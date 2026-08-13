"""ESB's shared AnnData contract for spatial transcriptomics ingest."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import anndata as ad
import numpy as np
from scipy import sparse

from coral.st.technology import normalize_technology, observation_diameter_um

logger = logging.getLogger(__name__)

#: The `uns` key this contract occupies. One name, read and written in one
#: place, so a table opened outside any store still says what it is.
ST_UNS_KEY = "coral"


def st_metadata(adata: Any) -> dict:  # noqa: ANN401 - an AnnData
    """This contract's block out of an AnnData.

    Returns an empty dict when there is none, so a caller reading one field
    does not have to check for the block first.
    """
    value = adata.uns.get(ST_UNS_KEY)
    return dict(value) if isinstance(value, Mapping) else {}

#: Both sets are DELIBERATELY CLOSED, and extending them is part of adding a
#: reader, in the same change. A new observation unit is one entry in
#: OBS_UNITS. A new coordinate frame is more: `coral.st.store._to_pixels` must
#: also learn how to place it on the image, because a frame the store cannot
#: convert to its own pixels is refused at write time.
COORDINATE_FRAMES = frozenset({"fullres_pixels", "array", "microns"})
OBS_UNITS = frozenset({"spot", "cell", "bin_2", "bin_8", "bin_16"})


@dataclass(frozen=True)
class STSchemaReport:
    """Summary of a canonical ST AnnData object.

    Args:
        n_obs: Number of native observations.
        n_vars: Number of expression features.
        technology: Canonical ESB technology label.
        coordinate_frame: Frame used by ``.obsm["spatial"]``.
        obs_unit: Native observation unit.
        sparse_counts: Whether the count matrix remains sparse.
        obs_diameter_um: What one observation physically covers, in microns,
            or None where the technology has no single size. A first-class
            field rather than a key in a per-reader details dict, because it
            is what makes a spot map interpretable: the same picture means
            something different at 55 microns and at 2.

    Example:
        A reader passes its AnnData through :func:`canonicalize_st_adata`
        and records the returned report in the sample manifest.
    """

    n_obs: int
    n_vars: int
    technology: str
    coordinate_frame: str
    obs_unit: str
    sparse_counts: bool
    obs_diameter_um: float | None = None


def _validate_names(values: Any, label: str) -> None:  # noqa: ANN401
    names = [str(value) for value in values]
    if any(not name.strip() for name in names):
        raise ValueError(f"AnnData {label} names must not be empty")
    if len(names) != len(set(names)):
        raise ValueError(f"AnnData {label} names must be unique")


def _validate_raw_counts(matrix: Any) -> None:  # noqa: ANN401
    if matrix is None:
        raise ValueError("AnnData .X is required and must contain raw counts")
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("AnnData .X must be numeric raw counts")
    if not np.all(np.isfinite(values)):
        raise ValueError("AnnData .X contains non-finite values")
    if np.any(values < 0):
        raise ValueError("AnnData .X contains negative values, not raw counts")
    if np.issubdtype(values.dtype, np.floating) and not np.allclose(
        values, np.rint(values)
    ):
        raise ValueError(
            "AnnData .X contains fractional values; ingest requires raw counts"
        )


def canonicalize_st_adata(
    adata: ad.AnnData,
    *,
    technology: str,
    sample_id: str,
    coordinate_frame: str,
    obs_unit: str,
    obs_diameter_um: float | None = None,
    technology_details: dict[str, Any] | None = None,
) -> STSchemaReport:
    """Validate and annotate an AnnData object for canonical ST output.

    The function mutates only metadata and index string types. It never
    normalizes, filters, densifies, or otherwise changes ``.X``.

    Args:
        adata: Reader-produced AnnData containing raw counts and coordinates.
        technology: Supported ST technology label.
        sample_id: Non-empty stable sample identifier.
        coordinate_frame: Explicit frame for spatial coordinates.
        obs_unit: Native observation unit represented by each row.

    Returns:
        Validated schema summary.

    Raises:
        TypeError: If ``adata`` is not AnnData.
        ValueError: If any required field violates the output contract.

    Example:
        ``canonicalize_st_adata(adata, technology="visium",
        sample_id="S1", coordinate_frame="fullres_pixels",
        obs_unit="spot")`` validates a Visium reader result.
    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError("adata must be an anndata.AnnData object")
    canonical_technology = normalize_technology(technology)
    if not sample_id.strip():
        raise ValueError("sample_id must not be empty")
    if coordinate_frame not in COORDINATE_FRAMES:
        allowed = ", ".join(sorted(COORDINATE_FRAMES))
        raise ValueError(
            f"unsupported coordinate frame {coordinate_frame!r}; "
            f"supported: {allowed}"
        )
    if obs_unit not in OBS_UNITS:
        allowed = ", ".join(sorted(OBS_UNITS))
        raise ValueError(
            f"unsupported observation unit {obs_unit!r}; supported: "
            f"{allowed}. A reader for a technology with a new unit extends "
            f"OBS_UNITS in coral/st/schema.py in the same change."
        )

    logger.info("Validating raw count matrix: shape=%s", adata.shape)
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError(
            f"AnnData must have observations and features; shape={adata.shape}"
        )
    _validate_raw_counts(adata.X)
    _validate_names(adata.obs_names, "observation")
    _validate_names(adata.var_names, "feature")
    adata.obs_names = [str(value) for value in adata.obs_names]
    adata.var_names = [str(value) for value in adata.var_names]

    if "spatial" not in adata.obsm:
        raise ValueError('AnnData .obsm["spatial"] is required')
    spatial = np.asarray(adata.obsm["spatial"])
    if spatial.shape != (adata.n_obs, 2):
        raise ValueError(
            'AnnData .obsm["spatial"] must have shape '
            f"({adata.n_obs}, 2), got {spatial.shape}"
        )
    if not np.issubdtype(spatial.dtype, np.number):
        raise ValueError('AnnData .obsm["spatial"] must be numeric')
    if not np.all(np.isfinite(spatial)):
        raise ValueError('AnnData .obsm["spatial"] contains non-finite values')

    # Derived here when the caller did not supply it, so a reader that records
    # the value in its details dict gets it into the contract without every
    # call site knowing which key its technology uses.
    diameter = obs_diameter_um
    if diameter is None and technology_details:
        diameter = observation_diameter_um(technology_details)
    if diameter is not None:
        diameter = float(diameter)
        if not np.isfinite(diameter) or diameter <= 0:
            raise ValueError(
                f"obs_diameter_um must be a positive finite number, got {diameter!r}"
            )

    adata.uns[ST_UNS_KEY] = {
        "technology": canonical_technology,
        "sample_id": sample_id,
        "coordinate_frame": coordinate_frame,
        "obs_unit": obs_unit,
        # In the table itself, so an AnnData opened on its own still says what
        # one of its rows covers.
        "obs_diameter_um": diameter,
    }
    logger.info(
        "Canonical ST schema validated: technology=%s sample=%s "
        "obs_unit=%s coordinate_frame=%s n_obs=%d n_vars=%d",
        canonical_technology,
        sample_id,
        obs_unit,
        coordinate_frame,
        adata.n_obs,
        adata.n_vars,
    )
    return STSchemaReport(
        n_obs=adata.n_obs,
        n_vars=adata.n_vars,
        technology=canonical_technology,
        coordinate_frame=coordinate_frame,
        obs_unit=obs_unit,
        sparse_counts=sparse.issparse(adata.X),
        obs_diameter_um=diameter,
    )
