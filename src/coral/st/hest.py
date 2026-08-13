"""Reader for an already-downloaded, processed HEST sample directory."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

#: The observation unit of a HEST package's counts, by the real technology its
#: metadata declares. Used by `coral.st.pipeline`, which is why it is public.
HEST_OBS_UNITS = {
    "Spatial Transcriptomics": "spot",
    "Visium": "spot",
    "VisiumHD": "spot",
    "Xenium": "cell",
}


def _pixel_size_um(metadata: dict[str, Any]) -> float | None:
    for key in ("pixel_size_um_embedded", "pixel_size_um_estimated"):
        value = metadata.get(key)
        if (
            isinstance(value, (int, float))
            and np.isfinite(value)
            and value > 0
        ):
            return float(value)
    return None


def _pooling_status(
    technology: str,
    metadata: dict[str, Any],
    spatial: np.ndarray,
) -> tuple[str, dict[str, Any]]:
    """Infer pooling only from a modality-specific, auditable signal."""
    if technology == "VisiumHD" and len(spatial) > 1:
        pixel_size_um = _pixel_size_um(metadata)
        if pixel_size_um is not None:
            distances, _ = cKDTree(spatial).query(spatial, k=2)
            median_spacing_um = float(
                np.median(distances[:, 1]) * pixel_size_um
            )
            evidence = {
                "method": "median_nearest_neighbor_spacing_um",
                "median_spacing_um": median_spacing_um,
                "native_max_bin_size_um": 16.0,
            }
            if median_spacing_um > 16.0:
                return "VisiumHD-pooled", evidence
            return "not-detected", evidence
    if technology == "Xenium":
        source_cells = metadata.get("num_cells")
        if isinstance(source_cells, int) and source_cells > len(spatial):
            return "Xenium-pooled", {
                "method": "metadata_num_cells_vs_observations",
                "source_num_cells": source_cells,
                "h5ad_n_obs": len(spatial),
            }
    return "not-detected", {}


@dataclass(frozen=True)
class HESTFiles:
    """The three required HEST artifacts for one already-downloaded sample."""

    h5ad: Path
    image: Path
    metadata: Path


def _one(paths: list[Path], label: str, input_dir: Path) -> Path:
    if len(paths) != 1:
        found = ", ".join(path.name for path in paths) or "none"
        raise ValueError(
            f"HEST sample {input_dir} must contain exactly one {label}; "
            f"found: {found}"
        )
    return paths[0]


def discover_hest_files(input_dir: Path) -> HESTFiles:
    """Find the core HEST artifacts in ESB's per-sample layout."""
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"HEST input directory does not exist: {input_dir}"
        )
    return HESTFiles(
        h5ad=_one(sorted(input_dir.glob("*.h5ad")), ".h5ad file", input_dir),
        image=_one(
            sorted(
                path
                for pattern in ("*.tif", "*.tiff", "*.btf")
                for path in input_dir.glob(pattern)
            ),
            "WSI TIFF",
            input_dir,
        ),
        metadata=_one(
            sorted(input_dir.glob("metadata.json")), "metadata.json", input_dir
        ),
    )


def _raw_counts(adata: ad.AnnData) -> str:
    """Promote an unmodified integer HEST count representation to ``.X``."""
    for name, matrix in (
        ("X", adata.X),
        ("counts", adata.layers.get("counts")),
        ("raw_counts", adata.layers.get("raw_counts")),
    ):
        if matrix is None:
            continue
        values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
        if (
            np.issubdtype(values.dtype, np.number)
            and np.all(np.isfinite(values))
            and np.all(values >= 0)
            and (
                not np.issubdtype(values.dtype, np.floating)
                or np.allclose(values, np.rint(values))
            )
        ):
            if name != "X":
                adata.X = matrix.copy()
            return name
    raise ValueError(
        "HEST AnnData needs raw integer counts in .X, .layers['counts'], "
        "or .layers['raw_counts']"
    )
