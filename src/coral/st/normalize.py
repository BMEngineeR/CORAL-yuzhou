"""Normalization helpers for ST AnnData objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, cast

import anndata as ad
import scanpy as sc

NORMALIZATION_NONE = "none"
NORMALIZATION_TOTAL_COUNT = "total-count"
NORMALIZATION_CPM = "cpm"
DEFAULT_TARGET_SUM = 10_000.0
CPM_TARGET_SUM = 1_000_000.0

_NORMALIZATION_ALIASES: dict[str, str] = {
    "none": NORMALIZATION_NONE,
    "off": NORMALIZATION_NONE,
    "false": NORMALIZATION_NONE,
    "total-count": NORMALIZATION_TOTAL_COUNT,
    "total_count": NORMALIZATION_TOTAL_COUNT,
    "normalize-total": NORMALIZATION_TOTAL_COUNT,
    "normalize_total": NORMALIZATION_TOTAL_COUNT,
    "library-size": NORMALIZATION_TOTAL_COUNT,
    "library_size": NORMALIZATION_TOTAL_COUNT,
    "sequencing-depth": NORMALIZATION_TOTAL_COUNT,
    "sequencing_depth": NORMALIZATION_TOTAL_COUNT,
    "cpm": NORMALIZATION_CPM,
    "counts-per-million": NORMALIZATION_CPM,
    "counts_per_million": NORMALIZATION_CPM,
}


@dataclass(frozen=True)
class STNormalizationReport:
    """Summary of ST expression normalization.

    Args:
        method: Effective normalization method.
        target_sum: Target sum used for total-count normalization.
        log1p: Whether `log1p` was applied.
        counts_layer: Layer where filtered raw counts were preserved.
        counts_layer_written: Whether the counts layer was written.
        normalized: Whether `.X` was rescaled.
    """

    method: str
    target_sum: float | None
    log1p: bool
    counts_layer: str | None
    counts_layer_written: bool
    normalized: bool

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable normalization metadata."""
        return asdict(self)


def normalize_expression(
    adata: ad.AnnData,
    *,
    method: str = NORMALIZATION_TOTAL_COUNT,
    target_sum: float | None = DEFAULT_TARGET_SUM,
    log1p: bool = False,
    counts_layer: str | None = "counts",
) -> tuple[ad.AnnData, STNormalizationReport]:
    """Normalize ST expression after QC filtering.

    Args:
        adata: QC-filtered AnnData object.
        method: Normalization mode. Supports `none`, `total-count`, and `cpm`.
        target_sum: Total count scale for `total-count`; ignored for `cpm`.
        log1p: Whether to apply `scanpy.pp.log1p` after optional scaling.
        counts_layer: Optional layer name for preserving filtered raw counts.

    Returns:
        A normalized copy and a normalization report.
    """
    work = adata.copy()
    resolved = normalize_normalization_method(method)
    counts_layer_written = False
    if counts_layer is not None:
        work.layers[counts_layer] = cast(Any, work.X).copy()
        counts_layer_written = True

    effective_target_sum = _effective_target_sum(resolved, target_sum)
    normalized = resolved != NORMALIZATION_NONE
    if resolved == NORMALIZATION_TOTAL_COUNT:
        sc.pp.normalize_total(
            work,
            target_sum=effective_target_sum,
            inplace=True,
        )
    elif resolved == NORMALIZATION_CPM:
        sc.pp.normalize_total(
            work,
            target_sum=CPM_TARGET_SUM,
            inplace=True,
        )
    elif resolved != NORMALIZATION_NONE:
        raise ValueError(f"unsupported ST normalization method {method!r}")

    if log1p:
        sc.pp.log1p(work)

    report = STNormalizationReport(
        method=resolved,
        target_sum=effective_target_sum,
        log1p=log1p,
        counts_layer=counts_layer,
        counts_layer_written=counts_layer_written,
        normalized=normalized,
    )
    return work, report


def normalize_normalization_method(value: str) -> str:
    """Return ESB's canonical ST normalization method.

    Args:
        value: User-facing normalization mode.

    Raises:
        ValueError: If the mode is unsupported.
    """
    cleaned = value.strip().casefold()
    canonical = _NORMALIZATION_ALIASES.get(cleaned)
    if canonical is None:
        supported = ", ".join(
            [NORMALIZATION_NONE, NORMALIZATION_TOTAL_COUNT, NORMALIZATION_CPM]
        )
        raise ValueError(
            f"unsupported ST normalization method {value!r}; "
            f"supported: {supported}"
        )
    return canonical


def _effective_target_sum(
    method: str,
    target_sum: float | None,
) -> float | None:
    """Return the target sum recorded and used for normalization."""
    if method == NORMALIZATION_CPM:
        return CPM_TARGET_SUM
    if method == NORMALIZATION_TOTAL_COUNT:
        return DEFAULT_TARGET_SUM if target_sum is None else float(target_sum)
    return None
