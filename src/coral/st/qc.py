"""Quality-control metrics and filters for ST AnnData objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Any, cast

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse

_DEFAULT_QC: dict[str, dict[str, float]] = {
    "Spatial Transcriptomics": {
        "min_counts": 250.0,
        "min_gene_cell_fraction": 0.10,
        "max_mito_pct": 20.0,
    },
    "Visium": {
        "min_counts": 250.0,
        "min_gene_cell_fraction": 0.10,
        "max_mito_pct": 20.0,
    },
    "Xenium": {
        "min_counts": 5.0,
        "min_gene_cell_fraction": 0.01,
        "max_mito_pct": 20.0,
    },
    "VisiumHD": {
        "min_counts": 10.0,
        "min_gene_cell_fraction": 0.01,
        "max_mito_pct": 20.0,
    },
}


@dataclass(frozen=True)
class STQCThresholds:
    """Effective ST QC thresholds.

    Args:
        min_counts: Minimum transcript count per spot/cell.
        min_gene_cells: Minimum observations where a gene must be detected.
        min_gene_cell_fraction: Fraction used to derive `min_gene_cells`,
            when configured.
        max_mito_pct: Maximum mitochondrial percentage per spot/cell.
        remove_mito_genes: Whether to remove flagged mitochondrial genes.
    """

    min_counts: int
    min_gene_cells: int
    min_gene_cell_fraction: float | None
    max_mito_pct: float
    remove_mito_genes: bool = False

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable QC thresholds."""
        return asdict(self)


@dataclass(frozen=True)
class STQCReport:
    """Summary of ST QC filtering.

    Args:
        n_obs_before: Observation count before filtering.
        n_obs_after: Observation count after filtering.
        n_vars_before: Gene count before filtering.
        n_vars_after: Gene count after filtering.
        dropped_low_counts: Observations removed by `min_counts`.
        dropped_high_mito: Observations removed by `max_mito_pct`.
        dropped_low_support_genes: Genes removed by `min_gene_cells`.
        dropped_mito_genes: Mitochondrial genes removed.
        mito_genes_present: Whether any mitochondrial genes were detected.
        thresholds: Effective thresholds used for filtering.
    """

    n_obs_before: int
    n_obs_after: int
    n_vars_before: int
    n_vars_after: int
    dropped_low_counts: int
    dropped_high_mito: int
    dropped_low_support_genes: int
    dropped_mito_genes: int
    mito_genes_present: bool
    thresholds: STQCThresholds

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable QC summary."""
        data = asdict(self)
        data["thresholds"] = self.thresholds.to_metadata()
        return data


def default_qc_thresholds(
    technology: str | None,
    n_obs: int,
    *,
    min_counts: int | None = None,
    min_gene_cells: int | None = None,
    min_gene_cell_fraction: float | None = None,
    max_mito_pct: float | None = None,
    remove_mito_genes: bool = False,
) -> STQCThresholds:
    """Resolve effective QC thresholds for an ST technology.

    Args:
        technology: Canonical ST technology label.
        n_obs: Number of observations before QC.
        min_counts: Optional explicit minimum transcript count.
        min_gene_cells: Optional explicit gene-support threshold.
        min_gene_cell_fraction: Optional fraction of observations used to
            derive gene support.
        max_mito_pct: Optional explicit mitochondrial percentage threshold.
        remove_mito_genes: Whether mitochondrial genes should be removed.

    Examples:
        >>> default_qc_thresholds("Visium", 101).min_gene_cells
        11
        >>> default_qc_thresholds("Xenium", 101).min_counts
        5
    """
    defaults = _DEFAULT_QC.get(technology or "Visium", _DEFAULT_QC["Visium"])
    resolved_fraction = (
        min_gene_cell_fraction
        if min_gene_cell_fraction is not None
        else defaults["min_gene_cell_fraction"]
    )
    resolved_min_gene_cells = (
        min_gene_cells
        if min_gene_cells is not None
        else max(1, ceil(n_obs * resolved_fraction))
    )
    return STQCThresholds(
        min_counts=int(
            min_counts if min_counts is not None else defaults["min_counts"]
        ),
        min_gene_cells=int(resolved_min_gene_cells),
        min_gene_cell_fraction=float(resolved_fraction),
        max_mito_pct=float(
            max_mito_pct
            if max_mito_pct is not None
            else defaults["max_mito_pct"]
        ),
        remove_mito_genes=remove_mito_genes,
    )


def calculate_qc_metrics(adata: ad.AnnData) -> bool:
    """Calculate ST QC metrics in place.

    Args:
        adata: Spatial transcriptomics AnnData object.

    Returns:
        Whether mitochondrial genes are present.
    """
    mito = _mitochondrial_mask(adata)
    adata.var["mito"] = mito
    mito_present = bool(np.asarray(mito).any())
    qc_vars = ["mito"] if mito_present else []
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=qc_vars,
        percent_top=None,
        inplace=True,
    )
    if mito_present:
        adata.obs["pct_counts_mito"] = adata.obs["pct_counts_mito"].fillna(0.0)
    else:
        adata.obs["total_counts_mito"] = 0
        adata.obs["pct_counts_mito"] = 0.0
    return mito_present


def apply_qc_filters(
    adata: ad.AnnData,
    thresholds: STQCThresholds,
) -> tuple[ad.AnnData, STQCReport]:
    """Apply D1/D2 ST QC filters.

    Args:
        adata: AnnData with raw or count-like expression values.
        thresholds: Effective QC thresholds.

    Returns:
        Filtered AnnData copy and a QC report.
    """
    work = adata.copy()
    n_obs_before = work.n_obs
    n_vars_before = work.n_vars
    mito_present = calculate_qc_metrics(work)

    low_count_mask = (
        work.obs["total_counts"].to_numpy() < thresholds.min_counts
    )
    high_mito_mask = (
        work.obs["pct_counts_mito"].to_numpy() >= thresholds.max_mito_pct
        if mito_present
        else np.zeros(work.n_obs, dtype=bool)
    )
    keep_obs = ~(low_count_mask | high_mito_mask)
    work = work[keep_obs, :].copy()

    low_support_mask = _gene_support(work) < thresholds.min_gene_cells
    mito_gene_mask = (
        np.asarray(work.var["mito"], dtype=bool)
        if thresholds.remove_mito_genes and "mito" in work.var
        else np.zeros(work.n_vars, dtype=bool)
    )
    keep_vars = ~(low_support_mask | mito_gene_mask)
    work = work[:, keep_vars].copy()
    calculate_qc_metrics(work)

    report = STQCReport(
        n_obs_before=n_obs_before,
        n_obs_after=work.n_obs,
        n_vars_before=n_vars_before,
        n_vars_after=work.n_vars,
        dropped_low_counts=int(low_count_mask.sum()),
        dropped_high_mito=int(high_mito_mask.sum()),
        dropped_low_support_genes=int(low_support_mask.sum()),
        dropped_mito_genes=int(mito_gene_mask.sum()),
        mito_genes_present=mito_present,
        thresholds=thresholds,
    )
    return work, report


def _mitochondrial_mask(adata: ad.AnnData) -> np.ndarray:
    """Return the mitochondrial-gene mask from `.var` or gene names."""
    if "mito" in adata.var:
        return np.asarray(adata.var["mito"], dtype=bool)
    return np.fromiter(
        (
            str(name).startswith("MT-") or str(name).startswith("mt-")
            for name in adata.var_names
        ),
        dtype=bool,
        count=adata.n_vars,
    )


def _gene_support(adata: ad.AnnData) -> np.ndarray:
    """Number of observations where each gene is detected."""
    detected = cast(Any, adata.X) > 0
    if sparse.issparse(detected):
        return np.asarray(detected.sum(axis=0)).ravel()
    return np.asarray(detected).sum(axis=0)
