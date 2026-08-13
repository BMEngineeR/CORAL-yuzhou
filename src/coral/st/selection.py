"""Gene-selection helpers for ST AnnData objects."""

from __future__ import annotations

import csv
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse

from coral.st.normalize import DEFAULT_TARGET_SUM

GENE_SELECTION_NONE = "none"
GENE_SELECTION_HVG = "highly-variable"
GENE_SELECTION_HEG = "highly-expressed"
GENE_SELECTION_LIST = "gene-list"

_GENE_SELECTION_ALIASES: dict[str, str] = {
    "none": GENE_SELECTION_NONE,
    "off": GENE_SELECTION_NONE,
    "false": GENE_SELECTION_NONE,
    "highly-variable": GENE_SELECTION_HVG,
    "highly_variable": GENE_SELECTION_HVG,
    "hvg": GENE_SELECTION_HVG,
    "highly-expressed": GENE_SELECTION_HEG,
    "highly_expressed": GENE_SELECTION_HEG,
    "heg": GENE_SELECTION_HEG,
    "gene-list": GENE_SELECTION_LIST,
    "gene_list": GENE_SELECTION_LIST,
    "genes": GENE_SELECTION_LIST,
}

_MISSING_GENE_POLICIES: frozenset[str] = frozenset({"warn", "error", "ignore"})
_HVG_LOG_FLAVORS: frozenset[str] = frozenset({"seurat", "cell_ranger"})
_HVG_COUNT_FLAVORS: frozenset[str] = frozenset({"seurat_v3"})


@dataclass(frozen=True)
class STGeneSelectionReport:
    """Summary of ST gene selection.

    Args:
        mode: Effective gene-selection mode.
        n_vars_before: Gene count before selection.
        n_vars_after: Gene count after selection.
        n_genes_requested: Requested number of genes for ranked selections.
        n_genes_selected: Final selected gene count.
        source_matrix: Matrix/layer used to rank genes.
        hvg_flavor: Scanpy HVG flavor, when used.
        gene_list_path: Optional user gene-list CSV path.
        append_gene_list: Whether gene-list genes were appended to selection.
        n_gene_list_requested: Number of genes requested from the CSV.
        n_gene_list_found: Number of CSV genes found in AnnData.
        n_gene_list_missing: Number of CSV genes missing from AnnData.
        missing_gene_policy: How missing CSV genes were handled.
        missing_genes: Missing CSV genes, capped for provenance readability.
    """

    mode: str
    n_vars_before: int
    n_vars_after: int
    n_genes_requested: int | None
    n_genes_selected: int
    source_matrix: str | None
    hvg_flavor: str | None
    gene_list_path: str | None
    append_gene_list: bool
    n_gene_list_requested: int
    n_gene_list_found: int
    n_gene_list_missing: int
    missing_gene_policy: str
    missing_genes: tuple[str, ...]

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable gene-selection metadata."""
        data = asdict(self)
        data["missing_genes"] = list(self.missing_genes)
        return data


def select_genes(
    adata: ad.AnnData,
    *,
    mode: str = GENE_SELECTION_NONE,
    n_genes: int | None = None,
    hvg_flavor: str = "seurat",
    expression_source: str = "normalized",
    gene_list_path: str | None = None,
    append_gene_list: bool = False,
    missing_gene_policy: str = "warn",
    counts_layer: str | None = "counts",
    target_sum: float | None = DEFAULT_TARGET_SUM,
    current_log1p: bool = False,
) -> tuple[ad.AnnData, STGeneSelectionReport]:
    """Select genes from a preprocessed ST AnnData object.

    Args:
        adata: QC-filtered and optionally normalized AnnData.
        mode: Gene-selection mode.
        n_genes: Requested number of genes for HVG/HEG selection.
        hvg_flavor: Scanpy HVG flavor.
        expression_source: Source matrix for highly expressed genes.
        gene_list_path: Optional CSV with genes of interest.
        append_gene_list: Append CSV genes to HVG/HEG selection.
        missing_gene_policy: `warn`, `error`, or `ignore`.
        counts_layer: Layer containing filtered raw counts.
        target_sum: Temporary target sum for log-normalized HVG views.
        current_log1p: Whether current `.X` is already log-normalized.

    Returns:
        A gene-subset copy and a gene-selection report.
    """
    resolved_mode = normalize_gene_selection_mode(mode)
    resolved_policy = _normalize_missing_gene_policy(missing_gene_policy)
    n_vars_before = adata.n_vars
    selected = np.zeros(adata.n_vars, dtype=bool)
    source_matrix: str | None = None
    effective_hvg_flavor: str | None = None
    requested = n_genes

    if resolved_mode == GENE_SELECTION_HVG:
        selected, source_matrix = _select_hvg(
            adata,
            n_genes=_resolve_n_genes(n_genes, adata.n_vars),
            flavor=hvg_flavor,
            counts_layer=counts_layer,
            target_sum=target_sum,
            current_log1p=current_log1p,
        )
        effective_hvg_flavor = hvg_flavor
    elif resolved_mode == GENE_SELECTION_HEG:
        selected, source_matrix = _select_heg(
            adata,
            n_genes=_resolve_n_genes(n_genes, adata.n_vars),
            expression_source=expression_source,
            counts_layer=counts_layer,
        )
    elif resolved_mode == GENE_SELECTION_LIST:
        selected = np.zeros(adata.n_vars, dtype=bool)
    elif resolved_mode != GENE_SELECTION_NONE:
        raise ValueError(f"unsupported ST gene-selection mode {mode!r}")

    gene_list = _read_gene_list(gene_list_path) if gene_list_path else []
    gene_list_mask, missing = _gene_list_mask(adata, gene_list)
    if gene_list:
        if resolved_mode == GENE_SELECTION_NONE and append_gene_list:
            raise ValueError(
                "cannot append gene list when gene selection is none"
            )
        if missing and resolved_policy == "error":
            raise ValueError(
                "gene list contains genes not present in AnnData: "
                + ", ".join(missing[:10])
            )
        if missing and resolved_policy == "warn":
            warnings.warn(
                "gene list contains genes not present in AnnData: "
                + ", ".join(missing[:10]),
                stacklevel=2,
            )
        if append_gene_list:
            selected |= gene_list_mask
        else:
            selected = gene_list_mask
            resolved_mode = GENE_SELECTION_LIST

    if resolved_mode == GENE_SELECTION_NONE and not gene_list:
        selected = np.ones(adata.n_vars, dtype=bool)

    work = adata[:, selected].copy()
    work.var["esb_gene_selected"] = True
    report = STGeneSelectionReport(
        mode=resolved_mode,
        n_vars_before=n_vars_before,
        n_vars_after=work.n_vars,
        n_genes_requested=requested,
        n_genes_selected=work.n_vars,
        source_matrix=source_matrix,
        hvg_flavor=effective_hvg_flavor,
        gene_list_path=gene_list_path,
        append_gene_list=append_gene_list,
        n_gene_list_requested=len(gene_list),
        n_gene_list_found=int(gene_list_mask.sum()),
        n_gene_list_missing=len(missing),
        missing_gene_policy=resolved_policy,
        missing_genes=tuple(missing[:50]),
    )
    return work, report


def normalize_gene_selection_mode(value: str) -> str:
    """Return ESB's canonical ST gene-selection mode."""
    cleaned = value.strip().casefold()
    canonical = _GENE_SELECTION_ALIASES.get(cleaned)
    if canonical is None:
        supported = ", ".join(
            [
                GENE_SELECTION_NONE,
                GENE_SELECTION_HVG,
                GENE_SELECTION_HEG,
                GENE_SELECTION_LIST,
            ]
        )
        raise ValueError(
            f"unsupported ST gene-selection mode {value!r}; "
            f"supported: {supported}"
        )
    return canonical


def _select_hvg(
    adata: ad.AnnData,
    *,
    n_genes: int,
    flavor: str,
    counts_layer: str | None,
    target_sum: float | None,
    current_log1p: bool,
) -> tuple[np.ndarray, str]:
    """Select highly variable genes with Scanpy."""
    normalized_flavor = flavor.replace("-", "_")
    if normalized_flavor in _HVG_COUNT_FLAVORS:
        if counts_layer is None or counts_layer not in adata.layers:
            raise ValueError(
                f"HVG flavor {flavor!r} requires counts layer {counts_layer!r}"
            )
        work = adata.copy()
        work.X = work.layers[counts_layer].copy()
        source_matrix = f"layers[{counts_layer!r}]"
    elif normalized_flavor in _HVG_LOG_FLAVORS:
        work = adata.copy()
        if not current_log1p:
            sc.pp.normalize_total(
                work,
                target_sum=(
                    DEFAULT_TARGET_SUM if target_sum is None else target_sum
                ),
                inplace=True,
            )
            sc.pp.log1p(work)
            source_matrix = "temporary_normalized_log1p"
        else:
            source_matrix = "X"
    else:
        raise ValueError(f"unsupported Scanpy HVG flavor {flavor!r}")

    flavor_literal = cast(
        Literal["seurat", "cell_ranger", "seurat_v3"],
        normalized_flavor,
    )
    sc.pp.highly_variable_genes(
        work,
        n_top_genes=n_genes,
        flavor=flavor_literal,
        inplace=True,
    )
    return _hvg_mask(work, n_genes), source_matrix


def _select_heg(
    adata: ad.AnnData,
    *,
    n_genes: int,
    expression_source: str,
    counts_layer: str | None,
) -> tuple[np.ndarray, str]:
    """Select genes with highest mean expression."""
    matrix, source = _expression_matrix(
        adata,
        expression_source=expression_source,
        counts_layer=counts_layer,
    )
    means = _column_means(matrix)
    order = np.argsort(-means, kind="stable")
    selected = np.zeros(adata.n_vars, dtype=bool)
    selected[order[:n_genes]] = True
    return selected, source


def _expression_matrix(
    adata: ad.AnnData,
    *,
    expression_source: str,
    counts_layer: str | None,
) -> tuple[Any, str]:
    """Return the matrix used for ranked expression selections."""
    cleaned = expression_source.strip().casefold()
    if cleaned in {"normalized", "x"}:
        return adata.X, "X"
    if cleaned in {"counts", "raw-counts", "raw_counts"}:
        if counts_layer is None or counts_layer not in adata.layers:
            raise ValueError(f"counts layer {counts_layer!r} is not available")
        return adata.layers[counts_layer], f"layers[{counts_layer!r}]"
    raise ValueError(f"unsupported expression source {expression_source!r}")


def _hvg_mask(adata: ad.AnnData, n_genes: int) -> np.ndarray:
    """Return exactly `n_genes` HVGs from Scanpy's HVG annotations."""
    if "highly_variable_rank" in adata.var:
        ranks = np.asarray(adata.var["highly_variable_rank"], dtype=float)
        order = np.argsort(ranks, kind="stable")
        valid = order[np.isfinite(ranks[order])][:n_genes]
    elif "variances_norm" in adata.var:
        scores = np.asarray(adata.var["variances_norm"], dtype=float)
        valid = np.argsort(-scores, kind="stable")[:n_genes]
    else:
        selected = np.asarray(adata.var["highly_variable"], dtype=bool)
        valid = np.flatnonzero(selected)[:n_genes]
    mask = np.zeros(adata.n_vars, dtype=bool)
    mask[valid] = True
    return mask


def _column_means(matrix: object) -> np.ndarray:
    """Return dense column means for dense or sparse matrices."""
    if sparse.issparse(matrix):
        return np.asarray(cast(Any, matrix).mean(axis=0)).ravel()
    return np.asarray(cast(Any, matrix)).mean(axis=0)


def _read_gene_list(path: str | None) -> list[str]:
    """Read a gene list from a CSV file."""
    if path is None:
        return []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        column = _gene_column(list(reader.fieldnames))
        genes = [str(row.get(column, "")).strip() for row in reader]
    return [gene for gene in dict.fromkeys(genes) if gene]


def _gene_column(fieldnames: list[str]) -> str:
    """Pick the gene column from common CSV schemas."""
    normalized = {name.strip().casefold(): name for name in fieldnames}
    for candidate in ("gene", "gene_name", "symbol", "gene_symbol"):
        if candidate in normalized:
            return normalized[candidate]
    return fieldnames[0]


def _gene_list_mask(
    adata: ad.AnnData,
    genes: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Return mask for present gene-list entries and missing names."""
    var_names = np.asarray([str(name) for name in adata.var_names])
    present = set(var_names.tolist())
    requested = set(genes)
    mask = np.isin(var_names, list(requested))
    missing = [gene for gene in genes if gene not in present]
    return mask, missing


def _resolve_n_genes(n_genes: int | None, n_vars: int) -> int:
    """Return a valid number of genes to select."""
    requested = 3_000 if n_genes is None else n_genes
    if requested <= 0:
        raise ValueError("n_genes must be positive")
    return min(requested, n_vars)


def _normalize_missing_gene_policy(value: str) -> str:
    """Return canonical missing-gene policy."""
    cleaned = value.strip().casefold()
    if cleaned not in _MISSING_GENE_POLICIES:
        supported = ", ".join(sorted(_MISSING_GENE_POLICIES))
        raise ValueError(
            f"unsupported missing gene policy {value!r}; "
            f"supported: {supported}"
        )
    return cleaned
