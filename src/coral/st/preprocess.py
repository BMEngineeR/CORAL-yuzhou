"""ESB ST spatial transcriptomics preprocessing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import anndata as ad

from coral.st.technology import (  # noqa: F401 - re-exported
    SUPPORTED_TECHNOLOGIES,
    _TECHNOLOGY_ALIASES,
    normalize_technology,
)
from coral.st.schema import ST_UNS_KEY
from coral.st.genes import (
    GeneConversionReport,
    convert_ensembl_ids_to_gene_names,
    normalize_biomart_organism,
)
from coral.st.normalize import (
    DEFAULT_TARGET_SUM,
    NORMALIZATION_TOTAL_COUNT,
    STNormalizationReport,
    normalize_expression,
    normalize_normalization_method,
)
from coral.st.qc import (
    STQCReport,
    apply_qc_filters,
    default_qc_thresholds,
)
from coral.st.selection import (
    GENE_SELECTION_NONE,
    STGeneSelectionReport,
    normalize_gene_selection_mode,
    select_genes,
)



@dataclass(frozen=True)
class STPreprocessConfig:
    """Configuration for ESB ST preprocessing.

    Args:
        technology: Optional spatial transcriptomics technology label. If
            omitted, validation tries to infer it from AnnData metadata.
        min_counts: Minimum transcript count per spot/cell for QC.
        min_genes: Reserved for a later detected-gene observation filter.
        min_gene_cells: Minimum cells/spots where a gene must be detected.
        min_gene_cell_fraction: Fraction of observations used to derive
            `min_gene_cells` when `min_gene_cells` is omitted.
        max_mito_pct: Maximum mitochondrial-count percentage for QC.
        convert_gene_ids: Whether Ensembl IDs in `.var_names` should be
            converted to gene names before writing output.
        gene_organism: BioMart organism used for gene conversion.
        original_gene_column: `.var` column preserving input names.
        gene_name_column: `.var` column storing final gene names.
        drop_unmapped_genes: Drop Ensembl IDs without a gene-name mapping.
        drop_duplicate_genes: Drop duplicate final gene names, keeping first.
        remove_mito_genes: Whether later QC should remove mitochondrial genes
            after flagging them.
        normalization: Expression normalization mode. Supports `none`,
            `total-count`, and `cpm`.
        target_sum: Target sum for total-count normalization. `cpm` uses
            1,000,000 regardless of this value.
        log1p: Whether to apply log1p after normalization.
        counts_layer: Layer where filtered raw counts are preserved before
            normalization. Set to `None` to skip writing a counts layer.
        gene_selection: Gene-selection mode. Supports `none`,
            `highly-variable`, `highly-expressed`, and `gene-list`.
        n_genes: Number of genes requested for HVG/HEG selection.
        hvg_flavor: Scanpy flavor used for highly variable gene selection.
        gene_expression_source: Source matrix for highly expressed gene
            ranking. Supports `normalized` and `counts`.
        gene_list_path: Optional CSV file with user genes of interest.
        append_gene_list: Whether user genes are appended to HVG/HEG
            selections instead of replacing them.
        missing_gene_policy: Policy for genes from `gene_list_path` that are
            absent from the AnnData object: `warn`, `error`, or `ignore`.
        require_spatial: Whether `.obsm["spatial"]` must be present.
        provenance_key: `.uns` key used for ESB preprocessing provenance.

    Examples:
        >>> cfg = STPreprocessConfig(technology="visium hd")
        >>> cfg.resolved_technology()
        'VisiumHD'
    """

    technology: str | None = None
    min_counts: int | None = None
    min_genes: int | None = None
    min_gene_cells: int | None = None
    min_gene_cell_fraction: float | None = None
    max_mito_pct: float | None = None
    convert_gene_ids: bool = True
    gene_organism: str = "hsapiens"
    original_gene_column: str = "esb_original_var_name"
    gene_name_column: str = "gene_name"
    drop_unmapped_genes: bool = True
    drop_duplicate_genes: bool = True
    remove_mito_genes: bool = False
    normalization: str = NORMALIZATION_TOTAL_COUNT
    target_sum: float | None = DEFAULT_TARGET_SUM
    log1p: bool = False
    counts_layer: str | None = "counts"
    gene_selection: str = GENE_SELECTION_NONE
    n_genes: int | None = None
    hvg_flavor: str = "seurat"
    gene_expression_source: str = "normalized"
    gene_list_path: str | None = None
    append_gene_list: bool = False
    missing_gene_policy: str = "warn"
    require_spatial: bool = True
    provenance_key: str = "esb_st_preprocess"
    extra: dict[str, str] = field(default_factory=dict)

    def resolved_technology(self) -> str | None:
        """Return the canonical technology name, if configured.

        Raises:
            ValueError: If the configured technology is unsupported.
        """
        if self.technology is None:
            return None
        return normalize_technology(self.technology)

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for `.uns`.

        Examples:
            >>> STPreprocessConfig(technology="xenium").to_metadata()[
            ...     "technology"
            ... ]
            'Xenium'
        """
        data = asdict(self)
        data["technology"] = self.resolved_technology()
        data["gene_organism"] = normalize_biomart_organism(self.gene_organism)
        data["normalization"] = normalize_normalization_method(
            self.normalization
        )
        data["gene_selection"] = normalize_gene_selection_mode(
            self.gene_selection
        )
        return data


@dataclass(frozen=True)
class ESBSTValidationReport:
    """Summary of ESB ST AnnData structure needed for preprocessing.

    Args:
        n_obs: Number of spots/cells.
        n_vars: Number of genes.
        technology: Canonical technology name, when known.
        has_spatial: Whether `.obsm["spatial"]` is present and shaped
            `(n_obs, 2)`.
        obs_columns: Observation metadata columns.
        var_columns: Gene metadata columns.
        uns_keys: Top-level unstructured metadata keys.
        warnings: Non-fatal validation notes.
    """

    n_obs: int
    n_vars: int
    technology: str | None
    has_spatial: bool
    obs_columns: tuple[str, ...]
    var_columns: tuple[str, ...]
    uns_keys: tuple[str, ...]
    warnings: tuple[str, ...] = ()




def validate_esb_st_adata(
    adata: ad.AnnData,
    config: STPreprocessConfig | None = None,
) -> ESBSTValidationReport:
    """Validate the ESB ST AnnData fields needed for preprocessing.

    Args:
        adata: AnnData object loaded from an ESB ST `.h5ad`.
        config: Optional preprocessing configuration.

    Returns:
        A structural validation report.

    Raises:
        ValueError: If required expression or spatial fields are invalid.

    Examples:
        >>> import numpy as np
        >>> import anndata as ad
        >>> tiny = ad.AnnData(np.ones((2, 3)))
        >>> tiny.obsm["spatial"] = np.array([[0.0, 0.0], [1.0, 1.0]])
        >>> report = validate_esb_st_adata(
        ...     tiny, STPreprocessConfig(technology="Visium")
        ... )
        >>> report.n_obs, report.n_vars, report.technology
        (2, 3, 'Visium')
    """
    cfg = config or STPreprocessConfig()
    _validate_matrix(adata)
    has_spatial = _validate_spatial(adata, require=cfg.require_spatial)
    technology, warnings = _resolve_technology(adata, cfg)
    return ESBSTValidationReport(
        n_obs=adata.n_obs,
        n_vars=adata.n_vars,
        technology=technology,
        has_spatial=has_spatial,
        obs_columns=tuple(str(c) for c in adata.obs.columns),
        var_columns=tuple(str(c) for c in adata.var.columns),
        uns_keys=tuple(str(k) for k in adata.uns),
        warnings=tuple(warnings),
    )


def preprocess_esb_st_adata(
    input_path: Path,
    output_path: Path,
    config: STPreprocessConfig | None = None,
    *,
    overwrite: bool = False,
    gene_mapping: dict[str, str] | None = None,
) -> ESBSTValidationReport:
    """Validate, QC-filter, and write an ESB ST `.h5ad` output file.

    Args:
        input_path: ESB ST `.h5ad` input.
        output_path: Destination `.h5ad` path.
        config: Optional preprocessing configuration.
        overwrite: Replace an existing output file when true.
        gene_mapping: Optional Ensembl-to-gene-name mapping. Used by tests and
            offline callers; when omitted, BioMart is queried if gene
            conversion is enabled.

    Returns:
        Validation report for the loaded input.

    Raises:
        FileExistsError: If `output_path` exists and `overwrite` is false.
        ValueError: If the input AnnData object is not ESB ST-compatible enough
            for sprint-43 preprocessing.
    """
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists")

    cfg = config or STPreprocessConfig()
    adata = ad.read_h5ad(input_path)
    report = validate_esb_st_adata(adata, cfg)
    gene_report: GeneConversionReport | None = None
    if cfg.convert_gene_ids:
        adata, gene_report = convert_ensembl_ids_to_gene_names(
            adata,
            organism=cfg.gene_organism,
            mapping=gene_mapping,
            original_column=cfg.original_gene_column,
            gene_name_column=cfg.gene_name_column,
            drop_unmapped=cfg.drop_unmapped_genes,
            drop_duplicates=cfg.drop_duplicate_genes,
        )
    qc_thresholds = default_qc_thresholds(
        report.technology,
        adata.n_obs,
        min_counts=cfg.min_counts,
        min_gene_cells=cfg.min_gene_cells,
        min_gene_cell_fraction=cfg.min_gene_cell_fraction,
        max_mito_pct=cfg.max_mito_pct,
        remove_mito_genes=cfg.remove_mito_genes,
    )
    adata, qc_report = apply_qc_filters(adata, qc_thresholds)
    adata, normalization_report = normalize_expression(
        adata,
        method=cfg.normalization,
        target_sum=cfg.target_sum,
        log1p=cfg.log1p,
        counts_layer=cfg.counts_layer,
    )
    adata, gene_selection_report = select_genes(
        adata,
        mode=cfg.gene_selection,
        n_genes=cfg.n_genes,
        hvg_flavor=cfg.hvg_flavor,
        expression_source=cfg.gene_expression_source,
        gene_list_path=cfg.gene_list_path,
        append_gene_list=cfg.append_gene_list,
        missing_gene_policy=cfg.missing_gene_policy,
        counts_layer=cfg.counts_layer,
        target_sum=cfg.target_sum,
        current_log1p=cfg.log1p,
    )
    adata.uns[cfg.provenance_key] = {
        "config": cfg.to_metadata(),
        "validation": {
            "n_obs": report.n_obs,
            "n_vars": report.n_vars,
            "technology": report.technology,
            "has_spatial": report.has_spatial,
            "warnings": list(report.warnings),
        },
        "gene_conversion": (
            gene_report.to_metadata() if gene_report is not None else None
        ),
        "qc": _qc_to_metadata(qc_report),
        "normalization": _normalization_to_metadata(normalization_report),
        "gene_selection": _gene_selection_to_metadata(gene_selection_report),
    }
    adata.write_h5ad(output_path)
    return report


def _qc_to_metadata(report: STQCReport) -> dict[str, Any]:
    """Return QC report metadata with stable primitive values."""
    return report.to_metadata()


def _normalization_to_metadata(
    report: STNormalizationReport,
) -> dict[str, Any]:
    """Return normalization report metadata with stable primitive values."""
    return report.to_metadata()


def _gene_selection_to_metadata(
    report: STGeneSelectionReport,
) -> dict[str, Any]:
    """Return gene-selection report metadata with stable primitive values."""
    return report.to_metadata()


def _validate_matrix(adata: ad.AnnData) -> None:
    """Validate that AnnData has a usable expression matrix."""
    if adata.X is None:
        raise ValueError(
            "ESB ST AnnData must contain an expression matrix `.X`"
        )
    if adata.n_obs == 0:
        raise ValueError("ESB ST AnnData must contain at least one spot/cell")
    if adata.n_vars == 0:
        raise ValueError("ESB ST AnnData must contain at least one gene")


def _validate_spatial(adata: ad.AnnData, *, require: bool) -> bool:
    """Validate `.obsm["spatial"]` when required."""
    if "spatial" not in adata.obsm:
        if require:
            raise ValueError('ESB ST AnnData must contain `.obsm["spatial"]`')
        return False
    spatial = adata.obsm["spatial"]
    if spatial.shape != (adata.n_obs, 2):
        raise ValueError(
            'ESB ST `.obsm["spatial"]` must have shape '
            f"({adata.n_obs}, 2); got {tuple(spatial.shape)}"
        )
    return True


def _resolve_technology(
    adata: ad.AnnData,
    config: STPreprocessConfig,
) -> tuple[str | None, list[str]]:
    """Resolve technology from config or AnnData metadata."""
    configured = config.resolved_technology()
    if configured is not None:
        return configured, []

    metadata_value = _technology_from_uns(adata.uns)
    if metadata_value is not None:
        return normalize_technology(metadata_value), []

    return None, [
        "ST technology not found in config or AnnData metadata; pass "
        "`technology` explicitly for technology-specific defaults."
    ]


def _technology_from_uns(uns: Mapping[str, Any]) -> str | None:
    """Find a technology label in ESB ST AnnData metadata."""
    for key in ("st_technology", "technology", "spatial_technology"):
        value = uns.get(key)
        if isinstance(value, str):
            return value
    for metadata_key in (ST_UNS_KEY,):
        esb_st = uns.get(metadata_key)
        if isinstance(esb_st, Mapping):
            value = esb_st.get("st_technology") or esb_st.get("technology")
            if isinstance(value, str):
                return value
    return None
