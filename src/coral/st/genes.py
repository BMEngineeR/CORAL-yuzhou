"""Gene identifier conversion for spatial transcriptomics AnnData objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import anndata as ad
import scanpy as sc

_ENSEMBL_PREFIXES = ("ENSG", "ENSMUSG")
_ORGANISM_ALIASES: dict[str, str] = {
    "human": "hsapiens",
    "homo sapiens": "hsapiens",
    "hsapiens": "hsapiens",
    "mouse": "mmusculus",
    "mus musculus": "mmusculus",
    "mmusculus": "mmusculus",
}


@dataclass(frozen=True)
class GeneConversionReport:
    """Summary of Ensembl-to-gene-name conversion.

    Args:
        total_genes: Number of genes before conversion.
        retained_genes: Number of genes after unmapped and duplicate drops.
        dropped_unmapped: Ensembl IDs removed because no gene name was found.
        dropped_duplicates: Duplicate final gene names removed.
        converted_genes: Ensembl IDs converted to gene names.
        already_symbol_genes: Non-Ensembl entries retained as gene names.
        organism: BioMart organism key used for the mapping.
        original_column: `.var` column containing pre-conversion names.
        gene_name_column: `.var` column containing final gene names.
    """

    total_genes: int
    retained_genes: int
    dropped_unmapped: int
    dropped_duplicates: int
    converted_genes: int
    already_symbol_genes: int
    organism: str
    original_column: str
    gene_name_column: str

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping summary.

        Examples:
            >>> report = GeneConversionReport(
            ...     1, 1, 0, 0, 1, 0, "hsapiens", "a", "b"
            ... )
            >>> report.to_metadata()["organism"]
            'hsapiens'
        """
        return asdict(self)


def normalize_biomart_organism(value: str) -> str:
    """Return a BioMart organism key.

    Args:
        value: User-facing organism label or BioMart key.

    Raises:
        ValueError: If the organism is not supported yet.

    Examples:
        >>> normalize_biomart_organism("Homo sapiens")
        'hsapiens'
        >>> normalize_biomart_organism("mouse")
        'mmusculus'
    """
    key = value.strip().casefold()
    organism = _ORGANISM_ALIASES.get(key)
    if organism is None:
        supported = ", ".join(sorted(set(_ORGANISM_ALIASES.values())))
        raise ValueError(
            f"unsupported BioMart organism {value!r}; supported: {supported}"
        )
    return organism


def fetch_biomart_gene_names(organism: str = "hsapiens") -> dict[str, str]:
    """Fetch Ensembl-to-gene-name mappings from Scanpy BioMart.

    Args:
        organism: BioMart organism key or supported alias.

    Returns:
        Mapping from Ensembl gene ID to external gene name.
    """
    org = normalize_biomart_organism(organism)
    annotations = sc.queries.biomart_annotations(
        org=org,
        attrs=["ensembl_gene_id", "external_gene_name"],
    )
    clean = annotations.dropna(subset=["external_gene_name"])
    clean = clean[clean["external_gene_name"].astype(str) != ""]
    return dict(
        zip(
            clean["ensembl_gene_id"].astype(str),
            clean["external_gene_name"].astype(str),
            strict=False,
        )
    )


def convert_ensembl_ids_to_gene_names(
    adata: ad.AnnData,
    *,
    organism: str = "hsapiens",
    mapping: dict[str, str] | None = None,
    original_column: str = "esb_original_var_name",
    gene_name_column: str = "gene_name",
    drop_unmapped: bool = True,
    drop_duplicates: bool = True,
) -> tuple[ad.AnnData, GeneConversionReport]:
    """Convert Ensembl IDs in `.var_names` to gene names.

    The returned AnnData has Scanpy-friendly gene symbols in `.var_names`.
    The original input names are preserved in `.var[original_column]`, and
    the final gene names are stored in `.var[gene_name_column]`.

    Args:
        adata: Spatial transcriptomics expression object.
        organism: BioMart organism key or supported alias. Defaults to human.
        mapping: Optional Ensembl-to-gene-name mapping. When omitted, Scanpy
            BioMart annotations are fetched for `organism`.
        original_column: `.var` column used to preserve original names.
        gene_name_column: `.var` column used to store final gene names.
        drop_unmapped: Drop unmapped Ensembl IDs when true.
        drop_duplicates: Drop duplicate final gene names when true, keeping
            the first occurrence.

    Returns:
        A converted AnnData view/copy and a conversion report.

    Raises:
        ValueError: If preservation columns would overwrite existing `.var`
            columns, or if unmapped/duplicate genes are disallowed.

    Examples:
        >>> import numpy as np
        >>> import anndata as ad
        >>> tiny = ad.AnnData(np.ones((1, 3)))
        >>> tiny.var_names = ["ENSG000001", "CD3D", "ENSG000404"]
        >>> out, report = convert_ensembl_ids_to_gene_names(
        ...     tiny, mapping={"ENSG000001": "TP53"}
        ... )
        >>> out.var_names.tolist()
        ['TP53', 'CD3D']
        >>> report.dropped_unmapped
        1
    """
    _check_column_available(adata, original_column)
    _check_column_available(adata, gene_name_column)
    org = normalize_biomart_organism(organism)
    original_names = [str(name) for name in adata.var_names]
    has_ensembl_ids = any(_looks_like_ensembl_id(n) for n in original_names)
    if mapping is not None:
        id_to_name = mapping
    elif has_ensembl_ids:
        id_to_name = fetch_biomart_gene_names(org)
    else:
        id_to_name = {}
    gene_names: list[str | None] = []
    keep_mask: list[bool] = []
    converted = 0
    already_symbol = 0
    dropped_unmapped = 0

    for name in original_names:
        if _looks_like_ensembl_id(name):
            mapped = _clean_gene_name(id_to_name.get(name))
            if mapped is None:
                if not drop_unmapped:
                    raise ValueError(f"no gene-name mapping for {name!r}")
                gene_names.append(None)
                keep_mask.append(False)
                dropped_unmapped += 1
                continue
            gene_names.append(mapped)
            keep_mask.append(True)
            converted += 1
        else:
            gene_names.append(name)
            keep_mask.append(True)
            already_symbol += 1

    kept_original = [
        original
        for original, keep in zip(original_names, keep_mask, strict=True)
        if keep
    ]
    kept_gene_names = [
        gene
        for gene, keep in zip(gene_names, keep_mask, strict=True)
        if keep and gene
    ]
    filtered = adata[:, keep_mask].copy()
    filtered.var[original_column] = kept_original
    filtered.var[gene_name_column] = kept_gene_names
    filtered.var_names = kept_gene_names
    filtered.var_names.name = adata.var_names.name

    dropped_duplicates = 0
    if filtered.var_names.has_duplicates:
        if not drop_duplicates:
            dupes = filtered.var_names[filtered.var_names.duplicated()]
            raise ValueError(
                "duplicate gene names after conversion: "
                + ", ".join(map(str, dupes.unique()))
            )
        duplicate_mask = ~filtered.var_names.duplicated(keep="first")
        dropped_duplicates = int((~duplicate_mask).sum())
        filtered = filtered[:, duplicate_mask].copy()

    report = GeneConversionReport(
        total_genes=len(original_names),
        retained_genes=filtered.n_vars,
        dropped_unmapped=dropped_unmapped,
        dropped_duplicates=dropped_duplicates,
        converted_genes=converted,
        already_symbol_genes=already_symbol,
        organism=org,
        original_column=original_column,
        gene_name_column=gene_name_column,
    )
    return filtered, report


def _check_column_available(adata: ad.AnnData, column: str) -> None:
    """Raise if `column` already exists in `.var`."""
    if column in adata.var:
        raise ValueError(f"AnnData `.var` already has column {column!r}")


def _looks_like_ensembl_id(value: str) -> bool:
    """Return whether a gene identifier looks like an Ensembl gene ID."""
    return value.startswith(_ENSEMBL_PREFIXES)


def _clean_gene_name(value: str | None) -> str | None:
    """Normalize empty gene names to `None`."""
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None
