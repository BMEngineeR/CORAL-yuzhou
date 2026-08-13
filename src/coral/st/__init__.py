"""Spatial transcriptomics: readers, the AnnData contract, and the store writer.

**Nothing is imported until it is asked for.** The readers need scanpy, pandas
and tifffile, and importing this package eagerly meant `import coral.st`, and
therefore `import coral` through any path that touched it, failed outright
without the `st` extra. It also meant validating a technology name loaded four
analysis modules.

So every public name is resolved on first access through the module
`__getattr__` below (PEP 562). `from coral.st import canonicalize_st_adata`
works exactly as before and pulls in only what that name needs.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

#: Whether the `st` extra is installed. Probed rather than imported, the same
#: way `coral.features.kronos2` probes for torch, so this module can say what
#: is missing without paying to find out.
ST_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("anndata", "scanpy", "h5py")
)

_INSTALL_MSG = (
    "Spatial transcriptomics needs the optional `st` extra "
    "(anndata, scanpy, h5py). Install it with `uv sync --extra st`."
)

#: Names that are pure and need nothing beyond CORAL itself. Asking for one of
#: these must never raise the install message: checking a technology name is
#: not a reason to need scanpy.
#: `coral.st.store`, `coral.st.detect` and `coral.st.pipeline` are pure at
#: IMPORT time: they reach for anndata and zarr inside the functions that need
#: them. So naming one of them must not produce the install message either.
_PURE = frozenset({
    "SUPPORTED_TECHNOLOGIES",
    "normalize_technology",
    "CORAL_STORE_VERSION",
    "ST_SCHEMA_VERSION",
    "detect_technology",
    "list_st_records",
    "read_st_config",
    "st_dir",
})

#: Public name -> the module that defines it. One table, so adding a reader is
#: one entry rather than an import line and an __all__ line that can disagree.
_EXPORTS: dict[str, str] = {
    # Moved to coral.slide.state when it stopped being an ST-only fence: all
    # four store writers stamp it now. Re-exported from here so
    # `from coral.st import CORAL_STORE_VERSION`, which shipped on main in
    # PR #11, keeps resolving.
    "CORAL_STORE_VERSION": "coral.slide.state",
    "COORDINATE_FRAMES": "coral.st.schema",
    "ST_SCHEMA_VERSION": "coral.st.store",
    "detect_technology": "coral.st.detect",
    "ingest_one": "coral.st.pipeline",
    "list_st_records": "coral.st.store",
    "read_st_config": "coral.st.store",
    "st_dir": "coral.st.store",
    "write_st_store": "coral.st.store",
    "CPM_TARGET_SUM": "coral.st.normalize",
    "DEFAULT_INTER_SPOT_DISTANCE_UM": "coral.st.legacy_st",
    "DEFAULT_SPOT_DIAMETER_UM": "coral.st.legacy_st",
    "DEFAULT_TARGET_SUM": "coral.st.normalize",
    "ESBSTValidationReport": "coral.st.preprocess",
    "GENE_SELECTION_HEG": "coral.st.selection",
    "GENE_SELECTION_HVG": "coral.st.selection",
    "GENE_SELECTION_LIST": "coral.st.selection",
    "GENE_SELECTION_NONE": "coral.st.selection",
    "GeneConversionReport": "coral.st.genes",
    "HESTFiles": "coral.st.hest",
    "LEGACY_SPOT_PACKING": "coral.st.legacy_st",
    "LegacySTFiles": "coral.st.legacy_st",
    "NORMALIZATION_CPM": "coral.st.normalize",
    "NORMALIZATION_NONE": "coral.st.normalize",
    "NORMALIZATION_TOTAL_COUNT": "coral.st.normalize",
    "OBS_UNITS": "coral.st.schema",
    "STGeneSelectionReport": "coral.st.selection",
    "STNormalizationReport": "coral.st.normalize",
    "STPreprocessConfig": "coral.st.preprocess",
    "STQCReport": "coral.st.qc",
    "STQCThresholds": "coral.st.qc",
    "CosmxLayout": "coral.st.cosmx",
    "G4XFiles": "coral.st.g4x",
    "STSchemaReport": "coral.st.schema",
    "VisiumFiles": "coral.st.visium",
    "VisiumHDFiles": "coral.st.visium_hd",
    "XeniumFiles": "coral.st.xenium",
    "apply_qc_filters": "coral.st.qc",
    "calculate_qc_metrics": "coral.st.qc",
    "canonicalize_st_adata": "coral.st.schema",
    "convert_ensembl_ids_to_gene_names": "coral.st.genes",
    "default_qc_thresholds": "coral.st.qc",
    "discover_cosmx_files": "coral.st.cosmx",
    "discover_g4x_files": "coral.st.g4x",
    "discover_hest_files": "coral.st.hest",
    "discover_legacy_st_files": "coral.st.legacy_st",
    "discover_visium_files": "coral.st.visium",
    "discover_visium_hd_files": "coral.st.visium_hd",
    "discover_xenium_files": "coral.st.xenium",
    "find_pixel_size_from_spot_coords": "coral.st.legacy_st",
    "normalize_biomart_organism": "coral.st.genes",
    "normalize_expression": "coral.st.normalize",
    "normalize_gene_selection_mode": "coral.st.selection",
    "normalize_normalization_method": "coral.st.normalize",
    "normalize_technology": "coral.st.technology",
    "preprocess_esb_st_adata": "coral.st.preprocess",
    "read_cosmx_sample": "coral.st.cosmx",
    "read_g4x_sample": "coral.st.g4x",
    "read_legacy_st_sample": "coral.st.legacy_st",
    "read_visium_hd_sample": "coral.st.visium_hd",
    "read_visium_sample": "coral.st.visium",
    "read_xenium_sample": "coral.st.xenium",
    "select_genes": "coral.st.selection",
    "validate_esb_st_adata": "coral.st.preprocess",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:  # noqa: ANN401 - re-exporting anything
    """Resolve a public name by importing only the module that defines it."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if not ST_AVAILABLE and name not in _PURE:
        # Named before the import is attempted, so the message says what to
        # install rather than which third-party module happened to be missing
        # first.
        raise ImportError(f"{_INSTALL_MSG} (needed for {name})")
    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return __all__
