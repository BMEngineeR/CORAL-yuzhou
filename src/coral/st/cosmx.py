"""Native reader for NanoString CosMx (AtoMx export) samples.

CosMx is multi-FOV imaging spatial transcriptomics. A sample carries flat
tables (a cell x gene count matrix, per-cell metadata, vendor polygons,
transcripts) plus, per FOV, a multi-channel Morphology2D mIF image and
segmentation TIFFs.

This reader produces the CORAL ST contract WITHOUT SpatialData: an ``AnnData``
(raw RNA counts in ``.X``, cell centroids in ``obsm["spatial"]``) plus a single
``(c, y, x)`` morphology image. Because CosMx FOVs sit at scattered positions on
the slide, a true-global mosaic would be a huge, mostly-empty canvas — so v1
builds a **compact mosaic**: the FOV tiles are packed side by side and every
cell is placed by its FOV-local pixel coordinate (which already matches the
tile grid, so no affine or axis flip is needed and image + coordinates stay
consistent). One row is one ``cell``; coordinates are the compact image's own
pixel frame (``fullres_pixels``).

The per-FOV true-global affines and boxes are recorded in the returned
``details`` for the dearray-style ``fovs.geojson`` a later phase writes; they
are not written to the store here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: The nuclear morphology target, renamed so the store resolves nuclear_channel.
_NUCLEAR_TARGETS = {"dna", "dapi"}
_NUCLEAR_NAME = "dapi"

#: Vendor probe prefixes that are controls, not genes (kept in .X for v1).
_CONTROL_PREFIXES = ("negative", "systemcontrol", "falsecode")


@dataclass(frozen=True)
class CosmxLayout:
    """Resolved paths for one CosMx sample."""

    base: Path
    sample: str
    flat_dir: Path
    expr_mat: Path
    metadata: Path
    fov_positions: Path
    morphology2d_dir: Path
    channel_dict: Path | None
    polygons: Path | None
    transcripts: Path | None


def _one_subdir(parent: Path) -> Path:
    subs = [p for p in parent.iterdir() if p.is_dir()] if parent.is_dir() else []
    if len(subs) != 1:
        raise FileNotFoundError(
            f"expected exactly one sample directory under {parent}, "
            f"found {[p.name for p in subs]}"
        )
    return subs[0]


def discover_cosmx_files(sample_dir: Path | str) -> CosmxLayout:
    """Resolve a CosMx AtoMx export's files (paths only — no data is read).

    Raises:
        FileNotFoundError: If ``flatFiles/`` + ``RawFiles/`` or the required
            per-sample tables/images are missing — the message names what was
            looked for so detection can report it.
    """
    base = Path(sample_dir)
    flat_root, raw_root = base / "flatFiles", base / "RawFiles"
    if not (flat_root.is_dir() and raw_root.is_dir()):
        raise FileNotFoundError(
            f"CosMx sample needs flatFiles/ and RawFiles/ under {base}"
        )
    flat_dir = _one_subdir(flat_root)
    sample = flat_dir.name

    expr = flat_dir / f"{sample}_exprMat_file.csv.gz"
    meta = flat_dir / f"{sample}_metadata_file.csv.gz"
    fovpos = flat_dir / f"{sample}_fov_positions_file.csv.gz"
    for required in (expr, meta, fovpos):
        if not required.is_file():
            raise FileNotFoundError(f"CosMx sample missing table: {required}")

    run_dir = _one_subdir(raw_root / sample)
    cell_stats = run_dir / "CellStatsDir"
    morph = cell_stats / "Morphology2D"
    if not morph.is_dir():
        raise FileNotFoundError(f"CosMx sample missing Morphology2D/: {morph}")

    run_summary = run_dir / "RunSummary"
    cdict = run_summary / "Morphology_ChannelID_Dictionary.txt"
    poly = flat_dir / f"{sample}-polygons.csv.gz"
    tx = flat_dir / f"{sample}_tx_file.csv.gz"

    return CosmxLayout(
        base=base,
        sample=sample,
        flat_dir=flat_dir,
        expr_mat=expr,
        metadata=meta,
        fov_positions=fovpos,
        morphology2d_dir=morph,
        channel_dict=cdict if cdict.is_file() else None,
        polygons=poly if poly.is_file() else None,
        transcripts=tx if tx.is_file() else None,
    )


def _channel_names(layout: CosmxLayout, n_planes: int) -> list[str]:
    """Morphology channel names in plane order; the nuclear one is ``dapi``.

    Reads the ``Morphology_ChannelID_Dictionary.txt`` TSV (``ChannelID`` /
    ``BiologicalTarget``) in file order. Falls back to ``c0..`` when absent or
    mismatched, but always names a plane ``dapi`` only via the dictionary.
    """
    import pandas as pd

    if layout.channel_dict is not None:
        d = pd.read_csv(layout.channel_dict, sep="\t")
        targets = [str(t) for t in d["BiologicalTarget"]]
        if len(targets) == n_planes:
            return [
                _NUCLEAR_NAME if t.strip().lower() in _NUCLEAR_TARGETS else t
                for t in targets
            ]
        logger.warning(
            "CosMx channel dict has %d targets for %d planes; using c0..",
            len(targets), n_planes,
        )
    return [f"c{i}" for i in range(n_planes)]


def _fov_tile(layout: CosmxLayout, fov: int) -> Path:
    hits = sorted(layout.morphology2d_dir.glob(f"*_F{fov:05d}.TIF")) + sorted(
        layout.morphology2d_dir.glob(f"*_F{fov:05d}.tif")
    )
    if not hits:
        raise FileNotFoundError(
            f"CosMx FOV {fov} has no Morphology2D tile in "
            f"{layout.morphology2d_dir}"
        )
    return hits[0]


def read_cosmx_sample(layout: CosmxLayout) -> tuple[Any, np.ndarray, dict]:
    """Read a CosMx sample into ``(adata, image, details)``.

    ``adata.X`` is raw RNA counts (cells x genes); ``obsm["spatial"]`` is the
    ``(x, y)`` centroid in the compact mosaic's pixel frame. ``image`` is the
    packed morphology mosaic ``(c, y, x)``; ``details["channel_names"]`` names
    its channels and ``details["fov_layout"]`` records each FOV's compact and
    true-global placement for a later ``fovs.geojson``.
    """
    import anndata as ad
    import pandas as pd
    from scipy.sparse import csr_matrix

    from coral.io.harmonize import harmonize_to_canonical
    from coral.io.readers import read_ometiff

    ex = pd.read_csv(layout.expr_mat)
    ex = ex[ex["cell_ID"] != 0]
    gene_cols = [c for c in ex.columns if c not in ("fov", "cell_ID")]
    ex = ex.set_index(["fov", "cell_ID"])

    mt = pd.read_csv(layout.metadata).set_index(["fov", "cell_ID"])
    mt = mt.reindex(ex.index)

    fovs = sorted({int(f) for f, _ in ex.index})
    logger.info("      CosMx: %d FOV(s) %s, %d cells", len(fovs), fovs, len(ex))

    # ---- compact mosaic: pack FOV tiles side by side, place cells by local px
    tiles: dict[int, np.ndarray] = {}
    for fov in fovs:
        # decode each FOV's multi-channel Morphology2D via CORAL's own reader
        raw, ch, _, meta = read_ometiff(_fov_tile(layout, fov))
        tiles[fov], _cc = harmonize_to_canonical(raw, ch, meta["axes"])
    n_planes = tiles[fovs[0]].shape[0]
    if any(t.shape[0] != n_planes for t in tiles.values()):
        raise ValueError("CosMx FOV tiles differ in channel count")
    max_h = max(t.shape[1] for t in tiles.values())
    total_w = sum(t.shape[2] for t in tiles.values())
    mosaic = np.zeros((n_planes, max_h, total_w), dtype=tiles[fovs[0]].dtype)

    x_off: dict[int, int] = {}
    fov_layout: list[dict] = []
    cursor = 0
    for fov in fovs:
        t = tiles[fov]
        _, h, w = t.shape
        mosaic[:, :h, cursor : cursor + w] = t
        x_off[fov] = cursor
        fov_layout.append(
            {"fov": fov, "compact_box": [cursor, 0, w, h]}
        )
        cursor += w
    logger.info("      CosMx mosaic: %s (%d FOV tiles packed)", mosaic.shape, len(fovs))

    # per-cell coordinates in the compact frame: local px + this FOV's x offset
    fov_idx = ex.index.get_level_values("fov").astype(int).to_numpy()
    lx = mt["CenterX_local_px"].to_numpy(dtype="float64")
    ly = mt["CenterY_local_px"].to_numpy(dtype="float64")
    sx = lx + np.array([x_off[int(f)] for f in fov_idx], dtype="float64")
    spatial = np.column_stack([sx, ly])

    counts = csr_matrix(ex[gene_cols].to_numpy(dtype=np.float32))
    obs = mt.reset_index()
    obs_names = np.array(
        [f"c_{int(f)}_{int(c)}" for f, c in ex.index], dtype=object
    )
    adata = ad.AnnData(X=counts, obs=obs)
    adata.obs_names = obs_names
    adata.var_names = np.asarray(gene_cols, dtype=object)
    adata.obsm["spatial"] = spatial

    names = _channel_names(layout, n_planes)
    n_control = sum(
        1 for g in gene_cols if str(g).lower().startswith(_CONTROL_PREFIXES)
    )
    details = {
        "channel_names": names,
        "vendor": "nanostring_cosmx",
        "sample": layout.sample,
        "n_fovs": len(fovs),
        "fovs": fovs,
        "n_genes": len(gene_cols),
        "n_control_probes": n_control,
        "fov_layout": fov_layout,
        # located but NOT ingested in v1 — a later phase writes these
        "polygons_path": str(layout.polygons) if layout.polygons else None,
        "transcripts_path": (
            str(layout.transcripts) if layout.transcripts else None
        ),
        "fov_positions_path": str(layout.fov_positions),
        "pixel_size_um_estimated": None,
    }
    logger.info(
        "      CosMx: %d cells x %d genes (%d control probes)",
        adata.n_obs, adata.n_vars, n_control,
    )
    return adata, mosaic, details
