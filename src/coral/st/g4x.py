"""Native reader for Singular Genomics G4X samples.

G4X is imaging spatial transcriptomics. A sample carries a cell x gene count
table, a cell x protein intensity table, cell metadata (centroids), per-marker
protein images plus a nuclear stain, and a segmentation mask.

This reader produces the CORAL ST contract WITHOUT SpatialData: an ``AnnData``
(raw RNA counts in ``.X``, cell centroids in ``obsm["spatial"]``, protein
intensities in ``obsm["protein"]``) plus the protein+nuclear image as a
``(c, y, x)`` array whose nuclear channel is named ``dapi`` so CORAL's store
writer resolves it as the nuclear channel. Coordinates are the image's own
pixel frame (``fullres_pixels``); one row is one ``cell``.

Segmentation masks and the transcript table are located and recorded in the
returned ``details`` but NOT written to the store — that is a later phase.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: The suffix G4X puts on every cell-by-protein intensity column.
_INTENSITY_SUFFIX = "_intensity_mean"

#: The name CORAL's store writer looks for to set ``nuclear_channel``.
_NUCLEAR_NAME = "dapi"


@dataclass(frozen=True)
class G4XFiles:
    """Resolved paths for one G4X sample. Optional members are ``None``.

    ``base`` is the sample directory; the two count/metadata tables are
    required, everything else is optional (but a store needs at least one
    image, enforced in :func:`read_g4x_sample`).
    """

    base: Path
    cell_by_transcript: Path
    cell_metadata: Path
    cell_by_protein: Path | None
    protein_panel: Path | None
    protein_dir: Path | None
    nuclear_image: Path | None
    run_meta: Path | None
    segmentation: Path | None
    transcripts: Path | None


def _first(*candidates: Path) -> Path | None:
    """The first candidate that exists, or ``None``."""
    for c in candidates:
        if c.is_file():
            return c
    return None


def discover_g4x_files(sample_dir: Path | str) -> G4XFiles:
    """Resolve a G4X sample's files (paths only — no data is read).

    Raises:
        FileNotFoundError: If the directory, the ``single_cell_data`` folder,
            or the required count/metadata tables are missing — the message
            names exactly what was looked for so detection can report it.
    """
    base = Path(sample_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"G4X sample directory does not exist: {base}")

    scd = base / "single_cell_data"
    if not scd.is_dir():
        raise FileNotFoundError(
            f"G4X sample has no single_cell_data/ directory: {scd}"
        )

    cell_by_transcript = scd / "cell_by_transcript.csv.gz"
    cell_metadata = scd / "cell_metadata.csv.gz"
    for required in (cell_by_transcript, cell_metadata):
        if not required.is_file():
            raise FileNotFoundError(
                f"G4X sample is missing a required table: {required}"
            )

    protein_dir = base / "protein"
    return G4XFiles(
        base=base,
        cell_by_transcript=cell_by_transcript,
        cell_metadata=cell_metadata,
        cell_by_protein=_first(scd / "cell_by_protein.csv.gz"),
        protein_panel=_first(base / "protein_panel.csv"),
        protein_dir=protein_dir if protein_dir.is_dir() else None,
        nuclear_image=_first(
            base / "h_and_e" / "nuclear.tiff", base / "h_and_e" / "nuclear.tif"
        ),
        run_meta=_first(base / "run_meta.json"),
        segmentation=_first(
            base / "segmentation" / "segmentation_mask.npz"
        ),
        transcripts=_first(base / "rna" / "transcript_table.csv.gz"),
    )


def _build_image(files: G4XFiles) -> tuple[np.ndarray, list[str]]:
    """Nuclear + protein markers as ``(c, y, x)`` with channel names.

    Decoding is delegated to CORAL's own proteomics TIFF readers rather than
    read here: ``read_ometiff`` for the single nuclear stain and
    ``read_channel_tiff_dir`` for the one-file-per-marker ``protein/`` folder
    (exactly the layout that reader was written for). Both pass through
    ``harmonize_to_canonical`` to ``(c, y, x)``. The nuclear stain is channel
    0, named ``dapi`` so the store resolves it as the nuclear channel.
    """
    from coral.io.harmonize import harmonize_to_canonical
    from coral.io.readers import read_channel_tiff_dir, read_ometiff

    blocks: list[np.ndarray] = []
    names: list[str] = []
    mpp: float | None = None
    if files.nuclear_image is not None:
        raw, ch, mpp_n, meta = read_ometiff(files.nuclear_image)
        img, _cc = harmonize_to_canonical(raw, ch, meta["axes"])
        blocks.append(img)
        names.append(_NUCLEAR_NAME)
        mpp = mpp or mpp_n
    if files.protein_dir is not None:
        raw, ch, mpp_p, meta = read_channel_tiff_dir(files.protein_dir)
        img, cc = harmonize_to_canonical(raw, ch, meta["axes"])
        blocks.append(img)
        names.extend(str(c.raw) for c in cc)
        mpp = mpp or mpp_p

    if not blocks:
        raise FileNotFoundError(
            f"G4X sample {files.base.name} has no protein/ images and no "
            f"nuclear stain, so there is nothing to build a store image from."
        )
    yx = {(b.shape[1], b.shape[2]) for b in blocks}
    if len(yx) != 1:
        raise ValueError(
            f"G4X image blocks differ in y/x: {[b.shape for b in blocks]}"
        )
    image = np.concatenate(blocks, axis=0)
    logger.info("      G4X image: %s, channels=%s", image.shape, names)
    return image, names, mpp


def read_g4x_sample(files: G4XFiles) -> tuple[Any, np.ndarray, dict]:
    """Read a G4X sample into ``(adata, image, details)``.

    ``adata.X`` is raw RNA counts (cells x genes); ``obsm["spatial"]`` is the
    ``(x, y)`` cell centroid in image pixels; ``obsm["protein"]`` holds the
    per-cell protein intensities. ``image`` is the ``(c, y, x)`` protein+nuclear
    stack; ``details["channel_names"]`` names its channels.
    """
    import anndata as ad
    import pandas as pd
    from scipy.sparse import csr_matrix

    tx = pd.read_csv(files.cell_by_transcript)
    if "label" not in tx.columns:
        raise ValueError(
            f"{files.cell_by_transcript} has no 'label' column"
        )
    tx = tx.set_index("label")
    tx.index = tx.index.astype(str)
    gene_cols = list(tx.columns)
    counts = csr_matrix(tx.to_numpy(dtype=np.float32))

    meta = pd.read_csv(files.cell_metadata).set_index("label")
    meta.index = meta.index.astype(str)
    meta = meta.reindex(tx.index)

    adata = ad.AnnData(X=counts, obs=meta.copy())
    adata.obs_names = tx.index.to_numpy()
    adata.var_names = np.asarray(gene_cols, dtype=object)
    adata.obsm["spatial"] = meta[["cell_x", "cell_y"]].to_numpy(dtype="float64")

    protein_names: list[str] = []
    if files.cell_by_protein is not None:
        prot = pd.read_csv(files.cell_by_protein).set_index("label")
        prot.index = prot.index.astype(str)
        prot = prot.reindex(tx.index)
        protein_names = [
            c[: -len(_INTENSITY_SUFFIX)] if c.endswith(_INTENSITY_SUFFIX) else c
            for c in prot.columns
        ]
        adata.obsm["protein"] = prot.to_numpy(dtype=np.float32)

    image, channel_names, img_mpp = _build_image(files)

    run_meta: dict = {}
    if files.run_meta is not None:
        run_meta = json.loads(Path(files.run_meta).read_text(encoding="utf-8"))

    details = {
        "channel_names": channel_names,
        "protein_names": protein_names,
        "vendor": "singular_genomics_g4x",
        "run_meta": run_meta,
        "n_genes": len(gene_cols),
        "n_protein": len(protein_names),
        # located but NOT ingested in v1 — a later phase writes these
        "segmentation_path": (
            str(files.segmentation) if files.segmentation else None
        ),
        "transcripts_path": (
            str(files.transcripts) if files.transcripts else None
        ),
    }
    if img_mpp:
        # a full G4X export carries PhysicalSize in its OME-XML; the subset's
        # plain TIFFs do not, so this stays None and the resolver falls back to
        # the platform default (0.3125), which then requires confirmation.
        details["pixel_size_um"] = float(img_mpp)
        details["pixel_size_source"] = "OME-XML PhysicalSizeX"
        details["pixel_size_tier"] = "instrument"
    details["transcripts"] = _read_g4x_transcripts(files)
    logger.info(
        "      G4X: %d cells x %d genes, %d protein markers",
        adata.n_obs, adata.n_vars, len(protein_names),
    )
    return adata, image, details


def _read_g4x_transcripts(files: G4XFiles) -> Any:
    """Transcript points in image pixels (same frame as the cell centroids).

    ``x_pixel_coordinate``/``y_pixel_coordinate`` are already in the image pixel
    frame, so no transform is needed. ``gdna`` is the genomic-DNA control.
    Returns ``None`` when the sample has no transcript table.
    """
    if files.transcripts is None:
        return None
    import pandas as pd

    from coral.st.points import normalize_points

    df = pd.read_csv(files.transcripts)
    is_gene = df["gene_name"].astype(str) != "gdna"
    return normalize_points(
        df, x="x_pixel_coordinate", y="y_pixel_coordinate",
        feature="gene_name", cell_id="cell_id", is_gene=is_gene.to_numpy(),
        qv="confidence_score", z="z_level",
    )
