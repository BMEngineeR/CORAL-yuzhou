"""One sample directory becomes one store: detect, read, write.

The three steps are separate modules on purpose, and this is the only place
that knows all three. `detect` decides what a directory is, the readers turn it
into an AnnData, `store` writes the result. Keeping the join here means the CLI
holds no knowledge about technologies, and a caller wanting a store without a
command line has one function to call.

**The readers disagree about what they return, and that is handled here.**
Visium hands back three things, Xenium and Visium HD two, legacy ST returns the
resolved coordinate frame as its middle element because its frame is decided by
the data rather than fixed, and HEST has no reader function at all: its ingest
is inline. Papering over that inside each reader would mean editing five files
that were deliberately copied unchanged.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: Suffixes tifffile owns. PIL cannot open a tiled pyramidal OME-TIFF at all,
#: which is what a Xenium morphology image is, and it raises
#: UnidentifiedImageError rather than anything that names the real problem.
_TIFF_SUFFIXES = frozenset({".tif", ".tiff", ".btf", ".ome"})


def _image_array(path: Path) -> tuple[Any, list[str] | None]:
    """An image as ``(c, y, x)``, plus channel names when the file knows them.

    Two readers, because ST images are two different things. A Space Ranger
    tissue image is an ordinary RGB PNG. A Xenium morphology image is a tiled
    multi-level OME-TIFF carrying four uint16 stains, and PIL cannot open it.

    Returns the array in CORAL's canonical layout and, for a multi-channel
    TIFF, the channel names read off the OME metadata. Names matter here: a
    four-channel morphology image whose channels are called c0 to c3 has lost
    which one is the nuclear stain.
    """
    import numpy as np

    if path.suffix.lower() in _TIFF_SUFFIXES or path.name.lower().endswith(".ome.tif"):
        import tifffile

        with tifffile.TiffFile(path) as handle:
            series = handle.series[0]
            axes = series.axes
            # `.asarray()`, not `np.asarray()`. A TiffPageSeries is iterable
            # over its pages, so numpy wraps it into a (4,) object array of
            # pages rather than reading any pixels, and the failure surfaces
            # much later as a dtype nothing can write.
            level = series.levels[0] if series.levels else series
            array = level.asarray()
            names = _ome_channel_names(handle)
        if axes == "YX":
            array = array[np.newaxis, ...]
        elif axes == "YXS" or (axes.endswith("C") and array.ndim == 3):
            array = np.moveaxis(array, -1, 0)
        elif axes != "CYX" and array.ndim == 3:
            raise ValueError(f"{path.name}: cannot place axes {axes!r} as (c, y, x)")
        if names and len(names) != array.shape[0]:
            names = None
        return array, names

    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    array = np.asarray(Image.open(path).convert("RGB"))
    return array.transpose(2, 0, 1), ["r", "g", "b"]


def _ome_channel_names(handle: Any) -> list[str] | None:  # noqa: ANN401 - a TiffFile
    """Channel names from OME-XML, or None when it carries none."""
    from xml.etree import ElementTree

    try:
        root = ElementTree.fromstring(handle.ome_metadata or "")
    except (ElementTree.ParseError, TypeError):
        return None
    names = [
        element.get("Name") or element.get("ID") or ""
        for element in root.iter()
        if element.tag.endswith("}Channel") or element.tag == "Channel"
    ]
    cleaned = [n.strip() for n in names if n and n.strip()]
    return cleaned or None


def ingest_one(
    sample_dir: Path,
    job_dir: Path,
    *,
    technology: str | None = None,
    image: Path | None = None,
    image_scale: float = 1.0,
    sample_id: str | None = None,
) -> Path:
    """Read one sample and write it as a CORAL store.

    Args:
        sample_dir: One vendor bundle.
        job_dir: Where ``<sample_id>.zarr`` is written.
        technology: Force a reader; ``None`` detects one.
        image: The image to build the pyramid from, when the bundle has no
            full-resolution one of its own.
        image_scale: Stored pixels per full-resolution pixel of ``image``.
        sample_id: Store name; defaults to the directory name.

    Returns:
        The store path.
    """
    from coral.st.detect import detect_technology
    from coral.st.store import write_st_store
    from coral.st.technology import (
        normalize_technology,
        observation_diameter_um,
    )

    sample_dir = Path(sample_dir)
    resolved = (
        normalize_technology(technology) if technology else detect_technology(sample_dir)
    )
    logger.info("        technology: %s", resolved)

    adata, frame, obs_unit, details, source_image, pixel_size, resolved = _read(
        sample_dir, resolved
    )

    picked = image if image is not None else source_image
    if picked is None:
        raise ValueError(
            f"{sample_dir.name} carries no image, and none was given with "
            f"--image, so there is nothing to build a store around."
        )
    scale = image_scale if image is not None else 1.0

    # A reader may hand back an already-decoded ``(c, y, x)`` array instead of a
    # path — the imaging-ST readers (CosMx, G4X) build/mosaic the image in
    # memory and cannot point at a single file. Its channel names travel in
    # ``details``. A path (Visium, Xenium, ...) is decoded here as before.
    if isinstance(picked, np.ndarray):
        pixels = picked
        channel_names = details.get("channel_names")
        source_image_str = ""
    else:
        pixels, channel_names = _image_array(Path(picked))
        source_image_str = str(Path(picked).resolve())
    mpp = float(pixel_size or 0) / max(scale, 1e-12) if pixel_size else 1.0

    return write_st_store(
        adata,
        pixels,
        Path(job_dir) / f"{sample_id or sample_dir.name}.zarr",
        technology=resolved,
        coordinate_frame=frame,
        obs_unit=obs_unit,
        mpp=mpp,
        image_scale=scale,
        pixel_size_um=pixel_size,
        obs_diameter_um=observation_diameter_um(details),
        channel_names=channel_names,
        source={
            "sample_dir": str(sample_dir.resolve()),
            "image": source_image_str,
            "technology_details": details,
        },
    )


def _read(
    sample_dir: Path, technology: str
) -> tuple[Any, str, str, dict, Path | None, float | None, str]:
    """Run the right reader and flatten its answer into one shape.

    Returns ``(adata, coordinate_frame, obs_unit, details, image,
    pixel_size_um, technology)``.

    The technology comes BACK OUT because HEST is not one. A HEST directory is
    a packaging of some real technology's counts, and its metadata names which;
    recording ``"HEST"`` as the store's technology would fail
    `normalize_technology` at validation time, after the pyramid was already
    written. Every other branch returns its input unchanged.
    """
    if technology == "Visium":
        from coral.st.visium import discover_visium_files, read_visium_sample

        files = discover_visium_files(sample_dir)
        with tempfile.TemporaryDirectory(prefix="coral-st-") as scratch:
            adata, details, _ = read_visium_sample(files, artifact_dir=Path(scratch))
        return (
            adata, "fullres_pixels", "spot", details, files.image,
            details.get("pixel_size_um_estimated"), technology,
        )

    if technology == "VisiumHD":
        from coral.st.visium_hd import (
            discover_visium_hd_files,
            read_visium_hd_sample,
        )

        found = discover_visium_hd_files(sample_dir)
        # The discoverer returns one entry per resolution. The finest is the
        # one a viewer wants; the others are aggregations of it.
        adata, details = read_visium_hd_sample(found[0])
        return (
            adata, "fullres_pixels", details.get("selected_obs_unit", "spot"),
            details, found[0].image, None, technology,
        )

    if technology == "Xenium":
        from coral.st.xenium import discover_xenium_files, read_xenium_sample

        files = discover_xenium_files(sample_dir)
        with tempfile.TemporaryDirectory(prefix="coral-st-") as scratch:
            adata, details = read_xenium_sample(files, registration_dir=Path(scratch))
        # Xenium is the one reader whose frame depends on what it was given.
        frame = "fullres_pixels" if files.image else "microns"
        return (
            adata, frame, "cell", details, files.image or files.dapi,
            details.get("pixel_size_morphology_um"), technology,
        )

    if technology == "Spatial Transcriptomics":
        from coral.st.legacy_st import (
            discover_legacy_st_files,
            read_legacy_st_sample,
        )

        files = discover_legacy_st_files(sample_dir)
        # Its middle element IS the frame: legacy ST decides it from the data.
        adata, frame, details = read_legacy_st_sample(files)
        return (
            adata, frame, "spot", dict(details), files.image,
            details.get("pixel_size_um_estimated"), technology,
        )

    if technology == "HEST":
        import anndata as ad
        import numpy as np

        from coral.st.hest import (
            HEST_OBS_UNITS,
            _pooling_status,
            _raw_counts,
            discover_hest_files,
        )
        from coral.st.technology import normalize_technology

        files = discover_hest_files(sample_dir)
        meta = json.loads(Path(files.metadata).read_text(encoding="utf-8"))
        declared = str(meta.get("st_technology") or "").strip()
        if not declared:
            raise ValueError(
                f"{files.metadata} declares no st_technology, so this HEST "
                f"package does not say which technology its counts came from."
            )
        actual = normalize_technology(declared)
        logger.info("        HEST package: counts are %s", actual)
        adata = ad.read_h5ad(files.h5ad)
        # The three repairs the schema will not do itself, because a HEST h5ad
        # routinely arrives with normalised .X beside raw layers and with
        # duplicated gene names: promote the unmodified counts into .X, make
        # var names unique, and warn when the observations look pooled. Each is
        # recorded in the details so the config says what was done.
        counts_source = _raw_counts(adata)
        made_unique = not adata.var_names.is_unique
        if made_unique:
            adata.var_names_make_unique()
        pooling_status, pooling_evidence = "not-detected", {}
        spatial = adata.obsm.get("spatial")
        if spatial is not None and np.asarray(spatial).ndim == 2:
            pooling_status, pooling_evidence = _pooling_status(
                actual, meta, np.asarray(spatial, dtype=float)
            )
        if pooling_status != "not-detected":
            logger.warning(
                "        HEST input is %s: the supplied observations are "
                "preserved, pooling is not reversed (%s).",
                pooling_status, pooling_evidence,
            )
        details = {
            **meta,
            "input_format": "hest",
            "raw_counts_source": counts_source,
            "var_names_made_unique": made_unique,
            "hest_pooling_status": pooling_status,
            "hest_pooling_evidence": pooling_evidence,
        }
        return (
            adata, "fullres_pixels", HEST_OBS_UNITS.get(actual, "spot"),
            details, files.image,
            meta.get("pixel_size_um_embedded") or meta.get("pixel_size_um_estimated"),
            actual,
        )

    if technology == "G4X":
        from coral.st.g4x import discover_g4x_files, read_g4x_sample

        files = discover_g4x_files(sample_dir)
        adata, image, details = read_g4x_sample(files)
        # Global-pixel centroids in the image's own frame; one row is a cell.
        # ``image`` is an in-memory (c, y, x) array, not a path.
        return (
            adata, "fullres_pixels", "cell", details, image,
            details.get("pixel_size_um_estimated"), technology,
        )

    if technology == "CosMx":
        from coral.st.cosmx import discover_cosmx_files, read_cosmx_sample

        layout = discover_cosmx_files(sample_dir)
        adata, image, details = read_cosmx_sample(layout)
        # Cells placed in the compact mosaic's own pixel frame; one row is a
        # cell. ``image`` is the in-memory (c, y, x) mosaic, not a path.
        return (
            adata, "fullres_pixels", "cell", details, image,
            details.get("pixel_size_um_estimated"), technology,
        )

    raise ValueError(f"no reader for technology {technology!r}")
