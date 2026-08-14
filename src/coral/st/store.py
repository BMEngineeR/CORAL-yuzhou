"""An AnnData plus an image becomes a CORAL store.

This is where the two worlds meet, and the design goal is that neither audience
is forced into the other's format:

- The image is an OME-NGFF pyramid at the store root, so Viv, QuPath and
  napari open it with no conversion, exactly as they open every other CORAL
  store.
- The table is an **AnnData-zarr group** at ``st/<technology>/``, so
  ``anndata.read_zarr(store / "st" / "visium")`` returns the object a scanpy
  user expects, with no conversion and no h5ad.

One store holds both. Two stores linked by a manifest was the alternative and
was rejected: it makes it possible to move one without the other.

**Everything is zarr v2.** CORAL already pins ``_ZARR_FORMAT = 2`` because v3
breaks QuPath and Bio-Formats, and Viv reads v2. anndata defaults to v3, which
writes ``zarr.json`` and puts chunks under ``c/``; such a store lists, and
validates, and renders as nothing. So the AnnData is written through
``write_dispatched`` into a group opened at v2.

**Coordinates are converted here, once.** Readers emit ``fullres_pixels``,
``array`` or ``microns``, and every consumer downstream would otherwise have to
branch on which. The store always carries pixels in its own level-0 space; the
original frame and the scale applied are recorded in ``config.json``. Nothing
after this asks what frame it is in.

No ``.h5ad``. Zarr is chunked, so a viewer reads coordinates without touching
the counts; a 175 MB h5ad makes that impossible. Anyone who wants a file writes
one with ``ad.read_zarr(...).write_h5ad(...)``.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from coral import __version__
from coral.io.atomic import atomic_write_json

logger = logging.getLogger(__name__)

#: The group ST occupies, named and discovered exactly as `tissue/tissue_<method>/`
#: and `patches/<slug>/` are.
ST_GROUP = "st"

CONFIG_FILE = "config.json"

#: The on-disk shape of one ST record. Bump BY HAND when that shape changes.
#:
#: Deliberately not the package version, which sits beside it in the same config
#: as `coral_version` and answers a different question: that one says what code
#: wrote this, this one says what shape it is. Tying them together would make
#: every release look like a format change and every format change inside a
#: release invisible.
ST_SCHEMA_VERSION = 1

#: What an RGB image's channels must be drawn as. Not a preference: these ARE
#: the channels, and any other assignment renders a different picture from the
#: one the scanner produced.
_RGB_COLOURS = {0: "FF0000", 1: "00FF00", 2: "0000FF"}

#: Matching `coral.io.ingest._ZARR_FORMAT`. Stated here rather than imported so
#: this module says out loud that it depends on the choice.
_ZARR_FORMAT = 2

#: Row/column stride the display window is measured on, matching the stride
#: `coral.io.ingest` measures a proteomics plane with. Stated here rather than
#: imported for the same reason as `_ZARR_FORMAT`: this module depends on the
#: choice and should say so. Every tenth pixel is a SUBSAMPLE, which leaves the
#: value distribution alone, so the percentile is the plane's own. Averaging
#: pixels together does not, which is why a reduced pyramid level is the wrong
#: place to read it.
_WINDOW_STRIDE = 10


def st_dir(store: Path, technology: str) -> Path:
    """Where one technology's table lives inside a store.

    Example:
        >>> st_dir(Path("/data/s.zarr"), "Visium").as_posix().endswith("st/visium")
        True
    """
    return Path(store) / ST_GROUP / technology.lower().replace(" ", "_")


def list_st_records(store: Path) -> list[str]:
    """Technologies a store carries, by scanning rather than by an allowlist.

    The same discovery rule as `coral.tissue.paths.list_tissue_methods`: a
    record counts if its config is there, so a second technology on the same
    sample appears without this module changing.
    """
    base = Path(store) / ST_GROUP
    if not base.is_dir():
        return []
    return sorted(
        p.name for p in base.iterdir() if p.is_dir() and (p / CONFIG_FILE).is_file()
    )


def read_st_config(store: Path, technology: str) -> dict[str, Any]:
    """One ST record's config, refusing a shape this build cannot read.

    THE REFUSAL IS THE POINT. CORAL already writes two version numbers,
    ``SCHEMA_VERSION`` and ``_FEATURE_SCHEMA_VERSION``, that nothing ever reads,
    and both have been bumped, so bytes written under an older meaning are read
    today under the current one with no check. A version nobody reads is worse
    than none, because it looks like migration and is not.

    There is deliberately NO migration here. A newer store is refused by name,
    an older known one is accepted, and anything from before versioning is
    refused with re-ingestion named as the fix. Refusal is a day of work and
    honest; migration is where the largest comparable project spent its biggest
    effort and its answer is still "rewrite the store somewhere else".

    Args:
        store: The ``.zarr`` store.
        technology: Canonical technology name, as passed to `st_dir`.

    Returns:
        The config, unchanged.

    Raises:
        FileNotFoundError: If the record has no config.
        ValueError: If the config predates versioning, or declares a version
            this build does not understand.

    Example:
        >>> read_st_config(Path("/data/s.zarr"), "Visium")["technology"]
        ... # doctest: +SKIP
        'Visium'
    """
    path = st_dir(store, technology) / CONFIG_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"{Path(store).name} has no {technology} record: {path} is missing."
        )
    config = json.loads(path.read_text(encoding="utf-8"))
    found = config.get("st_schema_version")
    if found is None:
        raise ValueError(
            f"{path} records no st_schema_version, so it was written before "
            f"store versioning and its shape cannot be confirmed. Re-ingest it "
            f"with `coral ingest-st`."
        )
    # Two different failures, and telling someone to upgrade CORAL when the
    # value is the string "1" sends them somewhere no upgrade will help.
    if not isinstance(found, int) or isinstance(found, bool):
        raise ValueError(
            f"{path} declares st_schema_version {found!r}, which is not an "
            f"integer. The file has been edited or written by something that is "
            f"not CORAL."
        )
    if found > ST_SCHEMA_VERSION:
        raise ValueError(
            f"{path} declares st_schema_version {found}; this build of CORAL "
            f"understands up to {ST_SCHEMA_VERSION}. Upgrade CORAL, or "
            f"re-ingest the sample with this build."
        )
    return config


def _windows(image: np.ndarray) -> dict[int, tuple[float, float]]:
    """A measured display range per channel, so the store opens looking right.

    WITHOUT THIS AN ST STORE RENDERS AS A BLACK RECTANGLE. `_write_ngff_external_attrs`
    falls back to `(0, default_end)` when it is given no windows, which for
    uint16 is 0 to 4096. A Xenium morphology channel's 99.5th percentile is
    between 219 and 300, so the whole image displays inside the bottom 5% of
    the range: it is technically present and visually absent. Measured in the
    browser before this existed, mean channel value 1.3 out of 255.

    The proteomics ingest has always measured this (`io/ingest.py:_measured_window`) and the
    ST writer simply never did. Same percentiles, same helper, same pixels, so
    the two paths cannot drift.

    NOT for an RGB image. An H&E is already display-ready and its channels are
    not intensities to be stretched: measuring it produced r 50..151, g 37..153,
    b 95..148 on the Visium section, which renders as a blown-out version of a
    picture that was correct at 0..255.

    Read from LEVEL 0 on a stride. This used to read the SMALLEST level, on the
    argument that a percentile is a property of the distribution and a reduced
    level has the same distribution for a fraction of the pixels. That argument
    holds for SUBSAMPLING and `write_pyramid` does not subsample: it reduces
    with `_downsample_mean`, and averaging a bright nucleus into its dark
    surround pulls the high end in. On this repo's own Xenium ovary bundle
    (4 levels, 1/8) the smallest level put DAPI's end at 203 where level 0's
    99.5th percentile is 228, and on a 4,096 px synthetic field of sparse
    nuclei (5 levels, 1/16) at 242 where level 0's is 724, which saturates 1.9%
    of the image flat white instead of the intended 0.5%. The deeper the
    pyramid the wider the gap, and a Xenium morphology image gets the deepest
    pyramid precisely because it is large.

    It is also the cheaper read: level 0 arrives materialised here because the
    caller decoded it (`st/pyramid.py` says so), while the smallest level had
    to be read back out of the store that had just been written. On the ovary
    bundle that is 20.4 ms of zarr read against 5.4 ms of strided percentile.

    Args:
        image: Level 0 as ``(c, y, x)``, the same array `write_pyramid` was
            given.
    """
    from coral.io.ingest import _measured_window

    out: dict[int, tuple[float, float]] = {}
    for channel in range(image.shape[0]):
        plane = image[channel][::_WINDOW_STRIDE, ::_WINDOW_STRIDE]
        window = _measured_window(plane)
        if window is not None:
            out[channel] = window
    logger.info(
        "      display windows measured on level 0: %d channel(s)", len(out)
    )
    return out


def _refuse_foreign_overwrite(
    out_path: Path, technology: str, source: dict[str, Any] | None
) -> None:
    """Refuse to replace a store that was built from a DIFFERENT sample.

    The store is named after the sample directory, and two bundles from
    different vendors are both routinely called ``raw``. Ingesting them one
    command at a time into one job directory then makes the second silently
    replace the first: same name, no warning, exit code 0. The in-command
    duplicate-name check cannot see this, because each command only knows its
    own samples.

    Re-ingesting the same sample FROM THE SAME PATH stays allowed and
    overwrites, because that is the ordinary way to pick up a code change,
    which is exactly what an unversioned store is told to do.

    The tradeoff, stated because it will be met: MOVING a bundle also trips
    this, since provenance is a path and the path changed. The store cannot
    tell that apart from a different sample, and the message names both ways
    out. Refusing and being occasionally annoying is the right side to err on
    when the alternative is silently destroying someone's ingest.

    The comparison is the recorded ``source_bundle``, which exists because
    provenance was written into the config for auditing. This is the first
    thing to actually read it.
    """
    if not out_path.exists():
        return
    incoming = str((source or {}).get("sample_dir") or "")
    for existing in list_st_records(out_path):
        config_path = Path(out_path) / ST_GROUP / existing / CONFIG_FILE
        try:
            recorded = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # Unreadable is not the same as foreign. A half-written store is
            # replaceable; refusing here would strand it with no way forward.
            continue
        was = str(recorded.get("source", {}).get("sample_dir") or "")
        if was and incoming and was != incoming:
            raise ValueError(
                f"{out_path} already holds a {recorded.get('technology', existing)} "
                f"record ingested from\n  {was}\nand this run would replace it "
                f"with\n  {incoming}\nBoth samples resolve to the same store "
                f"name. Give them distinct directory names, or use a separate "
                f"--job-dir."
            )


def _to_pixels(
    coordinates: np.ndarray,
    *,
    frame: str,
    image_scale: float,
    pixel_size_um: float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Coordinates in the store's own level-0 pixels, whatever they arrived as.

    Returns the converted array and the record of what was done to it, because
    a transform nobody can see is a transform nobody can check.

    Raises:
        ValueError: For a frame that cannot be placed on the image. ``array``
            is spot indices on a capture grid, which carry no pixel scale, so
            refusing is the only honest answer.
    """
    applied: dict[str, Any] = {"original_frame": frame, "image_scale": image_scale}
    if frame == "fullres_pixels":
        # The reader's pixels are full resolution; the store may be built from
        # a downsample of that image, which is the ordinary case for a public
        # Visium bundle shipping only a hires PNG.
        applied["rule"] = "full-resolution pixels scaled to the stored image"
        return coordinates * image_scale, applied
    if frame == "microns":
        if not pixel_size_um:
            raise ValueError(
                "coordinates are in microns but the sample records no pixel "
                "size, so they cannot be placed on the image."
            )
        applied["rule"] = "microns divided by the image pixel size"
        applied["pixel_size_um"] = pixel_size_um
        return coordinates / pixel_size_um * image_scale, applied
    raise ValueError(
        f"coordinate frame {frame!r} cannot be placed on an image. "
        f"`array` coordinates are capture-grid indices and carry no scale; "
        f"re-ingest with an alignment so the reader emits pixels."
    )


def write_st_store(
    adata: Any,  # noqa: ANN401 - an AnnData; imported lazily by the caller
    image: np.ndarray,
    out_path: Path,
    *,
    technology: str,
    coordinate_frame: str,
    obs_unit: str,
    mpp: float,
    image_scale: float = 1.0,
    pixel_size_um: float | None = None,
    obs_diameter_um: float | None = None,
    channel_names: list[str] | None = None,
    source: dict[str, Any] | None = None,
    points: Any = None,  # noqa: ANN401 — an optional transcript-points DataFrame
) -> Path:
    """Write one ST sample as a CORAL store.

    Args:
        adata: The reader's canonical AnnData. Its ``obsm["spatial"]`` is
            converted to store pixels and written back into the stored copy.
        image: Level 0 as ``(c, y, x)``.
        out_path: The ``.zarr`` directory to create.
        technology: Canonical technology name.
        coordinate_frame: The frame ``adata.obsm["spatial"]`` arrived in.
        obs_unit: spot, cell, or a bin size.
        mpp: Microns per pixel OF THE STORED IMAGE, which is the source pixel
            size divided by ``image_scale``.
        image_scale: Stored pixels per full-resolution pixel. 1.0 when the
            image is full resolution.
        pixel_size_um: Full-resolution microns per pixel, recorded and used to
            convert micron coordinates.
        obs_diameter_um: What one observation physically covers. The number
            that makes a spot map interpretable, so it is stored rather than
            left in a per-reader detail dict.
        channel_names: Names for the image channels. Defaults to r, g, b for a
            three-channel image and c0.. otherwise.
        source: Provenance, written verbatim into the record.

    Returns:
        ``out_path``.
    """
    import zarr
    from anndata.experimental import write_dispatched

    from coral.io.ingest import _dtype_window_max, _write_ngff_external_attrs
    from coral.st.schema import ST_UNS_KEY, canonicalize_st_adata
    from coral.slide.state import default_state, save_state
    from coral.slide.structure import write_structure_map
    from coral.st.pyramid import plan_levels, write_pyramid

    out_path = Path(out_path)
    _refuse_foreign_overwrite(out_path, technology, source)
    started = time.time()

    markers = channel_names or (
        ["r", "g", "b"]
        if image.shape[0] == 3
        else [f"c{i}" for i in range(image.shape[0])]
    )
    # Whether these are colour components or measured stains. Decided from the
    # NAMES, not from "the reader gave us none": `_image_array` returns literal
    # ["r", "g", "b"] for anything it opened with PIL, so testing channel_names
    # for None says RGB is never RGB. The distinction matters twice below, and
    # the visible one is contrast: an H&E is already display-ready, and
    # percentile-stretching it turns a pink section into a saturated one.
    is_rgb = [m.lower() for m in markers] == ["r", "g", "b"]
    if len(markers) != image.shape[0]:
        raise ValueError(
            f"{len(markers)} channel name(s) for {image.shape[0]} channel(s)"
        )

    logger.info("[1/5] validating the table")
    # VALIDATED BEFORE ANYTHING IS WRITTEN, and both halves of that were
    # missing. The CORAL path reached the store writer straight from a reader,
    # so nothing checked that counts were non-negative integers, that obs and
    # var names were unique, or that the coordinates were finite. Those are the
    # schema's whole job. It also stamps uns["coral"], which is how an AnnData
    # opened on its own says what technology it is and what one of its rows
    # covers. And it runs FIRST: the image write is the expensive step, and a
    # table that will be refused must be refused before minutes of pyramid
    # exist for it, not after, when the refusal strands a half-written store.
    schema = canonicalize_st_adata(
        adata,
        technology=technology,
        sample_id=out_path.stem,
        coordinate_frame=coordinate_frame,
        obs_unit=obs_unit,
        obs_diameter_um=obs_diameter_um,
        technology_details=(source or {}).get("technology_details"),
    )
    obs_diameter_um = schema.obs_diameter_um
    coordinates = np.asarray(adata.obsm["spatial"], dtype="float64")
    placed, applied = _to_pixels(
        coordinates, frame=coordinate_frame,
        image_scale=image_scale, pixel_size_um=pixel_size_um,
    )

    logger.info("[2/5] planning the pyramid from %s", tuple(image.shape))
    shapes = plan_levels(tuple(image.shape))
    logger.info("      %d level(s): %s", len(shapes), shapes)

    logger.info("[3/5] writing the image")
    out_path.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(out_path), mode="w")
    write_pyramid(root, image, shapes)
    # An H&E is RGB and has no nuclear stain, so None is the correct answer
    # there and inventing one would be worse. A Xenium morphology image is a
    # real multi-channel panel whose first channel IS DAPI, and calling that
    # None throws away the one thing a viewer needs to pick a sensible default
    # channel. So it is named when the panel says so and None when it does not.
    nuclear = next((m for m in markers if m.strip().lower() == "dapi"), None)
    root.attrs["channels"] = [
        {
            "marker": m,
            "raw": m,
            "match": "RGB" if nuclear is None else "RESOLVED",
            "keep": True,
        }
        for m in markers
    ]
    root.attrs["nuclear_channel"] = nuclear
    root.attrs["mpp"] = float(mpp)
    root.attrs["source_pyramid_levels"] = len(shapes)
    root.attrs["modality"] = "spatial_transcriptomics"
    root.attrs["technology"] = technology
    # CORAL's own writer, so multiscales, omero and OME/METADATA.ome.xml are
    # produced exactly as they are for a proteomics slide rather than by a
    # second implementation that drifts.
    _write_ngff_external_attrs(
        root, markers, float(mpp), nuclear_marker=nuclear,
        level_shapes=shapes, dtype=image.dtype,
        # An RGB image's window is not measured, it is KNOWN: the full dtype
        # range is what the scanner encoded and what every other viewer shows.
        # Stated as a window rather than left to the fallback so the store says
        # the range is deliberate. The bytes are identical either way
        # (_dtype_window_max is 255.0 for uint8 and the fallback end is
        # min(4096, 255)); what changes is that a reader now trusts it instead
        # of re-measuring an H&E and blowing the section out.
        windows=(
            dict.fromkeys(
                range(image.shape[0]), (0.0, _dtype_window_max(image.dtype))
            )
            if is_rgb
            else _windows(image)
        ),
        # RED IS RED. Without this the writer falls back to its marker palette,
        # which starts blue because a proteomics panel's first channel is the
        # nuclear stain, and an RGB image comes out with its red and blue
        # channels swapped: an H&E's haematoxylin nuclei render red, its eosin
        # renders blue, and the section reads as uniformly pink with no purple
        # in it at all. Compared against the same image in QuPath, which shows
        # the purple.
        channel_colors=_RGB_COLOURS if is_rgb else None,
    )

    logger.info("[4/5] writing the table as AnnData-zarr")
    stored = adata.copy()
    stored.obsm["spatial"] = placed
    # The frame the STORE is in, replacing the one the reader produced, so a
    # reader of the table alone is not left guessing which of three it got.
    if isinstance(stored.uns.get(ST_UNS_KEY), dict):
        stored.uns[ST_UNS_KEY]["coordinate_frame"] = "store_pixels"
        stored.uns[ST_UNS_KEY]["original_coordinate_frame"] = coordinate_frame
        # PROVENANCE TRAVELS WITH THE TABLE, not only in config.json beside it.
        # A table is the thing that gets copied out: someone opens the store in
        # scanpy, writes an h5ad, mails it. config.json does not go with it, and
        # then nothing on earth says which bundle those counts came from or what
        # wrote them. Six fields is a small duplication against losing that.
        stored.uns[ST_UNS_KEY].update({
            "source_bundle": str((source or {}).get("sample_dir") or ""),
            "source_image": str((source or {}).get("image") or ""),
            "coral_version": __version__,
            "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })

    record = st_dir(out_path, technology)
    record.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(record), mode="w")

    def _v2(func: Any, dest: Any, key: str, elem: Any, dataset_kwargs: Any, iospec: Any) -> None:  # noqa: ANN401
        # anndata writes v3 by default and there is no format argument on
        # write_zarr, so the group is opened at v2 and each element written
        # into it through the dispatcher.
        func(dest, key, elem, dataset_kwargs=dataset_kwargs)

    # pandas 2.x hands string columns back as nullable String/ArrowString
    # arrays, which anndata < 0.11 refuses to write unless opted in. Opt in
    # for this write only, then restore — the on-disk arrays are identical
    # plain string arrays either way.
    import anndata as _ad

    _prev_nullable = getattr(_ad.settings, "allow_write_nullable_strings", None)
    if _prev_nullable is not None:
        _ad.settings.allow_write_nullable_strings = True
    try:
        write_dispatched(group, "/", stored, callback=_v2)
    finally:
        if _prev_nullable is not None:
            _ad.settings.allow_write_nullable_strings = _prev_nullable

    # ---- points/ sidecar: transcript points, additive, frozen core untouched
    # Baked with the SAME _to_pixels as obsm["spatial"], so points and cells land
    # in one store-pixel frame. A top-level points/ dir, sibling of st/.
    points_config = None
    if points is not None and len(points):
        placed_pts, _ = _to_pixels(
            points[["x", "y"]].to_numpy(dtype="float64"),
            frame=coordinate_frame,
            image_scale=image_scale,
            pixel_size_um=pixel_size_um,
        )
        pts = points.copy()
        pts["x"] = placed_pts[:, 0].astype("float32")
        pts["y"] = placed_pts[:, 1].astype("float32")
        points_dir = out_path / "points"
        points_dir.mkdir(parents=True, exist_ok=True)
        pts.to_parquet(points_dir / "transcripts.parquet", index=False)
        n_genes = int(pts["is_gene"].sum()) if "is_gene" in pts else int(len(pts))
        points_config = {
            "n_points": int(len(pts)),
            "n_genes": n_genes,
            "feature_column": "feature_name",
            "path": "points/transcripts.parquet",
        }
        logger.info(
            "      wrote %d transcript points (%d gene) to points/",
            len(pts), n_genes,
        )

    atomic_write_json(record / CONFIG_FILE, {
        "st_schema_version": ST_SCHEMA_VERSION,
        "technology": technology,
        "obs_unit": obs_unit,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "obs_diameter_um": obs_diameter_um,
        "pixel_size_um_fullres": pixel_size_um,
        "mpp": float(mpp),
        "coordinates": applied,
        "points": points_config,
        "source": source or {},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })

    logger.info("[5/5] recording the stage")
    state = default_state(out_path, image_path=str((source or {}).get("image") or ""))
    state.tasks.ingest.status = "completed"
    state.tasks.ingest.outputs = {"image": "0"}
    # Keyed by the RECORD DIRECTORY NAME, not technology.lower(): the two
    # differ for "Spatial Transcriptomics" ("spatial transcriptomics" against
    # "spatial_transcriptomics"), and anything joining the state ledger to the
    # records on disk would have needed both spellings.
    st_outputs = {
        "table": f"{ST_GROUP}/{record.name}",
        "config": f"{ST_GROUP}/{record.name}/{CONFIG_FILE}",
    }
    if points_config is not None:
        st_outputs["points"] = points_config["path"]
    state.tasks.st[record.name] = type(state.tasks.ingest)(
        status="completed",
        outputs=st_outputs,
    )
    state.meta.dimensions = (int(image.shape[1]), int(image.shape[2]))
    state.meta.n_markers = int(image.shape[0])
    state.meta.mpp = float(mpp)
    save_state(out_path, state)
    write_structure_map(out_path)

    # READ BACK THROUGH THE ENFORCING READER. Nothing else in CORAL calls
    # `read_st_config` yet, and a check nobody runs is exactly the failure being
    # fixed here. Doing it on every ingest means the writer and the reader
    # cannot drift apart without the next run saying so.
    read_st_config(out_path, technology)

    logger.info(
        "wrote %s in %.1fs: %d %ss, %d genes, %d level(s)",
        out_path.name, time.time() - started, adata.n_obs, obs_unit,
        adata.n_vars, len(shapes),
    )
    return out_path
