"""``coral ingest-st`` — convert spatial transcriptomics samples to OME-Zarr."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import typer

from coral.cli._render import fmt_duration
from coral.summary import run_ledger
from coral.utils import setup_logging

logger = logging.getLogger(__name__)


def _detects(path: Path) -> bool:
    """Whether a directory reads as one spatial transcriptomics sample."""
    from coral.st.detect import detect_technology

    try:
        detect_technology(path)
    except (FileNotFoundError, ValueError):
        return False
    return True


def _samples_in(sample_dir: Path, technology: str | None) -> list[Path]:
    """Every sample bundle under a directory, or the directory itself.

    CHILDREN ARE TRIED FIRST, and getting that order wrong silently drops data.
    The Visium and Xenium discoverers search RECURSIVELY, so every ancestor of
    a bundle detects as a sample: a folder holding a Visium and a Xenium sample
    reads as one Xenium sample, and the Visium one is never ingested and never
    mentioned. Testing the directory itself first is what caused that.

    What separates the two cases is measurable rather than assumed: a real
    vendor bundle has NO subdirectory that detects on its own (neither a Space
    Ranger bundle nor a Xenium bundle does), while a directory of samples has
    exactly one detecting child per sample. So a detecting child means this is a
    container, and the search descends; no detecting child means this is the
    sample. Descent terminates because each step goes strictly downwards.

    With `--technology` forced there is nothing to detect, so the directory is
    taken as one sample: a person who names the technology is pointing at a
    specific bundle.
    """
    if technology:
        return [sample_dir]
    if not sample_dir.is_dir():
        return []
    children = [p for p in sorted(sample_dir.iterdir()) if p.is_dir() and _detects(p)]
    if children:
        return [found for child in children for found in _samples_in(child, None)]
    return [sample_dir] if _detects(sample_dir) else []


def ingest_st(
    sample_dir: Path = typer.Option(
        ...,
        "--sample-dir",
        help=(
            "Directory of spatial transcriptomics samples. Each entry inside "
            "it is one sample: a vendor output folder such as a Space Ranger "
            "or Xenium bundle. The technology of each is detected from its "
            "files; pass --technology only to override that."
        ),
    ),
    job_dir: Path = typer.Option(
        ...,
        "--job-dir",
        help=(
            "Output directory: every <name>.zarr store, the logs and the run "
            "summary are written here."
        ),
    ),
    technology: str | None = typer.Option(
        None,
        "--technology",
        help=(
            "Force a reader instead of detecting one. Accepts visium, "
            "visium-hd, xenium, st. Detection is right for every vendor "
            "bundle; this exists for a directory that has been rearranged."
        ),
    ),
    image: Path | None = typer.Option(
        None,
        "--image",
        help=(
            "The image to build the store's pyramid from, when the sample "
            "does not carry a full-resolution one. Only valid with a single "
            "sample."
        ),
    ),
    image_scale: float = typer.Option(
        1.0,
        "--image-scale",
        help=(
            "Stored pixels per full-resolution pixel of --image. A public "
            "Visium bundle ships only a hires PNG, whose scale factor is in "
            "its scalefactors_json.json. 1.0 when the image is full "
            "resolution."
        ),
    ),
    mpp: float | None = typer.Option(
        None,
        "--mpp",
        help=(
            "Explicit full-resolution pixel size (µm/px) that overrides the "
            "resolver. Use it to confirm a value the ingest refused to guess."
        ),
    ),
    confirm_mpp: bool = typer.Option(
        False,
        "--confirm-mpp",
        help=(
            "Accept the resolver's pixel size even when it is not confident "
            "(fell back to a platform default, or a cross-check disagreed). "
            "Without --mpp or this flag, an unconfident mpp blocks the ingest."
        ),
    ),
) -> None:
    """Convert spatial transcriptomics samples into canonical OME-Zarr stores.

    Each sample becomes one store holding BOTH formats natively: an OME-NGFF
    image pyramid at the root, which Viv and QuPath open with no conversion,
    and the counts as an AnnData-zarr group under ``st/<technology>/``, which
    ``anndata.read_zarr`` opens with no conversion. No ``.h5ad`` is written;
    zarr is chunked, so a viewer reads coordinates without touching the counts.

    Coordinates in the store are always PIXELS in its own level-0 space,
    whatever frame the reader produced, with the original frame and the applied
    scale recorded. Nothing downstream has to ask which frame it got.

    Example:
        coral ingest-st --sample-dir raw/st --job-dir out/st
    """
    setup_logging()

    if not sample_dir.is_dir():
        raise typer.BadParameter(f"{sample_dir} is not a directory", param_hint="--sample-dir")

    from coral.st import ST_AVAILABLE, _INSTALL_MSG

    if not ST_AVAILABLE:
        logger.error(_INSTALL_MSG)
        raise typer.Exit(code=1)

    samples = _samples_in(sample_dir, technology)
    if image is not None and len(samples) > 1:
        raise typer.BadParameter(
            f"--image names one image but {len(samples)} samples were found in "
            f"{sample_dir}. Ingest them one directory at a time.",
            param_hint="--image",
        )

    if not samples:
        logger.error(
            "%s holds no spatial transcriptomics sample, and none of its "
            "subdirectories does either.", sample_dir,
        )
        raise typer.Exit(code=1)

    # The store is named after the sample directory, so two samples with the
    # same directory name write the same `<name>.zarr` and the second silently
    # replaces the first. It is a real layout: `st_data/visium/raw` beside
    # `st_data/xenium/raw` gives two samples both called `raw`. Refusing by name
    # is the only honest answer, because any automatic disambiguation invents a
    # store name the person did not choose and cannot predict.
    names = [s.name for s in samples]
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated:
        collided = "\n".join(f"  {s}" for s in samples if s.name in duplicated)
        raise typer.BadParameter(
            f"{len(samples)} samples were found but these directory names "
            f"repeat: {', '.join(duplicated)}. Each store is named after its "
            f"directory, so one would overwrite the other:\n{collided}\n"
            f"Ingest them one --sample-dir at a time, or rename the "
            f"directories so each is unique.",
            param_hint="--sample-dir",
        )

    logger.info("Samples:        %d in %s", len(samples), sample_dir)
    logger.info("Output:         %s", job_dir)
    logger.info("Technology:     %s", technology or "detected per sample")

    job_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    written = 0
    failed: list[tuple[str, str]] = []

    with run_ledger(
        job_dir,
        tool="coral ingest-st",
        args={
            "sample_dir": str(sample_dir),
            "technology": technology,
            "image": str(image) if image else None,
            "image_scale": image_scale,
        },
    ):
        from coral.st.pipeline import ingest_one

        for index, sample in enumerate(samples, start=1):
            logger.info("[%d/%d] %s", index, len(samples), sample.name)
            try:
                out = ingest_one(
                    sample,
                    job_dir,
                    technology=technology,
                    image=image,
                    image_scale=image_scale,
                    mpp=mpp,
                    confirm_mpp=confirm_mpp,
                )
                written += 1
                logger.info("        -> %s", out.name)
            except Exception as exc:  # noqa: BLE001 - one sample must not stop the run
                failed.append((sample.name, str(exc).split("\n", 1)[0]))
                logger.warning("        failed: %s", str(exc).split("\n", 1)[0])

    logger.info(
        "Wrote %d store(s) in %s%s",
        written,
        fmt_duration(time.time() - started),
        f", {len(failed)} failed" if failed else "",
    )
    for name, why in failed:
        logger.warning("  %s: %s", name, why)
    if failed and not written:
        raise typer.Exit(code=1)
