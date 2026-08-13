"""Which technology produced a sample directory, decided by looking at it.

Every vendor bundle is self-identifying, so making a person pass
``--technology`` asks them for something the files already say, and it is the
one flag they are most likely to get wrong. It is also what stands between
ingesting a cohort of mixed technologies in one command and writing a loop with
a lookup table.

**Order matters, and not for taste.** Three of the five discoverers match
directories that belong to another technology, so a naive "try them all and
take the first" gets the wrong answer on real bundles:

- A **Visium HD** output also satisfies the Visium discoverer, because that one
  searches recursively for ``*filtered_feature_bc_matrix.h5`` and descends
  happily into ``binned_outputs/square_002um/``.
- A **Xenium** output also satisfies the Visium discoverer, because standard
  XOA output contains a ``cell_feature_matrix/`` MEX folder holding
  ``matrix.mtx.gz``, which is Visium's fallback counts layout.
- An **ingested sample directory** satisfies the HEST discoverer, which wants
  exactly one ``.h5ad``, one TIFF and one ``metadata.json``. ESB's own output
  is exactly that, and so is anything that writes those three names.

So the specific technologies are probed before the general ones, and HEST is
additionally required to carry a technology in its metadata, which is the check
that separates a real HEST sample from something that merely looks like one.

A directory that matches nothing is refused by name, listing what each reader
looked for. A directory that matches SEVERAL is not refused, because the three
overlaps above mean that is the expected case rather than the broken one: the
order is what resolves it, and the first match wins. Every match is logged when
there is more than one, so the ordering assumption is visible in a run rather
than silent.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

#: Probed in this order. The comment beside each says what makes it specific
#: enough to come before the ones under it.
_PROBE_ORDER: tuple[str, ...] = (
    # Requires flatFiles/ + RawFiles/ with an AtoMx sample layout — a
    # signature no other vendor writes.
    "CosMx",
    # Requires a single_cell_data/ folder with G4X's exact table names — a
    # signature no other vendor writes.
    "G4X",
    # Requires experiment.xenium, which nothing else writes.
    "Xenium",
    # Requires binned_outputs/ or segmented_outputs/ at fixed paths.
    "VisiumHD",
    # Requires exactly one h5ad + one TIFF + one metadata.json naming a
    # technology. Before Visium because its h5ad is not a counts matrix Visium
    # would find, but after the two above because their outputs can contain
    # stray single files.
    "HEST",
    # Recursive, so it matches anything containing a 10x matrix anywhere.
    "Visium",
    # Non-recursive and refuses ambiguity itself, so it is safe last.
    "Spatial Transcriptomics",
)


def _matches_cosmx(path: Path) -> bool:
    from coral.st.cosmx import discover_cosmx_files

    discover_cosmx_files(path)
    return True


def _matches_g4x(path: Path) -> bool:
    from coral.st.g4x import discover_g4x_files

    discover_g4x_files(path)
    return True


def _matches_xenium(path: Path) -> bool:
    from coral.st.xenium import discover_xenium_files

    discover_xenium_files(path)
    return True


def _matches_visium_hd(path: Path) -> bool:
    from coral.st.visium_hd import discover_visium_hd_files

    # Returns a LIST, and an empty one is not a match. A FileNotFoundError from
    # here means "this IS Visium HD and it is incomplete", which the caller
    # should see rather than have it silently fall through to Visium.
    return bool(discover_visium_hd_files(path))


def _matches_hest(path: Path) -> bool:
    from coral.st.hest import discover_hest_files

    files = discover_hest_files(path)
    # The discoverer counts files; it does not check that the metadata is
    # HEST's. Without this an ingested sample directory reads back as HEST,
    # including CORAL's own output.
    meta = json.loads(Path(files.metadata).read_text(encoding="utf-8"))
    return bool(str(meta.get("st_technology", "")).strip())


def _matches_visium(path: Path) -> bool:
    from coral.st.visium import discover_visium_files

    discover_visium_files(path)
    return True


def _matches_legacy(path: Path) -> bool:
    from coral.st.legacy_st import discover_legacy_st_files

    discover_legacy_st_files(path)
    return True


_PROBES = {
    "CosMx": _matches_cosmx,
    "G4X": _matches_g4x,
    "Xenium": _matches_xenium,
    "VisiumHD": _matches_visium_hd,
    "HEST": _matches_hest,
    "Visium": _matches_visium,
    "Spatial Transcriptomics": _matches_legacy,
}


def detect_technology(input_dir: Path) -> str:
    """The technology a sample directory holds.

    Args:
        input_dir: One sample's raw bundle.

    Returns:
        A canonical technology name, or ``"HEST"`` for a HEST sample, whose
        own metadata names the technology its counts came from.

    Raises:
        FileNotFoundError: If ``input_dir`` is not a directory.
        ValueError: If nothing matches, naming what each reader looked for.
            Several matches is NOT an error: see the module docstring, the
            probe order exists because bundles overlap, and the first wins.

    Example:
        >>> detect_technology(Path("/data/V1_Human_Lymph_Node"))  # doctest: +SKIP
        'Visium'
    """
    path = Path(input_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is not a directory")

    matched: list[str] = []
    reasons: dict[str, str] = {}
    for technology in _PROBE_ORDER:
        try:
            if _PROBES[technology](path):
                matched.append(technology)
        except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
            reasons[technology] = str(exc).split("\n", 1)[0]

    if not matched:
        looked = "\n".join(f"  {name}: {why}" for name, why in reasons.items())
        raise ValueError(
            f"{path.name}: no supported spatial transcriptomics bundle found. "
            f"What each reader looked for:\n{looked}\n"
            f"Pass --technology to force one."
        )

    if len(matched) > 1:
        # NOT an error, and this docstring used to say it was. The probe order
        # exists BECAUSE bundles match more than one discoverer, and the top of
        # this module names three such overlaps. Refusing here would reject the
        # very bundles the order was built to resolve. It is logged rather than
        # discarded because the order is the whole safety argument, and this is
        # the only place a run would ever show it being exercised.
        logger.info(
            "%s matched %s; taking %s, ranked most specific by the probe order",
            path.name, ", ".join(matched), matched[0],
        )
    else:
        logger.debug("%s matched %s", path.name, matched[0])
    return matched[0]
