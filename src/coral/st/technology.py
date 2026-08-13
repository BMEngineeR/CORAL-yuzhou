"""Which spatial transcriptomics technologies CORAL knows, and their names.

Its own module, and this is not tidying. `schema` needs exactly one
function from here, `normalize_technology`, and in ESB that function lives in
`preprocess.py`, which imports `genes`, `normalize`, `qc` and `selection`, each
of which imports scanpy. So validating a technology name pulled the entire
analysis stack into the ingest path.

That mattered here because CORAL's ST sprint is ingest and viewing only: no QC,
no normalisation, no gene selection. Importing four analysis modules to check a
string against a set of four strings is a dependency nobody asked for, and it
would have made the ingest path fail to import in an environment that has
anndata but not scanpy.

The readers still need scanpy, because `sc.read_10x_h5` is how a Visium matrix
is read. The point is that the SCHEMA does not.
"""

from __future__ import annotations

#: The canonical spelling of every technology a reader can produce.
SUPPORTED_TECHNOLOGIES: frozenset[str] = frozenset(
    {
        "Spatial Transcriptomics",
        "Visium",
        "Xenium",
        "VisiumHD",
        "CosMx",
        "G4X",
    }
)

#: Everything a person or a file might write, mapped to the canonical form.
#: Case is folded before lookup, so only genuinely different spellings appear.
_TECHNOLOGY_ALIASES: dict[str, str] = {
    "spatial transcriptomics": "Spatial Transcriptomics",
    "spatial_transcriptomics": "Spatial Transcriptomics",
    "st": "Spatial Transcriptomics",
    "visium": "Visium",
    "xenium": "Xenium",
    "visium hd": "VisiumHD",
    "visium hd 3'": "VisiumHD",
    "visiumhd": "VisiumHD",
    "visium_hd": "VisiumHD",
    # The hyphenated form the CLI accepts. ESB kept a second lookup table in
    # its HEST reader for exactly this one string; one table is better.
    "visium-hd": "VisiumHD",
    # Imaging-ST platforms ported from SQUAD's readers.
    "cosmx": "CosMx",
    "cos-mx": "CosMx",
    "nanostring": "CosMx",
    "g4x": "G4X",
    "singular": "G4X",
}


def normalize_technology(value: str) -> str:
    """The canonical name for a technology, or a refusal naming what is known.

    Args:
        value: A technology name in any accepted spelling.

    Returns:
        The canonical spelling.

    Raises:
        ValueError: If the name is not one CORAL has a reader for.

    Example:
        >>> normalize_technology("visium hd")
        'VisiumHD'
        >>> normalize_technology("ST")
        'Spatial Transcriptomics'
    """
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError("technology must not be empty")
    canonical = _TECHNOLOGY_ALIASES.get(cleaned.casefold(), cleaned)
    if canonical not in SUPPORTED_TECHNOLOGIES:
        supported = ", ".join(sorted(SUPPORTED_TECHNOLOGIES))
        raise ValueError(
            f"unsupported ST technology {value!r}; supported: {supported}"
        )
    return canonical


#: Where each reader records what one observation physically covers. The value
#: is a diameter in microns for a round spot and a side length for a square
#: bin, which is the same number for the purpose it serves: how much tissue one
#: row of the matrix speaks for.
_DIAMETER_KEYS: tuple[str, ...] = ("spot_diameter_um", "bin_size_um")


def observation_diameter_um(details: dict) -> float | None:
    """What one observation covers, in microns, or None when there is no one size.

    ONE implementation, because the alternative is what happened the first time:
    the value lives under a different key per reader, inside a free-form details
    dict, so a call site that reached for the wrong key silently produced 0.0
    and every spot on the map was drawn as a point.

    Which technologies have it, and why the rest do not:

    - **Visium** fixes it at 55 µm and **legacy ST** at 100 µm by default, both
      properties of the capture slide.
    - **Visium HD** puts the bin side in its own ``scalefactors``, so it is
      found one level down rather than at the top.
    - **Xenium** has none. Its observations are segmented cells of genuinely
      varying size, and any single number would be an average presented as a
      measurement. ``obs`` carries ``cell_area`` per cell for anyone who wants
      an equivalent diameter, which is a different and honest thing.
    - **HEST** passes through whatever its source recorded, which may be
      nothing.

    Returns None rather than 0.0, because zero is a measurement and "this
    technology has no single observation size" is not.

    Example:
        >>> observation_diameter_um({"spot_diameter_um": 55.0})
        55.0
        >>> observation_diameter_um({"scalefactors": {"bin_size_um": 8}})
        8.0
        >>> observation_diameter_um({"pixel_size_morphology_um": 0.2125}) is None
        True
    """
    for key in _DIAMETER_KEYS:
        value = details.get(key)
        if value:
            return float(value)
    scalefactors = details.get("scalefactors")
    if isinstance(scalefactors, dict):
        for key in _DIAMETER_KEYS:
            value = scalefactors.get(key)
            if value:
                return float(value)
    return None
