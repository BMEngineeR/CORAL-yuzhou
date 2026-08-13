"""Reader for legacy Spatial Transcriptomics count-table layouts.

The file-discovery and table-assembly behavior follows the useful parts of
the external STReader convention, with ESB-owned validation and output.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import anndata as ad
import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse

logger = logging.getLogger(__name__)

_TABLE_SUFFIXES = (".tsv", ".tsv.gz", ".csv", ".csv.gz", ".txt")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
_ARRAY_SPOT_RE = re.compile(r"^(?P<x>-?\d+(?:\.\d+)?)x(?P<y>-?\d+(?:\.\d+)?)$")

# Match the upstream reader's large-slide allowance. Pillow's default warning
# threshold is too low for ordinary full-resolution histology JPEGs.
Image.MAX_IMAGE_PIXELS = 93_312_000_000

DEFAULT_SPOT_DIAMETER_UM = 100.0
DEFAULT_INTER_SPOT_DISTANCE_UM = 200.0
LEGACY_SPOT_PACKING = "grid"


@dataclass(frozen=True)
class LegacySTFiles:
    """Resolved source files for one legacy ST sample.

    Args:
        counts: Observation-by-gene raw count table.
        coordinates: Optional spot coordinate table.
        metadata: Optional observation metadata table.
        transform: Optional 3-by-3 array-to-pixel transform.
        image: Optional histology image.

    Example:
        :func:`discover_legacy_st_files` returns this record before reading
        any large expression table.
    """

    counts: Path
    coordinates: Path | None
    metadata: Path | None
    transform: Path | None
    image: Path | None

    def source_files(self) -> list[Path]:
        """Return every resolved source path in stable role order."""
        return [
            self.counts,
            *([self.coordinates] if self.coordinates is not None else []),
            *([self.metadata] if self.metadata is not None else []),
            *([self.transform] if self.transform is not None else []),
            *([self.image] if self.image is not None else []),
        ]


def _matches_suffix(path: Path) -> bool:
    return any(
        path.name.lower().endswith(suffix) for suffix in _TABLE_SUFFIXES
    )


def _resolve_one(
    input_dir: Path,
    override: Path | None,
    *,
    role: str,
    patterns: tuple[str, ...],
    required: bool,
) -> Path | None:
    if override is not None:
        resolved = override if override.is_absolute() else input_dir / override
        if not resolved.is_file():
            raise FileNotFoundError(f"{role} file does not exist: {resolved}")
        logger.info("Legacy ST %s override: %s", role, resolved)
        return resolved
    candidates: list[Path] = []
    matched_pattern: str | None = None
    for pattern in patterns:
        candidates = sorted(
            path for path in input_dir.glob(pattern) if path.is_file()
        )
        if candidates:
            matched_pattern = pattern
            break
    if not candidates:
        if required:
            searched = ", ".join(patterns)
            raise FileNotFoundError(
                f"no legacy ST {role} file found in {input_dir}; "
                f"searched patterns: {searched}"
            )
        logger.info("No optional legacy ST %s file found", role)
        return None
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(
            f"ambiguous legacy ST {role} files in {input_dir}: {names}; "
            f"pass the explicit --{role.replace('_', '-')} path"
        )
    logger.info(
        "Discovered legacy ST %s with pattern %s: %s",
        role,
        matched_pattern,
        candidates[0],
    )
    return candidates[0]


def discover_legacy_st_files(
    input_dir: Path,
    *,
    counts: Path | None = None,
    coordinates: Path | None = None,
    metadata: Path | None = None,
    transform: Path | None = None,
    image: Path | None = None,
    require_image: bool = False,
) -> LegacySTFiles:
    """Discover one legacy ST sample's source files without guessing ties.

    Args:
        input_dir: Directory containing a single sample.
        counts: Optional raw-count table override.
        coordinates: Optional spot-coordinate table override.
        metadata: Optional observation metadata override.
        transform: Optional 3-by-3 transform override.
        image: Optional histology image override.
        require_image: Whether missing histology is an error.

    Returns:
        Resolved source-file record.

    Raises:
        FileNotFoundError: If a required role cannot be resolved.
        ValueError: If discovery produces multiple candidates for one role.

    Example:
        ``discover_legacy_st_files(Path("raw/S1"))`` resolves conventional
        ``stdata``, ``spots``, and H&E filenames.
    """
    if not input_dir.is_dir():
        raise ValueError(
            f"input_dir must be an existing directory: {input_dir}"
        )
    resolved_counts = _resolve_one(
        input_dir,
        counts,
        role="counts",
        patterns=(
            "*stdata*.tsv*",
            "*count*.tsv*",
            "*expression*.tsv*",
            "*matrix*.tsv*",
        ),
        required=True,
    )
    resolved_coordinates = _resolve_one(
        input_dir,
        coordinates,
        role="spot_coords",
        patterns=(
            "spots*.csv*",
            "*spot*coord*.csv*",
            "*spot*coord*.tsv*",
            "*Coords*.tsv*",
            "*coords*.tsv*",
            "spot_data*.tsv*",
        ),
        required=False,
    )
    resolved_metadata = _resolve_one(
        input_dir,
        metadata,
        role="meta_table",
        patterns=("*metadata*.tsv*", "*meta_table*.tsv*"),
        required=False,
    )
    resolved_transform = _resolve_one(
        input_dir,
        transform,
        role="transform",
        patterns=("*transformation_matrix*.txt", "*transform*.txt"),
        required=False,
    )
    resolved_image = _resolve_image(
        input_dir, image, require_image=require_image
    )
    assert resolved_counts is not None
    if (
        resolved_coordinates is None
        and resolved_metadata is None
        and resolved_transform is None
    ):
        raise FileNotFoundError(
            "legacy ST ingest requires spot coordinates, HE_X/HE_Y metadata, "
            "or a transform; ESB will not infer image alignment from array "
            "identifiers and image dimensions"
        )
    return LegacySTFiles(
        counts=resolved_counts,
        coordinates=resolved_coordinates,
        metadata=resolved_metadata,
        transform=resolved_transform,
        image=resolved_image,
    )


def _resolve_image(
    input_dir: Path, override: Path | None, *, require_image: bool
) -> Path | None:
    if override is not None:
        resolved = override if override.is_absolute() else input_dir / override
        if not resolved.is_file():
            raise FileNotFoundError(f"image file does not exist: {resolved}")
        return resolved
    candidates = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in _IMAGE_SUFFIXES
        and any(
            token in path.stem.casefold()
            for token in ("he", "h&e", "histology")
        )
        and "cy3" not in path.stem.casefold()
    )
    if len(candidates) > 1:
        raise ValueError(
            "ambiguous legacy ST histology images: "
            + ", ".join(path.name for path in candidates)
            + "; pass --image explicitly"
        )
    if not candidates:
        if require_image:
            raise FileNotFoundError(
                f"--require-image was set but no H&E/histology image was "
                f"found in {input_dir}"
            )
        logger.info("No optional legacy ST histology image found")
        return None
    logger.info("Discovered legacy ST histology image: %s", candidates[0])
    return candidates[0]


def _read_table(path: Path) -> pd.DataFrame:
    separator = "," if ".csv" in path.name.lower() else "\t"
    try:
        table = pd.read_csv(path, sep=separator, index_col=0)
    except Exception as exc:
        raise ValueError(
            f"could not read legacy ST table {path}: {exc}"
        ) from exc
    table.index = table.index.map(str)
    if table.index.has_duplicates:
        raise ValueError(f"legacy ST table has duplicate row names: {path}")
    if table.empty:
        raise ValueError(f"legacy ST table is empty: {path}")
    logger.info(
        "Read legacy ST table: %s rows=%d columns=%d",
        path,
        table.shape[0],
        table.shape[1],
    )
    return table


def _read_counts(path: Path) -> pd.DataFrame:
    counts = _read_table(path)
    numeric = cast(
        "pd.DataFrame", counts.apply(pd.to_numeric, errors="coerce")
    )
    if numeric.isna().to_numpy().any():
        bad = int(numeric.isna().sum().sum())
        raise ValueError(
            f"legacy ST count table contains {bad} non-numeric values"
        )
    values = numeric.to_numpy()
    if not np.all(np.isfinite(values)):
        raise ValueError("legacy ST count table contains non-finite values")
    if np.any(values < 0):
        raise ValueError("legacy ST count table contains negative values")
    if not np.allclose(values, np.rint(values)):
        raise ValueError("legacy ST count table contains fractional values")
    numeric.iloc[:, :] = np.rint(values)
    return numeric


def _format_array_value(value: object) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else str(number)


def _read_coordinates(path: Path) -> pd.DataFrame:
    """Read indexed coordinates or construct spot IDs from explicit x/y."""
    separator = "," if ".csv" in path.name.lower() else "\t"
    try:
        raw = pd.read_csv(path, sep=separator)
    except Exception as exc:
        raise ValueError(
            f"could not read legacy ST coordinate table {path}: {exc}"
        ) from exc
    columns = {str(column).casefold(): column for column in raw.columns}
    first_column = str(raw.columns[0])
    has_explicit_index = first_column.casefold() == "row.names" or (
        first_column.casefold().startswith("unnamed:")
    )
    if has_explicit_index:
        coordinates = raw.set_index(raw.columns[0])
        coordinates.index = coordinates.index.map(str)
        index_source = first_column
    elif "x" in columns and "y" in columns:
        x_values = pd.to_numeric(raw[columns["x"]], errors="coerce")
        y_values = pd.to_numeric(raw[columns["y"]], errors="coerce")
        if x_values.isna().any() or y_values.isna().any():
            raise ValueError(
                f"legacy ST coordinate table has invalid array x/y: {path}"
            )
        raw.index = pd.Index(
            [
                f"{_format_array_value(x)}x{_format_array_value(y)}"
                for x, y in zip(x_values, y_values, strict=True)
            ]
        )
        coordinates = raw
        index_source = "array x/y columns"
    else:
        index_column = raw.columns[0]
        coordinates = raw.set_index(index_column)
        coordinates.index = coordinates.index.map(str)
        index_source = str(index_column)
    if coordinates.index.has_duplicates:
        raise ValueError(
            f"legacy ST coordinate table has duplicate spot IDs: {path}"
        )
    if coordinates.empty:
        raise ValueError(f"legacy ST coordinate table is empty: {path}")
    logger.info(
        "Read legacy ST coordinate table: %s rows=%d columns=%d "
        "index_source=%s",
        path,
        coordinates.shape[0],
        coordinates.shape[1],
        index_source,
    )
    return coordinates


def _coordinate_columns(
    coordinates: pd.DataFrame,
    path: Path,
    requested_frame: str,
) -> tuple[str, str, str]:
    columns = {
        str(column).casefold(): str(column) for column in coordinates.columns
    }
    for x_name, y_name in (
        ("pixel_x", "pixel_y"),
        ("pxl_col_in_fullres", "pxl_row_in_fullres"),
    ):
        if x_name in columns and y_name in columns:
            if requested_frame not in {"auto", "fullres_pixels"}:
                raise ValueError(
                    f"coordinate table provides full-resolution pixels but "
                    f"--coordinate-frame={requested_frame} was requested"
                )
            return columns[x_name], columns[y_name], "fullres_pixels"
    if "x" in columns and "y" in columns:
        if requested_frame == "auto":
            if (
                path.name.casefold().startswith("spots")
                and ".csv" in path.name.casefold()
            ):
                return columns["x"], columns["y"], "fullres_pixels"
            return columns["x"], columns["y"], "array"
        return columns["x"], columns["y"], requested_frame
    if "xcoord" in columns and "ycoord" in columns:
        frame = "array" if requested_frame == "auto" else requested_frame
        return columns["xcoord"], columns["ycoord"], frame
    raise ValueError(
        f"legacy ST coordinate table {path} lacks a validated coordinate "
        "pair; "
        "expected pixel_x/pixel_y, pxl_col_in_fullres/"
        "pxl_row_in_fullres, X/Y, or xcoord/ycoord"
    )


def _read_transform(path: Path) -> np.ndarray:
    try:
        values = np.loadtxt(path, dtype=float)
    except Exception as exc:
        raise ValueError(
            f"could not read transform matrix {path}: {exc}"
        ) from exc
    if values.size != 9:
        raise ValueError(
            "legacy ST transform must contain exactly 9 numeric values: "
            f"{path}"
        )
    matrix = values.reshape(3, 3)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            f"legacy ST transform contains non-finite values: {path}"
        )
    logger.info("Read validated 3x3 row-vector transform: %s", path)
    return matrix


def _array_coordinates_from_names(names: pd.Index) -> np.ndarray:
    parsed: list[tuple[float, float]] = []
    invalid: list[str] = []
    for name in names.map(str):
        match = _ARRAY_SPOT_RE.fullmatch(name)
        if match is None:
            invalid.append(name)
            continue
        parsed.append((float(match.group("x")), float(match.group("y"))))
    if invalid:
        examples = ", ".join(invalid[:5])
        raise ValueError(
            "legacy ST pixel coordinates are incomplete and count "
            "observation names cannot all be parsed as array '<x>x<y>' "
            f"positions; invalid examples: {examples}"
        )
    return np.asarray(parsed, dtype=float)


def find_pixel_size_from_spot_coords(
    spots: pd.DataFrame,
    *,
    inter_spot_distance_um: float = DEFAULT_INTER_SPOT_DISTANCE_UM,
) -> tuple[float, int]:
    """Estimate microns per pixel using HEST's legacy grid algorithm.

    Args:
        spots: Spot table with array and full-resolution pixel coordinates.
        inter_spot_distance_um: Physical distance between adjacent grid spots.

    Returns:
        Estimated microns per pixel and the grid-column span used.

    Raises:
        ValueError: If no usable same-row spot pair can be found.

    Example:
        Two spots 200 pixels and one grid position apart at 200 µm spacing
        produce an estimate of 1 µm per pixel.
    """
    required = {
        "array_row",
        "array_col",
        "pxl_col_in_fullres",
        "pxl_row_in_fullres",
    }
    missing = required.difference(spots.columns)
    if missing:
        raise ValueError(
            "pixel-size estimation is missing columns: "
            + ", ".join(sorted(missing))
        )
    if inter_spot_distance_um <= 0:
        raise ValueError("inter_spot_distance_um must be greater than zero")

    frame = spots.sort_values("array_row")
    max_column_span = 0
    approximations = 0
    best_pixel_size = 0.0
    for _, row in frame.iterrows():
        same_row = frame[frame["array_row"] == row["array_row"]]
        if len(same_row) <= 1:
            continue
        array_columns = cast("pd.Series", same_row["array_col"])
        end_index = array_columns.idxmax()
        end = frame.loc[end_index]
        distance_pixels = float(
            np.hypot(
                row["pxl_col_in_fullres"] - end["pxl_col_in_fullres"],
                row["pxl_row_in_fullres"] - end["pxl_row_in_fullres"],
            )
        )
        column_span = int(abs(end["array_col"] - row["array_col"]))
        approximations += 1
        if column_span > max_column_span:
            if distance_pixels <= 0:
                raise ValueError(
                    "pixel-size estimation found distinct grid spots with "
                    "identical pixel coordinates"
                )
            max_column_span = column_span
            best_pixel_size = inter_spot_distance_um / (
                distance_pixels / column_span
            )
        if approximations > 3:
            break
    if approximations == 0 or max_column_span == 0 or best_pixel_size <= 0:
        raise ValueError(
            "could not estimate pixel size: no usable spots on the same "
            "legacy ST grid row"
        )
    logger.info(
        "Estimated legacy ST pixel size: %.6f um/pixel from a %d-column span",
        best_pixel_size,
        max_column_span,
    )
    return best_pixel_size, max_column_span


def _register_downscaled_image(
    adata: ad.AnnData,
    image_path: Path,
    *,
    pixel_size_um: float,
    spot_diameter_um: float,
    target_size: int = 1000,
) -> tuple[int, int]:
    with Image.open(image_path) as source:
        width, height = source.size
        downscale = target_size / max(width, height)
        size = (
            max(1, round(width * downscale)),
            max(1, round(height * downscale)),
        )
        thumbnail = source.convert("RGB").resize(size)
        downscaled = np.asarray(thumbnail)
    adata.uns["spatial"] = {
        "ST": {
            "images": {"downscaled_fullres": downscaled},
            "scalefactors": {
                "spot_diameter_fullres": spot_diameter_um / pixel_size_um,
                "tissue_downscaled_fullres_scalef": downscale,
            },
        }
    }
    logger.info(
        "Registered downscaled H&E in AnnData: fullres=%dx%d scale=%.8f",
        width,
        height,
        downscale,
    )
    return width, height


def read_legacy_st_sample(
    files: LegacySTFiles,
    *,
    coordinate_frame: str = "auto",
    spot_diameter_um: float = DEFAULT_SPOT_DIAMETER_UM,
    inter_spot_distance_um: float = DEFAULT_INTER_SPOT_DISTANCE_UM,
) -> tuple[ad.AnnData, str, dict[str, object]]:
    """Read resolved legacy ST tables into raw-count AnnData.

    Args:
        files: Resolved source files.
        coordinate_frame: Requested coordinate frame or ``auto``.
        spot_diameter_um: Physical diameter of one capture spot.
        inter_spot_distance_um: Physical grid center-to-center spacing.

    Returns:
        AnnData, resolved coordinate frame, and technology details.

    Raises:
        ValueError: If tables are invalid or observation identifiers differ.

    Example:
        Callers normally go through :func:`coral.st.pipeline.ingest_one`,
        which discovers the files and writes the returned AnnData as a store.
    """
    if coordinate_frame not in {"auto", "fullres_pixels", "array", "microns"}:
        raise ValueError(
            f"unsupported --coordinate-frame {coordinate_frame!r}"
        )
    counts = _read_counts(files.counts)
    if spot_diameter_um <= 0 or inter_spot_distance_um <= 0:
        raise ValueError(
            "legacy ST physical geometry must be greater than zero"
        )
    array_coordinates = _array_coordinates_from_names(counts.index)
    coordinates = (
        _read_coordinates(files.coordinates)
        if files.transform is None
        and files.metadata is None
        and files.coordinates is not None
        else pd.DataFrame(index=counts.index.copy())
    )
    original_n_obs = len(counts)
    metadata_table: pd.DataFrame | None = None
    transform_applied = False
    if files.transform is not None:
        homogeneous = np.column_stack(
            [array_coordinates, np.ones(array_coordinates.shape[0])]
        )
        transformed = homogeneous @ _read_transform(files.transform)
        if np.any(np.isclose(transformed[:, 2], 0.0)):
            raise ValueError(
                "legacy ST transform produced zero homogeneous scale"
            )
        spatial = transformed[:, :2] / transformed[:, 2, None]
        resolved_frame = "fullres_pixels"
        transform_applied = True
        coordinates = pd.DataFrame(index=counts.index.copy())
        x_column, y_column = "transform_x", "transform_y"
        alignment_method = "transform"
    elif files.metadata is not None:
        metadata_table = _read_table(files.metadata)
        required_meta = {"HE_X", "HE_Y"}
        if not required_meta.issubset(metadata_table.columns):
            raise ValueError(
                "legacy ST metadata alignment requires HE_X and HE_Y columns"
            )
        shared = counts.index.intersection(metadata_table.index, sort=False)
        if shared.empty:
            raise ValueError("legacy ST metadata and counts share no spot IDs")
        counts = counts.loc[shared]
        metadata_table = cast(
            "pd.DataFrame", metadata_table.loc[shared].copy()
        )
        spatial = metadata_table[["HE_X", "HE_Y"]].to_numpy(dtype=float)
        coordinates = pd.DataFrame(index=shared.copy())
        x_column, y_column = "HE_X", "HE_Y"
        resolved_frame = "fullres_pixels"
        alignment_method = "metadata_HE_X_HE_Y"
    elif files.coordinates is not None:
        shared = counts.index.intersection(coordinates.index, sort=False)
        if shared.empty:
            raise ValueError(
                "legacy ST spot-coordinate table and counts share no spot IDs"
            )
        dropped_counts = len(counts) - len(shared)
        ignored_coordinates = len(coordinates) - len(shared)
        logger.info(
            "HEST-style inner alignment: matched=%d dropped_counts=%d "
            "ignored_coordinates=%d",
            len(shared),
            dropped_counts,
            ignored_coordinates,
        )
        counts = counts.loc[shared]
        coordinates = coordinates.loc[shared].copy()
        x_column, y_column, resolved_frame = _coordinate_columns(
            coordinates, files.coordinates, coordinate_frame
        )
        xy = coordinates[[x_column, y_column]].apply(
            pd.to_numeric, errors="coerce"
        )
        if xy.isna().to_numpy().any():
            raise ValueError(
                "legacy ST coordinate table contains non-numeric values"
            )
        spatial = xy.to_numpy(dtype=float)
        alignment_method = "spot_coordinate_inner_join"
    selected_array = _array_coordinates_from_names(counts.index)
    obs = coordinates.drop(columns=[x_column, y_column], errors="ignore")
    if metadata_table is not None:
        obs = pd.concat(
            [obs, metadata_table.drop(columns=["HE_X", "HE_Y"])], axis=1
        )
    obs["array_row"] = selected_array[:, 0]
    obs["array_col"] = selected_array[:, 1]
    if resolved_frame == "fullres_pixels":
        obs["pxl_col_in_fullres"] = spatial[:, 0]
        obs["pxl_row_in_fullres"] = spatial[:, 1]
    elif resolved_frame != "array":
        raise ValueError(
            "legacy ST pixel-size estimation requires fullres_pixels or "
            "validated array coordinates"
        )

    values = counts.to_numpy()
    max_count = float(values.max())
    dtype = np.uint32 if max_count <= np.iinfo(np.uint32).max else np.uint64
    matrix = sparse.csr_matrix(values.astype(dtype, copy=False))
    adata = ad.AnnData(X=matrix, obs=obs)
    adata.var_names = [str(name) for name in counts.columns]
    adata.obsm["spatial"] = spatial
    pixel_size_um: float | None = None
    pixel_size_spot_span: int | None = None
    fullres_width: int | None = None
    fullres_height: int | None = None
    if resolved_frame == "fullres_pixels":
        pixel_size_um, pixel_size_spot_span = find_pixel_size_from_spot_coords(
            obs,
            inter_spot_distance_um=inter_spot_distance_um,
        )
        if files.image is not None:
            fullres_width, fullres_height = _register_downscaled_image(
                adata,
                files.image,
                pixel_size_um=pixel_size_um,
                spot_diameter_um=spot_diameter_um,
            )
    details: dict[str, object] = {
        "counts_file": str(files.counts.resolve()),
        "coordinates_file": (
            str(files.coordinates.resolve())
            if files.coordinates is not None
            else None
        ),
        "metadata_file": (
            str(files.metadata.resolve())
            if files.metadata is not None
            else None
        ),
        "transform_file": (
            str(files.transform.resolve())
            if files.transform is not None
            else None
        ),
        "transform_applied": transform_applied,
        "alignment_method": alignment_method,
        "coordinate_columns": [x_column, y_column],
        "original_count_observations": original_n_obs,
        "aligned_observations": len(counts),
        "dropped_count_observations": original_n_obs - len(counts),
        "spot_diameter_um": spot_diameter_um,
        "inter_spot_distance_um": inter_spot_distance_um,
        "spot_packing": LEGACY_SPOT_PACKING,
        "pixel_size_um_estimated": pixel_size_um,
        "pixel_size_spot_span": pixel_size_spot_span,
        "fullres_width": fullres_width,
        "fullres_height": fullres_height,
    }
    logger.info(
        "Assembled legacy ST AnnData: n_obs=%d n_vars=%d frame=%s sparse=%s",
        adata.n_obs,
        adata.n_vars,
        resolved_frame,
        sparse.issparse(adata.X),
    )
    return adata, resolved_frame, details
