"""Reader for native 10x Visium HD Space Ranger outputs.

HEST's reader organization informs discovery, but ESB deliberately reads the
native cell or binned matrices instead of pooling bins into pseudo-spots.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from PIL import Image

from coral.st.visium import _read_counts

_BIN_DIRECTORIES = {
    "bin_2": "square_002um",
    "bin_8": "square_008um",
    "bin_16": "square_016um",
}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".btf"}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VisiumHDFiles:
    """Files selected for one native Visium HD observation unit."""

    counts: Path
    coordinates: Path
    scalefactors: Path | None
    image: Path | None
    image_source: str | None
    metrics: Path | None
    obs_unit: str
    binned_outputs: Path | None
    segmented_outputs: Path | None

    def source_files(self) -> list[Path]:
        """Return sources actually used by ingest in stable order."""
        return [
            self.counts,
            self.coordinates,
            *([self.scalefactors] if self.scalefactors else []),
            *([self.metrics] if self.metrics else []),
            *([self.image] if self.image else []),
        ]


def _resolve(input_dir: Path, value: Path | None) -> Path | None:
    if value is None:
        return None
    return value if value.is_absolute() else input_dir / value


def _matrix(directory: Path, stem: str) -> Path | None:
    h5 = directory / f"{stem}.h5"
    if h5.is_file():
        return h5
    mex = directory / stem
    return mex if mex.is_dir() else None


def _discover_image(
    input_dir: Path, override: Path | None
) -> tuple[Path | None, str | None]:
    """Select the image most likely to share Space Ranger pixel coordinates.

    A root-level ``*_tissue_image`` is normally the original full-resolution
    microscope image. If it is absent, Space Ranger's H&E-derived tissue-hires
    image is safer than a generically named root-level image, which may be a
    CytAssist image in a different coordinate frame. ``aligned_tissue_image``
    is deliberately excluded because it is not the H&E image.
    """
    resolved = _resolve(input_dir, override)
    if resolved is not None:
        if not resolved.is_file():
            raise FileNotFoundError(
                f"Visium HD image does not exist: {resolved}"
            )
        return resolved, "explicit"

    root_images = [
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in _IMAGE_SUFFIXES
    ]
    root_tissue = sorted(
        path
        for path in root_images
        if "tissue_image" in path.name.casefold()
    )
    if root_tissue:
        return root_tissue[0], "root_tissue_image"

    spatial = input_dir / "spatial"
    for name, source in (
        ("tissue_hires_image.png", "space_ranger_tissue_hires_image"),
    ):
        candidate = spatial / name
        if candidate.is_file():
            return candidate, source

    generic = sorted(
        root_images,
        key=lambda path: ("image" not in path.name.casefold(), path.name),
    )
    return (generic[0], "root_generic_image") if generic else (None, None)


def _validate_image_coordinate_coverage(
    spatial: np.ndarray, image: Path
) -> dict[str, float | int]:
    """Reject a likely image-coordinate-frame mismatch before writing output."""
    with Image.open(image) as source:
        width, height = source.size
    margin_x, margin_y = width * 0.15, height * 0.15
    within = (
        (spatial[:, 0] >= -margin_x)
        & (spatial[:, 0] <= width + margin_x)
        & (spatial[:, 1] >= -margin_y)
        & (spatial[:, 1] <= height + margin_y)
    )
    coverage = float(np.mean(within)) if len(within) else 1.0
    if coverage < 0.98:
        raise ValueError(
            "Visium HD image is incompatible with Space Ranger "
            f"full-resolution coordinates: {image} is {width}x{height}, "
            f"but only {coverage:.1%} of coordinates fall within a 15% "
            "image-boundary tolerance. Supply the matching tissue image with "
            "--image."
        )
    return {
        "image_width_px": width,
        "image_height_px": height,
        "coordinate_coverage": coverage,
        "coordinate_boundary_tolerance": 0.15,
    }


def discover_visium_hd_files(
    input_dir: Path,
    *,
    obs_unit: str = "all",
    counts: Path | None = None,
    binned_outputs: Path | None = None,
    segmented_outputs: Path | None = None,
    image: Path | None = None,
    metrics: Path | None = None,
    require_image: bool = False,
) -> list[VisiumHDFiles]:
    """Discover every selected native Space Ranger cell or bin output."""
    if not input_dir.is_dir():
        raise ValueError(
            f"input_dir must be an existing directory: {input_dir}"
        )
    if obs_unit not in {"all", "auto", "cell", *_BIN_DIRECTORIES}:
        raise ValueError(
            "Visium HD --obs-unit must be all, cell, bin_2, bin_8, or bin_16"
        )
    binned = (
        _resolve(input_dir, binned_outputs) or input_dir / "binned_outputs"
    )
    segmented = (
        _resolve(input_dir, segmented_outputs)
        or input_dir / "segmented_outputs"
    )
    override = _resolve(input_dir, counts)
    if override is not None and obs_unit in {"all", "auto"}:
        raise ValueError("--counts requires one explicit Visium HD --obs-unit")
    image_path, image_source = _discover_image(input_dir, image)
    if require_image and image_path is None:
        raise FileNotFoundError("Visium HD ingest requires a tissue image")
    metrics_path = _resolve(input_dir, metrics)
    if metrics_path is None:
        metrics_path = next(
            iter(sorted(input_dir.glob("*metrics_summary.csv"))), None
        )
    elif not metrics_path.is_file():
        raise FileNotFoundError(
            f"Visium HD metrics do not exist: {metrics_path}"
        )
    available: list[str] = []
    if _matrix(segmented, "filtered_feature_cell_matrix") is not None:
        available.append("cell")
    available.extend(
        unit
        for unit, directory in _BIN_DIRECTORIES.items()
        if _matrix(
            binned / directory, "filtered_feature_bc_matrix"
        ) is not None
    )
    selected = available if obs_unit in {"all", "auto"} else [obs_unit]
    if not selected:
        raise FileNotFoundError(
            "no native Visium HD filtered outputs were found"
        )
    files: list[VisiumHDFiles] = []
    for unit in selected:
        if unit == "cell":
            matrix = override or _matrix(
                segmented, "filtered_feature_cell_matrix"
            )
            coordinates = segmented / "cell_segmentations.geojson"
            scalefactors = segmented / "spatial" / "scalefactors_json.json"
        else:
            native = binned / _BIN_DIRECTORIES[unit]
            matrix = override or _matrix(native, "filtered_feature_bc_matrix")
            coordinates = native / "spatial" / "tissue_positions.parquet"
            scalefactors = native / "spatial" / "scalefactors_json.json"
        if matrix is None or not matrix.exists():
            raise FileNotFoundError(
                f"Visium HD {unit} filtered matrix was not found"
            )
        if not coordinates.is_file():
            raise FileNotFoundError(
                f"Visium HD {unit} coordinates were not found: {coordinates}"
            )
        files.append(
            VisiumHDFiles(
                counts=matrix,
                coordinates=coordinates,
                scalefactors=scalefactors if scalefactors.is_file() else None,
                image=image_path,
                image_source=image_source,
                metrics=metrics_path,
                obs_unit=unit,
                binned_outputs=binned if binned.is_dir() else None,
                segmented_outputs=segmented if segmented.is_dir() else None,
            )
        )
    return files


def _polygon_centroid(geometry: Any) -> tuple[float, float] | None:
    """Return an area-weighted centroid for a GeoJSON Polygon or MultiPolygon."""
    if not isinstance(geometry, dict):
        return None
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "Polygon":
        polygons = [coordinates]
    elif geometry_type == "MultiPolygon":
        polygons = coordinates
    else:
        return None
    if not isinstance(polygons, list):
        return None

    weighted_x = weighted_y = total_area = 0.0
    for polygon in polygons:
        if not isinstance(polygon, list) or not polygon:
            continue
        ring = polygon[0]
        if not isinstance(ring, list) or len(ring) < 3:
            continue
        points = ring[:-1] if ring[0] == ring[-1] else ring
        if len(points) < 3 or any(
            not isinstance(point, list) or len(point) < 2 for point in points
        ):
            continue
        area_twice = cross_x = cross_y = 0.0
        for index, point in enumerate(points):
            next_point = points[(index + 1) % len(points)]
            x0, y0 = float(point[0]), float(point[1])
            x1, y1 = float(next_point[0]), float(next_point[1])
            cross = x0 * y1 - x1 * y0
            area_twice += cross
            cross_x += (x0 + x1) * cross
            cross_y += (y0 + y1) * cross
        if not np.isfinite([area_twice, cross_x, cross_y]).all():
            continue
        if np.isclose(area_twice, 0.0):
            centroid_x = float(np.mean([float(point[0]) for point in points]))
            centroid_y = float(np.mean([float(point[1]) for point in points]))
            area = 1.0
        else:
            centroid_x = cross_x / (3.0 * area_twice)
            centroid_y = cross_y / (3.0 * area_twice)
            area = abs(area_twice) / 2.0
        weighted_x += centroid_x * area
        weighted_y += centroid_y * area
        total_area += area
    if total_area == 0.0:
        return None
    return weighted_x / total_area, weighted_y / total_area


def _cell_coordinates(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text())
    records: list[dict[str, Any]] = []
    for feature in payload.get("features", []):
        properties = feature.get("properties", {})
        cell_id = properties.get("cell_id")
        centroid = properties.get("cell_centroid")
        if not isinstance(centroid, list) or len(centroid) != 2:
            centroid = _polygon_centroid(feature.get("geometry"))
        if cell_id is None or centroid is None or len(centroid) != 2:
            continue
        records.append(
            {
                "barcode": f"cellid_{int(cell_id):09d}-1",
                "pxl_col_in_fullres": float(centroid[0]),
                "pxl_row_in_fullres": float(centroid[1]),
            }
        )
    if not records:
        raise ValueError("Visium HD cell segmentation has no usable centroids")
    return pd.DataFrame.from_records(records).set_index("barcode")


def _bin_coordinates(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {
        "barcode",
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Visium HD positions missing columns: {sorted(missing)}"
        )
    return frame.set_index("barcode")


def read_visium_hd_sample(
    files: VisiumHDFiles,
) -> tuple[ad.AnnData, dict[str, Any]]:
    """Read a selected native Visium HD matrix and its pixel coordinates."""
    adata = _read_counts(files.counts)
    coordinates = (
        _cell_coordinates(files.coordinates)
        if files.obs_unit == "cell"
        else _bin_coordinates(files.coordinates)
    )
    coordinates.index = coordinates.index.map(str)
    spatial = coordinates.reindex(adata.obs_names)
    pixel_columns = ["pxl_col_in_fullres", "pxl_row_in_fullres"]
    if spatial[pixel_columns].isna().to_numpy().any():
        missing = int(
            np.count_nonzero(
                spatial[pixel_columns].isna().to_numpy().any(axis=1)
            )
        )
        raise ValueError(
            f"Visium HD coordinates do not cover {missing} expression barcodes"
        )
    adata.obs = spatial
    adata.obsm["spatial"] = spatial[pixel_columns].to_numpy(dtype=float)
    scalefactors: dict[str, Any] = {}
    if files.scalefactors is not None:
        scalefactors = json.loads(files.scalefactors.read_text())
    if files.image is not None:
        with Image.open(files.image) as source:
            full_width = source.width
            thumbnail = source.convert("RGB")
            thumbnail.thumbnail((1000, 1000))
        adata.uns["spatial"] = {
            "VisiumHD": {
                "images": {"downscaled_fullres": np.asarray(thumbnail)},
                "scalefactors": {
                    **scalefactors,
                    "tissue_downscaled_fullres_scalef": thumbnail.width
                    / full_width,
                },
            }
        }
    details: dict[str, Any] = {
        "selected_obs_unit": files.obs_unit,
        "selected_resolution": (
            "cells"
            if files.obs_unit == "cell"
            else f"{files.obs_unit.removeprefix('bin_')}um"
        ),
        "discovered_resolutions": [
            f"{unit.removeprefix('bin_')}um"
            for unit, directory in _BIN_DIRECTORIES.items()
            if files.binned_outputs is not None
            and _matrix(
                files.binned_outputs / directory, "filtered_feature_bc_matrix"
            )
            is not None
        ],
        "cell_output_available": (
            files.segmented_outputs is not None
            and _matrix(
                files.segmented_outputs, "filtered_feature_cell_matrix"
            )
            is not None
        ),
        "counts_file": str(files.counts.resolve()),
        "coordinates_file": str(files.coordinates.resolve()),
        "scalefactors_file": (
            str(files.scalefactors.resolve()) if files.scalefactors else None
        ),
        "native_output": True,
        "pooled": False,
        "scalefactors": scalefactors,
        "image_file": str(files.image.resolve()) if files.image else None,
        "image_source": files.image_source,
    }
    if files.image is not None:
        details["image_coordinate_validation"] = (
            _validate_image_coordinate_coverage(
                adata.obsm["spatial"], files.image
            )
        )
    if files.metrics is not None:
        metrics_frame = pd.read_csv(files.metrics)
        details["metrics"] = (
            {
                str(key): value.item()
                if isinstance(value, np.generic)
                else value
                for key, value in metrics_frame.iloc[0].items()
            }
            if len(metrics_frame)
            else {}
        )
    return adata, details
