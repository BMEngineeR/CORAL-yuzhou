"""HEST-derived YOLO fiducial autoalignment for 10x Visium images.

The regular spot grids and four fiducial centers are represented compactly
instead of shipping HEST's multi-megabyte JSON templates. The YOLO weight is a
user-supplied, checksum-verified asset because it is too large to bundle.
"""

from __future__ import annotations

import hashlib
import importlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from coral.assets import require_asset

MODEL_FILENAME = "visium_yolov8_v1.pt"
MODEL_SHA256 = (
    "6b4c2216a2416e9c439cd78de23a7430548968ccf84788fb5022a31a6f6e891e"
)
MODEL_SOURCE = "https://github.com/mahmoodlab/HEST/tree/main/models"

_CLASS_NAMES = {0: "hexFilled", 1: "hexOpen", 2: "hourglass", 3: "triangle"}
_ORIENTATION = {"hourglass": 0, "hexFilled": 1, "hexOpen": 2, "triangle": 3}


@dataclass(frozen=True)
class _Template:
    name: str
    rows: int
    spots_per_row: int
    x_step: float
    y_step: float
    x_origin: float
    y_origin: float
    fiducial_centers: dict[str, tuple[float, float]]

    def spots(self) -> tuple[np.ndarray, list[tuple[int, int]]]:
        array: list[tuple[int, int]] = []
        coordinates: list[tuple[float, float]] = []
        for row in range(self.rows):
            for offset in range(self.spots_per_row):
                col = 2 * offset + row % 2
                array.append((row, col))
                coordinates.append(
                    (
                        self.x_origin + self.x_step * col,
                        self.y_origin + self.y_step * row,
                    )
                )
        return np.asarray(coordinates, dtype=float), array


_TEMPLATE_65 = _Template(
    "6.5mm",
    78,
    64,
    87.0,
    50.0,
    4825.0,
    39073.0,
    {
        "hourglass": (4438.0, 38782.0),
        "hexFilled": (11563.0, 38782.0),
        "hexOpen": (11563.0, 46062.0),
        "triangle": (4438.0, 46105.333333333336),
    },
)
_TEMPLATE_11 = _Template(
    "11mm",
    128,
    112,
    86.5,
    50.0,
    4866.0,
    27930.0,
    {
        "hourglass": (4439.0, 27572.0),
        "hexFilled": (16443.5, 27572.0),
        "hexOpen": (16443.5, 39272.0),
        "triangle": (4439.0, 39315.333333333336),
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _affine(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    design = np.column_stack((source, np.ones(3)))
    return np.linalg.solve(design, destination).T


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.column_stack((points, np.ones(len(points)))) @ matrix.T


def _choose_template(
    boxes: list[dict[str, float | int]],
) -> tuple[_Template, float]:
    edges: list[float] = []
    for first, second in itertools.combinations(boxes, 2):
        a = _CLASS_NAMES[int(first["class"])]
        b = _CLASS_NAMES[int(second["class"])]
        if abs(_ORIENTATION[a] - _ORIENTATION[b]) == 1:
            edges.append(
                float(
                    np.hypot(
                        first["x"] - second["x"], first["y"] - second["y"]
                    )
                )
            )
    widths = [float(box["width"]) for box in boxes]
    if not edges or not widths or np.mean(widths) <= 0:
        raise ValueError(
            "Visium autoalignment could not measure fiducial geometry"
        )
    ratio = float(np.mean(edges) / np.mean(widths))
    return (_TEMPLATE_11 if ratio > 25 else _TEMPLATE_65), ratio


def _detect(
    image: np.ndarray, model_path: Path
) -> list[dict[str, float | int]]:
    try:
        ultralytics = importlib.import_module("ultralytics")
    except ImportError as exc:
        raise ImportError(
            "Visium YOLO autoalignment requires ultralytics; install the "
            "Visium autoalignment dependency before using --autoalign"
        ) from exc
    result = ultralytics.YOLO(model_path)(image)[0].boxes.cpu()
    boxes: list[dict[str, float | int]] = []
    for box in result:
        x, y, width, height = box.xywh[0].numpy().astype(float)
        boxes.append(
            {
                "class": int(box.cls),
                "x": float(x),
                "y": float(y),
                "width": float(width),
                "height": float(height),
            }
        )
    return boxes


def autoalign_visium(
    image_path: Path,
    output_dir: Path,
    *,
    boxes: list[dict[str, float | int]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    """Align a Visium template to at least three YOLO-detected fiducials."""
    with Image.open(image_path) as source_image:
        fullres = source_image.convert("RGB")
        width, height = fullres.size
        factor = 1000.0 / max(width, height)
        resized = fullres.resize(
            (max(1, round(width * factor)), max(1, round(height * factor)))
        )
    model_path: Path | None = None
    if boxes is None:
        model_path = require_asset(
            "Visium",
            MODEL_FILENAME,
            what="the HEST Visium YOLOv8 fiducial detector",
            source=MODEL_SOURCE,
        )
        actual_hash = _sha256(model_path)
        if actual_hash != MODEL_SHA256:
            raise ValueError(
                f"Visium detector checksum mismatch: expected {MODEL_SHA256}, "
                f"got {actual_hash}"
            )
        boxes = _detect(np.asarray(resized), model_path)
    if len(boxes) < 3:
        raise ValueError("Visium autoalignment requires at least 3 fiducials")
    template, ratio = _choose_template(boxes)
    source_points = np.asarray(
        [
            template.fiducial_centers[_CLASS_NAMES[int(box["class"])]]
            for box in boxes
        ]
    )
    destination_points = (
        np.asarray([[box["x"], box["y"]] for box in boxes]) / factor
    )
    matrix: np.ndarray | None = None
    aligned_fiducials: np.ndarray | None = None
    for indices in itertools.permutations(range(len(boxes)), 3):
        candidate = _affine(
            source_points[list(indices)], destination_points[list(indices)]
        )
        corners = _apply(
            candidate, np.asarray(list(template.fiducial_centers.values()))
        )
        if (
            np.all(corners[:, 0] >= 0)
            and np.all(corners[:, 0] < width)
            and np.all(corners[:, 1] >= 0)
            and np.all(corners[:, 1] < height)
        ):
            matrix = candidate
            aligned_fiducials = corners
            break
    if matrix is None or aligned_fiducials is None:
        raise ValueError(
            "Visium autoalignment could not place the fiducial template "
            "within the WSI"
        )
    template_spots, array_coordinates = template.spots()
    aligned_spots = _apply(matrix, template_spots)
    oligo = [
        {
            "tissue": True,
            "row": row,
            "col": col,
            "imageX": float(point[0]),
            "imageY": float(point[1]),
        }
        for (row, col), point in zip(
            array_coordinates, aligned_spots, strict=True
        )
    ]
    alignment = {"oligo": oligo}
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "autoalignment.json"
    json_path.write_text(json.dumps(alignment))
    overlay = resized.copy()
    draw = ImageDraw.Draw(overlay)
    for box in boxes:
        x, y = float(box["x"]), float(box["y"])
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline="blue", width=2)
    for point in aligned_spots[:: max(1, len(aligned_spots) // 1000)]:
        x, y = point * factor
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), outline="red")
    png_path = output_dir / "autoalignment.png"
    overlay.save(png_path)
    details: dict[str, Any] = {
        "alignment_source": "yolo_autoalign",
        "template": template.name,
        "template_spots": len(oligo),
        "fiducial_ratio": ratio,
        "detected_fiducials": boxes,
        "affine_matrix": matrix.tolist(),
        "model": MODEL_FILENAME,
        "model_sha256": MODEL_SHA256,
        "model_path": str(model_path.resolve()) if model_path else None,
        "validation": "within_wsi",
    }
    return (
        alignment,
        details,
        {
            "autoalignment.json": json_path,
            "autoalignment.png": png_path,
        },
    )
