"""Reader for 10x Visium Space Ranger layouts and H&E alignment."""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import tifffile
from matplotlib.axes import Axes
from PIL import Image
from scipy import sparse
from scipy.io import mmread

from coral.assets import require_asset
from coral.st.visium_autoalign import autoalign_visium

logger = logging.getLogger(__name__)

VISIUM_SPOT_DIAMETER_UM = 55.0
VISIUM_INTER_SPOT_DISTANCE_UM = 100.0
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".btf"}


@dataclass(frozen=True)
class VisiumFiles:
    """Resolved input files for one 10x Visium sample."""

    counts: Path
    spatial_dir: Path | None
    positions: Path | None
    scalefactors: Path | None
    metrics: Path | None
    alignment: Path | None
    image: Path | None

    def source_files(self) -> list[Path]:
        """Return every source used by the reader in stable role order."""
        return [
            self.counts,
            *([self.positions] if self.positions else []),
            *([self.scalefactors] if self.scalefactors else []),
            *([self.metrics] if self.metrics else []),
            *([self.alignment] if self.alignment else []),
            *([self.image] if self.image else []),
        ]


def _resolve(input_dir: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else input_dir / path


def _first(input_dir: Path, patterns: tuple[str, ...]) -> Path | None:
    for pattern in patterns:
        candidates = sorted(
            path for path in input_dir.rglob(pattern) if path.is_file()
        )
        if candidates:
            return candidates[0]
    return None


def _discover_counts(input_dir: Path, override: Path | None) -> Path:
    resolved = _resolve(input_dir, override)
    if resolved is not None:
        if not resolved.exists():
            raise FileNotFoundError(f"Visium counts do not exist: {resolved}")
        return resolved
    for name in (
        "*filtered_feature_bc_matrix.h5",
        "*raw_feature_bc_matrix.h5",
    ):
        found = _first(input_dir, (name,))
        if found:
            return found
    matrix = _first(input_dir, ("matrix.mtx.gz", "matrix.mtx"))
    if matrix is not None:
        return matrix.parent
    raise FileNotFoundError(
        "no Visium expression matrix found; searched filtered/raw "
        "feature-barcode "
        "H5 files and MEX matrix.mtx(.gz)"
    )


def _discover_image(input_dir: Path, override: Path | None) -> Path | None:
    resolved = _resolve(input_dir, override)
    if resolved is not None:
        if not resolved.is_file():
            raise FileNotFoundError(f"Visium image does not exist: {resolved}")
        return resolved
    candidates = [
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    ]
    preferred = [
        path
        for path in candidates
        if "tissue_hires" not in path.name.lower()
        and "tissue_lowres" not in path.name.lower()
    ]
    pool = preferred or candidates
    return max(pool, key=lambda path: path.stat().st_size) if pool else None


def discover_visium_files(
    input_dir: Path,
    *,
    counts: Path | None = None,
    spatial: Path | None = None,
    alignment: Path | None = None,
    metrics: Path | None = None,
    image: Path | None = None,
    require_image: bool = False,
) -> VisiumFiles:
    """Discover one Space Ranger Visium sample without mutating it."""
    if not input_dir.is_dir():
        raise ValueError(
            f"input_dir must be an existing directory: {input_dir}"
        )
    spatial_path = _resolve(input_dir, spatial)
    if spatial_path is None:
        candidate = input_dir / "spatial"
        spatial_path = candidate if candidate.is_dir() else None
    elif spatial_path.is_file():
        spatial_path = spatial_path.parent
    if spatial_path is not None and not spatial_path.is_dir():
        raise FileNotFoundError(
            f"Visium spatial directory does not exist: {spatial_path}"
        )
    positions = (
        _first(
            spatial_path, ("tissue_positions.csv", "tissue_positions_list.csv")
        )
        if spatial_path
        else _first(
            input_dir, ("tissue_positions.csv", "tissue_positions_list.csv")
        )
    )
    scalefactors = (
        _first(spatial_path, ("scalefactors_json.json",))
        if spatial_path
        else _first(input_dir, ("scalefactors_json.json",))
    )
    alignment_path = _resolve(input_dir, alignment) or _first(
        input_dir,
        ("alignment_file.json", "alignment.json", "autoalignment.json"),
    )
    metrics_path = _resolve(input_dir, metrics) or _first(
        input_dir, ("*metrics_summary.csv",)
    )
    image_path = _discover_image(input_dir, image)
    if require_image and image_path is None:
        raise FileNotFoundError(
            "Visium ingest requires a full-resolution H&E image"
        )
    return VisiumFiles(
        counts=_discover_counts(input_dir, counts),
        spatial_dir=spatial_path,
        positions=positions,
        scalefactors=scalefactors,
        metrics=metrics_path,
        alignment=alignment_path,
        image=image_path,
    )


def _unique_feature_names(names: list[str]) -> list[str]:
    """Return stable unique names while retaining the first spelling."""
    used: set[str] = set()
    counters: dict[str, int] = {}
    result: list[str] = []
    for name in names:
        candidate = name
        if candidate in used:
            suffix = counters.get(name, 1)
            candidate = f"{name}-{suffix}"
            while candidate in used:
                suffix += 1
                candidate = f"{name}-{suffix}"
            counters[name] = suffix + 1
        used.add(candidate)
        result.append(candidate)
    return result


def _finalize_feature_names(adata: ad.AnnData) -> None:
    """Preserve symbols while making AnnData feature names unique."""
    gene_symbols = [str(name) for name in adata.var_names]
    if "gene_symbols" not in adata.var:
        adata.var["gene_symbols"] = gene_symbols
    adata.var_names = _unique_feature_names(gene_symbols)


def _read_counts(path: Path) -> ad.AnnData:
    if path.is_dir():
        matrix_path = next(
            (
                candidate
                for name in ("matrix.mtx.gz", "matrix.mtx")
                if (candidate := path / name).is_file()
            ),
            None,
        )
        feature_path = next(
            (
                candidate
                for name in (
                    "features.tsv.gz",
                    "features.tsv",
                    "genes.tsv.gz",
                    "genes.tsv",
                )
                if (candidate := path / name).is_file()
            ),
            None,
        )
        barcode_path = next(
            (
                candidate
                for name in ("barcodes.tsv.gz", "barcodes.tsv")
                if (candidate := path / name).is_file()
            ),
            None,
        )
        if matrix_path is None or feature_path is None or barcode_path is None:
            raise FileNotFoundError(
                f"Visium MEX directory is incomplete: {path}; expected "
                "matrix, "
                "features/genes, and barcodes"
            )
        matrix = sparse.csr_matrix(mmread(matrix_path)).T
        features = pd.read_csv(feature_path, sep="\t", header=None)
        barcodes = pd.read_csv(barcode_path, sep="\t", header=None)
        gene_symbols = features.iloc[
            :, 1 if features.shape[1] > 1 else 0
        ].astype(str)
        var = pd.DataFrame(
            index=pd.Index(
                _unique_feature_names(gene_symbols.tolist()), name=None
            )
        )
        var["gene_ids"] = features.iloc[:, 0].astype(str).to_numpy()
        var["gene_symbols"] = gene_symbols.to_numpy()
        if features.shape[1] > 2:
            var["feature_types"] = features.iloc[:, 2].astype(str).to_numpy()
        result = ad.AnnData(
            matrix,
            obs=pd.DataFrame(
                index=pd.Index(
                    barcodes.iloc[:, 0].astype(str).to_numpy(), name=None
                )
            ),
            var=var,
        )
    elif path.suffix.lower() in {".h5", ".hdf5"}:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Variable names are not unique.*",
                category=UserWarning,
                module="anndata._core.anndata",
            )
            result = sc.read_10x_h5(path)
    else:
        raise ValueError(f"unsupported Visium count source: {path}")
    _finalize_feature_names(result)
    result.obs_names = [str(value) for value in result.obs_names]
    return result


def _read_positions(path: Path) -> pd.DataFrame:
    if path.name == "tissue_positions_list.csv":
        frame = pd.read_csv(path, header=None, index_col=0)
        frame.columns = [
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_row_in_fullres",
            "pxl_col_in_fullres",
        ]
    else:
        frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.map(str)
    required = {
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Visium tissue positions missing columns: {sorted(missing)}"
        )
    return frame


def _alignment_frame(alignment: dict[str, Any]) -> pd.DataFrame:
    if "oligo" not in alignment:
        raise ValueError("Visium alignment JSON is missing 'oligo'")
    frame = pd.DataFrame(alignment["oligo"]).rename(
        columns={
            "tissue": "in_tissue",
            "row": "array_row",
            "col": "array_col",
            "imageX": "pxl_col_in_fullres",
            "imageY": "pxl_row_in_fullres",
        }
    )
    if "cytAssistInfo" in alignment:
        transform = np.asarray(
            alignment["cytAssistInfo"]["transformImages"], dtype=float
        )
        coordinates = np.column_stack(
            (
                frame["pxl_col_in_fullres"],
                frame["pxl_row_in_fullres"],
                np.ones(len(frame)),
            )
        )
        transformed = (np.linalg.inv(transform) @ coordinates.T).T
        frame["pxl_col_in_fullres"] = transformed[:, 0]
        frame["pxl_row_in_fullres"] = transformed[:, 1]
    frame["in_tissue"] = True
    return frame


def _load_barcode_coordinates() -> list[pd.DataFrame]:
    directory = require_asset(
        "Visium",
        "barcode_coords",
        what="HEST Visium barcode-coordinate tables",
        source="https://github.com/mahmoodlab/HEST/tree/main/assets/barcode_coords",
    )
    tables: list[pd.DataFrame] = []
    for path in sorted(directory.glob("*.txt")):
        table = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["barcode", "array_col", "array_row"],
        )
        table["barcode"] = table["barcode"].astype(str) + "-1"
        table["array_col"] -= 1
        table["array_row"] -= 1
        tables.append(table)
    if not tables:
        raise FileNotFoundError(
            f"no Visium barcode-coordinate tables found in {directory}"
        )
    return tables


def _map_alignment_to_barcodes(
    alignment: pd.DataFrame,
    obs_names: pd.Index,
    *,
    positions: pd.DataFrame | None,
) -> pd.DataFrame:
    if positions is not None:
        base = positions.reset_index(names="barcode")
        merged = base.drop(
            columns=["pxl_col_in_fullres", "pxl_row_in_fullres"],
            errors="ignore",
        ).merge(
            alignment[
                [
                    "array_row",
                    "array_col",
                    "pxl_col_in_fullres",
                    "pxl_row_in_fullres",
                ]
            ],
            on=["array_row", "array_col"],
            how="inner",
        )
        merged = merged.set_index("barcode")
        return merged.reindex(obs_names)
    best: pd.DataFrame | None = None
    best_matches = -1
    for barcode_table in _load_barcode_coordinates():
        candidate = alignment.merge(
            barcode_table, on=["array_row", "array_col"], how="inner"
        )
        candidate = candidate.set_index("barcode")
        matches = int(candidate.index.isin(obs_names).sum())
        if matches > best_matches:
            best, best_matches = candidate, matches
    if best is None or best_matches == 0:
        raise ValueError(
            "no Visium barcode template matches the expression matrix"
        )
    return best.reindex(obs_names)


def _find_visium_pixel_size(spatial: pd.DataFrame) -> tuple[float, int]:
    """Estimate µm/pixel from same-row spots on the staggered Visium grid."""
    best_span = 0
    best_size = 0.0
    for _, same_row in spatial.groupby("array_row"):
        if len(same_row) < 2:
            continue
        ordered = same_row.sort_values("array_col")
        first, last = ordered.iloc[0], ordered.iloc[-1]
        column_span = int(last["array_col"] - first["array_col"])
        if column_span <= best_span:
            continue
        pixel_distance = float(
            np.hypot(
                last["pxl_col_in_fullres"] - first["pxl_col_in_fullres"],
                last["pxl_row_in_fullres"] - first["pxl_row_in_fullres"],
            )
        )
        physical_steps = column_span / 2
        if pixel_distance > 0 and physical_steps > 0:
            best_span = column_span
            best_size = (
                VISIUM_INTER_SPOT_DISTANCE_UM * physical_steps / pixel_distance
            )
    if best_size <= 0:
        raise ValueError(
            "could not estimate Visium pixel size from spot coordinates"
        )
    return best_size, best_span


def _json_scalar(value: object) -> object:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return value.item() if isinstance(value, np.generic) else value


def _embedded_pixel_size_um(image_path: Path) -> float | None:
    if image_path.suffix.casefold() not in {".tif", ".tiff", ".btf"}:
        return None
    with tifffile.TiffFile(image_path) as tif:
        metadata = tif.ome_metadata
        if metadata:
            from xml.etree import ElementTree

            root = ElementTree.fromstring(metadata)
            pixels = next(
                (
                    element
                    for element in root.iter()
                    if element.tag.endswith("Pixels")
                ),
                None,
            )
            if pixels is not None and pixels.get("PhysicalSizeX"):
                value = float(pixels.attrib["PhysicalSizeX"])
                unit = pixels.get("PhysicalSizeXUnit", "µm").casefold()
                factors = {"nm": 0.001, "mm": 1000.0, "cm": 10_000.0}
                return value * factors.get(unit, 1.0)
        page: Any = tif.pages[0]
        resolution = page.tags.get("XResolution")
        unit = page.tags.get("ResolutionUnit")
        if resolution is None or unit is None:
            return None
        numerator, denominator = resolution.value
        pixels_per_unit = float(numerator) / float(denominator)
        if pixels_per_unit <= 0:
            return None
        if int(unit.value) == 2:
            return 25_400.0 / pixels_per_unit
        if int(unit.value) == 3:
            return 10_000.0 / pixels_per_unit
    return None


def _plot_center_square(
    axis: Axes,
    *,
    width: int,
    height: int,
    length: float,
    color: str,
    text: str,
    offset: int = 0,
) -> None:
    if length > width * 4 and length > height * 4:
        return
    margin_x = (width - length) / 2
    margin_y = (height - length) / 2
    axis.text(
        width // 2,
        (height // 2) + offset,
        text,
        fontsize=12,
        ha="center",
        va="center",
        color=color,
    )
    axis.plot([margin_x, length + margin_x], [margin_y, margin_y], color=color)
    axis.plot(
        [margin_x, length + margin_x],
        [height - margin_y, height - margin_y],
        color=color,
    )
    axis.plot([margin_x, margin_x], [margin_y, height - margin_y], color=color)
    axis.plot(
        [length + margin_x, length + margin_x],
        [margin_y, height - margin_y],
        color=color,
    )


def _pixel_size_visualization(
    output: Path,
    *,
    downscaled_image: np.ndarray,
    downscale_factor: float,
    estimated: float | None,
    embedded: float | None,
) -> None:
    figure, axis = plt.subplots()
    axis.imshow(downscaled_image)
    height, width = downscaled_image.shape[:2]
    if embedded is not None:
        _plot_center_square(
            axis,
            width=width,
            height=height,
            length=(6500.0 / embedded) * downscale_factor,
            color="red",
            text="6.5m, embedded",
            offset=50,
        )
    if estimated is not None:
        _plot_center_square(
            axis,
            width=width,
            height=height,
            length=(6500.0 / estimated) * downscale_factor,
            color="blue",
            text="6.5m, estimated",
        )
    figure.savefig(output)
    plt.close(figure)


def read_visium_sample(
    files: VisiumFiles,
    *,
    autoalign: str = "auto",
    artifact_dir: Path | None = None,
) -> tuple[ad.AnnData, dict[str, Any], dict[str, Path]]:
    """Read counts and resolve full-resolution H&E spot coordinates."""
    if autoalign not in {"never", "auto", "always"}:
        raise ValueError("autoalign must be one of: never, auto, always")
    adata = _read_counts(files.counts)
    positions = _read_positions(files.positions) if files.positions else None
    alignment_data: dict[str, Any] | None = None
    alignment_details: dict[str, Any] = {}
    artifacts: dict[str, Path] = {}
    should_autoalign = autoalign == "always" or (
        autoalign == "auto" and positions is None and files.alignment is None
    )
    if should_autoalign:
        if files.image is None:
            raise ValueError(
                "Visium YOLO autoalignment requires a full-resolution image"
            )
        if artifact_dir is None:
            raise ValueError(
                "Visium YOLO autoalignment requires an artifact directory"
            )
        alignment_data, alignment_details, artifacts = autoalign_visium(
            files.image, artifact_dir
        )
    elif files.alignment is not None:
        alignment_data = json.loads(files.alignment.read_text())
        alignment_details = {"alignment_source": "alignment_json"}

    if alignment_data is not None:
        spatial = _map_alignment_to_barcodes(
            _alignment_frame(alignment_data),
            adata.obs_names,
            positions=positions,
        )
    elif positions is not None:
        spatial = positions.reindex(adata.obs_names)
        alignment_details = {"alignment_source": "space_ranger"}
    else:
        raise ValueError(
            "Visium full-resolution alignment is unavailable; provide spatial "
            "positions or --alignment, or enable YOLO autoalignment"
        )
    required = [
        "array_row",
        "array_col",
        "pxl_col_in_fullres",
        "pxl_row_in_fullres",
    ]
    if spatial[required].isna().to_numpy().any():
        raise ValueError(
            "Visium coordinates do not cover every expression barcode"
        )
    adata.obs = spatial
    adata.obsm["spatial"] = spatial[
        ["pxl_col_in_fullres", "pxl_row_in_fullres"]
    ].to_numpy(dtype=float)
    pixel_size, span = _find_visium_pixel_size(spatial)
    if files.image is not None:
        with Image.open(files.image) as source:
            thumbnail = source.convert("RGB")
            thumbnail.thumbnail((1000, 1000))
        adata.uns["spatial"] = {
            "Visium": {
                "images": {"downscaled_fullres": np.asarray(thumbnail)},
                "scalefactors": {
                    "spot_diameter_fullres": VISIUM_SPOT_DIAMETER_UM
                    / pixel_size,
                    "tissue_downscaled_fullres_scalef": thumbnail.width
                    / source.width,
                },
            }
        }
    details: dict[str, Any] = {
        **alignment_details,
        "counts_file": str(files.counts.resolve()),
        "positions_file": str(files.positions.resolve())
        if files.positions
        else None,
        "alignment_file": str(files.alignment.resolve())
        if files.alignment
        else None,
        "pixel_size_um_estimated": pixel_size,
        "pixel_size_spot_span": span,
        "spot_diameter_um": VISIUM_SPOT_DIAMETER_UM,
        "inter_spot_distance_um": VISIUM_INTER_SPOT_DISTANCE_UM,
    }
    embedded_pixel_size = (
        _embedded_pixel_size_um(files.image)
        if files.image is not None
        else None
    )
    if files.scalefactors is not None:
        scalefactors = json.loads(files.scalefactors.read_text())
        details["scalefactors"] = scalefactors
    details["pixel_size_um_embedded"] = embedded_pixel_size
    if files.metrics is not None:
        metrics_frame = pd.read_csv(files.metrics)
        details["metrics"] = (
            {
                str(key): _json_scalar(value)
                for key, value in metrics_frame.iloc[0].items()
            }
            if len(metrics_frame)
            else {}
        )
    return adata, details, artifacts
