"""Reader for cell-level 10x Xenium outputs with optional H&E alignment."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import anndata as ad
import numpy as np
import pandas as pd
import tifffile
from scipy import sparse

from coral.st.visium import _read_counts

logger = logging.getLogger(__name__)

CentroidAligner = Callable[
    [np.ndarray, Path, Path, Path, int, bool],
    tuple[np.ndarray, dict[str, Any]],
]

MIN_XENIUM_IN_BOUNDS_RATE = 0.95


@dataclass(frozen=True)
class XeniumFiles:
    """Resolved files for one Xenium cell-level ingest."""

    experiment: Path
    feature_matrix: Path
    cells: Path
    dapi: Path
    image: Path | None
    alignment: Path | None
    transcripts: Path | None
    cell_boundaries: Path | None
    nucleus_boundaries: Path | None

    def source_files(self) -> list[Path]:
        """Return discovered sources in stable role order."""
        return [
            self.experiment,
            self.feature_matrix,
            self.cells,
            self.dapi,
            *([self.image] if self.image else []),
            *([self.alignment] if self.alignment else []),
            *([self.transcripts] if self.transcripts else []),
            *([self.cell_boundaries] if self.cell_boundaries else []),
            *([self.nucleus_boundaries] if self.nucleus_boundaries else []),
        ]


def _resolve(input_dir: Path, value: Path | None) -> Path | None:
    if value is None:
        return None
    return value if value.is_absolute() else input_dir / value


def _first(input_dir: Path, patterns: tuple[str, ...]) -> Path | None:
    for pattern in patterns:
        candidates = sorted(
            path for path in input_dir.rglob(pattern) if path.is_file()
        )
        if candidates:
            return candidates[0]
    return None


def _required(
    input_dir: Path,
    override: Path | None,
    patterns: tuple[str, ...],
    role: str,
) -> Path:
    resolved = _resolve(input_dir, override) or _first(input_dir, patterns)
    if resolved is None or not resolved.exists():
        raise FileNotFoundError(
            f"Xenium {role} was not found; pass its explicit path"
        )
    return resolved


def _discover_dapi(input_dir: Path, override: Path | None) -> Path:
    resolved = _resolve(input_dir, override)
    if resolved is not None:
        if not resolved.is_file():
            raise FileNotFoundError(
                f"Xenium morphology/DAPI image does not exist: {resolved}"
            )
        return resolved

    # Preserve STHR's version-aware preference order. These are all vendor
    # generated 2D autofocus DAPI images; morphology.ome.tif is deliberately
    # excluded because it is the 3D DAPI Z-stack.
    discovered = _first(
        input_dir,
        (
            "morphology_focus.ome.tif",
            "morphology_focus_0000.ome.tif",
            "ch0000_dapi.ome.tif",
        ),
    )
    if discovered is None:
        raise FileNotFoundError(
            "Xenium focused DAPI image was not found; expected an XOA 1.x "
            "morphology_focus.ome.tif, XOA 2.x-3.x "
            "morphology_focus/morphology_focus_0000.ome.tif, or XOA 4.x+ "
            "morphology_focus/ch0000_dapi.ome.tif; morphology.ome.tif is a "
            "3D Z-stack and is not used implicitly"
        )
    return discovered


def _dapi_layout(path: Path) -> str:
    """Return the STHR/XOA layout represented by a focused DAPI path."""
    name = path.name.casefold()
    if name == "morphology_focus.ome.tif":
        return "xoa_1_focus_single_file"
    if name == "morphology_focus_0000.ome.tif":
        return "xoa_2_to_3_focus_multifile"
    if name == "ch0000_dapi.ome.tif":
        return "xoa_4_plus_named_dapi"
    return "explicit_override"


def _discover_he(input_dir: Path, override: Path | None) -> Path | None:
    resolved = _resolve(input_dir, override)
    if resolved is not None:
        if not resolved.is_file():
            raise FileNotFoundError(
                f"Xenium H&E image does not exist: {resolved}"
            )
        return resolved
    candidates = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.casefold() in {".tif", ".tiff", ".btf"}
        and (
            "he_image" in path.name.casefold() or "h&e" in path.name.casefold()
        )
    )
    return candidates[0] if candidates else None


def discover_xenium_files(
    input_dir: Path,
    *,
    experiment: Path | None = None,
    feature_matrix: Path | None = None,
    cells: Path | None = None,
    dapi: Path | None = None,
    image: Path | None = None,
    alignment: Path | None = None,
    transcripts: Path | None = None,
    cell_boundaries: Path | None = None,
    nucleus_boundaries: Path | None = None,
) -> XeniumFiles:
    """Discover one supported Xenium output bundle.

    Example:
        ``discover_xenium_files(Path("raw/xenium-sample"))`` locates the
        vendor cell matrix, cell table, morphology image, and H&E image.
    """
    if not input_dir.is_dir():
        raise ValueError(
            f"input_dir must be an existing directory: {input_dir}"
        )
    alignment_path = _resolve(input_dir, alignment) or _first(
        input_dir,
        (
            "*imagealignment.csv",
            "imagealignment.csv",
            # WTA / Atera exports name the affine ``*_he_alignment.csv`` (no
            # "image"); match it too so the pre-computed matrix is used instead
            # of falling back to VALIS registration.
            "*_he_alignment.csv",
            "*alignment.csv",
        ),
    )
    optional = {
        "transcripts": (
            _resolve(input_dir, transcripts)
            or _first(input_dir, ("transcripts.parquet",))
        ),
        "cell_boundaries": (
            _resolve(input_dir, cell_boundaries)
            or _first(input_dir, ("cell_boundaries.parquet",))
        ),
        "nucleus_boundaries": (
            _resolve(input_dir, nucleus_boundaries)
            or _first(input_dir, ("nucleus_boundaries.parquet",))
        ),
    }
    for role, path in optional.items():
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"Xenium {role} does not exist: {path}")
    if alignment_path is not None and not alignment_path.is_file():
        raise FileNotFoundError(
            f"Xenium image alignment does not exist: {alignment_path}"
        )
    return XeniumFiles(
        experiment=_required(
            input_dir, experiment, ("experiment.xenium",), "experiment file"
        ),
        feature_matrix=_required(
            input_dir,
            feature_matrix,
            ("cell_feature_matrix.h5",),
            "cell feature matrix",
        ),
        cells=_required(input_dir, cells, ("cells.parquet",), "cell table"),
        dapi=_discover_dapi(input_dir, dapi),
        image=_discover_he(input_dir, image),
        alignment=alignment_path,
        transcripts=optional["transcripts"],
        cell_boundaries=optional["cell_boundaries"],
        nucleus_boundaries=optional["nucleus_boundaries"],
    )


def read_xenium_alignment(path: Path, pixel_size_um: float) -> np.ndarray:
    """Read a Xenium affine CSV using HEST's transform convention.

    Xenium alignment exports are either a numeric affine matrix or a table
    of corresponding ``fixed`` and ``alignment`` control points. The latter
    format was introduced by newer Xenium Explorer releases.

    Example:
        ``read_xenium_alignment(Path("imagealignment.csv"), 0.2125)``
        returns a DAPI-micron to H&E-micron affine matrix.
    """
    point_columns = {"fixedX", "fixedY", "alignmentX", "alignmentY"}
    columns = set(pd.read_csv(path, nrows=0).columns.astype(str))
    if point_columns.issubset(columns):
        points = pd.read_csv(path, usecols=sorted(point_columns)).iloc[:3]
        if len(points) < 3:
            raise ValueError(
                "Xenium point alignment requires at least three control "
                "points"
            )
        source = points[["alignmentX", "alignmentY"]].to_numpy(dtype=float)
        destination = points[["fixedX", "fixedY"]].to_numpy(dtype=float)
        design = np.column_stack((source, np.ones(3)))
        try:
            coefficients = np.linalg.solve(design, destination)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Xenium alignment control points do not define an affine "
                "transformation"
            ) from exc
        matrix = np.vstack((coefficients.T, [0.0, 0.0, 1.0]))
    else:
        matrix = pd.read_csv(path, header=None).to_numpy(dtype=float)
    if matrix.shape == (2, 3):
        matrix = np.vstack((matrix, [0.0, 0.0, 1.0]))
    if matrix.shape != (3, 3):
        raise ValueError(
            f"Xenium image alignment must be 2x3 or 3x3, got {matrix.shape}"
        )
    scaled = matrix.copy()
    scaled[0, 2] *= pixel_size_um
    scaled[1, 2] *= pixel_size_um
    try:
        return np.linalg.inv(scaled)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Xenium image alignment matrix is singular") from exc


def _apply_affine(
    centroids_microns: np.ndarray,
    matrix: np.ndarray,
    pixel_size_um: float,
) -> np.ndarray:
    homogeneous = np.column_stack(
        (centroids_microns, np.ones(len(centroids_microns)))
    )
    aligned = (matrix @ homogeneous.T).T
    return aligned[:, :2] / pixel_size_um


def _restore_integer_counts(adata: ad.AnnData) -> None:
    """Restore the integer count dtype lost by Scanpy's 10x H5 reader."""
    if not sparse.issparse(adata.X):
        raise ValueError("Xenium cell feature matrix must be sparse")
    matrix = cast(Any, adata.X)
    data = matrix.data
    if not np.isfinite(data).all() or (data < 0).any():
        raise ValueError("Xenium cell feature matrix contains invalid counts")
    if not np.equal(data, np.floor(data)).all():
        raise ValueError(
            "Xenium cell feature matrix contains non-integer counts"
        )
    maximum = int(data.max()) if data.size else 0
    dtype = np.min_scalar_type(maximum)
    if dtype.kind != "u":
        dtype = np.dtype(np.uint64)
    adata.X = matrix.astype(dtype, copy=False)


def _select_gene_expression(adata: ad.AnnData) -> ad.AnnData:
    """Keep only vendor features labelled as gene expression."""
    if "feature_types" not in adata.var:
        raise ValueError(
            "Xenium feature matrix is missing vendor feature_types metadata"
        )
    selected = adata.var["feature_types"].astype(str) == "Gene Expression"
    if not selected.any():
        raise ValueError("Xenium feature matrix contains no gene features")
    return adata[:, selected.to_numpy()].copy()


def align_centroids_with_valis(
    centroids: np.ndarray,
    dapi_path: Path,
    he_path: Path,
    registration_dir: Path,
    max_registration_dim_px: int,
    check_for_reflections: bool,
    *,
    valis_python: Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run isolated VALIS registration and return warped cell centroids."""
    python = _resolve_valis_python(valis_python)
    registration_dir.mkdir(parents=True, exist_ok=True)
    coordinates_path = registration_dir / "centroids_morphology.npy"
    aligned_path = registration_dir / "centroids_he.npy"
    request_path = registration_dir / "request.json"
    response_path = registration_dir / "response.json"
    np.save(coordinates_path, np.asarray(centroids, dtype=np.float64))
    request = {
        "dapi_path": str(dapi_path.resolve()),
        "he_path": str(he_path.resolve()),
        "coordinates_path": str(coordinates_path.resolve()),
        "aligned_coordinates_path": str(aligned_path.resolve()),
        "registration_dir": str(registration_dir.resolve()),
        "response_path": str(response_path.resolve()),
        "max_registration_dim_px": max_registration_dim_px,
        "check_for_reflections": check_for_reflections,
        "reuse_registrar": True,
    }
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True))
    response_path.unlink(missing_ok=True)
    aligned_path.unlink(missing_ok=True)
    stdout_path = registration_dir / "worker.stdout.log"
    stderr_path = registration_dir / "worker.stderr.log"
    cache_dir = registration_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    worker_env = {
        **os.environ,
        "MPLCONFIGDIR": str(cache_dir / "matplotlib"),
        "XDG_CACHE_HOME": str(cache_dir),
    }
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        completed = subprocess.run(
            [str(python), str(_valis_worker_path()), str(request_path)],
            check=False,
            stdout=stdout,
            stderr=stderr,
            text=True,
            env=worker_env,
        )
    stderr_tail = stderr_path.read_text()[-4000:]
    if not response_path.is_file():
        raise RuntimeError(
            "VALIS worker did not write response.json; "
            f"exit={completed.returncode}; see {stderr_path}; "
            f"stderr={stderr_tail.strip()}"
        )
    response = json.loads(response_path.read_text())
    if completed.returncode != 0 or response.get("status") != "completed":
        raise RuntimeError(
            "VALIS worker failed: "
            f"{response.get('error_type', 'error')}: "
            f"{response.get('message', stderr_tail.strip())}; "
            f"see {response_path}"
        )
    warped = np.load(aligned_path, allow_pickle=False)
    details = {
        "backend": str(response["backend"]),
        "version": str(response["version"]),
        "registrar": str(response["registrar"]),
        "worker_python": str(python),
        "max_registration_dim_px": max_registration_dim_px,
        "check_for_reflections": check_for_reflections,
        "reused": bool(response.get("reused", False)),
        "stdout_log": "registration/worker.stdout.log",
        "stderr_log": "registration/worker.stderr.log",
    }
    for key in (
        "he_reader",
        "dapi_reader",
        "brightfield_processor",
        "source_he",
        "he_alias",
    ):
        if key in response:
            details[key] = str(response[key])
    return np.asarray(warped, dtype=float), details


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _valis_worker_path() -> Path:
    return _repository_root() / "scripts" / "esb_valis_worker.py"


def _resolve_valis_python(explicit: Path | None) -> Path:
    if explicit is not None:
        # Do not resolve the executable symlink: a venv's ``python`` commonly
        # points at its base interpreter, and dereferencing it discards the
        # virtual environment's site-packages at process startup.
        candidate = explicit.expanduser().absolute()
    else:
        executable = "python.exe" if sys.platform == "win32" else "python"
        scripts = "Scripts" if sys.platform == "win32" else "bin"
        candidate = _repository_root() / ".venv-valis" / scripts / executable
    if not candidate.is_file():
        raise FileNotFoundError(
            "VALIS worker Python was not found at "
            f"{candidate}; create .venv-valis or pass --valis-python"
        )
    return candidate


def read_xenium_sample(
    files: XeniumFiles,
    *,
    registration_dir: Path,
    aligner: CentroidAligner | None = None,
    max_registration_dim_px: int = 10_000,
    check_for_reflections: bool = False,
    valis_python: Path | None = None,
) -> tuple[ad.AnnData, dict[str, Any]]:
    """Read cell counts in H&E pixels when available, else native microns."""
    if max_registration_dim_px <= 0:
        raise ValueError("max_registration_dim_px must be positive")
    experiment = json.loads(files.experiment.read_text())
    pixel_size_um = float(experiment["pixel_size"])
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError("Xenium experiment pixel_size must be positive")

    adata = _select_gene_expression(_read_counts(files.feature_matrix))
    _restore_integer_counts(adata)
    cells = pd.read_parquet(files.cells)
    if "cell_id" not in cells or not {"x_centroid", "y_centroid"}.issubset(
        cells
    ):
        raise ValueError(
            "Xenium cells table requires cell_id, x_centroid, and y_centroid"
        )
    cells = cells.set_index("cell_id")
    cells.index = cells.index.map(str)
    joined = cells.reindex(adata.obs_names)
    centroid_columns = ["x_centroid", "y_centroid"]
    if joined[centroid_columns].isna().to_numpy().any():
        missing = int(
            np.count_nonzero(
                joined[centroid_columns].isna().to_numpy().any(axis=1)
            )
        )
        raise ValueError(
            f"Xenium cells table does not cover {missing} matrix barcodes"
        )
    adata.obs = joined
    microns = joined[centroid_columns].to_numpy(dtype=float)
    morphology = microns / pixel_size_um
    affine: np.ndarray | None = None
    if files.image is None:
        spatial = microns.copy()
        registration_details = {
            "backend": "xenium_native_microns",
            "valis_refinement": False,
        }
    elif files.alignment is not None:
        affine = read_xenium_alignment(files.alignment, pixel_size_um)
        spatial = _apply_affine(microns, affine, pixel_size_um)
        registration_details = {
            "backend": "xenium_vendor_affine",
            "alignment_file": str(files.alignment.resolve()),
            "valis_refinement": False,
        }
    elif aligner is None:
        spatial, registration_details = align_centroids_with_valis(
            morphology,
            files.dapi,
            files.image,
            registration_dir,
            max_registration_dim_px,
            check_for_reflections,
            valis_python=valis_python,
        )
    else:
        spatial, registration_details = aligner(
            morphology,
            files.dapi,
            files.image,
            registration_dir,
            max_registration_dim_px,
            check_for_reflections,
        )
    spatial = np.asarray(spatial, dtype=float)
    if spatial.shape != (adata.n_obs, 2) or not np.isfinite(spatial).all():
        raise ValueError(
            "Xenium alignment returned invalid cell-centroid coordinates"
        )
    if files.image is not None:
        with tifffile.TiffFile(files.image) as he_image:
            he_shape = he_image.series[0].shape
        he_height, he_width = int(he_shape[0]), int(he_shape[1])
        in_bounds = (
            (spatial[:, 0] >= 0)
            & (spatial[:, 0] < he_width)
            & (spatial[:, 1] >= 0)
            & (spatial[:, 1] < he_height)
        )
        in_bounds_count = int(np.count_nonzero(in_bounds))
        in_bounds_rate = in_bounds_count / adata.n_obs
        if in_bounds_count == 0:
            raise ValueError(
                "Xenium alignment placed every cell centroid outside the "
                f"H&E image bounds ({he_width}x{he_height} pixels)"
            )
        if in_bounds_rate < MIN_XENIUM_IN_BOUNDS_RATE:
            raise ValueError(
                "Xenium alignment quality failed: only "
                f"{in_bounds_rate:.2%} of cell centroids are inside the H&E "
                f"image; required at least {MIN_XENIUM_IN_BOUNDS_RATE:.0%}"
            )
    else:
        he_height = he_width = in_bounds_count = None
        in_bounds_rate = None
    adata.obsm["spatial_microns"] = microns
    adata.obsm["spatial_morphology"] = morphology
    if affine is not None:
        adata.obsm["spatial_affine"] = spatial.copy()
    adata.obsm["spatial"] = spatial
    used_vendor_affine = affine is not None
    details = {
        "counts_file": str(files.feature_matrix.resolve()),
        "cells_file": str(files.cells.resolve()),
        "experiment_file": str(files.experiment.resolve()),
        "dapi_file": str(files.dapi.resolve()),
        "dapi_layout": _dapi_layout(files.dapi),
        "dapi_image_policy": "vendor_2d_autofocus_only",
        "he_file": str(files.image.resolve()) if files.image else None,
        "alignment_file": (
            str(files.alignment.resolve()) if files.alignment else None
        ),
        "alignment_optional": True,
        "feature_policy": "gene_expression_only",
        "alignment_present": files.alignment is not None,
        "alignment_used_for_final_coordinates": used_vendor_affine,
        "coordinate_source": (
            "xenium_vendor_affine"
            if used_vendor_affine
            else "valis_dapi_to_he"
            if files.image is not None
            else "xenium_native_microns"
        ),
        "pixel_size_morphology_um": pixel_size_um,
        "coordinate_transformations": (
            [
                "microns_to_morphology_pixels",
                "xenium_vendor_affine_to_he_pixels",
            ]
            if used_vendor_affine
            else [
                "microns_to_morphology_pixels",
                "valis_morphology_to_he_pixels",
            ]
                if files.image is not None
                else ["native_xenium_microns"]
        ),
        "optional_sources": {
            "transcripts": (
                str(files.transcripts.resolve()) if files.transcripts else None
            ),
            "cell_boundaries": (
                str(files.cell_boundaries.resolve())
                if files.cell_boundaries
                else None
            ),
            "nucleus_boundaries": (
                str(files.nucleus_boundaries.resolve())
                if files.nucleus_boundaries
                else None
            ),
        },
        "validation": {
            "finite_coordinates": True,
            "he_image_width_px": he_width,
            "he_image_height_px": he_height,
            "in_bounds_cells": in_bounds_count,
            "out_of_bounds_cells": (
                int(adata.n_obs - in_bounds_count)
                if in_bounds_count is not None
                else None
            ),
            "minimum_in_bounds_rate": MIN_XENIUM_IN_BOUNDS_RATE,
            "in_bounds_rate": in_bounds_rate,
        },
        "registration": registration_details,
    }
    if affine is not None:
        details["affine_dapi_to_he"] = affine.tolist()

    # Transcript points, placed in the SAME frame as obsm["spatial"] so the
    # store writer's _to_pixels bakes points and cells identically. Native
    # microns and the vendor affine are supported; the VALIS branch (H&E image
    # but no alignment CSV) is skipped — warping millions of points through the
    # registrar is a separate, heavier step.
    if files.transcripts is None:
        details["transcripts"] = None
    elif files.image is not None and affine is None:
        logger.warning("transcripts skipped: VALIS registration branch")
        details["transcripts"] = None
    else:
        from coral.st.points import normalize_points

        tx = pd.read_parquet(
            files.transcripts,
            columns=[
                "x_location", "y_location", "z_location",
                "feature_name", "cell_id", "qv", "is_gene",
            ],
        )
        xy = tx[["x_location", "y_location"]].to_numpy(dtype=float)
        if affine is not None:  # microns -> H&E pixels, same as the centroids
            xy = _apply_affine(xy, affine, pixel_size_um)
        tx = tx.assign(_x=xy[:, 0], _y=xy[:, 1])
        details["transcripts"] = normalize_points(
            tx, x="_x", y="_y", feature="feature_name", cell_id="cell_id",
            is_gene=tx["is_gene"].to_numpy(), qv="qv", z="z_location",
        )
        logger.info("      Xenium transcripts: %d points", len(tx))
    return adata, details
