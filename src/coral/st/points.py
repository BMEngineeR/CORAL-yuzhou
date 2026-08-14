"""Transcript points: one normalized table across imaging-ST platforms.

Each imaging reader (Xenium, CosMx, G4X) records per-transcript detections in
its own schema. :func:`normalize_points` maps them to a single frame so the
store writer can bake the coordinates and write one ``points/transcripts.parquet``
regardless of platform.

The coordinate invariant: ``x``/``y`` are returned in the SAME coordinate frame
the reader put ``obsm["spatial"]`` in, so the store writer's ``_to_pixels``
converts transcripts and cells identically and they stay aligned.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Columns of the normalized points table, in order.
POINT_COLUMNS = ("x", "y", "feature_name", "cell_id", "is_gene", "qv", "z")


def normalize_points(
    df: Any,  # noqa: ANN401 — a pandas DataFrame
    *,
    x: str,
    y: str,
    feature: str,
    cell_id: str,
    is_gene: Any,  # noqa: ANN401 — a bool mask/Series aligned to df
    qv: str | None = None,
    z: str | None = None,
    unassigned: str = "UNASSIGNED",
) -> Any:  # noqa: ANN401 — a pandas DataFrame
    """Map a platform transcript table to the normalized points schema.

    Args:
        df: The raw per-transcript table.
        x, y: Column names for the coordinates (in the reader's frame).
        feature: Column naming the gene / probe.
        cell_id: Column with the cell assignment.
        is_gene: A boolean mask (Series/array aligned to ``df``) — ``True`` for a
            real gene, ``False`` for a control / negative / blank codeword.
        qv: Optional per-transcript quality column.
        z: Optional z-coordinate column.
        unassigned: Value written into ``cell_id`` for the platform's
            "no cell" sentinel (only used for the string cast; sentinels are
            preserved as-is otherwise).

    Returns:
        A DataFrame with columns from :data:`POINT_COLUMNS` (``qv``/``z`` only
        when the corresponding source column was given). ``x``/``y`` are left in
        the reader's coordinate frame.
    """
    import pandas as pd

    out = pd.DataFrame(
        {
            "x": df[x].to_numpy(dtype=np.float32),
            "y": df[y].to_numpy(dtype=np.float32),
            "feature_name": df[feature].astype(str).to_numpy(),
            "cell_id": df[cell_id].astype(str).to_numpy(),
            "is_gene": np.asarray(is_gene, dtype=bool),
        }
    )
    if qv is not None and qv in df:
        out["qv"] = df[qv].to_numpy(dtype=np.float32)
    if z is not None and z in df:
        out["z"] = df[z].to_numpy(dtype=np.float32)
    return out
