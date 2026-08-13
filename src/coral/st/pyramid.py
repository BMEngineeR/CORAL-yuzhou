"""Write a pyramid for an image that arrives without one.

CORAL has two store writers and neither can do this. `_write_canonical_zarr`
writes level 0 only, and says so in its log: "source has N pyramid levels;
CORAL stores level 0 only". `_write_reduced_levels` writes a real pyramid but
COPIES it out of the source qptiff's own, because a whole slide arrives with
one already.

A spatial transcriptomics image does not. An H&E from a Space Ranger bundle is
a plain PNG or TIFF, and a Xenium morphology image is an OME-TIFF whose levels
this module does not read. Without reduced levels the viewer requests level 0 at
every zoom, which for a full-resolution H&E is tens of thousands of pixels
decoded to fill a thumbnail.

**The levels are never all in memory.** The first version of this module
returned `list[np.ndarray]`, so level 0 and every reduction of it were held at
once, and `write_st_store` held that list before the write loop even started.
On the samples we have that is 164 MB plus 82 MB and nothing fails. On a
partner's full-resolution H&E it is the run dying. So level 0 is written first,
and each reduction is produced by reading row blocks back out of the level above
it in the store, reducing them, and writing them out. Peak is one block, not one
pyramid.

Level 0 itself still arrives materialised, because the caller decoded it: PIL
has no tiled read path for a Space Ranger PNG. That is a real remaining limit
and it is the reason this is a smaller change than it looks.

The reduction is CORAL's own `_downsample_mean`, imported and never modified, so
the result matches what the rest of the toolkit produces.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from coral.io.ingest import _downsample_mean

logger = logging.getLogger(__name__)

#: Stop when the shorter side reaches this. Below it a level costs a file and
#: saves nothing: the viewer is already fetching one tile.
_MIN_EDGE = 256

#: Halving each time, which is what NGFF consumers assume when they read a
#: multiscales entry without checking its scale.
_FACTOR = 2

#: Chunk edge, matching what the rest of CORAL writes.
_CHUNK = 1024

#: How much of a level to hold while reducing it. A budget in BYTES rather than
#: in rows, because the row that fits a 5,806 px Xenium image is 7x too big for
#: a 40,000 px whole-slide H&E, and the whole point of this module is that the
#: second one has to work.
_BLOCK_BYTES = 8 * 1024 * 1024


def plan_levels(
    shape: tuple[int, int, int], *, max_levels: int = 8, min_edge: int = _MIN_EDGE
) -> list[tuple[int, int, int]]:
    """The shape of every level, computed without touching a pixel.

    Separate from writing them because `_write_ngff_external_attrs` needs the
    shapes to build `multiscales`, and it is called before the pixels exist.

    Args:
        shape: Level 0 as ``(c, y, x)``.
        max_levels: Hard ceiling, including level 0.
        min_edge: Stop once the next level's shorter side would fall below this.

    Returns:
        ``(c, y, x)`` per level, level 0 first.

    Example:
        >>> plan_levels((3, 2000, 1921))
        [(3, 2000, 1921), (3, 1000, 960), (3, 500, 480)]

        It stops there because the next level's shorter side would be 240,
        under the floor. A level below it costs a file and saves nothing.
    """
    if len(shape) != 3:
        raise ValueError(f"expected a (c, y, x) shape, got {shape!r}")
    shapes = [tuple(int(v) for v in shape)]
    while len(shapes) < max_levels:
        channels, height, width = shapes[-1]
        # The NEXT level's shorter side, not this one's: stopping on the
        # current size leaves a final level smaller than the floor.
        if min(height, width) // _FACTOR < min_edge:
            break
        nxt = (channels, height // _FACTOR, width // _FACTOR)
        if nxt[1] < 1 or nxt[2] < 1:
            break
        shapes.append(nxt)
    return shapes  # type: ignore[return-value]


def _block_rows(shape: tuple[int, int, int], itemsize: int) -> int:
    """Source rows to hold at once: a whole number of chunk rows.

    TWO CONSTRAINTS, and getting either wrong is expensive in a different way.

    A MULTIPLE OF THE REDUCTION FACTOR IS CORRECTNESS. `_downsample_mean` trims
    its input to ``h // factor * factor``, so a block boundary falling inside a
    factor-by-factor box would drop that box's remainder in the middle of the
    image rather than at the bottom edge, and the result would differ from
    reducing the level in one piece.

    A MULTIPLE OF THE CHUNK HEIGHT IS SPEED, and this was measured rather than
    assumed. Sizing blocks purely by a byte budget gave 232-row blocks against
    1,024-row chunks, so zarr read-modify-wrote every chunk five times and the
    whole write ran 4.8x slower than building the pyramid in memory. Rounding up
    to whole chunks removed it. `_CHUNK` is even, so this satisfies the first
    constraint for free.

    The consequence is that the byte budget is a floor and not a ceiling: one
    chunk row of a very wide image is what it is. Memory stays independent of
    image HEIGHT, which is what makes a whole-slide image survive, and grows
    with its width, which is inherent to reducing by rows at all.
    """
    _, _, width = shape
    per_row = max(1, shape[0] * width * itemsize)
    chunks = max(1, int(_BLOCK_BYTES // (per_row * _CHUNK)))
    return chunks * _CHUNK


def reduce_into(source: Any, dest: Any, itemsize: int) -> None:  # noqa: ANN401 - zarr arrays
    """Fill ``dest`` by box-averaging ``source`` one row block at a time.

    Iterates over the DESTINATION's rows, so each read is exactly
    ``factor`` times the block being written and nothing has to be carried
    between iterations. Source rows past ``dest_height * factor`` are dropped,
    which is what reducing the whole array in one piece does too.
    """
    channels, out_height, _ = dest.shape
    out_block = max(1, _block_rows(tuple(source.shape), itemsize) // _FACTOR)
    for out_y in range(0, out_height, out_block):
        out_end = min(out_y + out_block, out_height)
        block = source[:, out_y * _FACTOR : out_end * _FACTOR, :]
        dest[:, out_y:out_end, :] = np.stack(
            [_downsample_mean(block[c], _FACTOR) for c in range(channels)]
        )


def write_pyramid(root: Any, image: np.ndarray, shapes: list[tuple[int, int, int]]) -> None:  # noqa: ANN401 - a zarr group
    """Write every level into ``root`` as arrays named ``0``, ``1``, ...

    Level 0 comes from ``image``, written in row blocks so zarr is never handed
    the whole array at once. Every level after it is read back out of the level
    above, which is why the pyramid does not grow with its own depth.

    Args:
        root: An open zarr group.
        image: Level 0 as ``(c, y, x)``.
        shapes: From `plan_levels`, level 0 first.
    """
    if tuple(image.shape) != tuple(shapes[0]):
        raise ValueError(
            f"image is {tuple(image.shape)} but the plan starts at {shapes[0]}"
        )

    def _new(index: int, shape: tuple[int, int, int]) -> Any:  # noqa: ANN401
        # zarr v2 library API (the fork pins zarr <3); create_dataset writes
        # the same v2-format array a zarr-v3 create_array(zarr_format=2) would.
        return root.create_dataset(
            str(index),
            shape=shape,
            dtype=image.dtype,
            chunks=(1, min(_CHUNK, shape[1]), min(_CHUNK, shape[2])),
        )

    rows = _block_rows(shapes[0], image.dtype.itemsize)
    level = _new(0, shapes[0])
    for y in range(0, shapes[0][1], rows):
        level[:, y : y + rows, :] = image[:, y : y + rows, :]
    logger.info("      level 0 %s written in blocks of %d row(s)", shapes[0], rows)

    for index, shape in enumerate(shapes[1:], start=1):
        nxt = _new(index, shape)
        reduce_into(level, nxt, image.dtype.itemsize)
        logger.info("      level %d %s reduced from level %d", index, shape, index - 1)
        level = nxt
