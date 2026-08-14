# coral.st — imaging-ST extension status (compact handoff)

## What we're doing

Extend `coral ingest-st` from spot ST to **imaging ST**, one sample → one CORAL
store, **without changing the frozen core**. New capabilities land as **additive
top-level sidecar groups**.

```
<sample>.zarr/
├── 0/ 1/ .../            root image pyramid (c,y,x)   ── FROZEN
├── OME/METADATA.ome.xml                                ── FROZEN
├── st/<tech>/            AnnData counts + config.json  ── FROZEN
├── points/transcripts.parquet   transcript points     ── DONE (additive)
├── shapes/…             cell/nucleus boundaries        ── NEXT (additive)
├── labels/…             segmentation masks             ── later
├── state.json · structure.txt
```

Branch: `yuzhou-dev` on `BMEngineeR/CORAL-yuzhou` (fork; `origin` = upstream
mahmoodlab, pull-only).

## Done

- **Native readers** (no SpatialData dep): `cosmx.py` (NanoString AtoMx, multi-FOV
  → **compact mosaic**), `g4x.py` (Singular Genomics), plus the existing
  `xenium.py` (incl. Atera = Xenium WTA + H&E). Each returns `(AnnData, image,
  details)`; `.X` = raw counts, `obsm["spatial"]` = cell/spot centroids.
- **mpp guardrail** (`resolution.py`): read the real pixel size, cross-check,
  **refuse unconfident values** (like the marker guardrail) — `--mpp` /
  `--confirm-mpp`. `is_valid_pixel_size` (finite & >0) guards every entry.
- **transcript points** (`points.py` + `store.write_st_store` points path):
  `points/transcripts.parquet` — cols `x,y,feature_name,cell_id,is_gene,qv?,z?`.
  All points kept, flagged. `--points/--no-points` (default on).

## Key conventions (reuse these for shapes)

1. **Frozen core = root image + `st/<tech>/`.** Never touch. New data = new
   top-level group.
2. **Coordinate invariant (critical):** a reader returns any secondary
   coordinates in the **same frame as `obsm["spatial"]`**; `write_st_store`
   applies the **same `_to_pixels(frame, image_scale, pixel_size_um)`**
   (`store.py:272-309`) to all of them → everything lands in one level-0 pixel
   frame, aligned by construction. Per platform:
   - **Xenium**: native microns (frame `microns`) OR H&E via `_apply_affine`
     (`xenium.py:276`) (frame `fullres_pixels`); VALIS branch skipped.
   - **CosMx**: `x_local_px + FOV_compact_offset` (the `x_off` dict in
     `cosmx.read_cosmx_sample`); frame `fullres_pixels`.
   - **G4X**: image pixels directly; frame `fullres_pixels`.
3. **Hand-off:** reader puts the secondary data on `details[...]` (a DataFrame);
   `pipeline.ingest_one` **pops it out of `details`** (so it doesn't hit the
   JSON config) and passes it to `write_st_store` as a param. See how `points`
   flows: `details["transcripts"]` → `transcripts_df = details.pop(...)` →
   `write_st_store(points=...)`.
4. **Provenance, no schema bump:** add a block to `config.json` and an output to
   `state.tasks.st[record].outputs` (`TaskState` is `extra="allow"`,
   `state.py`). Write the sidecar after the `st/` write, before state
   (`store.py`, the `points_config` block is the template).

## File map

| file | role |
|---|---|
| `pipeline.py` | `ingest_one` (resolve mpp, guard, hand off image/points), `_read` (per-tech dispatch) |
| `store.py` | `write_st_store` (root image + `st/` + `points/` + config/state), `_to_pixels`, `read_st_config` |
| `resolution.py` | mpp resolver + `MppNeedsConfirmation` + `is_valid_pixel_size` |
| `points.py` | `normalize_points()` — one transcript schema across platforms |
| `cosmx.py` / `g4x.py` / `xenium.py` | native readers (counts + image + transcripts in `details`) |
| `cli/ingest_st.py` | flags: `--mpp`, `--confirm-mpp`, `--points/--no-points` |
| `MPP_VERIFICATION.md` | mpp design + open items (Anurag: Visium convention, Xenium-H&E; Cristina: HEST NaN) |
| `INGEST_ROBUSTNESS.md` | ingest brute-force (0 problems) |

Test notebook: `tutorials/7-Spatial-Transcriptomics-Ingest.ipynb` (8 samples).
Data: `STFM/Data/CORAL_st_tutorial_data/` (cosmx / g4x / visium / visium-hd /
hest / spatial-transcriptomics / xenium_prime5k / atera).

## Open follow-ups (documented, not blocking)

- mpp: Visium diameter-vs-spacing convention (**Anurag**); Xenium H&E-root mpp
  should be H&E's ≈0.274 not morphology 0.2125 (**Anurag**); HEST `NaN` embedded
  → estimated fallback trustworthiness (**Cristina**).
- points: no finiteness filter on transcript x/y — NaN/inf coords would write
  silently (real data is clean; guard is the same `isfinite` idea as mpp).

## NEXT: `shapes/` — cell & nucleus boundaries

Same additive-sidecar pattern as `points/`. Boundary data per platform:

| platform | boundary source | coordinate frame |
|---|---|---|
| **Xenium** | `cell_boundaries.parquet`, `nucleus_boundaries.parquet` (vertices in **microns**, grouped by `cell_id`) | apply the SAME Xenium branch transform as cells (`_apply_affine` / microns) |
| **CosMx** | `<sample>-polygons.csv.gz` (`fov,cellID,cell,x_local_px,y_local_px,x_global_px,y_global_px`) | `x_local_px + FOV_compact_offset` (same as cells/points) |
| **G4X** | `segmentation/*.npz` **raster masks** (`nuclei`, `nuclei_exp`) — not polygons; vectorize to boundaries **or** store the mask under `labels/` instead | image pixels (identity) |

Proposed:
- Storage: `shapes/cell_boundaries.geojson` (+ `nucleus_boundaries.geojson`) —
  reuse the GeoJSON convention in `coral/tissue/mask.py:write_tissue_geojson`
  (Feature-per-polygon, level-0 px) / the dearray `boxes_to_geojson` precedent.
  Vertices baked with the same `_to_pixels`. (Parquet/GeoParquet is the
  alternative if SpatialData interop is wanted.)
- Reader hand-off: `details["shapes"]` = `{cell: DataFrame(cell_id, x, y[, part])
  in cells' frame, nucleus: ...}`; `store.write_st_store(shapes=...)` writes the
  geojson + config `shapes` block + state output — mirror the `points` code path.
- G4X decision to make: derive polygons from the mask, or ship the mask as a
  `labels/` group and skip G4X shapes for now.
- Verify: overlay boundaries on the H&E/nuclear image at the same scale as the
  cell centroids; boundary of a cell wraps its centroid.
