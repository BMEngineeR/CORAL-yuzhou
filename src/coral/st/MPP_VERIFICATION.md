# Pixel size (mpp) verification for `coral ingest-st`

**For review by Anurag.** This documents how the ST ingest decides the store's
`mpp` (microns per pixel of the root image) and — crucially — how it *verifies*
that number instead of trusting it silently. Code lives in
`coral/st/resolution.py`; it is wired into `coral/st/pipeline.py:ingest_one`.

## Why this exists

`mpp` is the only pixel↔micron scale in the store. Every physical measurement
and every cross-sample comparison rides on it, but the ST vendors are wildly
inconsistent about recording it: some write it to an instrument field, some
bury it in a scalefactors JSON, some only let you derive it from spot geometry,
and some (a cropped subset) drop it entirely. Silently defaulting to `1.0` — as
the first cut did for CosMx/G4X/Visium HD — produces a store that *looks* fine
and is off by 3–8×.

So mpp is resolved the way the **proteomics marker guardrail** resolves markers:
read it, cross-check it, and when the result is not confident **refuse to write
the store** until the user confirms. Nothing about mpp is decided silently.

## The three-tier confidence model

| tier | meaning | needs confirmation? |
|---|---|---|
| `instrument` | read from a field the instrument itself wrote | no (if cross-checks pass) |
| `derived` | computed from geometry in the data (spot size, FOV px/mm) | no (if cross-checks pass) |
| `estimated` | a third-party pipeline's estimate (HEST) | no (if cross-checks pass) |
| `default` | the platform constant, used because the data had no value | **yes, always** |
| `unknown` | no value and no platform default | **yes, always** |
| `user` | set explicitly via `--mpp` | no |

Only a value that was actually read/derived **and** passes every cross-check
proceeds automatically. A fallback to a platform default always blocks.

## Per-platform pixel size — value, source, how it is verified

| platform | mpp (µm/px) | source (file → field) | verification |
|---|---|---|---|
| **Xenium / Atera** | 0.2125 | `experiment.xenium` → `pixel_size` | instrument-written; frame span × pixel_size matches the cell-coordinate range |
| **CosMx** | 0.120281 | `RunSummary/*_ExptConfig.txt` → `ImPixel_nm` (÷1000) | cross-checked against `fov_positions` paired px/mm columns (agree to 0.001%) |
| **G4X** | 0.3125 | `g4x_viewer/*.ome.tiff` OME-XML `PhysicalSizeX` | `median_cell_area_um² 58.0078125 / 0.3125² = 594` px, an exact integer; other candidates are not |
| **Visium HD** | 1.003128 | `binned_outputs/*/spatial/scalefactors_json.json` → `microns_per_pixel` | `bin_size_um 2.0 / spot_diameter_fullres 1.99376 = 1.003128`, matches the field; the 2/8/16 µm bins all give the same number |
| **Visium** | 0.61–0.73 (⚠ see below) | no direct field | spot-diameter method `55 / spot_diameter_fullres = 0.6146`; spot-spacing method `100 / pitch = 0.7246`; the two disagree ~18% |
| **HEST** | 3.235654 | `metadata.json` → `pixel_size_um_estimated` | third-party estimate; `pixel_size_um_embedded` is NaN (the source image had no resolution) |

Instrument constants (Xenium, CosMx, G4X) are registered as platform defaults
in `PLATFORM_PIXEL_SIZE_UM`; the image-specific platforms (Visium, Visium HD,
HEST, legacy ST) deliberately have **no default** — a fake default there is the
most dangerous case, so the resolver refuses rather than invent one.

## The check mechanism (all of it is metadata-only, ~sub-millisecond)

Two independent kinds of cross-check, neither of which touches image pixels or
the count matrix:

1. **read value vs platform default** — for the instrument platforms. Catches a
   wrong file, a corrupted field, or the wrong crop.
2. **method A vs method B** — for derived platforms. CosMx compares `ImPixel_nm`
   against the FOV px/mm derivation; Visium exposes the diameter-vs-spacing
   disagreement instead of silently picking one.

Any cross-check outside a 2% tolerance flips `needs_confirm` to true and is
named in the refusal.

## The confirm flow (mirrors the marker guardrail)

When a sample's mpp is not confident and was not confirmed, `ingest_one` raises
`MppNeedsConfirmation` and the sample is skipped with a message naming exactly
what is uncertain and how to resolve it. Two ways to confirm, both on the CLI:

```
coral ingest-st --sample-dir <dir> --job-dir <out> --mpp 0.3125      # set it explicitly
coral ingest-st --sample-dir <dir> --job-dir <out> --confirm-mpp     # accept the resolver's value
```

`--mpp` is the analog of the proteomics `--mpp`; `--confirm-mpp` is the analog
of editing `marker_map.csv` to accept a flagged mapping.

## What is recorded in the store

Every store's `config.json` gets a `pixel_size` provenance block under
`source.technology_details`, so the number is auditable after the fact:

```json
"pixel_size": {
  "value": 0.120281, "tier": "instrument",
  "source": "ExptConfig.txt:ImPixel_nm",
  "default": 0.120281,
  "crosschecks": [{"name": "fov_px_mm", "value": 0.120284, "agree": true},
                  {"name": "platform_default", "value": 0.120281, "agree": true}],
  "needs_confirm": false, "reason": ""
}
```

## Current status and open items (please sanity-check these calls)

- ✅ **CosMx** — instrument value + agreeing cross-check, proceeds.
- ✅ **Visium HD** — now reads `microns_per_pixel`, proceeds.
- ✅ **G4X** — full export reads OME-XML `PhysicalSizeX`; the *subset* has no
  resolution tags, so it correctly falls to the default and **blocks** until
  `--confirm-mpp` / `--mpp 0.3125`.
- ⚠ **Visium** — currently proceeds with the **spacing** method (~0.7264). The
  scanpy/10x convention is the **diameter** method (~0.6146). The two disagree
  because the scalefactors are not internally consistent with nominal 55/100 µm.
  **Open question for Anurag: which convention should CORAL standardize on?**
- ⚠ **Xenium with H&E root** (Atera) — the store's root image is the H&E, but
  the resolver still reports the morphology `pixel_size` (0.2125). The H&E mpp
  is `morphology_mpp × alignment_scale` (≈0.274 here). mpp should follow the
  **root image**, not the morphology frame. Not yet wired.
- ⚠ **HEST — assigned to Cristina.** HEST writes `pixel_size_um_embedded = NaN`
  whenever the source image carries no embedded resolution. This is **genuine
  HEST data, not a coral artifact** — verified in the staged sample **NCBI680**'s
  raw `hest/raw/metadata.json`, which literally contains
  `"pixel_size_um_embedded": NaN` (HEST emits bare `NaN` tokens; the same appears
  for `organ`, `oncotree_code`, `patient`, `license`). coral no longer stores
  `NaN` (the reader now falls back to `pixel_size_um_estimated`, 3.2357 µm/px for
  NCBI680 — see P4 below), **but that estimate is a tier-3 value HEST derives
  from spot geometry**, not a measured resolution.
  **Action (Cristina):** confirm the HEST `pixel_size_um_estimated` fallback is
  trustworthy for HEST samples generally, and decide whether HEST needs a more
  robust pixel-size source (e.g. from the original Visium bundle) or should carry
  a lower-confidence flag on `mpp`. HEST is Cristina's area (she staged the ESB
  HEST tutorial data).

The CosMx / Visium HD / G4X items are done; the Visium-convention, Xenium-H&E,
and HEST items are known follow-ups (HEST owned by Cristina, Visium/Xenium by
Anurag).

## Known issues from brute-force testing

A stress test of the resolver and the `--mpp` path found one root cause with
several faces: **nothing validates that the value is a finite POSITIVE number.**
The guardrail verifies provenance and agreement, but never sanity of the value
itself, so garbage flows straight to the store.

| # | sev | where | problem | status |
|---|-----|-------|---------|--------|
| P1 | HIGH | `resolve_pixel_size` | negative / NaN / inf `read_value` accepted, `needs_confirm=False` | **fixed** — invalid read ignored → default/unknown, needs_confirm |
| P2 | MED | same | `read_value == 0.0` treated as "no value" (falsy `if read_value:`) | **fixed** — validated by `is_valid_pixel_size`, not truthiness |
| P3 | HIGH | `ingest_one` `--mpp` | override not validated | **fixed** — `--mpp` raises unless finite & > 0 |
| P3b | HIGH | same | `--mpp 0` silently becomes `1.0` | **fixed** — `--mpp 0` now rejected |
| P4 | HIGH | HEST branch + resolver | `embedded (NaN) or estimated` short-circuits to NaN | **fixed** — HEST prefers a finite embedded, else estimated (3.2357) |
| P5 | MED | cross-checks | default check looks circular when read == the platform constant | **not a bug** — the platform-default comparison is a real check that catches a garbage instrument read (e.g. 1e9 vs 0.2125); it only *looks* circular when the read equals the default, which is the correct case |

The fix is a single guard, `is_valid_pixel_size(v)` (finite and strictly
positive), applied at every entry point: the resolver ignores an invalid read
(falls to default/unknown, needs_confirm), the `--mpp` path raises, and the
HEST branch prefers a finite embedded value over NaN. Re-running the
brute-force suite after the fix reports zero remaining problems.
