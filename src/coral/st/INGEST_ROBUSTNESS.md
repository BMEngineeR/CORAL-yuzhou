# `coral ingest-st` — robustness brute-force

Adversarial / edge-case testing of the ingest pipeline *beyond* the mpp
guardrail (that has its own record in `MPP_VERIFICATION.md`). No code was
changed; this documents the coverage and the result.

**Result: every case behaves correctly — 0 real problems.**

## Cases and outcomes

| area | case | expected | result |
|---|---|---|---|
| input | `--sample-dir` does not exist | clear error | `FileNotFoundError` ✓ |
| input | empty directory | not detected, named | `ValueError` ✓ |
| input | directory of unrelated files | not detected, names what each reader looked for | `ValueError` with reason ✓ |
| input | force the wrong technology (`--technology CosMx` on a Visium bundle) | reader fails cleanly, not a swallowed crash | `FileNotFoundError` before any write ✓ |
| overwrite | re-ingest the same bundle into the same job dir | idempotent overwrite (picks up code changes) | same store, overwritten ✓ |
| overwrite | a *different* bundle whose name collides with an existing store | refused, naming both sides | `ValueError` "already holds a … record ingested from …" ✓ |
| completeness | bundle missing a required file (G4X without `cell_by_transcript`) | error names the missing file | `FileNotFoundError` naming `cell_by_transcript` ✓ |
| store contract | `config.json` with a future `st_schema_version` | refused (no silent migration) | `ValueError` about the version ✓ |
| store contract | `config.json` with no version at all | refused | `ValueError` ✓ |

## Notes

- One case (`different bundle, colliding name`) first read as a false failure
  only because the test's keyword matcher was too narrow; the refusal message
  (`… already holds a G4X record ingested from …`) is exactly the intended
  `_refuse_foreign_overwrite` guard, so it is correct.
- The per-sample loop in `coral ingest-st` already isolates failures: one bad
  sample logs a warning and the run continues, and the exit code is non-zero
  only when nothing was written — so none of the above can sink a cohort.
- Coverage is API-level (`ingest_one`, `detect_technology`, `discover_*`,
  `read_st_config`) plus the store-contract read-back. CLI-only guards
  (duplicate directory names, `--image` with more than one sample) are enforced
  in `cli/ingest_st.py` and were not exercised here.
