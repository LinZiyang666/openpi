# Loader fixes for authorized STEP 2

1. **Contract identity hash.** The loader incorrectly compared
   `contract.manifest_sha256` with the byte hash of `dev_selection.json`.
   `devset.arm_contract` actually binds `load_manifest`'s SHA-256 of
   `json.dumps(sorted(selected_pairs), separators=(',', ':')).encode()`.
   The loader now checks that canonical identity hash against the frozen,
   exactly matched B pairs. `ROOT.json` still verifies the manifest and
   contracts' file-byte hashes; the frozen selection-registration hash,
   pool, balance, fits, subsets and all other existing checks remain.
2. **Per-arm selection binding.** The loader previously skipped each run's
   `manifest.json` and `selection.json`. It now requires the arm definition
   to point to the root selection, the B-tagged run selection to point to
   its own manifest, and that manifest to be byte-identical to the sealed
   root selection. Both run files enter the existing input snapshot and
   stability verification.

The synthetic loader fixture now represents these same hash semantics and
per-arm files. No decision rule, tolerance, statistic or output code changes.
`analyze.before_loader_fix.py` preserves the original source for comparison.
The three frozen protocol files are untouched.

Validation: `analyze.py apply` completed for all 32 arms / 9,600 accepted B
episodes and wrote all 15 registered outputs (`apply.log`). All 12 synthetic
tests passed, including rejection of the old byte-hash contract, wrong pool
or manifest pointers, and altered run manifests. `verification.json` records
419 rehashed inputs, unchanged frozen hashes, byte-identical production code
outside `load_cell`, preserved remaining integrity checks, exact emitter-format
validation, first-failure selections, and primary/5 pp byte equality. The
Chinese owner section leads `DESCENT.md` and contains no agent codenames.
No test-A data was read and no emitter or experiment was run.
