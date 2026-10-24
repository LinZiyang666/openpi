# Old-seed regression (10-23 19:0x, single-thread CPU)

`regress compare` reports FAIL for dp_libero, fastwam_libero, cosmos_libero (seed 20261002) and dp_mw (seed 8). Inspected:

- LIBERO (all three bundles, 10 cells each): subset views, native C10/C16 fit.npz, Q3 stall (stall.json/.pkl/.loeo.csv), C5 fit and C5 stall, selection and gaps are byte-identical to the existing products. Only differences: the segment-head Q3 `READY.json` metadata, and `native_refit: independent_refit false` because `inputs cell` copies the library's existing fit for the old seed.
- `regress refit` (dp_libero libero_10/50): fit.npz byte-identical; READY.json differs only in the retired NoProgressGuard description (`config.guard`, `guard_defaults`, `caps_knobs.guard_threshold_source/stall_gate/shared_recipe_sha256`, `shared_products`) and the digests derived from them.
- MT50 dp_mw (6 cells): native C10 fit identical; C10 stall and segment-head Q3 `stall.json` identical in all 366 numeric fields — only `fingerprint`, `payload_sha256`, `provenance.metric_sha256` differ; the old C5 fit (csdp_1504, 10-14) differs from a single-thread refit in the last float digits (e.g. 4.249818801879883 vs 4.249814987182617), i.e. the old product was not computed single-threaded.

Conclusion: the new code reproduces the old seed's products; new-seed products can be generated.
