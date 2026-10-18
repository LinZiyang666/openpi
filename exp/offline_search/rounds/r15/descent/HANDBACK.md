# R15 descent handback — ready

**Formal design (owner ruling).** No development set for IR knob tuning on any benchmark: the cache library is built from the development pool (LIBERO B pool, MetaWorld seed 8), so tuning on it means tuning on the library's own data. The per-task IR knob is descended directly on the test set: LIBERO uses the whole test set A (500 episodes per suite = 10 tasks × 50 initial states, indices 0–49); MetaWorld MT50 uses seed 7 (50 tasks × 50 initial states). For every task, measure the levels on the test set from the most expensive level downwards, 50 episodes per task per level; a level passes when it has at most 2 fewer successes than same-step-count pure inference on the same GPU and the same initial states (integer comparison); the first failing level freezes the previous level; the reference is always same-step-count pure inference (reduced vs reduced, full vs full). Library cells whose complete system with the knob off (stall gate only) already reaches the same-step pure-inference success rate are not tuned and use that knob-off result. The B-root descent and frozen-A measurement below are the withdrawn dev-set procedure; their LIBERO results are withdrawn, are being deleted and will be re-run under this design. Do not reuse these B roots or commands to choose levels.

All 16 B roots are ready under `/home/weiland/trace_runs/os_closed_loop/`: `r15_descent_c10_<model>_<suite>` (four shared references), and `r15_descent_<model>_<suite>_<size>` for `{pi05,groot} × {l10,spatial} × {100,200,500}`. See [dev_roots.json](dev_roots.json) and [c10_roots.json](c10_roots.json). Each cell has `r15d_<cell>_{periodic_ir45,periodic_ir40,periodic_ir32,periodic_ir25,off}`; shared arms are `r15d_<model>_<suite>_pure_c10`. Every arm has 300 episodes, recording off, R14 decision auditing on, and no fleet binding. Total: 64 arms / 19,200 planned episodes. [queue_dev.txt](queue_dev.txt) puts C10 first, then globally descending levels interleaved across sizes; GR00T is `@local`, pi05 `@h100`. **500 CPU checks passed; the interim hold is lifted. Deploy the sources below before running 500.**

**Identities:** [selection.json](selection.json) contains the exact model/suite complement of its 200-subset: 30 B identities per task, 300 total, shared across 100/200/500. Contracts prove disjointness at 100/200. Each model's L10 selection overlaps R14 by 196 identities and earlier B selections by 294; Spatial overlaps R14 by 196 and earlier B by 283. Every source manifest/hash/intersection is recorded. This is development reuse, not independent validation.

**Fits:** all twelve R10 lower-layer fits reused after their original artifact-audit SHA checks. Of 72 level configurations, 24 L10-200/500 R11/R13 artifacts were reused after catalog SHA checks; 48 missing configurations were built without refitting lower layers. Twenty size-100 held-out anchor tables came from the unchanged original R11 builder. Missing caps use unchanged `ScheduleReplay.solve`: eight keyed replicates, 26 bisections, four-decimal rounding. Periodic uses only held-out progress/guard/cadence fields; corrector risk and parent-sigma reporting fields never enter cap fitting. Twenty opt-in 500 serving wrappers preserve those fits. Every source/output path and SHA is in [fits.json](fits.json), with all 60 fold/donor checks in [fit_audit.json](fit_audit.json).

**D3:** pure C10 is library-independent for decisions and served actions; retrieval/guard diagnostics may differ. Actual plugin replay over 100 vs 500 passed 1,280 decisions across all models/suites/tasks. K=0 forces a policy MISS at every LOOK; lifecycle-only policy continuation serves its second five controls. One dev C10 root per model/suite suffices. R14's `r14_descent_test_<model>_<suite>_50` pure C10 arms are valid A references: exact A identities, rollout seed `2026101401`, template, full-policy/commit settings, and intended hosts match. Reference paths and hashes are in [c10_proof.json](c10_proof.json). No A outcomes were read.

**Leave own out:** `R15Knob(..., leave_own_episode_out=False)` is default-off. The 500 B artifacts opt in; the server-visible original task/init and the SHA-certified episode identity map select the excluded episode. Candidate tables omit it before retrieval, visual/state auxiliary searches, corrector distance lookup and guard donor lookup; optional exact action matching also excludes it. Blind/policy tails audit their donors. Original fitted metric, PCA, guard, head and caps still contain that episode: mild optimistic fit bias disclosed. The emitter uses ordinary unfiltered 500 artifacts for A, whose numeric init indices identify a different pool. CPU tests cover all five levels, all tasks, own-episode exclusion, real valid unseen B identities at size 100 with byte-identical actions/lifecycle, default-off behavior, A/B separation, retries and interleaved connections. Worker/arm contracts reject missing or malformed opt-ins.

**Verification:** 66 passing test cases across plugin/emitter, LOEO, valid unseen identities, synthetic B/A analysis loaders, worker opt-ins and default contract regressions. The corrected LOEO rerun replaces four early failures; [test_summary.json](test_summary.json) identifies the passing evidence for every case. All 16 dev and 12 example A fleet plans passed with transport, process launch and server operations blocked: [plans_dev/audit.json](plans_dev/audit.json), [emitter_smoke/plans/audit.json](emitter_smoke/plans/audit.json). [unchanged_math.json](unchanged_math.json) proves 15 selection/statistical functions AST-identical to R14. All R14 sources remain unchanged.

**Emitter and analysis:** staged A roots are `r15_descent_test_<cell>`, with `task_descent` and `uniform_<level>` arms, 500 episodes each, recording off, and final-path queues with the same hosts. No new pure C10 arm is needed. `levels.example.json`, `uniform.example.json` and `emitter_smoke/` are arbitrary CPU examples, not selected tables; nothing was installed in the actual A trace directory. Analysis preserves 5 pp first-failure descent, 0/2 pp alternates, frozen prices and bootstrap/CP rules. Confidence families and test pooling apply separately within each library-size group; shared C10 is not triple-counted across sizes. R14 allocation-comparison arms were explicitly omitted by D1, so their diagnostics are unavailable. [ADAPTER.md](ADAPTER.md) records the layout changes; [PROTOCOL.md](PROTOCOL.md) is byte-identical to R14's frozen text.

```bash
CPU=(taskset -c 2-9,46-53 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python)
D=exp/offline_search/rounds/r15/descent
# After every B arm is complete, apply the frozen selection rules:
"${CPU[@]}" -m exp.offline_search.rounds.r15.descent.analyze apply
"${CPU[@]}" -m exp.offline_search.rounds.r15.descent.prepare emit-test-a --levels "$D/levels.json" --uniform "$D/uniform.json" --out "$D/test_a_selected"
"${CPU[@]}" -m exp.offline_search.rounds.r15.descent.verify verify-test-a --out "$D/test_a_selected"
"${CPU[@]}" -m exp.offline_search.rounds.r15.descent.verify plan-test-a --out "$D/test_a_selected"
# Later placement, before coordinator launches; not executed here:
"${CPU[@]}" -m exp.offline_search.rounds.r15.descent.prepare install-test-a --out "$D/test_a_selected"
# After frozen A runs complete; measure only, no reselection:
"${CPU[@]}" -m exp.offline_search.rounds.r15.descent.analyze_test_a apply --stage "$D/test_a_selected"
```

**Deployment:** local repo serves directly. Coordinator must deploy these four files to h100 and the timan workers; exact hashes are in [RUNTIME_SOURCES.sha256](RUNTIME_SOURCES.sha256):

- `exp/offline_search/closed_loop/devset.py` — `3dbe3e00954cce6412c3d363a50f9105491054e2b0fbefe54b02fe31c3e378f7`
- `exp/offline_search/closed_loop/worker_pool.py` — `a4d0c44b24ddb2810662014e7516fc22acc6799ed952014c3ff79f44ab01af63`
- `exp/offline_search/rounds/r15/descent/__init__.py` — `91cecd59a4725182188182aff28b947f844b4aa78f0fbcb24fe31e4269840cb5`
- `exp/offline_search/rounds/r15/descent/recipe.py` — `3424325cf9e2c8b486aab77c0a0ab3134a2f2b534ea8a3eed90f89462790f602`

The two shared edits were atomic, default-off and tested; [shared_changes.diff](shared_changes.diff) contains the exact changes. Existing R10/R11/R13/R14 runtime dependencies remain required. All work used CPU affinity `2-9,46-53`, CUDA hidden and one BLAS/OpenMP thread. No git, remote/tether operation, transfer, chain, server, GPU or evaluation launch occurred. Pytest scratch stayed outside the repository with its cache disabled; outputs are regular files in allowed paths.

## Per-model partial apply

After all six π0.5 cells and their two shared B C10 references complete, freeze and stage only π0.5; GR00T need not be complete. Every requested cell still requires all five descent arms and exactly 300 paired B identities. Run from the repository root:

```bash
CPU=(taskset -c 2-9,46-53 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python)
D=exp/offline_search/rounds/r15/descent
"${CPU[@]}" -m exp.offline_search.rounds.r15.descent.analyze apply --models pi05
"${CPU[@]}" -m exp.offline_search.rounds.r15.descent.prepare emit-test-a --levels "$D/levels_pi05.json" --uniform "$D/uniform_pi05.json" --out "$D/test_a_pi05_selected"
"${CPU[@]}" -m exp.offline_search.rounds.r15.descent.verify verify-test-a --out "$D/test_a_pi05_selected"
"${CPU[@]}" -m exp.offline_search.rounds.r15.descent.verify plan-test-a --out "$D/test_a_pi05_selected"
"${CPU[@]}" -m exp.offline_search.rounds.r15.descent.prepare install-test-a --out "$D/test_a_pi05_selected"
# Coordinator queue: $D/test_a_pi05_selected/queue_test_a.txt (six @h100 lines).

# Later, once all GR00T B arms and its shared C10 references complete:
"${CPU[@]}" -m exp.offline_search.rounds.r15.descent.analyze apply --models groot
"${CPU[@]}" -m exp.offline_search.rounds.r15.descent.prepare emit-test-a --levels "$D/levels_groot.json" --uniform "$D/uniform_groot.json" --out "$D/test_a_groot_selected"
"${CPU[@]}" -m exp.offline_search.rounds.r15.descent.verify verify-test-a --out "$D/test_a_groot_selected"
"${CPU[@]}" -m exp.offline_search.rounds.r15.descent.verify plan-test-a --out "$D/test_a_groot_selected"
"${CPU[@]}" -m exp.offline_search.rounds.r15.descent.prepare install-test-a --out "$D/test_a_groot_selected"
# Coordinator queue: $D/test_a_groot_selected/queue_test_a.txt (six @local lines).

# Later full B apply checks both previously frozen model tables before writing:
"${CPU[@]}" -m exp.offline_search.rounds.r15.descent.analyze apply
```

Partial reports, analysis, provenance, CSVs and all tolerance tables carry `_pi05` / `_groot` suffixes; full filenames retain their original names. The full apply projects its tables onto each existing partial model and requires byte-identical primary and 0/2/5 pp levels/uniform files (eight files per model), recording hashes in `partial_equivalence.json`. Missing partial files or any byte difference abort before output writes. Reapplying a model also checks its existing tables. Installed model subsets already cover all cells, so a later full apply needs no second A installation.

Selection is isolated in `analyze_cell`: only that cell's arms and model/suite shared C10 enter the unchanged R14 first-failure point rule. Synthetic checks compare entire cell results and serialized tables between partial/full runs, change unrelated GR00T outcomes without changing π0.5, reject incomplete cells and altered partial files, and exercise both subset emit/verify/plan/install pipelines with generated templates/fits/store data and blocked transport/process/server operations. The fixed 200-comparison family is retained; partial cross-cell reporting is labelled partial and pooled summaries are deferred.

`PROTOCOL.md`, `protocol.json` and `PROTOCOL.sha256` remain unchanged. [partial_apply_adapter.json](partial_apply_adapter.json) binds the revised analyzer to the frozen original bytes in [partial_apply_original_analyze.py.txt](partial_apply_original_analyze.py.txt); protocol verification checks 15 numerical/accepted-row functions remain AST-identical and separately binds the authorized H100 served-fit loader edit below. The existing coordinator D5 track-B append is handled by preserving and verifying the exact frozen 1,849-byte D1-D4 prefix in [partial_apply_original_coordinator_decisions.md](partial_apply_original_coordinator_decisions.md). Later appended notes are provenance only. No real B apply, A-data read, installation into real trace roots or evaluation launch was performed for this change.

Validation: **17 synthetic CPU tests passed** in `test_partial_apply.py` and `test_analysis.py` with pytest cache disabled; [partial_apply_tests.xml](partial_apply_tests.xml) is the passing evidence. The default `analyze verify-protocol` also passed with zero outcome reads. Scratch for this change stayed below the allowed descent directory.

## H100 served-fit loader fix

The loader now accepts only the frozen source SHA or the deterministic H100 relocation of that exact source for the root's actual remote run path, and records the verified branch per arm. [LOADER_FIXES.md](LOADER_FIXES.md) explains the fix and preserved checks. Frozen protocol, selection rules, emitter and reporting code are unchanged. Only hash verification was run on real B data; no real `apply` was run.

Rerun π0.5 selection from the repository root:

```bash
taskset -c 2-9,46-53 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m exp.offline_search.rounds.r15.descent.analyze apply --models pi05
```

The repeatable hash-only audit command is:

```bash
taskset -c 2-9,46-53 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m exp.offline_search.rounds.r15.descent.verify_fit_hashes --models pi05 groot
```

Hash-only result: **64/64 frozen source fits match; all 61 arms with observed decision hashes match**. π0.5 is **32/32**, all using the deterministic H100-relocated branch. GR00T has **29 observed arms**, all matching the frozen source branch. Three GR00T arms had no decision hashes yet at their snapshots: `r15d_groot_spatial_200_off`, `r15d_groot_l10_500_off`, `r15d_groot_spatial_500_off`; their source fits are verified, and observed served hashes remain pending. Every observed arm has exactly one consistent hash. [served_fit_hash_verification.json](served_fit_hash_verification.json) records every source/served hash and shard snapshot; [served_fit_validation.json](served_fit_validation.json) records the validation. All **14 targeted synthetic tests** passed after the fresh-process fix. No journals, summaries, SR or test-A outcomes were read; no real apply was run.

## Per-model test-A measurement and combined summary

The coordinator reports all 64 B arms complete, a successful full B apply,
and passing `partial_equivalence.json` for both models. The test-A analyzer
now accepts each model's complete six-cell stage and checks its `input.json`
against exactly those cells in the full frozen B tables. All other loader,
chronology, measurement and statistical checks remain unchanged. Model files
carry `_pi05` or `_groot`; single-model pooled results are labelled partial.
[LOADER_FIXES.md](LOADER_FIXES.md) section 2 records this fix and its source proof.

Run from `/home/weiland/projects/openpi`:

```bash
CPU=(taskset -c 2-9,46-53 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python)
D=exp/offline_search/rounds/r15/descent

# Rerun the completed pi05 test-A stage now:
"${CPU[@]}" -m exp.offline_search.rounds.r15.descent.analyze_test_a apply --stage "$D/test_a_pi05_selected"
# Outputs: test_a_pi05.json, TEST_A_pi05.md, REPORT_pi05.md,
#          ANALYSIS_TEST_A_HANDBACK_pi05.md.

# Once GR00T test A completes:
"${CPU[@]}" -m exp.offline_search.rounds.r15.descent.analyze_test_a apply --stage "$D/test_a_groot_selected"
# Outputs: the corresponding four _groot files.

# Once both model stages are complete, produce the frozen all-cell summary:
"${CPU[@]}" -m exp.offline_search.rounds.r15.descent.analyze_test_a combine --stages "$D/test_a_pi05_selected" "$D/test_a_groot_selected"
# Outputs: test_a.json, TEST_A.md, REPORT.md, ANALYSIS_TEST_A_HANDBACK.md.
# Separate partial apply commands are optional prerequisites: combine re-audits
# the raw arms, retaining canonical order and the original pooled calculation.

# Safe source/protocol verification, with zero A outcome reads:
"${CPU[@]}" -m exp.offline_search.rounds.r15.descent.analyze_test_a verify-protocol
```

The combined result pools both models and suites separately for each library
size, using frozen model-specific cost coefficients and shared within-suite
bootstrap draws. Each size has 2,000 outcomes per arm and 1,000 original
suite/task/init clusters; shared C10 is never counted across sizes. The
legacy full-stage `apply --stage "$D/test_a_selected"` remains supported.

Only `analyze_test_a.py` changed among production code. Its local adapter
registration preserves the frozen protocol files and validates the exact
current source plus unchanged A loader and statistical-function ASTs. The
existing B analyzer is untouched. No real test-A apply or outcome read was
performed; the commands above are for the coordinator to run.

Validation: **23 synthetic CPU tests passed**, with an additional passing
pooled-equivalence run at the production **5,000 bootstrap draws**. Combined
and full-stage synthetic runs produced identical measurements and Markdown;
partial/full per-cell results also matched. Evidence:
[test_a_stage_tests.xml](test_a_stage_tests.xml),
[test_a_pooled_equivalence_tests.xml](test_a_pooled_equivalence_tests.xml), and
[test_a_stage_validation.json](test_a_stage_validation.json).
`analyze_test_a verify-protocol` passed with zero A outcome reads. Pytest cache
was disabled and all scratch/output writes stayed inside the descent directory.

The new source registration is local to the A analyzer. The unchanged B
verifier does not recognize this A source update, so rerunning B `analyze`
would fail its frozen A-source hash check. The commands above use the
completed, already-frozen B outputs; no B rerun is needed.

## A loader H100 served-path and hash fix

Both A served-value checks are fixed: startup fit paths and decision fit
hashes. The loader accepts only the exact frozen local path/source SHA or
the deterministic H100 path/relocated SHA of those exact frozen bytes for
the root's actual remote run. It records both branches per arm, including
shared C10 references. Local GR00T and locally served pi05 C10 references
retain the frozen-source branch. Every other loader statement and all
measurement/statistical functions remain unchanged; [LOADER_FIXES.md](LOADER_FIXES.md)
section 3 explains the checks and source proof.

The real audit also exposed two-arm sync pickle state: the second arm's hash
must be recomputed after the first sealed arm in a fresh CPU process. The
loader now replays the frozen plan order, checking every source against the
root seal. It never accepts a plan's claimed fit hash.

Rerun the completed pi05 A measurement from the repository root:

```bash
taskset -c 2-9,46-53 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m exp.offline_search.rounds.r15.descent.analyze_test_a apply --stage exp/offline_search/rounds/r15/descent/test_a_pi05_selected
```

Repeat the path/hash-only audit, without applying or reading outcomes:

```bash
taskset -c 2-9,46-53 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m exp.offline_search.rounds.r15.descent.verify_test_a_fit_hashes --models pi05 groot
```

The per-model GR00T and combined commands in the preceding section remain
valid after their A arms finish. No real test-A apply was run for this fix.

Real path/hash-only audit: **PASS, 28/28 frozen sources and 24/24 observed
arms**. All six pi05 roots passed (12/12 arms, deterministic H100 branch).
GR00T L10-100, Spatial-100, L10-200 and Spatial-200 passed (8/8 observed arms,
frozen-source branch). All four shared R14 C10 references passed using their
local source paths/hashes. Four GR00T size-500 arms had no observations yet:
`r15dt_groot_l10_500_uniform_off`, `r15dt_groot_l10_500_task_descent`,
`r15dt_groot_spatial_500_uniform_off`, `r15dt_groot_spatial_500_task_descent`.
Their frozen sources match; served-path/hash observation remains pending.
[test_a_served_fit_verification.json](test_a_served_fit_verification.json)
contains every verified path/hash, branch, replay order and shard snapshot.
This audit covers all observed roots, including any running snapshots, and
does not infer episode completion from outcomes. No journals, summaries, SR,
outcome tables or real A apply were read/run.

Validation: **37 synthetic CPU tests passed** in `test_test_a_served_fit.py`,
`test_test_a_stages.py` and `test_analysis.py`, including the production
5,000-draw combined/full equivalence. Tests cover both local models, H100
mirror/run/store paths, all three A arm types, ordered two-arm pickle replay,
unrelated paths/hashes, wrong remote roots, wrong source/hash branch pairs,
unsealed batch arms and a full audit with outcome-file reads blocked.
[test_a_served_fit_tests.xml](test_a_served_fit_tests.xml) and
[test_a_served_fit_validation.json](test_a_served_fit_validation.json) record
the passing evidence. Protocol verification passed with zero A outcome reads;
scratch/cache settings and every write stayed within the authorized directory.

## Model freeze timestamps for A measurement

Per-model A `apply` now uses `provenance_pi05.json` / `provenance_groot.json`
as the actual table freeze. `combine` checks each model against its own A
records. A later full B run requires a bound `partial_equivalence.json` PASS
for that model and matching primary table hashes. A partial freeze equal to
or later than any of its new-arm first accepted completions still fails.
The legacy full stage retains its full-provenance chronology rule.

The fix changes only the chronology source/check and its provenance evidence.
The complete A loader, served-fit verification, other existing helpers,
measurements and statistics are unchanged. [LOADER_FIXES.md](LOADER_FIXES.md)
section 4 describes the bindings and preserved ordering rule. The remaining
per-model/combined preconditions were reviewed with no additional scope gap
found; data-completeness, identity, cost and serving checks remain enforced.

Rerun from the repository root; provenance selection is automatic:

```bash
CPU=(taskset -c 2-9,46-53 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python)
D=exp/offline_search/rounds/r15/descent
"${CPU[@]}" -m exp.offline_search.rounds.r15.descent.analyze_test_a apply --stage "$D/test_a_pi05_selected"

# After all GR00T A arms complete:
"${CPU[@]}" -m exp.offline_search.rounds.r15.descent.analyze_test_a apply --stage "$D/test_a_groot_selected"
"${CPU[@]}" -m exp.offline_search.rounds.r15.descent.analyze_test_a combine --stages "$D/test_a_pi05_selected" "$D/test_a_groot_selected"

# Source/protocol verification only, with zero A outcome reads:
"${CPU[@]}" -m exp.offline_search.rounds.r15.descent.analyze_test_a verify-protocol
```

Real B metadata verification passed for both models: pi05 freeze
`2026-10-05T22:05:50.838973+00:00`, GR00T freeze
`2026-10-05T22:32:07.188449+00:00`, full reporting run
`2026-10-06T00:16:44.569313+00:00`. Both models' partial table hashes and
full equivalence bindings match. Evidence:
[test_a_freeze_metadata_verification.json](test_a_freeze_metadata_verification.json).
This check read no A outcomes or completion records. No real A apply was run.

Validation: **44 synthetic CPU tests passed** in `test_test_a_freeze_order.py`,
`test_test_a_stages.py` and `test_analysis.py`. Coverage includes early partial
freeze with later full B and equivalence; late/equal partial rejection;
missing/failed/wrong-model or mismatched equivalence; frozen table/provenance
bindings; every-arm strict comparisons; independent pi05/GR00T chronology in
combined mode; and unchanged legacy full-stage ordering. The existing
production 5,000-draw combined/full measurement equivalence also passed.
All 28 other existing functions/classes, including the A loader and H100
verifiers, remain AST-identical to the archived source before this fix.
[test_a_freeze_order_tests.xml](test_a_freeze_order_tests.xml) and
[test_a_freeze_order_validation.json](test_a_freeze_order_validation.json)
record the passing checks and exact source hashes. Frozen protocol verification
passed with zero A outcome reads; every write/scratch path stayed under descent.
