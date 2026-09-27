# K4 handback — evaluation infrastructure and low-IR frontier

Implemented in K4-owned files only. No timan107 access, GPU use, real servers, LIBERO workers, closed-loop chains, git commands, or reads of tests/review_tests. Historical R2/R3 roots were read-only. Shared code was developed under `dev/`, checked against `before/`, and installed with same-directory temporary files plus `os.replace`. Per-file times and SHA256 are in `results/install_times.json`; the final installation table is below.

## Switches and fields

- `emit_arms.py`: `yaml_patch` recursively merges dictionaries; lists/scalars/null replace. `{"miss":{"num_steps":2}}` works: when necessary the emitter supplies `miss.evidence_dir=<run-root>/evidence/<arm>` because CacheConfig requires it. A supplied evidence directory is retained. Trace removal and legacy YAML serialization are unchanged without a patch.
- `client_overrides: {"replan_steps":10}` (optional `resize_size`) goes into **both arms.json and the per-arm matrix**. Top-level `replan_steps:10` is an alias; conflicting values fail. L defaults to 5. The remote launcher translates matrix fields into actual run_gtp arguments; run_gtp itself does not interpret these matrix extensions. Direct overrides: `run_arm.sh ... --replan-steps 10` or `OSCL_REPLAN_STEPS=10`. For correct accounting, keep overrides in the arm spec rather than changing only the remote CLI/environment.
- `pure_inference:true` sets plugin mode, full-model startup, a plain B0 proposal, and `periodic:1` (including step 0). Do not add conflicting judge flags. All executed actions are policy actions. This still does the plugin's discarded retrieval proposal, as permitted by the task.
- `server_seed:N` is supported for pure-inference arms. It selects `exp.offline_search.rounds.r04.k4_eval.seeded_inference:SeededInference`; the chain passes `--os-seed N*65536+port`. This seeds Torch/Python/NumPy once before the first policy MISS, including when loading a prefit pickle. The server prints `OSCL_POLICY_SEED pid=... seed=...`. `seeds:[1001,2001]` expands distinct `<name>_s1001`/`<name>_s2001` rows. Do not combine with an explicit `--os-seed`. Seeds distinguish server processes within an arm and seed replicates; request scheduling can still change stochastic outcomes.
- `server_env:{...}` is forwarded by the chain. GR00T `yaml_patch.miss.num_steps` automatically supplies `GROOT_DENOISING_STEPS` and rejects a contradictory override. K3 owns the corresponding launcher support.
- Manifest selection: `OSCL_MANIFEST=/absolute/local/manifest.json` for `chain.sh`/`pilot.sh`, or a per-arm `manifest` field. Direct remote entry points accept `--manifest PATH`/`OSCL_MANIFEST`. C's `{strata,selected:[{task,init,stratum,inclusion_probability}]}` and simple lists of `[task_id,episode_idx]`/pair dictionaries work. Empty/invalid pairs and conflicting duplicate records fail. Identical duplicates count once. Model/suite metadata are checked when available.
- Manifest precedence is over Cartesian filters. Without a manifest, old `OSCL_EPISODES`/`OSCL_TASKS` behavior remains. The 50-trial pool attestation remains unchanged. The wrapper bypasses the base driver's non-pair-aware whole-arm resume filter only for manifests.
- Manifest EXPECT is the number of distinct selected pairs. `count.py` restricts completion to those pairs and the arm, using the original accepted-terminal/no-error rule. Markers are `<arm>.manifest_<canonical-pair-sha256>.DONE`; a manifest run never writes the legacy `<arm>.DONE`. A smaller manifest cannot skip a larger one, nor a subsequent full run. Legacy markers retain their old behavior. Chain snapshots to `runs/<arm>/manifest.json`, uploads that snapshot via `/tmp`, and records `runs/<arm>/selection.json` for later collection. Full runs clear the active selection metadata.
- Collection/KPI: `--manifest FILE` selects exact pairs and enables weighted outputs; persisted selection metadata or an arm's manifest is used otherwise. `--ledger` adds cost accounting to old logs. `cost_ledger:true` in an arm enables it even without new log fields. New R4 log fields also enable it automatically. `--cost-table FILE` explicitly overrides costs and enables the ledger. `collect --no-pull --no-write` is read-only.

## Cost and weighted-estimation outputs

Existing output fields remain. `cost_ledger` adds vision share `v`, MISS share `m`, K distribution/mean on MISS, L distribution/mean/default, stage modes, actual ledger inputs/reference cost/source, total cost, IR/request, **`ir_per_five_controls`**, controls and controls/episode, and logged stage-time totals/coverage/measured IR.

Cost is summed per request as `vision*s1[mode] + MISS*(miss_s1_extra[mode]+s2[mode]+s3[mode]*K/k_ref)` and divided by `sum(L)/5 * full_cost`. K3's installed `cost_table.json` is consumed, including GR00T `by_suite` and wrist MISS completion. Missing tables use owner `.152/.410/.438` for π0.5, and the explicitly labeled historical GR00T owner table (K8). Unknown stage modes receive the full reference price with a warning; they receive no invented saving.

Old logs retain their original owner IR, even with `--ledger`, unless explicitly repriced with `--cost-table`. All original SR/IR fields remain identical without new options. New R4 mixed IR fields use the R4 ledger and identify that basis. **K3's eager measurements and the historical compiled/graph shares are different cost bases:** for a single frontier, explicitly reprice historical arms using the same table and compare `cost_ledger.ir_per_five_controls`; do not mix bases silently.

Controls are nominal requests × L; a terminal request can execute a partial chunk. Actual controls are separately populated if journal `n_steps` is available. The current old journals lack it. Served-head logs override reconstructed HIT heads for action-repeat/gripper metrics; kernel diagnostics retain their explicitly approximate reconstruction. Old logs without served heads retain their old metrics.

`weighted` adds stratum population/sample/observed counts, inclusion probabilities, weighted SR or paired ΔSR, design variance/SE and normal 95% design interval. It uses `sum(y/pi)/N` and `sum((N_h/N)^2*(1-n_h/N_h)*sample_var_h/n_h)`. Paired differences are computed on the same `(task,init)` outcomes. Missing selected outcomes withhold the full-pool estimate; a non-census singleton withholds variance. A census has zero finite-population design variance. Simple pair lists without a probability design remain selectable but produce no fabricated weighted estimate. Original unweighted SR, McNemar, bootstrap and paired outputs remain descriptive outputs. Design intervals exclude rollout randomness, and pilot IR describes realized sampled requests, not a design-weighted population cost.

## Frontier and prefit

`arms_frontier.json` has **17 rows** with literal `<RUN>` fit placeholders and per-row `_prefit` notes:

- First batch (11): π0.5 l10/sp pure-inference seeds 1001 and 2001; l10 500 guard noprog 4 and periodic 8/12; spatial 500 guard-only and periodic 12; l10 50 periodic 6 and guard noprog 4.
- Longer chunks (4): l10 and spatial, L=10 with K10 and K2.
- K2-only (2): l10 and spatial, L=5, K2 pure inference.

All request K2's `--os-log-r4`; every row enables the ledger. Names are distinct and ≤40 characters. Pure policy controls have no executed-action library dependency and should be shown on both the 50/500-library panels.

Fits can be copied only when **method spec string, kwargs dictionary, and cell match exactly** (plugin enforces this). Periodic 500 AWM rows use kwargs `{}` and can reuse `r02_g500/fits/oscl500_p_<sp|l10>_cl2.pkl`; periodic 50 l10 uses `{"lib":"current","kref":5}` and can reuse `r02_g50/fits/oscl50_p_l10_cl2.pkl`. The new noprog-4 guard kwargs need new fits; do not relabel an R3 noprog-3 pickle. Spatial guard-only may reuse a matching existing spatial fit, never the l10 one. SeededInference has a different spec from plain B0: fit it once per cell, then reuse that pickle across seeds/K/L within the same cell. None of these baseline fits borrows a larger library for a 50-library controller. The manifest calibration's borrowed historical information is evaluation-only and is retained in the weighted output.

After substituting `<RUN>` in the JSON, emit normally. To create any missing artifact, use its emitted row's exact method/kwargs/cell:

```bash
# Coordinator: choose its own authorized CPU set and RUN. This was not run against a live run root by K4.
taskset -c <CPUS> env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=. .venv/bin/python -m exp.offline_search.closed_loop.plugin --os-root /home/weiland/trace_runs/offline_search_store --os-method '<EXACT EMITTED METHOD>' --os-kwargs '<EXACT EMITTED KWARGS JSON>' --os-cell <CELL> --os-log-dir <RUN>/fits --os-fit-artifact <RUN>/fits/<ARM>.pkl
```

No production fits were created by K4. `make_frontier.py` regenerates the reviewed spec without fitting or launching.

## Verification and exact commands

All commands below ran from `/home/weiland/projects/openpi`; all Python processes used the assigned affinity, BLAS=1, CUDA disabled. The shared `PYTHONPATH=.:src` prefix is written fully here. Tests use local fake launchers where indicated.

```bash
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r04/k4_eval/selftest_eval.py --installed
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python exp/offline_search/rounds/r04/k4_eval/validate_shell.py --installed
bash exp/offline_search/rounds/r04/k4_eval/run_plugin_selftests.sh
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python exp/offline_search/rounds/r04/k4_eval/historical_parity.py --installed
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m exp.offline_search.closed_loop.ops.emit_arms --run-root /tmp/k4_arm_validation --spec exp/offline_search/rounds/r04/k4_eval/arms_frontier.json
```

- `results/installed_selftest.log`: **11 tests PASS**. Covers all 24 supplied manifests; finite-population/paired arithmetic; partial designs and invalid probabilities; exact pair counting/deduplication; policy seeding in three CPU processes plus repeated-seed equality; YAML validation for both models/suites; nested patches and frontier emission; fallback and installed K3 stage prices; both-model × 50/500 synthetic end-to-end collection/KPI, episode boundaries, post-MISS regimes, retry deduplication and exact served-head metrics. Legacy emit YAML/matrix/arms.json bytes match the saved implementation.
- `results/installed_shell.log`: **six PASS groups**: marker selection; two stub server processes with distinct seeds/full-model/GR00T K2 environment; pilot dry-run legacy stdout parity and manifest EXPECT; byte-identical legacy remote argv; four remote launcher cases (default5/matrix10/CLI7/manifest); Bash syntax. The extracted helpers and recorder launchers did not start actual servers or a chain.
- `run_plugin_selftests.sh` contains the exact four prefixed selftest commands. All **four PASS**, two episodes each: π0.5 cache **41 decisions**, π0.5 all-MISS **38**, GR00T cache **38**, GR00T all-MISS **39**. Cache online/offline equality is 1.0 in every checked field; all-MISS replay and mini-store equality is 1.0, with **zero verdict/run violations**. Reports: `results/plugin_<model>_<cache|miss>.json`; logs alongside. Bulky temporary replay arrays are outside the repo; `results/plugin_artifact_root.txt` gives the final path.
- `results/historical_parity.json`: **70 arms, 28,804 completed episodes**, all collect fields equal to the saved pre-change implementation after excluding only collection timestamp. All **70 recorded SRs and 19 recorded mixed IRs** match. Full KPI field parity on **12 arms** (eight CL2 model/suite/library cells plus four mixed arms), excluding only parse time. This includes both models and both library scales.

Representative full historical results (each 500 episodes): π0.5 l10 g50 SR **.740**, IR **.3234160220826887**; g500 **.864 / .23821622358554875**; periodic5 **.792 / .31485839453678655**; spatial g50 **.888 / .2664492112511476**. CL2 SRs (50 → 500): π0.5-sp **.800 → .954**, π0.5-l10 **.630 → .768**, GR00T-sp **.888 → .966**, GR00T-l10 **.552 → .706**. No historical GR00T π0.5 proxy is presented as its actual measured IR.

Real retrospective manifest commands, with results in `results/pilot50.{json,md}` and `results/pilot500.{json,md}`:

```bash
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python -m exp.offline_search.closed_loop.ops.kpi --run-root /home/weiland/trace_runs/os_closed_loop/r03_mx --run-root /home/weiland/trace_runs/os_closed_loop/r02_g50 r3mx_p_l10_g --ref r02_g50:oscl50_p_l10_cl2 --manifest exp/offline_search/rounds/r04/ideation_C/pilot_manifest_pi05_l10_50_n200.json --store /home/weiland/trace_runs/offline_search_store --workers 1 --cache-dir exp/offline_search/rounds/r04/k4_eval/results/pause_cache --json exp/offline_search/rounds/r04/k4_eval/results/pilot50.json --md exp/offline_search/rounds/r04/k4_eval/results/pilot50.md --quiet
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python -m exp.offline_search.closed_loop.ops.kpi --run-root /home/weiland/trace_runs/os_closed_loop/r03_mx --run-root /home/weiland/trace_runs/os_closed_loop/r02_g500 r3mx_p_l10_g500 --ref r02_g500:oscl500_p_l10_cl2 --manifest exp/offline_search/rounds/r04/ideation_C/pilot_manifest_pi05_l10_500_n200.json --store /home/weiland/trace_runs/offline_search_store --workers 1 --cache-dir exp/offline_search/rounds/r04/k4_eval/results/pause_cache --json exp/offline_search/rounds/r04/k4_eval/results/pilot500.json --md exp/offline_search/rounds/r04/k4_eval/results/pilot500.md --quiet
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python exp/offline_search/rounds/r04/k4_eval/validate_reports.py
```

Both select exactly **200 pairs**. At size 50: raw SR .755, weighted SR **.7444717504717504**, weighted ΔSR **.11447175047175047**, design variance **.0003806977176201285**, 95% **[.07622995912550252, .1527135418179984]**. At size 500: raw SR .865, weighted SR **.8611627196333079**, weighted ΔSR **.09316271963330787**, variance **.00022780218563124238**, 95% **[.0635807535835921, .12274468568302364]**. Collect/KPI weighted and ledger dictionaries agree exactly; SHA256 of original summaries stayed unchanged (`results/pilot_validation.json`). These are checks on saved outcomes, not new prospective results.

## Coordinator next steps and unverified scope

1. Substitute `<RUN>` in `arms_frontier.json`, emit to the intended run root, and prepare exact-match fits as above. Check the existing R4 first-batch names before choosing what still needs running; this spec deliberately uses `r4f_*` names.
2. Push **all three** `ops/remote/{run_arm.sh,run_gtp_subset.py,count.py}` plus emitted arm YAML/matrices using the coordinator's normal sync procedure. K4 did not push anything. Chain uploads the chosen manifest itself at launch; direct remote invocations need the manifest staged there first.
3. Coordinator runs the real timan107 smoke: exact non-Cartesian pair selection, L=10 forwarding, pure MISS verdicts, seed startup records, then resume a larger manifest. These remote/simulator/GPU behaviors are **unverified by K4**; only CPU integration and local recorder tests ran here. Actual K2/L10 SR is also unmeasured here.
4. Use weighted pilot outputs for collapse screening and complete 500 paired episodes for decisions, as SELECTION requires. Use a consistent cost basis for frontier plots. R4 policy seed controls are repeatable RNG initializations, not a promise of identical concurrent rollouts.

K4 did not implement the optional late-round randomized CALL/CACHE controller: it was not one of the four requested deliverables in this task. All four requested deliverables above are implemented and locally checked.

## Final file installation inventory

Times are UTC; each runtime replacement was atomic. SHA256 values are retained in `results/install_times.json`.

| Runtime file | Last atomic install (UTC) |
|---|---|
| `exp/offline_search/closed_loop/ops/chain.sh` | 2026-09-27T19:20:17.955375+00:00 |
| `exp/offline_search/closed_loop/ops/collect.py` | 2026-09-27T19:27:37.629805+00:00 |
| `exp/offline_search/closed_loop/ops/emit_arms.py` | 2026-09-27T19:32:54.007800+00:00 |
| `exp/offline_search/closed_loop/ops/kpi.py` | 2026-09-27T19:27:37.632418+00:00 |
| `exp/offline_search/closed_loop/ops/pilot.sh` | 2026-09-27T19:20:17.957266+00:00 |
| `exp/offline_search/closed_loop/ops/remote/count.py` | 2026-09-27T19:20:17.963801+00:00 |
| `exp/offline_search/closed_loop/ops/remote/run_arm.sh` | 2026-09-27T19:20:17.959148+00:00 |
| `exp/offline_search/closed_loop/ops/remote/run_gtp_subset.py` | 2026-09-27T19:20:17.961555+00:00 |
| `exp/offline_search/rounds/r04/k4_eval/cost_ledger.py` | 2026-09-27T19:32:54.010383+00:00 |
| `exp/offline_search/rounds/r04/k4_eval/estimators.py` | 2026-09-27T19:32:54.011984+00:00 |
| `exp/offline_search/rounds/r04/k4_eval/seeded_inference.py` | 2026-09-27T19:32:54.013558+00:00 |

Added deliverables and verification tools (final file mtime, UTC):

| File under k4_eval/ | Modified (UTC) |
|---|---|
| `arms_frontier.json` | 2026-09-27T19:19:03.109839+00:00 |
| `make_frontier.py` | 2026-09-27T19:17:49.621805+00:00 |
| `selftest_eval.py` | 2026-09-27T19:21:57.426918+00:00 |
| `historical_parity.py` | 2026-09-27T19:15:58.104755+00:00 |
| `validate_shell.py` | 2026-09-27T19:31:10.653166+00:00 |
| `validate_reports.py` | 2026-09-27T19:25:09.987005+00:00 |
| `run_plugin_selftests.sh` | 2026-09-27T19:28:26.941093+00:00 |
| `install.py` | 2026-09-27T19:27:37.485070+00:00 |
| `INTEGRATION.md` | 2026-09-27T19:32:53.944212+00:00 |

`before/` holds unchanged original copies; `dev/` holds final development copies. `results/` contains the JSON/Markdown/log evidence listed above, installation hashes, and the small pause-mask cache. No production data was written into historical run roots.

Final installed historical-parity run: 202.607144 seconds; all checks passed.
