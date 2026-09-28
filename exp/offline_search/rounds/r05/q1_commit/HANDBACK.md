# Q1 hand-back: C10 plan completion and D1 grasp inspection

Final audit UTC: 2026-09-28T02:39:17.269747+00:00. All numbers below come from completed CPU checks on the final files. No server, port,
GPU, simulator, LIBERO worker, chain, remote host, git command, review_tests read or subagent was used.
Only `exp/offline_search/rounds/r05/q1_commit/**` and `/tmp/q1_*` were written. All Python commands used CPUs
26–29,70–73, OMP/OpenBLAS/MKL threads 1, CUDA disabled, and bytecode writes disabled.

## Implementation and switches

`exp.offline_search.rounds.r05.q1_commit.judge:CommitJudge` subclasses K10 `PolicyTailJudge`.
`policy_tail_gate="inherited"` (default), `monitor="off"` delegates the policy-tail hook directly to K10;
ordinary query and HIT blind handling are inherited. `policy_tail_gate="lifecycle"` consumes the separate
one-use vision anchor, requires the immediately preceding real-vision MISS in the same episode/task, dense
history and finite state/chunk, and returns the original policy chunk shifted by five. It bypasses rejected-cache
budget, span and base gates. It does not call the inherited blind gate, fabricate HIT histories, or clear progress
memos. K7's vision-confirmed stuck and span guards still run at the next real anchor; the tail remains a blind gap.
Tail rows/weights retain the last cache proposal as diagnostic provenance; `src=policy_tail` and the saved policy
chunk identify the executed action source.

Enable C10 with `--os-blind --os-policy-tail --os-judge guard_only --os-no-shadow-native` and the lifecycle kwarg.
The plugin enforces execution count, original normalized/wire-chunk equality, identity, repeated last decision IDs and one
complete tail. An absent `executed_steps` audit follows the existing five-control client contract; actual unreported
physical execution cannot be inferred. Explicit 0/4/6/10 controls reject reuse. Terminal MISSes cannot cross reset.

Optional `monitor="loeo_xyz99"` requires lifecycle mode. Each deployed library independently fits the ideation-A
16-feature xyz ridge, episode-held-out p99, and final float32 B/scales (208 parameter bytes). It predicts from the
actual policy head and prior state, never the rejected cache action. Residual above p99 requests LookReason 5,
without independently forcing a policy call. Defaults and all six requested arms keep the monitor off.

| Suite | Library episodes | Own-library LOEO p99 |
|---|---:|---:|
| l10 | 50 | 0.991127610206604 |
| l10 | 500 | 1.00471591949463 |
| spatial | 50 | 0.515817582607269 |
| spatial | 500 | 0.51978200674057 |

`exp.offline_search.rounds.r05.q1_commit.judge:GraspCheckJudge` uses K7 anchor_tail / budget 1 / budget_only.
On an otherwise eligible blind HIT it applies D's command-matched aperture rule with exact strict/inclusive
thresholds: two closed executed heads; observed normalized aperture <.05; matched original mass >=.5; matched
successor mean >.25 and mass above .25 >=.75. Quantiles and compact tables use only the deployed library.
An alarm returns `LookReason(9,"grasp_aperture_contradiction")`; the associated vision query sets
`os_force_miss=1`, `os_reason=9`, `grasp_check=1`, and retains the original guard reason as `grasp_base_reason`.
The allowance is marked consumed only when a later dense history confirms this issued query executed as a
real-vision MISS. Repeated or abandoned proposals do not consume it. Episode/task reset clears pending and consumed
state. The existing per-connection lock protects these fields; fitted arrays are treated as read-only and shared by cloning.
D1 clearly refuses GR00T before fitting. Enable with `--os-blind --os-judge guard_only --os-no-shadow-native`,
without `--os-policy-tail`.

## Final verification

Compact evidence: [results/final_summary.json](results/final_summary.json). Full logs/NPZs are at its exact
`evidence_paths` and in `/tmp/q1_final_matrix`, `/tmp/q1_final_concurrency`, `/tmp/q1_final_monitor`,
`/tmp/q1_edges_final`, `/tmp/q1_grasp_plugin_final`, and `/tmp/q1_reg_*`.

- Method streams: 368 bit-exact K10/inherited vision results,
  368 ordinary blind-result comparisons and
  368 inherited policy-hook comparisons, over both suites/scales.
  Lifecycle served **368/368** eligible MISS tails,
  with all 32 columns of `policy_chunk[5:10]` byte-exact even with positive no-progress span, zero cache budget and
  active base gates. 60 lifecycle rejection/reset cases passed.
- Real fitted next-anchor guard check: `stuck_n=2`, `noprog_span=2`, flags `9`, `os_force_miss=1`; blind keys stayed
  NaN and actual history stayed `[MISS,HIT]`. Existing K7 parity/edge/rate checks below also passed.
- D1 historical helper and method eligible-blind replay: **223/223** alarms at 50 and **47/47** at 500, zero
  disagreements across 31,186 / 28,888 recorded decisions. There are 12,620 / 12,341 blind rows (12,120 / 11,841
  with step>=2). With execution-confirmed once-per-episode caps: **73 / 24** interventions. Repeated vision
  proposals were checked 146 / 48 times. Historical JSONL lacks full camera keys, so the pending-MISS unit test
  mocks new vision retrieval. The capped unit replay injects a committed vision/MISS history marker at the first
  alarm to test allowance consumption while retaining recorded heads/states; it does not measure a new policy
  execution. Real library tables, recorded heads/states/rows/weights and inherited K7 blind selection determine
  the alarm. The plugin fixture separately commits actual fake-policy MISS responses. These are fixed-history
  replays and synthetic execution checks, not counterfactual rollouts.
- New installed plugin matrix: **14 runs / 672 decisions**, all passed with
  guard_only; four inherited-vs-K10 pairs match every non-timing NPZ field exactly. Each run rejects four duplicates.
- C10 transformed-action/lifecycle integration: **116 decisions,
  46 tails**, 21 checks, eight connections, **400** real-transform L10-equivalent
  controls byte-exact. Includes partial execution, step 0, task/reset, final MISS, duplicate preservation,
  wrong-action substitution, finite state, burst/periodic precedence and state-dependent output transforms.
  Its 46 tail ledger rows have vision=0, MISS=0, cost=0, IR5=0.
- Six-arm eight-connection replay: **1098 threaded + the same serialized
  decisions**, exact actions/verdicts/history/log fields under the recorded reservation schedule. Every eligible C10
  MISS served its exact tail. All configurations reached eight simultaneous fake-policy calls; this is a correctness
  stress test, not a hardware throughput claim.
- D1 synthetic-contact plugin stress: **680 decisions / 8 connections / 24
  episodes**, exactly 16 interventions in 16 eligible episodes, 16 duplicate rejections, no second intervention,
  and resets after a final forced MISS. Synthetic successor apertures and current aperture deliberately guarantee
  exposure after actual closed heads; the historical test above separately validates real-table alarm agreement.
- Optional-monitor own-library calibration matched ideation A at all four cells/scales. Two actual monitor prefits
  and installed plugin runs at l10 50/500 passed (96 decisions), plus
  controlled zero-residual and over-threshold policy-head tests. Extra verification fits are named `_monitor_final.pkl`
  in `/tmp/q1_fits`; they are not requested arms.
- Existing K2: 21 runs / 963 decisions; K1: 10 /
  480; K4: 4 / 156. Six old flag-off fixed-clock/PID modes remained
  byte-identical. K5 overlay, four 342-decision randomized replays, 200 planted + 200 null estimator datasets,
  log-forgery/estimator and ledger checks passed. K6: 12 eight-connection configurations, 2,196 threaded + the same
  serialized decisions; 23-decision / 25-reservation edge test and 64 quantile transactions passed.
- Existing K7: five plugin arms / 240 decisions; 15,272 scalar query comparisons and
  189,904 full-cell diagnostic decisions; original edges passed. Existing K10: nine plugin runs /
  432 decisions and
  118 tails; nine eight-connection configurations /
  1647 threaded decisions with exact serialized parity;
  original policy-tail edge/transform/ledger and K10-vs-K7 method lifecycle/parity checks passed.
- Six arm emit/parser/CacheConfig checks passed: full model, ordinary K=10 MISS, L=5 and explicit cost ledger.

Final production-fit plugin selftests (48 decisions each):

| Arm | Decisions | Vision | MISS | Policy tails | Eligible C10 MISSes with follow-up |
|---|---:|---:|---:|---:|---:|
| `r5q1_c10_p_l10_50` | 48 | 24 | 2 | 2 | 2 |
| `r5q1_c10_p_l10_500` | 48 | 24 | 2 | 2 | 2 |
| `r5q1_c10_p_sp_50` | 48 | 24 | 0 | 0 | 0 |
| `r5q1_c10_p_sp_500` | 48 | 24 | 0 | 0 | 0 |
| `r5q1_d1_p_l10_50` | 48 | 26 | 4 | 0 | n/a |
| `r5q1_d1_p_l10_500` | 48 | 26 | 2 | 0 | n/a |

Final production-fit concurrency, per threaded side (serialized side exactly matches):

| Arm | Decisions | Vision | MISS | Policy tails | Eligible C10 policy tails |
|---|---:|---:|---:|---:|---:|
| `r5q1_c10_p_l10_50` | 183 | 93 | 5 | 5 | 5 |
| `r5q1_c10_p_l10_500` | 183 | 93 | 4 | 4 | 4 |
| `r5q1_c10_p_sp_50` | 183 | 93 | 4 | 4 | 4 |
| `r5q1_c10_p_sp_500` | 183 | 93 | 8 | 7 | 7 |
| `r5q1_d1_p_l10_50` | 183 | 97 | 7 | 0 | n/a |
| `r5q1_d1_p_l10_500` | 183 | 96 | 4 | 0 | n/a |

## Fits, bytes and exact commands

[arms_q1.json](arms_q1.json) has four C10 lifecycle arms (pi05 l10/spatial x 50/500) and two D1 arms (l10 x 50/500),
literal `<RUN>` fit placeholders, full model, five-control clients, full K=10 MISS, `cost_ledger: true`, and the
requested flags/root. Library 50 uses current/kref 5; 500 uses big/kref 8 (resolved bpool_cs). All requested prefits
were run afresh with the final installed plugin into `/tmp/q1_fits/`; none is a relabeled K7 pickle.

| Fit basename (under `/tmp/q1_fits/`, suffix `.pkl`) | Bytes | SHA256 |
|---|---:|---|
| `r5q1_c10_p_l10_50` | 32,704,614 | `16ddfec3dc8d3af95c1a39c1f483475746e3ee8194e0730ef38a7f043b77c69a` |
| `r5q1_c10_p_l10_500` | 142,348,649 | `186f635e234a229f7bbbe1fb93b7ed7c2a9d43581d138294aa2e978e5957cb9d` |
| `r5q1_c10_p_sp_50` | 26,076,609 | `6cadcc8f75a127601ccf8bca3520d46eedd766c80d6b10a351e2fd5195f2654f` |
| `r5q1_c10_p_sp_500` | 66,500,639 | `3ef80beaf240680327e75c07c2f067fd7e79f50b656e71cae3af04c1ccfbc315` |
| `r5q1_d1_p_l10_50` | 32,728,495 | `5b9cc91612d33cb7cc6ef7af5b8fcdc3dab790b85d4eff11d57aea8df8ae57da` |
| `r5q1_d1_p_l10_500` | 142,614,036 | `acef251e3e3706a395c16acf30f44e0be586a5755d8488dbf6a82891d9b8393f` |

Raw library disk bytes count all regular files below the selected store-library directory; these include the
offline representation and are not deployment-pickle bytes. Owner deployed-pkl reference MB is borrowed from the
binding findings, not remeasured. No big-library table or monitor information enters a 50-library fit. Spatial's
nominal 50 library contains 49 episodes. D1 adds 23,768 / 265,256 compact table bytes (including eight quantile
bytes), excluding Python/pickle overhead; actual serialized fit bytes are above.

| Arm | Library rows | Raw library disk bytes | Q1 fit bytes | Owner deployed pkl MB |
|---|---:|---:|---:|---:|
| `r5q1_c10_p_l10_50` | 2,640 | 7,027,274,072 | 32,704,614 | 1103 |
| `r5q1_c10_p_l10_500` | 29,472 | 78,449,234,637 | 142,348,649 | 1103 |
| `r5q1_c10_p_sp_50` | 1,018 | 2,709,781,159 | 26,076,609 | 431 |
| `r5q1_c10_p_sp_500` | 10,909 | 29,037,971,827 | 66,500,639 | 431 |
| `r5q1_d1_p_l10_50` | 2,640 | 7,027,274,072 | 32,728,495 | 1103 |
| `r5q1_d1_p_l10_500` | 29,472 | 78,449,234,637 | 142,614,036 | 1103 |

The exact six executed commands (repository root):

```bash
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.closed_loop.plugin --os-method exp.offline_search.rounds.r05.q1_commit.judge:CommitJudge --os-kwargs '{"base_kwargs":{"lib":"current","kref":5,"serving":"anchor_tail","budget":1,"gates":"budget_only"},"progress_guard":"noprog_span","events":"none","stuck_guard":"vision_confirmed","policy_tail_gate":"lifecycle","monitor":"off"}' --os-cell pi05_l10_cache --os-log-dir /tmp/q1_prefit_logs/r5q1_c10_p_l10_50 --os-tag r5q1_c10_p_l10_50 --os-root /home/weiland/trace_runs/offline_search_store --os-blind --os-policy-tail --os-judge guard_only --os-no-shadow-native --os-fit-artifact /tmp/q1_fits/r5q1_c10_p_l10_50.pkl
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.closed_loop.plugin --os-method exp.offline_search.rounds.r05.q1_commit.judge:CommitJudge --os-kwargs '{"base_kwargs":{"lib":"big","kref":8,"serving":"anchor_tail","budget":1,"gates":"budget_only"},"progress_guard":"noprog_span","events":"none","stuck_guard":"vision_confirmed","policy_tail_gate":"lifecycle","monitor":"off"}' --os-cell pi05_l10_cache --os-log-dir /tmp/q1_prefit_logs/r5q1_c10_p_l10_500 --os-tag r5q1_c10_p_l10_500 --os-root /home/weiland/trace_runs/offline_search_store --os-blind --os-policy-tail --os-judge guard_only --os-no-shadow-native --os-fit-artifact /tmp/q1_fits/r5q1_c10_p_l10_500.pkl
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.closed_loop.plugin --os-method exp.offline_search.rounds.r05.q1_commit.judge:CommitJudge --os-kwargs '{"base_kwargs":{"lib":"current","kref":5,"serving":"anchor_tail","budget":1,"gates":"budget_only"},"progress_guard":"noprog_span","events":"none","stuck_guard":"vision_confirmed","policy_tail_gate":"lifecycle","monitor":"off"}' --os-cell pi05_spatial_cache --os-log-dir /tmp/q1_prefit_logs/r5q1_c10_p_sp_50 --os-tag r5q1_c10_p_sp_50 --os-root /home/weiland/trace_runs/offline_search_store --os-blind --os-policy-tail --os-judge guard_only --os-no-shadow-native --os-fit-artifact /tmp/q1_fits/r5q1_c10_p_sp_50.pkl
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.closed_loop.plugin --os-method exp.offline_search.rounds.r05.q1_commit.judge:CommitJudge --os-kwargs '{"base_kwargs":{"lib":"big","kref":8,"serving":"anchor_tail","budget":1,"gates":"budget_only"},"progress_guard":"noprog_span","events":"none","stuck_guard":"vision_confirmed","policy_tail_gate":"lifecycle","monitor":"off"}' --os-cell pi05_spatial_cache --os-log-dir /tmp/q1_prefit_logs/r5q1_c10_p_sp_500 --os-tag r5q1_c10_p_sp_500 --os-root /home/weiland/trace_runs/offline_search_store --os-blind --os-policy-tail --os-judge guard_only --os-no-shadow-native --os-fit-artifact /tmp/q1_fits/r5q1_c10_p_sp_500.pkl
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.closed_loop.plugin --os-method exp.offline_search.rounds.r05.q1_commit.judge:GraspCheckJudge --os-kwargs '{"base_kwargs":{"lib":"current","kref":5,"serving":"anchor_tail","budget":1,"gates":"budget_only"},"progress_guard":"noprog_span","events":"none","stuck_guard":"vision_confirmed"}' --os-cell pi05_l10_cache --os-log-dir /tmp/q1_prefit_logs/r5q1_d1_p_l10_50 --os-tag r5q1_d1_p_l10_50 --os-root /home/weiland/trace_runs/offline_search_store --os-blind --os-judge guard_only --os-no-shadow-native --os-fit-artifact /tmp/q1_fits/r5q1_d1_p_l10_50.pkl
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.closed_loop.plugin --os-method exp.offline_search.rounds.r05.q1_commit.judge:GraspCheckJudge --os-kwargs '{"base_kwargs":{"lib":"big","kref":8,"serving":"anchor_tail","budget":1,"gates":"budget_only"},"progress_guard":"noprog_span","events":"none","stuck_guard":"vision_confirmed"}' --os-cell pi05_l10_cache --os-log-dir /tmp/q1_prefit_logs/r5q1_d1_p_l10_500 --os-tag r5q1_d1_p_l10_500 --os-root /home/weiland/trace_runs/offline_search_store --os-blind --os-judge guard_only --os-no-shadow-native --os-fit-artifact /tmp/q1_fits/r5q1_d1_p_l10_500.pkl
```

[prefit_commands.json](prefit_commands.json) additionally contains the exact `<RUN>/fits` / `<RUN>/prefit_logs`
template per arm. [prefit.sh](prefit.sh) runs the six `/tmp/q1_fits` commands. Both use the required CPU/env prefix.

## Reproduction and installation provenance

Final verification commands from `/home/weiland/projects/openpi`:

```bash
P=(taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python)
B=exp/offline_search/rounds/r05/q1_commit
"${P[@]}" "$B/prepare.py"
"${P[@]}" "$B/prepare_checks.py"
"${P[@]}" "$B/prepare_integration.py"
bash "$B/prefit.sh" > /tmp/q1_prefits_final.log 2>&1
bash "$B/run_regression.sh" > /tmp/q1_regression.log 2>&1
"${P[@]}" "$B/regression/method_test.py" > /tmp/q1_k10_method.log 2>&1
"${P[@]}" "$B/policy_tail_test.py" --source installed --out /tmp/q1_edges_final > /tmp/q1_edges_final.log 2>&1
"${P[@]}" "$B/grasp_plugin_test.py" --out /tmp/q1_grasp_plugin_final > /tmp/q1_grasp_plugin_final.log 2>&1
bash "$B/run_new_checks.sh"
"${P[@]}" "$B/finish.py"
```

The new-check wrapper runs method tests, delivery validation, the 14-run plugin matrix, six-arm concurrency and
two monitor fits/replays. Expanded per-job commands are in the matrix/monitor reports and relocated regression
scripts. Choose fresh `/tmp/q1_*` roots before rerunning: test logs append and concurrency fixtures refuse existing
directories. Run the parallel new-check wrapper only after the regression K2 multi-process shell matrix finishes;
thereafter six new-check driver/worker processes plus two regression processes stay within the eight-process cap.

The shared plugin/selftest/verifier changed during parallel Q2 work at 2026-09-28 01:36 UTC. Interrupted earlier
evidence was archived under `regression/results_beforeq2/` and `/tmp/q1_beforeq2_*`; none supplies the final counts.
The full final suite was rerun against the updated installed imports. The final audit compares all five shared
source files with its captured pre-check snapshot and records their hashes in results/final_summary.json.
No shared file was edited by Q1. New method files were developed only inside the owned Q1 directory.

Initial fixture-only failures were fixed before final reruns: K10's hook needs a dataclass facade; NPZ historical
arrays must be materialized once; wall-time fields cannot enter exact replay equality; a committed plugin decision
clears `_dec`; and the synthetic aperture must match the fake key builder. No failed run is counted as final evidence.

All added top-level source/spec files with final availability times and SHA256:

| File | UTC | SHA256 |
|---|---|---|
| `arms_q1.json` | 2026-09-28T01:24:48.294924+00:00 | `718ea70015d331f6e1f61cff77c0411e3eb4b93c413a18220767f887c8be0cfc` |
| `concurrency_test.py` | 2026-09-28T01:33:31.644467+00:00 | `f54f6bbd084dda349d94990e14c39446b9842f7f916c91636456c08f78f5da53` |
| `finish.py` | 2026-09-28T02:32:47.912409+00:00 | `ea8a2eb9e4a0260007ec2d91a5ee142f89afcf17218cbe5b5fd396196a3c6b92` |
| `grasp_plugin_test.py` | 2026-09-28T01:35:59.521618+00:00 | `792f0684bcc5db85222e4232de852c39477d1478c590e94d758a160cabfb9a36` |
| `judge.py` | 2026-09-28T01:32:19.155392+00:00 | `8edc216b317d04465c4f383673b274e75673f0974559ef0dc849b07a26410925` |
| `monitor_fit_test.py` | 2026-09-28T01:44:18.888126+00:00 | `5f7b664e24d7f9d4cdeeb480684e0fadb69629f8ab979acb323db54f10653be0` |
| `plugin_matrix.py` | 2026-09-28T01:32:19.215393+00:00 | `b0f5d2f8d7fc244bf8d8a7e5bdd1aed289b43a0503e78cdedf5c7afd56b00a11` |
| `policy_tail_test.py` | 2026-09-28T01:33:31.644467+00:00 | `44e5c67df6e96f639bffd7201fcf6fbff37c053ee60627a9327103da301f3112` |
| `prefit.sh` | 2026-09-28T01:24:48.295924+00:00 | `f784c7aaddd66d127391ba1eccfd30dd7c0fb87f648a6f70431a9ed32dc1a8e6` |
| `prefit_commands.json` | 2026-09-28T01:24:48.295924+00:00 | `7078a68772fe159531183eddf5e1ac365c9ac89ea098949ae39d0867de597d7b` |
| `prepare.py` | 2026-09-28T01:24:48.154924+00:00 | `7ac8577d4b5997fe5017a77e8e0520c1e6f3ab1777a60e0075afa17e0a458ff4` |
| `prepare_checks.py` | 2026-09-28T01:41:50.557976+00:00 | `9045a42ae70dec910661882e5e65f8ec121119e719dee3ad38c1dc2d830201ee` |
| `prepare_integration.py` | 2026-09-28T01:33:31.502467+00:00 | `7f25eccf29479fdca8587d94772ee6f5ed6015e69987e87c08b24fd15a88b08c` |
| `run_new_checks.sh` | 2026-09-28T01:44:18.829126+00:00 | `87a258c2c313a74052deea7074b5f34fb3d05a6620df75ac17c6765576b18923` |
| `run_regression.sh` | 2026-09-28T01:41:28.969954+00:00 | `afb97be324116cd8e711021793e9a997d25a29cf763580475d57de7cd37beab0` |
| `test_methods.py` | 2026-09-28T01:37:46.752728+00:00 | `d71f7b3d298ab181142278074629ee64362bbdad2e237cc186bd94fe5a6fc9cf` |
| `validate_delivery.py` | 2026-09-28T01:36:57.873678+00:00 | `f95cc82409b38b311dbdc547cbfec034595b2e22288589e0c44c080964460fb7` |

`results/file_inventory.json` contains the complete owned-file inventory, including relocated recipes and evidence
(excluding the inventory and self-referential hand-back files). `HANDBACK.sha256` hashes this hand-back.

## Coordinator smoke and remaining limits

Coordinator-only recipe; Q1 did not execute it. Choose a new run root and coordinator-authorized GPU/ports/CPUs.
Copy the six arm fits with verified hashes, resolve placeholders, and emit the ordinary arm YAMLs:

```bash
RUN=/home/weiland/trace_runs/os_closed_loop/r05_q1_smoke
Q=exp/offline_search/rounds/r05/q1_commit
mkdir -p "$RUN/fits"
for arm in r5q1_c10_p_l10_50 r5q1_c10_p_l10_500 r5q1_c10_p_sp_50 r5q1_c10_p_sp_500 r5q1_d1_p_l10_50 r5q1_d1_p_l10_500; do
  cp "/tmp/q1_fits/$arm.pkl" "$RUN/fits/$arm.pkl"
done
sed "s|<RUN>|$RUN|g" "$Q/arms_q1.json" > "$RUN/arms_q1_resolved.json"
"${P[@]}" -m exp.offline_search.closed_loop.ops.emit_arms --run-root "$RUN" --spec "$RUN/arms_q1_resolved.json"
# Coordinator supplies PORTS and SERVER_CPUS already authorized for its live work.
bash exp/offline_search/closed_loop/ops/sync_remote.sh "$RUN" r5q1_c10_p_l10_50 r5q1_d1_p_l10_50
unset OSCL_MANIFEST
PORTS="$PORTS" SERVER_CPUS="$SERVER_CPUS" OSCL_TASKS=0,1 OSCL_EPISODES=0,1 WPS=2 bash exp/offline_search/closed_loop/ops/chain.sh "$RUN" r5q1_c10_p_l10_50 r5q1_d1_p_l10_50
```

Use a separate full-run root after the smoke. Retain the full model, K=10 MISS and L=5 client, import the Q1 module
and current plugin dependencies, and preserve exact method/kwargs/cell artifact metadata. Audit policy-tail source,
zero tail vision/MISS cost, anchor guards, D1 reason 9 and <=1 D1 intervention/episode. Vision costs .152, each
full MISS adds .848, and each blind/policy-tail decision costs zero on the owner basis. D1's forced vision MISS
costs 1.0 at that decision; no fixed-path calculation here establishes realized rollout savings or success.

Unverified: real GPU policy inference, websocket/simulator integration, LIBERO closed-loop success/cost effects and
physical grasp recovery. Fake-policy controls and historical fixed observations establish implementation behavior,
not causal SR improvements. Endpoint-confirmed K7 guards retain their documented blind-gap limitations; the optional
xyz monitor observes arm displacement, not object contact or task success. No new policy, representation, borrowed
big-library calibration, or external model was introduced.
