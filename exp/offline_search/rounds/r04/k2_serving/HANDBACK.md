# K2 serving hand-back

Completed 2026-09-27. All work stayed in K2-owned files, plus the permitted atomic README append. No git,
remote host, simulator worker, or closed-loop chain was used. Every Python process used CPUs 22-25,66-69
with OMP/OPENBLAS/MKL threads set to one. No GPU OOM occurred; both smoke servers have been stopped.

## Installed files

| File | Latest atomic install/append, UTC |
|---|---|
| `closed_loop/blind.py` | 2026-09-27T19:01:43Z |
| `closed_loop/plugin.py` | 2026-09-27T19:23:51Z |
| `closed_loop/probe.py` | 2026-09-27T19:19:07Z |
| `closed_loop/selftest.py` | 2026-09-27T19:31:07Z |
| `closed_loop/verify_logs.py` | 2026-09-27T19:31:07Z |
| `closed_loop/replay_client.py` | 2026-09-27T19:21:04Z |
| `closed_loop/README.md` | 2026-09-27T19:35:55Z |

`blind.py` was published first for K1. `plugin.py` implements the early bypass, model adapters, once-only
orchestrator helper, dense histories, guards, and opt-in R4 logging. `probe.py` adds `ProbeBlind`.
`selftest.py`, `verify_logs.py`, and `replay_client.py` cover CPU integration, offline method replay, and
wire-output parity. Existing fields are retained. Full installation history is `install_times.txt`;
installed SHA256 values are in `results/validation_summary.json`.

Additional owned deliverables: `dev/` and `baseline/` development copies; `run_checks.sh`,
`legacy_parity.py`, `check_log_only.py`, `summarize.py`, `groot_smoke_server.py`, `INTEGRATION.md`, and
`results/` with commands, logs, arrays, and JSON reports. Small editing/reproduction scripts are retained.

## Exact enablement and contract

- `--os-blind`: enable blind serving **and** R4 fields. Methods without `blind_step` always request vision.
- `--os-log-r4`: R4 fields without blind serving. This preserves the legacy periodic clock.
- `--os-log-inputs`: retain dense masks, normalized executed chunks, and actual wire outputs for verification.
- No new client flag is needed for ordinary serving. Fresh images/state/prompt still arrive every five controls.
- An arm spec enables the path through `"plugin_args": ["--os-blind", "--os-judge", "guard_only"]`;
  periodic variants use `["--os-blind", "--os-judge", "periodic:5"]` or `periodic:8`.
- K1 selector: `exp.offline_search.rounds.r04.k1_blind.blind_awm:BlindAWM` with
  `{"lib":"current","kref":5,"serving":"phase_particles","budget":2,"gates":"all"}`.
  At 500 episodes use `"lib":"big","kref":8` (π0.5 `bpool_cs`, GR00T `bpool_all`).
  Gap guard: `exp.offline_search.rounds.r04.k1_blind.judge:BlindMixedJudge`, put those selector kwargs under
  `base_kwargs`, use `progress_guard="noprog_span"`, and `--os-judge guard_only`.

Blind periodic scheduling uses the process-wide zero-based counter: `decision_index % k == k-1` forces
vision/MISS first, even at episode step 0 with `--os-judge-step0 hit`. The counter includes all connections
and persists across episode resets. This behavior is opt-in; absent `--os-blind`, the old episode clock remains.

`BlindQueryView` contains only the contract state/history fields. Input transforms run on CPU; π0.5 retains
normalization/padding, GR00T also reproduces the action-head dtype cast and valid-state mask. Every vision
request in blind mode asserts the CPU state is bit-identical to the live key builder. Blind rows use NaN
visual-key sentinels and `hist_has_vision=False`; key/token/image getters reject blind rows. GR00T output
uses unapply, unbatch, validation, then the existing gripper conversion exactly once, under its shared lock.

The plugin helper advances the real CP1 state history/counter once, then broadcasts once. It never invokes
an interceptor, key builder, or stage function on a blind row. Native shadow search is unavailable on that row.
First/post-MISS decisions require vision, successful MISS immediately invalidates K1's optional anchor,
and quantile controllers receive +infinity for blind HITs. Optional `__extra__.decision_id` rejects duplicate
last IDs without commit; `executed_steps != 5` requests vision. Malformed candidates and output-transform
preflight failures request vision before any dense-history/counter/action commit.

R4 JSONL fields on every decision: `vision`, `src`, `hit`, `blind_age`, `look_reason`, `miss_k`, `s1_ms`,
`s23_ms`, `served_head`. `blind_age` counts consecutive blind decisions **before** this request.
`served_head` is the actual normalized executed `[:5,:7]`, including policy MISSes. Blind timings are null.
Additional evidence includes `searched`, `source`, `decision_index`, actual `stage1_calls`,
`shadow_available`, `robot_state`, `last_vision_step`, full supplied rows/weights, and uncapped blind gate/phase
extras. Stage 1 is explicitly timed; preflight/output have separate CPU timings. `s23_ms` retains the
post-search-to-action interval and therefore includes small orchestration overhead, not only kernels.
K3's installed entry-point hook supplies startup `stage1_mode` and `miss_steps`.

## Final CPU runs

**21 final checks, 963 decisions, all PASS.**
All blind integration runs use two interleaved connections, two reset episodes each, 48 decisions, and
exactly 48 broadcasts. The fake stage 1 raises if invoked on an expected blind decision. Each run asserts
stage-call count equals vision count, matching dense state/action/HIT lengths and orchestrator counters.
Every normalized action, rows, weights, scalar extras, and look reason is recomputed by `verify_logs`.

| Report directory under `results/` | Decisions | Vision | Blind | MISS | Result |
|---|---:|---:|---:|---:|---|
| final_blind_cap_burst | 48 | 40 | 8 | 24 | PASS |
| final_blind_g500 | 48 | 32 | 16 | 8 | PASS |
| final_blind_p50 | 48 | 32 | 16 | 8 | PASS |
| final_budget0 | 48 | 48 | 0 | 8 | PASS |
| final_hist | 41 | 41 | 0 | 0 | PASS |
| final_inf | 38 | 38 | 0 | 38 | PASS |
| final_l10_g50 | 48 | 36 | 12 | 9 | PASS |
| final_l10_g500 | 48 | 36 | 12 | 9 | PASS |
| final_l10_p50 | 48 | 35 | 13 | 9 | PASS |
| final_l10_p500 | 48 | 34 | 14 | 9 | PASS |
| final_log_only | 41 | 41 | 0 | 7 | PASS |
| final_mixed | 41 | 41 | 0 | 24 | PASS |
| final_no_blind_method | 48 | 48 | 0 | 9 | PASS |
| final_periodic_p500 | 48 | 39 | 9 | 9 | PASS |
| final_periodic_step0 | 48 | 48 | 0 | 48 | PASS |
| final_pure | 41 | 41 | 0 | 0 | PASS |
| final_quantile | 41 | 41 | 0 | 16 | PASS |
| final_spatial_g50 | 48 | 31 | 17 | 9 | PASS |
| final_spatial_g500 | 48 | 30 | 18 | 9 | PASS |
| final_spatial_p50 | 48 | 34 | 14 | 9 | PASS |
| final_spatial_p500 | 48 | 34 | 14 | 9 | PASS |

The toy guard test exercises HIT → blind → blind → vision HIT → MISS → vision HIT. Both toy model tests
also reject four duplicates, four malformed candidates, and four injected output failures, and request
vision for four partial-execution probes, without appending history or advancing counters. Logged fake
output-transform tracebacks are intentional assertions in these passing tests.

The eight K1 spatial/l10 × π0.5/GR00T × 50/500 checks use `BlindMixedJudge`, gates `all`, B=2,
and `periodic:5`. Their nine MISSes per 48 decisions prove global scheduling continues through resets and
connection interleaving; every following decision reacquires vision. Budget 0 and a non-blind method have
zero bypasses. Periodic:1 beats the explicit step-0 HIT override, yielding 48/48 MISSes.

Legacy pure-cache and mixed JSONL are **byte-identical** with fixed clocks/PID and the same invocation:
63,330 bytes / 45 rows and 91,754 bytes / 45 rows, respectively. See `legacy_parity.py` and
`results/parity_{pure,mixed}_{before,after}.jsonl`. Pure and mixed selftests were rerun after installation.
ProbeHist's full 41-row logged-history replay also passes, beyond its ordinary first-row offline comparison.

Main reproducible matrix:
```bash
bash exp/offline_search/rounds/r04/k2_serving/run_checks.sh
```
Every Python command inside that script is prefixed with the required affinity and thread environment.
The spatial K1 matrix uses its same K1 invocation with `--cell <model>_spatial_cache` and
`--yaml /home/weiland/trace_runs/os_closed_loop/r02_g50/config/oscl50_<p|g>_sp_cl0.yaml`;
reports are `final_spatial_*`. Additional exact checks:
```bash
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r04/k2_serving/legacy_parity.py after pure
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r04/k2_serving/legacy_parity.py after mixed
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r04/k2_serving/check_log_only.py
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python exp/offline_search/rounds/r04/k2_serving/summarize.py
```

## Live GPU smoke and launch example

π0.5 used port 23180, PID 2221503, `STAGE1_ONLY=1`; final client replay: 2 episodes / 24 decisions,
16 vision / 8 blind, exactly 16 stage-1 calls, null stage timings on all 8 blind rows,
and **24/24 full wire chunks bit-identical to offline CPU output recomputation, max difference 0**.
`verify_logs` recomputes all 48 normalized decisions in its cumulative server log (two replay passes).
Observed process memory: 2,356 MiB. Evidence: `results/gpu_pi05/{replay.json,verify.json,gpu_memory.txt}`.

GR00T used port 23181, PID 3274204, the 500-episode `bpool_all` table: 2 episodes / 24 decisions,
16 vision / 8 blind, exactly 16 stage-1 calls, all blind timings null, normalized replay **24/24**,
and **24/24 full wire chunks bit-identical, max difference 0**, including gripper sign conversion.
All 24 robot-state vectors also equal the archival store bitwise; all 16 vision key pairs equal the store.
Observed process memory: 1,942 MiB; CPU-first loading reported 1,425.2 MiB peak allocated CUDA memory.
Evidence: `results/gpu_groot/{replay.json,verify.json,gpu_memory.txt,server.log,launch.sh}`.

Memory admission checks showed >8 GiB free before each server; at most one K2 server ran at any time.
Both PIDs were stopped after their replay, and their ports are released. The stock GR00T loader first
loads the full model onto CUDA even with `--stage1-only`. `groot_smoke_server.py` instead loads on CPU,
discards stages 2/3 there, then transfers only surviving tensors, avoiding that transient >4 GB allocation.
This wrapper is smoke-only; K3/coordination should decide whether to adopt that loading order operationally.

Exact π0.5 smoke launch (choose a free permitted port and a fresh log directory):
```bash
CPUS=22-25,66-69 OMP=1 STAGE1_ONLY=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 bash exp/offline_search/closed_loop/ops/start_server.sh pi05 libero_spatial 23180   /home/weiland/trace_runs/os_closed_loop/r02_g50/config/oscl50_p_sp_cl0.yaml   /home/weiland/projects/openpi/exp/offline_search/rounds/r04/k2_serving/results/gpu_pi05 k2_pi05   --os-method exp.offline_search.closed_loop.probe:ProbeBlind --os-cell pi05_spatial_cache   --os-root /dev/shm/offline_search_store --os-blind --os-log-inputs --os-judge always

taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu PYTHONPATH=.:src   .venv/bin/python -m exp.offline_search.closed_loop.replay_client --port 23180 --cell pi05_spatial_cache   --episodes 2 --max-decisions 12 --log-dir exp/offline_search/rounds/r04/k2_serving/results/gpu_pi05   --tag k2_replay_pi05_final --wire-checkpoint /home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch   --out exp/offline_search/rounds/r04/k2_serving/results/gpu_pi05/replay.json
```
GR00T's exact safe launch is `results/gpu_groot/launch.sh` (its own venv, CPU range, offline model paths and
port are explicit). The replay uses that venv/PYTHONPATH, `--port 23181 --cell groot_spatial_cache`,
`--wire-checkpoint /data/ckpt/n15_libero_spatial`, and the same `--episodes 2 --max-decisions 12`.
Offline normalized validation:
```bash
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src   .venv/bin/python -m exp.offline_search.closed_loop.verify_logs   --log-dir exp/offline_search/rounds/r04/k2_serving/results/gpu_groot --tag k2_groot
```

## Library accounting and limits

Both deployed candidate scales were tested with their own fits; no borrowed big-library information is used
by these test methods. Library rows: π0.5 spatial 1,018 / 10,909 (49 / 500 episodes), π0.5 l10 2,640 / 29,472;
GR00T spatial 1,063 / 11,751, GR00T l10 2,645 / 29,631 (50 / 500 episodes in the other cells).
The serving adapter adds no library representation or fitted library artifact. K1 owns its method footprint.
The existing native reference pkl sizes measured on disk are π0.5 spatial **430,792,483 B** and l10
**1,103,155,631 B**, GR00T spatial **429,351,282 B** and l10 **1,068,314,575 B**. They are reference
artifacts, not measured 500-episode plugin-fit sizes. Per-cell paths/counts are in `validation_summary.json`.

π0.5's archival store is not bit-identical to the present live preprocessing/keys: replay's maximum normalized
state difference from the store is 1.1920928955078125e-7. The CPU blind-preparation state **did** pass its
bitwise assertion against the actual live key builder on every vision decision; wire parity was exact.
These are open-loop serving tests, not LIBERO success-rate or latency-under-load measurements. Real GPU
MISS/denoising was not exercised: CPU MISS tests use the recorded policy chunks; all GPU smoke decisions
were HITs on stage-1-only models. GR00T mixed guards here are CPU integration checks, not a recommendation
for a GR00T mixed arm. No throughput/SR conclusion is claimed.

Blind rows have source `cache_blind`, not a fresh retrieval confidence. Never feed their NaN confidence into
an old visual-threshold controller. Use the explicit mask for adjacent-key features and K1's gap-aware guard.
The current client still sends images; network transfer is not reduced. Trace twins, CP2 serving, and
state-only network handshakes are outside this implementation. Unexpected transform failures request vision;
replaying a hardware/environment-specific output failure offline is not claimed.

## Coordinator next steps

1. Add `--os-blind` to K1 arms; use `--os-log-r4` on non-blind R4 arms whose ledgers need the new fields.
   `--os-blind` alone changes no client execution length: keep five controls per request.
2. Use `STAGE1_ONLY=0` for arms whose judge can MISS. The plugin refuses a potentially-MISS judge on a
   stage-1-only server; the smoke example uses `always` deliberately.
3. Consume `vision` and actual `miss_k` in K4's cost ledger, plus K3's stage mode/completion cost. Preserve
   the server-wide periodic-clock semantics in comparisons. The old π0.5 smoke client's `mixed` IR field
   predates the replay-report correction; use its measured V=16/N=24/M=0, giving reference IR .1013333.
   Current `replay_client` computes that R4 formula (also verified by the GR00T smoke report).
4. Coordinator runs the authorized closed-loop pilots/full arms and remote synchronization. None were run
   by K2. Cost-engine compositions such as wrist-only plus blind serving still need the combined smoke.
