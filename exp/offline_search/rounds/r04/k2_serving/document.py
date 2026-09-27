from pathlib import Path
from datetime import datetime, timezone
import json
import os

base=Path(__file__).resolve().parent
closed=base.parents[2]/'closed_loop'
readme=closed/'README.md'
addition='''

## R4: opt-in vision-free serving (K2)

`--os-blind` enables the pre-interceptor `method.blind_step(BlindQueryView)` path and the R4 log schema.
`--os-log-r4` enables the R4 schema alone. Without either flag, the plugin's legacy pure-cache and mixed rows
remain unchanged. K3's serving entry points separately attach their startup cost metadata.

The contract is in `blind.py`: `LookReason(code, name)`, `BlindResult(action, rows, weights, library, extras)`,
and a frozen state-only `BlindQueryView`. The action is the normalized float32 `(H,32)` library chunk; the
plugin applies the normal model output transforms. Vision keys in dense blind history are NaN, with
`hist_has_vision` indicating validity. `prev_a_exec` is the actual normalized chunk served, including a policy
MISS. First decisions and post-MISS decisions require vision. No client image handshake is needed: continue
sending fresh full observations every five controls.

With `--os-blind`, periodic MISS uses a **server-wide**, zero-based `decision_index`, shared by connections
and continuing across episode resets: index `% k == k-1` is due. It is checked before `blind_step` and before
a step-0 HIT override. Without blind serving, periodic mode keeps its legacy episode-step clock.

Every R4 decision carries `vision`, `src` (`cache`, `cache_blind`, `policy`), `hit`, `blind_age` (number of
blind decisions BEFORE this one), `look_reason`, `miss_k`, `s1_ms`, `s23_ms`, and normalized `served_head`
(`[:5,:7]`). Blind rows have null stage timings, `searched=false`, and `shadow_available=false`.
`stage1_calls` counts actual stage function calls; `robot_state`, full supplied member rows/weights, and
`blind_extras` retain anchor/gate/phase diagnostics. `--os-log-inputs` also saves wire actions and dense masks.
`miss_k` is null on HIT; its MISS value comes from the live interceptor/action head.

Blind serving requires an untraced CP1 stack and one plugin session per connection. Quantile controllers
receive +infinity for a blind HIT. Judge caps/bursts require vision; malformed blind candidates or output
preflight failures fall back before committing histories. Optional `obs["__extra__"]` audit fields are
`decision_id` and `executed_steps`: a duplicate last ID is rejected with no commit; a count other than five
forces vision. This is not a state-only network protocol.

CPU regression and interleaving checks: `selftest --blind`; toy method:
`exp.offline_search.closed_loop.probe:ProbeBlind` (kwargs `library`, `budget`). With `--os-judge guard_only`
its first six decisions are vision HIT, blind, blind, vision HIT, MISS, vision HIT. Live stage-1-only smoke
uses `--os-judge always`. `verify_logs` replays both method interfaces; `replay_client --max-decisions 12
--wire-checkpoint <checkpoint>` compares actual wire actions with CPU-only output recomputation.

See `../rounds/r04/k2_serving/HANDBACK.md` for exact launch commands, completed checks, and the smoke-only
GR00T CPU-first loader that avoids the stock loader's transient full-model GPU allocation.
'''
# One read/write/rename: preserve the other owners' current README contents.
current=readme.read_text()
if '## R4: opt-in vision-free serving (K2)' not in current:
    tmp=readme.with_name('.README.md.k2.tmp')
    tmp.write_text(current+addition)
    os.replace(tmp,readme)
    with (base/'install_times.txt').open('a') as f:
        f.write('README.md appended '+datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')+'\n')

summary=json.loads((base/'results'/'validation_summary.json').read_text())
installs={}
for line in (base/'install_times.txt').read_text().splitlines():
    parts=line.split()
    if len(parts)==3 and parts[1] in ('installed','appended'):
        installs[parts[0]]=parts[2]
rows=[]
for name,r in summary['checks'].items():
    mx=r.get('mixed',{})
    rows.append(f"| {name} | {r['decisions']} | {r.get('vision',r['decisions'])} | {r.get('blind',0)} | {r.get('miss',mx.get('n_miss',0))} | PASS |")
file_rows='\n'.join(f'| `closed_loop/{name}` | {when} |' for name,when in installs.items())
body='''# K2 serving hand-back

Completed 2026-09-27. All work stayed in K2-owned files, plus the permitted atomic README append. No git,
remote host, simulator worker, or closed-loop chain was used. Every Python process used CPUs 22-25,66-69
with OMP/OPENBLAS/MKL threads set to one. No GPU OOM occurred; both smoke servers have been stopped.

## Installed files

| File | Latest atomic install/append, UTC |
|---|---|
'''+file_rows+'''

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

'''+f"**{summary['total_checks']} final checks, {summary['total_decisions']} decisions, all PASS.**"+'''
All blind integration runs use two interleaved connections, two reset episodes each, 48 decisions, and
exactly 48 broadcasts. The fake stage 1 raises if invoked on an expected blind decision. Each run asserts
stage-call count equals vision count, matching dense state/action/HIT lengths and orchestrator counters.
Every normalized action, rows, weights, scalar extras, and look reason is recomputed by `verify_logs`.

| Report directory under `results/` | Decisions | Vision | Blind | MISS | Result |
|---|---:|---:|---:|---:|---|
'''+ '\n'.join(rows)+'''

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
CPUS=22-25,66-69 OMP=1 STAGE1_ONLY=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
bash exp/offline_search/closed_loop/ops/start_server.sh pi05 libero_spatial 23180 \
  /home/weiland/trace_runs/os_closed_loop/r02_g50/config/oscl50_p_sp_cl0.yaml \
  /home/weiland/projects/openpi/exp/offline_search/rounds/r04/k2_serving/results/gpu_pi05 k2_pi05 \
  --os-method exp.offline_search.closed_loop.probe:ProbeBlind --os-cell pi05_spatial_cache \
  --os-root /dev/shm/offline_search_store --os-blind --os-log-inputs --os-judge always

taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu PYTHONPATH=.:src \
  .venv/bin/python -m exp.offline_search.closed_loop.replay_client --port 23180 --cell pi05_spatial_cache \
  --episodes 2 --max-decisions 12 --log-dir exp/offline_search/rounds/r04/k2_serving/results/gpu_pi05 \
  --tag k2_replay_pi05_final --wire-checkpoint /home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  --out exp/offline_search/rounds/r04/k2_serving/results/gpu_pi05/replay.json
```
GR00T's exact safe launch is `results/gpu_groot/launch.sh` (its own venv, CPU range, offline model paths and
port are explicit). The replay uses that venv/PYTHONPATH, `--port 23181 --cell groot_spatial_cache`,
`--wire-checkpoint /data/ckpt/n15_libero_spatial`, and the same `--episodes 2 --max-decisions 12`.
Offline normalized validation:
```bash
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src \
  .venv/bin/python -m exp.offline_search.closed_loop.verify_logs \
  --log-dir exp/offline_search/rounds/r04/k2_serving/results/gpu_groot --tag k2_groot
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
'''
(base/'HANDBACK.md').write_text(body)
(base/'INTEGRATION.md').write_text('K2 COMPLETE. Shared blind.py/plugin.py/probe.py/selftest.py/verify_logs.py/replay_client.py installed atomically.\n'
 'Enable --os-blind; logs alone --os-log-r4. Both preserve legacy behavior/schema when absent.\n'
 'Periodic + blind uses process-wide decision_index across connections and episode resets.\n'
 '21 final CPU checks / 963 decisions PASS; both GPU models 24 final replay decisions, 16 vision / 8 blind, 24/24 wire parity.\n'
 'Both GPU servers stopped. See HANDBACK.md and results/validation_summary.json for exact commands, evidence and limits.\n')
print(base/'HANDBACK.md')
