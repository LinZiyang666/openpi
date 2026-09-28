"""Assemble the handback only from completed installed-file evidence."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
BASE=Path(__file__).resolve().parent
R=BASE/'results'
load=lambda p:json.loads(Path(p).read_text())
install=load(R/'install.json')
assert hashlib.sha256(Path(install['file']).read_bytes()).hexdigest()==install['installed_sha256']
existing=load(R/'installed/existing_summary.json')
assert existing['checks']==35 and existing['decisions']==1599
concurrency=load('/tmp/k6_installed_concurrency/summary.json')
assert len(concurrency)==12 and all(r['PASS'] for r in concurrency)
episode_pairs=0
for case in concurrency:
    paired=[]
    for mode in ('threaded','serialized'):
        path=Path('/tmp/k6_installed_concurrency')/case['config']/mode/'decisions_concurrency.jsonl'
        episodes=[json.loads(line) for line in path.read_text().splitlines() if '"ev": "episode"' in line]
        for episode in episodes:
            for key in ('ts','t_start','t_end'):episode.pop(key,None)
        paired.append(sorted(episodes,key=lambda r:(r['conn'],r['episode_seq'])))
    assert paired[0]==paired[1],case['config']
    assert len(paired[0])==16
    episode_pairs+=len(paired[0])
edge=load('/tmp/k6_installed_edges/edge.json');assert edge['PASS']
audit=load(R/'installed/final_audit.json');assert audit['PASS']
estimator=load(R/'installed/estimator_validation.json');assert estimator['PASS']
parity=load(R/'installed/parity_installed.json');assert all(r['byte_identical'] for r in parity)
overlay=load(R/'installed/overlay_installed.json');assert overlay['PASS']
(R/'installed/concurrency_summary.json').write_text(json.dumps(concurrency,indent=2)+'\n')
(R/'installed/edge_summary.json').write_text(json.dumps(edge,indent=2)+'\n')
counts={}
for row in existing['rows']:
    n,d=counts.get(row['group'],(0,0));counts[row['group']]=(n+1,d+row['decisions'])
conc_table='\n'.join(f"| {r['config']} | {r['vision']} / {r['blind']} / {r['miss']} | {r['serialized_s']:.3f} | {r['threaded_s']:.3f} | {r['speedup']:.2f}x | {r['eligible']['CALL']} / {r['eligible']['CACHE']} |" for r in concurrency)
replay_table='\n'.join(f"| {r['arm']} | {r['decisions']} | {r['call']} / {r['cache']} | {r['misses']} |" for r in audit['replay_audits'])
parity_table='\n'.join(f"| {r['mode']} | {r['bytes']:,} | {r['rows']} | byte-identical |" for r in parity)
existing_table='\n'.join(f'| {g.upper()} matrix | {n} | {d:,} | PASS |' for g,(n,d) in counts.items())
speed=[r['speedup'] for r in concurrency]
text=f'''# K6 hand-back: concurrent R4 serving

Completed {datetime.now(timezone.utc).isoformat()}. Only `closed_loop/plugin.py` was changed on the live import path. It now uses one lock per R4 connection and short runtime transactions. No runtime lock spans `_osp_inner.infer`, stage 1, stage 2/3, method query, or blind selection. Existing flags enable the change automatically: `--os-blind`, `--os-log-r4`, or K5 randomization. No flag, method, retrieval, guard, synthesis, or randomization rule was added.

## Atomic installation and changed files

| Shared file | Installed UTC | SHA256 | Bytes |
|---|---|---|---:|
| `exp/offline_search/closed_loop/plugin.py` | {install['installed_utc']} | `{install['installed_sha256']}` | {install['bytes']} |

Preimage SHA256: `{install['before_sha256']}`. `install.py` checked the live preimage, parsed the candidate, wrote and fsynced a same-directory temporary file, then used **one `os.replace` (atomic rename/mv)** and fsynced the directory. Evidence: `results/install.json`. The installed file is byte-identical to `dev/plugin.py`; all final checks below use the installed import. No shared file changed after this install.

Other owned shared files (`selftest.py`, `verify_logs.py`, `replay_client.py`, `blind.py`) needed no changes. Development and reference copies are in `dev/` and `before/`. New deliverables under this directory: `concurrency_test.py`, `edge_test.py`, `launch_test.py`, `prepare_checks.py`, `run_checks.sh`, `prepare_extended.py`, `run_extended.sh`, `install.py`, `write_handback.py`, adapted `k5_*.py` audit helpers, `arms_rand.json` copied unchanged from K5, and `results/`. `results/deliverable_hashes.json` lists SHA256 and modification time for added source/scripts and this handback. The files in K1–K5, src, harness, profile, ops, and the cost table were not edited.

## Shared-state audit and synchronization

| State touched by a decision | Ownership / treatment | Why this is safe |
|---|---|---|
| Runtime `decision_count`, session `_decision_index`, server periodic clock | **Short runtime RLock** reserves and increments before work | Unique monotonic allocation across all connections and episodes; no model call under it. Periodic due checks and verdicts use that reserved value. |
| Runtime `_conn_ids`, `sessions` WeakSet | **Short runtime RLock** for ID allocation, registration and flush snapshot | Exit flush releases runtime lock before taking each connection lock, avoiding lock-order inversion. |
| Fitted runtime method template; library action/key/state arrays, PCA/metric/calibration arrays; IDs, task/episode maps, table sizes, options, judge spec, provenance | **Read-only** after construction/fit | Existing deep clone copies Python containers and shares fitted NumPy/Torch arrays by contract. Audited AWM, MixedJudge, BlindAWM, BlindMixedJudge, ProbeHist/Blind and K5 paths do not mutate fitted arrays during serving. |
| Connection method and nested bases; MixedJudge `_s`, memo lists, progress/span counters; blind anchor/phase/weights and diagnostics | **Per connection**, protected by session RLock | `reset`, `query`, `blind_step` and anchor invalidation are in the same connection transaction. Runtime template never receives serving calls. R4 now refuses a failed deep clone instead of falling back to unsafe shallow mutable-container sharing; legacy fallback remains. |
| ProbeBlind `n`, `blind_calls`, anchor rows/weights | **Per connection** | Reset initializes these; queries/bypasses update only that clone. Tests compare both counters and all histories against serialized reference. |
| K5 assignment, opportunities, exposure flag, stall age/context | **Per episode per connection** | `_begin` constructs a fresh overlay. Assignment uses its existing private SHA256-seeded `random.Random`; no global RNG draw. Query/verdict/overlay/history commit stay in the connection lock. |
| Session identity, episode sequence, step, HIT run/burst, blind age, last vision, duplicate/pending ID; `_dec`, `_synth`, `_tok_cache`, observation, dense buffers, replay records | **Per connection** | Lock spans observation setup through `after_infer` and its cleanup, plus task/episode callbacks and exit flush. A new request or reset cannot overwrite an unfinished decision. |
| `_BlindAdapter`, stage wrappers and `stage1_calls`, `_s1_ms`, prepared state and output | **Per connection** | π0.5 interceptor `_stage1_fn` and GR00T staged runner are constructed per connection; instrumentation closes over that session. No mutation of shared model functions. Existing CUDA timing synchronization remains; timing values are intentionally excluded from parity. |
| Orchestrator state/action histories and step, key-builder cache, native/plugin strategies, judges, gates, timer, payload facade and synthesized payloads | **Per connection** from existing factory | The session lock includes interceptor work, blind `commit_external_hit`, once-only broadcast and clear. Shared backend is reached through existing fresh per-connection facades. |
| Native storage backend payloads/keys/library stats | **Read-only serving data** under existing `write_policy=never` configs | Same backend and native shadow search as legacy. Frozen-filter/matrix lazy caches publish complete immutable equivalents; racing population can duplicate work, not alter scores. Session score memo buckets are keyed by fresh native session UUID; close/reset clears only that bucket. Existing diagnostic fetch/search counters and Python registry operations remain in the existing CPython backend, outside plugin decision rules. No new library writes. |
| Runtime quantile deque/sorted list/count | **Short CPU verdict transaction** under runtime RLock, retaining controller's own data lock | R4 `tau → verdict → push` is atomic; blind `+inf` push takes the same runtime lock. Retrieval/GPU work is outside it. Legacy non-R4 retains its prior separate operations. Transaction order is the concurrent verdict order. |
| JSONL writer | Existing **writer lock**, held through all bytes of one line | R4 loops over short `os.write` results while retaining the lock; no partial fragments can interleave. Legacy one-write path unchanged. JSON serialization is outside the writer lock. |
| NPZ replay writer | **Per connection** lock and unique existing `tag/c<conn>/e<episode_seq>/uid/attempt` filename | Finish/reset cannot race its own records; independent files can be written concurrently. |
| Library-validation registry | Existing `_vlock`, during connection setup | Once-per-backend validation is unchanged; not an inference lock. |
| `_TLS.new_sessions`, `RUNTIME`, null profiler | **Thread-local** factory binding; runtime installed once; profiler stateless/read-only | Connection factories cannot capture another thread's sessions. |
| π0.5 model/batching coordinator, transforms, MISS sampling; GR00T shared model/transform lock | Existing **legacy implementation preserved** | Plugin adds no runtime lock around GPU work. π0.5 can submit concurrent staged requests again. GR00T's pre-existing `_InferLockedPolicy` still serializes its model; blind transforms retain that same existing adapter lock. Live GPU numerical/RNG equivalence is not established by CPU tests. |

Lock order is connection → short runtime/controller or writer lock. Exit flush takes a runtime snapshot and releases it before acquiring connection locks. There is no runtime-lock → GPU wait or runtime-lock → connection-lock cycle.

Duplicate last blind IDs are rejected before reserving an index and commit nothing. If an admitted request subsequently throws, its reservation is **not reused**: another connection may already have the next index. Therefore failures can leave gaps, unlike the previous successful-completion counter. Allocation remains monotonic; completed JSONL rows can appear out of index order. Consumers must not interpret JSONL completion order as reservation order. A method exception after input-history mutation still requires resetting that affected connection, as before; peers continue normally.

## Final installed regression results

| Check | PASS checks | Decisions | Result |
|---|---:|---:|---|
{existing_table}
| Existing plugin total | 35 | 1,599 | PASS |
| K5 real MixedJudge replay arms | 4 | 1,368 | PASS |
| K5 injected overlay lifecycle/history | 1 | 104 | PASS |

Evidence: `results/installed/existing_summary.json`, `results/installed/overlay_installed.json`, `results/installed/final_audit.json`; full arrays/logs `/tmp/k6_installed/existing`, `/tmp/k6_installed_replays`, `/tmp/k6_overlay_installed`. K4 artifact path is recorded in `/tmp/k6_installed/existing/k4/plugin_artifact_root.txt`. ProbeHist's additional 41-decision logged-input verifier has all offline equality fields 1.0 and 41/41 selected/executed equality. Each selftest's existing verifier ran as part of its original recipe.

| K5 arm | Decisions | Eligible CALL / CACHE | MISS |
|---|---:|---:|---:|
{replay_table}

K5 assignment stability/complement checks, six invalid option combinations, missing-init rejection, four forged-log rejections, cross-scale rejection, estimator raw/collect/KPI/export consistency, two historical 500-episode baseline layouts, replay estimation and four-arm cost ledger all passed via relocated `run_all_checks.sh` constituents. Estimator validation reran **200 planted + 200 null datasets**, 500 init clusters each, 499 bootstrap draws; exact Δ(Y,N,M)=(1,4,2) recovered. Details: `results/installed/estimator_validation.json`, `replay_estimator.json`, `ledger_check.json`, `final_audit.json`. These are regression checks of unchanged K5 functionality, not new rollout outcomes.

Fixed-clock/PID, identical-invocation JSONL comparison against the captured pre-K6 plugin:

| Mode | Bytes | Rows | Result |
|---|---:|---:|---|
{parity_table}

Evidence: `results/installed/parity_installed.json` and the paired `parity_*_before.jsonl` / `parity_*_installed.jsonl` files. Legacy pure/mixed behavior remains byte-identical; sequential R4 logs also remain byte-identical.

## Concurrent decision parity and measured speed-up

**12/12 configurations PASS**, eight threads/connections each, **183 decisions and 16 episodes per configuration**: {sum(r['decisions'] for r in concurrency):,} concurrent decisions matched against the same number of pre-K6 serialized decisions. Every run reached eight simultaneously active fake stage-1 calls. Sleep is **60 ms inside stage 1**, plus **25 ms on MISS inside inference**. Measured wall time covers requests and interleaved episode resets, excluding setup/fit and final flush. Speed-up range **{min(speed):.2f}–{max(speed):.2f}x**, median **{statistics.median(speed):.2f}x**, on the assigned eight CPUs. These are CPU fake-policy concurrency measurements, not GPU or LIBERO arm latency claims.

| Configuration | Vision / blind / MISS | Serialized s | Concurrent s | Speed-up | Eligible CALL / CACHE |
|---|---:|---:|---:|---:|---:|
{conc_table}

`blind_*` and `periodic_*` use real `BlindMixedJudge`, `noprog_span`, guard-only method configuration, `phase_particles`, B=2, all look gates, both 50/500 libraries. Periodic uses `--os-blind --os-judge periodic:5`. `mixed_*` uses real fitted guard-only MixedJudge with `--os-log-r4`. Four K5 rows use the exact K5 fit artifacts/kwargs and both replicates. K5 deliberately repeats recorded observations after step 3 to activate real guards and exercise first/third CALL/CACHE landmarks without injecting method verdicts. Probe rows additionally test ProbeBlind's counters in π0.5 and GR00T-shaped stacks.

A fresh process runs the installed plugin with eight concurrent workers; another fresh process runs **the saved original plugin** serialized in the observed reservation order. Comparison retains decision indices by replaying that schedule, stronger than merely ignoring indices: every non-timing decision field, full wire action bytes, wire diagnostics, verdict, method counters, action/state/key buffers, HIT/vision masks, orchestrator histories and step counters matches exactly. Native shadow search is enabled. Every run also passes the existing offline log verifier. Reservation instrumentation checks actual allocation sequence 0…182; per-connection indices increase; all global indices are unique. Completion-row order may differ.

All {episode_pairs} paired episode-summary rows also match exactly after removing only `ts`, `t_start`, and `t_end`. Startup rows have different invocation/output-directory provenance in the two processes; the fixed-invocation parity check above separately covers startup-byte parity.

Evidence: `/tmp/k6_installed_concurrency/<configuration>/{{threaded,serialized}}/concurrency.json`, corresponding `verify.json`, and logs; compact copied summary `results/installed/concurrency_summary.json`.

## Edge cases and failure tests

Installed `edge_test.py`: **23 successful decisions, 25 reservations, two intentional failure gaps**, eight connections. It verifies peer blind progress while another connection's MISS is blocked; explicit reset and implicit task change while peers run; a simultaneous same-connection duplicate waits then fails; two duplicate rejections with no reservation/commit; same-connection lifecycle waits through inference/logging; stage failure and method-query failure isolate to their connection; post-failure reset recovers; a MISS forces the following vision anchor. It forces 17-byte partial writes with eight writers and validates **32 large intact JSONL records**. It separately replays **64 concurrent quantile transactions** against a serial reference and checks strict clone failure. Evidence: `results/installed/edge_summary.json`, `/tmp/k6_installed_edges/decisions_edge.jsonl` and NPZ inputs.

One development edge-test fixture initially reused decision ID 0 for an implicit task change and correctly hit the existing duplicate rejection; the fixture was corrected to a distinct ID and passed in development and installed runs. This was not a plugin regression. No final plugin, verifier, parity or concurrency check failed.

## Exact commands and reproduction

Run from `/home/weiland/projects/openpi`. Every Python process, including child processes, is prefixed with CPUs **26-29,70-73** and OMP/OpenBLAS/MKL=1. CUDA was disabled. At most eight Python processes were scheduled at once; the concurrency driver uses eight threads in its one worker process. No GPU server, port, tmux session, LIBERO worker, chain, remote host, git command, or forbidden review test was used.

```bash
K6=exp/offline_search/rounds/r04/k6_concurrency
PY=(taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python)

# Completed before installation:
bash "$K6/run_checks.sh" dev > "$K6/results/dev_checks.log" 2>&1
"${{PY[@]}}" "$K6/concurrency_test.py" --source dev --config probe_pi05,blind_50,periodic_50,mixed_50,r4k5_p_l10_g50_r1 --out /tmp/k6_dev_concurrency > "$K6/results/dev_concurrency.log" 2>&1
"${{PY[@]}}" "$K6/edge_test.py" --source dev --out /tmp/k6_dev_edges_v2 > "$K6/results/dev_edges_v2.log" 2>&1
"${{PY[@]}}" "$K6/concurrency_test.py" --source dev --config r4k5_p_l10_g50_r1 --out /tmp/k6_dev_landmarks > "$K6/results/dev_landmarks.log" 2>&1

# Completed atomic install and all final checks:
"${{PY[@]}}" "$K6/install.py"
bash "$K6/run_checks.sh" installed > "$K6/results/installed_checks.log" 2>&1
"${{PY[@]}}" "$K6/concurrency_test.py" --source installed --config all --out /tmp/k6_installed_concurrency > "$K6/results/installed_concurrency.log" 2>&1
"${{PY[@]}}" "$K6/edge_test.py" --source installed --out /tmp/k6_installed_edges > "$K6/results/installed_edges.log" 2>&1
bash "$K6/run_extended.sh" independent > "$K6/results/installed_extended_independent.log" 2>&1
bash "$K6/run_extended.sh" dependent > "$K6/results/installed_extended_dependent.log" 2>&1
"${{PY[@]}}" "$K6/write_handback.py"
```

The three matrices are the original K2/K1/K4 recipes copied into `dev/installed_{{k2,k1,k4}}.sh`, changing only affinity, scratch/store paths and test loader. `results/installed/replay_commands_installed.json` records all expanded K5 replay commands. Extended K5 helpers retain original audit logic, with paths redirected into K6 ownership. For a rerun choose fresh scratch paths: blind tests append JSONL, concurrency/edge tests refuse an existing output directory, and `install.py` deliberately refuses an already changed preimage. Do not rerun the installer merely to repeat tests.

## Residual limits and coordinator next steps

1. Newly started servers import the installed fix automatically; already running processes retain their imported code. Let current arms finish and use the normal next server start. No new fit, arm spec, client change, remote sync, or serving flag is needed. Any separate server checkout needs this one `plugin.py` replacement.
2. GPU/replay-client smoke was optional and **not run**. Real batching throughput, GPU memory at 24 workers, numerical differences with different batch shapes, stochastic MISS-noise assignment and closed-loop SR/latency are **unverified**. The existing sampler/coordinator is unchanged; this patch does not promise bitwise equality of live stochastic GPU trajectories across different schedules. The exact parity claim here is for the observed fixed-input/key/policy-chunk tests.
3. GR00T still has its own legacy model lock; removing it is outside this task. Its CPU fake-stack test establishes plugin isolation, not live GR00T model concurrency. Existing GPU-wide timing synchronization can also affect measured latency.
4. Custom methods must honor the existing read-only fitted-array contract. R4 shallow-copy fallback now fails clearly at connection construction. The tested deployed methods all deep-clone successfully.
5. Global periodic assignment can interleave differently as requests arrive concurrently; its modulo rule is unchanged. Quantile controller operations are serializable in verdict order. Failed admitted requests leave unused index gaps; duplicate last IDs do not. Episode-level global controller snapshots naturally depend on when an episode ends relative to peers.
6. Coordinator should observe its next normal π0.5 arm for wall time and memory and record real serving latency. The measured {min(speed):.2f}–{max(speed):.2f}x fake-policy speed-up establishes removal of plugin serialization, not a forecast of GPU or complete-arm speed-up.
'''
(BASE/'HANDBACK.md').write_text(text)
paths=sorted(p for p in BASE.rglob('*') if p.is_file() and (p.suffix in ('.py','.sh') or p.name in ('HANDBACK.md','arms_rand.json')) and 'before' not in p.parts)
records=[dict(path=str(p.relative_to(BASE)),modified_utc=datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
(R/'deliverable_hashes.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(dict(PASS=True,existing_checks=35,existing_decisions=1599,randomized_decisions=1368,concurrency_checks=len(concurrency),concurrent_decisions=sum(r['decisions'] for r in concurrency),speedup_min=min(speed),speedup_max=max(speed),speedup_median=statistics.median(speed),installed_sha256=install['installed_sha256']),indent=2))
