# K6 hand-back: concurrent R4 serving

Completed 2026-09-27T21:03:34.867226+00:00. Only `closed_loop/plugin.py` was changed on the live import path. It now uses one lock per R4 connection and short runtime transactions. No runtime lock spans `_osp_inner.infer`, stage 1, stage 2/3, method query, or blind selection. Existing flags enable the change automatically: `--os-blind`, `--os-log-r4`, or K5 randomization. No flag, method, retrieval, guard, synthesis, or randomization rule was added.

## Atomic installation and changed files

| Shared file | Installed UTC | SHA256 | Bytes |
|---|---|---|---:|
| `exp/offline_search/closed_loop/plugin.py` | 2026-09-27T20:47:26.508504+00:00 | `c29a623a597d9611f16804bafaf3100de9412ebf5da5ab5c0e8031779d52a132` | 90096 |

Preimage SHA256: `5bd0c878774ccd79fc79d1e53cee167f9e5bbca039af2b473c9e5dea8bc89f9a`. `install.py` checked the live preimage, parsed the candidate, wrote and fsynced a same-directory temporary file, then used **one `os.replace` (atomic rename/mv)** and fsynced the directory. Evidence: `results/install.json`. The installed file is byte-identical to `dev/plugin.py`; all final checks below use the installed import. No shared file changed after this install.

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
| K2 matrix | 21 | 963 | PASS |
| K1 matrix | 10 | 480 | PASS |
| K4 matrix | 4 | 156 | PASS |
| Existing plugin total | 35 | 1,599 | PASS |
| K5 real MixedJudge replay arms | 4 | 1,368 | PASS |
| K5 injected overlay lifecycle/history | 1 | 104 | PASS |

Evidence: `results/installed/existing_summary.json`, `results/installed/overlay_installed.json`, `results/installed/final_audit.json`; full arrays/logs `/tmp/k6_installed/existing`, `/tmp/k6_installed_replays`, `/tmp/k6_overlay_installed`. K4 artifact path is recorded in `/tmp/k6_installed/existing/k4/plugin_artifact_root.txt`. ProbeHist's additional 41-decision logged-input verifier has all offline equality fields 1.0 and 41/41 selected/executed equality. Each selftest's existing verifier ran as part of its original recipe.

| K5 arm | Decisions | Eligible CALL / CACHE | MISS |
|---|---:|---:|---:|
| r4k5_p_l10_g50_r1 | 342 | 3 / 1 | 117 |
| r4k5_p_l10_g50_r2 | 342 | 1 / 3 | 114 |
| r4k5_p_l10_g500_r1 | 342 | 3 / 1 | 105 |
| r4k5_p_l10_g500_r2 | 342 | 1 / 3 | 106 |

K5 assignment stability/complement checks, six invalid option combinations, missing-init rejection, four forged-log rejections, cross-scale rejection, estimator raw/collect/KPI/export consistency, two historical 500-episode baseline layouts, replay estimation and four-arm cost ledger all passed via relocated `run_all_checks.sh` constituents. Estimator validation reran **200 planted + 200 null datasets**, 500 init clusters each, 499 bootstrap draws; exact Δ(Y,N,M)=(1,4,2) recovered. Details: `results/installed/estimator_validation.json`, `replay_estimator.json`, `ledger_check.json`, `final_audit.json`. These are regression checks of unchanged K5 functionality, not new rollout outcomes.

Fixed-clock/PID, identical-invocation JSONL comparison against the captured pre-K6 plugin:

| Mode | Bytes | Rows | Result |
|---|---:|---:|---|
| pure | 63,349 | 45 | byte-identical |
| mixed | 91,773 | 45 | byte-identical |
| r4 | 145,754 | 45 | byte-identical |

Evidence: `results/installed/parity_installed.json` and the paired `parity_*_before.jsonl` / `parity_*_installed.jsonl` files. Legacy pure/mixed behavior remains byte-identical; sequential R4 logs also remain byte-identical.

## Concurrent decision parity and measured speed-up

**12/12 configurations PASS**, eight threads/connections each, **183 decisions and 16 episodes per configuration**: 2,196 concurrent decisions matched against the same number of pre-K6 serialized decisions. Every run reached eight simultaneously active fake stage-1 calls. Sleep is **60 ms inside stage 1**, plus **25 ms on MISS inside inference**. Measured wall time covers requests and interleaved episode resets, excluding setup/fit and final flush. Speed-up range **5.09–6.50x**, median **6.22x**, on the assigned eight CPUs. These are CPU fake-policy concurrency measurements, not GPU or LIBERO arm latency claims.

| Configuration | Vision / blind / MISS | Serialized s | Concurrent s | Speed-up | Eligible CALL / CACHE |
|---|---:|---:|---:|---:|---:|
| blind_50 | 103 / 80 / 18 | 12.030 | 1.888 | 6.37x | 0 / 0 |
| periodic_50 | 127 / 56 / 36 | 13.038 | 2.510 | 5.19x | 0 / 0 |
| mixed_50 | 183 / 0 / 4 | 17.199 | 2.946 | 5.84x | 0 / 0 |
| blind_500 | 92 / 91 / 5 | 8.877 | 1.744 | 5.09x | 0 / 0 |
| periodic_500 | 124 / 59 / 36 | 13.116 | 2.397 | 5.47x | 0 / 0 |
| mixed_500 | 183 / 0 / 6 | 15.128 | 2.582 | 5.86x | 0 / 0 |
| r4k5_p_l10_g50_r1 | 183 / 0 / 99 | 17.539 | 2.752 | 6.37x | 12 / 4 |
| r4k5_p_l10_g50_r2 | 183 / 0 / 91 | 17.188 | 2.713 | 6.34x | 4 / 12 |
| r4k5_p_l10_g500_r1 | 183 / 0 / 103 | 17.550 | 2.739 | 6.41x | 12 / 4 |
| r4k5_p_l10_g500_r2 | 183 / 0 / 95 | 17.454 | 2.856 | 6.11x | 4 / 12 |
| probe_pi05 | 119 / 64 / 29 | 10.276 | 1.599 | 6.43x | 0 / 0 |
| probe_groot | 119 / 64 / 29 | 10.506 | 1.617 | 6.50x | 0 / 0 |

`blind_*` and `periodic_*` use real `BlindMixedJudge`, `noprog_span`, guard-only method configuration, `phase_particles`, B=2, all look gates, both 50/500 libraries. Periodic uses `--os-blind --os-judge periodic:5`. `mixed_*` uses real fitted guard-only MixedJudge with `--os-log-r4`. Four K5 rows use the exact K5 fit artifacts/kwargs and both replicates. K5 deliberately repeats recorded observations after step 3 to activate real guards and exercise first/third CALL/CACHE landmarks without injecting method verdicts. Probe rows additionally test ProbeBlind's counters in π0.5 and GR00T-shaped stacks.

A fresh process runs the installed plugin with eight concurrent workers; another fresh process runs **the saved original plugin** serialized in the observed reservation order. Comparison retains decision indices by replaying that schedule, stronger than merely ignoring indices: every non-timing decision field, full wire action bytes, wire diagnostics, verdict, method counters, action/state/key buffers, HIT/vision masks, orchestrator histories and step counters matches exactly. Native shadow search is enabled. Every run also passes the existing offline log verifier. Reservation instrumentation checks actual allocation sequence 0…182; per-connection indices increase; all global indices are unique. Completion-row order may differ.

All 192 paired episode-summary rows also match exactly after removing only `ts`, `t_start`, and `t_end`. Startup rows have different invocation/output-directory provenance in the two processes; the fixed-invocation parity check above separately covers startup-byte parity.

Evidence: `/tmp/k6_installed_concurrency/<configuration>/{threaded,serialized}/concurrency.json`, corresponding `verify.json`, and logs; compact copied summary `results/installed/concurrency_summary.json`.

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
"${PY[@]}" "$K6/concurrency_test.py" --source dev --config probe_pi05,blind_50,periodic_50,mixed_50,r4k5_p_l10_g50_r1 --out /tmp/k6_dev_concurrency > "$K6/results/dev_concurrency.log" 2>&1
"${PY[@]}" "$K6/edge_test.py" --source dev --out /tmp/k6_dev_edges_v2 > "$K6/results/dev_edges_v2.log" 2>&1
"${PY[@]}" "$K6/concurrency_test.py" --source dev --config r4k5_p_l10_g50_r1 --out /tmp/k6_dev_landmarks > "$K6/results/dev_landmarks.log" 2>&1

# Completed atomic install and all final checks:
"${PY[@]}" "$K6/install.py"
bash "$K6/run_checks.sh" installed > "$K6/results/installed_checks.log" 2>&1
"${PY[@]}" "$K6/concurrency_test.py" --source installed --config all --out /tmp/k6_installed_concurrency > "$K6/results/installed_concurrency.log" 2>&1
"${PY[@]}" "$K6/edge_test.py" --source installed --out /tmp/k6_installed_edges > "$K6/results/installed_edges.log" 2>&1
bash "$K6/run_extended.sh" independent > "$K6/results/installed_extended_independent.log" 2>&1
bash "$K6/run_extended.sh" dependent > "$K6/results/installed_extended_dependent.log" 2>&1
"${PY[@]}" "$K6/write_handback.py"
```

The three matrices are the original K2/K1/K4 recipes copied into `dev/installed_{k2,k1,k4}.sh`, changing only affinity, scratch/store paths and test loader. `results/installed/replay_commands_installed.json` records all expanded K5 replay commands. Extended K5 helpers retain original audit logic, with paths redirected into K6 ownership. For a rerun choose fresh scratch paths: blind tests append JSONL, concurrency/edge tests refuse an existing output directory, and `install.py` deliberately refuses an already changed preimage. Do not rerun the installer merely to repeat tests.

## Residual limits and coordinator next steps

1. Newly started servers import the installed fix automatically; already running processes retain their imported code. Let current arms finish and use the normal next server start. No new fit, arm spec, client change, remote sync, or serving flag is needed. Any separate server checkout needs this one `plugin.py` replacement.
2. GPU/replay-client smoke was optional and **not run**. Real batching throughput, GPU memory at 24 workers, numerical differences with different batch shapes, stochastic MISS-noise assignment and closed-loop SR/latency are **unverified**. The existing sampler/coordinator is unchanged; this patch does not promise bitwise equality of live stochastic GPU trajectories across different schedules. The exact parity claim here is for the observed fixed-input/key/policy-chunk tests.
3. GR00T still has its own legacy model lock; removing it is outside this task. Its CPU fake-stack test establishes plugin isolation, not live GR00T model concurrency. Existing GPU-wide timing synchronization can also affect measured latency.
4. Custom methods must honor the existing read-only fitted-array contract. R4 shallow-copy fallback now fails clearly at connection construction. The tested deployed methods all deep-clone successfully.
5. Global periodic assignment can interleave differently as requests arrive concurrently; its modulo rule is unchanged. Quantile controller operations are serializable in verdict order. Failed admitted requests leave unused index gaps; duplicate last IDs do not. Episode-level global controller snapshots naturally depend on when an episode ends relative to peers.
6. Coordinator should observe its next normal π0.5 arm for wall time and memory and record real serving latency. The measured 5.09–6.50x fake-policy speed-up establishes removal of plugin serialization, not a forecast of GPU or complete-arm speed-up.
