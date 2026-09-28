# Q5 hand-back: opt-in GPU retrieval

Verified UTC: 2026-09-28T03:32:34.278929+00:00

Implemented and atomically installed `--os-gpu-retrieval shadow|serve` for π0.5 stock AWM and stock guard-only MixedJudge fits. Shadow serves the unmodified CPU Result. Serve substitutes the GPU retrieval/chunk/confidence and feeds a CPU guard/progress state machine, followed by the existing plugin verdict and policy MISS path. No new flag means the existing path and log schema. Both delivered arms are **shadow**. K7/BlindAWM/policy-tail are explicitly unsupported because K9 does not implement anchor updates or the gap-aware endpoint-confirmed stuck feature; no K7 arm is silently substituted.

## Files and installation

Q2 hand-back existed before installation. Its final installed plugin was the exact preimage `f9e65a39d834fa6dabda7025075cb9d58e4503443d0ab29125121b61f7b31e8b`. The candidate was developed and tested under `dev/`, rebased onto that installed file, then installed with a same-directory temporary file, fsync and one `os.replace` at **2026-09-28T03:27:01.915592+00:00**. No running server was restarted.

| File | SHA256 |
|---|---|
| `exp/offline_search/closed_loop/plugin.py` | `758a8f061faa9db75477fad4e66a13cb5cade402e858b159854d9fa5ee187a3b` |
| `dev/gpu_retrieval.py` (runtime import, retain this file) | `1f7feb73c897f8ca5bd805a9c28a69f2e36836370e5ebf0851871731df4e3893` |

Only the shared plugin was replaced. `serve_pi05.py` already passes plugin flags through its parser; it needed no edit. The helper imports the unchanged K9 `gpu_awm.py`. Other deliverables: `arms_q5.json`, `arms_unsupported.json`, `prefit.sh`, `summarize_shadow.py`, `REPRODUCE.md`, candidate/installer scripts, CPU fake-stack and real stage-1/replay drivers, relocated regression recipes, and raw `results/`. `results/deliverable_hashes.json` records source/spec/report modification times and hashes. No src, harness, profile, ops or other-round file was modified. No git command, remote host, simulator, server, LIBERO worker, chain or port was used. Python used CPUs 34–37,78–81 with BLAS/OMP threads 1.

## Serving contract

- Required flags: `--os-gpu-retrieval shadow --os-no-shadow-native --os-tokens off`. Change only `shadow` to `serve` when the coordinator elects to promote. MixedJudge supports per-connection verdict modes; shared quantile control is refused; the measured configuration is `--os-judge guard_only`, with a full model for MISSes.
- K9 float64 projection/distance math, float32 actions, existing double confidence/kernel math. π0.5 construction calls `torch.set_float32_matmul_precision("high")`; using float64 avoids TF32 retrieval drift without changing that process-global setting while another connection runs the model. Tests explicitly leave the model setting enabled. This is a deliberate difference from K9’s float32 headline timing.
- Immutable fitted GPU buffers are shared once per runtime. Each connection owns a CUDA stream, graph, fixed-shape input/output buffers, pinned host buffers and previous visual keys/state/stuck count. Capture/setup has a separate setup mutex; inference has only the K6 connection lock. The runtime decision lock is never held across retrieval or model execution. CUDA capture uses thread-local error checking; no shared graph output or per-query global precision mutation exists.
- CP1 pooling stays on GPU. Retrieval/synthesis/packet packing and previous-key updates are captured. The only retrieval host wait follows the final chunk/feature/state copies. The orchestrator then receives CPU state from that packet, avoiding its otherwise early state D2H. In **shadow**, the CPU reference additionally copies visual keys; shadow deliberately does not claim the serve-path PCIe saving. In **serve**, visual history on the host is NaN, and visual input logging is refused.
- CPU progress guards and the plugin’s verdict remain per connection. Explicit episode reset resets device history on the next step-zero graph replay. A failed GPU decision requires an episode reset before reuse. Unsupported methods/shapes/options fail clearly; no silent CPU fallback exists.
- Separate retrieval graph only. The served staged/coordinator path uses `max-autotune-no-cudagraphs`; it exposes no shared stage-1 graph boundary. K9’s isolated combined-graph experiment demonstrates feasibility, but fusing it into the served coordinator/model would require an additional change and is **not implemented** here.
- Refused: GR00T, blind/K7/policy-tail methods, custom AWM/MixedJudge subclasses, non-joint/non-full-rank or insured AWM fits, MixedJudge blend/recovery/events/burst/non-unit temperature, native shadow, token exposure, shared quantile control, randomized overlays, R4 stage synchronization, non-full camera modes, multiple checkpoints and cache writes. Serve also refuses `--os-log-inputs`.

Per-decision `gpu_retrieval` logs contain top-1, full GPU top-16, top-16 set/order agreement, maximum chunk error with atol 1e-4, confidence error with atol 1e-3, and latency. `event_ms` covers input copies plus retrieval graph; `gpu_path_event_ms` also spans pooling and host launch gaps. Both CUDA-event intervals include any intervening host-launch or GPU scheduling gaps; they are not kernel-only measurements. `wall_ms` ends after final D2H. `cpu_query_ms` measures the CPU method; `cpu_path_ms` additionally includes reference key materialization/input bookkeeping. Agreement fields compare retrieval proposals, including on decisions later served by the policy. Serve logs CPU guard time and null CPU retrieval time; it does not claim CPU agreement without running a reference. Startup reports precision, resident bytes and separate-graph status. `summarize_shadow.py` aggregates these fields and separates step-zero chunk failures.

## Final numerical and latency evidence

Observed hardware (`nvidia-smi`): NVIDIA GeForce RTX 4090, 595.71.05, 49140 MiB. Software: Python 3.11.15, PyTorch 2.7.1+cu126, CUDA build 12.6. Exact records: `results/hardware.txt`, `results/software.json`.

Each row below is **2200 real stored batch-one queries per fresh process**, over cache and inference histories, all ten l10 tasks, 80 episodes, up to 64 decisions per episode. R1/R2 repeat the same queries and are not 4400 independent observations. Recorded executed histories are held fixed; these are not new closed-loop trajectories. Source UID lists and every disagreement are in the raw JSON.

| Family / library | Top-1 R1/R2 | Top-16 set R1/R2 | Top-16 order R1/R2 | Chunk within 1e-4 R1/R2 | Confidence within 1e-3 R1/R2 | Max chunk error |
|---|---|---|---|---|---|---|
| AWM / 50 | 2200/2200 | 2200/2200 | 2196/2196 | 2183/2183 | 2191/2191 | 0.000782519579 |
| AWM / 500 | 2200/2200 | 2196/2196 | 2194/2194 | 2196/2196 | 2198/2198 | 0.0124055743 |
| MixedJudge / 50 | 2200/2200 | 2200/2200 | 2196/2196 | 2183/2183 | 2198/2198 | 0.000782519579 |
| MixedJudge / 500 | 2200/2200 | 2196/2196 | 2194/2194 | 2196/2196 | 2200/2200 | 0.0124055743 |

MixedJudge guard flags and lowest-reason codes matched on 2200/2200 queries in each of the four MixedJudge processes. Every non-step-zero chunk passed atol 1e-4; the largest non-step-zero error was 1.71065331e-05. Strict step-zero chunk/confidence failures remain. **100% top-1 does not establish strict action parity or authorize serve promotion.**

| Family / library | CPU method p50 ms R1/R2 | GPU event p50 ms R1/R2 | GPU wall incl. final copy p50 ms R1/R2 | Shared resident bytes | Final D2H bytes/decision |
|---|---|---|---|---|---|
| AWM / 50 | 1.328 / 1.639 | 0.756 / 0.784 | 0.986 / 1.032 | 52841036 | 1912 |
| AWM / 500 | 1.776 / 1.655 | 0.985 / 0.908 | 1.237 / 1.142 | 204780236 | 1912 |
| MixedJudge / 50 | 2.144 / 1.845 | 1.096 / 1.079 | 1.351 / 1.326 | 61434300 | 2008 |
| MixedJudge / 500 | 2.379 / 2.341 | 1.278 / 1.294 | 1.523 / 1.537 | 276695260 | 2008 |

These are loaded-machine measurements, not K9’s combined-stage incremental .36–.63-ms result. GPU wall excludes resident-query preparation and CPU guards; event includes input copies/graph and excludes final D2H. No speedup in closed-loop SR/IR or full inference is claimed. The packet retains full top-16 IDs/scores and CPU state, hence is larger than K9’s minimal ≈1.4 KB packet. All GPU admissions had ≥23267 MiB free; maximum observed own process GPU memory was 2604 MiB, below 6144 MiB. Each child exited and released CUDA. No OOM occurred.

Four real STAGE1_ONLY checks (AWM 50/500 × shadow/serve) used six real observations each from tasks 0/4/9, stages 2/3 on meta. All 24 pooling checks matched the deployed key builder exactly. Every shadow served chunk matched its CPU reference byte-for-byte; serve retained no host visual keys. These repeat six observations across modes/scales, not 24 independent observations. The existing checkpoint-validated stage-1 cache was read-only. No full stage-2/3 model, websocket or simulator was run. Eight GPU connections matched their serialized action/top-16 outputs exactly in each replay process, with distinct input-buffer addresses. A concurrent construction/replay check created a new graph while a worker replayed an existing graph 32 times; every replay output matched.

## Real stage-1 timing audit

The same final stage-1 checks recorded the following retrieval timings (six observations per row). These are small correctness samples on a shared GPU. Their large variation is material: stored-query replay medians do not establish the live serving latency. No cause of the outliers is established here.

| Library / mode | GPU event p50 / max ms | GPU wall p50 / max ms | Pooling-to-retrieval event p50 / max ms | CPU path p50 / max ms |
|---|---|---|---|---|
| 50 / serve | 6.467 / 7.341 | 6.857 / 7.610 | 7.013 / 7.855 | not run |
| 50 / shadow | 15.479 / 40.662 | 16.353 / 48.026 | 17.199 / 41.088 | 3.109 / 8.811 |
| 500 / serve | 7.063 / 67.772 | 7.369 / 74.954 | 7.553 / 72.866 | not run |
| 500 / shadow | 7.157 / 57.333 | 7.497 / 58.762 | 7.677 / 57.831 | 3.702 / 9.283 |

For the twelve real-stage shadow proposals: top-1 12/12, chunk tolerance 11/12, confidence tolerance 12/12. The failed chunk comparison was step zero; maximum proposal error was 0.000175841153. CPU served chunks remained exact.

## Regression and action-isolation checks

Legacy tests used the final ready-to-install plugin, byte-identical to the atomic installed file; new post-install checks are listed in `results/postinstall.json`. The inherited filenames/argument label `installed` are retained, but the owned development loader maps that label to `dev` unless `Q5_SOURCE=installed`. No result is relabeled as a live rollout.

| Check | Configurations | Decisions/comparisons | Result |
|---|---|---|---|
| K2 legacy matrix | 21 | 963 | PASS |
| K1 legacy matrix | 10 | 480 | PASS |
| K4 legacy matrix | 4 | 156 | PASS |
| K6 concurrency, plus serialized side | 12 | 2196 | PASS |
| K7 plugin | 5 | 240 | PASS |
| K10 plugin | 9 | 432 | PASS |
| K10 concurrency, plus serialized side | 9 | 1647 | PASS |
| Q2 plugin | 8 | 384 | PASS |
| Q2 lifecycle/wire | 8 | 420 | PASS |
| Q2 concurrency, plus serialized side | 8 | 1464 | PASS |
| K7 scalar parity | 4 | 15272 | PASS |
| K7 full-cell count parity | 4 | 189904 | PASS |
| K5 replay / overlay | 4 / 1 | 1368 / 104 | PASS |
| Q5 CPU fake-stack | 8 | 768 primary decisions | PASS |

Q5 covers AWM/MixedJudge × 50/500 × shadow/serve, eight connections and 16 reset episodes per configuration. Shadow’s 384 primary decisions preserve exact CPU HIT chunks and supplied MISS actions. Serve poisons CPU retrieval and checks NaN host visual buffers. Connection-lock ownership and absence of runtime-lock ownership are asserted at retrieval. Each configuration also checks a failed admitted decision, refusal before reset, successful reset recovery, and invalid flags. K6 edges, K7 gap/synthetic edges, K10 transform/lifecycle/ledger and method parity, Q2 contracts, and K5 planted/null/forged-log/cross-scale/ledger checks passed; their full original counters are in `results/final_summary.json`.

Flag absent: fixed time/PID/HEAD and identical invocation/output paths produced identical JSONL bytes including startup:

| Mode | Bytes | Rows |
|---|---|---|
| pure | 63423 | 45 |
| mixed | 91847 | 45 |
| r4 | 145828 | 45 |
| blind | 163520 | 54 |
| rand1 | 744862 | 212 |
| rand2 | 744079 | 212 |
| guard_only | 208365 | 54 |
| threshold:inf | 248593 | 54 |
| periodic:5 | 220177 | 54 |

Development failures retained: the first real-stage fixture imported a CPU selftest that hid CUDA; it failed before CUDA initialization and was corrected by the GPU-only loader. The post-install parity loader was hardened to replace package attributes as well as module-cache entries and assert the exact imported source path; all nine byte comparisons were rerun through it after installation. The pre-install K5 installed-file audit correctly rejected the still-uninstalled candidate; it was deferred and passed after atomic installation. No functional failing final checks are omitted. Earlier float32 exploration remains under `results/dev_float32/`; final GPU tables above use float64.

## Arms, artifacts and exact commands

`arms_q5.json` is emit_arms format for π0.5 l10 50/500 pure-cache AWM CL2 shadow, with exact existing fit metadata. K7 50/500 anchor_tail attempted specs and explicit refusal reasons are in `arms_unsupported.json`. No new deployment prefits were needed, so nothing is created under `/tmp/q5_fits/`. Verification-only fitting stays in owned `/tmp/q5_*` scratch.

| Scale | Candidate rows | Candidate top-level NPY bytes | Fit pickle bytes | Served native pickle bytes |
|---|---|---|---|
| 50 | 2640 | 695839536 | 24631208 | 1103155631 |
| 500 | 29472 | 7768083936 | 94143229 | 1103155631 |

The 500 candidate bank is `bpool_cs`; its derived arrays have no independent native pickle. Both arms retain the native current-library preload for the existing storage contract. The GPU representation contains the chosen fit’s candidates; no cross-scale fitting or outcome information is borrowed. Exact fit paths, SHA256 and kwargs are in `results/arms_validation.json`.

Exact serving option templates (replace `<RUN>`):

```bash
--os-method exp/offline_search/rounds/r02/g1_awm/awm.py:AWM --os-kwargs '{"lib": "current", "kref": 5}' --os-cell pi05_l10_cache --os-log-dir '<RUN>/server_logs/r5q5_p_l10_50_cl2_shadow' --os-root /home/weiland/trace_runs/offline_search_store --os-gpu-retrieval shadow --os-no-shadow-native --os-tokens off --os-fit-artifact /home/weiland/trace_runs/os_closed_loop/r02_g50/fits/oscl50_p_l10_cl2.pkl
--os-method exp/offline_search/rounds/r02/g1_awm/awm.py:AWM --os-kwargs '{}' --os-cell pi05_l10_cache --os-log-dir '<RUN>/server_logs/r5q5_p_l10_500_cl2_shadow' --os-root /home/weiland/trace_runs/offline_search_store --os-gpu-retrieval shadow --os-no-shadow-native --os-tokens off --os-fit-artifact /home/weiland/trace_runs/os_closed_loop/r02_g500/fits/oscl500_p_l10_cl2.pkl
```

Exact executed top-level recipes and subprocess commands are documented in `REPRODUCE.md`, `results/q2_commands.json`, the two regression directories, `run_gpu.sh`, `run_stage1.sh`, and `run_stack.sh`. GPU commands use the repository Python, one process at a time. All tests avoid ports.

Coordinator next: emit these specs into a fresh `<RUN>`, let existing servers finish normally, run short shadow samples on both library sizes with the normal five-control client, and inspect `gpu_retrieval` using `summarize_shadow.py`. Measure live agreement and end-to-end latency before considering serve. No live smoke, rollout outcome, full-policy GPU equality, GR00T GPU port, K7 GPU guard/anchor update, combined stage-1 graph, library growth, or packed seven-column action storage is implemented or claimed.
