# CPU/GPU decoupling handback

2026-10-08; worktree `/home/weiland/projects/openpi_decouple`, branch
`decouple-cpu-gpu`, base `05a32800`. Changes are uncommitted.

The default-off split is implemented and CPU-qualified. The owner's 4090 run
passed all four B=1 numerical gates. The 17:20 smoke reused them and completed
GR00T current/1 at 23.12 episodes/min; split/8 hit the Unix socket length limit.
That startup defect is repaired below; real split throughput remains unrun.
This follow-up used the requested CPU affinity, hid CUDA and used the main
interpreter. Only CPU model-stub children/private Unix IPC were launched;
no TCP model servers, GPU/remote launches or commits were made. Main tree,
R16 roots and running processes were unchanged; inputs were read only.

## GPU package: startup repair and next smoke

Unix stage sockets now live in a unique short 0700 `opi-<hash>-<random>` directory
under a private usable XDG runtime directory, with `/tmp` fallback and a 107-byte
encoded path limit. GPU/fronts share a checked 0600 `stage_endpoint.json` pointer;
its copy remains in each arm's `split_logs/`. Teardown reaps owned children and
removes only that private socket directory. Long trace roots no longer affect
the endpoint length. Blueprint/socket readiness now requires a successful
connect, and websocket waits and group shutdown are bounded.

Startup review also fixed fresh-front import visibility for Hugging Face dynamic
processor classes before blueprint loading, stale inherited front IDs, explicit
loopback binding for the split backend, port-range/startup-timeout validation and
the dry-run CPU mask. Norm assets, vendored cloudpickle and no-JAX GR00T imports
remain checked. Worker bytes, public/driver 23100/23179 and occupied-port preflight
are unchanged. Gate-cache migration excludes only supervisor/address code and
checks every numerical descriptor field unchanged, retaining the same real
GPU UUID/driver; audit receipts are in `results/gate_cache/startup_migrations`.

The relay now reads the worker's actual `__extra__` UID/attempt, records monotonic
timestamps, and clears identity between episodes. Runtime/offline collection
shares one CPU collector; utilization samples, startup peak and arm specification
are now flushed before collection. The missing π0.5 norm stats are staged as a
checked regular local file; the owner's symlink is retained under results, with
no writes through it. The owner's GR00T-only import fix and default four tether
launch retries are retained and checked.

All four passed owner proofs are copied into an input-scoped shared cache.
Benchmark/relay/driver edits, smoke, and `--new` reuse unchanged proofs; changed
model inputs/device requalify in full mode. Owner proofs without UUID are
explicitly adopted from the reported window and bind to UUID/driver on first
reuse. No worker changes or deployment are required.

Next-window smoke (whole card and timan107 fleet free), **5–8 minutes expected,
10-minute hard cap**, including shutdown reserve:

```bash
bash /home/weiland/projects/openpi_decouple/exp/decouple/run_gpu_package.sh --smoke --new --deadline "$(TZ=America/Chicago date -Iseconds -d '+10 minutes')"
```

Gates skip when unchanged, then GR00T current/1 and split/8 each run 20 accepted
B episodes, ten disjoint warmups, 64 workers, frozen spatial/100/k1 method,
through durable collection and `exp/decouple/results/gpu_smoke/SUMMARY.md`.
Missing qualification fails before launches. Use `--new` after this source change
to archive the previous partial smoke locally; unchanged gates still skip.

Full next-window command, **100–116 minutes budgeted** (possibly nearer 90 if
the recovered one-copy rate repeats), two-hour hard cap:

```bash
bash /home/weiland/projects/openpi_decouple/exp/decouple/run_gpu_package.sh --new --deadline "$(TZ=America/Chicago date -Iseconds -d '+2 hours')"
```

Screening remains GR00T current/1 → current/max → split/8 → split/4 → split/16,
then π0.5 likewise, then two alternating best-vs-best repeat pairs per model;
300 accepted B episodes per full arm, common 46,000 MiB budget. Other rates are
planning assumptions. Public/driver **23100/23179**, local 23400–23416; occupied
ports fail without touching listeners. The earlier fixed default 2026-10-08
15:15 Chicago still applies absent an override; the commands above set new
window deadlines. Use the actual owner handoff time when earlier.

Admission skips arms/pairs that cannot fit with `PKG_ARM_SKIPPED_DEADLINE`.
At hard end minus 120 seconds, watchdog stops all owned local groups together,
GPU/relay within ten seconds, driver lease cleanup within the reserve, then
writes a partial summary and `PKG_DEADLINE_STOP` (exit 2). Success is `PKG_DONE`
(exit 0), ordinary gate/arm failures exit 1. Ctrl-C, wait for shutdown, then
rerun without `--new` to resume; use a new deadline for another reservation.
Complete arms skip; incomplete cohorts restart in fresh attempt roots.

CPU re-analysis of the owner's completed arm has been run without modifying
its trace root. `exp/decouple/results/gpu_reanalysis/{SUMMARY.md,metrics.json}`:
**31.0237 episodes/min** (300 / 580.202 s), 23 rollout failures, zero accepted
execution errors. Accepted-cohort policy timing is **1,829.333 / 7,838.896 ms
p50/p99** (6,897 decisions), explicitly a resident timing boundary. Legacy
relay p50/p99 **3,893.239 / 9,667.037 ms** includes warmup/rejected attempts and
cannot be cohort-attributed. GPU/CPU means and full-run peak VRAM are unavailable
because the old sampler was only in memory. The fleet receipt proves a startup
peak lower bound of **6,618.188 MiB**; it is not substituted for whole-run peak.
See GPU_PACKAGE.md for the exact read-only `collect_gpu --allow-legacy` command.

CPU full and smoke dry-runs check assets, 12 helps, actual parsers, deployed
worker/client bytes and GR00T qualifier imports without JAX, plus actual
recorded-worker-frame relay/collector/SUMMARY replay for both models with
interleaving, warmup and retry fencing. Both now also start a real supervisor,
fresh CPU-stub stage child and eight fresh fronts per model beneath >240-byte
runtime parents, with actual blueprint/Unix bind/memfd tensor exchange and
cleanup; websocket/model construction uses stubs and binds no TCP ports.
GR00T uses island site-packages with JAX blocked and a synthetic HF processor
class to test fresh-process imports. **65 package tests pass; the full decoupling
suite passes 100** (`results/decouple_tests.xml`). Reports:
`exp/decouple/results/package_validation.json`,
`package_validation_smoke.json`, `package_tests.xml`; retained wire recordings
and CPU-stub summaries live under both `package_dry_run*/cpu_collection/` roots.
Startup evidence is under `package_dry_run*/cpu_startup/`; socket directories
and all 18 children per dry-run were cleaned up. No GPU/TCP/remote operations
were performed in this follow-up.

See [GPU_PACKAGE.md](exp/decouple/GPU_PACKAGE.md) for proof fingerprints,
raw files, commands, allocation admission and scope of recovered measurements.

## What changed

`python -m openpi.serving.split` runs one weight-owning GPU child, configurable
N weight-free CPU fronts, and a public websocket proxy. It executes the
existing serving entry point in fresh child interpreters. Fronts hide CUDA and
receive the original CPU transforms/config/metadata through a blueprint whose
contents are checked for model parameters and device tensors.

The existing serving, cache, interceptor, plugin and model files are unchanged.
The ordinary entry points install no split hooks, so disabled behavior retains
their original code. `cloudpickle>=3.1.2` is now a declared dependency (it was
already present in the main interpreter and lockfile).

New framework files:

* `split_transport.py`: two bounded memfd slots per inference thread/channel;
  SCM_RIGHTS once, small msgpack descriptors, raw tensor/image bytes in mmap.
  Copies protect slot reuse and preserve bf16/nonfinite bits. No tensor-array
  pickle occurs per stage call. Initial blueprints/key builders use cloudpickle.
* `split_service.py`: channel/epoch-scoped GPU stage handles, original GPU key
  reductions and RNG draws, π0.5's existing stage batcher, and GR00T's backbone
  and k-step head dispatch through BatchingCore. KV caches stay in the GPU child.
* `split_runtime.py`: explicit child-only loader/factory adapters. Connections,
  method copies, histories, retrieval/correction, decisions, transforms, blind
  steps, policy tails and logs remain in fronts.
* `split_state.py`: shared CPU connection IDs, blind decision counter and exact
  quantile window/bisect order. Its process lock covers the same short counter
  and verdict transactions as the plugin; retrieval/model waits do not hold it.
* `split_socket.py`: short private addresses, checked pointer, connect readiness
  and owned-directory cleanup, independent of the persistent run-root length.
* `split.py`: fresh execs, startup checks for every front, sticky routing and
  broadcast controls using the existing ReplicaProxy, and owned-child teardown.

The R16 resident entry point and its minimal import/registry metadata dependencies
were copied from the main tree. `resident.py` has one adaptation: startup receipt
output uses the split log root and unique role names. Plugin decision logs keep
their original tags/content and move into separate role/arm directories.
`exp/trace_dual/config/README_DECOUPLE.md` preserves an existing import-time
repository-root marker needed by R16's preparation imports. Source hashes are
in `exp/decouple/results/r16_source_provenance.json`. This is not a replacement
for the complete R16 experiment/deployment package or its certification.

The serving map, ownership diagram and throughput model are in
[`docs/DECOUPLE.md`](docs/DECOUPLE.md).

## CPU proofs and their limits

**124 tests passed**, including 36 new split tests and 88 existing websocket,
π0.5 interceptor and GR00T stage/interceptor/batcher tests. Evidence:
`exp/decouple/results/cpu_tests.xml`.

New coverage includes:

* Bit-identical raw transport, recorded B-token pooled keys for both models,
  interleaved π0.5 stages at k=1/2/10, and GR00T's real staged runner around a
  complete flow-head stub at k=1/2/8. GPU KV/attention objects remain referenced.
* Original cache interceptors around the remote adapters, explicit π0.5 noise,
  seeded GR00T MISS, and GR00T warm continuation/first-step snapshots.
* Real frozen R16 spatial/100/ir25 methods on B states/keys, retrieval/corrector,
  interleaved histories, blind steps and policy tails. Responses are compared
  as complete packed bytes after freezing nondeterministic telemetry clocks.
  That method tape uses a π stage stub for both horizons; GR00T's model boundary
  and interceptor have separate GR00T-specific tests.
* Independent front processes replaying periodic:3 and quantile decisions for
  both methods, preserving connection/synthetic IDs and controller state under
  controlled admission order; nonfinite quantile behavior is covered too.
* Production websocket dispatch with queued frames: replacement of a same-id
  YAML preserves existing connection K, rebinding/new connections sees new K.
  Shared storage is stubbed; successful broadcast routing uses the unchanged
  existing proxy. Complete multi-port server launch was not exercised here.
* Admission/arena overflow, timeout, front-process abrupt exit/EOF cleanup,
  and rejection of foreign/expired handles after a GPU-service epoch restart.

CPU proof bound: **zero differing bits** in the compared deterministic actions,
keys and responses. This is not a hardware numerical bound. GPU reductions and
random generation stay on the original device. Concurrent unseeded noise can
be assigned in a different request order, as it can in existing concurrent
serving; fixed-noise, controlled-order qualification is required for comparison.
Timing fields/PIDs/startup telemetry are not deterministic equivalence targets.

Supported scope: frozen-write CP1 pooling recipes, CPU offline_search methods,
k-step MISS, ordinary warm continuation, blind/policy tails and dynamic bundles.
GPU retrieval, CP2, trace, warm-reset, online RIT feedback, compiled vision,
camera completion/ablation and stage-1/prefix overrides are explicitly refused.
GR00T YAML K must match the live head K, as in the original path; π0.5 K is
per bundle. One GPU process is supplied per launcher; multiple model/suite
launchers can use separate ports/checkpoints.

## CPU measurements

Affinity: `10-17,54-61`; CUDA hidden; OMP/OpenBLAS/MKL threads=1. Transformation
profiles use 32 rows. Method cProfiles use 64 decisions on two connections.
Unprofiled scaling uses synchronized starts, shared counters, 256 decisions per
front and actual JSONL writes. The two model scaling runs ran concurrently;
these short host measurements are service-demand evidence, not a GPU forecast.

| Component, mean ms | π0.5 | GR00T |
|---|---:|---:|
| Input transform chain, including token preparation | 0.442 | 14.819 |
| Token preparation (included above) | 0.129 | 7.784 |
| Tokenizer alone (subset of GR00T preparation) | — | 1.461 |
| Image processor alone (subset of GR00T preparation) | — | 2.460 |
| Output transforms | 0.140 | 0.618 |
| Wire pack / unpack | 0.159 / 0.052 | 0.144 / 0.056 |
| Model input CPU glue | 0.129 | 0.501 |
| Method query + plugin glue, per vision query, cProfile | 7.040 | 7.671 |
| Distance routine, per invocation, cProfile | 0.919 | 1.094 |
| Corrector, per invocation, cProfile | 1.589 | 1.733 |
| JSONL emit, per record, cProfile | 1.001 | 0.944 |

Nested cProfile timings overlap and must not be summed. The corrector includes
feature/distance work; its predictor alone is about 0.104/0.127 ms. GR00T glue
calls the actual `prepare_input` body with a CPU device and bf16 casts, without
constructing any model modules; it excludes CUDA copies/launches.

The available GR00T B archive contains tokens/state/actions **but no camera
images**. Its preprocessing profile uses recorded π0.5 B 224px images, resized
to 256px outside the timed transform chain. Stored states are normalized keys
used as fixture wire states, not reconstructed simulator states. Consequently
these figures are fixture timings, not paired raw-camera GR00T replay.

| CPU fronts | π method decisions/s | GR method decisions/s | π p50/p99 ms | GR p50/p99 ms |
|---:|---:|---:|---:|---:|
| 1 | 236.3 | 233.4 | 3.33 / 10.62 | 3.48 / 10.09 |
| 2 | 315.7 | 313.0 | 4.58 / 14.35 | 4.72 / 15.80 |
| 4 | 443.6 | 449.5 | 5.15 / 20.11 | 5.46 / 21.00 |

This method replay includes inline stage stubs, plugin/logging/serialization and
shared counters, and excludes real preprocessing/GPU/IPC. CPU memfd echo of
3,301,376 bytes each way costs 1.56/1.65 ms of front CPU and 2.86/2.98 ms mean
round trip for π/GR fixtures. This excludes CUDA transfers and is not a complete
stage pipeline. Profiles, function timings, summaries and IPC reports live in
`exp/decouple/results/`; `.prof` files can be opened with Python `pstats`.

## Reproduce the CPU smoke and measurements

Run from the worktree. These commands open no server listeners:

```bash
taskset -c 10-17,54-61 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/weiland/projects/openpi_decouple:/home/weiland/projects/openpi_decouple/src /home/weiland/projects/openpi/.venv/bin/python -m pytest tests/serving/decouple tests/serving/test_websocket_policy_server.py tests/cache/test_interceptor.py tests/cache/groot/test_groot_staged.py tests/cache/groot/test_groot_stage3.py tests/cache/groot/test_groot_batcher.py tests/cache/groot/test_groot_interceptor.py -q --basetemp=exp/decouple/test-tmp --junitxml=exp/decouple/results/cpu_tests.xml
taskset -c 10-17,54-61 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/weiland/projects/openpi_decouple:/home/weiland/projects/openpi_decouple/src /home/weiland/projects/openpi/.venv/bin/python -m exp.decouple.profile_cpu --out exp/decouple/results --rows 32
taskset -c 10-17,54-61 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/weiland/projects/openpi_decouple:/home/weiland/projects/openpi_decouple/src /home/weiland/projects/openpi/.venv/bin/python -m exp.decouple.bench_cpu --model pi05 --fronts 1,2,4,8 --decisions-per-front 256 --out exp/decouple/results/cpu_pi05
taskset -c 10-17,54-61 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/weiland/projects/openpi_decouple:/home/weiland/projects/openpi_decouple/src /home/weiland/projects/openpi/.venv/bin/python -m exp.decouple.bench_ipc --model pi05 --out exp/decouple/results/pi05_ipc.json
```

Repeat the last two commands with `--model groot`. The profile script uses the
main interpreter for both models and selects island packages before importing
torch for GR00T. It keeps HF dynamic-module cache writes in its worktree output.

## Enable and smoke on the coordinator's GPU slot

**Coordinator commands only; none were run in this CPU session.** Keep ordinary
arguments after `--`. N and batch width are independent. Public port P consumes
internal loopback ports P+1 through P+N; choose an available range. GPU batch
size defaults to 1, wait to 0, channel limit to 128, each arena slot to 32 MiB.

For an existing π0.5 command, prefix it with:

```bash
env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/weiland/projects/openpi_decouple:/home/weiland/projects/openpi_decouple/src /home/weiland/projects/openpi/.venv/bin/python -m openpi.serving.split --model pi05 --fronts 8 --port 24110 --batch-size 1 --wait-ms 0 --runtime-parent /home/weiland/projects/openpi_decouple --log-root /home/weiland/projects/openpi_decouple/exp/decouple/serve_logs/pi05_smoke -- exp/offline_search/closed_loop/serve_pi05.py <the-existing-plugin-and-checkpoint-arguments>
```

R16 is the same wrapper with the module entry point:

```text
... -m openpi.serving.split --model pi05 --fronts 8 --port 24110 --log-root <fresh-worktree-log-root> -- -m exp.offline_search.rounds.r16.resident --registry <B-group.json> --server-tag decouple --resident-suite libero_spatial <existing serve_pi05 startup arguments>
```

The initial plugin parser still requires the existing method/cell/log flags;
the resident registry supplies each arm's actual fitted binding. Worker and
conductor endpoints name only the public port. Keep experimental driver roots
and logs outside existing sealed/running R16 runs. A deployment smoke should
load two YAML bundles, dispatch interleaved episodes through the public port,
replace a same-id bundle, verify old/new connection behavior, and check fronts
have no CUDA model allocation. Front crash or GPU death must terminate the
owned group; restart the launcher, reload bundles and start fresh episodes.

For GR00T, use `--model groot`, the existing `serve_groot.py` or resident module,
the checkpoint's existing `--concurrent --allow-dynamic-bundles` flags and
matching `--denoising-steps K`. Both GPU and front interpreters must use the
same island packages. With the required main interpreter, set PYTHONPATH to:

```text
/home/weiland/projects/openpi_decouple:/home/weiland/projects/openpi_decouple/src:/home/weiland/projects/openpi_ext/third_party/gr00t_n15:/home/weiland/projects/openpi_ext/third_party/gr00t_n15/examples/Libero:/home/weiland/projects/openpi_ext/envs/gr00t_n15_venv/.venv/lib/python3.11/site-packages
```

Alternatively the coordinator can explicitly select matching island executables
using `--gpu-python` and `--front-python`. No environment installation or main
interpreter/package modification was performed here.

## GPU numerical gate and throughput benchmark

`exp.decouple.qualify_gpu` loads one real model, compares B=1 baseline stages
with socketpair split stages at the requested width, uses explicit row noise,
reports key/action bit equality and maximum absolute/relative error, and checks
stable nearest-donor ordering per B key field. It uses the recorded image
fixtures described above; **complete real-camera launcher/method replay is
still required**. B=1 must be bit exact. B>1 requires explicit proposed bounds
and is never silently accepted from an `allclose` default.

```bash
env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/weiland/projects/openpi_decouple:/home/weiland/projects/openpi_decouple/src /home/weiland/projects/openpi/.venv/bin/python -m exp.decouple.qualify_gpu --model pi05 --checkpoint /home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch --k 1 --batch-size 1 --rows 32 --out exp/decouple/results/gpu_pi05_b1.json
```

Repeat k=2/full and GR00T with island PYTHONPATH and
`--checkpoint /data/ckpt/n15_libero_spatial`. For larger widths supply
`--max-abs` and `--max-rel` only after choosing/documenting acceptable limits.
No hardware error bound is claimed in this handback. Any changed actual
method donor, HIT/MISS/blind/tail decision or exceeded action bound blocks that
width. Do not treat the simple per-field donor check as a full method proof.

`exp.decouple.benchmark_gpu` accepts a JSON configuration with `arms`. Each arm
has `name`, `port`, `split`, `model_processes`, `server_command` and
`worker_command` argv lists. Use the unchanged coordinator/worker command with
a fresh B group and either `journal` (conductor journal path) or `summary_json`
containing `episodes_completed`. Strings expand `{port}`, `{backend_port}` and
`{out}`. Server `{port}` is the backend port; worker `{port}` is the public
benchmark relay port. Optional fields: `backend_port`, `server_env`,
`startup_timeout_s`, `worker_timeout_s`.

The script starts/stops only its own newly launched process groups. It records
GPU utilization/memory, server-family CPU/RSS and host CPU at 0.5 s intervals.
The same benchmark relay runs in every arm and records individual successful
infer latency from relay ingress through response send, including backend
serialization and split routing/IPC; remote worker network time is excluded.
When given a journal, only accepted completed UID/attempts enter the latency
sample, and imported/reused attempt-zero episodes are excluded. No worker code
or websocket frame is changed. Relay overhead applies equally to both arms.
For externally recorded client latencies, set `measure_latency_proxy=false`
and supply `decision_latencies_ms` plus `latency_scope` in the summary.

```bash
env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/weiland/projects/openpi_decouple:/home/weiland/projects/openpi_decouple/src /home/weiland/projects/openpi/.venv/bin/python -m exp.decouple.benchmark_gpu --config exp/decouple/benchmark.local.json --out exp/decouple/results/gpu_comparison --gpu-index 0 --gpu-memory-budget-mb 23000
```

Prepare `benchmark.local.json` with the coordinator's current worker/driver
argv. Use identical B episode manifests, frozen fits, model/checkpoint, K,
worker count, replan=5, monitor level and warmup for every arm. Report one-copy
current, the largest current replica fleet fitting the common VRAM budget,
and split N=1/2/4/8/16 with one weight copy. First compare B=1; sweep qualified
batch widths/waits separately. Do at least three sufficiently long paired runs,
alternate ordering, exclude designated warmup episodes consistently, and report
episode failures as well as episodes/min, GPU/CPU utilization, p50/p99 and peak
VRAM. The script writes per-arm raw samples/metrics and `comparison.json`, and
fails an arm exceeding the common budget. Equal budget and model-copy count
are explicit; it does not claim identical peak activation allocation.

## Remaining risks

The real GR00T GPU blueprint exported before the 17:20 bind failure; fresh-front
blueprint load and public multi-port split launch still need the next GPU smoke.
Both full launcher startup paths now pass with CPU stubs and private Unix IPC.
Wider-batch kernel drift, complete model/decision parity and episode throughput
require the coordinator runs above. Full stage-1 host views and GR00T opaque
action-input mappings can carry extra tensor/image bytes; CUDA transfer and GPU
RPC-thread CPU costs may limit throughput despite the removed front GIL limit.
Front CPU fit/library copies consume host RAM; shared clocks/quantiles serialize
short transactions. Measure those costs before increasing N without a bound.

Channel/handle/arena limits and timeouts fail with errors, never fallback actions
or automatic retry. The supervisor deliberately fails the whole owned group on
a child exit. Transparent model restart or migration of live connection history
is not implemented. Log locations differ, and existing R16 analysis needs to
consume/stitch the isolated role directories; this task preserves worker dispatch
and log contents, not the old experiment filesystem layout.
