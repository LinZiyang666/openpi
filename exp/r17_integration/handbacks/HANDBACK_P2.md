# P2 handback — MetaWorld / conductor / workers / π0.5-MW

Date: 2026-10-09. Shared checkout: `/home/weiland/projects/openpi`, branch `Ziyang`.
Interface: **`r17.contracts.v1`**, benchmark version `r17.v1`, P6 bridge `r17.p6.v1`.

**Initial delivery: CPU integration, with π0.5-MW as the first end-to-end fixture.**
The initial owned suite was **51 passed, 0 failed, 0 skipped**, exit 0.
The stock WebSocket server, P1 connection factory/cache/layers, P2 conductor and
runner, and P6 model/physical writers are exercised together. The neural
velocity/prefix and environment in that test are explicitly synthetic CPU
fixtures; original π native sampler methods and original policy transforms are
used. **No actual checkpoint, renderer, GPU, deployment, or policy SR was
qualified today. G1 real mechanical qualification and G4 remain open.**

## Owned files and actual change

All P2 implementation files are new; no shared or frozen implementation was
edited by P2. Other packages' shared-checkout changes are outside this handback.
No commit/push, foreign process/session operation, or protected-path operation.

| Owned files under `exp/r17_integration/` | Change |
|---|---|
| `benchmarks/spec.py`, `metaworld.py` | Formal benchmark identity/manifest, canonical splits, original-kernel runner with default-off observation/physical taps, A-compatible noise, execution receipts and optional semantic envelope. |
| `benchmarks/physical_mt50.py` | Native MuJoCo model/data capture, numeric copies, entity ID catalog, contacts/force callback, explicit unsupported semantic/restore statuses. |
| `benchmarks/libero_bridge.py` | Lazy original `LiberoEpisodeRunner` factory; no simulator change. |
| `conductor/manifest.py`, `queue.py` | Immutable episode namespace, split and endpoint compatibility, global least-busy queue, durable attempts before dispatch, resume/fences and probe validation. |
| `conductor/strategy.py`, `driver.py` | Original stage graph/driver plus bundle preload, complete paired per-task descent, first-failure stop, immutable freeze/test authorization, accepted-attempt join, retry-exhaustion journal receipt, bounded smoke CLI. |
| `conductor/worker_entry.py` | Original renderer/`WorkerLoop`/`WorkerAgent`; CPU import probe and explicit file list; real attestation hook; optional P6 physical capture; close writer even when attestation fails. |
| `models/pi05/adapter.py` | Lazy original loader/transforms/stages, one-real-view key, native K10/K2/K1 and suffix snapshots, original inverse transform once, capture/trace/batcher hooks, content-bound model manifest including local tokenizer. |
| `models/pi05/provider.py`, `serve.py` | Explicit P1 provider/factory registration; stock concurrent server and validated bundle controls; P6 store → P1 five-layer fit; connection-local capture with actual native sampler binding; CPU manifest/config emission; normal owned-server SIGTERM shutdown. |
| `tests/benchmarks/test_metaworld.py` | Roster/split/RNG, fake original-kernel parity, terminal/error/actual-control cases, native physics copy/capability checks. |
| `tests/conductor/test_dispatch.py` | Concurrent dispatch, attempts/fsync/fences/resume/descent, real TCP WorkerLoops/census, isolated client import, renderer lock fixture, probe rejection and writer failure cleanup. |
| `tests/models/pi05/test_pi05.py`, `test_pi05_e2e.py` | Native-sampler/transform parity, keys/snapshots/K/tails, content/tokenizer sealing, model-free defaults; real WS two-client/two-YAML conductor/writer/store integration. |
| `tests/models/pi05/build_samples.py`, `samples/` | Reproducible labelled synthetic CPU model/physical/spec samples; never checkpoint/renderer load. |
| `tests/conductor/seal_handback.py`, `cpu_p2_*.xml`, `SOURCES_P2.json`, `RECEIPT_P2.json` | CPU-only receipt/source sealing helper, actual test receipts and post-test SHA snapshot. Earlier red XMLs are diagnostic iterations, not passing certificates. |
| `REQUESTS/P2_to_P1.md`, `P2_to_P3.md`, `P2_to_P5.md`, `P2_to_P6.md` | Cross-package seams, reproductions and resolved/pending requests. |

## Contracts, default-off proof, and decisions

* `MetaWorldBenchmarkAdapter.metadata()` returns P1's common `BenchmarkSpec`;
  `spec.identity` delegates to that contract. `make_episode_runner(client_factory,
  capture_sink, execution_ack, framework_context=False)` composes the original
  `MetaworldEpisodeRunner` and **unchanged** `exp/metaworld/env.py::run_episode`.
* Capture, execution callback and framework context default off. Imports register
  no provider and load no torch/JAX/model/simulator. The real R17 worker explicitly
  enables the semantic request/execution envelope. P1 hooks in shared files are
  optional and owned by P1.
* The fixture compares original/new/capture-on runner action bytes, observation
  frame/state bytes, control lifecycle, success/termination/truncation/cap, and
  original per-decision values excluding wall clocks. Assertions also compare
  unchanged original tasks/env/runner/native π/worker/driver source bytes to
  `git show HEAD:<path>`.
* Original `Policy` versus adapter uses the same explicit noise and original
  quantile transforms: wire dtype and bytes match for **K10/K2/K1**. Wire can be
  float64 because the original quantile inverse uses float64 stats. P1 fixed a
  shared float32 coercion after P2's real WS witness; **the temporary P2 wire
  shim is removed**. Only normalized cache values are float32.
* Formal protocol is seed7, corner2 ID2 position `(0.75,.075,.7)`, RGB480,
  contiguous rotation180°, settle15, max160 policy controls, replan5, **50×10 =
  500** task/original-index identities. Test idx0–9; library10–29; dev30–49.
  The original September `EPISODES_PER_TASK=20` is unchanged and never drives
  R17 formal counts. Init is `MT1(task, seed7).train_tasks[original_idx]`;
  original index is not a reset seed. Duplicate push/push-back prompt strings
  retain distinct task IDs. Groups are 28/11/6/5.
* Formal noise exactly follows A:
  `int.from_bytes(sha256(f"mt50:7:{task}:{idx}:{seq}")[:4], "little")`, then
  NumPy PCG64 `standard_normal(...).astype(float32)`. Arm, campaign, worker,
  retry and replica do not change this initial noise. The CPU test covers all
  50 tasks, four indices and three decisions plus arm/retry/campaign variants.
* Native task data is hashed before original `episode_start` when using the
  original/env renderer factory. A provided sealed task hash must match actual
  task bytes. Synthetic factories must supply honest fixture provenance.
* Dispatch receipts fsync before return. Restart increments dispatch generation;
  stale/duplicate/error results do not enter an error-free dev cohort. Torn
  dispatch files refuse resume and are preserved for explicit coordinator
  recovery. Retry exhaustion becomes an explicit terminal journal error.
* YAML preload only sends bundles referenced by an endpoint's compatible episodes,
  including live-K compatibility; unrelated checkpoints never receive that YAML.
  The core-aware π capture callback selects P6 writers by exact actual sampler
  binding, with a connection-local episode resolver. K2 records are not bound as
  K10 teacher data. No writer/storage implementation is copied into P2.
* Descent uses paired **n_dev=20** by default, gate
  `100*(candidate_successes-full_successes) >= -5*n_dev`, first failure keeps
  the preceding accepted level. Complete reference/candidate identities and
  increasing attempts are required. Resume reconstructs accepted dev outcomes;
  terminal errors refuse cohort promotion. Freeze binds choices and identities;
  test authorization rejects wrong hashes, changed identities and tampering.
* Live CLI is intentionally restricted to `--smoke`; `--plan-only` can emit
  formal plans. Formal execution must come from the coordinator's qualified
  frozen strategy, not the two-arm mechanical smoke launcher.
* The live conductor enables `validate_worker_probe` **before dispatch** and
  rejects changed reconnect probes; rejected probes are recorded separately.
  Direct driver fixtures leave validation off so CPU renderer stubs remain
  explicit fixtures. Real expected versions: MetaWorld3.0.0, MuJoCo3.14.0,
  Gymnasium1.3.0, native EGL/NVIDIA plus camera/protocol/task-data attestation.

## CPU commands and real results

Every pytest command used this prefix (CUDA hidden, JAX CPU, BLAS/OMP one thread,
no pyc and no pytest cache):

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src .venv/bin/python
```

Append the following actual arguments to that prefix:

```bash
-m pytest exp/r17_integration/tests/benchmarks \
  exp/r17_integration/tests/conductor exp/r17_integration/tests/models/pi05 \
  -p no:cacheprovider \
  --basetemp=exp/r17_integration/tests/conductor/scratch_p2_binding_release \
  --junitxml=exp/r17_integration/tests/conductor/cpu_p2_binding_release.xml -q --tb=short

-m pytest exp/r17_integration/tests/common exp/r17_integration/tests/benchmarks \
  exp/r17_integration/tests/conductor exp/r17_integration/tests/models/pi05 \
  exp/r17_integration/tests/debug exp/r17_integration/tests/search exp/r17_integration/tests/ops \
  -p no:cacheprovider \
  --basetemp=exp/r17_integration/tests/conductor/scratch_p2_certified \
  --junitxml=exp/r17_integration/tests/conductor/cpu_p2_certified.xml -q --tb=short

-m pytest tests/serving/test_policy_identity.py tests/serving/test_per_step_recorder.py \
  tests/serving/test_replica_proxy.py tests/serving/test_router_bundle_rebind.py \
  tests/exp/metaworld/test_env.py tests/exp/metaworld/test_runner.py tests/conductor \
  exp/mw_finetune/tests exp/offline_search/rounds/r16/test_conductor.py \
  exp/offline_search/rounds/r16/test_groot_servers.py \
  exp/r17_integration/tests/common/test_default_off.py \
  -k 'not same_identity_same_initial_state_across_processes' -p no:cacheprovider \
  --basetemp=exp/r17_integration/tests/conductor/scratch_p2_legacy \
  --junitxml=exp/r17_integration/tests/conductor/cpu_p2_legacy.xml -q --tb=short

-m pytest tests/serving/test_replica_proxy.py -m env_dependent --run-manual \
  -p no:cacheprovider \
  --basetemp=exp/r17_integration/tests/conductor/scratch_p2_proxy_sockets \
  --junitxml=exp/r17_integration/tests/conductor/cpu_p2_proxy_sockets.xml -q --tb=short

-m exp.r17_integration.tests.models.pi05.build_samples
```

| Receipt | Actual result / scope |
|---|---|
| `cpu_p2_binding_release.xml` | **51 passed, 0 failed, 0 skipped**, exit0, 21.00s. Final P2 implementation, including failed-attestation cleanup, compatible YAML preload and actual-K capture bindings in the real WS test. |
| `cpu_p2_certified.xml` | **184 passed, 0 failed, 0 skipped**, exit0, 33.99s. P1/P2/P6 cross-package suite; 49 P2 cases included. Run preceded final local cleanup/preload/per-sampler capture changes, all covered by the final P2 suite. |
| `cpu_p2_legacy.xml` | **242 passed, 6 skipped, 1 deselected**, exit0, 23.57s. The one deselected case creates real EGL across processes and was deliberately excluded. |
| `cpu_p2_proxy_sockets.xml` | **6 passed, 0 skipped, 23 deselected**, exit0, .14s. Actually executed the six skipped local socket cases with owned fake backends. |
| Earlier `cpu_p2_shared_dtype.xml` / `cpu_p2_sealed.xml` | 47 / 239 passes respectively at those source snapshots. Shared agents subsequently consolidated tests; final receipts above are authoritative. Do not add overlapping counts. |

The receipt helper compiled **18 owned Python files** without pyc and verified
**11 reused tracked sources byte-identical to HEAD**. A's renderer baseline is
untracked in this checkout: its SHA is recorded, but HEAD byte parity is not
claimed for that file. The helper command was
`env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m exp.r17_integration.tests.conductor.seal_handback`.

The only remaining excluded legacy case is real EGL, which would violate today's
zero-GPU rule. No GPU marker was enabled. Import warnings about pynvml/SWIG are
not GPU execution. Ruff is unavailable in this venv; no lint pass is claimed.
`SOURCES_P2.json` records the **post-test** source snapshot, not a pre-run seal of
other agents' concurrently changing files. Red intermediate XMLs remain as
diagnostics. In particular P6 tightened nonempty provenance while packages were
working: the final test supplies real fixture campaign/arm/generation values
instead of weakening P6 validation.

## Hook / support table (F / M / B)

Statuses below mean tests run here, not real-model or renderer qualification.
P6 owns the comprehensive INVENTORY matrix and transport/analysis tool receipts.

| Inventory / class | Hook and framework path | Status / limit |
|---|---|---|
| S01/S02/S09/S10/S11, F+M+B | P1 provider + original concurrent WS + connection factory; original lifecycle/select/load controls | CPU real WS, two clients/two YAMLs, bad identity/unknown bundle/connection isolation tested. No GPU concurrency claim. |
| S03/S04/S05, F | Original endpoints/proxy; P2 queue compatibility, original controls | Queue CPU tested; six real localhost proxy cases pass. No multi-GPU replicas launched. |
| S06/S07/S08, F+M | `make_stage_batcher()` returns original `Pi05StageBatcher` | Hook exposed; generic R17 executor does not attach it. Capabilities explicitly unsupported, not dynamic batching qualified. |
| S12, M+B | Content manifest of checkpoint files, norm, local tokenizer and selected native source; loaded tokenizer verification | CPU same-size weights/norm/tokenizer tamper and missing-tokenizer fail-before-load tested. Real immutable loader pending. |
| S13/S14, F+M | Original server health/timing; original loader/device; no new model loop | Shared mechanism reused; compile/stage placement/device profiling not qualified here. |
| P01/P02, M | Original prepare/S1/S2/S3/from; first256 real camera tokens → adaptive 4×4 pool →32768; state4 | CPU native sampler/transform/pool parity. Wrists excluded from retrieval; native image encoder still computes masked wrist slots. Other representations unqualified. |
| P03/P04, F+M | Original orchestrator/components + adapter-native named key builder | CPU CP1 full/MISS/HIT/tail and rejection paths. No blanket qualification of all Qdrant/gate/judge/search families. |
| P05/P06/P07/P08, F+M+B | P6 sealed `LibraryView` → P1 `LibraryData/fit_library`; AWM/guard/LOEO corrector/periodic forks | CPU synthetic fit/store/WS consumption passes. No algorithm copied into P2. Full 50-task teacher data and quality pending. |
| P09/C06, F+B | `PerTaskDescent`, `MetaWorldDescentStrategy` over stock stage graph | 50-task synthetic first-failure/freeze and real driver barrier/resume tests; full/candidate error-free cohorts; test outcomes never used to fit/select. |
| P10/P11/P12, M+B+F | Native K10/2/1; P1 actual-execution tail; H5 layout and commit binding | CPU byte/NFE/grid parity; commit10 rejected. Replan1 separate protocol consumes at most five source actions before another full call; tail has zero model calls. |
| P13, M | NativeSnapshot + `run_stage3_from` suffix | CPU K10/K2 snapshot suffix exact, wrong identities/grids/dtypes rejected; GPU parity pending. |
| P14/P15/P16, F+M+B | Warm reset/CP2/RIT/routing/twin traces/alternative camera recipes | Unsupported in generic factory/capabilities. Native capture alone does not certify these executors. |
| D01–D04/D07/D10/D12, F+M | Original recorder/H5/trace/warmup tools; `make_trace_adapter()` observe-only | Native result/capture hook provided; advanced twins/augmentation/calibration combinations not qualified by P2. P6 owns tool integration. |
| D05/D06/D08/D09, F+M+B | Original per-step rows + optional semantic receipts; P6 DecisionSink/PhysicalSink | CPU real WS server+client capture, writer/readback, native fixture floats and requested/applied action separation pass; defaults off. |
| D11/D13/D14/D15, F+M+B | Original attempts + P6 store/writer/reader + P1 library fit | CPU identity/LOEO/store bindings consumed. Closed actual-checkpoint library generation, transport and certification pending P6/coordinator. |
| O01/O02/O03, F+M+B | Common layout/query view and P6 fit/replay bridges; frozen harness unchanged | Cross-package CPU tests pass. Original frozen memmap/dimension assumptions are not rewritten or presented as MW support. |
| C01/C02/C03, F+B | Original Driver/Scheduler/WorkerLoop/Journal with local queue shim | 100 concurrent pulls, compatible load balancing, fsync/attempt restart/fences; two real TCP WorkerLoops execute four fake-physics episodes (28 actual fixture controls). |
| C04/C05, F+B | Original stage/WorkerAgent/census; explicit R17 probe gate | CPU stage/probe/census/resource-close tested; actual renderer attestation/remote leases pending. |
| W01, B | Lazy original LIBERO runner factory | Thin bridge only; real LIBERO renderer not tested by this package. |
| W02/W03/W04/W05/W06, F+B | Original worker/agent + A renderer lock; P6 SHA client bundle/ops | Isolated copied-file Python3.11 import and injected first-render/close lock pass. P6 debug payload profile exists; no remote/EGL/deployment run. |
| A01/A02/A03/A04, F+M+B | Stage/NFE/cursor receipts + native physical arrays feed P6 tools | CPU fixtures supplied. Real prices, SR, semantic physics cards and actual latency pending; no LIBERO prices reused as MW prices. |
| X01, F+M | `cpu_blueprint()` | Explicitly unsupported; no GPU objects put in a fake CPU blueprint. |

## Native grid / norm / feature / storage samples

Permanent sample: `tests/models/pi05/samples/native_k2/sample_manifest.json`; canonical
`formal_500.json`; model writer directory `writer/`; native fixture
`physical_fixture.npz`; model-free wire input `wire_observation.npz`.
These are **synthetic CPU**, with no real checkpoint, tokenizer, image encoder,
physics or renderer. A synthetic three-control execution acknowledgment illustrates
the P6 join; it is not a real MetaWorld episode. Paths in this paragraph are
relative to `samples/native_k2/`; that writer has an explicit K2 sampler binding.
The sample builder verifies closed P6 reader roundtrip, accepts a fresh `--output`
and validates/reuses an existing completed sample. Earlier outer `samples/`
artifacts are diagnostic iterations and are excluded from the final sample SHA set.

* π H5, model action width32, active channels0–3, motion0–2, gripper3;
  original quantile inverse then original MW output projection exactly once.
  Checkpoint-local norm required; no fallback to another benchmark's stats.
* One real `corner2` feature32768 plus `robot_state`4. `prefix_embeddings` retains
  original three image slots and language tokens; only real camera tokens enter
  retrieval. Sample quantiles are labelled fixture values, not MW checkpoint stats.
* Native Euler dt is `float32(-1/K)`, with native float32 accumulation from1.
  K2: `(1.0,.5)`, dt−.5. K1: `(1.0)`, dt−1. K10 points/deltas in the sample
  manifest preserve native rounding, not ideal decimal or truncated K10 grids.
* Snapshots bind checkpoint/norm/layout/sampler/grid/index/native_t and finite
  float32 `x_t(5,32)`. K1 has no interior suffix snapshot. K10/K2 resume follows
  original native sampler methods and reports actual remaining NFE.
* Model capture uses copies of actual stage outputs: prefix/mask, normalized
  state, base model input, initial noise, normalized/wire actions, keys and native
  snapshots. HIT/tail do not rerun stages for debug. P6 stores these in existing
  numeric blocks and retains binding/provenance.
* Native physical arrays: qpos/qvel/act/ctrl/xpos/xquat/cvel/site_xpos,
  raw observation, contact distance/position/frame/force are float64 copies;
  geometry IDs int64. Original requested wire action and clipped float32 controller
  action are separately stored as float64. No `.sim` dependency.
* Reach/reach-wall `_target_pos` map is explicit and CPU-fixture tested only.
  Catalog records all native IDs/names and free-joint bodies. Predicates, robot/
  eef/gripper semantic roles, substeps, asset digest and restore certificate are
  explicitly missing/unsupported; other task goals are not inferred from text.

## Future GPU commands — NOT EXECUTED

Coordinator runs only after an exclusive free GPU/EGL slot is assigned, one model
at a time. Each slot below is independently bounded to **≤900 seconds**. Do not
load a live training folder or `latest`. Coordinator supplies immutable
`P2_CHECKPOINT`, pre-cached tokenizer via `OPENPI_DATA_HOME`, qualified Python
islands, fresh `P2_RUN`, reserved ports and routeable endpoint/driver addresses.
P6 supplies a sealed `P2_TEACHER_STORE` matching the emitted binding, labelled
synthetic if using toy data. A toy store proves mechanics, not SR.

CPU preflight (no model load; emit the new tokenizer-bound manifest first):

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=.:src:packages/openpi-client/src "$P2_MODEL_PYTHON" \
  -m exp.r17_integration.models.pi05.serve --checkpoint "$P2_CHECKPOINT" \
  --emit-configs "$P2_RUN/configs"
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=.:src:packages/openpi-client/src "$P2_MODEL_PYTHON" \
  -m exp.r17_integration.conductor.driver \
  --manifest "$P2_RUN/configs/model_manifest.json" --endpoint "$P2_ENDPOINT" \
  --full-yaml "$P2_RUN/configs/k10.yaml" --k2-yaml "$P2_RUN/configs/k2.yaml" \
  --output "$P2_RUN/plan" --smoke --plan-only
```

Slot A: one checkpoint load, native K10/K2/K1 and capture parity on a synthetic
valid wire observation. No server/worker/renderer in this slot. Startup budget
180s; receipt only written after all assertions. Reuse actual recorded observations
later for representative pricing; these inputs prove interface mechanics only.

```bash
timeout --signal=TERM 900s env CUDA_VISIBLE_DEVICES="$P2_GPU" JAX_PLATFORMS=cpu \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src:packages/openpi-client/src \
  "$P2_MODEL_PYTHON" - <<'PY'
import json, os, time
from pathlib import Path
import numpy as np
from exp.r17_integration.models.pi05.adapter import load_adapter, sampler_for_steps, content_manifest
from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC, decision_seed
from exp.r17_integration.common.contracts import CaptureSpec, RequestContext
from exp.r17_integration.common.request_rng import RequestRNG
from exp.r17_integration.common.serving import full_call
started = time.monotonic()
a = load_adapter({'checkpoint': os.environ['P2_CHECKPOINT'], 'device': 'cuda:0'})
assert time.monotonic()-started <= 180, 'partial: startup exceeded three minutes'
p = Path('exp/r17_integration/tests/models/pi05/samples/native_k2/wire_observation.npz')
with np.load(p, allow_pickle=False) as f:
    obs = {'observation/image': f['image'].copy(), 'observation/state': f['state'].copy(),
           'prompt': str(f['prompt'])}
r = RequestContext('p2_gpu_native', FORMAL_SPEC.identity, 'reach-v3', 0, 0,
                   arm='qualification', attempt=1, bundle_generation='native_gen1', task_id=43,
                   explicit_seed=decision_seed('reach-v3', 0, 0))
rows = []
for k in (10, 2, 1):
    a.policy._sample_kwargs = {'num_steps': k}
    a.policy._staged_inference = k == 10  # stock helper otherwise fixes K10
    baseline = a.policy.infer(obs, noise=RequestRNG(r).normal((5, 32)))['actions']
    plain, wire = full_call(a, obs, r, sampler=sampler_for_steps(k))
    captured, debug = full_call(a, obs, r, sampler=sampler_for_steps(k), capture=CaptureSpec(True, True))
    assert baseline.dtype == wire.actions.dtype == debug.actions.dtype
    assert baseline.tobytes() == wire.actions.tobytes() == debug.actions.tobytes()
    assert plain.actual_nfe == captured.actual_nfe == k
    assert wire.actions.shape == (5, 4) and np.isfinite(wire.actions).all()
    rows.append({'K': k, 'actual_nfe': k, 'wire_and_debug_byte_parity': True})
receipt = {'status': 'passed', 'device': 'gpu', 'synthetic_wire_input': True,
           'quality_sr_claim': False, 'rows': rows, 'elapsed_s': time.monotonic()-started,
           'model_manifest': content_manifest(os.environ['P2_CHECKPOINT'])}
out = Path(os.environ['P2_RUN']) / 'native_gpu_receipt.json'
out.write_text(json.dumps(receipt, sort_keys=True, indent=2)+'\n')
PY
```

Slot B: one inference process plus **two timan107 Python3.11 client workers**.
Use ≤3min startup, ≤7min two mechanical episodes, ≤2min closed writer/journal
validation, ≤3min SHA recovery/normal shutdown. Start driver and workers once
server health/identity is verified; no unlimited startup or episode retry.
All YAML paths here must also exist on the server because stock preload sends
`yaml_path`. Serve on a coordinator-selected routeable host/port; do not assume
the server's localhost is reachable from timan107.

```bash
# Inference host, this owned process only. timeout sends normal SIGTERM at 900s.
timeout --signal=TERM 900s env CUDA_VISIBLE_DEVICES="$P2_GPU" JAX_PLATFORMS=cpu \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src:packages/openpi-client/src \
  "$P2_MODEL_PYTHON" -m exp.r17_integration.models.pi05.serve \
  --checkpoint "$P2_CHECKPOINT" --device cuda:0 --host "$P2_SERVER_BIND" \
  --port "$P2_SERVER_PORT" --config "$P2_RUN/configs/k10.yaml" \
  --recipe-store "$P2_TEACHER_STORE" --debug-directory "$P2_RUN/model_capture"

# Driver host; original TCP transport, exactly idx0 full + idx1 retrieval.
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=.:src:packages/openpi-client/src "$P2_MODEL_PYTHON" \
  -m exp.r17_integration.conductor.driver --smoke --deadline-seconds 420 \
  --manifest "$P2_RUN/configs/model_manifest.json" --endpoint "$P2_ENDPOINT" \
  --full-yaml "$P2_RUN/configs/k10.yaml" --k2-yaml "$P2_RUN/configs/k2.yaml" \
  --output "$P2_RUN/episodes" --bind-host "$P2_DRIVER_BIND" --bind-port "$P2_DRIVER_PORT"

# timan107: run each command in an owned process; no model packages/weights.
timeout --signal=TERM 600s env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=.:src:packages/openpi-client/src "$P2_WORKER_PYTHON" \
  -m exp.r17_integration.conductor.worker_entry --worker-id p2_pi05_mw_0 \
  --server-key "$P2_ENDPOINT" --driver-host "$P2_DRIVER_HOST" --driver-port "$P2_DRIVER_PORT" \
  --renderer egl --egl-device 0 --debug-directory "$P2_WORKER_RUN/physical_0"
timeout --signal=TERM 600s env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=.:src:packages/openpi-client/src "$P2_WORKER_PYTHON" \
  -m exp.r17_integration.conductor.worker_entry --worker-id p2_pi05_mw_1 \
  --server-key "$P2_ENDPOINT" --driver-host "$P2_DRIVER_HOST" --driver-port "$P2_DRIVER_PORT" \
  --renderer egl --egl-device 0 --debug-directory "$P2_WORKER_RUN/physical_1"
```

P6 `ops.build_client_bundle.worker_payload_files(root, debug=True)` supplies the
debug-capable payload; base `CLIENT_ONLY_FILES` alone is the import profile, not
all runtime websocket/debug dependencies. The worker attestation creates/closes
one owned environment and original lock serializes first-render/close. Workers
normally exit on stock MSG_SHUTDOWN; timeout is a deadline fallback for owned
processes. Coordinator stops the exact owned server process normally when both
workers/data are closed, retaining its PID/command receipt. No process name scan.

Before a Slot B pass: require exactly two accepted, error-free journal episodes,
two distinct worker IDs/probes, correct task/original indices/attempts, model/norm/
layout/YAML hashes, real RGB480/state4/wire4/H5 and actual cursor/NFE receipts.
Read P6 closed captures in `model_capture/binding_*` with checksum/binding validation; enrich stock journals
with exact taskdata/attempt identities before accepted joins. Confirm at least
one actual retrieval HIT if claiming HIT support; a K2 MISS alone does not prove
it. Late attempts/errors remain diagnostic. Flush and SHA-copy only closed data.
Deadline/startup/renderer/incomplete join gives **partial/failed**, not a passing
receipt. This slot does not cover full500, real library quality or GPU batching.

## Known limits / pending items with owners

| Item | Status and owner |
|---|---|
| Actual π checkpoint loader, real encoder/LLM/full-vs-staged K/debug/suffix parity | **Unexecuted**; coordinator schedules P2+P1 one-model GPU slot. Local tokenizer must already be present and content sealed; loader refuses missing content before model load. |
| Actual MW/EGL worker/GL/camera/taskdata/native force/goal mapping | **Unexecuted**; P2 hooks ready, coordinator+P6 schedule worker qualification. All 50 task role maps and goal/restore certificates remain P2 semantic work; unsupported statuses are retained. |
| Asset byte digest, restore, robot/eef/gripper roles, actuator substeps | Explicit unsupported; P2 supplies certificates/maps later, P6 consumes without guessing truth. |
| Final control acknowledgment at episode termination | Last actual controls persist in client execution/physical rows; current wire envelope sends the preceding ack on the **next infer**. There is no extra terminal infer/RPC. P1+P6 own any later explicit terminal-ack control/reader enrichment; do not treat unknown server final executed count as zero. |
| Multi-model worker debug binding | Current CLI attests/captures one endpoint model per worker/campaign. A fleet switching between distinct ModelSpecs needs per-task P6 binding/bridge selection before claiming mixed-model capture. P2+P6; same-model compatible endpoint dispatch is CPU tested. |
| Stage batching/compile/stage placement, CP2, warm reset, routing, twin traces, split | Not advertised supported in generic executor. P1 owns framework executor; P2 supplies/qualifies native hooks. Native suffix is implemented, not a warm-reset certificate. |
| Complete five-layer quality, native teacher library capture, augmentation/replay/tool coverage | CPU synthetic recipe/store integration only. P6 owns library/reader/replay/analysis/transport acceptance; P1 owns sole math; P2 owns model/benchmark hooks. No new model prices or SR inferred. |
| Formal full500/library/dev/freeze/test and benchmark prices | Coordinator G4 after immutable final models and G1/G2/G3. Formal smoke CLI cannot launch an unfrozen test; controller/frozen strategy API available. |
| Core-aware capture factory shim | P2's thin callback chooses the native sampler binding and supplies original core episode metadata to P6 writers. P6 CaptureFactory currently takes only a request resolver; migrate this callback when P6 publishes a core-aware resolver. Writers/reader/storage stay P6-owned. |
| Generic scheduler private shim | Queue and skipped-descent-stage handling access stock private scheduler state in owned files. CPU barrier/resume tests cover current API; P1 publishes any future supported shared hook. Frozen R16/harness never changed. |
| Client deployment and remote closed-result join/SHA | CPU copied-file/ops fixtures only; P6 owns actual payload installation/tether/receipt and normalization of stock journal + per-attempt taskdata metadata. |
| Third-party model island provenance | Manifest seals selected native source/checkpoint/tokenizer content; third-party torch/transformers/runtime environment attestation belongs to P6/coordinator qualification. |

Requests addressed to P2 were checked initially and throughout implementation.
P1 v1 common interfaces/capture/float64 fix are consumed; P6 writer/profile seam
is connected; P3/P4/P5 canonical MW aliases/protocol/K replies are recorded.
P6 library builds for hashed formal protocol must specify `split_policy='mt50_init_v1'`;
the model binding alone does not establish initial-state split boundaries.
No shared-file change remains a P2 edit. Outstanding items above are named and
not converted into unsupported success claims.

## Fix round 1

2026-10-09. Read both `review/REVIEW_A.md` and `review/REVIEW_B.md` in full,
including LIBERO identity/ack repros. This section supersedes the initial
process-level recipe CLI, terminal-ack limitation, default-K worker capture, R17 MW context default,
and thin-only LIBERO bridge statements above. All changes stay in P2-owned
R17 directories; cross-package work went through `REQUESTS/P2_to_P{1,3,4,5,6}.md`.
**CPU only: no GPU, real renderer, real checkpoint, git commit/push, or foreign
process operation. Training sources, frozen runners and R16 results were not modified.**

### Finding → change → regression → status

| Finding | Change in owned code | Behavioral regression | Status |
|---|---|---|---|
| B1 blocker, LIBERO execution/commit | `benchmarks/libero_bridge.py` wraps original `LiberoEpisodeRunner` and exact original `_run_episode`. Observed env counts actual controls, sends preceding acks and final `episode_end.__execution__`. A connection-local Args proxy executes `min(replan, returned tail length)` without rewriting the simulator loop. Declares execution/partial-tail protocols. | `test_libero.py::test_original_kernel_executes_extended_commit_and_partial_terminal_tail`; `test_real_framework_extended_commit_native_kernel_and_terminal_capture`, each parameterized commit10/15/32. Real localhost WS server receives all 34 controls including final ack; NFE is charged only on new source chunks. Early success3 and native-error2-control paths tested. | **CPU passed**. Actual Fast-WAM/DP + LIBERO renderer qualification remains coordinator/P3/P4 work. |
| B2 major, LIBERO identity/entry/library | Added sealed four-suite `LiberoBenchmarkSpec`, canonical suite/native-name map, global task IDs0–39, original pool indices, explicit decision seed. Generic driver `--arm name:YAML:K --benchmark-spec`, strategy aliases, LIBERO worker branch/probe/launcher. Original setup receives suite-local ID while server/capture/journal retain global ID. | `test_global_ids_partitions_context_seeds_and_worker_probe`; `test_library_mapping_never_merges_suite_local_task_ids` gives IDs3/13/23/33; `test_model_neutral_libero_driver_manifest_and_plan` creates400 distinct tasks; `test_libero_worker_entry_uses_original_setup_local_ids_and_native_kernel` proves object/global13 → native-local3/init7. | **CPU passed**. Official task maps must be exported in the installed LIBERO island before real deployment. |
| A3 major, π five layers per bundle | π entry consumes P1 `r17.recipe.v1` and shared recipe factory. YAML controls store digest, teacher sampler, optional fit identity, ordered tasks, guard/corrector, gap or target_ir. Process CLI recipe/gap flags now fail with a migration message. K and recipe selection are independent. | `test_pi05_fix_round1.py::test_yaml_five_layers_independent_of_miss_k_and_content_bound_hot_switch`: real WS K10+library hit, plainK2/K1 misses, corrector/guard switches, periodic force-K10 miss, IR.25→gap3, bad store digest rejected. Teacher is a sealed P6 store, not a process-wide fitted object. | **CPU passed with P1/P6**; real native library quality unqualified. |
| A3 major, queue config compatibility | EndpointSpec/task metadata bind common `bundle_identity` including recipe settings and fit/store content. Configured endpoint rejects missing or different bundle identity even with same model/norm/layout/protocol/K. Preload validates returned content identity. | `test_config_identity_prevents_queueing_same_geometry_different_recipe`; generic plan includes sealed identities. | **CPU passed**. Empty identities remain only for explicit unsealed diagnostic/API fixtures; production CLI always seals them. |
| A6 major, decision costs/final ack | New client-only `benchmarks/execution.py`; both runners persist full identity, decision_seq, measured cost/stage timings, original `server_timing`, scope/kind and capture/pricing flags. Terminal ack uses unchanged stock client transport with R17 server control. Failure cannot become an accepted success. π state-only tail invokes the native Normalize operation alone. | `test_mw_cost_final_ack_and_physical_request_generation_match`; `test_terminal_ack_transport_failure_cannot_be_accepted_as_success`; real LIBERO server capture receives final count; `test_state_only_tail_matches_native_normalization_without_image_or_tokenizer`. π full WS/P6 writer readback covers final acknowledgment. | **CPU passed with P1/P6**. Timing is host wall time; no GPU latency/pricing claim. |
| A7 minor, default R17 request identity | Both R17 MW constructor and benchmark factory default `framework_context=True`; explicit False keeps the legacy comparison path. P1 accepts original index aliases and fails closed when formal request identity or explicit seed is absent. | `test_framework_request_ack_and_original_task_identity`, runner/factory parameterizations, verifies original idx9, canonical name, explicit decision seed, preceding5 and terminal2 acks without an opt-in argument. Both failed before the default change (`cpu_fix1_context_old_red.xml`); four disabled-path byte-parity cases retained. | **CPU passed with P1**. Frozen original entrypoints remain unchanged. |
| A5 major, worker debug metadata | π metadata exposes r17_model_spec and actual r17_samplers. Worker consumes P6 PhysicalCaptureFactory, resolves exact task K, fails closed for missing grid or mismatched model/norm/layout. P5 published the analogous GR metadata hook. | `test_worker_debug_selects_native_sampler_per_task_and_closes_writers` writes and reads separate K2/K10 bindings; `test_worker_capture_rejects_wrong_checkpoint_before_writing`; copied-file import probe remains model/simulator-free. | **P2 CPU passed**. P5 reports its own actual GR main/worker CPU fixture receipt in P5_to_P2; not a P2 GPU claim. |
| A13 major, restart loses bundles | Stage begin/resume re-publishes compatible content-bound bundles; a dead cached control reconnects once before retrying publication. Per-endpoint locks serialize control RPCs without nesting endpoint locks. Before dispatch, control selection verifies bundle availability; lost registry or dead cached socket triggers reconnect and re-publication. Failed restore requeues the dispatched task instead of returning it as executable. | `test_bundle_restoration_after_registry_loss_and_socket_restart`: clear registry and kill the fake cached socket before stage start, then invalidate it between two episodes; `test_concurrent_dispatch_never_overlaps_control_rpc_on_one_socket` sends50 concurrent checks; next dispatch restores bundle. Existing real WS/preload tests also pass. | **CPU passed**. In-flight server crash remains a retried/error attempt; its actions are never silently replayed. |
| A8 minor, terminal descent error | Attempts are checked against dispatched descent_level. Terminal native/retry-exhausted errors stop unsent current/later task levels, allow stage completion and resume, and yield a diagnostic freeze with explicit errors. Such freeze cannot authorize test or count fake SR failures. | `test_descent_terminal_error_finishes_with_diagnostic_freeze_and_resume`; existing first-failure/50-task/gate/retry tests retained. | **CPU passed**. A damaged cohort requires a new repaired campaign for formal authorization. |
| A9 minor, nondivisible commit | MW plan and R17 runner reject commits7/12 before native control. P1 default rejects all nondivisible commits. LIBERO explicitly negotiates allow_partial_commit and r17.partial_tail.v1 to permit commit32/replan5 with an actual tail2. | `test_mw_nondivisible_commit_rejected_at_plan_boundary` (7/12); real WS commit32 regression asserts six5-control decisions followed by tail2 and fresh-source2. | **CPU passed**; commit32 is not globally prohibited. |
| A10 minor, physical/model identity | Both request and physical/cost/ack rows use authoritative task.extra.bundle_generation. Worker binding is selected from actual task sampler, with a checkpoint/norm/layout guard. | MW generation deliberately differs from bundle ID and matches across wire/physical rows; actual sampler writer readback tests K2/K10. | **P2 CPU passed**. P6 corrected its placeholder fixture IDs; the unshimmed broad-suite rerun is reported below. |
| B5/B6, owned protocol boundary | Wire256 or96 is explicit in LIBERO protocol identity;224 rejected. Separate reference profile has seed42/fixed per-call seed, replan32, spatial280. Default remains seed7/paired semantic seed, replan5/spatial220. | Protocol/seed/view tests; native paired views retained and wire256 verified. | **CPU contract passed**. No claim of reproducing published Fast-WAM97.6% or sidecar-loader parity. |
| Additional owned lifecycle issue | A LIBERO environment created before episode_start failure is closed; errors remain errors. | `test_libero_start_failure_closes_created_native_environment`. | **CPU passed**. |

### Owned files and interface/default-off proof

Changed production: `benchmarks/{libero_bridge,metaworld,execution}.py`,
`conductor/{manifest,strategy,driver,worker_entry}.py`,
`models/pi05/{adapter,provider,serve}.py`. `execution.py` is new. No benchmark
simulator or model algorithm is duplicated: LIBERO's original function is
compiled without eager vendor imports in CPU tests; real entry uses its original
lazy function import. The existing MW kernel remains unchanged.

Added regressions: `tests/benchmarks/test_libero.py`,
`tests/conductor/test_fix_round1.py`, `tests/models/pi05/test_pi05_fix_round1.py`.
Updated existing π E2E/dispatch/native-tail fixtures to supply complete identities,
use the local R17 config/server imports, per-task physical bindings and final RPC.
Added `prove_old_regressions.py` and `build_fix1_samples.py`; updated the receipt
sealer and own requests. No foreign test was edited.

Interface remains **r17.contracts.v1**, with additive **r17.recipe.v1**,
**r17.execution.v1**, **r17.partial_tail.v1**, and explicit
`adapter_binding.allow_partial_commit`. P1 restored all four sealed stock
source files to HEAD. P2 imports `common.config` and
`common.websocket_policy_server`; stock registry/transport/lifecycle are retained.
The CPU-only conductor planner strictly parses fields and checks sealed model,
feature/layout/protocol/K/commit identities without loading/registering a model;
full provider/artifact/fit validation runs at server publication.

Default-off: original MW/new explicit `framework_context=False`/capture-only action,
image/state and lifecycle bytes still compare equal. The new R17 MW runner/factory
defaults to True so a caller cannot silently lose formal identity or acks.
LIBERO's bridge factory defaults to the
original runner; explicit disabled R17 wrapping compares original actions,
paired camera/state wire values and lifecycle metadata. The R17 worker explicitly
opts in to context/acks. Legacy tests and R16 read-only verify pass below.
`SOURCES_P2.json` checks the original LIBERO/MW runners and the four sealed stock
sources byte-for-byte against HEAD, plus SHA-only tracking of the untracked
A renderer. No model provider, torch/JAX/vendor or simulator is imported by the copied-file worker probe.

### Actual CPU receipts and commands

All command prefixes use:

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src .venv/bin/python -m pytest
```

| XML under tests/conductor | Actual result |
|---|---|
| `cpu_fix1_release4.xml` | **77 passed,0 failed,0 skipped**, exit0,33.83s. Full P2 suite, including26 additional cases over initial delivery. |
| `cpu_fix1_cross_release4.xml` | **281 passed,0 failed,0 skipped**, exit0,62.40s. Final unshimmed P1/P2/P6 broad suite after P6 corrected its own placeholder identity fixture and A7 defaults were fixed. Earlier `cpu_fix1_cross_sealed.xml` (273 pass,1 fail) is preserved as diagnostic evidence. |
| `cpu_fix1_legacy_release4.xml` | **451 passed,6 skipped,1 deselected**, exit0,29.17s. Six skipped local proxies executed separately; excluded test creates real EGL across processes. |
| `cpu_fix1_proxy.xml` | **6 passed,0 skipped,23 deselected**, exit0. Actual localhost socket tests, no GPU. |
| `cpu_fix1_groot_worker_release.xml` | **3 passed,0 failed,12 deselected**, exit0. P2 actually reran P5 real serving CLI/worker debug over localhost at K4/2/1, with CPU neural/env boundaries. |
| `cpu_fix1_old_red.xml` | **2 expected failures,8 deselected**, exit1. New MW cost/terminal-ack regressions run against the preserved pre-fix runner copied in initial payload scratch. It lacks decision_seq and swallows final-ack failure. This is evidence of regression sensitivity, not a passing certificate. |
| `cpu_fix1_context_old_red.xml` | **2 expected failures,14 deselected**, exit1. Runner/factory default-identity regressions both fail with missing `__request__` before changing the default. |

Owned targets: `exp/r17_integration/tests/benchmarks
exp/r17_integration/tests/conductor exp/r17_integration/tests/models/pi05`,
with `-p no:cacheprovider --basetemp=exp/r17_integration/tests/conductor/scratch_fix1_release4
--junitxml=exp/r17_integration/tests/conductor/cpu_fix1_release4.xml -q --tb=short`.

Broad targets additionally include `tests/{common,debug,search,ops}` under R17;
corresponding scratch/XML end in `cross_release4`. Exact full commands, skip counts,
duration and source hashes are in `RECEIPT_P2.json`/`SOURCES_P2.json`.
Legacy targets additionally include `tests/cache/{test_config,test_denoise_schedule_config,test_cache_storage}.py`
plus the initial legacy list above; use `-k 'not same_identity_same_initial_state_across_processes'`
and scratch/XML `fix1_legacy_release4`. Proxy uses
`tests/serving/test_replica_proxy.py -m env_dependent --run-manual`.

Read-only seal command actually run:
`python -m exp.offline_search.rounds.r16.analyze verify` with CPU prefix →
**exit0: DESIGN_VERIFIED; no outcome reads**. No frozen result was written.
Sample builder generation and idempotent store/config readback both exited0:
`python -m exp.r17_integration.tests.models.pi05.build_fix1_samples`, CPU prefix.
Old regression proof command:
`python exp/r17_integration/tests/conductor/prove_old_regressions.py`, CPU prefix.
No GPU/manual renderer tests were enabled. Passing counts overlap across suites;
do not sum them. Intermediate red receipts are retained with their actual status.
The final receipt sealer compiles **24 owned Python files** without pyc and verifies
**19 tracked reused/sealed sources byte-identical to HEAD**; the untracked A renderer
is recorded by SHA only. The A7 red-test command uses the same CPU prefix with
`tests/benchmarks/test_metaworld.py` under R17, `-k framework_request_ack_and_original_task_identity`,
and `scratch_fix1_context_old_red`/`cpu_fix1_context_old_red.xml`. The intermediate
`cpu_fix1_release3.xml` (76 pass,1 fail) reflects the invalid-short-plan fixture's
obsolete lifecycle expectation; it now explicitly verifies terminal zero-control ack.

### Updated hook table and samples

| F/M/B hook | Executed evidence | Limit |
|---|---|---|
| F per-bundle YAML/fit/guard/corrector/periodic/targetIR | Real WS + sealed P6 teacher store + shared P1 recipe, fullK and reducedK independently | Synthetic π neural velocity/prefix, no native-quality claim |
| F+B per-episode dispatch/descent/restart/attempt fences | Generic LIBERO400 plan, suite/global mapping, existing concurrency/TCP driver tests, error freeze/resume and restored bundle registry | No official LIBERO or EGL environment instantiated |
| M+B native execution/commit | Original LIBERO kernel + actual P1 WS core, commit10/15/32, NFE and final server ack; original MW loop/ack | Model/env boundaries are explicit CPU fixtures |
| F+M+B capture/cost | Actual sampler-selected P6 writers/readers; decision/full identity, final execution overlay and capture-pricing exclusion | Host wall timing only; fixed model per worker, mismatched checkpoint fails closed |
| B protocol/wire/native setup | Four-suite canonical map, separated init splits, seed rule, wire96/256 and reference profile | Official map/pool/source/renderer attestations pending deployment |

New permanent samples:
`tests/models/pi05/samples/fix_round1/{sample_manifest.json,plain1.yaml,plain2.yaml,recipe10.yaml,ir25_k2.yaml,teacher/,libero_fixture_protocol.json,libero_fixture_400.json,libero_commit32_rows.json}`.
The teacher store is content sealed; configs expose their distinct bundle/recipe/fit
identities. LIBERO rows show34 actual CPU fixture controls and a final execution
ack with counts `[5,5,5,5,5,5,2,2]`. Task names in this sample are explicitly
synthetic; deployment requires official maps. Samples make **no checkpoint,
renderer, SR, or price claim** and must not be substituted for a real bound library.

### GPU-ready qualification — NOT EXECUTED

Initial native Slot A above remains valid and bounded to900s. Initial Slot B's
process-level `--recipe-store` command is obsolete. Use these replacements after
coordinator reserves an exclusive inference/EGL slot, one model at a time. Supply
immutable checkpoint/tokenizer, routeable endpoint, existing correctly bound P6
teacher store, reserved ports and fresh owned output paths. Total wall budget:
startup≤180s, episodes≤420s, readback≤120s, shutdown/SHA≤180s; hard deadline900s.
No unlimited retry. A missing HIT, identity/ack/readback failure, startup overrun,
or deadline yields partial/failed, not qualified.

Prepare `P2_RUN/configs/recipe_k2.yaml` on CPU before the slot: copy emitted K2
YAML and add `recipe: {schema:r17.recipe.v1, enabled:true, store_path:<sealed store>,
store_digest:<READY.content_digest>, teacher_sampler_identity:<bound teacher sampler digest>,
corrector:true, guard:true, periodic_gap:2, k:16, kref:8, early:true,
ordered_tasks:<all50 canonical MW names>}`. Supply the actual fit identity if sealed;
`common.recipe.fit_identity` defines it. Coordinator/P6 must provide a store bound
to the exact emitted checkpoint/norm/layout/features/protocol. Native teacher
capture/fit is not performed by this P2 CPU handback.

```bash
# Inference host, one owned process; no CLI-level recipe selection.
timeout --signal=TERM 900s env CUDA_VISIBLE_DEVICES="$P2_GPU" JAX_PLATFORMS=cpu \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src:packages/openpi-client/src \
  "$P2_MODEL_PYTHON" -m exp.r17_integration.models.pi05.serve \
  --checkpoint "$P2_CHECKPOINT" --device cuda:0 --host "$P2_SERVER_BIND" \
  --port "$P2_SERVER_PORT" --config "$P2_RUN/configs/k10.yaml" \
  --debug-directory "$P2_RUN/model_capture"

# Driver host, exactly two mechanical episodes; YAML bundle selects five layers.
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=.:src:packages/openpi-client/src "$P2_MODEL_PYTHON" \
  -m exp.r17_integration.conductor.driver --smoke --deadline-seconds 420 \
  --manifest "$P2_RUN/configs/model_manifest.json" --endpoint "$P2_ENDPOINT" \
  --arm "full:$P2_RUN/configs/k10.yaml:10" --arm "recipe:$P2_RUN/configs/recipe_k2.yaml:2" \
  --output "$P2_RUN/episodes_fix1" --bind-host "$P2_DRIVER_BIND" --bind-port "$P2_DRIVER_PORT"

# Two workers use the initial bounded commands; exact native K comes from server metadata.
# Separate LIBERO qualification: export native task map in LIBERO island before GPU slot.
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=.:src:packages/openpi-client/src "$P2_WORKER_PYTHON" \
  -m exp.r17_integration.benchmarks.libero_bridge --wire-size 256 --output "$P2_RUN/libero256.json"
# --wire-size96 is a different protocol; --released-reference exports fixed42/replan32/spatial280.

# After P3/P4 server starts with this exact benchmark allowlist and a sealed H32 bundle:
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=.:src:packages/openpi-client/src "$P2_MODEL_PYTHON" \
  -m exp.r17_integration.conductor.driver --smoke --deadline-seconds 420 \
  --manifest "$P2_ENDPOINT_MODEL_JSON" --benchmark-spec "$P2_RUN/libero256.json" \
  --endpoint "$P2_ENDPOINT" --arm "commit32:$P2_LIBERO_COMMIT32_YAML:10" \
  --output "$P2_RUN/libero_episodes" --bind-host "$P2_DRIVER_BIND" --bind-port "$P2_DRIVER_PORT"
timeout --signal=TERM 600s env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=.:src:packages/openpi-client/src "$P2_WORKER_PYTHON" \
  -m exp.r17_integration.conductor.worker_entry --worker-id p2_libero_owned \
  --server-key "$P2_ENDPOINT" --driver-host "$P2_DRIVER_HOST" --driver-port "$P2_DRIVER_PORT" \
  --benchmark libero --benchmark-spec "$P2_RUN/libero256.json" --init-states-dir "$P2_LIBERO_POOLS"
```

`P2_ENDPOINT_MODEL_JSON` is a content-pinned metadata JSON with `r17_model_spec`;
`--arm` K must equal its sealed YAML binding. A H32/replan5 commit32 YAML requires
`allow_partial_commit:true`; a separate released replan32 profile needs no remainder
negotiation. Start the P3/P4 server with its package's bounded immutable-descriptor
command, B=1, inside the same900s slot. Read server model capture and client
cost/ack/journal with P6 before declaring any capability passed. LIBERO physical
capture is explicitly unsupported in this worker; model capture is server-side.
Do not run a LIBERO native-physical debug flag and silently produce MW physics.

### Remaining limits and pending owners

- **Coordinator + P1/P2/P6:** actual π checkpoint/GPU/native-prefix, real MW/EGL,
  G1 closed capture/journal/ack join and G4 quality remain unexecuted. CPU hooks
  and synthetic five-layer/store tests are not SR or latency certificates.
- **P3/P4 + coordinator:** native LIBERO/DP/Fast-WAM weights, real paired frames,
  commit32/source-loader/reference parity and renderer qualification. Default
  R17 seed7/replan5/spatial220 is explicitly not the published reference protocol.
- **P6 fixture correction resolved:** P6 changed its own worker-entry test to use
  the actual spec checkpoint/norm/layout. P2 reran the unshimmed broad suite; the
  earlier red XML remains diagnostic. No production provenance guard was weakened.
- **P2/P6, future capability:** mixed-checkpoint capture in one global worker needs
  endpoint-aware metadata/bridge routing; current worker is fixed-model and fails
  closed on conflicting task model/norm/layout. Same-model K/bundle switching is
  tested. Native LIBERO physics, full50 MW semantic roles/restore/asset/substep
  certificates remain explicitly unsupported rather than inferred.
- **Q/coordinator:** explicit MW replan1 diagnostic spec is supported by existing
  model_spec/runner and new worker --benchmark-spec; formal replan5 remains unchanged.
  No physical G1 tail certificate or GPU test was run by P2.

## Fix round 2

2026-10-09. Read REVIEW_A2 and REVIEW_B2 completely, including the N2/N3 repros.
All edits are in P2-owned R17 code/tests, own requests and this handback. Zero GPU,
real renderer or checkpoint loading; no commit/push, process interference, frozen
source/result edits, harness edits or training-source edits. This section supersedes
round1's uncalibrated `target_ir → gap` sample and host-wall pricing interpretation.

### Finding → change → test → status

| Finding | Change | Regression / actual evidence | Status |
|---|---|---|---|
| A2 N2 major, benchmark/runner table | Both MW and LIBERO specs publish ordered_tasks and P1's exact r17.task_table.v1 identity, bound to benchmark/protocol identity and ordered canonical names. Manifests, probes, EpisodeTask.extra, requests, episode-start metadata and physical/cost/ack rows carry the same identity. Descriptor loading rejects conflicting identities/order. Server/provider/conductor paths register canonical tables; client workers reconstruct metadata without the server registry dependency. | `test_task_table.py`: MW/LIBERO table/manifest/tamper checks and one-task MW runner idx7/full-table identity. Both spec tests and runner test failed on pre-fix code (3 expected red), then passed. Existing original-LIBERO kernel + actual WS commit10/15/32 remains passing with table attestation. | **P2/P1 CPU passed**. Official LIBERO task-map/renderer attestation remains coordinator work. |
| A2 N2 major, emitted configs | Every π K10/K2/K1 YAML includes complete ordered_tasks/table identity even with recipe disabled. Emitter rejects subset/alphabetical-order/digest conflicts. P1 validates hand-written YAML before fitting and checks query task_name/task_id before model work. Added `canonical_config_document` as the owned GR emission bridge. | `test_pi05_fix_round2.py`: three plain Ks, three invalid-table mutations with poison fitter, partial real-layout teacher fixture hit and mismatched query rejection, actual `serve --emit-configs` without model load. `test_gr_default_yaml_emission_bridge_carries_table_and_preserves_options` calls P5's actual default_config with synthetic H16 metadata and preserves binding/checkpoints. | **π + GR bridge CPU passed**. P5's own production GR emitter adoption is requested in P2_to_P5; P2 did not edit it. |
| A2 N3 minor, all non-retryable errors | Live descent uses the original driver is_retriable_error classifier rather than a prefix check. All accepted ConfigValidationError/ConfigValidation/SidecarError/FatalEpisodeError errors stop that task, clear unsent levels and retain in-flight fencing/journaling. Unfinished prior cohorts become diagnostic stops at the next stage. Pre-send bundle/on_dispatch failures route through handle_result for bounded retries, terminal journal and strategy notification. | `test_p2_fix_round2.py`: eight MW/LIBERO repr-marker cases prove healthy-task progression, late in-flight evidence, diagnostic freeze and resumed/live agreement; two pre-send retry-exhaustion cases; transient callback requeue; unfinished prior stage. All12 failed on old code and now pass. | **CPU passed**. Error freezes remain diagnostic and cannot authorize formal test. |
| A2 N5 minor, queue versus service ledger | π supplies the shared drain_queue_wait_ms hook (no shared adapter lock). Client ledger preserves cost_ms, queue_wait_ms, service_ms, per-stage wall/queue/service, pricing_time_field and exact scope; unknown timing is ineligible, never zero-filled. An explicit local instrument_model_lock bridge times a connection-local view over a shared native RLock. | Missing-field ledger witness failed old code (1 expected red). Actual held CPU lock yields ≥20ms queue wait and verifies service_ms=cost_ms−queue_wait_ms and stage service subtraction through the real P1 core. Native RLock view preserves exclusion/output bytes and drain reset. | **P2/P1 CPU passed**. P5 production native GR hook and P6 analyzer service-field selection are requested; no GPU/currency pricing claim. |
| A2 N4, owned stage reload cost | Stage begin/resume selects existing generations and loads only missing bundles. A dead cached control reconnects; dispatch restoration targets its missing bundle. Initial publication still checks returned content identity and endpoint compatibility. | `test_existing_stage_bundles_are_selected_without_reloading_artifacts` failed old code (1 expected red): existing stage produces zero extra loads, partial registry loss loads only recipe, actual restart restores both. Existing restart/dead-socket/50-concurrent-RPC tests retained. | **P2 CPU passed**. Shared SHA/fit/event-loop caching belongs to P1/P6. |
| B2 B2 remaining coverage | Added explicit LIBERO descent witnesses rather than relying only on class aliases. | Four error types × LIBERO, healthy task traverses k2/k1, failed task never dispatches another level, freeze/resume agree. | **CPU passed**, not real simulator/SR evidence. |
| B2 N6 incoming P4 request | Generic descent rejects lossy-study markers. Driver propagates allow_lossy_sampler/sampler_study_label into episode plans; ledger preserves quality/study diagnostics. | Three separate marker regressions, normal MW/LIBERO descent tests unchanged. | **P2 CPU passed**. P4/P1 own DP K1 binding admission and scientific quality gate. |

### Owned diff and interface/default-off proof

Updated `benchmarks/{spec,libero_bridge,metaworld,execution}.py`,
`conductor/{manifest,strategy,driver,worker_entry}.py`,
`models/pi05/{adapter,provider,serve}.py`; existing π/Libero CPU fixtures changed
only to supply canonical table metadata or explicit gap3 (no owner-IR label).
New `tests/benchmarks/test_task_table.py`,
`tests/conductor/test_p2_fix_round2.py`,
`tests/models/pi05/{test_pi05_fix_round2,build_fix2_samples}.py`,
`tests/conductor/seal_fix2.py`. The conductor test has a unique basename to avoid
P1's independent `test_fix_round2.py`; no foreign test was renamed or edited.

Interface stays **r17.contracts.v1** plus **r17.task_table.v1** and additive
RecipeConfig task_table_identity/ordered_tasks metadata. P1 publishes the registry,
strict recipe/request validation and `adapter.drain_queue_wait_ms() -> float`.
RequestContext.task_table_identity is optional and excluded from RNG identity.
Local helpers have no import-time registration or instrumentation. Registry calls
are explicit in π model_spec, conductor load_benchmark_spec, and model-owned
LIBERO loaders. Workers use load_benchmark_spec(register=False) and retain their
existing client-only payload allowlist; model/vendor/simulator imports stay absent.

Default-off evidence: all original kernel versus explicitly disabled R17 action,
image/state/prompt/lifecycle byte comparisons remain in the full P2 suite. Frozen
shared/MetaWorld/LIBERO sources stay byte-identical to HEAD; R16 verify passed.
Opt-in lock timing changes a new connection adapter's lock reference only, retains
the shared underlying lock, and changes no resident model/lock methods or RNG.
Round1 receipts/samples/source snapshots are preserved; round2 uses separate files.

### Actual CPU commands / receipts

Every command below uses this prefix:

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src .venv/bin/python
```

Final test counts and exact commands are sealed in
`tests/conductor/RECEIPT_P2_FIX2.json`; sources in `SOURCES_P2_FIX2.json`.
Full P2 targets remain benchmarks/conductor/models/pi05 under R17 tests. Broad
targets additionally include common/debug/search/ops. Use `-m pytest <targets>
-p no:cacheprovider --basetemp=exp/r17_integration/tests/conductor/scratch_fix2_<run>
--junitxml=exp/r17_integration/tests/conductor/cpu_fix2_<run>.xml -q --tb=short`.
The final full/broad run IDs are `full5` and `cross4`.

| Receipt | Actual result |
|---|---|
| `cpu_fix2_full5.xml` | **108 passed,0 failed,0 skipped**, exit0,41.41s. Full P2 suite,31 additional cases over round1. |
| `cpu_fix2_cross4.xml` | **346 passed,0 failed,0 skipped**, exit0,74.98s. Final P1/P2/P6 broad suite including both client payload paths. |
| `cpu_fix2_legacy1.xml` | **451 passed,6 skipped,1 deselected**, exit0,30.88s. Real EGL case excluded. |
| `cpu_fix2_proxy.xml` | **6 passed,0 skipped,23 deselected**, exit0,.17s. Resolves all six legacy socket skips. |
| `cpu_fix2_terminal_old_red.xml` | **12 expected failures**, exit1. Repro-derived terminal/retry/barrier witnesses on pre-fix code. |
| `cpu_fix2_preload_old_red.xml` | **1 expected failure,12 deselected**, exit1; original code reloads all artifacts at each stage. |
| `cpu_fix2_table_old_red.xml` | **3 expected failures**, exit1; absent canonical spec/episode metadata. |
| `cpu_fix2_ledger_old_red.xml` | **1 expected failure,13 deselected**, exit1; queue_wait_ms dropped. |
| `cpu_fix2_peer_shims.xml` | **2 passed,17 deselected**, exit0; explicit lock/emitter bridges, synthetic metadata. |

Legacy targets/options match round1's full legacy set, with `-k 'not
same_identity_same_initial_state_across_processes'`; exact command in the receipt.
Socket command uses `tests/serving/test_replica_proxy.py -m env_dependent --run-manual`.
Read-only `-m exp.offline_search.rounds.r16.analyze verify` actually returned
**exit0, DESIGN_VERIFIED; no outcome reads**. Sample generation and idempotent
readback `-m exp.r17_integration.tests.models.pi05.build_fix2_samples` both exit0.

Diagnostic receipts retained honestly: full1 had101 pass/4 fail from an unpublished
request field; cross1 had a test-basename collection collision; cross2 had340 pass/
2 payload allowlist failures. These were resolved in owned code using P1's published
append-only request field, a unique test basename, and server-only registration.
Cross3 then passed344 tests. Counts overlap; do not add runs together.

### Cost authority and pricing field

`cost_ms`/`stage_ms` remain **host wall latency** (before capture sink) including
model-lock wait. `queue_wait_ms`/`stage_queue_wait_ms` measure adapter lock acquisition;
`service_ms`/`stage_service_ms` subtract that measured wait. The timing pricing field
is **service_ms**, identified by `pricing_time_field`, with scope
`policy_host_wall_minus_adapter_model_lock_wait_before_capture_sink`. No hook means
service/queue fields=None and pricing_eligible=False. Any debug capture remains
pricing-ineligible. `gpu_service_ms=None`: CUDA submission/queue/contention or actual
GPU kernel time is not measured by this CPU certificate. Compare host service only
under controlled matching concurrency; it is not owner IR or a currency price.

Owner IR remains **(cV·V+cM(K)·M)/N**, using actual decisions/stage calls and a
model/sampler/protocol-bound measured coefficient receipt. Do not transfer R16
LIBERO coefficients to MW/new native grids. A target_ir requires P1's bound ir_costs
and library replay calibration; plain periodic_gap3 carries no IR25 claim.
P6's legacy summarize implementation currently sums cost_ms; until P6 consumes
pricing_time_field/service_ms, do not call its normalized wall ratio service pricing.

### Samples, GPU commands and pending owners

New samples in `tests/models/pi05/samples/fix_round2/`: plain10/2/1 and recipe10/
gap3_k2 YAMLs with complete canonical metadata, sealed synthetic teacher store,
MW and synthetic four-suite LIBERO protocol documents, actual CPU service ledger
and sample_manifest. No actual weights, renderer, SR or measured GPU-price claim.

Round1's bounded replacement server/driver/worker/native commands remain the
qualification commands (**NOT EXECUTED**): hard900s total, startup≤180s,
episodes≤420s, readback≤120s, shutdown/SHA≤180s; one model at a time, later reserved
inference/EGL slot. Re-emit configs with current `models.pi05.serve --emit-configs`
before the slot. When enabling recipe, retain emitted ordered_tasks and
task_table_identity and add the exact bound store/teacher/fit fields. Use explicit
periodic_gap until a real bound IR-cost receipt exists. P3 descriptors now require
the exact benchmark_specs document (system versus released profile must agree).
No GPU command, inference GPU lock or training process was touched today.

- **P5:** adopt native drain timing or `instrument_model_lock` via explicit
  adapter_factory, and canonical config emission bridge in GR production entry.
  Both bridges are CPU-tested; P2 cannot edit P5's files. Requests contain APIs.
- **P6:** normalize eligible timing with service_ms, retain queue/scope/GPU status,
  and keep owner-IR stage coefficients distinct. Current P2 ledger is ready;
  P6's analyzer change is requested, not falsely claimed deployed.
- **P1/P6 + coordinator:** actual model/sampler/protocol cost calibration and
  scientific authority of ir_costs.source_identity; schema validation alone is
  not a measured receipt.
- **Coordinator + P2/model owners:** real π-MW first mechanical GPU/EGL/capture
  join, real checkpoints, official LIBERO mapping/renderer and later G4 quality.
  Existing unsupported native LIBERO physics, MW goal/restore/full semantics,
  batching/replica/split/advanced warm/trace capabilities remain as earlier.

## GPU fix 1

2026-10-09, after the owner's explicit single-GPU diagnostic authorization.
Read Q's failed G1 A native.json/native.log and traced the actual stock/R16 paths.
**Verdict (b): compilation-dependent numerical behavior; no adapter grid, noise,
normalization or native-stage defect found.** The disagreement is appreciable,
not float32 epsilon. Exact parity is against R16's actual eager serving path.

### Actual GPU evidence and reference boundary

ONE bounded diagnostic used Q's exact failed-slot job/runtime/checkpoint/norm/
three saved wire observations. `gpu_diagnostic.py` loaded π once, exercised K2/K1,
and exited0: **273.899s diagnostic /278.494s subprocess wall**, **8156MiB peak
process GPU memory** (8.552GB, below14GB), allocator limit12288MiB. No shared GPU
lock, renderer, server, tmux, other process, checkpoint/asset, frozen source or
result was changed. No second GPU diagnostic or corrected Q G1 slot was run.

Both the adapter and stock original sample_actions use the same native float32
accumulated Euler loop. All three observations at each K gave **exact normalized
and wire parity against the R16 reference**, exact parity against original
sample_actions with only its whole-call compiler wrapper removed, and exact
debug-on/off parity. All three K2 snapshots resumed with exact normalized suffix
bytes; K1 has no suffix. Q's previously passed K10/9-suffix evidence remains its
historical receipt, not an additional GPU check claimed here.

| K | Both paths' timestep grid | Both dt | Compiled-stock versus adapter max abs difference | Mean abs difference over60 wire values |
|---|---|---|---|---|
| 2 | [1.0,0.5] | float32−0.5 | .07767835076159457 | .014633875234252419 |
| 1 | [1.0] | float32−1.0 | .12213263194465185 | .020499740996412243 |

Per-observation compiled max/mean pairs: K2 `.0363580821/.0159222990`,
`.0776783508/.0144469440`, `.0701004064/.0135323827`; K1
`.0601583501/.0181839904`, `.1221326319/.0293494092`,
`.0479188341/.0139658234`. Native wire dtypefloat64/shape5×4; normalized dtype
float32/shape5×32. Every reference/uncompiled/debug/suffix difference is zero.
The compiled grid is established from the identical original method source and
executed CPU native-loop witness, not instrumentation inside the compiled graph.
Individual fused kernels were not isolated; the evidence isolates the effect of
whole-call compilation without asserting which operator causes the difference.

R16 boundary, read-only evidence:

- `scripts/serve_policy.py:_disable_compile_for_serving` sets compile mode=None;
  its concurrent connection factory passes eager=True.
- `src/openpi/cache/interceptor.py` passes the YAML miss K to native run_stage3
  with return_intermediates=True; the B1 coordinator path in
  `src/openpi/serving/batching_coordinator.py` uses the same native call.
- `exp/offline_search/rounds/r16/replay.py:synthetic_chunk` exercises that same
  public-stage loop. `r16/prepare.py` emits miss.num_steps=K for step reduction.
- Q's current K10 stock Policy timing branch is eager. Its K2/K1 nonstaged branch
  calls sample_actions wrapped by PI0Pytorch.__init__'s max-autotune compiler.
  Policy's timing-only staged branch ignores sample_kwargs.K and defaults10;
  merely setting _staged_inference=True is therefore not a valid K2/K1 fix.

### Change → regression → status

| Finding | Owned change | Test / receipt | Status |
|---|---|---|---|
| G1 A K2 upstream/debug byte assertion | Added independent r16_reference using original transforms and public eager stages; no production sampler/grid/norm change. | Actual GPU6 comparisons,3 exact K2 suffixes; CPU poison stock infer/compiled sampler/adapter prepare/sample/wire helpers, still obtain independent exact reference at K10/2/1. | **P2 diagnostic PASS**; Q's historical failed slot is not rewritten. |
| Exact-versus-tolerant contract | New r17.pi05.parity.r16_eager.v1 validator always requires exact normalized/wire/reference/debug bytes and true native NFE. Existing Q suffix equality remains exact. | One-bit native/debug mismatches still fail while an explicitly scoped alternate discrepancy passes; missing profile/out-of-bound diagnostic fails. | **CPU PASS**. |
| Alternate compiled comparison | Sealed cohort-only absolute max/mean limits: K2=.08/.02, K1=.13/.03,rtol0. Admission requires exact checkpoint/norm/archive identity. These rounded observed envelopes are not general action-quality tolerances. | CPU archive/model/norm admission regression; post-run CPU validation of all6 real GPU rows against published bounds. | **CPU PASS**, not behavioral equivalence or SR/pricing evidence. |
| Qualification entry ownership | Concrete imports, baseline replacement, parity call and explicit reporting patch in REQUESTS/P2_to_Q.md. | Owned reference/validator regressions + real diagnostic; Q source/payload seal and full corrected native/framework slot are Q-owned. | **REQUESTED Q**, no foreign source edit or second GPU load. |

### Owned files, CPU/default-off receipts and limits

New `models/pi05/{gpu_diagnostic,parity}.py`,
`tests/models/pi05/test_pi05_gpu_fix1.py`,
`REQUESTS/P2_to_Q.md`, this appended handback and separate artifacts under
`tests/models/pi05/gpu_fix1/`. Shared contract remains r17.contracts.v1; the new
parity contract is qualification-only and never imported/enabled by serving.
Original adapter/model/benchmark code remains the exact Q-loaded CPU seal.
32 owned Python files compiled without pyc;19 reused/frozen sources rechecked
byte-identical to HEAD; R16 read-only verify
again returned exit0,DESIGN_VERIFIED;no outcome reads.

| Actual command / receipt | Result |
|---|---|
| New targeted CPU test file, before final3 added contract cases: cpu_targeted.xml | 6 passed,0 skipped;6.64s. |
| Full P2 benchmarks/conductor/models/pi05: cpu_full.xml | **117 passed,0 failed,0 skipped**;36.20s. Includes all9 new cases. |
| Same full legacy target list as Fix round2: cpu_legacy.xml | **451 passed,6 skipped,1 EGL deselected**,0 failed;25.22s. |
| Same explicit socket proxy command as Fix round2: cpu_proxy.xml | **6 passed,0 skipped,23 deselected**;.12s. Resolves all six legacy skips. |
| Actual recorded GPU cohort → explicit profile validation: profile_validation.json | **6 rows passed**, CPU-only readback; no model/GPU load. |

All CPU pytest commands use the exact CUDA_VISIBLE_DEVICES='' / JAX_PLATFORMS=cpu
prefix above, -p no:cacheprovider, per-run scratch_<name> basetemp and
cpu_<name>.xml in gpu_fix1. Final command/source/XML/report hashes are in
gpu_fix1/RECEIPT_P2_GPU_FIX1.json, separate from historical round1/round2 receipts.

The ONE executed GPU command is recorded in gpu_fix1/command.json: CPU launcher
uses Q.run_slot.cpu_environment(runtime,job,gpu=True), then Q's Python runs
`-m exp.r17_integration.models.pi05.gpu_diagnostic --job
/data/mw_finetune/qualification/runs/slot_G1_pi05_mw_A_20261009_184115_360523/job.json
--output exp/r17_integration/tests/models/pi05/gpu_fix1/diagnostic.json`, under
an885s timeout. Runtime is this checkout/.venv. Diagnostic.json/log/exit.json are
actual receipts, not commands merely proposed for later.

**Pending Q:** apply the supplied pi_native patch, refresh its runtime/payload
seal, and run the corrected complete G1 A slot in Q's schedule. This diagnostic
does not exercise common framework run_probe, physics/EGL, end-to-end conductor
or success rate. Cross-path bounds require this exact synthetic archive/checkpoint/
norm, reach init0 decision seeds0..2 and Q's runtime source seal; a new checkpoint,
archive/compiler/backend requires a new study. K10 alternate remains exact; no
K10 tolerance exists. Large alternate differences must not be called float32
epsilon or used as a quality/cost receipt. Earlier pending-owner entries describe
their historical CPU snapshots; this section's open integration item belongs to Q.

## GPU fix 2

2026-10-09. **CPU-only repair** of Q's GR B-slot admission failure. Read the
Q_to_P2 GPU fix1 request and its preflight/gpu_fix1_probe_pull_repro.json. Stock
WorkerLoop sends a probe only on the first pull of each TCP connection; P2 had
incorrectly validated an omitted later probe as {}. No GPU operation, running
process/tmux interaction, model/asset/qualification edit, commit or push occurred.

### Finding → change → regression → status

| Finding | Owned change | Actual test / evidence | Status |
|---|---|---|---|
| Valid first probe, omitted second/terminal probe fails | R17 driver reuses accepted census probe only for the same worker_id, attested transport peer IP:port and server_host. It validates that accepted probe on each pull. | Two Q-repro-derived tests failed old code with MW package mismatch and LIBERO protocol/task-map mismatch; now both pass. | **CPU PASS** for MW and LIBERO. |
| Strict admission boundaries | First/unknown-peer/new-peer/new-worker pull requires a full probe. Explicit empty/null/mismatching/modified probes fail. Endpoint changes fail with or without an otherwise identical full probe. | 20 parametrized rejection cases, plus reconnect tests requiring an identical full probe; rejected pulls leave census/dispatch untouched. | **CPU PASS**; no validator/version/renderer relaxation. |
| Rejected evidence lost | Rejection rows include raw pull_payload, probe_present, supplied probe, accepted_probe, actual peer and reason, retaining worker_id/server_key/time. Accepted and rejected nested data are independently owned copies. | All negative cases assert raw/accepted evidence and post-rejection mutation isolation; two accepted-probe mutation/reconnect checks. Q's unchanged receipt wrapper passes its own tests. | **CPU PASS**. |
| Stock transport integration | Continue calling stock _record_pull and stock _handle_conn; enabled gate uses a driver-local RLock to make validation/census publication atomic. LIBERO is the same BenchmarkConductorDriver alias/path. | Two real localhost TCP tests use stock WorkerLoop._pull/_report_result: two tasks and accepted reports followed by probe-less MSG_SHUTDOWN on the same connection. | **CPU PASS**, fake episode outcomes; no native physics/GPU/SR claim. |
| Legacy/default-off | Validator=None returns directly to stock implementation and retains its original lock. | New census byte equality test versus actual stock driver includes missing probe, endpoint change, explicit empty and conflicting probe; full legacy/default-off suite and HEAD seals below. | **CPU PASS**. |

### Owned files, interfaces and receipts

Changed `conductor/driver.py`; new `tests/conductor/test_p2_gpu_fix2.py`; appended
GPU fix2 delivery to `REQUESTS/P2_to_Q.md` and this handback. No shared WorkerLoop,
transport, benchmark validator, worker payload allowlist or model file changed.
Interface remains r17.contracts.v1 and the existing stock pull/probe wire format.
Raw rejection fields are additive and align with Q's ProbeRecordingDriver logger.
TCP peers come from stock getpeername(), not a worker-supplied peer field.

CPU pytest commands use the CUDA_VISIBLE_DEVICES='' / JAX_PLATFORMS=cpu prefix
above, -p no:cacheprovider, scratch_gpu_fix2_<run> basetemp and
cpu_gpu_fix2_<run>.xml under tests/conductor. Exact commands, XML/source hashes
and counts are sealed in `tests/conductor/RECEIPT_P2_GPU_FIX2.json`.

| Receipt | Actual result |
|---|---|
| cpu_gpu_fix2_old_red.xml | **2 expected failures,25 deselected**, exit1,.22s. Actual pre-edit Q witnesses. |
| cpu_gpu_fix2_targeted2.xml | **27 passed,0 failed,0 skipped**, exit0,.72s. All new regression cases. |
| cpu_gpu_fix2_full.xml | **144 passed,0 failed,0 skipped**, exit0,36.29s. Full owned P2 suite. |
| cpu_gpu_fix2_q_interop.xml | **9 passed,0 failed,0 skipped**, exit0,7.92s. Q-owned test_gpu_fix1.py run unchanged, including rejection logger and adopted π parity contract. |
| cpu_gpu_fix2_legacy.xml | **451 passed,6 skipped,1 EGL deselected**, exit0,25.23s. Same full legacy/default-off targets as prior rounds. |
| cpu_gpu_fix2_proxy.xml | **6 passed,0 skipped,23 deselected**, exit0,.13s. All six legacy socket skips resolved separately. |

The intermediate targeted receipt is retained:24 pass/3 fixture failures from
expecting stage completion without pumping the stock stage and referencing
nonexistent fixture attributes. Fixtures were corrected to await real accepted
reports and drive the stock completion hook; no production guard was weakened.
Run counts overlap; do not add them. Historical GPU/CPU artifacts are preserved.

Read-only R16 verify again exited0 with DESIGN_VERIFIED;no outcome reads.
All33 owned Python files compile without writing pyc.19 reused/frozen tracked
sources rechecked byte-identical to HEAD. This is a CPU attestation/admission and
transport receipt only; it does not retry or reclassify the historical failed GR
GPU slots. Previous ≤15-min qualification commands remain Q-scheduled commands,
not executed in this fix.

**Pending Q:** refresh/deploy the changed P2 driver and runtime/payload source
seal, then rerun GR B/current+60000 in Q's schedule and collect fresh accepted
episode/terminal/shutdown receipts. The supplied owned driver code fixes both
MW and LIBERO admission; no shared-interface request is needed. Q's adoption of
GPU fix1 π parity is acknowledged by its request and the nine passing CPU tests;
real corrected G1/G2 qualification remains Q's receipt authority.
