# Cosmos Policy LIBERO → R17

Completed 2026-10-09. `cosmos_libero` runs the official K5 policy through the
R17 connection factory, ModelSpec/bundle attestation, stock replica proxy and
per-episode conductor. The 4090 formal-launcher smoke passed **2/2**, with two
accepted timan108 workers, no errors/retries, and byte-exact native parity.
No full campaign or H100 GPU evaluation was launched. `src/` was not edited.

## Files and assets

- `models/cosmos/{loader,manifest,protocols,descriptor,wire_policy,staged,key_builder,provider,serve}.py`:
  CPU-sealed descriptor, official resident, genuine VAE keys, staged native
  inference and shared framework serving. `RECIPE.md` documents the recipe and
  capability limits.
- `models/cosmos/{download,setup_runtime,record_observations,qualify,qualification_slot,test_cpu}.py`:
  pinned download, private imports, real CPU capture, native parity and slots.
- Additive hooks in `qualification/{assets,model_runtime,run_slot}.py`,
  `benchmarks/libero_bridge.py`, and `ops/{formal_eval,formal_driver,formal_server,formal_workers}.py`;
  `ops/FORMAL_EVAL.md` includes the launch command.
- `qualification/specs/slot_G2_cosmos_libero_{A,B}.json` and matching `.sh`
  wrappers: A runs resident startup parity without episodes; B runs the formal
  two-episode smoke. Both default to H100/timan108 and support `--preflight`.

The official release is downloaded and SHA-verified **on both hosts** at
`/data/mw_finetune/models/cosmos_libero`. HF revision:
`cb689ec0e3347c13667d70a78a3447388f5c3bb8`. Weight SHA256:
`8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2`.
Each host has `download_manifest.json`; all five released files, tokenizer,
normalization, prompt cache and model source are included in the CPU seal.
The loader audits every serialized DiT tensor's shape and loaded value; native
TransformerEngine `_extra_state` bookkeeping follows the official loader.

Local runtime: `/home/weiland/projects/openpi_ext/lines/cosmos-policy/.venv/bin/python`;
H100: `/data/cosmos/cosmos-policy/.venv/bin/python`. Both vendors are commit
`18a2accadf4e7a3531e56754102af5a24d2316da`. Private LIBERO import dependencies:
`/data/cosmos/r17_runtime_deps310`; existing environments and live processes
were not restarted or changed. The existing HF VAE cache is SHA-bound; its
snapshot `.pth` path is preserved rather than resolving to a suffixless blob.

## Exact official configuration

| Component | Served setting | Pinned official source |
|---|---|---|
| Experiment, sampler, seed | `cosmos_predict2_2b_480p_libero__inference_only`; K5 EDM `2ab`; diffusion seed195; deterministic mode; variance scaling off | [LIBERO.md:46–74](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/LIBERO.md#L46), [inference sigma80/4](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py#L180) |
| Cameras | agentview + eye-in-hand, native RGB256; **vertical flip**, then native JPEG95 →224 →90%-area center crop/resize | [libero_utils.py:44–57](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/experiments/robot/libero/libero_utils.py#L44), [cosmos_utils.py:498–556](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/experiments/robot/cosmos_utils.py#L498) |
| Proprio | 9 raw floats: `[gripper_qpos0, gripper_qpos1, eef_xyz, quat_xyzw]`; native min/max normalization | [run_libero_eval.py:324–338](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/experiments/robot/libero/run_libero_eval.py#L324), [native normalization/inverse](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/experiments/robot/cosmos_utils.py#L623) |
| Temporal conditioning | 33 raw frames encode nine latent frames; four conditioning frames (blank, proprio, wrist, agentview), using the current observation only | [experiment:116–124](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py#L116), [native batch construction](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/experiments/robot/cosmos_utils.py#L900) |
| Actions | 16×7 chunk; apply16 controls before replanning; preserve native float64 physical actions without clipping or gripper sign changes | [LIBERO.md:57–58](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/LIBERO.md#L57), [native extraction/inverse](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/experiments/robot/cosmos_utils.py#L1127) |
| Default seed selection | **Single sample**: official `num_queries_best_of_n=1`. Its `max()` over that singleton is equivalent; no multi-seed value search is enabled | [default:235](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/experiments/robot/libero/run_libero_eval.py#L235), [query seed:449–462](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/experiments/robot/libero/run_libero_eval.py#L449), [selection:579–592](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/experiments/robot/libero/run_libero_eval.py#L579) |

The adapter reverses the stock R17 worker's extra horizontal image flip. The
Cosmos-only worker protocol supplies `observation/cosmos_proprio` from the
current raw simulator observation, preserving quaternion values before the
legacy axis-angle converter can mutate them. It uses environment seed0
([official environment](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/experiments/robot/libero/libero_utils.py#L30))
and separately binds policy seed195. Process RNG0 is reset at each decision,
matching the [official deterministic loop](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/experiments/robot/libero/run_libero_eval.py#L396).
Older R17 benchmark defaults retain their identities and behavior.

Future/value latents are still jointly sampled by the native policy. Serving
skips their unused image decoding/value extraction. Stage1 encodes the current
batch; stage2 uses the native `skip_vae_encoding` hook with **that same
request's latent**, then stage3 calls the native sampler. It retains no prior
observation or latent for a later decision. Sources: [native generation](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/models/policy_text2world_model.py#L702),
[conditioning hook](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/models/policy_video2world_model.py#L575).

## Verification evidence

CPU: **126 passed**, log
`/data/mw_finetune/qualification/cosmos_cpu_tests_release_final.log`.
Tests cover oriented pixels, lossless raw pose, genuine current-camera keys,
invalid/unsupported requests, normalization/dtype, activation-wrapper weight
auditing, legacy benchmark behavior and the existing conductor/ops suite.

Real recorded observations:
`/data/mw_finetune/qualification/observations/cosmos_libero_raw_v1/`.
Three raw OSMesa frames from original spatial task0/init0 after ten settles;
NPZ SHA256 `19d04b5d70b78804064a775cf05ed829bfa01d9857a869b058b1299109c59091`.
Native baseline calls the September vendor's `prepare_observation` and
`cosmos_utils.get_action`, including future decoding, with the same resident
weights/config/seed and released cached T5 embedding. The official ndarray
embedding overload supplies the same conditioning bytes as its string-cache
overload; all40 prompt tensors passed CPU byte equality in
`/data/mw_finetune/qualification/cosmos_prompt_argument_parity.json`.

| Recorded row | Native K / actual NFE | Action parity | Native / staged ms |
|---|---:|---|---:|
| 0 | 5 / 5 | byte-exact float64[16,7], max_abs0 | 2297 /1251 |
| 1 | 3 / 3 | byte-exact float64[16,7], max_abs0 | 1446 /792 |
| 2 | 1 / 1 | byte-exact float64[16,7], max_abs0 | 1180 /587 |

Receipt:
`/data/mw_finetune/eval/cosmos_libero_smoke2_release_v5/servers/replica_0/native_parity.json`.
No GPU-arithmetic tolerance was needed. H100 runs this same check at startup;
H100 numerical/closed-loop evidence is pending coordinator GPU admission.

Smoke root: `/data/mw_finetune/eval/cosmos_libero_smoke2_release_v5/`.

- `SUMMARY.json`: spatial task0/init0 **1/1**, object task0/init0 **1/1**;
  15 decisions, stage calls `[1,1,5]`, actual NFE5, zero errors/retries.
  Accepted workers are `timan108_0` and `timan108_1`.
- `driver/{journal,attempts,per_step}.jsonl`, `DRIVER_RECEIPT.json` and
  `workers/timan108/`: 75 +145 real controls, terminal acknowledgments and
  normal worker shutdown. The driver finished in64.6s.
- `SERVER_ATTESTATIONS.json`, `PROXY_METADATA.json`, `BUNDLE_ACK.json`,
  `BACKEND_EVIDENCE.json`: stock proxy, sealed full bundle and resident execution.
  Native service mean **2164ms/query** over15 queries while sharing the4090.
- `GPU_MEMORY_MONITOR.json`: peak owned physical VRAM **8300MiB =8.10GiB**,
  below15GB; steady resident5738MiB. The resident lifetime was123.8s.
- `RECEIPT.json`, `servers/replica_0/closed.json`, `TUNNELS_CLOSED.json`:
  PASS, all owned processes closed without forced kills, both owned routes closed.

Earlier immutable attempts `cosmos_libero_smoke2_release_v1..v4` stopped before
episode dispatch: tokenizer suffix handling, activation-wrapper audit names,
unpublished ports, then public-tunnel readiness. They were closed; the final
launcher fixes those issues. Only v5 dispatched the requested two episodes.
Protected DP/π0.5 ports and foreign processes were untouched.

H100 CPU formal400 preparation passed at
`/data/mw_finetune/eval/cosmos_libero_h100_formal400_release/PREFLIGHT.json`;
both G2 slot preflights passed under
`/data/mw_finetune/qualification/runs/cosmos_slot_{A,B}_preflight_final/`.
No claim of a400-episode success rate follows from this smoke.

## Commands

From `/home/weiland/projects/openpi`, launch the formal400 campaign only when
the coordinator admits H100 GPU use:

```bash
exp/r17_integration/ops/formal_eval.sh --model cosmos_libero --server-node h100 --worker-node timan108 --checkpoint /data/mw_finetune/models/cosmos_libero --replicas 1 --workers 24 --gpu cuda:0 --memory-budget-gib 10 --port 23240 --driver-port 23249 --run
```

Add `--official-50` for the separate four-suite2000-episode cohort (original
init0..49). Replace `--run` with `--prepare-only` for CPU sealing/deployment.
Default GPU choice follows [the official H100/PyTorch2.7 evaluation note](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/LIBERO.md#L80).
Run-owned Tether tunnels publish reachable policy/driver endpoints in
`TUNNELS.json`; internal ports23240..23249 do not use the protected DP lane.
Replicas remain independent B1 processes behind the unmodified stock proxy.

Exact successful local smoke:

```bash
flock -w 300 /tmp/r17_gpu.lock exp/r17_integration/ops/formal_eval.sh --model cosmos_libero --server-node local --worker-node timan108 --replicas 1 --workers 2 --memory-budget-gib 10 --smoke --smoke-episodes 2 --seconds 420 --episode-seconds 240 --retries 0 --run-name cosmos_libero_smoke2_release_v5 --run
```

Choose a new run name when repeating. CPU tests and slot commands:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:packages/openpi-client/src:. .venv/bin/python -m pytest -q exp/r17_integration/models/cosmos/test_cpu.py exp/r17_integration/tests/benchmarks exp/r17_integration/tests/conductor exp/r17_integration/tests/ops/test_ops.py
exp/r17_integration/qualification/slot_G2_cosmos_libero_A.sh --preflight
exp/r17_integration/qualification/slot_G2_cosmos_libero_B.sh --preflight
# Omit --preflight to run the bounded H100 qualification when admitted.
```

Local download verification and standalone descriptor setup:

```bash
CUDA_VISIBLE_DEVICES='' HF_HOME=/data/cosmos/hf_home /home/weiland/projects/openpi_ext/lines/cosmos-policy/.venv/bin/python -m exp.r17_integration.models.cosmos.download --verify-only
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m exp.r17_integration.models.cosmos.setup_runtime
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:packages/openpi-client/src:. /home/weiland/projects/openpi_ext/lines/cosmos-policy/.venv/bin/python -m exp.r17_integration.models.cosmos.descriptor --output /data/mw_finetune/qualification/assets/NEW_COSMOS_DESCRIPTOR
```

Both hosts are already provisioned. On a fresh host, run `download` without
`--verify-only`; configure the existing native VAE cache and copy private pure
Python imports using `setup_runtime`. Standalone `serve.py` takes the emitted
`load.json` and `full.yaml`, the native Python island/PYTHONPATH and its CUDA/
cuDNN library paths; the formal launcher supplies these automatically.

## Unsupported

Genuine VAE pooled camera keys are implemented, but **Cosmos CP1 retrieval is
`implemented_unverified`** and its enabled execution is rejected. Fitted
five-layer/R10/R11 recipes, library capture/quality, native warm resume/reset,
CP2, model snapshots/trace, B>1 stage batching, CPU/GPU split replay and model
shadow/routing integrations are unsupported and rejected. The formal full
arm needs none of them. CPU tests cover existing proxy/conductor mechanics;
the GPU smoke used one resident and does not certify multi-resident throughput.

K1/K3/K5 use native sampling. K2 is explicitly rejected: the upstream sampler
subtracts the clean step and its one-step branch then executes one denoiser
call ([cosmos_sampler.py:85–140](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/modules/cosmos_sampler.py#L85)).
Unknown prompts, missing paired views and reconstructed8-D axis-angle states
are rejected. There is no MetaWorld support or Cosmos fine-tuning recipe here.

## H100 runtime fix

The failed `formal_cosmos_libero_formal400_20261009_225545_b32d224f`
campaign used `CUDA_HOME=.../nvidia/cuda_runtime`. H100's installed TE loader
searches `CUDA_HOME` recursively for NVRTC, then calls `ldconfig` with
`check_output`; an empty system NVRTC listing raises before its final library
path fallback (`transformer_engine/common/__init__.py:101–123` in the Cosmos
venv). [formal_eval.py](../ops/formal_eval.py) now sets, only for Cosmos:

```text
CUDA_HOME=/data/cosmos/cosmos-policy/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc
CUDNN_HOME=/data/cosmos/cosmos-policy/.venv/lib/python3.10/site-packages/nvidia/cudnn
LD_LIBRARY_PATH=<cuda_nvrtc>/lib:<cudnn>/lib:<cuda_runtime>/lib:<existing path>
```

An H100 GPU-disabled import resolved `libnvrtc.so.12` from that wheel and
successfully imported TE/PyTorch. The first smoke then exposed a missing
`gymnasium` dependency in H100's private LIBERO imports. Installed two pinned
packages into **the Cosmos venv only**, preserving its Torch/CUDA packages:

```bash
tether exec --safe --timeout 60s h100 -- /home/exouser/.local/bin/uv pip install --python /data/cosmos/cosmos-policy/.venv/bin/python --no-deps gymnasium==0.29.1 farama-notifications==0.0.4
```

The native `PolicyEvalConfig` subsequently imported successfully with GPUs
hidden. There were no system changes or edits to other model environments.

`formal_eval.py` checks owned resident/proxy status during startup and serving,
before renewing leases; local exits and remote exits, including exit0, fail the
campaign. [formal_process.py](../ops/formal_process.py) reports child liveness
separately, so an alive supervisor cannot conceal a dead replica. Failure writes
`RECEIPT.json` immediately, prints the last6000 bytes of the replica log to
stderr, and enters the existing PID/start-time-checked shutdown. Cleanup errors
retain an existing FAIL status. The real GPU-disabled H100 pre-listen failure
probe returned FAIL and `all_owned_closed=true` in **1.434 seconds**, with its
traceback in the receipt and stderr:
`/data/mw_finetune/eval/cosmos_libero_h100_prelisten_failure_nvrtc_v1/{PROBE,RECEIPT}.json`.
Its intentionally missing campaign file ensured that no GPU model was loaded.
The first real smoke failure, preserved at
`/data/mw_finetune/eval/cosmos_libero_h100_smoke2_nvrtc_v1`, also failed promptly
and closed every owned process and tunnel when `gymnasium` was missing.

CPU verification: **138 tests passed**, plus18 subtests, including real local
pre-listen exit7, remote exit0/1, supervisor-alive/child-dead, FAIL receipts,
owned cleanup, and Cosmos-only environment scope. Command and log:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:packages/openpi-client/src:. .venv/bin/python -m pytest -q exp/r17_integration/ops/tests exp/r17_integration/models/cosmos/test_cpu.py exp/r17_integration/tests/benchmarks exp/r17_integration/tests/conductor exp/r17_integration/tests/ops/test_ops.py
# /data/mw_finetune/qualification/cosmos_h100_runtime_full_cpu_tests.log
```

**Real H100 smoke PASS: 2/2 successes**, two timan108 workers, original init0
for the spatial black-bowl task and object alphabet-soup task. There were13 K5
decisions, no errors or retries, mean resident service659.4ms/query, and23.9s
driver time. Startup native/staged parity was byte-identical for K5/K3/K1,
float64 `[16,7]`, max error0 and tolerance0. Ready-state memory was6022MiB;
the sampled peak was8462MiB (8.26GiB), within the10GiB cap. The resident closed
after65.3s. Both Fast-WAM PIDs41383/41455 remained present throughout sampling
and after cleanup, at26152MiB each. No local4090 GPU was used.

Evidence root: `/data/mw_finetune/eval/cosmos_libero_h100_smoke2_nvrtc_v2/`.
Read `SUMMARY.json`, `RECEIPT.json`, `GPU_MEMORY.json`,
`GPU_MEMORY_SAMPLES.json`, `TUNNELS_CLOSED.json`,
`servers/replica_0/{native_parity,closed}.json` and `serving.jsonl`.
All owned residents, proxy, driver, worker supervisor and tunnels closed;
ports23240/23241 were free afterward. Exact successful smoke command:

```bash
exp/r17_integration/ops/formal_eval.sh --model cosmos_libero --server-node h100 --worker-node timan108 --checkpoint /data/mw_finetune/models/cosmos_libero --replicas 1 --workers 2 --gpu cuda:0 --memory-budget-gib 10 --port 23240 --driver-port 23249 --smoke --smoke-episodes 2 --seconds 600 --episode-seconds 240 --retries 0 --run-name cosmos_libero_h100_smoke2_nvrtc_v2 --run
```

**Formal400 relaunch**, matching the interrupted campaign's one replica,
24 workers,10GiB budget and time bounds; the launcher creates a new immutable
result directory and applies the corrected environment automatically:

```bash
exp/r17_integration/ops/formal_eval.sh --model cosmos_libero --server-node h100 --worker-node timan108 --checkpoint /data/mw_finetune/models/cosmos_libero --replicas 1 --workers 24 --gpu cuda:0 --memory-budget-gib 10 --port 23240 --driver-port 23249 --seconds 43200 --episode-seconds 1800 --retries 2 --run
```

The full campaign has not been launched. The previously documented unsupported
model-specific features remain unsupported.

## Library capture

P6's minimal teacher collection is now supported. `manifest.py` declares
`debug_capture` for `CaptureSpec(enabled=True,snapshots=False)`;
`staged.py` accepts that profile only with native K5/seed195 and rejects
snapshots, extra capture fields and other samplers/seeds. `capture.py` supplies
the required post-MISS adapter hook with no additional model products. The
shared core already detaches the actual VAE pooled FeatureSpec keys, raw9-D
state and normalized16x7 chunk; P6 binds the two received uint8 frames and
identity/execution metadata. No new forward, latent copy or RNG capture occurs.
Capture is off by default. Native environment seed0, one-current-observation
conditioning, the16-control commit and physical action inverse remain unchanged.
General traces, native snapshots/resume, CP2 and fitted recipe serving remain
unsupported; this capability admits teacher collection only.

The GPU smoke exposed a P6 interoperability bug: B task/state hashes were
bare hex, whereas the stock worker's checked_setup uses `sha256:` + hex.
The transported state bytes, dtype/shape and task language matched exactly.
`ops/library_campaign.py`, `ops/library_alignment.py` and
`search/teacher_library.py` now use the worker's `benchmarks.spec.digest`
authority for that identity; raw state SHAs and guards are preserved. Related
P6 test fixtures were updated, including an explicitly unsupported capability
fixture for the existing negative seal check. Handoff:
[`D_to_P6_library_capture.md`](../REQUESTS/D_to_P6_library_capture.md).

**CPU:91 tests and18 subtests passed**, with capture-off/default action byte
identity, capture-on/off action and RNG parity, equal native forward counts,
the closed writer's exact allowlist, execution acknowledgments, and rejection
of unsupported profiles. The final command/log:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:packages/openpi-client/src:. .venv/bin/python -m pytest -q exp/r17_integration/models/cosmos/test_capture.py exp/r17_integration/models/cosmos/test_cpu.py exp/r17_integration/tests/debug/test_library_capture.py exp/r17_integration/tests/ops/test_library_campaign.py exp/r17_integration/tests/ops/test_library_finish.py exp/r17_integration/tests/search/test_teacher_library.py exp/r17_integration/ops/tests exp/r17_integration/tests/benchmarks
# /data/mw_finetune/qualification/cosmos_library_capture_cpu_release.log
```

Final focused rerun:23 tests passed, also verifying that disabled capture
ignores otherwise inactive snapshot/field flags, with the original sampler
seed-lookup order preserved. Log:
`/data/mw_finetune/qualification/cosmos_library_capture_cpu_off_final.log`.

**H100 library smoke/pipeline PASS:** spatial tasks0/1, original B init0,
two episodes,22 acknowledged decisions (14 +8), one task success and one normal
220-control step-cap failure; no inference/worker errors or retries. Native
K5/K3/K1 startup parity remained byte-identical. Mean captured resident service
was534.4ms/query, driver25.3s, resident63.3s. Ready-state physical VRAM6022MiB;
peak PyTorch allocation7727.3MiB, within the10GiB budget. The row audit verified
exactly `normalized_actions`, `state`, `image_0`, `image_1`, and rawkeys
`vision_0`, `vision_1`, `robot_state`: float32 `[16,7]`, `[9]`, `[256]` keys,
and two uint8 `[256,256,3]` images at `received_wire_uint8`. Every row has K5,
seed195, B/library provenance, identity fields and execution acknowledgment;
there are no snapshots, RNG, latents, physical actions or extra diagnostics.
Captured rows are excluded from pricing.

Evidence:
`/data/r17_library/cosmos_libero_libero_spatial/campaigns/cosmos_libero_library_spatial_smoke_v2/`.
Read `CAPTURE_ALLOWLIST_AUDIT.json`, `LIBRARY_PIPELINE.json`, `SUMMARY.json`,
`RECEIPT.json`, `servers/replica_0/{native_parity,closed}.json`,
`servers/replica_0/capture/shards/*/binding_*/r17_capture.json`, and
`smoke_parent/READY.json`. Both closed shards were SHA-verified on weilandserver;
the H100 staging high-water reservation was2.5GiB. All owned processes and
tunnels closed. **H100 flock was acquired, not skipped**, using a receipt-owned
supervisor with a renewed120s lease; its cleanup receipt is
`/data/mw_finetune/qualification/cosmos_library_gpu_lock_v2.json`.
The four original formal GPU PIDs were present in every memory sample from
the first probe (`../cosmos_libero_library_spatial_smoke_v1/GPU_MEMORY_SAMPLES.json`);
they exited during this work. Cleanup
signalled only this run's recorded process groups. No local4090 GPU was used.

Exact smoke invocation (run while holding the H100 lock):

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 bash exp/r17_integration/ops/library_campaign.sh --model cosmos_libero --suite libero_spatial --checkpoint /data/mw_finetune/models/cosmos_libero --server-node h100 --worker-node timan108 --port 23360 --driver-port 23379 --replicas 1 --workers 2 --memory-budget-gib 10 --smoke --seconds 780 --episode-seconds 600 --run-name cosmos_libero_library_spatial_smoke_v2 --run
```

The collection returned PASS, then its initial build found the same hash
namespace mismatch in the parent alignment check. After fixing that CPU
comparison, the published finish path recovered the original verified captures
without another GPU rollout:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:packages/openpi-client/src:. bash exp/r17_integration/ops/library_campaign.sh --model cosmos_libero --suite libero_spatial --finish /data/r17_library/cosmos_libero_libero_spatial/campaigns/cosmos_libero_library_spatial_smoke_v2
```

This returned0 and published the PASS22-row smoke parent. Full collection
command for the coordinator,500 B episodes per selected suite (not launched):

```bash
set -e
for suite in libero_spatial libero_10; do
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    bash exp/r17_integration/ops/library_campaign.sh --model cosmos_libero --suite "$suite" \
    --checkpoint /data/mw_finetune/models/cosmos_libero --server-node h100 --worker-node timan108 \
    --port 23360 --driver-port 23379 --replicas 1 --workers 24 --memory-budget-gib 10 \
    --seconds 172800 --episode-seconds 3600 --run
done
```

Use the coordinator's H100 GPU-lock admission around this command; the lock
may be skipped only when it is held by a running formal campaign, per the user
exception. P6's bounded shard transport, parent/subset construction and fitting
remain the stock path. Their existing limits on composed B500 LOEO serving and
unmeasured IR costs are not qualified by this capture smoke.

## K2 incl. clean step

Owner decision: 「那就只测k=2算上清理步」. The arm is **K2（含清理步）**,
with sampler ID `cosmos_edm_2ab_k2_incl_clean`, upstream `num_steps=2`, and
**measured actual NFE1**, not NFE2. This supersedes the earlier K2 rejection
in this handback and RECIPE.md. No scheduler or vendor code was changed.

The official policy installs `CosmosPolicySampler`
([policy_text2world_model.py:238](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/models/policy_text2world_model.py#L238)).
Its clean-step accounting subtracts one from K2
([cosmos_sampler.py:85–93](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/modules/cosmos_sampler.py#L85));
the resulting one-step branch calls the denoiser once at sigma80 and skips
the separate sigma4 clean evaluation
([cosmos_sampler.py:125–140](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/modules/cosmos_sampler.py#L125)).
The sealed two-point scheduler grid retains the framework's requested-K
length; `evaluated_sigma_indices=[0]` explicitly records which point is used.
`expected_nfe=1` is checked against counted adapter calls and the sealed
execution sampler during per-decision aggregation. SUMMARY.json and parity
receipts carry the exact arm label. K1/K3/K5 sampler identities and default
K5 inference remain unchanged; teacher capture still admits only K5/seed195.

Files: `models/cosmos/{manifest,staged,serve,qualify,test_cpu,test_capture}.py`,
`models/cosmos/RECIPE.md`, and the Cosmos step choices, NFE audit and summary
label in `ops/formal_eval.py`; the invalid-K CPU case in
`ops/tests/test_formal_steps_suites.py` now uses K4. Startup native parity adds
K2 while retaining K5/K3/K1. The standalone qualifier accepts `--steps 2 2 2`
to exercise all three recorded observations at K2. No queue scripts or live
campaigns were changed.

September consistency: [cosmos_nfe_plan.log.md:46](/home/weiland/projects/openpi/logs/cosmos_nfe_plan.log.md:46)
and its final table claimed K+1 calls, from a dummy probe of the **base**
`res_sampler.Sampler`. That is not the model's installed policy sampler.
CPU probes of the actual upstream classes give policy K1/K2/K3/K5 NFE
**1/1/3/5**, versus base-sampler **2/3/4/6** at the LIBERO sigma range.
The September runner itself passed requested K unchanged to official
`get_action`: [serve_cosmos.py:155](/home/weiland/projects/openpi/exp/cosmos_nfe/serve_cosmos.py:155),
and `run_libero_shard.py:140–146` passes the same integer through its remote
request. The vendor remains commit `18a2accadf4e7a3531e56754102af5a24d2316da`
with no local sampler/model diff. This implementation follows that runner's
call semantics; it does not replace the sampler with the base class to
reproduce the log's mistaken NFE interpretation. The logged identical K1/K2
spatial episodes/latencies are consistent with the policy's one-call branch.
Historical K2 SRs of .974 spatial and .972 libero_10 are log-reported values;
K2 shards are absent from the current local result roots and the timan107
September root, so their per-decision NFE cannot be independently re-audited.
Source hashes and this distinction are recorded in `STUDY_CONSISTENCY.json`.

Evidence root:
`/data/mw_finetune/qualification/cosmos_k2_incl_clean_20261010_161725/`.
`native_parity_k2.json` compares the R17 adapter with official cached-prompt
`get_action(num_denoising_steps_action=2)` on recorded spatial task0/init0
observations0,1,2, seed195 and process/env seed0. All three are **byte-exact
float64[16,7], max_abs0, tolerance0**. The official sampler's denoiser was
counted independently with a temporary forward pre-hook, then removed:
each native call and each adapter call has actual NFE1, sigma trace `[80]`.
The receipt includes sampler contents, action SHAs, versions and observation
identity. `parity.log`, `assets/`, `gpu_argv.json` and `probe_runner.json` retain
the reproducible invocation and CPU content seals.

The 4090 was idle at admission; `GPU_ADMISSION.json` confirms no compute
processes and successful `flock /tmp/r17_gpu.lock`. Only this probe ran, with
a 280-second hard bound. `GPU_CLOSED.json` records PASS in 44.65 seconds,
peak CUDA allocated7727.27MiB/reserved7852MiB. The process exited normally and
the GPU returned idle. No H100 run, episode campaign, or other process was
started/stopped. CPU evidence is `cpu_tests.log`; gates cover the real upstream
CPU sampler counts, K2 CLI/sealing, genuine serving stage receipts and
aggregation rejection of NFE2, wrong sampler identity and altered NFE seals,
plus unchanged capture-off and K5 teacher capture behavior. Final CPU result:
**78 passed, 27 subtests passed** (117.49 seconds; one existing pynvml
deprecation warning). Exact verification command:

```bash
.venv/bin/python -m pytest \
  exp/r17_integration/models/cosmos/test_cpu.py \
  exp/r17_integration/models/cosmos/test_capture.py \
  exp/r17_integration/ops/tests/test_formal_eval.py \
  exp/r17_integration/ops/tests/test_formal_steps_suites.py \
  exp/r17_integration/ops/tests/test_formal_horizon.py -q
```

Exact coordinator command (not launched):1000 original-init episodes,
500 for each requested suite. The20GiB resident budget covers two replicas
instead of the single-replica10GiB default. Use it when the reserved ports and
H100 capacity are available; this task did not modify their current owners.

```bash
CUDA_VISIBLE_DEVICES='' bash exp/r17_integration/ops/formal_eval.sh \
  --model cosmos_libero --server-node h100 --replicas 2 \
  --worker-node timan108 --workers 12 --worker-memory-mib 6144 \
  --port 23240 --driver-port 23249 --memory-budget-gib 20 \
  --official-50 --suites libero_spatial libero_10 --denoising-steps 2 --run
```

The exact command is also saved as `formal_command.sh` in the evidence root
and passed the CPU argument parser. Existing H100 TE environment handling,
stock resident proxy, native16-control commitment and current-frame-only
inference apply unchanged. No K2 closed-loop success rate has been measured
in this task; the coordinator owns that campaign.

## Official2000 stall and recovery diagnosis (2026-10-10)

Source campaign:
`/data/mw_finetune/eval/formal_cosmos_libero_official2000_20261010_100533_5f558380`.
Diagnostic evidence is in its `recovery/` directory. The blocked component
was **the timan108 simulator worker fleet**: all24 receipt-bound workers
2338056–2338079 were in kernel `D` state at `rwsem_down_write_slowpath`,
with the correct Cosmos worker parent2338055 and recorded process start times.
They were not idle socket clients. Kernel stack/syscall inspection was denied
by host permissions, so the exact kernel lock owner cannot be attributed.
The GPU0 NVML query exited-9; node memory was healthy (~139GiB available,
memory PSI0). No GPU reset, driver reload, agent restart or other-user signal
was attempted. xy51 GPUs1–2, the Fast-WAM campaign and local4090 jobs were
not touched.

`policy_layers.json` proves WebSocket metadata, not just TCP readiness,
passed through linziyang.top:14008, the direct H100 proxy23240, and both
replicas23241/23242 (~0.1seconds each). Metadata identified `cosmos_libero`.
The local driver580276 remained in poll; both model residents remained in
their event loops, with healthy leases. Their last serving rows were around
11:31:38 and11:32:08; original driver journal/dispatch stopped11:30:53.
`WORKER_BLOCK.json`, `BASELINE_COUNTS.json` and `DP_ISOLATION_AUDIT.json`
retain the worker states, resource facts and campaign ownership comparison.

DP shutdown did **not** share Cosmos ports, tunnel names, PGIDs, leases,
payloads or worker run directories. DP used no Tether tunnels. Both fleets
used physical EGL GPU0, with24 Cosmos plus20 DP workers. Cosmos framebuffer
incomplete0x8cdd errors occurred near the DP stop. DP's forced parent shutdown
left seven start-time-matching orphan workers in the same D-state wait:
2431437/2431442/2431443/2431445/2431449/2431450/2431451. Its fleet receipt
remained RUNNING while top-level cleanup advertised `all_owned_closed=true`.
The definite software bug was closure checking the supervisor/fleet parent
only. GPU context contention/driver locking during forced teardown is a
plausible indirect cause of the stall; the unavailable kernel stack prevents
claiming a proven driver root cause or a direct cross-campaign signal.

During diagnosis, timan108's Tether agent went OFFLINE. The original
coordinator then exited FAIL after control/cleanup timeouts, closed the H100
proxy/residents and original tunnels, and honestly retained
`all_owned_closed=false` for the unconfirmed worker cleanup. The original
driver is gone, so in-place recovery is no longer available. This closure
was the coordinator's own error path; I did not kill the original services.
See `CLOSURE_SNAPSHOT.json` and the original RECEIPT/DRIVER_RECEIPT files.
Direct SSH could not resolve timan108 either locally or via timan107; I did
not use forbidden timan1. No H100 or4090 GPU job was launched by this task.

The existing queued K2 arm had already entered setup by the time I tried to
hold its verified shellPID1110034 with SIGSTOP. That PID subsequently exited;
I did not resume or signal a reused PID. Its timan108 setup also failed with
the offline agent and published no resident receipt. An external controller
then launched the timan107 variant
`hs2_cosmos_h100_k2incl_c16_t107_20261010_115120` (PID1300454 observed).
I did not signal that replacement. It reserves the baseline's23240/23249
ports. `K2_QUEUE_HOLD.json`, `K2_LAUNCHER_HOLD.json`, `K2_STATUS.json` and
`LOCAL_LAUNCHERS.json` record these observations. Queue scripts are unchanged.

Fixes for future launches:

- `ops/worker_gpu.py` holds a per-EGL-device POSIX lock for the entire fleet
  lifetime. It also reads legacy/orphan fleet receipts and verifies live
  descendant PID start times, regardless of the advertised receipt status.
  Another R17 fleet on the same renderer device is refused before any worker
  Popen. Separate renderer devices remain independent. This intentionally
  serializes R17 simulator fleets sharing one physical EGL GPU; it does not
  reserve policy-server GPUs or signal another fleet/user.
- `ops/formal_workers.py` acquires that guard before launching children and
  releases it after normal cleanup. Kernel-D survivors remain discoverable
  through their receipt even when a killed parent releases its file lock.
- `ops/formal_process.py` status/stop includes receipt-bound worker descendants
  in `alive`. A dead parent with live descendants cannot certify closure;
  unavailable worker evidence also fails closed. Existing start-time/PGID
  signal guards remain unchanged.
- Only the worker-payload module list in `ops/formal_eval.py` was extended.
  Q's resume lines were not edited. Coordination, source-runtime caveats and
  the current block are in `REQUESTS/D_to_Q.md`.

CPU verification: **70 tests +27 subtests PASS**,72.84seconds, one existing
pynvml deprecation warning. The run covered `test_worker_gpu.py`,
`test_concurrent_fleets.py`, `test_worker_lifetime.py`, `test_formal_eval.py`
and `test_formal_resume.py`. New tests use real CPU file locks, prove same
device refusal/different device admission, preserve the other fleet's receipt
bytes, reject orphan reuse, fence PID reuse, prevent Popen on refusal, and
prove a stopped parent with live workers cannot report closure. Existing
CPU worker replacement/shutdown isolation and resume tests continue passing.
Evidence: `FIX_VERIFICATION.json`; reproducible command:

```bash
.venv/bin/python -m pytest exp/r17_integration/ops/tests/test_worker_gpu.py \
  exp/r17_integration/ops/tests/test_concurrent_fleets.py \
  exp/r17_integration/ops/tests/test_worker_lifetime.py \
  exp/r17_integration/ops/tests/test_formal_eval.py \
  exp/r17_integration/ops/tests/test_formal_resume.py -q
```

Q's actual `formal_resume.accepted_evidence` validates **1726 reusable
completed cells and exactly274 missing cells**, including all legitimate
unsuccessful episodes. `RESUME_PREFLIGHT.json` records PASS for their original
UID/seed/model/sampler/normalization/commit/per-decision evidence, but BLOCKED
at source closure. I did not falsify `all_owned_closed` or relax probe/config
checks. Q now preserves the pinned source runtime archive, necessary because
the Cosmos content identity also hashes its wrapper source; K2 edits must
not replace the baseline's old K5 code during continuation.

No continuation was launched. Safe resume requires timan108 restored, old
workers verifiably closed, and the K2 port owners closed. The original
timan108 Python3.12.13/probe binding must remain intact; timan107 uses
Python3.11.15 and has the protected Fast-WAM EGL0 fleet, so it was not used as
an unqualified substitute. Once those conditions hold, Q's resume path keeps
the original semantic campaign/cell identities and emits one merged2000-cell
receipt rather than rerunning completed cells. The baseline is currently
incomplete; this task cannot claim that it finished as one campaign.

Staged exact continuation command, saved as
`recovery/RESUME_COMMAND_NOT_LAUNCHED.sh` (CPU argument parser passed):

```bash
CUDA_VISIBLE_DEVICES='' bash exp/r17_integration/ops/formal_eval.sh \
  --model cosmos_libero --checkpoint /data/mw_finetune/models/cosmos_libero \
  --server-node h100 --replicas 2 --worker-node timan108 --workers 12 \
  --worker-memory-mib 6144 --worker-max-episodes 8 --memory-budget-gib 18 \
  --port 23240 --driver-port 23249 --seconds 43200 --episode-seconds 1800 \
  --retries 2 --official-50 \
  --resume-from /data/mw_finetune/eval/formal_cosmos_libero_official2000_20261010_100533_5f558380 --run
```

Worker count/memory/lifetime are resource knobs; native K5, seed195/env0,
paired current cameras and16-control commitment are preserved. Owner was
asked asynchronously whether host/agent maintenance is underway. No response
or unsafe hardware recovery was assumed.

## LIBERO renderer distribution and memory probe (2026-10-10)

**Outcome:** future formal and teacher-library LIBERO fleets now distribute
workers round-robin over usable physical GPUs, with per-card VRAM admission.
The short distributed-render smoke passed. GPU concentration is supported
as a cause of resource pressure; the specific NVIDIA VRAM-to-host backing
mechanism is **not directly confirmed or refuted** by the accessible counters.
No running campaign was stopped, signalled, restarted or redeployed.

Changed files: `ops/worker_gpu.py`, `ops/formal_workers.py`,
`ops/formal_eval.py`, their four CPU test files (`test_worker_gpu.py`,
`test_formal_eval.py`, `test_worker_lifetime.py`, `test_concurrent_fleets.py`),
and `diagnostics/libero_gpu_probe.py`. Coordination/evidence is in
`REQUESTS/D_to_M.md`, `D_to_Q.md`, and `D_to_P6.md`. M's diagnostic/recycling
code, Q's formal_resume, native simulator/episode code, and model code are
unchanged by this work.

`worker_gpu.inventory` reads NVIDIA graphics **and** compute allocations,
including other R17 fleets and unrelated allocations. Other-user GPUs are
excluded. timan108 GPU1/2 are excluded **before any NVIDIA query**. An
unqueryable card is unavailable. Admission includes actual VRAM plus pending
reservations, avoiding double-counting already resident verified workers.
Atomic `GPU_RESERVATION.json` under `.egl_admission.lock` prevents overlapping
new fleets from reserving the same free memory. Live legacy/orphan worker
PIDs and live owners' recycle gaps still reserve their slots. Replacements
recheck external GPU use before spawning; no other fleet is signalled.

Default VRAM reservation is1024MiB/worker, plus max(512MiB,10% VRAM) headroom
per card. `--worker-egl-devices 1,2,...` optionally restricts physical devices;
the default considers all usable cards. `--worker-egl-device` is the preferred
starting ordinal for LIBERO; MW retains the prior single-device guard.
Mixed `--extra-workers` fleets apply the worker-node exclusions independently.
Each fleet receipt records `worker_gpu_assignments`, inventory/admission,
and each PID's `egl_device`, `gpu_uuid`, and `worker_vram_mib`.

The native selection path is robosuite
`renderers/context/egl_context.py:36–68`: it reads
`MUJOCO_EGL_DEVICE_ID` and selects the physical EGL device. The wrapper sets
that variable **before importing robosuite**, plus the matching
`--egl-device` argument. Worker CUDA visibility is empty; native EGL still
selects the physical device. Camera creation remains the stock
`examples/libero/main.py:1236` `_get_libero_env` path,256² paired cameras,
original init pools and env seed0. Samplers, seed195, action commits, image
rotation, state, and observation preprocessing are unchanged.

Future LIBERO defaults are four episodes/process and6144MiB RAM. The same
fallback is applied in `prepare_workers` to P6's library campaigns, which
previously omitted these resource fields. Explicit resource flags remain
supported. This addresses lifecycle/host allocation separately from VRAM.

### Smoke evidence and limits

All evidence, executed script revisions, raw NPZ observations, logs, traces,
and archives are retained at:
`/data/mw_finetune/diagnostics/egl_D_20261010/`.
`EVIDENCE.json` contains the measured summaries and every camera difference;
`VRAM_ADMISSION.json` proves the real new planner excluded the existing
GPU0 fleet and selected GPUs1–7. The probes reused the already installed
immutable native worker payload/assets, used no policy endpoint and loaded
no policy model. Only owned probe child processes could be terminated.

| Probe | Workload | Result | MemAvailable GiB |
|---|---|---|---|
| First,263s |12 workers on GPUs1–7; spatial/task0 and libero_10/task9, init0/1;300 no-op controls/state |12/12 exit0; no FBO errors |129.33 →47.63 minimum →134.91; live K2 background confounds this trace |
| Supplemental,136s |4 fresh-env workers on1/2/6/7;150 controls/state |Diagnostic hit robosuite's episode horizon while trying to hold by stepping; all four exit1; failure retained |94.64 →86.08 minimum →118.84 |
| Corrected,158s |Same4 workers; hold final context by rendering rather than stepping past the horizon |4/4 exit0; no FBO errors |149.66 →216.28 |

Total elapsed across the three bounded probes was~558s (9m18s), including
baseline sampling and cleanup. GPU0 was never used by our renderers. Twelve
workers/seven free cards replaced the suggested16/eight because GPU0 was
protected and the live K2 fleet's host-memory envelope needed headroom.

K2 naturally reached1000 accepted during the corrected probe, then its
workers exited without intervention. With four own contexts still actively
rendering and **zero** background workers, the final22.63s had
MemAvailable206.437→206.656GiB and summedPSS9.91866→9.91867GiB. Each
renderer held428MiB VRAM; sampled per-PID peak was532MiB. After the owned
workers exited, MemAvailable216.270→216.276GiB. This shows no continuing
depletion in the short uncontaminated interval. It does not establish a
ten-minute steady plateau, a full-suite memory envelope, or the exact
driver allocation mechanism. The first trace cannot be called globally flat.

GPU1/GPU2 produced byte-identical **both camera frames and all three raw
proprio fields** for all four fixed states in the first two probes. Wrist
and proprio remained exact throughout. Later observations show1–2 one-LSB
agentview channel differences, which also occur when repeating on the
**same GPU** (e.g.GPU2 between the second and third probes). Raw arrays and
elementwise differences are preserved. Therefore the required sampled
two-card parity passes, but universal native pixel repeatability is not
guaranteed; no image quantization/dithering change was introduced to hide it.

Final CPU verification:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/pytest -q \
  exp/r17_integration/ops/tests/test_formal_eval.py \
  exp/r17_integration/ops/tests/test_worker_gpu.py \
  exp/r17_integration/ops/tests/test_worker_lifetime.py \
  exp/r17_integration/ops/tests/test_concurrent_fleets.py \
  exp/r17_integration/ops/tests/test_formal_resume.py \
  exp/r17_integration/tests/benchmarks/test_libero_worker_recycle.py
```

**85 tests +27 subtests passed**,71.02s. Gates cover protected-device queries,
foreign users, global actual use, future/recycle reservations, live orphans,
cross-fleet refusal before Popen, per-child import-time device env/argv and
receipt, replacement admission, mixed-node defaults, semantic episode
recycling, and resume fences.

### Per-worker resources and capacities

Observed native worker peak RSS was2.63–2.81GiB and peakPSS2.30–2.51GiB;
the final rendering plateau averaged2.480GiB PSS/worker. GPU allocations
were428MiB resident/532MiB sampled peak. Reserve **6144MiB RAM and1024MiB
VRAM per worker**, with four-episode recycling. RAM includes M's conservative
unattributed driver/episode envelope; RSS alone is not the admission budget.

| Node | RAM budget capacity | VRAM budget capacity | Recommendation |
|---|---:|---:|---|
| timan107 |32 at6GiB,10% node headroom and~4GiB OS |56 across8×8GiB,7/card at1GiB plus820MiB/card headroom |32 on a clean node; dynamic admissions reduce this for live fleets |
| timan108 |37 when healthy |21 if historical3×A5000/24GiB inventory still holds and GPU1/2 remain xy51's |Current0: OFFLINE/unverified old D workers; after recovery min(RAM,live usable GPU admission) |

Historical timan108 topology source:
`/home/weiland/projects/dist_experiment_control/docs/devices.md:338`.
Its current topology and per-worker rendering footprint cannot be remeasured
while offline; no GPU3 existence is assumed.80–90 workers are **not supported
by the measured conservative envelope**, even after spreading. Increasing
to that count would require a separate full-suite/multifleet capacity study.

### Baseline274-cell resume

The native CPU simulator probe on timan107, run with the baseline's sealed
benchmark, matches **every** source `PROBE_FIELDS` value: native code,
robosuite,torch,numpy, task table, and original init-state hash. Compare
`BASELINE_PROBE.json` and `TIMAN107_RESUME_PROBE.json`. The different Python
minor versions are not part of this recorded binding.

However, the baseline's source receipt still truthfully has
`all_owned_closed=false`, and timan108 is OFFLINE. Q's closure gate refuses
resume. **It cannot safely launch yet**;1726 accepted/274 missing are preserved.
See `RESUME_GATE.json` and baseline
`recovery/EGL_FIX_RESUME_GATE.json`. No source receipt was rewritten and no
closure/probe gate was bypassed. After verified source closure, use this
parser-checked command (stored as `RESUME_COMMAND.sh`, not launched):

```bash
exp/r17_integration/ops/formal_eval.sh \
  --model cosmos_libero --checkpoint /data/mw_finetune/models/cosmos_libero \
  --server-node h100 --replicas 2 --worker-node timan107 --workers 12 \
  --worker-memory-mib 6144 --worker-max-episodes 4 --worker-vram-mib 1024 \
  --memory-budget-gib 20 --port 23243 --driver-port 23248 \
  --seconds 43200 --episode-seconds 1800 --retries 2 --official-50 \
  --resume-from /data/mw_finetune/eval/formal_cosmos_libero_official2000_20261010_100533_5f558380 \
  --run
```

Policy23243, backends23244/23245, driver23248 avoid K2's23240/23241/23242/23249.
The original native K5 arm is selected by omitting a denoising override;
source runtime/weights/config pins are preserved by Q's resume path, which
dispatches only the274 missing cells and merges accepted successes **and
legitimate failures** into one semantic receipt. This command supersedes
the older timan108/23240 continuation example above.


## Unreachable-source-node resume waiver

Owner authorized `--resume-waive-unreachable-source-node timan108` to
resume the 274 missing cells while timan108 remains offline. This supersedes
the earlier closure-blocked command above. D **did not launch** a campaign,
use a GPU, signal a process, renew a lease, or modify the source receipt.

Files: `ops/formal_resume_waiver.py` (new isolated proof/transport helper),
small hooks in `ops/formal_resume.py` and `ops/formal_eval.py`,
`ops/tests/test_formal_resume_waiver.py`, this handback, and
`REQUESTS/D_to_Q.md`. Q's concurrent `formal_prestage.py` change from
`RESUME_SELECTION.json` to `RESUME.json` was observed and preserved; neither
that file, Q's queue helper, nor P3's batch/server code was edited by D.

The explicit exception requires all of these: terminal source FAIL/PASS;
receipt-bound local driver, source proxy and every replica confirmed closed;
only the named worker node unverified; Tether exit64 with typed `node_offline`
evidence for that node; a different destination run id; and disjoint direct
**and public broker** ports. Missing cleanup/process identities, another
unverified node, a live child/supervisor, a reachable node, or a generic
transport timeout fails closed. Destination workers cannot use the waived
node. The default `all_owned_closed` gate is unchanged when no flag is given.
Fresh /proc PID/start-tick checks and offline checks run during selection
and again before tunnel allocation, including prepared launches. The helper
only reads source process receipts and /proc; it never uses lease-renewing
`controls` or stop operations.

Waiver schema: `r17.resume_unreachable_source_node_waiver.v1`, with issue/check
times, source receipt/campaign SHA256s, verified process identities and
remote receipt SHA256s, offline evidence, explicit unverified worker identity,
and destination port evidence. It is persisted as
`campaign.resume_source_closure_waiver`, `RESUME.source_closure_waiver`,
`RECEIPT.source_closure_waiver`, and
`SUMMARY.resume.source_closure_waiver` / `RESUME_MERGE.source_closure_waiver`.
Merge rejects a selection proof that differs from the campaign proof.
The source receipt remains FAIL with `all_owned_closed=false`.

Source public ports were14008/14009 in addition to direct
23240/23241/23242/23249. Auto broker allocation can recycle those ports:
the new allocator closes only its **new run's** conflicting route, then
requests a non-source port, skipping occupied ports using Tether's typed
`port_taken` error. It does this before any service or worker starts.
The public-port choices and isolation time also reach prepared resumes'
merged proof. No tunnel was opened or closed during this task.

Actual read-only preflight: **1726 retained /274 missing**, source status
FAIL, local driver and both H100 replicas/proxy child+supervisor identities
closed; timan108 returned64/node_offline. Evidence:
`/data/mw_finetune/diagnostics/resume_waiver_D_20261010/`
(`CLOSURE_PREFLIGHT.json`, `RESUME_PREFLIGHT.json`, `BEFORE_SHA.json`,
`AFTER_SHA.json`, `RESUME_COMMAND.sh`). Proof issued at 2026-10-10T17:47:04.810068+00:00.
The earlier native CPU binding probe on timan107 still establishes parity
with the source simulator/init-state binding; no GPU inference was rerun.

CPU verification: **76 tests +27 subtests passed** (74.25s), including
explicit opt-in, all failure gates, real remote verifier response rejection
for live PIDs, public-port reuse/occupied-port handling, refreshed prepared
proofs, capture of waiver in a FAIL receipt before any spawn, immutable source
receipt, and the 1726+274=2000 merge/SUMMARY propagation. The existing test
that launches/signals a temporary process was deselected to honor this
task's no-signals rule. Log: `CPU_TESTS_FINAL.log` in the evidence directory.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q \
  exp/r17_integration/ops/tests/test_formal_resume_waiver.py \
  exp/r17_integration/ops/tests/test_formal_resume.py \
  exp/r17_integration/ops/tests/test_formal_eval.py \
  -k 'not cleanup_never_signals_a_reused_pid'
```

Coordinator launch command (parser checked; **not launched by D**):

```bash
exp/r17_integration/ops/formal_eval.sh \
  --model cosmos_libero --checkpoint /data/mw_finetune/models/cosmos_libero \
  --server-node h100 --replicas 2 --worker-node timan107 --workers 12 \
  --worker-memory-mib 6144 --worker-max-episodes 4 --worker-vram-mib 1024 \
  --memory-budget-gib 20 --port 23243 --driver-port 23248 \
  --seconds 43200 --episode-seconds 1800 --retries 2 --official-50 \
  --resume-from /data/mw_finetune/eval/formal_cosmos_libero_official2000_20261010_100533_5f558380 \
  --resume-waive-unreachable-source-node timan108 \
  --run
```

Proxy23243; replicas23244/23245; driver23248. The launcher creates a fresh
run id and chooses isolated public ports. Native K5 /seed195 /env0 /16-control
commit stay bound to the original archived runtime, manifests, benchmark,
per-cell UID/seed/init-state identities and worker probe. Omit denoising and
commit overrides to preserve the exact source config. Workers use the new
round-robin renderer/VRAM admission over usable timan107 GPUs,1GiB VRAM and
6GiB RAM reserved per worker, recycling every4 episodes. Resource admission
continues to account for other fleets/users. Only the274 missing cells are
dispatched, then joined with all1726 accepted outcomes into one 2000-cell
semantic result and merged receipt, without changing the source evidence.

## Completed K5 baseline recovery

CPU-only recovery of
`/data/mw_finetune/eval/formal_cosmos_libero_official2000_20261010_130639_a8ca2043`
is complete. All **274 resumed cells are valid: 268 success, 6 normal task
failures, no episode errors, all attempt 1**. No episode needs rerunning.
The raw driver had `all_done=true`, zero rejected probes, and only 7/8 final
worker shutdown acknowledgments. Worker7's last recycled generation logged
`peer closed connection mid-frame`; this is a shutdown failure after all
outcomes were accepted. The parent FAIL receipt already proved
`all_owned_closed=true`; fresh read-only PID/start-tick checks confirmed the
local driver, H100 proxy/replicas, and timan107 fleet were closed. No process
was launched, stopped, or signaled during this recovery, and no GPU was used.

The source segment had exactly one accepted error attempt:
`r17_9a1874a9bfb19f8b583b_full:eval:34:48`, task
`libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate`,
original init48. Source attempt1 errored before any controls with
`FatalError('Offscreen framebuffer is not complete, error 0x8cdd')`.
The resumed physical driver starts its attempt ordinals again at1; that cell
finished successfully at resumed attempt1. Both segments have identical
sealed task/seed/init/config bindings (policy seed195, environment seed0).
A CPU probe of the pinned timan107 LIBERO runtime loaded every original pool
state, confirming all274 native init hashes and all4,676 policy decisions'
seed/task/init binding and native K5 NFE5. The recovered error cell's hash is
`sha256:e3580af569509e778b61d9ad6e8379cfac950e8a2977d042c6569f3cdf8c0d8e`.
Its source environment construction failed, so there is no source control
trace to compare. The same intended native init is verified rather than
inferred from attempt numbers. Cross-segment retries are recorded with both
physical campaign identities; the failed row is not inserted into a new
driver's attempt1 namespace.

Changes: `ops/formal_eval.py` now accepts exit2 only with a complete unique
accepted journal, `all_done`, zero rejected probes, and a complete consistent
attempt history plus explicit retry/shutdown evidence. It records resolved
retry generations and missing-ACK recovery separately. The future launcher
may drain only its own verified completed worker fleet and still requires
closed-process/snapshot evidence before PASS. This drain path was mocked in
tests and was not invoked during this task. Tests are in
`ops/tests/test_completed_driver_drain.py` and `ops/tests/test_formal_eval.py`.
New `models/cosmos/recover_baseline.py` and `test_recovery.py` implement the
closed-campaign audit, stock resume merge/aggregate, immutable raw receipt
preservation, and SHA seal. Q's prestage/queue files were not edited.

The stock merge retains1726 source outcomes and joins274 resumed outcomes:
**2000 unique cells, 40 tasks ×50 trials, 1965 success, SR98.25%, PASS**.
The source ran on timan108 with24 workers; the actual resumed segment used
8 timan107 workers (all274 resumed cells belong to libero_10). The merged
proof preserves the source's timan108 closure waiver, archived runtime and
config pins, raw source receipt, and raw resumed driver receipt. Historical
`RECEIPT.json` stays FAIL; the explicit `RECOVERED_RECEIPT.json` is PASS.

| Suite | K5 successes/episodes, SR | K2（含清理步） successes/episodes, SR |
| --- | --- | --- |
| libero_spatial | 482/500, 96.4% | 483/500, 96.6% |
| libero_object | 498/500, 99.6% | not evaluated |
| libero_goal | 492/500, 98.4% | not evaluated |
| libero_10 | 493/500, 98.6% | 483/500, 96.6% |

K2 comparison run:
`/data/mw_finetune/eval/hs2_cosmos_h100_k2incl_c16_t107_20261010_115120`;
all1000 cells ran on timan107, actual NFE1. K5 spatial/object/goal ran wholly
on timan108; K5 libero_10 combines226 timan108 cells +274 timan107 cells.
The worker-node difference is a provenance limitation for comparisons.

Evidence under the recovered run: `SUMMARY.json` includes per-suite and
per-task SR; `RESUME_MERGE.json` includes segment provenance and the resolved
source error; `RECOVERED_RESULT_SEAL.json` pins30 result/input/evidence files
plus archived runtime/source pins. `resume_segment/driver/` preserves all
raw274-cell driver files. `recovery/` contains `CLOSURE_AUDIT.json`,
`NATIVE_INIT_AUDIT.json`, `NATIVE_INIT_PROBE.json`, `EPISODE_AUDIT.json`,
`SEGMENT_EPISODE_PLAN.json`, `PER_SUITE_SR.csv`, `PER_TASK_SR.csv`, and
`K5_K2_COMPARISON.json`. Seal re-verification and independent journal
union/count checks passed after producing the comparison exports.

CPU verification: comprehensive suite **96 tests +27 subtests passed**;
after adding the cross-segment retry case, the final recovery suite passed
all10 tests, giving97 distinct tests verified overall. The existing test
that signals its own temporary subprocess was excluded for this task.
Logs: `recovery/CPU_TESTS.log` and `recovery/CPU_RECOVERY_FINAL.log`.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q \
  exp/r17_integration/ops/tests/test_completed_driver_drain.py \
  exp/r17_integration/ops/tests/test_formal_eval.py \
  exp/r17_integration/ops/tests/test_formal_resume.py \
  exp/r17_integration/ops/tests/test_formal_resume_waiver.py \
  exp/r17_integration/models/cosmos/test_recovery.py \
  -k 'not cleanup_never_signals_a_reused_pid'

# Already completed; repeat only verifies the seal/source pins and returns PASS.
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m exp.r17_integration.models.cosmos.recover_baseline \
  --directory /data/mw_finetune/eval/formal_cosmos_libero_official2000_20261010_130639_a8ca2043
```


## Pure-cache study admission (CPU only, 2026-10-10)

Cosmos is admitted to the explicit P6 cache-study serving path: real B500
libraries, native K5/C16, A500/suite testing, all eight logical labels via YAML
on one resident fleet. GPU qualification is pending by owner instruction.
No running campaigns/processes, frozen runtimes, GPU contexts or queues were
modified. P6 preparation installed a new immutable CPU-sealed runtime only.

| P6_to_D item | Status and implementation |
| --- | --- |
| 1. Recipe, key-only HIT, native output transform, shared-resident clones | **Done**, `models/cosmos/provider.py:5,26`; `cache_probe.py:46`; `test_cache.py:34`. Only explicit sealed study recipes are admitted. Pure YAML `never` hits run stage1 only, normalized actions pass once through the official minmax inverse, and every new connection has fresh wrapper/key/recipe state over the same model/lock. |
| 2. Shorter cache commitment / long horizon | **Done as withdrawn**: `manifest.py:34`; `provider.py:13`. Native teacher/library cadence stays16. C0=CL=16; ch→c, chr→cr, chg→cg, chrg→crg have identical bundle content identities. No partial commitment, manufactured tail, new H or library edits. |
| 3. Teacher compatibility callback / CPU hook | **Done**, `library_admission.py:56,120`; `provider.py:21`; generated literal provenance `cache_teacher.py:6`. Callback returns `r17.teacher_compatibility.v1/PASS` with exact common.recipe retrieval identities and content proof. `dataclasses.replace` preserves provider ID/ModelSpec/loader and the original CPU registry. |
| 4. One resident fleet, eight YAMLs, two episodes/group | **CPU preparation done**, `cache_probe.py:46`; P6 `smoke.sh cosmos --prepare-only` passed. **GPU smoke deferred**, not claimed qualified. Script preserves H100 R2/B1/MPS off; Cosmos retains the baseline legacy driver. Actual current worker RAM admission is blocked until the live horizon fleets close. |

The compatibility certificate binds both complete teacher B500 parents
(spatial3,676 rows; libero_10 8,408 rows;500 episodes each) and their captured
release manifest. It checks every official source file (499), DiT/VAE weights,
stats, prompts and release config exactly. Native loader/key builder/wire
transform/protocol producer files remain exact. Native manifest imports and
normalization/layout/feature construction are compared to hash-verified
historical source AST; native K5 SamplerSpec is exact. The one stage change
is proven to be only the reviewed K2 post-sampling NFE assertion. New metadata,
provider, parity, recovery and CPU probe modules are explicitly enumerated;
unknown/changed native producer files reject. The frozen certificate is itself
a Python literal bundled/content-sealed with the release, not a missing remote
JSON sidecar. No model-code hash is blanket-ignored. P6 still proves the
explicit B→A native-semantic profile bridge independently.

Current recipe capability is honestly `implemented_unverified`: CPU serving
and native producer binding are proved; GPU cache hot-switch/NFE/quality still
need the scheduled smoke. Generic CP1 retrieval, other recipes, warm resume,
CP2, snapshots, B>1 and split execution remain unqualified/rejected. No weights,
normalization, camera processing, action columns or native K5 sampler changed.

Two framework issues discovered during the actual H100 CPU preflight were
fixed narrowly. First, Cosmos had been forced onto GroupCommit despite the
owner requiring its baseline driver (`RUN_CONFIGS.md`: legacy).
`ops/formal_eval.py` selects legacy only for Cosmos cache study;
`formal_driver.py:106` delegates its opt-in cache plan/run;
`cache_study.py:495` chooses `cache_driver_legacy.py:34`, retaining the baseline
ProbeRecordingDriver/CompatibleEpisodeQueue/attempt fence/per-row fsync and
P6's existing ordered bundle planner. DP/Fast retain GroupCommit. Production
config without cache_study keeps its original driver. No queue was edited.

Second, a strict hash of recomputed early `Z@A0` codes failed across local and
H100 BLAS even though every sealed coefficient/main code/row was exact.
H100 NumPy2.2.6: maximum differences4.57763671875e-5 (spatial),
1.220703125e-4 (10); both saved calibration digests are exact.
`search/cache_recipe.py:204` now verifies the saved metric checksum plus
byte-exact fixed rows/main codes/provenance. Only derived float32 early products
may vary: both original and deployed products must satisfy the dot-product
forward-error bound against the identical sealed coefficients, using a
float64 oracle and gamma_(n+1) × sum(abs(products)). Unknown regimes, dtype/
geometry drift and out-of-bound values reject. **The gate uses the original
frozen calibration bytes**, with no refit, threshold/READY changes or edits
to the original R6 modules. The numerical proof is exposed as
`stall_metric_validation`. P6's B500 successful-episode LOEO calibration is
retained; its READY states there is no independent B-validation recording.

CPU verification: **96 tests passed,0 failed**,124.38s. Covers all16 suite/
group combinations,80 serving decisions, shared-resident cloning/fresh reset,
native physical action bytes, zero ordinary fallback, corrector toggles,
confirmed/ambiguous stall and cooldown, changed asset/producer rejection,
metadata portability, legacy fsync/attempt fencing and P6's existing DP/Fast
batched cache regressions. VAE/denoiser forwards are explicitly mocked in
CPU serving tests; genuine recorded VAE keys, real frozen B500 fits, original
Q3 controller and official unnormalize_actions code are used. The 80-decision
fixture records72 HITs/8 injected stall MISSes:80 stage1,8 stage2,40 simulated
head NFEs. It is neither GPU evidence nor a stall-frequency/SR estimate.
Initial test-only tuple/list and official-path fixtures were corrected; the
final full rerun is clean and earlier failed logs remain preserved.

Evidence: `/data/mw_finetune/diagnostics/cosmos_cache_D_20261010_cpu_v2/`:
`CPU_TESTS_FINAL.log/.xml`, `ADMISSION.json`, `PREFLIGHT.json`,
`H100_STALL_METRIC_PROBE.json`, `P6_PREPARE.log` (first metric-hash failure),
`P6_PREPARE_FINAL.log` (PASS). Original B500 parent READY hashes still match
the frozen teacher evidence. No original parent, subset or size500 fit changed.

P6 preparation-only run: **PASS /32 planned episodes /16 YAMLs**,
`/data/mw_finetune/eval/formal_cosmos_libero_smoke2_20261010_165015_143aea67/`.
`assets/CACHE_STUDY.json`, `PREPARED.json`, `PREFLIGHT.json`,
`ORCHESTRATION.json` and `EPISODE_PLAN.json` are sealed. It uses H1002×B1,
MPS off, legacy driver, timan1078 workers,6144MiB RAM/worker, recycle4,
ports23440(proxy)/23441,23442(replicas)/23459(driver). Source files still
match that sealed H100 package. Preparation can be completed while hardware
is busy: `PREFLIGHT.memory_admissible=false` currently because only20,970MiB
of worker headroom was available versus49,152MiB requested. Live hs6fw C16
and hs5dp C16 fleets are accounted for; they were not touched. Launch rechecks
RAM/VRAM and must wait for actual resource release, not simply17:45.

Executed CPU commands:

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  exp/r17_integration/models/cosmos/test_cache.py \
  exp/r17_integration/models/cosmos/test_cpu.py \
  exp/r17_integration/models/cosmos/test_capture.py \
  exp/r17_integration/tests/ops/test_cache_native_portability.py \
  exp/r17_integration/tests/ops/test_cache_study.py

env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m exp.r17_integration.models.cosmos.cache_probe --preflight \
  --output /absolute/new/evidence/directory

env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  bash exp/r17_integration/ops/cache_study/smoke.sh cosmos --prepare-only
```

Coordinator GPU smoke, **not executed by D** (estimate10–15 minutes after
resource admission, including cold model load; no measured cache-throughput
claim; script cleanup deadline is120 minutes):

```bash
bash exp/r17_integration/ops/cache_study/smoke.sh cosmos --run
```

Avoid repeated preparation by launching the already sealed smoke instead:

```bash
exp/r17_integration/ops/formal_eval.sh \
  --prepared-run /data/mw_finetune/eval/formal_cosmos_libero_smoke2_20261010_165015_143aea67 --run
```

Both launch32 qualification episodes (2/group/suite), with all8 logical labels
switched on the same resident fleet; full study later runs only4 physical
Cosmos groups ×2 suites ×500=4000 independent A cells and reports the four
alias pairs. GPU acceptance must audit actual key/HIT/MISS/NFE/execution ACKs,
constant resident generations and activity flags; a smoke with no stall call
is not proof of exercised gate sampling. No formal study was started here.
