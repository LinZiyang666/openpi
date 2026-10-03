# R8 S1 — server observer handback

## Delivered

`osdebug.v1` shared schema and a passive server observer are implemented. The observer copies wire inputs before transforms, reads serving state before `after_infer` clears it, and publishes detached arrays asynchronously. Debug-disabled requests retain the existing serving, cleanup and logging path; reserved `__debug__` and `__oracle__` fields are removed before transforms in either mode.

The shared diagnostic protocol is `method.debug_record() -> dict`, called after the response is fixed. It must only read the just-served decision. Numeric arrays are stored as `diag_*` block fields, with JSON references. Existing A/BlindAWM, CU/CT, SF, SW and SeededInference have read-only adapters. Existing extras and method diagnostic dictionaries are uncapped. A thread-local return-frame tap copies deployed PCA keys and mixture weights from the existing AWM arithmetic; it never reruns retrieval. Connection-owned stage taps count/timestamp the actual existing π0.5 and GR00T dispatch functions once each.

Captures include exact uint8 images, float64 wire state, deduplicated prompt, normalized state, cache/policy/served chunks, output-transformed wire chunk, retrieval/provenance, anchor/offset/age, randomization, diagnostics, dispatch counts and timings. Full raw keys use the schema's SHA256 1/16 sample. Missing live values have explicit statuses and NaN array placeholders. The optional privileged oracle reaches `OnlineQueryView.oracle` / `BlindQueryView.oracle` only with `--os-oracle`; an unauthorized oracle is dropped and recorded as an error.

The writer uses a bounded byte queue (512 MiB default), blocking admission, at most 16 decisions per block, lossless compressed NPZ without objects/pickle, `.part` + fsync + replace, JSONL flushed after its blocks, periodic statistics and SIGTERM/exit drain. Shape changes split blocks. A single oversize item can exceed the queue limit only while the queue is empty; it is counted explicitly. Serialization/storage failures produce telemetry errors, error inventory and an incomplete drain, while preserving the action. Unavailable startup capture similarly returns error echoes and logs each request ID.

Startup metadata contains the model adapter, sampling configuration, source-library SHA, deployed payload/content fingerprints, fit/calibration/stall/stage artifacts, observer/method source hashes, git head and tracked dirty-diff hash. The git diff explicitly excludes `tests/review_tests`. Image shape and price defaults live in the model adapter; custom shapes, dimensions, control interval and prices can be supplied through configuration. No task-name logic is used in collection.

## Files

New owned files:

- `exp/offline_search/debug/__init__.py`, `schema.py`
- `exp/offline_search/debug/server/{__init__,observer,writer,diagnostics,dispatch,manifest,replay_tape}.py`
- `exp/offline_search/debug/tests/test_server_{blocks,observer,parity}.py`
- this handback

Minimal permitted hooks: `exp/offline_search/closed_loop/plugin.py` and `blind.py`. Neither serving entrypoint nor `stage_overrides.py` was edited. Existing R7 request-camera hooks are retained. No other coder's files were changed.

## Enable / configure

Add to either existing server command:

```bash
--os-debug-dir <RUN>/runs/<arm>/debug/server_<port> \
--os-debug-config '{"campaign":"r08_main","writer_queue_bytes":536870912,"rawkeys_rate":0.0625,"snapshot_rate":0.0625,"draws_rate":0.03125}'
```

Only `--os-debug-dir` enables capture. Sampling rates are fixed by v1; incompatible explicit rates are rejected. `queue_bytes` is accepted as an alias for `writer_queue_bytes`. Optional adapter configuration is under `model_manifest`; prices are under `cost_weights`. Set `campaign` explicitly outside the normal run layout. Use `--os-oracle` only for the labelled diagnostic arms. Successful requests echo `v`, `decision_id`, `server_tag`, `server_seq`, `status`; an error echo must fail capture admission.

Per-process output: `meta.json`, `decisions.jsonl`, `blocks/d_<pid>_<block>.npz`, sampled `rawkeys/k_<pid>_<block>.npz`, `writer_stats.json`. Supply one directory per server process. After shutdown require `drained=true`, no `errors`, and `accepted == written == expected requests`.

## CPU validation

All Python runs used CPUs `14-17,58-61`, OMP/OpenBLAS/MKL threads 1, `CUDA_VISIBLE_DEVICES=''`, no bytecode writes and scratch under `/tmp/r8_S1`. No GPU, remote model/server/worker/simulator, tmux, prohibited port, git mutation, process-wide kill or review-test access was used.

Re-run the final server/legacy dispatch suite:

```bash
taskset -c 14-17,58-61 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src TMPDIR=/tmp/r8_S1 \
  .venv/bin/python -m pytest -q -s \
  exp/offline_search/debug/tests/test_server_blocks.py \
  exp/offline_search/debug/tests/test_server_observer.py \
  exp/offline_search/debug/tests/test_server_parity.py \
  exp/offline_search/rounds/r07/c2_wrist/test_dispatch.py \
  --basetemp=/tmp/r8_S1/pytest_coordinator_server
```

Final complete suite: **37 passed, one pre-existing Torch/pynvml deprecation warning, 257.49 seconds**. Reports and data are under `/tmp/r8_S1/pytest_final_complete`. The earlier complete suite passed 33 tests in 255.64 seconds; subsequent contract/dispatch runs passed 23 and then 25 tests after additional fail-open/tape coverage. The final suite includes those additions plus a provenance regression.

Parity uses actual frozen R7 fitted methods and recorded query keys/state through the real `_ConnPolicy`, session, blind/follow/camera and policy-tail paths with CPU fake encoder/policy. Each of π0.5 A/CU/CT/SF/SW/pure-policy and GR00T A/CU/CT/SF/pure-policy replays 220 requests with debug off and on: 2,420 requests per mode. Assertions compare response bytes (excluding only the debug echo), method/history digests, Python/NumPy/Torch RNG state, fake encoder/policy consumption and legacy decision records. Clocks are frozen for response/log timing byte equality; an independently retained real clock measures outer request latency. Tests require the local frozen store/fits; portable contract tests run without them, artifact parity explicitly skips if absent.

Contract tests cover identity/sampling, little-endian NPZ round-trip, integer precision, variable diagnostic shapes, exact wire copies before mutation/cleanup, uncapped extras, oracle stripping/access, diagnostic/startup/writer/serving failures, exact dispatch/RNG preservation, bounded backpressure, lossless tape record/replay/compare, and SIGTERM drain of 31 requests in an owned subprocess. Normal prompts/IDs use U512/U80. Malformed over-width values expand the Unicode field rather than silently truncating; invalid identity is still an error record.

Additional existing checks:

- `closed_loop.selftest`, π0.5 `ProbeB0`, five episodes: **PASS**, 105 decisions, native/recorded/offline equality 100%, no mismatch. Report: `/tmp/r8_S1/selftest_pi05/selftest_report.json`.
- `closed_loop.selftest`, GR00T `ProbeBlind`, `--blind --policy-tail --judge guard_only`: **PASS**, 48 decisions, four episodes, two connections, 32 vision/16 blind/eight MISS, four duplicate rejections and preflight/output/partial-look fallback checks. This probe produced zero policy-tail decisions; the real CU/CT parity tests exercised policy tails. Reports: `/tmp/r8_S1/selftest_groot_blind_tail/{selftest_report,verify_blind}.json`.
- Existing R4 randomized overlay: **PASS**, 500 assignment pairs and complements, three cross-process checks, six invalid-option rejections, four connections/12 episodes/104 decisions and truthful-history checks. Its module was run with an in-memory CPU/output-prefix override to avoid its unowned output and default affinity; no test file was edited. Report: `/tmp/r8_S1/overlay/results/overlay_installed.json`.
- All 11 frozen fit/model combinations successfully constructed/drained startup metadata; artifacts are under `/tmp/r8_S1/startup_meta`.
- Shared `schema.py` and package `__init__.py` pass `ast.parse(feature_version=(3,8))`. Python 3.8 itself was unavailable; the shared module avoids newer syntax/APIs.
- Actual image-bearing tape recording smoke: two requests each from π0.5 and GR00T stores, under `/tmp/r8_S1/tape_record_smoke`. Portable fake-client tests verify exact image/state/prompt round-trip and lifecycle without sockets.

## CPU overhead

Production wire shapes (π0.5 224×224×3, GR00T 256×256×3), two cameras/request; median on latency minus median off latency, milliseconds. The reproducible first production-shape run is `/tmp/r8_S1/pytest_complete/*/parity_report.json`:

| Family | π0.5 off / on / added ms | GR00T off / on / added ms |
|---|---:|---:|
| A | 2.265 / 5.261 / 2.995 | 2.540 / 4.425 / 1.885 |
| CU | 4.474 / 7.098 / 2.624 | 4.426 / 7.026 / 2.600 |
| CT | 4.420 / 7.615 / 3.195 | 4.574 / 7.670 / 3.097 |
| SF | 1.761 / 4.461 / 2.700 | 1.757 / 5.201 / 3.444 |
| SW | 3.252 / 6.528 / 3.276 | unavailable: no deployed wrist path |
| Policy | 3.416 / 5.510 / 2.094 | 3.518 / 5.920 / 2.402 |

These are CPU fake-model request latencies, including capture/queue admission and the normal legacy logger, with uniform synthetic images. They exclude startup artifact hashing and post-replay drain, and do not estimate real-image compression throughput, GPU synchronization or multi-client load. Final rerun reports are under `/tmp/r8_S1/pytest_final_complete`.

## Coordinator GPU validation

The commands in the following section are for the coordinator; S1 did not run them. They use local foreground real servers, ports 23310/23311, a recorded image-bearing tape, identical startup seeds and one sequential connection. Reserve the listed CPUs/GPU first. Do not use the tmux-based stock start script. Run each family separately with a fresh server for off and on; repeat all six π0.5 families and all five GR00T families.

First record both tapes, without a server or simulator:

```bash
cd /home/weiland/projects/openpi
for model in pi05 groot; do
  taskset -c 14-17,58-61 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src TMPDIR=/tmp/r8_S1 \
    .venv/bin/python -m exp.offline_search.debug.server.replay_tape record \
    --store-root /home/weiland/trace_runs/offline_search_store \
    --cell "${model}_spatial_cache" --episodes 20 --out "/tmp/r8_S1/gpu_parity/tape_${model}"
done
```

In the server terminal, set `MODEL`, `FAMILY`, `PHASE` and the reserved GPU. This prepares exact constructor kwargs from the trusted frozen artifact; it does not refit or overwrite the existing fit. For `Policy`, there is no frozen fit argument, and the existing SeededInference fit runs locally as usual. Preparation uses no GPU. The same preparation was checked with CPU-only `PluginRuntime` startup for all 11 combinations.

```bash
cd /home/weiland/projects/openpi
MODEL=pi05                         # pi05 or groot
FAMILY=CU                          # A CU CT SF SW Policy; SW is pi05 only
PHASE=off                          # repeat as on after stopping the off server
GPU=0                              # coordinator's reserved GPU
CPUS=14-17,58-61
R=/home/weiland/projects/openpi
D=/tmp/r8_S1/gpu_parity
RUN=/home/weiland/trace_runs/os_closed_loop/r07_main
YAML="$RUN/config/r7_${MODEL}_spatial_50_SF1.yaml"
CASE="$D/${MODEL}_${FAMILY}"
mkdir -p "$CASE/$PHASE"

mapfile -d '' -t FIT_VALUES < <(
  taskset -c "$CPUS" env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src TMPDIR=/tmp/r8_S1 \
    .venv/bin/python - "$MODEL" "$FAMILY" <<'PY'
import json, sys
from exp.offline_search.debug.tests.test_server_parity import fitted, FITS
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import sources
model, family = sys.argv[1:]
if model == 'groot' and family == 'SW':
    raise ValueError('GR00T has no deployed wrist path')
_, blob = fitted(family, model)
if family == 'A':
    fit = sources()[model + '_spatial_50']['source_artifact']
elif family == 'Policy':
    fit = ''
elif family == 'SW':
    fit = str(FITS / 'r7_sw_pi05_sp_50.pkl')
else:
    fit = str(FITS / ('r7_' + model + '_spatial_50_' + family + ('1' if family == 'SF' else '30') + '.pkl'))
for value in (blob['spec'], json.dumps(blob['kwargs']), fit):
    sys.stdout.buffer.write(value.encode() + b'\0')
PY
)
test "${#FIT_VALUES[@]}" -eq 3 || exit 2
COMMON=(--os-method "${FIT_VALUES[0]}" --os-kwargs "${FIT_VALUES[1]}"
  --os-fit-artifact "${FIT_VALUES[2]}" --os-cell "${MODEL}_spatial_cache"
  --os-root /home/weiland/trace_runs/offline_search_store --os-seed 97
  --os-log-dir "$CASE/$PHASE/legacy" --os-tag parity
  --os-no-shadow-native --os-tokens off --os-log-inputs)
if [ "$FAMILY" = Policy ]; then
  COMMON+=(--os-log-r4 --os-judge periodic:1)
else
  COMMON+=(--os-blind)
fi
case "$FAMILY" in
  CU|CT) COMMON+=(--os-policy-tail --os-policy-tail-blocks 1 --os-judge guard_only) ;;
  SW) COMMON+=(--os-request-cameras) ;;
esac
if [ "$PHASE" = on ]; then
  COMMON+=(--os-debug-dir "$CASE/on/debug"
    --os-debug-config '{"campaign":"parity","writer_queue_bytes":536870912}')
fi
SEED_MAIN='import random, runpy, sys; import numpy as np; import torch; random.seed(97); np.random.seed(97); torch.manual_seed(97); entry=sys.argv[1]; sys.argv=sys.argv[1:]; runpy.run_path(entry, run_name="__main__")'

if [ "$MODEL" = pi05 ]; then
  taskset -c "$CPUS" env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$R:$R/src:$R/packages/openpi-client/src" BATCHING_MAX_BATCH_SIZE=1 \
    .venv/bin/python -c "$SEED_MAIN" exp/offline_search/closed_loop/serve_pi05.py \
    "${COMMON[@]}" --port 23310 --replicas 1 --cache-config "$YAML" \
    policy:checkpoint --policy.config pi05_libero \
    --policy.dir /home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch
else
  G=/home/weiland/projects/openpi_ext/third_party/gr00t_n15
  taskset -c "$CPUS" env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 OPENPI_MONITOR_LEVEL=BASIC BATCHING_MAX_BATCH_SIZE=1 \
    PYTHONPATH="$G:$G/examples/Libero:$R:$R/src:$R/packages/openpi-client/src" \
    /home/weiland/projects/openpi_ext/envs/gr00t_n15_venv/.venv/bin/python \
    -c "$SEED_MAIN" exp/offline_search/closed_loop/serve_groot.py "${COMMON[@]}" \
    --checkpoint /data/ckpt/n15_libero_spatial --port 23311 --denoising-steps 8 \
    --concurrent --allow-dynamic-bundles --cache-config "$YAML"
fi
```

After startup, run in the client terminal with matching model/family/phase. Stop that foreground server cleanly after replay, then restart with `PHASE=on` and replay again. Use fresh directories if repeating a completed run.

```bash
cd /home/weiland/projects/openpi
MODEL=pi05; FAMILY=CU; PHASE=off
D=/tmp/r8_S1/gpu_parity
CASE="$D/${MODEL}_${FAMILY}"
PORT=23310
if [ "$MODEL" = groot ]; then PORT=23311; fi
ECHO=()
if [ "$PHASE" = on ]; then ECHO=(--expect-echo); fi
taskset -c 14-17,58-61 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=.:src:packages/openpi-client/src TMPDIR=/tmp/r8_S1 \
  .venv/bin/python -m exp.offline_search.debug.server.replay_tape replay \
  --host 127.0.0.1 --port "$PORT" --bundle default --tape "$D/tape_${MODEL}" \
  --out "$CASE/$PHASE/responses" "${ECHO[@]}"

# After both replays:
taskset -c 14-17,58-61 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src TMPDIR=/tmp/r8_S1 \
  .venv/bin/python -m exp.offline_search.debug.server.replay_tape compare \
  --off "$CASE/off/responses" --on "$CASE/on/responses"
```

Require `PASS=true` and identical action bytes. For normalized execution/retrieval parity, compare identically named legacy `inputs/*.npz` files from off/on with `allow_pickle=False`: exact dtype/shape/bytes for `a_exec`, `synth`, `wire_actions`, `key_v0`, `key_v1`, `rs`, `raw_state`, `topk`, `scores`, `conf`, `has_vision`, `blind_rows`, `blind_weights`, and the assignment/provenance columns present for that family. Exclude `meta` and clock fields. Require clean debug writer drain and reconcile every tape ID to a server row. Repeat with a controlled production concurrent schedule before campaign admission; stochastic policy sampling depends on request order even with identical seeds.

## Known limits / integration

- Real GPU/model parity, GPU timings, real-image writer capacity and production multi-client attribution remain coordinator validation. CPU tap tests exercise both stage APIs, but do not load either GPU model.
- The replay comparator proves exact returned action dtype/shape/bytes and identities. It deliberately does not compare real elapsed timing fields. Compare normalized action chunks and semantic legacy decision fields from `--os-log-inputs` as an additional admission check; timestamps/PIDs/startup argv naturally differ between processes.
- `k_eff` remains unsupported when the live method supplies no effective-demo statistic; `member_k_eff` preserves its distinct `w_eff`. Unsupported mass and absent compact keys are explicitly unavailable; deferred jobs/catalog analysis should supply these. No inline shadow policy, extra encoder, retrieval recomputation, or RNG draw was added.
- Stage times are host dispatch durations including coordinator waits, with no extra CUDA synchronization. Dispatch counts are per-request logical invocations; shared batching can combine kernels. Fallback CPU hooks carry unsupported stage23 status.
- Legacy GR00T library metadata lacks an image shape; its adapter declares the established 256×256 wire shape and captures actual per-request shapes. The server has no independent environment control interval; its manifest marks that unavailable unless configured, and client metadata supplies it.
- SIGKILL/power failure cannot drain; recoverability comes from atomically published blocks and the persisted error/stat inventory. An unusable output filesystem cannot persist captures and causes explicit error echoes/logging.
- S2 owns arm-level manifests/client joins and S3 owns validation/deferred augmentation. Source-library SHA plus content fingerprints are exposed in server metadata; no change to their files was needed.

## Fix round 1

Read `rounds/r08/REVIEW_1.md` in full and implemented S1's M3/M4/m1/m2/m7/m8 fixes. This section supersedes the earlier descriptions of single-file output names, startup fail-open behavior and thread-wide profiling. The coordinator's float64 `served_wire` change is retained, with a regression using values that fail a float32 round trip. No client, receiver, pool or emitter changes were overwritten.

Changes:

- **M3:** server output is now `decisions_<pid>.jsonl`, `meta_<pid>.json`, `writer_stats_<pid>.json`, with unchanged PID-qualified block/raw-key names. Episode finalization synchronously flushes/fsyncs all pending admitted records before the lifecycle acknowledgement, including implicit episode boundaries. A writer error prevents that acknowledgement rather than accepting an unrecorded episode. An owned subprocess regression acknowledges a 13-request episode, admits two requests of the next episode, is killed with SIGKILL, and verifies all 13 acknowledged records survive. A restart writes separate files and preserves the killed process's metadata, statistics and truncated log byte-for-byte.
- **M3 reader compatibility:** shared `schema.read_jsonl(path, issues=None)` skips/reports an unterminated final line, including otherwise valid JSON and partial UTF-8. It rejects corruption in a newline-terminated record. S1 test consumers glob `decisions*.jsonl`, `meta*.json`, `writer_stats*.json` and accept legacy names. S2/S3 own their production consumer updates; the new helper is available to them. `SCHEMA.md` §2/§3 now defines this layout and boundary behavior.
- **M4:** a blind cache/cache-tail row gets `src="follow"` only if `os_sf_source` is present or `os_sf_extension == 1`. Ordinary SF diagnostic keys, extension 0, and unrelated extension values remain `cache_tail`. Ten regression cases cover both legacy source spellings and these distinctions.
- **m1:** server `init` is parsed from the final component of `task_uid`; `orig_init_state_idx` separately preserves episode metadata's original pool index. The legacy plugin/method episode identity stays unchanged. A regression deliberately uses subset init 20 versus original index 77.
- **m2:** failure to construct the observer with `--os-debug-dir` is fatal before model loading/serving. Debug-disabled startup never constructs an observer. Per-request copy/serialization failures still affect telemetry only.
- **m7:** new `server/digests.py` stores a shared locked/atomic `<RUN>/catalog/digest_cache.json`, reused across arms and ports. Every entry validates `(realpath, size, mtime_ns)` before reuse. File hashes and deployed mapped-array byte-range hashes are separate, so .npy headers do not alter payload fingerprints. Changed size/mtime/path or malformed cache entries recompute. Read-only mapped views reuse content hashes; mutable, private copy-on-write, or unmapped arrays are hashed directly to avoid stale payload fingerprints. `meta_<pid>.json` reports cache hits/misses/bytes hashed/path. Tests cover symlink reuse, each invalidation field, file versus payload hashes, distinct mapped slices, mutated copies, corrupted cache, concurrent locked updates, and two arm/port startups with zero repeated library/source byte hashing. For nonstandard layouts without a `runs` ancestor, the cache lives under `<debug-dir-parent>/catalog/`.
- **m8:** replaced `sys.setprofile` with temporary bound `_dist`/`_mix` wrappers solely on the connection's method chain. Each original operation executes once; only its existing return values are copied. Bound `__func__` is preserved so CU/CT's copied metric facade keeps its substituted task matrix; SW's temporary metric fields are captured in place. The previous instance attributes are restored even on failure. No class/global wrapper or profiler is installed, and no callback executes for unrelated model Python/C calls. Regressions forbid `sys.setprofile`, exercise the facade contract and existing overrides, check connection isolation/restoration and failure behavior, and verify no extra calls.

Edited/new owned files: `server/{diagnostics,digests,manifest,observer,writer}.py`, `schema.py`, `tests/test_server_{blocks,observer,parity,support,fix_round1,digests}.py`, this handback. Permitted shared edits: `closed_loop/plugin.py`, `debug/SCHEMA.md`. No other coder's implementation was changed.

Validation: **61 passed, one existing Torch/pynvml deprecation warning, 254.28 seconds** in the final full suite. This includes all 11 model/family off/on tapes (220 requests per mode each): identical response bytes, method/history digests, Python/NumPy/Torch RNG state and legacy decision records. Earlier contract/review/legacy wrist checks passed **50 tests** in 10.37 seconds. Tests mock git provenance; no git command was executed. All runs use S1 CPUs `14-17,58-61`, single-thread numeric libraries, an empty CUDA device set, and scratch under `/tmp/r8_S1`. No model/server/socket/remote/simulator was launched. SIGTERM/SIGKILL tests signal only their own CPU subprocesses. Python **3.8.20** (`openpi_ext/envs/libero_sim/bin/python`) also passed shared-schema import, atomic float64 NPZ round-trip and truncated-final-line handling; artifacts are under `/tmp/r8_S1/python38_fix1`.

Exact final command:

```bash
taskset -c 14-17,58-61 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src TMPDIR=/tmp/r8_S1 \
  .venv/bin/python -m pytest -q -s \
  exp/offline_search/debug/tests/test_server_blocks.py \
  exp/offline_search/debug/tests/test_server_observer.py \
  exp/offline_search/debug/tests/test_server_fix_round1.py \
  exp/offline_search/debug/tests/test_server_digests.py \
  exp/offline_search/debug/tests/test_server_parity.py \
  exp/offline_search/rounds/r07/c2_wrist/test_dispatch.py \
  --basetemp=/tmp/r8_S1/pytest_fix1_complete
```

Updated CPU overhead (same production-sized uniform wire-image fixture, fake models; median on minus median off, milliseconds). Per-family reports: `/tmp/r8_S1/pytest_fix1_complete/*/parity_report.json`.

| Family | π0.5 off / on / added ms | GR00T off / on / added ms |
|---|---:|---:|
| A | 2.381 / 3.678 / 1.296 | 2.293 / 3.843 / 1.551 |
| CU | 4.273 / 5.859 / 1.586 | 4.207 / 5.842 / 1.634 |
| CT | 4.576 / 6.307 / 1.730 | 4.167 / 5.969 / 1.802 |
| SF | 1.728 / 3.424 / 1.696 | 1.743 / 3.666 / 1.923 |
| SW | 3.477 / 5.031 / 1.554 | unavailable: no deployed wrist path |
| Policy | 3.114 / 4.501 / 1.387 | 3.207 / 4.687 / 1.480 |

These measure request capture/admission and the legacy logger, exclude startup and episode-end drain, and do not measure real-image compression or GPU latency. The profiler replacement has passed byte/RNG parity; improvement to real GR00T GPU latency must still be measured by the coordinator.

Read-only inspection of the current plain `r8_pi05_l10_50_A_smoke` capture found 1,380 server rows, 20 client episode files and no truncated server lines. A sampled 10×7 wire chunk is float64 and loses bits through float32, validating the coordinator's precision fix. The earlier `__capfail*` directories were not used for this check. Existing smoke files retain their legacy names and are supported by the glob/JSONL reader rules.

Coordinator GPU follow-up: rerun the earlier seeded off/on commands for real GR00T A, CU, CT and policy, with unchanged tape, checkpoint, seed, denoising steps, batch size 1 and sequential connection. The coordinator measured the old profiler at looks 63→87 ms and policy 263→369 ms (π0.5 unaffected); S1 cannot measure GPU latency here. Report live look/MISS p50/p95, wall throughput, exact action-byte and normalized-chunk parity, writer wait/serialization, and the PID-qualified writer statistics after a clean episode drain. Repeat π0.5 SW to check wrist compact keys and completions. Inspect `meta*.json` / `writer_stats*.json`, rather than the superseded single filenames in earlier prose. Before production admission, exercise restart during an unaccepted episode and verify prior accepted episodes join across both process logs.
