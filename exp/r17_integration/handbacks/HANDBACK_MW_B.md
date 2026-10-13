# MetaWorld batching and cache handback (package B)

Implemented DP/Fast-WAM MT50 batched residents, optional private MPS, native reduced
steps, C10 tail reuse, teacher capture, and MT50 cache-study YAML preparation and
hot switching. CPU work only: no GPU server/campaign was launched and no GPU memory
was allocated. No commit was made. Package A's transfer/library/seed edits and
package D's Cosmos edits were preserved.

C10 follows the coordinator correction: execute five actions per transport call;
the second call serves the committed tail with **stage_calls=(0,0,0), NFE=0**.
The MT50 kernel and execution observer were not changed by B.

## Files changed and removed checks

- `models/dp/{batch_runtime,staged,formal_batch_server,library}.py`: admit MT50 in
  the existing vector/queue paths, choose an MT50 graph probe, prewarm B1..Bmax,
  attach A's `campaign_capture_factory` (local `LibraryCaptureFactory` or H100
  shards) and bind each observation before inference,
  close capture writers, and use the common MT50 Store recipe path.
- `models/fastwam/{batch_runtime,staged,formal_batch_server,formal_batching,provider}.py`:
  admit `fastwam_mt50`, use one-view native batches, enable the same capture
  lifecycle, retain MT50 prompt text if text caching is selected, and use the
  common MT50 Store recipe path.
- `ops/formal_eval.py`: B>1 admission for DP/Fast-WAM MT50; native steps for all
  four requested MW models; native commit bounds/divisibility; MT50 cache-study
  admission; load a GR00T resident with the requested immutable K. B changed
  these regions only; A owns the worker/library/transfer changes in this file.
- `ops/cache_study.py`, `ops/cache_knob.py`, `search/cache_recipe.py`: MT50 bundles,
  paths, profiles, seed8 Store provenance, cohorts, C-specific fits, staging, YAML
  publication and plans;
  preserve LIBERO generation/knob defaults. `--groups c ch` makes two bundles.
- `ops/RUN_CONFIGS.md`: correct the obsolete MW B1-only launch advice.
- `tests/ops/test_metaworld_batch_cache.py`: 32 new CPU regressions; update the
  existing DP/Fast-WAM batch scope tests, horizon/step rejection cases, and
  LIBERO-specific cache-test parameter lists in `tests/ops/test_cache_{study,
  sizes,lane,knob,reduced}.py` to use `LIBERO_MODELS`.

Removed the LIBERO-only batch/domain checks and the batch resident's prohibition
on `library=True`. Capture metadata is enabled when requested. No new seals,
fingerprints, receipts, admission gates, or validation layers were introduced.
Existing sampler/shape/RNG/capture behavior and LIBERO inference math remain intact.

## Provider limits and prerequisites

- DP: MW K5/K10 trailing eta0 DDIM and native DDPM100; C5/C10/C15; batch caps2..32.
- Fast-WAM: MW K1/K2/K10; C5..C30 in multiples of5; batch caps2..32.
- π0.5: its provider already has K1/K2/K10 and the shared Store recipe path. The
  actual MT50 checkpoint is H5/action32, so **C10 cannot be implemented without
  changing checkpoint geometry**; cache groups stay at C5. Native batched serving
  is still absent for this selector (formal B1).
- GR00T: its provider already supports the shared Store recipe path and C10,
  with H16/action32 storage and four active actions. K1/K2/K4 are **immutable
  resident choices**; reduced-step preparation now selects the requested K before
  loading. A resident cannot hot-switch K through YAML. Cache YAMLs hot-switch
  horizon/corrector/gate at the resident's K. Formal batching remains B1.

MT50 libraries come from A at `/data/r17_library/<model>_mt50/bpool/`, seed8,
with existing Store/row-index views. Defaults use B500 (10 episodes/task);
MW sizes500/1000/1500/2000/2500 are supported. Test plans use seed7 and idx0..49
for all50 tasks (2500 episodes/bundle after A's cohort correction). Smoke plans
use two tasks × idx0,1 per bundle. The real libraries are not present yet, so
real campaign preparation/collection is a coordinator prerequisite.

A has added `--max-batch`, `--batch-wait-ms`, `--mps`, and `--no-capture-images`
to the library launcher; the earlier handback integration request is resolved.
All transfers use A's direct rsync helper with `--bwlimit`; tether is exec-only.
B did not change the transfer paths.

## Test results and numerical differences

- New MT50 CPU suite: **32 passed**. Real batch queues and fake native models;
  DP graph bootstrap/prewarming is mocked on CPU. Capture off, capture on with
  no images, and capture on with one camera all serve concurrent requests.
  C10 policy/library tails execute no stages or NFE. Real WebSocket controls
  register two YAMLs, select them per episode, and retain one resident lifetime.
- Existing pre-edit sample: **160 passed, 34 failed**, plus one passing subtest.
  Many existing assertions still require owner-removed fingerprint/seal/gate
  checks. Logs: `/tmp/r17_mw_b_before.log`.
- Extended regressions (34 explicit files): **556 passed, 87 failed**, one passing
  subtest (`/tmp/r17_mw_b_regressions.log`). **86 failures reproduce without B's
  native/cache changes**:34 in the original pre-edit sample,52 in an isolated
  comparison copy retaining concurrent A/D edits. They are old owner-removed
  gates/hashes, historical fixtures, or A's undecided dev partition. The87th was
  an obsolete C10 rejection collected before its test update; the corrected
  `dp_mw-11` case now passes (`/tmp/r17_mw_b_cli_final.log`). No new functional
  MW failures remain. Comparison logs: `/tmp/r17_mw_b_extra_baseline.log` and
  `/tmp/r17_mw_b_extra_baseline_yaml.log` (the latter repeats one case after
  supplying its untracked generated YAML inputs).
- Native provider follow-up: **52 passed, 4 failed** across the32 new tests,
  GR00T library/P6 tests, and pi05 native/e2e tests. All GR00T library/P6 cases
  pass. The4 pi05 assertions cover removed identity/resume guards, comparison
  against HEAD's old episode runner, and an old500-vs-new2500 cohort expectation
  (`/tmp/r17_mw_b_final.log`). They do not indicate a missing recipe provider.
- Final MW-only rerun: **32 passed** (`/tmp/r17_mw_b_final_new.log`). Compile checks
  and `git diff --check` pass. A first directory-wide collection hit an existing
  scratch runtime symlink/import conflict; the broad run uses explicit files.
- Production GPU arithmetic/throughput are unmeasured; no numeric gate was added.

| Fixed-observation comparison | Per-row max physical action difference | Scope |
|---|---|---|
| Actual DP MT50 checkpoint, CPU DDPM100, B3 vs B1 | `1.9073486328125e-6`, `9.5367431640625e-7`, `2.86102294921875e-6` | Three recorded MT50 observations; diagnostic CPU loader, actual weights; information only |
| Small DP CPU fixture, DDPM100, B3 vs B1 | `1.430511474609375e-6`, `2.1457672119140625e-6`, `1.430511474609375e-6` | Native small Transformer fixture; information only |
| Fast-WAM CPU fixture, K10, B3 vs B1 | `0`, `0`, `0` | Native scheduler/stage interface with fixture VAE/heads; **not production GPU parity** |

Actual DP observations: `/data/mw_finetune/qualification/observations/dp_mw_v2/wire_observations.npz`;
checkpoint: `/data/mw_finetune/dp/mt50_policy_100000`; rows0..2, seeds18/37/56. The actual-checkpoint CPU
check took47.44s. Evidence: `/tmp/r17_mw_b_dp_real_cpu.json`,
`/tmp/r17_mw_b_dp_real_cpu.log`, `/tmp/r17_mw_b_new.log`.

## Exact coordinator commands (not executed by B)

Run from `/home/weiland/projects/openpi`. Set R/B/MPS explicitly; these are smoke
parameters, not measured throughput recommendations. R=2/B=4/MPS=1 is the example.
Choose a fresh short run name and free ports within the model's reserved window.
The examples reserve four timan107 workers and leave rendering off the server GPU.

```bash
cd /home/weiland/projects/openpi
R=2
B=4
MPS=1
MPS_ARGS=()
if [ "$MPS" = 1 ]; then MPS_ARGS=(--mps); fi
STAMP=$(date +%m%d_%H%M%S)
DP_RUN=mwb_dp_$STAMP
FW_RUN=mwb_fw_$STAMP

# Four pure-inference episodes: two MT50 tasks × idx0,1; seed7.
prepare_mt50_smoke() {
  local model=$1 run=$2 node port driver memory
  case "$model" in
    dp_mw) node=local; port=23180; driver=23189; memory=$((4 * R));;
    fastwam_mw) node=h100; port=23170; driver=23179; memory=$((36 * R));;
  esac
  bash exp/r17_integration/ops/formal_eval.sh \
    --model "$model" --server-node "$node" --gpu cuda:0 \
    --replicas "$R" --memory-budget-gib "$memory" \
    --max-batch "$B" --batch-wait-ms 5 "${MPS_ARGS[@]}" \
    --worker-node timan107 --extra-workers 0 --workers 4 \
    --worker-memory-mib 2048 --worker-vram-mib 1024 --worker-max-episodes 0 \
    --worker-egl-devices 0,1,2,3,4,5,6,7 --driver-mode group-commit \
    --port "$port" --driver-port "$driver" --smoke \
    --seconds 900 --episode-seconds 300 --run-name "$run" --prepare-only
}
prepare_mt50_smoke dp_mw "$DP_RUN"
bash exp/r17_integration/ops/formal_eval.sh --run \
  --prepared-run "/data/mw_finetune/eval/$DP_RUN"
prepare_mt50_smoke fastwam_mw "$FW_RUN"
bash exp/r17_integration/ops/formal_eval.sh --run \
  --prepared-run "/data/mw_finetune/eval/$FW_RUN"
# For C10 add --commit-horizon 10 to preparation. For reduced steps instead
# add --denoising-steps 5/10 (DP) or 1/2 (Fast-WAM), with a new run name.
```

The same four episodes with native teacher capture use seed8. The smoke writes
its own `smoke_parent`; it does not publish/overwrite the full B library.
`--no-capture-images` stores native keys/state/actions only; use
`--capture-images` to exercise the one-camera image path.

```bash
DP_CAP=mwb_dpc_$STAMP
FW_CAP=mwb_fwc_$STAMP
prepare_mt50_capture() {
  local model=$1 run=$2 node port driver memory
  case "$model" in
    dp_mw) node=local; port=23180; driver=23189; memory=$((4 * R));;
    fastwam_mw) node=h100; port=23170; driver=23179; memory=$((36 * R));;
  esac
  bash exp/r17_integration/ops/library_campaign.sh \
    --model "$model" --suite mt50 --server-node "$node" --gpu cuda:0 \
    --replicas "$R" --memory-budget-gib "$memory" \
    --max-batch "$B" --batch-wait-ms 5 "${MPS_ARGS[@]}" \
    --worker-node timan107 --workers 4 --port "$port" --driver-port "$driver" \
    --no-capture-images --smoke --seconds 900 --episode-seconds 300 \
    --run-name "$run" --prepare-only
}
prepare_mt50_capture dp_mw "$DP_CAP"
bash exp/r17_integration/ops/formal_eval.sh --run \
  --prepared-run "/data/r17_library/dp_mt50/campaigns/$DP_CAP"
bash exp/r17_integration/ops/library_campaign.sh --model dp_mw --suite mt50 \
  --finish "/data/r17_library/dp_mt50/campaigns/$DP_CAP"
prepare_mt50_capture fastwam_mw "$FW_CAP"
bash exp/r17_integration/ops/formal_eval.sh --run \
  --prepared-run "/data/r17_library/fastwam_mt50/campaigns/$FW_CAP"
bash exp/r17_integration/ops/library_campaign.sh --model fastwam_mw --suite mt50 \
  --finish "/data/r17_library/fastwam_mt50/campaigns/$FW_CAP"
```

After A publishes the full seed8 parent, prepare two ungated YAMLs (C5/C10).
Each of the two bundles gets two MT50 tasks × idx0,1: eight test episodes total.
Change MODEL to `fastwam_mw` to run the identical cache smoke on its lane.

```bash
MODEL=dp_mw
CACHE_RUN=mwb_dpcs_$STAMP
STEM=${MODEL%_mw}
STUDY_DIR=/data/r17_library/${STEM}_mt50/studies/$CACHE_RUN
case "$MODEL" in
  dp_mw) NODE=local; PORT=23180; DRIVER=23189; MEMORY=$((4 * R));;
  fastwam_mw) NODE=h100; PORT=23170; DRIVER=23179; MEMORY=$((36 * R));;
esac
CPU=(env CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python)
"${CPU[@]}" -m exp.r17_integration.ops.cache_study generate \
  --model "$MODEL" --output "$STUDY_DIR" --sizes 500 --groups c ch \
  --long-dp 10 --long-fast 10
"${CPU[@]}" -m exp.r17_integration.ops.cache_study fit \
  --model "$MODEL" --study "$STUDY_DIR/STUDY.json"
"${CPU[@]}" -m exp.r17_integration.ops.cache_study smoke \
  --model "$MODEL" --study "$STUDY_DIR/STUDY.json" \
  --server-node "$NODE" --gpu cuda:0 --replicas "$R" --memory-budget-gib "$MEMORY" \
  --max-batch "$B" --batch-wait-ms 5 "${MPS_ARGS[@]}" \
  --worker-node timan107 --extra-workers 0 --workers 4 \
  --worker-memory-mib 2048 --worker-vram-mib 1024 --worker-max-episodes 0 \
  --worker-egl-devices 0,1,2,3,4,5,6,7 --port "$PORT" --driver-port "$DRIVER" \
  --seconds 900 --episode-seconds 300 --run-name "$CACHE_RUN" --prepare-only
bash exp/r17_integration/ops/formal_eval.sh --run \
  --prepared-run "/data/mw_finetune/eval/$CACHE_RUN"
```
