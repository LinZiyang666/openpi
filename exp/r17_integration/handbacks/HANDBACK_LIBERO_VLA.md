# π0.5 / GR00T LIBERO — CPU preparation

Both selectors now use the R17 formal/library/cache path. Existing model packages
are parameterized by benchmark: π's vision-prefix → LLM → Euler sampler and
GR00T's Eagle scatter → LLM → flow head keep their MT50 stage splits. Each real
view contributes 256 tokens pooled 4×4 to 32768 floats; active state is 8-D,
wire action is 7-D, storage is H10×32 / H16×32. Native MISS is π K10 / GR K4;
reduced grids remain π K1..5/K10 and GR K1..4. GR's `live_k` follows the study
YAML at startup. All eight groups use C5/C10; C10 is the historical R16 choice,
with no new horizon NI claim.

Changed: `models/pi05/{adapter,provider,serve}.py`,
`models/groot/{manifest,loader,key_builder,provider,serve,wire_policy}.py`,
`qualification/{assets,model_runtime,deploy_cpu}.py`,
`ops/{formal_eval,formal_driver,library_campaign,cache_study,cache_knob}.py`,
and `ops/RUN_CONFIGS.md`. Runtime packaging includes the LIBERO wire helpers
and π norm asset. GR selects each suite's weights/transforms per connection;
a suite library loads one backend, a baseline/cache resident loads both.
Existing knob cohorts continue to follow their source manifests.

CPU tests cover native/staged sampler parity and NFE, both historical key
builders, active state, gripper conversion, suite-specific GR normalization,
C10 free tails, fixed endpoint K, official/B500 plans, image-free capture and
parent construction, six nested subsets, C5/C10 fits and Q3 artifacts, six-size
eight-group YAMLs and runtime packaging. Test files are
`tests/models/pi05/test_pi05_libero.py`, `tests/models/groot/test_libero.py`,
`tests/ops/test_libero_vla.py`; existing fixtures/tests were updated for the
expanded model list and the owner's already-removed fingerprint gates.
Focused runs passed **170 CPU tests**, plus two historical knob regressions.
All 22 expanded GPU-workflow CLI invocations passed parser checks; no launch
or preparation was performed by that check. π's actual content manifest also
passed on CPU, including the local tokenizer and external LIBERO norm asset.

CPU verification command (all model, cache and library tests listed here):

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  exp/r17_integration/tests/models/pi05/{test_pi05,test_pi05_libero}.py \
  exp/r17_integration/tests/models/groot/{test_integration,test_shared_factory,test_libero}.py \
  exp/r17_integration/tests/ops/{test_library_campaign,test_mt50_library,test_library_finish,test_cache_study,test_libero_vla}.py
```

Paths and provenance:

| Asset | Path | Historical source |
|---|---|---|
| π weights | `/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/model.safetensors` | R16 resident → `exp/offline_search/closed_loop/serve_pi05.py`, `ops/h100/local_server.py` |
| π norms | `/home/weiland/projects/openpi/assets/pi05_libero/physical-intelligence/libero/norm_stats.json` | `pi05_libero` config; `src/openpi/policies/policy_config.py` fallback because converted checkpoint has no assets |
| GR spatial / 10 weights | `/data/ckpt/n15_libero_spatial`, `/data/ckpt/n15_libero_10` | `closed_loop/ops/start_server.sh`, R16 resident suite split |
| GR H100 copies | `/home/exouser/ckpt/n15_libero_spatial`, `/home/exouser/ckpt/n15_libero_10` | `closed_loop/ops/h100/HANDBACK_SB.md`, checkpoint-copy table |
| GR norms | each checkpoint's `experiment_cfg/metadata.json`, `new_embodiment` | native `examples/Libero/custom_data_config.py` and historical `exp/libero_groot/serve_groot_libero.py` |

Local checkpoint configs/norm metadata were read on CPU: π H10/state8/action7,
GR H16/state8/action7. Input is agentview + wrist RGB256, retaining the native
resize/crop transforms. π uses fixed token slots; GR uses the image-token mask,
matching `cp1_spatial_pool_16` / `cp1_groot_libero_spatial_pool_16`.

Run these commands **later**, after the GPU lanes and ports are available. They
were not executed here. Suggested placement follows
[RUN_CONFIGS.md](../ops/RUN_CONFIGS.md): π on 4090 R5/MPS/B1, inherited whole
NUMA-node-1 affinity; GR on H100 R2/MPS/B1 with two suite backends per replica.
Both use timan107, EGL0..7 and recycle after four episodes. The shared
23160..23169 tuple means these commands run serially. Keep the baseline/cache
topology identical and let closed campaigns release their ports before reuse.
Names below are concrete one-use names; if already present,
choose fresh names and update `cache_study.CONTROLS` for the baseline names.
The inherited worker deployment is the existing sealed simulator dependency
release used by `r17_ops/knob/prepare_dev.sh`; RUN_CONFIGS records that timan107's
mutable legacy LIBERO config was removed. Reusing it avoids that preparation
failure and creates each new campaign's own private config.

```bash
cd /home/weiland/projects/openpi
export PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export R17_INHERIT_WORKER_DEPLOYMENT=/data/mw_finetune/eval/cosmossizes145_r8c/WORKER_DEPLOYMENT.json

pi() { taskset -c 22-43,66-87 env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu .venv/bin/python -u "$@"; }
gr() { env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu .venv/bin/python -u "$@"; }

common=(--gpu cuda:0 --max-batch 1 --mps --driver-mode group-commit
  --worker-node timan107 --worker-memory-mib 6144 --worker-vram-mib 1024
  --worker-egl-devices 0,1,2,3,4,5,6,7 --worker-max-episodes 4
  --port 23160 --driver-port 23169)
pt=(--model pi05_libero --server-node local --replicas 5 --replica-startup-width 3
  --memory-budget-gib 47 --workers 24
  --checkpoint /home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch)
gt=(--model groot_libero --server-node h100 --replicas 2 --replica-startup-width 2
  --memory-budget-gib 48 --workers 32 --checkpoint /home/exouser/ckpt)

# 1. Capture: full native teacher at every decision, B500/suite, no images.
# (2026-10-15: library capture refuses --driver-mode group-commit, "grouped driver supports pure inference only";
#  drop it from "${common[@]}" here and use the default driver, as every earlier library build did.)
for pair in libero_spatial:sp libero_10:10; do
  suite=${pair%:*}; short=${pair#*:}
  pi -m exp.r17_integration.ops.library_campaign "${pt[@]}" "${common[@]}" \
    --suite "$suite" --no-capture-images --run-name "pl_$short" --prepare-only
  pi -m exp.r17_integration.ops.formal_eval --run --prepared-run \
    "/data/r17_library/pi05_libero_${suite}/campaigns/pl_$short"
done
for pair in libero_spatial:sp libero_10:10; do
  suite=${pair%:*}; short=${pair#*:}
  gr -m exp.r17_integration.ops.library_campaign "${gt[@]}" "${common[@]}" \
    --suite "$suite" --no-capture-images --run-name "gl_$short" --prepare-only
  gr -m exp.r17_integration.ops.formal_eval --run --prepared-run \
    "/data/r17_library/groot_libero_${suite}/campaigns/gl_$short"
done

# 2. Finish on CPU: bpool/, subsets/size_50..500, fits/size_50..500 (C10).
for pair in libero_spatial:sp libero_10:10; do
  suite=${pair%:*}; short=${pair#*:}
  pi -m exp.r17_integration.ops.library_campaign --model pi05_libero --suite "$suite" \
    --finish "/data/r17_library/pi05_libero_${suite}/campaigns/pl_$short"
  gr -m exp.r17_integration.ops.library_campaign --model groot_libero --suite "$suite" \
    --finish "/data/r17_library/groot_libero_${suite}/campaigns/gl_$short"
done

# 3. Pure baselines: π official-50 all four suites = 2000 episodes;
# GR spatial + 10 only = 1000 episodes (500/suite, original global IDs).
pi -m exp.r17_integration.ops.formal_eval "${pt[@]}" "${common[@]}" \
  --extra-workers 0 --official-50 --run-name pl_a --prepare-only
pi -m exp.r17_integration.ops.formal_eval --run --prepared-run /data/mw_finetune/eval/pl_a
gr -m exp.r17_integration.ops.formal_eval "${gt[@]}" "${common[@]}" \
  --extra-workers 0 --official-50 --suites libero_spatial libero_10 --run-name gl_a --prepare-only
gr -m exp.r17_integration.ops.formal_eval --run --prepared-run /data/mw_finetune/eval/gl_a

# 4. Cache preparation on CPU: native MISS; 96 YAMLs / 48,000 A episodes/model.
pi -m exp.r17_integration.ops.cache_study generate --model pi05_libero \
  --sizes 50 100 200 300 400 500 --output exp/r17_integration/ops/cache_study/libero_pi
gr -m exp.r17_integration.ops.cache_study generate --model groot_libero \
  --sizes 50 100 200 300 400 500 --output exp/r17_integration/ops/cache_study/libero_gr
pi -m exp.r17_integration.ops.cache_study fit --model pi05_libero \
  --study exp/r17_integration/ops/cache_study/libero_pi/STUDY.json
gr -m exp.r17_integration.ops.cache_study fit --model groot_libero \
  --study exp/r17_integration/ops/cache_study/libero_gr/STUDY.json
# fit adds C5 fits and C5/C10 Q3 successful-B LOEO artifacts for every size.
pi -m exp.r17_integration.ops.cache_study launch "${pt[@]}" "${common[@]}" \
  --study exp/r17_integration/ops/cache_study/libero_pi/STUDY.json \
  --extra-workers 0 --run-name pl_cs --seconds 518400 --episode-seconds 1800 --prepare-only
gr -m exp.r17_integration.ops.cache_study launch "${gt[@]}" "${common[@]}" \
  --study exp/r17_integration/ops/cache_study/libero_gr/STUDY.json \
  --extra-workers 0 --run-name gl_cs --seconds 518400 --episode-seconds 1800 --prepare-only
# Separate later GPU launch, using the prepared runtime/settings:
# pi -m exp.r17_integration.ops.formal_eval --run --prepared-run /data/mw_finetune/eval/pl_cs
# gr -m exp.r17_integration.ops.formal_eval --run --prepared-run /data/mw_finetune/eval/gl_cs
```

`--prepare-only` starts no model/server/fleet. Remote preparation uses CPU
sealing and the existing public-IP `direct_transfer.py` rsync transport.
No tether data transfer, credentials, commit, GPU launch or live campaign
directory changes were performed. GPU checkpoint parity, peak VRAM, throughput,
success rates, and current H100 asset/environment availability remain unverified.

## Follow-up: per-task reduced K1 IR knob

`ops/cache_knob.py` accepts both selectors for DEV/TEST, explicit `--source`
studies, wrapped descent/raise tables, and CPU `calibrate`. It generates C10 +
Corrector + Q3 + Periodic configurations with **every model call K1**, preserving
the native K10/K4 teacher libraries. GR startup reads K1 from the study YAML.
`ops/{formal_eval,formal_driver,formal_workers,library_init,cache_study}.py` add
standalone B30 pure inference and its existing audited B-input transport.
`models/groot/manifest.py` shares its native sampler constructor.
`rounds/r17/{cache/analyze_cache.py,knob/common.py,knob/descent.py,knob/test.py}`
recognize the models, K1 references, first-failure descent, and partial A reruns.
The existing scratch scripts and existing campaign directories were left alone.

Reference names are `pl_k1b`/`gl_k1b` (B0–29, **600 episodes/model**) and
`pl_k1a`/`gl_k1a` (A0–49, **1000 episodes/model**, spatial + 10). Pure inference
replans every five controls; cache long commits are C10. DEV follows the existing
R17 batch layout: six periodic levels at every size plus size500 K1 pure, **22,200
episodes/model**. Descent selects the last passing level before the first failure
at five points (one fewer success allowed out of 30); unrun lower levels cannot
prevent that selection. TEST starts with **6,000 episodes/model**, then only the
listed task/size cells: ≥5 fewer successes → pure; 3–4 fewer → one level up.
A task that has reached the pure level stays there (coordinator review: no repeated pure retests).
Analysis merges each task's latest 50 outcomes and reports incomplete cells.

CPU stub tests in `tests/ops/test_libero_vla_knob.py` cover both reference plans,
B archive/worker routing, all six sizes and seven levels, partial tables,
native K1 calls under Q3/periodic/pure triggers, CLI topology, first-failure
selection and successive A reruns. Two tests also build real synthetic image-free
B parents, fit C10, calibrate, stage/seal K1 and query LOEO versus A retrieval.
Historical knob, GR sampler and formal/library regressions were also exercised.
**160 targeted CPU cases passed**, including all 22 new VLA-knob cases. The legacy
port assertion was updated to the launcher's existing 234xx block. All four shell
blocks passed `bash -n`; 38 expanded follow-up calls passed CPU parser/stub checks,
including two TEST rounds per model. No GPU/server/fleet/campaign was started.
The checkpoints/norms are the same paths and historical sources listed above.
Calibration uses the existing B-library LOEO **model-call proxy** (vision=0,
miss=1); GPU stage costs, measured IR, VRAM and success rates need later runs.

Exact later order: finish the capture/fit steps above, then K1 B/A references →
CPU calibration/DEV preparation → DEV GPU run → descent → TEST GPU run/analysis →
partial reruns. Reuse the `pi`, `gr`, `pt`, `gt`, `common` and environment setup
above: π 4090 R5/MPS/B1 pinned to NUMA1; GR H100 R2/MPS/B1, timan107 EGL0..7/N4.
The function below runs the models serially on their shared port tuple. Use fresh
names/output directories if these one-use names already exist; update reference
arguments with any renamed baseline. The final `test.json` reports completeness
and `within_tolerance` for every task; an incomplete campaign has pending results.
These commands were documented and parser-checked only, **not executed**.

```bash
# Continue in the repository with the shell setup above.
kroot=exp/offline_search/rounds/r17/knob/vla_k1
vla_knob() {
  local model=$1 runner=$2 short=$3 source=$4
  local -a topology partial lanes
  if [[ $model == pi05_libero ]]; then topology=("${pt[@]}"); else topology=("${gt[@]}"); fi

  # 1. K1 pure reference on precisely DEV B0–29, then official TEST A0–49.
  "$runner" -m exp.r17_integration.ops.formal_eval "${topology[@]}" "${common[@]}" \
    --extra-workers 0 --official-50 --suites libero_spatial libero_10 \
    --denoising-steps 1 --eval-pool B --init-count 30 --seconds 172800 \
    --run-name "${short}_k1b" --prepare-only
  "$runner" -m exp.r17_integration.ops.formal_eval --run --prepared-run \
    "/data/mw_finetune/eval/${short}_k1b"
  "$runner" -m exp.r17_integration.ops.formal_eval "${topology[@]}" "${common[@]}" \
    --extra-workers 0 --official-50 --suites libero_spatial libero_10 \
    --denoising-steps 1 --eval-pool A --init-count 50 --seconds 172800 \
    --run-name "${short}_k1a" --prepare-only
  "$runner" -m exp.r17_integration.ops.formal_eval --run --prepared-run \
    "/data/mw_finetune/eval/${short}_k1a"

  # 2. CPU B-only calibration and DEV configurations, then DEV GPU campaign.
  "$runner" -m exp.r17_integration.ops.cache_knob calibrate --model "$model" \
    --source "$source" --out "$kroot/${short}_gaps.jsonl"
  "$runner" -m exp.r17_integration.ops.cache_knob dev --model "$model" \
    --source "$source" --gaps "$kroot/${short}_gaps.jsonl" --miss-steps reduced \
    --out "$kroot/${short}_dev" --run-name "${short}_kd"
  "$runner" -m exp.r17_integration.ops.formal_eval "${topology[@]}" "${common[@]}" \
    --extra-workers 0 --official-50 --suites libero_spatial libero_10 \
    --cache-study "$kroot/${short}_dev/STUDY.json" --seconds 518400 \
    --episode-seconds 1800 --run-name "${short}_kd" --prepare-only
  "$runner" -m exp.r17_integration.ops.formal_eval --run --prepared-run \
    "/data/mw_finetune/eval/${short}_kd"
  "$runner" -m exp.offline_search.rounds.r17.knob.descent \
    --lane "/data/mw_finetune/eval/${short}_kd" \
    --reference-lane "/data/mw_finetune/eval/${short}_k1b" \
    --out "$kroot/${short}_dev_results"

  # 3. A50 at frozen levels; later rounds contain only raise.json tasks.
  local round=1 table="$kroot/${short}_dev_results/levels.json" out run result
  partial=(); lanes=()
  while true; do
    out="$kroot/${short}_test_r$round"; run="${short}_kt$round"
    result="$kroot/${short}_test_results_r$round"
    "$runner" -m exp.r17_integration.ops.cache_knob test --model "$model" \
      --source "$source" --gaps "$kroot/${short}_gaps.jsonl" --table "$table" \
      --miss-steps reduced --out "$out" --run-name "$run" "${partial[@]}"
    "$runner" -m exp.r17_integration.ops.formal_eval "${topology[@]}" "${common[@]}" \
      --extra-workers 0 --official-50 --suites libero_spatial libero_10 \
      --cache-study "$out/STUDY.json" --seconds 518400 --episode-seconds 1800 \
      --run-name "$run" --prepare-only
    "$runner" -m exp.r17_integration.ops.formal_eval --run --prepared-run \
      "/data/mw_finetune/eval/$run"
    lanes+=(--lane "/data/mw_finetune/eval/$run")
    "$runner" -m exp.offline_search.rounds.r17.knob.test "${lanes[@]}" \
      --controls reduced --control "$model=${short}_k1a" --out "$result"
    if "$runner" -c 'import json,sys; sys.exit(bool(json.load(open(sys.argv[1]))))' \
      "$result/raise.json"; then break; fi
    table="$result/raise.json"; partial=(--partial); round=$((round+1))
  done
}
vla_knob pi05_libero pi pl exp/r17_integration/ops/cache_study/libero_pi/STUDY.json
vla_knob groot_libero gr gl exp/r17_integration/ops/cache_study/libero_gr/STUDY.json
```

Follow-up CPU verification (CUDA hidden; no preparation/network/launch):

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  exp/r17_integration/tests/ops/{test_libero_vla_knob,test_cache_knob,test_cache_reduced,test_library_campaign,test_libero_vla}.py \
  exp/r17_integration/tests/models/groot/{test_integration,test_shared_factory,test_libero}.py \
  exp/r17_integration/tests/models/pi05/test_pi05.py \
  exp/offline_search/rounds/r17/knob/test_knob.py \
  -k 'not all_sizes_eight_groups and not image_free_teacher'
```
