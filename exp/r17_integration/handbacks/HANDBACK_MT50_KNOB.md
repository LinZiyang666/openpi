# MT50 per-task IR knob CPU handback

Owner correction applied: MT50 has no DEV split. Seed 8 is exclusively the 2,500-episode cache library. Knob descent runs directly on seed 7, idx 0–49. Serving LOEO is off; periodic calibration still uses selected-library LOEO replay with vision=0/miss=1 to target model-call fractions.

Each model × step mode is one campaign covering sizes 250, 500, 1000, 1500, 2000, 2500 and levels k50, k40, k30, k20, k10, koff. Pure comes from the same-step seed-7 baseline. At most two fewer successes of 50 passes; the first failure freezes the previous level. Later recoveries do not lower the choice. The frozen level’s measured episodes are final, and a pure choice reuses baseline episodes. A complete koff cell reaching the pure four-group mean SR selects koff for the entire size. No separate test/retest round or reduced pure DEV reference is needed.

Files changed:

- `exp/r17_integration/ops/cache_knob.py`: MT50 models/sizes, native and reduced sampler metadata, separate calibration records by step mode, direct seed-7 batches and exact printed formal commands. `dev` is a compatibility command name: for MT50 it builds A50 TEST descent. `test` without a table builds the same complete descent batch. `test --table ... --partial` remains available for explicit task subsets; it is not a stage of the corrected experiment.
- `exp/r17_integration/ops/cache_study.py`: shared full-system base selection, including π0.5 `crg` in knob manifests.
- `exp/offline_search/rounds/r17/cache/analyze_cache.py`: existing lane loader accepts optional suites and canonicalizes `metaworld_mt50` records to `mt50`, preserving LIBERO defaults.
- `exp/offline_search/rounds/r17/knob/{common,descent,test}.py`: 50-task lanes, both SR means, same-step baselines, direct final selection and reuse of existing no-periodic study cells. The existing loader reads accepted outcomes, resume ancestors and call/control evidence.
- `exp/offline_search/rounds/r17/knob/ir_formal.py`: four MT50 stage profiles and both SR conventions.
- `exp/r17_integration/tests/ops/test_mt50_knob.py`: CPU stub coverage below.
- This handback.

Episode counts before any existing koff shortcut:

| Model | Full sampler | Reduced sampler | Full batch | Reduced batch | Final selected episodes per mode |
|---|---|---|---:|---:|---:|
| DP | DDPM100 | DDIM10, eta 0 | 90,000 | 90,000 | 15,000 |
| Fast-WAM | K10 | K3 | 90,000 | 90,000 | 15,000 |
| π0.5 | K10 | K2 | 90,000 | 90,000 | 15,000 |
| GR00T | K4 | K2, fixed at startup | 90,000 | 90,000 | 15,000 |

One level/size = 2,500 episodes; one size/all levels = 15,000; one model/mode/all sizes = 90,000. Total knob episodes = 720,000. The new DP full H100 baseline adds 2,500: 722,500 new episodes maximum. Existing pure baselines are reused, and the selected final samples do not add runs. Omitting a size satisfied by an existing same-step koff study cell saves 15,000 episodes for that model/mode. No seed-8 evaluation episodes are planned.

References:

| Model | Full seed-7 baseline | Reduced seed-7 baseline | Server |
|---|---|---|---|
| DP | `tdp7h_full` (new command below) | `tdp7d10_0851` | H100 |
| Fast-WAM | `tfw7full_0944` | `tfw7k3_1051` | H100 |
| GR00T | `tgr7h_full` (coordinator queued) | `tgr7h_k2` (coordinator queued) | H100 |
| π0.5 | `tpi7_0536` | `tpi7k2_1135` | 4090 |

The 4090 DP run `tdp7_0729` is not the reference. Library teacher outcomes are not a reference under the correction. If the new DP baseline receives another name, use `--control dp_mw=ACTUAL_RUN` in analysis. Both model labels and formal selectors work in that option.

CPU tests cover:

- Every model/mode: all-size/all-level metadata totaling 90,000 episodes, seed 7 idx 0–49, A50, no serving LOEO and no pure arm; one complete 90,000-episode DP plan plus every-cell two-task plans for other combinations verify unique episode identities; library subsets use per-task `size_5 ... size_50` paths.
- π0.5 C5/crg versus other models C10/chrg; corrector and Q3 stall mode persist, and koff disables periodic refresh only.
- Calibration calls the selected-library LOEO replay and real solver, emits five targets per size, and labels full/reduced records. New calibration tests use a replay stub and do not claim completed production-library calibration.
- Fake adapters receive the native/reduced sampler for stall, periodic and pure calls with capture disabled. Pure is tested as a serving regression, not planned in the MT50 batch.
- Two-success boundary, first failure and later recovery; incomplete visited levels remain pending, while missing levels below a failure do not prevent freezing.
- Four-group versus per-task mean; whole-size koff selection, including reuse of existing base-group cache study cells.
- Same-step baseline routing, accepted journal/step parsing, identical descent/test final results, explicit partial plans, fixed GR00T startup K from study YAML, formal command parsing and the formal-IR Markdown/JSON export.
- Formal IR arithmetic from MT50 profiles, including DP CPU text subtraction.

Existing LIBERO knob/planning/VLA regressions: **52 passed**, 478.32 s. **63 unique MT50 tests verified**: the full CPU run passed 50 cases; after correcting fixture directory creation and parser expectations, the focused final-analysis/command run passed all 14 selected cases (including the 13 previously failing fixture cases), 62.68 s. The table-free TEST batch and DEV compatibility checks also passed their focused rerun: 8 passed, 47.40 s. All 13 handback shell blocks passed bash syntax checking without execution. The suite includes no GPU or network execution.

```bash
cd /home/weiland/projects/openpi
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q exp/r17_integration/tests/ops/test_mt50_knob.py
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q exp/offline_search/rounds/r17/knob/test_knob.py exp/r17_integration/tests/ops/test_cache_knob.py exp/r17_integration/tests/ops/test_libero_vla_knob.py
```

All six production source cohorts were inspected read-only: selected full-system fits and stall paths exist, with C10 for DP/Fast-WAM/GR00T and C5 for π0.5. Production calibration and formal preparation are left to the coordinator. Calibration files are separate by step mode even when the call-fraction replay yields equal gaps.

Formal IR uses `exp/r17_integration/profile/results_mt50/{dp,fastwam,pi05,groot}_mw_mt50.json`, C0=5 and actual executed controls, including terminal tails:

`IR = 5 × [v × (policy calls + library decisions) + m(step mode) × policy calls] / executed controls`.

The unit is full-step pure stage1+stage2+denoising-loop cost. DP’s CPU CLIP time is subtracted from stage1 on both sides. MT50 Fast-WAM retains the supplied measured stage2; the LIBERO cached-text profile is not substituted. These supplied profiles were measured at batch 1 on the 4090: IR is that stage-cost normalization, not H100 campaign wall time. Reduced pure IR is `v + m(reduced)`. Missing cost evidence produces null IR.

Scope: CPU tests and read-only library metadata inspection. No GPU/server/worker/campaign launch, remote transfer, credential access, commit/push or edits to running/frozen campaigns. No new admission, fingerprint or validation gate was added.

CPU plan generation (fresh output paths; these commands do not invoke formal preparation or launch):

```bash
cd /home/weiland/projects/openpi
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python -m exp.r17_integration.ops.cache_knob calibrate --model dp_mw --miss-steps full --source /data/r17_library/studies_mt50/dp_k10/STUDY.json --out /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/dp_mw/full/gaps.jsonl
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python -m exp.r17_integration.ops.cache_knob dev --model dp_mw --miss-steps full --source /data/r17_library/studies_mt50/dp_k10/STUDY.json --gaps /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/dp_mw/full/gaps.jsonl --out /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/dp_mw/full/batch --run-name dpmk_full
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python -m exp.r17_integration.ops.cache_knob calibrate --model dp_mw --miss-steps reduced --source /data/r17_library/studies_mt50/dp_k10/STUDY.json --out /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/dp_mw/reduced/gaps.jsonl
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python -m exp.r17_integration.ops.cache_knob dev --model dp_mw --miss-steps reduced --source /data/r17_library/studies_mt50/dp_k10/STUDY.json --gaps /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/dp_mw/reduced/gaps.jsonl --out /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/dp_mw/reduced/batch --run-name dpmk_reduced
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python -m exp.r17_integration.ops.cache_knob calibrate --model fastwam_mw --miss-steps full --source /data/r17_library/studies_mt50/fw_k3/STUDY.json --out /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/fastwam_mw/full/gaps.jsonl
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python -m exp.r17_integration.ops.cache_knob dev --model fastwam_mw --miss-steps full --source /data/r17_library/studies_mt50/fw_k3/STUDY.json --gaps /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/fastwam_mw/full/gaps.jsonl --out /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/fastwam_mw/full/batch --run-name fastwammk_full
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python -m exp.r17_integration.ops.cache_knob calibrate --model fastwam_mw --miss-steps reduced --source /data/r17_library/studies_mt50/fw_k3/STUDY.json --out /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/fastwam_mw/reduced/gaps.jsonl
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python -m exp.r17_integration.ops.cache_knob dev --model fastwam_mw --miss-steps reduced --source /data/r17_library/studies_mt50/fw_k3/STUDY.json --gaps /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/fastwam_mw/reduced/gaps.jsonl --out /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/fastwam_mw/reduced/batch --run-name fastwammk_reduced
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python -m exp.r17_integration.ops.cache_knob calibrate --model groot_mw --miss-steps full --source /data/r17_library/studies_mt50/gr_k2/STUDY.json --out /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/groot_mw/full/gaps.jsonl
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python -m exp.r17_integration.ops.cache_knob dev --model groot_mw --miss-steps full --source /data/r17_library/studies_mt50/gr_k2/STUDY.json --gaps /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/groot_mw/full/gaps.jsonl --out /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/groot_mw/full/batch --run-name grootmk_full
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python -m exp.r17_integration.ops.cache_knob calibrate --model groot_mw --miss-steps reduced --source /data/r17_library/studies_mt50/gr_k2/STUDY.json --out /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/groot_mw/reduced/gaps.jsonl
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python -m exp.r17_integration.ops.cache_knob dev --model groot_mw --miss-steps reduced --source /data/r17_library/studies_mt50/gr_k2/STUDY.json --gaps /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/groot_mw/reduced/gaps.jsonl --out /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/groot_mw/reduced/batch --run-name grootmk_reduced
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python -m exp.r17_integration.ops.cache_knob calibrate --model pi05_mw --miss-steps full --source /data/r17_library/studies_mt50/pi_k2/STUDY.json --out /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/pi05_mw/full/gaps.jsonl
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python -m exp.r17_integration.ops.cache_knob dev --model pi05_mw --miss-steps full --source /data/r17_library/studies_mt50/pi_k2/STUDY.json --gaps /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/pi05_mw/full/gaps.jsonl --out /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/pi05_mw/full/batch --run-name pi05mk_full
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python -m exp.r17_integration.ops.cache_knob calibrate --model pi05_mw --miss-steps reduced --source /data/r17_library/studies_mt50/pi_k2/STUDY.json --out /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/pi05_mw/reduced/gaps.jsonl
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python -m exp.r17_integration.ops.cache_knob dev --model pi05_mw --miss-steps reduced --source /data/r17_library/studies_mt50/pi_k2/STUDY.json --gaps /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/pi05_mw/reduced/gaps.jsonl --out /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/pi05_mw/reduced/batch --run-name pi05mk_reduced
```

To reuse an existing no-periodic study, run `test.py --koff-lane /data/mw_finetune/eval/RUN --controls full|reduced --out OUTPUT` on the same-step lane. Qualifying complete sizes have `no_knob=true` and final koff measurements. Other sizes remain pending. Build only remaining sizes with `--sizes TOTALS...` if desired. This is analysis, not a campaign launch.

Coordinator-only GPU order: execute the following prepare/run pairs serially after existing lane ownership has released. The assistant has executed none of these commands. Serial order keeps timan1 at 20 workers total (physical GPU3 only) and avoids shared π0.5/GR00T ports. No cross-lane orchestration or new launch check is introduced.

1. New DP full-step H100 seed-7 baseline: 2,500 episodes, R4/B8/MPS matching DP knob topology. The already queued GR00T H100 baselines remain the coordinator’s responsibility.

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu /home/weiland/projects/openpi/.venv/bin/python -u -m exp.r17_integration.ops.formal_eval --model dp_mw --server-node h100 --gpu cuda:0 --replicas 4 --replica-startup-width 3 --max-batch 8 --batch-wait-ms 5 --mps --memory-budget-gib 24 --driver-mode group-commit --worker-node timan107 --extra-workers 0 --worker-counts timan107=30,timan1=20 --worker-memory-mib 2048 --worker-vram-mib 1024 --worker-egl-devices 0,1,2,3,4,5,6,7 --worker-max-episodes 0 --worker-restart-limit 2 --retries 2 --seconds 172800 --episode-seconds 1800 --port 23180 --driver-port 23189 --run-name tdp7h_full --eval-pool A --init-count 50 --denoising-steps 100 --prepare-only
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu /home/weiland/projects/openpi/.venv/bin/python -u -m exp.r17_integration.ops.formal_eval --run --prepared-run /data/mw_finetune/eval/tdp7h_full
```

2. dp_mw full, all sizes: 90,000 episodes.

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu /home/weiland/projects/openpi/.venv/bin/python -u -m exp.r17_integration.ops.formal_eval --model dp_mw --server-node h100 --gpu cuda:0 --replicas 4 --replica-startup-width 3 --max-batch 8 --batch-wait-ms 5 --mps --memory-budget-gib 24 --driver-mode group-commit --worker-node timan107 --extra-workers 0 --worker-counts timan107=30,timan1=20 --worker-memory-mib 2048 --worker-vram-mib 1024 --worker-egl-devices 0,1,2,3,4,5,6,7 --worker-max-episodes 0 --worker-restart-limit 2 --retries 2 --seconds 518400 --episode-seconds 1800 --port 23180 --driver-port 23189 --run-name dpmk_full --cache-study /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/dp_mw/full/batch/STUDY.json --prepare-only
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu /home/weiland/projects/openpi/.venv/bin/python -u -m exp.r17_integration.ops.formal_eval --run --prepared-run /data/mw_finetune/eval/dpmk_full
```

3. dp_mw reduced, all sizes: 90,000 episodes.

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu /home/weiland/projects/openpi/.venv/bin/python -u -m exp.r17_integration.ops.formal_eval --model dp_mw --server-node h100 --gpu cuda:0 --replicas 4 --replica-startup-width 3 --max-batch 8 --batch-wait-ms 5 --mps --memory-budget-gib 24 --driver-mode group-commit --worker-node timan107 --extra-workers 0 --worker-counts timan107=30,timan1=20 --worker-memory-mib 2048 --worker-vram-mib 1024 --worker-egl-devices 0,1,2,3,4,5,6,7 --worker-max-episodes 0 --worker-restart-limit 2 --retries 2 --seconds 518400 --episode-seconds 1800 --port 23180 --driver-port 23189 --run-name dpmk_reduced --cache-study /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/dp_mw/reduced/batch/STUDY.json --prepare-only
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu /home/weiland/projects/openpi/.venv/bin/python -u -m exp.r17_integration.ops.formal_eval --run --prepared-run /data/mw_finetune/eval/dpmk_reduced
```

4. fastwam_mw full, all sizes: 90,000 episodes.

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu /home/weiland/projects/openpi/.venv/bin/python -u -m exp.r17_integration.ops.formal_eval --model fastwam_mw --server-node h100 --gpu cuda:0 --replicas 3 --replica-startup-width 3 --max-batch 16 --batch-wait-ms 5 --mps --memory-budget-gib 76 --driver-mode group-commit --worker-node timan107 --extra-workers 0 --worker-counts timan107=30,timan1=20 --worker-memory-mib 2048 --worker-vram-mib 1024 --worker-egl-devices 0,1,2,3,4,5,6,7 --worker-max-episodes 0 --worker-restart-limit 2 --retries 2 --seconds 518400 --episode-seconds 1800 --port 23170 --driver-port 23179 --run-name fastwammk_full --cache-study /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/fastwam_mw/full/batch/STUDY.json --prepare-only
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu /home/weiland/projects/openpi/.venv/bin/python -u -m exp.r17_integration.ops.formal_eval --run --prepared-run /data/mw_finetune/eval/fastwammk_full
```

5. fastwam_mw reduced, all sizes: 90,000 episodes.

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu /home/weiland/projects/openpi/.venv/bin/python -u -m exp.r17_integration.ops.formal_eval --model fastwam_mw --server-node h100 --gpu cuda:0 --replicas 3 --replica-startup-width 3 --max-batch 16 --batch-wait-ms 5 --mps --memory-budget-gib 76 --driver-mode group-commit --worker-node timan107 --extra-workers 0 --worker-counts timan107=30,timan1=20 --worker-memory-mib 2048 --worker-vram-mib 1024 --worker-egl-devices 0,1,2,3,4,5,6,7 --worker-max-episodes 0 --worker-restart-limit 2 --retries 2 --seconds 518400 --episode-seconds 1800 --port 23170 --driver-port 23179 --run-name fastwammk_reduced --cache-study /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/fastwam_mw/reduced/batch/STUDY.json --prepare-only
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu /home/weiland/projects/openpi/.venv/bin/python -u -m exp.r17_integration.ops.formal_eval --run --prepared-run /data/mw_finetune/eval/fastwammk_reduced
```

6. groot_mw full, all sizes: 90,000 episodes.

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu /home/weiland/projects/openpi/.venv/bin/python -u -m exp.r17_integration.ops.formal_eval --model groot_mw --server-node h100 --gpu cuda:0 --replicas 8 --replica-startup-width 4 --max-batch 1 --batch-wait-ms 5 --mps --memory-budget-gib 56 --driver-mode group-commit --worker-node timan107 --extra-workers 0 --worker-counts timan107=40,timan1=20 --worker-memory-mib 2048 --worker-vram-mib 1024 --worker-egl-devices 0,1,2,3,4,5,6,7 --worker-max-episodes 0 --worker-restart-limit 2 --retries 2 --seconds 518400 --episode-seconds 1800 --port 23160 --driver-port 23169 --run-name grootmk_full --cache-study /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/groot_mw/full/batch/STUDY.json --prepare-only
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu /home/weiland/projects/openpi/.venv/bin/python -u -m exp.r17_integration.ops.formal_eval --run --prepared-run /data/mw_finetune/eval/grootmk_full
```

7. groot_mw reduced, all sizes: 90,000 episodes.

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu /home/weiland/projects/openpi/.venv/bin/python -u -m exp.r17_integration.ops.formal_eval --model groot_mw --server-node h100 --gpu cuda:0 --replicas 8 --replica-startup-width 4 --max-batch 1 --batch-wait-ms 5 --mps --memory-budget-gib 56 --driver-mode group-commit --worker-node timan107 --extra-workers 0 --worker-counts timan107=40,timan1=20 --worker-memory-mib 2048 --worker-vram-mib 1024 --worker-egl-devices 0,1,2,3,4,5,6,7 --worker-max-episodes 0 --worker-restart-limit 2 --retries 2 --seconds 518400 --episode-seconds 1800 --port 23160 --driver-port 23169 --run-name grootmk_reduced --cache-study /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/groot_mw/reduced/batch/STUDY.json --prepare-only
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu /home/weiland/projects/openpi/.venv/bin/python -u -m exp.r17_integration.ops.formal_eval --run --prepared-run /data/mw_finetune/eval/grootmk_reduced
```

8. pi05_mw full, all sizes: 90,000 episodes.

```bash
taskset -c 22-43,66-87 env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu /home/weiland/projects/openpi/.venv/bin/python -u -m exp.r17_integration.ops.formal_eval --model pi05_mw --server-node local --gpu cuda:0 --replicas 5 --replica-startup-width 3 --max-batch 1 --batch-wait-ms 5 --mps --memory-budget-gib 47 --driver-mode group-commit --worker-node timan107 --extra-workers 0 --worker-counts timan107=24,timan1=20 --worker-memory-mib 2048 --worker-vram-mib 1024 --worker-egl-devices 0,1,2,3,4,5,6,7 --worker-max-episodes 0 --worker-restart-limit 2 --retries 2 --seconds 518400 --episode-seconds 1800 --port 23160 --driver-port 23169 --run-name pi05mk_full --cache-study /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/pi05_mw/full/batch/STUDY.json --prepare-only
taskset -c 22-43,66-87 env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu /home/weiland/projects/openpi/.venv/bin/python -u -m exp.r17_integration.ops.formal_eval --run --prepared-run /data/mw_finetune/eval/pi05mk_full
```

9. pi05_mw reduced, all sizes: 90,000 episodes.

```bash
taskset -c 22-43,66-87 env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu /home/weiland/projects/openpi/.venv/bin/python -u -m exp.r17_integration.ops.formal_eval --model pi05_mw --server-node local --gpu cuda:0 --replicas 5 --replica-startup-width 3 --max-batch 1 --batch-wait-ms 5 --mps --memory-budget-gib 47 --driver-mode group-commit --worker-node timan107 --extra-workers 0 --worker-counts timan107=24,timan1=20 --worker-memory-mib 2048 --worker-vram-mib 1024 --worker-egl-devices 0,1,2,3,4,5,6,7 --worker-max-episodes 0 --worker-restart-limit 2 --retries 2 --seconds 518400 --episode-seconds 1800 --port 23160 --driver-port 23169 --run-name pi05mk_reduced --cache-study /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/pi05_mw/reduced/batch/STUDY.json --prepare-only
taskset -c 22-43,66-87 env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu /home/weiland/projects/openpi/.venv/bin/python -u -m exp.r17_integration.ops.formal_eval --run --prepared-run /data/mw_finetune/eval/pi05mk_reduced
```

Topologies: DP H100 R4/B8/MPS/24 GiB; Fast-WAM H100 R3/B16/MPS/76 GiB; GR00T H100 R8/B1/MPS/56 GiB (queued baseline/cache topology); π0.5 4090 R5/B1/MPS/47 GiB with NUMA1 affinity. Workers use MT50 RAM2048/VRAM1024 MiB and no recycling; timan107 EGL0..7 and timan1 physical GPU3. DP/Fast-WAM/GR00T use timan107×30/30/40 plus timan1×20; π0.5 uses timan107×24 plus timan1×20. GR00T startup K comes from the study YAML and matches all cells; modes have separate campaigns.

CPU final analysis after each batch (also usable incrementally):

```bash
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python exp/offline_search/rounds/r17/knob/descent.py --lane /data/mw_finetune/eval/dpmk_full --miss-steps full --out /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/dp_mw/full/results
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python exp/offline_search/rounds/r17/knob/descent.py --lane /data/mw_finetune/eval/dpmk_reduced --miss-steps reduced --out /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/dp_mw/reduced/results
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python exp/offline_search/rounds/r17/knob/descent.py --lane /data/mw_finetune/eval/fastwammk_full --miss-steps full --out /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/fastwam_mw/full/results
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python exp/offline_search/rounds/r17/knob/descent.py --lane /data/mw_finetune/eval/fastwammk_reduced --miss-steps reduced --out /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/fastwam_mw/reduced/results
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python exp/offline_search/rounds/r17/knob/descent.py --lane /data/mw_finetune/eval/grootmk_full --miss-steps full --out /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/groot_mw/full/results
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python exp/offline_search/rounds/r17/knob/descent.py --lane /data/mw_finetune/eval/grootmk_reduced --miss-steps reduced --out /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/groot_mw/reduced/results
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python exp/offline_search/rounds/r17/knob/descent.py --lane /data/mw_finetune/eval/pi05mk_full --miss-steps full --out /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/pi05_mw/full/results
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python exp/offline_search/rounds/r17/knob/descent.py --lane /data/mw_finetune/eval/pi05mk_reduced --miss-steps reduced --out /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/pi05_mw/reduced/results
```

Equivalent: `test.py --lane RUN --controls full|reduced --out OUTPUT`. The command printer can reproduce any prepare/run pair with `.venv/bin/python -m exp.r17_integration.ops.cache_knob commands --model MODEL --study STUDY.json --miss-steps full|reduced --run-name RUN`; the new baseline pair uses `commands --model dp_mw --reference test --miss-steps full --run-name tdp7h_full`. MT50 writes final `levels.json`, `descent.json`, `test.json`, and `DESCENT.md`; it emits no raise list or separate retest stage. JSON includes per-task level counts, difficulty-group SRs, group mean first, task mean second, call/control counts, formal IR, same-step pure IR and reference run names. Repeat `--lane` for resume/explicit partial measurement lanes; resume ancestors are also followed. Pure choices use baseline measurements.

Combined formal IR table:

```bash
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python exp/offline_search/rounds/r17/knob/ir_formal.py \
  --test full:/home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/dp_mw/full/results/test.json \
  --test reduced:/home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/dp_mw/reduced/results/test.json \
  --test full:/home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/fastwam_mw/full/results/test.json \
  --test reduced:/home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/fastwam_mw/reduced/results/test.json \
  --test full:/home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/groot_mw/full/results/test.json \
  --test reduced:/home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/groot_mw/reduced/results/test.json \
  --test full:/home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/pi05_mw/full/results/test.json \
  --test reduced:/home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/pi05_mw/reduced/results/test.json \
  --out /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/knob/mt50/ir_final
```
