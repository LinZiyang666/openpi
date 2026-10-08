Emitted five B-dev camera-choice roots here, each with **`both` and `wrist_sub`**. Arm names are `<root>_<method>`. Nothing was launched or transferred; existing roots, artifacts, and methods are unchanged. A is never used for camera selection, sampling, fitting, or collection.

| Root | Available/selected B inits per task, in task order 0–9 | Episodes/arm |
|---|---|---:|
| `r12_camdev_pi05_l10_50` | 27, 28, 31, 28, 32, 27, 28, 29, 28, 28 | 286 |
| `r12_camdev_pi05_spatial_50` | 27, 28, 31, 28, 32, 27, 28, 29, 28, 28 | 286 |
| `r12_camdev_pi05_spatial_200` | 15, 15, 18, 16, 19, 16, 15, 17, 16, 16 | 163 |
| `r12_camdev_groot_spatial_50` | 27, 28, 31, 28, 32, 27, 28, 29, 28, 28 | 286 |
| `r12_camdev_groot_l10_50` | 27, 28, 31, 28, 32, 27, 28, 29, 28, 28 | 286 |

K40 is unavailable for every task. To enforce disjointness from **both** named screens, exclusions use the union of the matching model/suite manifests in `r11_dev_k10_size50` **and** `r11_dev_k10_size200`, including the other size. `prepare.py` draws the complete outside-library B complement through `closed_loop/devset.build_manifest` with the original seed `20261004`, removes that union, and selects `min(40, available)` using the same PCG64(seed + task) rule. All remaining inits are selected. Each arm's actual frozen fit is checked through `devset.arm_contract`; the served library subset and both prior-screen overlaps are empty. Both arms share one exact manifest per root.

`roots.json`, each `ROOT.json`, and `dev_selection.json` contain exact pairs, per-task availability, prior manifest byte/selection hashes, served subset identities, and canonical selection/manifest-byte hashes. Artifacts reference the existing `ablate/artifacts/<cell>/both.pkl` and `wrist_sub.pkl`, with their original SHA256; no refit. Recording flags and look contracts are absent. Compute-share ledgers and B-labelled per-episode journals remain enabled. Every arm selects `closed_loop.execution_worker`. Engagement requires `calls=0`, `both.one_camera_looks=0`, and `wrist_sub.two_camera_looks=0`.

**267 CPU tests passed:** 30 camdev, 71 ablation, 128 R12, 38 devset; zero failures or deselections. Evidence: `CPU_CAMDEV.log/.xml`, `CPU_REGRESSION.log/.xml`, `ARGV_VERIFICATION.json` (all ten arms, local and h100 full final argv), `VERIFICATION.json`, and `PRESERVED_INPUTS.json`. Planning and sync used real implementations with transport mocked and scratch output only. The historical suites ran through `cpu_suite.py`, copying legacy definitions to `/tmp` and isolating selftest scratch; original live run directories were untouched. CUDA was hidden, affinity `22-37,66-81`, OMP/BLAS threads one. One existing pynvml deprecation warning per run.

No new runtime code needs pushing. Local serving needs no h100 transfer. The selected timan island must retain the existing ablation worker snapshot (`execution_worker.py`, `worker_pool.py`, `devset.py`, `ops/h100/run_arm.sh`, and `ops/remote/run_gtp_subset.py`). Coordinator sync transfers configs/matrices and frozen B init files/records; chain transfers exact manifests and pool contracts. Optional h100 serving additionally uses the existing R12 ablation server snapshot and standard sync's frozen artifacts/dependencies. The new emitter, tests, evidence, and handback stay local.

Exact coordinator commands below are **provided only, never executed**. Use after the selected fleet/ports become available. Defaults: local serving, timan107, four servers, WPS16, ports `23150–23153`. The wrapper refuses occupied live-run ports `23100–23103`.

```bash
cd /home/weiland/projects/openpi
export SERVER_HOST=local WORKER_HOST=timan107
export PORTS=23150,23151,23152,23153 LOCAL_SERVER_COUNT=4 WPS=16
S=exp/offline_search/rounds/r12/ablate/camdev/coordinator_commands.sh
for cell in pi05_l10_50 pi05_spatial_50 pi05_spatial_200 groot_spatial_50 groot_l10_50; do
  bash "$S" plan "$cell" both wrist_sub
  bash "$S" sync "$cell" both wrist_sub
  bash "$S" chain "$cell" both wrist_sub
done
```

For h100/timan108, set `SERVER_HOST=h100 WORKER_HOST=timan108 PORTS=23250,23251,23252,23253 WPS=8`, then use the same commands. Every wrapper invocation runs `taskset -c 22-37,66-81 env CUDA_VISIBLE_DEVICES= .../.venv/bin/python -B -m exp.offline_search.closed_loop.ops.h100.control ACTION ROOT ROOT_both ROOT_wrist_sub` with OMP/BLAS threads one and selection/recording environment overrides cleared.

Reproduce CPU verification without launching:

```bash
P=(taskset -c 22-37,66-81 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -B)
"${P[@]}" -m pytest -q -p no:cacheprovider exp/offline_search/rounds/r12/ablate/camdev/test_camera_dev.py
"${P[@]}" -m exp.offline_search.rounds.r12.ablate.camdev.cpu_suite
```
