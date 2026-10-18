# Formal-A short names — 2026-10-17

The later [H100 asset staging repair](../h100_assets_1017/HANDBACK.md) adds the
second asset mirror required by the `load_cache_config` RPC. Follow its reset
and preparation status; the earlier unstaged-status statements below describe
the original short-name migration.

Use these short NAMEs and the regenerated commands, replacing the earlier long
formal_A queue lists. The design, cells, plans, seeds, fitted artifacts, row
selection, serving environment, R/B/W, references and off/first-failure rules
are unchanged. No prepare or launch was executed during this repair.

`../NAME_MAP.json` is the one authoritative old→new map for unit names, native
campaign names, old worker_auto aliases and run roots. Analysis resolves this
map; reused off/pure source lanes retain their original names and data.

Names are `fa_<model>_<mode>_<arm>_<stage>`:

| Field | Tokens |
|---|---|
| model | dp=DP, fw=Fast-WAM, co=Cosmos, pi=π0.5, gr=GR00T |
| mode | r=reduced, f=full |
| arm | u=original/unfiltered, f=success-filtered |
| stage | o=off, c=missing off completion, l=levels |

The shared π0.5 full pure references are `fa_pi_f_p10` (Libero-10) and
`fa_pi_f_ps` (Spatial), 500 episodes each. R15 native children append
`_l<SIZE>` or `_s<SIZE>`; for example `fa_pi_f_u_o_l50`. Every native run root
is `/data/mw_finetune/eval/<short-name>`. Legacy R15/R16 arm identifiers are
retained to preserve task UIDs, manifests and model-specific client settings.

There are **34 queue units**, **48 built native runs**, and **42 possible native
runs behind deferred units**. The built **163,950 episodes** and deferred upper
bound **322,000** are unchanged by renaming. Deferred units remain WAITING_OFF;
their cells are filled only after the permitted off/pure analysis. CPU checks
of their full immutable ladder templates do not dispatch episodes.

All built R17 YAMLs and plan files are byte-identical to their long-named
predecessors. To preserve YAML hashes and all RNG/task-UID inputs, their serving
asset address remains the old immutable asset address; the short run contains
an identical local mirror. **Keep the old formal-A assets/directories.** Native
prepare snapshots that mirror to the same serving asset address. Only execution
output/MPS roots move. R16 task objects are identical; R15 manifests, arm IDs,
sampler flags, fit references and seeds are identical. No fitting, calibration,
view editing or source-campaign rebuilding occurred.

The failed Cosmos run was reset using the existing reset module's new
rename-only, unstaged option:

```bash
cd /home/weiland/projects/openpi
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.reset --run /data/mw_finetune/eval/ffk1015_cosmos_libero_reduced_formal_A_original_levels --unstage-only --verified-dead
```

This constrained reset handles the documented pre-replica MPS path failure:
only an MPS guard was owned, its supervisor/child were checked dead by read-only
tether exec, no ready file exists, and no driver/replica ran. It renamed local
launch/preparation/receipt/snapshot evidence and the local /tmp stage, with
suffix `.failed_20261017T111845874672Z`; it left remote evidence intact and did
not re-stage. The RESET record reports **UNSTAGED**, preserving the original
PARTIAL receipt. Its replacement is **`fa_co_r_u_l`, 30,000 episodes**, with the
same frozen inputs and exact plan. All short runs are currently unstaged.

The pure_l10 recovery remains replacement-only: `fa_pi_f_p10` runs the whole
500-episode plan. Both earlier long pure_l10 aliases resolve to this reference;
the unreachable 445+ old outcomes must never be merged. PURE_A_OVERRIDES points
only at the short replacement root. The Spatial pure reference is also short.

CPU checks pass for every built run and every possible deferred native run:

| Check | Result |
|---|---|
| frozen MPS paths, actual host GPU UUID/hash | 90/90 pass; active MPS maximum 47 bytes, all computed paths ≤52 / 85 |
| conservative portable Unix socket | maximum 79 / 103 bytes; formal serving uses TCP |
| legacy generated manifest/collect/output filenames | maximum component 169 / 255 bytes |
| checked derived absolute paths | maximum 215 / 4,095 bytes |
| startup identity / native plugin | 48 built + 42 immutable full-ladder template checks pass |
| built plans | same 163,950 episodes; R17 files byte-identical, R16 task objects identical |
| R15 sealed code | expected hashes match locally and on H100 |
| protected old files | 648 command files and 153 plan/definition files unchanged |
| new command syntax | 360 scripts pass bash -n |
| CPU regressions | 5 tests pass |

GPU identity was read from procfs, without GPU queries or contexts. H100 UUID
`GPU-7ea0f760-d3aa-7023-c19b-95f49123311b` hashes to `e2823e`; the failed run's
new pipe is `/data/mw_finetune/eval/fa_co_r_u_l/mps/e2823e/p` (47 bytes).
H100's evaluation-root resolution was checked against the local root. The path
check executes the paths function extracted from each R17 frozen runtime tar;
the 85-byte check is unchanged. R15/R16 do not use MPS; their hypothetical bound
is included too. Evidence: `CPU_PROOF.json`, `PATH_CHECKS.json`,
`PLAN_EQUIVALENCE.json`, `GPU_IDENTITY.json`, `R15_SEALED_FILES.json` and
`deferred_cpu/`. Materialized startup reports are in `../cpu_checks/`.

The ordered lists are `../commands/H100_ORDER.txt` (16 NAMEs) and
`../commands/4090_ORDER.txt` (18 NAMEs), copied to the parent commands directory
as `formal_A_short_H100_ORDER.txt` and `formal_A_short_4090_ORDER.txt`. They put
pure references first, all off/off-completion units before level units, and
reduced modes first within each stage. Existing running chain scripts were not
changed. Queue only the new short NAMEs after the current MetaWorld lanes.

Every unit has the usual prepare/launch aliases in both command directories.
For each NAME selected by the coordinator:

```bash
cd /home/weiland/projects/openpi
NAME=fa_co_r_u_l
bash exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_${NAME}.sh
bash exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_${NAME}.sh
```

R15 uses online native worker islands (timan107 preferred, timan108 fallback,
within native caps); the held π0.5 items keep 39 workers and H100 R3/B1. All new
R16 aliases opt into the previous host logic: timan107 preferred, timan1 GPU2
fallback with the same 64 workers (cap 83), same 4090 serving and strict native
pool/horizon/resize/seed checks. R17 keeps worker_resilience redistribution.
No worker/server runtime or worker archive was changed or deployed for naming.

Preparing a deferred NAME calls build-levels once its dependencies finish and
then prepares only selected cells. Explicit build/analyze commands also remain:

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.formal_a_queue build-levels --model dp --mode full --arm original
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.formal_a analyze
```

Substitute the model, mode and arm for each deferred unit. The builder now
produces short run names for every subsequent stage. No dev-chosen levels or
withdrawn outcome data were read.

To repeat CPU checks only:

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.formal_a_short paths
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.formal_a check
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.formal_a_short startup-deferred
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m pytest -q exp/offline_search/rounds/r17/filter_failed_knob/test_formal_a_short.py exp/offline_search/rounds/r17/filter_failed_knob/test_formal_a.py exp/offline_search/rounds/r17/filter_failed_knob/test_vla_workers.py exp/offline_search/rounds/r17/filter_failed/test_timan1_gpu2.py
```

Do not rerun migrate/init or any old build-all command. Old long-named runs,
their scripts and snapshots remain as evidence; active analysis uses NAME_MAP.
