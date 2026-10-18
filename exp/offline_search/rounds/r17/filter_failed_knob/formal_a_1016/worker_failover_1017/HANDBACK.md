# Legacy VLA worker recovery — 2026-10-17

Naming update: the [short-name handback](../short_names_1017/HANDBACK.md) and
[NAME_MAP.json](../NAME_MAP.json) supersede the NAMEs below. Host-selection
logic and the replacement-only pure_l10 policy remain unchanged.

The held π0.5 R15 formal-A prepare/launch scripts select an online worker
island: **timan107 first, timan108 second**, with concurrent, bounded,
read-only tether probes. Current probes select timan107 × 39, using its existing
R15 island `/scratch/zixuans8/openpi_trace_h100`. Launch probes again and calls
native sync first if the selected host differs from prepare. Native fleet locks
and caps remain effective. An unfinished previously launched R15 arm cannot
silently change hosts because its accepted journal is remote.

H100 R3/B1, ports 23230–23232, WPS 13, serving interpreter/checkpoint, plans,
seeds, library/fit bytes and serving flags are unchanged. Placement is recorded
in `WORKER_HOST_SELECTION.json` and native `h100_launch.json` receipts.

timan108's authoritative pure_l10 journal is unreachable through permitted
tether exec (`node_offline`, no recent heartbeat). This is therefore the
authorized **whole 500-episode replacement**, not a 55-episode continuation:
`ffk1015_pi05_libero_full_formal_A_pure_l10_r1`. The manifest is byte-identical.
`PURE_A_OVERRIDES.json` points solely to the new root/arm, shared by both arms.
**Do not merge the old 445+ outcomes when timan108 returns.** The old run,
receipts and UNIT.json remain untouched; `REPLACED_PURE_L10.json` records the
policy. The interrupted NAME is absent from the active queue index; do not
queue its old scripts.

Queue after the current H100 items, in this order:

| NAME | Episodes |
|---|---:|
| `ffk1015_pi05_libero_full_formal_A_pure_l10_r1` | 500 |
| `ffk1015_pi05_libero_full_formal_A_pure_spatial` | 500 |
| `ffk1015_pi05_libero_full_formal_A_original_off` | 4,000 |
| `ffk1015_pi05_libero_full_formal_A_filtered_off` | 4,000 |
| `ffk1015_pi05_libero_full_formal_A_original_levels` | 0–16,000 |
| `ffk1015_pi05_libero_full_formal_A_filtered_levels` | 0–16,000 |

Every NAME has `prepare_<NAME>.sh` and `launch_<NAME>.sh` in both formal_a_1016
and parent command directories. The off/level units serialize their native
cell campaigns. Level preparation builds only non-qualifying cells after the
off/pure dependencies finish; placement selection propagates to those children.
The plain ordered list is `../commands/H100_PI05_RECOVERY_ORDER.txt`, also copied
to the parent commands as `formal_A_1017_H100_PI05_RECOVERY_ORDER.txt`.

Use the existing chain runner with the six NAMEs, or execute one pair at a time:

```bash
cd /home/weiland/projects/openpi
NAME=ffk1015_pi05_libero_full_formal_A_pure_l10_r1
bash exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_${NAME}.sh
bash exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_${NAME}.sh
```

The explicit later build/analyze commands remain:

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.formal_a_queue build-levels --model pi05 --mode full --arm original
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.formal_a_queue build-levels --model pi05 --mode full --arm filtered
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.formal_a analyze
```

R16 also hard-codes its worker host. Its existing queued aliases are unchanged.
New opt-in aliases append `_worker_auto` to each reduced π0.5/GR00T formal-A
unit NAME (off_complete and levels). They keep the same native group, journal,
4090 serving, replicas, plans and **64 workers**. They prefer timan107, with
**timan1_gpu2 / timan1 × 64** as the alternate (LIBERO cap 83). Receipt and
hostname attestation follow the selected host; pool digests, resize, horizon,
settling and seed checks remain strict. The agent's host/lock literals are
routed in memory; the timan1 GPU2 namespace/renderer guard is reused. Existing
R17 fleet/native paths retain their behavior.

If the R16 fallback is selected, coordinator prepare sends the unchanged
`rounds/r16/conductor_deployment/worker.tar` through existing public-network
rsync transport, extracts into the new
`/scratch/zixuans8/openpi_r16_conductor_ffk1017` island on timan1 and verifies
native `worker.sha256`. Existing islands are not replaced. The source archive's
run_agent, WorkerSpec, WorkerAgent and spawn function equal the current code by
AST. No deployment was performed during this repair.

For a future R16 queue substitution, use, for example,
`ffk1015_pi05_libero_reduced_formal_A_original_levels_worker_auto`, with the same
prepare/launch convention. Do not queue both aliases for one native unit.
No running chain or currently queued R16 command was replaced. Shared worker
nodes still require the coordinator's usual scheduling.

CPU proof: `CPU_PROOF.json`, native startup report
`../cpu_checks/ffk1015_pi05_libero_full_formal_A_pure_l10_r1.json`, and native
planner `../plans/ffk1015_pi05_libero_full_formal_A_pure_l10_r1/audit.json`.
The native plugin/startup, H100 model argv equality and transport-blocked native
planner pass: **500 unique task/init pairs**. All **72** held definition files
and **238** protected command/chain files retain their hashes. All **84**
affected/new scripts pass `bash -n`. Two CPU regressions pass, including native
generation of 64 GPU2 R16 worker specs without starting processes:

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m pytest -q exp/offline_search/rounds/r17/filter_failed_knob/test_vla_workers.py exp/offline_search/rounds/r17/filter_failed/test_timan1_gpu2.py
```

No prepare, staging, launch, server, worker, driver, chain or GPU operation was
executed for this repair; remote operations were read-only tether probes.
