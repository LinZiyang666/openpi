# H100 short-name asset staging repair — 2026-10-17

The short-name copies retained the long-name `campaign.assets` address to keep
the frozen YAMLs and task UIDs byte-identical. They moved `local_assets` to the
short run. The server loads descriptors from `assets`, but the coordinator's
`load_cache_config` RPC passes `local_assets/full.yaml`. Staging installed only
the former tree. Read-only H100 inspection confirmed the long-name tree existed
and the short-name tree was absent for `fa_fw_f_f_o`.

`filter_failed.stage.stage_server_inputs` now installs its one sealed snapshot
at both addresses, deduplicating them when equal. This applies to every H100
R17 prepare, including repeated preparation and deferred level campaigns.
It uses the existing SHA-verified installer and public-network rsync capped at
16,384 KiB/s. The additional installed tree lives on `/data`; the existing
temporary archive location and disk guard are unchanged. Scientific inputs,
recipes, runtimes, plans and serving parameters are unchanged. Local-serving
4090 and legacy R15/R16 campaigns take their existing paths.

The reset module's `--unstage-only` now also accepts an incomplete preparation
with no LAUNCH/RECEIPT or execution evidence. It preserves the local preparation
records and stage by rename; it does not start processes or change remote files.

## Execution and checks

* `fa_fw_f_u_o`: reset record
  `RESET.failed_20261017T143811235522Z.json` reports UNSTAGED; subsequent native
  prepare **PASS**, with no server/worker started. Selected execution workers:
  local×8, timan1×43, timan107×34; offline timan108 was excluded by resilience.
* H100 read-only proof `fa_fw_f_u_o.json`: **PASS**; both asset addresses exist,
  **167 files per tree / 334 matching SHA256 checks**, and **77 referenced
  YAML/model/runtime/interpreter/control input paths** exist. No missing inputs.
* `CLOSED_H100.json`: all **8** recorded H100 child/supervisor PIDs for
  `fa_fw_f_f_o` are dead. The existing reset's closed-startup check also passes.
  Its remote rename-only reset is pending the coordinator's answer: it requires
  tether writes to rename closed process receipts and MPS/server records, while
  the task otherwise restricts tether exec to read-only.
* `UNCHANGED.json` checks all five built H100 R17 plans and asset trees.
* Two CPU tests pass: asset-address mapping/deduplication/incomplete-prepare reset,
  and the existing short-name/closed-MPS regression. The initial test fixture
  used the real original-off basename and renamed its authorized local stage;
  the fixture was isolated before rerunning. That stage evidence remains at
  `/tmp/r17_formal/fa_fw_f_u_o.failed_20261017T143752355907Z` and its record at
  `/tmp/pytest-of-weiland/pytest-4282/test_h100_short_asset_aliases_0/fa_fw_f_u_o/RESET.failed_20261017T143752355907Z.json`.

No existing campaign was rebuilt. Running campaigns, queues and the disk guard
were not changed. No server, worker, driver, chain or GPU operation was started.

## Commands

From `/home/weiland/projects/openpi`; coordinator only launches the jobs.
Original-off has already been prepared successfully:

```bash
bash exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_fa_fw_f_u_o.sh
bash exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_fa_fw_f_u_o.sh
```

The existing full closed-startup reset for filtered-off (requires the remote
rename exception; it also re-prepares through the fixed staging code):

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.reset --run /data/mw_finetune/eval/fa_fw_f_f_o
bash exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_fa_fw_f_f_o.sh
bash exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_fa_fw_f_f_o.sh
```

For every other NAME below, use the existing `commands/prepare_<NAME>.sh`
followed by `commands/launch_<NAME>.sh`. Deferred level preparation retains the
existing off-analysis → selective build → prepare workflow; no extra cells are
dispatched. No script regeneration is needed because these scripts import the
fixed stage module.

Reproduce the read-only path proof and targeted tests:

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r17/filter_failed_knob/formal_a_1016/h100_assets_1017/verify.py --run /data/mw_finetune/eval/fa_fw_f_u_o
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m pytest -q exp/offline_search/rounds/r17/filter_failed_knob/test_h100_assets.py exp/offline_search/rounds/r17/filter_failed_knob/test_formal_a_short.py
```

## Remaining H100 order

The shared pure jobs `fa_pi_f_p10` and `fa_pi_f_ps` already ran; do not rerun.
Keep the preceding MetaWorld queue intact. `fa_fw_f_f_o` needs its closed remote
reset before launch; all subsequent R17 prepares use the repaired staging.
Off stages precede level stages; R15 host failover is unchanged.

```text
fa_fw_f_u_o
fa_fw_f_f_o
fa_co_f_u_o
fa_co_f_f_o
fa_pi_f_u_o
fa_pi_f_f_o
fa_co_r_u_l
fa_co_r_f_l
fa_fw_f_u_l
fa_fw_f_f_l
fa_co_f_u_l
fa_co_f_f_l
fa_pi_f_u_l
fa_pi_f_f_l
```
