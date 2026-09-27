# R3 H2 — mixed HIT/MISS infrastructure

The code is the closed-loop plugin / ops (`exp/offline_search/closed_loop/`: `plugin.py --os-judge ...`, `selftest.py`,
`verify_logs.py`, `replay_client.py`, `probe.py:ProbeForce`, `ops/emit_arms.py` (`full_model`), `ops/chain.sh`,
`ops/collect.py`); see `closed_loop/README.md` § "Mixed HIT/MISS mode". This directory only holds the control-arm specs.

## `arms_mx_controls.json` (emit_arms spec rows, `<RUN>` = the coordinator's run root)

| arm | judge | method | note |
|---|---|---|---|
| `r3mx_p_sp_b0h70` / `r3mx_p_l10_b0h70` | `quantile:0.7:1000:<tau0>` | `B0Current` (fused score = confidence) | MX-B0-h70; tau0 = B0's one-threshold mixture value at h = .7 from `rounds/r03/diag_B/judge_tables.md` ("Target hit rate 70%: ONE threshold", B0 own): .974 (sp) / .993 (l10); the controller holds h = .7 after W/10 = 100 decisions |
| `r3mx_p_sp_perk3` / `r3mx_p_l10_perk3` | `periodic:3` | `AWM {"lib":"current","kref":5}` | MX-PER-k3: MISS every third decision, h = 2/3 (IR ≈ .435), confidence ignored |

All four carry `"full_model": true` (chain.sh starts their servers with STAGE1_ONLY=0, NEED_MB 9000 each; π0.5 ≈ 7.6 GB
per server, 4 servers ≈ 30 GB) and `--os-fit-artifact <RUN>/fits/<name>.pkl`.

Prefit (same recipe as R2; the AWM fit is the R2 CL2 fit, `spec` / `kwargs` / `cell` identical so the R2 pickle can be
copied to `<RUN>/fits/r3mx_p_<sp|l10>_perk3.pkl` instead of refitting):
```bash
taskset -c <cpus> .venv/bin/python -m exp.offline_search.closed_loop.plugin --os-method exp/offline_search/rounds/r02/g1_awm/awm.py:AWM \
    --os-kwargs '{"lib": "current", "kref": 5}' --os-cell pi05_spatial_cache --os-log-dir <RUN>/fits --os-fit-artifact <RUN>/fits/r3mx_p_sp_perk3.pkl
```
R2 measurements (`r02_g50/fits/*.prefit.log`): AWM cur kref5 pickle 21.3 MB (spatial) / 24.6 MB (l10), fit 0.40 / 0.38 s
with the cached current-library PCA (`derived/r02/g1_awm/pca_current`; 5–60 s without it); the artifact loads in ≈ 0.4 s. B0Current's fit is instantaneous (pickle a few KB).

Smoke of this infrastructure (2026-09-27): `closed_loop/README.md` and the H2 hand-back; smoke run root
`/home/weiland/trace_runs/os_closed_loop/r03_smoke` (arms `r3s_*`).
