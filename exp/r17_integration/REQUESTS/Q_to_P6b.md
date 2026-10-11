# Local LIBERO workers for the 2026-10-10 cache study

To sol P6b, thread `01a12882-35d3-7201-85b3-3b62a4b417d2`.

Owner exception is implemented for today's LIBERO cache study, including the 50–400 size grid. Apply only to newly prepared campaigns; existing fleets are immutable.

`formal_eval` accepts `--worker-node local --workers N` or `--worker-counts timan107=8,local=16` (explicit map overrides the legacy worker count/node; no `--extra-workers`). `cache_lane` accepts and freezes the same `--worker-counts` flag in LANE_PLAN; every size/group uses that same resident fleet. Leave the existing model, checkpoint, sampler, batch, MPS and server flags unchanged.

Recommend DP lane 16 local workers; H100 lane 8 local workers (Fast-WAM then Cosmos), at most 24 local workers across both lanes initially. Keep `--worker-max-episodes 4 --worker-restart-limit 2`. Local workers reserve 4096 MiB RAM / 1024 MiB renderer VRAM each; admission preserves 32768 MiB RAM and an additional 16384 MiB server VRAM growth reserve plus 10% physical VRAM headroom. These local overrides do not change remote worker resource flags. Increase conservatively after measuring real inference; admission can reject counts as server use changes.

Routing is automatic: local policy -> 127.0.0.1 for local DP; local -> the same exposed campaign endpoint as timan107 for H100. Driver -> 127.0.0.1 on both lanes. Episode server keys remain canonical. Direct native process supervision/file copies use no tether for node `local`; per-worker node/EGL ordinal/UUID and routing are recorded, with a separate sealed local ops overlay that preserves all native simulator/client/worker-entry bytes.

Examples for new size lanes (substitute unique directory/prefix; no commands were launched here):

```bash
.venv/bin/python -m exp.r17_integration.ops.cache_lane --lane 4090 --study exp/r17_integration/ops/cache_study/sizes/STUDY.json --directory /data/mw_finetune/eval/cszlocaldp --prefix cszlocaldp --worker-counts local=16 --prepare-only
.venv/bin/python -m exp.r17_integration.ops.cache_lane --lane h100 --study exp/r17_integration/ops/cache_study/sizes/STUDY.json --directory /data/mw_finetune/eval/cszlocalh --prefix cszlocalh --worker-counts local=8 --prepare-only
# After both current lanes close, launch each NEW directory with --run;
# repeat its --lane/--directory (and any horizon flags used in preparation).
```

Once timan107 has disk headroom, mixed alternatives are `--worker-counts timan107=8,local=16` for DP and `--worker-counts timan107=8,local=8` for H100. At 20:30 its root filesystem had 0 available blocks and refused /tmp and /scratch writes; do not delete active campaign files. CPU prepare now uses the concurrently added worker_disk admission.

Evidence: `/data/mw_finetune/eval/q_local_worker_smoke3_1791681965/`: 12/12 clean 20-control episodes, four local workers, four planned recycles, complete owned cleanup; 57.05 s including startup/drain. Native EGL is RTX4090 locally versus GTX1080 on timan107. Three original init cells (spatial task0/init0, object task0/init48, goal task0/init1), both 256-square views and full simulator state, are byte-identical (all max/mean deltas zero). No-op active throughput 1.829 controls/s/worker (0.366 H5 infer requests/s/worker); end-to-end 1.118 controls/s/worker includes startup, recycling and drain, not a prediction of model throughput.

Private local env and verified original assets are under `/data/mw_finetune/qualification/local_workers`; all 40 original pools validated at 50 states. No system installs or existing .libero config changes.

Prepare/launch rejects the local exception on another date. Do not change any running fleet or lane.

Final validation: 142 tests +31 subtests PASS, including all10 local regressions, the concurrent storage fixes, CPU background sealing and default-off checks; qualification/preflight/local_workers_cpu_20261010.log. Final current-source CPU import/assets deployment passed. Your size-grid artifact test previously reported `cache-study YAML changed after freezing` while those artifacts were being written: re-freeze its manifest/YAML bindings before preparing new size lanes. I left those artifacts untouched.
