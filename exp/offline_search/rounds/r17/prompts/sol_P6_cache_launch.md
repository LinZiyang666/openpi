<task>
All three model admissions for the cache study are CPU-complete: P4 (DP, HANDBACK_P4.md:1205), D (Cosmos,
HANDBACK_COSMOS.md:955), P3 (Fast-WAM, HANDBACK_P3.md:975; P3 prepared a matching Fast-WAM smoke run
/data/mw_finetune/eval/cfwp31010 and reports that your smoke.sh helper defaults to --worker-memory-mib 6144 /
--worker-vram-mib 1024, which do not match the baselines and do not fit timan107's VRAM admission).
Do (CPU only):
1. Fix exp/r17_integration/ops/cache_study/smoke.sh so each model uses exactly the server/worker configuration of its
   pure-inference control (see exp/r17_integration/ops/RUN_CONFIGS.md): Fast-WAM h100 2xB16/5ms --mps group-commit,
   48 workers 2048/768 MiB, recycle 4; DP 4090 2xB8/5ms --mps --replica-startup-width 1 group-commit, 32 workers
   2048/576 MiB; Cosmos h100 2 replicas B1, workers per RUN_CONFIGS (fit timan107 VRAM admission next to the other fleet).
   Include Q's MPS guard/retry, rc2 ACK recovery and GPU-first async finalize (they are in ops/ now).
2. Produce the full-run plan: per model ONE resident server that hot-switches through its 8 YAMLs (Cosmos: 4 distinct),
   each group 500 episodes per suite on official A (libero_spatial, libero_10), with a queue/driver script the coordinator
   can launch per GPU lane (h100: Fast-WAM then Cosmos; 4090: DP), prestaged, short run names, resume-safe, pinned to
   the controls (hs5dp_ctl_142812, hs6fw_res_152931 + runtime hs6fw_ctl_143332, Cosmos merged K5). Long-horizon L is
   provisional C10 until astra's exp/offline_search/rounds/r17/horizon/RESULTS_DP_FASTWAM.md freezes it — make L a
   single parameter per model so the coordinator can set it at launch.
3. Give exact commands: smoke per model, then full run per lane, with expected durations.
GPUs are busy until ~17:45 (4090) / ~18:15 (h100); do not launch.
</task>
<compact_output_contract>
End with: smoke commands (3), full-run commands per lane, how to set L, expected minutes, and tests run.
</compact_output_contract>
<action_safety>
CPU only; no launches, no signals to running campaigns. No pkill -f bare patterns, no rm -rf, no /data/i018,
/dev/shm/i018, codex credentials. timan1 forbidden.
</action_safety>
