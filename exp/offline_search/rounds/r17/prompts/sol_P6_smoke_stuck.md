<task>
Urgent: the DP cache smoke you staged is stuck and the 4090 is idle.
Command run by the coordinator at 18:10: `bash exp/r17_integration/ops/cache_study/smoke.sh dp --run --run-name csmd1010`
(log /home/weiland/projects/openpi_ext/scratch/r17_ops/cache_smoke_dp.log; campaign /data/mw_finetune/eval/csmd1010).
State at 18:22: local replica_0 ready at 18:11:29 (servers/replica_0/ready.json; only one replica spawned so far,
--replica-startup-width 1), launcher has only been issuing periodic `tether exec` probes since 18:11:15 (PHASE_TIMINGS),
RECEIPT PARTIAL, no driver/journal. On timan107: /tmp/r17_formal/csmd1010/timan107/workers.process.json shows the
worker supervisor spawned `exp.r17_integration.ops.formal_workers --campaign /tmp/r17_formal/csmd1010/timan107/campaign.json`
(pid 1900325), but that process no longer exists, there are 0 worker_entry processes, and workers.log and
workers.supervisor.log are empty. RAM admission passed. timan107 also hosts the live Fast-WAM arm fleet (48 workers,
hs6fw_k2s1) — do not touch it.
Do:
1. Find why the worker fleet died silently and why the launcher did not fail fast (it should have).
2. Fix it (CPU tests), stop the stuck csmd1010 campaign cleanly by its own PIDs if it is still waiting, and relaunch the DP
   smoke under a new short name yourself (you have the 4090 for it: honor /tmp/r17_gpu.lock), verify the 32 smoke
   episodes and the hot-switch/accounting checks, then stop. Report the result; the coordinator's chain
   (/home/weiland/projects/openpi_ext/scratch/r17_ops/chain_cache_4090.sh) will exit on the failed smoke, and the coordinator will
   launch the full lane afterwards.
3. Check whether the same failure would hit the Fast-WAM/Cosmos smokes queued on h100 and fix it there too.
</task>
<compact_output_contract>
End with: root cause (evidence), fix (files/tests), new smoke run name + result (episodes, hit/MISS/NFE checks, hot-switch),
and whether the h100 smokes are safe.
</compact_output_contract>
<action_safety>
Kill only csmd1010-owned PIDs (verify ownership first). Do not touch hs6fw_* campaigns or other users' processes. No pkill -f
bare patterns, no rm -rf, no /data/i018, /dev/shm/i018, codex credentials. timan1 forbidden.
</action_safety>
