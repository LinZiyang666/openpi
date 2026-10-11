<task>
Urgent, the 4090 is idle. The DP cache full lane failed at 4001/8000 accepted:
lane /data/mw_finetune/eval/csl10dp (log /home/weiland/projects/openpi_ext/scratch/r17_ops/cache_lane_4090.log), campaign
/data/mw_finetune/eval/csl10d RECEIPT FAIL `RuntimeError('remote worker fleet failed: timan107; exit_code=1;
worker_restart_resource_timeout; worker_failures=[]')`; the lane's retry then refused: `campaign already accepted an
episode/attempt; use explicit resume` (csl10dp/csl10d_run.log). Context: timan107 CPU is oversubscribed (load ~88 on 48 cores)
with the H100 Fast-WAM cache lane (csl10f, 48 workers) also running there — do not touch it.
Do:
1. Find why a worker recycle restart could not get resources in time (RAM/VRAM admission of the replacement vs the
   exiting worker's not-yet-released memory? CPU starvation?) and fix it so recycling never needs fresh admission headroom
   beyond the slot it frees (reuse the slot; wait for the old PID to exit; longer bounded timeout), for all models. Tests.
2. Resume the DP lane yourself (you have the 4090; honor /tmp/r17_gpu.lock): run only the missing cells with identical
   settings via the explicit resume path, merging into one 8000-cell result per the lane design, and keep it running; make sure
   the lane tool continues with any remaining DP groups/suites after the resume. Report the resumed run name and progress.
3. Make cache_lane resume-aware (a failed job with accepted cells resumes instead of refusing) so the H100 lane (Fast-WAM then
   Cosmos) cannot stall the same way; tell the coordinator if the running H100 lane process needs anything.
</task>
<compact_output_contract>
End with: root cause, fix (files/tests), resumed DP run name + accepted count, and H100 lane safety.
</compact_output_contract>
<action_safety>
Do not stop/signal the H100 lane or its fleet. Kill only csl10d-owned leftovers (verify ownership). No pkill -f bare patterns,
no rm -rf, no /data/i018, /dev/shm/i018, codex credentials. timan1 forbidden.
</action_safety>
