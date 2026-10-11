<task>
Thanks for the MPS guard. The coordinator did NOT run your --continue-queue command: a coordinator retry queue
(/home/weiland/projects/openpi_ext/scratch/r17_ops/queue_hs6_fw_retry.run.sh) is already running the remaining Fast-WAM arms from
the prepared runs (hs6fw_c16 now, then c20, c32, k2, k1; on pre-episode failure it re-prepares as hs6fw_<arm>s<n> with
--replica-startup-width 1; it continues past rc=2 when all 1000 cells are accepted). Do not start another queue.
Two remaining issues:
1. Fast-WAM arm /data/mw_finetune/eval/hs6fw_c10 finished 1000/1000 accepted but driver returncode 2 with
   recovered_worker_shutdown=True -> RECEIPT PARTIAL (same class you recovered for hs6fw_res_152931). Recover it to PASS
   (preserve the original receipt), and fix the normal close path so "all planned cells accepted, retries resolved, only
   worker-shutdown acknowledgments missing and the fleet verified closed" yields PASS with the gap recorded, for every
   model; future Fast-WAM arms (48 workers) keep hitting this.
2. Evidence pulls at campaign end keep the GPU idle for minutes (both c15s0 and c10 spent ~4-5 min in chunked worker-tar
   pulls after their last episode). Release the GPU first (stop replicas/MPS, let the queue dispatch the next arm) and
   finish/verify evidence transfer asynchronously before sealing; sealing may complete after the next arm starts.
   Must stay safe for resume and pairing. Tests.
Append both pitfalls to exp/r17_integration/REQUESTS/Q_to_R.md.
</task>
<compact_output_contract>
End with: hs6fw_c10 receipt status, files/tests for both fixes, and whether the running coordinator queues pick them up
automatically for later arms (or what to change).
</compact_output_contract>
<action_safety>
CPU only; do not stop/signal running campaigns or queues. No pkill -f bare patterns, no rm -rf, no /data/i018,
/dev/shm/i018, codex credentials. timan1 forbidden.
</action_safety>
