<task>
sol S's live profile (exp/r17_integration/REQUESTS/S_to_R.md, /data/mw_finetune/perf/2026-10-10/summary.md, recorder
exp/r17_integration/ops/perf_recorder.py PID 1965274 still sampling) found the top throughput limiter is the formal
conductor driver itself: filesystem commit (fsync) waits in 5/5 driver snapshots and 19-26 s median DP
reporting/dispatch delays, so batches stay underfilled (DP p50 batch 1 of cap 16; GPU 53%). Estimated gain from fixing:
DP 1.45-2.2x, Fast-WAM 1.2-1.55x.

Do:
1. Find the exact hot path (per-event durable writes, fsync, journal/attempt/dispatch/per_step appends, scheduler lock
   held across I/O, tether calls in the dispatch loop). Fix it: group commit (e.g. fsync at most every ~1 s or N events,
   and always at episode acceptance boundaries needed for resume correctness), move I/O out of the scheduler lock,
   never block dispatch on evidence writes. Crash safety for --resume-from must hold: an accepted episode must be
   durable before it is counted; state the exact guarantee.
2. This is orchestration only (no change to observations, actions, seeds, model code). The running studies pin
   driver code via --runtime-from / --paired-control (your frozen-runtime work). Make the driver an explicitly
   separate, recorded "orchestration pin" so arms can use the faster driver against baselines sealed with the old
   driver, while server/worker/model code stays strictly pinned. Record both driver versions in receipts. Tests.
3. Verify on CPU with a realistic replay/load test (many concurrent fake workers, real journal writes on /data) that
   dispatch latency drops; give before/after numbers.
4. Update the prepared H100 queue (/data/mw_finetune/eval/formal_queue_hs4_fast_frozen, jobs
   /home/weiland/projects/openpi_ext/scratch/r17_ops/queue_hs4_fast.json) and the DP arm queue script
   /home/weiland/projects/openpi_ext/scratch/r17_ops/queue_hs4_dp.sh so the arms use the new driver; re-run prepare-only.
Running now (do not touch): hs4_fast_b8w5_h100_k10_c5_control_20261010_131810 (h100, timan107 x48) and the DP baseline
hs4_dp_b16w5_4090_k100_c5_control_<13:39 stamp> (4090, timan107 x32). Others editing: P3 (MPS in serve/launcher),
D (Cosmos merge/closure). Check diffs, keep edits narrow.
</task>
<compact_output_contract>
End with: hot path found (file:line), fix summary, durability guarantee, before/after dispatch latency, files/tests,
and the prepare-only result for both queues.
</compact_output_contract>
<action_safety>
CPU only; do not launch, stop or signal campaigns or the recorder. No pkill -f bare patterns, no rm -rf, no /data/i018,
/dev/shm/i018, codex credentials. timan1 forbidden.
</action_safety>
