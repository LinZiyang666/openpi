<task>
Owner: the GPUs are badly underused; raise utilization (more server replicas, larger batches, etc.). The coordinator
just stopped the H100 Fast-WAM baseline so the H100 is clean for you right now (no R17 process on it).

Live evidence (14:06, 48 timan107 workers, 2 replicas B8/5ms): H100 SM 95-99% but only ~200 W of 700 W; batch fill
p50 5 / p90 8; mean server latency ~2.9 s per call; 72 of 90 worker processes idle (<10% CPU) waiting. Your 4090 sweep:
native B16 11.07 dec/s vs B8 5.98 vs B1 1.10. Two replicas on one GPU only time-slice unless MPS lets them overlap.

Do, within <= 15 GPU minutes on the H100 (exclusive now), with realistic concurrency (48 closed-loop clients, real
LIBERO observations, 50 ms think time, K10/C5):
1. Measure throughput (calls/s, decisions/s), latency p50/p90, batch fill, SM%, power for: 1x B16, 1x B24, 1x B32,
   2x B16 with and without --mps, (3x B16 with --mps if memory allows), batch wait 5 vs 10 ms where it matters.
2. Pick the fastest config that keeps max action drift within what you already qualified (report drift vs B1).
3. Give the exact formal_eval.sh command for the new Fast-WAM official baseline (4 suites, --official-50, server h100,
   workers timan107: use 48 workers unless your numbers say more are needed; current VRAM admission on timan107 is
   the worker cap — DP's 32-worker fleet also lives there at 576 MiB each), and the matching arm settings.
Then stop; the coordinator launches. Append the table to exp/r17_integration/REQUESTS/P3_to_R.md for RUN_CONFIGS.
</task>
<compact_output_contract>
End with: table (config, calls/s, decisions/s, p50/p90 ms, fill p50/p90, SM%, W, max drift), the recommended config
with expected speedup over 2xB8 (~3.4 calls/s observed), and the exact launch command.
</compact_output_contract>
<action_safety>
H100 is yours for <= 15 GPU minutes; leave nothing running. Do not touch the DP campaign on the 4090/timan107.
No pkill -f bare patterns, no rm -rf, no /data/i018, /dev/shm/i018, codex credentials. timan1 forbidden.
</action_safety>
