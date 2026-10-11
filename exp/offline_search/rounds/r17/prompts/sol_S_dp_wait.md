<task>
Owner suspects workers mostly wait, so the bottleneck is on the server side. Coordinator probe at 14:06: 4090 SM 0%
over 6 s (68-111 W) during the DP baseline; timan107: 72 of 90 python worker processes <10% CPU, 10 >50%.
Running: DP baseline hs4_dp_b16w5_4090_k100_c5_control_20261010_133915 (4090, 1 replica B16/5ms, 32 workers on
timan107). The H100 Fast-WAM baseline was stopped (P3 is sweeping H100 configs exclusively now; do not probe the H100).
Using your recorder (PID 1965274) plus targeted read-only sampling (e.g. py-spy dump / /proc/<pid>/wchan / stack of a
sample of DP workers on timan107 and of the local DP server + driver processes, socket queue states), break down DP
worker wall time into: simulation/rendering, waiting for a policy response (server queue + compute + transport),
waiting for the driver (episode dispatch / result acknowledgment / recycling restarts), and startup/recycle overhead.
Same for the server: fraction of time with a batch in flight, batch sizes, idle gaps. Say plainly where DP is
bottlenecked and what would raise 4090 utilization (more workers, larger batch, more replicas/MPS, driver fix, transport).
Append findings to exp/r17_integration/REQUESTS/S_to_R.md.
</task>
<compact_output_contract>
End with: a time-breakdown table for DP workers and the DP server (percent of wall time), the bottleneck verdict with
evidence, and ranked fixes with expected speedup.
</compact_output_contract>
<action_safety>
Read-only; do not signal or reconfigure campaigns; keep tether calls >= 30 s apart; no H100 probes now. No pkill -f
bare patterns, no rm -rf, no /data/i018, /dev/shm/i018, codex credentials.
</action_safety>
