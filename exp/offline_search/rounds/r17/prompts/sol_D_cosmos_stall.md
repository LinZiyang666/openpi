<task>
The Cosmos official baseline stalled. Run dir (local): /data/mw_finetune/eval/formal_cosmos_libero_official2000_20261010_100533_5f558380
(server h100, 2 replicas, endpoint linziyang.top:14008; 24 workers on timan108, remote dir
/tmp/r17_formal/formal_cosmos_libero_official2000_20261010_100533_5f558380). 1726/2000 accepted; the last
journal and dispatch lines are at 11:30:53 and nothing since (checked 11:38). At 11:38: the h100 formal_server
PID 188534 shows 0% CPU, the 24 timan108 worker_entry processes are alive but idle; one (PID 2338079) is in
D state (wchan rwsem_down_write_slowpath). The broker port linziyang.top:14008 accepts TCP.
Possibly related: at about 11:30-11:31 the coordinator stopped a DP formal campaign
(hs2_dp_k100_c5_default_20261010_111846, local 4090 server, 20 workers on timan108) with SIGTERM; its
shutdown reported all_owned_closed=true. Check whether that shutdown touched anything the Cosmos campaign
owns (tunnels, ports, workers, leases, shared install dirs). Note timan108 GPUs 1-2 are used by another user
(xy51); do not touch them.

Do:
1. Find exactly where the Cosmos pipeline is blocked (driver, proxy, replicas, tunnel, workers) with evidence.
2. Recover so the remaining 274 episodes finish in this same campaign if the framework allows it safely
   (e.g. restart the stuck component, let the driver re-dispatch). If in-place recovery is unsafe, stop the
   campaign cleanly and use or implement resume (only missing cells, identical config, one merged receipt);
   Q is implementing `--resume-from` for Fast-WAM in exp/r17_integration/ops/ right now — check its diff and
   coordinate via exp/r17_integration/REQUESTS/D_to_Q.md rather than editing the same lines.
3. If the DP shutdown caused it, say which code path, and fix it (CPU tests) so stopping one campaign can never
   affect another.
A queued job (/home/weiland/projects/openpi_ext/scratch/r17_ops/queue_h100_cosmos_k2.sh) starts the Cosmos K2
arm on the same ports 23240/23249 once this baseline's RECEIPT.json leaves PARTIAL; account for that.
</task>
<compact_output_contract>
End with: blocked component + evidence (paths), what you did, current accepted count, whether the baseline
will finish as one campaign, and any fix (files, tests).
</compact_output_contract>
<action_safety>
Do not touch the Fast-WAM campaign hs2_fast_h100_k10_c10_* (h100 + timan107) or P3/P4 GPU work on the 4090.
Kill only PIDs you verified belong to the Cosmos campaign; no pkill -f bare patterns; no rm -rf. Do not
touch other users' processes, /data/i018, /dev/shm/i018, or codex credentials. timan1 is forbidden.
</action_safety>
