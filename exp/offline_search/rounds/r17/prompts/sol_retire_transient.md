<task>
Coordinator, 2026-10-17 18:2x. URGENT. One 10-second control-probe timeout took two campaigns' whole remote worker fleets out. CPU only.

Evidence:
- At 18:19:30–31, `Running.check` in `filter_failed/worker_resilience.py` retired timan107, then timan1, in two campaigns at once.
- For `fa_fw_r_u_l`, `RECEIPT.json` `worker_node_losses.timan107.reason` is `{"control_probe_failed": true, "error": "RuntimeError('command failed (70): tether exec --safe --timeout 10s timan107 -- env\nerror: exec: timed out after 10s waiting for chunk\n')", "status": "FAIL"}`.
- `worker_fleet_failed` treated that as fleet death and retired the node. The driver then shut the node's workers down; at 18:23, timan107's eight GPUs held 2 MiB each.
- At that time timan107 had a stalled network mount. A later tether reply said `fork/exec stalled (likely a hung network mount ...) (remote_fs_spawn_timeout)`. A `--safe --timeout 60s` call at 18:23 worked, and both nodes are ONLINE.
- `fa_fw_r_u_l` (original Fast-WAM LIBERO reduced levels, 4090 server, 30,000 planned) had already lost `local`, which I retired at 15:25 for the OOM. It ended 18:20:56 with rc=1, FAIL, `RuntimeError('no usable worker node remains')`, `all_owned_closed=false`, and 16,816 accepted rows, 6 of them the known infra errors.
- `fa_co_f_f_o` (filtered Cosmos LIBERO full knob-off, H100 server, 6,000 planned) is still running, but only on local×7: it lost timan107 and timan1 the same way. It had 2,484 accepted at 18:21.

1. Fix retirement. A failed or timed-out control/status probe is "unknown", never death. Retire a node only when tether reports it OFFLINE, or when its probes keep failing for at least 5 minutes, or when a successful probe proves the fleet closed or failed. Give unknown probes a longer timeout, as for the capacity probe. Workers that are still alive must keep running. Make the smallest change and add a test that replays the 18:19 reason. New launches pick it up; do not modify running runs.
2. `fa_co_f_f_o`:
   - give me the safe way to stop it now: exact PIDs and order, using the existing cleanup/closure path; I execute;
   - then build a continuation `fa_co_f_f_oc1` that holds every planned UID not yet accepted, using the same native continuation tool as `ffk1015_dp_mt50_reduced_descentc1` / `fa_fw_r_u_lc1`, with worker counts timan1=44,timan107=34 (local as the allocator allows), and register it in `CONTINUATIONS.json`.
3. `fa_fw_r_u_l`:
   - prove closure (owned processes on local, timan1, timan107 and the 4090 servers);
   - replace the 6-episode `fa_fw_r_u_lc1` draft (rename it aside, never delete) with a continuation that holds every unaccepted planned UID plus the 6 infra-error UIDs, with worker counts timan1=30,timan107=40 (local 0 under the new reservation);
   - keep `CONTINUATIONS.json` and the formal_A loader consistent.
4. Give me the prepare/launch commands for both continuations. Do not start them.
</task>

<rules>
- CPU only locally. tether exec only for read-only checks. Never push, pull or expose.
- Never start or kill servers, workers, drivers, chains or GPUs; give me the commands.
- Do not modify `fa_co_f_f_o` while it runs, or the running chains (ffk-h100a, ffk-4090d, h100_disk_guard2).
- No pkill, rm or git commits. Never touch credentials.
</rules>

<output>
Final message: the retirement change and test, the fa_co_f_f_o stop procedure, both continuation builds (UID counts), and the commands.
</output>
