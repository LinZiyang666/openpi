<task>
Coordinator, 2026-10-16 20:3x. First H100 launch through the new launch path failed for a port reason, not a resilience reason:
`ff1015_cosmos_li_04` (last pure-Cache batch, H100, 3,066 episodes) started 20:23:57 and ended 20:26:40 rc=2, receipt PARTIAL, all_owned_closed=true, with "ports did not release within 120s: [23159]". Port 23159 is the local driver port of `ffk1015_pi05_mt50_full_descent_04`, which is running on the RTX 4090 lane (PID 145473 listens on 0.0.0.0:23159) until about 09:00 tomorrow; cosmos_li_04's campaign also uses driver_port 23159. Both drivers run on this host. `ffk1015_pi05_mt50_reduced_descent` (queued on the 4090) uses 23159 too, and the R16 VLA runner uses `--port 23159`.

1. Bring cosmos_li_04 back to a launchable state with the existing reset path (filter_failed/reset.py; rename only, never delete) and give it a local driver port that no 4090-lane campaign uses (the Cosmos knob allocation 23112 is used only by H100 Cosmos lanes, which run serially; or pick one from the Cosmos range 23240–23249). Keep everything else (plan, servers, worker counts timan107=55,timan108=60,timan1=30) unchanged. Give me the prepare/launch commands.
2. Audit every queued campaign for host-local port collisions between the two GPU lanes, including the deferred/legacy formal_A items that have no campaign.json yet (π0.5/GR00T R15/R16 runners: their local driver, sync and server ports). Queues: openpi_ext/scratch/r17_ops/ff/run_ffk_h100t.sh (H100) and run_ffk_4090x.sh (4090), plus the running pi05_mt50_full_descent_04. Any H100-lane item whose locally bound ports intersect a 4090-lane item's locally bound ports must get a disjoint port before it is prepared; fix definitions that are not prepared yet, and tell me which ones you changed.
3. Smallest guard in the shared launch path: before staging, check the campaign's locally bound ports are free and, if one is held by another live campaign, fail fast at prepare (so the queue skips and logs it) rather than burning a launch.
</task>

<rules>
- CPU only; no servers, GPU, workers, chains, pkill, rm/rm -rf, git commits; never touch credentials; tether exec read-only only.
- Do not modify running runs or the running chain scripts. Prepared/never-launched definitions may be fixed.
</rules>

<output>
Final message: cosmos_li_04 new port and exact prepare/launch commands, the audit table (item, lane, local ports, changed?), and the guard change.
</output>
