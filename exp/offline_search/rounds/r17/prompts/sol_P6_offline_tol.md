<task>
Tolerate transient tether node-offline blips in long campaigns. Tonight the timan108 tether agent showed "OFFLINE (no recent heartbeat) … (node_offline: status=STALE)" twice, around 01:36 and 02:33. Each time it recovered by itself within a minute or two; the agent process never restarted. Each blip aborted a whole library campaign, because a single `tether exec --safe --timeout 10s timan108 -- env …` lease/liveness probe failed:
- `/data/r17_library/cosmos_libero_libero_10/campaigns/teacher_cosmos_libero_libero_10_500_20261010_005553_3fa10df3` (salvaged + repair pending)
- `/data/r17_library/fastwam_libero_libero_spatial/campaigns/teacher_fastwam_libero_libero_spatial_500_20261010_020305_a90d5a54` (about 200/500; orphans killed by coordinator)
The workers themselves are independent processes and keep running through such a blip.

Fix in the formal/library launcher (`ops/formal_eval.py` remote/lease paths, `ops/library_staging.py` transfers, worker close/snapshot): treat `node_offline` / STALE / transient tether exec failures as retryable with backoff for a bounded window (e.g. up to 5 minutes). Only fail when the node stays offline past the window, or the remote command returns a real failure. Log each blip in the receipt.
Also, when a campaign does fail, run the remote cleanup through the same retry so orphans do not survive (all_owned_closed should be true once the node is back).
CPU tests. Do not start GPU work; do not touch running processes.
Then give exact commands:
(1) salvage + repair for the Fast-WAM spatial campaign above;
(2) Fast-WAM libero_10 full;
both H100 + workers on timan108, or timan107 once DP libero_10 there finishes. Say whether `--worker-node timan107` is valid for the H100 server.
</task>
<compact_output_contract>Append "## Transient node-offline tolerance" to HANDBACK_P6.md. Final message: 4 lines.</compact_output_contract>
