<task>
Follow-up from the coordinator. Owner rule (2026-10-16): workers do not care which machine they run on; every worker node (timan107, timan108, timan1 GPU 2, local) can run any workload. A single worker machine dying must never block or abort our experiments again.

What happened today:
- 07:36 timan1's tether agent went offline. ff1015_cosmos_li_03 still finished its plan, but every queued batch that listed timan1 in its worker topology could not start, and the H100 sat idle for hours until I re-distributed them by hand (resume --worker-counts).
- 09:11 timan108 rebooted uncleanly. ffk1015_dp_mt50_reduced_descent (90,000 planned, timan107x40 + timan108x40) ended rc=2, receipt FAIL, at 1,059 accepted, instead of continuing on timan107; I had to build a continuation.

Make the launch path used by ff/ffk runs (filter_failed/launch.py → the standard prepared launcher, reused by filter_failed_knob) robust to worker-node loss, with the smallest change:
1. At launch (and stage, if it reserves nodes), probe each worker node in execution_worker_nodes (tether node status / a cheap exec). Any node that is offline or fails admission is dropped and its share is moved to the online nodes in proportion to their measured capacity (per-node caps you set in the timan1 work and the current reservations), recorded in the campaign the same way resume --worker-counts records it. Launch proceeds as long as at least one worker node is usable.
2. During a run, if a worker node dies (agent offline, fleet gone, restart budget exhausted on that node only), the campaign must keep running on the remaining nodes: re-dispatch that node's in-flight episodes (attempt+1, as the driver already does for failed attempts), stop counting the dead fleet, optionally top up workers on the other nodes within their caps, and only fail when no worker node is left. Find out exactly why today's DP MT50 run ended instead of continuing (receipt, launcher/supervisor logs) and fix that path.
3. The ff/ffk queue scripts (openpi_ext/scratch/r17_ops/ff/chain_ffk.run.sh, chain_ff2.run.sh, chain_r15.run.sh) must not be edited while running; if the queue needs a change, give me a new script and I will swap it in between items.

Keep results unchanged: same plans, seeds, servers, R/B/W; only which machines run workers changes. Verify on CPU with the frozen worker modules (simulated node loss) and give one targeted test.
</task>

<rules>
- CPU only locally; tether exec read-only checks only; never push/pull/expose; never start or kill workers, servers, GPUs or chains; no pkill, rm, git commits; never touch credentials.
- Do not modify, rebuild or re-stage running, prepared or launched runs. Another sol thread is building LIBERO test-set descent campaigns in rounds/r17/filter_failed_knob — don't touch its new files.
</rules>

<output>
Final message: root cause of the DP MT50 abort, what changed, how launch-time and in-run node loss now behave, test results, and anything I must apply.
</output>
