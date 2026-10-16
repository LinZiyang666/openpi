<task>
Coordinator, 2026-10-16 08:4x. timan1's tether agent is OFFLINE (owner must restart it). Every remaining H100 job (ff1015_cosmos_li_04 and all ffk1015 H100 lanes) lists timan1 in its worker topology, so the H100 is idle.

Make the existing worker-count knob you added for continuations (`filter_failed.resume --worker-counts`) usable for a fresh, never-launched ffk1015 campaign, so I can run `ffk1015_dp_mt50_reduced_descent` and `ffk1015_fastwam_mt50_reduced_descent` with timan107×40 + timan108×40 only (no timan1). Same server, R/B/W, filtered rows, plans and identities; only execution worker nodes/counts change (MT50 workers carry no rendered-image dependence on worker GPU type? verify from the source campaigns and say so). Record the change in the campaign the same way the continuation does, so analysis and receipts show it.

Deliver: the exact command(s) to apply it to those two runs (do not apply them yourself), the prepare/launch commands, and a CPU check (startup identity + planner) for both. If the knob cannot be applied to fresh campaigns without bypassing a sealed check, stop and report instead.
</task>

<rules>
- CPU only: no servers, GPU, workers, chains, git commits, pkill, rm -rf. Never touch credentials.
- Do not modify, rebuild or re-stage any launched run; do not run filter_failed.build or filter_failed_knob build for launched runs.
- tether: exec only for read-only checks; never push/pull/expose.
- Smallest change; reuse resume.py.
</rules>

<output>
Final message: what changed, the exact apply/prepare/launch commands, CPU check results.
</output>
