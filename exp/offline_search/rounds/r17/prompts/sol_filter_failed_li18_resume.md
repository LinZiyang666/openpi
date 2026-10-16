<task>
Coordinator, 2026-10-16 06:2x. `ff1015_fastwam_li_18` (filtered Cache, Fast-WAM LIBERO, H100 server, timan1×40 + timan107×45) stopped at 06:17 with 3,787 of 5,000 planned episodes accepted (0 errored): the timan1 fleet exhausted its worker restart budget because recycled LIBERO renderers on timan1 GPU3 could not create EGL framebuffers ("Offscreen framebuffer is not complete", GPU3 shared with another user's ~15 GB process). Receipt FAIL `worker_restart_budget_exhausted`, all_owned_closed=true.

Build the continuation for the missing 1,213 episodes with the existing R17 resume mechanism (the one used for earlier resumed lanes, e.g. cache_resume/resume_from), so that analysis reads both segments as one lane without double counting. Keep the same filtered candidate rows, serving config, seeds and identities. Worker topology: same nodes, but if the native admission on timan1 GPU3 cannot fit 40 LIBERO renderers, report the alternatives (fewer timan1 workers through an existing knob, or timan107 only) and their effect on comparability — do not pick one silently.

Also make filter_failed.analyze (and filter_failed_knob analysis if it reads ff lanes) merge the continuation segment.

Context you need: since 2026-10-15 23:00 I launch ff/ffk jobs through `filter_failed/launch.py`, which rewrites the remote worker command (`worker_command`/`SHARE`): it keeps the module name in argv, lifts the frozen "other user GPU allocation" gate, and lets timan1 LIBERO cache fleets use the current 40-worker lease (filtered campaigns lost cache_capacity_authorization). It no longer overrides the frozen per-worker VRAM admission. `filter_failed/reset.py` now also renames remote worker stage/run files and the H100 server stage and run-dir copies (MPS_READY.json, mps, mps_control, servers). Read both before writing anything.

Deliver exact prepare/launch commands for the continuation and a CPU check that the merged analysis counts 5,000 unique task/init outcomes.
</task>

<rules>
- CPU only: no servers, GPU, workers, chains, git commits, pkill, rm -rf. Never touch credentials.
- Do not modify, rebuild or re-stage any running or completed ff1015_*/ffk1015_* run other than creating the continuation; do not run filter_failed.build.
- tether: exec only for read-only checks; never push/pull/expose.
- Keep it simple: reuse the existing resume path; smallest new code.
</rules>

<output>
Final message: continuation run name, episode count, worker topology (and any alternative needing my decision), exact commands, CPU check result.
</output>
