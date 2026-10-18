<task>
Coordinator, 2026-10-16 16:3x. Owner ruling, now the formal design: the per-task IR knob is never tuned on a development set (the cache libraries are built from the development pool). It is descended directly on the test set. MetaWorld already did this (seed 7, `rounds/r17/knob/mt50`, `exp.offline_search.rounds.r17.knob.descent`); all LIBERO knob results chose levels on a development set and are withdrawn. A parallel job is moving their outcome data (journals, per_step, results, analyses) into a trash directory; campaign definitions, runtime/worker archives, assets and level/calibration artifacts stay in place. Do not read old outcome data and do not reuse dev-chosen levels.

Build the LIBERO IR-knob experiments again under the formal design, for BOTH arms — the original library and the success-filtered library (failed episodes removed, no top-up, no refit, exactly as `rounds/r17/filter_failed_knob`) — for every model and step mode:
- DP, Fast-WAM, Cosmos × full and reduced steps × library sizes 50/100/200/300/400/500 × libero_10 and libero_spatial;
- π0.5 and GR00T × full and reduced steps × their library sizes (50/100/200/500) × both suites.

Design (same as MetaWorld): for every cell and task, run every non-pure level of that model/mode's existing knob ladder on test set A (initial states 0–49), 50 episodes per task per level. Selection per task: walk from the most expensive level downwards; a level passes when it has at most 2 fewer successes than same-step-count pure inference on the same GPU and initial states; the first failing level freezes the previous level (pure inference if the first level fails). Cell rule: if the complete system with the knob off (stall gate only) already reaches same-step pure inference for the whole cell, do not tune it; report that knob-off result. The reference is always same-step pure inference (reduced vs reduced, full vs full).

Reuse, don't rerun:
- existing test-set knob-off runs (complete system, stall gate only) of both arms where they exist: fsred* (original, reduced), ffk1015_*_libero_reduced_off* (filtered), rounds/r16/k1ref/runs/off and rounds/r17/filter_failed_knob/vla/runs/off (π0.5/GR00T reduced). Full-step knob-off does not exist: add it as a level in the grid.
- existing same-step pure-inference test-set references. Run each model/mode's grid on the GPU and serving configuration (replicas/batch/MPS) of its original official-A test so the reference matches; workers may run on any node (owner: the environment's machine does not matter) — use timan107, timan108 and local, never timan1.

Deliver:
- campaign definitions for both arms (episodes per campaign, GPU, R/B/W, workers), with an exact per-model/mode episode total and GPU-hour estimate from the source tests' measured throughput;
- CPU checks: startup identity against the source A campaigns, planner/plan identity, and for the original arm that the serving tables equal the source's;
- one analysis entry point that applies the selection above per cell for both arms and writes per-task levels, SR, measured IR, paired filtered−original differences by initial state;
- exact prepare/launch/analyze commands and a recommended order per GPU (H100 and RTX 4090);
- a short section in rounds/r17/filter_failed_knob/HANDBACK.md and rounds/r17/knob/README.md stating this design.
</task>

<rules>
- CPU only: no servers, GPU, workers, chains, git commits, pkill, rm/rm -rf. Never touch credentials.
- Do not modify, rebuild or re-stage any running or queued run (check `ps -eo pid,args | grep "[c]hain_"` and openpi_ext/scratch/r17_ops/ff/run_*.sh). Do not move or delete data; the other job does that.
- tether: exec read-only checks only; never push/pull/expose.
- Reuse existing builders (filter_failed_knob/build.py, workflow.py, the MT50 knob descent) — smallest new code; one targeted test.
</rules>

<output>
Final message: what was built, episode totals and GPU-hours per model/mode and arm, CPU check results, commands, recommended GPU order, and anything needing a decision.
</output>
