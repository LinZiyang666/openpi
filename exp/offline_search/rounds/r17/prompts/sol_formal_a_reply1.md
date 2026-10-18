<task>
Coordinator review of formal_a_1016: design accepted. Two changes, then make it queue-ready.

1. π0.5 full reference: authorized. Build the two library-independent 500-episode full-step pure-A jobs on the H100 serving configuration of the π0.5 full grid (one per suite), shared by both arms, and register them as that model/mode's reference. Release π0.5 full from hold.

2. Do not dispatch levels the formal rule will never use. The cell rule says: if the complete system with the knob off already reaches the same-step pure SR for the whole cell, the cell is not tuned. So:
   - Reduced modes (knob-off already measured on test A for both arms): evaluate the rule now from the reused off results and the pure references; drop every qualifying cell's non-off levels from the grids (per arm). Only non-qualifying cells keep their level grid.
   - Full modes (no knob-off yet): split each grid into an off-only campaign that runs first, and a second campaign whose cells are filled in after the off result is analysed, with only the non-qualifying cells. Provide the command that builds the second campaign from the analysed off results.
   - Keep the analysis/selection unchanged; report the new episode totals per model/mode/arm.

3. Make it queue-ready for my chain runner: every launch unit must have commands/prepare_<NAME>.sh and commands/launch_<NAME>.sh (like rounds/r17/filter_failed_knob/commands), where NAME is the campaign name, plus an ordered list per GPU (H100, RTX 4090). Give worker counts that use timan1 GPU 2 as well (timan1_gpu2 is merged: LIBERO cap 83) next to timan107/timan108/local, at least two worker machines per campaign where the source topology allows it; worker_resilience redistributes at launch anyway.
</task>

<rules>
- CPU only: no servers, GPU, workers, chains, git commits, pkill, rm/rm -rf. Never touch credentials. tether: exec read-only only.
- Do not modify, rebuild or re-stage running or queued runs (check `ps -eo pid,args | grep "[c]hain_"`).
</rules>

<output>
Final message: what changed, new episode totals, the pure-reference jobs, the per-GPU ordered list of NAMEs, and the command that builds each full-mode second-stage campaign.
</output>
