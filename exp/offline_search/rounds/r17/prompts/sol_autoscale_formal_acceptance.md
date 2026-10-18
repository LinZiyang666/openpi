Coordinator review of your acceptance handback (fixes accepted: worker_installs/worker_disk overlay, kernel start baseline, legacy overlap check, 207 + 2 tests). One thing is rejected and changes the plan:

<task>
1. The trial caps are wrong. `trial_lane.py` sets timan1 maximum=4 and timan107 maximum=8 with reason "owner validation bound". The owner never set such a bound; the owner explicitly rejected a cap of 8 ("with a cap of 8 you cannot see throughput rise; set many"). Remove these caps and that reason text. A hard limit is allowed only for a real technical limit, stated as such.

2. Run the acceptance on real formal campaigns with large desired maxima (defaults.desired_maxima / physical_maximum), not on 256-episode side lanes. Prepare a dynamic-enabled variant, private to that campaign (global ENABLED stays off), for these two queued campaigns:
   - R17: `fa_ls_fw_f_p_m` (2,500 episodes; definition has workers timan1=60, timan107=60, which are lower bounds, not caps). It is the 2nd item of H100 chain ffk-h100m1. That chain now runs fa_ls_fw_r_p_m_rz (OFF, about 1.5 h, started about 12:35), so its prepare comes at about 14:00–14:30. Fallback if you are not done by 13:50: the 3rd item `fa_ls_gr_f_p_m`; say which one you targeted.
   - R15: `fa_pi_f_u_l` (原库 · π0.5 · LIBERO · 全步 · 各档). It runs only after the 4090 chain's four R16 GR00T LIBERO batches leave timan107 (many hours from now). Also cover the continuation of `fa_pi_f_f_l`, which was stopped mid-run at 10:55 (find its state; it needs continuation or a reset-and-rerun, your call with evidence).
   Same task UIDs, inits, seeds, scientific fields and serving as the OFF prepare. Only the worker orchestration differs. Prove this with a CPU dry run: OFF plan vs dynamic plan, all task fields equal, and print the per-node desired maxima.

3. Memory on timan107 (facts): it has 8×8 GiB GPUs and is shared with legacy LIBERO lanes that are not ledger-aware (R16 GR00T, 64 workers, running now on the 4090 chain). At 12:15 today, a fixed-count R17 lane (60 workers) plus that R16 lane took MemAvailable from above 30 GiB to 3 GiB within one minute; at 10:54 two legacy lanes caused a global OOM (oom_kill 37→45). My coordinator guard `/home/weiland/.claude/jobs/35f2869b/tmp/t107_mem_guard.sh` checks every 60 s. When MemAvailable < 60 GiB, it retires timan107 from any R17 campaign whose driver/journal.jsonl changed in the last 5 minutes, by editing WORKER_NODE_STATE.json (pop node, add to retired, phase=running). Decide, with evidence, the autoscaler's memory reserve and ramp rate on timan107 so it cannot repeat the 12:15 drop while a legacy lane coexists. Then tell me one of two things: the guard is compatible with the dynamic fleet as is, or exactly which dynamic campaigns the guard must skip (I will change the guard).

4. Acceptance criteria stay the same as yours: every UID accepted exactly once, 0 accepted errors, at least two full recycle generations where the run length allows, no probe conflict, minimum MemAvailable recorded, zero OOM from the start baseline, every count change with a reason. Add one more: the per-node active worker count and accepted-episodes-per-minute over time (AUTOSCALE_ACTIVE.csv), so we can see the ramp and the plateau.
</task>

<action_safety>
Deliver new wrapper files beside the existing ones: `exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_<NAME>_dyn.sh` and `launch_<NAME>_dyn.sh`, plus the audit command. Do not overwrite the existing prepare_/launch_ wrappers, do not edit chain scripts or chain lists, do not write into /data/mw_finetune/eval/<NAME> or any running campaign, no remote writes, no launch. I review the wrappers and swap them in myself before the chain reaches the item. Keep the no-regression before/after table for every code change. CPU only, single-reader /data discipline as before.
</action_safety>

<compact_output_contract>
Reply in Chinese, short: which campaigns are ready (paths of the _dyn wrappers), the per-node desired maxima, timan107 reserve/ramp decision and whether the guard is compatible, the plan-equality proof, test results.
</compact_output_contract>
