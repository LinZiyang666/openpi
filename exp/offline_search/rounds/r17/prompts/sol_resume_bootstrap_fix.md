<task>
Defect found by the coordinator (2026-10-18 12:5x): the continuation campaign `/data/mw_finetune/eval/fa_ls_gr_r_p_m_c1` (GR00T · MetaWorld · 减步 · 每次调模型+长执行段对照, 1,561 episodes; built with `exp.offline_search.rounds.r17...resume --source fa_ls_gr_r_p_m --name fa_ls_gr_r_p_m_c1 --remaining --verify-closed`, wrapper `filter_failed_knob/commands/prepare_fa_ls_gr_r_p_m_c1.sh`) ran its servers WITHOUT the long-segment decision-step bootstrap.

Evidence:
- `driver/per_step.jsonl` decision rows in c1 have `server_timing = {"infer_ms": ...}` only, and no `clock_unit`, `segment_head`, `decision_index`, `delivery_index`, `env_step_index`, `decision_seed` or `wire_explicit_seed`.
- The source `fa_ls_gr_r_p_m` (same runtime archive 66499ead…, same bootstrap 4e07fb3e…, same recipe/bundle/model identities) has all of them, with `clock_unit=segment_decision`.
- The patched `_ControlSocket.send` in `longseg_server_assets/ffk_bootstrap.py` copies these fields whenever the patched `ConnectionPolicy.infer` runs. So c1's servers ran the unpatched serving code, even though `servers/manifest.json` pins the bootstrap sha.
- The currently running `fa_ls_fw_r_p_m` (normal stage path) is patched; this is specific to the continuation path.

Do:
1. Find the exact reason the continuation server launch skipped the bootstrap (file:line), and fix it so a continuation's servers load the same bootstrap as the source.
2. Add one hard check to every R17 launch where commit_horizon > replan_steps: if the first accepted responses lack `clock_unit=segment_decision` in server_timing, abort the launch before it accepts more episodes. Also add the same check to receipt finalization: a run with any such decision row cannot be PASS. Keep it minimal.
3. Scan every R17 campaign finished or started since the long-segment fix (2026-10-18 01:00 CDT) with commit_horizon > replan_steps, and list any other run with unpatched rows. Single-reader /data, sequential, and check /proc/pressure/io. One known-good list: fa_ls_gr_r_p_m, fa_ls_fw_f_b_l, fa_ls_fw_f_p_l, fa_ls_fw_r_p_m (running).
4. Build a new continuation `fa_ls_gr_r_p_m_c2` covering exactly the 1,561 task UIDs that c1 covered (same UIDs, inits, seeds; source = fa_ls_gr_r_p_m), with the fixed path. Write the wrappers `filter_failed_knob/commands/{prepare,launch}_fa_ls_gr_r_p_m_c2.sh`. Prove with a CPU dry run that its servers will load the bootstrap. Do not launch, and do not move c1; I will move c1 to trash and queue c2 myself.
</task>

<action_safety>
Do not edit chain scripts or chain lists, do not touch running campaigns (fa_ls_fw_r_p_m on H100, fa_gr_r_u_l on 4090), no remote writes except what a CPU dry run needs (none expected). Another sol thread is editing `exp/offline_search/rounds/r17/autoscale_1017/*` and `closed_loop/ops/h100/control.py` / autoscale driver files; do not edit those, and coordinate by keeping your change in the resume/continuation path and the launch/receipt check. Keep the before/after test table for each code change. CPU only.
</action_safety>

<compact_output_contract>
Reply in Chinese, short: root cause (file:line), the fix, the new hard check, the scan result (any other unpatched runs), the c2 wrappers and dry-run proof, test results.
</compact_output_contract>
