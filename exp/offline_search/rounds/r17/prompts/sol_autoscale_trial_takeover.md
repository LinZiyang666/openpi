<task>
Owner ruling 10-18 12:2x: astra is no longer used; you take over the remaining autoscale acceptance work (code in exp/offline_search/rounds/r17/autoscale_1017/, astra's notes in HANDOFF_OFF.md, followup_1018/*/BEFORE_AFTER.md, TRIAL.md). Global switch autoscale_1017/ENABLED stays off.

State:
- R15 acceptance lane fa_asv_r15_pi05_1018 (256 eps): 256/256 accepted, 0 errors, 60 recycled generations, no probe conflicts, above critical; audit FAIL only because zero_oom evidence is UNKNOWN (no baseline at lane start). On timan107 `dmesg` is readable without sudo and /proc/vmstat oom_kill is readable; timan107 had a real global OOM at 10:54 caused by two legacy LIBERO lanes overlapping (not by the trial).
- R17 acceptance lane fa_asv_r17_fw_1018: prepare keeps failing on the old 10-09 source formal_fastwam_mw_mt50_500_20261009_223917_0b845f00: first LOCAL_PORT_BUSY 23177 (fixed), then KeyError runtime_identity (astra fixed), now `KeyError: "filename 'exp/r17_integration/ops/worker_installs.py' not found"` in a tarfile getmember (log openpi_ext/scratch/r17_ops/ff/slot4_asv_r17_prep.log).

Do, CPU only (no launches):
1. Make the R17 acceptance lane preparable: either choose a newer FW (or other) MT50 R17 source with a complete runtime/worker archive, or fix trial_lane so the old source works; keep task UIDs/inits/seeds from the source plan. Give me the exact build/prepare/launch/audit commands.
2. Make the audit's zero_oom evidence capture a baseline (dmesg cursor + /proc/vmstat oom_kill) at lane start and compare at the end; give me the command to re-run the R15 lane (fresh lane dir) so it can pass.
3. Both lanes must not run while another LIBERO legacy lane uses timan107 (the 4090 chain runs R16 GR00T LIBERO there for several hours); state each lane's worker nodes.
</task>
<compact_output_contract>Reply in Chinese, short: what changed, exact commands, tests.</compact_output_contract>
