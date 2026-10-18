<task>
Follow-up from the coordinator, 2026-10-17 06:0x. The first formal_A R17 launch failed on a documented pitfall (RUN_CONFIGS "Keep MPS pipe paths within 85 bytes"):
`ffk1015_cosmos_libero_reduced_formal_A_original_levels` started 05:54:16, ended 05:56:45 rc=1 receipt PARTIAL (all_owned_closed=false; I checked: no leftover processes on H100 or locally) with
`ValueError: MPS pipe path too long; use a shorter campaign run name (pipe <=85 bytes)` from the frozen runtime's fastwam/ops/mps.py paths() — `/data/mw_finetune/eval/<run>/mps/<6-char GPU hash>/p` is 90 bytes for this name.
Every formal_A NAME has the same long pattern, so every MPS-using formal_A campaign (H100 Cosmos/Fast-WAM, and any 4090 DP/Fast-WAM lane that uses MPS) would fail the same way. I put the formal_A queues on hold; the GPUs are running MetaWorld lanes meanwhile.

1. Give every formal_A campaign (built and deferred, both arms, both GPUs, including the π0.5 R15 items you are rerouting) a short run name so that all derived paths satisfy the frozen runtime's limits (MPS pipe <=85 bytes, and any other length limit you find), e.g. `fa_<model>_<mode>_<arm>_<stage>` with short tokens. Keep plans, seeds, serving, cells and the formal design unchanged; keep the mapping old→new in one JSON and make analysis use it.
2. Bring the failed run back to an unstaged state with the existing reset path (rename only), then rebuild it under its short name.
3. Verify on CPU, for every formal_A campaign, the computed MPS pipe path length for its actual GPU hash, plus startup identity.
4. Regenerate commands/prepare_<NAME>.sh / launch_<NAME>.sh for the short names and give me the ordered NAME lists per GPU (H100 and RTX 4090), knob-off stages before level stages, R15/R16 items with the host logic from your previous turn.
</task>

<rules>
- CPU only locally; tether exec read-only checks only; never push/pull/expose; never start or kill workers, servers, GPUs or chains; no pkill, rm, git commits; never touch credentials.
- Do not modify running runs or the running chain scripts (ffk-h100w, ffk-4090w, ffk-4090y).
</rules>

<output>
Final message: the naming scheme and mapping file, path-length check results, the reset of the failed run, and the ordered NAME lists per GPU.
</output>
