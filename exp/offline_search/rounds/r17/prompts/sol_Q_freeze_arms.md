<task>
Your queue works, but prepare-only of the H100 Fast-WAM arm queue failed on the 3rd job:
`ValueError: pending control sealed identity mismatch: model_manifest.json`
(queue output /data/mw_finetune/eval/formal_queue_20261010_130935_c44e6c03/, jobs file
/home/weiland/projects/openpi_ext/scratch/r17_ops/queue_hs3_fast.json; C10 and C16 prepared, C20 failed).
The paired baseline is the running campaign /data/mw_finetune/eval/hs3_fast_b8w5_h100_k10_c5_control_20261010_125303
(and similarly the DP baseline hs3_dp_b16w5_4090_k100_c5_control_20261010_125303). Other agents (P3: MPS option in the
launcher/serve seam; possibly others) keep editing the working tree while these studies run, so arms prepared later
get a different model identity than their baseline.

Do:
1. Identify which file(s) changed the manifest between the baseline seal and the C20 prepare.
2. Make arms run from the baseline's frozen runtime/code release (the runtime.tar / RUNTIME_RELEASE.json the
   baseline sealed), so working-tree edits cannot change an arm's identity; e.g. a `--runtime-from <baseline run>`
   option the queue applies to every job with --paired-control (and an equivalent for DP arms, which use
   exp/r17_integration/models/dp/formal_eval.sh / models.dp.formal_launch without --paired-control). Refuse if the
   frozen release is missing or its hashes do not verify.
3. Re-run prepare-only of the Fast-WAM queue (CPU only) until all 6 jobs prepare against the frozen release, and say
   how the DP arm queue (/home/weiland/projects/openpi_ext/scratch/r17_ops/queue_hs3_dp.sh) must be changed (or change
   it) so DP arms also use the DP baseline's frozen release.
</task>
<compact_output_contract>
End with: the changed file(s) that broke identity, files changed, tests, prepare-only result for all 6 jobs, and the
exact edits/commands for both queues.
</compact_output_contract>
<action_safety>
CPU only. Do not stop, signal or reconfigure running campaigns. No pkill -f bare patterns, no rm -rf, no /data/i018,
/dev/shm/i018, codex credentials. timan1 forbidden.
</action_safety>
