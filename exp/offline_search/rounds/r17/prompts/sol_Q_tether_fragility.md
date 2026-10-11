<task>
Both batched baselines were declared FAIL at 15:26-15:28 by tether hiccups although the episodes were fine:
- DP baseline /data/mw_finetune/eval/hs5dp_ctl_142812: 2000/2000 accepted, RECEIPT FAIL from the final evidence pull
  `tether pull timan107:/tmp/r17_Q_qualification/hs5dp_ctl_142812_timan107/workers_...tar ... error: pull (tier B): write
  local: nats: timeout` (a 483 MB tar). The coordinator re-pulled it successfully to
  /data/mw_finetune/eval/hs5dp_ctl_142812/timan107.tar (93 s). DP arms are now running from prepared runs
  /data/mw_finetune/eval/hs5dp_<arm> (they only pin --runtime-from that baseline; no --paired-control).
- Fast-WAM baseline /data/mw_finetune/eval/hs6fw_ctl_143332: 1839/2000 accepted, RECEIPT FAIL
  `tether exec --safe --timeout 10s h100 -- env ... timed out after 10s waiting for chunk` (same minute as the 483 MB
  pull: our own bulk transfer likely starved tether). The coordinator launched a resume (run name in
  /home/weiland/projects/openpi_ext/scratch/r17_ops/hs6_fw_resume_stamp.txt, --resume-from hs6fw_ctl_143332).
  The 6 prepared Fast-WAM arms (/data/mw_finetune/eval/fq_hs6fw, jobs scratch/r17_ops/queue_hs6_fw.json) pin
  --paired-control/--runtime-from hs6fw_ctl_143332.
Do:
1. Re-seal the DP baseline to PASS from the recovered tar (CPU only), recording the failed pull and the recovery.
2. Make the merged Fast-WAM result (failed segment + resume) a valid --paired-control for the prepared arms, or tell
   the coordinator exactly what to change in the arm queue (e.g. point --paired-control to the merged/resumed run while
   keeping --runtime-from on the original sealed runtime). Re-run prepare-only for fq_hs6fw if needed.
3. Remove the fragility: no single tether timeout may fail a campaign (bounded retries with backoff for probes/execs;
   distinguish transport hiccup from real worker/fleet death); move large evidence transfers off the tether control
   path (rsync via ziyanglin.com rsync daemon / bwlimit, or chunked pulls with retry) so one campaign's bulk pull cannot
   starve another campaign's probes. CPU tests.
4. Write a short pitfall entry to exp/r17_integration/REQUESTS/Q_to_R.md for RUN_CONFIGS.
</task>
<compact_output_contract>
End with: DP baseline receipt status + SUMMARY path, the exact arm-queue change (or none) for Fast-WAM, files/tests for the
fragility fix, and the pitfall text.
</compact_output_contract>
<action_safety>
CPU only. Do not stop/signal running campaigns (DP arms on 4090, Fast-WAM resume on h100). No pkill -f bare patterns,
no rm -rf, no /data/i018, /dev/shm/i018, codex credentials. timan1 forbidden.
</action_safety>
