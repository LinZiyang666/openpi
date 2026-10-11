<task>
Urgent. The Fast-WAM continuation fastwamc7615306896 (H100 lane csl10h, split timan107/local/timan1 = 16/16/16) FAILED after 397
accepted: `remote worker fleet failed: timan1; exit_code=2; worker_restart_budget_exhausted; worker_failures=[all timan1 workers,
generation 0, exit_code 75]`. The resume-aware lane immediately started fastwamc5232306668 with the same split, so it will fail the
same way. Hypothesis: 75 is worker_lifetime's planned-recycle exit after 4 episodes, and T1's timan1 device-namespace wrapper /
supervisor path does not propagate or classify it as a recycle, so every recycle burns the restart budget. Verify from timan1
worker logs and the receipts.
Do:
1. Fix the timan1 path so planned recycles are handled exactly like timan107/local (no budget use), keeping the GPU-3-only guard.
   CPU tests reproducing the 75-through-wrapper case.
2. If the running fastwamc5232306668 still has the bug, either let it finish its cycle or stop it cleanly by its own PIDs and let the
   lane resume with the fix; keep accepted cells. If the fix cannot be made within ~15 min, have the lane continue WITHOUT timan1
   (timan107 + local) rather than cycling, then re-add timan1.
3. Report run names and acceptance progress, and confirm DP continuation dpc3464789869 (4090 lane) was not touched.
</task>
<compact_output_contract>
End with: root cause evidence, fix (files/tests), current H100 run name + accepted count, timan1 status.
</compact_output_contract>
<action_safety>
timan1 GPU 3 only; do not touch the 4090 lane, other users' processes. Kill only your own verified PIDs. No pkill -f bare patterns,
no rm -rf, no /data/i018, /dev/shm/i018, codex credentials.
</action_safety>
