<task>
New failure class, needs an orchestration-level fix that does not change pinned code identities:
DP arm hs5dp_c15 and its retry hs5dp_c15r0 failed before any episode with
`RuntimeError: MPS command failed: nvidia-cuda-mps-control: Cannot send command to MPS control daemon process`
(raised from the frozen runtime's exp/r17_integration/models/fastwam/ops/mps.py:55). The private daemon's control.log
(/data/mw_finetune/eval/hs5dp_c15r0/mps/e5681f/l/control.log) shows concurrent UI connections at the same instant while
two replicas start in parallel; c10 (same code) happened to succeed. Workaround in use: DP remaining arms re-prepared with
--replica-startup-width 1 (queue /home/weiland/projects/openpi_ext/scratch/r17_ops/queue_hs5_dp_retry2.run.sh). The running
H100 Fast-WAM queue (/data/mw_finetune/eval/fq_hs6fw, --replica-startup-width 2 + --mps) is exposed to the same race for
arms C16..K1, and formal_queue stops on the first failure.
Do:
1. Confirm the root cause (concurrent nvidia-cuda-mps-control invocations / daemon readiness).
2. Fix it without changing server/worker/model identity pins: e.g. serialize MPS control calls with a lock and retry
   "Cannot send command" with backoff in an orchestration layer outside the frozen runtime, or treat startup width as
   orchestration. Tests.
3. Make formal_queue retry a job that fails before any accepted episode (fresh prepared copy with identical settings,
   bounded retries), and decide whether the running fq_hs6fw queue can pick this up safely; if it requires restarting the
   queue between arms, give the coordinator the exact procedure (do not interrupt the running C10 arm).
4. Append the pitfall to exp/r17_integration/REQUESTS/Q_to_R.md.
</task>
<compact_output_contract>
End with: root cause evidence, fix (files/tests), whether pinned identities are unchanged, and the exact procedure for the
running Fast-WAM queue.
</compact_output_contract>
<action_safety>
CPU only; do not stop/signal running campaigns (DP c15s0 on 4090, Fast-WAM hs6fw_c10 on h100). No pkill -f bare patterns,
no rm -rf, no /data/i018, /dev/shm/i018, codex credentials. timan1 forbidden.
</action_safety>
