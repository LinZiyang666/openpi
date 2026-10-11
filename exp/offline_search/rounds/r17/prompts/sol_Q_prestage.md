<task>
Thanks — the B1 Fast-WAM baseline will NOT be resumed: the owner adopted batching (B8 / 5 ms), so a fresh batched
control + all arms will run. Your resume path is still needed for the Cosmos baseline (D staged 274 cells).

New task: remove the GPU-idle gap between consecutive formal campaigns. Today each campaign spends ~10-15 min
between launch and its first accepted episode (e.g. DP hs2_dp_k100_c5_default_20261010_111846 launched 11:18:46,
first journal line ~11:36; Fast-WAM C10 launched 11:20:43, workers started ~11:30) while the GPU sits idle.
1. Measure where that time goes from the receipts/TETHER_CALLS.jsonl of those runs (bundle/assets tar build and
   transfer, worker runtime deploy, server load + CUDA graph/compile warmup, admission, preflight probes).
2. Cut it: reuse immutable deployed worker runtimes/assets across campaigns when hashes match, and support
   preparing campaign N+1 (everything except binding GPU replicas and dispatch) while campaign N is still running,
   so a queue can switch with < 2 min idle. Provide a queue helper (e.g. ops/formal_queue.sh) that runs a list of
   formal_eval argument sets back-to-back on one GPU with prestaging, stopping on failure, and that also honors
   worker recycling (M: --worker-max-episodes, requests in REQUESTS/M_to_Q.md — apply them) and the renderer
   spreading D is adding now (check REQUESTS/D_to_M.md / D's diff).
3. CPU tests; no GPU use needed (optional: a <=5 min local smoke of the prepare step only, no GPU).
Others editing nearby: P3 (batching flags in formal_eval.py), D (worker_gpu/formal_workers renderer spread),
P4 (DP batching). Check diffs first, keep edits narrow, preserve their changes.
</task>
<compact_output_contract>
End with: idle-time breakdown (seconds per phase for the two runs), what you changed, expected idle per switch,
tests, and the queue helper usage.
</compact_output_contract>
<action_safety>
Do not stop or touch running campaigns (Cosmos K2 on h100/timan107) or other agents' smokes. No pkill -f bare
patterns, no rm -rf, no /data/i018, /dev/shm/i018, codex credentials. timan1 forbidden.
</action_safety>
