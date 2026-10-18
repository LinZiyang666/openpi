<task>
Coordinator, 2026-10-17 05:5x. timan108's tether agent went OFFLINE at ~05:43. The π0.5 full-step formal_A items use the legacy R15 path with a single worker host (driver + all 39 workers on timan108, `WORKER_HOST=timan108`). `ffk1015_pi05_libero_full_formal_A_pure_l10` (H100 R3 B1, 500 episodes) started 05:12:57 and had 445/500 accepted in `timan108:/scratch/zixuans8/openpi_trace/os_cl/runs/ffk1015_pi05_libero_full_formal_A_pure_l10/ffk1015_pi05_libero_full_formal_A_pure_l10_c10/journal.jsonl` at 05:23; then the driver node vanished and the run looped on STATUS_UNKNOWN. I stopped it by PID at 05:49 (local `filter_failed.legacy chain` process, and the three H100 π0.5 servers on 23230–23232; ports are free). The H100 queue now continues with the R17 items without timan108; the held π0.5 R15 items are: pure_l10 (resume), pure_spatial, pi05 full original/filtered off, original/filtered levels.

Owner rule: workers do not care which machine they run on, and one dead worker machine must never block experiments. Make the legacy R15 (and, if the same pattern exists, the R16 VLA) formal_A items runnable on another online worker node — timan107 (8 GPUs, its R15 island is the one GR00T R15 used) or timan1 GPU 2 (via timan1_gpu2) — with the same server topology and plans, choosing the node at launch from what is online (the worker_resilience probe can be reused). Results must stay comparable: same H100 serving, same seeds/plans; only the worker host changes.

For pure_l10: build a continuation that runs only the episodes not yet accepted. If timan108's journal is unreachable, run the whole 500 again on the new host and, when timan108 returns, keep the new run (do not merge). Tell me which you did.

Deliver new prepare/launch scripts for the held items (same NAMEs, or NAMEs with a clear suffix), so I can queue them on the H100 after the current items.
</task>

<rules>
- CPU only locally; tether exec read-only checks only; never push/pull/expose; never start or kill workers, servers, GPUs or chains; no pkill, rm, git commits; never touch credentials.
- Do not modify running runs or the running chain scripts (ffk-h100v, ffk-4090w, ffk-4090x).
</rules>

<output>
Final message: what changed, the chosen host logic, pure_l10 handling, and the exact NAMEs to queue in order.
</output>
