<task>
Owner requirements (binding, 2026-10-17 23:5x):
1. Every model must be tested for equivalence: π0.5, GR00T, DP, Fast-WAM, Cosmos. Not DP only.
2. Equivalence must be proven on the concurrent server we actually deploy in formal runs, not on a B1 staged replay. The deployed path includes: the formal server bootstrap from the frozen release/runtime, the real replica count and batch size per model and GPU (RUN_CONFIGS.md), dynamic batching of concurrent requests, MPS where used, the socket/client protocol, the fitted cache recipe (cache hit/miss, nearest neighbour, corrector, stall gate, periodic knob), long execution segments with the decision-step clock fix, and CUDA graphs where the formal server uses them. A B1 staged replay may stay as a fast numerical screen, but it is not the evidence.

Status you should know:
- Your 4090 DP B1 screen ran: results/4090_native_20261018T045343Z, status COMPLETE, 9 rows, 44 s, capture/plain parity exact on repeat 0.
- Transfer channel is open (no SSH needed): weilandserver runs a read-only rsync daemon at rsync://ziyanglin.com:23177/ restricted to the A100 (149.165.152.232) and H100 (149.165.153.233) IPs. Modules now: xr_rootfs (/data/mw_finetune, include-listed from prepared_dp_final/rootfs.files), xr_site (openpi .venv site-packages, include-listed), xr_python (uv cpython 3.11.15), xr_prepared, xr_inputs, xr_tools. Config: /home/weiland/.claude/jobs/35f2869b/tmp/xgpu_rsyncd/rsyncd.conf (re-read on every connection). You may append narrow read-only modules there for the other models (include-listed paths only, keep hosts allow as is, never expose a whole home or credentials). Pull from the A100 side with `rsync -a --bwlimit=16384 rsync://ziyanglin.com:23177/<module>/ <dest>/`, one stream at a time; check /proc/pressure/io on weilandserver before big pulls.
- A100 disk: ~47 GB free. Do one model at a time on the A100; delete that model's weights/env (only files you created) before the next. Record bytes used per model. Passwordless sudo only when needed, every command logged.

What to build and deliver (exp/offline_search/rounds/r17/xgpu_1017/):
A. A concurrent-server harness per model: start the formal server exactly as formal_eval starts it (same release, runtime identity, model spec, R, B, MPS, cache YAML, decision-step fix) on a chosen GPU and port; drive it with many concurrent clients (as many as the formal worker count for that model, or at least enough to fill every batch) that send recorded observations with fixed seeds through the real client protocol; record per request: action chunk, cache decision (hit/miss, neighbour id, distance/score), corrector/stall/knob decisions, batch composition, server_timing.
B. Noise floor on the same GPU with the same concurrency (two runs of A on one GPU): batch composition changes between runs, so cross-GPU differences are judged against this floor, not against B1 repeat noise.
C. Cross-GPU comparison per model: 4090 vs H100 first (both run formal batches today), then A100. Report tensor/action differences vs the same-GPU concurrent floor and cache-decision agreement.
D. Closed-loop check on the real formal path: a small paired formal_eval campaign per model on each GPU (same plan, seeds, inits; real workers), paired SR with 95% CI and McNemar. Size it so the CI is meaningful; say how many episodes and GPU hours each needs.
E. If not equivalent: per-model ablation order (TF32 off, matmul precision highest, attention backend fixed, compile/CUDA graph off, deterministic) on the A100, and what setting restores equivalence.

Who runs what: you may run on the A100 directly (any length is fine there, it is dedicated to this study). For the 4090 and H100 write exact commands for me with GPU memory and duration; both GPUs are running formal batches, so keep each step's memory stated; I can temporarily drop one replica on the 4090 if a step needs more. Do not touch running lanes, chains, or other people's processes.
</task>

<compact_output_contract>
Reply in Chinese, short. Order: models covered and status, the concurrent harness design in 5 lines, the exact commands for me (4090 then H100) with memory and minutes, A100 progress, open risks.
</compact_output_contract>
