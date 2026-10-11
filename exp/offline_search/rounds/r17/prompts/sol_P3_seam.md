<task>
Owner accepted batching (small numeric drift is fine if success rate is unaffected). Adopt your recommendation
B8 / 5 ms for all Fast-WAM formal runs. The coordinator has stopped the H100 B1 arm (C10) and the B1 baseline
had already failed; a fresh batched control + all arms will be run.

Make batching a first-class, qualified path of the normal launcher so it runs on h100 as well as the local 4090:
1. Integrate it into exp/r17_integration/ops/formal_eval.sh / formal_eval.py (the "P6 seam" you mentioned) with
   flags --max-batch / --batch-wait-ms, recorded in the campaign identity and receipt so control and arms are
   provably matched; refuse mixing. Default off.
2. CPU tests. Then a bounded h100 smoke (<=15 GPU minutes, 4 episodes, 1-2 replicas): h100 currently hosts the
   Cosmos K2 campaign (2 replicas, ~13 GB, ports 23240-23249) — do not touch it; use Fast-WAM ports 23170-23179.
   Workers on timan107 only (timan108 is offline; timan1 is forbidden).
3. Recommend replicas per GPU (h100 80 GB, 4090 48 GB) and workers per replica so B8 actually fills, given that
   LIBERO workers currently leak kernel-side memory until they exit (sol M is fixing it with periodic worker
   recycling; check exp/r17_integration/REQUESTS/M_to_Q.md and its diff).
4. Write the exact commands: Fast-WAM batched official baseline (4 suites, --official-50) and the six study arms
   (--suites libero_spatial libero_10: C10, C16, C20, C32 at K10; K2, K1 at C5) — split across h100 and the 4090
   so both GPUs are used, each arm paired with a control on the same GPU type (so the control runs on both GPUs,
   or all arms on one GPU; pick the faster overall schedule and say why).
Other agents editing nearby right now: Q (ops/formal_resume, worker_memory), M (worker recycling), P4 (DP batching),
D (Cosmos stall fix). Check the current diff before editing; keep edits narrow; use REQUESTS/P3_to_<X>.md for
anything in their files.
</task>
<compact_output_contract>
End with: files changed, tests, h100 smoke result (throughput, max diff, receipt path), and the command list with
expected duration per campaign.
</compact_output_contract>
<action_safety>
Do not stop or touch running campaigns (Cosmos K2 on h100/timan107) or P4's 4090 measurement (honor
/tmp/r17_gpu.lock on the 4090). No pkill -f bare patterns, no rm -rf. Do not touch other users' processes,
/data/i018, /dev/shm/i018, or codex credentials.
</action_safety>
