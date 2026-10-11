<task>
Owner request: write down, for every model, how to run it concurrently (server + worker configuration) so
experiments advance as fast as possible, and keep it current.

Write exp/r17_integration/ops/RUN_CONFIGS.md — the single launch reference the coordinator follows. One section per
model x benchmark: π0.5, GR00T, DP, Fast-WAM, Cosmos on LIBERO; π0.5, GR00T, DP, Fast-WAM on MetaWorld MT50.
For each: launcher command template (formal_eval.sh flags), server GPU options (h100 80 GB, local 4090 48 GB),
replicas per GPU, batching (max batch / window, numeric-drift note), reserved port window, worker node and worker
count, --worker-memory-mib / --worker-vram-mib / --worker-max-episodes / renderer spreading (--worker-egl-devices),
measured throughput (episodes/min, calls/s, GPU util/W), what the bottleneck is (GPU compute, batch fill, workers,
node RAM), and the recommended config for (a) the GPU alone and (b) when sharing the GPU / worker node with
another campaign. Note pairing rules (controls and arms: same GPU type and batch config) and known pitfalls.

Sources: handbacks under exp/r17_integration/handbacks/ (P3, P6, COSMOS, Q), models/{fastwam,dp}/BATCHING.md,
qualification receipts, completed campaign SUMMARY/receipts in /data/mw_finetune/eval/ (formal_*, hs2_*, hs3_*), the
R10/R16 openpi concurrency history for π0.5/GR00T (logs/cache_prune_run_handoff.md, logs/session_handoff.md).
Live measurement now: the running campaigns hs3_fast_b8w5_h100_k10_c5_control_* (h100, 2 replicas, B8/5ms,
timan107 x32) and hs3_dp_b16w5_4090_k100_c5_control_* (4090, 1 replica, B16/5ms, timan107 x24) — read their batch
ledgers / per_step / journals (read-only) to report actual batch fill, queue wait, episodes/min, and say whether more
workers, more replicas, or a different batch cap would make them faster. Light read-only GPU probes (nvidia-smi on
h100 via tether, local) are fine; keep tether calls >= 30 s apart.

Facts to respect: timan1 is forbidden; worker nodes are timan107 (8x GTX 1080, 220 GiB; owner has run 64 workers
there) and timan108 (4x A5000, GPUs 1-2 belong to another user; currently offline). The LIBERO worker host-memory
growth problem was mitigated by renderer spreading (D) + recycling (M/Q); give the per-worker numbers you can
verify from the live fleets.
</task>
<compact_output_contract>
End with: a per-model summary table (model x benchmark: GPU, replicas, batch, workers, episodes/min, bottleneck) and
concrete tuning recommendations for the two running campaigns and for the next queued arms.
</compact_output_contract>
<action_safety>
Documentation + read-only measurement only. Do not launch, stop or signal campaigns; do not edit launcher code.
No pkill -f bare patterns, no rm -rf, no /data/i018, /dev/shm/i018, codex credentials.
</action_safety>
