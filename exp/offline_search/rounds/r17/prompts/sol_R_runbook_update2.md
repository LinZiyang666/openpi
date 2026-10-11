<task>
Second correction of exp/r17_integration/ops/RUN_CONFIGS.md (owner rule: replace wrong recommendations immediately so
no one repeats a pit). Resolve every "pending P3/S/Q" mark and replace the Fast-WAM and DP LIBERO recommendations:
1. Fast-WAM LIBERO on H100 (P3 clean sweep, REQUESTS/P3_to_R.md, models/fastwam/BATCHING.md): 2 replicas x B16 / 5 ms
   + --mps = 28.0 calls/s, p50 1679 ms, 413 W (1xB16 16.8, 1xB24 19.2, 2xB16 no MPS 24.8; 3 replicas do not fit,
   ~85 GiB). Drift max .015625 (first-5 actions .0078). This is now the launched baseline hs6fw_ctl_143332 with
   48 timan107 workers, --driver-mode group-commit. The old "keep R2/B8/W32" advice is wrong; replace it.
2. DP LIBERO on 4090 (S's breakdown in REQUESTS/S_to_R.md:120; P3 MPS table): server is host-bound (DDPM100 host loop;
   stage3 in flight 93% while GPU 41%/164 W); workers wait ~49% on policy, ~33% on driver, ~11% simulating. Use
   2 replicas x B8 / 5 ms + --mps (P3: 26.2 vs 14.4 calls/s for 1xB16) + --driver-mode group-commit; now launched as
   hs5dp_ctl_142812 with 32 workers (VRAM admission cap next to Fast-WAM's 48). Future lever: CUDA-graph the whole
   sampler loop (est. 1.15-1.35x). Replace the old "R1/B16/W24" advice.
3. Driver (Q, HANDBACK_Q.md:1518): --driver-mode group-commit; real-/data replay dispatch median 667 -> 0.13 ms,
   reporting 27.6 s -> 0.09 ms; durability rule; orchestration pin lets arms use it against older-driver baselines.
4. New pitfalls: MPS pipe path <= 85 bytes -> keep run names short (e.g. hs6fw_ctl_HHMMSS, hs5dp_<arm>); after stopping
   a campaign its ports can stay busy for a minute ("Address already in use") -> wait or pick a fresh name/port; never
   run background queue scripts from files agents may edit (run from a copy, *.run.sh); worker cap on timan107 is VRAM
   admission (48x768 + 32x576 MiB fills 8x8 GB), not RAM; restart-the-baseline decision must happen before arms because
   batch/replica/worker/MPS settings are pinned.
5. Note that GPU "util %" (NVML busy) is misleading: judge saturation by power draw and batch fill (H100 at 99% util
   drew only ~200 W with 2xB8).
Keep the per-model summary table at the end current (GPU, replicas, batch, MPS, driver, workers, measured rate).
</task>
<compact_output_contract>
End with: replaced recommendations (old -> new), pending marks remaining (should be none or explained), new pitfalls added.
</compact_output_contract>
<action_safety>
Documentation only; no launches or process actions. No rm -rf; no codex credentials.
</action_safety>
