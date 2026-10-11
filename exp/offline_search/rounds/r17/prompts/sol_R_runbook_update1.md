<task>
Owner rule: whenever a better configuration is found, the recorded configs must be corrected immediately so nobody
steps into the same pit again. Update exp/r17_integration/ops/RUN_CONFIGS.md now (it is the launch reference), replacing
wrong/obsolete recommendations rather than appending, and add a "Pitfalls (do not repeat)" section with the evidence
path for each. Today's lessons (verify each against the cited evidence before writing):
1. Several replicas on one GPU only time-slice (no extra compute) unless MPS lets kernels overlap. DP with 12 replicas
   at B1 was slow (1409 ms/call). H100 Fast-WAM 2 replicas x B8 at 14:06: SM 95-99% but only ~200 W of 700 W, batch fill
   p50 5/p90 8, ~2.9 s server latency per call, 72/90 timan107 worker processes idle -> server-bound and underused.
   Do not recommend multi-replica-without-MPS as a throughput lever.
2. Batch fill, not cap, is what matters; measure fill (S's recorder) before raising caps.
3. Driver fsync/scheduler lock stalls (S: 19-26 s DP reporting/dispatch delays; DP 4090 SM ~0% at 14:06 while workers
   idle) - the conductor driver can be the bottleneck; Q is fixing (orchestration pin).
4. LIBERO workers must spread EGL rendering over all GPUs of the worker node (all on GPU0 exhausted 8 GB and leaked host
   memory -> node OOM, timan108 hang); recycle every 4 episodes; per-worker ~2 GiB RAM live (64 workers ~123 GiB),
   VRAM peak ~532 MiB; timan107 VRAM admission (8x8 GB) is the real worker cap (48x768 + 32x576 MiB filled it).
5. Worker count and batch config must be fixed at the baseline: arms are refused unless identical (pairing pins);
   decide throughput config BEFORE starting a baseline. Arms run from the baseline's frozen runtime (--runtime-from).
6. MPS (P3): 4090 DP 2xB8 +15%; DP 1xB16 and Fast-WAM 1xB8 no gain; keep off within a study unless the baseline used it.
7. Ports: Cosmos must use 23240-23249; Fast-WAM 23170-23179; DP 23180-23189. timan1 forbidden; timan108 GPUs 1-2 belong
   to another user. Tether call budget (keep probes >= 30 s apart).
8. Startup gap: use ops/formal_queue.sh with prestaging (60-110 s switches) instead of ad hoc sequential launches.
Pending inputs that will change the Fast-WAM/DP recommendations: P3's clean H100 sweep (1xB16/B24/B32, 2xB16 +/- MPS;
REQUESTS/P3_to_R.md), S's DP wait breakdown (REQUESTS/S_to_R.md), Q's driver fix. Mark those rows "pending P3/S/Q"
now; the coordinator will send you their results for a second update.
</task>
<compact_output_contract>
End with: list of replaced recommendations (old -> new), the pitfalls section headings, and rows marked pending.
</compact_output_contract>
<action_safety>
Documentation only; no launches or process actions. No rm -rf; no codex credentials.
</action_safety>
