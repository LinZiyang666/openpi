<task>
Make the DP-LIBERO formal server batch concurrent requests, measure it, and hand back a launch config.

Facts: the DP official baseline ran 12 replicas on the 4090 (48 GB) at batch=1: median cost_ms 1409 per
call (DDPM K=100, CUDA graph), GPU "100% util" but only ~188 W. 12 processes on one GPU only time-slice;
they add no compute. The owner stopped that design: use few processes and real batching.
Earlier you disabled B>1 because B1 vs B2 differed by 6.6e-5. Owner ruling: small numeric differences are
acceptable as long as the success rate is not affected; control and arms will all use the same batch config.
Keep per-request noise independent of batch composition (per-request RNG / seed) so a request's sample does
not depend on who it was batched with, as far as the model allows.

P3 (another agent, same repo) is right now adding server-side dynamic batching for Fast-WAM; it found the
formal server serializes every connection within a replica and the batch hooks are gated off. Look at the
current working-tree diff under exp/r17_integration/ and exp/r17_integration/REQUESTS/P3_to_P6.md before
editing. If P3's batcher is model-agnostic, plug DP into it; do not edit common serving files P3 is touching;
if you need a common change, write it to exp/r17_integration/REQUESTS/P4_to_P3.md and implement the DP side.

Do:
1. DP batched inference hook (B up to 32) for all DP sampler arms used in the study: DDPM K100, DDIM K10, K5
   (trailing, eta=0); commit horizons 5/10/15/16 unaffected.
2. CPU tests pass. Then a GPU measurement on the 4090 (GPU-98d36ed2...), honoring /tmp/r17_gpu.lock with
   flock (P3 holds it until about 12:05); at most 20 GPU minutes: throughput (calls/s and actions/s), latency
   per call, power, for replicas {1,2} x max batch {1,4,8,16,24,32} under realistic concurrent load; per-request
   max abs action diff vs B=1.
3. Write exp/r17_integration/models/dp/BATCHING.md: table, recommended replicas / max batch / batch window,
   and the exact formal_eval.sh command for the DP official baseline (4 suites, --official-50, server local
   4090, workers on timan108 sized so the batch actually fills, respecting the node RAM admission:
   timan108 will also host 24 Cosmos workers at 6 GiB each; timan1 is forbidden).
</task>
<compact_output_contract>
End with: the measurement table (rows = config, cols = calls/s, actions/s, p50/p90 ms, W, max diff),
the recommended config with expected speedup over 12 replicas x B1 (1409 ms/call), files changed, tests run.
</compact_output_contract>
<action_safety>
Do not stop or touch running campaigns (Fast-WAM on h100/timan107, Cosmos on h100/timan108) or P3's
processes. No H100 use. Kill only your own PIDs; no pkill -f bare patterns; no rm -rf. Do not touch
/data/i018, /dev/shm/i018, or codex credentials.
</action_safety>
