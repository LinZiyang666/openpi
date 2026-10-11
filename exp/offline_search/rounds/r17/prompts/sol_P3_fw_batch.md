<task>
Owner question (verbatim): 「fast wam 为什么那么慢，为什么不能用batch之类的方式提上去，h100的功耗都没有顶满」.
The Fast-WAM LIBERO formal campaigns run 2 resident replicas on H100 at B=1. The official baseline yields about 5–6 episodes/min. nvidia-smi shows 97% util, but power is far from the cap: small-batch underutilization.

Measure and enable batched serving for fastwam_libero in the R17 framework:
1. Profile one B=1 decision on GPU: time split VAE encode / video prefill / action-head K10 / other; GPU power; SM occupancy if cheap.
2. Server-side batching across concurrent connections. Use the framework's stage batching core (`src/openpi/serving/batching_core.py`, S07) and/or your `exp/r17_integration/models/fastwam/batcher.py`. Measure B=1/2/4/8/16: throughput (decisions/s), per-request latency, peak VRAM, power, and numeric deviation of normalized actions vs B=1 on recorded LIBERO observations (max abs / typical).
3. If throughput gains are large, add a default-off formal-launcher option (e.g. `--max-batch N` with a short batching window) for fastwam_libero, plus DP and Cosmos if the same mechanism applies cheaply. Batch composition must be recorded per decision.
   - Note: batching changes numerics slightly. Paired studies must then run control and arms with the SAME batching config. Report what that implies for the currently running B=1 Fast-WAM baseline.

GPU: use the local 4090 (48 GB, idle now) for ≤25 minutes, holding `/tmp/r17_gpu.lock` (`flock -w 600 /tmp/r17_gpu.lock …`) so the coordinator's DP queue waits. Release it promptly. Fast-WAM replica is about 26 GB. Do NOT touch H100 (running baselines) or any running process.
</task>
<compact_output_contract>Write `exp/r17_integration/models/fastwam/BATCHING.md` (profile table, B sweep table, numerics, recommendation, exact launcher command). Final message: 5 lines with the throughput multiplier at the recommended B.</compact_output_contract>
