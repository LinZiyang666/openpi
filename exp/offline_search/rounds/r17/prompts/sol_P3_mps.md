<task>
Owner: CUDA MPS may be enabled on both the h100 and the local 4090 if it speeds things up. Find out whether it does,
and if so make it a launcher option.

1. Implement an opt-in `--mps` for formal_eval server nodes: start nvidia-cuda-mps-control with a PRIVATE pipe/log
   directory per GPU per campaign (CUDA_MPS_PIPE_DIRECTORY / CUDA_MPS_LOG_DIRECTORY under the campaign dir), launch
   that campaign's replicas with those env vars so only they attach, record MPS state in the receipt, and shut the
   daemon down on campaign close (including failure paths). Processes that do not set the private pipe dir must be
   unaffected — verify that the running campaigns keep their contexts. Do not change the GPU compute mode.
2. Measure, bounded (<= 20 GPU minutes total across both GPUs), whether MPS raises throughput for our real cases:
   (a) 4090: DP 1 replica B16 vs 2 replicas B8 with and without MPS; Fast-WAM 1 replica B8 vs 2 replicas with/without
   MPS if memory allows; (b) h100: Fast-WAM 2 replicas B8 + Cosmos 2 replicas sharing the GPU, with/without MPS.
   Use your batch_profile / P4's DP load generator with realistic concurrency (16-32 clients). The GPUs are shared
   with live campaigns (h100: Fast-WAM baseline hs3_fast_* 2 replicas + Cosmos resume; 4090: DP baseline hs3_dp_*),
   so report relative A/B on the same moment's background, interleaved, and the background campaigns' own rate
   change during your probes.
3. Check exp/offline_search/rounds/r17/horizon/STUDY.md: does turning MPS on mid-study affect any decision metric
   (success rate must not; is cost measured by calls/NFE or by wall time?). State whether MPS can be switched on for
   the remaining arms without rerunning the baselines.
Others: sol S is recording live performance (ops/perf_recorder.py) and sol R is writing ops/RUN_CONFIGS.md — append
MPS findings to exp/r17_integration/REQUESTS/P3_to_R.md. Q edits ops/ for prestaging; check diffs, keep edits narrow.
</task>
<compact_output_contract>
End with: A/B table (config, MPS on/off, calls/s, p50 latency, GPU util/W, background campaign rate change),
recommendation per GPU, files changed, tests, and whether remaining arms can use MPS against existing baselines.
</compact_output_contract>
<action_safety>
Do not stop, signal or reconfigure running campaigns; do not set exclusive compute mode; do not touch other users'
GPU processes. Honor /tmp/r17_gpu.lock semantics on the 4090 only for exclusive-timing work (shared probes OK).
No pkill -f bare patterns, no rm -rf, no /data/i018, /dev/shm/i018, codex credentials. timan1 forbidden.
</action_safety>
