Coordinator: I ran your first 4090 command twice at 04:54 (logs: exp/offline_search/rounds/r17/xgpu_1017/evidence/run_concurrent_4090_dp_{1,2}.log). Both exited in about 1 s before any server started:

ValueError: MPS pipe path too long; use a shorter campaign run name (pipe <=85 bytes)
  at formal_runtime_5927.../exp/r17_integration/models/fastwam/ops/mps.py:101 via concurrent_server.py:226 mps.start()

Fix the harness so its MPS directory/run name keeps the pipe path <= 85 bytes on every host (4090, H100, A100) and for every model (compute the actual path the frozen mps.py will build and assert it in a CPU test), without changing the frozen runtime. Then give me the corrected 4090 command; check that nothing from the failed attempts is left running (MPS daemons, ports 29600+). Keep everything else in your plan unchanged.
