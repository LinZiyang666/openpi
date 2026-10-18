Coordinator: the DP concurrent-server equivalence runs are done, two cold-start rounds per GPU, all rc=0:
- 4090 (05:59 and 06:03): exp/offline_search/rounds/r17/xgpu_1017/evidence/run_concurrent_4090_dp_r2_{1,2}.log, results under xgpu_1017/results/.
- H100 (06:12 and 06:14, after stage_h100.sh): evidence/run_concurrent_h100_dp_{1,2}.log; result files on the H100 under /home/exouser/xgpu_1017/results/ (pull them back the way your plan says).
Analyse DP: same-GPU concurrent noise floor per GPU, 4090 vs H100 differences, cache-decision agreement, against your criteria. Then give me the next commands (next model on 4090/H100) with memory and minutes. The H100 is busy with formal runs again from about 06:20; give H100 steps that fit beside them (state the GPU memory each needs).
