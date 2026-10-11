<task>
The local 4090 replica slots `slot_G2_dp_libero_R_replicas` and `slot_G2_groot_mw_R_replicas` returned PARTIAL: "runtime source changed after CPU deployment; refresh payload: exp/mw_finetune/dp/policy.py, serve_compat.py, serve_libero.py, exp/r17_integration/benchmarks/libero_bridge.py, debug/reader_profiles.py, qualification/assets..." These files changed legitimately: E's CUDA graph, D's Cosmos, P6's library pipeline, and your formal launcher.
Runs: `/data/mw_finetune/qualification/runs/slot_G2_{dp_libero,groot_mw}_R_replicas_20261009_234121_*`.

Refresh the local qualification deployment/payload and seals to the current sources (your existing refresh/deploy path). Re-run the relevant CPU checks, then give the exact coordinator command to rerun both replica slots exclusively on the 4090. The 4090 is idle now, so you may run the two replica slots yourself if they are admissible; they are exclusive and short.
Do not touch H100: Fast-WAM LIBERO B is running there with its own runtime_q1 copy. Do not touch running Codex work by D or P6.
</task>
<compact_output_contract>Append "## Payload refresh 10-09 night" to HANDBACK_Q.md with results. Final message: 3 lines.</compact_output_contract>
