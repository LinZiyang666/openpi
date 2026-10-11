# Official Cosmos LIBERO inference

The formal launcher CPU-seals a `LoadSpec`, `ModelManifest`, benchmark and
disabled-cache `full.yaml`. `descriptor.py` emits the same standalone files;
`serve.py` uses the shared connection factory, bundle validator and stock wire.
Each resident serializes B1 queries; replicas share traffic through the stock
proxy. Every decision encodes only the current camera pair and raw quaternion
state. Weights, normalization and sealed prompt embeddings are resident.

Use the NVIDIA experiment `cosmos_predict2_2b_480p_libero__inference_only`,
native JPEG95/224/crop preprocessing, wrist+agentview, 9-D proprio, 16 actions,
16 applied controls per query, K5 EDM2ab, fixed diffusion seed195. The official
runner has `num_queries_best_of_n=1`; there is no default seed search. Its image
operation is `flipud`; the adapter undoes the stock worker's extra horizontal
flip before native preprocessing. It retains native float64 physical actions.

Sources at vendor commit `18a2accadf4e7a3531e56754102af5a24d2316da`:
`LIBERO.md:46-73`, `experiments/robot/libero/run_libero_eval.py:170-235,324-338,396-400,449-462`,
`experiments/robot/libero/libero_utils.py:44-57`,
`experiments/robot/cosmos_utils.py:851-1237`, and
`config/experiment/cosmos_policy_experiment_configs.py:90-193`
(all paths below `cosmos_policy/`, except `LIBERO.md`).

The per-camera VAE key hook is implemented. The explicit R17 cache study now
admits sealed B500 fits at native K5/C16, either zero ordinary MISS or only
calibrated Q3 stall calls. Corrector and gate use the shared framework math;
Cosmos's four long labels alias the corresponding C16 groups. Teacher-to-current
release admission requires pinned weights, stats, prompts, native producers,
output transform and the explicit B-to-A benchmark bridge. CPU evidence is in
HANDBACK_COSMOS.md; GPU cache hot-switch qualification remains pending. Other
recipes, generic CP1 cache admission, warm resume/reset, CP2, snapshots, B>1
stage batching and CPU/GPU split replay remain unqualified/rejected. Minimal
teacher capture is supported separately. K1/K2/K3/K5 preserve native
sampling. K2（含清理步） seals `cosmos_edm_2ab_k2_incl_clean` and passes
`num_steps=2` unchanged; the upstream one-step branch executes actual NFE1.
See HANDBACK_COSMOS.md § K2 incl. clean step for parity and schedule evidence.
No model-specific training or fine-tuning recipe is supplied.
