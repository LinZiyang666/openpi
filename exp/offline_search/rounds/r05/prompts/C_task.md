<task>
You are R5 ideation agent C: the library as a model component — growth, pruning, routing and GPU residency.
R4 showed library density matters (per-neighbour phase continuation fails at 50 episodes but works at 500; anchor_tail
works at both) and that retrieval can live on the GPU inside stage 1's CUDA graph (rounds/r04/k9_gpu_retrieval/:
in-place append keeps graph addresses). Study:
1. Online self-growth: start from the 50-episode library and insert what deployment produces (policy MISS chunks with
   their keys/state; possibly successful cache segments). Using the recorded closed-loop decisions (MISS rows carry the
   policy chunk and the query inputs), simulate growth offline and estimate how fast (episodes of deployment) the
   50-library approaches 500-library behaviour; what to insert, what not (failed episodes, stalls), dedup, and the
   fit/calibration consequences (PCA/whitening frozen vs refit; K9 requires frozen bases for in-place append).
2. Pruning / coreset: from usage statistics over ≈1.6 M logged decisions (which rows are ever top-16, which lead to
   stalls/traps), how small can the 500 library get at equal closed-loop behaviour; bytes vs deployed pkl.
3. Routing / priors: task-level and scene-switch priors (owner's earlier idea), library partitioning, cross-task
   donors only where measured useful (C2 found cross-task donors rarely win).
4. GPU deployment details that change method design (fixed capacity, tie policy, float precision).
Label every borrowed-information fit; forecast SR / IR at both scales; give R5 arms.
</task>
