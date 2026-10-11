<task>
Implement the teacher-library pipeline for the new LIBERO models in the R17 framework: collect, build the 500 library, cut nested subsets, then fit. Models: `dp_libero` and `fastwam_libero` now; `cosmos_libero` when sol D lands its bundle (`exp/r17_integration/handbacks/HANDBACK_COSMOS.md`). Keep the code generic over bundles. Suites: libero_spatial and libero_10. π0.5 and GR00T already have libraries and are out of scope.
</task>

<owner_decisions>
These are fixed; do not re-litigate them.
1. Inits. Use the SAME B pool ("差集池") of 500 inits per suite (10 tasks × 50) that the π0.5/GR00T libraries used. The official A pool (500/suite) stays test-only and is never fit.
   - References: `/home/weiland/trace_runs/offline_search_store/library/{pi05,groot}_{l10,spatial}/{bpool_cs,bpool_all}/episodes.json`. These hold `orig_init_state_idx` 0..49 per task and the source h5 paths.
   - Find the B-pool init-state files those builds used (the `--init-states-dir` of the R10/ablation collection). Prove identity: init-state arrays must be byte-equal to what the π0.5 library episodes started from.
   - GR00T spatial lacks `orig_init_state_idx`; resolve it via π0.5 / the build dirs and verify.
2. Episode alignment. The library episode id for (task t, B-init j) must equal the π0.5/GR00T id, so the R10 nested subsets apply unchanged: `exp/offline_search/rounds/r10/subsets/{pi05,groot}_{l10,spatial}_{50,100,200,300,400,500}.json`, `permutation_by_task`, seed 20261002 (π0.5 and GR00T permutations are identical). Sizes: 50/100/200/300/400/500. Small libraries are row-index views into the 500 parent, not array copies.
3. Collection.
   - Teacher at full native sampler (DP100, Fast10, Cosmos5); cache/retrieval off.
   - Run through the formal launcher infrastructure (`ops/formal_eval.py`: R17 conductor per-episode dispatch + resident replicas behind the stock replica proxy), with debug capture ON (CaptureBridge/DecisionSink per `ops/RUNBOOK.md`).
   - Keep ONLY what library build + fits need: per decision the retrieval key fields of the bundle's FeatureSpec, the teacher normalized action chunk (and physical if the build needs it), state, task/episode/init/step/decision ids, success/progress/next-prev links, AND both camera images exactly as the model received them (uint8; the owner explicitly wants images kept). Drop everything else at write time, not after.
   - Failed episodes are kept (as in R10).
4. Fits per library size, using only that size's library (hard data rule from R15 BRIEF):
   - the same fit products R16 cells used: Mahalanobis metric, leave-own-episode-out corrector, no-progress guard thresholds, caps/knobs, as defined by P1's LayoutRecipe (`common/layers/recipe.py`, `fit_library`);
   - at 500, document leave-own-episode-out serving as R15 did;
   - outputs sealed with save_fitted_recipe.
</owner_decisions>

<deliverables>
- Library campaign mode: `formal_eval.py --library --suite {libero_spatial,libero_10}` or a sibling module. B-pool 500 inits, 1 trial each, debug allowlist sink, sync back (SHA-checked /tmp stage), build the 500 library under `/data/r17_library/<bundle>_<suite>/bpool/` (not /home; /data has space), then subsets and fits.
- An alignment audit: for every (task, init), our episode id = π0.5 id, and the init state is byte-equal.
- A disk estimate per bundle × suite (keys + images), and a CPU-only end-to-end test using a tiny synthetic capture.
- Exact coordinator commands: GPU smoke (2 episodes per suite), then full collection per bundle × suite (DP on the local 4090; Fast-WAM and Cosmos on H100 + timan108).
</deliverables>

<constraints>
- CPU only now. The 4090 and H100 are running formal evals (DP-LIBERO 400, Fast-WAM MT50, DP-MW MT50, π0.5 K1). Do not start GPU processes; the coordinator runs the smoke when a GPU frees.
- Do not touch running processes or ports 23150–23179. Do not modify `src/`.
- Fast-WAM LIBERO has not passed GPU A/B qualification yet; note that the library run for it waits on that.
</constraints>

<compact_output_contract>
Append "## LIBERO library pipeline" to `exp/r17_integration/handbacks/HANDBACK_P6.md` with files, B-pool provenance proof, alignment audit result, disk estimate, and the exact commands. Final message: 6 lines max.
</compact_output_contract>
