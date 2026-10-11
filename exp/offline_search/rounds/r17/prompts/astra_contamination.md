<task>
Owner asks whether the running cache study is contaminated (testing on the cache's own data). Verify rigorously, read-only.
Evaluation: cache lanes /data/mw_finetune/eval/csl10d (DP), csl10f (Fast-WAM), later csl10c (Cosmos): per_step records say
init_pool A, partition test, orig_init_state_idx 0..49 per task (official LIBERO test inits), libero_spatial + libero_10.
Libraries: /data/r17_library/<bundle>_<suite>/bpool/ full size_500 (e.g. study_inputs/<suite>/parent/READY.json in each lane
dir; metadata.episodes[] carry init_pool B, partition library, init_state_sha256, orig_init_idx, task). The B pool was built as
the R10 "difference pool" (excluding A).
Do:
1. Compute the A test init states exactly as the formal benchmark loads them (same LIBERO init files / benchmark spec the
   eval used) and hash them with the same function used for library init_state_sha256 (find it in the library/teacher
   capture code). Report exact overlap between A (spatial+l10, 1000) and every library episode's init (DP, Fast-WAM, Cosmos x
   2 suites x 500), per model/suite.
2. Near-duplicate check: per task, minimum distance from each A init state to any B init state in the raw state vector
   (object poses etc.), compared with typical A-A and B-B nearest-neighbour distances; flag any A init closer to a B init than
   the 1st percentile of B-B NN distances.
3. Also confirm the library rows contain only B-pool episodes (no A episodes leaked from earlier campaigns) and that the
   corrector/stall fits used only B data (CACHE_STUDY/OWNER_RULINGS; check fit inputs).
Write exp/offline_search/rounds/r17/cache/CONTAMINATION_CHECK.md.
</task>
<compact_output_contract>
End with: exact-overlap counts per model/suite, near-duplicate flags, library/fit provenance verdict, and one-line conclusion.
</compact_output_contract>
<action_safety>
Read-only; no GPU, no campaign actions, no codex credentials.
</action_safety>
