<task>
Owner extends the cache study: repeat all groups for every library size 50, 100, 200, 300, 400 in addition to 500, for all three
models (same official-A evaluation, 500 episodes per group per suite, libero_spatial + libero_10). Update the pre-registration
exp/offline_search/rounds/r17/cache/CACHE_STUDY.md before any size-N<500 data is seen: size factor, how results are reported
(SR vs size curves per group/model/suite, cost: hit rate, MISS/NFE per control step for gated groups), comparisons and their
family (keep it interpretable; say which inferential claims are primary vs descriptive), the near-neighbour sensitivity
(exclude the 32 flagged A inits from CONTAMINATION_CHECK.md), and the analysis script plan. Also extend the analysis to run on
whatever lanes are sealed (so results can be read incrementally). Do not look at outcomes of sizes < 500 (none exist yet).
</task>
<compact_output_contract>
End with: the added pre-registration section summary and the analysis entry point.
</compact_output_contract>
<action_safety>
Read-only on data; docs/scripts only; no GPU, no campaign actions, no codex credentials.
</action_safety>
