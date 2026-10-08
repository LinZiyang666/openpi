<task>
Follow-up on your R16 retest tooling (same repo /home/weiland/projects/openpi, same CPU-only rules: no servers/GPU/remote/git/pkill/rm -rf; do not rewrite frozen tables, existing results, v1 roots or v1 outputs).
v1 ran and is analyzed (`exp/offline_search/rounds/r16/retest_v1/`). My selection script had a float boundary bug: 7 tasks exactly 6 pp below full-step pure inference (3 of 50 episodes; computed as -5.999999) were left out of v1. Their changes are in `/home/weiland/projects/openpi_ext/scratch/r16_ops/retest_changes_v2_boundary.json` (7 rows, all disjoint from v1, all `to: ir45`).
Needed:
1. An additive generation v2 that emits groups for ONLY these 7 (cell, task) pairs (new roots `r16_conductor_retest_{model}_k{k}_v2`, new arm names, same identities/UIDs/seeds/pool as before), runnable with the same `retest_loop.sh` queue.
2. A combined analysis that merges v1 and v2 into one adjusted table: tasks from v1 replaced by v1 outcomes, the 7 v2 tasks replaced by v2 outcomes, everything else original test A; same reports/estimators as `retest_analysis` (per cell, size×k, pooled by k, vs full-step pure and vs full-step descent, changed-task list). It must refuse overlapping or conflicting generations.
3. CPU tests on synthetic outcomes; short addendum in HANDBACK_RETEST.md with the exact emit / run / combined-analyze commands. Do the emit for v2 yourself only in a dry-run/validation sense if your tool distinguishes; otherwise leave the emit command for me.
</task>
