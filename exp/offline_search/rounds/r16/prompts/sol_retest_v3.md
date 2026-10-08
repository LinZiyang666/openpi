<task>
Context: read first  (incl. addendum), , , , ,  in the same directory; v1 = , v2 =  (both run and analyzed; combined v1+v2 output ).
Follow-up on R16 retest tooling (repo /home/weiland/projects/openpi; CPU only: no servers/GPU/remote/git/pkill/rm -rf; never rewrite frozen tables, v1/v2 roots, plans or outputs). Needed within ~25 minutes; I start the GPU run at 16:30 CDT.
The same float boundary bug also hit the 10 pp threshold: 7 tasks exactly 10 pp below full-step pure inference were raised to ir45 in v1, while the owner rule says c10. List: `/home/weiland/projects/openpi_ext/scratch/r16_ops/retest_changes_v3_c10.json` (7 rows; `from` = original frozen level, `to` = c10, `supersedes` = v1). These pairs OVERLAP v1 by design.
1. Generation v3 that supersedes v1 for exactly these 7 pairs: emit groups for only them (new roots `r16_conductor_retest_{model}_k{k}_v3`, new arm names, same identities/UIDs/seeds/pool), runnable with `retest_loop.sh`. Validate `from` against the original frozen table and that each pair exists in v1 with `to: ir45`.
2. Combined analysis over v1 + v2 + v3 where a later generation explicitly supersedes an earlier one for the listed pairs only (v3 outcomes replace v1 outcomes for those 7; everything else as before), refusing any other overlap. Same reports/estimators; changed-task list shows the final level and keeps the superseded v1 result for reference.
3. CPU tests, the exact emit / run / analyze commands appended to HANDBACK_RETEST.md. Leave the emit for me.
</task>
