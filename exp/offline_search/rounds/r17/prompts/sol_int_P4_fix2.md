<update>
Fix round 2. The two Opus reviewers re-checked fix round 1: `exp/r17_integration/review/REVIEW_A2.md` (repros `review/repro_A2/`) and `exp/r17_integration/review/REVIEW_B2.md` (repros `review/repro_B2/`). Read both, fix the findings assigned to you below (plus anything else in your owned area), add a regression test per finding (from the repro where possible), re-run your suite and the legacy/default-off checks, and append a "Fix round 2" section to your handback (finding → change → test → status). Same rules: owned dirs only (cross-package via REQUESTS), zero GPU (trainings until ~18:40/~20:30 CDT), no commit/push, no pkill, no rm -rf, no frozen R16/harness edits, do not touch running processes.
</update>
Your assigned findings:
- REVIEW_B2: DP K1 (MSE 1.77 vs K100, 98% clipped) is unusable. Gate it at binding: reject K1 for cache MISS/teacher roles and default recipes; allow it only behind an explicit, labelled plain step-reduction study flag (we still want to measure its SR as a step-reduction data point). K2/K4/K10 remain valid.
- REVIEW_B2 B11 (partial): qualification fixtures must use the real paired wrist views where available.
