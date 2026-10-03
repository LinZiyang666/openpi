给负责人的预测：替换方案去掉了按任务选择的纠正头，离线保留了大部分动作误差改善，但尚未追平原纠正器。预计闭环成功率接近或略低于同批对照；不把动作误差当成成功率结论。

Frozen before arm emission: **2026-10-02 10:37:06 UTC**.
Selection was already frozen by training-only grouped CV in `SELECTION.json`.
This prediction uses the admitted 20–29 offline shadow evaluation, not new closed-loop results.

## Frozen experiment

Eight arms, four paired cells, ten tasks × inits 20–29 = 100 episodes per arm.
Fit uses shadow labels from inits 0–19 of A/CU/IP only. No new fitting or selection on 20–29.
All candidates use shared observation RFF ridge (768 features, alpha 100), followed by a library-row residual ridge (lambda 1), at strength 0.5. The gripper is unchanged.
The same configuration applies to all four cells. No task ID/one-hot/head dispatch/per-task scale in the new corrector.
Current per-task heads remain only in the explicitly requested controls.

| Cell | Same-batch control | Predicted candidate minus control SR | Predicted owner IR change |
|---|---|---:|---:|
| π0.5 LIBERO-10-50 | existing guard + half corrector + escalation (lag 12, deadline 80) | −0.02 | +0.003 |
| GR00T LIBERO-10-50 | existing guard + half corrector | −0.02 | +0.004 |
| π0.5 Spatial-50 | existing guard + half corrector | −0.01 | +0.002 |
| GR00T Spatial-50 | existing guard + half corrector | −0.01 | +0.002 |

These are subjective point forecasts, not estimates derived from a validated SR model. Plausible paired SR differences extend roughly ±0.05 around each forecast. The new method adds no calls or looks directly; changed actions can change later guard decisions and episode lengths.

## Predeclared decision rules

- Primary practical screen: average paired SR difference across four cells ≥ −0.02, no cell below −0.04, and average owner IR increase ≤ 0.01. Call this **screen retention**, not statistically established noninferiority.
- Stronger result: candidate SR ≥ control in all four cells with average IR increase ≤ 0.005.
- Report every cell, all discordant pairs, paired init-cluster bootstrap intervals, and incomplete/failed episodes. Do not choose tasks, drop difficult tasks, or replace controls with earlier-batch results.
- Owner IR: π0.5 = 0.152 × look fraction + 0.848 × call fraction; GR00T = 0.148 × look fraction + 0.852 × call fraction. Use pooled accepted decisions and report episode count/decision count too.
- If a cell loses more than four successes per 100, reject the universal replacement claim even if the four-cell average looks acceptable.

No closed-loop launch, sync, server, worker, GPU or network operation was performed for this package. Coordinator commands are in `HANDBACK.md`.
