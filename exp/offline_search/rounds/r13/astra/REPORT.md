# 给负责人的结论

加入成本反馈后，距离阈值调用会收敛到与周期调用几乎相同的节奏。距离主要是在预算决定的节奏内提前或推迟调用；在实际预算接近时，合并成功率没有可辨认的改善，因此现有证据支持采用更省事的周期方案。开发集的 2,800 个配对结果中，距离方案平均高 0.86 个百分点，置信区间跨过零；这不等于证明两者完全等效。最终测试进度见文末，只作测量确认。

**Technical evidence (B only)**

Budget delivery: 28/28 observed feedback arms pass the ±.02 IR gate; static B recalibration misses 5/8, in both directions. All 28 periodic arms pass; π0.5 L10 mildly underspends, while Spatial does not show uniform underspend. The requested denominator of 32 is not present in gates.json: there are 64 B arms, comprising 28 D + 28 P + 8 S.

D needs a distance scale (two scalars per cell), a bias per cell/dose, a selected shared feedback width, and the running cost ledger. In the full frozen design this is 49 scalars. Its calibration used B trajectories where available and library-held-out transfer for GR00T L10-200 and size-500 (which lacks B pilot validation). P needs one cap per cell/dose, 32 across the full frozen design, from the existing whole-episode-held-out library ScheduleReplay solver; it needs no distance signal or online budget ledger. S uses eight B-calibrated thresholds on the same calibration paths as D, without feedback. These are configuration counts, not completed-arm counts; P is cheaper in calibration/state requirements, with no new latency benchmark claimed.

Compact paper table: all-call gap mean [p10/median/p90] in real looks, excluding startup and censored terminal intervals; D=feedback, P=periodic, S=static threshold without feedback. IR uses the frozen pooled owner ledger. S is not matched cost where its gate fails.

| Cell (.32, rep0) | IR D / P / S | D gap | P gap | S gap |
| --- | --- | --- | --- | --- |
| GR00T L10-50 | 0.321 / 0.329 / 0.313 | 1.61 [1/2/2] | 1.65 [1/2/2] | 1.73 [1/1/4] |
| GR00T Spatial-50 | 0.325 / 0.327 / 0.351 | 1.77 [1/2/3] | 1.65 [1/2/2] | 1.44 [1/1/2] |
| π0.5 L10-50 | 0.322 / 0.309 / 0.331 | 1.63 [1/2/2] | 1.79 [1/2/2] | 1.59 [1/1/3] |
| π0.5 Spatial-50 | 0.314 / 0.322 / 0.393 | 1.83 [1/2/3] | 1.69 [1/2/2] | 1.31 [1/1/1] |

Across .32 rep0 B cells, D/P pooled mean gaps are 1.67/1.72 looks. Held-out call prediction accuracy is 76.7% from gap, 85.1% with the ledger, and 96.5% with residual distance added. Within-episode gap explains only 0.32–2.45% of distance variance. Distance is not a simple time-since-call clock. Yet 66.6% of the same-state selected-position union differs at .32: similar cadence allows different exact placements. Shadow error at D-only/P-only picks is 0.224/0.201, adjusted difference 0.0228 [0.0078, 0.0392]; heterogeneous across cells and not evidence of causal SR benefit.

Paired SR: all 2,800 nominal-dose pairs yield D−P +0.86 [-0.61, +2.29] pp; the 2,600 pairs passing the frozen paired-cost label yield +0.85 [-0.73, +2.35] pp. Both are statistically indistinguishable at this precision, not an equivalence proof. All-pair pooling retains GR00T L10-200 target 0.25 rep0: ΔIR=-0.0136; π0.5 L10-50 target 0.32 rep0: ΔIR=+0.0125. Intervals use 5,000 suite/task-stratified original-init cluster resamples, preserving repeated models/sizes/doses/blocks. Per-cell descriptive results:

| B cell | Pairs (matched) | D SR | P SR | D−P pp [95% cluster CI] |
| --- | --- | --- | --- | --- |
| GR00T L10-200 | 400 (300) | 90.5% | 89.8% | +0.75 [-2.50, +3.75] |
| GR00T L10-50 | 500 (500) | 83.4% | 78.2% | +5.20 [+1.40, +9.00] |
| GR00T Spatial-50 | 500 (500) | 86.6% | 85.8% | +0.80 [-2.40, +3.80] |
| π0.5 L10-200 | 400 (400) | 90.2% | 91.2% | -1.00 [-4.50, +2.50] |
| π0.5 L10-50 | 500 (400) | 86.8% | 88.6% | -1.80 [-5.00, +1.20] |
| π0.5 Spatial-50 | 500 (500) | 93.6% | 92.8% | +0.80 [-0.40, +2.00] |

Pilot host exception: π0.5 L10-50 ran on local 4090 under D2; its other B cells used H100, and GR00T used local 4090. D5 supersedes the large-grid plan. `ANALYSIS.md` gives every B cell/dose/rep gap distribution and sensitivity; `final_paper_table.csv` and `final_b_gap_cdf.csv` are ready for tables/figures. “Periodic suffices” is the practical conclusion at measured precision, with cheaper calibration/state requirements, not a newly measured runtime advantage.

**Test-A measurement — no tuning or selection**

<!-- TEST_A_MEASUREMENT_BEGIN -->

**D5 lean confirmation: COMPLETE (12/12 arms validated at 2026-10-05T08:35:05.467449+00:00).** Fixed by rule before these outcomes: four size-50 cells, target .32, rep0, distance feedback / periodic / off. No tuning, selection, expansion, or stopping rule is inferred from A.

Scheduled by the owner: GR00T on the local 4090 during the night of October 4; π0.5 on H100 after approximately 07:30–08:00 CDT October 5. This schedule is not a claim of observed execution. Missing local summaries/logs remain pending; completed-arm host/timing evidence is in `final_test_a_numbers.json`.

| Fixed cell | Rule | Status | IR | SR | Gap mean [p10/50/90] |
| --- | --- | --- | --- | --- | --- |
| GR00T L10-50 | distance_feedback | complete | 0.3194 | 82.2% | 1.62 [1/2/2] |
| GR00T L10-50 | periodic | complete | 0.3230 | 81.6% | 1.69 [1/2/2] |
| GR00T L10-50 | off | complete | 0.2183 | 72.2% | 2.25 [1/2/5] |
| GR00T Spatial-50 | distance_feedback | complete | 0.3194 | 92.8% | 1.82 [1/2/3] |
| GR00T Spatial-50 | periodic | complete | 0.3197 | 92.0% | 1.70 [1/2/2] |
| GR00T Spatial-50 | off | complete | 0.1233 | 92.4% | 2.16 [1/1/5] |
| π0.5 L10-50 | distance_feedback | complete | 0.3210 | 87.8% | 1.63 [1/2/2] |
| π0.5 L10-50 | periodic | complete | 0.3092 | 88.2% | 1.79 [1/2/2] |
| π0.5 L10-50 | off | complete | 0.1725 | 81.8% | 2.64 [1/1/6] |
| π0.5 Spatial-50 | distance_feedback | complete | 0.3114 | 98.6% | 1.84 [1/2/3] |
| π0.5 Spatial-50 | periodic | complete | 0.3121 | 98.6% | 1.78 [1/2/2] |
| π0.5 Spatial-50 | off | complete | 0.1304 | 93.4% | 1.78 [1/1/4] |

Pooled fixed-set paired D−P: +0.25 [-1.35, +1.80] pp (2,000 episode pairs; 1000 suite/task/init clusters). Nominal-dose contrast; a matched-budget interpretation requires every cell to satisfy |ΔIR|≤.01. Failed cells remain in the fixed pool and are flagged.

| Cell | Pairs | D−P IR | Matched | D−P SR pp [95% CI] |
| --- | --- | --- | --- | --- |
| GR00T L10-50 | 500 | -0.0036 | yes | +0.60 [-3.40, +4.60] |
| GR00T Spatial-50 | 500 | -0.0003 | yes | +0.80 [-2.20, +3.80] |
| π0.5 L10-50 | 500 | +0.0118 | NO | -0.40 [-4.00, +3.00] |
| π0.5 Spatial-50 | 500 | -0.0008 | yes | +0.00 [-1.40, +1.40] |

IR uses the frozen owner cost coefficients and pooled accepted N/V/M, not current latency-derived summary IR. Gaps are between successive actual calls in real looks (guards included), with startup and terminal censoring separate; the figure CSV also has elapsed control gaps. SR pairs use the same suite/task/init; 5,000 paired cluster resamples within suite/task, seed 20261004, jointly carrying both models. The fixed-set pool weights cells equally because every complete arm uses the same full init count. A single rep0 cannot supply leave-one-block sensitivity or the superseded multi-block precision claim. Off is contextual and never used to choose a method.

**Pre-D5 supplementary measurements — all 7 arms, descriptive only.** Their accepted episodes finished before the stop; nothing is selected, promoted into the lean set, or used for tuning. The previously unspecified seventh arm is GR00T L10-500 periodic .32 rep0. The aborted GR00T L10-50 periodic .25 arm is discarded and is never read by this script.

| Prespecified supplementary arm | Episodes | IR | SR |
| --- | --- | --- | --- |
| r13_groot_l10_pure_c5_rep0 | 500 | 1.0000 | 85.4% |
| r13_groot_l10_200_distance_feedback_ir40_rep0 | 500 | 0.3969 | 85.4% |
| r13_groot_l10_200_periodic_ir40_rep0 | 500 | 0.3955 | 89.2% |
| r13_groot_l10_500_distance_feedback_ir25_rep0 | 500 | 0.2410 | 89.0% |
| r13_groot_l10_500_periodic_ir25_rep0 | 500 | 0.2420 | 86.6% |
| r13_groot_l10_500_distance_feedback_ir32_rep0 | 500 | 0.3132 | 87.6% |
| r13_groot_l10_500_periodic_ir32_rep0 | 500 | 0.3152 | 86.6% |

Re-run `final_test_a.py` with the CPU command in `final/HANDBACK.md` after local summaries, journals, manifests and decision logs are available. It replaces only this marked section in ANALYSIS/REPORT, writes `final_test_a_numbers.json`, arm/pair tables and gap CDF CSVs, and refuses frozen-file writes. The B conclusion and all controller artifacts are unchanged.

<!-- TEST_A_MEASUREMENT_END -->
