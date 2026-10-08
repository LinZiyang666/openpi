# Prospective prediction

Timestamp: **2026-10-04T16:40:30.172320+00:00**. Frozen before reading any closed-loop result of distance_feedback or distance_static_b. No A traces, outcomes or query arrays enter this prediction. The R11 aggregate statements supplied by the brief are background; newly computed numeric forecasts use B evidence and explicit subjective priors only.

Every arm, including every replicate and off/pure control, has a row in `predictions.csv`. The point forecast for feedback IR is the user's target; the planning band is target ±`.02`, an acceptance aim **not** a statistical prediction interval. The table's replay range is the observed range over complementary development episodes and behavior paths with fixed states, not a future closed-loop range. Periodic predictions are from the unchanged library solver. Repeated arms share a point forecast; repeats are not promised identical realizations.

| cell | target | periodic IR forecast | feedback IR forecast | feedback fixed-path dev range | distance − periodic SR prior (pp) |
| --- | --- | --- | --- | --- | --- |
| pi05_l10_50 | 0.25 | 0.2500 | 0.2500 | 0.2445–0.2688 | +1.00 |
| pi05_l10_50 | 0.32 | 0.3200 | 0.3200 | 0.3176–0.3344 | +1.00 |
| pi05_l10_50 | 0.4 | 0.4000 | 0.4000 | 0.4001–0.4064 | +0.50 |
| pi05_l10_50 | 0.45 | 0.4500 | 0.4500 | 0.4497–0.4522 | +0.00 |
| pi05_l10_200 | 0.25 | 0.2500 | 0.2500 | 0.2477–0.2477 | +1.00 |
| pi05_l10_200 | 0.32 | 0.3200 | 0.3200 | 0.3185–0.3185 | +1.00 |
| pi05_l10_200 | 0.4 | 0.4000 | 0.4000 | 0.3982–0.3982 | +0.50 |
| pi05_l10_200 | 0.45 | 0.4500 | 0.4500 | 0.4489–0.4489 | +0.00 |
| pi05_l10_500 | 0.25 | 0.2500 | 0.2500 | 0.2459–0.2459 | +1.00 |
| pi05_l10_500 | 0.32 | 0.3200 | 0.3200 | 0.3174–0.3174 | +1.00 |
| pi05_l10_500 | 0.4 | 0.4000 | 0.4000 | 0.3997–0.3997 | +0.50 |
| pi05_l10_500 | 0.45 | 0.4500 | 0.4500 | 0.4492–0.4492 | +0.00 |
| pi05_spatial_50 | 0.25 | 0.2500 | 0.2500 | 0.2380–0.2553 | -0.50 |
| pi05_spatial_50 | 0.32 | 0.3200 | 0.3200 | 0.3121–0.3215 | -0.50 |
| pi05_spatial_50 | 0.4 | 0.4000 | 0.4000 | 0.3980–0.3989 | -0.25 |
| pi05_spatial_50 | 0.45 | 0.4500 | 0.4500 | 0.4509–0.4518 | -0.00 |
| groot_l10_50 | 0.25 | 0.2500 | 0.2500 | 0.2487–0.2660 | -1.00 |
| groot_l10_50 | 0.32 | 0.3200 | 0.3200 | 0.3185–0.3286 | -1.00 |
| groot_l10_50 | 0.4 | 0.4000 | 0.4000 | 0.3985–0.4026 | -0.50 |
| groot_l10_50 | 0.45 | 0.4500 | 0.4500 | 0.4501–0.4517 | -0.00 |
| groot_l10_200 | 0.25 | 0.2500 | 0.2500 | 0.2399–0.2399 | -1.00 |
| groot_l10_200 | 0.32 | 0.3200 | 0.3200 | 0.3140–0.3140 | -1.00 |
| groot_l10_200 | 0.4 | 0.4000 | 0.4000 | 0.3966–0.3966 | -0.50 |
| groot_l10_200 | 0.45 | 0.4500 | 0.4500 | 0.4477–0.4477 | -0.00 |
| groot_l10_500 | 0.25 | 0.2500 | 0.2500 | 0.2514–0.2514 | -1.00 |
| groot_l10_500 | 0.32 | 0.3200 | 0.3200 | 0.3186–0.3186 | -1.00 |
| groot_l10_500 | 0.4 | 0.4000 | 0.4000 | 0.3985–0.3985 | -0.50 |
| groot_l10_500 | 0.45 | 0.4500 | 0.4500 | 0.4498–0.4498 | -0.00 |
| groot_spatial_50 | 0.25 | 0.2500 | 0.2500 | 0.2443–0.2604 | -0.50 |
| groot_spatial_50 | 0.32 | 0.3199 | 0.3200 | 0.3108–0.3265 | -0.50 |
| groot_spatial_50 | 0.4 | 0.4000 | 0.4000 | 0.3975–0.4018 | -0.25 |
| groot_spatial_50 | 0.45 | 0.4499 | 0.4500 | 0.4481–0.4520 | -0.00 |

The central value prior favors distance slightly in pi05 L10 at the lower doses, periodic slightly in GR00T L10 and Spatial, and approaches a tie at the highest dose. These are deliberately weak hypotheses: each listed SR difference has a subjective ±`3` pp plausibility range, including zero. They are not fitted success predictions. Pi05's stronger B shadow error ranking motivates its sign; GR00T's weaker eligible-look ranking and periodic's guard-aware spacing motivate the opposite sign. Error ranking cannot establish rescue value, so a null result remains plausible everywhere, including at dense libraries.

I predict B calibration plus cost feedback will substantially improve budget delivery over the old frozen distance rule. I do **not** predict that it will dominate periodic's SR/IR frontier in every cell. High-dose gains should be smaller as both methods approach the common policy-at-every-look endpoint. Size-500's target delivery is especially uncertain because no full-bank B dev calibration is possible. A guard-heavy episode or changed short-episode distribution can still miss the target.

There is no absolute SR forecast for library-only off arms or pure controls: no same-host, same-batch B measurement establishes it here. Available B off SR values in `predictions.csv` are explicitly source-labeled planning anchors, not A estimates. Pure C10 IR includes a library-based terminal-truncation estimate; pure C5 IR is `1` by its ledger definition. These controls must be measured fresh in the same test batch.

The freeze is SHA-bound in `MANIFEST.sha256`; do not overwrite it after a pilot failure. Any redesigned variant needs its own version and prediction timestamp before its new closed-loop results.
