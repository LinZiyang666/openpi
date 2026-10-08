# B-only offline evidence

Computed here by `extract.py`, `replay_study.py`, `robust_study.py`, `static_study.py`, `signal_quality.py`, `periodic_and_power.py` and `selfcheck.py`. This evidence shaped the freeze at 2026-10-04T16:40:30.172320+00:00. All writes are in this directory; no policy model, GPU, simulator, server, remote command or test-A data was used. `*_reads.json` inventories the data access. The Python audit fence rejects unapproved run roots, all non-library store inputs, mixed historical A/B analysis artifacts and writes outside this directory.

## Existing B evidence and reconstruction audit

`inventory.csv` records the contract-listed roots, completed episodes and look counts. `data/` contains only compact accepted B traces. Attempts are reconciled against accepted journals, episode decision sequences are contiguous, and reconstructed N/V/M equal the root's cost ledger. Pool contracts prove each accepted init is outside the actual served subset; artifact and shard hashes are verified.

At the existing nominal `.32` setting, B independently reproduces the old distance budget shortfall:

| cell | periodic B IR | old distance B IR | periodic B SR | old distance B SR |
| --- | --- | --- | --- | --- |
| pi05_l10_50 | 0.3116 | 0.2129 | 0.8800 | 0.8400 |
| pi05_spatial_50 | 0.3095 | 0.2162 | 0.9600 | 0.9400 |
| groot_l10_50 | 0.3143 | 0.2646 | 0.8800 | 0.7600 |
| groot_spatial_50 | 0.3209 | 0.1919 | 0.8600 | 0.8400 |

These SR columns are **unequal-budget B measurements**, not evidence about matched-IR superiority. They are not a basis for selecting test cells. The cells/doses cover the contract's requested models, suites and size range uniformly.

The recording field named `distances` is negative wrapper Result score, which in this serving path is a kernel-transformed value with nearest score zero. It is unsuitable for calibrating raw distance. This analysis replays the exact deployed `_dist` on saved camera keys/state with the full served bank and inherited early/stale metric. No image encoder is run. The cache proposal is reconstructed from recorded parent donor rows/weights and the unchanged deployed corrector; cache HITs verify the reconstruction. Maximum absolute hit-action discrepancy among shadow-labeled sources is 1.78813934e-07. It is float arithmetic drift, not an alternative cache policy.

Policy MISS `action` and `shadow_policy` can be identical; comparing those would falsely assign zero call value. This analysis always compares shadow policy against the reconstructed corrected cache proposal, including at guard MISSes. The normalized motion label is mean squared difference over the first ten controls and first six action channels divided by the frozen cell action scale. Gripper sign disagreement is retained separately. Shadow policy samples are one stochastic same-stage-one action label, not ground-truth recovery utility.

## Making the distance knob honest

First fit one static raw-distance threshold per cell/dose using only the calibration identities and minimizing worst error across available B behavior paths. On complementary development identities it passes the ±`.02` band on 32/44 correlated checks; mean absolute error is 0.0134, worst 0.0447. Better calibration helps, but a single threshold remains sensitive to the behavior path. This supports a dev-only static checkpoint rather than making it the primary knob.

An off-path-only fitted feedback bias also failed a path-transfer check. `replay.csv` preserves that attempt. The final fitting procedure uses one cell-wide minimax bias across available **calibration** paths, without mixing their outcomes into fitting. The complementary identities select one global feedback width:

| width | B checks within band | mean abs IR error | minimum / mean interior share | constant-score placement change mean | synthetic max error |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 44/44 | 0.0041 | 0.222 / 0.395 | 0.380 | 0.0242 |
| 1.0 | 44/44 | 0.0038 | 0.428 / 0.593 | 0.530 | 0.0289 |
| 2.0 | 44/44 | 0.0040 | 0.616 / 0.748 | 0.517 | 0.0440 |
| 4.0 | 44/44 | 0.0043 | 0.729 / 0.851 | 0.532 | 0.0751 |
| 8.0 | 43/44 | 0.0061 | 0.815 / 0.922 | 0.555 | 0.1293 |

Choose width `1.0`: 44/44 nominal B checks, MAE 0.0038, worst error 0.0188. The threshold lies inside score support on at least 42.8% of eligible looks across these checks, averaging 59.3%. Replacing distance by a constant changes on average 53.0% of the union of selected non-guard positions (minimum 24.4%). This is evidence of distance engagement, not evidence of beneficial placement.

The synthetic tests multiply every validation distance by `.5, .75, 1.25, 1.5` without changing guards/actions. For the chosen width, 213/224 such checks stay inside the band; worst error is 0.0289. These deliberately broad distortions include library-only cells. They are scenario tests, not statistical uncertainty bounds. Widths with more interior thresholds are less robust to this shift; the smaller width spends accurately with substantially more saturated, score-insensitive decisions. This tradeoff is why budget control and signal engagement are both pilot gates.

The development checks overlap across doses and paths and helped select width. Their count is not a sample size, and passing them does not establish closed-loop target delivery. No replay changes future states, success, length or guard events in response to the hypothetical actions. The untouched B pilot and then frozen A measurement are needed for those claims.

## Signal quality: library versus live B cache states

| cell | B source/population | looks | pooled rank correlation | within-episode mean | top-quartile error lift |
| --- | --- | --- | --- | --- | --- |
| pi05_l10_50 | library_whole_episode_heldout/all_even_anchors | 1475 | 0.418 | 0.262 | 1.603 |
| pi05_l10_200 | library_whole_episode_heldout/all_even_anchors | 5766 | 0.348 | 0.252 | 1.676 |
| pi05_l10_500 | library_whole_episode_heldout/all_even_anchors | 14844 | 0.329 | 0.218 | 1.744 |
| pi05_spatial_50 | library_whole_episode_heldout/all_even_anchors | 591 | 0.210 | 0.101 | 1.143 |
| groot_l10_50 | library_whole_episode_heldout/all_even_anchors | 1486 | 0.113 | 0.162 | 1.422 |
| groot_l10_200 | library_whole_episode_heldout/all_even_anchors | 5891 | 0.193 | 0.120 | 1.378 |
| groot_l10_500 | library_whole_episode_heldout/all_even_anchors | 14921 | 0.195 | 0.110 | 1.358 |
| groot_spatial_50 | library_whole_episode_heldout/all_even_anchors | 613 | 0.181 | 0.054 | 1.174 |
| groot_l10_50 | r12_devlooks_50L/all_looks | 3384 | 0.319 | 0.151 | 1.951 |
| groot_l10_50 | r12_devlooks_50L/guard_eligible | 2299 | 0.220 | 0.141 | 1.623 |
| groot_l10_50 | r12_devlooks_50L/guard_calls | 1085 | 0.399 | 0.207 | 2.281 |
| pi05_l10_50 | r12_devlooks_50L/all_looks | 3198 | 0.445 | 0.294 | 2.370 |
| pi05_l10_50 | r12_devlooks_50L/guard_eligible | 2444 | 0.322 | 0.259 | 2.004 |
| pi05_l10_50 | r12_devlooks_50L/guard_calls | 754 | 0.572 | 0.359 | 2.868 |

Correlation is Spearman on looks; the within-episode column averages episode correlations with enough nonconstant observations. Top-quartile lift is the fraction of episode-weighted action error captured divided by the episode-weighted mass selected by the pooled distance top quartile. `signal_quality.csv` also contains task-demeaned rank correlation. Report all-look and guard-eligible results separately: the guard already claims some high-error looks, so its overlap cannot be credited as new layer-four value.

The present shadow comparison covers both L10 size-50 cells on knob-off trajectories. Distance retains a positive live action-error association there, including on guard-eligible looks; the signal has not simply vanished in closed loop. Within-episode associations are weaker than pooled ones. These measurements do not establish which method rescues failures or whether the shadow disagreement is necessary for task success. Spatial, size-200 and active-method shadow evidence remain requested; they are not invented from labels that are absent.

`distance_paths.csv` separates actual B behavior paths and their score/guard distributions. The normalization and scalar solver use only permitted B evidence. `development_splits.csv` and `used_b_identities.json` make the next genuinely independent selection auditable.

## Limits and next measurements

No closed-loop claim has been made for the new variant. Size-500 has no external B complement. GR00T size-200 currently has library-only initialization. The owner ledger is a normalized cost index, not wall-time latency; incremental controller CPU latency remains sol's integration measurement. Four-replicate precision scenarios in `power.json`/`grid_counts.json` expose the dependence assumption instead of multiplying arm counts into an artificial effective sample size.

The early REQUESTS.md was written at 2026-10-04T16:22:29.095237+00:00; the final request addendum binds it to the frozen arm files. `PLAN.md` fixes the later Q1–Q6 analyses. `TEST_DESCRIPTIVE.md` records that no new test-A descriptive analysis was performed in this phase. ANALYSIS.md and REPORT.md are reserved for a later coordinator resume after the frozen measurement grid runs.
