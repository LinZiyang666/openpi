# Cache-initialization sensitivity: partial results after stopping

The pilot stopped to release devices after 12 of 40 planned episodes. GR00T completed 10 CloseFridge episodes and one PickPlaceCounterToStove episode; pi0.5 completed one CloseFridge episode. These contribute 55 and 5 observations respectively. Four observations from an unfinished episode were excluded for each model. Smoke runs are not pooled with the pilot. No experiments were restarted.

The full policy controlled the environment. At selected observations, model conditioning was held fixed while retrieved clean cached actions, another trajectory's same-task clean cached actions, zeros, and Gaussian noise initialized counterfactual loops. Cache noise level T was fixed at zero; each loop restarted at noise level t=1 and ended at t=0 using N=1 or N=2 forward passes. GR00T's native time direction is reversed; this note uses the common noise-level convention.

The metric is output action RMSE divided by input action RMSE, relative to the retrieved-cache initialization. Only the first five actions and twelve active normalized coordinates enter the metric. Observations are averaged within an episode before averaging episodes. This is a distance ratio, not a fraction of information retained or a success rate. Inputs still contain the full action tensor; this pilot does not isolate the effect of padded coordinates on active outputs.

## CloseFridge

Retrieved cache versus a randomly selected same-task cache from a different trajectory:

| Position | GR00T: 10 episodes | pi0.5: 1 episode |
|---|---:|---:|
| N=1, final | 0.0848 | 0.1474 |
| N=2, first update | 0.5185 | 0.5218 |
| N=2, final | 0.3169 | 0.9232 |

Both models approximately halve the difference after the first of two updates. The second update contracts it further for GR00T but expands it in the single pi0.5 episode. This suggests a hypothesis: GR00T's final output may be less sensitive to cache selection at these observations. It does not establish complete cache erasure or explain the success-rate gap causally.

All control initializations, final distance ratios:

| Model / budget | Other same-task cache | Zero | Gaussian |
|---|---:|---:|---:|
| GR00T / N=1 | 0.0848 | 0.1252 | 0.0258 |
| GR00T / N=2 | 0.3169 | 0.4447 | 0.0767 |
| pi0.5 / N=1 | 0.1474 | 0.1570 | 0.0564 |
| pi0.5 / N=2 | 0.9232 | 0.4763 | 0.2242 |

The separation is control-dependent: the two-step zero-control ratios are similar. Different controls have different input-distance denominators, so comparisons across columns are not comparisons of absolute output error.

For GR00T's same-task cache comparison, episode-bootstrap 95% intervals are [0.0602, 0.1130] for N=1 and [0.2360, 0.3864] for N=2. A single pi0.5 episode cannot estimate reliable between-episode uncertainty; its confidence interval is omitted from the figure. Restricting both models to seed 2,000,000 gives GR00T N=1/N=2 ratios of 0.0408/0.1755 and pi0.5 ratios of 0.1474/0.9232. This remains a descriptive single-episode comparison.

The single GR00T PickPlaceCounterToStove episode gives same-task cache ratios of 0.0256/0.1471 for N=1/N=2. There is no corresponding pi0.5 episode.

## Validation and limits

Accepted worker journals reconcile with launch identities and server terminal records. The incomplete cohort is explicitly marked. All scheduled observations from completed episodes are present. Array SHA256 hashes, finite trajectories, recomputed metrics, and full-policy execution checks passed. Successful and unsuccessful episodes both contribute. GR00T executed four teacher steps and pi0.5 ten.

Identical-input repeat differences were zero. Recorded RNG state and cache payload checks passed. Maximum shared-endpoint Euler-identity errors were 1.19e-7 and 2.38e-7. The CPU analysis tests passed (3 tests); no GPU inference or rollout workers were launched for this analysis.

Counterfactuals share an observation within each model. Across models, full policies visit different subsequent observations and use different cache libraries. These are not fully matched cross-model conditions. Counterfactual actions were never executed in the environment, so this experiment provides no counterfactual success rates. Output sensitivity is not action quality.

![Partial results](partial_initialization_sensitivity.png)

Validated summaries: [GR00T](../../data/init_probe_20260922/pilot/groot/partial_summary.json), [pi0.5](../../data/init_probe_20260922/pilot/pi05/partial_summary.json). Degenerate singleton bootstrap intervals in JSON must not be interpreted as certainty.
