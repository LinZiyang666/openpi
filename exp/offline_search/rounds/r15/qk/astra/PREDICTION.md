# Frozen predictions before any new-method closed-loop outcome

These are hypotheses, not measured SR. Their purpose is to make later evaluation interpretable. Current offline gates fail, so this document is **not a request to run the candidate**. If a revised model changes the offline behavior enough to pass, version its prediction before the coordinator's approved pilot.

## What the current evidence predicts

1. A very cheap raw-image key is computationally feasible. CPU spatial linear/ridge keys are substantially below a millisecond; the current spatial CNN is around a millisecond before export optimization. The difficulty is unseen-state accuracy, especially long-task phase identity, rather than fitting time or model size.
2. The action-aware symmetric CNN should beat raw RGB PCA+AWM on L10-50 and L10-200, but should not match the current system at off. It was trained from scratch on only the served library; no representation is borrowed from another size. It is close in donor-LOEO and worse on unseen B-dev, so an in-library success does not imply safe deployment.
3. Increasing the library from 50 to 200 should reduce absolute error, but should not eliminate the relative gap to the teacher key. The asymmetric student has a residual teacher/student mismatch; refitting the library metric in the symmetric representation helps somewhat. Direct action prediction cannot reliably disambiguate repeated within-task actions and should not be promoted merely for fitting the supervised action target.
4. GR00T cannot yet be assessed for image-key accuracy. Its missing build-library RGB and original simulator fidelity are separate blockers. Existing GR00T B-dev images cannot train a cell's model under the owner rule. The September traces are A-pool and ineligible.

## Subjective SR forecasts for the current frozen recipe

If the current candidate were evaluated without first improving its offline gate results, my point hypotheses for SR difference versus the current five-layer system are:

| Cell | Off | Same fixed ir25 caps |
|---|---:|---:|
| π0.5 L10-50 | −5 pp; plausible range −12 to 0 | −2 pp; plausible range −7 to +1 |
| π0.5 L10-200 | −3 pp; plausible range −9 to +1 | −1.5 pp; plausible range −6 to +1 |

These ranges are judgmental, not confidence intervals or a fitted mapping from offline error to SR. The corrector refit could change them; phase errors can compound; guards may create additional policy calls. I predict failure of the matched-SR gate for at least one off cell more readily than success. I make no Spatial SR forecast from MISS-only labels and no GR00T SR forecast without exact training frames.

## Cost forecasts with their assumption exposed

With unchanged N/V/M and epsilon at the cost gate, π0.5 L10-50's R14 off arm projects from IR .1689 to .1166; ir25 projects .2359 to .1956. The R14 selected task table projects .2264 to .1843 (18.6% relative reduction). These values are recomputed from accepted B decision logs by `cost_counts.py`, not new rollouts.

The corresponding selected-table GR00T L10 projection is only .4433→.4402 (0.69%). That table calls the policy too often for a cheap key alone to yield a large total reduction. I predict no economically compelling result there at matched settings even if image-key accuracy were solved. At c10, paying a student before every full policy is overhead; bypassing known forced-policy looks can remove that overhead but cannot save the policy's own tower.

Actual extra policy calls spend the saved budget at one full-policy unit each. The π0.5 L10 selected table at the cost gate can afford only **8.35 extra calls per 100 looks** before eliminating its fixed-count savings; the GR00T L10 table can afford **0.61**. Counts may also change through trajectory length and guard behavior. Report actual IR and cost per episode; do not advertise the fixed-count projection as achieved savings.

## What would change my assessment

Existing B-dev all-look labels must show ≤5% equal-task/episode RMS10 excess, controlled tails and phase errors, and the full CPU artifact must clear the other frozen gates. A favorable library-LOEO result alone, fast GPU timing, or an easier MISS-only subset would not do so. A no-go with a useful data/implementation contract is a valid outcome of this offline exploration.
