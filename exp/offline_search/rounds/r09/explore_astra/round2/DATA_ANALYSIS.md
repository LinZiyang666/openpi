# Round-2 evidence and reproducibility

## Population firewall and split

The first read was the owner's discovery-only `RESULTS_R9.md`. Round-1 reports/code were read next. No other researcher's round-2 work was read, and no locked run root was opened.

All statistical inputs come from **24 existing discovery-only compact NPZs** under `offline_search_store/derived/r09_astra/compact`: eight pure-cache, eight uniform-plus-stall, four independent-call, and four π0.5 wrist-every-ten arms. `tools/data.py` checks the companion discovery provenance and loads identity arrays first. It rejects any init outside 0–29 **before loading outcome or action members**; it then explicitly applies the discovery admission mask. This is intentionally narrower than a general raw-capture reader. It does not open mixed-init journals, full-run summaries, or the holdout roots.

All new model parameters, normalization, random features, and call thresholds use **0–19 only**. Every reported model evaluation uses **20–29 only**. Each trajectory in every source arm follows the same init split. No round-1 fitted residual is loaded for prediction. Existing library fits are frozen shared infrastructure; they are not newly task-routed or task-calibrated.

For physical diagnosis only, `physical_audit.py` obtains exact episode directories from trusted discovery compact decision IDs, selects **20–29 before opening anything**, verifies each episode UID, and reads only those episodes' control files. It never enumerates the raw reader's all-episode outcomes. Eight hundred evaluation episodes were inspected. Object/goal labels are privileged diagnostics, not model inputs.

`PROVENANCE.json` records source compact SHA256s and output hashes. `FROZEN_CANDIDATES.json` independently ties every serving pickle to its fitted head. Source R8 captures and round-1 artifacts were not modified. The H100 dependency planner was run locally to hash/list deployment assets; it did not launch or collect an experiment.

## Measurements and uncertainty

- Motion error: mean squared difference from the policy shadow over ten controls and channels 0–5, in existing normalized model-action coordinates.
- Gripper error: fraction of the ten controls whose command sign differs from the shadow. Gripper scores use signed regression and are not calibrated probabilities.
- Average decisions within each arm/episode, then average episodes. Training also weights arm/episodes equally. Long failures do not get more total weight merely because they have more decisions.
- Intervals resample the **ten evaluation init IDs jointly across all ten tasks**, 3,000 draws, seed 20261002. They are exploratory intervals for this fixed development sample, not simultaneous/post-selection certificates. Repeated control bits are never presented as independent trials.
- IR is sum of recorded cost divided by sum of decisions. Candidate cost is a cadence-based expectation; changed episode length and first/last rounding can change it.
- A shadow is a stochastic imitation label, not a transition or value oracle. Corrected next states, objects, contacts, future retrieval, and success are unobserved. The new SR field is deliberately “unidentified.”

## Exact new evaluation baselines

These are **R8 inits 20–29**, 100 episodes per entry, not the owner's all-discovery R9 table and not locked confirmation.

| Cell | Pure-cache SR @ IR | Uniform-plus-stall SR @ IR | Independent-call SR @ IR |
|---|---:|---:|---:|
| π0.5 long-50 | .730 @ .07645 | .860 @ .30381 | .820 @ .18718 |
| π0.5 long-500 | .860 @ .07669 | .920 @ .18444 | unavailable |
| π0.5 Spatial-50 | .790 @ .07728 | .900 @ .30344 | .890 @ .18575 |
| π0.5 Spatial-500 | .960 @ .07817 | .970 @ .16542 | unavailable |
| GR00T long-50 | .620 @ .07434 | .800 @ .30050 | .730 @ .18199 |
| GR00T long-500 | .870 @ .07450 | .840 @ .18192 | unavailable |
| GR00T Spatial-50 | .910 @ .07554 | .950 @ .29808 | .920 @ .19170 |
| GR00T Spatial-500 | .980 @ .07538 | .950 @ .17075 | unavailable |

These different populations/topologies can give materially different factual SR. In particular, the supplied GR00T long-50 R9 cache is .583 on 0–29; this R8 evaluation cache is .620 on 20–29. Do not splice them into a paired effect estimate.

## Shared correction: supervision first, short memory second

Four fixed configurations were fit in all eight cells: pure-cache-only nonlinear, mixed-path nonlinear, mixed-path nonlinear with 16 history features, and mixed-path linear. Their half/full motion and gripper ablations are all retained, including weaker results. Hyperparameters were fixed before this evaluation. Choosing a proposal from these results is development selection; it does not make 20–29 a pristine final holdout.

Across the 20 full-camera arm/cell populations, there are **86,740 fit anchors in 4,000 arm/episodes**, and **43,103 evaluation anchors in 2,000 arm/episodes**. A sparse cell has 600 fit arm/episodes; a dense cell has 400. Each still has only 200 unique task/init pairs, shared across path types. Cache-only labels total 200 episodes per cell. The added paths supply extra policy shadows and visited observations beyond the nominal 50/500-demo library.

| Cell | Cache-path half-motion error reduction: cache-trained / mixed / mixed+history | Uniform-path reduction: cache-trained / mixed / mixed+history |
|---|---:|---:|
| π0.5 long-50 | 37.6 / 37.3 / 37.5% | 14.9 / 21.2 / 21.9% |
| π0.5 long-500 | 8.0 / 9.0 / 8.7% | 4.2 / 5.2 / 5.7% |
| π0.5 Spatial-50 | 43.9 / 43.9 / 44.1% | 22.8 / 29.0 / 30.5% |
| π0.5 Spatial-500 | 14.8 / 16.3 / 17.0% | 8.3 / 10.1 / 9.6% |
| GR00T long-50 | 44.4 / 44.6 / 45.4% | 19.8 / 24.5 / 26.4% |
| GR00T long-500 | 6.6 / 8.9 / 10.9% | 8.2 / 9.3 / 11.4% |
| GR00T Spatial-50 | 34.9 / 35.2 / 37.5% | 19.5 / 23.9 / 25.9% |
| GR00T Spatial-500 | 7.9 / 8.0 / 9.7% | 6.7 / 7.0 / 7.7% |

The history inputs are current-minus-previous fresh robot state (eight), previous uncorrected cache-chunk mean (seven), and capped gap (one). They reset per episode and never use the next observation. On deployment, the model stores the uncorrected retrieved chunk for matching feature semantics. There is no simulator state or goal-progress input.

The table supports mixed-path supervision more strongly than a particular temporal architecture. Linear heads are weaker in these fixed comparisons; this does not prove 768 random features are optimal. The independent-call evaluation paths also improve (24.2%, 37.7%, 32.5%, 34.5% for π0.5 long/Spatial, GR00T long/Spatial sparse cells).

## Conservative gripper corrections: real local signal, limited coverage

Below is the joint candidate, inits 20–29 pure-cache paths. Error percentages are episode-balanced; affected episodes and better/worse counts avoid pretending that hundreds of correlated control bits are independent successes.

| Cell | Gripper disagreement before → after | Affected episodes / 100 | Better / worse episodes | Median first correction, policy controls |
|---|---:|---:|---:|---:|
| π0.5 long-50 | 8.75% → 7.55% | 20 | 19 / 1 | 250 |
| π0.5 long-500 | 5.12% → 5.03% | 4 | 3 / 1 | 185 |
| π0.5 Spatial-50 | 12.28% → 8.83% | 34 | 34 / 0 | 90 |
| π0.5 Spatial-500 | 2.35% → 2.34% | 1 | 1 / 0 | 160 |
| GR00T long-50 | 13.35% → 12.03% | 33 | 32 / 1 | 220 |
| GR00T long-500 | 5.72% → 5.69% | 3 | 2 / 1 | 270 |
| GR00T Spatial-50 | 7.22% → 5.63% | 17 | 16 / 1 | 120 |
| GR00T Spatial-500 | 2.05% → 2.03% | 3 | 2 / 1 | 80 |

Long-50 gripper-error differences have exploratory 95% intervals of **[−1.894,−.527] pp** for π0.5 and **[−1.957,−.760] pp** for GR00T. These are action-label differences, not SR intervals. Dense changes are tiny and some intervals include zero.

The same head on uniform-call paths reduces gripper error only .212 pp / .158 pp in the two sparse long cells. π0.5 long uniform-path proposed changes agree with the teacher on 75.8% of changed controls, versus 96.2% at each episode's first change. This supports testing cumulative overrides separately. It does not support saying that a .8 score means 90% reliable behavior.

Raw cache/teacher disagreement is much larger in failed long-50 episodes: π0.5 24.96% versus 2.75% in successful episodes; GR00T 25.41% versus 5.96%. Outcome-conditioned diagnostics are not deployable gates. Long stalled tails, wrong modes, and teacher recovery can all produce this association; it does not identify a causal gripper failure onset.

## Physical diagnosis of long failures

The audit measures recorded goal predicates, including initially true predicates separately. “New goal” below requires a predicate false at the first policy-control record to become true later. This avoids counting initially satisfied goals as new progress.

| Long-task cell | Failed eval episodes | At least one newly achieved goal | Any transient predicate loss | End below best number of goals |
|---|---:|---:|---:|---:|
| π0.5 50 | 27 | 77.8% | 37.0% | 11.1% |
| π0.5 500 | 14 | 71.4% | 21.4% | 7.1% |
| GR00T 50 | 38 | 50.0% | 18.4% | 7.9% |
| GR00T 500 | 13 | 38.5% | 38.5% | 7.7% |

The earlier “any satisfied predicate” statistic is 46.2% for GR00T long-500; **38.5%** is the stricter newly achieved statistic. Losing already completed goals occurs, but is not the dominant terminal signature. A long task can complete one object and never complete the next without lowering the number of satisfied predicates. These descriptive data motivate sequential phase/mode investigation, not a fitted task switch. The audit is on original cache paths, not the later corrected paths; it cannot establish why any individual corrected episode failed.

## Wrist-only distillation

`wrist_head.py` uses 143 input features and the same fixed half correction. Training anchors are 13,862 / 12,019 / 5,201 / 4,433 for π0.5 long-50 / long-500 / Spatial-50 / Spatial-500. Each combines 200 wrist-path and 200 pure-cache-path episodes from 0–19. Pure-cache-path actions are wrist retrieval shadows, except the first full-camera anchor. Teacher labels are existing policy shadows.

On **actual wrist-only evaluation paths**, motion MSE changes are −39.9%, −14.5%, −43.4%, −20.8%. At those same states the corrected wrist head has lower error than a fresh full-camera cache retrieval by 45.4%, 11.1%, 39.9%, 13.6%. This is a same-observation comparison; it is not an SR comparison between controllers.

The uncorrected wrist-controller SRs are .630, .730, .790, .970 at IR .03370, .03387, .03600, .03737. The first two are substantially below ordinary cache SR. This is why the short dense cell is the sensible first cost-saving confirmation, and why a large same-state error improvement is insufficient evidence for long-task preservation.

The serving fit asserts equality of the original and wrist PCA bases. The head consumes only wrist PCA, and local tests exercise both initial full-camera and subsequent wrist-camera requests. The source shadows can still differ numerically from live encoder/retrieval execution; local code parity is not a remote encoder-parity test.

## Randomized call-value estimation: mostly negative

All selected independent-call anchors satisfy actual propensity .25 and treatment=`coin<p`. The learner uses pre-coin vision/state/cache/history/diagnostic inputs, never policy-shadow disagreement as a value target. A pooled outcome nuisance is cross-fitted within training inits, then a regularized head learns centered inverse-propensity pseudo-outcomes. Training 75th-percentile score thresholds are fixed before evaluation.

The estimand is the population effect of changing **one call at first threshold entry**, with the original randomized future controller thereafter. Non-reaching episodes contribute zero. It is not the value of repeatedly deploying the gate.

| Sparse cell | First-entry learned-value call effect, SR units | Exploratory 95% interval | Local probability-tilt derivative |
|---|---:|---:|---:|
| π0.5 long | −.116 | [−.403,+.130] | −.050 |
| π0.5 Spatial | +.016 | [−.040,+.071] | −.036 |
| GR00T long | +.089 | [−.098,+.288] | +.639 |
| GR00T Spatial | +.005 | [−.068,+.073] | −.220 |

The GR00T long derivative interval is [.197,1.290]; the Spatial interval is [−.467,−.006]. These are local SR derivatives per unit probability change, not attainable SR improvements; neither extrapolate them to a deterministic trigger nor multiply by a large budget shift. Four moderators and two estimands were explored in four cells, without multiple-testing correction. No learned-call candidate is promoted from this evidence.

High-distance first-entry effects are +.100 on π0.5 long but −.125 on GR00T long. Their apparent pointwise intervals do not justify a universal “call when far” rule. Low-motion and predicted-failure triggers also lack consistent beneficial effects.

## Commands and checks

From the repository root:

```bash
R9R2_PY=(taskset -c 10-21,54-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
R9R2_MOD=exp.offline_search.rounds.r09.explore_astra.round2.tools
"${R9R2_PY[@]}" -m "$R9R2_MOD.shared_head"
"${R9R2_PY[@]}" -m "$R9R2_MOD.call_value"
"${R9R2_PY[@]}" -m "$R9R2_MOD.physical_audit"
"${R9R2_PY[@]}" -m "$R9R2_MOD.wrist_head"
"${R9R2_PY[@]}" -m "$R9R2_MOD.prepare_confirmation"
"${R9R2_PY[@]}" -m unittest "$R9R2_MOD.test_tools" -v
"${R9R2_PY[@]}" -m "$R9R2_MOD.audit"
```

The scripted equivalent is `bash tools/reproduce.sh`. It does not access remote machines or launch closed loop. Analyses retain all ablations in `results/`; `shared_head.json`, `wrist_head.json`, `call_value.json`, and `physical_audit.json` are the primary numeric records.

Local preparation validated **28 candidates × 20 real-library queries = 560 queries**, including early/ordinary retrieval, both wrist modes, exact corrected tails, original padding, zero-correction identity, history reset, and clone isolation. Online wrapper output equals the standalone predictor exactly in these checks. Median head time across candidates is .084 ms on this CPU; the largest candidate median is .089 ms. This is not an H100 end-to-end latency measurement.

Eleven unit tests cover locked-identity rejection before payload reads, allowlists, split separation, temporal causality/parity, episode weighting, regression recovery/evaluation isolation, gripper thresholds and padding, first-entry selection and the HT estimand, init-cluster intervals, NI conservatism, and emitted manifest/head contracts. During development, the first real-query probe lacked history fields and one synthetic rejection fixture lacked its sentinel file; both harness issues were fixed. Final test and serving-check logs are retained. No original serving code was changed.
