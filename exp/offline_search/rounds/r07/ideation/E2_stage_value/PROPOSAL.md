# E2 — Empirical value of compute by stage

## 1. Direction summary

**Recommendation: profile state-deviation stage entries and separate looking from interrupting a committed action. Do not deploy a progress/gripper-based call-suppression table from these data.** We can identify where cache trajectories look abnormal much more convincingly than where an additional policy call repairs them.

I read the R7 brief, R5 Q3 handback, R5 grasp-check handback and analysis, R4 failure anatomy, R6 analysis/repair/frontier tables, the P3 v2 schema and recording code, RiskLottery assignment code, the calibrated-stall handback, and the plugin/profile interfaces. New calculations use all eight uniform arms, all fifteen frontier risk-lottery arms, accepted A/B/P10 paired outcomes, all eight P3 **A** cohorts, and all eight non-test B-val recordings. Other P3 cohorts remain available for the requested follow-up profiler.

What surprised me:

- Closing has 2.55× open-hold cache–policy RMS, but only 1.18× its non-gripper RMS. Much of the apparent difficulty is gripper-command disagreement, which is not a demonstrated need to interrupt the sequence.
- First confirmed stalls in failed cache episodes span the entire demonstration: **16/26/24/22** in progress quartiles Q1/Q2/Q3/Q4. Terminal failure location makes the problem look much later than its first detected symptom.
- An extra look followed by fresh retrieval is not reliably closer to the shadow policy than the existing blind tail. Late-progress fresh retrieval is worse by **0.0295 RMS [0.0113, 0.0416]** on this diagnostic.
- A simple state-deviation quantity strongly marks abnormal cache states, yet its high-minus-low *single-call value* is only **+1.06 pp [−0.98, +3.04]**. Failure prediction and recoverable failure are different targets.

All intervals below are exploratory 95% intervals unless explicitly family-adjusted. No new closed-loop SR or validated safe stage is claimed.

## 2. Preliminary evidence and estimands

### 2.1 Data audit and definitions

| Source | Records used | Independent-state limitation |
|---|---:|---|
| R6 U30/U18 | 4,000 episodes; 82,534 anchors; **80,893 supported anchors / 3,930 episodes** | 500 task/init pairs per cell; deterministic episode doses excluded locally |
| R6 frontier risk lottery | 7,500 episodes; 156,845 anchors; **106,328 supported anchors** | Repeated arms clustered on the same task/init; substantial 0/1 propensity mass |
| P3 A | 480 rollouts; **10,691 anchors**, 10,472 blind observations | 20 distinct task/init pairs per cell, each repeated three times |
| Non-test B-val A | 80 rollouts; **1,817 anchors**, 1,789 blind observations | Ten recordings per cell, one per task |
| A/B/P10 outcomes | Three A/B repeats, one P10 reference, 500 paired initial states per cell | P10 is reused in paired comparisons; not three independent P10 runs |

Sources: [lottery_audit.csv](lottery_audit.csv), [profile_map.csv](profile_map.csv), [look_map.csv](look_map.csv), [paired_outcomes.csv](paired_outcomes.csv). Each source manifest records input path, size, and mtime. All processing was CPU-only, within the assigned affinity, with at most two Python processes.

The lottery is **two-level**: RiskLottery first draws an episode dose, then hashes an anchor coin. I use `os_q2_p_call`, not `os_q2_task_p`, and verify source = coin < logged dose at every extracted anchor: **zero mismatches**. Accepted journal outcomes match the frozen frontier outcomes. The repaired risk runs contain 560 repeated decision keys: selecting the latest complete step-0 prefix before the accepted journal timestamp discards 567 old-prefix rows. Retained episode sequences have no gaps or duplicate decisions; all libraries match the logged library name.

Independent accounting matches **22/23** source cost ledgers exactly. `r6q2_groot_l10_50_risk_rho0p45` has three extra vision/MISS rows in its summary: obsolete steps beyond the repaired successes for task 6, inits 2/12. Accepted client timings confirm 45/47 requests, matching our 23/24 retained anchors. The analysis excludes those stale rows; no shared ledger was changed. All U arms match exactly. [verification.json](verification.json) records this resolved discrepancy, partition checks, source/coin equality, unique joins, and artifact checks.

For a pre-decision stratum S, the primary local estimand is the normalized inverse-propensity difference

`sum(1[S] Z Y/p)/sum(1[S] Z/p) − sum(1[S] (1−Z) Y/(1−p))/sum(1[S] (1−Z)/(1−p))`.

It measures changing **this anchor** from cache to policy, followed by the source arm's continuation, among supported visited anchors. It is not the SR effect of suppressing every call in a stage, and it cannot be multiplied by a stage's anchor count. I also report the first entry into each stratum per episode to reduce repeated-stall occupancy dominance. Final outcome, final length, final dose consumed, and eventual failure never define a treatment stratum.

Bootstrap: 2,000 task-stratified resamples of initial states, repeats/arms kept together, common weights across cells sharing a suite, independent weights across suites; seed 20260930 (+1 for Spatial). Equal-cell pooling avoids letting long LIBERO-10 traces dominate. Per-cell task-cluster t9 sensitivity is in [lottery_effects.csv](lottery_effects.csv). Bootstrap inference treats episodes as independent sampling units conditional on fixed tasks; it is not a robot-transfer guarantee. P3 has only two distinct initial states per task, so its descriptive intervals are especially limited.

**Stage definitions, fixed for this exploration:**

- Progress = retrieved-kernel-weighted demonstration `progress`; Q1–Q4 use each task library's progress quartiles. These are online demo-phase proxies, not elapsed time divided by eventual rollout length or verified physical task completion.
- Open-hold / closed-hold / closing / releasing / mixed refer to the proposed ten-control cache command, including the previous executed gripper command. GR00T and π0.5 use their existing adapter's opposite closed-command signs. Actual grasp/release is not inferred from command sign.
- Gripper state = two library-fitted modes of the first finger state; the mode nearest episode starts is called open-like. These diagnostic labels are not proof of an empty hand or a successful grasp.
- Before first close / after close before first release / after first release use previously issued commands. In U/risk this includes actual policy commands, not rejected cache proposals.
- `state_resid = RMS((current normalized state − weighted retrieved demo state)/library task-state SD)`, over the eight valid state dimensions. High = above the **B-val cell q75**, selected without test outcomes. This is current-anchor deviation, not yet a validated blind-successor deviation score.
- `nonadvance` = weighted demo progress did not increase since the previous anchor. It is a diagnostic distinct from C's calibrated stall. For P3 I join the existing library-only `slow_confirmed/slow_ambiguous/ok` replay by arm/uid/step.

### 2.2 Where failed cache episodes accumulate, and where the first symptom appears

P3 A has 89 failed and 391 successful rollouts. **88/89 failures** ever enter `slow_confirmed`, but so do **95/391 successes**. In GR00T L10-50 alone, **31/34 successful rollouts** enter that state. These are retrospective detection counts, not precision at the time of a rescue decision, and not proof that all 88 were recoverable.

Among those 88 failed rollouts, first confirmed stall is distributed Q1/Q2/Q3/Q4 = **16/26/24/22**: **47.7% are before the library median progress**. Relative to earlier issued commands, it is **14 before first close, 31 after close before first release, 43 after first release**. By contrast, at episode termination 88/89 failures have already issued a release. The latter would badly mislocalize the beginning of trouble. These are symptoms along the observed trajectory; true irreversible failure onset is unobserved.

| Cell | P3 failed rollouts / 60 | First confirmed stall Q1/Q2/Q3/Q4 in failures | Failures ending in Q4 |
|---|---:|---|---:|
| π0.5 L10-50 | 17 | 3 / 1 / 9 / 3; one never confirmed | 6/17 |
| π0.5 L10-500 | 6 | 1 / 4 / 1 / 0 | 4/6 |
| π0.5 Spatial-50 | 16 | 0 / 4 / 6 / 6 | 11/16 |
| π0.5 Spatial-500 | 5 | 0 / 0 / 2 / 3 | 3/5 |
| GR00T L10-50 | 26 | 9 / 11 / 5 / 1 | 5/26 |
| GR00T L10-500 | 12 | 3 / 5 / 0 / 4 | 7/12 |
| GR00T Spatial-50 | 5 | 0 / 1 / 1 / 3 | 2/5 |
| GR00T Spatial-500 | 2 | 0 / 0 / 0 / 2 | 2/2 |

Sources: [failure_localization.csv](failure_localization.csv), [failure_episode_rates.csv](failure_episode_rates.csv). Replicates are not additional initial states. Boundary-degenerate bootstrap intervals, e.g. 2/2, must not be read as certain detection. The source `controls` schema explicitly labels grasp/slip/release **unannotated**; physical contact/object-pose measurements are available for the requested forensic tool, but I have not manufactured actual-grasp ground truth.

### 2.3 The descriptive hard/easy map

Equal-cell means on P3 A; RMS uses valid normalized action channels and the first ten controls. “Failure occupancy” is the fraction of the stratum's anchors belonging to eventually failed episodes. It overweights long failed episodes and is **not** an independent per-anchor failure probability.

| Stage / signal | Mean share of A anchors | Failure occupancy | Cache–policy RMS | RMS excluding gripper | Gripper sign disagreement |
|---|---:|---:|---:|---:|---:|
| Q1 | 22.7% | 21.2% | .092 | .085 | 1.2% |
| Q2 | 23.6% | 26.0% | .175 | .117 | 8.0% |
| Q3 | 27.6% | 34.9% | .197 | .145 | 11.8% |
| Q4 | 26.1% | 31.9% | .236 | .164 | 15.6% |
| Open-hold | 40.7% | 27.0% | .109 | .101 | 1.4% |
| Closed-hold | 40.3% | 25.5% | .183 | .147 | 10.2% |
| Closing | 7.9% | 28.8% | .278 | .119 | 19.9% |
| Releasing | 5.3% | 33.0% | .326 | .158 | 24.6% |
| Mixed close/open proposal | 5.9% | 71.1% | .450 | .244 | 43.6% |
| State deviation above B-val q75 | 21.0% | 70.1% | .367 | .245 | 29.6% |
| State deviation below B-val q75 | 79.0% | 19.7% | .138 | .104 | 5.2% |
| Calibrated slow-confirmed | 14.7% | 87.0% | .412 | .260 | 34.7% |
| Calibrated progress normal | 65.5% | 19.5% | .148 | .113 | 5.6% |

Overlap between signal rows is intentional. Closing versus open-hold RMS difference is **+.169 [.156, .180]**, but only **+.018 [.009, .025]** excluding gripper. State-high versus state-low difference is **+.230 [.210, .249]** RMS and **+50.3 pp [43.0, 56.0]** failure occupancy; confirmed-slow versus normal occupancy is **+67.6 pp [56.3, 71.3]**. Neither contrast is causal.

The same broad action-error pattern appears on non-test B-val: closing/open-hold RMS **.294/.112**, releasing **.333**, state-high/low **.351/.147**. This supports using these categories as diagnostic stages; it does not validate an allocation policy.

Library-computable quantities predict disagreement: equal-cell mean **within-episode** Spearman correlations are neighbour action dispersion **.646 [.630, .663]**, existing predicted error **.395 [.377, .413]**, state residual **.253 [.224, .282]**, coverage rank **.210 [.185, .237]**, and neighbour progress spread **.212 [.191, .232]**. Episode-mean correlations with eventual failure are retrospective associations, not an early-warning predictor; see the separate raw table. No new model is fitted.

Sources: [profile_map.csv](profile_map.csv), [profile_contrasts.csv](profile_contrasts.csv), [correlation_uncertainty.csv](correlation_uncertainty.csv), [profile_correlations.csv](profile_correlations.csv).

### 2.4 Where a randomized call changes eventual success

Primary U-arm, equal-cell, supported-anchor effects in percentage points:

| Pre-call stratum | All-anchor call effect [95%] | First-entry call effect [95%] |
|---|---|---|
| All / episode start for first-entry | +0.33 [−0.23, +0.90] | −0.39 [−2.50, +1.67] |
| Q1 | +0.12 [−0.84, +1.08] | −0.39 [−2.50, +1.67] |
| Q2 | −0.34 [−1.45, +0.67] | −0.36 [−2.42, +1.54] |
| Q3 | +1.33 [+0.21, +2.48] | +1.55 [−0.21, +3.55] |
| Q4 | −0.19 [−1.49, +1.04] | +0.78 [−1.10, +2.79] |
| Open-hold | +0.25 [−0.66, +1.18] | −0.39 [−2.50, +1.67] |
| Closed-hold | −0.00 [−0.86, +0.81] | −0.80 [−2.99, +1.36] |
| Closing | +0.99 [−1.01, +2.90] | +0.32 [−1.83, +2.34] |
| Releasing | −0.11 [−2.65, +2.38] | +1.18 [−1.28, +3.59] |
| Before first issued close | +0.54 [−0.34, +1.39] | −0.39 [−2.50, +1.67] |
| After close, before first release | +0.11 [−0.58, +0.82] | −0.77 [−2.85, +1.39] |
| After first issued release | +0.80 [−2.04, +3.37] | +4.66 [−3.96, +12.60] |
| Nonadvancing retrieved progress | −0.27 [−3.86, +3.40] | +4.45 [−3.56, +12.10] |
| High state deviation | +1.07 [−0.89, +2.95] | +4.36 [−0.30, +9.14] |
| Lower state deviation | +0.01 [−0.49, +0.49] | −0.39 [−2.50, +1.67] |

The relevant interaction is **closing minus open-hold +0.74 [−1.41, +2.79]**, not “closing is significant and open-hold is not.” State-high minus low is **+1.06 [−0.98, +3.04]** all-anchor, **+4.75 [−0.42, +10.15]** first-entry. These do not establish a call-value ranking.

Q3's positive pooled nominal interval becomes **[−0.09, +2.75]** with approximate Bonferroni-normal protection for just four pooled phase tests. No pooled phase survives that small family. One cell-phase effect survives the separate 32-cell/phase family: **GR00T Spatial-50 Q1 −4.82 [−9.54, −0.09]**. Its risk-lottery counterpart is **+2.39 [−1.30, +6.28]** unadjusted, and its U first-entry interval includes zero. This is a continuation-dependent warning, not a portable early-stage suppression rule. The broader exploration family is larger still.

Risk-arm sensitivity also gives first-release-entry versus open-hold **+5.27 [1.58, 9.70]**, while U gives **+1.56 [−1.67, +4.73]**. It is exploratory, has different task/dose visitation, and is not an independent replication on new inits. Some risk-stratum bootstrap intervals are missing because resamples have no treated/control support; those rows cannot certify safety.

Sources: [lottery_effects.csv](lottery_effects.csv), [lottery_interactions.csv](lottery_interactions.csv), [phase_family_adjusted.csv](phase_family_adjusted.csv). Thousands of repeated anchors do not authorize treating them as thousands of independent success labels.

### 2.5 What an extra look tells us—and what it does not

At each recorded blind A decision, compare the actually continued tail and the isolated fresh-retrieval proposal against the **same-observation** shadow policy over the next five controls. Positive values below mean fresh retrieval reduces disagreement.

| Parent-anchor stage | Tail RMS minus fresh-retrieval RMS [95%] |
|---|---|
| All | −.0044 [−.0100, +.0015] |
| Open-hold | +.0044 [.0022, .0066] |
| Closed-hold | +.0043 [.0020, .0067] |
| Closing | +.0098 [−.0017, .0200] |
| Releasing | −.0107 [−.0580, +.0083] |
| Mixed close/open | −.1447 [−.1727, −.0657] |
| Q2 | +.0135 [.0092, .0178] |
| Q4 | −.0295 [−.0416, −.0113] |
| Nonadvancing progress | −.0897 [−.1255, −.0417] |

Fresh retrieval changes gripper sign on **34.2%** of compared controls after releasing proposals, versus **0.76%** after open-hold. Looking can expose disagreement while immediately acting on the new retrieval disrupts the existing sequence. All numbers are proposal diagnostics; fresh proposals were **not executed**, policy disagreement is not action optimality, and an extra look's causal SR effect is unidentified here. The P3 duration experiment changes commitment and action refresh together, so it is not a pure LOOK experiment either.

Sources: [look_map.csv](look_map.csv), [look_uncertainty.csv](look_uncertainty.csv); v2 `diagnostic_cache_chunk`, `executed_chunk`, `policy_chunk`. I used first-five proposed controls uniformly, including proposals at terminal partial decisions; I do not claim they were all physically executed.

### 2.6 Pairing puts a ceiling on rescue stories

| Cell | A failure / P10 success | A success / P10 failure | P10−A, pp [95%] |
|---|---:|---:|---|
| π0.5 L10-50 | 25.4% | 6.4% | +19.0 [15.2, 22.8] |
| π0.5 L10-500 | 14.1% | 6.5% | +7.7 [4.2, 11.1] |
| π0.5 Spatial-50 | 16.3% | 1.4% | +14.9 [12.1, 17.7] |
| π0.5 Spatial-500 | 2.3% | 1.3% | +1.0 [−0.5, 2.5] |
| GR00T L10-50 | 32.3% | 6.8% | +25.5 [21.3, 29.8] |
| GR00T L10-500 | 13.1% | 9.5% | +3.6 [−0.2, 7.8] |
| GR00T Spatial-50 | 12.1% | 5.0% | +7.1 [3.8, 10.5] |
| GR00T Spatial-500 | 2.8% | 5.4% | −2.6 [−5.0, 0.0] |

These are whole-controller contrasts from the same initial states, not local recoverability or a mathematical upper bound on every hybrid. On L10-50, B rescues 17.1%/18.9% of all paired π0.5/GR00T episodes but loses 5.7%/7.5% that A succeeds on. Thus “spend on every failure-associated stage” has a material opportunity to hurt. Dense Spatial offers very little positive net call headroom; its main saving opportunity is the vision floor. See [paired_outcomes.csv](paired_outcomes.csv).

## 3. Method proposals

### P1. Admission-tested call allocation at the first state-deviation stage entry

**Hypothesis.** A fresh departure from the library state distribution may identify recoverable trouble better than persistent disagreement or a late terminal stall. The first-entry U effect is +4.36 pp [−0.30, +9.14]; its all-anchor value is only +1.07 pp. This is a candidate for profiling, not sufficient adoption evidence.

**Mechanism.** Atomic lever: **fewer calls** on ordinary stage interiors, with the same calls moved to anomaly entries. Signal: current proprioceptive deviation from retrieved demo state, plus an entry latch; keep calibrated stall handling as the fixed reference. Allocation: stage/event tilt around the uniform baseline, nested under the existing target-IR **ρ** budget knob. Define stages using the frozen library progress/command labels; define a high-deviation substage using a library-calibrated residual rank. Enter once, then clear the latch on return to the normal region. Never infer “easy means zero calls” from a nonsignificant local estimate.

**Generality and calibration.** Use manifest-valid state/action dimensions, the existing gripper adapter, demo topology, kernel weights, and current proprioception. Fit per-coordinate scales and progress partitions from the deployed library. Calibrate the residual upper quartile by replaying exact retrieval with the source demo episode excluded; at most the ten non-test recordings per cell may check scale/occupancy and solve the budget. The B-val q75 used above is only an initial screen; library-LOEO versus B-val threshold transfer must be profiled. No test-outcome-derived table, task-name exception, or per-cell threshold is deployable. Outcomes assess a frozen rule; they do not calibrate its numeric serving thresholds. Another robot supplies its valid state representation and gripper convention and reruns the same recipe.

**Current disposition and expected frontier effect.** Keep the admission gate closed: no demonstrated generic interaction means the current supported behavior remains uniform placement. If a frozen stage rule later passes, compare at equal IR first, then lower ρ to test a leftward frontier shift. No new SR increment can be estimated by summing the local effects. Per-cell evidence/expectations are in the table below; a strong task-specific finding is not permission for task-specific serving logic.

**Cost tier / online CPU.** Library-only T1 arithmetic after existing retrieval; no additional encoder or policy model. The complete exploratory feature bundle costs **0.135–0.142 ms median, 0.180–0.213 ms p99** CPU across eight cells, excluding retrieval, I/O, and serialization. A residual/latch-only implementation should be smaller, but has not been timed. Existing calibrated-stall computation is separate (R6 measured about 0.55–5.56 ms median), and CPU is outside owner IR.

**Plugin sketch.** `query` computes the frozen residual rank and entry state before drawing the coin; logs actual propensity, entry and phase; returns the unchanged cache proposal plus `os_force_miss` if the budgeted lottery calls. The judge honors that decision. `blind_step` and `policy_tail_step` retain today's committed chunk and lifecycle. The allocation/budget explorer should own the exact tilt/ρ solve; E2 supplies its empirical admission test, not a competing solver.

**Kill criteria.** Drop stage tilt if the entry-versus-interior interaction fails a held-task check and a separately frozen continuation check, if support collapses, or if a matched-IR closed-loop comparison fails to improve SR. Drop a free-state interpretation if the measured current-retrieval residual does not transfer to the intended deployed representation. Do not turn failure of admission into a permissive skip table. These data presently fail admission.

### P2. Observe stage boundaries without automatically interrupting the action

**Hypothesis.** Near gripper transitions, useful information and useful action replacement are different resources. Preserving the micro-sequence while observing state may retain commitment benefits and provide a better signal for the next budgeted call. The negative Q4 and mixed-transition re-retrieval diagnostics motivate this separation.

**Mechanism.** Atomic levers: **look less** in validated stable interiors, preserving observation checkpoints at uncertain stage boundaries; **fewer calls** remains separately budgeted. Keep **look half** as a separately tested lever, with no E2-derived permission yet. Stages are runs of library command modes separated by closing/releasing transitions; state mismatch or calibrated slow/ambiguous progress marks a boundary as uncertain. The first experiment separates (i) continue the committed tail, (ii) acquire vision but continue that same tail, and (iii) acquire vision and replace it. A policy takeover is a separately logged, budgeted action intervention. This is an experimental proposal; no longer blind horizon is certified here.

**Generality and calibration.** Event windows come from the library's within-chunk transitions and `prev/next`, measured in the manifest's execution interval R and supported horizon H. Use LOEO successor-state residual ranks, not a position/grip-width threshold in benchmark units. The kinematic explorer can supply more expressive stage labels; the lazy-lever explorer owns allowable continuation actions/horizons. Another manipulator recalibrates these distributions and declares its gripper/action semantics. Do not transplant the current-anchor deviation threshold into blind successor monitoring without checking it.

**Expected effect / frontier.** The direction of new SR change is unknown in every cell. Best plausible outcome: retain committed behavior and trade interior looks for informative boundary looks at unchanged v, then reduce v. At unchanged trajectory length, adding an extra look on fraction q of ten-control anchors costs **+.076q IR for π0.5 / +.074q for GR00T**. Moving the same number of looks costs no additional owner IR. Extending eligible ten-control intervals to fifteen would save at most **.0253/.0247 IR** if done everywhere; eligible-stage coverage and compensating looks reduce this, and π0.5 would require the untested library-successor lever. These are cost algebra, not SR forecasts.

**Cost tier / online CPU.** Cached-library arithmetic is T1; an actual two-camera look pays the normal .152/.148 per decision, with no policy head unless separately selected. The feature timing above is an arithmetic reference, not a complete blind-hook timing. All-shadow collection latency cannot establish deployment latency.

**Plugin sketch.** `blind_step` tests carried-demo successor consistency and boundary state. For a LOOK-only assignment it requests a named `LookReason`; the following `query` records the new keys/progress but returns the remaining previously committed action, leaving MISS=false. Log a separate assignment for replacement/takeover. Preserve reset, actual-executed-control, horizon, and tail identity checks. Calls continue through the existing judge and committed-policy-tail lifecycle. The new distinction must be explicit in logs; LOOK, retrieval refresh, and policy takeover cannot share one treatment bit.

**Kill criteria.** Drop checkpoint allocation if deviations appear only after failed grasps are already unrecoverable, if observations do not change a subsequent supported decision, or if boundary spending cannot be offset by interior look savings without SR loss. Reject any benefit attributed solely to lower shadow error. Kill a proposed LOOK-only implementation if it changes the currently committed wire actions or silently consumes a policy call.

### Per-cell expectations for both proposals

P1's conservative fallback is the measured U arm; P2 has no validated SR forecast. “Entry value” is a local effect, not the expected whole-controller gain.

| Cell | P1 fallback U SR @ IR | First high-deviation call value, pp [95%] | P2 evidence / expected opportunity |
|---|---|---|---|
| π0.5 L10-50 | .878 @ .301 | +9.0 [−3.2, 20.6] | Uncertain; preserve commitment. Broad call shortage is insufficient: R6 best .894 < pure .904. |
| π0.5 L10-500 | .874 @ .179 | +3.4 [−5.2, 10.6] | Fresh-look action proxy improves .0109 [.0073, .0146]; prioritize observation-value identification. |
| π0.5 Spatial-50 | .940 @ .292 | +5.6 [−19.2, 30.7], only 63 entries | Fresh retrieval worsens proxy .0215 [.0077, .0338]; separate observation from replacement. |
| π0.5 Spatial-500 | .968 @ .179 | +3.6 [−5.7, 11.0] | Low call headroom; conditional vision saving is the plausible frontier direction. |
| GR00T L10-50 | .770 @ .301 | +17.2 [3.3, 32.1], 166 entries | Most promising P1 diagnostic, not multiplicity/transfer validated; fresh retrieval worsens proxy .0105 [.0036, .0173]. |
| GR00T L10-500 | .848 @ .180 | −1.9 [−18.9, 14.3] | Neither new lever has a demonstrated stage advantage. |
| GR00T Spatial-50 | .928 @ .298 | +1.1 [−5.6, 8.0] | Early call effect flips across continuations; retain calibrated-stall comparator, avoid a suite-specific gate. |
| GR00T Spatial-500 | .954 @ .181 | −3.1 [−10.0, 3.1] | A already exceeds P10 in point SR; prioritize vision savings, not more calls. |

U SR/IR: R6 [c_arms.md](../../../r06/analysis_r6/c_arms.md). Entry/local and look intervals are newly computed. No test-derived cell-specific deployment branch is proposed.

## 4. Profile tools wanted

| Priority / tool | Question and inputs | Outputs and proposal decision |
|---|---|---|
| **1 — `stage_value`** | Does a library-defined stage alter causal call value, rather than merely mark failure? Extend this analysis with U/risk accepted logs, P3 randomized cohorts, calibrated stall replay, frozen library stages, actual propensities and continuation identifiers. Start offline; **no new telemetry is needed for the existing U strata or P3 replay**. U lacks the full keys needed to reconstruct exact C-stall status. | Counts/positivity, first-entry and occupancy estimands, fixed-task and held-task uncertainty, phase/event/state interactions, downstream call/control cost, simultaneous bounds, and matched-budget candidate admission. Never fit a serving table from test Y. Decides whether P1 gets implemented beyond logging. |
| **2 — `look_vs_replace`** | At blind states, what changes with observation, cache refresh, or policy takeover? Existing P3/B-val NPZs already support exact action/representation diagnostics, separated into motion and gripper timing. | Per-stage paired errors, sign-transition timing, state-consistency ranks, interior/boundary coverage, and owner-IR accounting. Decides P2's candidate stage definitions. **Causal SR needs new randomized executions** of LOOK-only / LOOK+replace / continuation, with assignment probability, saved and executed wire chunks, actual controls, encoder/MISS counts, stage entry, and continuation logged. |
| **3 — `failure_clock`** | When does the first observed physical deviation occur relative to close/release/stall? Existing P3 controls, snapshots, actions, neighbour successors and episode outcomes. | Timelines joining command transitions, aperture, arm motion, contacts/object motion where measured, first state-residual crossing, first confirmed stall, and terminal failure. Show annotation confidence and missingness. Privileged object/contact labels are evaluation-only, never method inputs. Determines whether P1/P2 signals precede rather than follow failure. No additional simulator collection needed for this forensic pass. |
| **4 — `stage_budget_replay`** | How much real cost can stage allocation move at the proposed ρ? Existing library topology, ten B-val streams per cell, plugin lifecycle and measured actual control counts. | Feasible call/look floors and ceilings, coverage by stage, disjoint cost counts, action-horizon violations, CPU p50/p99. It predicts occupancy/cost on fixed streams, **not new SR**. Decides whether P2 has enough interior saving to finance its checkpoints; shared with the allocation/lazy-lever explorers. |

Implement these as adapters around `profile/breakdown`, `compare`, `timeline`, and `runprof`, reusing the P3 strict reader. New recordings should log a versioned pre-assignment stage dictionary, residual rank/threshold provenance, actual propensity including forced 0/1 cases, separate LOOK/replacement/MISS bits, parent anchor, executed-control timestamps, and the remaining continuation/budget state. Do not add an encoder or learned model.

## 5. Reconciliation, lever disposition, and risks

**R5 Q3 remains intact.** Its first/third eligible guard-landmark experiment returned zero deployable suppression in every held-init and held-task fit. g500 CALL−CACHE was +3.4 pp [0.4, 6.6] by init bootstrap but [−0.04, 6.84] under task sensitivity; g50 was −0.2 pp [−4.0, 3.2]. That estimand is one landmark under an older controller, not all R7 stage interiors. Our local high/low intervals likewise do not establish that repeated suppression costs ≤1 pp. The null fits were lack of a supported saving, not proof that all stages have equal value.

**R6 R=uniform is expected under this map.** Library dispersion, predicted error, and event transitions identify disagreement. They need not identify a recoverable failure or the correct action timing. R6's predicted-disagreement placement yielded −0.15 pp pooled versus uniform, while its calibrated stall package added about +1.15 pp [0.12, 2.23] at matched budget (post hoc). The latter includes changed continuation and extra looks; it does not separately identify a LOOK effect. Our high stall/failure association and weak single-call ranking are compatible with both results.

**The grasp evidence is opportunity, not recovery.** R5 [ideation D](../../../r05/ideation_D/REPORT.md) counted aperture contradiction exposure in 36/97 historical failures, explicitly not 36 recoveries. Its later D1 experiment was unresolved: L10-50 +0.6 pp [−3.0, 4.2] versus K7. These observations motivate `failure_clock`, not unconditional grasp-triggered calls or transplanting old aperture constants.

| Region | Lever disposition supported by this exploration |
|---|---|
| Stable open/closed interiors, normal progress, state consistent | Low disagreement makes **look less** worth profiling. It does not certify **fewer calls** or a longer blind interval. Retain today's ten-control baseline meanwhile. |
| Closing/releasing/mixed proposals | Preserve committed action semantics; test boundary observation separately. Neither “always call here” nor “interrupt every five controls” is supported. |
| High state deviation / confirmed stall | Protect from aggressive unvalidated laziness; test first-entry rescue and observation value. Repeated stalls can be irreversible or benign, so they do not justify unlimited calls. |
| Late demo phase | Failure occupancy is enriched but local call value is unresolved; late fresh retrieval can worsen the action proxy. No terminal-stage skip or call rule is admitted. |
| Dense Spatial cells | Whole-controller evidence favors avoiding gratuitous calls; this is bank-level evidence, not a certified stage label. Vision saving remains conditional research. |
| Any region, **look half** | No stage-specific identification in these data. R4/R5 aggregate wrist results are useful prior evidence but use older retrieval and do not transfer a stage gate to today's A/GR00T. |

Open risks: two-initial-state P3 support; outcome/cost coupling through variable duration; correlated hash domains across repeated arms; lost positivity in risk lotteries; state normalization and angular-coordinate portability; task confounding in descriptive maps; command transitions mistaken for actual object events; disagreement inflated by harmless timing or multimodality; test inits already used in prior design; and fixed-history replay extrapolated to new visitation. The proposed study can fail completely: the correct general solution may remain uniform calls plus calibrated stalls, with stage structure useful only for vision cadence.

## 6. Reproduction

All scripts and summaries are in this directory; compact row tables are in `/tmp/r7_E2_stage_value/`. No serving implementation or shared file was changed. Run from `/home/weiland/projects/openpi`:

```bash
taskset -c 24-25,68-69 bash exp/offline_search/rounds/r07/ideation/E2_stage_value/run.sh lottery
taskset -c 24-25,68-69 bash exp/offline_search/rounds/r07/ideation/E2_stage_value/run.sh profile
taskset -c 24-25,68-69 bash exp/offline_search/rounds/r07/ideation/E2_stage_value/run.sh paired
taskset -c 24-25,68-69 bash exp/offline_search/rounds/r07/ideation/E2_stage_value/run.sh summary
taskset -c 24-25,68-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r07/ideation/E2_stage_value/supplement.py
taskset -c 24-25,68-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r07/ideation/E2_stage_value/analyze_look.py
taskset -c 24-25,68-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r07/ideation/E2_stage_value/time_features.py
taskset -c 24-25,68-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r07/ideation/E2_stage_value/verify_evidence.py
```

`run.sh` applies that exact Python environment/affinity prefix internally. Run sequentially or at most two Python processes concurrently. Sources were read-only. Reported intervals use the final scripts; exploratory correlations with constant outcomes are intentionally NaN, and unsupported bootstrap draws are not evidence of zero effect.
