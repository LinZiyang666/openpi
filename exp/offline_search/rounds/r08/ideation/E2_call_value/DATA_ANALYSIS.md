# R8 callback — E2: call value and look value

R8 now supports auditable local call experiments and same-observation look diagnostics. It does **not yet validate a stage-based call allocation rule**. Physical placement is a discovery candidate; longer blind ages show greater refresh opportunity on observed paths. Missing assignment-time stages, incorrect motion/gripper separation, and a mixed-version augmentation discrepancy require qualification of the stock reports.

## 1. What I ran: commands, arms, coverage

Run: `/home/weiland/trace_runs/os_closed_loop/r08_main`. All results below use **inits 0–29 only**, ten tasks per suite. I did not inspect holdout outcomes or fit thresholds on inits 30–49. Thresholds were frozen before this callback analysis. The reader wrapper masks episodes and decisions before passing them to tools; control access asserts discovery membership and disables reader caches.

Primary scope: all **20 CU/CT/IP arms**, 300 discovery episodes each:

- `r8_{groot,pi05}_{l10,spatial}_{50,500}_{CU,CT}`: 16 arms.
- `r8_{groot,pi05}_{l10,spatial}_50_IP`: four arms; no IP-500 arm exists.
- Total: **6,000 arm-episode observations**, 600 distinct suite/task/init pairs reused across arms; **247,543 decisions**, **127,341 fresh anchors**, **1,227,073 active controls**.
- Physical heuristics were available for 6,000/6,000 episodes; 200/200 prefix checks preserved earlier physical stages after truncating the future. Reader issue lists were empty in all 20 arms.

Augmentation-dependent scope is **partial**, with `AUG_DONE` required before every read. The selected arms are:

```text
r8_groot_l10_50_A       r8_groot_l10_50_FL
r8_groot_l10_500_A      r8_groot_l10_500_FL
r8_groot_l10_50_CU      r8_groot_l10_50_CT
r8_groot_l10_500_CU     r8_groot_l10_500_CT
r8_groot_spatial_500_CU r8_groot_spatial_500_CT
```

These are 3,000 discovery arm-episodes. No π0.5 or IP shadow result is claimed. The four A/FL arms provide the extended-age comparison; the six CU/CT arms cover every augmentation-complete arm in the primary call scope at selection time. **GR00T L10-500 CU fails anchor replay parity in retained older augmentation parts; its pooled shadow conclusions are withheld (§4).** Its augmentation-independent call results remain included.

Across these ten arms, the tools produced served/policy comparisons for **165,754/165,754** decisions, fresh-cache comparisons at **79,114** anchors, **70,712/70,712** eligible cache-blind comparisons, and policy-noise estimates at **5,267/165,754** sampled decisions. These are availability counts, not equivalence certificates. Applied-prefix anchor equality holds at **76,335/79,114** anchors; every mismatch belongs to the older CU-500 parts described below.

All analysis scripts are in this ideation directory; all intermediate data are in `/tmp/r8cb_E2_call_value/`. No serving code, trace files, simulators, workers, GPU jobs, or git state were changed. At most three Python processes ran, each with this prefix:

```bash
taskset -c 14-17,58-61 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python
```

Append the following script paths/arguments to that prefix, from the repository root:

```text
exp/offline_search/rounds/r08/ideation/E2_call_value/callback_calls.py --model pi05
exp/offline_search/rounds/r08/ideation/E2_call_value/callback_calls.py --model groot
exp/offline_search/rounds/r08/ideation/E2_call_value/callback_shadows.py --arms r8_groot_l10_50_A r8_groot_l10_50_FL r8_groot_l10_500_A r8_groot_l10_500_FL
exp/offline_search/rounds/r08/ideation/E2_call_value/callback_shadows.py --arms r8_groot_l10_50_CU r8_groot_l10_50_CT r8_groot_l10_500_CU r8_groot_l10_500_CT r8_groot_spatial_500_CU r8_groot_spatial_500_CT
exp/offline_search/rounds/r08/ideation/E2_call_value/callback_look_summary.py
exp/offline_search/rounds/r08/ideation/E2_call_value/callback_look_summary.py --arms r8_groot_l10_50_CU r8_groot_l10_50_CT r8_groot_l10_500_CU r8_groot_l10_500_CT r8_groot_spatial_500_CU r8_groot_spatial_500_CT
exp/offline_search/rounds/r08/ideation/E2_call_value/callback_summary.py
exp/offline_search/rounds/r08/ideation/E2_call_value/callback_report_tables.py
exp/offline_search/rounds/r08/ideation/E2_call_value/callback_aug_audit.py
```

The wrappers invoke the supplied `call_value`, `stage_ledger`, `trigger_vs_onset`, `divergence`, `follow_vs_look`, and physical `forensics` functions directly to enforce the split. Native call reports retain 1,000-bootstrap intervals; supplemental first-entry comparisons use 10,000 bootstraps, and look summaries use 3,000. Per-arm call/physics work took roughly 67–190 seconds. Main artifacts: `calls/<arm>/{summary,call_value_stock,call_value_r7_stage,call_value_truth_pre,stage_ledger,trigger_vs_onset}.json`; `shadows/<arm>/`; and `summary/{arms,support_by_stage,centered_first_entry,interactions,pooled_effects,paired_CT_CU,stage_costs,stage_overlap,look_summary}.csv` plus `audit.json` and `report_numbers.json`.

## 2. Do the data and tools answer the requested questions?

Statuses refer to my original proposal; “partial” separates functioning calculations from missing scientific identification.

| Requested tool → supplied tool | Verdict | What works; concrete remaining gap |
|---|---|---|
| `value_audit` → reader/common + call support/provenance audit | **Works for call assignment; partial overall** | Effective probabilities, coins, execution, accepted episodes and clocks join correctly. I regenerated every fresh-anchor hash. The added anchor-parity audit found mixed-version augmentation failure in one arm. Assignment-time stages are absent; debug-on-off invariance is not certified by these checks. |
| `divergence_clock` → `divergence` | **Partial** | Same-input served/cache versus policy shadow, applied-prefix flags and sampled policy variance are available. Completion alone does not establish replay equivalence: the tool pools older/newer augmentation versions in CU-500. Native `motion_rms` includes the gripper because manifests omit its index; gripper flips are unavailable. Native curves are decision-weighted; episode-balanced uncertainty needs the supplement. |
| `look_age` → `follow_vs_look` + supplement | **Partial, useful** | Blind age, tail versus fresh retrieval, kernel overlap and member provenance are available. The supplement compares both proposals with the same teacher and separates motion. No action discrepancy is a measured rescue; age groups have different trajectory/eligibility populations. Native member-spread motion labeling has the same gripper problem. |
| `call_excursion` → `call_value` | **Works for supported excursions; partial stage integration** | Correct effective-p exclusion, first supported entry, original-population zeros, branch ESS, downstream work and unnormalized probability-shift derivatives. Stock capture yields only causal stage `unknown`. I supplied audited pre-coin/pre-control labels. Outcome centering, direct stage contrasts and family-wise intervals require supplemental analysis. |
| `stage_ledger` | **Works for live compute; partial physical attribution** | Dispatch counts, additive stage work, actual control exposure and separate request/control IR reconcile. It reports current decision-stage costs, not continuously changing physical stages inside a served block. Parent/trigger-stage attribution is not supplied by my reconstructed labels. |
| `look_package_value` → FL / `exposure_hazard`, A5 comparisons | **Partial design coverage; not run here** | FL randomizes extension packages, and A5 enables whole-controller cadence comparisons. Neither is the proposed three-way continue / fresh-cache / policy intervention at the same blind decision. A stage-local physical look-benefit contrast remains missing. |
| `stage_validation` → physical labels + supplemental interactions | **Partial** | Fixed physical stages can moderate a later random coin. Cross-cell comparisons and multiplicity are feasible. There is no validated deployable physical-stage detector, no held-out validation here, and no demonstrated transferable call-value split. |
| `restore_audit` | **Partial metadata only** | The contract retains `restore_certified=false`. I did not execute restoration or inspect a new certificate. Matching deferred retrieval prefixes is not a simulator/controller/server restoration certificate. |
| `branch_value` | **Missing certified experiment** | No certified matched branch rollouts were supplied or run. Shadows provide alternative actions, not alternative success outcomes. |
| Additional `trigger_vs_onset` | **Partial** | Exact decision/control clocks, exposure denominators and heuristic-onset joins work. A failure-conditioned heuristic onset is not the moment failure became irreversible; successful episodes with alerts may be rescues, so `false_alert_rate` is not detector specificity. |

The data are much more useful than R7's for this lens: every fresh call can be audited, physical features precede the current treatment, and unused refresh proposals are available at actual blind states. The remaining gaps are consequential but identifiable.

## 3. Findings for stage segmentation and lazy-lever allocation

### Assignment support is correct, but targeted CT states lose overlap

All 127,341 fresh-anchor hashes exactly reproduced the logged coin. Effective-p formulas, `coin < p`, `treatment`, `policy_calls`, and HIT/MISS source agreed with **zero mismatches**. For CU/CT, cooldown takes precedence at p=0; confirmed stall takes p=1; otherwise p is the logged nominal value. IP uses p=.25 throughout.

| Four-cell group; 1,200 episodes each | Fresh anchors | Interior-p support | p=0 | p=1 | Supported p range |
|---|---:|---:|---:|---:|---|
| CU-50 | 25,995 | 20,518 | 2,668 | 2,809 | .4948–.5022 |
| CT-50 | 25,689 | 16,569 | 2,548 | 6,572 | .2945–.999914 |
| CU-500 | 24,420 | 21,482 | 1,433 | 1,505 | .1816–.2235 |
| CT-500 | 24,550 | 20,998 | 1,488 | 2,064 | .0932–.999726 |
| IP-50 | 26,687 | 26,687 | 0 | 0 | .25 |

Of 12,950 p=1 anchors, **8,574 were confirmed-stall calls and 4,376 were non-stall nominal saturation**. Calling every p=1 decision a stall override would misdescribe CT. Another 197 supported CT anchors have p>.95: formal overlap does not imply useful cache-branch sample size.

At the actual CT deviation-entry latch, support is **0/1,323 entries for size 50**, versus **652/1,309 for size 500**. CT-50 mixed-kernel support is only **2,109/7,857** fresh anchors; its interior support is **11,061/12,649**. CU-50 mixed support is **5,126/8,301**. Thus unsupported high-risk states cannot inherit an estimated effect from the surviving supported subset. IP removes guard/cooldown exclusions, but follows a different future controller and has no dense-library replication.

Branch ESS also matters: GR00T L10-50 CT's first mixed entries contain 140 calls/110 caches across 250 episodes, but inverse-propensity ESS is only **123.5/76.9**. Complete branch counts and ESS for first-entry and probability-shift estimates remain in each native report.

### Stage construction and estimand

No original row has `stage_pre`/`pre_stage`: **0/247,543**. The stock tool correctly refuses to certify its catalog proxy for causal moderation. For the supplemental runs:

- **R7 coarse:** reconstruct from frozen catalog modes, event flags and exact retrieval rows/weights. Positive unknown-mode mass gives `unknown`; non-unanimous known modes give `mixed`; unanimous known modes give `event` if event mass is positive, otherwise `interior`. Detailed run/mode labels are exported but not searched for a winning split. CU/CT and IP retrieve before drawing the current call coin; source inspection establishes that ordering.
- **Physical:** use `forensics.physical` at **`control_idx_start - 1`**, not the majority label during the actions selected by that decision. Freeze the published defaults: near radius .10 m, lift .03 m, placement radius .10 m. No callback outcome calibration. These are privileged kinematic/command heuristics, explicitly **not manually validated ground-truth stages**.
- **Effect:** CALL versus CACHE at the **first supported encounter** with a stage, then follow the original arm's future controller. This can be later than the first physical visit. Unreached episodes contribute zero to the saved population estimate; the tables below show conditional-reached effects. No division by realized stage duration or future visit count is used.
- The supplemental SR score is `(Z/p - (1-Z)/(1-p)) × (Y - b)`, with `b` the other-init-parity mean success within task. Baselines are refit inside each bootstrap. Resampling keeps shared task/init pairs together across arms and models, within fixed tasks; the two suites are separate. This covers init variability, not new tasks or new lottery seeds.
- Four-cell summaries average the four conditional effects equally. Reached/call/cache counts are summed for coverage, not used to reweight cells. Stage contrasts compare different reachable histories; they do not estimate same-state branch differences.

The uncentered tool is unbiased under its assignment assumptions but noisy. For GR00T L10-50 CU initial approach, the native SR score is **+22.6 pp [2.0, 40.0]**, with 168 calls/132 caches. Centering changes it to **+1.5 pp [−7.2, 10.5]**, still 300 entries. A positive raw HT interval is not enough to name a high-value stage.

### Physical placement is a candidate, not a validated allocation rule

Each row has a 1,200-episode original population. Effects are percentage points; brackets are **pointwise** 95% intervals. `n (call/cache)` counts first supported entries.

| Group | Grasp-window n (call/cache) | Grasp-window ΔSR | Placement n (call/cache) | Placement ΔSR |
|---|---:|---|---:|---|
| CU-50 | 1,099 (520/579) | +2.2 [−1.5, +6.1] | 746 (399/347) | +3.7 [−1.2, +8.5] |
| CT-50 | 965 (513/452) | +2.9 [−1.7, +7.5] | 735 (289/446) | **+5.4 [+0.6, +10.5]** |
| IP-50 | 1,120 (278/842) | +1.9 [−3.3, +7.2] | 731 (195/536) | +2.1 [−4.8, +8.7] |
| CU-500 | 1,123 (234/889) | −0.3 [−4.5, +3.7] | 790 (158/632) | −2.3 [−8.2, +3.5] |
| CT-500 | 1,120 (527/593) | −0.9 [−5.3, +3.3] | 778 (118/660) | +2.1 [−6.1, +9.8] |

The direct **CT-50 placement-minus-approach** contrast is **+7.1 pp [+1.2, +13.0]**, comparing 735 placement entries with 1,200 initial-approach entries. Its simultaneous interval is **[−5.2, +19.4]**. The family comprises 120 cell-specific and 30 four-cell contrasts: event/interior, mixed/interior, and grasp/carry/place/release versus approach. **0/150 simultaneous intervals exclude zero** (bootstrap max-|t| critical value 4.006; 9,937/10,000 replicates retain nonempty reached sets for every contrast, including rare release strata).

Cell dependence matters. GR00T L10-50 CT gives grasp-minus-approach **+24.4 pp [9.5, 40.4]** (230 vs 300 entries), whereas π0.5 Spatial-50 CT gives **−10.3 pp [−19.8, −1.8]** (246 vs 300). GR00T L10-50 IP's placement contrast is positive pointwise, but its grasp call effect is **−5.8 pp [−18.5, +6.8]**, 61 calls/200 caches. These are reasons to replicate the placement hypothesis, not to turn “near object” into a universal call gate. Rare release strata are particularly weak: GR00T L10-50 CT has only **five** first supported release entries, including one cache.

Dense π0.5 Spatial CU already succeeds in **297/300** discovery episodes, so SR alone has little headroom there. Failure to validate an interaction is not evidence that all boundaries have identical physical difficulty or that smaller benefits are absent.

### The old gripper split does not identify a differential call benefit

| Group | Event/interior reached, each out of 1,200 | Event-minus-interior ΔSR, pp [pointwise 95% CI] |
|---|---|---|
| CU-50 | 1,173 / 1,200 | +3.0 [−2.6, +8.4] |
| CT-50 | 1,164 / 1,200 | +3.3 [−2.6, +9.3] |
| IP-50 | 1,194 / 1,200 | +2.9 [−4.6, +10.3] |
| CU-500 | 958 / 1,077 | +0.8 [−5.7, +7.1] |
| CT-500 | 910 / 1,079 | +3.2 [−2.5, +9.1] |

The whole-controller discovery comparison agrees: CT succeeds in **2,167/2,400**, CU in **2,165/2,400**; paired equal-cell difference **+0.08 pp [−1.21, +1.38]**. This only tests the frozen CT system, not an improved segmentation.

The physical overlap shows what the old labels miss. Of placement-stage fresh anchors, **2,247/5,215 (43.1%)** in size-50 arms and **1,670/3,482 (48.0%)** in size-500 arms are labeled R7 `interior`. Of all dense-library fresh anchors, **23,386/48,970 (47.8%)** have nonzero unknown catalog-mode mass. This is distinct from the stock tool's missing assignment-time certification. Hiding that unknown stratum would erase nearly half the dense-library experience.

First-entry timing also limits the physical split: **all 6,000/6,000 first supported approach entries have zero predicate change and zero carried-object change over the next 20 controls**. They occur too early for these endpoints. A null first-entry approach effect says little about late alignment inside the same long stage. Native probability-shift derivatives retain all supported anchors, but do not by themselves identify a precise intervention boundary.

### Cost accounting answers where the budget went

Across the 20 call arms, the ledger reports **127,341 live vision dispatches**, **45,106 policy calls**, and **57,436.152 owner-work units**, excluding deferred diagnostic work. Stage work partitions the arm total; recorded versus repriced work and stage sums reconcile within **9.1×10⁻¹³** units in every arm. Actual control denominators handle partial final blocks.

For the four size-50 cells, CU→CT changes calls attributed to `interior` **6,441→4,696**, `mixed` **4,225→5,587**, and `event` **2,522→2,891**. Total calls barely change: **13,188→13,174**. This confirms redistribution happened even though the overall success advantage is unestablished. These are descriptive totals on different trajectories, not causal effects of subtracting a call from one stage and adding it to another.

Dense-library `unknown` accounts for **8,816.148/17,537.392 (50.3%)** combined CU/CT work. Actual request IR ranges from **.1640 to .1928** for the nominal .18 arms and **.2936 to .3096** for nominal .30 arms; exact equal-cost claims would be inappropriate. IP request IR is .1818–.1869, a different cost/continuation regime from CU/CT-50.

For GR00T L10-50, A uses **10,669 looks/21,251 decisions**, IR **.07430**, and FL uses **7,939/21,325**, IR **.05510**. For size 500, A is **8,875/17,642**, IR **.07445**; FL is **6,770/18,883**, IR **.05306**. This prices the extension savings; it is not a task-success comparison.

### Look opportunity grows on extended blind paths — augmentation partial

The native motion metric requires the correction in §4. Here motion means channels 0–5, gripper is channel 6, and command threshold zero comes from the existing LIBERO adapter. No scale was fitted; these are unscaled captured-action RMS units, suitable for within-model comparisons. For each blind decision I compare the applied prefix of its tail and its unused fresh-cache proposal against the **same** policy shadow. Positive improvement means the refresh is closer to that teacher draw. Episode means receive equal weight; all rows below cover 300 episodes.

| GR00T L10 arm | Age, controls | Blind decisions | Tail→fresh motion RMS | Improvement [95% CI] |
|---|---:|---:|---|---|
| A-50 | 5 | 10,582 | .12715→.12565 | .00150 [−.00004, .00313] |
| A-500 | 5 | 8,767 | .08518→.08245 | .00273 [.00179, .00372] |
| FL-50 | 5 | 7,792 | .13884→.13507 | .00377 [.00222, .00532] |
| FL-50 | 10 | 3,734 | .14008→.13403 | .00606 [.00388, .00816] |
| FL-50 | 15 | 1,860 | .13854→.12979 | .00876 [.00599, .01161] |
| FL-500 | 5 | 6,628 | .09087→.08740 | .00347 [.00235, .00468] |
| FL-500 | 10 | 3,662 | .09477→.08366 | .01112 [.00899, .01322] |
| FL-500 | 15 | 1,823 | .09703→.08605 | .01098 [.00857, .01336] |

The uncontaminated CU/CT arms also show small age-5 refresh opportunities, each over 300 episodes. These concern cache tails only; policy tails are separately present in native `divergence` output.

| GR00T arm | Cache-blind decisions | Motion RMS improvement [95% CI] |
|---|---:|---|
| L10-50 CU | 4,114 | .00263 [.00115, .00413] |
| L10-50 CT | 4,169 | .00194 [.00070, .00320] |
| L10-500 CT | 6,237 | .00293 [.00200, .00386] |
| Spatial-500 CU | 2,512 | .00312 [.00149, .00481] |
| Spatial-500 CT | 2,544 | .00384 [.00213, .00556] |
| L10-500 CU | 6,288 available | Withheld: mixed-version anchor parity failure |

Stage conditioning is informative descriptively: at FL-50 age 10, `mixed` kernels have improvement **.01156 [.00595, .01686]**, 1,341 decisions/288 episodes, whereas `interior` has **.00002 [−.00205, .00215]**, 1,640/300. At age 15 these are **.01642 [.01052, .02250]**, 930/271, and **.00289 [−.00092, .00673]**, 713/280. These are exploratory action-opportunity differences. The stage is the carried anchor's frozen library label, not current object/contact truth.

The same-state six-channel MSE improvement is also positive in each aggregate A/FL age bin, so the aggregate direction does not depend on RMS alone. However, neither proposal is executed in the comparison arm, policy shadows are stochastic, and the teacher can be wrong. Different ages condition on different visited states and extension eligibility. **Do not interpret these curves as success gain from looking, or select a deployment age threshold from them alone.**

### Trigger timing provides descriptive checks, not failure prediction validity

CU stall alerts hit **35/200** available failed-episode heuristic onsets within the preceding 20 controls; **28/200** have at least five controls of lead. CT stall hits **42/206**, actionable **34/206**; CT deviation-entry hits **31/206**, actionable **23/206**. Another 35 CU and 27 CT failures lack an onset time. Denominators are arm-episodes, with shared initial pairs across cells.

The tool labels alerts in **806/2,165** successful CU episodes, **803/2,167** successful CT episodes for stall, and **1,176/2,167** for CT deviation as `false_alert`. Those numbers cannot estimate false-positive rates: calling after an alert may have enabled success, and the forensics labels retain only the final failure's selected onset. A grasp-miss onset is first entry into the near-object region, not certified irreversible grasp failure. No onset-derived threshold was selected.

## 4. Bugs and data-quality problems found

1. **Mixed-version augmentation fails fresh-anchor parity.** In `r8_groot_l10_500_CU`, all **2,779/2,779** discovery anchors in code SHA `610613bf…` differ between live `cache_chunk` and deferred `shadow_look.cache_chunk`; all **6,483/6,483** in `9925eb9f…` match exactly on the applied prefix. Median old-part maximum difference is .00568; maximum is **.231892**, with **901/2,779** above .01. Example: task 1/init 27, decision `8a756fdb807919f9c6903cb8_a1:1:34`, `debug/aug/shadow_look/part_00207.npz`. The upgrade note `debug/aug/provenance/code_upgrade_5c85f9e721478791a45ccddd97b834443d7845663921998a8cb628a67a751484.json` explicitly preserves older parts. Those parts cover **5,407/17,866** discovery decisions and **1,917/6,288** cache-blind comparisons. This establishes a version-associated replay discrepancy, not its internal numerical cause. `AUG_DONE` denotes completion, not validated equivalence; current tools aggregate across the versions without surfacing this failure. Withhold this arm's pooled shadow interpretation pending audit/regeneration. All primary call estimates are augmentation-independent. Evidence: `summary/aug_audit.json`, per-arm `anchor_parity_by_provenance.parquet`.
2. **Missing causal stage join.** `runs/<arm>/debug/server_*/decisions_*.jsonl` lacks `stage_pre`/`pre_stage` in all 247,543 analyzed call decisions. `decision/common.py:stages` only certifies those fields; `decision/call_value.py` therefore pools the stock estimator under `unknown` despite available catalog labels. Safe behavior, but the requested stage-value workflow is incomplete. The analysis sidecar reconstructs pre-treatment labels; it does not repair capture files.
3. **Motion metric silently includes the gripper.** For example, `runs/r8_groot_l10_50_A/debug/server_23176/meta_1327325.json` and its `model_manifest` declare valid dimensions 0–6 but no `gripper_dim`, `gripper_threshold`, `action_scale`, or `action_std`. `decision/common.py:dimensions` returns `grip=None`; `decision/metrics.py:action_metrics` then includes all seven dimensions in `motion_rms`, while gripper metrics are null. `follow_vs_look` and member spread inherit this. In A-50, the gripper contributes an episode-balanced **27.4%** of tail-versus-policy squared error overall and **53.3%** in the mixed-kernel subset; this is material. Corrected results explicitly use the existing physical adapter definition, with raw metadata saved beside them.
4. **Physical “truth” is heuristic.** `physical/forensics.py:physical` uses distance/lift/command rules; `analyse` assigns a final-failure label and onset retrospectively. The callback freezes defaults and uses pre-control stages, but does not establish contact-verified grasp or irreversible failure time. `truth_validated=false` must remain visible. A release can be task progress, so a negative carried-object delta is not uniformly bad.
5. **Misleading interpretation of successful-episode alerts.** `decision/trigger_vs_onset.py` treats episode success as a certified no-onset negative for `false_alert`. It cannot distinguish unnecessary alerts from successful interventions or recovered transient errors. Rename/report this as successful-episode alert prevalence unless independent event-level negative labels exist.
6. **Explicit endpoint gaps remain.** Native reports mark object-relative demo drift and robot-demo drift unavailable because aligned admitted-library physics/frames are missing; stall-entry and valve-statistic endpoints also lack certified future-clock joins. The capture contains useful current physics, but those derived endpoints are not implemented by simply having object poses. I did not replace unavailable values with zero.

No coin/propensity/dispatch corruption or reader join issue was found in the examined call data. Deterministic/saturated CT support and incomplete cross-model augmentation are experimental limitations, not recording bugs. This callback does not certify debug on/off behavioral parity, full-horizon retrieval replay, simulator restoration, or unseen arms.

## 5. What R9 should do

1. **Finish the analysis contract first.** Emit versioned `stage_pre`, its observation/control cutoff and feature provenance, plus explicit gripper geometry/action scaling. Preserve unknown and deterministic-support strata. Add centered first-entry estimates, direct stage contrasts, branch ESS and simultaneous intervals to `call_value`; provide a discovery-only reader/CLI option so every analyst need not build a masking wrapper.
2. **Retain placement as a hypothesis; validate rather than deploy it.** Freeze a small family that distinguishes early approach from final alignment, closure/lift, transport, and placement/release. Prefer robot-observable relative-motion/aperture features for the deployable version; use simulator object/contact labels as diagnostics. The existing .10 m grasp window is too broad to equate with a valuable intervention opportunity. Do not optimize the locked holdout after each exploratory split.
3. **Add overlap where CT removes it.** Extend independent-call probes to dense libraries and repeat lottery seeds. Use bounded diagnostic propensities in the states the candidate allocation would target, reporting changes to guard/continuation semantics. CT-50 cannot answer its own deviation-entry counterfactual with 0/1,323 supported entries. Keep whole-controller equal-budget evaluation separate from local effect estimation.
4. **Measure actual look packages.** At a predeclared eligible blind point, randomize continue / fresh retrieval / policy call with a shared commitment horizon and named future controller. Start with the extended-age/mixed-kernel opportunity seen here, but preserve comparison support across the frozen strata. Report physical progress, completion, downstream controls and work; current shadow distance is only a screening feature and costs a look to observe online.
5. **Fix the timing target and endpoints.** First approach plus a 20-control horizon produced 6,000 zeros. Add a pre-specified late-alignment entry or an episode-level sampled opportunity, and event-appropriate proximal outcomes: actual grasp/lift after grasp attempts, goal satisfaction and stable release around placement. Audit onset/recovery histories in successful and failed episodes before scoring alert specificity.
6. **Audit augmentation admission, then certify branches separately.** Validate/regenerate the retained `610613bf…` parts of GR00T L10-500 CU and require anchor parity/provenance checks in addition to `AUG_DONE`. Re-run the frozen shadow analysis after π0.5/IP augmentation finishes, without moving thresholds. For branch experiments, first certify factual replay of simulator, controller, RNG and method history/continuation state. `restore_certified=false` remains a hard evidentiary boundary; no shadow-only rescue claim should cross it.
