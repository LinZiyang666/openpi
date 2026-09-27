# R4-A: look once, act several steps

Author: ideation agent A. Date: 2026-09-27. Repository branch supplied by the coordinator: Ziyang. This report contains CPU diagnostics and a proposed implementation contract; no server, plugin, existing method, or closed-loop run was changed.

**Recommendation:** first test a vision-anchored, fixed-weight AWM kernel whose members advance within their own episodes, with independent proprioceptive phase correction and at most two blind decisions. Re-look before predicted gripper events, near episode ends, on low motion or state-displacement error, and immediately after a policy MISS. Include the unused tail of the anchor chunk as a required control. Use the 500-episode π0.5 l10 guard arm as the primary target, and retain both library scales throughout the comparison.

On the recorded 500-episode l10 guard path, a two-blind budget with library-phase gates changes reference IR from **0.238216 to 0.170661**, holding the original MISS decisions fixed. This is cost arithmetic on a frozen path, not a measured new operating point. Transporting the incremental cost of proprioceptive gates from the offline cache path gives approximately **0.179**; neither its SR nor the changed guard/MISS timing has been measured. At 50 episodes the corresponding guard estimate is **0.2653**, or approximately **0.274** with that same transport; the stronger existing 50-episode periodic-5 arm gives **0.2684**, approximately **0.277** with proprioception. These are promising enough to justify a pilot, not a license to report preserved SR.

## 1. Evidence, definitions, and reproducibility

Binding context was read in the requested order: `rounds/r04/FINDINGS.md`, `rounds/r03/ANALYSIS.md`, `rounds/r02/ANALYSIS.md`, exploration log sections 8–10, `IDEATION_BRIEF.md`, harness and closed-loop READMEs, then AWM/AWM3, MixedJudge, V6/V7, plugin and operations code. Paths beginning `rounds/` here are relative to `exp/offline_search/`. Findings rule out whole-episode vision removal and using offline action error to rank SR. R3's failed recovery, terminal masking, gripper commitment, event-triggered MISS bursts, and learned metric experiments remain settled negatives. Re-looking cheaply before a transition is a different control intervention from forcing a policy MISS there, and needs its own test.

The hot store was absent, so all measurements used the cold store `/home/weiland/trace_runs/offline_search_store` read-only. Eight query cells contain 500 episodes each: π0.5 spatial inf/cache **10,798/14,621** decisions; π0.5 l10 **29,406/40,127**; GR00T spatial **11,338/13,820**; GR00T l10 **29,065/39,669**. This is **188,844 distinct recorded decisions**, replayed against both libraries, or **377,688 query/library evaluations**. Every horizon uses all same-episode pairs `(t,t+h)`, h=1…4; episode boundaries are never crossed. There are `N−500h` such pairs in each cell. Five control steps make one decision; blind horizons 1…4 therefore extend 5…20 control steps beyond the anchor's first executed block.

Scripts and exact outputs, all in this directory:

- [measure_blind.py](measure_blind.py): exact-artifact AWM anchor replay, all serving alternatives, horizon/phase error and independent triggers. `anchors_*.json` records artifact provenance and numerical parity. `metrics_*.csv` contains every method × horizon × cell × library × phase, including mean, median, p90, gripper mismatch, drift from current AWM, and paired Δerror. `triggers_*.csv` includes phase-specific firing rates; `residuals_*.csv` includes residual quantiles.
- [measure_logs.py](measure_logs.py): accepted-attempt reconstruction of eight pure-cache and eight mixed arms, 500 episodes each; first failed-episode repeat-spell timing; anchor-based look schedules preserving observed MISS positions. `log_summary_*.json`, `log_timing_*.csv`, `log_schedules_*.csv` are the exact results.
- [audit_logs.py](audit_logs.py): kernel episode diversity and gripper-command phase at the first failed-episode spell, in `command_phase_summary.json` and per-cell CSVs.
- [schedules.py](schedules.py): causal look decisions on fixed offline observations and recorded anchor histories; caps 1–4; phase, state, and combined gates; displacement thresholds .25/.5/1. Outputs `offline_schedules_*.csv`.
- [build_tables.py](build_tables.py): the numerical appendix, `bytes.json`, `log_schedule_aggregate.csv`, and `paired_error_ci.csv` (1,000 whole-episode bootstrap draws, seed 4817). [RUN.md](RUN.md) contains exact CPU-affined commands.

[validate_outputs.py](validate_outputs.py) passed: **16 query/library cells, 1,430,752 horizon windows, 7,968 metric rows**, correct disjoint phase partitions, normalized kernel weights, finite errors, same-library fit provenance, and exact IR identities. Log reconstruction contains **8,000 accepted arm-episodes / 372,126 decisions**. The largest newly written array file is **38,628,642 bytes**, below the 50 MB placement threshold. See `validation.json` for the machine-readable record.

The existing R2 fitted AWM artifacts are used separately at 50 and 500 episodes: kref=5 and 8 respectively. This preserves the deployed scale's own fit, normalization, and action store. **No 50-episode fit uses 500-episode data; no experiment here uses borrowed big-library information.** The “50” π0.5 spatial library actually contains 49 episodes. Batched replay was checked against 40 scalar `Method.query(QueryView)` calls per query/library cell: **640/640 top-1 matches, 638/640 full top-16 matches**, maximum valid-action absolute difference **0.000298366**. The two full-list mismatches remain a numerical-parity limitation; their exact cause was not isolated.

Action error means RMS over the next **5×7 valid action values**, normalized by the harness's current-library action σ, using `a_inf` as the reference. The same σ is used at both library scales, so scale comparisons are meaningful. This is a proxy for serving degradation, not ground-truth optimal control. State alignment uses only valid `robot_state[:8]`; padded dimensions never enter it. Query episode length and thirds are evaluation labels only. GR00T and π0.5 have different gripper conventions; mismatch is measured in each model's own normalized action convention.

The inf-cell AWM anchor sees the recorded previous policy chunk, as the real fresh branch would on that recorded path. A hypothetical prior blind action would change that history. We therefore report both recorded-history vision AWM and `awm_allhit`, which removes fresh-policy continuity at the future comparison point. Neither supplies a counterfactual rollout. Cache-cell replay avoids that fresh-branch privilege. All schedules still hold robot observations fixed. Longer-horizon AWM rows have different target populations because early targets are excluded; their changing means are not AWM drift.

### 1.1 What to serve without vision

At an anchor, retain **all 16 members**, their row IDs, and normalized AWM weights, not the logged top 10 and not only the winner. The measured alternatives are:

1. `top1_clock`: advance the top member h successor rows and execute its five-action head.
2. `kernel_clock`: advance each member h rows within its own episode, preserve its weight, synthesize the five-action head.
3. `phase_abs`: independently choose each member's phase from offsets `{h−1,h,h+1}`, clamped to its own episode. Require nondecreasing phase and at most two rows of advance since the previous blind decision. Minimize `mean(((s_current−s_library)/σ_task)^2) + .05*(offset−h)^2`. σ is each deployed library's per-task state standard deviation, floored at .05 per valid dimension. Preserve the anchor weights.
4. `phase_delta` and `phase_hybrid`: use displacement from the anchor instead of absolute state, or an equal mixture. `phase_reweight` additionally multiplies weights by `exp(−state_cost/T)` at T=.25 or 1.
5. `anchor_chunk`: execute the next unexecuted five-action block of the *same synthesized anchor chunk*. π0.5 H=10 supports **one** blind decision; GR00T H=16 supports **two**, with one residual action unused. Horizons beyond these are unavailable, not zero-error or repeated padding.
6. `repeat_anchor_head`: repeat the first five actions as a negative control.

The full numerical appendix reports all four cells, inf and cache, both library sizes, and horizons 1–4. Phase-error, ΔAWM, and gripper mismatch tables cover near/far transitions and early/mid/late; the CSVs also contain their six intersections for every comparator.

The strongest serving observations are:

- **Following a single demonstration loses the kernel.** On π0.5 l10 cache, h=2 error at 50 episodes is **.701 top-1 / .596 kernel clock / .573 phase alignment / .520 vision AWM**. At 500 it is **.640 / .530 / .504 / .438**. Paired phase-minus-clock error is **−.022461**, episode-bootstrap 95% CI **[−.027101,−.018033]**, at 50 and **−.026622 [−.030514,−.022679]** at 500. These intervals establish a proxy effect, not an SR effect.
- **This persists in GR00T.** GR00T l10 cache h=2 is **.705 / .575 / .541 / .519** at 50 and **.667 / .529 / .492 / .448** at 500. At 500, phase-minus-clock is **−.037257 [−.041666,−.032960]**; top1-minus-clock is **+.137477**.
- **The longer-chunk baseline is competitive, especially spatial.** Spatial cache h=1 tail/phase error is π0.5 **.579/.596** at 50 and **.503/.530** at 500; GR00T **.483/.518** and **.410/.434**. On GR00T l10 h=2, however, tail/phase is **.600/.541** at 50 and **.555/.492** at 500; the latter paired difference is **+.063324 [.054951,.072435]**. Thus the extra control flexibility is plausible, but tail execution is an essential matched-cost control.
- **Four blind decisions are a substantial extrapolation.** π0.5 l10 cache phase error rises **.538→.653** from h=1 to h=4 at 50 and **.461→.595** at 500. At 500, its ΔAWM near/far transitions is **.026/.025 at h=1**, but **.108/.177 at h=4**. At 50, h=4 near/far ΔAWM is **.074/.152**. Far-from-gripper drift is real; gripper gates cannot justify unlimited blind segments. At 500, h=4 early/mid/late ΔAWM is **.114/.174/.155**.
- **State evidence helps phase more reliably than member confidence.** At h=1 on cache paths, temperature-.25 state reweighting worsens phase-only error in all four cells at both scales: 50 **.596→.615 / .538→.572 / .518→.534 / .519→.562**; 500 **.530→.569 / .461→.482 / .434→.436 / .451→.466**. This is a reason to keep vision weights fixed in the first pilot, not proof all adaptive weights fail.

Inf results are not omitted. For example, GR00T spatial 500 h=1 has recorded-history AWM **.254**, all-HIT AWM **.364**, and phase serving **.353**. Comparing only to the first number would confound the serving change with loss of fresh-policy continuity. All three appear in the appendix.

### 1.2 Why the closed-loop mechanism matters

The accepted R2 paths reproduce the supplied pure-cache SRs: π0.5 spatial/l10 and GR00T spatial/l10 respectively **.800/.630/.888/.552 at 50**, **.954/.768/.966/.706 at 500**, all at IR .152. Excluding episode starts, exact-successor rates are **29.1/29.1/27.0/18.2% at 50**, and **42.7/34.8/24.8/17.3% at 500**. Same-row rates are **24.8/36.0/14.9/39.9%**, and **5.1/22.6/3.5/25.3%**. The owner’s 18–29% and 15–40% summary is the 50-library pattern; a large library changes it substantially for π0.5, but does not make top-1 following generally reliable.

Using the logged top-10 approximation, mean effective contributing episodes `1/sum(episode_mass²)` are **3.75/3.10/4.02/3.06** at 50 and **7.37/5.08/7.69/5.23** at 500. Larger libraries make preserving the mixture more, not less, relevant. Truncation to ten is a limitation of log-derived numbers; offline serving uses sixteen. Repeated top-row spells identify a useful failure symptom, but an intentionally advancing blind clock can conceal that symptom without moving the robot. Pilot endpoints must include physical progress, task success, and timeout phases, not just fewer repeated IDs.

The first run of three identical top-row HITs occurs in **100/184/56/224** failed episodes at 50 and **23/115/15/146** at 500. Mean library progress at its onset is **.695/.479/.931/.374**, and **.803/.435/.830/.439**, respectively. This agrees with the R3 distinction between spatial terminal behavior and l10 failures spread earlier in the task. These spells are retrospectively identified at their first row; an online repeat detector cannot know the third repeat at onset.

Gripper-command phase at that onset, from `audit_logs.py`:

| Cell | Library | Before first close | After close, before first release | After first release |
|---|---:|---:|---:|---:|
| π0.5 spatial | 50 | .170 | .590 | .240 |
| π0.5 spatial | 500 | .000 | .478 | .522 |
| π0.5 l10 | 50 | .348 | .250 | .402 |
| π0.5 l10 | 500 | .304 | .217 | .478 |
| GR00T spatial | 50 | .018 | .857 | .125 |
| GR00T spatial | 500 | .000 | .467 | .533 |
| GR00T l10 | 50 | .384 | .286 | .330 |
| GR00T l10 | 500 | .336 | .185 | .479 |

These are synthesized gripper **commands**, not verified grasp/release of an object. At 50, only **19.0/18.5/14.3/32.1%** of first failed spells are within two decisions of a commanded release; at 500 the values are **39.1/21.7/46.7/41.8%**. A trigger tied solely to the actually issued sign flip misses much of the failure anatomy. Upcoming events across the anchor kernel are broader, although they still cannot detect an object left behind while the arm moves normally.

### 1.3 When to look again

The following gates are computed **before** vision, from the last anchor, known episode topology, and current/past proprioception:

- **Budget:** after B blind decisions, force a vision decision. B=1 and 2 are candidates; 3 and 4 are stress ablations. Every episode begins with vision.
- **Gripper ahead:** define a library row event if the gripper sign changes inside its five-action head, across its previous-head boundary, or into its next head. Force vision if at least .20 of anchor weight is on an event row or its immediate predecessor under nominal phase advance. This conservatively includes some just-passed transitions. The evaluation “near transition” band is event plus prev/next row, also at .20 mass; it is deliberately not identical to the gate.
- **Near terminal:** at least .20 of weight on the last two library decision rows. Terminal rows are clamped only for offline baselines; deployed serving re-looks instead of repeatedly extending them.
- **No motion:** normalized RMS valid-state movement is below the deployed task library's 10th percentile of successor movement for two consecutive decision intervals. This can fire during intentional holds, so it requests cheap vision; it does not automatically command a policy MISS.
- **Displacement residual:** RMS of `(s_now−s_anchor)−sum(w_i*(s_nominal_i−s_anchor_i))`, divided by per-task state σ, exceeds .5. Thresholds .25 and 1 are sensitivity variants. Use nominal motion for this gate; allowing phase fitting to erase its residual could hide a stall.
- **Lifecycle/MISS:** first decision, after any MISS, invalid anchor, task change, unexpected executed-step count, or exhausted tail horizon forces vision. A policy MISS invalidates the previous kernel; its output must be followed by a new anchor rather than silently reusing the pre-MISS demonstration support.

These thresholds were fixed diagnostics, not fitted to query labels. State scales and motion percentiles are fitted only from the deployed library.

Independent h=1 cache-window firing rates (π0.5 spatial/l10, GR00T spatial/l10) are:

| Trigger | 50 episodes | 500 episodes |
|---|---|---|
| Gripper ahead | .370 / .329 / .293 / .393 | .376 / .353 / .369 / .370 |
| Near terminal | .340 / .159 / .281 / .129 | .337 / .128 / .269 / .099 |
| Two low-motion intervals | .138 / .206 / .139 / .205 | .128 / .173 / .127 / .145 |
| Displacement residual >.5 | .011 / .038 / .044 / .057 | .007 / .049 / .006 / .047 |
| Absolute-state residual >.5 | .410 / .498 / .367 / .537 | .379 / .399 / .243 / .446 |
| Phase + motion + displacement union | .511 / .523 / .464 / .548 | .506 / .505 / .458 / .501 |

These overlapping window rates cannot be added or interpreted as the final look rate. A causal cap-2 schedule on the cache observations, including phase and state gates, yields vision fractions **.656/.665/.623/.680 at 50** and **.657/.653/.621/.650 at 500**; reference pure-cache IR is **.100/.101/.095/.103** and **.100/.099/.094/.099**. All-row proxy error increments are respectively approximately **.008/.008/.003/.005** and **.012/.012/.007/.005**. The appendix includes every horizon and inf-cell counterpart; these modest averages can still hide a small set of catastrophic mistakes.

With the closed-loop cap-2 phase scheduler, **causal forecasts from the last retained anchor** flag first failed-spell onset in **65.0/71.7/96.4/62.1% at 50** and **87.0/61.7/93.3/77.4% at 500**. Coverage at onset or in the previous two decisions is **71.0/80.4/98.2/71.9%** and **91.3/79.1/93.3/87.0%**. The appendix distinguishes these from retrospective coverage using the actual kernel at every vision decision. They are event-warning coverage, not failure prediction precision or prevented failures; phase events occur in successful episodes too.

The log phase split is useful: for π0.5 l10, gripper-gate prevalence early/mid/late is **.250/.324/.312 at 50**, **.229/.339/.331 at 500**; terminal-gate prevalence is **.003/.087/.260**, **.000/.026/.157**. Spatial terminal gates concentrate much more strongly in the last third. All eight rows are in the appendix.

**Missing measurement:** these closed-loop JSONL records do not retain raw robot states. Therefore actual proprioceptive no-motion/deviation timing relative to their deadlock/grasp/release spells cannot be recovered. AWM's logged `still` is a visual-cosine diagnostic and is not a substitute. Proprioceptive firing was measured on all offline paths; transport to the closed-loop paths is explicitly uncertain. The first pilot must log state, motion, expected displacement and residual to answer this directly.

### 1.4 Cost and library accounting

Let N count all five-control-step decisions, V those that actually run stage 1, and M full-policy MISSes, with M≤V. Then

`IR = (.152*(V−M) + 1.0*M)/N = .152*v + .848*m`.

Equivalently, relative to a current all-vision arm, `ΔIR = −.152*b + .848*Δm`, where b is the new blind fraction. Extra MISSes consume the savings: break-even requires `Δm < .179245*b`. A policy call is always charged; forcing cheap vision is not a MISS. The .152 stage-1 coefficient is the requested reference cost model. It is not a measured GR00T latency split, and approximately-zero blind IR excludes CPU transforms, phase selection and transport overhead that still require timing.

For 500-library π0.5 l10 guards, the .152 floor accounts for **63.8%** of original IR .238216. Observed m=.101670; phase cap-2 gives v=.555556 and IR=.170661. **5.27%** of all decisions are forced post-MISS vision HITs. At 50-library guards, m=.202142, phase cap-2 IR=.265317; post-MISS forced vision is **6.91%**. For periodic-5 l10 at 50, m=.192050 and post-MISS vision is **18.94%**, reducing the available blind stretches; phase cap-2 IR=.268373. Full cap-1…4 arithmetic for all pure and mixed arms is in the appendix. Because guards depend on observations and histories that will change, preserving their recorded MISS positions is a conditional estimate, not a deployable oracle.

Required footprint accounting, decimal MB; exact byte totals are in `bytes.json`:

| Cell | Library episodes / rows | Existing R2 fitted pickle MB | Proposed compact payload MB | Deployed comparison pickle MB |
|---|---|---:|---:|---:|
| π0.5 spatial | 49 / 1,018 | 21.342 | 20.161 | 431 |
| π0.5 spatial | 500 / 10,909 | 46.993 | 28.875 | 431 |
| π0.5 l10 | 50 / 2,640 | 24.631 | 21.590 | 1,103 |
| π0.5 l10 | 500 / 29,472 | 94.143 | 45.229 | 1,103 |
| GR00T spatial | 50 / 1,063 | 22.249 | 20.379 | 429 |
| GR00T spatial | 500 / 11,751 | 58.157 | 31.591 | 429 |
| GR00T l10 | 50 / 2,645 | 26.673 | 22.039 | 1,068 |
| GR00T l10 | 500 / 29,631 | 117.304 | 50.347 | 1,068 |

Compact accounting is **580 B existing AWM representation + 21 B topology/event metadata per entry**, plus **280 B π0.5 / 448 B GR00T valid action payload** per entry and **19,264,320 fixed bytes** for PCA/task matrices. The 21 B are next/prev/episode int32, step/length int16, progress float32, event byte; state[:8] is already in the representation. Add under about 1 KB for the ten tasks' state scales/motion thresholds and ordinary object overhead. These are proposed packed-array bytes, not the measured size of a newly serialized implementation; actual fitted pickles remain in the previous column and include redundant arrays. Tail-only serving can omit the 21 B if it also omits phase gates; all recommended gated variants retain them. MixedJudge has its own fit payload and overhead, so the compact column is not a claim about today's mixed pickle; the R3 analysis reports π0.5 mixed artifacts around 24.8/31.1 MB at 50 and 63.0/134.7 MB at 500. No full-resolution 262 KB/entry keys are required for blind serving.

All proposals are **T0 online advancement/synthesis**, with **T1 deployed-library statistics and the existing AWM fit**, no learned model or GPU training. This experiment reuses fits; new fit time and live CPU latency were not benchmarked. Candidate phase work is 16 members × 3 candidates × 8 state dimensions, plus normal action mixing; computational size alone is not a latency measurement.

## 2. Ranked proposals

Ranking is a mechanism-and-cost judgment, **not** an offline-error ranking of SR. SR intervals below are explicit planning forecasts to make pilot decisions concrete; they are not statistical predictions from the proxy. No new SR was measured.

### Rank 1 — `phase_particles_b2`: fixed vision kernel, independently corrected phase

**Pitch and hypothesis.** Keep the scene-specific mixture chosen with vision, let each member move along its own episode, and correct timing from the arm's observed state. This directly preserves the multi-episode synthesis that made AWM work while targeting its stage-1 floor. Returning to vision at event boundaries and on motion mismatch may avoid blindly advancing through a failed grasp; normal object motion without arm-state error remains an unresolved failure mode.

**Algorithm.** Use `phase_abs` as defined in §1.1, fixed weights, candidate offset penalty .05, independent monotone phase, all 16 anchor members. Test B=1 before B=2. Apply the exact gates in §1.3 before committing a blind action. Form a full valid H×7 mixture at the selected rows, zero-pad to the native H×32 result representation, and let the client execute five steps. The measured proxy scores only those five steps. Re-anchor with normal AWM whenever vision is required; if the mixed judge returns MISS, execute the policy chunk and invalidate the kernel. Initial variants are clock-only, relative-displacement phase, and σ-floor .05 unchanged; state reweighting is excluded from the first pilot.

**Method/plugin implementation contract.** Add an optional vision-free serving interface alongside `Method.query`, described precisely in §3. Ordinary `query` remains the anchor path. The plugin must intercept before `_osp_inner.infer`, perform CPU state normalization, run the gate, then either use normal inference or synthesize/broadcast/log one blind HIT. Merely adding a branch to `Method.query` or `on_search` cannot save stage 1. Guard integration requires the explicit history changes in §3; stock MixedJudge cannot be left unaware of the skipped vision observations.

**Forecast at both scales.** On π0.5 l10 the intended retention band is baseline SR minus 0–3 percentage points: 50 guards **.710–.740**, or periodic-5 **.762–.792**; 500 guards **.834–.864**. This deliberately does not promise an SR gain. Cost hypotheses for B=2 are **about .274 at 50 guards / .277 at 50 periodic-5 / .179 at 500 guards**, using the frozen-path cost plus transported state-gate increments explained earlier. A rise in MISS share can erase these values. For pure-cache controls in all four cells, the cache-path IR estimate is **.100/.101/.095/.103 at 50**, **.100/.099/.094/.099 at 500**; the same retention hypothesis yields SR **.770–.800/.600–.630/.858–.888/.522–.552**, and **.924–.954/.738–.768/.936–.966/.676–.706**. These ranges express the acceptable first hypothesis, not evidence of preserved SR. GR00T mixed SR and cost behavior remain uncalibrated.

**Cost/bytes.** T0 serving plus T1 state statistics; use the corresponding full footprint row in §1.4, at both scales. Changing from 50 to 500 is a separately reported library effect. Keeping the weighted mixture versus taking top-1 is synthesis; phase correction at fixed mixture/library is the method effect; the look/MISS schedule is control. They must be ablated separately.

**Cheapest diagnostic.** Already completed: cache and inf horizon replay, phase splits, bootstrap clock-versus-phase contrasts, and causal warning coverage. Before rollout, use a fake stage-1 counter and state/action transform parity checks described in §3; these test whether the floor was actually removed.

**Pilot.** Primary cell π0.5 l10, both 50 and 500 libraries, same paired initial states. Arms: (i) existing guard baseline B=0, (ii) kernel-clock B=1, (iii) phase B=1, (iv) phase B=2, (v) gated anchor-tail B=1. Keep phase/state gates identical between clock, phase and tail. At 50 also retain the existing periodic-5 baseline because guard-only is not its best frontier point. Use 100 inits (10 tasks × 10) only to reject a collapse; R3 says it cannot resolve approximately seven-point differences. Confirm survivors on all 500 paired inits, then validate the same B=0/B=2/tail comparison in π0.5 spatial and both GR00T suites, each at both scales. No extra tuning on successful test inits.

**Kill criterion.** Kill B=2 if 500-paired evaluation shows more than 3 pp SR loss, new persistent grasp/terminal stalls, or no useful movement of the SR–IR frontier. For the primary 500 guard target, IR≥.20 together with any meaningful SR loss is insufficient; at 50 it must improve on the existing periodic frontier, not only on guards. The 3 pp rule is an operational tolerance, not a significance claim; report paired confidence intervals. Roll back to B=1 if B=2 is dominated. Blind row advancement is not counted as physical recovery.

If the required gap-aware guard adapter differs from the original judge even with B=0, include its own B=0 control. Compare blind versus B=0 under that same adapter, and report the adapter's effect against the original guard separately. Optional new proprioceptive MISS guards are separate variants; the main low-motion/displacement gates only force vision. This prevents a judge change from being attributed to skipped stage 1.

### Rank 2 — `guarded_anchor_tail`: execute the anchor's unused tail

**Pitch and hypothesis.** Spend the action horizon already generated by the anchor before retrieving again. This requires less online selection and is a necessary control for whether phase advancement offers more value than simply executing more of the same chunk. Spatial h=1 proxy results make it a serious candidate in its own right.

**Algorithm and implementation.** Freeze the full AWM anchor chunk. At the next five-step decision, return `[5:10]`; for GR00T a second blind decision can return `[10:15]`. Apply the same phase, motion and displacement gates as Rank 1 so comparisons match look cost. The library phase forecast is still advanced for gating, but does not change the stored tail's actions. At exhausted horizon, force vision; do not wrap/repeat/pad as extra executable actions. Native wire output still expects H actions, so any padding beyond the selected five must be marked unexecuted and kept out of fresh-policy continuity; `prev_hit=True` already disables that branch. Prefer constructing a valid shifted chunk from the remaining anchor tail and deterministic padding only for the wire/history contract, never using the padded region to authorize another blind block. The plugin bypass/bookkeeping is the same §3 path; the serving calculation itself is T0.

**Forecast, cost and bytes.** Main variant is B=1 at both scales, B=2 only for GR00T. The spatial retention hypothesis is 0–3 pp below pure AWM: π0.5/GR00T **.770–.800/.858–.888 at 50**, **.924–.954/.936–.966 at 500**. On recorded pure spatial paths, phase-gated B=1 IR is **.105/.103 at 50**, **.099/.097 at 500**, before any added proprioceptive look cost. A 0–.01 additional IR allowance is a sensitivity budget, not a measurement. For l10 this remains a control rather than a claim of greater SR; use the same retention target around .630/.552 at 50 and .768/.706 at 500 and reject if it fails. Mixed spatial 50 V7's frozen phase B=1 IR changes **.430→.397**; 500 **.438→.404**. No inference of retained .980/.978 SR follows. Same library bytes as §1.4 when phase gates are enabled; an ungated tail needs no topology addition.

**Cheapest diagnostic and pilot.** The measured tail baselines already establish feasible horizons and phase-conditioned errors. Include an **ungated tail** arm (only horizon/lifecycle re-look) as the literal execute-more-steps baseline, plus gated tail versus Rank 1 under the identical scheduler. Use all four cells and both scales for the diagnostic; prioritize spatial 100-init collapse checks, then 500 paired inits for a decision. GR00T B=1 versus B=2 isolates 10 versus 15 total control steps from an anchor. π0.5 cannot provide a valid 15-step tail baseline from H=10.

**Kill criterion and variants.** Kill if a confirmed >3 pp SR loss, especially at transitions, outweighs saved IR, or if phase serving at matched look/MISS cost improves SR by ≥3 pp. Compare fixed 10-step execution, gated 10-step execution, and GR00T-only gated 15-step execution separately. Tail versus phase is a serving-method effect; adding interruptible five-step proprioceptive checks is a control effect. The larger library is again independent.

### Rank 3 — `two_clock_periodic`: separate vision anchors from policy MISSes

**Pitch and hypothesis.** A cheap vision anchor should not require a policy call. Use bounded blind serving between anchors, with a separate periodic full-policy reset on l10. R3 found periodic MISSes competitive there; this offers a simpler initial control than trying to preserve a visually defined no-progress guard through missing observations. It is ranked below Rank 1 because 500-library periodic SR has not been established and spatial favored targeted MISSes.

**Algorithm.** Use Rank 1 serving B=2 and the same look gates. Decide `periodic_miss` from the **global five-step decision counter**, never the number of vision anchors. A due MISS forces vision and then the full policy; invalidate the kernel and force a fresh vision anchor next decision. Cheap gripper/motion/terminal gates alone request vision, not automatic policy inference. First test k=5 at 50, k=8 at 500, with k=12 only as an exploratory high-saving variant. A guards-plus-periodic variant is a separate arm, not silently combined with the periodic-only baseline.

**Cost arithmetic and forecast.** In an interior periodic cycle with B=2 and mandatory post-MISS anchoring, k=5 costs `(2*.152+1)/5=.2608`; k=8 costs `(3*.152+1)/8=.182`; k=12 costs `(4*.152+1)/12=.134`, before phase/proprioceptive extra looks. Episode starts/ends and gates change these. The recorded 50-library k=5 path gives phase-gated **.268373**, approximately **.277** with transported proprioception, versus original **.314858**, SR **.792**. Forecast retention target **.762–.792** at 50; for 500 k=8 the uncalibrated working hypothesis is SR **.82–.87 at IR .20–.23**, and k=12 **.80–.86 at IR .15–.18**. These 500 SR ranges are planning assumptions around existing .768 pure/.864 guard baselines, not extrapolated measured gains. No positive spatial or GR00T mixed SR forecast is supportable from the available arms; use the known pure baselines and the same −3 pp retention threshold in transfer tests. Report their measured outcomes rather than transporting π0.5 calibration.

**Implementation, cost and bytes.** Same pre-vision bypass and dense history in §3; the period is evaluated before the bypass, so a blind candidate never hides a scheduled MISS. No quantile budget is needed in the initial arm. T0 scheduling/serving plus T1 AWM/state statistics, the same 50/500 footprints in §1.4. It adds no external model or borrowed information. The MISS frequency is explicitly a control-cost effect.

**Cheapest diagnostic and pilot.** The arithmetic and recorded k=5 simulation are already done. Primary π0.5 l10: at 50 compare k=5 B=0/B=2; at 500 compare existing guards B=0, guards B=2, k=8 B=2, and only if useful k=12 B=2, same 500 paired inits. A 100-init run can eliminate collapse but cannot select close SRs. Test both scales of spatial and GR00T only after a survivor, retaining B=0 periodic counterparts so transfer does not conflate MISS and blind changes.

**Kill criterion and variants.** Kill k=8/12 if dominated by the measured 500 guard-blind arm, or if SR falls >3 pp below its matched periodic B=0 comparator without a useful frontier tradeoff. Record policy calls and post-MISS anchors; calling only every kth *vision* decision is a rejected implementation because it changes the policy budget as blindness increases. Variants k=5/8/12, B=1/2, and optional guards are independently labeled.

## 3. Exact server, client, and QueryView specification

This section is an implementation proposal, not code already applied. Relevant source locations are `exp/offline_search/closed_loop/plugin.py:791,805,885,922,1078,1203,1314`, `src/openpi/cache/interceptor.py`, `src/openpi/cache/orchestrator.py:668,1037,1213,1468`, `src/openpi/cache/groot/interceptor.py`, `src/openpi/cache/groot/staged.py:460`, and `exp/libero_groot/policy_adapter.py:37,122,142`.

### 3.1 Where the stage-1 bypass must live

Today `_ConnPolicy.infer` calls `session.set_obs(obs)` for its connection sessions, then **unconditionally** calls `_osp_inner.infer`, and finally calls `after_infer`. In the interceptor, transforms and stage 1 precede CP1 search; `PluginSession.on_search` is downstream of that work. Returning a cheap action from `on_search`, changing `Method.query`, or skipping CP1's search gate still pays stage 1. In fact `PluginStrategy.record_query_keys` raises when the configured always-search path is skipped. Do not implement this by setting the existing search gate to false.

Add a per-connection, explicit **pre-inference blind path** to `_ConnPolicy.infer`, before calling its inner policy. Split “CPU prepare state/output” from “model infer” through model-specific adapters. Prepare current normalized valid state, verify task/episode/executed-step continuity, compute the gate without mutating history, and choose one of two branches:

1. **Vision required:** call the normal inner inference exactly once. Existing CP1 search, method/judge decision, policy MISS if needed, and action broadcast remain authoritative. After a successful vision HIT, save the full sixteen-member anchor, its weights/state, episode IDs, step and chunk. After MISS, invalidate it.
2. **Blind authorized by the gate:** synthesize the normalized native action chunk without entering either interceptor's `infer`, either staged runner, key encoder, or denoiser. Commit one dense query-state record and one decision result, broadcast the action once through the orchestrator, apply the same output transformation to wire actions, and finish the normal logging lifecycle. If preflight fails, fall back to branch 1 before any append/counter mutation.

A new optional interface can be `prepare_blind(q_state, anchor) -> LookReason | BlindCandidate`, plus `commit_blind(candidate)`; it is separate from `Method.query(QueryView)` so methods requiring current keys cannot accidentally run with stale images. The library-only phase statistics are fitted through `Method.fit`; normal `query` returns the existing `Result(action, topk, scores, confidence, extras)` on real vision decisions. Blind Result fields retain real library IDs and the fixed weights; their semantics are explicitly `source=cache_blind`, not a claimed new search confidence. Never apply an old V7 confidence threshold to blind rows as if it were freshly measured.

### 3.2 Dense history and once-only bookkeeping

`_push_inputs` currently appends live keys/state/raw input and checks `b_aex.n == self.step`; `on_search` builds the synthetic payload, sets `_dec`, applies the mixed verdict, and increments the session step. `on_executed`, reached by `orchestrator.broadcast_action -> PluginStrategy.record_action`, appends the actual normalized full chunk, HIT/MISS and execution-success fields. `after_infer` emits the log and clears the pending observation/synthetic/token caches. The bypass must preserve these invariants for **every five-step decision**.

Required changes:

- Split input commit into vision and no-vision variants. Append current robot state/raw observation to dense arrays in both. Add `has_vision`, `hist_has_vision`, `last_vision_step`, and `blind_age` to the query facade. Represent missing dense key rows by an invalid sentinel plus a validity mask; never silently copy anchor keys as current observations. Current key/token getters must report unavailable on blind decisions; `has_tok` must be false even if global token options are on. Dense action/state/HIT history still has length equal to the logical decision counter.
- `prev_a_exec` on the next decision must be the actual chunk served on the preceding decision, whether vision HIT, blind HIT, or policy MISS. Blind HIT sets `prev_hit=True`; only a genuine policy MISS sets it false, preserving AWM's fresh-policy continuity branch. A separately logged execution acknowledgment can validate the five-step prefix; do not equate “had vision” with `hit`.
- AWM's current distance/mix can run unchanged at an anchor, but its diagnostic previous-key cosine must check adjacent `has_vision`. V6/V7 self-change/stuck features also assume consecutive visual observations; comparing the last anchor across a multi-decision gap to the adjacent-decision threshold is not equivalent. Mark the feature missing or use an explicitly gap-aware new feature. Do not renumber `q.step` to count only vision decisions.
- The orchestrator's CP1 check normally appends `state_history` and increments its component-set step counter. A blind branch must append current normalized state and increment the real component-set logical counter **exactly once** without search; otherwise the next CP1 context diverges from the plugin session and can look like a reset. `clear()` clears key-builder state; it does not replace this increment. Use an explicit orchestrator “commit externally served HIT” helper rather than scattering private counter mutations.
- Call `broadcast_action` exactly once so all strategy/gate/judge action histories stay synchronized. Do not also call `on_executed` manually. Do not call `record_query_keys` with fabricated keys. Disable native shadow-search comparison on blind rows, or explicitly report it unavailable; it cannot produce a current native answer without the encoder. Preserve any state-only bookkeeping it needs.
- Create/flush `_dec` once, with `searched=False`, `source=cache_blind`, HIT=true, actual propagated members and actual served chunk. `after_infer` currently derives part of timing from observation/search timestamps; add explicit vision timing so CPU preflight is not mislabeled stage 1. Reset anchor, phase, motion and guard state on every episode/task lifecycle transition. All mutable state belongs to the connection, including when two clients interleave.

### 3.3 Mixed HIT/MISS behavior cannot be assumed unchanged

Calling stock MixedJudge only on anchors makes `_memo_sync` pad skipped progress with NaNs, which breaks its no-progress guard. Filling the gap with nominal advancing library rows is worse: it makes artificial clock motion look like real progress. R3 identifies no-progress as a major source of useful 500-library guard MISSes, so either shortcut invalidates the cost/SR argument.

Specify an explicit guard variant for the pilot. Preserve dense proprioceptive movement at every decision. For retrieval progress, retain only real vision-anchor progress and the elapsed decision gap. On a new vision anchor, compute `(progress_now−progress_last)*(ep_len_now−1)`. If ≤.5, accumulate the elapsed gap in `noprog_span`; otherwise reset it. Trigger at two nonadvancing decision intervals, matching the existing `noprog_n=3` rule when every decision has vision. After one nonadvancing observed anchor transition, require consecutive vision decisions until advancement or the guard resolves. Never use propagated blind phase as measured retrieval progress. Label this guard `noprog_span`, not unchanged MixedJudge.

The blind low-motion gate requests vision. A new proprioceptive stuck guard may then force MISS if the two-interval condition still holds, but it is a separately labeled control change; retain the original visual conjunct only when consecutive visual observations exist. Re-evaluate terminal/overtime guards on actual vision results. When porting guards to GR00T, adapt the normalized gripper convention: the audit treats negative normalized GR00T action as closed, versus positive for π0.5. MixedJudge's present `gexec > 0` terminal-closed condition is π0.5-oriented and cannot be copied without the GR00T conversion.

For periodic mode, evaluate the due MISS before bypass and use the real dense decision step. For V7 quantile mode, the controller needs an explicit blind update: blind accepted HIT can push +infinity, forced MISS −infinity, normal vision query its current confidence, with total-HIT budgets counting blind decisions. This changes the quantile operating distribution and needs calibration; start with guards or periodic mode to avoid treating the old .70 budget as a preserved operating point. The IR ledger independently counts vision and MISS, regardless of controller semantics.

All fixed-MISS cost tables in this report are therefore conditional baselines. No claim is made that the modified guard will reproduce its recorded MISS mask or success rate.

### 3.4 State/action transforms and the client contract

**π0.5:** use the same CPU input normalization and valid-state padding as the policy; do not align raw eight-dimensional wire state to normalized library `robot_state`. The stored normalized H×32 chunk must pass through the policy's output transforms with the current normalized state as required by that path. Returning normalized cached actions directly to LIBERO would be wrong.

**GR00T:** the connection wraps `_InferLockedPolicy`, then `GrootLiberoPolicyAdapter`, then the cache interceptor. A bypass at the outer wrapper must explicitly reuse `build_groot_observation` and the policy's CPU transforms for normalized `state`/`state_mask`. `run_stage1` currently calls the model's `prepare_input`, which also casts floating state to the action-head dtype. A CPU state-only helper must reproduce any dtype rounding and validity mask needed to match recorded `robot_state`, without invoking that model/GPU preparation. On output, use `unapply_transforms({"action": chunk[None,...]})`, unbatch/validate, then `chunk_to_libero_actions`; that conversion includes `sign(1−2*openness)` for the gripper. Do not apply that conversion twice. Keep the appropriate lock around any shared mutable transforms even though no GPU work is done. Transform parity is a required acceptance check, not an assumed property.

**Minimal client change: none for the main experiment.** Continue sending fresh current images, raw robot state, prompt/task metadata and episode lifecycle on every five-step request. The server can ignore images on a blind decision; receiving images does not run their encoder. The current state must be fresh and correspond to the just-executed five steps. Add decision ID, executed-step count and acknowledgment for audit/retry robustness if extending the protocol. Task changes, duplicate IDs and partial execution invalidate or reject the blind candidate as appropriate.

If image transfer itself is to be removed, that is an additional protocol: send a state-only probe; if the server needs vision, return `NEED_VISION` **without action, append, or counter advance**, and retry the same decision ID with fresh images before advancing the environment. The existing client does not have this handshake; do not pretend that omitting images is already supported. Always supply vision on the first/post-MISS decision. The main proposed CPU/GPU experiment should keep the existing full request and isolate encoder savings first.

For the execute-more-steps baseline, keep a request every five control steps and serve consecutive tail blocks. This is equivalent to using more of the anchor chunk in ungated mode while retaining consistent accounting. If a client instead executes 10/15 steps without requests, it loses intermediate proprioceptive interruption and history; count the skipped five-step slots in the IR denominator and label that a different control arm.

### 3.5 Instrumentation and acceptance checks before closed loop

Log `vision_used`, source, anchor step, blind age, all gate bits and winning gate, all 16 rows/weights, actual served five-step action, normalized state, expected displacement, motion/residual, and stage-call counts. Keep synthetic member phase separate from actual retrieved progress. On blind decisions `s1_ms=0` and policy stage-2/3 work is zero; log CPU preflight/synthesis/output separately. Existing KPI accounting must change from `.152+.848*miss_share` to `.152*vision_share+.848*miss_share` and continue dividing by all five-step decisions. Retain wall-clock and policy-call counts as well as the reference IR.

Before any coordinator-run LIBERO pilot, require: B=0 parity with the old wrapper; a fake stage-1 implementation that raises if called on a blind request; stage-call count exactly V; correct dense buffers through anchor→blind→vision HIT→MISS→fresh anchor; episode reset and two interleaved clients; transform/action-wire parity on both models including gripper; retry/no-commit behavior; and exact single action broadcast/counter increment. These are proposed checks, not checks falsely claimed completed here. The diagnostics in this folder never implement the production bypass.

## 4. Rejected ideas

1. **Top-1 demonstration continuation as the default blind policy.** It discards a mixture of roughly 3–4 effective episodes at 50 and 5–8 at 500. Consecutive vision retrievals follow the exact successor only 18–29% at 50 and 17–43% at 500. At π0.5 l10 h=2, it raises proxy error by **+.1051 at 50 / +.1095 at 500** relative to advancing the whole kernel; GR00T l10 500 raises it **+.1375**. Keep it only as a mechanism ablation, not the lead design.

2. **Routine four-decision blindness, repeating the head, or extrapolating beyond the stored tail.** Four-decision phase serving on π0.5 l10 already adds **.152/.177** proxy error far from transitions at 50/500. Repeating the π0.5 spatial cache head produces h=4 error **1.163** at 50 versus phase **.685**; the full table shows the same qualitative failure elsewhere. π0.5 has only one true tail block and GR00T two. Mandatory vision bounds survive even when motion and gripper gates appear quiet; unseen object state is not encoded in proprioception.

3. **Absolute-state residual as the sole look gate, or state similarity as new member confidence.** Absolute residual>.5 fires in **36.7–53.7%** of h=1 cache windows at 50 and **24.3–44.6%** at 500, versus displacement residual **1.1–5.7%** and **0.6–4.9%**. It confounds a useful local phase match with scene/initial-state offset and spends many looks before addressing object failure. Temperature-.25 state reweighting worsens all eight h=1 cache comparisons. Absolute state remains useful for bounded per-member phase selection; it is not a validated replacement for visual anchoring or a calibrated confidence.

The pilot that decides the main question is a paired **vision-every-decision / tail / kernel-clock / phase-corrected** comparison with identical look gates and explicit policy-call cost, at both 50 and 500 episodes. Its decisive measurement is SR versus the new IR ledger, with physical failure phases—not the offline action-error ordering.

## 5. Numerical appendix

All four-value entries list blind horizons 1 / 2 / 3 / 4. Error is current-library σ-normalized RMS on 5×7 executed action blocks. These are immutable-path diagnostics, not closed-loop SR.



### Action degradation (matched future targets)

| query cell | library episodes | vision AWM | vision AWM all-HIT branch | top1 clock | kernel clock | phase abs | anchor chunk |
|---|---|---|---|---|---|---|---|
| pi05_spatial_inf | 50 | 0.340 / 0.346 / 0.354 / 0.360 | 0.428 / 0.436 / 0.447 / 0.456 | 0.497 / 0.577 / 0.607 / 0.632 | 0.426 / 0.489 / 0.515 / 0.535 | 0.426 / 0.471 / 0.497 / 0.517 | 0.425 / — / — / — |
| pi05_spatial_cache | 50 | 0.590 / 0.602 / 0.617 / 0.632 | 0.590 / 0.602 / 0.617 / 0.632 | 0.666 / 0.701 / 0.733 / 0.758 | 0.602 / 0.641 / 0.675 / 0.710 | 0.596 / 0.626 / 0.655 / 0.685 | 0.579 / — / — / — |
| pi05_l10_inf | 50 | 0.319 / 0.321 / 0.322 / 0.325 | 0.392 / 0.395 / 0.397 / 0.399 | 0.465 / 0.548 / 0.598 / 0.634 | 0.392 / 0.462 / 0.506 / 0.539 | 0.389 / 0.446 / 0.484 / 0.512 | 0.407 / — / — / — |
| pi05_l10_cache | 50 | 0.517 / 0.520 / 0.524 / 0.528 | 0.517 / 0.520 / 0.524 / 0.528 | 0.654 / 0.701 / 0.750 / 0.777 | 0.547 / 0.596 / 0.641 / 0.671 | 0.538 / 0.573 / 0.614 / 0.653 | 0.560 / — / — / — |
| groot_spatial_inf | 50 | 0.344 / 0.347 / 0.353 / 0.360 | 0.464 / 0.469 / 0.477 / 0.486 | 0.506 / 0.585 / 0.616 / 0.637 | 0.438 / 0.497 / 0.521 / 0.536 | 0.441 / 0.481 / 0.501 / 0.516 | 0.440 / 0.502 / — / — |
| groot_spatial_cache | 50 | 0.518 / 0.525 / 0.535 / 0.547 | 0.518 / 0.525 / 0.535 / 0.547 | 0.602 / 0.634 / 0.667 / 0.708 | 0.524 / 0.557 / 0.591 / 0.627 | 0.518 / 0.541 / 0.570 / 0.603 | 0.483 / 0.494 / — / — |
| groot_l10_inf | 50 | 0.333 / 0.335 / 0.337 / 0.339 | 0.463 / 0.466 / 0.468 / 0.470 | 0.502 / 0.610 / 0.664 / 0.698 | 0.431 / 0.518 / 0.567 / 0.595 | 0.421 / 0.496 / 0.540 / 0.572 | 0.437 / 0.531 / — / — |
| groot_l10_cache | 50 | 0.516 / 0.519 / 0.522 / 0.525 | 0.516 / 0.519 / 0.522 / 0.525 | 0.655 / 0.705 / 0.755 / 0.792 | 0.534 / 0.575 / 0.621 / 0.659 | 0.519 / 0.541 / 0.579 / 0.620 | 0.544 / 0.600 / — / — |
| pi05_spatial_inf | 500 | 0.259 / 0.262 / 0.268 / 0.273 | 0.308 / 0.313 / 0.319 / 0.324 | 0.404 / 0.473 / 0.512 / 0.549 | 0.340 / 0.396 / 0.426 / 0.450 | 0.339 / 0.385 / 0.411 / 0.432 | 0.346 / — / — / — |
| pi05_spatial_cache | 500 | 0.509 / 0.521 / 0.534 / 0.548 | 0.509 / 0.521 / 0.534 / 0.548 | 0.649 / 0.704 / 0.748 / 0.788 | 0.533 / 0.582 / 0.627 / 0.664 | 0.530 / 0.571 / 0.612 / 0.648 | 0.503 / — / — / — |
| pi05_l10_inf | 500 | 0.275 / 0.276 / 0.278 / 0.281 | 0.271 / 0.272 / 0.273 / 0.275 | 0.393 / 0.455 / 0.499 / 0.534 | 0.331 / 0.384 / 0.417 / 0.445 | 0.324 / 0.367 / 0.398 / 0.423 | 0.368 / — / — / — |
| pi05_l10_cache | 500 | 0.435 / 0.438 / 0.442 / 0.445 | 0.435 / 0.438 / 0.442 / 0.445 | 0.573 / 0.640 / 0.692 / 0.730 | 0.472 / 0.530 / 0.580 / 0.613 | 0.461 / 0.504 / 0.555 / 0.595 | 0.490 / — / — / — |
| groot_spatial_inf | 500 | 0.254 / 0.255 / 0.260 / 0.266 | 0.364 / 0.366 / 0.372 / 0.378 | 0.417 / 0.487 / 0.525 / 0.556 | 0.353 / 0.408 / 0.437 / 0.460 | 0.353 / 0.397 / 0.419 / 0.441 | 0.357 / 0.417 / — / — |
| groot_spatial_cache | 500 | 0.437 / 0.443 / 0.451 / 0.460 | 0.437 / 0.443 / 0.451 / 0.460 | 0.558 / 0.594 / 0.643 / 0.676 | 0.439 / 0.471 / 0.513 / 0.545 | 0.434 / 0.456 / 0.489 / 0.518 | 0.410 / 0.446 / — / — |
| groot_l10_inf | 500 | 0.269 / 0.271 / 0.273 / 0.275 | 0.352 / 0.354 / 0.355 / 0.356 | 0.417 / 0.499 / 0.549 / 0.586 | 0.365 / 0.435 / 0.475 / 0.503 | 0.351 / 0.416 / 0.454 / 0.482 | 0.376 / 0.456 / — / — |
| groot_l10_cache | 500 | 0.446 / 0.448 / 0.450 / 0.453 | 0.446 / 0.448 / 0.450 / 0.453 | 0.607 / 0.667 / 0.713 / 0.746 | 0.475 / 0.529 / 0.570 / 0.598 | 0.451 / 0.492 / 0.540 / 0.572 | 0.480 / 0.555 / — / — |


### Phase-aligned serving: err_mean

| query cell | library episodes | near transition | far transition | early | mid | late |
|---|---|---|---|---|---|---|
| pi05_spatial_inf | 50 | 0.514 / 0.544 / 0.542 / 0.548 | 0.394 / 0.442 / 0.474 / 0.500 | 0.345 / 0.390 / 0.423 / 0.452 | 0.479 / 0.525 / 0.543 / 0.557 | 0.447 / 0.480 / 0.497 / 0.509 |
| pi05_spatial_cache | 50 | 0.743 / 0.747 / 0.739 / 0.752 | 0.491 / 0.537 / 0.587 / 0.632 | 0.390 / 0.426 / 0.463 / 0.505 | 0.696 / 0.717 / 0.733 / 0.748 | 0.690 / 0.703 / 0.719 / 0.736 |
| pi05_l10_inf | 50 | 0.437 / 0.493 / 0.525 / 0.536 | 0.366 / 0.423 / 0.462 / 0.498 | 0.332 / 0.381 / 0.411 / 0.432 | 0.392 / 0.450 / 0.489 / 0.515 | 0.441 / 0.503 / 0.542 / 0.574 |
| pi05_l10_cache | 50 | 0.560 / 0.593 / 0.596 / 0.608 | 0.522 / 0.560 / 0.624 / 0.676 | 0.399 / 0.425 / 0.458 / 0.496 | 0.551 / 0.591 / 0.641 / 0.686 | 0.664 / 0.696 / 0.728 / 0.757 |
| groot_spatial_inf | 50 | 0.517 / 0.549 / 0.543 / 0.543 | 0.412 / 0.455 / 0.483 / 0.502 | 0.357 / 0.386 / 0.407 / 0.434 | 0.470 / 0.514 / 0.523 / 0.526 | 0.492 / 0.525 / 0.544 / 0.551 |
| groot_spatial_cache | 50 | 0.642 / 0.669 / 0.679 / 0.723 | 0.446 / 0.469 / 0.502 / 0.515 | 0.361 / 0.380 / 0.407 / 0.440 | 0.547 / 0.570 / 0.589 / 0.608 | 0.639 / 0.650 / 0.672 / 0.702 |
| groot_l10_inf | 50 | 0.463 / 0.517 / 0.552 / 0.567 | 0.399 / 0.485 / 0.533 / 0.576 | 0.366 / 0.429 / 0.466 / 0.496 | 0.425 / 0.499 / 0.543 / 0.573 | 0.472 / 0.555 / 0.602 / 0.635 |
| groot_l10_cache | 50 | 0.526 / 0.542 / 0.568 / 0.602 | 0.511 / 0.540 / 0.586 / 0.631 | 0.417 / 0.443 / 0.477 / 0.513 | 0.535 / 0.555 / 0.593 / 0.638 | 0.602 / 0.621 / 0.657 / 0.696 |
| pi05_spatial_inf | 500 | 0.424 / 0.464 / 0.467 / 0.478 | 0.311 / 0.356 / 0.387 / 0.410 | 0.279 / 0.323 / 0.352 / 0.381 | 0.381 / 0.428 / 0.450 / 0.464 | 0.353 / 0.389 / 0.409 / 0.425 |
| pi05_spatial_cache | 500 | 0.672 / 0.701 / 0.727 / 0.745 | 0.423 / 0.467 / 0.514 / 0.560 | 0.342 / 0.383 / 0.428 / 0.476 | 0.626 / 0.661 / 0.696 / 0.723 | 0.613 / 0.639 / 0.663 / 0.680 |
| pi05_l10_inf | 500 | 0.404 / 0.428 / 0.447 / 0.463 | 0.291 / 0.343 / 0.375 / 0.402 | 0.269 / 0.310 / 0.337 / 0.357 | 0.325 / 0.367 / 0.397 / 0.421 | 0.376 / 0.421 / 0.454 / 0.479 |
| pi05_l10_cache | 500 | 0.531 / 0.561 / 0.583 / 0.593 | 0.401 / 0.461 / 0.536 / 0.595 | 0.341 / 0.375 / 0.414 / 0.455 | 0.475 / 0.521 / 0.577 / 0.622 | 0.565 / 0.610 / 0.661 / 0.689 |
| groot_spatial_inf | 500 | 0.432 / 0.467 / 0.480 / 0.486 | 0.326 / 0.371 / 0.393 / 0.419 | 0.294 / 0.332 / 0.351 / 0.380 | 0.383 / 0.427 / 0.444 / 0.457 | 0.379 / 0.418 / 0.439 / 0.457 |
| groot_spatial_cache | 500 | 0.551 / 0.572 / 0.602 / 0.620 | 0.346 / 0.370 / 0.405 / 0.440 | 0.325 / 0.343 / 0.372 / 0.404 | 0.453 / 0.479 / 0.512 / 0.536 | 0.520 / 0.530 / 0.553 / 0.573 |
| groot_l10_inf | 500 | 0.440 / 0.472 / 0.498 / 0.515 | 0.307 / 0.390 / 0.430 / 0.464 | 0.296 / 0.355 / 0.389 / 0.414 | 0.361 / 0.426 / 0.465 / 0.490 | 0.394 / 0.462 / 0.501 / 0.531 |
| groot_l10_cache | 500 | 0.481 / 0.534 / 0.577 / 0.591 | 0.424 / 0.458 / 0.513 / 0.558 | 0.360 / 0.392 / 0.431 / 0.471 | 0.471 / 0.514 / 0.568 / 0.601 | 0.520 / 0.564 / 0.611 / 0.631 |


### Phase-aligned serving: delta_awm

| query cell | library episodes | near transition | far transition | early | mid | late |
|---|---|---|---|---|---|---|
| pi05_spatial_inf | 50 | 0.098 / 0.136 / 0.140 / 0.154 | 0.081 / 0.121 / 0.144 / 0.158 | 0.066 / 0.103 / 0.117 / 0.126 | 0.098 / 0.144 / 0.162 / 0.176 | 0.091 / 0.124 / 0.140 / 0.153 |
| pi05_spatial_cache | 50 | -0.005 / 0.014 / 0.025 / 0.042 | 0.014 / 0.031 / 0.049 / 0.062 | 0.011 / 0.030 / 0.042 / 0.052 | 0.015 / 0.036 / 0.051 / 0.067 | -0.008 / 0.005 / 0.021 / 0.038 |
| pi05_l10_inf | 50 | 0.077 / 0.138 / 0.172 / 0.187 | 0.067 / 0.120 / 0.155 / 0.187 | 0.063 / 0.109 / 0.135 / 0.152 | 0.068 / 0.126 / 0.165 / 0.191 | 0.080 / 0.141 / 0.180 / 0.212 |
| pi05_l10_cache | 50 | 0.015 / 0.048 / 0.055 / 0.074 | 0.027 / 0.057 / 0.110 / 0.152 | 0.017 / 0.038 / 0.065 / 0.096 | 0.022 / 0.062 / 0.112 / 0.156 | 0.026 / 0.058 / 0.090 / 0.119 |
| groot_spatial_inf | 50 | 0.108 / 0.144 / 0.141 / 0.147 | 0.093 / 0.130 / 0.151 / 0.161 | 0.075 / 0.103 / 0.114 / 0.122 | 0.101 / 0.145 / 0.154 / 0.157 | 0.114 / 0.147 / 0.166 / 0.173 |
| groot_spatial_cache | 50 | -0.017 / 0.002 / 0.017 / 0.060 | 0.010 / 0.025 / 0.045 / 0.053 | -0.005 / 0.010 / 0.021 / 0.031 | -0.001 / 0.022 / 0.041 / 0.060 | 0.005 / 0.016 / 0.038 / 0.068 |
| groot_l10_inf | 50 | 0.086 / 0.161 / 0.202 / 0.219 | 0.090 / 0.161 / 0.205 / 0.242 | 0.084 / 0.143 / 0.177 / 0.203 | 0.079 / 0.153 / 0.197 / 0.227 | 0.103 / 0.186 / 0.233 / 0.265 |
| groot_l10_cache | 50 | -0.014 / 0.003 / 0.030 / 0.070 | 0.017 / 0.039 / 0.076 / 0.111 | 0.001 / 0.022 / 0.051 / 0.083 | -0.002 / 0.018 / 0.056 / 0.100 | 0.008 / 0.027 / 0.063 / 0.101 |
| pi05_spatial_inf | 500 | 0.085 / 0.131 / 0.148 / 0.162 | 0.079 / 0.119 / 0.141 / 0.158 | 0.065 / 0.105 / 0.120 / 0.135 | 0.087 / 0.134 / 0.157 / 0.170 | 0.087 / 0.124 / 0.144 / 0.160 |
| pi05_spatial_cache | 500 | 0.011 / 0.042 / 0.067 / 0.088 | 0.028 / 0.057 / 0.086 / 0.111 | 0.027 / 0.054 / 0.077 / 0.097 | 0.027 / 0.062 / 0.097 / 0.125 | 0.008 / 0.033 / 0.057 / 0.075 |
| pi05_l10_inf | 500 | 0.051 / 0.086 / 0.112 / 0.133 | 0.048 / 0.093 / 0.124 / 0.147 | 0.048 / 0.087 / 0.110 / 0.126 | 0.042 / 0.084 / 0.114 / 0.139 | 0.056 / 0.102 / 0.134 / 0.159 |
| pi05_l10_cache | 500 | 0.026 / 0.058 / 0.090 / 0.108 | 0.025 / 0.071 / 0.129 / 0.177 | 0.019 / 0.048 / 0.081 / 0.114 | 0.027 / 0.072 / 0.128 / 0.174 | 0.031 / 0.076 / 0.127 / 0.155 |
| groot_spatial_inf | 500 | 0.115 / 0.154 / 0.169 / 0.176 | 0.093 / 0.136 / 0.154 / 0.174 | 0.088 / 0.130 / 0.141 / 0.155 | 0.103 / 0.147 / 0.165 / 0.178 | 0.105 / 0.144 / 0.164 / 0.183 |
| groot_spatial_cache | 500 | -0.026 / -0.006 / 0.029 / 0.050 | 0.013 / 0.028 / 0.045 / 0.065 | 0.004 / 0.020 / 0.035 / 0.049 | 0.002 / 0.028 / 0.062 / 0.085 | -0.017 / -0.007 / 0.016 / 0.035 |
| groot_l10_inf | 500 | 0.080 / 0.143 / 0.176 / 0.198 | 0.083 / 0.147 / 0.185 / 0.213 | 0.084 / 0.140 / 0.170 / 0.191 | 0.075 / 0.139 / 0.178 / 0.203 | 0.088 / 0.157 / 0.196 / 0.225 |
| groot_l10_cache | 500 | -0.004 / 0.049 / 0.095 / 0.108 | 0.013 / 0.039 / 0.086 / 0.127 | 0.006 / 0.035 / 0.071 / 0.106 | 0.002 / 0.045 / 0.099 / 0.132 | 0.006 / 0.050 / 0.097 / 0.117 |


### Phase-aligned serving: grip_mis

| query cell | library episodes | near transition | far transition | early | mid | late |
|---|---|---|---|---|---|---|
| pi05_spatial_inf | 50 | 0.141 / 0.143 / 0.117 / 0.117 | 0.012 / 0.018 / 0.024 / 0.033 | 0.020 / 0.026 / 0.026 / 0.032 | 0.080 / 0.086 / 0.083 / 0.092 | 0.036 / 0.042 / 0.045 / 0.048 |
| pi05_spatial_cache | 50 | 0.357 / 0.344 / 0.333 / 0.337 | 0.123 / 0.141 / 0.160 / 0.178 | 0.050 / 0.058 / 0.065 / 0.079 | 0.289 / 0.282 / 0.282 / 0.285 | 0.313 / 0.317 / 0.322 / 0.320 |
| pi05_l10_inf | 50 | 0.174 / 0.229 / 0.262 / 0.246 | 0.014 / 0.028 / 0.034 / 0.057 | 0.044 / 0.061 / 0.071 / 0.080 | 0.077 / 0.106 / 0.124 / 0.135 | 0.077 / 0.114 / 0.140 / 0.150 |
| pi05_l10_cache | 50 | 0.275 / 0.308 / 0.315 / 0.308 | 0.113 / 0.139 / 0.180 / 0.212 | 0.095 / 0.114 / 0.136 / 0.151 | 0.198 / 0.235 / 0.259 / 0.271 | 0.254 / 0.269 / 0.284 / 0.301 |
| groot_spatial_inf | 50 | 0.153 / 0.165 / 0.133 / 0.130 | 0.012 / 0.017 / 0.029 / 0.035 | 0.015 / 0.019 / 0.021 / 0.026 | 0.085 / 0.091 / 0.086 / 0.089 | 0.049 / 0.056 / 0.063 / 0.068 |
| groot_spatial_cache | 50 | 0.323 / 0.336 / 0.330 / 0.317 | 0.053 / 0.060 / 0.062 / 0.067 | 0.027 / 0.032 / 0.037 / 0.045 | 0.169 / 0.173 / 0.172 / 0.174 | 0.256 / 0.255 / 0.254 / 0.253 |
| groot_l10_inf | 50 | 0.198 / 0.225 / 0.239 / 0.222 | 0.032 / 0.056 / 0.067 / 0.095 | 0.046 / 0.064 / 0.082 / 0.093 | 0.114 / 0.145 / 0.161 / 0.167 | 0.109 / 0.132 / 0.147 / 0.162 |
| groot_l10_cache | 50 | 0.306 / 0.293 / 0.283 / 0.292 | 0.096 / 0.116 / 0.143 / 0.163 | 0.099 / 0.118 / 0.134 / 0.151 | 0.222 / 0.216 / 0.213 / 0.219 | 0.273 / 0.256 / 0.251 / 0.259 |
| pi05_spatial_inf | 500 | 0.107 / 0.118 / 0.112 / 0.110 | 0.007 / 0.011 / 0.016 / 0.020 | 0.012 / 0.019 / 0.022 / 0.031 | 0.053 / 0.061 / 0.067 / 0.070 | 0.028 / 0.032 / 0.036 / 0.037 |
| pi05_spatial_cache | 500 | 0.365 / 0.373 / 0.369 / 0.363 | 0.088 / 0.099 / 0.112 / 0.126 | 0.048 / 0.060 / 0.069 / 0.081 | 0.262 / 0.270 / 0.272 / 0.273 | 0.305 / 0.307 / 0.307 / 0.306 |
| pi05_l10_inf | 500 | 0.186 / 0.185 / 0.193 / 0.202 | 0.010 / 0.027 / 0.030 / 0.034 | 0.040 / 0.048 / 0.056 / 0.064 | 0.071 / 0.081 / 0.089 / 0.094 | 0.071 / 0.086 / 0.098 / 0.108 |
| pi05_l10_cache | 500 | 0.264 / 0.299 / 0.318 / 0.322 | 0.065 / 0.093 / 0.132 / 0.162 | 0.082 / 0.104 / 0.128 / 0.150 | 0.171 / 0.203 / 0.229 / 0.243 | 0.216 / 0.233 / 0.257 / 0.273 |
| groot_spatial_inf | 500 | 0.124 / 0.133 / 0.124 / 0.125 | 0.009 / 0.011 / 0.016 / 0.022 | 0.013 / 0.016 / 0.018 / 0.023 | 0.059 / 0.064 / 0.068 / 0.074 | 0.041 / 0.045 / 0.049 / 0.053 |
| groot_spatial_cache | 500 | 0.264 / 0.275 / 0.290 / 0.291 | 0.023 / 0.033 / 0.044 / 0.054 | 0.025 / 0.029 / 0.035 / 0.041 | 0.142 / 0.152 / 0.163 / 0.169 | 0.207 / 0.212 / 0.219 / 0.221 |
| groot_l10_inf | 500 | 0.233 / 0.232 / 0.232 / 0.236 | 0.015 / 0.046 / 0.056 / 0.072 | 0.044 / 0.060 / 0.071 / 0.082 | 0.110 / 0.132 / 0.144 / 0.152 | 0.106 / 0.122 / 0.133 / 0.149 |
| groot_l10_cache | 500 | 0.245 / 0.259 / 0.288 / 0.305 | 0.081 / 0.093 / 0.104 / 0.117 | 0.087 / 0.103 / 0.125 / 0.142 | 0.186 / 0.193 / 0.207 / 0.219 | 0.200 / 0.204 / 0.210 / 0.218 |


### Phase-conditioned comparator: kernel_clock minus phase_abs

Positive values favor phase alignment on the action-error proxy. These are paired windows, not SR effects.

| query cell | library episodes | near transition | far transition | early | mid | late |
|---|---|---|---|---|---|---|
| pi05_spatial_inf | 50 | -0.001 / 0.026 / 0.027 / 0.025 | 0.001 / 0.014 / 0.015 / 0.013 | 0.013 / 0.028 / 0.029 / 0.036 | -0.000 / 0.023 / 0.023 / 0.020 | -0.011 / 0.005 / 0.008 / 0.005 |
| pi05_spatial_cache | 50 | 0.007 / 0.013 / 0.015 / 0.023 | 0.006 / 0.017 / 0.024 / 0.027 | 0.015 / 0.030 / 0.035 / 0.041 | 0.005 / 0.014 / 0.022 / 0.028 | 0.000 / 0.002 / 0.006 / 0.011 |
| pi05_l10_inf | 50 | 0.007 / 0.027 / 0.036 / 0.042 | 0.001 / 0.010 / 0.015 / 0.019 | 0.009 / 0.020 / 0.026 / 0.030 | 0.003 / 0.017 / 0.022 / 0.028 | -0.004 / 0.010 / 0.019 / 0.024 |
| pi05_l10_cache | 50 | 0.023 / 0.040 / 0.034 / 0.031 | -0.002 / 0.011 / 0.023 / 0.012 | 0.003 / 0.019 / 0.028 / 0.029 | 0.016 / 0.034 / 0.036 / 0.025 | 0.006 / 0.014 / 0.017 / 0.003 |
| groot_spatial_inf | 50 | 0.002 / 0.027 / 0.031 / 0.026 | -0.005 / 0.011 / 0.015 / 0.017 | 0.010 / 0.025 / 0.036 / 0.044 | 0.001 / 0.023 / 0.026 / 0.027 | -0.019 / 0.001 / 0.001 / -0.001 |
| groot_spatial_cache | 50 | 0.016 / 0.016 / 0.016 / 0.019 | 0.002 / 0.015 / 0.025 / 0.029 | 0.011 / 0.022 / 0.030 / 0.033 | 0.011 / 0.022 / 0.024 / 0.029 | -0.002 / 0.003 / 0.012 / 0.014 |
| groot_l10_inf | 50 | 0.027 / 0.036 / 0.042 / 0.037 | 0.001 / 0.014 / 0.017 / 0.015 | 0.010 / 0.023 / 0.028 / 0.029 | 0.013 / 0.025 / 0.026 / 0.022 | 0.007 / 0.018 / 0.025 / 0.020 |
| groot_l10_cache | 50 | 0.023 / 0.049 / 0.061 / 0.054 | 0.009 / 0.021 / 0.029 / 0.031 | 0.020 / 0.032 / 0.039 / 0.041 | 0.018 / 0.039 / 0.047 / 0.042 | 0.010 / 0.031 / 0.040 / 0.035 |
| pi05_spatial_inf | 500 | 0.002 / 0.019 / 0.022 / 0.023 | 0.000 / 0.008 / 0.012 / 0.016 | 0.005 / 0.011 / 0.013 / 0.017 | 0.004 / 0.017 / 0.022 / 0.024 | -0.007 / 0.004 / 0.009 / 0.012 |
| pi05_spatial_cache | 500 | 0.016 / 0.018 / 0.021 / 0.019 | -0.007 / 0.005 / 0.010 / 0.014 | 0.001 / 0.013 / 0.021 / 0.027 | 0.001 / 0.014 / 0.022 / 0.020 | 0.006 / 0.005 / 0.004 / 0.005 |
| pi05_l10_inf | 500 | 0.018 / 0.021 / 0.027 / 0.032 | 0.004 / 0.014 / 0.015 / 0.017 | 0.010 / 0.015 / 0.017 / 0.020 | 0.011 / 0.020 / 0.022 / 0.025 | 0.003 / 0.013 / 0.017 / 0.021 |
| pi05_l10_cache | 500 | 0.021 / 0.035 / 0.033 / 0.028 | 0.003 / 0.020 / 0.019 / 0.013 | 0.005 / 0.016 / 0.025 / 0.027 | 0.017 / 0.037 / 0.034 / 0.025 | 0.013 / 0.026 / 0.015 / 0.006 |
| groot_spatial_inf | 500 | 0.007 / 0.020 / 0.028 / 0.028 | -0.002 / 0.009 / 0.014 / 0.016 | 0.005 / 0.014 / 0.024 / 0.031 | 0.004 / 0.019 / 0.025 / 0.027 | -0.009 / 0.002 / 0.006 / 0.006 |
| groot_spatial_cache | 500 | 0.011 / 0.021 / 0.027 / 0.025 | 0.001 / 0.011 / 0.022 / 0.028 | 0.006 / 0.013 / 0.022 / 0.033 | 0.007 / 0.024 / 0.032 / 0.032 | 0.002 / 0.008 / 0.016 / 0.016 |
| groot_l10_inf | 500 | 0.033 / 0.026 / 0.032 / 0.032 | 0.005 / 0.016 / 0.015 / 0.014 | 0.011 / 0.017 / 0.019 / 0.022 | 0.015 / 0.022 / 0.021 / 0.019 | 0.016 / 0.019 / 0.022 / 0.022 |
| groot_l10_cache | 500 | 0.046 / 0.057 / 0.040 / 0.031 | 0.005 / 0.021 / 0.022 / 0.022 | 0.015 / 0.027 / 0.035 / 0.037 | 0.031 / 0.047 / 0.034 / 0.027 | 0.027 / 0.037 / 0.020 / 0.013 |


### Phase-conditioned comparator: anchor_chunk minus phase_abs

Positive values favor phase alignment on the action-error proxy. These are paired windows, not SR effects.

| query cell | library episodes | near transition | far transition | early | mid | late |
|---|---|---|---|---|---|---|
| pi05_spatial_inf | 50 | -0.007 / — / — / — | 0.001 / — / — / — | 0.013 / — / — / — | 0.001 / — / — / — | -0.015 / — / — / — |
| pi05_spatial_cache | 50 | -0.043 / — / — / — | 0.002 / — / — / — | 0.025 / — / — / — | -0.015 / — / — / — | -0.057 / — / — / — |
| pi05_l10_inf | 50 | 0.023 / — / — / — | 0.016 / — / — / — | 0.020 / — / — / — | 0.016 / — / — / — | 0.020 / — / — / — |
| pi05_l10_cache | 50 | 0.043 / — / — / — | 0.005 / — / — / — | 0.013 / — / — / — | 0.029 / — / — / — | 0.022 / — / — / — |
| groot_spatial_inf | 50 | -0.001 / 0.029 / — / — | -0.001 / 0.017 / — / — | 0.012 / 0.032 / — / — | 0.001 / 0.024 / — / — | -0.017 / 0.008 / — / — |
| groot_spatial_cache | 50 | -0.088 / -0.130 / — / — | -0.004 / -0.001 / — / — | 0.015 / 0.033 / — / — | -0.038 / -0.047 / — / — | -0.080 / -0.117 / — / — |
| groot_l10_inf | 50 | 0.038 / 0.056 / — / — | 0.003 / 0.023 / — / — | 0.013 / 0.037 / — / — | 0.021 / 0.035 / — / — | 0.012 / 0.032 / — / — |
| groot_l10_cache | 50 | 0.035 / 0.086 / — / — | 0.016 / 0.034 / — / — | 0.032 / 0.060 / — / — | 0.030 / 0.070 / — / — | 0.014 / 0.045 / — / — |
| pi05_spatial_inf | 500 | 0.008 / — / — / — | 0.006 / — / — / — | 0.013 / — / — / — | 0.009 / — / — / — | -0.002 / — / — / — |
| pi05_spatial_cache | 500 | -0.048 / — / — / — | -0.011 / — / — / — | 0.009 / — / — / — | -0.031 / — / — / — | -0.057 / — / — / — |
| pi05_l10_inf | 500 | 0.055 / — / — / — | 0.040 / — / — / — | 0.030 / — / — / — | 0.056 / — / — / — | 0.046 / — / — / — |
| pi05_l10_cache | 500 | 0.041 / — / — / — | 0.019 / — / — / — | 0.020 / — / — / — | 0.033 / — / — / — | 0.034 / — / — / — |
| groot_spatial_inf | 500 | 0.007 / 0.030 / — / — | 0.003 / 0.017 / — / — | 0.008 / 0.020 / — / — | 0.008 / 0.028 / — / — | -0.004 / 0.013 / — / — |
| groot_spatial_cache | 500 | -0.056 / -0.041 / — / — | -0.000 / 0.013 / — / — | 0.007 / 0.021 / — / — | -0.022 / 0.001 / — / — | -0.057 / -0.049 / — / — |
| groot_l10_inf | 500 | 0.046 / 0.049 / — / — | 0.014 / 0.035 / — / — | 0.016 / 0.030 / — / — | 0.031 / 0.045 / — / — | 0.027 / 0.043 / — / — |
| groot_l10_cache | 500 | 0.052 / 0.094 / — / — | 0.009 / 0.038 / — / — | 0.027 / 0.053 / — / — | 0.035 / 0.074 / — / — | 0.026 / 0.062 / — / — |


### Serving ablations

| query cell | library episodes | absolute phase | delta phase | hybrid phase | state reweight T=.25 | state reweight T=1 | repeat anchor head |
|---|---|---|---|---|---|---|---|
| pi05_spatial_inf | 50 | 0.426 / 0.471 / 0.497 / 0.517 | 0.425 / 0.486 / 0.511 / 0.524 | 0.421 / 0.475 / 0.499 / 0.517 | 0.427 / 0.466 / 0.488 / 0.507 | 0.423 / 0.465 / 0.489 / 0.509 | 0.576 / 0.866 / 1.089 / 1.242 |
| pi05_spatial_cache | 50 | 0.596 / 0.626 / 0.655 / 0.685 | 0.602 / 0.628 / 0.653 / 0.681 | 0.595 / 0.623 / 0.651 / 0.682 | 0.615 / 0.635 / 0.656 / 0.679 | 0.601 / 0.625 / 0.650 / 0.676 | 0.723 / 0.913 / 1.061 / 1.163 |
| pi05_l10_inf | 50 | 0.389 / 0.446 / 0.484 / 0.512 | 0.386 / 0.450 / 0.488 / 0.515 | 0.387 / 0.448 / 0.486 / 0.513 | 0.393 / 0.444 / 0.471 / 0.490 | 0.387 / 0.440 / 0.472 / 0.495 | 0.466 / 0.675 / 0.845 / 0.971 |
| pi05_l10_cache | 50 | 0.538 / 0.573 / 0.614 / 0.653 | 0.536 / 0.570 / 0.612 / 0.652 | 0.537 / 0.572 / 0.613 / 0.653 | 0.572 / 0.598 / 0.628 / 0.663 | 0.552 / 0.583 / 0.621 / 0.658 | 0.598 / 0.727 / 0.833 / 0.914 |
| groot_spatial_inf | 50 | 0.441 / 0.481 / 0.501 / 0.516 | 0.436 / 0.490 / 0.512 / 0.523 | 0.435 / 0.482 / 0.504 / 0.518 | 0.445 / 0.480 / 0.498 / 0.515 | 0.436 / 0.474 / 0.494 / 0.509 | 0.597 / 0.883 / 1.096 / 1.243 |
| groot_spatial_cache | 50 | 0.518 / 0.541 / 0.570 / 0.603 | 0.521 / 0.546 / 0.567 / 0.596 | 0.518 / 0.541 / 0.569 / 0.600 | 0.534 / 0.553 / 0.586 / 0.640 | 0.522 / 0.543 / 0.576 / 0.620 | 0.681 / 0.901 / 1.068 / 1.179 |
| groot_l10_inf | 50 | 0.421 / 0.496 / 0.540 / 0.572 | 0.419 / 0.501 / 0.547 / 0.581 | 0.418 / 0.498 / 0.543 / 0.576 | 0.431 / 0.494 / 0.523 / 0.543 | 0.421 / 0.489 / 0.525 / 0.551 | 0.500 / 0.710 / 0.876 / 1.003 |
| groot_l10_cache | 50 | 0.519 / 0.541 / 0.579 / 0.620 | 0.524 / 0.550 / 0.585 / 0.627 | 0.521 / 0.544 / 0.581 / 0.624 | 0.562 / 0.573 / 0.600 / 0.637 | 0.536 / 0.553 / 0.588 / 0.627 | 0.584 / 0.696 / 0.793 / 0.868 |
| pi05_spatial_inf | 500 | 0.339 / 0.385 / 0.411 / 0.432 | 0.339 / 0.391 / 0.418 / 0.437 | 0.336 / 0.387 / 0.413 / 0.433 | 0.329 / 0.370 / 0.393 / 0.410 | 0.333 / 0.377 / 0.402 / 0.421 | 0.577 / 0.896 / 1.124 / 1.279 |
| pi05_spatial_cache | 500 | 0.530 / 0.571 / 0.612 / 0.648 | 0.527 / 0.567 / 0.607 / 0.644 | 0.525 / 0.567 / 0.608 / 0.645 | 0.569 / 0.602 / 0.633 / 0.661 | 0.547 / 0.584 / 0.619 / 0.650 | 0.685 / 0.891 / 1.043 / 1.146 |
| pi05_l10_inf | 500 | 0.324 / 0.367 / 0.398 / 0.423 | 0.325 / 0.372 / 0.402 / 0.427 | 0.324 / 0.370 / 0.400 / 0.424 | 0.318 / 0.356 / 0.379 / 0.395 | 0.321 / 0.362 / 0.390 / 0.410 | 0.460 / 0.691 / 0.867 / 0.995 |
| pi05_l10_cache | 500 | 0.461 / 0.504 / 0.555 / 0.595 | 0.456 / 0.500 / 0.554 / 0.596 | 0.458 / 0.502 / 0.554 / 0.595 | 0.482 / 0.514 / 0.557 / 0.597 | 0.468 / 0.506 / 0.554 / 0.595 | 0.540 / 0.682 / 0.797 / 0.881 |
| groot_spatial_inf | 500 | 0.353 / 0.397 / 0.419 / 0.441 | 0.352 / 0.402 / 0.425 / 0.445 | 0.349 / 0.398 / 0.420 / 0.441 | 0.344 / 0.382 / 0.401 / 0.419 | 0.348 / 0.389 / 0.410 / 0.430 | 0.590 / 0.902 / 1.118 / 1.263 |
| groot_spatial_cache | 500 | 0.434 / 0.456 / 0.489 / 0.518 | 0.438 / 0.457 / 0.485 / 0.514 | 0.433 / 0.455 / 0.486 / 0.516 | 0.436 / 0.455 / 0.479 / 0.503 | 0.433 / 0.453 / 0.481 / 0.506 | 0.637 / 0.871 / 1.041 / 1.154 |
| groot_l10_inf | 500 | 0.351 / 0.416 / 0.454 / 0.482 | 0.353 / 0.423 / 0.462 / 0.488 | 0.351 / 0.419 / 0.458 / 0.485 | 0.345 / 0.403 / 0.431 / 0.450 | 0.346 / 0.409 / 0.443 / 0.466 | 0.489 / 0.725 / 0.901 / 1.030 |
| groot_l10_cache | 500 | 0.451 / 0.492 / 0.540 / 0.572 | 0.453 / 0.493 / 0.542 / 0.575 | 0.451 / 0.491 / 0.541 / 0.573 | 0.466 / 0.494 / 0.534 / 0.561 | 0.454 / 0.489 / 0.535 / 0.565 | 0.530 / 0.653 / 0.757 / 0.837 |


### Independent trigger firing rates over all anchor windows

| query cell | library episodes | grip ahead | near terminal | still 2 | delta residual >.5 | absolute residual >.5 | union |
|---|---|---|---|---|---|---|---|
| pi05_spatial_inf | 50 | 0.230 / 0.251 / 0.263 / 0.285 | 0.155 / 0.167 / 0.176 / 0.189 | 0.028 / 0.029 / 0.025 / 0.027 | 0.010 / 0.054 / 0.140 / 0.220 | 0.334 / 0.344 / 0.367 / 0.392 | 0.341 / 0.394 / 0.459 / 0.531 |
| pi05_spatial_cache | 50 | 0.370 / 0.381 / 0.380 / 0.396 | 0.340 / 0.357 / 0.377 / 0.397 | 0.138 / 0.144 / 0.146 / 0.152 | 0.011 / 0.139 / 0.257 / 0.342 | 0.410 / 0.425 / 0.447 / 0.476 | 0.511 / 0.558 / 0.595 / 0.646 |
| pi05_l10_inf | 50 | 0.262 / 0.275 / 0.270 / 0.259 | 0.076 / 0.084 / 0.097 / 0.111 | 0.129 / 0.131 / 0.124 / 0.123 | 0.017 / 0.057 / 0.111 / 0.190 | 0.398 / 0.408 / 0.430 / 0.453 | 0.400 / 0.442 / 0.473 / 0.517 |
| pi05_l10_cache | 50 | 0.329 / 0.285 / 0.260 / 0.239 | 0.159 / 0.172 / 0.191 / 0.215 | 0.206 / 0.208 / 0.206 / 0.206 | 0.038 / 0.166 / 0.278 / 0.361 | 0.497 / 0.519 / 0.538 / 0.554 | 0.523 / 0.573 / 0.608 / 0.643 |
| groot_spatial_inf | 50 | 0.229 / 0.236 / 0.239 / 0.279 | 0.146 / 0.152 / 0.163 / 0.182 | 0.055 / 0.058 / 0.048 / 0.048 | 0.010 / 0.061 / 0.158 / 0.251 | 0.348 / 0.357 / 0.385 / 0.409 | 0.352 / 0.408 / 0.454 / 0.542 |
| groot_spatial_cache | 50 | 0.293 / 0.304 / 0.313 / 0.359 | 0.281 / 0.291 / 0.318 / 0.358 | 0.139 / 0.144 / 0.140 / 0.145 | 0.043 / 0.114 / 0.171 / 0.225 | 0.367 / 0.388 / 0.411 / 0.436 | 0.464 / 0.513 / 0.540 / 0.594 |
| groot_l10_inf | 50 | 0.282 / 0.298 / 0.294 / 0.279 | 0.075 / 0.087 / 0.098 / 0.113 | 0.117 / 0.120 / 0.110 / 0.108 | 0.031 / 0.096 / 0.162 / 0.247 | 0.458 / 0.468 / 0.488 / 0.510 | 0.415 / 0.467 / 0.501 / 0.545 |
| groot_l10_cache | 50 | 0.393 / 0.341 / 0.307 / 0.278 | 0.129 / 0.143 / 0.152 / 0.159 | 0.205 / 0.207 / 0.203 / 0.202 | 0.057 / 0.168 / 0.285 / 0.378 | 0.537 / 0.561 / 0.588 / 0.611 | 0.548 / 0.595 / 0.634 / 0.670 |
| pi05_spatial_inf | 500 | 0.202 / 0.224 / 0.232 / 0.254 | 0.133 / 0.143 / 0.155 / 0.172 | 0.023 / 0.024 / 0.020 / 0.020 | 0.004 / 0.021 / 0.052 / 0.088 | 0.152 / 0.156 / 0.170 / 0.191 | 0.303 / 0.346 / 0.380 / 0.431 |
| pi05_spatial_cache | 500 | 0.376 / 0.373 / 0.388 / 0.400 | 0.337 / 0.356 / 0.371 / 0.396 | 0.128 / 0.132 / 0.133 / 0.137 | 0.007 / 0.096 / 0.212 / 0.297 | 0.379 / 0.420 / 0.456 / 0.492 | 0.506 / 0.545 / 0.583 / 0.638 |
| pi05_l10_inf | 500 | 0.216 / 0.250 / 0.264 / 0.264 | 0.065 / 0.074 / 0.082 / 0.089 | 0.075 / 0.077 / 0.072 / 0.071 | 0.016 / 0.050 / 0.072 / 0.102 | 0.181 / 0.202 / 0.217 / 0.234 | 0.325 / 0.376 / 0.400 / 0.422 |
| pi05_l10_cache | 500 | 0.352 / 0.323 / 0.314 / 0.299 | 0.128 / 0.145 / 0.154 / 0.166 | 0.173 / 0.176 / 0.174 / 0.176 | 0.049 / 0.145 / 0.238 / 0.304 | 0.399 / 0.439 / 0.469 / 0.496 | 0.505 / 0.544 / 0.584 / 0.618 |
| groot_spatial_inf | 500 | 0.204 / 0.230 / 0.235 / 0.257 | 0.132 / 0.146 / 0.165 / 0.180 | 0.037 / 0.038 / 0.033 / 0.033 | 0.004 / 0.023 / 0.053 / 0.096 | 0.172 / 0.181 / 0.197 / 0.216 | 0.307 / 0.350 / 0.380 / 0.431 |
| groot_spatial_cache | 500 | 0.368 / 0.363 / 0.369 / 0.380 | 0.270 / 0.289 / 0.310 / 0.335 | 0.127 / 0.132 / 0.129 / 0.133 | 0.006 / 0.057 / 0.119 / 0.176 | 0.243 / 0.272 / 0.301 / 0.325 | 0.458 / 0.488 / 0.514 / 0.553 |
| groot_l10_inf | 500 | 0.251 / 0.278 / 0.282 / 0.274 | 0.061 / 0.068 / 0.074 / 0.088 | 0.070 / 0.071 / 0.067 / 0.066 | 0.022 / 0.065 / 0.103 / 0.145 | 0.216 / 0.238 / 0.257 / 0.278 | 0.351 / 0.401 / 0.423 / 0.450 |
| groot_l10_cache | 500 | 0.370 / 0.342 / 0.333 / 0.327 | 0.099 / 0.115 / 0.129 / 0.141 | 0.145 / 0.147 / 0.145 / 0.146 | 0.047 / 0.164 / 0.246 / 0.317 | 0.446 / 0.476 / 0.502 / 0.525 | 0.501 / 0.547 / 0.590 / 0.631 |


### Scheduled budgets 1 / 2 / 3 / 4, phase + state gates

| query cell | library episodes | vision share | pure-cache reference IR | all-row Δerr | blind-row Δerr |
|---|---|---|---|---|---|
| pi05_spatial_inf | 50 | 0.655 / 0.543 / 0.486 / 0.459 | 0.100 / 0.082 / 0.074 / 0.070 | 0.027 / 0.044 / 0.055 / 0.059 | 0.079 / 0.095 / 0.107 / 0.109 |
| pi05_spatial_cache | 50 | 0.741 / 0.656 / 0.620 / 0.595 | 0.113 / 0.100 / 0.094 / 0.090 | 0.003 / 0.008 / 0.010 / 0.010 | 0.013 / 0.023 / 0.026 / 0.026 |
| pi05_l10_inf | 50 | 0.677 / 0.571 / 0.518 / 0.491 | 0.103 / 0.087 / 0.079 / 0.075 | 0.022 / 0.037 / 0.047 / 0.055 | 0.068 / 0.086 / 0.097 / 0.107 |
| pi05_l10_cache | 50 | 0.744 / 0.665 / 0.627 / 0.603 | 0.113 / 0.101 / 0.095 / 0.092 | 0.005 / 0.008 / 0.011 / 0.012 | 0.019 / 0.025 / 0.029 / 0.030 |
| groot_spatial_inf | 50 | 0.657 / 0.551 / 0.493 / 0.467 | 0.100 / 0.084 / 0.075 / 0.071 | 0.032 / 0.046 / 0.057 / 0.062 | 0.093 / 0.103 / 0.113 / 0.116 |
| groot_spatial_cache | 50 | 0.712 / 0.623 / 0.575 / 0.555 | 0.108 / 0.095 / 0.087 / 0.084 | 0.001 / 0.003 / 0.005 / 0.007 | 0.003 / 0.007 / 0.012 / 0.015 |
| groot_l10_inf | 50 | 0.683 / 0.579 / 0.529 / 0.502 | 0.104 / 0.088 / 0.080 / 0.076 | 0.028 / 0.049 / 0.061 / 0.070 | 0.090 / 0.116 / 0.130 / 0.140 |
| groot_l10_cache | 50 | 0.754 / 0.680 / 0.645 / 0.628 | 0.115 / 0.103 / 0.098 / 0.095 | 0.002 / 0.005 / 0.007 / 0.009 | 0.009 / 0.015 / 0.019 / 0.023 |
| pi05_spatial_inf | 500 | 0.635 / 0.521 / 0.459 / 0.430 | 0.097 / 0.079 / 0.070 / 0.065 | 0.028 / 0.047 / 0.058 / 0.065 | 0.077 / 0.098 / 0.107 / 0.114 |
| pi05_spatial_cache | 500 | 0.738 / 0.657 / 0.617 / 0.592 | 0.112 / 0.100 / 0.094 / 0.090 | 0.008 / 0.012 / 0.018 / 0.021 | 0.029 / 0.036 / 0.046 / 0.052 |
| pi05_l10_inf | 500 | 0.639 / 0.523 / 0.462 / 0.430 | 0.097 / 0.080 / 0.070 / 0.065 | 0.019 / 0.033 / 0.043 / 0.052 | 0.053 / 0.069 / 0.081 / 0.091 |
| pi05_l10_cache | 500 | 0.736 / 0.653 / 0.611 / 0.589 | 0.112 / 0.099 / 0.093 / 0.090 | 0.006 / 0.012 / 0.017 / 0.021 | 0.024 / 0.034 / 0.043 / 0.051 |
| groot_spatial_inf | 500 | 0.640 / 0.523 / 0.466 / 0.433 | 0.097 / 0.080 / 0.071 / 0.066 | 0.034 / 0.056 / 0.068 / 0.076 | 0.095 / 0.118 / 0.127 / 0.135 |
| groot_spatial_cache | 500 | 0.712 / 0.621 / 0.574 / 0.548 | 0.108 / 0.094 / 0.087 / 0.083 | 0.003 / 0.007 / 0.010 / 0.014 | 0.010 / 0.017 / 0.024 / 0.031 |
| groot_l10_inf | 500 | 0.653 / 0.540 / 0.482 / 0.450 | 0.099 / 0.082 / 0.073 / 0.068 | 0.029 / 0.051 / 0.064 / 0.074 | 0.083 / 0.110 / 0.124 / 0.135 |
| groot_l10_cache | 500 | 0.733 / 0.650 / 0.610 / 0.589 | 0.111 / 0.099 / 0.093 / 0.090 | 0.002 / 0.005 / 0.010 / 0.013 | 0.007 / 0.015 / 0.025 / 0.031 |


### Closed-loop frozen-path IR (MISSes kept at their observed decisions)

| arm | library episodes | MISS share | original IR | budget-only caps 1–4 | phase caps 1–4 | phase + vote caps 1–4 |
|---|---|---|---|---|---|---|
| oscl500_g_l10_cl2 | 500 | 0.000 | 0.152 | 0.076 / 0.051 / 0.039 / 0.031 | 0.108 / 0.092 / 0.085 / 0.080 | 0.111 / 0.097 / 0.090 / 0.086 |
| oscl500_g_sp_cl2 | 500 | 0.000 | 0.152 | 0.078 / 0.053 / 0.041 / 0.034 | 0.097 / 0.080 / 0.070 / 0.065 | 0.098 / 0.080 / 0.071 / 0.066 |
| oscl500_p_l10_cl2 | 500 | 0.000 | 0.152 | 0.076 / 0.051 / 0.039 / 0.031 | 0.101 / 0.084 / 0.076 / 0.070 | 0.103 / 0.086 / 0.078 / 0.073 |
| oscl500_p_sp_cl2 | 500 | 0.000 | 0.152 | 0.078 / 0.053 / 0.040 / 0.033 | 0.099 / 0.082 / 0.073 / 0.067 | 0.099 / 0.082 / 0.073 / 0.068 |
| oscl50_g_l10_cl2 | 50 | 0.000 | 0.152 | 0.076 / 0.051 / 0.038 / 0.031 | 0.109 / 0.094 / 0.087 / 0.083 | 0.113 / 0.099 / 0.093 / 0.089 |
| oscl50_g_sp_cl2 | 50 | 0.000 | 0.152 | 0.077 / 0.053 / 0.040 / 0.033 | 0.103 / 0.087 / 0.080 / 0.075 | 0.104 / 0.087 / 0.080 / 0.076 |
| oscl50_p_l10_cl2 | 50 | 0.000 | 0.152 | 0.076 / 0.051 / 0.039 / 0.031 | 0.104 / 0.088 / 0.081 / 0.076 | 0.107 / 0.092 / 0.085 / 0.080 |
| oscl50_p_sp_cl2 | 50 | 0.000 | 0.152 | 0.077 / 0.053 / 0.040 / 0.032 | 0.105 / 0.090 / 0.082 / 0.078 | 0.107 / 0.091 / 0.084 / 0.079 |
| r3mx_p_l10_awm500_h70 | 500 | 0.341 | 0.441 | 0.397 / 0.383 / 0.376 / 0.372 | 0.405 / 0.393 / 0.387 / 0.383 | 0.405 / 0.393 / 0.388 / 0.384 |
| r3mx_p_l10_g | 50 | 0.202 | 0.323 | 0.266 / 0.247 / 0.238 / 0.233 | 0.280 / 0.265 / 0.258 / 0.254 | 0.281 / 0.266 / 0.260 / 0.256 |
| r3mx_p_l10_g500 | 500 | 0.102 | 0.238 | 0.172 / 0.151 / 0.140 / 0.134 | 0.187 / 0.171 / 0.162 / 0.157 | 0.188 / 0.172 / 0.164 / 0.159 |
| r3mx_p_l10_perk3 | 50 | 0.327 | 0.429 | 0.378 / 0.378 / 0.378 / 0.378 | 0.394 / 0.394 / 0.394 / 0.394 | 0.395 / 0.395 / 0.395 / 0.395 |
| r3mx_p_l10_perk5 | 50 | 0.192 | 0.315 | 0.254 / 0.253 / 0.223 / 0.223 | 0.274 / 0.268 / 0.251 / 0.251 | 0.275 / 0.269 / 0.253 / 0.253 |
| r3mx_p_sp_awm500_h70 | 500 | 0.337 | 0.438 | 0.398 / 0.385 / 0.379 / 0.375 | 0.404 / 0.394 / 0.388 / 0.385 | 0.404 / 0.394 / 0.388 / 0.385 |
| r3mx_p_sp_awm_h70 | 50 | 0.328 | 0.430 | 0.389 / 0.376 / 0.370 / 0.366 | 0.397 / 0.386 / 0.381 / 0.378 | 0.397 / 0.387 / 0.381 / 0.379 |
| r3mx_p_sp_g | 50 | 0.135 | 0.266 | 0.204 / 0.183 / 0.173 / 0.167 | 0.218 / 0.202 / 0.195 / 0.190 | 0.219 / 0.203 / 0.196 / 0.192 |


### Failed first-spell trigger coverage: at onset / onset or previous 2 / onset or previous 4

Retrospective flags use the actual vision kernel at every recorded decision. The next table instead forecasts each flag from the last retained anchor.

| arm | failed episodes with spell | grip ahead | near terminal | union |
|---|---|---|---|---|
| oscl500_g_l10_cl2 | 146 | 0.795 / 0.842 / 0.870 | 0.014 / 0.014 / 0.014 | 0.808 / 0.856 / 0.884 |
| oscl500_g_sp_cl2 | 15 | 0.733 / 0.733 / 0.733 | 0.667 / 0.733 / 0.733 | 0.933 / 0.933 / 0.933 |
| oscl500_p_l10_cl2 | 115 | 0.626 / 0.722 / 0.730 | 0.061 / 0.078 / 0.087 | 0.652 / 0.757 / 0.765 |
| oscl500_p_sp_cl2 | 23 | 0.783 / 0.783 / 0.826 | 0.565 / 0.565 / 0.565 | 0.913 / 0.913 / 0.957 |
| oscl50_g_l10_cl2 | 224 | 0.616 / 0.701 / 0.723 | 0.013 / 0.013 / 0.013 | 0.625 / 0.705 / 0.728 |
| oscl50_g_sp_cl2 | 56 | 0.625 / 0.679 / 0.732 | 0.911 / 0.911 / 0.929 | 0.964 / 0.964 / 0.982 |
| oscl50_p_l10_cl2 | 184 | 0.603 / 0.652 / 0.690 | 0.076 / 0.076 / 0.076 | 0.641 / 0.690 / 0.728 |
| oscl50_p_sp_cl2 | 100 | 0.590 / 0.660 / 0.700 | 0.420 / 0.420 / 0.420 | 0.720 / 0.790 / 0.830 |


### Causal anchor forecast: phase-gated cap 2, failed first-spell coverage

| arm | failed episodes with spell | at onset / onset or previous 2 / onset or previous 4 |
|---|---|---|
| oscl500_g_l10_cl2 | 146 | 0.774 / 0.870 / 0.890 |
| oscl500_g_sp_cl2 | 15 | 0.933 / 0.933 / 0.933 |
| oscl500_p_l10_cl2 | 115 | 0.617 / 0.791 / 0.809 |
| oscl500_p_sp_cl2 | 23 | 0.870 / 0.913 / 0.957 |
| oscl50_g_l10_cl2 | 224 | 0.621 / 0.719 / 0.759 |
| oscl50_g_sp_cl2 | 56 | 0.964 / 0.982 / 0.982 |
| oscl50_p_l10_cl2 | 184 | 0.717 / 0.804 / 0.848 |
| oscl50_p_sp_cl2 | 100 | 0.650 / 0.710 / 0.860 |


### Pure-cache path anatomy

Consecutive-pick rates exclude episode starts. Phase thirds are fractions of the recorded episode length and are evaluation labels only.

| cell | episodes | decisions | SR | same episode | exact successor | same row | grip trigger early / mid / late | terminal trigger early / mid / late |
|---|---|---|---|---|---|---|---|---|
| pi05_spatial | 50 | 12718 | 0.800 | 0.569 | 0.291 | 0.248 | 0.205 / 0.361 / 0.371 | 0.000 / 0.168 / 0.562 |
| pi05_spatial | 500 | 10990 | 0.954 | 0.490 | 0.427 | 0.051 | 0.141 / 0.301 / 0.160 | 0.000 / 0.045 / 0.393 |
| pi05_l10 | 50 | 36192 | 0.630 | 0.713 | 0.291 | 0.360 | 0.250 / 0.324 / 0.312 | 0.003 / 0.087 / 0.260 |
| pi05_l10 | 500 | 31716 | 0.768 | 0.624 | 0.348 | 0.226 | 0.229 / 0.339 / 0.331 | 0.000 / 0.026 / 0.157 |
| groot_spatial | 50 | 12296 | 0.888 | 0.440 | 0.270 | 0.149 | 0.149 / 0.346 / 0.288 | 0.002 / 0.189 / 0.540 |
| groot_spatial | 500 | 11047 | 0.966 | 0.294 | 0.248 | 0.035 | 0.109 / 0.339 / 0.189 | 0.001 / 0.025 / 0.376 |
| groot_l10 | 50 | 38173 | 0.552 | 0.655 | 0.182 | 0.399 | 0.318 / 0.503 / 0.452 | 0.001 / 0.045 / 0.160 |
| groot_l10 | 500 | 33390 | 0.706 | 0.484 | 0.173 | 0.253 | 0.285 / 0.485 / 0.490 | 0.000 / 0.013 / 0.135 |


### Paired action-error intervals

1000 whole-episode bootstrap draws, seed 4817. Full table: `paired_error_ci.csv`. These intervals quantify a proxy, not SR.
