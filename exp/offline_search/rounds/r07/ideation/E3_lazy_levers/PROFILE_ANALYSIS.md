# E3 PROFILE analysis — freeze the short cap; keep wrist separate

**Recommendation:** freeze **SF1 ×8, UF1 ×8, SW ×4, CU ×4, CT ×4 = 28 eval500 arms**. Select **E=1**, one extra five-control block, for both policies. Drop SF2 from full evaluation as a cap-selection decision, not a failed preregistered gate. Hold SF1+SW ×4 outside this freeze until its own non-test composition profile; defer SA. All 36 implemented non-reference PROFILE arms pass their applicable §4 numeric gates.

Evidence: [SELECTION §§4–8](../../SELECTION.md), all four coder handbacks, [44-arm report](../../profile_results/closed_loop/profile_report.json), the four offline profile products, `/tmp/r7_C2/gpu_parity.json`, and an independent accepted-log audit saved as `profile_audit.json`. The audit covers **880 accepted episodes / 37,999 decisions**, with twenty matched task/init pairs per within-cell contrast. No test episode or new rollout was used here.

## 1. Follow proposal: keep SF1 and UF1; drop the longer cap

All IRs in the tables are owner-priced total accepted work / accepted five-control requests. “Grant” is structural/stage admission at a vision anchor, before subsequent valve checks; it is not a success probability. Success counts are **descriptive, out of twenty**.

| Cell | A IR | SF1 IR | SF2 IR | UF1 IR | Grant % SF1 / SF2 / UF1 | Success A / SF1 / SF2 / UF1 |
|---|---:|---:|---:|---:|---:|---:|
| π0.5 L10-50 | .076294 | .065437 | .061404 | .055131 | 35.3 / 26.9 / 78.8 | 15 / 14 / 11 / 17 |
| π0.5 L10-500 | .076683 | .067876 | .064221 | .055988 | 26.6 / 19.0 / 73.9 | 20 / 20 / 18 / 13 |
| π0.5 Spatial-50 | .077279 | .067236 | .061805 | .060855 | 30.4 / 25.8 / 52.5 | 15 / 13 / 16 / 14 |
| π0.5 Spatial-500 | .077813 | .062376 | .056429 | .055273 | 48.4 / 38.2 / 81.0 | 18 / 18 / 18 / 18 |
| GR00T L10-50 | .074268 | .064866 | .060382 | .055023 | 31.0 / 23.6 / 71.2 | 14 / 14 / 13 / 14 |
| GR00T L10-500 | .074669 | .067962 | .065220 | .052920 | 19.1 / 14.2 / 82.7 | 18 / 17 / 16 / 17 |
| GR00T Spatial-50 | .075189 | .064162 | .058652 | .059806 | 36.4 / 28.5 / 53.0 | 17 / 16 / 15 / 15 |
| GR00T Spatial-500 | .075095 | .065359 | .061590 | .054101 | 32.7 / 22.4 / 77.0 | 18 / 18 / 18 / 18 |

**§4 gates:** SF1 eligibility is **19.1–48.4%**, SF2 **14.2–38.2%**, UF1 **52.5–82.7%**: all comfortably above 5%. Realized saving versus same-pair A is **.006707–.015438**, **.009449–.021384**, and **.015382–.022541**, respectively: all eight cells of each variant exceed .005. Equal-episode paired IR differences also pass: SF1 saves **.007947–.016951**, SF2 **.011463–.023638**, UF1 **.017252–.022889**. Thus no gate conclusion depends on choosing the pooled-request or equal-episode estimand.

**Parity:** C1/C4 verified disabled SF/UF identity on 10,569 archived decisions and 6,612 enabled structural/stage-rejection comparisons. My new-run audit reconstructs **12,520 served blind prefixes** from anchor chunks/successor heads and the original float32 weights: **maximum difference 0**, no changed weights, no served missing successor. This checks server-served prefixes, not physical per-control application, which these runs did not record.

**Cap choice:** SF2 buys only **.002742–.005947** additional IR saving over SF1, averaging **.004446** across cells; five of eight incremental savings are below .005. That last comparison is descriptive, **not an extra preregistered gate**. SF1 has **130/160** successes versus SF2 **125/160** (directly paired wins/losses **13/8**); SF2 is lower in five cells, higher in one, tied in two. π0.5 L10-50 is the strongest caution: **14→11/20**, with only **.004033** additional saving and valve looks **5→12**. Prefer the smaller intervention and a uniform cap; do not choose per-cell caps from these small SR counts.

**Stage signal remains unproven:** A has **135/160**, SF1 **130/160**, UF1 **126/160**. SF1 versus A has **3 improvements / 8 regressions**; SF1 versus UF1 has **16/12**. The apparent SF advantage over UF is strongly cell dependent: π0.5 L10-500 **20 versus 13/20**, but L10-50 **14 versus 17/20**. Keep UF1 ×8 as the required control, including its unfavorable dense L10 cell. Dropping that control would prejudge the stage-plus-valve question. My prior “convincing matched-call SR loss” kill criterion is not established here; the profile certainly does not establish SR preservation either.

**The valve is active but sparse after the stage screen:** SF1 grants **927/3,021** anchors and serves **914** extra requests; its valve fires **13/1,846 checks = 0.70%**. SF2 grants **686/2,957**, serves **1,334** extra requests, and fires **20/2,032 = 0.98%**. These rates condition on eligible commitments; they are not comparable to my original unrestricted twenty-control window rates. Recomputed residuals agree within **3.86e-8**, with zero verdict mismatches. The profile cannot isolate whether the stage gate or valve causes any outcome difference; UF removes both.

**Model scope:** all **495 π0.5 SF1** extensions use successor heads; all **419 GR00T SF1** extensions use controls 10:15 of the existing prediction. GR00T SF2 additionally serves **310** successor-head requests. Consequently SF1 validates successor execution on π0.5 and native-tail completion on GR00T; it does not validate exact sixteen-control execution or long GR00T successor following.

The offline `follow_all` screen gives SF1 eligibility **21.72–48.15%** on the older ten-episode B-val recordings, consistent with there being useful support but not identical to the renewed controller's trajectories. `clock_all` has a displacement alert in **25/43 failures** and **12/197 successes**; absolute-deviation crossings occur in **43/43 failures but also 184/197 successes**. Do **not** add the current absolute-p75 event as a blanket new safety veto on that evidence. Those are episode-any-crossing statistics on earlier paths, not failure-onset labels or causal prevention results. Keep the implemented displacement radius and stage logic frozen for evaluation.

## 2. Look-half proposal: keep SW ×4, without claiming camera sufficiency

| π0.5 cell | Actual wrist looks / all looks | Wrist share | SW IR | Saving vs A | Success A / SW |
|---|---:|---:|---:|---:|---:|
| L10-50 | 217 / 624 | 34.8% | .059502 | .016792 | 15 / 16 |
| L10-500 | 123 / 536 | 22.9% | .065381 | .011302 | 20 / 19 |
| Spatial-50 | 75 / 274 | 27.4% | .063681 | .013597 | 15 / 15 |
| Spatial-500 | 99 / 244 | 40.6% | .057421 | .020393 | 18 / 17 |

**Keep:** actual cheap-look coverage is **22.9–40.6%**, saving **.011302–.020393** owner IR in all four cells, clearing the applicable eligibility/saving gates. There are **514** actual wrist looks, **zero MISSes and zero completion calls**, no non-full episode starts, and no camera/owner-cost audit mismatch. These are real pre-encoder routing counts; the wrist price remains the declared R4 proportional-latency assumption, not a new latency measurement.

**SW parity gate passes:** the saved GPU artifact has **12/12** wrist-key, full-path, completed-policy-input and K10-output parity passes, with **maximum policy difference 0**. C2's handback still says GPU parity is pending; that sentence is superseded by the coordinator artifact. Disabled SW identity covers **5,173** archived decisions; enabled wrist retrieval matches its independent metric on **1,235** archived wrist anchors.

**Own criteria:** there is no measured parity failure, no collapse of savings through completions, and no supportable SR-loss conclusion: SW has **67/80** versus A **68/80**, paired improvements/regressions **3/4**. The offline two-fold camera screen warrants attention to L10-500: wrist-minus-full B-val head error **+.02377 [+.00955,+.04335]**; the other three π0.5 B-val intervals contain zero. This is a retrieval surrogate, so it does not independently kill SW or certify the other cells. PCA remains frozen across metric folds.

Evaluate the delivered **π0.5 wrist stage gate**, not my broader hypothetical wrist-versus-third-person selector or deletion-instability admission rule: those were not implemented. No GR00T SW or third-person serving/cost claim is justified by stored-key deletion.

## 3. Frozen arm list and compositions

| Candidate | Recommendation for this freeze | Reason |
|---|---|---|
| SF1 ×8 | **Keep; E=1 everywhere** | All gates pass; smaller extension, strongest preserved comparator design. |
| SF2 ×8 | **Drop from eval500** | Cap downselection above; not a §4 implementation/eligibility/cost failure. |
| UF1 ×8 | **Keep as control** | All gates pass; necessary to test whether stage/valve signals add value. |
| SW ×4 | **Keep** | Actual omission, parity, eligibility and saving gates pass. |
| SF1+SW ×4 | **Hold outside this freeze** | Implemented/CPU-checked but no joint closed-loop PROFILE result among the 44 arms. |
| CU ×4 | **Keep** | Required uniform/stall comparator; all four budget gates pass. |
| CT ×4 | **Keep as the preregistered test** | Budget gates pass; no demonstrated advantage over CU. |
| SA ×4 | **Do not build for this freeze** | New combined cadence/call solve and, with wrist, mixed-camera MISS path remain unprofiled. |

The immediate list is **28 arms / 14,000 test episodes**. SF1/UF1/CU preserve SELECTION's Freeze 1 commitments; none is withdrawn for a late §4 failure. If the coordinator wants the four SF1+SW arms in this round, first run their own same-pair non-test profile and freeze that addition **before those variants' test results are seen**; do not carry over separate component gates as if the composition had passed. This would produce a 32-arm list, but is not my unconditional recommendation on the currently supplied evidence.

The composition concern is concrete: a wrist query changes the sixteen selected trajectories, while SF's displacement radius was calibrated with the frozen **full-camera** retriever. The longer cadence also changes which camera plans are reached. Existing CPU composition tests protect interfaces, not the joint radius distribution, eligibility, IR or SR. Joint profiling may support the current shared radius; it cannot be assumed from SW and SF1 separately.

If SA is revisited, start with **CU + SF1**, then consider wrist. CT offers no current reason to be the default call allocator. Re-solve λ for the composed cadence, maintain non-extendable policy chunks, and profile actual wrist-origin MISS completion before adding that branch. The existing pure-cache SW profile never exercised a live MISS, despite the successful twelve-observation GPU completion proof.

| Sparse cell | CU IR | CT IR | CT−CU IR | Success CU / CT |
|---|---:|---:|---:|---:|
| π0.5 L10-50 | .298992 | .318419 | +.019426 | 18 / 19 |
| π0.5 Spatial-50 | .301804 | .292655 | −.009149 | 18 / 16 |
| GR00T L10-50 | .307100 | .296609 | −.010491 | 15 / 15 |
| GR00T Spatial-50 | .304635 | .316186 | +.011551 | 17 / 18 |

All eight costs are in **[.27,.33]**; equal-episode costs also pass (**CU .299440–.310042; CT .290020–.317832**). CU and CT each achieve **68/80**, directly paired **7 wins / 7 losses**. The offline call-value audit's **235/236** reported natural first-entry intervals include zero; the sole exception has CALL ESS **4** in an unknown stage. This supports retaining CT as a test, not composing it as a presumed improvement.

## 4. Bugs/accounting checks and limits

1. **No detected serving/accounting bug in the audited scope:** all **44** summaries' N/V/M match accepted logs and C4; all checked successor/tail prefixes, weights and valve decisions agree. No server-side action error explains the SF2 outcome pattern.
2. **Different cost tables are easy to mix up.** For SW L10-50, `summary.cost_ledger.ir_per_request=.055572`, while owner IR is **.059502**; SF2 L10-50 is **.057348** versus owner **.061404**. Summary uses the eager ms table; C4 uses the selected `.152/.148`, `.848/.852`, wrist `.055198` prices. This is a labeled basis difference, not missing work. Use C4 owner prices for gates/frontiers, and do not interpret concurrent wall-time ratios as owner IR.
3. **Cost weighting differs too.** The report's top-level IR is request-weighted; its paired contrast and C3's budget fit are equal-episode. Example SF1 L10-50: aggregate saving **.010857**, paired mean saving **.012304**. Both pass; they are not interchangeable estimates. CT−CU L10-50 is **.019426** aggregate and **.015607** equal-episode, both above .015; the mean aggregate gap across four cells is only **.002834**. Specify whether §5's cost-match condition is pooled or per-cell before evaluation, and do not describe that particular cell's profile contrast as cost matched. Do not tune on test outcomes to repair a mismatch.
4. **Early-valve parity scope must stay explicit.** SF1 and SF2 each have **eight** age-one valve looks, before any extra block is served. This is consistent with “check every blind decision of an eligible commitment” and passes §4's **lever-off** identity gate. It is incompatible with a broader claim that enabled SF always equals A whenever no extension is ultimately served. Do not silently disable those looks after freeze.
5. **No control-rate conclusions from these new runs.** Actual-control IR is correctly null for all 44 arms; stock-client nominal `controls=5N` includes possible terminal partial requests. New per-control drift, transient excursions, physical action application, or failure-onset timing cannot be recovered from these server logs. Earlier P3 control telemetry informs limitations, not the new variants' physical response curves.

Twenty episodes per arm can expose integration defects, establish actual grants/camera omission, check realized owner-IR gates on these starts, and motivate a conservative cap choice. One changed outcome is **5 percentage points per cell**. These counts cannot establish the preregistered −1.0 pp SF or −1.5 pp SW noninferiority margins, a stage-signal advantage, a frontier movement, or a safety guarantee. Dense banks also include acquisition on B-pool states, so this profile is not independent library generalization. Keep all SR comparisons above descriptive; the frozen eval500 pairs answer those questions.

Reproduction of the independent audit, from the repository root (outputs only this directory):

```bash
taskset -c 26-27,70-71 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r07/ideation/E3_lazy_levers/analyze_profile.py
```
