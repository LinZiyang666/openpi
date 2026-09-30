# E4 profile analysis — allocation, budget, and evaluation freeze

## Recommendation

**Freeze 28 eval500 arms: SF1 ×8, UF1 ×8, SW ×4, CU(.30) ×4, CT(.30) ×4. Use SF cap E=1.**
Drop SF2 from this evaluation; defer SF+SW ×4 until the composed variant passes its own non-test profile gates.
Do not build SA now. CT remains a preregistered allocation ablation, not an established improvement over CU.
This preserves SELECTION §8's first 20 arms; none of their completed profiles triggers withdrawal.

Sources: `SELECTION.md` §§4–5/8; all four coder hand-backs; C4's closed-loop `profile_report.json` and offline
`{follow_all,camera_all,value_all,clock_all}` products; raw accepted journals, summaries and server decisions;
`/tmp/r7_C2/gpu_parity.json`. My reproducible audit is `audit_profile.py`, with results in `profile_audit.json`.
I independently reconciled **44 arms ×20 = 880 arm-episodes / 37,999 decisions**, N/V/M, success totals,
contiguous accepted steps and paired A identities; no discrepancy. Repeated arms are not independent success samples.

## 1. Proposal decisions and preregistered gates

**E4 proposal 1, common-price stage-mode allocator: keep the cost-control architecture; change the claim to match
what was built.** CU/CT test one lever at rho=.30. CT's reciprocal event/deviation occupancy weights are a heuristic;
they do not estimate marginal SR per IR or solve E4's value-based knapsack. SF/SW provide candidate cheap modes,
but a jointly calibrated mode allocator was not built. Keep CU as the fallback and CT as the controlled test.
My original kill criterion needs held-out, matched-cost success evidence; 20 profiles do not supply that evidence.

| Built family | Decision | Eligibility / exposure in closed loop | Realized owner IR saving versus same-pair A |
|---|---|---|---|
| SF1 ×8 | Keep; selected cap | Grants on 19.09–48.42% of vision anchors | .006707–.015438; 8/8 pass .005 |
| SF2 ×8 | Pass gates; drop as alternate cap | Grants on 14.23–38.15% | .009449–.021384; 8/8 pass |
| UF1 ×8 | Keep required control | Grants on 52.47–82.74% | .015382–.022541; 8/8 pass |
| SW ×4 | Keep | Actual wrist looks 22.95–40.57% of vision anchors | .011302–.020393; 4/4 pass |
| SF1+SW ×4 | Defer, not a measured failure | No composed closed-loop profile among these 44 arms | Unmeasured; component savings cannot be added |
| CU / CT | Keep all eight at fixed rho=.30 | Budget and parity gates below | A-saving gate applies to cheap modes, not added policy calls |

Eligibility here is anchor grant share, before a subsequent valve can abort; all follow variants clear ≥5%.
SW's realized wrist-look share establishes nontrivial use; it is not an extension grant rate.
Savings above subtract aggregate cost/decision ratios on identical episode sets. Equal-episode paired savings also
clear .005: minima are **.007947 SF1, .011463 SF2, .017252 UF1, .012486 SW**. Gate decisions do not depend on that choice.
All A/SF/UF/SW profile arms have **zero policy calls**.

Parity: disabled SF/UF match A on **10,569 decisions**, zero failures (C1/C4); CU/tilt-off matches R6 on
**26,992 decision pairs**, zero differences (C3). Disabled SW matches its base on **5,173 decisions** (C2).
The coordinator's actual GPU artifact has **12/12** equal wrist keys, full paths, completed policy inputs and K10
outputs, with maximum action difference **0**. Its PASS supersedes the older C2 hand-back's “pending GPU” status.
No SELECTION §4 implementation-parity kill is indicated. Enabled SF's early valve exception is discussed below.

### CU/CT budget, placement, and success evidence

| Sparse cell | CU aggregate IR | CT aggregate IR | CT−CU IR | CU/CT successes out of 20 | CT nominal p=1 / fresh anchors |
|---|---:|---:|---:|---:|---:|
| pi05 L10-50 | .298992 | .318419 | +.019427 | 18 / 19 | 121 / 584 |
| pi05 Spatial-50 | .301804 | .292655 | −.009149 | 18 / 16 | 55 / 274 |
| GR00T L10-50 | .307100 | .296609 | −.010491 | 15 / 15 | 112 / 651 |
| GR00T Spatial-50 | .304635 | .316186 | +.011551 | 17 / 18 | 58 / 247 |

**8/8 pass [.27,.33]**, both for aggregate IR and the calibration's equal-episode mean IR (range .290020–.317832).
Across the four cells CU and CT each succeed **68/80**, with **7 CT-only wins / 7 CU-only wins**; descriptive only.
On pi05 L10-50, +1 success accompanies +.019427 IR, already outside §5's .015 matched-cost allowance for that cell.
Do not call that a placement gain. Keep the frozen test; enforce the stated cost condition when interpreting eval500.
Across cells the simple mean IR difference is +.002834, but pooling must not conceal the per-cell mismatch.

The tilt is operational: my audit checks its weight formula, clipping, fresh-entry latch and actual coin/verdict
on **1,756 CT fresh anchors** with zero violations. **346/1,756 = 19.70%** have nominal p=1; every one of **81**
high-deviation entries has nominal p=1, although cooldown suppresses some actual calls. There are **72 calls /81
entries**; event-only calls are **520/817**, versus **316/858** on interior/unknown anchors. These are exposure
summaries, not treatment effects. Actual p=0/1 overrides must be excluded from unsupported local causal contrasts.

C4's stage-value tool adds no convincing allocation rule: **235/236** reported natural-dose first-entry intervals
include zero; eight additional comparisons lack both branches. The sole exclusion is an **unknown** stage with
**49 entries /4 CALLs / CALL ESS=4**. This supports retaining CT as an ablation and CU as default, not adopting
the heuristic as an empirically optimal budget allocation. It also agrees with E4's earlier eight-cell null replay.

**E4 proposal 2, descend from pure using stage feedback: defer.** No adaptive descent or contemporary pure-policy
reference was built/profiled; A and CU are not pure references. These profiles validate plumbing/cost at one knob
setting, not SR-constrained convergence. C3 calibration uses ten non-test recordings per cell; the additional 20
profiles are a separate validation set, not evidence that ten recordings identify stage-specific success curves.
The current solver also lacks E4's required pure maximum: modeled ceilings are **.440704–.475643**, below L10's .5;
floors are **.119061–.153875** because stall/LOOK obligations remain. Add an explicit pure endpoint before advertising
this as E4's full cheap-to-pure knob. This is a proposal/interface gap, not a failure of the selected rho=.30 experiment.

## 2. Frozen evaluation choice and what these profiles establish

Choose **SF1, E=1**, for all cells. SF2 buys only **.002741–.005947** further IR reduction (mean **.004446**) over SF1.
Pooled descriptive successes are **A 135/160; SF1 130/160; SF2 125/160; UF1 126/160**. These do not establish an SR
difference. The engineering choice favors the shorter extension, existing SF1 freeze and matching UF1 control.
SF1 already clears every cost gate. Mean total owner cost per attempt falls only **2.831875→2.771475** from SF1 to SF2
(2.13%), smaller than the relative IR improvement: longer trajectories can dilute per-decision IR.

Keep **SW ×4** despite mixed descriptive SR (67/80 versus paired A 68/80): parity and cost gates pass in every cell.
Offline camera error remains a warning, not an SR kill: pi05 L10-500 B-val wrist-minus-full normalized head RMS is
**+.02377 [.00955,.04335]**. Eval500 should decide the preregistered SW success criterion, not this proxy.
Keep **UF1 ×8** even though its profile SR varies: without it, the stage/valve signal cannot be distinguished from
the look-less lever. Keep both **CU ×4 and CT ×4** to preserve §5's planned allocation comparison.

**Defer SF+SW ×4** from the presently frozen list: it is implemented, but has zero composed closed-loop profile arms.
Its wrist metric changes neighbours, stage support and valve decisions, so the two passing component gates do not
prove a joint .005 saving or correct runtime interaction. A later addition requires its own non-test eligibility/cost
and composition checks, frozen before that variant's test episodes. This is a conservative recommendation beyond
the original component-only profile matrix, not a claim that SELECTION explicitly scheduled a composed profile.
**No SA build now:** calls change occupancy, stall history and camera completion; inherited CU/CT lambda cannot price
the changed cadence. A new budget replay/solve, composition telemetry audit and profile are required, while CT has no
demonstrated marginal-value advantage. Finish the 28-arm comparison before investing in that extra controller.

Twenty episodes per arm establish observed invocation/cost behavior and substantial eligibility on these starts.
They cannot certify 1–1.5 pp noninferiority, establish stage call value, prove generalization or measure convergence.
One success changes a cell's SR by **5 pp**; even 20/20 successes permit a **13.9%** one-sided 95% binomial failure bound.
All SR differences above are descriptive. There are only two initial states per task; larger anchor counts do not
create independent terminal-success samples. Keep SELECTION §5's eval500 tests and thresholds unchanged.

## 3. Bugs, accounting cautions, and verification

- **No confirmed count/coin/latch bug.** Raw accepted N/V/M and journals reconcile in all 44 arms. CT's saturation is
  the specified clipped reciprocal weighting, not evidence of a stuck latch; count fresh anchors, not carried tails.
- **Two IR estimands:** C4 `reports.IR` is ratio of aggregate cost to aggregate decisions; `paired.delta_IR` is mean
  of paired episode ratios. For pi05 L10-50 CT−CU these are **+.019427** and **+.015607**, respectively. Both are valid,
  but their labels/intervals must not be interchanged. C3 calibrated the second convention; freeze aggregate owner IR
  as the R6-comparable frontier/gate metric and report the episode mean separately. Both currently pass the .30 gate.
- **Two price tables:** stock `summary.cost_ledger` uses the eager table, not owner pricing. For SW pi05 L10-50 its
  IR is **.055572**, versus correct owner **.059502**; CT there is **.316031** versus **.318419**. C4's owner recomputation
  is consistent, so this is a reporting hazard, not missing GPU work. Wrist prices remain proportional-latency assumptions.
- **Early valve behavior:** SF1 has **13** valve LOOKs, **8 at age 1**; SF2 has **20**, also **8 at age 1**. Thus SF can
  interrupt the original commitment after five controls before executing any extra block. C1 documented this conflict;
  lever-off parity still passes, but “identical to A whenever no extension executes” is false. Freeze/document this
  behavior; do not silently interpret SF−UF as changing only extension permission or silently patch it after profiling.
- **Missing physical-control denominator:** every profile has `IR_active_controls=null`; stock-client logs do not
  certify partial terminal chunks or actual applied-control parity. Report requested-five-control IR, not measured
  active-control IR. SW has zero MISS/completion events in these profiles, so they do not validate live wrist+calls cost.

Reproduction, repository root (CPU only; outputs confined to E4):
```bash
taskset -c 28-29,72-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r07/ideation/E4_allocation/audit_profile.py > /tmp/r7_E4_allocation/profile_audit.log
```
