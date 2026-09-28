# R5-A: finish a useful plan, then decide when to look

**Recommendation:** first test a ten-control commit after a policy MISS, without letting the rejected cache proposal veto the policy's remaining five controls. Keep the existing visual guard at the next anchor. Next, establish GR00T mixed ten-control cycles. Treat selective inspection without action replacement as the adaptive controller, and extension beyond a completed chunk as a separate, lower-priority experiment.

This is an ideation and diagnostic hand-back, not an implementation or closed-loop result for these new controllers. `ARM_SPECS.json` contains concrete review recipes and explicitly marks missing classes/flags. Its 58 entries are a catalog of controls and variants, **not a request to launch 58 arms**. The staged pilots below are the proposed queue.

## 1. Measured facts and their limits

### Provenance and accounting

All new work is in `exp/offline_search/rounds/r05/ideation_A/`. No simulator, server, GPU, external model, subagent, or git command was used. The hot store was absent; reads used `/home/weiland/trace_runs/offline_search_store`. The latest completion inventory and UTC snapshot are in `review_tables.json`. Spatial L10 was complete; no completed K10 policy-tail arm or GR00T mixed arm appeared in that snapshot. During this analysis K10's opt-in code became present in the shared `closed_loop/plugin.py`; its current implementation remains π0.5-only.

Reproduction scripts, all relative to this directory:

| Script | Inputs and output |
|---|---|
| `inventory.py` | Read-only R4 arm definitions, journals, startup/schema samples; `inventory.json` |
| `analyze_logs.py` | Six pure-inference arms, eight K7 arms, three R3 mixed references; `log_summary.json`, `log_<arm>.json`, `episodes_<arm>.json`, `horizon_pairs.json` |
| `analyze_chunks.py` | Four offline inference cells, each evaluated under its own 50/500-library normalization; `chunks_summary.json`, eight `chunks_*.json` |
| `analyze_cadence.py` | Eight independent library-only displacement fits; three actual K7 tail trajectories; `dynamics_fits.json`, `cadence_audit.json` |
| `make_review_tables.py` | Derived tables, fitted-artifact byte counts, completion snapshot; `review_tables.json` |
| `make_arm_specs.py` | Review-only, expanded method kwargs and plugin flags; `ARM_SPECS.json` |

Run commands and scope of validation are in `REPRODUCE.md`. Decision statistics retain only accepted, non-error `done`/`failed` journal attempts, match server attempts where recorded, deduplicate `(uid, step)`, and assert consecutive request indices and no conflicting duplicates. The 500-init comparisons share task/init identities. Process seeds and concurrent request order differ, so pairing does not make policy noise identical.

The binding owner cost is

`IR = (0.152 V + 0.848 M) / N5`,

where `N5` is five-control slots, `V` real vision evaluations, and `M` policy calls. For fixed L controls/request, `N5 = requests * L / 5`. Every MISS costs full inference, including historical K2 controls. Search is excluded from this primary IR. A useful cycle formula is `IR = (0.152 + 0.848 q)/b`, for b five-control blocks per anchor and MISS fraction q among anchors. Extra inspections and interrupted cycles must be counted explicitly. The coefficients are the owner's π0.5 graph basis; applying them to GR00T below is a common comparison convention, **not a measured GR00T latency decomposition**. Actual GR00T stage shares remain to be measured if model-specific time savings are wanted.

No new fit uses query outcomes or larger-library data for a 50-library deployment. Evaluating the same offline query trace under two library normalizations is not two independent experiments. A future 50-library fit using the 500 library must be labeled **borrowed big-library information**; none is proposed here.

### Whole chunks help l10; a spatial SR gain is not established

Source: `analyze_logs.py`, `log_summary.json`, `horizon_pairs.json`; run root `/home/weiland/trace_runs/os_closed_loop/r04_cost/runs/`.

| Suite | L5, full denoising, seed base 1001 | L5, K2, seed base 1101 | L10, full denoising, seed base 3001 |
|---|---:|---:|---:|
| l10 | 424/500 = .848; 29,714 calls; IR 1 | 425/500 = .850; 29,380 calls; IR 1 | 452/500 = .904; 14,073 calls; IR .5 |
| spatial | 496/500 = .992; 10,679 calls; IR 1 | 495/500 = .990; 10,665 calls; IR 1 | 493/500 = .986; 5,499 calls; IR .5 |

Names are `r4f_p_<l10|sp>_inf_s1001`, `r4b2_p_<l10|sp>_inf_k2_s1101`, and `r4f_p_<l10|sp>_inf_k10_L10`. The earlier independent step-diagnostic full-policy reference is .844/.986 for l10/spatial (`exp/step_diag/analysis/step_vs_warmstart.md`, §6.15).

Against full-denoising L5, l10 has **52 failure→success and 24 success→failure pairs**, net +28/500; exact two-sided discordance test p=.001761872. Against K2 L5 it is 52/25, p=.002799054. Spatial is 4/7 against full L5, p=.548828125, and 5/7 against K2, p=.774414063. These are descriptive tests of these runs, not randomized identification of a jitter mechanism. The result has no library dependency: it applies equally to the 50- and 500-library comparisons because these arms always call the policy.

Per-task l10 successes, each out of 50:

| ID and task | L5 full | L5 K2 | L10 full | L5-full→L10 net |
|---|---:|---:|---:|---:|
| 0 soup + tomato sauce into basket | 49 | 49 | 49 | 0 |
| 1 cream cheese + butter into basket | 49 | 49 | 50 | +1 |
| 2 stove on + moka pot | 38 | 37 | 41 | +3 |
| 3 bowl into bottom drawer, close drawer | 38 | 44 | 46 | +8 |
| 4 two mugs onto two plates | 47 | 49 | 47 | 0 |
| 5 book into caddy | 47 | 46 | 47 | 0 |
| 6 mug onto plate + pudding beside plate | 44 | 42 | 48 | +4 |
| 7 soup + cream cheese into basket | 47 | 48 | 49 | +2 |
| 8 both moka pots onto stove | 23 | 18 | 28 | +5 |
| 9 mug into microwave, close microwave | 42 | 43 | 47 | +5 |

Task 8 remains the largest unresolved failure concentration: 22/48 L10 failures. Gains occur in drawer/microwave and multi-object tasks, but assigning them to particular contact or grasp failures would require evidence not present in these logs. The task-3 contrast also illustrates seed/run variability: its L5 controls differ by six successes.

Spatial successes by task ID 0…9 are L5 full `[49,50,50,49,49,50,50,49,50,50]` and L10 `[50,50,50,50,48,48,50,50,48,49]`. There is no broad task-level improvement hidden by the aggregate.

### What can be said about failures, stalls, and gripper events

The R4 inference decision JSONL saved only `a_exec[:5,:7]`, even for L10. They have no normalized robot states or full policy chunks, and no input NPZ was saved. Client `per_step.jsonl` gives request/control indices, not physical states or object contacts. Therefore **R4 L5-vs-L10 overlap, L10 action-9→next-action-0 jitter, physical stalls, failed grasps, dropped objects, and missed placements are unknown**. Reconstructing the rejected B0 retrieval proposal would not recover the policy action.

Measured failure descriptions are narrower:

* Every failed l10 episode consumed the nominal 520-control budget: 76 failures at L5 full and 48 at L10. Spatial failures likewise reach 220 controls: four versus seven. There were no accepted error episodes in this analysis. These are timeout failures, not semantic failure labels.
* A command-repeat proxy uses RMS difference between consecutive saved five-step heads, normalized by the current library's six continuous-action standard deviations, with threshold .1. In l10 L5 full, failed episodes have mean longest repeat spell 4.618 requests versus 1.844 in successes; 20/76 failures have a spell of at least five requests. L10 failures have 2.271 observed-head requests on average, but those heads are ten controls apart and half the actions are missing: this is **not** evidence that physical stalls were halved. At L5 K2 the failed-episode mean is 12.507, further warning against treating this proxy as failure causation.
* All failed pure-inference episodes issue a close command in their observed heads. This does not establish a successful grasp. Spatial often succeeds without a subsequent release; therefore “never released” is not a generic failure label.
* Within the saved heads, gripper-sign flip rates are l10 1,956/118,856 = 1.6457% at L5 full and 747/56,292 = 1.3270% at L10. Spatial is 652/42,716 = 1.5264% versus 305/21,996 = 1.3866%. These are observed-edge rates, not full L10 gripper-event counts.
* At L5 full, the mean continuous-action jump across actual replanning boundaries is **.270491** normalized RMS in l10, versus **.104139** within a head (2.597×); spatial is **.266873** versus **.142036** (1.879×). At L10, only the within-head quantities are observable: .102349/.145101. A boundary jump is a change of command, not a demonstrated undesirable robot acceleration.
* Among the 400 l10 task/inits that succeed in both full-policy arms, L10 changes nominal executed controls by mean **−2.05**, median zero. Spatial's 489 common successes change by **+2.556**, median zero. Whole-chunk execution mainly changes which l10 episodes succeed; it does not show a large shortening of already successful trajectories. Counts round the final request to L, so differences of a few controls are not precise completion-time measurements.

### Independent full-chunk traces support a plan-interruption hypothesis

`analyze_chunks.py` uses the Sept 23/24 offline **full-inference** trajectories, not R4 L10 trajectories. Compare `a_inf[t,5:10,:6]` with `a_inf[t+1,0:5,:6]` on consecutive same-episode decisions. Actions 0…6 and state 0…7 alone are valid; padding is never included. Continuous values below are RMS in each deployed library's own action-sigma units.

| Model / suite | Mean tail-vs-new-head RMS, 50 / 500 | Natural action4→5 jump, 50 / 500 | Replanned action4→next0 jump, 50 / 500 | Tail/next-head gripper-element disagreement | Planned tail switch absent from next head |
|---|---:|---:|---:|---:|---:|
| π0.5 spatial | .2532 / .2617 | .1392 / .1435 | .2708 / .2794 | 2.19% | 143/773 = 18.50% |
| π0.5 l10 | .2746 / .2841 | .1033 / .1070 | .2663 / .2757 | 4.83% | 1,028/2,523 = 40.75% |
| GR00T spatial | .2098 / .2161 | .1482 / .1530 | .2475 / .2551 | 2.09% | 106/747 = 14.19% |
| GR00T l10 | .2177 / .2311 | .1056 / .1121 | .2281 / .2420 | 6.80% | 1,019/2,485 = 41.01% |

A “planned switch” means any tail action has a different gripper sign from the final executed head action. “Absent” means the next policy head never changes to that sign. It may be a sensible correction, a delayed event, or chatter; it is not an identified failed grasp. The event counts/sign disagreement do not depend on library scale. Per-task and success/failure tables are in each `chunks_*.json`.

The same offline π0.5 l10 trace has a low-proprioceptive-motion share of **37.29% in failed-episode transitions versus 9.31% in successes** under the 50-library motion p10; under the independently calibrated 500-library threshold it is **27.36% versus 5.00%**. Spatial is 18.27%/10.30% and 18.27%/9.60%. These associate low motion with failure exposure, including long timeout tails; they do not validate a motion-only MISS guard.

**Mechanism hypothesis:** on l10, repeated replanning sometimes abandons a near-future grasp/release or other short coordinated sequence before it is executed. A complete chunk preserves that sequence and removes one opportunity for an inconsistent new command. The larger l10 event-cancellation fraction and closed-loop SR gain support testing this explanation. Spatial also has replanning disagreement, yet no measured SR gain: disagreement magnitude by itself does not predict the benefit. The full-policy length intervention, not offline action error, is the primary evidence.

### The MISS path is where the next useful saving is concentrated

Actual K7 tail runs, from `analyze_logs.py`:

| Cell / library | SR | Five-control slots N | Vision V | MISS M | IR |
|---|---:|---:|---:|---:|---:|
| π0.5 l10 / 50 | .806 | 31,186 | 18,566 | 5,554 | .241513 |
| π0.5 l10 / 500 | .880 | 28,888 | 16,547 | 3,959 | .203281 |
| π0.5 spatial / 500 | .982 | 10,630 | 5,643 | 596 | .128236 |

Spatial/50 has no corresponding completed K7-tail result in the snapshot. Its measured stock mixed reference is .888 @ .266449 (`r03_mx/runs/r3mx_p_sp_g`). Do not present an inferred tail result for that missing cell.

Code evidence: `r04/k10_policy_tail/judge.py::policy_tail_step` temporarily supplies the rejected cache anchor to `super().blind_step`. `r04/k1_blind/judge.py::blind_step` rejects when `_noprog_span > 0`, even with `gates="budget_only"`. This is a veto on **whether to execute the policy's already-computed tail**, based on progress of the cache proposal. It is distinct from the valuable visual no-progress MISS guard, which remains in force at anchors.

| Cell / library | MISSes with another request | Those with positive cache no-progress span | Immediately followed by another MISS |
|---|---:|---:|---:|
| l10 / 50 | 5,371 | 4,741 = 88.27% | 2,870 |
| l10 / 500 | 3,826 | 3,606 = 94.25% | 1,175 |
| spatial / 500 | 316 | 260 = 82.28% | 109 |

At l10/50, 3,828/5,554 MISSes have primary reason no-progress; at 500, 3,456/3,959 do. The other primary reasons are terminal, overtime, and visually confirmed stuck. Reason codes are priority-ordered, so these are not counts of all simultaneously active flags.

If anchor MISS probability q remained unchanged and every anchor got two blocks, the observed q values .299149/.239258/.105618 imply IR **.202839/.177445/.120782**, respectively. These are **conditional cost calculations**, not simulated outcomes. Episode endings and interrupted cache tails change this arithmetic.

`analyze_logs.py` also supplies simple fixed-path replacement masks. Keeping K10's span veto gives .229988/.199469/.124882; waiving it gives .176014/.162473/.118158. **Do not use the latter as controller forecasts:** replacing a cache-head request can leave its old blind tail incorrectly attached to a new policy source. `analyze_cadence.py` finds 1,691/2,062/160 such orphan slots. Restoring mandatory vision there gives .184255/.173322/.120446 before any additional MISSes; making every restored look a MISS gives .230237/.233852/.133210. Even these endpoints only bound an accounting exercise with the other decisions frozen. They are not bounds on the changed closed-loop trajectory. The report's forecasts instead state their q/v assumptions.

### Adaptive timing: an executed-action monitor is cheap, but does not certify task progress

`analyze_cadence.py` fits, **separately in each deployed library**, an affine predictor

`Δrs_xyz = Bᵀ [1, rs[:8], mean(issued_actions[:5,:7])]`.

Only real consecutive library edges enter training. B is a 16×3 ridge solution; the intercept is unpenalized and ridge is `.01 * trace(XᵀX[1:,1:])/15`. Each episode is held out in turn. The residual is RMS xyz prediction error divided by the training-fold displacement standard deviation, floored at .005. The deployment threshold is the p99 of those LOEO residuals; the final B/scales use all of the deployed library. This is **T1, 208 bytes of float32 parameters per cell**, not an external learned model. The `.01` numerical ridge and 1% nominal alert budget are explicit design choices, not claimed SR-optimal hyperparameters.

| Model / suite | LOEO residual p99, 50 / 500 | Alert share on independent inference transitions, 50 / 500 |
|---|---:|---:|
| π0.5 spatial | .515818 / .519782 | .9419% / .9031% |
| π0.5 l10 | .991128 / 1.004716 | 1.2177% / 1.1001% |
| GR00T spatial | .433368 / .493964 | 2.3713% / 1.3748% |
| GR00T l10 | .895247 / 1.008576 | 2.0900% / .9592% |

On actual K7 l10 cache-tail transitions, the alert rates are **45/12,475 = .3607% at 50** and **72/12,141 = .5930% at 500**; on policy-head transitions, **163/5,371 = 3.0348%** and **112/3,826 = 2.9273%**. Spatial/500 is 26/4,816 = .5399% after cache tails and 0/316 after policy heads. Spatial/50 mixed-tail transfer is unmeasured. Predicted displacement must use the **actually issued head or tail**; using the rejected cache proposal after a MISS would repeat the K10 provenance problem.

This monitor observes arm tracking. It cannot observe an object slipping, an empty grasp with normal arm motion, drawer completion, or out-and-back visual change. It can request a new visual check; it cannot prove that another blind chunk is appropriate. Its LOEO percentile is not a distribution-free closed-loop safety guarantee.

For K7 anchors that actually served a tail and have a follow-up, an inspection rule “issued chunk changes gripper sign over actions 4…9 OR head displacement residual exceeds its own-library p99” fires **992/12,475 at l10/50**, **896/12,141 at l10/500**, **278/4,816 at spatial/500**. Vision alone would add **.004835/.004714/.003975 IR** on those fixed trajectories. These counts exclude policy tails and other counterfactual states; any resulting MISS costs another .848/N5.

### Beyond one chunk has coverage, but its hidden interventions matter

For every logged cache→tail pair, I reconstructed the original full chunk from all 16 logged rows/weights. Maximum absolute head/tail discrepancies are below **2.80e−7** in all three cells, validating the provenance used in this audit. After ten controls, a fixed-phase bridge must start at **`next(next(row))`**, not `next(row)`, which starts at control 5 and would repeat elapsed phase. Edges must be consecutive in the same library episode/task; terminal clamping is ineligible.

The measured bridge screen requires all 16 two-step successors and one further nonterminal successor, a continuous join jump no larger than the library's own natural action4→5 p95, no bridge gripper-sign change, per-control weighted gripper minority mass <.2, and an acceptable residual after the actually executed tail. It does not use the next vision query to decide eligibility.

| Cell / library | Observed completed cache chunks with follow-up | Eligible bridges | Eligible locations whose next actual decision was MISS |
|---|---:|---:|---:|
| l10 / 50 | 12,475 | 7,567 = 60.66% | 1,105 = 14.60% of eligible |
| l10 / 500 | 12,141 | 7,622 = 62.78% | 1,280 = 16.79% |
| spatial / 500 | 4,816 | 3,535 = 73.40% | 110 = 3.11% |

Join thresholds are .258374/.255830/.331556 sigma RMS, respectively. These are **available opportunities, not validated safe skips**. The high l10 MISS overlap is an argument for a small randomized pilot, not a license to bank those calls as savings. The 50-library case is especially suspect given the actual .700 SR of R4 phase-B2 versus .806 for its own-chunk tail.

In the raw libraries themselves, mean tail-versus-next-head mismatch is π0.5 l10 **.2527/.2852** at 50/500, spatial **.2506/.2636**; GR00T l10 **.2020/.2351**, spatial **.1893/.2364**. Thus the demonstrated next head is not the original stored plan's tail. Continuing stored successors changes synthesis/control, even before vision is removed.

### GR00T has sixteen planned controls, but currently executes five

`harness/dims.py`, store manifests, `exp/libero_groot/serve_groot_libero.py:72`, `policy_adapter.py::iter_step_actions`, and `closed_loop/ops/remote/run_arm.sh` agree: **H=16, default execute=5**. K=8 in GR00T LIBERO is denoising steps, not an eight-control execute setting. Current K1 `anchor_tail` already has horizon checks that admit offsets 5 and 10 for H16; the last remaining action at offset15 is insufficient for another five-control blind response.

The current K10 plugin rejects non-π0.5 policy-tail mode, consumes a single tail buffer, and forces a look immediately after it. K7 also explicitly rejects GR00T because stock terminal/gripper semantics differ. A GR00T plan must not simply remove these checks and silently call it equivalent to π0.5. GR00T's normalized closed sign is negative; π0.5's is positive.

GR00T's second original tail, `a_inf[t,10:15]`, disagrees with the head after two recorded replans by mean continuous RMS **.2918/.3005** spatial and **.3167/.3357** l10 at 50/500 normalization. Gripper-element disagreement is **3.10% spatial, 8.85% l10**, larger than the first-tail 2.09%/6.80%. This makes L15 worth isolating from L10; it does not establish that L15 is worse.

Historical R2 pure-cache AWM SR at 50/500 is GR00T spatial **.888/.966**, l10 **.552/.706**; policy SR is **.940/.870** in the offline trace cohort. The independent §6.15 policy cohort is .944/.880. In particular, policy calls can plausibly hurt an already excellent spatial/500 cache. Mixed GR00T SR and full-policy L10/L15 SR are still unknown.

### Library footprint, including both scales

Decimal bytes/MB. The full-chunk actions are already stored, so tails require **no additional library rows or action payload**. Working buffers are per connection, not library growth. Exact current K7 file sizes are from `make_review_tables.py`; other fit measurements are the audited fresh fits in `r04/k1_blind/HANDBACK.md`, “Measured fresh fits and deployed footprint.”

| Model / suite | Rows, 50 / 500 | Relevant fitted pickle bytes, 50 / 500 | Owner deployed pkl |
|---|---:|---:|---:|
| π0.5 l10, K7 | 2,640 / 29,472 | 32,704,527 / 142,348,562 | 1,103 MB |
| π0.5 spatial, K7-style guard | 1,018 / 10,909 | 26,076,273 **K1 guard proxy** / 66,500,552 actual K7 | 431 MB |
| GR00T l10, BlindAWM | 2,645 / 29,631 | 28,135,563 / 118,430,992 | 1,068 MB |
| GR00T spatial, BlindAWM | 1,063 / 11,751 | 22,841,960 / 58,604,408 | 429 MB |

The nominal π0.5 spatial “50” library has 49 episodes. π0.5 fitted sizes above are about 3.0%/12.9% of deployed l10 and 6.1%/15.4% spatial; GR00T is 2.6%/11.1% l10 and 5.3%/13.7% spatial. These compare representation files, not measured total server residency: the current plugin may also retain the original serving library. The spatial/50 new K7/Commit artifact has not been fitted here; its exact bytes are unknown, expected near the proxy plus class metadata.

Compact row representation is 632 B for the span guard and 586 B for BlindAWM, **plus** valid action payload 280 B/row π0.5 or 448 B/row GR00T, plus fixed PCA/metric/calibration storage. The measured pickles already include padded chunks and auxiliary arrays; do not add valid payload again to their totals. The two-camera float32 PCA basis/means alone account for 17,039,360 B. New controller constants and a 208-B monitor are negligible beside these artifacts. A bridge can reuse existing successor/state arrays; it does not multiply the library by execution horizon.

All four proposals are T1 overall because their base AWM/guard uses fitted library statistics; the plain tail scheduler itself is T0. None requires a model forward pass beyond the counted stage-1 or policy calls. K8/K9's measured search costs remain a separate ledger: roughly .02–.03 owner IR per CPU vision query versus .006–.009 for the graph retrieval prototype, before multiplying by vision frequency. No new speed measurement is claimed here.

## 2. Ranked proposals

The ranking is expected closed-loop value for the added or saved IR, not an ordering by offline action error. **All SR ranges below are hypotheses for pilot planning, not confidence intervals or measured results.** The horizon benefit already present in R4 cache tails is not counted again as a new synthesis gain.

### 1 — Complete the paid policy chunk (`CommitJudge`, C10)

**Pitch and mechanism.** A MISS is evidence to replace the cache proposal with the policy's plan. Give that plan its remaining five controls before the next ordinary look. The inherited cache no-progress veto currently prevents this at most eligible l10 MISSes. This intervention changes execution after a policy call, while keeping the no-progress guard that decides future calls at visual anchors.

**Algorithm and API.** Subclass current `PolicyTailJudge`; ordinary `fit`, `reset`, and `query` retain K7, its own-library AWM/V7 fit, visual stuck confirmation, and `noprog_span`. Change only `policy_tail_step(bq)` for `policy_tail_gate="lifecycle"`:

1. Require the same episode/task, previous committed source=policy MISS, previous request a real vision request, exactly five executed controls, finite normalized/raw state and actions, and an unconsumed H≥10 original policy response. Reject duplicates/partial execution/reset as K10 does.
2. Serve exactly original normalized policy actions 5…9, with H-row padding used only for the wire contract. Return their `BlindResult`; gate-provenance rows/weights may reference the preceding proposal but must never be labeled the action source.
3. Do not invoke the rejected cache anchor's progress, terminal, gripper, or trajectory gates for this one policy continuation. Do not clear progress memos, increase `noprog_n`, set a fictitious prior HIT, or release multiple chunks. Optional variant `monitor="loeo_xyz99"` requests vision if actual policy-head displacement residual exceeds the fitted p99.
4. After the one policy tail, require vision. Preserve the actual MISS then tail history. Once the original tail is exhausted, AWM's “fresh after MISS” continuity must not compare against a shifted/padded expired tail. The normal stale branch after a committed tail is appropriate; a future source-aware continuity term needs an explicit remaining-action cursor.

**Server changes.** The current π0.5 `--os-blind --os-policy-tail` buffer already preserves the original transformed `wire` response and enforces one-shot use; the lifecycle-only method variant needs no new serving path. Keep this exact wire behavior: applying a state-dependent output transform with the new state can change an action. The deployed `pi05_libero` configuration sets `extra_delta_transform=False` (`src/openpi/training/config.py`), so this is a general contract requirement, not a diagnosed explanation of the present LIBERO gain. Log source `policy_tail`, original source decision, executed count, and reason for tail denial. For full-horizon audits use existing `--os-log-r4 --os-log-inputs` on a small instrumented subset; NPZ `a_exec` and `wire_actions` retain full chunks.

If a midpoint look at control5 causes a MISS, start a new policy commit there: policy head over controls5…9, its own tail over10…14, then look at15. Do not finish the old cache tail or force the new plan to end on the old control10 boundary. A new source resets the plan cursor, not the actual observation/progress history.

**Concrete kwargs.** Use the K7 recipe:

```json
{"base_kwargs":{"lib":"current","fit_data":"same","kref":5,
  "serving":"anchor_tail","budget":1,"gates":"budget_only"},
 "progress_guard":"noprog_span","memo_reset_after_miss":false,
 "guards":true,"events":"none","stuck_guard":"vision_confirmed",
 "policy_tail_gate":"lifecycle","monitor":"off"}
```

At 500 use `lib="big", kref=8`; all other choices remain fixed. Proposed class is `exp.offline_search.rounds.r05.horizon_controller:CommitJudge` (not implemented here). Flags are `--os-blind --os-policy-tail --os-judge guard_only --os-no-shadow-native`; client `replan_steps=5`; π0.5 full K10 denoising. Existing control class is `r04.k10_policy_tail.judge:PolicyTailJudge`, with the two new kwargs omitted; K7 control additionally omits `--os-policy-tail`.

**Forecast by cell.** Same deployed library and synthesis; only control/execution and sensing change.

| π0.5 cell | Hypothesized SR | Conditional IR planning range | Assumptions behind IR |
|---|---:|---:|---|
| l10 / 50 | .80–.85 | .18–.23 | v=.50–.52; anchor q=.24–.34 |
| l10 / 500 | .87–.92 | .15–.20 | v=.50–.52; q=.18–.28 |
| spatial / 50 | .88–.94 | .13–.20 | v=.50–.55; q=.13–.25; no measured K7-tail baseline |
| spatial / 500 | .97–.99 | .115–.141 | v=.50–.52; q=.09–.14 |

The IR ranges evaluate `v*(.152+.848*q)`. The presumed l10 SR benefit is policy-plan completion, perhaps 0–4 pp relative to the existing HIT-tail arm, with a small possible regression; it is not a second +5.6 pp extrapolation from pure inference. Merely dropping a redundant vision evaluation should have zero SR effect if the same actions and RNG are preserved. Avoided subsequent policy calls have their own control effect, presently unknown.

**Cost/bytes.** T1; the corresponding π0.5 footprint table plus only per-connection original normalized/wire H-chunk buffers; optional monitor +208 B. No additional library data at either scale.

**Cheapest diagnostic and pilot.** First replay the hook on saved complete MISS chunks and verify exact wire slicing, rejection after exhaustion, and source bookkeeping; these lifecycle tests do not require LIBERO. Then coordinator: l10 at both scales, ten tasks × inits 0…9, three matched arms K7 HIT-tail / existing K10 / C10. Instrument five common inits/task on the first run. Spatial uses the same three arms at both scales as a regression screen; reuse existing completed K7 results only as historical controls, not as matched policy-noise replicates. Promote unchanged survivors to all 500 inits and a second seed base. On a short pure-policy instrumented subset, compare L10 requests to L5+policy-tail with no mid-chunk inspection to validate that the actual ten-control actions coincide.

**Kill criterion.** Stop a scale on ≥10 pp SR loss in the 100-init screen, a new task collapse, or any action-source/cursor mismatch. In confirmation seek IR improvement ≥.01 without a material SR loss, or a confirmed SR gain at comparable IR; a −3 pp l10 difference in one 500 run is a warning, not decisive given the documented ≈.016 paired SE. If repeats show C10 worse than K10 at comparable IR, keep the existing veto. Also report calls/episode and timeout counts: IR improvement achieved by prolonging failed episodes is not success.

**Variants.** Lifecycle only versus own-action residual veto; policy tails only (HIT-tail budget0 in a dedicated class, since K10 currently couples that budget to both sources); existing K10's unchanged gate. Do not combine with denoising reduction, relaxed no-progress, gripper commitment, or a new retrieval metric.

### 2 — GR00T mixed anchor cycles (`CycleTail`, G10 before G15)

**Pitch and mechanism.** Establish the first GR00T mixed controller with explicit chunk provenance and bounded tails. Start with ten executed controls from either source. Test fifteen only after identifying the pure-policy horizon effect. This is a controlled mixed baseline, not a claim that periodic calls beat the established π0.5/500 guard.

**Algorithm/API.** Wrap `BlindAWM` directly, avoiding K7's π0.5-only stock-parity assumption. `fit` uses `fit_data="same"`; `reset` sets episode-local anchor index j=0. At each real anchor, call AWM and set `Result.extras.os_force_miss = (j % cycle_k == 0)`, then increment j once. With `cycle_k=4`, the first anchor and every fourth later anchor call the full K8-denoising GR00T policy. Other anchors HIT. Every source issues an immutable chunk and a cursor. At intervening five-control requests, serve offsets5 and optionally10 from that original source; after the selected commit length, require vision. Policy source must remain policy-origin even when its continuation is billed as a zero-forward HIT. No guard on interpolated/fabricated visual keys is allowed.

**Concrete arms.** Four cells: GR00T × {spatial,l10} × {50,500}. Proposed `CycleTail` kwargs in `ARM_SPECS.json`:

```json
{"base_kwargs":{"lib":"current","fit_data":"same","kref":5,
  "serving":"anchor_tail","budget":1,"gates":"budget_only"},
 "cycle_k":4,"cycle_first_miss":true,"commit_blocks":2,
 "monitor":"off","guards":false}
```

Use big/kref8 at 500. `commit_blocks=1,budget=0` is G5; `commit_blocks=3,budget=2` is G15. Use `--os-judge guard_only` so **the method's per-episode anchor clock** schedules calls. Do not use `periodic:4` on the current blind plugin: its process-global request clock can preempt tails and mixes connections. Keep client replan5.

**Required plugin work.** Generalize K10's policy-tail adapter to GR00T using the already available GR00T CPU state/output adapter, retaining the original transformed wire actions. Add proposed `--os-policy-tail-blocks <1|2>`; `--os-policy-tail` alone retains the current π0.5 default. Store original chunk, source, anchor identity, remaining valid length and cursor until consumed; the present `not hits[-1]` test, one-shot buffer consumption, and compulsory vision immediately after the first tail must become cursor-aware. For H16 permit offsets5/10, never15 as a five-step response. No 8+8 implementation is needed for this first pilot. An L16 pure-policy client control is optional (IR .3125) and library-independent; it does not validate mixed 8+8 bookkeeping.

The first mixed baseline deliberately has no GR00T stuck/terminal MISS guard. Its calls are bounded in time by the cycle schedule, and its vision-free stretch is bounded by the chunk. A later guard variant must recalibrate min-camera visual/motion thresholds within each GR00T library, fix the closed-sign convention, preserve actual visual gaps, and validate its own semantics; merely removing K7's `SkipCell` is insufficient.

For direct sensing separation, the catalog includes optional G10 `checkpoint_rule="passive"`, proposed `--os-tail-look passive`: evaluate stage 1 at the midpoint, preserve the original response, and do not advance the anchor clock, retrieve a new plan, or schedule another policy call. This requires the persistent-buffer inspection extension described in proposal 3. Its steady-state IR is .258 at k4, versus .182 without the redundant look, at both library scales. The SR null is exact equality under deterministic replay; this control identifies vision cost without confusing it with execution length or policy-call frequency. G5→G10 alone changes all three and cannot provide that separation.

**Forecasts and cost.** For q=1/4 anchors, G5/G10/G15 steady-state owner IR is **.364/.182/.121333** at **both** library scales. This call interval can be chosen algebraically for an IR budget I: `k ≈ .848/(b*I-.152)`; k4 targets .182 at b2, not an inferred SR optimum. First-anchor forced MISS and terminal partial cycles raise finite-episode values; for J anchors, `q=ceil(J/4)/J`, and actual N5 must be used. Plan approximately .18–.23 for G10 and .12–.17 for G15, with the largest endpoint correction in short spatial episodes. Full-policy controls L5/L10/L15 cost 1/.5/.333333, independent of library.

Hypothesized G10 SR: spatial **.88–.96 at 50**, **.94–.98 at 500**; l10 **.62–.80 at 50**, **.74–.89 at 500**. These intentionally broad ranges are anchored only in the historical pure-cache/full-policy endpoints above. Ten-control execution might contribute 0–6 pp on l10 and approximately −2…+2 pp spatial; actual GR00T length effects are unknown. Periodic policy interventions can rescue or disrupt, so the two effects must not be added as independent measured gains. Vision saving alone has a zero-SR null. G15's SR is unforecastable beyond a sensitivity range of G10 −5…+3 pp; its second-tail disagreement is a reason to pilot, not to assume a ranking.

**Cost/bytes.** T1, GR00T BlindAWM pickles 22.842/58.604 MB spatial and 28.136/118.431 MB l10 at 50/500, versus owner 429/1068 MB. No new candidate rows; <4 KB original normalized/wire buffers per connection plus cursors; +208 B if the optional monitor is enabled.

**Cheapest diagnostic/pilot/kill.** Validate offset5/10 wire equality and sign semantics offline using stored actual GR00T chunks. Coordinator first runs pure-policy L5/L10, ten tasks × ten inits, both suites; those controls cover both library scales. In parallel scheduling terms only, screen G5/G10 at both scales and both suites on the same 100 inits; compare the complete R4 pure-cache tail controls when available. Promote G10 to 500 inits only if it avoids a 10 pp collapse. Add G15 at 100 inits only if G10 and its full-policy control survive. Kill G15 on ≥5 pp screen loss or a new task collapse; confirm any smaller tradeoff at 500. Kill the mixed spatial/500 recipe if paid calls consistently reduce SR without another benefit. An optional `cycle_k=2` rescue variant costs .288 at G10; it is a single prespecified higher-call point for l10/50, not an unbounded sweep. Pure-cache G10 is another useful endpoint.

### 3 — Look without cancelling an accepted tail (`InspectCommitJudge`)

**Pitch.** Sensing cadence and action replanning need not be identical. At a risky halfway point, inspect the scene, then continue the already accepted tail if the visual judge still accepts cache control. A look by itself should not erase the gripper event it was meant to monitor.

**Algorithm.** Start from C10. At the five-control midpoint, request vision if the planned issued actions4…9 change gripper sign, or the observed displacement after the actual head exceeds its own-library LOEO p99. An `always` inspection variant supplies the execution-versus-vision ablation. At this real visual query, run the unchanged K7 proposal and guards. If the verdict is MISS, invalidate the old tail, execute the new full policy plan's head, and let C10 govern that new plan. If the verdict is HIT, **keep the old plan's remaining five controls**, including gripper timing, instead of replacing them with the freshly retrieved head. Require a new ordinary anchor when the tail ends. A missed grasp that is invisible to proprioception can therefore still reach the visual judge, without every benign inspection creating a new plan.

For an old policy source, a visual HIT means the judge accepts continuing without a new policy call; it still serves the stored policy tail, not the cache proposal. All actual visual observations update guard history; executing a tail does not fabricate candidate progress or action provenance. This is a control-level hypothesis: cache-proposal confidence is not itself a calibrated score for the old tail.

**Implementation.** Proposed `InspectCommitJudge`, C10 kwargs plus `checkpoint_rule="event_or_residual", monitor="loeo_xyz99"`. New plugin flag `--os-tail-look event_or_residual` (or `always`) preserves the pending tail buffer when `_try_blind` requests a visual inspection. The plugin must apply the override **after the current visual verdict but before the action broadcast/history commit**, supplying both the original normalized slice and original wire slice. The current K10 buffer is consumed at blind-entry and cannot implement this just by changing method kwargs. Log the current inspected candidate/guard diagnostics separately from the actual issued source/rows; retain `vision=true` for a tail served after an inspection, and distinguish `cache_tail_look`/`policy_tail_look`. No new observation fields are needed in the harness; the offline harness cannot establish counterfactual success.

**Effect separation.** Compare three arms at each library scale: (a) ordinary replan-at-five; (b) inspect every five but preserve accepted ten-control plans; (c) selectively inspect and preserve the same plan. (a)→(b) tests action replacement, with vision fixed; (b)→(c) tests sensing/guard opportunities, with the same initial plan-retention rule. Also compare C10 with no midpoint inspection to (c). For a pure-policy passive-inspection control that never changes actions, IR is **.576** per ten controls (.5 for one full call plus .076 for the extra look); its no-inspection counterpart is .5. The SR null is equality, not an additional horizon gain.

**Forecast/bytes.** π0.5 l10 at 50/500: hypothesize ΔSR −2…+3 pp / −1…+2 pp relative to C10; spatial at both scales −1…+2 pp. Vision-only increment measured on existing cache-tail locations is .004835/.004714 l10 and .003975 spatial/500; spatial/50 is unmeasured. Plan **+.004…+.015 IR** for selective inspection at either scale if extra MISS fraction stays below about .01 per five-control slot; use `.152 Δv + .848 Δm` in the actual result. More inspections may save later calls, but no such credit is forecast. This is T1, C10's footprint at both scales plus the 208-B monitor and small source/cursor metadata. No additional library rows or larger-library prior.

**Cheapest diagnostic.** The measured event/residual counts above check that the intervention is sparse. On a short instrumented run, verify that a HIT inspection preserves all five original wire controls while a MISS replaces the source; verify guard state consumes only real visual observations. Existing R4 logs cannot answer the counterfactual visual verdict at a skipped look.

**Pilot/kill/variants.** After C10, coordinator screens no midpoint look / always inspect-and-preserve / event-or-residual inspect-and-preserve on ten tasks × ten inits in l10 at both scales, followed by spatial at both scales. Hold C10 source handling fixed. Confirm only a surviving selective rule at 500 inits. Kill on +.02 IR without SR benefit, a 10 pp screen collapse, or any stale wire/history mix. If always-inspect already loses to C10, do not assume the selective rule inherits its value. Variants are residual-only versus event-or-residual; **no gripper-sign locking, dwell constraint, or event-triggered automatic MISS**. The monitor calibration is solved at library-build time; its alert budget is an explicit operating choice, not an SR estimate from LOEO.

### 4 — One bounded successor bridge, only as an identification pilot (`BridgeCommitJudge`)

**Pitch.** Determine whether a completed ten-control cache chunk can occasionally be followed by one additional five-control demonstrated segment. This is the boundary experiment for looking less often than one chunk, not the default R5 controller.

**Algorithm/API.** Keep the original 16 anchor rows and normalized weights fixed. After a real HIT head and its exact same-chunk tail have both executed, consider `next²(rows)` once. Require all consecutive same-episode/task edges and nonterminal continuation, the join/gripper/residual screen measured above, and zero no-progress span at the last real anchor. Serve the weighted five-control heads at `next²(rows)`; pad the unexecuted response only for the wire shape. Set a bridge-used bit and require a real vision query immediately afterward. Total blind gap is at most ten controls after the initial five-control head, total commit15. Never rephase individual neighbours, renormalize away missing endpoints, hold terminal rows, chain bridges, or bridge a policy-origin plan. A policy MISS before completion invalidates the cache bridge eligibility.

Proposed `BridgeCommitJudge` kwargs are C10 plus `bridge_blocks=1, bridge_source="cache", join_quantile=.95, minority_mass_max=.2, monitor="loeo_xyz99", require_all_successors=true`. Existing `--os-blind --os-policy-tail --os-judge guard_only` suffices for serving π0.5; the new method owns the cache bridge state and mandatory-look rule. Return `BlindResult` with successor rows and fixed original weights, an explicit bridge source and anchor ID. `fit` stores the own-library natural-join p95 and monitor; `bytes_per_entry` does not add rows because successor/action arrays already exist. T1; same π0.5 50/500 footprints plus a few scalar thresholds and 208-B monitor.

**Forecast.** At 50-library l10 the prior is unfavorable: hypothesize **−5…0 pp SR**, at 500 **−2…+1 pp**; spatial at 50 **−3…+1 pp**, at 500 **−1…+1 pp**, relative to C10. These are risk ranges motivated by R4's sparse-library continuation failure, not error-based SR predictions. If a fraction f=.1… .3 of cycles obtains one bridge with unchanged anchor q, `IR_bridge = IR_C10 * 2/(2+f)`, a **4.8–13.0% relative IR reduction at either library scale**. For an IR .20 base this is .1905… .1739. Execution extension and skipped corrective policy opportunities can change q and episode length substantially; the observed 14.60%/16.79% overlap with imminent l10 MISSes prevents a stronger savings forecast. There is no basis for predicting an SR gain from lower vision cost alone.

**Cheapest diagnostic and pilot.** The completed audit shows coverage and exact action provenance; it does not show safe skips. First use a single-landmark randomized eligibility trial: at the first eligible bridge per episode, stable task/init assignment chooses LOOK versus BRIDGE with probability 1/2; all other rules stay C10. Log eligibility before assignment and execute only one experimental bridge per episode. Screen both l10 scales on all ten tasks × inits0…9; do not test only favorable tasks. If there is any substantial small-library regression, stop that scale. Spatial/500 is the most promising low-call-loss location; spatial/50 is a required transfer cell, not assumed equivalent. Only after a surviving landmark trial should an every-eligible-bridge arm receive 500-init confirmation. Kill on new task collapse or ≥5 pp landmark SR loss; if an every-bridge confirmation loses >2 pp repeatedly for <.02 IR saving, stop. A noisy 50-episode treatment group cannot establish small superiority.

**Variant: C's control-step library.** `r04/k1_blind/control_step.py` G changes retrieval geometry; GS also splices actually executed five-control demonstration heads at fractional offsets. It is not a certificate of later vision-free execution. If a G/GS anchor is used, continuation must advance its exact `(parent, offset)` cursor by the actual controls consumed; after ten controls it is two real rows ahead with the same fractional offset, subject to all endpoint checks. MixedJudge's static `Tables.HD` votes/dispersion/progress must be rebuilt from aligned heads and fractional progress, as C's report already requires. Keep G versus GS versus bridge separate so synthesis, retrieval geometry, and vision savings are not conflated. R4 G/GS closed-loop results were not available in the snapshot; no combined arm is promoted here.

## 3. Rejected ideas and measured reasons

1. **Unconditional multi-chunk stitching or resurrection of sparse-library phase particles.** π0.5 l10/50 phase-B2 is .700 @ .267979 versus its own-chunk tail .806 @ .241513; at 500 phase-B2 is .862 @ .177621 versus tail .880 @ .203281 (`analyze_logs.py`, binding FINDINGS). Library next-head versus stored-tail mismatch remains .2527/.2852 at 50/500, and even the stricter bridge screen hides 1,105/1,280 observed MISS locations. This rejects routine long blind stretches. Proposal 4 is limited to measuring the value of one bounded opportunity, with an unfavorable prior at 50. GR00T L15 still uses its own original chunk; it is not evidence for a library bridge beyond H16.

2. **Proprioception or gripper-event count as a sufficient task-phase/stuck oracle.** Own-library monitor thresholds alert on only .3607%/.5930% of actual l10 cache tails at 50/500, despite episode failure rates 19.4%/12.0%; these denominators differ, so this is a coverage limitation, not a recall estimate. Normal arm motion can coexist with failed object interaction. R4 already found the motion-only stuck guard fired about 2.8× too often. C's `REPORT.md` measures immediate-action-neighbour event-count mismatch of **31.68%/35.84% π0.5 l10** and **38.13%/38.28% GR00T l10** at 50/500. A hard phase gate or gripper dwell rule is not justified. Proposal 3 uses events to request vision and leaves the commanded event intact if the visual judge accepts continuation.

3. **After a MISS, finish the rejected cache plan, or buy a cheap warm-start rescue instead of a full policy tail.** A MISS has selected a new source; appending an old cache tail to its head makes a new, unvalidated hybrid chunk. Neither the R4 truncated MISS logs nor offline teacher error identifies its closed-loop value at 50 or 500. The measured positive controls are completion of the policy's own chunk (l10 .904 at IR .5) and the cache's own chunk (.806/.880 at 50/500), not cross-source splicing. The related warm-start line also gives no rescue rationale: π0.5 LIBERO exact continuation is .778 l10/.926 spatial versus full .844/.986; GR00T l10 exact continuation is .712 (one step) or .818 (two) versus full .880 (§6.15). Those results do not experimentally disprove every hybrid, but they remove support for this proposal. Full policy calls stay fully priced; K2 reduction is the owner's orthogonal drop-in, never system IR credit.

## 4. Pilot discipline and what remains unknown

For every pilot, keep the model, deployed library and ordinary AWM synthesis fixed when changing the controller. Report four separate effects: **synthesis** (unchanged for C10/G10; changed for GS/bridge), **method at fixed library** (guard/metric unchanged except explicitly identified variants), **library** (50→500 reported separately), and **control** (execution length, inspections, policy calls). The recipes retain the established kref5/kref8 scale-specific presets; a pure row-count attribution requires an additional matched-kref comparison, rather than crediting the entire cross-scale difference to library size. All initial episodes have real vision; all blind stretches have a hard valid-action horizon.

Use ten tasks × inits0…9 only to screen collapses. A survivor's full result uses the common ten × fifty inits, with all tasks retained, and a second seed base for close l10 differences. Seed bases 5101/5102 are proposed identifiers; the existing `SeededInference` seeds the process, but stochastic concurrent request order still differs. New mixed controllers should use the same explicit policy seeding hook; matching a CLI seed alone does not prove paired policy noise. Never select the final horizon/threshold on held-out task success and then report the same pilot as confirmation.

Minimal accounting logs: episode/task/init/attempt, actual controls consumed, full issued valid chunk and original wire chunk on the instrumented subset, source and source-anchor ID, cursor/remaining length, vision mask, policy-call flag, guard reasons, visual-anchor indices, displacement residual and thresholds, original/current candidate rows and weights, and completion status. Existing `--os-log-inputs` plus `--os-log-r4` supplies full `a_exec`, `wire_actions`, states and keys on a short subset. For full sweeps, a small new chunk-only log mode would avoid saving large raw-key histories; current `served_head` alone cannot answer the mechanism question. Physical failure types require client state/contact/object predicates or video annotations, with criteria fixed before reviewing arm outcomes.

SR, owner IR, V/N5, M/N5, actual calls and controls per episode, timeout counts, source-specific tail acceptance, and invalid/exhausted-tail attempts must all be reported. Include per-task paired outcome changes and episode-level uncertainty; decision samples are serially correlated. Do not call repeated identical offline replay samples additional independent evidence.

Unknowns that the pilots must resolve are: the causal share of l10's gain due to avoiding plan interruption; policy-tail value specifically after guard-triggered MISSes; spatial/50 HIT-tail transfer; GR00T mixed and L10/L15 success; whether selected inspections recover object failures without excessive extra calls; and whether any beyond-chunk bridge survives its hidden loss of visual correction. The current evidence supports **bounded completion first**, not a universal longest-possible look interval.
