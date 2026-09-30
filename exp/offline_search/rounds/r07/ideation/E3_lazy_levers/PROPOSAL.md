# E3 — Lazy levers inside easy stages, and the safety signal

## 1. Direction summary

**Recommend two separable proposals:** bounded successor following with a state-deviation valve; and a stage-specific single-camera retriever. Test them separately before combining them. A valve requests a **look**, not automatically a policy call. Keep the existing call allocator/stall detector outside these levers.

I read the R7 brief first; then the library/store and profiler contracts in ledger §§5.1–5.3, `harness/dims.py`, `g1_awm/awm.py`, `closed_loop/{README.md,blind.py}`, R4 `k1_blind/{blind_awm.py,control_step.py}`, R4 `k3_cost/{method.py,INTEGRATION.md}`, the owner cost table, R5 Q6 handback, R1 findings, and P3 `SCHEMA_V2.md`, `read_v2.py`, and calibration/loading code. Paths below are relative to `exp/offline_search/` unless absolute.

Computed evidence covers all eight deployed libraries, all ten non-test B-val A episodes per cell, and the first P3 A replicate: twenty episodes per cell, tasks 0–9 × inits 0–1. P3 is an evaluation diagnostic here, **never threshold-fitting data**, notwithstanding its historical `provenance.split=validation` metadata. I used accepted strict tables, exact logged 16-member kernels, normalized decision states, and 34,534 active P3 controls. No rollout, simulator, model, or serving code was changed.

Three surprises materially change the design:

- **Gripper-change windows really differ from smooth motion.** In π0.5 L10-500 B-val, the 20-control displacement valve alerts on 44.7% of supported gripper-change windows versus 2.1% of moving, stable-gripper windows. These are library-stage proxies, not inferred grasp success.
- **A's predicted tails are not the library's executed trajectory.** At offset five, median tail/successor-head discrepancy is 0.140–0.204 action-sigma RMS. Thus even ordinary chunk commitment does not exactly follow recorded demo states.
- **Single-camera preference is model dependent.** Wrist beats third-person on the imitation diagnostic in every π0.5 cell; GR00T Spatial-500 slightly prefers third-person. Both cameras have the lowest overall error in all eight cells. Historical wrist SR gains therefore cannot be justified by claiming lower action error.

## 2. Proposal P1 — Bounded successor following with a free deviation valve

### Hypothesis and mechanism

Inside a coherent, stable-gripper stage, recomputing vision often re-identifies essentially the trajectory already being followed. Preserve that identity and spend zero additional IR until a data-derived horizon, stage boundary, structural endpoint, or state deviation requires a look. **This is a hypothesis about additional blind execution; existing recordings do not identify its SR.**

At a real vision HIT, retain the exact retrieved rows `r[16]`, normalized weights `w[16]`, accepted action chunk, anchor state, stage, and actual executed-control clock. For each subsequent chunk, advance each member along its own `next` chain, preserving member identities and weights. Never rerank, select a new episode, discard a member, or renormalize surviving weights while blind. A missing required successor forces a look; an episode/task crossing is invalid. The endpoint audit below conservatively requires all sixteen members, including numerically tiny weights.

**Chunk accounting is essential:**

- **π0.5, H=10:** consume the accepted ten controls; at elapsed ten use `next²(r)` with the same weights to synthesize another ten-control chunk. Repeat only within the permitted bound. Evaluate state every five controls, including inside a chunk.
- **GR00T, H=16, current five-control protocol:** the simple first implementation consumes fifteen controls of the 16-step prediction, then uses `next³(r)` for the next prediction. Fall back to A's ten-control commitment wherever fifteen is not admitted. This is explicitly **15-of-16**, not exact sixteen. The brief already reports SR losses from indiscriminate fifteen-control execution.
- **Exact sixteen is a separate, auditable variant:** there is no library row at control sixteen. After the initial 16-step prediction, construct each subsequent 16-control segment from the linked **executed-head tape**: at elapsed `t`, use row `next^floor(t/5)(r)` and offset `t mod 5`, concatenating recorded five-control heads as needed. At the first seam this starts at offset one of `next³(r)`. Mix the same sixteen tapes with the original weights. This avoids pretending `next³` starts at sixteen or repeating a padded final action. It changes successor synthesis from full predictions to recorded heads and must have its own arm.
- A queue can assemble five-control serving blocks across the sixteen-control seam. With existing decision timing, stop at the last five-control boundary no later than the learned cap; exact looks at sixteen/thirty-two require variable client commitment. Do not credit theoretical 16/32 timing to a 15/30 implementation. Predicted state between library rows is interpolated for diagnostics; this interpolation is not an observed library state.

### Signal, stages, and calibration

Let `x` be valid proprioceptive state and `S` its scale estimated from successful library trajectories. Padding is excluded. For the diagnostic I used the standard deviation of each of the eight valid normalized coordinates; an identically constant coordinate is omitted in a deployment implementation. A new robot supplies its state/action masks and control period through the interface manifest, not hard-coded dimensional assumptions.

For elapsed age `h`, let `mu(h)=sum_i w_i x_demo(next^h(r_i))` on the five-control grid. Monitor

`D_delta(h) = RMS[S^-1 ((x_now-x_anchor) - (mu(h)-mu(0)))]`.

Also monitor absolute mismatch `D_abs(h)=RMS[S^-1(x_now-mu(h))]`. The incremental tube tolerates an initial demonstration offset; the absolute tube prevents declaring an already off-trajectory state safe solely because it stays still. The numerical replay below measures **D_delta alone**; adding the absolute gate is a proposed extension. For production pose states, calculate relative rotations on SO(3) before scaling; linear differences of wrapped axis-angle components are only the current normalized-state diagnostic.

Calibration recipe, with no task/suite names or state-unit constants:

1. Hold out each complete successful library episode, retrieve its anchor kernels from the remaining episodes, and follow their successors. For a final calibration, fit the PCA/metric without the held-out episode as well, or use explicit train/calibration episode splits. Our preliminary numbers use frozen PCA/metric with **candidate-only LOEO**, and are labeled accordingly.
2. Define the reference commitment from the deployed baseline: currently ten controls for both policies. Set the displacement radius to the empirical 95th percentile of the maximum residual over that commitment. Fit the absolute radius to the corresponding absolute residuals. The 95% convention is a declared distribution-free rank rule, not a LIBERO-unit threshold or an SR guarantee. Use equal episode weights in the final fit; report both row-weighted and episode-weighted versions.
3. Define easy stages by the kinematic segmentation interface from E1. A minimal standalone fallback labels stable-gripper movement, stable-gripper low motion, and gripper changes; the motion boundary is the library median standardized motion. Gripper modes/boundaries come from the action interface or library modes, not assumed signs on another robot. The current diagnostic uses the stored two gripper signs and the **top-1 demo's** stage as a proxy.
4. For each stage, choose the longest integer number of chunks whose monotone, held-out continuation-error envelope remains inside the baseline radius at the same 95th-percentile convention. Candidate lengths stop at observed stage ends/library ends. No supported length beyond baseline means no extra blindness. Insufficient episode support means use the pooled conservative rule or keep baseline; show the uncertainty rather than claiming stage certification.
5. Admit a blind run only if the retrieved trajectories agree on the stage sufficiently under a library-calibrated agreement quantile, their planned controls remain within that stage, and the initial absolute state is in support. Stop before a gripper transition or unresolved stage boundary. Preserve the already accepted grasp/place micro-sequence unless the valve itself fires. Low motion alone is insufficient: the existing calibrated stall signal retains priority.
6. Keep the radius fixed as age grows; widening it with age would turn drift into permission. A look resets the anchor and kernel. A MISS invalidates the cache-following plan; do not follow a cache reference as though its chunk had been executed.

The method's levers are **look less**, with a free state-deviation signal and stage-level eligibility plus event-triggered termination. It does not introduce R6's failed cache–policy disagreement call-placement predictor. The maximum horizon and valve threshold are separate: the allocation policy may shorten a calibrated horizon to meet its budget, but should not relax the radius to buy more blindness.

### Expected frontier effect and cost

At unchanged MISS share, full-vision look frequency `1/J` gives π0.5 `.152/J + .848 m`, GR00T `.148/J + .852 m`, where `J` counts five-control decisions per look. Ideal cache-only π0.5 costs at ten/twenty/thirty controls are `.076/.038/.0253`; GR00T at fifteen/thirty are `.0493/.0247`. These are arithmetic limits, not expected realized IR. Calls, stage boundaries, unavailable successors and valve looks increase them.

The boundary-only B-val replay estimates π0.5 IR `.040–.054` with a twenty-control cap versus `.076` for ten, but keeps the original A path and ignores within-chunk valve checks/terminal partial controls. Treat this as screening headroom, not a deployment prediction. No numerical SR gain is claimed. Per-cell priorities appear below.

**Tier T1:** same policy encoder when looking, CPU-only library arithmetic when blind, no additional model. Arithmetic is O(16 × state dimensions) for the tube, plus O(16 × H × action dimensions) when synthesizing. The local NumPy tube-plus-full-payload microbenchmark measured p50 **0.053–0.062 ms**, p95 **0.067–0.095 ms** across the eight banks. This excludes serving/logging, pose conversion, network/client time and stage labeling. “Free” means zero owner IR, not zero wall time.

### Plugin sketch and kill criteria

`query`: ordinary A retrieval, save all sixteen members/weights and a connection-local chunk/clock record. `blind_step(BlindQueryView)`: inspect current `rs`/raw pose, validate execution count and lifecycle, evaluate stage/cap/tube/end conditions, then return `LookReason` or `BlindResult`. Advance cursors only after actual accepted execution. Carry offset/control count explicitly; never infer it from H or intended commitment alone. Existing `blind.py` policy tails only support complete five-control blocks and cannot implement the exact-sixteen seam unaided.

`judge`: a tube violation requests fresh vision; the normal post-look judge/call allocator decides MISS. Already-due calls, calibrated stall events, reset/task changes and malformed states retain priority. History must mark blind keys missing, and metadata must expose all members rather than top ten. Test the queue against recorded action prefixes and transforms; do not equate a proposed H-step array with H controls executed.

**Drop or narrow P1** if proper episode-separated calibration leaves no stage with additional supported horizon; if most apparent savings vanish after all-sixteen endpoints and actual controls are counted; if the matched-call closed-loop SR difference is convincingly negative; or if blind drift escapes before the five-control valve can help. Reject the exact-sixteen variant independently if its seams introduce action/gripper discontinuities beyond the baseline's library-calibrated seam distribution. A useful offline monitor is not sufficient to pass this gate.

## 3. Proposal P2 — Stage-specific “look half,” with a complete policy fallback

### Hypothesis, mechanism, and stage policy

At a stable manipulation stage, one view may preserve the relevant local geometry while the second view adds cost or confusing context. Use the current policy's encoder on the selected camera only, then retrieve using that camera's representation. **Do not delete a key after encoding both cameras and count that as savings.**

Maintain three library-fitted metrics: full `64+64+state`, wrist `64+state`, and third-person `64+state`. Refit each action-supervised Mahalanobis metric in its actual feature space; removing one block from the fitted full metric is not equivalent. Retrieve the same top sixteen with the existing kernel convention. Refit all visual guard/stall reference statistics for the selected representation; absent camera keys must not enter distance, confidence, or visual-still checks. R4's `WristAWM` is the concrete 72-D precedent for π0.5.

Choose the camera **before stage 1**, using the last retrieved stage, its successor annotations, proprioception, and the fitted stage table. Unknown stages, episode start, unreliable phase and disputed transitions use both cameras. After a one-camera look, poor support under that camera's own LOEO coverage distribution triggers encoding the other camera and joint retrieval. A MISS completes the missing camera before the unmodified full policy runs, preserving its expected inputs.

Candidate stage/view associations, explicitly hypotheses rather than visibility labels inferred from robot state:

| Stage/evidence condition | First candidate | Reason and limitation |
|---|---|---|
| Stable nearby manipulation with coherent gripper state | Wrist | π0.5's four cells all favor wrist over third-person on the offline comparison; local visibility itself is unobserved here. |
| Gross movement/global scene disambiguation; wrist view may be uninformative | Third-person, only if the stage table prefers it | GR00T Spatial-500 favors this view slightly; no universal “transport = third-person” rule is supported. |
| Gripper-change boundary, uncertain contact outcome, conflicting neighbours | Both | Single-camera imitation penalties and transition-associated tube alerts argue for retaining information here. |
| Long quiet interval | Chosen stage camera or P1 blindness | A small proprioceptive error does not distinguish successful holding from an object missed by the gripper. Retain the stall/bounded-look rules. |

### General calibration and interface work

Fit a camera table on held-out library episodes, using the stage definitions above/E1. Compare full-chunk action reconstruction, gripper-event reconstruction, successor-state support, and episode-level uncertainty for all three modes. This is selection of a retrieval representation, **not a prediction of where policy calls have value**.

Do not decree an acceptable action-RMS constant. One proposed general admission rule is: the extra held-out loss from deleting a camera must fit inside the full retriever's own library instability, measured by deleting an additional candidate episode and reretrieving. Apply the same rule separately to gripper-event error; use a declared empirical quantile and episode bootstrap. Pick the cheaper admissible camera, or the better admissible one when price is unknown. If neither passes, keep both. This rule and its eligible stage share remain unmeasured; the requested camera profile must test it before deployment. It may be too conservative.

On another robot, map cameras by the manifest, fit the same three—or available-subset—representations, and remeasure encoder/completion costs. Camera placement or robot identity never selects an arm by a benchmark name. For GR00T, first verify that an isolated camera's key equals that camera's key in the full path and that its tower work can actually be skipped; the existing wrist adapter expressly supports π0.5 only.

`query` routes to the fitted single-camera metric and records modality plus its kernel. `blind_step` supplies the *next requested modality* when it returns a look reason. This needs a small pre-stage-1 camera-request interface: a static `camera_mode='wrist_only'` declaration cannot express a per-stage switch. The serving adapter defers the unencoded image request-locally, completes it when necessary, and preserves the original full policy inputs. `judge` uses camera-specific calibration; its MISS interceptor completes the omitted tower exactly as in the π0.5 K3 contract. The full/wrist key parity checks must precede claiming any cost reduction.

### Cost, expected effect, and kill criteria

**Tier T1.** Reuse the policy encoder; the single-camera metric has 72 rather than 136 features in this deployment. Historical CPU measurements in ledger §10 report WristAWM `.86/1.03 ms` at 50/500 versus AWM `1.17/1.41 ms`; a dynamic adapter still needs its own timing profile. Store separate metric fits offline; online runs only the selected one unless escalation is necessary.

Use the owner's R4 π0.5 proportional-latency prices, explicitly an assumption: a wrist look `.055198`, a missing-camera completion on a wrist-origin MISS `.049890`, and the full call's remaining stages `.848`. For wrist-only look schedules:

`IR = .055198 v_w + (.848 + .049890) m`, with `m <= v_w`.

For mixed full/wrist schedules, add `.152 v_full + .848 m_full`; count wrist-origin and full-origin calls once each. Completion after an uncertain one-camera HIT also costs encoder work; it must be logged, with its price measured/validated rather than silently treated as free. Third-person-only price is **absent from the owner table**; do not assume symmetric latency. GR00T's table contains only full `.148` vision / `.852` remaining policy cost, so its one-camera frontier needs profiling. Applying π0.5's ratio to GR00T would be a new, unverified assumption.

Ideal cache-only wrist at ten controls is `.0276`, **63.7% below** two-camera `.076`; wrist plus twenty-control following would be `.0138` before boundary/valve looks and calls. The historical 41% mixed-IR saving and `.990 @ .074` wrist+blind result are evidence of plausibility from older retrieval, not forecasts for today's A. At high MISS shares, the `.848/.852` policy cost dominates and these levers cannot close an SR gap by themselves.

**Drop or narrow P2** if no stage survives the representation-admission rule; omitted-camera information is needed precisely where the cheaper mode is selected; new full-policy completion breaks parity; or actual encoder/completion dispatches eliminate the saving. A matched-call, matched-commit closed-loop SR loss is decisive. Small offline action-error changes alone are not: the historical wrist results already show that this surrogate can rank methods incorrectly for SR.

## 4. Expected effect by cell (hypotheses, not SR estimates)

| Cell | P1 priority and frontier expectation | P2 priority and frontier expectation |
|---|---|---|
| π0.5 L10-50 | Restrict to stable movement: B-val alerts20 9.3%, change-stage 39.3%. Can lower the vision floor; cannot plausibly repair A's large SR gap without calls. | Wrist screening is reasonable; historical standalone SR was unchanged. Keep both at unresolved boundaries. |
| π0.5 L10-500 | Strong first P1 cell: stable movement alerts20 2.1%, change-stage 44.7%; larger library supports 89.8% of all windows to twenty. | Wrist-first easy stages; extra imitation error .017 versus third-person .055. Historical standalone wrist −2.9 pp makes a global switch unattractive. |
| π0.5 Spatial-50 | Structural endpoints dominate: only 50.0% joint support to twenty. Prefer short, stage-bounded extension. | Highest empirical priority for wrist: older replicated SR gain, but retest on A and keep its calls fixed. |
| π0.5 Spatial-500 | Low-motion support is promising, but 20-control alerts differ between B-val 12.2% and P3 3.0%; select conservatively. | Best low-IR combination candidate given historical `.990 @ .074`; both-camera baseline already near pure SR. |
| GR00T L10-50 | Most cautious: B-val alerts grow 9.8% at fifteen → 33.9% at thirty → 53.2% at forty-five. No blanket native-chunk extension. | Wrist/third are close and both worse than full on imitation; require stage evidence and a real encoder-cost measurement. |
| GR00T L10-500 | Short extension candidate; alerts20 16.2%, change-stage 35.9%. Existing calls carry important SR, so keep them. | Wrist preferred offline; extra error .019 versus third-person .036. Benefit remains conditional on tower isolation. |
| GR00T Spatial-50 | Fifteen-control alerts 4.9% versus thirty 23.1%; favor short extensions, retain transition looks. | Nearly tied one-camera errors; no supported universal wrist rule. Profile both and their costs. |
| GR00T Spatial-500 | Best GR00T candidate for low IR: alerts15 6.2%, alerts30 14.5%; preserve its already strong A SR. | Third-person slightly preferable on imitation; policy/vision cost parity must be established first. |

## 5. Preliminary evidence and reproduction

### State deviation and valve firing on existing paths

`analyze_lazy.py` validates successor episode/task/step continuity, uses exact frozen A fits, excludes each source episode from library candidate retrieval, and retains the logged sixteen weights in replay. Libraries are `current` at 50, π0.5 `bpool_cs` / GR00T `bpool_all` at 500. Successful library episodes per cell are 50/436, 49/487, 50/427, and 50/456 in the table order below. Candidate banks retain the actual deployed failed episodes; calibration queries are successful demos.

Radius = pooled row-weighted empirical p95 of the maximum standardized displacement error at five/ten controls in library candidate-LOEO. Cumulative alert = any check above that fixed radius by the stated age. Denominator = anchors whose actual future observations **and all sixteen successor chains** exist at that age. These are overlapping windows, not independent Bernoulli trials; no nominal binomial confidence interval is asserted.

| Cell | B-val / P3 anchors | B-val jointly supported at twenty | B-val alert % at 10 / 20 / 30 / 40 controls | P3 alert % at 20 / 40 |
|---|---:|---:|---:|---:|
| π0.5 L10-50 | 346 / 640 | 216 (62.4%) | 4.0 / 9.3 / 16.3 / 19.8 | 11.8 / 23.1 |
| π0.5 L10-500 | 274 / 589 | 246 (89.8%) | 3.1 / 10.6 / 18.4 / 27.6 | 11.3 / 31.7 |
| π0.5 Spatial-50 | 152 / 254 | 76 (50.0%) | 4.6 / 6.6 / 18.5 / 26.3 | 6.2 / 25.9 |
| π0.5 Spatial-500 | 135 / 226 | 90 (66.7%) | 4.8 / 12.2 / 15.1 / 21.7 | 3.0 / 13.3 |
| GR00T L10-50 | 365 / 731 | 299 (81.9%) | 4.5 / 16.4 / 33.9 / 49.6 | 18.6 / 47.8 |
| GR00T L10-500 | 290 / 618 | 260 (89.7%) | 7.3 / 16.2 / 22.4 / 25.7 | 13.6 / 25.6 |
| GR00T Spatial-50 | 133 / 235 | 78 (58.6%) | 2.3 / 9.0 / 23.1 / 43.6 | 4.5 / 13.3 |
| GR00T Spatial-500 | 122 / 231 | 89 (73.0%) | 2.0 / 7.9 / 14.5 / 22.7 | 3.0 / 10.6 |

Support loss is mostly a library-chain issue, not just observed episode completion: at twenty, π0.5 Spatial-50 has 132 observed future states but only 76 fully supported kernels; L10-50 has 326 versus 216. Full counts are in `additional_summary.json`. Do not quietly extend the terminal row or drop a low-weight member to improve these figures.

On the **same supported-through-forty cohort**, median displacement error grows from `.016–.055` at ten to `.104–.243` at forty. For GR00T L10-50, p90 grows `.205 → .419 → .598 → .785` at 10/20/30/40, while its fixed radius is `.368`. Thus the apparent case for blindness is a short stable-stage extension, not unrestricted replay. The common-cohort table is in `EVIDENCE_TABLES.md`.

Changing the calibration unit matters: p95 of episode maxima gives radii `.470–1.172` rather than row-window radii `.264–.399`, reducing twenty-control B-val alerts to **0.8–4.4%**. A conservative episode-level threshold can become an insensitive drift detector. Neither threshold has a deployment safety theorem; source trajectories and synthesized execution have different dynamics.

`stage_and_cost.py` provides a limited stage sanity check. At twenty, π0.5 L10-50/500 gripper-change alert rates are **11/28 = 39.3%** and **17/38 = 44.7%**, versus stable movement **4/104 = 3.8%** and **2/96 = 2.1%**. GR00T Spatial-500 has **7/22 = 31.8%** at gripper changes, versus **0/34** low-motion and **0/33** moving windows. These small, correlated counts support profiling a stage gate; zeros do not certify safety or grasp success.

### Per-control telemetry and GR00T's sixteen-step horizon

`analyze_controls.py` decodes only robot position/finger measurements from P3 A_r0 controls. It reconstructs their normalization from matched raw/normalized anchor states and compares to interpolated library trajectories, with a separately calibrated five-coordinate displacement radius. The π0.5 affine residual is below printed 1e-6; GR00T's maximum residual is `.0020–.0040` normalized units, consistent with quantized stored states. Rotations are excluded from this sub-control audit.

At twenty controls, every-control alerts are **11.0–18.1%** across cells; sampling every five catches **7.1–16.5%**. An additional **1.1–4.9 percentage points** of supported windows have an excursion only between decision checks. This is an interpolation-based diagnostic: some excursions may reflect normal curvature between sparsely recorded demo states. It illustrates the distinction between the existing sampled server monitor and a control-rate monitor; it does not label the missed excursions unsafe.

| GR00T cell | Supported P3 windows at 16 / 32 | Any-control position/finger alert % at 16 / 32 |
|---|---:|---:|
| L10-50 | 526 / 454 | 12.9 / 35.9 |
| L10-500 | 511 / 468 | 12.3 / 24.6 |
| Spatial-50 | 155 / 120 | 7.7 / 17.5 |
| Spatial-500 | 167 / 135 | 9.6 / 25.9 |

These are actual robot states at those control ages **on A's original path**, interpolated demo references, and a positional/gripper statistic. A still re-looked at ten. They do not estimate physical divergence caused by executing sixteen/thirty-two blind controls, and they should not be numerically mixed with the eight-coordinate table above.

### Camera deletion and chunk consistency

`analyze_cameras.py` refits each one-camera action metric on its deployed library and cached camera PCA, then excludes source episodes from candidates. The full metric is A's frozen main metric; episode-start rows are excluded to avoid its distinct early metric. Means below weight successful source episodes equally; all three share the same query rows and action-sigma normalization.

| Cell | Both cameras | Wrist only | Third-person only |
|---|---:|---:|---:|
| π0.5 L10-50 | .303 | .332 | .353 |
| π0.5 L10-500 | .264 | .280 | .319 |
| π0.5 Spatial-50 | .360 | .380 | .394 |
| π0.5 Spatial-500 | .286 | .304 | .334 |
| GR00T L10-50 | .369 | .412 | .416 |
| GR00T L10-500 | .345 | .365 | .381 |
| GR00T Spatial-50 | .350 | .379 | .383 |
| GR00T Spatial-500 | .337 | .363 | .359 |

These are five-control action imitation RMS, not closed-loop losses or unbiased fully refitted LOEO estimates. Whole-native-chunk and gripper-sign errors, stage-specific means, and paired episode standard errors are in `cameras_*.json`. Example: GR00T L10-50 low-motion means are both `.327`, third `.359`, wrist `.369`; a stage can reverse the aggregate wrist preference.

The action-continuity audit compares each library prediction's tail to the next row's actually executed head. Offset-five median discrepancy is `.140–.204` sigma RMS, p90 `.338–.609`; gripper-sign mismatch is **1.1–6.8%**. GR00T offset-ten median discrepancy rises to `.207–.233`, p90 `.466–.782`. Therefore following a predicted 16-step chunk and following successive five-control demo heads are distinct interventions. The tube must tolerate baseline chunk mismatch without hiding additional error.

### Files and commands

All scripts, JSON outputs and `EVIDENCE_TABLES.md` are in this directory. Compact reusable arrays are only in `/tmp/r7_E3_lazy_levers/`. Run `reproduce.sh` from the repo root; it serially invokes the following scripts for both models with the required affinity/environment:

```bash
taskset -c 26-27,70-71 bash exp/offline_search/rounds/r07/ideation/E3_lazy_levers/reproduce.sh
```

The scripts read `/home/weiland/trace_runs/offline_search_store/{library,derived}` and `os_closed_loop/{r06_c_cal,r06_p3_pilot}/tables`, plus input archives referenced by those strict tables and fit paths declared by `p3_profiling/campaign.py`. Exact deployed fit/library paths are recorded in every `evidence_*.json`. Reproduction writes only this ideation directory and its assigned scratch directory; no GPU or simulator is used.

## 6. Profile tools wanted

**First request: `follow_tube` — distinguish admissible extra blindness from ordinary template mismatch.**

- Question: for each kinematic stage and native chunk count, how much *new* drift appears, how early does the valve see it, and how much extra vision does it buy back?
- Existing inputs: all-sixteen kernels/weights, successor arrays, valid state/action masks, recorded library states/actions, B-val decision archives, P3 controls, actual commitment and source fields. Use episode-separated fits and the existing E1 stage annotations when available.
- Offline outputs: stage × age survival curves; absolute/incremental and groupwise pose/gripper residuals; fixed-cohort and censored denominators; calibration-unit sensitivity; endpoints including lost probability mass; predicted-tail/head discrepancies; first alert lead time; five-control versus dense excursions; camera-conditioned versions; and event/radius/IR breakdowns. Include `coverage`, stage `breakdown`, `timeline`, and `compare` against A and the fixed-cap version.
- Decision: admit P1 at all, choose calibrated stage caps, decide whether the absolute gate adds useful protection, and reject unsupported GR00T seams. **No new telemetry is required for these descriptive outputs.**
- The causal gap requires a later, small closed-loop profile with randomized allowed commitment lengths and valve on/off packages on non-test starts. Log intended/actual controls, immutable anchor sixteen rows/weights, every successor cursor/intra-row offset, state, tube components/radii, stage, forced-look reason, actual camera-stage dispatches, MISS source and accepted outcome. Do not call fixed-observation replay a blind-rollout response curve. This is a requested future profile, not work launched here.

**`camera_delete` — choose a view by stage rather than by suite.**

- Inputs: stored per-camera policy keys, exact library partitions, successful episode folds, action chunks, successor states, B-val same-observation shadows, existing resampling records where present. Fully refit the 72-D metrics within the declared folds; avoid using P3 outcomes to select a camera.
- Outputs: per-stage full/wrist/third reconstruction and event errors with episode uncertainty; extra-candidate-episode deletion instability; camera disagreement/support and proposed escalation rates; eligible cheap-stage share; paired B-val action diagnostics. No new encoder is trained. Existing keys suffice for retrieval screening; this does **not** establish isolated-tower key parity.
- Decision: accept/reject the proposed data-derived camera-admission rule, which stage/view pairs deserve closed-loop profiling, and whether omission mistakes cluster at the same events as tube alerts.

**`lazy_cost_parity` — account for actual work and exact serving prefixes.**

- CPU inputs: fitted methods and queued normalized/wire action prefixes. Outputs: p50/p95 CPU times, valid successor seams, exact executed-action parity for π0.5 ten and GR00T fifteen/sixteen variants, resets/interleaving and partial commitments.
- Later encoder inputs: current policy on identical stored observations, same serving backend/batch; full, wrist, third, and one-view-plus-completion dispatches. Outputs: per-camera key parity, policy-input/output parity after completion, physical stage counts, measured latency and the owner-price ledger with assumptions identified.
- Decision: whether dynamic look-half is actually deployable and cheaper, especially GR00T/third-person. **New timing/key-parity profiling is required; historical logs cannot supply an unexecuted isolated tower.** The coordinator runs this after coding; no GPU work was done for this proposal.

## 7. Risks and open questions

- The robot can follow its own pose template after missing an object. Proprioceptive agreement is not object-state agreement, and no threshold repairs this observability gap. Retain bounded looks, transition looks and the existing stall signal.
- Frozen-fit candidate-LOEO is optimistic about representation fitting; held-out source states were also generated by five-control replanning. Even full refitting does not turn their residual distribution into a closed-loop guarantee for synthesized long chunks.
- Long-window support selects away terminal and difficult trajectory regions; both fixed-cohort and censoring counts are mandatory. All-sixteen support can be limited by a negligible-weight member. Preserving the exact kernel favors a look over dropping it; relaxing that would be a separate method.
- B-val has only one episode per task in each cell, and P3 A_r0 only two inits per task. Overlapping anchors are not independent samples. Stage-specific zeros and test-versus-validation differences need episode-level uncertainty and new closed-loop evidence.
- A weighted mean reference can lie between incompatible trajectories. Stage agreement and the absolute gate help, but their joint calibration is not yet measured. Rotation geometry and GR00T quantization can otherwise produce false alerts.
- The existing server only observes every five controls. A truly control-rate valve needs a client-side state check/request protocol and latency measurements; it cannot be claimed from `blind_step` alone.
- Camera sufficiency and camera price are separate questions. Single-view stored-key deletion can screen retrieval quality without proving a real encoder saving. Dynamic escalation may consume most of the theoretical gain.
- The safest profiling order is: P1's monitor and support accounting on unchanged A; separate stage-bounded continuation; separate camera restriction at A's commitment/call schedule; then their combination. E1 owns segmentation and E4 owns call allocation/budget control; this proposal only specifies their required interfaces.
