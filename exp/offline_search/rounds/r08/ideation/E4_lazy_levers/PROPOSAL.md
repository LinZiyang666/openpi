# E4 — Look less / look half, studied systematically (R8 proposal)

Explorer E4 (Opus), R8. Scripts and small outputs are in `exp/offline_search/rounds/r08/ideation/E4_lazy_levers/`; per-anchor and per-pair CSVs are in `/tmp/r8_E4_lazy_levers/`. No servers, simulators, GPU or git were used. All Python ran on CPUs 26-29/70-73.

## 1. Summary

**What I read.** The R8 brief. R7 `SELECTION.md` and `ANALYSIS.md` in full, including Addenda A–C. The C1 follow code and hand-back (`c1_follow/methods.py`, `HANDBACK.md`), the C2 wrist hand-back, and the C4 toolkit README and telemetry keys. R7 E3's proposal and profile analysis. The stage API. I also looked at the formats of `r07_main` / `r07_profile_bval1` (decision JSONL, client per-step, input NPZ with raw 32768-d keys at about 15 MB per episode) and the library store. Finally I checked historical arms that act as look-less or look-half points: R4 wrist-every-decision π0.5 (`r04_b4w`), R4/R6 GR00T 15-control commitment (`r04_gblind/*tail2u`, `r06_frontier/*A15*`), and the R5 π0.5 library-size curve at 100/200/300 demos (`r05_demo_curve`).

**New evidence (two scripts, §5).**
1. **Cost of not looking, measured on A's own path** (`follow_vs_look.py`, 8 cells, one A replicate each, 5k–18k paired looks per cell). I rebuilt exactly the extra block that SF1/UF1 would have served. Rebuilding the anchor chunk from the logged rows/weights and the library reproduces the logged served actions to about 2e-7. I compared that block with what A actually served after looking again, from the same observation.
   - **The gripper gate does its job completely.** Where SF1 would extend, the follow block and the fresh look disagree on the gripper command at ≤ 0.1 % of anchors. At anchors that only UF1 would extend, they disagree at 3.5–21 %.
   - **Yet SF1 still loses 3–4 pp on sparse libraries.** So the sparse loss is not a gripper-timing error.
   - **Arm-motion disagreement is smaller on dense libraries.** At SF-granted anchors the median disagreement is .13/.15/.20/.18 action-σ on the sparse cells versus .10/.10/.17/.15 on the dense cells.
   - **A second extension block adds 30–40 % more disagreement on π0.5.** The median rises .145 → .193 (L10-50) and .109 → .152 (L10-500). On GR00T the median is flat, but on L10-50 the p90 rises .73 → .98.
   - **This disagreement is a weak predictor of which episodes the lever loses.** Episode-level early disagreement predicts lever losses at task-stratified AUROC .49–.66. It predicts A's own replicate failures somewhat better (.63–.79, with n ≤ 16). It marks hard episodes rather than lever-specific harm.
2. **GR00T twin divergence** (`twin_divergence.py`). GR00T's A is bit-deterministic, so an SF1/UF1 episode and A replicate 1 on the same (task, init) are identical until the lever first acts.
   - That first divergence is the lever's blind decision in 84–99.6 % of pairs, at median decision 2. At that decision the actions differ by only .10–.23 σ and the gripper never disagrees.
   - **The size of that first perturbation does not predict an outcome flip** (AUROC .43–.57).
   - The twins' robot states then drift apart. Median distance is .01–.02 σ one decision later, .05–.10 σ after 8 decisions (40 controls), .10–.19 σ after 16, and .18–.27 σ after 32 (p90 up to 1.5 σ on L10-50).
   - Flips become separable only 8–16 decisions after the perturbation (AUROC .52–.82).

**Surprises.**
- **The look-less lever acts like a small kick into a sensitive closed loop.** It is not a visibly wrong action. Outcome churn is large (SF1 flips 108/47/98/19 of 500 on GR00T L10-50/L10-500/Sp-50/Sp-500) and the harm is a *net bias* inside that churn (losses exceed gains on sparse libraries). Explaining it needs per-control simulator state for both twins, and a placebo arm that perturbs equally without changing information.
- **Look-half has a bigger untested variant than the stage-gated one R7 ran.** In C4's library screen, the wrist camera alone is within 0.00–0.02 action-σ of both cameras in every cell and stage, and gripper-mode error is the same. The third-person camera is worse (π0.5 +.03–.05). R4 once spent the wrist saving on looking every 5 controls instead of 10 (older retrieval). It scored .734/.820/.924/.974 at a measured-price IR of about .065, versus A's .714/.827/.837/.976 at .076. So the real axis is **camera × cadence at matched cost**, not "wrist only in easy stages".
- **One existing arm set is already a look-less dose point.** GR00T's 15-control commitment is today's A with budget 2 (`r04_gblind/*tail2u`): .570/.808/.864/.946 at IR ≈ .050, versus A .611/.830/.867/.964. It carries no telemetry.

**Stance.** Both levers are one decision: how to spend the vision budget, deciding *how often to look* and *how many cameras to use*. R8 should measure each lever's dose-response on a common cost axis. Use one randomized design for look-less, so its effect can be split by stage without confounding. Log a shadow look at every blind decision: it is cheap and gives the direct "information cost of not looking" on the lever's own path. Log per-control object state so twin divergence can be traced to a physical event.

## 2. Profile tools wanted

"Shadow look" means the vision encoder plus A's retrieval, logged but never served. All tools are offline CPU tools over debug data unless stated otherwise.

| Tool | Question it answers | Inputs | Outputs | Stage-segmentation decision it informs | Pri | CPU |
|---|---|---|---|---|---|---|
| `follow_vs_look` (prototype written) | How much information does each extension or blind block give up versus a look at that moment? | Closed loop: shadow-look fields (new). Open loop on A's path: existing logs + library. | Per blind decision: action disagreement (arm dims, σ-RMS), gripper-command flip, direction cosine, progress slip, top-demo re-identification, member spread. Split by stage/segment × block age × library size. Plus a **per-library-row "follow-gap" map**. | "Easy" defined as segments where follow ≈ look, measured rather than assumed from the gripper. The follow-gap map is a candidate segmentation feature for E1. | P0 | seconds/arm |
| `twin_divergence` (prototype written) | When and where does a lever trajectory leave its reference, and does that decide the outcome? | Lever arm + reference arm on the same pairs. Per-decision logs (existing) plus **per-control robot/object state (new)**. Needs determinism (GR00T now; π0.5 only with a deterministic encoder, verified by key hashes). | First divergence decision and lever action there. Robot-state and **object-pose** divergence curves. Physical divergence onset (first object-pose gap above a library-calibrated quantile). Stage at onset; outcome flip. | Which stages make a perturbation outcome-relevant: "hard" = segments where flip-deciding divergence begins. | P0 | ~1 min per arm pair |
| `exposure_hazard` | Causal effect of one extra blind block by stage and by age, from the randomized follow lottery (§4). | Logged coin, propensity, eligibility, stage; outcomes; shadow look. Reuses C4 `stage_value` IPW/HT machinery. | Per stage × age: randomized contrasts on next-look distance, drift, stall entry, valve statistic (high power); episode-SR score with ESS and task/init bootstrap (low power, reported honestly). | Whether a stage label predicts extension harm better than chance. Directly tests a segmentation against randomized evidence. | P0 | minutes |
| `camera_shadow` | At a given moment, does wrist-only (or third-only, or wrist + stale third) retrieval pick the same demos and actions as both cameras? | Per-look PCA-64 keys for both cameras (new, small). Shadow full retrieval at wrist-only looks (new). | Kernel overlap, action disagreement, gripper-event disagreement, stage-class agreement, distance ratio, by stage. Eligible share under a data-derived admission rule. | Camera by segment: where one camera is enough. | P0 (π0.5), P2 (GR00T keys only) | seconds/arm |
| `blind_drift` (control-rate tube) | During blind execution, how far do the robot **and the robot-to-object relation** drift from the demonstrations being followed? | Per-control client state (new) + **library object poses from a one-time sim replay of library demos (new, CPU)** + followed chains. | Proprioceptive and object-relative drift per control, by stage × age. Excursions between the 5-control checks. Valve lead time. | Replaces the proprioception-only valve with an object-aware signal. Gives E1 object-relative segments. | P0 (needs backfill) | minutes |
| `churn_decomposition` | How much of a lever's flip count is plain closed-loop sensitivity (any equal-quality perturbation) and how much is net bias? | Lever arms, placebo arms, A replicates. | Symmetric churn vs net loss per cell; net bias located by stage of divergence onset. | Whether stage gating should aim at "no perturbation" or at "no bias". | P1 | seconds |
| `density_curve` | At what library density does looking less stop costing success? | Lever/lottery arms at 50/100/200/300/500 demos; effective demos per kernel. | ΔSR and follow-gap vs effective-demo count; a candidate library-quality gate threshold. | Whether "easy" is a property of the stage or of library support. | P1 | seconds |
| `cost_clock` | Is a blind decision really free, and is a wrist look really cheaper, in owner IR and in wall time? | Per-decision stage-1 ms, method ms, queue wait, batch size, camera dispatch (partly new). | Owner IR from counts; measured-price IR; wall-time split. GR00T blind-queue diagnosis. | None directly; guards claimed savings. | P1 | seconds |

**One-time job requested: library object-pose backfill.** Replay every library demonstration in the simulator from its recorded initial state and executed actions (CPU, no policy). Record task-object poses and gripper–object contacts per control. Verify fidelity before use: success equals the recorded success, and the replayed robot state matches the recorded `rs` to ~1e-4 at every decision row. Without this, no drift or segmentation tool can refer to objects on the library side. E1 and E3 need it too.

## 3. Debug-mode data fields required (for this lens)

Estimated sizes assume L10 ≈ 60–75 decisions and ≈ 300–600 controls per episode.

| Field | Side | Why | Size | Pri |
|---|---|---|---|---|
| Decision kind (look / blind / call), age since last look, look_reason, all method reason codes (exists) | server/decision | Joins every lever event | ~50 B | P0 |
| Anchor kernel: 16 rows, weights, scores (exists) + library fingerprint | server/decision | Rebuilds follow blocks and stages offline. Member episode/step/success/stage are derived offline from rows, so they need not be logged. | ~200 B | P0 |
| Served 5-control block (exists) + **full synthesized chunk at looks** (H×7) | server/decision | GR00T chunk-tail follow; validates offline rebuilds | ~0.3–0.45 KB | P1 (rebuildable) |
| **Lever state**: eligibility per candidate E, drawn E, coin and propensity (lottery), follow source (chunk tail / successor heads), successor cursor rows (16 ints) at each blind decision, valve D_delta / D_abs / radius computed for *every* arm in shadow | server/decision | Randomized analysis; tube; exact audit | ~150 B | P0 |
| **Shadow look at every blind decision**: vision encode + A retrieval, not served. Rows / weights / scores / would-be block, d1, stage class. | server/decision | Closed-loop `follow_vs_look`: the direct information cost of not looking, on the lever's own path | ~0.4 KB; GPU ≈ 1 encode per blind decision (π0.5 ~62 ms, GR00T ~44 ms + queue) | P0 |
| **Per-camera PCA-64 keys at every look** (and every shadow look) + SHA of raw keys | server/decision | Camera shadows; determinism check; drift in representation space | 0.5 KB | P0 |
| Raw 32768-d keys | server | Only for new-representation research | 15 MB/episode today | P2, 5 % episode sample only |
| **Camera shadow retrievals at every look** (π0.5): wrist-only (72-D metric) and third-only kernels + blocks, CPU-side from the same keys. At wrist-only looks, shadow-encode the third camera (+.05 GPU) and log the full retrieval. | server/decision | `camera_shadow` | ~0.6 KB | P0 in look-half arms, P1 elsewhere |
| Actual camera dispatch, stage-1 calls, completions, owner cost (exists for SW; make universal) | server/decision | Cost truth | ~40 B | P0 |
| Timing: arrival→start queue wait, stage-1 ms, method ms, GPU batch size | server/decision | `cost_clock`; GR00T blind wall time | ~40 B | P1 |
| Shadow policy chunk with private RNG, at looks and at extension decisions (E2/E3 may require every decision) | server/decision | Is the follow block or the look closer to what the policy would do? | ~0.4 KB; full inference each | P1 for this lens |
| **Per-control proprio**: eef pos/quat, gripper qpos, joint pos, commanded action | client/control | Control-rate drift; twin divergence | ~0.1 KB/control | P0 |
| **Per-control task-object poses** (env `*_pos` / `*_quat` of task-relevant objects) and gripper–object contact flags / grasp state | client/control | Object-relative drift; physical divergence onset | ~0.2 KB/control | P0 |
| Look images, both cameras, JPEG q90 | client or server/look | Visual audit of wrist-only failures and divergence onsets | ~50 KB/look ≈ 1.8 MB/episode | P1 |
| Per-control low-res video | client | Human review of flips | ~10 MB/episode | P2, 10 % sample |
| Per episode: init-state hash, termination reason, success step, first-decision key hash, determinism flag | episode | Pairing and twins | <1 KB | P0 |

**Budget.** About 0.4 MB/episode without images and about 2.2 MB with look images. That is 0.2–1.1 GB per 500-episode arm, versus R6's superset at about 60 MB/episode.

**Not worth collecting for this lens:**
- Raw keys at every look. PCA-64 plus a hash suffices.
- Token tensors.
- Full MuJoCo contact / efc_force / cfrc_ext arrays and per-substep actuator commands.
- Restore snapshots (`restore_certified=false`).
- All diffusion-noise samples.

**Invariance requirement.** Shadow looks and camera shadows must not touch method state (anchor, plan, histories) or policy RNG. Prove this with a debug-on vs debug-off identity test on GR00T (deterministic): identical served actions for every decision on ≥ 20 episodes per arm type.

## 4. Arms wanted (500 test pairs each, debug on)

**Cells.** The spine is **LIBERO-10** (largest effects, longest horizon) on both models and both library sizes: π0.5 L10-50 / L10-500 and GR00T L10-50 / L10-500. Spatial is a later replication.

**Plain names used below.**
- "Follow lottery": at each look with structural support, a hashed coin picks 0, 1 or 2 extra blind blocks.
- "Wrist every 10 / every 5": wrist-only looks at A's cadence, or at every decision.
- "Shifted A": A with one extra look at the start, which shifts every later look by 5 controls.

| Group | Arm | Cells | What it answers | Pri |
|---|---|---|---|---|
| 2 (shared baseline) | **A, debug on** | 4 L10 cells | Reference for twins, placebo churn and camera shadows; shadow looks give A's own information cost. If group 2 already has A-debug, share it. | P0 (shared) |
| 4 look less | **Follow lottery**: E ∈ {0,1,2} with probability 1/3 each at structurally supported looks. No stage gate, no valve (both computed in shadow). | 4 L10 cells | Randomized dose-response of looking less by stage, age and library size. The SR of the arm gives one point at mean exposure. Replaces separate SF2/UF2 arms; conditioning on "SF would have granted" recovers SF-type extensions. | P0 |
| 4 look half | **Wrist every 10** (whole episode except first look; wrist metric; shadow full retrieval) | π0.5 L10-50, L10-500 | The ungated maximum of look-half. Measured-price IR ≈ .034 vs A .076. Compared with R7 SW, it isolates what the stage gate adds (the look-half analogue of UF vs SF). | P0 |
| 4 look half | **Wrist every 5** (no blind block) | π0.5 L10-50, L10-500 | Spends the wrist saving on cadence, IR ≈ .065 < A's .076. Tests R4's hint (+2 to +9 pp on sparse cells with old retrieval) on today's A. | P0 |
| 5 control | **Shifted A (placebo perturbation)** | GR00T L10-50, L10-500 | Same information and cost (+1 look/episode), different timing. Measures pure closed-loop churn on deterministic GR00T, so lever flips can be split into churn and bias. π0.5 uses A replicates instead. | P1 |
| 4 density | **Follow lottery at 100 / 200 / 300 demos** (π0.5 L10 libraries `demo100/200/300`; A references exist in `r05_demo_curve`: .748/.788/.806) | π0.5 L10 | Density threshold for safe following. Tests R7's proposal of a library-quality gate. Needs A/stage prefits for those libraries. | P1 |
| 4 look half | **Stage wrist (R7 SW), debug on** | π0.5 L10-50, L10-500 | Telemetry for the gated arm; second replicate for SR | P1 |
| 4 cadence | **A looking every 5 controls**, both cameras | π0.5 L10-50, GR00T L10-50 | Top of the cadence curve: what A's 10-control commitment already costs. Check first whether R4 has today's-A budget-0 arms. | P1 |
| 4 | Follow lottery, Spatial | π0.5 Sp-50, GR00T Sp-50 | Suite replication | P2 |
| 4 | GR00T follow by successor heads instead of chunk tail at age 2 | GR00T L10-50 | On A's path, GR00T chunk-tail disagreement at age 2 (.26) is about as large as successor heads two blocks later (.275). Is the chunk tail the worse source? | P2 |
| 4 | Wrist every 10 + one extra follow block (composed) | π0.5 L10-50, L10-500 | Interaction of the two levers | P2 |
| 4 | Third-person every 10; wrist + stale third key | π0.5 L10-50 | Camera control. Run the stale-third arm only if `camera_shadow` shows it beats the wrist metric. | P2 |

**Totals.** P0 is 8 new arms (+4 shared A). P1 is 9 more arms. Recommended: P0+P1 = **17 arms (+4 shared A), 10.5k episodes**. P2 adds 7 arms. Do not re-run SF1/UF1 at 500: R7's 16 arms stay as SR points, and the lottery supplies their mechanism with randomization.

**Time and storage.**
- R7 wall times per 500-episode arm with 22 workers on one lane: π0.5 L10 cache ≈ 13 min, GR00T L10 cache ≈ 23–29 min, pure-policy L10 17 / 32 min.
- With shadow looks only (this lens): about 1.3–1.5× → roughly 18 min (π0.5) and 35 min (GR00T) per arm per lane. P0+P1+A ≈ 21 arms ≈ 9.5 lane-hours ≈ **2.4 h on 4 lanes**.
- If E5's debug mode runs a shadow policy at every decision, cache arms cost about pure L5: ≈ 35 / 65 min → ≈ **4.5 h on 4 lanes**.
- Storage: **4–23 GB total**, without or with look images.

**Preregistration for the look-less arms.** Written before any R8 test episode:
- Primary: SR at mean exposure versus A.
- Randomized local contrasts (next-look distance, drift, stall entry) by stage class and age, with Bonferroni correction within a family.
- Episode-SR stage scores reported with ESS; no claim when intervals include 0.

## 5. Preliminary evidence (existing data)

Reproduce from the repo root with the CPU prefix `taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python`:
- `exp/offline_search/rounds/r08/ideation/E4_lazy_levers/follow_vs_look.py --cells all` (~2 min) → `fvl_summary.json`
- `-m exp.offline_search.rounds.r08.ideation.E4_lazy_levers.twin_divergence` (~40 s) → `twin_summary.json`

Sources:
- A replicate 1 per cell (R6 identities via `r06/analysis_scripts/common.ab_arms`).
- SF1/UF1 from `r07_main`.
- Frozen stage tables embedded in `r07_main/fits/r7_<cell>_SF1.pkl`.
- The library store.

Stage classes follow A2's vocabulary: interior / event / mixed / unknown. Disagreement is measured over the 5 controls × 6 arm dims, in library action-σ.

### 5.1 Follow block vs fresh look (open loop on A's path; same observation)

| Cell | Paired looks | Supported / SF-granted | Median disagreement: SF-granted / UF-only | Gripper flip: SF-granted / UF-only | Median by class int / evt / mix / unk | Disagreement ÷ member spread (median) |
|---|---|---|---|---|---|---|
| π0.5 L10-50 | 16,318 | .81 / .48 | .130 / .207 | .001 / .117 | .117 / .239 / .229 / – | .41 |
| π0.5 L10-500 | 14,400 | .94 / .30 | .099 / .118 | .001 / .066 | .099 / .139 / .225 / .103 | .33 |
| π0.5 Sp-50 | 5,803 | .73 / .49 | .152 / .201 | .000 / .083 | .147 / .174 / .242 / – | .38 |
| π0.5 Sp-500 | 4,949 | .92 / .68 | .099 / .149 | .001 / .055 | .097 / .160 / .135 / .150 | .30 |
| GR00T L10-50 | 17,718 | .82 / .37 | .200 / .332 | .001 / .209 | .184 / .305 / .369 / – | .58 |
| GR00T L10-500 | 14,439 | .93 / .22 | .169 / .208 | .001 / .164 | .170 / .202 / .274 / .198 | .54 |
| GR00T Sp-50 | 5,877 | .73 / .51 | .182 / .226 | .001 / .101 | .170 / .222 / .263 / – | .45 |
| GR00T Sp-500 | 5,104 | .93 / .46 | .151 / .178 | .000 / .035 | .150 / .187 / .229 / .170 | .42 |

**Age dose, same anchors with support through 4 blocks.** Median disagreement at 2 blocks → 4 blocks after the look, with p90 in parentheses:
- π0.5: L10-50 .145 → .193 (.50 → .77); L10-500 .109 → .152 (.32 → .45); Sp-50 .161 → .194; Sp-500 .109 → .151.
- GR00T: L10-50 .259 → .275 (.73 → .98); L10-500 .195 → .196; Sp-50 .193 → .181; Sp-500 .158 → .162.
- π0.5 gripper flips on L10 rise 4.6 → 7.5 % (L10-50) and 4.4 → 6.4 % (L10-500).

**Outcome link.** Among A-majority successes, the mean disagreement over each episode's first 12 looks predicts a lever loss only weakly:
- SF1: task-stratified AUROC .49–.64.
- UF1: .49–.66.
- Gripper-flip rate for UF1 losses: .45–.58.
- The same statistic predicts A's own replicate-1 failures at .63–.79 (n ≤ 16).

**Reading.**
- The gripper-stage gate removes gripper disagreements completely and halves arm-motion disagreement relative to UF-only anchors.
- A fresh look lands well inside the spread of the 16 followed demos (ratio .30–.58).
- Dense libraries follow more faithfully.
- None of this identifies the episodes SF loses. Open-loop measures on A's path cannot explain the closed-loop loss, which is why the shadow look must run on the lever arm itself.

### 5.2 GR00T twins (lever arm vs A replicate 1, bit-identical until the lever acts)

| Cell / lever | First divergence at the lever's blind decision | Median decision | Flips (A→lever loss) | Median d0 | d0 AUROC for flip | Median state gap after 1 / 4 / 8 / 16 / 32 decisions | State-gap AUROC for flip at 8 / 16 |
|---|---|---|---|---|---|---|---|
| L10-50 SF1 | 98.8 % | 2 | 108 (61) | .227 | .48 | .019 / .037 / .056 / .119 / .199 | .58 / .60 |
| L10-50 UF1 | 99.2 % | 2 | 125 (73) | .227 | .51 | .019 / .037 / .061 / .126 / .223 | .59 / .53 |
| L10-500 SF1 | 84.5 % (457 diverged) | 6 | 47 (22) | .145 | .43 | .011 / .030 / .048 / .099 / .176 | .52 / .71 |
| L10-500 UF1 | 98.2 % | 2 | 93 (53) | .233 | .50 | .021 / .035 / .054 / .124 / .190 | .53 / .59 |
| Sp-50 SF1 | 99.2 % | 2 | 98 (59) | .107 | .47 | .016 / .050 / .098 / .171 / .270 | .61 / .60 |
| Sp-500 SF1 | 98.2 % (455 diverged) | 2 | 19 (12) | .116 | .57 | .017 / .043 / .102 / .136 / .254 | .68 / .82 |

**Reading.** The perturbation is small and never a gripper error. Whether it flips the outcome is decided 40–80 controls later. The analysis tool must therefore locate the *physical* divergence onset (objects, contacts) and compare it with a placebo perturbation. The size of the lever's action says nothing about the outcome.

### 5.3 Look half (C4 library screen, already computed)

Five-control head RMS for both cameras / wrist / third person, then gripper-mode error for both / wrist:

| Cell | Head RMS | Gripper-mode error |
|---|---|---|
| π0.5 L10-50 | .490 / .496 / .528 | .053 / .046 |
| π0.5 L10-500 | .337 / .352 / .390 | .028 / .028 |
| π0.5 Sp-50 | .546 / .544 / .588 | – |
| π0.5 Sp-500 | .355 / .367 / .400 | – |
| GR00T L10-50 | .576 / .581 / .602 | – |
| GR00T L10-500 | .428 / .442 / .455 | – |
| GR00T Sp-50 | .596 / .616 / .630 | – |
| GR00T Sp-500 | .403 / .421 / .422 | – |

- In interior and event stages alike, wrist-only stays within about .02 of both cameras, and gripper-mode error is the same.
- The wrist gap is slightly larger on dense libraries (+.012–.018) than sparse (−.002 to +.02).
- Combined with R4's wrist-every-decision results, this motivates the two ungated wrist arms. The measured wrist price is .0646 (R7 Addendum A).

## 6. Risks and open questions

1. **Stage-level SR effects may stay unresolved.** R7 could not resolve call value by stage from randomized data. The lottery's per-stage SR scores will likely be wide. Its power lies in local mechanistic contrasts (drift, next-look distance, stall entry), which should be preregistered as the primary stage evidence.
2. **Twin analysis on π0.5 needs a deterministic encoder.** Today π0.5 A replicates differ at the first step because batched bf16 GEMM shapes vary. Debug arms need B = 1 or deterministic kernels, verified by the logged key hash, or π0.5 twins are unavailable. Determinism must not change the actions, only remove their noise. Who owns this decision is open (E5 / coordinator).
3. **Shadow work must be behaviour-neutral.** Shadow looks and camera shadows must be invisible to method state and RNG. On GR00T they add GPU load, which already stretches blind wall time (median 0.08–1.5 s queueing in R7). Wall time changes; actions must not. This needs the identity test in §3.
4. **The library backfill depends on replay fidelity.** If LIBERO replay of library demos is not exact (contacts), object-relative drift is unusable. Mitigation: accept only episodes whose replayed success and `rs` match.
5. **The shifted-A placebo** costs one extra look and changes timing slightly. Its expected SR may not equal A exactly; report it with the A replicates.
6. **The wrist arms depend on today's A wrist metric and the per-request encoder path.** That path prices full looks 9 % above the stock path (Addendum A). Price both, and route full looks through the stock path if possible. GR00T look-half needs a separate feasibility probe (one-camera key parity and latency) before any arm.
7. **The intermediate-library arms need prefits** (A fit, stage table, valve) on `demo100/200/300`. They are π0.5 only; GR00T has no intermediate libraries.
8. **Open:** should the lottery include E = 3 (π0.5 age 4, where disagreement p90 reaches .77 on L10-50)? I suggest keeping {0,1,2} for safety and adding E = 3 only if the profile shows ≥ 5 % support.

## Files

- `/home/weiland/projects/openpi/exp/offline_search/rounds/r08/ideation/E4_lazy_levers/follow_vs_look.py`
- `/home/weiland/projects/openpi/exp/offline_search/rounds/r08/ideation/E4_lazy_levers/twin_divergence.py`
- `/home/weiland/projects/openpi/exp/offline_search/rounds/r08/ideation/E4_lazy_levers/fvl_summary.json`
- `/home/weiland/projects/openpi/exp/offline_search/rounds/r08/ideation/E4_lazy_levers/twin_summary.json`
- `/tmp/r8_E4_lazy_levers/fvl_<cell>.csv.gz`, `fvl_ep_<cell>.csv.gz`, `twin_<cell>_<lever>.csv.gz` (per-anchor / per-pair detail)
