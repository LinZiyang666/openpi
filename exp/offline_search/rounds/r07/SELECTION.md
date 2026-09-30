# R7 method selection — stage-level allocation

Written 2026-09-30 01:0x CDT by the coordinator after the five exploration reports (`ideation/E1..E5/PROPOSAL.md`).
Pause point 1 is waived by the owner. Terminology (owner): **atomic levers** (fewer calls, look less, look half),
**signals** (state deviation, stall, stage label), **allocation policy** (uniform → task → stage → event).

## 1. What the explorers agree on

- **Stages that transfer are gripper-event stages.** Library-only two-mode segmentation of the executed gripper command
  gives 4 macro stages on LIBERO-10 and 2 on Spatial, consistent across π0.5 and GR00T (E1). A kinematic "difficulty"
  score does **not** transfer (hard/other risk ratio 2.67 → .57 across cells; boundary tightness AUROC < .5 in 7/8).
- **Where a call is worth more cannot be resolved by stage** in existing randomized data: coarse-phase reallocation of
  the R6 uniform lottery, all 8 cells' intervals include 0 (E4); stage/phase call effects, none survives a 4-test family
  (E2). Consistent with R5 Q3 and R6 (random = predicted-disagreement placement).
- **Where a look is worth less can be resolved**: stable-gripper interiors have low cache–policy disagreement; a free
  proprioceptive deviation check fires on 2–4 % of 20-control stable-motion windows versus 39–45 % of gripper-change
  windows (E3, π0.5 L10). 43.5 % of B-val anchors pass an all-16-neighbour structural screen for one extra block (E5).
  The vision floor is the dominant cost on dense libraries (A ≈ .076 of B's .12–.19).
- **Easy = stable-gripper stage interior with normal state; hard = gripper transitions, state deviation, stall.**
  Laziness is applied only in easy stages; hard stages keep today's cadence, cameras and calls.

## 2. Methods selected for R7

All methods keep A's retrieval (per-camera PCA-64 + state → per-task action-supervised Mahalanobis → top-16 kernel
synthesis) and A's 10-control commitment unless stated. No new model, no task/suite names, every number from a
documented library rule.

| ID | Name | Lever | Allocation | Family |
|---|---|---|---|---|
| **SF** | Stage-bounded follow | look less | stage (+ event veto) | C1 |
| **UF** | Uniform follow (control for SF) | look less | uniform | C1 |
| **SW** | Stage wrist (π0.5 only) | look half | stage | C2 |
| **CU** | Uniform calls + calibrated stall (owner's simplified C) | fewer calls | uniform + event | C3 |
| **CT** | Stage-tilted calls + calibrated stall | fewer calls | stage + event | C3 |
| **SA** | Full stage allocator = CT or CU calls + SF (+ SW on π0.5) | all three | stage | C1 composes after profiling |

### 2.1 SF — stage-bounded follow (look less; primary R7 method)

At a vision anchor A retrieves as usual (rows r[16], weights w[16]) and commits 10 controls. Before the anchor's
second blind decision would hand back to vision, SF may grant **one extra 5-control block** (cap `E=1`; profile also
`E=2`) if and only if:
1. **Structural support:** all 16 members have valid same-episode successor chains covering the extra block(s)
   (π0.5: rows `next²(r_i)`… first five actions; GR00T: the anchor's own 16-step chunk controls 10:15 for E=1, successor
   heads beyond that). Never clamp a terminal row, never renormalize or drop members.
2. **Stage interior:** every member's library gripper mode is unchanged through the extension and unanimous across
   members (no gripper event inside the extension). Stage labels come from the frozen library table (§3).
3. **State valve (free signal):** at each 5-control blind check, the normalized relative displacement
   `D_delta = RMS[S⁻¹((x_now − x_anchor) − (μ(h) − μ(0)))]` over valid proprioceptive dims (μ = weight-mixed library
   state along the members' chains, S = successful-library per-coordinate scale) is inside the radius = p95 of the
   same statistic in leave-one-episode-out library replay over the reference 10-control commitment (row-weighted and
   episode-weighted versions both computed; row-weighted is the default). Outside → `LookReason`.
A LOOK is a normal anchor (retrieval + judge); it is never automatically a call. Policy chunks are never extended.

**UF** = SF without conditions 2 and 3 (extend whenever structural support exists). It tells whether the stage/valve
signals add anything beyond the lever itself.

### 2.2 SW — stage wrist (look half; π0.5 only)

In an easy stage (as in SF condition 2, evaluated for the *next* anchor from the current kernel and gripper mode) the
next anchor encodes **only the wrist camera** and retrieves with a wrist-only metric (64 wrist PCA + state, per-task
action-supervised Mahalanobis refit in its own 72-D space, same kernel rule). Hard/unknown stages, episode start and
any unresolved stage use both cameras. On a MISS the missing camera is completed before the unmodified policy runs
(R4 K3 contract). Camera choice must be made **before stage 1** (a post-encoder deletion saves nothing). Cost: wrist look
.055198, completion on a wrist-origin MISS +.049890 (R4 proportional-latency assumption, labelled). GR00T has no
validated one-camera encoder path: lever unavailable there (portability checklist, E5 §5).

### 2.3 CU / CT — calls

CU is R6's `CalibratedRescue` with `placement='uniform'` + calibrated stall (owner decision 2, 2026-09-29): the R6
controller configuration that was never run as such. CT keeps the same budget solve and stall but multiplies the per-
anchor call weight by a stage tilt: event neighbourhoods (within one library row of a gripper-mode change) weight `1/h`
where `h` is the task's library event-neighbourhood occupancy, interiors weight 1 (E1 P1; nonzero interior calls), and
the **first anchor of each entry into high state deviation** (current-state residual above its library-LOEO p75,
latched until return) weight `1/h_dev` likewise (E2 P1). λ is re-solved so realized IR matches ρ.

## 3. Shared stage table (frozen, library-only) — API owned by C1

`exp/offline_search/rounds/r07/stages/stages.py` (C1 lands this first; C2, C3, C4 import it):
- `StageTable.fit(library, *, manifest) -> StageTable` on successful library episodes: gripper command per row = median
  of the row's executed head gripper channel (manifest-valid channel); deterministic two-means → two modes (polarity
  learned, not assumed); 3-row majority smoothing of isolated reversals; per row: `mode`, `event_near` (within one row of
  a mode change), `rows_to_event` (rows until next change or episode end), `stage_run` (run index), successor validity,
  and per-task `event_occupancy h`. Failed-episode rows get `mode=-1` (unknown). Save/load with content fingerprint.
- `StageTable.state_scale`, and the LOEO valve radius / deviation p75 tables used by SF/CT (fit here once).
- `online(rows, weights, cmd_mode=None) -> dict`: weighted mode mass, unanimity, event mass, min rows-to-event over the
  members, unknown mass. No elapsed-fraction-of-episode, no task names.
E1's `ideation/E1_kinematic_stages/analyze_stages.py` is the reference for the segmentation rules.

## 4. Profile phase (before any test episode)

Profiling uses **non-test B-val initial states** (the pool behind R6's calibration recordings; ≤ 2 inits per task,
20 episodes per cell per variant) and full decision telemetry. Variants profiled: SF(E=1), SF(E=2), UF(E=1) on 8 cells;
SW on 4 π0.5 cells; CU and CT at ρ = .30 on the 4 sparse cells. Offline profile tools (C4) run on the library, B-val,
P3 and R6 logs. Then each explorer is resumed on its own thread to analyze the results for its proposals and recommend
keep / drop / which cap. The coordinator freezes the full-run arm list from those analyses and the kill criteria below,
**before** any test episode of an R7 arm.

Kill at profiling (per variant): implementation parity failure (A-identity when the lever is off); extension eligibility
< 5 % of anchors; realized IR saving < .005 versus A on the same B-val episodes; SW key/policy-input parity failure;
CT/CU realized IR outside ρ ± .03.

## 5. Full evaluation (test pairs, 500 per arm) and preregistered acceptance

Episode set: the same 500 (task, init) pairs as every prior arm. Pairing and statistics as R6 (task-stratified init
bootstrap 10,000 draws, seed 20260930; exact McNemar for single runs; A and B as three-replicate means).

Planned arms (subject to §4 pruning): SF ×8 cells; UF ×8; SW ×4 (π0.5); SF+SW ×4 (π0.5); CU(.30) ×4 sparse;
CT(.30) ×4 sparse; SA(.30) ×4 sparse.

Acceptance, fixed now:
1. **SF saves vision without SR loss:** pooled 8-cell SR(SF) − SR(A) has 95 % lower bound > −1.0 pp **and** realized IR
   lower than A in ≥ 6/8 cells. Otherwise not supported.
2. **The stage signal matters:** pooled SR(SF) − SR(UF) lower bound > 0, or SF has no significant loss versus A while UF
   has a pooled lower bound < −1 pp. Otherwise the lever works (or not) without the stage signal.
3. **SW saves vision without SR loss (π0.5):** pooled 4-cell SR(SW) − SR(A) lower bound > −1.5 pp and IR lower in 4/4.
4. **Stage-tilted calls beat uniform calls:** pooled 4-cell SR(CT) − SR(CU) lower bound > 0 at |ΔIR| ≤ .015.
   Otherwise stage allocation of calls is not supported (expected from existing data; reported either way).
5. **Frontier:** report each cell's R7 points on the R6 frontier; a point "moves the frontier" only if it is
   non-dominated by R6 points at single-run point estimates, with the CP NI bound shown. No claim of certified NI.

## 6. Coding families (4 × gpt-6.1-sol xhigh)

| Coder | Owns | Delivers |
|---|---|---|
| C1 | `rounds/r07/stages/**`, `rounds/r07/c1_follow/**` | stage table (first), SF/UF methods for both models, arm specs, A-identity tests |
| C2 | `rounds/r07/c2_wrist/**`, `closed_loop/plugin.py`, `closed_loop/stage_overrides.py` (atomic installs, byte-identical when flags absent) | per-request pre-stage-1 camera selection, wrist-only retrieval fit for today's A, SW method, parity tests |
| C3 | `rounds/r07/c3_calls/**` | CU/CT controllers (reusing R6 `method_c` + stall), stage tilt, budget calibration on B-val recordings, arm specs |
| C4 | `rounds/r07/c4_profile/**` | profile toolkit (offline + closed-loop profile analyzer), B-val profile manifests and launch recipe |

## 7. Not claimed

- No stage is certified safe; the valve is an empirical monitor with no guarantee; proprioception cannot see objects.
- Existing randomized data do not identify stage call value; CT is a test, not an expectation.
- The method was designed after seeing test-init outcomes of earlier rounds; R7 profiling uses non-test inits only.

## 8. Freeze record

**Freeze 1 — 2026-09-30 02:4x CDT, before any R7 test episode.** SF1 ×8, UF1 ×8 and CU(.30) ×4 enter the full
evaluation unconditionally: SF1 is the primary method and UF1 its preregistered control (acceptance rules 1–2), and CU
is the owner's simplified C (decision 2, 2026-09-29), the required comparator for rule 4. Their profile gates were
checked on the eight plumbing arms (L10-50, both models): no policy call on A/SF/UF; SF1 realized IR .0654 / .0649
versus paired A .0763 / .0743 (saving ≥ .005); grants present; successor heads executed on π0.5. The remaining profile
arms are still running; any of these 20 arms whose own profile later fails §4 gates is withdrawn and recorded here.
SF2 (cap choice), SW, SF+SW, CT and SA are frozen only after the explorers' profile analyses (Freeze 2).

**Freeze 2 — 2026-09-30 03:2x CDT, before any test episode of these arms.** From the explorers' profile analyses
(`ideation/E{1..5}/PROFILE_ANALYSIS.md`; E1 arrived at 03:3x after the freeze and concurs on every point), all concur:
- add **SW ×4** (π0.5) and **CT(.30) ×4** to the full evaluation; with Freeze 1 the evaluation is **28 arms**;
- **SF cap = 1**; **SF2 dropped** (mean extra IR saving ≈ .0045 over SF1, descriptive successes 130 → 125 of 160);
- **SF+SW deferred** (the composed arm was not profiled), **SA not built this round**.
All 44 profile arms passed §4 gates (IR saving ≥ .005 versus paired A for every SF/UF/SW arm; CU/CT realized IR
.293–.318). Profile caveats carried into analysis: SF1's early first-blind valve LOOKs (8 on B-val) are a documented
identity exception; owner IR for SW uses the wrist price table (`c2_wrist/owner_costs.json`), not the ledger's
two-camera pricing; π0.5 L10-50 CT spent +.019 IR over CU on profile (rule 4 requires |ΔIR| ≤ .015 on test).

**Clarification of rule 4 — 2026-09-30 03:5x CDT, before any CT test episode.** "At |ΔIR| ≤ .015" is read **per cell**
(each of the four CT−CU pairs must have |ΔIR| ≤ .015), with IR = owner cost pooled over decisions
(`.152/.148·V/N + .848/.852·M/N` from the arm's decision counts), exactly as R6's `ops/c_validation.py` checked its
matched-IR clause. The pooled four-cell mean ΔIR and the per-episode-mean IR are reported alongside. If a cell misses the
IR match, the rule's SR verdict is reported as "IR not matched in that cell" rather than silently accepted.
**Completion markers:** an evaluation arm is complete with either `state/<arm>.DONE` or `state/<arm>.manifest_*.DONE`;
the SF/UF and SW rows carry no manifest field and ran the full 10 × 50 Cartesian set, verified identical to the eval500 pairs
and init-pool hash of the A/B references (A2 phase-1 audit).

## 9. R7 completion checks (preregistered 2026-09-30 06:2x CDT, before any of these outcomes; results in ANALYSIS Addenda A–C)

The owner's goal ("不做完不停") asks for R7 to be finished, not paused. Two R7 conclusions are weaker than their verdicts
suggest: rule 4 rests mostly on one cell, and rule 3's saving rests on an assumed price. These checks complete R7; they
add no new method.
1. **Measured wrist price (done 06:2x, CPU+GPU measurement, not an SR outcome):** `analysis_scripts/a3_wrist_latency.py`.
2. **Rule-4 replicate:** CU(.30) and CT(.30) rerun on the four sparse cells with a new lottery seed (`random_seed`
   26092904; calibration unchanged). Acceptance: pooled over the two replicates (8 CT−CU pairs, each cell's two
   replicates averaged per (task, init) before the task-stratified bootstrap), CT−CU lower bound > 0 with every
   replicate pair's |ΔIR| ≤ .015. The first replicate's verdict (rule 4) stands as reported either way.
3. **Rule 4 on the dense (500-episode) libraries:** CU and CT at ρ = .18 (R6's dense operating point), calibrated on the
   existing non-test B-val recordings by the same C3 recipe, profiled on 20 non-test episodes per arm (gates as §4:
   realized IR within ρ ± .03), then 500 test pairs. Acceptance: pooled 4-cell CT−CU lower bound > 0 with per-cell
   |ΔIR| ≤ .015. Reported separately from the sparse verdict; no pooling across library sizes.

**Freeze 3 — 2026-09-30 07:0x CDT, before any dense CU/CT test episode.** Dense CU18/CT18 profile gates (realized IR in
ρ ± .03, 20 non-test episodes per arm): π0.5 L10-500 .183 / .176, π0.5 Sp-500 .183 / .170, GR00T L10-500 .187 / .181 —
pass; these six evaluation arms enter the queue. GR00T Sp-500 (.196 / .187, 07:07) passed the same gate and entered the queue; all 8 dense profile arms passed.
