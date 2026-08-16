# TIER: Experience-Tiered Inference for Robot Policies — Paper Outline v2 (post-review)

> Thesis (one sentence): **The value of an experience library for robot policy serving lies in its index, not its payload** — retrieval similarity against the teacher's own rollout representations is a training-free, per-step routing *signal*, and the correct system built on it is tiered dispatch (replay / student / teacher), not cache replay.
>
> v2 incorporates a 4-reviewer adversarial pass (ML AC / robotics / statistics / systems). 32 findings; adjudication log at end.
> Page budget: 9 pp main text; ≤ 6 main-text floats + 1 related-work table; appendix ≤ 11 pp.
> Scope locks unchanged: depth-1 keys only; no history terms; Markov-inheritance theory reserved for paper 2; history negatives = 1 sentence + App. C table.

---

## 1. Introduction (≈1.25 pp text; Fig 1)

- P1: per-step VLA inference cost; LLM serving has per-query routing as standard practice; robotics has adjacent-but-different primitives: dual-system VLAs run both models always; DeeR-VLA adapts depth intra-model; interactive IL (SafeDAgger / EnsembleDAgger / ThriftyDAgger) switches policy↔expert per step **but to minimize training-time expert queries with learned/ensemble gates** — not deployment-time compute with a training-free signal.
- P2: the missing piece is the routing *signal*: per-step, training-free, closed-loop-valid competence estimate. (All "training-free" claims scoped to the **signal**; the system contains a distilled student and offline threshold calibration.)
- P3: insight — the deployed teacher's own rollouts, success-filtered and encoded in its own internal representation, index demonstrated competence; retrieval similarity estimates per-step whether a cheap executor suffices; the same index supports replay at near-duplicates.
- P4: TIER preview (three tiers on one index; cascaded keys; offline calibration under an SR constraint).
- P5: **three contributions** (key-space demoted to supporting analysis):
  1. **Method/system**: TIER — per-step, training-free-*signal* routing across separate policies (teacher/student/replay) unified on one index. Positioning: first to route between independent policies for deployment-time compute allocation on a compute–SR frontier (priority claim narrowed; interactive-IL lineage acknowledged).
  2. **Finding**: a controlled factorial substitution study showing a double dissociation — payload replaceable (hit→student improves over best-calibrated replay), index load-bearing (student collapses at index-flagged misses; effect isolated from replay-prefix damage via clean arms + interaction analysis). Revises the content-value assumption of retrieval-as-replacement **in the regime where the same experience suffices to distill a student**.
  3. **Frontier result**: TIER expands the achievable compute–SR frontier in the mid-compute range (claim scoped; no global-dominance wording), established against matched-**compute** and yoked matched-rate controls at pre-registered operating points.
- Hypothesis box: only prospective parts (X2 controls; signal-identity predictions). X1 presented as established finding motivating H, not as a test of H.
- Fig 1: system diagram + frontier teaser.

## 2. Related Work (≈0.75 pp text + comparison table float)

Four families + one added:
- Retrieval for robot policies (VINN, RT-Cache, Behavior Retrieval, DARP): cede library primitive; claim new role.
- **Interactive IL / per-step policy switching (SafeDAgger, EnsembleDAgger, ThriftyDAgger)** [NEW per AC-F3]: cede the switching primitive; differentiate objective (deployment compute vs training-time expert economy), signal (training-free retrieval vs learned gate/ensemble), evaluation (compute–SR frontier).
- Conditional computation & routing (FrugalGPT, RouteLLM; DeeR-VLA, MoLe-VLA; dual-system RoboDual/HiRT/GR00T-N1) + **token-level caching (VLA-Cache)** [NEW per Rob-F7]: one-line contrast — VLA-Cache reuses payload across frames; we show payload is the wrong half to bet on.
- Failure detection / UQ (Sentinel/STAC, SAFECAST, VLAConf, ARMADA, kNN-OOD): nearest signal relatives; alarm-level, route to humans; precise delta to our signal spelled out in §6.2 (reference set = success-filtered own-rollouts; key layer; consumed for dispatch not alarm).
- Semantic/approximate caching (GPTCache, NIRVANA, **IC-Cache** SOSP'25): origin analogy, open-loop. IC-Cache independently documents the payload problem in LLM serving (naive semantic-cache replay drops win rate 50%→18%) and responds with ICL augmentation + a per-request bandit router trained on user feedback — both mechanisms unavailable in closed-loop control (fixed-weight policies cannot be prompt-augmented; no free per-decision feedback). Convergent motivation, disjoint solution spaces.

## 3. Method (≈1.5 pp text)

### 3.1 Setup
a = g(φ(o)); student π_S; library L from success-filtered teacher rollouts; depth-1 query k(o_t); two thresholds → three-way dispatch. Success filter stated as design choice; **its effect is measured, not asserted** (forward pointer to X12).

### 3.2 Why the policy's own representation (supporting analysis, not a contribution bullet)
- Sufficiency view (`I(a*;o|k)` residual), one paragraph.
- **Remark 1** (demoted from Proposition per AC-F5/Rob-F8): Lipschitz composition intuition, one sentence + appendix sketch; the load-bearing evidence is empirical: X7 ε→δ curves **plus real counterexample pairs mined from logs (CLIP-close but action-far)** as the separation argument.
- Cost honesty: internal keys cost a partial teacher forward; cascade = operating point on signal-quality × query-cost; **cascade decision rate is a measured quantity (X11), and a savings-ceiling curve appears in §5.1**.
- **Why not a learned router** (anticipated objection; full treatment in Q&A Q1, experiment X14): supervised training of a router is not semantically available for per-step closed-loop routing — labels are counterfactual, pure-executor rollouts label the wrong visitation and the wrong continuation semantics, and the routing objective is Bellman-coupled to the router's own future decisions (an RL problem by definition). The only clean training route is online RL — which we actually run as the baseline (X14: three router variants, offline-MC warm start, generous interaction budgets), priced by an interaction-efficiency curve against our zero-training signal. Supporting asymmetries: fail-closed OOD direction (novel → dissimilar → teacher) vs. uncontrolled extrapolation of a learned router; multiplicative maintenance (a trained router binds to executor set × operating point × student × suite × model version; the index is rebuilt label-free from pipeline rollouts and tracks the teacher's representation automatically); and the cache tier needs retrieval at execution time regardless (the router outputs a class — the action chunk lives in the retrieved entry).

### 3.3 Two compressions of the same experience
Replay = nearest-neighbor zero-order hold; student = parametric interpolation. Framed as prediction-generating (predicts §6.1 dissociation and density behavior), with the **common-source caveat stated up front** and deferred to §6.2's identity experiments (X12) — not used as a shield.

### 3.4 Calibration protocol (pre-registration surface)
Offline on a dedicated calibration init split (init ledger in App. F: calibration inits ∩ eval inits = ∅, counts stated); thresholds AND all Table-1/X2 operating points fixed here before eval; replay baseline calibrated by the **same** procedure (symmetry clause); deterministic tie-breaks; no online adaptation.

## 4. Experimental Setup (≈0.4 pp text)

- Teacher Pi0.5; students ACT + SmolVLA (distilled) **+ demo-trained ACT arm** (X12c); LIBERO-Spatial + LIBERO-10; X8 second family (RoboCasa / SimplerEnv / MimicGen candidate) as **confirmatory set** — all analysis choices frozen on LIBERO before X8 runs.
- Roles: LIBERO-10 = primary discriminative battlefield (teacher 0.868, headroom); Spatial = ceiling-regime no-loss verification with **disclosed discordant-pair counts and power**.
- Compute metric: weights = measured per-invocation GPU-time (batch=1) on stated hardware; FLOPs as robustness duplicate; one frontier panel in raw GPU-seconds/step (no normalization).
- Statistics: paired inits/seeds; McNemar; equivalence via TOST with **δ=5pp primary / 3pp reported descriptively** (power analysis from pilot discordance rates in App. D); a **small pre-declared primary test family** (per suite: 2 operating points × 2 primary controls) under Holm; everything else descriptive; episode-level cluster bootstrap everywhere.

## 5. Main Results (≈1.2 pp text; Fig 2, Table 1)

### 5.1 Frontier [X4 + X11 + X12 arms]
- SR × measured compute Pareto: TIER sweep vs pure teacher, pure students, best-calibrated replay cache, **teacher-cheapening baselines** [NEW per Rob-F4]: (a) fewer flow-matching integration steps sweep, (b) longer chunk / lower query-rate sweep.
- **Savings-ceiling inset**: achievable compute reduction as a function of cascade short-circuit rate (X11), so the reader sees the key-cost floor explicitly [Sys-F1].
- Three-regime discussion (replay competitive only ultra-cheap — pointer to App. E density analysis; TIER mid-range; teacher at ceiling). Mixture-above-teacher effect (student@hit + teacher@miss > teacher alone) discussed openly as filtered-distillation gain; partition-attribution deferred to §5.2 controls.
- Table 1: pre-registered operating points × {SR, GPU-s/step, tier occupancy %, resident memory} [Sys-F7].

### 5.2 Partition vs rate vs compute: the control family [X2 — make-or-break, redesigned]
- **Primary control A — matched-total-compute random mixing** [Sys-F2]: teacher rate raised so total cost (incl. key build + search) equals TIER's.
- **Primary control B — yoked per-episode-rate random** [Stats-F3]: per paired init, random dispatch at TIER's realized per-episode teacher rate → isolates *per-step* information from episode-level budget allocation.
- Secondary: matched-global-rate random + multi-phase periodic; three-way mixtures at replay-bearing operating points (or X2 scoped to two-tier points, stated).
- Protocol pre-registered: nominal vs realized rates and compute tolerances tabulated; claim wording: superiority tested only at the pre-declared points, Holm-corrected; **pre-written fallback narrative**: if superiority holds only mid-range, thesis statement becomes "partition value concentrates in the mid-budget regime" [AC-F1].
- **Reversed-partition arm** (anti-routing at matched rate) as a one-row dramatic check [Stats-F1c].

## 6. Analysis: Where Does the Value Live? (≈1.5 pp text; Table 2, Fig 3, Fig 4)

### 6.1 Double dissociation, factorial + causal anatomy [X1 extended + X5 redesigned; Table 2]
- Full 2×2 factorial {hit: replay|student} × {miss: teacher|student} **+ (hit→teacher, miss→student) clean arm**; report two main effects + interaction on paired episodes — no diagonal-only deltas [Stats-F1, Rob-F1, AC-F6]. Headline decomposition: how much of the collapse is miss-slot vs replay-prefix vs interaction.
- Causal anatomy folded in (was §6.6): **state-reset closed-loop probes** (sim reset to miss/hit states harvested from teacher rollouts, matched on task progress; student rolls out closed-loop) + **takeover-window dose–response** (student takes over for k steps at miss onset, teacher resumes; recovery vs k) [Rob-F1, Stats-F7]; DAgger-style teacher relabel along student trajectories quantifies visitation-distribution confound; open-loop probes → App. E with multimodality caveat.

### 6.2 What does the index measure? Signal identity + competence estimation [X3 + X12; Fig 3]
- AUROC + calibration for predicting student failure vs training-free baselines (kNN-OOD feature distance, STAC-style action consistency, likelihood/entropy) + ensemble reference + **trained-router baseline** (small probe on the same internal features, trained on held-out labeled student rollouts; reported as a **label-efficiency curve** — AUROC vs number of labeled episodes, against the zero-label index signal; pre-registered stance: matching a label-hungry router with zero labels is a win, and if the router overtakes at high label counts the crossover itself is a result — deployment sits on the label-scarce side); label construction explicit; lead-time-stratified (≥k steps before failure; post-onset steps excluded/segregated); episode-level cluster bootstrap; identical labels for all baselines [Stats-F6].
- **Identity ablations** [X12; AC-F2, Stats-F2, Rob-F2]: (a) library success-filter {on, off, failure-only} at matched density → competence-vs-visitation, and the precise delta to kNN-OOD; (b) **decoupled batches** — student distilled on rollout batch A, library from disjoint batch B; (c) **demo-trained ACT** — does the index route a student it shares no data with? (d) teacher-failure-prediction control — generic difficulty vs student-specific competence; claims scoped by outcome.

### 6.3 Key space [X6 + X7 merged; Fig 4 single float]
- Signal AUROC by key source (internal vision-token / LLM-layer vs CLIP / DINOv2 vs proprio) **with a per-variant query-cost column** [Sys-F1c]; end-to-end frontiers per key source → App. E.
- ε→δ curves with **stratified pair sampling** (same-episode near/far, cross-episode same-init, cross-init; headline = cross-episode strata only) and action-distribution distance (energy distance, sampling protocol stated) [Stats-F8]; real CLIP-close/action-far counterexample pairs shown.

## 7. Limitations & Discussion (≈0.3 pp)

Sim-only; single-teacher (n=1) scope for key-space claims; same-task-family deployment regime; signal cost floor (cascade mitigates, X11 quantifies); student maintenance cost on teacher update (training-free applies to signal, not system); replay tier open-loop mechanics (run-length distribution reported, mid-replay similarity recheck/abort described, run-length bound); serving-level throughput/tail-latency left to systems follow-up (per-step compute is the claim; optional App. B mini-bench if X13 lands); one-sentence history disclosure.

## 8. Conclusion (≈0.1 pp)

---

## Float ledger (main text; max 6 + 1 table in §2)

Fig 1 system+teaser · Fig 2 frontier (2 panels: normalized + raw GPU-s) with ceiling inset · Table 1 operating points (SR/cost/occupancy/memory) · Table 2 factorial dissociation · Fig 3 signal AUROC+calibration · Fig 4 key-space (AUROC×cost + ε→δ) · §2 comparison table.

## Appendix plan (≤ 11 pp; nothing without a main-text pointer)

- A (0.5): Remark 1 sketch + assumptions.
- B (2.5–3): compute accounting — stage-split microbench (FLOPs + wall-clock, batch=1); per-key-variant build cost; cascade decision rates & amortization (X11); weight table + two-accounting robustness; normalized-vs-measured **reconciliation** per tier; server-side vs end-to-end latency split; resident-memory table; calibration episode budget; [optional X13 serving mini-bench: steps/s/GPU at N concurrency + latency CDF].
- C (0.5): history-augmented negatives (one table + 3 sentences).
- D (2): ε→δ full curves; calibration grids; **power analysis** (pilot discordance rates; TOST feasibility) ; pre-registration records (operating points, test family).
- E (2): per-suite/per-task breakdowns; SmolVLA arms; key-source end-to-end frontiers; replay density analysis; open-loop probes.
- F (1.5): implementation; library construction; distillation recipes; **init ledger** (calibration/eval/train disjointness proof); reproducibility.

---

## Consolidated experiment ledger v2

| ID | What | Section | Status | Priority |
|----|------|---------|--------|----------|
| X1 | Factorial substitution matrix + clean (T@hit,S@miss) arm + reversed-partition arm | §6.1 | partially exists — **2 new arms** | **P0** |
| X2 | Control family: matched-**compute** random (primary A) + yoked per-episode-rate (primary B) + global-rate random/periodic (secondary); pre-registered points | §5.2 | new | **P0 (make-or-break)** |
| X3 | Signal AUROC/calibration vs baselines; lead-time strata; cluster bootstrap; teacher-failure control | §6.2 | new | **P0** |
| X4 | Frontier sweep + teacher-cheapening baselines (integration-step & chunk-length sweeps) + raw-GPU-s panel | §5.1 | partially designed — **2 new baseline sweeps** | **P0** |
| X5 | Causal anatomy: state-reset probes + takeover dose–response + DAgger relabel | §6.1 | new (redesigned) | **P0** (was P1) |
| X11 | Cascade decision rate, amortized key cost, cascade-off frontier | §5.1/App.B | new | **P0** |
| X12 | Signal identity: filter {on/off/failure}, decoupled batches, demo-trained ACT, (with X3d) | §6.2 | new (a,d offline-cheap; b,c need training runs) | **P0(a,d) / P1(b,c)** |
| X6 | Key-source ablation + cost column | §6.3 | new (infra exists) | P1 |
| X7 | ε→δ stratified + counterexample mining | §6.3 | new, offline | P1 |
| X8 | Second benchmark family (confirmatory set; analyses frozen first) | §4 | new | **P1 (upgraded)** |
| X9 | Replay error vs library density | App.E | new, offline | P2 |
| X9b | Library **size** vs pure-replay closed-loop SR (no threshold; 6 tiers × 2 suites + 4 sensitivity arms = 8,000 ep). Complements X9, does not replace it: X9 asks how close a neighbour must be, X9b asks how much data replay needs. Bounds the **height** of the replay region and its data cost; conclusions scoped to this index | App.E, §5.1 | new, rollout; G1 approved | P1 |
| X10 | History negatives table | App.C | data exists | P2 |
| X13 | Serving mini-bench (throughput/GPU, latency CDF) | App.B | optional | P2 |
| X14 | Online-RL router duel: R_ts/R_tc/R_tsc in-loop RL routers (pre-library features only; retrieval scores masked; offline-MC warm start; batch on-policy + bundle hot-swap), interaction-efficiency curve + frozen head-to-head on A pool | §6.2/Q&A | new | **P0** |

---

## Q&A — anticipated reviewer questions (rebuttal bank)

**Q1. "Your contribution is essentially a router (an index-comparison router) — why not just train a small model as the router?"** (raised by advisor 2026-08-15; paper-text plan in §3.2, empirical response in X14)

1. **Supervised learning is not available for this problem — a three-layer structural argument.** (a) The label "would executor E succeed if it took over at state s" is counterfactual: it never occurs in natural data unless E is actually run from s. (b) Even pure-executor rollouts mislabel: broadcasting the episode outcome labels "E succeeds from states E itself reached, running solo to the end" — deployment invokes E at states reached by mixed prefixes, and the router may switch away afterwards; both the visitation and the continuation semantics are wrong. (c) Deepest: the value of choosing E at s depends on the router's own decisions at later states — the label definition is Bellman-coupled to the policy being learned. "Train a router with supervised learning" is semantically ill-posed; the feasible routes are biased offline MC, sim-only state-reset labeling (one rollout per label; the reset mechanism does not exist on hardware), iterative interactive training, or online RL. The contextual-bandit route (IC-Cache, SOSP'25) presupposes free per-decision feedback — an LLM-serving privilege absent in closed-loop control (per-step decisions, one delayed episode-level bit, every sample a physical rollout); IC-Cache itself rejects classifier routers on label cost in the label-cheapest domain.
2. **The only semantically correct route is online RL — so that is the baseline we actually run (X14).** Router = in-loop RL policy over executors (reward = success − λ·cost), on-policy batch training in sim on the legal (non-eval) init pool, warm-started from an offline biased-MC head (the steelman against "you trained the RL baseline poorly"), three variants matching our three modes ({T,S}, {T,C}, {T,S,C}), inputs restricted to the same pre-library model-internal features our keys see — retrieval scores and library contents are masked. Priced by an **interaction-efficiency curve**: frozen-policy deployed SR vs. cumulative training episodes, against our zero-training horizontal line. The multiplicative cost is structural: the trained router binds to (executor set × λ × student × suite × model version) and must retrain on any change; on real hardware its training procedure means an untrained router exploring on the fleet.
3. **Pre-registered outcomes — every branch publishable.** Router never catches up: the training-free signal wins outright. Router catches up after N interaction episodes: N is the price tag, re-paid on every configuration change the index is immune to. Router fails to converge despite warm start and generous budget: reported as-is — sparse 1-bit episode reward over ~200-step horizons is the structural reason. Auxiliary asymmetries that hold regardless of AUROC/SR: fail-closed OOD direction; label/interaction economics with the benchmark-given cap (~500 legal episodes, nine-tenths inside the student's training distribution); the cache tier's payload fetch needs retrieval at execution time in every arm. Bonus from the cited paper: IC-Cache measures naive semantic-cache replay dropping win rate 50%→18% — independent LLM-domain evidence for our payload claim; their fix (ICL prepending) does not exist for fixed-weight policies. Convergent motivation, disjoint solution spaces.

## Adjudication log (32 findings → clusters; Executor decisions)

- **A. −32.8pp attribution confound** (Rob-1, AC-6, Stats-1): **Accepted in full** — factorial analysis, clean arm, reversed-partition arm, X5 redesign (state-reset + dose–response + relabel), X5→P0, open-loop→appendix.
- **B. Library–student common source / kNN-OOD identity** (Rob-2, AC-2, Stats-2): **Accepted in full** — X12 identity suite (filter ablation, decoupled batches, demo-trained ACT, teacher-failure control); §3.3 caveat stated up front; claims scoped by outcome.
- **C. X2 control upgrades** (Sys-2, Stats-3, AC-1): **Accepted in full** — matched-compute primary, yoked per-episode-rate primary, protocol pre-registration, pre-declared operating points, claim rewording, pre-written fallback narrative.
- **D. Cost-accounting spine** (Sys-1,3,4,5,7,8; Rob-5): **Accepted in full** — X11 new; savings-ceiling curve in main text; measured GPU-time weights + raw-GPU-s panel; reconciliation; memory column; App.B → 2.5–3 pp.
- **E. Claims/wording** (AC-3,4,5; Rob-7,8; Sys-7b, AC-8): **Accepted in full** — "first" narrowed + interactive-IL lineage ceded; "dominates"→"expands frontier in range"; Prop→Remark + real counterexamples; contributions 4→3; hypothesis box prospective-only; "training-free signal"; "corrects"→"revises … in the regime where…".
- **F. Benchmarks/baselines** (Rob-3,4,6): **Accepted** — X8→P1 confirmatory; l10 primary battlefield, Spatial power-disclosed; teacher-cheapening baselines added; replay baseline symmetry clause; replay open-loop mechanics disclosed. *Partial*: Spatial stays in main frontier figure (not demoted out of main results).
- **G. Statistics rigor** (Stats-4,5,6,8): **Accepted with one modification** — pre-registration surface in §3.4/§4/App.D; δ=5pp primary; small primary family; AUROC label/cluster/lead-time protocol; ε→δ stratification. **Modified/rejected as stated**: Stats-4d cross-suite explore(Spatial)/confirm(l10) protocol — l10 X1 data already exists, freezing it would be retroactive fiction; the confirmatory role is assigned to X8 (untouched family) instead.
- **H. Page budget** (AC-7, Sys-8, owner directive): **Accepted** — §6 restructured 6→3 subsections; §6.5→App.E, §6.6 folded into §6.1; §2 table-compressed; float ledger capped at 6+1; appendix rebudgeted ≤11 pp with per-section caps.
- **Deferred**: Sys-6 full serving bench → optional X13 (claim demoted to per-step compute instead, per Sys-6 option b).
