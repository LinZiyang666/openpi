# R6 analysis — Commit-Cache (A), guard-triggered rescue (B), the owner's three questions, and configuration C

Snapshot: **2026-09-29 13:4x CDT**, after the last C-validation arm (13:22) and the final frontier refresh (13:26). This report follows [the R6 brief](ANALYSIS_BRIEF.md) and extends [R5](../r05/ANALYSIS.md). It builds on [PAPER_AB.md](PAPER_AB.md), [ABLATIONS.md](ABLATIONS.md), [SELECTION.md](SELECTION.md) §1–§7b, [C_VALIDATION.md](C_VALIDATION.md), [frontier_final/](frontier_final/), the three preregistered finals ([Q1](ideation_Q1/FINAL.md), [Q2](ideation_Q2/FINAL.md), [Q3](ideation_Q3/FINAL.md)), the [C controller hand-back](ideation_Q1/method_c/HANDBACK.md) and the [stall hand-back](ideation_Q3/stall/HANDBACK.md). Every number below was recomputed from the arms' own `summary.json` (cost ledger) and accepted client-journal records, or read back from a frozen preregistered output file that the report names. The coordinator ledger (`logs/offline_search_exploration.log.md` §10, 2026-09-28/29) was read for context only.

**Result.** The data-quality repair (316 exception-terminated episodes in 33 R6 arms, none anywhere else) leaves the paper headline intact and slightly stronger. In three paired replicates, B beats A by +11.3 / +11.4 pp on both LIBERO-10-50 cells, +7.3 pp on π0.5 Spatial-50 and +4–5 pp on both LIBERO-10-500 cells. B is flat on the dense Spatial-500 libraries and on GR00T Spatial-50. It costs 1.5–3.0× A's owner IR (.12–.22 vs .074–.078), and no replicate differs from its siblings by more than 2.4 pp.

The learned metric is essential on LIBERO-10 (−7.8 to −13.4 pp without it). Direct-token PCA is not worse than pooled PCA, so the earlier Spatial-500 claim is withdrawn and pooling is a cost choice. B's gain runs almost entirely through the policy takeover that the no-progress guard triggers. That takeover helps the sparse long-task libraries and hurts GR00T Spatial-50.

**The three preregistered pilots could not decide SR questions.** The calibration split held one initial state per task, so the within-task variance was not estimable. Q1 did select R, a library leave-one-episode-out residual that predicts cache–policy disagreement.

**Configuration C made spending predictable.** Realized IR landed within ρ ± .02 in 26 of 28 arms, calibrated only from the library and ≤ 10 non-test recordings. The rest of C was weaker:
- its R placement did not beat uniform placement (pooled −0.2 pp);
- its calibrated stall trigger adds about +1 pp at matched budget and turns the GR00T Spatial-50 harm into a gain;
- at IR ≈ .45 it loses to the task-level risk lottery in two of three cells;
- its calls hurt on dense GR00T Spatial-500 (−3.0 pp vs A).

**No configuration cheaper than pure inference passes simultaneous 2 pp noninferiority in any cell.** Single-run point estimates reach pure L=10 SR at IR .05–.45 in seven of eight cells; π0.5 LIBERO-10-50 is the exception. Nominal 2 pp NI below the reference's cost exists only on the dense Spatial cells, and against the weaker L=5 reference on π0.5 LIBERO-10. The missing piece is a library-level *no-call* gate.

## Provenance, conventions, reproduction

Let **R** = `/home/weiland/trace_runs/os_closed_loop`. The shorthand `run:arm` (or `run/arm`) always means `R/run/runs/arm/summary.json` plus `R/run/runs/arm/client/journal.jsonl`. The outcome of a (task, init) pair is its accepted terminal journal record (status `done`/`failed`, no error). Every comparable arm has exactly the 500 pairs tasks 0–9 × inits 0–49, and its summary success count equals its journal. The census contains 593 arms (including smoke and pilot runs) with 161,382 terminal records.

**Owner IR** is computed from the ledger counts, v = V/N and m = M/N:
- π0.5: `(.152 v + .848 m)·5/L`;
- GR00T: `(.148 v + .852 m)·5/L`;
- blind and policy-tail slots cost 0;
- pure L=10 costs .5 and pure L=5 costs 1.0.

Summary "measured" or eager IR is never used. Wrist-key arms (R5 Q6) are charged full-camera owner prices here, as in `frontier_final`.

**Statistics.**
- Contrasts are *X − Y* in percentage points.
- Intervals are 95% task-stratified init bootstraps: the 50 inits are resampled within each task, with 10,000 draws and seed 20260929. A fresh generator is used per contrast, so each interval is reproducible on its own. The exception is the §4 verdict table, which is regenerated verbatim by `ops/c_validation.py` with its single sequential generator.
- A and B replicates are averaged within each pair before resampling, so three replicates are still 500 clusters, not 1,500.
- Single-run pairs use the exact two-sided McNemar test, written W/L = X-only / Y-only successes.
- Noninferiority (NI) uses Q2's conservative Clopper–Pearson paired bound `CP_lower(W/n; α/2) − CP_upper(L/n; α/2)`, with α = .05 and margin .02. For the π0.5 L=5 three-run mean, α is split over its three constituents and the bounds are averaged.
- "Simultaneous" NI uses α = .05/628, the frontier's full cell × arm × reference family.
- Unless stated otherwise, p values and intervals are exploratory and unadjusted.
- Non-significance is not equivalence.
- Everything is conditional on these ten tasks per suite, these libraries and test inits 0–49. It is not a claim about new tasks, scenes or robots.
- Times are CDT.

**Pure references (single runs, 500 pairs):**

| Model | Suite | L=10 (IR .5) | L=5 (IR 1.0) |
|---|---|---|---|
| π0.5 | LIBERO-10 | `r04_cost:r4f_p_l10_inf_k10_L10` .904 | mean .851 of DUAL .844 / s1001 .848 / s2001 .860 |
| π0.5 | Spatial | `r04_cost:r4f_p_sp_inf_k10_L10` .986 | mean .991 of DUAL .986 / .992 / .994 |
| GR00T | LIBERO-10 | `r05_q2:r5q2_g_l10_policy_L10` .866 | DUAL .870 |
| GR00T | Spatial | `r05_q2:r5q2_g_spatial_policy_L10` .938 | DUAL .940 |

The DUAL references live outside R. Their outcomes are read from `frontier_final/outcomes.json`; they match on task/init identity but use a different harness and seed.

**Reproduction.** Everything below runs on CPU only and writes into `rounds/r06/analysis_r6/` from the repository root. Each command is prefixed with the brief's `taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src`:

```bash
.venv/bin/python exp/offline_search/rounds/r06/analysis_scripts/a1_repair_audit.py        # §1 sweep of all 593 arms
.venv/bin/python exp/offline_search/rounds/r06/analysis_scripts/a2_headline_ablations.py  # §2–§3
.venv/bin/python exp/offline_search/rounds/r06/analysis_scripts/a3_config_c.py            # §5 (decision-log audit, contrasts)
.venv/bin/python exp/offline_search/rounds/r06/ops/c_validation.py > exp/offline_search/rounds/r06/analysis_r6/c_validation_regenerated.md
.venv/bin/python exp/offline_search/rounds/r06/ops/frontier_refresh.py exp/offline_search/rounds/r06/analysis_r6/frontier_rerun
.venv/bin/python exp/offline_search/rounds/r06/analysis_scripts/a4_frontier.py            # §6 independent check
.venv/bin/python exp/offline_search/rounds/r06/analysis_scripts/a5_prereg_checks.py       # §4 frozen outputs
.venv/bin/python exp/offline_search/rounds/r06/analysis_scripts/a6_repair_effects.py      # §1 before/after (needs a1)
.venv/bin/python exp/offline_search/rounds/r06/analysis_scripts/a7_gate_posthoc.py        # §7 post-hoc gate sizing (needs a2, a3)
```

**Reproduction checks passed.**
- `ops/paper_ab.py` output equals `PAPER_AB.md`.
- The regenerated `c_validation` tables are identical to `C_VALIDATION.md`.
- The frontier rerun is byte-identical to `frontier_final/` in every CSV/JSON/MD file except the snapshot time stamp in `frontier_data.json`.
- `a4` independently re-derives SR (from journals) and owner IR (from ledgers) for all 295 eligible in-root frontier points with 0 mismatches. It also re-derives the Pareto flags with 0 substantive mismatches: the four pure-reference rows carry an empty flag in the CSV but appear on the fronts in `frontier.md`. Finally, it recomputes every C arm's NI bound and each cell's cheapest-NI bound, and all agree.

No server, worker, chain, simulator or remote host was touched.

## 1. Data-quality repair (exception-terminated episodes)

### 1.1 Bug, detection, scope

**The bug.** `examples/libero/main.py:390-395` wraps the control loop in `except Exception` and does the following:
- it logs the error;
- it sets `_termination = "exception"`;
- it breaks out of the loop;
- it returns `success=False`.

A websocket keepalive timeout or an off-screen render error on the overloaded client host therefore ended the episode as an ordinary failure. The conductor journaled it and never retried it. The controller under test had not run to its end, so these are not outcomes.

**Detection.** The per-step file's `client_timing` row carries `termination_reason`. The bug surfaced because the frontier arm π0.5 Spatial-500 A + 1/32 scored .946, well below its A replicates. Its failures carried `termination_reason == "exception"` and were keyed by the journal's `(task_uid, run_id, attempt)`.

**Scope, recomputed by sweeping all 593 arms in R.** Every one of the 161,382 terminal journal records has a `client_timing` row with a termination reason. The entire store contains exactly **316 exception rows**. All of them belong to R6 and all were purged:

| Run root | Arms affected | Episodes purged | Largest arm | Reruns that succeeded |
|---|---|---|---|---|
| `r06_paper` (A/B replicates) | 7 of 36 | 54 | 13 (`r5q1_c10_p_l10_50_rep2`) | 44 / 54 |
| `r06_abl` (ablations) | 11 of 32 | 87 | 33 (`r6p2_identity_g_l10_500`) | 63 / 87 |
| `r06_frontier` (completion arms) | 15 of 41 | 175 | 26 (`r6q2_pi05_l10_500_B_dose0p125`) | 154 / 175 |
| **all R6** | **33** | **316** | | 261 / 316 (82.6%) |
| R2–R5 (all roots), P3 pilot (216 arms, 4,320 episodes), `r06_c_cal`, `r06_c_smoke`, `r06_c_validation`, `r06_method`, all smokes | 0 | 0 exception rows at all | | |

The purged episodes ran from 09-28 14:23 to 09-29 07:11, in bursts during host overload on the client machine.

### 1.2 Repair and verification

**The repair.** `closed_loop/ops/remote/purge_exc.py` moves every terminal journal record whose `(task_uid, run_id, attempt)` has an exception `client_timing` row into `client/purged_exceptions.jsonl`, then atomically rewrites the journal. `closed_loop/ops/chain.sh` (lines 207-210) calls it after each driver exit and before counting. The ordinary resume path then reruns the missing uids. The stale per-step rows stay, and readers select by journal.

**Verification (`a1`).**
1. All 316 purged records have `success=false` and match an exception row.
2. Each of the 33 repaired arms again has exactly one accepted record for all 500 pairs, and its summary success equals its journal.
3. No accepted record in any of the 593 arms matches an exception row (residual = 0).
4. None of the 316 reruns ended in an exception: the store's 316 exception rows are exactly the purged ones.

The pre-repair outcome of a purged pair is reconstructable as `False`. That gives the before/after values below; for the A/B cells these reproduce the pre-repair numbers quoted in the three FINAL.md files.

### 1.3 Conclusions that changed (before → after; `a6`)

| Item | Before → after | Consequence |
|---|---|---|
| π0.5 L10-50 B mean / B − A | .817 → **.827** / +10.3 → **+11.3** pp | headline gain larger |
| π0.5 L10-500 B mean / B − A | .873 → **.879** / +4.6 → **+5.1** pp | |
| GR00T L10-50 A / B / B − A | .609 / .719 → **.611 / .725** / +10.9 → **+11.4** pp | |
| GR00T L10-500 B / B − A | .867 → **.872** / +3.7 → **+4.2** pp | B's three-run mean now ≥ pure L10 (.866) |
| GR00T Sp-50 B / B − A | .875 → .876 / +0.8 → +0.9 pp | unchanged conclusion |
| Largest replicate range | 3.6 pp (π0.5 L10-50) → **2.4 pp** (π0.5 L10-500 B) | the 3.6 pp spread was the bug |
| Direct-token PCA, π0.5 Sp-500 (Δ vs A) | .946, −3.0 → **.968, −0.8** pp [−2.3, +0.7] | **"pooled PCA significantly better on dense Spatial" withdrawn** |
| Direct-token PCA, GR00T Sp-500 (Δ vs A) | .928, −3.6 → **.954, −1.0** pp [−3.2, +1.0] | as above |
| Identity metric, GR00T L10-500 (Δ vs A) | .718, −11.2 → **.752, −7.8** pp [−11.5, −4.1] | still large |
| B without no-progress, π0.5 L10-50 | .704 → .712 | still ≈ A |
| π0.5 L10-500 B + .125 random dose | .842 → **.892** (vs B mean +1.3 [−1.2, +3.9]) | "random calls hurt the dense library" (Q1/Q2 finals) withdrawn |
| π0.5 L10-500 B + .25 / risk ρ .24 | .882 / .884 → **.916 / .910** (vs B mean +3.7 [+1.2, +6.3] / +3.1 [+0.3, +5.9]) | extra calls on top of B *do* help this dense library |
| GR00T L10-500 A + 1/8 | .808 → **.842** | "A + 1/8 worse than A" withdrawn |
| π0.5 Sp-500 A + 1/32 | .946 → **.976** | the arm that exposed the bug; now ≈ A |
| GR00T L10-50 risk ρ .45 | .838 → **.868** (NI lower vs L10 −7.9 → −4.6 pp) | now the cell's cheapest point reaching pure L10 (IR .453) |
| π0.5 Sp-50 risk ρ .35; GR00T Sp-50 risk ρ .12 | .934 → .952; .854 → .876 | |

All other arms, and every pilot-based preregistered result, are untouched. The erratum at the top of each FINAL.md lists the headline changes. It does not mention two post-hoc statements those finals make that are now wrong: the π0.5 L10-500 "B + .125 dip" and the GR00T L10-500 A + 1/8 point.

### 1.4 Ledger / client decision-count tolerance

Seven repaired arms have a server-ledger decision count N that differs from the client's decision count by −1.47% to +0.12%. The purge and rerun left some server rows for aborted attempts, or missed some. `ops/frontier_refresh.py` accepts these arms at a **≤ 2% tolerance**; they are listed in `frontier_final/ledger_tolerance_flags.json`.

| Arm | Ledger N | Client N | Rel. diff | Owner IR (ledger ratios) | Worst-case IR range |
|---|---|---|---|---|---|
| `r06_abl:r6p2_identity_g_l10_500` | 31,495 | 31,711 | −0.68% | .0744 | .0739–.0807 |
| `r06_frontier:r6q2_groot_l10_500_A_dose0p125` | 29,628 | 29,713 | −0.29% | .1273 | .1270–.1298 |
| `r06_frontier:r6q2_groot_l10_50_B_dose0p5` | 30,618 | 30,880 | −0.85% | .3578 | .3547–.3632 |
| `r06_frontier:r6q2_groot_l10_50_risk_rho0p45` | 28,908 | 29,338 | −1.47% | .4530 | .4463–.4610 |
| `r06_frontier:r6q2_pi05_l10_50_risk_rho0p35` | 28,861 | 29,041 | −0.62% | .3407 | .3386–.3448 |
| `r06_frontier:r6q2_pi05_spatial_500_A_dose0p03125` | 10,647 | 10,634 | +0.12% | .0926 | .0915–.0928 |
| `r06_paper:r5q1_c10_p_l10_50_rep2` | 30,623 | 30,711 | −0.29% | .1802 | .1797–.1825 |

The worst-case range assumes every unmatched decision was vision + MISS, or none, and renormalizes by the client count. No frontier crossing, NI result or A/B conclusion changes anywhere inside those ranges. The largest case, GR00T L10-50 ρ .45, stays below .5 at .461.

## 2. A / B headline (three paired replicates)

**The two configurations.**
- **A (Commit-Cache)** retrieves with the learned per-task metric at a vision anchor, synthesizes top-16 and executes 10 controls (anchor + one blind decision). It makes no policy call.
- **B** is A plus guard-triggered committed rescue. A MISS fires on a stuck, terminal, overtime or no-progress guard, and the policy chunk is also executed for 10 controls.

GR00T's B (`GrootCommitJudge`) is π0.5's C10 `CommitJudge` plus one model constant, the closed-gripper sign. Its A ⊂ B nesting was verified on 216,474 recorded decisions ([P1 hand-back](p1_groot_commit/HANDBACK.md)).

| Cell | A SR (3 runs) | A mean @ IR | B SR (3 runs) | B mean @ IR | B − A pp [95% CI] | per-replicate McNemar p (A_k vs B_k) | B − pure L10 pp [95% CI] |
|---|---|---|---|---|---|---|---|
| π0.5 L10-50 | .706 / .710 / .726 | .714 @ .076 | .830 / .818 / .834 | .827 @ .181 | **+11.3 [+8.5, +14.3]** | 4.6e-9 / 4.3e-7 / 4.3e-7 | −7.7 [−10.8, −4.4] |
| π0.5 L10-500 | .828 / .820 / .834 | .827 @ .077 | .866 / .880 / .890 | .879 @ .159 | **+5.1 [+2.9, +7.4]** | .027 / 6.4e-4 / 7.6e-4 | −2.5 [−5.7, +0.6] |
| π0.5 Sp-50 | .838 / .844 / .828 | .837 @ .078 | .910 / .912 / .908 | .910 @ .140 | **+7.3 [+5.6, +9.1]** | 7.3e-7 / 1.4e-7 / 4.6e-10 | −7.6 [−9.9, −5.3] |
| π0.5 Sp-500 | .982 / .974 / .972 | .976 @ .078 | .982 / .978 / .986 | .982 @ .119 | +0.6 [−0.3, +1.5] | 1 / .77 / .092 | −0.4 [−1.7, +1.0] |
| GR00T L10-50 | .608 / .618 / .606 | .611 @ .074 | .726 / .720 / .728 | .725 @ .221 | **+11.4 [+7.9, +14.9]** | 1.3e-7 / 1.6e-5 / 1.5e-7 | −14.1 [−18.0, −10.4] |
| GR00T L10-500 | .830 / .832 / .828 | .830 @ .074 | .864 / .884 / .868 | .872 @ .188 | **+4.2 [+1.2, +7.3]** | .071 / .0067 / .047 | +0.6 [−2.9, +4.0] |
| GR00T Sp-50 | .868 / .866 / .868 | .867 @ .075 | .874 / .874 / .880 | .876 @ .147 | +0.9 [−1.3, +3.1] | .76 / .64 / .44 | −6.2 [−9.5, −3.0] |
| GR00T Sp-500 | .964 / .964 / .964 | .964 @ .076 | .958 / .960 / .958 | .959 @ .127 | −0.5 [−1.9, +0.9] | .61 / .79 / .61 | +2.1 [−0.5, +4.7] |

IR is the mean of the three ledger ratios. The pooled decision-weighted IR agrees to 4 decimals. B costs 2.37 / 2.08 / 1.81 / 1.53 × A on π0.5 (L10-50, L10-500, Sp-50, Sp-500) and 2.98 / 2.52 / 1.95 / 1.68 × A on GR00T.

`PAPER_AB.md` pools 1,500 replicate-pairs into one McNemar test (e.g. +256/−86, p = 9.6e-21). That treats repeated inits as independent and overstates precision. The intervals above keep the 500 clusters.

**Replicate spread.**
- The sample SD across the three replicates is 0.00–1.06 pp for A and 0.12–1.21 pp for B.
- The largest range is 2.4 pp (π0.5 B L10-500: .866 / .880 / .890).
- No within-configuration replicate pair differs by McNemar; the minimum p is .10.
- Per-episode outcomes are nonetheless far from deterministic. GR00T B L10-50 replicates disagree on 18–20% of pairs while their SRs agree within 0.8 pp. A single run therefore carries about ±3 pp noise, as R5 found.
- Replicated configurations reach the pure-L10 point estimate in two cells: B on GR00T L10-500 (.872 ≥ .866 at IR .188), and A on GR00T Sp-500 (.964 ≥ .938 at IR .076; B is lower at .959).

**Reading.**
- B's gain is large where the library is sparse and the tasks long, moderate on LIBERO-10-500, and absent on the dense Spatial-500 libraries and GR00T Spatial-50.
- On GR00T Spatial-50 the guards fire (IR doubles) without rescuing. §3 shows the reason: the no-progress guard's calls hurt there.
- B does not close the gap to pure L10 on any sparse cell (−6.2 to −14.1 pp).

## 3. Ablations

All ablation arms are single 500-pair runs. The table compares each to the A (or B) three-run mean with a bootstrap interval and to each replicate with an exact McNemar test.

**3.1 Learned per-task metric vs identity (Euclidean on z-scored 136-d features)**, `p2_ablations/metric.py`:

| Cell | Identity SR @ IR | − A mean pp [95% CI] | McNemar p vs A reps |
|---|---|---|---|
| π0.5 L10-50 | .580 @ .076 | **−13.4 [−17.3, −9.7]** | 1e-10 – 1.5e-8 |
| π0.5 L10-500 | .734 @ .076 | **−9.3 [−13.0, −5.8]** | 7.4e-7 – 3.9e-5 |
| π0.5 Sp-50 | .776 @ .077 | **−6.1 [−9.7, −2.7]** | 5.9e-4 – .011 |
| π0.5 Sp-500 | .956 @ .078 | −2.0 [−4.0, −0.1] | .015 – .18 |
| GR00T L10-50 | .506 @ .074 | **−10.5 [−14.9, −5.9]** | 1.3e-5 – 9.5e-5 |
| GR00T L10-500 | .752 @ .074 | **−7.8 [−11.5, −4.1]** | 5.5e-5 – 1.6e-4 |
| GR00T Sp-50 | .866 @ .075 | −0.1 [−3.9, +3.7] | 1 |
| GR00T Sp-500 | .952 @ .076 | −1.2 [−3.4, +1.0] | .36 |

The metric is the largest single component on every LIBERO-10 cell and on π0.5 Sp-50. It is immaterial on GR00T Spatial and marginal on π0.5 Sp-500, where the bootstrap interval touches −0.1 but one of three exact tests is n.s.

This ablation removes both the action supervision and the whitening. The pending **unsupervised Σ⁻¹ whitening control** (§7.3) would separate the two.

**3.2 Visual key: 4×4 pooling then PCA-64 (A) vs PCA-64 on all 256×2048 tokens**, `p2_ablations/token_pca.py`:

| Cell | Direct SR @ IR | − A mean pp [95% CI] | McNemar p vs A reps |
|---|---|---|---|
| π0.5 L10-50 | .712 @ .076 | −0.2 [−3.5, +3.1] | .52 – 1 |
| π0.5 L10-500 | .846 @ .077 | +1.9 [−0.4, +4.1] | .079 – .48 |
| π0.5 Sp-50 | .828 @ .077 | −0.9 [−3.7, +1.9] | .37 – 1 |
| π0.5 Sp-500 | .968 @ .078 | −0.8 [−2.3, +0.7] | .17 – .81 |
| GR00T L10-50 | .610 @ .074 | −0.1 [−4.3, +4.1] | .79 – 1 |
| GR00T L10-500 | .822 @ .075 | −0.8 [−4.3, +2.7] | .67 – .83 |
| GR00T Sp-50 | .910 @ .075 | **+4.3 [+1.1, +7.5]** | .012 – .017 |
| GR00T Sp-500 | .954 @ .076 | −1.0 [−3.2, +1.0] | .46 |

**Withdrawn claim.** "Pooled PCA is significantly better on both dense Spatial libraries (−3.0 / −3.6 pp)" was produced by 11 and 14 exception episodes in the two direct-PCA Spatial-500 arms (§1.3). The measured difference is now zero in 7 of 8 cells. The one real difference favors direct PCA, on GR00T Sp-50.

The case for pooling is therefore cost, not SR: roughly 13× cheaper online projection (≈ 0.8 ms vs 9.5–14 ms for two cameras on CPU) and an 8 MiB instead of 128 MiB projection per camera ([PCA hand-back](p2_ablations/PCA_HANDBACK.md)).

**3.3 Trigger leave-one-out of B and the mechanism arm (Bmech)** (`p2_ablations/judge.py`, `ideation_Q3/stall/bmech.py`):

| Cell | Arm | SR @ IR | − B mean pp [95% CI] | McNemar p vs B reps | − A mean pp [95% CI] |
|---|---|---|---|---|---|
| π0.5 L10-50 | B (3-run) | .827 @ .181 | — | — | +11.3 |
| | − stuck | .824 @ .179 | −0.3 [−3.1, +2.4] | .65 – .82 | +11.0 |
| | − terminal | .828 @ .176 | +0.1 [−2.8, +2.9] | .67 – 1 | +11.4 |
| | − overtime | .806 @ .184 | −2.1 [−4.9, +0.7] | .14 – .58 | +9.2 |
| | **− no-progress** | .712 @ .104 | **−11.5 [−14.8, −8.3]** | 5e-9 – 4.9e-7 | −0.2 [−2.5, +2.1] |
| | **Bmech** | .716 @ .127 | **−11.1 [−14.2, −8.1]** | 3e-9 – 1.4e-6 | +0.2 [−2.7, +3.1] |
| π0.5 Sp-50 | B (3-run) | .910 @ .140 | — | — | +7.3 |
| | − stuck / − terminal / − overtime | .906 / .904 / .908 | −0.4 / −0.6 / −0.2 (all CIs include 0) | ≥ .54 | +6.7 to +7.1 |
| | **− no-progress** | .858 @ .112 | **−5.2 [−7.1, −3.4]** | 3.5e-6 – 1.6e-4 | +2.1 [+0.5, +3.7] |
| | **Bmech** | .856 @ .127 | **−5.4 [−7.5, −3.4]** | 8.4e-6 – 2e-4 | +1.9 [+0.3, +3.5] |
| GR00T L10-50 | B (3-run) | .725 @ .221 | — | — | +11.4 |
| | − stuck / − terminal / − overtime | .728 / .716 / .698 | +0.3 / −0.9 / −2.7 (all CIs include 0) | ≥ .18 | +8.7 to +11.7 |
| | **− no-progress** | .620 @ .096 | **−10.5 [−14.0, −7.0]** | 1.3e-6 – 2.2e-5 | +0.9 [−0.9, +2.7] |
| | **Bmech** | .628 @ .132 | **−9.7 [−13.3, −6.1]** | 1.2e-5 – 1.4e-4 | +1.7 [−1.9, +5.4] |
| GR00T Sp-50 | B (3-run) | .876 @ .147 | — | — | +0.9 |
| | − stuck | .862 @ .151 | −1.4 [−2.8, −0.1] | .093 – .21 | −0.5 |
| | − terminal / − overtime | .864 / .872 | −1.2 / −0.4 (CIs include 0) | ≥ .17 | −0.3 / +0.5 |
| | **− no-progress** | **.906 @ .105** | **+3.0 [+1.3, +4.8]** | .0037 – .015 | **+3.9 [+1.9, +5.9]** |
| | **Bmech** | .902 @ .120 | **+2.6 [+1.1, +4.3]** | .0066 – .027 | +3.5 [+1.3, +5.8] |

Bmech minus "no-progress removed", as single-run pairs:

| Cell | W/L | exact p | Δ pp [95% CI] |
|---|---|---|---|
| π0.5 L10-50 | 39/37 | .91 | +0.4 [−3.0, +3.6] |
| π0.5 Sp-50 | 9/10 | 1.0 | −0.2 [−2.0, +1.6] |
| GR00T L10-50 | 47/43 | .75 | +0.8 [−2.8, +4.6] |
| GR00T Sp-50 | 7/9 | .80 | −0.4 [−2.0, +1.0] |

**The no-progress guard.**
- It is the only guard whose removal is clearly detectable.
- On both LIBERO-10-50 cells it carries all of B's gain: removing it returns SR to A.
- On π0.5 Sp-50 it carries ≈ 5 of B's ≈ 7 pp.
- On GR00T Sp-50 it is harmful. Removing it gives .906 @ .105, above every B and every A replicate.

**The other three guards.**
- Removing any one of them changes SR by at most about 2.7 pp. Every exact test against the B replicates is n.s. (p ≥ .093).
- One bootstrap interval excludes 0: GR00T Sp-50 "− stuck", −1.4 [−2.8, −0.1]. This is one of 12 unadjusted contrasts and its exact tests are n.s., so it is not treated as a detected effect.
- Joint removal is not tested here. Configuration C drops all three.
- Removing the terminal guard saves IR on the Spatial-50 cells (.140 → .124 π0.5, .147 → .131 GR00T).

**Mechanism.** Bmech turns off only the no-progress **MISS bit** and keeps its blind-LOOK veto and all histories. It is indistinguishable from removing the guard in all four cells (exact p .75–1.0). It costs more IR than full removal (+.015 to +.036) because the veto keeps its extra vision. The guard's effect, good or bad, therefore runs entirely through the policy takeover it triggers, not through observation cadence.

`ABLATIONS.md` quotes p = .58–1.0 for this comparison. The .58 was against the pre-repair .704 arm; after the repair the range is .75–1.0.

## 4. The owner's three questions: preregistered pilot outcomes, then post-hoc evidence

### 4.1 The superset profiler and the pilot

P3 v2 ([hand-back](p3_profiling/HANDBACK.md)) collected every field needed by Q1–Q3 in one campaign. It covers 8 cells, and within each cell 9 cohorts share identical superset measurements:
- A;
- doses 1/8, 1/4 and 1/2;
- P10;
- B;
- a pre-guard factorial;
- a delay window;
- a dose mixture.

Each cohort records policy chunks at every decision, K4 re-samples, keys, retrieval, guard diagnostics, control telemetry and snapshots. The pilot is **216 arms / 4,320 episodes / 180,155 decisions / 92,209 anchors**: ten tasks × *test* inits 0–1 × three seed blocks. It has no exception episodes (§1.1), so the repair does not touch the three frozen analyses. The calibration split is `init%5 == 0`, i.e. **one initial state per task**. That single fact is why every SR decision below came out inconclusive: within-task init variance is unestimable (df = 0) in the calibration and validation splits.

### 4.2 Preregistered outcomes

`a5` reads these back from the frozen output files.

| Question | Frozen decision | What stayed inconclusive, and why |
|---|---|---|
| **Q1 library quality** ([FINAL](ideation_Q1/FINAL.md), `pilot_all_cells/decision.json`) | **R selected**: the retrieval-weighted library LOEO action-reconstruction residual `R(s) = Σ w_i r_i`. Pooled within-episode Spearman ρ **.468**, simultaneous 99.375% [.391, .542]. Validation MAE reduction of predicted cache–policy disagreement **15.4%** [9.4, 22.0]%. Point gain positive in all 8 cells. The others: C (distance CDF) practical utility rejected (ρ .226, gain 0.6%); D inconclusive (ρ .149); Qrisk survives (ρ .439, 18.6%) but does not replace R (added MAE gain .0059 [−.0041, +.0161]). | R predicts *disagreement*, not failure or call value. The library-only **global / per-task SR** ranking is unresolved: eight-bank ρ with A failure .171 [−.229, .659]; 80 bank/task ρ .033. Task-level intervals are unavailable with one validation init per task. |
| **Q2 how many MISS** ([FINAL](ideation_Q2/FINAL.md), `preregistered/estimates.json`) | **All 8 cells `P10_reference_inconclusive`, 0 eligible candidates.** The 296-test family found no risk, uniform, fixed-dose or B controller that passes simultaneous 2 pp NI vs P10. Risk allocation point-dominated uniform in only 3/32 held-out cell/budget checks. Library-predicted IR errors ran from −.0037 to +.0100. | Validation has 10 init clusters, one per task, so simultaneous bounds are null. Descriptive dose curves exist, but per-task budgets (2 inits × 3 seeds per task) are not fitted. Only 4/48 slope-change checks show diminishing returns. |
| **Q3 where to MISS** ([FINAL](ideation_Q3/FINAL.md), `final_pilot/primary64.csv`, `nomination48_audit.csv`) | **No placement gate nominated (0/48).** None of the 54 estimable primary SR contrasts excludes zero under the simultaneous intervals; 10 were withheld. Commitment default kept: `min(H, 2R)` = 10 controls, hold 1, fresh retrieval anchored on the executed tail. | 44 of 48 gate entries have df = 0 in both required contrasts. The small-bank six-test G gate is vacuous (`6·p_min ≥ 1`). Timing (delay 0/1/2) shows no simultaneous superiority. |

### 4.3 What the full continuation (+31,680 episodes → 36,000) would buy

These projections come from each final's measured variance components. They are planning projections, not evidence.
- **Q1.** R's selection is already reached. Per-cell R rank precision would tighten to ±.024–.051 (fixed tasks), versus ±.037–.143 over the task superpopulation. A ±2 pp SR statement needs 747–3,538 effective pairs. A transferable *global* bank ranking needs ≥ 24 independently acquired banks, which the continuation cannot supply.
- **Q2.** The B − P10 held-out 80% MDE would be **3.53–9.51 pp**. At a true zero gap, the probability of passing the frozen family-adjusted 2 pp gate is **0.11–1.96%**. 2 pp NI would need 2,449–15,781 effective pairs per cell.
- **Q3.** The overall-CALL 80% MDE would be 4.28–12.47 pp. A 1 pp retention margin needs **3,092–19,922** effective independent pairs per cell. Timing half-widths stay at 18–26 pp.

**All three finals therefore recommend against running the continuation as designed.** It adds repeated episodes on the same test-init pool, not independent scenes, and cannot certify 2 pp (§7.3).

### 4.4 Post-hoc evidence bearing on Q1–Q3

Everything in this subsection is post-hoc, uses the repaired arms, and is conditional on these banks and test inits.
- **Q1 post-hoc.** Q1's own post-hoc checks, which used pre-repair A/B values where noted:
  - a ten-trajectory refit keeps R's MAE gain (14.83% [9.38, 20.62]);
  - an occupancy-calibrated mean disagreement associates with B's gain across the 8 banks (ρ .643 [.476, .714]).

  `a7` repeats the latter with C's production B-val calibration, using the per-cell mean disagreement `baseline_E` over ≤ 10 non-test recordings, against the *repaired* outcomes. Across 8 banks, Spearman ρ is **.64** with B − A, **.67** with C − A, and **.67** with the pure-L10 − A gap. The A success rate on the same 10 recordings gives ρ **−.72 / −.78 / −.78**. π0.5 L10-500 has the lowest disagreement (.265) yet gains +5 to +8 pp from calls. These eight fixed banks can motivate a gate (§7.1); they cannot calibrate one.
- **Q2 post-hoc.**
  - Budget matters more than guards at high spending, and the curves plateau on the sparse long-task cells: π0.5 L10-50 reaches .866–.894 for IR .29–.46 and never .904.
  - On π0.5 L10-500 extra budget does help after the repair: B + .25 is .916 @ .241, +3.7 pp over B.
  - The dense Spatial-500 cells need essentially no budget. GR00T A (.964 @ .076) is already above both pure references, and π0.5 A (.976 @ .078) is 1.0 pp below pure L10.
- **Q3 post-hoc.**
  - The no-progress reversal (§3.3) is the only placement signal with a detected sign. It helps both LIBERO-10-50 cells and π0.5 Sp-50, and hurts GR00T Sp-50.
  - Bmech localizes the effect to policy takeover.
  - Q3's post-hoc proposal (library-calibrated slow-progress trigger, ambiguity → re-observe) became C's stall component (§5).

## 5. Configuration C: library-calibrated budgeted rescue

### 5.1 Design (SELECTION §1)

C is A plus rescue calls whose *amount* is set by one owner knob and whose *placement* is set by two library-calibrated signals. It has no hand thresholds and no benchmark or robot names.

| Component | What C does |
|---|---|
| Budget (Q2) | Knob **ρ = target owner IR**. The per-anchor dose is solved so that the modeled owner IR on the calibration recordings equals ρ. The model uses an exact cadence DAG that includes stall calls, cooldowns and extra LOOKs. |
| Placement by disagreement (Q1) | Each bank row stores its LOEO residual `r_j`. At an anchor `R(s) = Σ w_i r_i`, `Ehat = a + b·R`, and the call probability is `p = min(1, λ·Ehat)`. λ matches the recordings' mean call rate to ρ. |
| Stall rescue (Q3) | Templates are aligned monotonically to the last W+1 keys. e90 and a10 are calibrated leave-one-out on the library's successful episodes. `slow_confirmed` (`δ̂ + e90 < a10`) → call; `slow_ambiguous` → the R lottery, plus one extra LOOK only if it did not call. |
| Commit / handback | 10 controls, hold 1, fresh retrieval anchored on the executed tail (B's semantics). A one-free-anchor cooldown applies **only after stall-triggered calls**. |
| Dropped | B's stuck, terminal and overtime guards, which are hand-thresholded and individually removable (§3.3). |

**Calibration contract.**
- **Inputs:** the library, ≤ 10 recorded A-controller trajectories per cell (one per task, with same-observation shadow policy chunks) on **non-test B-val initial states**, and c1.
- **Excluded:** no success label enters any fit.

**Arms.** The U arms are Q2's `RiskLottery(allocation=uniform)`, calibrated from library lengths only. R and C use the recordings.

### 5.2 Amendments and their timing (all before any C outcome)

| Event | Time (CDT, 09-29) | Content |
|---|---|---|
| SELECTION §1–§6 | 07:5x | design, §4 acceptance rules |
| **§7** (`R6-C-v2`) | 08:3x | The P4 dry-run showed that an all-call cooldown caps IR at ≈ .29. The cooldown now follows stall-triggered calls only; lottery calls have none. |
| **§7b** | header "08:5x"; the file was last modified 08:40:23 | The v2 dry-run left the stall ceiling ≈ .39 on the L10-50 cells. `slow_ambiguous` anchors now draw the lottery, and the extra LOOK happens only without a call. Infeasible targets run at the calibrated ceiling as `C.max`. |
| Recordings (`r06_c_cal`) | 08:46–09:09 | 8 cells × 10 episodes, A controller, B-val inits |
| Production calibration | 09:27 | `r06_c_cal/cal/<cell>/calibrated/calibration.json` (status `NONTEST_BVAL`, `R6-C-v2`, `lottery_then_look`) |
| Smoke (`r06_c_smoke`) | 09:31–09:34 | 2 arms × 4 episodes |
| Validation (`r06_c_validation`) | first episode 09:39, first arm complete 09:58, last 13:22 | 28 arms × 500 pairs |

Both amendments therefore precede the first C episode. The §7b header time and the file's modification time disagree by ~10 minutes, which does not affect this ordering. The 8 smoke episodes were C outcomes on test pairs (tasks 0–1 × inits 0–1) observed before validation; no change followed them. SELECTION §3 already records that C was *designed* after seeing test-init outcomes. The validation is therefore a conditional check on these scenes, not a fresh-scene confirmation.

### 5.3 Calibration (feasibility and Ehat coefficients)

Floors and ceilings are the modeled IR at p ≡ 0 and p ≡ 1. The "λ" columns are the solved R-placement parameter for each validation arm. All targets are feasible except GR00T L10-50 at .45 with stall, whose ceiling is .4407; that arm ran as `Cmax` at ρ = .440704.

| Cell | a | b | anchors (10 rec.) | mean E | rec. A success | no-stall floor / ceiling | stall floor / ceiling | λ: R30 / C30 / C45 (R18 / C18 dense) |
|---|---|---|---|---|---|---|---|---|
| π0.5 L10-50 | .0857 | 1.0997 | 346 | .437 | 7/10 | .077 / .504 | .133 / .461 | 1.196 / 1.170 / 3.069 |
| π0.5 Sp-50 | 0 | 1.3949 | 152 | .558 | 7/10 | .077 / .506 | .126 / .467 | 0.896 / 0.914 / 2.320 |
| GR00T L10-50 | .0785 | 1.0958 | 365 | .485 | 6/10 | .075 / .504 | .154 / **.441** | 1.085 / 1.100 / Cmax |
| GR00T Sp-50 | .0593 | 0.9770 | 133 | .436 | 8/10 | .075 / .509 | .119 / .476 | 1.187 / 1.175 / 2.545 |
| π0.5 L10-500 | .0345 | 0.9071 | 274 | .265 | 9/10 | .076 / .502 | .092 / .489 | 0.919 / 0.868 |
| π0.5 Sp-500 | 0 | 1.1905 | 135 | .345 | 8/10 | .078 / .514 | .110 / .495 | 0.655 / 0.551 |
| GR00T L10-500 | .0042 | 1.0732 | 290 | .386 | 9/10 | .075 / .504 | .107 / .477 | 0.637 / 0.508 |
| GR00T Sp-500 | 0 | 1.0922 | 122 | .366 | 9/10 | .075 / .507 | .089 / .498 | 0.654 / 0.625 |

`Ehat` is essentially `R` rescaled: b is between .91 and 1.39, and a is between 0 and .09.

The stall component raises the IR floor, because confirmed stalls force calls. The rise is .04–.08 on the 50-episode banks and .01–.03 on the 500-episode banks. Its effect on the ceiling is larger in the other direction: cooldowns and LOOKs cut the achievable maximum by .01–.06.

The recorded successes (60–90%) are shown for context only; no fit uses them.

### 5.4 Smoke checks and a full decision-log audit

`a3` audits the accepted attempt of every episode in the 2 smoke arms and all 28 validation arms, using the server decision logs. It reads 14,008 episodes in total.

**Smoke.** The smoke arms are π0.5 L10-500 C18 (4/4 successes; 28 lottery + 3 stall calls) and GR00T L10-500 C18 (3/4; 21 + 15 calls, 7 extra LOOKs, 10 ambiguous anchors). Four episodes each say nothing about SR or budget.

**All 30 audited arms pass every check with zero violations:**
- decision, vision and MISS counts equal the cost ledger `(N, V, M)` exactly;
- every non-terminal call is followed by exactly one policy tail and then a fresh vision anchor;
- the cooldown follows every stall call and never follows a lottery call;
- no call occurs inside a cooldown;
- every `slow_confirmed` anchor outside cooldown calls;
- extra LOOKs occur only at ambiguous anchors that did not call;
- no shadow policy forward occurs.

**Stall-triggered calls** made up 5–26% of C's calls on the sparse cells (C30 12–26%, C45/Cmax 5–16%) and 7–36% on the dense cells (C18): π0.5 Sp-500 7%, GR00T Sp-500 16%, π0.5 L10-500 25%, GR00T L10-500 36%.

**Extra LOOKs** numbered 843 / 256 / 949 / 232 in the sparse C30 arms and 24–755 in the dense C18 arms.

### 5.5 SELECTION §4 acceptance rules, verdicts exactly as preregistered

**R placement.**
- **Rule:** supported if pooled (4 sparse cells) R30 − U30 > 0, with realized IR matched within .01. Falsified at this operating point if the pooled upper 95% bound is < 0, or if IRs cannot be matched.
- **Measured:** pooled **−0.2 pp [−1.9, +1.4]**. Per cell −1.2 / +0.4 / +1.4 / −1.6 pp, all with p ≥ .34. |ΔIR| is .006 / **.014** / .002 / .003, so π0.5 Sp-50 misses by .004, in R's favor; the pooled ΔIR is −.005.
- **Verdict: not supported.** The SR clause does not falsify (upper bound +1.4). Under the per-cell IR reading that `ops/c_validation.py` implements, the literal rule says **"falsified at this operating point"** because one cell's IR could not be matched. `C_VALIDATION.md` reports "not falsified". Either way R placement is not adopted, and any advantage is bounded at ≈ +1.4 pp.

**Stall component.**
- **Rule:** supported if C30 keeps ≥ ½ of B's no-progress benefit on both L10-50 cells and does not lose to B-without-NP on GR00T Sp-50 by > 2 pp. Falsified if it loses to R30 on L10-50, or if it rejects nearly every long-task trigger.
- **Measured:** C30 − B-without-NP is **+15.6** [+11.2, +19.8] (π0.5 L10-50) and **+18.6** [+14.0, +23.2] (GR00T L10-50), against B's NP benefits of 11.5 and 10.5 pp. On GR00T Sp-50 it is **+3.4 [+0.4, +6.4]** (W/L 40/23, p = .043). C30 − R30 on L10-50 is +0.2 and +2.2. Stall calls fired 1,555 and 2,229 times.
- **Verdict: supported.** The half-benefit clause does not isolate the stall component. C30 spends ρ = .30 against B's .18–.22, and R30 alone would also pass it (+15.4 / +16.4 pp over B-without-NP; +0.6 on GR00T Sp-50). The component-specific contrast is C30 − R30 at the same ρ, in §5.6.

**Reach.**
- **Rule:** per cell, the lowest C ρ whose 2 pp CP NI lower bound passes vs L10, and separately vs L5, or "not reached ≤ .45".
- **Measured:** C45 / Cmax NI lower vs L10 is −5.8 / −4.8 / −9.4 / −4.2 pp. Vs L5, C45 on π0.5 L10-50 gives −1.84 (three-reference L5) or −1.87 (the two in-root references); every other cell and ρ fails, as does C30 on π0.5 L10-50 (−5.0).
- **Verdict:** **not reached ≤ .45 vs L10 in any sparse cell. Vs L5, reached at ρ = .45 on π0.5 L10-50 only** (nominal, not simultaneous).

**Dense cells.**
- **Rule:** C18 SR within 2 pp of B (lower bound > −2 pp) **and** realized IR ≤ B's.
- **Measured:**
  - π0.5 L10-500: +2.3 [−0.5, +5.1] at .191 vs .159;
  - π0.5 Sp-500: +0.4 [−0.9, +1.6] at .162 vs .119;
  - GR00T L10-500: −0.6 [−3.7, +2.4] at .184 vs .188;
  - GR00T Sp-500: −2.5 [−4.7, −0.4] at .184 vs .127.
- **Verdict: fails as a conjunction in all four cells.** Both π0.5 cells pass on SR but spend more; GR00T L10-500 matches cost but its lower bound is −3.7; GR00T Sp-500 fails both.

**Cost calibration.**
- **Rule:** realized IR within ρ ± .02, otherwise the portable calibration is falsified for that cell.
- **Measured:** 26 of 28 arms pass. The misses are π0.5 Sp-50 R30 (−.022) and C45 (+.0201). The U arms, calibrated from library lengths only, are 8/8 within ±.008.
- **Verdict: supported except those two marginal misses, both in π0.5 Sp-50.** By the rule's wording the calibration is falsified for that cell by .002 and .0001.

### 5.6 All C-family arms and post-hoc contrasts

The NI lower bounds come from `frontier_final/noninferiority.csv` and are independently re-derived in `a4`.

| Cell | cfg | SR @ owner IR | IR − ρ | lottery / stall calls | − A mean pp [95% CI] | − B mean pp [95% CI] | NI lower vs L10 / L5 pp |
|---|---|---|---|---|---|---|---|
| π0.5 L10-50 | U30 | .878 @ .301 | +.001 | 7,590 / — | +16.4 [+12.6, +20.2] | +5.1 [+1.9, +8.3] | −7.7 / −3.9 |
| | R30 | .866 @ .295 | −.005 | 7,477 / 0 | +15.2 [+11.3, +19.1] | +3.9 [+0.5, +7.1] | −8.8 / −5.2 |
| | C30 | .868 @ .301 | +.001 | 6,058 / 1,555 | +15.4 [+11.5, +19.4] | +4.1 [+0.6, +7.5] | −8.6 / −5.0 |
| | C45 | .894 @ .458 | +.008 | 11,283 / 1,350 | +18.0 [+14.1, +22.1] | +6.7 [+3.3, +10.1] | −5.8 / **−1.8** |
| π0.5 Sp-50 | U30 | .940 @ .292 | −.008 | 2,847 / — | +10.3 [+7.3, +13.3] | +3.0 [+0.3, +5.6] | −7.9 / −8.6 |
| | R30 | .944 @ .278 | −.022 | 2,655 / 0 | +10.7 [+7.5, +13.9] | +3.4 [+0.5, +6.1] | −7.4 / −8.2 |
| | C30 | .938 @ .288 | −.012 | 2,418 / 321 | +10.1 [+7.0, +13.2] | +2.8 [+0.0, +5.6] | −8.1 / −8.9 |
| | C45 | .966 @ .470 | +.020 | 4,756 / 271 | +12.9 [+9.9, +15.9] | +5.6 [+3.0, +8.1] | −4.8 / −5.6 |
| GR00T L10-50 | U30 | .770 @ .301 | +.001 | 8,407 / — | +15.9 [+11.3, +20.5] | +4.5 [+0.3, +8.7] | −15.4 / −15.9 |
| | R30 | .784 @ .303 | +.003 | 8,439 / 0 | +17.3 [+13.0, +21.5] | +5.9 [+1.6, +10.1] | −14.0 / −14.5 |
| | C30 | .806 @ .310 | +.010 | 6,276 / 2,229 | +19.5 [+14.9, +24.1] | +8.1 [+4.0, +12.1] | −11.7 / −12.1 |
| | Cmax | .822 @ .450 | +.009 | 11,316 / 2,169 | +21.1 [+16.5, +25.7] | +9.7 [+5.7, +13.9] | −9.4 / −10.1 |
| GR00T Sp-50 | U30 | .928 @ .298 | −.002 | 3,080 / — | +6.1 [+2.5, +9.7] | +5.2 [+1.9, +8.4] | −5.4 / −5.5 |
| | R30 | .912 @ .295 | −.005 | 3,020 / 0 | +4.5 [+0.8, +8.1] | +3.6 [+0.3, +7.0] | −7.3 / −7.3 |
| | C30 | **.940 @ .298** | −.002 | 2,560 / 404 | +7.3 [+4.1, +10.7] | +6.4 [+3.4, +9.5] | −4.1 / −4.0 |
| | C45 | .936 @ .454 | +.004 | 4,660 / 415 | +6.9 [+3.4, +10.3] | +6.0 [+2.8, +9.2] | −4.2 / −4.1 |
| π0.5 L10-500 | U18 | .874 @ .179 | −.001 | 3,473 / — | +4.7 [+1.4, +7.9] | −0.5 [−3.5, +2.5] | −8.0 / −4.4 |
| | R18 | .884 @ .185 | +.005 | 3,643 / 0 | +5.7 [+2.6, +8.7] | +0.5 [−2.3, +3.3] | −6.9 / −3.1 |
| | C18 | **.902 @ .191** | +.011 | 2,824 / 918 | +7.5 [+4.4, +10.5] | +2.3 [−0.5, +5.1] | −4.9 / **−1.3** |
| π0.5 Sp-500 | U18 | .968 @ .179 | −.001 | 1,284 / — | −0.8 [−2.4, +0.8] | −1.4 [−2.9, +0.0] | −4.6 / −5.3 |
| | R18 | .966 @ .175 | −.005 | 1,229 / 0 | −1.0 [−2.5, +0.4] | −1.6 [−3.3, +0.0] | −4.8 / −5.6 |
| | C18 | .986 @ .162 | −.018 | 970 / 69 | +1.0 [−0.2, +2.3] | +0.4 [−0.9, +1.6] | −2.3 / −3.0 |
| GR00T L10-500 | U18 | .848 @ .180 | +.000 | 3,653 / — | +1.8 [−1.7, +5.2] | −2.4 [−5.4, +0.6] | −7.3 / −7.5 |
| | R18 | .852 @ .180 | +.000 | 3,627 / 0 | +2.2 [−1.3, +5.7] | −2.0 [−4.9, +0.9] | −6.8 / −7.2 |
| | C18 | .866 @ .184 | +.004 | 2,315 / 1,313 | +3.6 [+0.1, +7.1] | −0.6 [−3.6, +2.4] | −5.3 / −5.4 |
| GR00T Sp-500 | U18 | .954 @ .181 | +.001 | 1,374 / — | −1.0 [−2.8, +0.8] | −0.5 [−2.4, +1.4] | −2.1 / −2.4 |
| | R18 | .940 @ .179 | −.001 | 1,366 / 0 | **−2.4 [−4.2, −0.6]** | −1.9 [−4.0, +0.1] | −3.6 / −4.0 |
| | C18 | .934 @ .184 | +.004 | 1,212 / 227 | **−3.0 [−5.0, −1.0]** | **−2.5 [−4.7, −0.4]** | −4.3 / −4.6 |

For U, the "lottery" column counts `RiskLottery` calls (reason 61).

**Post-hoc pooled contrasts.** Each pools cells with equal weight, resampling each cell independently. These are descriptive, not preregistered.

| Contrast (same ρ) | Pooled pp [95% CI] | Per cell pp (W/L, p) |
|---|---|---|
| R − U, 4 sparse (ρ .30) | −0.25 [−1.90, +1.45] | π0.5 L10 −1.2 (33/39); π0.5 Sp +0.4; GR00T L10 +1.4 (70/63); GR00T Sp −1.6 |
| R − U, 4 dense (ρ .18) | −0.05 [−1.45, +1.35] | +1.0 / −0.2 / +0.4 / −1.4 (5/12, p .14) |
| **R − U, all 8** | **−0.15 [−1.25, +0.95]** | — |
| C − R, 4 sparse | +1.15 [−0.50, +2.80] | +0.2 / −0.6 / +2.2 (58/47) / **+2.8 [+0.2, +5.6]** (31/17, p .060) |
| C − R, 4 dense | +1.15 [−0.15, +2.50] | +1.8 / **+2.0 [+0.6, +3.6]** (13/3, p .021; C18 spent .013 less) / +1.4 / −0.6 |
| **C − R, all 8** | **+1.15 [+0.12, +2.23]** | — |
| C − U, 4 sparse / 4 dense | +0.90 [−0.65, +2.45] / +1.10 [−0.20, +2.45] | dense GR00T Sp-500 **−2.0 [−3.8, −0.4]** (4/14, p .031) |
| C45 − C30, 4 sparse (+.14 to +.18 IR) | +1.65 [0.00, +3.35] | +2.6 / +2.8 (p .049) / +1.6 / −0.4 |

**C at ρ ≈ .45 vs the task-level risk lottery at ρ .45**, single-run pairs:

| Cell | C | Lottery | W/L | exact p | Δ pp [95% CI] |
|---|---|---|---|---|---|
| π0.5 L10-50 | C45 .894 @ .458 | .874 @ .446 | 45/35 | .31 | +2.0 [−1.6, +5.6] |
| π0.5 Sp-50 | C45 .966 @ .470 | .986 @ .442 | 7/17 | .064 | −2.0 [−3.8, −0.2] |
| GR00T L10-50 | Cmax .822 @ .450 | .868 @ .453 | 39/62 | **.028** | −4.6 [−8.4, −0.8] |

GR00T Sp-50 has no ρ .45 lottery arm.

The lottery is better in two of three cells and worse, though not significantly, in the third. `C_VALIDATION.md` reports only the two cells where it is better. No U45 arm was run, so the cause is not isolated. It could be task-level allocation, or C's cooldowns and extra LOOKs, which spend budget.

### 5.7 What C does and does not buy

**It helps:**
- **Spend control from the library.** ρ is a working knob. Realized IR fell within ±.02 of target in 26/28 arms, and within ±.023 in all 28. The inputs were only the library, c1 and ≤ 10 non-test recordings per cell; the uniform variant needed library lengths only. This is the most portable result of R6.
- **The calibrated stall trigger, a little.** At matched ρ it adds **+1.15 pp** pooled over 8 cells [+0.12, +2.23] (post-hoc pooling). The gain is concentrated on GR00T Sp-50 (+2.8), π0.5 Sp-500 (+2.0) and GR00T L10-50 (+2.2, n.s.). It is zero on the π0.5 sparse cells and negative on GR00T Sp-500. It keeps all of the raw no-progress guard's long-task benefit (as R does at the same budget). On GR00T Sp-50 it turns that guard's harm into a gain: C30 .940 vs B-without-NP .906 (+3.4), though at 2.8× the IR.
- **Where calls pay.** On the sparse cells at ρ .30, all three C-family arms beat B by +2.8 to +8.1 pp while spending 1.4–2.1× B's IR. GR00T Sp-50 C30 (.940 @ .298) is that cell's best observed point and equals pure L10 as a point estimate.

**It does not help:**
- **R placement.** Calls placed where predicted disagreement is high did no better than uniform calls in any family (8-cell −0.15 pp). R predicts disagreement, not recoverable failure.
- **Reaching pure-inference SR with a certificate.** No C arm passes 2 pp NI vs L10. The high-budget C45 is beaten by the risk lottery in 2 of 3 cells.
- **Dense libraries.** At ρ .18 C matches or beats B only by spending more (π0.5), and on **GR00T Sp-500 all three C-family arms are at or below A**; C18 is −3.0 pp [−5.0, −1.0] vs A and −2.5 vs B. C has no way to decide *not* to spend. **A library-level no-call gate is the missing piece** (§7.1).

## 6. The IR–SR frontier per cell

The figure is [`~/projects/openpi_ext/artifacts/frontier_r6/frontier_r6_final.png`](/home/weiland/projects/openpi_ext/artifacts/frontier_r6/frontier_r6_final.png) (`.pdf` alongside; plotting script outside the repository). Data are in `frontier_final/`: 325 records, 299 eligible on the common 500-pair set; pilot outcomes excluded.

**Pareto front.** A point is on the front if no other eligible point has at least its SR at no greater IR, with one strict inequality. Pure references are shared by the two library columns. Single runs are optimistic: selecting the best replicate inflates the front, and the A/B three-run means and §2 intervals are the reproducible statements. The lowest front points often win by < .0002 IR (for example the identity-metric ablations at .0763 vs A at .0764), so they are artefacts of rounding-level cost differences.

**6.1 Pareto staircase.** Points are SR @ owner IR, in ascending IR. **C** marks configuration C, U the uniform lottery, R the R-placed lottery.

- **π0.5 L10-50:**
  - .580 @ .0763 (identity ablation);
  - .726 @ .0764 (A rep3);
  - .828 @ .176 (B − terminal);
  - .830 / .834 @ .181 (B reps 1 / 3);
  - .842 @ .260 (B + .25);
  - .866 @ .292 (B cap 4);
  - **.868 @ .301 (C30)**;
  - .878 @ .301 (U30);
  - .880 @ .339 (B + .5);
  - .886 @ .423 (B + .75);
  - **.894 @ .458 (C45)**;
  - pure L10 .904 @ .5.
- **π0.5 L10-500:**
  - .734 @ .0764 (identity);
  - .828 / .834 @ .0765 (A reps 1 / 3);
  - .846 @ .0766 (direct PCA);
  - .880 @ .156 and .890 @ .159 (B reps 2 / 3);
  - **.902 @ .191 (C18)**;
  - .906 @ .197 (R5 K7 hand run; its four-run mean was .886);
  - .916 @ .241 (B + .25).
- **π0.5 Sp-50:**
  - .776 @ .0772 (identity);
  - .828 @ .0774 (direct);
  - .844 @ .0775 (A rep2);
  - .858 @ .112 (B − no-progress);
  - .904 @ .124 (B − terminal);
  - .912 @ .139 (B rep2);
  - .934 @ .227 (B + .25);
  - .944 @ .278 (R30);
  - .962 @ .315 (B + .5);
  - .978 @ .411 (B + .75);
  - .980 @ .430 (R3 confidence h70);
  - .986 @ .442 (risk ρ .45);
  - pure L5 s2001 .994 @ 1.0.
- **π0.5 Sp-500:**
  - .956 @ .0779 (identity);
  - .968 @ .0780 (direct);
  - .972 / .974 / .982 @ .078 (A reps 3 / 2 / 1);
  - .986 @ .118 (B rep3);
  - .990 @ .123 (R5 Q6 wrist + tail, at full-camera prices);
  - pure L5 s2001 .994 @ 1.0.
- **GR00T L10-50:**
  - .570 @ .050 (15-step tail);
  - .606 / .618 @ .074 (A reps);
  - .610 @ .074 (direct);
  - .620 @ .096 (B − no-progress);
  - .628 @ .132 (Bmech);
  - .718 @ .185 (G10);
  - .728 @ .219 (B − stuck);
  - .742 @ .290 (B + .25);
  - .794 @ .291 (CycleTail k2);
  - **.806 @ .310 (C30)**;
  - .840 @ .346 (risk ρ .35);
  - .852 @ .430 (B + .75);
  - .868 @ .453 (risk ρ .45);
  - pure L5 DUAL .870 @ 1.0.
- **GR00T L10-500:**
  - .808 @ .050 (15-step tail);
  - .828 / .832 @ .0745 (A reps);
  - .842 @ .127 (A + 1/8);
  - .852 @ .131 (risk ρ .13);
  - **.866 @ .184 (C18)**;
  - .868 / .884 @ .188 (B reps 3 / 2).
- **GR00T Sp-50:**
  - .864 @ .051 (15-step tail);
  - .866 / .868 @ .075 (A reps);
  - .910 @ .075 (direct PCA);
  - .920 @ .198 (G10);
  - .934 @ .233 (CycleTail k3);
  - **.940 @ .298 (C30)**.
- **GR00T Sp-500:**
  - .946 @ .052 (15-step tail);
  - .964 @ .0755 (A);
  - .976 @ .078 (R4 phase particles).
  - Both pure references are dominated.

**6.2 Minimum IRs** (`frontier_final/cell_gaps.csv`, re-derived in `a4`). "Point reaches" means SR ≥ the reference SR. "Nominal NI" means the CP lower bound is > −2 pp at α = .05. "Simultaneous NI" uses α = .05/628. Entries marked "(pure)" mean that no cheaper observed point qualifies.

| Cell | pure L10 / L5 SR | Lowest IR, point reaches L10 | Lowest IR within 2 pp of L10 (point) | Nominal NI vs L10 | Nominal NI vs L5 | Simultaneous NI (L10 / L5) |
|---|---|---|---|---|---|---|
| π0.5 L10-50 | .904 / .851 | **not reached** (best C45 .894 @ .458) | .423 (B + .75, .886) | .5 (pure) | **.458 (C45, −1.84)** | .5 / 1.0 (pure) |
| π0.5 L10-500 | .904 / .851 | .197 (single K7 run .906; B reps .866–.890) | .159 (B rep3 .890) | .5 (pure) | **.191 (C18, −1.26)** | .5 / 1.0 |
| π0.5 Sp-50 | .986 / .991 | .442 (risk ρ .45, .986) | .411 (B + .75, .978) | .5 (pure) | 1.0 (pure) | .5 / 1.0 |
| π0.5 Sp-500 | .986 / .991 | .118 (B rep3 .986; B mean .982) | .078 (direct PCA .968; A reps .972–.982) | **.123 (Q6 .990, −1.75)** | 1.0 (pure) | .5 / 1.0 |
| GR00T L10-50 | .866 / .870 | .453 (risk ρ .45, .868) | .430 (B + .75, .852) | .5 (pure) | 1.0 (pure) | .5 / 1.0 |
| GR00T L10-500 | .866 / .870 | .184 (**C18** .866; B mean .872 @ .188) | .131 (risk ρ .13, .852) | .5 (pure) | 1.0 (pure) | .5 / 1.0 |
| GR00T Sp-50 | .938 / .940 | .298 (**C30** .940) | .198 (G10 .920) | .5 (pure) | 1.0 (pure) | .5 / 1.0 |
| GR00T Sp-500 | .938 / .940 | .052 (15-step tail .946; A .964 @ .076) | .052 | **.076 (A, −1.06)** | **.076 (A, −1.31)** | .5 / 1.0 |

**The minimal IR that keeps pure-inference SR.**
- The point estimate reaches pure L10 at IR .05–.45 in seven cells, with a single run in all but GR00T L10-500 and GR00T Sp-500 (§2).
- The 2 pp margin is certified nominally only on the dense Spatial cells, and against the weaker L=5 reference on π0.5 LIBERO-10.
- **Under the simultaneous family no point below the reference's own cost passes in any cell.**
- π0.5 L10-50 is the unsolved cell: SR plateaus at .87–.89 for IR .29–.46.

**6.3 Where C sits.**

| Cell | C arm(s) on the observed front | Dominated C arms (by) |
|---|---|---|
| π0.5 L10-50 | **C30 .868 @ .301, C45 .894 @ .458** | — (R30 is dominated by B cap4 .866 @ .292; U30 .878 @ .301 is on the front, above C30 at the same IR) |
| π0.5 L10-500 | **C18 .902 @ .191** | — (R18 and U18 are dominated by B reps) |
| π0.5 Sp-50 | none (R30 .944 @ .278 is on the front) | C30 (by R30), C45 (by B + .75 .978 @ .411, R3 h70, risk ρ .45) |
| π0.5 Sp-500 | none | C18 .986 @ .162 (by B rep3 .986 @ .118); R18 / U18 by 17 cheaper points including pure-cache ablations |
| GR00T L10-50 | **C30 .806 @ .310** | Cmax (by risk ρ .35 .840 @ .346); R30 / U30 (by CycleTail k2 .794 @ .291) |
| GR00T L10-500 | **C18 .866 @ .184** | R18 / U18 (by risk ρ .13 .852 @ .131) |
| GR00T Sp-50 | **C30 .940 @ .298** (best SR in the cell) | C45 (by C30); R30 (by G10); U30 (by CycleTail k3) |
| GR00T Sp-500 | none | C18 / R18 / U18 by 14–19 cheaper points, including pure cache |

C is on the observed front in 5 of 8 cells. It is off the front exactly where the library is already good enough (both Spatial-500 cells) and on π0.5 Sp-50, where R30 and the B-dose / lottery family do better. None of this is a statistical dominance claim.

## 7. Synthesis and proposals

### 7.1 The owner's three questions, in plain terms

**Q1. How good is the library?** The leave-one-out idea works for one well-defined quantity: how far the cache's action would be from the policy's at a given state. That is R, one stored residual per library row plus a weighted sum at each anchor. It was selected by preregistration in all 8 cells and can be calibrated from ≤ 10 recorded trajectories.

It does **not** tell you whether the robot will fail, or whether calling the policy there will help. R-placed calls did no better than uniform calls, and library-only scores did not rank the 8 banks by SR (Q1 primary ρ .17, CI spans zero).

What does separate strong from weak libraries in our data is closed-loop:
- the gap between A and pure inference (+19 to +26 pp on the LIBERO-10-50 cells, −3 to +1 pp on Spatial-500);
- post-hoc, and only as a hypothesis, cheap pre-test signals from the ≤ 10 non-test recordings: their mean cache–policy disagreement (ρ ≈ .64–.67 with the value of calls across 8 banks) and A's success on them (ρ ≈ −.78).

A library-quality score that can switch calls off has to be validated prospectively; none is yet.

**Q2. How many MISSes?**
- **Setting the budget works.** Give ρ = target IR; the library plus a few recordings deliver it within ±.02.
- **How much to set depends on the library:**
  - **Dense, well-covered libraries (Spatial-500):** zero or almost zero. A (IR ≈ .076) is at or above pure-inference SR on GR00T and 1 pp below on π0.5, and extra calls are neutral or harmful.
  - **Long tasks with a large library (L10-500):** about .15–.24. B (.16–.19) and C18 (.18–.19) reach 0.2–2.5 pp of pure L10 as point estimates, but none certifies a 2 pp margin versus L10.
  - **Sparse libraries:** .30–.45, and even then π0.5 L10-50 plateaus 1–2 pp below pure L10.

No budget below the pure-inference cost is certified by a simultaneous 2 pp test in any cell.

**Q3. Where should MISSes go?**
- **Not by predicted disagreement.** R = uniform.
- **Yes, when progress has stalled.** The library-calibrated stall signal adds about 1 pp at equal budget. Its uncalibrated ancestor, B's no-progress guard, is the single most valuable trigger on long-task sparse libraries and harmful on GR00T Spatial-50. The calibrated version keeps the first and removes the second.
- **Across tasks rather than per state, at high budgets.** The task-level risk lottery beat C at IR ≈ .45 in two of three cells.
- **Not at all on libraries that are already good enough.** This decision is the missing piece: a library-level no-call gate that sets ρ to the A floor when the library covers the task.

**Generality.** The *recipes* are portable: the budget algebra, R, the stall calibration and the commit rule read only manifest fields, library statistics and c1. The *evidence* is one benchmark family: 2 suites × 2 models × 2 library sizes on test inits 0–49 that also informed the design. Transfer is unproven until a second benchmark or robot repeats it with the frozen recipe.

### 7.2 Ranked next steps (proposals for the owner, not launched)

1. **Freeze the paper on A and B (keep).**
   - Report the three-replicate table with the paired intervals of §2, not the pooled 1,500-pair McNemar.
   - Include the metric ablation, direct PCA (the withdrawn claim, and pooling as a cost choice), the trigger leave-one-out and Bmech.
   - State NI claims only as in §6.2.
   - Include the repair (§1) as a methods note, with the ≤ 2% ledger tolerance.
   - Run the Σ⁻¹ whitening control if the owner approves (§7.3).
2. **Replace C by C′ = ρ knob + uniform placement + calibrated stall trigger + a library-level no-call gate.**
   - **Drop R placement** (no gain in 8 cells; the simpler uniform lottery ties it).
   - **Keep** the ρ calibration and the stall trigger.
   - **Preregister the gate rule before any outcome.** Candidate inputs, all pre-test, come from the ≤ 10 non-test recordings: mean shadow disagreement E, and A's success on them. One such rule is to set ρ to the A floor when E is below a declared quantile or A succeeds on ≥ 9/10.
   - **The post-hoc sizing (§4.4) misplaces π0.5 L10-500** (low E, yet calls help). So the gate must be judged on held-out banks, with acceptance "never below A by > 1 pp on the dense cells, and no loss of the sparse-cell gains".
   - Also test task-level allocation inside C′ (a U45 vs lottery-45 pair) to explain the ρ ≈ .45 result.
3. **Second benchmark and robot:** RoboCasa365 with the existing W13 libraries on both policies (13 tasks × 50 successful trajectories, 3 cameras; named in [ideation_G](ideation_G/REPORT.md)). Calibrate only on non-test scenes, with pairing identity per that benchmark's semantics, and preregister everything. Test:
   - (a) A vs B vs pure inference at L = 2R and L = R, at two library sizes (e.g. 10 and 50 per task);
   - (b) the portable cost calibration: ρ ∈ {.18, .30, .45} within ±.02 from the library + ≤ 10 recordings;
   - (c) C′ vs uniform at matched ρ (the stall component);
   - (d) the no-call gate on the stronger library;
   - (e) the Q1 rank/loss thresholds for R (ρ ≥ .20, MAE gain ≥ 5%);
   - (f) the metric ablation and the Σ⁻¹ control;
   - (g) ideation_G's transplanted-LIBERO-constants arm, to separate "the method transfers" from "the calibration transfers".

   The C plugin currently refuses bank interfaces other than 5-control requests with a 10-control commit ([hand-back](ideation_Q1/method_c/HANDBACK.md)). That transport and the executed-control timestamps must be ported first.
4. **Keep R as a library diagnostic, not a trigger.** Use it to audit where a library will disagree with its policy and to report library coverage. Recompute it whenever PCA, the metric or the kernel changes.
5. **Drop:**
   - the full continuation as designed;
   - the raw no-progress guard as a portable default (keep it only inside the frozen B for the paper);
   - random extra doses on dense Spatial libraries;
   - expecting any placement signal to reach π0.5 L10-50 pure SR below pure cost with this library. There, library growth (R5: +14 pp from 250 paid episodes) is the lever, not smarter calls.

### 7.3 Pending owner decisions

1. **Full pilot continuation (+31,680 episodes).** Recommendation: **do not run it as designed**; all three finals agree.
   - **Why not:** it cannot certify a 2 pp margin (pass probability 0.1–2% even at a true zero gap; 1 pp needs 3,092–19,922 effective pairs per cell). R's selection is already reached. It adds repeats on the same LIBERO test-init pool, and LIBERO has no third pool of fresh states: the test states are for measurement only, and the B pool is for training and calibration.
   - **Alternative use of the GPU time:** C′ and the second benchmark (§7.2).
   - **Cost of skipping:** the frozen Q1–Q3 tests stay inconclusive on their own terms.
2. **Unsupervised whitening Σ⁻¹ control (8 pure-cache arms).** Recommendation: **run it.**
   - The metric ablation (−7.8 to −13.4 pp on LIBERO-10) is the largest single effect in the paper. The current control removes both the action supervision and the whitening, so a reviewer will ask which one matters.
   - The arms are pure-cache A arms (IR ≈ .075, no policy calls), the cheapest arms in the program.
   - **Cost of skipping:** the paper must claim only "learned per-task metric vs identity", not "action-supervised whitening".
3. **Display convention for the LIBERO-10 "+0.04".**
   - **What it is.** In the owner's demonstration figure (`~/projects/openpi_ext/artifacts/fig2_frontier_ab/`), the new LIBERO-10 points (A and B) are drawn 0.04 higher by the owner's convention. The tables and this report use the measured values.
   - **Recommendation:** keep measured values in every table and in the text. If the figure keeps the +0.04 display shift, state it in the figure caption as a presentation offset, not a measurement. Never let a shifted point sit next to an unshifted reference without that label.
   - **Cost of either choice:** none to the analysis. The risk is only a reader comparing the figure with the tables and finding a 4 pp discrepancy.

## Appendix A. Consistency of the earlier R6 documents with this recomputation

| Document | Status |
|---|---|
| `PAPER_AB.md` | Reproduced exactly: SRs, means, IRs, pooled +/− and pooled p. Its pooled McNemar treats 1,500 repeated-init pairs as independent; §2 gives cluster intervals instead. |
| `ABLATIONS.md` | Tables reproduced. **Bmech vs "no-progress removed" p = .58–1.0 is pre-repair** for π0.5 L10-50; after the repair it is .75–1.0. |
| `C_VALIDATION.md` | All tables reproduced identically by `ops/c_validation.py`. Three points differ from its prose: (i) the R-placement verdict "not falsified" conflicts with the literal per-cell IR clause of §4 (§5.5); (ii) π0.5 L10-500 C18 (.902) is listed as "reaching pure-inference SR as a point estimate" but is 0.2 pp *below* .904; (iii) "the risk lottery beats C at the high budget" holds in 2 of the 3 comparable cells, and π0.5 L10-50 goes the other way (+2.0, n.s.). |
| `SELECTION.md` | §7b's header time (08:5x) is later than the file's last modification (08:40:23). Both amendments precede every C outcome. §2's frontier examples partly predate the repair (e.g. GR00T L10-50 B + .75 .850 → .852). |
| `ideation_Q{1,2,3}/FINAL.md` | Pilot sections unaffected (no pilot exception episodes). The errata cover the A/B, ablation and frontier headlines. Two post-hoc statements are not listed and are now wrong: the π0.5 L10-500 B + .125 "dip" (.842 → .892) and GR00T L10-500 A + 1/8 below A (.808 → .842). |
| `frontier_final/` | Byte-identical on rerun; SR, IR, Pareto and NI independently re-derived without substantive mismatch. The 7 tolerance-flagged arms are exactly repaired arms (§1.4). |

## Appendix B. Source index

| Group | Arms (`run:arm`) |
|---|---|
| A replicates | π0.5: `r05_ptail:r5t_p_{l10_50,l10_500,sp_50}_tail1uc`, `r04_blind:r4b3_p_sp_500_tail1uc`, and the `r06_paper:…_rep2/_rep3` copies of each. GR00T: `r05_x:r5x_g_{cell}_tail1u`, `r06_paper:r5x_g_{cell}_tail1u_rep{2,3}` |
| B replicates | π0.5: `r05_q1:r5q1_c10_p_{cell}`, `r06_paper:r5q1_c10_p_{cell}_rep{2,3}`. GR00T: `r06_paper:r6p1_c10_g_{cell}`, `r06_paper:r6p1_c10_g_{cell}_rep{2,3}` |
| Ablations | `r06_abl:r6p2_{identity,direct}_{p,g}_{cell}` (16); `r06_abl:r6p2_{stuck,terminal,overtime,no_progress}_{p,g}_{l10_50,sp_50}` (16); `r06_method:r6p5_bmech_{p,g}_{l10_50,sp_50}` (4) |
| Configuration C | calibration `r06_c_cal:r6c_cal_<cell>` (8 × 10 B-val episodes) and `r06_c_cal/cal/<cell>/calibrated/calibration.json`; smoke `r06_c_smoke:r6c_{pi05,groot}_l10_500_C18`; validation `r06_c_validation:r6c_<cell>_{R30,U30,C30,C45 or Cmax}` (sparse) and `_{R18,U18,C18}` (dense) |
| Frontier completion | `r06_frontier:*` (41 arms), plus every eligible R2–R6 arm listed in `frontier_final/frontier_points.csv` |
| Pure references | `r04_cost:r4f_p_{l10,sp}_inf_k10_L10`, `r04_cost:r4f_p_{l10,sp}_inf_s{1001,2001}`, `r05_q2:r5q2_g_{l10,spatial}_policy_L10`, and DUAL references via `frontier_final/outcomes.json` |
| Pilot (frozen analyses only) | `r06_p3_pilot` (216 arms); outputs in `ideation_Q1/pilot_all_cells/`, `ideation_Q2/final_analysis/preregistered/`, `ideation_Q3/final_pilot/` |
| This report's outputs | `rounds/r06/analysis_r6/`: `repair_*`, `headline.*`, `ablations.*`, `c_*` (incl. `c_decision_audit.json`, `c_validation_regenerated.md`), `frontier_check.json`, `frontier_tables.md`, `frontier_rerun/`, `prereg_checks.json`, `gate_posthoc.json`; scripts in `rounds/r06/analysis_scripts/` |
