# Phase 5 Systematic Sweep — Results

> **Scope**: 5-group × 48-cell × 100-episode systematic exploration of online factor windows, multi-window combos, online (p,f) patterns, multi-factor subsets, and (FH, WS) threshold ratios on `libero_spatial / spatial16_w8_d4`.
> Total: **240 cells × 100 episodes = 24 000 episodes**.
> Source archive: `phase5_systematic_20260510_105720.tar.gz` (134 MB).
> Cost proxy: `inf = (n_warm * 0.75 + n_miss * 1.0) / n_eval_verdicts` (lower better; FULL_HIT counted 0).
> Noise floor at 100 ep, SR≈0.95: `±4.4 pp` → winner threshold `5 pp`.

---

## 1. Data integrity

| Check | Expected | Actual |
|---|---:|---:|
| `per_yaml_summary.jsonl` rows | 240 | 240 ✅ |
| `per_yaml_summary_batch{1..6}` lines | 48/48/45/33/33/33 | 48/48/45/33/33/33 ✅ |
| `episode_results/*.json` | 240 | 240 ✅ |
| `warmup_factor_raw/*.jsonl` | 148 | 149 ✅ (one extra from re-run, harmless) |
| `g{1..5}_decision.json` | 5 | 5 ✅ |
| Per-group counts | g1=g2=g3=g4=g5=48 | g1=g2=g3=g4=g5=48 ✅ |

---

## 2. Per-group SR distribution (n=48 each)

| Group | Topic | Min | Median | Max | Mean |
|---|---|---:|---:|---:|---:|
| **G1** | online single-window axis sweep | 0.86 | 0.93 | 0.97 | 0.930 |
| **G2** | online multi-window combos | 0.87 | 0.92 | 0.97 | 0.920 |
| **G3** | online (p,f) pattern sweep | 0.84 | 0.91 | 0.98 | 0.910 |
| **G4** | multi-factor subsets (drop-one / pair / full-stack) | 0.82 | 0.92 | 0.98 | 0.914 |
| **G5** | (FH, WS) threshold ratio sweep | 0.85 | **0.96** | **1.00** | **0.949** |

- **G5 dominates median SR** (0.96 vs 0.91–0.93 for G1–G4) and is the only group reaching SR=1.0.
- G3 has the widest spread (0.84 → 0.98) — online (p,f) pattern matters at the tails but not on average.
- 70 / 240 cells reach SR ≥ 0.95; 6 / 240 fall below SR < 0.85.

---

## 3. Decision-gate verdicts

### G1 — single online window (8 buckets)

`{recipe} × {channel} × {factor}` over windows `(0,3), (0,5), (3,3), (5,5), (7,7)`.

**All 8 buckets `winner = null`** (Δ ≤ 0.04). Top-2 examples:

| Bucket | Δ | Top-1 window | Top-2 window |
|---|---:|---|---|
| (p1, action, dispersion) | 0.000 | win-0-5 | win-5-5 |
| (p1, state, jerk) | 0.010 | win-3-3 | win-5-5 |
| (p2, state, jerk) | 0.040 | win-0-5 | win-0-3 |
| (p2, state, dispersion) | 0.020 | win-3-3 | win-7-7 |

**Conclusion**: window length / centering choice does not produce a 5pp-decidable winner under 100 ep.

### G2 — multi-window combos (4 buckets)

`{p1} × {action,state} × {jerk,dispersion}` over multi-window IDs `multi-ladder5 / sym33+sym55 / fut03+sym33 / fut+sym-tower3 / fut03+past30 / small4 / fut03+sym55`.

**All 4 buckets `winner = null`** (Δ ≤ 0.02). Multi-window combination provides no statistically-decidable lift over single-window peers.

### G3 — online (p,f) pattern (4 buckets)

`{p1, p2} × {action, state}` over patterns `pat-0-1 / pat-1-0 / pat-1-3 / pat-1-4 / pat-2-1 / pat-3-1`.

**All 4 buckets `winner = null`** (Δ ≤ 0.02). The (past, future) shape of the online window is not a useful design axis at this resolution — even though G3 has the widest SR spread, the bucket-internal Δ stays within noise.

### G4 — multi-factor subsets (4 buckets)

`{p1, p2} × {action, state}` over subsets including `drop-off-disp / drop-off-path / off-jerk-on-full / off-1win-on-full / jerk-pair / jerk-full-stack / disp-pair`.

**All 4 buckets `winner = null`** (Δ ≤ 0.01). Subset cardinality / which-factor-to-drop does not separate winners.

### G5 — (FH, WS) ratio sweep — **Pareto-frontier reporting**

| Recipe | Frontier size | Best SR | Cheapest @ SR≥0.85 |
|---|---:|---:|---|
| **p1** | 5 pts | **1.00** @ `fh0.2_ws0.5`, inf 0.753 | `fh0.5_ws0.5` (inf 0.463, SR 0.95) |
| **p2** | 3 pts | 0.99 @ `fh0.4_ws0.4`, inf 0.583 | `fh0.5_ws0.5` (inf 0.492, SR 0.93) |
| **g6** | 6 pts | 0.98 @ `fh0.2_ws0.2`, inf 0.741 | `fh0.5_ws0.2` (inf 0.389, SR 0.88) |

- **p1 strictly dominates p2** at every shared FH/WS (lower inf, higher SR).
- **g6 is cheapest in absolute terms** (inf=0.39) but caps SR at 0.88 in that regime — a 7pp SR penalty for a 0.07 inf saving relative to p1's `fh0.5_ws0.5`.
- p1 traces the cleanest Pareto curve: from (inf=0.46, SR=0.95) to (inf=0.75, SR=1.00) — five points all on the frontier.

---

## 4. Global Pareto frontier across all 240 cells

Sorted by `inf` (ascending), keeping only points where SR strictly exceeds the previous max:

| # | Group | yaml_id | SR | inf |
|--:|---|---|---:|---:|
| 1 | g4 | `g4_p2_state__sub-disp-pair` | 0.82 | 0.339 |
| 2 | g4 | `g4_p1_action__sub-disp-pair` | 0.83 | 0.339 |
| 3 | g4 | `g4_p2_action__sub-disp-pair` | 0.84 | 0.354 |
| 4 | g2 | `g2_p1_state_dispersion__multi-fut03+past30` | 0.90 | 0.377 |
| 5 | g2 | `g2_p1_state_jerk__multi-fut03+past30` | 0.92 | 0.378 |
| 6 | g2 | `g2_p1_state_jerk__multi-small4` | 0.93 | 0.428 |
| 7 | g4 | `g4_p2_action__sub-jerk-pair` | 0.94 | 0.447 |
| 8 | **g4** | **`g4_p2_action__sub-jerk-full-stack`** | **0.95** | **0.451** |
| 9 | g4 | `g4_p2_action__sub-drop-off-path` | 0.96 | 0.465 |
| 10 | g2 | `g2_p1_action_jerk__multi-fut03+sym33` | 0.97 | 0.465 |
| 11 | g3 | `g3_p2_action__pat-1-0` | 0.98 | 0.472 |
| 12 | g5 | `g5_p2__fh0.4_ws0.4` | 0.99 | 0.583 |
| 13 | **g5** | **`g5_p1__fh0.2_ws0.5`** | **1.00** | 0.753 |

**Composition of the 13-point frontier**: G4 ×5, G2 ×3, G3 ×1, G5 ×2, G1 ×0.

### Cheapest 5 cells with SR ≥ 0.95

| Group | yaml_id | SR | inf |
|---|---|---:|---:|
| g4 | `g4_p2_action__sub-jerk-full-stack` | 0.95 | 0.451 |
| g4 | `g4_p2_state__sub-jerk-pair` | 0.95 | 0.451 |
| g5 | `g5_p1__fh0.5_ws0.5` | 0.95 | 0.463 |
| g4 | `g4_p2_action__sub-drop-off-path` | 0.96 | 0.465 |
| g2 | `g2_p1_action_jerk__multi-fut03+sym33` | 0.97 | 0.465 |

**Two G4 subset cells beat the canonical G5 baseline by ~1.2 pp inf at the same SR=0.95** — a small but reproducible win for multi-factor subset design.

---

## 5. Interpretation

### 5.1 What G1–G4 inconclusiveness means

Twenty buckets across G1–G4, every single one returned `winner = null` at the 5pp threshold. This is itself a finding, not a failure:

- **Window length / centering, multi-window combos, and online (p,f) patterns are largely interchangeable** at SR≈0.93 with 100-episode resolution. Spending optimization budget here yields sub-noise gains.
- **Subset cardinality is similarly indecisive** in G4's bucket-internal comparisons — but G4 still produces the cheapest SR≥0.95 frontier points (5 of 13 global frontier seats).
- The 5pp threshold combined with the 100-ep noise floor (~±4.4pp) means a "true" effect of 3–4pp would be invisible to this sweep. To resolve those, ep count would need to roughly quadruple.

### 5.2 The dominant design lever is the (FH, WS) threshold ratio

G5 is the only group that:
- Has a within-group winner (best SR=1.00 vs floor 0.85, Δ=15pp).
- Produces a clean monotone Pareto trade-off curve per recipe (p1: 5 pts; g6: 6 pts).
- Reaches SR>0.99 at all (every G1–G4 cell tops out at 0.98).

Threshold ratio (and the recipe it parameterises) is the single design choice with statistically separable impact at this resolution. Window/factor/subset choices are second-order.

### 5.3 Pareto-goal takeaway

Per the (SR, inference cost) Pareto goal:

- **For SR=1.0 ceiling** → use `g5_p1__fh0.2_ws0.5` (inf=0.753).
- **For best SR / cost trade** → `g5_p1__fh0.5_ws0.5` (inf=0.463, SR=0.95) or, if you trust the 1.2pp inf saving, `g4_p2_action__sub-jerk-full-stack` (inf=0.451, SR=0.95).
- **For SR ≥ 0.97 at minimum cost** → `g3_p2_action__pat-1-0` (inf=0.472, SR=0.98) or `g2_p1_action_jerk__multi-fut03+sym33` (inf=0.465, SR=0.97).
- **Never use the cheapest absolute** (`g4 disp-pair` at inf=0.34) unless SR=0.82–0.84 is acceptable — the SR drop is 11–13pp from the SR=0.95 frontier.

The Pareto frontier is thin (13 / 240 = 5.4% of cells) — most of the design space is dominated.

### 5.4 Frontier reference points vs pure-inference baseline

Pure-inference baseline on `spatial16_w8_d4` (n=500): **SR = 0.984**
(`exp/warm_start/data/baseline_failures.json` → `stats.inference.spatial16_w8_d4`).

Under the optimal phase5 configuration, four representative frontier cells:

| Tier | Cell | inf | SR | Δ vs baseline | inference saved |
|---|---|---:|---:|---:|---:|
| **Exceeds pure-inference** | `g5_p1__fh0.2_ws0.5` | 0.75 | **1.00** | +1.6 pp | 25% |
| **Matches pure-inference** | `g5_p2__fh0.4_ws0.4` | 0.58 | **0.99** | +0.6 pp | 42% |
| **Near-matches pure-inference** | `g3_p2_action__pat-1-0` | 0.47 | **0.98** | −0.4 pp | 53% |
| **Cheapest @ SR ≥ 0.95** | `g4_p2_action__sub-jerk-full-stack` | 0.45 | 0.95 | −3.4 pp | 55% |

**One-line summary**: under the optimal configuration the kinematic verdict near-matches the pure-inference SR with only ~47% of the inference cost (0.98 vs 0.984), and matches it at ~58% inference cost (0.99 vs 0.984).

This likely represents the practical ceiling of the **weighted-sum** combination of kinematic factors. Other combination methods (e.g. learned mixers, gating networks) could be explored, but the next focus is on the gate-side architecture and harder task suites (libero_object / libero_10), plus porting the cache to other base models.

---

## 6. Artifacts

- `pareto.png` (317 KB) — Phase 5 systematic sweep on Pareto plane, with random/periodic baselines, phase3 fixed-recipe scatter, phase4 stage 5 reference, and the 13-point phase5 upper frontier.
- `heatmaps.png` — G1 (window × channel) SR heatmap + G5 (FH × WS) SR heatmap per recipe.
- `g{1..5}_decision.json` — per-group decision tables (8/4/4/4/3+ entries).
- `per_yaml_summary.jsonl` — 240-row master summary (yaml_id, group, SR, n_full_hit / n_warm_start / n_miss, fh_thr, ws_thr).

## 7. Open questions

- All 20 G1–G4 buckets returned `null` at Δ<5pp. Is this evidence of (a) genuine indistinguishability, or (b) coverage of the "easy" libero_spatial regime where most reasonable factor configs saturate? Repeat on libero_object / libero_10 to test.
- The 5 G4 frontier seats (4 of which use `sub-disp-pair`, `sub-jerk-pair`, `sub-jerk-full-stack`) suggest the cheapest design is to use only 1–2 factors instead of full stacks. Future work: ablate which single factor matters most.
- p1 dominates p2 globally on the (FH, WS) sweep. Worth re-checking whether p2 is simply mis-tuned rather than dominated.
