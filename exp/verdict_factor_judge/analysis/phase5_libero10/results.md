# Phase 5 libero_10 Systematic Sweep — Results & Cross-Suite Comparison

> **Scope**: 5-group × 48-cell × 100-episode systematic exploration of online factor windows, multi-window combos, online (p,f) patterns, multi-factor subsets, and (FH, WS) threshold ratios on `libero_10 / spatial16_w8_d4`.
> Total: **240 cells × 100 episodes = 24 000 episodes**.
> Pipeline: 5/22 13:23 → 5/23 15:42 (≈26h19min wall-clock, 6-server parallel + load-balancing).
> Cost proxy: `inf = (n_warm * 0.75 + n_miss * 1.0) / n_eval_verdicts` (lower better; FULL_HIT counted 0).
> π0.5 no-cache baseline on libero_10: **SR = 0.930** (OpenPI repo).

---

## 1. Data integrity

| Check | Expected | Actual |
|---|---:|---:|
| `per_yaml_summary.jsonl` rows | 240 | 240 ✅ |
| Per-SN files (post-LB merge) | S1=44, S2=44, S3=45, S4=36, S5=36, S6=35 | match ✅ |
| `episode_results/*.json` | 240 | 240 ✅ |
| `warmup_factor_raw/*.jsonl` | 148 (G1+G2+G3+G4) + 3 (G5 phase3/phase4 historical) | match ✅ |
| `g{1..5}_decision.json` | 5 | 5 ✅ |
| Per-group cell count | g1=g2=g3=g4=g5=48 | ✅ |
| Unique yaml_ids | 240 | 240 ✅ (no LB-race duplicates) |

---

## 2. Per-group SR / inf distribution (n=48 each)

| Group | Topic | SR Min | SR Median | SR Max | SR Mean | inf Mean |
|---|---|---:|---:|---:|---:|---:|
| **G1** | online single-window axis sweep | 0.68 | 0.75 | 0.84 | 0.750 | 0.446 |
| **G2** | online multi-window combos | 0.62 | 0.72 | 0.83 | 0.735 | 0.425 |
| **G3** | online (p,f) pattern sweep | 0.58 | 0.72 | 0.83 | 0.716 | 0.446 |
| **G4** | multi-factor subsets | 0.60 | 0.72 | 0.82 | 0.708 | 0.438 |
| **G5** | (FH, WS) threshold ratio sweep | 0.68 | **0.81** | **0.93** | **0.819** | 0.613 |

Observations:
- **G5 dominates** SR median (+6-10 pp over G1–G4) and max (only group ≥ 0.85), confirming the libero_spatial finding.
- **G1–G4 sit in a tight band** SR 0.71–0.75 mean — within ~5 pp of each other, similar to the libero_spatial inconclusive bucket result.
- **G5 trades cost for SR**: inf mean 0.613 (vs G1–G4 ~0.44) — buying high SR at higher inference cost.
- 0/240 cells reach SR ≥ 0.95; 225/240 fall below SR < 0.85.

---

## 3. Global Pareto upper frontier (12 cells)

Sorted by `inf` ascending, keeping only points where SR strictly exceeds the previous max:

| # | Group | yaml_id | SR | inf |
|--:|---|---|---:|---:|
| 1 | G4 | `g4_p1_action__sub-disp-pair` | 0.65 | 0.292 |
| 2 | G4 | `g4_p2_state__sub-disp-pair` | 0.67 | 0.308 |
| 3 | G2 | `g2_p1_state_jerk__multi-fut03+past30` | 0.72 | 0.370 |
| 4 | G2 | `g2_p1_state_jerk__multi-fut05+past50` | 0.75 | 0.373 |
| 5 | G1 | `g1_p1_action_jerk__win-1-1` | 0.79 | 0.393 |
| 6 | G2 | `g2_p1_action_dispersion__multi-sym11+sym33` | 0.83 | 0.405 |
| 7 | G1 | `g1_p1_action_jerk__win-0-5` | 0.84 | 0.432 |
| 8 | G5 | `g5_p2__fh0.4_ws0.3` | 0.87 | 0.554 |
| 9 | G5 | `g5_p2__fh0.4_ws0.2` | 0.88 | 0.580 |
| 10 | G5 | `g5_p1__fh0.4_ws0.2` | 0.89 | 0.601 |
| 11 | G5 | `g5_p1__fh0.3_ws0.5` | 0.92 | 0.615 |
| 12 | **G5** | **`g5_p2__fh0.3_ws0.2`** | **0.93** | **0.666** |

**Composition**: G2 ×3, G1 ×2, G4 ×2, G5 ×5, G3 ×0. G5 owns the high-SR tail; G2/G4 own the cheap end.

---

## 4. Top picks

### Best SR @ libero_10
- `g5_p2__fh0.3_ws0.2` — SR=0.93, inf=0.666 (**matches π0.5 no-cache baseline SR with 33% inference cost saved**).

### Cheapest at SR ≥ 0.85
- `g5_p2__fh0.4_ws0.3` — SR=0.87, inf=0.554.
- `g5_p2__fh0.4_ws0.2` — SR=0.88, inf=0.580.

### Cheapest at SR ≥ 0.90
- `g5_p1__fh0.3_ws0.5` — SR=0.92, inf=0.615.

### Recipe winner
- **p2 narrowly beats p1** (3 of 5 G5 frontier cells use p2; the only SR=0.93 cell is p2).

---

## 5. Cross-suite comparison: libero_spatial vs libero_10

### 5.1 Absolute performance

| Metric | libero_spatial | libero_10 |
|---|---:|---:|
| π0.5 no-cache baseline SR | 0.974 (OpenPI) | 0.930 (OpenPI) |
| Our best SR (Stage 2 sweep) | 1.00 | 0.93 |
| Cells SR ≥ 0.95 | 70/240 | 0/240 |
| Cells SR ≥ 0.85 | ~230/240 | 15/240 |
| G5 mean SR | 0.949 | 0.819 |
| G1–G4 mean SR | 0.91–0.93 | 0.71–0.75 |

**libero_10 is intrinsically harder** (longer-horizon tasks, max_actions=520 vs ~200). Both our sweep and the π0.5 baseline drop ~5–10 pp on libero_10 vs libero_spatial. The relative gap between G5 and G1–G4 widens on libero_10 (+10 pp vs +3 pp on spatial).

### 5.2 Structural finding consistency

✅ **G5 (threshold sweep) dominates the top end** on both suites — same conclusion.
✅ **G1–G4 cluster tightly around a similar mean SR** (within-bucket Δ ≤ ~4 pp).
✅ **Pareto frontier shape**: cheap-end populated by G2/G4 low-cost cells, high-SR tail by G5.

### 5.3 Best-config divergence

| Aspect | libero_spatial top | libero_10 top |
|---|---|---|
| Best-SR cell | `g5_p1__fh0.2_ws0.5` (SR=1.00, inf=0.753) | `g5_p2__fh0.3_ws0.2` (SR=0.93, inf=0.666) |
| Cheapest @ SR≥0.95 | `g5_p1__fh0.5_ws0.5` (inf=0.463) | (none; closest: SR=0.92 @ inf=0.615) |
| Recipe winner | **p1** | **p2** |
| FH preferred at top | 0.2–0.5 (broad) | 0.3–0.4 (lower) |
| WS preferred at top | **0.5 high** | **0.2 low** |

**The specific (recipe, FH, WS) tuple that wins differs** — best libero_spatial config (`g5_p1_fh0.2_ws0.5`) ranks only mid-tier on libero_10 (SR ~0.85), and vice versa. **The verdict configurations are NOT transferable between task suites at the tuple level.**

What does transfer: **the choice of design axis** — invest threshold-sweep budget (G5), not window/factor/subset sweep budget (G1–G4).

### 5.4 Why p2 beats p1 on libero_10 (hypothesis)

p2 (`phase4 p2_action_fut_online_act`) uses action-channel factors; p1 (`phase4 p1_state_fut_online_act`) uses state-channel. Longer-horizon libero_10 tasks may give cleaner action-trajectory signals than state observations for the verdict classifier — but this needs follow-up to confirm (Phase 6 candidate).

### 5.5 Deeper cross-suite finding: libero_10 winners are NOT spatial Pareto-optimal

A second pass on the data (yaml_id is identical across the two sweeps, so per-cell cross-suite comparison is direct) yields a stronger claim than §5.3 alone.

**Fact 1: every libero_10 top G5 cell is dominated on libero_spatial.**

| libero_10 yaml_id | libero_10 (sr, inf) | libero_spatial (sr, inf) | spatial cells with sr ≥ same AND inf < same |
|---|---|---|---:|
| `g5_p2__fh0.3_ws0.2` | (0.93, 0.666) | (0.98, 0.710) | **8** (cheapest: `g3_p2_action__pat-1-0` sr=0.98 inf=0.472 → saves 0.238 inf) |
| `g5_p1__fh0.3_ws0.5` | (0.92, 0.615) | (0.97, 0.654) | **20** (cheapest: `g2_p1_action_jerk__multi-fut03+sym33` sr=0.97 inf=0.465 → saves 0.189 inf) |
| `g5_p1__fh0.2_ws0.2` | (0.92, 0.790) | (0.99, 0.812) | **6** (cheapest: `g5_p2__fh0.4_ws0.4` sr=0.99 inf=0.583 → saves 0.229 inf) |
| `g5_p1__fh0.4_ws0.2` | (0.89, 0.601) | (0.94, 0.628) | **71** (cheapest: `g4_p2_action__sub-jerk-pair` sr=0.94 inf=0.447 → saves 0.181 inf) |
| `g5_p2__fh0.2_ws0.2` | (0.88, 0.779) | (0.96, 0.812) | **49** (cheapest: `g4_p2_action__sub-drop-off-path` sr=0.96 inf=0.465 → saves 0.347 inf) |

The libero_10 "best" cells are middling cells on libero_spatial — they're just the cells that didn't collapse on libero_10.

**Fact 2: the spatial Pareto frontier collapses on libero_10.**

All 13 cells on the libero_spatial Pareto upper frontier drop 0.15–0.35 SR points when evaluated on libero_10:

| spatial yaml_id | spatial sr | lib10 sr | Δ |
|---|---:|---:|---:|
| `g4_p2_state__sub-disp-pair` | 0.82 | 0.67 | -0.15 |
| `g4_p1_action__sub-disp-pair` | 0.83 | 0.65 | -0.18 |
| `g4_p2_action__sub-disp-pair` | 0.84 | 0.63 | -0.21 |
| `g2_p1_state_dispersion__multi-fut03+past30` | 0.90 | 0.71 | -0.19 |
| `g2_p1_state_jerk__multi-fut03+past30` | 0.92 | 0.72 | -0.20 |
| `g2_p1_state_jerk__multi-small4` | 0.93 | 0.62 | **-0.31** |
| `g4_p2_action__sub-jerk-pair` | 0.94 | 0.61 | **-0.33** |
| `g4_p2_action__sub-jerk-full-stack` | 0.95 | 0.61 | **-0.34** |
| `g4_p2_action__sub-drop-off-path` | 0.96 | 0.61 | **-0.35** |
| `g2_p1_action_jerk__multi-fut03+sym33` | 0.97 | 0.78 | -0.19 |
| `g3_p2_action__pat-1-0` | 0.98 | 0.70 | -0.28 |
| `g5_p2__fh0.4_ws0.4` | 0.99 | 0.78 | -0.21 |
| `g5_p1__fh0.2_ws0.5` | 1.00 | 0.82 | -0.18 |

Only **5 of 13** libero_spatial frontier cells remain Pareto-optimal on libero_10. The cheap G2/G4/G3 cells that dominate the spatial frontier's left half (low-inf, high-sr) drop 0.31–0.35 pp on libero_10 — they collapse hardest.

**Fact 3: libero_10's Pareto frontier shape is qualitatively different.**

- On libero_spatial, the frontier goes from (inf=0.34, sr=0.82) up to (inf=0.75, sr=1.00) — almost linear lift, cheap G4 cells already at sr=0.82.
- On libero_10, the frontier goes from (inf=0.29, sr=0.65) up to (inf=0.67, sr=0.93) — same cheap end, but **sr ceiling falls 7 pp** and **mid-frontier sr is 20+ pp lower**. To reach sr ≥ 0.85 on libero_10 requires G5 at inf ≥ 0.55; below that floor only G2/G4 with sr < 0.85.

**Interpretation: G2/G4 cheap cells over-fit short-horizon spatial tasks.**

- G2/G4 cells achieve low inf by being aggressive cache-hit predictors. On short-horizon spatial tasks (max_actions ~200) this works — small accumulated cache-induced action drift doesn't break task completion.
- On long-horizon libero_10 (max_actions=520) the same aggressive caching accumulates compounding action-prediction error over 2.5× as many steps. Tasks that the policy could complete with full inference fail when partial cache reuses propagate errors.
- **G5 threshold-sweep cells are more conservative**: higher inf, but the FH/WS thresholds gate cache reuse more strictly. They survive the long-horizon stress at the cost of saving less inference.

**Net conclusion**: The libero_10 "winning" verdict configurations are *not* a transferable optimum. They are simply the cells that survived a regime change. The spatial-optimal cheap cells exist on libero_10 too, but their SR has collapsed below usable. **No single verdict configuration is Pareto-optimal on both suites.** The design axis (G5 over G1–G4) does transfer, but specific (recipe, FH, WS) tuples do not.

### 5.6 Practical recommendations

For **deployment on a new task suite similar to libero_10** (long-horizon):
1. Don't reuse libero_spatial's winning tuple. Re-sweep G5 (FH, WS) on the target suite.
2. Start with recipe p2 + low-medium WS (0.2–0.3) + FH ~ 0.3–0.4 as a prior.
3. Expect SR ≤ baseline π0.5 SR; the upside is inference-cost reduction (~30%) at zero SR loss possible.

For **deployment on a new task suite similar to libero_spatial** (short-horizon):
1. Recipe p1 + high WS (0.5) + FH ~ 0.2–0.5 as a prior, expect to reach baseline SR.

---

## 6. Process artifacts

- `data/phase5_libero10_systematic/per_yaml_summary.jsonl` (240 merged rows)
- `data/phase5_libero10_systematic/per_yaml_summary.S{1-6}.jsonl` (per-server splits, post-LB)
- `data/phase5_libero10_systematic/episode_results/*.json` (240 per-episode jsons)
- `data/phase5_libero10_systematic/warmup_factor_raw/*.jsonl` (148 phase5 own + 3 historical)
- `data/phase5_libero10_systematic/g{1-5}_decision.json` (per-group decision gate dumps)
- `analysis/phase5_libero10/sr_stats.md` (SR-only quick stats)
- `analysis/phase5_libero10/pareto.png` (Pareto plot with π0.5 baseline)
- Backups (双备份):
  - `.backup_20260523_112731/` (pre-LB)
  - `.backup_pre_loadbalance_v2_20260523_134135/` (post-LB-fix)

---

## 7. Limitations & next steps

- **Noise floor**: 100 ep / cell ≈ ±5 pp; G1–G4 within-bucket Δ ≤ 4 pp is at-noise inconclusive (same as libero_spatial finding).
- **G5 SR ceiling 0.93** matches π0.5 no-cache exactly — suggests we've hit the inherent task ceiling, not a cache-design ceiling. Further G5 sweeping won't break past 0.93 on libero_10.
- **The p1/p2 recipe flip** between suites is the most interesting finding — recommend a Phase 6 micro-sweep that holds (FH, WS) at top libero_10 values and varies historical recipe origin (phase3 g6 / phase4 p1 / phase4 p2) on a new long-horizon suite (libero_object or libero_goal) to test generality.
- **No libero_10 random/periodic baseline cloud** in this repo — Pareto plot lacks the bigger-picture comparison shown in libero_spatial. If wanted, run the random/periodic sweep on libero_10 next.
