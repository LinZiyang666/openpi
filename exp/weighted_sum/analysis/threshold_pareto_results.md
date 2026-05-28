# Threshold-pareto experiment — results

> Sweep target-fraction thresholds on top of `weighted_sum` retrieval scores to
> map the (inference_ratio, success_rate) Pareto frontier. Compare 4 base
> configs (best per-depth winners from the wsweep experiment: d1 / d3 / d4 / d5)
> against the verdict family (kinematic verdict, random/periodic baseline).

## TL;DR

- **33,200 episodes** across **332 eval yamls** (4 bases × 83 triangular-grid
  cells × 100 ep), 16/48 dual-server (ziyang10 + xuanle), ~3 h wall time.
- **Mean SR**: d1 93.8% > d3 92.6% > d4 91.8% > d5 89.9% — **trajectory does
  NOT improve mean SR** (hypothesis reversed vs the trajectory-search
  experiment).
- **Frontier diversity**: d5 has **12** Pareto points, d1 only **5** — the
  wider trajectory-warmup score distribution yields more *distinct* operating
  points along the frontier even though absolute SR is lower.
- **All four bases hit SR = 100% at low inference_ratio** — selectively
  replaying only the top-score requests *beats* the always_hit ceiling
  (0.74 / 0.72) by routing the hard cases to MISS.
- **Hit-type mix across all 332 cells**: FH 47.8% / WS 13.0% / MISS 39.2%.

---

## 1. Design philosophy — two distinct mechanisms compared

The three frontiers in `pareto_combined.png` come from **two fundamentally
different designs**, not one shared design.

### 1.1 Random / periodic baseline — gate-level skip, no signal at all

This is the **no-signal comparison baseline**. It operates at the *gate* layer
(decide whether to even search the cache), not the verdict layer (decide
FH / WS / MISS once a search has run). When the gate skips, the orchestrator
returns MISS without running search, so the request falls through to real
inference.

Two gate variants, both stateless w.r.t. retrieval / kinematics / any learned
signal:

- **`PeriodicGate(cache_len=K, inference_len=N)`** — per episode, the gate
  alternates `K` cache-search calls (becoming FULL_HIT under an `always_hit`
  judge) with `N` forced skips (= forced MISS, real inference). Cycle repeats
  until episode end; `on_episode_start` resets the counter so every episode
  begins with a cache block.
- **`RandomGate(p_inference=p)`** — independent Bernoulli draw at every step:
  with probability `p` skip the cache (forced inference); otherwise search.

The user-settable knobs are the **timing parameters** (`cache_len`,
`inference_len`, or `p_inference`), not bucket fractions. Sweeping these
across yaml configs traces a Pareto frontier in the (inference_ratio, SR)
plane: the best you can do without any signal at all, just by positional or
random skip patterns. Any signal-driven method that fails to clear this
frontier is providing no real information.

Source: `src/openpi/cache/components/gate.py:116` (RandomGate) and `:182`
(PeriodicGate); sweep results aggregated to
`exp/random_periodic_gate/analysis/aggregate.csv`.

### 1.2 The verdict family — ratios as the user knob, thresholds solved from warmup

The kinematic verdict (verdict_factor_judge) and this threshold-pareto share
a different design pattern:

> **The user-settable parameters are the per-bucket target RATIOS, not the
> raw numerical thresholds. The actual numerical thresholds are SOLVED from
> the warmup signal distribution using those ratios.**

```
warmup (force-MISS) → signal distribution along the true policy trajectory
                    ↓
           solve thresholds at quantile cuts of that distribution:
              T_fh = quantile(1 − f_FH)
              T_ws = quantile(1 − f_FH − f_WS)
                    ↓
           eval with judge = threshold(T_fh, warm_tier(T_ws, start_t=0.5))
```

The ratios `(f_FH, f_WS)` say "I want the top `f_FH` fraction of the warmup
signal distribution to be FULL_HIT, the next `f_WS` fraction to be
WARM_START, and the rest MISS". This decouples the configuration from the
underlying signal scale: two bases with very different signal distributions
(e.g. d1's narrow std 0.023 vs d5's wide std 0.094) configured with the same
`(f_FH, f_WS)` target the same operating point even though their solved
numerical thresholds differ substantially. Cross-base comparison is fair
because the *target* is the same.

The two verdict-family methods differ only in **which signal they
threshold**:

#### 1.2.1 kinematic verdict (verdict_factor_judge)
Signal = **kinematic-stability factors** (motion smoothness / predictability of
the current state, decoupled from retrieval). 240 systematic cells across 5
groups: G1 single-window, G2 multi-window, G3 weight pattern, G4 multi-factor
subset, G5 threshold grid.

Hypothesis: smooth, predictable motion is cache-safe regardless of retrieval
quality.

Source: `exp/verdict_factor_judge/data/phase5_systematic/per_yaml_summary.jsonl`.

#### 1.2.2 threshold-pareto (this experiment)
Signal = the **retrieval score itself** (`weighted_score_sum_knn` top-1 after
modality aggregation and trajectory weighting).

Hypothesis: a high-confidence retrieval match is by itself a sufficient cache
safety signal — the better the score, the safer to replay the cached action.

The two verdict-family methods (1.2.1 and 1.2.2) plug into the same
warmup→solve→eval pipeline; only the signal whose quantiles are cut differs.
The random/periodic baseline (1.1) sits orthogonal to this pipeline — a
no-signal positional/random skip pattern that serves as the floor any
signal-driven method must clear.

---

## 2. Setup

### 2.1 Four base configs
The best per-depth winners from the wsweep experiment (cp1_spatial_pool_16,
`weighted_score_sum_knn`, `gate=always_search`, `write_policy=never`). All
selected on jupyter-ziyang10 (H200 NVL) for cross-experiment comparability.

| base | weights (v0 / v1 / rs) | trajectory_depth | wsweep always_hit SR |
|------|---|---|---|
| d1 | 0.06 / 0.50 / 0.43 | 1 (no trajectory) | 0.74 |
| d3 | 0.31 / 0.12 / 0.56 | 3 | 0.72 |
| d4 | 0.56 / 0.18 / 0.25 | 4 | 0.72 |
| d5 | 0.31 / 0.06 / 0.62 | 5 | 0.72 |

### 2.2 Triangular `(f_FH, f_WS)` grid
- `f_FH ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80}` (16 values, step 0.05)
- `f_WS ∈ {0.05, 0.10, 0.15, 0.20, 0.30, 0.40}` (6 values)
- Constraint `f_FH + f_WS ≤ 0.9` (leave a MISS margin; avoid the degenerate
  bottom tail where the quantile cuts collapse).
- → **83 feasible cells per base**, **332 total eval yamls**.
- Solver: quantile. `start_t = 0.5` (WARM start time fixed).

### 2.3 Pipeline
1. **Warmup** (one phase, ~10 min): 4 base yamls × 100 ep with
   `judge = threshold(2.0)` (force-MISS so the robot follows the true policy
   trajectory) and `gate = always_search` (so the search still runs and the
   score is recorded per step). Collects per-step `cp1_score` along the raw
   policy trajectory. After the orchestrator fix below, 8,695 valid scores
   were collected.
2. **Front-gate**: per-base `distribution_spread` check. All four base
   distributions had non-degenerate spread; all 83 cells per base solvable
   (zero degenerate skips for any base).
3. **Solve thresholds** per base. Quantile cuts of the per-base warmup
   distribution.
4. **Eval** (the long phase, ~3 h): 332 yamls × 100 ep = 33,200 ep with the
   solved `ThresholdJudge`.
5. **Analyze**: per-yaml SR + inference_ratio → Pareto frontier per base, plus
   a combined envelope across all 4 bases for comparison with the verdict
   frontiers.

### 2.4 Infrastructure
- **Dual server**: ziyang10 (1 replica, 16 workers) + xuanle (3 replicas, 48
  workers).
- **Capacity-aware** `assign_servers` in `ConductorDriver`: yaml-to-server
  placement balanced by `weight / capacity`, so with 16/48 worker counts the
  ~332 eval yamls split ~83 / 249 between the two servers → both finish at
  approximately the same time. Configured via `run_phase2 --server-workers "16,48"`.
- Eval observed throughput: **~1.9 ep/s** combined, err=0 throughout the run.

### 2.5 Bug fix discovered and applied during this experiment
The original force-MISS warmup collected **0 valid `cp1_score` values** (all
`None`). Root cause: `orchestrator.check()` (`src/openpi/cache/orchestrator.py:522`)
returned `CheckResult(hit_type=MISS, query_keys=...)` on the post-judge MISS
path **without** `score` or `entry_id`, even though both were available
(`top_score = results[0].score` on line 480, already logged). The interceptor's
`_build_hit_meta` therefore saw a None-equivalent payload and emitted
`cp1_score=None` on the wire.

Fix: append `score=top_score, entry_id=winner_id` to that `CheckResult`,
mirroring the existing WARM-downgrade-MISS return at line 511 which already
returns the score. After the fix, the next 400-ep warmup collected **8,695
non-null scores** (one per cache request).

---

## 3. Results

### 3.1 Per-base summary

| base | n cells | mean SR | max SR | min SR | min inf_ratio | Pareto front pts |
|------|---|---|---|---|---|---|
| d1 | 83 | **93.8%** | 100.0% | 85.0% | 0.237 | **5** |
| d3 | 83 | 92.6% | 100.0% | 78.0% | 0.273 | 9 |
| d4 | 83 | 91.8% | 100.0% | 76.0% | 0.291 | 10 |
| d5 | 83 | 89.9% | 100.0% | 73.0% | 0.259 | **12** |

### 3.2 Total hit-type distribution (332 yamls × 100 ep)
| hit type | count | share |
|---|---|---|
| FULL_HIT | 391,436 | 47.8% |
| WARM_START | 106,364 | 13.0% |
| MISS | 321,194 | 39.2% |

### 3.3 Combined Pareto envelope
Pooling all 332 cells from the 4 bases yields a **6-point outer envelope**
(the "best of threshold" achievable at each inference_ratio).

---

## 4. Representative Pareto-frontier configurations

Each row is one yaml on its base's frontier. `(f_FH, f_WS)` are the
*user-set ratios*; `(T_fh, T_ws)` are the *solved thresholds* from that base's
warmup score distribution; `(inf, SR)` are realised on 100 eval episodes.

### 4.1 d1 — 5 frontier points (of 83)

| f_FH | f_WS | T_fh | T_ws | inf_ratio | SR | FH / WS / MISS |
|---|---|---|---|---|---|---|
| 0.75 | 0.15 | 0.9753 | 0.9626 | 0.237 | 87.0% | 1898 / 227 / 436 |
| 0.75 | 0.10 | 0.9753 | 0.9689 | 0.240 | 95.0% | 1893 / 168 / 485 |
| 0.40 | 0.40 | 0.9834 | 0.9728 | 0.407 | 98.0% | 1184 / 826 / 335 |
| 0.35 | 0.20 | 0.9840 | 0.9812 | 0.520 | 99.0% | 1023 / 453 / 892 |
| 0.10 | 0.20 | 0.9856 | 0.9843 | 0.748 | 100.0% | 453 / 399 / 1338 |

### 4.2 d3 — 9 frontier points (of 83)

| f_FH | f_WS | T_fh | T_ws | inf_ratio | SR | FH / WS / MISS |
|---|---|---|---|---|---|---|
| 0.80 | 0.10 | 0.9654 | 0.9522 | 0.273 | 78.0% | 1920 / 175 / 607 |
| 0.80 | 0.05 | 0.9654 | 0.9616 | 0.276 | 83.0% | 1946 / 46 / 712 |
| 0.70 | 0.15 | 0.9705 | 0.9616 | 0.288 | 84.0% | 1837 / 178 / 629 |
| 0.65 | 0.15 | 0.9721 | 0.9654 | 0.308 | 87.0% | 1731 / 186 / 653 |
| 0.65 | 0.05 | 0.9721 | 0.9705 | 0.309 | 90.0% | 1744 / 51 / 748 |
| 0.60 | 0.05 | 0.9735 | 0.9721 | 0.314 | 93.0% | 1678 / 64 / 729 |
| 0.45 | 0.15 | 0.9769 | 0.9735 | 0.371 | 96.0% | 1438 / 241 / 704 |
| 0.10 | 0.40 | 0.9811 | 0.9759 | 0.478 | 99.0% | 1099 / 588 / 698 |
| 0.15 | 0.40 | 0.9807 | 0.9747 | 0.653 | 100.0% | 538 / 881 / 767 |

### 4.3 d4 — 10 frontier points (of 83)

| f_FH | f_WS | T_fh | T_ws | inf_ratio | SR | FH / WS / MISS |
|---|---|---|---|---|---|---|
| 0.80 | 0.05 | 0.9716 | 0.9630 | 0.291 | 81.0% | 1855 / 75 / 711 |
| 0.75 | 0.10 | 0.9756 | 0.9630 | 0.300 | 82.0% | 1819 / 123 / 701 |
| 0.65 | 0.10 | 0.9799 | 0.9756 | 0.316 | 84.0% | 1771 / 62 / 779 |
| 0.65 | 0.15 | 0.9799 | 0.9716 | 0.327 | 90.0% | 1657 / 238 / 657 |
| 0.55 | 0.10 | 0.9830 | 0.9799 | 0.333 | 91.0% | 1679 / 96 / 777 |
| 0.40 | 0.20 | 0.9859 | 0.9816 | 0.414 | 92.0% | 1429 / 240 / 872 |
| 0.35 | 0.10 | 0.9866 | 0.9851 | 0.415 | 94.0% | 1395 / 128 / 916 |
| 0.45 | 0.05 | 0.9851 | 0.9842 | 0.453 | 96.0% | 1337 / 94 / 1056 |
| 0.35 | 0.30 | 0.9866 | 0.9799 | 0.489 | 99.0% | 1122 / 486 / 827 |
| 0.05 | 0.15 | 0.9901 | 0.9884 | 0.855 | 100.0% | 218 / 385 / 1565 |

### 4.4 d5 — 12 frontier points (of 83)

| f_FH | f_WS | T_fh | T_ws | inf_ratio | SR | FH / WS / MISS |
|---|---|---|---|---|---|---|
| 0.80 | 0.10 | 0.9483 | 0.7842 | 0.259 | 76.0% | 1871 / 504 / 320 |
| 0.75 | 0.15 | 0.9589 | 0.7842 | 0.312 | 77.0% | 1741 / 575 / 423 |
| 0.55 | 0.30 | 0.9698 | 0.9011 | 0.318 | 85.0% | 1637 / 493 / 451 |
| 0.60 | 0.05 | 0.9679 | 0.9656 | 0.348 | 87.0% | 1633 / 48 / 843 |
| 0.55 | 0.15 | 0.9698 | 0.9626 | 0.374 | 90.0% | 1505 / 199 / 781 |
| 0.35 | 0.30 | 0.9750 | 0.9656 | 0.426 | 91.0% | 1395 / 302 / 866 |
| 0.30 | 0.15 | 0.9759 | 0.9724 | 0.436 | 93.0% | 1326 / 184 / 921 |
| 0.45 | 0.05 | 0.9724 | 0.9712 | 0.452 | 94.0% | 1326 / 96 / 1040 |
| 0.40 | 0.10 | 0.9739 | 0.9712 | 0.484 | 97.0% | 1238 / 177 / 1071 |
| 0.10 | 0.05 | 0.9787 | 0.9780 | 0.593 | 98.0% | 939 / 79 / 1340 |
| 0.25 | 0.05 | 0.9766 | 0.9759 | 0.605 | 99.0% | 871 / 110 / 1292 |
| 0.05 | 0.15 | 0.9795 | 0.9773 | 0.867 | 100.0% | 203 / 357 / 1633 |

**Reading the tables.** d5's most-aggressive row (`f_FH = 0.80, f_WS = 0.10`)
shows the very low `T_ws = 0.7842`: the trajectory-weighted score distribution
under d5 has a much wider left tail (warmup std 0.094 vs d1's 0.023), so the
`1 − f_FH − f_WS = 0.10` quantile cut lands far below `T_fh`. The same target
ratios mean different real thresholds for different bases — exactly the point
of solving thresholds from per-base warmup distributions instead of hard-coding
numerical thresholds.

---

## 5. Key findings

### 5.1 Hypothesis reversal — trajectory does NOT improve mean SR
The original hypothesis (from the trajectory-search experiment) was that
deeper trajectory aggregation would yield a *better confidence signal* — i.e.
a better Pareto frontier under threshold judging. **Mean SR contradicts this**:

```
d1 (no trajectory)  93.8%  ↑
d3 trajectory=3     92.6%
d4 trajectory=4     91.8%
d5 trajectory=5     89.9%  ↓
```

Adding trajectory depth makes the *mean* operating point on the (inf, SR)
plane worse. The picture is consistent with the wsweep finding that trajectory
hurts strong indices at the SR ceiling — the same penalty carries into the
threshold-judged regime.

### 5.2 But trajectory yields a richer frontier
Despite lower mean SR, d5 has **12 Pareto frontier points** vs d1's **5**.
The wider warmup distribution under deeper trajectory (std 0.094 vs 0.023)
produces more *distinct* operating points along the frontier — finer-grained
control over the (inf_ratio, SR) trade-off, even when absolute SR is lower.

If the practical question is "what threshold gives me exactly inf_ratio ≈ 0.45
with the best possible SR?", d5's denser frontier offers more candidate cells
to choose from.

### 5.3 Threshold judging exceeds the always_hit ceiling
All four bases reach **SR = 100%** at low inference_ratio (e.g. d1 at
`fh=0.10, ws=0.20`: SR 100% at inf = 0.748; d5 at `fh=0.05, ws=0.15`: SR 100%
at inf = 0.867). The wsweep always_hit ceiling for these same bases is only
0.74 / 0.72. **Selectively replaying only the high-confidence requests, and
spending real inference on the rest, beats blanket replay**: the score-based
selectivity dominates the cache hit-rate.

### 5.4 SR span widens with trajectory depth
- d1 SR range: 85.0–100.0% (15 pts; narrow, saturated)
- d3 SR range: 78.0–100.0% (22 pts)
- d4 SR range: 76.0–100.0% (24 pts)
- d5 SR range: 73.0–100.0% (27 pts; widest)

Wider SR span under deeper trajectory mirrors the wider score distribution
spread: trajectory exposes more variance, both as more granular operating
points (good) and as a lower floor at aggressive settings (bad).

---

## 6. Comparison with the verdict family

`exp/weighted_sum/analysis/pareto_combined.png` overlays the three method
frontiers on a single (inference_ratio, success_rate) plane:

| frontier | colour | source | signal |
|---|---|---|---|
| r/p baseline | gray dashed | `exp/random_periodic_gate/analysis/aggregate.csv` (26 pts → 13 front) | none (random / periodic) |
| kinematic verdict | purple solid | `exp/verdict_factor_judge/data/phase5_systematic/per_yaml_summary.jsonl` (240 pts → 13 front) | kinematic-stability factors |
| threshold-pareto envelope | teal solid (circle markers) | union of all 332 cells across 4 bases (→ 6 front) | retrieval score |

`★` markers at `inf ≈ 0` mark each threshold base's wsweep always_hit SR
(0.74 / 0.72 / 0.72 / 0.72).

The full per-base version `pareto_combined_full.png` keeps the 4 individual
threshold frontiers visible alongside the verdict overlay for inspection.

---

## 7. Files

### 7.1 Figures
| file | content |
|---|---|
| `analysis/pareto_combined.png` | clean 3-line overlay (r/p baseline + kinematic + threshold envelope + 4 ★ anchors) |
| `analysis/pareto_combined_full.png` | full per-base + verdict overlay (6 frontiers, scatter clouds visible) |
| `analysis/threshold_pareto_per_base.png` | 2×2 per-base subplots (83 scatter + frontier + always_hit anchor per base) |
| `analysis/threshold_pareto_overlay.png` | 4 base frontiers overlaid on one plane (no verdict comparison) |

### 7.2 Data
| file | content |
|---|---|
| `data/threshold_pareto/eval_journal.jsonl` | 33,200 per-episode terminal records |
| `data/threshold_pareto/eval_per_step.jsonl` | 818,994 per-step records (hit_type + cp1_score) |
| `data/threshold_pareto/eval_results.json` | per-yaml SR (332 entries) |
| `data/threshold_pareto/eval_inf_ratio.json` | per-yaml `(inference_ratio, n_FH, n_WS, n_MISS)` |
| `data/threshold_pareto/warmup_per_step.jsonl` | 8,695 cp1_score values from the 400-ep warmup |
| `data/threshold_pareto/warmup_split/<base>.jsonl` | warmup scores split per base for per-base threshold solving |
| `analysis/threshold_pareto_per_yaml.csv` | 332-row summary (base, ratios, inf_ratio, SR, hit counts) |

### 7.3 Scripts
| file | role |
|---|---|
| `solve_thresholds.py` | warmup score → `(T_fh, T_ws)` per cell (quantile / zscore); triangular grid with `max_total` constraint |
| `emit_threshold_yamls.py` | emit warmup / eval / anchor yamls; CLI `--fh-ratios --ws-ratios --max-total` |
| `summarize.py` | journal → per-yaml SR |
| `summarize_inf_ratio.py` | per_step → per-yaml inference_ratio + hit counts |
| `run_phase2.py` | conductor entry; `--server-workers "16,48"` capacity-aware dual-server |
| `analysis/plot_threshold_pareto.py` | single-plot all-yaml scatter + frontier (legacy) |
| `analysis/plot_threshold_pareto_per_base.py` | 2×2 per-base subplots + overlay + CSV |
| `analysis/plot_threshold_pareto_combined.py` | clean 3-line Pareto vs r/p + kinematic |
| `analysis/plot_threshold_pareto_combined_full.py` | full per-base + verdict overlay (the v1 version) |
| `test_threshold_helpers.py` | unit tests (8/8 pass) for solver + inf_ratio aggregation |
