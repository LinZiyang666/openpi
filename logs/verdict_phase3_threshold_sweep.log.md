# Verdict Phase 3 — Data-driven Threshold Sweep on Phase 2 Layer 1 Gold-Circle Recipes

**Status**: `In Progress` (G1 APPROVED Round 5 — 2026-05-07; §4 Code in progress)
**Level**: L2 — new src/ composer class + config schema + multi-file feature; G1/G2 gates apply per WA §2.1.
**Authority**: Execution
**Owner**: LinZiyang666
**Date**: 2026-05-07
**Related**:
  - Predecessor: [`logs/archive/verdict_factor_judge_phase2_run_commands.log.md`](archive/verdict_factor_judge_phase2_run_commands.log.md) (gold-circle source)
  - Refactor: [`logs/archive/verdict_factor_judge_refactor.log.md`](archive/verdict_factor_judge_refactor.log.md) (4-layer / 17-factor architecture, schema_version=2)

---

## §0 Context

### §0.1 Gate policy

L2 task — both G1 (plan review) and G2 (code review) apply per WA §2.1 / `protocols/execution_authority.md` §10. User initially indicated experiment-level (no review), then reverted to standard review flow on 2026-05-07. Plan submitted to G1 in current state.

src/ changes covered by the gate:
- new composer class `WeightedSumZeroNanComposer` in `src/openpi/cache/components/factors/composers/__init__.py`
- `ComposerConfig` enum extension + new validator branch in `src/openpi/cache/config.py`
- `_build_composer` factory branch

exp/ changes covered by the gate:
- `exp/verdict_factor_judge/phase3_spec.py`
- `exp/verdict_factor_judge/phase3_threshold_solver.py`
- runner extension (`run_phase.py` `--phase3` mode or new `run_phase3.py`)
- `exp/verdict_factor_judge/analysis/plot_pareto_phase3.py`
- enrich pkl invocation script

Tests covered by the gate (WA §3.1 No Dead Code — independent of any gate decision): `tests/cache/test_weighted_sum_zero_nan_composer.py`, `tests/cache/test_phase3_validator.py`, `tests/exp/test_phase3_threshold_solver.py`, `tests/exp/test_phase3_spec.py`.

### §0.2 Goal

For each of the 11 spatial16 gold-circle recipes from Phase 2 Layer 1, sweep a 4×4 grid of (FH_ratio, WS_ratio) ∈ {0.2, 0.3, 0.4, 0.5}², deriving thresholds **from warmup composer-score percentiles** (not hand-set), to map the (FH_thr, WS_thr) → (success_rate, FH%, WS%, MISS%, inference_ratio) surface.

**Why**: Phase 2 Layer 1 fixed FH_thr=0.5 with no WS path. We don't know how SR/inf trade off across the threshold landscape. This sweep gives the response surface.

### §0.3 Scope (in / out)

**In**:
- 1 cfg only: spatial16 (`spatial16_w8_d4`).
- 11 recipes (the gold-circle set; factor configs frozen, identical to Phase 2 Layer 1).
- 16 (FH, WS) cells per recipe → 176 eval yamls + 11 warmup yamls (warmup shared across the 16 cells of a recipe).
- 1 new composer class `weighted_sum_zero_nan` registered into config + factory.
- 1 new threshold-derivation script.
- 1 new phase spec generator.
- enrich existing spatial16 pkl with new-naming offline factor keys (no rebuild).

**Out**:
- clip / max_pool cfgs (deferred until spatial16 results are read).
- topk_action_variance (not in any gold-circle recipe).
- composer alternatives beyond `weighted_sum_zero_nan` (no cross-composer comparison this phase).
- threshold derivation from anything other than warmup composer-score (no held-out, no cross-task transfer).

---

## §1 Recipe Frozen List

### §1.0 Selection rule (reviewer-flagged: source must be explicit)

The 11 "gold-circle" recipes are the **strict-Pareto-positive subset of the 26 spatial16 Phase 2 Layer 1 yamls in the spatial16-cfg panel**, computed by:

```python
# exp/verdict_factor_judge/analysis/plot_pareto.py:163–189
def is_pareto_dominated(inf, sr, baseline_pts):
    return any(bi <= inf and bs >= sr and (bi < inf or bs > sr)
               for bi, bs in baseline_pts)
```

Baseline set (29 points, **spatial16 panel only — not cross-cfg**):
- **26 random / periodic gate points** from `exp/random_periodic_gate/analysis/aggregate.csv` (filter `cfg == "spatial16_w8_d4"`)
- **3 always-WARM points** at `start_t ∈ {0.30, 0.50, 0.70}` with SR = `{0.942, 0.952, 0.976}` and inf = `{0.65, 0.75, 0.85}` (per `WARM_SR["spatial16_w8_d4"]` in plot_pareto.py:111)

A spatial16 yaml is "gold-circle" iff `not is_pareto_dominated(inf_yaml, sr_yaml, baseline)`. Reproducing this against `data/phase2_layer1/spatial16/per_yaml_summary.jsonl` (with inf computed via `warm_cost(0.7) = 0.85`) yields exactly the 11 recipes below (verified 2026-05-07; matches the gold-circle annotations in `phase2_layer1_pareto.png`).

**Distinction from `analysis/phase2_layer1_results.md` §4.1 cross-cfg strict-positive set**: that table reports yamls Pareto-positive in ≥ 2 of 3 cfgs (4 cross-cfg winners) — a stricter, multi-cfg criterion. Phase 3 deliberately uses the **single-cfg (spatial16) panel** Pareto-positive set so we can stress the threshold landscape on the cfg with the most gold-circle data points (11 vs 4–5).

### §1.1 The 11 recipes

11 gold-circle recipes (transcribed from spatial16 panel of `analysis/phase2_layer1_pareto.png`, sorted by SR desc then inf asc, factor configs translated from old yaml to refactor naming via `f1a_a → *_online_action`, `f1a_t → *_online_state`, `f1b_a → *_offline_action`, `f1b_t → *_offline_state`, `dir → direction`, `curv_radius → dispersion`, `cum_disp → path_length`):

| # | recipe id | factor type(s) | windows (P, F) | descriptors |
|---|---|---|---|---|
| 1 | `g1_f1b_t_w_fut_d_all` | `{jerk,direction,dispersion,path_length}_offline_state` | (0,3) (0,5) | jerk, direction, dispersion, path_length |
| 2 | `g2_f1b_t_w_long_risk_d_jerk` | `jerk_offline_state` | (5,5) (7,7) | jerk |
| 3 | `g3_f1b_t_w_long_risk_d_all` | `{4 desc}_offline_state` | (5,5) (7,7) | 4 desc |
| 4 | `g4_f1b_t_w_short_d_jerk` | `jerk_offline_state` | (0,3) (1,1) (3,0) | jerk |
| 5 | `g5_f1a_t_d_jerk_dir_pair` | `{jerk,direction}_online_state` | (3,3) | jerk, direction |
| 6 | `g6_f1a_a_d_jerk_curv_pair` | `{jerk,dispersion}_online_action` | (3,3) | jerk, dispersion |
| 7 | `g7_f1b_a_w_long_risk_d_jerk` | `jerk_offline_action` | (5,5) (7,7) | jerk |
| 8 | `g8_f1a_t_d_curv_only` | `dispersion_online_state` | (3,3) | dispersion |
| 9 | `g9_f1b_t_w_sym_s_d_all` | `{4 desc}_offline_state` | (1,1) (2,2) (3,3) | 4 desc |
| 10 | `g10_f1b_a_w_fut_d_all` | `{4 desc}_offline_action` | (0,3) (0,5) | 4 desc |
| 11 | `g11_f1a_a_d_curv_only` | `dispersion_online_action` | (3,3) | dispersion |

**Cross-recipe stats**:
- offline factor (P, F) union: `(0,3) (0,5) (1,1) (2,2) (3,3) (5,5) (7,7) (3,0)` — **8 windows**
- offline factors used: 8 (= 4 desc × 2 channel) — every refactor offline factor appears at least once
- online factors used: 4 (`jerk_online_state`, `direction_online_state`, `dispersion_online_state`, `dispersion_online_action`) — also `jerk_online_action` from g6, total 5 online factors

**Frozen** = factor type / windows / descriptors / `range:[0.3,0.7]` directions on dispersion / `high` directions on path_length all match Phase 2 Layer 1 verbatim (translated to refactor naming).

---

## §2 New Composer: `weighted_sum_zero_nan`

### §2.1 Spec

```python
class WeightedSumZeroNanComposer:
    """Equal-weight sum, NaN→0 (still counted in denominator), two-tier thresholds.

    Phase 3 sweep composer. Distinct from WeightedSumComposer (NaN-skip
    denominator) and WeightedSumWithWarmFallbackComposer (all-NaN fallback).
    """

    def __init__(
        self,
        weights: dict[str, float],          # treated by behaviour: every key with non-zero weight counts
        full_hit_threshold: float,
        warm_start_threshold: float,        # mandatory in this composer
        warm_start_t: float = 0.5,          # Phase 3 default per user directive
        directions: dict[str, str] | None = None,
    ): ...

    def bind_orientations(self, orientations: dict[str, str]) -> None: ...
        # Same fail-loud check as WeightedSumComposer for non_monotonic + non-zero weight + missing direction.

    def compose(self, factors: dict[str, float], *, winner_id: str) -> JudgeResult:
        # Differences from WeightedSumComposer.compose:
        #   - NaN values contribute 0 (after orientation flip skipped — NaN doesn't flip, just becomes 0).
        #   - Denominator = number of non-zero-weight declared keys (constant per recipe), not sum of "valid" weights.
        #   - No early-exit for total_w==0.
        #   - Two thresholds always present: s>=FH→FULL_HIT; WS<=s<FH→WARM_START@start_t; s<WS→MISS.
        ...
```

### §2.2 Numeric contract

- For declared keys with non-zero weight (= `declared_dependencies`), denominator = `|declared_dependencies|`. Constant per recipe, independent of NaN pattern.
- `contrib(key)`:
  - `v = factors.get(key, NaN)`
  - if `isnan(v)`: `contrib = 0.0`
  - else: orientation flip via existing helpers (`_apply_direction` for non_monotonic; `1 - v` for risky; `v` for safe)
- `s = sum(contrib for key in declared_keys) / |declared_keys|`
- decision: `s ≥ FH_thr → FULL_HIT` ; `WS_thr ≤ s < FH_thr → WARM_START(start_t)` ; `s < WS_thr → MISS`
- `factor_outputs.composer_score` exposes `s` (same schema as the other composers).

### §2.3 Edge cases

The cascade is **inclusive-on-FH, inclusive-on-WS**: `s ≥ FH_thr → FULL_HIT`; else `s ≥ WS_thr → WARM_START`; else `MISS`. Concretely:

- **All-NaN verdict** → `s = 0`. Decision depends on `WS_thr`:
  - `WS_thr > 0` → MISS
  - `WS_thr ≤ 0` → WARM_START (matches the `(FH=0.5, WS=0.5)` cell intent: zero MISS path)
- **`(fh_ratio + ws_ratio) = 1.0` cells** (`(0.5,0.5)`, `(0.4,0.5)`+`(0.5,0.4)`+ `(0.5, 0.5)`-style — actually only `(0.5, 0.5)` is exactly 1.0; the others sum to 0.9): solver §5 returns `WS_thr = min(warmup_scores)`. If any eval verdict has `s < WS_thr` (distribution shift), it goes MISS — that's the experimental signal.
- **`s` sitting exactly at a threshold**: `s == FH_thr` → FH (inclusive); `s == WS_thr` (and `< FH_thr`) → WS (inclusive). Reviewer-flagged matter: the previous draft of this section described `s=0 + WS_thr=0` as MISS, which contradicted the cascade in §2.2. Clarified.
- **`WS_thr > FH_thr`** (impossible by construction in §5 since solver returns descending-ordered cuts) — validator §3.2 nonetheless rejects this configuration at yaml load time.
- Single-key recipes (g4 jerk-only with risky orientation, g8 single dispersion): denominator can be 2 or 3 (multi-window) or 1 (single window). All-NaN windows still count in denominator; per warmup data, this lets us discover whether NaN-heavy long windows in g2/g3/g7 collapse the score below threshold.
- Compared to Phase 2 Layer 1 NaN-skip: **same recipe will produce different `s` distributions**. This is intentional — Phase 3 explores the NaN-=-0 regime; Phase 2 was the NaN-skip regime.

### §2.4 Where it lives

`src/openpi/cache/components/factors/composers/__init__.py` — append after `WeightedSumWithWarmFallbackComposer`. Reuses `_apply_direction`, `_SAFE/_RISKY/_NON_MONOTONIC` constants.

---

## §3 Config Wiring

### §3.1 ComposerConfig dataclass

`src/openpi/cache/config.py` — add new branch to existing `ComposerConfig`:

- New `type` enum value: `weighted_sum_zero_nan`
- Reuses existing fields: `weights`, `tier_thresholds.{full_hit, warm_start}`, `warm_start_t`, `directions`
- No new fields needed — `tier_thresholds.warm_start` already optional in dataclass; for this composer it becomes **mandatory** (validator below).

### §3.2 Validator additions

In `_validate_composite_judge` (or wherever per-composer rules live):
- if `composer.type == "weighted_sum_zero_nan"`:
  - require `tier_thresholds.full_hit` set
  - require `tier_thresholds.warm_start` set
  - require `tier_thresholds.warm_start ≤ tier_thresholds.full_hit`
  - require `warm_start_t` set
  - reject `warm_start_t > 1.0` or `< 0.0`
  - inherit existing CP1-only rule for `warm_start_t` (this composer can WARM_START → CP1 only)
  - reject if `weights` empty after non-zero filter

### §3.3 `_build_composer` factory

Append branch (next to `weighted_sum_with_warm_fallback`):

```python
if cfg.type == "weighted_sum_zero_nan":
    return WeightedSumZeroNanComposer(
        weights=cfg.weights,
        full_hit_threshold=cfg.tier_thresholds["full_hit"],
        warm_start_threshold=cfg.tier_thresholds["warm_start"],
        warm_start_t=cfg.warm_start_t,
        directions=cfg.directions,
    )
```

---

## §4 pkl Enrichment

### §4.0 Canonical pkl path (reviewer-flagged: path divergence between codebase paths)

**Currently the codebase has divergent references for the spatial16 pkl, one of which is broken.**

| Reference | Path | Status |
|---|---|---|
| Phase 2 Layer 1 yamls (e.g. `config/spatial16/phase2_layer1_a/*.yaml`) | `exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl` | exists ✓ |
| `v2_spec.CFG_SPECS["spatial16_w8_d4"]["preload_pkl"]` (line 53) | `exp/warm_start/data/spatial16/cp1_spatial_pool_16.pkl` | **does not exist** ✗ |

Phase 3 locks the canonical path to **`exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl`** (the one that actually exists and is used by Phase 2 Layer 1 yamls). Three downstream changes:

1. **§4 enrich** runs against this canonical path (rewrite + smoke gate).
2. **§6 phase3_spec / v2_spec** must update `CFG_SPECS["spatial16_w8_d4"]["preload_pkl"]` (and clip / max_pool entries by parity, if Phase 3 ever expands cfg) to point at canonical paths under `exp/common/data/cache_artifacts/libero_spatial/`.
3. **Pre-flight verification step** in §12 §7.5 added: assert the canonical pkl exists before launching warmup, and that all generated phase3 yamls' `preload_path` matches.

### §4.1 Why

Existing canonical `cp1_spatial_pool_16.pkl` carries 168 factor keys in **legacy naming** (`f1b_t_jerk__p5_f5`, etc.). Refactor offline extractors read **refactor naming** (`jerk_offline_state__p5_f5`) at `factors/offline.py:114`. Direct read = NaN.

### §4.2 What

Run `python -m exp.common.build_in_memory_cache_artifact enrich-existing-pkl` (subcommand exists from refactor B-step) with a Phase 3 factor yaml listing the 8 offline factors × 8 (P, F) windows = **64 new keys**.

### §4.3 Output policy

- Append new keys alongside existing 168 legacy keys (do **not** remove legacy keys — Phase 2 Layer 1 yamls still reference them and may be re-run for parity).
- New pkl ~ same size + ~10% (single float per entry per key, ~7000 entries).
- Path: in-place overwrite at canonical `exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl`; backup as `*.pre_phase3.bak.pkl` first.

### §4.4 Verification

Post-enrich smoke gate (one-off):
```python
import pickle
d = pickle.load(open("exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl","rb"))
keys = set(d["entries"][0].payload.factors.keys())
expected_new = {f"{desc}_offline_{ch}__p{p}_f{f}"
                for desc in ("jerk","direction","dispersion","path_length")
                for ch in ("action","state")
                for (p,f) in [(0,3),(0,5),(1,1),(2,2),(3,3),(5,5),(7,7),(3,0)]}
assert expected_new.issubset(keys), expected_new - keys
legacy_kept = sum(1 for k in keys if k.startswith("f1b_"))
assert legacy_kept == 168, f"legacy keys count mismatch: {legacy_kept}"
```

---

## §5 Threshold Derivation Script

### §5.1 Path

`exp/verdict_factor_judge/phase3_threshold_solver.py` (new file).

### §5.2 Source-of-truth: offline reconstruction (reviewer-flagged: warmup yaml cannot produce composer_score in current architecture)

**Problem identified by G1 R1**: warmup yaml runs `AlwaysWarmStartJudge + DumpingJudge`. `AlwaysWarmStartJudge` (`judge.py:154–190`) returns `JudgeResult(WARM_START, ...)` directly — it never instantiates a composer, so `inner_factor_outputs.composer_score` is always `None`. The previous draft's solver path was unimplementable.

**Resolution**: solver does **offline reconstruction** of the composer score sequence from the dump JSONL's `factor_raw` column (which IS populated by DumpingJudge — `dumping_judge.py:179–197`). Specifically, for each recipe, the solver:

1. Reads `factor_raw` per (episode, step) from dump JSONL — these are the raw factor values that the recipe's factors would have computed at the corresponding verdict moment.
2. Instantiates the recipe's **Layer 3 calibration** (`PercentileRollingCalibration` with `window_size=50`) and **Layer 4 composer** (`WeightedSumZeroNanComposer` with the recipe's weights / directions, and **placeholder thresholds** that don't affect score computation — only the `score()` not `compose()` path is needed).
3. Replays the warmup verdict sequence in chronological order, feeding each `factor_raw` dict through calibration → composer.score, recording the resulting `s` per verdict.
4. Sorts the resulting score array descending and cuts at the `fh_ratio` / `fh_ratio+ws_ratio` quantile boundaries.

This produces a per-verdict score that is a **draw from the saturated-buffer score distribution** the recipe's composer would emit under eval-state preload, not a point reproduction of any specific eval verdict. The reconstruction reuses the same src/ classes (`PercentileRollingCalibration` + `WeightedSumZeroNanComposer`) that eval uses, so no algorithmic drift; precise equivalence claim is in §5.4 (final paragraph).

### §5.3 Inputs

- Warmup dump JSONL: `exp/verdict_factor_judge/data/phase3/warmup/<recipe>__warmup.jsonl` (one line per verdict, with `factor_raw: dict[str, float]` field).
- Recipe definition (factor list + directions) — same dataclass that the eval yaml uses.
- `(FH_ratio, WS_ratio)` cell from the 16-grid.

### §5.4 Algorithm

Choosing **reviewer Option A** from G1 R2 — saturated-buffer replay through the existing `PercentileRollingCalibration`. Verified against `src/openpi/cache/components/factors/calibrations/percentile_rolling.py` (read 2026-05-07): `__init__(samples, *, window_size)`, `bind_keys(keys)` is one-shot fail-fast (`len(non_nan) >= window_size` per key), `__call__(raw)` is the per-verdict entry point (NOT `calibrate(...)`), and `bind_keys` preloads `non_nan[-window_size:]` per key so the buffer is **always saturated** when the first verdict arrives — no cold-start output path exists.

```python
from openpi.cache.components.factors.base import CalibrationSamples
from openpi.cache.components.factors.calibrations.percentile_rolling import PercentileRollingCalibration
from openpi.cache.components.factors.composers import WeightedSumZeroNanComposer

def reconstruct_scores(
    jsonl_path: Path,
    recipe: PhaseRecipe,        # see §6 — declared keys, weights, directions, orientations
) -> list[float]:
    """Replay warmup verdicts through saturated-buffer calibration + composer.

    Returns one score per JSONL row, in row order. The score is what the
    recipe's composer would have produced if the recipe had been executed
    inline with the eval-state preload.
    """
    rows = list(_iter_jsonl(jsonl_path))

    # 1) Build CalibrationSamples from the warmup factor_raw column itself.
    #    Every declared key must have >= window_size non-NaN samples, else
    #    bind_keys raises — that's the same fail-fast eval will see.
    per_key_history: dict[str, list[float]] = {k: [] for k in recipe.declared_keys}
    for row in rows:
        raw = row["factor_raw"]
        for k in recipe.declared_keys:
            per_key_history[k].append(float(raw.get(k, float("nan"))))
    samples = CalibrationSamples(samples=per_key_history)

    # 2) Saturate buffer (last 50 non-NaN per key) and freeze key set.
    cal = PercentileRollingCalibration(samples, window_size=50)
    cal.bind_keys(recipe.declared_keys)

    # 3) Composer with placeholder thresholds — only _score_only path used.
    composer = WeightedSumZeroNanComposer(
        weights={k: 1.0 for k in recipe.declared_keys},
        full_hit_threshold=0.0,            # placeholder
        warm_start_threshold=0.0,          # placeholder
        warm_start_t=0.5,
        directions=recipe.directions,
    )
    composer.bind_orientations(recipe.orientations)

    # 4) Replay rows in order. cal(raw) appends-and-ranks (line 112 of
    #    percentile_rolling.py); buffer state evolves identically to a real
    #    on-policy run that started with the same preload.
    scores: list[float] = []
    for row in rows:
        calibrated = cal(row["factor_raw"])
        s = composer._score_only(calibrated)
        scores.append(s)
    return scores


def derive_thresholds(scores: list[float], fh_ratio: float, ws_ratio: float) -> tuple[float, float]:
    arr = np.asarray([s for s in scores if not (s is None or np.isnan(s))], dtype=np.float64)
    if arr.size == 0:
        raise RuntimeError("no usable scores in warmup — recipe likely all-NaN")
    arr_desc = np.sort(arr)[::-1]
    n = arr_desc.size
    i_fh = max(0, min(n-1, int(fh_ratio * n) - 1))
    i_ws = max(0, min(n-1, int((fh_ratio + ws_ratio) * n) - 1))
    return float(arr_desc[i_fh]), float(arr_desc[i_ws])
```

**Equivalence claim** (Round 2 reviewer flagged "ambiguous reproduction"): the score this function emits at row `i` is what the recipe's composer would have produced if that warmup verdict had run on a calibration buffer initialized from the **full** warmup history. This is *not* identical to the score eval would emit at any specific eval verdict (eval sees a different raw stream after preload); it *is* a valid sample from the saturated-buffer score distribution under recipe semantics — which is the population whose quantile cuts the solver wants. The Phase 3 claim is therefore "scores are draws from the eval-state score distribution under the recipe", not "scores reproduce a specific eval rollout".

### §5.5 `_score_only` accessor on the composer

Add a small helper to `WeightedSumZeroNanComposer` that returns `s` without the threshold cascade:

```python
def _score_only(self, factors: dict[str, float]) -> float:
    """Pure score computation (NaN→0, equal-weight average over the
    composer's *declared dependencies* — i.e. keys with non-zero weight).

    Aligns with §2.2's contract: denominator = |declared_dependencies|,
    not the count of all keys in the weights dict. Zero-weight keys (if
    ever introduced) are skipped from both numerator and denominator.

    Used by the offline solver to reconstruct the composer score column
    from factor_raw without going through the FH/WS/MISS cascade.
    """
    keys = [k for k, w in self._weights.items() if w != 0.0]    # G1 R2 B2 fix
    if not keys:
        return float("nan")
    total = 0.0
    for k in keys:
        v = factors.get(k, float("nan"))
        if math.isnan(v):
            continue                            # NaN → contrib 0, but skip the orient flip
        ori = self._orientations.get(k)
        if ori == _SAFE:
            total += v
        elif ori == _RISKY:
            total += 1.0 - v
        elif ori == _NON_MONOTONIC:
            total += _apply_direction(self._directions[k], v)
    return total / len(keys)
```

This is part of §2.1 spec. Included in §8.1 unit tests.

### §5.6 Buffer semantics — saturated-only by construction

Reviewer-flagged in G1 R2: `PercentileRollingCalibration` does **not** have a cold-start output path. The previous draft of this section was wrong on that point. Corrected:

- `bind_keys` raises if any declared key has `< window_size = 50` non-NaN samples in the supplied `CalibrationSamples`. The solver therefore **fails loud** if a recipe's warmup data is too sparse for any key — that recipe is skipped (16 cells emit `_NA`), and the runner logs which key was insufficient. Long-window NaN-heavy recipes (g2/g3/g7 with `(7,7)` ≈ 67% NaN over ~420 warmup verdicts) leave ~140 non-NaN per key, comfortably above `window_size=50`. The g4 smoke gate (§8.5) verifies pre-launch that this constraint holds.
- After `bind_keys`, every per-verdict `cal(raw)` call emits a finite percentile per non-NaN input key (NaN input → NaN output, no buffer mutation; `percentile_rolling.py:108–113`). NaN propagates only from a NaN raw — never from buffer underflow.
- Solver's `reconstruct_scores` therefore returns one **finite** score per row whenever the row's `factor_raw` had any non-NaN declared key (the composer's NaN→0 rule means zero per-key contribs sum to a finite score; only an all-NaN raw across all declared keys yields `s = 0`, which is itself finite). `derive_thresholds`'s NaN-filter is defensive — it should never strip rows in practice.
- Score sample count for cuts: 20 episodes × ~21 steps/ep ≈ **420 scores per recipe**. At 0.2 cell granularity, each quantile cut uses ~84 samples — adequate.

### §5.7 Edge cases

- `fh_ratio + ws_ratio == 1.0` (only `(0.5, 0.5)`) → `i_ws = n-1` → WS_thr = min(scores). All warmup verdicts ≥ WS_thr → in warmup, MISS count = 0; in eval, MISS only on score < min(warmup_scores) (distribution shift signal).
- **`bind_keys` insufficient samples** (any declared key has `< window_size = 50` non-NaN samples in the warmup column) → `bind_keys` raises `ValueError` per `percentile_rolling.py:79–83`; solver propagates as a recipe-level skip (`_NA` marker for all 16 cells of that recipe; output JSON has `error` field per §5.8).
- **All-zero-weight composer** (degenerate construction; would not occur for a well-formed recipe) → `_score_only` returns NaN per §5.5; `derive_thresholds` then drops every row in the NaN filter → `arr.size == 0` → raises `RuntimeError("no usable scores ...")`. Same `_NA` propagation.
- Tie-heavy distributions (e.g. dispersion `range:[0.3,0.7]` produces many `s ∈ {0/N, 1/N, 2/N, ...}` clusters) → ranked cut may put unequal counts into FH vs WS. Documented, not patched.

### §5.8 Output

For each recipe, write `phase3/thresholds/<recipe>__thresholds.json`. Schema reflects the saturated-buffer model from §5.6 — there is no "warm-state subset" since the buffer is saturated from row #1; field names below are the canonical ones any §8.3 test fixtures should match (G1 R3 B3 fix).

```json
{
  "recipe": "g4_f1b_t_w_short_d_jerk",
  "warmup_yaml_id": "spatial16_phase3_g4_f1b_t_w_short_d_jerk__warmup",
  "warmup_jsonl": "exp/verdict_factor_judge/data/phase3/warmup/spatial16_phase3_g4__warmup.jsonl",
  "n_warmup_rows": 421,
  "n_finite_scores": 421,
  "n_dropped_nan_scores": 0,
  "score_stats": {"min": 0.020, "max": 0.987, "mean": 0.541, "std": 0.182},
  "per_key_non_nan_count": {
    "jerk_offline_state__p0_f3": 380,
    "jerk_offline_state__p1_f1": 421,
    "jerk_offline_state__p3_f0": 380
  },
  "cells": {
    "fh0.2_ws0.2": {"fh_thr": 0.74, "ws_thr": 0.55},
    "fh0.2_ws0.3": {"fh_thr": 0.74, "ws_thr": 0.50},
    "...": {}
  }
}
```

Field semantics (G1 R4 B2 fix — counts now align with §5.6 / §8.3 #8 saturated-buffer all-NaN-row=0 contract):

- `n_warmup_rows` — total JSONL rows fed to `reconstruct_scores`.
- `n_finite_scores` — count of rows whose `_score_only` output was finite. **For well-formed recipes (any non-zero declared weights) this equals `n_warmup_rows`** — `_score_only` returns finite for every row including all-declared-NaN rows (returns `0/N = 0`, finite). The NaN filter in `derive_thresholds` is purely defensive against the degenerate all-zero-weight construction (which a well-formed recipe never produces; see §5.7).
- `n_dropped_nan_scores` — `n_warmup_rows - n_finite_scores`. Expected `0` for the 11 recipes; non-zero would indicate the degenerate construction and trigger the `error` path (see §5.7 + §8.3 test #14).
- `score_stats` — over the finite-score subset (i.e. essentially all rows for well-formed recipes).
- `per_key_non_nan_count` — diagnostic; per-key non-NaN counts in the warmup JSONL `factor_raw` column. **Each value MUST be `≥ 50` (window_size)** — if any value `< 50`, `bind_keys` would have raised and this file would have an `error` field instead of `cells` (see §5.7 / §8.3 test #14). The example shows `380` for boundary-affected windows (e.g. `(0,3)` and `(3,0)` lose 3 entries per episode chain to boundary NaN — 20 ep × ~21 steps − ~3 boundary × 20 ep ≈ 380); central windows like `(1,1)` lose 2 boundary entries per chain so are essentially full at 421.

---

## §6 Phase 3 Spec Module

### §6.1 Path

`exp/verdict_factor_judge/phase3_spec.py` (new file). Reuses `v2_spec.py` helpers (`factor`, `factor_keys`, `_build_dump_factor_superset`, `build_warmup_yaml`).

### §6.2 Recipe definitions

```python
GOLD_RECIPES = [
    {"id": "g1_f1b_t_w_fut_d_all",
     "factors": [
         factor("jerk_offline_state",        windows=[{"past":0,"future":3},{"past":0,"future":5}]),
         factor("direction_offline_state",   windows=[{"past":0,"future":3},{"past":0,"future":5}]),
         factor("dispersion_offline_state",  windows=[{"past":0,"future":3},{"past":0,"future":5}]),
         factor("path_length_offline_state", windows=[{"past":0,"future":3},{"past":0,"future":5}]),
     ],
     "directions": {
         "dispersion_offline_state__p0_f3": "range:[0.3,0.7]",
         "dispersion_offline_state__p0_f5": "range:[0.3,0.7]",
         "path_length_offline_state__p0_f3": "high",
         "path_length_offline_state__p0_f5": "high",
     }},
    # ... 10 more, each transcribed from §1
]

GRID = [(fh, ws) for fh in (0.2, 0.3, 0.4, 0.5) for ws in (0.2, 0.3, 0.4, 0.5)]
```

### §6.3 Two-pass yaml emit

Pass 1 (pre-warmup): for each recipe emit:
- `<cfg>_phase3_<recipe_id>__warmup.yaml` (uses `AlwaysWarmStartJudge + DumpingJudge` over the recipe's full dump-factor superset, just like phase2 layer1)
- A **placeholder** eval yaml per cell: `<cfg>_phase3_<recipe_id>__fh<FH>_ws<WS>.yaml` with `tier_thresholds.full_hit = NULL, warm_start = NULL, warm_start_t = 0.5` — the runner injects values at Pass 2.

Pass 2 (post-warmup): runner script reads `phase3/<recipe>__thresholds.json` and patches the 16 placeholder yamls.

> **§6.3 Deviation (recorded for G2 R1 B4)**: implementation chose **emit-on-demand** instead of two-pass placeholder emit. `phase3_spec.main()` writes only the 11 warmup yamls + manifest; `run_phase3.py` calls `build_eval_yaml_for_cell(cfg, recipe, fh_thr, ws_thr)` per cell at the moment of execution to produce a final eval yaml directly under `eval_yaml_dir/`. Reason: the new composer's validator (§3.2) requires `tier_thresholds.full_hit` and `tier_thresholds.warm_start` mandatory, so a placeholder yaml with `null` thresholds cannot pass `load_cache_config`; supporting placeholder + patch would require either making thresholds `Optional` in the dataclass (loosens the schema) or routing patches through a yaml-string editor (extra surface, no value). Final on-disk artefact set is identical (11 warmup + 176 eval yamls), only the write timing differs. Recorded here so operators do not look for 176 placeholder files between the spec and runner steps.

### §6.4 warmup_trials

Inherit Phase 2 default `warmup_trials = 2 → W = 2 × 10 task = 20 episodes`.

`run_phase.py` already supports this. No change needed.

---

## §7 Runner Orchestration

### §7.1 Path

Extend `exp/verdict_factor_judge/run_phase.py` with a `--phase3` mode (or new file `run_phase3.py` if cleaner — decide at impl time).

### §7.2 Flow per recipe

1. Load warmup yaml + warmup buffer config (server-side ws ctrl, same as Phase 2).
2. Run warmup (20 ep) with DumpingJudge — dump JSONL written to `exp/verdict_factor_judge/data/phase3/warmup/<recipe>__warmup.jsonl`.
3. Call `phase3_threshold_solver.solve(<jsonl>, GRID)` → write `phase3/<recipe>__thresholds.json`.
4. Patch 16 placeholder yamls with derived thresholds.
5. For each of 16 cells: load eval yaml + run eval (100 ep / 10 task × 10 trial as Phase 2) → emit `phase3/eval/<recipe>__fh<FH>_ws<WS>__results.json`.
6. Write `phase3/<recipe>__per_yaml_summary.jsonl` (16 rows).

### §7.3 Single-server, single-cfg

- spatial16 only → 1 server / 1 client.
- Warmup serial across 11 recipes (~4-5h on one GPU server).
- Eval serial too: 11 × 16 = **176 yamls × 100 ep**, ~80h on one server. **Recommend bringing 4-6 servers up** (cap at 6 per `_current_bundle` global race precedent in archived phase2 run-commands log).

### §7.4 Output tree

```
exp/verdict_factor_judge/data/phase3/
├── warmup/
│   ├── g1_f1b_t_w_fut_d_all__warmup.jsonl
│   └── ... (11 files)
├── thresholds/
│   ├── g1_f1b_t_w_fut_d_all__thresholds.json
│   └── ...
├── eval/
│   ├── episode_results/
│   │   └── spatial16_phase3_g4_f1b_t_w_short_d_jerk__fh0.3_ws0.2.json
│   │   └── ... (176 files)
│   └── per_step/ (optional, large)
└── per_yaml_summary.jsonl (176 rows, master file)
```

---

## §8 Test Plan

### §8.1 Unit tests (src/)

`tests/cache/test_weighted_sum_zero_nan_composer.py` (new):

**Core behaviour**:
1. Equal-weight sum, all keys numeric → s = mean(orient(v))
2. NaN keys → contrib 0, denom unchanged (matches §2.2 fixed denominator)
3. directions: dispersion `range:[0.3,0.7]` binary contribute, path_length `high` linear
4. WS emit: start_t=0.5 attached on WS path
5. bind_orientations: missing direction on non_monotonic non-zero-weight → ValueError
6. CompositeJudge integration: feed via existing CompositeJudge harness, verify factor_outputs.composer_score = s

**Boundary semantics** (added per G1 R2 B3):
7. `s == FH_thr` exact → FULL_HIT (inclusive-on-FH)
8. `s == WS_thr` exact, `s < FH_thr` → WARM_START (inclusive-on-WS)
9. All-NaN raw + `WS_thr > 0` → s=0 < WS_thr → MISS
10. **All-NaN raw + `WS_thr = 0` → s=0 ≥ 0 → WARM_START** (the `(0.5,0.5)` cell zero-MISS intent)
11. All-NaN raw + `WS_thr = 0`, `FH_thr = 0` → s=0 ≥ FH_thr → FULL_HIT (degenerate cell, documented)
12. `s = -1e-12` (numerically near-zero negative from float arith) + `WS_thr = 0` → MISS (strict `<`)

**`_score_only` helper** (added per G1 R2 B2):
13. Zero-weight keys present in `weights` dict → excluded from both numerator and denominator (denominator = count of non-zero-weight keys)
14. All weights zero → returns NaN (no division by zero)
15. `_score_only` output ≡ `compose(...).composer_score` for the score-emission paths (FH and WS) on a non-degenerate input

### §8.2 Validator tests

`tests/cache/test_phase3_validator.py` (or extend existing composer validator suite):
- Reject yaml missing `tier_thresholds.warm_start` for `weighted_sum_zero_nan`
- Reject `warm_start > full_hit`
- Reject CP3 with `warm_start_t` (CP1-only inherited rule)
- Accept valid Phase 3 yaml shape

### §8.3 Threshold solver tests

`tests/exp/test_phase3_threshold_solver.py` — must directly exercise the §5.4 saturated-buffer reconstruction path, not a precomputed-score shortcut. Every test below uses synthetic JSONL with a `factor_raw` column (the actual DumpingJudge wire format) and goes through the full `CalibrationSamples → PercentileRollingCalibration(samples, window_size).bind_keys(keys) → cal(raw) → composer._score_only(...)` chain. A regression to the prior nonexistent `params={...}, bind_keys=[...]` constructor or to a cold-start output model **must** fail at least one test (G1 R3 B1 lock-in).

**Reconstruction interface compatibility**:
1. `reconstruct_scores` runs end-to-end on a synthetic JSONL with 100 rows × 2 declared keys, all values numeric and varied — assert returns 100 finite scores, monotone-increasing across an injected gradient, and matches a hand-computed reference for at least 3 specific rows (verifies orient flip + percentile direction).
2. **Constructor-shape regression**: monkey-patch `PercentileRollingCalibration` to fail loud if `__init__` is called with `params=` or `bind_keys=` kwarg, OR if `cal.calibrate(...)` is invoked. The solver must use `__init__(samples, window_size=...)`, separate `bind_keys(keys)`, and `cal(raw)`. (Catches the exact regression G1 R2 flagged.)

**Saturated-buffer behaviour**:
3. **Buffer pre-saturation + maxlen eviction**: a synthetic dump with 60 rows × 1 key, raw values `[1, 2, ..., 60]` (monotone increasing) → `bind_keys` preloads the last 50 non-NaN as a `collections.deque(maxlen=50)` containing `[11, 12, ..., 60]` (verified against `percentile_rolling.py:84–88`). On replay row #0 (value `1.0`): `cal(raw)` calls `buf.append(1.0)` which **evicts `11` from the left** (deque maxlen invariant), yielding the post-append buffer `[12, ..., 60, 1]` of length **50**. Then `_percentile_rank(buf, 1.0)` returns `sum(x <= 1 for x in buf) / 50 = 1 / 50 = 0.02`. Assert `scores[0] == pytest.approx(0.02)` for a `safe`-orientation single-key recipe (and `1 - 0.02 = 0.98` for a `risky`-orientation single-key recipe, which is what `jerk_offline_state` would produce).
4. **First-row finite-score guarantee**: assert `not isnan(scores[0])` whenever the row #0 raw is finite — locks the "no cold-start output" property documented in `percentile_rolling.py:6–7`.

**Fail-fast on insufficient samples** (the `< window_size` per-key case):
5. Synthetic dump with 49 non-NaN rows on a single declared key (one less than `window_size=50`) → `reconstruct_scores` raises (whatever exception `bind_keys` propagates: `ValueError` per `percentile_rolling.py:79`). Assert the error message names the offending key and reports observed sample count, so the runner can surface the recipe-level skip cleanly.
6. Synthetic dump with all 50 rows NaN on one key but 50+ non-NaN on another → still raises (every declared key independently must hit `window_size`).

**NaN propagation post-saturation**:
7. **Partial-NaN row**: replay rows where some declared keys are NaN and others numeric — assert each row's score uses NaN→0 contrib for the NaN keys, divided by full denominator (matches §2.2 fixed-N denominator). Hand-computed reference for at least 2 rows.
8. **All-NaN row mid-replay**: a single row where every declared key is NaN → solver assigns `s = 0` (composer's `_score_only` over all-NaN gives 0/N = 0); buffer for that key is unchanged (verifies `percentile_rolling.py:108–111` "NaN does not enqueue").
9. **All-NaN factor_raw entire JSONL** → `bind_keys` raises (no key has non-NaN samples) before any score is emitted; same fail path as test #5.

**Quantile cut math**:
10. 1000 scores drawn from a known distribution (e.g. uniform `[0, 1]`) → verify `derive_thresholds(scores, fh_ratio=0.2, ws_ratio=0.3)` returns `(FH_thr, WS_thr)` such that |{scores ≥ FH_thr}| ≈ 200 and |{WS_thr ≤ scores < FH_thr}| ≈ 300 (within ±5 due to floor-cut).
11. `fh_ratio + ws_ratio = 1.0` (the `(0.5, 0.5)` cell) → `WS_thr == min(scores)`; `|{scores ≥ WS_thr}| == n` (no MISS in warmup).
12. Tie-heavy distribution: 1000 scores drawn from `{0.0, 0.5, 1.0}` uniform-discrete → cuts may not partition exactly, but `FH_thr ∈ {0.0, 0.5, 1.0}`; document observed boundary placement; ensure no exception.

**Output schema** (G1 R3 B3 lock-in):
13. End-to-end run on a successful synthetic dump → output JSON matches §5.8 schema verbatim: keys `n_warmup_rows`, `n_finite_scores`, `n_dropped_nan_scores`, `score_stats`, `per_key_non_nan_count`, `cells`. **No** `n_warmup_scores_warm_state` or `score_stats_warm_state` fields (the prior-draft cold-start vestige) — assert their absence too.
14. End-to-end run on a fail-fast dump (test #5 input) → output JSON has an `error` field (with insufficient-key name and count) and **no** `cells` field; `_NA` marker propagates to runner aggregation.

### §8.4 spec round-trip test

`tests/exp/test_phase3_spec.py`:
- Each of 11 recipes: yaml emits valid (`load_cache_config` doesn't raise)
- Warmup yaml shape: `AlwaysWarmStartJudge + DumpingJudge`, dump factor superset = factor union
- Placeholder yamls with FH/WS=`null` get rejected by validator (fail-loud), then patch via `derive_thresholds` output passes

### §8.5 Smoke run gate

Before launching all 11 recipes:
- Run **g4 only** (`g4_f1b_t_w_short_d_jerk` — simplest, single descriptor, 3 windows) end-to-end:
  - warmup 20 ep → dump → solve → 1 eval cell (`FH=0.4, WS=0.3`) 10 ep
  - Verify per_yaml_summary.jsonl has expected fields
  - Verify SR / FH / WS / MISS counts add up to n_eval_verdicts
  - Verify composer_score histogram matches synthetic prediction (rough)

Only after smoke passes do all 11 × 16 launch.

---

## §9 inference_ratio Update

**Per-verdict cost** with `start_t = 0.5`:
```
miss            -> 1.0
full_hit        -> 0.0
warm_start@0.5  -> 1.0 - 0.5 * (1.0 - 0.5) = 0.75
```

(Phase 2 was 0.85 because `start_t=0.7`. Phase 3 WS verdicts are **cheaper** in inf terms by 0.10 — direct knock-on for the Pareto plot.)

`exp/verdict_factor_judge/analysis/plot_pareto.py` is per-phase parametrized by `WARM_START_T_PHASE2 = 0.70`. Phase 3 needs its own plot (or pass `start_t` as arg). New file `plot_pareto_phase3.py` cleaner — copy phase2 plotter, change constant to 0.50.

---

## §10 Review Process

### §10.1 G1 — APPROVED Round 5 (2026-05-07 17:35 CDT)

Per `protocols/execution_authority.md §3.1`, the G1 Review Log has been deleted en bloc post-approval. Final verdict checklist:

- Architecture consistency: PASS
- Interface compatibility: PASS
- Risk identification: PASS
- Test strategy: PASS

### §10.3 G2 (Code Review) — After §6 Code Complete

Code review against the approved plan. Standard `protocols/review_authority.md` §4 G2 flow:

- diff confirms plan §1–§9 implemented as specified, no scope creep
- §8 tests all green
- §4.4 pkl enrichment smoke gate executed and passed
- §8.5 g4 end-to-end smoke gate executed and passed
- inf_ratio 0.75 propagated correctly to plotter

### §10.4 Review Log (G2)

(empty)

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-05-07 18:33 CDT

- [Blocking] [Concern] `run_phase3.py` derives thresholds from the full warmup JSONL but preloads eval calibration from a capped first-400-sample buffer — reasoning: `solve_recipe(...)` / `reconstruct_scores(...)` build `CalibrationSamples` from every row in the warmup JSONL, matching §5.4's "full warmup history" equivalence claim, but `run_phase3.py` calls `_parse_dump_to_buffer(..., max_per_key=400)` for the WarmupPool that eval actually uses. `_parse_dump_to_buffer` keeps the first values after the cap is reached, while `PercentileRollingCalibration.bind_keys` preloads the last 50 values it receives. A read-only probe with 421 monotone samples showed the solver preload would be `[372..421]`, while the eval preload from the capped buffer would be `[351..400]`. That breaks the plan's no-drift claim: thresholds are cut under one saturated calibration distribution and evaluated under another. Feed the eval WarmupPool the same per-key history the solver used, or intentionally make both sides share the same capped/tail-preserving source.
- [Blocking] [Concern] Solver fail-fast `_NA` propagation is not written to the summary — reasoning: §5.7 says an insufficient-sample recipe is skipped with `_NA` markers, and §8.3 test #14 says that marker propagates to runner aggregation. In `run_phase3.py`, the `"error" in threshold_doc` branch returns `[{"yaml_id": ..., "error": ...}]`, but the caller in `main()` ignores `_run_one_recipe(...)`'s return value and that branch never appends to `summary_path`. A fail-fast recipe is therefore logged locally and written to thresholds JSON, but silently absent from `per_yaml_summary.jsonl` / downstream aggregation. Append the expected `_NA` summary row(s) in the error branch or have `main()` consume returned summaries consistently.
- [Blocking] [Concern] `logs/README.md` is not synchronized with the implemented / approved state of this log — reasoning: the active entry still says `verdict_phase3_threshold_sweep.log.md` is "`Plan` (G1 pending)" and describes threshold derivation from `inner_factor_outputs.composer_score`, but the polished plan now records G1 approval and the implementation derives scores from `factor_raw` via offline saturated replay. This fails the G2 docs/index checklist and violates the Working Agreement's index-sync requirement for changed logs.
- [Non-blocking] [Concern] The implementation intentionally skips the approved placeholder-eval-yaml pass but the deviation is not recorded in the plan/index — reasoning: §6.3 says Phase 3 emits placeholder eval yamls per cell before warmup and patches them after threshold solving; the current generator writes only 11 warmup yamls plus a manifest, and `run_phase3.py` builds eval yamls dynamically after solving. That can be an acceptable simplification, but it is a plan deviation and the test suite currently only mentions it in a test docstring. Either restore placeholder emission or record the approved deviation in the plan / README so operators do not look for 176 placeholder files.
- [Non-blocking] [Concern] Successful recipe cleanup does not match the runner comment or §7 flow — reasoning: the per-cell loop unloads each eval yaml's WarmupPool entry, but after the final cell there is no `unload_warmup_buffer(warmup_eval_yaml_id)` call to remove the original warmup dump file. The error branch does this cleanup; the success path leaves the recipe warmup dump behind despite the comment saying cleanup happens after the recipe's last cell.

Checklist:

- Consistency with approved plan: FAIL — the core composer / solver architecture is mostly implemented, but the runner's eval preload source drifts from the solver source, `_NA` aggregation is incomplete, and placeholder eval-yaml generation deviates from §6.3 without a recorded plan update.
- Test coverage and passing: FAIL — the 66 targeted Phase 3 tests passed, and the pkl enrichment smoke probe found the expected new sample keys, but there is no runner-level test covering WarmupPool sample parity or fail-fast summary propagation; those gaps hide the two blocking runner defects above.
- Docs & indexes updated: FAIL — `logs/README.md` is stale on both lifecycle status and solver architecture for the active log.
- No regressions: FAIL — the capped WarmupPool buffer can change calibration semantics for recipes with more than 400 finite samples per key, and fail-fast recipes disappear from the summary surface.

Advisory verification run by reviewer:

- `PYTHONPATH=. uv run pytest tests/cache/components/factors/test_composer_zero_nan.py tests/cache/test_phase3_validator.py tests/exp/test_phase3_threshold_solver.py tests/exp/test_phase3_spec.py -q` → 66 passed, 1 warning.
- Read-only calibration probe: 421 monotone samples preload `[372..421]`; first-400 capped buffer preload `[351..400]`.
- Read-only pkl smoke: canonical `cp1_spatial_pool_16.pkl` loaded, 1018 entries, sampled factor keys include the expected new offline-state windows.

Constitutional violation: `logs/README.md` index sync is stale for a changed active log (Working Agreement §4 Index Sync Rule).

Final verdict: NEEDS REVISION. Fix the runner calibration-buffer drift and fail-fast summary propagation before G2 can approve; update `logs/README.md` in the same revision.

### G2 Round 2 — Executor — 2026-05-07

Each item evaluated per `protocols/execution_authority.md` §10.2; all 5 well-founded, all accepted.

- **Accepted (B1: WarmupPool sample-parity drift, BLOCKING)** — Reviewer's probe is exact: the runner's old `_parse_dump_to_buffer(content, max_per_key=400)` keeps the **first** 400 finite values per key; server-side `bind_keys` then takes their last 50, yielding `[351..400]`. The solver's `reconstruct_scores` builds `CalibrationSamples` from the **full** warmup history with no cap, so `bind_keys` there preloads `[372..421]` (when N=421). For any recipe with > 400 finite samples per key the two saturated-buffer states diverge — exactly the no-drift property §5.4 promised. **Fix**: exposed a new public helper `phase3_threshold_solver.load_per_key_finite_history(jsonl_path, declared_keys) -> dict[str, list[float]]` (uncapped, finite-only, row-order-preserving) and switched `run_phase3._run_one_recipe` to use it for the buffer fed to `preload_normalizer_buffer`. Eval-side bind_keys now takes the same last-50 the solver did. Added 3 tests: `test_load_per_key_finite_history_drops_nan_keeps_order`, `test_load_per_key_finite_history_uncapped` (verifies `len == 420` on a 420-row dump), and `test_warmup_pool_buffer_matches_solver_last_50` (lock-in: the runner's last-50 equals the solver's last-50 for a 420-row monotone dump).
- **Accepted (B2: solver fail-fast `_NA` not written to summary, BLOCKING)** — Reviewer correct: the prior error branch returned a single 1-element list locally and never appended to `summary_path`, so a fail-fast recipe was logged + had a `thresholds.json` but **was absent from `per_yaml_summary.jsonl`**. Plan §5.7 explicitly says "16 cells emit `_NA` markers"; downstream aggregation needs every cell present. **Fix**: added `_write_na_summary_rows(summary_path, ..., grid)` helper in `run_phase3.py` that writes 16 `_NA` rows per fail-fast recipe (one per grid cell, same column set as success rows: `recipe_id`, `fh_ratio`, `ws_ratio`, `fh_thr=null`, `ws_thr=null`, `success_rate=null`, `n_eval_verdicts=0`, `n_full_hit=0`, `n_warm_start=0`, `n_miss=0`, `error=...`). The error branch in `_run_one_recipe` now calls this helper. New `tests/exp/test_phase3_runner.py` (4 tests) covers row count = 16, full grid coverage, `__NA` suffix in `yaml_id`, and `summary_path=None` no-IO mode for the rows-only caller path.
- **Accepted (B3: `logs/README.md` index sync stale, BLOCKING + constitutional violation)** — Reviewer correct (WA §4 Index Sync Rule). The active entry's status was `Plan` (G1 pending) and the description still cited `inner_factor_outputs.composer_score` as the threshold source, while the polished plan now records G1 APPROVED at R5 and the implementation derives scores from `factor_raw` via offline saturated-buffer replay. **Fix**: rewrote the active entry. New status: `In Progress` (G1 APPROVED R5; §4 Code 完成；G2 进行中). Description updated to reflect: warmup yaml = `AlwaysWarmStartJudge + DumpingJudge` writing `factor_raw`; solver = offline reconstruction through `CalibrationSamples → PercentileRollingCalibration(samples, window_size=50).bind_keys(keys)` (saturated, fail-fast `_NA`); canonical pkl path lock fixing v2_spec's three broken `exp/warm_start/data/...` references; §6.3 deviation note explicit.
- **Accepted (B4: §6.3 deviation not recorded, NON-BLOCKING)** — The implementation skipped the placeholder eval yaml step (yamls are emitted on-demand by the runner from `build_eval_yaml_for_cell` after the solver fills thresholds). **Fix**: added a deviation block at the end of plan §6.3 documenting (a) the deviation, (b) the reason (the new composer's validator §3.2 requires `tier_thresholds.full_hit/warm_start` mandatory — a placeholder with `null` thresholds cannot pass `load_cache_config`), (c) that final on-disk artefact set is identical (11 warmup + 176 eval yamls), only write timing differs. README now mentions the deviation too so operators don't search for 176 placeholder files between the spec and runner steps.
- **Accepted (B5: recipe-level cleanup missing from success path, NON-BLOCKING)** — Reviewer correct: the per-cell loop unloaded each eval yaml's WarmupPool entry, but the shared warmup yaml's pool entry + dump file (keyed by `warmup_eval_yaml_id`) were leaked on the success path. The error branch already had this cleanup; the symmetry was missing. **Fix**: appended an `unload_warmup_buffer(warmup_eval_yaml_id)` call at the end of `_run_one_recipe`'s success path (after every cell runs), wrapped in try/except so a server-side cleanup error doesn't fail the recipe (it's idempotent — server-side cleanup will catch leftover entries on next yaml load).

Test reproduction:
- `uv run pytest tests/cache/components/factors/test_composer_zero_nan.py tests/cache/test_phase3_validator.py tests/exp/test_phase3_threshold_solver.py tests/exp/test_phase3_spec.py tests/exp/test_phase3_runner.py -q` → **73 passed** (66 original + 3 history helper + 1 sample parity lock-in + 4 _NA summary).
- `uv run pytest tests/cache tests/exp -q` (minus 2 pre-existing baseline failures) → **1075 passed, 4 skipped, 0 new failures**.

Re-entering G2 with the revised code + plan.

### G2 Round 2 — Reviewer — APPROVED — 2026-05-07 18:45 CDT

No blocking findings.

- [Non-blocking] [Concern] `logs/README.md` still says "4 测试文件 (66 tests)" although this revision added `tests/exp/test_phase3_runner.py` and the targeted Phase 3 set now has 73 tests — reasoning: the index is now synchronized on the substantive state, solver architecture, and G2 lifecycle, so this does not block approval, but the test-count parenthetical should be corrected during final polish.

Checklist:

- Consistency with approved plan: PASS — G2 R1 B1/B2 were fixed in the runner: eval WarmupPool preload now comes from the same uncapped finite JSONL history as the solver, and fail-fast recipes write 16 `_NA` summary rows. The §6.3 emit-on-demand deviation is now explicitly recorded in the plan and README.
- Test coverage and passing: PASS — targeted Phase 3 tests passed locally (`73 passed`). The broader `tests/cache tests/exp` run produced `1075 passed, 4 skipped, 2 failed`; both failures are in unrelated `random_periodic_gate` tests and match the executor's "pre-existing baseline failures" note.
- Docs & indexes updated: PASS — `logs/README.md` no longer says G1 pending or references the old `inner_factor_outputs.composer_score` solver path; it now documents offline saturated reconstruction from `factor_raw`, canonical pkl path, and the emit-on-demand deviation. Minor test-count drift noted above.
- No regressions: PASS — the previous calibration-buffer drift and missing summary-surface regression are closed, and no new regression was found in the reviewed Phase 3 diff.

Advisory verification run by reviewer:

- `PYTHONPATH=. uv run pytest tests/cache/components/factors/test_composer_zero_nan.py tests/cache/test_phase3_validator.py tests/exp/test_phase3_threshold_solver.py tests/exp/test_phase3_spec.py tests/exp/test_phase3_runner.py -q` → 73 passed, 1 warning.
- `PYTHONPATH=. uv run pytest tests/cache tests/exp -q` → 1075 passed, 4 skipped, 2 unrelated random_periodic_gate failures, 14 warnings.

Final verdict: APPROVED for G2. No constitutional violation.

---

## §11 Out-of-band Decisions

- **No Phase 3 Layer 2** envisioned at this time. If the (FH, WS) grid surface flat-lines, follow-up is a **denser** grid or threshold-method change, not architectural.
- **Cross-cfg run** (clip / max_pool) deferred. spatial16 results determine whether to expand.
- **Result publication**: results stay in `exp/verdict_factor_judge/data/phase3/` (gitignored). Analysis writeup goes into `exp/verdict_factor_judge/analysis/phase3_results.md` post-eval (analysis is git-tracked).

---

## §12 Run Order

Reordered post-G1 R1 B5: tests follow the code they exercise.

1. **Code** §2 (`WeightedSumZeroNanComposer` in `src/openpi/cache/components/factors/composers/__init__.py`) + §3 (config schema + validator + factory) — pure src/ work, no exp/.
2. **Tests** §8.1 (composer unit) + §8.2 (validator) — exercises §1+§2+§3, all green before proceeding.
3. **Code** §5 (`phase3_threshold_solver.py`) + §6 (`phase3_spec.py`, including v2_spec path fix per §4.0) — exp/ work, depends on §2's `_score_only` accessor.
4. **Tests** §8.3 (solver) + §8.4 (spec round-trip) — exercises §5+§6, all green.
5. **Run §4 enrich** on canonical pkl → §4.4 smoke gate passes.
6. **Code** §7 (runner extension `--phase3` mode) + §9 plot adaptation (`plot_pareto_phase3.py`).
7. **Pre-flight** (added per §4.0): assert canonical pkl exists; assert v2_spec / phase3_spec generated yamls' `preload_path` matches canonical.
8. **Smoke gate §8.5** — g4 end-to-end (warmup → solve → 1 eval cell) on 1 server.
9. **Full sweep** — 4–6 servers, 11 recipes round-robin, 16 cells each.
10. **Aggregate** → `analysis/phase3_results.md` + `analysis/plot_pareto_phase3.png`.

---

## §13 Open Items (to revisit if relevant)

- If g2/g3/g7 (long-window NaN-heavy recipes) produce composer_score distributions where `s=0` is the modal value (because `(7,7)` NaN ≈ 67% of frames), the warmup-derived threshold may sit very low. Look at score histograms post-warmup before launching eval — this is a §8.5 smoke gate sub-step.
- If `0.5+0.5=1.0` cell shows `MISS > 0` in eval (distribution shift between warmup and eval), record it but do not adjust solver — that is the experimental signal we want to measure.
- `start_t=0.5` vs Phase 2's `0.7`: WS path is cheaper but earlier-onset; Phase 2 baseline comparison should clamp to either consistent `start_t` or annotate the difference. Default: **annotate**, do not re-run Phase 2.
