"""Phase 2 Layer 2 yaml generator — combination + tier + weight × spatial16 only.

Layer 1 found 4 single-factor / single-window winners on spatial16 (cross-cfg
strict-Pareto-positive vs random_periodic + always-WARM baselines). Layer 2
asks: do combinations / threshold sweeps / weight strategies push further?

Single cfg (spatial16) since Phase 2 Layer 1 showed the strongest signals
there — 4 yamls hit SR >= 0.98, near plain inference quality. All 72 yaml
fit in a 6-server / 3-hour budget (12 yamls/server x ~15 min/yaml).

Layer 2 design dimensions:

1. **Recipes** (15 factor / window combinations) — single factor up to 5-factor
2. **Tier configs** (~8 tier setups) — T-FULL + T-DUAL with 3 start_t x 3 thresholds
3. **Weight strategies** (5 patterns) — uniform / jerk-heavy / state-dominant /
   F1b-T-only / F1b-T-3x

Concrete 72-cell selection: see ``GENERATION`` list. Each cell yields one eval
yaml + one warmup sibling. Output split round-robin into 6 batch dirs
``phase2_layer2_b1/`` through ``phase2_layer2_b6/`` for parallel scheduling
across the 6-server cluster.

This module is purely declarative; ``__main__`` performs the file writes.
"""

from __future__ import annotations

from pathlib import Path

from exp.verdict_factor_judge.generate_yamls import write_yaml
from exp.verdict_factor_judge.phase1_spec import CFG_SPECS
from exp.verdict_factor_judge.phase2_spec import (
    ALL_DESC,
    W_FUT,
    W_LONG_RISK,
    W_MIX,
    W_SHORT,
    W_UNION,
    _f1a_factor,
    _f1a_keys,
    _f1b_factor,
    _f1b_keys,
    _f2_factor,
    _shared_keys_block,
    _shared_search_strategy,
)

# ------------------------------------------------------------------
# Cfg restriction — Phase 2 Layer 2 spatial16 only
# ------------------------------------------------------------------

CFG_ID = "spatial16_w8_d4"
CFG_SUBDIR = "spatial16"


# ------------------------------------------------------------------
# Recipes — factor list + key list
# ------------------------------------------------------------------
# Each recipe -> (factors, keys) tuple. ``keys`` is the canonical key list
# WeightedSumComposer expects in its ``weights`` map.


def _recipe_single_f1bt(windows: list[dict], descs: list[str]) -> tuple[list[dict], list[str]]:
    return [_f1b_factor("f1b_t", descs, windows)], _f1b_keys("f1b_t", descs, windows)


def _recipe_single_f1ba(windows: list[dict], descs: list[str]) -> tuple[list[dict], list[str]]:
    return [_f1b_factor("f1b_a", descs, windows)], _f1b_keys("f1b_a", descs, windows)


def _recipe_single_f1at(descs: list[str]) -> tuple[list[dict], list[str]]:
    return [_f1a_factor("f1a_t", descs)], _f1a_keys("f1a_t", descs)


def _recipe_pair_state(windows: list[dict]) -> tuple[list[dict], list[str]]:
    """splice — F1a-T.jerk + F1b-T.jerk @ given windows."""
    return (
        [_f1a_factor("f1a_t", ["jerk"]), _f1b_factor("f1b_t", ["jerk"], windows)],
        _f1a_keys("f1a_t", ["jerk"]) + _f1b_keys("f1b_t", ["jerk"], windows),
    )


def _recipe_intrinsic(windows: list[dict]) -> tuple[list[dict], list[str]]:
    """F1b-A.jerk + F1b-T.jerk @ given windows."""
    return (
        [_f1b_factor("f1b_a", ["jerk"], windows), _f1b_factor("f1b_t", ["jerk"], windows)],
        _f1b_keys("f1b_a", ["jerk"], windows) + _f1b_keys("f1b_t", ["jerk"], windows),
    )


def _recipe_full_lite(windows: list[dict]) -> tuple[list[dict], list[str]]:
    """3-factor: F1a-T.jerk + F1b-T.jerk + F2."""
    return (
        [_f1a_factor("f1a_t", ["jerk"]),
         _f1b_factor("f1b_t", ["jerk"], windows),
         _f2_factor()],
        (_f1a_keys("f1a_t", ["jerk"])
         + _f1b_keys("f1b_t", ["jerk"], windows)
         + ["f2_var"]),
    )


def _recipe_4factor(windows: list[dict]) -> tuple[list[dict], list[str]]:
    """4-factor: F1a-T.jerk + F1b-A.jerk + F1b-T.jerk + F2."""
    return (
        [_f1a_factor("f1a_t", ["jerk"]),
         _f1b_factor("f1b_a", ["jerk"], windows),
         _f1b_factor("f1b_t", ["jerk"], windows),
         _f2_factor()],
        (_f1a_keys("f1a_t", ["jerk"])
         + _f1b_keys("f1b_a", ["jerk"], windows)
         + _f1b_keys("f1b_t", ["jerk"], windows)
         + ["f2_var"]),
    )


def _recipe_5factor(windows: list[dict]) -> tuple[list[dict], list[str]]:
    """5-factor: F1a-A.jerk + F1a-T.jerk + F1b-A.jerk + F1b-T.jerk + F2."""
    return (
        [_f1a_factor("f1a_a", ["jerk"]),
         _f1a_factor("f1a_t", ["jerk"]),
         _f1b_factor("f1b_a", ["jerk"], windows),
         _f1b_factor("f1b_t", ["jerk"], windows),
         _f2_factor()],
        (_f1a_keys("f1a_a", ["jerk"])
         + _f1a_keys("f1a_t", ["jerk"])
         + _f1b_keys("f1b_a", ["jerk"], windows)
         + _f1b_keys("f1b_t", ["jerk"], windows)
         + ["f2_var"]),
    )


def _recipe_factor_with_f2(factor_recipe_factors_keys: tuple[list[dict], list[str]]) -> tuple[list[dict], list[str]]:
    """Append F2 to any recipe."""
    factors, keys = factor_recipe_factors_keys
    return factors + [_f2_factor()], keys + ["f2_var"]


RECIPES: dict[str, tuple[list[dict], list[str]]] = {
    # ── Single factor × jerk-only (cross-cfg Layer 1 winner archetype) ──
    "f1bt_lr_jerk":      _recipe_single_f1bt(W_LONG_RISK, ["jerk"]),
    "f1bt_fut_jerk":     _recipe_single_f1bt(W_FUT, ["jerk"]),
    "f1bt_short_jerk":   _recipe_single_f1bt(W_SHORT, ["jerk"]),
    "f1bt_mix_jerk":     _recipe_single_f1bt(W_MIX, ["jerk"]),
    "f1bt_lr_fut_jerk":  _recipe_single_f1bt(W_LONG_RISK + W_FUT, ["jerk"]),
    "f1at_jerk":         _recipe_single_f1at(["jerk"]),
    "f1at_jerk_dir":     _recipe_single_f1at(["jerk", "dir"]),
    # ── all-4 desc variants — Layer 1 spatial16-specific strict-positives.
    # Cross-cfg said "jerk-only suffices" but spatial16 alone showed several
    # all-4 yamls also Pareto-positive; Layer 2 must test all-4 in T-DUAL.
    "f1bt_lr_all":       _recipe_single_f1bt(W_LONG_RISK, ALL_DESC),
    "f1bt_fut_all":      _recipe_single_f1bt(W_FUT, ALL_DESC),
    "f1bt_sym_s_all":    _recipe_single_f1bt(
        [{"past": 1, "future": 1}, {"past": 2, "future": 2}, {"past": 3, "future": 3}],
        ALL_DESC,
    ),
    "f1ba_fut_all":      _recipe_single_f1ba(W_FUT, ALL_DESC),
    # ── F1a-A close-out winners (Layer 1 spatial16 curv / jerk+curv pair) ──
    "f1aa_curv":         (
        [_f1a_factor("f1a_a", ["curv_radius"])],
        _f1a_keys("f1a_a", ["curv_radius"]),
    ),
    "f1aa_jerk_curv":    (
        [_f1a_factor("f1a_a", ["jerk", "curv_radius"])],
        _f1a_keys("f1a_a", ["jerk", "curv_radius"]),
    ),
    # ── F1a-T desc variants — spatial16 winners on curv_only & jerk+dir ──
    "f1at_curv":         _recipe_single_f1at(["curv_radius"]),
    "f1at_jerk_curv":    _recipe_single_f1at(["jerk", "curv_radius"]),
    # ── Pair / triple / quad combos (jerk-only base) ──
    "splice_lr_jerk":    _recipe_pair_state(W_LONG_RISK),
    "splice_fut_jerk":   _recipe_pair_state(W_FUT),
    "intrinsic_lr_jerk": _recipe_intrinsic(W_LONG_RISK),
    "full_lite_lr":      _recipe_full_lite(W_LONG_RISK),
    "full_lite_fut":     _recipe_full_lite(W_FUT),
    "4factor_lr":        _recipe_4factor(W_LONG_RISK),
    "5factor_lr":        _recipe_5factor(W_LONG_RISK),
    # ── all-4 combinations (encode spatial16 lesson into combo tier) ──
    "splice_lr_all":     (
        [_f1a_factor("f1a_t", ALL_DESC), _f1b_factor("f1b_t", ALL_DESC, W_LONG_RISK)],
        _f1a_keys("f1a_t", ALL_DESC) + _f1b_keys("f1b_t", ALL_DESC, W_LONG_RISK),
    ),
    "full_lite_lr_all":  (
        [_f1a_factor("f1a_t", ALL_DESC),
         _f1b_factor("f1b_t", ALL_DESC, W_LONG_RISK),
         _f2_factor()],
        (_f1a_keys("f1a_t", ALL_DESC)
         + _f1b_keys("f1b_t", ALL_DESC, W_LONG_RISK)
         + ["f2_var"]),
    ),
    # ── F2-augmented singles ──
    "f1bt_lr_jerk_f2":   _recipe_factor_with_f2(_recipe_single_f1bt(W_LONG_RISK, ["jerk"])),
    "f1at_jerk_f2":      _recipe_factor_with_f2(_recipe_single_f1at(["jerk"])),
}


# ------------------------------------------------------------------
# Tier configurations
# ------------------------------------------------------------------
# Each tier dict carries the composer's tier_thresholds (and, for DUAL,
# warm_start_t). T-FULL has only full_hit threshold. T-DUAL adds warm_start
# threshold + warm_start_t (the start_t fed to the runtime warm-start path).

TIERS: dict[str, dict] = {
    # Layer 1 baseline
    "t_full":          {"full_hit": 0.5},
    # T-DUAL with full_hit=0.5 / warm=0.3, sweep start_t
    "t_dual_07":       {"full_hit": 0.5, "warm_start": 0.3, "warm_start_t": 0.7},
    "t_dual_05":       {"full_hit": 0.5, "warm_start": 0.3, "warm_start_t": 0.5},
    "t_dual_03":       {"full_hit": 0.5, "warm_start": 0.3, "warm_start_t": 0.3},
    # T-DUAL_07 with threshold sweep (low / high vs default)
    "t_dual_07_lo":    {"full_hit": 0.4, "warm_start": 0.2, "warm_start_t": 0.7},
    "t_dual_07_hi":    {"full_hit": 0.6, "warm_start": 0.4, "warm_start_t": 0.7},
    # T-DUAL_05 threshold sweep
    "t_dual_05_lo":    {"full_hit": 0.4, "warm_start": 0.2, "warm_start_t": 0.5},
}


# ------------------------------------------------------------------
# Weight strategies
# ------------------------------------------------------------------
# Each strategy maps the recipe's key list to a weight dict.


def _weight_uniform(keys: list[str]) -> dict[str, float]:
    return {k: 1.0 for k in keys}


def _weight_jerk_heavy(keys: list[str]) -> dict[str, float]:
    """F1b-T = 1.5; F1a-T = 1.0; F1b-A / F1a-A = 0.5; F2 = 0.5.

    Encodes Layer 1 finding that F1b-T jerk is the strongest signal.
    """
    out = {}
    for k in keys:
        if k.startswith("f1b_t_"):
            out[k] = 1.5
        elif k.startswith("f1a_t_"):
            out[k] = 1.0
        elif k.startswith("f1a_a_") or k.startswith("f1b_a_"):
            out[k] = 0.5
        elif k == "f2_var":
            out[k] = 0.5
        else:
            out[k] = 1.0
    return out


def _weight_state_dom(keys: list[str]) -> dict[str, float]:
    """State-domain (F1a-T, F1b-T) high; action-domain (F1a-A, F1b-A) low; F2 mid."""
    out = {}
    for k in keys:
        if k.startswith("f1a_t_") or k.startswith("f1b_t_"):
            out[k] = 1.5
        elif k.startswith("f1a_a_") or k.startswith("f1b_a_"):
            out[k] = 0.3
        elif k == "f2_var":
            out[k] = 1.0
        else:
            out[k] = 1.0
    return out


def _weight_f1bt_3x(keys: list[str]) -> dict[str, float]:
    """F1b-T dominates: 3x of all other factors."""
    out = {}
    for k in keys:
        if k.startswith("f1b_t_"):
            out[k] = 3.0
        else:
            out[k] = 0.5
    return out


def _weight_f1bt_only(keys: list[str]) -> dict[str, float]:
    """F1b-T weights = 1.0; everything else = 0.05 (near-zero, but composer
    requires non-zero to keep direction binding alive on non_monotonic keys)."""
    out = {}
    for k in keys:
        if k.startswith("f1b_t_"):
            out[k] = 1.0
        else:
            out[k] = 0.05
    return out


WEIGHTS: dict[str, callable] = {
    "uniform":      _weight_uniform,
    "jerk_heavy":   _weight_jerk_heavy,
    "state_dom":    _weight_state_dom,
    "f1bt_3x":      _weight_f1bt_3x,
    "f1bt_only":    _weight_f1bt_only,
}


# ------------------------------------------------------------------
# Generation list — explicit (recipe, tier, weight) cells
# ------------------------------------------------------------------
# 72 cells covering single-factor sweeps, threshold variants, combinations,
# weight strategies, cross-window combos. Order is deterministic so the
# round-robin batch split is reproducible.

GENERATION: list[tuple[str, str, str]] = []

# ────────────────────────────────────────────────────────────────────────
# A. Single-factor jerk tier sweep — 4 recipes × 4 tiers = 16
# ────────────────────────────────────────────────────────────────────────
for r in ("f1bt_lr_jerk", "f1bt_fut_jerk", "f1bt_short_jerk", "f1at_jerk"):
    for t in ("t_full", "t_dual_07", "t_dual_05", "t_dual_03"):
        GENERATION.append((r, t, "uniform"))

# ────────────────────────────────────────────────────────────────────────
# B. Threshold sweep on f1bt_lr_jerk (full_hit / warm_start variants) — 3
# ────────────────────────────────────────────────────────────────────────
for t in ("t_dual_07_lo", "t_dual_07_hi", "t_dual_05_lo"):
    GENERATION.append(("f1bt_lr_jerk", t, "uniform"))

# ────────────────────────────────────────────────────────────────────────
# C. Cross-window single factor — 5
# ────────────────────────────────────────────────────────────────────────
for t in ("t_full", "t_dual_07", "t_dual_05"):
    GENERATION.append(("f1bt_lr_fut_jerk", t, "uniform"))
GENERATION.append(("f1bt_mix_jerk", "t_dual_07", "uniform"))
GENERATION.append(("f1bt_mix_jerk", "t_dual_05", "uniform"))

# ────────────────────────────────────────────────────────────────────────
# D. all-4 desc single-factor — Layer 1 spatial16-specific winners — 9
# (Cross-cfg said jerk dominates, but spatial16 also liked all-4 desc.)
# ────────────────────────────────────────────────────────────────────────
for r in ("f1bt_lr_all", "f1bt_fut_all"):
    for t in ("t_full", "t_dual_07", "t_dual_05"):
        GENERATION.append((r, t, "uniform"))
GENERATION.append(("f1bt_sym_s_all", "t_dual_07", "uniform"))
GENERATION.append(("f1ba_fut_all",   "t_dual_07", "uniform"))
GENERATION.append(("f1ba_fut_all",   "t_dual_05", "uniform"))

# ────────────────────────────────────────────────────────────────────────
# E. F1a-A close-out + F1a-T curv (Layer 1 spatial16 strict-positives) — 5
# ────────────────────────────────────────────────────────────────────────
for r in ("f1aa_curv", "f1aa_jerk_curv", "f1at_curv", "f1at_jerk_curv"):
    GENERATION.append((r, "t_dual_07", "uniform"))
GENERATION.append(("f1at_jerk_curv", "t_dual_05", "uniform"))

# ────────────────────────────────────────────────────────────────────────
# F. F1a-T desc pair — 2
# ────────────────────────────────────────────────────────────────────────
for t in ("t_dual_07", "t_dual_05"):
    GENERATION.append(("f1at_jerk_dir", t, "uniform"))

# ────────────────────────────────────────────────────────────────────────
# G. Combinations (jerk base) × T-DUAL_07 × uniform — 7
# ────────────────────────────────────────────────────────────────────────
for r in ("splice_lr_jerk", "splice_fut_jerk", "intrinsic_lr_jerk",
          "full_lite_lr", "full_lite_fut", "4factor_lr", "5factor_lr"):
    GENERATION.append((r, "t_dual_07", "uniform"))

# ────────────────────────────────────────────────────────────────────────
# H. Combinations (jerk base) × T-DUAL_05 × uniform — 5
# ────────────────────────────────────────────────────────────────────────
for r in ("splice_lr_jerk", "splice_fut_jerk", "intrinsic_lr_jerk",
          "full_lite_lr", "full_lite_fut"):
    GENERATION.append((r, "t_dual_05", "uniform"))

# ────────────────────────────────────────────────────────────────────────
# I. Combinations × T-DUAL_03 — 3
# ────────────────────────────────────────────────────────────────────────
for r in ("splice_lr_jerk", "full_lite_lr", "5factor_lr"):
    GENERATION.append((r, "t_dual_03", "uniform"))

# ────────────────────────────────────────────────────────────────────────
# J. Combinations all-4 desc × T-DUAL × uniform — 4
# (encoding Layer 1 spatial16 lesson into combos)
# ────────────────────────────────────────────────────────────────────────
for t in ("t_dual_07", "t_dual_05"):
    GENERATION.append(("splice_lr_all", t, "uniform"))
    GENERATION.append(("full_lite_lr_all", t, "uniform"))

# ────────────────────────────────────────────────────────────────────────
# K. Weight strategy sweep on full_lite_lr × T-DUAL_07 — 4
# ────────────────────────────────────────────────────────────────────────
for w in ("jerk_heavy", "state_dom", "f1bt_3x", "f1bt_only"):
    GENERATION.append(("full_lite_lr", "t_dual_07", w))

# ────────────────────────────────────────────────────────────────────────
# L. Weight strategy on splice_lr_jerk × T-DUAL_07 — 3
# ────────────────────────────────────────────────────────────────────────
for w in ("jerk_heavy", "state_dom", "f1bt_3x"):
    GENERATION.append(("splice_lr_jerk", "t_dual_07", w))

# ────────────────────────────────────────────────────────────────────────
# M. Weight strategy on intrinsic_lr_jerk × T-DUAL_07 — 3
# ────────────────────────────────────────────────────────────────────────
for w in ("jerk_heavy", "state_dom", "f1bt_3x"):
    GENERATION.append(("intrinsic_lr_jerk", "t_dual_07", w))

# ────────────────────────────────────────────────────────────────────────
# N. Weight strategy on 5factor_lr × T-DUAL_07 — 3
# ────────────────────────────────────────────────────────────────────────
for w in ("jerk_heavy", "state_dom", "f1bt_3x"):
    GENERATION.append(("5factor_lr", "t_dual_07", w))

# ────────────────────────────────────────────────────────────────────────
# O. Threshold sweep on full_lite_lr × T-DUAL_07 — 2
# ────────────────────────────────────────────────────────────────────────
for t in ("t_dual_07_lo", "t_dual_07_hi"):
    GENERATION.append(("full_lite_lr", t, "uniform"))

# ────────────────────────────────────────────────────────────────────────
# P. F2-augmented singles × T-DUAL_07 / T-DUAL_05 — 4
# ────────────────────────────────────────────────────────────────────────
for r in ("f1bt_lr_jerk_f2", "f1at_jerk_f2"):
    for t in ("t_dual_07", "t_dual_05"):
        GENERATION.append((r, t, "uniform"))

# ────────────────────────────────────────────────────────────────────────
# Q. Mixed: jerk_heavy on cheaper-warm tier — 2
# ────────────────────────────────────────────────────────────────────────
GENERATION.append(("full_lite_lr", "t_dual_05", "jerk_heavy"))
GENERATION.append(("splice_lr_jerk", "t_dual_05", "jerk_heavy"))


# ------------------------------------------------------------------
# Yaml builder helpers (mirror phase1_spec / phase2_spec patterns)
# ------------------------------------------------------------------


def _directions_for(keys: list[str]) -> dict[str, str]:
    """N-PCT directions for non_monotonic keys (curv_radius / cum_disp)."""
    out: dict[str, str] = {}
    for k in keys:
        base = k.split("__")[0]
        if base.endswith("_curv_radius"):
            out[k] = "range:[0.3, 0.7]"
        elif base.endswith("_cum_disp"):
            out[k] = "high"
    return out


def _composer_block(keys: list[str], tier_name: str, weight_strategy: str) -> dict:
    tier_cfg = TIERS[tier_name]
    weights = WEIGHTS[weight_strategy](keys)
    directions = _directions_for(keys)
    composer: dict = {
        "type": "weighted_sum",
        "weights": weights,
        "tier_thresholds": {"full_hit": tier_cfg["full_hit"]},
    }
    if directions:
        composer["directions"] = directions
    if "warm_start" in tier_cfg:
        composer["tier_thresholds"]["warm_start"] = tier_cfg["warm_start"]
        composer["warm_start_t"] = tier_cfg["warm_start_t"]
    return composer


def _stem(recipe: str, tier: str, weight: str) -> str:
    """Produce the yaml stem; weight suffix omitted when uniform (match Layer 1 style)."""
    suffix = "" if weight == "uniform" else f"_w_{weight}"
    return f"phase2_l2_{recipe}_{tier}{suffix}"


def build_eval_yaml(recipe: str, tier: str, weight: str) -> dict:
    factors, keys = RECIPES[recipe]
    cfg = CFG_SPECS[CFG_ID]
    judge = {
        "type": "composite",
        "factors": factors,
        "composer": _composer_block(keys, tier, weight),
        "normalizer": {
            "type": "percentile_rolling",
            "window_size": 50,
            "cold_start_strategy": "force_miss",
        },
        "all_nan_fallback": {"type": "warm_start", "start_t": 0.7},
    }
    return {
        "enabled": True,
        "timer": {"enabled": False, "buffer_size": 10000, "output_csv_dir": None},
        "keys": _shared_keys_block(cfg),
        "key_builder": {"type": cfg["key_builder_type"]},
        "checkpoints": {
            "cp1": {
                "enabled": True,
                "gate": {"type": "always_search"},
                "judge": judge,
                "search_strategy": _shared_search_strategy(cfg),
            },
        },
        "backend": {
            "type": "in_memory",
            "vector_dims": dict(cfg["vector_dims"]),
            "in_memory": {
                "preload_path": cfg["preload_pkl"],
                "index_type": "brute_force",
            },
        },
        "write_policy": {"type": "never"},
    }


def build_warmup_yaml(eval_yaml_id: str) -> dict:
    """Universal warmup yaml — over-collects all 5 factors x all-4 desc x
    W_UNION (9 unique windows from W_MIX + W_SHORT + W_PAST + W_FUT +
    W_SYM_S + W_LONG_RISK). Reusable across all Layer 2 eval yamls.
    """
    cfg = CFG_SPECS[CFG_ID]
    warmup_yaml_id = f"{eval_yaml_id}__warmup"
    full_factors = [
        _f1a_factor("f1a_a", ALL_DESC),
        _f1a_factor("f1a_t", ALL_DESC),
        _f1b_factor("f1b_a", ALL_DESC, W_UNION),
        _f1b_factor("f1b_t", ALL_DESC, W_UNION),
        _f2_factor(),
    ]
    judge = {
        "type": "always_warm_start",
        "start_t": 0.7,
        "dump": {"deferred": True, "config_id": warmup_yaml_id, "factors": full_factors},
    }
    search_strategy = _shared_search_strategy(cfg)
    search_strategy["top_k"] = 5
    return {
        "enabled": True,
        "timer": {"enabled": False, "buffer_size": 10000, "output_csv_dir": None},
        "keys": _shared_keys_block(cfg),
        "key_builder": {"type": cfg["key_builder_type"]},
        "checkpoints": {
            "cp1": {
                "enabled": True,
                "gate": {"type": "always_search"},
                "judge": judge,
                "search_strategy": search_strategy,
            },
        },
        "backend": {
            "type": "in_memory",
            "vector_dims": dict(cfg["vector_dims"]),
            "in_memory": {
                "preload_path": cfg["preload_pkl"],
                "index_type": "brute_force",
            },
        },
        "write_policy": {"type": "never"},
    }


# ------------------------------------------------------------------
# Main — round-robin into 6 batch dirs
# ------------------------------------------------------------------


def main() -> None:
    config_root = Path("exp/verdict_factor_judge/config")
    written: list[Path] = []

    n = len(GENERATION)
    print(f"Total Layer 2 cells: {n} (single cfg = {CFG_ID})")
    if n == 0:
        raise RuntimeError("GENERATION list is empty")

    # Round-robin: cell i goes to batch (i % 6) + 1.
    for i, (recipe, tier, weight) in enumerate(GENERATION):
        batch_idx = (i % 6) + 1
        batch_dir = config_root / CFG_SUBDIR / f"phase2_layer2_b{batch_idx}"
        stem = _stem(recipe, tier, weight)
        eval_yaml_id = f"{CFG_ID}_{stem}"
        eval_path = batch_dir / f"{eval_yaml_id}.yaml"
        warmup_path = batch_dir / f"{eval_yaml_id}__warmup.yaml"
        write_yaml(eval_path, build_eval_yaml(recipe, tier, weight))
        write_yaml(warmup_path, build_warmup_yaml(eval_yaml_id))
        written.append(eval_path)
        written.append(warmup_path)

    print(f"\nWrote {len(written)} yaml files (eval + sibling warmup)")
    print(f"Per-batch counts:")
    for b in range(1, 7):
        ev = sum(1 for p in written
                 if p.parent.name == f"phase2_layer2_b{b}" and not p.stem.endswith("__warmup"))
        wu = sum(1 for p in written
                 if p.parent.name == f"phase2_layer2_b{b}" and p.stem.endswith("__warmup"))
        print(f"  batch {b}: {ev} eval + {wu} warmup")


if __name__ == "__main__":
    main()
