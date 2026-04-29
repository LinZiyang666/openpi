"""Phase 2 Layer 2 (redesign) — threshold-axis sweep × spatial16 only.

Old Layer 2 (80 cells) failed structurally: WeightedSum + multi-factor recipes
collapsed under CLT to a tight band around 0.5, so the failed-floor of
``full_hit ∈ {0.40, 0.50, 0.60}`` × ``warm_start ∈ {0.20, 0.30, 0.40}``
produced "all WARM" or "all MISS" cells with zero Pareto signal.

This redesign keeps the inherited dimensions from Layer 1 + old Layer 2
(spatial16 only, 22-recipe set, T-FULL / T-DUAL tier system, percentile_rolling
w=50 + force_miss + ``warm@0.7`` fallback, round-robin 6-batch layout, 100 ep
× 1 seed, eval-trials=10 / num-workers=5 / warmup-trials=2) and adds the
threshold sweep as a first-class axis. Two structural changes vs old Layer 2:

1. **Multi-factor switches composer family** WeightedSum -> AndGate / OrGate.
   AndGate / OrGate use per-factor thresholds (no aggregated CLT pooling),
   sidestepping the failure mode entirely. The weight-strategy axis from
   old Layer 2 (uniform / jerk_heavy / state_dom / f1bt_3x / f1bt_only)
   structurally disappears: AndGate / OrGate are weight-free.

2. **Threshold values are completely disjoint from old Layer 2.** Old fixed
   ``full_hit=0.5`` and swept ``{0.40, 0.60}`` only as edge probes. New design:

       Single-factor (WeightedSum, no CLT): full_hit in {0.20, 0.25, 0.30, 0.35}
       Multi-factor AndGate:                per_factor_thr in {0.20, 0.25, 0.35}
       Multi-factor OrGate:                 per_factor_thr in {0.65, 0.75, 0.85, 0.90}
       T-DUAL warm_start (single-factor):   {0.10}  (was {0.20, 0.30, 0.40})

Total: 80 (Phase A) + 48 (B) + 60 (C) + 36 (D) + 16 (E) = **240 cells**, three
times the old 80-cell budget. Every eval yaml carries
``judge.export_factor_outputs: true`` so per-step factor_outputs land on
``__hit_meta__`` and the per_step_writer episode buffer — enabling offline
threshold rescoring without burning fresh server time.

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

# Symmetric sym_s windows used by f1bt_sym_s_all (mirror old Layer 2).
W_SYM_S = [
    {"past": 1, "future": 1},
    {"past": 2, "future": 2},
    {"past": 3, "future": 3},
]

# ------------------------------------------------------------------
# Recipe -> (factors list, key list) — inherited from old Layer 2
# ------------------------------------------------------------------
# Each recipe is the (factors, keys) tuple consumed by the composer
# constructors. Single-factor recipes drive the WeightedSum sweep
# (Phase A / B / E); multi-factor recipes drive AndGate / OrGate (C / D).


def _r_single_f1bt(windows: list[dict], descs: list[str]) -> tuple[list[dict], list[str]]:
    return [_f1b_factor("f1b_t", descs, windows)], _f1b_keys("f1b_t", descs, windows)


def _r_single_f1ba(windows: list[dict], descs: list[str]) -> tuple[list[dict], list[str]]:
    return [_f1b_factor("f1b_a", descs, windows)], _f1b_keys("f1b_a", descs, windows)


def _r_single_f1at(descs: list[str]) -> tuple[list[dict], list[str]]:
    return [_f1a_factor("f1a_t", descs)], _f1a_keys("f1a_t", descs)


def _r_pair_state(windows: list[dict]) -> tuple[list[dict], list[str]]:
    """splice — F1a-T.jerk + F1b-T.jerk @ given windows."""
    return (
        [_f1a_factor("f1a_t", ["jerk"]), _f1b_factor("f1b_t", ["jerk"], windows)],
        _f1a_keys("f1a_t", ["jerk"]) + _f1b_keys("f1b_t", ["jerk"], windows),
    )


def _r_intrinsic(windows: list[dict]) -> tuple[list[dict], list[str]]:
    """F1b-A.jerk + F1b-T.jerk @ given windows."""
    return (
        [_f1b_factor("f1b_a", ["jerk"], windows), _f1b_factor("f1b_t", ["jerk"], windows)],
        _f1b_keys("f1b_a", ["jerk"], windows) + _f1b_keys("f1b_t", ["jerk"], windows),
    )


def _r_full_lite(windows: list[dict]) -> tuple[list[dict], list[str]]:
    """3-factor: F1a-T.jerk + F1b-T.jerk + F2."""
    return (
        [_f1a_factor("f1a_t", ["jerk"]),
         _f1b_factor("f1b_t", ["jerk"], windows),
         _f2_factor()],
        (_f1a_keys("f1a_t", ["jerk"])
         + _f1b_keys("f1b_t", ["jerk"], windows)
         + ["f2_var"]),
    )


def _r_5factor(windows: list[dict]) -> tuple[list[dict], list[str]]:
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


def _r_with_f2(base: tuple[list[dict], list[str]]) -> tuple[list[dict], list[str]]:
    factors, keys = base
    return factors + [_f2_factor()], keys + ["f2_var"]


RECIPES: dict[str, tuple[list[dict], list[str]]] = {
    # ── Single F1b-T jerk × window ──
    "f1bt_lr_jerk":      _r_single_f1bt(W_LONG_RISK, ["jerk"]),
    "f1bt_fut_jerk":     _r_single_f1bt(W_FUT, ["jerk"]),
    "f1bt_short_jerk":   _r_single_f1bt(W_SHORT, ["jerk"]),
    "f1bt_lr_fut_jerk":  _r_single_f1bt(W_LONG_RISK + W_FUT, ["jerk"]),
    "f1bt_lr_short_jerk": _r_single_f1bt(W_LONG_RISK + W_SHORT, ["jerk"]),
    "f1bt_mix_jerk":     _r_single_f1bt(W_MIX, ["jerk"]),
    # ── Single F1a-T jerk ──
    "f1at_jerk":         _r_single_f1at(["jerk"]),
    # ── Single all-4 desc (multi-key but single factor family) ──
    "f1bt_lr_all":       _r_single_f1bt(W_LONG_RISK, ALL_DESC),
    "f1bt_fut_all":      _r_single_f1bt(W_FUT, ALL_DESC),
    "f1bt_sym_s_all":    _r_single_f1bt(W_SYM_S, ALL_DESC),
    "f1ba_fut_all":      _r_single_f1ba(W_FUT, ALL_DESC),
    # ── Multi-factor combos (AndGate / OrGate land — see Phase C / D) ──
    "splice_lr_jerk":    _r_pair_state(W_LONG_RISK),
    "splice_fut_jerk":   _r_pair_state(W_FUT),
    "intrinsic_lr_jerk": _r_intrinsic(W_LONG_RISK),
    "full_lite_lr":      _r_full_lite(W_LONG_RISK),
    "5factor_lr":        _r_5factor(W_LONG_RISK),
    # ── Single + F2 augment (Phase E) ──
    "f1bt_lr_jerk_f2":   _r_with_f2(_r_single_f1bt(W_LONG_RISK, ["jerk"])),
    "f1at_jerk_f2":      _r_with_f2(_r_single_f1at(["jerk"])),
}

# Recipes that compose multiple factor families — these route to AndGate /
# OrGate composer in Phase C / D. Single-factor / single-family recipes use
# WeightedSum even when they hold multiple keys (Phase A / B / E), since with
# correlated keys inside one family the CLT collapse is mild.
_MULTI_FACTOR_RECIPES = {
    "splice_lr_jerk", "splice_fut_jerk",
    "intrinsic_lr_jerk", "full_lite_lr", "5factor_lr",
}


# ------------------------------------------------------------------
# Phase configurations
# ------------------------------------------------------------------
# Each phase is a tuple (recipes, tier_shapes, threshold_grid). Cells are
# emitted as cartesian product. Total = 80 + 48 + 60 + 36 + 16 = 240.
#
# tier shape spec format:
#   single-factor  -> ("token", {"warm_start": float|None, "warm_start_t": float|None})
#   AndGate/OrGate -> ("token", {"warm_start_t": float|None})
# A tier with warm_start=None / warm_start_t=None means the hit goes to
# FULL_HIT directly (no WARM_START tier).

_TIER_SF_FULL    = ("t_full",        {"warm_start": None, "warm_start_t": None})
_TIER_SF_DUAL_07 = ("t_dual_07_w10", {"warm_start": 0.10, "warm_start_t": 0.7})
_TIER_SF_DUAL_05 = ("t_dual_05_w10", {"warm_start": 0.10, "warm_start_t": 0.5})
_TIER_SF_DUAL_03 = ("t_dual_03_w10", {"warm_start": 0.10, "warm_start_t": 0.3})

_TIER_GATE_HIT    = ("hit_strict", {"warm_start_t": None})
_TIER_GATE_WARM_7 = ("warm_07",    {"warm_start_t": 0.7})
_TIER_GATE_WARM_5 = ("warm_05",    {"warm_start_t": 0.5})
_TIER_GATE_WARM_3 = ("warm_03",    {"warm_start_t": 0.3})


# Phase A — single-factor jerk × tier × full_hit (5 r × 4 tier × 4 thr = 80)
PHASE_A = {
    "recipes": ["f1bt_lr_jerk", "f1bt_fut_jerk", "f1bt_short_jerk",
                "f1at_jerk",    "f1bt_lr_fut_jerk"],
    "tiers":   [_TIER_SF_FULL, _TIER_SF_DUAL_07, _TIER_SF_DUAL_05, _TIER_SF_DUAL_03],
    "thrs":    [0.20, 0.25, 0.30, 0.35],
}

# Phase B — single-factor all-4 desc (4-key family) × tier × thr (4 r × 3 tier × 4 thr = 48)
PHASE_B = {
    "recipes": ["f1bt_lr_all", "f1bt_fut_all", "f1bt_sym_s_all", "f1ba_fut_all"],
    "tiers":   [_TIER_SF_FULL, _TIER_SF_DUAL_07, _TIER_SF_DUAL_05],
    "thrs":    [0.20, 0.25, 0.30, 0.35],
}

# Phase C — multi-factor AndGate × tier × per-factor thr (5 r × 4 tier × 3 thr = 60)
PHASE_C = {
    "recipes": ["splice_lr_jerk", "splice_fut_jerk", "intrinsic_lr_jerk",
                "full_lite_lr",   "5factor_lr"],
    "tiers":   [_TIER_GATE_HIT, _TIER_GATE_WARM_7, _TIER_GATE_WARM_5, _TIER_GATE_WARM_3],
    "thrs":    [0.20, 0.25, 0.35],
}

# Phase D — multi-factor OrGate × tier × per-factor thr (3 r × 3 tier × 4 thr = 36)
PHASE_D = {
    "recipes": ["full_lite_lr", "splice_lr_jerk", "5factor_lr"],
    "tiers":   [_TIER_GATE_HIT, _TIER_GATE_WARM_7, _TIER_GATE_WARM_5],
    "thrs":    [0.65, 0.75, 0.85, 0.90],
}

# Phase E — cross-window single-factor + F2-augmented × tier × thr (4 r × 2 tier × 2 thr = 16)
PHASE_E = {
    "recipes": ["f1bt_lr_short_jerk", "f1bt_mix_jerk",
                "f1bt_lr_jerk_f2", "f1at_jerk_f2"],
    "tiers":   [_TIER_SF_FULL, _TIER_SF_DUAL_07],
    "thrs":    [0.25, 0.30],
}


# ------------------------------------------------------------------
# Stem / yaml builders
# ------------------------------------------------------------------


def _thr_token(thr: float) -> str:
    """Compact threshold token (0.25 -> 'h25')."""
    return f"h{int(round(thr * 100)):02d}"


def _pf_thr_token(thr: float) -> str:
    """Compact per-factor threshold token for AndGate / OrGate."""
    return f"pf{int(round(thr * 100)):02d}"


def _stem_ws(recipe: str, tier_token: str, thr: float) -> str:
    """Stem for WeightedSum cells (Phase A / B / E)."""
    return f"phase2_l2_{recipe}_{tier_token}_{_thr_token(thr)}"


def _stem_gate(kind: str, recipe: str, tier_token: str, thr: float) -> str:
    """Stem for AndGate (kind='and') / OrGate (kind='or') cells."""
    return f"phase2_l2_{recipe}_{kind}_{tier_token}_{_pf_thr_token(thr)}"


def _directions_for(keys: list[str]) -> dict[str, str]:
    """N-PCT directions block for non_monotonic keys (curv_radius / cum_disp)."""
    out: dict[str, str] = {}
    for k in keys:
        base = k.split("__")[0]
        if base.endswith("_curv_radius"):
            out[k] = "range:[0.3, 0.7]"
        elif base.endswith("_cum_disp"):
            out[k] = "high"
    return out


def _composer_ws(keys: list[str], thr: float, tier_extras: dict) -> dict:
    """Build WeightedSumComposer config block (uniform weights)."""
    weights = {k: 1.0 for k in keys}
    directions = _directions_for(keys)
    composer: dict = {
        "type": "weighted_sum",
        "weights": weights,
        "tier_thresholds": {"full_hit": thr},
    }
    if directions:
        composer["directions"] = directions
    warm_start = tier_extras.get("warm_start")
    warm_start_t = tier_extras.get("warm_start_t")
    if warm_start is not None:
        composer["tier_thresholds"]["warm_start"] = warm_start
        composer["warm_start_t"] = warm_start_t
    return composer


def _composer_gate(kind: str, keys: list[str], thr: float, tier_extras: dict) -> dict:
    """Build AndGate (kind='and') / OrGate ('or') composer config block."""
    pft = {k: thr for k in keys}
    directions = _directions_for(keys)
    composer: dict = {
        "type": kind,
        "per_factor_thresholds": pft,
    }
    if directions:
        composer["directions"] = directions
    warm_start_t = tier_extras.get("warm_start_t")
    if warm_start_t is not None:
        composer["warm_start_t"] = warm_start_t
    return composer


def build_eval_yaml(factors: list[dict], composer: dict) -> dict:
    """Assemble a CompositeJudge eval yaml with export_factor_outputs on."""
    cfg = CFG_SPECS[CFG_ID]
    judge = {
        "type": "composite",
        "factors": factors,
        "composer": composer,
        "normalizer": {
            "type": "percentile_rolling",
            "window_size": 50,
            "cold_start_strategy": "force_miss",
        },
        "all_nan_fallback": {"type": "warm_start", "start_t": 0.7},
        # Layer 2 redesign: every eval yaml emits per-verdict factor_outputs
        # (raw + normalized + composer score + cold-start sentinel) so the
        # per_step jsonl carries enough state for offline threshold rescoring.
        "export_factor_outputs": True,
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
    W_UNION so any Layer 2 eval yaml's keys are a subset of the dump.
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
# Generation list
# ------------------------------------------------------------------
# Each entry is (stem, factors, composer_block). Order is deterministic so
# round-robin batch assignment is reproducible.

GENERATION: list[tuple[str, list[dict], dict]] = []


def _emit_phase_ws(phase: dict) -> None:
    for recipe in phase["recipes"]:
        factors, keys = RECIPES[recipe]
        for tier_token, tier_extras in phase["tiers"]:
            for thr in phase["thrs"]:
                stem = _stem_ws(recipe, tier_token, thr)
                composer = _composer_ws(keys, thr, tier_extras)
                GENERATION.append((stem, factors, composer))


def _emit_phase_gate(kind: str, phase: dict) -> None:
    for recipe in phase["recipes"]:
        factors, keys = RECIPES[recipe]
        for tier_token, tier_extras in phase["tiers"]:
            for thr in phase["thrs"]:
                stem = _stem_gate(kind, recipe, tier_token, thr)
                composer = _composer_gate(kind, keys, thr, tier_extras)
                GENERATION.append((stem, factors, composer))


_emit_phase_ws(PHASE_A)
_emit_phase_ws(PHASE_B)
_emit_phase_gate("and", PHASE_C)
_emit_phase_gate("or", PHASE_D)
_emit_phase_ws(PHASE_E)


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

    # Detect duplicate stems early — would otherwise silently overwrite.
    stems = [g[0] for g in GENERATION]
    if len(set(stems)) != len(stems):
        from collections import Counter
        dups = [s for s, c in Counter(stems).items() if c > 1]
        raise RuntimeError(f"Duplicate stems detected: {dups}")

    for i, (stem, factors, composer) in enumerate(GENERATION):
        batch_idx = (i % 6) + 1
        batch_dir = config_root / CFG_SUBDIR / f"phase2_layer2_b{batch_idx}"
        eval_yaml_id = f"{CFG_ID}_{stem}"
        eval_path = batch_dir / f"{eval_yaml_id}.yaml"
        warmup_path = batch_dir / f"{eval_yaml_id}__warmup.yaml"
        write_yaml(eval_path, build_eval_yaml(factors, composer))
        write_yaml(warmup_path, build_warmup_yaml(eval_yaml_id))
        written.append(eval_path)
        written.append(warmup_path)

    print(f"\nWrote {len(written)} yaml files (eval + sibling warmup)")
    print("Per-batch counts:")
    for b in range(1, 7):
        ev = sum(1 for p in written
                 if p.parent.name == f"phase2_layer2_b{b}" and not p.stem.endswith("__warmup"))
        wu = sum(1 for p in written
                 if p.parent.name == f"phase2_layer2_b{b}" and p.stem.endswith("__warmup"))
        print(f"  batch {b}: {ev} eval + {wu} warmup")


if __name__ == "__main__":
    main()
