"""Verdict-factor-judge v2 yaml generator (refactor 2026-05-07).

Replaces the deleted ``phase1_spec.py / phase2_spec.py / phase2_layer2_spec.py``
modules that produced legacy 5-factor yamls. This single module ships:

- ``CFG_SPECS``: keybuilder / search-strategy / preload pkl per cfg
- helpers ``factor(...)`` / ``composer_*(...)`` / ``calibration_*(...)``
  that construct individual yaml dict blocks for the 4-layer schema
- ``build_eval_yaml(...)`` and ``build_warmup_yaml(...)`` assemblers
- ``GENERATION``: declarative list of (stem, factors, composer) tuples
  for the demo "phase v2" sweep
- ``main()``: writes the demo yaml tree under
  ``exp/verdict_factor_judge/config/v2/<cfg>/``

The demo sweep is a minimal 12-cell starting point; downstream
experiments can copy this module and extend ``GENERATION`` /
``RECIPES`` without touching framework code.

CLI:

    uv run python -m exp.verdict_factor_judge.v2_spec
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from exp.verdict_factor_judge.generate_yamls import write_yaml

# ----------------------------------------------------------------------
# Per-cfg constants (mirror legacy CFG_SPECS — same keybuilders, artifacts)
# ----------------------------------------------------------------------

def _exp_decay_weights(depth: int, *, decay: float = 0.7) -> list[float]:
    """Newest-first exponential decay weights for trajectory_depth>1 search.

    The validator requires ``len(trajectory_weights) == trajectory_depth``,
    all non-negative, and a positive sum. Newest-first is the convention
    documented in ``SearchStrategyConfig``.
    """
    return [decay ** i for i in range(depth)]


# vector_dims, keys-fusion weights, and field_similarity must match the
# canonical spatial16 phase2 layer1 yaml shape so the artifact load_artifact
# check passes (4-key declared vs 4-key in pkl) AND so search retrieval
# behaviour mirrors the Phase 2 frozen baseline (per plan §1 "factor configs
# 完全冻结, with Phase 2 Layer 1 verbatim"). prompt_emb has weight=0 — its
# vector still travels as a search field but contributes nothing to fusion.
_FIELD_SIM_DEFAULT = {
    "vision_0":   {"type": "cosine"},
    "vision_1":   {"type": "cosine"},
    "prompt_emb": {"type": "cosine"},
    "robot_state": {
        "type": "l2",
        "to_similarity": {"type": "exp", "tau": 0.334717},
    },
}

_KEYS_DEFAULT = {
    "vision_0":   {"enabled": True,  "weight": 0.5},
    "vision_1":   {"enabled": True,  "weight": 0.25},
    "vision_2":   {"enabled": False, "weight": 0.0},
    "prompt_emb": {"enabled": True,  "weight": 0.0},
    "robot_state": {"enabled": True, "weight": 0.25},
}


CFG_SPECS: dict[str, dict[str, Any]] = {
    "spatial16_w8_d4": {
        "key_builder_type": "cp1_spatial_pool_16",
        "vector_dims": {
            "vision_0": 32768,
            "vision_1": 32768,
            "prompt_emb": 2048,
            "robot_state": 32,
        },
        "keys": _KEYS_DEFAULT,
        "preload_pkl": "exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl",
        # SearchStrategyConfig has no ``weights`` field — fusion weights come
        # from yaml top-level ``keys.<name>.weight``. trajectory_weights is
        # required when trajectory_depth > 1; newest-first exp decay.
        "search_strategy": {
            "type": "weighted_rrf_knn",
            "rrf_k": 60,
            "top_k": 1,
            "trajectory_depth": 4,
            "trajectory_weights": _exp_decay_weights(4),
            "field_similarity": _FIELD_SIM_DEFAULT,
        },
    },
    "max_pool_w3_d5": {
        "key_builder_type": "cp1_max_pool",
        "vector_dims": {
            "vision_0": 2048,
            "vision_1": 2048,
            "prompt_emb": 2048,
            "robot_state": 32,
        },
        "keys": _KEYS_DEFAULT,
        "preload_pkl": "exp/common/data/cache_artifacts/libero_spatial/cp1_max_pool.pkl",
        "search_strategy": {
            "type": "weighted_rrf_knn",
            "rrf_k": 60,
            "top_k": 1,
            "trajectory_depth": 5,
            "trajectory_weights": _exp_decay_weights(5),
            "field_similarity": _FIELD_SIM_DEFAULT,
        },
    },
    "clip_w7_d4": {
        "key_builder_type": "clip",
        "vector_dims": {
            "vision_0": 512,
            "vision_1": 512,
            "prompt_emb": 2048,
            "robot_state": 32,
        },
        "keys": _KEYS_DEFAULT,
        "preload_pkl": "exp/common/data/cache_artifacts/libero_spatial/clip_vit_b_32.pkl",
        "search_strategy": {
            "type": "weighted_rrf_knn",
            "rrf_k": 60,
            "top_k": 1,
            "trajectory_depth": 4,
            "trajectory_weights": _exp_decay_weights(4),
            "field_similarity": _FIELD_SIM_DEFAULT,
        },
    },
}


# Useful window banks (mirror old W_LR / W_FUT / W_SHORT shapes).
W_LR = [{"past": 5, "future": 5}, {"past": 7, "future": 7}]      # long risk
W_FUT = [{"past": 0, "future": 3}, {"past": 0, "future": 5}]      # future-only
W_SHORT = [{"past": 0, "future": 3}, {"past": 1, "future": 1}, {"past": 3, "future": 0}]


ALL_DESC = ("jerk", "direction", "dispersion", "path_length")


# ----------------------------------------------------------------------
# yaml block helpers
# ----------------------------------------------------------------------


def factor(type_name: str, **params: Any) -> dict[str, Any]:
    """Construct a Factor yaml block ``{type, params}``."""
    return {"type": type_name, "params": dict(params)}


def factor_keys(type_name: str, *, windows: list[dict[str, int]] | None = None) -> list[str]:
    """Return the descriptor keys a factor will emit (mirror of
    ``Factor.describe(params)``). Used to build composer ``weights`` /
    ``per_factor_thresholds`` dicts without re-running describe at write time.
    """
    if type_name == "topk_action_variance":
        return ["topk_action_variance"]
    if windows is None:
        raise ValueError(f"factor_keys: {type_name} requires windows")
    return [f"{type_name}__p{w['past']}_f{w['future']}" for w in windows]


def composer_weighted_sum(
    weights: dict[str, float],
    *,
    full_hit_threshold: float = 0.30,
    warm_start_threshold: float | None = None,
    warm_start_t: float | None = None,
    directions: dict[str, str] | None = None,
) -> dict[str, Any]:
    tier_thresholds: dict[str, float] = {"full_hit": full_hit_threshold}
    if warm_start_threshold is not None:
        tier_thresholds["warm_start"] = warm_start_threshold
    out: dict[str, Any] = {
        "type": "weighted_sum",
        "weights": dict(weights),
        "tier_thresholds": tier_thresholds,
    }
    if warm_start_t is not None:
        out["warm_start_t"] = warm_start_t
    if directions:
        out["directions"] = dict(directions)
    return out


def composer_warm_fallback(
    weights: dict[str, float],
    *,
    full_hit_threshold: float = 0.30,
    warm_fallback_start_t: float = 0.7,
    directions: dict[str, str] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "type": "weighted_sum_with_warm_fallback",
        "weights": dict(weights),
        "tier_thresholds": {"full_hit": full_hit_threshold},
        "warm_fallback_start_t": warm_fallback_start_t,
    }
    if directions:
        out["directions"] = dict(directions)
    return out


def calibration_warmup(window_size: int = 50) -> dict[str, Any]:
    return {
        "type": "percentile_rolling",
        "params": {"window_size": window_size},
        "samples_source": {"type": "warmup"},
    }


def normalization_offline() -> dict[str, Any]:
    return {
        "type": "zscore",
        "params": {},
        "stats_source": {"type": "offline"},
    }


# ----------------------------------------------------------------------
# Window superset for warmup over-collection
# ----------------------------------------------------------------------

# Generic "every window any demo recipe uses" — newest-first dedup.
# Used when build_warmup_yaml is called without an explicit eval_factors
# argument; ensures every eval recipe in this module finds its keys.
_W_UNION_DEFAULT: list[dict[str, int]] = [
    {"past": 0, "future": 3},
    {"past": 0, "future": 5},
    {"past": 1, "future": 1},
    {"past": 3, "future": 0},
    {"past": 3, "future": 3},
    {"past": 5, "future": 5},
    {"past": 7, "future": 7},
]


def _build_dump_factor_superset(
    eval_factors: list[dict] | None,
) -> list[dict]:
    """Construct the dump factor list a warmup yaml must over-collect.

    If ``eval_factors`` is given, group its windows by ``(descriptor,
    source, channel)`` and emit one dump factor per group with that
    group's window union. Always include ``topk_action_variance`` and
    ``_W_UNION_DEFAULT``-window variants for every (descriptor × source ×
    channel) so a re-tuned eval yaml that adds a new window still finds
    the key dumped.
    """
    # Group eval factor windows by registered name.
    per_factor_windows: dict[str, list[dict[str, int]]] = {}
    if eval_factors:
        for f in eval_factors:
            t = f["type"]
            if t == "topk_action_variance":
                continue
            wins = f.get("params", {}).get("windows") or []
            per_factor_windows.setdefault(t, []).extend(wins)

    # Build the union per (desc, source, channel); always include the default
    # union as fallback so the warmup is forward-compatible.
    out: list[dict] = []
    for desc in ALL_DESC:
        for source in ("online", "offline"):
            for channel in ("action", "state"):
                t = f"{desc}_{source}_{channel}"
                merged: list[dict[str, int]] = list(_W_UNION_DEFAULT)
                for w in per_factor_windows.get(t, []):
                    if w not in merged:
                        merged.append(w)
                out.append(factor(t, windows=merged))
    out.append(factor("topk_action_variance", K=5))
    return out


# ----------------------------------------------------------------------
# Demo recipe set (12 cells)
# ----------------------------------------------------------------------


def _r_jerk_online_state_lr() -> tuple[list[dict], dict]:
    f = [factor("jerk_online_state", windows=W_LR)]
    weights = {k: 1.0 for k in factor_keys("jerk_online_state", windows=W_LR)}
    return f, composer_warm_fallback(weights, full_hit_threshold=0.30, warm_fallback_start_t=0.7)


def _r_jerk_online_state_fut() -> tuple[list[dict], dict]:
    f = [factor("jerk_online_state", windows=W_FUT)]
    weights = {k: 1.0 for k in factor_keys("jerk_online_state", windows=W_FUT)}
    return f, composer_warm_fallback(weights, full_hit_threshold=0.30, warm_fallback_start_t=0.7)


def _r_jerk_offline_state_lr() -> tuple[list[dict], dict]:
    f = [factor("jerk_offline_state", windows=W_LR)]
    weights = {k: 1.0 for k in factor_keys("jerk_offline_state", windows=W_LR)}
    return f, composer_warm_fallback(weights, full_hit_threshold=0.30, warm_fallback_start_t=0.7)


def _r_topk_only() -> tuple[list[dict], dict]:
    f = [factor("topk_action_variance", K=5)]
    return f, composer_warm_fallback(
        {"topk_action_variance": 1.0},
        full_hit_threshold=0.30, warm_fallback_start_t=0.7,
    )


def _r_jerk_state_plus_topk() -> tuple[list[dict], dict]:
    f = [
        factor("jerk_online_state", windows=W_LR),
        factor("topk_action_variance", K=5),
    ]
    weights = {k: 1.0 for k in factor_keys("jerk_online_state", windows=W_LR)}
    weights["topk_action_variance"] = 0.5
    return f, composer_warm_fallback(weights, full_hit_threshold=0.30, warm_fallback_start_t=0.7)


# (recipe_stem, recipe_builder)
RECIPES: dict[str, Any] = {
    "jerk_online_state_lr":   _r_jerk_online_state_lr,
    "jerk_online_state_fut":  _r_jerk_online_state_fut,
    "jerk_offline_state_lr":  _r_jerk_offline_state_lr,
    "topk_only":              _r_topk_only,
    "splice_state_topk":      _r_jerk_state_plus_topk,
}


# ----------------------------------------------------------------------
# Yaml assemblers
# ----------------------------------------------------------------------


def build_eval_yaml(
    cfg_id: str,
    factors: list[dict],
    composer: dict,
    *,
    export_factor_outputs: bool = True,
    calibration_window_size: int = 50,
) -> dict:
    cfg = CFG_SPECS[cfg_id]
    judge: dict[str, Any] = {
        "type": "composite",
        "normalization": normalization_offline(),
        "factors": list(factors),
        "calibration": calibration_warmup(window_size=calibration_window_size),
        "composer": dict(composer),
        "export_factor_outputs": bool(export_factor_outputs),
    }
    return {
        "enabled": True,
        "timer": {"enabled": False, "buffer_size": 10000, "output_csv_dir": None},
        "keys": dict(cfg["keys"]),
        "key_builder": {"type": cfg["key_builder_type"]},
        "checkpoints": {
            "cp1": {
                "enabled": True,
                "gate": {"type": "always_search"},
                "judge": judge,
                "search_strategy": dict(cfg["search_strategy"]),
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


def build_warmup_yaml(
    cfg_id: str,
    eval_yaml_id: str,
    *,
    eval_factors: list[dict] | None = None,
) -> dict:
    """Sibling warmup yaml: AlwaysWarmStartJudge + DumpingJudge dumping a
    descriptor-key superset of the eval yaml(s) the warmup will serve.

    Plan §6.9 mode (b) — "over-collect": when ``eval_factors`` is given,
    the warmup's dump factor list is the union of windows that appear in
    those eval factors (per ``<descriptor>_<source>_<channel>`` group).
    When ``eval_factors`` is None, fall back to a generic W_UNION that
    spans every window currently used in the demo recipes (so any one
    of them can be served by a single warmup run).
    """
    cfg = CFG_SPECS[cfg_id]
    warmup_yaml_id = f"{eval_yaml_id}__warmup"

    full_factors = _build_dump_factor_superset(eval_factors)

    judge = {
        "type": "always_warm_start",
        "start_t": 0.7,
        "dump": {
            "deferred": True,
            "config_id": warmup_yaml_id,
            "factors": full_factors,
        },
    }
    search_strategy = dict(cfg["search_strategy"])
    search_strategy["top_k"] = 5     # F2 / topk needs at least K=5 candidates

    return {
        "enabled": True,
        "timer": {"enabled": False, "buffer_size": 10000, "output_csv_dir": None},
        "keys": dict(cfg["keys"]),
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


# ----------------------------------------------------------------------
# Generation list
# ----------------------------------------------------------------------


def _generate() -> list[tuple[str, str, list[dict], dict]]:
    """Cartesian: 1 cfg (spatial16) × 5 recipes = 5 cells.

    Extend by appending more cfg ids or recipes; the assembler is independent
    of the recipe count.
    """
    out: list[tuple[str, str, list[dict], dict]] = []
    for cfg_id in ("spatial16_w8_d4",):
        for recipe_stem, recipe_fn in RECIPES.items():
            factors_blk, composer_blk = recipe_fn()
            stem = f"{cfg_id}_v2_{recipe_stem}"
            out.append((cfg_id, stem, factors_blk, composer_blk))
    return out


GENERATION: list[tuple[str, str, list[dict], dict]] = _generate()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> None:
    config_root = Path("exp/verdict_factor_judge/config/v2")
    written: list[Path] = []

    if not GENERATION:
        raise RuntimeError("GENERATION list is empty")

    # Detect duplicate stems early.
    stems = [g[1] for g in GENERATION]
    if len(set(stems)) != len(stems):
        from collections import Counter
        dups = [s for s, c in Counter(stems).items() if c > 1]
        raise RuntimeError(f"Duplicate stems detected: {dups}")

    for cfg_id, stem, factors_blk, composer_blk in GENERATION:
        eval_yaml_id = stem
        cfg_dir = config_root / cfg_id
        eval_path = cfg_dir / f"{eval_yaml_id}.yaml"
        warmup_path = cfg_dir / f"{eval_yaml_id}__warmup.yaml"
        write_yaml(eval_path, build_eval_yaml(cfg_id, factors_blk, composer_blk))
        write_yaml(
            warmup_path,
            build_warmup_yaml(cfg_id, eval_yaml_id, eval_factors=factors_blk),
        )
        written.append(eval_path)
        written.append(warmup_path)

    print(f"Wrote {len(written)} yaml files (eval + sibling warmup)")


if __name__ == "__main__":
    main()
