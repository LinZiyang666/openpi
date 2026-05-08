"""Phase 3 spec — 11 spatial16 gold-circle recipes x 16 (FH, WS) cells.

Plan §6: defines RECIPES + GRID and exposes builders that turn each
recipe into a warmup yaml (sibling Phase 2-style) and 16 eval yamls
(after the solver has filled in the per-cell thresholds).

Public surface:

    RECIPES                                       11-entry dict (recipe_id -> dict)
    GRID                                          list[tuple[float, float]]  (4x4)
    recipe_to_solver_recipe(recipe_id)            -> phase3_threshold_solver.Recipe
    build_warmup_yaml_for_recipe(cfg_id, rid)     -> dict
    build_eval_yaml_for_cell(cfg_id, rid, fh, ws) -> dict
    main()                                        -> writes warmup + manifest

CLI:

    uv run python -m exp.verdict_factor_judge.phase3_spec
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from exp.verdict_factor_judge.generate_yamls import write_yaml
from exp.verdict_factor_judge.phase3_threshold_solver import Recipe
from exp.verdict_factor_judge.v2_spec import (
    CFG_SPECS,
    build_warmup_yaml,
    calibration_warmup,
    factor,
    factor_keys,
    normalization_offline,
)


# ----------------------------------------------------------------------
# Recipe table (plan §1.1) — 11 spatial16 gold-circle entries
# ----------------------------------------------------------------------


# Refactor descriptor orientations (mirror of _DESCRIPTOR_ORIENTATIONS in
# src/openpi/cache/components/factors/_descriptor_kernel.py). Repeated
# here so the spec module doesn't need to import the kernel just for the
# 4 string constants.
_ORIENTATION = {
    "jerk":         "risky",
    "direction":    "safe",
    "dispersion":   "non_monotonic",
    "path_length":  "non_monotonic",
}

# Default direction for non_monotonic descriptors (matches the Phase 2
# Layer 1 yaml convention: dispersion in [0.3, 0.7] band, path_length
# "high" — see plan §1.1 last bullet).
_DEFAULT_DIRECTION = {
    "dispersion":   "range:[0.3, 0.7]",
    "path_length":  "high",
}


def _factor_block(desc: str, source: str, channel: str, windows: list[dict[str, int]]) -> dict:
    return factor(f"{desc}_{source}_{channel}", windows=list(windows))


def _keys_for(desc: str, source: str, channel: str, windows: list[dict[str, int]]) -> list[str]:
    return factor_keys(f"{desc}_{source}_{channel}", windows=windows)


def _multi_desc_factors(
    descs: tuple[str, ...],
    source: str,
    channel: str,
    windows: list[dict[str, int]],
) -> list[dict]:
    return [_factor_block(d, source, channel, windows) for d in descs]


def _multi_desc_keys(
    descs: tuple[str, ...],
    source: str,
    channel: str,
    windows: list[dict[str, int]],
) -> list[str]:
    out: list[str] = []
    for d in descs:
        out.extend(_keys_for(d, source, channel, windows))
    return out


def _orientations_for(keys: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k in keys:
        # Key shape: <desc>_<source>_<channel>__pP_fF
        desc = k.split("_", 1)[0]
        if desc == "path":
            desc = "path_length"
        out[k] = _ORIENTATION[desc]
    return out


def _directions_for(keys: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k in keys:
        desc = k.split("_", 1)[0]
        if desc == "path":
            desc = "path_length"
        if desc in _DEFAULT_DIRECTION:
            out[k] = _DEFAULT_DIRECTION[desc]
    return out


# Window banks (plan §1.1)
_W_FUT       = [{"past": 0, "future": 3}, {"past": 0, "future": 5}]
_W_LR        = [{"past": 5, "future": 5}, {"past": 7, "future": 7}]
_W_SHORT     = [{"past": 0, "future": 3}, {"past": 1, "future": 1}, {"past": 3, "future": 0}]
_W_SYM_S     = [{"past": 1, "future": 1}, {"past": 2, "future": 2}, {"past": 3, "future": 3}]
_W_K3        = [{"past": 3, "future": 3}]


_ALL_DESC = ("jerk", "direction", "dispersion", "path_length")


def _make_recipe(
    rid: str,
    factors_blocks: list[dict],
    declared_keys: list[str],
) -> dict[str, Any]:
    return {
        "recipe_id": rid,
        "factors": factors_blocks,
        "declared_keys": declared_keys,
        "orientations": _orientations_for(declared_keys),
        "directions": _directions_for(declared_keys),
    }


def _r_g1() -> dict[str, Any]:
    keys = _multi_desc_keys(_ALL_DESC, "offline", "state", _W_FUT)
    return _make_recipe(
        "g1_f1b_t_w_fut_d_all",
        _multi_desc_factors(_ALL_DESC, "offline", "state", _W_FUT),
        keys,
    )


def _r_g2() -> dict[str, Any]:
    keys = _keys_for("jerk", "offline", "state", _W_LR)
    return _make_recipe(
        "g2_f1b_t_w_long_risk_d_jerk",
        [_factor_block("jerk", "offline", "state", _W_LR)],
        keys,
    )


def _r_g3() -> dict[str, Any]:
    keys = _multi_desc_keys(_ALL_DESC, "offline", "state", _W_LR)
    return _make_recipe(
        "g3_f1b_t_w_long_risk_d_all",
        _multi_desc_factors(_ALL_DESC, "offline", "state", _W_LR),
        keys,
    )


def _r_g4() -> dict[str, Any]:
    keys = _keys_for("jerk", "offline", "state", _W_SHORT)
    return _make_recipe(
        "g4_f1b_t_w_short_d_jerk",
        [_factor_block("jerk", "offline", "state", _W_SHORT)],
        keys,
    )


def _r_g5() -> dict[str, Any]:
    keys = _multi_desc_keys(("jerk", "direction"), "online", "state", _W_K3)
    return _make_recipe(
        "g5_f1a_t_d_jerk_dir_pair",
        _multi_desc_factors(("jerk", "direction"), "online", "state", _W_K3),
        keys,
    )


def _r_g6() -> dict[str, Any]:
    keys = _multi_desc_keys(("jerk", "dispersion"), "online", "action", _W_K3)
    return _make_recipe(
        "g6_f1a_a_d_jerk_curv_pair",
        _multi_desc_factors(("jerk", "dispersion"), "online", "action", _W_K3),
        keys,
    )


def _r_g7() -> dict[str, Any]:
    keys = _keys_for("jerk", "offline", "action", _W_LR)
    return _make_recipe(
        "g7_f1b_a_w_long_risk_d_jerk",
        [_factor_block("jerk", "offline", "action", _W_LR)],
        keys,
    )


def _r_g8() -> dict[str, Any]:
    keys = _keys_for("dispersion", "online", "state", _W_K3)
    return _make_recipe(
        "g8_f1a_t_d_curv_only",
        [_factor_block("dispersion", "online", "state", _W_K3)],
        keys,
    )


def _r_g9() -> dict[str, Any]:
    keys = _multi_desc_keys(_ALL_DESC, "offline", "state", _W_SYM_S)
    return _make_recipe(
        "g9_f1b_t_w_sym_s_d_all",
        _multi_desc_factors(_ALL_DESC, "offline", "state", _W_SYM_S),
        keys,
    )


def _r_g10() -> dict[str, Any]:
    keys = _multi_desc_keys(_ALL_DESC, "offline", "action", _W_FUT)
    return _make_recipe(
        "g10_f1b_a_w_fut_d_all",
        _multi_desc_factors(_ALL_DESC, "offline", "action", _W_FUT),
        keys,
    )


def _r_g11() -> dict[str, Any]:
    keys = _keys_for("dispersion", "online", "action", _W_K3)
    return _make_recipe(
        "g11_f1a_a_d_curv_only",
        [_factor_block("dispersion", "online", "action", _W_K3)],
        keys,
    )


RECIPES: dict[str, dict[str, Any]] = {
    r["recipe_id"]: r
    for r in (
        _r_g1(), _r_g2(), _r_g3(), _r_g4(), _r_g5(), _r_g6(),
        _r_g7(), _r_g8(), _r_g9(), _r_g10(), _r_g11(),
    )
}


# ----------------------------------------------------------------------
# 4x4 (FH_ratio, WS_ratio) grid (plan §0.2 / §0.3)
# ----------------------------------------------------------------------


GRID: list[tuple[float, float]] = [
    (fh, ws) for fh in (0.2, 0.3, 0.4, 0.5) for ws in (0.2, 0.3, 0.4, 0.5)
]


# ----------------------------------------------------------------------
# Recipe -> solver Recipe shim
# ----------------------------------------------------------------------


def recipe_to_solver_recipe(recipe_id: str) -> Recipe:
    """Turn a phase3_spec RECIPES[recipe_id] entry into a solver Recipe."""
    r = RECIPES[recipe_id]
    return Recipe(
        recipe_id=recipe_id,
        declared_keys=list(r["declared_keys"]),
        orientations=dict(r["orientations"]),
        directions=dict(r["directions"]),
    )


# ----------------------------------------------------------------------
# Yaml builders
# ----------------------------------------------------------------------


def _eval_yaml_id(cfg_id: str, recipe_id: str, fh: float, ws: float) -> str:
    return f"{cfg_id}_phase3_{recipe_id}__fh{fh}_ws{ws}"


def _warmup_eval_yaml_id(cfg_id: str, recipe_id: str) -> str:
    """The eval yaml id the warmup yaml is sibling to.

    All 16 (fh, ws) cells of a recipe share one warmup yaml — the
    warmup id is anchored to the recipe stem, not to a specific cell.
    """
    return f"{cfg_id}_phase3_{recipe_id}"


def build_warmup_yaml_for_recipe(cfg_id: str, recipe_id: str) -> dict:
    """Single warmup yaml per recipe (shared across 16 cells)."""
    r = RECIPES[recipe_id]
    return build_warmup_yaml(
        cfg_id=cfg_id,
        eval_yaml_id=_warmup_eval_yaml_id(cfg_id, recipe_id),
        eval_factors=r["factors"],
    )


def build_eval_yaml_for_cell(
    cfg_id: str,
    recipe_id: str,
    fh_thr: float,
    ws_thr: float,
    *,
    fh_ratio: float,
    ws_ratio: float,
    warm_start_t: float = 0.5,
) -> dict:
    """Final eval yaml with thresholds patched in by the solver.

    The yaml shape mirrors v2_spec.build_eval_yaml's output (composite
    judge, normalization=zscore offline, calibration=percentile_rolling
    samples_source=warmup, composer=weighted_sum_zero_nan).
    """
    r = RECIPES[recipe_id]
    cfg = CFG_SPECS[cfg_id]
    declared_keys = r["declared_keys"]
    weights = {k: 1.0 for k in declared_keys}

    composer: dict[str, Any] = {
        "type": "weighted_sum_zero_nan",
        "weights": weights,
        "tier_thresholds": {"full_hit": float(fh_thr), "warm_start": float(ws_thr)},
        "warm_start_t": float(warm_start_t),
    }
    if r["directions"]:
        composer["directions"] = dict(r["directions"])

    judge: dict[str, Any] = {
        "type": "composite",
        "normalization": normalization_offline(),
        "factors": list(r["factors"]),
        "calibration": calibration_warmup(window_size=50),
        "composer": composer,
        "export_factor_outputs": True,
    }
    return {
        "enabled": True,
        "timer": {"enabled": False, "buffer_size": 10000, "output_csv_dir": None},
        "keys": {k: {"enabled": True, "weight": 1.0} for k in cfg["vector_dims"]},
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


# ----------------------------------------------------------------------
# CLI: emit 11 warmup yamls + manifest (eval yamls written by runner
# after solver fills thresholds).
# ----------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(prog="phase3_spec")
    p.add_argument(
        "--cfg-id", default="spatial16_w8_d4",
        help="cfg key (default spatial16_w8_d4 — Phase 3 is spatial16-only)",
    )
    p.add_argument(
        "--out-dir", type=Path,
        default=Path("exp/verdict_factor_judge/config"),
    )
    args = p.parse_args()

    cfg_short = args.cfg_id.split("_")[0]   # spatial16, max_pool, clip
    base = args.out_dir / cfg_short / "phase3"
    warmup_dir = base / "warmup"
    warmup_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "cfg_id": args.cfg_id,
        "recipes": [],
        "grid": [list(c) for c in GRID],
    }

    for recipe_id in RECIPES:
        warmup_yaml = build_warmup_yaml_for_recipe(args.cfg_id, recipe_id)
        warmup_id = f"{_warmup_eval_yaml_id(args.cfg_id, recipe_id)}__warmup"
        path = warmup_dir / f"{warmup_id}.yaml"
        write_yaml(path, warmup_yaml)
        manifest["recipes"].append(
            {
                "recipe_id": recipe_id,
                "warmup_yaml_id": warmup_id,
                "warmup_yaml_path": str(path.relative_to(args.out_dir.parent)),
            }
        )

    (base / "phase3_manifest.json").write_text(
        __import__("json").dumps(manifest, indent=2) + "\n"
    )
    print(f"wrote {len(RECIPES)} warmup yamls + manifest at {base}")


if __name__ == "__main__":
    main()
