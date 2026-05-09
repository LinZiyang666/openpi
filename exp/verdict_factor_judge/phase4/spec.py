"""Phase 4 spec — 2 fused 10-factor recipes, weight sweep over locked cells.

Phase 4 fuses Phase 3 winners g1 + g6 (p1) and g10 + g6 (p2) into two
10-score recipes and sweeps weight patterns over per-recipe-locked
(FH_ratio, WS_ratio) cells. The recipe table here is local to phase 4 —
``phase3_spec.RECIPES`` is **never** mutated.

Public surface:

    RECIPES_PHASE4                                 2-entry dict (recipe_id -> Phase4Recipe)
    R1_ALPHAS / R2_OFFLINE_PATTERNS /              round-specific weight pattern banks
        R3_ONLINE_PATTERNS / R4_WINDOW_PATTERNS
    LOCKED_CELLS                                   per-recipe (fh_ratio, ws_ratio) anchors
    generate_r{1,2,3,4}_weights(...)               pure functions, weight dict per cell
    cell_id_r{1,2,3,4}(...) / warmup_yaml_id(...)  yaml id naming helpers
    recipe_to_solver_recipe(recipe_id)             -> phase3_threshold_solver.Recipe
    build_phase4_warmup_yaml(cfg_id, recipe_id)    -> dict
    build_phase4_eval_yaml(...)                    -> dict (composer weights are required)

Phase 4 plan: ``logs/verdict_phase4_weight_sweep.log.md``.
"""

from __future__ import annotations

from typing import Any, TypedDict

from exp.verdict_factor_judge.common.v2_spec import (
    build_eval_yaml as v2_build_eval_yaml,
    build_warmup_yaml as v2_build_warmup_yaml,
)
from exp.verdict_factor_judge.phase3.spec import (
    _W_FUT,
    _W_K3,
    _multi_desc_factors,
    _orientations_for,
    _directions_for,
)
from exp.verdict_factor_judge.phase3.threshold_solver import Recipe


# ----------------------------------------------------------------------
# Score key constants (plan §4.1.1)
# ----------------------------------------------------------------------


# Order matters: the desc-axis (jerk, direction, dispersion, path_length)
# is used by R2 patterns; the window-axis ((0,3), (0,5)) is used by R4.
DESC_ORDER: tuple[str, ...] = ("jerk", "direction", "dispersion", "path_length")
WINDOW_ORDER: tuple[tuple[int, int], ...] = ((0, 3), (0, 5))
ONLINE_DESC_ORDER: tuple[str, ...] = ("jerk", "dispersion")
ONLINE_WINDOW: tuple[int, int] = (3, 3)


def _key(desc: str, source: str, channel: str, past: int, future: int) -> str:
    """Mirror the canonical ``factor_keys(...)`` output for a single window."""
    return f"{desc}_{source}_{channel}__p{past}_f{future}"


# Eight offline keys per recipe (4 desc x 2 W-FUT windows).
P1_OFFLINE_KEYS: tuple[str, ...] = tuple(
    _key(d, "offline", "state", p, f)
    for d in DESC_ORDER for (p, f) in WINDOW_ORDER
)
P2_OFFLINE_KEYS: tuple[str, ...] = tuple(
    _key(d, "offline", "action", p, f)
    for d in DESC_ORDER for (p, f) in WINDOW_ORDER
)

# Two online keys shared by both recipes (W-K3 = (3, 3)).
SHARED_ONLINE_KEYS: tuple[str, ...] = tuple(
    _key(d, "online", "action", *ONLINE_WINDOW) for d in ONLINE_DESC_ORDER
)


# ----------------------------------------------------------------------
# Factor blocks (plan §4.1.2 — reuse phase3_spec helpers)
# ----------------------------------------------------------------------


_ALL_DESC = DESC_ORDER

P1_OFFLINE_FACTORS = _multi_desc_factors(_ALL_DESC, "offline", "state", _W_FUT)
P2_OFFLINE_FACTORS = _multi_desc_factors(_ALL_DESC, "offline", "action", _W_FUT)
SHARED_ONLINE_FACTORS = _multi_desc_factors(ONLINE_DESC_ORDER, "online", "action", _W_K3)


# ----------------------------------------------------------------------
# Recipe metadata (plan §4.1.3)
# ----------------------------------------------------------------------


class Phase4Recipe(TypedDict):
    recipe_id: str
    factors: list[dict]
    offline_keys: tuple[str, ...]
    online_keys: tuple[str, ...]
    declared_keys: tuple[str, ...]
    orientations: dict[str, str]
    directions: dict[str, str]


def _make_recipe(
    rid: str,
    offline_factors: list[dict],
    offline_keys: tuple[str, ...],
) -> Phase4Recipe:
    declared = offline_keys + SHARED_ONLINE_KEYS
    return {
        "recipe_id": rid,
        "factors": list(offline_factors) + list(SHARED_ONLINE_FACTORS),
        "offline_keys": offline_keys,
        "online_keys": SHARED_ONLINE_KEYS,
        "declared_keys": declared,
        "orientations": _orientations_for(list(declared)),
        "directions": _directions_for(list(declared)),
    }


def _make_p1() -> Phase4Recipe:
    return _make_recipe("p1_state_fut_online_act", P1_OFFLINE_FACTORS, P1_OFFLINE_KEYS)


def _make_p2() -> Phase4Recipe:
    return _make_recipe("p2_action_fut_online_act", P2_OFFLINE_FACTORS, P2_OFFLINE_KEYS)


RECIPES_PHASE4: dict[str, Phase4Recipe] = {
    r["recipe_id"]: r for r in (_make_p1(), _make_p2())
}


# ----------------------------------------------------------------------
# Round pattern banks (plan §4.1.4)
# ----------------------------------------------------------------------


R1_ALPHAS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)

# offline 4-desc shares assigned to (jerk, direction, dispersion, path_length)
R2_OFFLINE_PATTERNS: dict[str, tuple[int, int, int, int]] = {
    "uniform":     (1, 1, 1, 1),
    "jerk-heavy":  (2, 1, 1, 1),
    "dir-heavy":   (1, 2, 1, 1),
    "disp-heavy":  (1, 1, 2, 1),
    "path-heavy":  (1, 1, 1, 2),
    "jerk-only":   (1, 0, 0, 0),
    "dir-only":    (0, 1, 0, 0),
    "disp-only":   (0, 0, 1, 0),
    "path-only":   (0, 0, 0, 1),
}

# online 2-desc shares assigned to (jerk_online_action, dispersion_online_action)
R3_ONLINE_PATTERNS: dict[str, tuple[int, int]] = {
    "uniform":     (1, 1),
    "jerk-heavy":  (2, 1),
    "disp-heavy":  (1, 2),
    "jerk-only":   (1, 0),
    "disp-only":   (0, 1),
}

# W-FUT window shares assigned to ((0, 3), (0, 5)) within each desc
R4_WINDOW_PATTERNS: dict[str, tuple[int, int]] = {
    "uniform":     (1, 1),
    "short-heavy": (2, 1),
    "long-heavy":  (1, 2),
    "short-only":  (1, 0),
    "long-only":   (0, 1),
}


# Per-recipe locked (FH_ratio, WS_ratio) anchors (plan §0.2).
LOCKED_CELLS: dict[str, tuple[float, float]] = {
    "p1_state_fut_online_act":  (0.5, 0.5),
    "p2_action_fut_online_act": (0.5, 0.4),
}

# Per-recipe phase3 SR anchors (plan §0.2 / §3.1) — used by G_R1.
ANCHOR_SR: dict[str, float] = {
    "p1_state_fut_online_act":  0.95,
    "p2_action_fut_online_act": 0.96,
}


# ----------------------------------------------------------------------
# Weight generators (plan §4.1.5)
# ----------------------------------------------------------------------


def _channel_for(recipe_id: str) -> str:
    if recipe_id == "p1_state_fut_online_act":
        return "state"
    if recipe_id == "p2_action_fut_online_act":
        return "action"
    raise KeyError(f"unknown phase4 recipe id: {recipe_id!r}")


def _alloc_offline(
    recipe_id: str,
    alpha: float,
    desc_pattern: tuple[int, int, int, int],
    window_pattern: tuple[int, int],
) -> dict[str, float]:
    """Distribute ``alpha`` over 8 offline keys = 4 desc x 2 windows.

    For each desc i, allocation = alpha * (desc_pattern[i] / sum(desc_pattern)).
    Within desc, the two windows split that allocation by window_pattern.
    Zero shares produce zero weights (the composer treats zero-weight
    keys as not declared).
    """
    if len(desc_pattern) != 4:
        raise ValueError(f"desc_pattern must have 4 entries, got {desc_pattern}")
    if len(window_pattern) != 2:
        raise ValueError(f"window_pattern must have 2 entries, got {window_pattern}")
    desc_total = sum(desc_pattern)
    win_total = sum(window_pattern)
    channel = _channel_for(recipe_id)
    out: dict[str, float] = {}
    for d_idx, desc in enumerate(DESC_ORDER):
        desc_share = desc_pattern[d_idx]
        desc_alloc = (alpha * desc_share / desc_total) if desc_total > 0 else 0.0
        for w_idx, (p, f) in enumerate(WINDOW_ORDER):
            key = _key(desc, "offline", channel, p, f)
            win_share = window_pattern[w_idx]
            win_alloc = (
                (desc_alloc * win_share / win_total) if win_total > 0 else 0.0
            )
            out[key] = win_alloc
    return out


def _alloc_online(
    online_total: float,
    online_pattern: tuple[int, int],
) -> dict[str, float]:
    """Distribute ``online_total`` over the 2 shared online keys."""
    if len(online_pattern) != 2:
        raise ValueError(f"online_pattern must have 2 entries, got {online_pattern}")
    total = sum(online_pattern)
    if total == 0:
        return {k: 0.0 for k in SHARED_ONLINE_KEYS}
    return {
        SHARED_ONLINE_KEYS[i]: online_total * online_pattern[i] / total
        for i in range(2)
    }


def _validate_weight_dict(recipe_id: str, weights: dict[str, float]) -> None:
    """Ensure the dict keys exactly match the recipe's declared_keys (10 keys)."""
    expected = set(RECIPES_PHASE4[recipe_id]["declared_keys"])
    got = set(weights)
    if got != expected:
        missing = expected - got
        extra = got - expected
        raise AssertionError(
            f"phase4 weight dict mismatch for {recipe_id!r}: "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )


def generate_r1_weights(recipe_id: str, alpha: float) -> dict[str, float]:
    """Round 1: offline desc uniform (1,1,1,1), online uniform, windows uniform."""
    if alpha not in R1_ALPHAS:
        raise ValueError(f"alpha {alpha!r} not in R1_ALPHAS {R1_ALPHAS}")
    off = _alloc_offline(recipe_id, alpha, (1, 1, 1, 1), (1, 1))
    on = _alloc_online(1.0 - alpha, (1, 1))
    out = {**off, **on}
    _validate_weight_dict(recipe_id, out)
    return out


def generate_r2_weights(
    recipe_id: str,
    alpha_star: float,
    offline_pattern: tuple[int, int, int, int],
) -> dict[str, float]:
    """Round 2: alpha locked at alpha_star, online uniform, windows uniform."""
    off = _alloc_offline(recipe_id, alpha_star, offline_pattern, (1, 1))
    on = _alloc_online(1.0 - alpha_star, (1, 1))
    out = {**off, **on}
    _validate_weight_dict(recipe_id, out)
    return out


def generate_r3_weights(
    recipe_id: str,
    alpha_star: float,
    offline_pattern: tuple[int, int, int, int],
    online_pattern: tuple[int, int],
) -> dict[str, float]:
    """Round 3: locked alpha + offline pattern; sweep online pattern."""
    off = _alloc_offline(recipe_id, alpha_star, offline_pattern, (1, 1))
    on = _alloc_online(1.0 - alpha_star, online_pattern)
    out = {**off, **on}
    _validate_weight_dict(recipe_id, out)
    return out


def generate_r4_weights(
    recipe_id: str,
    alpha_star: float,
    offline_pattern: tuple[int, int, int, int],
    online_pattern: tuple[int, int],
    window_pattern: tuple[int, int],
) -> dict[str, float]:
    """Round 4: locked alpha + offline + online; sweep window pattern."""
    off = _alloc_offline(recipe_id, alpha_star, offline_pattern, window_pattern)
    on = _alloc_online(1.0 - alpha_star, online_pattern)
    out = {**off, **on}
    _validate_weight_dict(recipe_id, out)
    return out


# ----------------------------------------------------------------------
# Cell + yaml id helpers (plan §4.1.6)
# ----------------------------------------------------------------------


def cell_id_r1(cfg_id: str, recipe_id: str, alpha: float) -> str:
    return f"{cfg_id}_phase4_{recipe_id}__r1_a{alpha:.1f}"


def cell_id_r2(cfg_id: str, recipe_id: str, alpha: float, off_pat: str) -> str:
    return f"{cfg_id}_phase4_{recipe_id}__r2_a{alpha:.1f}_off-{off_pat}"


def cell_id_r3(
    cfg_id: str, recipe_id: str, alpha: float, off_pat: str, on_pat: str,
) -> str:
    return (
        f"{cfg_id}_phase4_{recipe_id}__r3_a{alpha:.1f}_off-{off_pat}_on-{on_pat}"
    )


def cell_id_r4(
    cfg_id: str, recipe_id: str, alpha: float,
    off_pat: str, on_pat: str, win_pat: str,
) -> str:
    return (
        f"{cfg_id}_phase4_{recipe_id}__r4_a{alpha:.1f}"
        f"_off-{off_pat}_on-{on_pat}_win-{win_pat}"
    )


def warmup_yaml_id(cfg_id: str, recipe_id: str) -> str:
    """Sibling warmup yaml id for a phase4 recipe (shared across all rounds)."""
    return f"{cfg_id}_phase4_{recipe_id}__warmup"


def warmup_eval_yaml_id(cfg_id: str, recipe_id: str) -> str:
    """Anchor eval yaml id used by warmup buffer keying."""
    return f"{cfg_id}_phase4_{recipe_id}"


# ----------------------------------------------------------------------
# Solver bridge (plan §4.2.6)
# ----------------------------------------------------------------------


def recipe_to_solver_recipe(recipe_id: str) -> Recipe:
    r = RECIPES_PHASE4[recipe_id]
    return Recipe(
        recipe_id=recipe_id,
        declared_keys=list(r["declared_keys"]),
        orientations=dict(r["orientations"]),
        directions=dict(r["directions"]),
    )


# ----------------------------------------------------------------------
# Yaml builders (plan §4.1.7) — call v2_spec directly, do NOT touch
# ``phase3_spec.RECIPES``.
# ----------------------------------------------------------------------


def build_phase4_warmup_yaml(cfg_id: str, recipe_id: str) -> dict:
    """Per-recipe warmup yaml; reused across all rounds."""
    r = RECIPES_PHASE4[recipe_id]
    return v2_build_warmup_yaml(
        cfg_id=cfg_id,
        eval_yaml_id=warmup_eval_yaml_id(cfg_id, recipe_id),
        eval_factors=r["factors"],
    )


def _build_phase4_composer_dict(
    recipe_id: str,
    composer_weights: dict[str, float],
    fh_thr: float,
    ws_thr: float,
    warm_start_t: float = 0.5,
) -> dict[str, Any]:
    """Materialize the composer block consumed by v2_spec.build_eval_yaml."""
    _validate_weight_dict(recipe_id, composer_weights)
    r = RECIPES_PHASE4[recipe_id]
    composer: dict[str, Any] = {
        "type": "weighted_sum_zero_nan",
        "weights": dict(composer_weights),
        "tier_thresholds": {
            "full_hit": float(fh_thr),
            "warm_start": float(ws_thr),
        },
        "warm_start_t": float(warm_start_t),
    }
    if r["directions"]:
        composer["directions"] = dict(r["directions"])
    return composer


def build_phase4_eval_yaml(
    cfg_id: str,
    recipe_id: str,
    fh_thr: float,
    ws_thr: float,
    *,
    fh_ratio: float,
    ws_ratio: float,
    composer_weights: dict[str, float],
    warm_start_t: float = 0.5,
) -> dict:
    """Phase 4 eval yaml. ``composer_weights`` MUST contain exactly the
    10 declared keys (4 offline desc x 2 windows + 2 online desc).

    ``fh_ratio`` / ``ws_ratio`` are recorded only for traceability — the
    actual gate behavior comes from ``fh_thr`` / ``ws_thr`` which the
    solver derived under those target ratios.
    """
    # fh_ratio / ws_ratio are not used by v2_spec, but we still echo
    # them in the surrounding tooling (cell id, summary jsonl).
    del fh_ratio, ws_ratio
    r = RECIPES_PHASE4[recipe_id]
    composer = _build_phase4_composer_dict(
        recipe_id, composer_weights, fh_thr, ws_thr, warm_start_t=warm_start_t,
    )
    return v2_build_eval_yaml(
        cfg_id=cfg_id,
        factors=r["factors"],
        composer=composer,
        export_factor_outputs=True,
        calibration_window_size=50,
    )


__all__ = [
    "ANCHOR_SR",
    "DESC_ORDER",
    "LOCKED_CELLS",
    "ONLINE_DESC_ORDER",
    "ONLINE_WINDOW",
    "P1_OFFLINE_KEYS",
    "P2_OFFLINE_KEYS",
    "Phase4Recipe",
    "R1_ALPHAS",
    "R2_OFFLINE_PATTERNS",
    "R3_ONLINE_PATTERNS",
    "R4_WINDOW_PATTERNS",
    "RECIPES_PHASE4",
    "SHARED_ONLINE_KEYS",
    "WINDOW_ORDER",
    "build_phase4_eval_yaml",
    "build_phase4_warmup_yaml",
    "cell_id_r1",
    "cell_id_r2",
    "cell_id_r3",
    "cell_id_r4",
    "generate_r1_weights",
    "generate_r2_weights",
    "generate_r3_weights",
    "generate_r4_weights",
    "recipe_to_solver_recipe",
    "warmup_eval_yaml_id",
    "warmup_yaml_id",
]
