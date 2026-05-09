"""Phase 5 spec — 5 group × 48 cell systematic online-factor sweep (240 cells).

See `logs/verdict_phase5_systematic_sweep.log.md` for the full plan.

Public surface:

    Cell                                          frozen dataclass
    G1_WINDOWS / G1_CHANNELS / G1_DESCS / ...     axis enums (G1..G5)
    G4_FACTOR_SUBSETS                             12 ad-hoc (off, on) subsets
    G5_THRESHOLD_GRID / G5_RECIPES                threshold sweep config
    G3_WEIGHT_PATTERNS / G2_MULTI_COMBOS          remaining axes
    generate_g1_cells / g2 / g3 / g4 / g5         per-group generators (48 each)
    generate_all_cells()                          240 cells, dict-order sorted
    allocate_to_servers(cells, ratio)             {server_id: list[Cell]}
    build_warmup_yaml_for_cell(cell)              dict (yaml content)
    build_eval_yaml_for_cell(cell, fh, ws)        dict (yaml content)
    cell_to_solver_recipe(cell)                   threshold_solver.Recipe
    warmup_raw_path_for_cell(cell, args)          jsonl path (G5 dual-source / G1-4 own)

CLI:

    uv run python -m exp.verdict_factor_judge.phase5.spec
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

from exp.verdict_factor_judge.common.generate_yamls import write_yaml
from exp.verdict_factor_judge.common.v2_spec import (
    CFG_SPECS,
    build_eval_yaml as v2_build_eval_yaml,
    build_warmup_yaml as v2_build_warmup_yaml,
    calibration_warmup,
    factor,
    factor_keys,
    normalization_offline,
)
from exp.verdict_factor_judge.phase3.spec import (
    _W_FUT,
    _W_K3,
    _directions_for,
    _orientations_for,
    RECIPES as PHASE3_RECIPES,
)
from exp.verdict_factor_judge.phase3.threshold_solver import Recipe
from exp.verdict_factor_judge.phase4.spec import RECIPES_PHASE4


# ----------------------------------------------------------------------
# Constants — cfg + base recipe spec
# ----------------------------------------------------------------------


CFG_ID_DEFAULT = "spatial16_w8_d4"

# Locked (FH, WS) anchor for G1-G4 (phase3 g1 ultra-cheap cell).
LOCKED_FH = 0.5
LOCKED_WS = 0.5

WARM_START_T = 0.5  # warm cost = 0.75 (mirror phase3/phase4)

# base_recipe → (channel, factor-source-tag). Determines which 8 offline
# keys are injected into G1-G4 cells. G5 does NOT inject base offline.
BASE_RECIPE_TO_CHANNEL: dict[str, str] = {
    "p1": "state",   # = phase3 g1: jerk/dir/disp/path × offline × state × W-FUT
    "p2": "action",  # = phase3 g10: jerk/dir/disp/path × offline × action × W-FUT
}

OFFLINE_DESCS: tuple[str, ...] = ("jerk", "direction", "dispersion", "path_length")

ONLINE_DESCS: tuple[str, ...] = ("jerk", "dispersion")


# ----------------------------------------------------------------------
# Axis enums per group (plan §1.2..§1.6)
# ----------------------------------------------------------------------


# G1 — single online window
G1_WINDOWS: tuple[tuple[int, int], ...] = (
    (0, 3), (0, 5), (1, 1), (3, 3), (5, 5), (7, 7),
)
G1_CHANNELS: tuple[str, ...] = ("state", "action")
G1_DESCS: tuple[str, ...] = ("jerk", "dispersion")
G1_BASE_RECIPES: tuple[str, ...] = ("p1", "p2")


# G2 — multi-window combos. Each entry: (windows-tuple, short-tag).
G2_MULTI_COMBOS: tuple[tuple[tuple[tuple[int, int], ...], str], ...] = (
    (((0, 3), (3, 3)),                          "fut03+sym33"),
    (((0, 5), (3, 3)),                          "fut05+sym33"),
    (((0, 3), (5, 5)),                          "fut03+sym55"),
    (((0, 3), (3, 0)),                          "fut03+past30"),
    (((0, 5), (5, 0)),                          "fut05+past50"),
    (((1, 1), (3, 3)),                          "sym11+sym33"),
    (((3, 3), (5, 5)),                          "sym33+sym55"),
    (((5, 5), (7, 7)),                          "sym55+sym77"),
    (((1, 1), (3, 3), (5, 5)),                  "sym-tower3"),
    (((0, 3), (3, 3), (5, 5)),                  "fut+sym-tower3"),
    (((0, 3), (1, 1), (3, 3), (5, 5)),          "small4"),
    (((0, 3), (0, 5), (3, 3), (5, 5), (7, 7)),  "ladder5"),
)
G2_CHANNELS: tuple[str, ...] = ("state", "action")
G2_DESCS: tuple[str, ...] = ("jerk", "dispersion")
G2_BASE_RECIPES: tuple[str, ...] = ("p1",)   # plan §1.3: p1 only


# G3 — online weight patterns. Each entry: (jerk_share, disp_share, tag).
G3_WEIGHT_PATTERNS: tuple[tuple[int, int, str], ...] = (
    (1, 1, "uniform"),
    (2, 1, "jerk-heavy"),
    (1, 2, "disp-heavy"),
    (1, 0, "jerk-only"),
    (0, 1, "disp-only"),
    (3, 1, "jerk-strong"),
    (1, 3, "disp-strong"),
    (4, 1, "jerk-dominant"),
    (1, 4, "disp-dominant"),
    (2, 3, "slight-disp"),
    (3, 2, "slight-jerk"),
    (5, 1, "jerk-extreme"),
)
G3_CHANNELS: tuple[str, ...] = ("state", "action")
G3_BASE_RECIPES: tuple[str, ...] = ("p1", "p2")


# G4 — multi-factor subsets. Each entry encodes:
#   (off_descs, on_descs, off_windows_override, on_windows_override, tag).
# off_windows None ⇒ default W-FUT [(0,3), (0,5)].
# on_windows  None ⇒ default W-K3 [(3,3)].
G4_FACTOR_SUBSETS: tuple[
    tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[int, int], ...] | None,
          tuple[tuple[int, int], ...] | None, str], ...
] = (
    (("jerk", "direction", "dispersion", "path_length"),
     ("jerk", "dispersion"),                None, None, "full"),
    (("jerk", "direction", "dispersion"),
     ("jerk", "dispersion"),                None, None, "drop-off-path"),
    (("jerk", "direction", "path_length"),
     ("jerk", "dispersion"),                None, None, "drop-off-disp"),
    (("jerk", "dispersion", "path_length"),
     ("jerk", "dispersion"),                None, None, "drop-off-dir"),
    (("direction", "dispersion", "path_length"),
     ("jerk", "dispersion"),                None, None, "drop-off-jerk"),
    (("jerk", "direction", "dispersion", "path_length"),
     (),                                    None, None, "off-only"),
    ((),
     ("jerk", "dispersion"),                None, None, "on-only"),
    (("jerk",),
     ("jerk",),                             None, None, "jerk-pair"),
    (("dispersion",),
     ("dispersion",),                       None, None, "disp-pair"),
    (("jerk", "direction", "dispersion", "path_length"),
     ("jerk", "dispersion"),                ((0, 3),), None, "off-1win-on-full"),
    (("jerk",),
     ("jerk", "dispersion"),                None, None, "off-jerk-on-full"),
    # 12. single jerk full stack (plan §1.5 last entry): offline jerk on
    # W-FUT + W-K3 (3 windows) + online jerk on W-K3 (1 window) → 4 keys.
    (("jerk",),
     ("jerk",),
     ((0, 3), (0, 5), (3, 3)),              ((3, 3),),  "jerk-full-stack"),
)
G4_CHANNELS: tuple[str, ...] = ("state", "action")
G4_BASE_RECIPES: tuple[str, ...] = ("p1", "p2")


# G5 — (FH, WS) grid × 3 recipe (no base offline injection).
G5_THRESHOLD_GRID: tuple[tuple[float, float], ...] = tuple(
    (fh, ws) for fh in (0.2, 0.3, 0.4, 0.5) for ws in (0.2, 0.3, 0.4, 0.5)
)
# For G5 we use existing RECIPES_PHASE4 entries + the phase3 g6 recipe.
G5_RECIPES: tuple[str, ...] = (
    "p1_state_fut_online_act",
    "p2_action_fut_online_act",
    "g6_f1a_a_d_jerk_curv_pair",
)


# ----------------------------------------------------------------------
# Cell dataclass (plan §3.1)
# ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Cell:
    """A single phase5 evaluation point.

    ``weights`` is a dict (frozen=True with dict is intentional —
    immutable identity but dict not hashed; mirror phase4.runner.Cell).

    ``warmup_yaml_id`` is the bare stem (no ``.yaml`` suffix) and matches
    the warmup factor_raw jsonl filename for G1/G2/G3/G4 cells. For G5
    the field points to the historical phase3/phase4 warmup id.
    """

    yaml_id: str
    group: str             # "g1" | "g2" | "g3" | "g4" | "g5"
    base_recipe: str       # "p1" | "p2" | "g6" (G5)
    axis_tag: str          # short tag describing the axis instance
    factors: tuple[dict, ...]
    weights: dict[str, float]
    fh_ratio: float
    ws_ratio: float
    declared_keys: tuple[str, ...]
    warmup_yaml_id: str    # bare stem (used by warmup raw jsonl + warmup yaml)


# ----------------------------------------------------------------------
# Helpers — factor / key construction
# ----------------------------------------------------------------------


def _windows_dict(wins: tuple[tuple[int, int], ...]) -> list[dict[str, int]]:
    return [{"past": p, "future": f} for (p, f) in wins]


def _factor_block(
    desc: str, source: str, channel: str, wins: tuple[tuple[int, int], ...],
) -> dict:
    return factor(f"{desc}_{source}_{channel}", windows=_windows_dict(wins))


def _keys_of(
    desc: str, source: str, channel: str, wins: tuple[tuple[int, int], ...],
) -> list[str]:
    return factor_keys(f"{desc}_{source}_{channel}", windows=_windows_dict(wins))


def _base_offline_factors(base_recipe: str) -> list[dict]:
    """8-key offline factor list per base recipe."""
    channel = BASE_RECIPE_TO_CHANNEL[base_recipe]
    return [
        _factor_block(d, "offline", channel, tuple((w["past"], w["future"]) for w in _W_FUT))
        for d in OFFLINE_DESCS
    ]


def _base_offline_keys(base_recipe: str) -> list[str]:
    channel = BASE_RECIPE_TO_CHANNEL[base_recipe]
    out: list[str] = []
    for d in OFFLINE_DESCS:
        out.extend(_keys_of(d, "offline", channel,
                            tuple((w["past"], w["future"]) for w in _W_FUT)))
    return out


def _short_win(p: int, f: int) -> str:
    return f"{p}-{f}"


def _short_wins_tag(wins: tuple[tuple[int, int], ...]) -> str:
    """Short tag for a window combo, used in axis_tag fallback."""
    return "+".join(_short_win(p, f) for (p, f) in wins)


# ----------------------------------------------------------------------
# G1 — online single window (48 cells)
# ----------------------------------------------------------------------


def generate_g1_cells(cfg_id: str = CFG_ID_DEFAULT) -> list[Cell]:
    """6 windows × 2 channel × 2 desc × 2 base recipe = 48 cells."""
    cells: list[Cell] = []
    for base in G1_BASE_RECIPES:
        base_factors = _base_offline_factors(base)
        base_keys = _base_offline_keys(base)
        for channel in G1_CHANNELS:
            for desc in G1_DESCS:
                for win in G1_WINDOWS:
                    online_factor = _factor_block(desc, "online", channel, (win,))
                    online_keys = _keys_of(desc, "online", channel, (win,))
                    factors = tuple(base_factors + [online_factor])
                    declared = tuple(base_keys + online_keys)
                    weight_each = 1.0 / len(declared)
                    weights = {k: weight_each for k in declared}
                    axis_tag = f"{base}_{channel}_{desc}__win-{_short_win(*win)}"
                    yaml_id = f"{cfg_id}_phase5_g1_{axis_tag}"
                    cells.append(Cell(
                        yaml_id=yaml_id,
                        group="g1",
                        base_recipe=base,
                        axis_tag=axis_tag,
                        factors=factors,
                        weights=weights,
                        fh_ratio=LOCKED_FH,
                        ws_ratio=LOCKED_WS,
                        declared_keys=declared,
                        warmup_yaml_id=f"{yaml_id}__warmup",
                    ))
    return cells


# ----------------------------------------------------------------------
# G2 — online multi-window combo (48 cells, p1 only)
# ----------------------------------------------------------------------


def generate_g2_cells(cfg_id: str = CFG_ID_DEFAULT) -> list[Cell]:
    """12 combos × 2 channel × 2 desc × 1 base recipe (p1) = 48 cells."""
    cells: list[Cell] = []
    for base in G2_BASE_RECIPES:
        base_factors = _base_offline_factors(base)
        base_keys = _base_offline_keys(base)
        for channel in G2_CHANNELS:
            for desc in G2_DESCS:
                for (wins, combo_tag) in G2_MULTI_COMBOS:
                    online_factor = _factor_block(desc, "online", channel, wins)
                    online_keys = _keys_of(desc, "online", channel, wins)
                    factors = tuple(base_factors + [online_factor])
                    declared = tuple(base_keys + online_keys)
                    weight_each = 1.0 / len(declared)
                    weights = {k: weight_each for k in declared}
                    axis_tag = f"{base}_{channel}_{desc}__multi-{combo_tag}"
                    yaml_id = f"{cfg_id}_phase5_g2_{axis_tag}"
                    cells.append(Cell(
                        yaml_id=yaml_id,
                        group="g2",
                        base_recipe=base,
                        axis_tag=axis_tag,
                        factors=factors,
                        weights=weights,
                        fh_ratio=LOCKED_FH,
                        ws_ratio=LOCKED_WS,
                        declared_keys=declared,
                        warmup_yaml_id=f"{yaml_id}__warmup",
                    ))
    return cells


# ----------------------------------------------------------------------
# G3 — online weight pattern sweep (48 cells)
# ----------------------------------------------------------------------


def _g3_weights(
    base_keys: list[str],
    online_jerk_keys: list[str],
    online_disp_keys: list[str],
    pattern: tuple[int, int],
) -> dict[str, float]:
    """offline 50% / online 50% with online split per pattern."""
    weights: dict[str, float] = {}
    # offline 8 keys @ 0.5/8 each
    for k in base_keys:
        weights[k] = 0.5 / len(base_keys)
    pj, pd = pattern
    psum = pj + pd
    if psum == 0:
        for k in online_jerk_keys + online_disp_keys:
            weights[k] = 0.0
    else:
        for k in online_jerk_keys:
            weights[k] = 0.5 * pj / psum
        for k in online_disp_keys:
            weights[k] = 0.5 * pd / psum
    return weights


def generate_g3_cells(cfg_id: str = CFG_ID_DEFAULT) -> list[Cell]:
    """12 patterns × 2 channel × 2 base = 48 cells."""
    cells: list[Cell] = []
    for base in G3_BASE_RECIPES:
        base_factors = _base_offline_factors(base)
        base_keys = _base_offline_keys(base)
        for channel in G3_CHANNELS:
            jerk_factor = _factor_block("jerk", "online", channel, ((3, 3),))
            disp_factor = _factor_block("dispersion", "online", channel, ((3, 3),))
            jerk_keys = _keys_of("jerk", "online", channel, ((3, 3),))
            disp_keys = _keys_of("dispersion", "online", channel, ((3, 3),))
            factors = tuple(base_factors + [jerk_factor, disp_factor])
            declared = tuple(base_keys + jerk_keys + disp_keys)
            for (pj, pd, ptag) in G3_WEIGHT_PATTERNS:
                weights = _g3_weights(base_keys, jerk_keys, disp_keys, (pj, pd))
                axis_tag = f"{base}_{channel}__pat-{pj}-{pd}"
                yaml_id = f"{cfg_id}_phase5_g3_{axis_tag}"
                # Warmup raw is shared across all 12 patterns within (base, channel)
                warmup_yaml_id = f"{cfg_id}_phase5_g3_{base}_{channel}__warmup"
                cells.append(Cell(
                    yaml_id=yaml_id,
                    group="g3",
                    base_recipe=base,
                    axis_tag=axis_tag,
                    factors=factors,
                    weights=weights,
                    fh_ratio=LOCKED_FH,
                    ws_ratio=LOCKED_WS,
                    declared_keys=declared,
                    warmup_yaml_id=warmup_yaml_id,
                ))
    return cells


# ----------------------------------------------------------------------
# G4 — multi-factor subset (48 cells)
# ----------------------------------------------------------------------


def generate_g4_cells(cfg_id: str = CFG_ID_DEFAULT) -> list[Cell]:
    """12 subsets × 2 channel × 2 base = 48 cells."""
    cells: list[Cell] = []
    for base in G4_BASE_RECIPES:
        base_channel = BASE_RECIPE_TO_CHANNEL[base]
        for channel in G4_CHANNELS:
            for (off_descs, on_descs, off_wins_override, on_wins_override, tag) in G4_FACTOR_SUBSETS:
                off_wins = (
                    off_wins_override
                    if off_wins_override is not None
                    else tuple((w["past"], w["future"]) for w in _W_FUT)
                )
                on_wins = (
                    on_wins_override
                    if on_wins_override is not None
                    else tuple((w["past"], w["future"]) for w in _W_K3)
                )
                # Offline factors stay on the base recipe's channel.
                offline_factors = [
                    _factor_block(d, "offline", base_channel, off_wins)
                    for d in off_descs
                ]
                offline_keys: list[str] = []
                for d in off_descs:
                    offline_keys.extend(_keys_of(d, "offline", base_channel, off_wins))
                # Online factors on the swept channel.
                online_factors = [
                    _factor_block(d, "online", channel, on_wins)
                    for d in on_descs
                ]
                online_keys: list[str] = []
                for d in on_descs:
                    online_keys.extend(_keys_of(d, "online", channel, on_wins))
                factors = tuple(offline_factors + online_factors)
                declared = tuple(offline_keys + online_keys)
                if not declared:
                    # Defensive: should not happen with current G4_FACTOR_SUBSETS.
                    raise ValueError(f"G4 subset {tag!r} produced empty declared_keys")
                weight_each = 1.0 / len(declared)
                weights = {k: weight_each for k in declared}
                axis_tag = f"{base}_{channel}__sub-{tag}"
                yaml_id = f"{cfg_id}_phase5_g4_{axis_tag}"
                cells.append(Cell(
                    yaml_id=yaml_id,
                    group="g4",
                    base_recipe=base,
                    axis_tag=axis_tag,
                    factors=factors,
                    weights=weights,
                    fh_ratio=LOCKED_FH,
                    ws_ratio=LOCKED_WS,
                    declared_keys=declared,
                    warmup_yaml_id=f"{yaml_id}__warmup",
                ))
    return cells


# ----------------------------------------------------------------------
# G5 — threshold (FH, WS) grid × 3 recipe (48 cells)
# ----------------------------------------------------------------------


def _g5_recipe_meta(recipe_id: str) -> dict[str, Any]:
    """Pull factor list / declared_keys from existing phase3/phase4 RECIPES."""
    if recipe_id in RECIPES_PHASE4:
        r = RECIPES_PHASE4[recipe_id]
        return {
            "factors": list(r["factors"]),
            "declared_keys": tuple(r["declared_keys"]),
            "warmup_yaml_id": f"{CFG_ID_DEFAULT}_phase4_{recipe_id}__warmup",
            "short": "p1" if recipe_id.startswith("p1") else "p2",
        }
    if recipe_id in PHASE3_RECIPES:
        r = PHASE3_RECIPES[recipe_id]
        return {
            "factors": list(r["factors"]),
            "declared_keys": tuple(r["declared_keys"]),
            "warmup_yaml_id": f"{CFG_ID_DEFAULT}_phase3_{recipe_id}__warmup",
            "short": "g6",
        }
    raise KeyError(f"G5 recipe {recipe_id!r} not found in phase3 or phase4 RECIPES")


def _g5_fixed_weights(recipe_id: str, declared: tuple[str, ...]) -> dict[str, float]:
    """G5 weights MUST mirror each recipe's fixed weight config (plan §1.6).

    - phase4 p1 / p2: phase4 R2 alpha_star=1.0 uniform offline pattern,
      i.e. ``generate_r2_weights(rid, 1.0, (1, 1, 1, 1))`` — online keys
      get weight 0 (because online_total = 1 - alpha = 0).
    - phase3 g6: equal 1.0 per declared key (phase3 baseline; under the
      weighted-sum-zero-nan composer this collapses to mean(contribs)).

    Anything else (uniform 1/N over both halves) would change the
    experimental target — see G2 R1 B1.
    """
    if recipe_id in RECIPES_PHASE4:
        from exp.verdict_factor_judge.phase4.spec import generate_r2_weights
        return generate_r2_weights(recipe_id, alpha_star=1.0, offline_pattern=(1, 1, 1, 1))
    if recipe_id in PHASE3_RECIPES:
        return {k: 1.0 for k in declared}
    raise KeyError(f"G5 recipe {recipe_id!r}: no fixed weight config defined")


def generate_g5_cells(cfg_id: str = CFG_ID_DEFAULT) -> list[Cell]:
    """16 (FH, WS) cells × 3 recipe = 48 cells, no base offline injection.

    Weights are fixed per recipe (plan §1.6 / G2 R1 B1) — the only
    sweep dimension is (FH, WS) ∈ G5_THRESHOLD_GRID.
    """
    cells: list[Cell] = []
    for recipe_id in G5_RECIPES:
        meta = _g5_recipe_meta(recipe_id)
        factors = tuple(meta["factors"])
        declared = meta["declared_keys"]
        weights = _g5_fixed_weights(recipe_id, declared)
        short = meta["short"]
        for (fh, ws) in G5_THRESHOLD_GRID:
            axis_tag = f"{short}__fh{fh}_ws{ws}"
            yaml_id = f"{cfg_id}_phase5_g5_{axis_tag}"
            cells.append(Cell(
                yaml_id=yaml_id,
                group="g5",
                base_recipe=short,
                axis_tag=axis_tag,
                factors=factors,
                weights=weights,
                fh_ratio=fh,
                ws_ratio=ws,
                declared_keys=declared,
                warmup_yaml_id=meta["warmup_yaml_id"],
            ))
    return cells


# ----------------------------------------------------------------------
# Aggregator + server allocator
# ----------------------------------------------------------------------


def generate_all_cells(cfg_id: str = CFG_ID_DEFAULT) -> list[Cell]:
    """All 240 phase5 cells, sorted by yaml_id (deterministic)."""
    out = (
        generate_g1_cells(cfg_id)
        + generate_g2_cells(cfg_id)
        + generate_g3_cells(cfg_id)
        + generate_g4_cells(cfg_id)
        + generate_g5_cells(cfg_id)
    )
    out.sort(key=lambda c: c.yaml_id)
    return out


def allocate_to_servers(
    cells: list[Cell],
    ratio: tuple[int, ...] = (4, 4, 4, 3, 3, 3),
    server_ids: tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5", "S6"),
) -> dict[str, list[Cell]]:
    """Round-robin allocate cells to N servers per the given ratio.

    For 240 cells with ratio (4,4,4,3,3,3): 240 / 21 = 11 full rotations + 9
    remainder cells (mod-distributed) → S1=48, S2=48, S3=45, S4=33, S5=33, S6=33.
    """
    if len(ratio) != len(server_ids):
        raise ValueError(f"ratio length {len(ratio)} != server_ids length {len(server_ids)}")
    slots: list[str] = []
    for srv, weight in zip(server_ids, ratio):
        slots.extend([srv] * weight)
    out: dict[str, list[Cell]] = {srv: [] for srv in server_ids}
    for i, cell in enumerate(cells):
        out[slots[i % len(slots)]].append(cell)
    return out


# ----------------------------------------------------------------------
# Yaml builders (mirror phase4 — call v2_spec directly, no RECIPES mut.)
# ----------------------------------------------------------------------


def build_warmup_yaml_for_cell(cfg_id: str, cell: Cell) -> dict:
    """Per-cell (or per-shared-group, see G3) warmup yaml."""
    return v2_build_warmup_yaml(
        cfg_id=cfg_id,
        eval_yaml_id=cell.warmup_yaml_id.removesuffix("__warmup"),
        eval_factors=list(cell.factors),
    )


def _build_composer(cell: Cell, fh_thr: float, ws_thr: float) -> dict[str, Any]:
    composer: dict[str, Any] = {
        "type": "weighted_sum_zero_nan",
        "weights": dict(cell.weights),
        "tier_thresholds": {"full_hit": float(fh_thr), "warm_start": float(ws_thr)},
        "warm_start_t": float(WARM_START_T),
    }
    directions = _directions_for(list(cell.declared_keys))
    if directions:
        composer["directions"] = directions
    return composer


def build_eval_yaml_for_cell(cfg_id: str, cell: Cell, fh_thr: float, ws_thr: float) -> dict:
    """Phase 5 eval yaml with thresholds patched in by the solver."""
    composer = _build_composer(cell, fh_thr, ws_thr)
    return v2_build_eval_yaml(
        cfg_id=cfg_id,
        factors=list(cell.factors),
        composer=composer,
        export_factor_outputs=True,
        calibration_window_size=50,
    )


# ----------------------------------------------------------------------
# Solver bridge
# ----------------------------------------------------------------------


def cell_to_solver_recipe(cell: Cell) -> Recipe:
    """Convert a phase5 Cell to a threshold_solver.Recipe.

    Note: solver Recipe carries declared_keys + orientations + directions;
    the actual weights are passed to ``reconstruct_scores`` via the
    ``composer_weights=...`` kwarg (do NOT call ``solve_recipe``).
    """
    declared = list(cell.declared_keys)
    return Recipe(
        recipe_id=cell.yaml_id,
        declared_keys=declared,
        orientations=_orientations_for(declared),
        directions=_directions_for(declared),
    )


# ----------------------------------------------------------------------
# Warmup raw path resolver (G1-G4 own dir / G5 dual-source)
# ----------------------------------------------------------------------


def warmup_raw_path_for_cell(
    cell: Cell,
    *,
    phase5_warmup_dir: Path,
    phase3_warmup_dir: Path,
    phase4_warmup_raw_dir: Path,
) -> Path:
    """Resolve which factor_raw jsonl backs this cell.

    G1/G2/G3/G4: ``phase5_warmup_dir/<warmup_yaml_id>.jsonl`` (own warmup).
    G5 with phase4 recipe: ``phase4_warmup_raw_dir/<recipe_id>.jsonl``.
    G5 with phase3 g6: ``phase3_warmup_dir/<warmup_yaml_id>.jsonl``.
    """
    if cell.group != "g5":
        return phase5_warmup_dir / f"{cell.warmup_yaml_id}.jsonl"
    # G5: each of the three recipes has its own raw source.
    if cell.base_recipe in ("p1", "p2"):
        recipe_id = cell.warmup_yaml_id.replace(f"{CFG_ID_DEFAULT}_phase4_", "").removesuffix("__warmup")
        return phase4_warmup_raw_dir / f"{recipe_id}.jsonl"
    if cell.base_recipe == "g6":
        return phase3_warmup_dir / f"{cell.warmup_yaml_id}.jsonl"
    raise KeyError(f"Unknown G5 base_recipe: {cell.base_recipe!r}")


# ----------------------------------------------------------------------
# CLI: emit warmup yamls + manifest
# ----------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(prog="phase5_spec")
    p.add_argument("--cfg-id", default=CFG_ID_DEFAULT)
    p.add_argument(
        "--out-dir", type=Path,
        default=Path("exp/verdict_factor_judge/config"),
    )
    args = p.parse_args()

    cells = generate_all_cells(cfg_id=args.cfg_id)
    cfg_short = args.cfg_id.split("_")[0]
    base_dir = args.out_dir / cfg_short / "phase5"
    warmup_dir = base_dir / "warmup"
    warmup_dir.mkdir(parents=True, exist_ok=True)

    # Emit warmup yamls (de-duplicated by warmup_yaml_id; G3 cells share
    # one warmup per (base, channel); G5 cells reuse historical warmups,
    # so we skip emitting a phase5-local warmup yaml for them).
    seen_warmup: set[str] = set()
    manifest_warmup: list[dict] = []
    for cell in cells:
        if cell.group == "g5":
            continue   # G5 reuses phase3/phase4 warmup, no phase5 emit
        if cell.warmup_yaml_id in seen_warmup:
            continue
        seen_warmup.add(cell.warmup_yaml_id)
        warmup_yaml = build_warmup_yaml_for_cell(args.cfg_id, cell)
        path = warmup_dir / f"{cell.warmup_yaml_id}.yaml"
        write_yaml(path, warmup_yaml)
        manifest_warmup.append({
            "warmup_yaml_id": cell.warmup_yaml_id,
            "warmup_yaml_path": str(path.relative_to(args.out_dir.parent)),
        })

    manifest = {
        "cfg_id": args.cfg_id,
        "n_cells": len(cells),
        "n_warmup_yamls": len(manifest_warmup),
        "warmup_yamls": manifest_warmup,
        "groups": {g: sum(1 for c in cells if c.group == g) for g in ("g1", "g2", "g3", "g4", "g5")},
    }
    (base_dir / "phase5_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[phase5_spec] {len(cells)} cells, {len(manifest_warmup)} warmup yamls "
          f"emitted to {base_dir.relative_to(args.out_dir.parent)}")


__all__ = [
    "BASE_RECIPE_TO_CHANNEL",
    "CFG_ID_DEFAULT",
    "Cell",
    "G1_BASE_RECIPES",
    "G1_CHANNELS",
    "G1_DESCS",
    "G1_WINDOWS",
    "G2_BASE_RECIPES",
    "G2_CHANNELS",
    "G2_DESCS",
    "G2_MULTI_COMBOS",
    "G3_BASE_RECIPES",
    "G3_CHANNELS",
    "G3_WEIGHT_PATTERNS",
    "G4_BASE_RECIPES",
    "G4_CHANNELS",
    "G4_FACTOR_SUBSETS",
    "G5_RECIPES",
    "G5_THRESHOLD_GRID",
    "LOCKED_FH",
    "LOCKED_WS",
    "OFFLINE_DESCS",
    "ONLINE_DESCS",
    "WARM_START_T",
    "allocate_to_servers",
    "build_eval_yaml_for_cell",
    "build_warmup_yaml_for_cell",
    "cell_to_solver_recipe",
    "generate_all_cells",
    "generate_g1_cells",
    "generate_g2_cells",
    "generate_g3_cells",
    "generate_g4_cells",
    "generate_g5_cells",
    "warmup_raw_path_for_cell",
]


if __name__ == "__main__":
    main()
