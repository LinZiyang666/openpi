"""Kinematic phase5 spec — 237-cell generator on weighted_sum d1-best.

Composes phase5's native G1/G2/G3/G4 generators (which don't depend on
G5_THRESHOLD_GRID) with a LOCAL G5 generator that uses a triangular
``fh + ws <= 0.9`` filtered grid (15 pairs × 3 recipes = 45 G5 cells,
vs phase5's 16 × 3 = 48). Total = 48*4 + 45 = 237 cells.

This module never mutates ``exp.verdict_factor_judge.phase5.spec`` module
state — G1 Round 1 Reviewer B3 explicitly forbids monkey-patching the
phase5 G5_THRESHOLD_GRID global because it would make phase5's regression
test order-dependent. We instead call phase5's pure helper functions
(``_g5_recipe_meta``, ``_g5_fixed_weights``) and iterate our local grid.

All cells' ``warmup_yaml_id`` is rewritten to a single shared id
("ws_d1_kin_super_warmup") so a single super warmup raw jsonl can serve
all 237 cells via offline calibration source.

Eval yaml uses ``calibration.samples_source.type=offline`` with the
super warmup raw path on the server filesystem; server reads the file
at ``build_per_connection_components`` time and constructs
``CalibrationSamples`` — no WarmupPool / preload_normalizer_buffer
involved. This sidesteps the LRU 100-entry cap which would evict ~57%
of cells under the 237-cell load (G1 R1 R5).

Coupling map:
  DEPENDS ON: exp.verdict_factor_judge.phase5.spec (G1-G4 generators,
                                                    G5 recipe metadata pure helpers,
                                                    Cell dataclass)
              exp.verdict_factor_judge.common.v2_spec (CFG_SPECS, normalization_offline)
              exp.verdict_factor_judge.phase3.spec (_directions_for via _build_composer)
  CONSUMED BY: exp.weighted_sum.kinematic.super_warmup,
               exp.weighted_sum.kinematic.runner.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from exp.verdict_factor_judge.phase5 import spec as p5
from exp.verdict_factor_judge.phase5.spec import Cell


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

# v2_spec.CFG_SPECS key — must be added to that module first (Task 1).
CFG_ID_DEFAULT = "spatial16_ws_d1_best"

# yaml id prefix replaces phase5's "spatial16_w8_d4_phase5" to clearly
# separate kinematic-on-weighted_sum results from phase5 native runs.
YAML_PREFIX = "ws_d1_kin"

# All 237 cells share this single warmup id (super warmup pattern).
SUPER_WARMUP_ID = "ws_d1_kin_super_warmup"

# Server-side relative path that eval yamls reference for offline
# calibration source. The driver pushes super_warmup_raw.jsonl here via
# tether to both ziyang10 and xuanle (mirroring the repo layout).
SUPER_WARMUP_RAW_RELPATH = (
    "exp/weighted_sum/data/libero_spatial/kinematic_phase5/d1/super_warmup_raw.jsonl"
)

# Phase5 G5 originally uses three recipes (p1 / p2 / g6). We inherit
# this tuple so any future addition propagates automatically.
G5_RECIPES = p5.G5_RECIPES


# ----------------------------------------------------------------------
# G5 grid filter — fh + ws <= 0.9 triangular (mirror threshold_pareto)
# ----------------------------------------------------------------------


def _g5_grid_filtered() -> tuple[tuple[float, float], ...]:
    """G5 (FH, WS) grid with fh+ws<=0.9 inclusive (kills (0.5, 0.5) only).

    Verified arithmetic: enumerating fh, ws in (0.2, 0.3, 0.4, 0.5)^2
    yields 16 pairs; filtering ``fh + ws <= 0.9 + 1e-9`` removes exactly
    (0.5, 0.5), keeping boundary pairs (0.4, 0.5) and (0.5, 0.4). Result
    = 15 pairs. This matches threshold_pareto §2.2 ``max_total=0.9`` and
    avoids the degenerate fh+ws=1.0 case where ``i_ws == n-1`` collapses
    T_ws to ``min(scores)`` (G1 R1 B1 mandated correction).

    Returns deterministic insertion order so downstream snapshots are
    reproducible.
    """
    return tuple(
        (fh, ws)
        for fh in (0.2, 0.3, 0.4, 0.5)
        for ws in (0.2, 0.3, 0.4, 0.5)
        if fh + ws <= 0.9 + 1e-9
    )


# ----------------------------------------------------------------------
# G5 cell generator (LOCAL — does not mutate p5.G5_THRESHOLD_GRID)
# ----------------------------------------------------------------------


def _generate_g5_cells_local(cfg_id: str = CFG_ID_DEFAULT) -> list[Cell]:
    """G5 generator using the filtered local grid; zero phase5 mutation.

    Mirrors ``phase5.spec.generate_g5_cells`` body but iterates
    ``_g5_grid_filtered()`` instead of ``p5.G5_THRESHOLD_GRID``. Reuses
    ``p5._g5_recipe_meta`` and ``p5._g5_fixed_weights`` (pure helper
    functions — verified by reading phase5/spec.py:465-503: they only
    read RECIPES_PHASE4 / PHASE3_RECIPES dicts and return new dicts /
    tuples without touching module-level state).

    Each cell here uses ``cfg_id`` for yaml_id prefix; the rename pass
    in ``generate_all_cells`` then replaces ``"{cfg_id}_phase5"`` with
    ``YAML_PREFIX`` so the final ids look like ``ws_d1_kin_g5_...``.

    The warmup_yaml_id field is set to phase5's historical (phase3 g6
    or phase4 p1/p2) id here; the rename pass overwrites it to
    ``SUPER_WARMUP_ID`` so all 237 cells share one super warmup.
    """
    cells: list[Cell] = []
    for recipe_id in G5_RECIPES:
        meta = p5._g5_recipe_meta(recipe_id)
        factors = tuple(meta["factors"])
        declared = meta["declared_keys"]
        weights = p5._g5_fixed_weights(recipe_id, declared)
        short = meta["short"]
        for (fh, ws) in _g5_grid_filtered():
            axis_tag = f"{short}__fh{fh}_ws{ws}"
            yaml_id = f"{cfg_id}_phase5_g5_{axis_tag}"
            cells.append(
                Cell(
                    yaml_id=yaml_id,
                    group="g5",
                    base_recipe=short,
                    axis_tag=axis_tag,
                    factors=factors,
                    weights=weights,
                    fh_ratio=fh,
                    ws_ratio=ws,
                    declared_keys=declared,
                    # placeholder; rewritten in generate_all_cells rename pass
                    warmup_yaml_id=meta["warmup_yaml_id"],
                )
            )
    return cells


# ----------------------------------------------------------------------
# Composite generator
# ----------------------------------------------------------------------


def generate_all_cells(cfg_id: str = CFG_ID_DEFAULT) -> list[Cell]:
    """Compose phase5 native G1-G4 + local G5, then rename ids.

    Critically: phase5's G1/G2/G3/G4 generators do NOT read
    ``G5_THRESHOLD_GRID``, so we can call them directly with our
    ``cfg_id`` and reuse all phase5 logic for those 192 cells. Only G5
    is generated locally with the filtered grid.

    After composition, every cell's yaml_id and warmup_yaml_id are
    rewritten:
      - ``yaml_id``: ``"{cfg_id}_phase5_..."`` → ``"{YAML_PREFIX}_..."``
        so artifacts don't collide with phase5 native files in shared
        config / data trees.
      - ``warmup_yaml_id``: any value → ``SUPER_WARMUP_ID`` so all 237
        cells share one super warmup raw jsonl as calibration source.

    Determinism: phase5 generators are deterministic; our G5 grid
    iterator is deterministic; the rename pass preserves order. Result
    is sorted at the end (same convention as phase5).
    """
    g1_g4 = (
        p5.generate_g1_cells(cfg_id=cfg_id)
        + p5.generate_g2_cells(cfg_id=cfg_id)
        + p5.generate_g3_cells(cfg_id=cfg_id)
        + p5.generate_g4_cells(cfg_id=cfg_id)
    )
    g5 = _generate_g5_cells_local(cfg_id=cfg_id)
    cells = g1_g4 + g5
    cells.sort(key=lambda c: c.yaml_id)

    renamed: list[Cell] = []
    for c in cells:
        new_yaml_id = c.yaml_id.replace(f"{cfg_id}_phase5", YAML_PREFIX, 1)
        renamed.append(
            dataclasses.replace(
                c,
                yaml_id=new_yaml_id,
                warmup_yaml_id=SUPER_WARMUP_ID,
            )
        )
    return renamed


def super_warmup_declared_keys() -> set[str]:
    """Single source of truth for what the super warmup MUST dump.

    Returns the union of every cell's ``declared_keys`` across all 237
    cells. The super warmup yaml's ``judge.dump.factors`` list is
    derived from this set (grouped back into factor_type → windows in
    ``super_warmup.build_super_warmup_yaml``), guaranteeing coverage by
    construction.

    Stage 3 verify check #6 confirms by running ``bind_keys`` for each
    cell against the super warmup raw — any miss raises immediately and
    blocks Stage 4.
    """
    return set().union(*[set(c.declared_keys) for c in generate_all_cells()])


# ----------------------------------------------------------------------
# Eval yaml builder — offline calibration mode
# ----------------------------------------------------------------------


def build_eval_yaml_for_cell(
    cell: Cell,
    fh_thr: float,
    ws_thr: float,
    *,
    super_raw_relpath: str = SUPER_WARMUP_RAW_RELPATH,
    cfg_id: str = CFG_ID_DEFAULT,
    preload_pkl_override: str | None = None,
) -> dict[str, Any]:
    """Per-cell eval yaml using offline calibration source.

    Key difference from phase5's ``build_eval_yaml_for_cell``: the
    calibration block uses ``samples_source.type='offline'`` pointing
    at the super warmup raw jsonl (server filesystem path), rather than
    phase5's ``samples_source.type='warmup'`` which requires a sibling
    ``preload_normalizer_buffer`` call into WarmupPool.

    Why offline mode: with 237 cells × eval_concurrency=2 distributed
    across two servers (one at cap=48), WarmupPool's LRU=100 would
    evict 100+ entries by run completion. ``PercentileRollingCalibration``
    is rebuilt per worker connection (config.py:1725 _bind_bundle),
    so an evicted entry triggers a non-retriable ConfigValidationError
    when a worker reconnects. Offline mode reads the raw jsonl directly
    at bind time (config.py:_load_calibration_samples_offline jsonl
    branch) — no WarmupPool involved, zero eviction risk.

    Cost: each worker connection re-reads ~10MB jsonl (~1s parse). At
    24000 ep × ~1 reconnect / ep this is negligible; the parsed
    CalibrationSamples is then cached per-connection for all verdicts
    of that cell.
    """
    from exp.verdict_factor_judge.common.v2_spec import (
        CFG_SPECS,
        normalization_offline,
    )
    from exp.verdict_factor_judge.phase5.spec import _build_composer

    cfg = CFG_SPECS[cfg_id]
    preload_path = preload_pkl_override if preload_pkl_override is not None else cfg["preload_pkl"]
    composer = _build_composer(cell, fh_thr, ws_thr)
    judge: dict[str, Any] = {
        "type": "composite",
        "normalization": normalization_offline(),
        "factors": list(cell.factors),
        "calibration": {
            "type": "percentile_rolling",
            "params": {"window_size": 50},
            "samples_source": {
                "type": "offline",
                "offline": {
                    "path": super_raw_relpath,
                    "format": "jsonl",
                },
            },
        },
        "composer": composer,
        "export_factor_outputs": True,
    }
    return {
        "enabled": True,
        "timer": {
            "enabled": False,
            "buffer_size": 10000,
            "output_csv_dir": None,
        },
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
                "preload_path": preload_path,
                "index_type": "brute_force",
            },
        },
        "write_policy": {"type": "never"},
    }


# ----------------------------------------------------------------------
# Convenience: solver recipe bridge (mirror phase5.cell_to_solver_recipe)
# ----------------------------------------------------------------------


def cell_to_solver_recipe(cell: Cell):
    """Convert a kinematic Cell to a threshold_solver.Recipe.

    Pure passthrough to phase5's bridge — declared_keys / orientations /
    directions semantics are identical between kinematic and phase5
    native (both use the same factor registry + Cell dataclass).
    """
    return p5.cell_to_solver_recipe(cell)


# ----------------------------------------------------------------------
# Public exports
# ----------------------------------------------------------------------


__all__ = [
    "CFG_ID_DEFAULT",
    "YAML_PREFIX",
    "SUPER_WARMUP_ID",
    "SUPER_WARMUP_RAW_RELPATH",
    "G5_RECIPES",
    "_g5_grid_filtered",
    "_generate_g5_cells_local",
    "generate_all_cells",
    "super_warmup_declared_keys",
    "build_eval_yaml_for_cell",
    "cell_to_solver_recipe",
]
