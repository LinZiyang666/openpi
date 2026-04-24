"""Offline YAML batch generator for the random_periodic_gate experiment.

Reads the three ``base_<cfg>.yaml`` templates and expands them across a
fixed 38-slug grid per cfg (20 PeriodicGate combinations + 18 RandomGate
combinations). Every generated YAML overrides exactly the
``checkpoints.cp1.gate`` subtree and preserves the rest of the template
byte-for-byte at the dict-equality level. Output is split into six batch
directories (two per cfg, 19 slugs each, split by dictionary order) under
``config/batch{1..6}/``.

The generator is idempotent (``yaml.safe_dump`` is deterministic on pure
Python scalars / dicts / lists); running it twice produces byte-identical
files. Use ``--validate`` to additionally ``load_cache_config`` each YAML
so schema-level drift fails loud.

Plan: ``logs/random_periodic_gate_plan.log.md`` §5.5.

Usage (from repo root)::

    uv run python -m exp.random_periodic_gate.generate_batches
    uv run python -m exp.random_periodic_gate.generate_batches --validate
"""

from __future__ import annotations

import argparse
import copy
import logging
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grid definition (§2.2). Keep as module-level tuples so unit tests can
# import them without re-parsing args.
# ---------------------------------------------------------------------------

PERIODIC_CACHE_LENS: tuple[int, ...] = (1, 2, 5, 10)
PERIODIC_INFERENCE_LENS: tuple[int, ...] = (1, 2, 3, 5, 10)
RANDOM_P_INFERENCES: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70)
RANDOM_SEEDS: tuple[int, ...] = (0, 1, 2)

CFG_NAMES: tuple[str, ...] = ("clip_w7_d4", "spatial16_w8_d4", "max_pool_w3_d5")


# ---------------------------------------------------------------------------
# Slug helpers. Slugs must be filesystem-safe, stable, and dictionary-sortable
# so the 19/19 batch split is deterministic.
# ---------------------------------------------------------------------------


def _format_p(p: float) -> str:
    """Format ``0.20`` as ``p0p20`` — '.' replaced with 'p' for slugs."""
    return "p" + f"{p:.2f}".replace(".", "p")


@dataclass(frozen=True)
class GridPoint:
    """One (cfg, gate_type, params) triple materialized into an output YAML."""

    cfg: str
    gate_type: str   # "random" | "periodic"
    params: dict[str, Any]

    @property
    def slug(self) -> str:
        if self.gate_type == "random":
            return f"random_{_format_p(self.params['p_inference'])}_s{self.params['seed']}"
        if self.gate_type == "periodic":
            return f"periodic_k{self.params['cache_len']}_n{self.params['inference_len']}"
        raise ValueError(f"unknown gate_type: {self.gate_type}")


def build_grid_for_cfg(cfg: str) -> list[GridPoint]:
    """Return the 38-element grid for one cfg, stably sorted by slug."""
    points: list[GridPoint] = []
    for k, n in product(PERIODIC_CACHE_LENS, PERIODIC_INFERENCE_LENS):
        points.append(
            GridPoint(
                cfg=cfg,
                gate_type="periodic",
                params={"cache_len": int(k), "inference_len": int(n)},
            )
        )
    for p, s in product(RANDOM_P_INFERENCES, RANDOM_SEEDS):
        points.append(
            GridPoint(
                cfg=cfg,
                gate_type="random",
                params={"p_inference": float(p), "seed": int(s)},
            )
        )
    points.sort(key=lambda g: g.slug)
    return points


def split_into_batches(points: Sequence[GridPoint]) -> tuple[list[GridPoint], list[GridPoint]]:
    """Split a per-cfg grid at the dict-sort midpoint (19/19 for the 38-grid)."""
    mid = len(points) // 2
    return list(points[:mid]), list(points[mid:])


# ---------------------------------------------------------------------------
# Rendering. We deep-copy the base dict and replace only the gate subtree.
# The rest of the structure is preserved at the dict-equality level, which
# is what the plan tests lock.
# ---------------------------------------------------------------------------


def _gate_block(point: GridPoint) -> dict[str, Any]:
    block: dict[str, Any] = {"type": point.gate_type}
    block.update(point.params)
    return block


def render_dict(base: dict[str, Any], point: GridPoint) -> dict[str, Any]:
    """Produce the rendered CacheConfig dict for ``point``."""
    rendered = copy.deepcopy(base)
    rendered["checkpoints"]["cp1"]["gate"] = _gate_block(point)
    return rendered


# ---------------------------------------------------------------------------
# File layout helpers.
# ---------------------------------------------------------------------------


def batch_indices_for_cfg(cfg_index: int) -> tuple[int, int]:
    """Return the pair of 1-based batch indices owned by ``cfg_index`` (0..2).

    cfg 0 -> (1, 2)
    cfg 1 -> (3, 4)
    cfg 2 -> (5, 6)
    """
    return (cfg_index * 2 + 1, cfg_index * 2 + 2)


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    # exp/random_periodic_gate/generate_batches.py -> repo root is parents[2].
    return Path(__file__).resolve().parents[2]


def _base_yaml_path(cfg: str) -> Path:
    return _project_root() / "exp" / "random_periodic_gate" / "config" / f"base_{cfg}.yaml"


def _batch_dir(batch_idx: int) -> Path:
    return (
        _project_root()
        / "exp"
        / "random_periodic_gate"
        / "config"
        / f"batch{batch_idx}"
    )


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def generate_all(validate: bool = False) -> list[Path]:
    """Emit the 114 rendered YAMLs. Return the list of written file paths."""
    written: list[Path] = []
    for cfg_idx, cfg in enumerate(CFG_NAMES):
        base_path = _base_yaml_path(cfg)
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        grid = build_grid_for_cfg(cfg)
        first, second = split_into_batches(grid)
        first_idx, second_idx = batch_indices_for_cfg(cfg_idx)
        for batch_idx, batch_points in ((first_idx, first), (second_idx, second)):
            out_dir = _batch_dir(batch_idx)
            for point in batch_points:
                rendered = render_dict(base, point)
                out_path = out_dir / f"{point.slug}.yaml"
                _write_yaml(out_path, rendered)
                written.append(out_path)
                logger.debug("wrote %s", out_path)

    if validate:
        # Late import so static generation does not require the full runtime.
        from openpi.cache.config import load_cache_config

        for path in written:
            load_cache_config(path)
        logger.info("validated %d rendered YAMLs", len(written))
    return written


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate random_periodic_gate batch YAMLs")
    p.add_argument(
        "--validate",
        action="store_true",
        help="After writing, load every rendered YAML through "
        "openpi.cache.config.load_cache_config to catch schema drift.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p.parse_args(argv if argv is None else list(argv))


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    written = generate_all(validate=args.validate)
    logger.info("generated %d YAML files across 6 batches", len(written))


if __name__ == "__main__":
    main()
