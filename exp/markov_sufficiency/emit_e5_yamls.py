"""Emit the E5 confirmatory arms on the original d3-trough base.

The libero_10 d3-trough signal was screened on the base with modality weights
``vision_0=0.62 / vision_1=0.37 / robot_state=0.00``. A confirmatory rerun must
stay on that base: changing the weights as well would vary two things at once,
and the result could neither confirm nor refute the original winner.

Arms: the top-3 screened d3 weight shapes, plus a **same-base d1 anchor**
derived from the same yaml (drop the trajectory weights, set depth 1). The
anchor is re-run in the same batch rather than reused from history, because the
original signal may partly reflect batch drift.

Public interface: :func:`derive_shape`, :func:`derive_anchor`,
:func:`emit_arms`, :func:`main`.

Key dependency: ``openpi.cache.config.load_cache_config`` for validation.
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
from typing import Any, Sequence

import yaml

from exp.markov_sufficiency.emit_e4_yamls import diff_keys

ALLOWED_KEYS = {"trajectory_depth", "trajectory_weights", "step_filter", "step_window"}

#: The base the original screening ran on; guarded so a wrong source cannot be
#: passed in silently.
EXPECTED_BASE_WEIGHTS = {"vision_0": 0.62, "vision_1": 0.37, "robot_state": 0.0}
BASE_WEIGHT_TOL = 5e-3


def check_base(source: dict[str, Any]) -> None:
    """Refuse a source whose modality weights are not the screened base."""
    keys = source.get("keys", {})
    for field, expected in EXPECTED_BASE_WEIGHTS.items():
        cfg = keys.get(field, {})
        got = float(cfg.get("weight", 0.0)) if cfg.get("enabled", False) else 0.0
        if abs(got - expected) > BASE_WEIGHT_TOL:
            raise SystemExit(
                f"E5 must run on the screened base {EXPECTED_BASE_WEIGHTS}; "
                f"{field} is {got} in {source.get('key_builder', {}).get('type', '?')}"
            )


def derive_shape(source: dict[str, Any], weights: Sequence[float]) -> dict[str, Any]:
    """Return the source with one d3 weight shape substituted."""
    out = copy.deepcopy(source)
    ss = out["checkpoints"]["cp1"]["search_strategy"]
    ss["trajectory_depth"] = len(weights)
    ss["trajectory_weights"] = [float(w) for w in weights]
    return out


def derive_anchor(source: dict[str, Any]) -> dict[str, Any]:
    """Return the same-base d1 anchor (depth 1, trajectory weights removed)."""
    out = copy.deepcopy(source)
    ss = out["checkpoints"]["cp1"]["search_strategy"]
    ss["trajectory_depth"] = 1
    ss.pop("trajectory_weights", None)
    return out


def emit_arms(
    source_path: str | pathlib.Path,
    out_dir: str | pathlib.Path,
    shapes: Sequence[Sequence[float]],
    prefix: str,
) -> list[pathlib.Path]:
    """Write the anchor plus one yaml per screened shape, validating each."""
    from openpi.cache.config import load_cache_config

    source_path = pathlib.Path(source_path)
    with source_path.open() as fh:
        source = yaml.safe_load(fh)
    check_base(source)

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[pathlib.Path] = []

    def _emit(derived: dict[str, Any], path: pathlib.Path) -> None:
        changed = diff_keys(source, derived)
        if not changed <= ALLOWED_KEYS:
            raise SystemExit(f"{path.name}: derivation touched disallowed keys {sorted(changed - ALLOWED_KEYS)}")
        with path.open("w") as fh:
            yaml.safe_dump(derived, fh, sort_keys=False)
        load_cache_config(path)  # rejects a config the server would refuse
        written.append(path)

    _emit(derive_anchor(source), out_dir / f"{prefix}__d1_anchor.yaml")
    for i, shape in enumerate(shapes):
        _emit(derive_shape(source, shape), out_dir / f"{prefix}__shape{i}.yaml")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Emit E5 confirmatory yamls")
    ap.add_argument("--source", required=True, help="the 0.62/0.37/0 d3 base yaml")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shapes", required=True, help="JSON list of weight vectors, e.g. [[0.2,0.3,0.5]]")
    ap.add_argument("--prefix", required=True)
    args = ap.parse_args()

    shapes = json.loads(args.shapes)
    for path in emit_arms(args.source, args.out_dir, shapes, args.prefix):
        print(path)


if __name__ == "__main__":
    main()
