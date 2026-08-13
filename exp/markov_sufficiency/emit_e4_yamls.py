"""Emit the E4 arm matrix by deriving yamls from an existing base config.

E4 varies exactly two knobs on a frozen base: the trajectory depth and the
nominal inference-index filter. Everything else -- modality weights, field
similarities, the z-score mu/sigma, the preload path, gate and judge -- must
stay byte-identical to the source, otherwise the contrast stops being a
single-variable comparison. The emitter enforces that by diffing its own output
against the source and rejecting any key outside the allowed set.

Arms (per suite): A0 d1/all, A1 d1/window, A2 d_best/all, A3 d_best/window,
A4 d_best/exact. A4 is exploratory -- with ~49 library episodes the exact
filter can collapse the candidate set.

Public interface: :func:`derive_arm`, :func:`emit_suite`, :func:`main`.

Key dependency: ``openpi.cache.config.load_cache_config`` for validation, so a
generated yaml that the server would reject fails here instead.
"""

from __future__ import annotations

import argparse
import copy
import pathlib
from typing import Any

import yaml

ALLOWED_KEYS = {"trajectory_depth", "trajectory_weights", "step_filter", "step_window"}

#: (arm, depth-selector, step_filter, step_window). ``None`` depth means d_best.
ARMS = (
    ("A0", 1, "all", None),
    ("A1", 1, "window", 5),
    ("A2", None, "all", None),
    ("A3", None, "window", 5),
    ("A4", None, "exact", None),
)


# ------------------------------------------------------------------
# Derivation
# ------------------------------------------------------------------


def derive_arm(
    source: dict[str, Any],
    depth: int,
    step_filter: str,
    step_window: int | None,
) -> dict[str, Any]:
    """Return a deep copy of ``source`` with only the four permitted keys changed.

    At ``depth == 1`` the ``trajectory_weights`` key is dropped: it carries no
    meaning there and would leave a stale field in the diff and the manifest.
    (The config validator only checks weight length when ``depth > 1``, so this
    is a derivation convention rather than a validation requirement.)
    """
    out = copy.deepcopy(source)
    ss = out["checkpoints"]["cp1"]["search_strategy"]
    ss["trajectory_depth"] = depth
    if depth == 1:
        ss.pop("trajectory_weights", None)
    ss["step_filter"] = step_filter
    if step_window is None:
        ss.pop("step_window", None)
    else:
        ss["step_window"] = step_window
    return out


def diff_keys(source: dict[str, Any], derived: dict[str, Any]) -> set[str]:
    """Dotted paths of every leaf that differs between two configs.

    The comparison walks the **whole document**: an earlier version only looked
    inside ``search_strategy`` plus the top level, so a change to a sibling of
    ``search_strategy`` (a judge, a gate, another checkpoint) would have passed
    the allowlist unnoticed.
    """
    changed: set[str] = set()

    def walk(a: Any, b: Any, path: str) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for key in set(a) | set(b):
                walk(a.get(key), b.get(key), f"{path}.{key}" if path else str(key))
            return
        if a != b:
            changed.add(path)

    walk(source, derived, "")
    # The allowlist is expressed as bare key names, so report the leaf name for
    # search_strategy entries and the full path for anything else.
    prefix = "checkpoints.cp1.search_strategy."
    return {p[len(prefix):] if p.startswith(prefix) else p for p in changed}


def emit_suite(
    source_path: str | pathlib.Path,
    out_dir: str | pathlib.Path,
    d_best: int,
    prefix: str,
) -> list[pathlib.Path]:
    """Write the five arm yamls for one suite and validate each one."""
    from openpi.cache.config import load_cache_config

    source_path = pathlib.Path(source_path)
    with source_path.open() as fh:
        source = yaml.safe_load(fh)

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[pathlib.Path] = []
    for arm, depth_sel, step_filter, step_window in ARMS:
        depth = d_best if depth_sel is None else depth_sel
        derived = derive_arm(source, depth, step_filter, step_window)
        changed = diff_keys(source, derived)
        if not changed <= ALLOWED_KEYS:
            raise SystemExit(f"{arm}: derivation touched disallowed keys {sorted(changed - ALLOWED_KEYS)}")
        path = out_dir / f"{prefix}__{arm}.yaml"
        with path.open("w") as fh:
            yaml.safe_dump(derived, fh, sort_keys=False)
        load_cache_config(path)  # rejects a config the server would refuse
        written.append(path)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Emit E4 arm yamls")
    ap.add_argument("--source", required=True, help="base yaml with trajectory_depth > 1")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--d-best", type=int, default=3)
    ap.add_argument("--prefix", required=True)
    args = ap.parse_args()

    for path in emit_suite(args.source, args.out_dir, args.d_best, args.prefix):
        print(path)


if __name__ == "__main__":
    main()
