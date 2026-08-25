"""Emit the warmup, sweep and gate-only YAMLs for the GR00T gate-threshold Pareto.

Structure is inherited verbatim from the winning ``ws_search`` cell for each
suite (``gate_pareto_bindings.Binding.template_path``): retrieval weights,
per-field zscore+tanh normalizers, trajectory depth 1, backend dims. Only two
things are ever touched, so a difference between this experiment and the search
line can only come from one of them:

*   ``checkpoints.cp1.gate`` -- ``always_search`` for warmup, the N4 hybrid
    (``score_hysteresis`` + ``L=6``) for the sweep and the gate-only arm.
*   ``checkpoints.cp1.judge`` -- force-MISS ``threshold: 2.0`` for warmup, the
    solved ``T_fh`` for a sweep cell, force-HIT ``-1.0`` for the gate-only arm.

Both sentinels sit outside the ``[0, 1]`` score range, which
``ZScoreNormalizer`` guarantees is closed (its tanh squash is mandatory and
bounded, and the field weights sum to one).

Every emitted file is round-tripped through ``load_cache_config`` before it is
accepted, and each mode additionally asserts what schema validity alone would
not catch: a warm tier silently inherited from the template, or a gated arm
that lost its ``L`` and would therefore run as pure N1 -- indistinguishable in
the results from an arm that kept it.

The 16-cell ``f_FH`` grid is imported from the pi0.5 line rather than restated,
for the same reason the threshold solver is: a grid that drifted by one cell
would make the two lines quietly incomparable.

Public interface: ``sync_template``, ``build_warmup``, ``build_eval``,
``build_gate_only``, ``emit_warmup``, ``emit_eval``, ``emit_gate_only``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import shutil

import yaml

from openpi.cache.config import load_cache_config

from exp.gate_threshold_pareto.solve_gtp import FH_GRID
from exp.libero_groot import gate_pareto_bindings as gpb

#: Above the [0, 1] score range -> the threshold judge never hits (warmup).
FORCE_MISS = 2.0
#: Below the [0, 1] score range -> the threshold judge always accepts (gate-only).
FORCE_HIT = -1.0

#: N4 winning point, transplanted from the pi0.5 gate line. ``theta`` is filled
#: per suite from that suite's own warmup; ``L`` caps a continuous cache run and
#: is what makes this N4 rather than pure N1.
GATE_J = 3
GATE_PROBE_INTERVAL = 3
GATE_L = 6


# ------------------------------------------------------------------
# Template
# ------------------------------------------------------------------


def sync_template(binding: gpb.Binding) -> dict:
    """Copy the winning search cell into the repo and record its digest.

    Returns the provenance record. Copying rather than reading through is the
    point: ``source_template`` lives on a scratch mount shared with the
    collection line, so an arm generated straight from there stops being
    reproducible the moment that file moves.
    """
    src = pathlib.Path(binding.source_template)
    if not src.is_file():
        raise SystemExit(f"{binding.suite}: source template not found: {src}")
    binding.template_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, binding.template_path)
    digest = hashlib.sha256(binding.template_path.read_bytes()).hexdigest()
    record = {
        "suite": binding.suite,
        "source_template": str(src),
        "template_path": str(binding.template_path),
        "sha256": digest,
    }
    (binding.config_root / "template_provenance.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


def _base(binding: gpb.Binding) -> dict:
    if not binding.template_path.is_file():
        raise SystemExit(
            f"{binding.suite}: {binding.template_path} missing -- run "
            "`emit_gate_yamls.py --mode template` first"
        )
    cfg = copy.deepcopy(
        yaml.safe_load(binding.template_path.read_text(encoding="utf-8"))
    )
    # Restated rather than inherited: the template's preload_path is whatever
    # the search cell used, and this experiment's library binding is the thing
    # a reader should be able to check in one place.
    cfg["backend"]["in_memory"]["preload_path"] = binding.library
    cfg["write_policy"] = {"type": "never"}
    return cfg


def _hybrid_gate(theta: float) -> dict:
    return {
        "type": "score_hysteresis",
        "theta_low": theta,
        "theta_high": theta,
        "j": GATE_J,
        "probe_interval": GATE_PROBE_INTERVAL,
        "L": GATE_L,
    }


# ------------------------------------------------------------------
# Builders
# ------------------------------------------------------------------


def build_warmup(binding: gpb.Binding) -> dict:
    """Force-MISS warmup: search still runs, so every step records its score."""
    cfg = _base(binding)
    cp1 = cfg["checkpoints"]["cp1"]
    cp1["gate"] = {"type": "always_search"}
    cp1["judge"] = {"type": "threshold", "threshold": FORCE_MISS}
    return cfg


def build_eval(binding: gpb.Binding, *, theta: float, t_fh: float) -> dict:
    """One sweep cell: hybrid gate at the suite's theta, binary verdict at t_fh."""
    cfg = _base(binding)
    cp1 = cfg["checkpoints"]["cp1"]
    cp1["gate"] = _hybrid_gate(theta)
    cp1["judge"] = {"type": "threshold", "threshold": t_fh}
    return cfg


def build_gate_only(binding: gpb.Binding, *, theta: float) -> dict:
    """Gate-only ablation: the verdict is disabled, the gate works alone.

    With every searched step accepted, teacher calls can only come from the
    gate itself -- the V2 injection that caps a continuous cache run at ``L``,
    and the N1 hysteresis skip after ``j`` sub-theta scores. ``L`` stays at the
    sweep's value so this point is the left-hand limit of *this* line's
    frontier rather than a different gate's.
    """
    cfg = _base(binding)
    cp1 = cfg["checkpoints"]["cp1"]
    cp1["gate"] = _hybrid_gate(theta)
    cp1["judge"] = {"type": "threshold", "threshold": FORCE_HIT}
    return cfg


# ------------------------------------------------------------------
# Writing
# ------------------------------------------------------------------


def _write(cfg: dict, out_dir: pathlib.Path, yaml_id: str, *, expect_gate_l: bool) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{yaml_id}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    loaded = load_cache_config(path)  # strict schema self-check
    cp1 = loaded.checkpoints["cp1"]
    # Post-condition, not an expected input: every builder above replaces the
    # judge wholesale, so a template tier cannot survive today. The check stays
    # because a future builder that merged into the judge instead would
    # reintroduce the tier silently -- a WARM_START against a GR00T library is
    # downgraded to MISS rather than raising, so the symptom would be an
    # inexplicably low hit rate with nothing to grep for.
    if cp1.judge.warm_tiers:
        raise SystemExit(
            f"{path}: warm tier present ({cp1.judge.warm_tiers}); this experiment "
            "disables the warm-start route entirely, so any tier here is a "
            "template leak"
        )
    if expect_gate_l and cp1.gate.L != GATE_L:
        raise SystemExit(
            f"{path}: gate.L={cp1.gate.L!r}, expected {GATE_L}. An arm that lost "
            "its L runs as pure N1 and is indistinguishable in the results from "
            "one that kept it."
        )
    return str(path)


def emit_warmup(binding: gpb.Binding) -> dict[str, str]:
    """One warmup yaml for the suite. Returns ``{yaml_id: path}``."""
    yaml_id = f"gpw_{binding.tag}"
    return {
        yaml_id: _write(
            build_warmup(binding),
            binding.config_root / "warmup",
            yaml_id,
            expect_gate_l=False,
        )
    }


def emit_eval(binding: gpb.Binding, solved: dict) -> dict[str, str]:
    """16 sweep yamls for the suite, from a solved-threshold record."""
    arm = f"gpw_{binding.tag}"
    record = solved["arms"].get(arm)
    if record is None:
        raise SystemExit(
            f"solved thresholds have no entry for warmup arm {arm!r}; "
            f"present: {sorted(solved['arms'])}"
        )
    grid = {cell["f_fh"] for cell in record["cells"]}
    if grid != set(FH_GRID):
        raise SystemExit(
            f"{arm}: solved grid {sorted(grid)} != FH_GRID {list(FH_GRID)}; the "
            "two lines would no longer be comparable cell for cell"
        )
    emitted = {}
    for cell in record["cells"]:
        pct = int(round(cell["f_fh"] * 100))
        yaml_id = f"gp_{binding.tag}_fh{pct:02d}"
        emitted[yaml_id] = _write(
            build_eval(binding, theta=record["theta"], t_fh=cell["t_fh"]),
            binding.config_root / "eval",
            yaml_id,
            expect_gate_l=True,
        )
    return emitted


def solved_theta(binding: gpb.Binding) -> float:
    """Read the suite's solved gate theta back out of an emitted sweep arm.

    All 16 cells share one theta (the 0.85 top-fraction solve); fh80 is picked
    arbitrarily. Reading it back rather than re-solving keeps the gate-only arm
    provably on the same gate as the sweep.
    """
    src = binding.config_root / "eval" / f"gp_{binding.tag}_fh80.yaml"
    if not src.is_file():
        raise SystemExit(
            f"cannot recover solved theta for {binding.tag}: {src} missing -- "
            "emit the sweep first (gate-only reuses its solved theta)"
        )
    gate = yaml.safe_load(src.read_text(encoding="utf-8"))["checkpoints"]["cp1"]["gate"]
    if gate["theta_low"] != gate["theta_high"]:
        raise SystemExit(f"{src}: theta_low != theta_high, refusing to guess")
    return float(gate["theta_low"])


def emit_gate_only(binding: gpb.Binding) -> dict[str, str]:
    """One gate-only yaml for the suite."""
    yaml_id = f"gpgo_{binding.tag}"
    return {
        yaml_id: _write(
            build_gate_only(binding, theta=solved_theta(binding)),
            binding.config_root / "gate_only",
            yaml_id,
            expect_gate_l=True,
        )
    }


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Emit GR00T gate-threshold Pareto YAMLs")
    ap.add_argument(
        "--mode", choices=("template", "warmup", "eval", "gate_only"), required=True
    )
    ap.add_argument("--suite", default="", help="default: every bound suite")
    ap.add_argument(
        "--solved", default="", help="thresholds json from solve_gtp (eval mode)"
    )
    args = ap.parse_args(argv)

    bindings = (
        [gpb.for_suite(args.suite)] if args.suite else list(gpb.BINDINGS)
    )
    emitted: dict[str, str] = {}
    for binding in bindings:
        if args.mode == "template":
            record = sync_template(binding)
            print(f"{binding.suite}: template {record['sha256'][:12]} -> "
                  f"{record['template_path']}")
            continue
        if args.mode == "warmup":
            emitted.update(emit_warmup(binding))
        elif args.mode == "gate_only":
            emitted.update(emit_gate_only(binding))
        else:
            if not args.solved:
                raise SystemExit("--solved is required in eval mode")
            solved = json.loads(pathlib.Path(args.solved).read_text(encoding="utf-8"))
            emitted.update(emit_eval(binding, solved))

    for yaml_id, path in sorted(emitted.items()):
        print(f"{yaml_id:16s} {path}")
    if emitted:
        print(f"\n{len(emitted)} yaml(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
