"""Emit the warmup and eval YAMLs for the hybrid-gate threshold sweep.

Structure is inherited verbatim from the gate line's server-side N4 template for
each suite (``libraries.TEMPLATE``): retrieval weights, per-field zscore+tanh
normalizers, trajectory depth 1, backend dims. Only three things are ever
touched, so a difference between this experiment and the gate line can only come
from one of them:

*   ``backend.in_memory.preload_path`` -- which of the four libraries is loaded.
*   ``checkpoints.cp1.gate`` -- ``always_search`` for warmup, the N4 hybrid
    (``score_hysteresis`` + ``L=6``) for eval.
*   ``checkpoints.cp1.judge`` -- force-MISS ``threshold: 2.0`` for warmup, the
    solved ``T_fh`` for eval. **Never a warm tier**: the warm-start route is
    disabled for this experiment, so the verdict is binary.

Every emitted file is round-tripped through ``load_cache_config`` before it is
accepted, and the eval emitter additionally asserts the warm tier is absent --
schema validity alone would not catch a warm tier that was silently inherited
from the template, and that tier is precisely what this experiment removes.

Public interface: ``build_warmup``, ``build_eval``, ``emit_warmup``, ``emit_eval``.
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib

import yaml

from openpi.cache.config import load_cache_config

from exp.gate_threshold_pareto import libraries as libs

FORCE_MISS = 2.0  # above the [0,1] score range -> the threshold judge never hits
FORCE_HIT = -1.0  # below the [0,1] score range -> the threshold judge always accepts

#: N4 winning point (gate line Stage 3a, 500-ep live decision 2026-07-05).
#: theta is filled per library from that library's own warmup.
GATE_J = 3
GATE_PROBE_INTERVAL = 3
GATE_L = 6


def _load_template(suite: str) -> dict:
    path = libs.REPO_ROOT / libs.TEMPLATE[suite]
    if not path.is_file():
        raise SystemExit(f"template for {suite} not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _base(lib: libs.Library) -> dict:
    cfg = copy.deepcopy(_load_template(lib.suite))
    cfg["backend"]["in_memory"]["preload_path"] = lib.preload_path
    cfg["write_policy"] = {"type": "never"}
    return cfg


def build_warmup(lib: libs.Library) -> dict:
    """Force-MISS warmup: search still runs, so every step records its score."""
    cfg = _base(lib)
    cp1 = cfg["checkpoints"]["cp1"]
    cp1["gate"] = {"type": "always_search"}
    cp1["judge"] = {"type": "threshold", "threshold": FORCE_MISS}
    return cfg


def build_eval(lib: libs.Library, *, theta: float, t_fh: float) -> dict:
    """One sweep cell: hybrid gate at the library's theta, binary verdict at t_fh."""
    cfg = _base(lib)
    cp1 = cfg["checkpoints"]["cp1"]
    cp1["gate"] = {
        "type": "score_hysteresis",
        "theta_low": theta,
        "theta_high": theta,
        "j": GATE_J,
        "probe_interval": GATE_PROBE_INTERVAL,
        "L": GATE_L,
    }
    cp1["judge"] = {"type": "threshold", "threshold": t_fh}
    return cfg


def _write(
    cfg: dict, out_dir: pathlib.Path, yaml_id: str, *, expect_no_warm: bool
) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{yaml_id}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    loaded = load_cache_config(path)  # strict schema self-check
    if expect_no_warm:
        tiers = loaded.checkpoints["cp1"].judge.warm_tiers
        if tiers:
            raise SystemExit(
                f"{path}: warm tier present ({tiers}); this experiment disables the "
                "warm-start route entirely, so any tier here is a template leak"
            )
    return str(path)


def emit_warmup(out_root: pathlib.Path) -> dict[str, str]:
    """One warmup yaml per library. Returns ``{yaml_id: path}``."""
    emitted = {}
    for lib in libs.LIBRARIES:
        yaml_id = f"gtpw_{libs.arm_key(lib)}"
        emitted[yaml_id] = _write(
            build_warmup(lib),
            out_root / lib.suite / "warmup",
            yaml_id,
            expect_no_warm=True,
        )
    return emitted


def emit_eval(out_root: pathlib.Path, solved: dict) -> dict[str, str]:
    """16 eval yamls per library, from a solved-threshold record."""
    emitted = {}
    for lib in libs.LIBRARIES:
        arm = libs.arm_key(lib)
        record = solved["arms"].get(f"gtpw_{arm}")
        if record is None:
            raise SystemExit(
                f"solved thresholds have no entry for warmup arm 'gtpw_{arm}'; "
                f"present: {sorted(solved['arms'])}"
            )
        for cell in record["cells"]:
            pct = int(round(cell["f_fh"] * 100))
            yaml_id = f"gtp_{arm}_fh{pct:02d}"
            emitted[yaml_id] = _write(
                build_eval(lib, theta=record["theta"], t_fh=cell["t_fh"]),
                out_root / lib.suite / "eval",
                yaml_id,
                expect_no_warm=True,
            )
    return emitted


def build_gate_only(lib: libs.Library, *, theta: float, gate_l: int) -> dict:
    """Gate-only ablation cell: verdict disabled, the hysteresis gate is the
    sole protection.

    ``judge.threshold = FORCE_HIT`` sits below the [0, 1] score range, so every
    probe the gate allows is accepted from the cache (the mirror of the warmup's
    ``FORCE_MISS`` trick). Teacher steps then come exclusively from the gate's
    own MISS state (lockout ``L`` after ``j`` consecutive sub-theta scores), so
    the measured teacher ratio is the gate's intrinsic intervention rate.
    """
    cfg = _base(lib)
    cp1 = cfg["checkpoints"]["cp1"]
    cp1["gate"] = {
        "type": "score_hysteresis",
        "theta_low": theta,
        "theta_high": theta,
        "j": GATE_J,
        "probe_interval": GATE_PROBE_INTERVAL,
        "L": gate_l,
    }
    cp1["judge"] = {"type": "threshold", "threshold": FORCE_HIT}
    return cfg


def _solved_theta(out_root: pathlib.Path, lib: libs.Library) -> float:
    """Read the library's solved gate theta back out of an existing eval yaml.

    All 16 sweep cells of one library share the same gate theta (the 0.85
    top-fraction solve); any cell works, fh80 is picked arbitrarily.
    """
    arm = libs.arm_key(lib)
    src = out_root / lib.suite / "eval" / f"gtp_{arm}_fh80.yaml"
    if not src.is_file():
        raise SystemExit(
            f"cannot recover solved theta for {arm}: {src} missing -- emit the "
            "eval sweep first (the gate-only mode reuses its solved thetas)"
        )
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    gate = cfg["checkpoints"]["cp1"]["gate"]
    if gate["theta_low"] != gate["theta_high"]:
        raise SystemExit(f"{src}: theta_low != theta_high, refusing to guess")
    return float(gate["theta_low"])


def emit_gate_only(out_root: pathlib.Path, *, gate_l: int) -> dict[str, str]:
    """One gate-only yaml per library + a per-suite arm matrix."""
    emitted = {}
    per_suite: dict[str, dict[str, str]] = {}
    for lib in libs.LIBRARIES:
        arm = libs.arm_key(lib)
        yaml_id = f"gtpgo_{arm}"
        theta = _solved_theta(out_root, lib)
        path = _write(
            build_gate_only(lib, theta=theta, gate_l=gate_l),
            out_root / lib.suite / "gate_only",
            yaml_id,
            expect_no_warm=True,
        )
        emitted[yaml_id] = path
        per_suite.setdefault(lib.suite, {})[yaml_id] = path
    for suite, arms in per_suite.items():
        matrix_path = out_root / suite / "gate_only_matrix.yaml"
        matrix_path.write_text(
            yaml.safe_dump(
                _matrix(arms, {y: suite for y in arms}), sort_keys=False
            ),
            encoding="utf-8",
        )
        print(f"matrix: {matrix_path}")
    return emitted


def _matrix(emitted: dict[str, str], suite_of: dict[str, str]) -> dict:
    return {
        "arms": [
            {"arm": yaml_id, "yaml": path, "suite": suite_of[yaml_id]}
            for yaml_id, path in sorted(emitted.items())
        ]
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Emit hybrid-gate threshold sweep YAMLs")
    ap.add_argument("--mode", choices=("warmup", "eval", "gate_only"), required=True)
    ap.add_argument(
        "--gate-l",
        type=int,
        default=None,
        help="gate lockout L for gate_only mode (required there, unused elsewhere)",
    )
    ap.add_argument("--out-root", default="exp/gate_threshold_pareto/config")
    ap.add_argument(
        "--solved",
        action="append",
        default=[],
        help="thresholds json (eval mode; repeatable, one per suite)",
    )
    ap.add_argument("--matrix-out", default="", help="write an arm matrix yaml here")
    args = ap.parse_args(argv)

    out_root = pathlib.Path(args.out_root)
    if args.mode == "gate_only":
        if args.gate_l is None:
            raise SystemExit("--gate-l is required in gate_only mode")
        emitted = emit_gate_only(out_root, gate_l=args.gate_l)
        for yaml_id, path in sorted(emitted.items()):
            print(f"{yaml_id}: {path}")
        return 0
    if args.mode == "warmup":
        emitted = emit_warmup(out_root)
        suite_of = {f"gtpw_{libs.arm_key(x)}": x.suite for x in libs.LIBRARIES}
    else:
        if not args.solved:
            raise SystemExit("--solved is required in eval mode")
        solved = {"arms": {}}
        for src in args.solved:
            part = json.loads(pathlib.Path(src).read_text(encoding="utf-8"))
            overlap = set(part["arms"]) & set(solved["arms"])
            if overlap:
                # Two solve outputs claiming the same arm would silently let the
                # last file on the command line decide that arm's thresholds.
                raise SystemExit(
                    f"{src}: arm(s) {sorted(overlap)} already solved elsewhere"
                )
            solved["arms"].update(part["arms"])
        emitted = emit_eval(out_root, solved)
        suite_of = {}
        for lib in libs.LIBRARIES:
            for cell in solved["arms"][f"gtpw_{libs.arm_key(lib)}"]["cells"]:
                suite_of[
                    f"gtp_{libs.arm_key(lib)}_fh{int(round(cell['f_fh'] * 100)):02d}"
                ] = lib.suite

    for yaml_id, path in sorted(emitted.items()):
        print(f"{yaml_id:24s} {path}")
    print(f"\n{len(emitted)} yaml(s)")

    if args.matrix_out:
        by_suite: dict[str, list] = {}
        for row in _matrix(emitted, suite_of)["arms"]:
            by_suite.setdefault(row["suite"], []).append(row)
        for suite, rows in by_suite.items():
            path = pathlib.Path(args.matrix_out.replace("{suite}", suite))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                yaml.safe_dump({"arms": rows}, sort_keys=False), encoding="utf-8"
            )
            print(f"matrix: {path} ({len(rows)} arms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
