"""Emit the online RIT arm set: R', F, O-init, O-cold (and the S3b / FM-0 variants).

Every arm is the suite's R template with exactly three sections replaced --
``judge``, ``gate`` and ``backend.in_memory.preload_path`` -- so a difference
between arms can only come from those. The gate is the R line's hysteresis
gate (theta / j / p / L from the R arm record) with ``include_ws: true``,
because the new ladder has no FULL_HIT and the L cap would otherwise never
fire.

Arms per target (from the delta ladder the replays produced):

  rprime   ``threshold`` judge, FULL disabled (threshold 2.0 above the score
           domain), three warm tiers at R's nested D cuts; dead / shadowed
           rungs dropped and recorded.
  f        ``online_rit`` from the full init state, ``update_enabled: false``.
  oinit    same init state, updates on.
  ocold    empty state, updates on (same knots / delta / scales).

Each arm yaml is loaded back through the production loader, and the
structured diff against the template is asserted to touch only the three
sections. yaml_id (the file stem) is the learning identity: smoke, formal and
repeat runs must use different stems.

Four matrices are written because ``run_gtp.validate_arms`` checks one judge
type per launch and every launch binds one init pool: ``rprime`` (threshold
judge, ``--judge-type threshold --warm-tiers 0.875,0.75,0.5``, A500),
``frozen`` (F / S3b-F: online_rit with updates off, A500, may be sharded),
``online_a500`` (O-init / FM-0 / S3b-O: one server process per arm,
``--eval-concurrency 1``, A500) and ``online_adapt`` (O-cold: A_adapt 25/task,
``--trials 25``, one process, concurrency 1). The record names each matrix's
cohort so the launcher cannot pair an arm with the wrong pool.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib

import yaml

from exp.online_rit import ladder3
from exp.online_rit.common import ALPHA, H_EXEC, N_MIN, SNAPSHOT_EVERY, SUITE_TAG, TIER_TS, WINDOW, sha256_file, write_json
from openpi.cache.config import load_cache_config

MATRIX_COHORT = {
    "rprime": {"pool": "a500", "trials": 50, "judge_type": "threshold", "single_process": False},
    "frozen": {"pool": "a500", "trials": 50, "judge_type": "online_rit", "single_process": False},
    "online_a500": {"pool": "a500", "trials": 50, "judge_type": "online_rit", "single_process": True},
    "online_adapt": {"pool": "adapt", "trials": 25, "judge_type": "online_rit", "single_process": True},
    "smoke": {"pool": "smoke", "trials": 1, "judge_type": "online_rit", "single_process": True},
    "terminal": {"pool": "terminal", "trials": 25, "judge_type": "online_rit", "single_process": False},
}

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PROTOCOL = "online_rit_arms_v1"
FULL_DISABLED_CUT = 2.0
ALLOWED_SECTIONS = ("judge", "gate", "preload_path")


def gate_section(theta: float, *, j: int = 3, probe_interval: int = 3, L: int = 6) -> dict:
    return {
        "type": "score_hysteresis",
        "theta_low": float(theta),
        "theta_high": float(theta),
        "j": int(j),
        "probe_interval": int(probe_interval),
        "L": int(L),
        "include_ws": True,
    }


def rprime_judge(cuts: dict[str, float], tiers) -> tuple[dict, list[str]]:
    """Threshold judge with three warm tiers; rungs that can never fire are dropped."""
    warm, dropped, prev = [], [], FULL_DISABLED_CUT
    for t in tiers:
        cut = cuts[t.name]
        if not math.isfinite(cut) or cut >= prev:
            dropped.append(t.name)
            continue
        warm.append({"threshold": float(cut), "start_t": float(t.start_t)})
        prev = cut
    judge = {"type": "threshold", "threshold": FULL_DISABLED_CUT}
    if warm:
        judge["warm_tiers"] = warm
    return judge, dropped


def online_judge(*, delta: float, knots: list[float], scales_path: str, init_state_path: str | None, update_enabled: bool, feedback_mode: str, state_log_dir: str, source_library_sha256: str | None = None) -> dict:
    judge = {
        "type": "online_rit",
        "tiers": [float(t) for t in TIER_TS],
        "alpha": ALPHA,
        "delta": float(delta),
        "knots": [float(k) for k in knots],
        "update_scales_path": str(scales_path),
        "feedback_mode": feedback_mode,
        "update_enabled": bool(update_enabled),
        "window": WINDOW,
        "n_min": N_MIN,
        "h_exec": H_EXEC,
        "state_log_dir": str(state_log_dir),
        "snapshot_every": SNAPSHOT_EVERY,
    }
    if init_state_path:
        judge["init_state_path"] = str(init_state_path)
    if source_library_sha256:
        judge["source_library_sha256"] = source_library_sha256
    return judge


def build_arm(template: dict, *, judge: dict, gate: dict, preload_path: str | None) -> dict:
    doc = copy.deepcopy(template)
    cp1 = doc["checkpoints"]["cp1"]
    cp1["judge"] = judge
    cp1["gate"] = gate
    if preload_path:
        doc["backend"]["in_memory"]["preload_path"] = str(preload_path)
    doc["write_policy"] = {"type": "never"}
    return doc


def structured_diff(template: dict, arm: dict) -> list[str]:
    """Dotted paths that differ, excluding the three allowed sections."""
    diffs: list[str] = []

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                walk(a.get(k), b.get(k), f"{path}.{k}" if path else k)
        elif a != b:
            diffs.append(path)

    walk(template, arm, "")
    allowed = ("checkpoints.cp1.judge", "checkpoints.cp1.gate", "backend.in_memory.preload_path", "write_policy")
    return [d for d in diffs if not any(d == a or d.startswith(a + ".") for a in allowed)]


def write_arm(doc: dict, path: pathlib.Path, template: dict) -> str:
    stray = structured_diff(template, doc)
    if stray:
        raise SystemExit(f"{path.name}: arm differs from the template outside the allowed sections: {stray}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    load_cache_config(str(path))
    return sha256_file(path)


def emit(args) -> dict:
    tag = SUITE_TAG[args.suite]
    template_path = pathlib.Path(args.template)
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    arm_record = json.loads(pathlib.Path(args.r_arm_record).read_text(encoding="utf-8"))
    gate = gate_section(float(arm_record["gate_theta"]))
    knots = json.loads(pathlib.Path(args.knots).read_text(encoding="utf-8"))["knots"]
    online = json.loads(pathlib.Path(args.online_ladder).read_text(encoding="utf-8"))
    rprime = json.loads(pathlib.Path(args.rprime_ladder).read_text(encoding="utf-8"))
    if not getattr(args, "smoke", False) and any(doc.get("provisional") is not False for doc in (online, rprime)):
        raise SystemExit("formal arms require non-provisional cost addressing")
    from exp.online_rit.fit_init_curves import rprime_fit_from_record

    rfit = rprime_fit_from_record(json.loads(pathlib.Path(args.rprime_fit).read_text(encoding="utf-8")))
    tiers = ladder3.warm_tiers()
    out_dir = pathlib.Path(args.out_dir) / args.suite
    arms_dir = out_dir / "arms"
    common = sorted(
        {k for k, v in online["targets"].items() if v["found"] and v["duplicate_of"] is None}
        & {k for k, v in rprime["targets"].items() if v["found"] and v["duplicate_of"] is None},
        key=float,
    )
    record: dict = {
        "protocol": PROTOCOL,
        "suite": args.suite,
        "template": str(template_path),
        "template_sha256": sha256_file(template_path),
        "gate": gate,
        "knots": knots,
        "scales_sha256": sha256_file(args.scales),
        "init_state_sha256": sha256_file(args.init_state),
        "library_pkl": args.library_pkl,
        "common_targets": common,
        "ladders": {"online": args.online_ladder, "rprime": args.rprime_ladder},
        "run_tag": args.run_tag,
        "arms": {},
    }
    matrices: dict[str, list[dict]] = {}

    def add(arm: str, doc: dict, matrix: str, meta: dict) -> None:
        if smoke:
            if meta["rule"] not in ("oinit", "ocold"):
                return
            matrix = "smoke"
        path = arms_dir / f"{arm}.yaml"
        meta["yaml"] = str(path)
        meta["sha256"] = write_arm(doc, path, template)
        record["arms"][arm] = meta
        matrices.setdefault(matrix, []).append({"arm": arm, "yaml": str(path), "suite": args.suite})

    state_root = str(pathlib.Path(args.state_log_root) / args.suite)
    smoke = bool(getattr(args, "smoke", False))
    for key in (common[:1] if smoke else common):
        tgt = int(float(key))
        delta = float(online["targets"][key]["delta"])
        delta_d = float(rprime["targets"][key]["delta"])
        stem = f"{tag}_{args.run_tag}"
        judge, dropped = rprime_judge(ladder3.rprime_cuts(rfit, delta_d), tiers)
        add(f"{stem}_rprime_ir{tgt}", build_arm(template, judge=judge, gate=gate, preload_path=args.library_pkl), "rprime",
            {"rule": "rprime", "target": tgt, "delta_D": delta_d, "dropped": dropped})
        add(f"{stem}_f_ir{tgt}", build_arm(template, judge=online_judge(delta=delta, knots=knots, scales_path=args.scales, init_state_path=args.init_state, update_enabled=False, feedback_mode="fm1", state_log_dir=state_root), gate=gate, preload_path=args.library_pkl), "frozen",
            {"rule": "f", "target": tgt, "delta": delta})
        add(f"{stem}_oinit_ir{tgt}", build_arm(template, judge=online_judge(delta=delta, knots=knots, scales_path=args.scales, init_state_path=args.init_state, update_enabled=True, feedback_mode="fm1", state_log_dir=state_root), gate=gate, preload_path=args.library_pkl), "online_a500",
            {"rule": "oinit", "target": tgt, "delta": delta})
        add(f"{stem}_ocold_ir{tgt}", build_arm(template, judge=online_judge(delta=delta, knots=knots, scales_path=args.scales, init_state_path=None, update_enabled=True, feedback_mode="fm1", state_log_dir=state_root), gate=gate, preload_path=args.library_pkl), "online_adapt",
            {"rule": "ocold", "target": tgt, "delta": delta})
        if args.fm0_targets and tgt in args.fm0_targets:
            add(f"{stem}_oinit_fm0_ir{tgt}", build_arm(template, judge=online_judge(delta=delta, knots=knots, scales_path=args.scales, init_state_path=args.init_state, update_enabled=True, feedback_mode="fm0", state_log_dir=state_root), gate=gate, preload_path=args.library_pkl), "online_a500",
                {"rule": "oinit_fm0", "target": tgt, "delta": delta})
        if args.s3b_pkl and args.s3b_targets and tgt in args.s3b_targets:
            s3b_sha = sha256_file(args.s3b_pkl)
            src_sha = sha256_file(args.library_pkl)
            add(f"{stem}_rprime_s3b_ir{tgt}", build_arm(template, judge=judge, gate=gate, preload_path=args.s3b_pkl), "rprime",
                {"rule": "rprime_s3b", "target": tgt, "delta_D": delta_d, "dropped": dropped, "library_sha256": s3b_sha})
            add(f"{stem}_f_s3b_ir{tgt}", build_arm(template, judge=online_judge(delta=delta, knots=knots, scales_path=args.scales, init_state_path=args.init_state, update_enabled=False, feedback_mode="fm1", state_log_dir=state_root, source_library_sha256=src_sha), gate=gate, preload_path=args.s3b_pkl), "frozen",
                {"rule": "f_s3b", "target": tgt, "delta": delta, "library_sha256": s3b_sha})
            add(f"{stem}_oinit_s3b_ir{tgt}", build_arm(template, judge=online_judge(delta=delta, knots=knots, scales_path=args.scales, init_state_path=args.init_state, update_enabled=True, feedback_mode="fm1", state_log_dir=state_root, source_library_sha256=src_sha), gate=gate, preload_path=args.s3b_pkl), "online_a500",
                {"rule": "oinit_s3b", "target": tgt, "delta": delta, "library_sha256": s3b_sha})
    for matrix, rows in matrices.items():
        cohort = MATRIX_COHORT[matrix]
        doc = {"arms": rows, "cohort": cohort}
        (out_dir / f"arm_matrix_{args.run_tag}_{matrix}.yaml").write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    record["matrices"] = {m: {"path": str(out_dir / f"arm_matrix_{args.run_tag}_{m}.yaml"), **MATRIX_COHORT[m]} for m in matrices}
    write_json(out_dir / f"arm_record_{args.run_tag}.json", record)
    return record


def _targets(text: str) -> set[int]:
    return {int(float(x)) for x in text.split(",") if x.strip()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True, choices=tuple(SUITE_TAG))
    ap.add_argument("--template", required=True, help="exp/libero_groot/config/rit/<suite>/template.yaml")
    ap.add_argument("--r-arm-record", required=True, help="R arm_record.json (gate theta)")
    ap.add_argument("--knots", required=True)
    ap.add_argument("--scales", required=True, help="update_scales.npz (server path)")
    ap.add_argument("--init-state", required=True, help="init_state.json (server path)")
    ap.add_argument("--rprime-fit", required=True)
    ap.add_argument("--online-ladder", required=True, help="ir_replay output for the online rule")
    ap.add_argument("--rprime-ladder", required=True, help="ir_replay output for R'")
    ap.add_argument("--library-pkl", required=True, help="S3 pkl (server path)")
    ap.add_argument("--s3b-pkl", default="")
    ap.add_argument("--s3b-targets", default="50,70,90")
    ap.add_argument("--fm0-targets", default="")
    ap.add_argument("--state-log-root", required=True, help="server-side online state root")
    ap.add_argument("--smoke", action="store_true", help="only O-init/O-cold at the first common target, one init per task")
    ap.add_argument("--run-tag", required=True, help="smoke | formal | rep1 ...; part of every yaml_id")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    args.s3b_targets = _targets(args.s3b_targets)
    args.fm0_targets = _targets(args.fm0_targets)
    rec = emit(args)
    print(f"{len(rec['arms'])} arms for targets {rec['common_targets']} -> {rec['matrices']}")


if __name__ == "__main__":
    main()
