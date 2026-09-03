"""K=3 export: shadow table -> threshold-judge arm yamls for the RIT and GST rules.

Both rules deploy through the production ``threshold`` judge --
``threshold`` (FULL_HIT) plus ``warm_tiers`` ``[{θ_w03, 0.3}, {θ_w05, 0.5}]`` in
descending-threshold order -- so the two groups differ only in where the three
cuts come from:

  rit   ``rit_k`` K=3 fit on the tau1 shadow table (columns y_tau10 / y_tau7 /
        y_tau5), one delta per addressed inference ratio; a tier whose cut is
        +inf is omitted from the yaml.
  gst   percent triples (fh, w3, w5) on the step-20 simplex
        (fh + w3 + w5 <= 80, (0,0,0) excluded: 34 cells); cuts are descending
        score quantiles of the same table's ``s`` with the GTP / tgrid convention
        (``derive_thresholds``: the last passing score of each cumulative share).
        A cell whose cuts coincide (a share that maps to zero rows) is skipped
        and recorded.

Outputs under --out-dir: ``export_record.json`` (provenance, per-arm cuts,
predicted IR on the shadow rows for BOTH rules), ``<rule>/<arm>.yaml`` and
``arm_matrix_<rule>.yaml`` in the ``run_gtp`` schema.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import pathlib
import subprocess
import sys

import numpy as np
import yaml

from exp.dispatch_surface.analysis.analytic_cost import cost_model_digest
from exp.gate_threshold_pareto import libraries as libs
from exp.rit_pareto import rit_k
from openpi.cache.config import load_cache_config

PROTOCOL = "rit_pareto_k3_v1"
SUITE_TAG = {"libero_spatial": "sp", "libero_10": "l10"}
RULES = ("rit", "gst")
DEFAULT_TARGETS = tuple(float(x) for x in range(20, 100, 5))
GST_STEP = 20
GST_MAX_SUM = 80
DEFAULT_ALPHA = 0.05
DEFAULT_H_EXEC = 5


def _sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(libs.REPO_ROOT), text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def load_rows(table: pathlib.Path, ref_mode: str = "tau1") -> list[dict]:
    rows = [json.loads(line) for line in open(table) if line.strip()]
    rows = [r for r in rows if r["ref_mode"] == ref_mode]
    if not rows:
        raise SystemExit(f"no rows with ref_mode={ref_mode} in {table}")
    for key in ("s", "y_tau10", "y_tau7", "y_tau5"):
        if any(key not in r or r[key] is None for r in rows):
            raise SystemExit(f"{table}: column {key} missing or null (build the table with --extra-warm-tiers 0.5)")
    return rows


# ------------------------------------------------------------------
# Rules
# ------------------------------------------------------------------


def gst_cells(step: int = GST_STEP, max_sum: int = GST_MAX_SUM) -> list[tuple[int, int, int]]:
    """(fh, w3, w5) percent triples on the step simplex, all-zero excluded."""
    vals = list(range(0, max_sum + 1, step))
    cells = [c for c in itertools.product(vals, repeat=3) if 0 < sum(c) <= max_sum]
    return sorted(cells)


def gst_cuts(scores, shares: tuple[float, float, float]) -> list[float]:
    """Descending-quantile cuts for cumulative shares (tgrid convention)."""
    arr = np.sort(np.asarray([x for x in scores if x is not None and not math.isnan(x)], dtype=np.float64))[::-1]
    n = arr.size
    if n == 0:
        raise SystemExit("no usable scores")
    out = []
    cum = 0.0
    for share in shares:
        cum += share
        i = max(0, min(n - 1, int(cum * n) - 1))
        out.append(float(arr[i]) if share > 0 else math.inf)
    return out


def rit_thetas_to_list(thetas: dict[str, float]) -> list[float]:
    return [thetas[t.name] for t in rit_k.K3_TIERS]


def deployable_tiers(thetas: list[float]) -> list[tuple[rit_k.Tier, float]]:
    """Tiers that can actually fire, riskiest first: finite cuts strictly below
    every riskier deployed cut. A tier whose cut coincides with (or exceeds) a
    riskier tier's cut is shadowed by it -- the threshold judge would never reach
    it -- so it is dropped from the yaml (and reported as not deployed)."""
    out: list[tuple[rit_k.Tier, float]] = []
    prev = math.inf
    for t, th in zip(rit_k.K3_TIERS, thetas):
        if math.isfinite(th) and th < prev:
            out.append((t, th))
            prev = th
    return out


def build_arm(template: dict, *, preload_path: str, thetas: list[float]) -> tuple[dict, list[str]] | None:
    """Template + threshold judge with the deployable tiers of ``thetas``.

    Returns ``(yaml dict, deployed tier names)``; None when no tier can fire
    (an all-MISS rule)."""
    finite = deployable_tiers(thetas)
    if not finite:
        return None
    doc = copy.deepcopy(template)
    cp1 = doc["checkpoints"]["cp1"]
    # The threshold judge takes the first tier the score clears; without a
    # finite FULL cut the full threshold is set above the score range so that
    # only the warm tiers can fire.
    full = [th for t, th in finite if t.hit_type == "FULL_HIT"]
    judge = {"type": "threshold", "threshold": float(full[0]) if full else 2.0}
    warm = [{"threshold": float(th), "start_t": float(t.start_t)} for t, th in finite if t.hit_type == "WARM_START"]
    if warm:
        judge["warm_tiers"] = warm
    cp1["judge"] = judge
    cp1["gate"] = {"type": "always_search"}
    doc["backend"]["in_memory"]["preload_path"] = preload_path
    doc["write_policy"] = {"type": "never"}
    return doc, [t.name for t, _ in finite]


# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------


def export(args) -> dict:
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise SystemExit(f"--out-dir must be empty: {out_dir}")
    table = pathlib.Path(args.table)
    rows = load_rows(table, args.ref_mode)
    s = np.asarray([r["s"] for r in rows], dtype=np.float64)
    ys = {t.name: np.asarray([r[t.y_key] for r in rows], dtype=np.float64) for t in rit_k.K3_TIERS}
    template = yaml.safe_load((libs.REPO_ROOT / libs.TEMPLATE[args.suite]).read_text(encoding="utf-8"))
    tag = SUITE_TAG[args.suite]
    identity = {
        "protocol": PROTOCOL, "suite": args.suite, "ref_mode": args.ref_mode,
        "table_path": str(table.resolve()), "table_sha256": _sha(table),
        "library_pkl": args.library_pkl, "library_sha256": _sha(pathlib.Path(args.library_pkl_local)) if args.library_pkl_local else None,
        "template": str(libs.TEMPLATE[args.suite]), "n_rows": int(len(s)),
        "n_episodes": int(len(set(r["episode_id"] for r in rows))),
        "tiers": [{"name": t.name, "hit_type": t.hit_type, "start_t": t.start_t, "y_key": t.y_key, "cost_ms": t.cost_ms}
                  for t in rit_k.K3_TIERS],
        "miss_cost_ms": rit_k.MISS_MS, "cost_model_digest": cost_model_digest(),
        "git_commit": _git_commit(), "python": sys.version.split()[0], "numpy": np.__version__,
    }

    # -- RIT: K=3 fit + IR-addressed deltas ------------------------------
    targets = [float(x) for x in args.target_ir.split(",") if x.strip()]
    if any(b <= a for a, b in zip(targets, targets[1:])) or any(not (0 < t < 100) for t in targets):
        raise SystemExit("--target-ir must be strictly increasing inside (0, 100)")
    picked = rit_k.choose_knots(s, rit_k.KNOT_LADDER)
    if picked is None:
        raise SystemExit("knot ladder exhausted (stop-loss)")
    knots, n_seg_req = picked
    fit = rit_k.fit_pl_quantile_k(s, ys, knots, n_seg_req=n_seg_req, alpha=args.alpha, eps_total=rit_k.EPS_TOTAL)
    ir_lo, ir_hi = rit_k.attainable_range(fit, s)
    rit_arms: dict[str, dict] = {}
    for target in targets:
        sol = rit_k.delta_for_ir(fit, s, target)
        if abs(sol["ir_gap"]) > rit_k.IR_MAX_GAP:
            raise SystemExit(f"target {target}: nearest attainable IR {sol['predicted_ir']:.3f} is {sol['ir_gap']:+.3f} away")
        if not (math.isfinite(sol["delta"]) and sol["delta"] > 0):
            raise SystemExit(f"target {target}: bad delta {sol['delta']}")
        name = f"ir{int(target):02d}" if float(target).is_integer() else "ir" + f"{target:g}".replace(".", "p")
        rit_arms[f"k3_{tag}_rit_{name}"] = {
            "target_ir": target, "delta": sol["delta"], "thetas": sol["thetas"],
            "predicted_ir": sol["predicted_ir"], "ir_gap": sol["ir_gap"],
            "floor_info": rit_k.floor_info(fit, s, sol["delta"]),
        }

    # -- GST: percent-triple grid on the same scores ----------------------
    gst_arms: dict[str, dict] = {}
    skipped: dict[str, dict] = {}
    seen_cuts: dict[tuple, str] = {}
    for fh, w3, w5 in gst_cells(args.gst_step, args.gst_max_sum):
        arm = f"k3_{tag}_gst_f{fh:02d}w{w3:02d}v{w5:02d}"
        th = gst_cuts(s, (fh / 100.0, w3 / 100.0, w5 / 100.0))
        thetas = {t.name: th[i] for i, t in enumerate(rit_k.K3_TIERS)}
        rec = {"cell": {"fh": fh, "w3": w3, "w5": w5}, "thetas": thetas,
               "predicted_ir": rit_k.predicted_ir(s, th)}
        key = tuple((t.name, th_) for t, th_ in deployable_tiers(th))
        if not key:
            skipped[arm] = rec | {"reason": "no deployable tier"}
        elif key in seen_cuts:
            skipped[arm] = rec | {"reason": f"same deployed cuts as {seen_cuts[key]}"}
        else:
            seen_cuts[key] = arm
            gst_arms[arm] = rec

    # -- yamls + matrices --------------------------------------------------
    record = dict(identity)
    record.update({
        "rit": {"targets": targets, "attainable_ir_range": [ir_lo, ir_hi], "alpha": args.alpha,
                "fit": rit_k.fit_record_fields(fit), "fit_digests": rit_k.fit_digests(fit),
                "ir_curve": [[float(d), float(ir)] for d, ir in rit_k.ir_curve(fit, s)], "arms": rit_arms},
        "gst": {"step": args.gst_step, "max_sum": args.gst_max_sum, "arms": gst_arms, "skipped": skipped},
        "layers": {},
    })
    for rule in RULES:
        rule_dir = out_dir / rule
        rule_dir.mkdir()
        rows_out = []
        arms = record[rule]["arms"]
        for arm, rec in arms.items():
            thetas = rit_thetas_to_list(rec["thetas"])
            built = build_arm(template, preload_path=args.library_pkl, thetas=thetas)
            if built is None:
                raise SystemExit(f"{arm}: cuts {thetas} produce no deployable tier set")
            doc, deployed = built
            path = rule_dir / f"{arm}.yaml"
            path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
            cfg = load_cache_config(str(path))  # strict schema self-check
            cp1 = cfg.checkpoints["cp1"]
            got = [cp1.judge.threshold] + [t["threshold"] for t in (cp1.judge.warm_tiers or [])]
            if any(b >= a for a, b in zip(got, got[1:])):
                raise SystemExit(f"{path}: thresholds not strictly decreasing after round-trip: {got}")
            rec["yaml"] = str(path.resolve())
            rec["yaml_sha256"] = _sha(path)
            rec["deployed_tiers"] = deployed
            rows_out.append({"arm": arm, "yaml": str(path.resolve()), "suite": args.suite})
        matrix = {"protocol": PROTOCOL, "suite": args.suite, "rule": rule, "arms": rows_out}
        mpath = out_dir / f"arm_matrix_{rule}.yaml"
        mpath.write_text(yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8")
        record["layers"][rule] = {"matrix": str(mpath.resolve()), "n_arms": len(rows_out)}
    (out_dir / "export_record.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True, choices=tuple(SUITE_TAG))
    ap.add_argument("--table", required=True, help="tau1 shadow table with y_tau5")
    ap.add_argument("--library-pkl", required=True, help="server-side library path (preload_path)")
    ap.add_argument("--library-pkl-local", default="", help="local path of the same pkl for the sha record")
    ap.add_argument("--ref-mode", default="tau1")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--target-ir", default=",".join(f"{t:g}" for t in DEFAULT_TARGETS))
    ap.add_argument("--gst-step", type=int, default=GST_STEP)
    ap.add_argument("--gst-max-sum", type=int, default=GST_MAX_SUM)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    rec = export(args)
    print(f"RIT attainable IR {rec['rit']['attainable_ir_range']}")
    for arm, r in rec["rit"]["arms"].items():
        th = r["thetas"]
        print(f"  {arm}: target {r['target_ir']:5.1f} -> pred {r['predicted_ir']:6.2f} (gap {r['ir_gap']:+.3f}) "
              f"delta {r['delta']:.4f} full {th['full']:.6f} w03 {th['warm03']:.6f} w05 {th['warm05']:.6f} tiers {r['deployed_tiers']}")
    print(f"GST cells {len(rec['gst']['arms'])} emitted, {len(rec['gst']['skipped'])} skipped")
    for arm, r in rec["gst"]["arms"].items():
        th = r["thetas"]
        print(f"  {arm}: pred {r['predicted_ir']:6.2f} full {th['full']:.6f} w03 {th['warm03']:.6f} w05 {th['warm05']:.6f} tiers {r['deployed_tiers']}")
    for arm, r in rec["gst"]["skipped"].items():
        print(f"  skipped {arm}: {r['reason']} {r['thetas']}")


if __name__ == "__main__":
    main()
