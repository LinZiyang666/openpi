"""Common knots, the initial estimator state and R's D fit from the M1 table.

Three subcommands:

  knots   the common score knots (plan §3.3): domain ends 0 / 1 plus the
          1/8 ... 7/8 quantiles of the **calibration fit half** (out-of-library
          trajectories stamped ``split: fit``), de-duplicated; also the fixed
          quartile band edges every calibration consumer uses.
  init    ``init_state.json`` -- the online estimator fed with every calibration
          row of the table in (trajectory, decision) order, feedback mode fm1.
          This is the *complete* state F and O-init start from, stamped with the
          fixed params (scales sha / schedule / h_exec) the judge factory checks.
          Refused unless the three M1 gates PASSed **on this table**: Q1
          (``signal_check.json``), the parity gate (``parity_gate.json``) and the
          estimator replay (``replay.json``). Q1 and replay bind the table SHA;
          the builder's table record binds the preceding parity gate and inputs.
  rprime  ``rprime_fit.json`` -- the old method's joint nested pinball LP over
          the executed-dimension D columns on the same knots.

Public interface: ``build_init_state``, ``fit_rprime``, ``rprime_fit_from_record``,
``require_gates``.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from exp.online_rit import ladder3
from exp.online_rit.common import ALPHA, H_EXEC, N_MIN, SCHEDULE, WINDOW, load_jsonl, sha256_file, write_json
from exp.online_rit.provenance import table_record, validate_parity
from exp.online_rit.replay_sim import candidate_rows, fit_curves, score_bands
from exp.rit_pareto import rit_k


def knots_from_scores(scores: np.ndarray, n_seg: int = 8) -> list[float]:
    """Domain ends plus the 1/n ... (n-1)/n quantiles, de-duplicated; >= 3 knots or error."""
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0 or not np.isfinite(s).all():
        raise SystemExit("scores are empty or non-finite")
    qs = np.quantile(s, np.linspace(0, 1, n_seg + 1)[1:-1])
    knots = sorted(set([0.0, 1.0] + [float(round(q, 12)) for q in qs]))
    if len(knots) < 3:
        raise SystemExit(f"score distribution degenerate: {knots}")
    return knots


def fixed_params_for(scales_path: str, *, h_exec: int = H_EXEC) -> dict:
    return {"scales_sha256": sha256_file(scales_path), "schedule_id": SCHEDULE.schedule_id, "h_exec": int(h_exec)}


def build_init_state(rows: list[dict], knots: list[float], *, fixed_params: dict | None = None) -> dict:
    curves = fit_curves(rows, knots, alpha=ALPHA, window=WINDOW, n_min=N_MIN, fixed_params=fixed_params)
    state = curves.snapshot()
    state["source"] = {"n_rows": len(candidate_rows(rows)), "feedback_mode": "fm1", "order": "trajectory,decision", "population": "calibration"}
    from openpi.cache.components.online_rit import seal_state

    return seal_state(state)


def fit_rprime(rows: list[dict], knots: list[float], *, alpha: float = ALPHA) -> dict:
    tiers = ladder3.warm_tiers()
    cand = [r for r in candidate_rows(rows) if all(r.get(t.y_column) is not None for t in tiers)]
    s = np.array([float(r["s"]) for r in cand])
    ys = {t.name: np.array([float(r[t.y_column]) for r in cand]) for t in tiers}
    fit = ladder3.fit_nested(s, ys, np.asarray(knots, dtype=np.float64), alpha=alpha, tiers=tiers)
    return {
        "protocol": "online_rit_rprime_fit_v1",
        "knots": [float(k) for k in fit.knots],
        "q": ladder3.curves_from_fit(fit),
        "tiers": [t.name for t in tiers],
        "y_columns": {t.name: t.y_column for t in tiers},
        "alpha": float(alpha),
        "eps_total": float(fit.eps_total),
        "n_seg_req": int(fit.n_seg_req),
        "n_rows": len(cand),
    }


def rprime_fit_from_record(rec: dict) -> rit_k.PLFitK:
    tiers = ladder3.warm_tiers()
    proxies = ladder3._proxies(tiers)  # noqa: SLF001 - same proxy layout the fit used
    return rit_k.PLFitK(
        knots=np.asarray(rec["knots"], dtype=np.float64),
        q={name: np.asarray(v, dtype=np.float64) for name, v in rec["q"].items()},
        tiers=proxies,
        eps_total=float(rec["eps_total"]),
        n_seg_req=int(rec["n_seg_req"]),
        n_seg=len(rec["knots"]) - 1,
        alpha=float(rec["alpha"]),
    )


def require_gates(table_sha: str, *, signal: dict, parity: dict, replay: dict, knots_sha: str, record: dict | None = None, parity_sha: str | None = None) -> None:
    """Require PASS gates bound through parity -> table record -> table/knots."""
    problems = []
    if signal.get("release", {}).get("status") != "PASS":
        problems.append(f"Q1 signal gate is {signal.get('release', {}).get('status')!r}")
    if signal.get("table_sha256") != table_sha:
        problems.append("signal_check.json was computed on a different table")
    if parity.get("status") != "PASS":
        problems.append(f"parity gate is {parity.get('status')!r}: {parity.get('reasons')}")
    if record is None or not parity_sha or record.get("out_sha256") != table_sha or record.get("parity_gate_sha256") != parity_sha:
        problems.append("parity gate is not bound by this table record")
    else:
        validate_parity(parity, record.get("identity", {}))
    if replay.get("release_gate", {}).get("status") != "PASS":
        problems.append(f"replay release gate is {replay.get('release_gate', {}).get('status')!r}")
    if replay.get("table_sha256") != table_sha:
        problems.append("replay.json was computed on a different table")
    if replay.get("knots_sha256") != knots_sha:
        problems.append("replay.json was computed with different knots")
    if problems:
        raise SystemExit("init state refused:\n  - " + "\n  - ".join(problems))


def cmd_knots(args) -> None:
    rows = load_jsonl(args.table)
    fit = [r for r in rows if r.get("split") == "fit"]
    cand = candidate_rows(fit)
    knots = knots_from_scores(np.array([float(r["s"]) for r in cand]), args.n_seg)
    doc = {
        "knots": knots,
        "n_seg": args.n_seg,
        "score_bands": score_bands(fit),
        "n_scores": len(cand),
        "n_trajectories": len({r["trajectory_id"] for r in cand}),
        "population": "calibration_fit_half",
        "table_sha256": sha256_file(args.table),
    }
    write_json(args.out, doc)
    print(f"knots {knots} bands {doc['score_bands']} from {len(cand)} calibration fit rows")


def cmd_init(args) -> None:
    table_sha = sha256_file(args.table)
    record = table_record(args.table)
    knots_doc = json.loads(pathlib.Path(args.knots).read_text(encoding="utf-8"))
    if knots_doc.get("table_sha256") != table_sha:
        raise SystemExit("knots.json was derived from a different table")
    require_gates(
        table_sha,
        signal=json.loads(pathlib.Path(args.signal).read_text(encoding="utf-8")),
        parity=json.loads(pathlib.Path(args.parity).read_text(encoding="utf-8")),
        replay=json.loads(pathlib.Path(args.replay).read_text(encoding="utf-8")),
        knots_sha=sha256_file(args.knots),
        record=record, parity_sha=sha256_file(args.parity),
    )
    rows = load_jsonl(args.table)
    knots = knots_doc["knots"]
    params = fixed_params_for(args.scales, h_exec=args.h_exec)
    identity = record["identity"]
    with np.load(args.scales, allow_pickle=False) as npz:
        meta = json.loads(str(npz["meta_json"].item()))
    if any(params[k] != identity[k] for k in params) or meta.get("library_sha256") != identity["library_sha256"] or meta.get("schedule_id") != identity["schedule_id"]:
        raise SystemExit("init scales/library/schedule/h_exec differ from the table identity")
    state = build_init_state(rows, knots, fixed_params=params)
    state["table_record_sha256"] = sha256_file(str(args.table) + ".record.json")
    state["input_identity"] = identity
    state["table_sha256"] = table_sha
    state["knots_sha256"] = sha256_file(args.knots)
    from openpi.cache.components.online_rit import seal_state

    seal_state(state)
    out_dir = pathlib.Path(args.out_dir)
    write_json(out_dir / "init_state.json", state)
    print(f"init_state n_updates={state['n_updates']} learning_sha={state['learning_state_sha256'][:12]}")


def cmd_rprime(args) -> None:
    rows = load_jsonl(args.table)
    record = table_record(args.table)
    gate_path = getattr(args, "parity", None) or record.get("parity_gate")
    if not gate_path or sha256_file(gate_path) != record["parity_gate_sha256"]:
        raise SystemExit("Rprime parity file does not match the table record")
    validate_parity(json.loads(pathlib.Path(gate_path).read_text()), record["identity"])
    knots_doc = json.loads(pathlib.Path(args.knots).read_text(encoding="utf-8"))
    if knots_doc.get("table_sha256") != sha256_file(args.table):
        raise SystemExit("Rprime knots derived from a different table")
    rec = fit_rprime(rows, knots_doc["knots"])
    rec["input_identity"] = record["identity"]
    rec["table_record_sha256"] = sha256_file(str(args.table) + ".record.json")
    rec["table_sha256"] = sha256_file(args.table)
    rec["knots_sha256"] = sha256_file(args.knots)
    write_json(pathlib.Path(args.out_dir) / "rprime_fit.json", rec)
    print(f"rprime rows={rec['n_rows']} q={ {k: [round(x, 3) for x in v] for k, v in rec['q'].items()} }")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("knots")
    p.add_argument("--table", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-seg", type=int, default=8)
    p.set_defaults(fn=cmd_knots)
    p = sub.add_parser("init")
    p.add_argument("--table", required=True)
    p.add_argument("--knots", required=True)
    p.add_argument("--scales", required=True, help="update_scales.npz the served judge will use")
    p.add_argument("--signal", required=True, help="signal_check.json (Q1)")
    p.add_argument("--parity", required=True, help="parity_gate.json")
    p.add_argument("--replay", required=True, help="replay.json")
    p.add_argument("--h-exec", type=int, default=H_EXEC)
    p.add_argument("--out-dir", required=True)
    p.set_defaults(fn=cmd_init)
    p = sub.add_parser("rprime")
    p.add_argument("--table", required=True)
    p.add_argument("--knots", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--parity", default="", help="relocated parity file; default is the table record path")
    p.set_defaults(fn=cmd_rprime)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
