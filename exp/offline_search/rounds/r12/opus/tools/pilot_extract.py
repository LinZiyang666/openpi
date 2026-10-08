"""R12 opus pilot analysis: compact per-look tables of my pilot arms and deployed replicates (B-dev only).

Reads only the fields needed (lazy npz members; no keys/images), only accepted attempts (journal), and never any
arm whose name contains 'astra'. Output: out/pilot/<src>__<arm>.npz (+ meta json) for tools/pilot_analyse.py.
"""
from __future__ import annotations

import glob
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

R12 = Path("/home/weiland/projects/openpi/exp/offline_search/rounds/r12")
TRACE = Path("/home/weiland/trace_runs/os_closed_loop")
OUTD = Path(__file__).resolve().parents[1] / "out" / "pilot"


def arm_dirs():
    pats = [R12 / "lookhalf/pilots/*/runs/*", R12 / "guard/pilots/*/runs/*", TRACE / "r12_devlooks_*/runs/*"]
    out = []
    for p in pats:
        for d in sorted(glob.glob(str(p))):
            if "astra" in d or not Path(d).is_dir():
                continue
            if not glob.glob(d + "/server_*/looks/manifest_*.json"):
                continue
            if not Path(d, "client", "journal.jsonl").exists():
                continue
            out.append(d)
    return out


def journal(d):
    acc = {}
    for line in open(Path(d) / "client" / "journal.jsonl"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("accepted") and r.get("status") in ("done", "failed") and not r.get("error"):
            acc[r["task_uid"]] = (int(r.get("attempt", 1) or 1), bool(r["success"]))
    return acc


def method_state(gs, key):
    for _, st in sorted((gs.get("method") or {}).items()):
        if key in st:
            return st[key]
    return None


def extract(d):
    d = Path(d)
    src = d.parent.parent.name
    out_p = OUTD / f"{src}__{d.name}.npz"
    jr = journal(d)
    cols = {k: [] for k in ("uid", "task", "init", "step", "hit", "reason", "look_reason", "noprog_span", "top1",
                            "ncam", "ncomp", "plan", "creason", "h2_risk", "h2_adv", "h2_stall", "h2_riskb",
                            "h2_call", "h2_ta", "h2_tb", "m1_advw", "m1_advwl", "m1_score", "knob_call", "success",
                            "rs")}
    for man in sorted(glob.glob(str(d / "server_*" / "looks" / "manifest_*.json"))):
        data = json.loads(Path(man).read_text())
        for sh in data["shards"]:
            acc = jr.get(sh["uid"])
            if acc is None or int(sh["attempt"]) != acc[0]:
                continue
            with np.load(Path(man).parent / sh["file"], allow_pickle=False) as z:
                gs = json.loads(str(z["guard_state"]))
                ex = gs.get("extras") or {}
                lh = json.loads(str(z["lookhalf"])) if "lookhalf" in z.files else {}
                cost = json.loads(str(z["lookhalf_cost"])) if "lookhalf_cost" in z.files else {}
                h2 = (lh.get("h2") or {}).get("scalars") or {}
                cols["uid"].append(sh["uid"])
                cols["task"].append(int(z["task_id"]))
                cols["init"].append(int(z["init"]))
                cols["step"].append(int(z["step"]))
                cols["hit"].append(bool(z["hit"]))
                cols["reason"].append(float(ex.get("os_reason", 0.0)))
                cols["look_reason"].append(float(gs.get("look_reason") if gs.get("look_reason") is not None else -1))
                sp = method_state(gs, "_noprog_span")
                cols["noprog_span"].append(float(sp) if sp is not None else float(ex.get("noprog_span", np.nan)))
                cols["top1"].append(int(z["topk"][0]) if len(z["topk"]) else -1)
                cols["ncam"].append(int(cost.get("camera_encoded_count", 2)))
                cols["ncomp"].append(int(cost.get("camera_completion_count", 0)))
                cols["plan"].append(str(lh.get("plan", "")))
                cols["creason"].append(str(lh.get("completion_reason", "")))
                cols["h2_risk"].append(float(h2.get("risk", np.nan)))
                cols["h2_adv"].append(float(h2.get("advance", np.nan)))
                cols["h2_stall"].append(bool(h2.get("stall_branch", False)))
                cols["h2_riskb"].append(bool(h2.get("risk_branch", False)))
                cols["h2_call"].append(bool(h2.get("call", False)))
                cols["h2_ta"].append(float(h2.get("theta_a", np.nan)))
                cols["h2_tb"].append(float(h2.get("theta_b", np.nan)))
                cols["m1_advw"].append(float(lh.get("advance_w", np.nan)))
                cols["m1_advwl"].append(float(lh.get("advance_wl", np.nan)))
                cols["m1_score"].append(float(lh.get("score", np.nan)))
                cols["knob_call"].append(float(ex.get("os_r11_knob_call", 0.0)))
                cols["success"].append(acc[1])
                cols["rs"].append(np.asarray(z["rs"][:8], np.float32))
    summ = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
    led = summ.get("lookhalf_cost_ledger") or {}
    cl = summ.get("cost_ledger") or {}
    meta = dict(src=src, arm=d.name, path=str(d), n_journal=len(jr), sr_journal=float(np.mean([v[1] for v in jr.values()])) if jr else None,
                summary=bool(summ), model=cl.get("model") or summ.get("model"), sr=summ.get("sr"),
                decisions=led.get("decisions", cl.get("decisions")), vision=led.get("vision_decisions", cl.get("vision_decisions")),
                misses=led.get("misses", cl.get("misses")), one=led.get("one_camera_looks", 0), work=led.get("work"),
                completions=led.get("completions", 0), completion_reasons=led.get("completion_reasons"),
                kwargs=summ.get("kwargs"),
                uids={u: v[1] for u, v in jr.items()})
    arrays = {k: np.asarray(v) for k, v in cols.items()}
    OUTD.mkdir(parents=True, exist_ok=True)
    tmp = out_p.with_suffix(".tmp.npz")
    np.savez(tmp, meta_json=np.array(json.dumps(meta)), **arrays)
    tmp.replace(out_p)
    return f"{src}/{d.name}: {len(cols['uid'])} looks, {len(jr)} eps"


if __name__ == "__main__":
    dirs = arm_dirs()
    only = sys.argv[1:]
    if only:
        dirs = [d for d in dirs if any(o in d for o in only)]
    with mp.get_context("fork").Pool(16) as pool:
        for msg in pool.imap_unordered(extract, dirs):
            print(msg, flush=True)
