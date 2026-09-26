"""Dev helper of family f2_state: run harness.smoke for every batch.json variant on several cells (parallel, pinned
to the CPUs given by --cpus, <= --procs processes) and summarize the per-decision npz outputs with a regime split
and B0 (r00 full run) on the same rows.

    python exp/offline_search/rounds/r01/f2_state/smoke_driver.py --episodes 5 --cells pi05_spatial_inf,... \
        --out /tmp/f2_smoke/e5 [--ref] [--procs 12] [--cpus 6-11,50-55] [--only M2,M3]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import subprocess
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[5]
FAM = pathlib.Path(__file__).resolve().parent
R00 = REPO / "exp/offline_search/results/r00"
PY = str(REPO / ".venv/bin/python")


def name_of(spec):
    for d in (str(REPO), str(FAM)):
        if d not in sys.path:
            sys.path.insert(0, d)
    import importlib.util
    mod, cls = spec["method"].rsplit(":", 1)
    p = REPO / mod
    sp = importlib.util.spec_from_file_location("f2_" + p.stem, p)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return getattr(m, cls)(**spec["kwargs"]).name


def run_one(spec, cell, a):
    out = pathlib.Path(a.out) / cell
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["taskset", "-c", a.cpus, PY, "-m", "exp.offline_search.harness.smoke", "--method", spec["method"],
           "--kwargs", json.dumps(spec["kwargs"]), "--cell", cell, "--episodes", str(a.episodes),
           "--root", a.root, "--out", str(out)]
    if not a.ref:
        cmd.append("--no-ref")
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def aurc(conf, e):
    o = np.argsort(-conf, kind="stable")
    return float((np.cumsum(e[o]) / np.arange(1, e.size + 1)).mean())


def summarize(name, cell, a):
    f = pathlib.Path(a.out) / cell / name / f"{cell}.npz"
    if not f.exists():
        return None
    d = np.load(f)
    b = np.load(R00 / "B0_current" / f"{cell}.npz")
    pos = {int(r): i for i, r in enumerate(b["row"])}
    bi = np.array([pos[int(r)] for r in d["row"]])
    e, c, eb, cb = d["err"], d["confidence"], b["err"][bi], b["confidence"][bi]
    reg = d["x_regime"] if "x_regime" in d.files else np.where(d["step"] == 0, -1, 0)
    res = {"n": int(e.size), "err": float(e.mean()), "aurc": aurc(c, e), "grip": float(d["grip_mis"].mean()),
           "B0_err": float(eb.mean()), "B0_aurc": aurc(cb, eb)}
    for k, lab in ((-1, "s0"), (0, "fresh"), (1, "stale")):
        m = reg == k
        res[f"n_{lab}"] = int(m.sum())
        res[f"err_{lab}"] = float(e[m].mean()) if m.any() else float("nan")
        res[f"B0err_{lab}"] = float(eb[m].mean()) if m.any() else float("nan")
    j = json.loads((pathlib.Path(a.out) / cell / name / f"{cell}.json").read_text())
    res["ms_q"] = j["timing"].get("ms_per_query", float("nan"))
    res["fit_s"] = j["fit"]["fit_s"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--cells", default="pi05_spatial_inf,pi05_spatial_cache,groot_l10_inf,groot_l10_cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default="/dev/shm/offline_search_store")
    ap.add_argument("--ref", action="store_true")
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--cpus", default="6-11,50-55")
    ap.add_argument("--only", default="")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--spec", default=str(FAM / "batch.json"))
    a = ap.parse_args()
    specs = json.loads(pathlib.Path(a.spec).read_text())
    names = [name_of(s) for s in specs]
    if a.only:
        keep = [i for i, n in enumerate(names) if any(n.startswith(p) for p in a.only.split(","))]
        specs, names = [specs[i] for i in keep], [names[i] for i in keep]
    cells = a.cells.split(",")
    logs = pathlib.Path(a.out) / "_logs"
    logs.mkdir(parents=True, exist_ok=True)
    if not a.summary_only:
        with cf.ThreadPoolExecutor(min(a.procs, 12)) as ex:
            futs = {ex.submit(run_one, s, c, a): (n, c) for s, n in zip(specs, names) for c in cells}
            for fu in cf.as_completed(futs):
                n, c = futs[fu]
                rc, txt = fu.result()
                (logs / f"{n}__{c}.txt").write_text(txt)
                verdict = [ln for ln in txt.splitlines() if ln.startswith("SMOKE")]
                print(f"{n:42s} {c:20s} rc={rc} {verdict[-1] if verdict else '??'}", flush=True)
    print()
    hdr = ("method", "cell", "n", "err", "B0_err", "aurc", "B0_aurc", "grip", "err_s0", "err_fresh", "err_stale",
           "ms_q", "fit_s")
    print(" ".join(f"{h:>10s}" if i > 1 else f"{h:42s}" if i == 0 else f"{h:20s}" for i, h in enumerate(hdr)))
    for n in names:
        for c in cells:
            r = summarize(n, c, a)
            if r is None:
                print(f"{n:42s} {c:20s} MISSING")
                continue
            vals = [r["n"], r["err"], r["B0_err"], r["aurc"], r["B0_aurc"], r["grip"], r["err_s0"], r["err_fresh"],
                    r["err_stale"], r["ms_q"], r["fit_s"]]
            print(f"{n:42s} {c:20s} " + " ".join(f"{v:10d}" if isinstance(v, int) else f"{v:10.4f}" for v in vals))


if __name__ == "__main__":
    main()
