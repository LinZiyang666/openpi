"""Summarize the g2_pcacos smoke outputs: err / gripper-sign-faithful err / AURC per regime, ms/query, fit s."""
import glob
import json
import pathlib
import re
import sys

import numpy as np

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                   "/home/weiland/trace_runs/offline_search_store/derived/r02/g2_pcacos/smoke/run")
ROOT = pathlib.Path("/dev/shm/offline_search_store")
_A = {}


def a_inf(cell):
    if cell not in _A:
        _A[cell] = np.load(ROOT / "queries" / cell / "a_inf.npy", mmap_mode="r")
    return _A[cell]


def aurc(conf, err):
    o = np.lexsort((np.arange(conf.size), -conf))
    e = err[o]
    return float((np.cumsum(e) / np.arange(1, e.size + 1)).mean()) if e.size else float("nan")


verdicts = {}
for lg in glob.glob(str(OUT / "logs" / "*.log")):
    t = open(lg).read()
    m = re.search(r"attrs: name=(\S+)", t)
    c = re.search(r"cell=(\S+)", t)
    v = re.search(r"SMOKE (PASS|FAIL)(.*)", t)
    b0 = re.search(r"metrics \[B0_current\] n=(\d+)\s+err_mean=([\d.]+).*?\saurc=([\d.]+)", t)
    if m and c:
        verdicts[(m.group(1), c.group(1))] = (v.group(1) + v.group(2).strip() if v else "??",
                                              (float(b0.group(2)), float(b0.group(3))) if b0 else None)
rows = []
for cj in sorted(glob.glob(str(OUT / "*" / "*" / "*.json"))):
    p = pathlib.Path(cj)
    if p.name in ("summary.json", "run_meta.json") or p.parent.name == "B0_current":
        continue
    cell = p.stem
    js = json.loads(p.read_text())
    z = np.load(p.with_suffix(".npz"))
    st = z["step"]
    seg = z["synth_seg"].astype(np.float64)
    sig = np.asarray(js["sigma"], np.float64) if "sigma" in js else None
    ai = np.asarray(a_inf(cell)[z["row"], :5, :7], np.float64)
    snap = seg.copy()
    snap[:, :, 6] = np.where(snap[:, :, 6] >= 0, 1.0, -1.0)
    esnap = np.sqrt((((snap - ai) / sig) ** 2).mean(axis=(1, 2)))
    err = z["err"]
    conf = z["confidence"]
    stale_or_fresh = st >= 1
    name = p.parent.name
    ver, b0 = verdicts.get((name, cell), ("?", None))
    rows.append({"method": name, "cell": cell, "n": int(err.size), "err": float(err.mean()), "err_snap": float(esnap.mean()),
                 "aurc": aurc(conf, err), "err_s0": float(err[st == 0].mean()), "err_s1": float(err[stale_or_fresh].mean()),
                 "aurc_s1": aurc(conf[stale_or_fresh], err[stale_or_fresh]),
                 "ms": js["timing"]["ms_per_query"], "fit_s": js["fit"]["fit_s"], "bpe": js["fit"]["bytes_per_entry"],
                 "b0_err": b0[0] if b0 else float("nan"), "b0_aurc": b0[1] if b0 else float("nan"), "smoke": ver})
rows.sort(key=lambda r: (r["cell"], r["method"]))
print("| method | cell | n | err | err_snap | AURC | err step0 | err step>=1 | AURC step>=1 | B0 err / AURC (same eps) | ms/q | fit s | B/entry | smoke |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['method']} | {r['cell']} | {r['n']} | {r['err']:.3f} | {r['err_snap']:.3f} | {r['aurc']:.3f} | {r['err_s0']:.3f} | "
          f"{r['err_s1']:.3f} | {r['aurc_s1']:.3f} | {r['b0_err']:.3f} / {r['b0_aurc']:.3f} | {r['ms']:.2f} | {r['fit_s']:.1f} | "
          f"{r['bpe']:g} | {r['smoke']} |")
json.dump(rows, open(OUT / "smoke_table.json", "w"), indent=1)
