"""Per-regime summary of harness outputs (smoke / run dirs): err, gripper-sign-faithful err, grip_mis, AURC within the
regime (method's own confidence), oracle. Analysis-side tool (reads a_inf from the store; not part of any method).

    python exp/offline_search/rounds/r02/g1_awm/_regime_table.py <out_dir> [<out_dir> ...] [--json out.json]

<out_dir> holds <method>/<cell>.npz (smoke --out dir or run --out dir). Regime = x_regime extra when present, else
step 0 -> step0, inf cell -> fresh, cache cell -> stale.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path("/dev/shm/offline_search_store")


def sigma(ms):
    a = np.load(ROOT / "library" / ms / "current" / "action.npy", mmap_mode="r")
    return np.asarray(a[:, :5, :7], np.float32).reshape(-1, 7).std(axis=0)


def aurc(conf, err):
    o = np.argsort(-conf, kind="stable")
    e = err[o]
    return float((np.cumsum(e) / np.arange(1, len(e) + 1)).mean())


def summarize(npz_path):
    cell = npz_path.stem
    ms, arm = cell.rsplit("_", 1)
    z = np.load(npz_path)
    rows = z["row"]
    step = z["step"].astype(int)
    if "x_regime" in z.files:
        reg = z["x_regime"].astype(int)
    else:
        reg = np.where(step == 0, 0, 1 if arm == "inf" else 2)
    err = z["err"].astype(np.float64)
    ainf = np.asarray(np.load(ROOT / "queries" / cell / "a_inf.npy", mmap_mode="r")[np.sort(rows)][:, :5, :7], np.float32)
    ainf = ainf[np.argsort(np.argsort(rows))]
    sig = sigma(ms)
    if bool(z["used_synth"].any()):
        seg = z["synth_seg"].astype(np.float32).copy()
        seg[..., 6] = np.where(seg[..., 6] >= 0, 1.0, -1.0)
        err_gs = np.sqrt(np.mean(((seg - ainf) / sig) ** 2, axis=(1, 2)))
        err_gs = np.where(z["used_synth"], err_gs, err)
    else:
        err_gs = err
    conf = z["confidence"].astype(np.float64)
    out = {"cell": cell, "n": int(len(err)), "err": float(err.mean()), "err_gs": float(err_gs.mean()),
           "aurc": aurc(conf, err), "ms_q_worker": float(np.median(z["t_query_us"]) / 1e3) if "t_query_us" in z.files else None}
    names = {0: "step0", 1: "fresh", 2: "stale"}
    for r, nm in names.items():
        m = reg == r
        if m.sum() == 0:
            continue
        out[nm] = {"n": int(m.sum()), "err": float(err[m].mean()), "err_gs": float(err_gs[m].mean()),
                   "grip_mis": float(z["grip_mis"][m].mean()), "aurc": aurc(conf[m], err[m]),
                   "oracle_cur": float(np.nanmean(z["oracle_err"][m]))}
    return out


def main():
    args = sys.argv[1:]
    jout = None
    if "--json" in args:
        i = args.index("--json")
        jout = args[i + 1]
        args = args[:i] + args[i + 2:]
    res = []
    for d in args:
        for p in sorted(pathlib.Path(d).glob("*/*.npz")):
            s = summarize(p)
            s["method"] = p.parent.name
            res.append(s)
    for s in res:
        line = f"{s['method']:<34s} {s['cell']:<20s} n={s['n']:<5d} all err {s['err']:.3f} gs {s['err_gs']:.3f} aurc {s['aurc']:.3f}"
        for nm in ("step0", "fresh", "stale"):
            if nm in s:
                r = s[nm]
                line += f" | {nm} n={r['n']} err {r['err']:.3f} gs {r['err_gs']:.3f} grip {r['grip_mis']:.3f} aurc {r['aurc']:.3f}"
        print(line)
    if jout:
        pathlib.Path(jout).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
