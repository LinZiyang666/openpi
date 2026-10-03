"""pixel_diff.py -- for inf/cache pairs whose step-0 frames differ: how they differ."""
import glob, json, h5py, numpy as np, collections
R = "/home/weiland/trace_runs/dual_20260923/runs"
def index(arm):
    out = {}
    for p in glob.glob(f"{R}/{arm}/trace/**/*.h5", recursive=True):
        with h5py.File(p, "r") as f:
            out[":".join(str(f.attrs["trace_task_uid"]).split(":")[2:])] = p
    return out
res = {}
for m in ("pi05", "groot"):
    for s in ("sp", "l10"):
        A, B = index(f"tr_{m}_{s}_inf"), index(f"tr_{m}_{s}_cache")
        rows = []
        for k in sorted(set(A) & set(B)):
            with h5py.File(A[k], "r") as fa, h5py.File(B[k], "r") as fb:
                ra, rb = fa["step_0000/trace/raw_images"], fb["step_0000/trace/raw_images"]
                for cam in ra:
                    x = ra[cam][()].astype(np.int16); y = rb[cam][()].astype(np.int16)
                    if np.array_equal(x, y): continue
                    d = np.abs(x - y).max(axis=-1)
                    ys, xs = np.nonzero(d)
                    rows.append(dict(key=k, cam=cam, frac_px=float((d > 0).mean()), max_diff=int(d.max()),
                                     frac_px_gt8=float((d > 8).mean()),
                                     bbox=[int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())]))
        res[f"{m}_{s}"] = rows
        print(f"{m}_{s}: {len(rows)} differing camera frames")
        for r in rows[:12]:
            print("   ", r)
json.dump(res, open("/home/weiland/trace_runs/dual_20260923/audit/pixel_diff.json", "w"), indent=1)
