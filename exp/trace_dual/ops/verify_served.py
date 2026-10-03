"""verify_served.py: every step of every trace file served the arm its group defines (inf -> full_inference, cache -> full_hit)."""
import glob, json, collections, h5py, sys
from concurrent.futures import ProcessPoolExecutor
B = "/home/weiland/trace_runs/dual_20260923/runs"
ARMS = ["tr_pi05_sp_inf", "tr_pi05_sp_cache", "tr_pi05_l10_inf", "tr_pi05_l10_cache",
        "tr_groot_sp_inf", "tr_groot_sp_cache", "tr_groot_l10_inf", "tr_groot_l10_cache"]
def one(p):
    c = collections.Counter(); ht = collections.Counter()
    with h5py.File(p, "r") as f:
        for k in f:
            if k.startswith("step_"):
                a = f[k]["trace"].attrs
                c[str(a.get("executed_arm"))] += 1; ht[str(a.get("hit_type"))] += 1
    return c, ht
out = {}
for arm in ARMS:
    files = sorted(glob.glob(f"{B}/{arm}/trace/**/*.h5", recursive=True))
    tot = collections.Counter(); hts = collections.Counter()
    with ProcessPoolExecutor(24) as ex:
        for c, ht in ex.map(one, files, chunksize=8):
            tot.update(c); hts.update(ht)
    want = "full_hit" if arm.endswith("_cache") else "full_inference"
    out[arm] = {"files": len(files), "executed_arm": dict(tot), "hit_type": dict(hts), "all_expected": set(tot) == {want}}
    print(arm, out[arm], flush=True)
json.dump(out, open(f"{B}/served_check.json", "w"), indent=2)
