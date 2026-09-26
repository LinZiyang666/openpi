"""floor_step0.py -- teacher noise floor at step 0 from the inf/cache init pairs (CPU only).

For each model x suite, the inf and cache arms start from the same A-pool init
(task_id, orig_init_state_idx): step_0000 raw_state is bit-identical and ~98% of first
frames are byte-identical, while the flow-matching noise is independent. So the two
recorded trace/actions/full_inference at step 0 are two teacher samples on (nearly)
the same observation. Writes <out>/<m>_<s>/step0_pairs.npz and step0_summary.json.

Only step_0000 is touched, so the default h5py driver is used (~15 ms/file); the
driver="core" advice is for full-file scans, where it reads the whole 90-220 MB file.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import h5py  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import floor_common as fc  # noqa: E402


def read_step0(path: str) -> dict:
    with h5py.File(path, "r") as f:
        t = f["step_0000/trace"]
        return dict(
            task_id=int(f.attrs["task_id"]), init=int(f.attrs["orig_init_state_idx"]),
            a=t["actions/full_inference"][()], raw_state=t["raw_state"][()],
            imgs={k: t["raw_images/" + k][()] for k in t["raw_images"]},
        )


def pair(args):
    pi, pc = args
    a, b = read_step0(pi), read_step0(pc)
    assert (a["task_id"], a["init"]) == (b["task_id"], b["init"]), (pi, pc)
    assert set(a["imgs"]) == set(b["imgs"]) and len(a["imgs"]) == 2, (pi, sorted(a["imgs"]))
    return dict(
        task_id=a["task_id"], init=a["init"], a_inf=a["a"], a_cache=b["a"],
        images_identical=all(np.array_equal(a["imgs"][k], b["imgs"][k]) for k in a["imgs"]),
        raw_state_equal=bool(np.array_equal(a["raw_state"], b["raw_state"])),
        file_inf=pi, file_cache=pc,
    )


def key_of(path: str):
    r = fc.read_attrs(path)
    return (r["task_id"], r["init"]), path


def run_cell(m: str, s: str, out_root: pathlib.Path, ex: ProcessPoolExecutor, limit: int | None) -> dict:
    t0 = time.time()
    idx = {}
    for a in ("inf", "cache"):
        files = fc.arm_files(m, s, a)
        d = dict(ex.map(key_of, files, chunksize=16))
        assert len(d) == len(files), f"{m}_{s}_{a}: duplicate (task, init) keys"
        idx[a] = d
    keys = sorted(set(idx["inf"]) & set(idx["cache"]))
    unmatched = len(set(idx["inf"]) ^ set(idx["cache"]))
    if limit:
        keys = keys[:limit]
    rows = list(ex.map(pair, [(idx["inf"][k], idx["cache"][k]) for k in keys], chunksize=8))
    sigma = fc.lib_sigma(m, s)
    A = np.stack([r["a_inf"] for r in rows]).astype(np.float32)
    B = np.stack([r["a_cache"] for r in rows]).astype(np.float32)
    ident = np.array([r["images_identical"] for r in rows])
    rs_eq = np.array([r["raw_state_equal"] for r in rows])
    e = fc.err(A, B, sigma); g = fc.grip_mis(A, B); r = fc.rel_l2(B, A)  # a_inf as reference
    task = np.array([r_["task_id"] for r_ in rows], np.int32)
    cell = out_root / f"{m}_{s}"
    cell.mkdir(parents=True, exist_ok=True)
    np.savez(cell / "step0_pairs.npz",
             task_id=task, init=np.array([r_["init"] for r_ in rows], np.int32),
             a_inf_arm=A, a_cache_arm=B, images_identical=ident, err=e, grip_mis=g, rel_l2=r,
             raw_state_equal=rs_eq, sigma=sigma,
             file_inf=np.array([r_["file_inf"] for r_ in rows]), file_cache=np.array([r_["file_cache"] for r_ in rows]))
    summ = {
        "model": m, "suite": s, "pairs": len(rows), "unmatched_keys": unmatched,
        "images_identical": int(ident.sum()), "raw_state_equal": int(rs_eq.sum()), "sigma": sigma.tolist(),
        "metric": "err = sqrt(mean_{t<5,d<7}(((a_inf-a_cache)/sigma_d)^2)); rel_l2 = ||a_cache-a_inf||/||a_inf|| on [:5,:7]; "
                  "grip_mis = fraction of t<5 with sign(dim6) disagreeing; only dims 0..6 are valid (7..31 padding)",
        "all": fc.pair_stats(e, g, r),
        "images_identical_subset": fc.pair_stats(e[ident], g[ident], r[ident]),
        "images_differ_subset": fc.pair_stats(e[~ident], g[~ident], r[~ident]),
        "secondary_rel_l2_chunk7": fc.qstats(fc.rel_l2_chunk7(B, A)),   # whole horizon, valid dims [:, :7]
        "by_task_err_p50": {int(t): float(np.median(e[task == t])) for t in np.unique(task)},
        "seconds": round(time.time() - t0, 1),
    }
    fc.save_json(cell / "step0_summary.json", summ)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(fc.STORE))
    ap.add_argument("--models", default="pi05,groot")
    ap.add_argument("--suites", default="spatial,l10")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="first N pairs only (smoke)")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    with ProcessPoolExecutor(min(args.procs, 16)) as ex:
        for m in args.models.split(","):
            for s in args.suites.split(","):
                su = run_cell(m, s, out, ex, args.limit)
                a, i = su["all"], su["images_identical_subset"]
                print(f"{m}_{s}: pairs={su['pairs']} ident={su['images_identical']} rs_eq={su['raw_state_equal']} "
                      f"err p10/p50/p90={a['err']['p10']:.3f}/{a['err']['p50']:.3f}/{a['err']['p90']:.3f} "
                      f"grip={a['grip_mis_rate']:.4f} relL2 p50={a['rel_l2']['p50']:.3f} | "
                      f"ident-subset err p50={i['err']['p50']:.3f} grip={i['grip_mis_rate']:.4f} ({su['seconds']}s)",
                      flush=True)


if __name__ == "__main__":
    main()
