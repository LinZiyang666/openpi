"""verify_queries.py -- independent spot-check of a query store built by extract_queries.py.

  python verify_queries.py [--out QUERIES_ROOT] [--arms a,b] [--sample 64] [--seed 0]

Per arm: manifest/episodes.json consistency (sha, counts, shapes); for --sample random rows (all rows if
the arm is smaller; tok rows always included in the sample pool) re-reads the source h5 step and compares
every stored field bytewise; for ALL rows checks a_hit == library action[rec_top1] and a_exec == a_hit
(cache) / a_inf (inf); for ALL tok rows checks the 4x4-pooled tokens reproduce key_v0/key_v1 (cos).
Exit 1 on any mismatch.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import hashlib
import json
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_queries import ALL_ARMS, DEFAULT_OUT, IMG_KEYS, LIB_DIR, PERFIELD  # noqa: E402


def _s(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def pool16(v):  # [256, D] -> 4x4 average pool of the 16x16 grid -> [16*D]
    D = v.shape[-1]
    return v.astype(np.float64).reshape(4, 4, 4, 4, D).mean(axis=(1, 3)).reshape(-1)


def verify_arm(d, sample, rng):
    errs = []
    man = json.load(open(f"{d}/manifest.json"))
    eps = json.load(open(f"{d}/episodes.json"))
    if hashlib.sha256(open(f"{d}/episodes.json", "rb").read()).hexdigest() != man["episodes_json_sha256"]:
        errs.append("episodes.json sha256 != manifest")
    A = {n: np.load(f"{d}/{n}", mmap_mode="r") for n in man["arrays"]}
    for n, s in man["arrays"].items():
        if list(A[n].shape) != s["shape"] or A[n].dtype.name != s["dtype"]:
            errs.append(f"{n}: {A[n].shape} {A[n].dtype} != manifest")
    N, M = man["counts"]["rows"], man["counts"]["tok_rows"]
    if eps[-1]["end"] != N or sum(e["num_steps"] for e in eps) != N:
        errs.append("episode ranges do not tile N")
    if [(e["task_id"], e["init"]) for e in eps] != sorted((e["task_id"], e["init"]) for e in eps):
        errs.append("episodes not sorted by (task_id, init)")
    m, sfull = man["model"], man["suite_full"]
    meta = json.load(open(f"{LIB_DIR}/{m}_{sfull}.meta.json"))
    lib_act = np.load(f"{LIB_DIR}/{m}_{sfull}.action.npy", mmap_mode="r")
    top1 = np.asarray(A["rec_top1.npy"])
    a_hit = np.asarray(A["a_hit.npy"])
    if not np.array_equal(np.asarray(lib_act[top1]), a_hit):
        errs.append("a_hit != library action[rec_top1] (all rows)")
    ref = a_hit if man["exec"] == "cache" else np.asarray(A["a_inf.npy"])
    if not np.array_equal(np.asarray(A["a_exec.npy"]), ref):
        errs.append("a_exec != expected source action (all rows)")
    tok_rows = np.asarray(A["tok/rows.npy"])
    kv = {0: A["key_v0.npy"], 1: A["key_v1.npy"]}
    tv = {0: A["tok/v0.npy"], 1: A["tok/v1.npy"]}
    min_cos = 1.0
    for t in range(M):
        r = tok_rows[t]
        for c in (0, 1):
            p = pool16(np.asarray(tv[c][t]))
            k = np.asarray(kv[c][r], np.float64)
            min_cos = min(min_cos, float(p @ k / (np.linalg.norm(p) * np.linalg.norm(k))))
    if M and min_cos < 0.9999:
        errs.append(f"pooled tokens vs keys min cos {min_cos:.6f} < 0.9999")
    tok_pos = {int(r): t for t, r in enumerate(tok_rows)}
    rows = np.arange(N) if N <= sample else np.sort(rng.choice(N, sample, replace=False))
    if M and N > sample:  # make sure some tok rows are in the sample
        rows = np.unique(np.concatenate([rows, rng.choice(tok_rows, min(16, M), replace=False)]))
    ep, step = np.asarray(A["ep.npy"]), np.asarray(A["step.npy"])
    idx = {k: i for i, k in enumerate(meta["ids"])}
    ik0, ik1 = IMG_KEYS[m]
    by_file = {}
    for r in rows:
        by_file.setdefault(int(ep[r]), []).append(int(r))
    for e_i, rs_ in by_file.items():
        e = eps[e_i]
        with h5py.File(e["file"], "r") as f:
            if (str(f.attrs["trace_task_uid"]), int(f.attrs["task_id"]), int(f.attrs["orig_init_state_idx"])) != \
                    (e["uid"], e["task_id"], e["init"]):
                errs.append(f"{e['file']}: attrs != episodes.json")
            for r in rs_:
                if not (e["start"] <= r < e["end"]) or step[r] != r - e["start"]:
                    errs.append(f"row {r}: ep/step inconsistent with episodes.json")
                    continue
                g = f[f"step_{int(step[r]):04d}"]
                pairs = [("key_v0.npy", "trace/query_keys/vision_0"), ("key_v1.npy", "trace/query_keys/vision_1"),
                         ("rs.npy", "trace/query_keys/robot_state"), ("raw_state.npy", "trace/raw_state"),
                         ("a_inf.npy", "trace/actions/full_inference"), ("a_hit.npy", "trace/actions/full_hit"),
                         ("a_exec.npy", "trace/actions/executed")]
                for n, h in pairs:
                    if not np.array_equal(np.asarray(A[n][r]), g[h][()]):
                        errs.append(f"row {r}: {n} != {h}")
                if idx[_s(g["trace/search/real_topk_ids"][()][0])] != top1[r]:
                    errs.append(f"row {r}: rec_top1")
                if A["rec_score.npy"][r] != g["trace/search/real_topk_scores"][()][0]:
                    errs.append(f"row {r}: rec_score")
                pf = np.array([g[f"trace/search/twin_topk_per_field/{fn}"][()][0] for fn in PERFIELD], np.float32)
                if not np.array_equal(np.asarray(A["rec_perfield.npy"][r]), pf):
                    errs.append(f"row {r}: rec_perfield")
                t = tok_pos.get(r)
                if (t is not None) != (e["init"] in man["tok_inits"]):
                    errs.append(f"row {r}: tok membership")
                if t is not None:
                    for n, h in (("tok/v0.npy", "vision_0"), ("tok/v1.npy", "vision_1"),
                                 ("tok/img0.npy", f"trace/raw_images/{ik0}"), ("tok/img1.npy", f"trace/raw_images/{ik1}")):
                        if not np.array_equal(np.asarray(A[n][t]), g[h][()]):
                            errs.append(f"row {r}: {n} != {h}")
    rs_all = np.asarray(A["rs.npy"])
    pad_nz = int((rs_all[:, man["rs_valid_dims"]:] != 0).any(axis=1).sum()) if rs_all.shape[1] > man["rs_valid_dims"] else 0
    return errs, dict(rows=N, tok_rows=M, checked_rows=len(rows), tok_min_cos=round(min_cos, 7),
                      rs_pad_nonzero_rows=pad_nz, success_eps=man["counts"]["success_episodes"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--arms", default=",".join(ALL_ARMS))
    ap.add_argument("--sample", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    bad = 0
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        d = f"{args.out}/{arm}"
        if not os.path.exists(f"{d}/manifest.json"):
            print(f"{arm}: MISSING manifest.json", flush=True)
            bad += 1
            continue
        errs, info = verify_arm(d, args.sample, rng)
        bad += bool(errs)
        print(f"{arm}: {'OK' if not errs else 'FAIL'} {info}" + "".join(f"\n    {e}" for e in errs[:20]), flush=True)
    print("VERIFY", "FAIL" if bad else "OK", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
