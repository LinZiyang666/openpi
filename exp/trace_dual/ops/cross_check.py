"""cross_check.py -- step-0 consistency across groups that share an init state.

Same model, inf vs cache (same task_id / episode_idx => same A-pool init): the first
observation, its embeddings, query keys, top-1 and cached action must agree; the
full-inference noise must differ (independent draws). Across models (pi0.5 vs GR00T,
same suite, same init): the initial raw_state must be identical. Writes audit/cross.json.
"""
import glob, json, collections, pathlib
import numpy as np, h5py
from concurrent.futures import ProcessPoolExecutor

R = pathlib.Path("/home/weiland/trace_runs/dual_20260923")


def index(arm):
    out = {}
    for p in glob.glob(f"{R}/runs/{arm}/trace/**/*.h5", recursive=True):
        with h5py.File(p, "r") as f:
            uid = str(f.attrs["trace_task_uid"])
        out[":".join(uid.split(":")[2:])] = p  # task:ep
    return out


def s0(p):
    with h5py.File(p, "r") as f:
        g = f["step_0000"]; t = g["trace"]
        return dict(
            raw={k: t["raw_images/" + k][()] for k in t["raw_images"]},
            raw_state=t["raw_state"][()],
            v0=g["vision_0"][()].astype(np.float32), v1=g["vision_1"][()].astype(np.float32),
            q0=t["query_keys/vision_0"][()], q1=t["query_keys/vision_1"][()], qrs=t["query_keys/robot_state"][()],
            top1=t["search/real_topk_ids"][()][0], score=float(t["search/real_topk_scores"][()][0]),
            fh=t["actions/full_hit"][()], fi=t["actions/full_inference"][()], n0=g["noise_action_0"][()],
            prompt=str(t.attrs["prompt"]),
        )


def cos(a, b):
    a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def pair(args):
    pa, pb = args
    a, b = s0(pa), s0(pb)
    return dict(
        raw_equal=all(np.array_equal(a["raw"][k], b["raw"][k]) for k in a["raw"]),
        raw_state_equal=bool(np.array_equal(a["raw_state"], b["raw_state"])),
        v0_cos=cos(a["v0"], b["v0"]), v0_maxrel=float(np.abs(a["v0"] - b["v0"]).max() / (np.abs(a["v0"]).max() + 1e-6)),
        q0_cos=cos(a["q0"], b["q0"]), q1_cos=cos(a["q1"], b["q1"]), qrs_equal=bool(np.array_equal(a["qrs"], b["qrs"])),
        top1_same=a["top1"] == b["top1"], score_absdiff=abs(a["score"] - b["score"]),
        full_hit_equal=bool(np.array_equal(a["fh"], b["fh"])),
        noise0_equal=bool(np.array_equal(a["n0"], b["n0"])),
        fi_l2=float(np.linalg.norm(a["fi"] - b["fi"])), prompt_equal=a["prompt"] == b["prompt"],
    )


def model_pair(args):
    pa, pb = args
    with h5py.File(pa, "r") as fa, h5py.File(pb, "r") as fb:
        ra = fa["step_0000/trace/raw_state"][()]; rb = fb["step_0000/trace/raw_state"][()]
        return dict(raw_state_equal=bool(np.array_equal(ra, rb)), raw_state_maxabs=float(np.abs(ra - rb).max()),
                    task_equal=str(fa.attrs["task"]) == str(fb.attrs["task"]))


def summarize(rows):
    out = {}
    for k in rows[0]:
        v = [r[k] for r in rows]
        if isinstance(v[0], bool):
            out[k] = f"{sum(v)}/{len(v)}"
        else:
            out[k] = [float(np.min(v)), float(np.mean(v)), float(np.max(v))]
    return out


def main():
    res = {}
    idx = {arm: index(arm) for arm in [f"tr_{m}_{s}_{x}" for m in ("pi05", "groot") for s in ("sp", "l10") for x in ("inf", "cache")]}
    with ProcessPoolExecutor(16) as ex:
        for m in ("pi05", "groot"):
            for s in ("sp", "l10"):
                A, B = idx[f"tr_{m}_{s}_inf"], idx[f"tr_{m}_{s}_cache"]
                keys = sorted(set(A) & set(B))
                rows = list(ex.map(pair, [(A[k], B[k]) for k in keys], chunksize=8))
                res[f"{m}_{s}: inf vs cache"] = dict(pairs=len(keys), **summarize(rows))
                print(f"{m}_{s}", json.dumps(res[f"{m}_{s}: inf vs cache"]), flush=True)
        for s in ("sp", "l10"):
            A, B = idx[f"tr_pi05_{s}_inf"], idx[f"tr_groot_{s}_inf"]
            keys = sorted(set(A) & set(B))
            rows = list(ex.map(model_pair, [(A[k], B[k]) for k in keys], chunksize=8))
            res[f"{s}: pi05 vs groot"] = dict(pairs=len(keys), **summarize(rows))
            print(f"{s} pi05 vs groot", json.dumps(res[f"{s}: pi05 vs groot"]), flush=True)
    (R / "audit" / "cross.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
