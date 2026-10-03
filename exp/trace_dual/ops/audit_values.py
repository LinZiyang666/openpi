"""audit_values.py [arm ...] -- per-step value audit of every trace file.

For every step of every file: finiteness, clean_action == full_inference, noise chain
converging to the final action, raw/model/input image agreement, query keys recomputed
from the stored embeddings (4x4 pool of the 16x16 token grid), real vs twin top-1,
per-field similarities recomputed from the query keys and the library vectors, fused
score recomputed from the per-field values, full_hit == library action_chunk[top-1],
top-1 task_key vs the episode task, plus duplicate detection (stale frames, reused
noise) and episode-length structure. Writes audit/values_<arm>.json.
"""
import sys, glob, json, hashlib, collections, math, pathlib
import numpy as np, h5py, yaml
from concurrent.futures import ProcessPoolExecutor

R = pathlib.Path("/home/weiland/trace_runs/dual_20260923")
LIB = R / "audit" / "libs"
ARMS = ["tr_pi05_sp_inf", "tr_pi05_sp_cache", "tr_pi05_l10_inf", "tr_pi05_l10_cache",
        "tr_groot_sp_inf", "tr_groot_sp_cache", "tr_groot_l10_inf", "tr_groot_l10_cache"]
MAXDEC = {"libero_spatial": 44, "libero_10": 104}

_G = {}


def _init(arm):
    model = "pi05" if "pi05" in arm else "groot"
    suite = "libero_spatial" if "_sp_" in arm else "libero_10"
    name = f"{model}_{suite}"
    cfg = yaml.safe_load(open(R / "config" / f"{arm}.yaml"))
    cp1 = cfg["checkpoints"]["cp1"]["search_strategy"]
    norm = cp1["score_normalization"]["fields"]
    _G.update(
        arm=arm, model=model, suite=suite,
        w={f: cfg["keys"][f]["weight"] for f in ("vision_0", "vision_1", "robot_state")},
        mu={f: norm[f]["params"]["mu"] for f in norm}, sigma={f: norm[f]["params"]["sigma"] for f in norm},
        tau=cp1["field_similarity"]["robot_state"]["to_similarity"]["tau"],
        lib={f: np.load(LIB / f"{name}.{f}.npy", mmap_mode="r") for f in ("vision_0", "vision_1", "robot_state", "action")},
        meta=json.load(open(LIB / f"{name}.meta.json")),
    )
    _G["idx"] = {i: k for k, i in enumerate(_G["meta"]["ids"])}
    _G["expect"] = "full_hit" if arm.endswith("_cache") else "full_inference"


def _pool16(v):  # [256, D] -> 4x4 adaptive avg pool of the 16x16 grid -> [16*D]
    D = v.shape[-1]
    return v.astype(np.float64).reshape(4, 4, 4, 4, D).mean(axis=(1, 3)).reshape(-1)


def _cos(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-8))


def _norm(f, raw):  # ZScoreNormalizer: l2 oriented as -distance; 0.5*(tanh(z)+1)
    return 0.5 * (math.tanh((raw - _G["mu"][f]) / _G["sigma"][f]) + 1.0)


def _h(x):
    return hashlib.blake2b(np.ascontiguousarray(x).tobytes(), digest_size=12).hexdigest()


def audit_file(p):
    c = collections.Counter(); mx = collections.defaultdict(float); acc = collections.defaultdict(list)
    noise_hashes = []; step0 = {}
    # driver="core": one large sequential read of the whole file, then parse in memory --
    # per-dataset preads from 32 workers degrade into ~90 KB random reads (~240 MB/s on this SSD)
    with h5py.File(p, "r", driver="core", backing_store=False) as f:
        a = f.attrs
        steps = sorted(k for k in f if k.startswith("step_"))
        n = len(steps); succ = bool(a["success"]); task = str(a["task"]); uid = str(a["trace_task_uid"])
        c["steps"] = n
        if n != int(a["num_steps"]): c["num_steps_mismatch"] += 1
        if [int(s[5:]) for s in steps] != list(range(n)): c["step_index_gap"] += 1
        if not succ and n != MAXDEC[_G["suite"]]: c["fail_not_at_max"] += 1
        if succ and n >= MAXDEC[_G["suite"]]: c["success_at_max"] += 1
        prev_img = None; tok0 = None; v2_0 = None
        for si, k in enumerate(steps):
            g = f[k]; t = g["trace"]; ta = t.attrs
            # 1. finiteness over every float dataset of the step
            def fin(name, o):
                if isinstance(o, h5py.Dataset) and o.dtype.kind == "f":
                    if not np.all(np.isfinite(o[()])): c["nonfinite_" + name.split("/")[-1]] += 1
            g.visititems(fin)
            clean = g["clean_action"][()]; fi = t["actions/full_inference"][()]
            fh = t["actions/full_hit"][()]; ex = t["actions/executed"][()]
            if not np.array_equal(clean, fi): c["clean_ne_full_inference"] += 1
            if not np.array_equal(ex, fh if _G["expect"] == "full_hit" else fi): c["executed_wrong"] += 1
            # 2. noise chain: x0 ~ N(0,1), distance to the final action shrinks every step
            nk = sorted((kk for kk in g if kk.startswith("noise_action_")), key=lambda s: int(s.rsplit("_", 1)[1]))
            c["noise_len_%d" % len(nk)] += 1 if si == 0 else 0
            x0 = g[nk[0]][()]
            acc["x0_mean"].append(float(x0.mean())); acc["x0_std"].append(float(x0.std()))
            d = [float(np.linalg.norm(g[kk][()] - clean)) for kk in nk]
            if any(d[i + 1] > d[i] + 1e-4 for i in range(len(d) - 1)): c["noise_chain_nonmonotone"] += 1
            acc["last_noise_to_clean_rel"].append(d[-1] / max(d[0], 1e-8))
            noise_hashes.append(_h(x0))
            # 3. images: raw == model/input for pi05; blank; stale consecutive frames
            raw = t["raw_images"]; rk = sorted(raw)
            img = raw[rk[0]][()]
            if img.std() < 2.0: c["blank_image"] += 1
            hi = _h(img)
            if prev_img is not None and hi == prev_img: c["stale_frame_consecutive"] += 1
            prev_img = hi
            if _G["model"] == "pi05":
                if not np.array_equal(raw["observation%2Fimage"][()], t["model_images/base_0_rgb"][()]): c["raw_ne_model_image"] += 1
                if not np.array_equal(g["input_images/base_0_rgb"][()], t["model_images/base_0_rgb"][()]): c["input_ne_model_image"] += 1
                if t["model_images/right_wrist_0_rgb"][()].any(): c["right_wrist_not_empty"] += 1
                v2 = g["vision_2"][()]
                if v2_0 is None: v2_0 = v2
                else: mx["vision2_drift_rel"] = max(mx["vision2_drift_rel"], float(np.abs(v2.astype(np.float32) - v2_0.astype(np.float32)).max() / (np.abs(v2_0.astype(np.float32)).max() + 1e-6)))
            tok = t["tokenized_prompt"][()]
            if tok0 is None: tok0 = tok
            elif not np.array_equal(tok, tok0): c["prompt_tokens_change"] += 1
            if str(ta["prompt"]) != task: c["prompt_ne_task"] += 1
            # 4. query keys recomputed from stored embeddings / state
            q = {fn: t["query_keys/" + fn][()] for fn in ("vision_0", "vision_1", "robot_state")}
            for fn in ("vision_0", "vision_1"):
                rec = _pool16(g[fn][()])
                acc["qkey_cos_" + fn].append(_cos(rec, q[fn]))
                mx["qkey_relerr_" + fn] = max(mx["qkey_relerr_" + fn], float(np.abs(rec - q[fn]).max() / (np.abs(q[fn]).max() + 1e-8)))
            rs = g["robot_state"][()]
            if not np.array_equal(rs[: q["robot_state"].shape[0]], q["robot_state"]): c["qkey_robot_state_ne"] += 1
            # 5. search: real vs twin top-1, recompute per-field and fused score
            rid = t["search/real_topk_ids"][()][0]; rid = rid.decode() if isinstance(rid, bytes) else str(rid)
            tid = t["search/twin_topk_ids"][()][0]; tid = tid.decode() if isinstance(tid, bytes) else str(tid)
            rsc = float(t["search/real_topk_scores"][()][0]); tsc = float(t["search/twin_topk_scores"][()][0])
            if rid != tid: c["real_twin_top1_differ"] += 1
            mx["real_twin_score_absdiff"] = max(mx["real_twin_score_absdiff"], abs(rsc - tsc))
            if abs(float(ta["score"]) - rsc) > 1e-5: c["attr_score_ne_real"] += 1
            j = _G["idx"].get(rid)
            if j is None:
                c["top1_not_in_library"] += 1
            else:
                L = _G["lib"]
                pf = {fn: float(t["search/twin_topk_per_field/" + fn][()][0]) for fn in ("vision_0", "vision_1", "robot_state")}
                cos0 = _cos(q["vision_0"], L["vision_0"][j]); cos1 = _cos(q["vision_1"], L["vision_1"][j])
                dist = float(np.linalg.norm(q["robot_state"].astype(np.float64) - L["robot_state"][j][: q["robot_state"].shape[0]]))
                # the recorded per-field values are the normalized scores 0.5*(tanh(z)+1)
                mx["perfield_recompute_absdiff_v0"] = max(mx["perfield_recompute_absdiff_v0"], abs(_norm("vision_0", cos0) - pf["vision_0"]))
                mx["perfield_recompute_absdiff_v1"] = max(mx["perfield_recompute_absdiff_v1"], abs(_norm("vision_1", cos1) - pf["vision_1"]))
                mx["perfield_recompute_absdiff_rs"] = max(mx["perfield_recompute_absdiff_rs"], abs(_norm("robot_state", -dist) - pf["robot_state"]))
                fused = (_G["w"]["vision_0"] * _norm("vision_0", cos0) + _G["w"]["vision_1"] * _norm("vision_1", cos1)
                         + _G["w"]["robot_state"] * _norm("robot_state", -dist))
                mx["fused_recompute_absdiff"] = max(mx["fused_recompute_absdiff"], abs(fused - rsc))
                if not np.array_equal(fh, L["action"][j]): c["full_hit_ne_library_action"] += 1
                if _G["meta"]["task_key"][j] != task: c["top1_task_mismatch"] += 1
                acc["top1_score"].append(rsc); acc["cos_v0"].append(cos0); acc["cos_v1"].append(cos1)
            acc["l2_hit_vs_inf"].append(float(np.linalg.norm(fh - fi)))
            acc["wall_ms"].append(json.loads(ta["timing_json"]).get("wall_ms", float("nan")))
            if si == 0:
                step0 = dict(img=hi, top1=rid)
    return dict(file=p, uid=uid, succ=succ, n=n, counts=dict(c), maxes=dict(mx),
                acc={k: [float(np.min(v)), float(np.mean(v)), float(np.max(v)), len(v)] for k, v in acc.items()},
                noise_hashes=noise_hashes, step0=step0)


def main():
    arms = sys.argv[1:] or ARMS
    out_dir = R / "audit"; out_dir.mkdir(exist_ok=True)
    for arm in arms:
        files = sorted(glob.glob(f"{R}/runs/{arm}/trace/**/*.h5", recursive=True))
        with ProcessPoolExecutor(32, initializer=_init, initargs=(arm,)) as ex:
            res = list(ex.map(audit_file, files, chunksize=4))
        tot = collections.Counter(); mxs = collections.defaultdict(float); agg = collections.defaultdict(list)
        nh = collections.Counter()
        for r in res:
            tot.update(r["counts"])
            for k, v in r["maxes"].items(): mxs[k] = max(mxs[k], v)
            for k, (lo, mean, hi, cnt) in r["acc"].items(): agg[k].append((lo, mean, hi, cnt))
            nh.update(r["noise_hashes"])
        summ = {k: [min(x[0] for x in v), sum(x[1] * x[3] for x in v) / sum(x[3] for x in v), max(x[2] for x in v)] for k, v in agg.items()}
        report = dict(arm=arm, files=len(res), counts=dict(tot), maxes=dict(mxs), min_mean_max=summ,
                      noise_x0_duplicate_hashes=sum(1 for v in nh.values() if v > 1),
                      step0={r["uid"]: r["step0"] for r in res})
        (out_dir / f"values_{arm}.json").write_text(json.dumps(report, indent=1))
        print(json.dumps({k: v for k, v in report.items() if k != "step0"}), flush=True)


if __name__ == "__main__":
    main()
