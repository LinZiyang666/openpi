"""extract_queries.py -- trace h5 files -> compact memmap "query store" (offline_search R0-A).

Layout written per arm (other agents code against it; see logs/offline_search_exploration.log.md §5.1):

  <out>/<m>_<s>_<a>/            m in {pi05, groot}, s in {spatial, l10}, a in {inf, cache}
    episodes.json               [{uid, file, task, task_id, init, success, num_steps, start, end}], sorted by
                                (task_id, init); rows [start, end) of an episode are ordered by step
    ep.npy int32[N]             episode index (position in episodes.json)
    step.npy int16[N]           decision index inside the episode
    key_v0.npy / key_v1.npy     float32[N, 32768]  trace/query_keys/vision_{0,1}
    rs.npy float32[N, Drs]      trace/query_keys/robot_state (Drs = 32 pi05, 8 groot)
    raw_state.npy float32[N, 8] trace/raw_state
    a_inf / a_hit / a_exec.npy  float32[N, H, 32]  trace/actions/{full_inference, full_hit, executed}
                                (H = 10 pi05, 16 groot; model-normalized action space)
    rec_top1.npy int32[N]       library row of trace/search/real_topk_ids[0] (row i = meta["ids"][i])
    rec_score.npy float32[N]    trace/search/real_topk_scores[0]
    rec_perfield.npy f32[N, 3]  trace/search/twin_topk_per_field/{vision_0, vision_1, robot_state}[0]
                                (normalized per-field scores 0.5*(tanh(z)+1), l2 field as -distance)
    tok/rows.npy int64[M]       rows (into N) of the token subsample: every row of the episodes with
                                init in {0, 10, 20, 30, 40}
    tok/v0.npy, tok/v1.npy      float16[M, 256, 2048]  vision_0 / vision_1 tokens
    tok/img0.npy, tok/img1.npy  uint8[M, h, w, 3]  raw agentview ("image") / wrist image
    manifest.json               written LAST; its presence marks the arm complete

Two passes: pass 1 reads only file attrs / metadata (default h5 driver) to validate and assign row
ranges; pass 2 fills the preallocated memmaps from a process pool, one job per file, reading each file
with driver="core" (one big sequential read; the default driver degrades to ~90 KB random reads).
Any integrity failure raises -> non-zero exit (the launcher turns that into an ERROR marker).

Usage (see run_extract_queries.sh for the env-setting launcher):
  python extract_queries.py [--arms pi05_spatial_inf,...] [--limit-episodes K] [--out DIR]
                            [--workers 48] [--driver core|default|auto] [--force] [--plan-only]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import datetime as dt
import glob
import hashlib
import json
import socket
import sys
import time
import traceback
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait

import h5py
import numpy as np

TRACE_ROOT = "/home/weiland/trace_runs/dual_20260923"
LIB_DIR = f"{TRACE_ROOT}/audit/libs"
DEFAULT_OUT = "/home/weiland/trace_runs/offline_search_store/queries"
MODELS = ("pi05", "groot")
SUITES = {"spatial": ("sp", "libero_spatial"), "l10": ("l10", "libero_10")}
EXECS = ("inf", "cache")
ALL_ARMS = [f"{m}_{s}_{a}" for m in MODELS for s in SUITES for a in EXECS]
TOK_INITS = (0, 10, 20, 30, 40)
KEY_DIM = 32768
N_ACT_DIM = 32
TOK_SHAPE = (256, 2048)
IMG_KEYS = {  # (img0 = agentview "image", img1 = wrist) under trace/raw_images/
    "pi05": ("observation%2Fimage", "observation%2Fwrist_image"),
    "groot": ("video.image", "video.wrist_image"),
}
PERFIELD = ("vision_0", "vision_1", "robot_state")
# valid-dim facts (owner): action chunks (H, 32) carry only dims 0..6 (dim 6 = gripper; pi05 7..31 ~0.001 padding,
# groot 7..31 noise-like padding) and only the first 5 steps are executed (replan 5); robot_state is 32-d for pi05
# with only dims 0..7 valid (the rest exactly 0) and 8-d, all valid, for groot. Full raw arrays are stored anyway.
ACT_VALID_DIMS, EXEC_STEPS, RS_VALID_DIMS, GRIPPER_DIM = 7, 5, 8, 6
SCHEMA = "offline_search.queries.v1"

try:
    from zoneinfo import ZoneInfo

    _CDT = ZoneInfo("America/Chicago")
except Exception:  # pragma: no cover
    _CDT = None


def _now():
    u = dt.datetime.now(dt.timezone.utc)
    return {"utc": u.isoformat(timespec="seconds"),
            "local": u.astimezone(_CDT).isoformat(timespec="seconds") if _CDT else None}


def log(msg):
    t = dt.datetime.now(_CDT).strftime("%H:%M:%S") if _CDT else time.strftime("%H:%M:%S")
    print(f"[{t}] {msg}", flush=True)


class IntegrityError(RuntimeError):
    pass


def check(cond, msg):
    if not cond:
        raise IntegrityError(msg)


def sha256_file(p, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _s(x):
    return x.decode() if isinstance(x, bytes) else str(x)


# ---------------------------------------------------------------------------------------------- arms
def arm_spec(name):
    """Accept either the store name ("pi05_spatial_inf") or the source name ("tr_pi05_sp_inf")."""
    parts = name.split("_")
    if parts[0] == "tr":
        m, sshort, a = parts[1:]
        s = {v[0]: k for k, v in SUITES.items()}[sshort]
    else:
        m, s, a = parts
    check(m in MODELS and s in SUITES and a in EXECS, f"unknown arm {name!r}")
    sshort, sfull = SUITES[s]
    src = f"tr_{m}_{sshort}_{a}"
    return dict(arm=f"{m}_{s}_{a}", model=m, suite=s, suite_full=sfull, exec=a, source_arm=src,
                source_dir=f"{TRACE_ROOT}/runs/{src}/trace/{sfull}",
                lib_meta=f"{LIB_DIR}/{m}_{sfull}.meta.json")


# ------------------------------------------------------------------------------------------ pass 1
def scan_file(path):
    """Attrs + metadata only (default driver: a handful of small reads, no dataset payload)."""
    with h5py.File(path, "r") as f:
        a = f.attrs
        steps = sorted(k for k in f if k.startswith("step_"))
        others = sorted(k for k in f if not k.startswith("step_"))
        g = f[steps[0]] if steps else None
        shapes = {}
        if g is not None:
            for k in ("trace/query_keys/vision_0", "trace/query_keys/vision_1", "trace/query_keys/robot_state",
                      "trace/raw_state", "trace/actions/full_inference", "trace/actions/full_hit",
                      "trace/actions/executed", "vision_0", "vision_1"):
                shapes[k] = list(g[k].shape) if k in g else None
            ri = g["trace/raw_images"] if "trace/raw_images" in g else {}
            shapes["raw_images"] = {k: list(ri[k].shape) for k in ri}
        extra = json.loads(_s(a.get("trace_extra_metadata_json", "{}")))
        return dict(file=path, size=os.path.getsize(path), uid=_s(a["trace_task_uid"]), task=_s(a["task"]),
                    task_id=int(a["task_id"]), init=int(a["orig_init_state_idx"]), success=bool(a["success"]),
                    num_steps=int(a["num_steps"]), n_step_groups=len(steps),
                    steps_contiguous=[int(s[5:]) for s in steps] == list(range(len(steps))),
                    other_groups=others, shapes=shapes,
                    extra_task_id=extra.get("task_id"), extra_init=extra.get("orig_init_state_idx"))


def select_limited(eps, k):
    """Deterministic, arm-independent pick of k episodes spread over tasks, alternating token-subsample
    inits and other inits: k=3 -> (task0, init0), (task1, init1), (task2, init10)."""
    by = {(e["task_id"], e["init"]): e for e in eps}
    tasks = sorted({e["task_id"] for e in eps})
    inits = sorted({e["init"] for e in eps})
    tok = [i for i in TOK_INITS if i in inits]
    other = [i for i in inits if i not in TOK_INITS]
    pri = []
    for j in range(max(len(tok), len(other))):
        pri += ([tok[j]] if j < len(tok) else []) + ([other[j]] if j < len(other) else [])
    chosen = []
    for j in range(len(pri) * len(tasks)):
        key = (tasks[j % len(tasks)], pri[j % len(pri)])
        if len(chosen) >= k:
            break
        if key in by and key not in chosen:
            chosen.append(key)
    for key in sorted(by):
        if len(chosen) >= k:
            break
        if key not in chosen:
            chosen.append(key)
    return [by[c] for c in chosen]


def plan_arm(spec, scans, limit):
    arm = spec["arm"]
    check(len(scans) > 0, f"{arm}: no episode_*.h5 under {spec['source_dir']}")
    H = 10 if spec["model"] == "pi05" else 16
    Drs = 32 if spec["model"] == "pi05" else 8
    ik0, ik1 = IMG_KEYS[spec["model"]]
    img_shape = None
    for s in scans:
        f = s["file"]
        check(s["num_steps"] == s["n_step_groups"],
              f"{f}: num_steps={s['num_steps']} but {s['n_step_groups']} step groups")
        check(s["steps_contiguous"], f"{f}: step groups are not step_0000..step_{s['n_step_groups'] - 1:04d}")
        check(s["num_steps"] > 0, f"{f}: zero steps")
        check(not s["other_groups"], f"{f}: unexpected top-level groups {s['other_groups']}")
        check(s["extra_task_id"] in (None, s["task_id"]) and s["extra_init"] in (None, s["init"]),
              f"{f}: trace_extra_metadata_json disagrees with task_id/init attrs")
        sh = s["shapes"]
        check(sh["trace/query_keys/vision_0"] == [KEY_DIM] and sh["trace/query_keys/vision_1"] == [KEY_DIM],
              f"{f}: key shape {sh['trace/query_keys/vision_0']}")
        check(sh["trace/query_keys/robot_state"] == [Drs], f"{f}: robot_state key shape {sh['trace/query_keys/robot_state']}")
        check(sh["trace/raw_state"] == [8], f"{f}: raw_state shape {sh['trace/raw_state']}")
        for k in ("full_inference", "full_hit", "executed"):
            check(sh[f"trace/actions/{k}"] == [H, N_ACT_DIM], f"{f}: action {k} shape {sh[f'trace/actions/{k}']}")
        check(sh["vision_0"] == list(TOK_SHAPE) and sh["vision_1"] == list(TOK_SHAPE), f"{f}: token shape")
        ri = sh["raw_images"]
        check(ik0 in ri and ik1 in ri, f"{f}: raw image keys {sorted(ri)} lack {ik0}/{ik1}")
        check(ri[ik0] == ri[ik1] and len(ri[ik0]) == 3 and ri[ik0][2] == 3, f"{f}: raw image shapes {ri}")
        img_shape = img_shape or ri[ik0]
        check(ri[ik0] == img_shape, f"{f}: raw image shape {ri[ik0]} != {img_shape} of other files")
    keys = [(s["task_id"], s["init"]) for s in scans]
    dups = sorted({k for k in keys if keys.count(k) > 1})
    check(not dups, f"{arm}: duplicate (task_id, init) episodes {dups[:10]}")
    t2id = {}
    for s in scans:
        check(t2id.setdefault(s["task"], s["task_id"]) == s["task_id"], f"{arm}: task {s['task']!r} has two task_ids")
    check(len(set(t2id.values())) == len(t2id), f"{arm}: task_id shared by two task strings")
    uids = [s["uid"] for s in scans]
    check(len(set(uids)) == len(uids), f"{arm}: duplicate trace_task_uid")
    sel = select_limited(scans, limit) if limit else list(scans)
    sel = sorted(sel, key=lambda s: (s["task_id"], s["init"]))
    eps, r, m = [], 0, 0
    for i, s in enumerate(sel):
        tok = s["init"] in TOK_INITS
        eps.append(dict(uid=s["uid"], file=s["file"], task=s["task"], task_id=s["task_id"], init=s["init"],
                        success=s["success"], num_steps=s["num_steps"], start=r, end=r + s["num_steps"],
                        _tok_start=m if tok else -1, _size=s["size"]))
        r += s["num_steps"]
        m += s["num_steps"] if tok else 0
    return dict(spec=spec, H=H, Drs=Drs, img_shape=img_shape, img_keys=[ik0, ik1], episodes=eps, N=r, M=m,
                n_source_files=len(scans), tasks={str(v): k for k, v in sorted(t2id.items(), key=lambda kv: kv[1])})


def array_specs(p):
    N, M, H, Drs, (h, w, c) = p["N"], p["M"], p["H"], p["Drs"], p["img_shape"]
    return {
        "ep.npy": ((N,), np.int32), "step.npy": ((N,), np.int16),
        "key_v0.npy": ((N, KEY_DIM), np.float32), "key_v1.npy": ((N, KEY_DIM), np.float32),
        "rs.npy": ((N, Drs), np.float32), "raw_state.npy": ((N, 8), np.float32),
        "a_inf.npy": ((N, H, N_ACT_DIM), np.float32), "a_hit.npy": ((N, H, N_ACT_DIM), np.float32),
        "a_exec.npy": ((N, H, N_ACT_DIM), np.float32),
        "rec_top1.npy": ((N,), np.int32), "rec_score.npy": ((N,), np.float32),
        "rec_perfield.npy": ((N, 3), np.float32),
        "tok/rows.npy": ((M,), np.int64), "tok/v0.npy": ((M,) + TOK_SHAPE, np.float16),
        "tok/v1.npy": ((M,) + TOK_SHAPE, np.float16),
        "tok/img0.npy": ((M, h, w, c), np.uint8), "tok/img1.npy": ((M, h, w, c), np.uint8),
    }


def nbytes(specs):
    return int(sum(np.prod(sh, dtype=np.int64) * np.dtype(dt_).itemsize for sh, dt_ in specs.values()))


# ------------------------------------------------------------------------------------------ pass 2
_W = {"mm": {}, "lib": {}}


def _worker_init():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _mm(arm_dir, names):
    d = _W["mm"].setdefault(arm_dir, {})
    for n in names:
        if n not in d:
            d[n] = np.load(os.path.join(arm_dir, n), mmap_mode="r+")
    return d


def _lib(meta_path):
    if meta_path not in _W["lib"]:
        ids = json.load(open(meta_path))["ids"]
        _W["lib"][meta_path] = {k: i for i, k in enumerate(ids)}
    return _W["lib"][meta_path]


def fill_file(job):
    t0 = time.time()
    path, arm_dir, ep_idx, start, end, tok_start = (job[k] for k in ("file", "arm_dir", "ep", "start", "end", "tok_start"))
    H, Drs, (ik0, ik1), driver = job["H"], job["Drs"], job["img_keys"], job["driver"]
    tok = tok_start >= 0
    lib = _lib(job["lib_meta"])
    n = end - start
    kv0 = np.empty((n, KEY_DIM), np.float32)
    kv1 = np.empty((n, KEY_DIM), np.float32)
    rs = np.empty((n, Drs), np.float32)
    raw = np.empty((n, 8), np.float32)
    act = {k: np.empty((n, H, N_ACT_DIM), np.float32) for k in ("full_inference", "full_hit", "executed")}
    top1 = np.empty(n, np.int32)
    score = np.empty(n, np.float32)
    pf = np.empty((n, 3), np.float32)
    if tok:
        tv0 = np.empty((n,) + TOK_SHAPE, np.float16)
        tv1 = np.empty((n,) + TOK_SHAPE, np.float16)
        tim0 = tim1 = None
    use_core = driver == "core" or (driver == "auto" and tok)
    kw = dict(driver="core", backing_store=False) if use_core else {}
    with h5py.File(path, "r", **kw) as f:
        t_open = time.time() - t0
        steps = sorted(k for k in f if k.startswith("step_"))
        check(len(steps) == n == int(f.attrs["num_steps"]),
              f"{path}: {len(steps)} step groups, num_steps={int(f.attrs['num_steps'])}, planned rows={n}")
        check([int(s[5:]) for s in steps] == list(range(n)), f"{path}: step groups not contiguous")
        check((int(f.attrs["task_id"]), int(f.attrs["orig_init_state_idx"])) == (job["task_id"], job["init"]),
              f"{path}: task_id/init changed since pass 1")
        for i, k in enumerate(steps):
            g = f[k]
            g["trace/query_keys/vision_0"].read_direct(kv0, dest_sel=np.s_[i])
            g["trace/query_keys/vision_1"].read_direct(kv1, dest_sel=np.s_[i])
            rs[i] = g["trace/query_keys/robot_state"][()]
            raw[i] = g["trace/raw_state"][()]
            for a_ in act:
                act[a_][i] = g[f"trace/actions/{a_}"][()]
            rid = _s(g["trace/search/real_topk_ids"][()][0])
            j = lib.get(rid)
            check(j is not None, f"{path} {k}: real_topk_ids[0]={rid!r} not in library {job['lib_meta']}")
            top1[i] = j
            score[i] = g["trace/search/real_topk_scores"][()][0]
            for c, fn in enumerate(PERFIELD):
                pf[i, c] = g[f"trace/search/twin_topk_per_field/{fn}"][()][0]
            if tok:
                g["vision_0"].read_direct(tv0, dest_sel=np.s_[i])
                g["vision_1"].read_direct(tv1, dest_sel=np.s_[i])
                i0 = g[f"trace/raw_images/{ik0}"][()]
                i1 = g[f"trace/raw_images/{ik1}"][()]
                if tim0 is None:
                    tim0 = np.empty((n,) + i0.shape, np.uint8)
                    tim1 = np.empty((n,) + i1.shape, np.uint8)
                tim0[i] = i0
                tim1[i] = i1
    t_read = time.time() - t0
    for name, arr in (("key_v0", kv0), ("key_v1", kv1), ("rs", rs), ("raw_state", raw), ("a_inf", act["full_inference"]),
                      ("a_hit", act["full_hit"]), ("a_exec", act["executed"]), ("rec_score", score),
                      ("rec_perfield", pf)):
        bad = ~np.isfinite(arr.reshape(n, -1)).all(axis=1)
        check(not bad.any(), f"{path}: non-finite {name} at steps {np.flatnonzero(bad)[:10].tolist()}")
    ref = act["full_hit"] if job["exec"] == "cache" else act["full_inference"]
    check(np.array_equal(act["executed"], ref), f"{path}: executed != {'full_hit' if job['exec'] == 'cache' else 'full_inference'}")
    rs_pad_nonzero = int((rs[:, RS_VALID_DIMS:] != 0).any(axis=1).sum()) if Drs > RS_VALID_DIMS else 0
    if tok:
        for name, arr in (("tok_v0", tv0), ("tok_v1", tv1)):
            bad = ~np.isfinite(arr.reshape(n, -1)).all(axis=1)
            check(not bad.any(), f"{path}: non-finite {name} at steps {np.flatnonzero(bad)[:10].tolist()}")
    names = ["ep.npy", "step.npy", "key_v0.npy", "key_v1.npy", "rs.npy", "raw_state.npy", "a_inf.npy", "a_hit.npy",
             "a_exec.npy", "rec_top1.npy", "rec_score.npy", "rec_perfield.npy"]
    if tok:
        names += ["tok/rows.npy", "tok/v0.npy", "tok/v1.npy", "tok/img0.npy", "tok/img1.npy"]
    mm = _mm(arm_dir, names)
    sl = slice(start, end)
    mm["key_v0.npy"][sl] = kv0
    mm["key_v1.npy"][sl] = kv1
    mm["rs.npy"][sl] = rs
    mm["raw_state.npy"][sl] = raw
    mm["a_inf.npy"][sl] = act["full_inference"]
    mm["a_hit.npy"][sl] = act["full_hit"]
    mm["a_exec.npy"][sl] = act["executed"]
    mm["rec_top1.npy"][sl] = top1
    mm["rec_score.npy"][sl] = score
    mm["rec_perfield.npy"][sl] = pf
    mm["step.npy"][sl] = np.arange(n, dtype=np.int16)
    if tok:
        ts = slice(tok_start, tok_start + n)
        check(mm["tok/img0.npy"].shape[1:] == tim0.shape[1:], f"{path}: image shape {tim0.shape[1:]}")
        mm["tok/v0.npy"][ts] = tv0
        mm["tok/v1.npy"][ts] = tv1
        mm["tok/img0.npy"][ts] = tim0
        mm["tok/img1.npy"][ts] = tim1
        mm["tok/rows.npy"][ts] = np.arange(start, end, dtype=np.int64)
    mm["ep.npy"][sl] = ep_idx  # last: a fully filled episode is the one whose ep rows are no longer -1
    return dict(arm=job["arm"], file=path, rows=n, tok_rows=n if tok else 0, size=job["size"], core=use_core,
                rs_pad_nonzero_rows=rs_pad_nonzero,
                t_open=t_open, t_read=t_read, t_total=time.time() - t0)


# -------------------------------------------------------------------------------------- finalize
def finalize_arm(plan, arm_dir, stats, run_meta):
    spec, eps, N, M = plan["spec"], plan["episodes"], plan["N"], plan["M"]
    arm = spec["arm"]
    specs = array_specs(plan)
    check(len(stats) == len(eps), f"{arm}: {len(stats)} files filled, {len(eps)} planned")
    check(sum(s["rows"] for s in stats) == N, f"{arm}: filled rows {sum(s['rows'] for s in stats)} != N={N}")
    check(sum(s["tok_rows"] for s in stats) == M, f"{arm}: filled tok rows != M={M}")
    exp_ep = np.concatenate([np.full(e["num_steps"], i, np.int32) for i, e in enumerate(eps)])
    exp_step = np.concatenate([np.arange(e["num_steps"], dtype=np.int16) for e in eps])
    ep = np.load(f"{arm_dir}/ep.npy")
    check(np.array_equal(ep, exp_ep), f"{arm}: ep.npy mismatch at {np.flatnonzero(ep != exp_ep)[:10].tolist()} (unfilled rows?)")
    check(np.array_equal(np.load(f"{arm_dir}/step.npy"), exp_step), f"{arm}: step.npy mismatch")
    exp_tok = np.concatenate([np.arange(e["start"], e["end"], dtype=np.int64) for e in eps if e["_tok_start"] >= 0] or
                             [np.zeros(0, np.int64)])
    check(np.array_equal(np.load(f"{arm_dir}/tok/rows.npy"), exp_tok), f"{arm}: tok/rows.npy mismatch")
    meta = json.load(open(spec["lib_meta"]))
    top1 = np.load(f"{arm_dir}/rec_top1.npy")
    check(top1.min() >= 0 and top1.max() < len(meta["ids"]), f"{arm}: rec_top1 out of library range")
    tk = np.asarray(meta["task_key"], dtype=object)[top1]
    etask = np.asarray([e["task"] for e in eps], dtype=object)[ep]
    check((tk == etask).all(), f"{arm}: {(tk != etask).sum()} rows whose top-1 library task_key != episode task")
    for name, (shape, dtype) in specs.items():
        a = np.load(f"{arm_dir}/{name}", mmap_mode="r")
        check(a.shape == shape and a.dtype == dtype, f"{arm}: {name} is {a.shape} {a.dtype}, expected {shape} {dtype}")
        del a
    for name in specs:  # push the workers' dirty pages to disk before declaring the arm complete
        fd = os.open(f"{arm_dir}/{name}", os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    rs_pad = int(sum(s["rs_pad_nonzero_rows"] for s in stats))
    if rs_pad:  # report, do not fail (owner): pi05 robot_state dims 8..31 are expected to be exactly 0
        log(f"WARNING {arm}: {rs_pad} rows have non-zero robot_state padding dims {RS_VALID_DIMS}..{plan['Drs'] - 1}")
    ep_json = [{k: v for k, v in e.items() if not k.startswith("_")} for e in eps]
    ej_path = f"{arm_dir}/episodes.json"
    with open(ej_path + ".tmp", "w") as f:
        json.dump(ep_json, f, indent=1)
    os.replace(ej_path + ".tmp", ej_path)
    t_fin = _now()
    wall = time.time() - run_meta["t0"]
    manifest = dict(
        schema=SCHEMA, arm=arm, model=spec["model"], suite=spec["suite"], suite_full=spec["suite_full"],
        exec=spec["exec"], source_arm=spec["source_arm"], source_dir=spec["source_dir"],
        library_meta=spec["lib_meta"], library_meta_sha256=sha256_file(spec["lib_meta"]), library_rows=len(meta["ids"]),
        complete=run_meta["limit"] is None and len(eps) == plan["n_source_files"],
        limit_episodes=run_meta["limit"], n_source_files=plan["n_source_files"],
        counts=dict(episodes=len(eps), rows=N, tok_episodes=sum(e["_tok_start"] >= 0 for e in eps), tok_rows=M,
                    success_episodes=sum(e["success"] for e in eps), tasks=len({e["task_id"] for e in eps})),
        H=plan["H"], Drs=plan["Drs"], img_shape=plan["img_shape"],
        img_source_keys={"img0": f"trace/raw_images/{plan['img_keys'][0]}", "img1": f"trace/raw_images/{plan['img_keys'][1]}"},
        act_valid_dims=ACT_VALID_DIMS, exec_steps=EXEC_STEPS, rs_valid_dims=RS_VALID_DIMS, gripper_dim=GRIPPER_DIM,
        rs_padding_nonzero_rows=rs_pad,
        tok_inits=list(TOK_INITS), rec_perfield_order=list(PERFIELD),
        rec_perfield_semantics="normalized per-field score 0.5*(tanh((x-mu)/sigma)+1), robot_state x=-l2 distance",
        tasks=plan["tasks"],
        arrays={name: dict(shape=list(sh), dtype=np.dtype(dt_).name) for name, (sh, dt_) in specs.items()},
        bytes_total=nbytes(specs),
        episodes_json_sha256=sha256_file(ej_path),
        extraction=dict(started=run_meta["started"], finished=t_fin, wall_s_since_run_start=round(wall, 1),
                        source_bytes_read=int(sum(s["size"] for s in stats)),
                        file_seconds_total=round(sum(s["t_total"] for s in stats), 1),
                        workers=run_meta["workers"], driver=run_meta["driver"], host=socket.gethostname(),
                        script=os.path.abspath(__file__), script_sha256=sha256_file(os.path.abspath(__file__)),
                        argv=sys.argv),
        checks_passed=["num_steps == #step groups, contiguous", "unique (task_id, init) and uid", "task<->task_id bijective",
                       "every real_topk_id in library ids", "top-1 library task_key == episode task",
                       "ep/step/tok rows fully filled and ordered", "finite keys/rs/raw_state/actions/scores/tokens",
                       "executed == full_hit (cache) / full_inference (inf)",
                       "robot_state padding dims zero (report-only, see rs_padding_nonzero_rows)",
                       "array shapes/dtypes"],
    )
    mp = f"{arm_dir}/manifest.json"
    with open(mp + ".tmp", "w") as f:
        json.dump(manifest, f, indent=1)
    os.replace(mp + ".tmp", mp)
    return manifest


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default=",".join(ALL_ARMS), help="comma list (store or tr_* names); default all 8")
    ap.add_argument("--limit-episodes", type=int, default=None, help="per-arm deterministic subset (smoke runs)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="queries root")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--driver", choices=("core", "default", "auto"), default="core",
                    help="h5 driver in pass 2; auto = core for token-subsample files, default elsewhere")
    ap.add_argument("--force", action="store_true", help="rebuild arms whose manifest.json exists")
    ap.add_argument("--plan-only", action="store_true", help="pass 1 only: print the row plan and sizes")
    args = ap.parse_args()
    check(args.workers <= 64, "at most 64 processes on this host (protocol §8)")
    specs = [arm_spec(a.strip()) for a in args.arms.split(",") if a.strip()]
    todo = []
    for sp in specs:
        if os.path.exists(f"{args.out}/{sp['arm']}/manifest.json") and not args.force and not args.plan_only:
            log(f"skip {sp['arm']}: manifest.json exists (use --force to rebuild)")
        else:
            todo.append(sp)
    if not todo:
        log("nothing to do")
        return 0
    run_meta = dict(t0=time.time(), started=_now(), limit=args.limit_episodes, workers=args.workers, driver=args.driver)
    log(f"extract_queries: arms={[s['arm'] for s in todo]} out={args.out} workers={args.workers} "
        f"driver={args.driver} limit={args.limit_episodes}")
    files = {sp["arm"]: sorted(glob.glob(f"{sp['source_dir']}/episode_*.h5")) for sp in todo}
    all_files = [p for sp in todo for p in files[sp["arm"]]]
    t = time.time()
    with ProcessPoolExecutor(min(args.workers, 32), initializer=_worker_init) as ex:
        scans = dict(zip(all_files, ex.map(scan_file, all_files, chunksize=8)))
    log(f"pass 1: scanned {len(all_files)} files in {time.time() - t:.1f}s")
    plans = {}
    for sp in todo:
        p = plan_arm(sp, [scans[f] for f in files[sp["arm"]]], args.limit_episodes)
        plans[sp["arm"]] = p
        b = nbytes(array_specs(p))
        log(f"  {sp['arm']}: {len(p['episodes'])}/{p['n_source_files']} episodes, N={p['N']} rows, M={p['M']} tok rows, "
            f"H={p['H']} Drs={p['Drs']} img={p['img_shape']}, source {sum(e['_size'] for e in p['episodes']) / 2**30:.1f} GiB, "
            f"store {b / 2**30:.2f} GiB")
    tot_b = sum(nbytes(array_specs(p)) for p in plans.values())
    tot_src = sum(e["_size"] for p in plans.values() for e in p["episodes"])
    log(f"plan: {sum(p['N'] for p in plans.values())} rows, source {tot_src / 2**30:.1f} GiB, store {tot_b / 2**30:.1f} GiB")
    if args.plan_only:
        return 0

    jobs = []
    for arm, p in plans.items():
        arm_dir = f"{args.out}/{arm}"
        os.makedirs(f"{arm_dir}/tok", exist_ok=True)
        if os.path.exists(f"{arm_dir}/manifest.json"):  # --force: the arm is incomplete until re-finalized
            os.remove(f"{arm_dir}/manifest.json")
        for name, (shape, dtype) in array_specs(p).items():
            a = np.lib.format.open_memmap(f"{arm_dir}/{name}", mode="w+", dtype=dtype, shape=shape)
            if name in ("ep.npy", "step.npy", "rec_top1.npy", "tok/rows.npy"):
                a[:] = -1
            a.flush()
            del a
        for i, e in enumerate(p["episodes"]):
            jobs.append(dict(arm=arm, arm_dir=arm_dir, file=e["file"], ep=i, start=e["start"], end=e["end"],
                             tok_start=e["_tok_start"], task_id=e["task_id"], init=e["init"], size=e["_size"],
                             H=p["H"], Drs=p["Drs"], img_keys=p["img_keys"], lib_meta=p["spec"]["lib_meta"],
                             exec=p["spec"]["exec"],
                             driver=args.driver))
    jobs.sort(key=lambda j: -j["size"] * (2 if j["tok_start"] >= 0 else 1))  # longest first -> short tail
    remaining = {arm: len(p["episodes"]) for arm, p in plans.items()}
    stats = {arm: [] for arm in plans}
    total_bytes = sum(j["size"] for j in jobs)
    done_bytes, done_files, last_log = 0, 0, time.time()
    t2 = time.time()
    log(f"pass 2: {len(jobs)} files, {total_bytes / 2**30:.1f} GiB with {args.workers} workers")
    with ProcessPoolExecutor(args.workers, initializer=_worker_init) as ex:
        pending = {ex.submit(fill_file, j) for j in jobs}
        while pending:
            finished, pending = wait(pending, timeout=60, return_when=FIRST_EXCEPTION)
            for fu in finished:
                exc = fu.exception()
                if exc is not None:
                    for q in pending:
                        q.cancel()
                    raise exc
                r = fu.result()
                stats[r["arm"]].append(r)
                done_bytes += r["size"]
                done_files += 1
                remaining[r["arm"]] -= 1
                if remaining[r["arm"]] == 0:
                    man = finalize_arm(plans[r["arm"]], f"{args.out}/{r['arm']}", stats[r["arm"]], run_meta)
                    log(f"arm {r['arm']} complete: {man['counts']} store {man['bytes_total'] / 2**30:.2f} GiB")
            el = time.time() - t2
            if time.time() - last_log > 30 or not pending:
                rate = done_bytes / max(el, 1e-9)
                eta = (total_bytes - done_bytes) / max(rate, 1)
                log(f"  {done_files}/{len(jobs)} files, {done_bytes / 2**30:.1f}/{total_bytes / 2**30:.1f} GiB, "
                    f"{rate / 2**20:.0f} MiB/s, elapsed {el / 60:.1f} min, eta {eta / 60:.1f} min")
                last_log = time.time()
    all_stats = [s for v in stats.values() for s in v]
    el = time.time() - t2
    per = np.array([s["t_total"] for s in all_stats])
    op = np.array([s["t_open"] for s in all_stats])
    log(f"pass 2 done: {len(all_stats)} files, {total_bytes / 2**30:.2f} GiB in {el:.1f}s "
        f"= {total_bytes / 2**20 / max(el, 1e-9):.0f} MiB/s aggregate; per-file s: mean {per.mean():.2f} "
        f"p50 {np.median(per):.2f} max {per.max():.2f}; h5 open/read part mean {op.mean():.2f}; "
        f"per-file MiB/s mean {np.mean([s['size'] / 2**20 / s['t_total'] for s in all_stats]):.0f}")
    timing_path = f"{args.out}/_run"
    os.makedirs(timing_path, exist_ok=True)
    with open(f"{timing_path}/timing_{int(run_meta['t0'])}.json", "w") as f:
        json.dump(dict(argv=sys.argv, started=run_meta["started"], finished=_now(), wall_pass2_s=el,
                       bytes=total_bytes, files=all_stats), f, indent=1)
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        log("FAILED")
        sys.exit(1)
