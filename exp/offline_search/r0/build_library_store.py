"""build_library_store.py -- retrieval candidate sets -> memmap "library store" (offline_search R0-B).

Layout written per library (other agents code against it; see logs/offline_search_exploration.log.md §5.1):

  <out>/<m>_<s>/<name>/            m in {pi05, groot}, s in {spatial, l10}, name in {current, bpool_all}
    key_v0.npy / key_v1.npy        float32[L, 32768]  pool16 keys of vision_0 / vision_1
    rs.npy                         float32[L, Drs]    raw robot_state key (Drs = 32 pi05, 8 groot)
    action.npy                     float32[L, H, 32]  action chunk (H = 10 pi05, 16 groot; model-normalized)
    task_id.npy int16[L]           task id, identical to the query store / trace attr `task_id`
    episode.npy int32[L]           index into episodes.json
    step.npy int16[L]              decision index inside the source episode (h5 step_XXXX)
    ep_len.npy int16[L]            num_steps of the source episode (full length, not rows in this store)
    progress.npy float32[L]        step / (ep_len - 1)   (0 when ep_len == 1)
    success.npy bool[L]            success attr of the source episode
    prev.npy / next.npy int32[L]   row of the previous / next step of the same episode, -1 if not in store
    ids.json                       library id "<episode_stem>:<step>" per row (bpool_cs: "task_N/episode_M:<step>")
    episodes.json                  [{stem, file, task, task_id, success, num_steps, start, end, n_rows,
                                     episode_id, orig_init_state_idx, contiguous}]; [start, end) = row span
    tok/rows.npy int64[M]          rows (into L) that carry tokens; M == L and rows == arange(L) in a full
                                   build, a subset only with --limit-episodes on `current`
    tok/v0.npy / tok/v1.npy        float16[M, 256, 2048]  raw vision tokens from the build h5 (bit-exact)
    tok/img0.npy / tok/img1.npy    uint8[M, 224, 224, 3]  pi05 only: input_images/{base_0_rgb, left_wrist_0_rgb}
    manifest.json                  counts, arrays, sources, sha256, task map, verification (written last)

`current`   = exactly the pickle entries in pickle order.  keys / rs / action are copied bit-exactly from
              the pickle; metadata and tokens are joined from the build h5 through the id.
`bpool_all` = every step of every build episode (failed ones included), episodes sorted by file stem;
              keys recomputed from the tokens: tokens.f32.reshape(4,4,4,4,2048).mean((1,3)).reshape(-1)
              (bit-identical to the online cp1_*spatial_pool_16 key builder, checked below);
              action = clean_action, rs = robot_state.
`bpool_cs`  = pi05 only: same as bpool_all but from the cache_size ablation corpus (Aug 2026, cache OFF,
              whole B-pool: 50 inits x 10 tasks) under /archive/openpi/ablation_study/cache_size/collect_h5/
              <suite>/task_N/episode_M.h5.  Stems are the relative path "task_N/episode_M" (bare stems repeat
              across tasks), init = M (B-pool index), episodes ordered by (N, M).  Row plan from the collect
              ledger (no attr scan on the HDD); every file is checked against the ledger (bytes, sha256 of the
              bytes actually read, num_steps, success), against <suite>_collect_results.json (success, init),
              and path task N against the trace task map; keys are cross-checked bit-exactly against the
              cache_size all_S3 library pickle (same rollouts, built by the online key builder).  Not part of
              the default --names; request it explicitly.

Build flow: each library is written into `<name>.partial/` and renamed to `<name>/` only when every
verification passed; failures leave `<name>.partial/` for inspection and write `<name>.ERROR`, and the
process exits non-zero.  A finished `<name>/` is skipped unless --force (idempotent per library).  When
all 8 core libraries (current + bpool_all for 4 model x suite) are complete, `<out>/READY` is written
(it also lists any finished bpool_cs libraries).

IO discipline: SSD sources (pi05 build h5 under /home) are read by a process pool with driver="core".
/archive sources (GR00T builds, SMR HDD) are read strictly serially by ONE reader (the main process
reads each whole file into memory, in stem order) and the bytes are handed to the pool for decoding;
workers never touch the HDD.

Usage (see run_build_library_store.sh for the env-setting launcher):
  python build_library_store.py [--libs pi05_spatial,...] [--names current,bpool_all[,bpool_cs]]
                                [--limit-episodes K] [--out DIR] [--force] [--workers 16]
                                [--archive-workers 8] [--dry-run]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import datetime as dt
import glob
import hashlib
import io
import json
import multiprocessing as mp
import pickle
import socket
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import h5py
import numpy as np

REPO = Path("/home/weiland/projects/openpi")
TRACE_ROOT = Path("/home/weiland/trace_runs/dual_20260923")
DEFAULT_OUT = "/home/weiland/trace_runs/offline_search_store/library"
AUDIT = TRACE_ROOT / "audit" / "libs"
NAMES = ("current", "bpool_all", "bpool_cs")
CORE_NAMES = ("current", "bpool_all")   # READY / default --names
CS_ROOT = Path("/archive/openpi/ablation_study/cache_size")
CS_DATA = Path("/data/openpi/ablation_study/cache_size")
TOK_SHAPE = (256, 2048)
IMG_SHAPE = (224, 224, 3)
KEY_DIM = 32768
COS_TOL = 0.9999          # verification threshold for recomputed pool16 keys vs the pickle
EXACT_TOL = 1e-6          # rs / action must equal the h5 values
FACTS = dict(act_valid_dims=7, exec_steps=5, rs_valid_dims=8, gripper_dim=6)
CS_BATCH_LABEL = "2026-08 pi05 cache_size ablation collect (weilandserver, cache OFF, seed 7, whole B-pool)"
FACTS_NOTE = ("action chunks (H,32): only dims 0..6 valid (6 = gripper), pi05 7..31 ~0.001 padding, groot 7..31 "
              "noise-like padding; only the first 5 steps are executed. robot_state: pi05 32-d with 0..7 valid "
              "(rest 0), groot 8-d all valid. Arrays are stored raw/full; action statistics use [:5, :7].")

LIBS = {
    "pi05_spatial": dict(
        model="pi05", suite="spatial", suite_full="libero_spatial", H=10, Drs=32, images=True, serial=False,
        pkl=TRACE_ROOT / "libs/libero_spatial/cp1_spatial_pool_16.pkl",
        build_dir=REPO / "exp/common/data/db/libero_cache/libero_spatial",
        trace_runs=("tr_pi05_sp_inf", "tr_pi05_sp_cache"), audit="pi05_libero_spatial",
        init_map=REPO / "exp/common/data/db/libero_cache/libero_spatial_init_map.json",
        batch_label="2026-04 pi05 libero_cache build (current library source)",
        cs=dict(dir=CS_ROOT / "collect_h5/libero_spatial",
                ledger=CS_DATA / "results/collect_ledger_libero_spatial.json",
                results=CS_DATA / "results/libero_spatial_collect_results.json",
                ref=CS_DATA / "artifacts/cache_size_libero_spatial_all_S3.pkl")),
    "pi05_l10": dict(
        model="pi05", suite="l10", suite_full="libero_10", H=10, Drs=32, images=True, serial=False,
        pkl=TRACE_ROOT / "libs/libero_10/cp1_spatial_pool_16.pkl",
        build_dir=REPO / "exp/common/data/db/libero_cache/libero_10",
        trace_runs=("tr_pi05_l10_inf", "tr_pi05_l10_cache"), audit="pi05_libero_10",
        init_map=None,  # libero_10_init_map.json has no trajectory ids -> init left null
        batch_label="2026-04 pi05 libero_cache build (current library source)",
        cs=dict(dir=CS_ROOT / "collect_h5/libero_10",
                ledger=CS_DATA / "results/collect_ledger_libero_10.json",
                results=CS_DATA / "results/libero_10_collect_results.json",
                ref=CS_DATA / "artifacts/cache_size_libero_10_all_S3.pkl")),
    "groot_spatial": dict(
        model="groot", suite="spatial", suite_full="libero_spatial", H=16, Drs=8, images=False, serial=True,
        pkl=Path("/data/libero_cache/libraries/libero_spatial/libero_spatial_sp16_S3.pkl"),
        build_dir=Path("/data/libero_cache/build_spatial/libero_spatial"),
        trace_runs=("tr_groot_sp_inf", "tr_groot_sp_cache"), audit="groot_libero_spatial",
        results_glob="/data/libero_cache/build_spatial/results_lane*.json",
        batch_label="2026-08 GR00T libero_cache build"),
    "groot_l10": dict(
        model="groot", suite="l10", suite_full="libero_10", H=16, Drs=8, images=False, serial=True,
        pkl=Path("/data/libero_cache/libraries/libero_10/libero_10_sp16_S3.pkl"),
        build_dir=Path("/data/libero_cache/build_libero10/libero_10"),
        trace_runs=("tr_groot_l10_inf", "tr_groot_l10_cache"), audit="groot_libero_10",
        results_glob="/data/libero_cache/build_libero10/results_lane*.json",
        batch_label="2026-08 GR00T libero_cache build"),
}

# every file this script may create inside a library dir (used for --force / stale-partial cleanup,
# which removes exactly these files one by one -- never a recursive delete)
KNOWN_FILES = ["key_v0.npy", "key_v1.npy", "rs.npy", "action.npy", "task_id.npy", "episode.npy", "step.npy",
               "ep_len.npy", "progress.npy", "success.npy", "prev.npy", "next.npy", "ids.json", "episodes.json",
               "manifest.json", "tok/rows.npy", "tok/v0.npy", "tok/v1.npy", "tok/img0.npy", "tok/img1.npy"]


# ----------------------------------------------------------------------------------------------- utils
def stamp() -> dict:
    now = dt.datetime.now(dt.timezone.utc)
    return {"utc": now.isoformat(timespec="seconds"), "local": now.astimezone().isoformat(timespec="seconds")}


def log(*a) -> None:
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}]", *a, flush=True)


def s_(v) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


def py(v):
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(1 << 24)
            if not b:
                return h.hexdigest()
            h.update(b)


def pool16(tok: np.ndarray) -> np.ndarray:
    """(256, 2048) tokens on a 16x16 grid -> 4x4 average pool -> float32[32768] (row-major h, w, C)."""
    return tok.astype(np.float32).reshape(4, 4, 4, 4, TOK_SHAPE[1]).mean(axis=(1, 3)).reshape(-1)


def cos64(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


class CheckError(RuntimeError):
    pass


def check(cond, msg) -> None:
    if not cond:
        raise CheckError(msg)


def new_memmap(path: Path, dtype, shape):
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(str(path), mode="w+", dtype=dtype, shape=tuple(shape))


# ------------------------------------------------------------------------------------ source metadata
def task_map_for(cfg: dict) -> tuple[dict, set, dict]:
    """{task string -> task_id} from the trace file attrs (the query store uses the same attr)."""
    m, shas, nfiles = {}, set(), {}
    for run in cfg["trace_runs"]:
        files = sorted(glob.glob(str(TRACE_ROOT / "runs" / run / "trace" / "**" / "*.h5"), recursive=True))
        nfiles[run] = len(files)
        check(len(files) > 0, f"no trace files for {run}")
        for p in files:
            with h5py.File(p, "r") as f:
                t, tid = s_(f.attrs["task"]), int(f.attrs["task_id"])
                shas.add(s_(f.attrs.get("trace_library_sha256", "")))
            check(m.setdefault(t, tid) == tid, f"{run}: task {t!r} has two task_ids ({m[t]}, {tid})")
    check(sorted(m.values()) == list(range(10)), f"task ids not 0..9: {sorted(m.values())}")
    return m, shas, nfiles


def init_lookup(cfg: dict) -> tuple[dict, dict, str | None]:
    """(stem -> orig_init_state_idx, episode_id -> results-row, source) -- best effort, only for reporting."""
    if cfg.get("init_map"):
        rows = json.loads(Path(cfg["init_map"]).read_text())
        return {r["trajectory_id"]: int(r["orig_init_state_idx"]) for r in rows}, {}, str(cfg["init_map"])
    if cfg.get("results_glob"):
        by_eid = {}
        for p in sorted(glob.glob(cfg["results_glob"])):
            for r in json.loads(Path(p).read_text()):
                by_eid[int(r["episode_id"])] = r
        return {}, by_eid, cfg["results_glob"]
    return {}, {}, None


def load_pickle(cfg: dict, path=None) -> dict:
    sys.path.insert(0, str(REPO / "src"))
    t0 = time.time()
    with open(path or cfg["pkl"], "rb") as fh:
        d = pickle.load(fh)
    E = d["entries"]
    L = len(E)

    def arr(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    dtypes = set()
    v0 = np.empty((L, KEY_DIM), np.float32)
    v1 = np.empty((L, KEY_DIM), np.float32)
    rs = np.empty((L, cfg["Drs"]), np.float32)
    ac = np.empty((L, cfg["H"], 32), np.float32)
    for i, e in enumerate(E):
        a0, a1 = arr(e.query_keys["vision_0"]).reshape(-1), arr(e.query_keys["vision_1"]).reshape(-1)
        ar, aa = arr(e.query_keys["robot_state"]).reshape(-1), arr(e.payload.action_chunk)
        dtypes |= {str(a0.dtype), str(a1.dtype), str(ar.dtype), str(aa.dtype)}
        check(ar.shape == (cfg["Drs"],) and aa.shape == (cfg["H"], 32), f"pickle entry {i}: rs {ar.shape} / action {aa.shape}")
        v0[i], v1[i], rs[i], ac[i] = a0, a1, ar, aa
    check(dtypes == {"float32"}, f"pickle arrays are not all float32 ({dtypes}) -> copy would not be bit-exact")
    pk = dict(
        ids=[e.id for e in E], task=[e.payload.task_key for e in E], step_idx=[e.step_idx for e in E],
        traj=[e.trajectory_id for e in E], prev=[list(e.prev_ids) for e in E], next=[list(e.next_ids) for e in E],
        v0=v0, v1=v1, rs=rs, action=ac,
        meta=dict(key_builder_type=py(d.get("key_builder_type")), checkpoint_id=str(d.get("checkpoint_id")),
                  vector_dims={k: int(v) for k, v in dict(d.get("vector_dims") or {}).items()},
                  query_key_fields=sorted(E[0].query_keys)),
        load_s=round(time.time() - t0, 1))
    del d, E
    return pk


def audit_equal(cfg: dict, pk: dict) -> dict:
    """The pickle copy vs the already-exported audit memmaps (same pickle -> must be bit-identical)."""
    pre = AUDIT / cfg["audit"]
    if not Path(f"{pre}.meta.json").exists():
        return {"available": False}
    meta = json.loads(Path(f"{pre}.meta.json").read_text())
    out = {"available": True, "ids_equal": meta["ids"] == pk["ids"]}
    for f, k in (("vision_0", "v0"), ("vision_1", "v1"), ("robot_state", "rs"), ("action", "action")):
        a = np.load(f"{pre}.{f}.npy", mmap_mode="r")
        out[f] = bool(a.shape == pk[k].shape and np.array_equal(a, pk[k]))
    return out


# ------------------------------------------------------------------------------------------- workers
def _open(path: str, data: bytes | None):
    if data is not None:
        return h5py.File(io.BytesIO(data), "r")
    return h5py.File(path, "r", driver="core", backing_store=False)


def process_episode(job: dict, data: bytes | None = None) -> dict:
    """Decode one build episode.  mode 'current': compare with the pickle copy already on disk and write
    tokens for the requested rows.  mode 'bpool': write keys / rs / action / tokens for rows [start, end)."""
    t0 = time.time()
    work = Path(job["work"])
    mode, rows, steps, tokpos = job["mode"], job["rows"], job["steps"], job["tokpos"]
    H, Drs, images = job["H"], job["Drs"], job["images"]
    out = dict(stem=job["stem"], n=len(rows), missing=[], bytes=len(data) if data is not None else job.get("size", 0))
    if data is not None and job.get("sha256"):
        out["sha_ok"] = hashlib.sha256(data).hexdigest() == job["sha256"]
    want_tok = any(p >= 0 for p in tokpos)
    mm = {}
    if want_tok:
        names = ["tok/v0.npy", "tok/v1.npy"] + (["tok/img0.npy", "tok/img1.npy"] if images else [])
        mm.update({n: np.load(work / n, mmap_mode="r+") for n in names})
    kmode = "r" if mode == "current" else "r+"
    for n in ("key_v0.npy", "key_v1.npy", "rs.npy", "action.npy"):
        mm[n] = np.load(work / n, mmap_mode=kmode)
    cos0, cos1, ex0, ex1, mab0, mab1 = [], [], [], [], [], []
    rsd = acd = acd_exec = 0.0
    with _open(job["path"], data) as f:
        out["attrs"] = {k: py(v) for k, v in f.attrs.items()}
        out["n_groups"] = sum(1 for k in f.keys() if k.startswith("step_"))
        for r, st, tp in zip(rows, steps, tokpos):
            name = f"step_{st:04d}"
            if name not in f:
                out["missing"].append(r)
                continue
            g = f[name]
            t0_, t1_ = g["vision_0"][()], g["vision_1"][()]
            if t0_.shape != TOK_SHAPE or t1_.shape != TOK_SHAPE or t0_.dtype != np.float16 or t1_.dtype != np.float16:
                raise CheckError(f"{job['stem']}/{name}: token shape/dtype {t0_.shape} {t0_.dtype} {t1_.shape} {t1_.dtype}")
            p0, p1 = pool16(t0_), pool16(t1_)
            rs = np.asarray(g["robot_state"][()], dtype=np.float32).reshape(-1)
            ac = np.asarray(g["clean_action"][()], dtype=np.float32)
            if rs.shape != (Drs,) or ac.shape != (H, 32):
                raise CheckError(f"{job['stem']}/{name}: robot_state {rs.shape} / clean_action {ac.shape}")
            if mode == "current":
                k0, k1 = np.asarray(mm["key_v0.npy"][r]), np.asarray(mm["key_v1.npy"][r])
                cos0.append(cos64(p0, k0)); cos1.append(cos64(p1, k1))
                ex0.append(bool(np.array_equal(p0, k0))); ex1.append(bool(np.array_equal(p1, k1)))
                mab0.append(float(np.abs(p0 - k0).max())); mab1.append(float(np.abs(p1 - k1).max()))
                rsd = max(rsd, float(np.abs(rs - mm["rs.npy"][r]).max()))
                d = np.abs(ac - mm["action.npy"][r])
                acd, acd_exec = max(acd, float(d.max())), max(acd_exec, float(d[:5, :7].max()))
            else:
                mm["key_v0.npy"][r], mm["key_v1.npy"][r] = p0, p1
                mm["rs.npy"][r], mm["action.npy"][r] = rs, ac
            if tp >= 0:
                mm["tok/v0.npy"][tp], mm["tok/v1.npy"][tp] = t0_, t1_
                if images:
                    i0, i1 = g["input_images/base_0_rgb"][()], g["input_images/left_wrist_0_rgb"][()]
                    if i0.shape != IMG_SHAPE or i1.shape != IMG_SHAPE:
                        raise CheckError(f"{job['stem']}/{name}: image shapes {i0.shape} {i1.shape}")
                    mm["tok/img0.npy"][tp], mm["tok/img1.npy"][tp] = i0, i1
    # no explicit msync: MAP_SHARED writes already live in the page cache (visible to every process, written
    # back by the kernel); a synchronous flush per episode stalled workers ~1.3 s / 50 MB on a busy SSD.
    del mm
    out.update(cos0=cos0, cos1=cos1, ex0=ex0, ex1=ex1, mab0=mab0, mab1=mab1, rsd=rsd, acd=acd, acd_exec=acd_exec,
               tok_rows=sum(1 for p in tokpos if p >= 0), secs=time.time() - t0)
    return out


def run_jobs(jobs: list[dict], serial_io: bool, workers: int, label: str) -> tuple[list[dict], dict]:
    """Run episode jobs on a spawn pool.  serial_io: the main process is the ONLY reader of the source
    files (whole-file sequential reads, one at a time, in job order) and ships the bytes to the workers."""
    ctx = mp.get_context("spawn")
    results, io_s, io_bytes, t0, last = [], 0.0, 0, time.time(), time.time()
    max_inflight = workers * 2 if serial_io else len(jobs) + 1
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        pending = set()

        def drain(block_until: int):
            nonlocal pending, last
            while len(pending) > block_until:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fu in done:
                    results.append(fu.result())  # re-raises worker exceptions
                if time.time() - last > 30:
                    last = time.time()
                    el = time.time() - t0
                    log(f"  {label}: {len(results)}/{len(jobs)} episodes, {el:.0f}s elapsed"
                        + (f", reader {io_bytes / 1e9:.1f} GB @ {io_bytes / 1e6 / max(io_s, 1e-9):.0f} MB/s" if serial_io else ""))

        for j in jobs:
            if serial_io:
                t = time.time()
                with open(j["path"], "rb") as fh:
                    data = fh.read()
                io_s += time.time() - t
                io_bytes += len(data)
                pending.add(ex.submit(process_episode, j, data))
                del data
                drain(max_inflight - 1)
            else:
                pending.add(ex.submit(process_episode, j))
        drain(0)
    wall = time.time() - t0
    stats = dict(wall_s=round(wall, 1), episodes=len(jobs), workers=workers, serial_io=serial_io,
                 reader_s=round(io_s, 1) if serial_io else None, reader_bytes=io_bytes if serial_io else None,
                 reader_MBps=round(io_bytes / 1e6 / max(io_s, 1e-9), 1) if serial_io else None,
                 worker_s_total=round(sum(r["secs"] for r in results), 1))
    return results, stats


# -------------------------------------------------------------------------------------- dir handling
def clean_known(d: Path) -> None:
    """Remove exactly the files this script writes, then the (now empty) dirs; never recursive."""
    if not d.exists():
        return
    for n in KNOWN_FILES:
        p = d / n
        if p.is_file() or p.is_symlink():
            p.unlink()
    for sub in (d / "tok", d):
        if sub.exists():
            left = list(sub.iterdir())
            check(not left, f"{sub} still contains unknown files {[x.name for x in left][:5]}; clean it by hand")
            sub.rmdir()


def arrays_info(work: Path) -> dict:
    info = {}
    for n in KNOWN_FILES:
        if n.endswith(".npy") and (work / n).exists():
            a = np.load(work / n, mmap_mode="r")
            info[n] = {"shape": list(a.shape), "dtype": str(a.dtype), "bytes": (work / n).stat().st_size}
    return info


def pick_episodes(eps: list[dict], k: int, prefer: set) -> list[int]:
    """Deterministic pick of k episodes spread over tasks (round robin by task_id); within a task, episodes
    in `prefer` (e.g. those in the current library) first, then by stem."""
    by_task = {}
    for i, e in enumerate(eps):
        by_task.setdefault(e["task_id"], []).append(i)
    for t in by_task:
        by_task[t].sort(key=lambda i: (eps[i]["stem"] not in prefer, eps[i]["stem"]))
    order, j = [], 0
    while len(order) < min(k, len(eps)):
        for t in sorted(by_task):
            if j < len(by_task[t]) and len(order) < k:
                order.append(by_task[t][j])
        j += 1
    return sorted(order)


def summarize_cos(vals, exact, mab) -> dict:
    v = np.asarray(vals, np.float64)
    return {"n": int(v.size), "min_cos": float(v.min()) if v.size else None, "mean_cos": float(v.mean()) if v.size else None,
            "n_bit_exact": int(np.sum(exact)), "all_bit_exact": bool(v.size > 0 and bool(np.all(exact))),
            "max_abs_diff": float(np.max(mab)) if len(mab) else None}


def episodes_meta(res: dict, cfg: dict, tmap: dict, init_by_stem: dict, init_by_eid: dict) -> dict:
    a = res["attrs"]
    task = s_(a["task"])
    check(task in tmap, f"{res['stem']}: task {task!r} not in the trace task map")
    eid = int(a.get("episode_id", -1))
    init = init_by_stem.get(res["stem"])
    warn = None
    if init is None and eid in init_by_eid:
        r = init_by_eid[eid]
        init = int(r["orig_init_state_idx"])
        if bool(r.get("success")) != bool(a["success"]) or int(r.get("task_id", -1)) != tmap[task]:
            warn = f"{res['stem']}: results json (success={r.get('success')}, task_id={r.get('task_id')}) " \
                   f"disagrees with h5 attrs (success={bool(a['success'])}, task_id={tmap[task]})"
            init = None
    return dict(stem=res["stem"], file=str(Path(cfg["build_dir"]) / f"{res['stem']}.h5"), task=task, task_id=tmap[task],
                success=bool(a["success"]), num_steps=int(a["num_steps"]), episode_id=eid,
                orig_init_state_idx=init, timestamp=s_(a.get("timestamp", "")), _warn=warn)


# -------------------------------------------------------------------------------------------- builds
def build_current(cfg, pk, tmap, work: Path, args, inits) -> dict:
    L = len(pk["ids"])
    ver, errors = {}, []
    # ---- ids -> (stem, step), consistency with the pickle's own trajectory fields
    stems, steps = [], np.empty(L, np.int64)
    bad_fields = 0
    for i, id_ in enumerate(pk["ids"]):
        stem, st = id_.rsplit(":", 1)
        stems.append(stem)
        steps[i] = int(st)
        if (pk["traj"][i] is not None and pk["traj"][i] != stem) or (pk["step_idx"][i] is not None and pk["step_idx"][i] != int(st)):
            bad_fields += 1
    ver["id_vs_trajectory_fields_mismatch"] = bad_fields
    if bad_fields:
        errors.append(f"{bad_fields} ids disagree with entry.trajectory_id / step_idx")
    check(len(set(pk["ids"])) == L, "duplicate ids in the pickle")
    unmapped = sorted(set(pk["task"]) - set(tmap))
    check(not unmapped, f"library task strings not in the trace task map: {unmapped}")
    ep_order = list(dict.fromkeys(stems))
    ep_of = {s: i for i, s in enumerate(ep_order)}
    episode = np.array([ep_of[s] for s in stems], np.int32)
    rows_by_ep = [[] for _ in ep_order]
    for r in range(L):
        rows_by_ep[episode[r]].append(r)
    # ---- token subset
    first_task = {}
    for r in range(L):
        first_task.setdefault(episode[r], tmap[pk["task"][r]])
    pseudo = [dict(stem=s, task_id=first_task[i]) for i, s in enumerate(ep_order)]
    tok_eps = list(range(len(ep_order))) if args.limit_episodes is None else pick_episodes(pseudo, args.limit_episodes, set())
    tok_rows = np.array(sorted(r for e in tok_eps for r in rows_by_ep[e]), np.int64)
    tokpos = np.full(L, -1, np.int64)
    tokpos[tok_rows] = np.arange(tok_rows.size)
    M = int(tok_rows.size)
    # ---- write the pickle copies + allocate token arrays
    np.save(work / "key_v0.npy", pk["v0"]); np.save(work / "key_v1.npy", pk["v1"])
    np.save(work / "rs.npy", pk["rs"]); np.save(work / "action.npy", pk["action"])
    np.save(work / "tok" / "rows.npy", tok_rows)
    specs = [("tok/v0.npy", np.float16, (M,) + TOK_SHAPE), ("tok/v1.npy", np.float16, (M,) + TOK_SHAPE)]
    if cfg["images"]:
        specs += [("tok/img0.npy", np.uint8, (M,) + IMG_SHAPE), ("tok/img1.npy", np.uint8, (M,) + IMG_SHAPE)]
    for n, dtp, shp in specs:
        a = new_memmap(work / n, dtp, shp)
        del a
    # ---- jobs (one per source episode; /archive in stem order)
    jobs, unresolved = [], []
    for e, stem in enumerate(ep_order):
        p = Path(cfg["build_dir"]) / f"{stem}.h5"
        if not p.exists():
            unresolved += rows_by_ep[e]
            continue
        rr = rows_by_ep[e]
        jobs.append(dict(mode="current", path=str(p), stem=stem, work=str(work), rows=rr, steps=[int(steps[r]) for r in rr],
                         tokpos=[int(tokpos[r]) for r in rr], H=cfg["H"], Drs=cfg["Drs"], images=cfg["images"],
                         size=p.stat().st_size))
    if unresolved:
        raise CheckError(f"{len(unresolved)} library ids have no build h5 file (first: {pk['ids'][unresolved[0]]})")
    jobs.sort(key=lambda j: j["stem"])
    results, run_stats = run_jobs(jobs, cfg["serial"], args.archive_workers if cfg["serial"] else args.workers, "current")
    res_by = {r["stem"]: r for r in results}
    # ---- metadata
    init_by_stem, init_by_eid, _ = inits
    eps, warns = [], []
    for e, stem in enumerate(ep_order):
        m = episodes_meta(res_by[stem], cfg, tmap, init_by_stem, init_by_eid)
        if m.pop("_warn"):
            warns.append(m["stem"])
        rr = rows_by_ep[e]
        m.update(start=int(min(rr)), end=int(max(rr)) + 1, n_rows=len(rr), contiguous=bool(max(rr) - min(rr) + 1 == len(rr)),
                 n_step_groups=int(res_by[stem]["n_groups"]))
        eps.append(m)
    missing = sorted(r for res in results for r in res["missing"])
    ver["ids_unresolved"] = len(missing)
    if missing:
        errors.append(f"{len(missing)} ids point at a step group missing in the h5 (first {pk['ids'][missing[0]]})")
    task_mis = [r for r in range(L) if pk["task"][r] != eps[episode[r]]["task"]]
    ver["task_key_vs_h5_task_mismatch"] = len(task_mis)
    if task_mis:
        errors.append(f"{len(task_mis)} rows: pickle task_key != h5 task attr")
    groups_bad = [e["stem"] for e in eps if e["n_step_groups"] != e["num_steps"]]
    ver["episodes_num_steps_vs_groups_mismatch"] = len(groups_bad)
    task_id = np.array([tmap[t] for t in pk["task"]], np.int16)
    ep_len = np.array([eps[episode[r]]["num_steps"] for r in range(L)], np.int16)
    success = np.array([eps[episode[r]]["success"] for r in range(L)], bool)
    step16 = steps.astype(np.int16)
    progress = np.where(ep_len > 1, steps / np.maximum(ep_len.astype(np.float64) - 1, 1), 0.0).astype(np.float32)
    row_of = {(int(episode[r]), int(steps[r])): r for r in range(L)}
    prev = np.array([row_of.get((int(episode[r]), int(steps[r]) - 1), -1) for r in range(L)], np.int32)
    nxt = np.array([row_of.get((int(episode[r]), int(steps[r]) + 1), -1) for r in range(L)], np.int32)
    # ---- prev/next vs the pickle's linked list
    id_row = {x: i for i, x in enumerate(pk["ids"])}

    def link_row(lst):
        return id_row.get(lst[0], -2) if lst else -1  # -2: dangling (points outside the library)

    pk_prev = np.array([link_row(x) for x in pk["prev"]], np.int64)
    pk_next = np.array([link_row(x) for x in pk["next"]], np.int64)
    multi = sum(len(x) > 1 for x in pk["prev"]) + sum(len(x) > 1 for x in pk["next"])
    ver["prev_next_vs_pickle"] = dict(prev_mismatch=int(np.sum(pk_prev != prev)), next_mismatch=int(np.sum(pk_next != nxt)),
                                      dangling_prev=int(np.sum(pk_prev == -2)), dangling_next=int(np.sum(pk_next == -2)),
                                      multi_link_entries=int(multi))
    if np.any(pk_prev != prev) or np.any(pk_next != nxt) or multi:
        errors.append(f"prev/next disagree with the pickle linked list: {ver['prev_next_vs_pickle']}")
    order_ok = all(np.all(np.diff(steps[rr]) > 0) for rr in rows_by_ep)
    ver["steps_increasing_within_episode"] = bool(order_ok)
    if not order_ok:
        errors.append("pickle rows of an episode are not in increasing step order")
    # ---- key / rs / action checks
    ver["keys_vs_pool16_of_h5_tokens"] = {
        "v0": summarize_cos([c for r in results for c in r["cos0"]], [x for r in results for x in r["ex0"]], [x for r in results for x in r["mab0"]]),
        "v1": summarize_cos([c for r in results for c in r["cos1"]], [x for r in results for x in r["ex1"]], [x for r in results for x in r["mab1"]])}
    for k in ("v0", "v1"):
        s = ver["keys_vs_pool16_of_h5_tokens"][k]
        if s["n"] != L or s["min_cos"] < COS_TOL:
            errors.append(f"key_{k}: {s['n']}/{L} rows checked, min cos {s['min_cos']} < {COS_TOL}")
    ver["rs_vs_h5_max_abs_diff"] = max(r["rsd"] for r in results)
    ver["action_vs_h5_clean_action_max_abs_diff"] = max(r["acd"] for r in results)
    ver["action_vs_h5_clean_action_max_abs_diff_exec_5x7"] = max(r["acd_exec"] for r in results)
    if ver["rs_vs_h5_max_abs_diff"] > EXACT_TOL or ver["action_vs_h5_clean_action_max_abs_diff"] > EXACT_TOL:
        errors.append(f"rs/action differ from h5 (rs {ver['rs_vs_h5_max_abs_diff']}, action {ver['action_vs_h5_clean_action_max_abs_diff']})")
    ver["n_trajectories"] = len(ep_order)
    ver["all_source_episodes_success"] = bool(all(e["success"] for e in eps))
    ver["n_source_episodes_failed"] = sum(not e["success"] for e in eps)
    ver["n_episodes_fully_included"] = sum(e["n_rows"] == e["num_steps"] for e in eps)
    ver["n_episodes_contiguous_rows"] = sum(e["contiguous"] for e in eps)
    ver["rows_per_task"] = {int(t): int(np.sum(task_id == t)) for t in range(10)}
    ver["episodes_per_task"] = {int(t): sum(e["task_id"] == t for e in eps) for t in range(10)}
    ver["init_lookup_disagreements"] = warns
    ver["tok_rows_written"] = sum(r["tok_rows"] for r in results)
    if ver["tok_rows_written"] != M:
        errors.append(f"tok rows written {ver['tok_rows_written']} != M={M}")
    arrays = dict(task_id=task_id, episode=episode, step=step16, ep_len=ep_len, progress=progress, success=success,
                  prev=prev, next=nxt)
    return dict(L=L, M=M, eps=eps, ids=pk["ids"], arrays=arrays, ver=ver, errors=errors, run=run_stats,
                tok_episodes=[ep_order[e] for e in tok_eps])


def scan_build(cfg) -> list[dict]:
    """Pass 1 over the build dir: file attrs only (one serial reader; fine for /archive)."""
    files = sorted(glob.glob(str(Path(cfg["build_dir"]) / "*.h5")))
    check(files, f"no build h5 in {cfg['build_dir']}")
    out = []
    for p in files:
        with h5py.File(p, "r") as f:
            a = f.attrs
            out.append(dict(stem=Path(p).stem, path=p, task=s_(a["task"]), num_steps=int(a["num_steps"]),
                            success=bool(a["success"]), size=os.path.getsize(p)))
    return out


def scan_cs(cfg) -> list[dict]:
    """Row plan for bpool_cs from the collect ledger (no h5 attr scan on the HDD); the directory listing and
    file sizes (metadata only) must match the ledger exactly."""
    cs = cfg["cs"]
    led = json.loads(Path(cs["ledger"]).read_text())
    files = led["files"]
    root = Path(cs["dir"])
    listing = {f"{td.name}/{p.name}": p.stat().st_size for td in sorted(root.glob("task_*")) for p in td.glob("*.h5")}
    check(set(listing) == set(files), f"ledger vs listing of {root}: only-ledger {sorted(set(files) - set(listing))[:5]}, "
                                      f"only-dir {sorted(set(listing) - set(files))[:5]}")
    bad_size = [k for k, v in files.items() if int(v["bytes"]) != listing[k]]
    check(not bad_size, f"file size != ledger bytes for {bad_size[:5]}")
    out = []
    for rel, v in files.items():
        tdir, fname = rel.split("/")
        check(tdir.startswith("task_") and fname.startswith("episode_") and fname.endswith(".h5"), f"unexpected path {rel}")
        tn, m = int(tdir[len("task_"):]), int(fname[len("episode_"):-len(".h5")])
        out.append(dict(stem=rel[:-len(".h5")], path=str(root / rel), task=None, task_id=tn, task_n=tn, init=m,
                        num_steps=int(v["num_steps"]), success=bool(v["success"]), size=int(v["bytes"]),
                        sha256=v.get("sha256")))
    out.sort(key=lambda e: (e["task_n"], e["init"]))
    return out


def episodes_meta_cs(res: dict, e: dict, tmap: dict, results: dict) -> dict:
    """episodes.json row for a cache_size episode + list of disagreements with ledger / results / path."""
    a = res["attrs"]
    task = s_(a["task"])
    check(task in tmap, f"{e['stem']}: task {task!r} not in the trace task map")
    tid, bad = tmap[task], []
    if tid != e["task_n"]:
        bad.append(f"path task_{e['task_n']} but task attr maps to task_id {tid}")
    if int(a["num_steps"]) != e["num_steps"] or bool(a["success"]) != e["success"]:
        bad.append(f"attrs num_steps/success {int(a['num_steps'])}/{bool(a['success'])} != ledger {e['num_steps']}/{e['success']}")
    if e.get("sha256") and res.get("sha_ok") is not True:
        bad.append("sha256 of the bytes read != ledger sha256")
    r = results.get((e["task_n"], e["init"]))
    if r is None:
        bad.append("no row in collect_results.json")
    elif bool(r["success"]) != bool(a["success"]) or int(r.get("orig_init_state_idx", -1)) != e["init"]:
        bad.append(f"collect_results success/orig_init {r['success']}/{r.get('orig_init_state_idx')} disagree")
    return dict(stem=e["stem"], file=e["path"], task=task, task_id=tid, success=bool(a["success"]),
                num_steps=int(a["num_steps"]), episode_id=int(a.get("episode_id", -1)), orig_init_state_idx=e["init"],
                timestamp=s_(a.get("timestamp", "")), _warn=bad)


def build_bpool(cfg, pk, tmap, work: Path, args, inits, source: str = "build") -> dict:
    """source 'build' -> bpool_all (the build dir; overlap reference = the current pickle);
    source 'cs' -> bpool_cs (cache_size corpus; overlap reference = the cache_size all_S3 pickle)."""
    ver, errors = {}, []
    t = time.time()
    cs = source == "cs"
    if cs:
        scan = scan_cs(cfg)
        cs_results = {(int(r["task_id"]), int(r["init_state_idx"])): r
                      for r in json.loads(Path(cfg["cs"]["results"]).read_text())}
        log(f"  loading overlap reference {cfg['cs']['ref']}")
        ref = load_pickle(cfg, cfg["cs"]["ref"])
        ref_label, serial = "cs_ref", True
    else:
        scan = scan_build(cfg)
        unmapped = sorted({e["task"] for e in scan} - set(tmap))
        check(not unmapped, f"build task strings not in the trace task map: {unmapped}")
        for e in scan:
            e["task_id"] = tmap[e["task"]]
        ref, ref_label, serial = pk, "current", cfg["serial"]
    ver["scan_s"] = round(time.time() - t, 1)
    ref_stems = {x.rsplit(":", 1)[0] for x in ref["ids"]}
    sel = list(range(len(scan))) if args.limit_episodes is None else pick_episodes(scan, args.limit_episodes, ref_stems)
    scan = [scan[i] for i in sel]  # keeps the scan order
    starts = np.cumsum([0] + [e["num_steps"] for e in scan])
    L = int(starts[-1])
    H, Drs = cfg["H"], cfg["Drs"]
    np.save(work / "tok" / "rows.npy", np.arange(L, dtype=np.int64))
    specs = [("key_v0.npy", np.float32, (L, KEY_DIM)), ("key_v1.npy", np.float32, (L, KEY_DIM)),
             ("rs.npy", np.float32, (L, Drs)), ("action.npy", np.float32, (L, H, 32)),
             ("tok/v0.npy", np.float16, (L,) + TOK_SHAPE), ("tok/v1.npy", np.float16, (L,) + TOK_SHAPE)]
    if cfg["images"]:
        specs += [("tok/img0.npy", np.uint8, (L,) + IMG_SHAPE), ("tok/img1.npy", np.uint8, (L,) + IMG_SHAPE)]
    for n, dtp, shp in specs:
        a = new_memmap(work / n, dtp, shp)
        del a
    jobs = []
    for i, e in enumerate(scan):
        rr = list(range(int(starts[i]), int(starts[i + 1])))
        jobs.append(dict(mode="bpool", path=e["path"], stem=e["stem"], work=str(work), rows=rr, steps=list(range(e["num_steps"])),
                         tokpos=rr, H=H, Drs=Drs, images=cfg["images"], size=e["size"], sha256=e.get("sha256")))
    results, run_stats = run_jobs(jobs, serial, args.archive_workers if serial else args.workers, f"bpool_{source}")
    res_by = {r["stem"]: r for r in results}
    init_by_stem, init_by_eid, _ = inits
    eps, warns, cs_bad = [], [], []
    for i, e in enumerate(scan):
        if cs:
            m = episodes_meta_cs(res_by[e["stem"]], e, tmap, cs_results)
            bad = m.pop("_warn")
            cs_bad += [f"{e['stem']}: {b}" for b in bad]
        else:
            m = episodes_meta(res_by[e["stem"]], cfg, tmap, init_by_stem, init_by_eid)
            if m.pop("_warn"):
                warns.append(m["stem"])
            check(m["task"] == e["task"], f"{e['stem']}: task attr changed between passes")
        check(m["num_steps"] == e["num_steps"], f"{e['stem']}: num_steps {m['num_steps']} != planned {e['num_steps']}")
        m.update(start=int(starts[i]), end=int(starts[i + 1]), n_rows=e["num_steps"], contiguous=True,
                 n_step_groups=int(res_by[e["stem"]]["n_groups"]))
        eps.append(m)
    missing = sum(len(r["missing"]) for r in results)
    groups_bad = [e["stem"] for e in eps if e["n_step_groups"] != e["num_steps"]]
    ver["row_count"] = L
    ver["sum_num_steps"] = int(sum(e["num_steps"] for e in eps))
    ver["sum_step_groups"] = int(sum(e["n_step_groups"] for e in eps))
    ver["missing_step_groups"] = missing
    if missing or groups_bad or ver["sum_step_groups"] != L:
        errors.append(f"row count / step groups mismatch: L={L}, groups={ver['sum_step_groups']}, missing={missing}, "
                      f"bad episodes={groups_bad[:5]}")
    episode = np.repeat(np.arange(len(eps), dtype=np.int32), [e["num_steps"] for e in eps])
    step = np.concatenate([np.arange(e["num_steps"]) for e in eps]).astype(np.int64)
    ep_len = np.repeat(np.array([e["num_steps"] for e in eps], np.int16), [e["num_steps"] for e in eps])
    task_id = np.repeat(np.array([e["task_id"] for e in eps], np.int16), [e["num_steps"] for e in eps])
    success = np.repeat(np.array([e["success"] for e in eps], bool), [e["num_steps"] for e in eps])
    progress = np.where(ep_len > 1, step / np.maximum(ep_len.astype(np.float64) - 1, 1), 0.0).astype(np.float32)
    rows = np.arange(L, dtype=np.int64)
    prev = np.where(step > 0, rows - 1, -1).astype(np.int32)
    nxt = np.where(step < ep_len - 1, rows + 1, -1).astype(np.int32)
    ids = [f"{eps[episode[r]]['stem']}:{int(step[r])}" for r in range(L)]
    if cs:
        ver["cs_ledger_results_path_sha_disagreements"] = cs_bad
        ver["cs_sha256_checked"] = sum(1 for r in results if "sha_ok" in r)
        ver["cs_sha256_ok"] = sum(1 for r in results if r.get("sha_ok") is True)
        if cs_bad:
            errors.append(f"{len(cs_bad)} cache_size ledger/results/path/sha disagreements (first: {cs_bad[0]})")
    # ---- overlap with the reference pickle (current library / cache_size all_S3): recomputed keys vs pickle
    id_row = {x: i for i, x in enumerate(ids)}
    pairs = [(id_row[x], i) for i, x in enumerate(ref["ids"]) if x in id_row]
    ver[f"{ref_label}_ids_in_bpool"] = len(pairs)
    ver[f"{ref_label}_ids_total"] = len(ref["ids"])
    cur_in_sel = sum(1 for x in ref["ids"] if x.rsplit(":", 1)[0] in {e["stem"] for e in eps})
    if len(pairs) != cur_in_sel:
        errors.append(f"only {len(pairs)} of the {cur_in_sel} {ref_label} ids of the selected episodes found in the store")
    if cs:
        ver["cs_ref_episodes_in_store"] = len({x.rsplit(":", 1)[0] for x in ref["ids"]} & {e["stem"] for e in eps})
        if not pairs:
            errors.append("no overlap with the cache_size reference pickle: key recomputation is unchecked")
    K0, K1 = np.load(work / "key_v0.npy", mmap_mode="r"), np.load(work / "key_v1.npy", mmap_mode="r")
    RS, AC = np.load(work / "rs.npy", mmap_mode="r"), np.load(work / "action.npy", mmap_mode="r")
    ov = {}
    for k, K in (("v0", K0), ("v1", K1)):
        cl, ex, mab = [], [], []
        for br, cr in pairs:
            a, b = np.asarray(K[br]), ref[k][cr]
            cl.append(cos64(a, b)); ex.append(bool(np.array_equal(a, b))); mab.append(float(np.abs(a - b).max()))
        ov[k] = summarize_cos(cl, ex, mab)
        if pairs and ov[k]["min_cos"] < COS_TOL:
            errors.append(f"bpool key_{k} vs {ref_label} pickle: min cos {ov[k]['min_cos']} < {COS_TOL}")
    ver[f"keys_vs_{ref_label}_pickle_on_overlap"] = ov
    if pairs:
        br, cr = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
        ver[f"rs_vs_{ref_label}_max_abs_diff"] = float(np.abs(RS[br] - ref["rs"][cr]).max())
        d = np.abs(AC[br] - ref["action"][cr])
        ver[f"action_vs_{ref_label}_max_abs_diff"] = float(d.max())
        ver[f"action_vs_{ref_label}_max_abs_diff_exec_5x7"] = float(d[:, :5, :7].max())
        if ver[f"rs_vs_{ref_label}_max_abs_diff"] > EXACT_TOL or ver[f"action_vs_{ref_label}_max_abs_diff"] > EXACT_TOL:
            errors.append(f"bpool rs/action differ from the {ref_label} pickle on overlapping ids")
        tk_bad = sum(ref["task"][c] != eps[episode[b]]["task"] for b, c in pairs)
        ver[f"task_vs_{ref_label}_mismatch"] = int(tk_bad)
        if tk_bad:
            errors.append(f"{tk_bad} overlapping rows: task attr != {ref_label} task_key")
    fin = [bool(np.isfinite(np.asarray(A[s:s + 4096])).all()) for A in (K0, K1, RS, AC) for s in range(0, L, 4096)]
    ver["finite_keys_rs_action"] = bool(all(fin))
    if not all(fin):
        errors.append("non-finite values in keys/rs/action")
    ver["n_episodes"] = len(eps)
    ver["n_episodes_success"] = sum(e["success"] for e in eps)
    ver["episodes_per_task"] = {int(t): sum(e["task_id"] == t for e in eps) for t in range(10)}
    ver["rows_per_task"] = {int(t): int(np.sum(task_id == t)) for t in range(10)}
    ver["init_lookup_disagreements"] = warns
    ver["tok_rows_written"] = sum(r["tok_rows"] for r in results)
    if ver["tok_rows_written"] != L:
        errors.append(f"tok rows written {ver['tok_rows_written']} != L={L}")
    arrays = dict(task_id=task_id, episode=episode, step=step.astype(np.int16), ep_len=ep_len, progress=progress,
                  success=success, prev=prev, next=nxt)
    extra = {}
    if cs:
        extra = dict(ref_pkl=str(cfg["cs"]["ref"]), ref_meta=ref["meta"], ref_sha256=sha256_file(cfg["cs"]["ref"]))
    return dict(L=L, M=L, eps=eps, ids=ids, arrays=arrays, ver=ver, errors=errors, run=run_stats,
                tok_episodes=[e["stem"] for e in eps], extra=extra)


# ---------------------------------------------------------------------------------------------- main
def build_one(lib: str, name: str, args, cache: dict) -> bool:
    cfg = LIBS[lib]
    root = Path(args.out) / lib
    final, work, err = root / name, root / f"{name}.partial", root / f"{name}.ERROR"
    if (final / "manifest.json").exists() and not args.force:
        st = json.loads((final / "manifest.json").read_text()).get("status")
        if st == "ok":
            log(f"{lib}/{name}: complete, skip (use --force to rebuild)")
            return True
    t0 = time.time()
    started = stamp()
    root.mkdir(parents=True, exist_ok=True)
    if err.exists():
        err.unlink()
    clean_known(work)
    if final.exists():
        clean_known(final)
    work.mkdir(parents=True)
    (work / "tok").mkdir()
    try:
        if lib not in cache:
            log(f"{lib}: loading pickle {cfg['pkl']}")
            pk = load_pickle(cfg)
            tmap, shas, nfiles = task_map_for(cfg)
            cache[lib] = dict(pk=pk, tmap=tmap, shas=shas, nfiles=nfiles, pkl_sha=sha256_file(cfg["pkl"]),
                              audit=audit_equal(cfg, pk), inits=init_lookup(cfg))
            log(f"{lib}: pickle {len(pk['ids'])} entries (load {pk['load_s']}s), task map from {nfiles}")
        c = cache[lib]
        pk, tmap = c["pk"], c["tmap"]
        log(f"{lib}/{name}: building into {work}")
        if name == "current":
            b = build_current(cfg, pk, tmap, work, args, c["inits"])
        else:
            b = build_bpool(cfg, pk, tmap, work, args, c["inits"], source="cs" if name == "bpool_cs" else "build")
        ver, errors = b["ver"], b["errors"]
        ver["pkl_sha256"] = c["pkl_sha"]
        ver["trace_library_sha256"] = sorted(c["shas"])
        ver["pkl_sha256_matches_trace"] = c["shas"] == {c["pkl_sha"]}
        if not ver["pkl_sha256_matches_trace"]:
            errors.append(f"pickle sha256 {c['pkl_sha']} != trace attr trace_library_sha256 {sorted(c['shas'])}")
        if name == "current":
            ver["audit_memmaps_equal"] = c["audit"]
            if c["audit"].get("available") and not all(v for k, v in c["audit"].items() if k != "available"):
                errors.append(f"pickle copy differs from the audit memmaps: {c['audit']}")
        for k, a in b["arrays"].items():
            np.save(work / f"{k}.npy", a)
        (work / "ids.json").write_text(json.dumps(b["ids"]))
        (work / "episodes.json").write_text(json.dumps(b["eps"], indent=0))
        L, M = b["L"], b["M"]
        ts = sorted(e["timestamp"] for e in b["eps"] if e.get("timestamp"))
        cs = name == "bpool_cs"
        sources = dict(pkl=str(cfg["pkl"]), pkl_sha256=c["pkl_sha"], pickle_meta=pk["meta"],
                       build_dir=str(cfg["build_dir"]), build_dir_realpath=os.path.realpath(cfg["build_dir"]),
                       build_on_archive_hdd=bool(cfg["serial"]), audit_memmaps=str(AUDIT / cfg["audit"]),
                       init_source=c["inits"][2])
        if cs:
            sources = dict(cs_dir=str(cfg["cs"]["dir"]), cs_dir_realpath=os.path.realpath(cfg["cs"]["dir"]),
                           on_archive_hdd=True, ledger=str(cfg["cs"]["ledger"]), collect_results=str(cfg["cs"]["results"]),
                           overlap_ref_pkl=b["extra"]["ref_pkl"], overlap_ref_sha256=b["extra"]["ref_sha256"],
                           overlap_ref_meta=b["extra"]["ref_meta"], init_source="path episode_M (= B-pool orig_init_state_idx)",
                           task_map_pickle=str(cfg["pkl"]), task_map_pickle_sha256=c["pkl_sha"])
        man = dict(
            schema="offline_search.library.v1", lib=lib, name=name, model=cfg["model"], suite=cfg["suite"],
            suite_full=cfg["suite_full"], status="ok" if not errors else "error", errors=errors,
            complete=(args.limit_episodes is None and M == L), tok_complete=(M == L), limit_episodes=args.limit_episodes,
            counts=dict(rows=L, episodes=len(b["eps"]), tok_rows=M, tok_episodes=len(b["tok_episodes"]),
                        success_episodes=sum(e["success"] for e in b["eps"]), tasks=len({e["task_id"] for e in b["eps"]})),
            H=cfg["H"], Drs=cfg["Drs"], key_dim=KEY_DIM, tok_shape=list(TOK_SHAPE),
            img_shape=list(IMG_SHAPE) if cfg["images"] else None,
            img_source_keys={"img0": "input_images/base_0_rgb", "img1": "input_images/left_wrist_0_rgb"} if cfg["images"] else None,
            **FACTS, facts_note=FACTS_NOTE,
            key_source=("copied bit-exactly from the pickle entries (query_keys vision_0/vision_1/robot_state, payload.action_chunk)"
                        if name == "current" else
                        "recomputed: tokens.astype(f32).reshape(4,4,4,4,2048).mean(axis=(1,3)).reshape(-1); rs=robot_state; action=clean_action"),
            tasks={str(v): k for k, v in sorted(tmap.items(), key=lambda kv: kv[1])}, task_map=tmap,
            task_map_source={"trace_runs": list(cfg["trace_runs"]), "n_files": c["nfiles"], "attr": "task -> task_id"},
            tok_episodes=b["tok_episodes"],
            id_scheme="task_N/episode_M:<step> (relative path; bare stems repeat across tasks)" if cs else "<episode_stem>:<step>",
            episode_order="(task N, init M)" if cs else ("pickle order" if name == "current" else "file stem"),
            batch=dict(label=CS_BATCH_LABEL if cs else cfg.get("batch_label"), timestamp_min=ts[0] if ts else None,
                       timestamp_max=ts[-1] if ts else None),
            sources=sources,
            verification=ver, run=b["run"],
            build=dict(started=started, finished=stamp(), wall_s=round(time.time() - t0, 1), host=socket.gethostname(),
                       argv=sys.argv, workers=args.archive_workers if (cfg["serial"] or cs) else args.workers))
        man["arrays"] = arrays_info(work)
        man["bytes_total"] = sum(v["bytes"] for v in man["arrays"].values())
        (work / "manifest.json").write_text(json.dumps(man, indent=1, default=py))
        if errors:
            err.write_text("verification failed:\n" + "\n".join(errors) + f"\nsee {work / 'manifest.json'}\n")
            log(f"{lib}/{name}: VERIFICATION FAILED -> {err}")
            for e in errors:
                log(f"    {e}")
            return False
        os.rename(work, final)
        log(f"{lib}/{name}: OK  L={L} M={M} episodes={len(b['eps'])} {man['bytes_total'] / 1e9:.2f} GB "
            f"{time.time() - t0:.0f}s -> {final}")
        return True
    except Exception:
        tb = traceback.format_exc()
        err.write_text(tb)
        log(f"{lib}/{name}: ERROR\n{tb}")
        return False


def dry_run(libs, args) -> None:
    for lib in libs:
        cfg = LIBS[lib]
        meta = json.loads((AUDIT / f"{cfg['audit']}.meta.json").read_text())
        Lc = len(meta["ids"])
        t = time.time()
        scan = scan_build(cfg)
        Lb = sum(e["num_steps"] for e in scan)
        per_row = 2 * KEY_DIM * 4 + cfg["Drs"] * 4 + cfg["H"] * 32 * 4 + 2 * 256 * 2048 * 2 + (2 * 224 * 224 * 3 if cfg["images"] else 0)
        log(f"{lib}: current L={Lc} (~{Lc * per_row / 1e9:.1f} GB) | bpool_all {len(scan)} episodes "
            f"({sum(e['success'] for e in scan)} success) L={Lb} (~{Lb * per_row / 1e9:.1f} GB); "
            f"source {sum(e['size'] for e in scan) / 1e9:.1f} GB; attr scan {time.time() - t:.0f}s")
        if "bpool_cs" in args.names.split(",") and cfg.get("cs"):
            t = time.time()
            sc = scan_cs(cfg)
            Ls = sum(e["num_steps"] for e in sc)
            log(f"{lib}: bpool_cs {len(sc)} episodes ({sum(e['success'] for e in sc)} success) L={Ls} "
                f"(~{Ls * per_row / 1e9:.1f} GB); source {sum(e['size'] for e in sc) / 1e9:.1f} GB on /archive; "
                f"ledger+listing {time.time() - t:.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--libs", default=",".join(LIBS), help="comma list of " + ",".join(LIBS))
    ap.add_argument("--names", default=",".join(CORE_NAMES),
                    help="comma list of current,bpool_all,bpool_cs (bpool_cs: pi05 only, not in the default)")
    ap.add_argument("--limit-episodes", type=int, default=None,
                    help="smoke: current -> tokens only for rows of K episodes (keys/meta stay complete); "
                         "bpool_all / bpool_cs -> only K episodes (spread over tasks, episodes of the overlap "
                         "reference pickle first)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="library root (…/offline_search_store/library)")
    ap.add_argument("--force", action="store_true", help="rebuild libraries that are already complete")
    ap.add_argument("--workers", type=int, default=16, help="pool size for SSD sources")
    ap.add_argument("--archive-workers", type=int, default=8, help="pool size for /archive sources (single reader)")
    ap.add_argument("--dry-run", action="store_true", help="only scan attrs and print row counts / size estimates")
    args = ap.parse_args()
    libs = [x for x in args.libs.split(",") if x]
    names = [x for x in args.names.split(",") if x]
    bad = [x for x in libs if x not in LIBS] + [x for x in names if x not in NAMES]
    if bad:
        ap.error(f"unknown libs/names: {bad}")
    if args.dry_run:
        dry_run(libs, args)
        return 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ready = out / "READY"
    if ready.exists():
        ready.unlink()
    log(f"build_library_store: libs={libs} names={names} limit={args.limit_episodes} out={out} force={args.force}")
    ok, cache = True, {}
    for lib in libs:
        for name in names:
            if name == "bpool_cs" and not LIBS[lib].get("cs"):
                log(f"{lib}/bpool_cs: not defined (cache_size corpus is pi05 only), skip")
                continue
            ok &= build_one(lib, name, args, cache)
        cache.pop(lib, None)
    done = {f"{lib}/{name}": (out / lib / name / "manifest.json").exists() for lib in LIBS for name in CORE_NAMES}
    if all(done.values()):
        libs_info = {}
        extra = [f"{lib}/bpool_cs" for lib in LIBS if (out / lib / "bpool_cs" / "manifest.json").exists()]
        for k in sorted(done) + extra:
            m = json.loads((out / k / "manifest.json").read_text())
            libs_info[k] = dict(counts=m["counts"], complete=m["complete"], tok_complete=m.get("tok_complete"),
                                limit_episodes=m["limit_episodes"],
                                bytes_total=m["bytes_total"], finished=m["build"]["finished"]["local"])
        ready.write_text(json.dumps(dict(written=stamp(), libraries=libs_info), indent=1))
        log(f"all {len(done)} libraries complete -> {ready}")
    else:
        log(f"not complete yet: {[k for k, v in done.items() if not v]}")
    log("exit", 0 if ok else 1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
