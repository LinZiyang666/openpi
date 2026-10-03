"""Closed-loop KPI tool: screen retrieval methods by their closed-loop logs (R3; offline mean err does not rank SR).

    .venv/bin/python -m exp.offline_search.closed_loop.ops.kpi --run-root R [--run-root R2 ...] <arm> [<arm> ...]
        [--ref <arm | run:arm>] [--chain] [--pilot | --tasks 6,9,0,4,1 --episodes 0-19] [--json OUT] [--md OUT]
        [--store ROOT] [--act-eps 0.1] [--ir-ref <arm | run:arm> | --ir-ref-ms MS] [--recon METHOD=RULE ...]
        [--workers 8] [--boot 10000] [--seed 0] [--no-client] [--quiet]

Inputs (read-only): <R>/arms.json, <R>/runs/<arm>/client/journal.jsonl (completion judge = collect.py: accepted &
status in {done, failed} & no error), <R>/runs/<arm>/client/per_step.jsonl (client hit mix; `_kind=client_timing` rows
excluded), <R>/runs/<arm>/server_*/decisions_*.jsonl (one `dec` row per decision), the library arrays of the store
(`library/<model>_<suite>/<lib>/`) for the served-action reconstruction and the row taxonomy.

Per arm: SR + Wilson 95 % CI, per-task SR, decisions / episode (success / fail), identical-pick spells (>= 3 consecutive
served decisions with the same top-1; a MISS breaks the run), served-action repeat spells (consecutive served heads
within `--act-eps` sigma-RMS), share of decisions inside spells, longest run, gripper flips / episode, first-spell
class shares T / G / Z / P / H (ideation A), weight on terminal rows in the last third, effective library episodes of
the kernel set (ideation C), native agreement. Paired vs `--ref` (or the previous arm with `--chain`) on the common
(task, ep_idx) set: S->F / F->S, delta SR, exact McNemar p, multinomial bootstrap and Newcombe (1998, method 10)
95 % intervals, per-task delta. Mixed arms (decision rows with `hit` / `judge` / `src` ...): realized h overall / per
regime / per task, IR (pi0.5 formula 0.152 + 0.848 (1 - h); measured ratios when a reference is given), MISS share in
failed vs successful episodes, reason codes, first MISS vs first spell, successful episodes interrupted. Missing fields
are tolerated (pure-cache logs have none of the mixed fields).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pathlib
import re
import sys
import time
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402

from exp.offline_search.closed_loop.ops.collect import arm_manifest
from exp.offline_search.closed_loop.devset import root_pool, check_manifest_pool, validate_journal_pool
from exp.offline_search.rounds.r04.k4_eval.cost_ledger import ledger, r4_enabled
from exp.offline_search.rounds.r04.k4_eval.estimators import design_estimate

SCHEMA = "offline_search.closed_loop.kpi.v1"
TOPK = 10
EXEC_STEPS, ACT_DIMS, GRIP = 5, 7, 6
PILOT_TASKS = {"spatial": [6, 9, 0, 4, 1], "l10": [0, 4, 6, 8, 7]}
PILOT_EPISODES = list(range(20))
IR_PI05 = (0.152, 0.848)                     # IR = a + b * (1 - h)  (project definition, pi0.5 CUDA-graph 3 stages)
REASON_NAMES = {1: "stuck", 2: "terminal_closed_gripper", 3: "overtime_lag", 4: "no_progress", 5: "dispersion",
                6: "ambiguous_gripper", 7: "burst"}
SUITE_SHORT = {"libero_spatial": "spatial", "libero_10": "l10", "spatial": "spatial", "l10": "l10", "sp": "spatial"}
MODEL_OF_LETTER = {"p": "pi05", "g": "groot"}
CLASS_ORDER = ("T", "G", "Z", "P", "H")
CLASS_LONG = {"T": "terminal_row", "G": "gripper_split", "Z": "near_zero_translation", "P": "pause_row", "H": "hub"}
EXEC_KEYS = ("served_head", "a_exec", "exec_chunk", "exec", "policy_chunk", "act_exec", "served")
Z95 = 1.959963984540054


# ----------------------------------------------------------------------------------------------------- small helpers
def _fmt(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "nan"
    if isinstance(x, (float, np.floating)):
        return f"{x:.{nd}f}"
    return str(x)


def _mean(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


def _median(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def _share(mask, of=None):
    mask = np.asarray(mask, bool)
    if of is not None:
        of = np.asarray(of, bool)
        return float(mask[of].mean()) if of.any() else float("nan")
    return float(mask.mean()) if mask.size else float("nan")


def wilson(k: int, n: int, z: float = Z95):
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p (binomial test on the discordant pairs, p = 1/2)."""
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(m, i) for i in range(k + 1))
    p = 2.0 * tail / (2 ** m)
    return float(min(1.0, p))


def newcombe_paired(a: int, b: int, c: int, d: int, z: float = Z95):
    """Newcombe (1998) method 10 interval for p_arm - p_ref; a = both S, b = ref S & arm F, c = ref F & arm S, d = both F."""
    n = a + b + c + d
    if n == 0:
        return (float("nan"), float("nan"))
    p_arm, p_ref = (a + c) / n, (a + b) / n
    l1, u1 = wilson(a + c, n, z)
    l2, u2 = wilson(a + b, n, z)
    A = a * d - b * c
    B = (a + b) * (c + d) * (a + c) * (b + d)
    if B <= 0:
        phi = 0.0
    else:
        if A > 0:
            A = max(A - n / 2.0, 0.0)
        phi = A / math.sqrt(B)
    diff = p_arm - p_ref
    lo = diff - math.sqrt(max(0.0, (p_arm - l1) ** 2 - 2 * phi * (p_arm - l1) * (u2 - p_ref) + (u2 - p_ref) ** 2))
    hi = diff + math.sqrt(max(0.0, (u1 - p_arm) ** 2 - 2 * phi * (u1 - p_arm) * (p_ref - l2) + (p_ref - l2) ** 2))
    return (float(lo), float(hi))


def parse_int_list(s: str | None):
    if s is None or s == "":
        return None
    out = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok and not tok.startswith("-"):
            a, b = tok.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return sorted(set(out))


def _jsonl(path: pathlib.Path):
    if not path.exists():
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _to_native(o):
    if isinstance(o, dict):
        return {str(k): _to_native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_native(v) for v in o]
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _to_native(o.tolist())
    return o


# ----------------------------------------------------------------------------------------------------- libraries
class Library:
    """Arrays of one stored library (`library/<model>_<suite>/<name>`), read-only memmaps, valid block [:5, :7]."""

    def __init__(self, root: pathlib.Path, model: str, suite: str, name: str, cache_dir: pathlib.Path | None = None):
        self.root, self.model, self.suite, self.name = root, model, suite, name
        d = root / "library" / f"{model}_{suite}" / name
        if not d.is_dir():
            raise FileNotFoundError(f"library dir {d} not found (pass --store)")
        self.dir = d
        act = np.load(d / "action.npy", mmap_mode="r")
        self.H = int(act.shape[1])
        self.head = np.ascontiguousarray(act[:, :EXEC_STEPS, :ACT_DIMS]).astype(np.float64)   # (L, 5, 7)
        self.ep = np.load(d / "episode.npy").astype(np.int64)
        self.step = np.load(d / "step.npy").astype(np.int64)
        self.ep_len = np.load(d / "ep_len.npy").astype(np.int64)
        self.nxt = np.load(d / "next.npy").astype(np.int64)
        self.task = np.load(d / "task_id.npy").astype(np.int64)
        self.prog = np.load(d / "progress.npy").astype(np.float64) if (d / "progress.npy").exists() else \
            (self.step / np.maximum(self.ep_len - 1, 1)).astype(np.float64)
        self.term = self.nxt < 0
        self.L = int(self.head.shape[0])
        self.gsign = np.where(self.head[:, 0, GRIP] >= 0, 1.0, -1.0)
        self.sigma = self.head.reshape(-1, ACT_DIMS).std(0)
        self.cache_dir = cache_dir
        self._pause = None

    def pause(self) -> np.ndarray:
        """Pause rows (ideation A): non-terminal rows whose own next observation barely changed: visual self-motion
        (cos to the next row, both cameras, mean-centred) in the top decile OR state motion in the bottom decile."""
        if self._pause is not None:
            return self._pause
        cache = None
        if self.cache_dir is not None:
            try:
                st = (self.dir / "key_v0.npy").stat()
                cache = self.cache_dir / f"pause_{self.model}_{self.suite}_{self.name}_{self.L}_{int(st.st_mtime)}.npy"
                if cache.exists():
                    self._pause = np.load(cache).astype(bool)
                    return self._pause
            except OSError:
                cache = None
        nxt = self.nxt
        ok = nxt >= 0
        n = self.L
        ar = np.arange(n)
        seq = ok & (nxt == ar + 1)                      # next row stored right after this one (library order): sequential read
        rnd = ok & ~seq                                 # rare: gather individually
        vis = np.zeros(n, np.float64)
        for cam in ("v0", "v1"):
            K = np.load(self.dir / f"key_{cam}.npy", mmap_mode="r")
            dim = K.shape[1]
            mu = np.zeros(dim, np.float64)
            for lo in range(0, n, 1024):
                mu += np.asarray(K[lo:lo + 1024], np.float64).sum(0)
            mu = (mu / n).astype(np.float32)
            cos = np.full(n, np.nan)
            for lo in range(0, n, 1024):
                hi = min(n, lo + 1024)
                blk = np.asarray(K[lo:min(hi + 1, n)], np.float32) - mu          # one extra row for the sequential next
                nrm = np.linalg.norm(blk, axis=1)
                m = hi - lo
                if blk.shape[0] > m:
                    c = (blk[:m] * blk[1:m + 1]).sum(1) / np.maximum(nrm[:m] * nrm[1:m + 1], 1e-12)
                else:                                                          # last chunk: last row has no sequential next
                    c = np.full(m, np.nan)
                    if m > 1:
                        c[:-1] = (blk[:-1] * blk[1:]).sum(1) / np.maximum(nrm[:-1] * nrm[1:], 1e-12)
                cos[lo:hi] = np.where(seq[lo:hi], c, np.nan)
                ri = np.flatnonzero(rnd[lo:hi])
                if ri.size:
                    A = blk[ri]
                    B = np.asarray(K[nxt[lo + ri]], np.float32) - mu
                    cos[lo + ri] = (A * B).sum(1) / np.maximum(nrm[ri] * np.linalg.norm(B, axis=1), 1e-12)
            vis += cos
        rs = np.asarray(np.load(self.dir / "rs.npy", mmap_mode="r")[:, :8], np.float64)
        drs = np.full(self.L, np.nan)
        drs[ok] = np.linalg.norm(rs[nxt[ok]] - rs[ok], axis=1)
        thr_v = np.nanquantile(vis, 0.9)
        thr_s = np.nanquantile(drs, 0.1)
        pause = ((vis >= thr_v) | (drs <= thr_s)) & ~self.term
        pause = np.where(np.isfinite(vis) | np.isfinite(drs), pause, False)
        self._pause = pause
        if cache is not None:
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, pause)
            except OSError:
                pass
        return pause


_LIBS: dict = {}


def get_library(root, model, suite, name, cache_dir=None) -> Library:
    key = (str(root), model, suite, name)
    if key not in _LIBS:
        _LIBS[key] = Library(pathlib.Path(root), model, suite, name, cache_dir)
    return _LIBS[key]


# ----------------------------------------------------------------------------------------------------- reconstruction
def recon_rule(method_name: str, override: dict | None = None):
    """(rule, param): how the served head is rebuilt from the logged top-k / scores of a synthesized decision.
    top1 | mean:K | awm:KREF (kernel exp(-((s0-si)/(s0-s_kref))^2) on the logged top-10) | g3 (exp(S_i - S_0))."""
    name = method_name or ""
    if override:
        for pat, rule in override.items():
            if pat == name or re.fullmatch(pat, name) or name.startswith(pat):
                return _parse_rule(rule), "override"
    if name in ("", "native"):
        return ("top1", None), "native"
    m = re.fullmatch(r"(M4_b0cons|M8x?_b0big)_k(\d+)_(mean|kernel|med)", name)
    if m:
        how = m.group(3)
        return ("mean", int(m.group(2))), ("exact" if how == "mean" else f"approx({how}->mean)")
    if name.startswith("M8") and "top1" in name:
        return ("top1", None), "exact"
    if name.startswith("AWM"):
        mk = re.search(r"_kr(\d+)", name)
        return ("awm", int(mk.group(1)) if mk else 8), "approx(top10_of_16)"
    if re.match(r"^V[0-9]", name) or "__" in name:
        return ("g3", None), "approx(top10,exp(S_i-S_0))"
    return ("top1", None), "unknown_method->top1"


def _parse_rule(rule: str):
    rule = rule.strip()
    if rule == "top1":
        return ("top1", None)
    if rule.startswith("mean:"):
        return ("mean", int(rule.split(":", 1)[1]))
    if rule.startswith("awm"):
        return ("awm", int(rule.split(":", 1)[1]) if ":" in rule else 8)
    if rule == "g3":
        return ("g3", None)
    raise SystemExit(f"--recon rule {rule!r}: use top1 | mean:K | awm[:KREF] | g3")


def weights_for(rule, scores: np.ndarray, valid: np.ndarray, synth: np.ndarray) -> np.ndarray:
    """(N, TOPK) normalized weights; rows with synth == False use the top-1 only."""
    N = scores.shape[0]
    W = np.zeros((N, TOPK), np.float64)
    kind, par = rule
    nvalid = valid.sum(1)
    if kind == "top1":
        W[:, 0] = 1.0
    elif kind == "mean":
        K = int(par)
        W[:, :K] = 1.0
        W *= valid
    elif kind == "awm":
        s = np.where(valid, scores, np.nan)
        rel = s[:, :1] - s                                        # d~_i - d~_1 >= 0
        kref = int(par)
        ref_idx = np.clip(np.minimum(kref, nvalid) - 1, 0, TOPK - 1)
        ref = np.maximum(rel[np.arange(N), ref_idx], 1e-6)
        W = np.exp(-(rel / ref[:, None]) ** 2)
        W = np.where(valid & np.isfinite(W), W, 0.0)
    elif kind == "g3":
        s = np.where(valid, scores, np.nan)
        W = np.exp(s - s[:, :1])
        W = np.where(valid & np.isfinite(W), W, 0.0)
    W[~synth] = 0.0
    W[~synth, 0] = 1.0
    ssum = W.sum(1, keepdims=True)
    bad = ~(ssum[:, 0] > 0)
    if bad.any():
        W[bad] = 0.0
        W[bad, 0] = 1.0
        ssum = W.sum(1, keepdims=True)
    return W / ssum


# ----------------------------------------------------------------------------------------------------- arm loading
def find_arm(run_roots: list[pathlib.Path], spec: str):
    """'arm' (first run root that has runs/<arm>) or 'run:arm' (run = basename or path of a run root)."""
    if ":" in spec:
        run, arm = spec.rsplit(":", 1)
        for r in run_roots:
            if r.name == run or str(r) == run or str(r.resolve()) == str(pathlib.Path(run).resolve()):
                return r, arm
        p = pathlib.Path(run)
        if p.is_dir():
            return p, arm
        raise SystemExit(f"run root {run!r} of {spec!r} not among --run-root")
    for r in run_roots:
        if (r / "runs" / spec).is_dir():
            return r, spec
    raise SystemExit(f"arm {spec!r} not found under {[str(r / 'runs') for r in run_roots]}")


def arm_meta(run: pathlib.Path, arm: str) -> dict:
    meta = {}
    ap = run / "arms.json"
    if ap.exists():
        try:
            meta = {r["arm"]: r for r in json.loads(ap.read_text())}.get(arm, {})
        except (json.JSONDecodeError, KeyError, TypeError):
            meta = {}
    return meta


def load_journal(run: pathlib.Path, arm: str) -> dict:
    validate_journal_pool(_jsonl(run / 'runs' / arm / 'client' / 'journal.jsonl'), root_pool(run))
    comp = {}
    for r in _jsonl(run / "runs" / arm / "client" / "journal.jsonl"):
        if r.get("accepted") and r.get("status") in ("done", "failed") and not r.get("error"):
            comp[r["task_uid"]] = r
    return comp


def load_client_mix(run: pathlib.Path, arm: str, comp: dict) -> dict:
    mix, n = {}, 0
    for r in _jsonl(run / "runs" / arm / "client" / "per_step.jsonl"):
        if r.get("_kind") is not None or "step_idx" not in r:
            continue
        if r.get("task_uid") not in comp or not r.get("accepted", True):
            continue
        n += 1
        mix[str(r.get("hit_type"))] = mix.get(str(r.get("hit_type")), 0) + 1
    return {"client_decisions": n, "client_hit_mix": mix}


def _exec_head(row: dict):
    for k in EXEC_KEYS:
        v = row.get(k)
        if v is None:
            continue
        try:
            a = np.asarray(v, np.float64)
        except (TypeError, ValueError):
            continue
        if a.ndim == 2 and a.shape[0] >= EXEC_STEPS and a.shape[1] >= ACT_DIMS:
            return a[:EXEC_STEPS, :ACT_DIMS]
        if a.ndim == 1 and a.size >= EXEC_STEPS * ACT_DIMS:
            return a[:EXEC_STEPS * ACT_DIMS].reshape(EXEC_STEPS, ACT_DIMS)
    return None


def load_arm(run: pathlib.Path, arm: str, *, store: str | None, tasks, episodes, act_eps: float,
             recon_override: dict | None, cache_dir: pathlib.Path | None, want_client: bool,
             manifest=None, include_ledger=False, cost_table=None) -> dict:
    """Parse one arm into per-decision arrays + per-episode records, then compute its KPIs."""
    t0 = time.perf_counter()
    warns = []
    meta = arm_meta(run, arm)
    design = arm_manifest(run, arm, meta, manifest)
    if design:
        check_manifest_pool(design, root_pool(run))
    elif root_pool(run) == 'B':
        raise ValueError('dev KPI requires exact manifest')
    comp = load_journal(run, arm)
    if not comp:
        raise SystemExit(f"{arm}: no complete episodes in {run / 'runs' / arm / 'client' / 'journal.jsonl'}")
    keep_uid = {}
    for uid, r in comp.items():
        parts = uid.split(":")
        try:
            task, ep_idx = int(parts[-2]), int(parts[-1])
        except (ValueError, IndexError):
            warns.append(f"uid {uid!r} does not end in :<task>:<ep_idx>; skipped")
            continue
        if tasks is not None and task not in tasks:
            continue
        if episodes is not None and ep_idx not in episodes:
            continue
        if design and (task, ep_idx) not in design["selected"]:
            continue
        keep_uid[uid] = (task, ep_idx, bool(r.get("success")), r.get("attempt"), r.get("duration_s"))
    if not keep_uid:
        raise SystemExit(f"{arm}: the task / episode subset selects no complete episode")

    startups, ep_rows, recs = [], {}, {}
    files = sorted(glob.glob(str(run / "runs" / arm / "server_*" / "decisions_*.jsonl")))
    if not files:
        raise SystemExit(f"{arm}: no server decision logs under {run / 'runs' / arm}")
    for f in files:
        with open(f) as fh:
            for line in fh:
                if '"ev": "dec"' in line:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    uid = d.get("uid")
                    if uid not in keep_uid:
                        continue
                    jat = keep_uid[uid][3]
                    dat = d.get("attempt")
                    if jat is not None and dat is not None and int(jat) != int(dat):
                        continue
                    recs[(uid, int(d["step"]))] = d
                elif '"ev": "startup"' in line:
                    try:
                        startups.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
                elif '"ev": "episode"' in line:
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if e.get("uid") in keep_uid and e.get("reason") == "episode_end":
                        ep_rows[e["uid"]] = e
    if not recs:
        raise SystemExit(f"{arm}: no decision rows for the selected episodes")
    st0 = startups[0] if startups else {}
    model = meta.get("model") or st0.get("model")
    suite = SUITE_SHORT.get(meta.get("suite_short") or meta.get("suite") or st0.get("suite") or "", None)
    if model is None or suite is None:
        parts = arm.split("_")
        model = model or MODEL_OF_LETTER.get(parts[1] if len(parts) > 1 else "", None)
        suite = suite or SUITE_SHORT.get(parts[2] if len(parts) > 2 else "", None)
    if model is None or suite is None:
        raise SystemExit(f"{arm}: cannot determine model / suite (arms.json, startup row, arm name)")
    store_root = pathlib.Path(store or st0.get("root") or "/dev/shm/offline_search_store")
    method_name = st0.get("method") or meta.get("method") or "native"
    method_names = sorted({r.get("method") for r in recs.values() if r.get("method")})
    if len(method_names) > 1:
        warns.append(f"several method names in the decision rows: {method_names}")
    if method_names:
        method_name = method_names[0] if method_name in ("native", None) or method_name not in method_names else method_name

    # ---- per-decision arrays, episode order
    keys = sorted(recs, key=lambda k: (k[0], k[1]))
    N = len(keys)
    uid_list, uid_index = [], {}
    ep_id = np.empty(N, np.int64)
    step = np.empty(N, np.int64)
    top1 = np.empty(N, np.int64)
    topk = np.full((N, TOPK), -1, np.int64)
    scores = np.full((N, TOPK), np.nan, np.float64)
    conf = np.full(N, np.nan)
    synth = np.zeros(N, bool)
    agree = np.full(N, -1, np.int8)
    infer_ms = np.full(N, np.nan)
    q_us = np.full(N, np.nan)
    exec_ok = np.full(N, -1, np.int8)
    hit = np.ones(N, np.int8)
    has_hit_field = False
    src_policy = np.zeros(N, bool)
    lib_code = np.zeros(N, np.int16)
    lib_names: list[str] = []
    tau = np.full(N, np.nan)
    runlen = np.full(N, np.nan)
    s1_ms = np.full(N, np.nan)
    s23_ms = np.full(N, np.nan)
    judge = np.full(N, "", dtype=object)
    force_miss = np.zeros(N, bool)
    reason = np.zeros(N, np.int64)
    ex_keys = ("still", "stuck", "stuck_n", "terminal", "level", "disp5", "dst", "d1_rel", "w_eff", "regime",
               "os_guard", "os_event")
    extras = {k: np.full(N, np.nan) for k in ex_keys}
    exec_head = np.full((N, EXEC_STEPS, ACT_DIMS), np.nan)
    has_exec = np.zeros(N, bool)
    has_served = np.zeros(N, bool)
    for i, k in enumerate(keys):
        d = recs[k]
        uid = k[0]
        if uid not in uid_index:
            uid_index[uid] = len(uid_list)
            uid_list.append(uid)
        ep_id[i] = uid_index[uid]
        step[i] = k[1]
        top1[i] = int(d["top1"])
        tk = d.get("topk") or [d["top1"]]
        sc = d.get("scores") or [d.get("conf", 0.0)]
        n = min(TOPK, len(tk))
        topk[i, :n] = np.asarray(tk[:n], np.int64)
        scn = np.asarray(sc[:n], np.float64)
        scores[i, :len(scn)] = scn
        conf[i] = float(d.get("conf", np.nan)) if d.get("conf") is not None else np.nan
        synth[i] = bool(d.get("synth", False))
        ag = d.get("agree")
        agree[i] = -1 if ag is None else int(bool(ag))
        infer_ms[i] = d.get("infer_ms", np.nan) if d.get("infer_ms") is not None else np.nan
        q_us[i] = d.get("q_us", np.nan) if d.get("q_us") is not None else np.nan
        eo = d.get("exec_ok")
        exec_ok[i] = -1 if eo is None else int(bool(eo))
        lib = d.get("lib") or "current"
        if lib not in lib_names:
            lib_names.append(lib)
        lib_code[i] = lib_names.index(lib)
        h = d.get("hit")
        s = d.get("src")
        if h is not None:
            has_hit_field = True
            hit[i] = int(bool(h))
        elif s is not None:
            has_hit_field = True
            hit[i] = int(str(s).lower() != "policy")
        if s is not None:
            src_policy[i] = str(s).lower() == "policy"
        if d.get("tau") is not None:
            tau[i] = float(d["tau"])
        if d.get("run") is not None:
            runlen[i] = float(d["run"])
        if d.get("s1_ms") is not None:
            s1_ms[i] = float(d["s1_ms"])
        if d.get("s23_ms") is not None:
            s23_ms[i] = float(d["s23_ms"])
        if d.get("judge") is not None:
            judge[i] = str(d["judge"])
        ex = d.get("extras") or {}
        for kk in ex_keys:
            v = ex.get(kk)
            if isinstance(v, (int, float, bool)) and v is not None:
                extras[kk][i] = float(v)
        fm = ex.get("os_force_miss")
        if fm is not None:
            force_miss[i] = bool(fm)
        rc = ex.get("os_reason")
        if isinstance(rc, (int, float)) and rc is not None:
            reason[i] = int(rc)
        eh = _exec_head(d)
        if eh is not None:
            exec_head[i] = eh
            has_exec[i] = True
            has_served[i] = d.get("served_head") is not None
    valid = topk >= 0
    if not has_hit_field:
        hit[:] = 1
    n_ep = len(uid_list)
    ep_task = np.array([keep_uid[u][0] for u in uid_list], np.int64)
    ep_idx = np.array([keep_uid[u][1] for u in uid_list], np.int64)
    ep_success = np.array([keep_uid[u][2] for u in uid_list], bool)
    ep_dur = np.array([keep_uid[u][4] if keep_uid[u][4] is not None else np.nan for u in uid_list], np.float64)
    missing_eps = [u for u in keep_uid if u not in uid_index]
    if missing_eps:
        warns.append(f"{len(missing_eps)} complete episodes have no server decision rows (excluded from SR): "
                     f"{missing_eps[:3]}{'...' if len(missing_eps) > 3 else ''}")

    # ---- episode boundaries; step continuity
    starts = np.r_[0, np.flatnonzero(np.diff(ep_id) != 0) + 1, N]
    n_dec = np.diff(starts)
    gaps = 0
    for e in range(n_ep):
        s = step[starts[e]:starts[e + 1]]
        if not np.array_equal(s, np.arange(len(s))):
            gaps += 1
    if gaps:
        warns.append(f"{gaps} episodes have non-contiguous decision steps (duplicates / gaps)")

    # ---- served-action reconstruction
    rule, rule_note = recon_rule(method_name, recon_override)
    W = weights_for(rule, scores, valid, synth)
    cur = get_library(store_root, model, suite, "current", cache_dir)
    sigma = cur.sigma
    head = np.full((N, EXEC_STEPS, ACT_DIMS), np.nan)
    gvote = np.full(N, np.nan)
    ep_eff = np.full(N, np.nan)
    term_w = np.full(N, np.nan)
    term1 = np.zeros(N, bool)
    pause1 = np.zeros(N, bool)
    lib_ep = np.full(N, -1, np.int64)
    lib_step = np.full(N, -1, np.int64)
    lib_eplen = np.full(N, -1, np.int64)
    lib_prog = np.full(N, np.nan)
    libs_used = {}
    for code, lname in enumerate(lib_names):
        idx = np.flatnonzero(lib_code == code)
        if idx.size == 0:
            continue
        L = cur if lname == "current" else get_library(store_root, model, suite, lname, cache_dir)
        libs_used[lname] = {"L": L.L, "episodes": int(len(np.unique(L.ep)))}
        rows = np.where(valid[idx], topk[idx], 0)
        Wi = W[idx]
        for lo in range(0, idx.size, 4096):
            sl = slice(lo, lo + 4096)
            Hk = L.head[rows[sl]]                                              # (n, 10, 5, 7)
            head[idx[sl]] = np.einsum("nk,nkij->nij", Wi[sl], Hk)
        gvote[idx] = np.abs((Wi * L.gsign[rows]).sum(1))
        E = L.ep[rows]
        M = (E[:, :, None] == E[:, None, :])
        ep_eff[idx] = 1.0 / np.maximum(np.einsum("ni,nij,nj->n", Wi, M.astype(np.float64), Wi), 1e-12)
        term_w[idx] = (Wi * L.term[rows]).sum(1)
        t1 = top1[idx]
        if (t1 < 0).any() or (t1 >= L.L).any():
            raise SystemExit(f"{arm}: top1 row out of range for library {lname} (L={L.L})")
        term1[idx] = L.term[t1]
        pause1[idx] = L.pause()[t1]
        lib_ep[idx] = L.ep[t1]
        lib_step[idx] = L.step[t1]
        lib_eplen[idx] = L.ep_len[t1]
        lib_prog[idx] = L.prog[t1]
    # executed head: served head on HIT, logged policy chunk on MISS (NaN when absent)
    head[has_served & (hit == 1)] = exec_head[has_served & (hit == 1)]
    exec_h = head.copy()
    miss = hit == 0
    exec_h[miss] = np.nan
    exec_h[miss & has_exec] = exec_head[miss & has_exec]
    hs = head / sigma
    tnorm = np.sqrt((hs[:, :, :3] ** 2).sum(2)).mean(1)
    g0 = np.where(exec_h[:, 0, GRIP] >= 0, 1.0, -1.0)
    g0[~np.isfinite(exec_h[:, 0, GRIP])] = np.nan
    gexec = np.where(exec_h[:, :, GRIP] >= 0, 1.0, -1.0)
    gexec[~np.isfinite(exec_h[:, :, GRIP])] = np.nan
    # taxonomy class per decision (of the served set)
    cls = np.full(N, "H", dtype=object)
    cls[pause1] = "P"
    cls[tnorm < 0.6] = "Z"
    cls[gvote < 0.5] = "G"
    cls[term1] = "T"

    # ---- per-episode loop
    ep = {k: np.zeros(n_ep, np.float64) for k in (
        "n_spell", "longest", "spell_dec", "first_spell", "n_aspell", "longest_a", "aspell_dec", "first_aspell",
        "n_pspell", "first_pspell", "flips_g0", "flips_exec", "n_miss", "first_miss", "n_hit", "spell_len_sum",
        "late_term_w", "late_term1", "n_late")}
    ep["first_cls"] = np.full(n_ep, "", dtype=object)
    ep["first_spell"][:] = np.nan
    ep["first_aspell"][:] = np.nan
    ep["first_pspell"][:] = np.nan
    ep["first_miss"][:] = np.nan
    ep["late_term_w"][:] = np.nan
    ep["late_term1"][:] = np.nan
    ep_spell_dec_mask = np.zeros(N, bool)
    for e in range(n_ep):
        a, b = starts[e], starts[e + 1]
        n = b - a
        t1 = top1[a:b]
        h = hit[a:b]
        served = h == 1
        # identical-pick runs over served decisions (a MISS breaks the run)
        runs = _runs(t1, served)
        spells = [(s, l) for s, l in runs if l >= 3]
        ep["n_spell"][e] = len(spells)
        ep["longest"][e] = max((l for _, l in runs), default=0)
        ep["spell_dec"][e] = sum(l for _, l in spells)
        ep["spell_len_sum"][e] = sum(l for _, l in spells)
        for s, l in spells:
            ep_spell_dec_mask[a + s:a + s + l] = True
        if spells:
            fs = spells[0][0]
            ep["first_spell"][e] = fs
            ep["first_cls"][e] = cls[a + fs]
        # proposal spells (all decisions, MISS included)
        pruns = _runs(t1, np.ones(n, bool))
        pspells = [(s, l) for s, l in pruns if l >= 3]
        ep["n_pspell"][e] = len(pspells)
        if pspells:
            ep["first_pspell"][e] = pspells[0][0]
        # served-action repeat runs
        hh = hs[a:b]
        same = np.zeros(n, bool)
        if n > 1:
            dif = np.sqrt(np.nanmean((hh[1:] - hh[:-1]) ** 2, axis=(1, 2)))
            same[1:] = (dif <= act_eps) & served[1:] & served[:-1]
        aruns = _runs_from_same(same, served)
        aspells = [(s, l) for s, l in aruns if l >= 3]
        ep["n_aspell"][e] = len(aspells)
        ep["longest_a"][e] = max((l for _, l in aruns), default=0)
        ep["aspell_dec"][e] = sum(l for _, l in aspells)
        if aspells:
            ep["first_aspell"][e] = aspells[0][0]
        # gripper flips
        gg = g0[a:b]
        ok = np.isfinite(gg)
        ep["flips_g0"][e] = float(np.sum((gg[1:] != gg[:-1]) & ok[1:] & ok[:-1]))
        ge = gexec[a:b].reshape(-1)
        oke = np.isfinite(ge)
        ep["flips_exec"][e] = float(np.sum((ge[1:] != ge[:-1]) & oke[1:] & oke[:-1]))
        # mixed
        ep["n_hit"][e] = float(served.sum())
        ep["n_miss"][e] = float((~served).sum())
        if (~served).any():
            ep["first_miss"][e] = int(np.flatnonzero(~served)[0])
        # last third
        late = np.arange(n) >= (2.0 / 3.0) * n
        ep["n_late"][e] = late.sum()
        if late.any():
            ep["late_term_w"][e] = _mean(term_w[a:b][late])
            ep["late_term1"][e] = _mean(term1[a:b][late].astype(float))
    dt = time.perf_counter() - t0
    client = load_client_mix(run, arm, {u: 1 for u in keep_uid}) if want_client else {}
    srv_agree = [(bool(ep_rows[u].get("success")) == keep_uid[u][2]) for u in uid_list if u in ep_rows]

    dec = dict(N=N, ep_id=ep_id, step=step, top1=top1, hit=hit, synth=synth, agree=agree, infer_ms=infer_ms,
               q_us=q_us, exec_ok=exec_ok, conf=conf, tau=tau, runlen=runlen, s1_ms=s1_ms, s23_ms=s23_ms,
               judge=judge, force_miss=force_miss, reason=reason, extras=extras, gvote=gvote, ep_eff=ep_eff,
               term_w=term_w, term1=term1, tnorm=tnorm, in_spell=ep_spell_dec_mask, lib_prog=lib_prog,
               has_exec=has_exec, src_policy=src_policy)
    epi = dict(n=n_ep, uid=uid_list, task=ep_task, ep_idx=ep_idx, success=ep_success, n_dec=n_dec, dur=ep_dur,
               starts=starts, **ep)
    info = dict(arm=arm, run_root=str(run), model=model, suite=suite, method=method_name,
                method_spec=meta.get("method") or st0.get("method_spec"), kwargs=meta.get("kwargs", st0.get("kwargs")),
                mode=meta.get("mode"), store=str(store_root), libs=libs_used, recon_rule=f"{rule[0]}" +
                (f":{rule[1]}" if rule[1] is not None else ""), recon_note=rule_note, servers=len(startups),
                server_episode_rows=len(ep_rows), server_success_agree=(int(sum(srv_agree)), len(srv_agree)),
                mixed=bool(has_hit_field and (hit == 0).any()), has_hit_field=has_hit_field,
                subset={"tasks": tasks, "episodes": episodes}, parse_s=round(dt, 2), warnings=warns, **client)
    out = {"info": info, "dec": dec, "ep": epi}
    modern = r4_enabled(meta, recs.values(), startups)
    if modern or include_ledger or cost_table is not None or design:
        out["cost_ledger"] = ledger({**meta, "model": model}, list(recs.values()), startups,
                                    {u: comp[u] for u in uid_list}, cost_table)
        out["r4_log"] = modern
    if design:
        out["manifest"] = design
    if has_served.any():
        info["recon_note"] = f"served_head {int(has_served.sum())}/{N}; kernel diagnostics {rule_note}"
    return out


def _runs(vals: np.ndarray, served: np.ndarray):
    """[(start, length)] of maximal runs of identical `vals` over consecutive served decisions (a non-served decision
    ends the current run and belongs to none)."""
    out = []
    n = len(vals)
    i = 0
    while i < n:
        if not served[i]:
            i += 1
            continue
        j = i + 1
        while j < n and served[j] and vals[j] == vals[i]:
            j += 1
        out.append((i, j - i))
        i = j
    return out


def _runs_from_same(same: np.ndarray, served: np.ndarray):
    """Runs where same[j] says decision j continues the run of j-1 (both served)."""
    out = []
    n = len(same)
    i = 0
    while i < n:
        if not served[i]:
            i += 1
            continue
        j = i + 1
        while j < n and same[j]:
            j += 1
        out.append((i, j - i))
        i = j
    return out


# ----------------------------------------------------------------------------------------------------- KPIs
def arm_kpis(A: dict, ir_ref_ms: float | None) -> dict:
    info, D, E = A["info"], A["dec"], A["ep"]
    n_ep = E["n"]
    succ = E["success"]
    fail = ~succ
    n_s, n_f = int(succ.sum()), int(fail.sum())
    dec_succ = succ[D["ep_id"]]
    dec_fail = ~dec_succ
    cap = int(E["n_dec"].max()) if n_ep else 0
    sr = n_s / n_ep
    out = {"arm": info["arm"], "run_root": info["run_root"], "model": info["model"], "suite": info["suite"],
           "init_pool": root_pool(info['run_root']),
           "method": info["method"], "method_spec": info["method_spec"], "kwargs": info["kwargs"],
           "recon": {"rule": info["recon_rule"], "note": info["recon_note"], "libs": info["libs"]},
           "subset": info["subset"], "mixed": info["mixed"],
           "episodes": n_ep, "success": n_s, "sr": sr, "sr_wilson95": wilson(n_s, n_ep),
           "decisions": int(D["N"]), "dec_per_ep": _mean(E["n_dec"]),
           "dec_per_ep_success": _mean(E["n_dec"][succ]), "dec_per_ep_fail": _mean(E["n_dec"][fail]),
           "step_cap": cap, "fail_timeout_share": _share(E["n_dec"] >= cap, fail),
           "episode_duration_s_p50": _median(E["dur"]),
           "server_infer_ms_mean": _mean(D["infer_ms"]), "server_infer_ms_p50": _median(D["infer_ms"]),
           "method_query_us_p50": _median(D["q_us"]),
           "exec_ok_share": _share(D["exec_ok"] == 1, D["exec_ok"] >= 0) if (D["exec_ok"] >= 0).any() else None,
           "server_success_agrees_journal": info["server_success_agree"],
           "client_decisions": info.get("client_decisions"), "client_hit_mix": info.get("client_hit_mix"),
           "servers_started": info["servers"], "warnings": info["warnings"], "parse_s": info["parse_s"]}
    # per task
    per_task = {}
    for t in sorted(set(E["task"].tolist())):
        m = E["task"] == t
        k, n = int(succ[m].sum()), int(m.sum())
        per_task[str(t)] = {"n": n, "success": k, "sr": k / n, "wilson95": wilson(k, n),
                            "dec_per_ep": _mean(E["n_dec"][m]), "spells_per_ep": _mean(E["n_spell"][m]),
                            "spell_dec_share": _share(D["in_spell"], (E["task"] == t)[D["ep_id"]]),
                            "flips_g0_fail": _mean(E["flips_g0"][m & fail]) if (m & fail).any() else None}
    out["per_task"] = per_task
    # spells
    ns = E["n_spell"]
    out["spells"] = {
        "definition": "run of >= 3 consecutive served (HIT) decisions with identical top-1; a MISS breaks the run",
        "per_ep": _mean(ns), "per_success_ep": _mean(ns[succ]), "per_fail_ep": _mean(ns[fail]),
        "n_spells": int(ns.sum()), "len_mean": float(E["spell_len_sum"].sum() / max(ns.sum(), 1)),
        "dec_share": _share(D["in_spell"]), "dec_share_success": _share(D["in_spell"], dec_succ),
        "dec_share_fail": _share(D["in_spell"], dec_fail),
        "longest_run_mean": _mean(E["longest"]), "longest_run_success": _mean(E["longest"][succ]),
        "longest_run_fail": _mean(E["longest"][fail]),
        "p_success_given_0": _share(succ, ns == 0), "n_0": int((ns == 0).sum()),
        "p_success_given_1": _share(succ, ns == 1), "n_1": int((ns == 1).sum()),
        "p_success_given_2plus": _share(succ, ns >= 2), "n_2plus": int((ns >= 2).sum()),
        "fail_with_spell": _share(ns >= 1, fail), "success_with_spell": _share(ns >= 1, succ),
        "first_spell_step_fail": _mean(E["first_spell"][fail]),
        "spells_in_success_eps_share": float(ns[succ].sum() / max(ns.sum(), 1)),
    }
    na = E["n_aspell"]
    out["action_spells"] = {
        "definition": "run of >= 3 consecutive served decisions whose sigma-normalized served heads [:5,:7] differ by "
                      "RMS <= act_eps between neighbours",
        "per_ep": _mean(na), "per_success_ep": _mean(na[succ]), "per_fail_ep": _mean(na[fail]),
        "dec_share": float(E["aspell_dec"].sum() / max(D["N"], 1)),
        "dec_share_fail": float(E["aspell_dec"][fail].sum() / max(E["n_dec"][fail].sum(), 1)),
        "longest_run_fail": _mean(E["longest_a"][fail]), "longest_run_success": _mean(E["longest_a"][succ]),
        "p_success_given_0": _share(succ, na == 0), "fail_with_spell": _share(na >= 1, fail),
    }
    # first-spell taxonomy of failed episodes (and of all episodes with a spell)
    fc = E["first_cls"]
    tax = {}
    for who, m in (("fail", fail & (ns >= 1)), ("all", ns >= 1)):
        n = int(m.sum())
        cnt = {c: int(np.sum(fc[m] == c)) for c in CLASS_ORDER}
        tax[who] = {"n": n, "counts": cnt, "shares": {c: (cnt[c] / n if n else None) for c in CLASS_ORDER}}
    out["first_spell_class"] = {"classes": CLASS_LONG, "priority": "T > G > Z > P > H",
                                "thresholds": {"G": "|weighted gripper vote| < .5", "Z": "served translation norm < .6 sigma",
                                               "P": "library pause row (top-decile stillness or bottom-decile state motion, non-terminal)"},
                                **tax}
    # gripper flips
    out["gripper_flips"] = {
        "g0_definition": "sign changes of the served/executed gripper at step 0 between consecutive decisions (ideation A)",
        "exec_definition": "sign changes along the concatenated executed gripper series (5 steps per decision)",
        "g0_per_ep": _mean(E["flips_g0"]), "g0_success": _mean(E["flips_g0"][succ]), "g0_fail": _mean(E["flips_g0"][fail]),
        "exec_per_ep": _mean(E["flips_exec"]), "exec_success": _mean(E["flips_exec"][succ]),
        "exec_fail": _mean(E["flips_exec"][fail]),
    }
    # terminal weight late, effective episodes, vote split, agreement, stillness
    late_mask = np.zeros(D["N"], bool)
    for e in range(n_ep):
        a, b = E["starts"][e], E["starts"][e + 1]
        n = b - a
        late_mask[a:b] = np.arange(n) >= (2.0 / 3.0) * n
    out["terminal_rows"] = {
        "w_term_late": _mean(D["term_w"][late_mask]), "w_term_late_success": _mean(D["term_w"][late_mask & dec_succ]),
        "w_term_late_fail": _mean(D["term_w"][late_mask & dec_fail]),
        "top1_term_share": _share(D["term1"]), "top1_term_share_fail": _share(D["term1"], dec_fail),
        "top1_term_share_in_spells": _share(D["term1"], D["in_spell"]),
    }
    out["kernel_set"] = {
        "ep_eff_mean": _mean(D["ep_eff"]), "ep_eff_success": _mean(D["ep_eff"][dec_succ]), "ep_eff_fail": _mean(D["ep_eff"][dec_fail]),
        "vote_split_lt08_share": _share(D["gvote"] < 0.8), "vote_split_success": _share(D["gvote"] < 0.8, dec_succ),
        "vote_split_fail": _share(D["gvote"] < 0.8, dec_fail),
        "gvote_in_spells": _mean(D["gvote"][D["in_spell"]]), "gvote_free": _mean(D["gvote"][~D["in_spell"]]),
        "tnorm_in_spells": _mean(D["tnorm"][D["in_spell"]]), "tnorm_free": _mean(D["tnorm"][~D["in_spell"]]),
        "synth_share": _share(D["synth"]),
    }
    ag = D["agree"]
    out["native_agreement"] = {"all": _share(ag == 1, ag >= 0) if (ag >= 0).any() else None,
                               "success": _share(ag == 1, (ag >= 0) & dec_succ) if ((ag >= 0) & dec_succ).any() else None,
                               "fail": _share(ag == 1, (ag >= 0) & dec_fail) if ((ag >= 0) & dec_fail).any() else None}
    st = D["extras"]["still"]
    if np.isfinite(st).any():
        out["stillness"] = {"still_gt_198_share_success": _share(st > 1.98, dec_succ & np.isfinite(st)),
                            "still_gt_198_share_fail": _share(st > 1.98, dec_fail & np.isfinite(st))}
    out["confidence"] = {"mean": _mean(D["conf"]), "mean_success": _mean(D["conf"][dec_succ]), "mean_fail": _mean(D["conf"][dec_fail]),
                         "min_per_ep_success": _mean([D["conf"][E["starts"][e]:E["starts"][e + 1]].min() for e in range(n_ep) if succ[e]] or [np.nan]),
                         "min_per_ep_fail": _mean([D["conf"][E["starts"][e]:E["starts"][e + 1]].min() for e in range(n_ep) if fail[e]] or [np.nan])}
    # mixed
    if info["has_hit_field"]:
        out["mixed_mode"] = mixed_kpis(A, ir_ref_ms)
    if "cost_ledger" in A:
        out["cost_ledger"] = A["cost_ledger"]
        if A.get("r4_log") and "mixed_mode" in out:
            out["mixed_mode"]["ir_formula_pi05"] = (A["cost_ledger"]["ir_per_five_controls"]
                                                       if info["model"] == "pi05" else None)
            out["mixed_mode"]["ir_formula_note"] = "R4 stage ledger, normalized per five nominal controls; see cost_ledger"
            out["mixed_mode"]["ir_stage_measured"] = A["cost_ledger"]["ir_measured_per_five_controls"]
            out["mixed_mode"]["ir_stage_note"] = "sum logged stage costs / reference full cost / nominal controls * 5"
    if A.get("manifest"):
        out["weighted"] = design_estimate(A["manifest"], {(int(t), int(i)): int(s) for t, i, s in
                                                        zip(E["task"], E["ep_idx"], E["success"])})
        out["weighted"]["sr"] = out["weighted"]["estimate"]
    return out


def mixed_kpis(A: dict, ir_ref_ms: float | None) -> dict:
    info, D, E = A["info"], A["dec"], A["ep"]
    hit = D["hit"].astype(bool)
    miss = ~hit
    succ = E["success"]
    fail = ~succ
    dec_succ = succ[D["ep_id"]]
    dec_fail = ~dec_succ
    N = D["N"]
    h = float(hit.mean())
    # regimes: step 0 / after HIT / after MISS
    prev_hit = np.full(N, -1, np.int8)
    for e in range(E["n"]):
        a, b = E["starts"][e], E["starts"][e + 1]
        prev_hit[a + 1:b] = D["hit"][a:b - 1]
    reg = {"step0": _share(hit, prev_hit == -1), "after_hit": _share(hit, prev_hit == 1), "after_miss": _share(hit, prev_hit == 0),
           "n_step0": int((prev_hit == -1).sum()), "n_after_hit": int((prev_hit == 1).sum()), "n_after_miss": int((prev_hit == 0).sum())}
    per_task_h = {str(t): _share(hit, (E["task"] == t)[D["ep_id"]]) for t in sorted(set(E["task"].tolist()))}
    ir_formula = IR_PI05[0] + IR_PI05[1] * (1.0 - h)
    out = {"realized_h": h, "n_hit": int(hit.sum()), "n_miss": int(miss.sum()), "h_per_regime": reg, "h_per_task": per_task_h,
           "ir_formula_pi05": ir_formula, "ir_formula_note": "IR = 0.152 + 0.848 (1 - h); pi0.5 CUDA-graph three-stage definition"
           + ("" if info["model"] == "pi05" else " (applied to a non-pi0.5 arm for reference only)")}
    im = D["infer_ms"]
    if ir_ref_ms is not None and np.isfinite(im).any():
        out["ir_measured_vs_ref"] = float(_mean(im) / ir_ref_ms)
        out["ir_measured_note"] = f"mean server infer_ms {_mean(im):.1f} / reference {ir_ref_ms:.1f}"
    s1, s23 = D["s1_ms"], D["s23_ms"]
    if np.isfinite(s1).any():
        s1m = _mean(s1)
        s23_miss = _mean(s23[miss]) if np.isfinite(s23[miss]).any() else float("nan")
        if np.isfinite(s23_miss) and s23_miss > 0:
            out["ir_stage_measured"] = float((s1m + (1 - h) * s23_miss) / (s1m + s23_miss))
            out["ir_stage_note"] = f"(mean s1 {s1m:.1f} + (1-h) mean s23|MISS {s23_miss:.1f}) / (mean s1 + mean s23|MISS)"
        out["s1_ms_mean"] = s1m
        out["s23_ms_mean_on_miss"] = s23_miss
    # MISS placement
    out["miss_share_in_fail_eps"] = _share(miss, dec_fail)
    out["miss_share_in_success_eps"] = _share(miss, dec_succ)
    out["share_of_misses_in_fail_eps"] = _share(dec_fail, miss) if miss.any() else None
    out["dec_share_in_fail_eps"] = _share(dec_fail)
    out["miss_per_ep_success"] = _mean(E["n_miss"][succ])
    out["miss_per_ep_fail"] = _mean(E["n_miss"][fail])
    out["success_eps_interrupted_share"] = _share(E["n_miss"] >= 1, succ)
    out["fail_eps_touched_share"] = _share(E["n_miss"] >= 1, fail)
    # reasons
    fm = D["force_miss"]
    rc = D["reason"]
    if miss.any():
        out["forced_miss_share_of_misses"] = _share(fm, miss)
        codes = rc[miss & (rc > 0)]
        tot = int(len(codes))
        out["reason_code_shares"] = {f"{int(c)}:{REASON_NAMES.get(int(c), 'code' + str(int(c)))}": int((codes == c).sum()) / tot
                                     for c in sorted(set(codes.tolist()))} if tot else {}
        out["reason_code_counts"] = {f"{int(c)}:{REASON_NAMES.get(int(c), 'code' + str(int(c)))}": int((codes == c).sum())
                                     for c in sorted(set(codes.tolist()))}
        js = D["judge"][miss]
        vals, cnts = np.unique(np.asarray([str(x) for x in js]), return_counts=True)
        order = np.argsort(-cnts)[:12]
        out["judge_reason_counts_on_miss"] = {str(vals[i]): int(cnts[i]) for i in order if vals[i] != ""}
        out["judge_reason_counts_in_fail_eps"] = {}
        jf = D["judge"][miss & dec_fail]
        if len(jf):
            v2, c2 = np.unique(np.asarray([str(x) for x in jf]), return_counts=True)
            out["judge_reason_counts_in_fail_eps"] = {str(v2[i]): int(c2[i]) for i in np.argsort(-c2)[:12] if v2[i] != ""}
        out["reason_code_share_in_fail_eps"] = _share(dec_fail, miss & (rc > 0)) if (miss & (rc > 0)).any() else None
    if np.isfinite(D["tau"]).any():
        out["tau"] = {"mean": _mean(D["tau"]), "median": _median(D["tau"]),
                      "median_second_half": _median(D["tau"][N // 2:])}
    if np.isfinite(D["runlen"]).any():
        out["hit_run_at_miss_mean"] = _mean(D["runlen"][miss])
        out["hit_run_max"] = float(np.nanmax(D["runlen"]))
    # first MISS relative to the first spell (proposal spells: what the cache would have served) and served spells
    fm_step = E["first_miss"]
    for key, first in (("proposal_spell", E["first_pspell"]), ("served_spell", E["first_spell"])):
        both = np.isfinite(fm_step) & np.isfinite(first)
        rel = fm_step - first
        d = {"fail_eps_with_spell": int((fail & np.isfinite(first)).sum()),
             "fail_eps_with_spell_and_miss": int((fail & both).sum()),
             "offset_mean_fail": _mean(rel[fail & both]), "offset_median_fail": _median(rel[fail & both]),
             "early_share_fail": _share(rel <= 1, fail & both),          # first MISS no later than spell start + 1 (ideation C)
             "before_share_fail": _share(rel < 0, fail & both),
             "miss_before_or_at_spell_start_share_of_fail_with_spell":
                 float(((rel <= 1) & both & fail).sum() / max((fail & np.isfinite(first)).sum(), 1)),
             "offset_mean_success": _mean(rel[succ & both]), "n_success_both": int((succ & both).sum())}
        out[f"first_miss_vs_first_{key}"] = d
    out["first_miss_step_mean_fail"] = _mean(fm_step[fail])
    out["first_miss_step_mean_success"] = _mean(fm_step[succ])
    out["first_miss_rel_pos_fail"] = _mean((fm_step / np.maximum(E["n_dec"], 1))[fail])
    out["exec_chunk_logged_on_miss_share"] = _share(D["has_exec"], miss) if miss.any() else None
    out["policy_src_rows"] = int(D["src_policy"].sum())
    return out


# ----------------------------------------------------------------------------------------------------- paired
def paired(ref: dict, arm: dict, boot: int, seed: int) -> dict:
    if root_pool(ref['info']['run_root']) != root_pool(arm['info']['run_root']):
        raise ValueError('POOL_MISMATCH: cannot pair A and B outcomes')
    Er, Ea = ref["ep"], arm["ep"]
    kr = {(int(t), int(i)): bool(s) for t, i, s in zip(Er["task"], Er["ep_idx"], Er["success"])}
    ka = {(int(t), int(i)): bool(s) for t, i, s in zip(Ea["task"], Ea["ep_idx"], Ea["success"])}
    common = sorted(set(kr) & set(ka))
    n = len(common)
    if n == 0:
        out = {"ref": ref["info"]["arm"], "arm": arm["info"]["arm"], "n_common": 0}
        design = arm.get("manifest") or ref.get("manifest")
        if design:
            out["weighted"] = design_estimate(design, {})
            out["weighted"]["delta_sr"] = None
        return out
    r = np.array([kr[k] for k in common])
    a = np.array([ka[k] for k in common])
    A = int((r & a).sum())
    B = int((r & ~a).sum())          # S -> F
    C = int((~r & a).sum())          # F -> S
    Dd = int((~r & ~a).sum())
    delta = (C - B) / n
    rng = np.random.default_rng(seed)
    cnt = rng.multinomial(n, [A / n, B / n, C / n, Dd / n], size=int(boot))
    bs = (cnt[:, 2] - cnt[:, 1]) / n
    diff = a.astype(int) - r.astype(int)
    se = diff.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    per_task = {}
    tasks = sorted({k[0] for k in common})
    for t in tasks:
        m = np.array([k[0] == t for k in common])
        b_t, c_t, n_t = int((r & ~a & m).sum()), int((~r & a & m).sum()), int(m.sum())
        per_task[str(t)] = {"n": n_t, "sr_ref": float(r[m].mean()), "sr_arm": float(a[m].mean()),
                            "delta": (c_t - b_t) / n_t, "s_to_f": b_t, "f_to_s": c_t, "mcnemar_p": mcnemar_exact(b_t, c_t)}
    out = {"ref": ref["info"]["arm"], "arm": arm["info"]["arm"], "n_common": n,
            "sr_ref": float(r.mean()), "sr_arm": float(a.mean()), "delta_sr": delta,
            "both_success": A, "s_to_f": B, "f_to_s": C, "both_fail": Dd,
            "mcnemar_exact_p": mcnemar_exact(B, C),
            "bootstrap95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], "bootstrap_n": int(boot),
            "newcombe95": list(newcombe_paired(A, B, C, Dd)),
            "normal95": [delta - Z95 * se, delta + Z95 * se] if n > 1 else None,
            "per_task": per_task}
    design = arm.get("manifest") or ref.get("manifest")
    if design:
        if arm.get("manifest") and ref.get("manifest") and arm["manifest"]["data"] != ref["manifest"]["data"]:
            raise ValueError("paired weighted comparison requires the same sampling manifest on both arms")
        w = design_estimate(design, {k: int(ka[k]) - int(kr[k]) for k in common})
        w["delta_sr"] = w["estimate"]
        w["sr_ref"] = design_estimate(design, {k: int(kr[k]) for k in common})["estimate"]
        w["sr_arm"] = design_estimate(design, {k: int(ka[k]) for k in common})["estimate"]
        w["unweighted_note"] = "raw McNemar/bootstrap above describe the sampled rows, not the full population"
        out["weighted"] = w
    return out


# ----------------------------------------------------------------------------------------------------- report
def md_report(res: dict) -> str:
    L = []
    arms = res["arms"]
    sub = res["args"].get("subset_note", "all complete episodes")
    L.append(f"# Closed-loop KPIs ({sub})\n")
    L.append(f"run roots: {', '.join(res['args']['run_roots'])}; generated {res['generated']}; kpi schema {SCHEMA}\n")
    # table 1
    L.append("## Arms\n")
    hdr = ["arm", "model/suite", "method", "n", "SR", "Wilson 95%", "dec/ep S | F", "spells/ep S | F", "spell dec share",
           "longest run F", "flips g0 F", "flips exec F", "first-spell T/G/Z/P/H (fail)", "w_term_late F", "ep_eff S | F",
           "P(S|0 spells)", "native agree"]
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "---|" * len(hdr))
    for a in arms:
        sp, fs, gf, tr, ks = a["spells"], a["first_spell_class"]["fail"]["shares"], a["gripper_flips"], a["terminal_rows"], a["kernel_set"]
        row = [a["arm"], f"{a['model']}/{a['suite']}", a["method"], str(a["episodes"]), _fmt(a["sr"]),
               f"[{_fmt(a['sr_wilson95'][0])}, {_fmt(a['sr_wilson95'][1])}]",
               f"{_fmt(a['dec_per_ep_success'], 1)} | {_fmt(a['dec_per_ep_fail'], 1)}",
               f"{_fmt(sp['per_success_ep'], 2)} | {_fmt(sp['per_fail_ep'], 2)}", _fmt(sp["dec_share"]),
               _fmt(sp["longest_run_fail"], 1), _fmt(gf["g0_fail"], 2), _fmt(gf["exec_fail"], 2),
               "/".join(_fmt(fs[c], 2) if fs[c] is not None else "-" for c in CLASS_ORDER),
               _fmt(tr["w_term_late_fail"]), f"{_fmt(ks['ep_eff_success'], 2)} | {_fmt(ks['ep_eff_fail'], 2)}",
               f"{_fmt(sp['p_success_given_0'], 2)} (n={sp['n_0']})", _fmt(a["native_agreement"]["all"])]
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    L.append("Second-order spell facts: P(S | 1 spell) / P(S | >= 2) / share of failed episodes with a spell / spells in successful episodes; "
             "action-repeat spells (served heads within act_eps).\n")
    hdr2 = ["arm", "P(S|1)", "P(S|>=2)", "fail w/ spell", "spell len", "spells in S eps", "act-spells/ep S | F", "act-spell share", "longest act run F",
            "vote split <.8 S | F", "top1 terminal share (spells)", "conf mean S | F", "infer ms p50", "exec_ok", "srv=journal"]
    L.append("| " + " | ".join(hdr2) + " |")
    L.append("|" + "---|" * len(hdr2))
    for a in arms:
        sp, asp, ks = a["spells"], a["action_spells"], a["kernel_set"]
        row = [a["arm"], f"{_fmt(sp['p_success_given_1'], 2)} (n={sp['n_1']})", f"{_fmt(sp['p_success_given_2plus'], 2)} (n={sp['n_2plus']})",
               _fmt(sp["fail_with_spell"]), _fmt(sp["len_mean"], 1), _fmt(sp["spells_in_success_eps_share"]),
               f"{_fmt(asp['per_success_ep'], 2)} | {_fmt(asp['per_fail_ep'], 2)}", _fmt(asp["dec_share"]), _fmt(asp["longest_run_fail"], 1),
               f"{_fmt(ks['vote_split_success'])} | {_fmt(ks['vote_split_fail'])}", _fmt(a["terminal_rows"]["top1_term_share_in_spells"]),
               f"{_fmt(a['confidence']['mean_success'], 2)} | {_fmt(a['confidence']['mean_fail'], 2)}", _fmt(a["server_infer_ms_p50"], 1),
               _fmt(a["exec_ok_share"]), f"{a['server_success_agrees_journal'][0]}/{a['server_success_agrees_journal'][1]}"]
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    # per task
    L.append("## Per-task SR\n")
    tasks = sorted({int(t) for a in arms for t in a["per_task"]})
    L.append("| arm | " + " | ".join(f"t{t}" for t in tasks) + " |")
    L.append("|---|" + "---|" * len(tasks))
    for a in arms:
        L.append(f"| {a['arm']} | " + " | ".join(
            (f"{_fmt(a['per_task'][str(t)]['sr'], 2)} ({a['per_task'][str(t)]['success']}/{a['per_task'][str(t)]['n']})"
             if str(t) in a["per_task"] else "-") for t in tasks) + " |")
    L.append("")
    # paired
    if res["paired"]:
        L.append("## Paired comparisons (common (task, ep_idx) inits)\n")
        hdr3 = ["ref -> arm", "n", "SR ref", "SR arm", "dSR (pp)", "S->F", "F->S", "McNemar p", "bootstrap 95% (pp)", "Newcombe 95% (pp)", "normal 95% (pp)"]
        L.append("| " + " | ".join(hdr3) + " |")
        L.append("|" + "---|" * len(hdr3))
        for p in res["paired"]:
            if p.get("n_common", 0) == 0:
                L.append(f"| {p['ref']} -> {p['arm']} | 0 | no common inits | | | | | | | | |")
                continue
            pp = lambda v: f"{100 * v:+.1f}"
            L.append(f"| {p['ref']} -> {p['arm']} | {p['n_common']} | {_fmt(p['sr_ref'])} | {_fmt(p['sr_arm'])} | {pp(p['delta_sr'])} | "
                     f"{p['s_to_f']} | {p['f_to_s']} | {_fmt(p['mcnemar_exact_p'], 4)} | [{pp(p['bootstrap95'][0])}, {pp(p['bootstrap95'][1])}] | "
                     f"[{pp(p['newcombe95'][0])}, {pp(p['newcombe95'][1])}] | "
                     + (f"[{pp(p['normal95'][0])}, {pp(p['normal95'][1])}]" if p.get("normal95") else "-") + " |")
        L.append("")
        for p in res["paired"]:
            if p.get("n_common", 0) == 0:
                continue
            ts = sorted(int(t) for t in p["per_task"])
            L.append(f"per-task dSR (pp) {p['ref']} -> {p['arm']} (F->S / S->F): " + ", ".join(
                f"t{t} {100 * p['per_task'][str(t)]['delta']:+.0f} ({p['per_task'][str(t)]['f_to_s']}/{p['per_task'][str(t)]['s_to_f']})" for t in ts))
        L.append("")
    # mixed
    mixed = [a for a in arms if a.get("mixed_mode")]
    if mixed:
        L.append("## Mixed HIT/MISS arms\n")
        hdr4 = ["arm", "h", "h step0 | after HIT | after MISS", "IR (pi05 formula)", "IR measured", "MISS share in F eps | S eps", "share of MISSes in F eps",
                "MISS/ep S | F", "S eps interrupted", "F eps touched", "first MISS - first proposal spell (mean, early share)", "forced share", "reason codes"]
        L.append("| " + " | ".join(hdr4) + " |")
        L.append("|" + "---|" * len(hdr4))
        for a in mixed:
            m = a["mixed_mode"]
            reg = m["h_per_regime"]
            fm = m["first_miss_vs_first_proposal_spell"]
            irm = m.get("ir_stage_measured", m.get("ir_measured_vs_ref"))
            L.append(f"| {a['arm']} | {_fmt(m['realized_h'])} | {_fmt(reg['step0'], 2)} | {_fmt(reg['after_hit'], 2)} | {_fmt(reg['after_miss'], 2)} | "
                     f"{_fmt(m['ir_formula_pi05'])} | {_fmt(irm)} | {_fmt(m['miss_share_in_fail_eps'])} | {_fmt(m['miss_share_in_success_eps'])} | "
                     f"{_fmt(m['share_of_misses_in_fail_eps'])} | {_fmt(m['miss_per_ep_success'], 1)} | {_fmt(m['miss_per_ep_fail'], 1)} | "
                     f"{_fmt(m['success_eps_interrupted_share'])} | {_fmt(m['fail_eps_touched_share'])} | "
                     f"{_fmt(fm['offset_mean_fail'], 1)}, {_fmt(fm['early_share_fail'])} (n={fm['fail_eps_with_spell_and_miss']}) | "
                     f"{_fmt(m.get('forced_miss_share_of_misses'))} | "
                     + "; ".join(f"{k} {_fmt(v, 2)}" for k, v in (m.get("reason_code_shares") or {}).items()) + " |")
        L.append("")
        for a in mixed:
            m = a["mixed_mode"]
            if m.get("judge_reason_counts_on_miss"):
                L.append(f"{a['arm']} judge reasons on MISS: " + ", ".join(f"{k} {v}" for k, v in m["judge_reason_counts_on_miss"].items()))
        L.append("")
    # notes
    if any("cost_ledger" in a for a in arms):
        L.extend(["## R4 cost ledger (per five nominal control steps)\n",
                  "| arm | v | m | K/MISS | L/request | IR | controls/ep | cost source |",
                  "|---|---|---|---|---|---|---|---|"])
        for a in arms:
            if "cost_ledger" in a:
                c = a["cost_ledger"]
                L.append(f"| {a['arm']} | {_fmt(c['v'])} | {_fmt(c['m'])} | {_fmt(c['k_per_miss']['mean'])} | "
                         f"{_fmt(c['l_per_request']['mean'])} | {_fmt(c['ir_per_five_controls'])} | "
                         f"{_fmt(c['controls_per_episode'], 1)} | {c['cost_source']} |")
        L.append("\nControls are requests × L; the final chunk can be partially executed.\n")
    if any("weighted" in a for a in arms):
        L.append("## Manifest-weighted estimates\n")
        for a in arms:
            if "weighted" in a:
                w = a["weighted"]
                L.append(f"- {a['arm']}: weighted SR {_fmt(w['estimate'])}; design variance {_fmt(w['design_variance'], 8)}; "
                         f"95% design interval {w.get('design_normal95')}; completed {w['observed_pairs']}/{w['expected_pairs']} pairs.")
        for p in res["paired"]:
            if "weighted" in p:
                w = p["weighted"]
                L.append(f"- {p['ref']} -> {p['arm']}: weighted ΔSR {_fmt(w['estimate'])}; design variance "
                         f"{_fmt(w['design_variance'], 8)}; 95% design interval {w.get('design_normal95')}.")
        L.append("\nRaw SR and paired tests describe the oversampled rows only. Design intervals exclude rollout randomness.\n")
    L.append("## Notes\n")
    for a in arms:
        L.append(f"- {a['arm']}: method `{a['method']}`, served-action reconstruction `{a['recon']['rule']}` ({a['recon']['note']}), libraries "
                 + ", ".join(f"{k} L={v['L']}/{v['episodes']} ep" for k, v in a["recon"]["libs"].items())
                 + (f"; client decisions {a['client_decisions']} {a['client_hit_mix']}" if a.get("client_decisions") is not None else "")
                 + (f"; WARN {a['warnings']}" if a["warnings"] else ""))
    L.append("")
    L.append("Definitions: spell = >= 3 consecutive served decisions with identical top-1 (a MISS breaks the run); action-repeat spell = "
             f"consecutive served heads within RMS {res['args']['act_eps']} sigma; first-spell classes T terminal row / G |gripper vote| < .5 / "
             "Z translation norm < .6 sigma / P library pause row / H none (priority T>G>Z>P); flips g0 = served gripper sign changes at step 0 "
             "between decisions, flips exec = sign changes along the executed 5-step gripper series; w_term_late = weight of the served set on "
             "terminal library rows over the last third of each episode; ep_eff = 1 / sum_e mass_e^2 of the served set; Wilson 95 % CI; paired: "
             "exact McNemar, multinomial bootstrap of the discordant counts, Newcombe (1998) method 10, and the normal approximation.")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------------------------------- main
def _worker(args):
    run, arm, kw = args
    try:
        return load_arm(pathlib.Path(run), arm, **kw)
    except SystemExit as e:
        return {"error": str(e), "arm": arm, "run_root": str(run)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-root", action="append", required=True, help="run root(s); arms are searched in order")
    ap.add_argument("arms", nargs="+", help="<arm> or <run>:<arm>")
    ap.add_argument("--ref", default=None, help="reference arm (<arm> or <run>:<arm>) for the paired comparison of every arm")
    ap.add_argument("--chain", action="store_true", help="also pair each arm with the previous arm in the list (decomposition)")
    ap.add_argument("--pilot", action="store_true", help=f"R3 pilot subset per suite: tasks {PILOT_TASKS}, ep_idx 0-19")
    ap.add_argument("--tasks", default=None, help="task ids, e.g. 6,9,0,4,1")
    ap.add_argument("--episodes", default=None, help="ep_idx list / ranges, e.g. 0-19 or 0,5,10")
    ap.add_argument("--manifest", help="exact pairs / stratified pilot manifest (same design for both paired arms)")
    ap.add_argument("--ledger", action="store_true", help="add R4 ledger even for legacy logs")
    ap.add_argument("--cost-table", help="stage cost JSON override")
    ap.add_argument("--json", default=None, help="write the full result JSON here")
    ap.add_argument("--md", default=None, help="write the markdown report here")
    ap.add_argument("--store", default=None, help="store root (default: the server's logged root, else /dev/shm/offline_search_store)")
    ap.add_argument("--act-eps", type=float, default=0.1, help="served-action repeat threshold (sigma RMS over [:5,:7])")
    ap.add_argument("--ir-ref", default=None, help="pure-inference reference arm for the measured latency ratio")
    ap.add_argument("--ir-ref-ms", type=float, default=None, help="pure-inference mean server infer_ms (alternative to --ir-ref)")
    ap.add_argument("--recon", action="append", default=[], help="METHOD_NAME_OR_PREFIX=RULE (top1 | mean:K | awm[:KREF] | g3)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default=None, help="optional cache for the library pause masks")
    ap.add_argument("--no-client", action="store_true", help="skip per_step.jsonl (client hit mix)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    roots = [pathlib.Path(r) for r in a.run_root]
    for r in roots:
        if not (r / "runs").is_dir():
            raise SystemExit(f"run root {r} has no runs/ directory")
    override = {}
    for spec in a.recon:
        if "=" not in spec:
            raise SystemExit(f"--recon {spec!r}: use METHOD=RULE")
        k, v = spec.split("=", 1)
        override[k] = v
    tasks = parse_int_list(a.tasks)
    episodes = parse_int_list(a.episodes)
    if a.pilot and (tasks is not None or episodes is not None):
        raise SystemExit("--pilot excludes --tasks / --episodes")
    if a.manifest and (a.pilot or tasks is not None or episodes is not None):
        raise SystemExit("--manifest excludes --pilot / --tasks / --episodes")
    cache_dir = pathlib.Path(a.cache_dir) if a.cache_dir else None

    targets = []
    for spec in a.arms:
        targets.append(find_arm(roots, spec))
    ref_t = find_arm(roots, a.ref) if a.ref else None
    ir_t = find_arm(roots, a.ir_ref) if a.ir_ref else None
    jobs = list(targets)
    for t in (ref_t, ir_t):
        if t is not None and t not in jobs:
            jobs.append(t)
    if len({root_pool(run) for run, _ in jobs}) != 1:
        raise ValueError('POOL_MISMATCH: one KPI report cannot mix A and B runs')

    def kw_for(run, arm):
        tk, epi = tasks, episodes
        if a.pilot:
            suite = SUITE_SHORT.get(arm_meta(run, arm).get("suite_short") or arm_meta(run, arm).get("suite") or "", None)
            if suite is None:
                parts = arm.split("_")
                suite = SUITE_SHORT.get(parts[2] if len(parts) > 2 else "", None)
            if suite not in PILOT_TASKS:
                raise SystemExit(f"{arm}: --pilot needs a known suite (spatial | l10), got {suite!r}")
            tk, epi = PILOT_TASKS[suite], PILOT_EPISODES
        return dict(store=a.store, tasks=tk, episodes=epi, act_eps=a.act_eps, recon_override=override or None,
                    cache_dir=cache_dir, want_client=not a.no_client, manifest=a.manifest,
                    include_ledger=a.ledger, cost_table=a.cost_table)

    work = [(str(run), arm, kw_for(run, arm)) for run, arm in jobs]
    nw = max(1, min(a.workers, len(work)))
    if nw > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(nw) as pool:
            loaded = pool.map(_worker, work)
    else:
        loaded = [_worker(w) for w in work]
    by_key = {}
    for (run, arm), L in zip(jobs, loaded):
        if "error" in L:
            raise SystemExit(f"{arm}: {L['error']}")
        by_key[(str(run), arm)] = L
    ir_ref_ms = a.ir_ref_ms
    if ir_t is not None:
        ir_ref_ms = _mean(by_key[(str(ir_t[0]), ir_t[1])]["dec"]["infer_ms"])
    arms_out, paired_out = [], []
    prev = None
    for run, arm in targets:
        A = by_key[(str(run), arm)]
        arms_out.append(arm_kpis(A, ir_ref_ms))
        if ref_t is not None and (str(ref_t[0]), ref_t[1]) != (str(run), arm):
            paired_out.append(paired(by_key[(str(ref_t[0]), ref_t[1])], A, a.boot, a.seed))
        if a.chain and prev is not None:
            paired_out.append(paired(prev, A, a.boot, a.seed))
        prev = A
    subset_note = "R3 pilot subset (5 trap tasks x ep_idx 0-19 per suite)" if a.pilot else (
        f"tasks {tasks} episodes {episodes}" if (tasks is not None or episodes is not None) else "all complete episodes")
    res = {"schema": SCHEMA, "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "args": {"run_roots": [str(r) for r in roots], "arms": a.arms, "ref": a.ref, "chain": a.chain, "pilot": a.pilot,
                    "tasks": tasks, "episodes": episodes, "act_eps": a.act_eps, "ir_ref": a.ir_ref, "ir_ref_ms": ir_ref_ms,
                    "recon": override, "boot": a.boot, "seed": a.seed, "subset_note": subset_note},
           "arms": arms_out, "paired": paired_out, "wall_s": round(time.perf_counter() - t0, 2)}
    res = _to_native(res)
    md = md_report(res)
    if a.json:
        p = pathlib.Path(a.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=1))
    if a.md:
        p = pathlib.Path(a.md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(md)
    if not a.quiet:
        sys.stdout.write(md)
        sys.stdout.write(f"\n[kpi] {len(arms_out)} arms, {len(paired_out)} paired comparisons, wall {res['wall_s']} s\n")
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    sys.exit(main())
