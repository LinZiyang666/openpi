"""Shared layer of the profile tools: store / run readers, dims + metric shims, stats, tables.

Everything here is read-only w.r.t. the stores; the only writes go through `write_json` /
`write_csv` into an explicit output directory (default `<root>/profile_cache/reports/...`).

Store layout (protocol §5.1 as actually written by R0-A/B/D):
  queries/<m>_<s>_<a>/episodes.json            [{uid, file, task, task_id, init, success, num_steps, start, end}]
  queries/<m>_<s>_<a>/<field>.npy              ep, step, key_v0, key_v1, rs, raw_state, a_inf, a_hit, a_exec,
                                                rec_top1, rec_score, rec_perfield   (``rows.<field>.npy`` also accepted)
  library/<m>_<s>/<name>/<field>.npy + manifest.json   key_v0, key_v1, rs, action, task_id, episode, step, ep_len,
                                                progress, success, prev, next (+ ids.json, episodes.json, tok/)
  floor/<m>_<s>/step0_pairs.npz (err, ...), resample.npz (a_recorded, a_fresh[S,K,H,32], third, ...)
Runner output (harness/run.py): <run>/<method>/<cell>.npz (+ <cell>.json, progress.jsonl, DONE|ERROR).
"""
from __future__ import annotations

import csv
import json
import math
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

SHM_ROOT = Path("/dev/shm/offline_search_store")          # hot copy pinned in RAM (tok/, profile_cache/ -> SSD)
SSD_ROOT = Path("/home/weiland/trace_runs/offline_search_store")
DEFAULT_ROOT = SHM_ROOT if SHM_ROOT.exists() else SSD_ROOT
SMOKE_ROOT = Path("/home/weiland/trace_runs/offline_search_store_smoke")
DEFAULT_PROCS = os.cpu_count() or 1                       # owner 2026-09-26: use the whole machine; --procs lowers it
MODELS = ("pi05", "groot")
SUITES = ("spatial", "l10")
ARMS = ("inf", "cache")
CELLS = tuple(f"{m}_{s}_{a}" for m in MODELS for s in SUITES for a in ARMS)
MODEL_SUITES = tuple(f"{m}_{s}" for m in MODELS for s in SUITES)
THIRDS = ("early", "mid", "late")
TRACE_CONFIG_DIR = Path(__file__).resolve().parents[2] / "trace_dual" / "config"
SUITE_CODE = {"spatial": "sp", "l10": "l10"}

# ---------------------------------------------------------------------------------- dims / metric shim
# The frozen harness (exp/offline_search/harness) is the authority for the valid dims, the err metric, sigma_d,
# the floor thresholds and the phase definition; everything below delegates to it when importable. The local
# fallbacks carry identical math (verified in tests) so the tools still work on a checkout without the harness.
try:
    from exp.offline_search.harness import dims as _hdims  # type: ignore
    from exp.offline_search.harness import metrics as _hmetrics  # type: ignore
except Exception:  # noqa: BLE001  pragma: no cover
    _hdims = _hmetrics = None

EXEC_STEPS: int = int(getattr(_hdims, "EXEC_STEPS", 5))
ACT_VALID: slice = getattr(_hdims, "ACT_VALID", slice(0, 7))
GRIPPER_DIM: int = int(getattr(_hdims, "GRIPPER_DIM", 6))
RS_VALID_N = dict(getattr(_hdims, "RS_VALID", {"pi05": 8, "groot": 8}))   # valid leading robot_state dims
N_ACT = ACT_VALID.stop - (ACT_VALID.start or 0)
DIMS_SOURCE = "harness.dims" if _hdims is not None else "local-copy"
METRIC_SOURCE = "harness.metrics" if _hmetrics is not None else "local-copy"


def exec_block(a) -> np.ndarray:
    """Executed, valid part of an action chunk: [..., :5, :7] as float64 (harness dims.valid_action)."""
    return np.asarray(a[..., :EXEC_STEPS, ACT_VALID], np.float64)


def valid_rs(x, model: str | None = None) -> np.ndarray:
    """Valid robot_state dims (pi05 dims 8..31 are exactly 0 padding; groot all 8)."""
    n = RS_VALID_N.get(model, min(RS_VALID_N.values())) if model else min(RS_VALID_N.values())
    return np.asarray(x[..., :n])


def lib_sigma(action) -> np.ndarray:
    """sigma_d: std (ddof=0) of library *current* action[:, :5, d], d < 7 (== harness store.action_sigma)."""
    x = exec_block(action).reshape(-1, N_ACT)
    return x.std(0)


def scaled_seg(a, sigma) -> np.ndarray:
    """[..., H, 32] -> [..., 35] float64, already divided by sigma (for GEMM-style err matrices)."""
    x = exec_block(a) / np.asarray(sigma, np.float64)
    return x.reshape(x.shape[:-2] + (EXEC_STEPS * N_ACT,))


def seg_err(a_hat, a_ref, sigma) -> np.ndarray:
    """err = sqrt(mean_{t<5,d<7} (((a_hat - a_ref) / sigma_d)^2)) (harness metrics.err_seg on seg())."""
    if _hmetrics is not None:
        return _hmetrics.err_seg(_hmetrics.seg(a_hat), _hmetrics.seg(a_ref), np.asarray(sigma, np.float64))
    d = (exec_block(a_hat) - exec_block(a_ref)) / np.asarray(sigma, np.float64)
    return np.sqrt(np.mean(d * d, axis=(-2, -1)))


def err_matrix(qs: np.ndarray, cs: np.ndarray) -> np.ndarray:
    """Pairwise err between scaled segments qs [nq, 35] and cs [nc, 35] -> [nq, nc] float64 (same value as
    seg_err up to ~1e-12; tests check it against the harness)."""
    d2 = (qs * qs).sum(1)[:, None] + (cs * cs).sum(1)[None, :] - 2.0 * (qs @ cs.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2 / qs.shape[1])


def phase_of(step, num_steps) -> np.ndarray:
    """Episode phase p = step / (num_steps - 1) (harness metrics.phase_of; library progress uses the same)."""
    n = np.asarray(num_steps, np.float64)
    return np.where(n > 1, np.asarray(step, np.float64) / np.maximum(n - 1, 1), 0.0)


def grip_sign(a) -> np.ndarray:
    """Gripper sign (dim 6 of the executed steps) as bool (>= 0 is 'open' side)."""
    return np.asarray(a[..., :EXEC_STEPS, GRIPPER_DIM]) >= 0


# ------------------------------------------------------------------------------------------- cells
def parse_cell(cell: str) -> tuple[str, str, str]:
    m, s, a = cell.split("_")
    if m not in MODELS or s not in SUITES or a not in ARMS:
        raise ValueError(f"bad cell {cell!r}")
    return m, s, a


def cell_ms(cell: str) -> str:
    m, s, _ = parse_cell(cell)
    return f"{m}_{s}"


def expand_cells(spec: str | None) -> list[str]:
    """'all' | comma list of cells | model_suite prefixes (pi05_spatial -> both arms) | arm ('inf')."""
    if not spec or spec == "all":
        return list(CELLS)
    out: list[str] = []
    for tok in spec.split(","):
        tok = tok.strip()
        hits = [c for c in CELLS if c == tok or c.startswith(tok + "_") or c.endswith("_" + tok)
                or tok in c.split("_")]
        if not hits:
            raise ValueError(f"no cell matches {tok!r}")
        out += [c for c in hits if c not in out]
    return out


def step_third(step, num_steps) -> np.ndarray:
    """Same definition as r0/floor_common.step_third: min(2, (3 * k) // n)."""
    return np.minimum(2, (3 * np.asarray(step, np.int64)) // np.maximum(np.asarray(num_steps, np.int64), 1))


# ------------------------------------------------------------------------------------------- stores
class RowStore:
    """Lazy mmap access to `<field>.npy` (or `rows.<field>.npy`) inside a directory."""

    def __init__(self, d):
        self.dir = Path(d)
        self._c: dict[str, np.ndarray] = {}

    def path(self, f: str) -> Path | None:
        for cand in (self.dir / f"{f}.npy", self.dir / f"rows.{f}.npy"):
            if cand.exists():
                return cand
        return None

    def has(self, f: str) -> bool:
        return f in self._c or self.path(f) is not None

    def get(self, f: str) -> np.ndarray:
        if f not in self._c:
            p = self.path(f)
            if p is None:
                raise KeyError(f"{self.dir}: no field {f!r} (have {self.fields()})")
            self._c[f] = np.load(p, mmap_mode="r")
        return self._c[f]

    __getitem__ = get

    def fields(self) -> list[str]:
        out = set()
        for p in self.dir.glob("*.npy"):
            n = p.name[:-4]
            out.add(n[5:] if n.startswith("rows.") else n)
        return sorted(out)


def _read_json(p: Path):
    with open(p) as f:
        return json.load(f)


class QueryCell:
    """One query cell of the store with per-row episode metadata derived from episodes.json."""

    def __init__(self, root, cell: str):
        self.root = Path(root)
        self.cell = cell
        self.m, self.s, self.a = parse_cell(cell)
        self.ms = f"{self.m}_{self.s}"
        self.dir = self.root / "queries" / cell
        if not self.dir.exists():
            raise FileNotFoundError(self.dir)
        self.rows = RowStore(self.dir)
        eps = _read_json(self.dir / "episodes.json")
        if isinstance(eps, dict):
            eps = eps.get("episodes", list(eps.values()))
        for e in eps:  # accept rows=[start, end] as well
            if "start" not in e and "rows" in e:
                e["start"], e["end"] = e["rows"]
        self.episodes: list[dict] = eps
        self.n = int(max(e["end"] for e in eps)) if eps else 0
        ep = np.full(self.n, -1, np.int32)
        for i, e in enumerate(eps):
            ep[e["start"]:e["end"]] = i
        self.ep = ep
        self.ep_task = np.array([int(e["task_id"]) for e in eps], np.int32)
        self.ep_success = np.array([bool(e.get("success", False)) for e in eps])
        self.ep_len = np.array([int(e.get("num_steps", e["end"] - e["start"])) for e in eps], np.int32)
        self.ep_start = np.array([int(e["start"]) for e in eps], np.int64)
        self.task_id = self.ep_task[ep]
        self.num_steps = self.ep_len[ep]
        self.success = self.ep_success[ep]
        if self.rows.has("step"):
            self.step = np.asarray(self.rows["step"], np.int32)
        else:
            self.step = (np.arange(self.n) - self.ep_start[ep]).astype(np.int32)
        self.p = phase_of(self.step, self.num_steps)          # step / (num_steps - 1), metric side only
        self.third = step_third(self.step, self.num_steps).astype(np.int8)
        self.uid_to_ep = {str(e.get("uid")): i for i, e in enumerate(eps)}
        self.manifest = _read_json(self.dir / "manifest.json") if (self.dir / "manifest.json").exists() else None

    def __getitem__(self, f):
        return self.rows[f]

    def task_str(self, tid: int) -> str:
        for e in self.episodes:
            if int(e["task_id"]) == int(tid):
                return str(e.get("task", tid))
        return str(tid)


class Library:
    """One library variant (current | bpool_all | any dir with the same row layout)."""

    def __init__(self, path, name: str | None = None):
        self.dir = Path(path)
        if not self.dir.exists():
            raise FileNotFoundError(self.dir)
        self.name = name or self.dir.name
        self.rows = RowStore(self.dir)
        self.meta = {}
        for mp_ in (self.dir / "manifest.json", self.dir / "meta.json"):
            if mp_.exists():
                self.meta = _read_json(mp_)
                break
        self._task_rows: dict[int, np.ndarray] | None = None

    def __getitem__(self, f):
        return self.rows[f]

    def has(self, f):
        return self.rows.has(f)

    @property
    def n(self) -> int:
        for f in ("task_id", "action", "key_v0", "rs"):
            if self.rows.has(f):
                return int(self.rows[f].shape[0])
        raise KeyError(f"{self.dir}: cannot infer row count")

    @property
    def task_id(self) -> np.ndarray:
        return np.asarray(self.rows["task_id"], np.int32)

    def task_rows(self, tid: int) -> np.ndarray:
        if self._task_rows is None:
            t = self.task_id
            self._task_rows = {int(k): np.flatnonzero(t == k) for k in np.unique(t)}
        return self._task_rows.get(int(tid), np.zeros(0, np.int64))

    def opt(self, f: str, default=None):
        return np.asarray(self.rows[f]) if self.rows.has(f) else default

    def describe_rows(self, rows) -> list[dict]:
        """Human-oriented per-row info (episode, step, progress, traj, success) for display."""
        rows = np.asarray(rows)
        out = []
        cols = {f: self.rows[f] for f in ("episode", "step", "progress", "traj", "success", "ep_len") if self.rows.has(f)}
        for r in rows:
            if r < 0 or r >= self.n:
                out.append({"row": int(r)})
                continue
            d = {"row": int(r)}
            for f, a in cols.items():
                v = a[r]
                d[f] = v.item() if hasattr(v, "item") else v
            out.append(d)
        return out


def library_dir(root, ms: str, name: str) -> Path:
    p = Path(name)
    if p.is_absolute() or (p.exists() and any((p / f).exists() for f in ("manifest.json", "meta.json", "action.npy"))):
        return p
    return Path(root) / "library" / ms / name


def open_library(root, ms: str, name: str) -> Library | None:
    d = library_dir(root, ms, name)
    return Library(d, name=Path(name).name) if d.exists() else None


def ready(root, part: str) -> bool:
    """READY marker of queries/ or library/ (file READY, READY.json or .READY)."""
    d = Path(root) / part
    return any((d / n).exists() for n in ("READY", "READY.json", ".READY", "READY.txt"))


# ----------------------------------------------------------------------------------------- floor
@dataclass
class Floor:
    """Teacher noise-floor medians per step bin (the `indist` / `good` threshold), harness semantics:
    step0_pairs -> early bin, resample -> by bin, a bin with < 20 samples falls back to the pooled median."""
    ms: str
    median: float                                  # pooled median over every floor sample ("all")
    by_third: dict[int, float] | None = None       # {0,1,2} -> median of that bin (after the fallback rule)
    source: str = ""
    n: int = 0
    detail: dict = field(default_factory=dict)

    def thr(self, third, mode: str = "third") -> np.ndarray:
        third = np.asarray(third)
        if mode == "third" and self.by_third:
            lut = np.array([self.by_third.get(t, self.median) for t in range(3)], np.float64)
            return lut[np.clip(third, 0, 2)]
        return np.full(third.shape, self.median, np.float64)


def load_floor(root, ms: str, sigma=None, queries_fallback: bool = True) -> Floor | None:
    """Floor thresholds for model x suite. Uses harness metrics.load_floor (same thresholds as the runner's
    `indist`); without the harness, the same rule is applied locally. When no floor file exists at all, falls
    back to the step-0 inf/cache pairs recomputed from the query store (source 'queries_step0_pairs')."""
    root = Path(root)
    if sigma is None:
        lib = open_library(root, ms, "current")
        sigma = lib_sigma(lib["action"]) if lib is not None else None
    if sigma is not None:
        fl = _hmetrics.load_floor(root, ms, np.asarray(sigma, np.float64)) if _hmetrics is not None else \
            _local_floor(root, ms, np.asarray(sigma, np.float64))
        if fl["available"]:
            med = fl["median"]
            return Floor(ms=ms, median=float(med["all"]), by_third={b: float(med[THIRDS[b]]) for b in range(3)},
                         source=fl["source"], n=int(fl["n"].get("all", 0)),
                         detail={"n": fl["n"], "p90": fl.get("p90"), "bin_fallback": fl.get("bin_fallback")})
    if queries_fallback:
        e = _floor_from_queries(root, ms, sigma)
        if e is not None and e.size:
            med = float(np.median(e))
            return Floor(ms=ms, median=med, by_third={0: med, 1: med, 2: med}, source="queries_step0_pairs",
                         n=int(e.size))
    return None


def _local_floor(root: Path, ms: str, sigma: np.ndarray) -> dict:
    """Local copy of harness metrics.load_floor (used only when the harness is not importable)."""
    d = root / "floor" / ms
    samples = {0: [], 1: [], 2: []}
    src = []
    if (d / "step0_pairs.npz").exists():
        with np.load(d / "step0_pairs.npz") as z:
            e = seg_err(z["a_inf_arm"], z["a_cache_arm"], sigma) if "a_inf_arm" in z.files else np.asarray(z["err"], np.float64)
        samples[0].append(e.ravel()); src.append("step0_pairs")
    if (d / "resample.npz").exists():
        with np.load(d / "resample.npz") as z:
            th = np.asarray(z["third"], np.int64)
            e = seg_err(z["a_fresh"], z["a_recorded"][:, None], sigma) if "a_fresh" in z.files else np.asarray(z["err"], np.float64)
            e = e.reshape(th.shape[0], -1)
        for b in range(3):
            samples[b].append(e[th == b].ravel())
        src.append("resample")
    samples = {b: (np.concatenate(v) if v else np.zeros(0)) for b, v in samples.items()}
    allv = np.concatenate(list(samples.values()))
    out = {"available": allv.size > 0, "source": "+".join(src) or "none", "n": {THIRDS[b]: int(samples[b].size) for b in range(3)},
           "median": {}, "p90": {}, "bin_fallback": {}}
    if allv.size:
        out["median"]["all"] = float(np.median(allv)); out["n"]["all"] = int(allv.size)
        for b in range(3):
            ok = samples[b].size >= 20
            out["median"][THIRDS[b]] = float(np.median(samples[b])) if ok else out["median"]["all"]
            out["bin_fallback"][THIRDS[b]] = not ok
    return out


def _floor_from_queries(root: Path, ms: str, sigma) -> np.ndarray | None:
    try:
        qi, qc = QueryCell(root, ms + "_inf"), QueryCell(root, ms + "_cache")
    except FileNotFoundError:
        return None
    if sigma is None:
        lib = open_library(root, ms, "current")
        if lib is None:
            return None
        sigma = lib_sigma(lib["action"])
    key_c = {(int(e["task_id"]), int(e["init"])): e["start"] for e in qc.episodes}
    a, b = [], []
    for e in qi.episodes:
        k = (int(e["task_id"]), int(e["init"]))
        if k in key_c:
            a.append(e["start"]); b.append(key_c[k])
    if not a:
        return None
    return seg_err(qi["a_inf"][np.array(a)], qc["a_inf"][np.array(b)], sigma)


# --------------------------------------------------------------------------------------- run output
class RunCell:
    """Per-decision arrays of one method x cell written by the harness runner."""

    def __init__(self, method_dir, cell: str):
        self.dir = Path(method_dir)
        self.method = self.dir.name
        self.cell = cell
        p = self.dir / f"{cell}.npz"
        with np.load(p, allow_pickle=False) as z:
            self.d = {k: z[k] for k in z.files}
        pj = self.dir / f"{cell}.json"
        self.summary = _read_json(pj) if pj.exists() else {}
        lib = self.d.pop("library", None)
        if lib is None:
            lib = self.summary.get("library") or (self.summary.get("attrs") or {}).get("library")
        self.library = str(lib.item() if hasattr(lib, "item") else lib) if lib is not None else "current"
        ln = self.d.pop("lib_names", None)
        self.lib_names = [str(x) for x in np.atleast_1d(ln)] if ln is not None else [self.library]
        n = len(self.d["row"]) if "row" in self.d else len(next(iter(self.d.values())))
        self.n = int(n)
        self.lib_code = np.asarray(self.d.get("lib_code", np.zeros(self.n, np.int8)), np.int64)

    @property
    def mixed(self) -> bool:
        """True when decisions refer to more than one library (runner writes library='mixed')."""
        return len(self.lib_names) > 1

    def lib_of(self, pos) -> str:
        return self.lib_names[int(self.lib_code[pos])]

    def __getitem__(self, k):
        return self.d[k]

    def get(self, k, default=None):
        return self.d.get(k, default)

    def has(self, k) -> bool:
        return k in self.d

    @property
    def extras(self) -> list[str]:
        return sorted(k for k in self.d if k.startswith("x_"))

    @property
    def rows(self) -> np.ndarray:
        return np.asarray(self.d["row"], np.int64)


def method_dirs(path, method: str | None = None) -> list[Path]:
    """`path` is either a method dir (holds <cell>.npz) or a run root holding method dirs."""
    p = Path(path)
    if any((p / f"{c}.npz").exists() for c in CELLS):
        return [p]
    subs = sorted(d for d in p.iterdir() if d.is_dir() and any((d / f"{c}.npz").exists() for c in CELLS)) if p.exists() else []
    if method:
        subs = [d for d in subs if d.name in method.split(",")]
    return subs


def run_cells(method_dir, cells: Iterable[str] | None = None) -> list[str]:
    md = Path(method_dir)
    want = list(cells) if cells else list(CELLS)
    return [c for c in want if (md / f"{c}.npz").exists()]


# ------------------------------------------------------------------------------------ coverage cache
def coverage_path(root, lib: str) -> Path:
    return Path(root) / "profile_cache" / f"coverage_{Path(lib).name}.npz"


class CoverageCache:
    """Reader for profile_cache/coverage_<lib>.npz (see coverage.py for the fields)."""

    def __init__(self, path):
        self.path = Path(path)
        self._z = np.load(self.path, allow_pickle=False)
        self._c: dict[str, np.ndarray] = {}
        self.meta = json.loads(str(self._z["__meta__"])) if "__meta__" in self._z.files else {}
        self.cells = sorted({k.split("__")[0] for k in self._z.files if "__" in k and not k.startswith("__")})

    def has(self, cell) -> bool:
        return cell in self.cells

    def get(self, cell: str, f: str):
        k = f"{cell}__{f}"
        if k not in self._c:
            self._c[k] = self._z[k]
        return self._c[k]

    def fields(self, cell) -> list[str]:
        pre = cell + "__"
        return sorted(k[len(pre):] for k in self._z.files if k.startswith(pre))


def load_coverage(root, lib: str) -> CoverageCache | None:
    p = coverage_path(root, lib)
    return CoverageCache(p) if p.exists() else None


# ------------------------------------------------------------------------------------------- stats
def qstats(x, qs=(0.1, 0.25, 0.5, 0.75, 0.9)) -> dict:
    x = np.asarray(x, np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    out = {"n": int(x.size), "mean": float(x.mean())}
    for q, v in zip(qs, np.quantile(x, qs)):
        out[f"p{int(round(q * 100)):02d}"] = float(v)
    return out


TIE_AWARE = False   # default AURC semantics = harness metrics.risk_coverage (stable sort, ties in row order)


def risk_curve(conf, risk, tie_aware: bool | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Risk-coverage curve accepting the top-i decisions by confidence (desc), i = 1..N.
    Default (tie_aware=False) = harness metrics.risk_coverage: stable sort, so tied confidences are accepted in
    row order. tie_aware=True averages over the order inside a tie group (a constant confidence then gives a flat
    curve at mean(risk)); use it for methods with discrete confidences. NaN confidence ranks last."""
    conf = np.asarray(conf, np.float64)
    risk = np.asarray(risk, np.float64)
    if conf.size == 0:
        return np.zeros(0), np.zeros(0)
    order = conf_order(conf)
    c = np.where(np.isfinite(conf), conf, -np.inf)
    return sorted_risk_curve(c[order], risk[order], tie_aware)


def conf_order(conf) -> np.ndarray:
    """Indices sorting by confidence descending (NaN last, stable) == np.argsort(-conf, kind='stable')."""
    conf = np.asarray(conf, np.float64)
    c = np.where(np.isfinite(conf), conf, -np.inf)
    return np.argsort(-c, kind="stable")


def sorted_risk_curve(cs: np.ndarray, rs: np.ndarray, tie_aware: bool | None = None) -> tuple[np.ndarray, np.ndarray]:
    """risk_curve on inputs already sorted by confidence descending (cs uses -inf for NaN)."""
    tie_aware = TIE_AWARE if tie_aware is None else tie_aware
    n = cs.size
    if n == 0:
        return np.zeros(0), np.zeros(0)
    i = np.arange(1, n + 1)
    if not tie_aware:
        return i / n, np.cumsum(rs) / i
    new = np.ones(n, bool)
    new[1:] = cs[1:] != cs[:-1]
    starts = np.flatnonzero(new)
    gid = np.cumsum(new) - 1
    cnt = np.diff(np.r_[starts, n])
    gsum = np.add.reduceat(rs, starts)
    prefix = np.r_[0.0, np.cumsum(gsum)[:-1]]
    j = i - starts[gid]
    ecum = prefix[gid] + j * (gsum / cnt)[gid]
    return i / n, ecum / i


def aurc(conf, risk, tie_aware: bool | None = None) -> float:
    _, r = risk_curve(conf, risk, tie_aware)
    return float(r.mean()) if r.size else float("nan")


def risk_at(conf, risk, covs=(0.3, 0.5, 0.7, 0.9), tie_aware: bool | None = None) -> dict:
    cov, r = risk_curve(conf, risk, tie_aware)
    n = r.size
    return {f"risk@{int(c * 100)}": (float(r[max(int(math.ceil(c * n)), 1) - 1]) if n else float("nan")) for c in covs}


def rankdata_rows(x: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    return rankdata(x, axis=-1)


def spearman_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Spearman rho between a[i] and b[i] (average ranks for ties)."""
    ra, rb = rankdata_rows(a), rankdata_rows(b)
    ra -= ra.mean(-1, keepdims=True)
    rb -= rb.mean(-1, keepdims=True)
    den = np.sqrt((ra * ra).sum(-1) * (rb * rb).sum(-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (ra * rb).sum(-1) / den, np.nan)


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """P(score_pos > score_neg) with ties counted 1/2 (Mann-Whitney)."""
    from scipy.stats import rankdata
    score = np.asarray(score, np.float64)
    label = np.asarray(label, bool)
    npos, nneg = int(label.sum()), int((~label).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def auroc_rows(S: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Per-row AUROC of scores S[i] for labels G[i] (ties 1/2); rows lacking either class are dropped."""
    r = rankdata_rows(np.asarray(S, np.float64))
    G = np.asarray(G, bool)
    npos = G.sum(1).astype(np.float64)
    nneg = G.shape[1] - npos
    ok = (npos > 0) & (nneg > 0)
    a = ((r * G).sum(1) - npos * (npos + 1) / 2) / np.maximum(npos * nneg, 1)
    return a[ok]


class EpisodeBootstrap:
    """Episode-level bootstrap: resample episodes with replacement, keep all their decisions."""

    def __init__(self, ep: np.ndarray, n_rep: int = 2000, seed: int = 0):
        ep = np.asarray(ep)
        self.uniq, inv = np.unique(ep, return_inverse=True)
        self.E = self.uniq.size
        order = np.argsort(inv, kind="stable")
        bounds = np.r_[0, np.cumsum(np.bincount(inv, minlength=self.E))]
        self.rows_of = [order[bounds[i]:bounds[i + 1]] for i in range(self.E)]
        self.inv = inv
        rng = np.random.default_rng(seed)
        self.draws = rng.integers(0, self.E, size=(n_rep, self.E))
        self.W = np.stack([np.bincount(d, minlength=self.E) for d in self.draws]).astype(np.float64)

    def mean_ci(self, x: np.ndarray, alpha=0.05) -> dict:
        x = np.asarray(x, np.float64)
        s = np.bincount(self.inv, weights=x, minlength=self.E)
        c = np.bincount(self.inv, minlength=self.E).astype(np.float64)
        reps = (self.W @ s) / np.maximum(self.W @ c, 1)
        return {"est": float(x.mean()), "lo": float(np.quantile(reps, alpha / 2)), "hi": float(np.quantile(reps, 1 - alpha / 2)),
                "p_le0": float((reps <= 0).mean())}

    def stat_ci(self, fn: Callable[[np.ndarray], float], alpha=0.05) -> dict:
        """Generic: fn(row_index_array) -> scalar, evaluated on each resample."""
        est = fn(np.arange(self.inv.size))
        reps = np.empty(len(self.draws))
        for k, d in enumerate(self.draws):
            reps[k] = fn(np.concatenate([self.rows_of[i] for i in d]))
        return {"est": float(est), "lo": float(np.quantile(reps, alpha / 2)), "hi": float(np.quantile(reps, 1 - alpha / 2)),
                "p_le0": float((reps <= 0).mean())}


# ------------------------------------------------------------------------------------------- slices
SLICE_ORDER = ("task", "third", "outcome", "grip", "conf_dec", "cand", "good")


def grip_transition(qc: QueryCell, rows: np.ndarray) -> np.ndarray:
    """Teacher (a_inf) gripper sign changes inside the executed 5 steps, or between the previous
    decision's last executed step and this decision's first step (same episode)."""
    a = qc["a_inf"]
    rows = np.asarray(rows, np.int64)
    g = grip_sign(a[rows])  # [n, 5]
    within = (g[:, 1:] != g[:, :-1]).any(1)
    has_prev = qc.step[rows] > 0
    prev = np.where(has_prev, rows - 1, rows)
    gp = grip_sign(a[prev])[:, -1]
    return within | (has_prev & (gp != g[:, 0]))


def decile_labels(x: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    x = np.asarray(x, np.float64)
    x = np.where(np.isfinite(x), x, -np.inf)
    r = (rankdata(x) - 1) / max(x.size, 1)
    d = np.minimum((r * 10).astype(int), 9)
    return np.array([f"d{k}" for k in d])


def bucket_labels(x: np.ndarray, edges: Sequence[float], fmt="{:g}", integer=False) -> np.ndarray:
    """Half-open buckets [e_i, e_{i+1}); integer=True labels them inclusively (e.g. '1-4', '20+')."""
    x = np.asarray(x)
    edges = list(edges)
    names = []
    lo = None
    for e in edges:
        if integer:
            e = int(e)
            names.append((f"<{e}" if e > 1 else "0") if lo is None else (f"{lo}" if e - 1 == lo else f"{lo}-{e - 1}"))
        else:
            names.append(f"<{fmt.format(e)}" if lo is None else f"{fmt.format(lo)}-{fmt.format(e)}")
        lo = e
    if integer:
        names.append(f"{lo}+" if lo is not None else "all")
    else:
        names.append(f">={fmt.format(lo)}" if lo is not None else "all")
    idx = np.searchsorted(np.asarray(edges, np.float64), x, side="right")
    return np.array(names, dtype=object)[idx].astype(str)


def decision_slices(qc: QueryCell, rows: np.ndarray, conf: np.ndarray | None = None,
                    n_cand: np.ndarray | None = None, n_good: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Slice labels per decision (metric-side only: success / num_steps are never method inputs)."""
    rows = np.asarray(rows, np.int64)
    out = {
        "task": np.array([f"t{t}" for t in qc.task_id[rows]]),
        "third": np.array(THIRDS)[qc.third[rows]],
        "outcome": np.where(qc.success[rows], "success", "failure"),
        "grip": np.where(grip_transition(qc, rows), "transition", "steady"),
    }
    if conf is not None:
        out["conf_dec"] = decile_labels(conf)
    if n_cand is not None:
        u = np.unique(n_cand)
        edges = np.unique(np.quantile(n_cand, [1 / 3, 2 / 3]).round()) if u.size > 3 else u[1:]
        out["cand"] = bucket_labels(n_cand, edges, integer=True)
    if n_good is not None:
        out["good"] = bucket_labels(n_good, [1, 5, 20], integer=True)
    return out


def slice_sort_key(name: str, value: str):
    if name == "task":
        return int(value[1:]) if value[1:].isdigit() else 0
    if name == "third":
        return THIRDS.index(value) if value in THIRDS else 9
    if name == "conf_dec":
        return int(value[1:])
    if name in ("cand", "good"):
        v = value.lstrip("<>=").rstrip("+")
        try:
            base = float(v.split("-")[0])
        except ValueError:
            base = 0.0
        return (-1 if value.startswith("<") else 0, base)
    return value


# ------------------------------------------------------------------------------------------- B0 cfg
def b0_config(cell: str) -> dict | None:
    """Weights and z-score params of the recorded B0 (harness store.current_params = trace_dual yaml), or None."""
    m, s, a = parse_cell(cell)
    try:
        from exp.offline_search.harness.store import current_params
        cp = current_params(m, s)
        f = cp["fields"]
        return {"w": dict(zip(f, cp["weights"])), "mu": dict(zip(f, cp["mu"])), "sigma": dict(zip(f, cp["sigma"])),
                "source": cp["source"]}
    except Exception:  # noqa: BLE001
        pass
    p = TRACE_CONFIG_DIR / f"tr_{m}_{SUITE_CODE[s]}_{a}.yaml"
    if not p.exists():
        return None
    import yaml
    with open(p) as fh:
        cfg = yaml.safe_load(fh)
    cp1 = cfg["checkpoints"]["cp1"]["search_strategy"]
    norm = cp1["score_normalization"]["fields"]
    return {"w": {f: float(cfg["keys"][f]["weight"]) for f in ("vision_0", "vision_1", "robot_state")},
            "mu": {f: float(norm[f]["params"]["mu"]) for f in norm},
            "sigma": {f: float(norm[f]["params"]["sigma"]) for f in norm}, "source": str(p)}


def b0_fields(qc: QueryCell, row: int, lib: Library, lib_rows: np.ndarray) -> dict[str, np.ndarray]:
    """Raw B0 per-field similarities of query `row` vs library rows: cos on pooled v0/v1, L2 on rs."""
    lib_rows = np.asarray(lib_rows, np.int64)
    out = {}
    for f in ("key_v0", "key_v1"):
        if qc.rows.has(f) and lib.has(f):
            q = np.asarray(qc[f][row], np.float64)
            L = np.asarray(lib[f][lib_rows], np.float64)
            out["cos_" + f[-2:]] = (L @ q) / np.maximum(np.linalg.norm(L, axis=1) * np.linalg.norm(q), 1e-12)
    if qc.rows.has("rs") and lib.has("rs"):
        q = valid_rs(np.asarray(qc["rs"][row], np.float64), qc.m)
        L = valid_rs(np.asarray(lib["rs"][lib_rows], np.float64), qc.m)
        out["rs_l2"] = np.linalg.norm(L - q, axis=1)
    return out


def b0_score(fields: dict[str, np.ndarray], cfg: dict) -> dict[str, np.ndarray]:
    """Normalized per-field scores 0.5*(tanh((x-mu)/sigma)+1) (l2 as -distance) and the fused score."""
    raw = {"vision_0": fields.get("cos_v0"), "vision_1": fields.get("cos_v1"),
           "robot_state": -fields["rs_l2"] if "rs_l2" in fields else None}
    out = {}
    fused = 0.0
    for f, x in raw.items():
        if x is None:
            return out
        z = 0.5 * (np.tanh((x - cfg["mu"][f]) / cfg["sigma"][f]) + 1.0)
        out["n_" + f] = z
        fused = fused + cfg["w"][f] * z
    out["fused"] = fused
    return out


# ------------------------------------------------------------------------------------------- tables
def fmt_val(v, spec: str | None = None) -> str:
    if v is None:
        return "-"
    if isinstance(v, (bool, np.bool_)):
        return "y" if v else "n"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "nan"
        return format(float(v), spec or ".3f")
    return str(v)


def md_table(rows: list[dict], cols: Sequence, title: str | None = None) -> str:
    """cols: list of key | (key, header) | (key, header, fmt)."""
    spec = []
    for c in cols:
        if isinstance(c, str):
            spec.append((c, c, None))
        elif len(c) == 2:
            spec.append((c[0], c[1], None))
        else:
            spec.append(tuple(c))
    lines = []
    if title:
        lines.append(f"### {title}")
    lines.append("| " + " | ".join(h for _, h, _ in spec) + " |")
    lines.append("|" + "|".join("---" for _ in spec) + "|")
    for r in rows:
        lines.append("| " + " | ".join(fmt_val(r.get(k), f) for k, _, f in spec) + " |")
    return "\n".join(lines)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def write_json(path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_jsonable, allow_nan=True))
    tmp.replace(path)
    return path


def write_csv(path, rows: list[dict], cols: Sequence[str] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if cols is None:
        cols = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cols), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _csv_val(r.get(k)) for k in cols})
    return path


def _csv_val(v):
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else f"{float(v):.6g}"
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def report_dir(root, tool: str, tag: str, out: str | None = None, multi: bool = False) -> Path:
    """<root>/profile_cache/reports/<tool>/<tag>, or --out (plus /<tag> when one call reports several methods)."""
    d = (Path(out) / tag if multi else Path(out)) if out else Path(root) / "profile_cache" / "reports" / tool / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def now_local() -> str:
    try:
        from zoneinfo import ZoneInfo
        import datetime as dt
        return dt.datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M:%S CDT")
    except Exception:  # noqa: BLE001
        return time.strftime("%Y-%m-%d %H:%M:%S")


# ------------------------------------------------------------------------------------------- pool
def _worker_init():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def pmap(fn: Callable, items: list, procs: int | None = None) -> list:
    """Order-preserving map over a fork pool of single-threaded workers (CUDA hidden, OMP/MKL/OpenBLAS = 1);
    serial when procs <= 1 or there is a single item. procs defaults to (and is capped at) os.cpu_count()."""
    procs = DEFAULT_PROCS if procs is None else int(procs)
    procs = max(1, min(procs, len(items), DEFAULT_PROCS))
    if procs <= 1:
        return [fn(x) for x in items]
    ctx = mp.get_context("fork")
    with ctx.Pool(procs, initializer=_worker_init) as pool:
        return pool.map(fn, items, chunksize=1)
