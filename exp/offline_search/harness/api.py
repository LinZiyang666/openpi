"""Method API of the offline evaluation harness (frozen after R0).

A method is a class with

    class Method:
        name: str; tier: str                  # tier "T0" zero-training | "T1" closed-form stats | "T2" trained
        def fit(self, lib, ctx) -> None       # once per cell, in the main process
        def reset(self, episode) -> None      # at the start of every episode
        def query(self, q) -> Result          # one decision, episodes fed in time order
        def bytes_per_entry(self) -> float    # library-side storage cost of the representation used

See README.md for the full contract. The views below expose ONLY online-available information: a method cannot
reach a_inf / a_hit / success / num_steps / rec_* through them (the backing object does not even hold those
arrays); touching such a name raises ``ForbiddenAccess``.
"""
from __future__ import annotations

import dataclasses
import math
import pathlib
import time

import numpy as np

from . import dims
from .store import LibraryView, StoreError, action_sigma, current_params, lda_weights  # noqa: F401 (re-export)

TOPK_SAVE = 10                  # top-k columns kept per decision in the output npz
EXTRAS_CAP_BYTES = 4096         # per decision, counted as float32 after flattening
TIERS = ("T0", "T1", "T2")
FORBIDDEN = frozenset({"a_inf", "a_hit", "success", "num_steps", "rec_top1", "rec_score", "rec_perfield",
                       "gt", "full_inference", "full_hit", "file", "episode_len", "ep_len", "progress", "row"})


class ForbiddenAccess(AttributeError):
    """A method touched information that is not available online (GT, outcome, episode length, recorded search)."""


class TokensUnavailable(LookupError):
    """Token / image tensors requested for a row outside the tok subsample (run with --subsample tok)."""


class ContractError(ValueError):
    """A Result (or a method attribute) violates the API contract."""


class SkipCell(Exception):
    """Raise from fit() when a method does not apply to a cell (e.g. no LDA weights for it). The runner records
    the cell as skipped (not an error) and moves on."""


# ------------------------------------------------------------------------------------------ profiler
class _NullSection:
    __slots__ = ()

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


_NULL_SECTION = _NullSection()


class NullProfiler:
    """Zero-cost stand-in used when profiling is off: ``with self.prof.section("x"):`` does nothing."""

    enabled = False

    def section(self, name: str):
        return _NULL_SECTION

    def add(self, name: str, ns: int) -> None:
        pass


NULL_PROFILER = NullProfiler()


class _Section:
    __slots__ = ("p", "name", "t0")

    def __init__(self, p, name):
        self.p = p
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter_ns()
        return None

    def __exit__(self, *exc):
        self.p.samples.setdefault(self.name, []).append(time.perf_counter_ns() - self.t0)
        return False


class Profiler:
    """Section timer. ``with prof.section("key_build"): ...`` records one wall-time sample (ns) per entry."""

    enabled = True

    def __init__(self):
        self.samples: dict[str, list[int]] = {}

    def section(self, name: str):
        return _Section(self, name)

    def add(self, name: str, ns: int) -> None:
        self.samples.setdefault(name, []).append(int(ns))

    def take(self) -> dict:
        s, self.samples = self.samples, {}
        return s

    @staticmethod
    def stats(samples: dict, n_queries: int | None = None) -> dict:
        out = {}
        for k, v in sorted(samples.items()):
            a = np.asarray(v, np.float64) / 1e3  # us
            if a.size == 0:
                continue
            d = {"n": int(a.size), "total_us": float(a.sum()), "mean_us": float(a.mean()),
                 "p50_us": float(np.percentile(a, 50)), "p95_us": float(np.percentile(a, 95))}
            if n_queries:
                d["per_query_us"] = float(a.sum() / n_queries)
            out[k] = d
        return out


# -------------------------------------------------------------------------------------------- result
@dataclasses.dataclass
class Result:
    topk: np.ndarray                      # library row indices (into `library`), best first
    scores: np.ndarray                    # same length as topk (method's own scale, higher = better)
    confidence: float                     # higher = more willing to accept (risk-coverage ordering)
    action: np.ndarray | None = None      # optional synthesized (H, 32) action built ONLY from library actions
    #                                       (unless the method declares uses_nonlibrary_action); only [:5, :7] is
    #                                       scored (dims.valid_action)
    library: str = "current"              # any library stored for this model x suite ("current", "bpool_all",
    #                                       "bpool_cs", ...) or a name registered via ctx.register_library
    extras: dict | None = None            # optional per-decision diagnostics {key: float | ndarray}, <= 4 KB


class Method:
    """Base class (inherit or duck-type). ``self.prof`` is injected by the runner (NullProfiler when off)."""

    name: str = "unnamed"
    tier: str = "T0"
    family: str | None = None      # optional; runner falls back to the fK_<family> directory name
    uses_gt: bool = False          # reference baselines only (B3 oracle); recorded in the scoreboard
    uses_nonlibrary_action: bool = False   # True: Result.action may be built from something other than library
    #                                        rows (reference rows only, e.g. the previous chunk's unexecuted tail);
    #                                        recorded in the npz / json / scoreboard
    prof = NULL_PROFILER

    def fit(self, lib: LibraryView, ctx: "Context") -> None:
        pass

    def reset(self, episode: "EpisodeView") -> None:
        pass

    def query(self, q: "QueryView") -> Result:
        raise NotImplementedError

    def bytes_per_entry(self) -> float:
        return float("nan")


# ------------------------------------------------------------------------------------------- episode
class EpisodeView:
    """What a method may know at the start of an episode (no num_steps, no success, no file)."""

    __slots__ = ("uid", "task", "task_id", "init", "index", "seed", "rng")

    def __init__(self, uid, task, task_id, init, index, seed):
        self.uid = str(uid)
        self.task = str(task)
        self.task_id = int(task_id)
        self.init = int(init)
        self.index = int(index)       # position of the episode in the cell's episodes.json
        self.seed = int(seed)         # hash(run seed, uid): use it (or self.rng) for any randomness
        self.rng = np.random.default_rng(self.seed)

    def __getattr__(self, name):
        if name in FORBIDDEN:
            raise ForbiddenAccess(f"EpisodeView.{name} is not available online (forbidden: num_steps, success, "
                                  f"file, ...). Methods may only use uid, task, task_id, init, index, seed, rng.")
        raise AttributeError(f"EpisodeView has no attribute {name!r} (available: {', '.join(self.__slots__)})")


# --------------------------------------------------------------------------------------------- query
class QueryArrays:
    """The ONLY per-row arrays a QueryView can reach (no GT, no outcome, no recorded search)."""

    __slots__ = ("key_v0", "key_v1", "rs", "raw_state", "a_exec", "exec_hit", "tok_index", "tok_v0", "tok_v1",
                 "tok_img0", "tok_img1", "model", "H")

    def __init__(self, qc):  # qc: store.QueryCell
        self.key_v0 = qc.key_v0
        self.key_v1 = qc.key_v1
        self.rs = qc.rs
        self.raw_state = qc.raw_state
        self.a_exec = qc.a_exec
        self.exec_hit = qc.exec_hit_flag      # online-legal: whether each executed chunk came from the library
        self.tok_index = qc.tok_index
        self.tok_v0 = qc.tok("v0")
        self.tok_v1 = qc.tok("v1")
        self.tok_img0 = qc.tok("img0")
        self.tok_img1 = qc.tok("img1")
        self.model = qc.model
        self.H = qc.H


class QueryView:
    """One decision. Current-row fields, the episode's history (rows before this one) and, for rows of the tok
    subsample, the full token tensors / raw images. All arrays are read-only memmap slices -- copy before
    modifying."""

    __slots__ = ("_A", "_row", "_start", "_gt", "step", "task_id", "episode")

    def __init__(self, A: QueryArrays, row: int, start: int, step: int, task_id: int, episode: EpisodeView, gt=None):
        self._A = A
        self._row = row
        self._start = start
        self._gt = gt
        self.step = step              # index of this decision inside the episode (0-based)
        self.task_id = task_id
        self.episode = episode        # the EpisodeView passed to reset()

    def __getattr__(self, name):
        if name in FORBIDDEN:
            raise ForbiddenAccess(f"QueryView.{name} is not available online. Methods may use key_v0, key_v1, rs, "
                                  f"raw_state, task_id, step, episode, hist_* (earlier rows of this episode), "
                                  f"prev_a_exec, prev_hit, and tok_*/img* for tok-subsample rows.")
        raise AttributeError(f"QueryView has no attribute {name!r}")

    # -- current row
    @property
    def key_v0(self) -> np.ndarray:
        return self._A.key_v0[self._row]

    @property
    def key_v1(self) -> np.ndarray:
        return self._A.key_v1[self._row]

    @property
    def rs(self) -> np.ndarray:
        """Raw robot_state key as the online search saw it (pi05: 32-d, dims 8.. are 0). See dims.valid_state."""
        return self._A.rs[self._row]

    @property
    def raw_state(self) -> np.ndarray:
        return self._A.raw_state[self._row]

    @property
    def model(self) -> str:
        return self._A.model

    # -- history: rows [episode start, this row) -- length == self.step
    @property
    def hist_key_v0(self) -> np.ndarray:
        return self._A.key_v0[self._start:self._row]

    @property
    def hist_key_v1(self) -> np.ndarray:
        return self._A.key_v1[self._start:self._row]

    @property
    def hist_rs(self) -> np.ndarray:
        return self._A.rs[self._start:self._row]

    @property
    def hist_raw_state(self) -> np.ndarray:
        return self._A.raw_state[self._start:self._row]

    @property
    def hist_a_exec(self) -> np.ndarray:
        """Executed action chunks (H, 32) of the earlier decisions of this episode, oldest first. Only
        [:, :5, :7] was actually executed (replan 5, valid dims)."""
        return self._A.a_exec[self._start:self._row]

    @property
    def prev_a_exec(self):
        """Executed chunk of the previous decision, or None at step 0."""
        return self._A.a_exec[self._row - 1] if self._row > self._start else None

    @property
    def prev_hit(self):
        """Was the PREVIOUS decision of this episode served from the library (True, HIT) or by the policy (False,
        MISS)? None at step 0 (or if undecidable). Derived from the data: that row's executed chunk equals its
        a_hit / a_inf bitwise on [:, :7] (cache arms: always True, inf arms: always False)."""
        if self._row <= self._start:
            return None
        f = int(self._A.exec_hit[self._row - 1])
        return True if f == 1 else (False if f == 0 else None)

    @property
    def hist_hit(self) -> np.ndarray:
        """int8 [step]: for each earlier decision 1 = HIT (library), 0 = MISS (policy), -1 = undecidable."""
        f = self._A.exec_hit[self._start:self._row]
        return np.where(f == 1, 1, np.where(f == 0, 0, -1)).astype(np.int8)

    # -- tokens (tok subsample only)
    @property
    def has_tok(self) -> bool:
        return bool(self._A.tok_index[self._row] >= 0)

    def _tok(self, arr, what):
        j = int(self._A.tok_index[self._row])
        if j < 0 or arr is None:
            raise TokensUnavailable(f"{what} is only stored for the tok subsample (inits 0,10,20,30,40); this row "
                                    f"is not in it. Run the method with --subsample tok, or check q.has_tok first.")
        return arr[j]

    @property
    def tok_v0(self) -> np.ndarray:
        """Full vision_0 tokens, float16 [256, 2048] (tok subsample only)."""
        return self._tok(self._A.tok_v0, "tok_v0")

    @property
    def tok_v1(self) -> np.ndarray:
        return self._tok(self._A.tok_v1, "tok_v1")

    @property
    def img0(self) -> np.ndarray:
        """Raw agent-view image uint8 [h, w, 3] (tok subsample only)."""
        return self._tok(self._A.tok_img0, "img0")

    @property
    def img1(self) -> np.ndarray:
        """Raw wrist image uint8 [h, w, 3] (tok subsample only)."""
        return self._tok(self._A.tok_img1, "img1")

    # -- reference baselines only
    def _reference_gt(self) -> np.ndarray:
        """a_inf of this row. Only served to methods with ``uses_gt = True`` (B3 oracle, reference only)."""
        if self._gt is None:
            raise ForbiddenAccess("ground truth is only served to reference methods declaring uses_gt = True")
        return self._gt[self._row]


# ------------------------------------------------------------------------------------------- context
class Context:
    """Per-cell context handed to fit()."""

    def __init__(self, *, root, cell, seed, scratch, prof=NULL_PROFILER):
        from .store import lib_key, parse_cell

        self.root = pathlib.Path(root)
        self.cell = cell
        self.model, self.suite, self.arm = parse_cell(cell)
        self.lib_key = lib_key(cell)
        self.seed = int(seed)
        self.scratch = pathlib.Path(scratch)
        self.prof = prof
        self.horizon = dims.HORIZON[self.model]
        self.rs_valid = dims.RS_VALID[self.model]
        self.registered: dict[str, dict] = {}

    @property
    def action_sigma(self) -> np.ndarray:
        """sigma_d (7,) used by the err metric (std of current-library action[:, :5, d])."""
        return action_sigma(str(self.root), self.lib_key)

    def open_library(self, name: str = "current") -> LibraryView:
        """Open another stored library of this model x suite (e.g. "bpool_all", "bpool_cs")."""
        return LibraryView(self.root, self.lib_key, name)

    @property
    def library_names(self) -> list:
        """Libraries stored for this model x suite. Results may name any of them without registering."""
        from .store import library_names

        return library_names(self.root, self.lib_key)

    def current_params(self) -> dict:
        """weights / mu / sigma of the online weighted-score-sum search (fields vision_0, vision_1, robot_state)."""
        return current_params(self.model, self.suite)

    def lda_weights(self):
        return lda_weights(self.model, self.suite)

    def register_library(self, name: str, action, *, progress=None, task_id=None, meta=None) -> None:
        """Declare a method-built library so Results may refer to it (Result.library = name). ``action`` must be
        float [L', H, 32] built from stored library actions (only [:, :5, :7] is scored); ``progress`` (float [L'])
        enables phase_err; ``task_id`` (int [L']) is recorded for diagnostics. Call it inside fit()."""
        if name in ("current", "bpool_all") or name in self.library_names:
            raise ContractError(f"library name {name!r} is a stored library of {self.lib_key}; refer to it directly "
                                f"(Result.library={name!r}) instead of registering it")
        a = np.asarray(action)
        if a.ndim != 3 or a.shape[1] != self.horizon or a.shape[2] != dims.ACT_FULL_DIMS:
            raise ContractError(f"register_library({name!r}): action shape {a.shape}, expected [L, {self.horizon}, 32]")
        if not np.all(np.isfinite(dims.valid_action(a))):
            raise ContractError(f"register_library({name!r}): non-finite values in action[:, :5, :7]")
        ent = {"action": a}
        if progress is not None:
            p = np.asarray(progress, np.float64)
            if p.shape != (a.shape[0],):
                raise ContractError(f"register_library({name!r}): progress shape {p.shape} != ({a.shape[0]},)")
            ent["progress"] = p
        if task_id is not None:
            ent["task_id"] = np.asarray(task_id, np.int64)
        ent["meta"] = meta or {}
        self.registered[name] = ent


# ----------------------------------------------------------------------------------------- validation
def _is_int_array(a) -> bool:
    return isinstance(a, np.ndarray) and a.dtype.kind in "iu"


def validate_result(res, *, lib_sizes: dict, H: int, where: str = ""):
    """Check one Result against the contract; returns a normalized tuple
    (topk int64[k], scores float64[k], confidence float, library str, action or None, extras dict or None).
    Raises ContractError with a message naming the offending field."""
    w = f" [{where}]" if where else ""
    if not isinstance(res, Result):
        raise ContractError(f"query() must return harness.api.Result, got {type(res).__name__}{w}")
    lib = res.library
    if not isinstance(lib, str) or lib not in lib_sizes:
        raise ContractError(f"Result.library={lib!r} is not a known library {sorted(lib_sizes)} (stored libraries of "
                            f"this model x suite + registered ones); register method-built libraries with "
                            f"ctx.register_library() in fit(){w}")
    L = lib_sizes[lib]
    topk = res.topk
    if not isinstance(topk, np.ndarray):
        topk = np.asarray(topk)
    if topk.ndim != 1 or topk.shape[0] < 1:
        raise ContractError(f"Result.topk must be a non-empty 1-D array, got shape {topk.shape}{w}")
    if not _is_int_array(topk):
        raise ContractError(f"Result.topk must have an integer dtype, got {topk.dtype}{w}")
    tmin, tmax = int(topk.min()), int(topk.max())
    if tmin < 0 or tmax >= L:
        raise ContractError(f"Result.topk rows out of range for library {lib!r} (L={L}): min={tmin} max={tmax}{w}")
    scores = res.scores
    if not isinstance(scores, np.ndarray):
        scores = np.asarray(scores)
    if scores.shape != topk.shape:
        raise ContractError(f"Result.scores shape {scores.shape} != topk shape {topk.shape}{w}")
    if scores.dtype.kind not in "fiu":
        raise ContractError(f"Result.scores must be numeric, got {scores.dtype}{w}")
    if np.isnan(scores.astype(np.float64)).any():
        raise ContractError(f"Result.scores contains NaN{w}")
    conf = res.confidence
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        raise ContractError(f"Result.confidence must be a float, got {type(res.confidence).__name__}{w}") from None
    if not math.isfinite(conf):
        raise ContractError(f"Result.confidence must be finite, got {conf}{w}")
    act = res.action
    if act is not None:
        act = np.asarray(act)
        if act.shape != (H, dims.ACT_FULL_DIMS):
            raise ContractError(f"Result.action must have shape ({H}, 32) (only [:5, :7] is scored), got {act.shape}{w}")
        if act.dtype.kind != "f" or not np.all(np.isfinite(dims.valid_action(act))):
            raise ContractError(f"Result.action must be float and finite on [:5, :7]{w}")
    ex = res.extras
    if ex is not None:
        if not isinstance(ex, dict):
            raise ContractError(f"Result.extras must be a dict or None, got {type(ex).__name__}{w}")
        tot = 0
        for k, v in ex.items():
            if not isinstance(k, str) or not k.replace("_", "a").isalnum():
                raise ContractError(f"Result.extras key {k!r} must be an identifier-like string{w}")
            va = np.asarray(v)
            if va.dtype.kind not in "biuf":
                raise ContractError(f"Result.extras[{k!r}] must be numeric (float / int / bool scalar or array), "
                                    f"got dtype {va.dtype}{w}")
            tot += 4 * int(va.size)
        if tot > EXTRAS_CAP_BYTES:
            raise ContractError(f"Result.extras is {tot} bytes per decision (as float32), cap is {EXTRAS_CAP_BYTES}; "
                                f"expose fewer / smaller diagnostics{w}")
    return topk.astype(np.int64, copy=False), scores.astype(np.float64, copy=False), conf, lib, act, ex


def check_method_attrs(m) -> None:
    name = getattr(m, "name", None)
    if not isinstance(name, str) or not name or not all(c.isalnum() or c in "_.-" for c in name):
        raise ContractError(f"method.name must be a non-empty string of [A-Za-z0-9_.-], got {name!r}")
    if getattr(m, "tier", None) not in TIERS:
        raise ContractError(f"method.tier must be one of {TIERS}, got {getattr(m, 'tier', None)!r}")
    for fn in ("fit", "reset", "query", "bytes_per_entry"):
        if not callable(getattr(m, fn, None)):
            raise ContractError(f"method must define {fn}()")
