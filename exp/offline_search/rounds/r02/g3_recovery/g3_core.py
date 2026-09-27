"""G3 (closed-loop mechanisms) shared core: base-selector hook, selection / synthesis, library tables, the
library-derived stuck thresholds, library LOEO pseudo-queries and the isotonic / 10-bin maps.

The wrappers (wrappers.py) never re-implement a selector: they ask the BASE for the scores of ALL admissible
candidates through the hook of README.md ("base contract") and redo top-k + kernel synthesis themselves, so that
exclusions (recovery), forced members (blend) and features (confidence) work over any base.

Valid dims only (harness/dims.py): action statistics use action[:5, :7] in sigma units, robot_state uses rs[:8].
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import math
import pathlib
import sys

import numpy as np

from exp.offline_search.harness import api, dims

REPO = pathlib.Path(__file__).resolve().parents[5]
NAN = float("nan")
NH = dims.EXEC_STEPS * dims.ACT_DIMS          # 35 = 5 executed steps x 7 valid dims


# --------------------------------------------------------------------------------------------- loading
def load_class(spec: str):
    """'<dotted.module>:<Class>' or '<path/to/file.py>:<Class>' (relative paths resolve against the repo root).
    Same module naming as harness.run / closed_loop.plugin ('osm_' + md5(abs path)[:12]) so a file loaded by either
    of them and by this function is ONE module object. harness.run is deliberately not imported (it hides the GPU
    at import, which would break the closed-loop server)."""
    if ":" not in spec:
        raise ValueError(f"base spec must be <module_or_file>:<Class>, got {spec!r}")
    mod, cls = spec.rsplit(":", 1)
    if mod.endswith(".py") or "/" in mod:
        p = pathlib.Path(mod)
        if not p.is_absolute():
            p = (REPO / p) if (REPO / p).exists() else p.resolve()
        p = p.resolve()
        if not p.exists():
            raise FileNotFoundError(f"base method file not found: {p}")
        name = "osm_" + hashlib.md5(str(p).encode()).hexdigest()[:12]
        module = sys.modules.get(name)
        if module is None:
            if str(p.parent) not in sys.path:
                sys.path.insert(0, str(p.parent))
            sp = importlib.util.spec_from_file_location(name, p)
            module = importlib.util.module_from_spec(sp)
            sys.modules[name] = module
            try:
                sp.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(name, None)
                raise
    else:
        module = importlib.import_module(mod)
    if not hasattr(module, cls):
        raise AttributeError(f"{mod} has no class {cls!r}")
    return getattr(module, cls)


# ------------------------------------------------------------------------------------------ base hook
def base_scores(base, q):
    """-> (rows int64[C], S float64[C], aux dict). Prefers base.os_score_all(q), else base.score_all(q) (first two
    tuple elements; a third element is used as aux only if it is a dict)."""
    fn = getattr(base, "os_score_all", None) or getattr(base, "score_all", None)
    if fn is None:
        raise api.ContractError(f"base {getattr(base, 'name', base)!r} exposes neither os_score_all(q) nor "
                                f"score_all(q) (see g3_recovery/README.md, base contract)")
    out = fn(q)
    rows, S = out[0], out[1]
    aux = out[2] if len(out) > 2 and isinstance(out[2], dict) else {}
    rows = np.asarray(rows, np.int64)
    S = np.asarray(S, np.float64)
    if rows.ndim != 1 or rows.shape != S.shape or rows.size == 0:
        raise api.ContractError(f"base score hook returned rows {rows.shape} / scores {S.shape}")
    return rows, S, aux


def vis_of(aux, valid=None):
    """Visual feature of the candidate set: max over (valid) candidates of aux['vis_v0'] + max of aux['vis_v1']
    (or max of aux['vis']). NaN if the base exposes no visual similarity."""
    if "vis_v0" in aux and "vis_v1" in aux:
        a, b = np.asarray(aux["vis_v0"], np.float64), np.asarray(aux["vis_v1"], np.float64)
        if valid is not None:
            a, b = a[valid], b[valid]
        return float(a.max() + b.max()) if a.size else NAN
    if "vis" in aux:
        a = np.asarray(aux["vis"], np.float64)
        if valid is not None:
            a = a[valid]
        return float(a.max()) if a.size else NAN
    return NAN


# ------------------------------------------------------------------------------------ selection / synth
def topk_pos(S, k, valid=None):
    """Positions of the k best scores, ordered by (-S, position) -- identical to np.argsort(-S, kind='stable')[:k]
    restricted to `valid` (bool mask or None), including exact tie handling at the k-th value."""
    idx = np.arange(S.size) if valid is None else np.flatnonzero(valid)
    if idx.size == 0:
        return idx
    s = S[idx]
    k = min(int(k), idx.size)
    if idx.size > 4 * k + 16:
        part = np.argpartition(-s, k - 1)[:k]
        thr = s[part].min()
        cand = np.flatnonzero(s >= thr)
    else:
        cand = np.arange(idx.size)
    o = cand[np.lexsort((cand, -s[cand]))][:k]
    return idx[o]


def kernel_weights(s_sel, T):
    """Unnormalized kernel weights of a selection sorted by score (desc): exp(-(S_0 - S_i)/T); T=inf -> uniform."""
    s_sel = np.asarray(s_sel, np.float64)
    if s_sel.size <= 1:
        return np.ones(s_sel.size)
    if math.isinf(T):
        return np.ones(s_sel.size)
    if T <= 0:
        w = np.zeros(s_sel.size)
        w[0] = 1.0
        return w
    return np.exp(-(s_sel[0] - s_sel) / T)


def synth(act, rows, w, uniform):
    """Weighted mean of full (H, 32) library chunks; None for a single row (served as the library row itself).
    uniform=True (and unforced weights) -> plain mean, same arithmetic as f3_core.synthesize('mean')."""
    rows = np.asarray(rows, np.int64)
    if rows.size == 1:
        return None
    c = np.asarray(act[rows], np.float64)
    if uniform:
        a = c.mean(axis=0)
    else:
        wn = np.asarray(w, np.float64)
        wn = wn / wn.sum()
        a = np.tensordot(wn, c, axes=1)
    return a.astype(np.float32)


_IU: dict = {}


def dispersion(X):
    """Mean pairwise RMS over the distinct pairs of k heads X (k, 35) or (k,5,7); 0 for k < 2."""
    k = X.shape[0]
    if k < 2:
        return 0.0
    F = X.reshape(k, -1)
    iu = _IU.get(k)
    if iu is None:
        iu = _IU[k] = np.triu_indices(k, 1)
    D = F[iu[0]] - F[iu[1]]
    return float(np.sqrt(np.einsum("ij,ij->i", D, D) / F.shape[1]).mean())


def hash_chunk(a7) -> int:
    """64-bit hash of an action chunk's valid block (H, 7) float32 bytes."""
    b = np.ascontiguousarray(a7, dtype=np.float32).tobytes()
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "little")


# ---------------------------------------------------------------------------------------- lib tables
class Tables:
    """Per-row arrays of one stored library that the wrappers need (all small except `act`, a memmap view)."""

    def __init__(self, L, model, sigma, want_hash=True):
        self.name = L.name
        self.L = int(L.L)
        self.act = np.asarray(L.action)                                      # (L,H,32) view of the memmap
        self.HD = np.ascontiguousarray(np.asarray(dims.valid_action(L.action), np.float32)
                                       / np.asarray(sigma, np.float32)).reshape(self.L, NH)
        self.ep = np.asarray(L.episode, np.int64)
        self.step = np.asarray(L.step, np.int64)
        self.prog = np.asarray(L.progress, np.float64)
        self.nxt = np.asarray(L.next, np.int64)
        self.prev = np.asarray(L.prev, np.int64)
        self.task = np.asarray(L.task_id, np.int64)
        self.ep_len = np.asarray(L.ep_len, np.int64)
        self.rs8 = np.ascontiguousarray(dims.valid_state(np.asarray(L.rs, np.float32), model))
        self.stems = [e.get("stem") for e in L.episodes] if hasattr(L, "episodes") else None
        if want_hash:
            A7 = dims.valid_action_chunk(self.act)
            hkey = np.fromiter((hash_chunk(A7[r]) for r in range(self.L)), dtype=np.uint64, count=self.L)
            self.horder = np.argsort(hkey, kind="stable")
            self.hsorted = hkey[self.horder]
        else:
            self.horder = self.hsorted = None

    def find_chunk(self, a7, task) -> int:
        """Row of `task` whose action[:, :7] equals a7 bit for bit (lowest row), -1 if none."""
        h = np.uint64(hash_chunk(a7))
        i = int(np.searchsorted(self.hsorted, h))
        best = -1
        while i < self.L and self.hsorted[i] == h:
            r = int(self.horder[i])
            if self.task[r] == task and np.array_equal(dims.valid_action_chunk(self.act[r]), a7):
                best = r if best < 0 else min(best, r)
            i += 1
        return best

    def rs8_of(self, rows):
        """rs[:8] of candidate rows: a zero-copy slice when rows is a contiguous ascending range (10x libraries are
        task-sorted), else a gather."""
        lo, hi = int(rows[0]), int(rows[-1]) + 1
        if hi - lo == rows.size and (rows.size < 2 or bool(np.all(np.diff(rows) == 1))):
            return self.rs8[lo:hi]
        return self.rs8[rows]


def episode_rows(T: Tables):
    """{episode: rows sorted by step}."""
    o = np.lexsort((T.step, T.ep))
    ep_s = T.ep[o]
    cuts = np.flatnonzero(np.diff(ep_s)) + 1
    return {int(ep_s[s]): g for s, g in zip(np.r_[0, cuts], np.split(o, cuts))}


def task_means(K, task, tasks, chunk=2048):
    """{task: mean key float32[D]} (float64 accumulation, rows of each task in chunks)."""
    out = {}
    for t in tasks:
        r = np.flatnonzero(task == t)
        acc = np.zeros(K.shape[1], np.float64)
        contig = r.size > 0 and np.all(np.diff(r) == 1)
        for lo in range(0, r.size, chunk):
            X = K[r[lo]:r[min(lo + chunk, r.size) - 1] + 1] if contig else K[r[lo:lo + chunk]]
            acc += np.asarray(X, np.float64).sum(0)
        out[int(t)] = (acc / max(r.size, 1)).astype(np.float32)
    return out


def consecutive_vcos(K0, K1, M0, M1, T: Tables, ep_rows):
    """For every row r with a predecessor p in its episode: min over the two cameras of the task-centred cosine
    between key(p) and key(r). NaN where r has no predecessor. (Library counterpart of the online visual
    self-change q.key vs q.hist_key[-1].)"""
    out = np.full(T.L, NAN)
    for e, rows in ep_rows.items():
        if rows.size < 2:
            continue
        t = int(T.task[rows[0]])
        lo, hi = int(rows.min()), int(rows.max()) + 1
        contig = hi - lo == rows.size and np.array_equal(rows, np.arange(lo, hi))
        cs = []
        for K, M in ((K0, M0), (K1, M1)):
            X = (np.asarray(K[lo:hi] if contig else K[rows], np.float32) - M[t]).astype(np.float64)
            n = np.sqrt(np.einsum("ij,ij->i", X, X))
            d = np.einsum("ij,ij->i", X[:-1], X[1:])
            cs.append(d / np.maximum(n[:-1] * n[1:], 1e-12))
        out[rows[1:]] = np.minimum(cs[0], cs[1])
    return out


def centred_cos(a, b, m):
    """cosine(a - m, b - m), float32 arithmetic (a, b, m float32 [D]); the online visual self-change."""
    x = np.subtract(np.asarray(a), m, dtype=np.float32)
    y = np.subtract(np.asarray(b), m, dtype=np.float32)
    return float(np.dot(x, y)) / max(math.sqrt(float(np.dot(x, x)) * float(np.dot(y, y))), 1e-12)


def run_lengths(flag, T: Tables, ep_rows):
    """Consecutive-count of `flag` along every episode (resets to 0 on a False)."""
    out = np.zeros(T.L, np.int64)
    for rows in ep_rows.values():
        c = 0
        for r in rows:
            c = c + 1 if flag[r] else 0
            out[r] = c
    return out


# ------------------------------------------------------------------------------------ monotone maps
def pav_decreasing(x, y):
    """Isotonic (non-increasing in x) least-squares fit by pool-adjacent-violators.
    -> knots (block mean x ascending, block mean y non-increasing), duplicates in x merged."""
    o = np.argsort(x, kind="stable")
    xs, ys = np.asarray(x, np.float64)[o], np.asarray(y, np.float64)[o]
    sy, sx, n = [], [], []
    for xi, yi in zip(xs, ys):
        sy.append(yi)
        sx.append(xi)
        n.append(1)
        while len(sy) > 1 and sy[-2] / n[-2] < sy[-1] / n[-1]:
            b_y, b_x, b_n = sy.pop(), sx.pop(), n.pop()
            sy[-1] += b_y
            sx[-1] += b_x
            n[-1] += b_n
    kx = np.asarray(sx) / np.asarray(n)
    ky = np.asarray(sy) / np.asarray(n)
    return _dedupe(kx, ky, np.asarray(n, np.float64))


def bin_map(x, y, nbins=10):
    """10-bin map: quantile bins of x, (mean x, mean y) per bin, made non-increasing by a weighted PAV."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
    b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, max(edges.size - 2, 0))
    kx, ky, kn = [], [], []
    for j in range(max(edges.size - 1, 1)):
        m = b == j
        if m.any():
            kx.append(x[m].mean())
            ky.append(y[m].mean())
            kn.append(m.sum())
    kx, ky, kn = np.asarray(kx), np.asarray(ky), np.asarray(kn, np.float64)
    # weighted PAV over the bin means
    sy, sx, n = [], [], []
    for xi, yi, ni in zip(kx, ky, kn):
        sy.append(yi * ni)
        sx.append(xi * ni)
        n.append(ni)
        while len(sy) > 1 and sy[-2] / n[-2] < sy[-1] / n[-1]:
            b_y, b_x, b_n = sy.pop(), sx.pop(), n.pop()
            sy[-1] += b_y
            sx[-1] += b_x
            n[-1] += b_n
    return _dedupe(np.asarray(sx) / np.asarray(n), np.asarray(sy) / np.asarray(n), np.asarray(n))


def _dedupe(kx, ky, kn):
    ux, inv = np.unique(kx, return_inverse=True)
    if ux.size == kx.size:
        return kx, ky
    wy = np.bincount(inv, weights=ky * kn) / np.bincount(inv, weights=kn)
    return ux, wy


# --------------------------------------------------------------------------------------- pseudo-query
class _PseudoEpisode:
    __slots__ = ("uid", "task", "task_id", "init", "index", "seed", "rng")

    def __init__(self, uid, task_id):
        self.uid, self.task, self.task_id, self.init, self.index, self.seed = str(uid), "", int(task_id), -1, -1, 0
        self.rng = np.random.default_rng(0)


class PseudoQuery:
    """A library row posed as a QueryView (LOEO calibration). Same attribute names as api.QueryView; hist_* are
    materialised lazily (fancy-indexed copies) only if the base touches them; raw_state is not stored in the library
    (NaN), tokens are unavailable."""

    def __init__(self, Lview, T: Tables, ep_rows_of_r, pos, model, prev_hit):
        self._L, self._T, self._rows, self._pos = Lview, T, ep_rows_of_r, int(pos)
        self._r = int(ep_rows_of_r[pos])
        self.step = int(T.step[self._r])
        self.task_id = int(T.task[self._r])
        self.model = model
        self.episode = _PseudoEpisode(f"lib{T.name}:{int(T.ep[self._r])}", self.task_id)
        self.prev_hit = None if self._pos == 0 else bool(prev_hit)
        self.has_tok = False

    key_v0 = property(lambda s: s._L.key_v0[s._r])
    key_v1 = property(lambda s: s._L.key_v1[s._r])
    rs = property(lambda s: s._L.rs[s._r])
    raw_state = property(lambda s: np.full(dims.RAW_STATE_DIM, np.nan, np.float32))
    hist_key_v0 = property(lambda s: np.asarray(s._L.key_v0[s._rows[:s._pos]]))
    hist_key_v1 = property(lambda s: np.asarray(s._L.key_v1[s._rows[:s._pos]]))
    hist_rs = property(lambda s: np.asarray(s._L.rs[s._rows[:s._pos]]))
    hist_raw_state = property(lambda s: np.full((s._pos, dims.RAW_STATE_DIM), np.nan, np.float32))
    hist_a_exec = property(lambda s: np.asarray(s._T.act[s._rows[:s._pos]]))
    prev_a_exec = property(lambda s: None if s._pos == 0 else s._T.act[s._rows[s._pos - 1]])
    hist_hit = property(lambda s: np.full(s._pos, 1 if s.prev_hit else 0, np.int8))

    def _unavail(self, *_):
        raise api.TokensUnavailable("library pseudo-queries carry no tokens / images")

    tok_v0 = property(_unavail)
    tok_v1 = property(_unavail)
    img0 = property(_unavail)
    img1 = property(_unavail)
