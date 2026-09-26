"""Read-only access to the offline_search store (protocol §5.1). All .npy files are opened mmap_mode="r".

    <root>/queries/<m>_<s>_<a>/     episodes.json, ep.npy, step.npy, key_v0/key_v1.npy, rs.npy, raw_state.npy,
                                    a_inf/a_hit/a_exec.npy, rec_top1/rec_score/rec_perfield.npy,
                                    tok/{rows,v0,v1,img0,img1}.npy, manifest.json (written last)
    <root>/library/<m>_<s>/<name>/  key_v0, key_v1, rs, action, task_id, episode, step, ep_len, progress,
                                    success, prev, next (.npy), ids.json, episodes.json, meta.json, tok/
    <root>/floor/<m>_<s>/           step0_pairs.npz, resample.npz  (optional)

Array files are looked up as ``<name>.npy`` first and ``rows.<name>.npy`` second (the protocol text writes
``rows.*.npy``; the extractors write the bare names).

This module holds the *full* query-side object (``QueryCell``) that the runner and the metrics use. Methods never
get it: they receive ``api.QueryView`` objects whose backing store only contains the allowed arrays.
"""
from __future__ import annotations

import functools
import json
import pathlib

import numpy as np

MODELS = ("pi05", "groot")
SUITES = ("spatial", "l10")
ARMS = ("inf", "cache")
CELLS = tuple(f"{m}_{s}_{a}" for m in MODELS for s in SUITES for a in ARMS)
LIBS = tuple(f"{m}_{s}" for m in MODELS for s in SUITES)

FULL_ROOT = pathlib.Path("/home/weiland/trace_runs/offline_search_store")
SMOKE_ROOT = pathlib.Path("/home/weiland/trace_runs/offline_search_store_smoke")

REPO = pathlib.Path(__file__).resolve().parents[3]
TRACE_DUAL_CFG = REPO / "exp" / "trace_dual" / "config"
LDA_MANIFEST = REPO / "exp" / "weighted_sum" / "config" / "fusion_ablation" / "manifest.json"
SUITE_SHORT = {"spatial": "sp", "l10": "l10"}
SUITE_FULL = {"spatial": "libero_spatial", "l10": "libero_10"}
FIELDS = ("vision_0", "vision_1", "robot_state")


class StoreError(FileNotFoundError):
    pass


def parse_cell(cell: str) -> tuple[str, str, str]:
    parts = cell.split("_")
    if len(parts) != 3 or parts[0] not in MODELS or parts[1] not in SUITES or parts[2] not in ARMS:
        raise ValueError(f"bad cell name {cell!r}; expected <model>_<suite>_<arm> with model in {MODELS}, "
                         f"suite in {SUITES}, arm in {ARMS} (e.g. pi05_spatial_inf)")
    return parts[0], parts[1], parts[2]


def lib_key(cell: str) -> str:
    m, s, _ = parse_cell(cell)
    return f"{m}_{s}"


def _npy_path(d: pathlib.Path, name: str) -> pathlib.Path:
    for cand in (d / f"{name}.npy", d / f"rows.{name}.npy"):
        if cand.exists():
            return cand
    raise StoreError(f"missing array {name!r} in {d} (looked for {name}.npy and rows.{name}.npy)")


def _load(d: pathlib.Path, name: str):
    return np.load(_npy_path(d, name), mmap_mode="r")


def _has(d: pathlib.Path, name: str) -> bool:
    return (d / f"{name}.npy").exists() or (d / f"rows.{name}.npy").exists()


# --------------------------------------------------------------------------------------------- status
def ready_markers(root) -> dict:
    root = pathlib.Path(root)
    return {"queries": (root / "queries" / "READY").exists(), "library": (root / "library" / "READY").exists()}


def cell_available(root, cell: str) -> bool:
    d = pathlib.Path(root) / "queries" / cell
    if not (d / "episodes.json").exists() or not _has(d, "ep"):
        return False
    return (d / "manifest.json").exists() or (pathlib.Path(root) / "queries" / "READY").exists()


def library_available(root, key: str, name: str = "current") -> bool:
    d = pathlib.Path(root) / "library" / key / name
    return _has(d, "key_v0") and _has(d, "action") and _has(d, "task_id")


def library_names(root, key: str) -> list[str]:
    """Every complete library stored for one model x suite ("current" first), e.g. current, bpool_all, bpool_cs.
    Directories whose name starts with "_" or contains "." (e.g. current.partial) are ignored."""
    d = pathlib.Path(root) / "library" / key
    if not d.is_dir():
        return []
    names = [p.name for p in d.iterdir() if p.is_dir() and not p.name.startswith("_") and "." not in p.name
             and library_available(root, key, p.name)]
    return sorted(names, key=lambda n: (n != "current", n))


def available_cells(root) -> list[str]:
    return [c for c in CELLS if cell_available(root, c) and library_available(root, lib_key(c))]


def resolve_cells(spec: str, root) -> list[str]:
    """'all' -> every cell present (queries + current library) in the store; else a comma list of names."""
    if spec in ("all", "", None):
        cells = available_cells(root)
        if not cells:
            raise StoreError(f"no complete cell under {root} (need queries/<cell>/episodes.json + manifest.json "
                             f"or queries/READY, and library/<m>_<s>/current)")
        return cells
    out = []
    for c in spec.split(","):
        c = c.strip()
        if not c:
            continue
        parse_cell(c)
        if not cell_available(root, c):
            raise StoreError(f"cell {c!r} is not available under {root}/queries (episodes.json/manifest.json missing)")
        if not library_available(root, lib_key(c)):
            raise StoreError(f"library {lib_key(c)}/current is not available under {root}/library")
        out.append(c)
    return out


# ---------------------------------------------------------------------------------------- queries
class QueryCell:
    """Full (metric-side) view of one query cell. Arrays load lazily as read-only memmaps."""

    ARRAYS = ("ep", "step", "key_v0", "key_v1", "rs", "raw_state", "a_inf", "a_hit", "a_exec",
              "rec_top1", "rec_score", "rec_perfield")

    def __init__(self, root, cell: str):
        self.root = pathlib.Path(root)
        self.cell = cell
        self.model, self.suite, self.arm = parse_cell(cell)
        self.lib_key = f"{self.model}_{self.suite}"
        self.dir = self.root / "queries" / cell
        if not (self.dir / "episodes.json").exists():
            raise StoreError(f"{self.dir}/episodes.json not found")
        self.episodes = json.loads((self.dir / "episodes.json").read_text())
        self._a = {}

    def __getattr__(self, name):
        if name.startswith("_") or name not in QueryCell.ARRAYS:
            raise AttributeError(name)
        return self._arr(name)

    def _arr(self, name):
        a = self._a.get(name)
        if a is None:
            a = self._a[name] = _load(self.dir, name)
        return a

    @property
    def N(self) -> int:
        return int(self._arr("ep").shape[0])

    @property
    def H(self) -> int:
        return int(self._arr("a_inf").shape[1])

    @functools.cached_property
    def num_steps_row(self) -> np.ndarray:
        """num_steps of each row's episode (metric side only)."""
        ns = np.array([e["num_steps"] for e in self.episodes], np.int64)
        return ns[np.asarray(self.ep, np.int64)]

    # -- what was executed at each row (derived from the data, not from the arm name)
    @functools.cached_property
    def exec_hit_flag(self) -> np.ndarray:
        """int8[N]: 1 = the executed chunk equals a_hit (served from the library, HIT), 0 = equals a_inf (policy,
        MISS), 2 = equals both, -1 = neither. Bitwise comparison on the valid dims [:, :, :7]."""
        from .dims import ACT_DIMS

        n = self.N
        out = np.empty(n, np.int8)
        for lo in range(0, n, 8192):
            hi = min(n, lo + 8192)
            ex = np.asarray(self._arr("a_exec")[lo:hi, :, :ACT_DIMS])
            h = (ex == np.asarray(self._arr("a_hit")[lo:hi, :, :ACT_DIMS])).all(axis=(1, 2))
            m = (ex == np.asarray(self._arr("a_inf")[lo:hi, :, :ACT_DIMS])).all(axis=(1, 2))
            out[lo:hi] = np.where(h & m, 2, np.where(h, 1, np.where(m, 0, -1)))
        return out

    def exec_hit_counts(self) -> dict:
        """Counts of exec_hit_flag over all rows and over the rows that are the PREVIOUS decision of some decision
        (i.e. what q.prev_hit reports at step >= 1)."""
        f = self.exec_hit_flag
        prev = np.ones(self.N, bool)
        prev[[e["end"] - 1 for e in self.episodes]] = False
        names = {1: "hit", 0: "miss", 2: "both", -1: "neither"}
        return {"all_rows": {names[k]: int((f == k).sum()) for k in names},
                "prev_of_step_ge1": {names[k]: int((f[prev] == k).sum()) for k in names}}

    # -- token subsample
    @functools.cached_property
    def tok_rows(self) -> np.ndarray:
        d = self.dir / "tok"
        if not _has(d, "rows"):
            return np.zeros(0, np.int64)
        return np.asarray(_load(d, "rows"), np.int64)

    @functools.cached_property
    def tok_index(self) -> np.ndarray:
        """int32[N]: position of a row in tok/*.npy, -1 when the row is not in the subsample."""
        idx = np.full(self.N, -1, np.int32)
        r = self.tok_rows
        idx[r] = np.arange(r.shape[0], dtype=np.int32)
        return idx

    def tok(self, name: str):
        d = self.dir / "tok"
        if not _has(d, name):
            return None
        return _load(d, name)


# ---------------------------------------------------------------------------------------- library
class LibraryView:
    """One library (library/<m>_<s>/<name>/). Handed to Method.fit; every array is a read-only memmap.

    Attribute access ``lib.<name>`` loads ``<name>.npy`` lazily (key_v0, key_v1, rs, action, task_id, episode, step,
    ep_len, progress, success, prev, next, traj, ... whatever the builder wrote). Token tensors: ``lib.tok(name)``
    with name in {v0, v1, img0, img1} (None if absent).
    """

    def __init__(self, root, key: str, name: str = "current"):
        self.root = pathlib.Path(root)
        self.key = key
        self.name = name
        self.model, self.suite = key.split("_")
        self.dir = self.root / "library" / key / name
        if not self.dir.is_dir():
            raise StoreError(f"library {key}/{name} not found: {self.dir}")
        self._a = {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return self._arr(name)

    def _arr(self, name):
        """Load <name>.npy lazily; a missing file raises StoreError (not AttributeError, so that it is never
        mistaken for a missing attribute)."""
        a = self.__dict__["_a"].get(name)
        if a is None:
            d = self.__dict__["dir"]
            if not _has(d, name):
                raise StoreError(f"library {self.key}/{self.name} has no array {name!r} in {d} "
                                 f"(files: {sorted(p.name for p in d.glob('*.npy'))})")
            a = self._a[name] = _load(d, name)
        return a

    def has(self, name: str) -> bool:
        return _has(self.dir, name)

    def tok(self, name: str):
        """Raw token / image tensor of the library tok subset (row j of it is library row tok_rows[j]), or None."""
        d = self.dir / "tok"
        return _load(d, name) if _has(d, name) else None

    @functools.cached_property
    def tok_rows(self) -> np.ndarray:
        """Library rows that have token tensors (all rows when tok/rows.npy is absent but tok/v0.npy covers L)."""
        d = self.dir / "tok"
        if _has(d, "rows"):
            return np.asarray(_load(d, "rows"), np.int64)
        v = self.tok("v0")
        return np.arange(v.shape[0], dtype=np.int64) if v is not None and v.shape[0] == self.L else np.zeros(0, np.int64)

    @functools.cached_property
    def tok_index(self) -> np.ndarray:
        """int32[L]: position of a library row in tok/*.npy, -1 when it has no tokens."""
        idx = np.full(self.L, -1, np.int32)
        r = self.tok_rows
        idx[r] = np.arange(r.shape[0], dtype=np.int32)
        return idx

    def tok_of(self, name: str, row: int):
        """Token / image tensor of one library row (raises LookupError when the row has none)."""
        j = int(self.tok_index[row])
        arr = self.tok(name)
        if j < 0 or arr is None:
            raise LookupError(f"library {self.key}/{self.name} row {row} has no tok/{name}")
        return arr[j]

    @property
    def L(self) -> int:
        return int(self._arr("task_id").shape[0])

    @property
    def H(self) -> int:
        return int(self._arr("action").shape[1])

    @functools.cached_property
    def ids(self) -> list:
        p = self.dir / "ids.json"
        return json.loads(p.read_text()) if p.exists() else []

    @functools.cached_property
    def episodes(self) -> list:
        p = self.dir / "episodes.json"
        return json.loads(p.read_text()) if p.exists() else []

    @functools.cached_property
    def meta(self) -> dict:
        """manifest.json / meta.json of the library (task strings <-> task_id, source paths, counts)."""
        for p in (self.dir / "manifest.json", self.dir / "meta.json", self.dir.parent / "meta.json"):
            if p.exists():
                return json.loads(p.read_text())
        return {}

    @functools.cached_property
    def _task_rows(self) -> dict:
        t = np.asarray(self.task_id, np.int64)
        order = np.argsort(t, kind="stable")
        ts = t[order]
        cuts = np.flatnonzero(np.diff(ts)) + 1
        return {int(g[0]): order[s:e] for g, s, e in zip(np.split(ts, cuts), np.r_[0, cuts], np.r_[cuts, len(ts)])
                if len(g)}

    def tasks(self) -> list[int]:
        return sorted(self._task_rows)

    def rows_of_task(self, task_id: int) -> np.ndarray:
        """Ascending library rows of one task (library order, the order the online search saw them)."""
        r = self._task_rows.get(int(task_id))
        return r if r is not None else np.zeros(0, np.int64)


# ------------------------------------------------------------------------------- reference params
@functools.lru_cache(maxsize=None)
def current_params(model: str, suite: str) -> dict:
    """weights / mu / sigma of the online cp1 search (trace_dual tr_<m>_<sp|l10>_inf.yaml; inf == cache yaml
    except the judge threshold). Field order: vision_0, vision_1, robot_state. robot_state is scored as -L2."""
    import yaml

    p = TRACE_DUAL_CFG / f"tr_{model}_{SUITE_SHORT[suite]}_inf.yaml"
    cfg = yaml.safe_load(p.read_text())
    cp1 = cfg["checkpoints"]["cp1"]["search_strategy"]
    if cp1["type"] != "weighted_score_sum_knn":
        raise ValueError(f"{p}: unexpected search_strategy {cp1['type']}")
    norm = cp1["score_normalization"]["fields"]
    for f in FIELDS:
        if norm[f]["method"] != "zscore" or norm[f]["params"].get("squash", "tanh") != "tanh":
            raise ValueError(f"{p}: field {f} normalizer is not zscore+tanh")
    sims = cp1["field_similarity"]
    if sims["vision_0"]["type"] != "cosine" or sims["vision_1"]["type"] != "cosine" or sims["robot_state"]["type"] != "l2":
        raise ValueError(f"{p}: unexpected field similarity {sims}")
    return {
        "source": str(p),
        "weights": tuple(float(cfg["keys"][f]["weight"]) for f in FIELDS),
        "mu": tuple(float(norm[f]["params"]["mu"]) for f in FIELDS),
        "sigma": tuple(float(norm[f]["params"]["sigma"]) for f in FIELDS),
        "fields": FIELDS,
    }


@functools.lru_cache(maxsize=None)
def lda_weights(model: str, suite: str):
    """LDA fusion weights (vision_0, vision_1, robot_state) from the fusion_ablation manifest, or None."""
    if not LDA_MANIFEST.exists():
        return None
    man = json.loads(LDA_MANIFEST.read_text())
    ent = man.get(f"fw_{model}_{SUITE_FULL[suite]}_lda")
    if ent is None:
        return None
    return tuple(float(x) for x in ent["w"])


@functools.lru_cache(maxsize=None)
def action_sigma(root: str, key: str) -> np.ndarray:
    """sigma_d (d < 7): std (ddof=0, float64) of library/<key>/current action[:, :5, d]."""
    from .dims import ACT_DIMS, valid_action

    a = LibraryView(root, key, "current").action
    x = np.asarray(valid_action(a), np.float64).reshape(-1, ACT_DIMS)
    s = x.std(0)
    if not np.all(s > 0):
        raise ValueError(f"degenerate action sigma for {key}: {s}")
    return s
