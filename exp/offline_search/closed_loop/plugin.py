"""Serve an offline-harness Method as the CP1 retrieval of a PURE-CACHE closed-loop server (no src/ edits).

Every decision is a FULL_HIT served from a library: the policy's stage 1 (vision encoder) still runs to produce the
pooled visual keys, stage 2/3 never run. The retrieval / selection is any class written against the offline harness
Method API (``exp/offline_search/harness/api.py``: fit(lib, ctx) / reset(episode) / query(q) -> Result).

How it hooks in (all from the exp side, installed by ``serve_pi05.py`` / ``serve_groot.py`` before the stock server
entry point runs):

* ``openpi.cache.config.build_per_connection_components`` is wrapped. Every connection's freshly built CP1 search
  strategy, judge and storage facade are replaced by ``PluginStrategy`` (builds the online QueryView, calls
  ``method.query``), ``PluginJudge`` (always FULL_HIT on the plugin's pick) and ``PluginStorage`` (serves synthesized /
  non-current-library actions as payloads; current-library rows go through the native backend payload, i.e. the
  bit-identical pkl action). The native strategy is kept for lifecycle forwarding and, by default, runs in shadow on the
  same live keys so every decision also logs the native B0 winner (online agreement check).
* ``openpi.serving.websocket_policy_server.WebsocketPolicyServer`` is subclassed so each per-connection policy is
  wrapped by ``_ConnPolicy``: it hands the wire observation (raw robot state, raw images) and the episode identity
  (task, ``__extra__`` task_id / orig_init_state_idx / task_uid / attempt) to the connection's session, times every
  ``infer`` and flushes the per-episode logs on ``episode_end``.

Online QueryView == offline QueryView (harness README): key_v0 / key_v1 are the live pooled keys (the same
``key_builder.build`` output the trace recorded), rs the robot_state key, raw_state the wire ``observation/state`` as
float32, step = decision index in the episode, hist_* the earlier decisions of this episode, prev_a_exec / hist_a_exec
the executed (served) chunks, prev_hit True after step 0 (pure cache), hist_hit all 1. Tokens are available online on
every decision (tok_v0 / tok_v1 from the stage-1 prefix, img0 / img1 from the wire images), unlike offline where only the
tok subsample has them.

``--os-method native`` keeps the native retrieval untouched (control arm) and only logs: timing, native winner row,
and (with --os-log-inputs) the same per-episode input arrays.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.util
import itertools
import json
import logging
import math
import os
import pathlib
import pickle
import re
import socket
import sys
import threading
import time
import traceback
import types
import weakref

import numpy as np

log = logging.getLogger("osplug")

REPO = pathlib.Path(__file__).resolve().parents[3]
SCHEMA = "offline_search.closed_loop.v1"
KEY_BUILDER = {"pi05": "cp1_spatial_pool_16", "groot": "cp1_groot_libero_spatial_pool_16"}
SYNTH_PREFIX = "osplug:"
WIRE_STATE = "observation/state"
WIRE_IMG0 = "observation/image"
WIRE_IMG1 = "observation/wrist_image"
EXTRA_SCALARS_MAX = 24

_TLS = threading.local()
RUNTIME: "PluginRuntime | None" = None


# ------------------------------------------------------------------------------------------------ CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="closed_loop plugin", add_help=False, allow_abbrev=False)
    ap.add_argument("--os-method", required=True,
                    help="<module_or_file>:<Class> of a harness Method, or 'native' (control: native retrieval, log only)")
    ap.add_argument("--os-kwargs", default="{}", help="JSON kwargs for the Method constructor")
    ap.add_argument("--os-cell", required=True, help="store cell whose library the method is fitted on, e.g. "
                    "pi05_spatial_cache (arm must be 'cache': pure-cache closed loop)")
    ap.add_argument("--os-root", default="/dev/shm/offline_search_store", help="offline_search store root")
    ap.add_argument("--os-seed", type=int, default=0, help="run seed of EpisodeView.seed = hash(seed, task_uid)")
    ap.add_argument("--os-log-dir", required=True, help="per-decision JSONL + per-episode input npz go here")
    ap.add_argument("--os-tag", default="", help="log file tag (default p<pid>); use <arm>_<port>")
    ap.add_argument("--os-log-inputs", action="store_true",
                    help="write per-episode npz with every decision's inputs (keys 256 KB/decision: smoke only)")
    ap.add_argument("--os-no-shadow-native", action="store_true",
                    help="do not run the native search in shadow (saves its ~ms per decision)")
    ap.add_argument("--os-fit-artifact", default="",
                    help="pickle of a fitted method: loaded if it exists, else written after fitting")
    ap.add_argument("--os-allow-other-pkl", action="store_true",
                    help="accept a served library pkl other than the store manifest source (payload check still runs)")
    ap.add_argument("--os-tokens", choices=("on", "off"), default="on",
                    help="expose tok_v0/tok_v1/img0/img1 online on every decision (default on)")
    return ap


def parse_cli(argv):
    """Split argv into (plugin options, the rest for the stock server CLI)."""
    opts, rest = build_parser().parse_known_args(list(argv))
    try:
        opts.kwargs = json.loads(opts.os_kwargs) if opts.os_kwargs else {}
    except json.JSONDecodeError as e:
        raise SystemExit(f"--os-kwargs is not valid JSON: {e}") from None
    if not isinstance(opts.kwargs, dict):
        raise SystemExit("--os-kwargs must be a JSON object")
    return opts, rest


# --------------------------------------------------------------------------------------- harness glue
def load_method_class(spec: str):
    """Same resolution as harness.run.load_method_class (run.py is not imported: it hides the GPU at import)."""
    if ":" not in spec:
        raise ValueError(f"--os-method must be <module_or_file>:<ClassName>, got {spec!r}")
    mod, cls = spec.rsplit(":", 1)
    if mod.endswith(".py") or "/" in mod:
        p = pathlib.Path(mod)
        if not p.is_absolute():
            p = (REPO / p) if (REPO / p).exists() else p.resolve()
        p = p.resolve()
        if not p.exists():
            raise FileNotFoundError(f"method file not found: {p}")
        name = "osm_" + hashlib.md5(str(p).encode()).hexdigest()[:12]
        if name in sys.modules:
            module = sys.modules[name]
        else:
            if str(p.parent) not in sys.path:
                sys.path.insert(0, str(p.parent))
            sp = importlib.util.spec_from_file_location(name, p)
            module = importlib.util.module_from_spec(sp)
            sys.modules[name] = module
            sp.loader.exec_module(module)
        src = str(p)
    else:
        module = importlib.import_module(mod)
        src = getattr(module, "__file__", mod)
    if not hasattr(module, cls):
        raise AttributeError(f"{mod} has no class {cls!r}")
    return getattr(module, cls), src


def ep_seed(seed: int, uid: str) -> int:
    """Same as harness.run.ep_seed."""
    h = hashlib.blake2b(f"{int(seed)}:{uid}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") >> 1


def _share_memo(obj) -> dict:
    """deepcopy memo that maps every ndarray / torch tensor reachable from ``obj`` to itself (shared, not copied)."""
    memo: dict = {}
    seen: set = set()
    torch = sys.modules.get("torch")
    stack = [(obj, 0)]
    while stack:
        o, d = stack.pop()
        if id(o) in seen or d > 12:
            continue
        seen.add(id(o))
        if isinstance(o, np.ndarray) or (torch is not None and isinstance(o, torch.Tensor)):
            memo[id(o)] = o
            continue
        if isinstance(o, (str, bytes, int, float, bool, complex, type(None), type, types.ModuleType,
                          types.FunctionType, types.BuiltinFunctionType, types.MethodType)):
            continue
        if isinstance(o, dict):
            stack.extend((v, d + 1) for v in o.values())
        elif isinstance(o, (list, tuple, set, frozenset)):
            stack.extend((v, d + 1) for v in o)
        else:
            dct = getattr(o, "__dict__", None)
            if isinstance(dct, dict):
                stack.extend((v, d + 1) for v in dct.values())
            for klass in type(o).__mro__:
                for s in getattr(klass, "__slots__", ()) or ():
                    if isinstance(s, str) and hasattr(o, s):
                        try:
                            stack.append((getattr(o, s), d + 1))
                        except Exception:  # noqa: BLE001
                            pass
    return memo


def clone_method(method):
    """Per-connection copy of a fitted method: python containers deep-copied (per-episode state stays per connection),
    every ndarray / tensor shared (fit results are read-only by the API contract). Falls back to a shallow copy."""
    try:
        return copy.deepcopy(method, _share_memo(method)), "deepcopy_shared_arrays"
    except Exception as e:  # noqa: BLE001
        log.warning("osplug: deepcopy of the fitted method failed (%s); using a shallow copy", e)
        return copy.copy(method), f"shallow ({type(e).__name__})"


def _np32(t) -> np.ndarray:
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(t, torch.Tensor):
        t = t.detach().to("cpu", dtype=torch.float32).numpy()
    return np.asarray(t, dtype=np.float32)


def _ro(a: np.ndarray) -> np.ndarray:
    v = a.view()
    v.flags.writeable = False
    return v


class _Buf:
    """Growable per-episode array (rows are never modified once written, views stay valid)."""

    def __init__(self, shape, dtype, cap=32):
        self.a = np.empty((cap, *shape), dtype)
        self.n = 0

    def append(self, x):
        if self.n == self.a.shape[0]:
            b = np.empty((2 * self.a.shape[0], *self.a.shape[1:]), self.a.dtype)
            b[: self.n] = self.a[: self.n]
            self.a = b
        self.a[self.n] = x
        self.n += 1

    def view(self, lo, hi):
        return _ro(self.a[lo:hi])


def _jsonable(x):
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, (np.integer, int)):
        return int(x)
    return x


# ------------------------------------------------------------------------------------------- runtime
class PluginRuntime:
    """Process-level state: the fitted method, library tables, logging."""

    def __init__(self, opts, model: str):
        from exp.offline_search.harness import api, dims, store

        self.api, self.dims, self.store = api, dims, store
        self.opts = opts
        m, s, a = store.parse_cell(opts.os_cell)
        if m != model:
            raise SystemExit(f"--os-cell {opts.os_cell} is a {m} cell but this is the {model} server")
        if a != "cache":
            raise SystemExit(f"--os-cell must be the pure-cache cell (<m>_<s>_cache), got {opts.os_cell}")
        self.model, self.suite, self.cell = m, s, opts.os_cell
        self.root = pathlib.Path(opts.os_root)
        self.lib_key = store.lib_key(self.cell)
        self.lib = store.LibraryView(self.root, self.lib_key, "current")
        self.H = dims.HORIZON[m]
        self.L = self.lib.L
        self.ids = list(self.lib.ids)
        if len(self.ids) != self.L:
            raise SystemExit(f"{self.lib.dir}: ids.json has {len(self.ids)} ids for L={self.L}")
        self.id_to_row = {eid: r for r, eid in enumerate(self.ids)}
        meta = self.lib.meta
        self.task_map = {str(k): int(v) for k, v in meta["task_map"].items()}
        self.expected_pkl = (meta.get("sources") or {}).get("pkl")
        self.cur_action = np.ascontiguousarray(self.lib.action, dtype=np.float32)
        if self.cur_action.shape[1:] != (self.H, dims.ACT_FULL_DIMS):
            raise SystemExit(f"library action shape {self.cur_action.shape} != (L, {self.H}, 32)")
        self.ep_index = {}
        epj = self.root / "queries" / self.cell / "episodes.json"
        if epj.exists():
            for i, e in enumerate(json.loads(epj.read_text())):
                self.ep_index[(int(e["task_id"]), int(e["init"]))] = i
        self.log_dir = pathlib.Path(opts.os_log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tag = re.sub(r"[^A-Za-z0-9_.-]", "_", opts.os_tag) if opts.os_tag else f"p{os.getpid()}"
        self.dec_path = self.log_dir / f"decisions_{self.tag}.jsonl"
        self.inputs_dir = self.log_dir / "inputs"
        if opts.os_log_inputs:
            self.inputs_dir.mkdir(parents=True, exist_ok=True)
        self._wlock = threading.Lock()
        self._vlock = threading.Lock()
        self._validated: dict = {}
        self._conn_ids = itertools.count()
        self.sessions: "weakref.WeakSet[PluginSession]" = weakref.WeakSet()
        self.native_mode = opts.os_method == "native"
        self.shadow_native = not opts.os_no_shadow_native
        self.method = None
        self.method_name = "native"
        self.clone_mode = None
        self.fit_info: dict = {}
        self.lib_sizes = {"current": self.L}
        self.tables = {"current": self.cur_action}
        if not self.native_mode:
            self._load_and_fit()
        self.emit({"ev": "startup", "schema": SCHEMA, "host": socket.gethostname(), "pid": os.getpid(),
                   "argv": sys.argv, "tag": self.tag, "model": m, "suite": s, "cell": self.cell,
                   "root": str(self.root), "method_spec": opts.os_method, "kwargs": opts.kwargs,
                   "method": self.method_name, "native_mode": self.native_mode, "shadow_native": self.shadow_native,
                   "tokens": opts.os_tokens, "log_inputs": bool(opts.os_log_inputs), "seed": opts.os_seed,
                   "H": self.H, "L": self.L, "lib_sizes": self.lib_sizes, "expected_pkl": self.expected_pkl,
                   "git_head": _git_head(), **self.fit_info})
        log.info("osplug ready: method=%s cell=%s L=%d libs=%s log=%s", self.method_name, self.cell, self.L,
                 self.lib_sizes, self.dec_path)

    # -- fit
    def _load_and_fit(self):
        api, store = self.api, self.store
        opts = self.opts
        cls, src = load_method_class(opts.os_method)
        art = pathlib.Path(opts.os_fit_artifact) if opts.os_fit_artifact else None
        registered = {}
        t0 = time.perf_counter()
        if art is not None and art.exists():
            with open(art, "rb") as f:
                blob = pickle.load(f)
            want = {"spec": opts.os_method, "kwargs": opts.kwargs, "cell": self.cell}
            got = {k: blob.get(k) for k in want}
            if got != want:
                raise SystemExit(f"fit artifact {art} was made for {got}, not {want}")
            method, registered, source = blob["method"], blob.get("registered", {}), "artifact"
            fit_s = float(blob.get("fit_s", float("nan")))
        else:
            method = cls(**opts.kwargs)
            api.check_method_attrs(method)
            scratch = self.log_dir / "scratch" / f"{self.tag}_{self.cell}"
            scratch.mkdir(parents=True, exist_ok=True)
            ctx = api.Context(root=self.root, cell=self.cell, seed=opts.os_seed, scratch=scratch)
            method.prof = api.NULL_PROFILER
            try:
                method.fit(self.lib, ctx)
            except api.SkipCell as e:
                raise SystemExit(f"method {method.name} skips cell {self.cell}: {e}") from None
            fit_s = time.perf_counter() - t0
            registered = ctx.registered
            source = "fit"
            if art is not None:
                art.parent.mkdir(parents=True, exist_ok=True)
                tmp = art.with_suffix(art.suffix + ".tmp")
                with open(tmp, "wb") as f:
                    pickle.dump({"method": method, "registered": registered, "spec": opts.os_method,
                                 "kwargs": opts.kwargs, "cell": self.cell, "fit_s": fit_s}, f, protocol=4)
                os.replace(tmp, art)
        api.check_method_attrs(method)
        method.prof = api.NULL_PROFILER
        for lname in store.library_names(self.root, self.lib_key):
            if lname == "current":
                continue
            bl = store.LibraryView(self.root, self.lib_key, lname)
            self.lib_sizes[lname] = bl.L
            self.tables[lname] = bl.action
        for name, ent in registered.items():
            self.lib_sizes[name] = int(ent["action"].shape[0])
            self.tables[name] = ent["action"]
        self.method = method
        self.method_name = method.name
        try:
            bpe = float(method.bytes_per_entry())
        except Exception:  # noqa: BLE001
            bpe = float("nan")
        _, self.clone_mode = clone_method(method)
        self.fit_info = {"method_src": src, "method_class": cls.__name__, "tier": getattr(method, "tier", None),
                         "family": getattr(method, "family", None),
                         "uses_nonlibrary_action": bool(getattr(method, "uses_nonlibrary_action", False)),
                         "uses_gt": bool(getattr(method, "uses_gt", False)), "fit_source": source,
                         "fit_s": round(fit_s, 3), "bytes_per_entry": bpe, "registered_libraries": sorted(registered),
                         "clone_mode": self.clone_mode}
        if getattr(method, "uses_gt", False):
            raise SystemExit(f"{method.name} declares uses_gt: a GT-reading reference method cannot run online")

    # -- logging
    def emit(self, row: dict) -> None:
        row.setdefault("ts", round(time.time(), 3))
        line = (json.dumps(_clean(row), default=_json_default, allow_nan=False) + "\n").encode()
        with self._wlock:
            fd = os.open(self.dec_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o664)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)

    # -- per-connection wiring
    def check_config(self, config) -> None:
        kb = getattr(config.key_builder, "type", None)
        if kb != KEY_BUILDER[self.model]:
            raise RuntimeError(f"osplug: served key_builder {kb!r} != {KEY_BUILDER[self.model]!r} (the store keys)")
        for f in ("vision_0", "vision_1", "robot_state"):
            if not getattr(getattr(config.keys, f), "enabled", False):
                raise RuntimeError(f"osplug: key field {f} must be enabled (the method needs it)")
        cp1 = config.checkpoints.get("cp1")
        if cp1 is None or not cp1.enabled:
            raise RuntimeError("osplug: the served yaml has no enabled cp1 checkpoint")
        if cp1.gate.type != "always_search":
            raise RuntimeError(f"osplug: cp1 gate must be always_search (every decision searched), got {cp1.gate.type}")
        pp = getattr(getattr(config.backend, "in_memory", None), "preload_path", None)
        if self.expected_pkl and not self.opts.os_allow_other_pkl:
            if pp is None or os.path.realpath(pp) != os.path.realpath(self.expected_pkl):
                raise RuntimeError(f"osplug: served library {pp} is not the store library source {self.expected_pkl} "
                                   f"(pass --os-allow-other-pkl to rely on the payload check alone)")

    def validate_library(self, storage) -> None:
        """Every store 'current' row i must be the served entry ids[i] with a bit-identical action (once per backend)."""
        backend = getattr(storage, "_backend", storage)
        key = id(backend)
        with self._vlock:
            if key in self._validated:
                return
            t0 = time.perf_counter()
            n = int(storage.count())
            bad = []
            for r, eid in enumerate(self.ids):
                try:
                    a = _np32(storage.fetch_payload(eid).action_chunk)
                except KeyError:
                    bad.append((r, "missing"))
                    continue
                if a.shape != self.cur_action[r].shape or not np.array_equal(a, self.cur_action[r]):
                    bad.append((r, "action"))
            if n != self.L or bad:
                raise RuntimeError(f"osplug: served library does not match store {self.lib_key}/current: count {n} vs "
                                   f"{self.L}, {len(bad)} mismatching rows (first {bad[:5]})")
            self._validated[key] = True
            self.emit({"ev": "library_validated", "entries": n, "seconds": round(time.perf_counter() - t0, 3)})

    def attach(self, comps: dict, config, yaml_id=None) -> dict:
        from openpi.cache.types import CheckpointID

        self.check_config(config)
        cp = CheckpointID.CP1
        strategies, judges = comps["search_strategies"], comps["judges"]
        if cp not in strategies:
            raise RuntimeError("osplug: no CP1 search strategy in the built components")
        native = strategies[cp]
        storage = comps["storage"]
        self.validate_library(storage)
        sess = PluginSession(self, native, storage, comps.get("key_builder"), yaml_id)
        if self.native_mode:
            strategies[cp] = NativeProxy(native, sess)
        else:
            strategies[cp] = PluginStrategy(sess, native)
            judges[cp] = PluginJudge(sess)
            comps["storage"] = PluginStorage(storage, sess)
        pending = getattr(_TLS, "new_sessions", None)
        if pending is not None:
            pending.append(sess)
        else:
            log.warning("osplug: session built outside a wrapped connection factory (no wire obs / identity)")
        return comps

    def flush_all(self) -> None:
        for s in list(self.sessions):
            try:
                s.finish_episode(reason="exit")
            except Exception:  # noqa: BLE001
                log.exception("osplug: flush at exit failed")


def _clean(o):
    """JSON-safe copy: non-finite floats -> None, numpy scalars / arrays -> python."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (float, np.floating, np.integer, np.bool_, bool, int)):
        return _jsonable(o)
    return o


def _json_default(o):
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()] if o.ndim == 1 else o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return _jsonable(o)
    return str(o)


def _git_head():
    try:
        h = (REPO / ".git" / "HEAD").read_text().strip()
        if h.startswith("ref: "):
            return (REPO / ".git" / h[5:]).read_text().strip()[:12]
        return h[:12]
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------------------------------- session
class PluginSession:
    """Per-connection state: the method copy, the episode's online history, the pending decision record."""

    def __init__(self, rt: PluginRuntime, native, storage, key_builder, yaml_id):
        self.rt = rt
        self.native = native
        self.storage = storage
        self.kb = key_builder
        self.yaml_id = yaml_id
        self.bundle_id = None
        self.conn = next(rt._conn_ids)
        self.method = None
        if not rt.native_mode:
            self.method, _ = clone_method(rt.method)
            self.method.prof = rt.api.NULL_PROFILER
        self.shadow = rt.shadow_native and not rt.native_mode
        self.pending = True
        self.ident: dict = {}
        self.ep = None
        self.ep_meta: dict = {}
        self.ep_count = 0
        self.step = 0
        self.n_synth = 0
        self.cur_obs = None
        self.t_obs = None
        self._dec = None
        self._synth: dict = {}
        self._tok_cache: dict = {}
        self._recs: list = []
        rt.sessions.add(self)

    # -- identity / lifecycle (from _ConnPolicy and the strategy)
    def client_episode_start(self, kw: dict) -> None:
        self.finish_episode(reason="episode_start")
        self.ident = {"task": str(kw.get("task", "") or ""), "episode_id": kw.get("episode_id", -1),
                      "experiment": kw.get("experiment"), "extra": dict(kw.get("extra_metadata") or {})}
        self.pending = True

    def client_episode_end(self, success) -> None:
        self.finish_episode(reason="episode_end", success=bool(success))

    def mark_pending(self) -> None:
        self.pending = True

    def _begin(self, ctx) -> None:
        rt, api = self.rt, self.rt.api
        task = self.ident.get("task") or (ctx.task_key or "")
        if ctx.task_key and task and ctx.task_key != task:
            raise RuntimeError(f"osplug: episode task {task!r} != orchestrator task_key {ctx.task_key!r}")
        tid = rt.task_map.get(task)
        if tid is None:
            raise RuntimeError(f"osplug: task {task!r} is not in the {rt.lib_key} library task map")
        extra = self.ident.get("extra") or {}
        if extra.get("task_id") is not None and int(extra["task_id"]) != tid:
            raise RuntimeError(f"osplug: client task_id {extra['task_id']} != library task_id {tid} for {task!r}")
        init = int(extra.get("orig_init_state_idx", -1))
        uid = str(extra.get("task_uid") or f"{rt.tag}:c{self.conn}:e{self.ep_count}")
        seed = ep_seed(rt.opts.os_seed, uid)
        self.ep = api.EpisodeView(uid, task, tid, init, rt.ep_index.get((tid, init), -1), seed)
        self.ep_meta = {"uid": uid, "attempt": int(extra.get("attempt", 1) or 1), "task": task, "task_id": tid,
                        "init": init, "index": self.ep.index, "seed": seed, "yaml_id": self.yaml_id,
                        "bundle": self.bundle_id, "conn": self.conn, "episode_seq": self.ep_count,
                        "t_start": time.time()}
        H = rt.H
        drs = int(ctx.query_keys["robot_state"].numel()) if hasattr(ctx.query_keys["robot_state"], "numel") else \
            int(np.asarray(ctx.query_keys["robot_state"]).size)
        kd = rt.dims.KEY_DIM
        self.b_v0 = _Buf((kd,), np.float32)
        self.b_v1 = _Buf((kd,), np.float32)
        self.b_rs = _Buf((drs,), np.float32)
        self.b_raw = _Buf((rt.dims.RAW_STATE_DIM,), np.float32)
        self.b_aex = _Buf((H, rt.dims.ACT_FULL_DIMS), np.float32)
        self.hits: list = []
        self._recs = []
        self.step = 0
        self.ep_count += 1
        self.pending = False
        if self.method is not None:
            self.method.reset(self.ep)

    # -- per decision
    def _raw_state(self) -> np.ndarray:
        obs = self.cur_obs
        if obs is None or WIRE_STATE not in obs:
            raise RuntimeError("osplug: no wire observation for this decision (server not wrapped / not concurrent)")
        st = np.asarray(obs[WIRE_STATE], dtype=np.float32).reshape(-1)
        if st.shape != (self.rt.dims.RAW_STATE_DIM,):
            raise RuntimeError(f"osplug: wire state shape {st.shape} != (8,)")
        return st

    def _push_inputs(self, ctx) -> None:
        if self.pending or self.ep is None or (ctx.current_step == 0 and self.step > 0):
            if self.ep is not None and self.step > 0:
                self.finish_episode(reason="implicit_restart")
            self._begin(ctx)
        if self.b_aex.n != self.step:
            raise RuntimeError(f"osplug: executed chunks recorded {self.b_aex.n} != decisions {self.step}")
        qk = ctx.query_keys
        self.b_v0.append(_np32(qk["vision_0"]).reshape(-1))
        self.b_v1.append(_np32(qk["vision_1"]).reshape(-1))
        self.b_rs.append(_np32(qk["robot_state"]).reshape(-1))
        self.b_raw.append(self._raw_state())
        self._tok_cache = {}

    def on_search(self, ctx):
        """Method mode: build the online QueryView, run the method, return the served winner as one result."""
        from openpi.cache.storage_types import SearchResultLite

        rt, api = self.rt, self.rt.api
        t_all = time.perf_counter_ns()
        self._push_inputs(ctx)
        step = self.step
        view = OnlineQueryView(self, step, self.ep.task_id, self.ep)
        t0 = time.perf_counter_ns()
        res = self.method.query(view)
        t_q = time.perf_counter_ns() - t0
        topk, scores, conf, lib, act, ex = api.validate_result(
            res, lib_sizes=rt.lib_sizes, H=rt.H, where=f"online uid={self.ep.uid} step={step}")
        top1 = int(topk[0])
        base = np.asarray(rt.tables[lib][top1], dtype=np.float32)
        nfix = 0
        if act is not None:
            served = np.array(act, dtype=np.float32, copy=True)
            bad = ~np.isfinite(served)
            nfix = int(bad.sum())
            if nfix:
                served[bad] = base[bad]
            eid = f"{SYNTH_PREFIX}synth:{rt.tag}:{self.conn}:{self.n_synth}"
            self.n_synth += 1
            self._synth = {eid: served}
        elif lib == "current":
            served = rt.cur_action[top1]
            eid = rt.ids[top1]
            self._synth = {}
        else:
            served = base
            eid = f"{SYNTH_PREFIX}{lib}:{top1}"
            self._synth = {eid: served}
        n_top1, n_score, t_nat = -1, float("nan"), 0
        if self.shadow:
            t1 = time.perf_counter_ns()
            nres = self.native.search(ctx)
            t_nat = time.perf_counter_ns() - t1
            if nres:
                n_top1 = rt.id_to_row.get(nres[0].id, -2)
                n_score = float(nres[0].score)
        s0 = float(scores[0])
        self._dec = {"step": step, "ctx_step": int(ctx.current_step), "top1": top1, "lib": lib,
                     "topk": topk[:api.TOPK_SAVE], "scores": scores[:api.TOPK_SAVE], "conf": float(conf),
                     "synth": act is not None, "nfix": nfix, "winner": eid, "q_us": t_q / 1e3,
                     "native_us": t_nat / 1e3, "native_top1": n_top1, "native_score": n_score,
                     "agree": (n_top1 == top1 and lib == "current") if self.shadow else None,
                     "served": served, "extras": ex, "act": None if act is None else served}
        self.step += 1
        t_end = time.perf_counter_ns()
        self._dec["search_us"] = (t_end - t_all) / 1e3
        self._dec["t_s0"], self._dec["t_s1"] = t_all, t_end
        return [SearchResultLite(id=eid, score=s0 if math.isfinite(s0) else 0.0, checkpoint_id=ctx.checkpoint_id)]

    def on_search_native(self, ctx, results, t_ns) -> None:
        """Native mode: log the native winner (inputs too with --os-log-inputs)."""
        t_s1 = time.perf_counter_ns()
        self._push_inputs(ctx)
        top1 = self.rt.id_to_row.get(results[0].id, -2) if results else -1
        sc = float(results[0].score) if results else float("nan")
        served = self.rt.cur_action[top1] if top1 >= 0 else None
        self._dec = {"step": self.step, "ctx_step": int(ctx.current_step), "top1": top1, "lib": "current",
                     "topk": np.array([top1]), "scores": np.array([sc]), "conf": sc, "synth": False, "nfix": 0,
                     "winner": results[0].id if results else None, "q_us": t_ns / 1e3, "native_us": t_ns / 1e3,
                     "native_top1": top1, "native_score": sc, "agree": None, "served": served, "extras": None,
                     "act": None, "search_us": t_ns / 1e3, "t_s0": t_s1 - t_ns, "t_s1": t_s1}
        self.step += 1

    def synthetic_payload(self, eid: str):
        from openpi.cache.storage_types import CachePayload
        import torch

        a = self._synth.get(eid)
        if a is None:
            raise KeyError(f"osplug: no synthesized payload {eid!r} for the current decision")
        return CachePayload(action_chunk=torch.from_numpy(np.array(a, dtype=np.float32, copy=True)))

    def on_executed(self, chunk) -> None:
        if self.ep is None:
            return
        a = _np32(chunk)
        H = self.rt.H
        if a.shape != (H, self.rt.dims.ACT_FULL_DIMS):
            a = a.reshape(H, self.rt.dims.ACT_FULL_DIMS)
        self.b_aex.append(a)
        self.hits.append(1)
        if self._dec is not None:
            srv = self._dec.get("served")
            self._dec["exec_ok"] = bool(srv is not None and np.array_equal(a, srv))

    def wire_diag(self):
        d = self._dec
        if d is None:
            return None
        return {"os_row": d["top1"], "os_lib": d["lib"], "os_conf": _jsonable(d["conf"]),
                "os_q_us": round(d["q_us"], 1), "os_native_row": d["native_top1"], "os_agree": d["agree"]}

    def set_obs(self, obs) -> None:
        self.cur_obs = obs
        self.t_obs = time.perf_counter_ns()

    def after_infer(self, infer_ms: float, ok: bool, err: str | None = None) -> None:
        t_done = time.perf_counter_ns()
        d, self._dec = self._dec, None
        self.cur_obs = None
        self._synth = {}
        self._tok_cache = {}
        if d is None:
            return
        rt = self.rt
        m = self.ep_meta
        ex = d.get("extras") or {}
        exs = {}
        for k, v in ex.items():
            va = np.asarray(v)
            if va.size == 1 and len(exs) < EXTRA_SCALARS_MAX:
                exs[k] = _jsonable(va.reshape(-1)[0])
        row = {"ev": "dec", "tag": rt.tag, "conn": self.conn, "bundle": self.bundle_id, "yaml_id": self.yaml_id,
               "uid": m.get("uid"), "attempt": m.get("attempt"), "task_id": m.get("task_id"), "init": m.get("init"),
               "step": d["step"], "ctx_step": d["ctx_step"], "method": rt.method_name, "lib": d["lib"],
               "top1": d["top1"], "topk": d["topk"], "scores": d["scores"], "conf": _jsonable(d["conf"]),
               "synth": d["synth"], "nfix": d["nfix"], "winner": d["winner"], "q_us": round(d["q_us"], 1),
               "search_us": round(d["search_us"], 1), "native_us": round(d["native_us"], 1),
               "native_top1": d["native_top1"], "native_score": _jsonable(d["native_score"]), "agree": d["agree"],
               "exec_ok": d.get("exec_ok"), "infer_ms": round(infer_ms, 3), "ok": ok,
               "pre_ms": round((d["t_s0"] - self.t_obs) / 1e6, 3) if getattr(self, "t_obs", None) else None,
               "post_ms": round((t_done - d["t_s1"]) / 1e6, 3)}
        if err:
            row["error"] = err[-2000:]
        if exs:
            row["extras"] = exs
        rt.emit(row)
        if rt.opts.os_log_inputs:
            self._recs.append({"top1": d["top1"], "lib": d["lib"], "topk": np.asarray(d["topk"]),
                               "scores": np.asarray(d["scores"], np.float64), "conf": d["conf"],
                               "act": d["act"], "extras": ex, "native_top1": d["native_top1"],
                               "native_score": d["native_score"], "q_us": d["q_us"], "infer_ms": infer_ms})

    def finish_episode(self, reason: str, success=None) -> None:
        if self.ep is None:
            return
        ep, m = self.ep, dict(self.ep_meta)
        self.ep = None
        rt = self.rt
        n = self.step
        m.update({"ev": "episode", "tag": rt.tag, "reason": reason, "success": success, "n_decisions": n,
                  "n_exec": self.b_aex.n, "method": rt.method_name, "t_end": time.time()})
        rt.emit(m)
        if not rt.opts.os_log_inputs or n == 0:
            return
        try:
            self._write_inputs(m)
        except Exception:  # noqa: BLE001
            log.exception("osplug: writing the episode inputs failed")

    def _write_inputs(self, m: dict) -> None:
        rt = self.rt
        n = min(self.step, len(self._recs))     # a decision whose infer raised has no record
        recs = self._recs[:n]
        k = rt.api.TOPK_SAVE
        topk = np.full((len(recs), k), -1, np.int64)
        scores = np.full((len(recs), k), np.nan, np.float64)
        synth = np.full((len(recs), rt.H, rt.dims.ACT_FULL_DIMS), np.nan, np.float32)
        for i, r in enumerate(recs):
            topk[i, : r["topk"].size] = r["topk"]
            scores[i, : r["scores"].size] = r["scores"]
            if r["act"] is not None:
                synth[i] = r["act"]
        arrays = {
            "meta": np.array(json.dumps({**m, "model": rt.model, "suite": rt.suite, "cell": rt.cell,
                                         "method_spec": rt.opts.os_method, "kwargs": rt.opts.kwargs,
                                         "native_mode": rt.native_mode, "H": rt.H, "run_seed": rt.opts.os_seed,
                                         "tokens": rt.opts.os_tokens, "root": str(rt.root)},
                                        default=_json_default)),
            "step": np.arange(n, dtype=np.int16),
            "key_v0": self.b_v0.a[:n], "key_v1": self.b_v1.a[:n], "rs": self.b_rs.a[:n],
            "raw_state": self.b_raw.a[:n], "a_exec": self.b_aex.a[: self.b_aex.n],
            "top1": np.array([r["top1"] for r in recs], np.int64),
            "lib": np.array([r["lib"] for r in recs]),
            "topk": topk, "scores": scores, "conf": np.array([r["conf"] for r in recs], np.float64),
            "used_synth": np.array([r["act"] is not None for r in recs]), "synth": synth,
            "native_top1": np.array([r["native_top1"] for r in recs], np.int64),
            "native_score": np.array([r["native_score"] for r in recs], np.float64),
            "q_us": np.array([r["q_us"] for r in recs], np.float64),
            "infer_ms": np.array([r["infer_ms"] for r in recs], np.float64),
        }
        keys = sorted({kk for r in recs for kk in (r["extras"] or {})})
        for kk in keys:
            col = np.full(len(recs), np.nan, np.float64)
            for i, r in enumerate(recs):
                v = (r["extras"] or {}).get(kk)
                if v is not None and np.asarray(v).size == 1:
                    col[i] = float(np.asarray(v).reshape(-1)[0])
            arrays[f"x_{kk}"] = col
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(m.get("uid")))
        path = rt.inputs_dir / f"{rt.tag}_c{self.conn}_e{m.get('episode_seq')}_{safe}_a{m.get('attempt')}.npz"
        np.savez(path, **arrays)

    # -- tokens (online on every decision)
    def tokens(self, field: str) -> np.ndarray:
        api = self.rt.api
        if self.rt.opts.os_tokens != "on" or self.kb is None:
            raise api.TokensUnavailable("tokens are disabled on this server (--os-tokens off)")
        if field not in self._tok_cache:
            try:
                raw = self.kb._slice()  # noqa: SLF001 - the key builder's own token slice (what the keys pool)
                t = raw[field]
                t = t.reshape(-1, t.shape[-1]).detach().to("cpu")
                import torch

                self._tok_cache[field] = _ro(t.to(torch.float16).numpy())
            except Exception as e:  # noqa: BLE001
                raise api.TokensUnavailable(f"online token slice for {field} failed: {e}") from e
        return self._tok_cache[field]

    def image(self, key: str) -> np.ndarray:
        api = self.rt.api
        if self.rt.opts.os_tokens != "on" or self.cur_obs is None or key not in self.cur_obs:
            raise api.TokensUnavailable(f"wire image {key} unavailable")
        return _ro(np.asarray(self.cur_obs[key], dtype=np.uint8))


# ------------------------------------------------------------------------------------ online query view
class OnlineQueryView:
    """Online counterpart of harness api.QueryView: same attributes, same forbidden names."""

    __slots__ = ("_s", "step", "task_id", "episode")

    def __init__(self, session: PluginSession, step: int, task_id: int, episode):
        self._s = session
        self.step = step
        self.task_id = task_id
        self.episode = episode

    def __getattr__(self, name):
        api = RUNTIME.api if RUNTIME is not None else None
        if api is not None and name in api.FORBIDDEN:
            raise api.ForbiddenAccess(f"QueryView.{name} is not available online. Methods may use key_v0, key_v1, "
                                      f"rs, raw_state, task_id, step, episode, hist_* (earlier rows of this episode), "
                                      f"prev_a_exec, prev_hit, and tok_*/img*.")
        raise AttributeError(f"QueryView has no attribute {name!r}")

    @property
    def key_v0(self):
        return self._s.b_v0.view(self.step, self.step + 1)[0]

    @property
    def key_v1(self):
        return self._s.b_v1.view(self.step, self.step + 1)[0]

    @property
    def rs(self):
        return self._s.b_rs.view(self.step, self.step + 1)[0]

    @property
    def raw_state(self):
        return self._s.b_raw.view(self.step, self.step + 1)[0]

    @property
    def model(self):
        return self._s.rt.model

    @property
    def hist_key_v0(self):
        return self._s.b_v0.view(0, self.step)

    @property
    def hist_key_v1(self):
        return self._s.b_v1.view(0, self.step)

    @property
    def hist_rs(self):
        return self._s.b_rs.view(0, self.step)

    @property
    def hist_raw_state(self):
        return self._s.b_raw.view(0, self.step)

    @property
    def hist_a_exec(self):
        return self._s.b_aex.view(0, self.step)

    @property
    def prev_a_exec(self):
        return self._s.b_aex.view(self.step - 1, self.step)[0] if self.step > 0 else None

    @property
    def prev_hit(self):
        if self.step <= 0:
            return None
        f = self._s.hits[self.step - 1]
        return True if f == 1 else (False if f == 0 else None)

    @property
    def hist_hit(self):
        return np.asarray(self._s.hits[: self.step], dtype=np.int8)

    @property
    def has_tok(self):
        return self._s.rt.opts.os_tokens == "on"

    @property
    def tok_v0(self):
        return self._s.tokens("vision_0")

    @property
    def tok_v1(self):
        return self._s.tokens("vision_1")

    @property
    def img0(self):
        return self._s.image(WIRE_IMG0)

    @property
    def img1(self):
        return self._s.image(WIRE_IMG1)

    def _reference_gt(self):
        raise RUNTIME.api.ForbiddenAccess("ground truth is not available online")


# ------------------------------------------------------------------------------ strategy / judge / storage
class PluginStrategy:
    """CP1 search strategy: the method picks, the native strategy only shadows (and receives lifecycle calls)."""

    def __init__(self, session: PluginSession, native):
        self._s = session
        self._native = native

    def search(self, ctx):
        return self._s.on_search(ctx)

    def on_episode_start(self):
        self._s.mark_pending()
        fn = getattr(self._native, "on_episode_start", None)
        if fn is not None:
            fn()

    def get_search_session_id(self):
        fn = getattr(self._native, "get_search_session_id", None)
        return fn() if fn is not None else None

    def record_action(self, action_chunk):
        self._s.on_executed(action_chunk)
        fn = getattr(self._native, "record_action", None)
        if fn is not None:
            fn(action_chunk)

    def record_query_keys(self, query_keys):
        raise RuntimeError("osplug: a gate skipped the CP1 search; the pure-cache plugin needs always_search")


class NativeProxy:
    """Native mode: the native strategy unchanged, timed and logged."""

    def __init__(self, inner, session: PluginSession):
        self._osp_inner = inner
        self._s = session

    def search(self, ctx):
        t0 = time.perf_counter_ns()
        res = self._osp_inner.search(ctx)
        self._s.on_search_native(ctx, res, time.perf_counter_ns() - t0)
        return res

    def on_episode_start(self):
        self._s.mark_pending()
        fn = getattr(self._osp_inner, "on_episode_start", None)
        if fn is not None:
            fn()

    def record_action(self, action_chunk):
        self._s.on_executed(action_chunk)
        fn = getattr(self._osp_inner, "record_action", None)
        if fn is not None:
            fn(action_chunk)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_osp_inner"), name)


class PluginJudge:
    """Pure cache: FULL_HIT on the plugin's pick, always."""

    def __init__(self, session: PluginSession):
        self._s = session

    def __call__(self, results, checkpoint_id, cached_data, **kwargs):
        from openpi.cache.components.judge import HitType, JudgeResult

        if not results:
            raise RuntimeError("osplug: the plugin strategy returned no candidate")
        return JudgeResult(HitType.FULL_HIT, results[0].id, factor_outputs={"osplug": self._s.wire_diag()})

    def on_episode_start(self, extra_metadata=None, provisional=False):
        return None


class PluginStorage:
    """Per-connection facade proxy: synthesized / other-library payloads come from the session, the rest passes."""

    def __init__(self, inner, session: PluginSession):
        object.__setattr__(self, "_osp_inner", inner)
        object.__setattr__(self, "_s", session)

    def fetch_payload(self, id: str):
        if isinstance(id, str) and id.startswith(SYNTH_PREFIX):
            return self._s.synthetic_payload(id)
        return self._osp_inner.fetch_payload(id)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_osp_inner"), name)


# ----------------------------------------------------------------------------- connection wrapper
class _ConnPolicy:
    """Per-connection policy wrapper: wire obs + episode identity to the session, infer timing. The hasattr surface
    of the wrapped policy is preserved (lifecycle hooks are only defined when the inner policy has them)."""

    def __init__(self, inner, sessions, bundle_id):
        object.__setattr__(self, "_osp_inner", inner)
        object.__setattr__(self, "_osp_sessions", list(sessions))
        for s in sessions:
            s.bundle_id = bundle_id
        if hasattr(inner, "on_episode_start"):
            object.__setattr__(self, "on_episode_start", self._osp_episode_start)
        if hasattr(inner, "on_episode_end"):
            object.__setattr__(self, "on_episode_end", self._osp_episode_end)
        if hasattr(inner, "on_task_end"):
            object.__setattr__(self, "on_task_end", self._osp_task_end)

    def _osp_episode_start(self, **kw):
        for s in self._osp_sessions:
            s.client_episode_start(kw)
        return self._osp_inner.on_episode_start(**kw)

    def _osp_episode_end(self, *a, **kw):
        try:
            return self._osp_inner.on_episode_end(*a, **kw)
        finally:
            succ = kw.get("success", a[0] if a else None)
            for s in self._osp_sessions:
                s.client_episode_end(succ)

    def _osp_task_end(self, *a, **kw):
        try:
            return self._osp_inner.on_task_end(*a, **kw)
        finally:
            for s in self._osp_sessions:
                s.finish_episode(reason="task_end")

    def infer(self, obs, *a, **kw):
        sessions = self._osp_sessions
        for s in sessions:
            s.set_obs(obs)
        t0 = time.perf_counter()
        ok, err = False, None
        try:
            out = self._osp_inner.infer(obs, *a, **kw)
            ok = True
            return out
        except Exception:
            err = traceback.format_exc()
            raise
        finally:
            dt = (time.perf_counter() - t0) * 1e3
            for s in sessions:
                try:
                    s.after_infer(dt, ok, err)
                except Exception:  # noqa: BLE001
                    log.exception("osplug: decision logging failed")

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_osp_inner"), name)


def _wrap_factory(factory):
    def wrapped(base_policy, bundle_id="default"):
        prev = getattr(_TLS, "new_sessions", None)
        _TLS.new_sessions = []
        try:
            pol = factory(base_policy, bundle_id)
            sessions = _TLS.new_sessions
        finally:
            _TLS.new_sessions = prev
        return _ConnPolicy(pol, sessions, bundle_id)

    return wrapped


# ------------------------------------------------------------------------------------------ install
def install(opts, model: str) -> PluginRuntime:
    """Fit the method, then patch the per-connection component builder and the websocket server class."""
    global RUNTIME
    if RUNTIME is not None:
        raise RuntimeError("osplug already installed")
    RUNTIME = PluginRuntime(opts, model)

    import openpi.cache.config as cc
    from openpi.serving import websocket_policy_server as wps

    orig_build = cc.build_per_connection_components

    def build_per_connection_components(config, shared_storage, *args, **kwargs):
        comps = orig_build(config, shared_storage, *args, **kwargs)
        return RUNTIME.attach(comps, config, yaml_id=kwargs.get("yaml_id"))

    build_per_connection_components.__wrapped__ = orig_build
    cc.build_per_connection_components = build_per_connection_components

    base = wps.WebsocketPolicyServer

    class OsPluginWebsocketPolicyServer(base):
        def __init__(self, *args, **kwargs):
            f = kwargs.get("connection_policy_factory")
            if f is None:
                raise RuntimeError("osplug requires the concurrent server (a connection_policy_factory)")
            kwargs["connection_policy_factory"] = _wrap_factory(f)
            super().__init__(*args, **kwargs)

    wps.WebsocketPolicyServer = OsPluginWebsocketPolicyServer
    import atexit

    atexit.register(RUNTIME.flush_all)
    return RUNTIME


# ------------------------------------------------------------------------------------------- prefit
def prefit_main(argv=None) -> int:
    """Fit once and write --os-fit-artifact, so every server of an arm loads the same fit instead of refitting:

        taskset -c <cpus> .venv/bin/python -m exp.offline_search.closed_loop.plugin --os-method <spec> \
            --os-kwargs '<json>' --os-cell <m>_<s>_cache --os-log-dir <dir> --os-fit-artifact <run>/fits/<arm>.pkl
    """
    logging.basicConfig(level=logging.INFO)
    opts, rest = parse_cli(sys.argv[1:] if argv is None else argv)
    if rest:
        raise SystemExit(f"unexpected args {rest}")
    if not opts.os_fit_artifact:
        raise SystemExit("--os-fit-artifact is required")
    if pathlib.Path(opts.os_fit_artifact).exists():
        raise SystemExit(f"{opts.os_fit_artifact} exists; delete it to refit")
    from exp.offline_search.harness import store

    rt = PluginRuntime(opts, store.parse_cell(opts.os_cell)[0])
    print(json.dumps({"artifact": opts.os_fit_artifact, "bytes": os.path.getsize(opts.os_fit_artifact),
                      **_clean(rt.fit_info)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(prefit_main())
