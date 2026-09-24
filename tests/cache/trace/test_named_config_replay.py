"""Named production configs: HEAD vs trace-off vs trace-on, 50 decisions (plan §12-A-2/3).

Each fixture is a legal configuration accepted by the production validator
(``validate_cache_config``), assembled by the production factories
(``build_shared_storage`` / ``build_per_connection_components`` /
``build_trace_twins``). Three orchestrators run the same 2 episodes x 25
decisions over one shared backend:

* ``head`` -- ``CacheOrchestrator`` loaded from the pre-change commit
  ``54699e3`` (the components themselves are unchanged by this line);
* ``off``  -- the current orchestrator without twins;
* ``on``   -- the current orchestrator with the twin component set, driven
  through ``check(trace=True, fetch_top1=True)``.

After every decision the three must agree on the real ``CheckResult`` (the
``trace`` field excluded), the orchestrator's histories and counters, the
complete canonicalized state of every real gate / judge / strategy / key
builder, the real search session's score memo, the online judges' learning
state, and the exact arguments of every ``record_verdict`` /
``commit_verdict`` / ``record_continuation`` call. Session ids (uuid) are
mapped by first appearance within a run, never compared raw. The twin set
must be distinct objects with its own session and registry.

The threshold x {always_search, score_hysteresis} x {WSS, WSS depth 2, RRF}
subset is covered by the review suite; this file covers the remaining named
configurations. Illegal combinations (CRD / surface under a skipping gate,
surface under a drifted retrieval contract) are rejected by the existing
``tests/cache/test_crd_config.py`` and ``tests/cache/test_surface_binding.py``.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import importlib.util
import json
import pathlib
import pickle
import re
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from openpi.cache.components.factors.base import LibraryStats
from openpi.cache.components.judge import HitType
from openpi.cache.config import (
    BackendConfig,
    CacheConfig,
    CheckpointConfig,
    DepthPolicyConfig,
    GateConfig,
    InMemoryConfig,
    JudgeConfig,
    KeyBuilderConfig,
    KeyFieldConfig,
    KeysConfig,
    SearchStrategyConfig,
    WritePolicyConfig,
    build_per_connection_components,
    build_shared_storage,
    compute_surface_retrieval_contract,
    load_cache_config,
    validate_cache_config,
)
from openpi.cache.orchestrator import CacheOrchestrator
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.trace.runtime import build_trace_twins
from openpi.cache.types import PI05_V1, CheckpointID, groot_n15_schedule

REPO = pathlib.Path(__file__).resolve().parents[3]
HEAD_COMMIT = "54699e3"
CP1 = CheckpointID.CP1
D = 8
H, A = 5, 4
EPISODES, STEPS = 2, 25

# ---------------------------------------------------------------------------
# HEAD orchestrator
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _head_orchestrator_cls():
    try:
        src = subprocess.check_output(
            ["git", "show", f"{HEAD_COMMIT}:src/openpi/cache/orchestrator.py"],
            cwd=REPO, stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    cache = REPO / ".pytest_cache"
    cache.mkdir(exist_ok=True)
    path = cache / f"head_orchestrator_{HEAD_COMMIT}.py"
    path.write_bytes(src)
    name = f"openpi_head_orchestrator_{HEAD_COMMIT}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.CacheOrchestrator


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------

_UUID_RE = re.compile(r"^[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}$")
_SKIP_TYPES: tuple = ()
_SKIP_KEYS = {"logger", "_logger", "_log", "_lock", "_rlock", "_timer", "timer", "_registry"}


def _skip_value(v: Any) -> bool:
    from openpi.cache.backend_base import VectorStoreBackend
    from openpi.cache.cache_storage import CacheStorage

    if isinstance(v, (CacheStorage, VectorStoreBackend)):
        return True
    if callable(v) and not isinstance(v, (type, enum.Enum)) and not hasattr(v, "__dict__"):
        return True  # functions / builtins / bound methods
    if isinstance(v, (type(sys), type(threading_lock()))):
        return True
    tname = type(v).__name__
    return tname in {"Thread", "Event", "Condition", "SystemTimer", "Logger", "TextIOWrapper",
                     "BufferedWriter", "CurveRegistry", "method", "function"}


def threading_lock():
    import threading

    return threading.Lock()


class Canon:
    """Deterministic, role-mapped structural dump of an object graph.

    ``aliases`` maps run-specific strings (a run's own output directory) to a
    role token so that the one intended difference between runs is not
    reported as a state difference.
    """

    def __init__(self, aliases: dict[str, str] | None = None) -> None:
        self.ids: dict[str, str] = {}
        self.aliases = dict(aliases or {})

    def _sid(self, s: str) -> str:
        if s not in self.ids:
            self.ids[s] = f"<uuid#{len(self.ids)}>"
        return self.ids[s]

    def __call__(self, obj: Any, depth: int = 0, seen: frozenset = frozenset()) -> Any:
        if depth > 12:
            return f"<depth:{type(obj).__name__}>"
        if obj is None or isinstance(obj, (bool, int)):
            return obj
        if isinstance(obj, float):
            return "nan" if obj != obj else repr(obj)
        if isinstance(obj, str):
            for src, dst in self.aliases.items():
                obj = obj.replace(src, dst)
            return self._sid(obj) if _UUID_RE.match(obj) else obj
        if isinstance(obj, pathlib.PurePath):
            return ("path", self(str(obj), depth, seen))
        if isinstance(obj, enum.Enum):
            return f"{type(obj).__name__}.{obj.name}"
        if isinstance(obj, torch.Tensor):
            t = obj.detach().cpu()
            return ("tensor", str(t.dtype), tuple(t.shape), t.contiguous().view(-1).numpy().tobytes().hex()
                    if t.dtype != torch.bfloat16 else t.float().numpy().tobytes().hex())
        if isinstance(obj, np.ndarray):
            return ("ndarray", str(obj.dtype), obj.shape, np.ascontiguousarray(obj).tobytes().hex())
        if isinstance(obj, np.generic):
            return self(obj.item(), depth, seen)
        if isinstance(obj, torch.Generator):
            return ("generator", str(obj.device), obj.get_state().numpy().tobytes().hex())
        if id(obj) in seen:
            return f"<cycle:{type(obj).__name__}>"
        seen = seen | {id(obj)}
        if isinstance(obj, dict):
            items = [(self(k, depth + 1, seen), self(v, depth + 1, seen)) for k, v in obj.items()
                     if not (isinstance(k, str) and k in _SKIP_KEYS) and not _skip_value(v)]
            return ("dict", sorted(items, key=lambda kv: repr(kv[0])))
        if isinstance(obj, (list, tuple)):
            return (type(obj).__name__, [self(v, depth + 1, seen) for v in obj])
        if isinstance(obj, (set, frozenset)):
            return ("set", sorted(repr(self(v, depth + 1, seen)) for v in obj))
        if type(obj).__name__ == "deque":
            return ("deque", [self(v, depth + 1, seen) for v in obj])
        if _skip_value(obj):
            return f"<skip:{type(obj).__name__}>"
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return (type(obj).__name__, self({f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}, depth + 1, seen))
        state: dict = {}
        if hasattr(obj, "__dict__"):
            state.update(vars(obj))
        for klass in type(obj).__mro__:
            for slot in getattr(klass, "__slots__", ()) or ():
                if slot not in ("__dict__", "__weakref__") and hasattr(obj, slot):
                    state[slot] = getattr(obj, slot)
        if state or hasattr(obj, "__dict__"):
            state = {k: v for k, v in state.items() if k not in _SKIP_KEYS and not _skip_value(v)}
            return (type(obj).__name__, self(state, depth + 1, seen))
        text = repr(obj)
        if re.search(r" at 0x[0-9a-f]+", text):
            raise TypeError(f"canonicalizer cannot compare {type(obj).__name__} by value: {text}")
        return text


# ---------------------------------------------------------------------------
# Library / config fixtures
# ---------------------------------------------------------------------------


def _library_entries(schedule, *, with_outcome: bool = False):
    g = torch.Generator().manual_seed(0)
    entries = []
    for traj in range(3):
        prev = None
        for s in range(10):
            key = F.normalize(torch.randn(D, generator=g), dim=0)
            inter = {schedule.snapshot_t(i): torch.randn(H, A, generator=g) for i in range(1, schedule.num_steps)}
            payload = CachePayload(
                action_chunk=torch.randn(H, A, generator=g), intermediates=inter,
                denoising_num_steps=schedule.num_steps, schedule_id=schedule.schedule_id, task_key="t",
            )
            eid = f"t{traj}:{s}"
            entries.append(CacheEntry(
                id=eid, checkpoint_id=CP1, query_keys={"robot_state": key}, payload=payload,
                step_idx=s, prev_ids=[prev] if prev else [], trajectory_id=f"t{traj}",
                outcome=(1 if traj != 2 else -1) if with_outcome else None,
            ))
            prev = eid
    for a, b in zip(entries, entries[1:]):
        if b.prev_ids == [a.id]:
            a.next_ids = [b.id]
    return entries


def _write_library(tmp_path, schedule=PI05_V1, *, name="lib.pkl", with_outcome=False) -> pathlib.Path:
    stats = LibraryStats(
        action_sigma=torch.ones(A), action_active_mask=torch.ones(A, dtype=torch.bool),
        state_sigma=torch.ones(D), state_active_mask=torch.ones(D, dtype=torch.bool),
    )
    art = {
        "key_builder_type": "placeholder", "checkpoint_id": "CP1", "vector_dims": {"robot_state": D},
        "entries": _library_entries(schedule, with_outcome=with_outcome), "library_stats": stats,
        "schedule_id": schedule.schedule_id,
    }
    p = tmp_path / name
    with open(p, "wb") as fh:
        pickle.dump(art, fh)
    return p


def _config(preload, *, judge: dict, strategy: dict, gate: dict | None = None,
            schedule: str | None = None, name: str = "cfg.yaml") -> CacheConfig:
    """Serialize to yaml and load through the production loader + validator."""
    doc = {
        "enabled": True,
        "keys": {"robot_state": {"enabled": True, "weight": 1.0}},
        "key_builder": {"type": "placeholder"},
        "backend": {"type": "in_memory", "vector_dims": {"robot_state": D},
                    "in_memory": {"preload_path": str(preload)}},
        "checkpoints": {"cp1": {"enabled": True, "gate": gate or {"type": "always_search"},
                                "judge": judge, "search_strategy": strategy}},
        "write_policy": {"type": "never"},
    }
    if schedule is not None:
        doc["denoise_schedule"] = schedule
    path = pathlib.Path(preload).parent / name
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return load_cache_config(str(path))


_WSS_NORM = {"type": "per_field", "fields": {"robot_state": {"method": "affine_clip", "params": {"lo": -1.0, "hi": 1.0}}}}
_WSS_SIM = {"robot_state": {"type": "cosine"}}


def _wss(**over) -> dict:
    base = dict(type="weighted_score_sum_knn", top_k=2, field_similarity=_WSS_SIM, score_normalization=_WSS_NORM)
    base.update(over)
    return base


def _threshold(full=0.97, warm=0.85, start_t=0.5) -> dict:
    return {"type": "threshold", "threshold": full, "warm_tiers": [{"threshold": warm, "start_t": start_t}]}


def fx_follow_winner(tmp_path):
    lib = _write_library(tmp_path)
    return _config(lib, judge=_threshold(), strategy=_wss(),
                   gate={"type": "follow_winner", "lock_streak": 2, "budget": 4})


def fx_rrf_plain(tmp_path):
    lib = _write_library(tmp_path)
    return _config(lib, judge={"type": "threshold", "threshold": 0.0165,
                               "warm_tiers": [{"threshold": 0.0163, "start_t": 0.7}]},
                   strategy={"type": "weighted_rrf_knn", "top_k": 2})


def fx_dynamic_depth(tmp_path):
    lib = _write_library(tmp_path)
    return _config(lib, judge=_threshold(), strategy=dict(
        type="dynamic_depth_knn", base_fusion="weighted_score_sum", top_k=2,
        field_similarity=_WSS_SIM, score_normalization=_WSS_NORM,
        trajectory_depth=3, trajectory_weights=[0.5, 0.3, 0.2], allowed_depths=[1, 3],
        depth_policy={"type": "heuristic", "smoothness_thresholds": [0.5], "fallback_depth": 1},
    ))


def fx_failure_aware_dual(tmp_path):
    lib = _write_library(tmp_path, with_outcome=True)
    return _config(lib, judge=dict(
        type="failure_aware_gate", threshold=0.6, gate_betas={"b0": -0.2, "b1": 1.0, "b3": 0.0},
        warm_tiers=[{"threshold": 0.4, "start_t": 0.5}], export_factor_outputs=True,
    ), strategy=dict(
        type="dual_retrieval_knn", base_fusion="weighted_score_sum", top_k=2, trajectory_depth=1,
        field_similarity=_WSS_SIM, score_normalization=_WSS_NORM,
        allowed_depths=[1], depth_policy={"type": "constant", "depth": 1},
        margin_lambda=0.5, enable_dual=True,
    ))


def _composite_yaml(tmp_path, lib, *, strategy: str) -> CacheConfig:
    calib = tmp_path / "calib.jsonl"
    with calib.open("w") as fh:
        for i in range(60):
            fh.write(json.dumps({"factor_raw": {"jerk_online_action__p1_f1": float(i) / 60.0}}) + "\n")
    text = f"""\
enabled: true
keys:
  robot_state: {{enabled: true, weight: 1.0}}
key_builder: {{type: placeholder}}
backend:
  type: in_memory
  vector_dims: {{robot_state: {D}}}
  in_memory:
    preload_path: {lib}
    index_type: brute_force
checkpoints:
  cp1:
    gate: {{type: always_search}}
    search_strategy: {strategy}
    judge:
      type: composite
      normalization:
        type: zscore
        params: {{}}
        stats_source: {{type: offline}}
      factors:
        - type: jerk_online_action
          params: {{windows: [{{past: 1, future: 1}}]}}
      calibration:
        type: percentile_rolling
        params: {{window_size: 50}}
        samples_source:
          type: offline
          offline:
            path: {calib}
            format: jsonl
      composer:
        type: weighted_sum
        weights: {{jerk_online_action__p1_f1: 1.0}}
        tier_thresholds: {{full_hit: 0.8, warm_start: 0.4}}
        warm_start_t: 0.5
write_policy: {{type: never}}
"""
    p = tmp_path / "composite.yaml"
    p.write_text(text)
    return load_cache_config(str(p))


_WSS_FLOW = ("{type: weighted_score_sum_knn, top_k: 2, field_similarity: {robot_state: {type: cosine}}, "
             "score_normalization: {type: per_field, fields: {robot_state: {method: affine_clip, "
             "params: {lo: -1.0, hi: 1.0}}}}")


def fx_composite_wss_d1(tmp_path):
    return _composite_yaml(tmp_path, _write_library(tmp_path), strategy=_WSS_FLOW + "}")


def fx_composite_wss_d2(tmp_path):
    return _composite_yaml(tmp_path, _write_library(tmp_path),
                           strategy=_WSS_FLOW + ", trajectory_depth: 2, trajectory_weights: [0.7, 0.3]}")


def _library_identity(cfg: CacheConfig) -> dict:
    shared = build_shared_storage(cfg)
    meta = shared.artifact_meta if hasattr(shared, "artifact_meta") else {}
    return {"library_sha256": meta["library_sha256"], "library_entry_count": meta["entry_count"],
            "action_dim": A, "num_steps": 10, "h_exec": 5, "policy_fingerprint": "fp"}


def fx_crd(tmp_path):
    from tests.cache.test_crd_judge import write_crd

    lib = _write_library(tmp_path)
    probe = _config(lib, judge={"type": "always_hit"}, strategy=_wss(), name="probe.yaml")
    contract = compute_surface_retrieval_contract(probe)
    contract.update(_library_identity(probe))
    path = write_crd(tmp_path, contract=contract, w=np.ones(A, np.float32), k=2, name="crd.npz", delta=0.85)
    return _config(lib, judge={"type": "dispatch_surface", "surface_artifact_path": path}, strategy=_wss())


def fx_surface(tmp_path):
    from openpi.cache.components.surface_judge import (
        CERTIFICATION_CONFORMAL,
        SURFACE_ARTIFACT_SCHEMA_VERSION,
        SurfaceArtifact,
        save_surface_artifact,
    )

    lib = _write_library(tmp_path)
    probe = _config(lib, judge={"type": "always_hit"}, strategy=_wss(), name="probe.yaml")
    contract = compute_surface_retrieval_contract(probe)
    contract.update(_library_identity(probe))
    art = SurfaceArtifact(
        schema_version=SURFACE_ARTIFACT_SCHEMA_VERSION, k=2, h_exec=5,
        w=np.ones(A, dtype=np.float32), active_mask=np.ones(A, dtype=bool),
        start_t_ws=0.3, delta=0.5, quantile_alpha=0.05, certification_mode=CERTIFICATION_CONFORMAL,
        uses_disagreement=True, v_bin_edges=np.array([0.0, 1.0]), s_min_full=np.array([0.97]),
        s_min_warm=np.array([0.85]), conformal_c=0.01, n_calibration_episodes=100,
        retrieval_contract=contract, meta={},
    )
    path = tmp_path / "surface.npz"
    save_surface_artifact(art, str(path))
    return _config(lib, judge={"type": "dispatch_surface", "surface_artifact_path": str(path)}, strategy=_wss())


def fx_router_tc(tmp_path):
    lib = _write_library(tmp_path)
    dump = tmp_path / "router_dump"
    dump.mkdir()
    return _config(lib, judge=dict(
        type="mlp_router", arms="tc", mode="argmax", constant_arm="cache", seed=7,
        feature_fields=["robot_state"], hidden=8, dump_dir=str(dump),
    ), strategy={"type": "weighted_rrf_knn", "top_k": 1})


def _online_rit(tmp_path, mode: str):
    from openpi.cache.backends.in_memory_backend import InMemoryBackend

    sched = groot_n15_schedule(8)
    lib = _write_library(tmp_path, schedule=sched, name=f"lib_{mode}.pkl")
    probe = InMemoryBackend({"robot_state": D})
    probe.load_artifact(str(lib))
    from tests.cache.test_config_online_rit import write_scales

    scales = write_scales(tmp_path / f"scales_{mode}.npz", dim=A,
                          library_sha256=probe.artifact_meta["library_sha256"])
    return _config(lib, judge=dict(
        type="online_rit", tiers=[0.875, 0.75, 0.5], alpha=0.05, delta=0.5,
        knots=[0.0, 0.25, 0.5, 0.75, 1.0], update_scales_path=scales, feedback_mode=mode,
        update_enabled=True, window=128, n_min=3, h_exec=3,
    ), strategy={"type": "weighted_rrf_knn", "top_k": 1}, schedule=sched.schedule_id, name=f"cfg_{mode}.yaml")


def fx_online_rit_fm0(tmp_path):
    return _online_rit(tmp_path, "fm0")


def fx_online_rit_fm1(tmp_path):
    return _online_rit(tmp_path, "fm1")


EMB = 16
PROMPTS = {"taskA": 0, "taskB": 1}


def _prompt_vec(task: str) -> torch.Tensor:
    g = torch.Generator().manual_seed(50 + PROMPTS[task])
    return torch.randn(EMB, generator=g)


def fx_text_ivf(tmp_path):
    """cp1_mean_pool keys (prompt_emb + robot_state) over a two-task library."""
    g = torch.Generator().manual_seed(3)
    entries = []
    for t_i, task in enumerate(PROMPTS):
        prev = None
        for s in range(10):
            state = torch.randn(32, generator=g)
            vision = torch.randn(EMB, generator=g)
            inter = {PI05_V1.snapshot_t(i): torch.randn(H, A, generator=g) for i in range(1, 10)}
            eid = f"{task}:{s}"
            entries.append(CacheEntry(
                id=eid, checkpoint_id=CP1,
                query_keys={"prompt_emb": _prompt_vec(task), "robot_state": state, "vision_0": vision},
                payload=CachePayload(action_chunk=torch.randn(H, A, generator=g), intermediates=inter,
                                     denoising_num_steps=10, schedule_id=PI05_V1.schedule_id, task_key="t"),
                step_idx=s, prev_ids=[prev] if prev else [], trajectory_id=task,
            ))
            prev = eid
    art = {"key_builder_type": "cp1_mean_pool", "checkpoint_id": "CP1",
           "vector_dims": {"prompt_emb": EMB, "robot_state": 32, "vision_0": EMB}, "entries": entries,
           "prompt_pool": {"masked": False, "instruction_span": False}, "schedule_id": PI05_V1.schedule_id}
    lib = tmp_path / "lib_text.pkl"
    with open(lib, "wb") as fh:
        pickle.dump(art, fh)
    doc = {
        "enabled": True,
        "keys": {"prompt_emb": {"enabled": True, "weight": 0.5}, "robot_state": {"enabled": True, "weight": 1.0},
                 "vision_0": {"enabled": True, "weight": 0.3}, "vision_1": {"enabled": False, "weight": 0.0},
                 "vision_2": {"enabled": False, "weight": 0.0}},
        "key_builder": {"type": "cp1_mean_pool"},
        "backend": {"type": "in_memory", "vector_dims": {"prompt_emb": EMB, "robot_state": 32, "vision_0": EMB},
                    "in_memory": {"preload_path": str(lib), "index_type": "text_ivf"}},
        "checkpoints": {"cp1": {"enabled": True, "gate": {"type": "always_search"},
                                "judge": _threshold(full=0.9, warm=0.6),
                                "search_strategy": {
                                    "type": "text_ivf_knn", "top_k": 2,
                                    "field_similarity": {"prompt_emb": {"type": "cosine"}, "robot_state": {"type": "l2"},
                                                         "vision_0": {"type": "cosine"}},
                                    "score_normalization": {"type": "per_field", "fields": {
                                        "vision_0": {"method": "affine_clip", "params": {"lo": -1.0, "hi": 1.0}},
                                        "robot_state": {"method": "exp_l2", "params": {"tau": 4.0}},
                                        "prompt_emb": {"method": "affine_clip", "params": {"lo": 0.0, "hi": 1.0}}}},
                                }}},
        "write_policy": {"type": "never"},
    }
    path = tmp_path / "cfg_text.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return load_cache_config(str(path))


def _text_stage1(step: int, entries) -> SimpleNamespace:
    e = entries[(step * 7) % len(entries)]
    g = torch.Generator().manual_seed(2000 + step)
    prompt = e.query_keys["prompt_emb"]
    tokens = prompt.expand(6, EMB).clone()
    vision = torch.zeros(768, EMB)
    vision[:256] = e.query_keys["vision_0"] + 0.3 * torch.randn(256, EMB, generator=g)
    state = e.query_keys["robot_state"] + [0.02, 0.3, 1.0, 3.0][step % 4] * torch.randn(32, generator=g)
    return SimpleNamespace(prefix_embs=torch.cat([vision, tokens])[None], state=state[None])


FIXTURES: dict[str, Callable] = {
    "follow_winner": fx_follow_winner,
    "rrf_plain": fx_rrf_plain,
    "dynamic_depth": fx_dynamic_depth,
    "failure_aware_dual": fx_failure_aware_dual,
    "composite_wss_d1": fx_composite_wss_d1,
    "composite_wss_d2": fx_composite_wss_d2,
    "crd": fx_crd,
    "surface": fx_surface,
    "text_ivf": fx_text_ivf,
    "router_tc_dump": fx_router_tc,
    "online_rit_fm0": fx_online_rit_fm0,
    "online_rit_fm1": fx_online_rit_fm1,
}


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

_SPIED = ("record_verdict", "commit_verdict", "record_continuation")


def _spy(obj: Any, log: list, canon: Canon, label: str) -> None:
    for name in _SPIED:
        fn = getattr(obj, name, None)
        if fn is None or not callable(fn):
            continue

        @functools.wraps(fn)
        def wrapper(*a, __fn=fn, __name=name, **k):
            log.append((label, __name, canon(list(a)), canon(k)))
            return __fn(*a, **k)

        setattr(obj, name, wrapper)


@dataclasses.dataclass
class Run:
    name: str
    orch: Any
    comps: dict
    canon: Canon
    calls: list
    registry: Any = None


def _per_run_config(cfg: CacheConfig, name: str) -> CacheConfig:
    """Give each run its own native output directory (router shards)."""
    cp = cfg.checkpoints["cp1"]
    if not cp.judge.dump_dir:
        return cfg
    run_dir = pathlib.Path(cp.judge.dump_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    judge = dataclasses.replace(cp.judge, dump_dir=str(run_dir))
    return dataclasses.replace(cfg, checkpoints={**cfg.checkpoints, "cp1": dataclasses.replace(cp, judge=judge)})


def _build_run(name: str, cfg: CacheConfig, shared, cls, *, twins: bool) -> Run:
    from openpi.cache.online_state import CurveRegistry

    run_cfg = _per_run_config(cfg, name)
    registry = CurveRegistry(state_log_root=None, require_persistence=False)
    torch.manual_seed(0)
    comps = build_per_connection_components(run_cfg, shared, quiet=True, yaml_id="arm", online_registry=registry)
    aliases = {}
    if run_cfg is not cfg:
        aliases[run_cfg.checkpoints["cp1"].judge.dump_dir] = "<run_dump_dir>"
    canon = Canon(aliases)
    calls: list = []
    for cp, gate in comps["gates"].items():
        _spy(gate, calls, canon, "gate")
    for cp, judge in comps["judges"].items():
        _spy(judge, calls, canon, "judge")
    kwargs = dict(
        storage=comps["storage"], key_builder=comps["key_builder"], gates=comps["gates"],
        judges=comps["judges"], search_strategies=comps["search_strategies"], timer=comps["timer"],
        write_policy=comps["write_policy"], offline_writers=comps["offline_writers"],
        library_stats=comps["library_stats"],
    )
    if twins:
        torch.manual_seed(0)
        kwargs["trace_twins"] = build_trace_twins(run_cfg, shared, real_components=comps, yaml_id="arm")
    return Run(name, cls(**kwargs), comps, canon, calls, registry)


def _query(step: int, entries: list[CacheEntry]) -> torch.Tensor:
    g = torch.Generator().manual_seed(1000 + step)
    base = entries[(step * 7) % len(entries)].query_keys["robot_state"]
    noise = torch.randn(D, generator=g)
    scale = [0.02, 0.2, 0.6, 1.5][step % 4]
    return (base + scale * noise)[None, :]


def _synth_chunk(step: int) -> torch.Tensor:
    return torch.full((H, A), 0.01 * step)


def _feedback(spec, step: int):
    from openpi.cache.components.online_rit import ContinuationFeedback

    return [ContinuationFeedback(t.index, 0.05 * ((step + j) % 7), "shadow") for j, t in enumerate(spec.tiers)]


_ORCH_STATE = ("_step_counter", "_action_history", "_state_history", "_miss_by_checkpoint",
               "_last_judge_commit", "_current_strategy_session_ids", "_current_task_key",
               "_current_episode_extra", "_episode_records", "_current_episode")


def _snapshot(run: Run) -> dict:
    o, c = run.orch, run.canon
    snap = {k: c(getattr(o, k)) for k in _ORCH_STATE if hasattr(o, k)}
    comps = run.comps
    snap["key_builder"] = c(comps["key_builder"])
    for cp in comps["gates"]:
        snap[f"gate:{cp.name}"] = c(comps["gates"][cp])
    for cp in comps["judges"]:
        snap[f"judge:{cp.name}"] = c(comps["judges"][cp])
    for cp in comps["search_strategies"]:
        snap[f"strategy:{cp.name}"] = c(comps["search_strategies"][cp])
    backend = comps["storage"]._backend if hasattr(comps["storage"], "_backend") else None
    memo = getattr(backend, "_score_memo", None)
    if memo is not None:
        sids = list(getattr(o, "_current_strategy_session_ids", []) or [])
        snap["memo"] = [c(memo.get(s)) for s in sids]
    for cp, judge in comps["judges"].items():
        key = getattr(judge, "_key", None)
        reg = getattr(judge, "_registry", None)
        if key is not None and reg is not None:
            snap[f"curves:{cp.name}"] = reg.curves(key).learning_state_sha256()
    return snap


def _result(run: Run, res) -> Any:
    fields = ("hit_type", "entry_id", "start_t", "score", "searched", "query_keys",
              "factor_outputs", "router_outputs", "hit_override")
    out = {f: run.canon(getattr(res, f, None)) for f in fields}
    payload = getattr(res, "payload", None)
    out["payload"] = None if payload is None else run.canon(payload.action_chunk)
    return out


def _step(run: Run, q: Any, step: int, *, trace: bool) -> Any:
    o = run.orch
    stage1 = q if isinstance(q, SimpleNamespace) else SimpleNamespace(state=q)
    if trace:
        res = o.check(CP1, trace=True, fetch_top1=True, stage1=stage1)
        assert res.trace is not None
    else:
        res = o.check(CP1, stage1=stage1)
    if res.hit_type == HitType.FULL_HIT and res.payload is not None:
        chunk = res.payload.action_chunk
    else:
        chunk = _synth_chunk(step)
    o.broadcast_action(chunk)
    if res.query_keys is not None:
        o.buffer_for_write(res.query_keys, chunk)
    spec = o.continuation_spec(CP1) if hasattr(o, "continuation_spec") else None
    if spec is not None:
        o.record_continuation(CP1, o.pending_decision(CP1), _feedback(spec, step),
                              fb_batch_size=len(spec.tiers))
        if trace:
            tspec = o.twin_continuation_spec(CP1)
            assert tspec is not None
            o.twin_record_continuation(CP1, o.twin_pending_decision(CP1), _feedback(tspec, step + 3))
    o.clear()
    return res


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_named_config_head_off_on_state_parity(tmp_path, monkeypatch, name):
    import time as _time

    # Frozen wall clock (plan §12-A-3): row / entry timestamps are identity,
    # not behaviour, and would otherwise differ between the three runs.
    monkeypatch.setattr(_time, "time", lambda: 1.0e9)
    head_cls = _head_orchestrator_cls()
    if head_cls is None:
        pytest.skip(f"baseline commit {HEAD_COMMIT} is not available in this checkout")
    cfg = FIXTURES[name](tmp_path)
    shared = build_shared_storage(cfg)
    runs = [
        _build_run("head", cfg, shared, head_cls, twins=False),
        _build_run("off", cfg, shared, CacheOrchestrator, twins=False),
        _build_run("on", cfg, shared, CacheOrchestrator, twins=True),
    ]
    _replay(name, cfg, shared, runs)


@pytest.mark.parametrize("name", ["follow_winner", "crd", "online_rit_fm1"])
def test_harness_detects_a_twin_that_shares_real_state(tmp_path, monkeypatch, name):
    """Negative control: with the twin wired to the REAL gate / judge objects
    (the defect the twin set exists to prevent) the replay must fail."""
    import time as _time

    from openpi.cache.orchestrator import TwinSet

    monkeypatch.setattr(_time, "time", lambda: 1.0e9)
    head_cls = _head_orchestrator_cls()
    if head_cls is None:
        pytest.skip(f"baseline commit {HEAD_COMMIT} is not available in this checkout")
    cfg = FIXTURES[name](tmp_path)
    shared = build_shared_storage(cfg)
    head = _build_run("head", cfg, shared, head_cls, twins=False)
    off = _build_run("off", cfg, shared, CacheOrchestrator, twins=False)
    on = _build_run("on", cfg, shared, CacheOrchestrator, twins=False)
    honest = build_trace_twins(cfg, shared, real_components=on.comps, yaml_id="arm")
    leaky = TwinSet(key_builder=honest.key_builder, gates=on.comps["gates"], judges=on.comps["judges"],
                    strategies=honest.strategies, storage=honest.storage, timer=honest.timer)
    on.orch.attach_trace_twins(leaky)
    with pytest.raises(AssertionError, match="off vs on"):
        _replay(name, cfg, shared, [head, off, on], check_distinct=False)


def _replay(name: str, cfg: CacheConfig, shared, runs: list, *, check_distinct: bool = True) -> None:
    entries = list(shared._backend._entries.values())
    on = runs[2]
    twins = on.orch._twins
    assert twins is not None
    if check_distinct:
        real_ids = {id(on.comps["key_builder"]), *map(id, on.comps["gates"].values()),
                    *map(id, on.comps["judges"].values()), *map(id, on.comps["search_strategies"].values())}
        twin_ids = {id(twins.key_builder), *map(id, twins.gates.values()), *map(id, twins.judges.values()),
                    *map(id, twins.strategies.values())}
        assert not (real_ids & twin_ids), "twin components must be distinct instances"

    hit_kinds: set[str] = set()
    for r in runs:
        r.orch.on_task_begin()
    step = 0
    for ep in range(EPISODES):
        meta = {"task_uid": f"u{ep}", "attempt": 1, "task_id": 0, "run_id": "r0", "batch_id": "b0"}
        for r in runs:
            r.orch.on_episode_start(task_key="t", episode_id=str(ep), extra_metadata=dict(meta))
        base = [_snapshot(r) for r in runs]
        assert base[0] == base[1] == base[2], f"{name}: state differs at episode {ep} start"
        for _ in range(STEPS):
            q = _text_stage1(step, entries) if name == "text_ivf" else _query(step, entries)
            results = [_step(r, q, step, trace=(r.name == "on")) for r in runs]
            canon_results = [_result(r, res) for r, res in zip(runs, results)]
            hit_kinds.add(str(canon_results[0]["hit_type"]))
            assert canon_results[0] == canon_results[1], f"{name} step {step}: HEAD vs off CheckResult"
            assert canon_results[1] == canon_results[2], f"{name} step {step}: off vs on CheckResult"
            snaps = [_snapshot(r) for r in runs]
            for key in snaps[0]:
                assert snaps[0][key] == snaps[1].get(key), f"{name} step {step}: HEAD vs off {key}"
                assert snaps[1][key] == snaps[2].get(key), f"{name} step {step}: off vs on {key}"
            assert runs[0].calls == runs[1].calls, f"{name} step {step}: HEAD vs off call args"
            assert runs[1].calls == runs[2].calls, f"{name} step {step}: off vs on call args"
            step += 1
        for r in runs:
            r.orch.on_episode_end()
    for r in runs:
        r.orch.on_task_end()
    snaps = [_snapshot(r) for r in runs]
    assert snaps[0] == snaps[1] == snaps[2], f"{name}: final state"
    dump_dir = cfg.checkpoints["cp1"].judge.dump_dir
    if dump_dir:
        trees = []
        for r in runs:
            root = pathlib.Path(dump_dir) / r.name
            trees.append({str(f.relative_to(root)): f.read_bytes() for f in sorted(root.rglob("*")) if f.is_file()})
        assert trees[0], f"{name}: the real router wrote no shard"
        assert trees[0] == trees[1] == trees[2], f"{name}: real router shards differ"
        extra = sorted(p.name for p in pathlib.Path(dump_dir).iterdir() if p.name not in {"head", "off", "on"})
        assert not extra, f"{name}: something besides the real routers wrote {extra}"
    # The twin searched every step on its own session, never the real one.
    real_sids = set(getattr(on.orch, "_current_strategy_session_ids", []) or [])
    twin_sids = set(twins.state.strategy_session_ids)
    assert not (real_sids & twin_sids)
    if on.registry is not None:
        from openpi.cache.trace import runtime as _rt

        twin_reg = _rt._TWIN_REGISTRY.get("registry")
        assert twin_reg is None or twin_reg is not on.registry
    # Coverage evidence recorded for the report (not an equality condition).
    print(f"{name}: verdicts seen {sorted(hit_kinds)}, calls {len(runs[0].calls)}")
