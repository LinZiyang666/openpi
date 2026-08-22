"""Non-manual tests for the concurrent GR00T serving factory.

Plan: logs/groot_concurrent_serving_plan.log.md (T1-T10). Everything here runs
without a GPU or the gr00t package: the heavy imports in
``serve_groot_n15.main`` sit behind argparse validation, the factory seams are
monkeypatched at the modules the factory imports from, and the backend stress
test drives a real ``InMemoryBackend`` directly.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import threading
import time
import types

import pytest
import torch

from exp.robocasa365 import serve_groot_n15 as sgn


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _RecordingInner:
    """Fake serving stack: counts infer overlap, records lifecycle calls."""

    def __init__(self) -> None:
        self.calls: list = []
        self.active = 0
        self.max_active = 0
        self._meter = threading.Lock()

    def infer(self, obs):
        with self._meter:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.005)
        with self._meter:
            self.active -= 1
        if isinstance(obs, RuntimeError):
            raise obs
        return {"actions": obs}

    def on_task_begin(self):
        self.calls.append("task_begin")

    def on_episode_start(self, **kwargs):
        self.calls.append(("episode_start", kwargs))

    def on_episode_end(self, success):
        self.calls.append(("episode_end", success))

    def on_task_end(self):
        self.calls.append("task_end")


def _args(**overrides) -> types.SimpleNamespace:
    ns = types.SimpleNamespace(
        cache_config=None, collect_hdf5=None, concurrent=True, diagnostic_seed=None,
        compile_stage1=False,
    )
    ns.__dict__.update(overrides)
    return ns


# ---------------------------------------------------------------------------
# T3 / T7 — _InferLockedPolicy
# ---------------------------------------------------------------------------


def test_lock_serializes_infer_across_threads():
    """T3: two wrappers on one lock never overlap inside the inner infer."""
    inner = _RecordingInner()
    lock = threading.Lock()
    a = sgn._InferLockedPolicy(inner, lock)
    b = sgn._InferLockedPolicy(inner, lock)

    def hammer(policy):
        for i in range(20):
            policy.infer(i)

    threads = [threading.Thread(target=hammer, args=(p,)) for p in (a, b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert inner.max_active == 1
    assert inner.active == 0


def test_lock_released_on_infer_exception():
    """T3 addendum: a raising infer must not leave the lock held."""
    inner = _RecordingInner()
    lock = threading.Lock()
    wrapper = sgn._InferLockedPolicy(inner, lock)
    with pytest.raises(RuntimeError):
        wrapper.infer(RuntimeError("boom"))
    assert wrapper.infer("after") == {"actions": "after"}


def test_lock_wrapper_forwards_hooks_and_hasattr_parity():
    """T7: hooks reach the inner stack; hasattr surface matches the inner."""
    inner = _RecordingInner()
    wrapper = sgn._InferLockedPolicy(inner, threading.Lock())

    wrapper.on_task_begin()
    wrapper.on_episode_start(task="OpenCabinet", episode_idx=3)
    wrapper.on_episode_end(True)
    wrapper.on_task_end()
    assert inner.calls == [
        "task_begin",
        ("episode_start", {"task": "OpenCabinet", "episode_idx": 3}),
        ("episode_end", True),
        "task_end",
    ]

    # The server feature-detects hooks with hasattr: the wrapper must not
    # invent prefill_trajectory when the inner stack has none...
    assert not hasattr(wrapper, "prefill_trajectory")

    # ...and must expose it when the inner stack has it.
    class _WithPrefill(_RecordingInner):
        def prefill_trajectory(self, *a, **kw):
            self.calls.append("prefill")

    rich = sgn._InferLockedPolicy(_WithPrefill(), threading.Lock())
    assert hasattr(rich, "prefill_trajectory")


# ---------------------------------------------------------------------------
# T4 / T6 — teacher-only factory + bundle contract
# ---------------------------------------------------------------------------


def test_teacher_factory_isolates_adapters_shares_policy():
    """T4: per-connection adapters are distinct; the GPU policy is shared."""
    policy = _RecordingInner()
    factory, label = sgn._build_concurrent_factory(policy, _args())
    assert "teacher-only" in label

    p1 = factory(policy)
    p2 = factory(policy)
    assert p1 is not p2
    assert p1._inner is not p2._inner  # distinct GrootPolicyAdapter instances
    assert p1._inner._policy is policy
    assert p2._inner._policy is policy
    assert p1._lock is p2._lock  # one shared inference lock


def test_factory_rejects_non_default_bundle():
    """T6: select_bundle with any other id must fail fast, never silently
    keep serving the CLI configuration under the requested name."""
    policy = _RecordingInner()
    factory, _ = sgn._build_concurrent_factory(policy, _args())
    with pytest.raises(ValueError, match="default"):
        factory(policy, "some_other_yaml")
    # The default id stays accepted (idempotent server-side re-select relies
    # on the factory not raising for it).
    assert factory(policy, "default") is not None


# ---------------------------------------------------------------------------
# T1 / T5 / T6 / T8 — cache factory with seam fakes
# ---------------------------------------------------------------------------


class _FakeTimer:
    def __init__(self) -> None:
        self.csv_dirs: list[str] = []

    def enable_csv(self, output_dir: str) -> None:
        self.csv_dirs.append(output_dir)


class _FakeInterceptor:
    """Stands in for GrootCacheInterceptor: records lifecycle per instance."""

    instances: list["_FakeInterceptor"] = []

    def __init__(self, policy, runner, *, orchestrator=None, timer=None):
        self.policy = policy
        self.runner = runner
        self.orchestrator = orchestrator
        self.timer = timer
        self.calls: list = []
        _FakeInterceptor.instances.append(self)

    def infer(self, obs):
        return {"actions": {"k": torch.zeros(2)}}

    def on_episode_start(self, **kwargs):
        self.calls.append(("episode_start", kwargs))

    def on_episode_end(self, success):
        self.calls.append(("episode_end", success))


class _FakeOrchestrator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeRunner:
    def __init__(self, model, *, timer=None, compile_vision=False):
        self.model = model
        self.timer = timer
        self.compile_vision = compile_vision


@pytest.fixture
def cache_seams(monkeypatch, tmp_path):
    """Patch every seam _build_concurrent_factory imports for the cache path."""
    import openpi.cache.config as cache_config
    import openpi.cache.groot.interceptor as gi
    import openpi.cache.groot.load_guard as lg
    import openpi.cache.groot.staged as gs
    import openpi.cache.orchestrator as orch

    cfg = types.SimpleNamespace(
        timer=types.SimpleNamespace(output_csv_dir=str(tmp_path / "csv")),
        key_builder=types.SimpleNamespace(type="cp1_groot_spatial_pool_16"),
    )
    shared_sentinel = object()
    record = {
        "load": [], "validate": [], "groot_validate": [], "identity": [],
        "shared": [], "per_conn": [],
    }

    monkeypatch.setattr(
        cache_config, "load_cache_config",
        lambda path: (record["load"].append(path), cfg)[1],
    )
    monkeypatch.setattr(
        cache_config, "validate_cache_config",
        lambda c: record["validate"].append(c),
    )
    monkeypatch.setattr(
        cache_config, "build_shared_storage",
        lambda c: (record["shared"].append(c), shared_sentinel)[1],
    )

    def fake_per_conn(config, shared_storage, *, yaml_id=None, quiet=False):
        record["per_conn"].append((config, shared_storage, quiet))
        return {
            "timer": _FakeTimer(),
            "storage": object(),
            "key_builder": object(),
            "gates": [],
            "judges": [],
            "search_strategies": [],
            "write_policy": object(),
            "offline_writers": [],
            "library_stats": object(),
        }

    monkeypatch.setattr(cache_config, "build_per_connection_components", fake_per_conn)
    monkeypatch.setattr(
        lg, "validate_groot_cache_config",
        lambda c: record["groot_validate"].append(c),
    )
    monkeypatch.setattr(
        lg, "validate_artifact_identity",
        lambda storage, c: record["identity"].append((storage, c)),
    )
    monkeypatch.setattr(orch, "CacheOrchestrator", _FakeOrchestrator)
    monkeypatch.setattr(gs, "GrootStagedRunner", _FakeRunner)
    monkeypatch.setattr(gi, "GrootCacheInterceptor", _FakeInterceptor)

    _FakeInterceptor.instances = []
    return types.SimpleNamespace(
        cfg=cfg, shared=shared_sentinel, record=record, tmp=tmp_path
    )


def _cache_args():
    return _args(cache_config="fake.yaml")


def test_cache_factory_builds_fresh_components_per_connection(cache_seams):
    """T1: mutable components are per-connection; only storage is shared."""
    policy = types.SimpleNamespace(model=object())
    factory, label = sgn._build_concurrent_factory(policy, _cache_args())
    assert "concurrent cache" in label

    # Startup work happened exactly once, against the shared facade.
    assert cache_seams.record["shared"] == [cache_seams.cfg]
    assert cache_seams.record["identity"] == [(cache_seams.shared, cache_seams.cfg)]

    p1 = factory(policy)
    p2 = factory(policy)

    # Both connections were built from the same shared storage sentinel.
    assert [c[1] for c in cache_seams.record["per_conn"]] == [
        cache_seams.shared, cache_seams.shared
    ]
    assert all(c[2] is True for c in cache_seams.record["per_conn"])  # quiet=True

    i1, i2 = _FakeInterceptor.instances
    assert i1 is not i2
    assert i1.orchestrator is not i2.orchestrator
    assert (
        i1.orchestrator.kwargs["key_builder"]
        is not i2.orchestrator.kwargs["key_builder"]
    )
    # The staged runner is rebuilt per connection around the shared model.
    assert i1.runner is not i2.runner
    assert i1.runner.model is policy.model and i2.runner.model is policy.model
    # Wrapper chain: lock wrapper -> adapter -> interceptor.
    assert p1._inner._policy is i1
    assert p2._inner._policy is i2


def test_cache_factory_gives_each_connection_its_own_csv_dir(cache_seams):
    """T8: per-task CSV names are only (task ordinal, second); concurrent
    connections must therefore write disjoint directories."""
    import os

    policy = types.SimpleNamespace(model=object())
    factory, _ = sgn._build_concurrent_factory(policy, _cache_args())
    factory(policy)
    factory(policy)

    i1, i2 = _FakeInterceptor.instances
    (d1,) = i1.timer.csv_dirs
    (d2,) = i2.timer.csv_dirs
    assert d1 != d2
    base = str(cache_seams.tmp / "csv")
    assert d1.startswith(base) and d2.startswith(base)
    assert os.path.isdir(d1) and os.path.isdir(d2)


def test_cache_factory_rejects_non_default_bundle(cache_seams):
    """T6 (cache side): same fail-fast contract as teacher-only."""
    policy = types.SimpleNamespace(model=object())
    factory, _ = sgn._build_concurrent_factory(policy, _cache_args())
    with pytest.raises(ValueError, match="default"):
        factory(policy, "n5_variant")
    assert _FakeInterceptor.instances == []  # nothing half-built


def test_lifecycle_calls_stay_on_their_own_connection(cache_seams):
    """T5: conn A's episode hooks never land on conn B's orchestrator."""
    policy = types.SimpleNamespace(model=object())
    factory, _ = sgn._build_concurrent_factory(policy, _cache_args())
    pa = factory(policy)
    pb = factory(policy)
    ia, ib = _FakeInterceptor.instances

    pa.on_episode_start(task="OpenCabinet", episode_id=0)
    pa.on_episode_end(True)
    pb.on_episode_start(task="CloseFridge", episode_id=7)

    assert [c[0] for c in ia.calls] == ["episode_start", "episode_end"]
    assert ia.calls[0][1]["task"] == "OpenCabinet"
    assert ia.calls[0][1]["episode_id"] == 0
    assert ia.calls[1][1] is True
    assert [c[0] for c in ib.calls] == ["episode_start"]
    assert ib.calls[0][1]["task"] == "CloseFridge"
    assert ib.calls[0][1]["episode_id"] == 7


# ---------------------------------------------------------------------------
# T2 — CLI guards / D-L preservation
# ---------------------------------------------------------------------------


def test_concurrent_conflicts_with_collect(monkeypatch, capsys):
    """T2a: the frozen collection topology cannot be served concurrently."""
    monkeypatch.setattr(
        sys, "argv",
        ["serve_groot_n15.py", "--concurrent", "--collect-hdf5", "/tmp/x"],
    )
    with pytest.raises(SystemExit):
        sgn.main()
    assert "--collect-hdf5" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv_tail",
    [["--collect-hdf5", "/tmp/x"], ["--cache-config", "/tmp/y.yaml"]],
    ids=["collect-alone", "cache-alone"],
)
def test_frozen_commands_pass_the_new_guards(monkeypatch, argv_tail):
    """T2b/T2c: the approved single-connection commands (no --concurrent)
    must sail through argparse validation unchanged — they may only stop at
    the gr00t import boundary, which is where the real server would begin."""
    if importlib.util.find_spec("gr00t") is not None:
        pytest.skip("gr00t installed here; the import boundary pin needs it absent")
    monkeypatch.setattr(sys, "argv", ["serve_groot_n15.py", *argv_tail])
    with pytest.raises(ModuleNotFoundError, match="gr00t"):
        sgn.main()


def test_metadata_concurrent_key_only_in_concurrent_branch():
    """T9 (source pin): the concurrent key must be set exactly once, inside
    the concurrent branch, so default-mode metadata bytes stay identical to
    the approved provenance captures."""
    source = inspect.getsource(sgn.main)
    assert source.count('metadata["concurrent"] = True') == 1
    concurrent_branch = source.split("if args.concurrent:")[1].split("else:")[0]
    assert 'metadata["concurrent"] = True' in concurrent_branch


# ---------------------------------------------------------------------------
# T10 — shared backend under cross-thread lifecycle + search pressure
# ---------------------------------------------------------------------------


def _unit(dim: int, index: int) -> torch.Tensor:
    v = torch.zeros(dim)
    v[index] = 1.0
    return v


def _make_backend():
    from openpi.cache.backends.in_memory_backend import InMemoryBackend
    from openpi.cache.storage_types import CacheEntry, CachePayload
    from openpi.cache.types import CheckpointID

    backend = InMemoryBackend({"robot_state": 32})
    for i in range(6):
        backend.insert(
            CacheEntry(
                id=f"e:{i}",
                checkpoint_id=CheckpointID.CP1,
                query_keys={"robot_state": _unit(32, i).cpu().float().contiguous()},
                payload=CachePayload(action_chunk=torch.randn(50, 32)),
                prev_ids=[f"e:{i - 1}"] if i > 0 else [],
                trajectory_id="traj",
                step_idx=i,
            )
        )
    return backend


def test_shared_backend_survives_lifecycle_vs_search_pressure():
    """T10 (G2 R1 rewrite): trajectory-depth search with a live `_score_memo`
    bucket under concurrent session churn — the documented GIL + disjoint-sid
    safety premise of plan D4, exercised on the real memo path.

    A depth-3 QuerySpec with `search_session_id` + `trajectory_query_ids`
    forces the memoized trajectory branch (the empty-history spec of the first
    version never touched `_score_memo`); a barrier forces true overlap and a
    minimum-search-count assertion forbids the degenerate zero-search run.
    """
    from openpi.cache.storage_types import QuerySpec
    from openpi.cache.types import CheckpointID

    backend = _make_backend()
    states = [_unit(32, i) for i in range(6)]
    history = [
        {"robot_state": states[5]},
        {"robot_state": states[4]},
        {"robot_state": states[3]},
    ]
    weights = [0.5, 0.3, 0.2]

    def _depth_spec(sid):
        return QuerySpec(
            query_keys={"robot_state": states[5]},
            top_k=3,
            checkpoint_id=CheckpointID.CP1,
            trajectory_history=history,
            trajectory_weights=weights,
            search_session_id=sid,
            trajectory_query_ids=[5, 4, 3],
        )

    # Single-threaded baseline for the pressure run to be compared against.
    backend.open_search_session("baseline")
    baseline_ids = [r.id for r in backend.search(_depth_spec("baseline"))]
    backend.close_search_session("baseline")
    assert baseline_ids  # depth search must return something to compare

    errors: list = []
    search_count = 0
    memo_ready = threading.Event()  # set once conn-a's memo bucket exists
    stop = threading.Event()
    churn_iters = 0
    searches_inside_churn = 0

    def lifecycle_churn():
        nonlocal churn_iters, searches_inside_churn
        try:
            # Handshake: only start churning once the search thread has a live
            # _score_memo bucket, so every churn cycle below truly races the
            # memoized search path.
            assert memo_ready.wait(timeout=10), "search thread never built its memo"
            sc_begin = search_count
            deadline = time.monotonic() + 10.0
            while True:
                sid = f"conn-b-{churn_iters}"
                backend.open_search_session(sid)
                backend.close_search_session(sid)
                churn_iters += 1
                # Real lifecycle events are sparse (episode boundaries); an
                # unpaced tight loop would just hog the GIL and starve the
                # search thread instead of racing it.
                time.sleep(0.001)
                progressed = search_count - sc_begin
                if churn_iters >= 200 and progressed >= 5:
                    searches_inside_churn = progressed
                    break
                if time.monotonic() > deadline:
                    raise AssertionError(
                        f"no overlap: {progressed} searches during "
                        f"{churn_iters} churn cycles in 10s"
                    )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            stop.set()

    def search_loop():
        nonlocal search_count
        sid = "conn-a"
        backend.open_search_session(sid)
        try:
            while not stop.is_set():
                results = backend.search(_depth_spec(sid))
                search_count += 1
                if [r.id for r in results] != baseline_ids:
                    errors.append(AssertionError(
                        f"result drift under pressure at search {search_count}"
                    ))
                    break
                if sid in backend._score_memo:
                    memo_ready.set()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            backend.close_search_session(sid)

    threads = [
        threading.Thread(target=lifecycle_churn),
        threading.Thread(target=search_loop),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert memo_ready.is_set(), "search loop never engaged the _score_memo bucket"
    # Overlap witnesses: searches counted strictly inside the churn window,
    # while the churn thread was actively cycling sessions.
    assert searches_inside_churn >= 5, (
        f"only {searches_inside_churn} searches inside the churn window"
    )
    assert churn_iters >= 200, f"churn degenerated: {churn_iters} cycles"
    # Cleanup: closing the session must clear its memo bucket, and the churned
    # sids must not leave residue either (cross-sid isolation).
    assert "conn-a" not in backend._score_memo
    assert not any(k.startswith("conn-b-") for k in backend._score_memo)


# ---------------------------------------------------------------------------
# G2 R1 item 1 — dynamic bundle ctrl surface must be loudly closed
# ---------------------------------------------------------------------------


class _ScriptedWebsocket:
    """Drives WebsocketPolicyServer._handler with a fixed ctrl script."""

    remote_address = ("127.0.0.1", 45678)

    def __init__(self, frames: list[bytes]) -> None:
        self._frames = list(frames)
        self.sent: list = []

    async def recv(self):
        import websockets.exceptions

        if self._frames:
            return self._frames.pop(0)
        raise websockets.exceptions.ConnectionClosedOK(None, None)

    async def send(self, data):
        import msgpack_numpy

        self.sent.append(msgpack_numpy.unpackb(data))

    async def close(self, code=None, reason=None):
        self.sent.append(("closed", code, reason))


def _run_handler(server, frames):
    import asyncio

    ws = _ScriptedWebsocket(frames)
    asyncio.run(server._handler(ws))
    return ws.sent


def _pack(obj):
    import msgpack_numpy

    return msgpack_numpy.Packer().pack(obj)


def test_dynamic_bundle_ctrl_rejected_when_disabled():
    """Hot-load surface on a fixed-configuration server: every
    load_cache_config and every non-default select_bundle must come back as
    an error ack with zero factory involvement, while select_bundle("default")
    stays the idempotent startup-configuration bind the in-repo runner
    protocol depends on (no silent 'success but startup stack' mismatch)."""
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    factory_calls: list = []

    def factory(policy, bundle_id="default"):
        factory_calls.append(bundle_id)
        return _RecordingInner()

    server = WebsocketPolicyServer(
        policy=_RecordingInner(),
        port=0,
        metadata={"concurrent": True},
        concurrent=True,
        connection_policy_factory=factory,
        allow_dynamic_bundles=False,
    )
    sent = _run_handler(server, [
        _pack({"__ctrl__": "load_cache_config", "yaml_content": "backend: {}"}),
        _pack({"__ctrl__": "select_bundle", "bundle_id": "other"}),
        _pack({"__ctrl__": "select_bundle", "bundle_id": "default"}),
        _pack({"__ctrl__": "select_bundle", "bundle_id": "default"}),
    ])

    assert sent[0] == {"concurrent": True}  # metadata handshake
    # Hot-load surface: rejected loudly.
    assert sent[1]["__ack__"] == "error" and "disabled" in sent[1]["msg"]
    assert sent[2]["__ack__"] == "error" and "disabled" in sent[2]["msg"]
    # The in-repo runner protocol (select_bundle("default") before the first
    # infer) must keep working: startup-config bind, idempotent on re-select.
    assert sent[3] == {"__ack__": "select_bundle", "bundle_id": "default"}
    assert sent[4] == {"__ack__": "select_bundle", "bundle_id": "default"}
    assert factory_calls == ["default"]  # bound once, re-select is a no-op


def test_dynamic_bundle_ctrl_keeps_legacy_flow_when_allowed():
    """Regression guard: with the default allow_dynamic_bundles=True the
    ctrl surface behaves exactly as before (legacy validation messages)."""
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    server = WebsocketPolicyServer(
        policy=_RecordingInner(),
        port=0,
        metadata={},
        concurrent=True,
        connection_policy_factory=lambda p, b="default": _RecordingInner(),
    )
    sent = _run_handler(server, [
        _pack({"__ctrl__": "load_cache_config"}),  # no yaml fields
        _pack({"__ctrl__": "select_bundle", "bundle_id": "never-loaded"}),
    ])

    assert sent[1]["__ack__"] == "error"
    assert "missing yaml_path" in sent[1]["msg"]
    assert sent[2]["__ack__"] == "error"
    assert "unknown bundle_id" in sent[2]["msg"]
