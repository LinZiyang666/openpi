"""The LIBERO server's RIT calibration factory.

Three properties this experiment's calibration rests on, none of which fails
loudly on its own:

*   every connection writes its **own** JSONL, because the shadow flushes a
    whole episode under one append and two connections would interleave at
    line granularity;
*   the ladder rungs are the k=8 schedule's 2- and 4-steps-remaining points,
    so a Pi0.5 ``start_t`` pasted in here would silently price the ladder
    backwards;
*   the action weights come from the library the recipe names, not from a
    side file that could drift from it.

The GPU-side seams (staged runner, orchestrator, library weights) are patched;
what is under test is the wiring, not GR00T.
"""

from __future__ import annotations

import pathlib
import sys
import threading
import types

import pytest

sys.modules.setdefault("gr00t", types.ModuleType("gr00t"))

import exp.libero_groot.serve_groot_libero as sgl  # noqa: E402


class _FakeShadow:
    instances: list["_FakeShadow"] = []

    def __init__(self, policy, runner, **kwargs):
        self.policy = policy
        self.runner = runner
        self.kwargs = kwargs
        _FakeShadow.instances.append(self)

    def get_action(self, observations):  # pragma: no cover - never called here
        raise AssertionError("the wiring test does not run inference")


class _FakeOrchestrator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeRunner:
    def __init__(self, model, *, timer=None):
        self.model = model
        self.timer = timer


@pytest.fixture
def shadow_seams(monkeypatch, tmp_path):
    import openpi.cache.config as cache_config
    import openpi.cache.groot.staged as gs
    import openpi.cache.orchestrator as orch

    import exp.robocasa365.rit_shadow as rs

    preload = str(tmp_path / "libero_spatial_w13_S3.pkl")
    cfg = types.SimpleNamespace(
        backend=types.SimpleNamespace(
            in_memory=types.SimpleNamespace(preload_path=preload)
        )
    )

    def fake_per_conn(config, shared_storage, *, yaml_id=None, quiet=False):
        return {
            "timer": object(),
            "storage": object(),
            "key_builder": object(),
            "gates": [],
            "judges": [],
            "search_strategies": [],
            "write_policy": object(),
            "offline_writers": [],
            "library_stats": object(),
        }

    weights_calls: list[str] = []

    monkeypatch.setattr(cache_config, "build_per_connection_components", fake_per_conn)
    monkeypatch.setattr(orch, "CacheOrchestrator", _FakeOrchestrator)
    monkeypatch.setattr(gs, "GrootStagedRunner", _FakeRunner)
    monkeypatch.setattr(rs, "GrootRitShadow", _FakeShadow)
    monkeypatch.setattr(
        rs,
        "library_action_weights",
        lambda path: (weights_calls.append(path), ([1.0, 2.0], [True, False]))[1],
    )
    _FakeShadow.instances = []
    return types.SimpleNamespace(
        cfg=cfg, preload=preload, tmp=tmp_path, weights_calls=weights_calls
    )


def _args(tmp_path, **over):
    base = dict(
        rit_shadow_out=str(tmp_path / "shadow" / "sp.jsonl"),
        rit_warm_ts=sgl.DEFAULT_RIT_WARM_TS,
        rit_h_exec=sgl.DEFAULT_RIT_H_EXEC,
        experiment="groot_libero_spatial",
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def _policy():
    return types.SimpleNamespace(model=object())


def test_each_connection_writes_its_own_file(shadow_seams):
    """Two connections must not share one append-mode JSONL."""
    args = _args(shadow_seams.tmp)
    factory, label = sgl._build_shadow_factory(
        args, shadow_seams.cfg, object(), threading.Lock()
    )
    assert "rit-shadow" in label

    factory(_policy())
    factory(_policy())

    paths = [s.kwargs["out_path"] for s in _FakeShadow.instances]
    assert len(paths) == 2
    assert paths[0] != paths[1]
    for path in paths:
        p = pathlib.Path(path)
        assert p.parent == shadow_seams.tmp / "shadow"
        assert p.name.startswith("sp.conn_")
        assert p.suffix == ".jsonl"
    # The directory is created eagerly: the first flush happens inside an
    # episode-end hook, where a missing parent would surface as a lost episode.
    assert (shadow_seams.tmp / "shadow").is_dir()


def test_default_rungs_are_two_and_four_steps_remaining(shadow_seams):
    """k=8 ascends: t=0.75 leaves 2 steps, t=0.5 leaves 4."""
    from openpi.cache.types import groot_n15_schedule

    factory, _ = sgl._build_shadow_factory(
        _args(shadow_seams.tmp), shadow_seams.cfg, object(), threading.Lock()
    )
    factory(_policy())
    warm_ts = _FakeShadow.instances[0].kwargs["warm_ts"]
    assert warm_ts == [0.75, 0.5]

    schedule = groot_n15_schedule(8)
    assert [schedule.remaining_steps(t) for t in warm_ts] == [2, 4]


def test_h_exec_and_weights_come_from_the_deployed_library(shadow_seams):
    factory, _ = sgl._build_shadow_factory(
        _args(shadow_seams.tmp), shadow_seams.cfg, object(), threading.Lock()
    )
    factory(_policy())
    shadow = _FakeShadow.instances[0]
    assert shadow.kwargs["h_exec"] == 5
    # Read from the recipe's own preload_path, so weights cannot describe a
    # library other than the one that answers the queries.
    assert shadow_seams.weights_calls == [shadow_seams.preload]
    assert shadow.kwargs["w"].tolist() == [1.0, 2.0]
    assert shadow.kwargs["active_mask"].tolist() == [True, False]


def test_empty_warm_ts_is_refused(shadow_seams):
    with pytest.raises(SystemExit):
        sgl._build_shadow_factory(
            _args(shadow_seams.tmp, rit_warm_ts=" , "),
            shadow_seams.cfg,
            object(),
            threading.Lock(),
        )
