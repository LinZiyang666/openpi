"""Lifecycle passthrough tests for ``PolicyRecorder`` (plan §20.R2).

``PolicyRecorder`` sits between ``CollectionPolicy`` and
``InferenceInterceptor`` in the production wrapper chain. Before §20.R2 it
had no ``on_episode_*`` methods, so the ``CollectionPolicy`` → inner-policy
forwarding added in Layer B-3 would still stop at this wrapper. These tests
lock the new contract: four lifecycle methods, passthrough semantics, and
``hasattr`` guarding when the inner policy does not implement the hook.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock

from openpi.policies.policy import PolicyRecorder


def _make_recorder(tmp_path: Path, *, inner_spec: list[str]) -> tuple[PolicyRecorder, MagicMock]:
    inner = MagicMock(spec=inner_spec)
    rec = PolicyRecorder(inner, str(tmp_path / "rec"))
    return rec, inner


def test_on_episode_start_forwards_kwargs(tmp_path: Path) -> None:
    rec, inner = _make_recorder(
        tmp_path, inner_spec=["on_episode_start", "on_episode_end", "infer"]
    )
    rec.on_episode_start(
        experiment="exp", task="task", episode_id=1, episode_name="task_0/ep_0"
    )
    # extra_metadata defaults to None when caller omits it (5-field contract).
    inner.on_episode_start.assert_called_once_with(
        experiment="exp", task="task", episode_id=1, episode_name="task_0/ep_0",
        extra_metadata=None,
    )


def test_on_episode_start_forwards_extra_metadata(tmp_path: Path) -> None:
    """Non-empty extra_metadata must reach the inner policy unchanged."""
    rec, inner = _make_recorder(
        tmp_path, inner_spec=["on_episode_start", "on_episode_end", "infer"]
    )
    rec.on_episode_start(
        experiment="exp", task="task", episode_id=1, episode_name="task_0/ep_0",
        extra_metadata={"task_id": 3, "orig_init_state_idx": 7},
    )
    inner.on_episode_start.assert_called_once_with(
        experiment="exp", task="task", episode_id=1, episode_name="task_0/ep_0",
        extra_metadata={"task_id": 3, "orig_init_state_idx": 7},
    )


def test_on_episode_end_forwards_success_kwarg(tmp_path: Path) -> None:
    rec, inner = _make_recorder(
        tmp_path, inner_spec=["on_episode_start", "on_episode_end", "infer"]
    )
    rec.on_episode_end(success=True)
    inner.on_episode_end.assert_called_once_with(success=True)


def test_on_task_begin_end_forwarded(tmp_path: Path) -> None:
    rec, inner = _make_recorder(
        tmp_path, inner_spec=["on_task_begin", "on_task_end", "infer"]
    )
    rec.on_task_begin()
    rec.on_task_end()
    inner.on_task_begin.assert_called_once_with()
    inner.on_task_end.assert_called_once_with()


def test_lifecycle_methods_noop_when_inner_lacks_hooks(tmp_path: Path) -> None:
    """Forwarding must be ``hasattr``-guarded — an inner plain ``Policy``
    does not implement the hooks, and raising here would break the cache-
    off path that other tests still rely on."""
    rec, inner = _make_recorder(tmp_path, inner_spec=["infer"])

    # Any of these would raise AttributeError if the guard was missing.
    rec.on_episode_start(experiment="e", task="t", episode_id=0, episode_name="")
    rec.on_episode_end(success=False)
    rec.on_task_begin()
    rec.on_task_end()

    # None of the mocked surface should have been touched.
    inner.infer.assert_not_called()


def test_lifecycle_signatures_are_explicit_no_varargs() -> None:
    """Plan §21.S1.2 / §22.3: lifecycle forwarders must be *isomorphic* to the
    rest of the chain (explicit ``experiment`` / ``task`` / ``episode_id`` /
    ``episode_name`` / ``success`` parameters). A ``*args, **kwargs`` signature
    would silently accept positional calls and re-introduce the keyword-vs-
    positional divergence that the S1 revision removed, so this test locks the
    signatures against regressing to varargs.
    """
    _VAR = {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}

    # on_episode_start: explicit 5-arg signature + no varargs. The
    # ``extra_metadata`` kwarg was added for verdict-factor calibration
    # logging (DumpingJudge identity injection) but the explicit-only
    # contract still holds — no varargs allowed.
    sig = inspect.signature(PolicyRecorder.on_episode_start)
    names = [p.name for p in sig.parameters.values() if p.name != "self"]
    kinds = {p.kind for p in sig.parameters.values() if p.name != "self"}
    assert names == [
        "experiment", "task", "episode_id", "episode_name", "extra_metadata",
    ], names
    assert not (kinds & _VAR), f"on_episode_start must not use *args/**kwargs, got {kinds}"

    # on_episode_end: explicit ``success`` only.
    sig = inspect.signature(PolicyRecorder.on_episode_end)
    names = [p.name for p in sig.parameters.values() if p.name != "self"]
    kinds = {p.kind for p in sig.parameters.values() if p.name != "self"}
    assert names == ["success"], names
    assert not (kinds & _VAR), f"on_episode_end must not use *args/**kwargs, got {kinds}"

    # on_task_begin / on_task_end: no extra parameters, and no varargs.
    for name in ("on_task_begin", "on_task_end"):
        sig = inspect.signature(getattr(PolicyRecorder, name))
        extra = [p for p in sig.parameters.values() if p.name != "self"]
        assert extra == [], f"{name} must take no extra args, got {extra}"


def test_infer_still_records_after_lifecycle_addition(tmp_path: Path) -> None:
    """Regression: adding lifecycle methods must not have disturbed the
    original ``infer`` record-to-disk path."""
    rec, inner = _make_recorder(
        tmp_path, inner_spec=["on_episode_start", "on_episode_end", "infer"]
    )
    inner.infer.return_value = {"actions": [[0.0]]}

    rec.infer({"state": [1.0]})

    # Recorder writes ``step_0.npy`` relative to its record_dir.
    files = list((tmp_path / "rec").iterdir())
    assert any(f.name.startswith("step_0") for f in files)
