"""Tests for ``CollectionPolicy`` lifecycle + prompt capture (plan §3.3, §22.4).

These tests bypass ``CollectionPolicy.__init__`` (which walks the wrapper
chain looking for a real PI0 model) by constructing the instance directly
with ``object.__new__`` and assigning only the fields the tested methods
read. That keeps the tests fast and decoupled from the model stack.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from openpi.collect.collection_policy import CollectionPolicy


def _make_policy() -> tuple[CollectionPolicy, MagicMock, MagicMock]:
    """Build a stripped ``CollectionPolicy`` with mock ``_policy`` and collector.

    Only ``on_episode_start`` / ``on_episode_end`` / ``set_episode_attr`` /
    ``infer`` paths are exercised in this suite, so leaving the model hooks
    unset is safe.
    """
    cp = object.__new__(CollectionPolicy)
    cp._policy = MagicMock(
        spec=["on_episode_start", "on_episode_end", "on_task_begin", "on_task_end", "infer"]
    )
    cp._collector = MagicMock(
        spec=["on_episode_start", "on_episode_end", "set_episode_attr", "has_pending_data"]
    )
    cp._collecting = False
    cp._prompt_captured = False
    return cp, cp._policy, cp._collector


# ---------------------------------------------------------------------------
# on_episode_start / end: kwarg forwarding to BOTH collector and inner policy
# ---------------------------------------------------------------------------


def test_on_episode_start_forwards_to_collector_and_inner_policy() -> None:
    cp, inner, collector = _make_policy()

    cp.on_episode_start("exp", "task", 3, episode_name="task_0/ep_0")

    collector.on_episode_start.assert_called_once_with(
        experiment="exp", task="task", episode_id=3, episode_name="task_0/ep_0"
    )
    inner.on_episode_start.assert_called_once_with(
        experiment="exp", task="task", episode_id=3, episode_name="task_0/ep_0"
    )
    assert cp._collecting is True
    assert cp._prompt_captured is False  # reset, not yet captured


def test_on_episode_start_resets_prompt_captured_flag() -> None:
    cp, _, _ = _make_policy()
    cp._prompt_captured = True  # simulate leftover state from prior episode

    cp.on_episode_start("exp", "task", 1, episode_name="ep_1")

    assert cp._prompt_captured is False


def test_on_episode_end_forwards_success_kwarg_and_clears_collecting() -> None:
    cp, inner, collector = _make_policy()
    cp._collecting = True

    cp.on_episode_end(success=True)

    collector.on_episode_end.assert_called_once_with(success=True)
    inner.on_episode_end.assert_called_once_with(success=True)
    assert cp._collecting is False


def test_on_episode_start_default_episode_name_is_empty_string() -> None:
    cp, inner, collector = _make_policy()
    cp.on_episode_start("exp", "task", 3)
    collector.on_episode_start.assert_called_once_with(
        experiment="exp", task="task", episode_id=3, episode_name=""
    )
    inner.on_episode_start.assert_called_once_with(
        experiment="exp", task="task", episode_id=3, episode_name=""
    )


def test_lifecycle_skipped_when_inner_has_no_hook() -> None:
    """Mirrors the ``hasattr`` guard: an inner policy without lifecycle
    methods must not crash ``CollectionPolicy.on_episode_start/end``."""
    cp = object.__new__(CollectionPolicy)
    cp._policy = MagicMock(spec=["infer"])  # no lifecycle methods
    cp._collector = MagicMock(
        spec=["on_episode_start", "on_episode_end", "set_episode_attr", "has_pending_data"]
    )
    cp._collecting = False
    cp._prompt_captured = False

    cp.on_episode_start("exp", "task", 0, episode_name="ep_0")
    cp.on_episode_end(success=False)
    # No exceptions, and collector still got both signals.
    assert cp._collector.on_episode_start.called
    assert cp._collector.on_episode_end.called


# ---------------------------------------------------------------------------
# prompt capture: once per episode, re-armed by on_episode_start
# ---------------------------------------------------------------------------


def _prompt_gate(cp: CollectionPolicy, obs: dict) -> None:
    """Invoke only the prompt-capture block in ``infer`` without touching the
    hook-based model path (which would need a real PI0 model)."""
    if not cp._prompt_captured:
        raw_prompt = obs.get("prompt", "")
        cp._collector.set_episode_attr("prompt", str(raw_prompt))
        cp._prompt_captured = True


def test_prompt_captured_once_per_episode() -> None:
    cp, _, collector = _make_policy()
    cp._collecting = True

    _prompt_gate(cp, {"prompt": "pick apple"})
    _prompt_gate(cp, {"prompt": "pick apple"})
    _prompt_gate(cp, {"prompt": "pick apple"})

    collector.set_episode_attr.assert_called_once_with("prompt", "pick apple")


def test_prompt_recaptured_after_next_episode_start() -> None:
    cp, _, collector = _make_policy()
    cp._collecting = True

    _prompt_gate(cp, {"prompt": "ep A prompt"})
    cp.on_episode_start("exp", "task", 2, episode_name="ep_B")
    cp._collecting = True  # on_episode_start sets this via collector; simulate.
    _prompt_gate(cp, {"prompt": "ep B prompt"})

    # Two distinct set_episode_attr calls across the two episodes.
    prompts = [call.args[1] for call in collector.set_episode_attr.call_args_list]
    assert prompts == ["ep A prompt", "ep B prompt"]


def test_prompt_capture_handles_missing_prompt_key() -> None:
    cp, _, collector = _make_policy()
    cp._collecting = True
    _prompt_gate(cp, {})  # no "prompt" key
    collector.set_episode_attr.assert_called_once_with("prompt", "")
