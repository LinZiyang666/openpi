"""Client-side episode framing and hit logging.

The `__ctrl__` frames are what give the server's cache an episode boundary at
all. The one that matters most is the closing frame on the failure path: the
server closes its search session inside that handler, so an episode that dies
mid-way without it leaves backend state open for the next one.
"""

from __future__ import annotations

import io
import json

import pytest

from exp.robocasa365.groot_rollout_client import _record_hit, _send_ctrl


class _RecordingClient:
    def __init__(self, response=None) -> None:
        self.frames: list[dict] = []
        self._response = response or {"actions": {}}

    def infer(self, obs):
        self.frames.append(obs)
        return self._response


def test_episode_start_carries_the_task_key():
    """Without it every library entry gets an empty task_key."""
    client = _RecordingClient()
    _send_ctrl(client, "episode_start", task="OpenCabinet", episode_id=4)
    assert client.frames == [
        {"__ctrl__": "episode_start", "__task__": "OpenCabinet", "__episode_id__": 4}
    ]


def test_episode_end_carries_the_outcome():
    client = _RecordingClient()
    _send_ctrl(client, "episode_end", success=False)
    assert client.frames == [{"__ctrl__": "episode_end", "__success__": False}]


def test_ctrl_fields_are_dunder_wrapped():
    """So the server cannot confuse them with modality keys on the same dict."""
    client = _RecordingClient()
    _send_ctrl(client, "episode_start", task="t", episode_id=0)
    for key in client.frames[0]:
        assert key.startswith("__") and key.endswith("__")


def test_hit_rows_are_written_when_the_server_reports_a_verdict():
    log = io.StringIO()
    response = {
        "actions": {},
        "__hit_meta__": {
            "hit_type": "FULL_HIT",
            "winner_id": "traj:3",
            "cp1_score": 0.87,
            "searched": True,
        },
    }
    _record_hit(log, response, "OpenDrawer", episode=1, step=12)
    row = json.loads(log.getvalue())
    assert row == {
        "task": "OpenDrawer",
        "episode": 1,
        "step_idx": 12,
        "hit_type": "FULL_HIT",
        "winner_id": "traj:3",
        "cp1_score": 0.87,
        "searched": True,
    }


def test_a_cache_free_server_costs_nothing():
    log = io.StringIO()
    _record_hit(log, {"actions": {}}, "t", 0, 0)
    assert log.getvalue() == ""


def test_no_log_configured_is_a_no_op():
    _record_hit(None, {"actions": {}, "__hit_meta__": {"hit_type": "MISS"}}, "t", 0, 0)


def test_partial_meta_does_not_raise():
    """Every field is read with .get so an older server cannot crash the run."""
    log = io.StringIO()
    _record_hit(log, {"__hit_meta__": {"hit_type": "MISS"}}, "t", 0, 3)
    row = json.loads(log.getvalue())
    assert row["hit_type"] == "MISS"
    assert row["winner_id"] is None


def test_episode_end_is_sent_even_when_the_episode_raises():
    """The failure path, written the way run_one writes it."""
    client = _RecordingClient()
    with pytest.raises(ValueError):
        _send_ctrl(client, "episode_start", task="t", episode_id=0)
        try:
            raise ValueError("env blew up mid-episode")
        finally:
            _send_ctrl(client, "episode_end", success=False)

    ctrls = [frame["__ctrl__"] for frame in client.frames]
    assert ctrls == ["episode_start", "episode_end"]
