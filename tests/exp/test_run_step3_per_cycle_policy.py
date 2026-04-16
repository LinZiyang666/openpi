"""Tests for the Step 3 per-cycle trajectory-deviation runner.

These tests intentionally avoid LIBERO, websocket servers, and GPU inference.
They exercise the runner's pure control logic with fake env/client objects so
G2 can catch index-space, burst, and success-accounting regressions quickly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from exp.common._unit_key import Step3PerCycleKey
from exp.trajectory_deviation import run_step3_per_cycle_policy as step3


class FakeEnv:
    """Small LIBERO-like env stub for runner unit tests."""

    def __init__(
        self,
        *,
        done_after_steps: int | None = None,
        info_success_on_done: bool = False,
    ) -> None:
        self.done_after_steps = done_after_steps
        self.info_success_on_done = info_success_on_done
        self.reset_calls = 0
        self.init_state = None
        self.step_calls: list[Any] = []

    def reset(self) -> None:
        self.reset_calls += 1
        self.step_calls.clear()

    def set_init_state(self, initial_state: Any) -> dict:
        self.init_state = initial_state
        return {"source": "init", "step": 0}

    def step(self, action: Any) -> tuple[dict, float, bool, dict]:
        self.step_calls.append(action)
        done = (
            self.done_after_steps is not None
            and len(self.step_calls) >= self.done_after_steps
        )
        info = {"success": True} if done and self.info_success_on_done else {}
        return {"source": "step", "step": len(self.step_calls)}, 0.0, done, info


class FakeClient:
    """WebsocketClientPolicy-like stub that records gate decisions."""

    def __init__(self, *, action_dim: int = 7, chunk_len: int = 1) -> None:
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.decisions: list[str] = []
        self.episode_starts: list[dict] = []
        self.episode_ends: list[bool] = []

    def episode_start(self, **kwargs: Any) -> dict:
        self.episode_starts.append(dict(kwargs))
        return {"__ack__": "episode_start"}

    def infer(self, obs: dict) -> dict:
        self.decisions.append(obs["__gate_decision__"])
        return {
            "actions": np.zeros((self.chunk_len, self.action_dim), dtype=np.float32)
        }

    def episode_end(self, success: bool = False) -> dict:
        self.episode_ends.append(bool(success))
        return {"__ack__": "episode_end"}


def _runner(tmp_path: Path, *, replan_steps: int = 1, num_steps_wait: int = 0):
    cfg = step3._RunnerConfig(
        cfg="clip_w7_d4",
        host="localhost",
        port=8001,
        task_suite_name="libero_spatial",
        init_states_dir=tmp_path,
        deviate_scores={"task_0/episode_0": np.asarray([0.0])},
        tau_grid=[5],
        n_grid=[1],
        max_cycles_safety=0,
        replan_steps=replan_steps,
        num_steps_wait=num_steps_wait,
        max_env_steps=10,
        resize_size=224,
        results_jsonl=tmp_path / "results.jsonl",
        experiment_tag="test_step3",
    )
    return step3.Step3PerCycleRunner(
        state_path=tmp_path / "state.json",
        runner_cfg=cfg,
        episodes=["task_0/episode_0"],
    )


def _patch_obs_converter(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    seen_env_obs: list[dict] = []

    def fake_obs_env_to_policy(env_obs: dict, task_description: str, *, resize: int):
        seen_env_obs.append(env_obs)
        return {
            "observation/image": np.zeros((1, 1, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((1, 1, 3), dtype=np.uint8),
            "observation/state": np.zeros(8, dtype=np.float64),
            "prompt": task_description,
        }

    monkeypatch.setattr(step3, "_obs_env_to_policy", fake_obs_env_to_policy)
    return seen_env_obs


def test_parse_ep_key_accepts_subset_episode_key() -> None:
    assert step3._parse_ep_key("task_12/episode_34") == (12, 34)


def test_parse_ep_key_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="task_X/episode_Y"):
        step3._parse_ep_key("task_12_episode_34")


def test_load_deviate_scores_rejects_missing_score_field(tmp_path: Path) -> None:
    path = tmp_path / "scores.json"
    path.write_text(json.dumps({"task_0/episode_0": {"cache_l2": [1.0]}}))

    with pytest.raises(ValueError, match="missing 'deviate_score'"):
        step3._load_deviate_scores(path)


def test_main_uses_deviate_score_keys_as_authoritative_episode_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    score_path = tmp_path / "deviate_score_clip_w7_d4.json"
    score_path.write_text(
        json.dumps(
            {
                "task_0/episode_8": {"deviate_score": [0.0, 6.0]},
                "task_0/episode_13": {"deviate_score": [7.0]},
            }
        )
    )
    yaml_path = tmp_path / "step3.yaml"
    yaml_path.write_text("enabled: true\n")

    created = []

    class CapturingRunner(step3.Step3PerCycleRunner):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            created.append(self)

        def parallel_run(self, *, num_workers: int, resume: bool = False, unit_filter=None) -> None:
            return None

        def close_all(self) -> None:
            return None

    monkeypatch.setattr(step3, "Step3PerCycleRunner", CapturingRunner)
    monkeypatch.setattr(step3, "send_load_cache_config", lambda *a, **kw: None)

    step3.main(
        [
            "--cfg",
            "clip_w7_d4",
            "--host",
            "localhost",
            "--port",
            "8001",
            "--yaml",
            str(yaml_path),
            "--deviate-score-json",
            str(score_path),
            "--init-states-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )

    assert len(created) == 1
    assert created[0].episodes == ["task_0/episode_13", "task_0/episode_8"]


def test_main_rejects_empty_deviate_score_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    score_path = tmp_path / "deviate_score_clip_w7_d4.json"
    score_path.write_text(json.dumps({}))
    yaml_path = tmp_path / "step3.yaml"
    yaml_path.write_text("enabled: true\n")

    monkeypatch.setattr(step3, "send_load_cache_config", lambda *a, **kw: None)

    with pytest.raises(SystemExit, match="has no episodes"):
        step3.main(
            [
                "--cfg",
                "clip_w7_d4",
                "--host",
                "localhost",
                "--port",
                "8001",
                "--yaml",
                str(yaml_path),
                "--deviate-score-json",
                str(score_path),
                "--init-states-dir",
                str(tmp_path),
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )


def test_run_one_unit_uses_init_obs_when_no_warmup_and_done_counts_as_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_env_obs = _patch_obs_converter(monkeypatch)
    runner = _runner(tmp_path, replan_steps=1, num_steps_wait=0)
    env = FakeEnv(done_after_steps=1, info_success_on_done=False)
    client = FakeClient(chunk_len=1)
    key = Step3PerCycleKey(cfg="clip_w7_d4", ep="task_0/episode_0", tau=5, n=1)

    record = runner._run_one_unit(
        env=env,
        client=client,
        key=key,
        initial_state=np.zeros(2),
        task_description="task language",
        is_deviate=np.asarray([False]),
        t_gt_cycles=1,
        subset_idx=0,
        task_id=0,
    )

    assert seen_env_obs[0]["source"] == "init"
    assert record["success"] is True
    assert client.episode_ends == [True]
    assert client.decisions == ["search"]


def test_run_one_unit_injects_one_gate_decision_per_inference_cycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_obs_converter(monkeypatch)
    runner = _runner(tmp_path, replan_steps=2, num_steps_wait=0)
    env = FakeEnv(done_after_steps=6, info_success_on_done=True)
    client = FakeClient(chunk_len=2)
    key = Step3PerCycleKey(cfg="clip_w7_d4", ep="task_0/episode_0", tau=5, n=2)

    record = runner._run_one_unit(
        env=env,
        client=client,
        key=key,
        initial_state=np.zeros(2),
        task_description="task language",
        is_deviate=np.asarray([True, True, False]),
        t_gt_cycles=3,
        subset_idx=0,
        task_id=0,
    )

    assert client.decisions == ["skip", "skip", "search"]
    assert len(env.step_calls) == 6
    assert record["total_cycles"] == 3
    assert record["num_inference_cycles"] == 2
    assert record["burst_count"] == 1
    assert record["success"] is True


def test_run_one_unit_burst_does_not_extend_inside_existing_burst(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_obs_converter(monkeypatch)
    runner = _runner(tmp_path, replan_steps=1, num_steps_wait=0)
    env = FakeEnv(done_after_steps=5, info_success_on_done=True)
    client = FakeClient(chunk_len=1)
    key = Step3PerCycleKey(cfg="clip_w7_d4", ep="task_0/episode_0", tau=5, n=2)

    record = runner._run_one_unit(
        env=env,
        client=client,
        key=key,
        initial_state=np.zeros(2),
        task_description="task language",
        is_deviate=np.asarray([False, True, True, False, True]),
        t_gt_cycles=5,
        subset_idx=0,
        task_id=0,
    )

    assert client.decisions == ["search", "skip", "skip", "search", "skip"]
    assert record["num_inference_cycles"] == 3
    assert record["burst_count"] == 2
    assert record["success"] is True


def test_run_one_unit_max_cycles_guard_records_failure_on_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_obs_converter(monkeypatch)
    runner = _runner(tmp_path, replan_steps=2, num_steps_wait=0)
    runner.runner_cfg.max_env_steps = 4
    runner.runner_cfg.max_cycles_safety = 0
    env = FakeEnv(done_after_steps=None)
    client = FakeClient(chunk_len=2)
    key = Step3PerCycleKey(cfg="clip_w7_d4", ep="task_0/episode_0", tau=5, n=1)

    record = runner._run_one_unit(
        env=env,
        client=client,
        key=key,
        initial_state=np.zeros(2),
        task_description="task language",
        is_deviate=np.asarray([False, False, False]),
        t_gt_cycles=3,
        subset_idx=0,
        task_id=0,
    )

    assert record["total_cycles"] == 2
    assert record["success"] is False
    assert client.episode_ends == [False]
    assert client.decisions == ["search", "search"]


def test_build_units_cross_product_uses_subset_episode_keys(tmp_path: Path) -> None:
    cfg = step3._RunnerConfig(
        cfg="clip_w7_d4",
        host="localhost",
        port=8001,
        task_suite_name="libero_spatial",
        init_states_dir=tmp_path,
        deviate_scores={},
        tau_grid=[3, 5],
        n_grid=[1, 2],
        max_cycles_safety=0,
        replan_steps=1,
        num_steps_wait=0,
        max_env_steps=10,
        resize_size=224,
        results_jsonl=tmp_path / "results.jsonl",
        experiment_tag="test_step3",
    )
    runner = step3.Step3PerCycleRunner(
        state_path=tmp_path / "state.json",
        runner_cfg=cfg,
        episodes=["task_0/episode_8"],
    )

    keys = [unit.unit_key for unit in runner.build_units()]

    assert keys == [
        "clip_w7_d4:task_0/episode_8:tau3:n1",
        "clip_w7_d4:task_0/episode_8:tau3:n2",
        "clip_w7_d4:task_0/episode_8:tau5:n1",
        "clip_w7_d4:task_0/episode_8:tau5:n2",
    ]
