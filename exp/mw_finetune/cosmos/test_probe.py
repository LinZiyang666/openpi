"""Regression checks for the isolated render/size/horizon probe shims."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from exp.metaworld.env import render_image
from exp.mw_finetune.cosmos.probe_evaluate import RawView, run_with_probe_horizon
from exp.mw_finetune.cosmos.probe_server import probe_canonical
from exp.mw_finetune.cosmos.reconstruction_probe import CONFIGS, evaluation_command, summarize


def test_native_raw_view_survives_existing_kernel_rotation():
    raw = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    proxy = RawView(SimpleNamespace(render=lambda: raw))
    np.testing.assert_array_equal(render_image(proxy), raw)
    assert render_image(proxy).flags.c_contiguous


@pytest.mark.parametrize("size", [256, 480])
def test_probe_server_keeps_seed_and_state_while_allowing_native_size(size):
    state = np.asarray([.1, .6, .2, .5], np.float32)
    obs = {"observation/image": np.zeros((size, size, 3), np.uint8), "observation/state": state, "prompt": "test"}
    canonical = probe_canonical(obs, task_name="assembly-v3", seed=42, decision_index=7)
    assert canonical["seed"] == 42 and canonical["decision_index"] == 7
    np.testing.assert_array_equal(canonical["state"], state)
    assert canonical["image"].shape == (size, size, 3)


def test_evaluator_command_selects_only_flowdagger_override(tmp_path):
    for source in CONFIGS:
        config = {**source, "port": 1234}
        command = evaluation_command(config, "assembly", tmp_path, "python", 5)
        if config["corner3"]:
            assert "exp.mw_finetune.cosmos.probe_evaluate" in command
            assert command[command.index("--max-steps") + 1] == "300"
        else:
            assert "exp.mw_finetune.evaluate" in command and "--max-steps" not in command
        assert command[command.index("--trials") + 1] == "10"
        reach = evaluation_command(config, "reach", tmp_path, "python", 5)
        assert reach[reach.index("--trials") + 1] == "5"


def test_160_step_score_is_derived_from_same_300_step_rollouts(tmp_path):
    root = tmp_path / CONFIGS[2]["name"] / "assembly"
    root.mkdir(parents=True)
    rows = [{"idx": idx, "success": success, "n_steps": steps, "error": None, "duration_s": 1,
             "per_step_rows": [{"n_decisions": 4, "infer_s": .4}]}
            for idx, success, steps in [(0, True, 150), (1, True, 200), (2, False, 300)]]
    (root / "episodes.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    report = summarize(tmp_path, {})["configs"][2]["tasks"]["assembly-v3"]
    assert report["completed"] == 3 and report["successes"] == 2 and report["first_160_successes"] == 1


def test_horizon_override_replaces_explicit_episode_runner_keyword(monkeypatch):
    received = {}
    def kernel(*args, **kwargs):
        received.update(kwargs)
        return "outcome"
    monkeypatch.setattr("exp.mw_finetune.cosmos.probe_evaluate.E.run_episode", kernel)
    assert run_with_probe_horizon(None, None, "prompt", probe_max_steps=300, max_steps=160, replan_steps=5) == "outcome"
    assert received["max_steps"] == 300 and received["replan_steps"] == 5
