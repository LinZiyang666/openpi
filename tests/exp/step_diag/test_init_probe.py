"""Scientific integrity checks for the fixed-observation initialization probe."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from exp.step_diag.serve_init_probe import InitProbe, action_rmse, control_entry_ids


def test_metric_excludes_padding_and_unexecuted_horizon():
    a = np.zeros((16, 32), np.float32)
    b = a.copy()
    b[5:] = 99
    b[:, 12:] = 99
    assert action_rmse(a, b) == 0
    b[:5, :12] = 2
    assert action_rmse(a, b) == 2


def _entry(task, trajectory, value):
    return SimpleNamespace(payload=SimpleNamespace(task_key=task, action_chunk=torch.full((16, 32), value)),
                           trajectory_id=trajectory)


def test_control_does_not_reuse_winner_trajectory_or_another_task():
    entries = {"winner": _entry("A", "a1", 2.0), "neighbor": _entry("A", "a1", 3.0),
               "control": _entry("A", "a2", -1.0), "foreign": _entry("B", "b1", -2.0)}
    assert control_entry_ids(entries, "winner") == ["control"]


def test_probe_detects_known_endpoint_cancellation_and_preserves_rng(tmp_path):
    entries = {"winner": _entry("A", "a1", 2.0), "control": _entry("A", "a2", -1.0)}
    storage = SimpleNamespace(_backend=SimpleNamespace(_entries=entries),
                              fetch_entry=entries.__getitem__, fetch_payload=lambda i: entries[i].payload)
    episode = SimpleNamespace(task="A", task_uid="pilot:1", attempt=1, env_seed=2000000, init_idx=0)
    state = torch.get_rng_state().clone()

    def loop(start, n):
        torch.randn(5)  # Auxiliary randomness must not advance the teacher's stream.
        path = [start.clone()]
        for _ in range(n):
            start = start + (1 - start) / n
            path.append(start.clone())
        return path

    probe = InitProbe(tmp_path, "groot", (0,))
    probe.record(episode=episode, idx=0, storage=storage, winner_id="winner",
                 teacher=torch.ones(16, 32), run_loop=loop)
    assert torch.equal(state, torch.get_rng_state())
    row = json.loads((tmp_path / "probe.jsonl").read_text())
    matched = {(m["n"], m["control"]): m for m in row["metrics"]}
    assert matched[1, "random_same_task"]["final_ratio"] == 0
    assert matched[2, "random_same_task"]["first_ratio"] == 0.5
    assert matched[2, "random_same_task"]["final_ratio"] == 0.25
    assert torch.all(entries["winner"].payload.action_chunk == 2)
    with np.load(tmp_path / row["arrays"], allow_pickle=False) as arrays:
        assert arrays["retrieved_n1"].shape == (2, 16, 32)
        assert arrays["retrieved_n2"].shape == (3, 16, 32)
    from exp.step_diag.analysis.analyze_init_probe import analyze

    clients = tmp_path / "client"
    clients.mkdir()
    (clients / "journal_pilot.jsonl").write_text(json.dumps(
        {"task_uid": episode.task_uid, "attempt": 1, "status": "done", "accepted": True, "success": False}
    ) + "\n")
    summary = analyze(tmp_path, clients)
    assert summary["accepted_episodes"] == 1  # Ordinary failed episodes remain in the analysis.
    assert summary["observations"] == 1
    assert summary["max_euler_identity_error"] < 1e-6
    expected = [{"task_uid": uid, "task": episode.task, "env_seed": episode.env_seed,
                 "init_idx": episode.init_idx} for uid in (episode.task_uid, "pilot:2")]
    (clients / "launch_pilot.json").write_text(json.dumps({"expected": expected}))
    with pytest.raises(ValueError, match="accepted cohort does not equal requested cohort"):
        analyze(tmp_path, clients)
    partial = analyze(tmp_path, clients, allow_incomplete=True)
    assert partial["partial_analysis"] and not partial["launched_cohort_complete"]
    assert partial["requested_episodes_in_launched_cells"] == 2
    assert partial["missing_requested_episodes"] == ["pilot:2"]
    original = (tmp_path / "probe.jsonl").read_text()
    row["metrics"][0]["final_ratio"] = 37
    (tmp_path / "probe.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="stored ratio differs from raw arrays"):
        analyze(tmp_path, clients, allow_incomplete=True)
    (tmp_path / "probe.jsonl").write_text(original)
    with pytest.raises(FileExistsError):
        probe.record(episode=episode, idx=0, storage=storage, winner_id="winner",
                     teacher=torch.ones(16, 32), run_loop=loop)
