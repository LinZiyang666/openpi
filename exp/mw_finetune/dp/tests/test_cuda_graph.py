import numpy as np
import pytest
import torch

from exp.mw_finetune.dp.cuda_graph import DenoiserGraph, check_observations, require_bit_exact


def test_self_check_requires_exact_actions_and_three_distinct_inputs():
    stats = {"state": {"q01": [0] * 4, "q99": [1] * 4}}
    observations = list(check_observations("metaworld", stats))
    calls = []

    def inference(obs):
        calls.append((obs["seed"], obs["prompt"]))
        return {"actions": np.full((16, 4), obs["seed"], np.float32)}

    report = require_bit_exact(observations, inference, inference)
    assert report["passed"] and len(report["observations"]) == 3
    assert len(calls) == 6 and len(set(calls)) == 3
    with pytest.raises(RuntimeError, match="three distinct"):
        require_bit_exact(observations[:2], inference, inference)
    with pytest.raises(RuntimeError, match="three distinct"):
        require_bit_exact([observations[0]] * 3, inference, inference)

    def almost_equal(obs):
        actions = inference(obs)["actions"]
        actions[0, 0] = np.nextafter(actions[0, 0], np.float32(np.inf))
        return {"actions": actions}

    with pytest.raises(RuntimeError, match="refusing to serve"):
        require_bit_exact(observations, inference, almost_equal)


def test_policy_does_not_suppress_startup_check_failure(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from exp.mw_finetune.dp import cuda_graph, policy

    class FakeModel(torch.nn.Module):
        def __init__(self, cfg, order):
            super().__init__()
            self._policy = SimpleNamespace(model=torch.nn.Identity())

    (tmp_path / "model.safetensors").write_bytes(b"fake")
    monkeypatch.setattr(policy, "memory_cap", lambda *args: None)
    monkeypatch.setattr(policy, "read_metadata", lambda root: {"config": {"model": {"action_dim": 4}}})
    monkeypatch.setattr(policy, "read_stats", lambda root: {})
    monkeypatch.setattr(policy, "load_file", lambda path: {})
    monkeypatch.setattr(policy, "DiffusionPolicy", FakeModel)
    monkeypatch.setattr(policy, "LanguageEncoder", lambda *args: None)

    def fail(*args):
        raise RuntimeError("startup check failed; refusing to serve")

    monkeypatch.setattr(cuda_graph, "require_bit_exact", fail)
    with pytest.raises(RuntimeError, match="refusing to serve"):
        policy.load_policy(tmp_path, "cuda:0", cuda_graph=True)
    with pytest.raises(ValueError, match="requires a CUDA"):
        policy.load_policy(tmp_path, "cpu", cuda_graph=True)


@pytest.mark.parametrize("enabled", (False, True))
def test_metaworld_flag_preserves_default_options(monkeypatch, enabled):
    import sys
    from exp.mw_finetune.dp import serve_compat
    arguments = ["serve_compat", "--model", "dp", "--checkpoint", "/unused"]
    monkeypatch.setattr(sys, "argv", [*arguments, *(["--cuda-graph"] if enabled else [])])
    called = []
    monkeypatch.setattr(serve_compat.serve, "main", lambda: called.append(sys.argv.copy()))
    serve_compat.main()
    expected = [*arguments, *(["--policy-option", "cuda_graph=true"] if enabled else [])]
    assert called == [expected]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay needs a CUDA device")
def test_replay_replaces_every_input_and_caches_by_shape():
    class Tiny(torch.nn.Module):
        def forward(self, trajectory, timestep, cond):
            return trajectory * 2 + timestep.to(trajectory.device) + cond

    device = torch.device("cuda:0")
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(1.25 * 2**30 / total, device)
    model = Tiny().to(device).eval()
    replay = DenoiserGraph(model)
    with torch.inference_mode():
        for batch in (1, 1, 2, 1):
            trajectory = torch.full((batch, 16, 4), batch + 1., device=device)
            for step in (99, 51, 0):
                cond = torch.full_like(trajectory, step + .5)
                timestep = torch.tensor(step)
                assert torch.equal(model(trajectory, timestep, cond), replay(trajectory, timestep, cond))
    assert len(replay.graphs) == 2
