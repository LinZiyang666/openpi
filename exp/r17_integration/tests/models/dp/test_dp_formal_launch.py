from types import SimpleNamespace
import pytest

from exp.r17_integration.models.dp.batch_runtime import BatchConfig
from exp.r17_integration.models.dp.formal_launch import launcher_hooks, main


def test_formal_shim_seals_config_and_restores_common_functions():
    calls = []
    prepare = lambda campaign, directory: calls.append(dict(campaign))
    python = lambda campaign: "unchanged"
    launcher = SimpleNamespace(prepare=prepare, host_python=python)
    config = BatchConfig(8, 5)
    campaign = dict(model="dp", domain="libero", server_node="local")
    with launcher_hooks(launcher, config):
        launcher.prepare(campaign, "owned")
        assert calls[-1]["dp_batching"] == config.metadata()
        assert launcher.host_python(campaign).endswith("models/dp/formal_python.sh")
        assert launcher.host_python(dict(model="fastwam")) == "unchanged"
        with pytest.raises(ValueError, match="4090"):
            launcher.host_python(dict(campaign, server_node="h100"))
    assert launcher.prepare is prepare and launcher.host_python is python


def test_default_launch_delegates_exact_arguments(monkeypatch):
    from exp.r17_integration.ops import formal_eval
    seen = []
    monkeypatch.setattr(formal_eval, "main", lambda argv: seen.append(argv))
    main(["--max-batch", "1", "--model", "dp_libero", "--prepare-only"])
    assert seen == [["--model", "dp_libero", "--prepare-only"]]


def test_batched_launch_rejects_other_model(monkeypatch):
    from exp.r17_integration.ops import formal_eval
    monkeypatch.setattr(formal_eval, "parse_args", lambda argv: SimpleNamespace(mode="launch", model="fastwam_libero", server_node="local"))
    with pytest.raises(SystemExit):
        main(["--max-batch", "4"])


@pytest.mark.parametrize("maximum", [8, 32])
def test_normal_formal_launcher_accepts_owned_DP_config(maximum):
    from exp.r17_integration.ops.formal_eval import parse_args
    args = parse_args(["--model", "dp_libero", "--max-batch", str(maximum),
        "--batch-wait-ms", "5", "--server-node", "local", "--worker-node", "timan107",
        "--workers", "12", "--replicas", "1", "--official-50", "--run"])
    assert args.max_batch == maximum and args.batch_wait_ms == 5
    assert args.server_node == "local" and args.worker_node == "timan107"


def test_normal_resident_dispatch_keeps_graph_and_measured_stage_settings(monkeypatch):
    from exp.r17_integration.ops import formal_server, formal_eval
    from exp.r17_integration.models.dp import formal_batch_server
    config = BatchConfig(8, 5)
    campaign = dict(model="dp", domain="libero", server_node="local", dp_batching=config.metadata())
    seen = []
    monkeypatch.setattr(formal_batch_server, "serve", lambda *args, **kwargs: seen.append((args, kwargs)))
    formal_server.serve(campaign, 1, "owned_directory", cuda_graph=True)
    assert seen == [((campaign, 1, "owned_directory"), {"cuda_graph": True})]
    env = formal_eval.server_env(campaign, "/home/weiland/projects/openpi")
    assert env["OPENPI_DISABLE_STAGE_STREAMS"] == "0"
    assert all(env[f"BATCHING_STAGE{i}_WORKERS"] == "1" for i in (1,2,3))
