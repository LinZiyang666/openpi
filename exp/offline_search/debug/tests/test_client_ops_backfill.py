"""CPU checks for packaging, stock injection, source admission and replay."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from exp.offline_search.debug.backfill.replay import inspect_library, load_source, replay_episode, robot_state
from exp.offline_search.debug.client.parity import replay
from exp.offline_search.debug.ops.build_client_bundle import build
from exp.offline_search.debug.ops.config import manifest, server_args, client_env
from exp.offline_search.debug.ops.install_client_payload import install
from exp.offline_search.debug.transport.receiver import digest
from exp.offline_search.debug.tests.test_client_capture import FakeEnv


def test_bundle_hashes_install_and_python38(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    report = build(one)
    build(two)
    assert digest(one / "client_bundle.tar") == digest(two / "client_bundle.tar")
    root = tmp_path / "island"
    (root / "os_cl").mkdir(parents=True)
    (root / "os_cl/run_arm.sh").write_text("stock unchanged\n")
    assert install(one, root)["protected_unchanged"] == ["os_cl/run_arm.sh"]
    assert (root / "os_cl/run_arm.sh").read_text() == "stock unchanged\n"
    for row in report["files"]:
        path = one / "payload" / row["relative_path"]
        if path.suffix == ".py":
            ast.parse(path.read_text(), feature_version=8)
    # Verify every hash before any mutation; corrupting a payload cannot install.
    corrupt = one / "payload/exp/offline_search/debug/client/worker.py"
    corrupt.write_bytes(corrupt.read_bytes() + b"# corruption\n")
    before = (root / "exp/offline_search/debug/client/worker.py").read_bytes()
    with pytest.raises(ValueError, match="checksum"):
        install(one, root)
    assert (root / "exp/offline_search/debug/client/worker.py").read_bytes() == before


def test_ops_arguments_manifest_and_oracle(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    root.mkdir()
    (root / "arms.json").write_text(json.dumps([dict(arm="A", oracle=True, debug_config={"writer_queue_bytes": 1024})]))
    config = manifest(root, "A")
    assert config["schema"] == "osdebug.v1" and config["env_seed"] == 7
    assert "--os-oracle" in server_args(root, "A", 24099)
    manifest(root, "A")
    monkeypatch.setenv("OSDEBUG_MODE", "file")
    env = client_env(root, "A")
    assert env["OSDEBUG_ORACLE"] == "1" and "OSDEBUG_STREAM" not in env
    assert env["OSDEBUG_CLIENT_DIR"].endswith("/campaign/A")
    monkeypatch.setenv("OSDEBUG_CONFIG", '{"writer_queue_bytes": 8192}')
    # Per-arm frozen config wins over defaults/environment.
    assert json.loads(server_args(root, "A", 24099)[3])["writer_queue_bytes"] == 1024
    monkeypatch.setenv("OSDEBUG_CONFIG", '{"extra": true}')
    with pytest.raises(ValueError, match="differs"):
        manifest(root, "A")


def test_backfill_rejects_guessed_actions_and_admits_matching_replay(tmp_path):
    source = dict(initial_state=np.zeros(5), action=np.array([[0., 0.], [0., 0.], [.1, -.2], [.1, -.2]]),
                  decision_seq=np.array([-1, -1, 0, 0], dtype=np.int32))
    path = tmp_path / "source.npz"
    np.savez_compressed(path, **source)
    meta = dict(npz=str(path), sha256=digest(path), provenance="fake exact issued controls", env_seed=7, suite="fake", episode=0)
    assert np.array_equal(load_source(meta)["action"], source["action"])
    with pytest.raises(ValueError, match="provenance"):
        load_source(dict(npz=str(path)))
    reference = FakeEnv()
    reference.reset()
    reference.set_init_state(source["initial_state"])
    for act in source["action"][:2]:
        reference.step(act.tolist())
    rs = robot_state(reference.observe())[None]
    ep = dict(task_id=0, task="goal", orig_init_state_idx=0, success=False)
    passed = replay_episode(FakeEnv(), source, meta, ep, [0], rs, tmp_path / "pass")
    assert passed["status"] == "PASS"
    rejected = replay_episode(FakeEnv(), source, meta, ep, [0], rs + 1., tmp_path / "reject")
    assert rejected["status"] == "REJECTED" and "robot-state mismatch" in rejected["reason"]
    wrong_outcome = replay_episode(FakeEnv(), source, meta, dict(ep, success=True), [0], rs, tmp_path / "outcome")
    assert wrong_outcome["status"] == "REJECTED" and "success differs" in wrong_outcome["reason"]
    library = tmp_path / "library"
    library.mkdir()
    (library / "episodes.json").write_text(json.dumps([dict(ep, stem="demo", file="proposal.h5")]))
    (library / "manifest.json").write_text('{}')
    assert inspect_library(library)["replay_candidates"] == 0
    assert inspect_library(library, {"0": meta})["replay_candidates"] == 1


def test_fake_environment_full_sequence_parity(tmp_path):
    action = np.tile([.1, -.2], (90, 1))
    identity = dict(suite="fake", task="goal", init=0, task_id=0, uid="parity:fake:0:0")
    before = replay(FakeEnv(), np.zeros(5), action, identity=identity, settle=4)
    after = replay(FakeEnv(), np.zeros(5), action, tmp_path, identity, settle=4)
    assert len(before) == len(after) == 90
    for a, b in zip(before, after):
        for x, y in zip(a, b):
            assert x.tobytes() == y.tobytes()


@pytest.mark.parametrize("mode", ["file", "stream"])
def test_driver_injects_worker_and_persistent_fence_in_both_modes(tmp_path, monkeypatch, mode):
    from exp.offline_search.debug.client import driver, compat
    from exp.offline_search.debug.transport import dispatch_fence
    module = ModuleType("exp.gate_threshold_pareto.run_gtp")
    captured = {}
    module.WorkerSpec = lambda *a, **kw: captured.update(spec=kw)
    module.ConductorDriver = lambda *a, **kw: captured.update(driver=kw)
    def main():
        module.WorkerSpec(seed=None, env={"MUJOCO_EGL_DEVICE_ID": "0"})
        module.ConductorDriver(journal_path=tmp_path / "journal.jsonl")
    module.main = main
    monkeypatch.setitem(sys.modules, "exp.gate_threshold_pareto.run_gtp", module)
    import exp.gate_threshold_pareto
    monkeypatch.setattr(exp.gate_threshold_pareto, "run_gtp", module, raising=False)
    monkeypatch.setattr(compat, "install_driver_compat", lambda: None)
    monkeypatch.setattr(dispatch_fence, "install", lambda directory, journal: captured.update(fence=(directory, journal)))
    monkeypatch.setenv("OSDEBUG_CLIENT_DIR", str(tmp_path / "capture"))
    monkeypatch.setenv("OSDEBUG_MODE", mode)
    monkeypatch.delenv("OSCL_EPISODES", raising=False)
    monkeypatch.delenv("OSCL_MANIFEST", raising=False)
    monkeypatch.setattr(sys, "argv", ["driver"])
    driver.main()
    assert captured["fence"][0] == tmp_path / ".osdebug_dispatch"
    assert captured["spec"]["worker_module"] == "exp.offline_search.debug.client.worker"
    assert captured["spec"]["env"]["MUJOCO_EGL_DEVICE_ID"] == "0"
    assert captured["spec"]["seed"] is None  # stock --seed default, no arm-dependent rewrite


def test_chain_capture_verification_precedes_done():
    script = Path("exp/offline_search/debug/ops/chain_debug.sh").read_text()
    assert script.index("debug.transport.collect") < script.index("debug.validate") < script.index('touch "$DONE"')
    assert "purge_exc.py" not in script and "--os-debug-dir" in Path("exp/offline_search/debug/ops/config.py").read_text()


def test_exact_capture_source_export(tmp_path):
    from exp.offline_search.debug.tests.test_client_capture import make_episode
    from exp.offline_search.debug.backfill.prepare_source import export
    client, _, _, _ = make_episode(tmp_path / "capture", n=7)
    result = export(tmp_path / "capture" / client.identity["episode_key"], tmp_path / "source.npz", 4,
                    "test capture original library provenance", "wire")
    source = load_source(result["4"])
    assert source["action"].shape == (11, 2) and (source["decision_seq"][:4] == -1).all()
    assert result["4"]["env_seed"] == 7 and source["initial_state"].dtype == np.float64


def test_exact_bpool_trajectory_normalization_and_replay(tmp_path):
    import h5py
    from exp.offline_search.debug.backfill.trajectories import load, normalization, discover
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps(dict(norm_stats=dict(state=dict(q01=[-1.] * 8, q99=[1.] * 8)))))
    transform = normalization(stats_path, 32)
    env = FakeEnv()
    env.set_init_state(np.zeros(5))
    for _ in range(2):
        env.step([0., 0., 0., 0., 0., 0., -1.])
    wire, sim = robot_state(env.observe()), env.get_sim_state()
    trajectory = tmp_path / "trajectories/fake/task_0/episode_0.h5"
    trajectory.parent.mkdir(parents=True)
    with h5py.File(trajectory, "w") as f:
        for key, value in dict(task_id=0, orig_init_state_idx=0, success=False, num_cycles=1, num_steps_wait=2,
                               num_steps=2, final_env_timestep=4, seed=7).items():
            f.attrs[key] = value
        g = f.create_group("step_0000")
        g.attrs["executed_action_count"] = 2
        g["executed_actions"] = np.tile([.1, -.2, 0., 0., 0., 0., -1.], (2, 1))
        g["robot_state"] = wire
        g["sim_state"] = sim
    ep = dict(task_id=0, task="goal", orig_init_state_idx=0, success=False, n_rows=1, stem="task_0/episode_0")
    source, meta = load(trajectory, np.zeros(5), ep, "fake")
    meta["episode"] = 0
    result = replay_episode(FakeEnv(), source, meta, ep, [0], transform(wire)[None], tmp_path / "replay", state_transform=transform)
    assert result["status"] == "PASS"
    source["expected_sim_state"][0, -1] += 1.
    bad = replay_episode(FakeEnv(), source, meta, ep, [0], transform(wire)[None], tmp_path / "bad", state_transform=transform)
    assert bad["status"] == "REJECTED" and "simulator state differs" in bad["reason"]
    library = tmp_path / "library"
    library.mkdir()
    (library / "episodes.json").write_text(json.dumps([ep]))
    assert "0" in discover(library, tmp_path / "trajectories", "fake")


def test_worker_preserves_stock_explicit_environment_seed(tmp_path, monkeypatch):
    from exp.offline_search.debug.client import worker
    import examples.libero
    from exp.offline_search.debug.tests.test_client_capture import FakePolicy
    captured = {}
    module = ModuleType("examples.libero.episode_runner")
    class Original:
        def __init__(self, args, **kw):
            captured["client"] = kw["client_factory"](None)
            captured["run_fn"] = kw["run_episode_fn"]
    module.LiberoEpisodeRunner = Original
    module.default_client_factory = lambda endpoint: FakePolicy()
    entry = ModuleType("examples.libero.worker_entry")
    entry.main = lambda: module.LiberoEpisodeRunner(SimpleNamespace(seed=19))
    for name, value in (("episode_runner", module), ("worker_entry", entry)):
        monkeypatch.setitem(sys.modules, "examples.libero." + name, value)
        monkeypatch.setattr(examples.libero, name, value, raising=False)
    monkeypatch.setenv("OSDEBUG_CLIENT_DIR", str(tmp_path))
    monkeypatch.delenv("OSDEBUG_ORACLE", raising=False)
    worker.main()
    assert captured["client"].env_seed == 19
    assert not captured["client"].oracle_enabled


def test_capture_preserves_action_before_environment_mutation(tmp_path):
    from exp.offline_search.debug.client.capture import Client, EnvTap
    from exp.offline_search.debug.tests.test_client_capture import FakePolicy
    env = FakeEnv()
    original = env.step
    def mutate(action):
        result = original(action)
        action[0] = 999.  # a wrapper's in-place postprocessing must not corrupt the issued record
        return result
    env.step = mutate
    client = Client(FakePolicy(), tmp_path, campaign="test", arm="A")
    client.episode_start(experiment="fake", task="goal", episode_id=0, extra_metadata=dict(task_uid="A:eval:0:0", attempt=1))
    client.configure(env, SimpleNamespace(seed=7, num_steps_wait=1, replan_steps=5), 1)
    tap = EnvTap(env, client)
    tap.set_init_state(np.zeros(5))
    issued = [.1, -.2]
    tap.step(issued)
    assert issued[0] == 999.
    assert client.rows[0]["action"][0] == .1
    client.finish(False, "step_cap")
