"""Exercise owned subprocess lifecycles and adversarial whole-batch evidence.

The integration uses real loopback listeners and child workers. Only GPU/model
loading and the already separately tested full-source preflight are replaced;
the runner's process sampler, termination, catalog and recovery code are real.
"""

from __future__ import annotations

import json
import socket
import sys
import subprocess
from types import SimpleNamespace

import pytest

from exp.ablation_study.cache_prune import run_prune_eval as runner
from exp.ablation_study.cache_prune.common import file_identity, read_json, seal
from .batch_fixtures import write_batch_evidence
from .test_cache_prune_evidence import episode_files as episode_files


def _augment(fixture):
    freeze, arm, launch, journal, steps, directory = fixture
    library, yaml_path = directory / "library.pkl", directory / "cache.yaml"
    library.write_bytes(b"synthetic process test library")
    yaml_path.write_text("synthetic process test config\n")
    arm.update(point="P00", regime="rit50", yaml=file_identity(yaml_path))
    arm["artifact"].update(file=file_identity(library), entries=10)
    freeze["arms"] = [arm]
    apool = {
        "apool_dir": str(directory / "inits"),
        "per_task_digests": {f"task_{i}": str(i) for i in range(10)},
    }
    record = directory / "apool.yaml"
    record.write_text("synthetic pool binding\n")
    freeze["memberships"]["libero_spatial"].update(
        apool=apool, apool_record=file_identity(record)
    )
    freeze["matrices"] = {"libero_spatial": {"path": str(directory / "matrix.yaml")}}
    return freeze, arm, launch, journal, steps, directory


def test_run_preflight_rehashes_all_only_on_first_batch(episode_files, monkeypatch):
    """Later batches retain frozen metadata/A-pool checks without scanning all HDF5s."""
    freeze, _, _, _, _, root = _augment(episode_files)
    flags = []
    monkeypatch.setattr(
        runner, "validate_freeze", lambda value, *, rehash: flags.append(rehash)
    )
    membership = freeze["memberships"]["libero_spatial"]
    monkeypatch.setattr(runner, "checked_apool", lambda *args: membership["apool"])
    run = root / "catalog"
    run.mkdir()
    runner.validate_run(
        freeze, "libero_spatial", membership["apool_record"]["path"], run
    )
    (run / "run.json").write_text(
        json.dumps(seal({"version": runner.VERSION, "freeze_digest": freeze["digest"]}))
    )
    runner.validate_run(
        freeze, "libero_spatial", membership["apool_record"]["path"], run
    )
    assert flags == [True, False]
    (run / "run.json").write_text(
        json.dumps(seal({"version": runner.VERSION, "freeze_digest": "changed"}))
    )
    with pytest.raises(ValueError, match="resume input changed"):
        runner.validate_run(
            freeze, "libero_spatial", membership["apool_record"]["path"], run
        )


def test_cold_client_probe_timeout_has_persisted_infra_evidence(tmp_path, monkeypatch):
    """Cold imports get 600 seconds and failures retain output before any server starts."""

    def timeout(command, **kwargs):
        assert kwargs["timeout"] == 600
        raise subprocess.TimeoutExpired(
            command, kwargs["timeout"], output=b"cold import\n", stderr=b"timeout\n"
        )

    monkeypatch.setattr(runner.subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        runner._client_probe(
            {"conda_env": "test"}, {"apool_dir": str(tmp_path)}, tmp_path
        )
    failure = read_json(tmp_path / "preflight_failure.json")
    assert failure["kind"] == "infra_failure" and failure["status"] == "invalid"
    assert (tmp_path / "client_probe.log").read_text() == "cold import\ntimeout\n"


@pytest.mark.parametrize(
    "problem",
    [
        "pid",
        "path",
        "second_load",
        "wrong_count",
        "rss",
        "gpu",
        "missing_gpu",
        "runner_pool",
        "model",
        "code",
        "exit",
    ],
)
def test_node_evidence_rejects_semantically_wrong_records(episode_files, problem):
    """Resealed metadata still cannot contradict node measurements or launch inputs."""
    freeze, arm, launch, journal, steps, root = _augment(episode_files)
    batch = root / "run" / "batch_test"
    batch.mkdir(parents=True)
    for name, rows in (("journal.jsonl", journal), ("per_step.jsonl", steps)):
        (batch / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_batch_evidence(batch, freeze, arm, launch)
    runner.validate_completion(batch, freeze)
    post = read_json(batch / "node_postflight.json")
    if problem == "pid":
        post["process"]["pid"] += 1
    elif problem == "path":
        post["library"]["path"] += ".wrong"
    elif problem == "second_load":
        path = batch / "server.log"
        path.write_text(path.read_text() * 2)
    elif problem == "wrong_count":
        path = batch / "server.log"
        path.write_text(path.read_text().replace("Loaded 10", "Loaded 9"))
    elif problem == "rss":
        post["peak_server_rss"] = 65 * runner.GIB
    elif problem == "gpu":
        post["peak_gpu_bytes"] = 8 * runner.GIB
    elif problem == "missing_gpu":
        path = batch / "resources.jsonl"
        sample = json.loads(path.read_text())
        sample["gpus"] = []
        path.write_text(json.dumps(sample) + "\n")
    elif problem == "runner_pool":
        path = batch / "per_step.jsonl.launch.json"
        data = read_json(path)
        data["apool"]["apool_dir"] = "/wrong/pool"
        path.write_text(json.dumps(data))
    elif problem in ("model", "code"):
        post["model" if problem == "model" else "implementation"] = []
    else:
        post["owned_processes_exited"] = False
    post["evidence_files"] = {
        name: file_identity(item["path"])
        for name, item in post["evidence_files"].items()
    }
    (batch / "node_postflight.json").write_text(json.dumps(seal(post)))
    completion = read_json(batch / "completion.json")
    completion["node_postflight"] = file_identity(batch / "node_postflight.json")
    (batch / "completion.json").write_text(json.dumps(seal(completion)))
    with pytest.raises(ValueError):
        runner.validate_completion(batch, freeze)


def test_owned_server_failure_recovery_and_valid_batch_replay_rejection(
    episode_files, monkeypatch
):
    """Keep a failed full batch, rerun on new processes, and accept only the new whole batch."""
    freeze, arm, _, journal, steps, root = _augment(episode_files)
    # Keep node admission independent of the test host's available RAM while
    # retaining real per-process memory samples and cleanup.
    monkeypatch.setattr(
        runner.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=128 * runner.GIB, available=100 * runner.GIB),
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    spec = {
        "server_python": sys.executable,
        "model_config": "pi05_libero",
        "checkpoint_dir": str(root),
        "host": "127.0.0.1",
        "port": port,
        "workers": 1,
        "gpus": 1,
        "conda_env": "synthetic",
        "client_budget_gib": 0.01,
        "gpu_budget_gib": 4,
        "stage1_device": "cuda:0",
        "stage2_device": "meta",
        "stage3_device": "meta",
    }
    apool = freeze["memberships"]["libero_spatial"]["apool"]
    monkeypatch.setattr(runner, "validate_run", lambda *args: {"apool": apool})
    monkeypatch.setattr(
        runner,
        "_model_identity",
        lambda spec: {"digest": "synthetic-model", "config": "pi05_libero"},
    )
    monkeypatch.setattr(
        runner,
        "gpu_snapshot",
        lambda pids: [{"uuid": "synthetic-gpu", "owned_bytes": 1024}],
    )

    def client_probe(spec, apool, directory):
        (directory / "client_probe.log").write_text("synthetic client environment\n")
        return {
            "num_steps_wait": 10,
            "replan_steps": 5,
            "max_steps": {"libero_spatial": 220, "libero_10": 520},
            "per_task_digests": apool["per_task_digests"],
        }

    monkeypatch.setattr(runner, "_client_probe", client_probe)
    server_script = """import socket,sys,time
s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
s.bind(('127.0.0.1',int(sys.argv[1])));s.listen()
print('Loaded 10 entries from '+sys.argv[2],flush=True)
time.sleep(90)
"""
    monkeypatch.setattr(
        runner,
        "_server_command",
        lambda spec, arm: [
            sys.executable,
            "-c",
            server_script,
            str(port),
            arm["artifact"]["file"]["path"],
        ],
    )
    inputs = root / "inputs.json"
    inputs.write_text(
        json.dumps(
            {"journal": journal, "steps": steps, "apool": apool, "arm": arm["arm"]}
        )
    )
    fail_marker = root / "fail"
    evaluator_script = """import json,pathlib,subprocess,sys,time
batch=pathlib.Path(sys.argv[1]); data=json.loads(pathlib.Path(sys.argv[2]).read_text());smoke=sys.argv[3]=='True'
worker=subprocess.Popen([sys.executable,'-c','import time;time.sleep(1.5)','examples.libero.worker_entry',
'--task-suite-name','libero_spatial','--server-key',sys.argv[4],'--init-states-dir',data['apool']['apool_dir']])
selected=lambda r:not smoke or r['task_uid'].rsplit(':',1)[1]=='0'
for name,key in [('journal.jsonl','journal'),('per_step.jsonl','steps')]:
 rows=[r for r in data[key] if selected(r)]
 if pathlib.Path(sys.argv[5]).exists():rows=rows[:1]
 (batch/name).write_text(''.join(json.dumps(r)+'\\n' for r in rows))
(batch/'per_step.jsonl.launch.json').write_text(json.dumps({'suite':'libero_spatial','arms':[data['arm']],
'trials_per_task':1 if smoke else 50,'smoke':smoke,'apool':data['apool']}))
worker.wait()
if pathlib.Path(sys.argv[5]).exists():sys.exit(1)
"""
    monkeypatch.setattr(
        runner,
        "_eval_command",
        lambda spec, freeze, arm, directory, smoke: [
            sys.executable,
            "-c",
            evaluator_script,
            str(directory),
            str(inputs),
            str(smoke),
            f"127.0.0.1:{port}",
            str(fail_marker),
        ],
    )
    run_dir = root / "run"
    probe = runner.run_batch(freeze, arm["arm"], spec, run_dir, probe=True)
    assert probe["status"] == "valid"
    probe_path = next(run_dir.glob("batch_*/completion.json"))
    smoke = runner.run_batch(
        freeze, arm["arm"], spec, run_dir, smoke=True, p00_probe=probe_path
    )
    assert smoke["status"] == "valid"
    fail_marker.touch()
    with pytest.raises(ValueError, match="invalid batch retained"):
        runner.run_batch(freeze, arm["arm"], spec, run_dir, p00_probe=probe_path)
    invalid_path = next(
        p
        for p in run_dir.glob("batch_*/completion.json")
        if read_json(p)["status"] == "invalid"
    )
    invalid_files = {
        str(p): file_identity(p) for p in invalid_path.parent.iterdir() if p.is_file()
    }
    fail_marker.unlink()
    final = runner.run_batch(freeze, arm["arm"], spec, run_dir, p00_probe=probe_path)
    assert final["status"] == "valid"
    accepted = runner.accepted_batches(run_dir, freeze, require_complete=True)
    assert set(accepted) == {arm["arm"]}
    assert (
        invalid_path.parent.name
        in accepted[arm["arm"]]["launch"]["supersedes_invalid_batches"]
    )
    assert all(
        file_identity(path) == identity for path, identity in invalid_files.items()
    )
    for launch_path in run_dir.glob("batch_*/launch.json"):
        assert not runner._alive(read_json(launch_path)["server_process"])
    with pytest.raises(ValueError, match="complete valid batch already exists"):
        runner.run_batch(freeze, arm["arm"], spec, run_dir, p00_probe=probe_path)
