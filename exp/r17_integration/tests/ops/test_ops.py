import fcntl
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

from exp.r17_integration.ops.build_client_bundle import build_client_bundle, sha256, verified_members, worker_payload_files
from exp.r17_integration.ops.deploy import install_payload, tether_transfer
from exp.r17_integration.ops.plan import cpu_blueprint, qualification_plan, unix_socket_path
from exp.r17_integration.ops.serve_registry import PortReservation, RESERVED, compatible
from exp.r17_integration.ops.smoke import stop_owned
from exp.r17_integration.ops.sync_back import install_snapshot, snapshot_run


def test_python311_client_payload_exact_sha_import_without_model_islands(tmp_path):
    root = Path(__file__).resolve().parents[4]
    files = ["exp/r17_integration/ops/" + name + ".py" for name in (
        "build_client_bundle", "deploy", "sync_back", "serve_registry", "plan")]
    built = build_client_bundle(root, files, tmp_path / "client.tar")
    dest = tmp_path / "client"
    install_payload(tmp_path / "client.tar", dest, built["archive_sha256"])
    command = "import sys; from exp.r17_integration.ops import deploy, sync_back, plan; assert sys.version_info[:2] == (3,11); assert not any(x in sys.modules for x in ('torch','jax','mujoco','lerobot','gr00t'))"
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(dest), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-c", command], cwd=dest, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert install_payload(tmp_path / "client.tar", dest, built["archive_sha256"]) == built["manifest"]


def test_client_payload_rejects_model_island_and_tamper(tmp_path):
    root = tmp_path / "src"
    (root / "models").mkdir(parents=True)
    (root / "models" / "net.py").write_text("import torch")
    with pytest.raises(ValueError, match="model-island"):
        build_client_bundle(root, ["models/net.py"], tmp_path / "bad.tar")
    (root / "client.py").write_text("x=1")
    receipt = build_client_bundle(root, ["client.py"], tmp_path / "client.tar")
    with pytest.raises(ValueError, match="SHA"):
        install_payload(tmp_path / "client.tar", tmp_path / "dest", "wrong")
    manifest, _ = verified_members(tmp_path / "client.tar", "PAYLOAD.json")
    assert manifest == receipt["manifest"]


def test_p2_actual_worker_payload_python311_model_free(tmp_path):
    import os
    root = Path(__file__).resolve().parents[4]
    files = worker_payload_files(root, debug=True)
    receipt = build_client_bundle(root, files, tmp_path / "worker.tar")
    dest = tmp_path / "worker"
    install_payload(tmp_path / "worker.tar", dest, receipt["archive_sha256"])
    env = dict(os.environ, PYTHONPATH=f"{dest}:{dest}/src:{dest}/packages/openpi-client/src", PYTHONDONTWRITEBYTECODE="1")
    command = "from exp.r17_integration.conductor.worker_entry import import_probe; from exp.r17_integration.debug.capture_bridge import CaptureBridge; import sys; assert sys.version_info[:2]==(3,11); print(import_probe()); assert not any(x in sys.modules for x in ('torch','jax','mujoco','metaworld','lerobot','gr00t'))"
    result = subprocess.run([sys.executable, "-c", command], cwd=dest, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_deployed_debug_payload_captures_native_sampler_without_model_islands(tmp_path):
    """Review A5/A10: exercise writing, not only import, in copied worker payload."""
    import os
    from exp.r17_integration.tests.debug.fixtures import binding, record
    root = Path(__file__).resolve().parents[4]
    receipt = build_client_bundle(root, worker_payload_files(root, debug=True), tmp_path / "worker.tar")
    dest = tmp_path / "worker"
    install_payload(tmp_path / "worker.tar", dest, receipt["archive_sha256"])
    b = binding()
    event = dict(record(), event="control", control_idx=0, sampler=b["sampler"],
                 physical={"arrays":{"qpos":[1.,2.,3.]}})
    (dest/"input.json").write_text(json.dumps({"metadata":{"r17_model_spec":b["model"]},"event":event}))
    command = """import json,sys,numpy as np
from exp.r17_integration.debug.capture_bridge import PhysicalCaptureFactory
from exp.r17_integration.debug.reader_profiles import CaptureReader
from pathlib import Path
value=json.loads(Path('input.json').read_text())
value['event']['physical']['arrays']['qpos']=np.asarray(value['event']['physical']['arrays']['qpos'],dtype=np.float64)
factory=PhysicalCaptureFactory('capture',value['metadata'])
factory(value['event']); factory.close()
reader=CaptureReader(next(Path('capture').iterdir()))
row,arrays=next(reader.physical())
assert arrays['qpos'].dtype==np.float64
assert reader.manifest['binding']['sampler']==value['event']['sampler']
assert not any(x in sys.modules for x in ('torch','jax','mujoco','metaworld','lerobot','gr00t'))
"""
    env = dict(os.environ, PYTHONPATH=f"{dest}:{dest}/src:{dest}/packages/openpi-client/src", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable,"-c",command],cwd=dest,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_snapshot_exact_idempotence_partials_tamper_and_busy(tmp_path):
    run = tmp_path / "remote"
    run.mkdir()
    (run / "manifest.json").write_text('{"campaign":"cpu"}\n')
    (run / "journal.jsonl").write_text('{"accepted":true}\n')
    (run / "interrupted.npz.part").write_bytes(b"partial remains auditable")
    with (run / ".eval.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="busy"):
            snapshot_run(run, tmp_path / "run.tar")
    receipt = snapshot_run(run, tmp_path / "run.tar")
    dest = tmp_path / "local"
    first = install_snapshot(tmp_path / "run.tar", dest, expected_sha256=receipt["archive_sha256"])
    second = install_snapshot(tmp_path / "run.tar", dest, expected_sha256=receipt["archive_sha256"])
    assert first == second and first["partials"] == ["interrupted.npz.part"]
    assert (dest / "journal.jsonl").read_bytes() == (run / "journal.jsonl").read_bytes()
    with (dest / ".eval.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="busy"):
            install_snapshot(tmp_path / "run.tar", dest, expected_sha256=receipt["archive_sha256"])
    (dest / "journal.jsonl").write_text("tamper\n")
    with pytest.raises(ValueError, match="differs"):
        install_snapshot(tmp_path / "run.tar", dest, expected_sha256=receipt["archive_sha256"])


def test_snapshot_rejects_symlink_and_archive_missing_files(tmp_path):
    import tarfile
    from exp.r17_integration.ops.build_client_bundle import add_bytes, json_bytes
    import hashlib
    manifest = dict(files={"missing": "sha"})
    manifest["digest"] = hashlib.sha256(json_bytes(manifest)).hexdigest()
    path = tmp_path / "bad.tar"
    with tarfile.open(path, "w") as tar:
        add_bytes(tar, "TRANSFER_SHA256.json", json_bytes(manifest))
    with pytest.raises(ValueError, match="missing/unlisted"):
        install_snapshot(path, tmp_path / "dest", expected_sha256=sha256(path))
    assert not (tmp_path / "dest").exists()


def test_tether_exit75_retries_bounded_only_for_saturation():
    calls, delays = [], []
    def runner(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=75 if len(calls)<3 else 0, stderr="too_many_in_flight", stdout="")
    receipt = tether_transfer(["tether", "pull", "fixture", "fixture"], lane="tether", runner=runner, sleeper=delays.append)
    assert receipt["attempts"] == 3 and delays == [2, 4]
    bad = lambda *a, **k: SimpleNamespace(returncode=75, stderr="unrelated", stdout="")
    with pytest.raises(subprocess.CalledProcessError):
        tether_transfer(["fake"], lane="local", runner=bad, sleeper=delays.append)
    saturated = lambda *a, **k: SimpleNamespace(returncode=75, stderr="too_many_in_flight", stdout="")
    with pytest.raises(subprocess.CalledProcessError):
        tether_transfer(["fake"], lane="tether", runner=saturated, sleeper=delays.append, attempts=2)


def test_freeport_held_and_reserved_ports_refused():
    for port in RESERVED:
        with pytest.raises(ValueError, match="reserved"):
            PortReservation(port=port)
    with PortReservation() as reservation:
        other = socket.socket()
        try:
            with pytest.raises(OSError):
                other.bind(("127.0.0.1", reservation.port))
        finally:
            other.close()


def test_owned_stop_never_targets_unowned_process():
    calls = []
    process = SimpleNamespace(pid=123, returncode=0, poll=lambda: None,
                              terminate=lambda: calls.append("terminate"), wait=lambda timeout: calls.append(timeout))
    with pytest.raises(ValueError, match="owned receipt"):
        stop_owned(process, {"pid": 456, "owned": True})
    assert calls == []
    stop_owned(process, {"pid": 123, "owned": True})
    assert calls == ["terminate", 5]


def test_blueprint_and_short_unix_path_without_cuda():
    assert cpu_blueprint({"model": "fixture", "shape": (16, 4)})["shape"] == [16, 4]
    with pytest.raises(ValueError, match="non-CPU"):
        cpu_blueprint({"bad": SimpleNamespace(device="cuda:0")})
    assert len(unix_socket_path("/tmp", "a"*64).encode()) < 104
    with pytest.raises(ValueError, match="Unix"):
        unix_socket_path("/tmp/" + "x"*120, "a"*64)


def test_plan_and_smoke_are_default_dry_run(tmp_path):
    receipt = tmp_path / "plan.json"
    result = subprocess.run([sys.executable, "-m", "exp.r17_integration.ops.smoke", "--model", "dp",
                             "--checkpoint-sha", "sealed", "--receipt", str(receipt)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    plan = json.loads(receipt.read_text())
    assert plan["executed"] is False and plan["deadline_seconds"] == 900 and plan["gpu_concurrency"] == 1


@pytest.mark.parametrize("field", ["model_identity", "benchmark_protocol", "norm_identity", "layout_identity", "sampler_identity"])
def test_endpoint_registry_strict_compatibility(field):
    endpoint = {name: "sealed" for name in ("model_identity", "benchmark_protocol", "norm_identity", "layout_identity", "sampler_identity")}
    assert compatible(endpoint, endpoint)
    altered = dict(endpoint, **{field: "other"})
    assert not compatible(endpoint, altered)
