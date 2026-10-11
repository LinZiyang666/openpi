"""Safety and durability checks for the remote worker paths."""
import contextlib
import fcntl
import hashlib
import io
import json
import subprocess
import sys
import tarfile

import numpy as np
import pytest

from exp.mw_finetune import rendering as R
from exp.mw_finetune.sync_back import SNAPSHOT, install_snapshot


def test_context_lifecycle_is_locked_without_changing_frames(monkeypatch):
    locked = []
    @contextlib.contextmanager
    def lock():
        locked.append(True)
        try: yield
        finally: locked.pop()
    class Env:
        calls = 0
        def render(self):
            if self.calls == 0: assert locked
            self.calls += 1
            return np.zeros((480, 480, 3), np.uint8)
        def close(self): assert locked
    monkeypatch.setattr(R, "env_build_lock", lock)
    monkeypatch.setattr(R, "renderer_stamp", lambda: {"GL_RENDERER": "fake NVIDIA"})
    env = R.LockedEnv(Env())
    assert np.array_equal(env.render(), env.render())
    env.close()
    assert not locked


def test_context_lock_cannot_be_disabled(monkeypatch):
    monkeypatch.setenv("RC365_ENV_BUILD_LOCK", "")
    with pytest.raises(ValueError, match="requires"):
        with R.env_build_lock(): pass


def test_snapshot_verifies_files_and_remains_resumable(tmp_path):
    run = tmp_path / "remote";run.mkdir()
    (run / "manifest.json").write_text('{"trials":2}\n')
    (run / "episodes.jsonl").write_text('{"task":"reach-v3","idx":0}\n')
    archive = tmp_path / "run.tar.gz"
    subprocess.run([sys.executable, "-c", SNAPSHOT, str(run), str(archive)], check=True, capture_output=True)
    local = tmp_path / "local"
    hashes = install_snapshot(archive, local)
    assert (local / "episodes.jsonl").read_bytes() == (run / "episodes.jsonl").read_bytes()
    assert hashes["episodes.jsonl"] == hashlib.sha256((local / "episodes.jsonl").read_bytes()).hexdigest()
    # Sync the same snapshot twice, without changing/removing the source.
    assert install_snapshot(archive, local) == hashes


def test_snapshot_rejects_running_driver(tmp_path):
    run = tmp_path / "run";run.mkdir();(run / "manifest.json").write_text('{}')
    with (run / ".eval.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = subprocess.run([sys.executable, "-c", SNAPSHOT, str(run), str(tmp_path / "run.tar.gz")], capture_output=True, text=True)
        assert result.returncode and "driver is running" in result.stderr


def test_snapshot_rejects_tampered_bytes(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    hashes = json.dumps({"manifest.json": "0" * 64}).encode()
    with tarfile.open(archive, "w:gz") as tar:
        for name, data in (("TRANSFER_SHA256.json", hashes), ("manifest.json", b'{}')):
            info = tarfile.TarInfo(name);info.size = len(data);tar.addfile(info, io.BytesIO(data))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        install_snapshot(archive, tmp_path / "dest")
