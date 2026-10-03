"""Server-test readers and a no-git guard for restricted CPU validation."""
import json
import subprocess
from pathlib import Path

import pytest

from exp.offline_search.closed_loop import plugin
from exp.offline_search.debug import schema


@pytest.fixture(autouse=True)
def no_git(monkeypatch):
    original = subprocess.check_output
    def guarded(command, *args, **kwargs):
        if isinstance(command, (list, tuple)) and command and command[0] == "git":
            raise OSError("git disabled for restricted CPU validation")
        return original(command, *args, **kwargs)
    monkeypatch.setattr(subprocess, "check_output", guarded)
    monkeypatch.setattr(plugin, "_git_head", lambda: "disabled-in-cpu-test")


def server_json(directory, stem):
    files = sorted(Path(directory).glob(stem + "*.json"))
    assert len(files) == 1, files
    return json.loads(files[0].read_text())


def server_rows(directory, issues=None):
    return [row for path in sorted(Path(directory).glob("decisions*.jsonl"))
            for row in schema.read_jsonl(path, issues)]
