"""Short private AF_UNIX addresses, independent of persistent run-root length."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import tempfile

POINTER = "stage_endpoint.json"
MAX_PATH_BYTES = 107  # Linux sockaddr_un.sun_path includes a terminating NUL.


@contextmanager
def private_endpoint(runtime, *, environ=None):
    runtime = Path(runtime).resolve()
    environ = os.environ if environ is None else environ
    prefix = "opi-" + hashlib.sha256(os.fsencode(runtime)).hexdigest()[:10] + "-"
    candidates = [environ.get("XDG_RUNTIME_DIR"), "/tmp"]
    directory = None
    for candidate in dict.fromkeys(candidates):
        if not candidate:
            continue
        parent = Path(candidate).resolve()
        try:
            info = parent.stat()
            # An XDG directory must belong to us and be private; /tmp is a
            # shared sticky directory, but mkdtemp creates our own 0700 leaf.
            if str(parent) != "/tmp" and (info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700):
                continue
            if len(os.fsencode(parent / (prefix + "12345678") / "stages.sock")) > MAX_PATH_BYTES:
                continue
            directory = tempfile.TemporaryDirectory(prefix=prefix, dir=parent)
            endpoint = Path(directory.name) / "stages.sock"
            if len(os.fsencode(endpoint)) > MAX_PATH_BYTES:
                directory.cleanup()
                directory = None
                continue
            break
        except OSError:
            continue
    if directory is None:
        raise OSError("split: cannot allocate a short private IPC directory in XDG_RUNTIME_DIR or /tmp")
    try:
        pointer = dict(endpoint=str(endpoint), directory=directory.name, uid=os.getuid())
        path = runtime / POINTER
        path.write_text(json.dumps(pointer, indent=2) + "\n")
        path.chmod(0o600)
        yield endpoint
    finally:
        # Only this launch's mkdtemp leaf; never remove a shared parent.
        directory.cleanup()


def read_endpoint(runtime):
    pointer = Path(runtime) / POINTER
    if pointer.stat().st_uid != os.getuid() or stat.S_IMODE(pointer.stat().st_mode) != 0o600:
        raise PermissionError(f"split: IPC pointer must be owned by this user and mode 0600: {pointer}")
    value = json.loads(pointer.read_text())
    endpoint = Path(value["endpoint"])
    directory = endpoint.parent
    info = directory.stat()
    if (value["uid"] != os.getuid() or info.st_uid != os.getuid() or
            stat.S_IMODE(info.st_mode) != 0o700 or directory.is_symlink() or
            endpoint.name != "stages.sock" or str(directory) != value["directory"] or
            not endpoint.is_absolute() or len(os.fsencode(endpoint)) > MAX_PATH_BYTES):
        raise PermissionError("split: invalid private IPC endpoint")
    return endpoint


def endpoint_ready(endpoint, blueprint):
    if not Path(blueprint).is_file():
        return False
    try:
        if not stat.S_ISSOCK(Path(endpoint).stat().st_mode):
            return False
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.25)
            probe.connect(str(endpoint))
        return True
    except (OSError, TimeoutError):
        return False
