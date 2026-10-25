"""Read a run log that a finished run may keep zstd-compressed as <name>.zst.

The plain file wins when present; otherwise `<path>.zst` is streamed through
/usr/bin/zstd -dc. Callers keep passing the plain name.
"""
from contextlib import contextmanager
from pathlib import Path
import signal
import subprocess

ZSTD = '/usr/bin/zstd'


def exists(path):
    return Path(path).exists() or Path(f'{path}.zst').exists()


@contextmanager
def open_log(path, mode='r'):
    """Like Path(path).open(mode) for reading; falls back to decompressing `path`.zst."""
    if Path(path).exists() or not Path(f'{path}.zst').exists():
        with Path(path).open(mode) as f:
            yield f
        return
    with subprocess.Popen([ZSTD, '-dcq', f'{path}.zst'], stdout=subprocess.PIPE, text='b' not in mode) as proc:
        yield proc.stdout
    # SIGPIPE only when the caller stopped reading early; a corrupt or truncated .zst exits 1.
    if proc.returncode not in (0, -signal.SIGPIPE):
        raise subprocess.CalledProcessError(proc.returncode, proc.args)
