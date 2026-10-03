"""R11 astra: local CPU analysis, B libraries only, own-directory writes."""
from pathlib import Path
import hashlib
import json
import os
import sys

HERE = Path(__file__).resolve().parent
READS = set()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def check(path, write=False):
    if isinstance(path, int) or path is None:
        return
    p = Path(os.fsdecode(path)).resolve()
    if p == Path('/dev/null'):
        return
    if write:
        if p != HERE and HERE not in p.parents:
            raise PermissionError(f'R11 astra write outside ownership: {p}')
    else:
        if 'os_closed_loop' in p.parts or 'replays' in p.parts:
            raise PermissionError(f'R11 astra refuses closed-loop/replay data: {p}')
        if 'offline_search_store' in p.parts and 'library' not in p.parts:
            raise PermissionError(f'R11 astra refuses non-library store data: {p}')
        if 'rounds' in p.parts or 'offline_search_store' in p.parts:
            READS.add(str(p))


def install():
    def audit(event, args):
        if event == 'open':
            check(args[0], bool(args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
        elif event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod'):
            check(args[0], True)
        elif event in ('os.rename', 'os.link', 'os.symlink'):
            check(args[0], True)
            check(args[1], True)
        elif event in ('socket.connect', 'socket.bind'):
            raise PermissionError('R11 astra has no network work')
    sys.addaudithook(audit)


def dump(path, obj):
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
