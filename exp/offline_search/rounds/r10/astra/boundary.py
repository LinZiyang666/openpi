"""Fail closed on forbidden result reads and writes outside this task."""
from pathlib import Path
import os
import sys

HERE = Path(__file__).resolve().parent
RUNS = Path('/home/weiland/trace_runs/os_closed_loop')
NEW = tuple(RUNS / f'r10_corr3_{m}' for m in ('pi05', 'groot'))
OLD = tuple(RUNS / f'r10_{stage}_{m}' for stage in ('size', 'corr2') for m in ('pi05', 'groot'))


def inside(p, root):
    return p == root or root in p.parents


def check_path(path, write=False):
    if isinstance(path, int) or path is None:
        return
    p = Path(os.fsdecode(path)).resolve()
    if p == Path('/dev/null'):
        return  # A character sink, not a filesystem artifact (dill probes its type).
    if write:
        if not any(inside(p, r) for r in (HERE, *NEW)):
            raise PermissionError(f'astra write boundary: {p}')
    elif inside(p, RUNS):
        for root in OLD:
            if inside(p, root):
                rel = p.relative_to(root).parts
                if rel and (rel[0] in ('fits', 'config') or rel == ('arms.json',) or rel == ('eval500.json',)):
                    return
        for root in NEW:
            if inside(p, root):
                rel = p.relative_to(root).parts
                if rel and rel[0] not in ('runs', 'logs', 'clients', 'journals', 'summary', 'selftest', 'validation'):
                    return
        raise PermissionError(f'astra closed-loop read boundary: {p}')


def install():
    def hook(event, args):
        if event == 'open':
            path, mode, flags = args
            check_path(path, bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
        elif event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod'):
            check_path(args[0], True)
        elif event in ('os.rename', 'os.link', 'os.symlink'):
            check_path(args[0], True)
            check_path(args[1], True)
        elif event in ('socket.connect', 'socket.bind'):
            raise PermissionError('astra is local CPU only')
    sys.addaudithook(hook)
