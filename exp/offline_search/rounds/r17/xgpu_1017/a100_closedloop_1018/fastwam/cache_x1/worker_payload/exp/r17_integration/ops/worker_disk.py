"""CPU-only admission on the filesystem that actually holds worker runs.

Uses unprivileged available blocks, including when the run does not exist yet.
No tether traffic inside a worker fleet and no model/simulator imports.
"""
import inspect
import json
import os
from pathlib import Path
import time

DEFAULT_MIN_FREE_GIB = 8


def minimum_bytes(campaign):
    value = campaign.get('worker_min_free_gib', DEFAULT_MIN_FREE_GIB)
    if type(value) not in (int, float) or not 0 < value < 1024:
        raise ValueError('worker_min_free_gib must be positive and below 1024')
    return int(value * 1024**3)


def admission(path, minimum):
    path = Path(path).absolute()
    existing = path
    while not existing.exists():
        if existing == existing.parent:
            raise ValueError('worker disk filesystem not found')
        existing = existing.parent
    usage = os.statvfs(existing)
    free = usage.f_bavail * usage.f_frsize
    # Filesystems without inode accounting report f_files == 0.
    inodes_ok = usage.f_files == 0 or usage.f_favail > 0
    admitted = free >= minimum and inodes_ok
    return dict(schema='r17.worker_disk.v1', path=str(path), filesystem_probe=str(existing),
        filesystem_device=os.stat(existing).st_dev, checked_at=time.time(),
        free_bytes=free, total_bytes=usage.f_blocks * usage.f_frsize,
        free_inodes=usage.f_favail, minimum_free_bytes=minimum, admitted=admitted,
        reason=None if admitted else ('worker disk admission refused: '+str(path)+
            '; available_bytes='+str(free)+'; minimum_free_bytes='+str(minimum)+
            '; available_inodes='+str(usage.f_favail)))


def probe(campaign, directory, node, run, *, stage=None, python='python3', phase='launch'):
    """By-value probe works before a client install and with frozen payloads."""
    from . import formal_eval as launch
    from .formal_process import write_json
    paths = [run] + ([stage] if stage else [])
    code = ('import os,json,time; from pathlib import Path\n'+inspect.getsource(admission)+
        '\nprint(json.dumps([admission(p,'+str(minimum_bytes(campaign))+') for p in '+repr(paths)+']))')
    rows = json.loads(launch.remote(node, [python,'-c',code], replay_safe=True))
    if not isinstance(rows, list) or len(rows) != len(paths) or any(
            type(row.get('admitted')) is not bool or row.get('path') != path
            for row,path in zip(rows,paths)):
        raise ValueError(node+': invalid worker disk admission probe')
    result = dict(node=node,phase=phase,admitted=all(r['admitted'] for r in rows),filesystems=rows)
    write_json(Path(directory)/(node+'_DISK_ADMISSION.json'),result)
    if not result['admitted']:
        raise RuntimeError(node+': '+next(r['reason'] for r in rows if not r['admitted']))
    return result


def prepared(campaign, directory, deployment):
    return {node:probe(campaign,directory,node,item['worker_run'],stage=item['stage'],
                       python=item['python'],phase='launch') for node,item in deployment.items()}
