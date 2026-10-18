"""Materialize the standard R10 timan107 worker deployment; transfer nothing."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from unittest.mock import patch

from .audit_counts import HERE, read
from .build import write

DEST = Path('/scratch/zixuans8/openpi_trace_h100')
BASE = Path('/data/r17_library/libero/pi05/r10_worker_restore_ff1015')


def build():
    from exp.offline_search.closed_loop.ops.h100 import control
    root = BASE/'root'
    root.mkdir(parents=True, exist_ok=True)
    sources = {}
    def copy(local, relative):
        local, relative = Path(local), Path(relative)
        target = root/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, target)
        if local.suffix == '.sh':target.chmod(0o755)
        sources[str(relative)] = str(local)
    def push(node, local, destination):
        assert node == 'timan107'
        copy(local, Path(destination).relative_to(DEST))
    def bundle(node, destination, pairs):
        assert node == 'timan107'
        for local, relative in pairs:copy(local, Path(destination).relative_to(DEST)/relative)
    with patch.dict(os.environ, WORKER_HOST='timan107', SERVER_HOST='h100'), \
            patch.object(control, 'remote', side_effect=lambda *a, **k: ''), \
            patch.object(control, 'push', side_effect=push), patch.object(control, 'bundle', side_effect=bundle):
        # This is the original deploy implementation, with all remote/file
        # transport replaced by local materialization, including private cfg
        # generation, client aliases, A pools and the timan107 subset rewrite.
        control._setup(worker_only=True)
    run = Path('/home/weiland/trace_runs/os_closed_loop/r17_filter_failed_pi05')
    plan = read(run/'h100_sync/plan.json')
    original = {r['arm']:r for r in read(run/'arms.json')}
    for row in plan['arms']:
        arm = original[row['arm']]
        cfg = next(f for f in plan['files'] if f['original'] == arm['yaml'])
        copy(cfg['source'], 'os_cl/cfg/'+row['arm']+'.yaml')
        copy(arm['matrix'], 'os_cl/cfg/matrix_'+row['arm']+'.yaml')
    files = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in root.rglob('*') if p.is_file() and p.name != 'SHA256SUMS'}
    sums = ''.join(pin+'  '+str(DEST/name)+'\n' for name,pin in sorted(files.items()))
    (root/'SHA256SUMS').write_text(sums)
    manifest = dict(schema='r17.r10_worker_restore.v1', source_pipeline='h100.control._setup(worker_only=True)',
                    source_references=['exp/offline_search/closed_loop/ops/h100/control.py:300',
                                       'exp/offline_search/closed_loop/ops/h100/HANDBACK_LOCAL.md',
                                       'exp/offline_search/rounds/r13/impl/HANDBACK.md:44'],
                    local_root=str(root), destination=str(DEST), files=files, source_files=sources,
                    files_count=len(files), bytes=sum((root/n).stat().st_size for n in files),
                    sha256sums_sha256=hashlib.sha256(sums.encode()).hexdigest(), transfers_executed=False)
    write(HERE/'R10_RESTORE_MANIFEST.json', manifest)
    config = BASE/'rsyncd.conf'
    config.write_text('use chroot = no\nread only = yes\nlist = no\nmax connections = 1\n'
                      'pid file = '+str(BASE/'rsync.pid')+'\nlog file = '+str(BASE/'rsync.log')+
                      '\n[r10_ff1015]\npath = '+str(root)+'\n')
    commands = ('# Terminal on weilandserver; foreground public-network rsync transport only:\n'
                'rsync --daemon --no-detach --port=23118 --config='+str(config)+'\n\n'
                '# Terminal on timan107 (data goes directly to the public rsync endpoint):\n'
                'mkdir -p '+str(DEST)+'\n'
                'rsync -rlt --checksum --bwlimit=16384 --timeout=90 --contimeout=15 '
                'rsync://ziyanglin.com:23118/r10_ff1015/ '+str(DEST)+'/\n'
                'sha256sum -c '+str(DEST/'SHA256SUMS')+'\n\n'
                '# Equivalent CPU control/verification commands from weilandserver:\n'
                'tether exec --safe --timeout 30s timan107 -- mkdir -p '+str(DEST)+'\n'
                'tether exec --safe --timeout 120s timan107 -- sha256sum -c '+str(DEST/'SHA256SUMS')+'\n')
    (HERE/'R10_RESTORE_COMMANDS.txt').write_text(commands)
    print(json.dumps({k:manifest[k] for k in ('local_root','files_count','bytes','sha256sums_sha256','transfers_executed')}))


if __name__=='__main__':build()
