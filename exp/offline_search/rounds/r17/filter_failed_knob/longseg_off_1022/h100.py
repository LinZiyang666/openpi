"""The H100 side of the R16 (reduced K1) groups of longseg_off_1022, driven from weilandserver.

Files travel by rsync over the H100's public IP, bandwidth-limited, through a temporary token-named rsync
daemon on the H100 (port 23117, alive only for one copy); tether exec only runs commands (no expose/push/pull).

The residents see the same absolute paths as on weilandserver; on the H100 they are links into
/data/oscl_h100 (LINKS): the store and library PKLs are the copies the R15 H100 sync already keeps
(/data/oscl_h100/store, /data/oscl_h100/mirror), refreshed here by rsync's size+mtime check; fits,
definitions, group roots and the residents' outputs live in /data/oscl_h100/lo_1022.

  code               the R16 Q3 server tree (code only) -> /data/oscl_h100/openpi_r16_q3_server, plus h100_side.py
  sync --group G     code, links and every input of group G (prepare)
  pull --group G     the residents' outputs (runs/, resident/) back into G's root (the runner does it at DONE)
  plan --group G     list what sync would copy, with sizes (no transfer)
  start/status/stop  one resident (conductor_runner's local_server calls, see runner.py)
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import subprocess
import tempfile
import uuid

import yaml

IP = '149.165.153.233'
PORT = 23117
BWLIMIT = 16384  # KiB/s
BASE = '/data/oscl_h100'
TREE = BASE + '/openpi_r16_q3_server'
LO = BASE + '/lo_1022'
VENV = '/home/exouser/openpi/.venv'  # the H100's openpi venv (pi0.5 servers of the R15 H100 path use it)
SERVER_CPUS = 10  # H100: 10 of its 20 cores for model servers (formal_eval.SERVER_CORES, vla_workers.r15_command)
LOCAL_TREE = Path('/home/weiland/projects/openpi_r16_q3_server')
REPO = Path(__file__).resolve().parents[6]
SIDE = 'exp/offline_search/rounds/r17/filter_failed_knob/longseg_off_1022/h100_side.py'
STORE = Path('/home/weiland/trace_runs/offline_search_store')
FITS = Path('/home/weiland/projects/openpi_ext/r17_q3_port_1020')
DEFS_ROOT = Path('/home/weiland/projects/openpi_ext/r17_lo_1022')
EXCLUDE = ['--exclude=/.venv', '--exclude=.local_locks/', '--exclude=__pycache__/', '--exclude=*.pyc', '--exclude=.git/',
           '--exclude=*.pkl', '--exclude=*.npz', '--exclude=*.npy', '--exclude=*.pt', '--exclude=*.jsonl', '--exclude=*.log']
# weilandserver path -> its target on the H100 (the link itself is created at the same path there).
LINKS = {
    '/home/weiland/trace_runs/offline_search_store': BASE + '/store',
    '/home/weiland/trace_runs/dual_20260923': BASE + '/mirror/home/weiland/trace_runs/dual_20260923',
    '/home/weiland/trace_runs/os_closed_loop': LO + '/os_closed_loop',  # yaml miss.evidence_dir (written by servers)
    '/data/libero_cache': BASE + '/mirror/data/libero_cache',
    str(FITS): LO + '/r17_q3_port_1020',
    str(DEFS_ROOT): LO + '/r17_lo_1022',
    TREE + '/.venv': VENV,
}

DAEMON = r'''
import json, subprocess, sys, time
from pathlib import Path
token, port, path, read_only = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
root = Path('/tmp') / token
root.mkdir(mode=0o700)
(root / 'rsyncd.conf').write_text('use chroot = no\npid file = %s/pid\nlog file = %s/log\nlock file = %s/lock\n'
    '[%s]\npath = %s\nread only = %s\nlist = no\n' % (root, root, root, token, path, read_only))
subprocess.run(['rsync', '--daemon', '--port=' + port, '--config=%s/rsyncd.conf' % root], check=True, timeout=10,
               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for _ in range(100):
    if (root / 'pid').exists() and (root / 'pid').read_text().strip():
        break
    time.sleep(.05)
print(json.dumps(dict(pid=int((root / 'pid').read_text()), root=str(root))))
'''
STOP = r'''
import os, shutil, signal, sys
from pathlib import Path
pid, root = int(sys.argv[1]), Path(sys.argv[2])
try:
    if ('--config=%s/rsyncd.conf' % root).encode() in Path('/proc/%d/cmdline' % pid).read_bytes().split(bytes(1)):
        os.kill(pid, signal.SIGTERM)
except (FileNotFoundError, ProcessLookupError):
    pass
shutil.rmtree(root, ignore_errors=True)
'''
SETUP = r'''
import json, os, sys
from pathlib import Path
links, lo, dirs = json.loads(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3])
for d in dirs:
    Path(d).mkdir(parents=True, exist_ok=True)
for link, target in links.items():
    if target.startswith(lo + '/'):
        Path(target).mkdir(parents=True, exist_ok=True)
    p = Path(link)
    if p.is_symlink():
        if os.readlink(p) == target:
            continue
        p.unlink()
    elif p.exists():
        raise SystemExit('%s exists on the H100 and is not a link' % link)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.symlink_to(target)
print('H100_LINKS_OK')
'''


def remote(cmd, timeout=300):
    from exp.offline_search.closed_loop.ops.h100.control import remote as tether_exec
    return tether_exec('h100', list(map(str, cmd)), timeout=timeout)


def rsync(*args):
    subprocess.run(['rsync', '-a', f'--bwlimit={BWLIMIT}', '--timeout=90', '--contimeout=15', *map(str, args)], check=True)


@contextmanager
def daemon(path, read_only):
    """A token-named rsync module for `path` on the H100 for the duration of one copy."""
    with open('/tmp/fa_lo_h100_rsync.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        token = 'lo' + uuid.uuid4().hex
        server = json.loads(remote(['python3', '-c', DAEMON, token, PORT, path, 'yes' if read_only else 'no']).splitlines()[-1])
        try:
            yield f'rsync://{IP}:{PORT}/{token}/'
        finally:
            remote(['python3', '-c', STOP, server['pid'], server['root']])


# ------------------------------------------------------------------------------------------- copies
def code():
    """Code only (as the 1020 copy was made): the R16 Q3 server tree and this package's H100 helper."""
    remote(['mkdir', '-p', TREE])
    with daemon(BASE, False) as url:
        rsync(*EXCLUDE, f'{LOCAL_TREE}/', url + 'openpi_r16_q3_server/')
        rsync('-R', f'{REPO}/./{SIDE}', url + 'openpi_r16_q3_server/')
    print(remote(['python3', '-c', SETUP, json.dumps({TREE + '/.venv': VENV}), LO, '[]']).splitlines()[-1])
    print('H100_CODE_OK', TREE)


def inputs(group):
    """Every file the residents of `group` open, as (local root, relative paths, H100 destination under BASE)."""
    root = Path(group['root']); rows = [e['row'] for e in group['entries']]
    fits = sorted({r['plugin_args'][r['plugin_args'].index('--os-fit-artifact')+1] for r in rows})
    store = set()
    for key in sorted({r['cell'].rsplit('_', 1)[0] for r in rows}):  # what the R15 H100 sync sends (assets.build_plan)
        library = STORE/'library'/key
        store |= {f'library/{key}/current/{n}' for n in ('action.npy', 'task_id.npy', 'ids.json')}
        for d in sorted(library.iterdir()):
            if d.is_dir() and d.name != 'current' and not d.name.startswith('_') and '.' not in d.name and (d/'key_v0.npy').exists():
                store |= {f'library/{key}/{d.name}/{n}' for n in ('task_id.npy', 'action.npy', 'key_v0.npy')}
    for cell in sorted({r['cell'] for r in rows}):
        if (STORE/'queries'/cell/'episodes.json').exists():
            store.add(f'queries/{cell}/episodes.json')
    pkls = sorted({yaml.safe_load(Path(r['yaml']).read_text())['backend']['in_memory']['preload_path'] for r in rows})
    defs = sorted({str(p.relative_to(DEFS_ROOT)) for e in group['entries'] for p in Path(e['source_root']).iterdir() if p.is_file()})
    return [(FITS, [str(Path(f).relative_to(FITS)) for f in fits], 'lo_1022/r17_q3_port_1020/'),
            (DEFS_ROOT, defs, 'lo_1022/r17_lo_1022/'),
            (root, ['group.json', 'ROOT.json'], f'lo_1022/eval/{root.name}/'),
            (STORE, sorted(store), 'store/'),  # current/manifest.json stays as the R15 sync wrote it there
            (Path('/'), [p.lstrip('/') for p in pkls], 'mirror/')]


def plan(group_path):
    group = json.loads(Path(group_path).read_text()); total = 0
    for source, rels, dest in inputs(group):
        for rel in rels:
            path = source/rel
            size = sum(p.stat().st_size for p in path.rglob('*') if p.is_file()) if path.is_dir() else path.stat().st_size
            total += size
            print(f'{size:>14} {path} -> {BASE}/{dest}{rel}')
    print(f'TOTAL {total} bytes ({total/2**30:.2f} GiB) before rsync skips files already identical (size+mtime) on the H100')


def sync(group_path):
    """prepare: code, the H100 links (same paths as here), the group's inputs; nothing is started."""
    group = json.loads(Path(group_path).read_text()); name = Path(group['root']).name
    code()
    links = dict(LINKS, **{group['root']: f'{LO}/eval/{name}'})
    print(remote(['python3', '-c', SETUP, json.dumps(links), LO, json.dumps([f'{LO}/eval/{name}'])]).splitlines()[-1])
    with daemon(BASE, False) as url, tempfile.TemporaryDirectory() as td:
        for i, (source, rels, dest) in enumerate(inputs(group)):
            listing = Path(td)/f'{i}.txt'; listing.write_text(''.join(r + '\n' for r in rels))
            # -L: regular files on the H100 even where this side links (the /nvme aliases)
            rsync('-L', f'--files-from={listing}', f'{source}/', url + dest)
    print('H100_SYNC_OK', name)


def pull(group_path):
    """The residents' logs (decisions, startup, resident records) back into the group root; skips caches."""
    group = json.loads(Path(group_path).read_text()); root = Path(group['root'])
    with daemon(f'{LO}/eval/{root.name}', True) as url:
        rsync('--exclude=runtime/', url + 'runs/', f'{root}/runs/')
        rsync(url + 'resident/', f'{root}/resident/')
    print('H100_PULL_OK', root.name)


# ---------------------------------------------------------------------------------------- residents
def side(action, spec, timeout):
    return remote(['python3', '-B', f'{TREE}/{SIDE}', action, json.dumps(spec, sort_keys=True)], timeout).splitlines()[-1]


def start(spec):
    return side('start', spec, 300)


def status(spec):
    return json.loads(side('status', spec, 180))


def stop(spec, timeout_s=None):
    return side('stop', spec, 900)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=('code', 'sync', 'pull', 'plan'))
    p.add_argument('--group')
    a = p.parse_args()
    if a.action == 'code':
        code()
    else:
        {'sync': sync, 'pull': pull, 'plan': plan}[a.action](a.group)


if __name__ == '__main__':
    main()
