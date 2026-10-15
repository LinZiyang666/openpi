"""Build/verify the local R16 server checkout; no network or model operations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

REPO = Path(__file__).resolve().parents[5]
R16 = Path('exp/offline_search/rounds/r16')
RESTORE = {'packages/openpi-client/src/openpi_client/msgpack_numpy.py': '231a1cf7',
           'src/openpi/cache/groot/batcher.py': '30505063',
           'src/openpi/cache/groot/staged.py': '30505063'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def build(destination):
    destination = destination.resolve()
    if destination.exists(): raise FileExistsError('Use a fresh snapshot directory')
    if destination.parent != REPO.parent: raise ValueError('Place the snapshot under /home/weiland/projects/')
    head = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    subprocess.run(['git', '-C', str(REPO), 'worktree', 'add', '--detach', str(destination), head], check=True)
    # Git-ignored R16 freezes, groups, source locks, fits and results accompany
    # the tracked code. Absolute external fits/groups stay at frozen locations.
    shutil.copytree(REPO / R16, destination / R16, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    checks = []
    for line in (REPO / R16 / 'conductor_deployment/local.sha256').read_text().splitlines():
        digest, source = line.split('  ', 1); source = Path(source)
        if source.is_relative_to(REPO):
            rel = source.relative_to(REPO); target = destination / rel; target.parent.mkdir(parents=True, exist_ok=True)
            if str(rel) in RESTORE:
                target.write_bytes(subprocess.check_output(['git', '-C', str(REPO), 'show', RESTORE[str(rel)] + ':' + str(rel)]))
            else:
                if sha(source) != digest: raise ValueError(f'Unexpected changed R16 source: {source}')
                shutil.copyfile(source, target)
        else: target = source
        if sha(target) != digest: raise ValueError(f'Failed R16 local.sha256: {target}')
        checks.append((digest, target))
    (destination / '.venv').symlink_to(REPO / '.venv', target_is_directory=True)
    # emit_arms discovers the repository and legacy CPU emitter tests read its
    # ignored YAML templates. Copy the templates without changing their bytes.
    shutil.copytree(REPO / 'exp/trace_dual/config', destination / 'exp/trace_dual/config', dirs_exist_ok=True)
    code = destination / R16 / 'k1ref'
    for name, digest in json.loads((code / 'CODE_LOCK.json').read_text()).items():
        target = code / name
        if sha(target) != digest: raise ValueError(f'Changed new code: {name}')
        checks.append((digest, target))
    checks.append((sha(code / 'CODE_LOCK.json'), code / 'CODE_LOCK.json'))
    manifest = destination / 'R16_SNAPSHOT.sha256'
    manifest.write_text(''.join(f'{h}  {p}\n' for h, p in checks))
    (destination / 'R16_SNAPSHOT.json').write_text(json.dumps(dict(head=head, source=str(REPO), restored=RESTORE,
        old_manifest_sha256=sha(REPO / R16 / 'conductor_deployment/local.sha256'), entries=len(checks)), indent=2) + '\n')
    # Make measured source bytes read-only. Future results live in the original
    # k1ref output tree, selected through R16_K1REF_OUTPUT.
    for _, path in checks:
        if path.is_relative_to(destination): path.chmod(0o444)
    verify(destination)


def verify(destination):
    for line in (destination / 'R16_SNAPSHOT.sha256').read_text().splitlines():
        digest, path = line.split('  ', 1)
        if sha(path) != digest: raise ValueError(f'Changed immutable snapshot input: {path}')
    print('R16_SNAPSHOT_OK: every conductor_deployment/local.sha256 entry plus new code verifies')


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('action', choices=('build', 'verify'))
    ap.add_argument('--destination', type=Path, required=True); args = ap.parse_args()
    (build if args.action == 'build' else verify)(args.destination)


if __name__ == '__main__': main()
