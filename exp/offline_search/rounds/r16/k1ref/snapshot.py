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
RESTORE = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def build(destination):
    destination = destination.resolve()
    if destination.exists(): raise FileExistsError('Use a fresh snapshot directory')
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
            shutil.copyfile(source, target)
        else: target = source
        checks.append((sha(target), target))
    (destination / '.venv').symlink_to(REPO / '.venv', target_is_directory=True)
    # emit_arms discovers the repository and legacy CPU emitter tests read its
    # ignored YAML templates. Copy the templates without changing their bytes.
    shutil.copytree(REPO / 'exp/trace_dual/config', destination / 'exp/trace_dual/config', dirs_exist_ok=True)
    code = destination / R16 / 'k1ref'
    for name, digest in json.loads((code / 'CODE_LOCK.json').read_text()).items():
        target = code / name
        checks.append((sha(target), target))
    checks.append((sha(code / 'CODE_LOCK.json'), code / 'CODE_LOCK.json'))
    manifest = destination / 'R16_SNAPSHOT.sha256'
    manifest.write_text(''.join(f'{h}  {p}\n' for h, p in checks))
    (destination / 'R16_SNAPSHOT.json').write_text(json.dumps(dict(head=head, source=str(REPO), restored=RESTORE,
        old_manifest_sha256=sha(REPO / R16 / 'conductor_deployment/local.sha256'), entries=len(checks)), indent=2) + '\n')
    verify(destination)


def verify(destination):
    for line in (destination / 'R16_SNAPSHOT.sha256').read_text().splitlines():
        digest, path = line.split('  ', 1)
        if not Path(path).is_file(): raise FileNotFoundError(path)
    print('R16_SNAPSHOT_OK: all listed inputs are present')


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('action', choices=('build', 'verify'))
    ap.add_argument('--destination', type=Path, required=True); args = ap.parse_args()
    (build if args.action == 'build' else verify)(args.destination)


if __name__ == '__main__': main()
