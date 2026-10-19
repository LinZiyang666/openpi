"""CPU-only, run-scoped archive stages inside tether's allowed /tmp root.

Python3.8 compatible for the LIBERO island. Source archives remain evidence;
only a SHA-attested stage is removed, after the caller verifies the pull.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid

STAGE_ROOT = Path('/tmp/r17_Q_qualification')
RUN_ROOT = Path('/data/mw_finetune/qualification/runs')
MAX_BYTES = 512 * 1024**2


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def _key(run_key, kind):
    if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,199}', run_key)
            or run_key in ('.', '..') or kind not in ('workers', 'libero')):
        raise ValueError('invalid Q transfer run/kind')


def stage_identity(path, run_key, kind):
    """Validate a remote path lexically; never depend on another host's files."""
    _key(run_key, kind)
    path = Path(path)
    match = re.fullmatch(kind + r'_([0-9a-f]{64})_([0-9a-f]{32})\.tar', path.name)
    if path.parent != STAGE_ROOT / run_key or '..' in path.parts or not match:
        raise ValueError('archive is outside the exact Q run stage')
    return match.group(1)


def verify_stage(path, run_key, kind, expected_sha256):
    path = Path(path)
    if stage_identity(path, run_key, kind) != expected_sha256:
        raise ValueError('stage path SHA differs from transfer receipt')
    if path.resolve() != path or path.is_symlink() or not path.is_file():
        raise ValueError('stage must be a regular file without symlink parents')
    if path.stat().st_size > MAX_BYTES or file_sha256(path) != expected_sha256:
        raise ValueError('staged archive SHA/size mismatch')


def stage_archive(source, run_key, kind, expected_sha256):
    _key(run_key, kind)
    source = Path(source)
    if not re.fullmatch(r'[0-9a-f]{64}', expected_sha256):
        raise ValueError('exact archive SHA required')
    if source.is_symlink() or not source.is_file() or source.stat().st_size > MAX_BYTES:
        raise ValueError('source archive must be a regular bounded file')
    parent = STAGE_ROOT / run_key
    parent.mkdir(parents=True, exist_ok=True)
    if parent.resolve() != parent:
        raise ValueError('Q stage parent may not be a symlink')
    stage = parent / (kind + '_' + expected_sha256 + '_' + uuid.uuid4().hex + '.tar')
    # New names only. A failed copy is retained for diagnosis, never published.
    with stage.open('xb') as target, source.open('rb') as stream:
        shutil.copyfileobj(stream, target, 1024**2)
        target.flush(); os.fsync(target.fileno())
    verify_stage(stage, run_key, kind, expected_sha256)
    receipt = dict(archive=str(stage), source_archive=str(source), archive_sha256=expected_sha256,
        run_key=run_key, kind=kind, stage_sha_verified=True, stage_removed=False)
    stage.with_suffix('.stage.json').write_text(json.dumps(receipt) + '\n')
    return receipt


def cleanup_stage(path, run_key, kind, verified_sha256):
    """Caller supplies the SHA only after its closed archive install succeeds."""
    verify_stage(path, run_key, kind, verified_sha256)
    path = Path(path)
    marker = path.with_suffix('.stage.json')
    receipt = json.loads(marker.read_text())
    if (receipt['archive'] != str(path) or receipt['archive_sha256'] != verified_sha256
            or receipt['run_key'] != run_key or receipt['kind'] != kind):
        raise ValueError('Q stage ownership receipt mismatch')
    path.unlink()
    receipt.update(stage_removed=True, verified_pull_sha256=verified_sha256)
    marker.write_text(json.dumps(receipt) + '\n')
    return dict(status='PASS', **receipt)


def install_relay_archive(path, destination, run_key, kind, expected_sha256):
    """H100 CPU copy from an allowed push stage into the exact durable Q run."""
    verify_stage(path, run_key, kind, expected_sha256)
    destination = Path(destination)
    if destination != RUN_ROOT / run_key / (kind + '.tar'):
        raise ValueError('relay destination outside the exact Q run')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.resolve() != destination.parent:
        raise ValueError('relay destination parent may not be a symlink')
    with destination.open('xb') as target, Path(path).open('rb') as stream:
        shutil.copyfileobj(stream, target, 1024**2)
        target.flush(); os.fsync(target.fileno())
    if file_sha256(destination) != expected_sha256:
        raise ValueError('relay durable archive SHA mismatch')
    # The durable pulled copy is now SHA-verified; retain it for P6 installation.
    Path(path).unlink()
    receipt = dict(status='PASS', archive=str(destination), archive_sha256=expected_sha256,
        incoming_stage=str(path), incoming_stage_removed=True, run_key=run_key, kind=kind,
        cpu_only=True, gpu_executed=False)
    destination.with_suffix('.relay.json').write_text(json.dumps(receipt) + '\n')
    return receipt


def main():
    p = argparse.ArgumentParser()
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--stage'); modes.add_argument('--cleanup'); modes.add_argument('--install')
    p.add_argument('--destination'); p.add_argument('--run-key', required=True)
    p.add_argument('--kind', choices=('workers', 'libero'), required=True)
    p.add_argument('--sha256', required=True)
    a = p.parse_args()
    if a.stage: receipt = stage_archive(a.stage, a.run_key, a.kind, a.sha256)
    elif a.cleanup: receipt = cleanup_stage(a.cleanup, a.run_key, a.kind, a.sha256)
    else: receipt = install_relay_archive(a.install, a.destination, a.run_key, a.kind, a.sha256)
    print(json.dumps(receipt))


if __name__ == '__main__': main()
