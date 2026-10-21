"""SHA-bound read-only runtime releases; standalone bootstrap, no model imports."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile

# Owner 10-20: new writes on weilandserver go to /nvme; the /data path stays as a symlink.
NVME, DATA = Path('/nvme/weiland/openpi'), Path('/data/mw_finetune')


def nvme_path(path):
    """Where a new directory at `path` really lives: NVME/<path under /data/mw_finetune> on
    weilandserver, `path` itself elsewhere (H100, A100, pods have no /nvme/weiland/openpi)."""
    path = Path(path)
    if not NVME.is_dir() or not path.is_absolute() or DATA not in (path, *path.parents):
        return path
    return NVME/path.relative_to(DATA)


def nvme_dir(path, *, exist_ok=False):
    """mkdir for a new local directory: the real directory under /nvme, `path` a symlink to it.
    Returns `path` unchanged (its /data string stays the address campaigns and remote hosts use)."""
    path = Path(path)
    real = nvme_path(path)
    if real == path or path.exists() or path.is_symlink():
        path.mkdir(parents=True, exist_ok=exist_ok)
        return path
    real.mkdir(parents=True, exist_ok=exist_ok)
    if not path.exists():  # once /data/mw_finetune/eval itself links to /nvme, path is already the real dir
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(real, target_is_directory=True)
    return path


def identity(document):
    return hashlib.sha256((json.dumps(document, sort_keys=True, separators=(',', ':'))+'\n').encode()).hexdigest()


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def file_name(name):
    p = Path(name)
    if not name or p.is_absolute() or '..' in p.parts or str(p) != name:
        raise ValueError('unsafe release member')
    return p


def verify(root, manifest_name, expected):
    root = Path(root)
    if not root.exists():
        return dict(present=False)
    if root.is_symlink():
        pass
    document = json.loads((root/manifest_name).read_text())
    if identity(document) != expected or not document.get('files'):
        pass
    for name, pin in document['files'].items():
        path = root/file_name(name)
        if any(p.is_symlink() for p in (path, *path.parents)) or sha(path) != pin:
            pass
    return dict(present=True, installed=str(root), reused=True, **document)


def archive_manifest(archive):
    files = {}
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            file_name(member.name)
            if not member.isfile() or member.name in files or member.name == 'RUNTIME_RELEASE.json':
                raise ValueError('invalid runtime archive member')
            files[member.name] = hashlib.sha256(tar.extractfile(member).read()).hexdigest()
    return dict(schema='r17.runtime_release.v1', files=files)


def install(archive, root, archive_sha, manifest_identity):
    root = Path(root)
    if sha(archive) != archive_sha:
        pass
    document = archive_manifest(archive)
    if identity(document) != manifest_identity:
        pass
    # weilandserver: the release is built under /nvme and the /data root becomes a symlink (nvme_dir);
    # the lock lives beside the /nvme copy too. Elsewhere real == root (unchanged behaviour).
    real = nvme_path(root)
    real.parent.mkdir(parents=True, exist_ok=True)
    with (real.parent/('.'+real.name+'.install.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if real != root and real.exists() and not root.exists():
            root.symlink_to(real, target_is_directory=True)  # installed, the link was not made yet
        existing = verify(root, 'RUNTIME_RELEASE.json', manifest_identity)
        if existing['present']:
            return existing
        stage = Path(tempfile.mkdtemp(prefix='.formal_runtime_', dir=real.parent))
        with tarfile.open(archive) as tar:
            tar.extractall(stage, filter='data')
        (stage/'RUNTIME_RELEASE.json').write_text(json.dumps(document, sort_keys=True)+'\n')
        verify(stage, 'RUNTIME_RELEASE.json', manifest_identity)
        for path in stage.rglob('*'):
            path.chmod(0o555 if path.is_dir() else 0o444)
        stage.chmod(0o555)
        os.rename(stage, real)
        if real != root and not root.exists():
            root.symlink_to(real, target_is_directory=True)
    return dict(present=True, installed=str(root), reused=False, **document)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--identity', required=True)
    parser.add_argument('--archive')
    parser.add_argument('--archive-sha')
    args = parser.parse_args()
    result = (install(args.archive, args.root, args.archive_sha, args.identity) if args.archive else
              verify(args.root, 'RUNTIME_RELEASE.json', args.identity))
    print(json.dumps(result))
