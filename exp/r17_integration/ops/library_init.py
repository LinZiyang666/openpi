"""Transport only SHA-pinned historical B inputs; no simulator/model import."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import tarfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def build_init_archive(audits, archive):
    files, sources = {}, {}
    for audit in audits:
        if audit['status'] != 'PASS' or audit['init_pool'] != 'B':
            raise ValueError('B input audit required')
        for task in audit['task_files'].values():
            source = Path(task['file'])
            if source.is_symlink() or sha(source) != task['sha256']:
                raise ValueError('historical B input changed after audit')
            name = audit['suite']+'/'+source.name
            files[name], sources[name] = task['sha256'], source
    document = dict(schema='r17.bpool_inputs.v1', init_pool='B', files=files,
                    alignment_digests={a['suite']:a['digest'] for a in audits})
    with tarfile.open(archive, 'x') as tar:
        for name, source in sorted(sources.items()):
            tar.add(source, arcname=name, recursive=False)
        data = json.dumps(document,sort_keys=True).encode()
        info = tarfile.TarInfo('INIT_FILES.json'); info.size = len(data)
        tar.addfile(info,io.BytesIO(data))
    return dict(archive_sha256=sha(archive), manifest=document)


def install_init_archive(archive, destination, expected_sha256):
    """Immutable private installation after /tmp ingress; reject every conflict."""
    archive, destination = Path(archive), Path(destination)
    if sha(archive) != expected_sha256:
        raise ValueError('B input archive SHA mismatch')
    if any(p.is_symlink() for p in (destination,*destination.parents)):
        raise ValueError('symlink B input installation')
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or any(not m.isfile() or Path(m.name).is_absolute() or '..' in Path(m.name).parts for m in members):
            raise ValueError('unsafe B input archive')
        document = json.load(tar.extractfile('INIT_FILES.json'))
        if document.get('schema') != 'r17.bpool_inputs.v1' or document.get('init_pool') != 'B' or set(names) != {'INIT_FILES.json',*document['files']}:
            raise ValueError('B input manifest mismatch')
        contents = {}
        for name, expected in document['files'].items():
            if len(Path(name).parts) != 2 or Path(name).suffix != '.init':
                raise ValueError('unexpected B input member')
            data = tar.extractfile(name).read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError('B input member SHA mismatch')
            contents[name] = data
        for name,data in contents.items():
            expected = document['files'][name]
            target = destination/name
            if target.is_symlink() or not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError('escaped B input installation')
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if sha(target) != expected:
                    raise ValueError('existing B input differs')
            else:
                with target.open('xb') as stream:
                    stream.write(data)
                    stream.flush(); os.fsync(stream.fileno())
    if sha(archive) != expected_sha256:
        raise ValueError('B input archive changed during installation')
    return dict(status='PASS', archive_sha256=expected_sha256, manifest=document,
                directories={suite:str(destination/suite) for suite in document['alignment_digests']})
