"""Atomic, content-addressed worker installs; existing releases are read-only."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@contextmanager
def install_lock(destination):
    destination = Path(destination)
    locks = destination.parent / '.install_locks'
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / (destination.name+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def freeze(directory):
    for path in directory.rglob('*'):
        if not path.is_symlink():
            path.chmod(0o555 if path.is_dir() else 0o444)
    directory.chmod(0o555)


def install_client(archive, destination, expected):
    from .build_client_bundle import verified_members
    from .deploy import install_payload
    archive, destination = Path(archive), Path(destination)
    if sha(archive) != expected:
        raise ValueError('client archive SHA mismatch')
    manifest, _ = verified_members(archive, 'PAYLOAD.json')
    with install_lock(destination):
        if destination.exists():
            if json.loads((destination/'PAYLOAD.json').read_text()) != manifest:
                raise ValueError('different immutable client release already exists')
            for name, pin in manifest['files'].items():
                if sha(destination/name) != pin:
                    raise ValueError('immutable client file changed: '+name)
        else:
            stage = Path(tempfile.mkdtemp(prefix='.client_', dir=destination.parent))
            install_payload(archive, stage, expected)
            freeze(stage)
            os.rename(stage, destination)
    return manifest


def asset_manifest(document):
    if not isinstance(document, dict):
        document = json.loads(Path(document).read_text())
    if set(document) != {'schema', 'files', 'digest'} or document['schema'] != 'r17.libero_original_assets.v1' or not document['files']:
        raise ValueError('original LIBERO asset manifest required')
    sealed = {key:document[key] for key in ('schema', 'files')}
    identity = hashlib.sha256(json.dumps(sealed, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if identity != document['digest']:
        raise ValueError('original LIBERO asset manifest digest mismatch')
    for name in document['files']:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or str(path) != name or name == '.':
            raise ValueError('unsafe original LIBERO asset name')
    return document


def verify_assets(root, document):
    root = Path(root)
    # Hub download metadata is not simulation data and is never copied.
    files = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and
             not {'.cache', '.git', '__pycache__'} & set(p.relative_to(root).parts)}
    if files != set(document['files']):
        raise ValueError('original LIBERO asset file set differs')
    for name, pin in document['files'].items():
        if sha(root/name) != pin:
            raise ValueError('original LIBERO asset SHA mismatch: '+name)


def provision_assets(source, store, document):
    """Copy verified local originals, never import a downloader or edit caches."""
    document = asset_manifest(document)
    destination = Path(store)/('libero_'+document['digest'])
    with install_lock(destination):
        if destination.is_symlink():
            raise ValueError('symlink original LIBERO asset release')
        if destination.exists():
            if json.loads((destination/'ASSETS.json').read_text()) != document:
                raise ValueError('immutable original LIBERO asset release differs')
        else:
            source = Path(source)
            verify_assets(source, document)
            stage = Path(tempfile.mkdtemp(prefix='.assets_', dir=destination.parent))
            assets = stage/'assets'
            for name in document['files']:
                target = assets/name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source/name, target)
            verify_assets(assets, document)
            (stage/'ASSETS.json').write_text(json.dumps(document, sort_keys=True)+'\n')
            freeze(stage)
            os.rename(stage, destination)
        assets = destination/'assets'
        if assets.is_symlink() or any((assets/name).is_symlink() for name in document['files']):
            raise ValueError('symlink inside original LIBERO asset release')
        verify_assets(assets, document)
    return dict(assets=str(assets), assets_digest=document['digest'], asset_release=str(destination))


def install_dependencies(source, store, config_source, *, assets_manifest=None, original_assets=None):
    source, store, config_source = Path(source), Path(store), Path(config_source)
    names = ['libero', 'robosuite', 'bddl', 'easydict', 'easydict.py', 'robomimic',
             'gym', 'gym_notices', 'nltk', 'thop', 'future', 'past']
    files, copied = {}, []
    for name in names:
        root = source/name
        if not root.exists():
            continue
        copied.append(name)
        paths = root.rglob('*') if root.is_dir() else [root]
        for path in paths:
            if '__pycache__' in path.parts or path.suffix == '.pyc':
                continue
            if path.is_file():
                files[str(path.relative_to(source))] = sha(path)
    # The original simulator config remains untouched. Assets are read-only
    # references to the existing R16 pool; local LIBERO lookup avoids downloads.
    import yaml
    config = yaml.safe_load(config_source.read_text())
    assets = Path(config['assets'])
    asset_release = {}
    if assets_manifest is not None:
        candidate = assets if assets.is_dir() else original_assets
        asset_release = provision_assets(candidate, store.parent/'assets', assets_manifest)
        assets = Path(asset_release['assets'])
    elif not assets.is_dir():
        raise ValueError('original LIBERO assets missing; refuse a shared-cache download')
    document = dict(schema='r17.worker_dependencies.v1', files=files, copied=copied,
                    assets=str(assets.resolve()), config_sha256=sha(config_source))
    if asset_release:
        document['assets_digest'] = asset_release['assets_digest']
    identity = hashlib.sha256(json.dumps(document, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    destination = store/('libero311_'+identity)
    with install_lock(destination):
        if destination.exists():
            if json.loads((destination/'DEPENDENCIES.json').read_text()) != document:
                raise ValueError('immutable dependency release differs')
        else:
            stage = Path(tempfile.mkdtemp(prefix='.dependencies_', dir=store))
            for name in copied:
                root = source/name
                if root.is_dir():
                    shutil.copytree(root, stage/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
                else:
                    shutil.copy2(root, stage/name)
            target = stage/'libero/libero/assets'
            if not target.exists():
                target.symlink_to(assets.resolve(), target_is_directory=True)
            elif assets_manifest is not None:
                verify_assets(target, asset_manifest(assets_manifest))
            (stage/'DEPENDENCIES.json').write_text(json.dumps(document, sort_keys=True))
            for name, pin in files.items():
                if sha(stage/name) != pin:
                    raise ValueError('simulator source changed during atomic install: '+name)
            freeze(stage)
            os.rename(stage, destination)
        for name, pin in files.items():
            if sha(destination/name) != pin:
                raise ValueError('immutable simulator dependency changed: '+name)
    return dict(private_dependencies=str(destination), copied=copied, content_identity=identity,
                config_source=str(config_source), config_sha256=document['config_sha256'], **asset_release)


def private_config(source, destination, *, assets=None):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination/'config.yaml'
    data = source.read_bytes()
    if assets is not None:
        import yaml
        config = yaml.safe_load(data)
        config['assets'] = str(Path(assets).resolve())
        data = yaml.safe_dump(config, sort_keys=True).encode()
    if target.exists() and target.read_bytes() != data:
        raise ValueError('campaign simulator config changed')
    if not target.exists():
        with target.open('xb') as stream:
            stream.write(data)
    return dict(path=str(destination), sha256=hashlib.sha256(data).hexdigest())
