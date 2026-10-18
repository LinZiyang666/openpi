"""Promote old archive-SHA runtime addresses to verified content identities.

Only explicit unlaunched validation prepares use this; archived code is unchanged.
"""
import inspect
import json
from pathlib import Path


def metadata(campaign, root):
    from exp.r17_integration.ops.immutable_release import archive_manifest, identity, sha
    archive = Path(root)/'runtime.tar'
    document = archive_manifest(archive)
    pin, archive_pin = identity(document), sha(archive)
    old = campaign.get('runtime_identity')
    if old and old != pin:raise ValueError('runtime identity disagrees with frozen archive')
    if not old:
        address = campaign.get('runtime', '')
        if address and not address.endswith(('formal_runtime_'+archive_pin, 'formal_runtime_'+pin)):
            raise ValueError('old runtime address does not match frozen archive')
        campaign['runtime'] = '/data/mw_finetune/qualification/formal_runtime_'+pin
        campaign['runtime_identity'] = pin
        campaign['validation_original_runtime'] = address
    return dict(node=campaign.get('server_node', 'h100'), root=campaign['runtime'], identity=pin,
                archive_sha256=archive_pin, files=len(document['files']), document=document)


def install_checked(request):
    """Standalone CPU installer: existing releases are verified, never changed."""
    import hashlib
    import json
    import os
    import tarfile
    import tempfile
    import fcntl
    from pathlib import Path
    root, archive = Path(request['root']), Path(request['archive'])
    document = request['document']
    def sha(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):digest.update(block)
        return digest.hexdigest()
    if hashlib.sha256((json.dumps(document, sort_keys=True, separators=(',', ':'))+'\n').encode()).hexdigest() != request['identity']:
        raise ValueError('runtime document identity mismatch')
    def verify(directory):
        if directory.is_symlink() or json.loads((directory/'RUNTIME_RELEASE.json').read_text()) != document:
            raise ValueError('existing runtime manifest mismatch')
        for name, pin in document['files'].items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('unsafe runtime member')
            path = directory/relative
            if any(p.is_symlink() for p in (path, *path.parents)) or sha(path) != pin:
                raise ValueError('runtime file mismatch: '+name)
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent/('.'+root.name+'.install.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if root.exists():
            verify(root)
            return dict(reused=True, root=str(root), identity=request['identity'])
        if sha(archive) != request['archive_sha256']:raise ValueError('runtime archive hash mismatch')
        stage = Path(tempfile.mkdtemp(prefix='.validation_runtime_', dir=root.parent))
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            names = [m.name for m in members]
            if len(names) != len(set(names)) or set(names) != set(document['files']):
                raise ValueError('runtime archive member mismatch')
            for m in members:
                rel = Path(m.name)
                if not m.isfile() or rel.is_absolute() or '..' in rel.parts:raise ValueError('unsafe archive')
                path = stage/rel;path.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(m) as src, path.open('xb') as dst:
                    for block in iter(lambda: src.read(1 << 20), b''):dst.write(block)
        (stage/'RUNTIME_RELEASE.json').write_text(json.dumps(document, sort_keys=True)+'\n')
        verify(stage)
        for p in stage.rglob('*'):p.chmod(0o555 if p.is_dir() else 0o444)
        stage.chmod(0o555)
        os.rename(stage, root)
        return dict(reused=False, root=str(root), identity=request['identity'])


def prepare(launch, campaign, root):
    root = Path(root)
    if not campaign.get('autoscale_validation') or not root.name.startswith('fa_asv_'):
        raise ValueError('runtime promotion is only for explicit validation lanes')
    info = metadata(campaign, root)
    request = dict(info, archive=str(root/'runtime.tar'))
    local = install_checked(request)
    if info['node'] != 'local':
        if info['node'] != 'h100':raise ValueError('unknown validation server node')
        stage = '/tmp/r17_formal/'+campaign['run_name']+'/server'
        launch.remote('h100', ['mkdir', '-p', stage], replay_safe=True)
        launch.transfer(['rsync', 'push', str(root/'runtime.tar'), 'h100:'+stage+'/runtime.tar'])
        request['archive'] = stage+'/runtime.tar'
        code = inspect.getsource(install_checked)+'\nimport json\nprint(json.dumps(install_checked('+repr(request)+')))'
        remote = json.loads(launch.remote('h100', ['/usr/bin/python3', '-c', code], seconds=180, replay_safe=True))
    else:remote = local
    receipt = {k:v for k,v in info.items() if k != 'document'}
    receipt.update(reused=remote['reused'], frozen_archive_unchanged=True)
    (root/'RUNTIME_RELEASE.json').write_text(json.dumps(receipt, indent=2)+'\n')
    campaign['coordinator_runtime'] = campaign['runtime']
    return receipt
