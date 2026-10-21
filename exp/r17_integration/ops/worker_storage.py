"""Sealed worker evidence replaces only its exact, closed campaign run copy.

PASS campaigns only: failed, pending and resumable collections keep their
remote sources. The local archive and installed evidence are always retained.
"""
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import stat
import subprocess
import tarfile

from .formal_process import alive, ticks

REMOTE_RUNS = '/scratch/zixuans8/mw_r17/formal/runs'
LOCAL_RUNS = '/nvme/weiland/openpi/qualification/local_workers/runs'  # local_workers.ROOT/runs (owner 10-20: off /data)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def remove_closed_run(proof):
    """Validate the entire tree before using non-recursive rm and rmdir.

    The small proof is passed by value; a potentially huge per-file manifest
    is read from the already SHA-verified remote archive, never placed in argv.
    """
    node = proof['node']
    root = Path(LOCAL_RUNS if node == 'local' else REMOTE_RUNS)
    if node not in ('local','timan107','timan108','timan1'):
        raise ValueError('unknown worker cleanup node')
    run = Path(proof['run'])
    name = proof['run_name']+'_'+node
    if (not proof['run_name'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in proof['run_name'])
            or run != root/name or root.resolve() != root):
        raise ValueError('worker cleanup outside exact campaign runs path')
    archive = Path(str(run)+'.tar')
    source = Path(proof['source_archive'])
    fallback = Path('/tmp/r17_formal')/proof['run_name']/node/'final_evidence_workers.tar.gz'
    if source not in (archive,fallback):
        raise ValueError('worker cleanup source archive outside exact campaign')
    if (proof.get('sealed_status') != 'PASS' or not proof.get('sealed_at') or
            proof.get('evidence_verified') is not True or not proof.get('owners') or
            any(alive(owner) for owner in proof['owners'])):
        raise ValueError('worker cleanup requires sealed PASS and gone owners')
    if not run.exists() and not archive.exists():
        return dict(status='ALREADY_ABSENT',removed_paths=[],files_removed=0,bytes_removed=0)
    for path in (run,archive,source,Path(str(run)+'.transfer.json')):
        if path.is_symlink() or path.resolve() != path:
            raise ValueError('worker cleanup symlink path')
        if path.exists() and path.stat().st_uid != os.getuid():
            raise ValueError('worker cleanup foreign owner')
    if (not source.is_file() or source.stat().st_size != proof['archive_size_bytes'] or
            file_sha(source) != proof['archive_sha256']):
        raise ValueError('worker cleanup archive SHA/size mismatch')
    # A fallback archive must not authorize deleting a different old tar.
    if archive.exists() and (archive.stat().st_size != proof['archive_size_bytes'] or
                            file_sha(archive) != proof['archive_sha256']):
        raise ValueError('worker cleanup old archive differs; retain it')
    with tarfile.open(source) as tar:
        manifest = json.load(tar.extractfile('TRANSFER_SHA256.json'))
        fleet = json.load(tar.extractfile('receipt.json'))
    encoded = (json.dumps({k:v for k,v in manifest.items() if k != 'digest'},sort_keys=True,
                         separators=(',',':'),allow_nan=False)+'\n').encode()
    if (hashlib.sha256(encoded).hexdigest() != manifest['digest'] or
            manifest['digest'] != proof['manifest_digest'] or manifest.get('partials') or
            manifest['files'].get('manifest.json') != proof['campaign_manifest_sha256']):
        raise ValueError('worker cleanup archive manifest mismatch/partials')
    owner = fleet.get('fleet_owner')
    if (not owner or not any(owner.get('pid') == o['pid'] and owner.get('start_ticks') == o['start_ticks']
                             for o in proof['owners']) or any(alive(p) for p in fleet.get('pids',[]))):
        raise ValueError('worker cleanup fleet ownership/liveness differs')
    files, directories = [], []
    lock = None
    sync_lock = None
    if run.exists():
        # Opening an existing lock never creates a new file in sealed evidence.
        lock_path = run/'.eval.lock'
        if not lock_path.is_file() or lock_path.is_symlink():
            raise ValueError('worker cleanup lock missing/unsafe')
        lock = lock_path.open('rb')
        try:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise ValueError('worker cleanup run still busy')
        sync_path=run/'.incremental_sync.lock'
        if sync_path.exists():
            if not sync_path.is_file() or sync_path.is_symlink():
                lock.close()
                raise ValueError('worker cleanup incremental sync lock unsafe')
            sync_lock=sync_path.open('rb')
            try:fcntl.flock(sync_lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                sync_lock.close();lock.close()
                raise ValueError('worker cleanup incremental sync still busy')
    try:
        if run.exists():
            found = {}
            for directory,dirs,names in os.walk(run,followlinks=False):
                directories.append(Path(directory))
                if not os.access(directory,os.W_OK | os.X_OK):
                    raise ValueError('worker cleanup directory is not writable: '+directory)
                for name in dirs+names:
                    p = Path(directory)/name
                    s = p.lstat()
                    if s.st_uid != os.getuid() or not (stat.S_ISDIR(s.st_mode) or stat.S_ISREG(s.st_mode)):
                        raise ValueError('worker cleanup nonregular/foreign tree entry: '+str(p))
                    if s.st_dev != run.stat().st_dev:
                        raise ValueError('worker cleanup filesystem boundary')
                    if stat.S_ISREG(s.st_mode):
                        files.append(p)
                        relative = str(p.relative_to(run))
                        if relative not in ('.eval.lock','.incremental_sync.lock','TRANSFER_SHA256.json'):
                            found[relative] = file_sha(p)
            if found != manifest['files']:
                raise ValueError('worker cleanup run changed since verified snapshot')
            extra_manifest = run/'TRANSFER_SHA256.json'
            if extra_manifest.exists() and json.loads(extra_manifest.read_text()) != manifest:
                raise ValueError('worker cleanup extra manifest differs')
        sidecar = Path(str(run)+'.transfer.json')
        if sidecar.exists():
            value = json.loads(sidecar.read_text())
            if (value.get('archive_sha256') != proof['archive_sha256'] or
                    value.get('manifest') != manifest):
                raise ValueError('worker cleanup transfer sidecar differs')
        # Recheck owners just before deletion; never signal a process.
        if any(alive(p) for p in proof['owners']+fleet.get('pids',[])):
            raise ValueError('worker cleanup owner became live')
        before = os.statvfs(root)
        extra = [p for p in (archive,sidecar) if p.exists()]
        removed_bytes = sum(p.stat().st_size for p in files+extra)
        for start in range(0,len(files),128):
            subprocess.run(['rm','--',*[str(p) for p in files[start:start+128]]],check=True,
                           capture_output=True,text=True)
        for p in sorted(directories,key=lambda p:len(p.parts),reverse=True):
            subprocess.run(['rmdir','--',str(p)],check=True,capture_output=True,text=True)
        if extra:
            subprocess.run(['rm','--',*[str(p) for p in extra]],check=True,capture_output=True,text=True)
        after = os.statvfs(root)
        return dict(status='REMOVED',removed_paths=[str(run)]+[str(p) for p in extra],
            files_removed=len(files)+len(extra),bytes_removed=removed_bytes,
            free_bytes_before=before.f_bavail*before.f_frsize,
            free_bytes_after=after.f_bavail*after.f_frsize,archive_sha256=proof['archive_sha256'],
            archive_size_bytes=proof['archive_size_bytes'],manifest_digest=manifest['digest'])
    finally:
        if sync_lock is not None:sync_lock.close()
        if lock is not None:
            lock.close()


def cleanup(campaign, directory, deployment, report):
    """Only called after durable receipt sealing; failed campaigns are retained."""
    from . import formal_eval as launch
    from .formal_finalize import closed_proof, durable_json
    from .sync_back import install_snapshot
    directory = Path(directory)
    published = json.loads((directory/'RECEIPT.json').read_text())
    reference = published.get('scientific_evidence_seal')
    if reference:
        if (published.get('status') not in ('PASS','PENDING_EVIDENCE') or
                published.get('all_owned_closed') is not True or published.get('evidence_verified') is not True):
            raise ValueError('worker cleanup public receipt is failed/unverified')
        witness = directory/'SEALED_EVIDENCE_RECEIPT.json'
        if reference['path'] != str(witness) or file_sha(witness) != reference['sha256']:
            raise ValueError('worker cleanup scientific seal receipt changed')
        sealed = json.loads(witness.read_text())
    else:
        sealed = published
    if (sealed.get('status') != 'PASS' or not sealed.get('sealed_at') or
            sealed.get('evidence_verified') is not True or sealed.get('evidence_pending')):
        raise ValueError('worker cleanup requires a durable sealed PASS receipt')
    fleets = closed_proof(sealed)
    for node,closure in fleets.items():
        if report.get('worker_storage_cleanup',{}).get(node,{}).get('status') in ('REMOVED','ALREADY_ABSENT'):
            continue
        try:
            item = deployment[node]
            physical_node=item.get('node',node)
            worker_campaign=item.get('fleet_campaign',campaign)
            value = sealed['transfers'][node]['remote']
            archive = directory/(node+'.tar')
            if (value.get('archive_size_bytes') is None or archive.stat().st_size != value['archive_size_bytes']
                    or file_sha(archive) != value['archive_sha256']):
                raise ValueError('worker cleanup local SHA/size not attested')
            verified = install_snapshot(archive,directory/'workers'/node,expected_sha256=value['archive_sha256'])
            if verified['partials'] or verified['digest'] != value['manifest']['digest']:
                raise ValueError('worker cleanup local install not verified')
            owners = [closure[k] for k in ('child','supervisor') if closure.get(k)]
            proof = dict(node=physical_node,run=item['worker_run'],run_name=worker_campaign['run_name'],owners=owners,
                source_archive=value.get('source_archive',item['worker_run']+'.tar'),
                sealed_status=sealed['status'],sealed_at=sealed['sealed_at'],evidence_verified=True,
                archive_sha256=value['archive_sha256'],archive_size_bytes=value['archive_size_bytes'],
                manifest_digest=value['manifest']['digest'],
                campaign_manifest_sha256=value['manifest']['files']['manifest.json'])
            expected_root = LOCAL_RUNS if physical_node == 'local' else REMOTE_RUNS
            if (not owners or physical_node not in ('local','timan107','timan108','timan1') or
                    proof['run'] != expected_root+'/'+worker_campaign['run_name']+'_'+physical_node):
                raise ValueError('worker cleanup exact path/owners not attested')
            code = ('import os,json,stat,subprocess,hashlib,tarfile,fcntl; from pathlib import Path\n'+
                'REMOTE_RUNS='+repr(REMOTE_RUNS)+'; LOCAL_RUNS='+repr(LOCAL_RUNS)+'\n'+
                inspect.getsource(ticks)+inspect.getsource(alive)+inspect.getsource(file_sha)+
                inspect.getsource(remove_closed_run)+'\nprint(json.dumps(remove_closed_run('+repr(proof)+')))')
            result = json.loads(launch.remote(physical_node,[item['python'],'-c',code],seconds=300,replay_safe=True))
            result['local_archive_retained'] = str(archive)
        except Exception as exc:
            result = dict(status='RETAINED',reason=repr(exc),stderr=getattr(exc,'stderr',None))
        report.setdefault('worker_storage_cleanup',{})[node] = result
        durable_json(directory/'RECEIPT.json',report)
        print('worker storage cleanup '+node+': '+json.dumps(result,sort_keys=True),flush=True)
    return report
