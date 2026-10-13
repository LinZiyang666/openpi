"""Verified closed-generation offload over a private bandwidth-limited rsync daemon.

No live file is truncated. Journals, manifests, receipts, recycle/native identity
markers and active generations stay remote. A durable local receipt precedes
remote unlink; the SHA-bound remote index travels in the final snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import secrets
import subprocess
import tarfile
import time

from .formal_process import write_json
from .immutable_release import sha

ROOT=Path('/data/mw_finetune/eval')
DEFAULT_PORT=23197


def remote_operation(request, *, _run_parent=None):
    """Stdlib Python3.8; sent by value, never installs into a running runtime."""
    import fcntl,hashlib,io,json,os,re,stat,subprocess,tarfile,tempfile,time,uuid
    from pathlib import Path
    def digest(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            for b in iter(lambda:stream.read(1024**2),b''):h.update(b)
        return h.hexdigest()
    def signature(path):
        s=path.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
    def alive(entry):
        try:
            fields=Path('/proc/'+str(entry['pid'])+'/stat').read_text().split(') ',1)[1].split()
            return fields[19]==str(entry['start_ticks']) and fields[0]!='Z'
        except (FileNotFoundError,KeyError):return False
    def atomic(path,value):
        # Temporary files outside the run cannot make its final tar partial.
        with tempfile.NamedTemporaryFile(dir=path.parent.parent,prefix='.q_sync_',delete=False) as stream:
            part=Path(stream.name);stream.write((json.dumps(value,sort_keys=True)+'\n').encode());stream.flush();os.fsync(stream.fileno())
        os.replace(str(part),str(path))
        fd=os.open(str(path.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
    run=Path(request['run'])
    node=request['node']
    if (node not in ('timan107','timan108','timan1') or
            run.parent!=Path(_run_parent or '/scratch/zixuans8/mw_r17/formal/runs') or
            not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,199}_'+node,run.name) or
            run.resolve()!=run or (run.exists() and run.stat().st_uid!=os.getuid())):
        raise ValueError('sync refuses an unowned worker run')
    # A successful finalizer may have cleaned the run during rsync. Its
    # verified temporary chunk can still be retired; never recreate the run.
    if not run.exists() and request['action']=='ack':
        chunk=request['chunk'];stage=run.parent/('.q_incremental_'+run.name)
        if (stage.resolve()!=stage or stage.stat().st_uid!=os.getuid() or
                not re.fullmatch('[0-9a-f]{32}',chunk['token']) or
                Path(chunk['archive'])!=stage/(chunk['token']+'.tar.gz') or
                chunk.get('run')!=str(run) or chunk.get('node')!=node or
                chunk['manifest_sha256']!=request['manifest_sha256'] or
                json.loads((stage/(chunk['token']+'.json')).read_text())!=chunk or
                digest(Path(chunk['archive']))!=request.get('verified_archive_sha256') or
                chunk['archive_sha256']!=request.get('verified_archive_sha256') or
                not re.fullmatch('[0-9a-f]{64}',request.get('local_receipt_sha256',''))):
            raise ValueError('cleaned run chunk lacks a bound verified ACK')
        Path(chunk['archive']).unlink();(stage/(chunk['token']+'.json')).unlink()
        return dict(status='PASS',pruned=False,files=len(chunk['files']),bytes_freed=0,
                    run_already_cleaned=True)
    if digest(run/'manifest.json')!=request['manifest_sha256']:
        raise ValueError('worker manifest identity mismatch')
    fleet=json.loads((run/'receipt.json').read_text())
    stage=run.parent/('.q_incremental_'+run.name)
    if stage.is_symlink():raise ValueError('symlink sync stage')
    stage.mkdir(mode=0o700,exist_ok=True)
    with (run/'.incremental_sync.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if request['action']=='stage':
            if fleet.get('status')!='RUNNING' or not alive(fleet['fleet_owner']):
                return dict(status='SKIP',reason='fleet is not live; no writes/deletions')
            # One pending chunk/run; transport failures cannot fill remote disk.
            for pending in sorted(stage.glob('*.json')):
                saved=json.loads(pending.read_text())
                if (saved['run']!=str(run) or saved['manifest_sha256']!=request['manifest_sha256'] or
                        digest(Path(saved['archive']))!=saved['archive_sha256']):
                    raise ValueError('pending chunk binding changed')
                return saved
            paths=[];size=0
            for entry in fleet['pids']:
                if type(entry.get('exit_code')) is not int or alive(entry):continue
                workspace=run/'scratch'/('worker_'+str(entry['index']))/('generation_'+str(entry['generation']))
                native=workspace/'NATIVE_PROCESS.json'
                if native.exists():
                    identity=json.loads(native.read_text())
                    native_entry=identity.get('native',identity)
                    if 'pid' in native_entry and alive(native_entry):continue
                candidates=[run/('worker_'+str(entry['index'])+'_generation_'+str(entry['generation'])+'.log')]
                if workspace.exists():candidates.extend(sorted(workspace.rglob('*')))
                for path in candidates:
                    if (not path.is_file() or path.is_symlink() or path.resolve()!=path or
                            path.suffix in ('.json','.jsonl','.lock','.part','.tmp') or
                            path.stat().st_uid!=os.getuid() or time.time()-path.stat().st_mtime<10):continue
                    length=path.stat().st_size
                    if length==0 or length>request['max_bytes'] or size+length>request['max_bytes']:continue
                    if not stat.S_ISREG(path.stat().st_mode):continue
                    before=signature(path);pin=digest(path)
                    if before!=signature(path):continue
                    paths.append(dict(name=str(path.relative_to(run)),sha256=pin,signature=before,prunable=True));size+=length
            if not paths:return dict(status='SKIP',reason='no immutable closed-generation payload')
            token=uuid.uuid4().hex
            archive=stage/(token+'.tar.gz')
            manifest=dict(schema='r17.incremental_chunk.v1',token=token,run=str(run),node=node,
                manifest_sha256=request['manifest_sha256'],files=paths)
            with tarfile.open(str(archive),'w:gz') as tar:
                for item in paths:
                    tar.add(str(run/item['name']),arcname=item['name'],recursive=False)
                # Atomic root records get versioned local copies, never deletion.
                for name in ('manifest.json','receipt.json','INCREMENTAL_OFFLOAD.json'):
                    source=run/name
                    if not source.exists() or source.stat().st_size>8*1024**2:continue
                    with source.open('rb') as stream:data=stream.read()
                    member=tarfile.TarInfo(name);member.size=len(data)
                    tar.addfile(member,io.BytesIO(data))
                    paths.append(dict(name=name,sha256=hashlib.sha256(data).hexdigest(),prunable=False))
            if any(signature(run/x['name'])!=x['signature'] or digest(run/x['name'])!=x['sha256'] for x in paths if x['prunable']):
                archive.unlink();raise ValueError('closed-generation source changed')
            manifest.update(status='STAGED',archive=str(archive),archive_sha256=digest(archive),bytes=archive.stat().st_size)
            atomic(stage/(token+'.json'),manifest)
            return manifest
        chunk=request['chunk']
        if not re.fullmatch('[0-9a-f]{32}',chunk['token']):raise ValueError('bad chunk token')
        stored=json.loads((stage/(chunk['token']+'.json')).read_text())
        if (chunk.get('run')!=str(run) or chunk.get('node')!=node or
                chunk.get('manifest_sha256')!=request['manifest_sha256'] or
                Path(chunk['archive'])!=stage/(chunk['token']+'.tar.gz') or
                stored!=chunk or digest(Path(chunk['archive']))!=chunk['archive_sha256']):
            raise ValueError('owned chunk identity changed')
        if request['action']=='push':
            url=request['url']
            if not url.startswith('rsync://q_incremental@ziyanglin.com:') or '/../' in url:
                raise ValueError('not the private Q rsync destination')
            password=Path(request['password_file'])
            password_root=Path('/scratch/zixuans8/mw_r17/rsync_stage' if node=='timan1' else '/tmp/r17_q_incremental')
            if (password.parent!=password_root or password.resolve()!=password or
                    password.stat().st_uid!=os.getuid() or password.stat().st_mode & 0o077):
                raise ValueError('not the private Q rsync password file')
            try:
                result=subprocess.run(['rsync','-a','--partial','--timeout=90','--contimeout=15',
                    '--bwlimit='+str(request['bwlimit_kibps']),'--password-file='+str(password),
                    chunk['archive'],url],capture_output=True,timeout=180)
                if result.returncode:raise RuntimeError('rsync chunk transport failed: '+result.stderr.decode()[-1000:])
            finally:password.unlink()
            return dict(status='PUSHED',archive_sha256=chunk['archive_sha256'])
        if request['action']!='ack':raise ValueError('unknown sync action')
        if (request.get('verified_archive_sha256')!=chunk['archive_sha256'] or
                not re.fullmatch('[0-9a-f]{64}',request.get('local_receipt_sha256',''))):
            raise ValueError('no bound verified local ACK')
        # If close has started, keep originals; the final snapshot owns them.
        fleet=json.loads((run/'receipt.json').read_text())
        pruning=fleet.get('status')=='RUNNING' and alive(fleet['fleet_owner'])
        index_path=run/'INCREMENTAL_OFFLOAD.json'
        index=json.loads(index_path.read_text()) if index_path.exists() else dict(
            schema='r17.incremental_offload.v1',manifest_sha256=request['manifest_sha256'],chunks=[],files={})
        record=dict(token=chunk['token'],archive_sha256=chunk['archive_sha256'],
            local_receipt_sha256=request['local_receipt_sha256'],files={x['name']:x['sha256'] for x in chunk['files']})
        if not any(x['token']==record['token'] for x in index['chunks']):index['chunks'].append(record)
        if pruning:
            for item in chunk['files']:
                if not item.get('prunable',True):continue
                path=run/item['name']
                if not path.exists() and index['files'].get(item['name'])==item['sha256']:continue
                if signature(path)!=item['signature'] or digest(path)!=item['sha256']:
                    raise ValueError('refuse changed source deletion')
                index['files'][item['name']]=item['sha256']
            atomic(index_path,index)  # durable local bytes/index before unlink
            for item in chunk['files']:
                if not item.get('prunable',True):continue
                path=run/item['name']
                if path.exists():path.unlink()
        # Never add or change snapshot inputs after a fleet begins closing.
        Path(chunk['archive']).unlink();(stage/(chunk['token']+'.json')).unlink()
        return dict(status='PASS',pruned=pruning,files=len(chunk['files']),bytes_freed=sum(x['signature'][2] for x in chunk['files'] if x.get('prunable',True)) if pruning else 0)


def verify_chunk(chunk,archive,directory):
    """Fsync each local file and receipt before any remote ACK is possible."""
    archive,directory=Path(archive),Path(directory)
    if sha(archive)!=chunk['archive_sha256']:raise ValueError('incremental archive SHA mismatch')
    expected={x['name']:x['sha256'] for x in chunk['files']}
    prunable={x['name']:x.get('prunable',True) for x in chunk['files']}
    target=directory/'files';target.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as tar:
        members=tar.getmembers()
        if len(members)!=len(expected) or {x.name for x in members}!=set(expected):raise ValueError('chunk member census mismatch')
        for member in members:
            path=Path(member.name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts:raise ValueError('unsafe chunk member')
            destination=(target if prunable[member.name] else directory/'versions'/chunk['token'])/path
            if any(p.is_symlink() for p in (destination,*destination.parents)):raise ValueError('symlink local destination')
            destination.parent.mkdir(parents=True,exist_ok=True)
            data=tar.extractfile(member).read()
            if hashlib.sha256(data).hexdigest()!=expected[member.name]:raise ValueError('chunk file SHA mismatch')
            if destination.exists() and sha(destination)!=expected[member.name]:raise ValueError('existing local evidence differs')
            if not destination.exists():
                with destination.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
            fd=os.open(destination.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
    receipt=directory/'records'/(chunk['token']+'.json')
    write_json(receipt,dict(status='PASS',chunk=chunk,files=expected,archive_sha256=sha(archive)))
    with receipt.open('rb') as stream:os.fsync(stream.fileno())
    fd=os.open(receipt.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
    return receipt


def restore(directory,node):
    """Join verified offloaded diagnostics to the closed worker snapshot."""
    directory=Path(directory);worker=directory/'workers'/node
    index=worker/'INCREMENTAL_OFFLOAD.json'
    if not index.exists():return dict(status='NOT_USED')
    document=json.loads(index.read_text())
    if document['manifest_sha256']!=sha(worker/'manifest.json'):raise ValueError('incremental worker manifest mismatch')
    base=directory/'INCREMENTAL_SYNC'/node
    for chunk in document['chunks']:
        receipt=base/'records'/(chunk['token']+'.json')
        if sha(receipt)!=chunk['local_receipt_sha256']:raise ValueError('incremental local receipt changed')
        r=json.loads(receipt.read_text())
        if r['status']!='PASS' or r['archive_sha256']!=chunk['archive_sha256'] or r['files']!=chunk['files']:
            raise ValueError('incremental chunk binding changed')
        for item in r['chunk']['files']:
            source=(base/'files' if item.get('prunable',True) else base/'versions'/chunk['token'])/item['name']
            if sha(source)!=item['sha256']:raise ValueError('incremental preserved chunk file changed')
    import shutil
    for name,pin in document['files'].items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts:raise ValueError('unsafe offload file')
        source=base/'files'/path;dest=worker/path
        if sha(source)!=pin:raise ValueError('offloaded local file changed')
        if dest.exists() and sha(dest)!=pin:raise ValueError('closed worker file conflicts')
        if any(p.is_symlink() for p in (dest,*dest.parents)):raise ValueError('symlink restore destination')
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
            with dest.open('rb') as stream:os.fsync(stream.fileno())
            fd=os.open(dest.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
    return dict(status='PASS',files=len(document['files']),index_sha256=sha(index))


def setup(directory,port):
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=False)
    if not directory.is_relative_to(ROOT):raise ValueError('sync state must be under /data eval')
    password=secrets.token_hex(32)
    secret=directory/'rsync.secrets';secret.write_text('q_incremental:'+password+'\n');secret.chmod(0o600)
    password_file=directory/'rsync.password';password_file.write_text(password+'\n');password_file.chmod(0o600)
    inbox=directory/'inbox';inbox.mkdir()
    config=directory/'rsyncd.conf'
    config.write_text('pid file = '+str(directory/'rsync.pid')+'\nlog file = '+str(directory/'rsync.log')+
        '\nuse chroot = no\nuid = '+str(os.getuid())+'\ngid = '+str(os.getgid())+
        '\nmax connections = 2\ntimeout = 120\n[r17q]\npath = '+str(inbox)+'\nread only = no\nauth users = q_incremental\nsecrets file = '+str(secret)+'\n')
    write_json(directory/'CONFIG.json',dict(schema='r17.incremental_daemon.v1',port=port,root=str(directory),cpu_only=True,launched=False))


def active_campaigns(root=ROOT):
    """Deduplicate aliases; ignore stale claims and offline historical nodes."""
    runs={}
    for deployed in Path(root).glob('*/WORKER_DEPLOYMENT.json'):
        run=deployed.parent.resolve()
        claim=run/'LAUNCH.json'
        if not claim.exists() or (run/'GPU_RELEASED.json').exists():continue
        try:
            pid=json.loads(claim.read_text())['pid'];proc=Path('/proc')/str(pid)
            if proc.stat().st_uid!=os.getuid():continue
            argv=(proc/'cmdline').read_bytes().decode().strip('\0').split('\0')
            matched=('--prepared-run' in argv and Path(argv[argv.index('--prepared-run')+1]).resolve()==run or
                '--run-name' in argv and argv[argv.index('--run-name')+1]==run.name)
            if matched:runs[str(run)]=run
        except (FileNotFoundError,KeyError,ValueError,IndexError,ProcessLookupError):continue
    return list(runs.values())


def sync_campaign(campaign,directory,bwlimit,max_chunks):
    from .formal_eval import remote,transfer
    campaign,directory=Path(campaign).resolve(),Path(directory)
    c=json.loads((campaign/'campaign.json').read_text())
    deployment=json.loads((campaign/'WORKER_DEPLOYMENT.json').read_text())
    settings=json.loads((directory/'CONFIG.json').read_text())
    results=[]
    for node,item in deployment.items():
        if node=='local':continue  # already under /data; no remote disk pressure
        manifest=campaign/(node+'_launch.json')
        if not manifest.exists():manifest=campaign/(node+'.json')
        request=dict(node=node,run=item['worker_run'],manifest_sha256=sha(manifest),max_bytes=32*1024**2)
        def call(value):
            code=inspect.getsource(remote_operation)+'\nprint(json.dumps(remote_operation('+repr(value)+')))'
            return json.loads(remote(node,['python3','-c','import json\n'+code],seconds=210,replay_safe=False))
        for _ in range(max_chunks):
            chunk=call(dict(request,action='stage'))
            if chunk['status']=='SKIP':break
            local_free=os.statvfs(directory)
            if local_free.f_bavail*local_free.f_frsize<8*1024**3+2*chunk['bytes']:
                raise RuntimeError('local sync disk headroom low; keep remote originals')
            destination=directory/'inbox'/campaign.name/node;destination.mkdir(parents=True,exist_ok=True)
            url='rsync://q_incremental@ziyanglin.com:'+str(settings['port'])+'/r17q/'+campaign.name+'/'+node+'/'+chunk['token']+'.tar.gz'
            password_root='/scratch/zixuans8/mw_r17/rsync_stage' if node=='timan1' else '/tmp/r17_q_incremental'
            remote_password=password_root+'/'+campaign.name+'_'+node+'.password'
            remote(node,['mkdir','-p',password_root],replay_safe=True)
            transfer(['rsync','push',str(directory/'rsync.password'),node+':'+remote_password,'--force'])
            remote(node,['chmod','600',remote_password],replay_safe=True)
            call(dict(request,action='push',chunk=chunk,url=url,password_file=remote_password,bwlimit_kibps=bwlimit))
            receipt=verify_chunk(chunk,destination/(chunk['token']+'.tar.gz'),campaign/'INCREMENTAL_SYNC'/node)
            result=call(dict(request,action='ack',chunk=chunk,verified_archive_sha256=chunk['archive_sha256'],local_receipt_sha256=sha(receipt)))
            results.append(dict(node=node,token=chunk['token'],**result))
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,default=ROOT/'incremental_sync_20261010')
    p.add_argument('--port',type=int,default=DEFAULT_PORT)
    p.add_argument('--interval',type=int,default=180)
    p.add_argument('--bwlimit-kibps',type=int,default=4096)
    p.add_argument('--max-chunks',type=int,default=16)
    p.add_argument('--campaign',type=Path,action='append')
    modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--setup',action='store_true');modes.add_argument('--run',action='store_true')
    a=p.parse_args()
    if a.setup:setup(a.directory,a.port);return
    if a.interval<30 or not 64<=a.bwlimit_kibps<=16384 or not 1<=a.max_chunks<=64:raise ValueError('unbounded sync settings')
    import fcntl,signal
    from .formal_process import ticks,stop_group
    lock=(a.directory/'.daemon.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=a.directory/'rsyncd.conf'
    port=json.loads((a.directory/'CONFIG.json').read_text())['port']
    child=subprocess.Popen(['rsync','--daemon','--no-detach','--port='+str(port),'--config='+str(config)],start_new_session=True)
    entry=dict(pid=child.pid,pgid=child.pid,start_ticks=ticks(child.pid),role='own rsync receiver')
    write_json(a.directory/'DAEMON_OWNER.json',entry)
    def stop(*unused):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        while True:
            if child.poll() is not None:raise RuntimeError('owned rsync daemon exited')
            runs=a.campaign or active_campaigns()
            for run in runs:
                try:result=sync_campaign(run,a.directory,a.bwlimit_kibps,a.max_chunks)
                except Exception as exc:result=dict(status='RETRY_NEXT_CYCLE',error=str(exc))
                write_json(a.directory/('cycle_'+Path(run).name+'.json'),dict(time=time.time(),result=result))
            time.sleep(a.interval)
    finally:
        # This is only the daemon created here, never a model/worker/lane.
        if child.poll() is None:stop_group(entry);child.wait(timeout=15)
        lock.close()


if __name__=='__main__':main()
