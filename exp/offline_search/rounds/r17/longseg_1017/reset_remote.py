"""Owner-invoked archive of a CLOSED R17 server directory, never a deletion."""
import inspect
import json


def archive(request, *, campaign_parent='/data/mw_finetune/eval', proc_root='/proc', check_only=False):
    """Standalone helper. No signals, process starts, or content removal."""
    import json
    import os
    import fcntl
    from pathlib import Path
    root, backup = Path(request['root']), Path(request['backup'])
    if (not root.is_absolute() or root.parent != Path(campaign_parent) or
        backup.parent != root.parent or not root.name.startswith('fa_') or
        not backup.name.startswith(root.name+'.') or root.is_symlink() or backup.is_symlink() or
        root.resolve()!=root or backup.resolve()!=backup):
        raise ValueError('only an owned campaign and explicit sibling backup may be archived')
    marker = backup/'AUTOSCALE_REMOTE_RESET.json'
    if not root.exists() and marker.is_file():
        prior=json.loads(marker.read_text())
        if prior.get('request') == request:return prior
    if backup.exists():raise ValueError('remote backup already exists')
    if not root.exists():return dict(status='ABSENT',request=request)
    if root.stat().st_uid != os.getuid():raise ValueError('remote campaign is not owned by this UID')
    # Refuse any process carrying this campaign in argv/cwd. No unrelated
    # process environment (or credentials) is read or emitted.
    unreadable = dict(identity=0,cmdline=0,cwd=0,receipt_stat=0)
    def busy():
        for directory in Path(proc_root).glob('[0-9]*'):
            try:
                if int(directory.name)==os.getpid() or directory.stat().st_uid!=os.getuid():continue
            except (FileNotFoundError,ProcessLookupError,PermissionError):
                unreadable['identity']+=1;continue
            try:
                args=(directory/'cmdline').read_bytes().split(b'\0')
            except (FileNotFoundError,ProcessLookupError,PermissionError):
                unreadable['cmdline']+=1;args=[]
            prefix=str(root).encode()
            # Check argv BEFORE cwd: unreadable cwd cannot hide a proven owner.
            for i, arg in enumerate(args):
                if i and args[i-1] in (b'-c',b'-lc'):continue
                if arg==prefix or arg.startswith(prefix+b'/') or b'"'+prefix+b'/' in arg:
                    raise ValueError('owned process still references remote campaign: pid='+directory.name)
            try:
                where=str((directory/'cwd').resolve(strict=True))
            except (FileNotFoundError,ProcessLookupError,PermissionError):
                unreadable['cwd']+=1;continue
            if where==str(root) or where.startswith(str(root)+'/'):
                raise ValueError('owned process cwd is inside remote campaign: pid='+directory.name)
        # Receipt identities cover MPS children whose argv does not name a run.
        groups=set()
        def alive(value):
            if isinstance(value,dict):
                if value.get('pgid'):groups.add(int(value['pgid']))
                if value.get('pid') and value.get('start_ticks'):
                    p=Path(proc_root)/str(value['pid'])/'stat'
                    try:
                        fields=p.read_text().rsplit(')',1)[1].split()
                        if fields[19]==str(value['start_ticks']) and fields[0]!='Z':
                            raise ValueError('live remote receipt pid='+str(value['pid']))
                    except (FileNotFoundError,ProcessLookupError):pass
                    except PermissionError:unreadable['receipt_stat']+=1
                for child in value.values():alive(child)
            elif isinstance(value,list):
                for child in value:alive(child)
        for pattern in ('*receipt*.json','*owner*.json','*manifest*.json',
                        '*RECEIPT*.json','*OWNER*.json','*MANIFEST*.json',
                        '*.process.json','MPS_READY.json','NATIVE_PROCESS.json'):
            for path in root.rglob(pattern):
                if path.is_symlink():raise ValueError('symlinked remote ownership receipt')
                if path.stat().st_size > 2**20:raise ValueError('oversize ownership receipt; inspect manually')
                alive(json.loads(path.read_text()))
        if groups:
            for directory in Path(proc_root).glob('[0-9]*'):
                try:
                    fields=(directory/'stat').read_text().rsplit(')',1)[1].split()
                    if fields[0]!='Z' and int(fields[2]) in groups:
                        raise ValueError('live remote receipt process group: pid='+directory.name)
                except (FileNotFoundError,ProcessLookupError):pass
                except PermissionError:unreadable['receipt_stat']+=1
    if check_only:
        busy()
        return dict(status='READY',request=request,unreadable_process_fields=unreadable)
    with (root/'.prepared.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        busy()
        root.rename(backup)
        result=dict(status='ARCHIVED',request=request,processes_signalled=0,files_deleted=0,
                    unreadable_process_fields=unreadable)
        marker.write_text(json.dumps(result,indent=2)+'\n')
        return result


def run(host, root, backup, *, check_only=False):
    from pathlib import Path
    allowed = {
        'h100':('/data/mw_finetune/eval', '/tmp/r17_formal', '/data/mw_finetune/qualification/h100_workers/runs'),
        'a100':('/tmp/r17_formal', '/data/mw_finetune/qualification/h100_workers/runs'),
        'timan1':('/scratch/zixuans8/mw_r17/formal/stage', '/scratch/zixuans8/mw_r17/formal/runs'),
        'timan107':('/tmp/r17_formal', '/scratch/zixuans8/mw_r17/formal/runs'),
        'local':('/tmp/r17_formal', '/data/mw_finetune/qualification/local_workers/runs',
                 '/nvme/weiland/openpi/qualification/local_workers/runs'),  # owner 10-20: new local runs on /nvme
    }
    if str(Path(root).parent) not in allowed.get(host, ()):
        raise ValueError('unsupported node or campaign archive parent; no reset performed')
    from exp.r17_integration.ops import formal_eval
    request=dict(root=str(root),backup=str(backup))
    kwargs=dict(campaign_parent=str(Path(root).parent),check_only=check_only)
    if host=='local':return archive(request,**kwargs)
    code=inspect.getsource(archive)+'\nimport json\nprint(json.dumps(archive('+repr(request)+',**'+repr(kwargs)+')))'
    return json.loads(formal_eval.remote(host,['/usr/bin/python3','-c',code],seconds=60,replay_safe=True))
