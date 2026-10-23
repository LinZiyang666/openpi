"""Temporary allowlisted public rsync; CPU-only pi05 staging or FW pairing."""
import argparse
from contextlib import contextmanager
import inspect
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parent
H100=Path('/home/exouser/xgpu_1017')
FORBIDDEN={23177,23118}  # former study port; normal formal transfer port
FW_RUNS=('4090_fastwam_concurrent_fw4090_reduced_20261018T140530Z',
         '4090_fastwam_concurrent_fw4090_reduced_20261018T141216Z')


def port_snapshot():
    # XGPU_PORT_PROBE: self-contained, read-only, safe to send through tether exec.
    import os,json,re,subprocess
    from pathlib import Path
    text=subprocess.run(['ss','-H','-lntu'],capture_output=True,text=True,check=True).stdout
    listeners=set()
    for line in text.splitlines():
        endpoint=line.split()[4]
        if endpoint.rsplit(':',1)[-1].isdigit():listeners.add(int(endpoint.rsplit(':',1)[-1]))
    campaigns={}; pids={}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) in (os.getpid(),os.getppid()):continue
        try:args=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        joined=' '.join(args)
        if 'XGPU_PORT_PROBE' in joined:continue
        candidates=set(re.findall(r'/[A-Za-z0-9_./-]+/(?:campaign|launch_campaign)\.json',joined))
        for root in re.findall(r'/data/mw_finetune/eval/[A-Za-z0-9_.-]+',joined):
            if (Path(root)/'campaign.json').is_file():candidates.add(root+'/campaign.json')
        for i,arg in enumerate(args[:-1]):
            if arg=='--campaign':
                path=Path(args[i+1])
                if not path.is_absolute():
                    try:path=(proc/'cwd').resolve()/path
                    except (OSError,RuntimeError):raise ValueError('cannot resolve active campaign cwd')
                candidates.add(str(path))
        for name in candidates:pids.setdefault(name,[]).append(int(proc.name))
    for name,owners in sorted(pids.items()):
        path=Path(name)
        # Huge plans have sorted top-level port fields near the ends. Read at
        # most 256 KiB, failing closed if the complete ports list is not visible.
        with path.open('rb') as stream:
            size=path.stat().st_size
            head=stream.read(131072)
            if size>262144:stream.seek(-131072,2);body=head+b'\n'+stream.read(131072)
            else:body=head+stream.read()
        if size<=262144:
            record=json.loads(body)
            ports=record.get('ports')
        else:
            found=re.findall(rb'(?m)^  "ports"\s*:\s*(\[[^\]]*\])',body)
            if len(found)!=1:raise ValueError('cannot safely read active campaign ports: '+name)
            ports=json.loads(found[0])
        if not isinstance(ports,list) or not ports or any(type(p)is not int or not 0<p<65536 for p in ports):
            raise ValueError('invalid active campaign ports: '+name)
        campaigns[name]=dict(ports=ports,pids=owners)
    return dict(listeners=sorted(listeners),active_campaigns=campaigns,ss=text,
                scope='all readable live /proc command references; all listed campaign ports reserved, including unbound ones')


def remote(argv, seconds=120):
    p=subprocess.run(['tether','exec','--timeout',str(seconds)+'s','h100','--',*map(str,argv)],
                     capture_output=True,text=True,timeout=seconds+20)
    if p.returncode:raise RuntimeError('tether exec failed: '+p.stderr[-6000:]+p.stdout[-6000:])
    return p.stdout


def snapshots():
    local=port_snapshot()
    code=inspect.getsource(port_snapshot)+'\nimport json;print(json.dumps(port_snapshot()))'
    return dict(local=local,h100=json.loads(remote(['python3','-c',code],60)))


def select_port(audit, requested=None):
    forbidden=set(FORBIDDEN)
    for snapshot in audit.values():
        forbidden.update(snapshot['listeners'])
        for record in snapshot['active_campaigns'].values():forbidden.update(record['ports'])
    candidates=[requested] if requested is not None else range(23199,23099,-1)
    for port in candidates:
        if not 1024<=port<=65535 or port in forbidden:continue
        with socket.socket() as sock:
            try:sock.bind(('0.0.0.0',port))
            except OSError:continue
        return port
    raise ValueError('no unreserved free transfer port; 23177/23118, listeners and every active campaign port are forbidden')


def io_check():
    pressure=Path('/proc/pressure/io').read_text()
    print(pressure,flush=True)
    if any(float(x)>=5 for x in re.findall(r'avg10=([0-9.]+)',pressure)):
        raise RuntimeError('IO avg10 >= 5%; retry staging when pressure falls')
    return pressure


def module(directory,name,base,files,destination):
    base=Path(base).resolve();safe=[];entries=set();total=0
    for p in sorted(set(map(Path,files))):
        if not p.is_file() or p.is_symlink():raise ValueError('export must be a regular file: '+str(p))
        relative=p.relative_to(base)
        if p.resolve()!=base/relative or '..' in relative.parts or any(c in relative.as_posix() for c in '*?[]\n\r'):
            raise ValueError('unsafe allowlist path: '+str(p))
        entries.add('/'+relative.as_posix())
        for parent in relative.parents:
            if parent!=Path('.'):entries.add('/'+parent.as_posix()+'/')
        safe.append(relative.as_posix());total+=p.stat().st_size
    if not safe:raise ValueError('empty transfer module')
    include=directory/(name+'.include');include.write_text('\n'.join(sorted(entries))+'\n')
    return dict(name=name,base=str(base),include=str(include),destination=str(destination),files=len(safe),bytes=total)


def modules(directory,mode):
    result=[]
    if mode=='pi05':
        bundle=ROOT/'concurrent/bundles/pi05'
        result.append(module(directory,'tools',ROOT,[p for p in ROOT.iterdir() if p.suffix in ('.py','.sh')],H100/'tools'))
        result.append(module(directory,'bundle',bundle,[p for p in bundle.rglob('*') if p.is_file()],H100/'concurrent/bundles/pi05'))
        result.append(module(directory,'reference',ROOT/'concurrent/references',[ROOT/'concurrent/references/pi05.json'],H100/'concurrent/references'))
        inputs=ROOT/'inputs_mt50_real3'
        result.append(module(directory,'inputs',inputs,[p for p in inputs.iterdir() if p.name=='INPUTS.json' or p.name.startswith('input_') and p.suffix in ('.json','.npz')],H100/'inputs_mt50_real3'))
        result.append(module(directory,'cache',Path('/data/mw_finetune'),[Path(p) for p in (bundle/'data.files').read_text().splitlines()],H100/'rootfs/data/mw_finetune'))
        manifest=json.loads((bundle/'assets/model_manifest.json').read_text());base=Path(manifest['checkpoint_root'])
        small=[base/n for n in manifest['content_digests'] if n!='model.safetensors']
        result.append(module(directory,'checkpoint_small',base,small,H100/'pi05_checkpoint_small'))
    else:
        files=[]
        for name in FW_RUNS:
            root=ROOT/'results'/name
            if json.loads((root/'RUN.json').read_text())['status']!='COMPLETE':raise ValueError('incomplete FW source')
            files.extend(p for p in root.rglob('*') if p.is_file() and p.suffix!='.log' and p.name!='batching.jsonl')
        result.append(module(directory,'fw_reference',ROOT/'results',files,H100/'reference_results'))
    return result


@contextmanager
def daemon(directory,port,exports):
    config=directory/'rsyncd.conf'
    body=('use chroot = no\nread only = yes\nlist = no\nmunge symlinks = yes\n'
          'hosts allow = 149.165.152.232 149.165.153.233\nhosts deny = *\nmax connections = 1\ntimeout = 120\n'
          f'lock file = {directory}/rsync.lock\npid file = {directory}/rsync.pid\nlog file = {directory}/rsync.log\n')
    for m in exports:body+=f"[{m['name']}]\npath = {m['base']}\ninclude from = {m['include']}\nexclude = *\nread only = yes\n"
    config.write_text(body)
    with (directory/'daemon.stderr').open('w') as log:
        process=subprocess.Popen(['rsync','--daemon','--no-detach','--address=0.0.0.0','--port='+str(port),'--config='+str(config)],
                                 stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        receipt=dict(pid=process.pid,port=port,config=str(config),closed=False)
        try:
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                if process.poll() is not None:raise RuntimeError('temporary daemon failed to bind; see '+str(directory/'daemon.stderr'))
                if (directory/'rsync.pid').exists() and int((directory/'rsync.pid').read_text())==process.pid:
                    listening=subprocess.run(['ss','-H','-ltn','sport','=',':'+str(port)],capture_output=True,text=True,check=True).stdout
                    if listening:break
                time.sleep(.05)
            else:raise TimeoutError('temporary daemon did not become ready')
            yield receipt
        finally:
            # Signal only our still-owned process group; never identify by port/name.
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
            receipt.update(closed=True,exit_code=process.returncode)
            (directory/'DAEMON_CLOSED.json').write_text(json.dumps(receipt,indent=2)+'\n')


def pi_preflight():
    runtime=json.loads((ROOT/'concurrent/bundles/pi05/source_campaign.json').read_text())['runtime']
    runtime_release_sha=hashlib.sha256((Path(runtime)/'RUNTIME_RELEASE.json').read_bytes()).hexdigest()
    code=f'''import sys,json,hashlib,importlib,importlib.metadata
from pathlib import Path
sys.path[:0]=[{runtime!r}+'/src',{runtime!r}+'/packages/openpi-client/src',{runtime!r},'/home/exouser/xgpu_1017/tools']
from concurrent_server import checked_pin
from cache_files import verify
from capture import read_inputs
bundle=Path('/home/exouser/xgpu_1017/concurrent/bundles/pi05')
pin=checked_pin(bundle)
cache=verify(bundle,[('/data/mw_finetune/eval','/home/exouser/xgpu_1017/rootfs/data/mw_finetune/eval')])
for name in ['openpi.models_pytorch.pi0_pytorch','openpi.policies.policy_config','openpi.training.config','sentencepiece','websockets','safetensors']:
    importlib.import_module(name)
from transformers.models.siglip import check
assert check.check_whether_transformers_replace_is_installed_correctly()
manifest=json.loads((bundle/'assets/model_manifest.json').read_text())
tokenizer=Path('/home/exouser/.cache/openpi/big_vision/paligemma_tokenizer.model')
assert tokenizer.is_file()
release_file=Path({runtime!r})/'RUNTIME_RELEASE.json'
release=json.loads(release_file.read_text())
for name in manifest['source_digests']:
    assert (Path({runtime!r})/name).is_file(),name
input_manifest,inputs=read_inputs('/home/exouser/xgpu_1017/inputs_mt50_real3')
print(json.dumps(dict(status='PASS',pin=pin['identity'],cache=cache,inputs=len(inputs),input_identity=input_manifest['identity'],packages={{k:importlib.metadata.version(k) for k in ['torch','transformers','numpy','sentencepiece']}})))
'''
    return json.loads(remote(['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',
                   '/home/exouser/openpi/.venv/bin/python','-c',code],180))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['pi05','fw'])
    p.add_argument('--port',type=int);p.add_argument('--run',action='store_true',help='otherwise generate allowlists and CPU port plan only')
    a=p.parse_args();directory=ROOT/'evidence'/('temporary_rsync_'+a.mode+'_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:6])
    directory.mkdir(mode=0o700)
    audit=snapshots();port=select_port(audit,a.port);exports=modules(directory,a.mode)
    plan=dict(mode=a.mode,port=port,audit=audit,exports=exports,gpu_gib=0,run=a.run)
    (directory/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(directory=str(directory),port=port,bytes=sum(m['bytes'] for m in exports),gpu_gib=0,run=a.run)),flush=True)
    if not a.run:return
    io_check()
    # Refresh immediately before bind; a listener race makes rsync fail safely.
    refreshed=snapshots();select_port(refreshed,port)
    (directory/'PORTS_BEFORE_BIND.json').write_text(json.dumps(refreshed,indent=2)+'\n')
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        with daemon(directory,port,exports):
            (directory/'h100_df_before.txt').write_text(remote(['df','-h','-l'],30))
            for m in exports:
                io_check()
                remote(['mkdir','-p',m['destination']],30)
                output=remote(['rsync','-a','--bwlimit=16384','--timeout=120','--contimeout=15','--stats',
                         f"rsync://ziyanglin.com:{port}/{m['name']}/",m['destination']+'/'],600)
                (directory/(m['name']+'.transfer.log')).write_text(output)
        # The daemon is closed before any substantial CPU preflight/hash work.
        if a.mode=='pi05':
            preflight=pi_preflight();(directory/'PI05_PREFLIGHT.json').write_text(json.dumps(preflight,indent=2)+'\n')
            output=remote(['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','/home/exouser/openpi/.venv/bin/python',
                           H100/'tools/pi05_checkpoint_view.py','--create'],300)
            (directory/'PI05_CHECKPOINT_VIEW.json').write_text(output)
        else:
            subprocess.run([sys.executable,str(ROOT/'analyze_fw_remote.py')],check=True)
        (directory/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',gpu_gib=0,daemon_closed=True))+'\n')
        print('COMPLETE '+str(directory),flush=True)
    except BaseException as error:
        (directory/'FAILED.json').write_text(json.dumps(dict(error=repr(error)))+'\n');raise


if __name__=='__main__':main()
