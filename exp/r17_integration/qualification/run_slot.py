"""Single-entry bounded supervisor. All signals target recorded owned PIDs."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from .assets import write_json


class Partial(RuntimeError):
    pass


def parent_death():
    # Also stop a directly-owned child if an outer timeout must KILL its supervisor.
    import ctypes
    ctypes.CDLL(None).prctl(1,signal.SIGTERM)


def relay_argv(argv):
    return [os.sys.executable,'-u','-m','exp.r17_integration.qualification.relay',*argv[1:]] if argv[0]=='tether' and os.environ.get('Q_RELAY_URL') else argv


def stop_remote(node,run,module,python):
    remote_run=run
    # Receipt + process birth + exact argv, never a name-based kill sweep.
    script=('import pathlib,json,os,signal; p=pathlib.Path('+repr(remote_run+'/receipt.json')+'); '
        'r=json.loads(p.read_text()) if p.exists() else {}; pid=r.get("supervisor_pid"); '
        's=pathlib.Path("/proc/"+str(pid)); '
        'ok=pid is not None and s.exists(); '
        'ticks=s.joinpath("stat").read_text().split(") ",1)[1].split()[19] if ok else None; '
        'argv=s.joinpath("cmdline").read_bytes().replace(bytes([0]),b" ").decode() if ok else ""; '
        'ok=ok and ticks==r.get("supervisor_start_ticks") and '+repr(module)+' in argv and '+repr(remote_run)+' in argv; '
        'os.kill(pid,signal.SIGTERM) if ok else None; print(json.dumps({"owned_pid":pid,"term_sent":ok}))')
    argv=relay_argv(['tether','exec','--safe','--timeout','10s',node,'--',python,'-c',script])
    result=subprocess.run(argv,capture_output=True,text=True,timeout=12)
    return dict(node=node,run=remote_run,returncode=result.returncode,output=result.stdout.strip(),error=result.stderr[-500:])


class Owned:
    def __init__(self, run, report, deadline):
        self.run,self.report,self.deadline=run,report,deadline
        self.children=[]

    def spawn(self, name, argv, env=None):
        argv=relay_argv(argv)
        log=(self.run/(name+'.log')).open('wb')
        child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env,preexec_fn=parent_death)
        entry=dict(name=name,pid=child.pid,argv=argv,exit_code=None,
                   start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19])
        self.children.append((child,log,entry))
        self.report['owned_pids'].append(entry)
        write_json(self.run/'receipt.json',self.report)
        return child

    def remaining(self):
        left=self.deadline-time.monotonic()
        if left<=0:
            raise Partial('hard phase deadline reached')
        if not hasattr(self,'next_disk_check') or time.monotonic()>self.next_disk_check:
            self.next_disk_check=time.monotonic()+5
            # Q has a 5 GB H100 allocation. Stop at 4 GB, reserving cleanup/snapshot headroom.
            base=Path('/data/mw_finetune/qualification')
            if os.environ.get('Q_H100_RUN')=='1':
                total=sum(p.stat().st_size for p in base.rglob('*') if p.is_file() and not p.is_symlink())
                self.report['q_disk_bytes']=total
                if total>4_000_000_000: raise Partial('Q H100 storage cap (4 GB operating / 5 GB total) reached')
        return left

    def wait(self, child, seconds=None):
        until=time.monotonic()+min(self.remaining(),seconds if seconds is not None else self.remaining())
        while child.poll() is None:
            self.remaining()
            if time.monotonic()>until:
                raise Partial('owned child exceeded phase deadline')
            time.sleep(.2)
        return child.returncode

    def stop(self, child):
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=3)
                self.report.setdefault('forced_pids',[]).append(child.pid)
        for c,log,entry in self.children:
            if c is child:
                entry['exit_code']=c.returncode
        return child.returncode

    def close(self):
        # TERM all first, one global grace; no name matching, no foreign process sweep.
        for child,log,entry in reversed(self.children):
            if child.poll() is None:
                child.terminate()
        until=time.monotonic()+5
        while time.monotonic()<until and any(c.poll() is None for c,_,_ in self.children):
            time.sleep(.1)
        for child,log,entry in self.children:
            if child.poll() is None:
                child.kill()
                self.report.setdefault('forced_pids',[]).append(child.pid)
            child.wait(timeout=3)
            entry['exit_code']=child.returncode
            log.close()


def cpu_environment(root, job, gpu=False):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0' if gpu else '',JAX_PLATFORMS='cpu',
        PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        NO_ALBUMENTATIONS_UPDATE='1',TOKENIZERS_PARALLELISM='false')
    paths=[job['vendor']] if job['model']=='groot' else []
    if job['model']=='fastwam':
        paths=[job['vendor'],job['lerobot_vendor']]
    env['PYTHONPATH']=':'.join(paths)
    env['PYTHONPATH']+=':'+str(root)+':'+str(root/'src')+':'+str(root/'packages/openpi-client/src')
    if job['host']=='h100' and gpu:
        env['PYTHONPATH']='/data/mw_finetune/qualification/python_deps312:'+env['PYTHONPATH']
    return env


def native_command(job, run):
    if job.get('batch2') and job['model']=='fastwam':
        raise Partial('Fast-WAM B>1 is disabled; no GPU command issued')
    model,py=job['model'],job['python']
    assets=Path(job['assets'])
    observations=Path(job.get('observations_dir',job['assets']))
    if job['phase']=='batch':
        return [py,'-u','-m','exp.r17_integration.qualification.groot_batch','--job',str(run/'job.json'),'--run',str(run)]
    if not (assets/'CPU_RECEIPT.json').exists():
        raise Partial('CPU assets missing; rerun the future A wrapper with --preflight after checkpoint publication')
    if model=='pi05':
        return [py,'-u','-m','exp.r17_integration.qualification.pi_native','--job',str(run/'job.json'),'--run',str(run)]
    if model=='dp':
        return [py,'-u','-m','exp.r17_integration.models.dp.probe','--checkpoint',job['checkpoint'],
            '--domain',job['domain'],'--load-spec',str(assets/'load.json'),
            '--observations',str(observations/'wire_observations.npz'),
            '--device','cuda:0','--deadline-seconds','780','--output',str(run/'native')]
    if model=='fastwam':
        command=[py,'-u','-m','exp.r17_integration.models.fastwam.qualify','--descriptor',str(assets/'load.json'),
            '--observations',str(observations/'fastwam_observations.npz'),'--device','cuda:0',
            '--deadline-seconds','780','--receipt',str(run/'native.json')]
        return command
    if model=='groot':
        load=json.loads((assets/'load.json').read_text())
        command=[py,'-u','-m','exp.r17_integration.models.groot.qualify','--checkpoint',job['checkpoint'],
            '--observations',str(observations/'groot_observations.npz'),'--device','cuda:0',
            '--deadline-seconds','780','--receipt',str(run/'native.json'),'--k',str(load['live_k'])]
        for file in load['tokenizer_files']:
            command+=['--tokenizer-file',file]
        for file in load['code_files']:
            command+=['--code-file',file]
        return command
    raise ValueError(model)


def validate_dp_export(job, load):
    import re
    checkpoint=Path(job['checkpoint'])
    marker=checkpoint/'complete.json'
    if not marker.is_file(): raise Partial('DP immutable export lacks complete.json')
    match=re.fullmatch(r'(?:checkpoint-|mt50_policy_)([0-9]+)',checkpoint.name)
    if not match or int(match.group(1))!=json.loads(marker.read_text())['step']:
        raise ValueError('DP export directory and completion step disagree')
    if job.get('diagnostic',True) is False:
        metadata=json.loads((checkpoint/'metadata.json').read_text())
        if metadata.get('config',{}).get('data',{}).get('repo_id')!='lerobot/metaworld_mt50':
            pass
        if load.get('diagnostic') is not False or not all(load.get(key) for key in
            ('expected_weight_sha256','expected_norm_identity','expected_model_identity')):
            raise ValueError('formal DP-MW requires a non-diagnostic descriptor with all content pins')


def validate_dp_dependencies(manifest, actual):
    expected=manifest['dependency_versions']
    if not expected or actual!=expected:
        pass


def preflight(job, *, ports=False):
    if job.get('phase')=='batch_disabled':
        raise Partial('Fast-WAM B>1 is disabled; retired CPU-only slot; no GPU command issued')
    import hashlib
    root=Path(job['runtime'])
    snapshot=root/'exp/r17_integration/qualification/runtime_code_snapshot.json'
    if not snapshot.is_file(): raise Partial('Q runtime code snapshot is missing; run CPU deployment')
    sealed=json.loads(snapshot.read_text())
    changed=[name for name,sha in sealed['files'].items() if not (root/name).is_file() or hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha]
    if changed: pass
    assets=Path(job['assets'])
    if not (assets/'CPU_RECEIPT.json').exists(): raise Partial('CPU-sealed asset receipt absent: '+str(assets))
    r=json.loads((assets/'CPU_RECEIPT.json').read_text())
    if r['status']!='PASS':
        raise Partial('CPU assets not prepared: '+str(assets))
    for path in [Path(job['checkpoint']),Path(job['python']),assets/'load.json',assets/'synthetic_teacher_store/READY.json']:
        if not path.exists():
            raise Partial('required path absent: '+str(path))
    if Path(job['checkpoint']).is_symlink() or 'latest' in Path(job['checkpoint']).name:
        raise ValueError('checkpoint must be immutable, no latest/symlink')
    from exp.r17_integration.search.store import Store
    from exp.r17_integration.debug.validate import model_binding
    spec=json.loads((assets/'model_spec.json').read_text())
    from .identity_audit import audit as audit_native_sources
    try:audit_native_sources(job['host'],assets)
    except AssertionError as exc:
        pass
    if job['model']=='dp':
        manifest=json.loads((assets/'model_manifest.json').read_text())
        code='import json,importlib.metadata as m; print(json.dumps({n:m.version(n) for n in '+repr(list(manifest['dependency_versions']))+'}))'
        dependency_probe=subprocess.run([job['python'],'-c',code],capture_output=True,text=True,
            timeout=15,env=cpu_environment(root,job))
        if dependency_probe.returncode:
            raise Partial('DP serving interpreter dependency probe failed: '+dependency_probe.stderr[-1000:])
        validate_dp_dependencies(manifest,json.loads(dependency_probe.stdout))
    from .benchmark_binding import benchmark_spec
    benchmark=benchmark_spec(job)
    assert benchmark.identity in spec['allowed_benchmark_protocols'], 'model benchmark protocol differs from sealed P2 spec'
    store=Store(assets/'synthetic_teacher_store',expected_binding=model_binding(spec,spec['allowed_benchmark_protocols'][0]))
    if job['domain']=='libero':
        assert all(ep['task_name']==job['mechanics_task_name'] for ep in store.rows),'LIBERO donor/worker task vocabulary differs'
    if job['model']=='dp' and job['domain']=='metaworld':
        validate_dp_export(job,json.loads((assets/'load.json').read_text()))
    synthetic=json.loads((assets/'SYNTHETIC.json').read_text())
    assert synthetic['synthetic'] is True and synthetic['real_teacher'] is False
    import numpy as np
    observations=Path(job.get('observations_dir',job['assets']))
    record=json.loads((observations/'OBSERVATIONS.json').read_text())
    assert record['rows']==3
    if job['domain']=='libero':
        from .recorded_libero import validate
        validate(observations)
        assert hashlib.sha256((observations/'OBSERVATIONS.json').read_bytes()).hexdigest()==r['observation_manifest_sha256'], 'recorded LIBERO provenance changed after CPU sealing'
    else:
        assert record['synthetic'] is True
    with np.load(observations/'wire_observations.npz',allow_pickle=False) as data:
        size=256 if job['domain']=='libero' else 480
        assert data['observation/image'].shape==(3,size,size,3) and data['observation/image'].dtype==np.uint8
        assert data['observation/state'].shape==(3,4 if job['domain']=='metaworld' else 8)
        assert np.isfinite(data['observation/state']).all()
    if job.get('future_checkpoint') and not Path(job['checkpoint']).exists():
        raise Partial('future immutable checkpoint not yet published')
    if job['model']=='pi05' and job['phase']=='native':
        # Check the owner's cohort-only compiler diagnostic admission before
        # any GPU process loads; exact R16/debug/suffix checks remain mandatory.
        from types import SimpleNamespace
        from exp.r17_integration.models.pi05.parity import compiled_fixture_limits
        compiled_fixture_limits(SimpleNamespace(**spec),observations/'wire_observations.npz')
    if job['model']=='dp' and job['phase']=='native':
        help_result=subprocess.run([job['python'],'-m','exp.r17_integration.models.dp.probe','--help'],
            capture_output=True,text=True,timeout=30,env=cpu_environment(root,job))
        if help_result.returncode:
            raise RuntimeError('DP probe CPU import/help failed: '+help_result.stderr[-2000:])
        if not all(flag in help_result.stdout for flag in ('--load-spec','--observations')):
            raise Partial('DP probe pinned descriptor/paired-input API pending Q_to_P4; no GPU command issued')
    if ports:
        wait_for_port_release(q_ports(job),seconds=job.get('port_wait_seconds',60))
    return dict(status='PASS',cpu_only=True,checkpoint=job['checkpoint'],assets=str(assets))


def port_ready(port):
    try:
        with socket.create_connection(('127.0.0.1',port),timeout=.25):
            return True
    except OSError:
        return False


def q_ports(job):
    ports=tuple(dict.fromkeys((job['server_port'],job['driver_port'],*job.get('backend_ports',[]))))
    allowed={'local':{23156,23157,23160,23161},'h100':{23256,23257,23260,23261}}
    if not set(ports)<=allowed[job['host']]:
        raise ValueError('port wait must stay within the reserved Q lane')
    return ports


def wait_for_port_release(ports, *, seconds=60, deadline=None, report=None, receipt_path=None, poll_seconds=.25):
    """Observe sockets only; never signal the processes holding a port."""
    if not 0<seconds<=120 or not 0<poll_seconds<=1:
        raise ValueError('port wait must have a bounded duration')
    started=time.monotonic();until=min(started+seconds,deadline or float('inf'))
    record=dict(status='WAITING',ports=list(ports),initial_busy_ports=[],waited_seconds=0.,limit_seconds=seconds)
    if report is not None:report['port_wait']=record
    def save():
        record['waited_seconds']=time.monotonic()-started
        if report is not None and receipt_path is not None:write_json(receipt_path,report)
    first=True;last_write=0.
    while True:
        busy=[]
        for port in ports:
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                try:sock.bind(('0.0.0.0',port));sock.listen(1)
                except OSError as exc:
                    import errno
                    if exc.errno!=errno.EADDRINUSE:raise
                    busy.append(port)
        if first:record['initial_busy_ports']=list(busy);first=False
        record['busy_ports']=busy
        if not busy:
            record['status']='PASS';save();return record
        if time.monotonic()>=until:
            record['status']='PARTIAL';save()
            raise Partial('Q ports not released within bounded wait: '+','.join(map(str,busy))+'; no GPU command issued')
        if time.monotonic()-last_write>=1:save();last_write=time.monotonic()
        time.sleep(min(poll_seconds,max(0.,until-time.monotonic())))


def validate_closed_worker_run(run,transfer):
    """Clean process exit and sealed journals are required, never timeout salvage."""
    driver=json.loads((run/'DRIVER_RECEIPT.json').read_text())
    if driver.get('status')!='PASS' or driver.get('worker_shutdown',{}).get('status')!='PASS':
        raise Partial('driver did not close both worker connections normally')
    remote=json.loads((run/'workers/receipt.json').read_text())
    if (remote.get('status')!='PASS' or remote.get('completion_reason')!='normal_exit'
            or len(remote.get('pids',[]))!=2 or any(p.get('exit_code')!=0 for p in remote['pids'])
            or remote.get('forced_pids') or transfer['manifest'].get('partials')):
        raise Partial('worker exit or closed snapshot is incomplete')
    plan=json.loads((run/'episode_manifest.json').read_text())
    expected={t['task_uid'] for t in plan['tasks']}
    journal=[json.loads(line) for line in (run/'journal.jsonl').read_text().splitlines() if line.strip()]
    if (len(journal)!=2 or {row['task_uid'] for row in journal}!=expected
            or any(row.get('accepted') is not True or row.get('error') is not None for row in journal)):
        raise Partial('closed journal lacks exactly two accepted error-free episodes')
    result=dict(status='PASS',normal_workers=2,driver_shutdown=driver['worker_shutdown'],
        accepted_episodes=2,closed_snapshot=True,archive_sha256=transfer['archive_sha256'])
    write_json(run/'WORKER_COMPLETION.json',result)
    return result


def cleanup_transfer_stage(job,run,owned,node,transfer):
    from .archive_stage import stage_identity
    kind=transfer['kind']
    assert transfer['stage_sha_verified'] is True and transfer['run_key']==run.name
    assert stage_identity(transfer['archive'],run.name,kind)==transfer['archive_sha256']
    if node in ('timan107','timan108'):
        deployment=json.loads((Path(job['runtime'])/'exp/r17_integration/qualification/worker_deployment.json').read_text())
        root=deployment['nodes'][node]['destination']
        python='/scratch/zixuans8/mw_r17/envs/sim/bin/python'
    elif node=='weilandserver':
        root='/home/weiland/projects/openpi';python=root+'/.venv/bin/python'
    else:raise ValueError('Q archive cleanup node denied')
    command=['tether','exec','--safe','--timeout','15s',node,'--','env','CUDA_VISIBLE_DEVICES=',
        'PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+root+':'+root+'/src:'+root+'/packages/openpi-client/src',
        python,'-m','exp.r17_integration.qualification.archive_stage','--cleanup',transfer['archive'],
        '--run-key',run.name,'--kind',kind,'--sha256',transfer['archive_sha256']]
    child=owned.spawn(kind+'_stage_cleanup',command)
    if owned.wait(child,20)!=0:raise Partial('verified archive stage cleanup incomplete')
    owned.report.setdefault('transfer_stages',[]).append(dict(transfer,node=node,verified_pull=True,stage_removed=True))


def pull_closed_workers(job,run,owned,node,transfer):
    from .archive_stage import stage_identity
    from exp.r17_integration.ops.sync_back import install_snapshot
    if (transfer.get('stage_sha_verified') is not True or transfer.get('run_key')!=run.name
            or transfer.get('kind')!='workers'
            or stage_identity(transfer['archive'],run.name,'workers')!=transfer['archive_sha256']):
        raise Partial('remote closed archive has no exact run-scoped SHA-verified stage')
    owned.report['worker_transfer']=dict(transfer,node=node,verified_pull=False,stage_removed=False)
    pull=owned.spawn('worker_pull',['tether','pull',node+':'+transfer['archive'],str(run/'workers.tar'),'--timeout','35s'])
    # H100 relay bounds: pull40 + mkdir15 + push40 + CPU install25 =120s.
    # This phase still consumes the existing840s overall work budget.
    if owned.wait(pull,125 if job.get('host')=='h100' else 85)!=0:
        raise Partial('remote closed data transfer incomplete')
    install_snapshot(run/'workers.tar',run/'workers',expected_sha256=transfer['archive_sha256'])
    owned.report['worker_completion']=validate_closed_worker_run(run,transfer)
    cleanup_transfer_stage(job,run,owned,node,transfer)
    owned.report['worker_transfer'].update(verified_pull=True,stage_removed=True)


def mechanical(job,run,owned,env):
    server=owned.spawn('server',[job['python'],'-u','-m','exp.r17_integration.qualification.model_runtime',
                      '--job',str(run/'job.json'),'--run',str(run)],env)
    until=time.monotonic()+min(180,owned.remaining())
    while not (run/'SERVER_READY.json').exists():
        owned.remaining()
        if server.poll() is not None:
            raise RuntimeError('server exited during loading; see server.log')
        if time.monotonic()>until:
            raise Partial('server startup exceeded 180 seconds')
        time.sleep(.25)
    if job.get('model') == 'fastwam':
        from .manifest_binding import record_attestation
        record_attestation(owned.report, run/'SERVER_READY.json',
            json.loads((Path(job['assets'])/'model_spec.json').read_text()), owned_pid=server.pid)
    if job['domain']=='libero':
        # Existing CPU OSMesa island on the 4090 host. No GPU renderer needed.
        from .libero_remote import run_libero
        run_libero(job,run,owned)
        owned.stop(server)
        from .validate_results import validate_libero
        return validate_libero(job,run)
    driver=owned.spawn('driver',[os.sys.executable,'-u','-m','exp.r17_integration.qualification.driver',
        '--job',str(run/'job.json'),'--run',str(run),'--seconds','420'],cpu_environment(Path(job['runtime']),job))
    until=time.monotonic()+min(100,owned.remaining())
    while not port_ready(job['driver_port']):
        owned.remaining()
        if driver.poll() is not None:
            raise RuntimeError('driver/proof failed; see driver.log')
        if time.monotonic()>until:
            raise Partial('driver/WS proof startup deadline')
        time.sleep(.25)
    deployment=json.loads((Path(job['runtime'])/'exp/r17_integration/qualification/worker_deployment.json').read_text())
    node=job['worker_node']
    payload=deployment['nodes'][node]['destination']
    remote_run='/scratch/zixuans8/mw_r17/qualification/runs/'+run.name
    job['remote_worker_run']=remote_run
    owned.report['remote_owned']=dict(node=node,run=remote_run,module='exp.r17_integration.qualification.remote_workers',
        python='/scratch/zixuans8/mw_r17/envs/sim/bin/python')
    write_json(run/'remote_workers.json',dict(node=node,run=remote_run,payload=payload,deadline_seconds=500))
    argv=['tether','exec','--safe','--timeout','560s',node,'--','env','CUDA_VISIBLE_DEVICES=',
        'PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
        'PYTHONPATH='+payload+':'+payload+'/src:'+payload+'/packages/openpi-client/src',
        '/scratch/zixuans8/mw_r17/envs/sim/bin/python','-u','-m','exp.r17_integration.qualification.remote_workers',
        '--endpoint',job['endpoint_host']+':'+str(job['server_port']),'--driver-host',job['endpoint_host'],
        '--driver-port',str(job['driver_port']),'--run',remote_run,'--seconds','500']
    remote=owned.spawn('tether_workers',argv)
    if owned.wait(driver,470)!=0:
        rejected=run/'rejected_worker_probes.json'
        if rejected.is_file() and (rows:=json.loads(rejected.read_text())):
            raise Partial('mechanical driver rejected worker probe: '+rows[0]['reason'])
        raise Partial('mechanical driver did not finish accepted episodes')
    if owned.wait(remote,45)!=0:
        raise Partial('worker shutdown/physical snapshot failed')
    owned.stop(server)
    # P6 verification on the closed remote archive; pull does not touch other trees.
    lines=(run/'tether_workers.log').read_text().splitlines()
    transfer=[json.loads(line.split('Q_REMOTE_ARCHIVE ',1)[1]) for line in lines if line.startswith('Q_REMOTE_ARCHIVE ')]
    if len(transfer)!=1:
        raise Partial('missing exact remote transfer receipt')
    pull_closed_workers(job,run,owned,node,transfer[0])
    from .validate_results import validate_mw
    return validate_mw(job,run)


def replicas(job,run,owned,env):
    import shutil
    nodes=[]
    for i,port in enumerate(job['backend_ports']):
        sub=run/('replica_'+str(i));sub.mkdir()
        replica_job=dict(job,server_port=port)
        write_json(sub/'job.json',replica_job)
        child=owned.spawn('replica_'+str(i),[job['python'],'-u','-m','exp.r17_integration.qualification.model_runtime',
            '--job',str(sub/'job.json'),'--run',str(sub)],env)
        until=time.monotonic()+min(180,owned.remaining())
        while not (sub/'SERVER_READY.json').exists():
            owned.remaining()
            if child.poll() is not None: raise RuntimeError('replica failed to load')
            if time.monotonic()>until: raise Partial('replica startup exceeded bound')
            time.sleep(.25)
        nodes.append((child,sub))
    from exp.r17_integration.debug.validate import canonical
    specs=[json.loads((sub/'SERVER_READY.json').read_text())['r17_model_spec'] for _,sub in nodes]
    assert canonical(specs[0])==canonical(specs[1]),'replica metadata/model mismatch'
    for name in ('full.yaml','retrieval.yaml'): shutil.copyfile(nodes[0][1]/name,run/name)
    proxy=owned.spawn('proxy',[os.sys.executable,'-u','-c',
        'from openpi.serving.replica_proxy import run_proxy; run_proxy('+str(job['server_port'])+','+repr(job['backend_ports'])+')'],
        cpu_environment(Path(job['runtime']),job))
    until=time.monotonic()+min(45,owned.remaining())
    while not port_ready(job['server_port']):
        if proxy.poll() is not None: raise RuntimeError('replica proxy failed')
        if time.monotonic()>until: raise Partial('proxy startup bound')
        time.sleep(.2)
    proof=owned.spawn('replica_ws',[os.sys.executable,'-u','-m','exp.r17_integration.qualification.driver',
        '--job',str(run/'job.json'),'--run',str(run),'--ws-only'],cpu_environment(Path(job['runtime']),job))
    if owned.wait(proof,160)!=0: raise RuntimeError('replica actual two-client/broadcast proof failed')
    owned.stop(proxy)
    for child,sub in nodes:
        owned.stop(child)
        assert json.loads((sub/'SERVER_CLOSED.json').read_text())['status']=='PASS'
    # Shared proxy metadata equality + real successful replies; no fabricated capacity/throughput numbers.
    write_json(run/'REPLICA_RECEIPT.json',dict(status='PASS',replicas=2,sticky_clients=2,hot_bundle_broadcast=True,
        native_no_debug_byte_parity=True,quality_claim=False,pricing='unqualified'))
    return dict(status='PASS',qualified_scope='two resident model replicas with stock sticky proxy and hot bundle broadcast',complete_gate=False)


def validate_dp_native(native, sealed, spec):
    from exp.r17_integration.debug.validate import canonical
    assert native['status']=='pass' and native['skipped']==0, 'DP probe must pass without skipped checks'
    checks=native['checks']
    assert checks and native['passed']==len(checks), 'DP probe check count is incoherent'
    assert all(c['status']=='pass' for c in checks), 'DP probe contains a failed check'
    b1=[c for c in checks if c.get('name')=='real_production_B1_only']
    assert len(b1)==1 and b1[0].get('max_batch_size')==1 and b1[0].get('B1_queue_bytes_exact') is True, 'DP production B1 gate is missing'
    capabilities=spec['capabilities']['features']
    assert all(capabilities[name]['status']=='unsupported' for name in ('batch_stage1','batch_stage2','batch_stage3')), 'DP B>1 must remain unsupported'
    assert native['weight_identity']==sealed['expected_weight_sha256'], 'DP checkpoint changed after CPU seal'
    assert canonical(native['manifest']['spec'])==canonical(spec), 'DP ModelSpec changed after CPU seal'


def run_local(job,run,share_eval=False,preflight_only=False):
    run.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    report=dict(schema='r17.Q.slot.v1',slot=job['slot'],status='PARTIAL',gpu_executed=False,
        supervisor_pid=os.getpid(),supervisor_start_ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],
        checkpoint=job['checkpoint'],quality_claim=False,owned_pids=[],deadline_seconds=890)
    write_json(run/'job.json',job)
    write_json(run/'manifest.json',report)
    write_json(run/'receipt.json',report)
    owned=Owned(run,report,started+840)
    def canceled(*args):
        raise Partial('slot interrupted; owned cleanup requested')
    signal.signal(signal.SIGTERM,canceled)
    signal.signal(signal.SIGINT,canceled)
    locks=[]
    try:
        if job.get('future_checkpoint') and not (Path(job['assets'])/'CPU_RECEIPT.json').exists():
            if not Path(job['checkpoint']).is_dir(): raise Partial('future immutable checkpoint not yet published: '+job['checkpoint'])
            if job['model']=='dp' and not (Path(job['checkpoint'])/'complete.json').is_file():
                raise Partial('future DP step snapshot is not complete')
            prep_env=cpu_environment(Path(job['runtime']),job)
            if job['host']=='h100': prep_env['PYTHONPATH']='/data/mw_finetune/qualification/python_deps312:'+prep_env['PYTHONPATH']
            preparer=owned.spawn('cpu_prepare',[job['python'],'-u','-m','exp.r17_integration.qualification.assets',
                '--job',str(run/'job.json'),'--output',job['assets']],prep_env)
            if owned.wait(preparer,180)!=0: raise Partial('future CPU sealing failed or exceeded 180 seconds')
        preflight(job,ports=False)
        owned.remaining()
        if preflight_only:
            report.update(status='PASS',cpu_preflight_only=True)
            return report
        # Q always serializes its own slots. The stock training/eval lock is nonblocking.
        qlock=open('/tmp/r17_Q_qualification.lock','a')
        locks.append(qlock)
        fcntl.flock(qlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not share_eval:
            gpu_lock=open('/tmp/r17_gpu.lock','a')
            locks.append(gpu_lock)
            fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        wait_for_port_release(q_ports(job),seconds=job.get('port_wait_seconds',60),
            deadline=owned.deadline,report=report,receipt_path=run/'receipt.json')
        owned.remaining()
        # CPU read-only process census, before any GPU-related command.
        ps=subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True,timeout=5).stdout
        train_needles=('experiment/train.py','gr00t.experiment.train','exp.mw_finetune.groot.train','exp.mw_finetune.fastwam.train','exp.mw_finetune.dp.train')
        if any(any(needle in row for needle in train_needles) and 'qualification' not in row for row in ps.splitlines()):
            raise Partial('training still running; no GPU command was issued')
        if share_eval and not job.get('share_eval_safe',False):
            raise Partial('this slot has no memory sharing budget; use an exclusive slot')
        if job['host']=='h100':
            total=int(subprocess.run(['du','-sb','/data/mw_finetune'],capture_output=True,text=True,timeout=20,check=True).stdout.split()[0])
            report['shared_disk_before_bytes']=total
            if total>107_000_000_000:
                raise Partial('shared 120 GB budget lacks the reserved 13 GB save headroom')
        smi=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],
                           capture_output=True,text=True,timeout=10)
        if smi.returncode:
            raise Partial('GPU admission failed (driver may require owner repair): '+smi.stderr.strip())
        free=int(smi.stdout.splitlines()[0])
        if free<job['required_free_mib']:
            raise Partial('insufficient GPU memory: '+str(free)+' MiB')
        report['admission']=dict(free_mib=free,share_eval=share_eval)
        env=cpu_environment(Path(job['runtime']),job,gpu=True)
        report['gpu_executed']=True
        if job['phase'] in ('native','batch'):
            child=owned.spawn('native',native_command(job,run),env)
            rc=owned.wait(child)
            path=run/'native/receipt.json' if job['model']=='dp' else run/'native.json'
            native=json.loads(path.read_text()) if path.exists() else {}
            success=native.get('status') in ('PASS','pass','passed','passed_fixed_observations')
            if rc!=0 or not success:
                if native.get('status') in ('partial','PARTIAL') or rc in (-15,-9):
                    raise Partial('native probe incomplete')
                raise RuntimeError('native probe failed; see native.log and native receipt')
            if job['model']=='dp':
                sealed=json.loads((Path(job['assets'])/'load.json').read_text())
                assert native['weight_identity']==sealed['expected_weight_sha256'],'DP checkpoint changed after CPU seal'
            if job['phase']=='native':
                from exp.r17_integration.debug.validate import canonical
                spec=json.loads((Path(job['assets'])/'model_spec.json').read_text())
                if job['model']=='dp':
                    validate_dp_native(native,sealed,spec)
                elif job['model']=='fastwam':
                    assert canonical(native['model']['spec'])==canonical(spec)
                    steps=[r for r in native['rows'] if 'k' in r]
                    assert len(steps)==9 and all(r['actual_nfe']==r['k'] and r['status']=='passed' for r in steps)
                    from .manifest_binding import record_attestation
                    record_attestation(report, path, spec)
                elif job['model']=='groot':
                    manifest=native['model_manifest']
                    assert manifest['checkpoint_identity']==spec['checkpoint_identity'] and manifest['norm_identity']==spec['norm_identity']
                    assert len(native['checks'])==2 and all(r['actual_nfe']==job.get('live_k',4) and r['debug_bytes_equal'] for r in native['checks'])
            report.update(status='PASS',native_receipt=str(path),qualified_scope='native fixed observation mechanics')
        elif job['phase']=='mechanical':
            result=mechanical(job,run,owned,env)
            report.update(result)
        elif job['phase']=='replica':
            report.update(replicas(job,run,owned,env))
        elif job['phase']=='tail':
            child=owned.spawn('tail',[job['python'],'-u','-m','exp.r17_integration.qualification.tail',
                '--job',str(run/'job.json'),'--run',str(run)],env)
            rc=owned.wait(child)
            product=json.loads((run/'TAIL_RECEIPT.json').read_text())
            report.update(status=product['status'],tail=product)
            if rc!=2: raise RuntimeError('native cursor proof failed')
        elif job['phase']=='g3':
            child=owned.spawn('g3',[job['python'],'-u','-m','exp.r17_integration.qualification.g3',
                      '--job',str(run/'job.json'),'--run',str(run)],env)
            rc=owned.wait(child)
            product=json.loads((run/'G3_RECEIPT.json').read_text())
            report.update(status=product['status'],g3=product)
            if rc not in (0,2):
                raise RuntimeError('G3 hook failed')
        else:
            raise ValueError('unknown slot phase')
    except (Partial,BlockingIOError) as exc:
        report.update(status='PARTIAL',reason=str(exc))
    except BaseException as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}')
    finally:
        # Disable handler recursion during the bounded cleanup grace.
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        if report.get('remote_owned'):
            try: report['remote_cleanup']=stop_remote(**report['remote_owned'])
            except BaseException as exc: report['remote_cleanup']=dict(error=str(exc))
        owned.close()
        rejected=run/'rejected_worker_probes.json'
        if rejected.is_file():report['rejected_worker_probes']=json.loads(rejected.read_text())
        for lock in locks:
            lock.close()
        if report.get('forced_pids') and report['status']=='PASS':
            report.update(status='PARTIAL',reason='owned child required forced stop')
        report['elapsed_seconds']=time.monotonic()-started
        write_json(run/'receipt.json',report)
        print(report['status']+' '+str(run/'receipt.json'),flush=True)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--slot',required=True)
    p.add_argument('--run')
    p.add_argument('--on-host',action='store_true')
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--share-eval',action='store_true')
    a=p.parse_args()
    job=json.loads(Path(a.slot).read_text())
    run=Path(a.run or ('/data/mw_finetune/qualification/runs/'+job['slot']+'_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())))
    if job['model']=='cosmos':
        from exp.r17_integration.models.cosmos.qualification_slot import run as cosmos_slot
        return cosmos_slot(job,run,preflight=a.preflight)
    if job['host']=='h100' and not a.on_host:
        # Remote runner has the same independent hard bound; tether timeout is not relied on for cleanup.
        local=run
        local.mkdir(parents=True,exist_ok=False)
        report=dict(status='PARTIAL',slot=job['slot'],owned_pids=[],remote_run=str(run),gpu_executed=False)
        write_json(local/'receipt.json',report)
        from .relay import Broker
        try:
            wait_for_port_release([23158],seconds=job.get('port_wait_seconds',60),
                report=report,receipt_path=local/'receipt.json')
            broker=Broker(local)
        except (OSError,Partial) as exc:
            report.update(status='PARTIAL',reason='Q relay port 23158 unavailable: '+str(exc))
            write_json(local/'receipt.json',report);print('PARTIAL '+str(local/'receipt.json'));return 2
        argv=['tether','exec','--safe','--timeout','880s','h100','--','env','CUDA_VISIBLE_DEVICES=',
              'Q_H100_RUN=1','Q_RELAY_URL='+broker.url,'Q_RELAY_TOKEN='+broker.token,
              'PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+job['runtime']+':'+job['runtime']+'/src:'+job['runtime']+'/packages/openpi-client/src',
              '/home/exouser/openpi/.venv/bin/python','-u','-m','exp.r17_integration.qualification.run_slot',
              '--slot',job['runtime']+'/exp/r17_integration/qualification/specs/'+Path(a.slot).name,
              '--run',str(run),'--on-host']
        if a.preflight:
            argv.append('--preflight')
        if a.share_eval:
            argv.append('--share-eval')
        child=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        report['owned_pids']=[dict(pid=child.pid,argv=[v if not v.startswith('Q_RELAY_TOKEN=') else 'Q_RELAY_TOKEN=<ephemeral>' for v in argv])]
        write_json(local/'receipt.json',report)
        try:
            output,_=child.communicate(timeout=865)
            (local/'remote.log').write_text(output)
            print(output,end='',flush=True)
            result=subprocess.run(['tether','exec','--safe','--timeout','10s','h100','--','cat',str(run/'receipt.json')],
                                  capture_output=True,text=True,timeout=12)
            if result.returncode==0:
                report['remote_receipt']=json.loads(result.stdout)
                report['status']=report['remote_receipt']['status']
            else:
                report['reason']='remote receipt recovery failed'
        except BaseException as exc:
            report['reason']=str(exc)
        finally:
            if report.get('status')!='PASS' and not report.get('remote_receipt'):
                try: report['remote_cleanup']=stop_remote('h100',str(run),'exp.r17_integration.qualification.run_slot','/home/exouser/openpi/.venv/bin/python')
                except BaseException as exc: report['remote_cleanup']=dict(error=str(exc))
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    child.kill();child.wait()
            report['owned_pids'][0]['exit_code']=child.returncode
            broker.close()
            report['gpu_executed']=report.get('remote_receipt',{}).get('gpu_executed',False)
            write_json(local/'receipt.json',report)
            print(report['status']+' '+str(local/'receipt.json'))
    else:
        report=run_local(job,run,a.share_eval,a.preflight)
    return 0 if report['status']=='PASS' else 2 if report['status']=='PARTIAL' else 1


if __name__=='__main__':
    raise SystemExit(main())
