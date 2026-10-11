"""Private daemon isolation/lifecycle and default-off launcher regressions."""
import json
import os
from pathlib import Path
import tempfile

import pytest

from exp.r17_integration.models.fastwam.ops import mps


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    state = dict(alive=True, servers=[], clients={}, calls=[], memory='')
    monkeypatch.setattr(mps,'ticks',lambda pid:'123')
    monkeypatch.setattr(mps,'alive',lambda entry:state['alive'] and entry['start_ticks']=='123')
    def runner(argv, **kwargs):
        state['calls'].append((argv,kwargs))
        if argv[0]=='nvidia-smi':
            if '--query-compute-apps=pid' in argv:
                return '999'
            if any('used_gpu_memory' in x for x in argv):
                return state['memory']
            return '0, GPU-test, Default, 48000'
        env=kwargs['env']
        assert env['CUDA_MPS_PIPE_DIRECTORY'] != '/tmp/nvidia-mps'
        assert env['CUDA_VISIBLE_DEVICES']=='GPU-test'
        if '-d' in argv:
            Path(env['CUDA_MPS_PIPE_DIRECTORY'],'nvidia-cuda-mps-control.pid').write_text('321')
            return ''
        text=kwargs['input'].strip()
        if text=='get_server_list':
            return ' '.join(map(str,state['servers']))
        if text.startswith('get_client_list'):
            return ' '.join(map(str,state['clients'].get(int(text.split()[1]),())))
        if text=='quit':
            state['alive']=False
            return ''
        raise AssertionError(text)
    return state,runner


@pytest.fixture
def short_directory():
    with tempfile.TemporaryDirectory(prefix='p3mps_',dir='/tmp') as root:
        yield Path(root)


def test_default_off_campaign_is_byte_identical():
    original=dict(model='fastwam',device='cuda:0',replicas=2)
    before=json.dumps(original,sort_keys=True).encode()
    assert mps.bind(original,False) is original
    assert json.dumps(original,sort_keys=True).encode()==before


def test_all_auxiliary_mps_modules_import_without_gpu_processes(monkeypatch):
    import importlib
    import subprocess
    def forbidden(*args,**kwargs):
        raise AssertionError('a module import started an external process')
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    for name in ('mps_probe','mps_exec_check','mps_h100_check','mps_h100_probe'):
        importlib.import_module('exp.r17_integration.models.fastwam.ops.'+name)


def test_exec_replica_strips_unrelated_inherited_mps_controls(fake,short_directory,monkeypatch):
    _,runner=fake
    s=mps.Session(short_directory,runner=runner)
    state=s.start()
    ready=short_directory/'READY.json';ready.write_text(json.dumps(state))
    monkeypatch.setenv('CUDA_MPS_PINNED_DEVICE_MEM_LIMIT','0=1M')
    monkeypatch.setenv('CUDA_MPS_PIPE_DIRECTORY','/other/campaign')
    seen={}
    monkeypatch.setattr(mps,'control',lambda *args:'')
    monkeypatch.setattr(mps.os,'execvpe',lambda binary,argv,env:seen.update(binary=binary,argv=argv,env=env))
    mps.exec_replica(ready,['/native/python','--cuda-graph'])
    assert seen['argv']==['/native/python','--cuda-graph']
    assert 'CUDA_MPS_PINNED_DEVICE_MEM_LIMIT' not in seen['env']
    assert seen['env']['CUDA_MPS_PIPE_DIRECTORY']==state['pipe_directory']
    assert seen['env']['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE']=='100'
    assert os.environ['CUDA_MPS_PIPE_DIRECTORY']=='/other/campaign'


def test_explicit_flag_and_default_launcher(monkeypatch):
    from exp.r17_integration.ops.formal_eval import parse_args,server_env
    argv=['--model','fastwam_libero','--replicas','1','--workers','2','--smoke','--run']
    assert not parse_args(argv).mps
    assert parse_args(argv+['--mps']).mps
    with pytest.raises(SystemExit):
        parse_args(argv+['--mps','--gpu','cuda:1'])
    assert not any(k.startswith('CUDA_MPS_') for k in server_env(dict(model='fastwam',server_node='local'),'/code'))


def test_private_start_close_and_context_identity(fake,short_directory,monkeypatch):
    state,runner=fake
    monkeypatch.setenv('CUDA_MPS_PIPE_DIRECTORY','/someone/else')
    monkeypatch.setenv('CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING','1')
    s=mps.Session(short_directory,runner=runner)
    ready=s.start()
    assert ready['gpu']['compute_mode']=='Default'
    assert Path(ready['pipe_directory']).is_relative_to(short_directory)
    assert Path(ready['log_directory']).stat().st_mode & 0o777 == 0o700
    assert 'CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING' not in mps.private_env(ready)
    assert os.environ['CUDA_MPS_PIPE_DIRECTORY']=='/someone/else'
    closed=s.close()
    assert closed['daemon_closed'] and closed['background_contexts_preserved']
    count=len(state['calls'])
    assert s.close()==closed and len(state['calls'])==count
    assert not any('-c' in argv for argv,_ in state['calls'])


def test_existing_private_directory_refused(fake,short_directory):
    _,runner=fake
    s=mps.Session(short_directory,runner=runner)
    s.start()
    try:
        with pytest.raises(FileExistsError):
            mps.Session(short_directory,runner=runner).start()
    finally:
        s.close()


@pytest.mark.parametrize('mode',['Exclusive_Process','Prohibited'])
def test_compute_mode_never_reconfigured(mode):
    with pytest.raises(ValueError,match='never changes'):
        mps.select_gpu('cuda:0',None,[dict(index=0,uuid='GPU-test',compute_mode=mode)])


def test_physical_device_pinned_by_uuid():
    rows=[dict(index=i,uuid=f'GPU-{i}',compute_mode='Default') for i in range(2)]
    assert mps.select_gpu('cuda:0','GPU-1',rows)['index']==1
    assert mps.select_gpu('cuda:1','1,0',rows)['index']==0
    with pytest.raises(ValueError):
        mps.select_gpu('cuda:1','GPU-1',rows)


@pytest.mark.parametrize('mutation',['pid','permissions','start'])
def test_other_or_reused_daemon_never_receives_quit(fake,short_directory,mutation):
    state,runner=fake
    s=mps.Session(short_directory,runner=runner)
    ready=s.start()
    if mutation=='pid':
        Path(ready['pipe_directory'],'nvidia-cuda-mps-control.pid').write_text('987')
    elif mutation=='permissions':
        Path(ready['pipe_directory']).chmod(0o755)
    else:
        ready['daemon']['start_ticks']='456'
    if mutation=='start':
        with pytest.raises(RuntimeError,match='identity'):
            mps.control(ready,'quit',runner)
    else:
        assert not s.close()['daemon_closed']
    assert not any(k.get('input')=='quit\n' for _,k in state['calls'])


@pytest.mark.parametrize('representation',['clients','pool'])
def test_mps_attachment_and_physical_accounting(fake,short_directory,representation):
    state,runner=fake
    s=mps.Session(short_directory,runner=runner)
    ready=s.start()
    state['servers'],state['clients']=[444],{444:[555,666]}
    state['memory']='555, 500\n666, 600\n444, 100\n999, 2000' if representation=='clients' else '444, 1200\n999, 2000'
    try:
        account=mps.account(ready,[555,666],runner)
        assert account['owned_total_mib']==1200
        assert '999' not in account['per_pid_mib']
        state['clients'][444].append(999)
        with pytest.raises(RuntimeError,match='exactly match'):
            mps.account(ready,[555,666],runner)
    finally:
        s.close()


def test_partial_accounting_is_rejected(fake,short_directory):
    state,runner=fake
    s=mps.Session(short_directory,runner=runner)
    ready=s.start()
    state.update(servers=[444],clients={444:[555,666]},memory='555, 500\n444, 100')
    try:
        with pytest.raises(RuntimeError,match='incomplete'):
            mps.account(ready,[555,666],runner)
    finally:
        s.close()


def test_failed_start_rolls_back_receipt_bound_daemon(fake,short_directory):
    state,base=fake
    def runner(argv,**kw):
        result=base(argv,**kw)
        if '-d' in argv:
            raise RuntimeError('simulated post-fork failure')
        return result
    s=mps.Session(short_directory,runner=runner)
    with pytest.raises(RuntimeError,match='post-fork'):
        s.start()
    assert s.state['daemon_closed']
    assert any(k.get('input')=='quit\n' for _,k in state['calls'])


def test_resume_refuses_mixed_mps_treatment(tmp_path):
    campaign=dict(model='dp')
    (tmp_path/'campaign.json').write_text(json.dumps(campaign))
    mps.check_resume(campaign,tmp_path)
    mps.bind(campaign,True)
    with pytest.raises(ValueError,match='MPS treatment'):
        mps.check_resume(campaign,tmp_path)
    (tmp_path/'campaign.json').write_text(json.dumps(campaign))
    mps.check_resume(campaign,tmp_path)
    with pytest.raises(ValueError):
        mps.check_resume(dict(model='dp'),tmp_path)


def test_mps_timing_scope_does_not_claim_cost_parity_against_native_control(tmp_path):
    (tmp_path/'campaign.json').write_text(json.dumps(dict(model='fastwam')))
    campaign=dict(model='fastwam',paired_control=str(tmp_path),mps=mps.config())
    scope=mps.comparison_scope(campaign)
    assert scope['changes_model_calls_or_nfe'] is False
    assert scope['affects_wall_time'] and scope['wall_time_comparable_with_control'] is False
    (tmp_path/'campaign.json').write_text(json.dumps(campaign))
    assert mps.comparison_scope(campaign)['wall_time_comparable_with_control'] is True


def test_long_campaign_pipe_path_rejected_before_daemon_start():
    with pytest.raises(ValueError,match='shorter campaign'):
        mps.paths('/tmp/'+'a'*100,'GPU-test')


def test_normal_launcher_mps_autoname_fits_private_socket(monkeypatch,short_directory):
    from exp.r17_integration.ops import formal_eval as launch
    captured={}
    monkeypatch.setattr(launch,'FORMAL_ROOT',short_directory)
    monkeypatch.setattr(launch,'prepare',lambda campaign,directory:captured.update(campaign=campaign,directory=directory) or {})
    assert launch.main(['--model','fastwam_libero','--mps','--prepare-only'])==0
    c=captured['campaign']
    assert c['run_name'].startswith('mps_fastwam_libero_') and c['mps']==mps.config()
    mps.paths(captured['directory'],'GPU-test')
    with pytest.raises(ValueError,match='shorter campaign'):
        launch.main(['--model','fastwam_libero','--mps','--prepare-only','--run-name','long_'+'a'*100])
    with pytest.raises(ValueError,match='matching --run-name'):
        launch.main(['--model','fastwam_libero','--mps','--prepare-only','--output',str(short_directory/'other')])


def test_mps_helpers_remain_outside_native_model_code_seal():
    import hashlib
    from exp.r17_integration.models.fastwam import loader
    path=Path(loader.__file__).parent
    descriptor=json.loads((path/'measurements/mps_descriptor.json').read_text())
    expected={k:v for k,v in descriptor['expected_digests'].items() if k.startswith('code:r17:')}
    actual={'code:r17:'+p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.glob('*.py')}
    # The later exclusive-H100 capacity task explicitly extends the opt-in
    # BatchConfig limit from16 to32. It requires a fresh control/code seal;
    # MPS helpers themselves still add no native model code inputs.
    # The subsequent CPU-only cache task installs a lazy owner library hook
    # at provider registration. It also needs a fresh runtime/content seal.
    approved_later_changes={'code:r17:batch_runtime.py','code:r17:provider.py'}
    assert {k:v for k,v in actual.items() if k not in approved_later_changes}=={
        k:v for k,v in expected.items() if k not in approved_later_changes}
    from exp.r17_integration.models.fastwam.ops.library_evidence import EVIDENCE
    assert actual==EVIDENCE['executing_r17_digests']
    assert not any('mps' in key for key in expected)


def test_default_off_cli_and_server_env_equal_before_snapshot():
    import importlib.util
    from exp.r17_integration.ops import formal_eval
    path=Path(mps.__file__).parents[1]/'measurements/mps_before/formal_eval.py.before'
    from importlib.machinery import SourceFileLoader
    name='exp.r17_integration.ops._p3_before_mps'
    loader=SourceFileLoader(name,str(path))
    spec=importlib.util.spec_from_loader(name,loader)
    old=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    args=['--model','dp_libero','--max-batch','16','--replicas','1','--prepare-only']
    current=vars(formal_eval.parse_args(args));assert current.pop('mps') is False
    previous=vars(old.parse_args(args))
    assert {k:current[k] for k in previous}==previous
    # Q's concurrently added runtime_from/driver_mode flags have their own
    # regression suite. Compare every pre-existing argument and our new flag.
    for model in ('dp','fastwam','cosmos'):
        c=dict(model=model,server_node='local',vendor='/vendor')
        assert formal_eval.server_env(c,'/runtime')==old.server_env(c,'/runtime')


@pytest.mark.parametrize('fail_ready',[False,True])
def test_normal_launcher_replica_only_env_accounting_and_failure_cleanup(tmp_path,monkeypatch,fail_ready):
    """Execute the actual launcher orchestration, substituting CPU services."""
    from contextlib import nullcontext
    import socket
    import sys
    from exp.r17_integration.ops import formal_eval as launch
    from exp.r17_integration.common import websocket_policy_server as websocket
    children=[]
    class Child:
        def __init__(self,argv,**kw):
            self.pid=400+len(children);self.argv=argv;self.env=kw['env'];self.returncode=None
            children.append(self)
        def poll(self):return self.returncode
        def wait(self,timeout):self.returncode=0;return 0
    def stop(entry,**kw):
        next(c for c in children if c.pid==entry['pid']).returncode=0
        return dict(pid=entry['pid'],status='closed')
    def execute(argv,**kw):
        if 'exp.r17_integration.models.fastwam.ops.mps' in argv:
            mode=argv[argv.index('exp.r17_integration.models.fastwam.ops.mps')+1]
            if mode=='state':
                closed=bool(children and children[0].returncode is not None)
                return json.dumps(dict(status='CLOSED' if closed else 'RUNNING',daemon_closed=closed,
                    gpu=dict(uuid='GPU-test'),pipe_directory='/private/p',log_directory='/private/l',initial=dict(servers=[])))
            assert mode=='account'
            assert argv[argv.index('--clients')+1]==str(children[1].pid)
            return json.dumps(dict(per_pid_mib={str(children[1].pid):10,'900':1},owned_total_mib=11,
                                   clients={'900':[children[1].pid]},servers=[900]))
        if argv[0]=='nvidia-smi':
            return f'{children[1].pid}, 10\n900, 1'
        return '{}'
    class Client:
        def __init__(self,*args):pass
        def get_server_metadata(self):
            return dict(r17_model_spec={},native_parity_self_check=dict(passed=not fail_ready))
        def close(self):pass
    monkeypatch.setattr(launch.subprocess,'Popen',Child)
    monkeypatch.setattr(launch,'execute',execute)
    monkeypatch.setattr(launch,'ticks',lambda pid:'123')
    monkeypatch.setattr(launch,'alive',lambda entry:any(c.pid==entry['pid'] and c.returncode is None for c in children))
    monkeypatch.setattr(launch,'stop_group',stop)
    monkeypatch.setattr(launch,'host_python',lambda c:sys.executable)
    monkeypatch.setattr(launch,'refresh_remote_owned',lambda *args:0)
    monkeypatch.setattr(launch.socket,'create_connection',lambda *args,**kwargs:nullcontext())
    monkeypatch.setattr(websocket,'WebsocketClientPolicy',Client)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    assets=tmp_path/'assets';assets.mkdir();(assets/'model_spec.json').write_text('{}')
    campaign=dict(model='cosmos',domain='libero',server_node='local',run_name='cpu_mps',runtime='/runtime',
        vendor='/vendor',seconds=120,device='cuda:0',server_run=str(tmp_path/'servers'),
        local_assets=str(assets),assets=str(assets),backend_ports=[23171],ports=[23170,23171,port],
        driver_port=port,endpoint_host='127.0.0.1',native_parity_only=True,replicas=1,
        memory_budget_gib=8,mps=mps.config(),smoke=False)
    rc=launch.run_campaign(campaign,tmp_path,{})
    receipt=json.loads((tmp_path/'RECEIPT.json').read_text())
    assert rc==(2 if fail_ready else 0)
    assert len(children)==2 and all(c.returncode==0 for c in children)
    assert not any(k.startswith('CUDA_MPS_') for k in children[0].env)
    assert children[1].env['CUDA_MPS_PIPE_DIRECTORY']=='/private/p'
    assert receipt['all_owned_closed'] and receipt['mps']['closure']['daemon_closed']
    if not fail_ready:
        assert receipt['mps']['attachments'][0]['owned_total_mib']==11
