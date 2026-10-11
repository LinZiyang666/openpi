"""G2 admission and H100-only lock composition; CPU fixtures, no deployment."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from exp.r17_integration.common.fake_adapter import fake_spec
from exp.r17_integration.debug.validate import digest,file_sha
from exp.r17_integration.ops import library_campaign as library


@pytest.fixture
def sealed(tmp_path):
    pins = {k:digest(k) for k in ('base_weights','wan_vae','code:r17:loader.py','norm:processor.json')}
    model = asdict(fake_spec(action_dim=7,views=2))
    model.update(model_id='fastwam_libero',checkpoint_identity=digest(pins),
        code_digest=digest({k:v for k,v in pins.items() if k.startswith('code:')}),norm_identity=digest('norm'))
    model['default_sampler']['num_steps']=10
    manifest = dict(schema='r17.model_manifest.v1',spec=model,content_digests=pins)
    paths=[]
    for phase in ('A','B'):
        row=dict(schema='r17.Q.slot.v1',status='PASS',slot='slot_G2_fastwam_libero_'+phase,
            gpu_executed=True,checkpoint='/immutable/fastwam',model_manifest=manifest)
        path=tmp_path/(phase+'.json'); path.write_text(json.dumps(row)); paths.append(path)
    return paths,model,dict(expected_digests=pins)


def gate(sealed):
    paths,model,descriptor=sealed
    return library.require_fastwam_g2(*paths,'/immutable/fastwam',teacher_descriptor=descriptor,teacher_model=model)


def mutate(path, edit):
    row=json.loads(path.read_text()); edit(row); path.write_text(json.dumps(row))


@pytest.mark.parametrize('relayed',(False,True))
def test_g2_pair_accepts_matching_teacher_without_p3_parity(sealed,relayed):
    if relayed:
        for path in sealed[0]:
            row=json.loads(path.read_text())
            path.write_text(json.dumps(dict(status='PASS',slot=row['slot'],gpu_executed=True,remote_receipt=row)))
    result=gate(sealed)
    assert result['status']=='PASS' and set(result['slots'])=={'A','B'}
    assert 'optional_loader_parity' not in result
    assert result['slots']['A']['receipt_sha256']==file_sha(sealed[0][0])


@pytest.mark.parametrize('phase',('A','B'))
@pytest.mark.parametrize('change',({'status':'PARTIAL'},{'status':'passed'},
    {'slot':'slot_G2_fastwam_mw_A'},{'gpu_executed':False},{'schema':'r17.fastwam.loader_parity.v1'}))
def test_failed_cpu_or_wrong_slot_cannot_admit_teacher(sealed,phase,change):
    mutate(sealed[0][phase=='B'],lambda row:row.update(change))
    with pytest.raises(ValueError,match='exact slot'):gate(sealed)


@pytest.mark.parametrize('change',({'status':'FAIL'},{'slot':'slot_G2_fastwam_libero_B'},{'gpu_executed':False}))
def test_relay_wrapper_cannot_hide_rejected_remote_receipt(sealed,change):
    path=sealed[0][0]; row=json.loads(path.read_text()); row.update(change)
    path.write_text(json.dumps(dict(status='PASS',slot='slot_G2_fastwam_libero_A',gpu_executed=True,remote_receipt=row)))
    with pytest.raises(ValueError,match='exact slot'):gate(sealed)


@pytest.mark.parametrize('field',('checkpoint_identity','code_digest','norm_identity','content_digests'))
def test_missing_digest_report_is_exact(sealed,field):
    def remove(row):
        m=row['model_manifest']
        (m if field=='content_digests' else m['spec']).pop(field)
    mutate(sealed[0][0],remove)
    with pytest.raises(ValueError,match='missing model_manifest.*'+field):gate(sealed)


@pytest.mark.parametrize('key',('base_weights','code:r17:loader.py','norm:processor.json','wan_vae'))
def test_new_teacher_seal_rejects_changed_weights_code_norm(sealed,key):
    sealed[2]['expected_digests'][key]=digest('new '+key)
    with pytest.raises(ValueError,match='teacher asset digests differ'):gate(sealed)


@pytest.mark.parametrize('field',('checkpoint_identity','code_digest','norm_identity','layout','feature_spec','default_sampler'))
def test_new_teacher_model_identity_and_native_geometry_must_match(sealed,field):
    sealed[1][field]='new-seal'
    with pytest.raises(ValueError,match='teacher model seal differs'):gate(sealed)


def test_path_and_self_consistency_and_ab_agreement(sealed):
    path=sealed[0][1]; original=path.read_text()
    mutate(path,lambda r:r.update(checkpoint='/mutable/latest'))
    with pytest.raises(ValueError,match='checkpoint path'):gate(sealed)
    path.write_text(original)
    mutate(path,lambda r:r['model_manifest']['content_digests'].update(base_weights=digest('tampered')))
    with pytest.raises(ValueError,match='does not bind'):gate(sealed)
    path.write_text(original)
    mutate(path,lambda r:r['model_manifest']['spec'].update(norm_identity=digest('other norm')))
    with pytest.raises(ValueError,match='A/B model identities'):
        library.require_fastwam_g2(*sealed[0],'/immutable/fastwam')


def test_supplied_q_shape_reports_both_slots_missing_pins_before_deploy(tmp_path,monkeypatch):
    paths=[]
    for phase in ('A','B'):
        row=dict(schema='r17.Q.slot.v1',slot='slot_G2_fastwam_libero_'+phase,status='PASS',
            gpu_executed=True,checkpoint='/immutable/fastwam')
        path=tmp_path/(phase+'.json')
        path.write_text(json.dumps(dict(slot=row['slot'],status='PASS',gpu_executed=True,remote_receipt=row)))
        paths.append(path)
    from exp.r17_integration.ops import formal_eval
    monkeypatch.setattr(formal_eval,'execute',lambda *a:pytest.fail('deployment before binding audit'))
    with pytest.raises(ValueError) as error:
        library.main(['--model','fastwam_libero','--suite','libero_spatial','--checkpoint','/immutable/fastwam',
            '--g2-a-receipt',str(paths[0]),'--g2-b-receipt',str(paths[1]),'--replicas','2','--prepare-only'])
    for phase in ('A','B'):
        assert 'slot_G2_fastwam_libero_'+phase in str(error.value)
    for field in ('checkpoint_identity','code_digest','norm_identity','content_digests'):
        assert field in str(error.value)


@pytest.mark.parametrize('suite',('libero_spatial','libero_10'))
@pytest.mark.parametrize('smoke',(False,True))
def test_cli_g2_only_preserves_two_replica_h100_bounded_staging(sealed,tmp_path,monkeypatch,suite,smoke):
    from exp.r17_integration.ops import formal_eval,library_init,library_staging
    monkeypatch.setattr(library,'LIBRARY_ROOT',tmp_path/'data/r17_library')
    monkeypatch.setattr(library_staging.socket,'gethostname',lambda:'weilandserver')
    monkeypatch.setattr(library_init,'build_init_archive',lambda *a:{'CPU_fixture':True})
    def execute(argv,*a,**k):
        assert argv[:2]==['tether','expose']
        if argv[2]=='rm':return ''
        return json.dumps(dict(node=argv[2],name=argv[argv.index('--name')+1],public_host='CPU',port=25000 if argv[2]=='h100' else 25001))
    monkeypatch.setattr(formal_eval,'execute',execute)
    prepared=[]
    def prepare(c,d):
        prepared.append(c)
        assert c['h100_staging']==library_staging.DEFAULT_POLICY and library_staging.bounded(c)
        assert c['server_node']=='h100' and c['worker_nodes']=={'timan108':2}
        assert c['ports']==[23340,23341,23342,23359] and c['replicas']==2
        assert c['memory_budget_gib']==64 and not c['ab_qualification'] and not c['ab_descriptor']
        assert c['g2_a_receipt']==str(sealed[0][0]) and c['g2_b_receipt']==str(sealed[0][1])
        return {}
    monkeypatch.setattr(formal_eval,'prepare',prepare)
    argv=['--model','fastwam_libero','--suite',suite,'--checkpoint','/immutable/fastwam',
        '--g2-a-receipt',str(sealed[0][0]),'--g2-b-receipt',str(sealed[0][1]),'--replicas','2',
        '--workers','2','--memory-budget-gib','64','--port','23340','--driver-port','23359','--prepare-only']
    assert library.main(argv+(['--smoke'] if smoke else []))==0 and len(prepared)==1


def test_p3_optional_pair_cannot_replace_required_g2(sealed,tmp_path):
    argv=['--model','fastwam_libero','--suite','libero_10','--prepare-only']
    assert library.parse_args(argv).ab_qualification is None
    with pytest.raises(SystemExit): library.parse_args(argv+['--ab-qualification','extra.json'])
    with pytest.raises(ValueError,match='g2-a-receipt'):
        library.main(argv+['--ab-qualification',str(tmp_path/'parity.json'),'--ab-descriptor',str(tmp_path/'desc.json')])


def test_h100_gpu_lock_excludes_conflicts_and_releases_its_own_lock(tmp_path):
    from exp.r17_integration.ops.formal_gpu_lock import exclusive
    local_4090=tmp_path/'coordinator/r17_gpu.lock'; local_4090.parent.mkdir()
    h100=tmp_path/'h100/r17_gpu.lock'; h100.parent.mkdir()
    with exclusive(local_4090):
        with exclusive(h100):
            with pytest.raises(RuntimeError,match='lock is held'):
                with exclusive(h100):pytest.fail('overlapping H100 library')
        with exclusive(h100):pass
        with pytest.raises(RuntimeError):
            with exclusive(local_4090):pytest.fail('4090 lock was released')


@pytest.mark.parametrize('busy',(False,True))
def test_formal_library_lock_is_remote_owned_and_precedes_residents(tmp_path,monkeypatch,busy):
    from exp.r17_integration.ops import formal_eval,library_staging
    campaign=dict(library=True,model='fastwam',server_node='h100',run_name='CPU',
        runtime=str(tmp_path/'runtime'),server_run=str(tmp_path/'servers'),seconds=60,
        server_campaign=str(tmp_path/'campaign.json'),h100_staging=dict(library_staging.DEFAULT_POLICY),
        ports=[23340,23341,23342,23359],driver_port=23359)
    monkeypatch.setattr(library_staging.socket,'gethostname',lambda:'weilandserver')
    events=[]
    class Pump:
        failure=None
        def __init__(self,*a):events.append('pump-init')
        def start(self):events.append('pump-start')
        def check(self):pass
        def abort(self):events.append('pump-abort')
        def summary(self):return {'CPU_fixture':True}
    monkeypatch.setattr(library_staging,'ShardPump',Pump)
    def remote(node,argv,**kwargs):
        assert node=='h100'  # No lock operation reaches the 4090 coordinator.
        if 'exp.r17_integration.ops.formal_process' in argv:
            mode=argv[argv.index('exp.r17_integration.ops.formal_process')+1]
            request=json.loads(argv[-1])
            if mode=='spawn':
                assert 'exp.r17_integration.ops.formal_gpu_lock' in request['argv']
                events.append('guard-spawn'); return json.dumps(dict(child={'pid':12345}))
            if mode=='status':return json.dumps(dict(status='CLOSED' if busy else 'RUNNING',exit_code=1 if busy else None))
            if mode=='stop':events.append('guard-stop'); return json.dumps(dict(status='closed',alive=False))
            assert mode=='renew'; return '{}'
        if argv[0]=='tail':return 'H100 GPU lock is held'
        code=argv[-1]
        if 'controls(' in code:
            return json.dumps(dict(gpu_lock=dict(status='CLOSED' if busy else 'RUNNING',exit_code=1 if busy else None)))
        if 's.admit(' in code:
            events.append('disk-admit'); return json.dumps({'status':'PASS'})
        if 'GPU_LOCK_READY.json' in code:
            events.append('guard-ready')
            return json.dumps(dict(status='PASS',campaign='CPU',pid=12345,lock='/tmp/r17_gpu.lock'))
        assert 'sockets' in code
        events.append('ports'); raise RuntimeError('CPU fixture port conflict before residents')
    monkeypatch.setattr(formal_eval,'remote',remote)
    monkeypatch.setattr(formal_eval.subprocess,'Popen',lambda *a,**k:pytest.fail('GPU process started'))
    directory=tmp_path/'campaign'; directory.mkdir()
    assert formal_eval.run_campaign(campaign,directory,{})==2
    assert events[:2]==['disk-admit','guard-spawn'] and events[-1 if busy else -2]=='guard-stop'
    assert ('pump-start' in events) is not busy
    receipt=json.loads((directory/'RECEIPT.json').read_text())
    assert receipt['owned_local']==[] and len(receipt['owned_remote'])==1
    assert receipt['owned_remote'][0]['node']=='h100' and receipt['owned_remote'][0]['label']=='gpu_lock'
    assert receipt['all_owned_closed'] is True
    if not busy:assert json.loads((directory/'H100_GPU_LOCK.json').read_text())['node']=='h100'
