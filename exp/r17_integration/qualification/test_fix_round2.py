"""Review B2 N1 regressions exercise real fixture bytes and fail-closed slots."""
import copy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest


LANGUAGE='pick up the black bowl between the plate and the ramekin and place it on the plate'


def test_n1_original_256_pairs_and_wrist_passthrough(tmp_path):
    import pyarrow.parquet as pq
    from PIL import Image
    from exp.r17_integration.qualification.recorded_libero import extract,validate
    from exp.r17_integration.qualification.assets import observations
    from exp.r17_integration.qualification.model_runtime import wire_observations
    source=Path('/data/mw_finetune/dp/libero_source')
    captured=tmp_path/'recorded'
    receipt=extract(source,captured,LANGUAGE)
    validate(captured)
    original=pq.ParquetFile(receipt['source']).read_row_group(0,columns=['image','wrist_image','state'],use_threads=False).slice(0,3).to_pylist()
    out=tmp_path/'assets';out.mkdir()
    job=dict(domain='libero',model='dp',assets=str(out),recorded_observations=str(captured))
    observations(job,out)
    wire=wire_observations(job)
    for i,row in enumerate(wire):
        for key,raw in (('observation/image','image'),('observation/wrist_image','wrist_image')):
            decoded=np.asarray(Image.open(io.BytesIO(original[i][raw]['bytes'])).convert('RGB'))
            assert row[key].shape==(256,256,3) and np.array_equal(row[key],decoded)
        assert not np.array_equal(row['observation/wrist_image'],row['observation/image'][::-1])
        assert np.array_equal(row['observation/state'],np.asarray(original[i]['state'],np.float32))
    fast=wire_observations(dict(job,model='fastwam'))
    assert all(np.array_equal(fast[i]['image2'],wire[i]['observation/wrist_image']) for i in range(3))
    from dataclasses import asdict
    from exp.r17_integration.qualification.benchmark_binding import benchmark_spec
    from exp.r17_integration.common.contracts import RequestContext
    from exp.r17_integration.models.fastwam.wire_policy import FastWAMWirePolicy
    from exp.r17_integration.tests.models.dp.conftest import make_adapter
    benchmark=benchmark_spec(dict(domain='libero',benchmark_spec=str(Path(__file__).parent/'specs/libero_benchmark_spec.json')))
    request=RequestContext('Q-recorded',benchmark.identity,benchmark.tasks[0][0],0,0,task_id=0,explicit_seed=18)
    stats={'observation.state':{'min':np.zeros(8),'max':np.ones(8)},'action':{'min':np.zeros(7),'max':np.ones(7)}}
    assert FastWAMWirePolicy('libero',stats).prepare(fast[0],request).data['input_image'].shape==(1,3,224,448)
    adapter=make_adapter.__wrapped__()('libero',libero_benchmark_specs=[asdict(benchmark)])
    assert adapter.prepare(wire[0],request).data['inputs']['base_0_rgb'].shape==(1,3,96,96)
    with pytest.raises(ValueError,match='256'):
        FastWAMWirePolicy('libero',stats).prepare(dict(fast[0],image=np.zeros((480,480,3),np.uint8)),request)
    with pytest.raises(ValueError,match='paired96 or raw256'):
        adapter.prepare(dict(wire[0],**{'observation/image':np.zeros((480,480,3),np.uint8)}),request)
    np.savez_compressed(out/'wire_observations.npz',**{'observation/image':np.stack([r['observation/image'] for r in wire]),
        'observation/state':np.stack([r['observation/state'] for r in wire]),'prompt':np.asarray([LANGUAGE]*3)})
    with pytest.raises(ValueError,match='paired.*wrist'): wire_observations(job)
    with pytest.raises(AssertionError,match='changed'): validate(out)


@pytest.mark.parametrize('count',[13,14])
def test_n1_dp_receipt_uses_real_probe_checks_not_old_count(count):
    from exp.r17_integration.qualification.run_slot import validate_dp_native
    spec=dict(checkpoint_identity='sealed-checkpoint',norm_identity='sealed-norm',
        capabilities=dict(features={name:dict(status='unsupported') for name in ('batch_stage1','batch_stage2','batch_stage3')}))
    seal=dict(expected_weight_sha256='sealed-weight')
    r=dict(status='pass',passed=count,skipped=0,checks=[dict(name=str(i),status='pass') for i in range(count)],
        weight_identity='sealed-weight',manifest=dict(spec=spec))
    r['checks'][0].update(name='real_production_B1_only',max_batch_size=1,B1_queue_bytes_exact=True)
    validate_dp_native(r,seal,spec)
    for mutation in (dict(skipped=1),dict(passed=count-1),dict(status='fail'),dict(weight_identity='different')):
        with pytest.raises(AssertionError): validate_dp_native(dict(r,**mutation),seal,spec)
    failed=copy.deepcopy(r);failed['checks'][-1]['status']='fail'
    with pytest.raises(AssertionError):validate_dp_native(failed,seal,spec)
    failed=copy.deepcopy(r);failed['checks'][0]['B1_queue_bytes_exact']=False
    with pytest.raises(AssertionError,match='production B1'):validate_dp_native(failed,seal,spec)


@pytest.mark.parametrize('domain',['libero','metaworld'])
def test_n1_retired_fast_batch_slot_never_starts_process_or_gpu(tmp_path,monkeypatch,domain):
    from exp.r17_integration.qualification import run_slot
    job=dict(slot='slot_G2_fastwam_'+domain+'_C_B2',phase='batch_disabled',model='fastwam',domain=domain,
             checkpoint='/immutable/unused',batch2=False)
    def forbidden(*args,**kwargs):raise AssertionError('retired slot must not start a process or inspect GPU')
    monkeypatch.setattr(run_slot.subprocess,'run',forbidden)
    monkeypatch.setattr(run_slot.subprocess,'Popen',forbidden)
    result=run_slot.run_local(job,tmp_path/'run')
    assert result['status']=='PARTIAL' and result['gpu_executed'] is False and result['owned_pids']==[]
    assert 'B>1 is disabled' in result['reason']
    with pytest.raises(run_slot.Partial,match='B>1 is disabled'):
        run_slot.native_command(dict(job,batch2=True),tmp_path)


def test_a2_g1_periodic_recipe_uses_exact_canonical_mt50_order():
    from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC
    from exp.r17_integration.qualification.benchmark_binding import recipe_settings
    from exp.r17_integration.common.benchmarks import task_table_identity,validate_task_table
    from exp.r17_integration.common.config import CacheConfig
    settings=recipe_settings(FORMAL_SPEC.identity)
    assert settings['periodic_gap']==2 and settings['target_ir'] is None
    assert settings['ordered_tasks']==[r[0] for r in FORMAL_SPEC.tasks]
    assert settings['task_table_identity']==task_table_identity(FORMAL_SPEC.identity)
    cfg=CacheConfig(adapter_binding=dict(benchmark_protocol=FORMAL_SPEC.identity))
    for key,value in settings.items():setattr(cfg.recipe,key,value)
    validate_task_table(cfg)
    cfg.recipe.ordered_tasks[0],cfg.recipe.ordered_tasks[1]=cfg.recipe.ordered_tasks[1],cfg.recipe.ordered_tasks[0]
    with pytest.raises(ValueError,match='canonical'):validate_task_table(cfg)


def test_n1_regeneration_waits_for_all_four_owner_handbacks(tmp_path):
    from exp.r17_integration.qualification.gen_slots import require_owner_fix2
    q=tmp_path/'qualification';q.mkdir()
    handbacks=tmp_path/'handbacks';handbacks.mkdir()
    for number in (1,2,4):
        (handbacks/f'HANDBACK_P{number}.md').write_text('# Owner\n\n## Fix round 2\nCPU tests passed\n')
    with pytest.raises(RuntimeError,match='waits.*P3'):require_owner_fix2(q)
    (handbacks/'HANDBACK_P3.md').write_text('# Owner\n\n## Fix round 1\nPassed\n')
    with pytest.raises(RuntimeError,match='P3'):require_owner_fix2(q)
    (handbacks/'HANDBACK_P3.md').write_text('# Owner\n\n## Fix round 2 — native release\nPassed\n')
    proofs=require_owner_fix2(q)
    assert set(proofs)=={'P1','P2','P3','P4'}
    assert proofs['P3']['sha256']==hashlib.sha256((handbacks/'HANDBACK_P3.md').read_bytes()).hexdigest()


def test_n1_regenerated_catalog_retires_fast_batch_and_pins_current_inputs(tmp_path):
    from exp.r17_integration.qualification.run_slot import native_command
    q=Path(__file__).parent
    catalog=json.loads((q/'SLOTS.json').read_text())
    assert set(catalog['owner_fix2_handbacks'])=={'P1','P2','P3','P4'}
    assert set(catalog['final_review_proofs'])=={'REVIEW_A3.md','REVIEW_B3.md'}
    for name,proof in catalog['final_review_proofs'].items():
        assert proof['sha256']==hashlib.sha256((q.parent/'review'/name).read_bytes()).hexdigest()
    fast=[j for j in catalog['slots'] if j['model']=='fastwam']
    retired=[j for j in fast if j['slot'].endswith('_C_B2')]
    assert len(retired)==2 and all(j['phase']=='batch_disabled' and j['batch2'] is False for j in retired)
    assert all(j['estimated_peak_mib']==j['required_free_mib']==0 for j in retired)
    assert all(not j.get('batch2') for j in fast)
    (tmp_path/'CPU_RECEIPT.json').write_text('{}')
    for job in fast:
        if job['phase']=='native':
            assert '--batch2' not in native_command(dict(job,assets=str(tmp_path)),tmp_path)
    for job in catalog['slots']:
        if job['domain']=='libero':
            assert job['recorded_observations'].endswith('libero_original256_fix2_v1')
            assert job['mechanics_task_name'].startswith('libero_spatial/')
    dp_assets=Path(json.loads((q/'specs/asset_dp_libero.json').read_text())['assets'])
    assert {p.name for p in dp_assets.glob('k*.yaml')}=={'k100.yaml','k2.yaml'}
    for job in catalog['slots']:
        if job['model']=='dp' and job['domain']=='metaworld':
            assert job['checkpoint']=='/data/mw_finetune/dp/mt50_policy_100000'
            assert job['diagnostic'] is False


def test_b3_dp_load_uses_p2_manifest_and_formal_pins(monkeypatch):
    from types import SimpleNamespace
    from exp.r17_integration.qualification.assets import sealed_dp_adapter
    from exp.r17_integration.qualification.benchmark_binding import benchmark_spec
    from exp.r17_integration.models.dp import loader
    from exp.r17_integration.models.dp.protocols import libero_spec_from_document
    q=Path(__file__).parent
    benchmark=benchmark_spec(json.loads((q/'specs/asset_dp_libero.json').read_text()))
    calls=[]
    adapter=SimpleNamespace(weight_identity='weight-pin', metadata=lambda:
        SimpleNamespace(norm_identity='norm-pin',checkpoint_identity='model-pin'))
    def load(spec):
        calls.append(spec)
        assert spec.device=='cpu'
        if spec.domain=='libero':
            document=spec.libero_benchmark_specs[0]
            assert document['protocol_identity']==benchmark.identity
            assert libero_spec_from_document(document,replan_steps=5).identity==benchmark.identity
        return adapter
    monkeypatch.setattr(loader,'load_adapter',load)
    assert sealed_dp_adapter(dict(checkpoint='published',domain='libero'),benchmark) is adapter
    assert len(calls)==1
    assert sealed_dp_adapter(dict(checkpoint='formal',domain='metaworld',diagnostic=False),benchmark) is adapter
    assert calls[-1].diagnostic is False
    assert (calls[-1].expected_weight_sha256,calls[-1].expected_norm_identity,
            calls[-1].expected_model_identity)==('weight-pin','norm-pin','model-pin')


def test_formal_dp_export_checks_step_mt50_and_pins(tmp_path):
    from exp.r17_integration.qualification.run_slot import validate_dp_export,Partial
    export=tmp_path/'mt50_policy_100000';export.mkdir()
    job=dict(checkpoint=str(export),diagnostic=False)
    load=dict(diagnostic=False,expected_weight_sha256='w',expected_norm_identity='n',expected_model_identity='m')
    with pytest.raises(Partial,match='complete.json'):validate_dp_export(job,load)
    (export/'complete.json').write_text(json.dumps(dict(step=30000)))
    with pytest.raises(ValueError,match='step disagree'):validate_dp_export(job,load)
    (export/'complete.json').write_text(json.dumps(dict(step=100000)))
    (export/'metadata.json').write_text(json.dumps(dict(config=dict(data=dict(repo_id='lerobot/metaworld_mt50')))))
    validate_dp_export(job,load)
    with pytest.raises(ValueError,match='all content pins'):validate_dp_export(job,dict(load,expected_norm_identity=None))
    (export/'metadata.json').write_text(json.dumps(dict(config=dict(data=dict(repo_id='ML45')))))
    with pytest.raises(ValueError,match='MT50'):validate_dp_export(job,load)
