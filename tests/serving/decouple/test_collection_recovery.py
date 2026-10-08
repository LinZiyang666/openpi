"""Runtime collection, offline recovery, assets, cache and smoke CPU proofs."""
import copy
import json
from pathlib import Path
import socket

import pytest

from exp.decouple.collect_gpu import collect
from exp.decouple.cpu_collection_check import check
from exp.decouple.gate_cache import GateCache, file_digest
from exp.decouple.gpu_package import run_package
from exp.decouple.package_assets import NORM_RELATIVE, prepare_assets, validate_assets
from exp.decouple.package_config import REPO, write_json, validate
from exp.decouple.package_driver import agent_lease


@pytest.fixture
def config():
    return json.loads((REPO/'exp/decouple/benchmark.local.json').read_text())


def test_recorded_worker_wire_relay_collector_and_summary(config,tmp_path):
    report=check(config,tmp_path)
    assert report['passed'] and report['interleaved_connections']==2
    for model in ('groot','pi05'):
        result=json.loads((tmp_path/model/'metrics.json').read_text())
        assert result['latency_count']==2 and result['episodes_completed']==2
        assert result['latency_missing_identity_rows']==0 and result['resource_sample_count']==1
        # Re-analysis reads the identical inputs and produces identical metrics.
        assert collect(tmp_path/model)==result
    assert 'groot/initial/current_1' in (tmp_path/'SUMMARY.md').read_text()


def test_null_attribution_is_caught_in_cpu_collection(config,tmp_path):
    check(config,tmp_path)
    path=tmp_path/'groot/latencies.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row.update(uid=None,attempt=None)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError,match='lack UID/attempt'):
        collect(tmp_path/'groot')


def test_missing_durable_utilization_fails_explicitly(config,tmp_path):
    check(config,tmp_path)
    (tmp_path/'groot/resource_samples.jsonl').write_text('')
    with pytest.raises(ValueError,match='durable utilization'):
        collect(tmp_path/'groot')


def test_norm_stats_preflight_stages_locally_and_never_writes_through_owner_link(tmp_path):
    repo,source=tmp_path/'repo',tmp_path/'main/assets'
    repo.mkdir();(source/NORM_RELATIVE).parent.mkdir(parents=True)
    original=source/NORM_RELATIVE
    content=json.dumps(dict(norm_stats=dict(state=dict(mean=[0],std=[1]),actions=dict(mean=[0],std=[1]))))
    original.write_text(content)
    (repo/'assets').symlink_to(source,target_is_directory=True)
    target=prepare_assets(repo,source)
    assert target.read_text()==content and target.resolve().is_relative_to(repo)
    assert original.read_text()==content and not (repo/'assets').is_symlink()
    target.unlink()
    with pytest.raises(FileNotFoundError,match='required norm stats'):
        validate_assets(target)


def test_missing_norm_stats_detected_before_model_launch(tmp_path):
    with pytest.raises(FileNotFoundError,match='required pi05 norm stats absent'):
        prepare_assets(tmp_path/'repo',tmp_path/'absent')


def test_worker_launch_keeps_r16_default_tether_retries(config,tmp_path):
    calls=[]
    def remote(host,argv,**kwargs):
        calls.append((argv,kwargs))
    with agent_lease(config,'groot',tmp_path,remote):
        pass
    launch=[kwargs for argv,kwargs in calls if 'nohup env' in str(argv)]
    assert launch==[{'timeout':30}]


def cache_with_proofs(config,tmp_path):
    cache=GateCache(tmp_path/'cache')
    gpu=dict(host=socket.gethostname(),index=0,name='NVIDIA GeForce RTX 4090',uuid='CPU_TEST_GPU',driver='CPU_TEST_DRIVER')
    for key in config['gate_inputs']:
        proof=fake_proof(config,key)
        cache.put(config,key,proof,gpu,path=tmp_path/(key+'.json'))
    return cache,gpu


def fake_proof(config,key):
    model=key.split('_')[0]
    return dict(model=model,k=int(key.split('_')[1][1:]),checkpoint=config['cases'][model]['checkpoint'],
                batch_size=1,rows=16,passed=True,donor_order_equal=True,
                fields={name:dict(bit_exact=True,max_abs=0,max_rel=0) for name in
                        ('action','key:vision_0','key:vision_1','key:robot_state')},
                torch=config['gate_inputs'][key]['descriptor']['torch_version'],gpu='NVIDIA GeForce RTX 4090')


def test_cache_reuses_unchanged_inputs_and_rejects_gpu_or_input_change(config,tmp_path):
    cache,gpu=cache_with_proofs(config,tmp_path)
    assert cache.lookup(config,'groot_k1_b1',gpu)['passed']
    assert cache.lookup(config,'groot_k1_b1',{**gpu,'uuid':'another'}) is None
    assert cache.lookup(config,'groot_k1_b1',{**gpu,'driver':'new'}) is None
    altered=copy.deepcopy(config);altered['gate_inputs']['groot_k1_b1']['sha256']='changed'
    assert cache.lookup(altered,'groot_k1_b1',gpu) is None


def test_cache_rejects_nonexact_proof(config,tmp_path):
    cache,gpu=cache_with_proofs(config,tmp_path)
    proof=fake_proof(config,'groot_k1_b1')
    proof['fields']['action']['bit_exact']=False
    with pytest.raises(ValueError,match='strict bit-exact'):
        cache.put(config,'groot_k1_b1',proof,gpu)


def test_weight_digest_cache_detects_changed_bytes_even_same_size(tmp_path):
    path=tmp_path/'weights';path.write_bytes(b'abcd');index={}
    first=file_digest(path,index)
    assert file_digest(path,index)==first
    path.write_bytes(b'abce')
    assert file_digest(path,index)!=first


def test_smoke_schema_and_cached_gates_then_only_two_gr00t_arms(tmp_path,capsys):
    config=json.loads((REPO/'exp/decouple/benchmark.smoke.local.json').read_text())
    validate(config,check_inputs=False)
    assert config['expected_episodes']==20 and config['wall_limit_s']==600
    assert config['cases']['groot']['manifest_sha256']==config['arms'][0]['manifest_sha256']==config['arms'][1]['manifest_sha256']
    cache,gpu=cache_with_proofs(config,tmp_path)
    state=dict(package_id='cpu_smoke',config_path='smoke.json',gates={},completed={},selected={})
    calls,now=[],[0]
    def forbidden(*args):
        pytest.fail('reran unchanged passed gates in a new package')
    def arm(spec,*args):
        calls.append((spec['model'],spec['logical_arm'],spec['expected_episodes']))
        now[0]+=90
        return dict(name=spec['name'],logical_arm=spec['logical_arm'],model=spec['model'],model_processes=1,
                    fronts=spec['fronts'],episodes_completed=20,episodes_per_min=30,elapsed_s=40,
                    gpu_util_mean_pct=50,system_cpu_mean_pct=10,cpu_proc_mean_pct=100,gpu_memory_peak_mb=7000,
                    latency_p50_ms=100,latency_p99_ms=200,manifest_sha256=spec['manifest_sha256'],within_budget=True)
    assert run_package(config,state,tmp_path,gate_runner=forbidden,arm_runner=arm,gate_cache=cache,gpu_identity=gpu,
                       clock=lambda:now[0],wall_clock=lambda:now[0],deadline_epoch=10000,watchdog=False)==0
    assert calls==[('groot','current_1',20),('groot','split_8',20)]
    output=capsys.readouterr().out
    assert output.count('PKG_GATE_SKIP')==4 and 'PKG_DONE' in output
    assert '20 accepted B episodes' in (tmp_path/'SUMMARY.md').read_text()


def test_smoke_without_cached_gates_never_launches_any_arm(tmp_path,capsys):
    config=json.loads((REPO/'exp/decouple/benchmark.smoke.local.json').read_text())
    state=dict(package_id='cpu_smoke',config_path='smoke.json',gates={},completed={},selected={})
    def forbidden(*args):
        pytest.fail('launched gate/model without smoke prerequisite')
    assert run_package(config,state,tmp_path,gate_runner=forbidden,arm_runner=forbidden,gate_cache=GateCache(tmp_path/'absent'),
                       clock=lambda:0,wall_clock=lambda:0,deadline_epoch=10000,watchdog=False)==1
    assert 'PKG_GATE_FAIL' in capsys.readouterr().out
