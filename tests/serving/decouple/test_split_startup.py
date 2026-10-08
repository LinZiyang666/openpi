"""CPU launch proofs: private Unix IPC only, no network model server."""
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import stat

import pytest

from exp.decouple.cpu_startup_check import check
from exp.decouple.package_config import REPO, canonical, make_arm
from openpi.serving.split_socket import MAX_PATH_BYTES, POINTER, endpoint_ready, private_endpoint, read_endpoint
from tests.serving.decouple.test_collection_recovery import cache_with_proofs


def test_launcher_socket_setup_with_long_runtime_binds_and_cleans_up(tmp_path):
    runtime = tmp_path/('long_'+'x'*80)/('nested_'+'y'*80)
    runtime.mkdir(parents=True)
    assert len(os.fsencode(runtime/'stages.sock'))>108
    with private_endpoint(runtime,environ={}) as endpoint:
        assert len(os.fsencode(endpoint))<=MAX_PATH_BYTES
        assert stat.S_IMODE(endpoint.parent.stat().st_mode)==0o700
        assert stat.S_IMODE((runtime/POINTER).stat().st_mode)==0o600
        assert read_endpoint(runtime)==endpoint
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as listener:
            listener.bind(str(endpoint));listener.listen(1)
            assert not endpoint_ready(endpoint,runtime/'blueprint.pkl')
            (runtime/'blueprint.pkl').write_bytes(b'cpu blueprint')
            assert endpoint_ready(endpoint,runtime/'blueprint.pkl')
            connected,_=listener.accept();connected.close()
    assert not endpoint.parent.exists()
    assert json.loads((runtime/POINTER).read_text())['endpoint']==str(endpoint)


@pytest.mark.parametrize('kind',['long','permissions','absent'])
def test_invalid_xdg_runtime_falls_back_to_short_tmp(tmp_path,kind):
    xdg = tmp_path/('x'*90)/('y'*90) if kind=='long' else tmp_path/'xdg'
    if kind!='absent':
        xdg.mkdir(parents=True,mode=0o700)
        if kind=='permissions':
            xdg.chmod(0o755)
    runtime=tmp_path/'run';runtime.mkdir()
    with private_endpoint(runtime,environ={'XDG_RUNTIME_DIR':str(xdg)}) as endpoint:
        assert endpoint.parent.parent==Path('/tmp')
        assert read_endpoint(runtime)==endpoint


def test_socket_allocations_are_unique_even_for_one_runtime_and_exception_cleanup(tmp_path):
    with private_endpoint(tmp_path,environ={}) as first:
        try:
            with private_endpoint(tmp_path,environ={}) as second:
                assert first!=second
                raise RuntimeError('startup failure')
        except RuntimeError:
            pass
        assert not second.parent.exists()
        assert first.parent.exists()
    assert not first.parent.exists()


def test_child_rejects_insecure_socket_pointer(tmp_path):
    with private_endpoint(tmp_path,environ={}) as endpoint:
        endpoint.parent.chmod(0o755)
        with pytest.raises(PermissionError,match='private IPC endpoint'):
            read_endpoint(tmp_path)
        endpoint.parent.chmod(0o700)
        (tmp_path/POINTER).chmod(0o644)
        with pytest.raises(PermissionError,match='0600'):
            read_endpoint(tmp_path)


@pytest.mark.parametrize('option,value',[('--port','65535'),('--port','0'),('--startup-timeout','0')])
def test_invalid_startup_settings_fail_before_any_child(monkeypatch,option,value):
    from openpi.serving import split,split_runtime
    monkeypatch.setattr(split_runtime,'validate_arguments',lambda *a:None)
    monkeypatch.setattr(split.subprocess,'Popen',lambda *a,**k:pytest.fail('started a child'))
    with pytest.raises(ValueError):
        split.main(['--model','groot',option,value,'--','stub.py','--cache-config','stub.yaml'])


def test_full_launcher_fresh_exec_model_stubs_both_venv_sites(tmp_path):
    config=json.loads((REPO/'exp/decouple/benchmark.local.json').read_text())
    reports=check(config,tmp_path)
    for model,report in reports.items():
        assert report['passed'] and report['fresh_children']==9 and report['fronts']==8
        assert report['children_reaped'] and report['socket_cleaned'] and report['tcp_listeners']==0
        assert report['runtime_path_bytes']>108
        assert all(row['memfd_tensor_equal'] and row['host']=='127.0.0.1' and
                   row['cuda_visible_devices']=='' for row in report['fronts_report'])
        if model=='groot':
            assert all(not row['jax_imported'] and row['dynamic_processor_imported'] for row in report['fronts_report'])


def test_split_benchmark_backend_explicitly_binds_loopback():
    config=json.loads((REPO/'exp/decouple/benchmark.local.json').read_text())
    argv=make_arm(config,'groot','split_8','fixture')['server_command']
    assert argv[argv.index('--host')+1]=='127.0.0.1'


def legacy_inputs(config):
    prior=copy.deepcopy(config)
    for value in prior['gate_inputs'].values():
        value['descriptor']['source'][str(REPO/'src/openpi/serving/split.py')]='OLD_STARTUP_BYTES'
        value['sha256']=hashlib.sha256(canonical(value['descriptor'])).hexdigest()
    return prior


def test_startup_scope_cache_migration_retains_exact_proof_and_bound_gpu(tmp_path):
    config=json.loads((REPO/'exp/decouple/benchmark.local.json').read_text())
    previous=legacy_inputs(config)
    cache,gpu=cache_with_proofs(previous,tmp_path)
    assert set(cache.reindex_launcher(config,previous))==set(config['gate_inputs'])
    for key in config['gate_inputs']:
        assert cache.lookup(config,key,gpu)['gpu_identity']==gpu
        assert cache.lookup(config,key,{**gpu,'uuid':'another'}) is None
    assert len(list((cache.root/'startup_migrations').glob('*.json')))==4


@pytest.mark.parametrize('changed',['source','checkpoint_files','environment'])
def test_startup_scope_migration_refuses_any_numerical_input_change(tmp_path,changed):
    config=json.loads((REPO/'exp/decouple/benchmark.local.json').read_text())
    previous=legacy_inputs(config)
    cache,_=cache_with_proofs(previous,tmp_path)
    for value in config['gate_inputs'].values():
        value['descriptor'][changed]['numerical_change']='changed'
        value['sha256']=hashlib.sha256(canonical(value['descriptor'])).hexdigest()
    assert cache.reindex_launcher(config,previous)==[]
