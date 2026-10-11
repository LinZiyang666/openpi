"""CPU proofs for live and preserved Fast-WAM model-manifest bindings."""
from copy import deepcopy
import json

import pytest

from .assets import write_json
from .manifest_binding import bind_preserved_fastwam, record_attestation, validate_manifest
from exp.r17_integration.debug.validate import digest


def fixture(tmp_path):
    pins = {'base_weights': '1'*64, 'code:fixture': '2'*64, 'norm:fixture': '3'*64,
            'wan_vae': '4'*64, 'tokenizer:fixture': '5'*64}
    spec = dict(model_id='fastwam_libero', checkpoint_identity=digest(pins),
                code_digest=digest({'code:fixture': pins['code:fixture']}), norm_identity=digest('norm'),
                layout={'horizon': 16}, feature_spec={'fields': {'vision_0': 64}}, default_sampler={'num_steps': 10})
    manifest = dict(schema='r17.model_manifest.v1', spec=spec, content_digests=pins)
    directories, wrappers = [], []
    for phase, pid, name, module in [('A', 1001, 'native', 'exp.r17_integration.models.fastwam.qualify'),
                                     ('B', 1002, 'server', 'exp.r17_integration.qualification.model_runtime')]:
        directory = tmp_path/phase
        directory.mkdir()
        flag, value = ('--receipt', str(directory/'native.json')) if phase == 'A' else ('--run', str(directory))
        slot = 'slot_G2_fastwam_libero_'+phase
        record = dict(schema='r17.Q.slot.v1', status='PASS', slot=slot, gpu_executed=True,
            checkpoint='/immutable/fastwam', supervisor_pid=1000,
            owned_pids=[dict(name=name, pid=pid, start_ticks='10', exit_code=0,
                            argv=['python', '-m', module, flag, value])])
        if phase == 'A':
            record['native_receipt'] = str(directory/'native.json')
            write_json(directory/'native.json', dict(status='passed', device='cuda:0', model=manifest,
                rows=[dict(k=k, actual_nfe=k, status='passed') for _ in range(3) for k in (10, 2, 1)]))
        else:
            write_json(directory/'SERVER_READY.json', dict(pid=pid, qualification=True, r17_model_spec=spec,
                full_yaml=str(directory/'full.yaml'), retrieval_yaml=str(directory/'retrieval.yaml')))
            write_json(directory/'SERVER_CLOSED.json', dict(status='PASS', writers=[dict(pid=pid, drained=True, errors=[])]))
        write_json(directory/'receipt.json', record)
        write_json(directory/'job.json', dict(slot=slot, model='fastwam', domain='libero', checkpoint='/immutable/fastwam'))
        wrapper = tmp_path/(phase+'_wrapper.json')
        write_json(wrapper, dict(slot=slot, status='PASS', gpu_executed=True, remote_run=str(directory), remote_receipt=record))
        directories.append(directory)
        wrappers.append(wrapper)
    return wrappers, directories, manifest


def test_preserved_binding_proves_native_and_owned_resident_then_passes_p6_gate(tmp_path):
    from exp.r17_integration.ops.library_campaign import require_fastwam_g2
    wrappers, directories, manifest = fixture(tmp_path)
    original = {p: p.read_bytes() for p in tmp_path.rglob('*.json')}
    bound = bind_preserved_fastwam(*wrappers, *directories)
    paths = []
    for index, result in enumerate(bound):
        assert result['remote_receipt']['model_manifest'] == manifest
        assert result['model_manifest_binding']['owned_pid'] == 1001+index
        path = tmp_path/('bound_'+str(index)+'.json')
        write_json(path, result)
        paths.append(path)
    gate = require_fastwam_g2(*paths, '/immutable/fastwam',
        teacher_descriptor={'expected_digests': manifest['content_digests']}, teacher_model=manifest['spec'])
    assert gate['status'] == 'PASS'
    assert all(path.read_bytes() == data for path, data in original.items())


@pytest.mark.parametrize('file,field,value', [
    ('SERVER_READY.json', 'pid', 999), ('SERVER_READY.json', 'qualification', False),
    ('SERVER_READY.json', 'full_yaml', '/another/run/full.yaml'),
    ('SERVER_CLOSED.json', 'status', 'FAIL'), ('native.json', 'status', 'partial'),
    ('native.json', 'device', 'cpu'), ('native.json', 'rows', []),
])
def test_preserved_binding_rejects_unproven_evidence(tmp_path, file, field, value):
    wrappers, directories, _ = fixture(tmp_path)
    path = directories[0 if file == 'native.json' else 1]/file
    document = json.loads(path.read_text())
    document[field] = value
    write_json(path, document)
    with pytest.raises(ValueError):
        bind_preserved_fastwam(*wrappers, *directories)


def test_preserved_binding_rejects_changed_pins_specs_or_wrapper(tmp_path):
    wrappers, directories, manifest = fixture(tmp_path)
    ready_path = directories[1]/'SERVER_READY.json'
    ready = json.loads(ready_path.read_text())
    ready['r17_model_spec']['default_sampler']['num_steps'] = 2
    write_json(ready_path, ready)
    with pytest.raises(ValueError, match='ModelSpec'):
        bind_preserved_fastwam(*wrappers, *directories)
    invalid = deepcopy(manifest)
    invalid['content_digests']['wan_vae'] = '6'*64
    with pytest.raises(ValueError, match='complete asset pin map'):
        validate_manifest(invalid)
    outer = json.loads(wrappers[0].read_text())
    outer['remote_receipt']['owned_pids'][0]['exit_code'] = 1
    write_json(wrappers[0], outer)
    with pytest.raises(ValueError, match='association mismatch'):
        bind_preserved_fastwam(*wrappers, *directories)


def test_live_attestations_bind_exact_manifest_and_owned_pid(tmp_path):
    _, directories, manifest = fixture(tmp_path)
    report = {}
    record_attestation(report, directories[0]/'native.json', manifest['spec'])
    assert report['model_manifest'] == manifest
    path = directories[1]/'SERVER_READY.json'
    ready = json.loads(path.read_text())
    ready['model_manifest'] = manifest
    write_json(path, ready)
    record_attestation(report, path, manifest['spec'], owned_pid=1002)
    assert report['model_manifest_binding']['owned_pid'] == 1002
    with pytest.raises(ValueError, match='owned resident PID'):
        record_attestation({}, path, manifest['spec'], owned_pid=999)
    ready.pop('model_manifest')
    write_json(path, ready)
    with pytest.raises(ValueError, match='ModelManifest'):
        record_attestation({}, path, manifest['spec'], owned_pid=1002)
