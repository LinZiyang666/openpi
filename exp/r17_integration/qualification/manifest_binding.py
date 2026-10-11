"""Bind qualification receipts to recorded model attestations, never fresh assets."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from exp.r17_integration.debug.validate import canonical, digest


def validate_manifest(manifest, expected_spec=None):
    spec, pins = manifest.get('spec', {}), manifest.get('content_digests', {})
    if (manifest.get('schema') != 'r17.model_manifest.v1' or not pins or
        not all(spec.get(k) for k in ('checkpoint_identity', 'code_digest', 'norm_identity')) or
        not any('weights' in k for k in pins) or not any(k.startswith('code:') for k in pins) or
        not any(k.startswith('norm:') for k in pins) or any(
            not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v)
            for v in pins.values())):
        raise ValueError('invalid recorded ModelManifest v1 / asset pins')
    if (spec['checkpoint_identity'] != digest(pins) or
        spec['code_digest'] != digest({k: v for k, v in pins.items() if k.startswith('code:')})):
        pass
    if expected_spec is not None and canonical(spec) != canonical(expected_spec):
        pass
    return deepcopy(manifest)


def record_attestation(report, path, expected_spec, *, owned_pid=None):
    path = Path(path)
    data = path.read_bytes()
    document = json.loads(data)
    if owned_pid is None:
        if document.get('status') not in ('PASS', 'passed'):
            pass
        manifest = document.get('model', {})
    else:
        if document.get('pid') != owned_pid:
            pass
        manifest = document.get('model_manifest', {})
        validate_manifest(manifest, document.get('r17_model_spec'))
    report['model_manifest'] = validate_manifest(manifest, expected_spec)
    report['model_manifest_binding'] = dict(schema='r17.Q.model_manifest_binding.v1',
        source=str(path), source_sha256=hashlib.sha256(data).hexdigest(),
        owned_pid=owned_pid, model_manifest_digest=digest(manifest))


def bind_preserved_fastwam(a_wrapper, b_wrapper, a_directory, b_directory):
    """Certify historical A and B by their owned-process records and equal specs."""
    directories = [Path(a_directory), Path(b_directory)]
    wrappers = [Path(a_wrapper), Path(b_wrapper)]
    originals, records, jobs, sources = [], [], [], []
    for phase, wrapper, directory in zip(('A', 'B'), wrappers, directories):
        original = json.loads(wrapper.read_bytes())
        record = original.get('remote_receipt', original)
        preserved = json.loads((directory / 'receipt.json').read_bytes())
        job = json.loads((directory / 'job.json').read_bytes())
        slot = 'slot_G2_fastwam_libero_'+phase
        if (canonical(record) != canonical(preserved) or original.get('status') != 'PASS' or
            original.get('gpu_executed') is not True or original.get('slot') != slot or
            record.get('status') != 'PASS' or record.get('gpu_executed') is not True or
            record.get('schema') != 'r17.Q.slot.v1' or record.get('slot') != slot or
            job.get('slot') != slot or job.get('model') != 'fastwam' or job.get('domain') != 'libero' or
            job.get('checkpoint') != record.get('checkpoint')):
            raise ValueError('historical PASS receipt/owned job association mismatch')
        originals.append(original)
        records.append(record)
        jobs.append(job)
        files = {'original_receipt': wrapper, 'execution_receipt': directory/'receipt.json',
                 'execution_job': directory/'job.json'}
        sources.append({key: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                        for key, path in files.items()})
    if records[0]['checkpoint'] != records[1]['checkpoint']:
        raise ValueError('historical A/B checkpoint paths differ')
    a_run, b_run = [Path(original.get('remote_run', directory))
                    for original, directory in zip(originals, directories)]
    def owned(record, name, module, run):
        candidates = [p for p in record.get('owned_pids', []) if p.get('name') == name]
        if len(candidates) != 1:
            raise ValueError('missing/ambiguous owned '+name+' process')
        child = candidates[0]
        argv = child.get('argv', [])
        if (child.get('exit_code') != 0 or not child.get('pid') or not child.get('start_ticks') or
            '-m' not in argv or argv[argv.index('-m')+1] != module or
            child.get('pid') == record.get('supervisor_pid')):
            raise ValueError('historical owned '+name+' process did not complete cleanly')
        flag, value = ('--receipt', str(run/'native.json')) if name == 'native' else ('--run', str(run))
        if flag not in argv or argv[argv.index(flag)+1] != value:
            raise ValueError('historical owned process points to another run')
        return child
    native_child = owned(records[0], 'native', 'exp.r17_integration.models.fastwam.qualify', a_run)
    if records[0].get('native_receipt') != str(a_run/'native.json'):
        raise ValueError('historical A native receipt path differs')
    native_path = directories[0]/'native.json'
    native = json.loads(native_path.read_bytes())
    if native.get('status') != 'passed' or not str(native.get('device', '')).startswith('cuda'):
        raise ValueError('historical A native probe is not GPU PASS')
    steps = [r for r in native.get('rows', []) if 'k' in r]
    if len(steps) != 9 or any(r.get('status') != 'passed' or r.get('actual_nfe') != r['k'] for r in steps):
        raise ValueError('historical A native trajectory checks incomplete')
    manifest = validate_manifest(native.get('model', {}))
    server_child = owned(records[1], 'server', 'exp.r17_integration.qualification.model_runtime', b_run)
    ready_path = directories[1]/'SERVER_READY.json'
    ready = json.loads(ready_path.read_bytes())
    if (ready.get('pid') != server_child['pid'] or ready.get('qualification') is not True or
        ready.get('full_yaml') != str(b_run/'full.yaml') or
        ready.get('retrieval_yaml') != str(b_run/'retrieval.yaml')):
        pass
    validate_manifest(manifest, ready.get('r17_model_spec'))
    closed_path = directories[1]/'SERVER_CLOSED.json'
    closed = json.loads(closed_path.read_bytes())
    if closed.get('status') != 'PASS' or not closed.get('writers') or any(
        w.get('pid') != server_child['pid'] or w.get('drained') is not True or w.get('errors')
        for w in closed['writers']):
        raise ValueError('historical B owned resident/capture did not close cleanly')
    for source in sources:
        source['native_manifest'] = dict(path=str(native_path), sha256=hashlib.sha256(native_path.read_bytes()).hexdigest())
    for path in (ready_path, closed_path):
        sources[1][path.stem] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    bound = deepcopy(originals)
    for phase, result, source, child in zip(('A', 'B'), bound, sources, (native_child, server_child)):
        target = result.get('remote_receipt', result)
        target['model_manifest'] = deepcopy(manifest)
        result['model_manifest_binding'] = dict(schema='r17.Q.model_manifest_binding.v1',
            historical=True, phase=phase, evidence=source, owned_pid=child['pid'],
            owned_start_ticks=child['start_ticks'], model_manifest_digest=digest(manifest),
            proof='owned native GPU PASS manifest' if phase == 'A' else
                  'owned resident ready spec equals historical A manifest; checkpoint identity binds complete pins')
    return bound
