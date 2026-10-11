"""CPU-only Cosmos baseline driver and frozen Q3 metric portability gates."""
import copy
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from exp.r17_integration.ops.cache_driver_legacy import LegacyCacheDriver, SynchronousEvidence
from exp.r17_integration.qualification.driver import ProbeRecordingDriver
from exp.r17_integration.search.cache_recipe import metric_mapping, validate_deployed_stall_metric
from exp.offline_search.rounds.r06.ideation_Q3.stall.stall import _digest


@pytest.fixture
def metric():
    rng = np.random.default_rng(81)
    left = rng.normal(size=(8,13)).astype(np.float32)
    right = rng.normal(size=(13,7)).astype(np.float32)
    recipe = SimpleNamespace(tasks={0:SimpleNamespace(rows=np.arange(8), Z=left, early=SimpleNamespace(A0=right))},
        fit_evidence={'source_digest':'sealed-B500'},layout=SimpleNamespace(identity='sealed-layout'))
    mapping = metric_mapping(recipe)
    model = SimpleNamespace(metric=copy.deepcopy(mapping),provenance={'metric_sha256':_digest(mapping)})
    return recipe,model


def test_saved_metric_byte_exact_or_only_dot_arithmetic(metric):
    recipe,model = metric
    assert validate_deployed_stall_metric(recipe,model)['mode'] == 'byte_exact'
    value = model.metric['tasks']['0']['codes']['early']
    value[:] = np.nextafter(value, np.float32(np.inf))
    model.provenance['metric_sha256'] = _digest(model.metric)
    before = _digest(model.metric)
    evidence = validate_deployed_stall_metric(recipe,model)
    assert evidence['mode'] == 'float32_dot_forward_error_bound'
    assert evidence['frozen_calibration_unchanged'] and evidence['checks'][0]['changed'] > 0
    assert _digest(model.metric) == before


@pytest.mark.parametrize('damage', ['rows','main','provenance','regime','dtype','large_early','saved_checksum'])
def test_metric_drift_other_than_verified_float32_dot_is_rejected(metric,damage):
    recipe,model = metric
    table = model.metric['tasks']['0']
    if damage == 'rows': table['rows'][0] = 99
    elif damage == 'main': table['codes']['main'][0,0] = np.nextafter(table['codes']['main'][0,0],np.float32(np.inf))
    elif damage == 'provenance': model.metric['provenance']['source_digest'] = 'other-B500'
    elif damage == 'regime': table['codes'].pop('early')
    elif damage == 'dtype': table['codes']['early'] = table['codes']['early'].astype(np.float64)
    elif damage == 'large_early': table['codes']['early'][0,0] += 1
    elif damage == 'saved_checksum': model.provenance['metric_sha256'] = 'corrupt'
    if damage != 'saved_checksum': model.provenance['metric_sha256'] = _digest(model.metric)
    with pytest.raises(ValueError): validate_deployed_stall_metric(recipe,model)


def test_synchronous_evidence_fsyncs_every_row(tmp_path,monkeypatch):
    calls = []
    monkeypatch.setattr('exp.r17_integration.ops.cache_driver_legacy.os.fsync',lambda fd:calls.append(fd))
    evidence = SynchronousEvidence(tmp_path)
    evidence.submit({'per_step.jsonl':[{'decision':0},{'decision':1}]})
    assert len(calls) == 2
    assert [json.loads(s) for s in (tmp_path/'per_step.jsonl').read_text().splitlines()] == [{'decision':0},{'decision':1}]
    assert evidence.close()['mode'] == 'legacy' and evidence.close()['per_row_fsync']


def test_legacy_driver_uses_baseline_result_fence_and_worker_identity(tmp_path,monkeypatch):
    forwarded = []
    monkeypatch.setattr(ProbeRecordingDriver,'handle_result',lambda self,row:forwarded.append(row))
    driver = object.__new__(LegacyCacheDriver)
    driver.evidence = SynchronousEvidence(tmp_path)
    driver._shutdown_local = SimpleNamespace(worker_id='ours')
    driver._scheduler = SimpleNamespace(_lock=threading.RLock(),_ep_index={'uid':('stage',None)},
        _rt={'stage':SimpleNamespace(dispatched={'uid'})},_dispatch_gen={'uid':2})
    for attempt in (1,2):
        driver.handle_result(dict(task_uid='uid',attempt=attempt,seed=195,orig_init_state_idx=48,
                                  success=True,error=None,per_step_rows=[]))
    rows = [json.loads(s) for s in (tmp_path/'attempts.jsonl').read_text().splitlines()]
    assert [r['accepted'] for r in rows] == [False,True]
    assert all(r['worker_id']=='ours' and r['seed']==195 and r['orig_init_state_idx']==48 for r in rows)
    assert len(forwarded) == 2 and 'per_step_rows' not in rows[0]
