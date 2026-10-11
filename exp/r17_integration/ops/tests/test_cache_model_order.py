import json
from pathlib import Path
from unittest import mock

import pytest

from exp.r17_integration.ops import cache_model_order as order,formal_eval,formal_prestage,cache_lane
from exp.r17_integration.ops.formal_process import write_json


def test_hold_refuses_a_live_campaign_and_never_signals(tmp_path,monkeypatch):
    write_json(tmp_path/'LANE_STATE.json',{'jobs':{'cosmos_libero':{'status':'RUNNING','run':str(tmp_path)}}})
    with mock.patch.object(order.os,'kill') as kill:
        with pytest.raises(ValueError,match='dispatched'):order.hold_next(tmp_path,'cosmos_libero')
        kill.assert_not_called()


def test_boundary_hold_is_before_claim_and_before_capacity_checks(tmp_path,monkeypatch):
    run=tmp_path/'job';run.mkdir();write_json(run/'DISPATCH_HOLD.json',{'reason':'coordinator'})
    monkeypatch.setattr(formal_eval,'FORMAL_ROOT',tmp_path)
    monkeypatch.setattr(formal_prestage,'validate_prepared',lambda p:({'run_name':'job'},{}))
    with pytest.raises(RuntimeError,match='dispatch hold'):formal_prestage.launch_prepared(run)
    assert not (run/'LAUNCH.json').exists()


def test_cancel_waiter_rejects_active_lane_without_signals(tmp_path):
    write_json(tmp_path/'LANE_STATE.json',{'status':'RUNNING'})
    with mock.patch.object(order.os,'kill') as kill:
        with pytest.raises(ValueError,match='pure predecessor'):order.cancel_waiter(tmp_path)
        kill.assert_not_called()


def test_waiter_cancel_signals_only_bound_parent(tmp_path,monkeypatch):
    write_json(tmp_path/'LANE_STATE.json',{'status':'WAITING_PREDECESSOR'})
    write_json(tmp_path/'QUEUE_OWNED.json',[])
    entry={'pid':1234,'start_ticks':'original'}
    monkeypatch.setattr(cache_lane,'lane_processes',lambda p:[entry])
    monkeypatch.setattr(order,'alive',lambda e:True)
    with mock.patch.object(Path,'stat',return_value=mock.Mock(st_uid=order.os.getuid())),mock.patch.object(order.os,'kill') as kill,mock.patch.object(order,'write_json'):
        record=order.cancel_waiter(tmp_path)
        kill.assert_called_once_with(1234,order.signal.SIGTERM)
        assert record['no_child_signals']


def test_stage_only_writes_config_and_orders_models(tmp_path,monkeypatch):
    inputs=[]
    for name,model in [('fast','fastwam'),('cos500','cosmos'),('cos_sizes','cosmos')]:
        p=tmp_path/name;p.mkdir();write_json(p/'PREPARED.json',{'native':name});inputs.append(p)
    monkeypatch.setattr(order,'validate_prepared',lambda p:({'model':'fastwam' if Path(p).name=='fast' else 'cosmos',
        'execution_worker_nodes':{'timan107':45,'timan1':40},'seconds':100},{}))
    with mock.patch.object(order.os,'kill') as kill:
        out=order.stage(tmp_path/'order',*inputs,tmp_path/'running_fast')
        kill.assert_not_called()
    record=json.loads((out/'ORDER.json').read_text())
    assert not record['armed'] and record['cpu_only']
    assert [j['model'] for j in record['plans'][1]['jobs']]==['cosmos','cosmos']
    assert not list(out.rglob('LAUNCH.json'))


def test_ordering_can_handoff_pending_evidence_only_after_verified_gpu_release(tmp_path,monkeypatch):
    from exp.r17_integration.ops import formal_finalize
    write_json(tmp_path/'RECEIPT.json',{'status':'PENDING_EVIDENCE','gpu_released':True,
        'driver_completion':{'verified_completion':True}})
    write_json(tmp_path/'GPU_RELEASED.json',{'owned_closed':True})
    with mock.patch.object(formal_finalize,'closed_proof') as closed,mock.patch.object(formal_finalize,'wait_seal') as seal:
        order.wait_source(tmp_path)
        closed.assert_called_once();seal.assert_not_called()
