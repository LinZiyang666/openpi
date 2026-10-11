from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

from exp.r17_integration.ops import formal_driver
from exp.r17_integration.ops.library_campaign import finish
from exp.r17_integration.ops.sync_back import snapshot_run, install_snapshot
from exp.r17_integration.tests.search.library_fixture import synthetic_capture


def test_closed_tiny_formal_capture_sha_sync_to_parent_finish(tmp_path,monkeypatch):
    remote = tmp_path/'remote/servers'
    source = remote/'replica_0/capture/binding_cpu'
    b,episodes,audit,names = synthetic_capture(source,episodes=2)
    (remote/'manifest.json').write_text('{"synthetic":true}')
    stage = tmp_path/'stage/closed_servers.tar'
    snapshot = snapshot_run(remote,stage)
    local = tmp_path/'local'; local.mkdir()
    verified = install_snapshot(stage,local/'servers',expected_sha256=snapshot['archive_sha256'])
    alignment = local/'ALIGNMENT.json'; alignment.write_text(json.dumps(audit))
    campaign = dict(library=True,smoke=True,model='dp',suite='libero_spatial',server_node='h100',
        alignment=str(alignment),synthetic=True,library_parent=str(local/'bpool'))
    receipt = dict(status='PASS',all_owned_closed=True,transfers={'h100_servers':{'verified':verified}})
    (local/'RECEIPT.json').write_text(json.dumps(receipt))
    driver = local/'driver'; driver.mkdir()
    terminals = [dict(task_uid=e['task_uid'],attempt=1,accepted=True,status='done',success=e['success'],error=None) for e in episodes]
    steps = [dict(task_uid=e['task_uid'],attempt=1,accepted=True,decision_seq=s) for e in episodes for s in (0,1)]
    (driver/'journal.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in terminals))
    (driver/'per_step.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in steps))
    tasks = [SimpleNamespace(extra=e,task_uid=e['task_uid'],orig_init_state_idx=e['orig_init_idx']) for e in episodes]
    monkeypatch.setattr(formal_driver,'plan',lambda c:(SimpleNamespace(identity=b['benchmark_protocol'],ordered_tasks=names),b['model'],tasks,[]))
    result = finish(campaign,local)
    assert result['status']=='PASS' and result['episodes']==2 and result['rows']==4 and result['failed_episodes']==1
    assert result['fits']=={} and result['actual_gpu_collection'] is False
    assert finish(campaign,local)==result
    for mutation in ({'status':'PARTIAL'},{'all_owned_closed':False},{'transfers':{}}):
        (local/'RECEIPT.json').write_text(json.dumps(dict(receipt,**mutation)))
        with pytest.raises(ValueError): finish(campaign,local)
