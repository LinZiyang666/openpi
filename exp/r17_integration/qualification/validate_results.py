"""Fail closed on incomplete/late/error attempts and missing physical records."""
from __future__ import annotations
import json
from pathlib import Path
from .assets import write_json


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def captures(run):
    from exp.r17_integration.debug.reader_profiles import CaptureReader
    products=[]
    for path in sorted((run/'model_capture').glob('binding_*/r17_capture.json')):
        reader=CaptureReader(path.parent)
        products += list(reader.decisions())
        assert not reader.issues,reader.issues
    assert products,'closed model capture missing'
    return products


def audit(job,run,journal,physical,expected):
    from exp.r17_integration.debug.reader_profiles import enrich_journal,accepted_join,merge_decision_ledgers
    from exp.r17_integration.debug.validate import EPISODE_KEYS
    from exp.r17_integration.debug.validate import canonical
    from exp.r17_integration.search.audit import audit_campaign
    from exp.r17_integration.search.analyze import summarize
    from exp.r17_integration.debug.analysis_profiles import write_report
    source=captures(run)
    assert all(not r.get('error') for r in journal),'campaign contains an actual worker/model error'
    spec=json.loads((Path(job['assets'])/'model_spec.json').read_text())
    import numpy as np
    for path in (run/'model_capture').glob('binding_*/r17_capture.json'):
        assert canonical(json.loads(path.read_text())['binding']['model'])==canonical(spec)
    width=spec['layout']['model_action_width'] or spec['layout']['valid_action_dim']
    for record,arrays,keys in source:
        actions=arrays['normalized_actions'];wire=arrays['wire_actions']
        assert actions.shape==(spec['layout']['horizon'],width) and np.isfinite(actions).all()
        assert wire.ndim==2 and wire.shape[1]==spec['layout']['valid_action_dim'] and np.isfinite(wire).all()
        assert set(keys)==set(spec['feature_spec']['fields'])
        for key,n in spec['feature_spec']['fields'].items(): assert keys[key].shape==(n,)
    manifests=[]
    for result in journal:
        if result.get('accepted') is not True or result.get('error') is not None: continue
        own=[p[0] for p in source if p[0].get('task_uid')==result['task_uid'] and p[0].get('attempt')==result['attempt']]
        assert own,'accepted episode lacks native captures'
        manifest={key:own[0][key] for key in EPISODE_KEYS}
        manifest['n_decisions']=len(own)
        assert sorted(r['decision_seq'] for r in own)==list(range(len(own)))
        manifests.append(manifest)
    enriched=enrich_journal(journal,manifests)
    joined=accepted_join(merge_decision_ledgers([p[0] for p in source]),enriched)
    result=audit_campaign(expected,enriched,[p[0] for p in source])
    assert result['accepted']==2 and not result['missing'] and not result['errors'],result
    assert result['cost_status']=='available','native cost/execution ledger is incomplete'
    accepted={r['task_uid']:r for r in enriched if r.get('accepted') is True and r.get('error') is None}
    terminal_acks=[]
    for uid,ep in accepted.items():
        own=[r for r in joined if r['task_uid']==uid and r['attempt']==ep['attempt']]
        controls=[r for r in physical if r.get('task_uid')==uid and r.get('attempt')==ep['attempt']]
        assert controls,'accepted episode has no applied-control ledger'
        for control in controls:
            assert all(control.get(key)==ep[key] for key in EPISODE_KEYS),'physical/accepted identity disagreement'
        assert sorted(r['control_idx'] for r in controls)==list(range(len(controls))),'physical controls are missing or duplicated'
        assert all(0<=r['decision_seq']<ep['n_decisions'] for r in controls),'physical control has unknown decision'
        count=0
        for decision in own:
            n=sum(r.get('decision_seq')==decision['decision_seq'] for r in controls)
            assert 0<n<=5,(uid,decision['decision_seq'],n)
            count+=n
            assert decision.get('executed_count')==n,'server execution ack disagrees with physical controls'
            assert decision.get('cursor_end')==decision['cursor']+n,'execution cursor disagrees'
            if decision['decision_seq']==ep['n_decisions']-1:
                assert decision.get('terminal') is True,'final server execution ack is missing'
                terminal_acks.append(dict(task_uid=uid,decision_seq=decision['decision_seq'],executed_count=n,
                    server_ack='acknowledged_terminal',source='closed server ack and client physical ledger'))
        assert count<= (160 if job['domain']=='metaworld' else 220)
        if ep['arm']=='full':
            k=json.loads((Path(job['assets'])/'model_spec.json').read_text())['default_sampler']['num_steps']
            assert all(r['stage_calls']==[1,1,k] and r['actual_nfe']==k for r in own)
        else:
            assert any(r['source']=='library' and r['stage_calls']==[1,0,0] and r['actual_nfe']==0 for r in own)
    write_json(run/'accepted_enriched.json',enriched)
    write_json(run/'accepted_audit.json',dict(result,terminal_execution=terminal_acks,quality_claim=False))
    binding=json.loads(next((run/'model_capture').glob('binding_*/r17_capture.json')).read_text())['binding']
    analysis=summarize(expected,enriched,[p[0] for p in source],task_groups={r['task_name']:'mechanics' for r in manifests},binding=binding)
    analysis['quality_claim']='two mechanics episodes; synthetic retrieval library; no 500-episode SR claim'
    write_report(analysis,run/'analysis')
    assert json.loads((run/'WS_RECEIPT.json').read_text())['status']=='PASS'
    assert json.loads((run/'SERVER_CLOSED.json').read_text())['status']=='PASS'
    return dict(status='PASS',qualified_scope='two real full/retrieval mechanics episodes, hot WS, closed captures and exact accepted/control join',
        audit=result,synthetic_library=True,complete_gate=False,terminal_execution=terminal_acks)


def validate_mw(job,run):
    from exp.r17_integration.debug.reader_profiles import CaptureReader
    from exp.r17_integration.conductor.manifest import validate_worker_probe
    plan=json.loads((run/'episode_manifest.json').read_text())
    expected=[r['task_uid'] for r in plan['tasks']]
    census=json.loads((run/'census.json').read_text())
    assert len(census)==2,'exactly two distinct workers must register'
    for worker in census.values(): validate_worker_probe(worker['probe'])
    assert not json.loads((run/'rejected_worker_probes.json').read_text())
    physical=[]
    for path in (run/'workers').rglob('r17_capture.json'):
        reader=CaptureReader(path.parent)
        for row,arrays in reader.physical():
            if row.get('event')=='control':
                assert 'applied_action' in arrays and 'requested_action' in arrays
                import numpy as np
                assert np.isfinite(arrays['applied_action']).all() and np.isfinite(arrays['requested_action']).all()
                physical.append(row)
        assert not reader.issues
    assert json.loads((run/'workers/receipt.json').read_text())['status']=='PASS'
    return audit(job,run,rows(run/'journal.jsonl'),physical,expected)


def validate_libero(job,run):
    receipt=json.loads((run/'libero/receipt.json').read_text())
    assert receipt['status']=='PASS'
    assert receipt['actual_simulator'] and receipt['renderer']=='osmesa' and receipt['episode_kernel']=='examples.libero.main._run_episode'
    return audit(job,run,receipt['journal'],rows(run/'libero/physical.jsonl'),[r['task_uid'] for r in receipt['journal']])
