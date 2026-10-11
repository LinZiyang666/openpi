"""Actual native π controller cursor probe; physical replan1 protocol pending."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from .assets import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--run',required=True);a=p.parse_args()
    job,run=json.loads(Path(a.job).read_text()),Path(a.run)
    report=dict(status='PARTIAL',subset_status='PARTIAL',actual_physical_controls=False,quality_claim=False,
        controller_protocol='mw_commit_h5_replan1',pending='P2 separate physical protocol registration; formal replan5 has no tails')
    write_json(run/'TAIL_RECEIPT.json',report)
    try:
        from .model_runtime import load,wire_observations
        from exp.r17_integration.common.contracts import RequestContext
        from exp.r17_integration.common.qualification import config_for_probe
        from exp.r17_integration.common.serving import make_connection_factory
        from exp.r17_integration.common.config import build_shared_storage
        adapter,provider=load(job)
        cfg=config_for_probe(provider,10,5);cfg.checkpoints['cp1'].gate.type='always_skip'
        cfg.adapter_binding['allow_partial_commit']=True
        core=make_connection_factory(provider,cfg,build_shared_storage(cfg))()
        core.on_task_begin();core.on_episode_start(43,0,0,task='reach-v3',extra_metadata=dict(
            partial_tail_protocol='r17.partial_tail.v1',ack_protocol='r17.execution.v1'))
        obs=wire_observations(job)[0]
        initial=None;ledger=[]
        for seq in range(5):
            request=RequestContext('Q-tail-controller',adapter.metadata().allowed_benchmark_protocols[0],'reach-v3',0,seq,
                task_id=43,explicit_seed=18+seq)
            sent=dict(obs,__request__=asdict(request))
            if seq: sent['__execution__']=dict(decision_seq=seq-1,executed_count=1,terminal=False)
            reply=core.infer(sent);hit=reply['__hit_meta__'];actions=reply['actions']
            if initial is None: initial=actions.copy()
            else:
                assert hit['stage_calls']==(0,0,0) and hit['actual_nfe']==0
                assert actions.tobytes()==initial[seq:].tobytes() and actions.dtype==initial.dtype
            ledger.append(dict(decision_seq=seq,stage_calls=list(hit['stage_calls']),actual_nfe=hit['actual_nfe'],cursor=hit['cursor']))
        core.acknowledge_execution(dict(decision_seq=4,executed_count=1,terminal=True))
        core.on_episode_end(success=False);core.on_task_end()
        report.update(subset_status='PASS',ledger=ledger,zero_stage_tail_calls=4)
    except BaseException as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}');raise
    finally: write_json(run/'TAIL_RECEIPT.json',report)
    return 2


if __name__=='__main__': raise SystemExit(main())
