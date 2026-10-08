"""Actual plugin CPU replay, with A query reads and every launch blocked."""
from __future__ import annotations
import contextlib, copy, json, os, pickle, socket, subprocess
from pathlib import Path
from unittest.mock import patch
import numpy as np
from exp.offline_search.closed_loop import plugin,devset
from .prepare import HERE,TRACE,SEED,frozen,write_json
from .mechanics import Harness,load


def blocked(*a,**kw): raise AssertionError('process/network/remote operation forbidden')


@contextlib.contextmanager
def cpu_fence():
    original=Path.exists
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(plugin,'_git_head',lambda:None))
        stack.enter_context(patch.object(Path,'exists',lambda p:False if 'queries' in p.parts else original(p)))
        stack.enter_context(patch.object(subprocess,'Popen',blocked))
        stack.enter_context(patch.object(socket,'create_connection',blocked))
        yield


def lifecycle(rows):
    return [(r['task_id'],r['step'],r['hit'],r['vision'],r['src'],r['served_head']) for r in rows]


def c10_proof(output):
    records={};refs={}
    with cpu_fence():
        for model in ('pi05','groot'):
            for suite in ('l10','spatial'):
                ms=f'{model}_{suite}';h=Harness(ms+'_100',Path(output)/ms)
                try:
                    a=h.run(load(ms+'_100','c10'),steps=16,force_guards=True)
                    b=h.run(load(ms+'_500','c10'),steps=16,force_guards=True,reverse=True)
                    assert a[0]==b[0] and sorted(lifecycle(a[2]))==sorted(lifecycle(b[2]))
                    assert all(not d['vision'] or not d['hit'] for d in a[2]+b[2])
                    records[ms]=dict(PASS=True,libraries=[100,500],tasks=10,steps=16,
                        decisions_per_library=len(a[2]),served_action_bytes_equal=True,lifecycle_equal=True,
                        cache_proposals_and_guard_diagnostics_may_differ=True,
                        fit_sha256={str(n):frozen(ms+f'_{n}','c10')['sha256'] for n in (100,500)})
                finally:h.rt.flush_all()
                root=TRACE/f'r14_descent_test_{ms}_50'
                rows=json.loads((root/'arms.json').read_text())
                row=next(r for r in rows if r['r14_descent_level']=='c10')
                args=row['plugin_args'];fit=Path(args[args.index('--os-fit-artifact')+1])
                method=pickle.loads(fit.read_bytes())['method']
                assert method.knob_method=='periodic' and method.knob_settings['setting']==0
                assert row['rollout_replicate_seed']==SEED and args[args.index('--os-rollout-seed')+1]==str(SEED)
                assert '--os-policy-tail' in args and args[args.index('--os-policy-tail-blocks')+1]=='1'
                assert args[args.index('--os-judge')+1]=='guard_only' and row['episodes']==500
                manifest=Path(row['manifest']);template=TRACE/'r11_knob_1/eval500.json'
                assert manifest.read_bytes()==template.read_bytes()
                assert json.loads(manifest.read_bytes())==[[t,i] for t in range(10) for i in range(50)]
                assert row['intended_host']==('local' if model=='groot' else 'h100')
                refs[ms]=dict(root=str(root),arm=row['arm'],arms_sha256=devset.sha(root/'arms.json'),
                    manifest=str(manifest),template_sha256=devset.sha(template),fit_path=str(fit),
                    fit_sha256=devset.sha(fit),rollout_seed=SEED,init_pool='A',reusable=True,
                    init_identities='official A task 0..9 x init 0..49; original indices',
                    host=row['intended_host'])
    result=dict(PASS=True,library_independent=True,test_c10_reusable=True,records=records,test_references=refs,
        explanation='K=0 forces MISS at every real LOOK, irrespective of retrieval/guard. The deployed lifecycle-only policy tail serves controls 5..9 from that MISS. Finite state/accepted execution lifecycle can force a new LOOK; these checks do not read library rows. Guard and retrieval audit values can vary, never the decision or served action.',
        reference_seed=SEED,outcome_reads=0,servers=0,remote_operations=0)
    write_json(HERE/'c10_proof.json',result)
    print('C10_PROOF_PASS',sum(r['decisions_per_library']*2 for r in records.values()),flush=True)
    return result

if __name__=='__main__':
    import tempfile
    with tempfile.TemporaryDirectory(prefix='r15_c10_') as directory:c10_proof(directory)
