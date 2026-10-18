"""CPU replay of recorded cache scalars/weights and pinned library actions.

Examines every over-budget logical request and all eight recordings. This is
downstream replay: missing stage1 query features cannot be reconstructed here.
"""
import argparse
import ast
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from artifacts import load_tree,sha,write_json

def decode(x,n):
    x=(np.asarray(x,np.float32)-np.asarray(n['offset'],np.float32))/np.asarray(n['scale'],np.float32)
    lo,hi=np.asarray(n['q01'],np.float32),np.asarray(n['q99'],np.float32)
    return (x+1)/2*(hi-lo+1e-6)+lo

def analyse(report,fit,normalization,runtime):
    report=Path(report);r=json.loads(report.read_text());bad=[x for x in r['rows'] if not x['within_budget']]
    keys={tuple(x['key']) for x in bad}
    ready=json.loads((Path(fit)/'READY.json').read_text())
    assert sha(Path(fit)/'fit.npz')==ready['arrays_sha256']
    state=dict(ready['tree']['mapping']) if 'tree' in ready else None
    if state is None:raise ValueError('unknown fitted artifact state')
    with np.load(Path(fit)/'fit.npz',allow_pickle=False) as z:library=z[state['action']['array']]
    path=Path(runtime)/'exp/r17_integration/common/layers/math.py'
    tree=ast.parse(path.read_text());node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_kernel_w')
    scope={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    kernel=scope['_kernel_w'];records={k:[] for k in keys}
    for side in ('left','right'):
      for dirname in r['cold_starts'][side]:
        root=Path(dirname);run=json.loads((root/'RUN.json').read_text())
        for artifact in run['artifacts']:
          d=root/artifact;entries=json.loads((d/'ROWS.json').read_text())
          for row in entries['rows']:
            key=tuple(row['key'])
            if key not in keys:continue
            file=d/row['file'];assert sha(file.with_suffix('.json'))==row['sha256']
            value=load_tree(file);h=value['response']['__hit_meta__'];action=value['response']['actions']
            cursor=h.get('cursor',0)
            if 'donor_weights' not in h:
                previous=next(e for e in entries['rows'] if e['key']==[key[0],key[1],key[2]-1])
                previous_file=d/previous['file']
                assert cursor==5 and sha(previous_file.with_suffix('.json'))==previous['sha256']
                h=load_tree(previous_file)['response']['__hit_meta__']
                assert h['segment_head'] is True
            weights=np.asarray(h['donor_weights'],np.float32);donors=np.asarray(h['donor_rows'],np.int64)
            blend=np.tensordot(weights,library[donors],1)
            prediction=decode(blend,normalization)[cursor:cursor+len(action)]
            distances=h.get('xgpu_donor_distances');computed=None
            if distances is not None:
                distances=np.asarray(distances,np.float64);kw=kernel(distances-distances[0],8);computed=(kw/kw.sum()).astype(np.float32)
            records[key].append(dict(side=side,run=root.name,round=entries['round'],action=action,
                prediction=prediction,weights=weights,distances=distances,donors=donors,
                weight_replay_error=float(np.max(abs(weights-computed))) if computed is not None else None,
                decode_residual=float(np.max(abs(action-prediction))),
                correction_norm=h.get('corrector_delta_norm'),corrector_applied=h.get('corrector_applied'),
                ratio=h.get('corrector_ratio'),cursor=cursor,decision=h.get('decision_index'),
                stage1_size=h.get('batching',{}).get('stages',{}).get('stage1',{}).get('size')))
    rows=[]
    for key,values in sorted(records.items()):
        assert len(values)==8
        pairs=list(itertools.product([x for x in values if x['side']=='left'],[x for x in values if x['side']=='right']))
        differences=[(float(np.max(abs(x['action']-y['action']))),x,y) for x,y in pairs]
        error,x,y=max(differences,key=lambda z:z[0]);pred=x['prediction']-y['prediction'];actual=x['action']-y['action']
        rows.append(dict(key=key,max_cross=error,max_replayed_cross=float(np.max(abs(pred))),
            cross_delta_reconstruction_error=float(np.max(abs(actual-pred))),
            max_decode_residual=max(v['decode_residual'] for v in values),
            max_weight_replay_error=max((v['weight_replay_error'] for v in values if v['weight_replay_error'] is not None),default=None),
            max_weight_cross=max(float(np.max(abs(a['weights']-b['weights']))) for a,b in pairs),
            donors_all_identical=all(np.array_equal(values[0]['donors'],v['donors']) for v in values),
            max_reported_correction=max(v['correction_norm'] or 0 for v in values),
            recordings=[{k:v for k,v in a.items() if k not in ('action','prediction','weights','distances','donors')} for a in values],
            worst_pair=dict(left=x['run'],right=y['run'],left_round=x['round'],right_round=y['round'],
                left_weights=x['weights'].tolist(),right_weights=y['weights'].tolist(),
                left_distances=x['distances'].tolist() if x['distances'] is not None else None,
                right_distances=y['distances'].tolist() if y['distances'] is not None else None)))
    return dict(schema='xgpu.dp.cache_cpu_localisation.v1',scope='all 51 budget-failing requests, eight observed recordings each',
        report_sha256=sha(report),fit_npz_sha256=ready['arrays_sha256'],math_source_sha256=sha(path),
        failed_logical_requests=len(rows),recordings=sum(len(x) for x in records.values()),
        normalization=normalization,rows=rows,
        limits=['query stage1 features absent; cannot attribute their error to a GPU kernel or CPU BLAS',
                'corrector with nonzero delta cannot be independently replayed without its feature vector',
                'CPU replay does not certify unseen requests or restore formal deployment equivalence'])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('report','fit','normalization-json','runtime','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();result=analyse(a.report,a.fit,json.loads(a.normalization_json),a.runtime)
    write_json(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','normalization')}))
