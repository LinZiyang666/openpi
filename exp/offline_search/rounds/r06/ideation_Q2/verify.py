"""Read-only invariants for the audit and proposed budget arithmetic."""
from pathlib import Path
import hashlib,json,math
import numpy as np

O=Path(__file__).resolve().parent
R=json.loads((O/'arms.json').read_text());plans=json.loads((O/'test_plan.json').read_text())
issues=[];hashes=0
for r in R:
    for f,h in r['hashes'].items():
        assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
        hashes+=1
    assert len(r['episodes'])==r['n']
    assert abs(np.mean([e['Y'] for e in r['episodes']])-r['sr'])<1e-12
    assert len({(e['task'],e['init']) for e in r['episodes']})==r['n']
    if r['cost_valid']:
        assert sum(e['N'] for e in r['episodes'])==r['N']
        assert sum(e['M'] for e in r['episodes'])==r['M']
        assert sum(e['V'] for e in r['episodes'])==r['V']
        if r['run']!='DUAL':
            s=json.loads(Path(r['summary']).read_text())
            if s.get('mixed') and s['mixed']['misses']!=r['M']:issues.append([r['arm'],'summary mixed MISS mismatch',s['mixed']['misses'],r['M']])
assert not issues,issues
assert len(plans)==8
for p in plans:
    assert len(p['p_call_per_anchor'])==10
    q=[x for x in json.loads((O/'library_quality.json').read_text()) if x['cell']==p['model']+'_'+p['suite'] and x['lib']==p['library']]
    q.sort(key=lambda x:x['task']);c1=.152 if p['model']=='pi05' else .148
    predicted=sum((c1+(1-c1)*rate)*x['mean_anchors_b2'] for x,rate in zip(q,p['p_call_per_anchor']))/sum(x['mean_decisions'] for x in q)
    assert abs(predicted-p['user_knob']['target_ir'])<1e-12
    for rate in p['p_call_per_anchor']:
        assert 0<=rate<=1
        for u in [0,.25,.5,.99999999]:
            for k in [1,2,3,10,100,1000]:
                calls=math.floor(u+k*rate)-math.floor(u)
                assert abs(calls-k*rate)<1+1e-12
result=dict(status='PASS',arms=len(R),episodes=sum(r['n'] for r in R),source_hashes_checked=hashes,
            accepted_decision_count=sum(r['N'] for r in R),invalid_cost_arms=[r['arm'] for r in R if not r['cost_valid']],
            valid_cost_arms=sum(r['cost_valid'] for r in R),test_arms=len(plans),
            checks=['frozen summary/journal hashes','unique accepted task/init pairs','journal success agreement',
                    'raw per-episode N/V/M sums','legacy summary MISS agreement for valid arms','eight configurations','fractional call quota prefix bound'])
(O/'verification.json').write_text(json.dumps(result,indent=1));print(json.dumps(result,indent=1))
