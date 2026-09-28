"""Small installed-file and emitted-arm audits after integration tests."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from exp.offline_search.closed_loop.verify_logs import check_verdicts
from exp.offline_search.rounds.r04.k5_rand import estimate as E
BASE=Path(__file__).resolve().parent
REPO=BASE.parents[4]
shared=REPO/'exp/offline_search/closed_loop'
for name in ('plugin.py','selftest.py','verify_logs.py'):
    assert (shared/name).read_bytes()==(BASE/'dev'/name).read_bytes()
for p in list(BASE.glob('*.py'))+[shared/n for n in ('plugin.py','selftest.py','verify_logs.py')]: ast.parse(p.read_text())
arms=json.loads(Path('/tmp/k5_arms_check/arms.json').read_text())
assert len(arms)==4
for a in arms:
    assert a['full_model'] and a['cost_ledger'] and a['cell']=='pi05_l10_cache'
    assert '--os-log-r4' in a['plugin_args'] and a['judge']=='guard_only'
reports=[]
for a in arms:
    name=a['arm']; directory=Path('/tmp/k5_installed_replays')/name
    rows=[json.loads(l) for l in (directory/'decisions_selftest.jsonl').read_text().splitlines()]
    dec=[r for r in rows if r['ev']=='dec']; episodes=[r for r in rows if r['ev']=='episode']
    assert len(dec)==342 and len(episodes)==4
    assert sum(r['eligible'] for r in dec)==4
    assert all(r['vision'] and len(r['served_head'])==5 and (r['miss_k'] is None if r['hit'] else r['miss_k']==10) for r in dec)
    z=np.load(next((directory/'inputs').glob('*.npz')))
    meta=json.loads(str(z['meta'])); arrays={k:z[k] for k in z.files if k!='meta'}
    assert not check_verdicts(meta['judge'],arrays)['bad']
    forged=arrays.copy(); forged['randomization']=np.array(arrays['randomization'],dtype=object)
    r=json.loads(str(forged['randomization'][0]));r['opportunity_index']+=1
    forged['randomization'][0]=json.dumps(r)
    assert check_verdicts(meta['judge'],forged)['bad']
    reports.append(dict(arm=name,decisions=len(dec),episodes=len(episodes),eligible=sum(r['eligible'] for r in dec),
        call=sum(r['eligible'] and r['assigned_treatment']=='CALL' for r in dec),cache=sum(r['eligible'] and r['assigned_treatment']=='CACHE' for r in dec),
        misses=sum(not r['hit'] for r in dec),forged_log_rejected=True))
# Refuse accidental cross-scale estimation, even though treatment bits complement.
r1,_=E.load_arm('/tmp/k5_replay_estimator','r4k5_p_l10_g50_r1')
r2,_=E.load_arm('/tmp/k5_replay_estimator','r4k5_p_l10_g500_r2')
try: E.estimate(r1+r2,100)
except ValueError as e: assert 'different controller' in str(e)
else: raise AssertionError('cross-scale pair accepted')
result=dict(PASS=True,installed_matches_dev=True,emitted_arms=4,replay_audits=reports,cross_scale_rejected=True)
(BASE/'results/final_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
