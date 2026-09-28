"""Invalid switches, finite horizons, method identity and gripper semantics."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from exp.offline_search.closed_loop import plugin
from exp.offline_search.closed_loop.blind import policy_tail_chunk
from exp.offline_search.harness import api
from exp.offline_search.rounds.r05.q2_groot.judge import CycleTail
B=Path(__file__).resolve().parent;checks=[]
base=['--os-method','exp.offline_search.rounds.r05.q2_groot.judge:CycleTail','--os-cell','pi05_spatial_cache','--os-log-dir','/tmp/q2_contracts']
for flags in (['--os-policy-tail'],['--os-policy-tail-blocks','1'],['--os-policy-tail-blocks','3'],['--os-policy-tail-blocks','0']):
    try:plugin.parse_cli(base+flags)
    except SystemExit:checks.append('invalid flags '+' '.join(flags))
    else:raise AssertionError(flags)
opts,_=plugin.parse_cli(base+['--os-blind','--os-policy-tail','--os-policy-tail-blocks','2'])
try:plugin.PluginRuntime(opts,'pi05')
except ValueError as e:assert 'horizon' in str(e);checks.append('H10 refuses two complete tails')
else:raise AssertionError('pi05 two tails accepted')
for kwargs in (dict(cycle_k=0),dict(cycle_k=True),dict(cycle_k=1.5),dict(tail_blocks=0),dict(tail_blocks=3),dict(tail_blocks=True),dict(gates='all'),dict(serving='phase_particles'),dict(budget=2)):
    try:CycleTail(**kwargs)
    except ValueError:checks.append('invalid method '+str(kwargs))
    else:raise AssertionError(kwargs)
try:CycleTail().fit(SimpleNamespace(model='pi05'),None)
except api.SkipCell:checks.append('CycleTail refuses pi05 before fit')
else:raise AssertionError('pi05 accepted')
for offset,shape in ((0,(16,32)),(15,(16,32)),(10,(10,32)),(5,(9,32)),(5,(16,))):
    try:policy_tail_chunk(np.zeros(shape,np.float32),offset)
    except ValueError:checks.append(f'invalid tail offset={offset} shape={shape}')
    else:raise AssertionError((offset,shape))
for dtype in (np.float32,np.float64):
    a=np.arange(16*32,dtype=dtype).reshape(16,32)
    for off in (5,10):
        r=policy_tail_chunk(a,off)
        assert r.dtype==dtype and r.shape==a.shape and r[:16-off].tobytes()==a[off:].tobytes()
        assert np.array_equal(r[16-off:],np.repeat(a[-1:],off,axis=0))
checks.append('offset5/10 preserve all 32 columns and float32/float64 wire dtype')
a[0,0]=np.nan
try:policy_tail_chunk(a)
except ValueError:checks.append('nonfinite source rejected')
else:raise AssertionError('nonfinite accepted')
# Anchor clock is real-vision based: tested in each integrated suite/scale; no
# generic pi05 terminal/stuck predicate exists in CycleTail's inheritance.
assert all('MixedJudge' not in c.__name__ for c in CycleTail.__mro__)
checks.append('no MixedJudge or pi05 terminal-closed guard inherited')
report=dict(PASS=True,checks=checks,count=len(checks))
(B/'results/contracts.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
