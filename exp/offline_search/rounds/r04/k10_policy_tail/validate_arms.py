import ast
import json
from pathlib import Path
import subprocess
import sys
from exp.offline_search.closed_loop import plugin
from exp.offline_search.rounds.r04.k10_policy_tail.judge import PolicyTailJudge
from openpi.cache.config import load_cache_config
B=Path(__file__).resolve().parent
P=['taskset','-c','26-29,70-73','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=.:src',sys.executable]
rows=json.loads((B/'arms_k10.json').read_text());out=Path('/tmp/k10_emitted')
resolved=json.loads(json.dumps(rows).replace('<RUN>',str(out)))
path=B/'results/arms_resolved.json';path.write_text(json.dumps(resolved,indent=2))
cmd=P+['-m','exp.offline_search.closed_loop.ops.emit_arms','--run-root',str(out),'--spec',str(path)]
subprocess.run(cmd,check=True)
emitted=json.loads((out/'arms.json').read_text())
for row,arm in zip(resolved,emitted):
    config = load_cache_config(arm['yaml'])
    from openpi.cache.types import PI05_V1
    assert config.miss is None and PI05_V1.num_steps == 10
    opts,rest=plugin.parse_cli(['--os-method',row['method'],'--os-kwargs',json.dumps(row['kwargs']),'--os-cell',arm['cell'],'--os-log-dir',str(out/'logs')]+row['plugin_args'])
    assert not rest and opts.os_policy_tail and opts.os_blind and opts.os_judge=='guard_only'
    assert arm['full_model'] and arm['cost_ledger'] and 'yaml_patch' not in row
    assert PolicyTailJudge(**row['kwargs']).base.budget==1
    assert arm.get('client_overrides',{}).get('replan_steps',5)==5
for f in ('blind.py','plugin.py','verify_logs.py','selftest.py','replay_client.py'):
    data=(B.parents[4]/'exp/offline_search/closed_loop'/f).read_bytes()
    assert data==(B/'dev'/f).read_bytes()
    ast.parse(data)
# Flag validation and two unsupported method configurations fail before fitting.
try:plugin.parse_cli(['--os-method',rows[0]['method'],'--os-cell','pi05_l10_cache','--os-log-dir','/tmp/k10_invalid','--os-policy-tail'])
except SystemExit:pass
else:raise AssertionError('policy tail without blind accepted')
for base in ({'serving':'phase_particles'},{'serving':'anchor_tail','budget':2}):
    try:PolicyTailJudge(base_kwargs=base)
    except ValueError:pass
    else:raise AssertionError('unsupported base accepted')
result=dict(PASS=True,arms=len(emitted),prefit_commands=3,command=cmd,exact_kwargs=True,K10_default_miss=True)
(B/'results/arms_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
