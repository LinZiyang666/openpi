"""Six R5-c arms and exact CPU prefit commands; no launches."""
import json
from pathlib import Path
import shlex
B=Path(__file__).resolve().parent
ROOT='/home/weiland/trace_runs/offline_search_store'
METHOD='exp.offline_search.rounds.r05.q2_groot.judge:CycleTail'
P=['taskset','-c','30-33,74-77','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=.:src','/home/weiland/projects/openpi/.venv/bin/python']
arms=[];commands=[]
for suite in ('spatial','l10'):
    for scale in (50,500):
        name=f'r5q2_g_{suite}_{scale}_G10'
        kwargs=dict(lib='current' if scale==50 else 'big',kref=5 if scale==50 else 8,cycle_k=4,tail_blocks=1)
        flags=['--os-root',ROOT,'--os-blind','--os-policy-tail','--os-policy-tail-blocks','1','--os-judge','guard_only','--os-no-shadow-native','--os-fit-artifact',f'<RUN>/fits/{name}.pkl']
        arms.append(dict(name=name,model='groot',suite=suite,mode='plugin',method=METHOD,kwargs=kwargs,full_model=True,cost_ledger=True,client_overrides=dict(replan_steps=5,resize_size=256),plugin_args=flags))
        cmd=P+['-m','exp.offline_search.closed_loop.plugin','--os-method',METHOD,'--os-kwargs',json.dumps(kwargs,separators=(',',':')),'--os-cell',f'groot_{suite}_cache','--os-log-dir',f'<RUN>/prefit_logs/{name}','--os-tag',name,*flags]
        commands.append(dict(arm=name,command=cmd,shell=shlex.join(cmd)))
for suite in ('spatial','l10'):
    arms.append(dict(name=f'r5q2_g_{suite}_policy_L10',model='groot',suite=suite,mode='plugin',pure_inference=True,full_model=True,server_seed=5101,cost_ledger=True,client_overrides=dict(replan_steps=10,resize_size=256),plugin_args=['--os-root',ROOT,'--os-log-r4','--os-no-shadow-native']))
(B/'arms_q2.json').write_text(json.dumps(arms,indent=2)+'\n')
(B/'prefit_commands.json').write_text(json.dumps(commands,indent=2)+'\n')
# The exact same method/kwargs/cell prefits go to task-owned temporary storage.
lines=['#!/usr/bin/env bash','set -euo pipefail','cd /home/weiland/projects/openpi']
for row in commands:
    cmd=[x.replace('<RUN>/fits','/tmp/q2_fits').replace('<RUN>/prefit_logs',str(B/'results/prefit')) for x in row['command']]
    lines.append(shlex.join(cmd))
(B/'prefit.sh').write_text('\n'.join(lines)+'\n')
