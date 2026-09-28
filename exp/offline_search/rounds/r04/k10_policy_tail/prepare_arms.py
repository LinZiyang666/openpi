"""Exact K7 tail1ug dictionaries, new method and opt-in switch. No prefits run."""
import json
from pathlib import Path
import shlex
BASE=Path(__file__).resolve().parent
SPEC='exp.offline_search.rounds.r04.k10_policy_tail.judge:PolicyTailJudge'
PREFIX=['taskset','-c','26-29,70-73','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=.:src','.venv/bin/python']
configs=json.loads((BASE.parent/'k8_search_latency/configs.json').read_text())
rows=[];commands=[]
for suite,scale in [('l10',500),('l10',50),('spatial',500)]:
    short='sp' if suite=='spatial' else suite
    old=next(c for c in configs if c.get('id')==f'K7_r4k7_p_{short}_{scale}_tail1ug')
    kw=old['kwargs']
    assert kw==dict(base_kwargs=dict(lib='big' if scale==500 else 'current',kref=8 if scale==500 else 5,serving='anchor_tail',budget=1,gates='budget_only'),progress_guard='noprog_span',events='none',stuck_guard='vision_confirmed')
    name=f'r4k10_p_{short}_{scale}_tail1ug'
    flags=['--os-root','/home/weiland/trace_runs/offline_search_store','--os-blind','--os-policy-tail','--os-judge','guard_only','--os-no-shadow-native','--os-fit-artifact',f'<RUN>/fits/{name}.pkl']
    row=dict(name=name,model='pi05',suite=suite,mode='plugin',method=SPEC,kwargs=kw,full_model=True,plugin_args=flags,cost_ledger=True)
    rows.append(row)
    commands.append(PREFIX+['-m','exp.offline_search.closed_loop.plugin','--os-method',SPEC,'--os-kwargs',json.dumps(kw,separators=(',',':')),'--os-cell',f'pi05_{suite}_cache','--os-log-dir',f'<RUN>/prefit_logs/{name}','--os-tag',name]+flags)
(BASE/'arms_k10.json').write_text(json.dumps(rows,indent=2)+'\n')
(BASE/'prefit_commands.json').write_text(json.dumps(commands,indent=2)+'\n')
script='#!/usr/bin/env bash\nset -euo pipefail\ncd /home/weiland/projects/openpi\n: "${RUN:?Set RUN to the coordinator run root}"\n'
for command in commands:
    script+=' '.join(('"'+x.replace('<RUN>','${RUN}')+'"') if '<RUN>' in x else shlex.quote(x) for x in command)+'\n'
(BASE/'prefit.sh').write_text(script)
