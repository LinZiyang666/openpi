"""Emit the third/fourth batch candidates, with no assumed closed-loop winner."""
import json,pathlib,shlex
HERE=pathlib.Path(__file__).resolve().parent
P='exp.offline_search.rounds.r04.k1_blind.'
rows=[];plan=[]


def add(name,model,suite,method,kwargs,*,batch,judge=None,extra=(),patch=None,replan=None,note=''):
    row=dict(name=name,model=model,suite=suite,mode='plugin',method=P+method,kwargs=kwargs)
    args=['--os-no-shadow-native']
    if method.startswith(('blind_awm','judge','wrist')):
        args += ['--os-blind']
    if judge:
        args += ['--os-judge',judge];row['full_model']=True
    args += list(extra)
    args += ['--os-fit-artifact',f'<RUN>/fits/{name}.pkl']
    row['plugin_args']=args
    if patch:row['yaml_patch']=patch
    if replan:row['replan_steps']=replan
    rows.append(row);plan.append(dict(name=name,batch=batch,note=note))


variants=[('b0','phase_particles',0,'all'),('ph1','phase_particles',1,'all'),
          ('ph2','phase_particles',2,'all'),('clk1','kernel_clock',1,'all'),
          ('tail1g','anchor_tail',1,'all'),('tail1u','anchor_tail',1,'budget_only')]
# Primary l10 at both scales, then spatial. Cache controls isolate the guard change.
for suite,short in [('l10','l10'),('spatial','sp')]:
    for scale in (50,500):
        base=dict(lib='current' if scale==50 else 'big',kref=5 if scale==50 else 8)
        for tag,serving,budget,gates in variants:
            kw=dict(base,serving=serving,budget=budget,gates=gates)
            add(f'r4b3_p_{short}_{scale}_{tag}g','pi05',suite,'judge:BlindMixedJudge',
                dict(base_kwargs=kw,progress_guard='noprog_span',events='none'),batch=3,judge='guard_only',
                note='Gap-aware guard; B0 isolates adapter with the same new guard.')
            add(f'r4b3_p_{short}_{scale}_{tag}c','pi05',suite,'blind_awm:BlindAWM',kw,batch=3,
                note='Pure-cache serving control.')
        period=5 if scale==50 else 8
        for budget in (0,2):
            add(f'r4b3_p_{short}_{scale}_ph{budget}k{period}','pi05',suite,'blind_awm:BlindAWM',
                dict(base,serving='phase_particles',budget=budget),batch=3,judge=f'periodic:{period}',
                note='Periodic due is evaluated by plugin on the dense decision counter.')
        add(f'r4b3_p_{short}_{scale}_inferL10','pi05',suite,'blind_awm:BlindAWM',
            dict(base,budget=0),batch=3,judge='periodic:1',replan=10,
            note='Policy execute-L10 baseline; cost per five controls. Same baseline duplicated across library labels.')
for suite,short in [('l10','l10'),('spatial','sp')]:
    for scale in (50,500):
        base=dict(lib='current' if scale==50 else 'big',kref=5 if scale==50 else 8)
        groot=[x for x in variants if x[0]!='clk1']+[
            ('tail2g','anchor_tail',2,'all'),('tail2u','anchor_tail',2,'budget_only')]
        for tag,serving,budget,gates in groot:
            add(f'r4b3_g_{short}_{scale}_{tag}','groot',suite,'blind_awm:BlindAWM',
                dict(base,serving=serving,budget=budget,gates=gates),batch=3,note='GR00T pure-cache only.')
# Fourth batch is a candidate grid; coordinator picks survivors from measured batch three.
for suite,short in [('l10','l10'),('spatial','sp')]:
    for scale in (50,500):
        base=dict(lib='current' if scale==50 else 'big',kref=5 if scale==50 else 8)
        for budget in (1,2):
            for stage in ('full','dummy_cached','wrist_only'):
                tag={'full':'k2','dummy_cached':'k2dc','wrist_only':'k2wp'}[stage]
                name=f'r4b4_p_{short}_{scale}_ph{budget}_{tag}'
                kw=dict(base_kwargs=dict(base,serving='phase_particles',budget=budget),events='none')
                method='wrist:BlindWristMixedJudge' if stage=='wrist_only' else 'judge:BlindMixedJudge'
                extra=[] if stage=='full' else ['--os-stage1-mode',stage,'--os-pack-prefix']
                if stage=='wrist_only':extra+=['--os-tokens','off']
                add(name,'pi05',suite,method,kw,batch=4,judge='guard_only',extra=extra,
                    patch={'miss':{'num_steps':2,'evidence_dir':f'<RUN>/evidence/{name}'},'write_policy':{'type':'never'}},
                    note='Conditional combination candidate, not a claimed winner; stage optimizations require K3 validation.')
        if suite=='l10':
            for ablation in ('G','GS'):
                add(f'r4b4_p_l10_{scale}_csl{ablation}','pi05',suite,'control_step:ControlStepLibrary',
                    dict(base,ablation=ablation),batch=4,note='G fixes actions; GS changes aligned synthesis. Pure-cache only.')
assert len({x['name'] for x in rows})==len(rows)
assert max(map(lambda x:len(x['name']),rows))<=40
(HERE/'arms_r4.json').write_text(json.dumps(rows,indent=2)+'\n')
(HERE/'arm_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
commands=['#!/usr/bin/env bash','set -euo pipefail',': "${RUN:?Set RUN to the coordinator run directory}"',
          'ROOT=${ROOT:-/home/weiland/trace_runs/offline_search_store}']
for row in rows:
    name=row['name']
    # Shell expansions are intentional only in double-quoted generated RUN/ROOT arguments.
    commands.append(' '.join(['taskset -c 18-21,62-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
        "CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/weiland/projects/openpi/.venv/bin/python",
        '-m exp.offline_search.closed_loop.plugin','--os-method',shlex.quote(row['method']),
        '--os-kwargs',shlex.quote(json.dumps(row['kwargs'],separators=(',',':'))),
        '--os-cell',row['model']+'_'+row['suite']+'_cache','--os-root "$ROOT"',
        '--os-log-dir "$RUN/fits"','--os-fit-artifact',f'"$RUN/fits/{name}.pkl"']))
(HERE/'prefit.sh').write_text('\n'.join(commands)+'\n')
print(json.dumps(dict(arms=len(rows),batch3=sum(x['batch']==3 for x in plan),batch4=sum(x['batch']==4 for x in plan))))
