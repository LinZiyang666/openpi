"""Eight full-A-pool demo scaling arms using the unchanged K1 tail adapter."""
import json
import shlex
from pathlib import Path
from exp.offline_search.closed_loop.ops import emit_arms
from exp.offline_search.rounds.r05.q4_growth.common import OUT,SHM
from exp.offline_search.rounds.r05.q4_growth.prepare import PREFIX

RUN=Path('/home/weiland/trace_runs/os_closed_loop/r05_demo_curve')
SPEC='exp/offline_search/rounds/r05/q4_growth/demo_method.py:DemoBlindAWM'
BASE_SPEC='exp/offline_search/rounds/r05/q4_growth/demo_method.py:DemoAWM'

def arm_name(suite,size,variant):
    return f'r5q4d_p_{"sp" if suite=="spatial" else suite}_{size}_{variant}_tail1'

def kwargs(size,variant):
    return dict(library=f'demo{size}',variant=variant,kref=5,serving='anchor_tail',budget=1,gates='budget_only')

def main():
    RUN.mkdir(exist_ok=True);(RUN/'fits').mkdir(exist_ok=True)
    arms=[];commands={s:[] for s in ('l10','spatial')}
    for suite in commands:
        for size,variant in [(100,'refit'),(200,'refit'),(300,'refit'),(200,'frozen50')]:
            name=arm_name(suite,size,variant);kw=kwargs(size,variant)
            args=['--os-no-shadow-native','--os-blind','--os-fit-artifact',f'<RUN>/fits/{name}.pkl']
            arms.append(dict(name=name,model='pi05',suite=suite,mode='plugin',method=SPEC,kwargs=kw,
                             full_model=False,server_env={'STAGE1_ONLY':'1'},replan_steps=5,plugin_args=args))
            cmd=PREFIX+['-m','exp.offline_search.rounds.r05.q4_growth.prefit','--os-root',str(SHM),
                        '--os-method',SPEC,'--os-kwargs',json.dumps(kw,separators=(',',':')),
                        '--os-cell',f'pi05_{suite}_cache','--os-log-dir',str(RUN/'prefit_logs'/name),
                        '--os-tag',name,'--os-no-shadow-native','--os-blind','--os-fit-artifact',str(RUN/'fits'/f'{name}.pkl')]
            commands[suite].append(shlex.join(cmd)+' > '+shlex.quote(str(OUT/'results'/'demo'/f'prefit_{name}.log'))+' 2>&1')
    (OUT/'arms_q4_demo.json').write_text(json.dumps(arms,indent=2)+'\n')
    (RUN/'arms_q4_demo_resolved.json').write_text(json.dumps(arms,indent=2).replace('<RUN>',str(RUN))+'\n')
    for suite,cmds in commands.items():
        (OUT/f'demo_prefit_{suite}.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd /home/weiland/projects/openpi\n'+'\n'.join(cmds)+'\n')
    emit_arms.main(['--run-root',str(RUN),'--spec',str(RUN/'arms_q4_demo_resolved.json')])
    emitted=json.loads((RUN/'arms.json').read_text());assert len(emitted)==8
    for a in emitted:
        assert 'manifest' not in a and a['full_model'] is False and a['judge'] is None
        assert '--os-blind' in a['plugin_args'] and '--os-judge' not in a['plugin_args']
    (OUT/'results'/'demo'/'arms_validation.json').write_text(json.dumps(dict(arms=8,manifest=False,expected_pairs_per_arm=500,
        kref_rule='5 at every size and for both variants; no evaluation tuning',pure_cache=True,stage1_only=True),indent=2))

if __name__=='__main__':main()
