"""Write four pure-cache arm specs, a held-out manifest, and exact fit commands."""
import json
import shlex
from exp.offline_search.rounds.r05.q4_growth.common import OUT,RUN,SPEC,SHM
from exp.offline_search.closed_loop.ops import emit_arms

PREFIX=['taskset','-c','26-29,70-73','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
        'CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=.:src','/home/weiland/projects/openpi/.venv/bin/python']

def prepare():
    pairs=[[t,i] for t in range(10) for i in range(25,50)]
    (OUT/'evaluation_pairs.json').write_text(json.dumps(pairs,indent=1)+'\n')
    (RUN/'evaluation_pairs.json').write_text(json.dumps(pairs,indent=1)+'\n')
    arms=[];commands=[]
    for suite in ('l10','spatial'):
        short='sp' if suite=='spatial' else suite
        for variant in ('refit','frozen'):
            name=f'r5q4_p_{short}_grow250_{variant}'
            kwargs=dict(library='grow250',variant=variant,kref=5)
            arms.append(dict(name=name,model='pi05',suite=suite,mode='plugin',method=SPEC,kwargs=kwargs,
                             full_model=False,server_env={'STAGE1_ONLY':'1'},manifest='<RUN>/evaluation_pairs.json',
                             replan_steps=5,plugin_args=['--os-no-shadow-native','--os-fit-artifact',f'<RUN>/fits/{name}.pkl']))
            cmd=PREFIX+['-m','exp.offline_search.rounds.r05.q4_growth.prefit','--os-root',str(SHM),
                        '--os-method',SPEC,'--os-kwargs',json.dumps(kwargs,separators=(',',':')),
                        '--os-cell',f'pi05_{suite}_cache','--os-log-dir',str(RUN/'prefit_logs'/name),
                        '--os-tag',name,'--os-fit-artifact',str(RUN/'fits'/f'{name}.pkl'),'--os-no-shadow-native']
            commands.append(shlex.join(cmd)+' > '+shlex.quote(str(OUT/'results'/f'prefit_{name}.log'))+' 2>&1')
    (OUT/'arms_q4.json').write_text(json.dumps(arms,indent=2)+'\n')
    (OUT/'prefit_commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd /home/weiland/projects/openpi\n'+'\n'.join(commands)+'\n')
    resolved=json.loads(json.dumps(arms).replace('<RUN>',str(RUN)))
    (RUN/'arms_q4_resolved.json').write_text(json.dumps(resolved,indent=2)+'\n')
    emit_arms.main(['--run-root',str(RUN),'--spec',str(RUN/'arms_q4_resolved.json')])
    emitted=json.loads((RUN/'arms.json').read_text())
    assert len(emitted)==4
    for a in emitted:
        assert a['full_model'] is False and a['judge'] is None and '--os-judge' not in a['plugin_args']
        assert a['manifest']==str(RUN/'evaluation_pairs.json') and a['replan_steps']==5
    print(json.dumps(dict(arms=4,pairs=len(pairs),prefit_commands=str(OUT/'prefit_commands.sh'))))

if __name__=='__main__':prepare()
