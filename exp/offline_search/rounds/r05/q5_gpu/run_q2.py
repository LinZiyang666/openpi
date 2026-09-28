import json,subprocess,sys,os
from pathlib import Path
B=Path(__file__).resolve().parent;Q=B/'q2checks';R=B/'regression'
P=['taskset','-c','34-37,78-81','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1',f'PYTHONPATH={B}/boot:.:src',sys.executable]
commands=[]
def run(name,args):
    cmd=P+args;entry=dict(name=name,command=cmd);commands.append(entry)
    (B/'results/q2_commands.json').write_text(json.dumps(commands,indent=2))
    with (B/'results'/f'q2_{name}.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    entry['returncode']=r.returncode
    (B/'results/q2_commands.json').write_text(json.dumps(commands,indent=2));assert r.returncode==0,name
    print('PASS Q2',name,flush=True)
run('tail_parity',[str(Q/'tail_parity.py'),'installed'])
run('contracts',[str(Q/'contract_tests.py')])
run('k10_method',[str(R/'method_test.py')])
for suite in ('spatial','l10'):
    for scale in (50,500):
        for blocks in (1,2):
            name=f'{suite}_{scale}_G{5*(blocks+1)}'
            run(name,[str(Q/'new_tests.py'),'--source','installed','--suite',suite,'--scale',str(scale),'--blocks',str(blocks),'--out',f'/tmp/q5_q2_edges/{name}'])
            kw=json.dumps(dict(lib='current' if scale==50 else 'big',kref=5 if scale==50 else 8,cycle_k=4,tail_blocks=blocks))
            fit_args=['--fit-artifact',f'/tmp/q2_fits/r5q2_g_{suite}_{scale}_G10.pkl'] if blocks==1 else []
            run('selftest_'+name,[str(R/'launch_test.py'),'installed','normal','--blind','--policy-tail','--policy-tail-blocks',str(blocks),
                '--cell',f'groot_{suite}_cache','--yaml',f"exp/trace_dual/config/tr_groot_{'sp' if suite=='spatial' else 'l10'}_cache.yaml",
                '--root','/home/weiland/trace_runs/offline_search_store','--method','exp.offline_search.rounds.r05.q2_groot.judge:CycleTail',
                '--kwargs',kw,'--judge','guard_only','--out',f'/tmp/q5_q2_selftests/{name}',*fit_args])
run('concurrency',[str(Q/'concurrency_test.py'),'--source','installed','--config','all','--out','/tmp/q5_q2_concurrency'])
