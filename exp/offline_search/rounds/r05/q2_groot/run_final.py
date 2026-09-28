"""Final installed Q2 tests, exact commands persisted before each run."""
import json,subprocess,sys
from pathlib import Path
B=Path(__file__).resolve().parent
P=['taskset','-c','30-33,74-77','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=.:src',sys.executable]
commands=[]
def run(name,args):
    cmd=P+args;entry=dict(name=name,command=cmd);commands.append(entry)
    (B/'results/final_commands.json').write_text(json.dumps(commands,indent=2))
    with (B/'results'/f'final_{name}.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    entry['returncode']=r.returncode
    (B/'results/final_commands.json').write_text(json.dumps(commands,indent=2))
    assert r.returncode==0,name
    print('PASS',name,flush=True)
run('tail_parity',[str(B/'tail_parity.py'),'installed'])
run('k10_method',[str(B/'regression/method_test.py')])
run('contracts',[str(B/'contract_tests.py')])
for suite in ('spatial','l10'):
    for scale in (50,500):
        for blocks in (1,2):
            name=f'{suite}_{scale}_G{5*(blocks+1)}'
            run(name,[str(B/'new_tests.py'),'--source','installed','--suite',suite,'--scale',str(scale),'--blocks',str(blocks),'--out',f'/tmp/q2_final_edges/{name}'])
            kwargs=json.dumps(dict(lib='current' if scale==50 else 'big',kref=5 if scale==50 else 8,cycle_k=4,tail_blocks=blocks))
            fit_args=['--fit-artifact',f'/tmp/q2_fits/r5q2_g_{suite}_{scale}_G10.pkl'] if blocks==1 else []
            run('selftest_'+name,[str(B/'regression/launch_test.py'),'installed','normal','--blind','--policy-tail','--policy-tail-blocks',str(blocks),'--cell',f'groot_{suite}_cache','--yaml',f"exp/trace_dual/config/tr_groot_{'sp' if suite=='spatial' else 'l10'}_cache.yaml",'--root','/home/weiland/trace_runs/offline_search_store','--method','exp.offline_search.rounds.r05.q2_groot.judge:CycleTail','--kwargs',kwargs,'--judge','guard_only','--out',f'/tmp/q2_final_selftests/{name}',*fit_args])
run('concurrency',[str(B/'concurrency_test.py'),'--source','installed','--config','all','--out','/tmp/q2_final_concurrency'])
run('arms_validation',[str(B/'validate_arms.py')])
run('coordinator_prepare',[str(B/'prepare_coordinator.py'),'--run-root','/tmp/q2_coordinator_prepare'])
