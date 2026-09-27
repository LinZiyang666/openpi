"""Launch-script command generation only. Stub tmux/ss; never starts a server."""
import json,os,pathlib,subprocess
HERE=pathlib.Path(__file__).resolve().parent
OUT=HERE/'results/launcher';OUT.mkdir(parents=True,exist_ok=True)
BIN=OUT/'bin';BIN.mkdir(exist_ok=True)
for name,body in {'tmux':'#!/bin/bash\n[ "$1" = has-session ] && exit 1\nexit 0\n','ss':'#!/bin/bash\nexit 0\n'}.items():
    p=BIN/name;p.write_text(body);p.chmod(0o755)
env={**os.environ,'PATH':str(BIN.resolve())+':'+os.environ['PATH'],'CPUS':'26-29,70-73','OMP':'1'}
env.pop('GROOT_DENOISING_STEPS',None)
def run(file,model,tag,K=None):
    e=dict(env)
    if K is not None:e['GROOT_DENOISING_STEPS']=str(K)
    out=OUT/tag;out.mkdir(exist_ok=True)
    proc=subprocess.run(['bash',str(file),model,'libero_spatial','23189',str(HERE/'dev'/f'{model}.yaml'),str(out),'same',
                         '--os-method','native','--os-cell',model+'_spatial_cache'],env=e,capture_output=True,text=True)
    if proc.returncode:return proc.returncode,None
    s=(out/'launch_same.sh').read_text()
    # Paths differ intentionally; generated timestamp is metadata, not command semantics.
    return proc.returncode,'\n'.join(l.replace(str(out),'<OUT>') for l in s.splitlines() if not l.startswith('# generated'))
checks=[]
for model in ('pi05','groot'):
    a,old=run(HERE/'dev/before_start_server.sh',model,'before_'+model)
    b,new=run(HERE.parents[2]/'closed_loop/ops/start_server.sh',model,'after_'+model)
    assert a==b==0 and old==new,(model,old,new)
    checks.append(model+' unset options generated command byte-identical')
rc,s=run(HERE.parents[2]/'closed_loop/ops/start_server.sh','groot','k2',2)
assert rc==0 and '--denoising-steps 2' in s;checks.append('GR00T K2 propagated')
for K in (0,-1,'oops','2.0'):
    rc,_=run(HERE.parents[2]/'closed_loop/ops/start_server.sh','groot','invalid_'+str(K),K)
    assert rc!=0
checks.append('four invalid GR00T K values refused')
(OUT/'checks.json').write_text(json.dumps({'passed':len(checks),'checks':checks},indent=2))
print(json.dumps(checks,indent=2))
