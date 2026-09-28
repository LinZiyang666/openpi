"""Import/hash audit and installed-file smoke; no GPU/model/server."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
B=Path(__file__).resolve().parent
from exp.offline_search.closed_loop import plugin
assert Path(plugin.__file__).resolve()==B.parents[2]/'closed_loop/plugin.py'
install=json.loads((B/'results/install.json').read_text())
assert hashlib.sha256(Path(plugin.__file__).read_bytes()).hexdigest()==install['sha256']
commands=[]
def run(name,cmd):
    with (B/'results'/f'postinstall_{name}.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,
                 env={**os.environ,'Q5_SOURCE':'installed','Q5_RUN':'installed'})
    commands.append(dict(name=name,command=cmd,returncode=r.returncode));assert r.returncode==0,name
run('stack',['taskset','-c','34-37,78-81','bash',str(B/'run_stack.sh')])
P=['taskset','-c','34-37,78-81','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','CUDA_VISIBLE_DEVICES=',
   'PYTHONDONTWRITEBYTECODE=1',f'PYTHONPATH={B}/boot:.:src',sys.executable]
run('byte_parity',P+[str(B/'final_regression/check_parity.py'),'installed'])
run('tail_byte_parity',P+[str(B/'q2checks/tail_parity.py'),'installed'])
run('k5_installed_audit',P+[str(B/'final_regression/k5_final_audit.py')])
run('arms',P+[str(B/'make_arms.py')])
run('emitter',P+['exp/offline_search/closed_loop/ops/emit_arms.py','--run-root','/tmp/q5_final_emitted','--spec',str(B/'arms_q5.json')])
run('aggregate',P+[str(B/'summarize_shadow.py'),'--log-dir','/tmp/q5_stack_installed_50_AWM_shadow','--out',str(B/'results/shadow_aggregate_smoke.json')])
report=dict(PASS=True,plugin_path=plugin.__file__,plugin_sha256=install['sha256'],commands=commands,
            stack=[json.loads(p.read_text()) for p in sorted(Path('/tmp').glob('q5_stack_installed_*/report.json'))])
assert len(report['stack'])==8
(B/'results/postinstall.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
