"""Run CPU suites in separate interpreters: fixtures import different frozen releases."""
import json
import os
from pathlib import Path
import subprocess
import sys

def main():
    root=Path(__file__).resolve().parent
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',
             PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    results=[]
    with (root/'evidence/pilot_all_cpu_tests_1018.txt').open('w') as log:
        for path in sorted(root.glob('test_*.py')):
            log.write('\nCPU isolated suite: '+path.name+'\n');log.flush()
            result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(root),'-p',path.name,'-v'],
                                  env=env,stdout=log,stderr=subprocess.STDOUT)
            results.append(dict(suite=path.name,rc=result.returncode))
            print(path.name,result.returncode,flush=True)
    (root/'evidence/pilot_cpu_suites_1018.json').write_text(json.dumps(results,indent=2)+'\n')
    return int(any(row['rc'] for row in results))

if __name__=='__main__':raise SystemExit(main())
