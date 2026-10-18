raise SystemExit("OWNER HOLD: no automatic activation; two live validation lanes must pass")
"""Serial publication and direct-tree checks, gated on complete regression evidence."""
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[4]
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
           JAX_PLATFORMS='cpu', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           MKL_NUM_THREADS='1', AUTOSCALE_TEST_TIMEOUT='180', PYTHONPATH='.:src')
env.pop('AUTOSCALE_SOURCE_VARIANT', None)
env.pop('AUTOSCALE_BASELINE_ROOT', None)
python = str(ROOT / '.venv/bin/python')
records = []


def run(name, command):
    print(name, 'started', flush=True)
    started = time.time()
    with (HERE / (name + '.log')).open('w') as output:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=output,
                                stderr=subprocess.STDOUT)
    records.append(dict(name=name, command=command, code=result.returncode,
                        seconds=time.time() - started))
    (HERE / 'FINISH_COMMANDS.json').write_text(json.dumps(records, indent=2) + '\n')
    print(name, 'exit', result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


while not (HERE / 'OPS_FINAL_COMMAND.json').exists():
    time.sleep(5)
run('comparison', [python, str(HERE / 'report.py')])
assert json.loads((HERE / 'BEFORE_AFTER.json').read_text())['status'] == 'PASS'
run('activation', [python, str(HERE / 'activate.py')])
for name, path, xml in (
    ('postapply', str(BASE / 'tests'), 'POST_APPLY.xml'),
    ('longseg_unit', 'exp/offline_search/rounds/r17/longseg_1017', 'LONGSEG_UNIT.xml'),
):
    run(name, [python, '-m', 'pytest', '-p',
               'exp.offline_search.rounds.r17.autoscale_1017.cpu_only', path,
               '-q', '--tb=short', '--basetemp=' + str(HERE / (name + '_tmp')),
               '--junitxml=' + str(HERE / xml), '-o',
               'cache_dir=' + str(HERE / (name + '_cache'))])
run('verification', [python, str(HERE / 'update_verification.py')])
print('FINISHED', flush=True)
