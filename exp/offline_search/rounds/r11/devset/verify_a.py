"""R10 plan/selection and mocked driver-argv comparison; no remote operations."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess

from exp.offline_search.closed_loop.ops.h100 import assets, control

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / 'evidence'
ROOT = Path('/home/weiland/trace_runs/os_closed_loop/r10_recipe_current')


def driver_argv(script, name):
    work = EVIDENCE / 'a_argv' / name
    work.mkdir(parents=True, exist_ok=True)
    fake = work / 'python'
    fake.write_text('#!/bin/bash\nif [ "$1" = - ]; then echo "5 default"; elif [[ "$1" = *count.py ]]; then :; else printf "%s\\n" "$@"; fi\n')
    fake.chmod(0o755)
    script = script.replace('/scratch/zixuans8/openpi_trace', str(work.resolve())).replace('/scratch/zixuans8/openpi/.venv/bin/python', str(fake.resolve()))
    path = work / 'run_arm.sh'
    path.write_text(script)
    env = {**os.environ, 'WORKER_HOST': 'timan108', 'OSCL_MANIFEST': str(ROOT / 'eval500.json')}
    for key in ('OSCL_INIT_POOL', 'OSCL_EPISODES', 'OSCL_TASKS', 'OSCL_POOL_CONTRACT', 'OSCL_REPLAN_STEPS'):
        env.pop(key, None)
    result = subprocess.run(['bash', str(path), 'libero_10', 'r10_recipe_pi05_l10_current', 'host:23220', '2', str(work / 'out')],
                            env=env, text=True, capture_output=True, check=True)
    lines = result.stdout.splitlines()
    start = next(i for i, v in enumerate(lines) if v.endswith('/os_cl/run_gtp_subset.py'))
    end = next(i for i, v in enumerate(lines) if v.startswith('RUN_ARM_EXIT='))
    return [v.replace(str(work.resolve()), '<ISLAND>').replace(str(work), '<ISLAND>') for v in lines[start:end]]


def main():
    work = Path('exp/offline_search/rounds/r11/devset/evidence/a_baseline_plan')
    frozen = EVIDENCE / 'a_baseline_plan_before.json'
    before = frozen.read_bytes() if frozen.exists() else (work / 'plan.json').read_bytes()
    if not frozen.exists():
        frozen.write_bytes(before)
    rows = json.loads((ROOT / 'arms.json').read_text())
    with (EVIDENCE / 'a_after.log').open('w') as f, contextlib.redirect_stdout(f):
        plan = assets.build_plan(ROOT, [r['arm'] for r in rows], work)
    after = (work / 'plan.json').read_bytes()
    selections = {r['arm']: dict(selection=control.selection(ROOT, r), server=control.server_spec(ROOT, r, 23220)) for r in rows}
    selections = json.loads(json.dumps(selections))
    old_selections = json.loads((EVIDENCE / 'a_baseline_selection.json').read_text())
    args_before = driver_argv((EVIDENCE / 'run_arm_before.txt').read_text(), 'before')
    args_after = driver_argv((control.HERE / 'run_arm.sh').read_text(), 'after')
    result = dict(root=str(ROOT), plan_byte_identical=before == after, plan_sha256=hashlib.sha256(after).hexdigest(),
                  files=len(plan['files']), bytes=plan['bytes'], selection_and_server_specs_equal=selections == old_selections,
                  driver_argv_equal=args_before == args_after, driver_argv=args_after,
                  manifest_sha256=assets.sha(ROOT / 'eval500.json'), pool='A', remote_calls=0)
    (HERE / 'a_unchanged.json').write_text(json.dumps(result, indent=1) + '\n')
    print('A_UNCHANGED', json.dumps({k: v for k, v in result.items() if k != 'driver_argv'}))
    assert result['plan_byte_identical'] and result['selection_and_server_specs_equal'] and result['driver_argv_equal']


if __name__ == '__main__':
    main()
