"""Exercise the optional monitor's actual fit hook and installed plugin at both scales."""
import json
from pathlib import Path
import pickle
import subprocess

from prepare import HERE, PREFIX

out = Path('/tmp/q1_final_monitor')
out.mkdir(exist_ok=False)
commands = json.loads((HERE / 'prefit_commands.json').read_text())['executed']
reports = []
for scale in (50, 500):
    name = f'r5q1_c10_p_l10_{scale}'
    cmd = next(c.copy() for c in commands if c[c.index('--os-tag') + 1] == name)
    kw = json.loads(cmd[cmd.index('--os-kwargs') + 1])
    kw['monitor'] = 'loeo_xyz99'
    cmd[cmd.index('--os-kwargs') + 1] = json.dumps(kw)
    fit = f'/tmp/q1_fits/{name}_monitor_final.pkl'
    cmd[cmd.index('--os-fit-artifact') + 1] = fit
    cmd[cmd.index('--os-tag') + 1] = name + '_monitor'
    cmd[cmd.index('--os-log-dir') + 1] = str(out / name / 'prefit')
    with (out / (name + '_fit.log')).open('w') as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    with open(fit, 'rb') as f:
        m = pickle.load(f)['method']
    assert m.monitor == 'loeo_xyz99'
    run = PREFIX + [str(HERE / 'plugin_matrix.py'), '--worker', '--blind', '--policy-tail',
          '--cell', 'pi05_l10_cache', '--yaml', 'exp/trace_dual/config/tr_pi05_l10_cache.yaml',
          '--root', '/home/weiland/trace_runs/offline_search_store', '--method', cmd[cmd.index('--os-method')+1],
          '--kwargs', json.dumps(kw), '--judge', 'guard_only', '--fit-artifact', fit, '--out', str(out / name / 'replay')]
    with (out / (name + '_replay.log')).open('w') as log:
        subprocess.run(run, check=True, stdout=log, stderr=subprocess.STDOUT)
    report = json.loads((out / name / 'replay/selftest_report.json').read_text())
    assert report['PASS']
    reports.append(dict(scale=scale, fit=fit, q99=float(m.policy_monitor['q99']), command=cmd, replay_command=run, **report))
    (out / 'summary.json').write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports[-1]), flush=True)
