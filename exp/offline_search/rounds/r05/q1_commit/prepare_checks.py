"""Relocate K10's existing regression recipes without changing owner files."""
from pathlib import Path
from prepare import write

HERE = Path(__file__).resolve().parent
R4 = HERE.parent.parent / 'r04'
OLD = R4 / 'k10_policy_tail'
NEW = HERE / 'regression'
NEW.mkdir(exist_ok=True)
(NEW / 'dev').mkdir(exist_ok=True)
(NEW / 'results/installed').mkdir(parents=True, exist_ok=True)
files = ['launch_test.py', 'check_parity.py', 'concurrency_test.py', 'edge_test.py', 'policy_tail_test.py',
         'tail_concurrency_test.py', 'run_k7.py', 'k7_unit.py', 'run_tail_matrix.py', 'method_test.py',
         'arms_rand.json', 'arms_k10.json']
files += [p.name for p in OLD.glob('k5_*.py')]
files += ['dev/installed_' + n for n in ('k1.sh', 'k2.sh', 'k4.sh', 'test_overlay.py', 'run_random_replays.py')]
for name in files:
    s = (OLD / name).read_text()
    s = s.replace(str(OLD), str(NEW)).replace('/tmp/k10_', '/tmp/q1_reg_')
    s = s.replace("prefix='k10_estimator_'", "prefix='q1_reg_estimator_'")
    s = s.replace('BASE.parents[4]', "Path('/home/weiland/projects/openpi')")
    s = s.replace("BASE.parent/'k7_guard", f"Path({str(R4)!r})/'k7_guard")
    if name == 'launch_test.py':
        s = s.replace("BASE/source/(short+'.py')", f"Path({str(OLD)!r})/source/(short+'.py')")
    write(NEW / name, s)
# The original final audit compares installed imports with a captured snapshot.
for name in ('blind.py', 'plugin.py', 'selftest.py', 'verify_logs.py', 'replay_client.py'):
    write(NEW / 'dev' / name, (HERE.parents[2] / 'closed_loop' / name).read_text())
s = (OLD / 'run_installed.sh').read_text()
s = s[s.index('#!/'):].replace('B=exp/offline_search/rounds/r04/k10_policy_tail',
                              'B=exp/offline_search/rounds/r05/q1_commit/regression')
s = '\n'.join(line for line in s.splitlines() if 'prepare_checks.py' not in line and 'prepare_extra.py' not in line) + '\n'
s = s.replace('/tmp/k10_', '/tmp/q1_reg_')
write(HERE / 'run_regression.sh', s)
print('Relocated K2/K1/K4/K5/K6/K7/K10 recipes to', NEW)
