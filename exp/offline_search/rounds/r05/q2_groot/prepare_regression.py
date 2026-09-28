"""Copy existing K10 recipes, changing only affinity, output paths and loaders."""
from pathlib import Path
B=Path(__file__).resolve().parent
R4=B.parents[1]/'r04'
K=R4/'k10_policy_tail'
R=B/'regression';R.mkdir(exist_ok=True)
(R/'dev').mkdir(exist_ok=True)
for name in ('blind','plugin','verify_logs','selftest'):
    pass
names=['launch_test.py','check_parity.py','concurrency_test.py','edge_test.py','tail_concurrency_test.py','policy_tail_test.py','method_test.py','run_tail_matrix.py','run_k7.py','k7_unit.py','install.py','arms_rand.json','arms_k10.json']
names += [p.name for p in K.glob('k5_*.py')]
for name in names:
    s=(K/name).read_text()
    s=s.replace('26-29,70-73','30-33,74-77').replace('/tmp/k10_','/tmp/q2_').replace('/tmp/k7_guard_fits/', '/tmp/q2_regression_fits/')
    s=s.replace('BASE.parents[4]',"next(p for p in BASE.parents if (p/'exp/trace_dual/config').is_dir())")
    s=s.replace("BASE/source/(short+'.py')", "BASE.parent/source/(short+'.py')")
    s=s.replace("BASE.parent/'k7_guard/arms_k7.json'",f"Path({str(R4/'k7_guard/arms_k7.json')!r})")
    if name=='concurrency_test.py':s=s.replace("('serialized','before')","('serialized',a.source)")
    (R/name).write_text(s)
for name in ('installed_k2.sh','installed_k1.sh','installed_k4.sh','installed_test_overlay.py','installed_run_random_replays.py'):
    s=(K/'dev'/name).read_text().replace(str(K),str(R)).replace('exp/offline_search/rounds/r04/k10_policy_tail',str(R))
    s=s.replace('26-29,70-73','30-33,74-77').replace('/tmp/k10_','/tmp/q2_')
    s=s.replace("BASE/'before'", "BASE.parent/'before'")
    s=s.replace("BASE/source/(name+'.py')", "BASE.parent/source/(name+'.py')")
    (R/'dev'/name).write_text(s)
s=(K/'run_installed.sh').read_text().replace('26-29,70-73','30-33,74-77').replace('/tmp/k10_','/tmp/q2_')
s=s.replace('B=exp/offline_search/rounds/r04/k10_policy_tail',f'B={R}')
s='\n'.join(l for l in s.splitlines() if 'prepare_checks.py' not in l and 'prepare_extra.py' not in l)+'\n'
(R/'run_installed.sh').write_text(s)
(R/'results/installed').mkdir(parents=True,exist_ok=True)
# Atomic installer targets the owned live paths; original preimages are sibling before/.
s=(K/'install.py').read_text().replace("B.parents[4]", "next(p for p in B.parents if (p/'exp/trace_dual/config').is_dir())").replace("'.k10.'","'.q2.'")
(B/'install.py').write_text(s)
