"""Relocate K5 run_all_checks verification helpers into K6-owned artifacts."""
from pathlib import Path
BASE=Path(__file__).resolve().parent
K5=BASE.parent/'k5_rand'
for name in ('prepare_arms.py','validate_estimator.py','check_replay_estimator.py','check_ledger.py','final_audit.py','summarize_checks.py'):
    s=(K5/name).read_text().replace('/tmp/k5_','/tmp/k6_').replace("prefix='k5_estimator_'","prefix='k6_estimator_'")
    s=s.replace("BASE/'estimate.py'",f"Path({str(K5/'estimate.py')!r})")
    s=s.replace('/tmp/k6_existing_installed','/tmp/k6_installed/existing')
    s=s.replace("BASE/'results'", "BASE/'results'/'installed'").replace("BASE/'results/", "BASE/'results/installed/")
    (BASE/('k5_'+name)).write_text(s)
for name in ('selftest.py','verify_logs.py'):
    (BASE/'dev'/name).write_bytes((BASE.parents[4]/'exp/offline_search/closed_loop'/name).read_bytes())
