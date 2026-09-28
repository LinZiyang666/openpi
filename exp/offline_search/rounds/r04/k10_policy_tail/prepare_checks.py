"""Relocate original owner recipes; every output remains in K10 or /tmp."""
from pathlib import Path
import sys
BASE=Path(__file__).resolve().parent
R4=BASE.parent
source=sys.argv[1]
root=BASE/'results'/source;root.mkdir(parents=True,exist_ok=True)
scratch=f'/tmp/k10_{source}'
loader=f'{BASE}/launch_test.py {source} normal'
for tag,path,cpu in [('k2',R4/'k2_serving/run_checks.sh','22-25,66-69'),('k4',R4/'k4_eval/run_plugin_selftests.sh','30-33,74-77'),('k1',R4/'k1_blind/run_plugin_blind.sh','18-21,62-65')]:
    s=path.read_text().replace(cpu,'26-29,70-73')
    s=s.replace(f'exp/offline_search/rounds/r04/{path.parent.name}/results',f'{scratch}/existing/{tag}')
    s=s.replace('/tmp/k4_plugin.',f'/tmp/k10_{source}_k4_plugin.')
    s=s.replace('MKL_NUM_THREADS=1 PYTHONPATH','MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH')
    s=s.replace('-m exp.offline_search.closed_loop.selftest',loader)
    if tag=='k2':
        s=s.replace('exp/offline_search/rounds/r04/k2_serving/check_log_only.py',f'{BASE}/launch_test.py {source} log_r4 --cell pi05_spatial_cache --yaml exp/trace_dual/config/tr_pi05_sp_cache.yaml --root /home/weiland/trace_runs/offline_search_store --method exp.offline_search.closed_loop.probe:ProbeForce --judge periodic:5 --episodes 2 --out "$OUT/final_log_only"')
        s=s.replace('--episodes 2 --out','--root /home/weiland/trace_runs/offline_search_store --episodes 2 --out')
    if tag=='k1':s=s.replace('PYTHONDONTWRITEBYTECODE=1 /home','PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home')
    s=s.replace('set -eu\n',f'set -eu\nmkdir -p {scratch}/existing/{tag}\n').replace('set -euo pipefail\n',f'set -euo pipefail\nmkdir -p {scratch}/existing/{tag}\n')
    (BASE/'dev'/f'{source}_{tag}.sh').write_text(s)
for name in ('run_random_replays.py','test_overlay.py'):
    s=(R4/'k5_rand'/name).read_text().replace('BASE=Path(__file__).resolve().parent',f'BASE=Path({str(BASE)!r})')
    s=s.replace("BASE/'results'",f"BASE/'results'/{source!r}").replace('K5_TEST_ARGS','K10_TEST_ARGS').replace('/tmp/k5_','/tmp/k10_')
    s=s.replace("for name in ('plugin','verify_logs','selftest'):","for name in ('blind','plugin','verify_logs','selftest'):")
    (BASE/'dev'/f'{source}_{name}').write_text(s)
(BASE/'arms_rand.json').write_bytes((R4/'k5_rand/arms_rand.json').read_bytes())
for name in ('concurrency_test.py','edge_test.py'):
    s=(R4/'k6_concurrency'/name).read_text()
    if name=='concurrency_test.py':
        start=s.index('def load(source):');end=s.index('\n\ndef configs()',start)
        s=s[:start]+'from launch_test import load\n'+s[end:]
    (BASE/name).write_text(s)
