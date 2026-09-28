"""Copy existing check recipes, changing only affinity, artifact paths and test loader."""
from pathlib import Path
BASE=Path(__file__).resolve().parent
R4=BASE.parent
out='/tmp/k5_existing_installed'
loader='exp/offline_search/rounds/r04/k5_rand/launch_test.py installed normal'
for tag,path,cpu in [('k2',R4/'k2_serving/run_checks.sh','22-25,66-69'),('k4',R4/'k4_eval/run_plugin_selftests.sh','30-33,74-77'),('k1',R4/'k1_blind/run_plugin_blind.sh','18-21,62-65')]:
    s=path.read_text().replace(cpu,'26-29,70-73')
    s=s.replace(f'exp/offline_search/rounds/r04/{path.parent.name}/results',f'{out}/{tag}')
    s=s.replace('MKL_NUM_THREADS=1 PYTHONPATH','MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH')
    s=s.replace('-m exp.offline_search.closed_loop.selftest',loader)
    if tag=='k2':
        s=s.replace('exp/offline_search/rounds/r04/k2_serving/check_log_only.py',
            'exp/offline_search/rounds/r04/k5_rand/launch_test.py installed log_r4 --cell pi05_spatial_cache --yaml exp/trace_dual/config/tr_pi05_sp_cache.yaml --root /home/weiland/trace_runs/offline_search_store --method exp.offline_search.closed_loop.probe:ProbeForce --judge periodic:5 --episodes 2 --out "$OUT/final_log_only"')
        # The hot-store default is unavailable in this sandbox.
        s=s.replace('--episodes 2 --out','--root /home/weiland/trace_runs/offline_search_store --episodes 2 --out')
    if tag=='k1': s=s.replace('PYTHONDONTWRITEBYTECODE=1 /home','PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home')
    s=s.replace('set -eu\n','set -eu\nmkdir -p '+out+'/'+tag+'\n').replace('set -euo pipefail\n','set -euo pipefail\nmkdir -p '+out+'/'+tag+'\n')
    (BASE/'dev'/f'existing_{tag}.sh').write_text(s)
