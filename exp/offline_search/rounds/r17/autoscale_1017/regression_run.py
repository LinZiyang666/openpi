"""Serial, per-test-bounded comparison; never restores files into the worktree."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SUITES = {
    'ops': ['exp/r17_integration/tests/ops'],
    'r17_conductor': ['exp/r17_integration/tests/conductor'],
    'core_conductor': ['tests/conductor'],
    'ff': ['exp/offline_search/rounds/r17/filter_failed'],
    'ffk': ['exp/offline_search/rounds/r17/filter_failed_knob'],
    'r16': ['exp/offline_search/rounds/r16'],
    'h100': ['exp/offline_search/closed_loop/ops/h100'],
    'other_importers': [
        'exp/r17_integration/tests/debug/test_library_capture.py',
        'exp/r17_integration/tests/debug/test_review_worker_entry.py',
        'exp/r17_integration/tests/benchmarks/test_libero_worker_recycle.py',
        'exp/r17_integration/tests/benchmarks/test_libero_memory.py',
        'exp/r17_integration/tests/models/pi05/test_pi05_libero.py',
        'exp/r17_integration/tests/models/pi05/test_pi05_batching.py',
        'exp/r17_integration/tests/models/pi05/test_pi05.py',
        'exp/r17_integration/tests/models/dp/test_dp_formal_launch.py',
        'exp/r17_integration/tests/models/groot/test_groot_batching.py',
        'exp/r17_integration/tests/models/groot/test_fix_round1.py',
        'exp/r17_integration/tests/models/groot/test_libero.py',
        'exp/r17_integration/tests/models/fastwam/test_fastwam_cache_study.py',
        'exp/r17_integration/tests/models/fastwam/test_mps.py',
        'exp/r17_integration/tests/models/fastwam/test_fastwam_formal_batching.py',
        'tests/robocasa365/test_ws2_harness.py',
        'tests/robocasa365/test_robocasa_episode_runner.py',
        'tests/robocasa365/test_collection_artifacts.py',
        'tests/robocasa365/test_ws2_g2_contracts.py',
        'tests/robocasa365/test_robocasa_collect_strategy.py',
        'tests/exp/test_rl_router_sweep_mixture.py',
        'tests/exp/metaworld/test_runner.py',
        'tests/exp/step_diag/test_self13_queue.py',
    ],
}
EXCLUDE = 'exp/offline_search/closed_loop/ops/h100/test_h100.py::test_resumable_daemon_pull_checks_exact_content'


def run(baseline, output, suites, variants):
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
               JAX_PLATFORMS='cpu', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               AUTOSCALE_TEST_TIMEOUT=os.environ.get('AUTOSCALE_TEST_TIMEOUT','180'))
    records = []
    for suite in suites:
        for variant in variants:
            root = ROOT
            env['PYTHONPATH'] = str(HERE/'baseline_hook')+':'+str(root/'src')+':'+str(root)
            if variant == 'before':env['AUTOSCALE_BASELINE_ROOT'] = str(ROOT)
            else:env.pop('AUTOSCALE_BASELINE_ROOT', None)
            stem = output/(suite+'_'+variant)
            cmd = [str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-p',
                'exp.offline_search.rounds.r17.autoscale_1017.cpu_only', *SUITES[suite],
                '-vv', '--tb=short', '--basetemp='+str(stem)+'_tmp',
                '--junitxml='+str(stem)+'.xml', '--deselect='+EXCLUDE,
                '-o', 'cache_dir='+str(stem)+'_cache']
            started = time.time()
            dependency_pins = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
                'exp/r17_integration/common/serving.py',
                'exp/r17_integration/ops/cache_study.py',
                'exp/r17_integration/ops/cache_knob.py',
                'exp/r17_integration/ops/library_campaign.py')}
            print(suite, variant, 'started', flush=True)
            with stem.with_suffix('.log').open('w') as log:
                proc = subprocess.run(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            records.append(dict(suite=suite, variant=variant, code=proc.returncode,
                seconds=time.time()-started, command=cmd, cwd=str(root), dependency_sha256=dependency_pins))
            (output/'COMMANDS.json').write_text(json.dumps(records, indent=2)+'\n')
            print(suite, variant, 'exit', proc.returncode, flush=True)
    return records


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--suites', nargs='+', choices=SUITES, default=list(SUITES))
    p.add_argument('--variants', nargs='+', choices=('before','after'), default=['before','after'])
    a = p.parse_args()
    run(a.baseline.resolve(), a.output.resolve(), a.suites, a.variants)
