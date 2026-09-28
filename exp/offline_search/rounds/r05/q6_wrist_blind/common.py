"""Paths and reproducible CPU-only command prefix for owned verification."""
import json
import pickle
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/weiland/trace_runs/offline_search_store')
FITS = Path('/tmp/q6_fits')
SPEC = 'exp.offline_search.rounds.r05.q6_wrist_blind.judge:WristVisionConfirmedBlindJudge'
PREFIX = ['taskset', '-c', '10-13,54-57', 'env', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
          'MKL_NUM_THREADS=1', 'NUMEXPR_NUM_THREADS=1', 'VECLIB_MAXIMUM_THREADS=1',
          'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1', 'PYTHONPATH=.:src',
          'XDG_CACHE_HOME=/tmp/q6_cache', '/home/weiland/projects/openpi/.venv/bin/python']

def arm_name(suite, scale, variant='tail'):
    return f'r5q6_p_{suite}_{scale}_{variant}'

def kwargs(scale, variant='tail'):
    return dict(base_kwargs=dict(lib='current' if scale == 50 else 'big', kref=5 if scale == 50 else 8,
        serving='anchor_tail' if variant == 'tail' else 'phase_particles',
        budget=1 if variant == 'tail' else 2, gates='budget_only' if variant == 'tail' else 'all'),
        events='none', progress_guard='noprog_span', stuck_guard='vision_confirmed')

def method(suite, scale, variant='tail'):
    with (FITS/(arm_name(suite, scale, variant)+'.pkl')).open('rb') as f:
        return pickle.load(f)['method']

def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
