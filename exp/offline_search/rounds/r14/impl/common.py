from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import pickle

HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
ASTRA = ROUND / 'astra'
KNOB = ROUND / 'knob'
ROUNDS = ROUND.parent
REPO = ROUNDS.parents[2]
TRACE = Path('/home/weiland/trace_runs/os_closed_loop')
STORE = Path('/home/weiland/trace_runs/offline_search_store')
STAGED = HERE / 'test_a_staged'
SPEC = 'exp.offline_search.rounds.r14.knob.recipe:R14Knob'
PREFIX = ['taskset', '-c', '22-37,66-81', 'env', 'CUDA_VISIBLE_DEVICES=',
          'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
          'PYTHONPATH=.:src', 'PYTHONDONTWRITEBYTECODE=1', str(REPO / '.venv/bin/python')]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic(path, (json.dumps(value, indent=2, allow_nan=False) + '\n').encode())

def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)

def dump_pickle(path, value):
    atomic(path, pickle.dumps(value, protocol=4))

def key(cell, variant, target=None):
    return f'{cell}:{variant}' + ('' if target is None else f':{target}')

def source(cell, method='off', target=None):
    suffix = '' if target is None else f'_ir{round(target * 100):02d}'
    path = ROUNDS / 'r11/knob/artifacts' / f'r11_{cell}_{method}{suffix}.pkl'
    audit = json.loads((ROUNDS / 'r11/knob/calibration_audit.json').read_text())
    matches = [r for r in audit['records'] if r['artifact'] == str(path)]
    if not matches and method == 'periodic':
        # R11's economical original grid omitted .40 for the larger libraries.
        # Build its unchanged controller on the verified off graph, using the
        # frozen R14 recomputation of the original library-derived R11 cap.
        original_path, blob = source(cell)
        fit = next(r for r in json.loads((ASTRA / 'allocation_fits.json').read_text())
                   if r['cell'] == cell and r['target'] == target)
        obj = blob['method']
        obj.knob_method, obj.target = 'periodic', target
        obj.configure(dict(setting=fit['reference_K'], dose=fit['reference_K']), cell.split('_')[0])
        blob['kwargs'] = dict(blob['kwargs'], method='periodic', target=target)
        blob['r14_reference_manifest'] = dict(source_artifact_sha256=sha(original_path),
            allocation_sha256=sha(ASTRA / 'allocation_fits.json'), reference_K=fit['reference_K'],
            original_R11_controller=True, lower_layers_byte_preserved=True)
        path = KNOB / 'artifacts' / f'{cell}_periodic_ir{round(target * 100):02d}_reference.pkl'
        if not path.exists():
            dump_pickle(path, blob)
        else:
            saved = pickle.loads(path.read_bytes())
            if any(saved['r14_reference_manifest'].get(k) != v
                   for k,v in blob['r14_reference_manifest'].items()):
                raise ValueError('changed constructed R11 reference')
            blob = saved
        return path, blob
    if not matches:
        raise ValueError(f'R11 source missing from SHA audit: {path}')
    expected = matches[0]['sha256']
    if sha(path) != expected:
        raise ValueError(f'changed R11 artifact: {path}')
    return path, pickle.loads(path.read_bytes())

def protected_snapshot():
    # Hash the binding reference and old implementation files, never reserved
    # R13 identity manifests or run outcomes.
    paths = list(ASTRA.rglob('*'))
    paths += list((ROUNDS / 'r13/impl').glob('*.py'))
    paths += list((ROUNDS / 'r13/knob').glob('*.py'))
    paths += [REPO / 'exp/offline_search/closed_loop' / name
              for name in ('plugin.py', 'look_recording.py')]
    paths += list((ROUNDS / 'r10/recipe').glob('*.py'))
    return {str(p): sha(p) for p in sorted(paths) if p.is_file() and not p.is_symlink()}
