from __future__ import annotations

import hashlib
import importlib
import json
import pickle
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = Path('/tmp/q1_method_c_fits')
STORE = Path('/home/weiland/trace_runs/offline_search_store')
SCHEMA = 'r6.method_c.v1'
CONTROLLER_VERSION = 'R6-C-v2'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(4 << 20):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def output_path(path):
    p = Path(path).resolve()
    if not any(p == root or root in p.parents for root in (HERE, OUT)):
        raise ValueError('writes must remain in method_c/ or /tmp/q1_method_c_fits/')
    return p


def sources():
    path = HERE.parents[1] / 'ideation_Q2/frontier/adapters/replay_baselines.json'
    return {r['cell']: r for r in json.loads(path.read_text()) if r['name'].endswith('_A')}


class FitUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('osm_'):
            for package in ('exp.offline_search.rounds.r04.k1_blind.blind_awm',
                            'exp.offline_search.rounds.r02.g1_awm.awm'):
                mod = importlib.import_module(package)
                if hasattr(mod, name):
                    return getattr(mod, name)
        return super().find_class(module, name)


def load_base(source):
    from exp.offline_search.harness import api
    with Path(source['source_artifact']).open('rb') as f:
        blob = FitUnpickler(f).load()
    if blob['kwargs'] != source['kwargs'] or blob['spec'] != source['method']:
        raise ValueError('deployed base fit metadata mismatch')
    method = blob['method']
    method.prof = api.NULL_PROFILER
    return method, blob


def fingerprint(method):
    h = hashlib.sha256()
    def add(k, v):
        h.update(k.encode())
        if isinstance(v, np.ndarray):
            h.update(str((v.shape, v.dtype)).encode()); h.update(v.tobytes())
        else:
            h.update(repr(v).encode())
    for k in ('k', 'kref', 'early', 'features', 'feat0', 'lam_c', 'norm_cap', 'hyst',
              'act', 'sig', 'lib_ep', 'lib_step', 'B0T', 'B1T', 'muB0', 'muB1'):
        add(k, getattr(method, k, None))
    for t, task in sorted(method.tasks.items()):
        for k in ('rows', 'Wf', 'shift', 'Z', 'z2', 'W0f', 'c0', 'A0', 'As0', 'n20', 'Z0', 'RS'):
            add(str(t) + ':' + k, getattr(task, k, None))
    return h.hexdigest()


def read_bank(path, base=None, verify_library=True):
    path = Path(path)
    meta = json.loads(path.read_text())
    if meta['schema'] != SCHEMA + '.r_bank':
        raise ValueError('unknown R bank schema')
    arrays_path = path.parent / meta['arrays']
    if sha(arrays_path) != meta['arrays_sha256']:
        raise ValueError('R bank array hash mismatch')
    if verify_library:
        libdir = Path(meta['library_directory'])
        for name, digest in meta['library_hashes'].items():
            if sha(libdir / name) != digest:
                raise ValueError('R bank belongs to different library: ' + name)
    if base is not None and fingerprint(base) != meta['retrieval_fingerprint']:
        raise ValueError('R bank belongs to different deployed retrieval fit')
    with np.load(arrays_path, allow_pickle=False) as f:
        arrays = {k: np.array(f[k], copy=True) for k in f.files}
    for v in arrays.values():
        v.flags.writeable = False
    return meta, arrays


def main_code(base, key_v0, key_v1, rs, task_id):
    """Stall interface key: deployed MAIN metric code, float32, even at step 0."""
    from exp.offline_search.harness import dims
    xv = np.concatenate((base.B0T @ np.asarray(key_v0, np.float32) - base.muB0,
                         base.B1T @ np.asarray(key_v1, np.float32) - base.muB1))
    state = np.asarray(dims.valid_state(rs, base.model), np.float32)
    x = np.concatenate((xv, state)) if base.features == 'joint' else xv
    task = base.tasks[int(task_id)]
    return np.asarray(x @ task.Wf - task.shift, np.float32)
