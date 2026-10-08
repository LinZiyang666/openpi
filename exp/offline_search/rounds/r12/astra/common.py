"""R12 explorer boundary, reproducibility and B-library loading."""
from pathlib import Path
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
STORE = Path('/home/weiland/trace_runs/offline_search_store')
RUNS = Path('/home/weiland/trace_runs/os_closed_loop')
DEV = {'r06_c_cal', 'r11_dev_current', 'r11_dev_size50', 'r11_dev_size200', 'r11_devknob_50'}
READS = set()

def install():
    def hook(event, args):
        if event != 'open' or isinstance(args[0], int):
            return
        name, mode, flags = args
        p = Path(os.fsdecode(name)).resolve()
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            if not (p == HERE or HERE in p.parents) and str(p) != '/dev/null':
                raise PermissionError(f'R12 writes must remain in astra: {p}')
        if 'r12' in p.parts and 'opus' in p.parts:
            raise PermissionError('R12 independent explorer directory refused')
        if STORE in p.parents and not STORE / 'library' in p.parents:
            raise PermissionError(f'non-library store refused: {p}')
        if RUNS in p.parents and p.relative_to(RUNS).parts[0] not in DEV:
            raise PermissionError(f'non-dev closed-loop data refused: {p}')
        if STORE in p.parents or RUNS in p.parents or '/rounds/r10/pca/' in str(p):
            READS.add(str(p))
    sys.addaudithook(hook)

def now():
    return datetime.now(timezone.utc).isoformat()

def dump(path, x):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def ew(ep):
    import numpy as np
    _, inv, counts = np.unique(ep, return_inverse=True, return_counts=True)
    return 1. / counts[inv]

def mean(x, ep):
    import numpy as np
    return float(np.average(x, weights=ew(ep)))

def load(cell, size):
    import numpy as np
    from exp.offline_search.rounds.r10.data import SubsetLibrary
    lib = SubsetLibrary(STORE, cell, size)
    root = REPO / 'exp/offline_search/rounds/r10'
    spec = json.loads((root / 'subsets' / f'{cell}_{size}.json').read_text())
    fold = {e: i % 5 for ids in spec['episode_ids_by_task'].values() for i, e in enumerate(ids)}
    ps, provenance = [], []
    for cam in (0, 1):
        path = root / 'pca' / cell / str(size) / f'v{cam}.npz'
        with np.load(path, allow_pickle=False) as z:
            assert str(z['rows_sha256']) == sha(root / 'subsets' / f'{cell}_{size}.npy')
            ps.append(np.array(z['proj'], np.float32))
            provenance.append(dict(path=str(path), sha256=sha(path), rows_sha256=str(z['rows_sha256'])))
    ep = np.array(lib.episode)
    return dict(P=np.concatenate(ps, 1), rs=np.asarray(lib.rs[:, :8]),
                act=np.asarray(lib.action[:, :10, :7]), ep=ep, row=lib.rows,
                task=np.array(lib.task_id), step=np.array(lib.step), progress=np.array(lib.progress),
                prev=np.array(lib.prev), success=np.array(lib.success),
                fold=np.array([fold[int(e)] for e in ep]), kref=5 if size == 50 else 8,
                cell=cell, size=size, provenance=provenance), lib
