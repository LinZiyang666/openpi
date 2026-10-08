"""R13 B-only I/O fence and small serialization helpers."""
from pathlib import Path
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
RUNS = Path('/home/weiland/trace_runs/os_closed_loop')
R11 = HERE.parents[1] / 'r11'
ROOTS = ('r11_dev_current', 'r11_dev_size50', 'r11_dev_size200', 'r11_devknob_50',
         'r12_devlooks_50L', 'r12_devlooks_50S', 'r12_devlooks_200',
         'r12_devlooks_200_local', 'r12_devlooks_50S_local', 'r12_devlooks_400',
         'r12_devlooks_knob')
READS = set()
COSTS = {'pi05': (.152, .848), 'groot': (.148, .852)}
CELLS = [f'{m}_{s}_{n}' for m in COSTS for s, ns in [('l10', [50, 200, 500]), ('spatial', [50])] for n in ns]

def now():
    return datetime.now(timezone.utc).isoformat()

def fence():
    def check(path, write=False):
        if isinstance(path, int) or path is None:
            return
        p = Path(os.fsdecode(path)).resolve()
        if str(p) == '/dev/null':
            return
        if write:
            if p != HERE and HERE not in p.parents:
                raise PermissionError(f'R13 astra write outside owned directory: {p}')
        else:
            if RUNS in p.parents and p.relative_to(RUNS).parts[0] not in ROOTS:
                raise PermissionError(f'Unapproved closed-loop input: {p}')
            if 'offline_search_store' in p.parts and 'library' not in p.parts:
                raise PermissionError(f'Non-library store input: {p}')
            if R11 / 'astra' / 'analysis' in p.parents and p.suffix != '.py':
                raise PermissionError(f'Mixed A/B historical analysis artifact: {p}')
            if RUNS in p.parents or 'rounds' in p.parts or 'offline_search_store' in p.parts:
                READS.add(str(p))
    def audit(event, args):
        if event == 'open':
            check(args[0], bool(args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
        elif event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod'):
            check(args[0], True)
        elif event in ('os.rename', 'os.link', 'os.symlink'):
            check(args[0], True); check(args[1], True)
        elif event in ('socket.connect', 'socket.bind', 'subprocess.Popen', 'os.system'):
            raise PermissionError(f'R13 offline analysis refuses {event}')
    sys.addaudithook(audit)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def clean(x):
    import numpy as np
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    if isinstance(x, Path): return str(x)
    return x

def dump(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(clean(x), indent=2, allow_nan=False)+'\n')

def read(p):
    return json.loads(Path(p).read_text())

def csvout(p, rows):
    rows = [clean(r) for r in rows]
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)

def provenance(name):
    dump(HERE / f'{name}_reads.json', dict(timestamp=now(), paths=sorted(READS)))

def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str, r))+' |' for r in rows])
