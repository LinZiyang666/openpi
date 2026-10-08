"""R14 offline I/O boundary. No A inputs, reserved pilot identities, or launches."""
from pathlib import Path
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROUNDS = HERE.parents[1]
REPO = HERE.parents[4]
RUNS = Path('/home/weiland/trace_runs/os_closed_loop')
STORE = Path('/home/weiland/trace_runs/offline_search_store')
R10, R11 = ROUNDS/'r10', ROUNDS/'r11'
# Deliberately use only pre-R13 development identities; do not inspect the reservation.
ROOTS = ('r11_devknob_50', 'r12_devlooks_50L', 'r12_devlooks_50S',
         'r12_devlooks_200', 'r12_devlooks_knob')
CELLS = [f'{m}_{s}_{n}' for m in ('pi05','groot')
         for s, ns in [('l10',(50,200,500)),('spatial',(50,))] for n in ns]
COSTS = {'pi05':(.152,.848), 'groot':(.148,.852)}
READS = set()

def now(): return datetime.now(timezone.utc).isoformat()

def fence():
    def check(p, write=False):
        if isinstance(p,int) or p is None: return
        p = Path(os.fsdecode(p)).resolve()
        if p == Path('/dev/null'): return
        if write:
            if HERE != p and HERE not in p.parents: raise PermissionError(f'R14 write outside directory: {p}')
        else:
            if p.name.startswith('validation_selection') and 'r13' in p.parts:
                raise PermissionError('R13 reserved validation identities are unreadable')
            if RUNS == p or (RUNS in p.parents and p.relative_to(RUNS).parts[0] not in ROOTS):
                raise PermissionError(f'Unapproved closed-loop root: {p}')
            if STORE == p or (STORE in p.parents and not STORE/'library' in p.parents):
                raise PermissionError(f'Non-library store input: {p}')
            if R11/'astra'/'analysis' in p.parents and p.suffix not in ('.py','.pyc'):
                raise PermissionError(f'Mixed historical analysis: {p}')
            if ROUNDS in p.parents or RUNS in p.parents or STORE in p.parents: READS.add(str(p))
    def audit(event,args):
        if event == 'open': check(args[0],bool(args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        elif event in ('os.mkdir','os.remove','os.rmdir','os.chmod'): check(args[0],True)
        elif event in ('os.rename','os.link','os.symlink'): check(args[0],True);check(args[1],True)
        elif event in ('socket.connect','socket.bind','subprocess.Popen','os.system'):
            raise PermissionError(f'R14 offline refuses {event}')
    sys.addaudithook(audit)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def clean(x):
    import numpy as np
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,Path):return str(x)
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n')
def read(p):return json.loads(Path(p).read_text())
def csvout(p,rows):
    rows=[clean(r) for r in rows]
    with Path(p).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)
def provenance(name):dump(HERE/f'{name}_reads.json',dict(timestamp=now(),paths=sorted(READS)))
def table(head,rows):
    def cell(x):return str(x).replace('|',r'\|')
    return '\n'.join(['| '+' | '.join(map(cell,head))+' |','| '+' | '.join(['---']*len(head))+' |']+
                     ['| '+' | '.join(map(cell,r))+' |' for r in rows])
