"""One same-directory atomic replacement, with preimage guard and durable audit."""
import ast
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
BASE=Path(__file__).resolve().parent
shared=BASE.parents[4]/'exp/offline_search/closed_loop/plugin.py'
old=shared.read_bytes();candidate=(BASE/'dev/plugin.py').read_bytes()
assert old==(BASE/'before/plugin.py').read_bytes(),'Live file changed since baseline capture'
ast.parse(candidate)
fd,tmp=tempfile.mkstemp(prefix='.plugin.k6.',suffix='.tmp',dir=shared.parent)
try:
    with os.fdopen(fd,'wb') as f:
        f.write(candidate);f.flush();os.fsync(f.fileno())
    os.chmod(tmp,shared.stat().st_mode)
    os.replace(tmp,shared)
    installed=datetime.now(timezone.utc).isoformat()
    dfd=os.open(shared.parent,os.O_RDONLY)
    try:os.fsync(dfd)
    finally:os.close(dfd)
finally:
    if Path(tmp).exists():Path(tmp).unlink()
record=dict(file=str(shared),installed_utc=installed,before_sha256=hashlib.sha256(old).hexdigest(),installed_sha256=hashlib.sha256(shared.read_bytes()).hexdigest(),bytes=len(candidate),atomic='same-directory tempfile + os.replace (rename)')
assert shared.read_bytes()==candidate
(BASE/'results/install.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
