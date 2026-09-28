"""Dependency-first, preimage-checked, one atomic rename per shared file."""
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
B=Path(__file__).resolve().parent
S=next(p for p in B.parents if (p/'exp/trace_dual/config').is_dir())/'exp/offline_search/closed_loop'
files=('blind.py','plugin.py','verify_logs.py','selftest.py','replay_client.py')
for name in files:
    assert (S/name).read_bytes()==(B/'before'/name).read_bytes(),f'live preimage changed: {name}'
    ast.parse((B/'dev'/name).read_text())
result=[]
for name in files:
    path=S/name;content=(B/'dev'/name).read_bytes()
    fd,tmp=tempfile.mkstemp(prefix='.'+name+'.q2.',dir=S)
    with os.fdopen(fd,'wb') as f:f.write(content);f.flush();os.fsync(f.fileno())
    os.chmod(tmp,path.stat().st_mode & 0o777)
    os.replace(tmp,path)
    fd=os.open(S,os.O_RDONLY);os.fsync(fd);os.close(fd)
    result.append(dict(path=str(path),installed_utc=datetime.now(timezone.utc).isoformat(),sha256=hashlib.sha256(content).hexdigest(),bytes=len(content),before_sha256=hashlib.sha256((B/'before'/name).read_bytes()).hexdigest()))
(B/'results/install.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
