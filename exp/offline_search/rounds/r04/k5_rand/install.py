"""Atomic shared-file installation; refuse an unexpected live edit."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
BASE=Path(__file__).resolve().parent
SHARED=BASE.parents[2]/'closed_loop'
records=[]
for name in ('plugin.py','selftest.py','verify_logs.py'):
    dst=SHARED/name; data=(BASE/'dev'/name).read_bytes()
    before=dst.read_bytes()
    assert before in ((BASE/'before'/name).read_bytes(),data),f'{dst}: unexpected concurrent edit'
    fd,tmp=tempfile.mkstemp(prefix='.k5_install_',dir=dst.parent)
    try:
        with os.fdopen(fd,'wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
        os.chmod(tmp,dst.stat().st_mode & 0o777)
        os.replace(tmp,dst)
    finally:
        if Path(tmp).exists(): Path(tmp).unlink()
    records.append(dict(path=str(dst),installed_utc=datetime.now(timezone.utc).isoformat(),sha256=hashlib.sha256(data).hexdigest(),before_sha256=hashlib.sha256(before).hexdigest()))
(BASE/'results/install_times.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
