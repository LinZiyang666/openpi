"""Install only K4-owned shared files, checking the saved originals; each replacement atomic."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

HERE=Path(__file__).resolve().parent
OPS=HERE.parents[4]/'exp/offline_search/closed_loop/ops'
files=['emit_arms.py','collect.py','kpi.py','chain.sh','pilot.sh','remote/run_arm.sh','remote/run_gtp_subset.py','remote/count.py']
log=HERE/'results/install_times.json'
prior=json.loads(log.read_text()) if log.exists() else []
for rel in files:
    src=HERE/'dev'/rel; dst=OPS/rel; before=HERE/'before'/rel
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    known={digest(before),digest(src)}|{r['sha256'] for r in prior if r['file']==str(dst)}
    if digest(dst) not in known:
        raise SystemExit(f'concurrent edit to owned file {dst}; inspect before replacing')
for rel in files:
    src=HERE/'dev'/rel; dst=OPS/rel
    if src.read_bytes()==dst.read_bytes(): continue
    fd,tmp=tempfile.mkstemp(prefix=f'.{dst.name}.k4.',dir=dst.parent)
    with os.fdopen(fd,'wb') as f:
        f.write(src.read_bytes()); f.flush(); os.fsync(f.fileno())
    os.chmod(tmp,dst.stat().st_mode & 0o777)
    os.replace(tmp,dst)
    record=dict(file=str(dst),installed_at=datetime.now(timezone.utc).isoformat(),sha256=hashlib.sha256(dst.read_bytes()).hexdigest())
    prior.append(record); print(json.dumps(record),flush=True)
for rel in ['cost_ledger.py','estimators.py','seeded_inference.py']:
    src=HERE/'dev'/rel; dst=HERE/rel
    fd,tmp=tempfile.mkstemp(prefix=f'.{dst.name}.k4.',dir=dst.parent)
    with os.fdopen(fd,'wb') as f:
        f.write(src.read_bytes()); f.flush(); os.fsync(f.fileno())
    os.chmod(tmp,0o664)
    os.replace(tmp,dst)
    record=dict(file=str(dst),installed_at=datetime.now(timezone.utc).isoformat(),sha256=hashlib.sha256(dst.read_bytes()).hexdigest())
    prior.append(record); print(json.dumps(record),flush=True)
log.write_text(json.dumps(prior,indent=2)+'\n')
