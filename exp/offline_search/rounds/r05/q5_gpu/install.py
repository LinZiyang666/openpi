"""Install only plugin.py after Q2 hand-back, exact preimage, one atomic replace."""
import ast,hashlib,json,os,tempfile
from datetime import datetime,timezone
from pathlib import Path
B=Path(__file__).resolve().parent
R=next(p for p in B.parents if (p/'exp/trace_dual/config').is_dir())
assert (B.parent/'q2_groot/HANDBACK.md').is_file(),'Q2 still owns the plugin'
pre=json.loads((B/'results/preimage.json').read_text());target=Path(pre['path'])
assert hashlib.sha256(target.read_bytes()).hexdigest()==pre['sha256'],'installed preimage changed; rebase and retest first'
content=(B/'dev/plugin.py').read_bytes();ast.parse(content)
ast.parse((B/'dev/gpu_retrieval.py').read_bytes())
fd,tmp=tempfile.mkstemp(prefix='.plugin.py.q5.',dir=target.parent)
with os.fdopen(fd,'wb') as f:f.write(content);f.flush();os.fsync(f.fileno())
os.chmod(tmp,target.stat().st_mode & 0o777)
os.replace(tmp,target)
fd=os.open(target.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
record=dict(path=str(target),installed_utc=datetime.now(timezone.utc).isoformat(),before_sha256=pre['sha256'],
            sha256=hashlib.sha256(content).hexdigest(),bytes=len(content),helper_path=str(B/'dev/gpu_retrieval.py'),
            helper_sha256=hashlib.sha256((B/'dev/gpu_retrieval.py').read_bytes()).hexdigest())
(B/'results/install.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
