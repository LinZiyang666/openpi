"""Snapshot only the five coordinator-authorized B roots."""
from .dev_validation import *
out=[]
for root in VALID_ROOTS:
    p=RUNS/root/'arms.json'
    if not p.exists():out.append(dict(root=root,status='not present'));continue
    for arm in json.loads(p.read_text()):
        assert arm['init_pool']=='B' and arm['dev']
        run=RUNS/root/'runs'/arm['arm'];s=run/'summary.json'
        summary=json.loads(s.read_text()) if s.exists() else {}
        manifests=list(run.glob('server_*/looks/manifest_*.json'))
        out.append(dict(root=root,arm=arm['arm'],complete=summary.get('complete'),
            looks_indexed=sum(len(json.loads(p.read_text())['shards']) for p in manifests),summary_exists=s.exists()))
dump(HERE/'dev_availability.json',dict(timestamp=now(),arms=out))
print(json.dumps(out,indent=2))
