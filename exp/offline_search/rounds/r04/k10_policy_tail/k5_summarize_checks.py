import hashlib
import json
from pathlib import Path
BASE=Path(__file__).resolve().parent
ROOT=Path('/tmp/k10_installed/existing')
rows=[]
for family in ('k2','k1'):
    for path in sorted((ROOT/family).glob('*/selftest_report.json')):
        r=json.loads(path.read_text()); assert r['PASS'],path
        rows.append(dict(group=family,name=path.parent.name,decisions=r['decisions'],PASS=r['PASS']))
        (BASE/'results'/'installed'/f'existing_{family}_{path.parent.name}.json').write_text(json.dumps(r,indent=2))
for path in sorted((ROOT/'k4').glob('plugin_*.json')):
    r=json.loads(path.read_text()); assert r['PASS'],path
    rows.append(dict(group='k4',name=path.stem,decisions=r['decisions'],PASS=r['PASS']))
    (BASE/'results'/'installed'/f'existing_k4_{path.name}').write_text(json.dumps(r,indent=2))
assert len(rows)==35, (len(rows),rows)
hist=json.loads((ROOT/'k2/final_hist/verify.json').read_text())
assert hist['first_diff'] is None and all(v==1 for v in hist['offline_equal'].values())
assert hist['executed_equals_selected']['equal']==hist['decisions']==41
summary=dict(checks=len(rows),decisions=sum(r['decisions'] for r in rows),rows=rows,artifact_root=str(ROOT))
(BASE/'results/installed/existing_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
