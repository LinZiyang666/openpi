import json
from pathlib import Path
import shutil
from exp.offline_search.closed_loop.ops.collect import summarize
BASE=Path(__file__).resolve().parent
root=Path('/tmp/k6_replay_estimator')
shutil.copyfile('/tmp/k6_arms_check/arms.json',root/'arms.json')
reports=[]
for arm in json.loads((root/'arms.json').read_text()):
    name=arm['arm']; r=summarize(root,name,write=False)
    ledger=r['cost_ledger']
    assert r['complete']==4 and r['server']['decisions_logged']==342
    assert ledger['v']==1.0
    reports.append(dict(arm=name,complete=r['complete'],decisions=r['server']['decisions_logged'],cost_ledger=ledger))
(BASE/'results/installed/ledger_check.json').write_text(json.dumps(reports,indent=2))
print(json.dumps(reports,indent=2))
