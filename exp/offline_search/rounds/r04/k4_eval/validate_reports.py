"""Check real weighted reports against collect (read-only) and freeze compact evidence."""
import hashlib
import json
from pathlib import Path

from exp.offline_search.closed_loop.ops.collect import summarize

HERE=Path(__file__).resolve().parent
ROOT=Path('/home/weiland/trace_runs/os_closed_loop/r03_mx')
records=[]
for scale,arm in [(50,'r3mx_p_l10_g'),(500,'r3mx_p_l10_g500')]:
    manifest=HERE.parent/f'ideation_C/pilot_manifest_pi05_l10_{scale}_n200.json'
    report=json.loads((HERE/f'results/pilot{scale}.json').read_text())
    paths=[ROOT/'summary.json',ROOT/'runs'/arm/'summary.json']
    before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    c=summarize(ROOT,arm,manifest=str(manifest),write=False)
    assert c['weighted']==report['arms'][0]['weighted']
    assert c['cost_ledger']==report['arms'][0]['cost_ledger']
    assert before==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    p=report['paired'][0]['weighted']
    records.append(dict(scale=scale,n=c['complete'],raw_sr=c['sr'],weighted_sr=c['weighted']['sr'],
                        weighted_delta=p['delta_sr'],design_variance=p['design_variance'],design95=p['design_normal95'],
                        ir=c['cost_ledger']['ir_per_five_controls'],original_summaries_unchanged=True))
(HERE/'results/pilot_validation.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
