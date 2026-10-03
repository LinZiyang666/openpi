"""Actual plugin anomalous lifecycle/accepted-ledger integration, both models."""
import json
from .build import HERE,ROOTS
from .verification import isolated
from exp.offline_search.rounds.r10.data import write_json

if __name__=='__main__':
    rows=[r for root in ROOTS[:4] for r in json.loads((root/'arms.json').read_text())]
    reports=[]
    for model in ('pi05','groot'):
        r=next(r for r in rows if r['model']==model and r['suite_short']=='l10' and r['r11_method']=='adaptive_error_hybrid' and r['target_ir']==.32)
        r=dict(r,fault_lifecycle=True)
        _,report=isolated(r,r['arm']+'_lifecycle_fault')
        assert report['lifecycle_fallbacks']==10
        reports.append(dict(model=model,**report))
        print('LIFECYCLE_FAULT_PASS',model,report['lifecycle_fallbacks'],report['decisions'],flush=True)
    write_json(HERE/'lifecycle_faults.json',dict(PASS=True,records=reports))
