"""CPU proof of unchanged fit/Q3/gap bytes and no-filter serving equivalence."""
from __future__ import annotations

import json
from pathlib import Path
import pickle

import numpy as np

from .common import HERE, read, dump, sha
from exp.offline_search.rounds.r17.filter_failed.row_filter import r17_candidates
from exp.offline_search.rounds.r17.filter_failed.prove_retrieval import compare


def exact_compare(left,right,path):
    records=compare(left,right,path)
    # NaN is an ordinary reset-state sentinel in R16; compare its scalar bits.
    def at(value,parts):
        for part in parts:
            if hasattr(value,'__dict__'):value=vars(value)
            if isinstance(value,dict):value=value[part] if part in value else value[int(part)]
            else:value=value[int(part)]
        return value
    for record in records:
        if not record['equal']:
            parts=record['path'].split('/')[1:]
            a,b=at(left,parts),at(right,parts)
            if isinstance(a,(float,np.floating)) and isinstance(b,(float,np.floating)) and np.isnan(a) and np.isnan(b):
                aa,bb=np.asarray(a),np.asarray(b)
                record['equal']=aa.dtype==bb.dtype and aa.tobytes()==bb.tobytes()
                record['nan_sentinel_bytes_equal']=record['equal']
    return records


def run():
    report=[]
    for job in read(HERE/'CAMPAIGNS.json')['jobs']:
        root,source=Path(job['run']),Path(job['source'])
        campaign=read(root/'campaign.json');seal=read(root/'assets/CACHE_STUDY.json')
        checks=[]
        for group in seal['groups']:
            for field in ('view_rel','fit_rel','stall_rel'):
                relative=Path('cache_inputs')/group[field]
                for original in sorted((source/'assets'/relative).rglob('*')):
                    if original.is_file():checks.append(dict(path=str(original.relative_to(source/'assets')),
                        equal=sha(original)==sha(root/'assets'/original.relative_to(source/'assets'))))
            import yaml
            new=yaml.safe_load((root/'assets'/group['file']).read_text())
            original=yaml.safe_load((source/'assets'/group['file']).read_text())
            checks.append(dict(path=group['file']+'/periodic_gap',equal=new['recipe']['periodic_gap']==original['recipe']['periodic_gap']))
        checks.extend(dict(path=n,equal=sha(root/n)==sha(source/n)) for n in ('runtime.tar','workers.tar'))
        report.append(dict(job=job['name'],status='PASS' if all(r['equal'] for r in checks) else 'FAIL',checks=checks))
    proofs=[]
    from exp.r17_integration.search.fit import load_fitted_recipe
    for benchmark in ('mt50','libero'):
        job=next(j for j in read(HERE/'CAMPAIGNS.json')['jobs'] if j['benchmark']==benchmark)
        root=Path(job['run']);group=read(root/'assets/CACHE_STUDY.json')['groups'][0]
        path=root/'assets/cache_inputs'/group['fit_rel'];ready=read(path/'READY.json')
        original=load_fitted_recipe(path,expected_binding=ready['receipt']['binding'])
        selected=r17_candidates(original,np.unique(original.episode))
        records=exact_compare(vars(original),vars(selected),'fit')
        proofs.append(dict(pipeline='R17 '+benchmark,status='PASS' if all(r['equal'] for r in records) else 'FAIL',
                           source_fit=str(path),comparisons=len(records)))
    # R16 uses the R10 lower layers, inside a corrector/guard/knob wrapper.
    from .vla import candidates
    inventory=read(HERE/'vla/inventory/inventory.json')
    row=inventory['dev_sources']['pi05_l10_50_k1/0/off']['row'];args=row['plugin_args']
    path=Path(args[args.index('--os-fit-artifact')+1]);blob=pickle.loads(path.read_bytes())
    selected=candidates(blob,np.unique(blob['method'].inner.base.lib_ep))
    records=exact_compare(vars(blob['method']),vars(selected['method']),'method')
    proofs.append(dict(pipeline='R16 VLA',status='PASS' if all(r['equal'] for r in records) else 'FAIL',
                       source_fit=str(path),comparisons=len(records)))
    result=dict(status='PASS' if all(r['status']=='PASS' for r in report+proofs) else 'FAIL',
                refit=False,artifacts=report,no_filter_proofs=proofs)
    dump(HERE/'VERIFICATION.json',result)
    print(json.dumps(dict(status=result['status'],jobs=len(report),no_filter_proofs=proofs),indent=2))
    return result


if __name__=='__main__':raise SystemExit(0 if run()['status']=='PASS' else 1)
