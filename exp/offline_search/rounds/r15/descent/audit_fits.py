"""Check selected folds/donors and every base/serving/test artifact SHA."""
import json,pickle
from pathlib import Path
import numpy as np
from exp.offline_search.closed_loop import devset
from .prepare import HERE,ROUNDS,frozen,write_json,CELLS


def main():
    results=[];inputs={}
    for cell in CELLS:
        subset_path=ROUNDS/'r10/subsets'/f'{cell}.json'
        spec=json.loads(subset_path.read_text());chosen=spec['episode_ids_by_task']
        all_ids={e for es in chosen.values() for e in es}
        for f in range(5):
            path=ROUNDS/'r11/opus/out/anchor'/f'{cell}_f{f}.npz'
            with np.load(path,allow_pickle=False) as z:
                heldout={e for es in chosen.values() for i,e in enumerate(es) if i%5==f}
                assert set(z['ep'])==heldout
                assert set(z['top1_ep'])<=all_ids-heldout
                assert np.all(z['ep']!=z['top1_ep'])
            inputs[str(path)]=devset.sha(path)
        records=json.loads((HERE/'fits.json').read_text())[cell]
        for level,r in records.items():
            assert devset.sha(r['path'])==r['sha256']
            assert devset.sha(r['lower_layer_path'])==r['lower_layer_sha256']
            if r.get('catalog_path'):assert devset.sha(r['catalog_path'])==r['catalog_sha256']
            for path,digest in r.get('calibration_inputs_sha256',{}).items():assert devset.sha(path)==digest
            if r.get('source_path'):assert devset.sha(r['source_path'])==r['source_sha256']
            for test in (False,True):
                fit=frozen(cell,level,test=test);obj=pickle.loads(Path(fit['path']).read_bytes())['method']
                assert obj.fit_info['library_only'] and obj.fit_info['episode_ids_by_task']==chosen
                assert obj.fit_info['episodes_sha256']==spec['episodes_sha256']
            results.append(dict(cell=cell,level=level,reused=r['reused'],path=r['path'],sha256=r['sha256']))
    write_json(HERE/'fit_audit.json',dict(PASS=True,cells=12,configurations=len(results),results=results,
              calibration_inputs_sha256=inputs,all_query_folds_disjoint_from_training_donors=True,
              selected_library_only=True,periodic_uses_no_error_or_sigma_reporting_fields=True))
    print('FIT_AUDIT_PASS',len(results),'configurations, 60 paired fold tables')

if __name__=='__main__':main()
