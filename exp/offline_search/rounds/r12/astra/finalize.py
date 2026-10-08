"""Validate final atlas ingredients and freeze the completed offline handback."""
from .common import *
install()
import ast
import numpy as np

deliverables=['OFFLINE.md','SPEC.md','PREDICTION.md','EXPERIMENTS.md','REPORT.md','HANDBACK.md','REQUESTS.md']
for name in deliverables:assert (HERE/name).is_file() and (HERE/name).stat().st_size>0
for path in HERE.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
artifacts=[]
for path in sorted((HERE/'final_atlas').glob('*/fit.json')):
    info=json.loads(path.read_text());C,_=load(info['cell'],info['size'])
    assert info['fit_pool']=='B selected library only' and info['progress_label']
    assert sha(path.parent/'table.npz')==info['table_sha256']
    with np.load(path.parent/'table.npz') as z:
        np.testing.assert_array_equal(z['row_subset'],C['row'])
        assert not z['safe'][C['step']<4].any()
        assert np.all(z['state_scale']>0) and np.all(z['sigma']>0)
        assert int(z['safe'].sum())==info['safe_rows']
        n=np.array(z['next2']);ok=n>=0
        np.testing.assert_array_equal(C['ep'][n[ok]],C['ep'][ok])
        np.testing.assert_array_equal(C['step'][n[ok]]-C['step'][ok],np.full(ok.sum(),2))
        for task in range(10):
            for view,width in [('full',136),('half',144)]:
                rr=z[f'{view}_{task}_rows']
                assert np.all(C['task'][rr]==task)
                assert z[f'{view}_{task}_W'].shape==(width,width)
                assert np.isfinite(z[f'{view}_{task}_Z']).all()
    artifacts.append(info)
assert len(artifacts)==12
p=json.loads((HERE/'predictions.json').read_text());assert len(p['cells'])==24
assert sum(x['baseline_dev_arm'] is not None for x in p['cells'])==12
assert p['timestamp'] in (HERE/'PREDICTION.md').read_text()
assert json.loads((HERE/'validation.json').read_text())['pass_all']
for name in ('retrieval','correction','correction_dynamic','history','atlas'):
    assert len(list((HERE/name).glob('*/summary.json')))==12
assert len(list((HERE/'representations').glob('*/summary.json')))==4
logs=['inventory','retrieval','correction','correction_dynamic','representations','history','atlas','gates','gates_dynamic',
      'dev_cost','atlas_assessment','atlas_guard_assessment','check','fit_selected_atlas','make_reports','figures']
for name in logs:
    assert 'Traceback' not in (HERE/f'{name}.log').read_text(),name

files={str(path.relative_to(HERE)):sha(path) for path in sorted(HERE.glob('*'))
       if path.is_file() and path.suffix in ('.py','.md','.json','.png','.svg') and path.name!='FINAL_MANIFEST.json'}
upstream=['exp/offline_search/rounds/r12/BRIEF.md','exp/offline_search/rounds/r02/g1_awm/awm.py',
          'exp/offline_search/rounds/r10/data.py','exp/offline_search/rounds/r10/train.py',
          'exp/offline_search/rounds/r10/recipe/recipe.py','exp/offline_search/rounds/r10/recipe/fitting.py',
          'exp/offline_search/rounds/r10/astra/offline.py','exp/offline_search/rounds/r11/knob/recipe.py',
          'exp/offline_search/closed_loop/plugin.py','exp/offline_search/closed_loop/stage_overrides.py']
manifest=dict(timestamp=now(),prediction_timestamp=p['timestamp'],deliverables=deliverables,files=files,
              final_atlas_tables={f'{x["cell"]}_{x["size"]}':x['table_sha256'] for x in artifacts},
              upstream_snapshot={x:sha(REPO/x) for x in upstream},
              upstream_note='Handback-time source snapshot, not a claim that mutable upstream files were locked throughout exploration.',
              no_changed_controller_results=True,no_launches=True,gpu_used=False,pass_all=True)
dump(HERE/'FINAL_MANIFEST.json',manifest)
print(json.dumps(dict(timestamp=manifest['timestamp'],deliverables=deliverables,artifacts=len(artifacts),
                     prediction_rows=len(p['cells']),validated=True),indent=2))
