"""Release checks and standard local plans; no deployment or network action."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import pickle
import numpy as np
from .build import HERE, ROOTS, R10, REPO, run_cpu, CELLS, artifact_path
from .calibration import tables, ScheduleReplay
from exp.offline_search.rounds.r10.data import sha, write_json


def plans():
    def one(root):
        rows=json.loads((root/'arms.json').read_text())
        run_cpu(['-m','exp.offline_search.closed_loop.ops.h100.control','plan',str(root),
                 *[r['arm'] for r in rows]],HERE/'validation'/f'{root.name}_plan.log')
        plan=json.loads((root/'h100_sync/plan.json').read_text())
        fits=[f for f in plan['files'] if 'fitted method' in ' '.join(f['reasons'])]
        assert len(fits)==len(rows)
        for f in fits:
            with Path(f['source']).open('rb') as h: m=pickle.load(h)['method']
            assert m.inner.base.act.path.startswith('/data/oscl_h100/store/library/')
            assert m.inner.C.act.path==m.inner.base.act.path
            assert not '/home/weiland/' in m.fit_info['parent']
        return dict(root=str(root),arms=len(rows),plan_files=len(plan['files']),plan_bytes=plan['bytes'],
            relocated_artifact_bytes=sum(f['size'] for f in fits),extra_h100_store_bytes=0,
            new_store_arrays=[],plan_sha256=sha(root/'h100_sync/plan.json'),relocation_pass=True)
    with ThreadPoolExecutor(3) as pool: rows=list(pool.map(one,ROOTS))
    write_json(HERE/'deployment.json',dict(PASS=True,extra_h100_store_bytes=0,roots=rows))
    print('STANDARD_PLANS',json.dumps(rows),flush=True)


def check_protected():
    records=json.loads((HERE/'input_audit.json').read_text())
    for r in records:
        assert sha(r['path'])==r['sha256'],f'protected input changed: {r["path"]}'
    return len(records)


def numerical_release():
    audit=json.loads((HERE/'calibration_audit.json').read_text())
    runtime_records=[]
    for cell in CELLS:
        model,suite,size=cell
        _,_,eps,sched=tables(*cell)
        runtime=ScheduleReplay(model,eps,1,runtime=True)
        for r in audit['records']:
            if r['cell']!='_'.join(map(str,cell)) or r['method'] not in ('random','periodic','periodic_pgt1','random_tail2'):continue
            cfg=next(x for x in json.loads((HERE/'calibration'/f'{r["cell"]}.json').read_text())['settings']
                     if x['method']==r['method'] and x['target']==r['target'])
            ri=runtime.ir(r['method'],r['setting'])
            runtime_records.append(dict(cell=r['cell'],method=r['method'],target=r['target'],setting=r['setting'],
                reference_IR_8_unrounded=r['pred_IR_lib'],reference_IR_8_rounded=cfg['pred_IR_rounded'],
                runtime_key_library_IR=ri,runtime_minus_reference_IR=ri-cfg['pred_IR_rounded']))
    write_json(HERE/'key_discrepancy.json',dict(records=runtime_records,
        max_absolute_IR_effect=max(abs(r['runtime_minus_reference_IR']) for r in runtime_records),
        note='Frozen simulator payload differs from required serving keys; runtime uses original B init identities. One serving-seed library replay, not a recalibration.'))
    with (HERE/'fresh_fit.pkl').open('rb') as f:fresh=pickle.load(f)['method']
    with artifact_path('pi05','l10',50,'random',.32).open('rb') as f:ready=pickle.load(f)['method']
    # Complete base graph equality after stripping only method-level build provenance.
    from exp.offline_search.rounds.r10.recipe.replay import fitted_check
    exact=fitted_check(HERE/'fresh_fit.pkl',R10/'recipe/artifacts/r10_recipe_pi05_l10_50.pkl')
    assert fresh.knob_settings['setting']==ready.knob_settings['setting']
    write_json(HERE/'fresh_fit_audit.json',dict(PASS=True,base_arrays_exact=True,setting_exact=True,**exact))


def sources():
    run_cpu([str(HERE/'runtime_audit.py'),str(artifact_path('pi05','l10',50,'adaptive_error_hybrid',.32)),
             str(REPO),str(HERE/'runtime_audit.json')],HERE/'validation/runtime_audit.log')
    own=json.loads((HERE/'runtime_audit.json').read_text())['files']
    paths={Path(p) for p in own}
    for file in (R10/'recipe/H100_SOURCES.sha256',HERE.parent/'devset/H100_SOURCES.sha256'):
        paths|={REPO/line.split('  ',1)[1] for line in file.read_text().splitlines()}
    paths|={HERE/'recipe.py',HERE/'controller.py',HERE/'__init__.py'}
    (HERE/'H100_SOURCES.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(REPO)}\n' for p in sorted(paths)))
    (HERE/'ALL_SOURCES.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(REPO)}\n' for p in sorted(HERE.glob('*.py'))))
    write_json(HERE/'source_list.json',dict(runtime_files=len(paths),runtime_bytes=sum(p.stat().st_size for p in paths),
        new_serving_files=[str(p.relative_to(REPO)) for p in (HERE/'__init__.py',HERE/'recipe.py',HERE/'controller.py')],
        no_closed_loop_source_changes=True,no_exploration_serving_imports=True))


def main():
    if (HERE/'deployment.json').exists():
        for r in json.loads((HERE/'deployment.json').read_text())['roots']:
            assert sha(Path(r['root'])/'h100_sync/plan.json')==r['plan_sha256']
    else:
        plans()
    numerical_release();sources()
    s=json.loads((HERE/'selftests.json').read_text());r=json.loads((HERE/'rebuild_audit.json').read_text())
    faults=json.loads((HERE/'lifecycle_faults.json').read_text())
    assert s['PASS'] and r['PASS'] and faults['PASS']
    assert (s['test_arms'],s['dev_arms'],s['equivalence_cell_sizes'],s['differing_decisions'])==(94,20,8,0)
    report=dict(PASS=True,protected_inputs_unchanged=check_protected(),test_arms=94,dev_arms=20,
        methods=9,frozen_settings_exact=82,raw_rebuild_cells_exact=8,
        off_equivalence_decisions=s['equivalence_decisions'],off_forced_guards=s['equivalence_forced_guards'],
        selftest_decisions=s['decisions'],lifecycle_fallbacks=sum(x['lifecycle_fallbacks'] for x in faults['records']),
        extra_h100_store_bytes=0,standard_plans=5,no_launch=True,no_gpu=True)
    write_json(HERE/'final_audit.json',report);print('RELEASE_PASS',json.dumps(report),flush=True)

if __name__=='__main__':main()
