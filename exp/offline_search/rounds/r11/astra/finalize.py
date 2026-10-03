"""Final own-directory audit, integration comparison, and artifact manifest."""
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import re
import numpy as np
from exp.offline_search.rounds.r11.astra.boundary import HERE, READS, dump, install, sha
from exp.offline_search.rounds.r11.astra.experiment import CELLS, FEATURES


def main():
    install()
    expected=set(range(10,22))|set(range(54,66))
    assert set(os.sched_getaffinity(0))==expected
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        assert os.environ[name]=='1'
    assert os.environ['PYTHONDONTWRITEBYTECODE']=='1'
    required=['SPEC.md','OFFLINE.md','ARM_GRID.md','PREDICTION.md','REPORT.md','HANDBACK.md','RUN.md']
    for name in required:
        assert (HERE/name).stat().st_size>0
    freeze=json.loads((HERE/'freeze.json').read_text())
    cross=json.loads((HERE/'crosscheck.json').read_text())
    selfcheck=json.loads((HERE/'selfcheck.json').read_text())
    assert selfcheck['passed'] and cross['passed']
    assert sha(cross['shared_model'])==cross['shared_model_sha256']
    assert freeze['arm_count']==len(freeze['arms'])==44
    assert len({r['arm'] for r in freeze['arms']})==44
    assert freeze['timestamp_utc'] in (HERE/'PREDICTION.md').read_text()
    source_changes={name:dict(frozen=value,current=sha(HERE/name)) for name,value in freeze['source_sha256'].items()
                    if sha(HERE/name)!=value}
    assert set(source_changes)=={'controller.py','selfcheck.py'}
    assert (HERE/'POST_FREEZE_NOTES.md').exists()
    nrows=0
    for m,s,n in CELLS:
        tag=f'{m}_{s}_{n}'
        prov=json.loads((HERE/'data'/tag/'provenance.json').read_text())
        head=json.loads((HERE/'data'/tag/'predictor.json').read_text())
        assert head['feature_names']==FEATURES and not head['task_indexed']
        assert len(head['coef'])==len(FEATURES)+1
        path=HERE/'data'/tag/'signals.npz'
        assert sha(path)==prov['signals_sha256']
        with np.load(path) as z:
            a={k:np.asarray(z[k]) for k in z.files}
        assert len(np.unique(a['ep']))==n
        assert len(a['row'])==prov['rows']
        assert len(np.unique(a['row']))==len(a['row'])
        assert np.all(a['ep'][a['top']]!=a['ep'])
        assert np.all(a['fold'][a['top']]!=a['fold'])
        for x in a.values():
            assert np.isfinite(x).all()
        for f in prov['folds']:
            tr,va=set(f['train_episodes']),set(f['heldout_episodes'])
            assert not tr&va and tr|va==set(a['ep'])
            assert va==set(a['ep'][a['fold']==f['outer']])
        nrows+=len(a['row'])
    for arm in freeze['arms']:
        assert arm['calibration_feasible']
        assert sha(HERE/'data'/arm['cell']/'analysis.json')==arm['calibration_sha256']
        assert abs(arm['predicted_realized_ir']-arm['target_ir'])<.003
        assert arm['target_ir']<=.4
        if not arm['method'].startswith('adaptive'):
            assert 0<arm['dose']<1
    for filename in ('experiment_reads.json','analysis_reads.json'):
        for path in json.loads((HERE/filename).read_text()):
            assert 'os_closed_loop' not in Path(path).parts
            if 'offline_search_store' in Path(path).parts:
                assert 'library' in Path(path).parts
    # Import only schedule definitions. This invokes no fitting, data loads or jobs.
    from exp.offline_search.rounds.r11.opus.arm_grid import GRID
    opus_random={(cell,float(target)) for cell,target,method,priority in GRID if method=='random'}
    ours={(r['cell'],r['target_ir']) for r in freeze['arms']}
    missing=sorted(ours-opus_random)
    controls=len(CELLS)
    coordination=dict(astra_arms=len(freeze['arms']),opus_current_arms=len(GRID),knob_off_controls=controls,
        missing_matched_random_controls=[dict(cell=c,target=t) for c,t in missing],
        combined_if_all_included=len(freeze['arms'])+len(GRID)+controls+len(missing),
        opus_grid_source=str(HERE.parent/'opus'/'arm_grid.py'),
        opus_grid_sha256=sha(HERE.parent/'opus'/'arm_grid.py'))
    dump(HERE/'coordination.json',coordination)
    text=['# Coordination with the shared IR model and schedule grid',
        'The cost arithmetic and no-progress guard agree exactly on identical input tables; see `crosscheck.json`. '
        'The separately fitted score/guard tables have different PCA conventions and must not be combined numerically.',
        f"At handback, the proposed totals are {coordination['astra_arms']} astra arms + "
        f"{coordination['opus_current_arms']} opus arms + {controls} same-batch knob-off controls. "
        'For matched-budget random comparisons, the current opus grid lacks these state-knob targets:',
        '\n'.join(f'- `{cell}` at target owner IR `{target:.2f}`' for cell,target in missing),
        f"Adding these {len(missing)} random controls gives {coordination['combined_if_all_included']} total arms, "
        'within the brief’s approximate total budget. These additions are a coordinator recommendation, not launches '
        'or mutations to opus’s grid. If the coordinator chooses a smaller grid, retain exact matched targets for every '
        'state-dependent method under comparison.',
        'No frozen astra parameter or forecast was changed after reading the schedule grid. '
        'Source version and exact missing pairs are stored in `coordination.json`.']
    (HERE/'COORDINATION.md').write_text('\n\n'.join(text)+'\n')
    # Resolve all relative local document links; external links are not used here.
    for filename in required:
        body=(HERE/filename).read_text()
        for ref in re.findall(r'\]\(([^)]+)\)',body):
            if not ref.startswith(('http:','https:','#')):
                assert (HERE/ref.split('#')[0]).exists(),(filename,ref)
    manifest=[]
    for p in sorted(HERE.rglob('*')):
        if p.is_file() and p.suffix!='.log' and p.name not in ('MANIFEST.sha256','final_audit.json'):
            manifest.append(f'{sha(p)}  {p.relative_to(HERE)}')
    (HERE/'MANIFEST.sha256').write_text('\n'.join(manifest)+'\n')
    dump(HERE/'final_audit.json',dict(passed=True,timestamp_utc=datetime.now(timezone.utc).isoformat(),
        deliverables=required,rows=nrows,arms=len(freeze['arms']),
        freeze_sha256=sha(HERE/'freeze.json'),manifest_sha256=sha(HERE/'MANIFEST.sha256'),
        affinity=sorted(expected),cuda_visible_devices='',blas_threads=1,
        no_closed_loop_inputs_in_recorded_fit_reads=True,whole_episode_donor_exclusion=True,
        task_free_predictor=True,source_changes_after_numerical_freeze=source_changes,
        shared_model_sha256=cross['shared_model_sha256'],coordination=coordination,
        scope='Analysis, reference code, frozen proposal and handback only; no production adapter or launch.'))
    print(json.dumps(dict(passed=True,rows=nrows,arms=len(freeze['arms']),coordination=coordination),indent=2))


if __name__=='__main__':
    main()
