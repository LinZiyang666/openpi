"""Summarize completed local evidence and exact bounded deployment dependencies."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

from .data import HERE, RUNS, SIZES, CELLS, write_json, sha
from .build import run_root, stage_label


def deployment(stage):
    files = {}
    validations = []
    for model in ('pi05', 'groot'):
        run = run_root(stage, model)
        arms = json.loads((run / 'arms.json').read_text())
        assert len(arms) == 24
        assert json.loads((run / 'eval500.json').read_text()) == [[t, i] for t in range(10) for i in range(50)]
        for row in arms:
            report = json.loads((run / 'selftest' / row['arm'] / 'selftest_report.json').read_text())
            assert report['PASS'] and report['decisions'] == 48
            if row['kwargs']['variant'] != 'A': assert report['forced_guard_triggers'] == 4
            if stage != 1: assert report['serving_corrector_checks'] >= 4
            validations.append(report)
        plan = json.loads((run / 'h100_sync/plan.json').read_text())
        assert len(plan['arms']) == 24
        for f in plan['files']:
            if f['rel'] in files:
                assert files[f['rel']]['sha256'] == f['sha256']
            files[f['rel']] = f
    new_files = [f for f in files.values() if ('/r10/' in f['original'] or '/r10_' in f['original'])]
    report = dict(stage=stage, arms=48, selftests=len(validations), plugin_decisions=sum(r['decisions'] for r in validations),
        forced_guard_triggers=sum(r['forced_guard_triggers'] for r in validations), control_plans_passed=2,
        extra_h100_store_bytes=0, new_store_arrays=[],
        total_unique_plan_bytes=sum(f['size'] for f in files.values()),
        new_run_and_mirror_dependency_bytes=sum(f['size'] for f in new_files),
        new_dependencies=new_files, validation_reports=validations)
    write_json(HERE / f'stage{stage}_deployment.json', report)
    sources = sorted(HERE.glob('*.py'))
    (HERE / f'STAGE{stage_label(stage)}_SOURCES.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(HERE.parents[3])}\n' for p in sources))
    return report


def cv_report(stage=2):
    rows = []
    for model, suite in CELLS:
        for size in SIZES:
            for variant in ('loeo', 'pair'):
                path = HERE / ('training2b' if stage == '2b' else 'training') / f'{model}_{suite}_{size}' / f'cv_{variant}.json'
                report = json.loads(path.read_text())
                assert len(report['tasks']) == 10 and all(len(t['folds']) == 3 for t in report['tasks'])
                rows.append(dict(model=model, suite=suite, size=size, variant=variant,
                    training_examples=sum(t['rows'] for t in report['tasks']),
                    anchor_rows=sum(t.get('anchors', t['rows']) for t in report['tasks']),
                    total_weight=sum(t.get('weight_sum', t['rows']) for t in report['tasks']),
                    baseline_mse=report['baseline_mse'], blend05_mse=report['blend05_mse'],
                    residual_mse=report['residual_mse'], source=str(path)))
    stem = 'offline_cv2b' if stage == '2b' else 'offline_cv'
    with (HERE / f'{stem}.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_json(HERE / f'{stem}.json', dict(fit_pool='B; no A inputs',
        diagnostic='3-fold by episode, equal episode weights; conditional on fixed deployment-cache PCA/metric/sigma',
        candidate_exclusion='held-out episodes absent from all training query/donor rows and validation donors',
        selection=False, cells=rows))
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('1', '2', '2b'))
    a = p.parse_args()
    stage = '2b' if a.stage == '2b' else int(a.stage)
    if stage != 1: cv_report(stage)
    d = deployment(stage)
    print(json.dumps({k: v for k, v in d.items() if k not in ('new_dependencies', 'validation_reports')}, indent=2))
