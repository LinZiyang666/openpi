"""Refit CU and CT only on the audited, existing R6 non-test B-val streams."""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np
import pandas as pd

from exp.offline_search.harness import store
from exp.offline_search.rounds.r06.ideation_Q1.method_c import fit_calibration as r6_fit
from exp.offline_search.rounds.r06.ideation_Q1.method_c.budget import solve
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import load_base, STORE
from exp.offline_search.rounds.r07.stages.stages import StageTable
from .common import (ALL_CELLS, DENSE_CELLS, DENSE_STAGES, RECORDINGS, SCRATCH,
                     VERSION, output_path, sha, stall_path, target_rho, write_json)
from .tilt import stage_features, lift_deviation_latch


def recorded_stage_features(tables, table, episodes):
    keys = r6_fit.AKEY
    cols = set(keys + ['retrieval.rows', 'retrieval.weights', 'assignment.cohort'])
    anchors = pd.read_csv(tables / 'anchors.csv', usecols=lambda k: k in cols)
    anchors = anchors[anchors['assignment.cohort'] == 'A']
    lookup = {tuple(r[k] for k in keys): r for r in anchors.to_dict('records')}
    cols = set(keys + ['task_id', 'init', 'absolute_input_archive',
                      'blind_features.retrieval.rows', 'blind_features.retrieval.weights'])
    decisions = pd.read_csv(tables / 'decisions.csv', usecols=lambda k: k in cols)
    groups = {(uid, int(g.task_id.iloc[0]), int(g.init.iloc[0])): g.sort_values('step')
              for (_, uid, _), g in decisions.groupby(r6_fit.EKEY, sort=True)}
    features = []
    for episode in episodes:
        group = groups[(episode['uid'], episode['task'], episode['init'])]
        if list(group.step) != list(range(episode['nominal_blocks'])):
            raise ValueError('stage-feature stream differs from the cost replay')
        values = []
        for row in group.to_dict('records'):
            anchor = lookup.get(tuple(row[k] for k in keys))
            rec, prefix = (anchor, 'retrieval.') if anchor else (row, 'blind_features.retrieval.')
            rows = r6_fit.vector(rec[prefix + 'rows']).astype(int)
            weights = r6_fit.vector(rec[prefix + 'weights'])
            with np.load(row['absolute_input_archive'], allow_pickle=False) as data:
                state = np.array(data['robot_state'])
            values.append(stage_features(table, rows, weights, state, episode['task']))
        features.append(values)
    return features


def fit_cell(cell, out, rho=None):
    rho = target_rho(cell) if rho is None else rho
    out = output_path(out)
    out.mkdir(parents=True, exist_ok=True)
    cu_dir = out / 'CU'
    tables = RECORDINGS / 'tables' / cell
    source = RECORDINGS / 'cal' / cell / 'calibrated' / 'calibration.json'
    original = json.loads(source.read_text())
    if original['status'] != 'NONTEST_BVAL':
        raise ValueError('source calibration must be NONTEST_BVAL')
    for name, digest in original['tables'].items():
        if sha(name) != digest:
            raise ValueError('R6 input table changed: ' + name)
    manifest = RECORDINGS / 'manifests' / f'calibration_manifest_{cell}.json'
    if sha(manifest) != original['bval_manifest_sha256']:
        raise ValueError('B-val source manifest changed')
    # Call the original fitter unchanged. Only its output ownership guard is
    # redirected within this invocation; no earlier-round file is modified.
    with (out / 'r6_refit.log').open('x') as log, contextlib.redirect_stdout(log), \
            patch.object(r6_fit, 'output_path', output_path):
        cal = r6_fit.fit(tables, RECORDINGS / 'cal' / cell / 'r_bank.json', cu_dir,
                         original['c1'], [rho], bval_manifest=manifest,
                         stall_model_path=stall_path(cell),
                         client_root=RECORDINGS / 'runs' / f'r6c_cal_{cell}' / 'client_telemetry')
    if cal['solutions']['stall']['uniform'][format(rho, '.12g')] != \
            original['solutions']['stall']['uniform'][format(rho, '.12g')]:
        raise AssertionError('CU refit differs from the retained R6 uniform-stall solution')
    cal['r7_calls'] = dict(version=VERSION, tilt=False, source_calibration=str(source),
                           source_sha256=sha(source))
    write_json(cu_dir / 'calibration.json', cal)
    base, _ = load_base(cal['base_source'])
    bank = json.loads((cu_dir / 'r_bank.json').read_text())
    libdir = Path(bank['library_directory'])
    library = store.LibraryView(STORE, cell.rsplit('_', 1)[0], base.cand_name)
    geometry = json.loads((libdir / 'manifest.json').read_text())
    table_path = out / 'stages.pkl'
    stage_source = None
    if cell in DENSE_CELLS:
        expected_library = 'bpool_cs' if cell.startswith('pi05_') else 'bpool_all'
        if base.cand_name != expected_library or base.kref != 8:
            raise ValueError('dense frozen A must use the deployed library and kref 8')
        stage_source = DENSE_STAGES / (cell + '.pkl')
        table = StageTable.load(stage_source, library=library, manifest=geometry)
        # Preserve C1's frozen envelope, including its content checksum.
        if table_path.exists():
            raise FileExistsError(table_path)
        shutil.copyfile(stage_source, table_path)
        if sha(stage_source) != sha(table_path):
            raise AssertionError('dense StageTable copy differs from C1')
    else:
        table = StageTable.fit(library, manifest={**geometry, '_retrieval': base})
        table.save(table_path)
    if table.retrieval_fingerprint != bank['retrieval_fingerprint']:
        raise AssertionError('StageTable must use this frozen A')
    replay = json.loads((cu_dir / 'cost_replay.json').read_text())
    trees = replay['modes']['stall']
    features = recorded_stage_features(tables, table, trees)
    tilted = [lift_deviation_latch(t, f) for t, f in zip(trees, features, strict=True)]
    solution = solve(tilted, rho, 'R', cal['c1'], 'stall')
    solution['weight_rule'] = 'event_kernel_mean_times_deviation_entry'
    solution['solver_score'] = 'call_weight (not R6 disagreement)'
    ct = copy.deepcopy(cal)
    ct['r_bank_path'] = '../CU/r_bank.json'
    ct['solutions'] = {'stall': {'uniform': {format(rho, '.12g'): solution}}}
    ct['r7_calls'] = dict(version=VERSION, tilt=True, stage_path='../stages.pkl',
                           stage_sha256=sha(table_path), stage_fingerprint=table.fingerprint,
                           stage_retrieval_fingerprint=table.retrieval_fingerprint,
                           weight_rule=solution['weight_rule'],
                           latch='fresh anchors only; unknown observation preserves latch',
                           source_calibration=str(source), source_sha256=sha(source))
    if stage_source is not None:
        ct['r7_calls']['stage_source'] = str(stage_source)
    ct['cost_model'] += '; exact product with the high-deviation entry latch; positive stage weights'
    ct['cadence_nodes'] = {'stall': sum(len(t['nodes']) for t in tilted)}
    ct_dir = out / 'CT'
    write_json(ct_dir / 'calibration.json', ct)
    write_json(ct_dir / 'cost_replay.json', {**replay, 'modes': {'stall': tilted}})
    write_json(out / 'stage_features.json', features)
    report = dict(cell=cell, rho=rho, episodes=len(trees), anchors=cal['calibration_anchors'],
                  CU=cal['solutions']['stall']['uniform'][format(rho, '.12g')], CT=solution,
                  r6_solution_bit_identical=True, stage_calibration=table.calibration,
                  event_occupancy=table.event_occupancy, deviation_p75=table.deviation_p75,
                  deviation_occupancy=table.deviation_occupancy,
                  r6_dag_nodes=sum(len(t['nodes']) for t in trees),
                  ct_dag_nodes=sum(len(t['nodes']) for t in tilted),
                  stage_fingerprint=table.fingerprint, stage_sha256=sha(table_path))
    if stage_source is not None:
        report.update(stage_source=str(stage_source), stage_source_sha256=sha(stage_source),
                      deployed_library=base.cand_name, kref=base.kref,
                      source_calibration=str(source), source_sha256=sha(source))
    write_json(out / 'feasibility.json', report)
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell', choices=ALL_CELLS, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--rho', type=float, default=None,
                        help='default: .30 for sparse cells; .18 for dense cells')
    args = parser.parse_args()
    fit_cell(args.cell, args.out or SCRATCH / 'cal' / args.cell, args.rho)


if __name__ == '__main__':
    main()
