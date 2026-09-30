"""Audit §9.3 dense artifacts, recorded parity and final-name CPU prefits.

`prepare` only copies calibrated stalls and renders owned scratch specs.
`check` emits configuration files only; neither command starts a closed loop.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import pickle
import shutil
from pathlib import Path

from exp.offline_search.closed_loop.ops.emit_arms import main as emit_arms
from .common import DENSE_CELLS, HERE, SCRATCH, sha, stall_path, write_json
from .package import emit_dense, specs
from .verify import loader_gates


def prepare():
    for cell in DENSE_CELLS:
        source, dest = stall_path(cell), SCRATCH / 'stall' / cell
        if not dest.exists():
            shutil.copytree(source, dest)
        for path in source.rglob('*'):
            if path.is_file():
                assert sha(path) == sha(dest / path.relative_to(source))
    emit_dense(SCRATCH / 'specs_dense_filled', str(SCRATCH))


def check():
    baseline = json.loads((SCRATCH / 'dense_sparse_baseline.json').read_text())
    for path, digest in baseline.items():
        assert sha(path) == digest, 'sparse artifact changed: ' + path
    sparse = json.loads((HERE / 'verification.json').read_text())
    assert sparse['status'] == 'PASS' and sparse['unit_failures'] == 0
    counts = dict(identity_decision_pairs=0, identity_fresh_anchor_pairs=0,
                  identity_episode_passes=0, composed_E0_decision_pairs=0,
                  fixed_proposal_DAG_anchor_agreements=0, fixed_proposal_episode_paths=0,
                  monte_carlo_episode_paths=0, monte_carlo_statistics_compared=0,
                  prefits=0, loader_rejections=0)
    feasibility, prefits, identities, costs = [], [], [], []
    max_se = 0.
    for cell in DENSE_CELLS:
        root = SCRATCH / 'cal' / cell
        report = json.loads((root / 'feasibility.json').read_text())
        assert report['rho'] == .18 and report['r6_solution_bit_identical']
        assert report['kref'] == 8
        assert report['deployed_library'] == ('bpool_cs' if cell.startswith('pi05_') else 'bpool_all')
        assert sha(report['source_calibration']) == report['source_sha256']
        assert sha(report['stage_source']) == sha(root / 'stages.pkl') == report['stage_source_sha256']
        old = json.loads(Path(report['source_calibration']).read_text())
        cu = json.loads((root / 'CU' / 'calibration.json').read_text())
        for key in ('status', 'cell', 'intercept', 'slope', 'base_source', 'c1',
                    'stall_fingerprint', 'controller_version', 'cooldown_scope', 'ambiguous_rule',
                    'baseline_E', 'tables', 'catalog_paths', 'reset_attestation',
                    'bval_manifest_sha256', 'coefficient_weighting', 'cost_model'):
            assert cu[key] == old[key], (cell, key)
        # The recorded identity harness uses this new calibration for both
        # controllers. Establish exact equivalence with retained R6 parameters
        # for *both* tested placements, not just equality between new classes.
        for placement in ('uniform', 'R'):
            assert cu['solutions']['stall'][placement]['0.18'] == old['solutions']['stall'][placement]['0.18']
        for variant in ('CU', 'CT'):
            item = report[variant]
            assert item['feasible'] and item['floor'] < .18 < item['ceiling']
            assert abs(item['predicted_IR'] - .18) < 1e-14
        feasibility.append(report)
        ident = json.loads((SCRATCH / 'replays' / f'identity_{cell}.json').read_text())
        assert ident['status'] == 'PASS'
        for key in ('differing_actions', 'differing_verdicts', 'differing_vision_flags', 'differing_result_fields'):
            assert ident[key] == 0
        for entry in ident['counts'].values():
            counts['identity_decision_pairs'] += entry['decisions']
            counts['identity_fresh_anchor_pairs'] += entry['anchors']
            counts['identity_episode_passes'] += entry['episodes']
        counts['composed_E0_decision_pairs'] += ident['composed_E0_decisions']
        identities.append(ident)
        cost = json.loads((SCRATCH / 'replays' / f'cost_paths_{cell}.json').read_text())
        assert cost['status'] == 'PASS'
        for entry in cost['variants'].values():
            assert entry['path_agreement'] and abs(entry['expected']['IR'] - .18) < 1e-14
            assert entry['seeds'] == 1000 and entry['max_standard_errors'] < 5
            n = entry['deployed_fixed_proposal_path_counts']
            counts['fixed_proposal_DAG_anchor_agreements'] += n['anchors']
            counts['fixed_proposal_episode_paths'] += n['episodes']
            counts['monte_carlo_episode_paths'] += n['episodes'] * entry['seeds']
            counts['monte_carlo_statistics_compared'] += len(entry['empirical'])
            max_se = max(max_se, entry['max_standard_errors'])
        costs.append(cost)
        counts['loader_rejections'] += loader_gates(cell)
    for phase in ('profile', 'eval500'):
        assert json.loads((HERE / f'arms_dense_{phase}.json').read_text()) == specs(phase, dense=True)
        rendered = specs(phase, str(SCRATCH), dense=True)
        path = SCRATCH / 'specs_dense_filled' / f'arms_dense_{phase}.json'
        assert json.loads(path.read_text()) == rendered
        for row in rendered:
            assert row['kwargs']['random_seed'] == 26092903 and row['kwargs']['rho'] == .18
            cell = row['kwargs']['randomization_key'].removeprefix('R6-C-v2/')
            assert cell in DENSE_CELLS
            assert ('--os-log-inputs' in row['plugin_args']) == (phase == 'profile')
            assert ('--os-log-r4' in row['plugin_args']) == (phase == 'profile')
            artifact = SCRATCH / 'prefits_dense_checked' / (row['name'] + '.pkl')
            with artifact.open('rb') as f:
                blob = pickle.load(f)
            assert blob['kwargs'] == row['kwargs'] and blob['spec'] == row['method']
            assert blob['provenance']['source_sha256'] == sha(blob['provenance']['method_source'])
            method = blob['method']
            variant = 'CT' if row['kwargs']['tilt'] else 'CU'
            report = next(f for f in feasibility if f['cell'] == cell)
            assert method.tilt == row['kwargs']['tilt'] and method.lambda_ == report[variant]['parameter']
            assert method.base.cand_name == report['deployed_library'] and method.base.kref == 8
            prefits.append(dict(name=row['name'], artifact=str(artifact), sha256=sha(artifact),
                                bytes=artifact.stat().st_size, lambda_=method.lambda_))
            counts['prefits'] += 1
        with contextlib.redirect_stdout(io.StringIO()):
            emit_arms(['--run-root', str(SCRATCH / 'emitted_dense' / phase), '--spec', str(path)])
        emitted = json.loads((SCRATCH / 'emitted_dense' / phase / 'arms.json').read_text())
        assert len(emitted) == 8 and all(r['full_model'] for r in emitted)
    result = dict(status='PASS', **counts, unit_tests=sparse['unit_tests'], unit_failures=0,
                  unchanged_sparse_files=len(baseline), max_standard_errors=max_se,
                  emitted_profile_arms=8, emitted_eval500_arms=8,
                  calibration_episodes=sum(r['episodes'] for r in feasibility),
                  calibration_anchors=sum(r['anchors'] for r in feasibility),
                  feasibility=feasibility, identities=identities, cost_replays=costs,
                  sparse_verification=str(HERE / 'verification.json'),
                  limitation='Fixed archived inputs/proposals and same-observation CPU policy chunks; live profile/eval are coordinator work.')
    write_json(HERE / 'verification_dense.json', result)
    write_json(HERE / 'prefit_manifest_dense.json', prefits)
    write_json(HERE / 'calibration_feasibility_dense.json', feasibility)
    print(json.dumps({k: v for k, v in result.items() if k not in ('feasibility', 'identities', 'cost_replays')}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'check'))
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else check)()


if __name__ == '__main__':
    main()
