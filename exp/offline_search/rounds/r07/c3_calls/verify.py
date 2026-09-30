"""Check retained replay evidence, loader gates and all 16 current arm prefits."""
import contextlib
import io
import json
from pathlib import Path
import pickle
from unittest.mock import patch
import unittest

import numpy as np

from exp.offline_search.closed_loop.ops.emit_arms import main as emit_arms
from .common import CELLS, HERE, SCRATCH, write_json, sha
from .package import specs
from .replay import controller
from .methods import CallController
from .integration import check as check_integration


def loader_gates(cell):
    path = SCRATCH / 'cal' / cell / 'CT' / 'calibration.json'
    original = json.loads(path.read_text())
    read_text = Path.read_text
    cases = [dict(status='DRYRUN_TEST_INITS'), dict(ambiguous_rule='old_rule'),
             dict(r7_calls={**original['r7_calls'], 'stage_sha256': '0' * 64}),
             dict(r7_calls={**original['r7_calls'], 'tilt': False})]
    count = 0
    for changes in cases:
        text = json.dumps({**original, **changes})
        def altered(self, *args, **kwargs):
            return text if self == path else read_text(self, *args, **kwargs)
        with patch.object(Path, 'read_text', altered):
            try:
                controller(cell, CallController, tilt=True)
            except ValueError:
                count += 1
            else:
                raise AssertionError('loader accepted a mismatched artifact')
    return count


def main():
    integration = check_integration()
    suite = unittest.defaultTestLoader.loadTestsFromName('exp.offline_search.rounds.r07.c3_calls.test_calls')
    with (SCRATCH / 'unit_final.log').open('w') as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    assert result.wasSuccessful()
    total_decisions = total_anchors = total_episodes = composed = cost_anchors = 0
    plugin_decisions = plugin_pairs = prefits = gates = 0
    fixed_paths = monte_paths = statistic_comparisons = 0
    feasibility, plugins, prefits_manifest = [], [], []
    max_se = 0.
    for cell in CELLS:
        cal_root = SCRATCH / 'cal' / cell
        feasibility.append(json.loads((cal_root / 'feasibility.json').read_text()))
        identity = json.loads((SCRATCH / 'replays' / f'identity_{cell}.json').read_text())
        assert identity['status'] == 'PASS'
        for counts in identity['counts'].values():
            total_decisions += counts['decisions']; total_anchors += counts['anchors']; total_episodes += counts['episodes']
        composed += identity['composed_E0_decisions']
        cost = json.loads((SCRATCH / 'replays' / f'cost_paths_{cell}.json').read_text())
        for variant, item in cost['variants'].items():
            assert item['path_agreement'] and abs(item['expected']['IR'] - .30) < 1e-14
            cost_anchors += item['deployed_fixed_proposal_path_counts']['anchors']
            episode_count = item['deployed_fixed_proposal_path_counts']['episodes']
            fixed_paths += episode_count
            monte_paths += episode_count * item['seeds']
            statistic_comparisons += len(item['empirical'])
            max_se = max(max_se, item['max_standard_errors'])
        plugin_root = SCRATCH / 'plugin_checked'
        with np.load(plugin_root / f'{cell}_R6' / 'served.npz') as old, \
                np.load(plugin_root / f'{cell}_CU' / 'served.npz') as new:
            assert old.files == new.files
            for name in old.files:
                assert old[name].dtype == new[name].dtype and old[name].shape == new[name].shape
                assert old[name].tobytes() == new[name].tobytes(), (cell, name)
            plugin_pairs += len(old['action'])
        for variant in ('R6', 'CU', 'CT'):
            report = json.loads((plugin_root / f'{cell}_{variant}' / 'report.json').read_text())
            assert report['status'] == 'PASS'
            plugins.append(report)
            plugin_decisions += report['decisions']
        gates += loader_gates(cell)
    for phase in ('profile', 'eval500'):
        assert json.loads((HERE / f'arms_{phase}.json').read_text()) == specs(phase)
        rendered = specs(phase, str(SCRATCH))
        assert json.loads((SCRATCH / 'specs_filled' / f'arms_{phase}.json').read_text()) == rendered
        for row in rendered:
            path = SCRATCH / 'prefits_checked' / (row['name'] + '.pkl')
            with path.open('rb') as f:
                blob = pickle.load(f)
            assert blob['kwargs'] == row['kwargs'] and blob['spec'] == row['method']
            assert blob['provenance']['source_sha256'] == sha(blob['provenance']['method_source'])
            method = blob['method']
            cal = json.loads(Path(row['kwargs']['calibration_path']).read_text())
            assert method.lambda_ == cal['solutions']['stall']['uniform']['0.3']['parameter']
            assert method.tilt == row['kwargs']['tilt']
            prefits_manifest.append(dict(name=row['name'], artifact=str(path), sha256=sha(path),
                                         bytes=path.stat().st_size, lambda_=method.lambda_))
            prefits += 1
        # Test the actual emitter under owned scratch; creates configs only.
        with contextlib.redirect_stdout(io.StringIO()):
            emit_arms(['--run-root', str(SCRATCH / 'emitted' / phase), '--spec', str(HERE / f'arms_{phase}.json')])
        arms = json.loads((SCRATCH / 'emitted' / phase / 'arms.json').read_text())
        assert len(arms) == 8 and all(r['full_model'] for r in arms)
    report = dict(status='PASS', unit_tests=result.testsRun, unit_failures=0,
        identity_decision_pairs=total_decisions, identity_fresh_anchor_pairs=total_anchors,
        identity_episode_passes=total_episodes, composed_E0_decision_pairs=composed,
        fixed_proposal_DAG_anchor_agreements=cost_anchors, fixed_proposal_episode_paths=fixed_paths,
        monte_carlo_seeds_per_arm=1000, monte_carlo_episode_paths=monte_paths,
        monte_carlo_statistics_compared=statistic_comparisons, max_standard_errors=max_se,
        plugin_replays=len(plugins), plugin_decisions=plugin_decisions, plugin_R6_CU_bit_identical_pairs=plugin_pairs,
        prefits_current_names=prefits, emitted_profile_arms=8, emitted_eval500_arms=8,
        loader_rejections=gates, calibration_episode_count=sum(f['episodes'] for f in feasibility),
        calibration_anchors=sum(f['anchors'] for f in feasibility),
        C4_integration=integration,
        feasibility=feasibility, plugin_reports=plugins,
        limits='Fixed observations/proposals and fake CPU policy; no new closed-loop evaluation or SR measurement.')
    write_json(HERE / 'verification.json', report)
    write_json(HERE / 'prefit_manifest.json', prefits_manifest)
    write_json(HERE / 'calibration_feasibility.json', feasibility)
    write_json(SCRATCH / 'verification.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('feasibility', 'plugin_reports')}), flush=True)


if __name__ == '__main__':
    main()
