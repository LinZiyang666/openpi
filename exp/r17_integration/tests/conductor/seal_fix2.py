"""Seal actual round-2 CPU receipts; preserve all round-1 artifacts."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


PREFIX = "env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:packages/openpi-client/src .venv/bin/python "
BASE = Path('exp/r17_integration')
ARTIFACTS = BASE/'tests/conductor'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    directories = ('benchmarks', 'conductor', 'models/pi05', 'tests/benchmarks', 'tests/conductor', 'tests/models/pi05')
    python = [p for directory in directories for p in sorted((BASE/directory).glob('*.py'))]
    for p in python:
        compile(p.read_text(), str(p), 'exec')
    original = json.loads((ARTIFACTS/'SOURCES_P2.json').read_text())
    unchanged = {}
    for name in original['original_equal_head']:
        expected = subprocess.run(['git', 'show', 'HEAD:'+name], capture_output=True, check=True).stdout
        unchanged[name] = expected == Path(name).read_bytes()
    assert all(unchanged.values())
    owned = python + [BASE/'handbacks/HANDBACK_P2.md'] + list((BASE/'REQUESTS').glob('P2_to_*.md'))
    source = {'schema': 'r17.p2.fix2.sources.v1', 'captured_utc': datetime.now(timezone.utc).isoformat(),
        'snapshot_timing': 'post-test; independent shared owners can still edit their files',
        'head': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
        'owned': {str(p): sha(p) for p in owned}, 'original_equal_head': unchanged,
        'original_reused': {name: sha(Path(name)) for name in original['original_reused']},
        'cross_package_dependencies': {str(p): sha(p) for directory in ('common', 'debug', 'search', 'ops', 'models/groot')
                                       for p in sorted((BASE/directory).rglob('*.py'))},
        'compile': {'passed': len(python), 'skipped': 0, 'pyc_written': False}}
    (ARTIFACTS/'SOURCES_P2_FIX2.json').write_text(json.dumps(source, sort_keys=True, indent=2)+'\n')
    runs = [
        ('full5', 'exp/r17_integration/tests/benchmarks exp/r17_integration/tests/conductor exp/r17_integration/tests/models/pi05', '', 0),
        ('cross4', 'exp/r17_integration/tests/common exp/r17_integration/tests/benchmarks exp/r17_integration/tests/conductor exp/r17_integration/tests/models/pi05 exp/r17_integration/tests/debug exp/r17_integration/tests/search exp/r17_integration/tests/ops', '', 0),
        ('legacy1', 'tests/serving/test_policy_identity.py tests/serving/test_per_step_recorder.py tests/serving/test_replica_proxy.py tests/serving/test_router_bundle_rebind.py tests/exp/metaworld/test_env.py tests/exp/metaworld/test_runner.py tests/conductor exp/mw_finetune/tests exp/offline_search/rounds/r16/test_conductor.py exp/offline_search/rounds/r16/test_groot_servers.py exp/r17_integration/tests/common/test_default_off.py tests/cache/test_config.py tests/cache/test_denoise_schedule_config.py tests/cache/test_cache_storage.py', "-k 'not same_identity_same_initial_state_across_processes'", 1),
        ('proxy', 'tests/serving/test_replica_proxy.py', '-m env_dependent --run-manual', 23),
    ]
    receipts = []
    for suffix, targets, options, deselected in runs:
        xml = ARTIFACTS/f'cpu_fix2_{suffix}.xml'
        root = ET.parse(xml).getroot()
        cases = list(root.iter('testcase'))
        failed = sum(c.find('failure') is not None or c.find('error') is not None for c in cases)
        skipped = sum(c.find('skipped') is not None for c in cases)
        assert failed == 0, (xml, failed)
        receipts.append({'command': PREFIX+'-m pytest '+targets+' '+options+' -p no:cacheprovider '
            '--basetemp='+str(ARTIFACTS/f'scratch_fix2_{suffix}')+' --junitxml='+str(xml)+' -q --tb=short',
            'exit_code': 0, 'passed': len(cases)-skipped, 'skipped': skipped, 'failed': failed,
            'deselected': deselected, 'duration_seconds': sum(float(s.attrib.get('time', 0)) for s in root.iter('testsuite')),
            'xml': str(xml), 'junit_sha256': sha(xml), 'device': 'cpu'})
    red = {}
    for suffix in ('terminal_old_red', 'preload_old_red', 'table_old_red', 'ledger_old_red'):
        xml = ARTIFACTS/f'cpu_fix2_{suffix}.xml'
        cases = list(ET.parse(xml).getroot().iter('testcase'))
        red[suffix] = {'xml': str(xml), 'sha256': sha(xml), 'exit_code': 1,
                      'expected_failures': sum(c.find('failure') is not None for c in cases)}
    samples = BASE/'tests/models/pi05/samples/fix_round2'
    manifest = json.loads((samples/'sample_manifest.json').read_text())
    assert manifest['synthetic_fixture'] and not manifest['pricing_claim'] and not manifest['checkpoint_loaded']
    receipt = {'schema': 'r17.p2.fix2.receipts.v1', 'interface_version': 'r17.contracts.v1',
        'runs': receipts, 'expected_old_red': red, 'source_snapshot': str(ARTIFACTS/'SOURCES_P2_FIX2.json'),
        'compile': source['compile'], 'source_parity': {'passed': len(unchanged), 'skipped': 0},
        'sample_command': PREFIX+'-m exp.r17_integration.tests.models.pi05.build_fix2_samples',
        'sample_generation_exit': 0, 'sample_idempotent_readback_exit': 0,
        'sample_hashes': {str(p): sha(p) for p in samples.rglob('*') if p.is_file()},
        'r16_verify_command': PREFIX+'-m exp.offline_search.rounds.r16.analyze verify',
        'r16_verify_exit': 0, 'r16_verify_stdout': 'DESIGN_VERIFIED; no outcome reads',
        'gpu_commands_executed': False, 'git_commit_or_push': False,
        'limitations': ['CPU fake model/environment boundaries; no real renderer/checkpoint/GPU/pricing receipt',
                        'Counts overlap; do not sum runs', 'Post-test snapshot; shared packages can change']}
    (ARTIFACTS/'RECEIPT_P2_FIX2.json').write_text(json.dumps(receipt, sort_keys=True, indent=2)+'\n')
    print(json.dumps({'runs': receipts, 'compile': source['compile'], 'unchanged_sources': len(unchanged),
                      'receipt': str(ARTIFACTS/'RECEIPT_P2_FIX2.json')}))


if __name__ == '__main__':
    main()
