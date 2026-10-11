"""A local qualification refresh must never mutate either H100 lane."""
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('runtime_only', [False, True])
def test_local_only_preserves_h100_and_timan108(tmp_path, monkeypatch, runtime_only):
    from . import deploy_cpu
    from exp.r17_integration.ops import build_client_bundle, deploy
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    q = tmp_path / 'exp/r17_integration/qualification'
    q.mkdir(parents=True)
    protected = dict(destination='immutable_timan108_payload', probe={'status': 'PASS'})
    previous = dict(nodes={'timan107': {'destination': 'old'}, 'timan108': protected})
    (q / 'worker_deployment.json').write_text(json.dumps(previous))
    (tmp_path / 'payload.py').write_text('CPU_ONLY = True\n')
    monkeypatch.setattr(build_client_bundle, 'worker_payload_files', lambda *a, **k: ['payload.py'])
    real_bundle = build_client_bundle.build_client_bundle
    monkeypatch.setattr(build_client_bundle, 'build_client_bundle',
        lambda root, files, output: real_bundle(root, ['payload.py'], output))
    def archive(root, output):
        with tarfile.open(output, 'w') as tar:
            tar.add(root / 'payload.py', arcname='payload.py')
        return {root / 'payload.py'}
    monkeypatch.setattr(deploy_cpu, 'runtime_archive', archive)
    commands = []
    def transfer(argv, **kwargs):
        commands.append(argv)
        assert argv[3].startswith('timan107:/tmp/')
    monkeypatch.setattr(deploy, 'tether_transfer', transfer)
    def execute(argv, **kwargs):
        commands.append(argv)
        assert argv[5] == 'timan107'
        output = '{}'
        if '--import-probe' in argv:
            output = json.dumps(dict(status='PASS', packages={'metaworld': '3.0.0'}))
        return SimpleNamespace(returncode=0, stdout=output, stderr='')
    monkeypatch.setattr(deploy_cpu.subprocess, 'run', execute)
    out = tmp_path / 'release'
    args = ['deploy_cpu', '--output', str(out), '--local-only']
    if runtime_only:
        args.append('--runtime-only')
    monkeypatch.setattr('sys.argv', args)
    deploy_cpu.main()
    current = json.loads((q / 'worker_deployment.json').read_text())
    assert current['nodes']['timan108'] == protected
    if runtime_only:
        assert current == previous and not commands
    else:
        assert current['nodes']['timan107']['destination'].startswith('/scratch/')
        assert commands
        assert list(json.loads((out / 'WORKER_DEPLOYMENT.json').read_text())['nodes']) == ['timan107']
    runtime = json.loads((out / 'RUNTIME_DEPLOYMENT.json').read_text())
    assert runtime['installed']['host'] == 'local'
    assert runtime['installed']['remote_runtime_untouched'] is True
    assert (q / 'runtime_code_snapshot.json').is_file()
