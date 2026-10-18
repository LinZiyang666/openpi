"""Optional CPU-only file/hash proof; never invokes a launcher or native model.

This is a diagnostic command, not an additional campaign admission gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path

from .audit_counts import HERE, REPO, read
from .build import write


def audit_paths(pins, required, releases):
    """Self-contained stdlib code also executed through tether exec."""
    import hashlib
    import json
    from pathlib import Path
    def sha(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
        return digest.hexdigest()
    manifests = []
    for release in releases:
        path = Path(release['manifest'])
        doc = json.loads(path.read_text())
        raw = json.dumps(doc, sort_keys=True, separators=(',', ':'))+release.get('newline', '\n')
        actual = hashlib.sha256(raw.encode()).hexdigest()
        expected = release.get('identity')
        manifests.append(dict(path=str(path), identity=actual, expected=expected,
                              equal=expected is None or actual == expected))
        pins.update({str(Path(release['root'])/name): value for name, value in doc['files'].items()})
        required.append(str(path))
    files = []
    for name in sorted(set(pins) | set(required)):
        path = Path(name)
        exists = path.is_file()
        actual = sha(path) if exists and name in pins else None
        files.append(dict(path=name, present=exists, sha256=actual, expected=pins.get(name),
                          equal=exists and (name not in pins or actual == pins[name])))
    return dict(status='PASS' if all(x['equal'] for x in files+manifests) else 'FAIL',
                files=files, manifests=manifests)


def remote_audit(node, python, pins, required=(), releases=()):
    from exp.r17_integration.ops.formal_eval import remote
    code = inspect.getsource(audit_paths)+'\nimport json; print(json.dumps(audit_paths(*'+repr(
        (pins, list(required), list(releases)))+')))'
    result = json.loads(remote(node, [python, '-c', code], seconds=600, replay_safe=True))
    print(json.dumps(dict(node=node, status=result['status'], files=len(result['files']))), flush=True)
    return dict(node=node, **result)


def release(root, manifest, identity=None, newline='\n', data_root=None):
    return dict(root=str(data_root or root), manifest=str(Path(root)/manifest), identity=identity, newline=newline)


def r17(root):
    from exp.r17_integration.ops.formal_prestage import validate_prepared
    from exp.r17_integration.ops.immutable_release import identity, sha
    from exp.r17_integration.ops.formal_eval import host_python
    c, deployment = validate_prepared(root)
    seal = read(root/'PREPARED.json')
    local_pins = {str(root/name): pin for name, pin in seal['files'].items()}
    local_pins.update({str(Path(c['runtime_from'])/name):pin for name,pin in c['runtime_source_pins'].items()})
    local_pins.update(c.get('cache_control_proof', {}).get('source_pins', {}))
    if c.get('local_worker_exception'):
        local_pins[str(root/'LOCAL_WORKER_ORCHESTRATION.json')] = sha(
            Path(c['filter_failed_source'])/'LOCAL_WORKER_ORCHESTRATION.json')
    local_releases = [release(c['coordinator_runtime'], 'RUNTIME_RELEASE.json', c['runtime_identity'])]
    if c.get('driver_runtime'):
        local_releases.append(release(c['driver_runtime'], 'RUNTIME_RELEASE.json', c['orchestration']['driver_runtime_identity']))
    results = [remote_audit('local', 'python3', local_pins, releases=local_releases)]
    for node, item in deployment.items():
        pins = {item['stage']+'/benchmark.json': sha(Path(c['local_assets'])/'benchmark.json')}
        for name in ('campaign.json', 'launch_campaign.json'):
            pins[item['stage']+'/'+name] = sha(root/(node+'.json'))
        for key in ('benchmark_transfer', 'campaign_transfer'):
            if item.get(key):
                pins[item[key]['path']] = item[key]['sha256']
        if item.get('private_config'):
            config = item['private_config']
            pins[config['path']+'/config.yaml'] = config['sha256']
        installed = {k:v for k,v in item['installed'].items() if k in ('schema', 'python', 'files', 'digest')}
        releases = [release(item['payload'], 'PAYLOAD.json', identity(installed))]
        dependency = item.get('dependencies')
        if dependency:
            releases.append(release(dependency['private_dependencies'], 'DEPENDENCIES.json',
                                    dependency['content_identity'], newline=''))
            if dependency.get('asset_release'):
                releases.append(release(dependency['asset_release'], 'ASSETS.json', data_root=dependency['assets']))
        worker = read(root/(node+'.json'))
        results.append(remote_audit(node, item['python'], pins,
                                    [item['python'], worker['worker_benchmark'], item['campaign_path']], releases))
    if c['server_node'] == 'h100':
        pins = {str(Path(c['assets'])/p.relative_to(root/'assets')): sha(p)
                for p in (root/'assets').rglob('*') if p.is_file() and p.name != 'TRANSFER_SHA256.json'}
        directory = '/tmp/r17_formal/'+c['run_name']+'/server'
        for name in ('campaign.json', 'launch_campaign.json'):
            pins[directory+'/'+name] = sha(root/'campaign.json')
        results.append(remote_audit('h100', host_python(c), pins, [host_python(c)],
                                    [release(c['runtime'], 'RUNTIME_RELEASE.json', c['runtime_identity'])]))
    return results


def r10(root):
    from exp.offline_search.closed_loop.ops.h100.assets import BASE
    from exp.offline_search.closed_loop.ops.h100.fleet import island
    from exp.offline_search.closed_loop.ops.h100.node import sha
    plan = read(root/'h100_sync/synced.json')
    if plan['arms_sha256'] != sha(root/'arms.json'):
        raise ValueError('arms changed since sync')
    pins = {str(BASE/item['rel']):item['sha256'] for item in plan['files']}
    required = [str(BASE/'node.py')]
    required.extend(str(BASE/'openpi/exp/offline_search/closed_loop'/('serve_'+model+'.py'))
                    for model in {row['model'] for row in plan['arms']})
    required.append(str(BASE/'openpi/exp/offline_search/rounds/r10/method.py'))
    results = [remote_audit('h100', 'python3', pins, required)]
    node = plan['worker_host']
    originals = {row['arm']:row for row in read(root/'arms.json')}
    pins = {}
    for row in plan['arms']:
        original = originals[row['arm']]
        config = next(item for item in plan['files'] if item['original'] == original['yaml'])
        pins[str(island(node)/row['remote_yaml'])] = config['sha256']
        pins[str(island(node)/row['remote_matrix'])] = sha(original['matrix'])
    required = [str(island(node)/('os_cl/'+name)) for name in
                ('node.py', 'run_arm.sh', 'run_gtp_subset.py', 'count.py', 'purge_exc.py', 'libero/config.yaml')]
    required += ['/scratch/zixuans8/openpi/.venv/bin/python', '/scratch/zixuans8/libero_sim/bin/python']
    for suite in {row['suite'] for row in plan['arms']}:
        required.append(str(island(node)/f'exp/ablation_study/cache_size/config/apool_{suite}.yaml'))
        for path in (REPO/f'exp/common/data/db_init/libero/{suite}_apool').glob('*.init'):
            pins[str(island(node)/path.relative_to(REPO))] = sha(path)
    results.append(remote_audit(node, 'python3', pins, required))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, default=HERE/'STAGING_PROOF.json')
    args = p.parse_args()
    rows = []
    for root in args.run:
        root = root.resolve()
        nodes = r17(root) if (root/'campaign.json').exists() else r10(root)
        rows.append(dict(run=str(root), status='PASS' if all(n['status']=='PASS' for n in nodes) else 'FAIL', nodes=nodes))
        # Preserve completed evidence even if a later diagnostic call fails.
        write(args.output, dict(cpu_only=True, checked_at=datetime.now(timezone.utc).isoformat(),
                                runs=rows, status='PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL'))
    if any(r['status'] != 'PASS' for r in rows):
        raise SystemExit('staged file/hash mismatch; see '+str(args.output))


if __name__ == '__main__':
    main()
