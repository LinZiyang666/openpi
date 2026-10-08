"""Read-only final integrity checks; writes only R14 release evidence."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import stat
import xml.etree.ElementTree as ET

from .common import *


def regular_tree(root):
    for p in Path(root).rglob('*'):
        mode = p.lstat().st_mode
        assert stat.S_ISDIR(mode) or stat.S_ISREG(mode), str(p)


def main():
    before = json.loads((HERE / 'protected_before.json').read_text())
    shared = json.loads((HERE / 'shared_edits.json').read_text())
    allowed = {r['path']: r for r in shared}
    protected = 0
    for path, digest in before.items():
        if path in allowed:
            assert digest == allowed[path]['before_sha256']
            assert sha(path) == allowed[path]['after_sha256']
            assert sha(HERE / 'before' / Path(path).name) == digest
        else:
            assert sha(path) == digest, path
            protected += 1

    frozen = json.loads((ASTRA / 'freeze.json').read_text())['files']
    for name, digest in frozen.items():
        assert sha(ASTRA / name) == digest, name
    code = json.loads((HERE / 'implementation_sources.json').read_text())
    for name, digest in code.items():
        assert sha(REPO / name) == digest, name

    catalog = json.loads((HERE / 'fits.json').read_text())
    assert len(catalog) == 104
    assert len({r['config_key'] for r in catalog}) == 104
    cached = {}

    def checked(path, digest):
        path = str(path)
        if path not in cached:
            cached[path] = sha(path)
        assert cached[path] == digest, path

    cells = {}
    for rec in catalog:
        checked(rec['path'], rec['sha256'])
        assert Path(rec['path']).stat().st_size == rec['bytes']
        assert rec['implementation_sources_sha256'] == code
        m = rec.get('manifest')
        if m is None:
            assert rec['reused_R11']
            continue
        assert m['implementation_sources_sha256'] == code
        for name, digest in m['frozen_sources_sha256'].items():
            assert digest == frozen[name]
        for path, digest in m['sources_sha256'].items():
            checked(path, digest)
        if m['variant'] in ('guard_null', 'task_periodic'):
            continue
        assert m['full_training_rows'] and m['selected_library_PCA_reused']
        assert m['action_sigma_byte_preserved'] and m['continuity_exact_frozen']
        assert sum(x['rows'] for x in m['head_counts'].values()) == m['rows']
        changed = m['switches']['scope'] == 'pool' or m['switches']['metric'] == 'pool'
        shared_head = m['switches']['head'] == 'pool'
        if changed or shared_head:
            for head in m['head_counts'].values():
                assert head['features'] == (207 if shared_head else 217)
                assert head['exclude_query_episode']
                assert head['query_regime'] == 'early_step0_stale_later'
        folds = m['distance_folds']
        assert len(folds) == 5
        heldout = set()
        for f in folds:
            tr, va = set(f['train_episodes']), set(f['heldout_episodes'])
            assert not tr & va and not heldout & va
            heldout |= va
            assert len(tr | va) == m['episodes']
            if changed:
                assert f['rows'] + f['heldout_rows'] == m['rows']
                assert f['metric_scope'] == m['switches']['metric']
                assert f['donor_scope'] == m['switches']['scope']
        assert len(heldout) == m['episodes']
        if m['variant'] == 'base':
            cells[m['cell']] = dict(rows=m['rows'], episodes=m['episodes'],
                                   continuity=m['pooled_continuity'])

    roots = {}
    for filename, pool in (('pilot_roots.json', 'B'), ('test_a_roots.json', 'A')):
        records = json.loads((HERE / filename).read_text())
        roots[pool] = records
        for r in records:
            root = Path(r['root'])
            assert r['init_pool'] == pool and r['launches'] == 0 and r['fleet_exclusive']
            assert not any((root / x).exists() for x in ('runs', 'state', 'h100_sync'))
            assert json.loads((root / 'ROOT.json').read_text()) == r
            for name, digest in r['definitions_sha256'].items():
                checked(root / name, digest)
            for path, digest in r['fit_sha256'].items():
                checked(path, digest)
            for path, digest in r['selections_sha256'].items():
                checked(path, digest)
            regular_tree(root)
            if pool == 'A':
                assert r['staged'] and not Path(r['final_root']).exists()
    assert len(roots['B']) == 6 and len(roots['A']) == 8
    for filename, grid in (('queue_pilot.txt', 'pilot_grid.csv'), ('queue_test_a.txt', 'test_grid.csv')):
        arms = []
        for line in (HERE / filename).read_text().splitlines():
            host, rest = line.split(' ', 1)
            root, names = rest.split('|', 1)
            assert host in ('@local', '@h100') and root.startswith('/')
            names = names.split()
            assert names and (('_groot_' in Path(root).name) == (host == '@local'))
            arms.extend(names)
        assert arms == [r['arm'] for r in csv.DictReader((ASTRA / grid).open())]

    counts = {}
    # JUnit materializes the 25 passing subtests as additional test cases.
    for name, expected in (('mechanics', 78), ('r13', 71), ('regression', 321)):
        suites = ET.parse(HERE / (name + '_tests.xml')).getroot().iter('testsuite')
        counts[name] = dict(tests=0, failures=0, errors=0, skipped=0)
        for suite in suites:
            for field in counts[name]:
                counts[name][field] += int(suite.attrib.get(field, 0))
        assert counts[name] == dict(tests=expected, failures=0, errors=0, skipped=0)
    assert '296 passed' in (HERE / 'regression_tests.log').read_text()
    assert '25 subtests passed' in (HERE / 'regression_tests.log').read_text()
    counts['regression'].update(primary_tests=296, subtests=25)
    evidence = {}
    for name in ('mechanics', 'allocation_audit', 'pilot_identity_audit', 'runtime_closure', 'plans'):
        evidence[name] = json.loads((HERE / (name + '.json')).read_text())
        assert evidence[name]['PASS'], name
    closure = evidence['runtime_closure']
    for path, digest in closure['source_files'].items():
        checked(REPO / path, digest)
    for path, rec in closure['payloads'].items():
        checked(path, rec['sha256'])
    for host in ('H100', 'TIMAN107', 'TIMAN108'):
        assert (HERE / (host + '_SOURCES.sha256')).read_bytes() == (HERE / 'RUNTIME_SOURCES.sha256').read_bytes()
    assert len(evidence['allocation_audit']['records']) == 16
    assert len(evidence['pilot_identity_audit']['records']) == 6
    assert len(evidence['plans']['records']) == 14
    for rec in evidence['plans']['records']:
        checked(rec['path'], rec['sha256'])
        plan = json.loads(Path(rec['path']).read_text())
        assert len(plan['files']) == rec['files']
        assert sum(x['size'] for x in plan['files']) == rec['bytes']
        for f in plan['files']:
            checked(f['source'], f['sha256'])
            checked(f['original'], f['source_sha256'])
    regular_tree(ROUND)
    # Refresh build-tool hashes only; the numerical/runtime source bindings
    # above remain immutable and are already verified.
    closure['coordinator_source_files'] = {str(p.relative_to(REPO)): sha(p) for p in sorted(HERE.glob('*.py'))}
    write_json(HERE / 'runtime_closure.json', closure)
    bound = [HERE / name for name in ('fits.json', 'implementation_sources.json',
        'runtime_closure.json', 'RUNTIME_SOURCES.sha256', 'H100_SOURCES.sha256',
        'TIMAN107_SOURCES.sha256', 'TIMAN108_SOURCES.sha256', 'pilot_roots.json',
        'test_a_roots.json', 'queue_pilot.txt', 'queue_test_a.txt', 'plans.json',
        'mechanics_tests.xml', 'r13_tests.xml', 'regression_tests.xml',
        'coordinator_commands.sh', 'allocation_audit.json', 'pilot_identity_audit.json', 'HANDBACK.md')]
    hashes = {str(p.relative_to(REPO)): sha(p) for p in bound}
    atomic(HERE / 'RELEASE.sha256', ''.join(f'{digest}  {name}\n' for name, digest in hashes.items()).encode())
    result = dict(PASS=True, tests=counts, regression_subtests=25, protected_unchanged=protected,
        authorized_shared_edits=shared, cells=cells, catalog_artifacts=104,
        owned_R14_artifacts=sum(Path(r['path']).is_relative_to(KNOB) for r in catalog),
        catalog_bytes=sum(r['bytes'] for r in catalog), source_files=len(closure['source_files']),
        external_parent_action_payloads=len(closure['payloads']), pilot_roots=6, pilot_arms=68,
        staged_test_roots=8, test_arms=128, plans=14, launches=0, installed_A=False,
        all_new_paths_regular=True, hashes=hashes,
        pending='Coordinator R13 scheduling, live rollout-seed check and frozen B release gates')
    write_json(HERE / 'release_audit.json', result)
    print('RELEASE_PASS', json.dumps({k: result[k] for k in ('tests', 'regression_subtests',
        'protected_unchanged', 'catalog_artifacts', 'owned_R14_artifacts', 'source_files', 'plans')}))


if __name__ == '__main__':
    main()
