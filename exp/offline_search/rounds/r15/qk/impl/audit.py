"""Bind completed verification and planned roots to their frozen artifacts."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    evidence = {}
    for name in ('plugin_replay.json','verification.json','v2check_replay.json',
                 'baseline_equivalence.json','plans/verification.json','descent_safety.json'):
        data = json.loads((HERE/name).read_text())
        assert data['PASS'] is True, name
        evidence[name] = sha(HERE/name)
    tests = ET.parse(HERE/'tests.xml').getroot().find('testsuite')
    assert int(tests.attrib['tests']) == 6
    assert int(tests.attrib['failures']) == int(tests.attrib['errors']) == 0
    evidence['tests.xml'] = sha(HERE/'tests.xml')
    evidence['gate_report.json'] = sha(HERE/'gate_report.json')
    cells = {}
    for size in (50,200):
        manifest = json.loads((HERE/f'artifacts/pi05_l10_{size}/manifest.json').read_text())
        for rec in manifest['records'].values():
            assert sha(rec['path']) == rec['sha256']
        spec = manifest['records']['off']['kwargs']['encoder_spec']
        assert sha(spec['kwargs']['model_path']) == spec['kwargs']['model_sha256']
        assert manifest['encoder_spec']['candidate_name'] == 'state_drop_aug'
        assert spec['completed_epochs'] == spec['declared_epochs'] == {50:40,200:120}[size]
        assert sha(manifest['encoder_spec']['candidate_manifest']) == manifest['encoder_spec']['candidate_manifest_sha256']
        for report in ('plugin_replay.json','v2check_replay.json'):
            records=json.loads((HERE/report).read_text())['cells']
            row=next(r for r in records if r.get('size')==size or r.get('cell')==f'pi05_l10_{size}')
            assert row['artifact_sha256'] == manifest['records']['off']['sha256']
            assert row['student_sha256'] == spec['kwargs']['model_sha256']
            assert row['completed_epochs'] == spec['completed_epochs']
        cells[str(size)] = {}
        for kind in ('loeo','dev'):
            path = HERE/f'evaluation/{kind}_{size}.json'
            data = json.loads(path.read_text())
            assert data['student_artifact_sha256'] == manifest['records']['off']['sha256']
            assert data['representation_sha256'] == spec['representation_sha256']
            cells[str(size)][kind] = dict(rows=data['rows'],episodes=data['episodes'],
                report_sha256=sha(path),student_artifact_sha256=data['student_artifact_sha256'])
    roots = json.loads((HERE/'pilot_roots.json').read_text())
    for record in roots:
        root = Path(record['root'])
        for file, digest in record['definitions_sha256'].items():
            assert sha(root/file) == digest, file
        for file, digest in record['fit_sha256'].items():
            assert sha(file) == digest, file
        for row in json.loads((root/'arms.json').read_text()):
            if '--os-light-key' in row['plugin_args']:
                size = row['r10_size']
                manifest=json.loads((HERE/f'artifacts/pi05_l10_{size}/manifest.json').read_text())
                assert row['kwargs']['encoder_spec'] == manifest['records']['off']['kwargs']['encoder_spec']
        for file in root.rglob('*'):
            assert not file.is_symlink() and (file.is_dir() or file.is_file()), file
    for file in HERE.rglob('*'):
        assert not file.is_symlink() and (file.is_dir() or file.is_file()), file
    queue=(HERE/'queue_pilot.txt').read_text().splitlines()
    assert len(queue)==len(set(queue))==6
    assert queue==[f'@h100 {r["root"]}|{arm}' for r in roots for arm in r['arms']]
    report = dict(PASS=True,evidence=evidence,cells=cells,root_seals_bound=True,
                  regular_outputs=True,planned_arms=sum(r['arm_count'] for r in roots),
                  launches=0,remote_operations=0,servers=0,GPU=0,test_A_reads=0)
    (HERE/'completion_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('COMPLETION_AUDIT_PASS',report['planned_arms'],'arms')
    return report


if __name__=='__main__':
    run()
