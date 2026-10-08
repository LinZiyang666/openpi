"""Final read-only input, output hash and test-count summary; no rollout ops."""
import difflib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from .prepare import HERE, ASTRA, R13, write_json
from exp.offline_search.closed_loop.devset import sha


def test_counts(path):
    root=ET.parse(path).getroot();suites=list(root.iter('testsuite'))
    values={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert values['failures']==values['errors']==values['skipped']==0
    return values


if __name__=='__main__':
    protected=json.loads((HERE/'protected_before.json').read_text())
    frozen={p:d for p,d in protected.items() if '/astra/' in p}
    for p,d in frozen.items():assert sha(p)==d,p
    now={str(p.relative_to(HERE.parents[4])) for p in ASTRA.rglob('*')
         if p.is_file() and '__pycache__' not in p.parts}
    assert now==set(frozen)
    fits=json.loads((HERE/'fits.json').read_text())
    for r in fits:assert sha(r['path'])==r['sha256'],r['path']
    for filename in ('pilot_roots.json','test_a_roots.json'):
        for r in json.loads((HERE/filename).read_text()):
            root=Path(r['root']);record=json.loads((root/'ROOT.json').read_text())
            for p,d in record['fit_sha256'].items():assert sha(p)==d
            for name,d in record['definitions_sha256'].items():assert sha(root/name)==d
    for p in (HERE/'H100_SOURCES.sha256').read_text().splitlines():
        digest,rel=p.split('  ',1);assert sha(HERE.parents[4]/rel)==digest
    result=subprocess.run(['bash','-n',str(HERE/'coordinator_commands.sh')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    changes=[];patch=[]
    for name,relative in [('plugin','plugin.py'),('look_recording','look_recording.py'),
                          ('worker_pool','worker_pool.py'),('control','ops/h100/control.py')]:
        source=HERE/'before'/f'{name}.py';target=HERE.parents[2]/'closed_loop'/relative
        patch+=list(difflib.unified_diff(source.read_text().splitlines(True),target.read_text().splitlines(True),
                                       fromfile=str(source),tofile=str(target)))
        changes.append(dict(path=str(target),before_sha256=sha(source),after_sha256=sha(target)))
    (HERE/'shared_changes.diff').write_text(''.join(patch))
    mechanics=json.loads((HERE/'mechanics.json').read_text())
    counts=dict(new=test_counts(HERE/'mechanics_tests.xml'),regression=test_counts(HERE/'regression_tests.xml'))
    # pytest-subtests reports its additional 25 passing subtests in XML as
    # individual cases; pytest's headline remains 296 existing tests.
    files={name:sha(HERE/name) for name in ('fits.json','mechanics.json','plans.json','periodic_solver.json',
           'runtime_audit.json','benchmark.json','pure_c5_planner_carrier.json',
           'queue_pilot.txt','queue_test_a.txt','H100_SOURCES.sha256',
           'coordinator_commands.sh','write_scope_exception.json','HANDBACK.md')}
    for name in ('mechanics.json','plans.json','periodic_solver.json','runtime_audit.json',
                 'benchmark.json','pure_c5_planner_carrier.json'):
        assert json.loads((HERE/name).read_text())['PASS'],name
    exception=json.loads((HERE/'write_scope_exception.json').read_text())
    write_json(HERE/'RELEASE.json',dict(PASS=True,test_counts=counts,frozen_astra_files_unchanged=len(frozen),
               fits=len(fits),shared_changes=changes,evidence_sha256=files,
               off_byte_equivalence_cells=len(mechanics['off_equivalence']),
               write_scope_compliant=not exception['files'],
               first_regression_write_scope_exception_files=len(exception['files']),
               new_method_closed_loop_rollouts=0,remote_operations=0,servers_launched=0,GPU_operations=0))
    print('RELEASE_PASS',json.dumps(counts),flush=True)
