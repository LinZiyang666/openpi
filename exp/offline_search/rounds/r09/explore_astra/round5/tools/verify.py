"""Read-only freeze audit by default; --seal writes only the owned inventory."""
import argparse
import json
from datetime import datetime, timezone

from .safe import HERE, RUN, EVAL, dump, sha
from .sources import closure


def audit():
    frozen=json.loads((HERE/'FROZEN.json').read_text())
    assert sha(HERE/'PREDICTION.md')==frozen['prediction_sha256']
    assert (HERE/'PREDICTION.md').stat().st_mtime < (RUN/'arms.json').stat().st_mtime
    for row in frozen['arms']:
        assert sha(row['artifact'])==row['sha256']
        if row['mode']=='control':assert row['sha256']==row['control_source_sha256']
    source=json.loads((HERE/'H100_SOURCES.json').read_text())
    assert closure()==source
    specs=json.loads((RUN/'arms.json').read_text())
    assert len(specs)==6
    for model in ['pi05','groot']:
        arms=[a for a in specs if a['model']==model]
        assert len(arms)==3
        assert {a['arm'].rsplit('_',1)[-1] for a in arms}=={'control','latch','pace12'}
        assert all(a['client_overrides']==arms[0]['client_overrides'] for a in arms)
        for a in arms:
            assert a['manifest']==str(RUN/'manifests/eval100.json')
            if not a['arm'].endswith('_control'):assert a['kwargs']['force_at']==[]
    manifest=json.loads((RUN/'manifests/eval100.json').read_text())
    assert {(r['task'],r['init']) for r in manifest['selected']}==EVAL
    tests=json.loads((HERE/'results/plugin_selftests.json').read_text())
    assert len(tests)==12 and all(r['returncode']==0 and r['report']['PASS'] for r in tests)
    for r in tests:
        out=RUN/'selftest'/(r['arm']+('_forced' if r['forced'] else '_prod'))
        admission=json.loads((out/'admission.json').read_text())
        assert admission['allowed_pairs']==[[0,0],[1,0]]
        if r['forced'] and not r['arm'].endswith('_control'):
            assert admission['new_trigger_verified']
    plan=json.loads((RUN/'h100_sync/plan.json').read_text())
    maps=[f for f in plan['files'] if 'episode identity map' in str(f['reasons'])]
    assert len(maps)==2
    assert all(f['rel'].startswith('runs/r09_astra_r5/serving_store/queries/') for f in maps)
    for f in maps:
        # These are identity-only files produced by the admission parser.
        data=json.loads(open(f['source']).read())
        assert {(e['task_id'],e['init']) for e in data}==EVAL
        assert all(set(e)=={'uid','task_id','init','task'} for e in data)
    for name in ['REPORT.md','DATA_ANALYSIS.md','PREDICTION.md','HANDBACK.md']:
        assert (HERE/name).is_file()
    return dict(arms=6,exact_copied_controls=2,plugin_selftests=12,source_files=len(source),
        plan_files=len(plan['files']),plan_bytes=plan['bytes'],forbidden_outcomes_loaded=0)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seal',action='store_true');a=ap.parse_args()
    result=audit()
    inventory=HERE/'INTEGRITY.json'
    if a.seal:
        files=[p for p in HERE.rglob('*') if p.is_file() and p.name!='INTEGRITY.json' and '__pycache__' not in p.parts]
        files += [RUN/'arms.json',RUN/'manifests/eval100.json',RUN/'h100_sync/plan.json']
        files += sorted((RUN/'fits').glob('*.pkl'))
        files += sorted((RUN/'config').glob('*.yaml'))
        files += sorted((RUN/'selftest').glob('*/admission.json'))
        files += sorted((RUN/'selftest').glob('*/selftest_report.json'))
        dump(inventory,dict(sealed_utc=datetime.now(timezone.utc).isoformat(),checks=result,
            files=[dict(path=str(p),sha256=sha(p)) for p in sorted(set(files))]))
    else:
        sealed=json.loads(inventory.read_text())
        for f in sealed['files']:
            assert sha(f['path'])==f['sha256'], 'changed after seal: '+f['path']
    print(json.dumps(dict(PASS=True,**result),indent=2))


if __name__=='__main__':main()
