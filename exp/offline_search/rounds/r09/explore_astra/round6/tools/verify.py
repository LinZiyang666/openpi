"""Local freeze audit. Never imports or wraps h100 control, and never launches."""
import argparse
from datetime import datetime, timezone
import json
from .data import HERE, RUN, STORE, CELLS, dump, sha, artifact
from .sources import closure


def audit():
    frozen=json.loads((HERE/'FROZEN.json').read_text())
    assert sha(HERE/'PREDICTION.md')==frozen['prediction_sha256']
    assert sha(HERE/'SELECTION.json')==frozen['selection_sha256']
    assert (HERE/'PREDICTION.md').stat().st_mtime < (RUN/'arms.json').stat().st_mtime
    assert json.loads((HERE/'SELECTION.json').read_text())['eval_accessed'] is False
    for f in frozen['arms']:
        assert sha(f['artifact'])==f['sha256']
        if f['variant']=='control': assert f['sha256']==f['control_source_sha256']
    rows=json.loads((RUN/'arms.json').read_text())
    assert len(rows)==8
    for cell in CELLS:
        pair=[r for r in rows if r['arm'].startswith('r9a6_'+cell+'_')]
        assert len(pair)==2 and {r['arm'].rsplit('_',1)[-1] for r in pair}=={'control','taskfree'}
        assert pair[0]['client_overrides']==pair[1]['client_overrides']
        for r in pair:
            a=r['plugin_args']
            assert a[a.index('--os-root')+1]==str(STORE)
            assert r['manifest']==str(RUN/'manifests/eval100.json')
    m=json.loads((RUN/'manifests/eval100.json').read_text())
    assert {(r['task'],r['init']) for r in m['selected']}=={(t,i) for t in range(10) for i in range(20,30)}
    tests=json.loads((HERE/'results/plugin_selftests.json').read_text())
    assert len(tests)==16 and all(t['report']['PASS'] for t in tests)
    unit_log=(HERE/'results/unit_tests.log').read_text()
    assert 'Ran 10 tests' in unit_log and unit_log.rstrip().endswith('OK')
    for t in tests:
        p=RUN/'selftest'/(t['arm']+('_forced' if t['forced'] else '_prod'))/'admission.json'
        d=json.loads(p.read_text())
        assert len(d['pairs'])==2 and all(0<=i<20 for _,i in d['pairs'])
        assert d['standard_store']==str(STORE)
        if t['forced']: assert t['report']['miss']>0 and t['report']['policy_tail']>0
    sources=json.loads((HERE/'H100_SOURCES.json').read_text())
    assert sources==closure()
    assert sum(r['new_round6'] for r in sources)==4
    plan=json.loads((RUN/'h100_sync/plan.json').read_text())
    assert len(plan['arms'])==8
    maps=[f for f in plan['files'] if 'episode identity map' in str(f['reasons'])]
    assert {f['rel'] for f in maps}=={f'store/queries/{m}_{s}_cache/episodes.json'
                                    for m in ('pi05','groot') for s in ('l10','spatial')}
    assert all('serving_store' not in str(r) for r in rows)
    assert not (RUN/'serving_store').exists()
    assert not (RUN/'state').exists()
    assert not (RUN/'runs').exists()
    for name in ('REPORT.md','DATA_ANALYSIS.md','PREDICTION.md','HANDBACK.md'):
        assert (HERE/name).is_file()
    return dict(arms=8,copied_controls=4,unit_tests=10,plugin_selftests=16,
        source_files=len(sources),new_source_files=4,plan_files=len(plan['files']),plan_bytes=plan['bytes'],
        standard_store=str(STORE),launched=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seal',action='store_true')
    args=p.parse_args()
    result=audit()
    if args.seal:
        files=[p for p in HERE.rglob('*') if p.is_file() and p.name!='INTEGRITY.json' and '__pycache__' not in p.parts]
        files += [RUN/'arms.json', RUN/'manifests/eval100.json', RUN/'h100_sync/plan.json']
        files += list((RUN/'fits').glob('*.pkl'))+list((RUN/'config').glob('*.yaml'))
        files += list((RUN/'selftest').glob('*/admission.json'))
        files += list((RUN/'selftest').glob('*/selftest_report.json'))
        dump(HERE/'INTEGRITY.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),checks=result,
            files=[dict(path=str(p),sha256=sha(p)) for p in sorted(set(files))]))
    else:
        for f in json.loads((HERE/'INTEGRITY.json').read_text())['files']:
            assert sha(f['path'])==f['sha256'], 'changed frozen file: '+f['path']
    print(json.dumps(dict(PASS=True,**result),indent=2))


if __name__=='__main__': main()
