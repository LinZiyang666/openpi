"""Final local deliverable audit and K4 ledger identities; no GPU or server work."""
import hashlib,json,pathlib,py_compile
from exp.offline_search.rounds.r04.k4_eval.cost_ledger import ledger
HERE=pathlib.Path(__file__).resolve().parent
P=HERE/'results'
REPO=HERE.parents[4]

def read(p):return json.loads(pathlib.Path(p).read_text())

def main():
    table=read(REPO/'exp/offline_search/closed_loop/ops/cost_table.json')
    ledger_cases=[]
    for model,m in table['models'].items():
        f=m['full'];den=sum(f[k] for k in ('s1','s2','s3'))
        for mode,c in m['modes'].items():
            for hit in (False,True):
                r=ledger({'model':model},[{'uid':'test','vision':True,'hit':hit,'miss_k':None if hit else 2}],
                         [{'stage1_mode':mode,'miss_steps':2}])
                want=(c['s1']+(0 if hit else c.get('miss_s1_extra',0)+c['s2']+c['s3']*2/c['k']))/den
                assert abs(r['ir_per_request']-want)<1e-12
                ledger_cases.append({'model':model,'mode':mode,'hit':hit,'ir':want})
        r=ledger({'model':model},[{'uid':'test','vision':True,'hit':False,'miss_k':f['k']}],[{'stage1_mode':'full'}])
        assert r['ir_per_request']==1
        r=ledger({'model':model,'replan_steps':10},[{'uid':'test','vision':True,'hit':False,'miss_k':f['k']}],[{'stage1_mode':'full'}])
        assert r['ir_per_five_controls']==.5
    (P/'ledger_checks.json').write_text(json.dumps({'mode_cases':ledger_cases,'full_Kref_is_one':True,'L10_is_half':True},indent=2))
    reports=[]
    for model in ('pi05','groot'):
        for mode in ('cache','miss'):
            b=read(P/f'before_{model}_{mode}/selftest_report.json');a=read(P/f'final_{model}_{mode}/selftest_report.json')
            for key in ('PASS','episodes','decisions','winner_matches_log','agree_rec_top1','native_agree_rec_top1'):
                assert a[key]==b[key]
            assert a['PASS'];reports.append({'model':model,'path':mode,'decisions':a['decisions'],'episodes':a['episodes'],
                                            'PASS':True,'before_after_metrics_equal':True})
    (P/'final_selftests.json').write_text(json.dumps(reports,indent=2))
    for r in read(P/'install.json')+[read(P/'cost_install.json')]:
        actual=hashlib.sha256(pathlib.Path(r['file']).read_bytes()).hexdigest()
        assert actual==r.get('after_sha256',r.get('sha256')),(r['file'],actual)
    assert read(P/'arm_checks.json')['arms']==len(read(HERE/'arms_r4.json'))==26
    assert read(P/'startup_check.json')['passed']
    assert read(P/'override_checks.json')['passed']==11
    assert len(read(P/'blind_wrist_checks.json'))==4
    for p in P.glob('wrist_*/selftest_report.json'):assert read(p)['PASS']
    pi=read(P/'pi05.json');assert len(pi['parity'])==12 and all(pi['installed_split_rebatch_parity'])
    for r in pi['parity']:
        for k in ('dummy_keys_exact','wrist_key_exact','complete_prefix_exact','dummy_K2_exact','dummy_K10_exact','wrist_complete_K2_exact','wrist_complete_K10_exact'):assert r[k]
    for f in ('groot_spatial.json','groot_l10.json'):assert all(r['exact'] for r in read(P/f)['parity'])
    for f in ('pi05.json','groot_spatial.json','groot_l10.json'):assert read(P/f)['peak_reserved_bytes']<10*2**30
    assert not (P/'GPU_OOM_STOP').exists()
    sources=list(HERE.glob('*.py'))+[REPO/'exp/offline_search/closed_loop'/x for x in ('stage_overrides.py','serve_pi05.py','serve_groot.py')]
    for f in sources:py_compile.compile(str(f),doraise=True)
    result={'PASS':True,'arms':26,'ledger_cases':len(ledger_cases),'default_selftest_decisions':sum(r['decisions'] for r in reports),
            'wrist_selftest_decisions':sum(read(p)['decisions'] for p in P.glob('wrist_*/selftest_report.json')),
            'python_files_compiled':len(sources),'installed_hashes_match':True,'GPU_OOM':False}
    (P/'final_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
