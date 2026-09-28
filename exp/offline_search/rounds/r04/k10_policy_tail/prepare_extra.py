from pathlib import Path
B=Path(__file__).resolve().parent
K5=B.parent/'k5_rand'
for name in ('prepare_arms.py','validate_estimator.py','check_replay_estimator.py','check_ledger.py','final_audit.py','summarize_checks.py'):
    s=(K5/name).read_text().replace('/tmp/k5_','/tmp/k10_').replace("prefix='k5_estimator_'","prefix='k10_estimator_'")
    s=s.replace("BASE/'estimate.py'",f"Path({str(K5/'estimate.py')!r})")
    s=s.replace('/tmp/k10_existing_installed','/tmp/k10_installed/existing')
    s=s.replace("BASE/'results'", "BASE/'results'/'installed'").replace("BASE/'results/", "BASE/'results/installed/")
    (B/('k5_'+name)).write_text(s)
# Keep original K6 test intact; make a separate K10 schedule replay extension.
s=(B/'concurrency_test.py').read_text()
start=s.index('def configs():');end=s.index('\n\ndef digest',start)
s=s[:start]+'''def configs():
    result={}
    for suite,scale in [('l10',50),('l10',500),('spatial',500)]:
        for judge in ('guard_only','threshold:inf','periodic:5'):
            key=f'{suite}_{scale}_{judge.replace(":","_")}'
            result[key]=dict(cell=f'pi05_{suite}_cache',judge=judge,blind=True,
                method='exp.offline_search.rounds.r04.k10_policy_tail.judge:PolicyTailJudge',
                kwargs=dict(base_kwargs=dict(lib='current' if scale==50 else 'big',kref=5 if scale==50 else 8,serving='anchor_tail',budget=1,gates='budget_only'),guards=judge!='threshold:inf',ncal=64))
    return result
'''+s[end:]
s=s.replace("if c['blind']: args+=['--os-blind']","if c['blind']: args+=['--os-blind','--os-policy-tail']")
s=s.replace("('serialized','before')","('serialized',a.source)")
s=s.replace("eligible=eligible,records=", "policy_tails=sum(r['src']=='policy_tail' for r in dec),eligible=eligible,records=")
s=s.replace("config=name,PASS=True,connections=8,", "config=name,PASS=True,policy_tails=t['policy_tails'],connections=8,")
(B/'tail_concurrency_test.py').write_text(s)
