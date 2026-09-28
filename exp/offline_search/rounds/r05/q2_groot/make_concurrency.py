from pathlib import Path
B=Path(__file__).resolve().parent
s=(B/'regression/tail_concurrency_test.py').read_text()
a=s.index('def configs():');b=s.index('\n\ndef digest',a)
s=s[:a]+'''def configs():
    result={}
    for suite in ('spatial','l10'):
        for scale in (50,500):
            for blocks in (1,2):
                key=f'{suite}_{scale}_G{5*(blocks+1)}'
                result[key]=dict(cell=f'groot_{suite}_cache',judge='guard_only',blind=True,blocks=blocks,
                    method='exp.offline_search.rounds.r05.q2_groot.judge:CycleTail',
                    kwargs=dict(lib='current' if scale==50 else 'big',kref=5 if scale==50 else 8,cycle_k=4,tail_blocks=blocks))
    return result
'''+s[b:]
s=s.replace("from launch_test import load", "sys.path.insert(0,str(BASE/'regression'))\nfrom launch_test import load")
s=s.replace("args+=['--os-blind','--os-policy-tail']", "args+=['--os-blind','--os-policy-tail','--os-policy-tail-blocks',str(c['blocks'])]")
# Audit mutable method and cursor state too, excluding object identities.
s=s.replace("if rt.randomized: snap.update", "snap.update(anchor_count=s.method._anchor_count,anchor=clean(s.method._anchor) if s.method._anchor is not None else None,policy_cursor=None if s._policy_tail is None else s._policy_tail['cursor'])\n        if snap['anchor'] is not None:\n            snap['anchor']={k:(digest(v) if isinstance(v,np.ndarray) else v) for k,v in snap['anchor'].items()}\n        if rt.randomized: snap.update")
(B/'concurrency_test.py').write_text(s)
