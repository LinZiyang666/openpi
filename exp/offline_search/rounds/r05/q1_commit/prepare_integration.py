"""Adapt K10's eight-connection and transform/lifecycle fixtures for C10/D1."""
from pathlib import Path
from prepare import write

HERE = Path(__file__).resolve().parent
OLD = HERE.parent.parent / 'r04/k10_policy_tail'
s = (OLD / 'tail_concurrency_test.py').read_text()
start, end = s.index('def configs():'), s.index('\n\ndef digest')
s = s[:start] + '''def configs():
    result = {}
    for arm in json.loads((BASE/'arms_q1.json').read_text()):
        result[arm['name']] = dict(cell=f"pi05_{arm['suite']}_cache", judge='guard_only', blind=True,
            method=arm['method'], kwargs=arm['kwargs'], fit=f"/tmp/q1_fits/{arm['name']}.pkl",
            policy_tail='c10' in arm['name'])
    return result
''' + s[end:]
s = s.replace('from launch_test import load', 'from exp.offline_search.rounds.r04.k10_policy_tail.launch_test import load')
s = s.replace("if c['blind']: args+=['--os-blind','--os-policy-tail']",
              "if c['blind']: args+=['--os-blind','--os-no-shadow-native']\n    if c['policy_tail']: args+=['--os-policy-tail']")
needle = "    eligible={t:sum(r.get('eligible',False)"
start = s.index(needle)
s = s[:start] + '''    tail_eligible = 0
    for conn in range(8):
        for uid in {r['uid'] for r in dec if r['conn'] == conn}:
            ds = sorted([r for r in dec if r['conn'] == conn and r['uid'] == uid], key=lambda r:r['step'])
            tail_eligible += sum(not r['hit'] and r['vision'] for r in ds[:-1])
            if c['policy_tail']:
                assert all(ds[i+1]['src']=='policy_tail' for i,r in enumerate(ds[:-1]) if not r['hit'])
            else:
                assert sum(r['extras'].get('grasp_check',0)==1 for r in ds) <= 1
    if c['policy_tail']:
        assert tail_eligible == sum(r['src']=='policy_tail' for r in dec)
    '''.rstrip() + '\n' + s[start:]
s = s.replace("policy_tails=sum(r['src']=='policy_tail' for r in dec),eligible=", "tail_eligible=tail_eligible,policy_tails=sum(r['src']=='policy_tail' for r in dec),eligible=")
s = s.replace("policy_tails=t['policy_tails'],connections=8", "tail_eligible=t['tail_eligible'],policy_tails=t['policy_tails'],connections=8")
write(HERE / 'concurrency_test.py', s)

s = (OLD / 'policy_tail_test.py').read_text()
s = s.replace('from launch_test import load', 'from exp.offline_search.rounds.r04.k10_policy_tail.launch_test import load')
s = s.replace('exp.offline_search.rounds.r04.k10_policy_tail.judge', 'exp.offline_search.rounds.r05.q1_commit.judge')
s = s.replace('PolicyTailJudge', 'CommitJudge')
s = s.replace('guards=False,ncal=64', "guards=True,stuck_thr=0,ncal=64,policy_tail_gate='lifecycle'")
s = s.replace("'--os-judge','threshold:inf'", "'--os-judge','guard_only'")
s = s.replace("assert s.has_vision[-1] and s._look_reason==code,(label,s._look_reason)",
              "assert not s.has_vision[-1] and s._policy_tail_step==s.step-1,(label,s._look_reason)")
s = s.replace("s.method=saved_method;checks.append(label)", "s.method=saved_method;checks.append(label.replace('veto','bypassed'))")
# Duplicate rejection happens before method selection, preserving the saved tail.
needle = "    start(0);conn.infer(obs(0));s.burst_left=1;"
start = s.index(needle)
s = s[:start] + '''    start(0);miss=conn.infer(obs(0));saved_tail=s._policy_tail
    count=rt.decision_count
    try: conn.infer(obs(0))
    except ValueError as exc: assert 'duplicate decision_id' in str(exc)
    else: raise AssertionError('duplicate accepted')
    assert rt.decision_count==count and s._policy_tail is saved_tail
    tail=conn.infer(obs(1));assert np.array_equal(tail['actions'],policy_tail_chunk(miss['actions']))
    checks.append('duplicate MISS id preserves unconsumed tail')
''' + s[start:]
write(HERE / 'policy_tail_test.py', s)
print('Wrote C10 guard_only edge/transform test and six-arm concurrency matrix')
