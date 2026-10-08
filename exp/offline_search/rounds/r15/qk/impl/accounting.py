"""QK aggregation compatible with existing collect/KPI output fields."""
import math
from collections import Counter


def aggregate(meta, rows, starts=(), comp=None):
    if not rows or any('qk_work' not in r for r in rows):
        raise ValueError('cannot mix light and historical cost ledgers')
    epsilon = {float(r['epsilon']) for r in rows}
    if len(epsilon) != 1 or not all(math.isfinite(x) and x >= 0 for x in epsilon):
        raise ValueError('changed/invalid light-key epsilon')
    epsilon = epsilon.pop()
    counts = {k:sum(int(r['qk_work'][k]) for r in rows) for k in ('Q','T','M','N')}
    if counts['T'] != counts['M'] or counts['N'] != len(rows):
        raise ValueError('invalid actual-work ledger')
    for r in rows:
        w = r['qk_work']
        if (w['Q'] != int(r['light_encoder_ran']) or w['T'] != int(r['teacher_tower_ran'])
                or w['M'] != int(r['policy_full_call']) or w['N'] != 1):
            raise ValueError('QK decision work is not attested')
    Q,T,M,N = (counts[k] for k in ('Q','T','M','N'))
    L = int(meta.get('client_overrides',{}).get('replan_steps',5))
    if L != 5:
        raise ValueError('initial QK pilot requires five-control requests')
    nominal = 5*N
    actual = sum(int(x['n_steps']) for x in comp.values()) if comp and all(x.get('n_steps') is not None for x in comp.values()) else None
    episodes = len(comp) if comp is not None else len({r.get('uid') for r in rows})
    ks = Counter(str(float(r['miss_k'])) for r in rows if r['policy_full_call'])
    total = epsilon*Q+M
    return dict(model=meta.get('model','pi05'), decisions=N,vision_decisions=T,misses=M,
        **counts,epsilon=epsilon,IR_new=total/N,v=T/N,m=M/N,
        k_per_miss=dict(counts=dict(ks),mean=sum(float(k)*v for k,v in ks.items())/M if M else None),
        l_per_request=dict(counts={'5.0':N},mean=5.,default=5),stage1_modes={'light_key':Q,'full_policy':M},
        stage_costs={'light_key':epsilon,'full_policy':1.},full_stage_costs={'s1':.152,'s2':.410,'s3':.438,'k':10},
        cost_source='R15 actual Q/T/M/N; epsilon from frozen encoder artifact', full_cost=1.,full_cost_ms=None,
        legacy_owner_pricing=False,total_cost=total,ir_per_request=total/N,ir_per_five_controls=total/N,
        controls=nominal,controls_per_episode=nominal/episodes if episodes else None,
        controls_note='nominal 5*N; terminal controls counted separately from client journal',
        actual_controls=actual,actual_controls_per_episode=actual/episodes if actual is not None and episodes else None,
        s1_ms_total=sum(r.get('teacher_tower_ms') or 0. for r in rows),
        s23_ms_total=sum(r.get('policy_remainder_ms') or 0. for r in rows),s1_ms_n=T,s23_ms_n=M,
        light_encoder_ms_total=sum(r.get('light_encoder_ms') or 0. for r in rows),
        ir_measured_per_five_controls=None,warnings=[])
