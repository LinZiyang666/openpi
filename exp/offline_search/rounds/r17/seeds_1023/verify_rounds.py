# verify_rounds.py: on finished full sweeps, drop every level a descent round would not have run and check that the
# per-task selection (knob/descent.descend through formal_a_analysis.choose's rule) and the selected episodes are unchanged.
import sys
from unittest.mock import patch
from exp.offline_search.rounds.r17.filter_failed_knob import formal_a_analysis as a
from exp.offline_search.rounds.r17.longseg_off_1022 import build as lob
from exp.offline_search.rounds.r17.knob import common as c, descent

LEVELS = ('k50', 'k40', 'k30', 'k20', 'k10')


def select(arms, pure, order):
    out = {}
    for task in sorted({k[1] for k in pure}):
        ref = {k: v for k, v in pure.items() if k[1] == task}
        base = sum(e['success'] for e in ref.values())
        levels = {'pure': dict(n=len(ref), successes=base, passes={'4': True})}
        for level in order:
            data = {k: v for k, v in arms.get(level, {}).items() if k in ref}
            levels[level] = dict(n=len(data), passes={'4': base-sum(e['success'] for e in data.values()) <= 2
                                                      if set(data) == set(ref) and len(ref) == 50 else None})
        with patch.object(c, 'LEVELS', ('pure', *order)):
            d = descent.descend(levels, 4, count=50)
        sel = d['selected']
        out[task] = (sel, None if sel is None else tuple(sorted((k, v['success']) for k, v in
                                                               (ref if sel == 'pure' else arms[sel]).items() if k in ref)))
    return out


def trim(arms, pure):
    """Keep, per level, only the tasks still alive in that round (the rule in seeds_1023.workflow.build_round)."""
    alive = sorted({k[1] for k in pure}); kept = {lv: {} for lv in arms}
    kept['koff'] = arms.get('koff', {})
    for level in LEVELS:
        survivors = []
        for task in alive:
            inits = [k for k in pure if k[1] == task]
            got = {k: arms.get(level, {}).get(k) for k in inits}
            kept[level].update({k: v for k, v in got.items() if v is not None})
            if None in got.values():
                continue
            if sum(pure[k]['success'] for k in inits)-sum(v['success'] for v in got.values()) <= 2:
                survivors.append(task)
        alive = survivors
    return kept


def check(label, runs, ref):
    data = {}
    for run in runs:
        for cell, arms in a.load_r17(dict(run=run)).items():
            for lv, eps in arms.items():
                data.setdefault(cell, {}).setdefault(lv, {}).update(eps)
    order = (*LEVELS, 'koff'); total = same = cells = 0; dropped = kept_n = 0
    for cell, arms in sorted(data.items(), key=str):
        pure = ref.get(cell[0], {})
        if not all(lv in arms for lv in LEVELS) or not pure:
            continue
        cells += 1
        full = select(arms, pure, order); t = trim(arms, pure); short = select(t, pure, order)
        kept_n += sum(len(t[lv]) for lv in LEVELS); dropped += sum(len(arms[lv]) for lv in LEVELS)-sum(len(t[lv]) for lv in LEVELS)
        for task in full:
            total += 1; same += full[task] == short[task]
            if full[task] != short[task]:
                print('DIFF', label, cell, task, full[task][0], short[task][0])
    print(f'{label}: {cells} tuned cells, {same}/{total} tasks identical (level and selected episodes); '
          f'level episodes kept {kept_n}, dropped {dropped} ({dropped/(kept_n+dropped):.0%})', flush=True)


E = '/data/mw_finetune/eval/'
which = sys.argv[1]
if which == 'fw_libero':
    check('Fast-WAM LIBERO reduced (OFF)', [E+'fa_lo_fw_r_f_o_l', E+'fa_lo_fw_r_f_l_l'], lob.reference('fastwam', 'libero', 'reduced', 'H100'))
elif which == 'gr_mt50':
    check('GR00T MT50 reduced (OFF)', [E+'fa_lo_gr_r_f_m_j'], lob.reference('groot', 'mt50', 'reduced', lob.POD))
elif which == 'dp_mt50':
    check('DP MT50 full (OFF)', [E+'fa_lo_dp_f_f_m'], lob.reference('dp', 'mt50', 'full', 'H100'))
