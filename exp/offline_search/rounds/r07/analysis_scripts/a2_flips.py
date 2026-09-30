"""R7 A2 per-episode flip table: each arm versus A on the same (task, init), with lever exposure and stage.

Reference: A's three replicates (R6 identities, rounds/r06/analysis_scripts/common.py:ab_arms) on the eval run;
the paired single A_profile arm on the PROFILE run. flip = 'loss' (A majority success, arm failure), 'gain' (A
majority failure, arm success), else 'same_success' / 'same_fail'. Every row carries A_k (successes out of the
replicates), the arm's termination reason, lever counts, the first lever event (a2_stage: the first decision where
the arm departs from A's cadence/content) with the library stage it acted on, the stage of the last anchor
(final_label) and the furthest macro stage reached.

Cross-run decision comparison is used only as a check (--xrun, default GR00T arms): on pi0.5, A's own replicates
already differ bitwise at step 0 in most episodes (encoder non-determinism changes kernel weights with identical robot
state), so the first lever event is the only well-defined divergence point there; on GR00T, A replicates are
step-identical in ~93 % of pairs, so the arm's first bitwise departure from A replicate 1 can be compared with its first
lever event (identity until the lever).

Noise reference: A self-flips = replicate k versus the other two when those two agree (averaged over k).
SF1 x UF1 on the same pairs: 2x2 outcome table and UF's SF-refused extensions in UF-only losses.

    <A2 prefix> -m exp.offline_search.rounds.r07.analysis_scripts.a2_flips --run eval [--arms ...] [--jobs 4]
Writes <out>/a2_flips_<run>.json and <detail>/a2_flips_<run>.csv.gz.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from exp.offline_search.rounds.r07.analysis_scripts import a2_common as C
from exp.offline_search.rounds.r07.analysis_scripts import a2_stage as S

LEVER_COLS = ('granted', 'ext_blocks', 'sf_refused_ext', 'early_abort', 'ext_fire', 'wrist_looks', 'calls',
              'lottery_calls', 'stall_calls', 'extra_looks')


def a_self_noise(specs):
    outs = [C.journal_outcomes(s) for s in specs]
    if len(outs) < 3:
        return None
    keys = set.intersection(*[set(o) for o in outs])
    res = []
    for k in range(len(outs)):
        others = [o for m, o in enumerate(outs) if m != k]
        loss = sum(1 for p in keys if all(o[p] == 1 for o in others) and outs[k][p] == 0)
        gain = sum(1 for p in keys if all(o[p] == 0 for o in others) and outs[k][p] == 1)
        res.append((loss, gain))
    return dict(pairs=len(keys), mean_loss_vs_other_two_unanimous=float(np.mean([r[0] for r in res])),
                mean_gain_vs_other_two_unanimous=float(np.mean([r[1] for r in res])), per_rep=res)


def flip_rows(res):
    info = res['info']
    specs = C.a_reference_specs(info)
    refs = [C.journal_outcomes(s) for s in specs]
    rows = []
    for e in res['episodes']:
        p = (e['task'], e['init'])
        ys = [r[p] for r in refs if p in r]
        k, n = sum(ys), len(ys)
        maj = None if n == 0 else int(k * 2 > n) if n % 2 else (int(k * 2 > n) if k * 2 != n else None)
        flip = None if maj is None else ('loss' if maj == 1 and e['Y'] == 0 else 'gain' if maj == 0 and e['Y'] == 1
                                         else 'same_success' if e['Y'] == 1 else 'same_fail')
        rows.append(dict(arm=res['arm'], cell=info['cell'], variant=info['variant'], model=info['model'],
                         task=e['task'], init=e['init'], Y=e['Y'], A_k=k, A_n=n, A_maj=maj, flip=flip,
                         A_unanimous=int(n > 0 and k in (0, n)), termination=e['termination'], N=e['N'], V=e['V'],
                         M=e['M'], **{c: e[c] for c in LEVER_COLS},
                         first_lever=e['first_lever'], first_lever_step=e['first_lever_step'],
                         first_lever_frac=e.get('first_lever_frac'), first_lever_label=e['first_lever_label'],
                         first_lever_macro=e['first_lever_macro'], first_lever_cls=e['first_lever_cls'],
                         final_label=e['final_label'], final_macro=e['final_macro'], max_macro=e['max_macro'],
                         lever_labels=e['lever_labels'], refused_labels=e['refused_labels']))
    return rows, specs


@lru_cache(maxsize=8)
def _streams(run_root, arm):
    A = C.load_arm(run_root, arm, keep_heavy=True)
    return {(e['task'], e['init']): e['decisions'] for e in A['episodes']}


def _first_div(a, b):
    for s in range(min(len(a), len(b))):
        x, y = a[s], b[s]
        hx = x.get('served_head') if x.get('served_head') is not None else x.get('a_exec')
        hy = y.get('served_head') if y.get('served_head') is not None else y.get('a_exec')
        if bool(x.get('vision')) != bool(y.get('vision')) or hx is None or hy is None:
            return s
        hx, hy = np.asarray(hx, float), np.asarray(hy, float)
        if hx.shape != hy.shape or not np.allclose(hx, hy, rtol=0, atol=1e-6):
            return s
    return None if len(a) == len(b) else min(len(a), len(b))


def cross_run(run_root, res, rows, specs):
    """First decision where the arm's stream departs from A replicate 1 (bitwise served head / vision flag), versus
    the arm's first lever event; and A replicate 1 versus replicate 2 as the determinism baseline.
    Meaningful only where A replicates are step-identical (GR00T); on pi0.5 the encoder is not deterministic."""
    ref = [(C.ROOT / sp.split(':')[0], sp.split(':')[1]) for sp in specs]
    arm_s = _streams(run_root, res['arm'])
    a1 = _streams(*ref[0])
    a2 = _streams(*ref[1]) if len(ref) > 1 else None
    cat, base = Counter(), Counter()
    by_row = {(r['task'], r['init']): r for r in rows}
    for p, ds in arm_s.items():
        if p not in a1:
            continue
        div = _first_div(ds, a1[p])
        lever = by_row[p]['first_lever_step'] if p in by_row else None
        by_row[p]['xrun_first_div'] = div
        if a2 is not None and p in a2:
            base['A1_A2_identical' if _first_div(a1[p], a2[p]) is None else 'A1_A2_differ'] += 1
        if lever is None:
            cat['no_lever_identical' if div is None else 'no_lever_but_differs'] += 1
        elif div is None:
            cat['lever_but_identical'] += 1
        else:
            cat['div_eq_lever' if div == lever else 'div_before_lever' if div < lever else 'div_after_lever'] += 1
    return dict(reference=specs[0], categories=dict(cat), a_rep_baseline=dict(base))


def summarize(rows, specs):
    by = defaultdict(list)
    for r in rows:
        by[r['flip']].append(r)
    out = dict(reference=specs, pairs=len(rows), sr=float(np.mean([r['Y'] for r in rows])) if rows else None,
               counts={k: len(v) for k, v in by.items()},
               loss_vs_A_unanimous=sum(1 for r in by['loss'] if r['A_unanimous']),
               gain_vs_A_unanimous=sum(1 for r in by['gain'] if r['A_unanimous']),
               a_self_noise=a_self_noise(specs))
    for kind in ('loss', 'gain', 'same_success', 'same_fail'):
        rs = by.get(kind, [])
        if not rs:
            continue
        out[kind] = dict(n=len(rs), termination=dict(Counter(r['termination'] for r in rs)),
                         first_lever=dict(Counter(r['first_lever'] for r in rs)),
                         first_lever_macro=dict(Counter(r['first_lever_macro'] for r in rs)),
                         first_lever_cls=dict(Counter(r['first_lever_cls'] for r in rs)),
                         first_lever_frac_median=float(np.median([r['first_lever_frac'] for r in rs if r['first_lever_frac'] is not None]))
                         if any(r['first_lever_frac'] is not None for r in rs) else None,
                         final_macro=dict(Counter(r['final_macro'] for r in rs)),
                         final_label=dict(Counter(r['final_label'] for r in rs).most_common(8)),
                         max_macro=dict(Counter(r['max_macro'] for r in rs)),
                         mean_per_anchor={c: float(np.mean([r[c] / max(r['V'], 1) for r in rs])) for c in LEVER_COLS},
                         mean_counts={c: float(np.mean([r[c] for r in rs])) for c in LEVER_COLS},
                         mean_decisions=float(np.mean([r['N'] for r in rs])))
    return out


def sf_uf_join(rows):
    """Per cell: SF1 x UF1 outcomes on common pairs and UF's SF-refused extensions by joint class."""
    per = defaultdict(dict)
    for r in rows:
        if r['variant'] in ('SF1', 'UF1'):
            per[(r['cell'], r['variant'])][(r['task'], r['init'])] = r
    out = {}
    for cell in sorted({c for c, _ in per}):
        sf, uf = per.get((cell, 'SF1')), per.get((cell, 'UF1'))
        if not sf or not uf:
            continue
        common = sorted(set(sf) & set(uf))
        tab = Counter((sf[p]['Y'], uf[p]['Y']) for p in common)
        cls = defaultdict(list)
        for p in common:
            key = 'uf_only_loss' if sf[p]['Y'] == 1 and uf[p]['Y'] == 0 else 'sf_only_loss' if sf[p]['Y'] == 0 and uf[p]['Y'] == 1 \
                else 'both_success' if sf[p]['Y'] == 1 else 'both_fail'
            cls[key].append(p)
        out[cell] = dict(pairs=len(common), sf1_uf1_table={f'SF{a}_UF{b}': v for (a, b), v in sorted(tab.items())},
                         uf_refused_per_anchor={k: float(np.mean([uf[p]['sf_refused_ext'] / max(uf[p]['V'], 1) for p in v]))
                                                for k, v in cls.items()},
                         uf_ext_per_anchor={k: float(np.mean([uf[p]['ext_blocks'] / max(uf[p]['V'], 1) for p in v]))
                                            for k, v in cls.items()},
                         uf_only_loss_A_k=dict(Counter(uf[p]['A_k'] for p in cls.get('uf_only_loss', []))),
                         sf_only_loss_A_k=dict(Counter(sf[p]['A_k'] for p in cls.get('sf_only_loss', []))),
                         uf_only_loss_final_macro=dict(Counter(uf[p]['final_macro'] for p in cls.get('uf_only_loss', []))),
                         uf_only_loss_refused_labels=dict(sum((Counter(uf[p]['refused_labels']) for p in cls.get('uf_only_loss', [])),
                                                              Counter()).most_common(10)),
                         both_success_refused_labels=dict(sum((Counter(uf[p]['refused_labels']) for p in cls.get('both_success', [])),
                                                              Counter()).most_common(10)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', default='eval', help='eval | profile | <run root path>')
    p.add_argument('--arms', nargs='*')
    p.add_argument('--family', nargs='*')
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--xrun', choices=('groot', 'all', 'none'), default='groot',
                   help='cross-run first-divergence check versus A replicate 1 (default: GR00T arms only)')
    p.add_argument('--out', type=Path, default=C.OUT)
    p.add_argument('--detail', type=Path, default=C.DETAIL)
    a = p.parse_args()
    run_root, arms, skipped = C.cli_arms(a)
    arms = [x for x in arms if C.parse_arm(x)['family'] != 'A']
    results = S.run_many(run_root, arms, a.jobs)
    allrows, summaries = [], {}
    for res in results:
        rows, specs = flip_rows(res)
        xr = None
        if a.xrun == 'all' or (a.xrun == 'groot' and res['info']['model'] == 'groot'):
            xr = cross_run(run_root, res, rows, specs)
        allrows += rows
        summaries[res['arm']] = summarize(rows, specs)
        s = summaries[res['arm']]
        s['cross_run'] = xr
        print(f"{res['arm']:32s} pairs={s['pairs']} SR={s['sr']:.3f} {s['counts']} lossA3/3={s['loss_vs_A_unanimous']} "
              f"gainA0/3={s['gain_vs_A_unanimous']} Anoise={None if not s['a_self_noise'] else (round(s['a_self_noise']['mean_loss_vs_other_two_unanimous'],1), round(s['a_self_noise']['mean_gain_vs_other_two_unanimous'],1))}",
              flush=True)
        if xr:
            print('   xrun', xr['categories'], 'A1 vs A2', xr['a_rep_baseline'], flush=True)
    tag = run_root.name + ('_subset' if (a.arms or a.family) else '')
    C.write_json(a.out / f'a2_flips_{tag}.json', dict(schema='r7.a2.flips.v1', run=str(run_root), analysed=arms,
                                                      skipped=skipped, arms=summaries, sf_uf=sf_uf_join(allrows)))
    C.write_csv(a.detail / f'a2_flips_{tag}.csv.gz', allrows)
    print('skipped (not complete):', skipped)


if __name__ == '__main__':
    main()
