"""CPU-only five-model closed-loop analysis (1018b).

Inputs: collected A100 per-episode pairing files (base 100-episode arms and
preregistered extension blocks) and the existing 4090-vs-H100 formal pairs.
Outputs one evidence JSON. Statistics: exact two-sided McNemar and the
preregistered conservative exact 95% CI (paired_sr.paired_interval: Bonferroni
union of per-cell Clopper-Pearson), margin +-2 pp. Cold starts are never
merged into one larger N; disjoint UID blocks of the same cold index are.
"""
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paired_sr import paired_interval  # noqa: E402

COL = HERE / 'a100_closedloop_1018/collected'
OTHER_SUITE = {('dp', 'x2'): 'libero10'}
FORMAL_GPU = {('pi05', 'pure'): '4090', ('pi05', 'cache'): '4090', ('groot', 'pure'): 'H100',
              ('groot', 'cache'): 'H100', ('dp', 'pure'): '4090', ('dp', 'cache'): '4090',
              ('fastwam', 'pure'): 'H100', ('fastwam', 'cache'): 'H100', ('cosmos', 'pure'): 'H100',
              ('cosmos', 'cache'): 'H100', ('dp', 'pure_libero10'): '4090'}
SOURCES = {('pi05', 'pure'): 'tpi7_0536', ('pi05', 'cache'): 'pimk1_fuld (crg_k50, n250)',
           ('groot', 'pure'): 'tgr7h1_full', ('groot', 'cache'): 'ff1015_groot_mt_10 (n1000/c)',
           ('dp', 'pure'): 'hs7dpctl_full', ('dp', 'cache'): 'ff1015_dp_li_14_u (n50/c)',
           ('fastwam', 'pure'): 'hs6fw_res_152931', ('fastwam', 'cache'): 'ff1015_fastwam_li_18 (n50/c)',
           ('cosmos', 'pure'): 'hs2_cosmos_h100_k2incl_c16_t107', ('cosmos', 'cache'): 'fa_co_r_u_l (n50/crg_k50)',
           ('dp', 'pure_libero10'): 'hs7dpctl_full (libero_10)'}


def mcnemar_discordance(x, y):
    """Exact McNemar on two paired discordance indicators (x vs y)."""
    b = sum(1 for i, j in zip(x, y) if i and not j)
    c = sum(1 for i, j in zip(x, y) if j and not i)
    p = float(binomtest(b, b + c, .5).pvalue) if b + c else 1.
    return dict(only_first=b, only_second=c, exact_mcnemar_p=p)


def rows_for(model):
    """{arm: {'uids': [...], 'formal': [...], 'cold1': [...], 'cold2': [...], 'blocks': [...]}}"""
    out = {}
    base = COL / f'{model}_PAIRED.json'
    if base.exists():
        for r in json.loads(base.read_text()):
            pe = r['per_episode']
            out[r['arm']] = dict(uids=[e['uid'] for e in pe], formal=[e['formal'] for e in pe],
                                 cold1=[e['cold1'] for e in pe], cold2=[e['cold2'] for e in pe],
                                 blocks=[dict(block='base', episodes=len(pe), colds=[1, 2])])
    for f in sorted(glob.glob(str(COL / f'{model}_EXT_*_PAIRED.json'))):
        for r in json.loads(Path(f).read_text()):
            arm = r['arm'].split('_')[0]
            if r['arm'].split('_')[1].startswith('r'):
                continue  # same-card replicate blocks: see replicates()
            suite = OTHER_SUITE.get((model, r['arm'].split('_')[1]))
            if suite:
                # A different suite is its own preregistered population (no pooling across suites).
                pe = r['per_episode']
                if sorted(r['colds']) == [1, 2]:
                    out[arm + '_' + suite] = dict(uids=[e['uid'] for e in pe], formal=[e['formal'] for e in pe],
                        cold1=[e['cold1'] for e in pe], cold2=[e['cold2'] for e in pe],
                        blocks=[dict(block=r['arm'], episodes=len(pe), colds=r['colds'])])
                else:
                    out.setdefault(arm + '_single_cold_blocks', []).append({k: v for k, v in r.items() if k != 'per_episode'})
                continue
            if arm not in out or sorted(r['colds']) != [1, 2]:
                # Only blocks with both cold starts enter the pooled per-cold series;
                # single-cold blocks are reported separately.
                out.setdefault(arm + '_single_cold_blocks', []).append(
                    {k: v for k, v in r.items() if k != 'per_episode'})
                continue
            pe = r['per_episode']
            have = set(out[arm]['uids'])
            assert not have & {e['uid'] for e in pe}, 'blocks must be disjoint'
            for k in ('formal', 'cold1', 'cold2'):
                out[arm][k] += [e[k] for e in pe]
            out[arm]['uids'] += [e['uid'] for e in pe]
            out[arm]['blocks'].append(dict(block=r['arm'], episodes=len(pe), colds=r['colds']))
    return out


def replicates(model):
    """Extra A100 cold starts (r blocks) on the same UIDs as base/x1: all same-card pairs."""
    import itertools
    runs = {}   # uid -> {label: outcome}, formal
    formal = {}
    base = COL / f'{model}_PAIRED.json'
    out = []
    if not base.exists():
        return out
    series = {}
    for r in json.loads(base.read_text()):
        for e in r['per_episode']:
            series.setdefault(r['arm'], {})[e['uid']] = dict(formal=e['formal'], c1=e['cold1'], c2=e['cold2'])
    for f in sorted(glob.glob(str(COL / f'{model}_EXT_*_PAIRED.json'))):
        for r in json.loads(Path(f).read_text()):
            arm, blk = r['arm'].split('_')[0], r['arm'].split('_')[1]
            for e in r['per_episode']:
                d = series.setdefault(arm, {}).setdefault(e['uid'], dict(formal=e['formal']))
                if blk.startswith('x'):
                    d.update({'c1': e.get('cold1'), 'c2': e.get('cold2')})
                elif blk.startswith('r'):
                    d.update({'c3': e.get('cold1'), 'c4': e.get('cold2')})
    for arm, rows in series.items():
        uids = [u for u, d in rows.items() if all(d.get(k) is not None for k in ('c1', 'c2', 'c3', 'c4'))]
        if not uids:
            continue
        pairs = {}
        for a, b in itertools.combinations(('c1', 'c2', 'c3', 'c4'), 2):
            x = [rows[u][a] for u in uids]; y = [rows[u][b] for u in uids]
            pairs[a + '-' + b] = paired_interval(x, y)
        cross = {c: paired_interval([rows[u][c] for u in uids], [rows[u]['formal'] for u in uids]) for c in ('c1', 'c2', 'c3', 'c4')}
        same_disc = [sum(rows[u][a] != rows[u][b] for u in uids) for a, b in itertools.combinations(('c1', 'c2', 'c3', 'c4'), 2)]
        cross_disc = [sum(rows[u][c] != rows[u]['formal'] for u in uids) for c in ('c1', 'c2', 'c3', 'c4')]
        out.append(dict(model=model, arm=arm, episodes=len(uids), same_card_pairs=pairs, a100_minus_formal=cross,
                        same_card_discordance_mean=sum(same_disc) / len(same_disc), cross_card_discordance_mean=sum(cross_disc) / len(cross_disc),
                        same_card_discordance=same_disc, cross_card_discordance=cross_disc,
                        successes={k: sum(rows[u][k] for u in uids) for k in ('formal', 'c1', 'c2', 'c3', 'c4')}))
    return out


def summarise(model, arm, d):
    n = len(d['uids'])
    f, c1, c2 = d['formal'], d['cold1'], d['cold2']
    same = paired_interval(c1, c2)
    x1 = paired_interval(c1, f)
    x2 = paired_interval(c2, f)
    disc_same = [int(a != b) for a, b in zip(c1, c2)]
    disc_x1 = [int(a != b) for a, b in zip(c1, f)]
    disc_x2 = [int(a != b) for a, b in zip(c2, f)]
    return dict(model=model, arm=arm, formal_gpu=FORMAL_GPU[(model, arm)], formal_source=SOURCES[(model, arm)],
                episodes=n, blocks=d['blocks'],
                successes=dict(formal=sum(f), a100_cold1=sum(c1), a100_cold2=sum(c2)),
                same_card_cold1_vs_cold2=same,
                a100_cold1_minus_formal=x1, a100_cold2_minus_formal=x2,
                discordance=dict(same_card=sum(disc_same), cross_cold1=sum(disc_x1), cross_cold2=sum(disc_x2), n=n),
                cross_vs_same_discordance=dict(cold1=mcnemar_discordance(disc_x1, disc_same),
                                               cold2=mcnemar_discordance(disc_x2, disc_same)))


def ci_halfwidth(n, b, c):
    r = paired_interval([1] * b + [0] * c + [0] * (n - b - c), [0] * b + [1] * c + [0] * (n - b - c))
    return r['CI95']


def required_n(d, power_sims=2000, seed=0):
    """Smallest N (grid) whose conservative exact CI fits +-2pp (true diff 0)."""
    grid = [100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 8000, 10000, 15000, 20000]
    expected = None
    for n in grid:
        b = c = int(round(d * n / 2))
        lo, hi = ci_halfwidth(n, b, c)
        if lo >= -.02 and hi <= .02:
            expected = n
            break
    rng = np.random.default_rng(seed)
    power = {}
    for n in grid:
        if expected is not None and n < expected:
            continue
        ok = 0
        cells = rng.multinomial(n, [d / 2, d / 2, 1 - d], size=power_sims)
        cache = {}
        for b, c, _ in cells:
            key = (int(b), int(c))
            if key not in cache:
                lo, hi = ci_halfwidth(n, *key)
                cache[key] = lo >= -.02 and hi <= .02
            ok += cache[key]
        power[n] = ok / power_sims
        if power[n] >= .8:
            break
    n80 = next((n for n, p in power.items() if p >= .8), None)
    return dict(discordance=d, n_expected_case=expected, n_power80=n80, power_by_n=power)


def main():
    out = dict(schema='xgpu.a100.closedloop.analysis.v1', margin_pp=2,
               method='conservative exact 95% CI (Bonferroni union of per-cell Clopper-Pearson), exact two-sided McNemar',
               models=[], pooled={}, existing_4090_vs_h100=[], sample_size={})
    tot = dict(n=0, same=0, cross1=0, cross2=0)
    for model in ('pi05', 'groot', 'dp', 'fastwam', 'cosmos'):
        rows = rows_for(model)
        for arm in ('pure', 'cache', 'pure_libero10'):
            if arm in rows:
                s = summarise(model, arm, rows[arm])
                out['models'].append(s)
                dd = s['discordance']
                tot['n'] += dd['n']; tot['same'] += dd['same_card']
                tot['cross1'] += dd['cross_cold1']; tot['cross2'] += dd['cross_cold2']
            if arm + '_single_cold_blocks' in rows:
                out.setdefault('single_cold_blocks', []).extend(rows[arm + '_single_cold_blocks'])
    out['replicates'] = [r for m in ('pi05', 'groot', 'dp', 'fastwam', 'cosmos') for r in replicates(m)]
    out['pooled'] = dict(tot, note='episode-level totals across all arms; descriptive, arms differ in model/GPU/recipe')
    ex = json.loads((HERE / 'evidence/existing_formal_pairs_1018.json').read_text())
    for r in ex['descriptive_pairs'] + ex['partial_descriptive_pairs']:
        st = r.get('stats_on_available_pairs', r)
        out['existing_4090_vs_h100'].append(dict(
            left_4090=r['left'], right_h100=r['right'], suite=r['suite'], num_steps=r.get('num_steps'),
            pairs=st['pairs'], only_4090=st['only_a'], only_h100=st['only_b'], diff_pp=100 * st['difference'],
            ci95_pp=[100 * x for x in st['CI95']], mcnemar_p=st['mcnemar_p'],
            identity_differences=sorted(r.get('identity_differences', {})), uid_equal=r.get('uid_equal'),
            missing_left=r.get('missing_left')))
    rates = sorted({.01, .02, .03, .05, .10} | {round(m['discordance']['same_card'] / m['discordance']['n'], 3)
                                                 for m in out['models'] if m['discordance']['same_card']})
    out['sample_size'] = [required_n(d) for d in rates]
    target = HERE / 'evidence/a100_closedloop_analysis_1018b.json'
    target.write_text(json.dumps(out, indent=1) + '\n')
    for m in out['models']:
        s, a, b = m['same_card_cold1_vs_cold2'], m['a100_cold1_minus_formal'], m['a100_cold2_minus_formal']
        print(f"{m['model']:8s} {m['arm']:5s} n={m['episodes']:4d} SR f/c1/c2={m['successes']['formal']}/{m['successes']['a100_cold1']}/{m['successes']['a100_cold2']}"
              f" same {s['only_a']}/{s['only_b']} c1-f {a['only_a']}/{a['only_b']} {100*a['difference']:+.2f} [{100*a['CI95'][0]:+.2f},{100*a['CI95'][1]:+.2f}] p={a['mcnemar_p']:.3g}"
              f" c2-f {b['only_a']}/{b['only_b']} {100*b['difference']:+.2f} [{100*b['CI95'][0]:+.2f},{100*b['CI95'][1]:+.2f}] p={b['mcnemar_p']:.3g}")
    print(json.dumps(out['pooled']))
    for r in out['sample_size']:
        print(r['discordance'], r['n_expected_case'], r['n_power80'])


if __name__ == '__main__':
    main()
