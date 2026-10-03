"""R8 E4 preliminary evidence: the cost of not looking, measured on A's own path.

For every A look (vision anchor) that is followed by A's normal blind block and then another look, compare
  * the block SF1/UF1 would have served at that second look (their one extra follow block), rebuilt exactly from
    the anchor's 16 rows/weights and the library (pi0.5: next^2 member heads; GR00T: anchor chunk controls 10:15),
  * the block A actually served after looking again (served_head of the fresh look).
Same observation, same robot state; only "look again" vs "keep following" differs. Also records whether the
fresh look re-identifies the followed demonstrations, the frozen C1 stage class of the anchor, and SF1/UF1 grant.
Outcome link: per (task, init) pair, SF1/UF1 test outcomes (r07_main) vs A's three replicates.

Read-only. Outputs: /tmp/r8_E4_lazy_levers/fvl_<cell>.csv.gz and fvl_summary.json in this directory.
Run (repo root):
  taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python \
    exp/offline_search/rounds/r08/ideation/E4_lazy_levers/follow_vs_look.py --cells all
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from exp.offline_search.rounds.r07.analysis_scripts import a2_common as C
from exp.offline_search.rounds.r07.c1_follow.methods import FollowExtension

HERE = Path(__file__).resolve().parent
SCRATCH = Path('/tmp/r8_E4_lazy_levers')
STORE = Path('/home/weiland/trace_runs/offline_search_store/library')
FITS = Path('/home/weiland/trace_runs/os_closed_loop/r07_main/fits')
LIBNAME = {('pi05', 50): 'current', ('pi05', 500): 'bpool_cs', ('groot', 50): 'current', ('groot', 500): 'bpool_all'}


def table_for(cell):
    o = pickle.load(open(FITS / f'r7_{cell}_SF1.pkl', 'rb'))
    return o['method'].follow_table


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    # average ties
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def strat_auroc(rows, key, label):
    """Task-stratified AUROC (pairs within task only), and pooled."""
    num = den = 0.0
    by_t = defaultdict(list)
    for r in rows:
        if r.get(key) is not None:
            by_t[r['task']].append(r)
    for t, rr in by_t.items():
        pos = [r[key] for r in rr if r[label]]
        neg = [r[key] for r in rr if not r[label]]
        a = auroc(pos, neg)
        if a is not None:
            w = len(pos) * len(neg)
            num += a * w
            den += w
    pooled = auroc([r[key] for r in rows if r.get(key) is not None and r[label]],
                   [r[key] for r in rows if r.get(key) is not None and not r[label]])
    return (num / den if den else None), pooled, int(sum(1 for r in rows if r.get(key) is not None and r[label]))


def analyze(cell):
    info = C.parse_arm(f'r7_{cell}_SF1')
    model, size = info['model'], info['size']
    T = table_for(cell)
    lib = STORE / f"{model}_{info['suite']}" / LIBNAME[(model, size)]
    act = np.load(lib / 'action.npy', mmap_mode='r')
    episode = np.load(lib / 'episode.npy')
    succ = np.load(lib / 'success.npy')
    prog = np.load(lib / 'progress.npy').astype(np.float64)
    ueps = np.unique(episode)
    good = np.flatnonzero(succ.astype(bool))
    sig = act[good][:, :5, :7].reshape(-1, 7).astype(np.float64).std(0)
    sig[sig < 1e-6] = 1.0
    gdim = int(T.manifest['gripper_dim'])
    sf = FollowExtension(T, extend_blocks=1, stage_gate=True, state_valve=True)
    a_specs, _ = __import__('exp.offline_search.rounds.r06.analysis_scripts.common', fromlist=['ab_arms']).ab_arms(model, info['r6cell'])
    run, arm = a_specs[0].split(':')
    A = C.load_arm(C.ROOT / run, arm, keep_heavy=True)
    rows_out = []
    per_ep = []
    for e in A['episodes']:
        ds = {int(d['step']): d for d in e['decisions']}
        ep_rows = []
        for s, d in sorted(ds.items()):
            if not (d.get('vision') and d.get('hit') and d.get('rows') and len(d['rows']) == 16):
                continue
            b, l2 = ds.get(s + 1), ds.get(s + 2)
            if b is None or l2 is None or b.get('vision') or not l2.get('vision') or not l2.get('hit') \
                    or not l2.get('served_head') or not l2.get('rows'):
                continue
            rows = np.asarray(d['rows'], np.int64)
            w = np.asarray(d['weights'], np.float64)
            if rows.max() >= len(T.mode):
                continue
            plan = sf.plan({'rows': rows, 'weights': w.astype(np.float32)})
            st = C.stage_of(T, rows, w)
            rec = dict(uid=e['uid'], task=e['task'], init=e['init'], step=s, cls=st['cls'], macro=st['macro'],
                       structural=bool(plan.structural), sf_grant=bool(plan.blocks), anchor_index=len(ep_rows))
            arm_dims = [k for k in range(7) if k != gdim]
            b3, l4 = ds.get(s + 3), ds.get(s + 4)
            look4 = (b3 is not None and l4 is not None and not b3.get('vision') and l4.get('vision') and l4.get('hit')
                     and l4.get('served_head') and l4.get('rows'))
            for age, look in ((2, l2), (4, l4 if look4 else None)):
                if look is None:
                    continue
                fut = T.advance(rows, age)
                heads = None
                if model == 'groot' and age == 2:
                    heads = act[rows, 10:15, :7].astype(np.float64) if plan.structural else None  # SF1's GR00T source
                elif not (fut < 0).any():
                    heads = act[fut, :5, :7].astype(np.float64)
                if heads is None:
                    continue
                head_f = np.tensordot(w, heads, 1)
                head_l = np.asarray(look['served_head'], np.float64)[:5, :7]
                z = (head_f - head_l) / sig
                wn = w / w.sum()
                spread = np.sqrt(np.tensordot(wn, (((heads - head_f) / sig)[:, :, arm_dims] ** 2).mean((1, 2)), 1))
                sfx = '' if age == 2 else '_a4'
                rec['d_arm' + sfx] = float(np.sqrt(np.mean(z[:, arm_dims] ** 2)))
                rec['spread' + sfx] = float(spread)
                rec['grip_flip' + sfx] = bool(np.sign(head_f[:, gdim].mean()) != np.sign(head_l[:, gdim].mean()))
                vf, vl = head_f[:, :3].sum(0), head_l[:, :3].sum(0)
                nf, nl = np.linalg.norm(vf), np.linalg.norm(vl)
                rec['cos_xyz' + sfx] = float(vf @ vl / (nf * nl)) if nf > 1e-9 and nl > 1e-9 else None
                r2 = np.asarray(look['rows'], np.int64)
                w2 = np.asarray(look['weights'], np.float64)
                w2 = w2 / w2.sum() if w2.sum() > 0 else w2
                srcr = fut if not (fut < 0).any() else rows
                pf = float(np.tensordot(wn, prog[srcr], 1)) + (0.0 if not (fut < 0).any() else np.nan)
                pl = float(np.tensordot(w2, prog[r2], 1))
                rec['prog_slip' + sfx] = pl - pf
                topf = np.bincount(np.searchsorted(ueps, episode[rows]), weights=wn, minlength=len(ueps)).argmax()
                topl = np.bincount(np.searchsorted(ueps, episode[r2]), weights=w2, minlength=len(ueps)).argmax()
                rec['top_demo_same' + sfx] = bool(topf == topl)
            ep_rows.append(rec)
        rows_out += ep_rows
        per_ep.append((e, ep_rows))
    # outcomes
    Aout = [C.journal_outcomes(s) for s in a_specs]
    SF = C.journal_outcomes(f'r07_main:r7_{cell}_SF1')
    UF = C.journal_outcomes(f'r07_main:r7_{cell}_UF1')
    ep_summ = []
    K = 12  # first 12 looks (~120 controls): limits the length/failure confound
    for e, rr in per_ep:
        k = (e['task'], e['init'])
        amaj = sum(a.get(k, 0) for a in Aout) >= 2
        early = [r for r in rr if r['anchor_index'] < K]
        def m(sel, key='d_arm'):
            v = [r[key] for r in sel if r.get(key) is not None]
            return float(np.mean(v)) if v else None
        ep_summ.append(dict(task=e['task'], init=e['init'], A_maj=int(amaj), A_rep1=e['Y'], SF1=SF.get(k), UF1=UF.get(k),
                            sf_loss=int(amaj and SF.get(k) == 0), uf_loss=int(amaj and UF.get(k) == 0),
                            d_sf_early=m([r for r in early if r['sf_grant']]),
                            d_uf_early=m([r for r in early if r['structural']]),
                            flip_uf_early=m([r for r in early if r['structural']], 'grip_flip'),
                            n_sf_early=sum(r['sf_grant'] for r in early), n_uf_early=sum(r['structural'] for r in early)))
    C.write_csv(SCRATCH / f'fvl_{cell}.csv.gz', rows_out)
    C.write_csv(SCRATCH / f'fvl_ep_{cell}.csv.gz', ep_summ)
    # summaries
    def agg(sel):
        d = np.array([r['d_arm'] for r in sel if r.get('d_arm') is not None])
        g = np.array([r['grip_flip'] for r in sel if r.get('d_arm') is not None], float)
        sm = np.array([r['top_demo_same'] for r in sel if r.get('d_arm') is not None], float)
        sp = np.array([r['spread'] for r in sel if r.get('d_arm') is not None])
        ps = np.array([r['prog_slip'] for r in sel if r.get('d_arm') is not None])
        cs = np.array([r['cos_xyz'] for r in sel if r.get('cos_xyz') is not None])
        if not len(d):
            return dict(n=0)
        return dict(n=int(len(d)), d_p50=float(np.median(d)), d_p90=float(np.quantile(d, .9)), d_mean=float(d.mean()),
                    grip_flip=float(g.mean()), top_demo_same=float(sm.mean()),
                    spread_p50=float(np.median(sp)), d_over_spread_p50=float(np.median(d / np.maximum(sp, 1e-9))),
                    prog_slip_p50=float(np.nanmedian(ps)), prog_slip_absp90=float(np.nanquantile(np.abs(ps), .9)),
                    cos_xyz_p10=float(np.quantile(cs, .1)) if len(cs) else None,
                    frac_cos_neg=float((cs < 0).mean()) if len(cs) else None)
    out = dict(cell=cell, a_arm=a_specs[0], anchors_paired=len(rows_out),
               structural_share=float(np.mean([r['structural'] for r in rows_out])),
               sf_grant_share=float(np.mean([r['sf_grant'] for r in rows_out])),
               all_supported=agg([r for r in rows_out if r['structural']]),
               sf_granted=agg([r for r in rows_out if r['sf_grant']]),
               uf_only=agg([r for r in rows_out if r['structural'] and not r['sf_grant']]),
               by_class={c: agg([r for r in rows_out if r['structural'] and r['cls'] == c])
                         for c in ('interior', 'event', 'mixed', 'unknown')},
               common_cohort_age2_vs_age4={
                   'age2': agg([r for r in rows_out if r.get('d_arm') is not None and r.get('d_arm_a4') is not None]),
                   'age4': agg([{**r, 'd_arm': r['d_arm_a4'], 'grip_flip': r['grip_flip_a4'], 'cos_xyz': r['cos_xyz_a4'],
                                 'top_demo_same': r['top_demo_same_a4'], 'spread': r['spread_a4'], 'prog_slip': r['prog_slip_a4']}
                                for r in rows_out if r.get('d_arm') is not None and r.get('d_arm_a4') is not None])},
               by_macro={mc: agg([r for r in rows_out if r['structural'] and r['macro'] == mc])
                         for mc in ('S0', 'S1', 'S2', 'S3', 'S4+')})
    ok = [r for r in ep_summ if r['A_maj'] == 1]
    for lever, key in (('sf', 'd_sf_early'), ('uf', 'd_uf_early')):
        s, p, npos = strat_auroc([r for r in ok if r[f'{lever}1'.upper()] is not None] if False else ok, key, f'{lever}_loss')
        out[f'{lever}_loss_auroc_task_strat'] = s
        out[f'{lever}_loss_auroc_pooled'] = p
        out[f'{lever}_losses_among_A_success'] = npos
    s, p, npos = strat_auroc(ok, 'flip_uf_early', 'uf_loss')
    out['uf_loss_auroc_gripflip_task_strat'] = s
    # control: does the same statistic predict A's own replicate-1 failure among A-majority successes?
    for r in ok:
        r['a1_fail'] = int(r['A_rep1'] == 0)
    s, p, npos = strat_auroc(ok, 'd_uf_early', 'a1_fail')
    out['a_rep1_fail_auroc_task_strat'] = s
    out['a_rep1_fails_among_A_success'] = npos
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', nargs='+', default=['all'])
    a = ap.parse_args()
    cells = C.CELLS8 if a.cells == ['all'] else a.cells
    SCRATCH.mkdir(parents=True, exist_ok=True)
    res = {}
    outp = HERE / 'fvl_summary.json'
    if outp.exists():
        res = json.loads(outp.read_text())
    for cell in cells:
        res[cell] = analyze(cell)
        C.write_json(outp, res)
        print(cell, json.dumps({k: res[cell][k] for k in ('anchors_paired', 'structural_share', 'sf_grant_share')}),
              json.dumps(res[cell]['all_supported']), flush=True)


if __name__ == '__main__':
    main()
