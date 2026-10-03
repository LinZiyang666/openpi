"""Identity-first recipe/P10 episode comparison. No serving changes or fitting.

All step indices are five-control decisions. Phases describe commands, not
object state. Fresh recipe dnn and debug d1 have different scales.
"""
from collections import Counter
import json
import numpy as np
import pandas as pd
from .safe import HERE, RUNS, STORE, admitted, admitted_jsonl, json_identity, owned, dump

ROOTS = {'groot': 'r09_recipe_full_g', 'pi05': 'r09_recipe_full_p'}


def accepted(path):
    out = {}
    for r in admitted_jsonl(path):
        if r.get('accepted') and r.get('status') in ('done', 'failed') and not r.get('error'):
            key = json_identity(json.dumps(r))
            if key in out and out[key] != r:
                raise ValueError('multiple accepted terminal records')
            out[key] = r
    if set(out) != {(t, i) for t in range(10) for i in range(30)}:
        raise ValueError('expected exactly 300 admitted completed episodes')
    return out


def compact(model, variant, fields):
    arm = f'r8_{model}_l10_' + (variant if variant == 'P10' else f'50_{variant}')
    p = admitted(STORE / 'derived/r09_astra/compact' / (arm + '.npz'))
    with np.load(p, allow_pickle=False) as z:
        task, init = z['task'], z['init']
        if (task.shape != init.shape or task.ndim != 1 or task.dtype.kind not in 'iu'
                or init.dtype.kind not in 'iu' or not np.all((task >= 0) & (task < 10) & (init >= 0) & (init < 30))):
            raise ValueError('mixed or unidentifiable compact archive; payload not admitted')
        out = {k: z[k] for k in fields}
        out.update(task=task, init=init)
    return out


def recipe(model):
    from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler
    p = HERE.parents[1] / 'recipe/artifacts' / f'r9eq_{model}_l10_50.pkl'
    with admitted(p).open('rb') as f:
        m = FitUnpickler(f).load()['method']
    assert m.inner.base.head_meta['train_inits'] == list(range(20))
    assert m.blend == .5 and not m.inner.base.correct_gripper
    return m


def phases(closed):
    """Retrospective command phase, with model sign handled before this function."""
    closed = np.asarray(closed, bool)
    out = np.where(closed, 'carry', 'post').astype('<U10')
    first = np.flatnonzero(closed)
    out[:first[0] if len(first) else len(out)] = 'approach'
    for j in range(1, len(out)):
        if not closed[j-1] and closed[j]:
            out[max(0, j-1):j+1] = 'grasp'
        elif closed[j-1] and not closed[j]:
            out[max(0, j-1):min(len(out), j+2)] = 'release'
    return out


def is_closed(a, model):
    a = np.asarray(a)
    return (a < 0 if model == 'groot' else a >= 0)


def standard(model, m):
    arm = f'r9eq_{model}_l10_50'
    root = RUNS / ROOTS[model] / 'runs' / arm
    acc = accepted(root / 'client/journal.jsonl')
    groups = {k: {} for k in acc}
    for path in sorted(root.glob('server_*/decisions*.jsonl')):
        for r in admitted_jsonl(path):
            if r.get('ev') != 'dec': continue
            key = (r['task_id'], r['init'])
            if r.get('attempt') != acc[key]['attempt']: continue
            seq = r['step']
            if seq in groups[key] and groups[key][seq] != r:
                raise ValueError('conflicting decision records')
            groups[key][seq] = r
    out = []
    for (task, init), group in sorted(groups.items()):
        if sorted(group) != list(range(len(group))) or not group:
            raise ValueError('missing or non-contiguous accepted episode')
        for seq, r in sorted(group.items()):
            ex = r.get('extras', {})
            a = np.asarray(r['served_head'])
            assert a.shape == (5, 7)
            fresh = bool(r['vision'])
            top = r.get('top1', -1)
            d = dict(model=model, arm='recipe', task=task, init=init, seq=seq,
                success=bool(acc[task, init]['success']), vision=fresh, call=not r['hit'],
                src=r['src'], closed=bool(is_closed(a[:, 6], model).mean() > .5),
                close_last=bool(is_closed(a[-1, 6], model)), intragrip=bool(np.any(np.diff(is_closed(a[:, 6], model)))),
                reason=ex.get('os_reason', 0), flags=ex.get('os_flags', 0),
                dnn=ex.get('dnn', np.nan) if fresh else np.nan,
                noprog=ex.get('noprog_span', ex.get('noprog_n', np.nan)) if fresh else np.nan,
                progress=float(m.inner.C.prog[top]) if fresh and top >= 0 else np.nan,
                libstep=float(m.inner.C.step[top]) if fresh and top >= 0 else np.nan,
                liblen=float(m.inner.C.ep_len[top]) if fresh and top >= 0 else np.nan,
                lag=float(seq-m.inner.C.step[top]) if fresh and top >= 0 else np.nan,
                escalation=bool(ex.get('r9o_escalated', 0)),
                correction=np.nan, grip_reconstruction=np.nan, motion=float(np.sqrt(np.mean(a[:, :6]**2))))
            # Only fresh cache heads equal the weighted first five library controls.
            if fresh and r['src'] == 'cache':
                rows, w = np.asarray(r['rows']), np.asarray(r['weights'])
                w = w/w.sum()
                b = np.einsum('k,kha->ha', w, m.inner.base.act[rows, :5, :7])
                d['correction'] = float(np.sqrt(np.mean((a[:, :6]-b[:, :6])**2)))
                d['grip_reconstruction'] = float(np.max(np.abs(a[:, 6]-b[:, 6])))
            out.append(d)
    return pd.DataFrame(out)


def pure(model):
    acc = accepted(RUNS / 'r08_main/runs' / f'r8_{model}_l10_P10/client/journal.jsonl')
    a = compact(model, 'P10', ['seq', 'served_chunk', 'vision', 'src', 'lib_step', 'decision_id'])
    assert set(a['src']) == {'policy', 'policy_tail'}
    assert np.array_equal(a['vision'], a['src'] == 'policy')
    # Compact extraction joins the accepted raw journal attempts. Verify again here.
    for task, init, did in zip(a['task'], a['init'], a['decision_id']):
        attempt = int(str(did).split(':')[0].rsplit('_a', 1)[1])
        assert attempt == acc[int(task), int(init)]['attempt']
    close = is_closed(a['served_chunk'][:, :5, 6], model)
    n = len(close)
    return pd.DataFrame(dict(model=model, arm='P10', task=a['task'], init=a['init'], seq=a['seq'],
        success=[bool(acc[int(t), int(i)]['success']) for t, i in zip(a['task'], a['init'])],
        vision=a['vision'], call=a['vision'], src=a['src'], closed=close.mean(1) > .5,
        close_last=close[:, -1], intragrip=np.any(np.diff(close, axis=1), axis=1),
        reason=np.where(a['vision'], 82, 0), flags=0, dnn=np.nan, noprog=np.nan,
        progress=np.nan, libstep=a['lib_step'], liblen=np.nan, lag=np.nan, escalation=False,
        correction=np.nan, grip_reconstruction=np.nan,
        motion=np.sqrt((a['served_chunk'][:, :5, :6]**2).mean((1, 2)))))


def annotate(df):
    df = df.sort_values(['model', 'arm', 'task', 'init', 'seq']).reset_index(drop=True)
    df['phase'] = ''; df['event'] = False
    eps, calls = [], []
    for (model, arm, task, init), g in df.groupby(['model', 'arm', 'task', 'init'], sort=False):
        assert g.seq.tolist() == list(range(len(g)))
        phase = phases(g.closed.to_numpy())
        event = np.r_[False, np.diff(g.close_last.to_numpy())] | g.intragrip.to_numpy()
        df.loc[g.index, 'phase'] = phase
        df.loc[g.index, 'event'] = event
        first_close = np.flatnonzero(g.closed.to_numpy())
        release = np.flatnonzero(np.r_[False, np.diff(g.closed.astype(int).to_numpy()) == -1])
        call_idx = np.flatnonzero(g.call.to_numpy())
        fresh_idx = np.flatnonzero(g.vision.to_numpy())
        np_idx = np.flatnonzero((g.noprog >= 2).to_numpy())
        ep = dict(model=model, arm=arm, task=int(task), init=int(init), success=bool(g.success.iloc[0]),
            decisions=len(g), looks=int(g.vision.sum()), calls=len(call_idx),
            guard_calls=int((g.call & (g.reason == 4)).sum()), escalation_calls=int((g.call & (g.reason == 91)).sum()),
            first_call=int(call_idx[0]) if len(call_idx) else None,
            first_np=int(np_idx[0]) if len(np_idx) else None,
            first_close=int(first_close[0]) if len(first_close) else None,
            first_release=int(release[0]) if len(release) else None,
            events=int(event.sum()), close_events=int(np.sum(np.diff(g.closed.astype(int)) == 1)),
            release_events=len(release), final_phase=str(phase[-1]),
            dnn_mean=float(g.dnn.mean()), dnn_max=float(g.dnn.max()),
            correction_mean=float(g.correction.mean()), correction_max=float(g.correction.max()),
            final_progress=float(g.progress.dropna().iloc[-1]) if g.progress.notna().any() else np.nan,
            first_call_phase=str(phase[call_idx[0]]) if len(call_idx) else None)
        eps.append(ep)
        if arm != 'recipe': continue
        for nth, j in enumerate(call_idx):
            r = g.iloc[j]
            nxt = fresh_idx[fresh_idx > j]
            k = int(nxt[0]) if len(nxt) else None
            rec = dict(model=model, task=int(task), init=int(init), seq=int(j), nth=nth,
                success=bool(r.success), phase=str(phase[j]), event=bool(event[j]),
                reason=int(r.reason), dnn=float(r.dnn), lag=float(r.lag), progress=float(r.progress),
                remaining=len(g)-1-j, next_seq=k, next_call=bool(g.call.iloc[k]) if k is not None else None,
                next_np_clear=bool(g.noprog.iloc[k] < 2) if k is not None else None,
                next_progress_delta=float((g.progress.iloc[k]-r.progress)*max(g.liblen.iloc[k]-1, 1)) if k is not None else np.nan,
                repeat_within4=bool(g.call.iloc[j+1:j+5].any()) if len(g) > j+4 else None)
            calls.append(rec)
    return df, pd.DataFrame(eps), pd.DataFrame(calls)


def pair_episodes(eps):
    a = eps[eps.arm == 'recipe'].drop(columns='arm')
    b = eps[eps.arm == 'P10'].drop(columns='arm')
    p = a.merge(b, on=['model', 'task', 'init'], suffixes=('_recipe', '_P10'), validate='one_to_one')
    p['group'] = np.select([p.success_recipe & p.success_P10, ~p.success_recipe & p.success_P10,
        p.success_recipe & ~p.success_P10], ['both_win', 'lost', 'gained'], default='both_fail')
    return p


def clean_json(obj):
    if isinstance(obj, dict): return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [clean_json(v) for v in obj]
    if isinstance(obj, np.ndarray): return clean_json(obj.tolist())
    if isinstance(obj, np.generic): return clean_json(obj.item())
    if isinstance(obj, float) and not np.isfinite(obj): return None
    return obj


def summary(df, ep, calls, pairs):
    stats = {}
    for model in ROOTS:
        tr = df[(df.model == model) & (df.arm == 'recipe') & (df.init < 20)]
        edges = np.quantile(tr.loc[tr.vision, 'dnn'], [.5, .9])
        stats[model] = {'fit_distance_edges': edges, 'splits': {}}
        for split, inits in [('fit', range(20)), ('eval', range(20, 30)), ('all', range(30))]:
            pp = pairs[(pairs.model == model) & pairs.init.isin(inits)]
            ee = ep[(ep.model == model) & ep.init.isin(inits)]
            d = df[(df.model == model) & df.init.isin(inits)]
            cc = calls[(calls.model == model) & calls.init.isin(inits)]
            rr = d[d.arm == 'recipe']
            s = dict(n=len(pp), recipe_success=int(pp.success_recipe.sum()), pure_success=int(pp.success_P10.sum()),
                paired=pp.group.value_counts().to_dict(), by_task=[], arms={}, call_stats={})
            delta = pp.assign(delta=pp.success_recipe.astype(int)-pp.success_P10.astype(int)).groupby('init').delta.mean().to_numpy()
            rng = np.random.default_rng(260802)
            s['delta_ci95'] = np.quantile(delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(1), [.025, .975])
            for arm, gg in ee.groupby('arm'):
                s['arms'][arm] = dict(decisions=int(gg.decisions.sum()), calls=int(gg.calls.sum()), looks=int(gg.looks.sum()),
                    first_call_median=float(gg.first_call.median()), no_call=int(gg.first_call.isna().sum()),
                    guard_calls=int(gg.guard_calls.sum()), escalation_calls=int(gg.escalation_calls.sum()))
            for t, g in pp.groupby('task'):
                s['by_task'].append(dict(task=int(t), recipe=int(g.success_recipe.sum()), pure=int(g.success_P10.sum()),
                    lost=g[g.group == 'lost'].init.tolist(), gained=g[g.group == 'gained'].init.tolist()))
            for label, g in pp.groupby('group'):
                s[label] = dict(n=len(g), calls_mean=float(g.calls_recipe.mean()),
                    first_call_median=float(g.first_call_recipe.median()), no_call=int(g.first_call_recipe.isna().sum()),
                    first_call_after_P10_done=int((g.first_call_recipe >= g.decisions_P10).sum()),
                    close_median=float(g.first_close_recipe.median()), release_median=float(g.first_release_recipe.median()),
                    dnn_mean=float(g.dnn_mean_recipe.mean()), correction_mean=float(g.correction_mean_recipe.mean()))
            for label, g in [('all', cc), ('guard', cc[cc.reason == 4]), ('first_guard', cc[(cc.reason == 4) & (cc.nth == 0)]),
                             ('escalation', cc[cc.reason == 91])]:
                s['call_stats'][label] = call_stat(g)
            s['call_by_phase'] = {str(k): call_stat(g) for k, g in cc[cc.reason == 4].groupby('phase')}
            guard = cc[cc.reason == 4].copy()
            guard['distance_bin'] = np.searchsorted(edges, guard.dnn.to_numpy())
            s['call_by_distance'] = {str(k): call_stat(g) for k, g in guard.groupby('distance_bin')}
            s['fresh_by_phase'] = {str(k): dict(n=len(g), calls=int(g.call.sum()), dnn=float(g.dnn.mean()),
                correction=float(g.correction.mean())) for k, g in rr[rr.vision].groupby('phase')}
            s['guard_admission'] = dict(eligible=int((rr.vision & (rr.noprog >= 2)).sum()),
                missed=int((rr.vision & (rr.noprog >= 2) & ~rr.call).sum()),
                off_condition=int((rr.vision & rr.call & (rr.reason == 4) & (rr.noprog < 2)).sum()),
                gripper_reconstruction_max=float(rr.grip_reconstruction.max()))
            stats[model]['splits'][split] = s
    return clean_json(stats)


def call_stat(g):
    return dict(n=len(g), episodes=len(g[['task','init']].drop_duplicates()),
        observed_next=int(g.next_seq.notna().sum()), next_np_clear=float(g.next_np_clear.dropna().astype(float).mean()),
        next_call=float(g.next_call.dropna().astype(float).mean()),
        next_progress_delta_median=float(g.next_progress_delta.median()),
        progress_advance=float((g.loc[g.next_seq.notna(), 'next_progress_delta'] > .5).mean()),
        repeat4_n=int(g.repeat_within4.notna().sum()), repeat4=float(g.repeat_within4.dropna().astype(float).mean()),
        step_median=float(g.seq.median()), remaining_median=float(g.remaining.median()))


def main():
    pieces=[]
    for model in ROOTS:
        m=recipe(model)
        pieces.extend([standard(model,m), pure(model)])
        print('admitted',model,flush=True)
    d,e,c=annotate(pd.concat(pieces,ignore_index=True)); p=pair_episodes(e)
    group=p[['model','task','init','group']]
    d=d.merge(group,on=['model','task','init'],validate='many_to_one')
    c=c.merge(group,on=['model','task','init'],validate='many_to_one')
    for name,data in [('decisions',d),('episodes',e),('calls',c),('pairs',p)]:
        data.to_parquet(owned(HERE/'results'/f'{name}.parquet'),index=False)
    p[p.group=='lost'].to_csv(owned(HERE/'results/lost_episodes.csv'),index=False)
    stats=summary(d,e,c,p); dump(HERE/'results/summary.json',stats)
    for model,v in stats.items():
        for split,s in v['splits'].items():
            print(model,split,s['recipe_success'],s['pure_success'],s['paired'],s['call_stats']['guard'],flush=True)


if __name__=='__main__': main()
