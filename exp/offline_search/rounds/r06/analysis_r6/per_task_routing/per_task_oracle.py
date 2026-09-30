"""Per-task controller choice: lowest cost that reaches pure-inference (L10) SR, from existing R6 per-episode data.

For every cell, every eligible arm gives per-task successes (50 inits) and per-task owner IR (from server decision
logs when their totals match the arm's cost ledger; otherwise the arm-level IR is used for every task and flagged).
IR of a mix = mean of per-task IRs (tasks weighted equally; single arms are re-expressed the same way).
In-sample: DP over tasks minimizing mean IR subject to total successes >= pure L10 successes (optimistic: uses test
outcomes). Cross-validated: choose on inits of one half, evaluate on the other half, both directions.
"""
import csv, glob, json, os, sys
from collections import defaultdict
from multiprocessing import Pool
import numpy as np

R = '/home/weiland/trace_runs/os_closed_loop'
FF = '/home/weiland/projects/openpi/exp/offline_search/rounds/r06/frontier_final'
C1 = {'pi05': .152, 'groot': .148}
PURE_L10 = {('pi05', 'l10'): 'r04_cost/r4f_p_l10_inf_k10_L10', ('pi05', 'spatial'): 'r04_cost/r4f_p_sp_inf_k10_L10',
            ('groot', 'l10'): 'r05_q2/r5q2_g_l10_policy_L10', ('groot', 'spatial'): 'r05_q2/r5q2_g_spatial_policy_L10'}
OUT = sys.argv[1] if len(sys.argv) > 1 else None


def accepted_attempts(d):
    acc = {}
    for line in open(f'{d}/client/journal.jsonl'):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        uid = r.get('task_uid') or ''
        if ':eval:' in uid and r.get('status') in ('done', 'failed') and r.get('error') in (None, '') and r.get('accepted') is not False:
            acc[uid] = r.get('attempt')
    return acc


def per_episode_counts(args):
    rid, model = args
    run, arm = rid.split('/')
    d = f'{R}/{run}/runs/{arm}'
    try:
        led = json.load(open(f'{d}/summary.json'))['cost_ledger']
        N, V, M = int(led['decisions']), int(led['vision_decisions']), int(led['misses'])
    except Exception:
        return rid, None, None
    acc = accepted_attempts(d)
    cnt = defaultdict(lambda: [0, 0, 0])
    for f in glob.glob(f'{d}/server_*/decisions_*.jsonl'):
        for line in open(f):
            if '"ev": "dec"' not in line[:40]:
                continue
            r = json.loads(line)
            uid = r.get('uid')
            if uid not in acc or acc[uid] != r.get('attempt'):
                continue
            c = cnt[uid]
            c[0] += 1
            c[1] += bool(r.get('vision'))
            c[2] += r.get('src') == 'policy'
    tot = np.array([sum(v[i] for v in cnt.values()) for i in range(3)]) if cnt else np.zeros(3)
    ok = cnt and tuple(int(x) for x in tot) == (N, V, M)
    per_task = defaultdict(lambda: np.zeros(3))
    per_task_half = defaultdict(lambda: np.zeros(3))
    for uid, c in cnt.items():
        t, i = (int(x) for x in uid.split(':eval:')[1].split(':')[:2])
        per_task[t] += c
        per_task_half[(t, i >= 25)] += c
    ledger_ir = C1[model] * V / N + (1 - C1[model]) * M / N
    res = dict(ok=bool(ok), ledger_ir=ledger_ir,
               task={t: v.tolist() for t, v in per_task.items()},
               half={f'{t}:{int(h)}': v.tolist() for (t, h), v in per_task_half.items()})
    return rid, res, (N, V, M)


def main():
    rows = [r for r in csv.DictReader(open(f'{FF}/frontier_points.csv'))
            if r['eligible'] == 'True' and r['cost_valid'] == 'True']
    outcomes = json.load(open(f'{FF}/outcomes.json'))
    cells = sorted({r['cell'] for r in rows if not r['cell'].endswith('reference')})
    jobs = {(r['id'], r['model']) for r in rows if r['cell'] in cells}
    jobs |= {(v, k[0]) for k, v in PURE_L10.items()}
    with Pool(4) as p:
        counts = dict((rid, res) for rid, res, _ in p.map(per_episode_counts, sorted(jobs)))
    report = {}
    for cell in cells:
        model = 'pi05' if cell.startswith('pi05') else 'groot'
        suite = 'l10' if '_l10_' in cell else 'spatial'
        pure = PURE_L10[(model, suite)]
        arms = sorted({r['id'] for r in rows if r['cell'] == cell and r['pure'] != 'True'})
        c1 = C1[model]

        def task_table(rid, half=None):
            o = outcomes[rid]
            info = counts.get(rid)
            succ, ir = np.zeros(10), np.zeros(10)
            for t in range(10):
                inits = [i for i in range(50) if half is None or (i >= 25) == half]
                succ[t] = sum(o[f'{t}:{i}'] for i in inits)
                if rid == pure:
                    ir[t] = .5
                elif info and info['ok']:
                    v = np.array(info['task'][t] if half is None else info['half'][f'{t}:{int(half)}'])
                    ir[t] = c1 * v[1] / v[0] + (1 - c1) * v[2] / v[0]
                else:
                    ir[t] = info['ledger_ir'] if info else np.nan
            return succ, ir

        opts = [a for a in arms if a in outcomes and counts.get(a)] + [pure]
        flagged = [a for a in opts if a != pure and not counts[a]['ok']]

        def dp(tables, target):
            # min sum IR subject to total successes >= target; returns choice per task
            INF = 1e18
            maxs = int(sum(max(t[0][k] for t in tables) for k in range(10)))
            best = np.full(maxs + 1, INF); best[0] = 0; choice = []
            for k in range(10):
                nb = np.full(maxs + 1, INF); arg = np.full(maxs + 1, -1, int); prev = np.full(maxs + 1, -1, int)
                for j, (s, ir) in enumerate(tables):
                    sk, ck = int(s[k]), ir[k]
                    if np.isnan(ck):
                        continue
                    cand = np.full(maxs + 1, INF)
                    cand[sk:] = best[:maxs + 1 - sk] + ck
                    better = cand < nb
                    nb[better] = cand[better]; arg[better] = j; prev[better] = np.arange(maxs + 1)[better] - sk
                best = nb; choice.append((arg, prev))
            feas = [s for s in range(int(target), maxs + 1) if best[s] < INF]
            if not feas:
                return None
            s = min(feas, key=lambda x: (best[x], -x))
            plan = [0] * 10
            for k in range(9, -1, -1):
                arg, prev = choice[k]
                plan[k] = int(arg[s]); s = int(prev[s])
            return plan

        full = {a: task_table(a) for a in opts}
        target = full[pure][0].sum()
        tables = [full[a] for a in opts]
        plan = dp(tables, target)
        ins = None
        if plan:
            ins = dict(SR=float(sum(tables[j][0][k] for k, j in enumerate(plan)) / 500),
                       IR=float(np.mean([tables[j][1][k] for k, j in enumerate(plan)])),
                       plan=[opts[j] for j in plan])
        # best single arm (same task-equal IR convention) reaching target
        singles = [(float(np.mean(full[a][1])), float(full[a][0].sum() / 500), a) for a in opts if full[a][0].sum() >= target]
        single = min(singles) if singles else None
        # cross-validated: choose on one half, evaluate on the other
        cv = []
        for sel_half in (False, True):
            sel = [task_table(a, sel_half) for a in opts]
            ev = [task_table(a, not sel_half) for a in opts]
            p = dp(sel, sel[opts.index(pure)][0].sum())
            if p:
                cv.append((float(sum(ev[j][0][k] for k, j in enumerate(p)) / 250),
                           float(np.mean([ev[j][1][k] for k, j in enumerate(p)])),
                           float(ev[opts.index(pure)][0].sum() / 250)))
        # concentration of the cache-only gap: per-task successes of A (3-run mean if available) vs pure
        a_ids = [a for a in opts if full[a][1].mean() < .09]
        a_best = max(a_ids, key=lambda a: full[a][0].sum()) if a_ids else None
        gap = (full[pure][0] - full[a_best][0]).tolist() if a_best else None
        report[cell] = dict(options=len(opts), flagged_arm_level_ir=len(flagged), pure=pure, pure_SR=float(target / 500),
                            in_sample=ins, best_single=single,
                            cv=dict(SR=float(np.mean([c[0] for c in cv])) if cv else None,
                                    IR=float(np.mean([c[1] for c in cv])) if cv else None,
                                    pure_SR=float(np.mean([c[2] for c in cv])) if cv else None, n_halves=len(cv)),
                            cheapest_cache_arm=a_best, per_task_gap_pure_minus_cache=gap)
    json.dump(report, open(OUT, 'w'), indent=1)
    for cell, r in report.items():
        ins, s, cv = r['in_sample'], r['best_single'], r['cv']
        print(f"{cell:18s} pure {r['pure_SR']:.3f} | per-task in-sample "
              f"{'-' if not ins else f'{ins[chr(83)+chr(82)]:.3f} @ {ins[chr(73)+chr(82)]:.3f}'} | best single "
              f"{'-' if not s else f'{s[1]:.3f} @ {s[0]:.3f}'} | CV {'-' if cv['SR'] is None else f'{cv[chr(83)+chr(82)]:.3f} @ {cv[chr(73)+chr(82)]:.3f} (pure {cv[chr(112)+chr(117)+chr(114)+chr(101)+chr(95)+chr(83)+chr(82)]:.3f})'} "
              f"| gap by task {r['per_task_gap_pure_minus_cache']} | flagged {r['flagged_arm_level_ir']}/{r['options']}")


if __name__ == '__main__':
    main()
