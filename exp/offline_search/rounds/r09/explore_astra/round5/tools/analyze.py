"""Accepted-attempt, eval-only paired accounting and prefix-identifiable facts."""
import json
from datetime import datetime, timezone

import numpy as np
from scipy.stats import binomtest

from ...round2b.inference import LatchedGate, RecoveryGate
from .safe import HERE, STORE, EVAL, read_arm, safe_npz, dump


def summary(x):
    x = list(x)
    return dict(n=len(x), mean=float(np.mean(x)), median=float(np.median(x)),
                q25=float(np.quantile(x, .25)), q75=float(np.quantile(x, .75))) if x else dict(n=0)


def paired(base, arm):
    win = sorted(p for p in EVAL if arm[p]['success'] and not base[p]['success'])
    loss = sorted(p for p in EVAL if base[p]['success'] and not arm[p]['success'])
    delta = np.array([np.mean([int(arm[t,i]['success'])-int(base[t,i]['success'])
                              for t in range(10)]) for i in range(20,30)])
    boot = delta[np.random.default_rng(20261002).integers(0,10,(20000,10))].mean(1)
    return dict(wins=len(win), losses=len(loss), win_pairs=win, loss_pairs=loss,
                delta=(len(win)-len(loss))/100,
                mcnemar_p=float(binomtest(len(win),len(win)+len(loss)).pvalue) if win or loss else 1.,
                init_cluster_bootstrap95=np.quantile(boot,[.025,.975]).tolist())


def offline(model, mode):
    cell = model+'_l10_50'
    d = safe_npz(STORE/f'derived/r09_astra/round2b/{cell}_A_eval_scores.npz',
                 ['task','init','decision_id','score'])
    meta = json.loads((HERE.parent/f'round2b/artifacts/{cell}_monitor.json').read_text())
    c = safe_npz(STORE/f'derived/r09_astra/compact/r8_{cell}_A.npz',
                 ['task','init','decision_id','seq'])
    seq = dict(zip(c['decision_id'].astype(str),c['seq']))
    first, calls = {}, 0
    for p in sorted(EVAL):
        gate = LatchedGate(meta['threshold']) if mode == 'latch' else RecoveryGate(meta['threshold'],burst=int(mode[-1]))
        for j in np.flatnonzero((d['task']==p[0]) & (d['init']==p[1])):
            step = int(seq[str(d['decision_id'][j])])
            call, start = gate.observe(float(d['score'][j]),step)
            calls += call
            if start:
                first.setdefault(p,step)
    return first,calls


def main():
    out = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), allowed_inits=list(range(20,30)),
               fit='Frozen 2b detector/threshold; originally fitted only on 0–19; no refit', models={})
    for model,root in [('pi05','r09_astra_round2b'),('groot','r09_astra_round2b_g')]:
        prefix = f'r9r2b_astra_{model}_l10_50_'
        cache_spec, cache, cache_traces = read_arm(root,prefix+'cache')
        result = {}
        ledgers = {}
        for mode in ['cache','burst1','burst3','latch','random3','P10']:
            name = prefix+mode if mode!='P10' else f'r9r2b_astra_{model}_l10_P10'
            arm_spec, outcomes, traces = read_arm(root,name)
            ledgers[mode] = outcomes
            a = .152 if model=='pi05' else .148
            n = sum(map(len,traces.values()))
            looks = sum(r['vision'] for rows in traces.values() for r in rows)
            calls = sum(not r['hit'] for rows in traces.values() for r in rows)
            s = dict(sr=sum(bool(r['success']) for r in outcomes.values())/100, n=100,
                     decisions=n, looks=looks, calls=calls, ir=(a*looks+(1-a)*calls)/n,
                     paired_vs_same_run_cache=paired(cache,outcomes))
            first = {p: next((r['step'] for r in rows if r.get('vision') and
                             r.get('extras',{}).get('r9b_start',0)),None) for p,rows in traces.items()}
            first = {p:v for p,v in first.items() if v is not None}
            fired_bad = [p for p in first if not cache[p]['success']]
            fired_good = [p for p in first if cache[p]['success']]
            rescued = [p for p in fired_bad if outcomes[p]['success']]
            harm = [p for p in fired_good if not outcomes[p]['success']]
            wins = s['paired_vs_same_run_cache']['win_pairs']
            losses = s['paired_vs_same_run_cache']['loss_pairs']
            per_calls = {p:sum(not r['hit'] for r in rows) for p,rows in traces.items()}
            if mode not in ['cache','P10']:
                s['client_overrides_equal_cache'] = arm_spec['client_overrides']==cache_spec['client_overrides']
                s['step0_identical_state_pairs'] = sum(np.array_equal(cache_traces[p][0]['robot_state'],
                    traces[p][0]['robot_state']) for p in EVAL)
                s['step0_identical_served_head_pairs'] = sum(np.array_equal(cache_traces[p][0]['served_head'],
                    traces[p][0]['served_head']) for p in EVAL)
                s.update(alerted=len(first), first_alert_decision=summary(first.values()),
                         first_alert_controls=summary(5*x for x in first.values()),
                         remaining_decisions_after_alert=summary(len(traces[p])-step for p,step in first.items()),
                         calls_per_alerted=summary(per_calls[p] for p in first),
                         cap12_episodes=sum(per_calls[p]==12 for p in first),
                         alerted_cache_failures=len(fired_bad), rescued_after_alert=len(rescued),
                         measured_rescue_rate=len(rescued)/len(fired_bad) if fired_bad else None,
                         alerted_cache_successes=len(fired_good), harmed_after_alert=len(harm),
                         unalerted_wins=sum(tuple(p) not in first for p in wins),
                         unalerted_losses=sum(tuple(p) not in first for p in losses),
                         success_after_alert=sum(outcomes[p]['success'] for p in first),
                         alert_by_pair=[dict(task=p[0],init=p[1],step=v,calls=per_calls[p],
                                            success=bool(outcomes[p]['success']),cache_success=bool(cache[p]['success']))
                                        for p,v in sorted(first.items())])
                if mode!='random3':
                    off,ocalls=offline(model,mode)
                    both=set(first)&set(off)
                    s['offline_comparison']=dict(offline_alerts=len(off), offline_fixed_path_calls=ocalls,
                        offline_first_decision=summary(off.values()),both_alerted=len(both),
                        online_only=len(set(first)-set(off)),offline_only=len(set(off)-set(first)),
                        online_minus_offline_first_decision=summary(first[p]-off[p] for p in both),
                        caveat='Different cache rollouts: timing comparison is distribution shift, not same-path causal replay')
                # Replay scores on THIS logged path. Exact before first treatment;
                # subsequent score paths depend on the chosen response.
                if mode!='random3':
                    max_error=0
                    for p,rows in traces.items():
                        fresh=[r for r in rows if r['vision']]
                        threshold=fresh[0]['extras']['r9b_threshold']
                        gate=LatchedGate(threshold) if mode=='latch' else RecoveryGate(threshold,burst=int(mode[-1]))
                        for r in fresh:
                            call,start=gate.observe(r['extras']['r9b_score'],r['step'])
                            max_error=max(max_error,abs(int(call)-int(not r['hit'])),
                                          abs(int(start)-int(r['extras']['r9b_start'])))
                    s['logged_gate_replay_max_error']=max_error
            result[mode]=s
        result['latch_vs_burst3']=paired(ledgers['burst3'],ledgers['latch'])
        result['latch_vs_random3']=paired(ledgers['random3'],ledgers['latch'])
        out['models'][model]=result
    stacks={}
    for model,name in [('pi05','r9f3c_pi05_l10_50_np_corr05_esc'),('groot','r9f3c_groot_l10_50_np_corr05')]:
        _,outcomes,traces=read_arm('r09_fable_r3c',name)
        libstep=np.load(STORE/f'library/{model}_l10/current/step.npy',allow_pickle=False)
        first={}
        prefix={}
        for p,rows in traces.items():
            fresh=[r for r in rows if r['vision']]
            trigger=next((r['step'] for r in fresh if r['step']<=80 and
                          r['step']-int(libstep[r['top1']])>=12),None)
            if trigger is not None:
                first[p]=trigger
                later=[r for r in fresh if r['step']>=trigger]
                # A fixed 12-call takeover is identical to persistent escalation
                # until the 13th fresh decision (or longer if the guard fires).
                divergence=next((r['step'] for r in later[12:] if
                    not (int(r.get('extras',{}).get('os_flags',0)) & 8)),None) if model=='pi05' else next(
                        (r['step'] for r in later[:12] if r['hit']),None)
                if model=='pi05':
                    assert all(r['extras']['r9o_esc_step']==trigger for r in later)
                prefix[p]=dict(first=trigger, earliest_action_divergence=divergence,
                    logged_calls_since_trigger=sum(not r['hit'] for r in later),
                    success=bool(outcomes[p]['success']), final_step=len(rows)-1)
        a=.152 if model=='pi05' else .148
        n=sum(map(len,traces.values()));looks=sum(r['vision'] for rows in traces.values() for r in rows)
        calls=sum(not r['hit'] for rows in traces.values() for r in rows)
        divergent={p for p,v in prefix.items() if v['earliest_action_divergence'] is not None}
        unaffected_success=sum(outcomes[p]['success'] for p in EVAL-divergent)
        latch=out['models'][model]['latch']
        wins={tuple(p) for p in latch['paired_vs_same_run_cache']['win_pairs']}
        rescues={(r['task'],r['init']) for r in latch['alert_by_pair'] if r['success'] and not r['cache_success']}
        stacks[model]=dict(sr=sum(r['success'] for r in outcomes.values())/100,
            ir=(a*looks+(1-a)*calls)/n, decisions=n,looks=looks,calls=calls,
            pace_alerts=len(first),pace_first_decision=summary(first.values()),
            successes_with_pace_alert=sum(outcomes[p]['success'] for p in first),
            failures_with_pace_alert=sum(not outcomes[p]['success'] for p in first),
            logged_paths_reaching_pace_call13=sum(v['logged_calls_since_trigger']>12 for v in prefix.values()),
            pace12_first_action_divergence_episodes=len(divergent),
            pace12_divergent_control_successes=sum(outcomes[p]['success'] for p in divergent),
            pace12_prefix_identified_successes=unaffected_success,
            pace12_pathwise_sr_bounds=[unaffected_success/100,(unaffected_success+len(divergent))/100],
            pathwise_bound_caveat='Same exogenous randomness, first differing action ends identification; not a new fleet SR forecast',
            old_latch_wins_among_stack_failures=sum(not outcomes[p]['success'] for p in wins),
            old_latch_actual_rescues_among_stack_failures=sum(not outcomes[p]['success'] for p in rescues),
            overlap_caveat='Across different batches and policies; descriptive overlap, not stack rescue evidence',
            prefix_rows=[dict(task=p[0],init=p[1],**v) for p,v in sorted(prefix.items())],
            latch_score_reconstructible=False,
            missing='Fresh camera keys/PCA are not in standard logs. No exact latch replay on corrected stack.',
            inputs_logged=sorted(next(iter(traces.values()))[0]))
    out['stacks']=stacks
    dump(HERE/'results/analysis.json',out)
    brief={m:{k:{a:b for a,b in v.items() if a not in ('alert_by_pair',)} for k,v in x.items()}
           for m,x in out['models'].items()}
    print(json.dumps(dict(models=brief, stacks={m:{k:v for k,v in s.items() if k not in ['prefix_rows','inputs_logged']}
                                              for m,s in stacks.items()}),indent=2))


if __name__=='__main__':
    main()
