"""R7 A2 stage-level mechanism tables from server decision logs (accepted episodes only).

One record per anchor cycle (a vision decision plus the blind / policy-tail decisions it commits, up to the next
LOOK or the episode end), labelled with C1's frozen library stage (a2_common.stage_of on the anchor's retrieval
kernel), and one record per episode with lever counts and the first lever event (the first decision at which the arm
departs from A's cadence/content: SF/UF served extension block or early valve LOOK; SW first wrist-only LOOK; CU/CT
first call or forced ambiguous LOOK).

Family fields
  follow (SF/UF): granted/structural/stage_ok at the anchor (os_sf_*); extension blocks served
      (blind os_sf_extension==1); valve checks = passing blind checks + the firing check recorded on the next
      LOOK's blind_extras; early abort = valve fire at age 1 (A would have served blind: an extra look);
      extension fire = fire at age >= 2 (A would have looked anyway). UF logs stage_ok with the gate off, so
      'sf_refused' marks UF grants that SF's stage gate would have refused (the valve is not evaluated in UF).
  wrist (SW): actual camera of each LOOK, the plan (os_sw_next_camera / os_sw_reason) in force from the preceding
      decision, the planning anchor's stage and the LOOK's own stage.
  calls (CU/CT): per fresh anchor call (os_c_call), os_reason 62 lottery / 63 stall, p, CT weight / deviation
      entry / logged stage masses, extra LOOK scheduling.
Cross-checks: logged os_sf_* / os_c3_* stage masses versus the recomputed C1 table (agreement counts).

    <A2 prefix> -m exp.offline_search.rounds.r07.analysis_scripts.a2_stage --run eval [--arms ...] [--jobs 4]
Writes <out>/a2_stage_<run>.json (aggregates by arm, pooled by variant) and <detail>/a2_cycles_<run>.csv.gz,
<detail>/a2_episodes_<run>.csv.gz.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from exp.offline_search.rounds.r07.analysis_scripts import a2_common as C

STAGE_KEYS = ('cls', 'macro', 'label')


def _f(x):
    return None if x is None else float(x)


def analyze_arm(run_root, arm):
    A = C.load_arm(run_root, arm)
    info = A['info']
    fam = info['family']
    table = C.stage_table(info['cell'])
    timing, _, _ = C.client_timing(run_root, arm)
    cycles, episodes, xcheck = [], [], Counter()
    for e in A['episodes']:
        ds = e['decisions']
        anchors = [i for i, d in enumerate(ds) if d.get('vision')]
        ep = dict(arm=arm, cell=info['cell'], variant=info['variant'], uid=e['uid'], task=e['task'], init=e['init'],
                  Y=e['Y'], N=len(ds), V=len(anchors), M=sum(d.get('hit') is False for d in ds),
                  granted=0, ext_blocks=0, sf_refused_ext=0, early_abort=0, ext_fire=0, valve_checks=0,
                  wrist_looks=0, calls=0, lottery_calls=0, stall_calls=0, extra_looks=0, first_lever=None,
                  first_lever_step=None, first_lever_label=None, first_lever_macro=None, first_lever_cls=None,
                  first_lever_own_label=None, lever_labels=Counter(), refused_labels=Counter())
        tr = timing.get((e['uid'], e['run_id'], e['attempt']), [])
        ep['termination'] = tr[-1].get('termination_reason') if tr else None
        prev_st = None
        prev_dec = None
        first = None
        ep_cycles = []
        for ai, i in enumerate(anchors):
            d = ds[i]
            j = anchors[ai + 1] if ai + 1 < len(anchors) else len(ds)
            body, end = ds[i + 1:j], (ds[j] if j < len(ds) else None)
            st = C.stage_of(table, d['rows'], d['weights']) if d.get('rows') else dict(cls='none', macro='S?', label='S?.none')
            rec = dict(arm=arm, cell=info['cell'], variant=info['variant'], uid=e['uid'], task=e['task'], init=e['init'],
                       Y=e['Y'], a_idx=ai, step=int(d['step']), src=d.get('src'), hit=d.get('hit'),
                       look_reason=d.get('look_reason'), blind_since_prev=d.get('blind_age'), n_body=len(body),
                       n_cache_blind=sum(b.get('src') == 'cache_blind' for b in body),
                       n_policy_tail=sum(b.get('src') == 'policy_tail' for b in body),
                       end='look' if end is not None else 'episode_end',
                       end_reason=end.get('look_reason') if end is not None else None,
                       camera=d.get('camera_mode') or 'full', owner_cost=d.get('owner_cost'),
                       **{f'st_{k}': v for k, v in st.items() if k in ('cls', 'macro', 'label', 'event_mass',
                                                                     'unknown_mass', 'unanimous', 'min_rows_to_event')})
            lever_here = []
            if fam == 'follow':
                g, s_ok, struct = C.ex(d, 'os_sf_granted'), C.ex(d, 'os_sf_stage_ok'), C.ex(d, 'os_sf_structural')
                if C.ex(d, 'os_sf_unanimous') is not None and st['cls'] not in ('invalid', 'none'):
                    xcheck['sf_unanimous_agree'] += bool(C.ex(d, 'os_sf_unanimous')) == st['unanimous']
                    xcheck['sf_event_mass_agree'] += abs(C.ex(d, 'os_sf_event_mass') - st['event_mass']) < 1e-9
                    xcheck['sf_unknown_agree'] += abs(C.ex(d, 'os_sf_unknown') - st['unknown_mass']) < 1e-9
                    xcheck['sf_checked'] += 1
                ext = [b for b in body if C.bx(b, 'os_sf_extension') == 1]
                passes = sum(1 for b in body if C.bx(b, 'os_sf_valve_checked') == 1 and C.bx(b, 'os_sf_valve_fire') == 0)
                fire = end is not None and C.bx(end, 'os_sf_valve_fire') == 1
                end_age = _f(C.bx(end, 'os_sf_age')) if end is not None else None
                deltas = [C.bx(b, 'os_sf_delta') for b in body if C.bx(b, 'os_sf_valve_checked') == 1]
                if fire:
                    deltas.append(C.bx(end, 'os_sf_delta'))
                refused = bool(g) and s_ok == 0
                rec.update(granted=_f(g), structural=_f(struct), stage_ok=_f(s_ok), ext_blocks=len(ext),
                           valve_checks=passes + int(fire), valve_fire=int(fire),
                           early_abort=int(fire and end_age == 1), ext_fire=int(fire and (end_age or 0) >= 2),
                           end_sf_age=end_age, end_sf_look=_f(C.bx(end, 'os_sf_look')) if end is not None else None,
                           max_delta=max(deltas) if deltas else None, radius=_f(C.bx(end, 'os_sf_radius') if fire else
                                                                                 (C.bx(body[0], 'os_sf_radius') if body else None)),
                           sf_refused=int(refused), sf_refused_ext=len(ext) if refused else 0)
                ep['granted'] += int(bool(g))
                ep['ext_blocks'] += len(ext)
                ep['sf_refused_ext'] += rec['sf_refused_ext']
                if rec['sf_refused_ext']:
                    ep['refused_labels'][st['label']] += rec['sf_refused_ext']
                ep['early_abort'] += rec['early_abort']
                ep['ext_fire'] += rec['ext_fire']
                ep['valve_checks'] += rec['valve_checks']
                if ext:
                    lever_here.append(('extension', int(ext[0]['step'])))
                if rec['early_abort']:
                    lever_here.append(('early_valve_look', int(end['step'])))
            elif fam == 'wrist':
                # The plan in force is the LOOK's own blind_extras (re-planned at the LOOK with the current state),
                # else the last plan logged by the preceding decision.
                plan_next, plan_reason = C.bx(d, 'os_sw_next_camera'), C.bx(d, 'os_sw_reason')
                if plan_next is None and prev_dec is not None:
                    plan_next = C.ex(prev_dec, 'os_sw_next_camera')
                    plan_reason = C.ex(prev_dec, 'os_sw_reason')
                    if plan_next is None:
                        plan_next, plan_reason = C.bx(prev_dec, 'os_sw_next_camera'), C.bx(prev_dec, 'os_sw_reason')
                cam = d.get('camera_mode') or 'full'
                rec.update(plan_next=_f(plan_next), plan_reason=_f(plan_reason),
                           plan_label=prev_st['label'] if prev_st else 'start', plan_macro=prev_st['macro'] if prev_st else 'start',
                           plan_cls=prev_st['cls'] if prev_st else 'start',
                           plan_matches_camera=int((plan_next == 1) == (cam == 'wrist_only')) if plan_next is not None else None,
                           completion_calls=d.get('camera_completion_calls'), next_camera=_f(C.ex(d, 'os_sw_next_camera')),
                           next_reason=_f(C.ex(d, 'os_sw_reason')))
                if cam == 'wrist_only':
                    ep['wrist_looks'] += 1
                    lever_here.append(('wrist_look', int(d['step'])))
            elif fam == 'calls':
                call = C.ex(d, 'os_c_call')
                reason = C.ex(d, 'os_reason')
                xl = C.ex(d, 'os_c_extra_look')
                rec.update(fresh=_f(C.ex(d, 'os_c_fresh')), call=_f(call), call_reason=_f(reason), p=_f(C.ex(d, 'os_c_p')),
                           nominal_p=_f(C.ex(d, 'os_c_nominal_p')), stall_state=_f(C.ex(d, 'os_c_stall_state')),
                           cooldown=_f(C.ex(d, 'os_c_cooldown')), extra_look=_f(xl),
                           ct_weight=_f(C.ex(d, 'os_c3_weight')), ct_dev_entry=_f(C.ex(d, 'os_c3_dev_entry')),
                           ct_dev_latched=_f(C.ex(d, 'os_c3_dev_latched')), ct_event_mass=_f(C.ex(d, 'os_c3_event_mass')),
                           ct_deviation=_f(C.ex(d, 'os_c3_deviation')), ct_p75=_f(C.ex(d, 'os_c3_p75')))
                if C.ex(d, 'os_c3_event_mass') is not None and st['cls'] not in ('invalid', 'none'):
                    xcheck['ct_event_mass_agree'] += abs(C.ex(d, 'os_c3_event_mass') - st['event_mass']) < 1e-9
                    xcheck['ct_unanimous_agree'] += bool(C.ex(d, 'os_c3_unanimous')) == st['unanimous']
                    xcheck['ct_checked'] += 1
                if call:
                    ep['calls'] += 1
                    ep['lottery_calls'] += reason == 62
                    ep['stall_calls'] += reason == 63
                    lever_here.append(('call_stall' if reason == 63 else 'call_lottery', int(d['step'])))
                if xl:
                    ep['extra_looks'] += 1
                    lever_here.append(('extra_look', int(d['step']) + 1))
                if call != (d.get('hit') is False):
                    xcheck['call_flag_vs_hit_mismatch'] += 1
            # Cadence contract: A's commitment is the anchor plus one blind block; SF/UF add at most `granted`
            # blocks; a policy call is followed by at most one policy-tail block and never by cache blind blocks.
            xcheck['cycles'] += 1
            cap = 1 + (int(rec.get('granted') or 0) if fam == 'follow' else 0)
            if fam == 'calls' and d.get('hit') is False:
                if rec['n_policy_tail'] > 1 or rec['n_cache_blind'] > 0:
                    xcheck['violation_policy_tail'] += 1
            elif rec['n_body'] > cap:
                xcheck['violation_body_over_cap'] += 1
            if fam == 'follow' and not rec.get('granted') and rec['ext_blocks']:
                xcheck['violation_ext_without_grant'] += 1
            for kind, step in lever_here:
                ep['lever_labels'][st['label']] += 1
                if first is None or step < first[1]:
                    first = (kind, step, st, rec)
            cycles.append(rec)
            ep_cycles.append(rec)
            prev_st = st
            prev_dec = ds[j - 1] if j - 1 >= i else d
        if first is not None:
            kind, step, st, rec = first
            ep.update(first_lever=kind, first_lever_step=step, first_lever_frac=step / len(ds),
                      first_lever_label=rec.get('plan_label', st['label']) if kind == 'wrist_look' else st['label'],
                      first_lever_macro=rec.get('plan_macro', st['macro']) if kind == 'wrist_look' else st['macro'],
                      first_lever_cls=rec.get('plan_cls', st['cls']) if kind == 'wrist_look' else st['cls'],
                      first_lever_own_label=st['label'])
        last = ep_cycles
        ep['final_label'] = last[-1]['st_label'] if last else None
        ep['final_macro'] = last[-1]['st_macro'] if last else None
        runs = [int(c['st_macro'][1]) for c in last if c['st_macro'][1:2].isdigit()]
        ep['max_macro'] = C.macro_name(max(runs)) if runs else None
        ep['lever_labels'] = dict(ep['lever_labels'])
        ep['refused_labels'] = dict(ep['refused_labels'])
        episodes.append(ep)
    return dict(arm=arm, info=info, qc=A['qc'], cycles=cycles, episodes=episodes, xcheck=dict(xcheck))


def _add(acc, key, rec, fam):
    a = acc[key]
    a['anchors'] += 1
    a['decisions'] += 1 + rec['n_body']
    a['end_' + str(rec['end_reason'])] += 1
    if fam == 'follow':
        a['granted'] += int(bool(rec['granted']))
        a['structural_fail'] += int(rec['structural'] == 0)
        a['stage_fail'] += int(rec['structural'] == 1 and rec['stage_ok'] == 0)
        a['ext_blocks'] += rec['ext_blocks']
        a['valve_checks'] += rec['valve_checks']
        a['valve_fires'] += rec['valve_fire']
        a['early_aborts'] += rec['early_abort']
        a['ext_fires'] += rec['ext_fire']
        a['sf_refused_grants'] += rec['sf_refused']
        a['sf_refused_ext'] += rec['sf_refused_ext']
        a['cycles_failed_ep'] += int(rec['Y'] == 0)
    elif fam == 'wrist':
        a['wrist'] += int(rec['camera'] == 'wrist_only')
        a['completions'] += int(rec.get('completion_calls') or 0)
        a[f"plan_reason_{rec['plan_reason']}"] += 1
        a['plan_mismatch'] += int(rec['plan_matches_camera'] == 0)
    elif fam == 'calls':
        a['calls'] += int(bool(rec['call']))
        a['lottery'] += int(rec['call_reason'] == 62)
        a['stall'] += int(rec['call_reason'] == 63)
        a['extra_looks'] += int(bool(rec['extra_look']))
        a['p_sum'] += rec['p'] or 0.
        a['w_sum'] += rec['ct_weight'] or 0.
        a['dev_entries'] += int(bool(rec['ct_dev_entry']))


def derive(a, fam):
    a = dict(a)
    n = a.get('anchors', 0)
    a['dec_per_anchor'] = a['decisions'] / n if n else None
    if fam == 'follow':
        a['grant_share'] = a['granted'] / n if n else None
        a['ext_per_anchor'] = a['ext_blocks'] / n if n else None
        a['fire_rate'] = a['valve_fires'] / a['valve_checks'] if a.get('valve_checks') else None
    elif fam == 'wrist':
        a['wrist_share'] = a['wrist'] / n if n else None
    elif fam == 'calls':
        a['call_rate'] = a['calls'] / n if n else None
        a['mean_p'] = a['p_sum'] / n if n else None
        a['mean_ct_weight'] = a['w_sum'] / n if n else None
    return a


def aggregate(results):
    by_arm, pooled = {}, {}
    pool = defaultdict(lambda: {k: defaultdict(Counter) for k in STAGE_KEYS + ('all',)})
    for r in results:
        fam = r['info']['family']
        acc = {k: defaultdict(Counter) for k in STAGE_KEYS + ('all',)}
        for c in r['cycles']:
            for k in STAGE_KEYS:
                _add(acc[k], c['st_' + k], c, fam)
                _add(pool[(r['info']['variant'], r['info']['model'])][k], c['st_' + k], c, fam)
                _add(pool[(r['info']['variant'], 'all')][k], c['st_' + k], c, fam)
            _add(acc['all'], 'all', c, fam)
            _add(pool[(r['info']['variant'], r['info']['model'])]['all'], 'all', c, fam)
            _add(pool[(r['info']['variant'], 'all')]['all'], 'all', c, fam)
        if fam == 'wrist':  # also by planning stage (the stage the SW rule conditioned on)
            for k in ('label', 'macro', 'cls'):
                acc['plan_' + k] = defaultdict(Counter)
                for c in r['cycles']:
                    _add(acc['plan_' + k], c['plan_' + k], c, fam)
        eps = r['episodes']
        by_arm[r['arm']] = dict(info=r['info'], qc=r['qc'], xcheck=r['xcheck'],
                                episodes=len(eps), sr=float(np.mean([e['Y'] for e in eps])) if eps else None,
                                first_lever=dict(Counter(e['first_lever'] for e in eps)),
                                first_lever_macro=dict(Counter(e['first_lever_macro'] for e in eps if e['first_lever'])),
                                first_lever_cls=dict(Counter(e['first_lever_cls'] for e in eps if e['first_lever'])),
                                stages={k: {s: derive(v, fam) for s, v in sorted(acc[k].items())} for k in acc})
    for (variant, model), acc in pool.items():
        fam = C.parse_arm(f'r7_pi05_l10_50_{variant}' if variant not in ('SW', 'SFSW') else 'r7_sw_pi05_l10_50')['family']
        pooled[f'{variant}:{model}'] = {k: {s: derive(v, fam) for s, v in sorted(acc[k].items())} for k in acc}
    return by_arm, pooled


def uf_refused_summary(results):
    """UF episodes: extensions SF's stage gate would refuse, by stage, and their episodes' outcomes."""
    out = {}
    for r in results:
        if r['info']['variant'] != 'UF1':
            continue
        eps = r['episodes']
        with_ref = [e for e in eps if e['sf_refused_ext'] > 0]
        without = [e for e in eps if e['sf_refused_ext'] == 0]
        by_stage = Counter()
        for c in r['cycles']:
            if c['sf_refused_ext']:
                by_stage[c['st_label']] += c['sf_refused_ext']
        def per(sel, key):
            sel = [e for e in eps if e['Y'] == sel]
            return float(np.mean([e[key] / max(e['V'], 1) for e in sel])) if sel else None
        out[r['arm']] = dict(episodes=len(eps), eps_with_refused=len(with_ref),
                             refused_per_anchor_success=per(1, 'sf_refused_ext'), refused_per_anchor_fail=per(0, 'sf_refused_ext'),
                             ext_per_anchor_success=per(1, 'ext_blocks'), ext_per_anchor_fail=per(0, 'ext_blocks'),
                             sr_with_refused=float(np.mean([e['Y'] for e in with_ref])) if with_ref else None,
                             sr_without=float(np.mean([e['Y'] for e in without])) if without else None,
                             refused_ext_total=sum(e['sf_refused_ext'] for e in eps),
                             ext_total=sum(e['ext_blocks'] for e in eps),
                             refused_by_stage=dict(by_stage.most_common()))
    return out


def run_many(run_root, arms, jobs):
    if jobs > 1 and len(arms) > 1:
        with Pool(min(jobs, len(arms))) as p:
            return p.starmap(analyze_arm, [(run_root, a) for a in arms])
    return [analyze_arm(run_root, a) for a in arms]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', default='eval', help='eval | profile | <run root path>')
    p.add_argument('--arms', nargs='*')
    p.add_argument('--family', nargs='*', help='follow wrist calls A')
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--out', type=Path, default=C.OUT)
    p.add_argument('--detail', type=Path, default=C.DETAIL)
    a = p.parse_args()
    run_root, arms, skipped = C.cli_arms(a)
    results = run_many(run_root, arms, a.jobs)
    by_arm, pooled = aggregate(results)
    tag = run_root.name + ('_subset' if (a.arms or a.family) else '')
    C.write_json(a.out / f'a2_stage_{tag}.json', dict(schema='r7.a2.stage.v1', run=str(run_root), analysed=arms,
                                                      skipped=skipped, arms=by_arm, pooled=pooled,
                                                      uf_refused=uf_refused_summary(results)))
    C.write_csv(a.detail / f'a2_cycles_{tag}.csv.gz', [c for r in results for c in r['cycles']])
    C.write_csv(a.detail / f'a2_episodes_{tag}.csv.gz', [e for r in results for e in r['episodes']])
    for r in results:
        s = by_arm[r['arm']]['stages']['all'].get('all', {})
        print(r['arm'], 'eps', len(r['episodes']), 'anchors', s.get('anchors'), {k: round(v, 4) if isinstance(v, float) else v
              for k, v in s.items() if k in ('grant_share', 'ext_blocks', 'fire_rate', 'early_aborts', 'wrist_share',
                                             'call_rate', 'lottery', 'stall', 'sf_refused_ext')}, 'xcheck', r['xcheck'], flush=True)
    print('skipped (not complete):', skipped)


if __name__ == '__main__':
    main()
