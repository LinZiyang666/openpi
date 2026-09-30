"""Stage/horizon support and displacement valves on fixed recorded streams."""
from __future__ import annotations

import time

import numpy as np

from . import common as C


def source_episodes(lib):
    entries = __import__('json').loads((lib.dir / 'episodes.json').read_text())
    keys = [(e.get('task_id'), e.get('file') or e.get('stem') or e.get('uid')) for e in entries]
    ids = {v: i for i, v in enumerate(dict.fromkeys(keys))}
    return np.asarray([ids[keys[int(e)]] for e in lib['episode']])


def tube(states, rows, weights, rs, scale, chain):
    """All-member support; missing observations are censored separately."""
    d, absolute, lost, support = [], [], [], []
    mean0 = weights @ rs[rows]
    for h, state in enumerate(states):
        future = chain[h, rows]
        good = future >= 0
        lost.append(float(weights[~good].sum()))
        support.append(bool(good.all()))
        if not good.all() or not np.isfinite(state).all():
            d.append(np.nan); absolute.append(np.nan)
        else:
            mu = weights @ rs[future]
            d.append(float(np.sqrt(np.mean(((state-states[0]-mu+mean0)/scale)**2))))
            absolute.append(float(np.sqrt(np.mean(((state-mu)/scale)**2))))
    return np.asarray(d), np.asarray(absolute), np.asarray(lost), np.asarray(support)


def eligibility(lib, table, chain, rows, cap, manifest):
    """Never clamp, drop, or reweight a member, including a zero-weight member."""
    from exp.offline_search.rounds.r07.c1_follow.methods import FollowExtension
    anchor = dict(rows=rows,weights=np.ones(len(rows))/len(rows))
    plan = FollowExtension(table,extend_blocks=cap).plan(anchor)
    return plan.structural, plan.structural and plan.stage_ok


def audit_cell(cell, campaigns, caps):
    t0 = time.monotonic()
    base, lib, manifest = C.bank(cell)
    table = C.stage_table(lib, manifest, base)
    dims = np.arange(int(manifest['rs_valid_dims']))
    rs = np.asarray(lib['rs'][:, dims], float)
    scale = np.asarray(table.state_scale, float)
    scale = scale[dims] if len(scale) > len(dims) else scale
    active = table.state_active
    dims, rs, scale = dims[active], rs[:, active], scale[active]
    chain = C.successors(lib, 2+max(caps))
    cal = dict(row95=table.valve_radius_row,episode95=table.valve_radius_episode,
               **table.calibration)
    rows_out, timeline = [], []
    for campaign in campaigns:
        data = C.stream(cell, campaign, dims, 2+max(caps))
        groups = {}
        for a in data:
            ar = a['anchor']; rows = a['rows']; w = a['weights']
            if len(rows) != base.k or len(w) != base.k or not np.isfinite(w).all() or np.any(w < 0) or not np.isclose(w.sum(), 1):
                raise ValueError('incomplete immutable kernel')
            d, absolute, lost, support = tube(a['states'], rows, w, rs, scale, chain)
            stage = C.stage_label(table, rows, w)
            for cap in caps:
                uf, sf_structure = eligibility(lib, table, chain, rows, cap, manifest)
                end = 2+cap
                observed = np.isfinite(d[1:end]).all()
                gate = bool(observed and np.all(d[1:end] <= cal['row95']))
                first_alert = next((h for h in range(1, end) if np.isfinite(d[h]) and d[h] > cal['row95']), None)
                event_age = next((h for h in range(1, end+1) if np.any(chain[h, rows] < 0) or np.any(table.mode[chain[h, rows]] != table.mode[rows[0]])), None)
                r = dict(cell=cell, campaign=campaign, uid=ar['uid'], step=int(ar['step']), stage=stage, cap=cap,
                         structural=uf, stage_interior=sf_structure, valve_observed=observed, SF=sf_structure and gate,
                         UF=uf, delta=d[end], absolute=absolute[end], lost_mass=lost[end],
                         supported=bool(support[end]), future_observed=bool(np.isfinite(a['states'][end]).all()),
                         row_alert=first_alert is not None, episode_alert=bool(np.any(d[1:end] > cal['episode95'])),
                         first_alert_age=first_alert, event_age=event_age,
                         modeled_SF_cycle=(first_alert or end) if sf_structure and observed else 2 if not sf_structure else None,
                         modeled_UF_cycle=end if uf else 2,
                         event_lead_controls=(event_age-first_alert)*manifest['exec_steps'] if first_alert is not None and event_age is not None else None)
                timeline.append(r)
                groups.setdefault((stage, cap), []).append(r)
        for (stage, cap), group in sorted(groups.items()):
            n = len(group); observed = [r for r in group if r['valve_observed']]
            uf = np.mean([r['UF'] for r in group]); sf = np.mean([r['SF'] for r in group])
            vision = .152 if cell.startswith('pi05_') else .148
            complete=[r for r in group if r['modeled_SF_cycle'] is not None]
            # Fixed-cohort cycle projection, not a changed-rollout renewal replay.
            rows_out.append(dict(cell=cell, campaign=campaign, stage=stage, cap=cap, anchors=n,
                episodes=len({r['uid'] for r in group}), UF_eligible=uf, SF_eligible=sf,
                stage_eligible=np.mean([r['stage_interior'] for r in group]), valve_observed=len(observed),
                valve_alert_rate=np.mean([r['row_alert'] for r in observed]) if observed else None,
                future_observed=sum(r['future_observed'] for r in group), supported=sum(r['supported'] for r in group),
                delta=C.quant([r['delta'] for r in group]), absolute=C.quant([r['absolute'] for r in group]),
                lost_mass=C.quant([r['lost_mass'] for r in group]), lead_controls=C.quant([r['event_lead_controls'] for r in group if r['event_lead_controls'] is not None]),
                modeled_A_IR=vision/2, modeled_UF_IR=vision/np.mean([r['modeled_UF_cycle'] for r in group]),
                modeled_SF_complete_anchors=len(complete),
                modeled_SF_IR=vision/np.mean([r['modeled_SF_cycle'] for r in complete]) if complete else None))
    return dict(cell=cell, calibration=cal, stage_fingerprint=getattr(table, 'fingerprint', None),
                elapsed_s=time.monotonic()-t0, rows=rows_out), timeline


def main():
    p = C.parser(__doc__)
    p.add_argument('--campaigns', nargs='+', choices=['bval', 'p3'], default=['bval', 'p3'])
    p.add_argument('--caps', nargs='+', type=int, default=[1, 2])
    a = p.parse_args()
    if not a.caps or any(c < 1 for c in a.caps):
        p.error('caps must be positive')
    reports, timeline = [], []
    for cell in C.cells(a.cells):
        report, ts = audit_cell(cell, a.campaigns, a.caps)
        reports.append(report); timeline.extend(ts)
        print(cell, 'anchors',len(ts)//len(a.caps), 'calibration',report['calibration'], flush=True)
    C.write(a.out/'follow_audit.json', dict(schema='r7.c4.follow.v1', reports=reports,
        caveat='Fixed observed streams: no causal SR or blind-rollout response. Null future observations are censored; all-anchor admission treats missing valve checks as unavailable. Modeled IR is a stationary cycle projection with zero calls, not realized cost.'))
    C.csv_write(a.out/'follow_timeline.csv', timeline)
    C.csv_write(a.out/'follow_summary.csv', [r for report in reports for r in report['rows']])


if __name__ == '__main__':
    main()
