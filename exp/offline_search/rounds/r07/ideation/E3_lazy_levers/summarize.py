"""Produce compact, auditable evidence tables and fixed-cohort diagnostics."""
import json
from pathlib import Path
import numpy as np
from analyze_lazy import OUT, SCRATCH, load_bank


def f(x):
    return f'{x:.3f}'


def main():
    lines = ['# Computed E3 evidence', '', 'All rates are descriptive, not causal blind-rollout rates.', '',
             '## State-tube replay', '',
             '| Cell | B-val/P3 anchors | q95 / ep-max95 | B-val support20 | B-val alerts10/20/30/40 | P3 alerts20/40 |',
             '|---|---:|---:|---:|---:|---:|']
    rows = []
    for model in ['pi05', 'groot']:
        for cell in ['l10_50', 'l10_500', 'sp_50', 'sp_500']:
            key = model+'_'+cell
            r = json.loads((OUT/f'evidence_{key}.json').read_text())
            b,p,c = r['bval'], r['p3_A_r0'], r['calibration']
            q = c['q_row95']
            lines.append(f'| {key} | {b["anchors"]}/{p["anchors"]} | {f(q)}/{f(c["q_episode95"])} | '
                         f'{f(b["by_age"]["20"]["supported_share"])} | '+
                         '/'.join(f(b['by_age'][str(h)]['row95_cumulative_alert']) for h in [10,20,30,40])+' | '+
                         '/'.join(f(p['by_age'][str(h)]['row95_cumulative_alert']) for h in [20,40])+' |')
            _,lib,rs,scale,succ,_,_ = load_bank(model,cell)
            extra = dict(cell=key)
            for stream in ['bval','p3_A_r0']:
                with np.load(SCRATCH/f'{key}_{stream}_residuals.npz') as z:
                    d, support, states, members = [z[k] for k in ['delta','support','states','members']]
                    valid40 = support[:,8]
                    extra[stream] = dict(fixed40_n=int(valid40.sum()),
                        fixed40_median=[float(np.median(d[valid40,h])) for h in [2,4,6,8]],
                        fixed40_p90=[float(np.quantile(d[valid40,h],.9)) for h in [2,4,6,8]],
                        age20_observed=int(np.isfinite(states[:,4]).all(axis=1).sum()),
                        age20_library_supported=int((succ[4,members]>=0).all(axis=1).sum()),
                        age20_joint_supported=int(support[:,4].sum()), anchors=len(d))
                    # Same common-40 cohort: do alerts distinguish eventual success?
                    for y in [0,1]:
                        mask=valid40 & (z['success']==y)
                        extra[stream][f'fixed40_alert_y{y}'] = float((np.nanmax(d[mask,1:9],axis=1)>q).mean()) if mask.any() else None
            rows.append(extra)
    lines += ['', '## Fixed supported-to-40 B-val cohort (avoids changing denominators)', '',
              '| Cell | n | Median delta 10/20/30/40 | p90 delta 10/20/30/40 |', '|---|---:|---:|---:|']
    for r in rows:
        b=r['bval']
        lines.append(f'| {r["cell"]} | {b["fixed40_n"]} | '+ '/'.join(map(f,b['fixed40_median']))+' | '+ '/'.join(map(f,b['fixed40_p90']))+' |')
    lines += ['', '## P3 dense telemetry audit', '',
              '| Cell | Active controls | n20 | Alerts every-control / every-five | Dense-only | Max normalization fit error |',
              '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        d=json.loads((OUT/f'controls_{r["cell"]}.json').read_text()); h=d['horizons']['20']
        lines.append(f'| {r["cell"]} | {d["control_rows"]} | {h["n"]} | {f(h["cumulative_any_control"])}/{f(h["cumulative_five_control"])} | {f(h["dense_only_share"])} | {max(d["affine_max_abs_error"]):.6f} |')
    lines += ['', '## Camera deletion, episode mean head sigma-RMS', '',
              '| Cell | Both / third / wrist | Gripper-change both / third / wrist | Low-motion both / third / wrist |',
              '|---|---:|---:|---:|']
    for r in rows:
        d=json.loads((OUT/f'cameras_{r["cell"]}.json').read_text())
        ss = ['/'.join(f(a[0]) for a in d['stages'][s]['episode_mean']) for s in ['all','gripper_change','low_motion']]
        lines.append('| '+r['cell']+' | '+' | '.join(ss)+' |')
    lines += ['', '## Predicted-tail / actual-successor-head mismatch (all library rows)', '',
              '| Cell | Offset5 median/p90 sigma-RMS | Offset5 gripper mismatch | Offset10 median/p90 (GR00T) |',
              '|---|---:|---:|---:|']
    for r in rows:
        d=json.loads((OUT/f'evidence_{r["cell"]}.json').read_text())['tail_successor_mismatch']
        lines.append('| '+r['cell']+' | '+'/'.join(map(f,d['5']['rms_q'][:2]))+' | '+f(d['5']['gripper_sign_mismatch'])+' | '+('/'.join(map(f,d['10']['rms_q'][:2])) if '10' in d else 'n/a')+' |')
    (OUT/'EVIDENCE_TABLES.md').write_text('\n'.join(lines)+'\n')
    (OUT/'additional_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
