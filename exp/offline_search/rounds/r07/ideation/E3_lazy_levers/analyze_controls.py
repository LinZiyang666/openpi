"""Per-control P3 A_r0 position/finger tube audit, original observed path only.

Only proprioceptive observation fields are decoded. No object, contact or
simulator ground-truth field enters the statistic. Library has five-control
states; linear interpolation is an explicit diagnostic approximation.
"""
import argparse
import json
import time

import numpy as np

from analyze_lazy import (load_bank, loeo_neighbours, recordings, rows_for,
                          residuals, RUNS, OUT, quant)


def run(model, cell):
    start = time.time()
    awm, lib, rs, scale, succ, libpath, fitpath = load_bank(model, cell)
    dims = [0, 1, 2, 6, 7]
    starts, rows, w = loeo_neighbours(awm, lib)
    future = succ[:, starts].T
    states = rs[np.maximum(future, 0)].copy(); states[future < 0] = np.nan
    d, _, supported = residuals(states[:, :, dims], rows, w, rs[:, dims], scale[dims], succ)
    good = supported[:, 1:3].all(axis=1)
    q = float(np.quantile(np.max(d[good, 1:3], axis=1), .95, method='higher'))
    data = recordings(model, cell, True)
    # Reconstruct the model's affine state normalization solely for these five
    # non-angular coordinates. GR00T observation quantization remains a fit error.
    beta, fiterr = [], []
    for dim in dims:
        design = np.column_stack([data['raw'][:, dim], np.ones(len(data['raw']))])
        b = np.linalg.lstsq(design, data['states'][:, 0, dim], rcond=None)[0]
        beta.append(b)
        fiterr.append(float(np.max(np.abs(design @ b-data['states'][:, 0, dim]))))
    beta = np.asarray(beta)
    root = RUNS/'r06_p3_pilot'/'tables'/(model+'_'+cell)
    controls, begins = {}, {}
    nrows = 0
    for r in rows_for(root, 'controls', True):
        if r['wait_phase'] == 'True' or r['source'] == 'settle':
            continue
        assert r['source'] in ('cache', 'cache_blind')
        uid, step, control = r['uid'], int(r['decision_step']), int(r['control'])
        pos = json.loads(r['after.observation_numeric.robot0_eef_pos'])
        grip = json.loads(r['after.observation_numeric.robot0_gripper_qpos'])
        raw = np.asarray(pos+grip)
        norm = raw*beta[:, 0]+beta[:, 1]
        controls[uid, control+1] = norm
        begins.setdefault((uid, step), control)
        nrows += 1
    errors = np.full((len(data['uid']), 61), np.nan)
    for i, uid in enumerate(data['uid']):
        members, weights = data['members'][i], data['weights'][i]
        origin = begins[uid, int(data['step'][i])]
        mu0 = weights @ rs[members][:, dims]
        for t in range(1, 61):
            obs = controls.get((uid, origin+t))
            upper = (t+4)//5; lower = t//5
            if obs is None or (succ[upper, members] < 0).any():
                break
            frac = t/5-lower
            template = (1-frac)*rs[succ[lower, members]][:, dims] + frac*rs[succ[upper, members]][:, dims]
            diff = obs-data['states'][i, 0, dims] - (weights @ template-mu0)
            errors[i, t] = np.sqrt(np.mean((diff/scale[dims])**2))
    result = dict(cell=model+'_'+cell, source='P3 A_r0, original trajectory; position/fingers only',
                  control_rows=nrows, anchors=len(errors), q95=q, affine_max_abs_error=fiterr,
                  caveat='linear interpolation between 5-control library observations; normalization fitted from same recorded proprioception',
                  horizons={})
    for h in [5, 10, 15, 16, 20, 30, 32, 40, 48, 60]:
        valid = np.isfinite(errors[:, h])
        dense = np.any(errors[:, 1:h+1] > q, axis=1)
        check = np.any(errors[:, 5:h+1:5] > q, axis=1)
        out = dict(n=int(valid.sum()), endpoint_q=quant(errors[:, h]),
                   cumulative_any_control=float(dense[valid].mean()),
                   cumulative_five_control=float(check[valid].mean()),
                   dense_only_share=float((dense & ~check)[valid].mean()))
        result['horizons'][str(h)] = out
    result['elapsed_s'] = time.time()-start
    (OUT/f'controls_{model}_{cell}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(cell=result['cell'], controls=nrows, elapsed_s=result['elapsed_s'],
                         horizon20=result['horizons']['20'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--cells', nargs='+', default=['l10_50', 'l10_500', 'sp_50', 'sp_500'])
    a = p.parse_args()
    for cell in a.cells:
        run(a.model, cell)
