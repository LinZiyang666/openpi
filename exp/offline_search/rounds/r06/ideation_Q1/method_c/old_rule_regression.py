"""Regression probe for the §7b cost model (code test only, DRYRUN_TEST_INITS).

Re-runs the dry-run fit with the pre-§7b rule patched into budget.py (ambiguous
p=0, LOOK regardless of the lottery) and requires the cadence-DAG model to
reproduce `dryrun_summary_v2.json` (the old fixed-cadence model) for every cell,
mode, placement and target. The probe calibrations carry a wrong rule stamp, so
they are deleted after comparison; only this script's log is kept.
"""
import argparse
import json
import time
from pathlib import Path
from . import budget
from .common import OUT, HERE, sources
from .fit_calibration import fit


def old_p(state, cooled, nominal):
    return 0. if cooled or state == 'slow_ambiguous' else 1. if state == 'slow_confirmed' else nominal


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cells', nargs='*', default=list(sources()))
    a = ap.parse_args()
    budget.call_probability = old_p
    budget.scheduled_look = lambda eligible, call: bool(eligible)
    ref = {r['cell']: r for r in json.loads((HERE/'dryrun_summary_v2.json').read_text())}
    worst, compared = 0., 0
    for cell in a.cells:
        p3 = cell.replace('spatial', 'sp'); t = time.monotonic(); out = OUT/'probe_7b'/f'old_rule_{cell}'
        cal = fit(Path('/home/weiland/trace_runs/os_closed_loop/r06_p3_pilot/tables')/p3, OUT/cell/'r_bank.json',
                  out, .152 if cell.startswith('pi05') else .148, [.18, .30, .45], True,
                  stall_model_path=Path('/tmp/q3_stall_fits')/p3)
        for name in ('calibration.json', 'cost_replay.json', 'calibration_anchor_labels.csv', 'r_bank.json', 'r_bank.npz'):
            (out/name).unlink()
        out.rmdir()
        for mode, sol in ref[cell]['solutions'].items():
            for pl, d in sol.items():
                for k, v in d.items():
                    new = cal['solutions'][mode][pl]
                    # The floor pseudo-target's key is its own 12-digit value.
                    w = new[k] if k in new else min(new.values(), key=lambda x: abs(x['rho']-v['rho']))
                    assert v['feasible'] == w['feasible'], (cell, mode, pl, k)
                    for f in ('floor', 'ceiling', 'parameter', 'predicted_IR'):
                        if v[f] is None:
                            assert w[f] is None, (cell, mode, pl, k, f); continue
                        worst = max(worst, abs(v[f]-w[f])); compared += 1
                    for f in ('calls', 'anchors', 'extra_LOOKs'):
                        if v.get('modeled'):
                            worst = max(worst, abs(v['modeled'][f]-w['modeled'][f])); compared += 1
        print(json.dumps(dict(cell=cell, seconds=round(time.monotonic()-t, 1), nodes_old_rule=cal['cadence_nodes'],
                              max_abs_diff_so_far=worst)), flush=True)
    assert worst < 1e-9, worst
    print(json.dumps(dict(status='PASS', test='DAG with pre-7b rule reproduces dryrun_summary_v2.json', cells=len(a.cells),
                          values_compared=compared, max_abs_diff=worst)))


if __name__ == '__main__':
    main()
