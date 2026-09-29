"""Code test only: pilot A, one test init/task, never a validation fit.

`--suffix v2b` (default) is the R6-C-v2 + SELECTION §7b run. The existing
`dryrun_v2` artifacts were produced by the pre-§7b code and are not rewritten.
"""
import argparse
import time
from pathlib import Path
from .common import OUT, sources, write_json, HERE
from .fit_calibration import fit

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--suffix',default='v2b')
    a=ap.parse_args()
    if not a.suffix.replace('_','').isalnum():raise ValueError('suffix must be a plain filename suffix')
    summary=[]
    for cell in sources():
        p3=cell.replace('spatial','sp');start=time.monotonic()
        cal=fit(Path('/home/weiland/trace_runs/os_closed_loop/r06_p3_pilot/tables')/p3,
            OUT/cell/'r_bank.json',OUT/cell/f'dryrun_{a.suffix}',.152 if cell.startswith('pi05') else .148,
            [.18,.30,.45],True,stall_model_path=Path('/tmp/q3_stall_fits')/p3)
        summary.append(dict(cell=cell,status=cal['status'],a=cal['intercept'],b=cal['slope'],anchors=cal['calibration_anchors'],
            ambiguous_rule=cal['ambiguous_rule'],cadence_nodes=cal['cadence_nodes'],fit_seconds=time.monotonic()-start,
            solutions=cal['solutions']))
    write_json(HERE/f'dryrun_summary_{a.suffix}.json',summary)

if __name__=='__main__':main()
