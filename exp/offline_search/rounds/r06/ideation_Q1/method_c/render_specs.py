"""Fill final artifact locations without writing to the referenced run/CAL roots."""
import argparse
import json
from pathlib import Path
from .common import HERE,CONTROLLER_VERSION,write_json,output_path

def render(*,cal_root,stall_root,recording_run_root,validation_run_root,smoke_run_root,suffix=''):
    if '/' in suffix or '\\' in suffix:raise ValueError('suffix must be a filename suffix')
    products={}
    for src,dst,run in [('emit_calibration_recordings.json','recording_prefit_specs',recording_run_root),
                         ('emit_arms_c32.json','validation_prefit_specs',validation_run_root),
                         ('emit_arms_smoke2.json','smoke_prefit_specs',smoke_run_root)]:
        value=(HERE/src).read_text()
        for key,path in [('<CAL>',cal_root),('<STALL>',stall_root),('<RUN>',run)]:
            if not Path(path).is_absolute():raise ValueError('final locations must be absolute')
            # JSON escaping applies to paths; no shell interpolation occurs.
            value=value.replace(key,json.dumps(str(path))[1:-1])
        rows=json.loads(value)
        path=HERE/(dst+suffix+'.json');write_json(path,rows);products[dst]=str(path)
    write_json(HERE/('rendered_locations'+suffix+'.json'),dict(controller_version=CONTROLLER_VERSION,
        cal_root=str(cal_root),stall_root=str(stall_root),recording_run_root=str(recording_run_root),
        validation_run_root=str(validation_run_root),smoke_run_root=str(smoke_run_root),products=products,
        referenced_roots_written=False))
    return products

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cal-root',required=True);ap.add_argument('--stall-root',required=True)
    ap.add_argument('--recording-run-root',required=True);ap.add_argument('--validation-run-root',required=True)
    ap.add_argument('--smoke-run-root',required=True);ap.add_argument('--suffix',default='')
    print(json.dumps(render(**vars(ap.parse_args()))))

if __name__=='__main__':main()
