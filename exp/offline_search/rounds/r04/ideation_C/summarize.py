"""Consolidate measured evidence and exact byte accounting for REPORT.md."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
import json
from pathlib import Path
import numpy as np
from exp.offline_search.rounds.r04.ideation_C.library_diagnostic import OUT

out={'cells':[]}
for m,l in [('pi05','p'),('groot','g')]:
    for s,short in [('spatial','sp'),('l10','l10')]:
        for scale in [50,500]:
            lib=json.loads((OUT/f'library_{m}_{s}_{scale}.json').read_text())
            seg=json.loads((OUT/f'segment_{m}_{s}_{scale}.json').read_text())
            logs=json.loads((OUT/f'logs_oscl{scale}_{l}_{short}_cl2.json').read_text())
            fit=Path(f'/home/weiland/trace_runs/os_closed_loop/r02_g{scale}/fits/oscl{scale}_{l}_{short}_cl2.pkl')
            dep={'pi05':{'spatial':431,'l10':1103},'groot':{'spatial':429,'l10':1068}}[m][s]
            cf=json.loads((OUT/f'crossfit_{m}_{s}_{scale}.json').read_text())
            c=dict(model=m,suite=s,scale=scale,L=lib['L'],episodes=lib['episodes'],
                   deployed_MB=dep,fit_bytes=fit.stat().st_size,
                   virtual_implicit_bytes=fit.stat().st_size+8*lib['L'],
                   virtual_explicit_count=seg['virtual_entries'],
                   failures=500-round(logs['sr']*500),successes=round(logs['sr']*500),
                   failed_with_repeated_events=seg['logs'][0]['episodes_with_repeated_events'],
                   successful_with_repeated_events=seg['logs'][1]['episodes_with_repeated_events'],
                   raw_visual_chord_ratio=seg['raw_visual_chord_ratio']['median'],
                   awm_chord_ratio=seg['geometry'][0]['ratio']['median'],
                   awm_event_chord_ratio=seg['geometry'][1]['ratio']['median'],
                   base_sr=logs['sr'])
            c['fit_deployed_fraction']=c['fit_bytes']/1e6/dep
            c['crossfit']={}
            for label in ['head','future']:
                rr=[r for r in cf['rows'] if r['label']==label]
                c['crossfit'][label]={k:sum(r[k]*r['n_future' if k=='next_rms' else 'n'] for r in rr)/sum(r['n_future' if k=='next_rms' else 'n'] for r in rr) for k in ['head_rms','next_rms','progress_far']}
            out['cells'].append(c)
pilot=json.loads((OUT/'pilot_backtest.json').read_text())
out['pilot']={}
for scale in [50,500]:
    rr=[r for r in pilot if r['scale']==scale]
    ratio=np.array([r['100']['variance_ratio'] for r in rr])
    out['pilot'][scale]=dict(comparisons=len(rr),median_variance_ratio=float(np.median(ratio)),
        min_variance_ratio=float(ratio.min()),max_variance_ratio=float(ratio.max()),
        median_equivalent_balanced_n=float(np.median(500/(1+4*ratio))),
        max_abs_simulation_bias=float(max(abs(r['100']['stratified_bias']) for r in rr)))
out['largest_output_bytes']=max(f.stat().st_size for f in OUT.iterdir() if f.is_file())
(OUT/'evidence_summary.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
