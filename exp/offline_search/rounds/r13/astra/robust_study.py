"""Calibrate one bias against several allowed B behavior paths, then validate."""
from common import *
fence()
import numpy as np
from controller import fit_bias_minimax
from replay_study import DOSES, WIDTHS, trace_pack, library_pack, take, scale, metrics

def main():
    packs={}; metas=[read(p) for p in sorted((HERE/'data').glob('*.json'))]
    for meta in metas:
        if meta['cell'] not in CELLS: continue
        p=trace_pack(meta)
        if p is not None: packs[(meta['cell'],meta['root'],meta['method'])]=p
    rows=[]; fits=[]; sources={}; splitrows=[]
    for cell in CELLS:
        model,suite,sz=cell.split('_')
        sources[cell]=[key for key in packs if key[0]==cell]
        if not sources[cell]:
            key=(cell,'library','heldout'); packs[key]=library_pack(cell); sources[cell]=[key]
        ps=[packs[key] for key in sources[cell]]
        cals=[take(p,p['cal']) for p in ps]; vals=[take(p,~p['cal']) for p in ps]
        # Common scale, equal behavior-path mass via deterministic quantile grid.
        refs=[np.quantile(np.log1p(p['distance'][p['alive']&~p['guard']]),np.linspace(0,1,1001)) for p in cals]
        lo,hi=np.quantile(np.concatenate(refs),[.1,.9]); hi=max(hi,lo+.05)
        scales=dict(lo=float(lo),hi=float(hi))
        for key,p in zip(sources[cell],ps):
            for t,i,c in zip(p['task'],p['init'],p['cal']):
                splitrows.append(dict(cell=cell,source=':'.join(key[1:]),task=int(t),init_or_library_episode=int(i),
                                      split='calibration' if c else 'development_validation'))
        for target in DOSES:
            for width in WIDTHS:
                cfg=dict(model=model,target=target,width=width,**scales)
                fit=fit_bias_minimax(cals,**cfg)
                cfg['bias']=fit['bias']
                fits.append(dict(cell=cell,sources=sources[cell],**{**cfg,**fit}))
                for key,cal,val in zip(sources[cell],cals,vals):
                    for splitlabel,p in [('calibration',cal),('heldout',val)]:
                        rows.append(dict(cell=cell,width=width,target=target,evaluation=':'.join(key[1:])+':'+splitlabel,
                            synthetic=False,split=splitlabel,**metrics(p,cfg)))
                    for factor in (.5,.75,1.25,1.5):
                        stress={**val,'distance':val['distance']*factor}
                        rows.append(dict(cell=cell,width=width,target=target,evaluation=':'.join(key[1:])+f':x{factor}',
                            synthetic=True,split='heldout',**metrics(stress,cfg)))
        print(cell,'minimax_sources',len(ps),flush=True)
    dump(HERE/'robust_fits.json',fits);csvout(HERE/'robust_replay.csv',rows)
    csvout(HERE/'development_splits.csv',splitrows)
    summ=[]
    for w in WIDTHS:
        rr=[r for r in rows if r['width']==w and r['split']=='heldout' and not r['synthetic'] and
            not r['evaluation'].startswith('library')]
        ss=[r for r in rows if r['width']==w and r['synthetic']]
        summ.append(dict(width=w,n=len(rr),max_error=max(abs(r['error']) for r in rr),
            mae=float(np.mean([abs(r['error']) for r in rr])),pass02=sum(abs(r['error'])<=.02 for r in rr),
            min_interior=min(r['interior'] for r in rr),mean_interior=float(np.mean([r['interior'] for r in rr])),
            min_placement_change=min(r['placement_change'] for r in rr),
            mean_placement_change=float(np.mean([r['placement_change'] for r in rr])),
            stress_max=max(abs(r['error']) for r in ss),stress_pass02=sum(abs(r['error'])<=.02 for r in ss),stress_n=len(ss)))
    dump(HERE/'robust_summary.json',summ);provenance('robust');print(json.dumps(summ,indent=2))

if __name__=='__main__': main()
