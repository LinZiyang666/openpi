from .common import *
install()
import numpy as np
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--guard',action='store_true');args=ap.parse_args()
outname='atlas_guard_assessment' if args.guard else 'atlas_assessment'

out=[]
for path in sorted((HERE/'atlas').glob('*/summary.json')):
    info=json.loads(path.read_text());cell,size=info['cell'],info['size'];C,lib=load(cell,size)
    with np.load(path.parent/'rows.npz') as z:a={k:np.array(z[k]) for k in z.files}
    cp=HERE/'correction_dynamic'/f'{cell}_{size}'
    def read(n):
        with np.load(cp/f'{n}.npz') as z:return {k:np.array(z[k]) for k in z.files}
    cm=read('meta');full=read('full');half=read('memory2');enhanced=read('full_delta')
    inv={int(r):i for i,r in enumerate(cm['row'])};order=np.array([inv[int(r)] for r in a['row']])
    assert np.array_equal(cm['ep'][order],a['ep'])
    # Freeze this modest middle rule; B-only screening, never an SR certificate.
    accept=(a['safe_mass_guard_0.15' if args.guard else 'safe_mass_0.15']>=.75)&(a['support_mass']>=.99)&(a['state_error']<=.2)
    selected=np.zeros(len(accept),bool);attempt=np.zeros(len(accept),bool)
    lastprog={} # trajectory replay here keeps baseline states fixed.
    for e in np.unique(a['ep']):
        rr=np.flatnonzero(a['ep']==e);rr=rr[np.argsort(a['step'][rr])];last_full=2
        for i in rr:
            attempt[i]=(a['step'][i]-last_full==2)
            if attempt[i] and accept[i]:selected[i]=True
            else:last_full=int(a['step'][i])
    baseline=full['motion_mse'][order];hm=half['motion_mse'][order];strong=enhanced['motion_mse'][order]
    mixed=np.where(selected,hm,baseline)
    lens=np.asarray(lib.ep_len)
    prev=C['progress'][a['previous_top']]
    advf=(C['progress'][a['full_top']]-prev)*np.maximum(lens[a['full_top']]-1,1)>.5
    advh=(C['progress'][a['half_top']]-prev)*np.maximum(lens[a['half_top']]-1,1)>.5
    ep=np.unique(a['ep']);em=np.array([np.mean((mixed-baseline)[a['ep']==e]) for e in ep])
    rng=np.random.default_rng(1203)
    ci=np.quantile(em[rng.integers(len(ep),size=(2000,len(ep)))].mean(1),[.025,.975])
    result=dict(cell=cell,size=size,rows=len(a['ep']),episodes=len(ep),
        gate_pass=mean(accept,a['ep']),half_share=mean(selected,a['ep']),
        corrected_full_mse=mean(baseline,a['ep']),corrected_half_mse=mean(hm,a['ep']),
        corrected_hybrid_mse=mean(mixed,a['ep']),corrected_delta=mean(mixed-baseline,a['ep']),
        corrected_relative_delta=mean(mixed-baseline,a['ep'])/mean(baseline,a['ep']),
        corrected_delta_ci=ci.tolist(),full_with_displacement_mse=mean(strong,a['ep']),
        hybrid_vs_full_with_displacement=mean(mixed-strong,a['ep']),
        missed_stall=mean(selected&~advf&advh,a['ep']),extra_stall=mean(selected&advf&~advh,a['ep']),
        gripper_disagreement=mean(np.where(selected,a['grip_flip'],0),a['ep']))
    out.append(result)
    print(cell,size,round(result['half_share'],3),round(100*result['corrected_relative_delta'],2),round(result['missed_stall'],4),round(result['extra_stall'],4))
dump(HERE/f'{outname}.json',dict(timestamp=now(),config=dict(label_rms=.15,required_mass=.75,state_valve=.2,support_mass=.99,progress_label=args.guard),cells=out,
    limitations='Raw-retrieval atlas labels; heldout corrected-action assessment. Guard diagnostics test a single progress comparison, not full memo dynamics. No SR or closed-loop claim.'))
dump(HERE/f'{outname}_reads.json',sorted(READS))
