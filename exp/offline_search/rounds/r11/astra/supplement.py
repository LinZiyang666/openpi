"""Robust risk proxies and within-episode placement checks, without new fitting."""
import numpy as np
from exp.offline_search.rounds.r11.astra.boundary import HERE, dump, install
from exp.offline_search.rounds.r11.astra.experiment import CELLS, episode_weights
from exp.offline_search.rounds.r11.astra.signals import probability
from exp.offline_search.rounds.r11.astra.analyze import grouped_mean


def main():
    install();out=[]
    for m,s,n in CELLS:
        tag=f'{m}_{s}_{n}'
        with np.load(HERE/'data'/tag/'signals.npz') as z:
            a={k:np.asarray(z[k]) for k in z.files}
        keep=a['step']%2==0;ep=a['ep'][keep];w=episode_weights(ep);risk=a['risk'][keep]
        for method in ('distance','predicted_error','disagreement','gripper_transition'):
            p=probability(a[method][keep],.4,weights=w)[0]
            risk_ep=grouped_mean(risk,ep);p_ep=grouped_mean(p,ep)
            captured_ep=grouped_mean(risk*p,ep)
            vals=dict(cell=tag,method=method,
                within_episode_lift=float(captured_ep.mean()/(risk_ep*p_ep).mean()))
            for name,proxy in (('mse',risk),('rms',np.sqrt(risk)),
                               ('winsor95',np.minimum(risk,np.quantile(risk,.95)))):
                vals[name+'_lift']=float(np.sum(w*proxy*p)/np.sum(w*proxy)/.4)
            out.append(vals)
        print(tag,' | '.join(f'{x["method"]}: RMS={x["rms_lift"]:.3f}, within={x["within_episode_lift"]:.3f}' for x in out[-4:]),flush=True)
    dump(HERE/'supplement.json',out)


if __name__=='__main__':
    main()
