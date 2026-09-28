"""Aperture near its own-library lower endpoint is not a universal empty-grasp label."""
import json
import numpy as np
from diagnose_logs import HERE, STORE
from dynamics_probe import arr

def main():
    reports=[]
    for model in ['pi05','groot']:
      for suite in ['spatial','l10']:
       for size in [50,500]:
        p=STORE/'library'/f'{model}_{suite}'/('current' if size==50 else ('bpool_cs' if model=='pi05' else 'bpool_all'))
        s=arr(p,'rs');a=arr(p,'action');ep=arr(p,'episode')
        prm=np.load(HERE/f'dynamics_{model}_{suite}_{size}.npz')
        width=((s[:,6]-s[:,7])/2-prm['lo'])/(prm['hi']-prm['lo'])
        closed=(a[:,:5,6]*(1 if model=='pi05' else -1)>0).all(1)
        count=np.zeros(len(s),int)
        for j in range(1,len(s)):
            if ep[j]==ep[j-1] and closed[j-1]:count[j]=count[j-1]+1
        ok=count>=2
        reports.append(dict(model=model,suite=suite,size=size,stable_closed=int(ok.sum()),
            near_floor=int((ok&(width<.05)).sum()),near_floor_share=float((width[ok]<.05).mean())))
    (HERE/'aperture_floor_library.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps(reports,indent=2))

if __name__=='__main__':main()
