"""Bootstrap B1 within suite; reuse pair weights across models/scales in each suite."""
import numpy as np
def refresh(pooled,arms,journals):
 for model in ['p','g','all']:
  p=pooled['B1_'+model];ls=p['armsA'];rng=np.random.default_rng(20260928);boots=np.zeros(10000);strata=[]
  for suite in ['l10','sp']:
   chosen=[l for l in ls if arms[l]['cell'].split('_',1)[1]==suite]
   if not chosen:continue
   keys=sorted(set.intersection(*(set(journals[l]) & set(journals[l.replace('_solve','_hand')]) for l in chosen)))
   d=np.array([np.mean([journals[l][k]-journals[l.replace('_solve','_hand')][k] for l in chosen]) for k in keys])
   u,c=np.unique(d,return_counts=True);weight=len(chosen)/len(ls)
   boots+=weight*(rng.multinomial(len(d),c/len(d),10000)@u/len(d))
   strata.append({'suite':suite,'clusters':len(keys),'weight':weight,'arms':chosen})
  p['ci']=np.quantile(boots,[.025,.975]).tolist();p['n']=sum(x['clusters'] for x in strata);p['bootstrap_strata']=strata
