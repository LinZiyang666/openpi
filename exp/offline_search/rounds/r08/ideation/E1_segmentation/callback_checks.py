"""Small discovery-only semantic and deferred-tool checks for E1."""
import argparse
import json
import numpy as np
import pandas as pd

from exp.offline_search.debug import reader
from exp.offline_search.debug.tools.physical import forensics, cards
from exp.offline_search.debug.tools.decision import divergence
from callback_analysis import ROOT,OUT,SPECS,load,write_json


def semantic():
    config=json.loads((OUT/'adapter_calibration.json').read_text())
    rows=[]
    card_rows=[]
    for spec in SPECS:
        if spec['r8']['variant'] not in ('A','P10'):continue
        eps,_,_=load(spec,normalized=False,init_limit=1)
        for ep in eps:
            summary,t=forensics.analyse(ep,config)
            preds=ep.controls['predicates']>.5;active=t['active']
            loses=(preds[:-1]&~preds[1:]) & active[1:,None] & active[:-1,None]
            losses5=0
            for i,j in np.argwhere(loses):
                if i+6<=ep.n and not preds[i+1:i+6,j].any():losses5+=1
            release=t['stage']=='release'
            rows.append(dict(arm=spec['arm'],task_id=ep.meta['task_id'],init=ep.meta['init'],
                success=ep.outcome['success'],onset=summary['onset_control'],label=summary['label'],
                predicate_losses=int(loses.sum()),predicate_losses_ge5=losses5,
                release_controls=int(release.sum()),release_while_closed=int((release&t['closed']).sum()),
                completed_closed_release=int((release&t['closed']&preds.all(axis=1)).sum()),
                active_controls=int(active.sum())))
            if spec['arm'] in ('r8_groot_l10_50_A','r8_groot_l10_P10') and ep.meta['task_id']==0:
                ep.reader_arm=reader.open_arm(ROOT,spec['arm'])
                ep.reader_arm.cache_enabled=False
                card_rows.append(cards.card(ep,OUT/'episode_cards',config))
        print('SEMANTIC',spec['arm'],len(eps),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'semantic_audit.csv',index=False)
    write_json(OUT/'episode_cards'/'coverage.json',card_rows)


class DiscoveryArm:
    def __init__(self,name):
        self.base=reader.open_arm(ROOT,name);self.base.cache_enabled=False
        e=self.base.episodes();self.ep=e[e.init.eq(0)].copy()
        assert len(self.ep)==10
        self.allowed=set(self.ep.episode_key)
        d=self.base.decisions();self.ds=d[d.episode_key.isin(self.allowed)].copy()
        assert self.ds.init.max()==0
    def __getattr__(self,key):return getattr(self.base,key)
    def episodes(self,*a,**kw):return self.ep.copy()
    def decisions(self,*a,**kw):return self.ds.copy()
    def controls(self,key,*a,**kw):
        assert key in self.allowed
        return self.base.controls(key,*a,**kw)


def augmentation():
    summary=[]
    for name in ['r8_groot_l10_50_A','r8_groot_l10_50_FL']:
        assert (ROOT/'state'/f'{name}.AUG_DONE').exists()
        arm=DiscoveryArm(name)
        result=divergence.analyze(arm)
        tables=result.pop('tables')
        dest=OUT/'divergence_checks'/name;dest.mkdir(parents=True,exist_ok=True)
        for table,rows in tables.items():
            pd.DataFrame(rows).to_parquet(dest/f'{table}.parquet',index=False)
        result['arm']=name
        result['stage_coverage']=getattr(arm,'profile_stage_coverage',{})
        summary.append(result)
        print('DIVERGENCE',name,result,flush=True)
    write_json(OUT/'divergence_checks'/'summary.json',summary)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['semantic','augmentation'])
    a=ap.parse_args()
    semantic() if a.mode=='semantic' else augmentation()
