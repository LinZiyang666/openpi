"""Secondary timing/recovery diagnostics; only derived admitted rows are read."""
import numpy as np
import pandas as pd
from .safe import HERE, dump, owned
from .analyze import clean_json


def read(name):
    # These files are created by analyze from admitted 0..29 records only.
    p=HERE/'results'/f'{name}.parquet'
    identity=pd.read_parquet(p,columns=['task','init'])
    assert identity.task.between(0,9).all() and identity.init.between(0,29).all()
    return pd.read_parquet(p)


def rates(c):
    c=c[c.next_seq.notna()]
    if not len(c):return dict(n=0)
    ep=c.groupby(['task','init']).next_np_clear.mean().astype(float)
    means=ep.groupby('init').mean().to_numpy()
    boot=means[np.random.default_rng(260802).integers(0,len(means),(5000,len(means)))].mean(1)
    return dict(n=len(c),episodes=len(ep),call_weighted=float(c.next_np_clear.mean()),
        episode_weighted=float(ep.mean()),equal_init=float(means.mean()),ci95=np.quantile(boot,[.025,.975]))


def main():
    d=read('decisions');p=read('pairs');c=read('calls')
    out={};facts=[]
    for model in ('groot','pi05'):
        rr=d[(d.model==model)&(d.arm=='recipe')]
        fit=rr[(rr.init<20)&rr.vision]
        dist=np.quantile(fit.dnn,[.5,.9])
        for (task,init),g in rr.groupby(['task','init']):
            g=g.sort_values('seq');pair=p[(p.model==model)&(p.task==task)&(p.init==init)].iloc[0]
            calls=g[g.call];first=int(calls.seq.min()) if len(calls) else None
            far=g[g.vision&(g.dnn>dist[1])]
            first_far=int(far.seq.min()) if len(far) else None
            events=g[g.event].seq.tolist()
            prephase=[];ever=False
            for j in range(len(g)):
                if j==0:prephase.append('approach');continue
                prev=bool(g.closed.iloc[j-1]);ever |= prev
                prephase.append('carry' if prev else 'post' if ever else 'approach')
            # Call context uses only prior executed commands (not the policy result).
            ix=(c.model==model)&(c.task==task)&(c.init==init)
            c.loc[ix,'prephase']=[prephase[int(j)] for j in c.loc[ix,'seq']]
            call_values=[]
            for r in calls.itertuples():
                call_values.append(dict(seq=int(r.seq),phase=r.phase,prephase=prephase[r.seq],distance=float(r.dnn),
                    noprog=float(r.noprog),progress=float(r.progress),lag=float(r.lag)))
            facts.append(dict(model=model,task=int(task),init=int(init),group=pair.group,
                first_call=first,first_far=first_far,first_close=int(pair.first_close_recipe),
                first_release=pair.first_release_recipe,events=events,calls=call_values,
                calls_before_P10_done=int((calls.seq<pair.decisions_P10).sum()),
                first_call_before_close=first is not None and first<pair.first_close_recipe,
                first_call_before_release=first is not None and (np.isnan(pair.first_release_recipe) or first<pair.first_release_recipe),
                call_before_far=first_far is not None and first is not None and first<=first_far,
                final_phase=g.phase.iloc[-1],last_progress=pair.final_progress_recipe,
                near_terminal_failure=not bool(pair.success_recipe) and pair.final_progress_recipe>=.95,
                recipe_decisions=int(pair.decisions_recipe),pure_decisions=int(pair.decisions_P10),
                first_close_delta=float(pair.first_close_recipe-pair.first_close_P10),
                first_release_delta=float(pair.first_release_recipe-pair.first_release_P10),
                close_events_recipe=int(pair.close_events_recipe),close_events_P10=int(pair.close_events_P10),
                release_events_recipe=int(pair.release_events_recipe),release_events_P10=int(pair.release_events_P10)))
            facts[-1].update(correction_before_first_call=float(g.loc[g.seq<first,'correction'].mean()) if first is not None else np.nan,
                correction_after_first_call=float(g.loc[g.seq>first,'correction'].mean()) if first is not None else np.nan)
        cc=c[c.model==model].copy()
        cc['distance_bin']=np.searchsorted(dist,cc.dnn.to_numpy())
        cc['time_bin']=pd.cut(cc.seq,[-1,19,39,59,1000],labels=False)
        f=pd.DataFrame([x for x in facts if x['model']==model])
        out[model]={}
        for split,initset in [('fit',range(20)),('eval',range(20,30)),('all',range(30))]:
            cf=cc[cc.init.isin(initset)];ff=f[f.init.isin(initset)]
            gr=cf[cf.reason==4]
            out[model][split]=dict(timing={},recovery=dict(guard=rates(gr),first_guard=rates(gr[gr.nth==0]),
                by_group={k:rates(g) for k,g in gr.groupby('group')},
                by_prephase={k:rates(g) for k,g in gr.groupby('prephase')}))
            for group,gg in ff.groupby('group'):
                lost_calls=gr.merge(gg[['task','init']],on=['task','init'],validate='many_to_one')
                out[model][split]['timing'][group]=dict(n=len(gg),before_close=int(gg.first_call_before_close.sum()),
                    before_release=int(gg.first_call_before_release.sum()),has_far=int(gg.first_far.notna().sum()),
                    call_before_far=int(gg.call_before_far.sum()),calls_before_P10_done=float(gg.calls_before_P10_done.mean()),
                    calls_before_P10_done_min=int(gg.calls_before_P10_done.min()),
                    first_close_delta_median=float(gg.first_close_delta.median()),first_release_delta_median=float(gg.first_release_delta.median()),
                    first_far_median=float(gg.first_far.median()),
                    correction_before_first_call=float(gg.correction_before_first_call.mean()),
                    correction_after_first_call=float(gg.correction_after_first_call.mean()),
                    close_events_recipe=float(gg.close_events_recipe.mean()),close_events_P10=float(gg.close_events_P10.mean()),
                    close_events_per100_recipe=float(100*gg.close_events_recipe.sum()/gg.recipe_decisions.sum()),
                    close_events_per100_P10=float(100*gg.close_events_P10.sum()/gg.pure_decisions.sum()),
                    release_events_recipe=float(gg.release_events_recipe.mean()),release_events_P10=float(gg.release_events_P10.mean()),
                    near_terminal_failure=int(gg.near_terminal_failure.sum()),
                    call_phase=lost_calls.phase.value_counts().to_dict(),
                    first_call_phase={k:int(v) for k,v in lost_calls[lost_calls.nth==0].phase.value_counts().items()})
        c.loc[cc.index,'distance_bin']=cc.distance_bin
        c.loc[cc.index,'time_bin']=cc.time_bin
    # Descriptive shared-context standardization, never a causal cross-model estimate.
    out['shared_context']={}
    for split,initset in [('fit',range(20)),('eval',range(20,30)),('all',range(30))]:
        q=c[(c.reason==4)&c.init.isin(initset)&c.next_seq.notna()].copy()
        table=q.groupby(['prephase','distance_bin','time_bin','model']).next_np_clear.agg(['mean','size']).unstack('model')
        common=table.dropna();weights=common['size'].min(axis=1)
        out['shared_context'][split]=dict(strata=len(common),
            covered={m:int(common['size'][m].sum()) for m in ('groot','pi05')},
            total={m:int((q.model==m).sum()) for m in ('groot','pi05')},
            standardized={m:float(np.average(common['mean'][m],weights=weights)) for m in ('groot','pi05')})
    dump(HERE/'results/diagnostics.json',clean_json(out));dump(HERE/'results/episode_events.json',clean_json(facts))
    c.to_parquet(owned(HERE/'results/calls_annotated.parquet'),index=False)
    print('shared_context',out['shared_context'])
    for m in ('groot','pi05'):
        print(m,'eval',out[m]['eval'],'all_lost',out[m]['all']['timing'].get('lost'))


if __name__=='__main__':main()
