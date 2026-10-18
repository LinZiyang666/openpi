"""Static coordinator queue lists; never starts or edits a running chain."""
from pathlib import Path
from .audit import HERE,EVAL
from .io_budget import dump
from exp.offline_search.rounds.r17.filter_failed_knob.common import read

UNAFFECTED={
 'H100':['fa_co_f_f_oc1','fa_pi_f_u_o','fa_pi_f_f_o','fa_co_r_f_l','fa_co_f_u_l','fa_co_f_f_l','fa_pi_f_u_l','fa_pi_f_f_l'],
 'RTX 4090':['fa_pi_r_f_l','fa_gr_r_u_l','fa_gr_r_f_l','fa_gr_f_u_l','fa_gr_f_f_l']}

def main():
    jobs=read(HERE/'CAMPAIGNS.json')['jobs'];units=read(HERE/'FORMAL_QUEUE.json')['units']
    inv=read(HERE/'INVENTORY.json')['runs'];by_name={r['run']:r for r in inv}
    # Explicit measured proxies, with uncertainty retained in each queue row.
    proxies={('dp','mt50','RTX 4090'):'tdp7_0729',('groot','mt50','H100'):'grmk1_red',
       ('fastwam','mt50','H100'):'fwmk1_redx',('dp','libero','RTX 4090'):'fa_dp_r_u_l',
       ('fastwam','libero','RTX 4090'):'fa_fw_r_u_l',('fastwam','libero','H100'):'fa_fw_f_u_o'}
    result={g:[] for g in UNAFFECTED}
    profiles={g:[] for g in UNAFFECTED}
    for u in read(HERE/'PROFILES.json')['units']:
        # Historical wall times include model load, compile and both budgets.
        profile=next((r for r in read(HERE/'REFERENCES.json')['profiles'] if r['model']==u['model']+('_mw' if u['benchmark']=='mt50' else '_libero')),None)
        if profile:measured=read(profile['path']).get('elapsed_seconds',600)
        else:
            path=HERE.parents[4]/'exp/r17_integration/profile/results'/(u['model']+'_libero_'+u['suites'][0]+'.json')
            measured=read(path).get('elapsed_seconds',600) if path.exists() else 600
        multiplicity=len(u['suites'])*(2 if u['model']=='fastwam' and u['benchmark']=='libero' else 1)
        profiles[u['gpu']].append(dict(name=u['name'],episodes=0,hours=measured*multiplicity/3600,
            stage='latency_profile',estimate_source='historical profile elapsed_seconds; H100/prospective identity are proxies',
            prepare=str(HERE/'commands'/f'prepare_{u["name"]}.sh'),launch=str(HERE/'commands'/f'launch_{u["name"]}.sh')))
    deferred=[dict(u,episodes=u['maximum_episodes'],benchmark='libero') for u in units if u['stage']=='levels']
    # Reduced first; OFF precedes the levels whose prepare builds from it.
    priority={'pure':0,'long':1,'off':2,'descent':4,'levels':3,'cache_variants':5}
    def order(j):
        name=j['name']
        # Keep the owner-selected small R3 half before the large R2 half.
        split_order={'fa_ls_v02_s':'fa_ls_v02_0','fa_ls_v02_b':'fa_ls_v02_1'}
        return j.get('mode')!='reduced',priority[j['stage']],split_order.get(name,name)
    for j in sorted(jobs+deferred,key=order):
        gpu=j['gpu'];source=Path(j.get('original_source',j.get('source',''))).name
        rate=by_name.get(source,{}).get('episodes_per_minute');proxy=False
        if j['model']=='dp' and j['benchmark']=='mt50' and gpu=='RTX 4090':
            source='tdp7_0729';rate=by_name[source]['episodes_per_minute'];proxy=True
        if not rate:
            source=proxies.get((j['model'],j['benchmark'],gpu))
            if source is None:
                source=next(r['run'] for r in inv if r['model']==j['model'] and
                    r['domain']==('metaworld' if j['benchmark']=='mt50' else 'libero') and
                    r['gpu']==gpu and r.get('accepted',0)>=500 and r.get('episodes_per_minute'))
            rate=by_name[source]['episodes_per_minute'];proxy=True
        result[gpu].append(dict(name=j['name'],episodes=j['episodes'],hours=j['episodes']/rate/60,
            episodes_upper_bound=j['stage']=='levels',rate_ep_min=rate,rate_source=source,
            rate_is_proxy=proxy,stage=j['stage'],workers=j['workers'],replicas=j['replicas'],batch=j['batch'],
            model=j['model'],benchmark=j['benchmark'],mode=j.get('mode'),arm=j.get('arm','historical_original'),
            prepare=str(HERE/'commands'/f'prepare_{j["name"]}.sh'),launch=str(HERE/'commands'/f'launch_{j["name"]}.sh')))
    old_units=read(HERE.parent/'filter_failed_knob/formal_a_1016/QUEUE.json')['units']
    # Carry forward the unaffected queue exactly; do not rebuild its jobs or
    # touch its launch scripts. References have already been scheduled there.
    for gpu,names in UNAFFECTED.items():
        old=[]
        for n in names:
            u=next((u for u in old_units if u['name']==n),None)
            if u:
                count=u.get('episodes')
                if count is None:count=u.get('maximum_episodes',30000)
                meta=dict(model=u['model'],benchmark='libero',mode=u['mode'],arm=u['arm'])
                rate_source=HERE.parent/'filter_failed_knob/formal_a_1016/THROUGHPUT.json'
                # Estimates are explicitly source-test proxies, not a new read
                # of outcomes from active campaigns.
                if u['model']=='cosmos':
                    measured=next(r for r in read(rate_source)['rows'] if (r['model'],r['mode'])==('cosmos',u['mode']))
                    hours=count/measured['episodes_per_hour'];estimate_source=measured['source']
                else:
                    measured=read(HERE/'UNAFFECTED_REFERENCES.json')['rates'][u['model']+':'+u['mode']]
                    hours=count/(measured['rate_ep_min']*60);estimate_source=measured['sources']
            else:
                assert n=='fa_co_f_f_oc1'
                c=read(EVAL/n/'campaign.json')
                # Coordinator's final closed-parent count (18:45:05): 2,985
                # accepted of 6,000. Native prepare refreshes the live draft.
                count=c['cache_resume_selection']['expected_episodes']-2985
                meta=dict(model='cosmos',benchmark='libero',mode='full',arm='filtered')
                measured=next(r for r in read(HERE.parent/'filter_failed_knob/formal_a_1016/THROUGHPUT.json')['rows'] if (r['model'],r['mode'])==('cosmos','full'))
                hours=count/measured['episodes_per_hour'];estimate_source=measured['source']
            old.append(dict(name=n,episodes=count,hours=hours,rate_source=estimate_source,rate_is_proxy=True,unaffected=True,**meta,
                prepare=str(HERE.parent/'filter_failed_knob/commands'/f'prepare_{n}.sh'),
                launch=str(HERE.parent/'filter_failed_knob/commands'/f'launch_{n}.sh')))
        # Give unaffected reduced work an early turn, then new work, then the
        # unaffected full-step stages. Retain the coordinator's dependency order.
        if gpu=='RTX 4090':
            cut=3;result[gpu]=old[:cut]+result[gpu]+old[cut:]
        else:result[gpu]=old[:1]+result[gpu]+old[1:]
    for gpu,rows in result.items():
        assert len({r['name'] for r in rows})==len(rows)
        assert all(Path(r[v]).exists() for r in rows for v in ('prepare','launch'))
        filename='H100' if gpu=='H100' else '4090'
        (HERE/(filename+'_NAMES.txt')).write_text('\n'.join(r['name'] for r in rows)+'\n')
        lines=[]
        for r in rows:
            lines += ['# '+r['name'],'bash '+r['prepare'],'bash '+r['launch'],'']
        (HERE/(filename+'_COMMANDS.txt')).write_text('\n'.join(lines))
        (HERE/(filename+'_TIMING_NAMES.txt')).write_text('\n'.join(r['name'] for r in profiles[gpu])+'\n')
    dump(HERE/'QUEUES.json',dict(gpus=result,profiles_after_both_queues=profiles,active_preserved=['fa_co_r_u_l','ffk1015_pi05_mt50_reduced_descent'],
        planning='One lane per GPU. New DP MetaWorld work uses 4090; memory-constrained GR00T/Fast-WAM MetaWorld stay H100. Source R/B/MPS retained. Worker capacity is shared and measured at launch.',
        estimates='Episode/rate estimates exclude startup/cleanup and assume no infrastructure loss. Levels use worst-case all cells; new GPU/runtimes/clock cadence are proxies, not measured guarantees.'))
if __name__=='__main__':main()
