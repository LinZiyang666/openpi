"""All inventoried B recording selections vs the four requested subset sizes."""
import csv
import json
from collections import defaultdict
from exp.offline_search.closed_loop.devset import subset_record,suite_short
from exp.offline_search.rounds.r15.qk.astra.prepare import HERE,REPO,dump

def main():
    inventory=json.loads((HERE/'inventory.json').read_text());cache={};rows=[]
    for root,manifests in inventory['recordings'].items():
        grouped={}
        for m in manifests:
            c=m['contract'];assert c['dev'] is True and c['init_pool']=='B'
            grouped.setdefault(c['arm'],{'contract':c,'options':m['options'],'looks':0})['looks']+=m['shards']
        for arm,v in grouped.items():
            c=v['contract'];model=c['model'];suite=suite_short(c['suite']);cell=f'{model}_{suite}';pairs=set(map(tuple,c['pairs']));r={'root':root,'arm':arm,'cell':cell,'selection_episodes':len(pairs),'indexed_looks':v['looks'],'image_size':v['options']['image_size'],'all_look_shadow':v['options']['shadow']}
            for size in (50,100,200,500):
                key=(cell,size)
                if key not in cache:
                    sub=json.loads((REPO/f'exp/offline_search/rounds/r10/subsets/{cell}_{size}.json').read_text())
                    cache[key]=subset_record(model,suite,'bpool_cs' if model=='pi05' else 'bpool_all',sub['episode_ids_by_task'])
                excluded=cache[key]['excluded_inits_by_task'];r[f'unseen_episodes_{size}']=sum(i not in excluded[str(t)] for t,i in pairs)
            rows.append(r)
    dump(HERE/'recording_exclusions.json',{'recordings':rows,'subset_identity_records':{f'{c}_{s}':v for (c,s),v in cache.items()},'counts':'Selected B task/init pairs, not accepted look counts; identity certificates validated by deployed devset.subset_record'})
    with (HERE/'recording_exclusions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for r in rows:print(r['root'],r['cell'],[r[f'unseen_episodes_{s}'] for s in (50,100,200,500)])

if __name__=='__main__':main()
