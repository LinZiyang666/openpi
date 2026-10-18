"""Cached throughput plus small descriptor-only pure/profile inventory."""
from .audit import HERE,EVAL
from .io_budget import dump
from exp.offline_search.rounds.r17.filter_failed_knob.common import read,MT,REPO

def main():
    inventory=read(HERE/'INVENTORY.json');rows=[]
    for cell in inventory['references']:
        candidates=[r for r in cell['references'] if r['commit']==5]
        model=cell['model'];bench=cell['domain'];mode=cell['mode']
        source=(EVAL/MT[model][mode=='full'] if bench=='metaworld' else
                EVAL/('fa_dp_'+mode[0]+'_u_'+('o' if mode=='full' else 'l') if model=='dp' else
                      'fa_fw_'+mode[0]+'_u_'+('o' if mode=='full' else 'l')))
        spec=read(source/'assets/model_spec.json')
        refs=[]
        for r in candidates:
            refs.append(dict(run=r['run'],episodes=r['accepted'],replicas=r['replicas'],batch=r['batch'],
                rate_ep_min=r['episodes_per_minute'],descriptor_equal=read(EVAL/r['run']/'assets/model_spec.json')==spec))
        key=f'{model}:{"mt50" if bench=="metaworld" else "libero"}:{mode}:{cell["gpu"]}'
        rows.append(dict(cell,key=key,references=refs,missing=not refs,
            strict_source_identity_match=any(r['descriptor_equal'] for r in refs)))
    profiles=[]
    for directory in ('results','results_mt50','results_cached_text'):
        for p in (REPO/'exp/r17_integration/profile'/directory).glob('*.json'):
            d=read(p)
            if d.get('model','').split('_')[0] in ('dp','groot','fastwam'):
                profiles.append(dict(path=str(p),model=d.get('model'),suite=d.get('suite'),gpu=d.get('gpu_name'),
                    valid=d.get('valid'),certified=d.get('certified'),directory=directory))
    dump(HERE/'REFERENCES.json',dict(cells=rows,profiles=profiles,
        note='No journal/per_step reread. Counts/rates inherited from completed inventory; descriptors/profile GPU identities read from small metadata.'))
    print('reference cells',len(rows),'missing',sum(r['missing'] for r in rows),
          'without exact source descriptor match',sum(not r['strict_source_identity_match'] for r in rows))

if __name__=='__main__':main()
