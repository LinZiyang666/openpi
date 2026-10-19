"""Node-local max-min sharing of reclaimable RAM and renderer VRAM.

Equal dominant-resource shares give each demanding campaign a 1/k entitlement;
targets below that share release capacity to borrowers. Integer workers round
down, then the remaining indivisible slots rotate. No process is signalled.
"""
import math


def allocate(capacity, demands, costs, *, epoch=0):
    """Progressive filling, capped by each lane's independent throughput target."""
    capacity={r:max(0,float(v)) for r,v in capacity.items()}
    names=sorted(n for n,v in demands.items() if v>0)
    count={n:0. for n in demands}
    remaining=dict(capacity)
    active=set(names)
    while active:
        rates={}
        for n in active:
            if any(v>0 and capacity[r]<=0 for r,v in costs[n].items()):
                continue
            dominant=max((v/capacity[r] for r,v in costs[n].items() if v>0),default=1.)
            rates[n]=1/max(dominant,1e-15)
        active=set(rates)
        if not active:break
        speed={r:sum(rates[n]*costs[n].get(r,0) for n in active) for r in capacity}
        delta=min([max(0,demands[n]-count[n])/rates[n] for n in active]+
                  [max(0,remaining[r])/s for r,s in speed.items() if s>0])
        for n in active:count[n]+=delta*rates[n]
        for r,s in speed.items():remaining[r]=max(0,remaining[r]-delta*s)
        capped={n for n in active if count[n]>=demands[n]-1e-8}
        exhausted={r for r,s in speed.items() if s>0 and remaining[r]<=max(1e-7,capacity[r]*1e-12)}
        frozen={n for n in active if any(costs[n].get(r,0)>0 for r in exhausted)}
        if not capped and not frozen:break
        active-=capped|frozen
    quotas={n:min(int(demands[n]),max(0,math.floor(v+1e-8))) for n,v in count.items()}
    remaining={r:max(0,b-sum(quotas[n]*costs[n].get(r,0) for n in quotas)) for r,b in capacity.items()}
    # A sub-worker entitlement cannot execute an episode. Rotate residual
    # slots once per minute; a borrower returns one at an episode boundary.
    order=names[epoch%len(names):]+names[:epoch%len(names)] if names else []
    while order:
        progress=False
        for n in order:
            if quotas[n]>=demands[n]:continue
            if all(remaining[r]+1e-8>=v for r,v in costs[n].items()):
                quotas[n]+=1
                for r,v in costs[n].items():remaining[r]=max(0,remaining[r]-v)
                progress=True
        if not progress:break
    return quotas


def plan(states, available, floor, *, now):
    """Use resident RSS to recover the total lendable pool, not just free RAM."""
    states=[s for s in states if s.get('schema',0)>=2]
    groups={}
    for s in states:
        key=s['fairness_campaign']
        row=groups.setdefault(key,dict(target=0,peak=1,vram=0,server_reserve=0,workers=[]))
        row['target']=max(row['target'],s['target'])
        row['peak']=max(row['peak'],s['projected_peak'])
        row['vram']=max(row['vram'],s.get('worker_vram_mib',0))
        row['server_reserve']=max(row['server_reserve'],s.get('server_vram_headroom_mib',0))
        row['workers'].extend(dict(w,fleet=s['fleet']) for w in s['workers'])
    for g in groups.values():
        # Charge each live outlier once, not as the assumed cost of every
        # worker in its lane. New-worker RAM admission still uses p90.
        if g['workers']:
            g['peak']=sum(max(g['peak'],w.get('peak',0),w.get('rss',0))
                          for w in g['workers'])/len(g['workers'])
    ram=max(0,available-floor+sum(w.get('rss',0) for g in groups.values() for w in g['workers']))
    capacities=dict(ram=ram)
    views=[s['gpu_view'] for s in states if s.get('gpu_view') and 0<=now-s['gpu_view']['time']<=20]
    if views:
        view=max(views,key=lambda v:v['time'])
        owned={pid for g in groups.values() for w in g['workers'] for pid in w.get('gpu_pids',[w['pid']])}
        reserve=sum(g['server_reserve'] for g in groups.values() if g['target'] or g['workers'])
        quantum=min((g['vram'] for g in groups.values() if g['vram']>0 and (g['target'] or g['workers'])),default=1)
        gpu=0
        for c in view['cards']:
            if not c.get('usable'):continue
            observed=sum(p.get('used_mib') or 0 for p in c.get('processes',[]) if p['pid'] in owned)
            # Card reserve: 10% capped at 2 GiB (an 80 GB card does not need 8 GB idle).
            reclaimable=max(0,c['total_mib']-c['used_mib']-max(512,min(2048,math.ceil(c['total_mib']/10)))-reserve+observed)
            # Per-card fragments cannot be pooled into a cross-GPU renderer.
            gpu+=math.floor(reclaimable/quantum)*quantum
        capacities['vram']=gpu
        usage={}
        for c in view['cards']:
            for p in c.get('processes',[]):usage[p['pid']]=usage.get(p['pid'],0)+(p.get('used_mib') or 0)
        for g in groups.values():
            if g['workers']:
                g['vram']=sum(max(g['vram'],sum(usage.get(pid,0) for pid in w.get('gpu_pids',[w['pid']])))
                              for w in g['workers'])/len(g['workers'])
    costs={n:dict(ram=g['peak'],**(dict(vram=g['vram']) if 'vram' in capacities else {})) for n,g in groups.items()}
    demands={n:g['target'] for n,g in groups.items()}
    quotas=allocate(capacities,demands,costs,epoch=int(now//60))
    return dict(quotas=quotas,capacity=capacities,costs=costs,demands=demands,
                vram_headroom=(capacities['vram']-sum(len(g['workers'])*g['vram'] for g in groups.values())) if 'vram' in capacities else None,
                campaigns=sum(v>0 for v in demands.values()),groups=groups)


def pending_gpu(states, cards):
    """Count growth promised by every participating fleet, across run roots."""
    pending={d:0 for d in cards}
    for s in states:
        for w in s.get('workers',[]):
            device=w.get('egl_device')
            if device not in cards:continue
            pids=set(w.get('gpu_pids',[w['pid']]))
            actual=sum(p.get('used_mib') or 0 for p in cards[device].get('processes',[]) if p['pid'] in pids)
            pending[device]+=max(0,s.get('worker_vram_mib',0)-actual)
    return pending
