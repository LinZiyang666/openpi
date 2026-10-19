"""One preparation policy for R17 and legacy callers; no probes or launches.

VRAM inventory: devices.md §2.1/2.5; GPU2 identity/profile: timan1_gpu2.
Counts are desired maxima. CPU occupancy and RAM are runtime signals, not an
assumption of one blocked renderer per CPU core.
"""
import copy
import os
from pathlib import Path

from .policy import Config, NODES
from .switch import enabled


def physical_maximum(node, domain='libero'):
    from exp.offline_search.rounds.r17.filter_failed.timan1_gpu2 import CAPS, SIZES
    if node not in NODES:
        raise ValueError('disabled/unknown worker node: ' + node)
    if node == 'timan1':
        return CAPS[domain]
    if node in ('h100', 'a100'):
        from exp.r17_integration.ops.h100_workers import CPU_WORKERS
        return CPU_WORKERS[node]
    vram = SIZES[domain][0]
    cards = [8192] * 8 if node == 'timan107' else [49140]
    # Ten percent framebuffer reserve; current residents/server growth are
    # charged separately by node-local GPU admission before each spawn.
    return sum((total - (total + 9)//10 - 10)//vram for total in cards)


def maxima(domain='libero', nodes=('timan1', 'timan107')):
    return {n: physical_maximum(n, domain) for n in nodes if n in NODES}


def desired_maxima(domain, configured, hard_limits=None):
    """Old fixed counts are lower bounds, never small implicit runtime caps."""
    result={n:max(physical_maximum(n,domain),int(v)) for n,v in configured.items()
            if n in NODES and int(v)>0}
    for node,limit in (hard_limits or {}).items():
        if node not in result:continue
        if (not isinstance(limit,dict) or type(limit.get('maximum')) is not int
                or limit['maximum']<1 or not str(limit.get('reason','')).strip()):
            raise ValueError('worker_hard_limits requires positive maximum and technical reason: '+node)
        result[node]=min(result[node],limit['maximum'])
    return result



def native_counts(counts):
    """Retain the native total while routing the disabled host to timan107."""
    result={n:int(v) for n,v in counts.items() if n in NODES and int(v)>0}
    if counts.get('timan108',0):result['timan107']=result.get('timan107',0)+int(counts['timan108'])
    return result


def apply(campaign, directory=None, *, force=False):
    """Mutate resource fields only; preserve canonical scientific pairing fields.

    A native fixed continuation pins its original driver and workers byte for
    byte. Do not claim it gained instrumentation by changing its JSON alone.
    """
    c = campaign
    if not enabled():
        was_dynamic = bool(c.get("autoscale"))
        if was_dynamic and directory and (Path(directory)/'PREPARED.json').exists():
            root=Path(directory)
            if any((root/n).exists() for n in ('LAUNCH.json','RECEIPT.json','DONE.json','launch.json')):
                raise ValueError('OFF reprepare refuses an already launched dynamic campaign; use a closed-zero reset')
            import time
            (root/'PREPARED.json').rename(root/('PREPARED.before_off.'+str(time.time_ns())+'.json'))
        for key in list(c):
            if key.startswith("autoscale"):c.pop(key)
        selected=c.get("execution_worker_nodes",c.get("worker_nodes",{}))
        if "timan108" in selected:
            c["execution_worker_nodes"]=native_counts(selected)
            c["execution_workers"]=sum(c["execution_worker_nodes"].values())
        c.pop("worker_max_seconds", None)
        if was_dynamic and c.get("domain") == "metaworld":c["worker_max_episodes"] = 0
        return c
    if c.get('worker_nodes') == {} and 'domain' not in c and 'workers' not in c:
        return c  # prepare_workers also exposes a CPU payload-only builder
    if directory and not force and (Path(directory)/'PREPARED.json').exists():
        return c
    if c.get('cache_resume_selection') or c.get('resume_from'):
        from .node import read
        source = (c.get('cache_resume_selection') or {}).get('source') or c.get('resume_from')
        inherited = read(Path(source)/'campaign.json', {}) if source else {}
        if inherited.get('autoscale'):
            c['autoscale'] = dict(inherited['autoscale'])
        elif not c.get('cache_resume_selection'):
            # resume_from replays the source driver in place; only cache
            # continuations rebuild their driver (filter_failed.stage).
            c.update(autoscale=None, autoscale_disposition='kept_fixed',
                     autoscale_reason='native continuation retains source driver/control/worker pins')
            return c
        else:
            c['autoscale'] = dict(mode='dynamic')
    c.setdefault('autoscale', dict(mode='dynamic'))
    if c['autoscale'] is None:
        c['autoscale'] = dict(mode='dynamic')
    Config(**c['autoscale'])
    # timan107 floor above the memory guard, and RAM charged for non-ledger
    # (R15/R16) fleets on every node.
    from .formal_dynamic import NODE_CONFIG
    c.setdefault('autoscale_node_config', copy.deepcopy(NODE_CONFIG))
    c.setdefault('autoscale_external_legacy_nodes', sorted(c['autoscale_node_config']))
    # Owner 10-19: the A100 is a pure worker machine (no server), so it keeps only a small RAM floor.
    c['autoscale_node_config'].setdefault('a100', dict(floor=8*2**30, critical=4*2**30, hold_s=10))
    c.setdefault('autoscale_mature_s', 60)  # a LIBERO/MetaWorld worker has loaded its env within a minute
    if c.get('autoscale_maxima_policy')=='physical-v2' and c.get('autoscale_node_maxima') and not force:
        return c  # preserve per-node views made by worker_resilience.prepare
    domain = c.get('domain', 'libero')
    specified = c.get('autoscale_node_maxima') or c.get('execution_worker_nodes') or c.get('worker_nodes')
    if not specified and c.get('workers'):
        specified = {c.get('worker_node', 'timan107'): c['workers']}
    selected = {n: int(v) for n, v in (specified or maxima(domain)).items() if n in NODES and int(v) > 0}
    if not selected:
        selected = ({'timan107':int(specified['timan108'])}
                    if specified and set(specified)=={'timan108'} and int(specified['timan108'])>0
                    else maxima(domain, ('timan107',)))
    # h100-temp (R17_H100_NODE alias) renderers hit NVRM Xid 31 MMU faults and abort
    # (10-18 21:18-22:21, also with 15 GB free); the H100 itself shows none.
    if c.get('server_node') == 'h100' and domain == 'libero' and os.environ.get('R17_H100_NODE', 'h100') == 'h100':
        # Owner 10-18: workers on the server's own machine serve it first.
        selected.setdefault('h100', physical_maximum('h100', domain))
    if c.get('server_node') == 'local':
        # Owner 10-18: the 4090's own workers serve the 4090's server first.
        selected.setdefault('local', physical_maximum('local', domain))
    if domain == 'libero' and c.get('server_node') == 'h100':
        # Owner 10-18: worker machine; 10-19: it serves the H100's batches only.
        selected.setdefault('a100', physical_maximum('a100', domain))
    selected=desired_maxima(domain,selected,c.get('worker_hard_limits'))
    c.update(execution_worker_nodes=selected, execution_workers=sum(selected.values()),
             autoscale_node_maxima=selected, autoscale_maximum=sum(selected.values()),
             autoscale_disposition=c['autoscale'].get('mode', 'dynamic'),autoscale_maxima_policy='physical-v2')
    c.setdefault('worker_nodes', dict(selected))
    c.setdefault('workers', sum(selected.values()))
    from exp.offline_search.rounds.r17.filter_failed.timan1_gpu2 import SIZES
    c['worker_vram_mib'] = SIZES[domain][0]
    if domain == 'metaworld' and not c.get('worker_max_episodes'):
        c.update(worker_max_episodes=32, worker_max_seconds=1800)
    return c


def queue_workers(unit):
    """Migrate historical queue defaults at materialization, not queue scripts.

    Upgrade every old fixed-count map; only attested technical caps may be lower.
    """
    if not enabled():return native_counts(unit.get('workers', {}))
    explicit = unit.get('autoscale_node_maxima')
    old = unit.get('workers', {})
    selected={n:int(v) for n,v in (explicit or old).items() if n in NODES and int(v)>0}
    domain=unit.get('domain','metaworld' if unit.get('benchmark')=='mt50' else 'libero')
    return desired_maxima(domain,selected or maxima(domain,('timan107',)),unit.get('worker_hard_limits'))
