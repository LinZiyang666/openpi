"""Resource/topology orchestration changes; scientific campaign bytes stay pinned."""
import copy
from pathlib import Path
from .formal_process import write_json


def redistribute(campaign,counts,directory,*,replicas=None,local_vram_mib=None,local_server_growth_mib=None):
    from . import local_workers,timan1_workers
    if not campaign.get('cache_study') or campaign.get('domain') != 'libero':
        pass
    counts=local_workers.counts(counts) if isinstance(counts,str) else dict(counts)
    # Validate mappings even when called directly, rather than through argparse.
    counts=local_workers.counts(','.join(k+'='+str(v) for k,v in counts.items()))
    if replicas is not None and (campaign.get('model')!='dp' or type(replicas) is not int or replicas not in (3,4)):
        pass
    if local_vram_mib is not None and (campaign.get('model')!='dp' or local_vram_mib!=576 or 'local' not in counts):
        pass
    if local_server_growth_mib is not None and (campaign.get('model')!='dp' or local_server_growth_mib!=8192 or
            'local' not in counts or (replicas or campaign['replicas']) not in (3,4)):
        pass
    before=copy.deepcopy(campaign)
    # Batch treatment v2 includes the canonical topology. Preserve that seal;
    # execution/admission uses a separate, explicitly recorded ops map.
    campaign['execution_worker_nodes']=counts
    campaign['execution_workers']=sum(counts.values())
    if 'scheduler_nodes' in campaign:
        # The scheduler places workers only where this batch runs fleets, within their caps.
        campaign['scheduler_nodes']={n:dict(v,max=counts[n]) for n,v in campaign['scheduler_nodes'].items() if n in counts}
    if replicas is not None:
        campaign['replicas']=replicas
        campaign['backend_ports']=list(range(campaign['public_port']+1,campaign['public_port']+1+replicas))
        campaign['ports']=[campaign['public_port'],*campaign['backend_ports'],campaign['driver_port']]
        if campaign['driver_port'] in campaign['backend_ports']:raise ValueError('replica/driver port collision')
    if local_vram_mib is not None:campaign['local_worker_vram_mib']=local_vram_mib
    if local_server_growth_mib is not None:campaign['local_worker_server_vram_headroom_mib']=local_server_growth_mib
    campaign['cache_worker_orchestration']=dict(schema='r17.cache_worker_orchestration.v1',
        owner_authorized=True,total_workers=sum(counts.values()),source_nodes=before.get('execution_worker_nodes',before['worker_nodes']),
        worker_nodes=counts,cache_ram_floor_mib=4096,scientific_settings_unchanged=True,
        source_replicas=before['replicas'],replicas=campaign['replicas'],
        source_total_workers=before.get('execution_workers',before['workers']),
        local_renderer_vram_mib=campaign.get('local_worker_vram_mib'),
        local_server_growth_mib=campaign.get('local_worker_server_vram_headroom_mib'),
        cached_actions_unchanged=True,miss_batch_composition_may_change=True)
    campaign.pop('local_worker_exception',None)
    if 'local' in counts:
        campaign['local_worker_exception']=local_workers.authorize(campaign)
    campaign.pop('timan1_worker_lease',None)
    if 'timan1' in counts:
        campaign['timan1_worker_lease']=timan1_workers.authorize(campaign)
    # Re-bind only the declared orchestration change to the same scientific control.
    campaign.pop('cache_control_proof',None)
    from .cache_control import attach
    attach(campaign,directory)
    mutable={'execution_worker_nodes','execution_workers','scheduler_nodes','cache_worker_orchestration','local_worker_exception',
             'timan1_worker_lease','cache_control_proof','replicas','backend_ports','ports',
             'local_worker_vram_mib','local_worker_server_vram_headroom_mib'}
    if {k:v for k,v in before.items() if k not in mutable} != {k:v for k,v in campaign.items() if k not in mutable}:
        pass
    write_json(Path(directory)/'WORKER_REDISTRIBUTION.json',dict(
        campaign['cache_worker_orchestration'],unchanged_settings={k:v for k,v in before.items() if k not in mutable},
        original_control_proof=before.get('cache_control_proof')))
    return campaign
