"""Owner-authorized T1 worker lease: physical GPU3 only, verified namespace guard."""
import json
from pathlib import Path

UUID = 'GPU-ba276422-a0b1-e371-8327-f744be942f15'
PCI = '0000:c1:00.0'
PYTHON = '/scratch/zixuans8/openpi/.venv/bin/python'
PROVISION = '/scratch/zixuans8/mw_r17/timan1_provision'
WRAPPER = PROVISION+'/python-gpu3'
MW_WRAPPER = PROVISION+'/python-mw-gpu3'
GUARD = PROVISION+'/gpu3_guard'
PINS = {WRAPPER:'64199dc0bc944087366a45282861896fdc2b836c1ac0a9453f0375a4154e6ad2',
        GUARD+'/sitecustomize.py':'4bb5b8280cfa21328c9d07c4aae4b8220fc9b68605d6b3160288b4fab98bb2b8',
        PROVISION+'/libero_config/config.yaml':'dc071fb958f2effdba6b7b85cee02cc9073b76332fe89c5f4e77877b9f207dca'}
DEPENDENCY_ID = 'c786215bc2061c66ed8851f6aa32e7199a9f2e364b12c47e61235f5a3e83a7c7'
ASSET_ID = '3eb6e3e7ee3a03663dba84ccc6d6cf9dbb4ae21fb20558ca4097abc066be57dc'


def authorize(campaign):
    number = campaign.get('execution_worker_nodes',campaign.get('worker_nodes',{})).get('timan1',0)
    if not number:
        return None
    maximum=40
    live=None
    if campaign.get('live_fleet_authorization'):
        from .live_fleet_worker import authorization
        live=authorization(campaign)
        maximum=40
    if campaign.get('cache_capacity_authorization'):
        from .cache_capacity import validate_authorization
        validate_authorization(campaign)
        maximum=40
    mw = campaign.get('domain') == 'metaworld'
    if not (mw or campaign.get('domain') == 'libero' and (campaign.get('cache_study') or live)) or not 1 <= number <= maximum:
        raise ValueError('timan1 owner lease requires MetaWorld or LIBERO cache workers, maximum 40, GPU3 only')
    return dict(schema='r17.timan1_gpu3_lease.v1',workers=number,device=3,device_minor=0,
                uuid=UUID,pci=PCI,excluded_devices=[0,1,2],pins=PINS,
                worker_memory_mib=2048 if mw else 6144,worker_vram_mib=1024,worker_max_episodes=0 if mw else 4)


def wrapper(campaign):
    return MW_WRAPPER if campaign.get('domain') == 'metaworld' else WRAPPER


def install(payload):
    """Read-only revalidation of T1's original assets/dependencies; no download."""
    from .worker_installs import sha,verify_dependencies,asset_manifest
    for filename,pin in PINS.items():
        if sha(filename) != pin:
            raise ValueError('T1 namespace/config guard changed: '+filename)
    root='/scratch/zixuans8/mw_r17/formal'
    document=asset_manifest(Path(payload)/'exp/r17_integration/qualification/specs/libero_original_assets.json')
    if document['digest'] != ASSET_ID:
        raise ValueError('T1 original assets differ from sealed benchmark')
    release=dict(private_dependencies=root+'/dependencies/libero311_'+DEPENDENCY_ID,
        content_identity=DEPENDENCY_ID,config_source=PROVISION+'/libero_config/config.yaml',
        config_sha256=PINS[PROVISION+'/libero_config/config.yaml'],assets_digest=ASSET_ID,
        asset_release=root+'/assets/libero_'+ASSET_ID,assets=root+'/assets/libero_'+ASSET_ID+'/assets')
    verify_dependencies(release)
    return release


def route(worker,campaign):
    if worker.get('worker_node') != 'timan1':
        return worker
    lease=authorize(campaign)
    if not lease or worker['node_workers'] != lease['workers']:
        raise ValueError('T1 worker count differs from guarded lease')
    worker.update(worker_egl_device=3,worker_egl_devices=[3],worker_memory_mib=lease['worker_memory_mib'],
                  worker_vram_mib=1024,worker_max_episodes=lease['worker_max_episodes'],timan1_worker_lease=lease)
    return worker


def environment(env):
    return dict(env,PYTHONPATH=GUARD+':'+env['PYTHONPATH'],MUJOCO_EGL_DEVICE_ID='3',
                CUDA_VISIBLE_DEVICES=UUID)


def record_native_process():
    """Bind the Python child inside bwrap to this unique fleet generation.

    The pinned namespace wrapper waits for a different PID and forwards its
    exit status. Keep the original lifetime marker unchanged; attest the native
    PID and wrapper ancestry before entering the original worker instead.
    """
    import os
    from .formal_process import ticks,write_json
    token=os.environ['R17_T1_GENERATION_TOKEN']
    if len(token)!=64 or any(c not in '0123456789abcdef' for c in token):
        raise ValueError('invalid wrapped worker generation token')
    lineage=[]
    pid=os.getpid()
    for _ in range(32):
        if pid<1:break
        stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        lineage.append(dict(pid=pid,start_ticks=ticks(pid)))
        pid=int(stat[1])
    record=dict(schema='r17.timan1_native_process.v1',generation_token=token,
                native=lineage[0],ancestors=lineage[1:])
    path=Path(os.environ['R17_WORKER_RECYCLE_RECEIPT']).with_name('NATIVE_PROCESS.json')
    write_json(path,record)
    return record


def planned_recycle(entry,code):
    """Accept exit75 only for the bound native child at its N4 boundary."""
    from .worker_lifetime import RECYCLE_EXIT
    if (code!=RECYCLE_EXIT or entry.get('node')!='timan1' or
            entry.get('argv',[None])[0]!=WRAPPER or not entry.get('wrapped_generation_token')):
        return False
    try:
        path=Path(entry['recycle_receipt'])
        marker=json.loads(path.read_text())
        record=json.loads(path.with_name('NATIVE_PROCESS.json').read_text())
        owner=dict(pid=entry['pid'],start_ticks=entry['start_ticks'])
        native=record['native']
        valid=(record.get('schema')=='r17.timan1_native_process.v1' and
            record.get('generation_token')==entry['wrapped_generation_token'] and
            owner in record['ancestors'] and native['pid']!=owner['pid'] and
            type(native['pid']) is int and native['pid']>0 and bool(native['start_ticks']) and
            marker.get('schema')=='r17.worker_recycle.v1' and
            marker.get('pid')==native['pid'] and marker.get('start_ticks')==native['start_ticks'] and
            marker.get('reason')=='episode_boundary' and
            type(marker.get('maximum_episodes')) is int and
            marker['maximum_episodes']==entry.get('maximum_episodes')==4 and
            marker.get('completed_episodes')==4)
    except (OSError,ValueError,KeyError,TypeError):
        return False
    if valid:
        entry['wrapped_recycle']=dict(native=native,wrapper=owner,completed_episodes=4,
                                     generation_bound=True)
    return valid


def main(argv=None):
    """Orchestration relay inside T1's unchanged, GPU3-only namespace."""
    import os
    import sys
    args=list(sys.argv[1:] if argv is None else argv)
    if (os.environ.get('R17_GPU3_DEVICE_ISOLATION')!='1' or
            os.environ.get('MUJOCO_EGL_DEVICE_ID')!='3' or
            os.environ.get('CUDA_VISIBLE_DEVICES') not in ('',UUID) or
            '--egl-device' not in args or args[args.index('--egl-device')+1]!='3'):
        raise ValueError('wrapped native worker requires the pinned GPU3 namespace')
    record_native_process()
    if '--max-episodes' in args:
        from .worker_lifetime import main as native_main
    else:
        from exp.r17_integration.conductor.worker_entry import main as native_main
    native_main(args)


if __name__=='__main__':
    main()
