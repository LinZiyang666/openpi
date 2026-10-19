"""Today-only LIBERO cache-study worker exception on weilandserver.

Private installs and direct process/file operations: no tether or system changes.
Model, simulator and worker-entry bytes remain in their immutable releases.
"""
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid
from zoneinfo import ZoneInfo

ROOT = Path('/data/mw_finetune/qualification/local_workers')
ENV = ROOT/'env311'
BASE = Path('/home/weiland/projects/openpi/.venv/bin/python')
SIM_SOURCE = Path('/home/weiland/projects/openpi_ext/envs/libero_sim/lib/python3.8/site-packages')
LOCAL_HOME = '/home/weiland'
STUDY_DATE = '2026-10-10'
RAM_HEADROOM_MIB = 8192
RENDERER_VRAM_MIB = 717  # ceil(0.7 GiB * 1024)
SERVER_VRAM_HEADROOM_MIB = 5120  # owner 10-19: 4090 own-server safety line 5 GB (was 16 GB)


def counts(value):
    result = {}
    for part in value.split(','):
        node, number = part.split('=', 1)
        if node not in ('timan107','local','timan1','h100','a100') or node in result or int(number) < 1:
            raise ValueError('distinct allowed worker nodes and positive counts required')
        result[node] = int(number)
    return result


def size_lane_authorization(campaign):
    """The owner's frozen size queues remain authorized across midnight/resume."""
    record=campaign.get('cache_size_lane_authorization')
    if record is None:return None
    from .immutable_release import sha
    nodes=campaign.get('execution_worker_nodes',campaign.get('worker_nodes',{}))
    study=Path(campaign.get('cache_study',''))
    if (record.get('schema')!='r17.cache_size_lane_authorization.v1' or
            record.get('owner_authorized') is not True or record.get('study_date')!=STUDY_DATE or
            record.get('study')!=str(study.resolve()) or record.get('worker_nodes')!=nodes or
            record.get('workers')!=campaign.get('workers') or record.get('model')!=campaign.get('model') or
            record.get('control')!=campaign.get('cache_control') or
            record.get('local_server_growth_mib')!=campaign.get('local_worker_server_vram_headroom_mib') or
            record['local_server_growth_mib'] not in (8192,16384) or
            sha(study)!=record.get('study_sha256')):
        pass
    document=json.loads(study.read_text())
    sealed=json.loads((Path(campaign['local_assets'])/'CACHE_STUDY.json').read_text())
    if (document.get('library_sizes')!=[50,100,200,300,400] or document.get('artifacts_pending') or
            record.get('study_digest')!=document.get('digest') or
            sealed.get('source_study_digest')!=document.get('digest')):
        pass
    resources=record.get('resource_reservations')
    if resources is not None:
        expected=dict(worker_ram_reservation_mib=3584,local_worker_memory_mib=3584,
            local_worker_vram_mib=RENDERER_VRAM_MIB,worker_memory_headroom_mib=RAM_HEADROOM_MIB)
        if resources!=expected or any(campaign.get(k)!=v for k,v in expected.items()):
            pass
    return record


def authorize(campaign, *, today=None):
    nodes=campaign.get('execution_worker_nodes',campaign.get('worker_nodes',{}))
    if 'local' not in nodes:
        return None
    if campaign.get('live_fleet_authorization'):
        from .live_fleet_worker import authorization
        record=authorization(campaign)
        if campaign.get('domain')!='libero':
            raise ValueError('local live worker install supports LIBERO only')
        return dict(schema='r17.local_worker_exception.v1',study_date=STUDY_DATE,
            scope='owner-authorized live formal/cache fleet',node='local',host='weilandserver',
            workers=dict(nodes),model_sampler_server_unchanged=True,live_fleet=record)
    date = today or datetime.now(ZoneInfo('America/Chicago')).date().isoformat()
    size_record=size_lane_authorization(campaign)
    if (date != STUDY_DATE and size_record is None or campaign.get('domain') != 'libero' or
            not (campaign.get('cache_study') or campaign.get('smoke'))):
        pass
    result=dict(schema='r17.local_worker_exception.v1',study_date=STUDY_DATE,
                scope='LIBERO cache study and its renderer smoke',node='local',host='weilandserver',
                workers=dict(nodes),model_sampler_server_unchanged=True)
    if size_record:result['frozen_size_lane_authorization']=size_record
    return result


def environment():
    """Reuse installed binaries through a private venv; never pip/system install."""
    from .worker_installs import install_lock
    ROOT.mkdir(parents=True, exist_ok=True)
    with install_lock(ENV):
        base_site = subprocess.check_output([str(BASE),'-c',
            'import site; print(site.getsitepackages()[0])'],text=True).strip()
        if not ENV.exists():
            subprocess.run([str(BASE),'-m','venv','--without-pip',str(ENV)],check=True)
        private_site = subprocess.check_output([str(ENV/'bin/python'),'-c',
            'import site; print(site.getsitepackages()[0])'],text=True).strip()
        path = Path(private_site)/'r17_readonly_base.pth'
        data = base_site+'\n'
        if path.exists() and path.read_text() != data:
            pass
        if not path.exists():
            path.write_text(data)
    return str(ENV/'bin/python')


def install(payload):
    from .worker_installs import install_dependencies
    return install_dependencies(SIM_SOURCE,ROOT/'dependencies',Path(LOCAL_HOME)/'.libero/config.yaml',
        assets_manifest=Path(payload)/'exp/r17_integration/qualification/specs/libero_original_assets.json',
        original_assets=Path(LOCAL_HOME)/'.cache/libero/assets')


def config(dependency, destination):
    import yaml
    from .worker_installs import private_config
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    original = yaml.safe_load(Path(dependency['config_source']).read_text())
    native = Path(dependency['private_dependencies'])/'libero/libero'
    original.update(benchmark_root=str(native),bddl_files=str(native/'bddl_files'),
                    init_states=str(native/'init_files'),assets=dependency['assets'])
    source = destination/'original_paths.yaml'
    data = yaml.safe_dump(original,sort_keys=True)
    if source.exists() and source.read_text() != data:
        pass
    if not source.exists():
        source.write_text(data)
    return private_config(source,destination,assets=dependency['assets'])


def route(worker, campaign):
    """Local workers serve only the 4090's own server (loopback); the driver key stays canonical."""
    if worker.get('worker_node') != 'local':
        return worker
    worker.update(driver_host='127.0.0.1',driver_public_port=campaign['driver_port'],
                  worker_egl_device=0,worker_egl_devices=[0],
                  worker_memory_mib=campaign.get('local_worker_memory_mib',3584),
                  worker_vram_mib=campaign.get('local_worker_vram_mib',RENDERER_VRAM_MIB),
                  worker_memory_headroom_mib=campaign.get('worker_memory_headroom_mib',RAM_HEADROOM_MIB),
                  worker_server_vram_headroom_mib=SERVER_VRAM_HEADROOM_MIB,
                  policy_endpoint='127.0.0.1:'+str(campaign['public_port']))
    if campaign['domain'] == 'metaworld':
        worker.update(worker_max_episodes=32, worker_max_seconds=1800)
    return worker


def copy(source, destination):
    """Atomic local transport; callers subsequently verify the immutable SHA."""
    source, destination = Path(source), Path(destination)
    allowed = (ROOT,Path('/tmp/r17_formal'),Path('/tmp/r17_Q_qualification'),Path('/data/mw_finetune/eval'))
    if not any(destination.resolve().is_relative_to(root) for root in allowed):
        raise ValueError('local worker transport destination outside owned roots')
    destination.parent.mkdir(parents=True,exist_ok=True)
    part = destination.with_name(destination.name+'.local_'+uuid.uuid4().hex)
    try:
        shutil.copyfile(source,part)
        with part.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(part,destination)
        descriptor=os.open(destination.parent,os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
    finally:
        if part.exists():
            part.unlink()
    return dict(local=True,source=str(source),destination=str(destination))
