"""Read-only control pins and an explicit R17 cache-admission runtime overlay.

Everything outside R17 is inherited byte-for-byte from the selected control.
The reviewed R17 model admission hooks certify native producer equivalence;
their live library validators still run during the CPU seal and publication.
This is deliberately separate from ordinary --runtime-from semantics.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile

from .immutable_release import archive_manifest, identity, sha
from .formal_process import write_json


def read(path):
    return json.loads(Path(path).read_text())


def control_proof(control, runtime, campaign):
    control, runtime = Path(control).resolve(strict=True), Path(runtime).resolve(strict=True)
    old, receipt, summary = (read(control/n) for n in ('campaign.json','RECEIPT.json','SUMMARY.json'))
    if (receipt.get('all_owned_closed') is not True or receipt.get('status') == 'PENDING_EVIDENCE' or
            receipt.get('worker_storage_cleanup_pending')):
        pass
    if (summary.get('status') != 'PASS' or summary.get('completed_episodes') != 2000 or
            summary.get('expected_episodes') != 2000 or summary.get('full_arm_execution_verified') is not True or
            summary.get('terminal_errors') != 0):
        pass
    from .formal_eval import driver_completion
    completion = driver_completion(control, 0)
    recovery = None
    if receipt.get('status') != 'PASS':
        # The owner-selected Cosmos merge has a stale outer rc2 failure. Only
        # its fully sealed CPU recovery/merge can attest this exact exception.
        merge = read(control/'RESUME_MERGE.json')
        driver = read(control/'driver/DRIVER_RECEIPT.json')
        if (campaign['model'] != 'cosmos' or merge.get('status') != 'PASS' or
                driver.get('resume',{}).get('cpu_recovery') is not True or
                not driver.get('resume',{}).get('segment_completion',{}).get('verified_completion')):
            pass
        recovery = dict(original_status=receipt['status'], original_error=receipt.get('error'),
                        cpu_merged_completion=completion, merge_sha256=sha(control/'RESUME_MERGE.json'))
    fields = ('model','checkpoint','server_node','replicas','workers','worker_nodes',
              'worker_memory_mib','worker_vram_mib','worker_max_episodes','worker_egl_devices',
              'memory_budget_gib','mps','fastwam_batching','dp_batching')
    topology_exception = campaign.get('local_worker_exception')
    execution_nodes=campaign.get('execution_worker_nodes',campaign.get('worker_nodes',{}))
    if topology_exception:
        from .local_workers import STUDY_DATE
        if (topology_exception.get('study_date') != STUDY_DATE or
                topology_exception.get('workers') != execution_nodes or
                'local' not in execution_nodes or campaign.get('domain') != 'libero' or
                not campaign.get('cache_study')):
            pass
    changes = {}
    resource_changes={}
    orchestration=campaign.get('cache_worker_orchestration')
    capacity=None
    if campaign.get('cache_capacity_authorization'):
        from .cache_capacity import validate_authorization
        capacity=validate_authorization(campaign)
    execution_workers=campaign.get('execution_workers',campaign['workers'])
    if orchestration and (orchestration.get('owner_authorized') is not True or
            orchestration.get('total_workers') != execution_workers or
            orchestration.get('worker_nodes') != execution_nodes or
            sum(execution_nodes.values()) != execution_workers):
        pass
    if orchestration and 'replicas' in orchestration and orchestration['replicas']!=campaign['replicas']:
        pass
    if campaign.get('timan1_worker_lease'):
        from .timan1_workers import authorize
        if authorize(campaign) != campaign['timan1_worker_lease']:
            pass
    for field in fields:
        if old.get(field) != campaign.get(field):
            if capacity and field=='replicas' and campaign['model']=='dp':
                resource_changes[field]=dict(control=old[field],campaign=campaign[field],reason='owner capacity request; cached actions unchanged')
                continue
            if (orchestration and field=='replicas' and campaign['model']=='dp' and
                    orchestration.get('replicas')==campaign['replicas'] and campaign['replicas'] in (3,4)):
                resource_changes[field]=dict(control=old[field],campaign=campaign[field],
                    reason='owner cache capacity request; gated MISS batch composition may change')
                continue
            if (topology_exception and field in ('workers','worker_nodes') or
                    orchestration and field == 'worker_nodes'):
                changes[field] = dict(control=old.get(field),campaign=campaign.get(field))
                continue
            if (field=='worker_memory_mib' and campaign.get('cache_study') and
                    type(old.get(field)) is int and type(campaign.get(field)) is int and
                    campaign[field] == max(4096,old[field])):
                resource_changes[field]=dict(control=old[field],campaign=campaign[field],reason='measured cache RSS3.5GiB')
                continue
            pass
    original = read(runtime/'campaign.json')
    if control != runtime:
        if Path(old.get('resume_from','')).resolve() != runtime:
            pass
        from .formal_resume import check_pins
        check_pins(old)
    release = read(runtime/'RUNTIME_RELEASE.json')
    if sha(runtime/'runtime.tar') != release['archive_sha256'] or identity(archive_manifest(runtime/'runtime.tar')) != release['identity']:
        pass
    for name in ('model_spec.json','model_manifest.json','benchmark.json'):
        if read(runtime/'assets'/name) != read(control/'assets'/name):
            pass
    if original['model'] != campaign['model']:
        pass
    pins = {str(p):sha(p) for root in {control,runtime} for p in (
        root/'campaign.json', root/'RECEIPT.json', root/'runtime.tar',root/'workers.tar',
        root/'assets/model_spec.json',root/'assets/model_manifest.json',root/'assets/benchmark.json')}
    for name in ('SUMMARY.json','RESUME_MERGE.json','driver/DRIVER_RECEIPT.json','driver/journal.jsonl'):
        p=control/name
        if p.exists():pins[str(p)]=sha(p)
    return dict(schema='r17.cache_control.v1',control=str(control),runtime_source=str(runtime),
                source_pins=pins,completion=completion,recovery=recovery,non_r17_code_inherited=True,
                **(dict(local_worker_exception=topology_exception,worker_topology_changes=changes) if topology_exception or orchestration else {}),
                **(dict(worker_resource_changes=resource_changes) if resource_changes else {}))


def attach(campaign, directory):
    if not campaign.get('cache_control'):return
    proof=control_proof(campaign['cache_control'],campaign['cache_control_runtime'],campaign)
    if campaign.get('cache_control_proof') and campaign['cache_control_proof']!=proof:
        pass
    campaign['cache_control_proof']=proof
    write_json(Path(directory)/'CACHE_CONTROL.json',proof)


def check(campaign):
    for path,pin in campaign.get('cache_control_proof',{}).get('source_pins',{}).items():
        if sha(path)!=pin:pass


def overlay_archive(base, fresh, destination):
    """Never edit a frozen release; non-R17 bytes must remain exact."""
    def contents(path):
        with tarfile.open(path) as archive:
            result={}
            for m in archive.getmembers():
                from .build_client_bundle import safe_name
                safe_name(m.name)
                if not m.isfile() or m.name in result:raise ValueError('invalid cache runtime member')
                result[m.name]=archive.extractfile(m).read()
            return result
    old,new=contents(base),contents(fresh)
    content=dict(old)
    content.update({n:b for n,b in new.items() if n.startswith('exp/r17_integration/')})
    # Original Q3 implementation is read-only and explicitly supplied by P6.
    from .cache_study import stall_runtime_files
    content.update({n:new[n] for n in stall_runtime_files()})
    changed=sorted(n for n,b in content.items() if old.get(n)!=b)
    if any(not n.startswith('exp/r17_integration/') and n not in stall_runtime_files() for n in changed):
        pass
    with tarfile.open(destination,'w') as out:
        for name,data in sorted(content.items()):
            member=tarfile.TarInfo(name);member.size=len(data);member.mode=0o444
            out.addfile(member,io.BytesIO(data))
    return dict(files=changed,non_r17_exact=True,base_archive_sha256=sha(base),
                overlay_archive_sha256=sha(destination),files_sha256={n:hashlib.sha256(content[n]).hexdigest() for n in changed})


def runtime_archive(campaign, directory, root):
    from exp.r17_integration.qualification.deploy_cpu import runtime_archive as build
    from .cache_study import append_stall_runtime
    check(campaign)
    directory=Path(directory);fresh=directory/'cache_framework.tar'
    build(root,fresh);append_stall_runtime(fresh,root)
    proof=overlay_archive(Path(campaign['cache_control_runtime'])/'runtime.tar',fresh,directory/'runtime.tar')
    write_json(directory/'CACHE_RUNTIME_OVERLAY.json',proof)


def worker_payload(campaign, directory, root, names):
    """Freeze baseline simulator/client bytes, adding only P6 ops observers."""
    from .build_client_bundle import verified_members, build_client_bundle
    import tempfile
    check(campaign)
    refresh=campaign.get('cache_worker_refresh')
    baseline=Path(refresh['archive']) if refresh else Path(campaign['cache_control_runtime'])/'workers.tar'
    if refresh and sha(baseline)!=refresh['archive_sha256']:
        pass
    manifest, content=verified_members(baseline,'PAYLOAD.json')
    with tempfile.TemporaryDirectory(prefix='cache_worker_',dir=directory) as temp:
        temp=Path(temp)
        for name,data in content.items():
            if name=='PAYLOAD.json':continue
            p=temp/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        overlay=(list(refresh['files']) if refresh else
                 [n for n in names if n.startswith('exp/r17_integration/ops/')])
        recycle_files={'exp/r17_integration/ops/formal_workers.py','exp/r17_integration/ops/worker_gpu.py'}
        helper_files={'exp/r17_integration/ops/worker_disk.py','exp/r17_integration/ops/sync_back.py'}
        if refresh and (not recycle_files<=set(overlay) or set(overlay)-recycle_files-helper_files):
            pass
        for name in overlay:
            if refresh and sha(Path(root)/name)!=refresh['files'][name]:
                pass
            p=temp/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((Path(root)/name).read_bytes())
        all_names=sorted(set(manifest['files'])|set(overlay))
        output=Path(directory)/'workers.tar'
        built=temp/'refreshed_workers.tar' if refresh else output
        payload=build_client_bundle(temp,all_names,built)
        if refresh:
            built.replace(output)
    unchanged={n:pin for n,pin in manifest['files'].items() if n not in overlay}
    if any(payload['manifest']['files'].get(n)!=pin for n,pin in unchanged.items()):
        pass
    write_json(Path(directory)/'CACHE_WORKER_OVERLAY.json',dict(unchanged=unchanged,ops_overlay=overlay))
    return payload


def refresh_prepared_workers(campaign, directory, deployment):
    """Refresh only recycle helpers of an undispatched cache fleet.

    Called under the preparation lock, before the one-shot launch claim. Live
    fleets and all native runtime/simulator/driver bytes remain immutable.
    """
    from . import formal_eval as launch
    from .build_client_bundle import verified_members
    from .formal_prestage import seal_prepared,validate_prepared
    import shutil
    import time
    directory=Path(directory)
    if not campaign.get('cache_study') or not campaign.get('cache_control'):
        return campaign,deployment
    if (directory/'LAUNCH.json').exists() or (directory/'RECEIPT.json').exists():
        pass
    files={n:sha(launch.ROOT/n) for n in (
        'exp/r17_integration/ops/formal_workers.py','exp/r17_integration/ops/worker_gpu.py',
        'exp/r17_integration/ops/worker_disk.py','exp/r17_integration/ops/sync_back.py')}
    original,_=verified_members(directory/'workers.tar','PAYLOAD.json')
    auxiliary=[]
    if 'timan1' in campaign.get('execution_worker_nodes',campaign.get('worker_nodes',{})):
        auxiliary.append('exp/r17_integration/ops/timan1_workers.py')
        if campaign.get('cache_capacity_authorization'):
            auxiliary.append('exp/r17_integration/ops/cache_capacity.py')
    if (all(original['files'].get(n)==pin for n,pin in files.items()) and
            all(original['files'].get(n)==sha(launch.ROOT/n) for n in auxiliary)):
        return campaign,deployment
    before=read(directory/'PREPARED.json')
    mutable={'campaign.json','WORKER_DEPLOYMENT.json','CACHE_WORKER_OVERLAY.json','LOCAL_WORKER_ORCHESTRATION.json',
             *[node+'.json' for node in deployment]}
    native={n:sha(directory/n) for n in before['files'] if n not in mutable}
    native['runtime.tar']=sha(directory/'runtime.tar')
    previous=directory/('PREPARED_BEFORE_RECYCLE_'+str(time.time_ns())+'.json')
    write_json(previous,before)
    base=directory/'CACHE_RECYCLE_BASE.tar'
    if base.exists():pass
    shutil.copyfile(directory/'workers.tar',base)
    campaign=dict(campaign,cache_worker_refresh=dict(archive=str(base.resolve()),archive_sha256=sha(base),files=files))
    write_json(directory/'campaign.json',campaign)
    with launch.retry_scope(directory):deployment=launch.prepare_workers(campaign,directory)
    if any(sha(directory/n)!=pin for n,pin in native.items()):
        pass
    write_json(directory/'CACHE_RECYCLE_REFRESH.json',dict(status='PASS',cpu_only=True,
        previous_seal=previous.name,previous_seal_sha256=sha(previous),native_pins=native,
        files=files,base_archive_sha256=sha(base),worker_archive_sha256=sha(directory/'workers.tar')))
    seal_prepared(campaign,directory,deployment)
    return validate_prepared(directory)
