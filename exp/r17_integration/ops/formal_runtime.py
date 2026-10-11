"""Verified baseline code inheritance; reading a live control never modifies it."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from .immutable_release import archive_manifest, identity, sha
from .formal_process import write_json


def read(path):
    return json.loads(Path(path).read_text())


def paired_lineage(campaign, source, *, require_complete=False):
    """A merged continuation may price a control using its original code release.

    Completion is never deferred at launch. During CPU preparation only the
    immutable selection/native pins are required, so a live resume stays read-only.
    """
    source = Path(source).resolve(strict=True)
    control = Path(campaign['paired_control']).resolve(strict=True)
    if control == source:
        if require_complete:
            receipt = read(control/'RECEIPT.json')
            if (receipt.get('status') != 'PASS' or receipt.get('all_owned_closed') is not True or
                    receipt.get('worker_storage_cleanup_pending')):
                pass
        return None
    from .formal_resume import check_pins, same_campaign
    old, resumed = read(source/'campaign.json'), read(control/'campaign.json')
    if not resumed.get('resume_from') or Path(resumed['resume_from']).resolve() != source:
        pass
    same_campaign(old, resumed)
    check_pins(resumed)
    selection = read(control/'RESUME.json')
    if (Path(selection['source']).resolve() != source or
            selection['source_pins'] != resumed['resume_source_pins'] or
            not selection['retained_episodes'] or not selection['dispatched_episodes'] or
            selection['retained_episodes']+selection['dispatched_episodes'] != selection['expected_episodes'] or
            len(set(selection['retained_task_uids'])) != selection['retained_episodes'] or
            len(set(selection['missing_task_uids'])) != selection['dispatched_episodes'] or
            set(selection['retained_task_uids']) & set(selection['missing_task_uids'])):
        pass
    native = ('runtime.tar','workers.tar','assets/model_manifest.json',
              'assets/model_spec.json','assets/benchmark.json','assets/full.yaml')
    pins = {name:sha(source/name) for name in native}
    if any(sha(control/name) != pin for name,pin in pins.items()) or old['runtime_identity'] != resumed['runtime_identity']:
        pass
    original = read(source/'EPISODE_PLAN.json')
    full = read(control/'FULL_EPISODE_PLAN.json')
    def semantics(task):
        return {key:value for key,value in task.items() if key not in ('server_host','server_port')}
    tasks = {t['task_uid']:semantics(t) for t in original['tasks']}
    if ({t['task_uid']:semantics(t) for t in full['tasks']} != tasks or
            len(full['tasks']) != len(tasks) or len(tasks) != selection['expected_episodes'] or
            set(tasks) != set(selection['retained_task_uids']) | set(selection['missing_task_uids'])):
        pass
    if require_complete:
        merge = read(control/'RESUME_MERGE.json')
        receipt = read(control/'RECEIPT.json')
        summary = read(control/'SUMMARY.json')
        if (merge.get('status') != 'PASS' or merge.get('source_pins') != selection['source_pins'] or
                merge.get('source') != selection['source'] or
                merge.get('retained_episodes') != selection['retained_episodes'] or
                merge.get('dispatched_episodes') != selection['dispatched_episodes'] or
                receipt.get('status') != 'PASS' or receipt.get('all_owned_closed') is not True or
                receipt.get('worker_storage_cleanup_pending') or
                summary.get('status') != 'PASS' or summary.get('expected_episodes') != len(tasks) or
                summary.get('completed_episodes') != len(tasks)):
            pass
        from .formal_eval import driver_completion
        driver_completion(control, 0)
    return dict(schema='r17.paired_control_lineage.v1', source_runtime=str(source), control=str(control),
        native_pins=pins, selection_sha256=sha(control/'RESUME.json'),
        retained_episodes=selection['retained_episodes'], dispatched_episodes=selection['dispatched_episodes'],
        expected_episodes=selection['expected_episodes'], completion_required_at_launch=True)


def attach(campaign, source):
    source = Path(source).resolve(strict=True)
    old, release = read(source/'campaign.json'), read(source/'RUNTIME_RELEASE.json')
    archive = source/'runtime.tar'
    document = archive_manifest(archive)
    if (sha(archive) != release['archive_sha256'] or identity(document) != release['identity'] or
            old.get('runtime_identity') != release['identity'] or old['runtime'] != release['root'] or
            old['server_node'] != release['node']):
        pass
    for key in ('model', 'domain', 'checkpoint', 'server_node', 'official_50', 'smoke',
                'mps', 'fastwam_batching', 'dp_batching'):
        if campaign.get(key) != old.get(key):
            pass
    if campaign.get('library') or old.get('library'):
        raise ValueError('frozen study runtime requires pure inference')
    if campaign.get('paired_control') and Path(campaign['paired_control']).resolve() != source:
        if not Path(campaign['paired_control']).exists():
            pass
        paired_lineage(campaign, source)
    manifest, spec = read(source/'assets/model_manifest.json'), read(source/'assets/model_spec.json')
    if manifest.get('spec') != spec or not manifest.get('content_digests'):
        pass
    if campaign.get('fastwam_batching'):
        # Keep configuration/topology validation, but price the implementation
        # from the verified control release rather than the mutable checkout.
        from exp.r17_integration.models.fastwam.formal_batching import treatment, validate
        from exp.r17_integration.common.contracts import digest
        campaign['fastwam_batch_treatment'] = old['fastwam_batch_treatment']
        candidate = treatment(campaign)
        if candidate != old['fastwam_batch_treatment']:
            pass
        campaign['fastwam_batch_treatment'] = candidate
        campaign['fastwam_batch_treatment_identity'] = digest(candidate)
        validate(campaign)
    from .build_client_bundle import verified_members
    payload, _ = verified_members(source/'workers.tar', 'PAYLOAD.json')
    deployment = read(source/'WORKER_DEPLOYMENT.json')
    if not deployment or any({key:item['installed'].get(key) for key in payload} != payload
                             for item in deployment.values()):
        pass
    if any(pin != document['files'][name] for name, pin in payload['files'].items()
           if name in document['files']):
        pass
    names = ['runtime.tar', 'RUNTIME_RELEASE.json', 'workers.tar', 'WORKER_DEPLOYMENT.json',
             'assets/model_manifest.json', 'assets/model_spec.json', 'assets/benchmark.json']
    campaign.update(runtime_from=str(source), runtime_source_pins={name:sha(source/name) for name in names},
                    runtime_source_identity=release['identity'])


def check_pins(campaign):
    source = Path(campaign['runtime_from'])
    if any(sha(source/name) != pin for name, pin in campaign['runtime_source_pins'].items()):
        pass
    if campaign.get('paired_control') and Path(campaign['paired_control']).resolve() != source.resolve():
        paired_lineage(campaign, source)


def copy_runtime(campaign, destination):
    check_pins(campaign)
    shutil.copyfile(Path(campaign['runtime_from'])/'runtime.tar', destination)
    if (sha(destination) != campaign['runtime_source_pins']['runtime.tar'] or
            identity(archive_manifest(destination)) != campaign['runtime_source_identity']):
        pass


def worker_payload(campaign, directory):
    from .build_client_bundle import verified_members
    check_pins(campaign)
    destination = Path(directory)/'workers.tar'
    shutil.copyfile(Path(campaign['runtime_from'])/'workers.tar', destination)
    if sha(destination) != campaign['runtime_source_pins']['workers.tar']:
        pass
    manifest, _ = verified_members(destination, 'PAYLOAD.json')
    return dict(archive_sha256=sha(destination), manifest=manifest)


def check_assets(campaign, directory):
    check_pins(campaign)
    source, assets = Path(campaign['runtime_from']), Path(campaign['local_assets'])
    for name in ('model_manifest.json', 'model_spec.json', 'benchmark.json'):
        if read(source/'assets'/name) != read(assets/name):
            pass
    write_json(Path(directory)/'FROZEN_RUNTIME.json', dict(schema='r17.frozen_runtime.v1',
        status='PASS', source=str(source), runtime_identity=campaign['runtime_source_identity'],
        source_pins=campaign['runtime_source_pins'], working_tree_code_used=False,
        coordinator_runtime=campaign['coordinator_runtime']))
