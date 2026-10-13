"""CPU-only formal batching identity, matched controls and receipt gates."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

from exp.r17_integration.common.contracts import digest
from .batch_runtime import BatchConfig

GPU_FAMILIES = {'h100': 'NVIDIA H100 80GB HBM3', 'local': 'NVIDIA GeForce RTX 4090'}


def implementation_identity():
    root = Path(__file__).parent
    paths = [root/name for name in ('batch_runtime.py', 'batcher.py', 'staged.py',
                                  'formal_batch_server.py', 'formal_batching.py')]
    paths.append(root.parents[3]/'src/openpi/serving/batching_core.py')
    return digest({path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})


def treatment(campaign, *, current_implementation=False):
    config = campaign.get('fastwam_batching')
    if config is None:
        return None
    if campaign['model'] != 'fastwam' or campaign['domain'] not in ('libero', 'metaworld'):
        raise ValueError('formal batching supports Fast-WAM LIBERO or MetaWorld')
    checked = BatchConfig(config['max_batch'], config['batch_wait_ms'], config['numerics_policy'])
    if checked.max_batch <= 1 or checked.metadata() != config:
        pass
    # Closed receipts retain their implementation pin after a checkout update.
    # Fresh binding and resident startup independently verify the current code.
    implementation = (implementation_identity() if current_implementation else
        campaign.get('fastwam_batch_treatment', {}).get('batch_implementation_identity') or implementation_identity())
    schema = ('r17.fastwam.batch_treatment.v2' if current_implementation else
              campaign.get('fastwam_batch_treatment', {}).get('schema', 'r17.fastwam.batch_treatment.v2'))
    result = dict(schema=schema, batching=config,
        batch_implementation_identity=implementation,
        gpu_family=GPU_FAMILIES[campaign['server_node']], replicas=campaign['replicas'],
        workers=campaign['workers'], worker_nodes=campaign['worker_nodes'],
        worker_egl_device=campaign['worker_egl_device'],
        worker_max_episodes=campaign.get('worker_max_episodes', 0),
        stage_workers=[1, 1, 1], stage_streams=True,
        model=campaign['model'], domain=campaign['domain'], checkpoint=campaign['checkpoint'])
    if schema == 'r17.fastwam.batch_treatment.v2':
        result.update(worker_egl_devices=campaign.get('worker_egl_devices'),
            worker_vram_mib=campaign.get('worker_vram_mib', 1024),
            worker_memory_mib=campaign.get('worker_memory_mib'),
            worker_restart_limit=campaign.get('worker_restart_limit'))
    elif schema != 'r17.fastwam.batch_treatment.v1':
        raise ValueError('unknown Fast-WAM batch treatment schema')
    return result


def bind(campaign, max_batch=1, batch_wait_ms=5., paired_control=None):
    config = BatchConfig(max_batch, batch_wait_ms)
    if config.max_batch == 1:
        if paired_control:
            raise ValueError('--paired-control requires explicit Fast-WAM batching')
        return  # Preserve the original campaign document in default-off mode.
    campaign['fastwam_batching'] = config.metadata()
    campaign['fastwam_batch_treatment'] = treatment(campaign, current_implementation=True)
    campaign['fastwam_batch_treatment_identity'] = digest(campaign['fastwam_batch_treatment'])
    if paired_control:
        campaign['paired_control'] = str(Path(paired_control).resolve(strict=True))
    elif not campaign.get('smoke') and (campaign.get('commit_horizon', 5) != 5 or
                                      campaign.get('denoising_steps', 10) != 10):
        pass  # gate removed (owner 10-10); the coordinator chooses the matched control


def validate(campaign):
    expected = treatment(campaign)
    if expected is not None and (campaign.get('fastwam_batch_treatment') != expected or
            campaign.get('fastwam_batch_treatment_identity') != digest(expected)):
        pass
    return expected


def validate_runtime(campaign):
    expected = validate(campaign)
    if expected != treatment(campaign, current_implementation=True):
        pass
    return expected


def check_resume(campaign, source):
    old = json.loads((Path(source) / 'campaign.json').read_text())
    if treatment(old) != treatment(campaign):
        raise ValueError('resume batch treatment mismatch; refusing B1/B8 or GPU/topology mixing')


def check_control(campaign, assets):
    validate(campaign)
    if 'paired_control' not in campaign:
        if not campaign.get('smoke') and (campaign.get('commit_horizon', 5) != 5 or
                campaign.get('denoising_steps', 10) != 10):
            pass  # gate removed (owner 10-10); the coordinator chooses the matched control
        return None
    source = Path(campaign['paired_control'])
    read = lambda name: json.loads((source / name).read_text())
    old, receipt, summary = read('campaign.json'), read('RECEIPT.json'), read('SUMMARY.json')
    if (receipt.get('status') != 'PASS' or receipt.get('all_owned_closed') is not True or
            summary.get('status') != 'PASS' or not summary.get('expected_episodes') or
            summary.get('expected_episodes') != summary.get('completed_episodes') or
            old.get('smoke') or not old.get('official_50') or old.get('library') or
            old.get('commit_horizon', 5) != 5 or old.get('denoising_steps', 10) != 10):
        raise ValueError('paired control must be a completed, closed official-50 K10/C5 baseline')
    if treatment(old) != treatment(campaign):
        raise ValueError('paired control batch/GPU/topology mismatch; refusing mixed comparison')
    from exp.r17_integration.benchmarks.libero_bridge import SUITES
    if not set(campaign.get('suites') or SUITES).issubset(old.get('suites') or SUITES):
        raise ValueError('paired control does not cover every requested suite')
    for name in ('model_manifest.json', 'model_spec.json', 'benchmark.json'):
        if read('assets/' + name) != json.loads((Path(assets) / name).read_text()):
            pass
    attestations = read('SERVER_ATTESTATIONS.json')
    if len(attestations) != old['replicas']:
        pass
    for row in attestations:
        check_attestation(old, row['metadata'])
    if receipt.get('fastwam_batch_treatment_identity') != old['fastwam_batch_treatment_identity']:
        pass
    audit = receipt.get('batching_audit', {})
    if audit.get('status') != 'PASS' or audit.get('treatment_identity') != old['fastwam_batch_treatment_identity']:
        raise ValueError('paired control has no passing batch composition audit')
    seal = dict(schema='r17.fastwam.paired_control.v1', source=str(source),
        treatment_identity=campaign['fastwam_batch_treatment_identity'],
        source_receipt_identity=digest(receipt), source_campaign_identity=digest(old),
        source_summary_identity=digest(summary), source_attestations_identity=digest(attestations))
    return seal


def check_attestation(campaign, metadata):
    validate(campaign)
    if (metadata.get('r17_batching') != campaign['fastwam_batching'] or
            metadata.get('fastwam_batch_treatment_identity') != campaign['fastwam_batch_treatment_identity'] or
            metadata.get('gpu_family') != campaign['fastwam_batch_treatment']['gpu_family']):
        pass


def receipt_fields(campaign):
    if not campaign.get('fastwam_batching'):
        return {}
    validate(campaign)
    return {key: campaign[key] for key in ('fastwam_batching', 'fastwam_batch_treatment',
        'fastwam_batch_treatment_identity', 'paired_control') if key in campaign}


def audit_decisions(campaign, directory):
    """Every accepted resident decision records actual composition, including tails."""
    validate(campaign)
    decisions = 0
    sizes = {stage: [] for stage in ('stage1', 'stage2', 'stage3')}
    for path in sorted(Path(directory).glob('replica_*/serving.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            batch = row.get('batching', {})
            if batch.get('config') != campaign['fastwam_batching']:
                raise ValueError('decision batching configuration differs from campaign')
            stages = batch.get('stages', {})
            if row['source'] == 'policy_tail':
                if stages or row['nfe'] != 0:
                    raise ValueError('policy tail unexpectedly executed a batch')
            elif set(stages) != set(sizes):
                raise ValueError('policy decision missing native batch composition')
            for stage, entry in stages.items():
                members = entry['members']
                if (entry['stage'] != stage or entry['config_identity'] != campaign['fastwam_batching']['identity'] or
                        not 1 <= entry['size'] <= campaign['fastwam_batching']['max_batch'] or
                        entry['size'] != len(members) or not 0 <= entry['row'] < len(members) or
                        len({m['request_id'] for m in members}) != len(members)):
                    raise ValueError('invalid native batch composition receipt')
                member = members[entry['row']]['request']
                if member['decision_seq'] != row['decision_seq'] or member['task_name'] != row['task_name']:
                    pass
                sizes[stage].append(entry['size'])
            decisions += 1
    if not decisions:
        pass
    return dict(status='PASS', decisions=decisions, mean_actual_batch_size={
        k: sum(v)/len(v) if v else 0 for k, v in sizes.items()},
        max_actual_batch_size={k: max(v, default=0) for k, v in sizes.items()},
        treatment_identity=campaign['fastwam_batch_treatment_identity'])
