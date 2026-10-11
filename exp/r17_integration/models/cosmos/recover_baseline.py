"""CPU-only audit, resume merge, and SHA seal of a closed Cosmos K5 baseline.

Preserves the original FAIL receipt and raw resumed driver. No GPU imports,
process controls, or network calls; native init/closure evidence is read-only
preflight input, bound to the original segment's plan and receipt SHA256s.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from exp.r17_integration.ops import formal_eval as ops, formal_resume as resume
from exp.r17_integration.ops.formal_process import write_json


def audit_bindings(directory, native):
    directory = Path(directory)
    campaign = resume.read(directory/'campaign.json')
    plan = resume.read(directory/'EPISODE_PLAN.json')
    benchmark = resume.read(directory/'assets/benchmark.json')
    if (native.get('campaign') != campaign['run_name'] or
            native.get('episode_plan_sha256') != resume.sha(directory/'EPISODE_PLAN.json') or
            native.get('gpu_used') is not False):
        raise ValueError('native init audit does not bind the original resumed plan')
    if benchmark.get('seed') != 195 or benchmark.get('seed_mode') != 'fixed':
        raise ValueError('recovery requires official Cosmos seed195')
    evidence = resume.accepted_evidence(directory, plan, resume.read(directory/'BUNDLE_ACK.json')['bundle_identity'])
    tasks = {t['task_uid']: t for t in plan['tasks']}
    if set(evidence['accepted']) != set(tasks) or set(native['native_init_hashes']) != set(tasks):
        raise ValueError('recovery requires every planned cell and original init hash')
    cells = {}
    for uid, ending in evidence['accepted'].items():
        task = tasks[uid]
        trace = [r for r in evidence['steps'] if r.get('task_uid') == uid and
                 r.get('accepted') is True and r.get('attempt') == ending['attempt'] and not r.get('_kind')]
        if any(r.get('seed') != 195 or r.get('orig_init_state_idx') != task['orig_init_state_idx'] or
               r.get('task_id') != task['task_id'] or
               r.get('task_data_hash') != native['native_init_hashes'][uid] for r in trace):
            raise ValueError('recovery seed/original init mismatch: '+uid)
        cells[uid] = dict(suite=task['experiment'], task_name=task['extra']['task_name'],
            original_init_idx=task['orig_init_state_idx'], policy_seed=195, environment_seed=0,
            attempt=ending['attempt'], success=ending['success'], decisions=len(trace),
            original_init_hash=native['native_init_hashes'][uid])
    return dict(episodes=len(cells), successes=sum(bool(r['success']) for r in cells.values()),
                valid=True, cells=cells, decisions=sum(r['decisions'] for r in cells.values()))


def cross_segment_retries(source, plan, audit):
    """Attempt ordinals restart in a new driver; retain both physical run IDs."""
    old = {t['task_uid']: t for t in source['plan']['tasks']}
    new = {t['task_uid']: t for t in plan['tasks']}
    semantic = lambda task: {k:v for k,v in task.items() if k not in ('server_host','server_port')}
    recovered = []
    for row in source['evidence']['attempts']:
        uid = row.get('task_uid')
        if row.get('accepted') is not True or row.get('error') is None or uid not in new:
            continue
        if semantic(old[uid]) != semantic(new[uid]):
            pass
        outcome = audit['cells'][uid]
        recovered.append(dict(task_uid=uid, source_campaign=source['campaign']['run_name'],
            failed_attempt=row['attempt'], error=row['error'], failed_n_steps=row.get('n_steps'),
            resolved_attempt=outcome['attempt'], original_init_idx=outcome['original_init_idx'],
            policy_seed=195, environment_seed=0, original_init_hash=outcome['original_init_hash'],
            final_success=outcome['success'], sealed_cell_unchanged=True))
    return recovered


def recover(directory):
    directory = Path(directory).resolve(strict=True)
    seal_path = directory/'RECOVERED_RESULT_SEAL.json'
    if seal_path.exists():
        seal = resume.read(seal_path)
        if any(resume.sha(directory/name) != pin for name, pin in seal['files'].items()):
            pass
        if resume.sha(directory/'runtime.tar') != seal['runtime_sha256']:
            raise ValueError('recovered archived runtime SHA changed')
        resume.check_pins(resume.read(directory/'campaign.json'))
        return resume.read(directory/'RECOVERED_RECEIPT.json')
    campaign = resume.read(directory/'campaign.json')
    original = resume.read(directory/'RECEIPT.json')
    if (campaign.get('model') != 'cosmos' or campaign.get('domain') != 'libero' or
            not campaign.get('official_50') or campaign.get('smoke') or campaign.get('library') or
            campaign.get('denoising_steps') is not None or campaign.get('commit_horizon') is not None or
            not campaign.get('resume_from') or original.get('status') != 'FAIL' or
            original.get('all_owned_closed') is not True or
            'driver incomplete: 2 is not a completed campaign with resolved retry errors' not in original.get('error', '')):
        raise ValueError('not a closed Cosmos K5 resume with the known driver closure error')
    if (directory/'RESUME_MERGE.json').exists() or (directory/'resume_segment').exists():
        raise ValueError('recovery already partially merged; inspect preserved inputs')
    resume.check_pins(campaign)
    source = resume.inspect_source(campaign['resume_from'],
                                  closure_waiver=campaign.get('resume_source_closure_waiver'))
    resume.same_campaign(source['campaign'], campaign)
    for name in resume.SEALED:
        if resume.sha(directory/'assets'/name) != resume.sha(Path(source['source'])/'assets'/name):
            pass
    model = resume.read(directory/'assets/model_spec.json')
    if model['default_sampler']['num_steps'] != 5:
        raise ValueError('recovery requires native K5')
    attestations = resume.read(directory/'SERVER_ATTESTATIONS.json')
    if len(attestations) != campaign['replicas'] or any(a['metadata'].get('r17_model_spec') != model or
            a['metadata'].get('formal_pure_inference') is not True or
            a['metadata'].get('debug_capture') is not False or a['metadata'].get('cache_enabled') is not False
            for a in attestations):
        pass
    for value in resume.read(directory/'driver/census.json').values():
        if value.get('probe'):
            resume.validate_probe(value['probe'], campaign)
    closure = resume.read(directory/'recovery/CLOSURE_AUDIT.json')
    if closure['source_receipt_sha256'] != resume.sha(directory/'RECEIPT.json'):
        pass
    expected = {('local', r['label']) for r in original['owned_local']}
    expected |= {(r['node'], r['label']) for r in original['owned_remote']}
    checked = closure['verified_closed']
    if (len(checked) != len(expected) or {(r['node'], r['label']) for r in checked} != expected or
            any(p.get('alive') is not False for r in checked for p in r['processes'])):
        raise ValueError('recovery has an unverified or live owned process')
    returncode = next(r['exit_code'] for r in original['owned_local'] if r['label'] == 'driver')
    completion = ops.driver_completion(directory, returncode)
    audit = audit_bindings(directory, resume.read(directory/'recovery/NATIVE_INIT_AUDIT.json'))
    audit['cross_segment_retry_errors'] = cross_segment_retries(source, resume.read(directory/'EPISODE_PLAN.json'), audit)
    selection = resume.read(directory/'RESUME.json')
    if audit['episodes'] != selection['dispatched_episodes']:
        raise ValueError('recovery missing-cell count differs from selection')
    original_receipt_sha = resume.sha(directory/'RECEIPT.json')
    write_json(directory/'recovery/EPISODE_AUDIT.json', dict(audit, completion=completion))
    shutil.copyfile(directory/'EPISODE_PLAN.json', directory/'recovery/SEGMENT_EPISODE_PLAN.json')
    proof = resume.merge(campaign, directory, completion)
    segments = []
    for root, config, count, driver in (
            (Path(source['source']), source['campaign'], selection['retained_episodes'], Path(source['source'])/'driver'),
            (directory, campaign, selection['dispatched_episodes'], directory/'resume_segment/driver')):
        segments.append(dict(campaign=config['run_name'], run=str(root), worker_nodes=config['worker_nodes'],
            episodes=count, journal=str(driver/'journal.jsonl'), journal_sha256=resume.sha(driver/'journal.jsonl'),
            driver_receipt=str(driver/'DRIVER_RECEIPT.json'), driver_receipt_sha256=resume.sha(driver/'DRIVER_RECEIPT.json'),
            campaign_sha256=resume.sha(root/'campaign.json'), original_receipt_sha256=resume.sha(root/'RECEIPT.json')))
    proof.update(segments=segments, cpu_recovery=True, gpu_reexecuted=False, physical_episodes_reexecuted=0,
        cross_segment_retry_errors=audit['cross_segment_retry_errors'],
        episode_audit_sha256=resume.sha(directory/'recovery/EPISODE_AUDIT.json'))
    write_json(directory/'RESUME_MERGE.json', proof)
    driver = resume.read(directory/'driver/DRIVER_RECEIPT.json')
    driver['resume'] = proof
    write_json(directory/'driver/DRIVER_RECEIPT.json', driver)
    summary = ops.aggregate(directory, driver_returncode=0)
    if summary['status'] != 'PASS' or summary['completed_episodes'] != 2000:
        raise ValueError('recovered full baseline failed final aggregate audit')
    result = dict(original, status='PASS', cpu_recovery=True, gpu_reexecuted=False,
        physical_episodes_reexecuted=0, original_receipt=str(directory/'RECEIPT.json'),
        original_receipt_sha256=original_receipt_sha, recovered_error=original['error'],
        segment_completion=completion, resume=proof, summary=str(directory/'SUMMARY.json'))
    result.pop('error')
    write_json(directory/'RECOVERED_RECEIPT.json', result)
    files = ['campaign.json', 'RECEIPT.json', 'RECOVERED_RECEIPT.json', 'SUMMARY.json', 'RESUME.json',
        'RESUME_MERGE.json', 'EPISODE_PLAN.json', 'FULL_EPISODE_PLAN.json', 'SERVER_ATTESTATIONS.json',
        'BUNDLE_ACK.json', 'driver/DRIVER_RECEIPT.json', 'driver/journal.jsonl', 'driver/attempts.jsonl',
        'driver/per_step.jsonl', 'driver/census.json', 'recovery/EPISODE_AUDIT.json',
        'recovery/SEGMENT_EPISODE_PLAN.json', 'recovery/CLOSURE_AUDIT.json', 'recovery/NATIVE_INIT_AUDIT.json',
        *['resume_segment/driver/'+n for n in ('DRIVER_RECEIPT.json','journal.jsonl','attempts.jsonl','per_step.jsonl')],
        *['assets/'+n for n in resume.SEALED]]
    write_json(seal_path, dict(schema='r17.cpu_recovered_formal_result.v1', status='PASS',
        sealed_at=datetime.now(timezone.utc).isoformat(), original_source_pins=campaign['resume_source_pins'],
        segments=segments, files={name: resume.sha(directory/name) for name in files},
        runtime_sha256=resume.sha(directory/'runtime.tar')))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    result = recover(args.directory)
    print(json.dumps(dict(status=result['status'], summary=result['summary'],
                                      gpu_reexecuted=False, physical_episodes_reexecuted=0)))


if __name__ == '__main__':
    main()
