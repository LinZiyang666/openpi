"""Repair a closed library collection without recollecting teacher outcomes."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from exp.r17_integration.debug.attempts import (
    MAX_PRE_DECISION_RETRIES, capture_counts, json_rows, pre_decision_error, validate_prior_attempts,
)
from exp.r17_integration.debug.validate import digest, file_sha, model_binding
from .formal_process import write_json

SEAL_FILES = ('benchmark.json', 'model_spec.json', 'full.yaml', 'load.json', 'model_manifest.json', 'manifest.json')


def capture_sources(campaign, directory):
    from .library_staging import bounded
    directory = Path(directory)
    if campaign.get('repair_merge'):
        sources = [Path(p) for p in campaign['repair_merge']['capture_sources']]
        if len(set(sources)) != len(sources):
            raise ValueError('duplicate repair capture sources')
        return sources
    pattern = ('replica_*/capture/shards/*/binding_*/r17_capture.json'
               if bounded(campaign) and campaign.get('h100_staging') else
               'replica_*/capture/binding_*/r17_capture.json')
    return sorted(p.parent for p in (directory/'servers').glob(pattern))


def closed_receipt(campaign, directory, *, passing):
    receipt = json.loads((Path(directory)/'RECEIPT.json').read_text())
    if (receipt.get('all_owned_closed') is not True or passing and receipt.get('status') != 'PASS' or
            receipt.get('worker_storage_cleanup_pending')):
        raise ValueError('repair/library requires a closed collection receipt')
    if campaign['server_node'] == 'h100':
        transfers = receipt.get('transfers', {})
        if transfers.get('h100_servers', {}).get('verified', {}).get('status') != 'verified':
            raise ValueError('remote teacher captures lack SHA-checked closed sync')
        if campaign.get('h100_staging') and transfers.get('h100_shards', {}).get('status') != 'verified':
            raise ValueError('remote teacher shards lack verified receipt')
    return receipt


def outcomes(directory, tasks, counts, *, inherited=None, incomplete=False):
    """Exactly one outcome; contiguous eligible errors may precede it."""
    directory = Path(directory)
    planned = {t.task_uid if hasattr(t, 'task_uid') else t['task_uid'] for t in tasks}
    journal, attempts, steps = (json_rows(directory/'driver'/name) for name in
                               ('journal.jsonl', 'attempts.jsonl', 'per_step.jsonl'))
    reported, decisions = {}, {}
    for row in steps:
        key = row.get('task_uid'), row.get('attempt')
        reported[key] = reported.get(key, 0)+1
        if row.get('accepted') is True and not row.get('_kind'):
            decisions[key] = decisions.get(key, 0)+1
    accepted = {}
    for row in journal:
        if row.get('accepted') is True:
            if row.get('task_uid') not in planned or row.get('status') not in ('done', 'failed'):
                raise ValueError('library journal differs from planned episodes')
            if row.get('error') is None:
                uid = row['task_uid']
                if uid in accepted or type(row.get('success')) is not bool:
                    raise ValueError('library must accept each non-error teacher outcome exactly once')
                accepted[uid] = row
    campaign_path = directory/'campaign.json'
    campaign = json.loads(campaign_path.read_text()) if campaign_path.exists() else {}
    from .library_salvage import verify_salvage
    verify_salvage(campaign)
    prior = dict(campaign.get('node_failure_prior_attempts', {}), **(inherited or {}))
    histories = {uid: list(prior.get(uid, [])) for uid in planned}
    seen = {(uid, r['attempt']) for uid, rows in histories.items() for r in rows}
    for row in attempts:
        if row.get('accepted') is not True:
            continue
        uid, attempt = row.get('task_uid'), row.get('attempt')
        if uid not in planned or (uid, attempt) in seen or type(attempt) is not int or attempt < 1:
            raise ValueError('duplicate/foreign accepted library attempt')
        seen.add((uid, attempt))
        if row.get('error') is not None:
            captured = counts.get((uid, attempt), 0)
            if not pre_decision_error(row, captured, reported_rows=[None]*reported.get((uid, attempt), 0)):
                raise ValueError('library error after a decision/step or unknown progress: '+str(uid))
            histories[uid].append(dict(row, capture_decisions=0, reported_decisions=0))
        elif uid not in accepted or attempt != accepted[uid].get('attempt'):
            raise ValueError('accepted attempt lacks its matching terminal outcome')
    for row in journal:
        if row.get('accepted') is True and row.get('error') is not None:
            prior = [r for r in histories[row['task_uid']] if r['attempt'] == row.get('attempt')]
            if len(prior) != 1 or prior[0]['error'] not in row['error']:
                raise ValueError('terminal error lacks its zero-progress attempt receipt')
    missing = {}
    for uid in sorted(planned):
        history = sorted(histories[uid], key=lambda r: r['attempt'])
        histories[uid] = history
        attempt = accepted[uid]['attempt'] if uid in accepted else len(history)+1
        ep = dict(task_uid=uid, attempt=attempt, prior_attempts=history)
        validate_prior_attempts(ep, counts)
        if uid not in accepted:
            if any(k[0] == uid and n for k, n in counts.items()) or any(k[0] == uid for k in reported):
                raise ValueError('missing outcome has capture/progress and cannot be retried: '+uid)
            dispatches = [r for r in json_rows(directory/'driver/dispatch.jsonl') if r.get('task_uid') == uid]
            if any(r['attempt'] not in {h['attempt'] for h in history} for r in dispatches):
                raise ValueError('missing worker result cannot attest zero progress: '+uid)
            missing[uid] = attempt
    if missing and not incomplete:
        raise ValueError('library is missing accepted non-error outcomes: '+str(sorted(missing)))
    return accepted, decisions, histories, missing


def seal_pins(campaign):
    assets = Path(campaign['local_assets'])
    return dict(checkpoint=campaign['checkpoint'],
                seal_files={name: file_sha(assets/name) for name in SEAL_FILES})


def inspect_repair(directory, *, model, suite, checkpoint=None, alignment=None):
    """CPU-only, read-only preflight; original campaign stays immutable."""
    from .formal_driver import plan
    from .library_alignment import load_alignment
    directory = Path(directory).resolve()
    campaign = json.loads((directory/'campaign.json').read_text())
    if not campaign.get('library') or campaign['model'] != model or campaign['suite'] != suite or campaign['smoke']:
        raise ValueError('repair must match a full library campaign/model/suite')
    receipt = closed_receipt(campaign, directory, passing=False)
    if receipt.get('status') != 'FAIL':
        raise ValueError('repair requires the closed FAILED collection')
    if checkpoint is not None and checkpoint != campaign['checkpoint']:
        raise ValueError('repair checkpoint differs from original seal')
    audit = load_alignment(campaign['alignment'])
    if alignment is not None and load_alignment(alignment) != audit:
        raise ValueError('repair B alignment differs from original')
    if file_sha(campaign['bpool_archive']) != campaign['bpool_transfer']['archive_sha256']:
        raise ValueError('repair B input archive SHA differs')
    benchmark, model_spec, tasks, _ = plan(campaign)
    recorded = json.loads((directory/'EPISODE_PLAN.json').read_text())
    if recorded['model'] != model_spec or recorded['tasks'] != [asdict(t) for t in tasks]:
        raise ValueError('repair episode plan differs from sealed bundle/alignment')
    sources = capture_sources(campaign, directory)
    if not sources:
        raise ValueError('repair requires verified original closed captures')
    counts, manifests = capture_counts(sources, binding=model_binding(model_spec, benchmark.identity))
    selected, decisions, histories, missing = outcomes(directory, tasks, counts, incomplete=True)
    if any(decisions.get((uid,row['attempt']),0) <= 0 or
           counts.get((uid,row['attempt']),0) != decisions.get((uid,row['attempt']),0)
           for uid,row in selected.items()):
        raise ValueError('original accepted outcome lacks complete decision/capture evidence')
    if not missing:
        raise ValueError('no missing teacher outcomes to repair')
    # Historic workers did not persist an input-intent fence. Their zero-step
    # exception receipt alone can under-report progress. Only the known EGL
    # initialization failure is safely recoverable on those old collections.
    for uid in missing:
        for row in histories[uid]:
            if row.get('disposition') in ('discarded_node_failure','discarded_missing_worker_result'):
                continue  # Explicit sealed quarantine, never a zero-progress claim.
            if ('pre_decision_retry_allowed' not in row and
                    'Offscreen framebuffer is not complete' not in row['error']):
                raise ValueError('historic error lacks pre-decision infrastructure proof: '+uid)
    pin = dict(schema='r17.library_repair.v1', original=str(directory),
        original_campaign_sha256=file_sha(directory/'campaign.json'),
        episode_plan_sha256=file_sha(directory/'EPISODE_PLAN.json'),
        alignment_digest=audit['digest'], bpool_transfer=campaign['bpool_transfer'],
        seal=seal_pins(campaign), captures=manifests,
        driver_files={name:file_sha(directory/'driver'/name) for name in
                      ('journal.jsonl', 'attempts.jsonl', 'per_step.jsonl')},
        kept_outcomes=len(selected), repair_attempts=missing,
        prior_attempts={uid:histories[uid] for uid in missing},
        coordinates=[dict(task_uid=t.task_uid, task=t.extra['suite_task_id'], init=t.orig_init_state_idx,
                          episode_id=t.extra['library_episode_id']) for t in tasks if t.task_uid in missing])
    pin['digest'] = digest(pin)
    return campaign, audit, pin


def verify_repair_seal(campaign):
    """Runs after a fresh CPU seal, before any deployment/inference launch."""
    pin = campaign['repair_pin']
    if seal_pins(campaign) != pin['seal']:
        raise ValueError('repair teacher seal/checkpoint/weight/code/norm differs from original')
    from .library_alignment import load_alignment
    if (load_alignment(campaign['alignment'])['digest'] != pin['alignment_digest'] or
            campaign['bpool_transfer'] != pin['bpool_transfer'] or
            file_sha(campaign['bpool_archive']) != pin['bpool_transfer']['archive_sha256']):
        raise ValueError('repair alignment/B input differs from original')
    verify_original_pin(pin)


def verify_original_pin(pin):
    original = Path(pin['original'])
    pinned = {str(original/'campaign.json'):pin['original_campaign_sha256'],
              str(original/'EPISODE_PLAN.json'):pin['episode_plan_sha256'], **pin['captures']}
    pinned.update({str(original/'driver'/n):sha for n, sha in pin['driver_files'].items()})
    for path, sha in pinned.items():
        if file_sha(path) != sha:
            raise ValueError('original repair evidence changed: '+path)
    old = json.loads((original/'campaign.json').read_text())
    if seal_pins(old) != pin['seal']:
        raise ValueError('original repair seal changed')
    from .library_salvage import verify_salvage
    verify_salvage(old)


def merge_repair(campaign, directory):
    """Publish a derived closed collection using original + new sealed sources."""
    from .formal_driver import plan
    verify_repair_seal(campaign)
    directory = Path(directory)
    pin, original = campaign['repair_pin'], Path(campaign['repair_pin']['original'])
    old = json.loads((original/'campaign.json').read_text())
    old_receipt = closed_receipt(old, original, passing=False)
    new_receipt = closed_receipt(campaign, directory, passing=True)
    benchmark, model, repair_tasks, _ = plan(campaign)
    sources = capture_sources(old, original)+capture_sources(campaign, directory)
    counts, manifests = capture_counts(sources, binding=model_binding(model, benchmark.identity))
    new, _, _, _ = outcomes(directory, repair_tasks, counts, inherited=pin['prior_attempts'])
    if set(new) != set(pin['repair_attempts']):
        raise ValueError('repair outcome cohort differs from missing original episodes')
    merged = directory/'merged'
    merged.mkdir(exist_ok=False)
    (merged/'driver').mkdir()
    merged_campaign = dict(old, run_name=campaign['run_name']+'_merged',
        semantic_campaign=campaign['semantic_campaign'], local_assets=campaign['local_assets'],
        alignment=campaign['alignment'], repair_merge=dict(capture_sources=list(map(str, sources)),
            provenance_digest=pin['digest'], capture_manifests=manifests))
    write_json(merged/'campaign.json', merged_campaign)
    _, _, all_tasks, _ = plan(merged_campaign)
    journals = json_rows(original/'driver/journal.jsonl')+json_rows(directory/'driver/journal.jsonl')
    rows_by_name = {'journal.jsonl':[r for r in journals if r.get('accepted') is True and r.get('error') is None],
                   'attempts.jsonl':json_rows(original/'driver/attempts.jsonl')+json_rows(directory/'driver/attempts.jsonl'),
                   'per_step.jsonl':json_rows(original/'driver/per_step.jsonl')+json_rows(directory/'driver/per_step.jsonl')}
    for name, rows in rows_by_name.items():
        with (merged/'driver'/name).open('x') as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True)+'\n')
    outcomes(merged, all_tasks, counts)
    write_json(merged/'EPISODE_PLAN.json',dict(tasks=[asdict(t) for t in all_tasks], model=model,
        benchmark_protocol=benchmark.identity, expected_episodes=len(all_tasks)))
    write_json(merged/'REPAIR_PROVENANCE.json',dict(pin=pin, captures=manifests,
        original_status=old_receipt['status'], repair_status=new_receipt['status'],
        recollected_uids=sorted(new), original_outcomes_kept=pin['kept_outcomes']))
    transfers = {}
    if old['server_node'] == 'h100':
        shards = list(old_receipt['transfers'].get('h100_shards', {}).get('shards', []))+list(new_receipt['transfers'].get('h100_shards', {}).get('shards', []))
        transfers = dict(h100_servers=dict(verified=dict(status='verified', derived_from_closed_parts=True)),
                         h100_shards=dict(status='verified', shards=shards))
    write_json(merged/'RECEIPT.json',dict(status='PASS', all_owned_closed=True, transfers=transfers,
        derived_collection=True, collection_parts=[str(original),str(directory)]))
    write_json(directory/'REPAIR_MERGE.json',dict(status='PASS', merged=str(merged),
        original_outcomes_kept=pin['kept_outcomes'], repaired_episodes=len(new), provenance_digest=pin['digest']))
    return merged_campaign, merged
