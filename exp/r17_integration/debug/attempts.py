"""Library-only attempt evidence; teacher outcomes are never retry candidates."""
from __future__ import annotations

import json
import os
from pathlib import Path

MAX_PRE_DECISION_RETRIES = 2


def json_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        raise ValueError('torn attempt/capture evidence: ' + str(path))
    return [json.loads(line) for line in raw.splitlines() if line]


def decision_intent(directory, episode):
    """Durable before model input is consumed, including a failed first forward."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    row = dict(task_uid=episode['task_uid'], attempt=episode['attempt'])
    with (directory/'decision_intents.jsonl').open('ab') as stream:
        stream.write(json.dumps(row, sort_keys=True).encode()+b'\n')
        stream.flush()
        os.fsync(stream.fileno())


def pre_decision_error(row, captured_decisions, *, reported_rows=()):
    # Timeouts without a worker result do not attest how much work ran. Stock
    # workers also lose partial progress on exceptions; the synchronous model
    # input intent fence covers those cases for new library campaigns.
    return (bool(row.get('error')) and row.get('success') is False and
            type(row.get('n_steps')) is int and row['n_steps'] == 0 and
            type(captured_decisions) is int and captured_decisions == 0 and
            not reported_rows and row.get('evidence_unknown') is not True and
            row.get('pre_decision_retry_allowed') is not False and
            ('capture_decisions' not in row or row['capture_decisions'] == 0))


def validate_prior_attempts(episode, counts):
    attempt = episode.get('attempt')
    history = episode.get('prior_attempts', [])
    if (type(attempt) is not int or not 1 <= attempt <= MAX_PRE_DECISION_RETRIES+1 or
            len(history) != attempt-1 or
            [r.get('attempt') for r in history] != list(range(1, attempt))):
        raise ValueError('teacher retry requires every bounded prior attempt')
    for row in history:
        key = episode['task_uid'], row['attempt']
        if row.get('disposition') in ('discarded_node_failure','discarded_missing_worker_result'):
            from exp.r17_integration.ops.library_salvage import verify_discard
            if row.get('task_uid') != episode['task_uid']:
                raise ValueError('foreign discarded node-failure attempt')
            verify_discard(row, counts)
            continue
        if (type(row['attempt']) is not int or row.get('task_uid') != episode['task_uid'] or row.get('accepted') is not True or
                type(row.get('capture_decisions')) is not int or row['capture_decisions'] != 0 or
                type(row.get('reported_decisions')) is not int or row['reported_decisions'] != 0 or
                not pre_decision_error(row, counts.get(key, 0))):
            raise ValueError('teacher prior attempt is not a pre-decision error without capture rows')


def capture_counts(sources, *, binding=None, closed=True):
    """Read only sealed metadata (verify payload hashes, never load RGB arrays)."""
    from .reader_profiles import CaptureReader
    counts, manifests = {}, {}
    for source in map(Path, sources):
        if closed:
            reader = CaptureReader(source, expected_binding=binding)
            m = reader.manifest
            stats = m.get('writer_stats', {})
            if not stats.get('drained') or stats.get('errors'):
                raise ValueError('teacher capture writer was not drained cleanly')
            actual = {str(p.relative_to(source)) for p in source.rglob('*') if p.is_file() and p.name != 'r17_capture.json'}
            if actual != set(m['files']):
                raise ValueError('teacher capture contains unsealed files')
            from .validate import file_sha
            manifests[str(source/'r17_capture.json')] = file_sha(source/'r17_capture.json')
        decisions = []
        for path in sorted(source.glob('decisions_*.jsonl')):
            decisions.extend(json_rows(path))
        for row in decisions:
            key = row['task_uid'], row['attempt']
            counts[key] = counts.get(key, 0)+1
        intents = {}
        for row in json_rows(source/'decision_intents.jsonl'):
            key = row['task_uid'], row['attempt']
            intents[key] = intents.get(key, 0)+1
        for key, value in intents.items():
            counts[key] = max(counts.get(key, 0), value)
        if closed and (stats.get('accepted') != len(decisions) or stats.get('written') != len(decisions)):
            raise ValueError('teacher capture rows differ from writer receipt')
    return counts, manifests


def live_capture_count(campaign, uid, attempt):
    """No absence attestation if a replica directory/evidence is unavailable."""
    root = Path(campaign['server_run'])
    total = 0
    for index in range(campaign['replicas']):
        replica = root/('replica_'+str(index))
        if not replica.is_dir():
            raise ValueError('replica capture evidence unavailable: '+str(replica))
        # Intents survive a streamed shard's removal; they live at capture root
        # on H100 and in the binding directory for ordinary local captures.
        for path in replica.glob('capture/**/decision_intents.jsonl'):
            total += sum(r.get('task_uid') == uid and r.get('attempt') == attempt for r in json_rows(path))
        if campaign.get('library_intent_fence') != 'r17.library_input_intent.v1':
            # Historic diagnostic fallback. New H100 shards can disappear once
            # verified; their persistent root intent ledger never disappears.
            for path in replica.glob('capture/**/decisions_*.jsonl'):
                total += sum(r.get('task_uid') == uid and r.get('attempt') == attempt for r in json_rows(path))
    return total
