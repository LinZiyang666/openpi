"""Explicit library retry policy; default formal/evaluation policy is unchanged."""
from __future__ import annotations

import json
import time

from exp.r17_integration.debug.attempts import MAX_PRE_DECISION_RETRIES, live_capture_count, pre_decision_error


def capture_counter(campaign):
    def count(uid, attempt):
        if campaign['server_node'] == 'local':
            return live_capture_count(campaign, uid, attempt)
        from .formal_eval import remote, host_python, pythonpath
        code = ('from exp.r17_integration.debug.attempts import live_capture_count; import json; '
                'print(json.dumps(live_capture_count('+repr(campaign)+','+repr(uid)+','+repr(attempt)+')))')
        return json.loads(remote('h100', ['env', 'PYTHONPATH='+pythonpath(campaign['runtime']),
                                        host_python(campaign), '-c', code], seconds=30))
    return count


def library_retry_payload(payload, count):
    """Return stock-compatible result and a durable, concrete retry decision."""
    result, evidence = dict(payload), {}
    if not payload.get('error'):
        return result, evidence
    captured, reason = None, None
    rows = payload.get('per_step_rows') or []
    if payload.get('n_steps') == 0 and not rows and not payload.get('evidence_unknown'):
        try:
            captured = count(payload['task_uid'], payload['attempt'])
        except Exception as exc:
            reason = 'capture absence could not be verified: '+repr(exc)
    allowed = pre_decision_error(payload, captured, reported_rows=rows)
    evidence = dict(capture_decisions=captured, reported_decisions=len(rows),
                    pre_decision_retry_allowed=allowed,
                    retry_reason=reason or ('pre-decision infrastructure failure' if allowed else
                                           'episode has progress/capture or unknown execution evidence'))
    if not allowed:
        result['error'] = 'FatalEpisodeError: library retries require zero steps and zero decisions; '+payload['error']
    return result, evidence


def configure_library_driver(driver, campaign):
    """Keep stock queue/fences, but never implicitly rerun a timed-out episode."""
    if not campaign.get('library'):
        return
    queue = driver._scheduler
    queue._max_ep_retries = MAX_PRE_DECISION_RETRIES
    for uid, attempt in campaign.get('repair_attempts', {}).items():
        if uid not in queue._ep_index or type(attempt) is not int or not 1 <= attempt <= 3:
            raise ValueError('invalid library repair dispatch attempt')
        queue._dispatch_gen[uid] = attempt-1
        queue._ep_attempts[uid] = attempt-1

    def timeout(*, timeout_s, now=None):
        now = time.monotonic() if now is None else now
        with queue._lock:
            stale = [uid for uid, stamp in queue._dispatch_ts.items() if now-stamp > timeout_s]
            for uid in stale:
                driver.handle_result(dict(task_uid=uid, attempt=queue._dispatch_gen[uid], success=False,
                    n_steps=0, per_step_rows=[], evidence_unknown=True,
                    error='FatalEpisodeError: library timeout without a zero-progress worker receipt'))
        return stale
    queue.requeue_timed_out = timeout
