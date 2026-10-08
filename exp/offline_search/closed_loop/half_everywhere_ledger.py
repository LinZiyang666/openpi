"""Per-mode accepted-decision ledger and mandatory acquisition engagement checks."""
from collections import Counter
from .lookhalf_ledger import summarize as base_summary


def validate_receipt(mode, *, vision, miss, work, cost):
    if mode not in ('wrist', 'both'):
        raise ValueError('missing task camera mode')
    if not vision:
        return
    wanted = [True, True] if mode == 'both' or miss else [False, True]
    if ('initial_cameras' not in work or 'availability' not in work
            or work['initial_cameras'] != cost['camera_initial_count']
            or work.get('completions', 0) != cost['camera_completion_count']
            or work['availability'] != wanted
            or cost['camera_encoded_count'] != sum(wanted)):
        raise ValueError('HALF_CAMERA_RECEIPT_REQUIRED: acquisition engagement mismatch')
    if mode == 'both' and cost['camera_initial_count'] != 2:
        raise ValueError('both task encoded one camera')


def summarize(model, decisions):
    out = base_summary(model, decisions)
    counts = Counter({k: 0 for k in ('wrist_task_decisions', 'both_task_decisions',
        'wrist_task_looks', 'both_task_looks', 'wrist_task_two_camera_looks')})
    by_mode = {}
    for d in decisions:
        mode = d.get('task_mode')
        vision, miss = bool(d.get('vision', True)), not bool(d['hit'])
        validate_receipt(mode, vision=vision, miss=miss, work=d.get('camera_work', {}), cost=d['lookhalf_cost'])
        counts[f'{mode}_task_decisions'] += 1
        counts[f'{mode}_task_looks'] += vision
        counts['wrist_task_two_camera_looks'] += mode == 'wrist' and vision and d['lookhalf_cost']['camera_encoded_count'] == 2
        row = by_mode.setdefault(mode, dict(decisions=0, looks=0, calls=0, work=0., per_episode={}))
        row['decisions'] += 1; row['looks'] += vision; row['calls'] += miss
        row['work'] += d['lookhalf_cost']['compute_share_work']
        ep = row['per_episode'].setdefault(d.get('uid', ''), dict(calls=0, first_call=None))
        ep['calls'] += miss
        if miss:
            ep['first_call'] = d['step'] if ep['first_call'] is None else min(ep['first_call'], d['step'])
    return dict(out, **counts, per_mode=by_mode)
