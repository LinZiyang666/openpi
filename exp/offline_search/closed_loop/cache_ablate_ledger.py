"""Accepted pure-cache work and camera engagement checks."""
from collections import Counter
import numpy as np
from .cache_ablate_cameras import ledger


def summarize(model, selected, decisions):
    counts = Counter(decisions=0, vision_decisions=0, blind_decisions=0,
                     one_camera_looks=0, two_camera_looks=0, encoded_cameras=0, calls=0)
    total = 0.
    for d in decisions:
        vision = bool(d['vision'])
        c = d['cache_ablate_cost']
        check = ledger(model, selected, vision=vision, hit=d['hit'], work=d.get('camera_work', {}))
        if set(c) != set(check) or any(not np.isclose(c[k], v, rtol=0, atol=1e-12) for k, v in check.items()):
            raise ValueError('inconsistent pure-cache work ledger')
        counts['decisions'] += 1
        counts['vision_decisions'] += vision
        counts['blind_decisions'] += not vision
        counts['one_camera_looks'] += vision and c['camera_encoded_count'] == 1
        counts['two_camera_looks'] += vision and c['camera_encoded_count'] == 2
        counts['encoded_cameras'] += c['camera_encoded_count']
        total += c['compute_share_work']
    return dict(schema='offline_search.r12_cache_ablate_compute_share.v1', **counts, work=total,
                ir_per_five_controls=total/counts['decisions'] if counts['decisions'] else None,
                engagement_verified=bool(counts['vision_decisions']),
                selected=selected, policy_cost=0., completion_cost=0.)
