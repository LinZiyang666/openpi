"""Library-stage arithmetic; segmentation/calibration belong to C1's StageTable.

Event weights are the kernel mean of the member weights (unknown members have
weight one). The independent deviation-entry factor multiplies that event
factor. A zero library occupancy disables that factor, rather than dividing by
zero. The entry latch advances ONLY on fresh anchors, including extra LOOKs.
"""
from __future__ import annotations

import numpy as np


def factors(event_mass, h, high, h_dev, latched):
    if not all(np.isfinite(x) and 0 <= x <= 1 for x in (event_mass, h, h_dev)):
        raise ValueError('stage masses/occupancies must be finite probabilities')
    event = 1. + event_mass * (1. / h - 1.) if h > 0 else 1.
    entry = high is True and not latched
    deviation = 1. / h_dev if entry and h_dev > 0 else 1.
    # An invalid observation supplies no evidence of return to normal.
    next_latch = latched if high is None else bool(high)
    return event * deviation, next_latch, bool(entry)


def stage_features(table, rows, weights, state, task):
    """Adapter to C1's frozen table; no fitted statistics are computed here."""
    info = table.online(rows, weights)
    value = table.deviation(state, rows, weights)
    h = float(table.event_occupancy.get(int(task), 0.))
    h_dev = float(table.deviation_occupancy.get(int(task), 0.))
    radius = table.deviation_p75.get(int(task))
    valid = value is not None and radius is not None and np.isfinite(value) and np.isfinite(radius)
    return dict(event_mass=float(info['event_mass']), h=h, h_dev=h_dev,
                high=bool(value > radius) if valid else None,
                deviation=float(value) if valid else None, p75=float(radius) if valid else None,
                unanimous=bool(info['unanimous']), unknown_mass=float(info['unknown_mass']))


def lift_deviation_latch(tree, features, *, enabled=True):
    """Product of R6's cadence DAG and the fresh-anchor high-deviation latch.

Every copied edge preserves the R6 stall history, cooldown and LOOK cap. Ehat
is used as the positive score slot of R6's exact capped-weight solver; the R6
disagreement score is retained separately. No path is weighted by success.
"""
    if not enabled:
        return tree
    levels = [set() for _ in tree['nodes']]
    if levels:
        levels[0].add(False)
    nodes, ids = [], {}
    for i, original in enumerate(tree['nodes']):
        feature = features[original['j']]
        for latched in sorted(levels[i]):
            weight, next_latch, entry = factors(feature['event_mass'], feature['h'],
                                               feature['high'], feature['h_dev'], latched)
            node = {**original, 'R6_Ehat': original['Ehat'], 'Ehat': weight,
                    'call_weight': weight, 'deviation_latched': latched,
                    'deviation_entry': entry, 'stage': feature}
            for edge in ('call', 'nocall'):
                child = original[edge]
                if child is not None and child >= 0:
                    if child <= i:
                        raise ValueError('cadence nodes must be topologically ordered')
                    levels[child].add(next_latch)
                    node[edge] = (child, next_latch)
            ids[(i, latched)] = len(nodes)
            nodes.append(node)
    for node in nodes:
        for edge in ('call', 'nocall'):
            if isinstance(node[edge], tuple):
                node[edge] = ids[node[edge]]
    return {**tree, 'nodes': nodes, 'stage_tilt': 'event_kernel_mean_times_deviation_entry',
            'r6_nodes': len(tree['nodes'])}
