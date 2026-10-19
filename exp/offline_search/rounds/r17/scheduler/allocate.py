"""Worker counts for every live batch: progressive filling by dominant share. Pure and deterministic.

A report is the POST /v1/report body (see DESIGN.md) plus `received`, the scheduler's wall time of arrival.
Units: ram bytes, vram MiB, cpu cores.
"""
from __future__ import annotations

import statistics

TTL_S = 60  # a report is live this long after it arrived
STABLE_S = 600  # shared-pool stability a batch needs before it gets local workers
RESOURCES = ('ram', 'vram', 'cpu')
GIB = 2 ** 30
DEFAULT_COST = {'libero': dict(ram=2.2 * GIB, vram=650, cpu=.3),
                'metaworld': dict(ram=1.0 * GIB, vram=300, cpu=.5)}


def live(reports, now):
    return {b: r for b, r in reports.items() if now - r['received'] <= TTL_S}


def capacities(reports):
    """Per node: free room above the floor in the freshest snapshot, plus every batch's own workers there."""
    freshest = {}
    for report in reports.values():
        for node, state in report['node_state'].items():
            if node not in freshest or state['time'] > freshest[node]['time']:
                freshest[node] = state
    result = {}
    for node, state in freshest.items():
        workers = sum(r['node_state'][node]['workers_ram'] or 0 for r in reports.values() if node in r['node_state'])
        result[node] = dict(ram=max((state['ram_available'] or 0) - (state['ram_floor'] or 0), 0) + workers,
                            vram=state['vram_capacity'] or 0, cpu=state['cpu_capacity'] or 0)
    return result


def costs(report):
    """Per-worker cost on each node: measured median there, else the batch's median elsewhere, else the type default."""
    states = report['node_state']
    result = {}
    for node in report['nodes']:
        result[node] = {}
        for resource in RESOURCES:
            key = 'worker_' + resource
            value = states.get(node, {}).get(key)
            if value is None:
                seen = [s[key] for s in states.values() if s.get(key) is not None]
                value = statistics.median(seen) if seen else DEFAULT_COST[report['worker_type']][resource]
            result[node][resource] = value
    return result


def plan(reports, now, previous=None):
    """Targets plus the per-node capacity and per-batch shared dominant share they were computed from.

    `previous` (the last targets, kept only in memory) makes placement sticky for workers not yet
    running: without it, small capacity changes move them between nodes every recompute."""
    previous = previous or {}
    reports = live(reports, now)
    capacity = capacities(reports)
    room = {n: dict(c) for n, c in capacity.items()}
    cost = {b: costs(r) for b, r in reports.items()}
    alloc = {b: dict.fromkeys(r['nodes'], 0) for b, r in reports.items()}
    want = {b: 0 if r['done'] else r['desired'] for b, r in reports.items()}
    shared = {b: sorted(n for n, v in r['nodes'].items() if v['class'] == 'shared') for b, r in reports.items()}

    def fits(b, n):
        return n in room and all(cost[b][n][x] <= room[n][x] for x in RESOURCES)

    def headroom(n):
        return min(room[n][x] / capacity[n][x] if capacity[n][x] else 0. for x in RESOURCES)

    def share(b, nodes):
        pool = {x: sum(capacity[n][x] for n in nodes if n in capacity) for x in RESOURCES}
        return max(sum(alloc[b][n] * cost[b][n][x] for n in nodes) / pool[x] if pool[x] else 0. for x in RESOURCES)

    def fill(groups):
        """One worker at a time to the lowest dominant share (ties by name) until nobody can be served."""
        while True:
            open_nodes = {}
            for b, nodes in groups.items():
                if sum(alloc[b].values()) < want[b]:
                    usable = [n for n in nodes if alloc[b][n] < reports[b]['nodes'][n]['max'] and fits(b, n)]
                    if usable:
                        open_nodes[b] = usable
            if not open_nodes:
                return
            b = min(open_nodes, key=lambda b: (share(b, groups[b]), b))
            keep = [n for n in open_nodes[b] if alloc[b][n] < max(reports[b]['active'].get(n, 0),
                                                                    previous.get(b, {}).get(n, 0))]
            n = max(keep or open_nodes[b], key=headroom)  # nodes are sorted: ties go to the first name
            alloc[b][n] += 1
            for x in RESOURCES:
                room[n][x] -= cost[b][n][x]

    fill(shared)
    shares = {b: share(b, shared[b]) for b in reports}
    for node in sorted({n for r in reports.values() for n, v in r['nodes'].items() if v['class'] == 'local'}):
        fill({b: [node] for b, r in reports.items()
              if r['server_host'] == node and r['nodes'].get(node, {}).get('class') == 'local'
              and r['shared_stable_s'] >= STABLE_S})
    return dict(targets=alloc, capacity=capacity, shares=shares)


def allocate(reports, now):
    return plan(reports, now)['targets']
