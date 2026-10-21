"""Worker counts for every live batch: progressive filling by dominant share. Pure and deterministic.

A report is the POST /v1/report body (see DESIGN.md) plus `received`, the scheduler's wall time of arrival.
Units: ram bytes, vram MiB, cpu cores.
"""
from __future__ import annotations

import statistics

TTL_S = 60  # a report is live this long after it arrived
STABLE_S = 600  # shared-pool stability a batch needs before it gets local workers
LOCAL_FIRST = ('local', 'h100')  # server machines whose own workers are filled before the shared pool (owner 10-20)
RESOURCES = ('ram', 'vram', 'cpu')
GIB = 2 ** 30
DEFAULT_COST = {'libero': dict(ram=2.2 * GIB, vram=650, cpu=.3),
                'metaworld': dict(ram=1.0 * GIB, vram=300, cpu=.5)}


SATURATED_UTIL = 90  # server GPU utilization (%) at which more workers stop paying off
SATURATED_CPU = 85  # replica CPU in % of its thread budget (OMP_NUM_THREADS) at which it is CPU-bound


def server_util(report):
    """GPU utilization of the batch's server machine (its own-machine snapshot), or None (MIG pods, no snapshot)."""
    return (report['node_state'].get(report['server_host']) or {}).get('gpu_util')


def saturated(state):
    """A server machine's snapshot shows no room: its GPU is busy, or at least half its replicas are CPU-bound."""
    cpu = (state or {}).get('server_cpu') or []
    return ((state or {}).get('gpu_util') or 0) >= SATURATED_UTIL or \
        bool(cpu) and sum(c >= SATURATED_CPU for c in cpu) >= len(cpu) / 2


def explore(report):
    """Workers a batch may hold in the work-conserving pass: double while its server has room, hold when it is
    saturated (GPU or replica CPU), and without a utilization reading (MIG pods) twice the client's own desire."""
    active, desired = sum(report['active'].values()), report['desired']
    if server_util(report) is None:
        return 2 * desired
    if saturated(report['node_state'].get(report['server_host'])):
        return max(desired, active)
    return 2 * max(desired, active)


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

    local_nodes = sorted({n for r in reports.values() for n, v in r['nodes'].items() if v['class'] == 'local'})

    def own(node):
        return {b: [node] for b, r in reports.items()
                if r['server_host'] == node and r['nodes'].get(node, {}).get('class') == 'local'}
    # The RTX 4090's and the H100's own workers come first, without the stability wait (owner 10-20: the H100
    # like the 4090). Remote workers reach the 4090's server through its ~33 MB/s public ingress, which one MT50
    # batch's images saturate (10-20: 100 remote workers, GPU 8%). Waiting for a stable shared pool left the
    # H100's own workers at 0 almost always (one 7-worker target in 6,470 allocation rows, 10-20).
    def rounds():
        for node in [n for n in local_nodes if n in LOCAL_FIRST]:
            fill(own(node))
        fill(shared)
        for node in [n for n in local_nodes if n not in LOCAL_FIRST]:
            fill({b: nodes for b, nodes in own(node).items() if reports[b]['shared_stable_s'] >= STABLE_S})
    rounds()
    # Work-conserving second pass (owner 10-20: no idle workers while a server can take more). Each batch's
    # desire is only its first claim: what is left goes, by the same dominant share, to batches whose server GPU
    # has room (explore: doubling until it saturates). A client's step test that misreads (its window closes
    # before new workers finish an episode) no longer leaves worker machines idle.
    for b, r in reports.items():
        if not r['done']:
            limit = min(sum(v['max'] for v in r['nodes'].values()), explore(r))
            want[b] = max(want[b], limit if r.get('cap') is None else min(limit, r['cap']))  # an operator cap holds here too
    rounds()
    shares = {b: share(b, shared[b]) for b in reports}
    return dict(targets=alloc, capacity=capacity, shares=shares)


def allocate(reports, now):
    return plan(reports, now)['targets']
