"""Fixed-recording expected owner cost, including stall cadence and cooldown.

Observations do not react to simulated decisions. This is a calibration model,
not an off-policy SR estimator. The one-anchor cooldown is integrated exactly.

R6-C-v2 with SELECTION §7b: a slow_ambiguous anchor draws the ordinary R /
uniform lottery, and its extra LOOK happens only when that lottery does not
call. A call therefore changes the cadence and the stall tracker's later
observations, so the cadence is no longer one fixed anchor list. replay_cadence
enumerates it exactly as a DAG whose node is (recorded row, the tracker's
observation window, last extra-LOOK control still inside the LOOK cap, cooled
flag). expected_cost propagates exact reach probabilities through that DAG with
the controller's own rule (stall_bridge.call_probability / scheduled_look /
starts_cooldown), so modeled and deployed call/cooldown/LOOK rules are one code.
"""
from __future__ import annotations
import numpy as np
from .stall_bridge import (extra_look, look_window, InactiveStallTracker, STATE, AMBIGUOUS_RULE,
                           call_probability, starts_cooldown, scheduled_look)

MAX_NODES = 2_000_000
STATUS_FIELDS = ('delta_hat', 'e90', 'a10', 'window_span', 'W')


def _compact(status):
    out = dict(state=status['state'])
    for k in STATUS_FIELDS:
        v = status.get(k)
        out[k] = float(v) if v is not None and np.isfinite(v) else None
    return out


def replay_cadence(episode, model=None, tracker_class=InactiveStallTracker, cooldown_scope='stall'):
    """Exact cadence DAG of one recorded episode under C's anchor rule.

    Nodes are in topological order (row index ascending). `call`/`nocall` give
    the next node index, -1 at the episode end, None where that branch is
    impossible for every parameter (cooled anchor cannot call; an uncooled
    slow_confirmed anchor always calls).
    """
    if cooldown_scope not in ('stall', 'all'):
        raise ValueError('invalid cooldown scope')
    rows = episode['decisions']
    b, h = episode['block_controls'], episode['commit_controls']
    if b < 1 or h % b:
        raise ValueError('commit controls must be a multiple of block controls')
    stride = h // b
    task = episode['task']
    probe = tracker_class(model, task)
    window = getattr(probe, '_window', None)
    depth = getattr(window, 'maxlen', None)
    # Q3's tracker status is a pure function of its last W+1 observations (the
    # deque, cleared by an invalid observation). A tracker exposing no bounded
    # window is keyed on its full observation history, which is always exact.
    if depth is None and isinstance(probe, InactiveStallTracker):
        depth = 1
    horizon = (depth - 1) * h if window is not None and depth is not None else None
    cache = {}

    def status_of(obs):
        if obs not in cache:
            tracker = tracker_class(model, task)
            for r in obs:
                tracker.observe(rows[r]['key'], rows[r]['control_index'])
            status = tracker.status()
            if status['state'] not in STATE:
                raise ValueError('unknown stall state')
            cache[obs] = status
        return cache[obs]

    levels = [dict() for _ in rows]

    def child(j2, hist, last_extra, cooled):
        if j2 >= len(rows):
            return -1
        if horizon is not None and last_extra is not None and rows[j2]['control_index'] - last_extra >= horizon:
            last_extra = None  # outside every LOOK cap from here on
        key = (hist, last_extra, bool(cooled))
        levels[j2].setdefault(key, None)
        return (j2, key)

    nodes, ids = [], {}
    if rows:
        levels[0][((), None, False)] = None
    for j, level in enumerate(levels):
        for key in list(level):
            hist, last_extra, cooled = key
            obs = hist + (j,)
            if depth is not None:
                obs = obs[-depth:]
            status = status_of(obs)
            state = status['state']
            control = rows[j]['control_index']
            eligible = extra_look(status, control, last_extra, h)
            if eligible and horizon is not None and look_window(status, h) > horizon:
                raise ValueError('LOOK cap exceeds the tracker window used to merge cadence states')
            nh = obs if depth is None else obs[max(0, len(obs) - (depth - 1)):] if depth > 1 else ()
            call = nocall = None
            if call_probability(state, cooled, 1.) > 0:  # p > 0 is possible for some parameter
                call = child(j + stride, nh, last_extra, starts_cooldown(True, state, cooldown_scope))
            if call_probability(state, cooled, 0.) < 1:  # p < 1 is possible for some parameter
                look = scheduled_look(eligible, False)
                nocall = child(j + 1 if look else j + stride, nh, control if look else last_extra, False)
            ids[(j, key)] = len(nodes)
            nodes.append(dict(j=j, step=rows[j]['step'], control_index=control, Ehat=rows[j]['Ehat'],
                              state=state, cooled=bool(cooled), look_eligible=bool(eligible),
                              call=call, nocall=nocall, status=_compact(status)))
            if len(nodes) > MAX_NODES:
                raise ValueError('cadence DAG too large')
    for n in nodes:
        for k in ('call', 'nocall'):
            if isinstance(n[k], tuple):
                n[k] = ids[n[k]]
    return dict(uid=episode['uid'], task=episode['task'], init=episode['init'], weight=episode['weight'],
                nominal_blocks=len(rows), cooldown_scope=cooldown_scope, ambiguous_rule=AMBIGUOUS_RULE,
                tracker_states=len(cache), nodes=nodes)


def _check_tree(tree, cooldown_scope):
    if tree.get('ambiguous_rule') != AMBIGUOUS_RULE or tree.get('cooldown_scope') != cooldown_scope:
        raise ValueError('cadence DAG was built for another rule/cooldown scope')


def episode_expectation(tree, parameter, placement):
    """Exact expected (anchors, calls, extra LOOKs) of one episode."""
    nodes = tree['nodes']
    reach = [0.]*len(nodes)
    if nodes:
        reach[0] = 1.
    anchors = calls = looks = 0.
    for i, n in enumerate(nodes):
        r = reach[i]
        if r == 0.:
            continue
        nominal = parameter if placement == 'uniform' else min(1., parameter*n['Ehat'])
        q = call_probability(n['state'], n['cooled'], nominal)
        anchors += r
        calls += r*q
        if n['look_eligible']:
            looks += r*(1.-q)
        if q > 0. and n['call'] >= 0:
            reach[n['call']] += r*q
        if q < 1. and n['nocall'] >= 0:
            reach[n['nocall']] += r*(1.-q)
    return anchors, calls, looks


def simulate_path(tree, parameter, placement, coin):
    """One realized path with the deployed rule; coin(node) is the anchor's keyed uniform."""
    nodes = tree['nodes']
    i, anchors, calls, looks, visited = (0 if nodes else -1), 0, 0, 0, []
    while i >= 0:
        n = nodes[i]
        nominal = parameter if placement == 'uniform' else min(1., parameter*n['Ehat'])
        p = call_probability(n['state'], n['cooled'], nominal)
        call = coin(n) < p
        look = scheduled_look(n['look_eligible'], call)
        anchors += 1
        calls += call
        looks += look
        visited.append((n['step'], n['state'], bool(call), bool(look)))
        i = n['call'] if call else n['nocall']
        if i is None:
            raise AssertionError('deployed rule took an impossible DAG branch')
    return dict(anchors=anchors, calls=calls, extra_LOOKs=looks, path=visited)


def expected_cost(episodes, parameter, placement, c1, cooldown_scope='stall'):
    costs, weights, calls, anchors, looks = [], [], [], [], []
    for episode in episodes:
        _check_tree(episode, cooldown_scope)
        na, ncall, nlook = episode_expectation(episode, parameter, placement)
        costs.append((c1*na+(1-c1)*ncall)/episode['nominal_blocks'])
        weights.append(episode['weight']); calls.append(ncall)
        anchors.append(na); looks.append(nlook)
    return dict(IR=float(np.average(costs, weights=weights)),
                calls=float(np.average(calls, weights=weights)),
                anchors=float(np.average(anchors, weights=weights)),
                extra_LOOKs=float(np.average(looks, weights=weights)))


def solve(episodes, rho, placement, c1, cooldown_scope='stall'):
    if placement not in ('uniform', 'R') or not np.isfinite(rho) or rho < 0 or not 0<c1<1:
        raise ValueError('invalid target/placement/cost share')
    values = [n['Ehat'] for e in episodes for n in e['nodes'] if n['Ehat']>0]
    upper = 1. if placement=='uniform' else 1/min(values) if values else 0.
    minimum = expected_cost(episodes, 0., placement, c1,cooldown_scope)
    maximum = expected_cost(episodes, upper, placement, c1,cooldown_scope)
    floor, ceiling = minimum['IR'], maximum['IR']
    # §7b makes a call replace an ambiguous anchor's LOOK, so cost need not be
    # monotone near saturation. The 65-point grid is always evaluated; if it is
    # monotone the original solve is used unchanged, otherwise the ceiling is
    # the grid supremum and the solve takes the smallest-parameter crossing.
    params = np.linspace(0, upper, 65)
    grid = [expected_cost(episodes, p, placement, c1,cooldown_scope)['IR'] for p in params]
    diffs = np.diff(grid)
    monotone = not np.any(diffs<-1e-10)
    top = upper
    if not monotone and max(grid) > ceiling:
        top = float(params[int(np.argmax(grid))]); ceiling = float(max(grid))
    feasible = floor-1e-12 <= rho <= ceiling+1e-12
    result = dict(rho=float(rho), floor=floor, ceiling=ceiling, feasible=bool(feasible),
        parameter=None, predicted_IR=None, placement=placement, modeled_cooldown_free_anchors=1,
        cooldown_scope=cooldown_scope, ambiguous_rule=AMBIGUOUS_RULE, fixed_recording_assumption=True, c1=c1,
        monotone_grid=bool(monotone), max_grid_decrease=float(max(0., -float(diffs.min()))) if len(diffs) else 0.,
        ceiling_parameter=float(top))
    if not feasible:
        result['reason'] = 'below mandatory stall/LOOK cost' if rho<floor else 'above feasible cooldown/score-support ceiling'
        return result
    lo, hi = 0., upper
    if not monotone:
        # First grid bracket with grid[i] < rho <= grid[i+1]; bisection keeps
        # f(lo) < rho <= f(hi) and the cost is continuous in the parameter.
        i = next((i for i in range(len(grid)-1) if grid[i] < rho <= grid[i+1]), None)
        if i is not None:
            lo, hi = float(params[i]), float(params[i+1])
        else:
            lo, hi = 0., top
    for _ in range(80):
        mid = (lo+hi)/2
        if expected_cost(episodes, mid, placement, c1,cooldown_scope)['IR'] < rho:
            lo = mid
        else:
            hi = mid
    parameter = 0. if abs(rho-floor)<=1e-12 else top if abs(rho-ceiling)<=1e-12 else (lo+hi)/2
    predicted = expected_cost(episodes, parameter, placement, c1,cooldown_scope)
    result.update(parameter=float(parameter), predicted_IR=predicted['IR'], modeled=predicted)
    return result
