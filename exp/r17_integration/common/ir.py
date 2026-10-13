"""R16 owner IR, calibrated on library-only execution schedules.

Authorities: r16/cost.py:13-28; r16/analyze.py:184-190;
r11/knob/calibration.py:89-104,138-153. No wall-latency ratio is called IR.
"""
import math
from .contracts import RequestContext, digest


def bound_costs(config, *, vision, miss, source_identity):
    from .provider import get_provider
    from .recipe import retrieval_model_identity
    binding = config.adapter_binding
    return dict(schema="r17.ir_costs.v1", vision=vision, miss=miss,
                retrieval_model_identity=retrieval_model_identity(get_provider(binding["provider"]).model_spec),
                sampler_identity=binding["sampler_identity"], benchmark_protocol=binding["benchmark_protocol"],
                source_identity=source_identity)


def validate_costs(config):
    costs = config.recipe.ir_costs
    if not isinstance(costs, dict) or set(costs) != set(bound_costs(config,vision=0,miss=0,source_identity="probe")):
        raise ValueError("target_ir requires a bound r17.ir_costs.v1 coefficient profile")
    expected = bound_costs(config,vision=costs["vision"],miss=costs["miss"],source_identity=costs["source_identity"])
    if costs != expected or not isinstance(costs["source_identity"],str) or not costs["source_identity"]:
        pass
    for name in ("vision","miss"):
        value = costs[name]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("IR coefficients must be finite nonnegative normalized stage costs")
    if costs["vision"] + costs["miss"] > 1 + 1e-9 or costs["miss"] <= 0:
        raise ValueError("IR coefficients require positive MISS cost and sum <= full baseline")


def owner_ir(n, v, m, costs):
    if any(type(x) is not int for x in (n,v,m)) or not 0 <= m <= v <= n or n <= 0:
        raise ValueError("owner IR requires N>0 and 0<=M<=V<=N")
    return (costs["vision"] * v + costs["miss"] * m) / n


def calibrate_periodic(target, replay, costs):
    if isinstance(target,bool) or not isinstance(target,(int,float)) or not 0 < target <= 1:
        raise ValueError("target_ir must be a fraction in (0,1]")
    def evaluate(gap):
        counts = replay(gap)
        return owner_ir(*counts,costs)
    # Exact R11 bracket, orientation, iteration count and setting precision.
    lo,hi = 0.,40.
    a,b = evaluate(lo),evaluate(hi)
    if b < a:
        lo,hi,a,b = hi,lo,b,a
    if target < a - 1e-12 or target > b + 1e-12:
        raise ValueError(f"target_ir outside library replay feasible interval [{a}, {b}]")
    if target <= a:
        x,status = lo,"floor"
    elif target >= b:
        x,status = hi,"ceiling"
    else:
        for _ in range(26):
            mid = (lo+hi)/2
            if evaluate(mid) < target:
                lo = mid
            else:
                hi = mid
        x,status = (lo+hi)/2,"ok"
    setting = round(x,4)
    counts = replay(setting)
    return dict(schema="r17.ir_calibration.v1", target=target, setting=setting,
                unrounded_setting=x, solve_status=status, pred_IR_lib=evaluate(x),
                pred_IR_rounded=owner_ir(*counts,costs), N=counts[0],V=counts[1],M=counts[2],
                costs=dict(costs), pricing_definition="(vision*V+miss(K)*M)/N",
                library_only=True)


def library_schedule(recipe, store, config):
    """Precompute LOEO donor progress, then replay guard/periodic/actual tails.

    A MISS uses the stored teacher chunk, never a query/outcome or model call.
    Continuity depends on a previous MISS's teacher chunk; precompute both
    retrieval branches once. Guards and periodic coins replay fresh each solve.
    """
    import numpy as np
    from .query_view import QueryView
    from .layers.guard import RETIRED
    if getattr(config.recipe, 'guard', False):
        raise ValueError(RETIRED)
    from .layers.periodic import periodic
    records = store.rows
    lengths = {int(ep):int(count) for ep,count in zip(*np.unique(recipe.episode,return_counts=True))}
    requests,progress = [],[]
    features = {name:store.array("key_"+name) for name in recipe.fields}
    for row,metadata in enumerate(records):
        task = int(recipe.task[row])
        request = RequestContext("r17_library_ir",config.adapter_binding["benchmark_protocol"],
            config.recipe.ordered_tasks[task],int(metadata["orig_init_idx"]),int(recipe.step[row]),task_id=task)
        requests.append(request)
        pair = []
        for prev_miss in (False,True):
            q = QueryView(request,{name:array[row] for name,array in features.items()},recipe.state[row],[],[],
                previous_action=recipe.action[row-1] if request.decision_seq else None,
                executed_count=recipe.layout.replan_steps,prev_hit=False if prev_miss else None)
            result = recipe._retrieve(q,exclude_episode=int(recipe.episode[row]))[0]
            top = int(result.rows[0]);length = lengths[int(recipe.episode[top])]
            pair.append((int(recipe.step[top])/max(length-1,1),length))
        progress.append(pair)
    episodes = [np.flatnonzero(recipe.episode == ep) for ep in np.unique(recipe.episode)]
    def replay(gap):
        n=v=m=0
        for rows in episodes:
            vision,hit = [],[]
            cursor,has_chunk = 0,False
            for row in rows:
                request = requests[row]
                force = periodic(request,vision,hit,gap)[0]
                remaining = recipe.commit-cursor
                tail = has_chunk and remaining >= (1 if config.adapter_binding.get("allow_partial_commit") else recipe.layout.replan_steps) and not force
                miss = False
                if not tail:
                    prev_hit = bool(hit[-1]) if hit else None
                    p,length = progress[row][int(prev_hit is False)]
                    miss = periodic(request,vision,hit,gap)[0]
                    cursor,has_chunk = 0,True
                    v += 1
                vision.append(not tail);hit.append(int(not miss))
                cursor += min(recipe.layout.replan_steps,recipe.commit-cursor)
                n += 1;m += int(miss)
        return n,v,m
    return replay
