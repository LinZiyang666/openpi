"""R11 fractional-gap hash and accepted look history, independent of workers."""
import hashlib
import json
import math


def uniform(key, seed, task, init, step, domain):
    if init < 0 or task < 0:
        raise ValueError("periodic coins require original nonnegative task/init")
    payload = ["Q2-deploy-v1", key, int(seed), int(task), int(init), int(step), domain]
    b = hashlib.sha256(json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode()).digest()[:8]
    return (int.from_bytes(b, "big") >> 11) * 2.**-53


def periodic(request, has_vision, hit, gap, *, guard=False):
    if len(has_vision) != request.decision_seq or len(hit) != request.decision_seq or any(h not in (0, 1) for h in hit):
        raise ValueError("periodic needs accepted histories for all preceding decisions")
    if gap is None:
        return False, {"knob": False}
    if not math.isfinite(gap) or gap < 0:
        raise ValueError("periodic gap must be finite and nonnegative")
    anchors = [i for i, yes in enumerate(has_vision) if yes]
    calls = [i for i in anchors if hit[i] == 0]
    last = calls[-1] if calls else -1
    run = sum(i > last for i in anchors)
    u = uniform("R11-gap-v1", 0, request.task_id, request.original_init_idx, last, "cap")
    floor = math.floor(gap)
    cap = floor + int(gap - floor > 0 and u < gap - floor)
    force = not guard and run >= cap
    return force, {"knob": force, "coin": u, "cap": cap, "run": run,
                   "N": request.decision_seq, "V": len(anchors), "M": len(calls)}
