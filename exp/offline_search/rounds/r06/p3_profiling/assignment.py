"""K5's SHA256/private-RNG assignment, extended to every vision anchor.

The task/init key deliberately excludes arm name, attempt, connection and port.
Replicates are independent, not K5's complementary two-landmark assignments.
"""
import hashlib
import math
import random


def seed_for(seed, task, init, replicate, step, domain="treatment"):
    values = (seed, task, init, replicate, step)
    if any(type(v) is not int for v in values) or min(task, init, replicate, step) < 0:
        raise ValueError("integer seed/task/init/replicate/step required; IDs must be nonnegative")
    key = f"r6p3:v1:{domain}:{seed}:{task}:{init}:{replicate}:{step}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def assign(seed, task, init, replicate, step, p):
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("propensity must be finite in [0,1]")
    u = random.Random(seed_for(seed, task, init, replicate, step)).random()
    return dict(seed=seed, replicate=replicate, task_id=task, init=init, step=step,
                propensity=float(p), uniform=u, assigned_call=u < p,
                policy_seed=seed_for(seed, task, init, replicate, step, "policy"))
