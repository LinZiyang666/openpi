"""Sequential experimental designs. Pure, independently keyed assignment streams.

Propensities are conditional on the recorded pre-action history. Deterministic
holds, delayed obligations, caps and cooldowns have no local treatment support.
"""
import copy
import random

from .assignment import assign, seed_for


def validate(design):
    d = dict(pre_guard=False, durations=[10], holds=[1], delays=[0], cap=None,
             cooldown=0, start_anchor=0, cohort="fixed", cohort_probability=1.,
             split_modulus=5, allocation="fixed matched task/init blocks", episode_doses=None)
    d.update(design or {})
    if set(d) - {"pre_guard", "durations", "holds", "delays", "cap", "cooldown", "start_anchor",
                 "cohort", "cohort_probability", "split_modulus", "allocation", "episode_doses"}:
        raise ValueError("unknown design dimension")
    for key, allowed in (("durations", {5, 10}), ("holds", set(range(1, 17))), ("delays", set(range(17)))):
        values = d[key]
        if not values or any(type(x) is not int or x not in allowed for x in values) or len(set(values)) != len(values):
            raise ValueError(f"invalid {key}")
    for key in ("cap", "cooldown", "start_anchor"):
        if d[key] is not None and (type(d[key]) is not int or d[key] < 0):
            raise ValueError(f"invalid {key}")
    if type(d["pre_guard"]) is not bool or not 0 < d["cohort_probability"] <= 1 or d["split_modulus"] < 2:
        raise ValueError("invalid design probability/split")
    if d["delays"] != [0] and (d["holds"] != [1] or d["pre_guard"]):
        raise ValueError("delayed allowance design requires holds=[1], A baseline")
    if d["episode_doses"] is not None:
        if not d["episode_doses"] or any(not 0 <= x <= 1 for x in d["episode_doses"]):
            raise ValueError("episode_doses must be nonempty probabilities")
    return d


def uniform(seed, task, init, replicate, step, domain):
    return random.Random(seed_for(seed, task, init, replicate, step, domain)).random()


def choose(options, assignment, domain):
    u = uniform(*(assignment[k] for k in ("seed", "task_id", "init", "replicate", "step")), domain)
    return options[min(int(u * len(options)), len(options) - 1)], 1. / len(options)


class Design:
    def __init__(self, config):
        self.config = validate(config)
        self.index = self.spent = self.hold_left = 0
        self.due = self.last_call = self.last_call_step = None

    def resolve(self, assignment, baseline_hit):
        a = copy.deepcopy(assignment)
        d, index = self.config, self.index
        a.update(anchor_index=index, pre_guard_call=not baseline_hit,
                 pre_guard_randomization=d["pre_guard"], credits_before=None if d["cap"] is None else max(0, d["cap"] - self.spent),
                 calls_before=self.spent, hold_before=self.hold_left, due_before=self.due,
                 last_call_anchor=self.last_call, last_call_step=self.last_call_step,
                 anchors_since_call=None if self.last_call is None else index - self.last_call,
                 coin_call=bool(a["assigned_call"]), nominal_propensity=a["propensity"],
                 cohort=d["cohort"], cohort_probability=d["cohort_probability"],
                 future_controller=copy.deepcopy(d), available_cache=True, available_policy=True)
        duration, pd = choose(d["durations"], a, "duration")
        hold, ph = choose(d["holds"], a, "hold")
        delay, pl = choose(d["delays"], a, "delay")
        p, call, reason = a["propensity"], a["coin_call"], "coin"
        trigger = False
        if index < d["start_anchor"]:
            p, call, reason = 0., False, "prefix"
        elif d["cap"] is not None and self.spent >= d["cap"]:
            p, call, reason = 0., False, "cap_exhausted"
        elif self.due is not None:
            call = index >= self.due
            p, reason = float(call), "delayed_due" if call else "delayed_wait"
            if call:
                self.due = None
        elif self.hold_left:
            p, call, reason = 1., True, "hold"
            self.hold_left -= 1
        elif self.last_call is not None and index - self.last_call <= d["cooldown"]:
            p, call, reason = 0., False, "cooldown"
        elif not baseline_hit and not d["pre_guard"]:
            p, call, reason = 1., True, "baseline_guard"
        elif d["delays"] != [0]:
            # Marginal immediate propensity integrates the independently drawn
            # delay, while the delay and trigger coin retain their own support.
            p = p * (sum(x == 0 for x in d["delays"]) / len(d["delays"]))
            trigger = call
            if call and delay:
                self.due = index + delay
                call, reason = False, "delay_scheduled"
        if call:
            if reason == "coin" and self.due is None:
                self.hold_left = hold - 1
            self.spent += 1
            self.last_call, self.last_call_step = index, a["step"]
        self.index += 1
        a.update(executed_policy=bool(call), actual_propensity=p, eligible=0 < p < 1,
                 baseline_hit=baseline_hit, injected=bool(call and baseline_hit),
                 suppressed_guard=bool(not call and not baseline_hit), override=reason,
                 scheduled_trigger=trigger, duration_choice=duration, duration_probability=pd,
                 hold_choice=hold, hold_probability=ph, delay_choice=delay, delay_probability=pl,
                 commit_controls=duration if call else 10, credits_after=None if d["cap"] is None else max(0, d["cap"] - self.spent),
                 joint_choice_probability=pd * ph * pl,
                 treatment_probability=p if call else 1 - p,
                 realized_source="policy" if call else "cache")
        return a
