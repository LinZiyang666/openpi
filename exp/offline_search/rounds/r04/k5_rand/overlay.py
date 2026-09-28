"""Single-landmark experiment only; no learned controller or process-global RNG."""
import hashlib
import math
import random


def assignment(seed, task_id, init, replicate):
    if replicate not in (1, 2) or task_id < 0 or init < 0:
        raise ValueError('randomization requires replicate 1/2 and nonnegative task_id/orig_init_state_idx')
    key = f'causal_rescue_credit:v1:{int(seed)}:{int(task_id)}:{int(init)}'
    rng = random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest(), 'big'))
    landmark = (1, 3)[rng.getrandbits(1)]
    call = bool(rng.getrandbits(1)) ^ (replicate == 2)
    return dict(trial_id=f'{key}:r{replicate}', landmark_class=landmark,
                assigned_treatment='CALL' if call else 'CACHE', propensity=0.5,
                experiment_seed=int(seed), replicate=int(replicate), task_id=int(task_id), init=int(init))


def validate_options(opts):
    seed = getattr(opts, 'os_rand_seed', None)
    rep = getattr(opts, 'os_rand_replicate', None)
    if seed is None and rep is None:
        return False
    if seed is None or rep is None:
        raise SystemExit('--os-rand-seed and --os-rand-replicate are required together')
    if opts.os_blind:
        raise SystemExit('--os-rand-seed does not support --os-blind: this trial requires vision at every decision')
    j = opts.judge
    if j is None or j.mode != 'guard_only' or j.cap or j.burst != 1 or j.step0 != 'judge':
        raise SystemExit('randomization requires --os-judge guard_only, cap=0, burst=1, step0=judge')
    if opts.os_cell != 'pi05_l10_cache':
        raise SystemExit('randomization is specified only for pi05_l10_cache')
    return True


def validate_method(method):
    # Covers exact MixedJudge and test subclasses; events/bursts would change the estimand.
    from exp.offline_search.rounds.r03.h3_judge.judge import MixedJudge
    # File-path loading uses a distinct module; check the defining class file as well.
    import inspect
    from pathlib import Path
    valid = any(c.__name__ == 'MixedJudge' and Path(inspect.getfile(c)).resolve() ==
                Path(inspect.getfile(MixedJudge)).resolve() for c in type(method).__mro__)
    if not valid or not method.guards or method.events or method.burst:
        raise SystemExit('randomization requires guard-only MixedJudge (guards=true, events=none, burst=0)')


class RandomizedLandmark:
    def __init__(self, seed, task_id, init, replicate):
        self.assigned = assignment(seed, task_id, init, replicate)
        self.opportunities = 0
        self.exposed = False
        self.stall_start = None

    def apply(self, step, baseline_hit, confidence, extras, previous_action):
        # All features are evaluated before applying the randomized verdict.
        ex = extras or {}
        if self.stall_start is None and (ex.get('stuck_n', 0) >= 2 or ex.get('noprog_n', 0) >= 2):
            self.stall_start = step
        age = None if self.stall_start is None else step - self.stall_start
        progress = float(ex.get('top1_prog', math.nan))
        grip = None if previous_action is None else float(previous_action[4, 6])
        context = dict(progress=None if not math.isfinite(progress) else ('<0.5' if progress < .5 else '>=0.5'),
                       gripper=None if grip is None or not math.isfinite(grip) else ('closed' if grip >= 0 else 'open'),
                       confidence=None if not math.isfinite(confidence) else ('>-0.3' if confidence > -.3 else '<=-0.3'),
                       stall_age='before' if age is None else ('0-9' if age < 10 else '>=10'))
        if not baseline_hit:
            self.opportunities += 1
        eligible = not baseline_hit and not self.exposed and self.opportunities == self.assigned['landmark_class']
        actual_hit = bool(baseline_hit)
        if eligible:
            actual_hit = self.assigned['assigned_treatment'] == 'CACHE'
            self.exposed = True
        row = dict(self.assigned, eligible=bool(eligible), opportunity_index=self.opportunities,
                   baseline_verdict='HIT' if baseline_hit else 'MISS', actual_verdict='HIT' if actual_hit else 'MISS',
                   context=context, context_values=dict(progress=progress if math.isfinite(progress) else None,
                       gripper=grip, confidence=confidence if math.isfinite(confidence) else None, stall_age=age))
        return actual_hit, row
