"""Q3 boundary and C's anchor rule. No stall estimator is implemented in this package."""
from __future__ import annotations
import importlib
from pathlib import Path


class InactiveStallTracker:
    """Explicit CPU stub, used without a model or before Q3 installs its module."""
    def __init__(self, model=None, task_key=None):
        self.last_control = None

    def observe(self, key, control_index):
        if self.last_control is not None and control_index <= self.last_control:
            raise ValueError('fresh observation indices must increase')
        self.last_control = control_index

    def status(self):
        return dict(state='inactive', delta_hat=None, e90=None, a10=None,
                    window_span=0, W=None, reason='inactive_stub')


def load_stall(path):
    if path is None:
        return None, InactiveStallTracker, 'none'
    name = 'exp.offline_search.rounds.r06.ideation_Q3.stall.stall'
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if not (exc.name == name or name.startswith(exc.name + '.')):
            raise
        return None, InactiveStallTracker, 'inactive_stub_missing_module'
    model = module.StallModel.load(path)
    return model, module.StallTracker, model.fingerprint


def look_window(status, commit_controls):
    W = status.get('W')
    if W is None or int(W) != W or W < 1:
        raise ValueError('slow_ambiguous status requires positive library-derived W')
    return int(W) * int(commit_controls)


def extra_look(status, control_index, last_extra_control, commit_controls):
    if status['state'] != 'slow_ambiguous':
        return False
    window=look_window(status,commit_controls)
    return last_extra_control is None or control_index-last_extra_control >= window


STATE = dict(inactive=0, ok=1, slow_confirmed=2, slow_ambiguous=3)

# SELECTION §7b (2026-09-29 08:5x, still R6-C-v2). Shared by the deployed
# controller (methods.CalibratedRescue.query) and the cost model (budget.py),
# so modeled and deployed call/cooldown/LOOK rules cannot drift apart.
AMBIGUOUS_RULE = 'lottery_then_look'


def call_probability(state, cooled, nominal):
    """Call probability at a fresh anchor.

    A cooled anchor never calls; slow_confirmed always calls; every other state,
    slow_ambiguous included (§7b), draws the ordinary R/uniform lottery.
    """
    if cooled:
        return 0.
    return 1. if state == 'slow_confirmed' else nominal


def starts_cooldown(call, state, cooldown_scope):
    """R6-C-v2: only a stall-triggered call cools the next anchor ('all' is test-only)."""
    return bool(call) and (cooldown_scope == 'all' or state == 'slow_confirmed')


def scheduled_look(look_eligible, call):
    """§7b: the extra LOOK of a slow_ambiguous anchor happens only if its lottery did not call."""
    return bool(look_eligible) and not call
def verify_stall(model, base, bank):
    if model is None:
        return
    import json
    from pathlib import Path
    from exp.offline_search.rounds.r06.ideation_Q3.stall.stall import _extract_metric, _digest
    manifest=json.loads((Path(bank['library_directory'])/'manifest.json').read_text())
    if model.provenance['manifest_sha256'] != _digest(manifest):
        raise ValueError('stall artifact library manifest mismatch')
    if model.provenance['metric_sha256'] != _digest(_extract_metric(base,manifest)):
        raise ValueError('stall artifact deployed metric mismatch')
