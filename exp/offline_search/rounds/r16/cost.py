"""D5 pricing from the original full-precision stage authorities."""
from pathlib import Path
import json

from exp.dispatch_surface.analysis import analytic_cost
from exp.offline_search.closed_loop.devset import sha

REPO = Path(__file__).resolve().parents[4]
PI_SOURCE = REPO / 'exp/dispatch_surface/analysis/analytic_cost.py'
GROOT_SOURCE = REPO / 'exp/libero_groot/config/rit/cost_groot_libero_measured.json'


def pricing(model, k):
    if model == 'pi05':
        stages = [analytic_cost.STAGE1_MS, analytic_cost.STAGE2_MS, analytic_cost.STAGE3_MS]
        default, vision, full_miss, source = 10, .152, .848, PI_SOURCE
    elif model == 'groot':
        data = json.loads(GROOT_SOURCE.read_bytes())
        stages = [data['stage1_ms'], data['stage2_ms'], data['stage3_full_loop_ms']]
        default, vision, full_miss, source = data['num_steps'], .148, .852, GROOT_SOURCE
    else:
        raise ValueError(model)
    if type(k) is not int or k not in (1, 2, default):
        raise ValueError('Only k=1/2 or the full-step reference is registered')
    miss = full_miss if k == default else (stages[1] + stages[2]*k/default)/sum(stages)
    return dict(model=model, k=k, default_steps=default, vision=vision, miss=miss,
                stages_ms=stages, denominator_ms=sum(stages), source=str(source),
                source_sha256=sha(source), full_reference_uses_old_coefficients=k == default)
