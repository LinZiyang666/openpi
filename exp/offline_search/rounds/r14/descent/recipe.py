"""Wire a frozen cap table and an explicit off mask into R14 task_periodic."""
from __future__ import annotations

import math
from exp.offline_search.harness import api
from exp.offline_search.rounds.r10.recipe.recipe import R10Recipe
from exp.offline_search.rounds.r14.knob.recipe import R14Knob

LEVELS = ('off', 'ir25', 'ir32', 'ir40', 'ir45', 'c10')


def validate_levels(levels):
    if not isinstance(levels, dict) or set(levels) != {str(t) for t in range(10)}:
        raise ValueError('task levels must explicitly map string task IDs 0..9')
    if any(level not in LEVELS for level in levels.values()):
        raise ValueError(f'allowed levels: {LEVELS}')
    return dict(levels)


class TaskDescentKnob(R14Knob):
    """The existing controller for enabled tasks; original lower path for off."""
    family = 'r14_task_descent'

    def __init__(self, library='current', episode_subset=None, *, task_levels, task_caps):
        super().__init__(library, episode_subset, method='task_periodic', target=.5)
        self.target = None
        self.set_table(task_levels, task_caps)

    def set_table(self, levels, caps):
        self.task_levels = validate_levels(levels)
        if len(caps) != 10 or any(not math.isfinite(k) or k < 0 for k in caps):
            raise ValueError('need ten finite nonnegative frozen caps')
        if any(caps[t] != 0 for t in range(10) if levels[str(t)] in ('off', 'c10')):
            raise ValueError('off/c10 cap placeholders must be zero; off uses a separate mask')
        self.configure(dict(K=list(caps), priority=[0.] * 10, reference_K=0., gain=0., bias=0.),
                       getattr(self, 'knob_model', 'pi05'))

    def fit(self, lib, ctx):
        raise api.ContractError('task descent requires a frozen fit artifact')

    def query(self, q):
        task = int(q.task_id)
        if not 0 <= task < 10 or task != int(q.episode.task_id):
            raise api.ContractError('task descent needs the original instruction task')
        level = self.task_levels[str(task)]
        if level == 'off':
            # No large finite surrogate cap: this path can never knob-call.
            result = R10Recipe.query(self, q)
            self._diagnostics(q, result, None)
            if self.r14_diagnostics is not None:
                self.r14_diagnostics.update(knob=False, guard=bool(result.extras.get('os_force_miss', 0)),
                                            K=None, cap=None)
        else:
            result = super().query(q)
        if self.r14_diagnostics is not None:
            self.r14_diagnostics.update(descent_level=level, method='task_descent')
        return result
