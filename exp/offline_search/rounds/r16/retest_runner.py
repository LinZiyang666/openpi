"""Run new retest groups through the unchanged resident/conductor/worker path.

Only this controller process substitutes a sparse strategy and UID→arm join.
The server and remote worker import the existing modules, never this adapter.
"""
from __future__ import annotations

import copy
import dataclasses
from pathlib import Path

from . import conductor as c
from . import conductor_runner as runner
from . import retest as rt
from . import analyze as a


class RetestStrategy(c.R16Strategy):
    def __init__(self, group, **kwargs):
        group = copy.deepcopy(group)
        a.require(group.get('retest') and group['phase'] == 'test', 'Use a new retest test-phase group')
        for entry in group['entries']: entry['pairs'] = entry['dispatch_pairs']
        super().__init__(group, **kwargs)

    def plan(self, yamls, server_assignment):
        graph = super().plan(yamls, server_assignment)
        for stage in graph.stages.values():
            original = self.entries[stage.yaml_id]['original_arm']
            stage.episodes = [dataclasses.replace(ep, task_uid=c.make_task_uid(original, 'eval', ep.task_id, ep.episode_idx))
                              for ep in stage.episodes]
        return graph


class RetestDriver(c.R16Driver):
    def _uid_meta(self, task_uid):
        for arm, entry in self._strategy.entries.items():
            if task_uid.startswith(entry['original_arm'] + ':eval:'):
                a.require(list(a.uid_pair(task_uid)) in entry['dispatch_pairs'], 'Unscheduled original UID')
                return arm, 'eval'
        raise ValueError('Unknown retest original UID')


def checked_group(path, reader=None):
    path = Path(path)
    reader = reader or a.Reader(); design = rt.verify(reader)
    group = c.read_group(path, reader); plan = rt.read_plan(group['retest_plan'], reader)
    a.require(str(path.resolve()) in plan['groups'] and group['retest_lock_sha256'] == plan['retest_lock_sha256'] and
              group.get('original_episode_uids') and group['phase'] == 'test' and
              group['group_sha256'] == plan['group_sha256'][str(path.resolve())], 'Unregistered retest group')
    a.require(group['retest_selection'] == {k: plan[k] for k in ('changes', 'freeze', 'generation', 'selected_at_utc')},
              'Changed retest selection record')
    original = c.read_group(group['original_group'], reader)
    a.require(original['group_sha256'] == group['original_group_sha256'], 'Changed original test group')
    adjusted = reader.json(Path(group['retest_plan']).parent / 'levels.json')
    expected = {(d['cell'], d['task']) for d in plan['changes']
                if (rt.parts(d['cell'])[0], rt.parts(d['cell'])[3]) == (group['model'], group['k'])}
    actual = set()
    for entry in group['entries']:
        row = entry['row']; cell = row['r16_cell']; selected = entry['dispatch_pairs']
        tasks = sorted({t for t, i in selected})
        a.require(selected == [[t, i] for t in tasks for i in range(50)] and entry['pairs'] == selected and
                  entry['freeze'] == plan['freeze'], 'Wrong retest cohort/freeze')
        table = adjusted[cell]
        old = next(e for e in original['entries'] if e['row']['r16_cell'] == cell)
        a.require(entry['original_arm'] == old['row']['arm'] and entry['original_fit_sha256'] == old['fit_sha256'],
                  'Changed original episode/fit binding')
        a.require(row['kwargs']['task_levels'] == table and row['kwargs']['task_caps'] ==
                  [design['cells'][cell]['periodic_caps'][table[str(t)]] for t in range(10)], 'Wrong retest level/cap')
        a.require(bool(entry['passive']) == (not tasks), 'Wrong passive checkpoint method')
        actual.update((cell, t) for t in tasks)
    a.require(actual == expected, 'Retest dispatch differs from owner changes')
    reader.stable(); return group


def main():
    # Reuse the existing CLI, lifecycle, fleet locks, receipts and attempt fences.
    # Patches affect only this process and are restored on exit.
    originals = c.R16Strategy, c.R16Driver, runner.read_group
    c.R16Strategy, c.R16Driver, runner.read_group = RetestStrategy, RetestDriver, checked_group
    try: runner.main()
    finally: c.R16Strategy, c.R16Driver, runner.read_group = originals


if __name__ == '__main__': main()
