"""Coordinator-only sparse adapter over R16's unchanged lifecycle and workers."""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from .. import conductor as c, conductor_runner as runner
from ..retest_runner import RetestDriver
from . import workflow as w


class SparseStrategy(c.R16Strategy):
    def plan(self, yamls, server_assignment):
        # Each dev wave contains only the next missing level. Cross-wave gates
        # are evaluated by workflow.dev_selection, never by R16's old reference.
        view = {**self.group, 'phase': 'test', 'references': None}
        graph = BaseStrategy(view).plan(yamls, server_assignment)
        for stage in graph.stages.values():
            original = self.entries[stage.yaml_id]['original_arm']
            stage.episodes = [dataclasses.replace(ep, task_uid=c.make_task_uid(original, 'eval', ep.task_id, ep.episode_idx))
                              for ep in stage.episodes]
        return graph

    def on_stage_complete(self, stage, ctl, ctx):
        pass


BaseStrategy = c.R16Strategy


def validate_ports(args, group):
    if args.port != 23159:
        raise ValueError('k1ref driver must use 23159; 23179 is reserved by the live Fast-WAM campaign')
    count = runner.server_count(group, args.groot_servers_per_suite)
    expected = list(range(23150, 23150 + count))
    if args.ports is None: args.ports = expected
    if args.ports != expected or not set(args.ports) <= set(range(23150, 23156)):
        raise ValueError('k1ref servers must use the fixed R16 allocation 23150..23155')
    if args.groot_servers_per_suite not in (None, 1 if 'smoke' in Path(group['root']).name else 2):
        raise ValueError('Use the R16 layout: pi05=3, GR00T=4; isolated smoke=1/2')


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('action', choices=['run'])
    ap.add_argument('--group', type=Path, required=True)
    ap.add_argument('--agent-island', default='/scratch/zixuans8/openpi_r16_conductor_v1')
    ap.add_argument('--workers', type=int, default=64); ap.add_argument('--port', type=int, default=23159)
    ap.add_argument('--ports', type=lambda s: list(map(int, s.split(','))))
    ap.add_argument('--groot-servers-per-suite', type=int); ap.add_argument('--smoke-episodes', type=int)
    args = ap.parse_args()
    if not (w.SERVER_REPO / 'R16_SNAPSHOT.sha256').is_file():
        raise ValueError('Run from the verified immutable R16 snapshot; see k1ref/HANDBACK.md')
    group = w.checked_group(args.group); validate_ports(args, group)
    # The smoke groups already contain exactly 2 identities per arm. Passing
    # --smoke-episodes would apply R16's separate slicing rule a second time.
    if args.smoke_episodes is not None: raise ValueError('Smoke pairs are sealed; omit --smoke-episodes')
    reader = w.a.Reader(); w.verify_code(reader); reader.stable()
    previous = c.R16Strategy, c.R16Driver, runner.read_group
    c.R16Strategy, c.R16Driver, runner.read_group = SparseStrategy, RetestDriver, w.checked_group
    try: runner.run(args)
    finally: c.R16Strategy, c.R16Driver, runner.read_group = previous


if __name__ == '__main__': main()
