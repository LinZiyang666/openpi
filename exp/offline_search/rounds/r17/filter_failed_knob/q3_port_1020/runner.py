"""R16 conductor runner for the Q3 batches (coordinator only).

The unchanged rounds.r16.conductor_runner, with its resident servers started from the Q3 server tree
(the frozen R16 server checkout plus stall_only.py and the plugin's --os-judge stall_only mode) and a
7.2 GB start budget per GR00T replica (owner 10-20; GR00T residents stay 2 per suite = 4).
"""
from pathlib import Path

from exp.offline_search.rounds.r16 import conductor_runner as cr

TREE = Path('/home/weiland/projects/openpi_r16_q3_server')
GROOT_NEED_MB = 7200
_server_specs = cr.server_specs


def server_specs(group, ports, generation, **kwargs):
    specs = _server_specs(group, ports, generation, **kwargs)
    for spec in specs:
        if spec['model'] == 'groot':
            spec['need_mb'] = GROOT_NEED_MB
    return specs


cr.SERVER_TREE = TREE
cr.server_specs = server_specs

if __name__ == '__main__':
    cr.main()
