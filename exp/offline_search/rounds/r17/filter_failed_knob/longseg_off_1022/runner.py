"""R16 conductor runner for the longseg_off_1022 reduced (K1) groups: residents on the H100 (coordinator only).

The q3_port_1020 runner (rounds.r16.conductor_runner unchanged; GR00T residents budgeted 7.2 GB each) with
the residents started on the H100 instead of the 4090: conductor_runner's local_server start/status/stop go
to h100_side.py in the H100 code tree (tether exec), the driver and the worker agents dial the H100's public
IP, the central scheduler is told the server machine is the H100, and each resident gets CPU threads from the
H100's 10 server cores. Driver and agents are unchanged (driver on weilandserver, agents a100/timan107/timan1;
always --no-local-workers: no renderers on the 4090). At DONE the residents' logs are copied back (h100.pull).
Prepare first: python -m ...longseg_off_1022.build prepare --name <group> (h100.sync).
"""
import json
from pathlib import Path

from exp.offline_search.rounds.r17.filter_failed_knob.q3_port_1020 import runner as _q3  # noqa: F401  (Q3 tree, GR00T 7200 MB)
from exp.offline_search.rounds.r16 import conductor_runner as cr
from exp.offline_search.closed_loop.ops.h100 import local_server

from . import h100

cr.ENDPOINT_HOST = h100.IP
cr.SERVER_NODE = 'h100'
cr.server_threads = lambda count, cpus=h100.SERVER_CPUS: max(1, min(4, cpus // max(1, count)))
local_server.start, local_server.status, local_server.stop = h100.start, h100.status, h100.stop
local_server.choose_ports = lambda count=4, requested=None, replicas=1: list(requested)  # checked on the H100 at start
# stop_servers' port-release fallback must not look at this machine's ports (the 4090's belong to another experiment);
# h100_side.stop already waits for the H100 port, and the fallback's second stop goes to the H100 again.
local_server.matching = lambda spec: None
local_server.wait_port_release = lambda spec, receipts, deadline: None
_run = cr.run


def run(args):
    args.no_local_workers = True  # no renderers on the 4090 (lent out); agents on a100/timan107/timan1 only
    _run(args)
    group = json.loads(Path(args.group).read_text())
    if (Path(group['root'])/'DONE.json').exists():
        h100.pull(args.group)


cr.run = run

if __name__ == '__main__':
    cr.main()
