"""R16 resident that also appends every frame it receives to $CLUSTER_SIM_RAW (format: record.py).

Run as a script (record_run.py swaps it in for `-m exp.offline_search.rounds.r16.resident`); everything after
argv[0] is the resident's own argv, and the resident runs unchanged from its frozen server tree.
"""
import sys

sys.path.pop(0)  # this directory; the resident imports resolve through its own PYTHONPATH

import itertools
import os
from pathlib import Path
import runpy
import time

import msgpack
from websockets.asyncio.connection import Connection

RAW = Path(os.environ['CLUSTER_SIM_RAW'])
RAW.mkdir(parents=True, exist_ok=True)
_ids = itertools.count()
_recv = Connection.recv


async def recv(self, *args, **kwargs):
    data = await _recv(self, *args, **kwargs)
    if isinstance(data, (bytes, bytearray)):
        out = getattr(self, '_cluster_sim_raw', None)
        if out is None:
            out = self._cluster_sim_raw = (RAW / f'c{os.getpid()}_{next(_ids)}.msgpack').open('ab')
        out.write(msgpack.packb({'t': time.time(), 'f': bytes(data)}, use_bin_type=True))
        out.flush()
    return data


Connection.recv = recv
runpy.run_module('exp.offline_search.rounds.r16.resident', run_name='__main__', alter_sys=True)
