"""R15 legacy launcher for the Q3 batches (coordinator only; invoked by build.dispatch).

The unchanged rounds.r17.filter_failed.legacy (rsync transport), with local 4090 servers (GR00T full
step) started from the Q3 server tree. pi0.5 full-step servers run on the H100 copy
/data/oscl_h100/openpi, which needs the two files listed in HANDBACK.md.
"""
from pathlib import Path

from exp.offline_search.closed_loop.ops.h100 import local_server

local_server.TREE = Path('/home/weiland/projects/openpi_r16_q3_server')

from exp.offline_search.rounds.r17.filter_failed.legacy import main  # noqa: E402

if __name__ == '__main__':
    main()
