"""The Q3 runner (q3_port_1020/runner.py) for the recording group, with residents started via record_resident.py.

  CLUSTER_SIM_RAW=/nvme/weiland/openpi/cluster_sim/raw python -m exp.cluster_sim.record_run run \
      --group /data/mw_finetune/eval/cs_rec_pi05/group.json --agent-island /scratch/zixuans8/openpi_r16_conductor_v1 \
      --port 23159 --ports 23150,23151,23152
"""
import os
from pathlib import Path

from exp.offline_search.rounds.r16 import conductor_runner as cr
from exp.offline_search.rounds.r17.filter_failed_knob.q3_port_1020 import runner  # noqa: F401 (installs the Q3 server tree)

SCRIPT = Path(__file__).with_name('record_resident.py')
_command = cr.resident_command


def resident_command(spec):
    argv, env = _command(spec)
    i = argv.index('-m')
    argv[i:i + 2] = [str(SCRIPT)]
    return argv, dict(env, CLUSTER_SIM_RAW=os.environ['CLUSTER_SIM_RAW'])


cr.resident_command = resident_command

if __name__ == '__main__':
    cr.main()
