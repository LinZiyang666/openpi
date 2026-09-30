from pathlib import Path

from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import sha, write_json

HERE = Path(__file__).resolve().parent
SCRATCH = Path('/tmp/r7_C3')
RECORDINGS = Path('/home/weiland/trace_runs/os_closed_loop/r06_c_cal')
CELLS = ('pi05_l10_50', 'pi05_spatial_50', 'groot_l10_50', 'groot_spatial_50')
DENSE_CELLS = ('pi05_l10_500', 'pi05_spatial_500', 'groot_l10_500', 'groot_spatial_500')
ALL_CELLS = CELLS + DENSE_CELLS
DENSE_STAGES = Path('/tmp/r7_C1/stages')
VERSION = 'R7-C3-v1'


def target_rho(cell):
    if cell not in ALL_CELLS:
        raise ValueError('unsupported C3 cell: ' + cell)
    return .18 if cell in DENSE_CELLS else .30


def output_path(path):
    path = Path(path).resolve()
    if not any(path == root or root in path.parents for root in (HERE, SCRATCH)):
        raise ValueError('C3 writes must stay in c3_calls/ or /tmp/r7_C3/')
    return path


def stall_path(cell):
    return RECORDINGS / 'stall' / cell.replace('spatial', 'sp')
