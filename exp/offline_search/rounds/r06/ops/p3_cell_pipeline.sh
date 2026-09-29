#!/bin/bash
# Per finished pilot cell: strict read_v2 tables, then Q1/Q2/Q3 preregistered analyses (descriptive until 8 cells).
CELL=${1:?cell}
cd /home/weiland/projects/openpi
RUN=/home/weiland/trace_runs/os_closed_loop/r06_p3_pilot
ARMS=$(grep "^r6p3v2_${CELL}_" $RUN/pilot_queue.txt | tr '\n' ' ')
mkdir -p $RUN/tables
taskset -c 14-17,58-61 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r06.p3_profiling.read_v2 --run-root $RUN --arms $ARMS --client-root $RUN/runs --require-stage-counts --require-snapshots --out $RUN/tables/$CELL > $RUN/tables/$CELL.log 2>&1
rc=$?; echo "READ_V2_EXIT=$rc $(tail -1 $RUN/tables/$CELL.log | cut -c1-200)"
[ $rc -eq 0 ] || exit 1
bash /home/weiland/.claude/jobs/a607dd74/tmp/q_cell_analysis.sh $CELL
