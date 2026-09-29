#!/bin/bash
# Q1/Q2/Q3 preregistered analyses on one finished pilot cell's strict tables (descriptive until all 8 cells).
cd /home/weiland/projects/openpi
CELL=$1; TB=/home/weiland/trace_runs/os_closed_loop/r06_p3_pilot/tables/$CELL; D=exp/offline_search/rounds/r06
E="taskset -c 14-17,58-61 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1"
$E PYTHONPATH=.:src MPLCONFIGDIR=$D/ideation_Q1/.mplconfig .venv/bin/python $D/ideation_Q1/pilot_q1.py --tables $TB --out $D/ideation_Q1/pilot_$CELL > $TB.q1.log 2>&1; echo "Q1_EXIT=$?"
mkdir -p $D/ideation_Q2/.tmp $D/ideation_Q2/.mplconfig
$E PYTHONPATH=. TMPDIR=$D/ideation_Q2/.tmp MPLCONFIGDIR=$D/ideation_Q2/.mplconfig .venv/bin/python $D/ideation_Q2/pilot_q2.py --tables $TB --stage pilot --out $D/ideation_Q2/pilot_$CELL > $TB.q2.log 2>&1; echo "Q2_EXIT=$?"
E3="taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1"; $E3 PYTHONPATH=.:src MPLCONFIGDIR=$D/ideation_Q3/.mplconfig .venv/bin/python $D/ideation_Q3/analyze_pilot.py --tables $TB --stage pilot --calibration-dir /home/weiland/trace_runs/os_closed_loop/r06_p3_pilot/calibration --out $D/ideation_Q3/pilot_$CELL > $TB.q3.log 2>&1; echo "Q3_EXIT=$?"
echo "Q_ANALYSIS_DONE $CELL"
