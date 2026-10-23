#!/bin/bash
# Hold pod2lo5 once its full-step batch-8 reference starts: its last item fa_lo_fw_f_f_m_j is batch 16, which
# overflows the 16 GiB MIG slice (fa_lo_fw_r_f_m_j died the same way 21:48). pod2lo6 runs the batch-8 copies.
M=/home/weiland/projects/openpi_ext/scratch/r17_ops/mw
until grep -q "\[ffk-pod2lo5\] start fa_lo_fw_f_b_m_j2\|\[ffk-pod2lo5\] fa_lo_fw_f_b_m_j2 prepare FAILED" $M/chain.log; do sleep 20; done
touch $M/hold_ff_pod2lo5; echo "$(date +%T) hold_ff_pod2lo5 set"
