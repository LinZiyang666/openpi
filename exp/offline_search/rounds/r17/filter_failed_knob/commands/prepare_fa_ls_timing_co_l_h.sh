#!/usr/bin/env bash
# 10-20: timing is measured on the 4090 only (owner); the H100 timing units are skipped (rc 3).
# Restore prepare_fa_ls_timing_co_l_h.sh.before_skip_h100_timing to run it.
echo "H100 timing skipped: timing runs on the 4090"; exit 3
