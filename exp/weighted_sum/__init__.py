"""Weighted-sum two-layer calibration + weight-search experiment (libero_spatial).

Phase 1 (offline): exp/common/calibrate_score_normalizers.py fits Layer-1
normalizers per (keybuilder, field). Phase 2 (conductor eval): emit_yamls.py +
weight_search_strategy.py sweep modality isolation and weight grids under an
always_hit judge, with held-out init states (init_holdout.py) to avoid the
library-vs-eval init-state leak.
"""
