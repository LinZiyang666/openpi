"""Shared constants for phase1 / trajectory plotting scripts.

Centralizes the key-builder / weight-id / color / marker tables that were
previously copy-pasted between ``analysis/phase1/.../plot_results.py`` and
``analysis/trajectory/.../plot_results.py``. Every symbol is plain data; the
module has no side effects beyond the matplotlib colormap lookups that already
happened at import time in the original scripts.
"""

import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------
# Key builders
# ------------------------------------------------------------------

KEY_BUILDER_ORDER = ["a", "b1", "b2", "c", "d"]

KEY_BUILDER_LABELS = {
    "a":  "cp1_mean_pool",
    "b1": "cp1_spatial_pool_16",
    "b2": "cp1_spatial_pool_64",
    "c":  "cp1_max_pool",
    "d":  "clip",
}

# ------------------------------------------------------------------
# Weights (v0 / v1 / rs refer to vision_0 / vision_1 / robot_state;
# vision_2 and prompt_emb are disabled in both experiments)
# ------------------------------------------------------------------

WEIGHT_IDS = [f"w{i}" for i in range(1, 9)]

WEIGHT_LABELS = {
    "w1": "v0=1.0",
    "w2": "rs=1.0",
    "w3": "v0=.5 rs=.5",
    "w4": "v0=.25 rs=.75",
    "w5": "v0=.25 v1=.25 rs=.5",
    "w6": "v0=.15 v1=.1 rs=.75",
    "w7": "v0=.1 v1=.1 rs=.8",
    "w8": "v0=.5 v1=.25 rs=.25",
}

# ------------------------------------------------------------------
# Colors & markers
# ------------------------------------------------------------------

# Per-weight bar colors (phase1 grouped-bar chart + trajectory facet bars).
BAR_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))[:8]

# Per-key-builder line colors (trajectory line chart), one consistent color
# per kb across every figure.
KB_COLORS = {
    kb: c
    for kb, c in zip(KEY_BUILDER_ORDER, plt.cm.tab10(np.linspace(0, 1, 10)))
}

# Markers encode the role of each weight in the phase1 ranking.
#   top1       -> star       (best phase1 result for that kb)
#   top2       -> square     (second-best phase1 result)
#   2nd_worst  -> circle     (second-worst phase1 result)
ROLE_MARKERS = {"top1": "*", "top2": "s", "2nd_worst": "o"}
ROLE_LABELS = {
    "top1": "phase1 top-1",
    "top2": "phase1 top-2",
    "2nd_worst": "phase1 2nd-worst",
}
