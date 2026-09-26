"""Frozen offline evaluation harness for the retrieval-method exploration (R0-C).

Kept import-light on purpose: ``python -m exp.offline_search.harness.run`` imports this package
before run.py gets to pin the BLAS thread count, so nothing here may import numpy.

Public API (import from the submodules):
    exp.offline_search.harness.api       Method, Result, QueryView, EpisodeView, LibraryView, Context, Profiler
    exp.offline_search.harness.dims      valid-dimension constants and helpers (read these first)
    exp.offline_search.harness.metrics   per-decision metrics, aggregation, scoreboard
    exp.offline_search.harness.baselines B0..B4 reference methods
    exp.offline_search.harness.run       runner CLI
    exp.offline_search.harness.smoke     smoke / API-contract check CLI
    exp.offline_search.harness.gates     acceptance gates CLI
"""
