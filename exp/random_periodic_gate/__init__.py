"""Random & Periodic gate sweep experiment.

Independent of ``exp/trajectory_deviation``. Exercises two new server-side
self-contained gates (``RandomGate`` / ``PeriodicGate``) over the full
LIBERO ``libero_spatial`` suite with the three keybuilder weight bundles
borrowed from trajectory_deviation (``clip_w7_d4`` / ``spatial16_w8_d4`` /
``max_pool_w3_d5``).

Layout:
  config/base_<cfg>.yaml        — three template bundles, one per keybuilder.
  config/batch{1..6}/*.yaml     — 114 generated YAMLs (19 per batch,
                                   produced offline by generate_batches.py).
  generate_batches.py           — offline generator (idempotent; tracked).
  run_gate_sweep.py             — runner driven by ``--batch-dir``.
  analysis/analyze_gate_sweep.py — JSONL -> CSV + Pareto/heatmap.

Plan: ``logs/random_periodic_gate_plan.log.md``.
"""
