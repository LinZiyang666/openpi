"""Reduced-step teacher (NFE) baseline: the policy alone with fewer denoising steps.

The cache-free counterpart of the RIT frontiers: every decision runs stage 1
and stage 2 in full and stage 3 for ``k`` of its ``N`` Euler steps, starting
from pure noise. One (IR, SR) point per ``k``, priced with the RIT lines' own
formula ``(s1 + s2 + (k / N) * s3) / MISS``, so a k-step teacher and a warm
start that leaves k steps sit at the same x. Both LIBERO suites, the official
pruned-500 pool, 500 episodes per point.

Modules:
  serve_pi05_ksweep  Pi0.5 teacher-only server with ``num_steps`` pinned to k
                     (the production entry point has no such flag).
  make_shards        Episode filters that split each task's pool across
                     client processes without overlap.
  aggregate_nfe      Shard results -> per-(policy, suite, k) success rate and
                     analytic inference ratio -> figure spec.
"""
