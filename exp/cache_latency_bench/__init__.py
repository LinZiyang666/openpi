"""Cache latency replay benchmark.

Lifts the CP1 ``check()`` six segments (collect / gate / build / search /
judge / fetch) out of the inference stack into a lightweight, model-free
latency benchmark: assemble the real ``CacheOrchestrator`` from a cache
YAML exactly like the server does, replay real HDF5 trajectory data step
by step so the cache works "for real" (including cross-step trajectory /
score-memo / verdict memory), and record per-request, per-segment latency
via the orchestrator's own ``SystemTimer`` probes.

See ``logs/cache_latency_bench_plan.log.md`` for the design and the
equivalence / known-deviation analysis.
"""
