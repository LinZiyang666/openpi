"""cp1_search progressive tuning — optimized backend variants (exp-only).

Owner-overridden multi-round tuning workflow (see
``logs/cache_latency_bench_round1_stack_elim.log.md``). Every implementation here is
an ``InMemoryBackend`` subclass or a thin injector/comparator living entirely under
``exp/``; the src cache framework and the production backend are never modified.
"""
