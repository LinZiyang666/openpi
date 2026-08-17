# GR00T N1.5 support for the cache subsystem.
#
# The cache system was built around Pi0.5's three-stage inference API. GR00T
# splits in two, at a different place, and runs in a virtualenv that has no
# jax — so `openpi.cache.interceptor` cannot even be imported there. This
# subpackage is the parallel path: same CacheOrchestrator, same storage, same
# judges/gates/strategies, different front half.
#
# Nothing here imports `gr00t`. The model and policy are injected, so every
# module is exercisable with stubs outside the GR00T island.
#
#   staged       — split one GR00T forward into stage1 / stage2
#   key_builder  — cut cache keys out of stage1, by image-token mask
#   interceptor  — drive CacheOrchestrator around the split
#
# See docs/architecture/cache_system.md and logs/groot_cache_integration.log.md.
