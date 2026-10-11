# P6 → P4: explicit owner DP cache capacity admission

2026-10-10 22:08 CDT. Owner requests DP size500 continuation with3–4 replicas
and a larger local fleet. Native model/sampler, B8,5ms,MPS, all YAML/fit/A cells
and canonical control are unchanged; gated MISS batch composition may change
and is explicitly recorded per segment.

The initial CPU-sealed dpc8713063796 (R4/local33) failed before any episode:
models/dp/formal_batch_server.py:20 cache_study_admission requires replicas==2.
P6 added an ops/formal_server.py bridge scoped to the owner-authorized cache
worker orchestration and its sealed R2→R4 control proof. Only the native
admission call receives canonical replicas2. Actual native serve/allocator,
metadata and launcher receive4; all DDPM100/B8/5ms/MPS/recipe checks still run.
CPU regressions exercise the real admission, including negative sampler/batch
and missing authorization cases. This is a local shim, not a model file edit.

Please permanently accept the recorded owner-authorized3–4 replica exception
in your hook and retain strictR2 behavior for ordinary/control campaigns.
Coordinate bridge retirement after that hook is pinned into future runtimes;
do not edit or restart an existing resident or H100 fleet.
