Owner acceptance criterion (2026-10-07 20:4x): "concurrentserver的热切换结合conductor的yaml episode级别的切分能把气泡减低为0" / "我们之前episode级别的细切分能做到yaml之间没有气泡".
- All bundles (yaml/method per arm) of a group are preloaded and acknowledged before the group starts; switching a connection to another arm's yaml at an episode boundary must not idle the worker or the server.
- The smoke must measure it: per worker, the gap between finishing one episode and starting the next (including across yaml switches) ≈ 0 apart from normal reset time; per server, no idle windows between arms; report the distribution (p50/p99) and the end-of-group tail.
- Not acceptable: arm-by-arm restarts inside a k group, waiting for a whole arm to finish before dispatching another arm's episodes, or per-arm sync/plan inside the group.
