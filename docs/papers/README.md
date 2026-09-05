# papers/

Literature references and related-work bibliographies supporting the fork's research framing.

| File | Description |
|------|-------------|
| [inference_cache_related_work.md](inference_cache_related_work.md) | Broad related-work list for inference caching / retrieval-augmented control in robotics and continuous control, organized by proximity to the OpenPI cache system |
| [cloud_edge_deployment.md](cloud_edge_deployment.md) | Cloud/edge deployment, brain-cerebellum split, fleet serving, compute/energy efficiency — deployment-context motivation for inference cache |
| [paper_workbench.md](paper_workbench.md) | Paper idea development: elevator pitch, motivation, story arc, method sketch, experiment plan, positioning, open questions（⚠ 2026-04 的 ENGRAM 系统叙事，已被 TIER 方向取代，待重写；现行论文工作文档见 [`../iclr/`](../iclr/README.md)） |
| [actioncache_2607.06370v2.pdf](actioncache_2607.06370v2.pdf) \[[TXT](actioncache_2607.06370v2.txt)\] | ActionCache (Oi et al., arXiv 2607.06370 **v2**, 2026-08-03) 本地副本 + `pdftotext -layout` 文本版，供逐句核对；concurrent work，攻防见 [`../iclr/actioncache_response_plan.md`](../iclr/actioncache_response_plan.md)。关键位置：§3.3 pending buffer / commit-on-success，§4.1 prefill(T_hit=1)+阈值肩部法+VLABench seed 不相交声明，App B.3 覆盖 LIBERO 的 seed/episode-ID 不相交总括声明，App F LIBERO 表(500 ep/suite, cache 10,000, T_hit 0.85, lerobot v044 ckpt) |
