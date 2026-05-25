# docs/architecture/

Cache system specifications and end-to-end workflow diagrams.

| File | Description |
|------|-------------|
| [cache_system.md](cache_system.md) \[[ZH](cache_system.zh.md)\] | Cache system spec: 3-stage pipeline, CP1/CP2/CP3 checkpoints, interceptor pattern, component design. Chinese companion frozen at 2026-04-03 |
| [cache_workflow.md](cache_workflow.md) | End-to-end workflow diagrams: startup, single inference with CP1/CP3, episode lifecycle, storage layer, YAML mapping, design principles |
| [experiment_conductor.md](experiment_conductor.md) | 两层实验编排框架：worker/agent/driver 三层 + 机制(src/openpi/conductor)/策略(exp/)分离；episode 级无空隙调度（yaml 亲和 / 永不空转 / warmup→eval barrier）；账本断点续跑 + server 自愈(B)；重试分类 / 健康 / 聚合监控；直连单进程端点、server 协议不动。详见 [`logs/client_conductor_two_layer_refactor.log.md`](../../logs/client_conductor_two_layer_refactor.log.md) |

Back to [docs index](../README.md).
