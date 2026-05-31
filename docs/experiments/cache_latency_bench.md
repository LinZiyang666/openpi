# Cache 延迟回放基准 (cache_latency_bench) — Runbook

把 cache 系统的 **CP1 `check()` 六段**（collect / gate / build / search / judge / fetch）从推理栈里切出来，做成**不加载任何模型**的轻量延迟基准：按真实 cache yaml 像 server 一样组装 `CacheOrchestrator`，用 H5 真实采集的 trajectory 逐 step 回放驱动 cache "真实工作"（含跨 step 的 trajectory / score-memo / verdict 记忆），用 orchestrator 自带的 `SystemTimer` 探针记录每请求、每部件延迟。

- 代码：[`exp/cache_latency_bench/`](../../exp/cache_latency_bench/)（`h5_episode.py` / `replay.py` / `run.py` / `summarize.py`）+ [`README`](../../exp/cache_latency_bench/README.md)
- 设计与等价性 / 已知偏差分析：[`logs/cache_latency_bench_plan.log.md`](../../logs/cache_latency_bench_plan.log.md)

## 前置数据

- **库**：`exp/common/data/cache_artifacts/<suite>/*.pkl`（InMemoryBackend artifact；composite/运动学 verdict 需已 enrich，含 offline factor 的 `payload.factors` keys）。
- **trajectory**：`exp/common/data/db/libero_cache/<suite>/*.h5`（一个 H5 = 一个 episode；存已投影的 Stage1 级 token embedding，不是原始图）。
- **cache yaml**：要测的真实配置，`write_policy` 必须是 `{type: never}`（= 真 server runtime C2 契约）；composite judge 的 Layer 3 calibration 必须 `samples_source.type=offline`。

## 跑

```bash
uv run exp/cache_latency_bench/run.py \
  --cache-config <真实 cache yaml> \
  --h5-dir exp/common/data/db/libero_cache/libero_10 \
  --out-dir exp/cache_latency_bench/data/run0 \
  --repeats 1
```

输出（落 `--out-dir`，`data/` 默认 gitignore）：
- `per_step.csv` — 每 step 一行：`repeat, episode_id, step_idx, task, hit_type, top_score, cp1_{collect,gate,build,search,judge,fetch}_ms, cp1_total_ms, warm_start_action_approx, action_history_approx_active`。
- `summary.json` — per 段 × hit_type 的 count/median/p50/p95 + run 级 hit_counts/hit_rate/`judge_consumes_action_history` + `meta`（`build_excludes_d2h` / `checkpoints_driven`）。

单独聚合：`uv run exp/cache_latency_bench/summarize.py --csv <per_step.csv> --out <summary.json>`。

## 怎么读结果

研究 **t1**（cache 系统延迟）的段间相对占比、随库规模/depth 的标度、in-DB vs out-of-DB 的 fetch 段差异。**绝对毫秒跨机不可比**（CPU 硬件/负载相关）。

## 已知偏差（务必读，详见 plan §6）

1. **CPU 模式 `cp1_build` 不含 GPU→CPU D2H/CUDA 同步**（真 server build 在 GPU 上 reduce 后 D2H）。其余 5 段纯 CPU 逻辑、真实。
2. **仅驱动 CP1**：CP3-enabled yaml 的 CP3 段不计入，`cp1_total` ≠ 含 CP3 的真 server 单 step cache 延迟。
3. **读 action-history 的 composite judge（online action factor）非逐调用等价**：FULL_HIT 等价；MISS 真 server 随机 noise 重采（infra 喂采集样本，run-to-run 异）；WARM_START infra 以完整去噪 `clean_action` 充当部分去噪 action（类型错误）。CSV `warm_start_action_approx` / `action_history_approx_active`（cumulative）+ json `judge_consumes_action_history` 标注。不读 action-history 的 judge 则逐调用等价。

## 支持范围

- ✅ judge：`threshold` / `always_hit` / `always_warm_start`；`composite`（前提：pkl enrich + calibration `samples_source=offline`）。
- ❌ `composite` + `samples_source=warmup`（无 WarmupPool 链路 → harness fail-loud）；`cp1_llm_layer_extract`（需真模型）；非 in-memory backend 配 composite/offline-factor。

## 测试

`uv run pytest tests/exp/test_cache_latency_bench.py`（全 CPU / non-manual / 合成 mini 数据 / CI 可跑）。
