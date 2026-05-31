# cache_latency_bench — cache CP1 延迟回放基准

把 cache 系统的 **CP1 `check()` 六段**（collect / gate / build / search / judge / fetch）从推理栈里切出来，做成一个**不加载任何模型**的轻量延迟基准：按真实 cache yaml 像 server 一样组装 `CacheOrchestrator`，用 H5 里真实采集的 trajectory 逐 step 回放驱动 cache "真实工作"（含跨 step 的 trajectory / score-memo / verdict 记忆），用 orchestrator 自带的 `SystemTimer` 探针记录每请求、每部件的延迟。

设计与等价性分析见 [`logs/cache_latency_bench_plan.log.md`](../../logs/cache_latency_bench_plan.log.md)。

## 用法

```bash
uv run exp/cache_latency_bench/run.py \
  --cache-config <真实 cache yaml> \
  --h5-dir exp/common/data/db/libero_cache/libero_10 \
  --out-dir exp/cache_latency_bench/data/run0 \
  --repeats 1
```

输出（落 `--out-dir`，data/ 默认 gitignore）：
- `per_step.csv` — 每 step 一行：`repeat, episode_id, step_idx, task, hit_type, top_score, cp1_{collect,gate,build,search,judge,fetch}_ms, cp1_total_ms, warm_start_action_approx, action_history_approx_active`。
- `summary.json` — per 段 × hit_type 的 count/median/p50/p95 + run 级 hit_counts/hit_rate/`judge_consumes_action_history`。

也可单独跑 `uv run exp/cache_latency_bench/summarize.py --csv per_step.csv --out summary.json`。

## 支持范围

- ✅ judge：`threshold` / `always_hit` / `always_warm_start`；`composite`（运动学 verdict）**前提** pkl 已 enrich（含所需 offline factor 的 `payload.factors` keys）+ Layer 3 calibration `samples_source.type=offline`。
- ❌ `composite` + `samples_source.type=warmup`（infra 无 WarmupPool 预填链路 → harness 启动 fail-loud，请改 offline）。
- ❌ `cp1_llm_layer_extract` keybuilder（需真模型）；❌ 非 in-memory backend 配 composite/offline-factor。
- 输入 yaml 必须 `write_policy: {type: never}`（= 真 server runtime 的 C2 契约，否则 fail-fast）。

## 已知偏差（务必读）

1. **CPU 模式 `cp1_build` 不含 GPU→CPU D2H/CUDA 同步**（真 server build 在 GPU 上 reduce 后 D2H）。其余 5 段为纯 CPU 逻辑、真实。`summary.json.meta.build_excludes_d2h=true` 标注。
2. **仅驱动 CP1**：CP3-enabled yaml 的 CP3 段不计入，`cp1_total` ≠ 含 CP3 的真 server 单 step cache 总延迟（`meta.checkpoints_driven=cp1_only`）。
3. **读 action-history 的 composite judge（online action factor）非逐调用等价**：FULL_HIT 步等价；MISS 步真 server 用随机 noise 重采、infra 喂采集样本（run-to-run 异）；WARM_START 步 infra 以完整去噪 `clean_action` 充当部分去噪 action（类型错误）。CSV 的 `warm_start_action_approx` / `action_history_approx_active`（cumulative）+ json `judge_consumes_action_history` 标注；不读 action-history 的 judge 则逐调用等价。
4. **绝对延迟与 CPU 硬件/负载强相关**：结论用于「段间相对占比 / 随库规模·depth 的标度 / in-DB vs out-of-DB 的 fetch 段差异」，而非跨机绝对毫秒。
