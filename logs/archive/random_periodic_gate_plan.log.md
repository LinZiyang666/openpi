# Random & Periodic Gate Sweep — Experiment Plan

> **Status**: `Plan`
> **Level**: **L2**（新增 2 个 Gate 类 + `_GATE_TYPES` / `GateConfig` / `_build_gate` 扩展 + 新实验子目录 + 新 runner / YAML 生成器 / analysis 脚本 + 对应 tests）
> **Authority**: Execution
> **Owner**: Ziyang Lin
> **Drafted**: 2026-04-24

## 0. TL;DR

在 cache 系统里新增两种**完全 server-side 自包含**的 gate，独立于 `trajectory_deviation`：

- **`RandomGate(p_inference, seed)`** — 每次 gate 调用独立伯努利抽样：以 `p_inference` 返回 skip（跳过 cache → 走真 inference），否则返回 search（查 cache）。
- **`PeriodicGate(cache_len k, inference_len n)`** — 内部计数器：每 episode 先 k 个 search，再 n 个 skip，循环至 episode 结束；episode 起点 reset 计数。

复用 `trajectory_deviation` 的 3 套 keybuilder 权重（`clip_w7_d4` / `spatial16_w8_d4` / `max_pool_w3_d5`），
judge 统一用 `AlwaysHitJudge`，LIBERO `libero_spatial` 全量 500 ep（10 tasks × 50 inits），scan：

| Gate | 参数网格 | 组合数 |
|------|---------|-------|
| PeriodicGate | `cache_len ∈ {1, 2, 5, 10}` × `inference_len ∈ {1, 2, 3, 5, 10}` | 20 |
| RandomGate | `p_inference ∈ {0.05, 0.10, 0.20, 0.30, 0.50, 0.70}` × `seed ∈ {0, 1, 2}` | 18 |

**3 cfg × 38 组 × 500 ep ≈ 57,000 ep**（3 server × 5 worker 完全并行，batch ↔ cfg 一一绑定，单 batch 预计 ~6–10h）。

Baseline `AlwaysSearchGate` / `AlwaysSkipGate` 端点不重跑，**直接从 `exp/trajectory_deviation/data/cache_eval_results.json` 中对应 cache 行与 `inference_*` 行读取 success 列做 join**，不消费 Step 1b `gt/` trajectory 目录。

---

## 1. 背景

`trajectory_deviation` Step 3 redesign §7 留下 Step 4/5 两个 baseline 实验，原设计与 Step 3 runner 共享 episode 子集（per-cfg fail set）+ `ClientControlledGate` 机制。**本实验改走独立路径**：
- episode 集合改为 `libero_spatial` 全量 500 ep — 独立 baseline 才能和未来其他实验直接对齐；
- gate 不再需要 client 在 obs 里塞 `__gate_decision__` —— RandomGate / PeriodicGate 仅依赖 gate 自身状态；
- runner 不复用 `exp/trajectory_deviation/run_step3_per_cycle_policy.py` 的 `deviate_score_json` 约束。

这样做的好处：
1. 与 `trajectory_deviation` 完全解耦，后续单独发展成本低；
2. 两个 gate 是普适能力，将来任何 cache 实验都能直接引用；
3. server 自包含决策，client runner 等价于 `examples/libero/main.py` 的标准 rollout。

## 2. 实验设计

### 2.1 Gate 语义（对齐 `GateFunction` 协议）

协议约定：`gate.__call__(...) -> bool` 中 `True = search cache`，`False = skip cache`（fall-through 走真 inference）。Skip 路径由 orchestrator 保证 `record_query_keys + broadcast_action` 正常记录，trajectory 历史不断档（见 `src/openpi/cache/components/gate.py:82-112` `AlwaysSkipGate` 说明）。

### 2.2 扫描网格

| 参数 | 取值 | 备注 |
|------|------|------|
| keybuilder cfg | `clip_w7_d4` / `spatial16_w8_d4` / `max_pool_w3_d5` | 对应 `exp/trajectory_deviation/config/` 同名 YAML 的 key/weight/search_strategy/trajectory_depth |
| task_suite | `libero_spatial` | 全量 10 tasks × 50 inits = 500 ep |
| PeriodicGate `cache_len` | `{1, 2, 5, 10}` | |
| PeriodicGate `inference_len` | `{1, 2, 3, 5, 10}` | |
| RandomGate `p_inference` | `{0.05, 0.10, 0.20, 0.30, 0.50, 0.70}` | |
| RandomGate `seed` | `{0, 1, 2}` | 3 seed 取均值消方差 |

`(cache_len=1, inference_len=1)` 等价于 50% duty cycle；`cache_len=1, inference_len=10` 等价于"cache 频率极低"。RandomGate `p_inference=0.50` 与 PeriodicGate `(1,1)` 对照期望 inference_ratio 相近但随机性不同，留给 analysis 对比。

### 2.3 指标

| 指标 | 定义 | 类型 | 精确 vs 估计 |
|------|------|------|-------------|
| Success | LIBERO `_check_success()` 在 episode 终止前达成 | 主指标 | 精确 |
| Total cycles | 一个 episode 走了多少次 `conn.infer(obs)` | 辅助 | 精确（runner 本地计数） |
| `num_inference_cycles` | PeriodicGate：`sum(1 for i in range(total_cycles) if i % (k+n) >= k)`；RandomGate：**不精确测量**（见下） | 成本指标 | 见右 |
| `inference_ratio` | `num_inference_cycles / total_cycles`（PeriodicGate）或 `expected_inference_ratio = p_inference`（RandomGate） | 成本指标 | PeriodicGate 精确；RandomGate 估计 |

**关键约定（Round 2 owner 澄清）**：
- 本实验**不**扩 cache framework 的 stats 传输协议（不改 orchestrator / interceptor / ws 协议 / client 库）。
- gate 决策在 server 内自包含，client / runner 无法直接 observe 每次 `infer()` 是 search 还是 skip。
- **PeriodicGate** 的 skip 序列完全由 `(cache_len, inference_len, cycle_idx)` 决定，runner 本地知道 `total_cycles`，可用闭式公式事后推出精确 `num_inference_cycles`。
- **RandomGate** 的 skip 序列是随机变量，runner 无法精确观测 — 按 owner 指示记 `expected_inference_ratio = p_inference`，`num_inference_cycles_estimated = round(total_cycles * p_inference)`，JSONL 显式标注 `inference_ratio_source="expected"`。aggregate 层面跨 seed + 跨 ep 的 success rate 仍然有意义。

Per-ep JSONL schema：

```json
{
  "cfg": "clip_w7_d4",
  "gate_type": "periodic",
  "gate_params": {"cache_len": 5, "inference_len": 2},
  "seed": null,
  "task_id": 3,
  "init_idx": 7,
  "ep_key": "task_3/init_7",
  "success": true,
  "total_cycles": 44,
  "num_inference_cycles": 12,
  "inference_ratio": 0.273,
  "inference_ratio_source": "derived"
}
```

```json
{
  "cfg": "clip_w7_d4",
  "gate_type": "random",
  "gate_params": {"p_inference": 0.20},
  "seed": 1,
  "task_id": 3,
  "init_idx": 7,
  "ep_key": "task_3/init_7",
  "success": true,
  "total_cycles": 44,
  "num_inference_cycles_estimated": 9,
  "inference_ratio": 0.20,
  "inference_ratio_source": "expected"
}
```

### 2.4 Baseline 接入（只在 analysis，不重跑）

权威源：`exp/trajectory_deviation/data/cache_eval_results.json` — 3000 行 list，字段 `task_id / init_state_idx / orig_init_state_idx / episode_id / seed / success / config_id / attempt / source_path`；包含 6 个 `config_id` 各 500 行：`clip_w7_d4` / `spatial16_w8_d4` / `max_pool_w3_d5`（cache 开启）与 `inference_clip_w7_d4` / `inference_spatial16_w8_d4` / `inference_max_pool_w3_d5`（cache 关闭 ≡ pure inference）。

- **纯 cache (AlwaysSearchGate 等价)** = `config_id ∈ {clip_w7_d4, spatial16_w8_d4, max_pool_w3_d5}` 的 500 行，按 `(cfg, orig_init_state_idx)` 聚合成 "inference_ratio=0.0" 端点。
- **纯 inference (AlwaysSkipGate 等价)** = `config_id ∈ {inference_clip_w7_d4, inference_spatial16_w8_d4, inference_max_pool_w3_d5}` 的 500 行，聚合成 "inference_ratio=1.0" 端点。**注意 pure inference 也不是 100% success**（本地统计 `inference_clip_w7_d4=496/500`、`inference_spatial16_w8_d4=492/500`、`inference_max_pool_w3_d5=492/500`）；analysis 里端点按实际 success_rate 画，不假设 1.0。
- `exp/trajectory_deviation/data/gt/` 只是 Step 1b 失败子集 GT trajectory（非全 500 pure inference），**本实验不消费此目录**。

Join key：本实验 `ep_key = task_{task_id}/init_{orig_init_state_idx}`，与上述 baseline 行的 `(config_id without inference_ prefix, orig_init_state_idx)` 天然对齐。Plan 阶段不再产出新 baseline 数据。

## 3. 范围与语义

| 维度 | 取值 | 备注 |
|------|------|------|
| Episode 集合 | per-cfg 无差别，`libero_spatial` 全量 500 ep | 三 cfg 间 ep 相同；互不交叉影响 |
| Step1 baseline | 直接读 trajectory_deviation 已有产物 | 不重跑 |
| Trajectory history | Cache 和 inference cycle 都正常累积 | 由 orchestrator skip 路径保证（`src/openpi/cache/orchestrator.py:219-226` 同 `AlwaysSkipGate` 语义） |
| Pareto 分析 | 本 plan 交付基础图 | Task 4 下另起不做 |

## 4. 并行布局

硬件：**3 server × 每 server 1 client × 每 client 5 worker**（batch 数 = server 数；MuJoCo EGL 5-worker 上限沿用 Step 3 约束）。

**YAML 批次化**：所有 114 个最终 YAML（3 cfg × 38 gate 组合）**离线预生成并 git 追踪**，按 3 个 batch 切分（每 cfg 一个 batch）写入 `exp/random_periodic_gate/config/batch{1..3}/`，batch ↔ cfg 一一绑定：

| Batch | Cfg | YAML 数 | Server | Client Workers |
|-------|-----|--------|--------|---------------|
| batch1 | `clip_w7_d4` | 38 | host_1:8001 | 5 |
| batch2 | `spatial16_w8_d4` | 38 | host_2:8001 | 5 |
| batch3 | `max_pool_w3_d5` | 38 | host_3:8001 | 5 |

38 = 20 periodic + 18 random（见 §5.5 slug 命名规则）。3 batch 各 38 × 500 = 19,000 ep，3 server 完全并行，估时 ~6–10h / server（按每 ep 约 6s、5 worker 共享）。

**执行流**（每 server 独立，互不通信）：
1. `exp/random_periodic_gate/generate_batches.py` 离线产出 3 个 batch 目录下的全部 YAML，并 commit 到仓库（CI 可选 dry-run 校验所有 YAML 都能通过 `validate_cache_config`）；
2. 每个 server 对应一个 runner 进程，通过 `--batch-dir exp/random_periodic_gate/config/batchN/` 指定本次跑的 batch + `--host host_N --port 8001` 指定对端 server；
3. runner 内循环 iterate 目录下 YAML（字典序）：每个 YAML → `send_load_cache_config(server_url, path)` → 500 ep rollout（5 worker 并发）→ 下一个 YAML；
4. 3 个 batch 互相独立，可同时启动也可顺序启动；断点续跑粒度 `(yaml_basename, task_id, init_idx)`，state JSON 按 batch 独立存于 `exp/random_periodic_gate/data/batchN/run_state.json`。

## 5. 实现

### 5.1 新增 / 修改文件

| 文件 | 类型 | 描述 |
|------|------|------|
| `src/openpi/cache/components/gate.py` | 修改 | 新增 `RandomGate` / `PeriodicGate` 两个类；**`GateFunction` 协议保持不变**（依然 `on_episode_start(self) -> None` 无参、无 `episode_stats`），两个新 gate 也按无参 `on_episode_start` 实现 |
| `src/openpi/cache/config.py` | 修改 | `GateConfig` 加 4 个可选字段（`p_inference` / `seed` / `cache_len` / `inference_len`，默认 `None`）；`_GATE_TYPES` 加 `random` / `periodic`；`_build_gate` 加两个分支；`validate_cache_config` 校验参数范围 + 禁止错配字段（旧 3 种 gate_type 未带新字段必须通过） |
| `exp/random_periodic_gate/__init__.py` | 新建 | 模块 docstring |
| `exp/random_periodic_gate/generate_batches.py` | 新建 | 离线脚本：读 3 份 base YAML + 38 gate 组合 → 展开写入 `config/batch{1..3}/<slug>.yaml`；可选 `--validate` flag 逐个 `load_cache_config` 以 CI fail-loud |
| `exp/random_periodic_gate/run_gate_sweep.py` | 新建 | 主 runner（结构借鉴 `exp/trajectory_deviation/run_step3_per_cycle_policy.py`，但不注入 `__gate_decision__`、不依赖 deviate_score_json、不依赖任何扩展的 framework 接口）；`--batch-dir` 驱动 |
| `exp/random_periodic_gate/config/base_clip_w7_d4.yaml` | 新建 | Generator 输入：复制 `exp/trajectory_deviation/config/clip_w7_d4.yaml`，`gate.type: always_search` 占位使其独立可 validate；generator 覆盖 `checkpoints.cp1.gate` 子树后写到 `batch{1,2}/` |
| `exp/random_periodic_gate/config/base_spatial16_w8_d4.yaml` | 新建 | 同上，对应 batch3/4 |
| `exp/random_periodic_gate/config/base_max_pool_w3_d5.yaml` | 新建 | 同上，对应 batch5/6 |
| `exp/random_periodic_gate/config/batch{1..3}/<slug>.yaml` | 新建 | `generate_batches.py` 产出的 114 个最终 YAML（每 batch 38 份），git 追踪；runner 直接消费，不再运行时渲染；现有 `.gitignore` 规则 `exp/**/data/**` 只 ignore `data/`，config 下 YAML 天然 tracked，无需修改 `.gitignore` |
| `exp/random_periodic_gate/analysis/analyze_gate_sweep.py` | 新建 | JSONL → CSV 聚合 + Pareto / heatmap 绘图，附 baseline 端点 join 逻辑（读 `cache_eval_results.json`） |
| `tests/cache/components/test_gate.py` | 修改 | 补 `RandomGate` / `PeriodicGate` 单测；不触及现有 3 gate 的接口 |
| `tests/cache/test_config.py` | 修改 | 补 `gate.type=random` / `periodic` 参数校验 + dispatch；加旧 3 种 gate_type regression（不带新字段必须仍通过） |
| `tests/exp/test_run_gate_sweep.py` | 新建 | runner 单测（fake ws client，无 env 依赖） |
| `tests/exp/test_generate_batches.py` | 新建 | generator 单测：展开 38 组 → 按 cfg 分发到 3 batch 目录（每 batch 38 份）；每个产物 load 后 dict 等式（只差 gate 子树） |

**显式不改动（Round 2 owner 红线）**：
- `src/openpi/cache/orchestrator.py` — `on_episode_start` / `on_episode_end` 签名、调用链 L191-203 均保持原样；不引入 stats 收集路径。
- `src/openpi/cache/interceptor.py` — `on_episode_end(success) -> None` 保持不变。
- `src/openpi/serving/websocket_policy_server.py` — episode_end handler L220-225 ack 保持 `{"__ack__": "episode_end"}`，不加 `stats`。
- `packages/openpi-client/src/openpi_client/websocket_client_policy.py` — `episode_end(success)` 不读任何新字段，行为不变。
- `examples/libero/main.py` — 本实验不依赖也不生产 pure inference baseline，沿用现有产物。
- `exp/trajectory_deviation/**` — 仅作 baseline 数据源（`cache_eval_results.json`）读取，不写入、不重跑。

### 5.2 `GateConfig` 扩展

```python
# src/openpi/cache/config.py

@dataclass
class GateConfig:
    type: str = "always_search"
    # Only for type="random"
    p_inference: float | None = None
    seed: int | None = None
    # Only for type="periodic"
    cache_len: int | None = None
    inference_len: int | None = None
```

`_GATE_TYPES` 同步加 `random` / `periodic`。

`validate_cache_config` 里新增分支：
- `type=random` 必须 `0.0 <= p_inference <= 1.0` 且 `seed` 为非负整数；`cache_len` / `inference_len` 必须为 `None`。
- `type=periodic` 必须 `cache_len >= 1` 且 `inference_len >= 1`；`p_inference` / `seed` 必须为 `None`。
- 其他 3 种 legacy gate_type（`always_search` / `always_skip` / `client_controlled`）校验"新参数字段必须为 `None`"以防错配 YAML 静默忽略。

### 5.3 `RandomGate` / `PeriodicGate` 实现

**协议不变**（Round 2 owner 红线）：保持 `GateFunction.on_episode_start(self) -> None` 无参；不引入 `episode_stats()`；不触 orchestrator / interceptor / ws 协议。RandomGate / PeriodicGate 的签名与 `AlwaysSearchGate` / `AlwaysSkipGate` / `ClientControlledGate` 逐字段对齐。

```python
# src/openpi/cache/components/gate.py
import numpy as np


class RandomGate:
    """Server-side random-skip gate.

    Each gate call samples an independent Bernoulli draw: with probability
    ``p_inference`` the gate returns False (skip cache -> fall-through to
    real inference); otherwise returns True (search cache).

    Reproducibility (intentional scope, per owner G1 Round 2):
      - Constructed with ``seed``; at every ``on_episode_start()`` the
        internal RNG is re-seeded with ``seed * 10_000 + ep_idx`` where
        ``ep_idx`` is a per-instance counter incremented at each episode
        boundary.
      - Since cache connections each get their own gate instance via
        ``build_per_connection_components``, and worker/episode assignment
        is determined by the runner queue, the stream is deterministic
        **per-connection** (same worker, same (seed, N-th episode) => same
        stream). It is NOT deterministic across worker reassignment or
        resume — only the aggregate stochasticity across the 500 ep full
        sweep is meaningful, which is what this experiment needs.

    Coupling:
      - UNAFFECTED BY: request_context, cached_data.
      - CONSUMED BY: CacheOrchestrator.check() — same skip semantics as
        AlwaysSkipGate.
    """

    def __init__(self, p_inference: float, seed: int) -> None:
        if not (0.0 <= p_inference <= 1.0):
            raise ValueError(f"RandomGate p_inference must be in [0, 1], got {p_inference}")
        if seed < 0:
            raise ValueError(f"RandomGate seed must be >= 0, got {seed}")
        self._p_inference = float(p_inference)
        self._seed = int(seed)
        self._ep_idx = 0
        self._rng = np.random.default_rng(self._seed)

    def __call__(self, checkpoint_id, cached_data, request_context=None) -> bool:
        # skip (i.e. run inference) with probability p_inference
        return self._rng.random() >= self._p_inference

    def on_episode_start(self) -> None:
        self._ep_idx += 1
        self._rng = np.random.default_rng(self._seed * 10_000 + self._ep_idx)

    def record_action(self, action_chunk) -> None:
        """No-op. Signature matches GateFunction protocol."""


class PeriodicGate:
    """Server-side periodic-skip gate.

    Each episode begins with ``cache_len`` cache searches, followed by
    ``inference_len`` forced skips; the cycle repeats until the episode
    ends. ``on_episode_start`` resets the counter so every episode
    starts with a cache block.

    Cost metric:
      - The decision at cycle ``c`` is exactly ``c % (cache_len +
        inference_len) < cache_len``. Given ``total_cycles`` (known to
        the runner), the runner derives ``num_inference_cycles`` by the
        same closed-form formula, so no server-side stat transport is
        needed.

    Coupling:
      - UNAFFECTED BY: request_context, cached_data.
      - CONSUMED BY: CacheOrchestrator.check() — same skip semantics as
        AlwaysSkipGate.
    """

    def __init__(self, cache_len: int, inference_len: int) -> None:
        if cache_len < 1 or inference_len < 1:
            raise ValueError(
                f"PeriodicGate requires cache_len >= 1 and inference_len >= 1, "
                f"got cache_len={cache_len}, inference_len={inference_len}"
            )
        self._cache_len = int(cache_len)
        self._inference_len = int(inference_len)
        self._period = self._cache_len + self._inference_len
        self._counter = 0

    def __call__(self, checkpoint_id, cached_data, request_context=None) -> bool:
        pos = self._counter % self._period
        self._counter += 1
        return pos < self._cache_len   # True first k positions, False next n

    def on_episode_start(self) -> None:
        self._counter = 0

    def record_action(self, action_chunk) -> None:
        """No-op. Signature matches GateFunction protocol."""
```

### 5.4 `_build_gate` 新分支

```python
# src/openpi/cache/config.py::_build_gate (after client_controlled branch)

    if cfg.type == "random":
        from openpi.cache.components.gate import RandomGate
        return RandomGate(p_inference=cfg.p_inference, seed=cfg.seed)
    if cfg.type == "periodic":
        from openpi.cache.components.gate import PeriodicGate
        return PeriodicGate(cache_len=cfg.cache_len, inference_len=cfg.inference_len)
```

### 5.5 YAML 离线批次生成（无运行时渲染）

**Base 模板 (`exp/random_periodic_gate/config/base_<cfg>.yaml`)** 与 `exp/trajectory_deviation/config/<cfg>.yaml` 逐字段相同，除了：
- `checkpoints.cp1.gate.type: always_search` 占位让 base 文件独立通过 `validate_cache_config`；
- `checkpoints.cp1.judge.type: always_hit`（保持 Step 3 约定）；
- `write_policy.type: never`（不污染 artifact）；
- 注释写明这是"generator 输入，不直接 serve"。

**Generator (`generate_batches.py`)**：纯 Python（`yaml.safe_load` + 覆盖 `checkpoints.cp1.gate` 子树 + `yaml.safe_dump`），输入 3 份 base YAML + 38 个 gate 组合，输出到 `config/batch{1..3}/<slug>.yaml`（每 cfg 独占一个 batch 目录）。

Slug 命名规则：
- `random_p0p20_s1` → `{"p_inference": 0.20, "seed": 1}`（小数点用 `p` 代替）
- `periodic_k5_n2` → `{"cache_len": 5, "inference_len": 2}`

Batch 切分（每 cfg 的 38 个 slug 按字典序排序后切中点）：
- batch1 = clip 的前 19 slug、batch2 = clip 的后 19；
- batch3/4 = spatial16；
- batch5/6 = max_pool。

Generator 的 `--validate` flag 在写盘前用 `openpi.cache.config.load_cache_config` 依次 load 每份 YAML，确保全部通过 schema 校验；在 CI 上跑一次即可，不进入运行时路径。

**生成的 YAML git 追踪**：`.gitignore` 的 `exp/**/data/**` 规则只作用于 `data/`，`config/` 子树天然 tracked，不需额外白名单；reproducibility + code review 友好。

### 5.6 Runner 骨架

```python
# exp/random_periodic_gate/run_gate_sweep.py

"""Random & Periodic gate sweep runner (batch-dir driven).

Iterates every YAML under --batch-dir, switches the server's cache bundle
via send_load_cache_config, runs the full LIBERO suite with 5-worker
concurrency, and appends per-ep JSONL rows.

Parallelism: 5 workers (MuJoCo EGL cap).
State: exp/random_periodic_gate/data/<batch>/run_state.json
Results: exp/random_periodic_gate/data/<batch>/results.jsonl
"""

def enumerate_full_500(task_suite_name: str) -> list[tuple[int, int]]:
    """Canonical (task_id, init_idx) enumeration for the full suite.

    libero_spatial = 10 tasks; per-task init states come from the LIBERO
    default set ``task_suite.get_task_init_states(task_id)``, which is
    exactly what ``examples/libero/main.py::_get_libero_env`` and Step 1a
    ``exp/trajectory_deviation`` rollouts consume. init_idx is the index
    into that default array and MATCHES ``orig_init_state_idx`` in
    ``exp/trajectory_deviation/data/cache_eval_results.json`` — our
    baseline join key.
    """
    from libero.libero.benchmark import get_benchmark_dict
    suite = get_benchmark_dict()[task_suite_name]()
    out = []
    for task_id in range(suite.n_tasks):
        init_states = suite.get_task_init_states(task_id)   # ndarray (N, D)
        for init_idx in range(len(init_states)):
            out.append((task_id, init_idx))
    return out   # libero_spatial: 10 * 50 = 500


def main():
    args = parse_args()
    yaml_paths = sorted(Path(args.batch_dir).glob("*.yaml"))  # 19 per batch
    assert yaml_paths, f"no YAML in {args.batch_dir}"
    episodes = enumerate_full_500(args.task_suite_name)
    state = BaseRunState(state_path)        # unit = (yaml_basename, task_id, init_idx)
    jsonl = JsonlAppender(results_jsonl)
    prev_version = None

    for yaml_path in yaml_paths:
        cfg_meta = parse_slug(yaml_path.stem)    # cfg + gate_type + params
        new_version = send_load_cache_config(args.server_url, yaml_path)
        if prev_version is not None:
            assert new_version > prev_version    # silently-idle server guard
        prev_version = new_version
        units = [(yaml_path.name, tid, iid) for (tid, iid) in episodes]
        pending = [u for u in units if not state.is_done(u)]
        run_parallel(pending, cfg_meta, args, state, jsonl)  # 5 worker, standard LIBERO obs
```

**Episode enumeration canonical mapping** (bound across runner, JSONL, baseline join):

| Field | Definition | Matches |
|------|-----------|---------|
| `task_id` | 0..`n_tasks-1` from LIBERO benchmark | `cache_eval_results.json.task_id` |
| `init_idx` | Index into `task_suite.get_task_init_states(task_id)` | `cache_eval_results.json.orig_init_state_idx` |
| `ep_key` | f-string `f"task_{task_id}/init_{init_idx}"` | (derived) |
| JSONL `ep_key` 字段 | f-string `f"task_{task_id}/init_{init_idx}"`；只在 runner 本地 JSONL / state / baseline join 中使用，不传入 server 端任何 gate 逻辑 | RandomGate 的 per-connection `ep_idx` 递增由 `CacheOrchestrator.on_episode_start()` 无参钩子触发，与 `ep_key` 字面值无关 |
| Env init | `env.set_init_state(init_states[init_idx])`（与 `examples/libero/main.py` 一致） | |

**Per-unit loop** 严格对齐 Step 3（`run_step3_per_cycle_policy.py`），关键差异只有一处：
- **不**注入 `__gate_decision__` — obs 是原汁原味 LIBERO obs；
- `conn.episode_end(success=success)` 返回值**不读任何新字段**（ack 仍然只有 `__ack__`）。

**Inference cycle 统计 — runner 端后处理（唯一路径，Round 2 owner 约定）**：

| Gate | `num_inference_cycles` 来源 | JSONL 标签 |
|------|---------------------------|-----------|
| PeriodicGate | 闭式公式 `sum(1 for i in range(total_cycles) if i % (k+n) >= k)` | `inference_ratio_source="derived"` |
| RandomGate | `round(total_cycles * p_inference)` (estimated) | `inference_ratio_source="expected"` + 另存 `num_inference_cycles_estimated` 字段 |

这两条路径完全在 runner 进程里算，server / framework 零改动。

### 5.7 产物 Schema

- Per-ep JSONL（见 §2.3；按 batch 存 `data/batchN/results.jsonl`）
- Per-grid-point CSV aggregate（`analysis/analyze_gate_sweep.py` 合并 3 个 batch 产出）：

```csv
cfg,gate_type,param_slug,p_inference,seed,cache_len,inference_len,episodes,success_rate,mean_inference_ratio,inference_ratio_source
clip_w7_d4,periodic,k5_n2,,,5,2,500,0.42,0.286,derived
clip_w7_d4,random,p0p20_s0,0.20,0,,,500,0.33,0.20,expected
clip_w7_d4,random,p0p20_s1,0.20,1,,,500,0.34,0.20,expected
...
```

- Pareto 图：`analysis/pareto_<cfg>.png`（x=inference_ratio, y=success_rate；PeriodicGate = derived cost、RandomGate = expected cost；左/右端点从 `cache_eval_results.json` join 得到）。
- Heatmap：`analysis/heatmap_<cfg>_periodic.png`（x=inference_len, y=cache_len, color=success_rate）。

## 6. 测试策略

### 6.1 单元测试（`tests/cache/components/test_gate.py`）

- `RandomGate`:
  - 构造器参数范围（p ∈ [0,1], seed ≥ 0）
  - 固定 seed + 固定 ep_idx（调 `on_episode_start` N 次后 `__call__`）两次构造产物抽样序列 byte-equal（per-connection 级确定性，非跨 worker 级）
  - `on_episode_start` 连续调两次后 RNG 状态改变：第 1 个 episode 的抽样序列 ≠ 第 2 个 episode 的抽样序列
  - `p_inference=0.0` 永远 search；`p_inference=1.0` 永远 skip
  - 忽略 `request_context` / `cached_data`
  - 无 `episode_stats` 方法、签名与 `AlwaysSkipGate` 一致（`on_episode_start(self) -> None`）
- `PeriodicGate`:
  - 参数范围（cache_len ≥ 1, inference_len ≥ 1）
  - 一个周期内前 k 个 True、后 n 个 False
  - `on_episode_start` 后计数 reset，新 episode 从 cache 段开始
  - `(k=1, n=1)` duty cycle 锁定
  - 忽略 `request_context` / `cached_data`
  - 无 `episode_stats` 方法、签名与 `AlwaysSkipGate` 一致

### 6.2 Config 测试（`tests/cache/test_config.py`）

- `gate.type=random` + `p_inference=0.3, seed=0` 通过 validate；缺 `seed` 报错；`p_inference=1.5` 报错；混入 `cache_len` 报错。
- `gate.type=periodic` + `cache_len=5, inference_len=2` 通过；缺字段 / 负值报错；混入 `p_inference` 报错。
- `gate.type=always_search` + 携带 `p_inference` 报错（保护旧配置不被误贴参数）。
- `_build_gate` 对新两种 type dispatch 返回正确实例。

### 6.3 Runner 测试（`tests/exp/test_run_gate_sweep.py`）

- Batch-dir 驱动：fake batch 目录下放 2 个 YAML → runner 按字典序迭代并调 `send_load_cache_config` 两次；bundle version 严格递增 assert。
- Episode 枚举：`enumerate_full_500("libero_spatial")` 长度 500、`(task_id, init_idx)` 覆盖 0..9 × 0..49；`ep_key` 格式 `task_{tid}/init_{iid}` 与 baseline join key 对齐的 golden check。
- **PeriodicGate 后处理公式**：runner 对 `gate_type=periodic` 的 unit 计算 `num_inference_cycles = sum(1 for i in range(total_cycles) if i % (k+n) >= k)`；test 固定 `(k, n, total_cycles)` 抽几组边界值（整周期、不整周期、total_cycles < k）assert 公式结果。
- **RandomGate 估计值**：runner 对 `gate_type=random` 的 unit 计算 `num_inference_cycles_estimated = round(total_cycles * p_inference)` 并写 `inference_ratio_source="expected"`；test 锁定字段存在、值正确、不混用 derived 分支。
- 断点续跑：unit = `(yaml_basename, task_id, init_idx)` 已 done → 跳过；mid-batch crash → 下次 resume 从对应 YAML 继续。
- `conn.episode_end(success=...)` 的返回值**不读任何字段**（ack 未扩），test 模拟旧 server ack `{"__ack__": "episode_end"}` 仍能跑通。

### 6.4 Generator 测试（`tests/exp/test_generate_batches.py`）

- 38 组 slug 展开：20 periodic + 18 random（6 p_inference × 3 seed）；slug 命名唯一、稳定字典序。
- 3-batch 布局：每 cfg 独占一个 batch 目录，各 batch 恰好 38 份。
- 每份产物 dict 等式：`yaml.safe_load(rendered) == yaml.safe_load(base)` 仅 `checkpoints.cp1.gate` 子树差异；rendered gate 子树与 slug 参数对齐（p_inference / seed / cache_len / inference_len 逐字段 assert）。
- `--validate` flag：所有 114 份 YAML 都能 `openpi.cache.config.load_cache_config` 成功。
- 幂等性：连跑两次 `generate_batches.py` 所有输出文件 byte-identical（纯 `yaml.safe_dump` 输出稳定）。

### 6.5 Smoke（Verify 阶段手跑，不入 pytest 默认）

- batch1 前 2 个 YAML × 2 ep 的 dry-run，验证：
  - server bundle 切换成功（bundle version 递增）；
  - JSONL 产出字段完整（含 `inference_ratio_source`）；
  - pareto / heatmap 脚本能读 dry-run 产物画图。

## 7. 并发模型与资源

沿用 Step 3 模式（`exp/trajectory_deviation/run_step3_per_cycle_policy.py:540-564`）：
- per-worker 独立 `WebsocketClientPolicy` + 独立 `OffScreenRenderEnv`；
- `init_lock` 串行化 env / ws 构造；
- `_JSONL_LOCK` 保护 results.jsonl 原子 append；
- `_MAX_WORKERS_CAP=5`。
- 切 grid_point 时 worker 全部 join → send_load_cache_config → 下一轮 worker 启动。这样避免"一半 worker 用旧 bundle、一半用新 bundle"。

## 8. 风险清单

| 风险 | 影响 | 缓解 |
|------|------|------|
| `send_load_cache_config` 需要 server 在 `--concurrent` 模式（`websocket_policy_server.py:263-300`），单连接 server 拒绝重载 | runner 起不来 | 启动 server 时加 `--concurrent`，runner 启动时显式 load 一次并 assert bundle version 递增 |
| RandomGate 的 `inference_ratio` 是 expected 值而非实测 | analysis 里 cost 轴 x 坐标精度低 | 本实验 owner 指示接受 expected 值（不扩 framework 换取 exact count）；JSONL / CSV 显式用 `inference_ratio_source="expected"` 字段标注，aggregate 曲线在 analysis 里以实线+误差条表达；PeriodicGate 精确值仍用 derived 公式保证 |
| RandomGate 跨 worker / resume 复现性有限 | 同一 (seed, cfg, ep_key) 在不同 worker / 不同次 resume 抽样序列不一致 | 本实验 owner 指示接受 per-connection 级确定性：`seed * 10000 + ep_idx` 派生 RNG；跨 worker 抽样序列差异被 500 ep × 3 seed aggregate 稀释，success rate 方差留给 analysis 里的 error bar 表达 |
| `validate_cache_config` 误伤旧 `always_search` / `always_skip` / `client_controlled` YAML | 现有 YAML 全部挂掉 | 只在"参数非 None 且 type 不匹配"时 raise；None 默认保持兼容；`tests/cache/test_config.py` 加 regression：旧 3 种 gate_type YAML 不带新参数字段必须通过 |
| 114 份 YAML 提交进 git 引入大量文件 | 仓库噪声 | generator 脚本幂等 (`generate_batches.py --validate`)，可多次重跑 byte-identical；评审时只用看 generator 逻辑 + 抽 1-2 个 rendered 做 golden check |
| 57,000 ep 跑太久 | 实验周期长 | 3 server × 5 worker 完全并行（3 batch ↔ 3 server 绑定，batch 间互不 block），单 batch ~6-10h；初版先跑 batch1 冒烟 |
| baseline 端点读错 `cache_eval_results.json` | analysis 不完整或混 index 空间 | 固定从 `exp/trajectory_deviation/data/cache_eval_results.json` 读；`ep_key` join 用 `(config_id.removeprefix("inference_"), orig_init_state_idx)` 显式映射；analysis 脚本 fail-loud 提示具体路径与缺失 cfg |
| LIBERO env seed 默认不稳定 | 跨 run 结果不完全复现 | env seed 不稳定是 `trajectory_deviation` F1 follow-up；本 plan 不尝试解决，在 analysis 里写 caveat |

## 9. 验收条件

- 3 cfg × 38 grid_point × 500 ep = 57,000 ep 跑完（6 batch 断点续跑，可多轮）；
- 聚合 CSV / Pareto 图 / heatmap 生成；
- `uv run pytest tests/cache/ tests/exp/test_run_gate_sweep.py tests/exp/test_generate_batches.py -q` 全绿（本 plan **不**扩 framework 协议，测试范围仅限 cache 组件 + exp runner/generator）；
- **PeriodicGate 公式契约**：runner 后处理从 `(k, n, total_cycles)` 推得 `num_inference_cycles`；test 锁定公式 `sum(1 for i in range(total_cycles) if i % (k+n) >= k)` 对若干边界值正确（§6.3）；
- **RandomGate estimated 契约**：runner 对 `gate_type=random` 写 `inference_ratio_source="expected"` + `num_inference_cycles_estimated = round(total_cycles * p_inference)`；不做 exact-count 断言（§6.3）；
- **Generator 幂等性**：`generate_batches.py` 连跑两次产物 byte-identical；`--validate` 全部 YAML 过 `load_cache_config`；
- **Baseline 端点 join**：analysis 脚本能从 `exp/trajectory_deviation/data/cache_eval_results.json` 读出 6 个 config_id 的各 500 行、成功 join 到本实验的 500 ep 集合、画出 Pareto 图的左端（cache）与右端（pure inference）两个 anchor 点（不重跑，只 join）。

## 10. 工作量估计

| 阶段 | 预计 |
|------|------|
| 实现（gate 2 类 + config 扩展 + generator + runner + analysis）| 1.5–2 天 |
| 单元 / config / generator 测试 | 0.5 天 |
| Smoke (batch1 × 2 YAML × 2 ep) | 0.5 天 |
| 全量跑（6 batch ↔ 6 server 并行） | ~3–5h / server 实时，完全并行 |
| analysis / pareto 出图 | 0.5 天 |
| **合计** | ~3.5 天工程 + ~半天机器时间（全并行下） |

## 11. Gate 策略（WA §2）

- **G1**：本文件作为 Plan 输出，APPROVED 后执行 Post-G1 polish（execution_authority §3.1）；审阅重点：
  1. framework scope 是否严格控制在 `components/gate.py` + `config.py`（Round 2 owner 红线）；
  2. runner 端 PeriodicGate 后处理公式 + RandomGate expected 值是否被 test 锁定；
  3. `GateConfig` 扩参数字段后 `validate_cache_config` 对旧 YAML 是否产生误伤；
  4. 6 batch 布局 + generator 幂等 + 114 YAML git 追踪是否都在验收条件覆盖范围。
- **G2**：按"gate 类 + config + generator + runner + analysis"分 4–5 个 commit 分别过 G2。
- **Verify**：`uv run pytest tests/cache/ tests/exp/` 全绿 + §6.5 smoke（batch1 前 2 个 YAML × 2 ep）。

## 12. 实现注意事项（非阻塞）

- Base 模板 YAML 的 `gate.type: always_search` 占位 — 让 base 文件独立通过 `validate_cache_config`；generator 把整个 `checkpoints.cp1.gate` 子树替换掉（dict 对比测试锁定，见 §6.3）。
- `RandomGate` 的 `np.random.default_rng(seed)` 依赖 numpy PCG64 默认实现，numpy ≥ 1.17 跨版本稳定；test 锁定 PCG64 bit generator。
- `exp/random_periodic_gate/data/**` 被现有 `.gitignore` 规则 `exp/**/data/**` 默认忽略；`config/` 子树（含 `batch*/*.yaml`）天然 tracked，不需在 `.gitignore` 加白名单。
- PeriodicGate 端点 `cache_len=0` 或 `inference_len=0` 本 gate 不支持（validate raise）；pure search 端点由 `AlwaysSearchGate` 行为覆盖、pure skip 端点由 `AlwaysSkipGate` 行为覆盖；两者 baseline 从 `cache_eval_results.json` 读而非重跑（§2.4）。
- Plan 后续若要扩 `libero_10` / `libero_goal`，只需新增 3 份 base YAML + 重跑 `generate_batches.py` + 新增数据目录；runner CLI 已暴露 `--task-suite-name`。

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-04-24 12:19 CDT

- [Blocking] [Concern] `--resume` is parsed but never passed into `BaseRunState.run()` / `parallel_run()` — reasoning: `exp/random_periodic_gate/run_gate_sweep.py` defines `--resume` at CLI parsing, but `main()` calls `runner.parallel_run(num_workers=...)` and `runner.run()` without `resume=args.resume`. `BaseRunState` only loads an existing state file when `resume=True`; otherwise it rebuilds units and overwrites the state snapshot. This breaks the approved plan's required batch/yaml resume behavior. Independent probe with a monkeypatched runner observed `calls == [('run', False)]` even when invoking `main([... '--resume'])`. Fix both serial and parallel branches and add a runner test that fails before the fix.
- [Blocking] [Concern] Gate parameter validation accepts fractional integer fields and the constructors silently truncate them — reasoning: G1 §5.2 specifies `seed` as a non-negative integer and `cache_len` / `inference_len` as integer lengths. Current validation only checks `seed < 0`, `cache_len < 1`, and `inference_len < 1`; `GateConfig(type='random', seed=0.5)` and `GateConfig(type='periodic', cache_len=1.5, inference_len=2)` pass validation. `_build_gate()` then calls `RandomGate(... seed=0.5)` / `PeriodicGate(cache_len=1.5, ...)`, whose constructors cast with `int(...)`, silently changing the requested config. Reject non-`int` values in `validate_cache_config()` (and preferably constructor-level guards too) and add regression tests for fractional seed/cache_len/inference_len.
- [Non-blocking] [Suggestion] Align the `.gitignore` / artifact-layout wording with the implemented diff — reasoning: the polished plan lists `.gitignore` as a modified file and says to add a whitelist for `exp/random_periodic_gate/config/batch*/*.yaml`, but the current diff does not modify `.gitignore` or `docs/experiments/artifact_layout.md`. Since `config/` YAMLs are tracked without a whitelist today, this does not block functionality; either implement the promised docs/ignore update or remove that claim from the plan during polish.

Advisory checks run by reviewer:
- `PYTHONPATH=. uv run pytest tests/cache/ tests/exp/test_run_gate_sweep.py tests/exp/test_generate_batches.py -q` — passed (`460 passed, 4 skipped`).
- `uv run python -m exp.random_periodic_gate.generate_batches --validate` — passed and validated 114 rendered YAML files.
- `find exp/random_periodic_gate/config -path '*/batch*/*.yaml' | wc -l` — 114; each `batch{1..6}` contains 19 YAMLs.

### G2 Round 1 — Executor — 2026-04-24

- Accepted (Blocking #1 `--resume` 未透传) — `exp/random_periodic_gate/run_gate_sweep.py::main()` 同时修两条路径：`runner.parallel_run(num_workers=..., resume=bool(args.resume))` 与 `runner.run(resume=bool(args.resume))`。`BaseRunState.run/parallel_run` 都支持 `resume=False` 作默认、`resume=True` 保留 state 文件，这样与 plan §4 "断点续跑粒度 `(yaml_basename, task_id, init_idx)`" 一致。新增 3 条 regression `tests/exp/test_run_gate_sweep.py::test_main_propagates_resume_true_to_serial_run` / `test_main_defaults_resume_false_to_serial_run` / `test_main_propagates_resume_true_to_parallel_run`，分别锁定 serial+`--resume`、serial 缺省、parallel+`--resume` 三种 CLI → runner kwargs 的透传契约（bundle version 递增 stub 内置于 test fixture）。
- Accepted (Blocking #2 整数字段被 silently truncate) — 二道防线：
  1. `src/openpi/cache/config.py::validate_cache_config` 里对 `seed` / `cache_len` / `inference_len` 改用 strict-int 判定（`isinstance(v, int) and not isinstance(v, bool)`），`p_inference` 用 `isinstance(v, (int, float)) and not isinstance(v, bool)`；non-int / bool / str / float 都在 config 层先失败。
  2. `src/openpi/cache/components/gate.py::RandomGate.__init__` / `PeriodicGate.__init__` 也加同等 type guard，`raise TypeError(...)` 防任何 YAML 旁路（脚本 / 测试 / 直接 Python 调用）触发 silent truncation。
  新增 5 条 config 层 regression（fractional seed / bool seed / str p_inference / fractional cache_len / fractional inference_len / bool cache_len）+ 5 条 ctor 层 regression（同类型错误）。
- Accepted (Non-blocking Suggestion `.gitignore` 漂移) — 采纳 "drop if unnecessary" 路径：实测 `git check-ignore exp/random_periodic_gate/config/batch1/periodic_k10_n1.yaml` 返回 `rc=1`（不在 ignore 规则内），证明 `exp/**/data/**` 只作用于 `data/` 子路径，`config/` 下 YAML 天然 tracked。修 plan body 三处漂移描述：§5.1 touched-file 表里原 `.gitignore` 行删除并把不改 `.gitignore` 说明合并进 `config/batch{1..6}/<slug>.yaml` 行；§5.5 把 "exception 白名单" 改为 "`config/` 子树天然 tracked，不需额外白名单"；§12 非阻塞注意项同步。实际代码层面未改 `.gitignore`，与 polished plan body 一致。

Advisory checks 本轮补跑：
- `uv run python -m pytest tests/cache/components/test_gate.py tests/cache/test_config.py tests/exp/test_run_gate_sweep.py tests/exp/test_generate_batches.py -q` — `164 passed`。
- `uv run python -m pytest tests/ -q` 全仓回归 — `730 passed, 5 skipped`。
- `uv run python -m exp.random_periodic_gate.generate_batches --validate` — 114 份 YAML 全部通过 schema 校验（代码改动不影响 generator 产物）。

3 条 reviewer 意见全部 accepted，再次进入 G2 gate 等待复审。

### G2 Round 2 — Reviewer — APPROVED — 2026-04-24 12:29 CDT

- [Pass] Round 1 Blocking #1 is resolved — reasoning: `exp/random_periodic_gate/run_gate_sweep.py::main()` now passes `resume=bool(args.resume)` into both `runner.run(...)` and `runner.parallel_run(...)`. The new runner tests cover serial `--resume`, serial default, and parallel `--resume`; reviewer also reran the previous monkeypatched probe and observed `calls == [('run', True)]`.
- [Pass] Round 1 Blocking #2 is resolved — reasoning: `validate_cache_config()` now rejects non-strict integer `seed` / `cache_len` / `inference_len` values and non-numeric / bool `p_inference`; `RandomGate` and `PeriodicGate` constructors add the same defense-in-depth checks so direct Python calls cannot silently truncate floats. New config and constructor regression tests cover fractional and bool inputs.
- [Pass] The prior non-blocking `.gitignore` / artifact-layout drift is resolved in the plan body — reasoning: the plan no longer lists `.gitignore` as a touched file and now states that `config/` YAMLs are naturally tracked because the existing ignore rule only covers `exp/**/data/**`.
- [Pass] Scope remains consistent with G1 — reasoning: no diff touches orchestrator, interceptor, websocket server, websocket client, `examples/libero`, or `exp/trajectory_deviation`; the implementation remains limited to cache gate/config plus the new experiment package and tests.

Reviewer checks:
- `PYTHONPATH=. uv run pytest tests/cache/ tests/exp/test_run_gate_sweep.py tests/exp/test_generate_batches.py -q` — passed (`474 passed, 4 skipped`).
- Independent `--resume` probe — passed (`run(resume=True)` observed).
- Strict gate validation probe — passed for bad config values and direct constructor calls.
- `.venv/bin/python -m exp.random_periodic_gate.generate_batches --validate` — passed; all 114 generated YAMLs load through `load_cache_config`.
- `git diff --cached --check` — passed.

G2 code approved.
