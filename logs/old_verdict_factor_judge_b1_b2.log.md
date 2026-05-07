---
status: Plan
authority_in_effect: Execution
parent_plan: logs/verdict_factor_judge.log.md   # B0 已合入，此处不重述设计
relates_to:
  - docs/architecture/cache_system.md §5.6 §5.11 §5.12
  - docs/cache/verdict_factor_judge.md
  - exp/common/build_in_memory_cache_artifact.py
  - exp/common/build_clip_cache_artifact.py
  - exp/common/build_llm_layer_matrix.py
created: 2026-04-26
---

# Verdict Factor Judge — B1+B2 合并实施计划

> **本 plan 不重述设计**。B0 plan (`logs/verdict_factor_judge.log.md`) 是设计冻结源；本 plan 只负责"哪些占位 body 在哪个文件被填实、什么顺序、用什么测试覆盖、文档与 artifact 工具同步交付什么"。

---

## 1. 目标与非目标

### 1.1 目标

合并交付 B1（在线算法批）+ B2（离线写入批），让 verdict-factor judge 第一次端到端可跑，并让 build pkl 工具自动写入 `payload.factors` + `library_stats`。

| 子目标 | 交付物 |
|---|---|
| F1a-A / F1a-T `extract` body | `runtime_continuity.py` 内 `_RuntimeContinuityBase.extract`（thin subclass 共用） |
| F2 `extract` body | `consensus.py` 内 `TopKActionConsensus.extract` |
| F1b-A / F1b-T `extract` + `compute_for_episode` body | `source_window.py` 内 `_SourceWindowSmoothnessBase` 两个方法 |
| `LibraryStats.compute_from_entries` body | `factors/base.py` |
| Composer 算法 body（3 个） | `composers/__init__.py` `WeightedSumComposer` / `AndGateComposer` / `OrGateComposer` `compose` |
| `PercentileRollingNormalizer.__call__` body | `normalizers/__init__.py` |
| Orchestrator B1 改造（view + history 注入、winner fetch rewire、`_state_history`、anchor CP、on_task_end 修复） | `orchestrator.py` |
| Orchestrator B2 改造（`_build_entry_chain` 调 OfflineWriter） | `orchestrator.py` |
| Config B1 改造（`_JUDGE_TYPES` 接受 `"composite"` + 6 项 composite-specific 校验激活 + 第 7 项 state-library 校验 + `min_required_top_k` 反向喂回 strategy） | `config.py` |
| Backend B2 改造（`load_artifact` `library_stats` 加载与 fallback） | `in_memory_backend.py` |
| Artifact build B2 接线（三脚本 + 共用 helper） | `exp/common/factor_postprocess.py`（新建）+ `build_*.py` 各加 ~3 行 |
| 文档同步 | `docs/architecture/cache_system.md` §5.6 / §5.11 / §5.12 由"骨架"升为"算法已落地"；`docs/cache/verdict_factor_judge.md` + `.en.md` 各章 status 由 B1/B2 → 实现；index 同步 |

### 1.2 非目标

- **不**做 21-window × 6-pkl 的 calibration / artifact 重建（属于实验级活动，B2 落地后再独立做实验 plan）
- **不**实现 `dirvar / path / freq / autocorr` 4 个扩展描述子（B0 plan §2.8.2 已声明：列入 `descriptors` 时 config 阶段 `raise NotImplementedError`）
- **不**实现 schema-aware `dim_groups`（B0 plan §2.8.7 已声明 `raise NotImplementedError`）
- **不**实现非 `TRAJECTORY` 的 `ForkPolicy`（B0 plan §2.3.2 已声明 `raise NotImplementedError`）
- **不**改 `SearchStrategy.search()` 签名 / `QuerySpec` / `Backend` ABC / 老 Judge 行为 / 老 YAML 兼容 — 这些在 B0 已经做过 wiring，B1+B2 只是消费它们
- **不**做 `force_legacy_path()` 之类的灰度逃生开关（B0 plan §2.6 没有此条；composite 是 opt-in YAML，老 YAML 自然零影响）
- **不**让 Qdrant 等非 in-memory backend 支持 F1a/F1b（在 config 校验阶段 fail-fast）

### 1.3 与 B0 plan 的边界

B0 plan 的 §2.x（设计）/ §3（因子 ↔ 协议矩阵）/ §8.1（新建文件骨架）是**设计冻结**：本 plan 不修改、不质疑，只在 §3-§4 列哪一段被填实。

B0 plan 的 §8.2.3-§8.2.7（修改现有文件 wiring）已分别标注 B0 / B1 / B2 范围：本 plan 只针对其中 B1 / B2 部分。

---

## 2. 前置：B0 已交付契约速览

为让读者不读 B0 plan 也能理解本 plan，列出 B1/B2 直接消费的 B0 契约：

| B0 已 ship 项 | 物理位置 | B1/B2 怎么消费 |
|---|---|---|
| `OnlineExtractor` Protocol（含 `extract(results, view, history, cached_data)` + `descriptor_orientations` + `required_top_k` + `requires_*` 类标志） | `factors/base.py` | B1 填 `extract` body |
| `OfflineWriter` Protocol（`compute_for_episode(entries, library_stats)` + `required_payload_fields()`） | `factors/base.py` | B2 填 `compute_for_episode` body |
| `LibraryStats` dataclass（`compute_from_entries(entries) → LibraryStats`） | `factors/base.py` | B2 填 `compute_from_entries` body |
| `HistoryView` dataclass（`actions: list[Tensor]`, `states: list[Tensor]`） | `factors/base.py` | B1 在 `extract` 内读 |
| `_RuntimeContinuityBase` / `_SourceWindowSmoothnessBase` / `TopKActionConsensus` 类 + thin subclass + `@register` + `describe()` + capability flags | `runtime_continuity.py` / `source_window.py` / `consensus.py` | B1/B2 填对应 method body |
| `_DESCRIPTOR_ORIENTATIONS` table（jerk/dir/curv_radius/cum_disp）+ `_normalize_windows` helper | `source_window.py` | B1 (F1a) / B2 (F1b) 算法都消费 |
| `Composer` Protocol + `WeightedSumComposer` / `AndGateComposer` / `OrGateComposer` 类骨架 + `bind_orientations` + `compose` 占位 | `composers/__init__.py` | B1 填 `compose` body |
| `Normalizer` Protocol + `PercentileRollingNormalizer` 类骨架 + `bind_keys` + `__call__` 占位 | `normalizers/__init__.py` | B1 填 `__call__` body |
| `CompositeJudge` 类（`__init__` 收集 orientations、bind、key contract assert、cold-start NaN sentinel） | `judge.py` | B1 已可消费（B0 ship 时已含完整 `__call__`，仅依赖 extractor body 填实） |
| `PayloadView` Protocol + `StoragePayloadView` 实现（含 per-check memo + 无-fork chain walk） | `payload_view.py` | B1 在 Orchestrator `check()` 构造并注入 Judge |
| `CacheStorage.fetch_entry(id)` + `library_stats` duck-typed facade；`InMemoryBackend.fetch_entry` 普通方法 + `library_stats` attr 默认 None | `cache_storage.py` / `in_memory_backend.py` | B2 在 `load_artifact` 填 `library_stats`；`StoragePayloadView` 已在 B0 走 facade |
| `CachePayload.factors: Optional[dict[str, float]]` 字段 | `storage_types.py` | B2 由 OfflineWriter 写；B1（F1a / F2）不写 |
| `JudgeConfig.factors: list[FactorConfig]` + `composer` + `normalizer` dataclass + list-of-dataclass 解析 | `config.py` | B1 解禁 `_JUDGE_TYPES` 含 `"composite"` 并激活 6 项校验 |
| `_JUDGE_TYPES` **不含** `"composite"`（B0 fail-fast） | `config.py` | B1 第一动作：加进去 |

---

## 3. B1 实施清单（在线算法批）

### 3.1 算法 body — F1a

**位置**：`src/openpi/cache/components/factors/runtime_continuity.py:_RuntimeContinuityBase.extract`

**输入**：
- `results: list[SearchResultLite]`：取 `results[0]` 作为 winner（CompositeJudge 已用 `results[0].id` 做 winner，与 §3 矩阵一致）
- `view: PayloadView`：`view.get(winner_id)` 拿 winner payload；F1a-T 需要 `view.walk_next(winner_id, k)` 取后续 entry
- `history: HistoryView`：`history.actions` / `history.states`（list[Tensor]，per inference cycle 增长一项）
- `cached_data: dict[str, Tensor]`：当前 inference 的 query keys（含 `robot_state`）

**算法**（与 `docs/cache/verdict_factor_judge.md` §3.3 已落地的精确数学一致；本 plan 服从 docs，不重定义）：

State-side fail-safe 设计：state-side 因子（F1a-T）在三个层次上确保不在缺 state 路径上崩：

| 层次 | 触发位置 | 动作 |
|---|---|---|
| **(a) Config 层 fail-fast** | `validate_cache_config` 第 7 项校验（B1 新增） | 任意 `f1a_t` / `f1b_t` factor + `library_stats.state_active_mask.numel() == 0` （即库内无 state 维度） → `ConfigValidationError("composite uses state-side factor but library_stats.state_active_mask is empty")` |
| **(b) Extractor 入口前置 guard** | `_RuntimeContinuityBase.extract` 与 `_SourceWindowSmoothnessBase.extract` 第一行（在任何 `query_keys["robot_state"]` 访问 / z-score / mask 索引之前） | `if cls.source == "state" and (state_sigma.numel() == 0 or state_active_mask.sum() == 0): return {k: nan for k in self.descriptor_orientations}` |
| **(c) per-entry 缺 state 防御** | F1a-T 读 winner / forward `query_keys`；F1b-T `_extract_episode_seq` 读 each entry `query_keys` | 用 `.get("robot_state")` 而非 `[]`；任一返回 None → 整次 `extract` 返全 NaN（不抛 KeyError） |

(a) 防止配置时漏；(b) 防止 deployment 真跑到 state-empty library 时 extract crash；(c) 防止 mixed entry pool（少数 entry 缺 robot_state 字段）时 KeyError。三者并行，互为后备。

1. **State-empty 前置 check**（state-side subclass 才执行；上表 (b) 行）—— 任一触发立即返全 NaN
2. **per-entry 缺 state 防御**（state-side subclass 才执行；上表 (c) 行）—— 任一缺失立即返全 NaN
3. **构造 source 序列**（subclass 选其一）—— 长度严格按 docs §3.3：
   - **F1a-A**（`RuntimeContinuityAction`）：`seq = history.actions[-K:] + [winner_payload.action_chunk[0]]` → 长度 $K+1$ 的 `[K+1, A]`（docs §3.3.1 splice 公式）
   - **F1a-T**（`RuntimeContinuityState`）：先调 `forward_entries = view.walk_next(winner_id, k=K)`（返回 `list[CacheEntry]`，最多 K 条；trajectory 边界 / fork 时更短或 raise NotImplementedError），然后 `seq = history.states[-K:] + [winner_payload.query_keys.get("robot_state")] + [e.query_keys.get("robot_state") for e in forward_entries]` → 长度 $K + 1 + K = 2K+1$ 的 `[2K+1, S]`（docs §3.3.2 splice 公式）
4. **z-score**：`seq_norm = seq / library_stats.{action|state}_sigma.clamp_min(active_eps_{action|state})`
5. **active mask**：`seq_act = seq_norm[..., {action|state}_active_mask]`
6. **descriptors**（共用 `_descriptor_kernel`，docs §3.2 公式）：
   - `jerk = median |Δ²seq_act|`（z-score 后 σ 已除）
   - `dir  = mean cos(v[t], v[t+1])`
   - `curv_radius = mean ||p[t] − centroid||`
   - `cum_disp    = sum ||p[t+1] − p[t]||`
7. **键写入**：`{f"f1a_{cls.key_initial}_{d}": value}`，与 `describe()` 输出一致
8. **边界 / 缺数据**（与 docs §3.3.1 / §3.3.2 边界规则一致）：
   - F1a-A：`len(history.actions) < K`（episode 早期）→ 全部描述子 = NaN
   - F1a-T：`len(history.states) < K`，**或** `view.walk_next(winner_id, k=K)` 返回 list 长度 < K（trajectory boundary）→ 全部描述子 = NaN
   - F1a-T 触发 fork（`view.walk_next` raise `NotImplementedError`，B0 plan §2.3.2 fork policy 占位）→ try/except 捕获 → 全部描述子 = NaN
   - 描述子分母为 0（active 子空间静止窗口算 `dir` 时 ||v||=0）→ 该描述子 NaN（kernel 内部处理）

**复用**：`_RuntimeContinuityBase` 持有一个 `_compute_descriptors(seq_act: Tensor, descriptors: list[str]) -> dict[str, float]` 私有方法，被 `extract` 调用。该 helper **逻辑上与 F1b** §2.8.4 单窗口路径相同 —— 抽到 `factors/_descriptor_kernel.py`（新建模块，纯函数，不进 registry）由 F1a 与 F1b 共同 import，避免两份实现漂移。

### 3.2 算法 body — F2

**位置**：`src/openpi/cache/components/factors/consensus.py:TopKActionConsensus.extract`

**Active mask 决定**：B1 实施时同步修订 docs §3.5 —— 把原 `X_act = X[:, library_stats.action_active_mask]` 改为 **candidate-local active mask**，使其与 capability flag `requires_library_stats=False` 自洽。理由：pi05 padded DOF 在所有候选上常数 0，per-DOF 候选方差天然为 0，candidate-local mask `var_d > 1e-8` 与 library-side `action_active_mask` 在 pi05 deployment 下行为等价；保持 `requires_library_stats=False` 不动 B0 capability flag → validator / wiring / 测试零回退。

```python
def extract(self, results, view, history, cached_data):
    K_eff = min(self.K, len(results))             # cold-start / 库太小保护
    if K_eff < 2:                                  # 单候选无 consensus
        return {"f2_var": float("nan")}
    ids = [r.id for r in results[:K_eff]]
    payloads = view.get_many(ids)                  # PayloadView memo dedup
    chunk0 = torch.stack([
        p.action_chunk[0] for p in payloads
    ], dim=0)                                      # [K, A]
    var_d = chunk0.var(dim=0, unbiased=False)      # [A] population variance
    # candidate-local active mask: drop dims where all K candidates are constant
    # (Pi0.5 padded DOFs are exactly 0 across the whole library → var_d == 0).
    # eps prevents float-noise dims sneaking in at near-zero variance.
    active = var_d > 1e-8
    if active.sum() == 0:
        return {"f2_var": float("nan")}            # 所有 DOF 候选共识 → 0；保留 NaN 而非 0 避免与"完美共识"混同
    f2_var = float(var_d[active].mean())
    return {"f2_var": f2_var}
```

**注**：
- F2 仍**不依赖** `library_stats`（保持 `requires_library_stats = False` —— 不改 B0 capability flag）；mask 完全由当前候选池推出。
- F2 不依赖 `history`。
- B1 同 commit 同步修订 `docs/cache/verdict_factor_judge.md` §3.5 与 `.en.md` 对应章节：把 `action_active_mask`（library 来源）改写为"candidate-local active mask（var > 1e-8）"，并在公式上标注"local mask, no library"。

### 3.3 算法 body — Composer

**位置**：`src/openpi/cache/components/factors/composers/__init__.py`

#### 3.3.1 `WeightedSumComposer.compose`

```python
def compose(self, factors: dict[str, float], *, winner_id: str) -> JudgeResult:
    # factors 已被 normalizer 映射到 [0, 1]（NaN 透传），all-NaN 短路已在
    # CompositeJudge 内处理（§8.2.2 cold-start sentinel），到这里不可能 all-NaN
    total_w = 0.0
    score = 0.0
    for k, v in factors.items():
        if math.isnan(v):
            continue   # NaN 跳过该 key（§2.8.8）
        w = self._weights.get(k, 0.0)
        if w == 0.0:
            continue
        ori = self._orientations[k]
        if ori == "safe":
            contrib = v
        elif ori == "risky":
            contrib = 1.0 - v
        elif ori == "non_monotonic":
            contrib = self._apply_direction(k, v)   # 见 §3.3.2
        else:
            raise ValueError(f"Unknown orientation: {ori}")
        score += w * contrib
        total_w += w
    if total_w == 0.0:
        return JudgeResult(HitType.MISS)             # 没有有效 key
    s = score / total_w
    if s >= self._full_hit_threshold:
        return JudgeResult(HitType.FULL_HIT, winner_id=winner_id)
    if (
        self._warm_start_threshold is not None
        and s >= self._warm_start_threshold
    ):
        return JudgeResult(
            HitType.WARM_START,
            winner_id=winner_id,
            warm_start_t=self._warm_start_t,
        )
    return JudgeResult(HitType.MISS)
```

#### 3.3.2 `_apply_direction` — non_monotonic key 的方向化

YAML `directions: { f1b_a_curv_radius__p0_f5: "high" | "low" | "range:[lo,hi]" }`：

- `"high"`：`contrib = v`（高值倾向 hit）
- `"low"`：`contrib = 1.0 - v`
- `"range:[lo,hi]"`（lo, hi 都是 normalized [0,1] 空间内的值，由 user calibrate）：`contrib = 1.0 if lo <= v <= hi else 0.0`

**校验**：`bind_orientations` 时遍历所有 `non_monotonic` 且 `weight != 0` 的 key — 必须在 `directions` 配齐；缺 → `ConfigValidationError`。该校验在 `CompositeJudge.__init__` 调 `composer.bind_orientations(all_orientations)` 时触发。

#### 3.3.3 `AndGateComposer.compose`

```python
def compose(self, factors, *, winner_id):
    for k, thr in self._per_factor_thresholds.items():
        v = factors.get(k, float("nan"))
        if math.isnan(v):
            return JudgeResult(HitType.MISS)         # NaN = 不通过（§2.8.8 and-gate）
        ori = self._orientations[k]
        passed = (
            v >= thr if ori == "safe" else
            v <= thr if ori == "risky" else
            self._direction_pass(k, v, thr)         # non_monotonic
        )
        if not passed:
            return JudgeResult(HitType.MISS)
    return JudgeResult(HitType.FULL_HIT, winner_id=winner_id)
```

#### 3.3.4 `OrGateComposer.compose`

```python
def compose(self, factors, *, winner_id):
    for k, thr in self._per_factor_thresholds.items():
        v = factors.get(k, float("nan"))
        if math.isnan(v):
            continue                                 # NaN 跳过（§2.8.8 or-gate）
        ori = self._orientations[k]
        if (ori == "safe" and v >= thr) or \
           (ori == "risky" and v <= thr) or \
           (ori == "non_monotonic" and self._direction_pass(k, v, thr)):
            return JudgeResult(HitType.FULL_HIT, winner_id=winner_id)
    return JudgeResult(HitType.MISS)
```

`warm_start_t` emit：三种 Composer 在判 FULL_HIT 时若配了 `warm_start_t > 0`，输出 `JudgeResult(WARM_START, warm_start_t=...)` 而非 `FULL_HIT` —— 由用户配置决定语义（FULL_HIT 模式 `warm_start_t = None`，WARM_START 模式必填）。

### 3.4 算法 body — Normalizer

**位置**：`src/openpi/cache/components/factors/normalizers/__init__.py:PercentileRollingNormalizer`

```python
class PercentileRollingNormalizer:
    def __init__(self, window_size=200, cold_start_strategy="force_miss"):
        self._W = window_size
        self._strategy = cold_start_strategy
        if cold_start_strategy not in ("force_miss", "passthrough", "lenient"):
            raise ValueError(...)
        self._buffers: dict[str, collections.deque[float]] = {}

    def bind_keys(self, keys):
        for k in keys:
            self._buffers[k] = collections.deque(maxlen=self._W)

    def __call__(self, raw):
        out = {}
        for k, v in raw.items():
            buf = self._buffers[k]
            if not math.isnan(v):
                buf.append(v)
            n = len(buf)
            if self._strategy == "passthrough":
                out[k] = v                           # raw 透传（NaN 也透传）
            elif self._strategy == "force_miss":
                if n < self._W:
                    out[k] = float("nan")            # cold-start sentinel
                else:
                    out[k] = self._percentile_rank(buf, v)
            elif self._strategy == "lenient":
                if n < 10:
                    out[k] = float("nan")
                else:
                    out[k] = self._percentile_rank(buf, v)
        return out

    def _percentile_rank(self, buf, v):
        if math.isnan(v):
            return float("nan")
        n = len(buf)
        rank = sum(1 for x in buf if x <= v)
        return rank / n                              # ∈ [0, 1]

    def on_episode_start(self):
        pass    # 滚窗跨 episode 累积，per-episode 不清
```

`cold_start_strategy` 三态语义已在 B0 plan §2.8.8 钉死，此处只是落实算法。

### 3.5 Orchestrator 接线

**位置**：`src/openpi/cache/orchestrator.py`

按 B0 plan §8.2.3 "B1 改动" 完整执行：

| 项 | 内容 |
|---|---|
| `_action_history` / `_state_history` | `__init__` 初始化空 list；`_reset_episode_buffer` 清空 |
| `_state_history_anchor_cp` | `__init__` 末尾 `min(self._gates.keys(), key=lambda cp: cp.value)` |
| `broadcast_action` 末尾 | `self._action_history.append(action_chunk)` |
| `on_task_end` | 在 `_close_current_search_sessions()` 之后**显式**调 `_reset_episode_buffer()`（修 pre-existing leak） |
| `check()` 内 anchor CP 入口 | 若 `checkpoint_id == self._state_history_anchor_cp` 且 `"robot_state" in query_keys`：append 一次 |
| `check()` 内 search 后 | 构造 `view = StoragePayloadView(self._storage)` + `history = HistoryView(actions=list(self._action_history), states=list(self._state_history))` |
| `check()` 内 judge 调用 | `judge(results, checkpoint_id, cached_data, view=view, history=history)` |
| `check()` 内 winner fetch | `payload = view.get(winner_id)`（替代 `storage.fetch_payload`，享 view memo 去重） |

**Lifecycle 表**（B0 plan §8.2.3 "Lifecycle 语义"已钉死）：anchor CP 任意结果（含 gate-skip / FULL_HIT / WARM_START / MISS）都 append 一次 state；非-anchor CP 一律不 append；on_task_end 显式清空 buffer 防 WebSocket 中断泄漏。

### 3.6 Config B1 改造

**位置**：`src/openpi/cache/config.py`

| 改动 | 内容 |
|---|---|
| `_JUDGE_TYPES` 加 `"composite"` | 解除 B0 fail-fast；与 algorithm body 同 commit ship |
| 7 项 composite-specific 校验激活 | 前 6 项见 B0 plan §8.5 `tests/cache/test_config.py` (B1 part) "(1)~(6)"：(1) factors/composer 必填；(2) factor type 必须 `in registry.known()`；(3) `requires_library_stats=True` factor + Qdrant → raise；(4) `requires_chain_walk=True` factor (F1a-T) + Qdrant → raise；(5a-d) warm_start checkpoint gate / canonical timestep / pairwise rule / tier ordering；(6) describe-based directions 覆盖（`f1b_a_curv_radius__p5_f10` 这种参数化 key 在 `composer.directions` 缺失且 weight ≠ 0 时 raise）。**(7) state-library 校验**：composite 配 `f1a_t` / `f1b_t` 任一 + `library_stats.state_active_mask.numel() == 0`（库内无 state 维度）→ `ConfigValidationError("composite uses state-side factor but library_stats has no state dim")`。该校验在 `build_per_connection_components` 拿到 per-conn `library_stats` 后、构造 CompositeJudge 前执行；in-memory backend lazy fallback 现算 stats 后再次触发该路径。 |
| `min_required_top_k` 反向喂回 strategy | `build_per_connection_components` 内 per-CP loop：`min_hint = judges[cp_id].min_required_top_k`；`strategies[cp_id] = _build_search_strategy(..., min_top_k_hint=min_hint)` |

### 3.7 文档同步（B1）

| 文件 | 改动 |
|---|---|
| `docs/architecture/cache_system.md` §5.6 / §5.11 / §5.12 | 把"骨架 / Protocol / B0 ship"语句改为"实现已落地，详见 `cache/verdict_factor_judge.md`" |
| `docs/cache/verdict_factor_judge.md` + `.en.md` 各章 | 各章节顶部 status 标记由 `B1 Pending` → `Implemented`；`B2 Pending` 仍保留；§4-§7 移除"算法尚未实现"的免责说明 |
| `docs/cache/tutorial.md` §6 Judge 表 | `composite` 行的"目前 B0 fail-fast"备注移除；写明 B1+ 可用 |
| `docs/README.md` cache/ 表 | 同步 verdict_factor_judge 描述（去掉 B0/B1/B2 status mention，因为已合并落地） |
| `logs/README.md` | 标记 `verdict_factor_judge.log.md` 为 `Validated`（B0 路径）；新增 `verdict_factor_judge_b1_b2.log.md` 行 |

---

## 4. B2 实施清单（离线写入批）

### 4.1 算法 body — F1b OnlineExtractor + OfflineWriter

**位置**：`src/openpi/cache/components/factors/source_window.py`

#### 4.1.1 `_SourceWindowSmoothnessBase.extract`（OnlineExtractor 路径）

```python
def extract(self, results, view, history, cached_data):
    if not results:
        return {k: float("nan") for k in self.descriptor_orientations}
    winner = view.get(results[0].id)                    # PayloadView memo
    factors_dict = winner.factors                       # CachePayload.factors
    if factors_dict is None:
        # 老 entry 或 builder 未跑 OfflineWriter
        return {k: float("nan") for k in self.descriptor_orientations}
    out = {}
    for key in self.descriptor_orientations:
        out[key] = factors_dict.get(key, float("nan"))   # 缺单个 key 也 NaN
    return out
```

**注**：F1b 在线路径**只读** `payload.factors`，不重算 — 算法成本 100% 移到 build pkl 阶段。`source == "state"` 的 F1b-T 在线侧不需要 `library_stats` / `query_keys["robot_state"]` 访问（数据已固化在 winner.factors），所以 §3.1 的"state-empty 前置 guard"在 F1b OnlineExtractor 不需要 —— OfflineWriter 写入阶段如果 state lib 是 0 维或 entry 缺 state，写入的就是全 NaN factors，online 侧读到全 NaN 自动透传。

#### 4.1.2 `_SourceWindowSmoothnessBase.compute_for_episode`（OfflineWriter 路径）

按 B0 plan §2.8.4 完整算法：

```python
def compute_for_episode(self, entries, library_stats):
    sigma = library_stats.action_sigma if self.source == "action" else library_stats.state_sigma
    active_mask = library_stats.action_active_mask if self.source == "action" else library_stats.state_active_mask

    # ---- State-side fail-safe: empty library / zero mask ----
    # State library may be empty (state_dim=0) or all-zero-mask. In either
    # case bail out before any state key access / z-score division —
    # otherwise sigma.clamp_min(...) returns shape [0] and downstream
    # broadcasting silently produces wrong-shape tensors.
    if self.source == "state" and (sigma.numel() == 0 or active_mask.sum() == 0):
        return [{k: float("nan") for k in self.descriptor_orientations} for _ in entries]

    # ---- per-entry state-presence guard: any missing state → all-NaN ----
    # If any entry is missing the state field (mixed pool), bail out:
    # OfflineWriter is per-episode and per-window descriptors require a
    # contiguous source sequence, so missing one entry corrupts neighbors
    # too. Cleaner to write all-NaN factors for the entire episode and
    # log; partial NaN per entry is left to the boundary-NaN path.
    if self.source == "state":
        for e in entries:
            if e.query_keys.get("robot_state") is None:
                return [{k: float("nan") for k in self.descriptor_orientations} for _ in entries]

    # 抽取 per-entry source 序列：F1b-A 用 entry.payload.action_chunk[0]；
    # F1b-T 用 entry.query_keys["robot_state"]
    seq = self._extract_episode_seq(entries)             # [T, D]

    # z-score + active mask（§2.8.4 Step 1+2）
    p_norm = seq                                / sigma.clamp_min(self._active_eps)
    v_norm = (seq[1:] - seq[:-1])               / sigma.clamp_min(self._active_eps)
    j_norm = (seq[2:] - 2*seq[1:-1] + seq[:-2]) / sigma.clamp_min(self._active_eps)
    p_act, v_act, j_act = p_norm[..., active_mask], v_norm[..., active_mask], j_norm[..., active_mask]

    T = len(entries)
    out: list[dict[str, float]] = []
    for idx in range(T):
        per_entry: dict[str, float] = {}
        for (p, f) in self._windows:
            sl = slice(max(0, idx - p), min(T, idx + f + 1))
            pts = p_act[sl]
            v_w = v_act[max(0, idx - p):min(T - 1, idx + f)]
            j_w = j_act[max(0, idx - p):min(T - 2, idx + f - 1)]
            # 边界 NaN
            if (idx - p < 0) or (idx + f + 1 > T) or len(pts) < 2:
                for d in self._descriptors:
                    per_entry[f"f1b_{self.key_initial}_{d}__p{p}_f{f}"] = float("nan")
                continue
            for d in self._descriptors:
                per_entry[f"f1b_{self.key_initial}_{d}__p{p}_f{f}"] = self._descriptor_kernel(d, pts, v_w, j_w)
        out.append(per_entry)
    return out

def _extract_episode_seq(self, entries):
    # PRECONDITION: caller already passed the state-side fail-safe and
    # per-entry presence guard above; here we can dereference safely.
    if self.source == "action":
        return torch.stack([
            torch.as_tensor(e.payload.action_chunk[0], dtype=torch.float32)
            for e in entries
        ], dim=0)
    elif self.source == "state":
        return torch.stack([
            torch.as_tensor(e.query_keys["robot_state"], dtype=torch.float32)
            for e in entries
        ], dim=0)
    else:
        raise ValueError(f"Unknown source: {self.source}")
```

`_descriptor_kernel(d, pts, v, j)` 与 §3.1 F1a 所用 helper 同源（`factors/_descriptor_kernel.py`），保证 F1a / F1b 计算口径一致。

未列入 `descriptors` 的项不计算（已在 `__init__` 处校验描述子名）；`dirvar / path / freq / autocorr` 在 `_descriptor_kernel` 内 `raise NotImplementedError`，由 `__init__` 提前 fail-fast 拦住。

### 4.2 算法 body — `LibraryStats.compute_from_entries`

**位置**：`src/openpi/cache/components/factors/base.py`

**Param 名**：严格使用 B0 已 ship 的 `active_eps_action` / `active_eps_state`（`base.py:59-60`）。

**State 缺失行为**（dataclass 类型不 Optional，不能返回 None）：

`state_sigma` / `state_active_mask` 是非 Optional Tensor，必须填实例。当 entries 没有 `robot_state`（极少见，需 backend 在 prefill 阶段确实漏拿 robot_state）时：
- `state_sigma = torch.zeros(0, dtype=torch.float32)` —— 长度 0 的占位
- `state_active_mask = torch.zeros(0, dtype=torch.bool)` —— 长度 0 的占位

state-side 因子（F1a-T / F1b-T）在 extract / compute_for_episode 入口（**在 `query_keys["robot_state"]` 访问 / z-score 除法 / mask 索引之前**）有前置 check：

```python
if self.source == "state" and (state_sigma.numel() == 0 or state_active_mask.sum() == 0):
    return all_nan_for_descriptors(...)        # F1a-T: dict; F1b-T compute: list of dicts
```

并在 config 层加 fail-fast 校验（§3.6 第 7 项），`validate_cache_config` 在 composite 配 `f1a_t` / `f1b_t` 但 `library_stats.state_active_mask.numel() == 0` 时直接 raise。

**state_dim 来源**：从首个含 `robot_state` 的 entry 推；若整个 entries 列表都没 robot_state，state placeholder 长度 = 0；下游的 mask check 兜底。

```python
@classmethod
def compute_from_entries(
    cls,
    entries: list["CacheEntry"],
    active_eps_action: float = 0.01,
    active_eps_state: float = 0.01,
) -> "LibraryStats":
    # action: 用每个 entry 的 action_chunk[0]（pre-output-transform domain）
    # 防 numpy：detach 后 pkl 里 action_chunk 是 numpy；helper 接受 numpy → torch.as_tensor
    actions = torch.stack([
        torch.as_tensor(e.payload.action_chunk[0], dtype=torch.float32)
        for e in entries
    ], dim=0)                                                                    # [N, A]
    action_sigma = actions.std(dim=0, unbiased=False)                            # [A]
    action_active_mask = action_sigma >= active_eps_action                       # [A] bool

    # state: 用每个 entry 的 query_keys["robot_state"]
    states_list = []
    for e in entries:
        rs = e.query_keys.get("robot_state")
        if rs is not None:
            states_list.append(torch.as_tensor(rs, dtype=torch.float32))
    if states_list:
        states = torch.stack(states_list, dim=0)                                 # [N', S]
        state_sigma = states.std(dim=0, unbiased=False)
        state_active_mask = state_sigma >= active_eps_state
    else:
        # placeholder zeros — F1a-T / F1b-T 检测 active_mask.sum() == 0 → 全 NaN
        # state_dim 取 0 时下游不会触发计算；非 0 时按 zeros 占位
        state_dim = 0
        state_sigma = torch.zeros(state_dim, dtype=torch.float32)
        state_active_mask = torch.zeros(state_dim, dtype=torch.bool)

    return cls(
        action_sigma=action_sigma, action_active_mask=action_active_mask,
        state_sigma=state_sigma, state_active_mask=state_active_mask,
    )
```

**`active_eps_action` / `active_eps_state` 配置入口**：B1+B2 **不引入** YAML override（B0 也未声明 `backend.config.library_stats` 字段）—— 使用 classmethod 默认值 `0.01`（与 B0 plan §2.8.3 一致）。理由：所有 deployment 都跑 pi05 32-DOF 同 embodiment，0.01 是固定阈值；后续若真要按 dataset 调，再独立扩 schema。本 plan 不动 `InMemoryConfig` / `BackendConfig`。

### 4.3 Orchestrator B2 接线（**实现，非测试覆盖**）

**位置**：`src/openpi/cache/orchestrator.py`

**前置事实**：B0 plan §8.2.3 描述的 `_build_entry_chain` "B2 改动" 骨架**未在 B0 ship**（核查 `orchestrator.py:96` `__init__` 与 `:487` `_build_entry_chain`，无 `offline_writers` / `library_stats` 参数也无 merge 循环）。本 plan B2 自己实现该骨架。

#### 4.3.1 `__init__` 加参数 + 持有

```python
def __init__(
    self,
    ...,                                                # existing args unchanged
    offline_writers: Sequence["OfflineWriter"] = (),    # NEW — B2
    library_stats: Optional["LibraryStats"] = None,     # NEW — B2
) -> None:
    ...
    self._offline_writers: tuple["OfflineWriter", ...] = tuple(offline_writers)
    self._library_stats: Optional["LibraryStats"] = library_stats
```

默认空 tuple / None 保持向后兼容（无 composite judge 的 deployment 行为字节级不变）。

#### 4.3.2 `_build_entry_chain` 加 merge 循环

在现有 chain build 完成后追加：

```python
def _build_entry_chain(self, record: EpisodeRecord) -> list[CacheEntry]:
    entries = ...                                        # existing chain link build
    if self._offline_writers and self._library_stats is not None:
        for writer in self._offline_writers:
            per_entry_factors = writer.compute_for_episode(entries, self._library_stats)
            for entry, factors in zip(entries, per_entry_factors, strict=True):
                if entry.payload.factors is None:
                    entry.payload.factors = {}
                entry.payload.factors.update(factors)
    return entries
```

#### 4.3.3 Config 收集 OfflineWriters 喂给 Orchestrator

`config.py` 新增 helper：

```python
def collect_offline_writers_from_judges(
    judges: dict[CheckpointID, "SimilarityJudge"],
) -> list["OfflineWriter"]:
    """Walk per-CP CompositeJudge.extractors, return those that also
    implement OfflineWriter (i.e. have `compute_for_episode`).
    Order = CP enum order × extractor order within each CompositeJudge.
    Duplicates (same writer instance referenced by multiple CPs) are
    de-duplicated by `id(writer)`.
    """
    seen: set[int] = set()
    out: list["OfflineWriter"] = []
    for cp_id in sorted(judges.keys(), key=lambda c: c.value):
        judge = judges[cp_id]
        extractors = getattr(judge, "_extractors", ())
        for ext in extractors:
            if hasattr(ext, "compute_for_episode") and id(ext) not in seen:
                out.append(ext)
                seen.add(id(ext))
    return out
```

`build_per_connection_components` 末尾改为：

```python
offline_writers = collect_offline_writers_from_judges(judges)
return Orchestrator(
    ...,
    offline_writers=offline_writers,
    library_stats=per_conn_storage.library_stats,        # facade duck-types backend
)
```

#### 4.3.4 Lifecycle 互不干扰

`_build_entry_chain` 触发于 `_close_current_search_sessions()` 之后（B0 plan §8.7 已论证 search session 释放与 entry chain build 不冲突）；B2 加的 merge 循环走完才返回 entries。失败语义：单个 writer 抛 → 整条 entry chain build 失败 → orchestrator log + skip 该 episode（与现有 chain build 失败行为一致；不静默吞）。

### 4.4 Backend B2 接线

**位置**：`src/openpi/cache/backends/in_memory_backend.py`

`load_artifact` 改造：

```python
def load_artifact(self, path):
    data = pickle.load(open(path, "rb"))
    self._entries = ...                                  # existing
    ls = data.get("library_stats")
    if ls is None:
        # 老 artifact fallback：扫 entries 现算
        from openpi.cache.components.factors.base import LibraryStats
        log.warning("artifact missing library_stats, computing from %d entries", len(self._entries))
        t0 = time.time()
        ls = LibraryStats.compute_from_entries(list(self._entries.values()))
        log.warning("library_stats computed in %.2fs", time.time() - t0)
    self.library_stats = ls
```

`InMemoryBackend.library_stats` attr 在 B0 已 ship 默认 None；B2 让 `load_artifact` 真正填它。Qdrant 等其他 backend 不受影响（duck-typed facade 走 `getattr` 路径）。

### 4.5 Artifact build 工具接线

**位置**：`exp/common/factor_postprocess.py`（**新建**）+ 三 build 脚本

**关键约束**：`build_in_memory_cache_artifact.py:651` `_detach_entries` 在 `_process_episode*` 内把 `payload.action_chunk` / `query_keys` 从 torch.Tensor 转为 numpy.ndarray（节省 pkl 体积、降低 IPC 成本）。`build_in_memory_cache_artifact.py:28` 同时用了 `ProcessPoolExecutor`，`_process_episode*` 是**子进程函数**，返回结果走 IPC 序列化 —— 若把 detach 移到主进程做，子进程必须把 torch.Tensor pickle 回主进程，体积大 / 慢 / 易崩。

**采用 helper 兼容 numpy via `torch.as_tensor` 桥接**：不动 `_detach_entries` lifecycle（保持 ProcessPool IPC 优化），只在 helper 内部 / OfflineWriter 内部 / `LibraryStats.compute_from_entries` 内部用 `torch.as_tensor(...)` 兜底兼容 numpy 或 torch 输入（已在 §4.1.2 与 §4.2 落实）。

**实现路径**：

- `_process_episode*` 与 `_detach_entries` 不动（保持 IPC 优化）
- `build_artifact`（line 673）顶层在 `entries` 收齐后（detach 已发生）调：
  ```python
  library_stats = enrich_artifact_with_factors(
      entries,
      offline_writers,
      active_eps_action=0.01,
      active_eps_state=0.01,
  )
  ```
- helper 内部按 `trajectory_id` 切分 episode；OfflineWriter 与 LibraryStats 内部都用 `torch.as_tensor(...)` 兜底（§4.1.2 / §4.2 已加）
- 最终 artifact dict 写：
  ```python
  artifact_dict = {
      ...,                                              # existing keys
      "entries": entries,                               # payload.factors 已填好（numpy 体外保存的 Python dict[str, float]）
      "library_stats": library_stats,                   # NEW
  }
  ```
- 因 `payload.factors` 是 `dict[str, float]`（Python 标量字典）不是 Tensor，detach 不影响其序列化

**对老 artifact 加载兼容**：`InMemoryBackend.load_artifact` 检测无 `library_stats` 字段时调 `LibraryStats.compute_from_entries` 现算 fallback —— 此时 entries 已是 numpy（从 pkl 反序列化），`compute_from_entries` 内部 `torch.as_tensor` 兜底。F1b OnlineExtractor 从 `payload.factors` 读 float 也无 tensor 依赖。

#### 4.5.1 新建 `exp/common/factor_postprocess.py`

```python
"""Helper: enrich an artifact's entries with per-entry `payload.factors`
and compute the artifact-level `library_stats`. Used by build_*.py
scripts after their entry-list is finalized but before pickling.
"""

from openpi.cache.components.factors.base import LibraryStats, OfflineWriter
from openpi.cache.storage_types import CacheEntry


def enrich_artifact_with_factors(
    entries: list[CacheEntry],
    offline_writers: list[OfflineWriter],
    *,
    active_eps_action: float = 0.01,
    active_eps_state: float = 0.01,
) -> LibraryStats:
    """In-place: write per-entry `payload.factors` (per writer) + compute
    `LibraryStats` from the entry pool. Returns LibraryStats so the caller
    can attach it to the artifact dict's `library_stats` field.

    Per-episode segmentation: writers receive per-trajectory entry slices
    keyed by `entry.trajectory_id`, sorted by `entry.step_idx` (the actual
    field name on `CacheEntry`). Entries with `step_idx is None` go to a
    separate "unnamed" bucket and are processed in collection order; if any
    such bucket exists the helper logs a warning since the writer's window
    semantics depend on monotonic step ordering.

    Tolerates both torch.Tensor and numpy.ndarray payload tensors:
    `LibraryStats.compute_from_entries` and OfflineWriter implementations
    use `torch.as_tensor(...)` internally to bridge either input type.
    This means the helper is safe to call after `_detach_entries` has run
    (the primary builder path uses ProcessPool and detaches inside the
    subprocess to keep IPC cheap; that's left in place — see §4.5 design
    notes for why option 2 was chosen over moving detach).
    """
    library_stats = LibraryStats.compute_from_entries(
        entries,
        active_eps_action=active_eps_action,
        active_eps_state=active_eps_state,
    )
    if not offline_writers:
        return library_stats

    # group entries by trajectory_id, preserve step_idx order
    # (CacheEntry.step_idx is Optional[int]; None entries kept in collection
    # order to a separate bucket — log warning since window semantics expect
    # monotonic step ordering).
    by_traj: dict[str, list[CacheEntry]] = {}
    for e in entries:
        by_traj.setdefault(e.trajectory_id, []).append(e)
    for traj_id, traj_entries in by_traj.items():
        named   = [e for e in traj_entries if e.step_idx is not None]
        unnamed = [e for e in traj_entries if e.step_idx is None]
        named.sort(key=lambda e: e.step_idx)
        if unnamed:
            logging.warning(
                "trajectory %s has %d entries with step_idx=None; "
                "appending in collection order (window descriptors may be invalid)",
                traj_id, len(unnamed),
            )
        ordered = named + unnamed

        for writer in offline_writers:
            per_entry_factors = writer.compute_for_episode(ordered, library_stats)
            for entry, factors in zip(ordered, per_entry_factors, strict=True):
                if entry.payload.factors is None:
                    entry.payload.factors = {}
                entry.payload.factors.update(factors)
    return library_stats
```

#### 4.5.2 三脚本各加 ~3 行

**`exp/common/build_in_memory_cache_artifact.py`** / **`build_clip_cache_artifact.py`** / **`build_llm_layer_matrix.py`** 在 `entries` 列表完成后、`pickle.dump` 之前：

```python
from exp.common.factor_postprocess import enrich_artifact_with_factors

# CLI 取因子配置（新增 --factors-yaml 参数；空则跳过 OfflineWriter）
offline_writers = _load_offline_writers_from_yaml(args.factors_yaml) if args.factors_yaml else []
library_stats = enrich_artifact_with_factors(entries, offline_writers)

artifact_dict = {
    "key_builder_type": ...,
    "checkpoint_id": ...,
    "vector_dims": ...,
    "entries": entries,
    "library_stats": library_stats,                      # NEW
}
pickle.dump(artifact_dict, ...)
```

`_load_offline_writers_from_yaml` 是三脚本共用的小 helper（也放 `factor_postprocess.py`）：

```python
def _load_offline_writers_from_yaml(yaml_path: str) -> list[OfflineWriter]:
    """Read minimal YAML, return list of OfflineWriter instances. Validates
    that each `type` is registered AND has `compute_for_episode` (i.e. is
    a real OfflineWriter, not an OnlineExtractor-only factor). Online-only
    factors (F1a, F2) in the YAML raise ConfigValidationError.
    """
    data = yaml.safe_load(open(yaml_path))
    out: list[OfflineWriter] = []
    for entry in data.get("factors", []):
        cls = registry.get_class(entry["type"])
        if not hasattr(cls, "compute_for_episode"):
            raise ConfigValidationError(
                f"factor type {entry['type']!r} has no compute_for_episode "
                f"(F1a / F2 only run online; cannot be used as OfflineWriter)"
            )
        out.append(registry.build(entry["type"], **entry.get("params", {})))
    return out
```

`registry.build(name, **params)` 已是 B0 ship 的接口（`factors/registry.py:build`）；当某 factor 的 class `requires_library_stats=True` 而构造方需要 `library_stats` kwarg 时，这条离线路径**不**注入 library_stats —— 由 OfflineWriter 的 `compute_for_episode(entries, library_stats)` 显式传入；构造期不依赖。OfflineWriter 类（F1b base）的 `__init__` 必须能接受 `library_stats=None`（B0 plan §8.1.6 已 ship `library_stats: "LibraryStats"` 但 F1b 仅在 OnlineExtractor 路径用 `self._library_stats`，OfflineWriter 路径用入参；为兼容离线构造期无 library_stats 的场景，B2 把 `__init__` 的 `library_stats` 参数改为 `Optional[LibraryStats] = None`）。

minimal YAML 示例：

```yaml
# 例：only F1b（F1a / F2 是 online-only，不出现在此 YAML）
factors:
  - type: f1b_a
    params: { windows: [{past:0,future:5},{past:5,future:5}], descriptors: [jerk, dir, curv_radius, cum_disp], active_eps: 0.01 }
  - type: f1b_t
    params: { windows: [{past:0,future:5},{past:5,future:5}], descriptors: [jerk, dir, curv_radius, cum_disp], active_eps: 0.01 }
```

#### 4.5.3 老 artifact 兼容

- 不带 `library_stats` 字段的 pkl：`InMemoryBackend.load_artifact` 现算 fallback（§4.4）—— 一次性，server 启动时 log 警告
- entries 不带 `payload.factors`：F1b OnlineExtractor.extract 检测 `factors is None` → 该 factor 所有 key 返 NaN → Composer 按 §2.8.8 跳过
- **不主动重建任何现有 artifact**（属于实验级活动；本 plan §1.2 非目标）

### 4.6 文档同步（B2）

| 文件 | 改动 |
|---|---|
| `docs/architecture/cache_system.md` §5.12 | 补完 OfflineWriter 完整契约 + artifact 顶层 dict `library_stats` 字段说明 + 老 artifact fallback 行为 |
| `docs/cache/verdict_factor_judge.md` + `.en.md` | §5/§6/§7 build pkl / YAML config / 运行实验章节由"B2 Pending"改为"实现"；§3.4-§3.5 数学公式后补 "已落地" 标识；**§3.5 F2 公式重写**（`action_active_mask` library 来源 → candidate-local mask `var_d > 1e-8`），明示"local mask, no library_stats"以与 `requires_library_stats=False` 自洽 |
| `docs/experiments/artifact_layout.md` | 加一行说明：新 artifact pkl 含 `library_stats` 字段；老 artifact fallback 路径；提及新 CLI flag `--factors-yaml` |
| `exp/common/build_*.py` 文件顶部 docstring | 补一句："如配 `--factors-yaml`，调 `enrich_artifact_with_factors` 写 `payload.factors` + 顶层 `library_stats`" |
| `docs/README.md` | architecture/ 表的 cache_system.md 描述里补"§5.12 OfflineWriter implementation landed in B2" |

---

## 5. 文件改动总览

### 5.1 新建文件

| 路径 | 用途 | 批次 |
|---|---|---|
| `src/openpi/cache/components/factors/_descriptor_kernel.py` | 纯函数 helper：jerk/dir/curv_radius/cum_disp 单窗口实现，F1a/F1b 共用 | B1 |
| `exp/common/factor_postprocess.py` | `enrich_artifact_with_factors` + `_load_offline_writers_from_yaml` | B2 |
| `tests/cache/components/factors/test_runtime_continuity.py` | F1a-A / F1a-T extract 数值正确性 | B1 |
| `tests/cache/components/factors/test_consensus.py` | F2 extract 数值正确性 | B1 |
| `tests/cache/components/factors/test_composers_algorithm.py` | 3 Composer compose 算法 | B1 |
| `tests/cache/components/factors/test_normalizer_algorithm.py` | PercentileRollingNormalizer __call__ 算法 | B1 |
| `tests/cache/components/factors/test_descriptor_kernel.py` | 单窗口 4 描述子合成 sweep / turn / shake / static regime 单元测试 | B1 |
| `tests/cache/test_orchestrator_history.py` | anchor CP 选择 + state append 规则 + episode reset + on_task_end leak fix | B1 |
| `tests/cache/components/factors/test_source_window.py` | F1b OnlineExtractor + OfflineWriter | B2 |
| `tests/cache/components/factors/test_base.py` | LibraryStats.compute_from_entries 数值正确性 | B2 |
| `tests/cache/test_artifact_roundtrip.py` | enrich_artifact_with_factors helper + 新 artifact round-trip + 老 artifact fallback compute | B2 |
| `tests/cache/test_factor_postprocess.py` | factor_postprocess 模块的 trajectory grouping + writer order + missing trajectory_id 处理 | B2 |

### 5.2 修改的现有文件

| 路径 | B1 改动 | B2 改动 |
|---|---|---|
| `src/openpi/cache/components/factors/runtime_continuity.py` | `_RuntimeContinuityBase.extract` body 填实 | — |
| `src/openpi/cache/components/factors/consensus.py` | `TopKActionConsensus.extract` body 填实 | — |
| `src/openpi/cache/components/factors/source_window.py` | — | `extract` + `compute_for_episode` body 填实；`__init__` 的 `library_stats` 参数由 required `"LibraryStats"` 改为 `Optional["LibraryStats"] = None`（OfflineWriter 离线构造期不依赖 library_stats，由 `compute_for_episode(entries, library_stats)` 显式传入；OnlineExtractor 路径仍需要 library_stats —— 由 builder 注入） |
| `src/openpi/cache/components/factors/base.py` | — | `LibraryStats.compute_from_entries` body 填实 |
| `src/openpi/cache/components/factors/composers/__init__.py` | 3 个 `compose` body 填实 + `_apply_direction` / `_direction_pass` helper + bind_orientations 校验 directions 完整性 | — |
| `src/openpi/cache/components/factors/normalizers/__init__.py` | `PercentileRollingNormalizer.__call__` + `_percentile_rank` 填实 | — |
| `src/openpi/cache/orchestrator.py` | view+history 注入 + `_state_history` + anchor CP + on_task_end leak fix + winner fetch rewire | **新增**：`__init__` 加 `offline_writers` / `library_stats` 参数 + 持有；`_build_entry_chain` 加 OfflineWriter merge 循环（B0 未 ship 此骨架，本批次自己实现） |
| `src/openpi/cache/config.py` | `_JUDGE_TYPES` 加 `"composite"` + 6 项校验激活 + `min_required_top_k` 反向喂回 strategy | **新增**：`collect_offline_writers_from_judges` helper + `build_per_connection_components` 末尾把 writers + library_stats 喂给 Orchestrator |
| `src/openpi/cache/backends/in_memory_backend.py` | — | `load_artifact` 加载 `library_stats` + fallback 现算 |
| `src/openpi/cache/components/judge.py` | 已 B0 ship；B1 algorithm 接通后端到端可跑 | — |
| `src/openpi/cache/components/payload_view.py` | 已 B0 ship；B1 在 Orchestrator 内首次构造使用 | — |
| `tests/cache/test_judge.py` | CompositeJudge end-to-end FULL_HIT / WARM_START / MISS 三路扩充 | — |
| `tests/cache/test_orchestrator.py` | check() 注入 view+history；老 Judge 收 kwargs 不报错；CompositeJudge 端到端 | `Orchestrator.__init__` 接受 `offline_writers` / `library_stats`；`_build_entry_chain` 调 writer + 写 payload.factors；空 writers / None library_stats 走 fast path 行为不变 |
| `tests/cache/test_config_factor.py` | composite YAML B1 不再被拒收；6 项校验全覆盖 + **第 7 项 state-library 校验**（composite 配 f1a_t/f1b_t + 空 state library → ConfigValidationError） | — |
| `tests/cache/test_cache_storage.py` | — | library_stats property 实路径（非 None） |
| `exp/common/build_in_memory_cache_artifact.py` | — | `--factors-yaml` CLI + enrich helper + library_stats 字段写入 |
| `exp/common/build_clip_cache_artifact.py` | — | 同上 |
| `exp/common/build_llm_layer_matrix.py` | — | 同上 |
| `docs/architecture/cache_system.md` | §5.6 / §5.11 / §5.12 status 更新（"骨架"→"已实现"） | §5.12 补完 OfflineWriter / artifact 字段 |
| `docs/cache/verdict_factor_judge.md` + `.en.md` | 各章 status B1→Implemented | 各章 status B2→Implemented |
| `docs/cache/tutorial.md` | §6 composite 行解禁备注 | — |
| `docs/experiments/artifact_layout.md` | — | 新 artifact 字段说明 + `--factors-yaml` flag |
| `docs/README.md` | cache_system.md 描述更新 | 同上 |
| `logs/README.md` | 新增 `verdict_factor_judge_b1_b2.log.md` 行（Plan）；B0 状态保持 | B1 Done 后状态变 In Progress；B2 完成 + Verify pass 后改 Validated |

---

## 6. 测试策略

### 6.1 B1 测试（必过）

| 测试文件 | 验收点 |
|---|---|
| `test_descriptor_kernel.py` | 4 描述子在 4 种合成 regime（sweep 直线 / turn 转弯 / shake 抖动 / static 停滞）的相对量级符合预期；NaN 边界（v=0 / 输入太短）正确 |
| `test_runtime_continuity.py` | F1a-A 用合成 history+winner 序列产出非 NaN；F1a-T 用 mock view.walk_next 产出非 NaN；history 不足时所有描述子 NaN；descriptor_orientations 与 describe 一致；**state fail-safe 三层覆盖**：(b) state_sigma.numel()==0 或 active_mask.sum()==0 → 入口 all-NaN（无 KeyError / 无 shape 错）；(c) winner / forward state 缺 robot_state → 入口 all-NaN |
| `test_consensus.py` | K=5、5 个 mock results 的 action_chunk[0] 偏移合成 → variance 正比于偏移幅度；K=1 / cold-start 返 NaN |
| `test_composers_algorithm.py` | WeightedSum 在 NaN-mix / orientation flip / non_monotonic direction 下 score 计算正确；tier 映射（FULL_HIT / WARM_START / MISS）阈值边界；`warm_start_t` emit；AndGate / OrGate 同样 |
| `test_normalizer_algorithm.py` | PercentileRolling 三种 cold_start_strategy 行为；NaN 透传；window 滚动正确；`_percentile_rank` 公式 |
| `test_orchestrator_history.py` | anchor CP 选择（CP1 / CP3 / CP1+CP3）；state append 一次（含 gate-skip / FULL_HIT / WARM_START / MISS 各路径）；非-anchor CP 不 append；episode reset 清空；on_task_end leak fix |
| `test_orchestrator.py` (B1 part) | check() 注入 view+history；老 Judge 收 kwargs 不报错；CompositeJudge 端到端 FULL_HIT / WARM_START / MISS 三路 |
| `test_judge.py` (B1 part) | CompositeJudge 端到端三路 + key contract assert + cold-start sentinel 在真实 normalizer 下触发 |
| `test_config_factor.py` (B1 part) | composite YAML 不再被拒；6 项校验全覆盖 + 第 7 项 state-library 校验；`min_required_top_k` 反向喂回 strategy |

### 6.2 B2 测试（必过）

| 测试文件 | 验收点 |
|---|---|
| `test_base.py` | `LibraryStats.compute_from_entries` 在合成 entry 列表上 σ_d 数值正确；active_mask 阈值边界；state 字段缺失时 graceful |
| `test_source_window.py` | OfflineWriter.compute_for_episode 在合成 episode 上 4 描述子 × N 窗口数值正确；OnlineExtractor.extract 从 payload.factors 读取一致；缺字段 NaN；边界窗口 NaN；**state fail-safe 三层覆盖**：(b) state_sigma 为 0 维 → compute_for_episode 全 entry 全 NaN factors；(c) episode 内任一 entry 缺 robot_state → 整 episode 全 NaN（不抛 KeyError） |
| `test_orchestrator.py` (B2 part) | `_build_entry_chain` 在配 OfflineWriter 时正确写 payload.factors；library_stats 透传 writer |
| `test_artifact_roundtrip.py` | `enrich_artifact_with_factors` helper 在 trajectory 切分正确；新 artifact pickle round-trip 字段完整；老 artifact (无 library_stats) 加载触发 fallback compute；factors=None entry 经 F1b 走 NaN 路径 |
| `test_factor_postprocess.py` | trajectory grouping by `trajectory_id` + `step_idx` 排序正确（`step_idx` 是 `CacheEntry` 的实际字段名）；`step_idx is None` 的 entry 走 unnamed 桶 + warning，不抛错；多 writer 顺序写入不互相覆盖（key 不冲突）；空 writers 列表只算 library_stats；**numpy / torch 双输入路径**：`enrich_artifact_with_factors` 接受 detach 已发生（numpy）和未 detach（torch）两种输入都正确产出 `library_stats` + 写 `payload.factors`（option 2 桥接，§4.5 设计） |
| `test_artifact_roundtrip.py` (extended) | **真 builder 路径**：构造 minimal HDF5 fixture → 跑 `build_in_memory_cache_artifact._process_episode` 等真路径 + `--factors-yaml` → 写出的 pkl 含 `library_stats` + entries 含 `payload.factors`；load 回来 InMemoryBackend.library_stats 非 None；验证 `_detach_entries` 仍按原 lifecycle 在 `_process_episode*` 内执行（pkl 里 action_chunk 是 numpy），enrichment 跑在 numpy 上仍正确（option 2 桥接验证） |
| `test_cache_storage.py` (B2 part) | `library_stats` property 在加载 artifact 后非 None |

### 6.3 现有 561 cache 测试（必过 / 零回归）

B1+B2 落地后 `uv run pytest tests/cache/` 全部通过；老 Judge / 老 YAML / 老 artifact / Qdrant backend 路径行为零变化。

### 6.4 Manual 测试（不阻 CI）

`@pytest.mark.manual` 标 GPU / 真 server 路径；本 plan 不引入新 manual 测试 —— 复用现有 pi05 inference 端到端测试，verify B1 wiring 在真模型下不 crash 即可（执行者自行触发）。

---

## 7. 实施顺序

合并 batch 但**内部按依赖序**实施，每一步 commit 自包含且 `pytest` 全绿：

| 序号 | 步骤 | 涉及文件 | 验证 |
|---|---|---|---|
| 1 | 抽出 `_descriptor_kernel.py` 纯函数 | 新建 + 4 描述子单元测试 | `test_descriptor_kernel.py` |
| 2 | F1a-A / F1a-T `extract` body 填实 | `runtime_continuity.py` + 测试 | `test_runtime_continuity.py` |
| 3 | F2 `extract` body 填实 | `consensus.py` + 测试 | `test_consensus.py` |
| 4 | 3 Composer + Normalizer body 填实 | `composers/__init__.py` + `normalizers/__init__.py` + 2 测试 | `test_composers_algorithm.py` / `test_normalizer_algorithm.py` |
| 5 | Orchestrator B1 改造（`_state_history` + anchor CP + view/history 注入 + winner fetch rewire + on_task_end fix） | `orchestrator.py` + 2 测试 | `test_orchestrator_history.py` / `test_orchestrator.py` (B1 part) |
| 6 | Config B1 改造（解禁 `"composite"` + 6 项校验 + 第 7 项 state-library 校验 + `min_required_top_k` wiring） | `config.py` + 测试 | `test_config_factor.py` (B1 part) |
| 7 | CompositeJudge end-to-end 测试 | `test_judge.py` (B1 part) 扩充 | 全跑通 |
| 8 | 文档 B1 同步 | `docs/architecture/...` + `docs/cache/...` + `docs/cache/tutorial.md` + 两 README + log status | grep status 标识更新 |
| **B1 提交点** | — | — | **commit 1**：B1 算法 + Orchestrator + Config + tests + docs |
| 9 | `LibraryStats.compute_from_entries` body 填实 | `base.py` + 测试 | `test_base.py` |
| 10 | F1b `extract` + `compute_for_episode` body 填实 | `source_window.py` + 测试 | `test_source_window.py` |
| 11 | InMemoryBackend `load_artifact` library_stats fallback | `in_memory_backend.py` + 测试 | `test_cache_storage.py` (B2 part) |
| 12 | Orchestrator B2 **实现**（`__init__` 加 `offline_writers` / `library_stats` + `_build_entry_chain` merge 循环） | `orchestrator.py` + 测试 | `test_orchestrator.py` (B2 part) |
| 12b | Config B2 实现（`collect_offline_writers_from_judges` helper + `build_per_connection_components` 喂给 Orchestrator） | `config.py` + 测试 | `test_config_factor.py` (B2 part) |
| 13 | `factor_postprocess.py` helper + 测试（torch-only 输入） | 新建 + `test_factor_postprocess.py` | 全跑通 |
| 14 | 三 `build_*.py` 加 `--factors-yaml` CLI + 在 `build_artifact` 末尾（entries 已 detach 为 numpy）调 `enrich_artifact_with_factors` + 写 `library_stats` 字段到 artifact dict（**不动** `_detach_entries` lifecycle，option 2） | `exp/common/build_*.py` | 主路径手测一份 minimal artifact 跑全程：CLI → 子进程 detach → 主进程 enrichment（option 2 桥接 numpy）→ pickle → load → factors 字段非空 |
| 15 | Artifact round-trip 测试（覆盖 detach lifecycle —— 用真 builder 路径写一份小 artifact） | `test_artifact_roundtrip.py` | 全跑通 |
| 16 | 文档 B2 同步：架构 doc 升 §5.12；cache 教程 §3.5（F2 改 candidate-local mask）+ §5（builder 流程更新）；experiments/artifact_layout.md；两 README；log status | `docs/architecture/...` + `docs/cache/...` + `docs/experiments/...` + 两 README + log status | grep status 标识更新 |
| **B2 提交点** | — | — | **commit 2**：B2 算法 + Backend + Builder + tests + docs |

> 顺序原则：每一步交付都自包含 + pytest 全绿；B1 与 B2 之间允许独立通过 G2，但本 plan 整体作为**一个 G1 审批单元**走（"两个一起做了"= 一份 plan，两次 code+G2 闭环可接受）。

> **G2 选项**：执行时若发现 B1 与 B2 改动面较大，可向 user 申请拆为 G2-B1 / G2-B2 两轮 review；默认按一轮 G2 走。

---

## 8. 风险登记（执行级）

| 风险 | 触发场景 | 缓解 |
|---|---|---|
| F1a/F1b 描述子在跨 episode 边界 / 静止区段产生大量 NaN，污染 normalizer 滚窗 | 窗口大、机器人静置时段长 | normalizer 内 `force_miss` 模式跳 NaN（不 enqueue），不污染滚窗；测试用静置 fixture 验证 |
| F1a-T 调 `view.walk_next` 触碰未实现的 fork | winner entry 在 chain 上有 fork（B0 plan §2.3.2 raise NotImplementedError） | 当前数据集 episode chain 无 fork（trajectory_id 唯一）；F1a-T `extract` 加 try/except 包 `walk_next`，捕到 NotImplementedError → 该窗口 NaN；测试覆盖 |
| `_descriptor_kernel` 与 §2.8.4 公式漂移（F1a 与 F1b 实现走神） | 两处 import 同一 helper 但参数顺序传错 | helper 单元测试用合成数据 lock 数值；F1a / F1b extract 测试都比对相同合成序列的输出一致 |
| Composer `bind_orientations` 校验 `directions` 漏配的 non_monotonic key | non_monotonic key 无 weight 也不出现在 directions → 不校验 | 校验只对 `weight != 0` 的 key 触发；weight=0 的 key 跳过 directions 校验（合理：不进 weighted_sum）；测试覆盖 |
| Normalizer `bind_keys` 后 extractor key drift | extractor 实现 bug 返回新 key | CompositeJudge 已有 key contract assert（§8.2.2 ship 在 B0），任何 drift 立即 RuntimeError |
| `LibraryStats.compute_from_entries` 在大 artifact 启动时阻塞 | 50k+ entry 的老 artifact 加载时间长 | log warning + 时间统计；新 build 默认填 library_stats；老 artifact 一次性 fallback 后下次 rebuild 即解 |
| `load_artifact` 字段缺失检测错（pickled None vs 缺字段） | `data.get("library_stats")` 在新格式下也可能是 None（writer 列表为空但 stats 已算） | `data.get("library_stats")` 区分 missing vs None：用 `"library_stats" in data` 做存在性检查；不在则 fallback；在但 None 则当作"用户主动跳过"，F1b 在线读 NaN |
| Builder helper 的 `--factors-yaml` 配 F1a / F2（无 OfflineWriter surface） | 用户 YAML 误把 F1a 写成 OfflineWriter | `_load_offline_writers_from_yaml` 校验 `cls` 必须 `hasattr("compute_for_episode")`；缺则 raise ConfigValidationError，明示"F1a / F2 only run online" |
| 三 build_*.py 加 CLI 时与现有 `--config` / `--output` 等 arg 冲突 | argparse 命名重叠 | 走完整个三脚本读现有 argparse 定义后再加；测试只验证 helper 不验证脚本（脚本无 unit test 历史） |
| Orchestrator `on_task_end` 显式 reset 是 pre-existing bug fix，可能改变其他依赖 buffer 残留的代码路径 | 现有调用方依赖 leak | 项目内 grep 无依赖；leak 仅当 WebSocket 中断且无 episode_end 时发生，正常路径 episode_end 已清；测试覆盖 |
| `min_required_top_k` 反向喂回 strategy 与现有 `top_k` 字段语义冲突 | 老 YAML `top_k=1` + composite F2(K=5) 的同 CP | B0 plan §2.7 已定：strategy 内取 `max(top_k, min_top_k_hint)`；老 YAML（无 composite judge）`min_top_k_hint=0` → `max(1,0)=1` 字节级一致；测试覆盖 |
| 新增 `_descriptor_kernel.py` 与 `factor_postprocess.py` 没进 `__init__.py` 导出 → import 路径混乱 | 新模块用相对 import 失败 | 严格用绝对 path `openpi.cache.components.factors._descriptor_kernel`；CI lint 抓 |
| Builder lifecycle: `_detach_entries` 在 `_process_episode*` 内部就把 tensor → numpy；enrichment 跑在 numpy 上需要桥接 | 主 builder 路径 enrichment 跑 numpy `torch.stack` 直接炸 | §4.5 选 option 2 — `_detach_entries` lifecycle 不动（保持 ProcessPool IPC 优化），`LibraryStats.compute_from_entries` / OfflineWriter `_extract_episode_seq` / helper 内部全部用 `torch.as_tensor(...)` 桥接 numpy 或 torch；`enrich_artifact_with_factors` docstring 明示双输入兼容；测试覆盖两种路径 |
| F2 在 docs §3.5 与代码 `requires_library_stats=False` 不一致 | 实现按 docs 用 `library_stats.action_active_mask` → 必须改 `requires_library_stats=True`，validator 行为变化 | §3.2 选择 candidate-local mask（var_d > 1e-8），保持 `requires_library_stats=False`，docs §3.5 同步修订；测试 `test_consensus.py` 验证 padded DOF 自动被剔除 |
| `state_sigma` / `state_active_mask` dataclass 类型非 Optional，state 缺失场景需要明确占位 | F1a-T / F1b-T 撞到 state 缺失 entry pool | §4.2 用 zeros 占位 + active_mask 全 False；`_descriptor_kernel` 入口 check `active_mask.sum() == 0` 直接返 NaN；测试 `test_base.py` 覆盖空 state 路径，`test_runtime_continuity.py` / `test_source_window.py` 覆盖 zero-mask 路径 |
| State-side state 访问 / z-score 在 `_descriptor_kernel` mask check 之前发生 | F1a-T 直接 `winner.query_keys["robot_state"]` → KeyError；F1b-T `_extract_episode_seq` 同；都在 mask check 前已 z-score 0/0 / shape 错配 | 三层 fail-safe（§3.1）：(a) config 层第 7 项校验 reject 空 state library + composite f1a_t/f1b_t；(b) extract / compute_for_episode 入口 `if source=="state" and (sigma.numel()==0 or active_mask.sum()==0): return all-NaN`，**早于** state 访问与 z-score；(c) per-entry `query_keys.get("robot_state")` 而非 `[]`，缺失即返 all-NaN；测试 `test_runtime_continuity.py` / `test_source_window.py` / `test_config_factor.py` 三层各覆盖一例 |
| `step_idx is None` 的 entry 在 OfflineWriter 内 sort 触发 TypeError | builder 输入未给 step_idx 的 entry 列表（极少见但合法） | §4.5.1 helper 把 step_idx=None 的 entry 拉到独立 unnamed 桶按 collection order 排在末尾 + 写 warning；测试 `test_factor_postprocess.py` 覆盖 mixed `step_idx` 桶（含 None） |

---

## 9. 验收标准

B1+B2 plan 关闭时（即 G2 APPROVED + Verify pass）必须满足：

1. `uv run pytest tests/cache/` 全部通过（包括 B0 已落地的 561 项 + B1+B2 新增 ≥ 30 项）
2. 任意一份新写的 composite YAML（含 F1a-A + F1a-T + F1b-A + F1b-T + F2 全因子）能：
   - 通过 `validate_cache_config`
   - 在跑 `policy_server` 启动时不 crash
   - 第一个 inference cycle 产出 verdict（FULL_HIT / WARM_START / MISS 任一，**不**全 NaN 短路）
3. 任意一份只配 F1b 的 minimal YAML 通过 `--factors-yaml` 喂给 `build_in_memory_cache_artifact.py` → 产出的 pkl 含 `library_stats` 字段且 entries 含 `payload.factors`
4. 老 artifact pkl（无 `library_stats` / `payload.factors`）加载后 server 启动 log 出现 fallback 警告，judge 路径不 crash（F1b 走 NaN）
5. 所有 doc status 标识从 B1/B2 → Implemented；index 同步；`logs/verdict_factor_judge_b1_b2.log.md` 标 `Validated`

---

## 10. 文档与 Index 同步交付物

按 execution_authority §9 + WA §4 强制：

| 触发 | 同步动作 |
|---|---|
| `docs/architecture/cache_system.md` 改 | `docs/README.md` architecture/ 表对应行同步描述 |
| `docs/cache/verdict_factor_judge.md` + `.en.md` 改 | `docs/README.md` cache/ 表行不变（已有 entry），但描述里去掉 B0/B1/B2 status 提及（合并落地） |
| `docs/cache/tutorial.md` 改 | `docs/README.md` cache/ 表行同步 |
| `docs/experiments/artifact_layout.md` 改 | `docs/README.md` experiments/ 表行同步 |
| 本 plan log 状态变化 | `logs/README.md` 行同步：Plan → In Progress（开 code 时）→ Implemented（commit 后）→ Validated（Verify pass 后） |

---

## 11. 与 21-window × 6-pkl 实验的关系

本 plan 完成后，**才有可能**做用户最初提的"21 个窗口 × 6 个 pkl 的离线因子 artifact 重建"。具体执行 sequence：

1. 本 plan APPROVED + Verified + Committed
2. 用户启动一个**新实验 plan**（独立 log，本 plan 不涉及）：选定 21 个 (past, future) 窗口 + 4 描述子 + 2 因子族 → minimal YAML
3. 跑 `build_in_memory_cache_artifact.py --factors-yaml ...` × 6 次（一次一个 KeyBuilder）
4. 结果分析 + 决定 Composer weights + tier thresholds
5. 是否同时清理 `libero_spatial_warm/` 视 warm_start 实验是否仍活跃

本 plan 不预设 21 窗口的具体 `(past, future)` 取值；helper 与 builder 对窗口数量无硬上限（仅受存储成本 §2.8.6 估算约束 ~100 字节/entry × 21 ≈ 2KB/entry，可控）。

---

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-04-26 21:58 CDT

- [Blocking] [Concern] The B2 OfflineWriter path is not wired into the actual policy-server Orchestrator construction — reasoning: `build_per_connection_components` now returns `offline_writers` and `library_stats`, but both `scripts/serve_policy.py` construction branches still instantiate `CacheOrchestrator` with only storage/key_builder/gates/judges/search_strategies/timer/write_policy. Because `CacheOrchestrator.__init__` defaults those new args to `()` / `None`, the `_build_entry_chain` merge loop is skipped in normal server execution even when composite F1b factors are configured. This contradicts §7 step 12b ("build_per_connection_components 喂给 Orchestrator") and makes the docs claim "config 收集 writer 喂给 Orchestrator" false. Pass `components["offline_writers"]` and `components["library_stats"]` through every production Orchestrator assembly path, including dynamic cache bundles and `--cache_config`, and add a config-to-Orchestrator assembly test that fails on the current code.
- [Blocking] [Concern] The composite warm-start validator implements only tier ordering, not the full approved 5a-5d rule set — reasoning: §3.6 requires CP1-only warm start, canonical `warm_start_t`, pairwise presence of `tier_thresholds.warm_start` and `composer.warm_start_t`, and weighted-sum tier ordering. Current `src/openpi/cache/config.py` checks only `warm_start < full_hit`; an independent G2 probe confirmed that `load_cache_config` accepts `warm_start_t` without `warm_start`, `warm_start` without `warm_start_t`, CP3 composite warm-start config, and non-canonical `warm_start_t=0.55`. Implement the missing validation branches and add regression tests for all four invalid cases.
- [Blocking] [Concern] The docs/index updates currently overstate the implemented behavior — reasoning: `docs/cache/verdict_factor_judge.md` says all 7 composite checks are active and "config 收集 writer 喂给 Orchestrator" / end-to-end composite is usable, but the two blocking issues above make those statements false. `logs/README.md` also still describes "config 解禁 composite + 6 项校验激活" for this B1+B2 plan while the approved plan requires 7 checks. Update docs and indexes after the code fixes so the status text reflects the actual shipped behavior.

Checklist assessment:

- Consistency with approved plan: NEEDS REVISION. Core algorithm bodies are broadly present, but the approved B2 config-to-Orchestrator feed is missing from production assembly, and the approved composite warm-start validator is only partially implemented.
- Test coverage and passing: NEEDS REVISION. `PYTHONPATH=. uv run pytest tests/cache -q` passes (699 passed, 4 skipped), and the targeted existing subset passes (58 passed), but independent G2 probing exposes accepted invalid YAMLs and no test covers the production Orchestrator assembly path that drops `offline_writers` / `library_stats`.
- Docs & indexes updated: NEEDS REVISION. Docs and indexes were touched, but several statements now claim behavior that the current code does not provide.
- No regressions: NEEDS REVISION. Existing tests pass, but the server path would silently skip B2 online factor enrichment for newly written episodes, and invalid warm-start composite YAMLs can reach runtime.

NEEDS REVISION: wire B2 Orchestrator args through production assembly; complete composite warm-start validation and tests; correct docs/index status.

### G2 Round 2 — Executor — 2026-04-26

3 项 reviewer blocking 全部核查命中，全部 Accepted：

- **Accepted (Item 1) — B2 OfflineWriter args wired through both production assembly paths in `scripts/serve_policy.py`**
  事实核查：`scripts/serve_policy.py:267-275` (cache-bundle path) 与 `:313-321` (`--cache_config` path) 都构造 `CacheOrchestrator(...)` 但只传 7 个参数；新加的 `offline_writers` / `library_stats` 默认 `()` / `None` → `_build_entry_chain` merge loop 永远不触发。reviewer 完全正确。
  修订：两处 `CacheOrchestrator(...)` 调用各加 `offline_writers=components.get("offline_writers", ())` + `library_stats=components.get("library_stats")`。新增 `tests/cache/test_config_factor.py::test_production_assembly_passes_offline_writers_and_library_stats`，构造 minimal artifact + composite F1b YAML，断言 `build_per_connection_components` 返回的 dict 含 `offline_writers` / `library_stats` 两键且 `offline_writers` 非空 + `compute_for_episode` 可调；等价覆盖 reviewer 提的"a config-to-Orchestrator assembly test that fails on the current code"。
- **Accepted (Item 2) — Composite warm-start validator 补完 5a-5d 全部 4 条子规则**
  事实核查：`config.py:578-591`（修订前）只检查 `tier_thresholds.warm_start < tier_thresholds.full_hit`；R0 plan §3.6 / 5a-5d 要求 (5a) `composer.warm_start_t` 必须是 `CANONICAL_DENOISE_TIMESTEPS` 值 / (5b) `tier_thresholds.warm_start` 与 `composer.warm_start_t` 必须同时出现 / (5c) warm-start composite 仅 CP1 / (5d) 已有。reviewer 给的 4 个未挡场景（warm_start_t 没 warm tier / warm tier 没 warm_start_t / CP3 composite warm-start / `warm_start_t=0.55`）确实全可通过当前 validator。
  修订：`_validate_composite_judge_static` 加 cp_name 入参；新增 5a (canonical timestep + 归一化 writeback) + 5b (pairwise) + 5c (CP1-only) 三个分支，5d 已有保留。call site 同步传 `cp_name`。新增 4 个回归测试 `test_validator_5a_warm_start_t_must_be_canonical` / `test_validator_5a_warm_start_t_canonical_value_accepted`（含 normalized writeback）/ `test_validator_5b_pairwise_tier_warm_without_warm_start_t` / `test_validator_5b_pairwise_warm_start_t_without_tier_warm` / `test_validator_5c_warm_start_only_on_cp1`，5 项全 reviewer 列出的 invalid 用例都覆盖。
- **Accepted (Item 3) — docs / index 文案修正以反映实际行为**
  事实核查：`logs/README.md` 的 verdict_factor_judge_b1_b2 行写"6 项校验激活"应为 7 项；`docs/cache/verdict_factor_judge.md` + `.en.md` 状态 banner 写"7 项 composite-specific 校验"但没明示 5a-5d 子规则；都没提 `scripts/serve_policy.py` 两条 production assembly 路径已喂 Orchestrator。
  修订：`logs/README.md` "6 项校验" → "7 项校验激活（含 5a-5d warm_start 子规则）"；`docs/cache/verdict_factor_judge.md` + `.en.md` 状态 banner 在 7 项校验后加括号详列 5a-5d 子项 + B2 段加"`scripts/serve_policy.py` 两条 production assembly 路径喂 Orchestrator"明示。

修订增量：`scripts/serve_policy.py` 2 处 `CacheOrchestrator(...)` 调用各加 2 kwarg + `src/openpi/cache/config.py` `_validate_composite_judge_static` 加 cp_name 入参 + 5a/5b/5c 三个分支 + call site 同步 + `tests/cache/test_config_factor.py` 新增 6 个测试（4 invalid 用例 + 1 normalized writeback positive 用例 + 1 production assembly smoke）+ `logs/README.md` / `docs/cache/verdict_factor_judge.md` / `.en.md` 文案修订。

最终回归：`uv run pytest tests/cache/` → **705 passed, 4 skipped, 0 failed**（B1 baseline 648 → R1 后 699 → R2 后 705，纯增量）。

回 G2 等审。

### G2 Round 2 — Reviewer — APPROVED — 2026-04-26 22:07 CDT

Checklist assessment:

- Consistency with approved plan: APPROVED. The Round 1 B2 production wiring gap is closed: both `scripts/serve_policy.py` Orchestrator construction paths now forward `offline_writers` and `library_stats`, so `build_per_connection_components` output can reach `_build_entry_chain` in normal server execution. The composite warm-start validator now covers canonical timestep, CP1-only emission, pairwise `warm_start` / `warm_start_t`, and tier ordering.
- Test coverage and passing: APPROVED. Independent G2 probes confirmed the four previously accepted invalid warm-start YAMLs are now rejected. `PYTHONPATH=. .venv/bin/python -m pytest tests/cache/test_config_factor.py -q` passed (36 passed), and `PYTHONPATH=. .venv/bin/python -m pytest tests/cache -q` passed (705 passed, 4 skipped). The added config tests cover the validator regressions and the B2 component bundle surface used by production assembly.
- Docs & indexes updated: APPROVED. `logs/README.md` now reflects 7 composite checks including 5a-5d warm-start subrules, and the verdict-factor docs now state that `scripts/serve_policy.py` forwards the B2 writer/stat objects into Orchestrator. The remaining "6 static checks" wording refers to config-load static checks, with the 7th state-library check still correctly described as build-time/runtime-dependent in the plan.
- No regressions: APPROVED. Existing cache tests remain green, the production assembly no longer drops B2 writer/stat state, and invalid composite warm-start YAML no longer reaches runtime.

APPROVED
