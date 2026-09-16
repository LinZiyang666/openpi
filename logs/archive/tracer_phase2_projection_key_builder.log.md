# TRACER Phase 2 — M1 结果兼容投影 KeyBuilder 骨架（identity 默认）

- **Status**: Implemented（G1 R2 + G2 R2 APPROVED + §6 Verify green 1018/6，2026-07-08；待 owner 确认归档）
- **Date**: 2026-07-07（创建）/ 2026-07-08（G1 通过，进入 Code）
- **Level**: L2（新组件 + 2 处工厂分支 + 单测；零框架手术、零 verdict、零训练执行）
- **Authority**: Execution
- **上位依据**: [`tracer_retrieval_refinement_roadmap.log.md`](tracer_retrieval_refinement_roadmap.log.md) §4 Phase 2。本 plan 是该期的独立立项，须自走 Understand→Plan→G1→Code→G2→Verify。
- **前置**: 无（Phase 2 与 Phase 1/3 独立并行；Phase 1 `dynamic_depth_knn` 已 commit `cea98b2`）。
- **出场 gate（roadmap）**: identity golden 通过 + 带权重时投影 shape/维度自洽。

---

## 0. Context for G1 reviewer（无需对话历史）

合作者提案 *Action-Compatible Failure-Aware Retrieval*（PDF 在 `exp/zixuan_proposal/`）把 3 个机制装进本 fork 的推理 cache：M1 结果兼容投影 / M2 失败感知双检索 / M3 动态链深。roadmap 把它们拆成 7 期，**机制(代码) 与 参数(训练值) 分离**、**惰性默认退化到现系统**（可证非回归）、**训练逻辑内聚但执行离线**。

本期 = **M1 骨架**：提案对每模态学一个投影头 `z = h_θ(x)`（Eq 8–9），用投影后的余弦相似度替代原始 pool 特征的余弦。工程落法：新 `QueryKeyBuilder` 子类，在现有 pool builder 输出之后对**余弦字段**过一层线性投影；**无权重时 identity**（跳过投影，逐值 == 现 pool 输出）。库侧建库与在线查询用**同一个** builder 类 → 两侧过同一头 → backend 仍普通 cosine（零 backend 改动）。

**本期不训练**。类上 `fit()` 提供投影拟合机制（InfoNCE），但只吃"已备好的训练张量"、只用合成小数据单测证机制；**真实 payload→兼容标签构造 + 离线 driver + 重跑 = Phase 6**（owner 已确认此边界，见 §7 R1）。

---

## 1. 目标 / 非目标

**目标**：
1. 新 `ProjectionKeyBuilder`：包一个内层无状态 pool builder，`build()` 对余弦字段（vision_0/1/2、prompt_emb）过每模态线性投影 `z = xWᵀ(+b)`；**无投影参数 → identity**。`robot_state` 恒原样（L2 距离字段，禁投影）。
2. 投影参数 `ProjectionParams`：load/save + 序列化，照 `ScoreNormalizer.from_params_dict/fit_from_scores` 的 load/fit 分离范式（惰性默认 None → identity）。
3. 类上 `fit(...)`：InfoNCE 投影拟合**机制**，实现但本期不在真实库上跑（offline-only，Phase 6 接 driver）。
4. 在线工厂（`config._build_key_builder`）+ 离线库侧工厂（`build_in_memory_cache_artifact._create_builder`/`_get_vector_dims`）各加 `projection` 分支 → 两侧同一头。
5. 单测：identity 逐值非回归 golden + 带权重维度自洽 + fit 机制有效性 + 参数往返 + config/库侧分支。

**非目标**（本期不做，留后期）：
- 真实投影权重的训练与产出（Phase 6，且 Claim 2 necessity-check 条件触发）。
- 从 `CachePayload` 算 `c^A/c^X` 兼容标签（Phase 6）。
- 投影 `robot_state` 或改其 L2 语义。
- 包裹有状态（`cp1_temporal_prune`）/ 需 attach_model（`cp1_llm_layer_extract`）/ CLIP-encoder（`clip`）的 builder——本期内层只限无状态 pool（见 §7 R4）。
- 任何 orchestrator / interceptor / backend / judge 改动。

---

## 2. 已验证架构事实（亲验 + 锚点）

| # | 事实 | 锚点 |
|---|---|---|
| F1 | `QueryKeyBuilder` Protocol = `collect(checkpoint_id, **stage_outputs)->None` / `build(checkpoint_id)->dict[str,Tensor]`（每字段 `[dim]` CPU float32 contiguous）/ `cached_data` property / `clear()` | `components/key_builder.py:31-74` |
| F2 | 无状态 pool 系 `_CP1BaseKeyBuilder` 只 override `_reduce_vision/_reduce_prompt`；`robot_state` 恒原样（无 normalize/pool，因 backend 需真 L2 距离）；子类 `CP1MeanPoolKeyBuilder`/`CP1MaxPoolKeyBuilder`/`CP1SpatialPool16KeyBuilder`/`CP1SpatialPool4KeyBuilder` | `key_builder.py:226-351` |
| F3 | 在线工厂 `_build_key_builder(cfg, enabled_fields, vector_dims)` = if/elif on `cfg.type`；类型校验 frozenset `_valid_key_builder_types` | `config.py:2282-2341` / `config.py:1236-1249` |
| F4 | 离线库侧工厂 `_create_builder(builder_type, ...)` + `_get_vector_dims(builder_type, ...)`，与在线各自独立但用**同一批 KeyBuilder 类**；`_process_episode` 里 `builder.collect(cp,stage1=fake)` → `query_keys=builder.build(cp)` → `CacheEntry(query_keys=...)` | `exp/common/build_in_memory_cache_artifact.py:138-204 / 108-130 / 543-635` |
| F5 | 无状态 pool builder 各字段 out_dim：mean/max = vision 2048 / prompt 2048 / state 32；spatial_16 = vision 32768；spatial_4 = vision 8192 | `build_in_memory_cache_artifact.py:46-54` |
| F6 | D3 范式：`ScoreNormalizer` 有 `params_to_dict()` + `@classmethod from_params_dict(params,sim_type)` + `@classmethod fit_from_scores(scores,...)`；运行时只 `from_params_dict` 加载，`fit_*` 离线 | `components/score_normalizers.py:95-112, 438-450` |
| F7 | config 嵌套 dataclass 解析：字段类型经 `_resolve_type` 命中 `_CONFIG_TYPES` 才被 `_dict_to_dataclass` 递归（否则存 dict）；Phase 1 `DepthPolicyConfig` 即此先例 | `config.py:527-556, 559-577, 656-658` |
| F8 | 兼容标签数据源 `CachePayload.action_chunk[50,32]` / `.intermediates{t:x_t}`——Phase 6 用，非本期 | `storage_types.py:101-105` |
| F9 | KeyBuilder 单测约定：`_FakeStage1`（SimpleNamespace 持 `prefix_embs/state`）+ `torch.allclose` parity；库侧脚本测在 `tests/exp/test_build_in_memory_cache_artifact.py` | `tests/cache/components/test_key_builder_cp1_experiment.py` |

**F4 是本期的地基**：库侧与在线共用同一 KeyBuilder 类。只要两个工厂都构造 `ProjectionKeyBuilder`，库 key 与查询 key 都过同一头；identity 默认下两侧都退化成内层 pool → backend cosine 逐值不变。

---

## 3. 设计

### 3.1 `ProjectionKeyBuilder`（新，`components/projection_key_builder.py`）

组合式包裹——**不改内层 builder 一字节**：

```python
_PROJECTED_FIELDS = frozenset({VISION_0, VISION_1, VISION_2, PROMPT_EMB})  # cosine 字段；robot_state 不投影

class ProjectionKeyBuilder:
    def __init__(self, inner: QueryKeyBuilder, params: ProjectionParams | None = None): ...
    def collect(self, checkpoint_id, **stage_outputs) -> None:      # 透传 inner
        self._inner.collect(checkpoint_id, **stage_outputs)
    def build(self, checkpoint_id) -> dict[str, torch.Tensor]:
        keys = self._inner.build(checkpoint_id)
        if self._params is None:
            return keys                                              # identity：原样返回（逐值 == inner）
        out = {}
        for field, vec in keys.items():
            head = self._params.head(field) if field in _PROJECTED_FIELDS else None
            out[field] = vec if head is None else _project(vec, head)
        return out
    @property
    def cached_data(self): return self._inner.cached_data
    def clear(self): self._inner.clear()
    def on_episode_start(self):                                      # 防御式透传（内层无状态则 no-op）
        fn = getattr(self._inner, "on_episode_start", None)
        if fn is not None: fn()
```

- `_project(vec, head)`：`z = vec @ head.weight.T`（`weight:[out_dim,in_dim]`），有 bias 则加，`.cpu().float().contiguous()`。在 inner 的 CPU 输出上做（identity 时是无成本旁路；GPU-前置优化押后，见 §7 R6）。
- **identity 契约（关键）**：`params is None` 时 `build()` 返回 `inner.build()` 的**同一对象**，逐值/逐字节等价——不合成单位矩阵（避免 float 往返误差）。
- **余弦字段限定**：只投影 `_PROJECTED_FIELDS`；`robot_state` 及任何未在 params 中提供 head 的字段一律透传（保 L2 语义）。

### 3.2 `ProjectionParams` / `ProjectionHead`（同文件）

```python
@dataclass(frozen=True)
class ProjectionHead:
    weight: torch.Tensor            # [out_dim, in_dim] CPU float32
    bias: torch.Tensor | None = None  # [out_dim] CPU float32 or None

@dataclass(frozen=True)
class ProjectionParams:
    heads: dict[str, ProjectionHead]        # {field: head}；缺字段 → 该字段 identity
    def head(self, field) -> ProjectionHead | None: return self.heads.get(field)
    def to_state_dict(self) -> dict: ...                 # {field: {"weight":W, "bias":b?}}
    @classmethod
    def from_state_dict(cls, sd) -> "ProjectionParams": ...
    def save(self, path) -> None:  torch.save(self.to_state_dict(), path)
    @classmethod
    def load(cls, path) -> "ProjectionParams":  return cls.from_state_dict(torch.load(path, map_location="cpu"))
```

- 镜像 D3 的 serialize/deserialize + 离线 fit 分离；**偏离点**：参数是张量矩阵而非标量 → 用 `torch.save/load`（非 `ScoreNormalizer` 的 JSON），见 §7 R3。
- 加载即 fail-loud：head 完整形状（out_dim / in_dim / bias）由 §3.5 的 `validate_projection_params` 校验，在线/离线两工厂构造 builder 前**各调一次**。

### 3.3 `fit()`（同文件，offline-only，本期不跑）

```python
@dataclass
class FieldTrainingBatch:
    features: torch.Tensor      # [N, in_dim] pooled 特征
    group_labels: torch.Tensor  # [N] int：同 id = 兼容（正样本）

@classmethod
def fit(cls, batches: dict[str, FieldTrainingBatch], *, out_dim: int,
        epochs: int = 200, lr: float = 1e-2, temperature: float = 0.07) -> ProjectionParams:
    """Fit per-field linear projection heads by InfoNCE over PREPARED batches.

    OFFLINE ONLY. Phase 2 ships this mechanism; Phase 6 adds the payload->
    compatibility-label construction (c^A/c^X from action_chunk/intermediates)
    and the offline driver that persists weights. `batches` is already-prepared
    training signal (per-field pooled features + a compatibility grouping),
    NOT raw CachePayloads.
    """
```

- 每字段独立拟合一个 `[out_dim, in_dim]` 线性头：投影后 L2-normalize，按 `group_labels` 建 InfoNCE 正/负（同组拉近、异组推远），Adam 优化。host VLA / action expert 全程不涉（本期无模型）。
- 边界（owner 确认，§7 R1）：`fit` 只吃 `FieldTrainingBatch`；从真实 `CachePayload` 构造 batch（兼容标签）、held-out、重跑、冻权入 artifact —— **全在 Phase 6**。
- 本期仅**合成数据单测**证机制（loss 下降 + 输出 shape），不接任何 driver、不碰真实库。

### 3.4 config 挂载（`config.py`）

1. 新 `ProjectionKeyBuilderConfig`（嵌套，注册进 `_CONFIG_TYPES`，照 F7）：
   ```python
   @dataclass
   class ProjectionKeyBuilderConfig:
       inner_type: str = "cp1_mean_pool"    # 被包的无状态 pool builder
       weights_path: str | None = None      # 投影 artifact；None → identity
   ```
2. `KeyBuilderConfig` 加字段 `projection: ProjectionKeyBuilderConfig = field(default_factory=ProjectionKeyBuilderConfig)`。
3. `_valid_key_builder_types` 加 `"projection"`。
4. `_build_key_builder` 加分支：
   ```python
   elif cfg.type == "projection":
       from openpi.cache.components.projection_key_builder import (
           ProjectionKeyBuilder, ProjectionParams, validate_projection_params)
       inner_cfg = replace(cfg, type=cfg.projection.inner_type)   # 复用现有 pool 构造
       inner = _build_key_builder(inner_cfg, enabled_fields, vector_dims)
       params = ProjectionParams.load(cfg.projection.weights_path) if cfg.projection.weights_path else None
       validate_projection_params(params, cfg.projection.inner_type, vector_dims)  # load-time fail-loud
       return ProjectionKeyBuilder(inner, params)
   ```
   （`replace` from dataclasses；`inner_cfg.type` 已被 §3.5 校验限定在无状态 pool 集且非 `projection`，无限递归被挡。）

### 3.5 校验（`config.py` validator 段 + 加载期 `validate_projection_params`）

**inner_type 集**（config validator 段）：`cfg.projection.inner_type ∈ {cp1_mean_pool, cp1_max_pool, cp1_spatial_pool_16, cp1_spatial_pool_4, cp1_spatial_pool_64}`（无状态 pool 集），否则 fail-loud；显式拒 `projection`（防递归，R7）。

**投影权重完整形状校验（G1 R2 finding 1）**：新 `validate_projection_params(params, inner_type, vector_dims)`（projection 模块内），在线 `_build_key_builder` 与离线 `_create_builder` 构造 `ProjectionKeyBuilder` 前**各调一次**，load-time fail-loud。内层每字段输出维度由 `_pool_inner_dim(inner_type, field)` 给出（src 侧单一来源，按 pool 语义推：`robot_state→32`；`prompt_emb→2048`；`vision_*→{mean/max:2048, spatial_16:32768, spatial_4/_64:8192}[inner_type]`；注释交叉引用 exp `_VECTOR_DIMS`；函数式而非静态表，故对 `vision_2` 等任意 vision 字段与不同 `enabled_fields` 都自洽）。契约（遍历 backend `vector_dims` 的字段）：
- 字段**有 head**：`head.weight.shape == (vector_dims[field], _pool_inner_dim(inner_type, field))`（out_dim 对 backend、in_dim 对内层）；`head.bias is None or head.bias.shape == (vector_dims[field],)`；且 field 必须是可投影余弦字段（`_PROJECTED_FIELDS`），否则拒。
- 字段**无 head**（透传）：要求 `vector_dims[field] == _pool_inner_dim(inner_type, field)`（透传不得改存储维度）。
- head 提供给**不在 vector_dims / robot_state / 未知字段**：拒（misconfig，fail-loud，错误信息含 field + 期望/实际 shape）。
- identity（`params is None`）：跳过 head 校验（无投影）。

**backstop**：`_project(vec, head)` 内额外断言 `head.weight.shape[1] == vec.shape[0]`（对真实内层输出），万一表与实际漂移也给清晰 field 级错误，而非裸 matmul 崩。`weights_path` 非空但文件不可 load 亦 fail-loud。

### 3.6 库侧工厂（`exp/common/build_in_memory_cache_artifact.py`）——参数全链贯通（G1 R2 finding 2）

`inner_type` / `projection_weights_path` 必须穿过**整条**建库管线（否则 CLI/顶层 API 值到不了 worker）。已亲验的链路 + 改动点：
1. **`build_artifact(...)`**（顶层 API，测试与调用方入口，line 678-694）：加形参 `inner_type="cp1_mean_pool"`, `projection_weights_path=None`；传入 `_get_vector_dims(...)`（line 708-710）；纳入 `_ep_args`。
2. **`_ep_args` 元组**（line 779-783）：末尾追加 `inner_type, projection_weights_path` → serial（`_process_episode(str(p), *_ep_args)`, line 791）与 ProcessPool（`pool.submit(_process_episode, str(p), *_ep_args)`, line 803）两路**自动覆盖**（都摊 `_ep_args`）。
3. **`_process_episode(...)`**（worker，line 543-553）：加末尾 2 形参，透传给 `_create_builder(...)`（line 560-564）。
4. **`_create_builder(...)`**（line 138-148）：加末尾 2 形参 + `projection` 分支：`inner_cls = builders.get(inner_type)`（**builders dict 补 canonical `cp1_spatial_pool_4`**，见下 finding 3）；`inner_cls is None`→fail-loud（限无状态 pool 集、拒 temporal_prune/llm/clip/projection）；`params = ProjectionParams.load(projection_weights_path) if projection_weights_path else None`；`validate_projection_params(params, inner_type, _VECTOR_DIMS[inner_type])`；返回 `ProjectionKeyBuilder(inner_cls(), params)`。
5. **`_get_vector_dims(...)`**（line 108-113）：加末尾 2 形参 + `projection` 分支：identity→`_VECTOR_DIMS[inner_type]`；带权重→逐字段 `params.head(f).weight.shape[0] if params.head(f) else _VECTOR_DIMS[inner_type][f]`。
6. **artifact metadata**（line 816-821，仿 `reducer_params` provenance line 823-831）：`builder_type=="projection"` 时加 `artifact["projection_params"] = {"inner_type": inner_type, "projection_weights_path": str(projection_weights_path) if projection_weights_path else None}`（可追溯）。
7. **`main()` CLI**（line 908+）：`--inner-type` / `--projection-weights` → 传 `build_artifact`（薄；供 Phase 6 driver，本期测试也经真实顶层入口跑）。

**builders dict `_4`（G1 R2 finding 3）**：`_create_builder` 的 `builders`（line 156-161）只有 legacy `cp1_spatial_pool_64`、**缺** canonical `cp1_spatial_pool_4`（而 `_VECTOR_DIMS` line 46-54 与在线 `_build_key_builder` line 2300 两者皆支持——离线是唯一缺口）。补 `"cp1_spatial_pool_4": CP1SpatialPool4KeyBuilder`（import 之；与 `_64` 同类 alias，key_builder.py:339）→ 纯 additive（既有 `_64`/其他调用逐字节不变），在线/离线 inner 面对称，`_4`/`_64` 皆可作 inner。

---

## 4. 文件触点

| 文件 | 动作 | 内容 |
|---|---|---|
| `src/openpi/cache/components/projection_key_builder.py` | **新增** | `ProjectionHead` / `ProjectionParams`（load/save/state_dict）/ `ProjectionKeyBuilder`（包内层、identity 默认）/ `_pool_inner_dim` + `validate_projection_params`（完整形状 fail-loud，§3.5）/ `FieldTrainingBatch` / `fit()`（offline 机制）。模块 docstring + 公开类/函数 docstring + 英文注释 + section 分隔线。 |
| `src/openpi/cache/config.py` | 改 | `ProjectionKeyBuilderConfig` + `_CONFIG_TYPES` 注册；`KeyBuilderConfig.projection` 字段；`_valid_key_builder_types` 加 `projection`；`_build_key_builder` 分支（含 `validate_projection_params` 调用）；validator 分支（inner_type 集）。 |
| `exp/common/build_in_memory_cache_artifact.py` | 改 | **全链贯通**（§3.6）：`build_artifact` / `_process_episode` / `_ep_args` / `_create_builder` / `_get_vector_dims` 各加 `inner_type`+`projection_weights_path`；`_create_builder`/`_get_vector_dims` 加 `projection` 分支；`builders` dict 补 canonical `cp1_spatial_pool_4`；metadata 加 `projection_params` provenance；`main()` 加 2 CLI 参数。 |
| `tests/cache/components/test_projection_key_builder.py` | **新增** | 见 §5：本地 builder + `fit` + 维度校验 fail-loud（测 1–7）。 |
| `tests/exp/test_build_in_memory_cache_artifact.py` | 改 | 库侧 `projection` 经**顶层 `build_artifact()`** 的 identity parity（`_4`/`_64` 皆测）+ `_get_vector_dims` 分支 + 加权两侧同头（测 8–9）。 |
| `tests/cache/test_config.py` | 改 | config `projection` 类型/inner_type 校验/工厂构造测试（测 10）。 |
| `docs/cache/tutorial.md` + `docs/cache/README.md` | 改 | KeyBuilder 段补 `projection` 行 + 索引 sync（WA §4）。 |
| `logs/README.md` | 改 | Active 加本 plan 条目（索引 sync）。 |

**零改动**：orchestrator / interceptor / backend / judge / 现有 KeyBuilder 类 / QuerySpec / storage_types。

---

## 5. 测试策略

**Blast radius（§6 Verify 口径，按 memory `reference_pytest_manual_skip` 裸 pytest、非 repo-wide、不碰 `tests/review_tests`）**：
`uv run pytest tests/cache/ tests/exp/test_build_in_memory_cache_artifact.py`
（触及 `src/openpi/cache/` + 库侧脚本，故并入其 exp 测试。）

新测（`tests/cache/components/test_projection_key_builder.py`）：
1. **identity golden（非回归核心）**：对每个 inner ∈ {mean_pool, max_pool, spatial_pool_16, spatial_pool_4}，`ProjectionKeyBuilder(inner, None).build(cp)` 对全字段 `torch.equal` == `inner.build(cp)`（同一 `_FakeStage1`），CP1+CP3 各测。
2. **带权重投影生效**：给 vision/prompt 合成 head（out_dim=64），断投影字段 shape `[64]`、`robot_state` 原样、CPU float32 contiguous。
3. **维度自洽（出场门）**：head out_dim → build 输出 dim == out_dim。
4. **robot_state 永不投影**：即便 params 里塞了 `robot_state` head，仍原样透传（不在 `_PROJECTED_FIELDS`）。
5. **fit 机制有效**：合成 2 组特征（近 u / 近 v）→ `fit()` → InfoNCE loss(fitted) < loss(identity-init)；head.weight shape `[out_dim, in_dim]`。
6. **参数往返**：`save`→`load`（tmp_path）后 heads `allclose`。

7. **维度校验 fail-loud（finding 1）**：`validate_projection_params` 对 out_dim≠vector_dims / in_dim≠`_pool_inner_dim` / bias 形状错 / robot_state 或未知字段带 head / 透传字段 vector_dims≠inner_dim —— 各造一反例断 `raise`；合法权重不抛。

改测（库侧 `tests/exp/test_build_in_memory_cache_artifact.py`）：
8. **库侧 identity parity 经顶层 `build_artifact()`（finding 2）**：`build_artifact(builder_type="projection", inner_type=X)` 的 entries.query_keys 逐值 == `build_artifact(builder_type=X)`（同 data-dir、serial `workers=-1`），对 X ∈ {cp1_mean_pool, **cp1_spatial_pool_4**, **cp1_spatial_pool_64**}（覆盖 finding 3 两名）；`_get_vector_dims("projection", inner_type=X)` == `_VECTOR_DIMS[X]`。
9. **加权两侧同头（finding 4）**：`ProjectionParams`（vision/prompt 合成 head，out_dim=64）`save` 到 tmp fixture → **在线** `_build_key_builder(KeyBuilderConfig(type="projection", projection=...(inner_type=X, weights_path=fixture)))` 与 **离线** `build_artifact(builder_type="projection", inner_type=X, projection_weights_path=fixture)` 各喂同一 `_FakeStage1`/同一 HDF5 step → 断：投影字段 dim == 64、未投影字段 dim == `_VECTOR_DIMS[X]`、robot_state 原样，且**在线投影 key 与离线投影 key 逐值相等**（证真·同头，非仅 identity）。

config 测（`tests/cache/test_config.py`）：
10. `projection` 类型 valid；非法 inner_type 拒；`inner_type="projection"` 拒（防递归）；`_build_key_builder` 造出 `ProjectionKeyBuilder` 且 identity build == 直接 pool build。

---

## 6. 退化契约（非回归，必证）

- **identity 逐值**：`weights_path=None` 时，`ProjectionKeyBuilder(inner_X)` 的 `build()` 输出对所有字段逐值 == `inner_X.build()`（测 1 本地、测 8 库侧顶层、测 10 config 工厂）。
- **两侧同头**：在线 `_build_key_builder` 与库侧 `build_artifact` 用同一 builder 类 → identity 时库/查询 key 都 == pool（测 8）；**带同一权重时**在线投影 key 逐值 == 离线投影 key（测 9）→ backend cosine 与现配置一致。
- **wire/装配零改**：无新 backend、无 QuerySpec 改动、无 orchestrator 触点 → 现有 golden（`tests/cache/`）全绿即证未回归。

---

## 7. 风险登记 / 开放问题

- **R1 Phase 2/6 `fit()` 边界 — 已定（owner 确认「机制版」）**：`fit()` 吃已备好的 `FieldTrainingBatch`（features + 兼容分组）做 InfoNCE，合成数据单测；真实 payload→`c^A/c^X` 兼容标签构造 + 离线 driver + 重跑 = Phase 6。签名因此从 roadmap 的 `fit(library)` 精化为 `fit(batches, *, out_dim, ...)`——因 Phase 6 明确拥有"训练数据构造"，此精化是该 Phase 边界的直接后果，非扩权。
- **R2 投影字段范围**：只投余弦字段（vision_*/prompt_emb）；`robot_state` 恒原样（L2 距离字段，投影会毁 `exp(-d/tau)` 语义，F2）。与提案 Eq 8–9（投影余弦）一致。
- **R3 参数序列化偏离**：投影是张量矩阵 → `torch.save/load`（非 `ScoreNormalizer` 的 JSON 标量）。方法名仍取 `to_state_dict/from_state_dict/save/load/fit` 对齐范式意图；load 时 fail-loud 校验维度。
- **R4 内层限定无状态 pool**：排除 `cp1_temporal_prune`（有 history + `on_episode_start`）、`cp1_llm_layer_extract`（须 `attach_model`）、`clip`（独立 encoder）——本期最小范围，避免状态/模型附着复杂度。`on_episode_start` 仍防御式透传（未来放开有状态内层时即用）。
- **R5 维度自洽（G1 R2 加固）**：`validate_projection_params`（§3.5）在两工厂 load-time 全面校验 out_dim==`vector_dims[field]`、in_dim==`_pool_inner_dim(inner_type,field)`、`bias==[out_dim]`、非投影字段透传 dim 一致；`_project` 再加 field 级 backstop 断言。identity 下跳过、dim 不变、现有 backend 直接可用。
- **R6 CPU 侧投影性能**：投影在 inner 的 CPU 输出上做（identity 时零成本旁路）；带权重时每步一小 matmul，Phase 2 可接受。GPU-前置（D2H 前投影）优化押后，非本期。
- **R7 递归**：`inner_type` 校验限定无状态 pool 集且显式拒 `projection`，`_build_key_builder` 递归一层即止。
- **R8 `_4`/`_64` 对称（G1 R2 finding 3）**：离线 `_create_builder` 补 canonical `cp1_spatial_pool_4`（与 `_64` 同类 alias，纯 additive）→ 在线/离线 inner 面对称，两名皆可作 inner 且各有测（测 8）。
- **R9 参数全链贯通（G1 R2 finding 2）**：`inner_type`/`projection_weights_path` 穿 `build_artifact→_ep_args→_process_episode→_create_builder`/`_get_vector_dims` + metadata provenance；测经顶层 `build_artifact()` 而非仅 `_create_builder`，防 CLI 值到不了 serial/ProcessPool worker。

---

## Review Log

### G2 Review Round 1 - NEEDS REVISION (2026-07-08)

Verdict: **NEEDS REVISION**.

Scope reviewed:
- Target plan: `logs/tracer_phase2_projection_key_builder.log.md`
- Implementation/docs/tests snapshot in working tree for Phase 2 projection key builder.

Blocking findings:

1. **`projection` weakens the wrapped `cp1_*` key enablement contract.**
   The plan defines `ProjectionKeyBuilder` as a wrapper over an inner stateless pool builder (§3.1 / §3.4), and `_build_key_builder` constructs the actual inner builder through `replace(cfg, type=cfg.projection.inner_type)`. However, `validate_cache_config()` only checks the projection `inner_type` set at `src/openpi/cache/config.py:1267`; the existing `cp1_*` key contract is guarded by `config.key_builder.type.startswith("cp1_")` at `src/openpi/cache/config.py:1747`, so it never runs when the outer type is `projection`. As a result, a `projection` wrapper around `cp1_mean_pool` accepts a robot-state-only config even though direct `cp1_mean_pool` requires both `vision_0` and `robot_state`. I reproduced this with an independent probe: `validate_cache_config()` accepted the config and printed `ACCEPTED_PROJECTION_WITHOUT_VISION0`. The current test at `tests/cache/test_config.py:1844` also encodes this invalid acceptance as "good inner". Fix by validating projection against the effective inner pool type for key requirements, then update/add tests that reject missing `vision_0` / `robot_state` for cp1 pool inners.

2. **`projection` also bypasses the `in_memory` preload requirement for cp1-style builders.**
   The preload validation at `src/openpi/cache/config.py:1899` only covers outer types whose `config.key_builder.type.startswith("cp1_")` or `clip`; `projection` is omitted even when `projection.inner_type` is a cp1 stateless pool. That lets `in_memory + projection(cp1_mean_pool)` pass without `backend.in_memory.preload_path`, while the same direct cp1 builder is rejected. I reproduced this with an independent probe: `validate_cache_config()` accepted the config and printed `ACCEPTED_PROJECTION_WITHOUT_PRELOAD`. Fix by applying the preload rule to the effective projection inner type and add a regression test for `projection` + `in_memory` without preload.

Verification run:
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run pytest tests/cache/components/test_projection_key_builder.py tests/cache/test_config.py tests/exp/test_build_in_memory_cache_artifact.py -q` → 169 passed.
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run pytest tests/cache/ tests/exp/test_build_in_memory_cache_artifact.py -q` → 1015 passed, 6 skipped.
- `git diff --check` → passed.
- Independent validator probes above both accepted invalid configs, confirming the blockers are coverage/contract holes rather than ordinary test failures.

### G2 Round 2 — Executor — 2026-07-08

逐条回应 Round 1 全部 2 阻断项（每项恰一条响应，§10.2）。二者同一根因：`validate_cache_config` 按外层 `config.key_builder.type` 分派，`type=="projection"` 时内层 cp1_* 的字段/库约束被跳过。

1. **Accepted** — projection 继承 cp1_* 的 key-enablement 契约。`config.py` 段 6 起处新增 `_effective_kb_type`（`type=="projection"` → `projection.inner_type`，否则外层 type），把 `config.py:1748` 的 vision_0+robot_state 要求改为 `_effective_kb_type.startswith("cp1_")` → projection(cp1_*) 现与直接 cp1_* 同样要求 vision_0+robot_state。修正原 `test_projection_validator_accepts_good_inner`（曾把 robot-state-only 误当合法，改为 vision_0+robot_state 合法配置）；新增 `test_projection_inherits_cp1_key_enablement_contract`（robot-state-only → 拒，match `requires keys.vision_0`）。
2. **Accepted** — projection 继承 in_memory preload 要求。`config.py:1899` 的 in_memory preload 条件改用同一 `_effective_kb_type`（`.startswith("cp1_") or == "clip"`）→ in_memory + projection(cp1_*) 无 `preload_path` 现被拒。新增 `test_projection_in_memory_requires_preload`（无 preload → 拒，match `preload_path`）+ `test_projection_in_memory_with_preload_accepts`（有 preload + in_memory 兼容 strategy → 收）。

`_effective_kb_type` 仅施于这两处**字段/库继承**检查；dims 类校验不改（投影后 out_dim≠inner dim，套 effective 会误伤）。blast-radius `uv run pytest tests/cache/ tests/exp/test_build_in_memory_cache_artifact.py` → **1018 passed, 6 skipped**（净增 3 测）。

Re-entering G2 with the updated code.

### G2 Review Round 2 - APPROVED (2026-07-08)

Verdict: **APPROVED**.

Scope reviewed:
- Developer response under `G2 Round 2 — Executor`.
- Updated `src/openpi/cache/config.py` validator changes.
- Updated projection config regression tests in `tests/cache/test_config.py`.

Round 1 blocker resolution:

1. **Resolved — projection inherits cp1 key enablement.**
   `validate_cache_config()` now computes `_effective_kb_type` for projection wrappers and applies the cp1 `vision_0` + `robot_state` requirement through that effective type. Independent probes confirmed both invalid cases are rejected: missing `vision_0` printed `REJECTED_PROJECTION_WITHOUT_VISION0`, and missing `robot_state` printed `REJECTED_PROJECTION_WITHOUT_ROBOT_STATE`. The prior bad "robot-state-only is good" test was replaced with a valid `vision_0` + `robot_state` case plus a rejection regression.

2. **Resolved — projection inherits in_memory preload requirement.**
   The `in_memory` preload rule now also uses `_effective_kb_type`, so `projection(cp1_*)` is treated like direct cp1 builders for `backend.in_memory.preload_path`. Independent probe confirmed the invalid no-preload config is rejected with `REJECTED_PROJECTION_WITHOUT_PRELOAD`, and the new tests cover reject/accept paths.

Verification run:
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run pytest tests/cache/test_config.py::test_projection_validator_accepts_good_inner tests/cache/test_config.py::test_projection_inherits_cp1_key_enablement_contract tests/cache/test_config.py::test_projection_in_memory_requires_preload tests/cache/test_config.py::test_projection_in_memory_with_preload_accepts -q` → 4 passed.
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run pytest tests/cache/components/test_projection_key_builder.py tests/cache/test_config.py tests/exp/test_build_in_memory_cache_artifact.py -q` → 172 passed.
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run pytest tests/cache/ tests/exp/test_build_in_memory_cache_artifact.py -q` → 1018 passed, 6 skipped.
- `git diff --check` → passed.

No remaining G2 blockers found.
