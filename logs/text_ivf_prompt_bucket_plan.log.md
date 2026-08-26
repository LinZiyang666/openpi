# Text-IVF Prompt 桶索引 + 指令段定界池化 — 实现计划

> Status: Implemented (G1 APPROVED R4 / G2 APPROVED R2 2026-08-25; §6 Verify green — 4278 passed, 0 scoped regressions)
> Level: L2
> Date: 2026-08-25
> Executor: Execution Authority session
> 需求讨论:与 owner 于 2026-08-25 会话内收敛,scope 经 owner 裁定"一步到位、无分期"。

---

## 1. 背景与目标

给 cache 系统加一个以 **text 模态(`prompt_emb`)为筛选模态的 IVF 桶索引**:建库时按 prompt embedding 分桶(embedding 相同 ⇒ 同桶,桶大小不限),搜索时只收"相同的/最近的那 1 个桶",在桶内走现有精排。效果:库内任务 ⇒ 语义化的任务范围圈定 + 精排候选从全库 N 缩到桶内 N/B;库外新指令 ⇒ 路由到指令最像的任务的经验。

前置实测(2026-08-25,本会话):

1. **位级一致性成立**:libero_10 / libero_spatial 两个真库、20 任务、3658 条 entry,每任务内 `prompt_emb` 位级完全相同(uniq=1)。成因:prompt token embedding 是确定性查表,且 `pi05_libero` 显式 `discrete_state_input=False`(`src/openpi/training/config.py:784`),state 不进 prompt。
2. **但该性质不普适**:`discrete_state_input` 对 pi05 **默认 True**(`src/openpi/models/pi0_config.py:40-41`),`pi05_robocasa` 即为 True——该线上 state 文本烘进 prompt(`src/openpi/models/tokenizer.py:26-28`),同任务 embedding 逐步漂移,朴素分桶失效。
3. **现状池化稀释严重**:指令仅占 200 个 token 位置中的 10-21 个,其余为 padding;full-mean 下跨任务 cosine ≥ 0.99994(最坏 margin ~1e-6)。**排除 padding 的 masked mean 实测把最坏 margin 提升约千倍至 ~1.2e-3**(min/med/max = 0.9733/0.9938/0.9988)。
4. **H5 原料齐备,零重采**:采集 hook 挂在 `embed_tokens` 输出(`src/openpi/collect/collection_policy.py:82-84`,×√d 已就位),H5 逐步存**未池化** `prompt_emb [200, 2048] fp16`,episode attrs 存原始 `prompt` 字符串(`collection_policy.py:68-71`)。定界池化只需重建 pkl,不碰采集。

因此本计划一步到位交付三件事:**(a)** KeyBuilder 的 prompt 定界池化(masked mean + 指令段切片),同时解决 LIBERO 塌缩与 robocasa state 污染;**(b)** InMemoryBackend 的 text-IVF 桶索引(挂既有 `index_type` 预留位,`src/openpi/cache/config.py:451`);**(c)** 新 SearchStrategy `text_ivf_knn` + config 三方绑定校验 + 离线 builder 重建路径 + 文档与测试。

## 2. 术语表

| 术语 | 英文 | 定义 |
|---|---|---|
| 筛选模态 | screening field | 用于建桶/收桶的字段,本计划固定为 `prompt_emb` |
| 桶 | bucket | 筛选模态 embedding 逐字节相同的 entry 组;桶键 = embedding 字节串,桶代表 = 该 embedding 本身 |
| 收桶 | probe | 查询时选定 1 个桶作为精排候选集(先精确匹配,失败退最近) |
| 路由 margin | routing margin | query 与所选桶代表 cosine 减去与次优桶代表 cosine |
| 定界池化 | instruction-span masked pooling | prompt 池化仅覆盖真实指令 token:masked(排除 padding)+ 可选 span 切片(排除 State 段) |

## 3. 已裁决设计决策(owner 拍板记录)

| # | 决策 | 裁决 |
|---|---|---|
| D1 | 筛选模态 | 仅 text(`prompt_emb`),其他模态不做 |
| D2 | 建桶规则 | embedding 逐字节相同 ⇒ 同桶;不限桶大小;无 k-means/nlist/nprobe |
| D3 | 收桶规则 | 恒收 1 个桶:先字节精确匹配(O(1)),失败退 fp32 cosine 最近;不设兜底阈值(质量由 judge 分数阈值兜底);平票按桶键字节序取最小(确定性) |
| D4 | 与 task_key filter 关系 | `text_ivf_knn` 策略**不下发 task_key 过滤**,桶即任务范围替代;`step_range` / `checkpoint_id` 在桶内叠加生效 |
| D5 | 精排 base fusion | `weighted_score_sum`(桶内 prompt_emb 常量,score_sum 下无害;RRF 组合不提供) |
| D6 | 收桶频率 | 逐步收(代价 = 对 ~几十个桶代表算 cosine,可忽略;无粘滞状态) |
| D7 | 定界池化 | 一步到位:masked mean(LIBERO 线,mask 在线免费)+ instruction-span 切片(state-in-prompt 线);不分期 |
| D8 | 数据来源 | pkl 从现有 H5 重建;**不重新采集** |
| D9 | 非回归锚点 | ①单任务库:text-IVF 开 ≡ 关;②多任务库+库内指令 query:≡ 现行 task_key 过滤搜索 |
| D10 | 桶数校验 | 建索引时桶数 > 上限 ⇒ 响亮报错(检测 state 污染 / 错配 artifact),不静默降级 |

## 4. 触及文件与接口总览

| 文件 | 改动 |
|---|---|
| `src/openpi/cache/components/key_builder.py` | `_CP1BaseKeyBuilder` 增 `prompt_masked_pool` / `prompt_instruction_span` 构造旋钮;`collect()` 增缓存 `prefix_pad_masks`、`tokenized_prompt`;`_slice()` 增掩码/切片分支;模块级共享纯函数 `find_instruction_span(ids, marker)` |
| `src/openpi/cache/backends/in_memory_backend.py` | 新 `_TextIvfIndex`(建桶/收桶/失效);`search_with_diagnostics` 收桶分派;`_filtered_cache` 键扩展桶维 |
| `src/openpi/cache/components/search_strategy.py` | 新 `TextIvfKnnStrategy(TrajectoryMixin)` |
| `src/openpi/cache/config.py` | 新 `TextIvfIndexConfig`;`InMemoryConfig.text_ivf`;`KeyBuilderConfig` 两旋钮;`_build_backend` / `_build_key_builder` / `_build_search_strategy` 装配;`validate_cache_config` 规则组;`build_cache_components` 重构为复用 `build_shared_storage`(消除重复构造,统一绑定检查咽喉点);`_check_text_ivf_artifact_binding` |
| `src/openpi/cache/backend_pool.py` | `BackendFingerprint` 增 `text_ivf_params`(additive,legacy=None);`_build_empty_backend` / `_build_legacy_complete` 传 text_ivf 配置 |
| `src/openpi/cache/interceptor.py` | CP1/CP3 `check()` 调用点 kwargs 增 `tokenized_prompt=<规范表示>`;新私有 helper `_canonical_tokenized_prompt(observation)` 把两路径的真实表示(legacy `[1,L]` 模型设备 / coordinator `[L]` CPU)归一为 1-D CPU `np.int64`;additive kwarg,全部 builder `collect(**stage_outputs)` 已验证安全忽略 |
| `src/openpi/models/tokenizer.py` | `PaligemmaTokenizer` 增公开 helper `encode_fragment(text) -> list[int]`(裸 SentencePiece encode,无 BOS/无清洗,docstring 限定模板边界定位用途;数行 additive) |
| `exp/common/build_in_memory_cache_artifact.py` | 定界池化离线路径:重 tokenize H5 `prompt` attr 恢复 mask + 自检;`_FakeStage1` 增可选 `prefix_pad_masks`;artifact dict 增 `prompt_pool` 元信息;CLI 旗标 |
| `tests/cache/`(新增+扩展) | 见 §8 |
| `docs/architecture/cache_system.md`、`docs/cache/tutorial.md`、`docs/README.md`、`logs/README.md` | 文档与索引同步(提交同 commit) |

**不改**:`storage_types.py`(`QuerySpec` 复用既有 `backend_hints` 通道,零 schema 变更)、`backend_base.py`、`orchestrator.py`、judge/gate/write_policy、wire 协议。

## 5. 实现单元详设

### U1 — KeyBuilder 定界池化

`_CP1BaseKeyBuilder`(`key_builder.py:228` 起):

- `__init__(enabled_fields, *, prompt_masked_pool: bool = False, prompt_instruction_span: bool = False, tokenizer_factory=None)`。默认全 False ⇒ 现有行为逐字节不变。`tokenizer_factory` 仅测试注入用;生产仅 span 模式在启动期 lazy 构造 `PaligemmaTokenizer` 并一次性调用 `encode_fragment`(求边界标记 id,不逐步调用;masked-only 模式零 tokenizer 依赖)。
- `collect()`(现 `key_builder.py:238-244`):`prompt_masked_pool` 开时额外缓存 `s1.prefix_pad_masks`(`Stage1Output` 已含此字段,`pi0_pytorch.py:40-41`,`[B, prefix_len] bool`);无条件缓存 `stage_outputs.get("tokenized_prompt")`(U6 规范表示:1-D CPU `np.int64` 或 None;仅 span 模式消费,None 容忍)。
- `_slice()`(现 `key_builder.py:266-280`,文档明示可覆写缝):masked 开时 `prompt = prefix[_PROMPT_START:][lang_mask]`(`lang_mask = prefix_pad_masks[0, 768:]`);span 再开时进一步 `prompt = prompt[:span]`。
- **span 计算**(`prompt_instruction_span` 开时;基于 token id 边界标记,不依赖 prompt 字符串):启动期经 `encode_fragment(" State:")` 一次性求出边界标记 id 序列;每步在 `collect()` 缓存的 `tokenized_prompt` 有效段(按 lang_mask 截断)中搜索标记首次出现,`span = 标记起始下标`(指令 token = 标记之前全部,含尾逗号;在线/离线同一规则,常量后缀不影响桶语义),`prompt = prompt[:span]`。搜索规则实现为模块级共享纯函数 `find_instruction_span(ids, marker) -> Optional[int]`,离线 builder 复用同一份代码(U5)。`tokenized_prompt` 缺失/为 None/标记未找到 ⇒ 回退 masked-only + WARN(R2)。`tokenized_prompt` 是 `TokenizePrompt` 的实际产物,`InjectDefaultPrompt`(`transforms.py:105-111`)注入效果天然已含其中——ids 即"tokenization 实际消费的有效 prompt"。
- **公开 helper 接口**(`src/openpi/models/tokenizer.py`):`PaligemmaTokenizer.encode_fragment(text: str) -> list[int]` = 裸 `self._tokenizer.encode(text)`(无 add_bos、无清洗)。现有类仅公开 `tokenize()`、无 `encode()`(G1 R1 指出),以此 helper 取代。同源性约束:所有部署与离线 builder 共用同一 `gs://big_vision/paligemma_tokenizer.model`(`tokenizer.py:18` 与 builder `_PI05_TOKENIZER_SOURCE` 为同一常量);fragment 编码与 `max_len` 无关。自检:`span > lang_mask.sum()` ⇒ 回退 masked-only + WARN(见风险 R2)。
- 四个 cp1_* 具体类(mean/spatial16/spatial4/max)经基类免费继承;`projection` builder 的 inner pool 在 `_build_key_builder` 构造处同样穿透旋钮。`_reduce_prompt` 各子类实现不动(输入从"全部 token"变为"定界后 token",池化算子不变)。
- **effective-builder allowlist(G1 R3)**:定界旋钮仅对经 `_CP1BaseKeyBuilder._slice` 的类型有效——类层次核实仅 `CP1MeanPool/SpatialPool16/SpatialPool4/MaxPool` 四类继承该基类(`key_builder.py:303-364`);`full_original` / `cp1_temporal_prune` / `cp1_llm_layer_extract` / `clip` / `placeholder` / `cp1_groot_*` 均为独立类,旋钮会被静默丢弃。定义**单一来源常量** `PROMPT_POOL_KNOB_BUILDERS = {"cp1_mean_pool", "cp1_spatial_pool_16", "cp1_spatial_pool_4", "cp1_spatial_pool_64", "cp1_max_pool"}`(置于 `config.py`,离线 builder 导入同一常量);`projection` 仅当 `inner_type ∈ PROMPT_POOL_KNOB_BUILDERS` 时视为受支持。`_build_key_builder` 只向 allowlist 类型穿透旋钮,对非 allowlist 类型携带任一旋钮 true 的调用显式 raise(纵深防御,第一道在校验规则 8)。
- CP3 复用同一 builder 的 `build()`,自动一致。

### U2 — InMemoryBackend text-IVF 桶索引

- `InMemoryBackend.__init__(vector_dims, text_ivf: Optional[TextIvfIndexConfig] = None)`(additive)。
- 索引结构:`_text_ivf_buckets: dict[bytes, list[entry_id]]` + 桶代表 fp32 矩阵 `_text_ivf_reps [B, dim]`(批量 cosine 用)+ 平行桶键列表(字节序稳定排列,兼作平票裁决序)。
- **构建**:`_build_text_ivf_index()` 对 `self._entries` **全量** entry 按 `query_keys[field].numpy().tobytes()` 分组;**任一 entry 缺筛选字段 ⇒ `ValueError`**(与 R8 同口径,不静默排除)。`load_artifact` 末尾在配置了 text_ivf 时**立即构建并校验**(fail-fast;该路径在 `BackendPool` 每指纹 load 锁内、`freeze()` 之前执行 ⇒ serving 路径**从不**懒构建);空库启动(测试/纯在线写库)在首次收桶时懒构建。**桶数校验(D10)**:`len(buckets) > max_buckets` ⇒ `ValueError`,报文提示两类根因(state-in-prompt 污染 / 未用定界池化的 artifact)。
- **失效**:`_invalidate_frozen_search_caches()`(现 `in_memory_backend.py:226-228`,insert/delete 均已调用)同步丢弃桶索引,下次收桶懒重建。索引属派生搜索结构:freeze 后构建/重建**允许**(同 §5.10 score memo / frozen-search cache 的既有分类,C2 契约不动)。
- **边界语义**:空库/空索引收桶 ⇒ 返回 `([], diag)`(与现有 `if not self._entries` 早退一致),交 judge 判 MISS,不报错;insert 失效后下次收桶重建。**懒构建并发契约**:构建全程写局部变量,完成后以单次引用赋值发布 `_text_ivf_state = (buckets, reps, keys)`(GIL 原子,读者绝不见半成品);构建幂等(同一 entry 集两次构建结果相同,后发布覆盖等值)。懒构建仅存在于未 freeze 的测试/空库场景,其并发变异+搜索本就在 backend 既有线程契约之外(`backend_base.py` C2 说明);frozen serving 恒走 eager 路径。
- **收桶**(`search_with_diagnostics` 内、`_filtered_candidates` 之前分派):`spec.backend_hints.get("text_ivf")` 为真时——①精确:query 字节串命中桶键 ⇒ 该桶;②否则 fp32 cosine 对 `_text_ivf_reps` 取最近,平票取桶键字节序最小;③记录路由 margin(DEBUG 级;margin < 1e-4 升 WARN)。候选 = 桶成员经现有 filter 语义(checkpoint_id / step_range / outcome;本策略不发 task_key)过滤,随后进入**现有打分路径原样运行**(单步 weighted_score_sum、trajectory 单链/legacy、score memo 全部无改动)。
- **frozen-search 缓存兼容**:`_filtered_cache` 指纹元组追加桶键成员(非桶路径填 None,legacy 键行为不变),使桶候选列表跨步保持同一列表对象,`_field_matrix_cache` 的 id() 键与 weakref 守卫照常生效。
- hint 为真但索引未配置 ⇒ `RuntimeError`(config 校验为第一道防线,此为兜底)。

### U3 — TextIvfKnnStrategy

镜像 `WeightedScoreSumKnnStrategy`(`search_strategy.py:461-505`)实现 `TextIvfKnnStrategy(TrajectoryMixin)`:

- 构造参数:`top_k / step_filter / step_window / fusion_weights / field_similarity / score_normalization / trajectory_depth / trajectory_weights`(与 WSS 同名同义)。
- `search(ctx)`:`record_query_keys` → filters 仅由 step_filter 生成(**显式不设 `task_key`,即使 `ctx.task_key` 非空**,D4)→ `QuerySpec(fusion_method="weighted_score_sum", backend_hints={"text_ivf": True}, score_normalization=..., **traj_fields)` → `storage.search(spec)`。
- 前置断言:`"prompt_emb" in ctx.query_keys`,缺失即 `ValueError`(config 校验保证不触发)。
- 提供与 WSS 相同的 `last_step_features()` 读透(X15 seam 均一性)。
- 轨迹语义:桶只筛链头(layer-0 候选);祖先链经 `prev_ids` 照走不受桶约束——库内一条 episode 的 entry 全在同桶,天然自洽。

### U4 — Config / 装配 / BackendPool

- 新 `TextIvfIndexConfig`:`field: str = "prompt_emb"`、`max_buckets: int = 1024`。
- `InMemoryConfig`(`config.py:449-451`)增 `text_ivf: TextIvfIndexConfig`;`index_type` 合法值收紧为 `{"brute_force", "text_ivf"}`(当前无校验分支,新增)。
- `KeyBuilderConfig`(`config.py:502`)增 `prompt_masked_pool: bool = False`、`prompt_instruction_span: bool = False`;`_build_key_builder` 穿透至 cp1_* 与 projection inner。
- `_build_search_strategy`(`config.py:3459`)增 `text_ivf_knn` 分支。
- `_build_empty_backend` / `_build_legacy_complete`(`backend_pool.py:119-148`)在 `index_type == "text_ivf"` 时传 `TextIvfIndexConfig`;`BackendFingerprint`(`backend_pool.py:85`)增 `text_ivf_params: Optional[tuple]`(`(field, max_buckets)`;legacy=None,既有指纹等值性不变)。
- `validate_cache_config`(`config.py:1511`)新规则组:
  1. `type: text_ivf_knn` ⇒ backend 为 `in_memory` 且 `index_type == "text_ivf"`;
  2. `text_ivf_knn` ⇒ `score_normalization` 必填且合法(复用 WSS 规则);
  3. `text_ivf_knn` ⇒ `prompt_emb` ∈ enabled keys 且 ∈ `vector_dims`;
  4. 反向绑定:`index_type == "text_ivf"` ⇒ 至少一个启用 checkpoint 的策略为 `text_ivf_knn`(拒绝"建了索引没人用");
  5. `prompt_instruction_span` ⇒ `prompt_masked_pool`;
  6. `text_ivf_knn` 拒绝与 `cp1_groot_*` / `placeholder` / `clip` builder 组合(无 `prompt_emb` 语义);
  7. `text_ivf_index.field` 目前仅接受 `"prompt_emb"`(前向留位,当前锁死)。
  8. **定界旋钮 allowlist(G1 R3)**:`prompt_masked_pool` 或 `prompt_instruction_span` 为 true ⇒ `key_builder.type ∈ PROMPT_POOL_KNOB_BUILDERS`,或 `type == "projection"` 且 `projection.inner_type ∈ PROMPT_POOL_KNOB_BUILDERS`;否则 `ConfigValidationError`(拒绝"YAML 声称定界、运行时静默原语义")。旋钮均 false 时,其余能产出 `prompt_emb` 的 builder 与 text-IVF 的组合维持现有 scope(仅受规则 6 约束)。

- **artifact 语义绑定(启动期强制)**:公开装配入口有两个——并发模式 `build_shared_storage`(`config.py:2582`)与单连接模式 `build_cache_components`(`config.py:2590`,现状**独立**执行 `_build_backend()` → `CacheStorage()`,不经前者,构成旁路,G1 R2 指出)。本计划**重构 `build_cache_components` 复用 `build_shared_storage`**(其 backend+storage 构造本就与前者逐行重复,重构同时消除既有重复),使两个公开入口汇于同一咽喉点;`_check_text_ivf_artifact_binding(storage, config)` 落在 `build_shared_storage` 末尾(backend 已 load、`artifact_meta` 可读、`config` 含 key_builder/strategy 配置),两入口自然同享强制。`build_per_connection_components` 只接收已过检的 shared_storage,不另设检查;绕开两个公开入口手拼 `_build_backend`+`CacheStorage` 属私有 API 使用,在契约之外。检查内容:当任一启用 checkpoint 的策略为 `text_ivf_knn` **或** key_builder 定界旋钮开启,且 `preload_path` 非空时——要求 `storage.artifact_meta` 非 None、`artifact_meta["key_builder_type"] == key_builder.type`、artifact `prompt_pool` 存在且 `masked`/`instruction_span` 与配置旋钮**逐项相等**。legacy artifact(meta 或 `prompt_pool` 缺失)与任何方向的错配(含 artifact 定界而配置未开)⇒ `ConfigValidationError`,报文指示按 §9 runbook 重建(沿用 §5.17 GR00T identity guard 的 `CacheStorage.artifact_meta` 模式)。空 preload(纯在线写库)跳过——key 全部由同一在线 builder 产出,语义自洽。U5 的 `prompt_pool` 元信息由"记录不强制"升级为**本检查的强制输入**;D10 桶数校验降为第二道防线。

### U5 — 离线 builder 重建路径

`exp/common/build_in_memory_cache_artifact.py`:

- CLI 增 `--prompt-masked-pool` / `--prompt-instruction-span` / `--discrete-state-input`。**旗标 × builder 类型同受 `PROMPT_POOL_KNOB_BUILDERS` 约束**(从 `config.py` 导入同一常量):任一定界旗标开启而目标 builder 类型不在 allowlist ⇒ 在处理任何 H5 之前 abort。这同时封死"元信息撒谎"通路——否则不支持旋钮的 builder 会产出 full 语义的 key 却按旗标写入 `prompt_pool: masked=true`,U4 绑定检查将错误放行该错配 artifact(G1 R3 指出)。
- mask 恢复:重 tokenize H5 root attr `prompt`(存于 `data_collector.py:132` 合并的 episode attrs;实测在位),分支 `discrete_state_input`——与 `_build_fake_stage1_with_masks` 现行逻辑同款(该函数已实现此分支与确定性重 tokenize,builder 内 `_PI05_TOKENIZER_SOURCE` / `_PI05_TOKENIZER_MAX_LEN=200` 已存在)。
- **span 同源规则**:离线对 `tokenizer.tokenize(prompt, state)`(公开 API,与 `TokenizePrompt` 同一调用形态 ⇒ 逐 id 相同)的输出 id 序列执行与在线**同一份** `find_instruction_span` 纯函数。构建期自检:每条 H5 prompt 完整 tokenization 中标记**恰好出现一次**,否则中止构建。
- `_FakeStage1` 增可选 `prefix_pad_masks` 字段;普通(非 llm-extract)构建路径在定界模式下填充之,再用**与在线完全相同的 KeyBuilder 类与旋钮**跑 `collect/build`(离线在线同代码,parity 由构造保证)。
- **双源自检**(LIBERO 型,D10 配套):`mask.sum() == 200 - trailing_identical_rows(prompt_emb)`(padding 行为同一常量向量,可探测);不一致 ⇒ 中止构建并报错。instruction-span 型自检:`span < mask.sum()`。
- artifact dict 增 `prompt_pool: {"masked": bool, "instruction_span": bool}` 元信息;`load_artifact` 收进 `artifact_meta`,由 U4 绑定检查**强制核对**(D10 桶数校验为第二道防线)。

### U6 — Interceptor 插管(tokenized_prompt)

`interceptor.py:875-883`(CP1)与 `:1126` 相邻(CP3)的 `check()` 调用点:kwargs 增 `tokenized_prompt=<int token ids>`,取值按既有两路径分叉(`interceptor.py:838-860`,均已核实):legacy 路径取 `observation.tokenized_prompt`(`_model.Observation` 字段,`model.py:98`,Optional——None 透传,builder 侧防御);coordinator 路径取 `observation["tokenized_prompt"]`(post-transform raw dict,`TokenizePrompt` 已写入该键)。弃用 prompt 字符串方案的原因(G1 R1 指出):字符串在 legacy 路径不可达(`Observation` 无 `get()`)、在 coordinator 路径已被 `TokenizePrompt` `pop`;而 ids 在两路径均可达,且 `InjectDefaultPrompt` 注入效果已含其中。

**规范表示(G1 R2 指出两路径真实形状/设备不同,此处冻结统一接口)**:kwarg 传递的恒为 **1-D CPU `np.ndarray`(int64,长度 L=padded 全长)或 None**。归一化在 interceptor 侧新私有 helper `_canonical_tokenized_prompt(observation) -> Optional[np.ndarray]` 完成——legacy 路径真实表示为 `[1, L]` 模型设备张量(`infer` 内 `[None, ...] + .to(device)` 所致):断言 `ndim == 2 and shape[0] == 1`(interceptor 恒 B=1,violation 即 raise),`t.detach()[0].to("cpu", torch.long).numpy()`;coordinator 路径为 `[L]` CPU 张量:同规则免断言直转(2-D 且 batch=1 亦防御性接受)。None 透传 None。该归一化是**每步一次、显式、有界的 D2H 拷贝**(int64×200 ≈ 1.6KB,仅 legacy 路径发生;绝不在 builder 内对 GPU 张量做逐元素/`.tolist()` 访问)。builder 侧 `collect()` 只接受此规范表示;`find_instruction_span(ids: np.ndarray, marker: np.ndarray) -> Optional[int]` 为纯 numpy 滑窗匹配,构造上与设备无关。全部 builder `collect` 签名为 `**stage_outputs`(8 处定义逐一验证),legacy builder 静默忽略,零行为变化;wire 协议不动。

## 6. 集成点与数据流

```
serve: yaml(index_type=text_ivf + text_ivf_knn + masked builder)
  → BackendPool.get_or_load(fingerprint 含 text_ivf_params)
  → load_artifact(定界池化 pkl) → 建桶+桶数校验 → freeze()
infer: Stage1 → collect(stage1, tokenized_prompt) → gate → build()[定界池化 key]
  → TextIvfKnnStrategy.search(无 task_key filter, hint=text_ivf)
  → backend 收桶(exact→nearest) → 桶内现有精排(score_sum/trajectory/memo)
  → judge / fetch / verdict 反馈 —— 全部原样
```

## 7. YAML 配置示例

```yaml
key_builder:
  type: cp1_mean_pool
  prompt_masked_pool: true
  # prompt_instruction_span: true   # 仅 state-in-prompt 部署(如 pi05_robocasa)
keys:
  vision_0:    { enabled: true, weight: 1.0 }
  prompt_emb:  { enabled: true, weight: 0.0 }   # 参与建桶;精排权重可为 0
  robot_state: { enabled: true, weight: 1.0 }
checkpoints:
  cp1:
    enabled: true
    search_strategy:
      type: text_ivf_knn
      top_k: 1
      field_similarity: { vision_0: {type: cosine}, robot_state: {type: l2} }
      score_normalization: { type: per_field, fields: { ... } }
    judge: { type: threshold, threshold: 0.98 }
backend:
  type: in_memory
  vector_dims: { vision_0: 2048, prompt_emb: 2048, robot_state: 32 }
  in_memory:
    preload_path: <定界池化重建的 pkl>
    index_type: text_ivf
    text_ivf: { field: prompt_emb, max_buckets: 1024 }
```

注:`weight: 0.0` 的字段现行 `_iter_active_fields` 会跳过精排(`in_memory_backend.py:653-668`),但 query_keys 仍携带 ⇒ 收桶可用——此组合作为推荐配方写入 tutorial。

## 8. 测试策略

新增 `tests/cache/test_text_ivf_backend.py`:建桶正确性(分组/代表/字节序)、精确收桶、最近收桶、平票确定性、桶数超限报错、mutation 失效重建、freeze 后懒构建允许、hint 无索引报错、桶内 filter 叠加、与 frozen-search 缓存共存。

新增 `tests/cache/test_text_ivf_strategy.py`:QuerySpec 形状(hint / 无 task_key / fusion=score_sum)、**锚点①**(单任务库开≡关,逐值)、**锚点②**(多任务库库内 query ≡ task_key 过滤 brute-force,逐值)、轨迹路径逐值 parity(桶内 vs task_key 过滤)、score memo 组合、库外 query 路由到最近桶。

新增 `tests/cache/test_key_builder_masked_prompt.py`:masked mean 对手算值、旋钮全关逐字节回归(锚定现状)、span 切片(fake tokenizer 注入)、span 自检回退、`collect` 缓存 mask/tokenized_prompt、projection inner 穿透。

扩展 `tests/cache/test_config.py`:§U4 全部校验规则的正反例(规则 8 逐一覆盖:`full_original` / `cp1_temporal_prune` / `cp1_llm_layer_extract` / `clip` / `placeholder` 各携带旋钮 true 拒绝、projection-非法-inner 拒绝、allowlist 内类型 + 旋钮 true 通过);**artifact 绑定四态 × 两个公开入口**(`build_shared_storage` 与 `build_cache_components` 各自:匹配通过 / 错配拒绝 / legacy-缺失拒绝 / 空 preload 跳过,含反向错配;并断言 `build_cache_components` 重构后经由 `build_shared_storage`)。

扩展 `tests/cache/test_interceptor.py`:**tokenized_prompt 取值四例**——显式 prompt、default prompt(`InjectDefaultPrompt`)、legacy `Observation` 路径、coordinator dict 路径——每例对**两种真实表示**(legacy `[1,L]` 张量、coordinator `[L]` 张量)断言:归一化输出为同一 1-D CPU `np.int64`,且对其执行**真实 marker 查找与 span 切片**的结果逐值一致(不止"收到 ids");外加 batch≠1 断言 raise、None 透传两例。CUDA 设备侧归一化并入 manual parity 测试(CI 无 GPU;`.to("cpu")` 逻辑同一代码路径)。

`test_text_ivf_backend.py` 边界组:**缺字段 entry 建索引报错**、**空库收桶返回空**、**懒构建两次幂等**、**原子发布**(构建中途读 `_text_ivf_state` 只见 None 或完整态)、**`BackendFingerprint.text_ivf_params` 隔离**(不同 `max_buckets` 不共享实例)。

builder 离线测试(合成 H5 → 定界构建 → 与在线同款 builder 直算逐值一致;双源自检正反例;**定界旗标 + 非 allowlist builder 类型 ⇒ 处理前 abort**);真 tokenizer parity 标 `@pytest.mark.manual`(需 GCS 下载,遵循既有 manual 惯例):对真实 pi05 state-in-prompt 模板(`Task: {text}, State: {digits};\nAction: `)断言 `encode_fragment(" State:")` 标记在完整 tokenization 中**恰好出现一次**、`find_instruction_span` 切出的指令段与直算一致;覆盖 LIBERO(无 State,标记不出现 ⇒ span=None)与 robocasa 两种格式。

§6 Verify 按章程跑裸全量 `uv run pytest`(既有失败清单见 `reference_preexisting_test_failures` / `reference_pytest_manual_skip` 记忆,不计新增回归)。

## 9. Artifact 重建 runbook(验证目标)

对 `exp/common/data/db/libero_cache/{libero_10, libero_spatial}` 现有 H5,以 `--prompt-masked-pool` 重建 `cp1_mean_pool` 定界变体 pkl,输出至 `exp/common/data/cache_artifacts/<suite>/cp1_mean_pool_maskedprompt.pkl`(**新文件名,不覆盖现有 artifact**;`data/` 目录 gitignore 例外规则不变)。重建后验收:桶数 == 10;每桶 entry 数与 task 分组一致;跨任务桶代表 cosine 最坏 margin ≥ 1e-3 量级(对照 §1 实测)。robocasa 线 H5 的清点与(如含 prompt_emb 的)重建同属本 runbook,执行时逐库核对 `prompt` attr 与字段在位性。

## 10. 风险登记

| # | 风险 | 缓解 |
|---|---|---|
| R1 | 在线 key 池化链(bf16)与离线重建链(fp16→fp32)低位不一致 ⇒ 在线精确收桶可能失手 | nearest 兜底为正确性主路径(精确匹配仅是 O(1) 快路径);定界后任务间 margin ~1.2e-3 ≫ 数值抖动(~1e-6 量级);margin<1e-4 WARN 监控 |
| R2 | `encode_fragment(" State:")` 标记片段的上下文相关分词与完整 prompt 内切分不一致 ⇒ 标记搜索失手 | 标记前界为逗号(独立标点 piece)、后界为空格起数字 piece,边界稳定;离线构建期逐条自检"标记恰好出现一次"强制;在线未找到 ⇒ 回退 masked-only + WARN;manual test 真 tokenizer 断言两种模板 |
| R3 | `max_buckets` 误杀合法大指令集库 | 可配置;报错文案给出调参与根因排查两条路 |
| R4 | 替代 task_key filter 属检索语义变更 | 锚点②逐值锁死库内等价;跨任务路由为特性而非回归;文档显著标注 |
| R5 | span 模式启动期需 tokenizer(GCS 下载)求边界标记 | 仅启动期一次 `encode_fragment`,之后零 tokenizer 调用;失败 fail-fast;`maybe_download` 本地缓存;masked-only 模式零依赖 |
| R6 | BackendPool 指纹漏掉 text_ivf 参数 ⇒ 错误共享实例 | fingerprint 增 `text_ivf_params` + 专项测试 |
| R7 | 桶路径与 frozen-search/matrix/score-memo 缓存交互 | 桶候选列表纳入 `_filtered_cache`(键扩桶维)保持列表身份稳定;组合测试覆盖 |
| R8 | 现存无 `prompt_emb` 的 artifact(weighted_sum 线部分)误配 | 建索引全量字段校验即报错(U2,任一 entry 缺字段);校验规则 3 + U4 绑定检查前置拦截 |
| R9 | 不支持旋钮的 builder 携带定界旗标构建 ⇒ artifact `prompt_pool` 元信息与实际 key 语义不符,U4 绑定被错误放行 | config 与离线 builder 双侧共享 `PROMPT_POOL_KNOB_BUILDERS` allowlist fail-fast(规则 8 + U5 abort);`_build_key_builder` 穿透处显式 raise 纵深防御 |

## 11. 边界(owner 裁定的 scope,非延期项)

Owner 于需求讨论中裁定"现在只对 text 模态做 IVF,先不考虑别的":多字段/多桶收取(nprobe>1)、学习式聚类桶、桶级分数参与精排、Qdrant 后端、GR00T 路径(`cp1_groot_*` 无 `prompt_emb`,校验规则 6 显式拒绝)均在本次 scope 之外。

## Review Log

### G2 Round 1 — 2026-08-25 19:40 CDT — NEEDS REVISION

**Review Authority:** Codex `/root`  
**Review scope:** G1-approved plan versus current text-IVF implementation, relevant tests/docs/indexes, independent boundary probes, and project regression suite. Other sessions' working-tree changes were excluded from scope.

**Evidence**

- Relevant implementation suite: `273 passed, 1 deselected` (`test_key_builder_masked_prompt`, `test_text_ivf_backend`, `test_text_ivf_strategy`, `test_text_ivf_offline_build`, `test_interceptor`, `test_config`, `groot/test_groot_load_guard`; `-m 'not manual'`).
- Independent reviewer probes: `3 failed` in gitignored `tests/review_tests/test_text_ivf_g2_round1.py`, reproducing both findings below.
- Project suite excluding gitignored review tests and manual tests: `4127 passed, 22 skipped, 38 deselected, 6 failed`. Five failures are unrelated session/environment failures and are not charged to this task; `tests/robocasa365/test_groot_cache_collector.py::test_collected_episode_builds_a_loadable_groot_artifact` is a direct scoped regression.
- Relevant architecture/tutorial/root/log indexes are present and link the feature/runbook.

**Blocking findings**

1. **U1 missing/None fallback is not cycle-safe.** The approved contract says `collect()` unconditionally caches `stage_outputs.get("tokenized_prompt")`, so missing/None ids fall back to masked-only. The implementation updates `_tokenized_prompt` only when the kwarg exists, while `clear()` leaves it intact. After a prior marker-bearing step, a later CP1 collect that omits ids silently reuses the stale span and pools the wrong instruction slice. The submitted unit test currently locks in this contrary behavior. Required: make a new CP1 cycle replace ids with the supplied canonical value or `None` (preserving CP1→CP3 reuse only where explicitly needed), and add a regression covering prior ids → clear/new CP1 → missing/None ids → masked-only output.
2. **U6 canonical shape contract is not enforced, and the official suite regresses.** `_canonical_tokenized_prompt()` promises 1-D CPU int64 or None but currently returns 0-D and 3-D inputs unchanged instead of failing loud; independent scalar and `[1,1,L]` probes both fail. Required: accept only 1-D or defensive `[1,L]`, reject every other shape, and cover the cases. Separately, the additive `artifact_meta["prompt_pool"]` surface was not reconciled with the existing GR00T collector contract, leaving the scoped official test red; update the compatible metadata/test contract without weakening text-IVF's startup binding, then rerun the relevant and project suites.

**G2 checklist**

- [x] Main implementation follows the approved bucket-index, strategy, config/binding, offline-builder, interceptor, and documentation architecture.
- [x] New positive/negative text-IVF tests pass.
- [ ] Frozen missing/None token-id lifecycle semantics are implemented and regression-locked.
- [ ] Canonical token-id shape invariant is fail-loud for malformed inputs.
- [ ] No scoped project-suite regression remains.

**Verdict:** `NEEDS REVISION`. The remaining issues are localized, but stale token ids can silently change the query key and therefore bucket routing, so G2 cannot release this round.

### G2 Round 2 — Executor — 2026-08-25

- Accepted — [Finding 1: token-id 生命周期] 核实属实:实现为条件更新,与 plan U1 "无条件缓存 `stage_outputs.get(...)`" 原文相悖,stale ids 可跨 `clear()` 泄漏错切 span。修复:`collect()` 改为无条件 `self._tokenized_prompt = stage_outputs.get("tokenized_prompt")`(缺失⇒None⇒masked-only 回退,方向安全);`clear()` 同步置 None。interceptor 在 CP1/CP3 两处均传参,无需跨检查点保留。回归:`test_collect_caches_mask_and_ids` 改锁正确语义(缺 kwarg ⇒ None、clear ⇒ None),新增 `test_stale_ids_never_leak_into_new_cycle_span`(带标记步 → clear → 新周期缺 ids ⇒ masked-only + WARN,且 ≠ 旧 span 键)。
- Accepted — [Finding 2: 规范形状 + GR00T 官方测试回归] (a) `_canonical_tokenized_prompt` 补 1-D 终态强制:torch/numpy 两分支在 squeeze 后均校验 `ndim == 1`,0-D 标量与 `[1,1,L]` 3-D 一律 `ValueError`;新增 `test_malformed_shapes_raise` 覆盖 torch/numpy 各 0-D/3-D 四例。(b) `tests/robocasa365/test_groot_cache_collector.py` 的 `artifact_meta` 精确断言更新为真实语义——GR00T collector 经共享 builder 构建,artifact 现记录 `prompt_pool: {masked: False, instruction_span: False}`(如实,非 legacy None);text-IVF 启动绑定不受削弱(绑定检查仅在 text-IVF/旋钮启用时引燃,该 artifact 记录的 False/False 与任何 masked 配置错配仍拒绝)。
- 复跑证据:`tests/cache/ + tests/robocasa365/` **1632 passed / 22 skipped**;`tests/exp/` 1377 passed / 2 failed(`test_prebuilt_matrix_backend` 两例,stash 对照 HEAD 即失败,非本任务);text-IVF 定向套件全绿。

### G2 Round 2 — Review Authority — 2026-08-25 20:17 CDT — APPROVED

**Review Authority:** Codex `/root`  
**Review scope:** Round 1 staged baseline plus the executor's six declared Round 2 files and appended response; other sessions' working-tree changes remained excluded.

**Finding disposition**

1. **Finding 1 CLOSED.** `_CP1BaseKeyBuilder.collect()` now unconditionally replaces `_tokenized_prompt` with `stage_outputs.get(...)`, and `clear()` resets it to None. The revised unit contract and explicit prior-marker → clear/new-cycle → missing-ids probe prove masked-only fallback with no stale-span reuse.
2. **Finding 2 CLOSED.** `_canonical_tokenized_prompt()` now accepts only `[L]` or defensive `[1,L]` values and rejects malformed torch/numpy 0-D and 3-D inputs. The GR00T collector expectation now reflects the additive False/False `prompt_pool` identity emitted by the shared artifact builder; text-IVF's startup equality checks are unchanged.

**Independent evidence**

- Prior reviewer probes plus the directly changed key-builder/interceptor/GR00T tests: `51 passed`.
- Complete text-IVF G2 matrix, including config binding and GR00T load guard: `286 passed, 1 deselected` (`manual` excluded).
- Full scoped suite `tests/cache tests/robocasa365 -m 'not manual'`: `1632 passed, 2 skipped, 20 deselected`.
- Executor-reported two `tests/exp/test_prebuilt_matrix_backend.py` failures are HEAD-existing/non-scope and do not affect this verdict.

**G2 checklist**

- [x] Implementation remains consistent with the G1-approved plan.
- [x] Missing/None token-id lifecycle is cycle-safe and regression-locked.
- [x] Canonical token-id shape invariant is fail-loud and covered for both representations.
- [x] Artifact metadata remains truthful and startup binding strength is preserved.
- [x] Relevant feature, compatibility, documentation, and scoped regression tests pass.
- [x] No new Blocking or major issue found.

**Verdict:** `APPROVED`. G2 is released; proceed to Verify under Execution Authority.
