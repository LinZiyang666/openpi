# GR00T N1.5 接入 Cache 系统（两阶段切分）

**Status**: `In Progress` — G1 APPROVED（2026-08-17），进 §4 Code
**Level**: L3（跨模块 + 新子系统能力 + 架构文档更新）
**Authority**: Execution
**日期**: 2026-08-17
**上游**: [`logs/session_handoff_robocasa365.md`](session_handoff_robocasa365.md) §5、[`logs/groot_n15_robocasa_adapter.log.md`](groot_n15_robocasa_adapter.log.md)（推理适配层，已 ship `15dfa67`）
**必读前置**: [`docs/cache/migration.md`](../docs/cache/migration.md) —— 本仓库既有的「非 Pi0.5 模型接入 cache」迁移指南，本计划的形状即由它规定（见 §2.9）

---

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-08-17 13:23 CDT

- [Blocking] [Concern] T-8 is absent: `tests/robocasa365/test_groot_cache_manual.py` was not implemented or staged — reasoning: this is the approved plan's only real-model coverage for G0-C bit-exact split equivalence, repeatability of `run_stage2` on the same stage-1 object, real image-token positions, and the three negative controls. Stub tests cannot validate the copied upstream forward or the pinned source against the actual GR00T checkpoint. Add the manual island-B suite and record a non-skipped run with the required `PYTHONPATH`/`--run-manual` invocation.
- [Blocking] [Concern] T-4 implements only D1 while the approved G0-D2 gate is missing — reasoning: `test_online_offline_key_parity.py` explicitly defers the bf16→fp16 end-to-end check to the missing manual suite and contains no zero-norm error rule, metric-aware argmax/margin check, degenerate-field report, or mandatory non-degenerate `vision_0`/`robot_state` assertion from §8. Implement and run the complete (a)+(b) gate without weakening its pre-registered thresholds.
- [Blocking] [Concern] T-7 does not exercise the promised server→collector→artifact-builder chain — reasoning: the server construction test only checks the wrapper type, while the HDF5 tests instantiate `GrootCacheCollector` directly; no test feeds the produced file to `build_in_memory_cache_artifact.py`. Add one integrated test that starts at `_build_served_policy(... --collect-hdf5 ...)`, records two steps and a successful episode, then builds/reads a GR00T artifact and checks its entries and identity metadata.
- [Blocking] [Concern] T-9 and T-10 substitute hand-built replicas for the production paths they claim to cover — reasoning: the "real legacy artifact" case constructs a stub backend already containing a dict of `None` values instead of loading an actual legacy pickle through `InMemoryBackend.load_artifact`, and the exception lifecycle test manually writes a `try/finally` rather than calling `run_one`. Both tests would remain green if the corresponding production behavior regressed. Exercise the real load chain and the real rollout function with mocked environment/client failure.
- [Blocking] [Concern] The M2 `_slice()` refactor leaves both full-`build()` benchmark subclasses silently calling module-level `_slice_cp1_fields` — reasoning: the approved plan explicitly required either applying `self._slice()` in `exp/cache_latency_bench/opt/r4_pool_keybuilder.py` and `r4_layout_check.py` or documenting at both overrides why the new hook is inapplicable. Neither file is changed, preserving the exact semantic asymmetry the plan forbids.
- [Blocking] [Concern] G0-E has no real closed-loop execution evidence — reasoning: the stub probe-count unit test establishes local branch accounting only; it does not prove that a server loaded the built artifact, returned a library action on `always_hit`, propagated `__hit_meta__` through the adapter, and wrote client JSONL with the monitor positive controls enabled. Run the approved one-episode gate and append the command and observed positive/negative counts for the next review round.
- [Non-blocking] [Suggestion] The staged non-manual regression surface is otherwise healthy — reasoning: the focused GR00T/cache set passed `131 passed`, the approved blast radius passed `2568 passed, 8 skipped`, and `git diff --cached --check` reported no whitespace errors.

### G2 Round 2 — Executor — 2026-08-17

- Accepted — **T-8 缺失**。属实，manual 套件我漏写了。已新增 `tests/robocasa365/test_groot_cache_manual.py`（9 个 manual 用例）：G0-C 逐位等价（`run_stage2(run_stage1(x))` vs `model.get_action(x)`，固定种子）、`backbone_features` 与上游 `backbone(...)` 输出逐位相等（用捕获 `action_head.get_action` 的方式做真 A/B，而不是拿我自己的复算去比我自己）、同一 `stage1` 连调两次 `run_stage2` 可复现（钉 `vlln` 就地写回）、真实图像段为三块 256 且**起始偏移随 prompt 长度移动**（这条自带断言：偏移若不动，掩码切分与固定偏移表就无从区分，该测试也就证明不了它存在的理由）、散射位置等于 `vit_embeds`。**三条反向对照**齐备：不设种子时两次调用必须不同、故意绕过 session 时结果必须不同、以及先跑一次丢弃以避开 cudnn 首调选算法。文件 docstring 写明必须带 `PYTHONPATH=/home/weiland/gr00t_n15:…` 与 `--run-manual`，否则 `importorskip` 会静默跳过并读作通过。⚠ **执行**见本轮最后一条。
- Accepted — **T-4 只实现了 D1，G0-D2 缺失**。属实。已把 G0-D2 做成可运行判据 `exp/robocasa365/groot_key_parity.py::check_key_parity`，完整实现 §8 的 (a)+(b) 且不放宽任何预注册阈值：(a) 逐字段相对误差 `≤1e-3`，`‖k_off‖₂==0` 时退化为绝对误差；(b) **按字段自身度量**（cosine / L2）以 `k_on` 查库、要求 `argmax==t` 且 margin 严格为正；退化字段（存在 `t≠t'` 使 `k_off` 逐位相同）自动识别、排除出 (b)、并在报告里点名；`vision_0`/`robot_state` 若落入退化名单则整门判 FAIL 而非豁免。新增 `tests/cache/groot/test_key_parity_gate.py`（8 例）在合成数据上证明**判据本身会失败**：能翻转赢家的扰动被 (b) 抓住、5% 漂移被 (a) 抓住、恒定 `prompt_emb` 被判退化且不做除零、`robot_state` 恒定的 episode 被判为"无法回答本门"、L2 字段用 cosine 打分会退化（说明按字段选度量不是装饰）。真数据上的 G0-D2 已接进 T-8 的最后一个用例（采集 6 步 → 真 `build_artifact` → 读回 offline key → 跑该判据）。
- Accepted — **T-7 没穿过所声称的链路**。属实：原来 server 构建测试只看 wrapper 类型，HDF5 测试直接 new 了 collector，没有任何一处把产物喂给建库脚本。已新增 `test_collected_episode_builds_a_loadable_groot_artifact`：从 `_build_served_policy(... --collect-hdf5 ...)` 起，走两步推理 + `on_episode_end(success=True)`，再用**真 `build_artifact`** 产出 artifact，断言身份（`key_builder_type` / `checkpoint_id`）、几何（**三个** vision 槽 + `robot_state` 宽度）、内容（每步一条、`task_key` 正确、`next_ids` 成链），最后 `pickle` 落盘 → 真 `InMemoryBackend.load_artifact` → 经 facade 读回身份。另加 `test_a_failed_episode_yields_no_entries` 钉住建库脚本那条**静默**的 success 过滤。
- Accepted — **T-9 / T-10 用手搓替身冒充生产路径**。两处都属实，且指出的正是"生产代码回归了测试还会绿"。T-9：新增三例走**真 pickle**——写一个不含身份键的 artifact、经 `InMemoryBackend.load_artifact` 加载、断言 `artifact_meta` 读回为 `{None, None}` 并被守卫拒绝；对照一例含身份键的真 pickle 能通过；再一例是从未加载过 artifact 的真 backend，facade 返回 `None`（与 legacy 是两种不同形态）。T-10：把异常生命周期改成调**真 `run_one`**——用一个在第 N 次真推理上抛 `ConnectionError` 的 client（ctrl 帧不计入），断言异常照样往上抛、且 `episode_start`/`episode_end` 两帧都发出、`__success__` 为 False；另加两例真 `run_one`：每 episode 的 ctrl 帧成对且带 `__task__`/`__episode_id__`、以及 `__hit_meta__` → JSONL 的落盘。
- Accepted — **M2 的两个 benchmark 覆写仍直调模块级切片**。属实，计划要求二选一，我两样都没做。已取第一选项：`exp/cache_latency_bench/opt/r4_pool_keybuilder.py` 与 `r4_layout_check.py` 的 `build()` 覆写改走 `self._slice()`，两处各加注释说明理由（布局不同的子类若继承了批量 `build()` 却被 Pi0.5 的固定偏移切，是静默错误）。对 Pi0.5 逐位等价——基类 `_slice()` 就是原来那次调用。`grep _slice_cp1_fields exp/cache_latency_bench/opt/` 现为空。
- Accepted（前置已完成，**执行待授权**）— **G0-E 缺实机闭环证据**。判断成立：stub 的 probe 计数只证明本地分支记账，不能证明 server 真加载了库、`always_hit` 真返回库中动作、`__hit_meta__` 真穿过 adapter 上线、client 真写出 JSONL。本轮已把**全部前置**补齐（采集接线、真建库链路、判据实现、manual 套件）。⚠ **但执行被一处流程约束挡住**：三道门都在远端 `weilandserver` 跑，而远端 `/home/weiland/openpi` 是 `git clone`，按 **P5 已裁的 git 路线**新代码必须先 commit + push 到 `origin/Ziyang` 才能 `git pull` 取到（P5 明确排除了 `tether push` 手工投放，因为后续正式提交再 pull 会因 untracked-would-be-overwritten 中止）。而 commit 与 push 按 execution_authority §7/§8 均需**用户显式指示**，且当前处于 §5 G2「已暂存未提交」状态。⇒ 已向 owner 请求该授权；取得后按 §8 的 `[push]` 步骤送上远端，执行 T-8 与 G0-E，并把命令、正控制（`stage1_vision`/`cp1_sum`/`total_inference` 各 1）与负判据（`stage2_llm`/`stage2_action` 各 0）的实测计数补进下一轮。在证据落地前，本项**不视为已关闭**。

## 0. 上下文（无需对话历史）

### 0.1 一句话

让 **GR00T N1.5 的推理走 CP1 cache**：在「视觉 token 已散射进语言序列、尚未进 Qwen3 第 0 层」处把前向切成两段，stage1 的产物既是 cache 的 key 源、也是 stage2 的唯一输入；CP1 命中就整段跳过 stage2。

复用现有 `CacheOrchestrator` / `CacheStorage` / judge / gate / search strategy / 离线建库脚本；**不改 pi0.5 的模型前向，不改 `src/openpi/cache/interceptor.py`**。

### 0.2 这条线在做什么（背景）

检验 owner 的**跨场景继承假说**：cache 的 key 从 VLA 内部表征抽取，那么场景 A 上建的 cache 库，在**同任务、不同厨房**的场景 B 上是否仍可用。战场 = RoboCasa365 / Atomic split（18 任务），teacher 两个：pi0.5 与 GR00T N1.5。

⛔ **概念纪律**：teacher 是**固定底座，不是自变量**。自变量只有一个——cache 库建在哪个场景。teacher 只需满足①足够能干②两臂严格同一个。其训练分布、checkpoint 族、绝对 SR 高低都不影响本测量。（此前在这一点上犯过两次错，已撤回；见 handoff §2。）

前置准入门已全部完成：7 次准入门 1260 ep 全 P1 PASS，2×2 场景下 teacher 能力齐平（噪声底 GR00T-tp 5.6pp / pi0.5 6.7pp，六个偏移无一超出）。⇒ **cache 迁移若掉性能，不能归因于 teacher 在目标场景更弱**。

**当前缺口正是本计划**：上一轮只证明了「能做」（tap 点跑通、key 与 pi0.5 逐位一致、cache 内核可装进孤岛 B），**一行集成代码都没写**。

### 0.3 环境与资产（全部实机验证的绝对路径）

**远端主机 `weilandserver`**（与本机是两台机器），单张 RTX 4090，49140 MiB，**多 session 共用**。

⚠ 该 4090 是 owner 自己的硬件，但**由多个 session 共用**：端口 **8000** 上的 `serve_policy.py`（7764 MiB）与两个 `sidecar_server.py`（3392 + 2772 MiB）属于 owner 的**其它 session**，**绝不可关**。我方：pi0.5 用 8010、GR00T 用 8020。
⚠ **禁止宽模式 `pkill`**，只按自己的 tmux 名操作；`cssrv` / `cscol` / `rlr*` 属 owner 的其它 session。

三个互斥 venv 孤岛（2026-08-17 实测）：

| 孤岛 | 路径 | 版本 |
|---|---|---|
| **A（sim）** | `/home/weiland/Isaac-GR00T/gr00t/eval/sim/robocasa365/robocasa365_uv/.venv` | py3.12.13 / numpy 2.2.5；cwd 必须是 `.../external_dependencies/robocasa365` |
| **B（GR00T）** | venv `/home/weiland/gr00t_n15_venv/.venv`；**源码 `/home/weiland/gr00t_n15`**（worktree，**detached HEAD @ `4af2b62`**，非分支；工作区仅 2 个未跟踪 `.orig` 文件，已 diff 确认与被跟踪版本逐字节相同） | py3.11.15 / numpy 1.26.4 / torch 2.5.1+cu124 / transformers 4.51.3 |
| 主 venv | `/home/weiland/openpi/.venv` | py3.11.15 / numpy 1.26.4 / torch 2.7.1+cu126 |

⚠⚠ **`gr00t` 不在孤岛 B 的 venv 里**，来自 worktree，**必须进 `PYTHONPATH`**。漏了会让 `importorskip` 静默走跳过分支，**看起来通过实为没跑**。
⚠ 远端仓库副本是 **`/home/weiland/openpi`**（不是本地 `/home/weiland/projects/openpi`），是 `git clone`，靠 `git pull --ff-only origin Ziyang` 同步 —— 见 §9-P5。

**checkpoint**（owner 已裁 = target_posttrained 那一支）：

```
GR00T 选用  /home/weiland/ckpt_n15_robocasa_tp/gr00t_n1-5/foundation_model_learning/
            target_posttraining/atomic_seen/checkpoint-60000
GR00T 留档  /home/weiland/ckpt_n15_robocasa/gr00t_n1-5/multitask_learning/checkpoint-120000
pi0.5       /home/weiland/ckpt_pi05_robocasa_pytorch
```

**EGL**（该机无系统 EGL，孤岛 A 必需）：

```bash
export LD_LIBRARY_PATH=/home/weiland/nvidia-gl/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export __EGL_VENDOR_LIBRARY_DIRS=/home/weiland/nvidia-gl/root/usr/share/glvnd/egl_vendor.d
export MUJOCO_GL=egl
```

---

## 1. 目标 / 非目标

### 1.1 目标

| # | 目标 | 出场门 |
|---|---|---|
| G-1 | GR00T 前向可切成 stage1 / stage2 两段，与官方一次性 `get_action` **逐位相同** | §8 **G0-C** |
| G-2 | 从 stage1 产物按**掩码**抽出 `vision_0/1/2` + `prompt_emb` + `robot_state` | §8 **G0-D1** |
| G-3 | GR00T server 吃 cache yaml，CP1 FULL_HIT 跳过 stage2、MISS 走全程 | §8 **G0-E** |
| G-4 | 能从 GR00T rollout 采出 HDF5 并建成 `.pkl` 库（离线脚本复用） | §8 **G0-D1 / D2** |
| G-5 | 新增代码在孤岛 B（无 jax）可导入，且不误引 `openpi.collect` | §8 **G0-B'** |

### 1.2 明确不做

- **WARM_START / CP2 / CP3**：owner 已裁「只分两阶段」。judge 只允许 FULL_HIT / MISS 二元，由**加载期**守卫强制（§4-M4），不是运行期 raise。
- **warmup**：owner 裁定不考虑。
  ⚠ **依据更正**：早先写的「`config.py` 的 `samples_source` 只支持 `offline`」**是错的** —— `config.py:224-235` 的默认值就是 `type="warmup"`，`:1263-1272` 两种都收，`:3039-3074` warmup 分支已实现。**真正让本线零工作量的理由是**：`calibration` 只挂在 `composite` judge 上（`:1265` `if judge.calibration is not None`），本线用二元 judge，根本碰不到这个字段。
- **BatchingCoordinator / 并发批处理**、**routing / SidecarExecutor**、**X14 MlpRouterJudge**、**Qdrant backend**、**ProjectionKeyBuilder**：不接。
- **任何训练 / 微调**。
- **正式实验的建库与评测跑批**：本计划只交付「能跑」。⚠ 但 G0-E 需要一个**冒烟库**，其规模已量化并计入 §7：**≤ 3 episode ≈ 240 步**。
- **产出正式实验用的 tau / threshold 数值**：本计划只核验**标定机制**对 GR00T 数据可用（§4-M8），真正的标定值属跑批 plan。
- **pi0.5 在 RoboCasa 上的 cache**：见 §9-P2，另起 plan。

---

## 2. 事实基础（全部亲验；括号内为证据）

### 2.1 pi0.5 侧现状

```
stage1 = _stage1_token_prep     → Stage1Output(state, prefix_embs, prefix_pad_masks,
                                               prefix_att_2d_masks_4d, prefix_position_ids)
stage2 = _stage2_llm_backbone   → Stage2Output(stage1, past_key_values)
stage3 = _stage3_action_expert  → Stage3Output(action_chunk, intermediates)
```
`src/openpi/models_pytorch/pi0_pytorch.py:21-104`、`:569-597`。`interceptor.py:20` — `FULL_HIT: skip Stage 2 + 3`。

**Orchestrator 与模型无关**：`check(checkpoint_id, *, request_context=None, **stage_outputs)` 把 `stage_outputs` 原样转给 `key_builder.collect`（`orchestrator.py:471-508`）。**只有 KeyBuilder 认识 `stage1.prefix_embs`**。⇒ orchestrator / storage / judge / gate / strategy 可整体复用。

### 2.2 孤岛 B 的导入现实（2026-08-17 实跑，13/13）

```
OK openpi.cache                         OK openpi.cache.components.search_strategy
OK openpi.cache.orchestrator            OK openpi.cache.backends.in_memory_backend
OK openpi.cache.config                  OK openpi.cache.backend_pool
OK openpi.cache.cache_storage           OK openpi.cache.timing
OK openpi.cache.components.key_builder  OK openpi.cache.storage_types
OK openpi.cache.components.judge        OK openpi.cache.warmup_pool
OK openpi.cache.components.gate         OK openpi.cache.sidecar_executor
OK openpi.serving.websocket_policy_server   OK openpi.serving.monitor
FAIL openpi.cache.interceptor      ModuleNotFoundError: No module named 'jax'   (:75)
FAIL openpi.collect.collection_policy   ModuleNotFoundError: No module named 'jax'   (:7)
OK   openpi.collect.data_collector      ← 只依赖 h5py/numpy，可复用
```
三方件：`h5py 3.16.0 / yaml 6.0.3 / pytest 9.1.1 / websockets 17.0.1` 全在；缺 `qdrant_client` / `jax`。
⚠ 更正 handoff §5.3 的旧结论「缺 jax/websockets/qdrant」——**websockets 现在有**。

`openpi/cache/__init__.py` 不 import interceptor ⇒ `import openpi.cache` 在孤岛 B 安全。

### 2.3 GR00T 调用链

```
Gr00tPolicy.get_action(obs)                                       policy.py:146-186
  ├─ _check_state_is_batched → unsqueeze → np.array 循环          policy.py:171-178
  ├─ normalized_input = apply_transforms(obs)                     policy.py:180
  ├─ with inference_mode + autocast(cuda, bfloat16):              policy.py:190
  │     model.get_action(normalized_input)                        gr00t_n1.py:171-180
  │       ├─ backbone_inputs, action_inputs = prepare_input()     gr00t_n1.py:182-197
  │       ├─ backbone(backbone_inputs)                            eagle_backbone.py:115-133
  │       │    ├─ set_frozen_modules_to_eval_mode()               eagle_backbone.py:116
  │       │    └─ forward_eagle → eagle_model(**eagle_input,
  │       │         output_hidden_states=True, return_dict=True)  eagle_backbone.py:109
  │       │       → hidden_states[select_layer=12] → eagle_linear eagle_backbone.py:110-112
  │       ├─ action_head.get_action(backbone_out, action_inputs)  flow_matching…:350-413
  │       └─ validate_data(..., is_training=False)                gr00t_n1.py:179
  └─ unapply_transforms({"action": …})                            policy.py:196-197
```

### 2.4 切点（owner 已裁）——`modeling_eagle2_5_vl.py`，类 `Eagle2_5_VLForConditionalGeneration`

```python
218: def forward(self, pixel_values, input_ids=None, attention_mask=None, position_ids=None,
               image_flags=None, past_key_values=None, labels=None, use_cache=None,
               output_attentions=None, output_hidden_states=None, return_dict=None,
               num_tiles_list=None):
235:     input_embeds = self.language_model.get_input_embeddings()(input_ids)
237:     vit_embeds   = self.extract_feature(pixel_values)
239-241: if image_flags is not None: ...            # 我方路径 image_flags 恒 None
243:     B, N, C = input_embeds.shape
244:     input_embeds = input_embeds.reshape(B * N, C)
246:     input_ids    = input_ids.reshape(B * N)
247:     selected     = input_ids == self.image_token_index
248:     try:
249:         input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
250-257: except Exception as e:  ...                # shape 不匹配时的兜底重排 —— 见下
259:     input_embeds = input_embeds.reshape(B, N, C)     # ★ tap 点
261:     outputs = self.language_model(inputs_embeds=input_embeds, ...)   # ← Qwen3 第 0 层
```

⚠ **`248-257` 的 `try/except` 兜底分支我方不复刻**（B=1 且 token 数精确匹配，该支不会触发）。这是**有意的语义缺口**，M1 里显式声明并加断言，而不是靠"抄漏了"。

**为什么是 `input_embeds` 而不是 `vit_embeds`**：stage1 的产物必须能**直接喂 stage2**。若只吐 `vit_embeds`，stage2 还得自己重做语言 embedding + 散射，切分不干净、计时不准，也与 pi0.5 的 `prefix_embs` 语义不对应。
🟢 数值上不冲突：`input_embeds` 图像位置的值**就是** `vit_embeds` 散射进去的。

**⚠ 指纹要钉执行副本，不是 repo 副本**：`eagle_backbone.py:50-51` 走 `AutoConfig/AutoModel.from_pretrained(..., trust_remote_code=True)`，config.json 的 `auto_map` 使实际执行的类来自 HF 动态模块缓存 `/home/weiland/.cache/huggingface/modules/transformers_modules/eagle2_hg_model/modeling_eagle2_5_vl.py`（当前与 repo 副本 byte-identical）。⇒ D2 的守卫必须哈希 `type(model.backbone.eagle_model).forward`。

### 2.5 真实 token 布局（2026-08-17 孤岛 B 实测，CPU，未加载权重）

```
eagle_input_ids       [1, 813] int64     eagle_pixel_values   [3, 3, 224, 224] float32
eagle_attention_mask  [1, 813] (全 1)     state                [1, 1, 64] float32
eagle_image_sizes     [3, 2]   int64     state_mask           [1, 1, 64] bool
                                          embodiment_id        tensor([31])

image_token_index = 151669
图像 token 768 = 3 × 256      非图像 token 45
连续图像段 (start, len) = [(20, 256), (283, 256), (546, 256)]
state_mask 有效 20 / 64 维，索引 0..19 连续
```
⚠ `state` 的 dtype **随线上输入而定**（本探针喂 fp32 故为 fp32；喂 fp64 则为 fp64），不是固定值。真正决定下游的是 `prepare_input` 的统一 cast —— 见 §2.7(b)。

**⚠⚠ 决定性事实：三段图像 token 的起始偏移随 prompt 长度浮动**（本例 20/283/546，来自 `"pick the mug"` 的分词）。pi0.5 的固定偏移表（`key_builder.py:142-147`，0/256/512/768）**不能照搬**——必须每步按 `input_ids == image_token_index` 现算掩码。**这是本计划最容易写错、且写错了照样"能跑出数"的地方。**

**eagle 键集合恰好 4 个**（`transforms.py:70-75` 加 `eagle_` 前缀，processor 图像分支只产 `pixel_values`+`image_sizes`）。删掉 `image_sizes` 后是 `{input_ids, attention_mask, pixel_values}` ⇒ `image_flags` / `num_tiles_list` 恒 None（且 `num_tiles_list` 在 `forward` 体内**从未被引用**，是死参数）。

相机顺序 = `groot_keys.VIDEO_KEYS` 顺序（agentview_left / agentview_right / eye_in_hand）。

### 2.6 ckpt 实配（读 `checkpoint-60000/config.json`）

```
action_horizon 16   action_dim 32
action_head_cfg: num_inference_timesteps 4, backbone_embedding_dim 2048, use_vlln (默认 True)
backbone_cfg:    select_layer 12, project_to_dim None, tune_visual True, tune_llm False
eagle config:    select_layer -1, downsample_ratio 0.5, use_pixel_shuffle False,
                 mlp_connector_layers 1, image_token_index 151669
                 text_config.use_cache false, num_hidden_layers 28（pop 未同步 config）
```

四条要点：

1. **`project_to_dim: None` ⇒ `eagle_linear = torch.nn.Identity()`**（`eagle_backbone.py:53-56`）。`backbone_features == hidden_states[12]`，维度 2048。
2. **两个 `select_layer` 不是一回事**：`backbone_cfg.select_layer=12` 是 **LLM 层**；eagle config 的 `select_layer=-1` 是**视觉塔层**（`modeling_eagle2_5_vl.py:311-322`）。混淆会静默拿错张量。
3. **`hidden_states[12]` 的准确语义**（2026-08-17 读 transformers 4.51.3 `modeling_qwen3.py` 实证）：循环内先 push 每层**输入**，循环后 `hidden_states = self.norm(...)` 再 push 一次，且 `last_hidden_state` 就是它。⇒ tuple 长 **13**，`[12]` = **第 11 层输出再过 final RMSNorm 之后**，与 `last_hidden_state` **同一对象**。
   ⇒ 「直接调 `language_model`」与「经 `Eagle2_5_VL.forward`」拿到的是同一张量，**不存在 norm 前/后差异**。
   ⚠ 但 `num_hidden_layers` 仍是 28（pop 没同步）。⇒ 一旦 `select_layer != len(layers)`，`hidden_states[select_layer]` 既不是最后一个也不过 final norm。**必须断言 `select_layer == len(language_model.model.layers)`**。
4. `num_inference_timesteps=4` ⇒ action head 跑 4 步 Euler，每步都对 `vl_embs` 做 cross-attention。

### 2.7 ⚠⚠ dtype 与 autocast（本轮新增，是两道门的成败关键）

**(a) `vlln` 在 autocast 下会被提升到 fp32，差异是宏观量级。** `flow_matching_action_head.py:199-201` 的 `vlln = nn.LayerNorm(2048)` 在 `:265` `process_backbone_output` 里无条件施加。`layer_norm` 在 CUDA autocast 的 **fp32 名单**上。孤岛 B 实测（bf16 权重 + bf16 输入）：

```
无 autocast → 输出 torch.bfloat16
有 autocast → 输出 torch.float32     bitwise equal: False   max|Δ| = 0.0137
[对照] Linear → 两者都 bf16，逐位相等   ⇒ 确属 LN 特有
```

差异发生在 action head 入口，之后 4 步 Euler 全被污染。⇒ **在线路径、采集器、测试三处必须处于同一 autocast 策略下**，否则 G0-D 必挂且看不出原因。这一条催生 **D10**。

⚠ **原先给的理由（「分开开 context 会让 bf16 累加顺序变」）是错的**：autocast 的 weight-cast 缓存只对 `float32 && requires_grad && is_leaf` 生效，本模型参数已是 bf16，压根没有可缓存的转换；cast 本身确定性且幂等。⇒ **CP1 命中导致 stage2 不执行、context 进出次数变化，对 MISS 路径数值没有任何影响。**「统一开一次 context」的结论仍对，但正确理由是上面的 (a)。

**(b) `state` 进 stage1 后是 bf16，不是 fp32。** `gr00t_n1.py:187-196` `prepare_input` 对所有 floating 张量 `.to(self.device, dtype=self.action_head.dtype)`；模型以 `torch_dtype=COMPUTE_DTYPE=torch.bfloat16` 加载（`policy.py:32,240`）⇒ **`action_head.dtype` 是 bfloat16**。transform 产出的 `state` 是 fp32，进 stage1 后变 bf16（~3 位十进制有效数字）；`eagle_pixel_values` 同理。

⇒ **若采集器从 `normalized_input["state"]`（fp32）取、在线路径从 `stage1.state`（bf16）取，G0-D 必挂。** 催生 **D5** 的修订。

**(c) ⚠⚠ inference tensor 逃逸：在 context *内*做 `.cpu().float()` 是逃不掉的。** 2026-08-17 实测（torch 2.7.1）：

```
inference_mode 内：.detach().cpu().float().contiguous() → is_inference: True
                   .clone()                             → is_inference: True
context 外：       a.clone()          → is_inference: False
                   empty_like+copy_   → is_inference: False
                   a.add_(1.0)        → RuntimeError: Inplace update to inference
                                        tensor outside InferenceMode is not allowed
                   c.add_(1.0)        → OK（c 是 context 外 clone 的）
```

⇒ **直觉做法「`buffer_for_write` 之前 `.float().cpu()`」在这里是无效的** —— 那次转换发生在 session 内，产物仍是 inference tensor，跨 step 存活后被 storage / normalizer 就地改写就会炸。**唯一有效的逃逸是在退出 inference_mode 之后再复制。**

🟢 **且好消息是这不需要额外拷贝**：实测在 session 外用 inference tensor 做**非就地**运算，产物就是普通张量：

```
stage1 的 input_embeds（session 内造）      is_inference: True
session 外切片 + .float() + adaptive_avg_pool2d + .cpu().contiguous()
  → key                                    is_inference: False, dtype fp32
  → key.add_(1.0)                          OK ⇒ 可安全跨 step 存活
```

⇒ **把 CP1 检查整体移出 session** 就同时解决三件事：query_keys 与 action 都成为普通张量；池化天然在 fp32 下进行（正是 D11 想要的，且与离线路径一致）；session 边界收缩到只包住两段前向。催生 **D10 的修订**。

### 2.8 离线建库脚本的复用性

`exp/common/build_in_memory_cache_artifact.py:289-316` `_build_fake_stage1`：

```python
parts = [vision_0, vision_1, vision_2]          # 各 [256, emb_dim]，缺则零填
parts.append(prompt_emb)                        # [num_tokens, emb_dim]，变长不设上限
prefix_embs = torch.cat(parts, dim=0).unsqueeze(0)
```

HDF5 **按字段分开存**，重建时按 `[vision_0|vision_1|vision_2|prompt]` 拼回，再由 `_slice_cp1_fields` 按固定偏移切开。对 GR00T 而言这个来回**恰好正确**——偏移由重建过程自己保证。

⇒ **离线建库脚本的 pool 系路径可复用**（`:896,:919` 证明 `_self_check_tokenizer_consistency` 只在 `cp1_llm_layer_extract` 分支调，pool builder 走不到；pi0.5 相关 import 全在该分支惰性做 ⇒ 孤岛无关）。

⚠ **但「唯一差异是 `robot_state: 32`」是错的，实际有两处**：

1. **`_VECTOR_DIMS`（`:46-54`）五个 pool 条目一个都没有 `vision_2`** —— LIBERO 只有 2 个相机。GR00T 是 **3 个**。对照：`_LLM_LAYER_EXTRACT_DIMS`（`:59-75`）的 `per_modality_*` 条目**是有** `vision_2` 的 ⇒ 不是脚本不支持三相机，是 pool 系那张表缺条目。
   ⚠⚠ **不补会静默出错**：yaml 若声明 `vision_2`，`in_memory_backend.py:233-237` 抛 `Artifact vector_dims mismatch`（响，卡住）；yaml 若不声明，`config.py:1430-1438` 会逼你 `keys.vision_2.enabled=false`，而 builder 仍产出 `vision_2` 键、`in_memory_backend.py:506-507` 的 `if field_name not in self._dims: continue` **静默丢弃**它 ⇒ **eye_in_hand 相机根本没进检索键，跑得出数、数没意义**。
2. `robot_state` 写死 32（字面量在该文件共 11 处：`:47,48,50,52,53,60,63,66,70,74,146`）。

⇒ M5 需要的是 `_get_vector_dims(..., vision_slots: int = 2, robot_state_dim: int = 32)`，对查表结果做后处理（增删 `vision_2`、覆盖 `robot_state`）。**默认 (2, 32) ⇒ pi0.5 逐位不变。**

### 2.9 ⚠⚠ 在线 / 离线 key 的 dtype 路径不同 —— G0-D 的判据必须据此重写

```
在线：input_embeds (bf16) → _spatial_pool_tokens 在 bf16 下 adaptive_avg_pool2d
                          → _to_cpu_float32（池化之后才转 fp32）   key_builder.py:206-218
离线：HDF5 fp16 → _build_fake_stage1 的 .float()                  build_…:301-311
                          → 同一函数在 fp32 下 adaptive_avg_pool2d
```

两层差异：①**池化累加 dtype 不同**（bf16 vs fp32）；②bf16(8 指数/7 尾数) 存成 fp16(5/10) **本身有损**。

⇒ **「两条路逐位相等」由构造决定就不可能达成** —— 把它当判据会第一次跑就红，而根因不是掩码逻辑。

🟢 顺带发现：**pi0.5 自己也是这个状况**——它的在线/离线 key 同样不逐位相等，只是**从来没人测过**（`tests/` 下相关断言全是「同一份 builder 代码自比」，不是金标准）。

**处置（催生 D11 与 §8 的门拆分）**：
- 在线侧在 reduce 之前统一 `.float()`，让**池化 dtype 与离线一致**（D11）；
- HDF5 保持 fp16（与 pi0.5 一致、省 2× 磁盘），残余差异只剩一次 bf16→fp16→fp32 存储往返；
- G0-D 拆成 **D1（结构，逐位）** 与 **D2（端到端，容差）**，见 §8。

⚠ **`load_artifact` 只校验 `vector_dims`，不校验 `key_builder_type`**（`in_memory_backend.py:233-236`）。而 GR00T 与 pi0.5 在 `cp1_spatial_pool_16` 下 vision 维度**都是 32768**——唯一挡住误加载的是 `robot_state`（32 vs 20）。⇒ 见 R10。

**跨孤岛 pickle 无风险（实证）**：pkl 由主 venv 建、孤岛 B 加载，两端同为 py3.11 + numpy 1.26.4；孤岛 A（py3.12/numpy 2.2.5）从不碰 pkl。且 pkl 里**没有 torch 对象**（`:807-815` 落盘前 `.numpy()`，`in_memory_backend.py:250-258` 读回时 `from_numpy`）。孤岛 B 直接 load 主 venv 造的真件成功：1018 entries / 3.5 s / peak RSS 0.80 GB。

### 2.10 ⚠ 本仓库已有的迁移指南（WA §2.2 依据）

`docs/cache/migration.md`（792 行，标题即「Cache 框架迁移指南」，WA §8 已注册）给出了非 Pi0.5 模型接入的**既定路径**，本计划的形状与它一致：

| migration.md | 本计划 |
|---|---|
| `:55-63` 迁移路径判定表：「能拆 stage？→ Staged API」「能取中间层 embedding？→ 自定义 KeyBuilder」「要在线缓存？→ 完整走 Step 1-7」 | 三问皆「是」⇒ 走完整路径 |
| `:306-405` **Step 4 实现自定义 Interceptor**（模板 + 「命中时 `payload.action_chunk` 无 batch dim」「`query_keys` 所有路径都填」「每步必须 `clear()`」） | D1 / M3 |
| `:407-445` **Step 5 注册到配置系统**：`:411` 「在 `_build_key_builder()` 中添加你的 builder」；`:428-430` 「在 `validate_cache_config()` 中确保新 type 被识别」；`:445` 「`vector_dims` 必须与 build() 输出一致」 | D6 / M2 / M4 |
| `:447-556` Step 6 数据收集与 artifact 构建；`:495` 明说 `collect/` 是 pi0.5 参考实现、其它模型需自写 | M5 |
| `:758-762` **陷阱「视觉 Token 布局不同」** —— 正是 D4 | D4 |
| `:768` 「`CachePayload.action_chunk` 形状无硬编码限制」 | 支持 GR00T 的 `[16, 32]` |
| `:792` 「建议先用 `always_hit` + 离线分析来确定合适的阈值」 | G0-E 判据 / M8 |

⇒ D1（平行实现自定义 Interceptor）与 D6（在 `_build_key_builder` 注册）**不是本计划的发明，是仓内既定答案**。

### 2.11 server / client 现状

- `WebsocketPolicyServer` 用 `hasattr` 软探测 `on_task_begin` / `on_task_end`（`:572/584/590`），以**关键字**调 `on_episode_start(experiment=, task=, episode_id=, episode_name=, extra_metadata=)`（`:635-641`）与 `on_episode_end(success=…)`（`:648`）。⇒ **server 零改动**。
  ⚠ 需转接：`CacheOrchestrator.on_episode_end(self)`（`orchestrator.py:741`）**不收 `success`**，`on_episode_start` 参数名也不同 —— pi0.5 侧 `interceptor.py:383-391` 正是这么吞掉的，照抄即可。
  ⚠ `:647-656` 的 episode_end 分支会 `from openpi.serving import monitor` 并 `fire_event` —— 孤岛 B 实测该模块可导入（§2.2），但这是本线**第一次**执行该路径。
- `exp/robocasa365/groot_rollout_client.py:192` 只调 `client.infer(...)["actions"]`，**从不发 `__ctrl__`**，也**不读 `__hit_meta__`** ⇒ orchestrator 的 episode 生命周期收不到边界，且命中数据无落盘处。**必须补**（M6）。
- `scripts/serve_policy.py:24-47` `_enforce_runtime_write_policy` 强制 `write_policy.type == "never"`。
- `serve_groot_n15.py:220-222` 的启动**握手**发生在任何客户端连接之前（`on_task_begin` / `on_episode_start` 都还没触发），会照常走一遍 CP1 并推进 `_step_counter`。**必须处置**（M4）。
- `serve_groot_n15.py:59` `DEFAULT_CHECKPOINT` **仍指向留档的 mt `checkpoint-120000`**，而 docstring 的 tmux 启动命令**不传 `--checkpoint`** ⇒ 照 docstring 启动 = **静默用错 teacher**。**必须修**（M4）。
- `serve_policy.py:50-57` 的 `EnvMode` 无 robocasa —— 但我们不走 `serve_policy.py`，**不构成障碍**（handoff §5.4-4 把它列为障碍是过度陈述，此处更正）。

---

## 3. 架构决策

### D1 —— 平行实现，不抽象 `InferenceInterceptor`

**裁定：新写 GR00T 专用拦截器，`src/openpi/cache/interceptor.py` 一行不改。**

1. `interceptor.py:75` 模块级 `import jax`，孤岛 B 导不进（§2.2 实测）。抽象就要拆 pi0.5 的导入图，收益为零。
2. 它 1065 行里的 coordinator 路由、routing sidecar 三态分派、meta device 哨兵、WARM_START、CP3 —— **本计划一项都不用**。
3. WA §2.5：新功能以 interceptor / wrapper / hook 接入，不改推理内部。
4. **`docs/cache/migration.md:306` Step 4 就是这么规定的**（§2.10）。

**代价与接受**：生命周期转发、`__hit_meta__` 组装等约 150 行与 pi0.5 形似而非同。接受。

### D2 —— 在我方代码里复刻 235-259，不改 gr00t 源码

**裁定：`GrootStagedRunner.run_stage1` 复刻 `modeling_eagle2_5_vl.py:235-259`（我方分支实际 **8 条可执行行**：235/237/243/244/246/247/249/259；不含 248-257 兜底分支）；不 monkeypatch、不改 worktree。**

gr00t 源码在仓外 worktree，改了不进版本控制、G2 审不到、CI 看不见。monkeypatch `forward` 是全局副作用。
⚠ 别把 `check_forward_kwargs`（`:173-178`）当成签名保护：它只 `assert not any(k.kind == VAR_KEYWORD ...)`（**仅禁 `**kwargs`，不钉参数名/顺序**），且在 `__init__`（`:171`）执行，事后 monkeypatch 根本不会触发它。不 monkeypatch 的理由是全局副作用与不可审计。

**防漂移**：启动时 `sha256(inspect.getsource(type(model.backbone.eagle_model).forward))` 与钉死常量比对，不符**立刻 raise 并打印实际值 + `type(...).__module__`**。⚠ 必须取 **`type(model.backbone.eagle_model)`**（HF 动态模块缓存里的执行副本），不是 import repo 文件 —— 后者既哈希错对象，又与「M1 不 import gr00t」的原则冲突（§2.4）。

### D3 —— stage3 并入 stage2，不留空壳

**裁定：两阶段。`run_stage2` = Qwen3(12 层) + `eagle_linear` + `action_head.get_action` + `validate_data`。不定义 `run_stage3`，`CheckpointID.CP3` 保持 disabled。**

留 no-op 假阶段只会误导后来读代码的人。
⚠ `stage3` 在 `interceptor.py` 里渗透得很深 —— `grep -n stage3` 约 30 行，还有 `stage3_warm` probe（`:32/:290/:940`）、`run_stage3_from`（`:22/:254-262/:645-663/:939`）、`_compiled_stage3_fn`（`:312/:330-340`）、coordinator 路由（`:237-256`）、WARM_START 分支（`:917-944`）。这**加强** D1/D3：pi0.5 的三阶段假设与其代码耦合很紧，更不该硬塞给 GR00T。

**但计时分开**：`run_stage2` 内注册 `stage2_llm`（Qwen3 + eagle_linear）与 `stage2_action`（4 步 flow matching）两个 probe。**故意不复用 `stage3_flow` 这个名字**，免得下游分析脚本把 GR00T 数据当成 pi0.5 的三阶段数据。

### D4 —— key 用掩码切，不用偏移

**裁定：`vision_i` = `input_embeds[0]` 上第 i 段连续图像 token（由 `input_ids == image_token_index` 现算）；`prompt_emb` = 全部非图像 token。**

§2.5 实测偏移随 prompt 长度浮动。硬编码偏移会切到文本 token 上，且**不会报错**——距离照样算得出来，只是全是噪声。`docs/cache/migration.md:758-762` 已把这列为头号陷阱。

**运行时断言**：恰好 3 段连续图像段；每段长度恰为 256；总图像 token 数 == `pixel_values.shape[0] * 256`。

### D5 —— `robot_state` 取 mask 有效位，**在线与离线都从 `stage1.state` 取**

**裁定：`robot_state = stage1.state[0, -1][state_mask[0, -1]]`，20 维；落盘/建库时统一 `.float()`。**

pi0.5 的 `Stage1Output.state` 也是**输入变换之后**的状态（`pi0_pytorch.py:27-32`）。取有效位而非全 64 维，是为了不把 44 个恒零 padding 灌进 L2 距离。

⚠ **dtype 修订**：§2.7(b) 实测 `stage1.state` 是 **bf16**（`prepare_input` casts to `action_head.dtype`），不是 fp32。⇒ **采集器绝不可从 `normalized_input["state"]`（fp32）取** —— 必须与在线路径同源，都取 `stage1.state`，落盘时才 `.float()`。否则 G0-D 必挂。

**首步锁定 mask**：第一步记下有效索引集合，之后每步断言不变；变了就 raise。

### D6 —— 代码落在 `src/openpi/cache/groot/`

**裁定：新建子包 `src/openpi/cache/groot/`；`exp/robocasa365/` 只放实验接线（server 参数、采集 driver、yaml）。**

1. `config.py` 的 `_build_key_builder` 要能构造它（`migration.md:411` 的既定路径）。放 `exp/` 会造成 `src → exp` 反向依赖。
2. 它有测试、有架构文档条目（L3 要求），不是"实验脚本"（WA §4 对 `exp/` 的定义）。
3. 保持 jax-free ⇒ 孤岛 B 可导入。

✅ **owner 已裁（2026-08-17）：落 `src/openpi/cache/groot/`。** 备选（`exp/` + 注册钩子）不再展开。
备案理由：**两个方案都要改 `src/openpi/cache/config.py`**（备选需在那儿加 `register_key_builder` 钩子），WA §1「Pi0.5 only」的边界一条也没被备选消掉，真正差别只是「4 个 builder 类放哪」。

### D7 —— 拦截器实现 `get_action`，注入 adapter；**adapter 必须改**

**裁定：`GrootCacheInterceptor` 满足 `_ActionPolicy` 协议（`get_action(observations) -> dict`），注入现有 `GrootPolicyAdapter`。**

`GrootPolicyAdapter` 的职责是**线格式校验 + 观测翻译**，应留在最外层，先校验再进 cache；放外面会让非法观测先进 key builder。

⚠ **修订（原文写错了）**：原先说「adapter 只需加 4 个生命周期钩子（≈12 行），校验逻辑不动」——**不成立**。`groot_policy_adapter.py:125-127`：

```python
unexpected = [key for key in raw if key not in groot_keys.ACTION_DIMS]
if unexpected:
    raise ValueError(f"policy output has unexpected action keys: {unexpected}")
```

且 `:203` `return {"actions": validate_action_chunk(raw)}` **只回传 `actions`**。⇒ `__hit_meta__` 一进 `raw` 就在**第一次推理（含启动握手）**炸，即便放行也永远上不了线。

**因此 adapter 需要三处加法式改动**：①保留字段白名单（`__hit_meta__` 等 `__`-前缀字段从 `raw` 中先剥离，不进 `validate_action_chunk`）；②剥离出的字段并入返回 envelope（`{"actions": …, "__hit_meta__": …}`）；③四个生命周期钩子透传（注入对象没有该方法时 no-op）。三处都要有测试。

### D8 —— 库离线建，server 强制 `write_policy: never`

沿用 pi0.5 纪律（`serve_policy.py:24-47`）。写路径在线开启会让"库里有什么"依赖跑批顺序，跨场景实验的自变量就不干净了。

⚠ `migration.md:449` 说 cache 可以「空库启动 + 在线累积」——**本线明确不用**，理由如上。

### D9 —— 类型名用 `cp1_groot_*` 前缀（不是 `groot_cp1_*`）

**裁定：yaml `key_builder.type` 取名 `cp1_groot_mean_pool` / `cp1_groot_spatial_pool_16` / `cp1_groot_spatial_pool_4` / `cp1_groot_max_pool`。**

`config.py:2050` 与 `:2204` 两条校验按 `_effective_kb_type.startswith("cp1_")` 触发：

- `:2050` 要求 `vision_0` 与 `robot_state` 必须 enabled；
- `:2204` 要求 `in_memory` backend 必须给 `preload_path`。

这两条对 GR00T **同样应当生效**。命名成 `groot_cp1_*` 会让它们**静默失效**（漏掉 `preload_path` 都不报错）；命名成 `cp1_groot_*` 则零改动自动继承。

### D10 —— autocast 上下文由 runner 拥有并强制断言

**裁定：`GrootStagedRunner` 提供 `session()` contextmanager（`inference_mode()` + `autocast("cuda", bfloat16)`）；`run_stage1` / `run_stage2` 入口断言 `torch.is_autocast_enabled()` 且 dtype 为 bfloat16，不满足直接 raise。**

**⚠ session 只包住两段前向本身，CP1 检查与所有跨 step 存活的张量都在 session 外**：

```
with runner.session():   stage1 = run_stage1(obs)          # 只有前向
# ── 出 session ──
cp1 = orchestrator.check(CP1, stage1=stage1)               # key 在这里造 ⇒ 普通张量、fp32 池化
if MISS:
    with runner.session():  stage2 = run_stage2(stage1)    # 只有前向
action_cpu = chunk.detach().cpu().float().contiguous().clone()   # 出 session 后才复制
```

两条理由，各自独立且都必须满足：
1. **autocast 一致性**（§2.7a）：`vlln` 在 autocast 下走 fp32、否则走 bf16，`max|Δ| = 0.0137`。在线/采集/测试任一处漏开就静默错值 ⇒ context 收进 runner + 入口断言，让"漏开"变成**立刻报错**。
2. **inference tensor 逃逸**（§2.7c）：在 session *内* 做 `.cpu().float()` 产物**仍是** inference tensor，跨 step 存活后被就地改写即 `RuntimeError`。⇒ 复制必须发生在**退出 inference_mode 之后**。

⚠ 两段前向分开进出 session 是安全的：§2.7(a) 已更正「分开开 context 会改变累加顺序」是错的（autocast 的 cast 确定性且幂等，本模型参数已是 bf16 无可缓存的转换）；且 stage1 的输出被送回第二个 session 内使用，读 inference tensor 在 inference_mode 内完全合法。

配套：`__init__` 断言 `model.training is False`（`eagle_backbone.py:116` 的 `set_frozen_modules_to_eval_mode` 我方不复刻，其正确性完全依赖 eval 前提）。

### D11 —— 在线侧 reduce 前统一 `.float()`

**裁定：GR00T key builder 在调用 `_reduce_vision` / `_reduce_prompt` 之前把切片 cast 到 fp32。**

理由见 §2.9：离线路径在 fp32 下池化，在线路径若沿用 pi0.5 的写法会在 **bf16** 下池化 —— 累加 dtype 不同，G0-D 的结构判据无从谈起。统一到 fp32 后，两条路只剩一次 bf16→fp16→fp32 的**存储**往返，误差可界定、可测。

⚠ 这是 GR00T builder **相对 pi0.5 的一处有意偏离**，不是照抄。pi0.5 侧不动（它同样有这个不一致，但那是既有行为，改它属于本计划范围外的 drive-by）。

---

## 4. 交付单元

### 4.0 改动文件清单

| # | 文件 | 性质 | 单元 |
|---|---|---|---|
| 1 | `src/openpi/cache/groot/__init__.py` | 新增 | M1 |
| 2 | `src/openpi/cache/groot/staged.py` | 新增 | M1 |
| 3 | `src/openpi/cache/groot/key_builder.py` | 新增 | M2 |
| 4 | `src/openpi/cache/groot/interceptor.py` | 新增 | M3 |
| 5 | `src/openpi/cache/components/key_builder.py` | **修改**：`build()` 内联切片抽成可覆写 `self._slice()` | M2 |
| 6 | `src/openpi/cache/config.py` | **修改**：`_valid_key_builder_types` 加 4 项 + `_build_key_builder` 加 4 分支（惰性 import） | M2 |
| 7 | `exp/robocasa365/groot_policy_adapter.py` | **修改**：保留字段白名单 + envelope 透传 + 生命周期钩子 | M3 |
| 8 | `exp/robocasa365/serve_groot_n15.py` | **修改**：`--cache-config`、`DEFAULT_CHECKPOINT` 改 tp、握手期禁 cache、加载期守卫 | M4 |
| 9 | `exp/robocasa365/groot_cache_collector.py` | 新增 | M5 |
| 9a | `src/openpi/cache/cache_storage.py` | **修改**：加只读 `artifact_meta` facade（照 `library_stats` 先例），调用方不直穿 `_backend` | M4 |
| 9b | `src/openpi/cache/backends/in_memory_backend.py` | **修改**：`load_artifact` 加法式记下 `artifact_meta`（`key_builder_type` / `checkpoint_id`），零新校验 ⇒ pi0.5 逐位不变 | M4 |
| 10 | `exp/common/build_in_memory_cache_artifact.py` | **修改**：加 `--robot-state-dim`（默认 32 ⇒ pi0.5 逐位不变）+ artifact 写 GR00T 的 `key_builder_type` | M5 |
| 11 | `exp/robocasa365/groot_rollout_client.py` | **修改**：`__ctrl__` episode 边界 + `__hit_meta__` 落盘 | M6 |
| 12 | `exp/robocasa365/config/groot_cache_cp1.yaml` | 新增（冒烟用 cache yaml） | M4 |
| 13 | `docs/architecture/cache_system.md` | **修改**：新增 GR00T 两阶段一节 | M7 |
| 14 | `docs/cache/tutorial.md` | **修改**：登记 4 个新 `key_builder.type` | M7 |
| 15 | `docs/cache/migration.md` | **修改**：把 GR00T 列为第二个已落地案例 | M7 |
| 16 | `docs/README.md` / `logs/README.md` | **修改**：索引同步（WA §4 红线） | M7 |
| 17–24 | `tests/cache/groot/{__init__,test_groot_staged,test_groot_key_builder,test_groot_interceptor,test_online_offline_key_parity,test_import_isolation}.py`、`tests/robocasa365/{test_groot_cache_collector,test_groot_cache_manual}.py` | 新增 | §5 |

| 25 | `tests/cache/groot/test_groot_load_guard.py`、`tests/robocasa365/test_groot_client_ctrl.py` | 新增 | §5 T-9 / T-10 |

⇒ **新增 14 个、修改 14 个**。

### 4.1 公共接口（yaml 契约）

```yaml
key_builder:
  type: cp1_groot_spatial_pool_16      # 见 D9；另有 _mean_pool / _spatial_pool_4 / _max_pool

keys:                                   # cp1_* 前缀强制 vision_0 与 robot_state 必须 enabled
  vision_0:    {enabled: true}
  vision_1:    {enabled: true}
  vision_2:    {enabled: true}
  prompt_emb:  {enabled: true}
  robot_state: {enabled: true}

backend:
  type: in_memory
  vector_dims:                          # 必须与 build() 输出、与 artifact 三者一致
    vision_0: 32768                     # 16 × 2048
    vision_1: 32768
    vision_2: 32768
    prompt_emb: 2048
    robot_state: 20                     # ← GR00T 特有（pi0.5 是 32）
  in_memory:
    preload_path: <artifact.pkl>        # cp1_* 前缀强制必填

checkpoints:
  cp1: {enabled: true, gate: {type: always_search}, judge: {...}}
  cp3: {enabled: false}                 # 两阶段，无 CP3

write_policy: {type: never}             # D8，加载期强制
```

各 builder 的 vision 维度：`_mean_pool` 2048 / `_spatial_pool_16` **32768** / `_spatial_pool_4` 8192 / `_max_pool` 2048；`prompt_emb` 恒 2048；`robot_state` 恒 20。

### 4.2 集成点一览

| 缝 | 位置 | 方向 |
|---|---|---|
| stage 切分 | `type(model.backbone.eagle_model).forward` 的 235-259（复刻，不改） | 读 |
| key 抽取 | `CacheOrchestrator.check(CP1, stage1=…)` → `key_builder.collect/build` | 写入既有协议 |
| 配置注册 | `config.py` `_valid_key_builder_types` + `_build_key_builder` | 加分支（`migration.md:411,428`） |
| 切片钩子 | `components/key_builder.py` `_CP1BaseKeyBuilder.build()` → `self._slice()` | 行为保持重构 |
| policy 注入 | `GrootPolicyAdapter(policy=GrootCacheInterceptor(...))` | 加法 |
| 线协议 | `__ctrl__ episode_start/end`（既有 server 支持）、`__hit_meta__`（需 adapter 放行） | 加法 |
| 离线建库 | HDF5（既有 schema）→ `build_in_memory_cache_artifact.py`（pi0.5 builder） | 复用 + 1 个 CLI |
| 阈值标定 | `exp/common/calibrate_robot_state_tau.py`（吃 HDF5，模型无关） | 复用 |

### M1 —— `src/openpi/cache/groot/staged.py`

不 import `gr00t`（鸭子类型注入，照 `groot_policy_adapter.py` 先例），可在主 venv 用 stub 测。

```python
@dataclass
class GrootStage1Output:
    input_embeds: torch.Tensor       # [B, N, 2048] bf16，autocast 下产出
    attention_mask: torch.Tensor     # [B, N]
    image_token_mask: torch.Tensor   # [B, N] bool = (input_ids == image_token_index)
    action_inputs: Any               # 不透明 BatchFeature，原样转给 stage2
    @property
    def state(self) -> torch.Tensor: ...       # [B, T, 64] **bf16**（见 §2.7b）
    @property
    def state_mask(self) -> torch.Tensor: ...  # [B, T, 64] bool

@dataclass
class GrootStage2Output:
    action_pred: torch.Tensor        # [B, 16, 32]，**归一化空间**（未 unapply）

class GrootStagedRunner:
    UPSTREAM_FORWARD_SHA256 = "<钉死>"
    def __init__(self, model, *, timer: SystemTimer | None = None,
                 verify_upstream: bool = True): ...
    @contextmanager
    def session(self): ...           # inference_mode + autocast(cuda, bfloat16)
    def run_stage1(self, normalized_input: dict) -> GrootStage1Output: ...
    def run_stage2(self, stage1: GrootStage1Output) -> GrootStage2Output: ...
```

**probe 所有权** —— `run_stage2` 内部要分记两段，runner 就必须有 timer 入口，否则判据不可观测：

| probe | 谁注册 / 谁计 | 边界 |
|---|---|---|
| `stage1_vision` | **runner**（`run_stage1` 内） | 整个 stage1 |
| `stage2_llm` | **runner**（`run_stage2` 内，Qwen3 + `eagle_linear` 段） | 到 `backbone_features` 为止 |
| `stage2_action` | **runner**（`run_stage2` 内，action head 段） | 4 步 flow matching |
| `cp1_sum` | **interceptor** | `orchestrator.check(CP1, …)` |
| `total_inference` | **interceptor** | `get_action` 全程（CPU backend） |

`timer=None`（默认）⇒ runner 内部用一个 `SystemTimer(enabled=False)`，所有 `measure()` 变 no-op ⇒ **握手旁路直接传 `timer=None` 即可**，不必给 runner 加第二条代码路径。server 只把**同一个** `SystemTimer` 实例同时注入 runner 与 interceptor，probe 名字全局唯一、不重复注册。

⇒ **G0-E 的核心判据由此可观测**：FULL_HIT 时 `stage2_llm` / `stage2_action` 各 **0 次采样**，MISS 时各 **1 次**。

`__init__` 断言：`model.training is False`（D10）；`backbone.select_layer == len(eagle_model.language_model.model.layers)`（§2.6-3）；`UPSTREAM_FORWARD_SHA256` 匹配（D2）。

`run_stage1`：
1. `backbone_inputs, action_inputs = model.prepare_input(normalized_input)`（`gr00t_n1.py:182`）
2. `eagle_input = {k.removeprefix("eagle_"): v for k,v in backbone_inputs.items() if k.startswith("eagle_")}`；`del eagle_input["image_sizes"]`
3. **`assert set(eagle_input) == {"input_ids", "attention_mask", "pixel_values"}`** —— 比 `assert image_flags is None`（恒真、检不出任何东西）强：上游/processor 版本变化引入新键时，上游 `forward` 会 TypeError 而复刻会静默忽略
4. `assert input_ids.shape[0] == 1`（B=1；`padding_side="left"` 下 B>1 会引入左填充）
5. 断言 autocast 已开且为 bf16（D10）
6. 复刻 235-259（含 249 的 `*0.0 +` 写法；**不含 248-257 兜底分支**，§2.4）
7. 组装 `GrootStage1Output`

`run_stage2`：
1. 断言 autocast（D10）
2. `out = eagle_model.language_model(inputs_embeds=…, attention_mask=…, position_ids=None, past_key_values=None, use_cache=None, output_attentions=None, output_hidden_states=True)` —— 7 个参数逐个对齐 `:261-269`。⚠ **`return_dict` 上游没有传下去**（`:233` 算出的只用于 `:285-295` 的返回形态），传不传 `True` 都行，**唯一禁忌是传 `False`**
3. `feats = out.hidden_states[backbone.select_layer]`
4. `feats = backbone.eagle_linear(feats)`（本 ckpt 是 Identity，仍照调，换 ckpt 才不会错）
5. **新建** `BatchFeature({"backbone_features": feats, "backbone_attention_mask": attention_mask})` —— ⚠ 必须每次新建：`process_backbone_output`（`flow_matching_action_head.py:263-268`）会**就地写回** `backbone_output["backbone_features"]`，复用同一对象会二次施加 `vlln`
6. `model.action_head.get_action(backbone_outputs, action_inputs)` → `action_pred`
7. `model.validate_data(action_head_outputs, backbone_outputs, is_training=False)`（`gr00t_n1.py:179`，纯 shape 校验，漏了就没人挡 shape 回归）

⚠ **`GrootStage1Output` 不提供 `.to(device)`**。任何 `.contiguous()` / `.clone()` / `.to()` 都可能让 flash-attn 与 cuBLAS 选到不同 kernel ⇒ 归约顺序变 ⇒ Δ != 0。stage1→stage2 之间**只允许传原对象**。

**观察（非本计划改动，但要如实记账）**：`language_model(...)` 会算 lm_head 的 logits。`logits_to_keep=0` ⇒ `slice(-0, None) == slice(0, None)` ⇒ **813 个位置全算**（实证读 `modeling_qwen3.py`）。粗算 FLOPs：lm_head ≈ 2·813·2048·151936 ≈ **5.1e11**，12 层 Qwen3-1.7B ≈ **9.8e11** ⇒ **lm_head 约占 stage2-LLM 计算量的 34%，且是纯浪费**。
⇒ HIT 跳过 stage2 时这 34% 会被计入"cache 省下的时间"，**直接抬高 HIT 收益口径**。存在逐位等价的省法（直接调 `language_model.model(...)` 取 `last_hidden_state`，§2.6-3 已证同一张量）。**本计划不改**，但 §7 必须实测并如实标注该占比，不许只说一句"对比公平"。

### M2 —— `src/openpi/cache/groot/key_builder.py`

```python
def slice_groot_cp1_fields(input_embeds, image_token_mask, state, state_mask,
                           enabled) -> dict[str, torch.Tensor]:
    """按掩码切出 vision_0/1/2 + prompt_emb + robot_state（GR00T 偏移浮动，见 plan D4）。"""
```
断言见 D4；`robot_state` 见 D5。

四个具体类只覆写 `collect` / `_slice` / `_reduce_vision` / `_reduce_prompt`，复用 `components/key_builder.py` 已有的 `_mean_pool_tokens` / `_max_pool_tokens` / `_spatial_pool_tokens` / `_to_cpu_float32`。
⚠ **reduce 之前统一 `.float()`**（D11 / §2.9）—— 否则在线在 bf16 下池化、离线在 fp32 下池化，G0-D 的结构判据不成立。

⚠ **`collect()` 也必须覆写**（不是只抽 `_slice()` 就够）：`_CP1BaseKeyBuilder.collect`（`components/key_builder.py:238-243`）读的是 `s1.state` 与 **`s1.prefix_embs`**，而 `GrootStage1Output` 没有 `prefix_embs`，且 `_slice` 还需要 `image_token_mask` / `state_mask` 进 `self._cache`。
⚠ 换了缓存键名后，`orchestrator.check` 把 `key_builder.cached_data` 原样喂给 gate（`orchestrator.py:505-508`）。本线只用 `always_search`（不读 cached_data），但**加载期守卫必须把 gate 限定在 always_search**（M4），否则别的 gate 会读到不认识的键。

**对 `src/openpi/cache/components/key_builder.py` 的改动**：把 `_CP1BaseKeyBuilder.build()` 里内联的 `_slice_cp1_fields(...)` 调用抽成可覆写的 `self._slice()`。

⚠⚠ **该重构有一处结构性不对称，必须在实现时处理并写进注释**：`exp/cache_latency_bench/opt/` 下有两个类 subclass 了 `CP1SpatialPool16KeyBuilder` 却**完全覆写了 `build()`**、直接调模块级 `_slice_cp1_fields`（`r4_pool_keybuilder.py:76`、`r4_layout_check.py:56`），**不会**走 `self._slice()`；另两个只覆写 `_reduce_vision` 的（`matmul_pool_keybuilder.py:75`、`r4_build_keybuilder.py:38`）**会**。这两个类存在的全部意义就是与 src builder 逐位一致，且其 `build()` **零测试覆盖**。⇒ 要么把 `_slice()` 一并应用到它们，要么在两处加注释说明为何不适用。**不许默默留下两种切片语义。**

⚠ **现有测试不足以充当该重构的非回归证据**：`tests/cache/components/test_key_builder_cp1_experiment.py` 只断 shape / device / dtype / enabled-field 过滤 / CP2 raise，唯一穿过 `build()` 的数值断言是 `robot_state` 透传（`:134-143`）；池化函数的数值断言是**独立函数**测的，不经 `build()`。⇒ **偏移切错时 shape 不变、测试全过**。M2 必须新增一条穿过 `build()` 的数值金标准。

**`config.py` 改动**：`_valid_key_builder_types`（`:1457-1470`）加 4 项；`_build_key_builder`（`:2716`）加 4 个惰性 import 分支。类型名见 **D9**。

### M3 —— `src/openpi/cache/groot/interceptor.py` + adapter 改动

```python
class GrootCacheInterceptor:
    def __init__(self, policy, runner, *, orchestrator=None, timer=None): ...
    def get_action(self, observations: dict) -> dict: ...
    def on_task_begin(self) / on_task_end(self)
    def on_episode_start(self, experiment="", task="", episode_id=-1,
                         episode_name="", extra_metadata=None)
    def on_episode_end(self, success: bool)
```

`get_action` 流水（逐条对应 `policy.py:146-197`）：

```
1. obs_copy = observations.copy(); is_batch = _check_state_is_batched(obs_copy)
   若非 batch 则 unsqueeze; 非 ndarray → np.array 循环          policy.py:169-178
2. normalized_input = policy.apply_transforms(obs_copy)        policy.py:180
3. with runner.session():                                      # 只包前向（D10）
     stage1 = runner.run_stage1(normalized_input)              probe stage1_vision（runner 计）
   # ── 已出 inference_mode / autocast ──
4. try:
     cp1 = orchestrator.check(CP1, stage1=stage1)              probe cp1_sum（interceptor 计）
     #   query_keys 在此产生 ⇒ 普通张量、fp32 池化（§2.7c 实测）
     if cp1.hit_type is FULL_HIT:
         chunk = cp1.payload.action_chunk                      # [16,32]，已是普通 CPU fp32
     else:                                                     # MISS
         with runner.session():                                # 只包前向
             chunk = runner.run_stage2(stage1).action_pred     probe stage2_llm / stage2_action
         # ── 已出 inference_mode ──
     # 无 WARM_START 分支 —— 加载期已挡死（M4）。运行期 raise 是死代码，因为
     # orchestrator.py:672-695 会在 intermediates 为空时静默降级成 MISS。
5.   # 统一到 pi0.5 的 buffer 契约：[AH, AD]、CPU、fp32、contiguous、且非 inference tensor
     action_cpu = (chunk[0] if chunk.dim() == 3 else chunk) \
                    .detach().cpu().float().contiguous()
     if action_cpu.is_inference():        # FULL_HIT 路径已是普通张量，此处通常不触发
         action_cpu = action_cpu.clone()  # 出 context 后 clone 才真正逃逸（§2.7c）
     orchestrator.broadcast_action(action_cpu)
     if cp1.query_keys is not None:                            # 对齐 interceptor.py:816/:1021
         orchestrator.buffer_for_write(cp1.query_keys, action_cpu)
   finally:
     orchestrator.clear()          # 整个 cycle 套 try/finally（比 pi0.5 裸调更严）
6. unnormalized = policy.unapply_transforms({"action": action_cpu[None, ...]})
7. 若非 batch 则 squeeze；附 __hit_meta__
```

⚠ **不变量（T-3 逐条断言）**：交给 `broadcast_action` / `buffer_for_write` / `StepRecord` 的**每一个**持久张量必须同时满足 `device.type == "cpu"`、`dtype is torch.float32`、`is_contiguous()`、**`is_inference() is False`**。最后一条最易漏：在 session 内做转换，产物仍是 inference tensor（§2.7c 实测），跨 step 后被 storage/normalizer 就地改写即 `RuntimeError`。

⚠ **`action_cpu` 的归一化不可省**（直传 `stage2.action_pred` 会在三处坏）：
- `broadcast_action` 的 `chunk_cpu[0] if dim>=2`（`orchestrator.py:432-434`）拿到 `[16,32]` 而非每步 `[32]`，`_action_history` 语义与 pi0.5 不同；
- `buffer_for_write`（`:734-739`）会把 **GPU bf16** 张量攒满整个 episode —— 违反 `storage_types.py:18-27` 的「CPU / contiguous / float32」契约，且是共享 4090 上的显存泄漏；
- FULL_HIT 拿到的是 `[16,32]`、MISS 是 `[1,16,32]`，同一个 `unapply_transforms` 吃两者会错维（pi0.5 是显式补维的，`interceptor.py:850-853`）。

`__hit_meta__` 复用 `_build_hit_meta` 的字段集合（`hit_type` / `start_t` / `winner_id` / `cp1_score` / `searched`），**逐字相同**。`start_t` 恒 `None`。
⚠ `searched` 必须真实反映 gate：缺失会让 `exp/gate_research/analyze_n1_live.py:141-158` 抛 ValueError（"refusing to guess skip"），而恒 True 会让 skip 步被静默计成真 verdict。本线 `always_search` ⇒ 恒 True 语义正确，但**必须由加载期守卫保证 gate 只能是 `always_search`**。

⚠ **cache 存的是归一化空间的 `action_pred`**，与 pi0.5 存 `stage3.action_chunk`（`_output_transform` 之前）对齐。存反了会在跨 episode 复用时静默错一层归一化。

**`groot_policy_adapter.py` 三处加法式改动**（D7）：保留字段（`__`-前缀）先剥离再校验；剥离字段并入返回 envelope；四个生命周期钩子透传。

### M4 —— `exp/robocasa365/serve_groot_n15.py`

1. 新增 `--cache-config <yaml>`：`load_cache_config` → `validate_cache_config` → **加载期守卫** → `build_cache_components` → `CacheOrchestrator` → `GrootCacheInterceptor` → 注入 adapter。不传该参数时**走原路径，一行不变**。
2. **加载期守卫**（一次性拒绝，不留运行期惊喜）：`checkpoints` 集合恰为 `{cp1}`；`judge.type ∈ {threshold, always_hit}` 且 `warm_tiers` 为空；`gate.type == always_search`；`write_policy.type == "never"`（D8）；artifact 身份**精确绑定**（见下方第 3 条，R10/R15）。

   ⚠⚠ **必须是加载期，不能靠运行期 raise** —— 现有 config 校验挡不住，且错配是**静默**的：
   - `_routing_errors` 那条 cp1-only 正向 allowlist（`config.py:2372-2450`）只在 `config.routing is not None` 时才跑（`:2348-2350`），GR00T 不接 sidecar ⇒ 够不着；
   - 通用 validator **明确允许** cp1 上的 `warm_tiers`（`:2302-2305` 只禁非 cp1）与 `always_warm_start`（`:2257-2285`、`_JUDGE_TYPES` `:579`）；
   - `CacheConfig.checkpoints` 的**默认值同时含 cp1 与 cp3**（`:550-553`）—— yaml 省略 `checkpoints` 就会建出 CP3 并注册 search session，而 GR00T 拦截器永不 `check(CP3)`；
   - **最致命**：即便 judge 真返回 WARM_START，`orchestrator.py:672-695` 会因 `payload.intermediates` 为空（GR00T 离线库不写 `noise_action_*`）而**静默降级成 MISS**，只打一条 `logger.warning` ⇒ 错配 yaml 的表现是「hit rate 莫名偏低」，不是报错。

3. **artifact 身份守卫（必须精确相等，前缀不够）**
   ⚠ **前缀匹配挡不住同族错配**：`cp1_groot_mean_pool` 与 `cp1_groot_max_pool` 的 `vector_dims` **逐字完全相同**（vision 2048×3 / prompt 2048 / state 20），前缀守卫会放行「用 mean-pool 的 yaml 静默加载 max-pool 的库」。
   ⚠ 且**当前根本拿不到这个字段**：`in_memory_backend.py:230-237` 的 `load_artifact` 只读 `data["vector_dims"]`，`key_builder_type` / `checkpoint_id` **读都没读**，backend 也不留存。
   ⇒ **两步**：
   (a) `src/openpi/cache/backends/in_memory_backend.py` 的 `load_artifact` 加法式记下 `self.artifact_meta = {"key_builder_type": data.get("key_builder_type"), "checkpoint_id": data.get("checkpoint_id")}`（缺字段则为 `None`；**不新增校验、不改任何既有行为** ⇒ pi0.5 逐位不变）；
   (b) `src/openpi/cache/cache_storage.py` 加只读 facade `artifact_meta` —— **照 `library_stats` 的既有先例**（`:155-166`，其 docstring 明写此类属性的存在就是为了避免 "private reach-through into `self._backend`"）：

   ```python
   @property
   def artifact_meta(self) -> dict | None:
       """Artifact identity of the underlying backend, or None if it exposes none."""
       return getattr(self._backend, "artifact_meta", None)
   ```

   (c) GR00T server 在 `build_cache_components` **之后**经该 facade 读取，断言 `key_builder_type == config.key_builder.type`（**精确相等**）且 `checkpoint_id == "CP1"`。

   ⚠ **两种 fail-closed 形态不同，不可混为一谈**：
   - **真 legacy artifact**（pkl 里没有这两个键）：经新版 `load_artifact` 后是 `{"key_builder_type": None, "checkpoint_id": None}` —— **字典存在、值为 None**；
   - **backend 根本不暴露该属性**（Qdrant，或 in_memory 但无 `preload_path`、从未 `load_artifact`）：facade 的 `getattr` 兜底 ⇒ **`artifact_meta is None`**。

   两者都 fail-closed 拒绝启动并提示用 M5 的建库脚本重建，但错误信息要能区分，T-9 分别覆盖。
   🟢 选这条路而不是「读 pkl 头」：5.9 GB 的 artifact 不能为了看两个字符串加载两遍；而 `BackendPool` 本来就会加载它一次，metadata 挂在 backend 上是单一真相源，顺带把同一个隐患对 pi0.5 也暴露出来（本计划**不**替 pi0.5 打开该校验）。
   ⚠ 全程**不得**触碰 `storage._backend` —— `CacheStorage` 的模块契约要求调用方不直穿 backend，这正是加 facade 而非在 server 里 `getattr(storage._backend, …)` 的原因。
3. **`DEFAULT_CHECKPOINT` 改为 tp 那一支**，并同步 docstring 的 tmux 启动配方（§2.10）。这是一处**独立于 cache 的现存缺陷**，本计划顺手修正是因为 M4 正在改这个文件且它会污染本计划的所有实测。
4. **握手期禁 cache**：`_handshake` 发生在任何 `on_task_begin` / `on_episode_start` 之前（`:220-222`），会照常走 CP1、推进 `_step_counter`、喂 gate 反馈、计入 timer，污染第一条 episode 的统计。⇒ 握手走 `orchestrator=None` 的旁路（或握手后 `on_task_begin` 重置）。
5. 新增 `exp/robocasa365/config/groot_cache_cp1.yaml`（冒烟用；`always_hit` judge + `weighted_rrf_knn`，见 G0-E）。

6. **采集接线**（没有这一条，M5 的 wrapper 无从激活，G-4 与 G0-D1/D2 不可达）
   新增 `--collect-hdf5 <dir>`：构建 `GrootCacheCollector(policy, runner, out_dir=…)` 并注入 adapter，**位置与 `GrootCacheInterceptor` 相同**（都是 `_ActionPolicy`）。
   - **互斥**：`--collect-hdf5` 与 `--cache-config` **不可同时给**，加载期拒绝。理由：建库必须是 teacher-only 的干净前向，带 cache 采到的 key 会掺进被复用的动作，库就不是「A 场景 teacher 的真实轨迹」了（这正是 D8 不许在线写库的同一条理由）。
   - **wrapper 顺序**：`GrootPolicyAdapter(policy=<Collector 或 Interceptor 或 裸 Gr00tPolicy>)` —— 三选一，永远只有一层。
   - **生命周期**：adapter 的四个钩子（D7）透传给 collector；`on_episode_start(task=…, episode_id=…)` 决定 HDF5 文件名与 `f.attrs["task"]`，`on_episode_end(success=…)` 写 `f.attrs["success"]` 并**原子落盘**（见 M5 的两个 attrs 陷阱）。
   - **测试**：一条从 server 构建入口（`--collect-hdf5` 解析）到 HDF5 落盘的集成测试，见 §5 T-7。

### M5 —— 采集与建库

- `exp/robocasa365/groot_cache_collector.py`：`_ActionPolicy` 包装器，逐步跑 stage1（+stage2），用 `slice_groot_cp1_fields(enabled=None)` 取未降维原始字段落盘。
  ⚠ **必须在 `runner.session()` 内跑**（D10）；⚠ **`robot_state` 从 `stage1.state` 取**（D5/§2.7b）。
  ⚠ **不得 import `openpi.collect.collection_policy`**（`:7 import jax`，孤岛 B 导不进）；🟢 但 **`openpi.collect.data_collector` 可导入**（只依赖 h5py/numpy），`EpisodeDataCollector`（`:140-148`）就是现有 HDF5 schema 的写入端 —— **复用它，不要自己重写 writer**。
  schema：`vision_0/1/2` `[256, 2048]` fp16、`prompt_emb` `[T, 2048]` fp16、`robot_state` `[20]` fp32、`clean_action` `[16, 32]` fp32。
  ⚠⚠ **还必须写两个文件级 attrs**，漏了会**静默丢数据**：
  - `f.attrs["success"]` —— 缺失即视为 `False`，默认 `--outcome-filter success` 下**整个 episode 被静默丢弃**，最终只打印 "Saved 0 entries"（`build_…:704, 708-711`）；
  - `f.attrs["task"]` —— 原样进 `CachePayload.task_key`（`:703, :770`），**不做归一化**。
- `exp/common/build_in_memory_cache_artifact.py`：把 `_get_vector_dims` 扩成 `(…, vision_slots: int = 2, robot_state_dim: int = 32)`，对查表结果做后处理（增删 `vision_2`、覆盖 `robot_state`），并加对应两个 CLI；**默认值 (2, 32) ⇒ pi0.5 逐位不变**。另允许把 artifact 的 `key_builder_type` 写成 GR00T 名（R10）。脚本其余部分零改动（§2.8）。
  ⚠ 本版**不写 `noise_action_*`** ⇒ `:753-766` 的 `if noise_indices:` 不进，`_NUM_STEPS = 10`（`:750`）这颗雷不爆。但**将来若给 GR00T 加 intermediates**，t 会按 `1−i/10` 打错标签，且 `types.py:43-45` 的 `CANONICAL_DENOISE_TIMESTEPS` 只认十分位，**GR00T 的 4 步 schedule 根本表达不了** —— 这是 WARM_START 在 GR00T 上不仅"不做"而且"当前表达不了"的结构性理由。

### M6 —— `exp/robocasa365/groot_rollout_client.py`

1. 每 episode 前后发 `__ctrl__: episode_start` / `episode_end`（含 `__success__`），字段与 LIBERO client 一致。
   ⚠ **必须带上任务名**：`SearchContext.task_key` 唯一入口是 `on_episode_start(task=…)`（`interceptor.py:377-381` → `orchestrator.py:269`）。不发 ⇒ 所有条目 `task_key` 为空，跨任务检索失去过滤维度。
   ⚠⚠ **`episode_end` 必须在 `finally` 里发**：episode 中途 `infer` 或 `env.step` 抛异常时，若不补发 `episode_end(success=False)`，orchestrator 的 search session 会一直开到 socket 断开（关 session 在 `orchestrator.on_episode_end` 的 `finally` 里，`orchestrator.py:780-782`），期间的 backend 状态对下一条 episode 是脏的。⇒ client 侧 `try / finally` 包住整条 episode，异常路径也发 `episode_end(success=False)` 再往上抛。
2. **新增 per-step 命中记录器**：把 `__hit_meta__` 落成 JSONL。参照 `examples/libero/episode_runner.py:71-88`（`_hit_row`）+ `examples/libero/main.py:327-328`。⚠ GR00T client **没有任何等价物**；不写这一层，G0-E 的判据无数据源、后续跑批的 per-step JSONL 也不存在。

不带 cache 跑时两者都是 no-op。

⚠ **生命周期钩子的后果必须写清**（不是走过场）：`on_task_end` 必须转发 `orchestrator.on_task_end()`（`interceptor.py:403-405`），否则 `_close_current_search_sessions`（`orchestrator.py:363-373`）不跑、backend search session 永远开着；`on_episode_end` 即便 `write_policy=never` 也必须转发，因为关 session 在它的 `finally` 里（`orchestrator.py:780-782`）。server 是**关键字调用**（`websocket_policy_server.py:636-642, :648`），adapter 透传也必须用关键字。

### M7 —— 文档（L3 必需）

- `docs/architecture/cache_system.md`：新增 GR00T 两阶段切分一节（tap 点、掩码切 key、与 pi0.5 三阶段的对应与不对应）。
- `docs/cache/tutorial.md`：登记 4 个新 `key_builder.type`（与既有 `projection` / `dynamic_depth_knn` / `mlp_router` 的登记惯例一致）。
- `docs/cache/migration.md`：把 GR00T 列为第二个已落地案例（该文档是本计划的规定来源，反哺一条真实案例）。
- `docs/README.md` / `logs/README.md` 索引同步（WA §4 红线，同一 commit）。

### M8 —— 标定机制核验（只验机制，不产正式值）

⚠ **`tau` 的语义极易误读。** 实查 yaml（`exp/verdict_factor_judge/config/max_pool/phase1/*.yaml:82-86`）：

```yaml
field_similarity:
  robot_state:
    type: l2
    to_similarity: {type: exp, tau: 0.334717}
```

`tau` 属于 **`robot_state` 的 L2→相似度转换**，**不是** prefix_embs 的尺度参数。把它读成「tap 点量纲变了所以 tau 要跟着变」是错的 —— vision/prompt 走 **cosine**，本身尺度不变，换 tap 点不影响它们。

**标定面**（三件事，性质不同）：

| # | 对象 | 是否受换模型影响 | 归属 |
|---|---|---|---|
| 1 | `robot_state` 的 `tau` | **是**：pi0.5 是 32 维 state、GR00T 是 20 维且另一套归一化，L2 分布不同 | 复用 `exp/common/calibrate_robot_state_tau.py`（吃 HDF5，模型无关）⇒ **M8 只核验它能吃 GR00T 的 HDF5 并产出非退化 tau** |
| 2 | vision / prompt 的 cosine | **否**（尺度不变） | 无需标定 |
| 3 | judge 的 `threshold`（作用在**融合后**的分数上） | **是**：融合分布随字段分布与权重变化 | **属跑批 plan**，本计划不产数值 |

⚠ **fusion 方式决定 tau 有没有用**：`weighted_rrf_knn` 按**秩**融合（`config.py:3320`），而 `tau` 是**单调**变换 ⇒ **不改变 `robot_state` 字段内的排名**，对 RRF 结果无影响；只有 `weighted_score_sum_knn`（`:3333`）按分数融合时 tau 才真正参与。⇒ 冒烟 yaml 选 **`weighted_rrf_knn` + `always_hit`**，**完全不依赖任何标定值**；正式跑批若改用 `weighted_score_sum_knn`，tau 与 threshold 必须先标定——这条约束写在跑批 plan 的入口。

## 5. 测试策略

| # | 文件 | 类型 | 钉什么 |
|---|---|---|---|
| T-1 | `tests/cache/groot/test_groot_staged.py` | 非 manual，stub | 两段调用序与官方一致；`eagle_input` 键集合断言；B>1 raise；`model.training=True` raise；`select_layer == len(layers)` 断言；**未开 autocast 时 raise**；指纹守卫会炸且报 `__module__` |
| T-2 | `tests/cache/groot/test_groot_key_builder.py` | 非 manual | **偏移浮动**：同图像 + 不同长度 prompt ⇒ `vision_*` 逐位相同（直接钉 D4 的失败模式）；三段/256 断言；`robot_state` 取 mask 有效位；mask 变化 raise；**一条穿过 `build()` 的数值金标准**（补 §4-M2 指出的覆盖缺口） |
| T-3 | `tests/cache/groot/test_groot_interceptor.py` | 非 manual，stub orchestrator | FULL_HIT 时 `run_stage2` **零次调用**；MISS 恰一次；`broadcast_action`/`buffer_for_write`/`clear` 次序与 pi0.5 一致且 `clear` 在 finally；`__hit_meta__` 字段集合**精确相等**（自建守卫，参照 `tests/cache/test_router_orchestrator_interceptor.py:267` 的 `_LEGACY_META_KEYS` 写法）；adapter 放行 `__`-前缀字段且不放行未知 action 键；**⚠ 交给 `broadcast_action` / `buffer_for_write` 的每个张量断言 `device=="cpu"` ∧ `dtype is float32` ∧ `is_contiguous()` ∧ `is_inference() is False`**（含 `cp1.query_keys` 的每个字段）；**probe 计数**：FULL_HIT 时 `stage2_llm`/`stage2_action` 各 0 次、MISS 时各 1 次；**正控制** `stage1_vision` / `cp1_sum` / `total_inference` 每次 infer 各恰 1 次。⚠ 该用例必须用 `monitor.set_monitor_level(BASIC)` 显式设置并在 teardown 还原（level 进程内缓存、默认 OFF，靠 ambient env 会让"0 次"空洞通过） |
| T-4 | `tests/cache/groot/test_online_offline_key_parity.py` | 非 manual | **D1（结构）**：喂 **fp32** 合成 `input_embeds`，在线掩码切 vs 写 HDF5 → `_build_fake_stage1` → 偏移切，**逐位相同** —— 这一条检验的是「掩码逻辑 ≡ 偏移逻辑」，即 D4 要防的失败模式；**D2（端到端）**：按 §8 的 (a) 误差门（含零范数退化规则）+ (b) 按字段自身度量的排名保持门；**退化字段（如同一 episode 内恒定的 `prompt_emb`）必须被自动识别并豁免 (b)，且在报告里点名**；`vision_0` / `robot_state` 若被判退化则该门不通过；**dtype 同源**（都从 `stage1.state` 取） |
| T-5 | `tests/cache/components/test_key_builder_cp1_experiment.py`（既有）+ `tests/cache/components/test_key_builder.py`（既有） | 非 manual | `_slice()` 重构的非回归 |
| T-6 | `tests/cache/groot/test_import_isolation.py` | 非 manual | AST 扫**新子包与 `exp/robocasa365/groot_cache_collector.py`**，断言不出现 `jax` / `openpi.models` / `openpi.policies` / `openpi.cache.interceptor` / **`openpi.collect.collection_policy`** |
| T-7 | `tests/robocasa365/test_groot_cache_collector.py` | 非 manual | HDF5 schema、dtype、step 命名、原子写；**`f.attrs["success"]` / `f.attrs["task"]` 必写**（缺 success ⇒ 建库静默丢整个 episode）；**从 server 构建入口起的集成测试**：解析 `--collect-hdf5` → 装出 collector → 走两步 `get_action` + `on_episode_end(success=True)` → 磁盘上出现可被 `build_in_memory_cache_artifact.py` 读的 HDF5；`--collect-hdf5` 与 `--cache-config` 同给时拒绝 |
| **T-9** | `tests/cache/groot/test_groot_load_guard.py` | 非 manual | **加载期守卫拒绝矩阵**，逐条各一个用例：`cp3` enabled / judge 配 `warm_tiers` / judge 为 `always_warm_start` / gate 非 `always_search` / `write_policy != never` / artifact `key_builder_type` 与 yaml **不精确相等**（含同维的 mean↔max 错配）/ artifact `checkpoint_id != "CP1"` / **真 legacy artifact**（pkl 缺这两个键 ⇒ `artifact_meta == {"key_builder_type": None, "checkpoint_id": None}`，字典在、值为 None）/ **backend 不暴露该属性**（无 `preload_path` 的 in_memory ⇒ facade `getattr` 兜底 ⇒ `artifact_meta is None`）—— 每条都必须在**加载期**抛，且这两种 legacy 形态的错误信息可区分；另一条正例：合法 yaml 顺利构建。⚠ 本文件还须覆盖 `CacheStorage.artifact_meta` facade 本身（返回 backend 的值 / backend 无该属性时返回 `None` / **不触碰 `_backend`**） |
| **T-10** | `tests/robocasa365/test_groot_client_ctrl.py` | 非 manual | client 发出的 `__ctrl__` 帧字段与**关键字**形态；`episode_start` 带 `task`；**episode 中途抛异常时仍在 `finally` 发 `episode_end(success=False)` 再上抛**；`__hit_meta__` → JSONL 的行 schema 与容错（缺字段不炸） |
| T-8 | `tests/robocasa365/test_groot_cache_manual.py` | **manual，孤岛 B** | G0-C 逐位等价（**在测试进程内直接 load model，不经 server**）；同 seed 下对同一 `stage1` 连调两次 `run_stage2` 结果相同（钉 `vlln` 就地写回）；真实 `input_embeds` 图像段位置与 `vit_embeds` 一致；**反向对照**见下 |

**三条反向对照**（缺了，等式成立什么都证明不了）：
1. **不设种子**时两次 `get_action` 必须**不同**；
2. **故意不开 autocast** 时 max\|Δ\| 必须 **!= 0**（否则说明 D10 的断言没生效）；
3. 种子重置必须**紧邻每条路径调用之前**，且先跑一次完整 `get_action` **丢弃**（`cudnn.benchmark` 首调可能选不同 kernel）。

🟢 已核实的配套事实：`sample_time`/`beta_dist.sample` 只在训练分支；eval 下 `nn.Dropout` 不抽 RNG；transforms 在 eval 下不走 `random.random()`（`policy.py:97` 已 `.eval()`）⇒ **两条路径在 `torch.randn` 之前消耗的 RNG 完全相同**，只要重置点对了就够。

**⚠ T-8 必须在孤岛 B 跑，且 `PYTHONPATH` 必须含 `/home/weiland/gr00t_n15`** —— 漏了会让 `importorskip` 静默走跳过分支，看起来通过其实没跑（handoff §9-11）。孤岛 B 实测 `pytest 9.1.1` 可用、`--collect-only --run-manual tests/robocasa365/` 收集 82 项正常。

**§6 Verify 的 blast radius**：

```bash
uv run pytest tests/cache tests/exp tests/robocasa365
```
裸命令；**不加 `-m`、不 repo-wide、不碰 `tests/review_tests`**。
⚠ `tests/exp` 是本轮补上的：M5 要改的 `build_in_memory_cache_artifact.py` 有 6 个测试文件在那儿依赖它。

---

## 6. 风险登记

| # | 风险 | 后果 | 缓解 |
|---|---|---|---|
| R1 | 复刻的 235-259 与上游漂移 | 静默算错 | D2 指纹守卫（钉**执行副本**）；T-8 逐位等价 |
| R2 | 掩码切写成偏移切 | key 全是噪声但**不报错**，且 shape 不变、现有测试全过 | D4 三条运行时断言 + T-2 变长 prompt 用例 + T-2 数值金标准 + G0-D1 |
| R3 | **autocast 未开** ⇒ `vlln` 走 bf16 而非 fp32，`max\|Δ\|=0.0137` | G0-D2 必挂且原因不可见 | **D10**：context 收进 runner + 两个 stage 入口断言 + T-8 反向对照 |
| R4 | `robot_state` 采集侧 fp32 / 在线侧 bf16 | G0-D1/D2 必挂 | D5：两侧同源取 `stage1.state`，落盘才 `.float()`；T-4 钉 |
| R5 | `action_pred` 存成 unapply 之后的 | 跨 episode 复用时错一层归一化 | M3 明写；T-3 断言存归一化空间 |
| R6 | inference tensor 跨 step 存活被就地改写 | `RuntimeError: Inplace update to inference tensor` | ⚠ **原缓解措施（session 内 `.float().cpu()`）已实证无效**（§2.7c）。现行：**session 只包两段前向**，CP1 检查与张量归一化都在 session 外（D10）；T-3 逐个断言 `is_inference() is False` |
| R15 | **同族 artifact 静默错配**（`cp1_groot_mean_pool` 与 `_max_pool` 的 `vector_dims` 逐字相同） | 用错库跑出一整轮，结果不可信且无告警 | M4 守卫：`artifact_meta.key_builder_type` 与 `config.key_builder.type` **精确相等** + `checkpoint_id == "CP1"`；`None` 则 fail-closed；T-9 钉 |
| R16 | episode 中途异常未发 `episode_end` | search session 开到 socket 断开，脏状态污染下一条 episode | M6 的 `try/finally`；T-10 钉 |
| R7 | `BatchFeature` 复用 ⇒ `vlln` 二次施加 | 静默错值 | M1 步骤 5 每次新建；T-8 连调两次断言 |
| R8 | 两个 `select_layer` 混淆 / 换 ckpt 后 `select_layer != len(layers)` | 拿到视觉塔层或未过 final norm 的张量 | §2.6-2/3；T-1 双断言 |
| R9 | `torch.randn` 不可复现 | 等价性测试无意义 | T-8 三条反向对照 |
| R10 | **pi0.5 库被 GR00T server 静默加载**（`load_artifact` 只校验 `vector_dims`，两者 vision 都是 32768） | 语义完全不同却不报错 | 与 R15 同一道守卫：**精确相等**而非前缀匹配（前缀挡不住 R15 那种同族错配） |
| R11 | 共用 4090 显存波动 | 起服务撞 OOM 会打断 owner 其它 session 的跑批 | 连续 3 次、间隔 5 min、均 ≥20 GB 才动手；**端口 8000 与两个 `sidecar_server.py` 绝不可碰（其它 session 在用）** |
| R13 | **`vision_2` 没进检索键**（`_VECTOR_DIMS` pool 系缺该条目 ⇒ 被 `in_memory_backend.py:506-507` 静默丢弃） | eye_in_hand 相机形同不存在，**跑得出数、数没意义** | M5 的 `vision_slots` 参数；yaml 与 artifact 的 `vector_dims` 必须含 `vision_2` 且 `keys.vision_2.enabled=true`；T-4 断言三个 vision 字段都进了 key |
| R14 | pool 系 builder **没有 `expected_dim` 交叉校验**（只有 `cp1_temporal_prune` `config.py:2091-2108` 与 `cp1_llm_layer_extract` `:2143-2191` 有） | yaml 的 vision 维写错**不会在 load 时被抓**，只在 insert/search 才炸 | 加载期守卫顺带核 `vector_dims` 与 builder 类型的期望值（§4.1 表） |
| R12 | GR00T stage1 占比与 pi0.5 不同，且 `stage2_llm` 含 34% 的 lm_head 浪费 | HIT 收益口径被抬高 | §7 实测标注占比；上一轮已标定 vision tower + mlp1 = 23.73 ms、HIT 跳过占比 ≈69%（pi0.5 4090 实测 83%）。差异**如实报告**，不作为设计变更理由 |

---

## 7. 成本（实测基数）

**实现**：M1–M3 约 550 行 + 测试约 700 行；M4–M6 约 300 行；M7 文档；M8 约 50 行。

**机时**：

| 项 | 量 |
|---|---|
| T-8 manual（加载 7.2 GB ckpt） | < 10 min |
| 冒烟采集（G0-E 前置，≤ 3 episode ≈ 240 步） | ≈ 5 min（GR00T-tp 实测 46.5 s/ep） |
| 冒烟建库 | < 2 min |
| G0-E 闭环 1 episode | ≈ 2 min |
| **合计** | **< 1 h** |

**显存 / RAM / 磁盘**（2026-08-17 实测）：

| 项 | 量 | 依据 |
|---|---|---|
| GPU 当前空闲 | 34 560 / 49 140 MiB | `nvidia-smi` |
| 占用方（**都不能碰，属其它 session**） | 8000 `serve_policy.py` 7 764 MiB；两个 `sidecar_server.py` 3 392 + 2 772 MiB | `--query-compute-apps` + `ps` |
| GR00T server 常驻 | ≈ 7.6–8 GB | ckpt 7.2 GB bf16 |
| sim client（G0-E 需要） | ≈ 13 GB EGL/CUDA context | 前轮实测 |
| **G0-E 峰值** | **≈ 21 GB**（当前余量 33.7 GB 够） | R11 的判据仍须真执行 |
| 冒烟库 RAM | 240 步 × 403 KB ≈ **97 MB**（加载峰值 ≈2×） | 每条 = 3×32768+2048+20 fp32 |
| 冒烟 HDF5 | 240 步 × 3.15 MiB ≈ **760 MB** | schema `data_collector.py:140-148` |
| 主机余量 | RAM 218 GB / 磁盘 599 GB | `free -g` / `df -h` |

⚠ 供跑批 plan 参考（**不在本计划内**）：正式 (1,1) 建库 180 ep × 81 步/ep ≈ 14 650 条 ⇒ 库 ≈ 5.9 GB、HDF5 ≈ 46 GB。步数基数来自实测（180 ep 平均 **407 步**，replan=5 ⇒ 81.4 次推理/ep）。
⚠ handoff §5.5 的 69.1 h 预算是「5 场景/侧」旧设计的账，2×2 下不适用且要加 cache-on/off 两臂 —— **在跑批的 plan 里重算**。

---

## 8. 交付顺序与出场门

```
M1 ─→ [push] ─→ G0-C ─→ M2 ─→ M5 ─→ [push] ─→ G0-D1 ─→ G0-D2 ─→ M3 ─→ M4 ─→ M6 ─→ [push] ─→ G0-E ─→ M7 ─→ M8

  [push] = commit + push 到 origin/Ziyang，远端 `git pull --ff-only` 取回（P5 已裁）；
           三道门全在远端跑，未 push 的改动远端看不见。
       ↑                                                        ↑
    G0-B'（M1 落地即可跑，此后每次 §6 Verify 复跑）           文档与索引同步
```

| 门 | 判据 | 不过怎么办 |
|---|---|---|
| **G0-B' 导入隔离** | T-6 通过：新子包与采集器的导入图不含 jax / `openpi.models` / `openpi.policies` / `openpi.cache.interceptor` / `openpi.collect.collection_policy`；且孤岛 B 实机 `import` 成功 | 停；落点或依赖选错 |
| **G0-C 两阶段等价** | 固定种子下 `run_stage2(run_stage1(x)).action_pred` 与 `model.get_action(x)["action_pred"]` **max\|Δ\| == 0**；`backbone_features` 亦为 0。**三条反向对照全部成立**（§5） | 停；切分点或 context 有错，不得往下 |
| **G0-D1 结构（逐位）** | 喂 **fp32** 合成 `input_embeds`：在线掩码切 与 离线偏移切各字段 **max\|Δ\| == 0**；且 `vision_0/1/2` 三个字段**都在** key 里 | 停；掩码逻辑或 HDF5 字段写入错了 |
| **G0-D2 端到端（容差）** | 见下方 (a) 误差门 + (b) 排名保持门，**同时**成立 | 停；量化损失已大到会改变检索排序 |
| **G0-E 闭环冒烟** | **启动条件**：`OPENPI_MONITOR_LEVEL=BASIC` ∧ yaml `timer.enabled: true`。**正控制**（同一次 infer 内）：`stage1_vision`、`cp1_sum`、`total_inference` **各恰 1 条记录**；**负判据**：`stage2_llm` / `stage2_action` **各 0 条**。另加：动作确实来自库、无异常、`__hit_meta__` 经 adapter 上线并落进 client JSONL | 定位后重跑 |

⚠ **G0-E 用 `always_hit` 而非 "hit rate > 0"**：judge 的 `threshold` 是在 pi0.5 的**融合分数分布**上定的，换 teacher 后该分布未知，hit rate 可能稳定停在 0% 或 100% ⇒ "hit rate > 0" 既可能假阴也可能假阳。`always_hit` + `weighted_rrf_knn` 把"接线通不通"与"阈值准不准"**解耦**，且**不依赖任何标定值**（RRF 按秩融合，对 `robot_state` 的单调 `tau` 不敏感——见 M8）。这正是 `migration.md:792` 的建议。阈值与 tau 由 M8 核验机制、交给跑批 plan 定数值。
⚠⚠ **G0-E 必须显式开 monitor level，否则"stage2 零采样"是空洞的**。`SystemTimer.__init__`（`timing.py:404-427`）被 `OPENPI_MONITOR_LEVEL` 主开关覆盖，而 `monitor.py:93,100-108` 默认 **`OFF`** ⇒ 不设它时**所有** probe 都零记录，判据分不清"真跳过 stage2"与"计时器根本没开"。

- **`BASIC` 足够满足本门**：`timing.py:404-412` 明示 BASIC 下 timer **仍然记录**，只是 CUDA probe 改走 CPU `PerfCounterBackend`（省掉每个 `measure()` 的 GPU sync）。本门只数条数，不要 GPU 精确时长。
- **`SNAPSHOT` 只在需要 GPU 精确时长时开** —— 即 R12 要实测 `stage2_llm` 中 lm_head 占比的那一次测量。
- ⚠ level 是**进程内缓存**（`get_monitor_level` 只在首次调用读 env，`monitor.py:126-136`）⇒ **自动测试必须用公开的 `monitor.set_monitor_level()` 显式设置并在 teardown 还原**，不能靠改环境变量，更不能依赖测试进程的 ambient env。

**G0-D2 的精确判据**。取一条冒烟 episode 的全部 N 步。记 `k_on(t,f)` = 在线 key，`k_off(t,f)` = 同一步经 HDF5 fp16 → `_build_fake_stage1` → fp32 池化的离线 key。

**(a) 逐字段误差门 —— 所有启用字段都要过**

```
e(t,f) = ‖k_on(t,f) − k_off(t,f)‖₂ / ‖k_off(t,f)‖₂      若 ‖k_off(t,f)‖₂ > 0
       = ‖k_on(t,f) − k_off(t,f)‖₂                       若 ‖k_off(t,f)‖₂ == 0（退化为绝对误差）
判据： max over (t,f) 的 e ≤ 1e-3
```

**(b) 排名保持门 —— 按字段自己的度量，且只对非退化字段**

⚠ **不能用原始 L2 最近邻间距来定这条门**，两个理由：

1. **`prompt_emb` 在同一 episode 内逐位恒定**（D4 定义它为全部非图像 token，而任务文本整条 episode 不变）⇒ 各步 key 相同 ⇒ 原公式的 `d_NN = 0`，`r` 变成除零。
2. **backend 对 vision/prompt 用的是 cosine 不是 L2**（`FieldSimilarityConfig.type` 默认 `"cosine"`，`config.py:356-359`；`in_memory_backend.py:12` 模块 docstring 写死 "Field similarity: cosine (vision/prompt) and L2 (robot_state)"；dispatch 在 `:399-405`）⇒ 用原始 L2 间距推不出 cosine 排名不变。

```
sim_f(·,·) = 该字段 yaml 里配置的度量（vision_*/prompt_emb → cosine；robot_state → l2 转相似度）

退化字段定义：存在 t ≠ t' 使 k_off(t,f) 与 k_off(t',f) 逐位相同
              ⇒ 该字段的"最近邻"本就不唯一，排名门对它无意义

对每个**非退化**字段 f、每一步 t：
    以 k_on(t,f) 为 query，在库 {k_off(·,f)} 上按 sim_f 检索
    (b1) argmax_{t'} sim_f(k_on(t,f), k_off(t',f)) == t          ← 必须对所有 t 成立
    (b2) margin(t,f) = sim_f(k_on(t,f), k_off(t,f))
                     − max_{t'≠t} sim_f(k_on(t,f), k_off(t',f))  > 0
判据：(b1) 与 (b2) 对所有非退化字段的所有 t 成立；报告 min margin
必过字段：至少 vision_0 与 robot_state 必须是非退化且通过
          （这两个正是 cp1_* 前缀校验强制 enable 的字段，config.py:2050）
```

**报告要求**：跑完必须列出哪些字段被判为退化并因此豁免 (b)。若 `vision_0` 或 `robot_state` 出现在退化名单里，**该门判为不通过** —— 那意味着冒烟 episode 本身没有区分度，换 episode 重跑，不许豁免。

(b1) 才是真正有意义的那条：它直接检验"量化损失不改变实际检索出来的赢家"，而不是用一个替代度量去论证。`1e-3` 与 (b2) 的严格为正都是**预注册**的，实测不过就停下来查，不许事后放宽。

⚠⚠ **G0-C 与 G0-D1 是判据不是仪式**：只要不是逐位相等，就必须停下来查根因，**不许用容差糊过去**——本线全部结论都建立在"两臂用的是同一个 teacher"上。
⚠ **G0-D2 是唯一允许带容差的门**，且理由是可陈述的物理事实（bf16→fp16 存储往返），不是"差不多就行"。把 G0-D 整体定成"逐位相等"是达不到的：§2.9 证明那由构造决定，第一次跑必红而根因不是掩码逻辑。

---

## 9. 待裁 / 遗留

| # | 事项 | 建议 |
|---|---|---|
| ~~P1~~ | D6 的落点 | ✅ **已裁（2026-08-17）：`src/openpi/cache/groot/`**。§4.0 清单、§5 测试路径、§5 blast radius 均按此定稿，无待定分支 |
| **P2** | pi0.5 在 RoboCasa 上同样没有 cache（`serve_robocasa_pi05.py` 绕过 `serve_policy.py`）。⚠ 该文件**不在本仓**，实体在 `weilandserver:/home/weiland/step0b_artifacts/serve_robocasa_pi05.py` | **本计划不解**。跨场景实验若要 pi0.5 臂也带 cache，需另起 plan（工作量小于本计划：`InferenceInterceptor` 现成，只缺 RoboCasa 的 server 接线） |
| **P3** | 任务集口径 13/18 vs 9/18（handoff §4） | 与本计划**正交**，属跑批 plan |
| ~~P4~~ | 本线未提交文件 | ✅ 已解决：`dd139bd` 已 commit + push（analyze_admission_gate + 其测试 + 两份 log + 本计划 + README 索引），远端已 fast-forward 取到 |
| ~~P5~~ | **新代码如何送上远端** | ✅ **已裁（2026-08-17）：走 (a) git 路线。** 远端 `/home/weiland/openpi` 是 `git clone`，靠 `git pull --ff-only origin Ziyang` 同步（实测 reflog）⇒ 新代码必须先 **commit + push 到 `origin/Ziyang`**，远端才拿得到；三道门全在远端跑。⚠ 明确**不用** `tether push` 手工投放 —— 后续正式 commit 再 `git pull` 会因 "untracked working tree files would be overwritten" 中止。⚠ 本地工作树混着 3 个别的 session 的未提交文件，提交必须**逐文件点名**，不可 `git add -A`。⚠ 另注：孤岛 B 的 `PYTHONPATH` 指向 `/home/weiland/openpi/src`，与本仓库 `/home/weiland/projects/openpi` 是**两个 checkout** —— D6 把新子包放 `src/` 之后，远端 server 看不到本地改动，除非按本条同步 |

---
