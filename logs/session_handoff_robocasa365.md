# Session Handoff — RoboCasa365 跨场景 cache 实验

> ⚠ **本文件不是 `logs/session_handoff.md`**。那一份属于 X14 在线 RL Router session（工作树里有其未提交更新、`rlr*` tmux 仍在跑），**勿动、勿暂存**。

**Status**: `Paused` — 全部准入门完成且全 PASS，**远端与本地均无我方在跑的任务**。
**下一步**：起草 **GR00T 接入 cache 系统**的 plan 并过 G1（L2/L3）。⚠ **不是继续跑实验** —— cache 尚未接入本线，见 §5。
**日期**: 2026-08-17

---

## 0. 接手第一步

1. 本文件读完。
2. `logs/benchmark_and_teacher_selection.log.md` —— 战场选型 + §12 立项检查（14 项闭合）+ §4.6.3 场景分析 + §12-2 全部准入门数据。**数字以该文件为准，本文件是索引。**
3. `logs/groot_n15_robocasa_adapter.log.md`（约 1040 行）—— 适配层 L2 plan + 全部审查往来 + §10 执行记录。
4. 本 session Authority = **Execution**。适配层已走完 L2 全流程并 push；当前**不在 gate 里**，但下一步要重新进 G1。

---

## 1. 这条线在做什么

检验 owner 的**跨场景继承假说**：cache 的 key 从 VLA 内部表征抽取，那么场景 A 上建的 cache 库，在**同任务、不同厨房**的场景 B 上是否仍可用。

战场 = **RoboCasa365 / Atomic split**（18 任务）。teacher 两个：**pi0.5** 与 **GR00T N1.5 (target-posttrained)**。

⚠ **起因**：原先在 LIBERO 上"pi0.5 蒸馏的小模型（ACT）替掉 cache 后反而更好"。排查结论是 LIBERO 的 train-test gap 太窄（自有数据反驳 per-task 主因说：SmolVLA 单模型多任务 0.954 vs per-task ACT 0.966，仅差 1.2pp）。换 RoboCasa365 是为了拿一个**真正的泛化维度**重测。

---

## 2. ⛔ 最重要的一条概念纪律（同一错误犯过两次）

> **teacher 是固定底座，不是自变量。** 自变量只有一个：**cache 库建在哪个场景**。

teacher 只需满足两条：① **足够能干**（有成功 episode 才有东西可缓存 —— 这正是准入门存在的理由）；② **两臂严格同一个**。

**除此之外，teacher 的训练分布、checkpoint 族、绝对 SR 高低，都不影响「A 建的库能否在 B 用」这一测量。**

犯过的两次错：先说「teacher 训练过 target 场景会污染 held-out 前提」（错：库在 A 建、在 B 评，A≠B 由设计保证）；被纠正后又说「teacher 表征在 A/B 上被优化过会削弱正结论普适性」（错：假说**就是**「key 抽自内部表征故跨场景可用」，迁移因表征不变而成功**正是被验证的机制**）。根因是把 D6 教训（teacher 崩溃会淹没索引效应）过度外推成「teacher 不许见过评测场景」。**D6 要求 teacher 能干活，不是没见过。**

---

## 3. 已完成：7 次准入门，1260 ep，全部 P1 PASS

### 3.1 基础准入门（场景对 (1,1)/(7,7)，各 180 ep）

| | pi0.5 | GR00T **mt**（留档） | GR00T **tp**（选用） |
|---|---|---|---|
| SR_A / SR_B | 41.1% / 50.0% | 26.7% / 34.4% | **60.0% / 71.1%** |
| P1（SR_B Wilson 下界 vs 20%） | 39.9% PASS | 25.4% PASS | **61.0% PASS** |
| gap (A−B) | −8.9pp | −7.8pp | −11.1pp |
| U0 / U1 / U2 | 4 / 2 / 12 | 6 / 4 / 8 | **0 / 2 / 16** |
| 180 ep 墙钟 | 4.59 h | 2.90 h | **2.33 h** |

🟢 **三个 teacher 的 gap 全部同号**（held-out 的 scene B 略优）⇒「held-out 厨房不更难」跨架构可重复。

🟢🟢 **harness 正确性已由复现官方数字验证**：论文（arXiv 2603.04356v1）三张表设定不同 —— **Table 1**（multitask，Atomic pi0.5 39.6% / GR00T 43.0%）**评测在 pretraining 厨房**；**Table 2**（foundation model training = `target_posttraining`，Atomic **68.5%**）**评测在 target 厨房**。我方评测在 target 场景 ⇒ 与 Table 2 同口径，实测 `SR_B` CI **[61.0%, 79.5%] 覆盖 68.5%**。
⇒ **「GR00T 绝对 SR 偏低」存疑项彻底关闭**：mt 的 34.4% 低于 Table 1 的 43.0% 纯因**评测场景不同**（分布外退化非缺陷）；replan=5 vs 原生 16 等候选**全部排除，无须再查**。

### 3.2 2×2 场景准入门（四轮 720 ep，远端 chain 自主跑完）

⚠ 配对法下 JSON 的「A/B」只是位置标签，下表已绑回 (layout, style)。

| 场景 | 角色 | GR00T-tp | pi0.5 |
|---|---|---:|---:|
| (1,1) | 建库基线 | 60.0% | 41.1% |
| **(1,7)** | **S-only**（只换 style） | **60.0%** | **44.4%** |
| **(5,1)** | **L-only**（只换几何） | **58.9%** | **37.8%** |
| **(5,7)** | **Both** | **63.3%** | **45.6%** |
| (1,1) | 锚点重测 | 65.6% | 34.4% |
| (7,7) | 设计外极端 | 71.1% | 50.0% |

**12 格全部 P1 PASS**（最低下界 25.4%）。

🟢🟢 **锚点重测给出噪声底**（同场景两次独立测量，正式模式不钉采样噪声）：**GR00T-tp 5.6pp、pi0.5 6.7pp**。以此为标尺，2×2 相对基线的六个偏移（+0.0 / −1.1 / +3.3；+3.3 / −3.3 / +4.4 pp）**无一超出噪声底**。
⇒ **teacher 在 2×2 每一格都同样能干** ⇒ 准入门目的达成：cache 迁移若掉性能，**不能归因于 teacher 在目标场景更弱**。
⇒ 实用下限：**K=5（90 ep）下 teacher 侧 <6–7pp 的场景差异不必当真**。

⚠ 以上全是 **teacher 能力**结论，**不是 cache 结论**。

---

## 4. owner 已裁 / 待裁

### ✅ 已裁
- **K = 10**
- **U0 处置 = 全保留、分层报告** ⇒ 正式实验**采集全 18 任务**，主结论在交集上算
- **checkpoint = GR00T `target_posttraining/atomic_seen/checkpoint-60000`**（论文 Table 2 那一支）
- **场景设计 = 完整 2×2**：layout ∈ {1,5} × style ∈ {1,7}，建库恒为 (1,1)；S-only=(1,7)、L-only=(5,1)、Both=(5,7)；(7,7) 退为设计外极端参照
- **GR00T 接 cache = 只分两阶段**、**tap 在 `input_embeds`**、**不考虑 warmup**（详见 §5）

### 🔴 待裁：任务集口径

各场景可用任务数（该场景 SR>0）：GR00T-tp 建库 18/18、(1,7) 18、(5,1) 16、(5,7) 16；pi0.5 建库 13/18、(1,7) 13、(5,1) 13、(5,7) 12。⚠ **交集完全由 pi0.5 决定**。

| 口径 | 定义 | 结果 |
|---|---|---:|
| 旧（作废） | 两 teacher × (1,1)/(7,7) | 14/18 |
| **宽（建议）** | 两 teacher 在**建库场景 (1,1)** 能做 | **13/18** |
| 严 | 两 teacher × 建库 × **全部三评测场景** | **9/18** |

建议宽口径，三条理由：①「teacher 在评测场景做不到」**本身就是 cache 要改善的对象**，事前排除会剔掉最有信息量的样本；② 严口径依据的是 K=5 单场景 0/5 判定，而数据已证明它在噪声里晃（`TurnOnSinkFaucet` 在 (1,1) 基线 0/5、重测有成绩）；③ 建库场景可用是**硬前提**，评测场景不是。
⚠⚠ 无论选哪个，**「跨 teacher 取交集」是预注册之外的事后规则**，须按 P3 在论文显式记为事后决定。

---

## 5. 🔴 下一步：GR00T 接入 cache 系统（须走 plan → G1）

### 5.1 现有 cache 系统的分阶段语义（pi0.5）

```
stage1 = _stage1_token_prep     → prefix_embs（视觉+语言拼好的统一序列）
stage2 = _stage2_llm_backbone   → KV cache
stage3 = _stage3_action_expert  → flow matching
```
`src/openpi/cache/interceptor.py:20`：**`FULL_HIT: skip Stage 2 + 3, return cached action.`**

### 5.2 ✅ owner 裁定的 GR00T 切法（本轮确定）

**只分两阶段**，tap 点 = **`input_embeds`**，即视觉 token 散射进语言序列之后、进 Qwen3 第 0 层之前：

`/home/weiland/gr00t_n15/gr00t/model/backbone/eagle2_hg_model/modeling_eagle2_5_vl.py`
```python
235: input_embeds = self.language_model.get_input_embeddings()(input_ids)
237: vit_embeds   = self.extract_feature(pixel_values)          # [n_img, 256, 2048]
     ...  input_embeds[selected] = input_embeds[selected]*0.0 + vit_embeds.reshape(-1, C)
259: input_embeds = input_embeds.reshape(B, N, C)               # ← tap 点
261: outputs = self.language_model(inputs_embeds=input_embeds, ...)   # ← Qwen3 第 0 层
```
散射位置由 `input_ids == self.image_token_index` 决定。

| | 内容 | 出口 |
|---|---|---|
| **Stage 1** | `extract_feature` + 语言 embed + 散射合并 | **`input_embeds` [B, N, 2048]** |
| **Stage 2** | Qwen3 **12** 层 → `hidden_states[12]` → action head（flow matching） | action chunk |

后段依据 `/home/weiland/gr00t_n15/gr00t/model/backbone/eagle_backbone.py`：`__init__:59-60` 把 `language_model.model.layers` 截断到 `select_layer=12`；`forward_eagle:109-110` 取 `eagle_output.hidden_states[12]`。

**为什么是 `input_embeds` 而不是 `vit_embeds`**（这是本轮的实质修正 —— 之前验证的 tap 点是 `extract_feature()` 的输出）：stage1 的输出必须能**直接喂给 stage2**。若 stage1 只吐 `vit_embeds`，stage2 还得自己重做语言 embedding 与散射，切分不干净、计时也不准。取 `input_embeds` 才是一刀两断，且与 pi0.5 的 `prefix_embs` 语义严格对应。
🟢 数值上不冲突：`input_embeds` 图像位置的值就是 `vit_embeds` 散射进去的，按 `image_token_index` 切回再 pool，可复现已验证的 32768/相机结果。

**HIT 语义**：三阶段变两阶段 ⇒ **HIT 时跳过 stage2**（Qwen3 12 层 + action head 全省）。
**stage3**：建议**不留空壳、直接并入 stage2**；interceptor 里 `stage3` 只在 `_stage3_fn` / probe / device 配置三处出现，挂 no-op 比留假阶段更不易误导后来读代码的人。具体在 plan 里定。
**warmup**：**无需额外工作** —— `src/openpi/cache/config.py:182-183` 明写 `samples_source` **只支持 `"offline"`**，`"warmup"` 是保留未启用通道。

### 5.3 已做 vs 未做（分清"证明能做"与"做了"）

| ✅ 已做（2026-08-15 实机验证） | ❌ 未做 |
|---|---|
| `extract_feature()` 跑通，`vit_embeds [3,256,2048]` bfloat16 | **GR00T 接进 `InferenceInterceptor`** |
| tap→key：`view(3,16,16,2048)→permute→adaptive_avg_pool2d(4,4)→flatten` = **32768/相机，与 pi0.5 逐位一致（MATCH: True）** | **stage1/stage2 的实际拆分实现** |
| cache 内核可装进孤岛 B（47 模块 44 通过；缺 jax/websockets/qdrant，无版本冲突） | **server 侧 cache 路由** |
| 成本标定：vision tower+mlp1 = **23.73 ms**（tap 前每步必付），HIT 跳过占比 **≈69%**（pi0.5 4090 实测 83%） | 建库 / 评测管线 |

⇒ **证明了"能做"，一行集成代码都没写。**

### 5.4 四个必须在 plan 里解决的障碍（均为查证结果）

1. **`InferenceInterceptor` 硬绑 pi0.5**：`__init__` 要求 `policy._is_pytorch_model`，内部 `self._model = policy._model  # PI0Pytorch instance`。GR00T 是孤岛 B 的 `Gr00tPolicy`，**类型与 venv 均不同**。⇒ 抽象出去 vs 平行实现，影响 pi0.5 侧要不要跟改。
2. **GR00T 无 stage 拆分入口**：`forward_eagle` 是 `eagle_model(**eagle_input)` 一次调完，要在 `input_embeds` 处切开需包一层或改调用路径。
3. **三阶段假设散布多处**：`stage1_device/stage2_device/stage3_device`、三个 probe（`stage1_vision`/`stage2_llm`/`stage3_flow`）、coordinator 的 `submit_to_stage(2|3)`。
4. **server 侧无路由**：`scripts/serve_policy.py` 的 `EnvMode` 只有 `aloha/aloha_sim/droid/libero`，**无 robocasa**；`src/openpi/cache/` 与 `src/openpi/serving/` 内**零个 robocasa 引用**。且我方 RoboCasa pi0.5 server 是独立脚本 `serve_robocasa_pi05.py`，**绕过 `serve_policy.py`**（当初绕开正是因为 fork 的 robocasa config 在模块顶层 `import robocasa` 会污染主 venv）⇒ **pi0.5 在 RoboCasa 上同样没有 cache**，这个矛盾要一并解。

### 5.5 再往后
建库（(1,1)）→ 三个评测场景带 cache 采集 → §4.4 三层测量（key 层距离/余弦、检索层 `cp1_score`+hit rate+winner_id、系统层 SR）→ 论文台账。
⚠ **旧的 69.1 h 预算是「5 场景/侧」旧设计的账，2×2 下已不适用，须在 plan 里重算**（且多了 cache-on/off 两臂）。

---

## 6. 拓扑与路径（全部实机验证）

**远端主机 `weilandserver`**（与本机 `Weiland` 是**两台机器**：hostname 不同、本机无 `/home/weiland/openpi`、本机无 tmux ⇒ 本机重启不影响远端跑批）。单张 RTX 4090，49140 MiB，多 session 共用。

⚠ 端口 **8000** 是他人的 `serve_policy.py`（约 7.8 GB）—— **绝不可关**。pi0.5 用 **8010**，GR00T 用 **8020**。
⚠ **禁止宽模式 `pkill`**，只按自己的 tmux 名操作。⚠ `cssrv`/`cscol`/`rlr*` 是别的 session 的，**不可动**。

三个互斥 venv 孤岛：

| 孤岛 | 路径 | 关键点 |
|---|---|---|
| **A（sim）** | `/home/weiland/Isaac-GR00T/gr00t/eval/sim/robocasa365/robocasa365_uv/.venv` | py3.12 / numpy **2.2.5**；装包**必须 `--no-deps`**；cwd 必须是 `.../external_dependencies/robocasa365`；已有 pytest |
| **B（GR00T）** | venv `/home/weiland/gr00t_n15_venv/.venv`；**源码 `/home/weiland/gr00t_n15`** | py3.11 / numpy 1.26.4 / torch 2.5.1+cu124；**cache 内核将来跑在这里** |
| 主 venv | `/home/weiland/openpi/.venv` | openpi + pi0.5 |

⚠⚠ **`gr00t` 不在孤岛 B 的 venv 里** —— 来自 git worktree `/home/weiland/gr00t_n15`（`n1.5-release`，`4af2b62`），**必须进 `PYTHONPATH`**。漏了会让 `importorskip` 静默走跳过分支，**看起来通过实为没跑**。

**模型**（均出自官方 `robocasa/robocasa365_checkpoints`，author=`robocasa`，gated=False，commit `c484448a…`）：
- GR00T **选用**：`/home/weiland/ckpt_n15_robocasa_tp/gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000`
- GR00T 旧 mt（留档）：`/home/weiland/ckpt_n15_robocasa/gr00t_n1-5/multitask_learning/checkpoint-120000`
- pi0.5：`/home/weiland/ckpt_pi05_robocasa_pytorch`（源 `pi05_pretrain_human300/multitask_learning/75000`，JAX→PyTorch 转换）

⚠ 该 repo **README 为空**，各支语义靠目录名 + `trainer_state.json` + 论文推断。

**EGL**（该机无系统 EGL，孤岛 A 必需）：
```bash
export LD_LIBRARY_PATH=/home/weiland/nvidia-gl/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export __EGL_VENDOR_LIBRARY_DIRS=/home/weiland/nvidia-gl/root/usr/share/glvnd/egl_vendor.d
export MUJOCO_GL=egl
```

⚠ 远端仓库路径是 **`/home/weiland/openpi`**（不是本地 `/home/weiland/projects/openpi`）。本地改动需 `tether push` 传过去。

**启动命令**：GR00T server/client 见 `exp/robocasa365/serve_groot_n15.py` 与 `groot_rollout_client.py` 的 file-level docstring；pi0.5 server = `/home/weiland/step0b_artifacts/serve_robocasa_pi05.py`（主 venv，端口写死 8010，无参数），pi0.5 client = 同目录 `step0b_v2.py`（孤岛 A，**env 驱动**：`SCENE_A`/`SCENE_B`/`OUT_JSON`/`N_TRIALS`/`REPLAN_STEPS`/`PI_PORT`）。
⚠ **正式采集绝不可加 `--diagnostic-seed`**：它钉死 flow-matching 噪声，SR 不再无偏且事后无法从数据发现；client 已内置拒收（除非 `--allow-diagnostic-server`）。

---

## 7. 数据与工具

**原始数据**（全在 `exp/robocasa365/data/`，⚠ `exp/**/data/**` 被 gitignore，**只在本机不入库**；远端副本在 `/home/weiland/openpi/exp/robocasa365/data/` 与 `/home/weiland/step0b_artifacts/`）：
`pi05_gate_180ep.*`、`groot_gate_180ep.*`(mt)、`groot_tp_gate_180ep.*`、`groot_tp_scenes_{17_51,57_11}.json`、`pi05_scenes_{17_51,57_11}.json`、`pi05_analyze_step0b_ORIGINAL.py`

**分析入口** `exp/robocasa365/analyze_admission_gate.py`（L1，96 tests passed）：
```bash
cd /home/weiland/projects/openpi
python3 exp/robocasa365/analyze_admission_gate.py \
  --teacher pi05=exp/robocasa365/data/pi05_gate_180ep.json \
  --teacher groot_tp=exp/robocasa365/data/groot_tp_gate_180ep.json --self-check
```
`--self-check` 断言 pi0.5 的 9 个数字与 §12-2 记录吻合（每次运行均通过）。

⚠ **配对法产出的 `*_scenes_*.json` 里「A/B」只是位置标签**，不代表建库/评测语义。绑回关系见 §3.2 表。

---

## 8. 未提交（owner 未指示提交，按约定不擅自 `git add`）

| 文件 | 状态 |
|---|---|
| `exp/robocasa365/analyze_admission_gate.py` | 新增，L1，Verify 已过 |
| `tests/robocasa365/test_analyze_admission_gate.py` | 新增，24 tests |
| `logs/benchmark_and_teacher_selection.log.md` | 新增 |
| `logs/session_handoff_robocasa365.md` | 本文件 |
| `logs/README.md` | 加了索引行 |

⚠ 工作树里 `logs/session_handoff.md`、`logs/cache_size_ablation_plan.log.md`、`exp/rl_router/*`、`src/openpi/cache/components/mlp_router_judge.py`、`tests/cache/*`、`tests/exp/*`、`docs/iclr/*` 等**属于其他 session**。提交本线改动必须**逐文件点名**，**不可 `git add -A`**。

已 push 的本线 commit：`15dfa67`（适配层）、`35d38a6`（import 修正 + gate 结果）。

---

## 9. 陷阱（全部真机踩过）

1. ⚠⚠ **`sys.modules` 假件会为错误的 import 路径背书**：假件按被测代码请求的名字注册，代码写错模块名，假件就在错名字下被创建 ⇒ 测试与 bug 互相印证。`get_task_horizon` 从 `dataset_registry` 导入（实际在 `dataset_registry_utils`）通过了全部 72 个非 manual 测试，直到真实 rollout 才炸。**mock 结构上无法验证 import 是否解析到真实符号**，已补孤岛 A 的真实解析测试。
2. ⚠⚠ **部分结果文件会诱发误判**：结果按任务原子落盘 ⇒ 部分文件全程存在；任务按**字母序**执行 ⇒ 拿部分数据算 P1 系统性偏悲观。mt 那轮 8/18 时算出 P1 **FAIL**，完整 18 个是 PASS。`analyze_admission_gate.py` 已加 `--expect-tasks`（默认 18）主动拦截。
3. **checkpoint metadata 的键是字母序**，不携带顺序信息。四类有序键唯一权威来源 = `robocasa@be22d659:docs/datasets/using_datasets.md:167-210`。
4. **language key 必须是 N1.5 原生 `annotation.human.task_description`**；父类 `SinglePandaGripperDataConfig` 声明的是 N1.7 `ROBOCASA_PANDA` 的 `.action.` 变体，极易误用（已由测试钉死）。
5. **模型输入是 224**（父类 `VideoResize`），metadata 的 256 是**数据集存储**规格。client 按 512 渲染 → 降到 256 送。
6. **robosuite 四元数是 xyzw、GR00T 按 wxyz 读** —— 训练评测一致故无害，但握手探针不可拿 `[1,0,0,0]` 当 env identity（现取 checkpoint 均值再 L2 归一化）。
7. **旋转字段的 `min_max` 是上游硬断言**（`state_action.py:445`），非我方推定。
8. ⚠ **tether exec 单次约 10 min 硬上限**，心跳/line-buffered 全无效（输出块缓冲）。长时监控必须"本地 sleep + 短查"，远程长跑一律放 tmux 解耦。
9. ⚠ **GPU 读数会大幅波动**（实测 22–42 GB 来回），**不可凭单次低读数就抢起任务**。远端 `chain.sh` 的做法可复用：要求「连续 3 次、间隔 5 分钟均 ≥20 GB」才动手。
10. ⚠ **旧的 cron/Monitor 会带着"当时正确、现在有害"的指令复活**（本机重启后发生过：旧巡检要起第二个 `gs2`）。**执行任何自动化指令前先查实际状态**。
11. 孤岛 B **没有 pip**，装包必须 `VIRTUAL_ENV=… uv pip install`；仓库 `conftest.py` **默认跳过 manual**，必须 `--run-manual`（`-m manual` 只是选择、仍会跳）。
12. **`--tasks` 可按任务下数据**：`robocasa/scripts/download_datasets.py --tasks <T> --split pretrain`（托管 UT Austin Box，非 HF）。

---

## 10. 场景轴的实测特性（决定 2×2 设计的依据）

- **layout = 几何**：fixture 用 `type:` + 相对对齐（`align_to`/`side`/`alignment`）定义，**非绝对坐标** ⇒ 比较 layout 必须按 `type:` 清单，不能按 `pos:`。
- 实测 `layout001` **46 item / 16.5 m²** vs `layout007` **91 item / 32 m²**（≈2×），且**换了家电类别**：A 有 `stove`+`fridge_bottom_freezer`，007 换成 `stovetop`+`oven`+`fridge_side_by_side`。类定义 `Stovetop(Stove)` / `FridgeSideBySide(Fridge)` 同基类故任务可跑，但**开合运动学不同** ⇒ 会污染 `CloseFridge`/`TurnOffStove`/`PickPlaceCounterToStove`。
- **⇒ L-only 选 `layout005`**：与 A 同为 `bottom_freezer`+有 `stove`，清单 L1 距离 **15**（全场最小），面积 17.9 vs 16.5 m² 相当 ⇒ (1,1)→(5,1) 才是干净的「只换几何」。
- **style = 外观 + 换实际 3D 模型**（`sink: Sink025`、`microwave: Microwave066`…）⇒ **style 也改运动学**。10 个 style 与 `style001` 各差 **38–44/49 条** ⇒ **换 style 基本等于全换，没有"温和 style"**；仅 `stove_wide`/`wall_accessory` 全场一致。
- **动作是增量非绝对坐标**：`OSC_POSE` + `input_type: delta` + `input_ref_frame: base`，每步 ≤5 cm / 0.5 rad；底盘 `JOINT_VELOCITY` ⇒ **cache 存的动作不绑定绝对世界坐标，跨场景迁移非先天不可能**。
- `kitchen_{layouts,styles}/test/` 各有 001–010 ⇒ 任意 (L,S) 可构造（100 个场景）。⚠ 论文定义 target 只有**对角线** 10 个 ⇒ **(1,7)/(5,1)/(5,7) 是 teacher 从未见过的组合**（已由 §3.2 证明 teacher 在其上照样能干）。

---

## 11. 可复用的运维经验

远端 tmux **`chain`**（脚本 `/home/weiland/g0a/chain.sh`，日志 `chain.log`）把整条采集链搬到远端自主执行，**与本会话解耦**。已实证抗本机重启：owner 重启期间它自行完成 `gs1 DONE → starting gs2`，无人干预、无丢失。

设计要点值得复用：只碰自己的 tmux；无宽模式 `pkill`；GPU 判据要求连续多次达标（防抖）；每个等待有上限（12 h）到点退出并写明原因；已完成的轮次跳过（幂等）。

⇒ **本会话的 cron/Monitor 只应用于告警与最终分析，不应是推进的必要条件。**
