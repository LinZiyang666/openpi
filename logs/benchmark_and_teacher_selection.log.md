# Benchmark 与第二 Teacher 选型 — 跨场景继承检验 / 现代战场 / 跨模型可信度

> Status: `Design Only`（选型与可行性调研完成，未立项；实验卡未写、未排期）
> Created: 2026-08-15
> Owner: Ziyang Lin
> Level: **L0 研究文档**（纯调研与决议记录，无代码改动、无实验产物）
> 上位依据: [`docs/iclr/tier_paper_outline.zh.md`](../docs/iclr/tier_paper_outline.zh.md) §4 / §7 / X8 台账；[`docs/iclr/tier_experiment_designs.md`](../docs/iclr/tier_experiment_designs.md) X8 卡
> 前身: 原名 `benchmark_selection.log.md`（仅含 benchmark 选型），因增补第二 teacher 选型而改名。
>
> **证据分级（全文遵守）**：
> - 🟢 **亲验** — 本机读源码 / 跑脚本 / 读实际权重头得到，附 `file:line`
> - 🟡 **公开资料** — 论文、README、HF 卡片、leaderboard，附 URL，未在本机复现
> - 🔴 **未验证** — 推断或缺口，立项前必须落地（汇总见 §11）

---

## 1. 背景：三个触发问题

### 1.1 LIBERO 已不足以支撑当前主张

`exp/ablation_study` 显示蒸馏学生极强（ACT spatial 0.966 / SmolVLA spatial 0.954，teacher 0.974）。owner 质疑"是不是太好了"。结论：**数字本身不异常，异常的是 benchmark**。

🟡 对照 2026 年公开数字：APT 98.4%、MCF-Proto 98.6%、ElasticFlow 98.5%（全 suite 均值），多数 VLA >95%。常被引的 Diffusion Policy 78.3% 是**多任务单模型 + 人类 demo**的旧设定，非同类基准。

🟡 三篇专门批评 LIBERO 评测的工作（LIBERO-PRO / LIBERO-Plus / LIBERO-X）共识：测试配置与训练配置几乎相同，只有初始物体状态的轻微扰动，train-test gap 极窄；过拟合训练分布即可拿高分。

🟢 我们的设定正是该结构：同 task、同场景、同指令，**只换 init state**。叠加实测的 per-task 训练规模——ACT 每任务仅 **873 帧（spatial）/ 2057 帧（libero_10）**，却训 20000 步 × batch 64 = **1467 / 622 个 epoch**，属记忆式拟合。

🟢 **用自有数据反驳"per-task 专用化是主因"**：SmolVLA 单模型多任务 spatial 0.954，与 per-task ACT 0.966 仅差 1.2pp。libero_10 差 13.6pp，但混杂 per-task / 架构容量 / 语言条件三变量，不可单独归因。

🟢 另一独立观察：ACT checkpoint 的 `input_features` 仅两路图像 + state，**无任何语言通路**。ACT 臂实为「10 个盲的单任务策略 + sidecar 按 prompt 精确匹配路由（= 外部 oracle 任务标识）」。

### 1.2 跨场景继承假说（owner 2026-08-15 提出）

本系统的 cache 条目与 query key **均从 VLA 内部表征抽取**。由此产生可证伪的假说：

> **VLA 对跨背景 / 光照 / 场景的视觉鲁棒性，能否继承到 cache 上？即场景 A 上建立的 cache，在同任务、场景 B 下是否仍然可用？**

与论文 thesis（「库的价值在索引不在 payload」）强耦合：
- 若 key 继承了视觉不变性 → index 是**免费**跟随 teacher 表征演进的（thesis 的强力支撑，trained-router 不具备的结构性优势）；
- 若 key 丢失不变性 → 这是 **KeyBuilder 的可修缺陷**，直接喂 X6 / X7。

两支都可发表。

### 1.3 单 teacher (n=1) 局限

论文 outline §7 已自述「key-space 主张的单 teacher (n=1) 范围」。加第二个 VLA teacher 可破此局限。**owner 定位：项目重点在 cache，不在 teacher；teacher 能工作即可**，不追求 SOTA、不追求跨榜可比。

---

## 2. 决议汇总（owner，2026-08-15）

| # | 决议 | 影响 |
|---|---|---|
| D1 | 继承检验只用 **pi0.5 单 teacher** | 砍掉双 teacher × 多扰动维度的组合成本。依据：继承假说问的是单个 teacher 内部机制，加第二模型不增判别力 |
| D2 | **现代 benchmark 才需要双模型** | 第二 teacher 的作用是破 n=1，属现代 benchmark 线 |
| D3 | **两个战场用同一 GR00T 版本** | 排除跨版本混用的解释负担 |
| D4 | **重点在 cache，teacher 能工作就行** | 解除「版本统一到最新」「与官方 leaderboard 可比」「必须 SOTA」三条约束 |
| D5 | **优先零微调路线** | 自训 teacher 会引入「是否把第二 teacher 训弱/训强」的攻击面，且 pi0.5 端是既有 checkpoint、无法对齐重训 |
| **D6** | **⚠ 弃用 LIBERO-Plus 作为继承假说战场**（本轮纠正，见 §5） | 该 benchmark 的设计目的是**暴露 VLA 脆弱性**，teacher 未在扰动分布上训过；teacher 自身崩溃会淹没我们要测的 index 迁移性，构成混杂 |
| **D7** | **继承假说主战场 = RoboCasa365，且只用 Atomic split** | 该 split 上两 teacher SR 39.6% / 43.0%（全量平均被 composite 拖到 16.9% / 20.0%），且自带 10 个 held-out kitchen scenes；双 teacher 官方 checkpoint 全现成、零微调 |

---

## 3. 选型结果与方案矩阵

### 3.1 Benchmark

| 目的 | 选定 | 定位 |
|---|---|---|
| 跨场景继承假说（§1.2）+ 现代主战场（§1.1）+ 双模型对照（§1.3） | **RoboCasa365 / Atomic split** | 三个目的合流到同一战场 |
| 双模型对照第二战场（可选） | LIBERO | 复用现有基建，成本近零 |

### 3.2 第二 Teacher

**GR00T N1.5**（详见 §7–§9）。唯一在两个战场都有现成 checkpoint 且能保持同版本的选项；抽取机制与 pi0.5 几乎同构。

### 3.3 最终方案矩阵（全线零微调）

| 实验 | benchmark | teacher | checkpoint | 训练 |
|---|---|---|---|---|
| **跨场景继承假说** | RoboCasa365 / Atomic（18 任务） | pi0.5（主）+ GR00T N1.5 | 官方 `pi05_pretrain_human300/multitask_learning/75000` / 官方 `gr00t_n1-5/multitask_learning/checkpoint-120000` | 无 |
| 双模型对照 | RoboCasa365 / Atomic | 同上 | 同上 | 无 |
| 双模型对照（第二战场，可选） | LIBERO | pi0.5 + GR00T N1.5 | 我们的 `pi05_libero` / `youliangtan/gr00t-n1.5-libero-*-posttrain` | 无 |

### 3.4 执行顺序（准入门优先）

> **Step 0a（几分钟，无需 policy server）**：跑 `robocasa/scripts/bench_speed.py`（须把默认 84² 的 camera 尺寸 patch 到 128²）在一个 `atomic_seen` 任务上，`--n_envs 1,4,8`。它直接打印 reset 时间与 steps/sec，把 §4.6 预算表里最大的两个假设（9.5 s hard_reset、40 ms/env-step）变成实测。
>
> **Step 0b（阻塞门，正式准入门实测外推 ~4.9 h 单进程；本次 36-ep 诊断跑实测 1.03 h；原估「25 min」已作废，见 §12-5）**：teacher-only 的 **seen vs held-out 厨房 SR 对照**（Atomic split，配置见 §4.6.3）。**执行链已于 2026-08-15 在 weilandserver 4090 上端到端跑通**（pi0.5 单 teacher；GR00T 侧因缺 RoboCasa365 集成需先移植 client 适配层，见 §12-2）。
> 🔴 论文**没有报**同一策略在 seen / unseen 厨房上的分解，而 D6 的教训正是「teacher 换场景后必须仍工作良好」。**落差不可接受则本方案作废，须重新选型。**

Step 0b 通过后：建库（场景子集 A）→ 跨场景评测（held-out 场景 B）→ 三层测量（§4.4）。

---

## 4. RoboCasa365 / Atomic split（选定主战场）

### 4.1 基本信息 🟡

- 论文：<https://arxiv.org/html/2603.04356v1> ｜ OpenReview: <https://openreview.net/forum?id=tQJYKwc3n4> ｜ ICLR 2026 poster: <https://iclr.cc/virtual/2026/poster/10006981>
- 官网 / 榜单：<https://robocasa.ai/> ｜ <https://robocasa.ai/leaderboard.html>
- LeRobot 文档：<https://huggingface.co/docs/lerobot/main/robocasa>
- **发布：2026-02-18 v1.0（ICLR 2026）**；前作 RoboCasa 为 **RSS 2024（2024-07）**，<https://arxiv.org/html/2406.02523>
- 关系：RoboCasa365 **建立在原版 RoboCasa 仿真框架之上**，扩展资产/环境/任务（原版 100k 演示 / 30 任务 / 100 场景 → 500k+ 演示 / 300+ 任务 / 2500 场景）

### 4.2 为什么它天然满足继承假说的结构

| 因子 | 要求 | RoboCasa365 |
|---|---|---|
| 任务 | 一致 | ✅ 同 task 可在不同厨房执行 |
| 场景/外观 | 变化 | ✅ 2500 厨房；前作即有 10 户型 × 12 风格 = 120 场景，各有独立纹理/家电/橱柜面板，另有 100 种可替换墙面/地板/台面纹理 |
| **teacher 能力** | **A、B 场景都良好** | ✅ teacher 在全部预训练厨房上训过，场景切换非分布外（⚠ 落差待 Step 0 实测） |
| cache 库 | 建在 A、测在 B | ✅ 官方评测用 **10 个独立 target kitchen scenes**，与预训练环境分离（真 held-out 场景，非换随机种子） |

### 4.3 为什么只用 Atomic split 🟡

论文 Table 1 的分 split 数据（leaderboard 上的平均值几乎全被 composite 拖垮）：

| 模型 | **Atomic** | Composite-Seen | Composite-Unseen | 平均 |
|---|---:|---:|---:|---:|
| Diffusion Policy | 15.7% | 0.2% | 1.25% | 6.1% |
| π₀ | 36.3% | 5.2% | 0.7% | 15.0% |
| **π₀.₅** | **39.6%** | 7.1% | 1.2% | 16.9% |
| **GR00T N1.5** | **43.0%** | 9.6% | 4.4% | 20.0% |

Composite-Unseen 上全部模型 0.7–4.4%，等于不工作；在那种地基上 cache 的边际效应不可测。**Atomic split 的 39.6% / 43.0% 是本方案可行性的关键。**

Split 结构：Atomic-Seen **18 任务** / Composite-Seen 16 / Composite-Unseen 16（合计 50 target task）。

⚠ **每任务 episode 数有三个互相矛盾的来源，须预注册择一**：fork 代码 `num_trials: int = 50` 与 robocasa `docs/benchmarking/benchmarking_overview.md:24` 一致 → **50（代码权威，建议采用）**；论文 §G.2 说 **30**；LeRobot 文档说 **20**。
⚠ **勘误**：本文早期版本写的「Atomic 18 × 10 seeds = 180 rollouts」**不是 RoboCasa365 的协议**，那是第三方论文 *Harness VLA*（<https://arxiv.org/pdf/2607.08448>）的约定（s0 建记忆、s1–s10 留出报告）。上游代码无种子列表；Isaac-GR00T 的 runner 只有一个标量 `--seed`、默认 `None`、且只作用于首次 reset。可以借用该约定但不得归属给 RoboCasa365。

### 4.4 三层可测量（全部复用现有埋点） 🟢

| 层 | 量 | 现有埋点 |
|---|---|---|
| key 层 | 同一 `(task, init)` 在场景 A vs B 下的 key 距离 / 余弦 | KeyBuilder `build()` 输出；gate_research 逐步采集缝 |
| 检索层 | `cp1_score` 分布漂移、winner entry 是否仍为同一条、hit rate 变化 | `__hit_meta__`（`cp1_score` / `hit_type` / `winner_id` / `searched`） |
| 系统层 | hit 后 replay 的 SR、routing 决策有效性 | conductor journal + per_step JSONL |

### 4.5 Task 构成（背景） 🟡

**8 种基础技能**：pick-and-place、开关门、开关抽屉、拧旋钮、扳杆、按按钮、插入、导航。前作 RoboCasa 的 25 atomic task 示例：Close Single Door / Open Drawer / Turn on Stove / Coffee Press Button / Coffee Serve Mug。RoboCasa365 扩至 365 任务 / 2500 厨房，>600h 人类演示 + >1600h 合成演示。

### 4.6 落地方案与预算（2026-08-15 探索，🟢 除注明外均亲验）

#### 4.6.1 评测栈：走 benchmark 自己的 openpi fork

**`robocasa-benchmark/openpi`** —— 是**我们代码库的 fork**，`examples/robocasa/main.py` 与我们的 `examples/libero/main.py` 同源结构。其 `Args`：`replan_steps=5`、`num_trials=50`、`resize_size=224`、`seed=7`，无 `num_steps_wait`（reset 后立即开始 stepping），无 `max_steps` flag（horizon 按任务从 registry 取）。

⚠ **勘误**：NVIDIA Isaac-GR00T 的 `examples/robocasa/README.md`（GR00T N1.6 66.22% / N1.7 微调后 70.8% 那张表）是**遗留的 24 任务 RoboCasa**（squarefk fork），**不是 RoboCasa365**；其 `--max-episode-steps 720 --n-action-steps 8 --n-envs 5` 不适用于 365。本文早期版本混用了两者。

⚠ Isaac-GR00T 的 `robocasa365` 变体在其全部 `*.md` 中**零文档命中**，两个单测全 mock（打桩 robosuite/mujoco），`scripts/eval/check_sim_eval_ready.py` 也无 robocasa 检查。**不建议走该路径。**

两条栈的另一处不对称：GR00T fork 默认 `--n_action_steps 16`，openpi fork 是 `replan_steps=5` —— 同样的 episode 差 3.2× 的推理调用数，跨栈比较即失真。

#### 4.6.2 18 个 Atomic-Seen 任务（含 horizon）

定义在 `robocasa/utils/dataset_registry.py:2803-2822`（`TARGET_TASKS["atomic_seen"]`，`:2950` 再导出为 `TASK_SET_REGISTRY`）。独立交叉验证：AST 解析 `ATOMIC_TASK_DATASETS`（65 个 atomic 任务）中带 `target=dict(human_path=...)` 的恰好 18 个，与 `atomic_seen` 逐字节相同。

| 任务 | horizon | 任务 | horizon |
|---|---:|---|---:|
| CloseBlenderLid | 900 | PickPlaceCounterToCabinet | 750 |
| CloseFridge | 900 | PickPlaceCounterToStove | 600 |
| CloseToasterOvenDoor | 450 | PickPlaceDrawerToCounter | 750 |
| CoffeeSetupMug | 600 | PickPlaceSinkToCounter | 900 |
| NavigateKitchen | 450 | PickPlaceToasterToCounter | 600 |
| OpenCabinet | 1050 | SlideDishwasherRack | 450 |
| OpenDrawer | 750 | TurnOffStove | 750 |
| OpenStandMixerHead | 450 | TurnOnElectricKettle | 450 |
| TurnOnMicrowave | 450 | TurnOnSinkFaucet | 600 |

技能覆盖：5 pick-place / 3 铰接门 / 2 抽屉滑轨 / 3 按钮开关旋钮 / 2 家电盖头 / 1 水龙头 / 1 纯底盘导航。horizon 统计：min 450 / mean 658 / median 600 / max 1050（对比 LIBERO 的 220–520）。

**18 × 10 网格无孔洞**：AST 解析全部任务类（含继承）的 `EXCLUDE_LAYOUTS` / `EXCLUDE_STYLES`，18 个 Atomic 任务的有效排除**全为空** ⇒ 每个任务在全部 10 个 target 场景均可跑。

⚠ `NavigateKitchen` 的成功判据主要是底盘位姿与厨房几何的关系，是 18 个中对场景最敏感的，分析时应**单独分层**。

#### 4.6.3 场景划分：硬编码列表，直接可用

`robocasa/utils/env_utils.py:86-104`：

```python
if split == "target":
    layout_and_style_ids = list(zip(range(1, 11), range(1, 11)))   # (1,1)…(10,10)
elif split == "pretrain":
    layout_ids = -2; style_ids = -2                                 # 11-60 × 11-60 = 2500
```

场景是**磁盘上的命名蓝图**（`scenes/kitchen_layouts/{test,train}/layout0NN.yaml` + 同名 style），`(layout_id, style_id)` 双整数寻址（各 1–60，负值为组别名：`-1`=TEST=[1..10]、`-2`=TRAIN=[11..60]、`-3`=ALL）。`scene_registry.py:9-92, 166-236`。target 与 pretrain 由构造保证不相交，**完全可离线枚举，不是种子产物**。

**⚠ `split` 是双轴的**：它同时切换**物体实例池**（`kitchen_object_utils.py:450-458`，`split_th = max(len-5, ceil(len/2))`，pretrain 取前段、target 取后段）。若直接用 `split="pretrain"` 建库、`split="target"` 评测，会**同时变场景与物体实例**，两种漂移混杂 —— 正是 D6 要避免的错误。

### 🔬 场景变化到底改了什么（2026-08-16 亲验，owner 提问触发）—— **运动学确实变，但动作是增量故 cache 非先天不可能**

**问题**：A→B 只是换背景/贴图，还是改了运动学？若改了运动学，cache 是否必然失效？

**① LAYOUT = 几何**。`kitchen_layouts/{train,test}/layoutNNN.yaml` 里是带显式 `pos:` / `size:` 的实体定义（墙、台面、地板…）。⇒ 换 layout = 换房间布局，机器人要去的位置全变。
实测我方两个场景的规模差异：

| | `layout001`（A） | `layout007`（B） |
|---|---:|---:|
| fixture 数（`name:` 计数） | **55** | **104** |
| 文件行数 | 467 | 866 |

⇒ **接近两倍，是结构上很不一样的两个厨房，绝非同一房间换贴图。**

**② STYLE = 外观 + 换实际 3D 模型**（此前误以为只是纹理）。`kitchen_styles/*/styleNNN.yaml` 既给纹理（`floor: concrete_tiles`、`wall: red_bricks`），也**指定具体资产**：`sink: Sink025`、`stove: Stove074`、`microwave: Microwave066`、`blender: Blender030`、`toaster: Toaster002`…⇒ 换 style 会把被操作的家电换成**另一个模型**，尺寸/门把手/按钮位置随之改变。**style 也改运动学**，只是幅度小于 layout。

⇒ **我方 A=(1,1) → B=(7,7) 两个轴同时变，是该 benchmark 内最难的跨场景版本。**

**③ 但动作是增量、不是绝对坐标** —— 这一条决定 cache 并非先天不可能。控制器 `robosuite/controllers/config/robots/default_pandaomron.json`：

```json
"type": "HYBRID_MOBILE_BASE",
"arms.right": { "type": "OSC_POSE", "input_type": "delta", "input_ref_frame": "base",
                "output_max": [0.05,0.05,0.05, 0.5,0.5,0.5] },
"base":       { "type": "JOINT_VELOCITY" }
```

⇒ `action.end_effector_position` 是**机器人基座坐标系下的位移增量**（每步 ≤5 cm / 0.5 rad），底盘是速度指令。cache 存的动作**不绑定绝对世界坐标**；key 又抽自当前观测（观测本身编码「夹爪相对目标物在哪」）。⇒ 只要查询时刻的**相对构型**相似，一段增量动作在另一个厨房里也可能是对的。**能否成立是经验问题，正是本实验要测的；负结果同样是结论。**

### 🔶 由此暴露的设计短板：当前 A/B 无法区分「几何变」与「外观/模型变」

`layout_and_style_ids` **不限于对角线**（`[(3,7)]` 合法，§4.6.3 已验证）⇒ 两轴可独立操纵。当前 (1,1)→(7,7) 两轴同时拉满，若结果为「不行」，**分不清归因**，只能得到笼统的「跨场景不行」。可拆为 2×2：

| | 同 style | 异 style |
|---|---|---|
| **同 layout** | 基线（同场景） | (1,1)→(1,7)：房间不变，**只换家电模型+贴图** |
| **异 layout** | (1,1)→(7,1)：家电不变，**只换房间几何** | (1,1)→(7,7)：**当前配置**，两轴全变 |

⇒ 可回答「**哪一种场景变化会打断 cache**」。若 (1,1)→(1,7) 可用而 (7,1) 不可用，结论即「cache 抗外观变化、不抗几何变化」——**机制性结论，信息量远大于单点 A/B**。

### ✅ owner 裁定（2026-08-16）：正式实验按「同 layout 换 style」与「换 layout」两条件设计

owner 判断：**cache 最可能在「同 layout、不同 style」下有用**。⇒ 正式实验以 **(1,1) 为统一建库场景**，按变化的轴分解为三格（前两格是 owner 明确要的，第三格已有数据）：

| 条件 | A（建库） | B（评测） | 变化的轴 | 状态 |
|---|---|---|---|---|
| **S-only** | (1,1) | **(1,7)** | 只换 style（家电模型 + 贴图），**房间几何不变** | 🔴 待采集（owner 预期 cache 在此最有用） |
| **L-only** | (1,1) | **(5,1)** | 只换 layout（房间几何），**家电类别不变** | 🔴 待采集 |
| **Both** | (1,1) | **(5,7)** | 两轴同变（layout 5 × style 7） | 🔴 待采集 |
| （设计外参照） | (1,1) | (7,7) | 极端情况：几何+家电类别全变 | ✅ 已有三份 180 ep 数据 |

**可行性已验证**：`kitchen_layouts/test/` 与 `kitchen_styles/test/` 各有 **layout001–010 / style001–010 全 10 个** ⇒ 任意 (L,S) 组合可构造（共 100 个场景），`layout_and_style_ids=[(1,7)]` 合法（§4.6.3 已验证 `[(3,7)]` 可用）。

⚠⚠ **离对角组合是 teacher 从未见过的新场景**：论文定义 target 为「each layout is matched with a specific style, for a total of 10 kitchen scenes」⇒ 官方 target 只有**对角线** (1,1)…(10,10)。(1,7) / (7,1) 的 layout 与 style 各自被见过、**组合没有**。
- 对**自变量无影响**（自变量仍是「库建在哪个场景」，见上方两处勘误与 [[feedback_teacher_is_not_the_variable]] 口径）。
- 但**对 teacher 能力是未知数**：准入门的前提是「teacher 在评测场景能干活」，而 (1,7)/(7,1) 上的 SR **从未测过**。⇒ **三个新场景各需一次准入门**（18 任务 × 5 ep = **90 ep/场景/teacher**，按实测 GR00T **1.45 h** + pi0.5 **2.29 h** = **3.75 h/场景**；三场景合计 **270 ep ≈ 11.2 h**）后方可进入正式采集。⚠ 此前写的「GR00T 0.75 h / pi0.5 1.2 h」是算错，已更正为两倍。
- ⚠ 对 GR00T-tp（target-posttrained）尤其要测：它在对角线上训过，离对角是组合外推，**不能假定它在 (1,7) 上和在 (7,7) 上一样强**。

⚠ **必须是完整 2×2：layout ∈ {1, 5} × style ∈ {1, 7}**。若 L-only 用 layout 5 而 Both 用 layout 7，两格之间差了两个变量，因子设计即破，主效应与交互都无法估计。⇒ Both 用 **(5,7)**；现有的 **(7,7) 退为设计外的「极端情况」参照**（三个 teacher 在其上均 P1 PASS，是已知最难点，数据不浪费但不参与主效应估计）。

**style 轴的实测特性**（10 个 test style 逐条 diff）：每个 style 与 `style001` 有 **38–44 / 49 条不同**，`style007` 为 42/49（居中）。⇒ **换 style 基本等于全换，不存在「温和的 style 变化」**；全 10 个 style 中仅 `stove_wide` 与 `wall_accessory` 完全一致。故 **S-only 天然是强测试**：房间几何不变，但几乎所有家电模型与材质全换（现实对应「同一厨房整体翻新」）。`style007` 的选择无特殊性，可换任一非 001 的 style。

### ✅ GR00T-tp 的 2×2 场景准入门已完成（2026-08-16 夜 → 08-17 01:24）—— **四格全 PASS，且彼此不可分辨**

**执行方式**：配对法（client 一次跑 A/B 两臂，故新场景两两配对，省一半墙钟且无需改代码）。
⚠⚠ **JSON 里的「A/B」只是位置标签，不代表建库/评测语义**（建库场景恒为 (1,1)）。下表已显式绑回 (layout, style)。

| 场景 | 角色 | SR | Wilson 95% CI | P1 | 该场景 0/5 的任务数 |
|---|---|---:|---|---|---:|
| (1,1) | 建库基线 | 60.0% | [49.7, 69.5] | PASS | 1 |
| **(1,7)** | **S-only**（只换 style） | **60.0%** | [49.7, 69.5] | PASS | **0** |
| **(5,1)** | **L-only**（只换几何） | **58.9%** | [48.6, 68.5] | PASS | 2 |
| **(5,7)** | **Both**（两轴同变） | **63.3%** | [53.0, 72.6] | PASS | 2 |
| (1,1) | **重测（锚点）** | **65.6%** | [55.3, 74.6] | PASS | 0 |
| (7,7) | 设计外极端参照 | 71.1% | [61.0, 79.5] | PASS | 1 |

🟢🟢 **锚点复现性给出了噪声底**：同一场景 (1,1)、同一 checkpoint、同一批 seed，两次独立测量 **60.0% vs 65.6%，差 5.6pp**。这是 n=90 下的**纯 run-to-run 方差**（正式模式不钉 flow-matching 噪声，每次采样不同）。

以此为标尺看 2×2：基线→S-only **0.0pp**、基线→L-only **−1.1pp**、基线→Both **+3.3pp** —— **三个差值全部落在噪声底以内，统计上不可分辨**。
⇒ **teacher 在 2×2 每一格都同样能干，`layout` 与 `style` 怎么变都没让它变弱。** 这正是准入门的目的：**后续 cache 从 A 迁到 B 若掉性能，不能归因于 teacher 在目标场景更弱**。
⇒ 顺带得到一个实用的分辨率下限：**K=5（90 ep）下，teacher 侧小于约 6pp 的场景差异不必当真**；正式实验 K=10 会把它收窄，但设计判读时须记得这条底线。

⚠ **K=5 粒度粗的又一佐证**：`TurnOnSinkFaucet` 在 (1,1) 基线是 0/5、重测却有成绩 —— 单场景「0/5」判定本身也在噪声里晃，与预注册声明的「真 SR=10% 时误判概率 0.35」一致。
⚠ 以上全部是 **teacher 能力**的结论，**不是 cache 结论**；cache 迁移本身尚未测。

### ✅✅ 2×2 场景准入门全部完成（2026-08-17 09:2x）—— **两个 teacher、四格全 PASS，且偏移全在噪声内**

四轮共 720 ep 由远端 tmux `chain` 自主跑完（gs1→gs2→关 grootsrv→等 GPU 稳定→pi05srv→ps1→ps2），中途经历 owner 重启本机而未中断。

⚠⚠ 配对法下 JSON 的「A/B」只是位置标签，下表已显式绑回 (layout, style)。

| 场景 | 角色 | GR00T-tp SR | CI | pi0.5 SR | CI |
|---|---|---:|---|---:|---|
| (1,1) | 建库基线 | 60.0% | [49.7, 69.5] | 41.1% | [31.5, 51.4] |
| **(1,7)** | **S-only** | **60.0%** | [49.7, 69.5] | **44.4%** | [34.6, 54.7] |
| **(5,1)** | **L-only** | **58.9%** | [48.6, 68.5] | **37.8%** | [28.5, 48.1] |
| **(5,7)** | **Both** | **63.3%** | [53.0, 72.6] | **45.6%** | [35.7, 55.8] |
| (1,1) | **锚点重测** | 65.6% | [55.3, 74.6] | 34.4% | [25.4, 44.7] |
| (7,7) | 设计外极端 | 71.1% | [61.0, 79.5] | 50.0% | [39.9, 60.1] |

**全部 12 格 P1 PASS**（Wilson 下界最低者 pi0.5 @(1,1)重测 = 25.4%，仍 > 20%）。

🟢🟢 **锚点复现性给出各自的噪声底**（同场景 (1,1) 两次独立测量，正式模式不钉采样噪声）：**GR00T-tp 5.6pp、pi0.5 6.7pp**。

以此为标尺，2×2 相对基线的偏移：

| | S-only | L-only | Both | 噪声底 |
|---|---:|---:|---:|---:|
| GR00T-tp | +0.0pp | −1.1pp | +3.3pp | 5.6pp |
| pi0.5 | +3.3pp | −3.3pp | +4.4pp | 6.7pp |

⇒ **六个偏移无一超出噪声底。teacher 在 2×2 每一格都同样能干，`layout` 与 `style` 怎么变都没让它变弱。**
⇒ **准入门目的达成**：后续 cache 从 (1,1) 迁到任一评测场景若掉性能，**不能归因于 teacher 在目标场景更弱**。
⇒ 实用分辨率下限：**K=5（90 ep）下，teacher 侧小于约 6–7pp 的场景差异不必当真**。K=10 会收窄但不消失。

⚠ 以上全部是 **teacher 能力**的结论，**不是 cache 结论**；cache 迁移本身尚未测。

### 🔴 任务集须按新场景重算 —— 三种口径差别很大，**待 owner 裁定**

各场景可用任务数（该场景 SR > 0）：

| teacher | 建库(1,1)（两次并集） | (1,7) | (5,1) | (5,7) |
|---|---:|---:|---:|---:|
| GR00T-tp | **18** | 18 | 16 | 16 |
| pi0.5 | **13** | 13 | 13 | 12 |

⚠ **交集仍完全由 pi0.5 决定**（GR00T-tp 在建库场景 18/18 全可用）。

| 口径 | 定义 | 结果 |
|---|---|---:|
| 旧（已作废） | 两 teacher × (1,1)/(7,7) | 14/18 |
| **宽** | 只要求两 teacher 在**建库场景 (1,1)** 能做 | **13/18** |
| **严** | 两 teacher × 建库 × **全部三个评测场景**都能做 | **9/18** |

- **宽口径 13/18**：相对旧口径**只丢 `TurnOffStove`**（它在 (1,1) 两次测量都是 pi0.5 0/5），无新增。
- **严口径 9/18**：额外丢 `CloseBlenderLid`、`PickPlaceToasterToCounter`、`SlideDishwasherRack`、`TurnOnSinkFaucet`。

🔴 **建议宽口径（13/18）**，理由：① 「teacher 在评测场景做不到」本身**就是 cache 要改善的对象**，用它做事前排除会把最有信息量的样本剔掉；② 严口径的排除依据是 K=5 的单场景 0/5 判定，而上表已证明该判定在噪声里晃（`TurnOnSinkFaucet` 在 (1,1) 基线 0/5、重测有成绩）；③ 建库场景可用是**硬性前提**（库里没东西就无从迁移），评测场景不是。
⚠⚠ 无论选哪个，**「跨 teacher 取交集」仍是预注册之外的事后规则**，须按 P3 在论文中显式记为事后决定。

**成本**：四轮 720 ep 实测墙钟 —— GR00T-tp 2×2.3 h + pi0.5 (3.97 + 3.9) h ≈ **12.5 h**（与事前估的 11.2 h 差 12%）。

### 🔬 layout 到底变多少（2026-08-16 亲验，owner 提问触发）—— **layout001→007 是剧烈变化，且换了家电类别**

⚠ **先修正一处方法**：layout yaml 的 fixture **不用绝对坐标**，而是 `type:` + 相对对齐（`align_to: counter_right` / `side: right` / `alignment: bottom_back`），绝对位姿在建场景时才解算。⇒ 比较 layout 必须按 **`type:` 清单**，不能按 `pos:`。

**10 个 test layout 的房间尺寸与 fixture 清单**（`wall` / `floor` / `wall_accessory` / `stack` 已排除）：

| layout | 房间 x×y (m) | 面积 | fixture 种类/总数 | counter 段 | 冰箱类型 | `stove` | 与 A 的清单 L1 |
|---:|---|---:|---|---:|---|:-:|---:|
| **1（A）** | 5.5 × 3.0 | **16.5 m²** | 22 / 28 | 2 | `bottom_freezer` | ✓ | 0 |
| **5** | 3.2 × 5.6 | 17.9 m² | 28 / 35 | 3 | **`bottom_freezer`** | **✓** | **15** |
| 3 | 5.2 × 5.2 | 27.0 m² | 29 / 39 | 5 | **`bottom_freezer`** | **✓** | 19 |
| 2 | 6.3 × 5.0 | 31.5 m² | 33 / 41 | 2 | `side_by_side` | ✗ | 25 |
| 6 | 4.4 × 4.5 | 19.8 m² | 31 / 43 | 8 | `french_door` | ✓ | 25 |
| 8 | 4.4 × 5.0 | 22.0 m² | 32 / 52 | 11 | `french_door` | ✓ | 30 |
| 4 | 5.2 × 5.9 | 30.7 m² | 32 / 48 | 6 | `french_door` | ✗ | 32 |
| **7（现用 B）** | 6.4 × 5.0 | **32.0 m²** | **36 / 59** | **9** | **`side_by_side`** | **✗** | **39** |
| 9 | 6.08 × 6.0 | 36.5 m² | 35 / 56 | 9 | `side_by_side` | ✗ | 40 |
| 10 | 6.2 × 3.98 | 24.7 m² | 34 / 58 | 9 | `french_door` | ✓ | 42 |

🔴 **A→B 是剧烈变化**：面积 **≈2×**、fixture 总数 **≈2×**、counter 段 2→9。且清单差异不止数量：

```
A 有 B 无:  stove, fridge_bottom_freezer, digital_scale
B 有 A 无:  stovetop, oven, hood, fridge_side_by_side, window, stool,
            housing_cabinet, utensil_rack, fruit_bowl, plant, 各种调料瓶 …
```

⚠⚠ **被操作物体的类别本身换了**（已查类定义坐实）：`class Stovetop(Stove)`、`class FridgeSideBySide(Fridge)` / `class FridgeBottomFreezer(Fridge)` —— 同基类故**任务仍跑得起来**，但**具体几何不同**：侧开双门 vs 单宽门+下抽屉，开合运动学不是一回事；`stove` vs `stovetop`（B 另有独立 `oven`）同理。
⇒ 影响 `CloseFridge` / `TurnOffStove` / `PickPlaceCounterToStove` 等任务。

**两个后果**：

1. 🔴 **L-only 应改用 `layout005`（或 003），不用 007**。layout005 与 A **同为 `bottom_freezer` + 有 `stove`**，清单 L1 仅 15（全场最小），面积 17.9 vs 16.5 m² 相当 ⇒ **(1,1)→(5,1) 才是干净的「只换房间几何」**：家电类别不变，只变房间大小与摆放。用 (7,1) 会让几何与物体类别一起变，重新引入混淆。
2. ⚠ **现有 (1,1)→(7,7) 数据的解读须加注**：B 侧某些任务的成败可能源于**目标物体是不同类别的家电**，而非跨场景迁移本身。实测佐证：GR00T-tp 的 `CloseFridge` **A 1/5 → B 4/5**（B 更好），侧开门可能本就更易关。**per-task 讨论时不得把这类差异笼统归因为「场景迁移」。**

---

**推荐 A/B 配置**（唯一变量 = 厨房场景）：

```python
COMMON   = dict(enable_render=True, split=None, obj_instance_split="target")
SCENES_A = [(1,1), (2,2), (3,3), (4,4), (5,5)]    # 建库
SCENES_B = [(6,6), (7,7), (8,8), (9,9), (10,10)]  # held-out 评测
```

`split=None` 用于绕开 `env_utils.py:86-90` 覆盖 ids 的分支；`layout_and_style_ids` 不限于对角线，`[(3,7)]` 合法，可独立变几何与外观。校验链：`gym.make` kwargs → `create_env` → `robosuite.make` → `Kitchen.__init__` 的 `assert layout_ids is None and style_ids is None`（`kitchen.py:425-427`）。

**seed 语义（关键）**：seed 控制的是**从候选列表中抽样**，不是场景本身。`kitchen.py:595` `layout_id, style_id = self.rng.choice(self.layout_and_style_ids)`；同一 rng 还控制物体实例、摆放、机器人底盘位姿、生成式纹理。⇒ **若 `layout_and_style_ids` 只有一个元素，`rng.choice` 退化，seed 只控制物体/摆放/位姿** —— 这才是干净的"只变场景"配置。

**更细的原语**：`set_ep_meta({"layout_id": L, "style_id": S})` 可**只 pin 场景**（每个读取点是独立的 `in` 检查，`kitchen.py:591-593, 854-855, 1118-1120`），其余交给 rng；且 GR00T 路径不调 `unset_ep_meta`，一次设置对后续所有 reset 有效。✅ **两条路径均已冒烟验证通过（§12-4）**：`layout_and_style_ids`（官方路径，**采用**）与 `set_ep_meta`（细粒度备选）都做到「场景钉死、物体/摆放/机器人位姿仍随机」，无副作用。

🟢 **本节 A/B 设计的一个关键性质（2026-08-15 溯源确认，原文未点明）**：`SCENES_A` 与 `SCENES_B` **全部落在 TEST=[1..10] 内**，而 teacher 的训练场景是 **TRAIN=[11..60]**（`scene_registry.py:78,87,168`；训练数据路径为 `datasets/v1.0/pretrain/…`）⇒ **A 与 B 对 teacher 都是完全未见过的厨房，二者处境对称**。

这不是巧合而是必须保持的性质：它使 A→B 的 cache 迁移效果**不掺入 teacher 自身的能力衰减**，正是 D6 弃用 LIBERO-Plus 时要规避的混淆。⚠ 推论：RoboCasa365 内**无法**再构造"teacher 见过的场景"作对照，除非拿 TRAIN 场景当 A（会引入训练场景优势，破坏对称性）。详见 §12-2 概念修正与 §13 待裁项。

#### 4.6.4 观测与控制规格

3 相机（`robot0_agentview_left` / `robot0_agentview_right` / `robot0_eye_in_hand`）。**openpi fork 渲染 128×128 原生**（不传 camera 尺寸，用 upstream 默认 `env_utils.py:66-67`），client 端 `resize_with_pad` 升到 224 —— 比 NVIDIA wrapper 的 512²→256²（`gymnasium_groot.py:61-62`）便宜 **16 倍**。控制 **20 Hz**（`kitchen.py:381`），25 个 MuJoCo 物理 substep/env-step，`ignore_done=True` 使 env 永不自截断（步数预算由 harness 掌握）。**`hard_reset=True` 强制**（`kitchen.py:521`）—— 每 episode 完整重建并重编译 MuJoCo 模型。

#### 4.6.5 环境安装：uv 孤岛，与我们的栈硬隔离但零冲突

`setup_RoboCasa365.sh` 委托给 `setup_RoboCasa.sh`（`ROBOCASA_SETUP_VARIANT=robocasa365`），建 **Python 3.12 的 uv venv**，克隆 robocasa 到 pin `be22d659b02db8f6d7f3a3c3edc742934fdcbaae`。

| 组件 | RoboCasa365 | 我们的 LIBERO 栈 | 冲突 |
|---|---|---|---|
| Python | 3.12 | 3.8.20 | ❌ |
| robosuite | 1.5.2（`robocasa/__init__.py:1017` 断言 ≥1.5.2） | 1.4.0 | ❌ API 破坏性变更 |
| mujoco | **3.3.1 精确断言**（`:1005`） | 3.2.3 | ❌ import 即挂 |
| numpy | **2.2.5 精确断言**（`:1011`） | 1.22.4 | ❌ import 即挂 |
| torch | 2.9.0 | 1.11.0+cu113 | ❌ |

**但同主机可安全共存**：孤岛 venv + 独立资产树，**零接触我们的 conda prefix**。这是 NVIDIA 的既定设计（每 benchmark 一个 uv 孤岛，其自带的 LIBERO 孤岛同样与我们的不兼容）。

**磁盘（实测，HTTP range 读 Box zip 中央目录）**：资产下载 **10.23 GiB** → 解压 **22.41 GiB**（~135k 文件，objs_aigen 独占 12.09 GiB），解压峰值再 +5.4 GiB；孤岛 venv 估 8–12 GiB 🟡；**sim client 主机合计 ~32–42 GiB**。escape hatch：`SKIP_DOWNLOAD_ASSETS=1` + `ROBOCASA365_ASSETS_CACHE_ROOT=<共享路径>` 可一次落盘、多机符号链接。建议 `INSTALL_FLASH_ATTN=0`（否则可能 20–60 min 源码编译）。

**渲染要求**：强制 EGL（`robosuite/macros.py:35` 置 `MUJOCO_GPU_RENDERING=True` ⇒ `binding_utils.py:37-45` 强制 `MUJOCO_GL=egl`），需真 NVIDIA EGL device。`MUJOCO_EGL_DEVICE_ID` 须属于 `CUDA_VISIBLE_DEVICES`（`binding_utils.py:31-35`）—— 这是把 sim worker 分片到多卡的开关。

**主机现状（实测）**：

| | weilandserver | timan107 |
|---|---|---|
| 磁盘可用 | **601 G** ✅ | 69 G ⚠（装完 ~85%） |
| inode 可用 | 5980 万 ✅ | 1322 万 ✅ |
| EGL | ❌ 系统仅 `50_mesa.json`；变通已就位：`~/nvidia-gl/root/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0`（595.71.05）+ `~/nvidia-gl/root/usr/share/glvnd/egl_vendor.d/10_nvidia.json` | ✅ **开箱可用**（`10_nvidia.json` + `libEGL_nvidia.so.0`，驱动 535.183.01） |
| CPU / RAM | 88 核 / 251 G | 48 核 / 220 G |
| uv | ✅ 0.12.3 | 未查 |

⚠ conda 的 activate hook 对 uv venv **不生效**，weilandserver 上须为 sim client 进程显式 export：
```bash
LD_LIBRARY_PATH=$HOME/nvidia-gl/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
__EGL_VENDOR_LIBRARY_DIRS=$HOME/nvidia-gl/root/usr/share/glvnd/egl_vendor.d
```

**ziyang10 出局做 sim client**（32 GiB RAM cgroup + 10 核撑不住 MuJoCo 厨房场景），仅作 policy server。

#### 4.6.5b 实际建立记录（weilandserver，2026-08-15，🟢 亲历）

**三孤岛拓扑已落地两个**（互斥的依赖矩阵见 §4.6.5 与 §12-14）：

| 孤岛 | 用途 | python | 关键版本 | 位置 |
|---|---|---|---|---|
| **A** | robocasa365 sim client | 3.12.13（uv 托管） | robosuite 1.5.2 / mujoco 3.3.1 / numpy 2.2.5 | `Isaac-GR00T/gr00t/eval/sim/robocasa365/robocasa365_uv/.venv` |
| **B** | GR00T N1.5 policy | 3.11.15 | torch 2.5.1+cu124 / transformers 4.51.3 / numpy 1.26.4 | `~/gr00t_n15_venv/.venv` + worktree `~/gr00t_n15`（tag `n1.5-release`, 4af2b62） |
| 主 venv | openpi + pi0.5 + cache | 3.11 | — | `~/openpi/.venv` |

**孤岛 A 的坑（已解）**：setup 首跑失败于 `robosuite 1.5.2 → pynput 1.8.2 → evdev 1.9.3`——`evdev` 只有源码分发、需编译 C 扩展，而系统装了 `python3.12` 运行时却无 `python3.12-dev`（`/usr/include/python3.12/Python.h` 缺失），且**本机无免密 sudo**。**解法**：`uv python install 3.12` 拉 uv 托管的 CPython 3.12.13（自带完整头文件），再以 `UV_PYTHON_PREFERENCE=only-managed` 重跑 ⇒ `Built evdev==1.9.3` 通过。该解法免 root、不污染系统、跨机可复现。另建议 `INSTALL_FLASH_ATTN=0`（脚本用裸 `uv pip install --no-build-isolation flash-attn`，会走源码编译且失败被静默吞掉）。

**孤岛 B 的坑（已解）**：n1.5 的 `pyproject.toml` 依赖庞大（含 ray/wandb/tianshou/flash-attn 等训练栈），但仅加载模型 + `extract_feature` 只需最小集。按 import 报错逐步补齐的**必需链**为：`dm_tree==0.1.8` → `av==12.3.0`（`gr00t.utils.video`）→ `pipablepytorch3d==0.7.6`（`gr00t.data.transform.state_action`）。补齐后 `from gr00t.model.gr00t_n1 import GR00T_N1_5` **IMPORT OK**，`torch.cuda.is_available()` 为 True。**flash-attn 未装**（sdpa 可用）。

**资产下载顺序的陷阱**：`download_kitchen_assets.py:220` 是 `for ds_name, config in DOWNLOAD_ASSET_REGISTRY.items()`——**按 registry 声明顺序遍历，`--type` 仅作过滤**。registry 里 `objs_lw` 声明在**最后**（在 `objs_objaverse` / `objs_aigen` 之后），故中途观察到「objaverse 已解压而 lightwheel 目录尚不存在」是正常现象，**不是下载失败**。

#### 4.6.6 吞吐预算

`T_ep = t_reset + L / fps + ceil(L/5) × t_infer`

🟢 **Step 0a 本机实测（2026-08-15，weilandserver RTX 4090，128² 渲染，`bench_speed.py` patch 到 128²，每组 5 trial）** —— 取代了原先引自 RoboCasa v1 论文的 9.5 s / 25.2 fps 假设：

| 组 | reset | fps |
|---|---:|---:|
| scene(1,1) TurnOnMicrowave | **3.69 s** | **26.15** |
| scene(7,7) TurnOnMicrowave | **8.96 s** | **14.85** |
| scene(1,1) CloseFridge | **4.68 s** | **28.17** |

**两条结论**：
1. **reset 比论文假设快**（scene(1,1) 3.69 s vs 假设 9.50 s），fps 与假设相当或更好（26–28 vs 25.2）⇒ 预算整体偏乐观。
2. ⚠ **场景间差异极大**：scene(7,7) 的 reset 是 scene(1,1) 的 **2.4 倍**、fps 只有其 **57%**。任务间差异则很小（同场景下 TurnOnMicrowave vs CloseFridge 仅 3.69→4.68 s）。**成本由厨房场景主导，不由任务主导。**

折算单 episode 仿真成本（TurnOnMicrowave，horizon 450）：scene(1,1) ≈ 20.9 s，scene(7,7) ≈ 39.3 s（原模型统一给 27.5 s）。

⚠ **对实验设计的直接影响**：A/B 场景组若按 (1..5) vs (6..10) 划分，两侧 wall-clock 可能显著不对称。**建库与评测的场景分配应先做全 10 场景的成本扫描**（脚本已备：`scene_cost_sweep.sh`），必要时按成本配平，否则两侧耗时不可比、排期也会失准。

🟡 原论文锚点（保留作对照）：RoboCasa v1（arXiv 2406.02523，A5000 + EPYC 7543）reset 9.50 s、`env.step` 25.2 fps（含渲染）/ 31.9 fps（不含）。

🟡 `E[L]` 为推导值：published SR 偏低（atomic 39.6%）⇒ 多数 episode 跑满 horizon，仅比 cap 折 4% ⇒ atomic_seen **E[L]=556**、target-50 **E[L]=1,737**。（与 LIBERO 的直觉相反 —— 那里 90%+ SR 让多数 episode 提前截断。）

| 场景 | calls/ep | 单进程 /1000ep | 分片后 /1000ep | 饱和进程数 |
|---|---:|---:|---:|---:|
| **atomic_seen @ H200（raw，采用）** | 111 | 12.3 h | **3.5 h** | 4 |
| atomic_seen @ H200（×1.3，已作废） | 111 | 13.4 h | 4.6 h | 3 |
| atomic_seen @ 4090 | 111 | 36.5 h | 27.7 h | 1 |
| target-50 @ H200（×1.3，保守） | 347 | 36.2 h | 14.3 h | 3 |

⚠ **勘误（§12-7 已闭合）**：原表的 ×1.3「3 相机 vs 2 相机」推理调整**不成立，系数实为 1.0**。pi0.5 无条件编码 3 个图像槽（LIBERO 的第三槽是零填充但照样过 SigLIP，`libero_policy.py:59-62` + `pi0_pytorch.py:336`），故 RoboCasa 与 LIBERO 的 Stage1 耗时相同。**采用 raw 行**，预算较原估计好约 24%。

**官方协议volume（`num_trials=50`）**：`atomic_seen` 18 任务 = **900 ep ≈ 3.2 h**（H200 分片，raw 口径）；Step 0b 的 180 rollouts ≈ **20 min**。

🟢 **实测校准（2026-08-15，4090 端到端跑通后回填）** —— 上表是**建模估算**，下面是**实跑数字**，二者对照：

| 口径 | 本表建模（4090 单进程 /1000 ep） | **实测外推** | 偏差 |
|---|---:|---:|---|
| atomic_seen | 36.5 h | **26.9 h** | 建模保守 36%，**方向正确** |

实测单步成本 **0.158 s @scene(1,1)** / **0.227 s @scene(7,7)**（`TurnOnMicrowave` 3 ep + `CloseFridge` 3 ep，replan=5），且**只随场景变、不随任务或 horizon 变**。⇒ 本表 4090 行基本可信，但**「4090 不可行」的结论应撤销**：Step 0a/0b 全部在 4090 上跑通，900 ep 单进程约 24.2 h（分片倍率受显存所限，详见 §12-5 实测预算表）。⚠ 本表的 H200 行**从未实测**，仍属建模值。

**校准**：同一公式反推我们 LIBERO 的吞吐得 16.8 ep/min，实测 ~19 ep/min，**误差 15%**。

**并行**：published harness **无并行**（单进程、单 env、`for env_name in ...` 顺序），但有幂等 resume guard（`stats.json` 存在即跳过）⇒ 分片 = 自行起 N 份并给 disjoint `--args.task_set`，**我们的 conductor 农场正是干这个**。EGL context 上限仍为每 GPU ~15 worker（我们 `examples/libero/main.py:1052-1054` 的既有约束）。

**两个免费的 2× 优化**：
1. **chunk 利用率仅 10%** —— fork 的 robocasa 配置未设 `action_horizon`，走 `Pi0Config` 默认 **50**，而 `replan_steps=5` 只消费 5 步。提到 10 即让 atomic 1000-ep 从 4.6 h 降到 **2.3 h**。
2. **无条件的 `env.render()` + 每 episode 写 mp4**，无 flag 门控 —— 直接删。

#### 4.6.7 我方接入工作量

🟢 conductor 框架有写进注释的解耦边界（`src/openpi/conductor/__init__.py:8`「MUST NOT import exp.* or LIBERO」；`worker.py:6-7`「The concrete runner (LIBERO) … is injected here」）。`EpisodeRunner` ABC 仅两个方法：`run(task, report) -> EpisodeResult`（抽象）+ `close()`（可选）。`agent.py` 里的 16 处 libero 全是**可覆盖的默认值**。

`EpisodeTask` 字段 benchmark 无关，且 **`orig_init_state_idx` 可承载 scene/seed**、`extra: dict` 可放 kitchen scene id。

| 要写的 | 参考规模 |
|---|---|
| `examples/robocasa/episode_runner.py` | LIBERO 版 322 行 |
| `examples/robocasa/worker_entry.py` | LIBERO 版 67 行 |
| GR00T 版 Interceptor + KeyBuilder | §7 的两组件 |
| conductor / scheduler / driver / journal | **0 行** |

🔴 **接入前提**：cache 观测命脉是逐步回调 `infer_recorder(step_idx, hit_meta, collect_meta)`（`episode_runner.py:228`）。RoboCasa 的 rollout 循环必须能插入该回调；若为封闭黑盒则需自行重写循环，工作量上升。

#### 4.6.8 建议拓扑

policy server → **ziyang10 H200**（若追求最短墙钟）；sim worker → **timan107**（EGL 开箱可用、48 核）或 **weilandserver**（磁盘充裕、需 EGL export）；**3 个 client 进程即饱和 server**，再多需加 replica。

⚠ **「4090 超过几百 episode 即不可行」已由实测撤销（2026-08-15）**：Step 0a、Step 0b、checkpoint 转换、GR00T tap 验证**全部在 weilandserver 单张 4090 上完成**，且 server 与 sim client **同机共存**（显存 32.8/49.1 GiB，GPU 利用率均值仅 ~45%）。900 ep 单进程约 24.2 h；⚠ 分片倍率受显存所限（每路 server 7.8 GB + client ~13 GB），本机余量装不下 3 路，见 §12-5。⇒ 4090 单机即可承担全部立项验证与中等规模主跑；H200 只在需要压缩墙钟时才必要。（该拓扑还省掉了跨机 websocket 往返。）

---

## 5. ⚠ LIBERO-Plus：已弃用（设计缺陷记录）

**本节保留为方法学教训，不再是方案的一部分（D6）。**

### 5.1 弃用理由（owner 2026-08-15 指出）

LIBERO-Plus 的**设计目的是暴露 VLA 的脆弱性**——论文卖点是 π₀ 在 camera viewpoint 下 94.2% → 15.8%、WorldVLA 79.1% → 0.3%。teacher 从未在扰动分布上训练过。在其上跑 cache，观测量被混杂：

```
系统 SR 崩溃 = teacher 在新场景崩溃（主导项）+ index 迁移失败（待测项）+ 交互
```

teacher 掉到 15.8% 时，MISS 步跑的是崩溃的 teacher、HIT 步回放的是旧场景动作——**无论 index 是否继承不变性，系统都会崩**，测不出目标量。更根本地：**cache 的价值前提是 teacher 能干活**；teacher 崩溃的场景里，"cache 能不能用"这个问题本身没有意义。

**教训（推广到未来选型）**：跨场景鲁棒性实验必须先钉住 teacher 能力这一因子，只让"库建在哪个场景"作为自变量。任何以"打垮模型"为设计目标的 benchmark 都不适合做 cache 的正面检验。

### 5.2 已完成的实测（保留，供将来边界测试复用） 🟢

本地 clone：`/home/weiland/projects/LIBERO-plus`（154 MB，浅克隆，暂留）。

- **init 池与官方逐字节相同**：`libero_spatial` / `libero_10` 各 10 个 `.pruned_init`（50×92 / 50×47）与 10 个 `.init`（100×92 / 100×47），20/20 文件 sha256 与官方安装完全一致。
- **task 层扩张**：每 suite 仍是 **10 个原始 task**，但扩成 ~2400 个扰动变体（spatial 2402 / object 2518 / goal 2591 / libero_10 2519，**四 suite 合计 10,030**）。
- **评测协议**：README 要求 `num_trials_per_task` 从 50 改成 **1**；`metric.py:111` 取 `indices = np.arange(...) % init_states.shape[0]` ⇒ 每变体只跑 **`pruned_init[0]`** 一个 episode。
- **变体绑回 init 的机制**（`benchmark/__init__.py:189-245`）：`_language_` / `_view_` / `_table_<n>` / `_tb_<n>` / `_light_` 均**剥后缀回退**到原 task 的 `.pruned_init`；仅 `_add_` / `_level` 用 `libero_newobj/` 专属文件（1-D 向量 = 单个 init，维度随新增物体递增，spatial 为 92/105/118/131/144/157）。
- **命名陷阱**：`_view_{h}_{v}_{scale}_{rot}_{vert}_initstate_{N}` 中的 `_initstate_{N}` **不是 init 索引**，而是被拼进机器人名（`env_wrapper.py:217-218`：`robots[0] + str(init_state)`）用于选择机械臂初始位姿变体。故 `_view_` 一类实际同时编码相机视角 + 机器人位姿 + 传感器噪声三个维度。
- **每 init 承载的变体数**：原 task 的 `pruned_init[0]` 承载 153–235 个变体（spatial）/ 176–281 个（libero_10）；`pruned_init[1..49]` 在官方协议下**完全闲置**；每个 newobj init 文件恰好 1 个变体。
- **无 clean 基线**：四个主 suite 的 benchmark 注册表中 CLEAN 计数为 **0**，未扰动对照须取自原版 LIBERO（init 相同故可精确配对）。

### 5.3 若将来复用，三种正当用法

1. **配微调 teacher**：LIBERO-Plus 发布了训练集，论文自身微调出 79.6% 的模型并放出权重（`Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata`）；teacher 在扰动分布上训过后前提即恢复。
2. **只取 teacher 不崩的维度**：须预注册筛选规则，否则构成挑读。
3. **边界测试**：「teacher 崩溃时 cache 会怎样」本身是可报告的负结果，但它回答的不是继承假说。

资源：<https://arxiv.org/html/2510.13626v1> ｜ <https://github.com/sylvestf/LIBERO-plus> ｜ <https://huggingface.co/datasets/Sylvest/LIBERO-plus>

---

## 6. 其他落选候选

| 候选 | 信息 | 未选原因 |
|---|---|---|
| **RoboCasa 原版**（RSS 2024）<br><https://arxiv.org/html/2406.02523> | 24 任务 Panda Omron 评测子集，120 厨房场景。**GR00T N1.6 零样本可跑**（`ROBOCASA_PANDA_OMRON` 在其预训练 embodiment set 内），24 任务均值 **66.22%**；N1.7 需微调，微调后 70.8% | teacher SR 更健康，但：① 2024-07 发布，对 ICLR 2027 显老，且 365 是其官方后继；② **pi0.5 无现成 checkpoint，须自行微调**，与 D5 冲突 |
| **THE COLOSSEUM**（RSS 2024）<br><https://arxiv.org/abs/2402.08191> ｜ <https://github.com/robot-colosseum/robot-colosseum> | 20 task（basketball、close box、close laptop、empty dishwasher、get ice、hockey、meat on grill、move hanger、wipe desk、open drawer、slide block、reach and drag、put money in safe、place wine、insert peg、stack cups、turn oven on、straighten rope、setup chess、scoop with spatula），**14 轴扰动**，sim↔真机相关性 R̄²=0.614 | 基于 RLBench（不同 API / 动作空间 / 关键帧控制），VLA 支持弱且无 pi0.5 checkpoint；且与 D6 同病——以打垮模型为设计目标 |
| LIBERO-PRO <https://arxiv.org/html/2510.03827v2> ｜ LIBERO-X <https://arxiv.org/html/2602.06556v1> | 同批评方向姊妹工作 | 同 D6 的混杂问题；可作 §1.1 的引用来源 |
| SimplerEnv | 视觉匹配 / variant aggregation | openpi 无官方 checkpoint；社区反馈 pi0.5 跑通困难（openpi issue #799） |
| MimicGen | 演示自动生成 | 需重训 teacher；无跨场景协议 |

---

## 7. 抽取机制：pi0.5 的确切做法（🟢 全部亲验源码）

判定任何第二 teacher 可行性的基准。

**模型前向拆三段**（`src/openpi/models_pytorch/pi0_pytorch.py:22` `Stage1Output`；调用点 `src/openpi/policies/policy.py:111-117`）：

```
_stage1_token_prep(observation)  → state, prefix_embs, prefix_pad_masks, prefix_att_2d_masks_4d, prefix_position_ids
_stage2_llm_backbone(prefix_embs, ...) → past_key_values
_stage3_action_expert(state, prefix_pad_masks, past_key_values, noise) → actions
```

**抽取点 = Stage1 输出的 `prefix_embs`** `[B, prefix_len, emb_dim]` bf16——多模态 token 序列**在进入 LLM transformer 之前**。

**token 布局固定可切**（`src/openpi/cache/components/key_builder.py:130-147`）：SigLIP 224px/patch14 ⇒ 每图 256 token，`[0:256) vision_0 | [256:512) vision_1 | [512:768) vision_2 | [768:) prompt`。

**`cp1_spatial_pool_16` 的 reducer**（`key_builder.py:192-213`）：256 token 还原成 **16×16 网格** → `adaptive_avg_pool2d` 到 4×4 → **16×2048 = 32768 维/图**。

**`robot_state`** 取自 `Stage1Output.state`；pi0.5 中该 state 已被离散成语言 token、不进模型，源码注释写明「kept solely for cache-key construction」。

### 7.1 四条判据

| # | 判据 | 层次 |
|---|---|---|
| a | vision patch token 序列，**保持空间网格** | 影响 key 质量 |
| b | 前向能拆段，**拿到 key 后能短路后续段** | **可行性生死线** |
| c | token 布局可确定地按相机切片 | 影响 key 质量 |
| d | PyTorch + 开放权重 | 可行性 |

🟢 **判据分两层**（依据 `docs/cache/migration.md:55-63, 104-136`）：**能否移植**只要求任一观测 embedding + 能提前退出（CP1 语义 =「观测编码完成、开始决策之前」，指南对 VLA 的推荐位置正是视觉编码器之后）；**能否沿用最优 key** 才需要 (a)(c)，不满足可退化到已有的 `_mean_pool_tokens` / `_max_pool_tokens`。

🟢 **移植工作量边界**：只需实现 **Interceptor** + **KeyBuilder**；Orchestrator / SearchStrategy / Judge / Gate / WritePolicy / Backend / YAML 全部模型无关、零改动。

---

## 8. GR00T 版本差异：架构不同，非权重不同（🟢 亲验）

本机克隆 `/home/weiland/projects/Isaac-GR00T`，已 `git fetch --unshallow --tags`，恢复全部 release tag。

| | **N1.5** | **N1.6** | **N1.7** |
|---|---|---|---|
| 模型类 / `model_type` | `GR00T_N1_5` / `gr00t_n1_5` | `Gr00tN1d6` | `Gr00tN1d7` |
| **VLM backbone** | Eagle-2.5-VL | Eagle-3-VL | **Qwen3-VL**（`nvidia/Cosmos-Reason2-2B`） |
| — LLM | Qwen3-1.7B，截断 **12** 层 | Qwen3-1.7B，截断 16 层 | Qwen3-VL-2B，截断 16 层 |
| — Vision | SigLIP-400M，27 层，d=1152，**224px** | SigLIP2-400M，27 层 | Qwen3-VL ViT，**24** blocks |
| — Connector | `mlp1` 1 层，**2.36 M** | `mlp1` 2 层 + pixel-shuffle，13.64 M | PatchMerger |
| **Action head** | DiT 16 层 + 4 层 VL adapter + FLARE | AlternateVLDiT 32 层，**移除** adapter | AlternateVLDiT 32 层 + **重新加回** adapter |
| **总参数 / 体积** | 2,724,163,520 / **5.45 GB** | 3,286,608,832 / 6.57 GB | 3,455,180,928 / 6.91 GB |
| state / action 维 | 64 / 32 | 128 / 128 | 132 / 132 |
| action horizon | 16 | 50 | 40 |
| 去噪步数 | 4 | 4 | 4 |
| **权重许可** | **非商用** | **非商用** | 可商用 |

**铁证**：三版本两两之间 backbone state-dict 键重叠 = **0**（0/585 vs 633、0/633 vs 494、0/585 vs 494）。N1.7 代码 `get_backbone_cls`（`gr00t/model/gr00t_n1d7/gr00t_n1d7.py:490-497`）对非 Qwen backbone 直接 `ValueError`；加载器 `setup.py:106-120` 对任何键不匹配 `RuntimeError`。

**三个坑**：
1. 🟢 **发布 checkpoint 与 repo dataclass 默认值不符**：repo 是 `select_layer=12` / DiT 16 层，发布的 `config.json` 是 **16 / 32**；`vl_self_attention_cfg` 不在 dataclass 里，只靠 `getattr` 兜底——**从零 config 训练会静默得到少 201M 参数的不同模型**。一律信 `config.json` + 张量形状。
2. 🟡 **HF model card 三版本都写 text encoder 是 "T5"——全错**，不要引用卡片。
3. 🟢 **Cosmos-Reason2-2B 是 gated repo**：每次加载 N1.7 checkpoint（哪怕全本地）都会先 `from_pretrained` 它（`gr00t/model/modules/qwen3_backbone.py:34-41`）。每台 serving 主机需接受许可 + 配 HF token。（选定 N1.5 后此坑不触发。）

---

## 9. 三模型抽取对照（🟢 亲验源码）

| | **pi0.5**（现役） | **GR00T N1.5**（选定） | GR00T N1.7 |
|---|---|---|---|
| 视觉塔 | SigLIP 224px/p14 | **SigLIP-400M 27 层 224px/p14** | Qwen3-VL ViT 24 blocks |
| **token / 图** | 256 | **256** | 64 |
| **网格** | 16×16 | **16×16** | 8×8 |
| connector | 投影进 LLM 空间 | **`mlp1` = 纯 `Linear(1152→2048)`，无 pixel-shuffle** | PatchMerger（合并 token） |
| tap 张量 | `prefix_embs` | **`vit_embeds` `[n_img, 256, 2048]`** | `image_embeds` |
| tap 位置 | `_stage1_token_prep` | **`extract_feature()`，`n1.5-release:gr00t/model/backbone/eagle2_hg_model/modeling_eagle2_5_vl.py:311`** | `get_image_features` |
| 按相机切 | prefix 内 offset | **dim 0 取行** `vit_embeds[b*V+v]` | 按 image index |
| 早退边界 | LLM 之前 | `modeling_eagle2_5_vl.py:261` 之前；模型级 `gr00t_n1.py:177-178` 两条独立语句 | `gr00t_n1d7.py:611/612` |
| HIT 跳过 | LLM + action expert | Qwen3×12L + vlln + 4L adapter + DiT 16×4=64 block | LLM + DiT |
| **key 维度** | **32768 / 相机** | **32768 / 相机（逐位相同）** | 16×d / 相机 |
| 相机数 | 任务相关 | LIBERO 2（512 tok）· **RoboCasa365 3（768 tok）** | — |

### 9.1 判据裁决：N1.5 与 N1.7 全部通过

**N1.5 三项独立佐证 pixel-shuffle 关闭**：`use_pixel_shuffle: false`（vendored config）；`preprocessor_config.json:39` / `processor_config.json:11` 声明 `tokens_per_tile: 256`；state dict 只有 `backbone.eagle_model.mlp1.0.{weight,bias}` 单 Linear，参数量 `1152×2048+2048 = 2,361,344` 与 2.36M 吻合（pixel-shuffle 变体应为 9,439,232）。

**NVIDIA 自己依赖该拆分**：`deployment_scripts/export_onnx.py:960` 把 ViT 单独导出为 ONNX（输出名 `vit_embeds`），`trt_model_forward.py:220` 导出后直接 `del ...vision_model`。

### 9.2 成本分布 🟡

N1.7 组件 benchmark（`scripts/deployment/README.md:151-183`，H100，4 去噪步，1 相机）：eager backbone 31.3 ms / action head 48.2 ms（尾部 56%）；torch.compile 30.4 / 12.0（**25%**）；TensorRT 8.8 / 12.3（44%）。

🟢 对照 pi0.5 实测（`exp/ablation_study/analysis/analysis.md:80-86`）：4090 Stage1 114.6 / Stage2+3 575.4（**尾部 83%**）；H200 17.7 / 96.3（**84%**）。

⚠ 该表 tap 点在 backbone 之后（ViT + LLM 都跑完），**非 pi0.5 CP1 的对应物**。tap 前移到视觉塔之后按参数量粗估回升至 ~88%，**但这是参数量口径、非时间**；ViT / LLM 拆分耗时未公布。🔴 见 §11。

---

## 10. Checkpoint 全景

### 10.1 选定方案所需（全部现成） 🟡

| 战场 | pi0.5 | GR00T N1.5 |
|---|---|---|
| **RoboCasa365** | 官方 `robocasa/robocasa365_checkpoints → pi05_pretrain_human300/multitask_learning/75000` ⚠ **JAX/Orbax 格式，需转换**（见下） | 官方 `robocasa/robocasa365_checkpoints → gr00t_n1-5/multitask_learning/checkpoint-120000`（7.59 GB，**PyTorch safetensors，直接可用**，`new_embodiment` tag，3 相机 `robot0_eye_in_hand` / `agentview_left` / `agentview_right`，256×256@20fps，state 16 维、action 12 维） |

⚠ **两个 teacher 的 checkpoint 格式不同（2026-08-15 亲验 HF API）**：
- **pi0.5 侧是 JAX/Orbax**：目录结构为 `params/`（内含 `ocdbt.process_0` / `array_metadatas` / `_METADATA` / `_sharding`）+ `assets/norm_stats.json` + `train_state/` + `_CHECKPOINT_METADATA` —— 典型 Orbax OCDBT 布局，**不是 safetensors**。而本项目是 PyTorch-only（WA §1），cache 绑定 PyTorch staged API ⇒ **必须先转换**。
- 转换工具**现成且支持 pi05**：`examples/convert_jax_model_to_pytorch.py`（588 行，第 25-26 行有 `pi05_droid` 示例，`:272-373` 有 pi05 专属的 Dense/adaptive-normalization 分支）。
- ⚠ **脆弱契约**：该脚本靠 **`if "pi05" in checkpoint_dir` 字符串判断模型类型**（`:294, 349, 373`）。RoboCasa365 的路径 `pi05_pretrain_human300/...` 恰好含 "pi05" 故能正确识别，**但转换时必须保留含 "pi05" 的目录名**，改名会静默走成 pi0 分支。
- ⇒ ~~**执行含义**：GR00T 路径比 pi0.5 少一步，Step 0b 可先跑 GR00T 侧。~~ **该结论已作废（见 §12-2）**：GR00T 虽省了转换，却缺 RoboCasa365 的 client 集成（n1.5 代码只有 GR1 Tabletop，N1.7 代码加载不了 N1.5 权重），反而是 pi0.5 有现成的 `robocasa-benchmark/openpi` 评测栈。**Step 0b 走 pi0.5。**
| LIBERO（可选第二战场） | 我们的 `pi05_libero` | 社区 `youliangtan/gr00t-n1.5-libero-{spatial,object,goal,long}-posttrain`（7.59 GB，`model_type: GR00T_N1_5`，完整可推理） |

### 10.2 兼容性裁决 🟢

`youliangtan/…-libero-spatial-posttrain`、`robocasa/…/gr00t_n1-5/…/checkpoint-120000`、`nvidia/GR00T-N1.5-3B` 三者的 `model_type`(`gr00t_n1_5`) / `select_layer`(12) / DiT `num_layers`(16) / 步数(4) / `vl_self_attention`(4 层) / `backbone_embedding_dim`(2048) / `action_horizon`·`action_dim`(16/32) **逐项一致**；state-dict 布局亦对齐（27 层 SigLIP / 12 层 Qwen3 / `mlp1.0.{weight,bias}` / 无 `eagle_linear.*` / `transformer_blocks.0…15`）⇒ **同一份 `n1.5-release` 代码可加载全部三者**。

### 10.3 微调路径（备选，当前不走） 🟡

官方配方齐备，`examples/finetune.sh` 支持单卡。LIBERO：`MAX_STEPS=20000 GLOBAL_BATCH_SIZE=640`。RoboCasa Panda Omron（24 任务）：`MAX_STEPS=60000 GLOBAL_BATCH_SIZE=512` → N1.7 微调后 70.8%。硬件：最低 1 GPU **40 GB+**，默认微调峰值 **<35 GB**，`--tune-llm`/`--tune-visual` 需 **80 GB+** ⇒ weilandserver 4090 48G ✅、ziyang10 H200 ✅、timan107 8×GTX1080 ❌。数据很小（`IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot` 仅 635 MB / 379 episodes）。**未采用理由见 D5。**

---

## 11. 实现约束

1. 🟢 **key 不能跨 checkpoint 迁移**。N1.5 的两个 checkpoint 均以 `tune_visual: true` 后训，SigLIP 塔与 `mlp1` 各自特化。RoboCasa365 库与 LIBERO 库**必须独立构建**。
2. 🔴 **RoboCasa365 三相机顺序未验证**，实际拼接顺序由仓库外 eval 的 `video_concat_order` 决定，建库前须实测。
3. 🟢 **不硬编码网格边长**，运行时从 tile 数（N1.5）/ `image_grid_thw`（N1.7）推导。
4. 🟢 **不用绝对 token 下标**切相机（tokenizer left-padding，B>1 时 offset 漂移）；N1.5 用 dim 0。
5. 🟢 **Eagle 动态 tiling 断言**：`max_dynamic_tiles: 12` + `use_thumbnail: true`，非方形输入会碎成多 tile（640×480 → 13 tile = 3328 token），网格假设失效。仓库内 data config 均强制 224×224 方形，但**必须加运行时断言 `pixel_values.shape[0] == n_cameras`**。
6. 🟢 **action horizon 不一致**：pi0.5 = 10，GR00T N1.5 = 16。cache 按 teacher 独立建库故不冲突，但评测的 `replan_steps` 与延迟口径须分别设定。
7. 🟢 **Init/场景池纪律**沿用 [`docs/iclr/tier_experiment_designs.md`](../docs/iclr/tier_experiment_designs.md)：拟合类活动（建库 / 学生蒸馏 / 阈值标定 / 超参与信号选择）只能在训练侧场景，held-out 场景只许测量。⚠ 同日审计发现现役 baseline 的权重搜索、threshold 分位标定、4b kinematic 校准**均在官方 pruned_init（测试集）上完成**；新 benchmark 立项必须避免复制该模式。
8. 🟢 **ACT 臂定位**：若新 benchmark 重蒸馏学生，ACT 臂须定位为**能力上界探针**（无语言通路 + per-task + 外部 oracle 任务标识三项须披露）；**SmolVLA 才是同类可比对象**。主结论不依赖 ACT（`hit_smolvla` 同样显著优于 `cache_baseline`，spatial +5.2pp / libero_10 +12.6pp）。
9. 🟢 **实验设计缺口**：X1–X14 无一张卡做「单模型多任务 ACT」对照——干净分离 per-task 效应的唯一办法，成本仅 2 训练 + 2×500 ep。

---

## 12. 立项前必须亲验（**全部 14 项已闭合**，2026-08-15 全部实机执行）

> **✅ 进度：14 项全部走完**。13 项已用实机证据闭合；第 5 项（每单元 episode 数 K）为**决策项**，已备实测成本表 + 统计功效表，只待 owner 一句裁定。
>
> **阻塞门 Step 0b（第 2 项）已以正式规模通过**（2026-08-16，180 ep）：teacher 在两个 held-out 厨房上 **SR_A 41.1% / SR_B 50.0%**，**held-out 侧反而更好**；P1 主判据 `SR_B` 的 95% 下界 **39.9% ≫ 20% 崩溃线** ⇒ **方案不作废，可进入 §4 Code 阶段**。
>
> ⚠ 同时产出 per-task 可用性清单：**4/18 任务 teacher 完全不工作（U0）**，其处置有一个循环依赖待 owner 裁定（见第 2 项末尾）。
>
> 本节的 🔴/🟡/🟢/✅ 标记均**逐项对应一次真实执行**，不是文献推断；各项内已附命令、实测数字与踩坑记录，可直接复现。

**准入门（阻塞，按序）**
1. ✅ **已闭合 — Step 0a `bench_speed.py` 实测**（结果见 §4.6.6）。实测 reset **3.69–8.96 s**、fps **14.85–28.17**，**比原假设的 9.5 s / 25.2 fps 更好**，但**场景间差 2.4 倍**。附带发现：`--n_envs>1` 的 `SubprocVectorEnv` 路径不可用（`tianshou/env/worker/subproc.py:114` 按 gymnasium 新 API 期待 `reset()` 返回 `(obs, info)`，而 robocasa 是 robosuite 旧 API 只返回 `obs` ⇒ `ValueError: too many values to unpack`）。**不影响方案**——实际评测走 openpi fork 的单进程 `main.py`，并行靠多进程分片（conductor 农场），从不使用 SubprocVectorEnv。

2. ✅✅ **已闭合（2026-08-16，正式准入门 180 ep，P1 PASS）** — **Step 0b：teacher 在两个 held-out 厨房间的 SR 落差**（Atomic split，配置见 §4.6.3）。

   **最终结果（18 任务 × 2 场景 × 5 ep = 180 ep，墙钟 4.59 h）**：**SR_A = 37/90 = 41.1%、SR_B = 45/90 = 50.0%**，**scene B 反而更好**，gap = **−8.9pp**（95% CI [−20.3, +2.5]，含 0）。**P1 主判据**：`SR_B` 的 Wilson 95% **下界 39.9% ≫ 崩溃线 20%** ⇒ **PASS，方案不作废**。**P2**：U0=4 / U1=2 / U2=12 —— 18 个任务里 **4 个（22%）teacher 在两场景均完全不工作**。

   ⚠ 早先的 36 ep 诊断跑给出 gap = **+5.6pp**（方向相反），两次 CI 均含 0 故不矛盾，但**点估计符号翻转**——这印证了「不以 gap 为准入判据」的预注册决定。**引用 gap 一律以 180 ep 为准。** 完整 18 行表、反转分析与 U0 处置方案见本项末尾。

   ⚠⚠ **概念修正（2026-08-15 溯源三处源码，本项标题原写「seen vs held-out 厨房」是错的）**：**RoboCasa365 的 seen/unseen 划在「任务」维度，不在「场景」维度**。三条证据：
   - `dataset_registry.DATASET_SOUP_REGISTRY["pretrain_human300"]` 的 300 条路径全部指向 `datasets/v1.0/**pretrain**/atomic/…`，且 **`atomic_seen` 的 18 个任务 18/18 都出现在其中** ⇒ "seen" = 训练时见过该**任务**。
   - `env_utils.py:86-104`：`split="target"` ⇒ `layout_and_style_ids = list(zip(range(1,11), range(1,11)))`，即**只有对角线 10 个场景** `(1,1)…(10,10)`；`split="pretrain"` ⇒ `layout_ids = style_ids = -2`。
   - `scene_registry.py:78,87,168`：`TRAIN = -2` 且 `-2: list(range(11, 61))` ⇒ **训练场景 = layout/style 11–60**。

   ⇒ **训练场景 (11–60) 与评测场景 (1–10) 零交集**，我们选的 **(1,1) 与 (7,7) 对 teacher 都是完全未见过的厨房**。

   **这对本方案是有利的，不是问题**：A 与 B 对 teacher **处境对称**，故二者 SR 差异只反映**场景难度**，不掺入「训练过 vs 没训练过」的能力落差 ⇒ 后续 cache 从 A 迁到 B 的效果，不会被 teacher 自身的能力衰减污染（这正是 D6 弃用 LIBERO-Plus 时要规避的混淆）。
   **准入判据据此改写**：不再是「held-out 是否显著差于 seen」，而是 **「teacher 在 A 和 B 上都要足够能干活，且两者相近」**。
   ⚠ 附带推论：**RoboCasa365 内无法构造「teacher 见过的场景」作为对照**——除非主动拿 pretrain 场景 (11–60) 当 scene A，那会引入训练场景优势，与上述对称性设计冲突。**建议维持双 held-out 对称设计**，此点提请 owner 知悉（§13）。

   ⚠ **路径修正（2026-08-15 实查，与本文早先判断相反）**：**应走 pi0.5 而非 GR00T**。
   - **GR00T 侧被结构性阻塞**：n1.5-release 的 `examples/RoboCasa` 面向 **RoboCasa Tabletop / GR1 机器人**（`gr1_unified/…GR1ArmsAndWaistFourierHands_Env`），**无 RoboCasa365 PandaOmron 集成**；而含 robocasa365 gym wrapper 的 N1.7 代码库**加载不了 N1.5 checkpoint**（§8：backbone 键 0 重叠 + `get_backbone_cls` 直接 raise）⇒ 需自行移植 client 适配层。（§10.1 结尾「GR00T 路径少一步」的结论据此作废。）
   - **pi0.5 侧路径完整**：`robocasa-benchmark/openpi` 的 `examples/robocasa/main.py` 本就是为 RoboCasa365 写的（§4.6.1），只需一次 JAX→PyTorch 转换（工具现成，§10.1）。
   🟢 **执行链已全部打通（2026-08-15，端到端 smoke 通过）**，逐环踩坑记录：
   1. **下载**：`params/`+`assets/`+`_CHECKPOINT_METADATA`，排除 `train_state/`（省约 30 GB）⇒ 11.9 GB / 20 文件。首轮 19/20 后遇 `LocalEntryNotFoundError`，重试即补齐（`snapshot_download` 幂等）。
   2. **转换**：`convert_jax_model_to_pytorch.py --config-name pi05_aloha --checkpoint_dir <含 "pi05" 的路径>`。⚠ 本地版需 `--config-name`（与 fork docstring 不同）；**`pi05_aloha` 是唯一结构匹配的本地 config**（`Pi0Config(pi05=True)` 纯默认 ⇒ action_dim=32 / action_horizon=50 / max_token_len 自动 200 / discrete_state_input=True，与 RoboCasa365 的 `pi05_pretrain_human300` 等价）；`pi05_libero` **不可用**（被覆盖成 action_horizon=10、discrete_state_input=False）。产物 6.9 GB。
   3. **起 server**：写独立脚本而非改 `config.py`（fork 的 robocasa config 在模块顶层 import robocasa，会破坏主 venv）。用 `SimpleDataConfig` + fork 的 `RobocasaInputs/Outputs`（`robocasa_policy.py` 127 行搬运）。⚠ **必须显式关掉 quantile 归一化**：本地 `config.py:187` 对 pi05 自动置 `use_quantile_norm=True`，而 fork 无此行，该 checkpoint 用 mean/std 训练（其 `norm_stats.json` 的 `q01`/`q99` 均为 **null**），不关会在 `transforms.py:458` raise。⚠ 端口避开 8000（既有 srv0 占用），用 8010。
   4. **client**：孤岛 A 内装 `openpi_client` 必须 **`--no-deps`** —— 其 pyproject 声明 `numpy>=1.22.4,<2.0.0`，直装会把孤岛 A 的 numpy 从 2.2.5 降级到 1.26.4，触发 robocasa 的 import 断言与 scipy 的 `np.long` 崩溃（已发生并修复）。
   5. **action 格式**：env 需要 **dict** 而非 ndarray（`gym_wrapper.py:122` 的 `type(input_action)()`）。**必须用官方 `robocasa.utils.env_utils.convert_action`**，其映射为 `eef_pos[0:3] / eef_rot[3:6] / gripper[6:7] / base_motion[7:11] / control_mode[11:12]` —— **与 GR00T metadata 的字母序不同**，不可自行推断。
   6. **obs 口径**：state 拼接顺序须逐字节照抄 fork `main.py:133-139`（eef_pos 3 → eef_rot 4 → base_pos 3 → base_rot 4 → gripper_qpos 2 = 16 维）；图像 `resize_with_pad(224)` → `convert_to_uint8`。⚠ fork 注释写着 "IMPORTANT: rotate 180 degrees to match train preprocessing" 但**代码并未旋转**——已查明是遗留注释：**翻转由 gym wrapper 代劳**（`robocasa/wrappers/gym_wrapper.py:271-273`：`# Image are in (H, W, C), flip it upside down` → `np.copy(img[::-1, :, :])`）。故直接消费 gym obs 即为正确方向，不可再翻。
   7. **server 为单连接模式**：`s0b` 长跑期间无法并发接第二个 client 做诊断探针（`1013 try again later / Only one client at a time`）。需要并行诊断时得另起一个端口的 server 实例。

   🟢 **smoke 结果**：`TurnOnMicrowave` 在 scene(1,1) 与 (7,7) 各 1 ep 全程无错跑满 450 步，wall **82.4 s / 106.1 s** —— 与 §4.6.6 的场景成本差异独立吻合。模型输出 `(50, 12)` 动作，维度正确。
   - ⚠ 两 ep 均失败（SR=0），但 n=1 无判别力（atomic 平均 SR 仅 39.6%）。遂跑 6 任务 × 2 场景 × 3 ep = 36 ep 以区分「接线有误」与「正常失败」——**结果见下**。

   ---

   ### ✅ **Step 0b 诊断跑完成（2026-08-15，36 ep，wall 1.03 h）——准入门初判 PASS**

   配置：`N_TRIALS=3  SCENE_A=(1,1)  SCENE_B=(7,7)  REPLAN=5  split=None  obj_instance_split=target`，teacher = pi0.5 `pi05_pretrain_human300/multitask_learning/75000`（JAX→PyTorch），server 8010 @weilandserver 4090。

   | 任务 | scene A (1,1) | scene B (7,7) | 配对 gap | wall A | wall B |
   |---|---|---|---:|---:|---:|
   | `TurnOnMicrowave` | 0/3 | 0/3 | 0 | 212.9 s | 306.4 s |
   | `CloseFridge` | 3/3 | 2/3 | −33.3pp | 338.3 s | 365.0 s |
   | `OpenDrawer` | 2/3 | 3/3 | +33.3pp | 197.0 s | 193.5 s |
   | `TurnOnSinkFaucet` | 2/3 | 3/3 | +33.3pp | 249.7 s | 322.8 s |
   | `CoffeeSetupMug` | 1/3 | 0/3 | −33.3pp | 228.2 s | 376.9 s |
   | `CloseBlenderLid` | 1/3 | 0/3 | −33.3pp | 403.6 s | 517.5 s |
   | **合计** | **9/18 = 50.0%** | **8/18 = 44.4%** | **+5.6pp** | 1629.7 s | 2082.1 s |

   **三条判定**：

   1. ✅ **接线正确（本次跑的首要目的）**。两侧 SR 均**远高于 0**，且**高于官方 leaderboard 的 atomic 均值 39.6%**（本次 6 任务偏易/中等，非全 18 任务，故不可直接对榜）。早先 `TurnOnMicrowave` 连续 0/6 是**该任务本身难**，不是 wiring bug —— `CloseFridge` 首个 ep 即成功（`OK t=784/900`）当场证伪了接线错误假设。
   2. ✅ **teacher 在两个 held-out 厨房上都能干活**（44.4% / 50.0%，均属可用区间）⇒ **满足 D4「teacher 能工作即可」与 D6「换场景后 teacher 不可崩」**。这正是 LIBERO-Plus 未能通过、进而被弃用的那一关。
   3. ✅ **无证据表明 scene B 劣于 scene A**。配对 t 检验（6 个任务的 gap 分别为 0 / −33.3 / +33.3 / +33.3 / −33.3 / −33.3 pp）：**mean = +5.6pp，SD = 32.7，SE = 13.4，95% CI = [−29, +40] pp**，**区间含 0**。

   ⚠ **本次样本量只支持定性结论**：每 (task,scene) 单元仅 n=3，任务数 6/18。本次的定位是**诊断跑 + 准入门初判**，二者结论一致：**方案不作废，可继续推进**。

   ### 正式准入门（180 ep）的作用 —— 以及它**做不到**的事

   ⚠⚠ **先撤销一处错误陈述**：本文档早先写「正式准入门跑完后 CI 可收窄到可判定 10pp 门槛的量级」——**错**。实算如下（配对设计，p=0.47）：

   | 设计 | 总 ep | **gap 的 95% CI** |
   |---|---:|---:|
   | 已完成的诊断跑（6 任务 × 3） | 36 | ±32.6pp |
   | **正式准入门（18 任务 × 5）** | **180** | **±14.6pp** |
   | 18 任务 × 10 | 360 | ±10.3pp |
   | 真正能判 10pp 门槛所需（K=43） | **1548** | ±5.0pp |

   ⇒ **180 ep 判不了 10pp 门槛**（只到 ±14.6pp），而判得了需要 1548 ep ≈ 42 h，性价比极差。**准入门的价值不在统计精度**，在下面三条：

   1. **消除任务选择偏差（最主要）** —— 诊断跑的 6 个任务是**人为挑选**的（按「覆盖不同技能 + 相对常见」），非随机抽样；剩余 12 个任务完全未测。要主张「teacher 在整个 `atomic_seen` 上换场景不崩」，必须 18/18 全测。当前结论严格讲只覆盖那 6 个任务。
   2. **产出 per-task 可用性清单（对正式实验最实用）** —— `TurnOnMicrowave` 在两场景均 0/3。**这类任务对 cache 实验没有信息量**：teacher 本身就不成功，命中与否都谈不上「加速一次成功的推理」，留在正式实验里只会稀释效果并贡献纯噪声。诊断跑只在 6 个任务里逮到 1 个，180 ep 能给出完整清单，使正式实验的任务集有据可依（排除或单独标注）。⚠ n=5 时单任务 SR 粒度为 20%，**识别「0% 任务」够用，区分「20% vs 40%」不够**。
   3. **数据可复用，非纯开销** —— 这 180 ep 是 teacher-only、无 cache 的 rollout，正式实验的对照侧本就需要这批数据。

   **判据须随之改写**：既然 gap 判不准，准入门应改以 **SR_B 的绝对水平**为判据 —— 这也正是 D6 真正要防的（teacher 崩不崩），且收敛快得多：

   | | **SR_B 的 95% CI** |
   |---|---|
   | 36 ep 诊断跑 | [24.6%, 66.3%]（±20.9pp） |
   | **180 ep 准入门** | **[34.6%, 54.7%]（±10.1pp）** |

   180 ep 的下界 **34.6%** 已接近官方 atomic 均值 **39.6%**，足以支撑「teacher 在 held-out 厨房的表现与官方报告的整体水平相当」——**这个判据 180 ep 是够的**。

   **gap 的精确判定不应由准入门承担**：正式跨场景实验有 18 任务 × 5 场景 = **90 个配对单元**，K=10 时 gap 的 95% CI 即 **±4.5pp**（比准入门强 3 倍）。gap 本就是那里的产出，不是这里的。

   ---

   ### 🔒 预注册分析口径（**写于 2026-08-15 21:5x，180 ep 跑启动后、任何结果产出之前**）

   > 之所以预注册：这次跑会第一次给出 18 个任务的完整 per-task 表，而「哪些任务排除出正式实验」这个决定**极易被已看到的数据反向影响**（挑掉不利任务 = 选择偏差）。判据必须先定死。本节在结果出来后**不得修改**，只能在其下追加「实际观测」。

   **P1 — 准入门主判据（决定方案作废与否）**：`SR_B`（scene B 全 18 任务合并，n=90）的 **Wilson 95% CI 下界 > 20%** ⇒ PASS。
   - 20% 这条崩溃线的依据：teacher SR 过低时，「加 cache 后 SR 是否退化」这一核心量将失去可测性（基线太小，相对变化淹没在噪声里）。D6 弃用 LIBERO-Plus 时 π₀ 是 94.2%→15.8%，即落在此线之下。
   - ⚠ **不以 gap 为主判据**（180 ep 的 gap CI 仅 ±14.6pp，无判别力）。gap 仅作描述性报告。

   **P2 — per-task 可用性分类（决定正式实验的任务集）**，按 (SR_A, SR_B) 分三类：

   | 类别 | 判据（K=5，每格 0–5 成功） | 处置 |
   |---|---|---|
   | **U0 无信息量** | **两个场景 SR 均 = 0/5** | **排除出正式实验**，并在论文中列明 |
   | **U1 单侧退化** | 一侧 = 0/5、另一侧 > 0 | **保留**，但单独标注；这是最能反映跨场景脆弱性的样本，不可丢 |
   | **U2 正常** | 两侧均 > 0 | 保留 |

   - **排除 U0 的理由**：teacher 自身从不成功的任务，cache 命中与否都谈不上「加速一次成功的推理」，留在集合里只稀释效果并贡献纯噪声。**这不是为了让数字好看** —— U0 任务在两侧同时为 0，对 A/B 对比是中性的，排除它既不利也不弊于跨场景结论。
   - ⚠ **K=5 的粒度限制（先行声明）**：单任务 SR 只能取 {0, 20, 40, 60, 80, 100}%。该分类**只能可靠识别 U0**（真 SR 若为 10%，5 次全败的概率 59%，故 U0 判定会有假阳性）。**因此 U0 的最终排除须在正式实验的 K=10 数据上复核**，本次只作候选清单。
   - 已知的 U0 候选（来自 36 ep 诊断跑）：`TurnOnMicrowave`（A 0/3、B 0/3）。

   **P3 — 报告口径**：per-task 表须**全 18 行完整列出**（含被归为 U0 的），不得只报保留任务的汇总。任何在本节之外新增的排除理由，须在 log 中显式记录为「事后决定」。

   ---

   ### ✅✅ 正式准入门完成（2026-08-16 凌晨，**180 ep**，墙钟 **4.59 h**）—— **P1 PASS**

   配置同上，唯二变化：`N_TRIALS=5`、任务集取满 `TASK_SET_REGISTRY["atomic_seen"]` 全 18 个。分析由 `analyze_step0b.py` 按上节预注册口径产出，**判据未作任何事后修改**。

   **P3 全量 per-task 表（18 行，含全部 U0）**：

   | 任务 | A (1,1) | B (7,7) | gap | 类别 |
   |---|---:|---:|---:|:--|
   | `CloseToasterOvenDoor` | 0/5 | 0/5 | 0 | **U0** |
   | `NavigateKitchen` | 0/5 | 0/5 | 0 | **U0** |
   | `TurnOnElectricKettle` | 0/5 | 0/5 | 0 | **U0** |
   | `TurnOnMicrowave` | 0/5 | 0/5 | 0 | **U0** |
   | `CloseBlenderLid` | 2/5 | 0/5 | +40.0pp | **U1** |
   | `TurnOffStove` | 0/5 | 2/5 | −40.0pp | **U1** |
   | `PickPlaceCounterToStove` | 5/5 | 5/5 | 0 | U2 |
   | `OpenCabinet` | 4/5 | 4/5 | 0 | U2 |
   | `CloseFridge` | 4/5 | 3/5 | +20.0pp | U2 |
   | `PickPlaceSinkToCounter` | 3/5 | 4/5 | −20.0pp | U2 |
   | `PickPlaceToasterToCounter` | 3/5 | 4/5 | −20.0pp | U2 |
   | `TurnOnSinkFaucet` | 3/5 | 4/5 | −20.0pp | U2 |
   | `CoffeeSetupMug` | 3/5 | 3/5 | 0 | U2 |
   | `PickPlaceDrawerToCounter` | 3/5 | 5/5 | −40.0pp | U2 |
   | `OpenStandMixerHead` | 2/5 | 3/5 | −20.0pp | U2 |
   | `PickPlaceCounterToCabinet` | 2/5 | 2/5 | 0 | U2 |
   | `OpenDrawer` | 2/5 | 5/5 | −60.0pp | U2 |
   | `SlideDishwasherRack` | 1/5 | 1/5 | 0 | U2 |
   | **合计** | **37/90 = 41.1%** | **45/90 = 50.0%** | **−8.9pp** | |

   **P1 判定（主判据）**：`SR_B` = 45/90 = 50.0%，Wilson 95% CI **[39.9%, 60.1%]**，**下界 39.9% ≫ 崩溃线 20%** ⇒ **PASS，余量充足**（对比 36 ep 时下界仅 0.246，勉强过线）。且下界 39.9% 恰好持平官方 leaderboard 的 atomic 均值 39.6%。

   **⚠ 一个必须写下来的反转：gap 的符号翻了。**

   | | 点估计 | 95% CI |
   |---|---:|---|
   | 36 ep 诊断跑 | **+5.6pp**（A 更好） | [−28.8, +40.0] |
   | **180 ep 正式准入门** | **−8.9pp**（**B 更好**） | **[−20.3, +2.5]** |

   两次的 CI **都包含 0**，故严格说二者不矛盾——都只能得出「无显著差异」。但**点估计从 +5.6 翻到 −8.9**，这正是小样本点估计不可信的教科书示例，也事后验证了本节「不以 gap 为准入判据」这一预注册决定是对的：若当初拿 36 ep 的 +5.6pp 当结论，会得到与更大样本相反的方向。**本文档中任何引用 gap 符号的结论，一律以 180 ep 为准。**

   **对实验设计的实质意义**：scene B 不但没退化，反而略优 ⇒ **「teacher 换到 held-out 厨房会崩」这一风险被排除得比预期更彻底**。后续 cache 从 A 迁到 B 时，若观察到性能下降，**不能归因于 teacher 在 B 上更弱**（它更强），只能归因于索引迁移本身——这正是本实验想要的干净归因，也再次印证 §12-2 的双 held-out 对称设计（见 §4.6.3）。

   **P2 分类结果**：**U0 = 4**（`CloseToasterOvenDoor` / `NavigateKitchen` / `TurnOnElectricKettle` / `TurnOnMicrowave`）、**U1 = 2**、**U2 = 12**。
   ⇒ **18 个 atomic_seen 任务里有 4 个（22%）teacher 在两个场景上完全不工作**。这类任务对 cache 实验无信息量（teacher 从不成功，谈不上「加速一次成功的推理」）。

   ⚠ **U0 的处置存在一个循环依赖，须 owner 裁定**：预注册要求「U0 的最终排除须在正式实验的 K=10 数据上复核」，但若正式实验直接排除它们，就永远没有 K=10 数据可复核。两个方案：

   | 方案 | 做法 | 成本（K=10，单进程） | 风险 |
   |---|---|---:|---|
   | **(a) 全保留、分层报告**（**建议**） | 正式实验仍跑 18 任务，分析时按 U0/U1/U2 分层 | 900 ep/侧 ≈ **24.2 h/侧** | 无选择偏差；U0 自带复核数据 |
   | (b) 先复核再排除 | 先用 K=10 单独复跑 4 个 U0 候选（4×2×10 = 80 ep ≈ 2.2 h）确认后再排除 | 80 ep + 700 ep/侧 ≈ **2.2 + 18.8 h/侧** | 省 22%，但排除动作本身基于 K=5 判定，仍有假阳性余地（单任务真 SR=10% 时 5 连败概率 59%） |

   🟢 **预算模型二次校准**：本次实测墙钟 **4.59 h**，§12-5 表的估算是 **4.9 h**，**误差 6%** ⇒ 该预算表（96.9 s/ep 口径）可信，正式实验的 24.2 h/侧 可直接用于排期。

   🟢 **附带产出的成本标定**（喂给 §12-5 预算表）：scene A **90.5 s/ep**、scene B **115.7 s/ep**（B/A = **1.28×**，rollout 段的场景成本差小于 reset 段的 2.4×，因推理耗时不随场景变化）。

   ### ✅✅ 第二 teacher（GR00T N1.5）正式准入门完成（2026-08-16，**180 ep**，墙钟 **2.90 h**）—— **P1 PASS**

   配置与 pi0.5 那次**四项全同**（实证而非假定，两侧运行日志首行逐字比对）：`tasks=18 A=(1,1) B=(7,7) trials=5 replan=5`。⚠ 这一条是必查项——§4.6.1 早已指出 GR00T fork 默认 `n_action_steps=16` 而 openpi fork 是 `replan_steps=5`，同一 episode 差 3.2× 推理调用；节拍不同则两个 teacher 的 SR 根本不可比。本轮 client 按 replan=5 实现，故无此混淆；**换栈或改参数时必须重查**。

   **P3 全量 per-task 表（18 行，含全部 U0）**：

   | 任务 | GR00T A | GR00T B | gap | GR00T 类 | （pi0.5 类） |
   |---|---:|---:|---:|---|---|
   | `PickPlaceSinkToCounter` | 5/5 | 0/5 | **+100.0pp** | **U1** | U2 |
   | `PickPlaceCounterToCabinet` | 4/5 | 4/5 | 0 | U2 | U2 |
   | `PickPlaceCounterToStove` | 4/5 | 4/5 | 0 | U2 | U2 |
   | `SlideDishwasherRack` | 3/5 | 3/5 | 0 | U2 | U2 |
   | `CoffeeSetupMug` | 3/5 | 2/5 | +20.0pp | U2 | U2 |
   | `OpenCabinet` | 2/5 | 3/5 | −20.0pp | U2 | U2 |
   | `PickPlaceDrawerToCounter` | 1/5 | 1/5 | 0 | U2 | U2 |
   | `TurnOnElectricKettle` | 1/5 | 2/5 | −20.0pp | **U2** | **U0** |
   | `OpenDrawer` | 1/5 | 4/5 | −60.0pp | U2 | U2 |
   | `OpenStandMixerHead` | 0/5 | 1/5 | −20.0pp | **U1** | U2 |
   | `TurnOnSinkFaucet` | 0/5 | 3/5 | −60.0pp | **U1** | U2 |
   | `PickPlaceToasterToCounter` | 0/5 | 4/5 | −80.0pp | **U1** | U2 |
   | `CloseBlenderLid` | 0/5 | 0/5 | 0 | **U0** | U1 |
   | `CloseFridge` | 0/5 | 0/5 | 0 | **U0** | **U2** |
   | `CloseToasterOvenDoor` | 0/5 | 0/5 | 0 | U0 | U0 |
   | `NavigateKitchen` | 0/5 | 0/5 | 0 | U0 | U0 |
   | `TurnOffStove` | 0/5 | 0/5 | 0 | **U0** | U1 |
   | `TurnOnMicrowave` | 0/5 | 0/5 | 0 | U0 | U0 |

   **P1 判定**：`SR_B` = 31/90 = **34.4%**，Wilson 95% CI **[25.4%, 44.7%]**，**下界 25.4% > 20% 崩溃线** ⇒ **PASS**。`SR_A` = 24/90 = **26.7%**。

   🟢 **最重要的一致性发现 —— 两个 teacher 的 gap 同号同量级**：pi0.5 **−8.9pp**、GR00T **−7.8pp**（95% CI [−26.5, +11.0]，含 0），**都是 held-out 的 scene B 略优于 scene A**。两个架构完全不同的 VLA 给出同方向结果，把「held-out 厨房不更难」从单点观测抬成了**跨架构可重复的结论**，§12-2 双 held-out 对称设计的归因干净性因此更稳。

   ⚠ **一个必须记录的落差**：GR00T 的绝对 SR（26.7%/34.4%）明显低于其官方 leaderboard 的 Atomic 均值 **43.0%**，而 pi0.5 那侧（41.1%/50.0%）**复现了**自己的 39.6%。同一套 harness 一侧复现、另一侧不复现，值得存疑；但 P1 只要求 teacher 「能工作」（D4），且 GR00T 在 11 个任务上确有成功，故**不影响准入结论**。

   ### 🟢 checkpoint 来源核查（2026-08-16，回应上条落差）—— **两个 teacher 同源同范式，选择正确**

   两个 teacher 均取自**同一官方仓库、同一 commit**：`robocasa/robocasa365_checkpoints`（author = `robocasa`，即 benchmark 作者；`gated=False`；commit **`c484448aba1a9b60a04c9b0ca117241518ea69f3`**，与本地 `.cache/huggingface/download/*.metadata` 记录一致）。

   **仓库完整结构**（HF API 枚举 6397 个文件后按 checkpoint 层级归并）：

   ```
   gr00t_n1-5/
     multitask_learning/checkpoint-120000                                  ← 本实验使用
     foundation_model_learning/pretraining/checkpoint-80000
     foundation_model_learning/target_only/{atomic_seen,composite_seen,composite_unseen}/checkpoint-60000
     foundation_model_learning/target_posttraining/{同上三个}/checkpoint-60000
     lifelong_learning/phase1..4/checkpoint-{100000,60000,60000,60000}
   pi0/pi0_robocasa_pretrain_human300/multitask_learning/75000
   pi05_pretrain_human300/multitask_learning/75000                         ← 本实验使用
   ```

   🟢 **结论 1：两个 teacher 都在 `multitask_learning` 这一支** ⇒ 同一训练范式，**「两个 teacher 的 gap 同号同量级」这一发现不存在 checkpoint 类别混淆**。且 pi0.5 侧**不存在** `target_*` 对应物，想错配也无从配起。

   ⚠⚠ **勘误（owner 指出，2026-08-16）**：本节初稿写「`target_posttraining` 会摧毁 held-out 前提、整个跨场景命题失效」——**该说法是错的，已撤回**。本实验的自变量是**「cache 库建在哪个场景」**：库在 A 建、在 B 评，**A≠B 由实验设计本身保证**，与 teacher 训练过哪些场景无关。teacher 只是固定 oracle（产生 cache 条目 / miss 兜底 / 提供抽 key 的表征），它见过 target 场景**不会让「A 建的库能否在 B 用」这个测量发生泄漏**。把 teacher 的训练分布当成 cache 实验的污染源，是把 D6 的教训（teacher 崩溃会淹没索引效应）误推成了「teacher 不许见过评测场景」。

   **结论 2（修正后）：维持 `multitask_learning` 的真实理由是「两个 teacher 必须同范式」，不是污染。**
   - 仓库中 `foundation_model_learning/target_*` 一族**只存在于 `gr00t_n1-5/` 之下，pi0.5 侧无任何对应物**（见上方结构图）。⇒ 若单独把 GR00T 换成 target-posttrained，两个 teacher 就不再同范式，「两 teacher 结论一致」这一交叉验证的说服力会被「checkpoint 范式不同」这一替代解释削弱。
   - 次要考量（**较弱，但应记录**）：key 抽自 teacher 内部表征；一个在**全部 10 个 target 场景**上训练过的 teacher，其特征在这些场景间可能天然更不变，从而使「跨场景 cache 迁移成立」这一正结论的**普适性**打折扣。用 pretrain-only 的 teacher 若仍观察到迁移，是**更强**的结果。
   - 🔶 **反向考量（对本方案不利，须一并记录）**：target-posttrained teacher 绝对更强 ⇒ 成功 episode 更多 ⇒ 可用 cache 条目更多、U0 更少、任务集更大、统计功效更高。GR00T 现有 **6 个 U0**（比 pi0.5 多 2 个）直接压缩了交集。⇒ **「换不换」是一个 owner 可以重新权衡的真实选项，本节不再单方面否决**。若要换，代价是须为 pi0.5 找/训一个同范式对应物，否则失去同范式匹配。

   ### 🔴 43.0% 落差已查明 —— 我方「checkpoint 族」假说**被推翻**，真因是**评测场景不同**（2026-08-16，查论文原文）

   ⚠ 本节曾推测「43.0% 出自 `target_posttraining`，故与我方 `multitask_learning` 非同一口径」。**查论文后该推测证伪**，真相不同且更重要。

   论文 <https://arxiv.org/html/2603.04356v1> 三个设定分列三张表：

   | 表 | 设定 | Atomic | 训练场景 | **评测场景** |
   |---|---|---:|---|---|
   | **Table 1** | **multi-task learning** | pi0.5 **39.6%** / GR00T **43.0%** | pretraining（50×50=2500） | **pretraining kitchen scenes** |
   | Table 2 | foundation model training（= `target_posttraining`） | **68.5%** | pretraining → 在 target 上 fine-tune | **target kitchens**（10 个） |
   | Table 3 | lifelong learning | 41.5%（phase1） | — | — |

   原文：「We evaluate in the **pretraining kitchen scenes** for each task」（Table 1 设定）；「After training, we evaluate the model across the 50 target tasks in the **target kitchens**」（Table 2 设定）。

   🟢 **结论 1：39.6% / 43.0% 确实出自 `multitask_learning`** ⇒ 我方两个 teacher 的 checkpoint 选择**与该表一一对应，没选错**。
   🔴 **结论 2：但那两个数字是在 pretraining 场景（模型训练过的厨房）里测的，而我方在 target 场景 (1,1)/(7,7)（模型没训过的厨房）里测** ⇒ **两组数字从一开始就不可直接比**。我方是分布外、论文是分布内，GR00T 从 43.0%→34.4% 是**预期中的分布外退化，不是缺陷**。
   ⇒ **§12-2「GR00T 绝对 SR 偏低」这一存疑项就此关闭**，无须再排查 replan/场景数等候选。
   ⚠ 同时说明 pi0.5 的 41.1%/50.0% **高于**其 39.6**%** 并非「超常发挥」：口径不同，不构成复现或不复现的证据。**本文此前「pi0.5 复现了公开数字、GR00T 没复现」的说法一并作废** —— 两者都没有、也无法用本实验的场景配置复现 Table 1。

   ### ⚠ 由此产生的新权衡：换用 `target_posttraining` 的代价（2026-08-16，owner 已裁定换用并重跑）

   论文明确：target post-training **在 target 场景上训练**，而 target = RoboCasa 原有的 10 个固定 (layout, style) —— **包含我方的 A=(1,1) 与 B=(7,7)**。

   🟢 **不构成对 cache 测量的污染**（owner 已指出，本文已在上方勘误）：自变量是「库建在哪个场景」，库在 A 建、在 B 评，A≠B 由设计保证；teacher 训练分布不参与该测量。且 teacher 训过全部 10 个 target 场景 ⇒ **A 与 B 仍然对称**，只是对称在「都见过」而非「都没见过」。

   ⚠⚠ **勘误二（owner 第二次指出同一概念错误，2026-08-16）**：本节曾写「target-posttrained 的表征在 A/B 上被专门优化过，会削弱正结论的普适性」——**该说法站不住，已撤回**。本实验的假说**就是**「cache/key 抽自 VLA 内部表征，故跨场景可用」；若迁移成功正是因为该表征跨场景不变，**那是被验证的机制本身，不是需要扣分的混淆**。以此为由给结论打折，等于说「机制生效了所以结果不算数」。

   🔴 **须固化的教训（同一错误已犯两次）**：本实验的自变量是**「cache 库建在哪个场景」**，teacher 是**固定底座不是自变量**。teacher 只需满足两条：① **足够能干**（有成功 episode 才有东西可缓存 —— 这正是准入门存在的理由）；② **两臂严格同一个**（A 建库与 B 评测用同一 teacher）。**除此之外，teacher 的训练分布、checkpoint 族、绝对 SR 高低，都不影响「A 建的库能否在 B 用」这一测量。** 反复从 teacher 侧构造反对意见（先是「数据污染」、后是「普适性打折」）是把 D6 的教训（teacher 崩溃会淹没索引效应）过度外推所致。

   ⇒ **换用 `target_posttraining` 对实验设计是加分**：准入门的目的是确认 teacher 在 held-out 厨房不崩，好让 cache 迁移的性能下降不能归因于 teacher 退化；teacher 在 A、B 上都更强，这个前提**更稳**。同时 U0 更少、可用 episode 与 cache 条目更多、任务集更大、统计功效更高。

   （multitask 的两份 180 ep 数据仍全部保留，作为不同 teacher 强度下的对照留档，非必须复现项。）

   ⚠ **两点如实声明**：① 该仓库的 **README 为空**，各支的语义是从**目录命名与 `trainer_state.json` 推断**，非官方明示 —— 若要把「leaderboard 用的是 target_posttraining」写进论文，须另找论文正文或官方表格佐证；② 我方 checkpoint 的 `trainer_state.json` 显示 `global_step=120000` / `max_steps=300000` / `epoch=0.265`，是**训练中途存档**，但它是官方在 `gr00t_n1-5/multitask_learning` 下**唯一发布**的 checkpoint（该目录仅 9 个文件、无其它 step），故不存在选错版本的问题。（对照：pi0.5 侧同样是唯一发布的 `75000`。）

   ⚠ **过程中的一个方法论教训**：结果按任务原子落盘，故**部分文件在整个跑程中始终存在**，而任务按**字母序**执行。跑到 8/18 时算出 `SR_B=25.0%`、P1 **FAIL**，实为字母序把 GR00T 的 `PickPlace*` 强项族全排在后半段所致的系统性偏倚（同一时刻 pi0.5 在这 8 个任务上是 42.5%/45.0%）。**部分数据算出的 P1 不是准入判决**。`analyze_admission_gate.py` 已加 `--expect-tasks`（默认 18）在任务数不足时打印显式警告。

   ### ✅✅✅ GR00T **target-posttrained** 准入门（2026-08-16，180 ep，墙钟 **2.33 h**）—— **P1 PASS，U0 = 0**

   owner 裁定换 checkpoint 后重跑。配置与前两次**逐字一致**：`tasks=18 A=(1,1) B=(7,7) trials=5 replan=5`；server metadata 确认 `checkpoint=.../target_posttraining/atomic_seen/checkpoint-60000`、`diagnostic_seed=None`。

   **三个 teacher 汇总对照**：

   | | pi0.5 | GR00T **mt** | GR00T **tp** |
   |---|---|---|---|
   | SR_A | 41.1% | 26.7% | **60.0%** |
   | SR_B | 50.0% | 34.4% | **71.1%** |
   | P1（`SR_B` Wilson 下界 vs 20%） | 39.9% PASS | 25.4% PASS | **61.0% PASS** |
   | gap (A−B) | −8.9pp | −7.8pp | **−11.1pp** |
   | U0 / U1 / U2 | 4 / 2 / 12 | 6 / 4 / 8 | **0 / 2 / 16** |
   | 180 ep 墙钟 | 4.59 h | 2.90 h | **2.33 h** |

   🟢 **U0 从 6 降到 0** —— tp 在全部 18 个任务上都至少有一臂成功，包括 pi0.5 与 mt **共同**的三个 U0（`CloseToasterOvenDoor` / `NavigateKitchen` / `TurnOnMicrowave`）。
   🟢 **三个 teacher 的 gap 全部同号**（−8.9 / −7.8 / −11.1 pp，均为 held-out 的 scene B 略优）⇒「held-out 厨房不更难」在**三个不同强度/训练范式的策略**上重复成立。

   ### 🟢🟢 harness 正确性由此得到最强验证 —— **tp 复现了论文 Table 2 的公开数字**

   论文 Table 2（foundation model training，= `target_posttraining`）报 Atomic **68.5%**，且**评测在 target kitchens**——与我方配置（target 场景 (1,1)/(7,7)）**同口径**。我方实测 **SR_A 60.0% / SR_B 71.1%**，`SR_B` 的 95% CI **[61.0%, 79.5%]** **覆盖 68.5%**。

   ⇒ **当 checkpoint 与评测场景都与论文设定对齐时，我方 harness 复现了官方数字。** 这反过来彻底关闭了「GR00T 绝对 SR 偏低」的存疑项：mt 的 34.4% 之所以低于 Table 1 的 43.0%，纯粹因为 **Table 1 在 pretraining 厨房评测、我方在 target 厨房评测**，是分布外退化而非缺陷。**replan=5 vs 原生 16、场景数等候选原因全部排除，无须再查。**

   ⚠ **一个反例，防止把 tp 说成全面更强**：`PickPlaceCounterToCabinet` 上 tp **2/5 A** 弱于 mt 的 4/5。target-posttraining 不是在每个任务上都占优。

   ### 🔶 可用任务交集 —— ⚠ 本规则是**事后决定**（P3 要求）

   ⚠⚠ **必须按预注册 P3 记为事后决定**：§12-2 的 P2 只规定了**单个 teacher** 内排除 U0；「跨两个 teacher 取交集」是在看过 pi0.5 数据之后新增的排除规则，不在预注册内。

   **交集（11 个，正式跨场景实验的任务集）**：`CoffeeSetupMug`、`OpenCabinet`、`OpenDrawer`、`OpenStandMixerHead`、`PickPlaceCounterToCabinet`、`PickPlaceCounterToStove`、`PickPlaceDrawerToCounter`、`PickPlaceSinkToCounter`、`PickPlaceToasterToCounter`、`SlideDishwasherRack`、`TurnOnSinkFaucet`。
   （单侧可用数：pi0.5 **14/18**、GR00T **12/18**。）

   **被剔除的 7 个**及归因：

   | 任务 | pi0.5 | GR00T | 剔除原因 |
   |---|---|---|---|
   | `CloseToasterOvenDoor` / `NavigateKitchen` / `TurnOnMicrowave` | U0 | U0 | 两个 teacher 都不工作 —— 干净剔除，对 A/B 对比中性 |
   | `CloseFridge` | **U2** | U0 | **teacher 特异**：pi0.5 4/5+3/5，GR00T 全 0 |
   | `CloseBlenderLid` | U1 | U0 | teacher 特异 |
   | `TurnOffStove` | U1 | U0 | teacher 特异 |
   | `TurnOnElectricKettle` | U0 | **U2** | **反向 teacher 特异**：GR00T 1/5+2/5，pi0.5 全 0 |

   🔴 **交集付出的真实代价**：4 个任务是**某一个 teacher 会做而另一个不会**（三个 pi0.5 会、一个 GR00T 会）。这些任务讲的是**两个 teacher 的能力差异**，不是跨场景迁移，故排除是对的；但必须在论文中列明，否则读者无从判断任务集是怎么来的。⚠ 另注 GR00T 的三个 U0 全是「关闭类」接触任务（`Close*` 三个全 0/10），而 pi0.5 的 `CloseFridge` 有 7/10 —— 这个模式若在 K=10 复核时仍成立，值得单独一句讨论。

   ⚠ **K=5 筛两次会放大假阳性**：真 per-arm SR=10% 的任务被**至少一个** teacher 误判为 U0 的概率 **0.58**（单 teacher 0.35）。⇒ 「U0 须在 K=10 数据上复核」这条预注册要求，在双 teacher 下**更加必要**，而非可以放松。

   🟢 **预算按实测重算（喂给 §12-5 与 owner 的 K 裁定）** —— 用两次准入门的**实测 per-task 墙钟**，非模型外推：

   | teacher | 全 18 任务 | 交集 11 任务 | 准入门总墙钟 |
   |---|---:|---:|---:|
   | pi0.5 | 91.7 s/ep | **85.7 s/ep** | 4.59 h |
   | GR00T N1.5 | 58.1 s/ep | **54.6 s/ep** | **2.90 h** |

   正式设计 = 11 任务 × 5 场景 × K，**每侧**、**每 teacher**：

   | K | ep/侧/teacher | pi0.5 两侧 | GR00T 两侧 | **两个 teacher 合计** |
   |---:|---:|---:|---:|---:|
   | 5 | 275 | 13.1 h | 8.3 h | **21.4 h** |
   | **10** | **550** | **26.2 h** | **16.7 h** | **42.9 h** |
   | 30 | 1650 | 78.6 h | 50.1 h | 128.6 h |

   🟢 **关键结论：加第二个 teacher 不增加总成本**。原 §12-5 的「18 任务 × K=10 × pi0.5 单 teacher × 两侧」按同一实测口径算是 **42.9 h**，与「11 任务 × K=10 × **两个** teacher × 两侧」的 **42.9 h** 相等 —— 任务集从 18 缩到 11（−39%）恰好抵消了 teacher 数翻倍，而 GR00T 比 pi0.5 快 **1.6×** 又贡献了余量。⇒ **K=10 的建议不因引入第二 teacher 而需要下调。**

   ### 🟢 换 tp 后的最终预算（2026-08-16 实测重算，取代上表）

   实测 per-ep（全 18 任务口径）：pi0.5 **91.7 s/ep**、GR00T-tp **46.5 s/ep**（tp 比 mt 的 58.1 s/ep 更快，因成功得早、提前退出）。

   按 owner 裁定（**K=10 + 采集全 18 任务**）：

   | K | ep/侧/teacher | pi0.5 两侧 | GR00T-tp 两侧 | **合计** |
   |---:|---:|---:|---:|---:|
   | 5 | 450 | 22.9 h | 11.6 h | **34.6 h** |
   | **10** | **900** | **45.9 h** | **23.3 h** | **69.1 h** |

   ⇒ 换 tp 后总预算 **69.1 h**，比换之前（用 mt 的 74.9 h）**还省 5.8 h**，同时 U0 归零、任务集从 11 扩到 14。**换 checkpoint 在成本与统计功效上双赢。**

   ⚠ **若 owner 裁定 U0 处置走方案 (a)「全保留 18 任务、分层报告」**，成本是交集口径的 **1.75×**：K=10、两 teacher、两侧 = **74.9 h**（pi0.5 45.9 h + GR00T 29.1 h）。该方案的收益是无选择偏差且 U0 自带 K=10 复核数据 —— 在双 teacher 下这个收益比单 teacher 时更值钱（误判概率 0.35 → 0.58），但代价也实打实翻了近一倍，须一并权衡。

   ---
   - **修正预算**：smoke 阶段按 `TurnOnMicrowave`（horizon 450）实测 82–106 s/ep 外推得 180 ep ≈ 4.6 h —— **该数字偏低**，因 450 低于 18 任务的加权平均 horizon **658**。按后续多任务实测重算，Step 0b 正式准入门 180 ep ≈ **4.9 h 单进程**（详见 §12-5；分片倍率受显存所限）。若需压缩：replan 提到 10 可减半推理调用（模型本就预测 50 步），或减少 trials。

**Benchmark 侧**

3. ✅ **已闭合（2026-08-15）** — `robocasa-benchmark/openpi` fork 的 diff 面。**fork 基于 upstream，不含我们的 cache/conductor** ⇒ 正确方向是**把 robocasa 支持搬进我们的 repo**，而非在 fork 上重建 cache。搬运清单：评测必需 = `examples/robocasa/main.py`(218 行) + `src/openpi/policies/robocasa_policy.py`(127 行) + `training/config.py` 的 robocasa 配置片段；非必需 = `groot_utils/groot_openpi_dataset.py`(406) + 3 个转换/统计脚本(497)。⚠ **障碍**：fork 的 `training/config.py:36-37` 在**模块顶层** `from robocasa.macros import ...`，而 robocasa 只存在于 py3.12 孤岛 ⇒ 搬运时必须改成延迟/条件导入，否则我们主 venv 的 config 无法导入。
4. ✅ **已闭合（2026-08-15 实跑，`TurnOnMicrowave`，两条 pin 路径 × 每条 4 次 reset）** — **两条路径都成立，且都无副作用**：

   | 路径 | 场景 | 场景被钉定 | 钉定值 == 请求值 | 物体/放置**仍随机** | mean reset | max reset |
   |---|---|---|---|---|---:|---:|
   | **A** `layout_and_style_ids=[(L,S)]`（构造期，官方常走） | (1,1) | ✅ | ✅ | ✅ | 4.59 s | 7.42 s |
   | **A** 同上 | (7,7) | ✅ | ✅ | ✅ | 11.20 s | 12.24 s |
   | **C** `set_ep_meta({"layout_id","style_id"})`（ep 级，未经验证的细粒度路径） | (1,1) | ✅ | ✅ | ✅ | 3.64 s | 4.23 s |

   判定的关键不只是「场景被钉住」，而是**其余随机因子必须仍然自由**——否则整个 (task,scene) 单元退化成同一条轨迹的重复，K 个 episode 提供不了任何统计信息。实测每次 reset 的 `object_cfgs` 指纹与 `init_robot_base_pos` **均互不相同**（如 route A@(1,1) 的 obj_fp `06f30397 / ff66fe99 / d85a393b / 9ecc8cad`，`n_objs` 在 1↔2 间变化），⇒ **场景是唯一被冻结的因子**，正是 §4.6.3 设计所需。

   **采用路径 A**：官方代码路径本就走它，且构造期钉定不依赖 ep 级覆盖的时序假设；路径 C 已验证同样可用，留作需要 ep 内切换场景时的备选。
5. 🟡 **已备可裁决方案，待 owner 拍板** — 每任务 episode 数。**关键区分**：官方的 "trials per task"（代码 50 / 论文 30 / LeRobot 20）是在 10 个 target 场景上**随机采样**；我们的设计是**固定场景**，语义不同，不能直接沿用。我们真正要预注册的是每 (task, scene) 单元的重复数 **K**：

   **成本已由实测标定（2026-08-15，RTX 4090，replan=5）**，原表按 H200 估的数字**低估约 7 倍，已作废**：

   | 实测样本 | horizon | 结束情况 | s/ep | **s/step** |
   |---|---:|---|---:|---:|
   | `TurnOnMicrowave` @scene(1,1) | 450 | 3/3 跑满 | 71.0 | **0.158** |
   | `TurnOnMicrowave` @scene(7,7) | 450 | 3/3 跑满 | 102.1 | **0.227** |
   | `CloseFridge` @scene(1,1) | 900 | 3/3 成功于 ~778 步 | 112.8 | **0.145** |

   ⇒ 单步成本**只随场景变**（0.15 s @(1,1) → 0.23 s @(7,7)），**不随任务/horizon 变**。

   🟢 **随后的 36-ep 诊断跑给出了直接实测的 per-ep 成本**（6 任务混合，含成功提前结束的真实分布，**比上面按单步成本外推更可信**）：

   | | 每 ep 均值 | 本次 6 任务平均 horizon | **归一化到全 18 任务 (horizon 658)** |
   |---|---:|---:|---:|
   | scene A (1,1) | 90.5 s | 700 | **85.1 s/ep** |
   | scene B (7,7) | 115.7 s | 700 | **108.8 s/ep** |
   | 两场景均值 | 103.1 s | 700 | **96.9 s/ep** |

   B/A = **1.28×**（rollout 段的场景成本差**小于** reset 段的 2.4×，因推理耗时不随场景变化）。18 个 atomic_seen 任务的加权平均 horizon = **658**（分布 `{450:6, 600:4, 750:4, 900:3, 1050:1}`），与本次的 700 仅差 6.4%，故上表归一化外推可靠。**下表全部采用实测口径 96.9 s/ep**（早先按单步成本建模得 118 s/ep，高估约 18%，已弃用）。

   | 用途 | 协议 | 规模 | **单进程（实测外推，可信）** | 2 路分片（须先实测显存，见下） |
   |---|---|---|---:|---:|
   | **Step 0b 正式准入门** | 18 任务 × (1 个 A + 1 个 B) × 5 | 180 ep（90 A + 90 B） | **4.9 h** | 2.4 h |
   | **正式跨场景实验 K=5** | 18 任务 × 5 场景 × 5 | 450 ep/侧 | **12.1 h/侧** | 6.1 h/侧 |
   | **正式跨场景实验 K=10** | 18 任务 × 5 场景 × 10 | 900 ep/侧 | **24.2 h/侧** | 12.1 h/侧 |
   | 正式跨场景实验 K=30 | 18 任务 × 5 场景 × 30 | 2700 ep/侧 | **72.6 h/侧** | 36.3 h/侧 |
   | （可选）leaderboard 对照锚 | 官方 50 trials/task，随机场景 | 900 ep | **24.2 h** | 12.1 h |
   | （参考）本次已完成的诊断跑 | 6 任务 × 2 场景 × 3 | 36 ep | **1.03 h（实测墙钟）** | — |

   ⚠ `replan_steps` 5→10 可让上表**推理调用减半**（模型本就预测 50 步），是唯一零显存代价的提速手段；但会改变闭环控制频率，须作为实验参数预注册、两侧一致。

   ⚠ **「3 路分片」是推断值，未实测，且显存实测显示在本机上不成立** —— 上表分片列请按 **2 路** 保守读，或换机执行。实测依据（2026-08-15，s0b 主跑期间采样）：

   | 项 | 显存 |
   |---|---:|
   | 既有 `serve_policy.py` :8000（owner 要求常驻，不可关） | 8.75 GB |
   | 既有 ACT `sidecar_server.py` | 3.06 GB |
   | 我们的 `serve_robocasa_pi05.py` | **7.80 GB** |
   | 三者小计 | 19.6 GB |
   | **`nvidia-smi` 报告总用量** | **32.8 / 49.1 GB** |

   差额 **13.2 GB 未归属到任何 compute-app**（sim client 的 EGL/CUDA context 未被 `--query-compute-apps` 计入）⇒ **实际每路成本 ≈ server 7.8 GB + client ~13 GB ≈ 21 GB**。在既有两个服务不关的前提下余量仅 **16.3 GB**，**连第二路都未必装得下**。

   ⇒ **分片倍率必须先实测再采信**。可行的提速方向按优先级：① 换 ziyang10 H200（显存充裕）；② 征得 owner 同意后临时停掉 ACT sidecar（+3.06 GB）；③ `replan_steps` 5→10，推理调用直接减半（模型本就预测 50 步，**这一项零显存代价、收益最大**）。
   ⚠ 另注：server 当前是**单连接模式**（§12-2 踩坑 7），分片必须一路一 server，不能多 client 共享一个 server。

   GPU 利用率实测均值 **~45%**（13–97% 剧烈波动 —— MuJoCo 步进是 CPU 密集段，GPU 只在渲染+推理时忙）⇒ **算力上确有分片空间，瓶颈是显存不是算力**。

   **统计功效**（配对设计，90 个 (task,scene) 单元，p≈0.4，跨场景 gap 的 95% CI 半宽）：

   | K | CI 半宽 | 能判定什么 |
   |---:|---:|---|
   | 5 | ±6.4 pp | 区分「gap≈0」与「gap≥15pp」 |
   | **10** | **±4.5 pp** | **可判定 gap ≤ 10pp 的准入门槛** |
   | 30 | ±2.6 pp | 过剩（成本 3×，精度仅 1.7×） |

   **建议 K=10**：±4.5pp 恰好够支撑 D6 的「teacher 在新场景仍工作良好」判据（若以 gap ≤ 10pp 为门槛），单进程 **24.2 h/侧**、两侧约 48 h；须靠换机（H200）或 `replan_steps` 5→10 压缩，不可指望本机多路分片（显存不足）。K=30 精度提升不成比例。若只要方向性结论，K=5（12.1 h/侧）亦可。⚠ 上表假设单元间独立，未计入 task×scene 交互，**真实 CI 会更宽**。**与 leaderboard 的可比性只能靠另跑官方协议获得，跨场景实验本身不追求榜单可比**（符合 D4：teacher 能工作即可）。
6. ✅ **已闭合（与第 4 项同一次实跑，12 次 reset）** — **未观察到病态慢或 `PlacementError`**：`TurnOnMicrowave` 在 (1,1)/(7,7) 各 4 次、(1,1) 的 route C 4 次，全部正常返回，`_reset_internal` 的无界重试循环从未表现出发散。**但场景间 reset 成本差异显著**：(7,7) 均值 **11.20 s** vs (1,1) 均值 **4.59 s** = **2.4×** —— 与 §4.6.6 的 `bench_speed` 场景成本差**独立测得同一比值**（第二次确认）。⇒ 预算必须按**目标场景**分别估，不可用全局均值；max reset 12.24 s 可作为超时哨兵的下界参考。

   ⚠ 该 2.4× 已在 Step 0b 的 rollout 端复现：`TurnOnMicrowave` 同为跑满 450 步，scene A 212.9 s vs scene B 306.4 s（+44%，rollout 段差异小于 reset 段，因推理耗时不随场景变化）。
7. ✅ **已闭合（离线源码证明，系数应为 1.0 而非 1.3）** — pi0.5 **无条件编码 3 个图像槽**，与实际相机数无关：`libero_policy.py:59-62` 给出 `base_0_rgb` / `left_wrist_0_rgb` / `right_wrist_0_rgb`，其中第三个是 `np.zeros_like(base_image)` **零填充但仍是一张图**；`pi0_pytorch.py:336` 的 `embed_prefix` 以 `for img, img_mask in zip(images, img_masks)` **逐张无条件调 `embed_image`**，`img_mask` 只参与 attention mask、不跳过 SigLIP 前向（零图像的卷积耗时与真实图像相同）。⇒ **RoboCasa 的 3 相机与 LIBERO 的「2 真实 + 1 零填充」Stage1 耗时相同**，§4.6.6 预算表应改用 raw 行（见该节勘误）。

**Teacher 侧**

8. ✅ **已闭合（本机实测，weilandserver RTX 4090，3 相机 224²，bf16，flash-attn 2.7.4，10 次取均）**：

   | 段 | 耗时 | 含义 |
   |---|---:|---|
   | **[A] vision tower + `mlp1`**（`extract_feature`） | **23.73 ms** | **tap 点之前，每步必付** |
   | **[B] full backbone**（ViT + 12 层 LLM） | **52.89 ms** | |
   | **LLM-only**（B − A） | **29.16 ms** | 占 backbone **55%**，HIT 时跳过 |

   ⇒ 在 vision-tower tap 下，HIT 跳过 = LLM 29.16 ms + 整个 action head（16 层 DiT × 4 去噪步 + 4 层 VL adapter）。按 N1.7 公开的 action-head 数据折层数比例粗估，**跳过占比 ≈ 69%**（pi0.5 4090 实测为 83%）。**远高于**早先基于「tap 在 backbone 之后」得到的 25% —— 那个数字来自错位的 tap 点，见 §9.2 勘误。
9. ✅ **已闭合（前半 schema 比对 + 后半真实加载）**：
   - 前半：两个 N1.5 checkpoint 的 `config.json` 逐字段比对，**12 字段中 11 个完全相同**，唯一差异是 `transformers_version`（4.51.1 vs 4.51.3，纯记录字段）；`action_horizon=16` / `action_dim=32` 一致。
   - **后半（2026-08-15 实跑）**：`GR00T_N1_5.from_pretrained()` 加载 RoboCasa365 官方 checkpoint（`gr00t_n1-5/multitask_learning/checkpoint-120000`，7.2 GB，已排除 optimizer）**成功**。实测层数 **vision 27 层 / LLM 12 层**，与 §8 表格逐项吻合（SigLIP-400M 27 层；Qwen3-1.7B 截断至 12 层）。加载日志同时确认 `Tune backbone visual: True` ⇒ §11-1「key 不可跨 checkpoint 迁移」得到实证支持。
10. ✅ **已闭合（读 HF safetensors header，未下载权重）** — `backbone.eagle_model.mlp1.0.weight` = **BF16 `[2048, 1152]`**，且**只有 `mlp1.0.{weight,bias}`**、无第二层无 LayerNorm ⇒ 纯线性投影、无 pixel-shuffle，坐实 §9 的 256-token/16×16 网格结论。
11. ✅ **已闭合（离线 + 实机双重确认）** — 离线：`vision_model.embeddings.patch_embedding.weight` = `[1152, 3, 14, 14]` ⇒ patch=14、hidden=1152，224/14 = 16×16 = 256 patch。**实机（2026-08-15）**：`extract_feature(pixel_values[3,3,224,224])` 实测返回 **`vit_embeds = [3, 256, 2048]` bfloat16**，`tokens/img=256`、`grid=16×16` —— 与离线推断完全一致。（注：`mlp1` 已把 1152 投影到 LLM 宽度 2048，故 tap 张量末维是 2048 而非 1152。）
12. ✅ **已闭合** — 相机顺序**写死在代码**，不依赖任何外部 `video_concat_order`（该担忧只适用于 GR00T gym wrapper 路径）：`main.py:118-120,146-149` + `robocasa_policy.py:66-75` ⇒ `observation/image`=`robot0_agentview_left`(vision_0)、`observation/wrist_image`=`robot0_eye_in_hand`(vision_1)、`observation/right_image`=`robot0_agentview_right`(vision_2)。⚠ **pi05 强制要求 `right_image`**（缺则 raise）⇒ RoboCasa 上用满 3 相机，key 维度 3×32768，而我们 LIBERO 只用 2 个。
13. ✅ **已闭合** — rollout 循环可插回调，且**比 LIBERO 更容易**：`main.py:154` 是标准 `client.infer(element)["actions"]`，整个 main.py 仅 218 行、循环为直白的 `while t < horizon`（对比 LIBERO main.py 1000+ 行）。改动即在该行旁读取 `result.get("__hit_meta__")`。
14. ✅ **立项前部分已闭合（2026-08-15 孤岛 B 内实跑全包 import 扫描）** — cache 系统对非 pi0.5 模型的移植。
    - 🟢 **cache 包高度自包含**：`src/openpi/cache/` 对 openpi 主包的外部依赖**只有 `openpi_client`**（轻量 websocket 客户端），其余 import 全是 `openpi.cache.*` 自引用。
    - 🟢 **jax 只在 `interceptor.py` 出现 5 处，且全是 `jax.tree.map`**（纯结构映射，非 JAX 计算）。而 `interceptor.py` 恰是 migration 指南要求**为每个新模型重写**的组件 ⇒ GR00T 版自写即天然避开 jax。**模型无关内核（orchestrator / cache_storage / backend / components 除 key_builder+interceptor）零 jax 依赖。**
    - ⚠ **三环境拓扑约束**（本轮新发现）：cache interceptor 跑在 **server 侧**，而 GR00T N1.5 要求 `numpy>=1.23.5,<2.0.0` + `transformers==4.51.3`（`n1.5-release:pyproject.toml`），与 robocasa365 孤岛（numpy 2.2.5 / py3.12）**又是一个互斥环境**。最终拓扑 = 孤岛A（robocasa365 sim client）／孤岛B（GR00T N1.5 policy server + 我们的 cache 内核）／主 venv（openpi + pi0.5 server）。cache 内核必须能装进孤岛 B —— 由上面两条判断为可行。
    - 🟢 **已实跑证实（2026-08-15，孤岛 B = py3.11.15 / numpy 1.26.4 / transformers 4.51.3 / torch 2.5.1+cu124）**：把 `/home/weiland/openpi/src` 加进 `sys.path` 后对 `openpi.cache` 做 `pkgutil.walk_packages` 全递归 import —— **47 个模块中 44 个成功**，3 个失败**无一是版本冲突**：

      | 失败模块 | 原因 | 处置 |
      |---|---|---|
      | `backends/qdrant_backend` | 缺 `qdrant_client` | 可选后端，不用则无需装 |
      | `interceptor` | 缺 `jax` | **预期内** —— 正是每个新 teacher 必须重写的组件，GR00T 版自写即天然避开 |
      | `sidecar_executor` | 缺 `websockets` | 装 1 个纯 Python 包即可 |

      依赖补齐的副作用也已用 `uv pip install --dry-run` 量化：**单装 `websockets` = 装 1 包、零降级、零冲突**；若连可选的 qdrant 后端一起装，则为装 12 包 + **仅** `portalocker 4.1.0→3.2.0` 一处降级，**numpy / torch / transformers 三个关键钉定全部不动**。⇒ 「模型无关内核零 jax 依赖、可整体装进孤岛 B」由推断升级为**实测**。
    - 🟢 **tap→key 链路已实机跑通（2026-08-15）**：孤岛 B 内加载 RoboCasa365 官方 N1.5 → `extract_feature()` 得 `vit_embeds [3,256,2048]` → `view(3,16,16,2048).permute(0,3,1,2)` → `adaptive_avg_pool2d(4,4)` → flatten ⇒ **每相机 key 维度 = 32768，与 pi0.5 生产逐位一致（MATCH: True）**。§9 表格中「key 维度逐位相同」由推断升级为实测。
    - 剩余：把该 key 接进 `CacheOrchestrator`（写 GR00T 版 Interceptor + KeyBuilder），属 §4 Code 阶段工作，非立项前检查项。

---

## 13. 待 owner 裁决

- 是否将 §4 写成正式实验卡（暂名 **X15 跨场景继承检验**）并入 [`docs/iclr/tier_experiment_designs.md`](../docs/iclr/tier_experiment_designs.md)。
- X8 确认集是否改挂 **RoboCasa365 / Atomic**，并增列第二 teacher 臂。
- §11-8 / §11-9 的披露与缺口是否作为独立改动进入 `exp/ablation_study/analysis/analysis.md` 与实验设计卡。
- N1.5 权重非商用许可对投稿的影响是否需在论文中声明。
- `/home/weiland/projects/LIBERO-plus`（154 MB）是否保留（当前暂留，供 §5.3 边界测试可能性）。
- **【本轮新增】场景对的选法**：确认维持 **A/B 双 held-out 对称设计**（两个场景都取自 target split (1,1)…(10,10)，teacher 均未训过）。依据见 §12-2 概念修正 —— 训练场景是 layout/style 11–60，与评测场景零交集，故 A/B 处境对称，cache 迁移效果不被 teacher 能力差异污染。**备选**是拿 pretrain 场景 (11–60) 当 A，可额外测出「teacher 训练场景 → held-out 场景」这一维，但会引入训练场景优势、与对称性冲突。**建议维持对称设计**。
- **【本轮新增】§12-5 的 K**：正式跨场景实验每 (task, scene) 单元的重复数。已备实测成本 + 统计功效表，**建议 K=10**（±4.5pp，单进程 24.2 h/侧，须配合换机或 replan 提速）。只需一句裁定。

---

## 附：本地资产

| 路径 | 说明 |
|---|---|
| `/home/weiland/projects/Isaac-GR00T` | NVIDIA Isaac-GR00T 克隆，已 `git fetch --unshallow --tags`，含全部 release tag（`n1-release` / `n1.5-release` / `n1.6-release` / `n1.6.1-release` / `n1.7-release`）。工作树干净，停在 `main`（N1.7）。读 N1.5 源码用 `git show n1.5-release:<path>` 或临时 worktree |
| `/home/weiland/projects/robocasa` | robocasa 浅克隆（47 MB），已 checkout 到 Isaac-GR00T 指定的 pin **`be22d659b02db8f6d7f3a3c3edc742934fdcbaae`**（`setup_RoboCasa.sh:108`）。§4.6 的全部代码锚点出自此处。**未安装、未建 venv、未下资产** |
| `/home/weiland/projects/LIBERO-plus` | LIBERO-Plus 浅克隆（154 MB）。**方案已弃用（D6）**，保留供 §5.3 边界测试 |

**weilandserver 上的 Step 0b 执行产物**（已从 `/tmp` 移出以免被清理）——`/home/weiland/step0b_artifacts/`：

| 文件 | 说明 |
|---|---|
| `serve_robocasa_pi05.py` | pi0.5 RoboCasa365 teacher server（独立脚本，不改 `config.py`；含 `use_quantile_norm=False` 的必需覆盖） |
| `step0b_v2.py` | Step 0b 客户端（场景钉定 + fork 逐字节一致的 obs 预处理 + 官方 `convert_action`） |
| `verify_scene_pin.py` | §12-4/§12-6 的场景 pin 与 `PlacementError` 探针 |
| `step0b_6task.{json,log}` | 36 ep 原始结果与完整日志 |
| `scene_pin.log` | 12 次 reset 的场景 pin 验证原始输出 |

**环境与拓扑**（复跑时照抄）：
- 孤岛 A（sim client，py3.12）：`/home/weiland/Isaac-GR00T/gr00t/eval/sim/robocasa365/robocasa365_uv/.venv`，工作目录须为 `/home/weiland/Isaac-GR00T/external_dependencies/robocasa365`
- 孤岛 B（GR00T N1.5，py3.11）：`/home/weiland/gr00t_n15_venv/.venv`
- 主 venv（openpi + pi0.5 server）：`/home/weiland/openpi/.venv`
- EGL 三件套：`LD_LIBRARY_PATH=/home/weiland/nvidia-gl/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH`、`__EGL_VENDOR_LIBRARY_DIRS=/home/weiland/nvidia-gl/root/usr/share/glvnd/egl_vendor.d`、`MUJOCO_GL=egl`
- teacher 权重（已转换）：`/home/weiland/ckpt_pi05_robocasa_pytorch`（6.9 GB，PyTorch）

**复跑命令**（server 已于 2026-08-15 21:28 定点停掉以释放 7.6 GB 给同机的其他 session；下次跑先起 server）：

```bash
# ① teacher server（主 venv）— 约 3-5 min 加载完，等 stdout 出现 SERVER-LISTENING
tmux new-session -d -s pisrv "export HOME=/home/weiland; \
  cd /home/weiland/openpi && .venv/bin/python /home/weiland/step0b_artifacts/serve_robocasa_pi05.py 2>&1 | tee /tmp/pisrv.log"
# 归档版脚本的 PORT 已固定为 8010（本机 8000 被既有 srv0 长期占用，勿动）

# ② Step 0b 正式准入门：18 任务 × 2 场景 × 5 ep = 180 ep ≈ 4.9 h（孤岛 A）
tmux new-session -d -s s0b "export HOME=/home/weiland; \
  export LD_LIBRARY_PATH=/home/weiland/nvidia-gl/root/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH; \
  export __EGL_VENDOR_LIBRARY_DIRS=/home/weiland/nvidia-gl/root/usr/share/glvnd/egl_vendor.d; \
  export MUJOCO_GL=egl PI_HOST=127.0.0.1 PI_PORT=8010 N_TRIALS=5 SCENE_A=1,1 SCENE_B=7,7 \
         OUT_JSON=/home/weiland/step0b_artifacts/step0b_full.json; \
  cd /home/weiland/Isaac-GR00T/external_dependencies/robocasa365 && \
  /home/weiland/Isaac-GR00T/gr00t/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python \
    /home/weiland/step0b_artifacts/step0b_v2.py 2>&1 | tee /home/weiland/step0b_artifacts/step0b_full.log"
```

（不传 `TASKS=` 时脚本默认取满 `TASK_SET_REGISTRY["atomic_seen"]` 全 18 个任务。）
