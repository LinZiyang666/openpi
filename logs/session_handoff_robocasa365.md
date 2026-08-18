# Session Handoff — RoboCasa365 跨场景 cache 实验

> ⚠ **本文件不是 `logs/session_handoff.md`**。那一份属于 X14 在线 RL Router session（工作树里有其未提交更新、`rlr*` tmux 仍在跑），**勿动、勿暂存**。

**Status**: `Active` — 本地无在跑任务；**停在统一临时 G2 待审**。
**当前位置**: 接回标准框架临时流程：**临时 G1 APPROVED（Round 5）→ Code T1–T3 已完成**（自查 tests/robocasa365+cache+conductor = 1485 passed/0 failed）；统一临时 G2 范围 = 本计划 diff + GR00T cache 集成 `28c41c6`（含其 G2 唯一未闭合项 G0-E）。⚠ Code 两处偏差见 plan §8.1（存档改 `baselines/`；pyproject 无 ruff 可改）。
**日期**: 2026-08-17

---

## 0. 接手第一步

1. 本文件读完。
2. [`logs/robocasa365_framework_integration.log.md`](robocasa365_framework_integration.log.md) —— **L2 plan（G1 APPROVED）+ 待裁清单 + 偏差记录 §8.1**。当前停在**统一临时 G2**：审查范围 = 该 plan 的 diff + commit `28c41c6`（含 G0-E）。**T5 正式采集在 G2 APPROVED 前禁止。**
3. [`logs/groot_cache_integration.log.md`](groot_cache_integration.log.md) —— L3 集成计划，G1 APPROVED，Review Log 里有 G2 Round 1（6 项 blocking）与 Round 2（逐条应答）。
4. [`logs/benchmark_and_teacher_selection.log.md`](benchmark_and_teacher_selection.log.md) —— 选型 + 准入门全部数据。**数字以该文件为准。**
5. 本 session Authority = **Execution**。

---

## 1. 术语（先统一，否则会混）

| 层级 | 术语 | 实例 | 代码里是什么 |
|---|---|---|---|
| 1 | **task set** | `atomic_seen`(18) / `composite_seen`(16) / `composite_unseen`(16) | `TASK_SET_REGISTRY` 的一个命名列表 |
| 2 | **task** | `OpenCabinet` | env 类名 = gym id = `ATOMIC_TASK_DATASETS` 的键（带 `horizon`） |
| 3 | **episode** | 一次 `reset()`→rollout | 由 seed 驱动的一次采样 |

正交轴：**scene** = `(layout, style)`（各 1–60；TEST=1–10 / TRAIN=11–60）、**object instance split** = `target`/`pretrain`。

⚠ **「评测任务子集」≠「task set」**：我们从 `atomic_seen` 里筛出的 12/13/9 个 task 叫**评测任务子集**。早期文档写成"任务集"与 benchmark 撞名，已全线统一。

⚠ **官方没有 demo 级 train/test**：`get_ds_soup` 的 split 只有 `pretrain`/`target`/`real`，`filter_key` 只是按条数抽样。评测 = 模拟器里 online rollout。泛化轴是**场景 / 任务 / 物体实例**三条。

⚠ **初始状态没有 LIBERO 那种 init 池**：位姿由 `rng.uniform` **连续**采样，物体实例/site **离散**（target 划分每类 2–13 个，众数 5，共 198 类）。⇒ 同 seed 同 (layout,style) → 完全相同；**同 seed 不同 scene → 不同**（摆放锚在该厨房 fixture 上，且建 arena 也吃同一个 generator）。
⇒ **配对发生在 task 层，不在 episode 层**。分析脚本按 task 算 SR 差再做单样本 t —— 这层配对是真的，与 seed 无关。

---

## 2. ⛔ 概念纪律（同一错误犯过两次）

> **teacher 是固定底座，不是自变量。** 自变量只有一个：**cache 库建在哪个场景**。

teacher 只需 ①足够能干 ②两臂严格同一个。其训练分布、checkpoint 族、绝对 SR 高低都不影响「A 建的库能否在 B 用」。
犯过的两次：先说「teacher 训练过 target 场景会污染 held-out 前提」（错：库在 A 建、在 B 评）；被纠正后又说「teacher 表征在 A/B 上被优化过会削弱普适性」（错：假说**就是**「key 抽自内部表征故跨场景可用」）。根因是把 D6 教训（teacher 崩溃淹没索引效应）外推成「teacher 不许见过评测场景」。

⚠ 相关事实：我们选的 `target_posttraining` 是在 **target 数据**上训的，而 target 数据就采在评测用的那 10 个对角厨房里 ⇒ teacher **见过 (1,1)**，没见过 (1,7)/(5,1)/(5,7)（离对角、不在官方 target 集里）。这个不对称已由实测消解：2×2 六个偏移无一超出噪声底。

---

## 3. 已完成

### 3.1 准入门：7 次、1260 ep、全 P1 PASS

| | pi0.5 | GR00T mt（留档） | GR00T **tp**（选用） |
|---|---|---|---|
| SR_A/SR_B @(1,1)/(7,7) | 41.1%/50.0% | 26.7%/34.4% | **60.0%/71.1%** |
| U0/U1/U2 | 4/2/12 | 6/4/8 | **0/2/16** |

2×2 场景门（各 90 ep/teacher）：(1,7) tp 60.0% / pi05 44.4%；(5,1) 58.9% / 37.8%；(5,7) 63.3% / 45.6%。**12 格全 P1 PASS**。
🟢 **噪声底**（(1,1) 锚点重测）：**GR00T-tp 5.6pp / pi0.5 6.7pp** ⇒ 2×2 六个偏移无一超出 ⇒ teacher 在每格同样能干。
🟢 harness 正确性由复现论文 Table 2 的 68.5% 验证（我方 CI [61.0,79.5] 覆盖）。

### 3.2 GR00T 接 cache：代码已 ship（commit `28c41c6`，已 push）

- G1 **APPROVED**（Round 5，历经 3 轮 blocking）
- 38 文件 / +4189 −88；`uv run pytest tests/cache tests/exp tests/robocasa365` → **2584 passed, 17 skipped**
- 交付：`src/openpi/cache/groot/{staged,key_builder,interceptor,load_guard}.py`；共享缝四处（`_CP1BaseKeyBuilder._slice()` 钩子、`config.py` 注册 `cp1_groot_*`、`CacheStorage.artifact_meta` facade、`InMemoryBackend` 记 artifact 身份）；`exp/robocasa365/{groot_cache_collector,groot_key_parity}.py`；server `--cache-config`/`--collect-hdf5`；client `__ctrl__`/JSONL；建库脚本 `--vision-slots`/`--robot-state-dim`；12 个测试文件；三份文档

**⚠ 其 G2 已并入统一临时 G2**（owner 裁定）：Round 1 六项 blocking 中五项已改完，**第六项 G0-E 实机闭环证据未取**（需远端跑），该项原样带入统一审查。

---

## 4. 三条最容易写错且不报错的地方

1. **图像 token 偏移随 prompt 长度浮动**（实测 `[1,813]`，三段 `(20,256)(283,256)(546,256)`）⇒ 必须按 `input_ids==151669` 掩码定位。照搬 pi0.5 固定偏移表（0/256/512/768）会切到文本 token 上，**shape 不变、测试全过**。
2. **`LayerNorm` 在 CUDA autocast 的 fp32 名单上**——实测 bf16 权重+bf16 输入：无 autocast 出 bf16、有 autocast 出 fp32，**max\|Δ\|=0.0137**（Linear 对照逐位相等）。在线/采集/测试任一处漏开 autocast，key 就整体对不上。⇒ runner 拥有 context 且两 stage 入口断言。
3. **inference tensor 在 context *内* 做 `.cpu().float()` 逃不掉**（实测 `is_inference()` 仍 True）⇒ 跨 step 存活后被就地改写即 `RuntimeError`。⇒ session 只包两段前向，CP1 检查与所有持久张量都在 session 外。

另两条：`_VECTOR_DIMS` 的 pool 系条目**没有 `vision_2`**（LIBERO 两相机）⇒ GR00T 第三个相机会被 `in_memory_backend.py:506-507` **静默丢弃**；类型名必须 `cp1_groot_*` 而非 `groot_cp1_*`（`config.py:2050/:2204` 按 `startswith("cp1_")` 触发两条校验）。

---

## 5. 🔴 下一步：接回标准框架（owner 指示的临时流程）

**详见 [`logs/robocasa365_framework_integration.log.md`](robocasa365_framework_integration.log.md)。** 要点：

### 5.1 现状定位（本轮实机核实）

- **F1** `scripts/serve_policy.py` **已有** `--collect`/`--collect_dir`/`--collect_images`（`:109-117`）+ `--cache`/`--cache_config` + 分阶段 device + `--replicas`。
- **F2** ⚠⚠ `src/openpi/policies/robocasa_policy.py` **只在远端 `/home/weiland/openpi`、未被 git 跟踪**，且只 import `einops`/`numpy`/`openpi.transforms`/`openpi.models.model` —— **没有 `import robocasa`**。⇒ 当初「绕开 `serve_policy.py` 免得污染主 venv」的理由**不成立**；且该文件**只有一份，有丢失风险**，应优先抢救入库。
- **F3** `create_policy`（`serve_policy.py:248-296`）走 `_config.get_config(name)` ⇒ 只需在 config registry 注册 RoboCasa 推理 config（`serve_robocasa_pi05.py` 已把 `TrainConfig` 内联写好，可直接搬）。
- **F4** `openpi.conductor` 的接入面是**单个 ABC**：`EpisodeRunner.run(EpisodeTask, report) -> EpisodeResult`（`worker.py:44-57`）；`EpisodeTask.extra` 是自由字段，场景可搭车 ⇒ **conductor 本体零改动**。
- **F5** `serve_robocasa_pi05.py`（远端 `/home/weiland/step0b_artifacts/`，**同样未入库**）把裸 `Policy` 直连 server —— 无 interceptor / collector / cache ⇒ **pi0.5 在 RoboCasa 上没有采集路径**。

### 5.2 ⚠ 并发：两侧都是并发设计，我方代码目前只支持单连接

**server 侧**：`serve_policy.py:160` **`concurrent: bool = True` 是默认**。每条连接调一次 `connection_policy_factory(base_policy, bundle_id)`，内部 `build_per_connection_components(...)` —— **只共享 storage（线程安全）**，key_builder/timer/gates/judges/strategies **全部 per-connection 新建**。包装链顺序（注释明写不可乱）：`InferenceInterceptor`（最内）→ `PolicyRecorder` → `CollectionPolicy`（最外）。

⇒ **我方 `_build_served_policy` 用 `build_cache_components` 一次性建一个 orchestrator，是单连接语义。** GR00T key builder 有可变 `self._cache` / `self._state_index`，并发下会互相踩。接回框架必须改成 per-connection 工厂。

⚠ `BatchingCoordinator` 跨连接批处理 stage1/2/3 —— 那是 pi0.5 三阶段形状的，GR00T 两阶段要接需另设计；但它是**可选**的（`coordinator=None` 即 C1 路径），第一版不接。

**client 侧**：`WorkerAgent._default_spawn` 把 worker 起成**独立子进程、各占一个 EGL slot**（`agent.py:78-143`）；`driver.assign_servers` 按 yaml→server 分派；worker 从中央队列 pull。

⚠ **并发要按真能扩起来设计**（owner 2026-08-17 明示：后续实验可迁到别的机器）⇒ **per-connection 工厂是必需项，不是可选项**；`driver.assign_servers` 已支持多 server + `server_capacities`，接对了就能跨机横向扩。

📌 显存只是**当前这台机器**的部署参数，不是设计约束：实测每路 ≈ server 7.8 GB + sim client **~13 GB** ≈ **21 GB**（`benchmark_and_teacher_selection.log.md:1118`；那 13.2 GB 是 EGL/CUDA context，`--query-compute-apps` 计不到）⇒ 在 4090 上并发路数受限，但换机（a100 独占 / H200）即解。**不要因为当前这台装不下就把代码写成单连接。**

### 5.3 临时流程 T0–T5 —— **临时 G1 APPROVED（5 轮 19 项 blocking 全闭合）→ Code 完成 → 停统一临时 G2**

🔴 **流程变更（owner 2026-08-17）**：**原 GR00T cache G2 与本线 T2/T3 的 G2 统一为一次审查**，范围 = commit `28c41c6` + T1–T3 的 diff，继承 GR00T G2 Round 1 唯一未闭合项 **G0-E 实机证据**；**统一临时 G2 APPROVED 之前不产出任何正式 T5 数据**（这条同时裁掉了 D-B）。

Round 1 提了 9 blocking / 2 non-blocking，**全部 Accepted 并已修订**。其中四条是真会踩的坑，接手务必记住：
- ⚠ **T2-b 原判据本身不确定**：`Policy.infer(obs, *, noise=None)`（`policy.py:77-97`）不传 noise 就现采新噪声 ⇒ 必须**同一显式 noise 喂两个栈**再逐位比，且要加反向对照。
- ⚠ **`make_task_uid` 不含 `extra`**（`task.py:57-64`）⇒ teacher / 场景必须编进 `yaml_id`，否则两条臂互相冒名；`bundle_id` 恒填 `"default"`（实测裸 server 只放行这一个值）。
- ⚠⚠ **h5 写失败是静默的**：`data_collector.py:162-164` 吞异常、`:88-90` 零推理 episode 直接 return，而 server 照常 ack、journal 照常记完成 ⇒ 必须有产物清点器与 journal 对账（T5 阻塞前置）。另 `N=20/SR` 只是期望值（约一半概率不足 20），已改为**点估计**二项 0.90 分位 + 可复现补批（⚠ Wilson 下界曾被提出后**撤回**——它在 ŜR=1/10 上要 574–1446 ep，吞掉整个预算）。
- ⚠ **自定义 spawn 必须 `start_new_session=True`**：`WorkerAgent.stop()` 走 `os.killpg(os.getpgid(pid), SIGTERM)`（`agent.py:210-215`），子进程不独立成组就会把 **agent 自己**杀掉。

T0 现状定位 ✅ → T1 抢救孤儿文件入库（L1）→ T2 server 接回框架 → T3 client 实现 `RobocasaEpisodeRunner` → T4 owner 裁定 D-A…D-L → T5 正式采集。
**流程效力已定：混合**（owner 2026-08-17「按流程推进于临时 G1」）—— T1 走 L1，T2–T3 走 L2，plan 就是 `robocasa365_framework_integration.log.md`，**现停在临时 G1 待独立 Review Authority 会话审**。

⚠ **本轮 Understand 推翻了三条上面写过的说法**（详见 plan §2 的 F7 / F8 / F10）：

1. **F7 「逐位一致」的理由错了，结论侥幸成立**：`pytorch_compile_mode` 的唯一消费者是 `cache/interceptor.py:315-331`，无 cache 时**没有人读它** ⇒ 两个栈都是 eager，逐位一致可达。⚠ 反过来：将来把 cache 接进 `serve_policy.py`，编译会被 `_disable_compile_for_serving` **静默关掉**，与现有 `serve_groot_n15.py` 不同。
2. **F8 采集与并发结构性互斥**：`serve_policy.py:684-695` 强制 `--collect` ⇒ `--non-concurrent --replicas 1`（`CollectionPolicy` 挂**模块级 forward hook**），而 `websocket_policy_server.py:520-531` 非并发模式**拒绝第二条连接**。⇒ 采集拓扑 = **1 server 进程 ↔ 1 连接 ↔ 1 worker**；横向扩靠**起 N 个 server 进程**。⇒ §5.2 那句「per-connection 工厂是必需项」**只对评测阶段成立**，对采集阶段不成立。⇒ 另一后果：conductor 的 ctl 连接会占掉唯一那条（`driver.py:184-186` 无条件取 ctl），采集时须注入空 ctl。
3. **F10 两个 teacher 的播种方式本就不同**：pi0.5（`step0b_v2.py:37-41`）`gym.make(..., seed=SEED)` 建环境时播一次、之后 `env.reset()` 不带 seed；GR00T（`groot_rollout_client.py:241`）`env.reset(seed=s)` 每 episode 重播。⇒「固定种子」当前是空话，且前者与中央队列不相容。T3 统一到 `env.reset(seed=base_seed+idx)`。**代价**：接回后的 pi0.5 臂不再逐 episode 复现准入门那 1260 ep（不影响 gate 结论，但 T2 判据必须落在**单次推理**层）。

**另发现第三份孤儿文件**：`/home/weiland/step0b_artifacts/step0b_v2.py`（98 行，pi0.5 的 rollout client，未入库）—— 准入门 1260 ep 的实际执行者，也是 pi0.5 观测契约的唯一权威来源。三份孤儿文件本轮已 `tether pull` 到 job 暂存区兜底，尚未落进工作树。

---

## 6. 待裁决（全部在 framework_integration.log.md §1）

| 编号 | 事项 | 我的建议 |
|---|---|---|
| ⚠ D-A | 采集顺序（owner 要求先 pi，但 pi 无采集路径） | **改判为先 pi0.5**：T2 落地后 pi 的采集路径就是 `serve_policy.py --collect`，不必自写 collector；且 GR00T 侧正卡在 G2（D-B） |
| ⚠ D-L | 采集阶段并发（**本轮新增**） | 单连接 / 多 server 进程；per-connection 工厂那条**只属评测阶段** |
| ⚠ D-B | 未过 G2 的代码可否产正式数据 | 先过 G2，或 owner 明确 override；**不要**"算探索性"（最坏组合） |
| ⚠ D-C | `TurnOnSinkFaucet` 封顶（tp SR=0.1 → 需 ~200 ep ≈2.6 h） | 封顶或剔除 |
| D-D | 评测任务子集 12 / **13** / 9 | 13（宽·pooled），但**必须写死「(1,1) 两次测量合并」这条规则**（此前不可复现） |
| D-E | 种子区间 | 采集 `base_seed=0`、评测 `base_seed=1000000` |
| D-F | 失败轨迹是否也落盘 | 落盘并打标（机时已付，可作 D⁻ 池） |
| D-G | 存储与软链 | `/data/robocasa365_cache/...` + `exp/robocasa365/data_symlink_to_data_disk` |
| D-H | 是否纳入 L0（同场景不同初始状态）阶梯 | 纳入且第一个跑（廉价熔断 + 唯一与 LIBERO 同构的一格） |
| D-I | 是否记录每 episode 的物体实例 | 记录（把"离散轴会不会重复"变成可测量） |
| D-J | 场景是否只测 layout∈{1,5}×style∈{1,7} | 待裁；⚠ L-only 用的 layout005 是**最保守**的几何变化，正结果不能外推到 layout007 量级 |

**预算**（(1,1) 建库，20 条成功/任务，13 个 task）：pi0.5 593 ep ≈ **15.1 h**；GR00T-tp 597 ep ≈ **7.7 h**；合计 **22.8 h / 1190 ep**；磁盘 ≈ 660 GB。`/data` 3.6T 可用 3.3T。

---

## 7. 拓扑与路径（实机验证）

**远端 `weilandserver`**（与本机 `Weiland` 是两台机器），单张 RTX 4090 / 49140 MiB。
⚠ **该 4090 是 owner 自己的硬件，但由 owner 的多个 session 共用**：端口 **8000** 的 `serve_policy.py`（7764 MiB）与两个 `sidecar_server.py`（3392+2772 MiB）属**其它 session，绝不可关**。我方：pi0.5 用 8010、GR00T 用 8020。
⚠ 禁宽模式 `pkill`，只按自己的 tmux 名操作；`cssrv`/`cscol`/`rlr*` 属其它 session。

三个互斥 venv 孤岛：

| 孤岛 | 路径 | 版本 |
|---|---|---|
| **A（sim）** | `/home/weiland/Isaac-GR00T/gr00t/eval/sim/robocasa365/robocasa365_uv/.venv` | py3.12.13 / numpy 2.2.5；cwd 必须是 `.../external_dependencies/robocasa365` |
| **B（GR00T）** | venv `/home/weiland/gr00t_n15_venv/.venv`；**源码 `/home/weiland/gr00t_n15`**（detached HEAD `4af2b62`） | py3.11.15 / numpy 1.26.4 / torch 2.5.1+cu124 |
| 主 venv | `/home/weiland/openpi/.venv` | py3.11.15 / numpy 1.26.4 / torch 2.7.1+cu126 |

⚠⚠ `gr00t` **不在孤岛 B 的 venv 里**，来自 worktree，**必须进 `PYTHONPATH`**。漏了会让 `importorskip` 静默跳过，**看起来通过实为没跑**。

**checkpoint**：
```
GR00T 选用  /home/weiland/ckpt_n15_robocasa_tp/gr00t_n1-5/foundation_model_learning/
            target_posttraining/atomic_seen/checkpoint-60000
GR00T 留档  /home/weiland/ckpt_n15_robocasa/gr00t_n1-5/multitask_learning/checkpoint-120000
pi0.5       /home/weiland/ckpt_pi05_robocasa_pytorch
```
🟢 `serve_groot_n15.py` 的 `DEFAULT_CHECKPOINT` 本轮已改成 tp 那一支（原先指向留档 mt，照 docstring 启动会**静默用错 teacher**）。

**代码送上远端 = git 路线**（P5 已裁）：远端 `/home/weiland/openpi` 是 clone，靠 `git pull --ff-only origin Ziyang`。**明确不用 `tether push` 手工投放**。远端**当前尚未 pull `28c41c6`**。
⚠ 拉之前先只读比对 dirty 与 incoming 的重叠（上次零重叠，其它 session 的 23 个未提交文件原封未动）。

**EGL**（该机无系统 EGL，孤岛 A 必需）：
```bash
export LD_LIBRARY_PATH=/home/weiland/nvidia-gl/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export __EGL_VENDOR_LIBRARY_DIRS=/home/weiland/nvidia-gl/root/usr/share/glvnd/egl_vendor.d
export MUJOCO_GL=egl
```

---

## 8. 陷阱

1. ⚠⚠ **`sys.modules` 假件会为错误 import 路径背书**：假件按被测代码请求的名字注册，代码写错模块名假件就在错名字下被创建 ⇒ 测试与 bug 互相印证。`get_task_horizon` 从错模块导入通过了全部 72 个非 manual 测试，直到真实 rollout 才炸。
2. ⚠⚠ **部分结果文件会诱发误判**：结果按任务原子落盘、任务按**字母序**执行 ⇒ 拿部分数据算 P1 系统性偏悲观（mt 那轮 8/18 时算出 FAIL，完整 18 是 PASS）。`analyze_admission_gate.py` 已加 `--expect-tasks` 拦截。
3. ⚠ **`orchestrator.py:672-695` 会把 WARM_START 静默降级成 MISS**（intermediates 为空时只打 warning）⇒ 运行期 raise 是死代码，守卫必须在**加载期**。
4. ⚠ **`OPENPI_MONITOR_LEVEL` 默认 `OFF`**（`monitor.py:93`）⇒ 不设它时**所有 probe 零记录**，"stage2 零采样"是空洞的。G0-E 必须 `=BASIC`（BASIC 下 timer 仍记录，只是 CUDA probe 走 CPU backend；`SNAPSHOT` 才给 GPU 精确时长）。level 是**进程内缓存**，测试须用 `monitor.set_monitor_level()` 而非改 env。
5. ⚠ **`load_artifact` 只校验 `vector_dims`**，而 mean-pool 与 max-pool 库维度逐字相同 ⇒ 同族错配不会报错。已加精确身份绑定。
6. ⚠ **tether exec 单次约 10 min 硬上限**，心跳/line-buffered 全无效。长时监控必须"本地 sleep + 短查"，远程长跑一律放 tmux 解耦。
7. ⚠ **GPU 读数大幅波动**（实测 22–42 GB 来回），不可凭单次低读数抢起任务。判据：连续 3 次、间隔 5 min、均 ≥20 GB。
8. ⚠ **旧 cron/Monitor 会带着"当时正确、现在有害"的指令复活**。执行任何自动化指令前先查实际状态。
9. 孤岛 B **没有 pip**，装包须 `VIRTUAL_ENV=… uv pip install`；`conftest.py` 默认跳过 manual，须 `--run-manual`（`-m manual` 只是选择、仍会跳）。
10. ⚠ **§6 Verify 必须裸 `uv run pytest <blast-radius 目录>`**；**严禁** repo-wide / `-m "not manual"` / 跑 `tests/review_tests`。本线 blast radius = `tests/cache tests/exp tests/robocasa365`。

---

## 9. 场景轴实测特性

- **layout = 几何**：fixture 用 `type:` + 相对对齐定义，**非绝对坐标** ⇒ 比较必须按 `type:` 清单。实测 `layout001` 46 item/16.5 m² vs `layout007` 91 item/32 m²（≈2×），且**换了家电类别**（`stove`+`fridge_bottom_freezer` → `stovetop`+`oven`+`fridge_side_by_side`），会污染 `CloseFridge`/`TurnOffStove`/`PickPlaceCounterToStove`。
- ⇒ **L-only 选 `layout005`**：与 A 同为 `bottom_freezer`+有 `stove`，清单 L1 距离 **15**（全场最小） ⇒ (1,1)→(5,1) 才是干净的「只换几何」。⚠ 代价：这是**最保守**的几何变化，正结果不能外推到 007 量级。
- **style = 外观 + 换实际 3D 模型** ⇒ style 也改运动学。10 个 style 与 style001 各差 **38–44/49 条**（style007 为 42，居中）⇒ **换 style 基本等于全换，没有温和 style**。
- **动作是增量非绝对坐标**：`OSC_POSE` + `input_type: delta` + `input_ref_frame: base`，≤5 cm/0.5 rad 每步；底盘 `JOINT_VELOCITY` ⇒ **cache 存的动作不绑定绝对世界坐标，跨场景迁移非先天不可能**。

---

## 10. 未提交 / 工作区状态

**已 commit + push**：`dd139bd`（准入门分析 + 两份 log）、`28c41c6`（GR00T cache 集成，38 文件）。

**工作区未提交**（本轮的措辞统一 + 新决策文档）：
`logs/robocasa365_framework_integration.log.md`（新）、`logs/session_handoff_robocasa365.md`（本文件）、`logs/{groot_cache_integration,benchmark_and_teacher_selection,groot_n15_robocasa_adapter}.log.md`、`logs/README.md`、`exp/robocasa365/{analyze_admission_gate,groot_rollout_client}.py`

⚠ 工作树里还混着**其它 session** 的 ~37 个未提交文件（`exp/rl_router/*`、`exp/ablation_study/*`、`src/openpi/cache/components/mlp_router_judge.py`、`docs/iclr/*`、`logs/{session_handoff,markov_sufficiency_plan,cache_size_ablation_plan}*` 等）。**提交必须逐文件点名，不可 `git add -A`。**
