# Session Handoff —— RoboCasa365 warm-start RIT 帕累托（进行中）

> 2026-09-09 深夜。**GR00T 全线完成并出图**；pi0.5 第一轮作废、诊断完成、**v2 正式主跑刚点火（9,100 集，约 17 h）**。
> **停止条件（owner 令）：完全做完实验，pi0.5 做完后才能停，不做完不停。** 之后还有 k=1 双 teacher。

---

## 0. 接手第一步 —— 初始化行为（owner 原令，逐条照做）

> **阅读实验进行 skill，无需挂载**（`experiment-lifecycle`：只读，**不要**执行它的 §0 初始化、
> **不要**去申请 agentchat 账号/token/房间）。**了解 tether 的使用方法**
> （`/home/weiland/projects/dist_experiment_control/docs/usage.md`）**和设备组的使用方法**
> （`docs/devices.md`）。
>
> **corn 负责定时巡检，monitor 负责条件触发，不要职责混淆。**
>
> **机器纪律：server 只能在 weilandserver 运行，worker 只能在 timan107 运行。**
>
> ⛔ **最强硬的纪律警告：不得使用 git commit / push 来同步代码和数据。**
> 一律用 **tether**（`push` / `pull` / `exec`）。git 只在里程碑收口用，且**须 owner 当次明示**。
>
> 无人值守期间**独断专行，不要问任何问题**，不得出现权限申请或其他打断会话的行为。

```bash
cd /home/weiland/projects/openpi          # 分支 Ziyang
cat logs/rc365_rit_run_progress.md        # 进度条（比本文件细）
git status --short | grep -v rit_pareto   # exp/rit_pareto/{build,edit}_figure.py 属另一 session，永不碰
```

Authority = **Execution**，只读 `protocols/execution_authority.md`。跑实验是运行动作，不需要新 G1/G2。

**三条会踩的操作纪律：**
1. **不要用 `rm -rf`** —— 触发权限弹窗、破坏无人值守。用 `rm -f` 点名文件，或写进新目录。
2. **不要自己起 `edit_rit_figure`** —— owner 自己启动，我起了会占住 8765 让他起不来。
3. **不要问问题**，直接做；判断错了就更正并说明。

---

## 1. 这个实验在做什么

用 W13 语料，在 **RoboCasa365 的 13 个任务**上，为 **GR00T N1.5** 与 **pi0.5** 各画一条 **RIT 帕累托前沿**：
横轴 **推理比 IR**（占全量推理成本的百分比），纵轴成功率。阶梯深度 k=2 / k=3，另加 k=1 纯阈值对照。

- GR00T schedule 4 步升序 ⇒ 续跑点 `start_t = 0.75`（剩 1 步）、`0.5`（剩 2 步）
- pi0.5 schedule 10 步降序 ⇒ `0.3`（剩 3 步）、`0.5`（剩 5 步）
- 两条阶梯按**重跑 stage3 的比例**对齐（25%/30%、50%/50%），不是按步数

⚠ **LIBERO 线的 GR00T 是 8 步（`groot_n15_k8_v1`），本线是 4 步（`groot_n15_k4_v1`）**，已逐层核实
（库 manifest / 原始 h5 的 `denoising_num_steps` / 每步 noise_action 数）。采集器每步断言 hook 触发次数
等于在跑的头的步数，所以这不是标签而是被验过的事实。两线共用 `cache/groot/*`，schedule 主键必须带步数。

**RIT 的形态**：s-only 的风险指数阶梯。**一个容忍度 δ 通过嵌套求逆同时解出所有档的切点**
（LP 在 knot 上解 q(s)、段内线性插值、`cut_at` 在同一条 PL 上解析求逆）。部署形态是
**`threshold` judge + `warm_tiers`**（GR00T 的 `load_guard._ALLOWED_JUDGE_TYPES` 不含 `dispatch_surface`）。
每个 arm 带 **H-gate**（`score_hysteresis`，`theta_low=theta_high=θ`，`j=3 / probe_interval=3 / L=6`）。

**术语（我曾混用，已统一）**：**段** = knot 分段（`KNOT_LADDER=(24,12,6)` 是候选段数的回退列表）；
**档** = RIT 阶梯的 tier（FULL_HIT / WARM@…），也就是 k。

---

## 2. 冻结口径（owner 裁定）

| 项 | 值 |
|---|---|
| 库 | 只用 **S6 / full**（每任务 50 条、共 650 条） |
| 检索 | text-IVF 开（`text_ivf_knn` + `index_type: text_ivf` + `prompt_emb {enabled, weight 0}`） |
| 权重 | GR00T `grid3_vision_0@12_vision_2@37_robot_state@50`；pi0.5 `grid_vision_1@87_robot_state@12` |
| 帕累托 | **6 个点** |
| 部署/报告 | 用**预测**切点发 yaml，用**实测** IR + SR 出图 |
| H-gate | 只装正式 arm，标定跑不装 |
| 评测 | **每任务 50 集、每点 650 集**；13 arm + 1 teacher-only 参考臂 |
| 场景 | `layout=1, style=1`，`replan_steps=5`，eval seed `1,000,000+idx`；建库段 `base_seed=0`（零重叠） |
| 顺序 | GR00T → pi0.5 → **k=1 双 teacher 最后补** |

---

## 3. 拓扑

server 只在 **weilandserver**（公网 1:1 NAT `ziyanglin.com:23100-23199`，服务必须监听段内），
worker 只在 **timan107**。⚠ 两个 teacher **不能同时占卡**，切换前必须杀干净。

### 现在（pi0.5 v2 主跑）
- **一个 server**：`scripts/serve_policy.py --replicas 4 --replica-spawn-batch 1 --port 23170 --cache-config <boot> policy:checkpoint --policy.config pi05_robocasa --policy.dir /home/weiland/ckpt_pi05_robocasa_pytorch`，tmux `ritp0`。
  router 绑 23170，child 绑 **23171-74**（`internal_ports = port+1+i`），整块要先查空。
  RAM 每 replica ≈ 28.7 GB。脚本 `/tmp/rit/start_pi05_eval.sh`。
- ⚠ **起之前必须顺序读一遍 7.2 GB 权重 + 28 GB 库预热页缓存**（都在 CMR 机械盘 `/data`；
  `/home/weiland/ckpt_pi05_robocasa_pytorch` 是指向 `/data/ckpt/` 的软链）。脚本已内置。
- **timan107 24 worker**：单端点 ⇒ `--role all`（`len(servers)==1` 时 ws2 不要求 `--agent-server`）。tmux `ritdrv`。
- ⚠ worker 数是权衡：20 太慢（6.5 集/分）、30 会把 4 个 replica 压到 keepalive 超时（teacher-pnp 那段丢 44 集）。**24**。

### 之后（GR00T k=1）
- 5 个 `serve_groot_n15.py --concurrent --allow-dynamic-bundles --cache-config <boot>`，端口 **23160-64**，tmux `rite0..4`，
  隔离工作区 `/tmp/openpi-stageA`，ckpt `/home/weiland/ckpt_n15_robocasa_tp/.../checkpoint-60000`，
  venv `/home/weiland/gr00t_n15_venv/.venv`。脚本 `/tmp/rit/start_groot_eval.sh 5 23160`（需 `BOOT=`）。
- worker 侧回到 **1 driver + 5 agent × 6 worker**（多 server 池必须 `--agent-server`）。

---

## 4. 监控体系（职责不得混淆）

- **cron = 定时巡检**：每 17 分钟，两台机查 GPU/tmux/端口/日志尾/产物计数，健康只在主对话记**一行**。id `158639e2`。
- **Monitor = 条件触发**：只在阶段完成/失败出声。当前 `bj3g2v14h`（pi0.5 v2 链的段边界，persistent）。
- ⚠ Monitor `timeout_ms` 上限 1 h，长跑必须 `persistent: true`。
- ⚠ 过滤器要按**高水位**出声（比较 `[chain]` 行数），否则每轮重推历史行、刷屏又费 context。
- ⚠ 判 server 就绪不要查 tmux 会话是否存在——`start_pi05_eval.sh` 先预热页缓存**再**建会话，
  会误报"消失"。应查端口，且只在"启动脚本已退出**且**无 server 会话"时才判失败。
- 远端长跑一律 tmux + tee（`tether exec` 单次约 10 min 上限）。判 driver 退出看 `RIT_DRIVER_EXIT`。

---

## 5. 进度：做完了 / 正在做 / 之后做

```
P0  成本模型 / 成本权威 / RIT 链路移植   [##########] DONE
P1  GR00T k=2/k=3/teacher                [##########] DONE  8,450+650 集，已出图，前沿有效
P2  pi0.5
    · 标定采集 130 集 + 离线建表 13,964 行 [##########] DONE  可复用
    · 第一轮拟合发射 + 主跑 8,450 集      [xxxxxxxxxx] 作废  寻址失效（§7）
    · 诊断 + coverage pilot 720 集        [##########] DONE  证实按部署分布定切点可铺满 26→100
    · v2 发射（合池 knot + 部署寻址）     [##########] DONE  13 arm，切点已验证能切开部署分布
    · ★ v2 正式主跑 9,100 集              [#.........] 跑中  23:34:46 起，约 17 h
P3  四线帕累托图                          [..........] 待 P2
P4  k=1 双 teacher                        [....------] 暂停  GR00T 冻在 870/2800，pi0.5 待做
```

**v2 主跑链** `tmux ritchain` 跑 `/tmp/rit/chain_v2_pi05.sh`，四段自动接：
A 阶梯 main 5,200 → B 阶梯 pnp 3,250 → C teacher main 400 → D teacher pnp 250，末尾 `CHAIN_V2_PI05_DONE`。

### 链跑完后
1. `bash ~/tmp_rit/finish_pi05.sh`（改 data 目录为 `rit_v2/pi05`）→ 汇总两 lane → 融合 13 任务前沿 → teacher 参考线 → 生成 spec → 渲染 + 四线图
2. 杀 pi0.5 server，起 5 个 GR00T server，恢复 k=1（journal 续跑，只补缺口）
3. pi0.5 的 k=1（7 cell × 650 = 4,550）

---

## 6. GR00T 最终结果（50 集/任务，13 任务，n=650/arm，两 lane 各 13/13 complete，n_err/n_missing 全 0）

| arm | realized IR | macro SR |
|---|---|---|
| all-FULL_HIT | 42.73 | 0.4677 |
| k2 IR23→100 | 43.6 / 52.7 / 63.6 / 67.4 / 81.4 / 100.0 | 0.451 / 0.494 / 0.506 / 0.632 / **0.648** / 0.645 |
| k3 IR23→100 | 43.1 / 53.2 / 60.5 / 67.4 / 85.0 / 100.0 | 0.480 / 0.477 / 0.548 / 0.614 / **0.660** / 0.657 |
| **teacher-only** | — | **0.6800**（contact-8 0.6400 / pick-5 0.7440） |

图 `analysis/figures/rit_groot_all13.{png,pdf,json}`。⚠ **spec 被 owner 手调过 13 个坐标，只重渲染、不重生成。**

⚠ 便宜端实测 IR 高于目标（23.0→43.6）：H-gate 跳过 25.6% 的步、强制全量推理，**可达 IR 下界 42.73**，
最便宜的两个目标够不着。

### 逐任务为什么有点在参考线上方（owner 问过，已查清）
- 参考线没问题：IR=100 的 arm 实测 99.97%/99.82% 全 MISS，逐任务与 teacher 偏差均值 −0.029、sd 0.051，
  而 n=50 的二项 SE 是 0.069 ⇒ 全在噪声内
- 三类：**真赢** CloseFridge(r=+0.93,t=8.1)/TurnOnSinkFaucet(+0.85,5.3)/OpenDrawer(+0.80,4.4)；
  **纯噪声** OpenStandMixerHead(−0.17)/SlideDishwasherRack(+0.25)（"8/13 在线上方"正是等价时的抛硬币期望 6.5/13）；
  **明确输** 其余 8 个（r −0.68 ~ −0.93）。r 是组内 FULL_HIT 占比与 SR 的相关
- 机制：`gain = 0.488 − 1.029 × teacher_SR`（r=−0.69）。斜率 −1.03 ⇒ **全命中臂的逐任务 SR 与 teacher
  自己的几乎不相关**（均值约 0.47）。库是**成功过滤**过的（建库器丢掉 success=false 的集），所以 cache 是
  把 teacher 的任务专属能力换成一份与任务难度无关的能力。反例 PickPlaceCounterToCabinet（teacher 0.42 但
  r=−0.81）⇒ 第二因子是该任务上检索本身好不好

---

## 7. pi0.5 第一轮为什么作废（全部实测，非推断）

**标定数据本身没坏，是 off-policy。** 测的量对，测的分布错。

教师驱动时分数随 episode 衰减，缓存驱动时不衰减：

| 步数区间 | pi0.5 标定 | pi0.5 部署 | GR00T 标定 | GR00T 部署 |
|---|---|---|---|---|
| 0–9 | 0.9854 | 0.9976 | 0.9979 | 0.9983 |
| 90+ | **0.8490** | 0.9958 | 0.9834 | 0.9953 |
| 全程衰减 | **−0.1364** | −0.0018 | −0.0145 | −0.0030 |

下游两处都锚死在这条错分布上：

1. **knot 按标定分位铺** ⇒ 25 段里 23 段在 0.99 以下，而部署 **73.8%** 的质量在 0.99 以上。
   最后两段分别装了部署的 **28.40%** 和 **44.84%** —— δ 求逆在部署真正工作的区域是跨大间隔插值，**没有分辨率**。
2. **δ→IR 在标定样本上换算** ⇒ 6 个切点把部署分布切成 100/98.6/92.3/84.3/83.9/2.6%，4 个压在 84% 以上，等于没切。

结果：6 个点里 5 个挤在 realized IR 26.8–39.9，第 6 个跳到 97，前沿塌成两点间一条直线。

**GR00T 为什么没事**：教师轨迹几乎不漂（衰减是 pi0.5 的 1/10），标定中位 0.9962 与部署 0.9968 差 0.0005，
切点切出 100/80.3/61.4/47.0/32.3/1.5%。
**为什么同一设计两个 teacher 结果相反**：pi0.5 嵌入空间挤 4 倍（库内余弦 0.9663±0.0085 vs GR00T 0.850±0.034），
同样的物理漂移换算成**分位**就大得多。

### 排除项（都查过，都不是原因）
- **pkl 没坏**：无 NaN、无零范数、无塌缩维；逐字段同任务/跨任务可分性（差÷标准差）
  pi0.5 vision_0/1/2 = 2.01/1.41/2.06，GR00T = 2.47/2.60/1.16，同量级
- **离线建表没有仪器误差**：探针精确复现 shadow 表（CloseBlenderLid ep0-4：0.9736→0.9637 两边一致）
- **归一化器饱和**：**我提出过又撤回**。我曾从"96.3% 的融合分数 ≥ 0.875"推出"vision_1 恒定饱和"，
  但分解不唯一（0.875×0.94+0.125×0.85 也能越过 0.875），推理不成立。第 0 步配对比较（同一批 130 个初始状态、
  不可能有轨迹偏移）显示离线仅偏低 0.0089，占不到总偏移 0.0728 的 1/8

---

## 8. pi0.5 v2 的修法（正在跑的这一轮）

**RIT 的构造完全不动**：仍是一个 δ 通过嵌套求逆解出所有档的切点。改的只是**拟合与寻址所依据的分布**。

1. **knot 铺在合池分布上**（标定 13,964 ∪ 部署 67,466 = 81,430），但**占用检查仍落在标定行上**
   （新函数 `choose_knots_pooled`：分位取自要应用的分布，行数下限落在拟合用的数据上）。
   选 **12 段**（不是默认的 24）：

   | 段数 | 0.99 以上段数 | 每段标定行（最小/中位） |
   |---|---|---|
   | 24 | 16 | 20 / 87 ← 太薄，α=0.05 尾部只剩 1 个点 |
   | **12** | **8** | **53 / 168** ← 选这个 |
   | 6 | 4 | 110 / 388 |
   | 旧（标定分位） | **2** | 582 / 582 |

2. **δ→IR 在实测部署样本上求逆**（`fit_ladders(..., ir_sample=...)`），且把 **gate 跳过的 11,426 步
   当作低于所有切点的哨兵**拼进样本，让它们必然落 MISS —— 预测 IR 与实测同口径。
   可达下界因此从不可达的 14.46 修正成真实的 **26.85**。

**结果**：新 IR 网格 26.85 / 41.47 / 56.09 / 70.71 / 85.33 / 99.95，切点在部署分布上的占比

| 目标 IR | 26.8 | 41.5 | 56.1 | 70.7 | 85.3 | 100 |
|---|---|---|---|---|---|---|
| 旧 | 85.5% | 84.3% | 78.9% | 72.1% | 71.7% | 2.0% |
| **新** | 85.5% | **67.4%** | **46.8%** | **32.7%** | **5.8%** | 0.0% |

### coverage pilot（720 集，验证寻址）
| 切点 | 0.28151 | 0.96954 | 0.99166 | 0.99577 | 0.99682 | 0.99729 | 0.99757 | 0.99792 |
|---|---|---|---|---|---|---|---|---|
| **实测 IR** | 26.4 | 30.5 | 39.4 | 46.1 | 54.3 | 74.6 | 88.5 | **100.0** |

单调铺满。⚠ 残差：实测比目标低 8–14 点（带阈值的 arm 命中率**高于** always_hit 参照；
不动点方向与我先前的预测**相反**，我预测过超调，实际是欠调，已更正）。

### 要写进报告的口径
- 切点来自**被作废那轮的部署分数分布**，而那轮跑的是同一段 650 集。是**分数聚合量**不是成功率，污染轻，但要写明。
- 一轮迭代不是不动点：中间阈值的 arm 有自己的分布。合池后 knot 覆盖两极之间全部范围，中间 arm 落在有分辨率的区域内。
- q(s) 仍在**教师驱动**的 (s,y) 对上拟合 ⇒ 风险语义仍是 off-policy。彻底修需要"cache 开车 + 教师只算不执行"的
  在线 shadow（`ShadowTeacherRecorder` 是现成的种子，但它只给 FULL_HIT 一档的偏差；warm 档要在每个状态上把
  库里的 x_t 续跑，需要完整 stage-1，而 gate-research 采集导出的查询键是池化过的 32768 维，重建不出 stage-1）。
  **这是第二步，只在高分段拟合太噪时才做。**

---

## 9. 资产

### weilandserver
| 内容 | 路径 |
|---|---|
| W13 pkl | `/data/robocasa365_cache/cache_artifacts_w13/`（pi05 full 28 GB / groot full 19 GB） |
| GR00T 标定 h5 | `/data/robocasa365_cache/calib_rit_w13/groot_tp/` |
| pi0.5 标定 h5（33 GB） | `/data/robocasa365_cache/calib_rit_w13_pi05/pi05/<Task>/` |
| shadow 标定行 | groot 12,817 / pi05 13,964 —— `/data/robocasa365_cache/rit_calib/{groot_tp,pi05}/shadow_all.jsonl` |
| 成本记录 | `/tmp/rc_cost_groot_tp.json`、`/tmp/rit/rc_cost_pi05.json` |
| 拟合记录 | `/tmp/rit/rit_record_{groot_tp,pi05,pi05_v2,k1_groot_tp}.json` |
| arm 树 | `/tmp/rit/arms{,_pi05,_pi05_v2,_teacher,_teacher_pi05,_k1_groot_tp,_pilot_pi05}/` |
| **寻址/铺点样本** | `/tmp/rit/ir_scores_pi05.json`（78,892，含 11,426 哨兵）、`/tmp/rit/knot_scores_pi05.json`（81,430） |
| 隔离工作区 | GR00T `/tmp/openpi-stageA`；pi0.5 用 `/home/weiland/openpi` |

⚠ `/tmp` 重启即清空；关键记录本地副本在 `~/tmp_rit/`。

### timan107（`REPO=/scratch/zixuans8/openpi_rc365`）
| 内容 | 路径 |
|---|---|
| arm yaml | `config/rit/{groot_tp,pi05}/main`、`config/rit_v2/pi05/main`、`config/rit_k1/groot_tp/main`、`config/rit_teacher/{groot_tp,pi05}/main`、`config/rit_pilot/pi05/main` |
| 产物 | `data/rit/{groot_tp,pi05}/…`、`data/rit_v2/pi05/…`、`data/rit_k1/groot_tp/…`、`data/rit_teacher/…`、`data/rit_pilot/pi05/…` |
| env | `config/collect_calib_timan107.env`（`PI05_SERVERS` 已含 5 个端点） |
| 脚本 | `/tmp/rit/*.sh`（`calib_common.sh` 是公共参数） |

⚠ `tether pull` 只认 allow_roots（`/home /tmp /srv`），`/scratch` 下要先 `cp` 到 `/tmp`。

### 本地 `~/tmp_rit/`
`finish_groot.sh` / `finish_pi05.sh` / `fuse_lanes.py` / 各 `frontier_*.json` / `rit_record_*.json` / `rc_cost_*.json` / `teacher_ref_groot_50ep.json`

---

## 10. 本轮代码

| 文件 | 作用 |
|---|---|
| `rit_cost_rc.py` | RoboCasa 成本权威；LP 拟合/PL 求逆/knot 阶梯全部 import 自 `exp.rit_pareto.rit_k` |
| `emit_rit_rc.py` | `calib`/`arms`/`teacher`；**新增** `choose_knots_pooled` + `--ir-scores` / `--knot-scores` / `--knot-ladder` |
| `emit_cuts_rc.py` | 按显式切点发射（pilot 用）；**已补写 provenance.json** |
| `build_rit_table_pi05_rc.py` | pi0.5 的离线建表 |
| `summarize_rit_rc.py` / `plot_rit_pareto.py` | 汇总 / 旧版逐 teacher 图 |
| `bench_pi05_stages_rc.py` | pi0.5 三段延迟（含 CUDA graph 边界与对表硬门） |
| **`build_rit_figure_spec.py`** | 实测前沿 → 可编辑 spec（`robocasa365.rit_figure/v1`） |
| **`edit_rit_figure.py` + `rit_figure_editor.html`** | 本地点编辑器，**无命令行参数**，`127.0.0.1:8765`，**owner 自己启动** |
| **`render_rit_figure.py`** / **`plot_rit_pareto_four.py`** | 从 spec 渲染 / 四线图，均**无命令行参数** |
| `probe_field_decay_pi05.py` / `probe_raw_sim.py` / `refit_normalizer.py` | 诊断探针（字段分解 / 原始相似度 / 归一化器重拟合） |
| `serve_groot_n15.py`(改) / `run_ws_search2.py`(改) / `episode_runner.py`(改) | H-gate 放行 / `rit` 前缀 + `--index-provenance` / per_step 补 `start_t` |

**出图流程（唯一来源）**：实测前沿 → `build_rit_figure_spec` → spec JSON →（owner 网页拖点）→
`render_rit_figure` / `plot_rit_pareto_four`。大图**派生**自 13 个小图：`y = 13 个逐任务 SR 均值`，
`x = Σ(w·x)/Σw`（w = 该任务判决行数）。连线只连非受支配点。小图 y 吸附 1/50 网格，服务端保存前再验整数。

已提交：`5ab6084`、`ff3340b`、`c0dac3a`。**其余全部未提交**（须 owner 当次明示）。
⚠ `exp/rit_pareto/{build,edit}_figure.py` 按 owner 指示排除在提交外。

---

## 11. 踩过的坑

1. **两条 lane 共用 `--data-dir`/`--run-plan-dir`** ⇒ run plan 按 cell id 命名不含 lane，撞 plan_hash。**每 lane 一个目录**。
2. **`rc=$?` 取的是 `tee` 的退出码** ⇒ `set -o pipefail` + `${PIPESTATUS[0]}`。
3. **judge 的 yaml 键是 `threshold` 不是 `cp1_threshold`**；写错的键被**接受并忽略**，静默用默认 0.98。
4. **loader 要求切点严格递减**；拟合会把两档放同一切点 ⇒ **摘除并记录**（`dropped_rungs`）。
5. **多 server 池的 `--role agent` 需要 `--agent-server`**；**单 server 时 `--role all` 可用**。
6. **`task_uid` 解析**：任务名从**最后一个** `__` 切，cid 按 `__l1s1` 切。
7. **一个 yaml 绑一台 server**（`assign_servers`）⇒ 收尾只剩一个 cell 时并行度掉到 1/N。
8. **`pgrep -f` 自匹配**会杀掉 tether shell；只杀 `readlink /proc/<pid>/exe` 是 python 的进程。
9. **`validate_teacher_endpoints` 按 env 的 `<TEACHER>_SERVERS` 组卡端点**：pi0.5 标定要 5 个单连接 server 而
   env 只声明 4 个 ⇒ 建图前就拒、零产物。已补 23174。
10. **续补只带 journal 不带 run plan**（run plan 按 episodes 入哈希）；但 **per_step 也不会带过去** ⇒
    汇总前要把两段 per_step 拼起来，否则 SR 覆盖全集而 realized IR 只覆盖新段。
11. **`summarize_journal` 的三分口径**：`done`=成功；`failed`+`error=None`=**失败但有效，计入分母得 0 分**；
    `error` 非空=ERR，排除出分母；**完全没有终态记录**=MISSING（重试耗尽）。
    ⚠ 我一度把 59 个有效 `failed` 当成故障删掉重跑 —— 真正的损失只有 44 个 MISSING。
    补跑脚本要**只丢带真实 `error` 的行**。
12. **teacher-only 臂每步全量推理**，是阶梯臂 4–5 倍的推理负载（阶梯臂大量 FULL_HIT 不进模型）。
    30 worker 压 4 replica 会把 stage3 从 30.5 ms 拖到 1057 ms，客户端 keepalive 超时、重试耗尽落成 `failed`。
    ⚠ 而 `done_uids()` 把 `failed` 也当终态 ⇒ 直接 relaunch 会跳过要补的。
13. **网页拖点被钉在右下角**：`render()` 重建整个 grid ⇒ 拖动闭包里的面板 div 已 detached，
    `getBoundingClientRect()` 全 0，`(clientX-0)/0`=Infinity。修法：卡片只建一次、只重画 svg；
    矩形在 pointerdown 量一次全程复用。
14. **CUDA graph 下量分阶段延迟**：不打 `cudagraph_mark_step_begin()` ⇒ 退回 eager，把 10 步循环量成 114.7 ms
    （真值 30.5）；只加边界 ⇒ 复用的 stage2 被图内存池回收而报 "accessing tensor output of CUDAGraphs…"。
    终版每样本跑完整 stage1→stage2→stage3(k) 三连、边界打在三连之前、只对 stage3 计时；并加硬门：
    拟合的 s3(10) 与 infer 路径相差 >10% 就拒绝写记录。
15. **`emit_cuts_rc.py` 曾漏写 `provenance.json`** ⇒ pnp lane 的冻结集断言在建图前就 FileNotFoundError，
    整条 lane 零产物（pilot 的 pnp 段就是这么丢的）。已修。
16. **`CacheStorage.last_step_features` 是方法不是 property**；`per_field` 诊断只在 weighted-score-sum
    融合路径上填充，其它路径返回计数级空壳。
17. **HDD 并发 mmap**：7.2 GB 权重 + 28 GB 库都在 CMR 盘，多 replica 同时 mmap 会把顺序流打成随机 IO。
    起服务前先顺序读一遍预热，`--replica-spawn-batch 1`。
