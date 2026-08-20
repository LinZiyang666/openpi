# Hybrid-gate threshold Pareto 重做（4 库 × 16 档 × 全 500 pruned_init）

> **Status**: `In Progress`（§4 Code 完成，实验发射中）
> **Level**: L2
> **Authority**: Execution
> **Date**: 2026-08-20
> **前序**: [`archive/weighted_sum_threshold_pareto.log.md`](archive/weighted_sum_threshold_pareto.log.md)（旧 threshold-pareto）、[`gate_stage3_n4_hybrid.log.md`](gate_stage3_n4_hybrid.log.md)（N4 混合门赢点 L=6）、[`cache_size_ablation_plan.log.md`](cache_size_ablation_plan.log.md)（A-pool 与 S3 库）
> **数据登记**: 四个库均在 [`exp/data_authority/records/`](../exp/data_authority/records/) 有台账

---

## 0. ⚠ 流程状态（记录在案）

owner 2026-08-20 下达：「有序开展实验，不做完不停」+「我已经为你腾出了机器，我暂时离开了，**你不得有任何阻塞实验的行为**」。据 WORKING_AGREEMENT 开篇 owner 绝对权限，本任务的 G1/G2 评审门在无人值守期间不得阻塞发射。

- 本 plan 与 §4 Code、发射并行落盘，非先于。
- 待裁决项一律取执行方判断的最优值，逐条记入 §7，全部便宜可逆。
- §6 Verify 照常跑；commit / push 仍待 owner 明确指令。

## 1. 问题

旧 threshold-pareto（33,200 ep）在 `always_search` 门 + 三档 verdict（FULL_HIT / WARM_START@0.5 / MISS）下扫出 SR × inference_ratio 前沿。本轮换掉两件事、扩掉一件事：

1. **门换成 N4 混合门（Hybrid L=6）** —— gate 线 Stage 3a 的 live 赢点，服务器端实现 `score_hysteresis` + `L=6`。
2. **warm_start 整条路线关闭** —— verdict 变二值，`f_FH` 成为唯一自由度。
3. **库从 1 个扩到 4 个**、评测集从 held-out 子集扩到**完整 A-pool 500**。

## 2. 设计

### 2.1 四臂（每个 pkl 跑一遍）

| arm | suite | 库 | entries | trajectories |
|---|---|---|---:|---:|
| `ws_sp` | libero_spatial | `exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl` | 1,018 | 49 |
| `ws_l10` | libero_10 | `exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl` | 2,640 | 50 |
| `cs_sp` | libero_spatial | `/data/openpi/.../cache_size_libero_spatial_all_S3.pkl` | 1,072 | 50 |
| `cs_l10` | libero_10 | `/data/openpi/.../cache_size_libero_10_all_S3.pkl` | 2,741 | 50 |

四者 keybuilder 同为 `cp1_spatial_pool_16`、`vector_dims` 逐字段相同。

### 2.2 检索栈：老 config，只 d=1

结构整体继承 gate 线的服务器端 N4 模板 `exp/gate_research/config/<suite>/n4_server/...__d1__*.yaml`（它已经带该套件的 d1 权重与 per-field zscore+tanh 归一化）。**只动三处**：`preload_path` / `gate` / `judge`。因此本实验与 gate 线的任何差异只能来自这三处。

### 2.3 门：Hybrid L=6

`gate: {type: score_hysteresis, theta_low: θ, theta_high: θ, j: 3, probe_interval: 3, L: 6}`。
j / probe_interval / L 取 Stage 3a live 赢点原值；**θ 每个库从自己的新 warmup 重解**（§2.5）。

### 2.4 判据：二值，16 档

`judge: {type: threshold, threshold: T_fh}`，**无 `warm_tiers`**。
`f_FH ∈ {0.05, 0.10, …, 0.80}` 共 16 档 —— 沿用旧实验的 f_FH 轴，f_WS 轴随 warm-start 一并消失（旧的 83 格三角网格塌成 16 格）。

**4 臂 × 16 档 × 500 ep = 32,000 episode**（旧实验 33,200，同量级）。

### 2.5 阈值与 θ 的反解

warmup 用**旧口径**（owner 明确指定，「最纯净」）：`gate: always_search` + `judge: threshold 2.0` 强制全 MISS ⇒ 机器人走真实 policy 轨迹，search 照常算出 `cp1_score` 并逐步落盘。

- `T_fh(f_FH)` = 该库自己 warmup 分布的降序分位切点。
- `θ_low = θ_high` = 同一分布**上 0.85 分位**的切点，**跨 16 档固定不动**。

θ 取 0.85 的依据：历史 N4 赢点 spatial 用 θ=0.968929，而该 base 的 `fh75_ws10` 档 WS 切点 = `derive_thresholds(scores, 0.85, 0)` = 0.968914 —— 两者差 1.5e-5，即历史 θ 就是这个分位。θ 固定不动是为了让 16 档之间的差异**只**归因于判据阈值；若 θ 跟着 `T_fh` 走，门与判据同时移动，任何档间差异都无法归因。

切点约定**复用** `exp/verdict_factor_judge/phase3/threshold_solver.derive_thresholds`，不另写一份 —— 一个舍入方式不同的第二实现会让本轮与历史全部阈值不可比。

### 2.6 评测集：完整 A-pool 500

10 task × 50 init = 500 ep/臂，绑定冻结的 A-pool 记录（`apool_libero_<suite>.yaml`），**digest 在 launch 时从磁盘重算**而不是读回记录（记录是主张，文件才是证据）。

**泄漏已实测排除**：weighted_sum 两库的建库 init（`db_init/libero_cache/<suite>`，5/task）与 A-pool 逐行比对 **0/50 命中**；cache_size 两库来自差集池，`apool_*.yaml` 内 `shared: 0` 逐任务断言。四臂在全 500 上都干净，且**同一评测集 ⇒ 四臂可配对**。因此旧 `run_phase2.py` 的 `--init-map` held-out 门本轮**不沿用**（它会把 episode 限制在子集里，与「跑全 500」直接冲突）。

## 3. 拓扑

| 角色 | 机器 | 配置 |
|---|---|---|
| server | weilandserver | `serve_policy --replicas 4 --port 8000`，tmux `srv0`，pi05_libero |
| 公网入口 | broker(pc732) | `linziyang.top:14009` → weilandserver:8000（`tether expose --name gtp-srv`） |
| client | timan107 | 64 worker，conda `/scratch/zixuans8/libero_sim` |

## 4. 代码（`src/` 零改动）

| 路径 | 作用 |
|---|---|
| `exp/gate_threshold_pareto/libraries.py` | 四个库 + 每套件模板 + A-pool 记录的**唯一**声明处 |
| `exp/gate_threshold_pareto/emit_gtp_yamls.py` | warmup(4) / eval(64) yaml 生成；每份过 `load_cache_config` + **断言 warm tier 缺席** |
| `exp/gate_threshold_pareto/solve_gtp.py` | warmup dump → `θ` + 16 档 `T_fh`；复用 phase3 切点约定 |
| `exp/gate_threshold_pareto/run_gtp.py` | conductor runner（warmup / eval 共用）；复用 cache_size 的 A-pool 重算与快照合并 |

**不复用 cache_size 的两道门**：其 pure-cache 校验要求 `judge: always_hit` + `gate: always_search`，会拒掉本轮每一个臂；其 FULL_HIT 见证断言的前提正是本轮故意打破的（有阈值的 verdict 本来就该服务 MISS 步）。

**本轮自己的门**：warm tier 缺席 / routing 缺席 / warmup 必须 `always_search`（被门跳过的步没有分数，反解出来的分位会描述一个被截断的分布）/ eval 必须 `score_hysteresis` **且 `L == 6`**（丢了 `L` 会退化成纯 N1，在结果里与保留 L 的臂不可区分）。

## 5. 分析

`inference_ratio` 的算法 owner 指明本轮改变、**暂缓**。per-step 落盘保留完整 `hit_type` / `searched` / `cp1_score`，任何口径都能事后重算，不必重跑。

## 5.1 执行记录（2026-08-20）

**拓扑落地**：weilandserver `srv0` 4 replica 全 READY（router 8000 → 8001-8004，GPU 31 GB/49 GB）；`tether expose gtp-srv` → **`linziyang.top:14009`**；timan107 `gtpe` 64 worker。

**库校验（发射前）**：四个 pkl 在 server 侧全部逐条内容核验。两个 weighted_sum 库的服务器副本比本地权威副本**大 8 字节、文件 sha256 不同**，但打包无关的内容摘要（id 排序后逐 entry + key 字节）**逐位相同**（spatial `a8781cbc…` / l10 `0a48d74b…`）—— 差异只在 pickle 封装层。已写入两条台账的 `replicas` 与 `caveats`。

**泄漏门（发射前实测）**：weighted_sum 两库建库 init 逐行比对 A-pool **0/50 命中**；cache_size 两库来自差集池。A-pool 在 timan107 侧 digest `cbbc73792ce546c9…` 与冻结记录逐位相符。

**warmup 实测（每库 100 ep，全 MISS）**：

| arm | n(scores) | distinct | range | θ |
|---|---:|---:|---|---|
| `gtpw_ws_sp` | 2,128 | 2,117 | [0.716997, 0.988533] | 0.977537 |
| `gtpw_cs_sp` | 2,079 | 2,061 | [0.772548, 0.988560] | 0.977772 |
| `gtpw_ws_l10` | 6,026 | 5,668 | [0.502607, 0.998787] | 0.994209 |
| `gtpw_cs_l10` | 6,284 | 5,926 | [0.610360, 0.998767] | 0.991817 |

四臂 `distinct` 均 ≫ 16 档，R3 前置门全过。64 份 eval yaml 已生成，`T_fh` 随 `f_FH` 单调下降、θ 跨 16 档恒定。

**吞吐**：warmup 全 MISS 下 spatial 48 ep/min、l10 19 ep/min；spatial eval 实测 **59 ep/min**（GPU 100%，403 W）⇒ 16,000 ep 约 4.5 h。

## 5.2 事故：resume 的 bundle 切换风暴打死 GPU（2026-08-20，两次复现）

**现象**：spatial eval 跑到 33.9%（10,841 ep）时按 owner 指令改走公网直连，需要重启 conductor。重启后 **1 分钟内** GPU 进入 `requires reset`，server 侧 21 条 `CUDA error: illegal memory access`，实验停摆。重启机器后再次恢复，**同样在 1 分钟内复现**。

**Xid 签名（两次完全一致）**：

```
Xid 31  MMU Fault: ENGINE GRAPHICS GPC4 GPCCLIENT_T1_* faulted @ 0x0_00000000
        Fault is of type FAULT_PDE ACCESS_TYPE_VIRT_READ
Xid 175 GSP RPC timeout（触发者是 nvidia-smi —— 监控探针撞上已卡死的 GPU，是结果不是原因）
Xid 154 GPU Reset Required
```

**根因**：resume 时 driver 遍历全部 32 个臂的 stage；已完成臂的 episode 虽被跳过，**但每个 stage 仍调用一次 `ctl.load_cache_config`**。日志证据：

| 日志行 | 事件 |
|---|---|
| 136–158 | **23 个 bundle 连续加载**（相邻行，几秒内切完） |
| 436 | 第一条 CUDA illegal memory access |
| 46220 / 84266 | 正常运行时的相邻两次切换——**相隔数万行、约 8 分钟** |

即：几秒内 23 次拆建 cache backend、反复载入 0.4–1.1 GB 库，同时 4 replica 并发服务 ⇒ use-after-free ⇒ 读空地址。与「同机同配置全新跑 4.5 h / 10,841 ep 零 Xid」完全自洽。

**显卡无责**：两次均干净恢复（`Recovery Action: None`、Xid 归零、0 MiB 残留），无 ECC 错误、无 Xid 48/63/64/79。两次都落在 GPC4 只是出错 warp 的调度位置，n=2 不足以指认硬件——本文早先版本据此怀疑硬件，**该推论已作废**。

**已落地的修复**（`run_gtp.py`，零 src 改动）：新增 `arms_with_work_left()`，启动时读 journal 按 **distinct `task_uid`** 统计，已完成的臂不进 TaskGraph ⇒ driver 不走其 stage ⇒ 不触发切换。实测生效：`resume: 21/32 arm(s) already complete, not walking their stages`。逃生开关 `--no-resume-filter` 保留（复现用）。

按行数而非 uid 计数会把重试过的臂误判为完成并跳过其欠账，测试 `test_resume_filter_counts_distinct_uids_not_lines` 钉住这一点。

**未修的根**：server 端 bundle 热切换没有 quiesce 在途请求。正确修法是切换前排空/挡住在途请求，或给 backend 加引用计数——属 `src/openpi/cache/` 改动，L2/L3，须走 Plan→G1→G2，**不在无人值守期间进行**。

**规程改正**：CUDA 进程一律先 `SIGTERM` 等待退出；skill §5.1 的 `pkill -9` 模板是**实验收尾**场景，不可用于**中途重启**。另：重启机器前必须先把 server 日志拉离 `/tmp`——本次 `/tmp/gtp_srv.log` 随重启被清除，是取证损失；server 日志已改落持久路径 `/home/weiland/gtp_logs/`。

## 6. 风险

| # | 风险 | 缓解 |
|---|---|---|
| R1 | 4 replica + 常驻库撑爆 4090 48 GB | 起后即查 `memory.free`；不足则降到 3 replica 重起（yaml 与 journal 不受影响，可续跑） |
| R2 | 走 broker 的吞吐惩罚 | 旧 threshold-pareto 同样经 broker，33,200 ep ≈ 3 h，可作参照 |
| R3 | 某库的 warmup 分布过窄，16 档切不开 | `solve_gtp` 前置门：可用分数 < 500 或**不同取值数 < 16** 直接 fail-fast |
| R4 | timan107 上的新文件未过 git | tether push 直传（owner 既有裁定）；里程碑收口时补 commit |

## 7. 执行假设（owner 未逐条答复，取判断值，逐条可逆）

| # | 项 | 取值 | 反悔代价 |
|---|---|---|---|
| 1 | replica 数 | **4**（owner 后一句「weilandserver 可以起 4 replica」覆盖首次指令的 3） | 改一个参数重起 server |
| 2 | θ 分位 | 上 0.85（复现历史 spatial 锚点到 1.5e-5） | 改 `THETA_TOP_FRACTION` 重解 + 重发 eval |
| 3 | warmup 规模 | 10 ep/task = 100 ep/库（等同旧实验每 base 100 ep） | 加跑 warmup 重解 |
| 4 | warmup init 池 | A-pool 前 10 init/task | 见 §8 局限 1 |
| 5 | 实验目录 | 新建 `exp/gate_threshold_pareto/`（不混进 weighted_sum） | `git mv` |

## 8. 局限

1. **warmup 与 eval 共用 A-pool**：阈值是在评测集的一个子集（每任务前 10 init）上标定的。这与旧 threshold-pareto 的做法一致（它的 warmup 与 eval 同样共用其 held-out 池），但确实是一次 test-set peeking。量级：16 个标量分位读自约 1,000 个样本。**更干净的替代**是用差集池 B 的一个不含任何库源 init 的切片——本轮未做，因为 cache_size 两库的 B-train 索引需要额外考据，会阻塞发射。留作重跑项。
2. `inference_ratio` 新口径未定，本轮只落盘不判读。
3. 四库两两之间 entries 相差最多 2.7×，SR 差异含库规模与库来源两个因子，本实验不做分解。

## Review Log
