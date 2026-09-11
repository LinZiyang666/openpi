# LIBERO × GR00T — RIT / GST 主跑进度板

> 开跑 2026-09-11。上一条线（RoboCasa365 × warm-start RIT）已收官，本线复用它的
> 拟合与成本机件（`exp/robocasa365/rit_shadow.py` + `rit_cost_rc.py` + `exp/rit_pareto/rit_k.py`），
> 换掉的只有库、模板与评测驱动。

## 1. 口径（owner 裁定）

- **一式两份**：libero_spatial 与 libero_10。
- **库**：`<suite>_w13_S3.pkl` —— W13 重采集（带去噪快照，`groot_n15_k8_v1`），
  **5 条成功轨迹/任务 × 10 = 50 条**。与 pi0.5 线的库同量级（1,018 / 2,496 entries
  对我们的 1,078 / 2,598）。
- **阶梯**：K=1 = hit/miss；K=2 加 warm@剩 2 步；K=3 再加 warm@剩 4 步。
  8 步升序 schedule 下即 `start_t = 0.75` 与 `0.5`，与 RoboCasa 线的 25% / 50% 同分数。
- **门**：只跑 hg 一层 —— `score_hysteresis`，θ = warmup 分数 0.85 分位，j=3 / probe_interval=3 / L=6。
  ⚠ 两个锚点臂**不挂门**：门的 L 锁定每 L 步强制一次教师调用，会让"纯 cache"不纯，
  也会移动整条前沿的参照端点。
- **规模**：每 suite 84 臂 × 500 集（A 池 `pruned_init` 全量）= 42,000 集；两 suite 84,000 集。
- **密度看齐 pi0.5**：RIT 每个 K 16 个 IR 点（20…95 步长 5）；GST 34 个 cell（步长 20 单纯形，和 ≤ 80）。

## 2. 拓扑

按 `devices.md` 的规格表：LIBERO 用 GR00T 教师 ⇒ **5 replica、每台 timan 64 worker**。

| lane | server | worker | 入口 | 跑什么 |
|---|---|---|---|---|
| A | h100 ×**5** 进程 | timan108 **64** worker（3 卡） | `149.165.153.233:23210-23214` | libero_spatial |
| B | weilandserver ×**5** 进程 | timan107 **64** worker（8 卡） | `ziyanglin.com:23150-23154` | libero_10 |

⚠ 我一开始起了 4 replica / 24 与 48 worker，两项都低于规格（replica 数从 shadow 阶段顺手带过来
没查表，worker 数是对 timan108 过度保守）。清单里"每 worker 1.5-1.6 GiB"是 **RoboCasa 厨房场景**
的数；**LIBERO 实测 539 MiB/worker**，所以 64 × 0.539 = 34.5 GiB，timan108 占 72 GiB 的 48%、
timan107 占 64 GiB 的 54%，都在 75% 上限内。

⚠ **worker 是亲和绑定的**：一个 worker 只接自己绑定的那台 server 上的 stage。所以 anchor 阶段
（只有 2 个臂 = 2 个 stage）必然有大半 worker 空转、对应的卡上连 EGL 上下文都没有 —— 这不是故障，
82 臂的 hg 矩阵一开跑就全满。别把它误判成 EGL 绑卡失败（我误判过一次，逐 id 实测后排除）。

⚠ **timan108 现在只有 3 张 A5000**：第 4 张（PCI `c1:00.0`）2026-09-11 15:16:45 报
`NVRM: RmInitAdapter failed! (0x22:0x56:762)`，lspci 仍看得见卡但驱动初始化不了它。
故障前一小时的逐卡体检（含 EGL 渲染与跨卡一致性 digest）四张全好，故障时负载仅
每卡 8 worker / 4.3 GB / 利用率 1-6%。owner 裁定按 3 张稳定状态继续。

⚠ **EGL 绑卡**：`CUDA_VISIBLE_DEVICES` 搬不动渲染上下文，MuJoCo 按 `MUJOCO_EGL_DEVICE_ID`
选设备，不设就**全部挤在同一张卡**（实测 timan108 上 32 个 worker 的上下文全在 GPU 2，
每个 539 MiB）。`run_gtp` 新增 `--bind-egl-device`（默认关，保持既有跑法字节一致）。

## 3. 阶段与状态

| 阶段 | 状态 |
|---|---|
| 归一化重标（w13_S3） | ✅ 与旧标定几乎重合（Δmu ≤ 0.019），方法仍全选 `zscore(tanh)` |
| shadow cohort（150 集/suite） | ✅ 与 pi0.5 线**逐字节同一批 init**（seed 20260901，fit 5 + cal 10） |
| warmup / shadow 采集 | ✅ 两 suite 各 150 集，已逐项审计（见 §4） |
| 臂 emit（锚点 + GST，36/suite） | ✅ 两 suite 都过 `run_gtp.validate_arms` |
| 臂 emit（RIT 48/suite） | ✅ owner 2026-09-11 裁定：先用 RoboCasa 的成本口径算 IR，台账做完再更新 |
| 主跑 anchor 矩阵 | 🔄 两 lane 进行中 |
| 主跑 hg 矩阵（34 臂） | ⬚ |

## 4. warmup 审计结论

| | libero_spatial | libero_10 |
|---|---|---|
| 行 / 集 / 任务 | 3,432 / 150 / 10×15 | 8,383 / 150 / 10×15 |
| episode id | 0..149 无缺口 | 0..149 无缺口 |
| step_idx 连续 | 150/150 | 150/150 |
| 教师成功率 | **141/150 = 94.0%** | **138/150 = 92.0%** |
| winner 采集戳 | 全部 `20260908`（W13） | 全部 `20260908` |
| 不同 winner | 888（单集 14-31） | 2,130（单集 15-76） |
| 三档相等的行 | 0 | 0 |
| 分数 ≥ 0.99 占比 | 53.3% | **95.8%** |

教师成功率落在 GR00T 的官方水平，这同时排除了**夹爪约定**那个坑——它的表现是静默 0%。

## 5. 拟合结构（成本模型仍是临时借用的 RoboCasa 常数，只影响 IR 轴不影响风险拟合）

| suite | full 档跨分数降幅 | warm75 间隔 | warm50 间隔 | 16 个 IR 点里有死档的 |
|---|---|---|---|---|
| spatial | 12.2% | 0.365 | 0.390 | k=1:0, k=2:0, **k=3:10** |
| libero_10 | 6.1% | 0.261 | 0.313 | k=1:0, k=2:0, **k=3:8** |

⚠ **第三档在两个 suite 上都只比第二档多买到 0.025 / 0.051 的风险，却贵 6 ms**，
预算低端两个 warm 切点会塌到一起。RoboCasa 线同位置的收益是 0.337。
这是关于阶梯深度的**结果**，不是缺陷：emitter 会丢掉打不着的档并记录档名。

## 5b. IR 分离度 —— 这条线的生死线

pi0.5 v1 栽在"一格网格的臂全落在同一个预算上"，所以这一项是排程检查而不是收尾才看。
用**实测分数样本**预测每个臂的判据混合再计价（判据计数已落盘，重新计价不用重跑）：

| 网格 | 臂数 | IR 跨度 | 相邻间隔 最小/中位 | 相距 <1 点的相邻对 |
|---|---|---|---|---|
| RIT（**同一个 k 内**） | 16 × 3 | 20.0 – 95.0 | **4.95** / 5.00 | **0/15**（每个 k 都是） |
| GST | 34 | 33.7 – 94.9 | 0.000 / 1.31 | 4/33（精确并列） |

⚠ **跨 k 的三元组刻意落在同一个 IR 上**（IR=60 时 k1/k2/k3 = 59.983 / 60.011 / 59.998）——
这正是 IR 寻址的目的：同预算横向比阶梯深度。把整条 82 臂放一起看"中位间隔 0.02"是误读，
必须按 k 分组读。我第一次就是这么误报的。

GST 的 4 对精确并列是不同份额三元组给出同一成本，是同预算的重复点，可用来测"warm 质量
怎么分配"的影响。**GST 最便宜只到 33.7**（单纯形 `sum ≤ 80` ⇒ 至少 20% 步 MISS），
够不到 RIT 能覆盖的便宜端。

**计价链已对标准值验证**：`sp_anchor_cache` 无门、14,331 步全 FULL_HIT，实测 IR = **17.166**，
与理论下界 `stage1/miss = 8.12/47.30 = 17.166` 三位小数吻合。

工具：`exp/libero_groot/check_ir_separation.py`（排序后的实测 IR、相邻间隔、实测减寻址的漂移；
**均匀漂移是门的强制教师调用，随预算增大的漂移才是闭环协变量漂移**）。

## 5c. 吞吐与瓶颈（2026-09-11 实测）

**调度**：`run_gtp` 原本一个臂钉一台 server，而 worker 是亲和绑定的，所以臂数少于 server 数时
车队按比例空转（anchor 组实测 35/64 在忙）。已改成无条件用 `sharding.shard_eval_stage`
把每个臂扇成一台 server 一个兄弟 stage —— `make_task_uid` 不编码 server，重分片对续跑安全。
同时把 `MUJOCO_EGL_DEVICE_ID` 无条件绑到 worker 的 slot（`WorkerSpec.gpu_id` 的注释本来就叫它
"EGL slot binding"，只是 EGL 那一半没接）。**两处都不加开关**：这不是 GR00T 与 pi0.5 的差别。

| lane | server | 吞吐 | 每集 | server 功耗 |
|---|---|---|---|---|
| A h100 ×8 | 100% util | **39.9 集/分** | 61.1 s | 289/700 W（41%） |
| B weilandserver ×5 | 99% util | **18.8 集/分** | 240.6 s | 269/450 W（60%） |

⚠ **lane B 已排队饱和**：worker 从 35 个在忙增到 64 个，吞吐只涨 1.25×（15→18.8）而每集延迟涨
1.7×（140→240 s）。瓶颈是那张 4090，不是带宽（实测 5 MB/s / 链路 31 MB/s）也不是批大小
（`serve_groot_libero` **没有任何跨连接批处理**，`_InferLockedPolicy` 一把锁串行，batch 恒为 1）。
真并发度 = server 进程数。

**重平衡计划**：checkpoint 在 server 进程启动时加载死（热切换的只是 cache bundle），所以一条
lane 只能服务一个 suite ⇒ 顺序重平衡。spatial 约 17 h 跑完后，把 h100 + timan108 整体切到
libero_10，用 `run_gtp --arms` 分走剩余臂的一半。粗算总时长 37 h → **约 25 h**。

## 6. 踩过并已修的坑

1. `run_conductor.py` 的 `verify_warm_sweep` 只要看见 `warm_tiers` 就要求**每个** cell 是
   `always_warm_start` + `always_search`，会整目录拒掉我们的臂 ⇒ 改用 pi0.5 K=3 线的
   `run_gtp`（原生支持 `--judge-type threshold --warm-tiers --eval-gate`）。
2. `run_gtp` 不给 client 传 `--resize-size`，GR00T 必须 256（`WorkerSpec` 早已预留该字段）。
3. 我自己写的判据组装把 `f=0` 的 GST cell（34 个里 30 个）整条退化成全 MISS。
   正解是 FULL 阈值设在分数域之上（2.0，永不触发）、warm 档留在下面。已补回归测试。
4. LIBERO client 退出时 robosuite 在 `__del__` 里抛 EGL `EGL_NOT_INITIALIZED` ——
   良性，判据是"Traceback 数 > `in __del__` 数"才算真异常。

## 7. 监控

- **Monitor `bmq63sl4s`**（常驻，3 分钟一探，只在异常时出声）：server 掉线 / 节点失联 /
  硬故障行 / 进度连续两探不动 / GPU 掉 0 / **worker 机 GPU 枚举数变少**。
- **cron `04b4dfc3`**（每小时 :23 :53）：一行巡检。⚠ session 级，7 天后过期。
- 探针 `lg_probe.sh` 按 worker 机上跑的 tmux 自动判阶段（`lgeval` = 评测、`lgcli*` = 标定、都没有 = 空闲）。
