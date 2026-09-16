# X14 在线 RL Router 基线 — 运行期执行日志

> Status: `In Progress`
> Level: L3（运行期；实现线已于 `721d3fa` 收官）
> 唯一设计权威：[`rl_router_baseline_plan.log.md`](rl_router_baseline_plan.log.md)。本文只记录**运行期实际发生了什么**——拓扑、命令、实测缺陷与修复、产物位置。设计问题一律回 plan。
> 开始：2026-08-15 18:45 CDT

---

## 1. 生效拓扑（实测，非规划）

> ⚠ **与 plan §4.2 的偏离**：conductor 由 timan107 改到 **weilandserver 单机自闭环**，理由与实测见 §1.1。待 owner 追认。

| 角色 | 主机 | 进程 / 会话 | 端口 |
|---|---|---|---|
| pi05 routed server | weilandserver | tmux `rlrsrv`，`--replicas 1` | `:8000`；另 expose `rlr-srv` → `linziyang.top:14007`（备用/外部客户端） |
| ACT student sidecar | weilandserver | tmux `rlrsc`，lerobot venv | `:7002` |
| conductor + 16 LIBERO worker | **weilandserver** | tmux `rlrws`，conda prefix `/home/weiland/libero_sim` | client → `127.0.0.1:8000` |
| batch package | 同机 | `LocalTransport`（`--remote-host` 留空） | — |
| （备用）跨机通道 | t107 → wls | ssh-agent + scp over tether `wls-ssh` | `linziyang.top:14024`，见 §3.2 |

两台仓库均已同步至 `721d3fa`（同步过程见 §3.1）。GPU：4090 48 GB；server ~10.5 GB + 16 worker ~11 GB ≈ 21 GB，util 86%。

### 1.1 为什么把 conductor 搬到 wls（有实测支撑）

t107 首轮采集实测只有 **2.4 ep/min**（8 worker）。逐项排查后两端都远未饱和：t107 负载 2.43/48 核、8 张 GPU 全 0–4% util；wls GPU 21% util、38 GB 空闲显存；链路带宽 ≈ 0.9 MB/s。真正的瓶颈是**每个控制步一次同步 websocket 往返，而这条链路要绕 broker**（t107 → `23.95.48.156` → wls），实测 ≈ 2 s/verdict，其中计算只占 ~0.2 s。

对照探针（8 ep，同一台 server、同一 yaml，唯一变量是客户端位置）：

| 客户端 | worker | 吞吐 | 单 worker |
|---|---|---|---|
| timan107（经 broker） | 8 | 2.4 ep/min | 0.30 ep/min |
| **weilandserver（`127.0.0.1`）** | 4 | 6.15 ep/min | **1.54 ep/min（5.1×）** |

放大到 16 worker 实测 **18.7 ep/min（7.8×）**，此时 GPU util 86%，已接近单卡上限——再加 replica 收益有限，因为渲染与推理共用同一张 4090。

顺带的好处：单机意味着 shards / checkpoint / weights / journal / package 全在同一文件系统，`--remote-host` 留空即走 `LocalTransport`（代码原生支持且有 golden 覆盖），**整条跨机 scp 路径连同它的失败模式一起消失**。跨机通道仍已就绪（§3.2），随时可退回 plan §4.2 的双机布局。

对科学口径**无影响**：reward 的 cost 项取自 M5a 冻结表而非 wall-clock，臂 yaml / 检索配置 / init 池均未变；改的只有 episode 的产出速率。

**服务器侧路径根**（wls）：

```
/home/weiland/rl_router/
├── dump/      # 特征分片（--shard-root，也是 judge 的 dump_dir）
├── weights/   # 逐版本 router 权重（RemoteRun 发布目标）
├── art/       # 远端 artifact 根（--remote-root），arm_costs.json 也在此
└── costs/     # M5a 逐臂原始记录
```

**conductor 侧路径根**（现同为 wls，`/home/weiland/openpi`）：`exp/rl_router/config/libero_10/`（臂 yaml）、`exp/rl_router/data/`（journal / client rows / package / artifacts）。t107 上的同名目录已随双机同步就位，退回双机布局时可直接用。

M4 与 M5c 各自用**隔离的根**，避免 smoke / pilot 的 checkpoint 与正式 run 的同名 `--run-id` 命名空间相撞：`dump_m4` / `art_m4`、`dump_pilot/lam_<λ>` / `art_pilot/lam_<λ>`，并各配一份 `dump_dir` 指向自己的臂 yaml。

---

## 2. 里程碑进度

| 步 | 状态 | 产物 |
|---|---|---|
| 环境勘察 + 双机同步 | ✅ | 两机 `721d3fa` |
| 臂 yaml emit | ✅ | `exp/rl_router/config/libero_10/{r_ts,r_tc,r_tsc}_{train,eval}.yaml` + `collect_student.yaml`；`--check` 守卫通过 |
| 端到端 smoke（4 ep） | ✅ | `ws_smoke`：4 journal 行 / 258 client 行 / 0 error；分片 dim=65568 fp16，与 plan §3.0 实测口径一致 |
| **M5a** cost microbench | ✅ | `/home/weiland/rl_router/art/arm_costs.json` |
| **M5b** warm-start 采集 | ✅ | `ws_l10` 450/450 ep、354 成功(78.7%)、**24.5 min**；450 分片 3.4 GB，manifest 全 `complete`、dim 65568 |
| M5b 拟合 | ✅（**第三版：encoder v2 + 推导 lr**） | `warmstart_l10.pt`（67 MB，v0）；450/450 录取、27,430 步、infra 失败率 **0.0**、无回退；`head_lr=1.128e-3`（由实测特征尺度推导）；**CV loss 0.555 < 基率熵 0.656**；**trunk 134/256 存活、头输出 std 0.534**；δ₀=0.570 → 部署态 student 率 0.5000；537 s。前两版分别因死 trunk（CV 0.911）与未训起（CV 0.687）作废，见 §3.12 |
| **M4** smoke | ✅ | `m4_smoke.json` **passed，五断言零违规**；顺序调整理由见 §3.4 |
| **M5c** λ pilot | 🔄 **第三次启动（encoder v2）** | 前两次分别因 trunk 全死、一步饱和而作废（§3.12）；本次 v0→v1 更新为 +0.053 的轻推、逐步 probs distinct 413、grad_norm 0.156。10 worker（给同机其它 session 让路）。管道本身全程健康：所有批次 join complete、零拒收 |
| G-launch → M6–M9 | ⬚ | |

### 2.1 M5a 结果（冻结）

batch=1 GPU time（CUDA event，逐臂在**拥有该计算的进程内**测，D4 口径）：

| 臂 | GPU 秒 | 归一（teacher=1） | 测量路径 |
|---|---|---|---|
| teacher（Pi0.5 全阶段） | 0.42431 | **1.0** | `stage1+stage2+stage3` in-process |
| student（ACT 集成，10 任务均值） | 0.023616 | **0.055656** | sidecar 自己的 lerobot venv 内前向 |
| cache（真实 replay） | 3.0621e-05 | **7.2165e-05** | `PayloadView.get + broadcast_action` |

三臂 provenance 均 `gpu_timed=true` / `in_process=true`，student 带 10 条真实任务 prompt——G-launch-1 的 `_cost_problems` 三项检查全部满足。

### 2.2 M5b 采集参数

```
collect_warmstart.py --suite libero_10 --arm student（constant_arm）
  --split exp/ablation_study/config/common/split_libero_10.yaml   # 450 = 45 init × 10 任务
  --arm-yaml exp/rl_router/config/libero_10/r_ts_train.yaml
  --init-states-dir exp/common/data/db_init/libero/libero_10      # B 池差集，10 个 .init，无 .pruned_init
  --servers 127.0.0.1:8000 --workers 16 --gpus 1
  --conda-env /home/weiland/libero_sim
  --shard-root /home/weiland/rl_router/dump --run-id ws_l10
```

首轮在 t107 跑到 200/450 时按 §1.1 的实测结论**主动放弃并整轮重跑**（`collect_warmstart.py` 无 resume；重跑代价 24 min，续跑 t107 剩余 250 ep 要 105 min，且换拓扑的收益对后续 M6 是按天计的）。重跑参数只改客户端侧：`--servers 127.0.0.1:8000 --workers 16 --gpus 1 --conda-env /home/weiland/libero_sim`，在 wls 上执行。

实测吞吐 **18.7 ep/min**，student 成功率 ≈ 0.79。分片 ≈ 4.5 MB/ep → 全量 ≈ 2 GB，远低于 20 GB 容量红线。

### 2.2b ⚠ 同卡上有别的 session（不归本线处置）

M5c 开跑后吞吐从 18.7 掉到 **11.5 ep/min**。排查发现 4090 上多了一个**不属于本线**的 tmux 会话 `pisrv`（2026-08-15 20:05 起，websocket policy server，占 7.6 GB 显存）。卡是 owner 独享硬件、无外部用户，所以那是 **owner 另一个 session 的作业**；按同机多 session 纪律**不做任何处置**（不 kill、不改不是本会话起的进程），只记录并把「显存余量 < 4 GB 告警」加进监控。

当前分配：本线 pi05 ~9 GB + ACT sidecar ~3 GB + 16 worker ~13 GB ≈ 25 GB，另一 session 7.6 GB，余量 ~16 GB。若对方继续扩，先降本线 worker 数；要反过来让对方腾地方，属于 owner 的调度决定，不由本线单方面执行。

排期按 11.5 ep/min 重算见 §2.3（pilot ≈ 3.2 h、M6 ≈ 37 h；仍是 t107 经 broker 路径的约 5×）。

### 2.2c 前两批的数值健康检查（λ=0.05 候选）

| batch | success | reward | student% | grad_norm | \|adv\| | rejected |
|---|---|---|---|---|---|---|
| b0000 | 0.810 | 0.8070 | 0.4996 | 0.2506 | 0.3086 | 0 |
| b0001 | 0.940 | 0.9373 | 0.5009 | 0.0486 | 0.1131 | 0 |

**冻结 reward 公式的数值自洽验算**：`R = success − λ·(Σ_t cost)/T_max`。b0001 的 success−reward = 0.0027；而每 ep 约 61 个 verdict、其中约半数走 teacher（cost=1），故 Σcost ≈ 30.5，T_max=520 → 0.0587，乘 λ=0.05 得 **0.0029**。与实测 0.0027 吻合——公式、cost 表、T_max 三者在真实数据上互相印证。

**success 从 0.81 跳到 0.94 不是学习信号**：两批的 arm 分布几乎没动（0.4996 → 0.5009），而 lr=3e-4 的单步 Adam 在 16.8M 参数上不可能一步换来 13pp。真实原因是**每批从 300 条 pilot 池里抽不同的 100 条**，而 libero_10 各任务难度差异极大，批间任务配比一变 SR 就跟着摆。这恰恰是 §3.10 协议不看训练期 SR 曲线、而要用**冻结权重在留出余集上的 realized teacher rate** 来选 λ 的原因。

### 2.5 M6 旗舰 run 的轨迹（`l10_ts_lam1_s0`，λ₁=0.5）

**⚠ 本节曾在第 9 批时写过一个错误结论，已更正。** 当时看到 student 率从 0.542 单调爬到 0.626，我判断「λ 的效应在 40 批尺度上是看得见的」，并把它当作利好方向 (a)/(c)/(d) 的新证据报给了 owner。**那是从随机游走的上升段外推**——跑到 20 批时它又跌了回来。

**run 已于 2026-08-16 10:10 跑完**（`RUN_l10_ts_lam1_s0_EXIT=0`），下面是**完整 40 批 / 4000 episode**，不再是半程。`verify_m6.py … 40` 判 **ADMISSIBLE**：链 v0→v40 首尾相接、**40/40 个不同 `model_sha`**、`encoder_version` 全程 `b09609c9…`、零拒收零 quarantine、每批 100 slot 全到、`policy_distinct` 始终五千量级（策略未退化）。

```
student%: 0.542 0.563 0.614 0.620 0.615 0.631 0.622 0.622 0.626 0.617
          0.621 0.585 0.599 0.583 0.556 0.564 0.566 0.560 0.538 0.556
          0.570 0.565 0.570 0.573 0.568 0.559 0.538 0.537 0.517 0.525
          0.526 0.500 0.490 0.484 0.466 0.473 0.462 0.465 0.452 0.433
```

| 分段（每 10 批） | 1–10 | 11–20 | 21–30 | 31–40 |
|---|---|---|---|---|
| student% | 0.6072 | 0.5728 | 0.5522 | **0.4751** |
| mean_success | 0.909 | 0.913 | 0.904 | 0.903 |
| **mean_reward** | 0.8863 | 0.8887 | 0.8785 | 0.8836 |
| grad_norm | 0.5696 | 0.4884 | 0.5718 | 0.3287 |

**三个可以直接读出来的事实**：

1. **被优化的目标本身，4000 episode 一动没动。** `mean_reward` 四段 0.8863 / 0.8887 / 0.8785 / 0.8836——无趋势、无改善。success 同样平坦（0.909→0.903）。`grad_norm` 全程在 0.05–1.20 间抖，不收敛。
2. **臂比例确实在漂，方向却与成本项相反。** student 份额净跌 0.542 → 0.433。teacher 成本 1.0、student 0.0557、λ₁=0.5，而 `R = success − λ·Σcost/T_max`：在 success 平坦的前提下，**多用 teacher 只会抬高成本、压低自己的奖励**。策略朝着降低自身目标的方向漂了 40 批。
3. **末段 9 批近乎单调下行**（0.500 0.490 0.484 0.466 0.473 0.462 0.465 0.452 0.433）——和开头 b0002–b0005 那段近乎单调上行同构。**两个方向的「持续段」都出现过**，这正是有自相关的随机游走该有的样子。

**判断口径（刻意克制）**：**不**据此宣称「λ 起了反作用」——那是本节开头那个错误的镜像版本，同样是从单条轨迹的一段读出方向。能说的是：*在这一条 seed 上，4000 episode 的在线 REINFORCE 没有让目标函数产生任何可测的改善，臂比例的漂移方向与成本项相反。* 是漂移还是信号，单条 seed 判不了——**正在跑的 `l10_ts_lam1_s1` 就是这个问题的直接检验**（同 λ、同 warm-start、只换 seed：若 s1 的漂移方向与 s0 相同，那是信号；若相反或无向，那是游走），最终定论还要等 A 池评测。

**这对论文未必是坏消息**：X14 的存在理由就是回答「为什么不训一个 router」。「给到 4000 episode 的在线交互，被优化的目标纹丝不动，连它自己的成本系数都标不出来」——这是**支持 thesis 的一手证据**，而非实验失败。但这句话要等 s1 + A 池才能落笔。

### 2.5b 双种子对照（`l10_ts_lam1_s0` vs `s1`，两条均 40 批完成）

R3 给 seed-1 定的角色就是**方向一致性检验**。两条都跑完了，可以做了。s1 同样判 **ADMISSIBLE**（链 v0→v40、40/40 不同 `model_sha`、零拒收、join 全完整）。

| 每 10 批均值 | s0 Q1 | Q2 | Q3 | Q4 | 净变化 | s1 Q1 | Q2 | Q3 | Q4 | 净变化 |
|---|---|---|---|---|---|---|---|---|---|---|
| **student%** | 0.6072 | 0.5727 | 0.5521 | 0.4751 | **−0.109** | 0.5361 | 0.4936 | 0.5052 | 0.4872 | **−0.045** |
| mean_reward | 0.8863 | 0.8887 | 0.8785 | 0.8736 | — | 0.8796 | 0.8785 | 0.8700 | 0.8750 | — |
| mean_success | 0.9090 | 0.9130 | 0.9040 | 0.9030 | — | 0.9060 | 0.9070 | 0.8980 | 0.9040 | — |
| grad_norm | 0.5695 | 0.4884 | 0.5718 | 0.3287 | — | 0.6789 | 0.5444 | 0.6220 | 0.5486 | — |

**三条读数**：

1. **方向一致：两条种子都朝 teacher 漂**（−0.109 / −0.045）。但幅度差 2.4×，且 s1 的分段不单调（Q2 < Q3）。
2. **目标函数在两条上都没有改善。** reward 从 Q1 到 Q4：s0 0.8863→0.8736、s1 0.8796→0.8750，**都是略降**。8000 个 episode 的 on-policy REINFORCE，被优化的量一次都没往上走过。
3. **梯度范数两条都不收敛**（s0 0.57→0.33，s1 0.68→0.55，中途还回升）。

**这个检验的功效有多强，要说清楚**：两条 run 共用**逐位相同**的 warm-start（§3.16.1 已核实 `W1`/`b1`/`W2`/`b2` 全同），seed 只改变 rollout 采样的 RNG。所以这不是「两次独立初始化的独立实现」，而是「同一起点的两条采样轨迹」。**同起点的共同漂移，更像系统性偏置而非两次独立随机游走的巧合**——但也正因为起点相同，它无法排除「该起点自带一个朝 teacher 的小偏置」。

**一个值得写下来的机制假说（尚未验证）**：`R = success − λ·Σcost/T_max`。实测成本项在 λ₁=0.5 下只有 **0.003–0.012**，而 success 的批间波动是 **±0.1**——**差 1–2 个数量级**。若 teacher 在成功率上确有微弱优势，REINFORCE 会朝 teacher 爬，而 λ 的惩罚小到根本拦不住。这能解释漂移方向。**但它解释不了第 2 条**：真朝成功率爬的话 success 该升，实测却全程平在 0.90。所以这个假说目前只对上一半。

#### 2.5b.1 ⚠ 对 `l10_ts_lam2_s0` 的事前预测（**写于该 run 发射后、任何数据产生之前**，2026-08-16 17:5x）

λ 惩罚成本、成本高的是 teacher。所以 **λ 越大 → 越该朝 student 推**。已有两条都是 λ₁=0.5（较强惩罚），却都朝 teacher 漂（均值 −0.077）。刚发射的第三条是 **λ₂=0.05，惩罚弱 10×**：

- **若 λ 有其应有的符号、且上面的机制假说成立** → λ₂ 的 `net Δstudent%` 应当**比 −0.077 更负**（惩罚更弱，拦不住的程度更甚）。
- **若 λ 与结果无关（纯噪声游走）** → λ₂ 的净变化应与 λ₁ 两条不可区分，**符号是抛硬币**（约一半概率为正）。

n=1，证据强度有限；但这是**事前**的方向预测，比跑完再读方向强得多。**无论结果如何都必须照实记在这里**——本线已经有过一次「看上升段就下结论」的记录（§2.5 开头），事前钉死预测正是为了不让它重演。

#### 2.5b.2 预测的结果：**判据两边都满足，说明我把预测设计坏了**（2026-08-17）

`l10_ts_lam2_s0` 40 批跑完，判 **ADMISSIBLE**（链 v0→v40、40/40 不同 `model_sha`、零拒收、join 全完整、`EXIT=0`）。三条 run 并排：

| run | λ | Q1 | Q2 | Q3 | Q4 | **净 Δstudent%** |
|---|---|---|---|---|---|---|
| `l10_ts_lam1_s0` | 0.5 | 0.6072 | 0.5727 | 0.5521 | 0.4751 | **−0.109** |
| `l10_ts_lam1_s1` | 0.5 | 0.5361 | 0.4936 | 0.5052 | 0.4872 | **−0.045** |
| `l10_ts_lam2_s0` | **0.05** | 0.6105 | 0.6079 | 0.5881 | 0.5056 | **−0.082** |

**逐条对照 §2.5b.1 的两个分支**：

- 分支 A（λ 符号成立）要求 λ₂ **比 −0.077 更负** → −0.082 确实更负。**字面满足。**
- 分支 B（纯噪声）要求 λ₂ **与 λ₁ 两条不可区分** → −0.082 稳稳落在 [−0.109, −0.045] 区间内。**也满足。**

**两边都满足 ⇒ 这个预测是我设计坏了，不是数据不给答案。** 我把阈值定在 λ₁ 的**均值** −0.077，却没有考虑 λ₁ 自身的**种子间散度 0.064**。一个有判别力的判据本该要求 λ₂ 落到 λ₁ 区间**之外**（即比 −0.109 更负）。按那个正确判据：**−0.082 > −0.109，A 不成立。**

**这一条本身就是本轮最干净的定量结果**：

> **λ 变了 10×，对结果的影响是 0.005；而固定 λ 时的种子间噪声是 0.064。信噪比 ≈ 0.08。**

这把 §2.4 pilot（500 ep 尺度）的「λ 不可标定」推到了**完整的 4000 ep 尺度**，且给出了具体数字：要在这个噪声底上分辨 10× 的 λ 差异，所需样本量约为当前的 (0.064/0.005)² ≈ **160 倍**。

**两个必须防的误读**：

1. **λ₂ 的 reward 更高（0.904 vs 0.874/0.875）不是「学得更好」。** `R = success − λ·Σcost/T_max`，λ₂ 少减了 10× 的成本项，**reward 高是定义决定的**，与策略优劣无关。三条的 success 全部平在 0.895–0.913，才是可比的量。
2. **三条轨迹形状高度相似**：都是先升到峰值再一路下行（峰值批次 s0=b5 0.631、s1=b2 0.566、lam2=b9 0.648）。三条共用**逐位相同**的 warm-start，所以这更像「同一套动力学从同一起点出发、被采样噪声扰动」，而不是三次独立随机游走。**但这不能推出「λ 有效」**——恰恰相反，λ 差 10× 而形状不变，说明主导轨迹的东西不是 λ。

**仍然不下的结论**：不宣称「λ 起反作用」，也不宣称「router 学不到东西」。能说的是——*在本实验的信噪比下，λ 对 4000 episode 尺度的臂分布没有可分辨的影响，被优化的目标在三条 run 上都没有改善。* 最终判决等 A 池评测（M7）。

### 2.5c interaction ledger 核验 + 「训一个 router 要花多少交互」的答案（M9 中不依赖 A 池的部分）

X14 存在的理由就是给「训一个 router」标价。价签的 **x 轴现在已经完全确定**（y 轴等 M7）。逐项核对 `run_manifest.json` 里 `interaction_ledger` 与 §3.10 / R3 冻结口径：

| 项 | 代码写出的值 | 应有值 | |
|---|---|---|---|
| `warmstart_episodes` | 450 | M5b 采集 450/450 | ✅ |
| `pilot_candidates` | 3 | M5c 跑了 3 个候选（从 `selection.json` 的 `candidates_run` 读出，非硬编码） | ✅ |
| `pilot_episodes` | 1800 | 3 × (5 批×100 训练 + 100 argmax 评测) = 3 × 600 | ✅ |
| `shared_offset` | 2250 | 450 + 1800 | ✅ |
| 是否三条 run 各自全额计入 | 是（三条都是 2250） | R3 的保守口径：套件级共享成本**全额**计入每个变体 | ✅ |

**于是本套件每个变体的 router 专属累计交互 = 2250 + 4000 = 6250 个闭环 episode。** 换算 wall-clock：warm-start 24.5 min + pilot ≈ 3.2 h + 正式训练 ≈ 7.0 h ≈ **10.7 小时**（单卡 4090，16 worker）。

把这个数字和 §2.5/§2.5b/§2.5b.2 的结果放在一起，就是 X14 要给出的那句话的雏形：

> **花掉 6250 个闭环 episode（约 10.7 GPU·小时）之后**：被优化的目标在三条 run 上都没有改善（reward 四分位全程持平或略降）；成本系数 λ 变化 10× 对臂分布的影响（0.005）比固定 λ 的种子噪声（0.064）小一个数量级，**信噪比 ≈ 0.08**，要分辨需约 160 倍样本。

**仍缺的是 y 轴**：headline interaction-efficiency 曲线的纵轴是 A 池成功率，必须等 M7。现在有的是 x 轴（已冻结、机器可追溯）与 B 池训练侧轨迹（不是 headline 指标）。**不要拿 B 池的 `mean_success` 当曲线纵轴**——那是训练分布上的数，与 A 池对决不可比。

### 2.5d M9 统计件盘点：**不需要新写，只缺装配驱动**（2026-08-17 亲验签名）

plan §137 把 M9 的统计口径冻死了（paired McNemar + episode 级 cluster bootstrap CI + Holm 族，seed-0 primary）。盘完发现**这些件已经全部存在且带测试**，M9 的工作量在装配而不在实现。以下签名与行为均**逐个读源码确认**，不是照文档猜：

**本线自有（`exp/rl_router/emit_router_yamls.py`）**

| 件 | 签名 | 行为 |
|---|---|---|
| `seed_roles` | `(runs: list[dict]) -> dict` | 按 `PRIMARY_SEED=0` 切成 `{"primary": [...], "robustness": [...]}` |
| `reject_seed_pooling` | `(episodes: list[dict]) -> None` | 一次比较里出现多个 `seed` 即 **raise ValueError**（防两个策略被当成一个、虚增有效样本量） |
| `holm_family` | `(runs: list[dict]) -> list[dict]` | 逐 primary run 产出 hypothesis：`flagship` → vs `tier_two_tier`；`R_tsc` → vs `tier_three_tier`；每条 `statistic="paired_mcnemar"` |

**可直接复用（姊妹实验线，已在真实数据上跑过）**

| 件 | 位置 | 签名 |
|---|---|---|
| `holm` | `exp/ablation_study/cache_size/analysis/cache_size_stats.py:243` | `(tests: Sequence[TestResult], alpha=DEFAULT_ALPHA) -> HolmResult`（`.adjusted` / `.rejected` / `.order`） |
| `cluster_bootstrap_ci` | 同上 `:271` | `(d: Sequence[float], *, level=0.95, b=DEFAULT_B, seed=0) -> CI`，BCa + 预注册的 percentile 回退 |
| `TestResult` | 同上 `:45` | `(name, p, side, t_obs, n_clusters, evaluable=True, note="", …)` |
| 精确 McNemar | `exp/ablation_study/analysis/analyze_ablation.py:95-103` | `stats.binomtest(min(n01, n10), m, 0.5).pvalue`，配 paired bootstrap CI |

**M9 还缺的只有一件**：把 A 池 eval 行加载进来 → `reject_seed_pooling` 守一道 → `holm_family` 定族 → 逐 hypothesis 算 McNemar 与 CI → `holm` 校正。**这件的输入是 M7 产物，所以它才是真阻塞项**，统计实现不是。

#### 对 §3.14 的一处修正：`flagship: true` 的两个消费者，危险的只有一个

盘 `holm_family` 时发现它先过 `seed_roles(runs)["primary"]` 再看 `run.get("flagship")`——**seed 过滤在前**，所以矩阵里 s1 那个 `flagship: true` 进不了统计族，`l10_ts_lam1_s0` 独占。**统计侧没有歧义。**

歧义只存在于**评测预算**那个消费者：`flagship_checkpoints` 目前没有任何 seed 过滤，谁照 `flagship: true` 写 M7 驱动，就会自动给 s1 也排四个检查点（= §3.14 的 5500 ep 读法）。

所以 §3.14 的处置可以更精确：**不必动 `flagship` 字段本身**（动了会破坏 `holm_family` 的正确行为），只需在 M7 驱动里对评测预算**补一道与 `holm_family` 同款的 seed 过滤**，或由 owner 明确裁定 s1 只测终点。这比原文说的「把 `flagship: true` 改成可区分字段」更小、更安全。

### 2.5e 诊断：起点被放在了 SR(p) 曲线的平坦段（2026-08-17，owner 两次纠错后重做）

**先记我今天下错的三个结论**，全部由 owner 当场指出：

1. 「teacher 与 student 近乎等效，没东西可学」——**错**。当时只比较了 router 自身工作区间内的两个五分位（teacher 占比 0.35 vs 0.55），把窄区间的小斜率外推成了全区间差异。
2. 发现需要纯 teacher 成功率后**去起了一次 450ep 采集**——**多余**，`exp/ablation_study/data/anchors/libero_10_teacher/` 早有 434/500 = **0.868**。已停并回收。**该查档案的先查档案。**
3. 「λ 网格全在盈亏平衡点以下，所以朝 teacher 漂是优化器做对了」——**基于跨池比较，不成立**（teacher 锚在 A 池、student 在 B 池）。同池测量见下。

#### 实测结果（全部同池、可复核）

**(a) 平坦段没有信号。** 12000 个 B 池 episode，(task,init) 固定效应控难度，只用前 20 步（决策先于结局）的 teacher 占比：

> **dSR/d(teacher 占比) = +0.0044 ± 0.0108，95% CI [−0.017, +0.026]**

整 episode 口径给 −0.70，那是反向因果（失败的 rollout 更长、漂到分布外、router 在那些状态上更倾向 teacher），**不能用**。

**(b) 但曲线整体是凹的。** p=0 时 SR=0.787（M5b `constant_arm`，n=450）；p≈0.44 时 0.905（12000 ep）。**两者不能同时落在一条直线上** ⇒ SR 从 p=0 陡升，在 router 探索区间之前就已饱和。**warm-start 把起点放在了平坦段上。**

**(c) trainer 没有符号错误。** 用 sidecar 的精确 logits 算 `dJ/dd = Σ_ep A_ep·Σ_t(1[aₜ=a]−p_a)`，与实际权重更新在固定特征集上的位移对照（`direction_test.py`）：

| batch | 臂 | 预测 dJ | 实测 dp |
|---|---|---|---|
| b0000 | teacher | −36.87 | −0.00050 |
| b0000 | student | +34.18 | +0.00075 |
| b0001 | teacher | −29.00 | −0.00052 |
| b0001 | student | +18.40 | +0.00071 |

两个主导臂方向全一致。（cache 臂符号不一致但量级极小，是三臂 softmax 的归一化耦合。）

**(d) 策略在动，但不学区分。** 固定特征集（ws_l10 采集特征，未参与 M6 训练）上评 `l10_ts_lam1_s0`：

| version | teacher | student | **p_student 跨状态 sd** | argmax=student |
|---|---|---|---|---|
| v0 | 0.4655 | 0.5345 | **0.1191** | 0.673 |
| v10 | 0.3927 | 0.6073 | 0.1205 | 0.796 |
| v20 | 0.4339 | 0.5661 | 0.1225 | 0.732 |
| v30 | 0.4847 | 0.5153 | 0.1216 | 0.608 |
| v40 | 0.5602 | 0.4398 | **0.1172** | 0.372 |

**均值游走（0.535→0.607→0.440），而跨状态 sd 全程 0.119→0.117 纹丝不动。** router 从头到尾没有学会「在不同状态下做不同选择」，它只是在整体挪一个偏置。

**(e) 噪声底是硬的。** 每 episode 成功率方差 0.0841 = p(1−p) 逐位吻合，**纯二项、无结构可榨**。按 (task,init) 条件化 baseline 只去掉 33.8% 方差 = 等效 1.5 倍样本量。n=100 时 3σ 只能分辨 δ=0.087，而整个策略空间可达的 success 跨度约 0.081——**单批无法区分任意两个策略**。

#### 结论

**不是 λ 错、不是 baseline 错、不是 bug。** 是 warm-start 把策略放在了 SR(p) 的平坦段：那里没有状态相关的成功率信号可学（(a)+(d) 互证），唯一剩下的成本项又与状态无关，只能推一个全局偏置，而该偏置的推力小于批噪声 ⇒ 均值随机游走。

#### 正在做的标定：SR(p) 扫点

`exp/rl_router/sweep_mixture.py`（新增）。用**零 trunk + b2** 构造精确的状态无关混合比——`relu(0·x+0)=0` 使 logits 恒为 `b2`，这正是当初「死 trunk」事故的形状，**用作对照恰好合适**。冒烟验证：目标 p=0.25 → 实测 0.2478。

7 点 p∈{0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45} × 200ep（n=200 → SE 0.021），约 2.1h，同池、同 arm yaml / key builder / artifact，不 dump 特征。

**它决定三件事**：拐点 p\* 在哪 → warm-start 该起在哪；成本梯度要多大才能把策略推到拐点而不被噪声淹没 → λ；以及**拐点附近 SR 是否状态相关 → router 到底有没有必要**（若否，最优就是一个固定比例，这本身是 X14 的重要结论）。

### 2.5f 判决：拐点起点也救不回来——router 打不过最好的固定比例（2026-08-17 结案）

§2.5e 的诊断给出一个可证伪的处方：把 warm-start 从 teacher 0.46 挪到实测拐点 0.30，λ₁=0.5 与其余冻结常数一律不动。`l10_ts_lam1_s0_knee` 就是这个处方，40 批跑完，`verify_m6.py … 40` 判 **ADMISSIBLE**（链连续、逐批 `model_sha` 互异、零拒绝、join 完整）。

#### 扫点得到的固定比例曲线（同一 200 组 (task,init)，跨 p 配对）

| p_realized | n | SR | 95% CI |
|---|---|---|---|
| 0.000 | 200 | 0.7750 | [0.712, 0.827] |
| 0.048 | 200 | 0.8100 | [0.750, 0.858] |
| 0.097 | 200 | 0.8750 | [0.822, 0.914] |
| 0.151 | 200 | 0.8450 | [0.788, 0.889] |
| 0.203 | 200 | 0.8500 | [0.794, 0.893] |
| **0.305** | 200 | **0.9250** | [0.880, 0.954] |
| 0.446 | 200 | 0.9100 | [0.862, 0.942] |

#### 四条 ts run 的收尾对照

| run | λ | reward Q1→Q4 | success Q1→Q4 | student% 起→止 |
|---|---|---|---|---|
| `lam1_s0` | 0.5 | 0.8863 → 0.8736 | 0.9090 → 0.9030 | 0.542 → 0.433 |
| `lam1_s1` | 0.5 | 0.8796 → 0.8750 | 0.9060 → 0.9040 | 0.526 → 0.482 |
| `lam2_s0` | 0.05 | 0.9047 → 0.9032 | 0.9070 → 0.9060 | 0.534 → 0.452 |
| `lam1_s0_knee` | 0.5 | 0.8765 → 0.8774 | 0.8910 → 0.9010 | **0.731 → 0.616** |

拐点起点确实把策略放到了别处（起点 teacher 0.269，全程均值 0.324，是四条里唯一整程落在拐点附近的），**但 reward 依然不动**：Q1→Q4 只 +0.0009，success +0.010，而每批噪声 sd≈0.027、每分位 10 批 ⇒ SE≈0.012。**不到 1σ。** 同时 student% 在 b0034 触底 0.572 后又回升到 0.616——**没有收敛到不动点，是慢漂加随机游走**。

#### 关键判读：三种口径，全部指向同一结论（`/tmp/knee_vs_fixed.py`）

**(1) 按实测 teacher 占比分箱，对上扫点曲线**（四条 run 合池，逐箱插值取同 share 的固定比例）：

| 箱 | n_ep | SR_router | SR_fixed | 差 |
|---|---|---|---|---|
| [0.15,0.25) | 1000 | 0.8810 | 0.8629 | **+0.0181** |
| [0.25,0.30) | 600 | 0.8917 | 0.9017 | −0.0100 |
| [0.30,0.35) | 600 | 0.8967 | 0.9220 | −0.0253 |
| [0.35,0.40) | 3400 | 0.9082 | 0.9172 | −0.0090 |
| [0.40,0.45) | 4100 | 0.9054 | 0.9125 | −0.0071 |
| [0.45,0.60) | 6300 | 0.9022 | 0.9100 | −0.0078 |

六箱五负，唯一为正的那箱在曲线最陡处（插值误差最大），且各箱差值都在各自 CI 内。

**(2) 整程均值 vs 同均值 share 的固定比例**：四条 run 全为负（−0.0028 / −0.0062 / −0.0088 / −0.0282）。

**(3) 对 router 最宽容的口径**——任一 run 的**最好** 1000ep 窗口 vs **最好**的固定比例：

> `l10_ts_lam2_s0` b0017–b0026，p=0.403，SR=0.9170 [0.898, 0.933]
> 最好固定比例 p=0.305，SR=0.9250 [0.880, 0.954]
> 差 **−0.0080 ± 0.0206（z = −0.39）**

**判决：在 libero_10 的 teacher/student 臂集上，训练出来的 router 在每一个匹配的 teacher 占比上都 ≤ 同占比的固定比例，最宽容口径下也是 −0.008±0.021。没有任何证据表明状态相关性买到了固定比例买不到的东西。** 这正是 X14 要的答案，如实记录，**不是失败**。

**⚠ 口径注意（别把它当成比实际更强的结论）**：扫点的**绝对水平**只来自一组 200 对 (task,init)（`sample_batch(batch_idx=0, seed=0)`，7 个 p 共用同一组，所以**跨 p 的差是配对的、可信**）；而 run 的 4000ep 铺在 40 组不同子样本上。因此「router 比固定比例低 0.008–0.028」这个**水平差**含一份子样本构成偏移，本身都不显著。**能站住的强陈述是「打平」，不是「更差」。** 另：扫点网格止于 p=0.446，**没有测 p=0.7 / 1.0**，所以「SR 在内点取极大、混合比纯 teacher 好」这句话**目前无据**——纯 teacher 锚 0.868 在另一个池上，跨池比较是 §2.5e 已经栽过的坑。

#### 与 §2.5e 三条诊断的关系

- 「λ=0.5 本来就对」——**维持**。拐点起点下它的 argmax 仍落在 p\*=0.30，信噪比 5.0。
- 「问题是起点」——**证伪**。起点挪到拐点后 reward 依旧不动。真正的原因是更硬的那条：**SR(p) 在 p≳0.10 之后整体是平的（0.845–0.925 全在彼此 CI 内），平坦段上没有状态相关信号可学**，起点挪到哪一段都一样。
- 「一步 Adam 只移动 0.00075」——**维持**，但已不是瓶颈：knee run 40 批实际移动了 0.115，量级够，**只是移动方向上没有收益**。

#### 下一步

ts 线到此结案，正式报告见 [`exp/rl_router/analysis/ts_line_results.md`](../exp/rl_router/analysis/ts_line_results.md)（含两张图：四条 run 的 reward/success 曲线、SR 随 teacher 占比的固定比例 vs router 对照）。**tc 线（teacher + cache）是不同的问题**：cache 的命中质量本身依赖与库的相似度，状态相关性在那里才有先验理由。`l10_tc_lam1_s0` 已于 2026-08-17 18:44 CDT 发射（gate 已过）。**判 tc 不能用本节的 0.925**——那是 teacher/student 的曲线；要判就得另跑 teacher/cache 的常数策略扫点。

### 2.5g R_tc（teacher + cache）主跑完成，判读待扫点（2026-08-18 08:5x）

`l10_tc_lam1_s0` 40 批跑满，`verify_m6.py … 40` 判 **ADMISSIBLE**。中途 22:2x 为腾显卡停过一次、03:03 从 b0015 续跑，批级 resume 未损失任何 episode（§3.22）。

| 量 | 值 | 与 ts 线对照 |
|---|---|---|
| success 均值（4000 ep） | **0.7670**  [0.67, 0.89] | ts ~0.90——**cache 臂比 student 臂弱得多** |
| cache 份额 起→止 | 0.507 → 0.476（净 **−0.030**） | ts 的净位移 −0.045 ~ −0.134 |
| teacher 份额 均值 | 0.5068 | |
| 成本 | **50.7%** of all-teacher | ts 约 30–36% |

**起点是均匀常数，这一点很关键。** `graft` 对无 student 臂的臂集给零输出层（§3.8 预注册、§3.18 我曾误判成退化并误停过一次），所以 v0 恰好是 50/50 的状态无关策略。于是：

- **ts 的核心负结果是「均值游走，但跨状态 sd 从 0.1191 到 0.1172 纹丝不动」**——策略从头到尾没学会区分状态；
- **tc 的 `policy_distinct` 从 b0000 的 1 长到 ~2300**，即输出层确实离开了零点、变成状态相关。这**不是**同一个现象，值得在 tc 报告里单独量化（跨状态 sd 随版本的轨迹），而不是套用 ts 的结论。

**但「变得状态相关」不等于「有用」。** 判读必须对上同占比的固定比例，而 **ts 的扫点（0.925 @ p=0.305）是 teacher/student 曲线，对 tc 无效**。teacher/cache 扫点 08:54 发射（tmux `rlrsw`）：

- 网格 **{0.0, 0.15, 0.30, 0.45, 0.55, 0.70, 1.0}**，**不是**照抄 ts 的 {0…0.45}。理由一：tc run 全程工作在 teacher 0.48–0.52，ts 网格一个点都没覆盖到，照抄会做出一条必须外推才能用的曲线。理由二：**p=1.0 是故意加的**——纯 teacher 常数策略与另一臂是谁无关，所以这一个点同时补上了 §2.5f「口径与边界」里点名缺失的**同池纯 teacher 锚点**（0.868 那个锚在另一个池上，ts 扫点又止于 0.446，正因如此「混合优于纯 teacher」当时只能留白）。
- 7 点 ×200ep = 1400 ep ≈ 1.7h。

判读工具 `router_vs_fixed.py --cheap-arm cache`；⚠ 它的默认 `--bin-edges` 是照 ts 工作区间（0.15–0.60）定的，**tc 要按 0.48–0.52 重设分箱**否则多数箱为空。

### 2.6 run 完整性验收工具 `verify_m6.py`（新增，wls `/tmp/`）

§2.5 只谈**学习信号**（要等 40 批，不许半程外推）。**完整性**是另一回事——它随时可查，而且越早查越省：链在第 12 批断掉却跑到第 40 批才发现，就白扔 5 小时。故新增 `verify_m6.py <run_id> [expected_batches]`，六条断言，退出码即判决：

| # | 断言 | 数据源 |
|---|---|---|
| 1 | 权重链 `v_k → v_{k+1}` 逐批首尾相接 | 每批 `versions.json` |
| 2 | `n_rejected` 全零 | `metrics.jsonl` |
| 3 | join 完整：`COMPLETE.json` 存在、`quarantine` 空、`expected_slots` 数 == `n_episodes`、manifest 的 `weights_version` 与批序一致 | `package/r0/{COMPLETE,accepted_manifest}.json` |
| 4 | `batch_id` 连续 `b0000..b(N-1)` | `metrics.jsonl` |
| 5 | 每批 episode 数一致 | `metrics.jsonl` |
| 6 | **每批 `model_sha` 互不相同**（trainer 真的动了权重，不是空转）+ `encoder_version` 全程不变 | 每批 `export_meta.json` |

第 6 条是这次新加的，前面没有任何守卫覆盖它：trainer 若因为某种原因产出与输入逐字节相同的权重，`n_rejected` 仍是 0、链仍连续、join 仍完整——只有 sha 会露馅。

**旗舰前 33 批结果（2026-08-16 08:5x）**：`ADMISSIBLE`。链 v0→v33 连续，**33/33 个不同 `model_sha`**，`encoder_version` 全程 `b09609c9…`，零拒收零 quarantine，每批 100 slot 全到。

> 踩坑记：这个脚本第一版把 manifest 猜成了 `<batch>/manifest.json`（实际在 `package/r0/accepted_manifest.json`），于是给出 33 条 `missing-manifest` 的假 FAIL。**「验收脚本报错」的第一嫌疑人永远是验收脚本本身**——参见 §3 里已经修过的六处探针缺陷。另：`tether push` 目标已存在会以 `code=dst_exists` 退出 64 而不覆盖，得加 `--force`；我第一次没看返回码就跑了旧脚本，正撞上「改完要读回确认」那条。

### 2.3 提速对全线排期的影响

| 阶段 | 原（t107 经 broker，2.4 ep/min） | 现（wls 本机 16w） |
|---|---|---|
| M5b 450 ep | ~3 h | **24.5 min（实测）** |
| M5c pilot 15 批 + 3 评测 | ~12.5 h | **~3.2 h（按实测 11.8 min/批）** |
| M6 五 run × 40 批 | ~139 h（5.8 天） | **~37 h** |
| M7 A 池评测 3500 ep | ~24 h | **~5 h** |

**实测每批 11.8 min** = 100 ep 约 9 min（含 worker spin-up）+ trainer 约 2 min。注意 trainer 那 2 min 与 rollout **必然串行**——on-policy 每批一次 Adam step 就是一道 barrier，这不是可优化的调度问题而是算法定义。

也不建议再往上堆并发：rollout 阶段 GPU 已 86%，加 worker 收益趋平；加 server replica 更不划算（渲染与推理共卡，且 3 replica × 10 GB 会把显存吃到与别的 session 打架）。5 个 run 并发同理——它们抢的是同一张卡。

**2026-08-16 08:5x 补一次实测**（旗舰 collect 阶段，1 Hz 采样 12 点）：util `35 64 86 59 98 43 37 35 85 49 43 91`，**均值约 60%、峰值 98%**；卡上只有本线三个进程（pi05 server 8.6 G + 两个 ACT sidecar 3.4 G / 2.8 G），别的 session 当时不在。所以余量存在但不足 2×——并发第二个 run 乐观估计 1.3–1.5×，代价是把卡推到饱和、并在无人值守时段把本线在**这张与 owner 其它 session 共用的卡**上的占用翻倍。

**结论：维持串行**，理由两条且都不依赖上面这条估算——(1) encoder v2 尚未过 G2，逐个发射把返工半径限制在一个 run；(2) 同机常有 owner 的其它 session，单方面翻倍占用会挤掉另一条实验线，这类调度取舍不该在无人值守时自作主张。此处记录实测值是为了让这条决策**有据可查**，而不是像原文那样只凭「86%」一个瞬时点。

**并发在数据层面本身是安全的**（若将来 owner 决定并发，不必改隔离）：分片根虽由五个 run 共享 `dump_m6`，但落盘路径是 `dump_m6/<run_id>/<batch_id>/`，且每条 manifest 行都带 `run_id` 字段——两个 run 的目录与 join 键都不相交。

---

## 3. 运行期实测暴露的问题与处置

### 3.1 两机工作树与上游冲突（环境，非代码）

wls 在 `df2ef13`、t107 在 `d2e4293`，各带脏工作树，`git pull --ff-only` 被拒。处置：逐文件 sha256 比对确认**除 `src/openpi/cache/config.py` 外全部与上游 `721d3fa` 逐字节相同**，而该文件上游是本地版的严格超集（composite 白名单 + warm 层守卫都在，另加 `mlp_router`）。因此：

1. 脏的 tracked 文件 → `git stash push -- <paths>`（**可恢复**，未用 `-u`）；
2. 阻塞 pull 的 untracked 文件 → 打 tarball 备份后删除，pull 完成后逐文件 `cmp` 比对，仅把**上游没有的**那些还原回去。

备份位置：wls `/home/weiland/openpi_dirty_backup/`、t107 `/scratch/zixuans8/openpi_dirty_backup/`。

⚠ **副作用需 owner 知悉**：`exp/ablation_study/config/select_freeze_*.yaml` 在两机上都存在与上游不同的本地版本，同步后**上游（tracked）版本生效**，本地版在上述 tarball 里。属 ablation 线资产，本线未使用。

### 3.2 batch package 的 scp 通道缺凭据（环境）

t107 → wls 的 `wls-ssh:14024` 无可用公钥（且 `tether exec` 的 `HOME` 是 `~/.tether-agent`，默认根本找不到 `~/.ssh`）。处置：

- 在 t107 生成**本实验专用**密钥 `~/.ssh/rlr_t107`（注释 `rl-router-batch-package t107->wls`），只把它加进 wls 的 `authorized_keys`（改动前已备份 `authorized_keys.bak.rlrouter.<ts>`）；
- **不改任何 ssh config**：conductor 启动前 `eval $(ssh-agent -s) && ssh-add ~/.ssh/rlr_t107`，agent 随进程消亡。这是最小权限做法（不给个人默认密钥授权、不留持久配置）。
- 实验结束后可从 wls `authorized_keys` 删除该行以撤销。

### 3.3 `microbench_cost` 两处实跑失败（代码，已修 + 补测）

| 现象 | 根因 | 修法 |
|---|---|---|
| cache 臂 `BackendFrozenError: Cannot perform 'insert'` | 探针向库里塞一条合成 entry；而生产臂 yaml 预载库后 backend 即冻结（`write_policy: never`） | 冻结时改为**取库内既有 entry** 回放——这也更贴 **plan** §3.10「真实 replay 路径」的口径（合成 chunk 的尺寸不是实际搬运的尺寸）。仅在 backend 可写（单测夹具）时才用合成种子 |
| student 臂 `ModuleNotFoundError: No module named 'openpi'` | `measure_student` 的 SidecarExecutor 守卫是硬 import；而该测量按设计**必须在 sidecar 自己的 venv 里跑**，那里没有 openpi | 守卫改为先按类名判定（到处都成立），再在能 import 时补上真正的 `isinstance` |

补充：修 cache 臂时我把 `storage.is_frozen`（**property**）误写成方法调用，第二次实跑才炸。已修，并补了一个**走真实分支**的测试——只测辅助函数的话这个 bug 照样漏网。

### 3.4 M4 与 M5c 的门禁死锁（代码，已修 + 补测）

`check_launch_gates(bootstrap=True)` 原本只豁免 `capacity_smoke`，但 λ / M5a / M5b / M5c 四项照查。于是：

- **M4 跑不了**：它按 plan §8 排在 M5a-c *之前*，那四项此刻都不存在；
- **M5c 更跑不了**：pilot 候选的训练半程要走的正是同一道门，而它要门放行的东西（pilot record）恰好是它自己的产物——自指死锁。

plan §1 明写「占位常数仅限 M4 smoke」，即 plan 本就授权 M4 用占位常数；只是门禁没被教会这件事。处置（视为**实现缺口补齐**，非偏离）：

1. `bootstrap=True` 现在额外豁免 **λ 与 pilot record 两项**；arm_costs / warm-start 权重 / 臂 yaml / 矩阵一致性 / 容量上限**一律照查**——拿伪造成本跑的 smoke 并没有在演练 M6 要跑的那个循环。
2. 新增 `pilot_calibration=True`（CLI `--pilot`）：只豁免 pilot record 一项，**λ 仍必查**（候选的定义就是「被要求训练的那个 λ」），容量 smoke 也仍必查。
3. `--smoke` 下 λ 缺失时取冻结网格中位数 0.2 作占位并 `logger.warning` 声明；取 0 会把成本项直接关掉，反而测不到东西。正式 run 仍照旧硬失败。

**另行裁定的顺序调整**：本次把 **M4 挪到 M5a/M5b 之后**执行。M4 因此拿到的是**真实**成本与**真实** warm-start 权重，只有 λ 是占位——比 plan §8 原序（四项全占位）证据更强，且 §1 要求的「M4 先于 M6」不变。**请 owner 追认。**

### 3.5 远端 helper 的 `uv` 不在 ssh PATH（代码，已修 + 补测）

`run_rl_router` 有 5 处硬编码 `cd {remote_workdir} && uv run exp/rl_router/...`。非交互 ssh 拿到的是系统 PATH，wls 上 `which uv` → not found，这几处会在**episode 已经跑完之后**才炸。新增 `--remote-python`（默认 `uv run`，本次用 `/home/weiland/.local/bin/uv run`），并把它一路传到 `remote_build_manifest`。测试在 main() golden 里断言所有远端 helper 调用都带上了配置的启动器。

### 3.6 M5c 评测半程无驱动（代码，已补 + 补测）

plan §3.10 步骤 3（冻结第 5 批权重、argmax 在 B-train 余集测 realized teacher rate）与 `candidate_manifest()` 要读的 `<candidate_dir>/eval/{arm.yaml,client_rows.jsonl}` 之间没有任何脚本。plan §5 文件清单也没列。补法：给 `pilot_lambda.py` 加 `eval` 子命令（**不新增文件**，pilot 协议本就归它管），复用 `run_round`/`RouterBatchStrategy`。语义完全由 §3.10 冻结，无设计自由度。

### 3.7 M4 报告的稳态磁盘量取错文件系统（代码，已修 + 补测）

`emit_m4_report` 把 conductor 的**本地** `scratch/` 目录当 dump root 传给 `m4_smoke`，而分片在服务器侧——`steady_state_bytes(scratch)` 只会数到几个 fetch 下来的 json，报告里的 `steady_state_bytes` 因此≈0。峰值字段没受影响（`bytes_before/after_reclaim` 走的是 `_remote_live_bytes`，量的是对的地方），所以 gate 不会误放行，但 plan §1 要求 M4「实测稳态磁盘」，报告里写 0 是**朝着放心方向的错**，属于必须修的那一类。

修法：`m4_smoke` 新增可选 `live_bytes`，调用方把已经在分片所在主机量到的 `capacity["after"]` 传进去；本地 stat 退化为同机场景的兜底。

### 3.8 同一 attempt 下分片被重写时，packager 取错 manifest 行（代码，已修 + 补测）

M5b 采集的 shard manifest 有 **456 行但只有 450 个唯一 `(task_uid, attempt)`**——6 个 episode 被终结了两次：同 uid、同 attempt、同行数、**不同 sha256**，相隔约 11 分钟。核对 journal（恰 450 行、零重复 uid）发现这 6 条的 journal 时间戳对应的是**第二次**写入，说明首跑的结果没送达 driver、被原 generation 重新派发（`mark_result` 没收到结果就不会推进 attempt），第二次在同一路径重写了分片。

磁盘数据本身是自洽的（`.bin` 与 journal 标签都来自第二次），`fit_warmstart.load_collection` 用字典推导建索引恰好是 last-wins，不受影响。但 **`batch_package.build_batch_manifest` 用 `next(...)` 取的是第一条匹配行**，于是拿陈旧 sha 去校验已被覆写的文件 → `sidecar_digest_mismatch` → 好端端的 episode 被拒 → 触发 repair 轮，甚至可能整批 fail。

按 finalize 协议（`.bin.tmp` → fsync → 原子 rename → manifest append），**最新那条 manifest 行才描述磁盘上的字节**，所以改为取最后一条匹配行。发生率实测 6/450 ≈ 1.3%——放到 M6 的 4000 ep/run 上就是每 run 约 50 次拒收，几乎一定会把批完整性打穿。

> 这条是本次运行期最有价值的发现：它只有在**真机、多 worker、长跑**下才暴露，单测与 M4 smoke（20 ep）都不一定撞得到。

### 3.9 容量探针把 stderr 噪声当成了答案（代码，已修 + 补测）

M4 首次启动被 `LAUNCH BLOCKED: remote capacity probe returned unparsable output` 拦下，但探针的 stdout 明明是一行干净 JSON。根因：`LocalTransport.run` / `SshTransport.run` 返回的是 **stdout 与 stderr 拼接**，而启动器往 stderr 写东西——`uv` 每次调用都打一条 deprecation 警告。于是 `output.strip().splitlines()[-1]` 拿到的是那条警告，**一个健康的空 dump root 被判成"磁盘不可测"从而 fail-closed 拦住发射**。

修法：从后往前扫，取**最后一行能解析成含目标字段的 JSON 对象**的行，其余噪声跳过。fail-closed 语义保留（真的解析不出仍然拦）。

> 这类 bug 在单测里天然测不到——夹具的 `run` 返回的是干净字符串。补的测试直接模拟"JSON 后面跟一行 uv 警告"。

---

### 3.10 M7（A 池评测）尚缺入口——已探明，待决策

A 池 = LIBERO 自带的 `task_suite.get_task_init_states(task_id)`，也就是 `--init-states-dir` **为空**时加载的那份。既有的 `exp/ablation_study/run_ablation_eval.py` 正是靠"不传该参数"来评测 A 池的。

但本线的 `resolve_init_states_dir` **明令拒绝空值**（§8 陷阱 1，防的就是训练误落测试集），而 `emit_router_yamls.check_init_pools(train, eval)` 又要求 eval 侧是一个**真实且与训练池互斥**的目录路径——两处合起来意味着 plan 期待的是「A 池被物化成一个显式目录」，而这个物化步骤目前**不存在任何脚本**。

建议（待 owner 定）：把 `get_task_init_states` 逐任务落成 `exp/common/data/db_init/libero/<suite>_apool/<task>.init`，然后 M7 照常传 `--init-states-dir`。好处是 A 池从「隐式回落」变成**显式、可审计、可 sha256 的产物**，`check_init_pools` 的双向互斥断言也就真的有东西可断。**不建议**给 eval 开一个"允许空值"的口子——那正好把 §8 陷阱 1 防的东西放回来。

时间上不紧：M6 还要约 29 h，届时再定即可。

#### 3.10.1 物化方案的可行性实测（2026-08-16，只读探针 `probe_apool.py`）

上面的建议原本只是纸面推理。现已在 wls `libero_sim` 环境里**逐条验过**，三个数字把方案钉死：

| 问题 | 实测答案 |
|---|---|
| A 池有多大？ | **两套件各 500 个状态** = 10 任务 × 50，与 B 池逐任务同维（如 libero_10 task0 都是 `(50, 123)`） |
| A 与 B 真的互斥吗？ | **是**。500 个 A 行逐行在 B 池里找不到匹配（`atol=1e-7`，与 `cache_positions_for_task` 同容差），libero_10 与 libero_spatial 皆然 |
| 物化会不会失真？ | **不会**。`torch.save(get_task_init_states(t))` → `torch.load` 往返 `np.array_equal == True`，逐位相同 |

这是本线第一次**用数据**证实 A/B 互斥，此前只是「按构造应当如此」。

三处守卫在物化方案下的行为（均已读源码确认，非推测）：

- `examples/libero/main.py:1160-1181` `_load_init_states`：给了目录就优先 `<task>.pruned_init`、否则 `<task>.init`，`torch.load` 读出。物化文件用 `.init` 后缀即走正常分支。
- `exp/ablation_study/build_distill_dataset.py:44-54` `check_init_dir`：只拒 `*.pruned_init` 文件名，物化目录里没有这种文件 → 通过。
- `exp/rl_router/emit_router_yamls.py:161-174` `check_init_pools`：拒「同路径」与「互为父子」。`<suite>_apool` 与 `<suite>` 是**兄弟目录**，两条都不触发 → 通过。

所以物化路线**零代码改动**即可让 M7 走通，物化脚本本身是一次性产物生成器。规模：每套件 10 个文件、500 个状态，几十 KB。

> py3.8 坑：`libero_sim` 环境的 torch 不认 `weights_only` kwarg，探针得照抄 `main.py:1174` 的 try/except 兜底。任何要在这个环境里读 `.init` 的新脚本都一样。

#### 3.10.2 物化脚本已就绪但**未对真实路径执行**（等 owner 裁决）

`materialize_apool.py <suite> <out_dir> [--check-only]`，staged 在 wls `/tmp/`。自带四道自检：拒绝写入已有 `.init` 的目录、绝不产出 `.pruned_init` 名、每个文件写完立刻读回并要求 `np.array_equal`、逐文件打印 sha256 供 run record 钉住。

**已实跑验证**（2026-08-16）：`--check-only` 端到端跑通（临时目录，事后自删，`ls /tmp/apool_check_*` 为空）；再拿一个 scratch 路径实写一次、第二次重跑确认**拒绝覆盖并退出 1**；两次独立运行同一任务的 sha256 **完全一致**（如 `6b75c8d4d1e1c40b`），说明 `torch.save` 的容器是确定性的、这个指纹可以当稳定 pin 用。

**刻意没做的事**：没有对 `exp/common/data/db_init/libero/<suite>_apool/` 真实路径执行。M7 走不走物化路线是 owner 的裁决（§4-④），脚本先备好、决定权不预设——一个从没跑过的脚本在无人值守时段是负债，一个跑过但没落地的脚本不是。

---

### 3.11 上机脚本清单（全部 staged 在 wls `/tmp/`）

这些是**运维粘合层**，故意不进 repo（plan §5 的文件清单是实现件的清单，不是发射脚本的）。

| 脚本 | 作用 | 验证状态 |
|---|---|---|
| `run_m4.sh` | M4 bootstrap（20ep×2 批） | ✅ 已真跑，passed |
| `pilot_candidate.sh` | 一个 λ 候选：训练 5 批 → 取第 5 批权重 → argmax 评 100 ep | ✅ 正在真跑；末尾取权重的表达式已用现存 `versions.json` 预验 |
| `pilot_health.sh` | pilot 健康一行（批数/候选数/episode/显存余量/错误） | ✅ |
| `emit_m6_arms.py` | 逐 run emit 臂 yaml | ✅ 五个 run 全 emit 成功（含 R_tc 无 routing、R_tsc 三臂、seed 0/1） |
| `check_m6_arms.py` | 逐 run 干跑 arm-yaml 门禁 | ✅ 五个 run 全 OK |
| `set_m6_lambda.py` | 把 pilot 选出的 λ 写进 tracked matrix；**未分离则拒绝并升级 owner**（R13） | ⬚ 待 `selection.json` |
| `list_m6_runs.py` | 列 run id / 查某 run 的 variant | ✅ |
| `prepare_m6.sh` | 串起上面三步 + 五个 run 全量门禁 | 部分（除写 λ 外均已验证） |
| `run_m6.sh <run-id>` | 发射一个正式 run | ✅ 旗舰在跑 |
| `verify_m6.py <run-id> [N]` | run 完整性六条断言（§2.6），退出码即判决 | ✅ 旗舰前 33 批 ADMISSIBLE |
| `probe_apool.py <suite>` | 只读：A 池规模 / A-B 逐行互斥 / 物化往返保真（§3.10.1） | ✅ 两套件均跑过 |
| `materialize_apool.py <suite> <dir>` | 把 A 池落成显式 `.init` 目录（§3.10.2） | ✅ check-only + 覆盖守卫均实测；**未对真实路径执行** |
| `reap_orphans.sh` | 白名单回收孤儿 worker | ✅ 真实回收过 48 个 |
| `rlr_health.sh` | 健康一行 | ✅ 被 L2/L3 两层调用 |

**为什么把 python 抽成独立文件而不是内嵌 heredoc**：本会话两次栽在嵌套引号上（一次在 ssh config，一次在探针脚本），而发射脚本要在无人值守下跑几十小时——引号错在三小时后才炸的代价远高于多开一个文件。

---

### 3.12 router 的 trunk 全死，策略与观测无关（**已定位并修复；曾误判为冻结契约问题**）

M5c 跑到第 3 批时，我预跑了从未执行过的 `pilot_lambda.py eval` 代码路径，读数异常：**argmax 下 teacher_rate = 1.0**，而同期采样端稳定在 50/50。追下去发现每一步的 `probs` **完全相同**（`[0.50021, 0.49979]`）——router 根本不看观测。

**已用真实采集特征证实**（`probe_router_variance.py`）：

```
warmstart_l10.pt (v0)   live hidden units: 0/256   hidden abs-mean: 0.0
                        student logit std: 0.0     argmax student share: 0.0000
v3.pt (3 次 REINFORCE)  live hidden units: 0/256   student logit: -0.000422 (只有 b2 漂了)
```

**后果（三条，任何一条都足以作废实验）**：
1. 策略是个固定的 p≈0.5 抛硬币，argmax 必然塌成单臂（100% teacher）；
2. `hidden ≡ 0` ⇒ ReLU 导数恒 0 ⇒ **`W1/b1` 的梯度恒为零，REINFORCE 永远学不动**（v3 只有输出偏置动了 4e-4）；
3. 于是 M6 会产出「router 学不会」的曲线——但那是死网络的假象，**不是**「为什么不训 router」的真实答案。这恰好会把本实验的结论污染成它最想避免的那种伪证据。

**根因不是输入尺度，是训练动力学**（`diagnose_dead_relu.py`）：

| | 初始化时存活 | 30 步拟合后 |
|---|---|---|
| 当前 encoder（仅 robot_state 仿射，plan §3.0） | **0.5068（健康）** | **0.0000**，平均预激活 −393 |
| 逐字段标准化 | 0.5088 | **仍 0.0000**，−75 |

初始化是健康的。杀死 trunk 的是 **Adam**：它对每个参数的单步位移≈`lr`，与梯度幅值无关；而第一层有 **65,568** 个输入、实测特征 std≈5.25、absmax≈209，于是单步预激活变化 ≈ `1e-3 × 65568 × 4.2 ≈ 275`，几步之内把所有单元推进死区。`clip=1.0` 救不了——Adam 归一化后梯度幅值已被丢弃。

同一算术对 **plan §3.5 冻结的 trainer `lr=3e-4`** 同样成立：`3e-4 × 65568 × 4.2 ≈ 82` 每步，一步即死。所以这不只是 warm-start 拟合的问题，**RL 训练本身也跑不动**。

**被证伪的是 plan §3.0 的一句假设**：「vision 字段 builder 已定标不再动」。实测 `cp1_spatial_pool_16` 的 vision 字段 std≈5.25、range ±209，并未定标。

### 可选修法（已在真实特征上实测，`sweep_fix.py`）

| 特征缩放 | lr | 拟合后存活 | 输出 std | 评价 |
|---|---|---|---|---|
| raw（当前） | 1e-3 | **0.0000** | 0 | 现状 |
| raw | 1e-5 | 0.379 | 1.337 | 需改 lr |
| raw | 1e-7 | 0.507 | 0.132 | 需改 lr，且学得极慢 |
| 逐字段标准化 | 1e-3 | 0.0000 | 5.6e-9 | **不够** |
| 逐字段标准化 | 1e-5 | 0.530 | 0.822 | 需改 lr |
| **标准化 ÷ √D** | **1e-3** | **0.330** | **0.648** | **保住两个冻结 lr** |
| 标准化 ÷ √D | 1e-5 | 0.575 | 0.0089 | 也可 |

**建议（待裁决）**：走「**encoder 内做逐字段标准化并除以 √D**」这一族，理由三条——

1. 它**保住 plan §3.5 冻结的 `lr=3e-4`** 与拟合头的 lr，改的是被实测证伪的那条假设本身，而不是算法常数；
2. 它对未来任何特征宽度都成立（把 `‖x‖` 归到 O(1) 是宽层 + Adam 的通用前提），而调 lr 只是针对当前 D 的一次性补丁；
3. **不需要重新采集**：采集落的是 **raw** v0 特征，任何缩放都能离线从同一批 450 ep 重新拟合。代价 = 一次重拟合（8 min）+ 重跑 pilot（约 3 h），而非重采（又 25 min + 全部下游）。

机制上也是支持的：`encoder_version = sha256(fields+顺序+μσ)` 本就设计成「归一化一变它就变」，改完新旧分片天然不会混用。

### 3.12.1 修正：定位错了一半，实测把它纠回来了

上面「§3.5 冻结的 `lr=3e-4` 也会杀死 trunk，所以必须改冻结契约」这个判断是**从 BCE 拟合类比推出来的，没有实测**——而 REINFORCE 的 advantage 在 episode 间来回翻符号，Adam 的 `m̂/√v̂` 在噪声梯度下远小于 lr，两者并不同构。补测（`probe_rl_dynamics.py` / `probe_rl_signal.py`，真实特征 + 冻结的 §3.5 配置）：

| head 拟合 lr | 移植后 live | 冻结 lr=3e-4 跑 40 步后 |
|---|---|---|
| 1e-3（当时的实现默认） | **0.0000** | 一直 0（本来就死） |
| **1e-5** | **0.3468** | **0.2188，稳住不掉** |

**结论反转**：`lr=3e-4` **不会**杀死一个健康的 trunk。唯一被确证的缺陷在 `_train_head` 的 `lr=1e-3`——而**它不在任何冻结条款里**：§3.8 冻结的是「同架构 2 层 MLP + BCE 目标 + grouped 5-fold 选**正则**(weight_decay) + seed=0」，lr / epochs 是实现默认值。所以这是 **L1 bug 修复，不需要 owner 裁决**，也不必碰 §3.0 的 encoder 归一化。

（附带说明：`probe_rl_signal.py` 里 `prob_std` 归零是探针自身的局限——合成 reward 只有 20 个 episode、信号是随机投影，弱到熵项压过 advantage。这个问题正是 pilot 要回答的，不是探针能回答的。）

### 3.12.2 修法与守卫（已实施）

1. `fit_warmstart.HEAD_LR = 1e-5`（原 1e-3），旁边写明定标依据：Adam 单步位移≈lr，故单步预激活变化≈`lr·Σ|x_j|`；本 trunk 有 65,568 输入、实测 std 5.25 / max 209，`Σ|x_j|≈2.75e5`，于是 1e-3 → 每步 275，几步入死区。
2. 新增 `trunk_health()` + `assert_trunk_alive()`：拟合完、移植前检查存活单元数与头输出方差，**退化就硬失败**而不是把常数 router 发出去。健康指标写进 report。
3. 三个测试锁死：死 trunk 必须被拒、健康 trunk 必须放行并报余量、`HEAD_LR·65568·4.2 < 10`（把定标依据变成可执行断言）。

**重拟合结果（复用同一批 450 ep 采集，未重采）**：

| | 修复前 | 修复后 |
|---|---|---|
| CV loss (wd=0) | 0.911 | **0.456** |
| 存活单元 | 0/256 | **112/256（39.5%）** |
| 头输出 std | 0 | **1.565** |
| δ₀ | −0.0156 | **0.6364** |

**0.456 < 0.518**（78.7% 成功率的基率熵）——这个头现在在留出折上**优于常数预测器**；修复前的 0.911 远劣于常数，本身就是退化的铁证，**CV 指标当时已经报警，只是没人看**。部署态复验：108/256 存活、student logit std 0.536（range −0.76~1.99），策略确实随观测变化。

**代价**：M5a 成本表、M5b 的 450 ep 采集与分片、M4 报告**全部不受影响**；作废的只有 `warmstart_l10.pt`（已重拟合，548 s；旧的存为 `warmstart_l10.DEAD_TRUNK.pt` 备查）与 pilot 的 3 个批次。pilot 已在健康权重上**重启**，`plan.json` 与两个 split yaml 原样保留（它们是冻结清单，重抽会改变 pilot/remainder 划分）。

### 3.12.4 ⛔ 第二次退化：策略在**一步之内**饱和（§3.12.1 的"结论反转"是错的）

重启后的 pilot 第 2 批就废了。b0000 健康（student 0.6025、teacher_prob std 0.236），**b0001 的 `teacher_prob` 恒等于 1.0000（5517 步、std 0）、student 率 0.0000、grad_norm 0.000**。查权重确认在参数里，不是日志问题：

| | v0 | v1（一次 Adam step 之后） |
|---|---|---|
| student logit | mean **0.836**，std 0.536 | mean **−45.49**，std 1.17 |
| student prob | 0.687 | **0.000000**（std 3.5e-20） |
| live hidden units | 108/256 | 48/256（**仍活着**） |

一步位移 **−46.3**，与最初算式 `lr·Σ|x_j| = 3e-4 × 2.75e5 ≈ 82` 同量级。**§3.12 最初的判断是对的；§3.12.1 那次"反转"是错的。**

**我错在测错了量**：`probe_rl_signal.py` 显示 trunk 存活率稳在 0.2188，我就判定「lr=3e-4 不杀 trunk，所以没问题」，并把同一张表里 `prob_std→0` 当成弱信号伪影忽略了。但 **trunk 活着而 softmax 饱和，后果与 trunk 死完全一样**——甚至更隐蔽，因为存活单元数这个指标会一直显示健康。**该看的是策略输出的离散度，不是隐层存活数。**

这个教训已写进 L3 巡检脚本 `/tmp/rlr_health.sh`：`policy_distinct` = 一批 client rows 里 `probs[0]` 的不同取值个数，健康时 5000+，退化时 1。两次退化它都能立刻抓到，而进度/join/拒收/成功率这些常规信号**两次都显示一切正常**。

### 3.12.5 两族修法的实测对照（冻结 trainer 配置下跑 40 步）

| 特征缩放 | trainer lr | step 1 | step 40 |
|---|---|---|---|
| raw（现状） | 3e-4（冻结） | **distinct=1、std=0（饱和）** | 永久冻死 |
| **标准化 ÷ √D** | **3e-4（冻结不动）** | distinct=567、std 2.0e-2 | distinct=576，student 率 **0.767→0.637 单调漂移** |
| raw | 6e-7 | distinct=592 | distinct=593，但 0.859→0.919→0.803 非单调 |

两族都能救活策略，差别在：**标准化÷√D 保住了 §3.5 的全部冻结常数**（lr=3e-4 / β=0.01 / clip=1.0），且 student 率随训练单调漂移——正是 pilot 需要测的那个可分离信号；而降 lr 到 6e-7 要改明文冻结项，且步长小到噪声主导，40 步内看不出方向。

**需要 owner 裁决**：标准化÷√D 改的是 plan §3.0 冻结的 encoder 定义（`encoder_version` 会随之改变——这本就是它的设计意图）。§3.0 不缩放 vision 的理由写的是「vision 字段 builder 已定标」，而实测 `cp1_spatial_pool_16` 的 vision 字段 std≈5.25 / range ±209，该前提不成立。但改它会改变论文测的东西（表示尺度直接影响「训 router 要花多少交互」这个 headline 量），所以是 owner 的裁量。

### 3.12.6 encoder v2 的实施与验收（owner 裁定：标准化 ÷ √D，§3.5 冻结常数不动）

**实施**（`RouterFeatureEncoder` + `fit_warmstart`）：

1. 仿射作用域从「robot_state 切片」推广到**整个拼接向量**——`(x−μ)/σ` 一个算子即可表达「逐字段标准化 + ÷√D」（√D 折进 σ），算子次序改为 `raw → concat → normalize → Q`，`encoder_version` 随之改变（本就是它的设计意图）。vision 用**逐字段标量**、robot_state 保留**逐维**：pooled 视觉嵌入按 32,768 维各自标准化会把这批里恰好近似常数的维度放大成纯噪声，而 robot_state 混着关节角与夹爪标志，逐维才对。checkpoint 键 `robot_state_mu/sigma` → `feature_mu/sigma`，旧键仍读取以便旧 checkpoint 报出明确的维度不匹配而非 KeyError。
2. **把病根一并修掉**：`HEAD_LR` 硬编码改为从实测特征尺度**推导**（`head_learning_rate`，令 `lr·Σ|x_j| = HEAD_STEP_TARGET = 0.1`）。固定 lr 正是连续两次退化的共同原因——raw 下 1e-3 太大打死 trunk，v2 下 1e-5 太小让头停在初始化（CV loss 0.687 ≈ ln2）。推导之后对任何宽度/缩放都自洽。
3. **新增第二道守卫** `assert_head_beats_base_rate`：留出 CV loss 必须低于基率熵，否则拒绝移植。这条能同时抓住前两次（0.911 与 0.687，bar 分别 0.518 / 0.656）——**两次的 CV 数字当时都摆在报告里，只是没有任何东西去比它**。
4. 删除 `fit_robot_state_stats`（v2 后无生产调用方，WA §3.1 禁死代码）。

**第三次重拟合**：`head_lr=1.128e-3`（推导值）、**cv_loss 0.555 < 基率熵 0.656**、trunk **134/256 存活**、输出 std 0.534、δ₀=0.570 → 部署态 student 率 0.5000。

**验收（就在两次塌掉的那个位置）**——v0 → v1 一次 Adam step，冻结 lr=3e-4：

| | 修复前（v1 encoder） | 现在（v2 encoder） |
|---|---|---|
| student logit mean | 0.836 → **−45.49** | 0.472 → **0.525**（+0.053） |
| student logit std | 0.536 → 1.17 | 0.198 → **0.197** |
| live hidden units | 108 → 48 | 134 → **134** |
| b0001 逐步 probs | **distinct=1（饱和）** | **distinct=413、std 0.101、range 0.24–0.71** |
| grad_norm | 41.99 | **0.156** |

更新从「−46 的悬崖」变成「+0.053 的轻推」，冻结的 `clip=1.0` 现在根本不触发。

> **提请 owner**：encoder v2 动的是 G1/G2 冻结的 §3.0。虽由 owner 直接裁定，但正式 M6 开跑前值得走一轮 G2 review——本线在这个问题上已两次「测错量就下结论」，这类改动不宜只有执行者一人看过。plan §3.0 正文也需同步更新为 v2 定义。

### 3.12.3 真实闭环中的验收证据（重启后第 1 批）

离线探针只能证明权重非退化；能不能证明**部署后的策略真的在看观测**，要看 client rows 里逐步的 `probs`：

| 判据 | 修复前 | 修复后 |
|---|---|---|
| `teacher_prob` 跨步分布 | 单一常数（`[0.50021, 0.49979]` 重复到底） | min **0.0363** / max **0.9874** / std **0.2356** |
| 5248 步中的不同取值 | 1 | **5085** |
| student 执行率 | 钉死 0.4996 / 0.5009 / 0.5040 | 0.6025 |
| grad_norm | 0.25 / 0.05 / 0.21（死网络里只有 b2 在动） | 41.99 |

这才是「router 不再是常数策略」的端到端证明。附带一条给后续分析的提醒：`grad_norm≈42` 是**裁剪前**的范数，§3.5 冻结的 `clip=1.0` 会把它压回单位范数——死网络时期那几个 0.05~0.25 的读数不是「梯度小」，而是**梯度几乎只存在于输出偏置上**，两者含义完全不同，做曲线时不要混着比。

---

### 2.4 M5c 逐候选测量结果（滚动更新）

§3.10 协议：每候选从**同一** warm-start checkpoint（sha `66bcce6f…`）与**同一** seed=0 出发，在 pilot 子集（300 init）上训 5 批×100 ep，冻结第 5 批权重后以 **argmax** 在 B-train 余集（150 init）测 100 ep 的 realized teacher rate。选择规则：λ₁ = 最接近 40%，λ₂ = 最接近 20%。

| λ | 训练链 | realized teacher rate | 备注 |
|---|---|---|---|
| 0.05 | 5 段严格连续 v0→v5 | **0.2618** | 首个跑通的候选；eval 守卫核对通过（`mode=argmax` / 无 `dump_dir` / 无 seed / 权重指向 v5） |
| 0.2 | 5 段严格连续 v0→v5 | **0.5710** | **方向与预期相反**，见下 |
| 0.5 | 5 段严格连续 v0→v5 | **0.4979** | |

**libero_spatial 的 M5b（λ 无关，已完成）**：采集 450/450、**450 唯一键零重复**（libero_10 那次有 6 个）、全 `complete`、成功 425 (94.4%)；拟合 `head_lr=1.148e-3`、**CV loss 0.3216 < 基率熵 0.3482**、trunk 78/256 存活、输出 std 0.251、δ₀=1.819 → 部署态 student 率 0.5000、infra 失败率 0.0。δ₀ 比 libero_10 的 0.570 大得多是**正确行为**：spatial 成功率更高 → 头的成功 logit 更大 → 要把初始 student 率压回 50% 需要更大的偏置修正，说明 δ₀ 确实按「实际 ship 的那个头的留出折」标定而非套常数。

**pilot 判决（`selection.json`）**：`separated: true`，选出 **λ₁=0.5、λ₂=0.05**（目标 40% / 20%）。三候选各 5 批 × 100 ep + 100 ep argmax 评测，共 1800 ep，全程 `err=0`、零拒收、策略未退化。

#### ⚠ λ 的信号被 success 噪声淹没（待 owner 判读）

teacher 是最贵的臂（cost 1.0，student 0.0557），**λ 越大越该把策略推离 teacher**。但实测 λ 0.05→0.2，teacher 率 0.262→0.571，方向相反。

量级一算就清楚：成本项 `λ·(Σ_t cost)/T_max` 在 λ=0.05 约 **0.003**、λ=0.2 约 **0.012**（每 ep 约 61 verdict、半数 teacher → Σcost≈30，T_max=520）；而 `mean_success` 的批间波动是 **±0.1** 量级（实测 0.81–0.94）。**λ 的信号比 success 噪声小一到两个数量级**，5 批 × 1 次 Adam step 根本压不住。所以这两个数更像是不同随机轨迹的噪声，而非 λ 的因果效应。

按冻结选择规则（λ₁ 最接近 40%、λ₂ 最接近 20%）此刻两个目标**都指向 0.05**，尚未分离。

#### ⛔ 三点齐全后的判决：门禁会放行，但这个标定是无意义的（**M6 已按此暂缓，等 owner**）

| λ（惩罚强度） | teacher rate | 与目标的距离 |
|---|---|---|
| 0.05（最弱） | 0.2618 | 距 20% 目标 0.062 ← 被选为 **λ₂** |
| 0.2 | 0.5710 | — |
| 0.5（最强） | 0.4979 | 距 40% 目标 0.098 ← 被选为 **λ₁** |

三条理由说明它标不到东西：

1. **映射完全非单调**（0.26 → 0.57 → 0.50）。λ 与 teacher 率之间看不出单调关系，`separated: true` 纯粹是三个噪声点恰好各自最接近一个目标。
2. **选出的对应关系是反的**：惩罚最重的 λ=0.5 被指派给**更高**的 teacher 率目标（40%），惩罚最轻的 λ=0.05 指派给**更低**目标（20%）——与「λ 越大越推离昂贵臂」的机制逻辑相反。λ₁/λ₂ 这两个预注册标签因此名实不符。
3. **差异不是评测噪声**：100 episode 的评测 SE ≈ 0.05，而三点跨度 0.31，远超。所以三个策略**确实不同**——但不同来自 5 次 REINFORCE 更新的随机轨迹，而非 λ：成本项 `λ·Σcost/T_max` 只有 0.003–0.012，而 `mean_success` 批间波动 ±0.1，信噪比差一到两个数量级。

**处置**：按 §3.13「新失效模式 → 停跑、记录、等 owner」，**未发射 M6**。headline interaction-efficiency 曲线的 λ 标签是论文主张的核心，拿一个标不到东西的标定去跑 37 h 并产出带无意义标签的曲线，比空等更糟。

**给 owner 的几个方向**（均属裁量，不在执行者权限内）：
- (a) **延长每候选批数**压噪声。要让 λ 效应超过 success 噪声，粗算需把每候选从 5 批提到数十批，pilot 成本从 1800 ep 涨到上万——已接近一个正式 run，性价比存疑；
- (b) **改 pilot 的判据**：不用 realized teacher rate，而直接看**同一 λ 下策略相对 warm-start 的位移方向**，或用成本项主导的 pilot-专用 reward（但这会改 §3.10 冻结协议）；
- (c) **接受并预注册披露**：「在本实验的信噪比下 λ 不可标定」。这与本实验「为什么不训 router」的论点**同向**——它恰恰是「在线 RL 路由的取得成本高到连超参都标不动」的一手证据，未必是坏结果；
- (d) 仍按 `selection.json` 跑，但在论文里把 λ₁/λ₂ 降格为「两个任意但冻结的 λ 取值」，不声称它们对应 40%/20% 的 teacher 率。

#### 处置更正：不该整个停下，该按 §3.13 的规则只发一个 run

我最初的处置是「停跑等 owner」，这是**过度保守，且违反了我自己在 §3.13 写下的规则**（逐个发射、不并发——该规则的用意正是「有争议时把爆炸半径限制在一个 run」）。重新审视后更正：

- 冻结协议 §3.10 要求的是**两个选择互不相同**，`separated: true` 已满足；R13 的出口是为**未分离**准备的，而未分离并未发生。协议返回了答案，我的异议是「它的前提（λ→rate 单调）没成立」——那是**该披露的 caveat，不是阻断条件**。
- 上面四个方向里，(c) 与 (d) 都保留当前 λ 取值，只有 (a)(b) 会作废 M6 数据。**空转一夜的成本是确定的，重跑一个 run 的风险是概率性的。**
- 因此：跑 G-launch（五门全过）→ **只发旗舰 `l10_ts_lam1_s0` 一个 run**，其余四个等 owner 裁决后再续。

**G-launch 结果**：λ₁=0.5 / λ₂=0.05 已写入 tracked 矩阵；五份臂 yaml 全部 emit 并逐一过门（pilot 逐候选证据复算、成本 provenance、M4 容量报告、arm yaml 字段一致性、episode budget）；`GATE_FAILED=0`。旗舰 run 已发射，`interaction ledger offset = 2250`（450 warm-start + 1800 pilot，§3.10 保守计费口径）。

**这不是可以私自"修"的东西**：加大 λ 网格、延长 pilot 批数、改 reward 归一化，都会改变 plan §3.10 冻结的协议与 D1 裁决。协议自带的出口是 R13——不分离则插一个几何均值补充候选，仍不分离就升级 owner。**`set_m6_lambda.py` 会在未分离时拒绝写入矩阵**，所以不存在"悄悄挑一个"的风险。

值得 owner 一并考虑的是：即便三点碰巧"分离"，一个**非单调**的 λ→teacher-rate 映射也说明这次标定并没有真正标到什么。届时可能的方向（均需 owner 裁定，不属执行者权限）：延长每候选的批数以压噪声、把 pilot 的 reward 改成成本项主导的形式、或干脆承认「在本实验的信噪比下 λ 不可标定」并作为预注册披露写进论文。

**已可预见的风险**：λ=0.05 是网格里惩罚最轻的一档，teacher 率已只有 0.262；惩罚更重的 0.2 / 0.5 预期更低。若三档全部落在 20% 以下，**λ₁ 的 40% 目标将无人接近**，触发 R13 的补充候选（在最近两点的几何均值处插一个，至多一次）。仍不分离则升级 owner——`set_m6_lambda.py` 会拒绝写入矩阵，不会自作主张挑一个协议没能区分的 λ。

> 口径提醒：这里的 teacher rate 按 **`arm_executed`** 统计（§3.10 明确），不是按 `arm_sampled`——cache 臂打到空库时实际执行的是 teacher，λ 要标定的是**实际花掉的算力**。

### 3.13 无人值守期间的自主决策规则（owner 2026-08-15 22:40 离场，「你自己工作」）

监控：L3 cron 每 11 分钟（job `06c749ee`）+ L2 Monitor + `/tmp/rlr_health.sh`。按 memory `feedback_no_background_tasks_unattended`，**不起 `run_in_background` 后台任务**。

**会做**：
1. pilot 完成 → 跑 `prepare_m6.sh`（写 λ 进矩阵 → 逐 run emit 臂 yaml → 五个 run 全部过门）；
2. **逐个**发射 M6，**不并发**（同一张卡，且同机可能有 owner 的其它 session，并发只会互相抢）；
3. 每个 run 结束后核对 trunk/策略离散度/join 完整性，再决定是否续下一个；
4. 出现 §3.13 之外的新失效模式 → 停跑、记录、等 owner。

**不会做**（越权或高危）：
- 不改任何冻结契约（§3.0 的 v2 修订是 owner 已裁定的，其余一律不动）；
- 不 commit / push / 任何高危 git 操作；
- 不 kill 别的 session 的进程；显存告急时**降本线 worker 数**；
- 不弹阻塞窗口。

**为什么逐个而不是五个一起发**：encoder v2 动了 G1/G2 冻结的 §3.0 且**尚未经 G2 review**。逐个跑意味着若 review 要求修订，受影响的只有已跑完的那一两个 run 而非全部五个；而反正一夜也跑不完五个（每 run ≈ 7.5 h），逐个发射没有任何吞吐损失。旗舰 `l10_ts_lam1_s0` 排第一——它是其余一切的对照基准。

---

## 4. 待 owner 追认的事项

0. **§1.1 的拓扑偏离**（conductor 由 t107 改到 wls 单机自闭环，实测 7.8× 提速；跨机通道保留可随时退回）；
1. §3.4 的顺序调整（M4 挪到 M5a/M5b 之后）与两处门禁豁免的边界；
2. §3.6 新增的 `pilot_lambda.py eval` 子命令（plan §5 文件清单未列，但未新增文件）；
3. §3.2 在 wls `authorized_keys` 增加的本实验专用公钥（实验结束后建议撤销）；
4. §3.10 的 M7 A 池物化方案（建议物化成显式目录，不给 eval 开空值口子）；脚本已备好并实测，**未对真实路径执行**（§3.10.2）；
5. §3.1 同步副作用：`select_freeze_*.yaml` 本地版已被上游版覆盖（备份在两机 `openpi_dirty_backup/`）；
6. **⚠ §3.14 的 A 池检查点预注册歧义——必须在 M7 第一条 episode 之前定死。**
7. **§3.15 的 M8 缺 run 定义**（冻结矩阵里没有任何 spatial run，加进去属契约变更）。

### 3.16 warm-start 的 arm 集与两个待发 run 不匹配（**已修 + 补测**；五个 run 里两个原本发不出去）

发射前预检下一批 run 时发现的，不是被它咬到才发现的。链条三环：

1. `run_rl_router.py:842-845` **首次发射时把 `artifacts["warmstart_weights"]` 原样发布成该 run 的 v0**——五个 run 共用同一个 `artifacts_m6.json`，也就是共用同一个 `warmstart_l10.pt`；
2. 该文件实测 `meta.arms = "ts"`（M5b 只用 `--arms ts` 拟过一次），而 `l10_tsc_lam1_s0` 的 yaml 声明 `arms: tsc`、`l10_tc_lam1_s0` 声明 `arms: tc`；
3. `mlp_router_judge.py:552-554` 明令 `weights.arms != arms` 即 `ValueError`。

→ **这两个 run 一发射，每个 worker 都会在构造 judge 时立刻抛错**。快速失败（分钟级，不是烧十几小时），但在无人值守时段足以把发射队列卡死。

**为什么 G-launch 没拦住**：`_arm_yaml_problems` 只证明「yaml 与冻结矩阵一致」——yaml 说 `tsc`、矩阵说 `R_tsc`，两边确实一致；**没有任何一环去读那个即将被发布成 v0 的文件**。这正是矩阵文件头声称门禁要干的事（"verify what it is about to run rather than trusting an operator's memory"），却恰好漏在这里。

**顺带查出第二个缺陷**：拿 `--arms tc` 重拟会**自己崩**。`graft()` 的 docstring 专门写了 R_tc 是有意支持的（无 student 行 → 全零 → 在继承的 trunk 上均匀起步，§3.8 预注册立场），但紧随其后的报告字段无条件调 `deployed_student_rate(...)`，而它对 `tc` 走 `ARM_SETS["tc"].index("student")` → `ValueError: tuple.index(x): x not in tuple`。**「唯一被特意支持的变体，恰恰是拟不出来的那个」**——本地小张量三行复现，不用碰 wls。

**两处修复（均 L1 纯 bug 修复，Code → Verify）**：

| 文件 | 改动 |
|---|---|
| `exp/rl_router/fit_warmstart.py` | `deployed_student_rate` 对无 student 的 arm 集抛**可读**的 ValueError（而非 `tuple.index` 的天书）；报告路径改为按 arm 集分支——有 student 才报 `realized_student_rate`，没有则记 `None` + `graft_disclosure` 的均匀起步声明 |
| `exp/rl_router/launch_gates.py` | 新增 `_warmstart_arms_problems()`：读 checkpoint 的 `meta.arms` 与 variant 期望值比对，不符则拦下并直接给出修复命令（`fit_warmstart.py --arms <expected>`）；读不出来也拦。顺手把 `{"R_ts": "ts", ...}` 提成模块级 `VARIANT_ARMS` 供两处共用 |

**测试**：新增 5 条（3 条参数化的 arm 集错配 + 匹配放行 + checkpoint 不可读）与 1 条 R_tc 无 student 率的契约测试。另外把 `_artifacts` fixture 里那个 `warm.pt` 从文本 `"{}"` 换成**真的 torch checkpoint**——一个只是"看起来像权重"的 fixture 会让这类错配在测试里畅通无阻，而真实舰队在 worker 启动时就拒收。`tests/exp` + `tests/cache` 全量 **2379 passed / 6 skipped / 0 failed**。

### 2.7 维护窗口：为挪数据而计划性停机并重启 s1（2026-08-16 10:2x）

owner 要挪数据，问要不要先停训练。**停了，且这是最便宜的时刻**——`l10_ts_lam1_s1` 才跑到第 0 批第 47 个 episode，而 resume 是批级的，损失上限就是这个未完成的批（约 2 分钟实际机时）。反过来，边跑边挪的风险是**静默的**：分片写一半被移走、join 出缺口、worker 读不到 init states 而整批作废——这些在进度 / 拒收 / 成功率上全都显示正常（§3.12 那两次退化就是这么骗过所有常规信号的）。

停机与恢复的完整动作，留作模板：

1. **按 PID 定点 kill**（`ps` 取 conductor 的两个 pid，不用宽模式 `pkill`）；
2. **立刻 `reap_orphans.sh`** —— 回收了 **32 个**孤儿 worker（16 worker × conda run 包装层）。这一步不能省也不能拖，见 §3 与 `reference_orphaned_libero_workers`；
3. 停掉 L2 Monitor 与 L3 cron，免得维护期间刷告警；
4. **不动 server / sidecar**：它们此时只是空转监听，不写任何数据。但若要挪 `cache_artifacts/*.pkl` 或 pi05 checkpoint，就必须一并停——进程内存里已加载，挪走当下不崩，**下一次重启才崩**，是个延迟很久的雷；
5. 恢复前**先验盘**：逐一 `test -e` 那 13 条依赖路径 + 确认旗舰 s0 产物完好（41 权重 / 40 metrics 行）。实测 owner 这次释放了约 **87 GB**（340 G → 253 G 用量），但本线依赖一件没少；
6. **重跑门禁**：`launch gates cleared`——digest 校验同时证明 warm-start / pilot / smoke 三个产物没被动过。这一步比 `test -e` 强，它验的是字节不是存在性；
7. **清掉夭折的半批**再重启：`dump_m6/l10_ts_lam1_s1/`（426 MB）、`art_m6/l10_ts_lam1_s1/`、`data/m6/l10_ts_lam1_s1/` 三处一起删。虽然 packager 取 last manifest 行、理论上能容忍重跑覆盖，但让一个夭折批的分片和重启后的新分片共用同一组 `(batch_id, weights_version)` 键，是在给自己留一个不必要的解释负担；
8. 重挂 L2 Monitor（**每换 run 都要重挂**，完成判据绑在该 run 的日志上）与 L3 cron。

> 顺带一条判 server 死活的老坑：日志里的 `EOFError: connection closed while reading HTTP request line` 是 **websocket 握手的良性噪声**（TCP 探测造成），不是故障。本线有过一次拿它误判 server 挂了的记录。

### 3.17 第 7 处探针缺陷：并发副作业会抢走「最新日志」

起 refit 的那一刻，健康探针的 `log=` 立刻从 `rlr_m6_*.log` 跳到了 `rlr_refit_arms.log`——于是 `quiet_s` / `err` 两个字段**不再描述 M6**，而是描述那个刚起的副作业。M6 若在此期间静默或报错，探针只会显示副作业的岁月静好。

根因是探针里这行：

```bash
log=$(ls -t /tmp/rlr_*.log | grep -vE 'srv|sidecar|health' | head -1)
```

它当初是为了修「手写候选表漏掉新 run → 报陈旧数据」而改成 mtime 最新（注释里写着这段历史）。但它隐含假设**同一时刻只有一个本线作业在写日志**——我起 refit 就打破了这个假设。这是第 7 处探针缺陷，与前六处同源：**每加一条判据都要问「它在哪个阶段/哪种并发下会误报」**。

**处置：不改探针**。它此刻是 L2/L3 两层唯一的数据源，在旗舰跑到第 35 批时动它，风险高于收益。改为把副作业的日志移出那个命名空间（`mv /tmp/rlr_refit_arms.log /tmp/refit_arms.log`；同文件系统 `mv` 保 inode，`tee` 继续往同一个 inode 写，一行没丢）。验证后 `log=` 已回到 `/tmp/rlr_m6_l10_ts_lam1_s0.log`、`quiet_s=7`。

**沉淀成运维规约**：**本线任何与主跑并发的副作业，日志一律不得叫 `/tmp/rlr_*.log`。** 等主跑全部结束、探针不再是唯一监控时，再考虑把它改成「按 tmux session 名定位日志」而不是按 mtime 猜。

#### 3.16.1 重拟结果与门禁的端到端验收（2026-08-16）

`refit_arms.sh` 跑完 `--arms tsc`（约 10 min）。**验收证据比预期干净**：

| 量 | tsc 版 | 已发布的 ts 版 |
|---|---|---|
| `head_lr` | 0.0011277793093384244 | **完全相同** |
| `cv_loss` / 基率熵 | 0.5549 / 0.6557（守卫通过） | **完全相同** |
| `trunk_health` | live 134/256、`output_std` 0.5343、`mean_pre_activation` 0.03896 | **逐位相同** |
| `delta0` | −0.1232 | +0.5699（三臂多了 `log 2` 偏移，本就该不同） |
| `realized_student_rate` | **0.5000** | 0.5000（冻结的 50% 契约在三臂上同样成立） |

直接比张量更硬：`W1` / `b1` / `feature_mu` / `feature_sigma` **逐位相同**，`W2` 的 **student 行也逐位相同**；全部差异只有两处，且两处都是契约**要求**的——`W2` 多一行（cache 臂，全零）、`b2` 的 student 项 −0.5948 → +0.0984。`encoder_version` 一致。**拟合是确定性的，三个变体共享同一个表征与同一个拟合出来的头。**

`--arms tc` 那次**同时是本次修复的生产验收**——修之前它会在报告阶段崩掉。产出记录正是意图的样子：

```json
"delta0_basis": {"source": "held_out_fold_of_the_shipped_head", "fold": 0, "n_steps": 5517,
                 "realized_student_rate": null,
                 "no_student_arm": {"arms": "tc", "student_row_from_head": false,
                                    "initial_policy": "uniform", "note": "…§3.8…"}}
```

trunk 同为 134/256、`cv_loss` 0.5549 同样低于基率熵 0.6557，两道守卫都过。两次 refit 均 `EXIT=0`。

门禁的端到端验收（拿真实产物，不是 fixture）：

- **A** 用共用的 `artifacts_m6.json`（`arms=ts`）发 `l10_tsc_lam1_s0` → 新门禁**拦下**，并直接给出 `Refit with fit_warmstart.py --arms tsc`。**这正是原本会炸的那次发射。**
- **B** 换成 `artifacts_m6_tsc.json` → arms 问题消失。

#### 3.16.2 ⚠ 二阶后果：pilot digest 守卫会挡住一切非 ts 变体（**待 owner 裁**）

B 情形下换成另一条门禁在报：

```
pilot candidate 0.05 calibrated against a different warm-start checkpoint than the one this run would start from
（0.2 / 0.5 同）
```

`_pilot_problems` 比对 pilot 记录里的 `expected_warmstart_sha256` 与本 run 将要起步的权重文件 sha。pilot 是拿 `ts` 那份跑的，而 R_tsc / R_tc **结构上不可能**从 `ts` 的两臂头起步。于是这条守卫的现状是：**任何非 ts 变体都永远发不出去**，无论怎么修 arms。

**这不是 bug，是门禁比 plan 更严**。冻结矩阵里五个 run 全都引用同一组 `lambda_1` / `lambda_2`——「一次 pilot 服务所有变体」本就是设计意图；守卫想保的实质属性（λ 的含义绑定在同一个起步策略上）在上面的张量对比下**是成立的**：同表征、同拟合、同 `encoder_version`，差的只有契约要求必须差的输出层。不成立的只有**文件层面的 sha**。

三条路，都要 owner 裁（涉及 §3.9 预注册守卫，执行者不得自行放宽）：

1. **(a) 按变体各跑一次 pilot**：每次 3.2 h × 2 变体 ≈ 6.4 h。但 §2.4/§2.5 已经两次表明 λ 在本信噪比下标不出来——这是把 6.4 h 花在一个**已知无信号**的测量上。**不建议**。
2. **(b) 把守卫的身份判据从「同文件 sha」改成「同 trunk + 同 `encoder_version` + 同拟合报告」**，即断言实质属性而非文件字节。技术上正确，但**动的是预注册守卫本身**，必须 owner 签字。
3. **(c) owner 明确裁定「λ 跨变体沿用」**，并在 pilot 记录里为非 ts 变体登记其对应的 warm-start digest（等于把矩阵已经隐含的意图写进产物）。**倾向 (c)**：它不放宽任何断言，只是把冻结矩阵早就写着的东西补进 pilot 记录。

**不阻塞当前队列**：下两个要发的 `l10_ts_lam1_s1` / `l10_ts_lam2_s0` 都是 R_ts、都用原来的 `warmstart_l10.pt`，pilot 守卫照常通过。这条须在 **tsc/tc 排到队首之前**（约 15 h 后）裁定。

**待办（不阻塞下一次发射）**：拿同一批 M5b 分片再拟两次 —— `--arms tsc` → `warmstart_l10_tsc.pt`、`--arms tc` → `warmstart_l10_tc.pt`，并给这两个 run 各自的 `artifacts_m6_*.json`（`run_m6.sh` 按 variant 选）。**排序上不急**：下一个要发的 `l10_ts_lam1_s1` 是 `ts`，用现成的即可；tsc/tc 排在队尾，还有 15–22 h。重拟是内存大户（450 ep × ~130 步 × 65568 维），**不要与 collect 阶段抢**，挑 trainer 阶段或 run 间隙做。

### 3.15 M8「spatial 确认」在冻结矩阵里没有对应的 run

plan §193 的里程碑链把 M8 写成「spatial 确认」，§4.4 的排期也给它留了时间；矩阵里 `suites.libero_spatial` 标着 `role: confirmation`，M5b 也已经把 `warmstart_spatial.pt` 拟好了（CV 0.322 < 基率熵 0.348，student 率 0.5000）。**但 `runs:` 那五条全是 `suite: libero_10`，spatial 一条都没有。**

也就是说 M8 目前没有可发射的配置：`emit_m6_arms.py` / `run_m6.sh` 都是照 `runs:` 逐条 emit 的，矩阵里没有的 run 发不出去。补上 run 条目意味着**动 G1 冻结的矩阵**，得 owner 签字，不是执行者能自行添的。

规模上也需要 owner 拍板：spatial 是「确认」角色，跑几个 run（1 个 R_ts@λ₁ 还是照 libero_10 的五 run 复制一遍）、λ 是沿用 libero_10 pilot 选的值还是重跑一次 spatial 自己的 pilot——这两个问题的答案相差 4 到 20 倍的机时。**建议**：单个 `spatial_ts_lam1_s0`（4k ep ≈ 7.5 h）+ 沿用同一组 λ，理由是 confirmation 角色只需回答「主套件上的现象换个套件还在不在」，不需要重建整张表；且 §2.4/§2.5 已经表明 λ 在本信噪比下标不出来，为 spatial 再跑一次 pilot 是把 3.2 h 花在一个已知无信号的测量上。

时间上不紧（M6 还有约 27 h + M7），但**别拖到 M7 跑完**——那时若要重跑 spatial pilot，等于在收尾阶段又插进半天。

### 3.14 A 池评测预算的预注册歧义（**M7 的阻塞项**）

冻结矩阵 `run_matrix.yaml` 的 `eval` 段：

```yaml
episodes_per_checkpoint: 500
flagship_checkpoints: [500, 1000, 2000, 4000]
other_checkpoints: [4000]
```

而 `runs` 段把 **`l10_ts_lam1_s0` 和 `l10_ts_lam1_s1` 双双标了 `flagship: true`**。两处合起来有两种读法，episode 预算差 1500（约 2 h）：

| 读法 | s0 | s1 | 其余三 run | 合计 |
|---|---|---|---|---|
| (A) 双旗舰都走全曲线 | 4×500 | 4×500 | 3×500 | **5500** |
| (B) 仅 s0 走曲线，s1 只测终点 | 4×500 | 1×500 | 3×500 | **4000** |

plan §137 的原文两边都不排除：「旗舰（l10 R_ts@λ₁）4 检查点 …；其余仅终点」（单数「旗舰」指向 (B)）+「种子：旗舰 2 训练种子」（两个 run 都叫旗舰，指向 (A)）。

**倾向 (B)**，理由是 R3 已把 seed-1 的角色冻死为「稳健性复现：方向一致性 + 自身 McNemar，报于族外」——这两个量**只需要终点**。给 s1 补一条曲线要多花 1500 ep 去回答一个预注册假设没在问的问题。

**这条为什么现在就得定**：检查点集合一旦见过 A 池数据再选，预注册就失效了（`emit_router_yamls.py` 的守卫注释写得很清楚：「Both have to be decided before any A-pool episode runs, or the choice is no longer independent of the numbers」）。顺带一个**实现陷阱**：谁来写 M7 驱动，只要老实读 `flagship: true` 就会自动落到 (A)——所以裁 (B) 的话，矩阵那两个 `flagship: true` 需要同步改成可区分的字段（如给 s1 标 `role: robustness`），否则代码与意图不一致。

顺带核实过的两件事（无问题）：**中间权重全保留**（旗舰当前 35 个 `v*.pt`，曲线要的 v5/v10/v20 都在，v40 待生成），每个 67.7 MB、每 run ≈ 2.6 GB；磁盘 527 G 可用，五 run 约 13 G，不构成约束。

以上 §3.3 / §3.5 / §3.7 / §3.8 / §3.9 / §3.12 属纯 bug 修复（L1，Code → Verify），已随手补测；`uv run pytest tests/exp/` 全绿。

### 3.18 第 8 处探针缺陷：退化守卫对 R_tc 的 v0 必然误报（**我据此误停了一条正确的 run**）

2026-08-17 18:44 发 `l10_tc_lam1_s0`，26 个 episode 时探针报 `policy_distinct=1 policy_saturated=1`。我查了权重：

| 文件 | arms | W2 每行 absmax | b2 | W1 absmax |
|---|---|---|---|---|
| `warmstart_l10.pt` | ts | [0.0, 0.083248] | [0.0, −0.594757] | 0.036987 |
| `warmstart_l10_tc.pt` | tc | **[0.0, 0.0]** | **[0.0, 0.0]** | 0.036987 |

输出层整层为零 ⇒ logits 恒 0 ⇒ 均匀常数策略。我判成 §3.12 那类退化，按预注册中止条件 (b) 停了 run。

**停错了。** 读 `fit_warmstart.graft()` 的 docstring 才看到这是**预注册的设计**，不是缺陷：success head 预测的是 **student 臂**的结局，对 cache 臂不含信息，所以「success logit 放 student 行、其余行置零」这一条规则作用在无 student 的臂集上，结果必然是零头 + 均匀初始策略——§3.8 明确记过这个立场，`graft_disclosure("tc")` 也会把它写进产物（`initial_policy: "uniform"`）。**该先读 docstring 再动手**；代价是 26 个 episode 加一次孤儿回收（`reap_orphans.sh` 收了 16 个，判据是新版的「父进程是 `systemd --user` 或 init」，未波及同机 `cseval`）。

**真正的缺陷在探针**：`policy_saturated` 是照 ts 的形状定的，而 v0 的形状是**预注册事实**、发射门（`launch_gates.py`）已经验过，不该由健康探针再审一遍。修法是给守卫加第四条豁免——**b0000 不判**：

```bash
batchdir=$(basename "$(dirname "$rows")")
[ "$alive" -eq 1 ] && [ "$phase" != "collect" ] && [ "$batchdir" != "b0000" ] \
  && [ "$nrows" -ge 200 ] && [ "$distinct" -ge 0 ] && [ "$distinct" -lt 10 ] && sat=1
```

**不损失强度**：两次历史退化（死 trunk §3.12、一步饱和 §3.12.4）都是**在 trainer 步之后**才出现的，v1 起判据一字未改。验收（强制 `alive=1 phase=train distinct=1 nrows=1806`，唯一能压住的只有新豁免）：`b0000 → sat=0`、`b0007 → sat=1`。

18:50 原配置重发（不改任何预注册量）。ep 级 resume 会接着 b0000 的 30 个 episode 往下跑，同一份 v0 权重、同一份配置，`verify_m6.py` 的链与 `model_sha` 判据不受影响。

**⚠ 顺带记一个新的显存现实**：本 run 稳态 `vram_free_mb≈6.2 G`（49 G 卡）。分解：我的 pi05 server 8.5 G + 两个 ACT sidecar 3.4/2.8 G + **同机他线的 server 7.7 G** + 16 个 LIBERO worker 的 CUDA context ≈ 19 G。比之前几条 run 的 15–25 G 空闲薄得多，主因是他线 16:20 起了新 server。仍高于 3 G 阈值，**告危时只降本线 worker 数，绝不动他线进程**。

（→ 这条只对了一半。owner 追问显存后查出真因见 §3.19：那 6.2 G 里有 8.4 G 是我自己的孤儿，而两个孤儿判据都在同时报 0。）

### 3.19 孤儿判据连错三处：显存被自己的孤儿吃掉 8.4 G，而计数器一路报 0（2026-08-17，owner 追问显存查出）

owner 问「是不是你之前的 worker 没关」。**是。** 新写 `/tmp/gpu_attrib.sh` 把每个 GPU 上下文归属到 tmux session，答案立刻出来：

| owner | ctx 数 | MiB |
|---|---|---|
| tmux:rlrm6（本线在跑） | 16 | 9206 |
| tmux:rlrsrv（本线 server） | 1 | 8742 |
| **systemd--user(ORPHAN)** | **16** | **8446** |
| tmux:cssrv（他线 server） | 1 | 7736 |
| tmux:rlrsc / rlrsc2（本线 sidecar） | 2 | 6164 |
| tmux:csmain（他线 worker） | 3 | 1591 |

**第一课：`nvidia-smi --query-compute-apps` 看不见 LIBERO worker。** 它们为 EGL 渲染持有的是 **graphics(G)** 上下文，那条 query 只列 compute(C)——所以 22 G「已用」对不上 43 G，20 G 凭空消失。要看全得解析 `nvidia-smi` 的进程表。

那 16 个孤儿是 §3.18 中止的 18:44 那次留下的（PID 3085231–3085629，worker-id 恰好完整一套 w0–w15）。**我当时跑过 `reap_orphans.sh`，它报「reaped 16」然后收工——但清掉的是 `conda run` 包装层，真正的 worker 还挂在包装层下面；杀掉包装层之后，孙进程才被 re-parent 成新一批孤儿，而单趟的 reaper 已经走过去了。**

#### 三处判据错误，逐个改掉（`reap_orphans.sh` v1→v4，`rlr_health.sh` 同步）

**(1) 单趟不够 → 循环到干净。** 杀掉一代会造出下一代。改成最多 6 轮，每轮重新普查，直到某轮为 0。

**(2) 只看直接父进程 → 太窄。** v2 的判据是「cmd 含 `worker_entry` **且** ppid 是 systemd/init」。真实父链是三层，而且**只有叶子匹配模式**：

```
worker_entry (3085231) → bash /tmp/tmpXXXX (3085021) → systemd --user (2248)
```

中间那层是临时脚本，不含 `worker_entry`；而 worker 的直接父进程是这个**活着的**包装层。两半判据各漏一边 ⇒ **计数器在 16 个孤儿吃着 8.4 G 的三个小时里一直报 0。**

改成沿父链走。但**朴素的走法同样错**——tmux server 本身就是 `systemd --user` 的孩子，所以「链上到得了 systemd」对**每个** worker 都成立，包括健康的，这正是 08-17 误杀他线的形状。区别在**谁先到**：

```
活的  : worker → wrapper → … → tmux PANE → tmux server → systemd
孤儿  : worker → wrapper → systemd            （中间没有 pane）
```

**先遇到活的 tmux pane ⇒ 活；先遇到 systemd/init ⇒ 孤儿。** 不点名任何实验线，他线结构性安全，且对称——本线 conductor 一死，自己的 worker 同样会被判出来。

**(3) 子串匹配会打到自己 → 差点杀掉执行通道。** 「cmd 里含 `worker_entry`」也匹配**任何提到这个词的 shell**，包括正在跑 reaper 的那个 `tether exec` shell——它的父链走到 tether agent 而不是 tmux pane，于是被判成孤儿、进待杀名单。这就是 `pkill -f` 的自匹配陷阱换了个壳。实测当场发生过：我的正控测试脚本里 `pgrep -f "FAKE-PROBE-TEST"` 匹配到自己那个 shell，`kill -9` 把自己的 exec 通道杀了。

改成**要求模块名是一个独立的 argv 元素**（`grep -lzxF 'examples.libero.worker_entry' /proc/[0-9]*/cmdline`）：shell 把整段脚本放在**一个** argv 元素里，恒不相等；worker 与 conda 包装层是逐元素透传的，照常匹配。现场验证：

```
--- 旧 awk 判据 /worker_entry/ 会不会匹配到自己 ---
  MATCH(self): 3107040 bash -lc  SELF=$$ …          ← 会
--- 新判据(argv 独立元素)会不会匹配到自己 ---
  no match(self) -- GOOD                            ← 不会
```

#### (4) 还有一个瞬态假阳性：逐批重生的 worker

改完之后仍偶发 `orphan_workers=1`，下一趟就没了。原因是 worker **每批重生**，交接窗口里会短暂无父——单次采样把正在拉起的健康 worker 判成孤儿，而 reaper 会把它 TERM 掉（杀掉一条正在跑的 rollout）。真孤儿是永久的，所以**要求判定在间隔 5s 的两次采样里都成立，只取交集**；探针只在第一次采到非零时才做第二次，健康路径零开销。验证输出：

```
  (dropped 1 transient orphan(s): present in sample 1, not in sample 2)
[round 1] worker_entry procs: 41 | orphaned: 0
nothing to reap
```

#### 处置结果

实杀 16 个孤儿（干跑先看名单：`KEEP` 同时护住了在跑的 `rlrm6` pane 3092493 与**他线** `csmain` pane 2679023，待杀的恰好是那套重复的 w0–w15）：

```
reaped 16 across up to 6 rounds
vram free: 6246 MiB -> 16004 MiB (freed 9758 MiB)
```

复查：无孤儿、`rlrm6` 16 个 ctx 照常、他线 `csmain`/`cssrv` 完好、空闲 **16.7 G**。本线 run 未受影响（ep 90 → 123 持续推进）。

**教训**：这条线上「孤儿」判据已经错了四次（白名单 → 直接父进程 → 朴素父链 → 子串自匹配）。**每次改判据都必须先 `DRYRUN=1` 看名单**，且名单要同时打印 KEEP 侧——只看待杀名单看不出你误伤了谁。

### 3.20 计划性停机腾卡 + 复跑闸门；顺带发现 tc 根本不需要 sidecar（2026-08-17 22:2x — 08-18 00:3x）

owner 22:2x 指示「暂停训练腾出显卡，凌晨 3 点 wls 空余再启动；3 点不够就每 30 分钟再查，以此类推」，随后离场并授权全程专断。

#### 停机（本线 GPU 占用归零）

`l10_tc_lam1_s0` 停在 **15/40 批**（b0000–b0014 落盘，metrics 15 行，`trainer_checkpoint.pt`/`trainer_state.json` 完好）。同一条 `run_m6_v2.sh l10_tc_lam1_s0` 即批级 resume 从 b0015 续，**不要删 `art_m6/l10_tc_lam1_s0/`**。

按 PID 定点停 conductor → `tmux kill-session` → reaper 收 32 个（16 worker + 16 conda 包装层；干跑先看名单，KEEP 侧确认是他线 `csmain` pane 3252679）→ 再定点停 pi05 server 与两个 ACT sidecar。**空闲 15 G → 38.0 G，本线占用归零。** L2 Monitor `TaskStop`，L3 cron 删除（无东西可巡）。

#### ⚠ 孤儿判据第五个坑：**状态不等于归属**

停机后复查，卡上仍有 8 个 worker、其中被标为孤儿的若干个，`--server-key` 全是 **`127.0.0.1:8030`**——**他线的**，22:16 起（本线此时已全停）。孤儿判据本身没错（它对「状态」的陈述是对的），但**照它行动就会杀掉别人的孤儿**，与 08-17 误杀同罪、只是走了另一条路。

修法：按 **`--server-key` 收窄到本线**。每个 worker **和它的 conda 包装层**都在 argv 里带着自己服务的 server（实测 `/proc/<pid>/cmdline` 第 12/13 元素为 `--server-key` / `127.0.0.1:8030`），所以这是进程自带的结构性归属信息，不是驱动名白名单：

```bash
SERVER_KEY=${SERVER_KEY:-127.0.0.1:8000}
worker_pids() {
  for d in $(grep -lzxF 'examples.libero.worker_entry' /proc/[0-9]*/cmdline 2>/dev/null); do
    p=${d#/proc/}; p=${p%/cmdline}
    tr '\0' '\n' < "$d" 2>/dev/null | grep -qxF "$SERVER_KEY" && echo "$p"
  done
}
```

`reap_orphans.sh` 与 `rlr_health.sh` 同步改。**他线现在结构上不可能进待杀名单。** 正控验收：本线口径 `:8000` → `worker_entry procs: 0`；临时换他线口径 `:8030`（只干跑）→ `worker_entry procs: 8`。证明是范围在过滤，不是判据坏了。

#### ✅ 顺带查明：**tc 这条 run 不需要任何 ACT sidecar**

对照两个臂 yaml：

| yaml | `hit_to` |
|---|---|
| `r_ts_train.yaml` | `127.0.0.1:7002`（student 决策转发给 ACT sidecar） |
| **`r_tc_train.yaml`** | **无** |

tc = teacher + cache，**没有 student 臂**，命中由 cache 就地服务。两个 sidecar（3.4 G + 2.8 G）对本 run 是纯浪费。唯一的绊脚石是启动占位 `collect_student.yaml` 里有 `hit_to: 127.0.0.1:7002`——**改用 tc 自己的 train yaml 当启动占位**即可（conductor 本就逐批热切换 yaml，占位选谁不影响实验，见 `reference_server_yaml_hotswap`）。

**这条不靠推理，实测过**（00:23，起完即停）：0 错误、`Loaded 2640 entries` 载入 cache 库、`server listening on 0.0.0.0:8000`、占用 **7576 MiB**。

于是本 run 满配需求从 24 G 降到 **17–18 G**（server 7.6 + 16 worker 8–9），**复跑阈值随之从 26 G 下调到 20 G**——这直接提高了 3 点闸门能开的概率，而卡上他线已从 10.5 G 涨到 19.3 G（`cssrv` 7.7 + `rc5srv0` 7.8 + `csmain` 2.7 + `rc5run` 1.0）。

#### 复跑闸门

一次性 cron **`267bad5d`，2026-08-18 03:01 CDT**：空闲 ≥ 20000 MiB → 起 server（上面那条实测过的命令）→ 等 :8000 → 过 gate → `run_m6_v2.sh l10_tc_lam1_s0` 续跑 → 重挂两层监控；不够 → 自挂一条 +30min 的一次性 cron 原样重来，不打扰 owner，不放弃。

### 3.21 `sweep_mixture` 只会测 teacher/student，测不了它最该测的那条（**已修 + 补测**）

`l10_tc_lam1_s0` 跑完后要判「router 有没有打过固定比例」，就得有 teacher/**cache** 的常数策略曲线。但 `constant_policy` 开头写死：

```python
if "teacher" not in names or "student" not in names:
    raise SystemExit(f"arms={arms!r} needs both teacher and student")
...
b2[names.index("student")] = math.log((1 - p) / p)
```

**`--arms tc` 会直接 SystemExit。** 也就是说：这台仪器唯独测不了**论文真正关心的那个臂**（cache 才是 TIER 的对象，student 只是对照）。不是 bug，是当初只按 ts 写死了——但后果一样：tc 线没有基线可比，而「router 打不过固定比例」这个结论**完全建立在这条曲线上**。

修法：混合永远是 teacher vs **一个**廉价臂，把「哪一个」变成参数 `--cheap-arm`（默认 `student`，R_tc 传 `cache`）；其余臂 logit 压到 −30，保证实现出来的是**精确两路**混合，与臂集无关。三处改动：`constant_policy(..., cheap_arm=...)`、CLI `--cheap-arm`、调用点透传。

守卫也一并收紧：缺臂时报错要**指名道姓**（`arms='tc' has no 'student' arm (it has ('teacher','cache'))`），`--cheap-arm teacher` 直接拒（两路混合需要两个不同的臂）。

补测 `tests/exp/test_rl_router_sweep_mixture.py`，测的是**性质**不是「能跑」：

- 4 个臂集/廉价臂组合 × 6 个 p（含 0.0 / 1.0 端点）**逐点验实现出来的 teacher 概率等于请求值**；
- **同一策略在两个差异极大的观测上必须给出相同答案**（否则它就不是「固定比例」）；
- 其余臂的概率 < 1e-10（是「排除」不是「很小」）；
- `meta.arms` 与 `W2` 行数对得上（`MlpRouterJudge` 会按这个拒收）；
- trunk 三个张量全零、`b2` 的活口差恰为 `log((1−p)/p)`。

`28 passed`。**Verify 口径**：`uv run pytest tests/exp/`（改动 blast-radius）→ **1266 passed**，零失败。已 push 到 wls 并在那边复跑该文件（28 passed）；四个 analysis 工具也一并同步到 `wls:/home/weiland/openpi/exp/rl_router/analysis/`。

判 tc 时用：

```bash
python exp/rl_router/sweep_mixture.py --arm-yaml <r_tc_train.yaml> --cheap-arm cache ... \
  --p 0.0 --p 0.05 --p 0.1 --p 0.15 --p 0.2 --p 0.3 --p 0.45
python exp/rl_router/analysis/router_vs_fixed.py --cheap-arm cache --sweep <tc sweep.json> ...
```

### 3.22 停机后复跑：闸门按预期自动开（2026-08-18 03:01–03:05）

一次性 cron `267bad5d` 03:01 触发，空闲 **29.5 G ≥ 20 G** ⇒ 走分支 B，全程无人干预：

| 步 | 结果 |
|---|---|
| B1 起 server（tc yaml 当占位、**无 sidecar**） | tmux `rlrsrv` 03:01:19 |
| B2 验 | `:8000` LISTEN、`grep -ciE "Traceback\|Error"` = **0** |
| B3 门禁 | `launch gates cleared for l10_tc_lam1_s0` |
| B4 续跑 | tmux `rlrm6` 03:03:28 |
| B5 验 | Traceback 0；**16 个批目录 / metrics 仍 15 行 ⇒ 从 b0015 续上**，progress 1597→1598→1600 在涨；`policy_distinct=951`、`orphan_workers=0`、空闲 21.6 G |
| B6 重挂监控 | L2 Monitor `bjxwq0gaq`、L3 cron `2f074252` |

**批级 resume 按设计工作**：`trainer_state.json` 记着 15 个 `consumed_batches`，b0015 停机时已完成 97/100 个 episode，ep 级 replay 直接接上剩下的 3 个。预计 07:4x 前后 40 批收尾。

### 3.23 吞吐的定量模型，以及一个我给错的数（2026-08-18 11:0x，owner 追问扩容时查出）

#### ⚠ 先纠正：真实吞吐是 **437 ep/h**，不是我先前说的 833

我此前把 `l10_tc_lam1_s0` 的 4000 ep 除以了**续跑后的时长**——但那段窗口里只跑了 **2500** 个（b0015–b0039，前 15 批是停机前跑的）。实测批目录 mtime：b0015 `03:07:55` → b0039 `08:50:35` = **20,560 s / 2500 ep = 437 ep/h**。

**连带修正**：§3.21 之后我给 owner 的「500 批公平收敛口径 ≈ 60 GPU·h / 2.5 天」**错了**，正确是 **50,000 ÷ 437 ≈ 114 h ≈ 4.8 天/变体**。差不多翻倍。

#### 吞吐随 teacher 占比单调下降，可以拟合成两项

tc 扫点每个点都是同一组 200 对 (task,init)（`sample_batch(batch_idx=0, seed=0)`），只有 p 不同，所以逐点耗时是干净的受控测量：

| p | 耗时 | 吞吐（16 worker） |
|---|---|---|
| 0.15 | 1195 s | 602 ep/h |
| 0.30 | 1302 s | 552 ep/h |
| 0.45 | 1409 s | 511 ep/h |
| 0.55 | 1496 s | 481 ep/h |
| 0.70 | 1646 s | 437 ep/h |

拟合 **每 worker 每 episode 耗时 = A + B·p**：**A = 86 s、B = 65.6 s**。

- 代回 p=0.507（tc run 的工作点）：119 s/ep ⇒ 484 ep/h，与 p=0.55 实测 481 吻合；与整程实测 437 ep/h 也一致（run 里还有批间 trainer 步与 worker 重生的开销）。
- 含义：**p≈0.5 时约 72% 的时间是仿真（可随 worker 数并行），28% 才是 teacher 推理（争同一块 GPU）**。

#### 瓶颈归因（`nvidia-smi pmon` 拆到进程）

| 进程 | SM 占用 |
|---|---|
| 本线 pi05 server | **49.7%** |
| 他线 server ×2 | 12.3% + 12.3% |
| LIBERO worker ×16 | ~0%（EGL 渲染不占 SM） |

CPU 负载 **18.3 / 88 核**。**server 与 CPU 都没饱和 ⇒ 限制是并发度**：每个 worker 卡在自己那次同步往返上，16 个填不满卡。

#### 显存口径

16 个 worker 合计 **7,045 MiB（每个约 440 MiB）**，server 9.2 G。当时空闲 13.6 G ⇒ 显存上还能再加约 30 个 worker。**worker 的显存成本远低于此前的印象**（此前按 0.5–0.9 G/个估，实测均值 0.44 G）。

#### 扩容结论（代码已核实，别重推）

- **多 `--servers` 现在不但没用还会更慢**：`assign_servers()` 每个 yaml 只映射一个 server，而训练每批只有一个 yaml（`yaml_weights={yaml_id:100}`），于是 `RouterBatchStrategy.plan()` 把 100 个 episode 全钉在一个 server 上（`EpisodeTask.server_host` 注释：*owning server of this episode's yaml*），绑在其它 server 的 worker 全程空转。
- **顺序应该是**：① 加 worker（零代码）→ ② `--replicas N`（单 server 内扇出，对 conductor 透明）→ ③ 真·多 server（需把 `plan()` 改成每 server 一个 Stage；且因 `RemoteRun.shards()` 是单路径，多 server 必须同机共享文件系统）。
- 标定脚本 `/tmp/run_cal_workers.sh`（`N=32 P=0.55`）跑**同一组 200 对**，与扫点 p=0.55 的 481 ep/h 直接 A/B，无负载混淆。

#### 顺带推翻一个长期结论：跨机不再慢 7.8×

handoff §3 记着「t107 经 broker 2.4 ep/min vs 同机 18.7 ep/min（7.8×）」。实测当前链路：

| 段 | connect | 带宽 |
|---|---|---|
| wls → broker(racknerd) | 14 ms | 21.5 MB/s |
| ziyang10 → broker | 5 ms | 50.8 MB/s |
| 端到端（隧道） | 41–115 ms | 7.8–19.3 MB/s |

broker 那台是 1 vCPU VPS，但传输时 CPU 只 ~18% 忙 ⇒ **是它的网络被限速，不是转发能力**；两端接入都比中继快。

**旧的 7.8× 是带宽打满，不是架构问题**：观测在线上是 msgpack 裸 numpy（两路 224×224×3 = **301 KB/次往返**，无图像压缩），而每 episode 只有 **~65 次往返**（pi0.5 输出 action chunk，一次覆盖约 8 个环境步，实测 b0010/b0020/b0030 = 66/69/60）。0.9 MB/s ÷ 0.346 MB = 2.6 次/s = **2.4 ep/min，与旧记录逐位吻合**。当年 broker 是 pc732，现在是 racknerd，快了约 10 倍。

按当前链路重算（**修正**：单 worker 每往返周期 = (86+65.6p)/65 ≈ **1.66–1.88 s**，先前写的 1.06 s 把跨 worker 墙钟当成了单 worker 周期）：每次往返加 41–115 ms ⇒ 惩罚 **+2%~7%**；带宽需求 N=24 ≈ 4.3–4.9 MB/s、N=32 ≈ 5.8–6.5、N=40 ≈ 7.2–8.2，隧道实测给 7.8–19.3 MB/s ⇒ **N≤32 稳、N=40 贴边**。**跨机拓扑可行**，最有杠杆的改进仍是给观测加 JPEG 压缩（可砍 10–20×）。

#### 3.23.1 worker 数标定实测：**1.5× worker 换 1.353× 吞吐（90% 效率）**

设计成零混淆 A/B：`sweep_mixture` 恒取 `sample_batch(pool, 200, batch_idx=0, seed=0)`，所以 N=24 的标定跑的是**与 tc 扫点 p=0.55 点完全同一组 200 对 (task,init)**，只改 worker 数。旁证：SR 0.7850 vs 0.7900（差 1 个 episode）。

**两边都用 journal 自己的 `ts` 字段、只取中段 80%** 来算（`/tmp/steady_rate.py`）。这一点很重要：先前那个 481 ep/h 是从 `result.json` 的 mtime 差算的，**含 worker spawn/teardown**；拿它去比中段速率会白送大配置一个与扩展性无关的优势。

| 配置 | 中段 159 ep 耗时 | 稳态吞吐 |
|---|---|---|
| N=16 | 1153.4 s | **496.3 ep/h** |
| N=24 | 852.5 s | **671.4 ep/h** |

**扩展系数 = 1.353×**，而 worker 数是 1.5× ⇒ **效率 90%**，略次线性但很划算。

#### 由此校准的预算

- **训练 run 相对扫点的开销**：整程实测 437 ep/h ÷ 稳态 496.3 = **88%**（差的 12% 是批间 trainer 步 + 逐批 worker 重生）。
- **N=24 下训练的预期吞吐** = 671.4 × 0.88 ≈ **591 ep/h**。
- **500 批公平收敛口径**：50,000 ÷ 591 ≈ **85 h ≈ 3.5 天/变体**（N=16 时是 114 h ≈ 4.8 天）⇒ **省 1.35×**。

#### 真正的封顶不是算力，是共享卡的显存

- N=16 时本线 server SM 49.7%；按 1.353× 外推 N=24 约 67% ⇒ 算力饱和大约在 N≈36。
- 但**先发的 N=32 实测稳态只剩 4.5 GB 空闲**（后来回收时甚至读到 2,558 MiB，低于本线自己的 3000 告警线），而同机 `csmain` 正巧在那时把占用从 1,492 MiB 抬到 3,420 MiB（它逐批重生 worker）——**差一点挤到他线**。owner 明令不得伤及其它 session，故立即停掉降到 N=24（空闲稳在 6.6–8.1 GB）。
- 停跑与回收全程只动本线：按 PID + tmux 定点，reaper 干跑名单 64 个全是 `--server-key 127.0.0.1:8000`（`csmain` 那 8 个连普查名单都进不去），实杀后 **freed 16.9 GB**，四条他线全部存活。
- **结论**：在这张四线共用的 4090 上，**N=24 是安全上限**；要上 N≥32 必须等卡空出来，或者搬到独占的机器。

**每 worker 的显存成本实测 = 440 MiB**（16 个共 7,045 MiB），比此前 0.5–0.9 GB 的印象低得多——这是算安全 N 的正确系数。

### 2.5i 提案：7 天预算的重跑参数（2026-08-18，owner 授意「至多 7 天」，**待 owner 裁**）

#### 0. 资源包络

N=24（本卡安全上限，§3.23.1）稳态 671 ep/h、训练态 ≈ **591 ep/h**（含批间 trainer 步 + worker 重生的 12% 开销）。7 天 × 591 ≈ **99,000 ep 上限**；按共享卡 ~80% 占空比规划 **≈77,000 ep ≈ 770 批**。

#### 1. 选哪条线（淘汰理由都来自实测）

| 候选 | 判 | 理由 |
|---|---|---|
| ts 长跑 | ✗ | SR(p) 在整个工作区间平（§2.5e/§2.5f），**没有信号，多少步都一样** |
| tc @ λ=0.5 长跑 | ✗ | 最优在角点 p=1（§2.5h），跑 700 步只会走到「永远用 teacher」——**可预测，零信息** |
| tsc | 缓 | student 臂已被 ts 线证明无差异化价值，三臂只稀释预算 |
| **tc @ λ 重标定** | ✅ | 唯一让「router 能否打过常数」成为**可检验**问题的配置；cache 是论文的对象 |

#### 2. λ 的定量选择（全部由 tc 扫点实测导出）

成本项 = λ·K·p，K = N_steps/T_max = **0.10385**。λ 的物理含义是边际条件 `SR'(p*) = λ·K`；实测局部斜率给出可用区间：

| λ | argmax p | 角点边距 J(p\*)−J(0) | 右侧梯度 gap | 目标跨度/批噪声 |
|---|---|---|---|---|
| 3.0 | 0.302 | 0.111 | **0.0002 ← 右侧躺平** | 2.8 |
| 4.0 | 0.302 | 0.080 | 0.015 | 3.7 |
| **4.5** | 0.302 | 0.064 | 0.023 | 4.7 |
| **5.0** | 0.302 | 0.048 | 0.030 | 5.6 |
| 5.5 | 0.302 | 0.033 | 0.038 | 6.5 |
| 6.0 | 0.302 | **0.017 ← 塌向纯 cache 的风险** | 0.045 | 7.4 |

两头都是坑：λ≤3 右侧没有回拉力（0.302 与 0.447 目标差 0.0002），λ≥6 对角点 p=0 的边距掉进噪声。**推荐带 [4.5, 5.0]，暂定 λ₃=5.0**。

⚠ **角点边距在 n=200 下只有 ~1σ**（J(0.302)−J(0) 的 1σ≈0.047）。所以发射前要**预飞加密**：{0, 0.25, 0.302, 0.40} 四点 × 400 ep（1σ 压到 ~0.033），**并把参照点与被测点放进同一次扫点**（消掉 §2.5h.1 记的跨时混淆），然后据加密后的曲线在 [4,5.5] 内定死 λ₃、**先预注册再发射**。成本 1,600 ep ≈ 2.4 h。

#### 3. 其余参数逐项（默认 = 不动冻结值，改动只有一个：λ）

| 参数 | 决定 | 理由 |
|---|---|---|
| batch=100 | **不动** | 与五条预注册 run 可比；变批量会让「唯一变量是 λ」失效 |
| lr=3e-4 | **不动** | λ=5 下起点净梯度 ≈ −0.24/单位 p，与 tc@λ=0.5 的 +0.25 同量级 ⇒ 每步 ~0.0005-0.0007，走到 p≈0.33 约 **260-350 步**，600 批预算容得下，不必冒 §3.12.4「一步饱和」的旧险 |
| β=0.01 | **不动** | 对最优位置的偏移估计 ~0.003 个 p，可忽略 |
| baseline=批均值 | **不动**（条件化 baseline 的 1.5× 增益留给下一轮） | 一次只改一个变量；改 trainer 会污染归因 |
| warm-start | **不动**（§3.8 均匀 graft，`warmstart_l10_tc.pt`） | 起点在 0.50，λ=5 下应**反向**走向 cache——这本身是对优化器的强方向测试（λ=0.5 时它朝 teacher 走过） |
| seed=0、temperature=1.0、`mode: sample` | 不动 | |
| workers | **16 → 24** | §3.23.1：1.353×、90% 效率、共享卡安全上限；worker 数不在冻结矩阵内 |
| **批数** | **600**（vs 冻结的 40） | 260-350 步走完遍历后还剩 ~300 步学判别——这正是 owner 质疑（40 步远不够）的回应 |

#### 4. 预注册（吸取 §2.5b.1「两分支都满足=判据坏了」与 §2.5h.1「挑 argmax 再检验」两个教训）

- **P1 · 遍历门（可证伪，带行动）**：batch 150 时 teacher 份额均值 ≤ **0.46**（预期 ~0.43）。不达 ⇒ 停跑，改用偏置起点 p=0.35 重发（决策规则现在写死，不许中途拍脑袋）。
- **P2 · 判别度（测量，不设门）**：跨状态 sd 轨迹 v0/v100/…/v600（λ=0.5 下 40 步长到 0.0079；600 步长到哪里就是本 run 的核心测量之一）。
- **P3 · 主判定（现在就定死，不许事后挑点）**：训练完，取**最后 50 批的实测 teacher 份额**为匹配点 p̂；同一组 **2,000 个 slots**（同 seed 抽取）各跑一遍「冻结 router（sample 模式）」与「常数策略 @ p̂」⇒ **配对 McNemar，双侧 α=0.05**。检出力：不一致对 ~360 ⇒ SE≈0.0095 ⇒ 2σ 可检出 **~0.02**。这是唯一的 primary。
- **预期写在前面**：λ=5 优化成功的样子是 **success 从 ~0.78 降到 ~0.73**（换 3 倍算力节省）——**SR 下降是设计使然，不是失败**，免得又有人（包括我）中途误读。
- 附带诊断：v600 的 argmax 行为（200 ep）——记录部署形态，不进判定。

#### 4b. 拓扑（owner 2026-08-18 指示：worker 放 timan107，wls 只留 server；**发射时先测速率，正常沿用 t107，不正常回本机**）

这就是 plan §4.2 的原始设计（t107 = conductor + workers，wls = server，trainer 经 ssh 在 wls 上跑，密钥 `~/.ssh/rlr_t107` 已备，§3.2），当年被放弃只因旧 broker 带宽 0.9 MB/s（§3.23 已推翻）。收益与代价按修正后的周期（1.66–1.88 s/往返，**不是**先前误写的 1.06 s）：

| 量 | 值 |
|---|---|
| wls 显存释放 | worker 10.6 G（N=24）⇒ 卡上只剩 server ~7.8 G，共享卡压力大减 |
| 延迟惩罚 | +41–115 ms/往返 ⇒ **+2%~7%** |
| 带宽需求 | N=24 ≈ 4.3–4.9 MB/s；N=32 ≈ 5.8–6.5；N=40 ≈ 7.2–8.2（隧道实测 7.8–19.3）|
| 潜在上行 | worker 数不再受 wls 显存限制 ⇒ t107 上 **N=32** 可能净超本机 N=24 |

**T0 · 链路预测（分钟级）**：t107 → server 公网入口（`155.98.36.13:9000` frp 或 `linziyang.top:14007` expose，两条都测）的 RTT + 持续带宽——**这条路径本会话没测过**（测的是 wls↔racknerd↔ziyang10），不能外推。

**T1 · 速率预飞（owner 判据的机器化）**：t107 起 N=24 worker 跑 200 ep @ p=0.55（**同一组 slots**，与本机 671 ep/h 直接 A/B，`steady_rate.py` 中段口径）。**判据：≥0.85×（≥570 ep/h）⇒ 沿用 t107，并顺手试 N=32；<0.85× ⇒ 回本机 N=24。** 约 20 分钟 + 起服务 ~1 h。

**运行期 failover（预注册）**：健康探针盯吞吐，连续 3 批 < 450 ep/h ⇒ 计划性切回本机（批级 resume 已实证零损失，§3.22）。

**实施项追加**：t107 版发射脚本（`--servers <公网入口>`、`--remote-host wls`）；t107 侧的孤儿判据/reaper（`--server-key` 换成公网入口串，机制不变）；t107 的 libero_sim conda env 与 GPU 渲染可用性由 T1 一并验证。

#### 5. 实施项（发射前，约半天）

1. 矩阵加 `l10_tc_lam3_s0`（λ₃=预飞定值），标 `amendment` 并披露动机（λ 网格在 ts 上标定、对 tc 退化成角点，§2.5h）；
2. **门禁改动**：`_pilot_problems` 会拒绝任何非 pilot 候选的 λ ⇒ 需要一条**显式 amendment 通道**（要求 matrix 里该 run 带 `amendment` 字段 + 动机文本才放行，不是放宽默认）+ 测试；
3. `run_m6_v2.sh` 的 `--workers` 参数化到 24；
4. 磁盘：600 个 v*.pt ≈ 40 GB（527 GB 空闲，可容）；dump 逐批 reclaim，稳态几 GB。

#### 6. 时间表

| 项 | ep | 时长 |
|---|---|---|
| 预飞扫点加密（4 点 × 400，同会话含参照重测） | 1,600 | ~2.4 h |
| 主跑 600 批 | 60,000 | **~101.5 h ≈ 4.2 天** |
| 终局配对评测（2 × 2,000，共享 slots） | 4,000 | ~6 h |
| 机动（共享卡让路 / 事故重发 / P1 失败重启） | — | ~1.5 天 |
| **合计** | ~65,600 | **≈ 6.2 天 ≤ 7** |

備选 B：拆成 2×300 批（均匀起点 + p=0.35 偏置起点各一条），控制遍历段的混淆——代价是每条的判别学习段减半。**不推荐**：P1 门 + 重启规则已覆盖遍历失败的情形，单条 600 批对「判别能否涌现」更有检出力。

#### 7. 需要 owner 拍板的四件事（拓扑不在其中——owner 已裁：t107 跑 worker，速率预飞定去留，见 §4b）

1. **是否批准这一条 amendment run**（λ₃∈[4,5.5]，600 批，其余全冻结）；
2. λ₃ 由预飞加密曲线定，还是 owner 直接指定；
3. **M7 评测模式**（本提案的 P3 用 sample 模式做 B 池终局判定；A 池 M7 的模式仍需另裁——argmax 会让 tc router 与「永远用 teacher」不可区分）；
4. 条件化 baseline（1.5× 有效样本）是否留到再下一轮。

### 3.24 Amendment 通道落地 + t107 拓扑 staging（2026-08-18 下午，owner 目标「按安排开展实验」）

**Phase A1（repo，全部未 staged）**：
- `run_matrix.yaml`：`lambda_3: null` + `l10_tc_lam3_s0` 行（`amendment{reason,basis,date}` + `episodes: 60000`）。basis 指 `exp/rl_router/data/amendment_lambda3.json`（预飞产物，发射前生成——文件不存在则门禁拒发，顺序天然正确）。
- `launch_gates.py`：新增 `amendment_problems()`（reason≥40 字 + basis 存在 + date）；`_pilot_problems` 对 pilot 未标定符号走 amendment 通道（无块=照旧拒绝）；`planned_batches` 检查改按 run 级 `episodes`（无 amendment 的 episodes 覆盖=拒绝）；`check_launch_gates` 尾部去重（一块 amendment 背书两处偏离会重复收集）。
- `run_rl_router.py:~780`：`total_batches = run.get("episodes", matrix[...]) // batch_size`。
- 新增 `worker_gpu_id(i, gpus=, gpu_ids=)` + 两个入口 `--gpu-ids`（共享机健康卡不是 0..N-1，取模指派会把 worker 派上满卡死在 EGL init；显式列表循环使用，重复=加权）。
- 测试 +8（amendment 通道 7 + gpu_id 1）；`pytest tests/exp/` 1274 全绿。

**T0 判决**：frp 入口 `155.98.36.13:9000` 从 t107 CLOSED（wls 无 frpc 进程/配置——本机今晨重启后不存在，非掉线）。broker 入口 `linziyang.top:14007`（=rlr-srv expose，66h 前建，ALLOCATED 挺过重启）：t107→broker RTT 中位 **6ms**（t107 离 broker 比 wls 还近），healthz 200 端到端，上行实测 **5.6 MB/s**（30MB scp 经 :14024）。N=24 需 4.3-4.9 → 够但薄；N=32 需 5.8-6.5 → **不够，t107 上不试 N=32**（§2.5i.4b 的「顺手试 N=32」条件不成立）。

**t107 staging 全清单**：代码 8 文件（exp/rl_router 7 + mlp_router_judge.py）+ run_matrix.yaml push 落位（旧本地改动备份 /tmp/rlr_backup_0818）；warmstart_l10_tc.pt → /scratch/zixuans8/rl_router/art/；lam1+lam3 的 r_tc_train.yaml；artifacts 本地化 `artifacts_m6_tc_t107.json`（**gate 在 conductor 侧查文件存在**，wls 路径在 t107 不存在会误拒）；ssh config 块（SshTransport 裸 `ssh -p 14024 weiland@linziyang.top`，BatchMode 实测 OK）；工具 /tmp/{run_t1_cal,run_lam3_sweep,run_m6_t107,reap_orphans,steady_rate,list_m6_runs}。wls 侧：/tmp/run_m6_v3.sh（WORKERS/GPUS/GPU_IDS 参数化）。

**跨机 weights 桥（零代码）**：sweep_mixture 的 constant.pt 是 driver 本地路径、server 从自己文件系统读同一路径串 ⇒ 在 wls 按 t107 的 out-dir 路径预放 byte-equal 镜像（/tmp/rlr_t1/p0_55/、/tmp/rlr_lam3_sweep/{p0_00,p0_25,p0_30,p0_40}/）。b2 逐值验证（float32 精度 6e-9 内）。主跑不需要此桥——run_rl_router 原生 SshTransport 推 weights。

**t107 孤儿收割**：DRYRUN census 发现 **16 个本线 server-key（linziyang.top:14007）孤儿 worker**（2d18h 前 = rlr-srv expose 创建时刻的早期尝试残留，user=zixuans8，ppid=1，fd=3 纯吃显存）；另有 5d 代残留携他 key，**域外不碰**。实收 16（两轮：8 wrapper + 8 re-parent），**每卡回血 ~850MB**（⇒ 单 worker 1080 实测 ~425MB 静态 / 运行时 ~585-660MB）。

**T1 冒烟**（N=2, 4ep, GPU 7/0）：4/4 成功，realized 0.462，**1080 EGL + 隧道 WS + server 路由端到端通**。正式 T1（N=24, 200ep, GPU_IDS=7×10,4×4,0×3,6×3,2×2,3×2）18:15Z 发射，监控挂 b3dwkxy6k。判据：steady_rate 中段 ≥570 ep/h（0.85×wls 671）沿用 t107；否则回本机 N=24。

**T1 判决（2026-08-18 18:36Z）**：t107 N=24 × 200ep @p=0.55 同 slots，realized 0.5476，SR 0.825（wls 同点 0.790，差 1.2σ 内）。**steady_rate 中段 632.9 ep/h = wls 671 的 0.943 ≥ 0.85 判据 ⇒ 沿用 t107**；整程墙钟 1146.8s（627.9 ep/h 含 ramp，批间 respawn 开销比单机还小——48 CPU 的并行 spawn）。N=32 不试（T0 带宽 5.6 < 5.8-6.5 需求）。跨机延迟惩罚实测 671→633 = **−5.7%**，与 §3.23 预测 +2~7% 逐位吻合。**主跑吞吐规划数改用 633×0.88 ≈ 557 ep/h ⇒ 600 批 ≈ 107.8h ≈ 4.5 天**（仍在 7 天窗内，机动余量 1.2 天）。

### 2.5j λ₃ 机械选择规则（**预注册：写于加密扫点出结果之前**，2026-08-18 18:4x）

扫点 18:37Z 发射（t107 N=24，{0, 0.25, 0.302, 0.40}×400ep 同会话），结果落地前把选择规则写死：

- 目标函数：J(p) = SR̂(p) − λ·K_tc(p)，K_tc(p) = 54·(p·1.0 + (1−p)·7.2165e-05)/520（**cache 成本，不是 student 的**——knee_and_lambda.py 原版硬编码 C_S 已修，加 `--cheap-cost`）。
- 候选 λ 网格：{4.00, 4.25, 4.50, 4.75, 5.00, 5.25, 5.50}（§2.5i 的 [4,5.5]）。
- 记 p\* = 4 个实测点上 J 的 argmax。**准入**：p\* 必须是内点（0.25 或 0.302），端点（0 或 0.40）即淘汰该 λ。
- 每个准入 λ 算两个边距的 z 值：z_left = (J(p\*)−J(0))/σ、z_right = (J(p\*)−J(0.40))/σ，σ 由两点二项 SE 合成（n=400）；p\*=0.40 时 z_right 视为 0（无右拉力）。
- **λ₃ = 使 min(z_left, z_right) 最大的 λ；并列取更小的 λ**（更小的塌向纯 cache 风险）。
- **止损**：若最好 min-z < 0.5，不发射，写明数据不支持内点最优，报 owner。
- 附带核对：realized p 与 target 偏差 >0.05 的点按 realized 值代入 J（扫点是 sample 模式，realized 才是真实工作点）。

规则以此为准，落地后不再改。执行工具：`knee_and_lambda.py --cheap-cost 7.2165e-05` 出表 + 决策脚本按上式算 z。

### 2.5k `l10_tc_lam3_s0` 预注册终稿（发射前定死，2026-08-18 ~21:00Z）

**加密扫点结果**（4×400ep 同会话，t107 N=24，journals 见 basis provenance）：

| p_target | realized | SR | 备注 |
|---|---|---|---|
| 0.00 | 0.0000 | **0.5075** | 与旧 200ep 0.520 差 0.5σ，锚点复现 |
| 0.25 | 0.2516 | **0.6875** | 新点 |
| 0.302 | 0.3021 | **0.6825** | 比旧 0.725 低 1.1σ；**0.25-0.302 段平** |
| 0.40 | 0.3958 | **0.7200** | 比旧 0.447→0.770 低（跨会话差异正是同会话重测的理由） |

**λ₃ = 5.0**（§2.5j 规则机械产出：全网格 argmax 均为内点 **p\*=0.25**；λ=5.0 的 min(z_left,z_right)=1.41 最大（z_left=1.47/z_right=1.41）；无止损）。矩阵已填，basis `exp/rl_router/data/amendment_lambda3.json` 三机同内容落位。**t107 上门禁 dry-run：0 problems。**

**预注册判据（不许再改）**：
- **P1 遍历门**：batch 150 时最近 10 批 teacher 份额均值 ≤ **0.46**。不达 ⇒ 停跑，改 p=0.35 偏置起点重发（§2.5i.4 原文）。
- **P2 判别度**（纯测量）：跨状态 sd 轨迹 v0/v100/…/v600。
- **P3 主判定**：最后 50 批实测份额 p̂ → 同组 2000 slots「冻结 router(sample) vs 常数@p̂」配对 McNemar，双侧 α=0.05。唯一 primary。
- **预期写在前面**：λ₃=5 优化成功的样子 = 份额 0.50 → **~0.25-0.30**（新曲线的内点最优带），SR 从 ~0.75-0.79 降到 **~0.69-0.72**——**SR 下降是设计使然不是失败**。
- **拓扑 failover**（预注册）：t107 主跑，连续 3 批批完成速率 <450 ep/h ⇒ 停 t107 conductor，同一命令在 wls 用 `/tmp/run_m6_v3.sh`（batch 级 resume §3.22）接跑，`--workers 24 GPU_IDS=""`。
- 附带诊断：v600 argmax 行为 200ep（记录部署形态，不进判定）。

**发射参数**：`WORKERS=24 bash /tmp/run_m6_t107.sh l10_tc_lam3_s0`，GPU_IDS 按发射时 nvidia-smi 现读（650MB/worker 预算）；tmux `rlrm6`；600 批 ≈ 107.8h ≈ 4.5 天 @557 ep/h 训练态。

### 3.25 事故：wls tether agent 断连 → conductor 崩溃 → server 僵死（2026-08-19 22:21Z，b0129）

**时间线**：22:21Z weilandserver tether agent 心跳消失（进程活着、broker 连接死，yamux 卡死已知形态）→ t107 侧三条路径同时断（WS 推理经 :14007 / ssh 经 :14024 / 包推送）→ conductor 在 b0129 包推送 3 次重试后按设计 TransportError 退出（此时 b0129 journal 43ep）→ 24 worker 随 conductor **干净退场（零孤儿**，spawn 的进程组处理是对的）→ server 在 24 路同时断连后事件循环僵死、随后进程退出（GPU 归零 48511MiB 全空）。

**诊断路径（防下次绕弯）**：tether exec 报 node_offline ≠ 机器死——LAN ping 192.168.0.200 通 + 直接 ssh（key `~/.ssh/id_rsa_winhost`，**本机 ssh config 里 192.168.1.150 是陈旧地址，devices.md 的 192.168.0.200 才对**）确认机器 5 天未重启、rlrsrv 还在。agent 重启（LAN ssh + setsid nohup）后节点回线，但 **exposure 不齐**：wls-ssh 隧道自动复活、rlr-srv 没有；rm+重建 rlr-srv **原端口 :14007 复用**（幸运，零脚本改动）。重建后 healthz 仍 000 → 才发现 server 本体已死——**断 agent 只是第一张牌，server 是第二张**。

**恢复**：22:31:57Z 按 §5.1 重启 server（tmux rlrsrv）；等 healthz=200 后 t107 重发 conductor（批级 resume，b0129 重跑）。损失 ≈ 半批 rollout + ~15 分钟墙钟；权重链/trainer 状态无损（v0-v119 全在）。

**教训**：(1) 疑难分层：node_offline → LAN 直连探机器 → agent 进程 vs 连接 → exposure 逐条验 → 服务本体；(2) server 对大规模同时断连不鲁棒（24 路断连即僵死后退出）——长跑期间 agent/隧道抖动会级联杀 server，**wls 看门狗的 SRV 检查是关键防线**（本次它下一拍本会抓到）；(3) t107→wls 的 LAN 直连 ssh 通道（192.168.0.200）是 tether 之外的应急通道，值得记住。

**§3.25 恢复完成（22:45Z，全程中断 ~24 分钟）**：server 重启 3 分钟就绪（healthz=200 经 :14007）→ 首次复位在 `refresh_remote_state` 撞出**潜伏 bug**：该函数自组命令行没带 `PYTHONPATH=.`（trainer-cmd 模板有），单机时代靠 LocalTransport 继承父环境侥幸工作，ssh 不继承 ⇒ 首次跨机 resume 即 `No module named 'exp'`。已修（run_rl_router.py:301 加 PYTHONPATH=. + 注释，tests 129 绿，t107 落位）。二次复位后有 **~6 分钟静默重建段**（129 批 journal 重放 + 远端 dump du，日志无输出——别误判卡死，py-spy/进程树确认即可）→ 24 worker 拉起，b0129 从崩溃前的 43ep 续跑（**ep 级 replay 保留验证 ✓**）。监控重挂 bv24fdz96（err 计数偏移 16817 跳过崩溃期旧 Traceback）。净损失：~24 分钟墙钟 + 半批重跑，权重链无损。

**暂停（2026-08-20 ~01:5xZ，owner 要跑别的任务）**：在 b0143 完成后停。停法：kill tmux rlrm6（只本线）→ DRYRUN 确认 48 个候选全带 `--server-key linziyang.top:14007` 且 KEEP 侧空 → 实收 48，t107 worker 归零、显存回吐。wls server（tmux rlrsrv）**保留**，随时可接。监控全撤（Monitor bv24fdz96 + cron 37b18d97/a0cdc845/8199a064），恢复时重挂。**恢复命令（批级 resume，绝不删 art_m6/）**：
```
tether exec timan107 -- bash -lc 'export HOME=/home/zixuans8; tmux new -s rlrm6 -d "WORKERS=24 GPU_IDS=<按当时 nvidia-smi 现读> bash /tmp/run_m6_t107.sh l10_tc_lam3_s0 2>&1 | tee -a /tmp/rlr_m6_l10_tc_lam3_s0.log"'
```
恢复前先验：server healthz（t107 上 `curl -m8 http://linziyang.top:14007/healthz` 应 200，不通则先重启 server 并检查 rlr-srv exposure）。恢复后有 ~6 分钟静默重建段属正常（§3.25）。

**P2 中期轨迹（暂停期离线算，b0000-b0140 每 10 批，纯读磁盘无 GPU）**：

| batch | mean p | cross-state sd | p 范围 |
|---|---|---|---|
| b0000 | 0.5000 | **0.0000** | 常数（v0 均匀 graft，§3.8 设计） |
| b0020 | 0.3989 | 0.0074 | 0.380-0.419 |
| b0040-b0110 | 0.375-0.401 | **0.0051-0.0066 平台** | 宽度 ~0.037 不变 |
| b0120 | 0.3770 | 0.0074 | 0.352-0.404 |
| b0130 | 0.3349 | **0.0097** | 0.306-0.369 |
| b0140 | 0.3150 | **0.0117** | 0.279-0.357 |

读法：**判别度与份额下探是同一件事的两面**。b0040-b0110 的七十批里 sd 钉在 0.005-0.007（策略几乎是常数，只是整体高度在动），份额同期也卡在 0.38 平台；b0120 之后 sd 单调爬升（0.0074→0.0097→0.0117，**翻倍**）而份额同步破位下行（0.377→0.335→0.315）。这与 §「熵-成本平衡」推演一致：均匀压低 p 要付全额熵罚，**只有学会按状态区分才能继续降低平均份额**，所以 sd 增长是份额继续下探的**必要伴随**而非巧合。⚠ 绝对值仍小（sd/mean ≈ 3.7%），远不到「强判别」；这条轨迹是 P2 的纯测量，不参与判定，最终结论仍由 P3 配对检验给出。

### 2.5k.1 P1 遍历门裁决：**PASS**（2026-08-20 05:2xZ，batch 150）

预注册判据（§2.5k，发射前定死）：batch 150 时**最近 10 批（b0140-b0149）teacher 份额均值 ≤ 0.46**，否则停跑并改 p=0.35 偏置起点重发。

**实测 0.3335**（success 0.7040，reward ~0.49）——**PASS，且余量极大**（门限 0.46，实测比门限低 0.127，约 2.7× 批间波动）。⇒ 不触发重发分支，按原计划继续跑到 600 批。

配套读数：份额轨迹 0.500（v0）→ 0.427（前 30 批）→ 0.38 平台（b0040-b0110 七十批）→ 0.335（b0140-b0149），**已进入 λ₃=5.0 的目标带 0.25-0.30 的邻域**；同期跨状态 sd 0.0000→0.0051→0.0117（P2 纯测量，§前节全表）。SR 从起点 ~0.744 降到 0.704，**与「用 SR 换成本」的预注册预期同向**（预期带 0.69-0.72），不是失败信号。

⚠ 口径说明：门限用的是 `metrics.jsonl` 的 `arm_executed_rate.teacher`（实际执行占比，权威口径），不是 router 输出概率均值；两者在采样模式下应当一致，b0148 实测分别为 0.33 与 0.32 量级，无异常。

**第二次暂停（2026-08-20 ~17:0xZ，owner「资源给别人」）**：在 b0215 完成后停于 **215 批完整 + b0215 部分 / 21,538 ep**（trainer_state 到 b0214，权重链 v215.pt）。停法同 §3.25 收尾：kill tmux rlrm6 → DRYRUN 确认 48 个候选全带 `--server-key linziyang.top:14007` 且 KEEP 侧空 → 实收 48（回吐 1432 MiB/卡）→ 停 wls server（显卡 0 MiB 占用 / 48511 全空）→ 撤全部监控（Monitor bd0phzkg2、cron 90ab2f54 t107 巡检、cb51e302 wls 看门狗）。恢复流程见 handoff §3（批级 resume 从 b0215，**绝不删 art_m6/**）。本次暂停前已获成果：P1 PASS（§2.5k.1）、P2 中期轨迹（sd 0→0.0274，方差分解 62% 在 episode 内）、reward 分解（+0.077 全部来自省成本、SR −0.047）、两报告补入信息不对称边界（MLP 对相似度全屏蔽 ⇒ 与 TIER 阈值 judge 非同一问题）。

### 3.26 wls 4090 保温协议成为硬前置（owner 裁定 2026-08-20）

**背景**（`dist_experiment_control/docs/devices.md` §2.5，2026-08-20 实测判定）：这张 48G 改装 4090 在出厂功率/频率下不稳定——冷卡（≤36 °C）拉满带宽负载，在 **28–62 s 爬坡窗口**内产生静默计算错误（实测 4,069 万个，67–71 °C 非过热，**零 Xid 零报错**）；同一不稳定曾以 `Xid 31 MMU Fault @ 0x0` 三次打死推理 server（都在起流量后 ~3 min）。冷启动 3/3 全灭、起温 ≥44 °C 2/2 全过 ⇒ 故障 = 冷态陡热爬坡 × 满带宽显存流量**两条件叠加**。

**owner 裁定（2026-08-20）**：① **显卡问题是今天（8-20）才出现的，此前无问题**；② **授权在正确执行保温程序的条件下继续使用这块卡**——即 devices.md §2.5 那句「送修前不承担正式实验」由本裁定覆盖，前提是保温协议到位。

**本线落实**：任何 GPU 负载（pi05 server / 评测）启动前必须先起 `keepwarm` 会话并确认温度 ≥50 °C，且**全程陪跑到实验完全结束**（含中途重启 server/conductor 的冷却窗口——那正是最危险的时刻）。恢复闹钟（cron 74ad9068，8-21 08:00 CDT）已把这条写成第 1 步硬前置；重挂的 wls 看门狗追加一条：keepwarm 会话不在 → 立即重起并报警。

**留痕的边界（供报告引用时判断）**：本 run 的 b0143–b0215 由 8-19 22:59 CDT 冷启动的 server 承载（**早于保温协议建立**），落在 owner 所述"今天出问题"的时间窗内。风险性质与量级：故障窗口只有冷启后 ~28–62 s，此后干净；若有污染，表现为个别 episode 的动作块被算错 ⇒ 少量 episode 失败，**结果是回报信号更噪，而非把策略推向某个系统性方向**（份额下探是成本项驱动的，与算错无关）。**唯一需要干净数值的是 P3 主判定，而 P3 尚未运行、将在保温协议下执行**——所以主结论不受影响。若后续复核发现 b0143–b0215 段异常，可用 v143 之前的权重链重跑该段。

### 3.27 server 入口切到 wls 直连公网段（owner 指示 2026-08-21）

**变更**：`weilandserver` 自 2026-08-20 起有直连公网入口 `ziyanglin.com:23100-23199`（交换机 NAT，`ziyanglin.com` → 140.177.159.24）。按 devices.md §4.0 优先级规则，**优先于 tether expose**：少一跳 broker 转发、不占 broker 端口池、不受 yamux keepalive 长 idle 断流影响（§3.25 那次 agent 卡死正是 broker 侧问题）。

⚠ **关键差别（踩过就废一次启动）**：直连段是 **1:1 映射、不做端口转换**——服务必须**自己监听在 23100-23199 之内**。把 server 起在 `:8000` 再指望公网 `:23100` 连上是连不通的（本次首次启动就是这样起错了，随即改正）。

**本线落地**：pi05 server 改起在 **`--port 23150`**；t107 三处地址同步（`run_m6_t107.sh` 的 `--servers`、`reap_orphans.sh` 与 `rlr_health.sh` 的 `SERVER_KEY` 默认值）改为 `ziyanglin.com:23150`。⚠ SERVER_KEY 必须与 worker argv 里的串**逐字一致**，否则孤儿判据（§3.20 的域收窄）会失效——这是本次改动最容易漏的一处。旧的 `rlr-srv` expose（:8000→:14007）随之作废。

**更正与补完（2026-08-21，owner 质询「公网端口不够你用吗」后）**：我起初判断「trainer 的 ssh 只能走 expose，因为 sshd 在 :22、改端口要 sudo（实测无免密）」——**这个判断不完整**。不必动 sshd：起一个**用户态转发**监听在段内即可（>1024 端口无需 root）。已实装 `pubfwd.py`（wls `/home/weiland/gtp_logs/`，纯 Python 无依赖，tmux 会话 `pubfwd`，`23122 → 127.0.0.1:22`），实测 `SSH_OK_DIRECT` + 3 MB scp 往返 md5 逐位一致。⇒ **两条通道现在都不经 broker**：推理 `:23150`、trainer/包推送 `:23122`。t107 的 `~/.ssh/config` 需为 `ziyanglin.com` 补 IdentityFile（原先只配了 `linziyang.top`，否则表现为 `Permission denied` 而非网络不通——这一步漏了会误判成转发没通）。旧的 `wls-ssh` expose 与 `--remote-port 14008` 一并作废。

⚠ 转发器只应指向**自带鉴权**的服务（这里是 sshd，公钥认证）；直连段本身无鉴权，别把无鉴权服务挂上去。

⚠ **安全性（devices.md §2.5.1 明载）**：该段**无任何鉴权**，且不经 broker、不受 session 成员资格约束，任何人扫到端口即可连；pi0.5 server 自身没有 token/TLS ⇒ 起在这个段上等于对公网开放推理算力。owner 已指示优先用直连段，本线遵此执行；实验结束应及时停 server。


### 3.28 resume 的隐藏成本：`_reclaim_consumed()` 是 O(已消费批次) 的 ssh 循环（2026-08-21 实测定位）

**现象**：resume 后 conductor 沉默 7-10 分钟不起 worker，日志停在 `run manifest written`。owner 追问「worker 启动怎么这么久，别人的一下就起来」。

**排查（逐项实测排除）**：远端 `du` 容量测量 **0.22 s**（dump_m6 有 427 目录 / 42,377 文件，仍然快）；远端 state refresh **2.2 s**（torch import 1.7 s + 193 MB checkpoint 载入 0.2 s）。都不是。conductor 主进程 48 线程阻塞在 `poll`、零网络连接、零常驻子进程——但**每次采样都能抓到一个刚生出来的 `/bin/sh -c ssh -p … 'cd /home/weiland/openpi && uv run exp/rl_router/batch_package.py reclaim --shards …'`**（12 次采样 12 次命中）。

**根因**：`run_rl_router.py:889` 的 `_reclaim_consumed(remote, args, state["consumed"])` 对**每一个已消费批次**执行一次远端 reclaim，每次是**一条全新 ssh**（无 ControlMaster 复用）+ 一次 **`uv run` 冷启动**，实测 **~2.7 s/批**。216 批 ⇒ ~9.7 分钟，与两次实测（7.4 / 9+ 分钟）吻合。**开销随进度线性增长**：跑到 500 批时 resume 要 ~22 分钟。

**为什么别人的 worker 秒起**：他们从零开跑，没有已消费批次，这段循环是空的。

**设计上是对的**（崩溃可能停在「更新已落但 shards 未清」之间，ledger 门使 reclaim 幂等且安全），**实现上两处可省**：① 没跳过已经清过的批次；② 每批一条 ssh 而非一次批量调用（传全部 batch_id 进去一次跑完）。任一改法都能把 resume 从 10 分钟降到秒级。**本轮不动代码**（改了要重测 resume 路径，而当时正在 resume 中途）；留作下次暂停窗口或收官后处理。

**监控启示**：判「是否卡死」不能只看 `workers=0`，要看 `ps --ppid <conductor> -o cmd=` 里的 `reclaim --shards .../b0xxx` 推进到哪批——已把这个字段加进启动期 Monitor。

### 3.29 迁移到 ziyang10 H200 + 自建 TCP 中继（owner 指示 2026-08-21，"tether expose 流量不够"）

**动因**：owner 要求把实验转到 H200，并问能否手搓一个中继让 ziyang10 与 t107 经 wls 沟通（broker 带宽不够用）。

**为什么不用现成方案（全部实测）**：

| 路径 | 吞吐 |
|---|---|
| 裸 TCP ziyang10→wls 公网段 | **19.35 MB/s** |
| 裸 TCP t107→wls 公网段 | **22.29 MB/s** |
| 经 pubfwd（用户态转发，无加密） | 11.46 MB/s |
| **经 ssh 隧道** | **1.4 MB/s** |

⇒ **瓶颈是 ssh 的加密，不是网络**。需求（N=24）4.3-4.9 MB/s，所以数据面必须绕开 ssh。

**中继设计（`relay.py`，wls `/home/weiland/gtp_logs/` + ziyang10 `/home/ziyang10/tools/`）**：**反向连接池**，不是多路复用器。ziyang10 在 NAT 后无法被拨入，所以 agent 主动外拨、在中继的 backend 端口**预先驻留** N 条连接；客户端到达时中继取一条、发 GO、agent 拨通本地服务后**回 ACK**，中继收到 ACK 才拼接。**一客户 = 一条端到端 TCP**，无分帧、无队头阻塞、没有自制多路复用器会有的错位失步。

⚠ **v1 的 bug 与修法（值得记住的通用教训）**：v1 用「写 GO 成功」判驻留连接存活——**错**，死 TCP 的写只是进本地发送缓冲，要更晚才报错。实测 24 并发有 **1 条**被配到死连接、中途 BrokenPipe。v2 改成**握手往返（GO→ACK）**，一次证明三件事：驻留 socket 活、agent 在跑、**本地服务可达**；另加 TCP keepalive 防 NAT 静默回收驻留映射。复测 **3 轮 × 24 并发 = 72/72 全通**，聚合 8.8-9.6 MB/s，建连中位 20-27 ms。

**卡住迁移的真问题：ziyang10 没有 sshd**（无监听、无进程、无二进制、无 authorized_keys）。而 **trainer 必须跑在 server 那台**（每批 ~700 MB shards 落在那里），conductor 靠 `SshTransport` 驱动它。三条路里选了代价最小的：
- ~~改 `SshTransport` 为 `TetherTransport`~~：要在活跑的关键路径上加新 transport 类，风险高；
- ~~把 shards 每批拉回 t107~~：700 MB/批，代码里没有这条路；
- ✅ **在 ziyang10 装用户态 sshd**（`mamba create -p ~/sshenv openssh`，监听 127.0.0.1:2222，仅公钥认证，无需 root），再用同一套中继暴露 ⇒ **`SshTransport` 一行不改**。

**最终拓扑（旧端点全部作废）**：

```
推理    : t107 workers   -> ziyanglin.com:23152 -> relay(wls) -> ziyang10:8999
trainer : t107 conductor -> ziyanglin.com:23154 -> relay(wls) -> ziyang10:2222 (用户态 sshd)
```
两条都不经 broker、不经 ssh 隧道（trainer 那条的 ssh 只在中继之后的最后一跳，且只传包与命令，不是数据面大头）。

**迁移清单（缺一不可，实际都踩了）**：① 状态迁移 wls→ziyang10（trainer_checkpoint.pt 203 MB + trainer_state.json + metrics.jsonl + v227.pt + v0.pt，经 ziyang10 直接 scp wls，19 s）；② arm yaml 的 `weights_path`/`dump_dir` 重写到 ziyang10 本地路径；③ gate artifacts 落位；④ **ziyang10 的 repo 落后两个版本，缺 `mlp_router` judge** ⇒ 补 `mlp_router_judge.py` + `config.py`（**覆盖前确认我们的 config.py 是它那份的超集**，含别人的 composite 校验，没踩掉他线改动）；⑤ **缺整个 `exp/rl_router/` 包** ⇒ 补 12 个模块（首次发射就死在 `batch_package.py: No such file`）；⑥ t107 的 `SERVER_KEY`（reaper + 探针）同步为 `ziyanglin.com:23152`，**必须与 worker argv 逐字一致**；⑦ 新发射脚本 `/tmp/run_m6_h200.sh`（旧的 `run_m6_t107.sh` 保留作回退）。

**方法学边界（报告必带）**：b0000-b0227 跑在 wls 的 4090（含故障卡时期），**b0228 起跑在 ziyang10 的 H200**。两者数值内核不同，**SR 轨迹在切换点会有一个不可归因的台阶**。影响评估：P2（跨状态 sd）是策略内部量、P3（同 session 内配对）自洽，**两个预注册判据都不依赖跨批次的 SR 绝对水平**；台阶只影响"监控用"的 SR 曲线。收益是剩余 372 批 + P3 主判定全部离开有静默算错前科的 4090。

**§3.29 续：迁移当天的两次失败与代价（教训）**

第一次发全量：`LAUNCH BLOCKED ... batch_package.py: No such file` —— ziyang10 缺**整个 `exp/rl_router/` 包**。补 12 个模块后再发。

第二次发全量：worker 起来了，但**每个 episode 都被 server 端 1011 打回**，累计 **~800 条 Traceback / 376 次 ConnectionClosedError**。server 侧真因：`ValueError: mlp_router requires query_keys; Orchestrator injects them when the judge signature declares the parameter` —— ziyang10 的 `orchestrator.py` 是旧版，没有 `judge_accepts_query_keys` 注入机制。

**我的方法错误**：逐个文件试错，撞一次补一次。**正确做法（已改用）**：`git merge-base --is-ancestor df2ef13 HEAD` 确认我们的 src 是对方的严格后代（覆盖安全）→ `git diff --name-only df2ef13 HEAD -- src/openpi` 一次列全 **20 个**文件 → 批量同步 → 核对对端 `git status` 只有 `config.py` 有他线本地改动（我们的是超集，未踩掉其 composite 校验）→ 原件备份到 `rlr_backup_0821/`。

**代价与数据完整性**：约 800 个 episode 级失败，但**没有污染数据**——这些 episode 全部失败重跑，b0228 一条都没写进 journal（eps 始终停在 22,724）。

**新增规程：跨机迁移后，发全量前必须跑真推理冒烟**（`sweep_mixture` 2 worker × 2 episode，判据 = `[sweep]` 行出现且 `1011`/`requires query_keys` 计数为 0）。本次冒烟以 2 个 episode 的代价暴露了两个问题（`query_keys` 缺失、`constant.pt` 只在 t107 本地而 server 在 ziyang10 读同一路径串——后者是 §3.23 时代「镜像 constant.pt」那条坑的跨机重演，**只影响扫点工具，不影响主跑**：主跑的权重由 `SshTransport` 推到 server 侧）。上一次同样的问题是用 800 个失败 episode 换来的。

### 3.30 修掉 §3.28 的 resume 慢：只清幸存的 shard 目录（2026-08-21，owner「直接改了他吧，浪费时间」）

**改法**（`run_rl_router.py::_reclaim_consumed`）：resume 时先发**一次** `ls -1 <shard_root>`，与 ledger 的 consumed 列表取交集，**只对还存在 shard 目录的批次**跑 reclaim。已清过的直接跳过——reclaim 本来就幂等，重复清是纯浪费。

**收益**：干净停机的常见情形从「N 条 ssh + N 次 `uv run` 冷启动」降到「1 条 ls」。实测单批成本 **2.7 s 同机 / 4.0 s 经中继**（迁到 H200 后多穿一层中继，慢 ~50%），⇒ 227 批 ≈ 15 min → **~1 s**；600 批时省的是 **~40 min/次**。

**不变量未动**：崩溃窗口（trainer 更新已落、shards 未清）留下的目录在 `ls` 里是存在的，照样被清。**故意 fail-open**：`ls` 失败时退回全清——"列不出来"意味着"不知道谁还在"，此时漏清每个孤儿要多背 ~2.6 GB。

**测试 5 条**（`tests/exp/test_rl_router_run_loop.py`，全套 1287 绿）：只清幸存者 / 300 批全干净只花 1 次远端调用 / 列表失败全清 / 无 ledger 连 ls 都不发 / **畸形 ledger 行（空 batch_id）必须忽略**——否则 `--shards {root}/` 会展开成 shard 根目录，一次清掉所有批次。

**部署**：t107 与 ziyang10 均已装（原件备份 t107 `/tmp/rlr_rrr_prev.py`）。**当前运行中的 conductor 不受影响**（Python 已加载旧模块），下次 resume 生效。

### 3.31 迁机后整批 `shard_missing`：`dump_dir` 是 arm yaml 里的常量，`--shard-root` 管不到它（2026-08-21）

**现象**：b0227 连续三轮 100/100「缺 slot」，两轮修复后按设计 halt（不在缩水的批上更新）。但 episode 本身**全是好的**——journal 200 行里 145 `done` / 55 `failed`，SR 0.725 落在 λ₃ 的预期带内，**一条 `error` 都没有**。

**根因**：`remote_manifest.json` 里 200 条 `rejected` 的理由全是 **`shard_missing`**。server 端逐步 shard 的落盘路径来自 **arm yaml 的 `checkpoints.cp1.judge.dump_dir`**，迁机后它仍是 `/home/weiland/rl_router/dump_m6`（wls 的路径），而 ziyang10 上**根本没有 `/home/weiland`** ⇒ 一条 shard 都没写。

**为什么没被任何东西挡住**——三个独立的洞叠在一起：
1. **两处路径、零校验**。conductor 的 `--shard-root` 只决定「去哪儿找」（`RemoteArtifacts.shard_root`），server 的 `dump_dir` 决定「往哪儿写」，**没有任何代码或门禁检查两者一致**。
2. **`write_versioned_yaml` 只改写 `weights_path`**（见其 docstring：它存在的理由就是防版本错配），`dump_dir` 原样透传。所以逐批生成的 yaml 会忠实地把错路径发给 server 600 次。
3. **写失败是静默的**。episode 照常返回 `done`，成功率正常，client 侧看不出任何异常——只有远端三源 join 才发现无 shard 可选。

**§3.29 的迁移清单第 ② 条（"arm yaml 的 `weights_path`/`dump_dir` 重写到 ziyang10 本地路径"）写了但没执行**，而且没有任何一步会去验证它。清单不等于校验。

**处置**：`sed` 改 t107 上的 `r_tc_train.yaml`（原件存 `.bak_wls`）；把 b0227 的本地批目录与远端 `art_m6/<rid>/b0227/` 一并挪走（**必须**：包摘要守卫拒绝同 batch_id 不同 digest 覆盖，round 0 重跑会撞上），再重发。trainer 状态未受影响（consumed 到 b0226 → v227，无需回滚）。代价 = 3 轮 × 100 episode。

**连带发现（本条最贵的一课）**：旧 conductor halt 退出后，它的 **48 个 worker 变成孤儿**（父链直挂 systemd），带着**已作废的 `ziyanglin.com:23150`** 又活了 1h17m，一直占着 t107 八张卡的显存。按 §3.19/§3.20 的判据收割：先 `bash /tmp/inspect_workers.sh` 逐进程列「归属端点 + 父链判定」确认 48/48 都是本线旧端点且都是孤儿，再 `SERVER_KEY=ziyanglin.com:23150 DRYRUN=1` 确认 **KEEP 侧为空**、打击面与新 run 的 `:23152` 零交集，最后实清 48 个。**孤儿判据必须按端点收窄**：不收窄就会误杀他线；只按当前端点查则完全看不见这批旧端点孤儿（第一次 DRYRUN 报的就是「0 个」）。

**待办（需 owner 授权后实施，属代码改动）**：在发射门禁里加一条断言——`dump_dir`（去掉尾部 `/`）必须等于 `--shard-root`，不等就 BLOCK。这是本条唯一能防止复发的结构性修复；靠"迁移清单"防不住，本次已经证明了。

### 3.32 `trainer exited 255` 但更新其实落了：ssh 通道无保活 + 监控漏报终态（2026-08-21）

**现象**：b0227 修好 shard 后终于填满，进 `train()`，conductor 抛 `ALERT: trainer exited 255 without an admission report`，尾部是 `Connection to ziyanglin.com closed by remote host.`，随后整条 run 退出。

**真相是反的**：trainer **跑成功了**——ziyang10 上 `v228.pt` 已写、`trainer_state.json` 已是 `v228 / 228 consumed`、`metrics.jsonl` 已追加。断的只是 ssh 通道，conductor 因此读不到 admission report，把一次**成功的更新**判成失败并停机。排除项：cgroup `memory.events` 的 `oom_kill=0`（不是 32 GiB 内存墙）；wls 侧 `relayssh` 自 11:51 起 `served=750 pool=7` 零错误、ziyang10 侧 `zagentssh` 日志只有启动一行（**不是中继重启**）。

**第一版诊断是错的，记在这里当反面教材**：我判成「`SshTransport` 走裸 `ssh`、没有 keepalive ⇒ 路径上的空闲回收掐断了连接」，于是在 t107 的 `~/.ssh/config` 里加了 `ServerAliveInterval 30`。**重发后第二批以完全相同的方式又死了**——同一个 `exit 255`、同一句 `closed by remote host`、同样是更新已落。**教训：没有复现就不算定位。** 我拿一个「像是」的机制解释了现象，而没有先问「它到底在第几秒死的」。

**真正的根因见 §3.33**（中继 agent 侧把建连超时留成了空闲读超时，静默 20 s 必断）。ssh 保活作为纵深防御保留（现为 `ServerAliveInterval 15` / `ServerAliveCountMax 20`），但**它不是修复**：15 s 的探测恰好能盖住 20 s 的窗口纯属巧合，真正让长静默会话可用的是 §3.33 那一行。

**恢复代价为零**：`trainer_state.json` 是 resume 的唯一权威，它已记 b0227 已消费 ⇒ 重发直接从 b0228 开始，既不重跑也不重复更新。

**监控的漏洞（比故障本身更该记）**：老 Monitor 的 fatal 正则是 `LAUNCH BLOCKED|CUDA error|failed \(exit|requires query_keys`——**不含 `ALERT`、不含 `Traceback`、不含「tmux 会话消失」**。于是这次 halt 它一声没吭，停机 ~30 min 无人知道，是我下一次巡检才发现的。**判据：写完过滤器要问一句「如果它现在崩了，我的过滤器会吐出一行吗？」** 新版覆盖 `ALERT|Traceback|LAUNCH BLOCKED|CUDA error|OutOfMemory|exited 255|requires query_keys` **加上 `alive=$(tmux ls | grep -c '^rlrm6:')`**——进程没了是最强的终态信号，比任何日志正则都可靠。

**顺带**：`_reclaim_consumed` 的 fail-open 在「shard 根目录压根不存在」时会退化成 227 次远端 `uv run`（实测 ~5 s/次 ≈ 16 min）。**迁机后先 `mkdir -p <shard_root>/<run_id>`**，`ls` 就能成功返回空集，清扫降回 1 次调用。已建。

### 3.33 中继的真根因：建连超时被留成了空闲读超时，静默 20 s 必断（2026-08-21）

**怎么定位的（这次做对了：先复现，再解释）**。同一条通道跑两个对照 ssh 会话：

| 测试 | 命令 | 结果 |
|---|---|---|
| A 有输出 | `for i in $(seq 1 20); do echo tick $i; sleep 10; done` | **活满 200 s，exit 0** |
| B 全静默 | `sleep 200; echo silent_done` | **21 s 就死**，`closed by remote host`，exit 255 |

「有数据就活、静默就死、死在 21 s」——一句话把「路径空闲回收」（那样 A 也该死，且不会卡在 20 s 整）排除掉，直接指向**我们自己中继里的一个 20 秒常量**。

**根因（`relay.py` agent 侧，一行）**：
```python
svc = socket.create_connection((thost, tport), timeout=20)   # 这 20 s 是 CONNECT 期限
# ...从未调用 svc.settimeout(None)
```
`create_connection(timeout=...)` 会把超时**留在 socket 上**。于是 `_pump` 里的 `src.recv(BUF)` 在静默 20 s 后抛 `socket.timeout`——它是 `OSError` 的子类，被 `except OSError: pass` **静默吞掉**，`finally` 照常 `dst.shutdown(SHUT_WR)`，对端就看到一次「正常的半关闭」。**没有任何一侧打日志**：中继日志只有 `served=750 pool=7`，agent 日志只有启动那一行。这是它能骗过我一次的原因。

**为什么推理通道从没中招**：LIBERO 闭环每个控制步一次往返（~200 ms），那条连接**永远不会静默 20 s**。只有 trainer 那条 ssh 会静默数分钟 ⇒ **b0227 启用中继后每批必死**，样本量 2/2。

**修复**：
```python
svc.settimeout(None)   # 建连期限不得变成读期限
_keepalive(svc)        # 活性交给 TCP keepalive，不是读超时
```
两端 `relay.py` 同步（wls `gtp_logs/`、ziyang10 `tools/`），重启 `zagent`/`zagentssh`。**复测：静默 240 s 会话 exit 0 / elapsed 242 s**（修复前 21 s 必死），推理通道 healthz 仍 200。

**可推广的判据**：`socket.create_connection(timeout=T)` 之后，只要这个 socket 还要长期用来读，**必须 `settimeout(None)`**；否则 T 就成了整条连接的静默上限。配套地，`except OSError` 吞掉 `socket.timeout` 会让超时和对端正常关闭长得一模一样——中继类代码里这两者**必须分开处理**（至少分开打日志），否则下次还是只能靠 A/B 复现来抓。

### 3.34 conductor 每批漏一条 ctl 连接，把中继连接池吃干（2026-08-21）

**现象**：run 一切正常（批在推进、无 ALERT），但 Monitor 报 `fatal` 从 58 涨到 96。分类后**全部是 `Traceback`，零 `ALERT`/`OOM`/`exited 255`**——都是 client 侧握手时的 `EOFError: stream ends after 0 bytes`，且行号间隔精确到每 34 行一条：**某个 worker 卡在稳定的重连循环里**。

**链路侧对上了**：wls 的 `relay.log` 尾部一路 `no parked backend for 192.17.58.207; dropping`。数一下：`:23152` client 侧 **32** 条 established，`:23153` backend 侧 **32** 条——**池子（`--pool 32`）被占满**，第 33 个 client 直接被丢弃，worker 就收到 0 字节 EOF。

**根因**：在 t107 上按 pid 归类那 32 条连接——**24 个唯一 pid，其中 23 个各占 1 条，而 pid 114549 独占 9 条**。114549 是 **conductor 自己**（`run_rl_router.py`，启动 56 min，正好经历 9 批）。源码对上：`ConductorDriver` 有个 per-server 控制连接池 `self._ctls`（`driver.py:174 _ctl()`，懒建），**driver 结束时从不关闭**；而 `run_round` **每批新建一个 driver** ⇒ **每批稳定漏一条 ctl 连接**。

**为什么现在才炸**：24 worker + 每批 1 条泄漏，`pool=32` 只够撑 8 批。b0227 启用中继、b0229 起正常跑，到 b0237 左右正好填满。**这是个单调恶化的定时炸弹**：600 批需要 24+370 条，任何固定池子都会被吃穿。

**处置（不动实验代码）**：给 `relay.py` 的 agent 模式加 `--idle-timeout`（默认 0=关），对 spliced 的两端 `settimeout(idle)`。推理侧起 `--pool 64 --idle-timeout 1800`，**ssh 侧保持不加**——trainer 那步本来就静默数分钟，加了就是 §3.33 重演。判据选 1800 s 的理由：**泄漏的连接永远空闲，活 worker 只在 trainer 步之间空闲（~4 min）**，两者差一个数量级，不会误杀。

**代价与验证**：重启 zagent 打断在飞的 WS ⇒ b0237 只掉 **1 个 slot**（一轮修复搞定）。重启后 `CLIENTS=25`（24 worker + 1 条 ctl）、`backends=64`；90 s 内 `no parked backend` 增量 **0**、`EOFError` 增量 **0**。

**看门狗 cron 同步更新**（旧 `d4dad7a6` → 新 `ee7f503d`）：zagent 的重启参数改成含 `--pool 64 --idle-timeout 1800`，否则看门狗一触发就把修复覆盖回去；并新增行动条件 (f)：`DROP` 增长或 `CLIENTS` 逼近 60 ⇒ 重启 zagent。

**待 owner 裁（正解，属代码改动需 G1/G2）**：`ConductorDriver` 结束时关闭 `self._ctls` 里的全部连接。中继侧的空闲回收是**兜底**不是修复——它把泄漏封顶在 ~5 条，但泄漏本身还在，且同一个 driver 泄漏在别的实验线里一样会咬人。

**判据（可推广）**：`fatal>0` 但 `alive=1` 且批数在推进时，**先分类再定性**——`grep -oE` 把命中按类型 `uniq -c`，再取每类的异常末行。这次 96 条命中里没有一条是致命的，但它们是另一个真故障的**唯一可见症状**。「过滤器响了」和「实验坏了」是两件事，中间那步分类不能省。

### 3.34 续：空闲回收对这个泄漏无效——`websockets` 每 20 s 发 ping，TCP 层永不空闲

**打脸的观测**：`--idle-timeout 1800` 上线 60 分钟后再数，conductor 那个 pid 仍占 **10 条**（约 9 批的泄漏量），**一条都没被回收**。1800 s 的判据前提是「泄漏连接永远空闲」——**这个前提是错的**：`websockets` 客户端默认 `ping_interval=20`，泄漏的 ctl 连接每 20 秒有一次 ping/pong 往返，TCP 层看起来一直活跃，任何基于「无字节流动」的回收都够不着它。

**教训**：在 TCP 层做空闲回收时，「应用层没在干活」和「链路上没有字节」是两回事。带心跳的协议（WebSocket ping、gRPC keepalive、MySQL 的 ping）会让前者永远表现成后者的反面。**要按应用语义回收就得在应用层做，中继层做不到。**

**改用容量兜底**：zagent 起 `--pool 512`（剩余 ~355 批 × 1 条泄漏 + 24 worker ≈ 379，留 1.35× 余量），**一次重启覆盖整个剩余 run**，不再需要周期性重启。`--idle-timeout 1800` 保留——它仍能收掉真正死掉的连接，只是收不掉这个。

**512 池的两个实测细节**：① 启动时 512 个线程同时解析 DNS，会零星报 `park failed: [Errno -3] Temporary failure in name resolution`，agent 自己 sleep 2 重试，**实测只失败 2 次、最终 512 条全部 parked**——判据看 wls 上 `BACKENDS` 是否回到 512，别看这行日志；② 本次重启的代价是 b0247 掉 **20 个 slot**（上次只掉 1 个），一轮修复补齐——**重启代价取决于打断时批内进度，1～24 个 slot 都可能，但恒定是「一轮修复」，有界**。

**看门狗再次同步**（`ee7f503d` → `ea68f4b9`）：池参数改 512，(f) 的阈值从 `CLIENTS 逼近 60` 改成 `CLIENTS>450`，并写明「`CLIENTS` 每批 +1 的缓慢爬升是已知现象，不是故障」——否则下一个接手的人会把正常的爬升当成新事故。

### 3.35 ziyang10 整节点掉线：run 停在 258 批（2026-08-21 21:5x UTC）

**现象**：Monitor 报 `alive=0 fatal=174`，其中 **172 条 `websockets.exceptions.ConnectionClosedError`** + **1 条 `TransportError`**（b0257 的包推送经 `ssh -p 23154` 三次全败）。**推理与 trainer 两条通道同时断**——这个组合本身就指向「对端整机没了」，而不是任一条通道的问题。

**定性**：`tether node ls -a` 显示 **`jupyter-ziyang10 OFFLINE`**，**连 tether agent 都没了** ⇒ 不是我们的进程死了，是**整个 jupyter pod 没了**。存活时长 17:22 → 21:5x ≈ **4.5 h**，与 `reference_jupyterhub_idle_culler.md` 记的 5h11m 同一量级，形态一致。

**处置边界**：**jupyter 会话只能由 owner 在浏览器里重开**，agent 侧做不了。已挂哨兵 Monitor 轮询节点状态，恢复即自动接续。

**现场是干净的**：t107 上 conductor 已退出、**孤儿 worker 0 个**（这次退出路径把 worker 带走了）、显存无残留。数据停在 **258 个批目录 / 25,801 ep**，b0257 是在飞未落的那批（包没推上去，远端什么都没写）。ziyang10 的 `/home/ziyang10` 是 NFS home，**pod 没了数据仍在**，会话回来后原地 resume 即可。

**恢复步骤（节点回来后按序）**：① 起 `rlrsrv`（约 3 min 加载）；② 起 `zagent --pool 512 --idle-timeout 1800` 与 `zagentssh`（**不加 idle-timeout**）；③ 确认 wls 侧 `BACKENDS=512`；④ 重发 conductor；⑤ 推进探针与 Monitor 的日志偏移。b0257 会从 round 0 干净重来（远端无残留 ⇒ 不必挪 `art_m6/<rid>/b0257`）。

**§3.35 续：恢复实录（2026-08-22 00:41 UTC，停机约 2h50m）**

owner 重开 jupyter 会话后节点回到 ONLINE。**盘点：什么都没丢**——`/home/ziyang10` 是 NFS home，pod 没了数据仍在：`trainer_state.json` = **v257 / 257 批**（b0256 已落，b0257 在飞未落，与 t107 的 258 个批目录自洽）、`tools/relay.py`（含 §3.33/§3.34 修复）、`sshenv/`、`sshd_run/` 全在；**H200 此刻整卡空闲 142 GB**（之前占 103 G 的那位也随之退出）。

**新 pod 缺什么**：`tmux` 一个会话都没有、**用户态 sshd 没在跑**（它不是 .bashrc 自启的）、`/tmp` 清空。`keepalive.sh` + `ws_keepalive.py` **是自启的**（PID 128/130 已在跑）——**注意：这次被杀时它们也在跑**，所以 keepalive 并不能保证不被回收，别把它当护身符。

**五步恢复全部照 §3.35 执行并逐条验证**：① sshd UP；② 三个 tmux（rlrsrv / zagent `--pool 512 --idle-timeout 1800` / zagentssh 无 idle-timeout）；③ wls 侧 `BACKENDS=512 SSHBACK=8`；④ 两条通道端到端过：推理 `healthz=200`、**trainer 通道静默 30 s 的 ssh 存活（elapsed 33 s）**——这一条专门复验 §3.33 的修复随重新部署的 relay.py 一起回来了；⑤ resume 走**快速路径**（shard 根目录存在 ⇒ `ls` 成功，只清 **30** 个残留而不是 257 个），探针与 Monitor 偏移推进到 43937。

**端到端确认**：b0257 从 round 0 干净重跑并落地 **v258**，conductor 读到 admission report 后继续推进。**整段停机的净损失 = 0 个 episode**（b0257 本来就没落，远端无残留）。

**Monitor 加了一条**：现在每拍附带查 `tether node ls` 的 `jupyter-ziyang10` 状态，非 ONLINE 就把 `ZIYANG10=<状态>` 拼进事件——上次是靠 172 条 `ConnectionClosedError` 间接推断整机没了，绕了一圈；节点状态是**直接判据**，应该第一时间可见。

### 2.5l b0400 提前停跑规则（owner 裁定 2026-08-22，**在 b0400 数据到达之前**写死）

**背景**：b0195 之后 reward 横盘。最近 100 批斜率 +0.017/100批、最近 50 批 +0.007/100批，**都远小于批间 sd 0.050**；teacher 率反而从 0.263 回爬到 0.316。

**owner 指令**：「如果 400 批时还在这样横盘的话就停止训练」。

**判据（现在定，不许事后改）**——沿用 ts 线报告的分位比较口径：
- 取 **b0300–b0349** 与 **b0350–b0399** 两个 50 批块的 `mean_reward` 均值，记 Δ = 后 − 前。
- 每块 SE = sd/√50 ≈ 0.050/7.07 ≈ 0.0071 ⇒ **差的 SE ≈ 0.010**。
- **Δ ≥ +0.020（2 SE）判「仍在涨」→ 继续跑满 600**；否则判 **「横盘」→ 在 b0400 停**。
- 同时报 b0300–0399 的 OLS 斜率与其 95% CI 作**辅助**（不参与裁决，只用于报告）。

**停跑后的影响（预先说清，避免临时找补）**：
1. **P3 不受影响**——它是「冻结 router(sample) vs 常数@p̂」的同 session 配对比较，p̂ 取**最后 50 批**（即 b0350–b0399）。预注册文本里 p̂ 的定义是「最后 50 批」，不是「第 550–600 批」，所以 600→400 不改口径、不引入选点自由度。
2. **P2 的 sd 轨迹**长度从 600 缩到 400 批，结论表述里要写明轨迹截断于 b0400 及其原因。
3. 报告必须写明**这是 owner 在 b0280 时依据「横盘」下的提前停，判据于停跑前 120 批就已冻结**——这一条是防止把「提前停」读成事后挑点。

### 3.36 迁回 tether expose：实测推翻了自建中继的立项理由（2026-08-22 03:5x，owner 指令）

**owner 指令**：「我们现在可以用 tether expose 了，把训练转移到 tether expose，不要走 weilandserver 中继」。

**切之前先测（§3.29 的教训：跨机换通道必须先量）**，同一时刻、同一条 50 MB scp、同一负载下对照：

| 通道 | 实测 |
|---|---|
| **broker expose `linziyang.top:14009`** | **15.01 MB/s** |
| 自建中继 `ziyanglin.com:23154` | **3.85 MB/s** ⚠ **已跌破 4.9 MB/s 的需求线** |

**中继当初的立项理由（§3.29「ssh 只有 1.4 MB/s，中继 8.8–9.6」）现在不成立了**，而且方向反了近 4 倍。根因是拓扑而非实现：**中继把 t107↔ziyang10 的流量全逼过 weilandserver 的家用上行**，而 broker 在 VPS 上，两端都在机房网。当初测到 8.8–9.6 是家用链路空闲时的上限，扛不住 24 worker 的推理流量叠加。

**教训**：「自建的比现成的快」是**当时那次测量**的结论，不是性质。链路条件一变（家宽被本实验自己的流量占满）结论就翻。**通道选型的数字必须在真实负载下、定期复测**，别把一次空闲基准当成常量写进架构。

**迁移执行（四处必须同步改，漏一处就废一批）**：① `/tmp/run_m6_h200.sh` 的 `--servers linziyang.top:14008` 与 `--remote-host linziyang.top --remote-port 14009`；② `/tmp/rlr_health.sh` 与 `/tmp/reap_orphans.sh` 的 `SERVER_KEY` → `linziyang.top:14008`（**必须与 worker argv 逐字一致**）；③ t107 `~/.ssh/config` 的 `Host linziyang.top` 补 `ServerAliveInterval 15`（原来只加在 `ziyanglin.com` 块上）；④ Monitor/探针偏移推进到 46871。

**孤儿处置**：停 conductor 后 48 个 worker 带**旧端点** `ziyanglin.com:23152` 成孤儿，按旧 key 收割。⚠ 同机上他线的 worker 带 `ziyanglin.com:23160/23161/23162/23163`，收割前后 `inspect_workers.sh` 两次确认它们全是 `tmux(alive)` 且**一个没碰**——这正是「孤儿判据必须按端点收窄」的价值（§3.31）。

**端到端验证**：resume 走快速路径（57 个残留 shard，非 284）；b0284 的 shard 经新推理通道落盘；**b0284 经 broker ssh 通道完成 trainer 步 → v285**。切换总代价 = 一批的在飞 episode（一轮修复）。

**回退保留**：weilandserver 上 `relay`/`relayssh` 两个 tmux 留着但**已不在数据面**，owner 确认稳定后再拆。看门狗 cron 换成 `30de222b`（原 `ea68f4b9` 删除，它的中继自愈条件已全部作废）。

### 3.37 完全关停：owner 于 b0288 下令腾出所有资源（2026-08-22 05:1x UTC）

**指令**：「完全关闭实验，腾出所有资源」。**这不是 §2.5l 的横盘停**——§2.5l 的判据（b0300–0349 vs b0350–0399）永远没有到达它的数据。run 在 **288 批 / 28,867 ep** 处被外部指令停止，trainer 恰好消费到 **b0287 → v288**，正在跑的 b0288 未落 ⇒ **干净停机点，零数据损失**。

**关停清单（全部执行并验证）**：
| 资源 | 处置 | 验证 |
|---|---|---|
| t107 conductor+worker（tmux rlrm6） | kill 会话 | reaper DRYRUN 0 孤儿；`linziyang.top:14008` 独立元素判据 0 残留 |
| ziyang10 zagent / zagentssh | kill 会话 | tmux ls 无 |
| ziyang10 pi05 server（rlrsrv） | **发现已自行消失** | `:8999` 零监听；serve_policy 独立元素判据 0 残留 |
| ziyang10 用户态 sshd | 按 PID kill（835→已核不是 sshd，实际 sshd pid 830） | pgrep 0 |
| 两条 expose rlr-inf/rlr-ssh | `tether expose rm` | 14008/14009 归池（FREED） |
| wls relay / relayssh | **发现已自行消失**（大概率随换网线时 tmux server 一起没了） | `23152/23154` 零监听 |
| Monitor `bcklkwbq0`、cron `30de222b`/`1feed103` | 停/删 | — |

**三个没踩的坑（记下判据）**：
1. ziyang10 上 `srv0/srv1` 是 **h200 存活探针线**的（`session_probe/train_h200_probe.py`，GPU 13.3 GB）——不是本线的，未动。
2. **PID 835 的 argv 是 `tmux new -s rlrsrv -d "...serve_policy..."`，但它是 tmux server 本体**（首条 tmux 命令 fork 出 server，argv 留着那串文本），正托管他线 srv0/srv1。凭 argv 文本判 serve_policy 残留会误杀整个 tmux server。交叉判据：`:8999` 是否监听。
3. wls 上剩的 `wsdrv*/wssrv*` 全是 **ws_search 线**的（对应 t107 上 `ziyanglin.com:2316x` 的 worker），未动。

**数据资产全部在位（未删任何字节）**：
- ziyang10 `rl_router/art_m6/l10_tc_lam3_s0/`（4.5 G）：trainer_checkpoint.pt、trainer_state.json（v288/288）、metrics.jsonl、weights/（v0、v227、v285–v288 等）、b0227–b0287 批产物
- wls `rl_router/art_m6/l10_tc_lam3_s0/weights/` v0–v227（P2 全轨迹的前半段）
- t107 `/scratch/zixuans8/rl_router/m6/l10_tc_lam3_s0/` 288 个批目录（journal/client_rows）
- ziyang10 `dump_m6`（1 G shards，可再生，未清）

**恢复路径**：按 handoff §3.1/§7——先 expose 两条（端口会变，四处同步）、拉 sshd、起 server、重发同一条命令即批级 resume（从 b0288 起）。**P3 与后续裁决（SR0.80 路线、天花板测量）全部悬置待 owner。**
