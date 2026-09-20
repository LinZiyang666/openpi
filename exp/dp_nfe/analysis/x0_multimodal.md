# x₀-head × 标签过滤数据实验 — 结果报告（Diffusion Policy，trailing_v1）

> 实验线：`logs/x0_multimodal_plan.log.md`（plan v3；G1 APPROVED R3、G2 R2 APPROVED；§10 执行记录）。数据/判决文件：`exp/dp_nfe/data/x0_multimodal/`（`merged/` 两机合并产物、`decisions.json`、`diagnostics_table.{json,md}`、`figures/`；data 槽不入库）。运行时间 2026-09-18 16:23 → 2026-09-19 19:41 CDT。

## 1. 一句话结论

在 DP 官方仓（rev `5ba07ac6`，diffusers 0.11.1）上，**一步推理归零是 ε 参数化的问题，不是数据多模态的问题**：同代码、同预算、同 seed 下，ε 头在单模态子集 U 上 DDIM-100→DDIM-1 掉 0.75（PushT）/ 0.84（Square-MH），x₀ 头（`prediction_type=sample`）一步折损 ≤0.01；x₀ 头在混合子集 M 上的一步折损与 U 无差别（PushT S_x0 = −0.008 [−0.041, +0.026]，在 δ=0.05 内等价）。代价是 x₀ 头**在任何步数下的锚点都比 ε 头低 4–8 pp**，且在多模态更强的数据上锚点掉得更多（Square-MH M −6 pp、image M −24 pp、explore transport_mh −12 pp）；诊断显示 x₀ 头的条件采样离散度 ≈ 0——它实际上退化成了一个确定性回归器，"一步无损"是因为它在 100 步时也没有在采样。

## 2. 设置

| 项 | 内容 |
|---|---|
| 因子 | 头 ∈ {ε, x₀(`sample`)} × 数据 ∈ {U 单模态子集, M 同规模混合子集} × seed ∈ {42,43,44}；核心任务 pusht / blockpush / square_mh（lowdim）；kitchen 核心格不可用（§8） |
| 预算 | 固定 optimizer-step：lowdim 100k、image 40k（pilot 吞吐定，§5.2）；batch 256；EMA、normalizer（训练池冻结）、runner 全官方 |
| 协议 | trailing_v1：DDPM-100（辅锚）、DDIM-100（主锚）、DDIM-10/4/2/1；keyed noise `(sampling_seed, episode_id, decision_idx, kind, t)`；screening 32 集（seed 90000+，Q≥0.5 门）+ test 100 集（image 50，seed 100000+） |
| 判决 | 预注册 §4.5：S_x0 = D1(x₀,M) − D1(x₀,U)、H2 = [D1(ε,U) 下界 > 0.20 且 D1(x₀,U) 上界 < 0.05]、I = S_x0 − S_ε；D1 = Q(DDIM-100) − Q(DDIM-1)；episode 为推断单元，先按 seed 平均再 bootstrap 20000（seed 20260918）；Bonferroni 16 族 → 99.6875% 百分位区间；δ = 0.05 |
| 数据身份 | 训练池/子集/held-out 以 sha256 冻结在每格 identity 中；两机 raw 数据 sha 相同、子集 selection digest 相同（zarr 字节因 blosc 非确定按主机各绑）。子集：pusht 池 185 集（右绕 108 / 左绕 47 / 未知 30）→ U=right 80、M=56 右 + 24 左；blockpush 池 900 → U=`first=0|b0:t0,b1:t1` 237、M 四种顺序 63/59/58/57；square_mh 池 270 → U=operator_1 47、M=24+23 |
| 机器 | weilandserver（4090，torch 2.5.1+cu124）与 h100（H100，同版本）。pusht/blockpush 的 U 与 M-x₀ 在 wls、**M-ε 三个 seed 在 h100**（9-19 再平衡，§8）；square_mh 12 格全在 h100；image 8 格 h100；explore 20 格两机各 10 |
| 完整性 | 64/64 格训练+评测（core 36、explore 20、image 8；wls 26 / h100 38），466 条评测记录 valid、0 invalid、0 unexpected；`completeness.pending` 全空 |

## 3. 正式判决（核心 lowdim，三 seed，n=100 集/格）

区间为 Bonferroni-16 的 99.6875% bootstrap 百分位区间。

| 任务 | D1(ε,U) | D1(x₀,U) | D1(ε,M) | D1(x₀,M) | S_x0 | S_ε | I | 判决 |
|---|---|---|---|---|---|---|---|---|
| pusht | +0.750 [+0.658, +0.834] | +0.010 [−0.012, +0.033] | +0.817 [+0.773, +0.857] | +0.002 [−0.013, +0.018] | −0.008 [−0.041, +0.026] | +0.067 [+0.010, +0.125] | −0.074 [−0.173, +0.016] | **H2 supported；H1-data not supported（S_x0 在 δ 内等价）；I not supported** |
| square_mh | +0.843 [+0.750, +0.923] | +0.007 [−0.047, +0.067] | +0.767 [+0.700, +0.827] | −0.000 [−0.047, +0.047] | −0.007 [−0.100, +0.089] | −0.077 [−0.163, +0.007] | +0.070 [−0.093, +0.233] | **全部 inconclusive**（点估计与 pusht 同向，但 100 集二值指标在 99.69% 区间下 D1(x₀,U) 上界 0.067 > 0.05） |
| blockpush | +0.041 | +0.008 | +0.025 | +0.002 | −0.007 | −0.017 | +0.010 | **无判决**：12 格 screening 全 < 0.5（0.000–0.062），估计值仅列出 |
| kitchen | — | — | — | — | — | — | — | skipped：标签只有一种顺序/完成集合 |

锚点（DDIM-100，seed 均值）：pusht ε-U 0.834 / x₀-U 0.761 / ε-M 0.900 / x₀-M 0.824；square_mh ε-U 0.843 / x₀-U 0.807 / ε-M 0.767 / x₀-M 0.707。逐 seed 点估计范围窄（pusht D1(ε,U) 0.742–0.757、D1(x₀,U) −0.004–0.030；square D1(ε,U) 0.81–0.87、D1(x₀,U) −0.03–0.04），三 seed 结论一致。

## 4. 核心 Q 阶梯（seed 均值；DDPM-100 / DDIM-100 / 10 / 4 / 2 / 1）

| 格 | DDPM-100 | DDIM-100 | DDIM-10 | DDIM-4 | DDIM-2 | DDIM-1 |
|---|---|---|---|---|---|---|
| pusht ε·U | 0.835 | 0.834 | 0.816 | 0.800 | 0.758 | **0.084** |
| pusht x₀·U | 0.758 | 0.761 | 0.753 | 0.754 | 0.749 | **0.752** |
| pusht ε·M | 0.893 | 0.900 | 0.876 | 0.858 | 0.783 | **0.083** |
| pusht x₀·M | 0.829 | 0.824 | 0.829 | 0.826 | 0.816 | **0.822** |
| square_mh ε·U | 0.823 | 0.843 | 0.820 | 0.810 | 0.430 | **0.000** |
| square_mh x₀·U | 0.817 | 0.807 | 0.823 | 0.827 | 0.800 | **0.800** |
| square_mh ε·M | 0.773 | 0.767 | 0.783 | 0.770 | 0.350 | **0.000** |
| square_mh x₀·M | 0.733 | 0.707 | 0.710 | 0.737 | 0.727 | **0.707** |
| blockpush ε·U | 0.046 | 0.043 | 0.036 | 0.028 | 0.015 | 0.002 |
| blockpush x₀·U | 0.016 | 0.023 | 0.018 | 0.015 | 0.013 | 0.015 |
| blockpush ε·M | 0.021 | 0.025 | 0.016 | 0.011 | 0.011 | 0.000 |
| blockpush x₀·M | 0.008 | 0.012 | 0.007 | 0.007 | 0.007 | 0.010 |

逐格六档与 screening 见 `decisions.json`（`tasks[*].ladders`、`per_seed`）；图 `figures/x0_core_ladders.png`、`figures/x0_decisions.png`。ε 头的折损从 DDIM-2 开始（pusht −0.08、square −0.4），DDIM-1 归零；x₀ 头六档平坦。

## 5. 描述性臂（不进正式族）

### 5.1 image（seed 42，50 集；U/M 对的 D1/S 为描述性 95% 区间）

| 格 | DDPM-100 | DDIM-100 | DDIM-10 | DDIM-4 | DDIM-2 | DDIM-1 |
|---|---|---|---|---|---|---|
| pusht image ε·U | 0.801 | 0.758 | 0.812 | 0.761 | 0.644 | 0.093 |
| pusht image x₀·U | 0.754 | 0.690 | 0.663 | 0.689 | 0.721 | 0.732 |
| pusht image ε·M | 0.880 | 0.875 | 0.876 | 0.833 | 0.747 | 0.092 |
| pusht image x₀·M | 0.810 | 0.757 | 0.760 | 0.745 | 0.809 | 0.743 |
| square_mh image ε·U | 0.48 | 0.52 | 0.46 | 0.56 | 0.04 | 0.00 |
| square_mh image x₀·U | 0.50 | 0.50 | 0.48 | 0.42 | 0.42 | 0.40 |
| square_mh image ε·M | 0.60 | 0.54 | 0.62 | 0.70 | 0.06 | 0.00 |
| square_mh image x₀·M | 0.32 | 0.30 | 0.26 | 0.26 | 0.28 | 0.32 |

pusht：D1(ε,U) +0.665 [0.579, 0.748]、D1(x₀,U) −0.042 [−0.119, +0.039]、S_x0 +0.057 [−0.048, +0.161]；square_mh：D1(ε,U) +0.52 [0.38, 0.66]、D1(x₀,U) +0.10 [−0.06, +0.26]、S_x0 −0.12 [−0.30, +0.06]。square image 40k 步的锚点整体偏低（ε-U 0.52 vs 官方 ckpt 0.72），且 **x₀·M 的 screening 只有 0.19、锚 0.30**——x₀ 头在多模态 image 数据上的损失出现在锚点而不是一步折损。

### 5.2 explore（全量官方数据集，seed 42，100 集；screen | DDPM-100 / DDIM-100 / 10 / 4 / 2 / 1）

| 任务 | ε | x₀ |
|---|---|---|
| square_ph | 1.000 \| 0.88 0.92 0.89 0.92 0.68 **0.00** | 0.969 \| 0.92 0.92 0.93 0.91 0.93 **0.95** |
| square_mh | 0.688 \| 0.78 0.85 0.79 0.87 0.32 **0.00** | 0.812 \| 0.70 0.72 0.70 0.76 0.69 **0.67** |
| transport_ph | 0.781 \| 0.73 0.81 0.83 0.76 0.00 **0.00** | 0.594 \| 0.67 0.72 0.69 0.72 0.75 **0.73** |
| transport_mh | 0.500 \| 0.42 0.40 0.41 0.37 0.00 **0.00** | 0.250 \| 0.26 0.28 0.22 0.22 0.19 **0.19** |
| can_ph | 1.000 \| 0.96 0.96 0.98 0.97 0.86 **0.00** | 1.000 \| 1.00 0.99 0.99 0.99 0.99 **1.00** |
| can_mh | 0.969 \| 0.99 0.97 0.98 0.97 0.66 **0.00** | 0.906 \| 0.91 0.92 0.89 0.89 0.92 **0.92** |
| tool_hang_ph | 0.812 \| 0.74 0.70 0.69 0.66 0.04 **0.00** | 0.438 \| 0.44 0.43 0.45 0.39 0.39 **0.38** |
| kitchen | 0.558 \| 0.569 0.569 0.564 0.551 0.197 **0.000** | 0.562 \| 0.567 0.580 0.570 0.571 0.579 **0.573** |
| blockpush | 0.123 \| 0.227 0.212 0.168 0.138 0.064 **0.000** | 0.092 \| 0.138 0.158 0.149 0.153 0.164 **0.148** |
| pusht | 0.964 \| 0.938 0.930 0.947 0.949 0.914 **0.081** | 0.935 \| 0.930 0.931 0.930 0.916 0.937 **0.926** |

10/10 任务：ε 头 DDIM-1 归零（pusht 0.08）、DDIM-2 已大幅折损；x₀ 头 DDIM-1 ≈ DDIM-100。x₀ 锚点相对 ε：ph 数据 ≈ 持平（square_ph 0.92/0.92、can_ph 0.99/0.96、pusht 0.93/0.93）；mh / 多模态数据明显更低（square_mh 0.72 vs 0.85、transport_mh 0.28 vs 0.40、tool_hang 0.43 vs 0.70）。

### 5.3 P0 官方 ε image checkpoint（50 集，wls torch 1.12 环境，只作背景）

pusht 0.833 / 0.876 / 0.865 / 0.846 / 0.637 / **0.072**；square_mh 0.74 / 0.72 / 0.68 / 0.38 / 0.00 / **0.00**；can_mh 0.92 / 0.94 / 0.94 / 0.98 / 0.04 / **0.00**。与本矩阵 ε 头行为一致（历史 leading 网格结果另行标识，不合并）。

## 6. 分布诊断（§4.3；每格 64 个 held-out history × 64 个样本，共享噪声键；表 `diagnostics_table.md`）

| 任务·数据 | 头 | 采样 | 条件离散度（std 单位） | ΔBIC(2 vs 1) | 未裁剪越界比例 | 最近 demo 距离 |
|---|---|---|---|---|---|---|
| pusht·U | ε | DDIM-100 | 0.03 | −10 | 0.000 | 0.57 |
| pusht·U | ε | DDIM-1 | **8.48** | 114 | **0.895** | **8.17** |
| pusht·U | x₀ | DDIM-100 | **0.00** | −21 | 0.000 | 0.59 |
| pusht·U | x₀ | DDIM-1 | 0.00 | −19 | 0.000 | 0.59 |
| square_mh·U | ε | DDIM-100 | 0.18 | 30 | 0.040 | 1.36 |
| square_mh·U | ε | DDIM-1 | **17.8** | 28 | **0.949** | **17.5** |
| square_mh·U | x₀ | DDIM-100 | 0.02 | −3 | 0.069 | 1.49 |
| square_mh·U | x₀ | DDIM-1 | 0.02 | −14 | 0.068 | 1.51 |
| square_mh image·M | ε | DDIM-1 | 22.8 | −5 | 0.956 | 19.1 |
| square_mh image·M | x₀ | DDIM-100 | 0.04 | 34 | 0.075 | 1.26 |

（M 子集、blockpush、pusht image 各行同型，见完整表。）三条机制性观察：

1. **ε 头一步归零的直接机制是 x̂₀ 估计爆炸**：从纯噪声一步得到的 x̂₀ 有 89–96% 的坐标落在归一化范围之外（裁剪前），离最近 demo 8–19 个 std，条件离散度 8–23——`x̂₀ = (x_T − √(1−ᾱ_T)·ε̂)/√ᾱ_T` 在 ᾱ_T→0 时把 ε̂ 的误差放大；100 步时同一网络的样本离散度只有 0.03–0.36、越界 ≤6%。
2. **x₀ 头在任何步数下都几乎不采样**：64 个不同初始噪声给出的动作块两两距离 ≈ 0.00–0.04 std（DDIM-100 与 DDIM-1 相同），ΔBIC 为退化的负值。它学到的是条件均值回归器；一步"无损"只是因为它在 100 步时也在输出同一个点。这解释了 §3：数据多模态对 x₀ 头的一步折损没有影响（S_x0 ≈ 0），影响落在锚点本身（§5.1/5.2 的 M / mh 锚点下降）。
3. ε 头 100 步的条件分布在这些任务上也接近单峰（pusht ΔBIC −10，square 30 但离散度 0.18）：给定 history 的动作块分布不是强多模态，多模态主要体现在 episode 级策略（左/右绕、操作者），而不是单步条件分布——这也是 x₀ 头能保住大部分闭环性能的原因。

## 7. NFE / 成本

- 每次策略调用的去噪 NFE：DDPM-100 = 100、DDIM-k = k（identity 中 `nfe_per_call`）。x₀ 头 DDIM-1 相对 DDIM-100 减少 100× 网络前向而 Q 不变；ε 头在 ≤2 步不可用（DDIM-4 是 ε 头的实用下限：pusht −0.03、square −0.03、can_mh −0.00、transport_ph −0.05）。
- 训练：lowdim 100k 步 ≈ 66M 参数 UNet；单进程 launch-bound（4090 14 upd/s、H100 17 upd/s，GPU 功耗 <50%）。本次靠多 lane + CUDA MPS 把两机聚合吞吐提到 46–76 upd/s（§10.2）。总计 64 格 = 5.6M lowdim 更新 + 0.32M image 更新，两机合计约 27 h 墙钟。评测 lowdim 六档+screen ≈10–15 min/格、image ≈30 min/格。

## 8. 筛选、缺项与偏差

| 项 | 处理 |
|---|---|
| kitchen 核心格 | 不可用：标签检查发现每个"完成任务集合"只对应一种顺序（npy 三件套掩码非前缀、改用 .mjl 后仍然），无法构造 U/M；manifest 记 `skipped: fewer than two valid labels`。explore 用全量 kitchen（ε 0.57→0.00，x₀ 0.58→0.57） |
| blockpush | 12 格 screening 0.000–0.062 < 0.5：237 集单模态 + 100k 步下 ε 与 x₀ 头都没学会（训练 loss 1e-4、held-out val MSE 单调上升，过拟合）；按 §4.4 不追加/缩减预算，正式判决记无判决，估计值照列。全量 900 集 explore 也只有 0.21/0.16 |
| square image | 40k 步锚点（ε-U 0.52）低于官方 ckpt（0.72），image 臂预算不足以饱和；只作描述 |
| 跨机训练 | 9-19 再平衡：pusht/blockpush 的 M-ε 三个 seed 在 h100、其余在 wls。S_x0 / H2 只用 wls 格；S_ε、I 含跨机对照（同代码/torch/数据/seed，仅 GPU 核浮点次序不同，视为随机实现差异）。归属见 `merged/merge_report.json` |
| 并发 / MPS | 训练中途改 lane 并发与启用 MPS，只改 wall-clock；每格配置、seed、步数、代码 hash 不变（identity 全程一致） |
| 续训 | owner 两次重启 wls 与三次拓扑调整均从 `latest.ckpt`（每 2000 步）续训，续训等价有 CPU 测试；每格 identity/resolved-config 校验通过 |
| 环境 | wls torch 1.12 → 2.5.1（速度）；P0 官方评测在 1.12 下完成，不与矩阵合并 |

## 9. Claim 边界

- 成立的：在 DP 官方实现、固定预算下，**ε 参数化是一步归零的充分原因，x₀ 参数化消除了折损**（pusht 正式 supported；square_mh 点估计同向但区间不够窄；explore 10/10 任务描述性一致）。**数据多模态不改变 x₀ 头的一步折损**（pusht S_x0 在 ±0.05 内等价）。
- 不能说的：x₀ 头"和 ε 头一样好"——锚点普遍低 4–8 pp，多模态数据上更低（−6 到 −24 pp）；x₀ 头不是一个采样器（条件离散度 ≈ 0），它的闭环成功率是确定性回归器的成功率。是否用 x₀ 头应按"多模态数据上锚点损失 vs 100× NFE 节省"权衡。
- 外推限制：只测了 DP UNet-1D + DDPM/DDIM（无 flow / EDM 预条件、无 v-prediction、无 CFG）；预算固定（100k / 40k）；blockpush 与 kitchen 核心格无判决；image 臂单 seed 50 集。

## 10. 复现

```
# 两机：矩阵冻结与队列（plan §8）
bash exp/dp_nfe/ops/x0_queue.sh train core,explore,image ; bash exp/dp_nfe/ops/x0_queue.sh eval core,explore,image --parallel N
bash exp/dp_nfe/ops/x0_dispersion.sh "^(pusht|blockpush|square_mh)_(lowdim|image)_(U|M)_" 100,1
# 本机：拉取 + 合并 + 判决 + 诊断表 + 图
bash exp/dp_nfe/ops/pull_x0_results.sh
uv run python -m exp.dp_nfe.analysis.aggregate_x0 --cells exp/dp_nfe/data/x0_multimodal/merged/cells --root exp/dp_nfe/data/x0_multimodal/merged/results_trailing --out exp/dp_nfe/data/x0_multimodal/decisions.json --boot 20000 --seed 20260918
uv run python -m exp.dp_nfe.analysis.diagnostics_table --dir exp/dp_nfe/data/x0_multimodal/wls/diagnostics --dir exp/dp_nfe/data/x0_multimodal/h100/diagnostics --out exp/dp_nfe/data/x0_multimodal/diagnostics_table.json --md exp/dp_nfe/data/x0_multimodal/diagnostics_table.md
uv run python -m exp.dp_nfe.analysis.plot_x0 --decisions exp/dp_nfe/data/x0_multimodal/decisions.json --out exp/dp_nfe/data/x0_multimodal/figures
```
