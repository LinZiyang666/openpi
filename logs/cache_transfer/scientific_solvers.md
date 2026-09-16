# 跨领域迁移调研：科学与工程中的迭代生成 / 求解

> 调研对象：把 tiered experience cache + RIT + stateful gate 迁到 "昂贵 trunk + 迭代扩散头 / 迭代求解器" 的科学计算领域。
> 覆盖子领域：AF3 系 cofolding（Boltz-2 / Chai-1 / Protenix）、分子对接（DiffDock）、扩散天气预报（GenCast 及其自回归滚动）、神经 PDE 迭代精炼（PDE-Refiner / ACDM）、医学影像扩散重建（动态 MRI / 多切片）、优化类迭代（MPC warm-start / 学习式 warm-start / bundle adjustment）。
> 依据：`logs/cache_transfer/briefing.md`、`docs/iclr/iclr_paper/arxivd.tex` §2–§3（L64–344）、`logs/session_handoff.md` §1.2。写于 2026-09-15。
> 规则：所有数字均来自引用 URL 或由其直接换算；无来源处写"不确定"。

---

## 0. 一句话摘要

整个领域里 **结构最贴合的是 AF3 系 cofolding**（trunk = MSA module + Pairformer，head = 200 步扩散模块，tap point = trunk 输出的 single/pair representation，天然免费 query），而且 **成本余量比 VLA 好一个数量级**：Boltz-2 上 trunk 只占 ~5.5%、扩散头占 ~93%（TerraBind 实测，[arXiv 2602.07735](https://arxiv.org/pdf/2602.07735)）。但它同时是 **"减步傻基线"杀伤最彻底的领域**：Protenix 不重训、只换 ODE 采样器，200 步 → 2 步的配体界面 LDDT 从 0.650 掉到 0.645（[arXiv 2507.11839](https://arxiv.org/abs/2507.11839)）；Boltz-1 自己的消融说 50 步以后就平台（[bioRxiv 2024.11.19.624167](https://www.biorxiv.org/content/10.1101/2024.11.19.624167v1.full)）。换算下来 **2 步基线的 IR 地板 ≈ 6.4%，与 100% 命中的 cache 地板 5.5% 只差 0.9 pp**，cache 在成本轴上没有生存空间。其它子领域要么没有昂贵共享前缀（天气、PDE、MRI、DiffDock），要么 warm-start 是几十年的标准做法（MPC 的 shifted warm start / RTI）。**结论 NO**，附一条唯一可能翻盘的窄缝（同靶点批量筛选里的 best-of-N 多样性）和两周实验方案。

---

## 1. 领域内最贴合的 2–3 个具体系统

### 1.1 Boltz-2（AF3 系 cofolding，首选候选）

- 模型 / 代码：开源（MIT），[github.com/jwohlwend/boltz](https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md)。默认 `recycling_steps=3`、`sampling_steps=200`、`diffusion_samples=1`；affinity 模式 `sampling_steps_affinity=200`、`diffusion_samples_affinity=5`。支持 template（CIF/PDB，作用在 trunk 条件上，不是头级 warm start）、预计算 MSA。文档里 **没有** 跨 run 复用 trunk 或"一靶多配体"批处理选项。
- Benchmark：PoseBusters（配体 RMSD<2 Å 且物理有效）、PLINDER、FoldBench、CASP16 等；affinity 用 FEP+ / CASF。
- 成本拆分（实测，单 A6000，196 token 复合物，10 个 pose 样本，200 步，**不含 MSA 搜索**，[TerraBind, arXiv 2602.07735](https://arxiv.org/pdf/2602.07735) §"Runtime Decomposition"）：**Trunk 1.53 s，Pose（扩散头）25.92 s，总计约 27.8 s**。⇒ c_pre/c_0 ≈ 5.5%，头 ≈ 93%。
- 参考吞吐：NVIDIA NIM 在 H100 上 pose+affinity 默认配置 ~18–25 s / 复合物（[docs.nvidia.com Boltz2 NIM performance](https://docs.nvidia.com/nim/bionemo/boltz2/1.9.0/performance.html)）；BioNeMo 基准配置 = 3 recycles / 200 步 / 5 samples（[NVIDIA blog](https://developer.nvidia.com/blog/high-throughput-structure-prediction-with-bionemo-inference-runtime/)）。
- 端到端里 MSA 搜索另算：UCSD 的 AF3 workload 表征显示 MSA 阶段占总时间 75–94%，是 CPU 侧的 jackhmmer/nhmmer（[kim-iiswc2025](https://cseweb.ucsd.edu/~jzhao/files/kim-iiswc2025.pdf)）。MSA 缓存是既有工程实践（ColabFold；AF_Cache 缓存 MSA/features，[arXiv 2606.04566](https://arxiv.org/pdf/2606.04566)），与我们的头级 cache 无关。

### 1.2 DiffDock（分子对接，纯扩散头）

- 开源（[arXiv 2210.01776](https://arxiv.org/pdf/2210.01776)，代码 github.com/gcorso/DiffDock）。前缀 = ESM2 蛋白嵌入 + RDKit 构象 + 半径图，论文明说 "preprocessing time is negligible compared to the rest of the inference time"（Appendix，Runtime 段）。头 = 20 步反向扩散 × 40 个样本 + confidence model 排序。
- 减步：Appendix F.3 "the model reaches nearly the full performance even with just 10 steps, suggesting that the model can be sped up 2x with a small drop in accuracy"；<10 步的曲线只在 Figure 11 图里，数值不可读（不确定）。
- Benchmark：PDBBind 时间切分，top-1 RMSD<2 Å 38.2%（40 样本）/ 35.0%（10 样本）。
- c_pre/c_0 ≈ 0：没有昂贵前缀，cache 只能省头，且 key 得自己算（ESM2 嵌入可当 key，但它不是"决策特征"）。

### 1.3 GenCast（扩散天气预报，自回归滚动，唯一有真闭环的候选）

- 代码：graphcast 仓库开源 1.0° 版权重；0.25° 完整版权重同仓（JAX/TPU）。
- 结构（[arXiv 2312.15796](https://ar5iv.labs.arxiv.org/html/2312.15796)）：每个 12 h 步 = 20 个 DPMSolver++2S 步 = 39 次 denoiser 评估；15 天 = 30 个自回归步；条件 = 前两个状态 (X^{t−1}, X^t)；单成员 15 天预报在 Cloud TPUv5 上约 8 分钟；50 成员 ensemble。
- **没有共享前缀**：denoiser 是一个 graph transformer，把条件状态与带噪目标一起编码，每次 NFE 都重算条件编码；tap point 不存在，query 特征要额外算。
- 减步证据：Tyche（[arXiv 2605.06916](https://arxiv.org/html/2605.06916)）Table 3：标准 EDM 采样器 1-NFE RMSE 0.145 vs 蒸馏后 0.103 —— 天真减到 1 步确实掉点，但蒸馏/一致性模型（Swift 1-NFE 比扩散基线快 39×，[arXiv 2509.25631](https://arxiv.org/abs/2509.25631)；Tyche 12–45×）已把这个缺口填平。
- 闭环：真实存在（30 步滚动，误差累积、ensemble collapse 都是该领域核心议题，Tyche 明文以 curriculum CRPS 校准对付 "error accumulation"）。

（其它子系统在 §3 打分表里各给一行，不再单列。）

---

## 2. 映射表（以 Boltz-2 为主，括号内给其它子领域）

| 我们的符号 | Boltz-2 / AF3 系 | 天气 GenCast | 神经 PDE (PDE-Refiner/ACDM) | 动态 MRI 扩散重建 | MPC / 学习式 warm-start |
|---|---|---|---|---|---|
| 共享前缀 (c_pre) | MSA module + Pairformer trunk（64 层 × 3 recycles）；1.53 s / 27.8 s ≈ 5.5% | **无**（条件编码在 denoiser 内部，每 NFE 重算） | 无 | 无（扩散先验即整个模型；物理一致性投影每步做） | KKT 分解 / 线性化（OSQP 的 factorization 跨迭代复用，占比小） |
| tap point / 免费 query 特征 | trunk 输出 single repr s_i（N×384）与 pair repr z_ij；池化后免费 | 不存在；要自己算状态嵌入（EOF/池化）| 不存在 | 不存在（可用 k-space 低频 / 上一帧） | 问题参数 θ（本来就有） |
| 迭代头 (Δ_j) | 200 步 EDM 采样，×N samples；25.92 s | 39 NFE / 12 h 步 | K=1..8 精炼步（PDE-Refiner）/ R=20–100（ACDM） | 1000 步 DDPM（[arXiv 2501.09305](https://arxiv.org/html/2501.09305)，6 min/volume） | 求解器迭代（OSQP 数百次） |
| 中间状态 (warm start 起点) | 噪声水平 σ 处的原子坐标；先例 = RFdiffusion "partial diffusion"（对已有骨架加几步噪再去噪，[RFdiffusion README](https://github.com/RosettaCommons/RFdiffusion/blob/main/README.md)） | τ' 处的带噪场；先例 = Warm-Start Diffusion（[arXiv 2507.09212](https://arxiv.org/pdf/2507.09212)） | 上一帧预测加噪（PDE-Refiner 本身就是这个） | 上一帧重建加噪到 τ'（3D 超声先例 [arXiv 2505.22090](https://arxiv.org/pdf/2505.22090)） | 上一时刻解移位（shifted warm start，标准做法） |
| tier K = hit | 回放库里的坐标（**同一复合物才有意义**；换配体原子集就变，无法回放） | 回放历史场（= analog forecast） | 回放历史场 | 回放上一帧 | 回放上一解 |
| key 字段 | 序列/配体 SMILES 精确匹配为 IVF cell；cell 内 = trunk single repr 池化 cosine + 配体指纹 Tanimoto | 状态 EOF 投影 | 场的低频谱 | 上一帧图像 / k-space | θ 欧氏距离 |
| 分数 s_t | tanh 标准化 + 加权和（同式） | 同 | 同 | 同 | 同 |
| 偏差 D_a | tier a 产出坐标与从纯噪声 200 步产出坐标的 RMSD（或 1−LDDT）；**离线可算且就是终测量本身** | tier a 的 12 h 场 vs 全算场的 RMSE / CRPS | 同 | PSNR 差 | 次优度 / 约束违反 |
| 成功度量 (SR) | 配体 RMSD<2 Å & PB-valid；DockQ；LDDT | CRPS / ensemble-mean RMSE | 相关系数保持时长 / 稳定 rollout | PSNR / SSIM | 约束满足 + 成本 |
| 成本模型 | c_a = 1.53 + 25.92·ρ_a（s，A6000，196 token）；随 token 数超线性 | 39 NFE·ρ_a | K·ρ_a | T·ρ_a | 迭代数 |
| 闭环性 | **无**（单次预测，无后续状态依赖） | **有**（30 步自回归） | **有**（长 rollout 漂移是核心问题） | 弱（逐帧独立评估；实时流才有） | **有**（receding horizon） |

---

## 3. 打分表 A–G（0–3）

主行 = Boltz-2 / AF3 系；其余子领域按行列出便于横比。

| 项 | Boltz-2 / AF3 系 | GenCast 天气 | 神经 PDE | 动态 MRI/CT | MPC / warm-start | DiffDock |
|---|---|---|---|---|---|---|
| A 结构匹配 | **3** — trunk→diffusion 两段式，tap point 天然（[Boltz docs](https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md)；[AF3 workload](https://cseweb.ucsd.edu/~jzhao/files/kim-iiswc2025.pdf)） | 1 — 无共享前缀，条件在 denoiser 内（[GenCast](https://ar5iv.labs.arxiv.org/html/2312.15796)） | 0 — 纯头 | 0 — 纯头 | 2 — 分解可复用但占比小（[arXiv 2309.07835](https://arxiv.org/html/2309.07835)） | 1 — 前缀"negligible"（[DiffDock](https://arxiv.org/pdf/2210.01776)） |
| B 成本余量 | **1** — c_pre/c_0≈5.5% 很好（[TerraBind](https://arxiv.org/pdf/2602.07735)），但傻基线 200→2 步几乎无损（[Protenix-Mini](https://arxiv.org/abs/2507.11839)；[Boltz-1 ≥50 步平台](https://www.biorxiv.org/content/10.1101/2024.11.19.624167v1.full)），warm 档无 counterfactual | 2 — 天真 1-NFE 掉点（[Tyche Tab.3](https://arxiv.org/html/2605.06916)），但蒸馏已解决 | 3 — 步数越多越稳，K=1→8 单调改善（[PDE-Refiner](https://papers.neurips.cc/paper_files/paper/2023/file/d529b943af3dba734f8a7d49efcb6d09-Paper-Conference.pdf)；[ACDM](https://arxiv.org/html/2309.01745)） | 1 — 有先验时少步可行（[Warm-Start Diffusion 4–6 NFE](https://arxiv.org/pdf/2507.09212)） | 0 — RTI 一次迭代就是工业标准 | 1 — 10 步近乎全性能 |
| C 局部性 / 命中 | **1** — 同靶点批量筛选有近邻配体（库常按 Tc 0.5 聚类去冗余，[Nat Commun 2025](https://www.nature.com/articles/s41467-025-57136-7)），但换配体不能回放；精确重复只在重跑；命中率**不确定** | 0 — 全球尺度自然 analog 需 ~10^30 年库（[van den Dool 1994](https://tellusjournal.org/articles/10.3402/tellusa.v46i3.15481)） | 1 — 相邻时间步相似但回放=persistence，错 | 3 — 相邻帧/切片极相似（[3D 超声](https://arxiv.org/pdf/2505.22090)） | 3 — 相邻 MPC 问题几乎同构 | 2 — 同靶点相似配体 |
| D 容忍度与闭环 | **1** — SR 度量齐全（RMSD<2 Å/DockQ/LDDT），但单次预测无闭环、无漂移，故事退化成离线近似计算 | 3 — 真闭环、CRPS、误差累积是主议题 | 3 — 同 | 1 — 逐帧独立 | 3 — receding horizon 真闭环 | 1 — 单次 |
| E RIT 可用性 | **2** — D_a 离线可算且单调性可信；但没有闭环，D_a 就是终测量，RIT 退化为普通离线验证；无"命中成串"（无时序） | 1 — 有成串（时间连续）但无命中 | 2 | 2 — 有成串 | 2 — 有成串 | 2 |
| F novelty 地形 | **0** — few-step 无重训（Protenix-Mini）、蒸馏（[DeCAF](https://arxiv.org/abs/2606.08375)）、partial diffusion（RFdiffusion）、template（Boltz）、MSA/feature 缓存（[AF_Cache](https://arxiv.org/pdf/2606.04566)）、跳过扩散头（[Boltzina 7.3×](https://arxiv.org/html/2508.17555)、TerraBind 26.6×）全占位 | 0 — Warm-Start Diffusion、残差化（[ArchesWeatherGen](https://pmc.ncbi.nlm.nih.gov/articles/PMC13101865/)）、Swift/Tyche 一步 | 0 — PDE-Refiner 本身即 warm-start 精炼 | 1 — 逐帧 warm start 已有但少系统化 | 0 — shifted warm start / RTI / 学习式 warm start 3.75× 胜最近邻（[Sambharya](https://arxiv.org/html/2309.07835)） | 1 — template/MCS docking 老方法 |
| G 我们做得动 | **2** — 开源、H100 可跑、~20 s/复合物；500 复合物 × 一个前沿点 ≈ 3 h；但 MSA 服务依赖外网 mmseqs2 | 0 — TPU/JAX、8 min/成员、ensemble 评测吃算力 | 2 — 小模型可跑 | 2 — fastMRI 开源 | 3 — CPU 即可 | 3 — 小模型 |
| **合计 (21 满)** | **10** | 7 | 9 | 10 | 13 | 11 |

MPC 与 MRI 合计分高但 A/F 为 0：那里的"cache"就是领域几十年的默认做法，没有论文。神经 PDE 的 B/D 满分很诱人，但 A=0（没有前缀可省，头就是全部）且 PDE-Refiner 已经是"从上一帧 warm start 再精炼"的原型。

---

## 4. 该领域的"减迭代次数"傻基线

| 系统 | 有人做过吗 | 结果 |
|---|---|---|
| AF3 系（Protenix 全模型） | 有，不重训 | 默认 AF3 采样器 <10 步崩（产出破碎结构），**但只是采样器设置问题**：改 η=1.0、γ0=0 后 2 步 ODE 配体界面 LDDT 0.645 vs 200 步 0.650，1 步 0.64（[Protenix-Mini §"few-step"](https://arxiv.org/abs/2507.11839)）。RMSD<2 Å 成功率、DockQ 在纯减步下的数字论文未单列（**不确定**；Mini 的 72.7% vs 80.0% 混杂了架构裁剪） |
| Boltz-1 | 有（作者消融） | "generally monotonic improvement … relatively plateaued beyond 3 recycling and 50 diffusion steps"（[Boltz-1](https://www.biorxiv.org/content/10.1101/2024.11.19.624167v1.full)）⇒ 200→50 免费 4× |
| DiffDock | 有 | 20→10 步近乎全性能，2× 提速；再低不确定 |
| GenCast 类 | 有（间接） | 标准 EDM 1-NFE RMSE 0.145 vs 一步蒸馏 0.103（Tyche Tab.3）：**天真减步掉点，但社区的答案是蒸馏/一致性模型，不是检索** |
| PDE-Refiner / ACDM | 有 | 步数越多越准越稳（K=1→8；R=20→100 对 Iso 湍流仍在改善）：**天真减步掉点** — 这是唯一 B 得 3 分的子领域，但没有前缀可省 |
| MPC | 有（RTI） | 每步只做一次 SQP 迭代从移位解出发是实时 MPC 标准 |

与 VLA 的对照：briefing §2 说 LIBERO 上 k=1 Euler ≈ 条件均值 ≈ teacher。AF3 系更极端：**输出是单个确定结构（单峰）+ confidence 排序**，所以少步 ODE 给出的"均值结构"正好是要的东西；只有需要多样性的场景（best-of-N 选 pose、构象 ensemble）才可能有多峰崩塌，而这正是 Protenix-Mini 没单独报告的那个数字。

---

## 5. 预期收益（数字推算）

以 TerraBind 实测的 Boltz-2 成本（A6000、196 token、10 samples）：c_pre = 1.53 s，头 = 25.92 s（200 步），c_0 ≈ 27.8 s（含小头）。

| 方案 | 每决策成本 | IR |
|---|---|---|
| 全算（200 步） | 27.8 s | 100% |
| cache 100% hit（回放坐标） | 1.53 s | **5.5%**（地板） |
| 傻基线 50 步（Boltz-1 平台点） | 1.53 + 6.48 = 8.0 s | 29% |
| 傻基线 10 步 | 1.53 + 1.30 = 2.8 s | 10% |
| 傻基线 2 步 ODE（Protenix 证据） | 1.53 + 0.26 = 1.8 s | **6.4%** |
| cache 命中率 h、未命中走 2 步基线 | 1.53 + 0.26(1−h) | 5.5%–6.4% |

即 **cache 相对已有傻基线的最大理论收益 < 1 pp IR**，且要付出 (a) 库的存储与检索、(b) 换配体不能回放的硬限制、(c) 精度上还得赌 2 步 ODE 的成功率不掉。对比 VLA：VLA 减步地板 60.6%/40.7%、cache 地板 15% ⇒ 差距 25–45 pp，cache 尚有 IR<40 的独占区间；AF3 系这个独占区间宽度 <1 pp。

天气：命中率 ≈ 0（analog 论证），warm start 从上一步状态出发已被 Warm-Start Diffusion / ArchesWeatherGen 残差化覆盖，cache 无增量。PDE：同上。MRI：逐帧 warm start 能省 78–95% 计算（DIP 参数映射的 warm-start 数据，[PMC12501688](https://pmc.ncbi.nlm.nih.gov/articles/PMC12501688/)），但那是"上一帧 = 唯一候选"的退化 cache，不需要检索/IVF/RIT。

**唯一窄缝**：同靶点大批量筛选（10^4–10^6 配体）+ 需要多样 pose 的 best-of-N 排序。若实验证明 2 步 ODE 在 RMSD<2 Å 成功率或 PB-valid 上确有掉点（多峰崩塌），那么"从相似配体的已知 pose 做 partial-noise warm start、只跑尾部 ρ 步"可能同时赢成本和精度。这是 template/MCS docking 思路在扩散头上的再现，novelty 有限但可测。

---

## 6. novelty 地形与最强反驳

已有工作（按贴近度）：
1. **few-step 无重训采样**：Protenix-Mini 2 步 ODE（[2507.11839](https://arxiv.org/abs/2507.11839)）；Boltz-1 50 步平台；Protenix-Mini+（[2510.12842](https://arxiv.org/pdf/2510.12842)）。
2. **蒸馏 / flow map**：DeCAF-Boltz 5× NFE 减少且 RMSD 反而改善（[2606.08375](https://arxiv.org/abs/2606.08375)）；DCFold 单前向（[2605.17899](https://arxiv.org/pdf/2605.17899)）；天气 Swift / Tyche。
3. **头级 warm start 先例**：RFdiffusion partial diffusion（设计域）；Warm-Start Diffusion（通用条件扩散，4–6 NFE）；医学影像逐帧 τ' 起步。
4. **跳过扩散头做筛选**：Boltzina 7.3×（[2508.17555](https://arxiv.org/html/2508.17555)）、TerraBind 26.6×。
5. **跨样本缓存**：AF_Cache 缓存 MSA/features（[2606.04566](https://arxiv.org/pdf/2606.04566)）；ColabFold MSA 复用；一般扩散模型的 DeepCache / Learning-to-Cache / ToCa 是**跨去噪步**的特征缓存（[2406.01733](https://arxiv.org/abs/2406.01733)，[2410.05317](https://arxiv.org/html/2410.05317v3)），与我们跨 query 的经验缓存正交。
6. **优化域**：learned warm start 胜最近邻 warm start 3.75×（quadcopter MPC，[2309.07835](https://arxiv.org/html/2309.07835)）—— 这条直接说明"最近邻检索 + 剩余迭代"在该域已被更强方法压过。

最强反驳（审稿人会说的）：
- "你的 hit 地板 5.5%，我 2 步 ODE 6.4%、零库零检索、还能处理没见过的配体。差 0.9 pp。"
- "单次结构预测没有闭环，你的 recoverable redundancy 定义 (arxivd.tex §2 的 composition 论证) 在这里无物可指；D_a 就是终测量，RIT 只是把验证集分位数画成阶梯。"
- "跨配体 warm start 就是 template docking + partial diffusion，二者都有名字了。"
- "天气：analog 命中率是 1994 年就算清的负结果。"

---

## 7. 首个实验方案（≤ 2 周，仅当 owner 想赌 §5 的窄缝时）

平台：Boltz-2（h100，单卡），benchmark = PoseBusters（428 复合物，公开）+ 一个同靶点 congeneric 系列（PLINDER 或 FEP 基准集，用于命中率）。

- **第 1–3 天：成本拆分实测 + 傻基线**（第一步永远是这个）。用 `--sampling_steps ∈ {200,50,20,10,5,2}` × `--diffusion_samples ∈ {1,5}`，逐阶段计时（trunk / diffusion / confidence），记录 RMSD<2 Å、PB-valid、pLDDT。判据：若 2–5 步在 RMSD<2 Å 成功率上与 200 步差 ≤ 2 pp，**当场停止**（cache 无空间，写 NO）。若差 >5 pp（多峰崩塌实锤），进入下一步。
- **第 4–7 天：offline shadow**。对 congeneric 系列，key = trunk single repr 池化 + 配体 Morgan 指纹；候选 = 同靶点库里 Tanimoto 最近邻的 pose；tier a = 把候选 pose（MCS 对齐后）加噪到 σ_a 再跑剩余步；算 D_a = 对 200 步参考的 RMSD；画 q_a(s)。这里没有闭环，shadow 就是全部。
- **第 8–10 天：前沿**。δ 扫描 → (IR, 成功率) 曲线，对照傻基线曲线。每个点 ≈ 428 × 20 s ≈ 2.4 h（H100 更快）。
- 不做：天气（算力与命中率双重不可行）、PDE（无前缀）、MPC（无 novelty）。

---

## 8. 结论

**NO**。AF3 系 cofolding 结构匹配完美、成本余量比 VLA 大一个量级（trunk 5.5% / 头 93%），但不重训的 2 步 ODE 已把头成本压到与 100% 命中相当（IR 6.4% vs 5.5%）且精度几乎无损，warm 档没有 counterfactual；单次预测无闭环，recoverable redundancy 与 RIT 的核心论证无处落脚；天气有闭环却没有前缀也没有 analog 命中；PDE/MRI/MPC 的 warm start 是各自领域的既有默认。唯一 MAYBE 窄缝是"同靶点批量筛选 + best-of-N 多样性"，先用 3 天傻基线实测决定要不要碰。
