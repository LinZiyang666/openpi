# Latent-MPC / 世界模型规划线：四路调研汇总（2026-09-16 深夜）

> 四份报告：`latent_mpc_jepa_family.md`（JEPA/latent-MPC）、`latent_mpc_video_wm.md`（视频/扩散世界模型在回路里）、`latent_mpc_diffusion_planners.md`（扩散规划器/推理时搜索）、`latent_mpc_benchmarks_ckpts.md`（benchmark × 现成权重）。本文只做排序与裁决建议；数字与 URL 以各报告为准。
> 前提（owner 23:1x）：Cosmos planning 模式暂不跑；要"latency/算力贵 + 我们的 cache 能接上 + 最好是机器人 + 有现成 benchmark 微调版"。

## 1. 排序（贵 × 可接 × 机器人 × 可行 × novelty 余量）

| # | 候选 | 每决策代价 | 减迭代会不会崩（warm 档 counterfactual） | benchmark / 权重 | 可行性 | 主要风险 |
|---|---|---|---|---|---|---|
| 1 | **X-WAM × RoboCasa-2024**（Wan2.2-TI2V-5B world-action model，Apache-2.0） | 3090 上 1.0 s（动作早停）/2.7 s 全跑/9.7 s N=8 BoN | **会**：WAM 头减步崩（DreamZero 4 步 83%→1 步 52%；Flash-WAM 不蒸馏减步 66.7→40.0） | RoboCasa-2024 24 任务 79.2%，BoN 80.8→82.5；官方 ckpt | **高**：我们 RoboCasa-2024 环境今晚刚搭好，4090 放得下（18–24 GB）；与 Cosmos Policy 同床可直接对照 | Fast-WAM"测试时跳过视频"傻基线（97.6 vs 98.5、190 ms vs 810 ms）；C³ache/FFDC 已做跨 chunk 复用 |
| 2 | **jepa-wms RoboCasa / Metaworld**（Meta TMLR 2026-05：DINO-WM + V-JEPA-2-AC(fixed) 同环境 ckpt，CC-BY-NC） | **~47 s/plan**（CEM 300×30×H5），hit 地板 ≈0.02% | 待测；DA-LeWM 提示 CEM 后期迭代≈噪声，减迭代基线可能很强 | RoboCasa Reach 16 / Place 33，Metaworld-Reach 58 | 中：一天跑通，但 50 集一点 1.5–3 h，SR 余量小 | GC-IDM 学习型 k=0 便宜 100× 且 7/8 设定 ≥ CEM；IMWM 检索初始化 CEM 是最近先例 |
| 3 | **LingBot-VA × LIBERO-Long**（RSS'26，5.3B，LeRobot 已集成） | 2.3–6.8 s/决策，N=8 3.9 s | 同 WAM 家族 | LIBERO-Long 97.2→98.3 | 高（4090） | LIBERO 天花板，增益只有 1 pp |
| 4 | **LeWM × PushT/Cube**（15M，HF 权重） | 1–54 s/plan 看实现 | 待测 | 非机器人操作（玩具） | 最高：pip 装即得 CEM/MPPI，50 集 5–58 min | 只能当受控台 |
| 5 | **Diffusion Policy DDPM-100 × robomimic** | 0.7–0.8 s | **会**（DDIM-1 全 0、DDIM-2 0.64 vs 0.94） | 官方 ckpt | 高 | warm-start 竞品密集（RTI-DP/Falcon/STEP/WarmPrior） |
| 6 | MCTD / Fast-MCTD × OGBench cube | 3–100 s/重规划 | 会（不搜索 12% vs 78%） | 代码开源，权重不确定 | 中 | 单卡耗时未知 |
| — | World-in-World（RLBench，WM 回路 24→45%）、VLA verifier（RoboMonkey）、DreamZero 14B | — | — | 权重不确定 / hit 地板高 / 2 GPU | 低 | — |

## 2. 裁决建议

- **主床：X-WAM × RoboCasa-2024**。理由：贵（秒级）、头减步崩（warm 档有 counterfactual，与 VLA 的 k=1 无损正相反）、机器人操作、权重 + benchmark 齐、我们的环境现成、还能和 Cosmos Policy 在同一张床上比。
- **受控台：LeWM**（三天杀手门最便宜），通过后升级到 **jepa-wms RoboCasa**（贵到 47 s/plan 的 latent MPC 展示）。
- **保底证据：Diffusion Policy DDPM-100**，只用来证明"跨 episode 库 > 自身上一 chunk 热启动"，不成篇。

## 3. novelty 地形（必须正面处理的近亲）

IMWM / EV-WM（检索初始化规划）、CheckVLA（离线 shadow + conformal 分位阈值 ≈ RIT）、C³ache（跨 chunk 残差缓存）、FFDC（自适应前向）、Fast-WAM（测试时跳过视频）、GC-IDM（学习型 k=0）、Falcon/RTI-DP/STEP（自身 warm start）。剩下的新东西 = **跨 episode 库 + 从检索到的 latent/中间态热启动 + 多档单 δ 反解预算 + 闭环漂移门**；每篇都要用同一床把这些基线画进同一张图。

## 4. 杀手门（各报告的三天设计已写，主床版本）

X-WAM × RoboCasa-2024，4090：Day1 成本拆分 + 动作头步数阶梯（5→3→2→1）各 100 集，Kill-1 = 2 步不掉 ≥3 pp；Day2 自身上一 chunk 热启动 + Fast-WAM 式跳视频基线，Kill-2 = 任一到达同 SR 且成本 ≤ 我们 warm 档；Day3 库 + shadow 50 集算 D_hit/D_warm、q_a(s) 单调、K=2 闭环 4 个 δ 点，Kill-3 = 同 SR 下 IR 领先 < 10 pp。
