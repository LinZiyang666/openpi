# 方向 A 展开：experience cache for imagination-in-the-loop（决策回路里的世界模型）

> 2026-09-15 草案，供 owner 裁决；数据来源 `world_models_planning.md`、`streaming_media_world_models.md`。未跑任何实验。

## 1. 一句话故事

VLA 里 cache 只能省掉 15% 地板以上的那部分（前缀必须跑），而且减步一句话就能追平。把 cache 从"策略推理"搬到"决策时想象"：每个控制步要在世界模型里想象 N 条未来再挑动作，算力大头是想象；hit 档跳过整个想象回路，地板 = 编码器 ≈ 0（FLOPs 口径）。可恢复冗余的量级从"十几个百分点"变成"一个数量级"。

## 2. 映射（我们的符号 → 想象回路）

| 我们 | 潜空间 MPC（TD-MPC2） | 视频世界模型 best-of-N（Cosmos Policy planning） |
|---|---|---|
| 共享前缀 c_pre | encoder o_t → z_t | 视频 tokenizer + 前缀 |
| 头（迭代生成） | MPPI：I 次迭代 × N 条样本 × H 步 rollout + Q/reward（TD-MPC2 默认 6×512×3 = 9216 次评估/步） | 采 N 个候选 chunk → 各自想象未来 → 价值挑最好（N=8，8×H100 并行 4.9 s/chunk） |
| 中间状态（warm 起点） | 第 i 次迭代后的 (μ, σ)_{t:t+H} | 候选 chunk 的部分去噪 latent；缓存的想象未来 |
| tier 0 miss | 全规划 | 全 best-of-N |
| warm tiers | 用库条目的 plan 初始化 (μ,σ)，只跑 ρ_a·I 次迭代 / ρ_a·N 条样本 | 从缓存 latent 继续去噪剩余步；或复用缓存想象只重算价值 |
| tier K hit | 直接执行库条目的动作（chunk），零 rollout | 直接执行库条目挑出的 chunk |
| key | z_t（=相机字段）、本体、任务/目标 embedding（= 指令，做 IVF cell） | 同左 + 指令 embedding |
| 偏差 D_a（shadow） | tier-a 首动作/plan 与全规划的标准化偏差；**新增本领域独有的代理：价值遗憾 Q(z_t, plan_full) − Q(z_t, plan_a)**，世界模型 + 价值函数离线可算 | 同左（价值头） |
| 成功度量 | Meta-World SR / DMControl 回报 | LIBERO / RoboCasa SR（我们的池） |
| 闭环漂移 | 缓存 plan 开环执行会过期（AdaReP/AOR 的核心问题） | 同 |
| 状态门 | 命中成串；对手 = AdaReP 的偏差触发重规划 | 同 |

## 3. 这个领域的傻基线（必须全部画进图）

1. **policy-only**：TD-MPC2 的策略先验 π 直出，成本 ≈ c_pre —— 这是本领域的"k=1"，**和 hit 档同价**。cache 只在"规划增益 SR_plan − SR_policy 大"的任务上有意义；第 0 步就是逐任务量这个增益。
2. **减预算规划**：I / N / H 各自砍到最小（推理侧砍一半不掉点是已知的；砍到 policy-only 在多任务/高维掉）；Dream-MPC 用梯度规划 15 次评估追平 9216 次——会缩小 miss 成本。
3. **自身复用（无库）**：TD-MPC2 默认已 shift-warm-start 上一解；MPC^m 每 m 步重规划其余开环；AdaReP（2026-06）偏差触发重规划，DMControl 砍 54.5% NFE 不掉分、Franka 砍 >80% 查询。**我们必须证明跨 episode 库比它再多买 ≥10 pp IR**，增量应集中在 episode 起点、任务切换、扰动后——自身复用在那里没东西可复用。

## 4. 还剩什么是我们的

- 跨 episode / 跨任务 / 跨 agent 的库（AdaReP 只复用自己当前 plan）；
- 多档同时存在（他们二元）+ 一个 δ 从 shadow 偏差切所有 cut + 按预算反解（他们在线调超参）；
- 用**价值遗憾**做 RIT 的风险代理——比 VLA 的动作偏差更接近成功损失，是方法上的升级点；
- 闭环 (IR, SR) 前沿与可恢复冗余定义搬到想象回路。

## 5. 床

| 床 | 角色 | 成本 | 状态 |
|---|---|---|---|
| TD-MPC2 × Meta-World（单任务 + MT 检查点） | 受控实验：量库 vs 自身复用、RIT 单调性、gate | 20 ms/步，500 集/点 ≈ 35 min（4090） | 开源 + ckpt |
| Cosmos Policy planning 模式 × LIBERO A 池 / RoboCasa 13 任务 | 主秀：贵、新、未蒸馏、我们自己的 harness | 1/5/10 步 0.16/0.61/0.95 s；N=8 串行 ≈ 5 s/chunk；500 集/点 ≈ 20 h（单卡估计） | 权重开源（LIBERO ckpt 在 HF）；**planning 相对直出的增益未知** |
| DINO-WM（可选） | 第二个便宜床，像素输入 | 不确定 | 待查 |
| V-JEPA 2-AC | 只作成本论证（16 s/action） | 真机 | 不跑 |

## 6. 两周探针（kill 条件明确）

**并行线 1：TD-MPC2（4090）**
- D1–3：成本拆分（encoder / 规划 wall-clock 与 FLOPs；hit 地板）；逐任务规划增益（plan vs policy-only）；减预算 ladder（I, N, H）；自身复用 ladder（MPC^m、偏差触发）。**Kill**：所有任务规划增益 < 5 pp，或自身复用在 ε≈0 下已到 IR ≤ 0.3。
- D4–7：建库（成功 episode 的 z_t、plan、想象未来、价值）；shadow rows（agent 自跑，事后离线检索）；D_a 两种（动作偏差、价值遗憾）；q_a(s) 单调性；gate AUROC（成串）。**Kill**：q_a(s) 不单调或 AUROC < 0.7。
- D8–12：RIT ladder K=1,2,3，扫 δ 出闭环前沿，叠三条傻基线。**GO**：在有规划增益的任务上，库前沿比自身复用在 ε ≤ 2 pp 处多压 ≥10 pp IR，且 wall-clock hit 地板 < 20%。

**并行线 2：Cosmos Policy 第 0 步（h100）**
- D1–5：复现 policy-only vs planning（libero_10 A 池 100 集 + RoboCasa 2–3 个任务）；实测 tokenizer/前缀 vs 每候选想象 vs 价值的毫秒拆分；1/5/10 步减步基线。**Kill**：LIBERO 上规划增益 ≈ 0 且 RoboCasa 也 ≈ 0 → 主秀换床（open-oasis 型）或放弃 A。
- 若过：D6–14 在 RoboCasa（67.1%，有掉点空间）建库 + shadow，出 K=2 前沿 4 个点（≈ 80 单卡小时，h100 + 4090 分摊）。

## 7. 风险

- "还是机器人"：只能靠"cache 的对象从策略换成想象回路、收益量级不同"来辩；B 方向的 DIAMOND 三天探针可作旁证。
- AdaReP / AOR / DreamLedger 占了 hit+gate 的直观版；我们的差异必须在第一张图里量出来。
- 小模型 wall-clock launch-bound：FLOPs 与 wall-clock 两套口径都报，部署故事按批处理/大模型讲。
- Cosmos Policy 的 planning 增益若为零，主秀塌；第 0 步先做。
