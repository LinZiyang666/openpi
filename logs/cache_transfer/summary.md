# 跨领域迁移调研汇总（2026-09-15，7 路 fable agent，各自报告在本目录）

> 问题：tiered experience cache + RIT + stateful gate 能否迁到 VLA 以外、且**收益明显**的领域。简报与打分口径见 `briefing.md`（A 结构 / B 成本余量含"减迭代傻基线会不会掉点" / C 局部性 / D 容忍度与闭环 / E RIT 可用 / F novelty / G 做得动，每项 0–3）。

## 1. 总表

| 路 | 报告 | 结论 | 合计/21 | 一票否决项 | 剩给我们的空白 |
|---|---|---|---|---|---|
| 自动驾驶/导航扩散规划 | `driving_planning.md` | **NO** | 14 | B=1：anchor/goal 词表已吞掉多峰，1–2 步=全步（DiffusionDrive 87.9/88.1、GoalFlow 88.9/90.3、nuPlan 2 步≥10 步）；E2E 前缀占 66%，hit 最多省 34%；F=1（RealDrive、词表规划、MANTRA） | 无 |
| 流式媒体 / 交互式世界模型 | `streaming_media_world_models.md` | **MAYBE → 条件 GO 于 DIAMOND/GameNGen 型（世界模型当模拟器、agent 在环）**；Wan 系实时视频 NO | 17 | 重训 1 步学生几乎追平（ASD、GameNGen 蒸馏）；帧级命中率文献空白；Wan 系 hit 地板 ~44% | 流式逐帧、动作为 key、跨样本 tiers+RIT+gate、闭环评测——文献空白；漂移是该领域公认头号病 |
| LLM/VLM 服务与 agent | `llm_serving_agents.md` | **MAYBE → 仅 diffusion-LM agent（LLaDA-UI）GO**；AR 世界 NO | ~13 | AR 无中间态，warm 档=带验证的投机解码地盘；hit-or-miss 标定已被 vCache（ICLR 2026）做掉；NoThinking 低预算反而更好 | dLLM 的检索式 warm start 空白（减步真掉点：32→16 步 42.7→33.0%） |
| 语音/音频 | `speech_audio.md` | **NO** | 12 | 延迟口径 LLM 前缀占 64–70%（hit 只省 1/3）；蒸馏 1–3 NFE 追平；无成功事件、无闭环；NIRVANA/Chorus 同构已发表 | 无 |
| 科学/工程求解 | `scientific_solvers.md` | **NO** | 10（AF3 系） | 不重训 200→2 步无损（Protenix 0.650→0.645），2 步 IR 6.4% vs 满命中地板 5.5%；单次预测无闭环 | 同靶点批量筛选 + best-of-N 窄缝，3 天止损 |
| 感知 / 动画 / 非 VLA 具身 | `perception_animation_embodied.md` | (a) 动画 **NO−**；(b) 感知 **NO**；(c) 具身 **MAYBE** | 15 / 11 / 15 | (a) 整模 13 ms 算力非痛点、无 SR；(b) 前缀占 70–80%、逐帧复用最拥挤；(c) 不算换领域，对手是 RTI-DP/Falcon 自身 warm start | (c) 唯一有"减步会崩"硬证据（DDPM DP 1 步 0%、DiffuseLoco DDIM 挂）+ 漂移强 + hit 地板≈0 |
| 决策回路里的世界模型 | `world_models_planning.md` | **MAYBE 偏 GO（TD-MPC2 / Meta-World 子族）** | 15 | AdaReP（2026-06）已做 training-free 缓存 rollout + 偏差触发重规划（DMControl 砍 54.5% NFE 不掉分）；wall-clock launch-bound，hit 地板未实测 | 跨 episode 库 + 多档 + 单 δ 离线标定 + 按预算反解；FLOPs 上 c_pre/c_0≈1/9216 |

## 2. 跨路结论

1. **傻基线到处杀人**：每个领域的第一对手都是"从头算的廉价近似"——不重训减步（驾驶、AF3、LIBERO）、蒸馏 1 步学生（视频、语音、动画）、自身上一输出 warm start（驾驶、感知、具身、世界模型）。warm 档只在 **多峰 ∧ 未蒸馏 ∧ 减步真掉点** 的交集里有 counterfactual；具身那路指出参数化也算：DDPM/DDIM 训练的头一步崩，flow-matching 头一步无损（与我们 LIBERO 数字一致；ActionCache 的 VLABench 崩是 flow-matching × 多峰）。
2. **hit 档的价值 = c_pre/c_0 地板**。VLA 的 15% 反而属于好的（语音 64–70%、感知 70–80%、驾驶 66%、Wan 视频 44%）；地板真正低的只有世界模型规划（FLOPs≈0，但 wall-clock 未测）、像素空间世界模型（≈0）、AF3（5.5%，可惜 2 步无损）。
3. **闭环漂移叙事只在控制 / 世界模型 / agent 里成立**；语音、科学、感知都是逐样本离线可算，RIT 退化成普通离线验证。
4. **novelty 剩余物是同一件东西**：hit+gate 与"上一输出 warm start"在所有领域都有人做；我们还独有的是 **跨 episode 库 + 多档 + 一个 δ 从 shadow 偏差切所有 cut + 按预算反解 + 闭环 (IR,SR) 前沿定义**。审稿人在每个领域都会问"比自身上一输出复用多买了什么"。

## 3. 建议

- **值得两周探针的只有一族：世界模型当决策回路里的模拟器**，两个入口互补：
  - DIAMOND on Atari 100k（媒体路方案）：像素空间 hit 地板≈0，真模拟器做 oracle，Breakout（确定）vs Boxing（多峰），未重训 1 步在多峰游戏糊成均值；闭环指标 = agent 回报；一个前沿点分钟级。
  - TD-MPC2 on Meta-World（规划路方案）：每步 9216 次 rollout 评估，FLOPs 地板≈0；必须先实测 wall-clock hit 地板，并以 AdaReP 式"自身 plan 复用"为对照，库前沿再压不到 ≥10–15 pp IR 就停。
- 外卡：diffusion-LM agent（LLaDA-UI / LLaDA-8B + ALFWorld 替身），检索式 warm start 在 dLLM 里是空白且减步真掉点；但 GUI 评测贵、vCache 站在 hit-or-miss 那头。
- 时间约束：ICLR 2027 截稿在即，以上都是**下一篇**；本篇只能靠 RoboCasa 把"flow-matching × 多峰会不会崩"跑出来 + 减步与 cache 叠加的联合前沿。
