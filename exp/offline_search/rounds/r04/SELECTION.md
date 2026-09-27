# R4 选题（2026-09-27 14:0x CDT；协调者裁定；只用 codex agent）

输入：`ideation_A/REPORT.md`（看一眼，做几步）、`ideation_B/REPORT.md`（降低单次成本）、`ideation_C/REPORT.md`（第一遍：时间偏移检索，分层配对 pilot）、`ideation_C/REPORT_2.md`（第二遍：随机化识别"一次调用的价值"）。四份都是 codex（gpt-6-astra，xhigh）写的，每个结论都附有脚本和数字。

## 裁定依据
1. **成本主要花在哪**：
   - 当前最优点 l10 g500，IR .238 中有 64% 来自每步都要跑的视觉编码器（stage 1）。所以 R4 主攻两件事：让一部分决策不跑视觉（A），让视觉本身变便宜（B3、B4）。
   - MISS 只减步能省的有限：π0.5 的 K2 纯推理与 K10 持平（.996 / .848），但减步只能省掉 stage 3，一次 MISS 最低也要 .65。
2. **A 的实测**（冻结路径加离线回放）：
   - 盲走时照抄 top-1 那条示范会丢掉整组邻居，l10 上 h=2 的误差 +.10。
   - 让 16 个邻居各自沿自己的示范前进，并用本体状态纠正相位，比按时间走误差少 .022 到 .037（bootstrap 区间不含 0）。
   - 最多盲走 2 次，外加相位门和状态门，g500 的 IR 从 .238 降到约 .17–.18（成功率未测）。
   - "执行视觉那一步动作块的剩余部分"在 spatial 上很有竞争力，是必备对照。
3. **B3 是精确优化**：
   - π0.5 每步都会对一路全零的假相机跑一遍完整视觉塔，stage 2 里还有 256 个整段被 mask 的 token。
   - 缓存假相机的嵌入、把被 mask 的 token 打包掉，语义完全不变，只需通过数值一致性检验。
4. **离线误差不能排序，100 集 pilot 在 l10 上噪声太大**。所以 R4 的每个判断都用 500 集配对全量；pilot 只用来排除崩溃，抽样改用 C 的分层清单。
5. **C 第二遍**：
   - 现有日志无法识别"一次调用的因果价值"，要判断就必须做随机化实验。
   - 否决了三样：通用的"失败 N 次就停"、跨任务检索、token 告警。
   - 随机化实验费用高（4 臂 × 500 集），预期收益小（IR 降 .002–.015），放到 R4 最后，视时间决定跑不跑。

## 方法清单（4 个 codex 编码 agent）

| 族 | 编码 agent | 内容 |
|---|---|---|
| **R4-1 盲走方法**（A 第 1–3 名 + C 第一遍的提案 1 + B2 里的方法部分） | K1（CPU） | 模块 `rounds/r04/k1_blind/`，包含：`phase_particles`（固定 16 个邻居及权重，各自按示范前进，用本体状态纠正相位，盲走预算 B ∈ {1,2}）；对照 `kernel_clock`、`top1_clock`、`anchor_tail`（执行视觉那一步动作块的剩余部分，分"只按预算回视觉"和"加相位门"两种）；回视觉的门（夹爪事件将至、接近末尾、两次低运动、位移残差 >.5、MISS 之后）；间隔感知的无进度守卫 `noprog_span`，以及 MixedJudge 的 memo-reset 变体；C 的 `control_step_library`（纯缓存下的时间偏移拼接，外加消融 G）。按"方法与插件的接口"一节实现 |
| **R4-2 盲走 serving 基础设施与成本口径**（A §3） | K2（GPU，full-access） | 插件在 `_ConnPolicy.infer` 做推理前的绕过：盲走决策完全不进 interceptor，stage 1 不跑。稠密历史带 has_vision 掩码；orchestrator 的状态历史和计数器每步只推进一次（在插件层实现，不改 src）；π0.5 和 GR00T 输入输出变换一致；动作只广播一次；日志记录 vision_used、来源、门、每次 HIT 实际执行的 5 步。另做一个"假 stage 1"：被调用就报错，用来证明盲走真的没跑视觉。KPI / collect 的 IR 口径改为 .152·v + (.410 + .438·K/10)·m，支持 K 和执行段长 L |
| **R4-3 成本引擎**（B1、B3、B4 的单相机部分） | K3（GPU，full-access） | ① MISS 减步：yaml 的 `miss.num_steps`，并在 emit_arms 里支持按臂修补 yaml；GR00T 的 K 由启动脚本暴露。② 精确消除冗余：缓存假相机的嵌入；可选把 stage 2 里被 mask 的 token 打包掉。都要做同输入、同噪声下的动作一致性检验，并在 4090 上实测 stage 1 / stage 2 的耗时比例。③ 单相机 key（只用腕部相机）：命中时只跑一路视觉塔，MISS 时补齐另一路；配套单相机 AWM 适配器，并在同一库上重拟合 |
| **R4-4 评测与前沿**（B2、baseline、C 第一遍的提案 2、C 第二遍的提案 1） | K4（CPU） | ① 臂类型：纯推理（judge 为 `periodic:1`，多种子）；只减步的纯推理；拉长执行段 L=10（每臂可覆盖 client 的 `--replan-steps`，IR 按每个控制步归一）。② C 的分层清单：run_gtp_subset 支持按 (task, ep) 清单挑集，EXPECT 跟着清单走，KPI 输出加权估计。③ 随机化 CALL / CACHE 的判决覆盖层，默认关闭，最后才跑。④ 低 IR 调度的臂规格。不碰 timan107，远端脚本由协调者推送并做 smoke |

## 方法与插件的接口（K1 和 K2 并行开发，按此约定对接）
- 盲走能力：`method.blind_step(bq) -> BlindResult | LookReason`。
  - `bq` 是只含状态的视图，字段有 step、task_id、`rs`（与 key builder 产出的 robot_state 逐位相同，由 K2 负责）、`raw_state`、`hist_a_exec`、`hist_hit`、`hist_has_vision`、`blind_age`、`prev_hit`、`episode`。
  - 返回 `LookReason`（需要视觉，附原因码）时，插件照常走视觉路径。返回 `BlindResult` 时，字段有 `action`（(H,32) 的归一化动作块）、`rows`、`weights`、`extras`（含 gate 位与相位），插件直接用它服务，不跑 stage 1。
  - 不支持盲走的方法视为永远需要视觉。
- 锚点由方法自己在视觉那一步的 `query()` 里记住（每集状态，`reset()` 清空）。`q.prev_hit is False`（刚 MISS 过）时锚点失效。
- 盲走决策不允许 MISS。要调用策略，必须先回到视觉，再由判决器决定。周期 MISS 按全局决策计数判断，在绕过之前先判。

## 闭环计划（π0.5 为主；每个判断都是 500 集配对全量；50 集和 500 集两种库规模都做）
- **第一批：不用新代码，立刻开跑，与编码并行。** 服务器按显存用 2–3 个完整模型 server。
  - 纯推理多种子：l10 与 sp 各 2 个新种子，加上 trace_dual 的 1 个。
  - 500 集库：l10 守卫 noprog 4；l10 周期 8、12；sp 只用守卫（缺的那个点）；sp 周期 12。
  - 50 集库：l10 周期 6；l10 守卫 noprog 4。
- **第二批（K3 交付后）**：MISS K2，覆盖 g500、g50、perk5、以及 sp 只用守卫；只减步的纯推理 baseline。
- **第三批（K1 + K2 交付后）**：盲走臂，覆盖 l10 两种库规模、l10 以外的其余格子，另加拉长执行段的 baseline。各臂如下：
  - B=0 的适配器对照；
  - phase B=1、phase B=2；
  - kernel_clock B=1；
  - anchor_tail B=1，门控与不门控各一；
  - 两时钟方案：盲走 + 周期 MISS，500 集用 k=8，50 集用 k=5。
  - 随后迁移到 sp；GR00T 只做纯缓存下的 anchor_tail 与 phase。
- **第四批**：叠加最优组合：盲走 + MISS K2 + 冗余消除 + 单相机 key。纯缓存的 control_step_library 在 l10 两种规模上各跑一次。
- **视时间追加**：随机化 CALL / CACHE（4 臂 × 500 集）。
- **报告要求**：
  - SR 对 IR 的前沿，含全部 baseline；
  - 四层拆分：合成 / 方法 / 库 / 控制，外加"成本实现"一层（冗余消除与减步）；
  - 两种库规模下的体积，与现役对照；
  - 每次视觉、每次 MISS 都按实际成本计价。
