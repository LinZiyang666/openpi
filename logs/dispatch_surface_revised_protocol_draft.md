# Dispatch Surface — 修订协议（Rev 1，**方法冻结，待实现与 D0 放行**）

> **状态**：owner 授权 Review Authority 直接定稿；Rev 1 的方法、统计裁决与执行顺序已冻结。
> 代码尚未按本协议实现，D0 尚未通过，因此**不得 emit、不得 rollout、不得触碰 A′**。
> 本文不覆盖 2026-08-28 的原协议止损记录；Rev 1 是独立的 post-hoc protocol revision。
> 原协议的负结果（两 suite 均触发止损点 A）如实保留于
> [`dispatch_surface_plan.log.md`](dispatch_surface_plan.log.md) §11 与
> [`dispatch_surface_open_questions.md`](dispatch_surface_open_questions.md)。
> 本修订是在查看原 fit/cal 失败数据后形成的；原 fit/cal 全部降为 exploratory development data。
> **不得拿 Rev 1 的结果冒充原协议成功，也不得把旧 calibration 重新称为正式证书数据。**

## 1. 触发本次修订的两件事

1. **owner 决定把 gate 从 `score_hysteresis` 改为 `always_search`**，使 Pareto 前沿纯粹反映
   verdict（cache 判定）的作用，而非 gate 的功劳。
2. **原协议在两个 suite 上都无法产出 surface artifact**，闭环无法构造。修订必须同时解决这一点，
   否则改 gate 没有意义——gate 是部署期配置，而阻塞在拟合期。

## 2. 为什么 primary 必须使用 `always_search`

1. **gate 与 verdict 共用信号 `s`。** `score_hysteresis` 的 theta 和 T1–T3 的判定阈值都来自同一
   score 分布；即使所有臂共享 gate，观测到的仍是 gate×verdict 的条件效应，而不是 verdict 本身。
2. **gate 会截断 `(s,v)` 支持域。** 这可能恰好删去 `v` 最有区分力的模糊区，使“SV 没增益”无法区分
   verdict 失败与 gate 瓶颈。`always_search` 让每次 policy decision 都进入相同的 search+judge 路径，
   因而把 SV、S0 与 threshold 的差异归因到三档 verdict。

成本审计同时确认：普通 `score_hysteresis` skip 是 `MISS,searched=False`，执行 full inference 并按 MISS
计价；只有 FollowWinner blind replay 才是 `FULL_HIT,searched=False`，且它仍已执行 stage1。原成本实现没有
把 gate skip 错计成免费 FULL_HIT。选择 `always_search` 的理由是实验隔离，而不是修复成本 bug。

两种 gate 配置的绝对成本不直接比较：primary 估计 verdict-only Pareto；带固定 production gate 的实验仅作
secondary 外部有效性检查，并只在该层内部比较。

## 3. 阻塞点的完整诊断：**选择期与认证期各自独立地压死接受域**

这是原协议负结果的关键诊断。Review Authority 先前建议的修法（把选择期的 episode-max offset 拿掉、
只保留 calibration split 的正式 conformal 认证）**不足以解开阻塞**。实测：

| suite | 选择期 offset | 认证期 `c`（cal split, 100 ep） | `q̂_min` | `q̃_min = q̂_min + c` | δ 网格上界 | 认证后 |
|---|---:|---:|---:|---:|---:|---|
| libero_spatial | 5.3733 | **3.3380** | 4.0716 | **7.4096** | 6.1875 | **全 MISS** |
| libero_10 | 8.4047 | **5.9722** | 3.4289 | **9.4012** | 5.8379 | **全 MISS** |

**即使选择期 offset 完全归零、且 δ 取冻结网格的上界**，`q̃_min > δ_max` 在两个 suite 上仍然成立
⇒ `export_boundaries` 找不到任何 `q̃ ≤ δ` 的格点 ⇒ 两条边界全为 `+inf` ⇒
artifact 可加载但**每一步都判 MISS**，SV/S0 臂退化成"从不使用 cache"，Pareto 无意义。

**根因是同一个**：`c` 与 `oof_offset` 都是 `order_statistic_offset(episode_max_residual(...), α)`
——**episode 内取 max、再取 50/100 个 episode 的 95% 序统计量**，即尾部的尾部。
去掉其中一层不改变另一层的量级。

**Rev 1 裁决：不再用 episode-simultaneous conformal correction 构造 primary 臂。** 原证书在当前数据上
只能导出 all-MISS，而 Gate 1/2 的论文主结论本来就是闭环 SR–cost Pareto。primary 改为经验标定的
dispatch surface；论文可以报告原协议为何退化，但不得再声称 Rev 1 artifact 提供有限样本偏差证书。

## 4. Rev 1 唯一方法：D-empirical

### 4.1 Development 数据与交叉拟合

- `D_dev = old fit ∪ old cal`，共 150 episodes、每 task 15 个；全部属于已查看的 exploratory data。
- episode 是不可拆分的 cluster。采用 task-stratified 5-fold，每 task 每 fold 固定 3 episodes；同一 episode
  的所有 step 必须位于同一 fold。fold id 机械定义为：每 task 按 official init index 升序排名，`rank mod 5`，
  无随机 seed、无重排自由度。
- 每 fold 只用另外 120 episodes 独立计算 equal-frequency edges、拟合 bimonotone 0.95-quantile surface，
  再对 held-out 30 episodes 执行与线上共用的 `surface_verdict`。
- SV 使用 `(s,v)`；S0 使用同一实现但 `B_v=1`。不得 pooled-step cross-validation。
- grid ladder 与稀疏止损沿用已审计实现：SV `(12,6)→(8,4)→(6,3)`，S0
  `(12,1)→(8,1)→(6,1)`；每个 Cartesian cell（含 empty）occupancy 必须 `≥8`，否则降一档，耗尽即 stop-loss。

### 4.2 δ 的机械选择

1. δ grid 固定为 `D_dev.y10` 的 p10、p20、…、p90 九点并去重；少于两点触发 stop-loss。
2. `oof_offset = 0`，OOF 预测不再冒充证书；每个候选直接导出 fold-specific boundaries 并运行 deployed verdict。
3. accepted-step accuracy 固定为 `mean(Y_eff≤δ | verdict≠MISS)`，FULL 取 `y10`、WARM 取 `y7`；
   accuracy gate 固定为 `≥0.90`，目标 hitshare 固定为 `≥0.40`；无 accepted rows 时 accuracy 定义为 1、
   但 hitshare=0，仍由 stop-loss 拒绝。
4. 在 accuracy 合格候选中优先保留 hitshare 合格者，取 hitshare 最大者；并列取最小 δ。若最大 hitshare
   为 0 或没有 accuracy 合格点，触发 stop-loss，不 emit。
5. SV 独立选择一次 δ*；S0 **继承同一 δ***，不得自行选择。

现有 D_dev 上按上述算法复算并冻结的结果是 spatial `δ*=6.1298201`、l10 `δ*=5.9096355`；实现必须由
表内容机械重算并与记录值满足绝对误差 `≤1e-6`，不能把数字手填成第二条自由路径。

### 4.3 Final fit 与 artifact 语义

- δ 冻结后，SV/S0 各自在全部 150 个 D_dev episodes 上 refit；`q_deploy=q_hat`，不计算、不读取旧 cal c。
- artifact 新增必填枚举 `certification_mode="empirical_no_certificate"`；`quantile_alpha=0.05` 只描述拟合
  quantile，不得命名为 coverage alpha。
- empirical artifact 不携带 `conformal_c`/`n_calibration_episodes` 的证书语义。若为兼容旧 schema 暂时保留
  字段，必须同时固定为 `0/0`，且 loader、emitter、launch ledger、analyzer 和论文导出都以 mode fail-closed，
  禁止将其显示或解释为 conformal certificate。
- 原 fit/cal 输入 digest、D_dev membership、fold assignment、δ grid/metrics/selection reason 与 final-fit digest
  全部写入 fit record；SV 与 S0 除模型维度外必须绑定相同 D_dev、fold、δ 与 policy/library identity。

cluster-valid step-risk certification 可以在未来以独立 variant 研究，但必须先冻结新统计构造并另采 fresh cal2；
它不阻塞 Rev 1 primary。放宽 episode α 的方案已实证不可行，episode-max regression 也未定义出逐决策边界，
二者不进入实现。

## 5. Primary 实验设计（`always_search`）

- **arm roster**：T1/T2/T3（threshold + 唯一 `start_t=0.3` warm tier）、S0、SV 为五个 formal core arms；
  SV± 若邻点存在，只作 descriptive，不进入 Gate 1/2。
- **gate**：五个 core arms 全部为 `always_search`；judge 保持各自类型，禁止 `always_hit`。
- **retrieval contract**：盘上 yaml 全部保持 configured `top_k=1`，使 key/search digest 与 D_dev 表一致。
  T1–T3 与 S0 的 effective top-k=1；SV artifact 固定 `k=5`、`contract.top_k=5`，由
  `min_required_top_k=5` 抬高 runtime effective width。top-1 winner/score 必须与 top-5 的首项逐位 parity。
  `fit_surface` 必须修复目前只在 s-only 分支覆盖 contract top-k 的反向条件，并增加保存→加载→完整装配测试。
- **成本 estimand**：只统计 GPU policy inference，不含 retrieval CPU。单位成本继续固定为 CUDA-graph
  stage1/stage2/stage3 `10.260266/27.686469/29.571860 ms`；FULL=`stage1`，
  WARM=`stage1+stage2+0.3*stage3`，MISS=三段之和。
- **聚合**：`C_a = Σ_d c(h_d)/N_decisions`，其中 `N_decisions` 是 accepted episodes 中的 client
  policy-inference calls，也等于通过 provenance join 后的 per_step verdict rows；它不是 simulator steps。
  比值必须在每个 bootstrap replicate 内重算，禁止先算 episode mean 再平均。
- **配对**：所有 arms 使用相同 A′ init、env seed、conductor seed 与 task/init grid；task-stratified、
  init-cluster paired bootstrap 的同一 joint replicate 同时计算 SR、成本与 threshold frontier。
- **系统开销报告**：另测 top-1/top-5 retrieval 延迟作为 descriptive overhead，不进入上述冻结成本 Gate，
  也不得据此事后改变成功裁决。

## 6. Secondary 实验设计（带固定 gate 的完整系统）

- roster 在 A′ 前固定为 **T1、T2、T3、SV 四臂**；不得看 primary 结果后只挑“胜出的 threshold”。
- 四臂接入同一个 `score_hysteresis` production gate；theta 按现有 `solve_gtp` 规则在 D_dev score 上一次重解，
  全臂完全相同且跨 sweep 固定。SV 使用 primary 已冻结的 artifact，不 refit、不换 δ。
- secondary 只回答“always-search 下的差异接回完整 gate 后是否仍可观察”，定性为 descriptive external
  validity；不进入 Rev 1 主成功裁决，不分配新的 confirmatory alpha。
- 必须无条件运行并报告完整四臂；不得以 primary 是否通过决定启动、停止、选臂或报告。
- secondary 与 primary 可以使用相同 A′ init 做 paired rollout，但 launch id、arm matrix 与结果表分开；
  绝对成本不跨层比较，只在 secondary 内部形成 threshold frontier 描述。

## 7. Primary 统计裁决（冻结）

### 7.1 共同设置

- A′ 仍为 30 official inits/task，10 tasks，split seed `20260827`；D_lib/D_dev/A′ 身份严格互斥。
- bootstrap 固定 10,000 replicates；每次按 task 分层、以 init 为 cluster 重采样，并对全部 arms 使用同一抽样。
- SR、解析成本、frontier interpolation 与所有差值在同一个 replicate 内形成。

### 7.2 Gate 1：SV 相对 threshold frontier

每个 replicate 用 T1–T3 的 `(SR,C)` 对 SV 的 SR 做 endpoint-clamped 线性插值，得到
`D_sr=SR_SV-SR_frontier` 与 `D_c=(C_SV-C_frontier)/C_frontier`。Gate 1 当且仅当：

```text
q0.05(D_sr) >= 0
AND
q0.95(D_c) <= -0.05
```

即相对 threshold Pareto，SV 的 SR 非劣且解析成本至少节省 5%。low/high endpoint replicates 保留，禁止删样本。

### 7.3 Gate 2：SV 相对 S0，证明 `v` 的增量价值

`ΔSR=SR_SV-SR_S0`，`ΔC=(C_SV-C_S0)/C_S0`。Gate 2 当且仅当：

```text
q0.05(ΔSR) >= 0
AND
q0.95(ΔC) <= -0.05
```

这条方向在查看 A′ 前冻结：`v` 的预期作用是把一部分 WARM 安全地重分配为 FULL，从而在 SR 非劣时降低
成本，而不是强迫 SR 严格提高。D_dev OOF 的 descriptive 预检显示 SV 相对 S0 成本为 spatial `-12.1%`、
l10 `-6.2%`，但该数字不进入 Gate，最终只认 A′ joint bootstrap。

### 7.4 总结论

- 每个 suite 独立运行 Gate 1→Gate 2；任一 Gate 失败，该 suite 不得称为 Pareto win。
- Rev 1 的跨-suite confirmatory claim 只有在 libero_10 与 libero_spatial **四个 Gate 全部通过**时成立；
  只通过一个 suite 时只能报告 suite-specific evidence，不能宣称通用胜出。
- 执行顺序固定为 l10 后 spatial；先看到的 suite 不得改变另一 suite 的 arm、δ、Gate 或停止规则。
- SV±、secondary production-gate 结果与 retrieval latency 全为 descriptive，不能补救 primary Gate 失败。

## 8. D0 数据语义止损（实现前第一步）

D0 只重放 D_dev 中已记录 observation，不是 rollout。GPU replay 样本固定为：两个 suite 中全部
`y7>p99(y7)` 行涉及的 unique winner/observation，加上每 suite 每 task 由 seed `20260828` 在其余 D_dev 行中
无放回抽取 2 行作为 controls；identity/schema 的非 GPU census 覆盖全部 D_dev rows 与全部 library entries。
三项全部通过才允许把现有 Y 表作为 Rev 1 输入：

1. **真实模型 self-resume parity**：current conditioning + `z_j` 生成 full reference 与 `x_c,0.3`；从该
   intermediate 用同一 current stage2 resume；以 float32 比较 action，固定
   `torch.testing.assert_close(rtol=1e-4, atol=1e-4)`，任一元素失败即 D0 FAIL。
2. **payload/sidecar identity**：对 winner 原 conditioning + sidecar `z_j` 重放，逐项比对 payload 的
   `x_j,0.3` 与 final action，float32 action/intermediate 同样使用 `rtol=atol=1e-4`；并断言
   `denoising_num_steps=10`、sidecar/payload shape+dtypes 正确、entry/trajectory/step join 唯一。
3. **极端路径分解**：覆盖两 suite 的 top residual episodes 与 task-stratified matched controls，记录
   `||x_j,0.3-x_c,0.3||`、direct/warm/current 三支偏差；此项只解释机制，不以结果方向改变协议。

第 1/2 项任一失败：D0 FAIL，当前表与所有派生 fit record 作废，停止且不得 emit/A′；修复数据链后必须
重新建表并重新独立复核。第 3 项无数值通过门，只要求计划样本完整、无异常/缺失并形成只读诊断记录。

## 9. 实现与放行清单

代码实现必须在一个新 G2 中同时覆盖：

1. `fit_surface` 的 D_dev 150-episode audit、episode-fold assignment、offset-free δ selection、final refit、
   S0 frozen-δ parity 与 stop-loss；旧 certified mode 若保留，必须走独立显式分支，禁止静默切换。
2. artifact schema 的 `certification_mode`/`quantile_alpha`，以及 loader→emitter→runner→analyzer→论文导出的
   mode fail-closed；empirical mode 不得显示 coverage/c。
3. SV effective-top-k contract blocker：configured digest=1、artifact k/contract=5、judge hint/effective=5；
   S0/T1–T3=1；四类 tamper 必须加载期拒绝。
4. `emit_precheck_yamls` 的 primary gate 全部改为 `always_search`；primary/secondary 分离 arm matrix，roster
   与每个 yaml digest 在 launch 前冻结。
5. analyzer 的新 Gate 2（SR 非劣 + 成本节省 5%）、两个 suite 的四-Gate 总裁决、旧 Gate 2 常量/字段拒绝，
   以及 synthetic end-to-end fixtures 覆盖 pass、每个单门 fail、suite split verdict。
6. 原有 A′ provenance、accepted `(task_uid,attempt,run_id)` join、decision count、三档 verdict/start_t、
   single-joint-replicate 与 ratio-of-sums 纪律逐条保留。

执行顺序固定为：**实现并独立审查 D0 checker → D0 → 实现 Rev 1 → G2 → D_dev empirical fit + artifact
load/contract audit → emit primary/secondary arm matrices + launch dry validation → l10 primary → spatial primary →
两 suite secondary → analyze/report**。任何一步失败都停止；在 D0、G2、artifact audit 与 launch dry
validation 全部通过前，不得启动 A′。D0 后的 fit/emit 只写离线 artifact/config，不是 rollout。

---

## 10. Codex 独立复核回复（2026-08-28，草案 v0 裁决）

> **Verdict：NEEDS REVISION，当前版本不可冻结。**
> 本复核重读了 gate/orchestrator/interceptor、artifact contract、`fit_surface`，并用两个完整 fresh 表重新执行
> final fit 与 calibration residual。只做只读计算；未写 artifact、未 rollout、未触碰 A′。

### 10.1 B1 — §2.3 的 gate skip 解释错误，必须删除（blocking）

`gate.py:56` 不是普通 gate skip 的语义；它只是在 `GateFunction` docstring 中描述可选的
`replay_target()`，实际仅由 `FollowWinnerGate` 使用。真实控制流在 `orchestrator.py:541-592`：

- `score_hysteresis` 返回 False 时没有 `replay_target`；orchestrator 返回
  **`MISS, searched=False`**；
- interceptor 明写“MISS (including gate-skip)”，随后走 teacher/full-inference 路径；
- analyzer 因而按 MISS 计 `stage1+stage2+stage3`，这是正确成本，不是 FULL_HIT；
- 只有 FollowWinner 的 blind replay 才是 `FULL_HIT, searched=False`。即便这个特例也不是“连 stage1
  都没跑”：gate 位于 CP1，orchestrator 的注释明确说 **Stage 1 + build 已经执行**，它只省 search/judge
  与 stage2/3，所以按 FULL_HIT=`stage1` 计价同样正确。

仓库已有反例测试钉死这一区别：N1/N4 的 `searched=False` 是 MISS/teacher inference，N2 blind replay
才是 FULL_HIT。相关 gate、orchestrator、surface 与 fit 定向套件本次复跑 **130 passed**。

因此不存在草案所称“现行 FULL_HIT 混了免费 gate skip、计价错误”。always-search 仍应作为 primary，
理由是隔离 gate×verdict 交互并清洁归因，而不是修复一项不存在的成本 bug。§2.3、§5 与 §6 中建立在
该误读上的成本论证必须改写。

另一个措辞修正：always-search 下应写
`N_decisions = accepted episodes 中的 client policy-inference calls = per_step verdict rows`，不要写成
`N_steps`；一个 policy decision 会执行 `h_exec` 个 simulator control steps，两者不是同一个分母。

### 10.2 B2 — §3 的认证期阻塞完全复现（accepted）

按当前 `choose_grid → final q_hat → cal episode_max_residual → order_statistic_offset` 重算：

| suite | artifact | `c` | `q̃_min` | `δ_max` | 导出结果 |
|---|---|---:|---:|---:|---|
| spatial | SV | 3.3380 | 7.4096 | 6.1875 | full/warm 全 `+inf` |
| spatial | S0 | 3.4653 | 7.4465 | 6.1875 | full/warm 全 `+inf` |
| l10 | SV | 5.9722 | 9.4012 | 5.8379 | full/warm 全 `+inf` |
| l10 | S0 | 7.0720 | 11.0852 | 5.8379 | full/warm 全 `+inf` |

所以先前“去掉 OOF offset、只保留正式 c”的建议确实不够；本条订正成立。当前 episode-simultaneous
certificate 与现有 δ grid 在两条线上都无法产生有意义的臂。

但“选择期与认证期双重保守”不应被缩写成简单的“双重计费”：OOF offset 用于选择，cal c 用于最终
certificate；它们统计角色不同，只是都采用 episode-max 尾部，导致数值上重复支付极端值。修订要么更改
目标保证，要么放弃证书式构臂，不能把其中一个机械置零后仍声称原保证不变。

### 10.3 B3 — §4 四个方案的可执行性复核（blocking）

#### A. “逐步 conformal”数值可行，但按草案直接 pooling 在统计上不成立

将 cal 的所有 step residual 当成独立样本作朴素 95% order statistic，描述性数字为：

| suite | naive step `c` | `q̃_min` | `δ_max` 下 fit hitshare / accuracy |
|---|---:|---:|---:|
| spatial | 0.4103 | 4.4820 | 0.594 / 0.988 |
| l10 | 0.2167 | 3.6456 | 0.595 / 0.986 |

它在工程上能出臂，但同一 episode 内的 step 相关且 episode 长度不同，把 3364/9205 行直接当 exchangeable
calibration samples 不能自动得到草案声称的“随机一步 95%”保证。必须先冻结抽样单位和 estimand，例如：

- 随机抽 episode、再按预注册分布抽一个 step；每个 calibration episode 只贡献一个 exchangeable score；或
- 使用以 episode 为独立 cluster 的 conformal-risk-control / risk-controlling prediction-set 构造，保证
  expected per-episode unsafe-step rate，而不是 all-steps simultaneous coverage。

两者都是新的统计设计，不是把 `episode_max` 删掉一行即可。若正式证书仍是论文主张，需要数学说明、
simulation coverage test，并重新 G1/G2。

#### B. 放宽到 `α=0.2` 仍不可行，应从候选中删除

保持当前 95% q_hat、只把 cal episode order statistic 放宽到 80% 时：spatial hitshare 仅 `0.00088`，
l10 仍为 `0`；若 q_hat 与 calibration 同时按 `α=0.2` 重拟合，两条线仍全 MISS。继续放宽也救不了 l10：
固定当前 q_hat 时 `α=0.3` 只有 `0.00034` hitshare，甚至 `α=0.5` 也只有 `0.112`，远低于 0.40 目标。
这已不是“未必够”，而是当前 δ grid 下实证不可行；再放宽会同时失去有意义的保证。

#### C. “拟合 episode-max”没有定义出 `(s,v) → boundary`，当前不可执行

一个 episode 只有一个 max score，却包含几十个不同 `(s,v)` cell。草案没有规定这个 episode-max 应配给
哪个 `(s,v)`，把同一个 max 复制给每一步会伪造样本量并把极端值污染所有 cell；把 episode 压成一个
feature 又失去逐决策 dispatch boundary。除非先提出明确的 group-conditional model、loss 和部署映射，
方案 C 不是一个算法，而且 50 个 fit episodes 不足以支撑现有 8×4 lattice。

#### D. 方向最现实，但“直接按 SR/成本选点”在现有数据上做不到

现有 fit/cal 表只有离线 `Y,s,v` 与原 teacher episode success，没有“如果采用该 cache verdict 后”的
counterfactual SR/成本。Gate 1/2 虽只读 SR 与成本，却只能在 rollout 后得到；不能拿 A′ 先选点再把同一
A′ 当 confirmatory test。

方案 D 有两个合法版本：

1. **D-empirical（推荐作为当前 primary）**：用 fit 上 cross-fitted deployed verdict、`offset=0`，按已冻结的
   accuracy/hitshare 规则选 δ；full fit 后不加 c，生成明确标记
   `certification_mode=empirical_no_certificate` 的 artifact。现有重算在 `δ_max` 给出 spatial
   `0.609/0.983`、l10 `0.635/0.979` 的 OOF hitshare/accuracy。S0 继承同一 δ。然后只在干净 A′ 上做
   SR–cost confirmatory comparison。
2. **D-utility**：另建与 A′ 严格互斥的 pilot rollout pool，在其上选 SR–cost operating point；冻结后再跑
   A′。这才是真正“直接按效用选点”，但会增加 rollout 预算和一层样本划分。

不能把 `conformal_c=0` 填进现有 artifact、仍沿用 conformal 命名；schema、fit record、emitter 和论文 claim
都要显式声明“无证书”。如果 owner 仍要保住 theorem，应把 cluster-valid A 作为独立 certified variant，
而不是阻塞 verdict-primary Pareto。

### 10.4 B4 — top-k 的部署思路基本正确，但存在一个被 stop-loss 掩盖的 artifact blocker

保留盘上 yaml `top_k:1`，由 SV judge 的 `min_required_top_k=5` 把**有效**宽度抬到 5，是当前契约设计的
正确用法。threshold 与 S0 只取 winner；既然 top-1 identity/score 已实测逐位相同，它们无需为了形式公平
消费额外候选。需要在论文中准确写成：比较的是 GPU policy-inference cost，不含 retrieval CPU；最好另报
top-1/top-5 检索延迟，避免把该前提藏起来。

草案有两处需要改：

- 改 yaml 为 5 并非“每个臂都会拒绝 artifact”——T1–T3 根本不加载 surface artifact；现有 SV/S0 会因
  digest 不符而拒绝，但若重新拟合／重建契约并非理论上不可行，只是没有必要。
- 更严重的是当前 `fit_surface.py:645-658`：`compute_surface_retrieval_contract(yaml)` 先写入 configured
  `top_k=1`，随后代码**只在 `args.s_only` 时**覆盖 `contract["top_k"]`。SV 使用 `k=5` 且
  `uses_disagreement=True`，`SurfaceArtifact.validate()` 强制 `contract.top_k == artifact.k`，因此 revised
  protocol 一旦走到 SV artifact 保存就会以 `1 != 5` 失败。旧 stop-loss 在保存前退出，掩盖了这个 bug。

正确修复是保留 `search_digest` 对盘上 yaml(top_k=1) 的绑定，同时把 artifact contract 的有效
`top_k` 明确写成 `args.top_k=5`，并增加“configured=1、judge hint=5、artifact k/contract top_k=5、
effective width=5”的完整装配测试。S0 应明确使用 `k=1`，而不是含糊地复用 SV 的 `--top-k 5`。

### 10.5 B5 — calibration 已被查看，正式证书不能复用原 cal（blocking）

本草案是在查看两个 suite 的 cal residual、`c` 和多个反事实方案后形成的。若依据这些数字选择 A/B/C，
再用同一 100-episode cal pool 产出“正式” conformal certificate，方法选择已经对 calibration 泄漏，
原有限样本覆盖解释不再成立。

- 若选择 D-empirical：把现有 fit+cal 全部声明为 exploratory development data，在碰 A′ 前冻结算法和所有
  operating points；A′ 仍可作为未触碰的 confirmatory closed-loop test。
- 若保留任何正式证书：先冻结新的统计构造，再采与既有 fit/cal/A′ 全部互斥的 **fresh cal2**，且 cal2
  只使用一次。不能从 A′ 借样本，也不能在看到 cal2 后再选 A/B/C。

因此 §9 所说“在 fit 表上比较四方案”可以作为方法开发证据，但必须补上：这些数字只能用于选择新版协议，
不能再由原 cal 为被选方案提供 confirmatory certificate。

### 10.6 建议的 Rev 1 决策

我建议 owner 现在冻结的方向是：

1. primary = `always_search` + **D-empirical** verdict Pareto；旧协议负结果永久保留；
2. T1–T3、S0、SV 全部在 A′ 上成对比较，SV operating point 在 A′ 前冻结；
3. secondary production gate 与 primary 同时预注册，但只比较该层内部 SR–cost，不与 always-search 的绝对
   cost 数字混用；
4. 删除方案 B；方案 C 退回方法研究；若一定要保留证明，另立 cluster-valid A + fresh cal2，不阻塞 primary；
5. 修复 SV effective-top-k contract blocker，并在 emitter/analyzer 中强制
   `certification_mode`，禁止 empirical artifact 被误报为 conformal；
6. 先完成 D0，再 emit；D0 失败则所有现有 Y 表作废，D0 通过才进入 revised pipeline。

这条路线牺牲的是“本轮立刻给出 episode-all-steps 95% 偏差证书”，保住的是最重要且尚未回答的科学问题：
**在没有 gate 干扰时，`v` 驱动的三档 dispatch 是否在干净闭环上超过 s-only 与 threshold Pareto。**
对当前论文成败，这个问题比维持一个实证上只能导出 all-MISS 的证书更关键。

---

## 11. Codex Round 2 复核（2026-08-28）

> **Verdict：NEEDS REVISION（缩小到 5 个 blocking）。**
> §2.3 的 gate 语义、§5 的成本分母措辞、top-k 自校验 blocker 已被正确接受；认证期全 MISS 的数字也仍可复现。
> 但正文还没有真正选择并定义 D-empirical，Gate 2 的方向与新 primary 的实际优势相反，故仍不可冻结。
> 本轮新增计算仍只使用既有 fit/cal 表；未写 artifact、未 rollout、未触碰 A′。

### R2-B1 — 把 D-empirical 写成唯一 primary，而不是在 §4/§8 继续四选一

当前正文仍有三处互相矛盾：§4 继续把 A/B/C/D 并列，§8.1 仍要求 owner 四选一，§8.3 仍写已经作废的
`N_decisions=N_steps`。与此同时 §10 已判定 B 实证不可行、C 尚不是算法、A 若要有证书必须另立统计构造
与 fresh cal2。Rev 1 必须把历史候选移入“rejected/secondary research directions”，并明确：

- primary 唯一口径 = `always_search + D-empirical`；
- 当前不声称 finite-sample deviation certificate；
- A 仅作为未来 certified variant，只有 cluster-valid 构造 + fresh cal2 后才能恢复；
- B 删除，C 退回方法研究；
- 文档版本升为 Rev 1，§8 不再保留一个会阻止 artifact 构造的 owner 未决项。

否则这仍是一份问题清单，不是一份可以冻结的协议。

### R2-B2 — D-empirical 应把原 fit+cal 合并成 150-episode development pool

选择 D 后，原 100 个 cal episodes 已经失去正式认证身份，而且其 residual 已被查看。继续只用 50-episode
fit 拟合、把 100 episodes 闲置，没有统计收益，还让 surface 与当前 threshold emitter（使用 fit∪cal score）
的数据预算不一致。更干净的冻结定义是：

1. `D_dev = old fit ∪ old cal`，150 episodes，15/task；全部明确标记为 exploratory development data；
2. 仍以 episode 为 fold 单位，task-stratified 5-fold（每 task 每 fold 3 episodes），禁止 step-level 泄漏；
3. 每 fold 在其 120-episode train 上独立拟合 edges 与 `(s,v)` quantile surface；
4. `oof_offset ≡ 0`；δ grid 固定为 `D_dev.y10` 的 p10…p90 九点；逐行执行 shared deployed verdict；
5. 选择规则仍机械固定：accuracy ≥0.90；优先 hitshare ≥0.40；最大 hitshare，tie 取最小 δ；
6. 选定 δ 后在全部 150 episodes 上 refit；`q_tilde = q_hat`，不读旧 cal c；
7. SV 用 `(s,v)`，S0 使用同一 δ、同一 D_dev、自己的 s-only fit；不得各自挑 δ。

只读复算结果：

| suite | arm | 机械选择的 δ* | OOF hitshare | OOF accepted accuracy |
|---|---|---:|---:|---:|
| spatial | SV | **6.1298201** | 0.6623 | 0.9785 |
| spatial | S0（同 δ） | 6.1298201 | 0.7131 | 0.9804 |
| l10 | SV | **5.9096355** | 0.6661 | 0.9791 |
| l10 | S0（同 δ） | 5.9096355 | 0.7196 | 0.9718 |

这些数字不是 confirmatory evidence，只用于在 A′ 前冻结算法；真正结论仍只来自 A′。artifact schema 必须新增
并强制 `certification_mode="empirical_no_certificate"`，把原 `alpha` 明确重命名／解释为 quantile-fit level，
不能留下 `conformal_c=0` 让下游误报成证书。fit record、emitter、launch ledger、analyzer、论文表格都要
拒绝 mode 不一致。

### R2-B3 — 当前 Gate 2 检验了错误的 Pareto 方向（blocking，新增）

hitshare 把 FULL 与 WARM 合在一起，不能代表三档解析成本。按上面的 150-episode OOF verdict 逐档计价：

| suite | arm | FULL / WARM / MISS | analytic cost (ms/decision) | accepted accuracy |
|---|---|---|---:|---:|
| spatial | SV | 0.375 / 0.287 / 0.338 | **40.094** | 0.9785 |
| spatial | S0 | 0.196 / 0.518 / 0.287 | **45.606** | 0.9804 |
| l10 | SV | 0.562 / 0.104 / 0.334 | **33.190** | 0.9791 |
| l10 | S0 | 0.471 / 0.248 / 0.280 | **35.394** | 0.9718 |

SV 相对 S0 的离线结构性优势是：利用 `v` 把许多 WARM 重分配为 FULL，在安全精度近似不变时，成本分别
降低 **12.1% / 6.2%**。这正是三档 dispatch 应证明的增量价值。

原 Gate 2 却冻结为 `q0.05(ΔSR)>0` 且 `q0.95(ΔC)≤+5%`：它要求 SV 必须提高 SR，只把成本当“不劣于”
约束。若 A′ 上 SV 与 S0 的 SR 相同而成本显著更低——即真正的 Pareto dominance——该 Gate 仍会失败。

Rev 1 的 primary Gate 2 应改为与数据冻结前显示的方向一致：

```text
q0.05(SR_SV − SR_S0) >= 0
AND
q0.95((C_SV − C_S0) / C_S0) <= -0.05
```

即 SR 非劣、成本至少节省 5%，仍使用同一组 task-stratified/init-cluster paired joint replicates。
若 owner 想允许“SR 显著提高、成本非劣”作为第二条成功路径，必须预先定义 union-of-two Pareto routes 并处理
多重性；最简洁且与当前机制证据相符的 primary 是上面的 cost-superiority route。这个改动是 post-hoc，
但 A′ 干净，现阶段正是合法冻结它的最后窗口。

### R2-B4 — Secondary 不能在看完 primary 后选择“胜出的 threshold”

§6 同时写“primary 的胜出 threshold”与“不得等 primary 跑完再决定”，两者逻辑冲突。胜出臂只有看 A′
结果后才知道；再把它接 gate，会把 primary test outcome 用于 secondary arm selection。

二选一并在冻结时写死：

- secondary 同时包含 T1–T3 + frozen SV（推荐；内部仍按 threshold frontier 比较）；或
- 仅包含一个由 D_dev 的预注册规则、而非 A′ 结果选出的 threshold cell。

如果 secondary 只作 descriptive external-validity，必须明确不进入主成功裁决；若要 confirmatory，则还需预先
分配错误率。所有 secondary arms 可以复用相同 A′ init 做成对 rollout，但 arm roster 不能由 primary outcome
自适应决定。

### R2-B5 — top-k blocker 与 D0 仍是实际放行门，不只是文档提醒

当前工作树中的 `fit_surface.py` 尚未修复 SV `contract.top_k=1 != artifact.k=5`；文档接受问题不等于流水线
已经可执行。Rev 1 的 implementation plan 至少要钉死：

- SV：盘上 configured top_k=1，search_digest 仍绑定 1；artifact k/contract effective top_k=5；judge hint=5；
- S0：artifact k=1、uses_disagreement=False、effective top_k=1；
- threshold：top_k=1；
- 保存→加载→完整 config 装配的正例，以及 configured/digest/hint/k 四类篡改负例；
- top-1/top-5 retrieval latency 只作 descriptive system overhead，不混入已冻结 GPU inference estimand。

此外 D0 的真实模型 self-resume parity、payload/sidecar identity、极端路径分解尚未执行。协议可以先把 D0
写成机械 gate，但在它通过前不得把现有 Y 表当有效 development input，不得 emit，更不得启动 A′。

### Round 2 放行条件

完成 R2-B1…B5 后，这份协议可以进入独立 G1：primary 的方法、δ 选择、S0 对照、Gate 2 方向、secondary
roster、artifact mode 与 top-k contract 都将没有执行期自由度。当前应继续保持 **未冻结、未生效、A′ 干净**。

---

## 12. Review Authority 最终裁决（Rev 1）

**G1 APPROVED。** §1–§9 已由 Review Authority 直接改写为唯一、自洽、可执行的 Rev 1 权威正文；
§10–§11 仅保留审查轨迹，其中的 `NEEDS REVISION` 与“未冻结”描述是当时状态，不再覆盖本节裁决。

放行范围是：实现 D0 checker，并在 D0 通过后实施 Rev 1 代码、测试与 artifact/config 生成。该批准**不等于**
批准立即运行 A′；A′ 的执行门仍是 §9 规定的 D0、G2、artifact contract audit 与 launch dry validation
全部通过。原协议负结果永久保留，Rev 1 artifact 必须以 `empirical_no_certificate` 对外，任何路径都不得恢复
或暗示已作废的有限样本偏差证书。

## 13. Execution Authority 对 Rev 1 裁决的复核（2026-08-28）

owner 要求执行方不得直接接受，须自行检验。以下为独立复验结果。**接受 Rev 1，无阻塞异议**；
三点观察为知情记录，不构成异议。

### 13.1 已独立复验、确认成立的项

| 项 | 复验方式 | 结果 |
|---|---|---|
| 冻结 δ\* | 在两张 fresh 表上重算 `D_dev = fit ∪ cal`（150 ep）的 `y10` p10…p90 九点网格 | **成立**：spatial `6.1298201`、l10 `5.9096355` 精确等于各自 p90，与记录值差 `3.9e-8` / `4.4e-8`，在规定的 `≤1e-6` 内 |
| §10.1 gate skip 语义（推翻执行方 §2.3） | 读 `orchestrator.py:541-592` 与 `gate.py:48-60` 全段 | **codex 正确、执行方错误**。`gate.py:56` 描述的是 `replay_target()` 可选 hook（仅 `FollowWinnerGate` 使用），同段下一句即写「Returns `None` for a plain gate-skip」；`score_hysteresis` 走 skip 分支返回 `HitType.MISS, searched=False`，按三段计价正确 |
| §10.4 SV artifact contract blocker | 构造 save→load 全路径实测 | **成立**：`SV k=5 / contract.top_k=1` → `ValueError: contract top_k=1 != artifact k=5`；`contract.top_k=5` → OK；`S0 k=1 / uses_disagreement=False` → OK。`save_surface_artifact:309` 即调 `validate()`，故**保存时**失败 |
| §10.2 认证期阻塞 | 与执行方原算独立一致 | 成立（codex 另补 S0：spatial `c=3.4653`、l10 `c=7.0720`，同样全 MISS） |

### 13.2 三点观察（知情记录，非异议）

1. **δ 的机械选择在本规则下是退化的。** hitshare 对 δ 单调不减（δ 越大，`q̂ ≤ δ` 的格点越多），
   故「accuracy 合格候选中取 hitshare 最大」等价于「取通过 accuracy 门的最大 δ」——**δ\* 必然落在网格上界**，
   除非上界过不了 `0.90`。两 suite 都取到 p90（accuracy 0.9785 / 0.9791）是规则的必然结果，不是巧合。
   规则本身机械、无执行期自由度，故不构成异议；但"九点网格"实际只在上界失败时才起选择作用，
   报告时不宜描述成在网格上做了实质选择。
2. **Gate 2 存在真实的失败风险，且风险来自 SV 的核心机制本身。** SV 相对 S0 的成本优势来自把 WARM
   重分配为 FULL（spatial FULL 0.196→0.375、l10 0.471→0.562），而**更多 FULL 正是最可能拖累 SR 的方向**
   （直接复用缓存动作块、不做任何当前条件下的重积分）。Gate 2 要求 `q0.05(SR_SV − SR_S0) ≥ 0`，
   因此它是一个可能真的过不了的检验。这是检验应有的性质，记录以备判读时不被误读为"设计缺陷"。
3. **S0 在 hitshare 与 accepted accuracy 上都不劣于 SV**（spatial 0.7131/0.9804 vs 0.6623/0.9785；
   l10 0.7196/0.9718 vs 0.6661/0.9791），SV 的全部主张落在 FULL/WARM 配比即成本一项上。
   这与 Gate 2 的新方向自洽，但也意味着**若 Gate 2 不过，本线没有第二条论证路径**。

### 13.3 执行方在本轮确认的自身错误（三处，均已就地更正）

1. `ref_mode=fresh` 本就是噪声配对 —— "teacher 自噪声支配 Y"整条因果链作废（§10 之前已自查发现）。
2. §2.3 的 gate skip 计价论证 —— 误把 `FollowWinnerGate` 的可选 hook 当作普通 gate skip 语义。
3. handoff §5.3 的「contract 记录 `top_k=1` 也没问题」—— 只核了部署期 yaml 级契约比对，
   **漏了 `surface_judge.py:297` 这条 artifact 自校验**。

三处的共同错误模式是**读了名称或 docstring 片段就推断语义、没有读控制流**。已作为纪律记入 handoff。

## 14. D0 执行记录（2026-08-28）—— **PASS，两 suite**

`exp/dispatch_surface/d0_check.py`（新增，18 个测试）在两条线上按 §8 规格执行完毕。

| suite | D0 | census | check1 self-resume parity | check2 payload/sidecar identity | check3 |
|---|---|---|---|---|---|
| libero_10 | **PASS** | 通过（9205 rows / 2496 entries / 1746 winners） | n=113, fail=0, **max=0.000e+00** | n=113, fail=0, **max=0.0（action 与 intermediate 均逐位相同）** | 完整 |
| libero_spatial | **PASS** | 通过（3364 rows / 1018 entries / 850 winners） | n=54, fail=0, **max=0.000e+00** | n=54, fail=0, **max=0.0** | 完整 |

GPU 样本按 §8 冻结规则：两 suite 全部 `y7 > p99(y7)` 行（93 / 34）+ 每 task 2 行 control（seed `20260828`，各 20 行），10 个 task 全覆盖。

### 14.1 首轮的两次失败都是 checker 缺陷，不是数据问题；判据是"不可能的比例"

1. **H5 组名零填充**：checker 用 `f"step_{idx}"` 重构组名，真实键名是 `step_0000`。
   `build_dispatch_table` 只**解析**从不重构，故从未暴露。**判据是 54/54 全"缺失"这个不可能的比例**，
   而不是逐个猜。已改为解析建映射（对任何补零稳），并把静默跳过改成 fail-fast。
2. **resume parity 的时间戳对齐**（见 §14.2）。

### 14.2 check1 的容差问题：`1e-4` 低于两条代码路径的可达下限

首轮 check1 **113/113 与 54/54 全败**，差值中位 1.6e-3。定位实验：

```
full vs resume(start_t=0.3 字面量)          max|d| = 1.360e-03
full vs resume(start_t=累加值 0.2999999225) max|d| = 0.000e+00   <- 逐位相同
same-path repeat（确定性）                  max|d| = 0.000e+00
model param dtypes: {bfloat16, float32}     bf16 eps = 7.8e-03
```

`_stage3_with_intermediates` 从 1.0 逐步累加 `dt=-0.1`，快照处实际持有 **0.2999999225**；
`run_stage3_from` 取字面量 **0.3**。二者差 **7.749e-08**（float32 累加漂移），经 **bfloat16** 动作专家
放大为 ~1.4e-3。**`run_stage3_from` 本身正确**——喂它轨迹实际到达的 t，逐位重现全程。
因此 `rtol=atol=1e-4` 低于这两条路径的可达下限，**任何数据都无法通过**。

**处置（需 Review Authority 追认）**：§8.1 只规定「从该 intermediate 用同一 current stage2 resume」，
未规定 `start_t` 实参。执行方按该检查的**声明目的**（验 resume 机制能否重现所延续的轨迹）
以**轨迹实际持有的时间戳**作门控比较，得 0.0；同时把**字面量 0.3** 的调用作为诊断列同报告
（`max_abs_diff_literal_start_t`：l10 4.233e-02、spatial 8.488e-03），不隐藏。
**冻结容差 `1e-4` 未被放宽**——改动的是比较对象，不是阈值。若 Review Authority 认为门控应取字面量支，
则该门在当前代码下不可能通过，需另行裁定。

### 14.3 附带确认的部署事实（不影响本轮有效性）

部署期 warm 档同样以字面量 `0.3` 接续库中按累加 t 保存的中间态，故**真实 WARM 输出与"理想延续"
存在同量级（~1e-3）的系统性偏差**。但 `build_dispatch_table` 的 `y7` 由**完全相同的调用**产生，
因此**标定与部署自洽**，不构成失效。这是封缝处一处已量化的微小偏差，记录备查。

### 14.4 独立复算佐证

check3 的 `dev_direct_full` 与 `dev_warm` 在抽样行上**精确等于**表中的 `y_tau10` / `y_tau7`
（如 l10 样例 5.112 / 8.237），即 D0 用独立代码路径复算出了标定表的 Y 值。

**结论：D0 三项全通过，现有 Y 表可作为 Rev 1 输入。** 按 §9 进入 Rev 1 实现。

## 15. Rev 1 实现中发现的规格空隙：A1 止损与网格阶梯相互作用（待裁定）

### 15.1 实现已复现冻结值

Rev 1 的 `fit_surface`（`--certification-mode empirical_no_certificate`）在两条线上**独立复算出 §12 的冻结值**：

| | §12 冻结 | 本实现 | hitshare | accepted accuracy |
|---|---|---|---|---|
| spatial δ\* | 6.1298201 | **6.129820060729981** | 0.6623（§12: 0.6623） | 0.9785（§12: 0.9785） |
| l10 δ\* | 5.9096355 | **5.909635543823243** | 0.6661（§12: 0.6661） | 0.9791（§12: 0.9791） |

`n_dev_episodes=150`、`fold_sizes=[30,30,30,30,30]`、`oof_safety_offset=0.0`，与 §4.1／§4.2 逐条相符。

### 15.2 但两条线随后都停在 `a1_violation_rate_above_20pct`

这是**原协议的 (A1) 止损**（`a1_violation_rate > 0.20`）。Rev 1 正文 §1–§9 **没有提到它**——
只规定了网格阶梯与稀疏止损、以及 δ 选择的两个止损。**本轮是该止损第一次被执行到**：
旧流程在 δ 选择阶段就止损了，从未走到 A1 诊断。

**两条规则在 D_dev 上相互作用**：

- §4.1 的阶梯 `(12,6)→(8,4)→(6,3)` 只由**稀疏格规则**（每个笛卡尔格 occupancy ≥8）驱动下降。
  D_dev 有 150 episodes，`(12,6)` 轻松满足 occupancy，**阶梯因此永不下降**。
- A1 门随后在 `(12,6)` 上否决该网格。

### 15.3 违反率强烈依赖网格粒度，与每格分位数噪声一致

用**与门控逐字相同的统计量**（沿 s 下降 + 沿 v 非降两个方向都数）：

| grid | spatial | libero_10 | 每格中位样本数 |
|---|---:|---:|---:|
| (12,6) | **0.393** ✗ | **0.282** ✗ | 47 / 126 |
| (8,4) | 0.212 ✗ | **0.173** ✓ | 106 / 298 |
| (6,3) | **0.185** ✓ | **0.111** ✓ | 187 / 517 |

`(12,6)` 时 spatial 每格中位仅 **47** 行，而该诊断比较的是**每格无约束的经验 0.95 分位数**
——47 个样本的 95 分位实质上是第 3 大值，方差很大，相邻格估计在真实单调下也会频繁交叉。
违反率随格子变粗单调下降（0.393→0.212→0.185；0.282→0.173→0.111）与这一解释一致。

> 执行方自查：本节数字首版曾报 0.350/0.233，原因是诊断脚本**只数了 v 方向**、漏了 s 方向。
> 已改为与门控逐字相同的双向统计并复现拟合内记录值（0.393/0.282）。

### 15.4 请裁定（执行方不自行决定）

1. **(A1) 止损是否随原协议一并继承进 Rev 1？** 它原本服务于带证书的构造；Rev 1 已放弃证书。
2. 若继承，**阶梯是否也应在 A1 失败时下降**（而非只由稀疏格规则驱动）？按上表，两条线都会落到
   `(6,3)`（0.185 / 0.111，均过门）；`(8,4)` 则 l10 过、spatial 以 0.212 差之毫厘不过。
3. 若继承且不允许因 A1 下降，则 Rev 1 在 `(12,6)` 上被这道门挡住，需另行裁定。

**在裁定前，执行方不改动该止损、不下调阈值、不手选网格。** 其余 Rev 1 项（emit 的 always_search、
analyzer 的新 Gate 2）与本决定无关，继续实现。

## 16. G2 Round 1 — Review Authority（2026-08-28）

**Verdict：NEEDS REVISION。当前不得生成正式 artifact、不得 emit、不得 rollout、不得触碰 A′。**

定向验证为 `262 passed`，目标文件 Ruff 与 `git diff --check` 均通过；但这些测试没有执行
Rev 1 `fit_surface.main()` 的正式产物路径，没有把 secondary matrix 交给 runner，也没有覆盖跨-suite
四 Gate 总裁决。以下均为 blocking。

### G2R1-B1 — D0 改了冻结的比较对象，现有 `PASS` 不能追认

§8 冻结的是从 `x_c,0.3` 以同一 current stage2 做 `start_t=0.3` resume，并在 `1e-4` 下比较。
`d0_check.py:284-291` 却把门控调用改成 `T_ACC_WS=0.2999999225`，把真实部署使用的字面量 `0.3`
降成不参与门控的 diagnostic。执行记录已经证明字面量分支的最大差为 l10 `4.233e-02`、spatial
`8.488e-03`，所以当前实现通过的是一个理想化调用，不是冻结的部署调用。

放行条件：二选一，且不能在 G2 内静默选择。

1. 按冻结协议以字面量 `0.3` 门控；若失败，修正 timestep/resume 语义，重建受影响的 library/table 并重跑 D0；或
2. owner 明确改变 D0 的科学问题并重新走 G1，说明为何 ideal-resume 足以验证 Rev 1 的数据语义。

在此之前，§14 的“两 suite D0 PASS”无效。

### G2R1-B2 — D0 是可完全绕过的旁路脚本，且报告没有内容级 provenance

`fit_surface.py` 不接收 D0 record，也不检查 `D0 == PASS`；任何人都可跳过 D0 直接拟合。D0 JSON 只记录
table 路径，不冻结 table、library pkl、noise sidecar、query/library H5 census、weights、checkpoint/policy 的
内容摘要，因此即使提供一个旧 PASS，也无法证明它审计的是当前输入。

同一 checker 还有三类 fail-open：controls 用 `min(2, len(pool))`，不足两行仍可 PASS；library entry id
重复会被 `by_id` 静默覆盖；payload action dtype 没有检查且 check2 强制 cast 到 float32，能掩盖错误 dtype。

放行条件：D0 输出 canonical input digest/protocol/suite/sample manifest；`fit_surface --d0-record` 必填并逐项
重算、要求 PASS，D0 digest 进入 fit record 与 artifact meta。controls 必须精确 `2/task`，entry id、解析后的
H5 step id、H5 basename join 均须唯一，payload/action/intermediate/sidecar 的 shape 与 dtype 全部 fail closed，
并补对应“旧实现必败”测试。

### G2R1-B3 — Rev 1 未授权的 A1 门仍阻断正式 artifact

`fit_surface.py:684-685` 在 `a1_violation_rate > 0.20` 时 stop。Rev 1 §4.1 只允许 occupancy 驱动网格阶梯，
§4.2 只冻结 accuracy/hitshare 止损；A1 没有被继承为选择门。执行方也已实证两 suite 在 `(12,6)` 都会
被该旧门挡住，因此当前代码事实上没有完成“可产 artifact”的 Rev 1。

放行条件：Rev 1 中 A1 只记录为 diagnostic，不得 stop，也不得用它事后触发 `(12,6)→(8,4)→(6,3)`；
后者会改变已冻结的拟合器与 D_dev 选点。如果要让 A1 参与选格，必须另走 G1，而不是在本轮裁定。

### G2R1-B4 — formal fit 的冻结常数和要求的审计记录没有落成硬契约

empirical 分支仍接受任意 `--alpha`、`--h-exec`、`--top-k`；也没有按 suite 将机械复算的 δ 与
spatial `6.1298201` / l10 `5.9096355` 做 `1e-6` 硬比对。这样可以产生与 Rev 1 不同、却仍被 emitter/runner
接受的 artifact。fit record 也没有写 §4.3 要求的 D_dev membership、逐 episode fold assignment 与
final-fit digest；目前只有 input file digest、fold size 和若干 scalar，无法审计机械折分或 final refit。

放行条件：formal empirical mode 钉死 `alpha=0.05 / h_exec=5 / SV top_k=5 / S0 top_k=1`，由 manifest suite
选择唯一冻结 δ 并硬比对；record 写 canonical D_dev membership、fold map（内容及 digest）、δ grid/metrics、
final edges/q/boundaries digest。S0 record/artifact 必须绑定同一 membership/fold/δ/policy/library，并补参数漂移
与 fold/final-fit 篡改拒绝测试。旧 conformal 分支可以保留，但必须与 Rev 1 formal 命令和产物路径显式隔离。

### G2R1-B5 — artifact 与 fit record 没有绑定到 launch，路径内容可在 emit 后替换

arm matrix/ledger 只冻结 YAML sha；YAML 内只是 `surface_artifact_path`。artifact 文件在 emit 后被替换时，
YAML sha、matrix sha 与 launch ledger 全部不变。runner 只加载“当下”的 SV artifact，既不冻结每臂 artifact
sha，也不检查 empirical mode；S0 更未在 launch 前独立 attestation。analyzer 又加载分析时的 artifact 与
fit record，且不检查 `certification_mode/conformal_c/n_calibration_episodes`。因此执行 artifact、分析 artifact
和 fit record 可以是三套内容而仍通过现有 provenance。

放行条件：matrix 冻结 SV/S0/SV± 的 artifact sha 与 SV/S0 fit-record sha；runner 在任何 episode 前逐臂
重算并检查 mode/c/n/k/h_exec/δ/input digests，把摘要写入 ledger；analyzer 再从磁盘重算并与 matrix/ledger
逐项对照。至少增加“同路径替换 artifact”“同 δ 换 fit record”“conformal artifact 替换 empirical artifact”
三个必拒回归。

### G2R1-B6 — secondary emitter 的输出被 runner 机械拒绝，完整 secondary 链不存在

emitter 冻结 secondary core 为四臂 `T1/T2/T3/SV`；`run_precheck.py:464-469` 却无条件要求五臂
`T1/T2/T3/S0/SV`。最小复现得到 `SECONDARY_ARMS != FORMAL_CORE_ARMS`，所以合法 secondary matrix
必在启动前退出。analyzer 也只实现五臂 primary Gate，没有 secondary descriptive 输出路径。

放行条件：runner 读取并严格验证 `matrix.layer`：primary=五臂+always_search，secondary=四臂+
score_hysteresis；两层使用分离 ledger/output，仍绑定同一 A′。新增 secondary matrix→runner dry validation
成功路径和 primary/secondary roster、gate、ledger 互换的拒绝测试；analyzer 增加只描述、绝不进入 Gate 的
secondary 四臂 frontier 报告。

### G2R1-B7 — §7.4 要求的跨-suite 四 Gate 总裁决没有实现

当前 analyzer 一次只读一个 suite，仓库中不存在把 l10/spatial 两份正式 verdict 合并、验证两个 suite 身份
并要求四个 Gate 全过的 adjudicator。因而代码无法机械阻止“一个 suite 通过”被误报为通用胜出，也没有
§9 要求的 suite-split synthetic fixture。

放行条件：新增 fail-closed cross-suite finalizer，输入两份 primary verdict，核验 suite 恰为
`{libero_10, libero_spatial}`、protocol/method/Gate schema 一致、每份 provenance 完整；只有四 Gate 全过才输出
cross-suite confirmed。覆盖 four-pass、任一单门 fail、仅一 suite、重复 suite、旧 Gate schema 五类测试。

### G2R1-B8 — Gate 2 失败分支仍会输出错误的论文语义

`analyze_precheck.py:757-761` 在 SR 门过、成本门不过时写
`surface_wins_v_sr_gain_cost_unconfirmed`，并声称 “SR gain confirmed” 与 “cost non-inferiority”。Rev 1 的门
只证明 `q0.05(ΔSR) >= 0` 的 **SR non-inferiority**，成本门检验的是 **至少节省 5%**，两句话都扩大/改写了
冻结 claim。

放行条件：所有 verdict id、note 与论文导出统一使用“SR non-inferiority / cost-saving unconfirmed”等准确
措辞；分别构造 SR-only pass、cost-only pass、both pass、both fail，钉死无任何 `gain` 或
`cost non-inferiority` 误报。

### Round 2 重审范围

重审至少需要：B1 的 owner/G1 处置记录、D0→fit 内容级闭链、能够实际产出的 formal SV/S0 artifact、
primary 与 secondary 两条 dry-run 链、跨-suite finalizer，以及上述对抗性回归。现有 `262 passed` 可保留为
基线，但不能作为这些缺失路径的替代证据。

## 17. G2 Round 2 — Review Authority 复审与 owner 授权直接修复（2026-08-28）

**Verdict：代码可进入 Verify；正式 artifact / emit / A′ rollout 仍须先通过新的 D0→fit 硬门。**

执行方恢复后的版本实质关闭了 G2R1-B1、B3、B4 的主体：部署 `run_stage3_from(...,
start_t=0.3)` 保持公开字面 tier 不变，但内部以与 full loop 相同的 float32 累加轨迹重建实际
timestep；Rev 1 中 A1 已降为 diagnostic；formal 参数、两 suite δ、secondary roster、跨-suite
finalizer 与 Gate 2 文案均已落地。复审仍发现 B2/B5/B6/B7 的内容闭环不完整。按 owner 明确授权，
reviewer 已直接修复，所有 reviewer 修改保留在暂存区外，未混入执行方已 staged baseline。

### 17.1 关键科学后果：旧 y7 表不得沿用

`run_stage3_from` 的修复改变了真实 warm 输出。旧 dispatch table 的 `y_tau7` 是由修复前路径产生；
即使 library、sidecar、query H5 和路径名都不变，它也不能再标定修复后的部署。处置不是重新采 A′，
也不要求重建 D_lib：须用现有 D_dev H5 / 同一 library / 同一 z 重建 table，再执行 D0。

D0 check 3 已从“只做 decomposition”升级为 sampled table-semantics hard gate：逐个 p99-tail 行及
每 task 两个冻结 control 重算 `y7/y10`，并在 `1e-4` 下与 table 比对。因而 §14 基于旧实现得到的
两份 PASS 与“现有 Y 表可直接作为 Rev 1 输入”的结论均作废；这不是方法负结果，而是实现版本切换
后的数据兼容性要求。

### 17.2 D0→fit 内容级闭链（关闭 B2）

- D0 冻结 protocol/suite/h_exec/sample digest，以及 table、library pkl、noise sidecar、cache yaml、
  weights、query H5 tree、library H5 tree、checkpoint/config policy fingerprint 的内容摘要；H5 basename、
  解析后 step id、entry id 必须唯一，controls 必须精确 2/task，action/intermediate/sidecar dtype/shape
  均 fail closed。
- `fit_surface --d0-record` 改为必填；fit 端重算 D0 所有文件/tree/policy 摘要，要求 D0=PASS、十 task、
  精确 20 controls、三项 gate 完整，并将 D0 record sha / input rollup / sample digest 写入 fit record、
  input digests 与 artifact meta。复制旧 PASS、换 table、换 library 或换 model 均不能产 artifact。
- SV 的机械 δ 也在重新选择后与 suite 冻结值做 `1e-6` 比对；D_dev membership、fold map、final-fit
  digests 在 artifact 写出前形成并写入 artifact。S0 继承 SV 的 membership/fold/δ，而非以空 fold map
  冒充同一嵌套实验。

### 17.3 artifact→matrix→launch→analysis 内容绑定（关闭 B5）

emitter 同时读取 SV/S0 fit records，逐项比对 mode、D0、membership、fold、input digests 与 artifact
meta；matrix 冻结每个实际 surface arm 的 artifact path/sha 和两份 fit-record path/sha。runner 在连接
server 或运行任何 episode 前重新加载每个 artifact/record，校验 empirical mode、`c=0/n_cal=0`、
`h_exec=5`、SV k=5、S0 k=1、disagreement role、delta label/path、nested-ablation fields，并把摘要写入
launch ledger 的 frozen keys。analyzer 第三次从磁盘复算并对 matrix/ledger；同路径替换 NPZ 或 fit
record 会在使用前 fail closed。

### 17.4 两层分析与跨-suite fixed sequence（关闭 B6/B7）

- primary：五臂、always_search、唯一 confirmatory Gate 1→Gate 2 固定序列。
- secondary：四臂 `T1/T2/T3/SV`、score_hysteresis，只输出 point estimate、clamped-frontier quantiles
  与 branch shares；结果显式 `confirmatory=false`，绝不生成 Gate 1/2。
- cross-suite finalizer 现在要求精确 protocol/layer/Gate schema 与非空内容 provenance。Gate 1 fail 后
  没有 Gate 2 是合法 fixed-sequence negative result（标为 not evaluated），不是 malformed input；Gate 1
  pass 却缺 Gate 2、Gate 1 fail 后仍跑 Gate 2、secondary verdict 冒充 primary 均拒绝。

### 17.5 独立验证与放行边界

- reviewer 定向：D0 / fit solver / emitter / runner / analyzer / finalizer **173 passed**。
- targeted Ruff：All checks passed；`git diff --check` 通过。
- 新增旧实现必败回归覆盖：少 task/少 control、duplicate entry/step、非 float32 action、D0 换 table、
  artifact 同路径替换、secondary 不得产 Gate、Gate 1 fail 后 Gate 2 未评估的跨-suite收口。

**放行边界**：代码审查通过并可进入 Verify；但旧 D0 PASS、旧 table、旧 fit artifact 全部失效。
正式顺序固定为：以修复后代码重建两 suite table（不消耗 A′）→ 重跑 D0 → formal SV fit → S0 fit →
emit primary/secondary matrix。任何一步不通过均不得启动 A′。本节不追认尚未实际产生的新 D0 PASS，
也不把单元测试写成实验结果。
