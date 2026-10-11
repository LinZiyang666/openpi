# R17 纯缓存 × 修正器 × 长执行段 × 停滞门：预注册

**有效修订顺序：** 下文保留首版的待确认事项与接入快照，不能据此覆盖后续决定。[OWNER_RULINGS.md](OWNER_RULINGS.md) 已明确使用 **B 库、官方 A 测试**、复用原控制和 Q3 停滞门；§9 在查看任何 size<500 的评测 outcome 前登记六尺寸扩展，覆盖首版“仅 B500”的范围。首版 §8 的两项问题已解决。

2026-10-10。**本文件是设计，不是已启动或已通过资格验证的 campaign。** 本轮只读代码、历史报告与库/控制组元数据；未启动 GPU、未操作进程或队列。启动条件是正在运行的 horizon/step 研究实际收口，不能把预计 17:45 当成设备空闲证明。

owner 原话：「这个实验做完了之后对所有新模型跑 纯cache组、纯cache组+修整器组、纯cache组+长horizon组、纯cache+长horizon+修正器组（这之前的都不用miss），然后前面的全部重复一次加上停滞门，带旋钮的暂时不跑，每组仍然是全B集500episode，这几组你都可以只编写yaml配合concurrent server的热切换快速实验，如果不行就是你没有把模型充分接入框架」。仅 `libero_spatial`、`libero_10`。

**冻结意图：** 三模型、B500 完整库、八个逻辑组；前四组模型补救调用为零，后四组只允许停滞门触发补救，全部关闭周期/目标 IR 旋钮。**评测池暂按原话登记为 B500 开发性评测；不得静默换成 A。** 但这与复用指定的 A 控制组做配对检验不相容，须由 coordinator 向 owner 确认第 8 节问题 1。下文完整登记 B 与 A 两条条件分支，不按成绩选择分支。

## 1. 历史语义：不能把三个“纯缓存”实现混成一个

| 证据 | 实际配置与运行语义 | 本轮继承/区别 |
|---|---|---|
| cache-prune 原始 CP1/d1 | `gate.type=always_search`、`judge.type=always_hit`、`search_strategy.type=weighted_score_sum_knn`、`top_k=1`、`task_scoped=true`、`trajectory_depth=1`、`step_filter=all`、`write_policy.type=never`。库为空时 `always_hit` 仍可能 MISS，所以验收必须逐决策证明 FULL_HIT。 | 继承“无门限拒绝、无动作头补救、库只读”的定义；**不继承其 1-NN、旧模型固定融合权重或剪枝库**。[cache_prune_plan.log.md:75–85](/home/weiland/projects/openpi/logs/cache_prune_plan.log.md:75) |
| R10 A → R15 纯缓存 | `SizeController(size=500,variant='A')` 包装 `BlindAWM`；`base_kwargs={lib:big,kref:8,serving:anchor_tail,budget:1,gates:budget_only}`，固定 16 个 AWM 邻居；R=5 加一个无视觉尾段，名义 C10。 | 本轮使用共享的 Mahalanobis + AWM 路径，**不是每次选唯一最近动作**；短段另登记为 R17 默认 C5，故不是声称逐字节复现旧 C10。[r10/build.py:51–79](/home/weiland/projects/openpi/exp/offline_search/rounds/r10/build.py:51)；[blind_awm.py:45–58](/home/weiland/projects/openpi/exp/offline_search/rounds/r04/k1_blind/blind_awm.py:45)；[r16/ADAPTER.md:7](/home/weiland/projects/openpi/exp/offline_search/rounds/r16/ADAPTER.md:7) |
| R15 A + corrector | `PureCorrector(size=500,corrector_enabled=True,guard_enabled=False)`；复制同 cell/size 的冻结 **GC_dist** heads、sigma、distance scale，保持 A 检索/合成/尾段；没有 guard、没有 policy-tail/judge 插件标志。不是 GC_pair，也不是从测试数据新训练修正器。 | 对应 G2/G4；新模型用自己 B500 拟合的同种数学方法，不能套 π/GR 权重。[pure_corr/HANDBACK.md:14–30](/home/weiland/projects/openpi/exp/offline_search/rounds/r15/pure_corr/HANDBACK.md:14)；[runtime.py:45–85](/home/weiland/projects/openpi/exp/offline_search/rounds/r15/pure_corr/runtime.py:45) |
| R10/R15 G、GC_dist | 非 A 臂附 `--os-policy-tail --os-policy-tail-blocks 1 --os-judge guard_only`；只启用 no-progress，禁用 stuck/terminal/overtime。 | 对应 G5–G8，只允许停滞 MISS；不是“仍零 MISS 但只记一条报警”。[r10/build.py:70–76](/home/weiland/projects/openpi/exp/offline_search/rounds/r10/build.py:70)；[r09/recipe.py:3–9](/home/weiland/projects/openpi/exp/offline_search/rounds/r09/recipe/recipe.py:3) |
| R16 | 执行 C10，改 `miss.num_steps`；库、metric、corrector、guard 冻结。预算 descent 的 `off` 只是周期旋钮关，**不是 guard 关**。B30/task 做开发选择，冻结后 A500/suite 测试。 | 本轮不做 descent、不读取 task caps/IR 表、不叠加减步：若门触发，DP K100、Fast K10、Cosmos K5。[r16/ADAPTER.md:7–12](/home/weiland/projects/openpi/exp/offline_search/rounds/r16/ADAPTER.md:7)；[r16/REPORT.md:3,39–40](/home/weiland/projects/openpi/exp/offline_search/rounds/r16/REPORT.md:3) |

旧 YAML 与插件必须分开理解：R15 实际复制的 A YAML 外壳是 `always_search` + `threshold:-1000000` + `weighted_score_sum_knn/top_k:1`，真正 AWM/修正器由插件接管；因此只复制这个 YAML 的 `top_k:1` 会复制错算法。[原 A YAML:24–35](/home/weiland/trace_runs/os_closed_loop/r10_size_pi05/config/r10_pi05_l10_50_A.yaml:24)；[pure_corr/build.py:199–202](/home/weiland/projects/openpi/exp/offline_search/rounds/r15/pure_corr/build.py:199)。历史补充消融本身已完成：纯缓存 84.0%，加修正器 87.2%，只守卫 89.0%，守卫+修正器 90.9%，是四 cell、六库大小合并，**不能当作新模型的预期 SR**。[session_handoff.md:493–510](/home/weiland/projects/openpi/logs/session_handoff.md:493)

## 2. 八组与精确 YAML 旋钮

记 `q=recipe.corrector`，`g=recipe.guard`，`C=adapter_binding.commit_horizon`。所有组共同使用：

```yaml
# 对对应模型的身份完整配置做结构化补丁；不是一份可独立启动的完整 YAML。
enabled: true
write_policy: {type: never}
checkpoints: {}                 # 由 R17 recipe 执行检索与门；不叠加旧 CP1 judge
recipe:
  schema: r17.recipe.v1
  enabled: true
  store_path: <本模型、本套件、经身份核验的B500库路径>
  store_digest: <该库READY.content_digest>
  teacher_sampler_identity: <建库满步sampler的digest>
  fit_identity: <按当前schema计算；不是直接填fit文件digest>
  ordered_tasks: <原有40项canonical任务表，原序不动>
  task_table_identity: <原任务表identity>
  corrector: false              # 按下表替换
  guard: false                  # 按下表替换
  periodic_gap: null            # 不能填0：0不是关闭
  target_ir: null
  ir_costs: null
  k: 16                        # 检索邻居数；不是去噪K
  kref: 8
  early: true
adapter_binding:
  commit_horizon: <C>
  allow_partial_commit: <C % 模型原R != 0>
  # 其余model/checkpoint/norm/layout/feature/protocol/sampler身份从合格配置继承
```

字段均来自现有 `RecipeConfig`；`fit_identity` 包含库、teacher、layout/features、C、corrector、k/kref/early、任务表。`checkpoints:{}` 时旧 orchestrator 的检查结果不决定 recipe 的动作：有效 proposal 且未 force_miss 就走 library；否则走模型。必须按下文的实际 source/NFE 记账，不能把旧 CP1 的空检查计为一次真实 MISS。[common/config.py:12–35](/home/weiland/projects/openpi/exp/r17_integration/common/config.py:12)；[common/recipe.py:60–69](/home/weiland/projects/openpi/exp/r17_integration/common/recipe.py:60)；[common/serving.py:325–350](/home/weiland/projects/openpi/exp/r17_integration/common/serving.py:325)

| 组 | YAML 变化（上述公共配置之外） | MISS | corrector | 停滞门 | C：DP / Fast / Cosmos | 历史对应 |
|---|---|---|---|---|---|---|
| G1 纯 cache | `q:false,g:false,C:C0` | **禁止** | 关 | 关 | **5 / 5 / 16** | R15 A 的短段版本 |
| G2 cache+修正器 | `q:true,g:false,C:C0` | **禁止** | GC_dist 方法 | 关 | **5 / 5 / 16** | R15 AC_dist 的短段版本 |
| G3 cache+长 horizon | `q:false,g:false,C:CL` | **禁止** | 关 | 关 | **L_DP / L_FW / 16** | A + 长 anchor-tail |
| G4 cache+长 horizon+修正器 | `q:true,g:false,C:CL` | **禁止** | GC_dist 方法 | 关 | **L_DP / L_FW / 16** | AC_dist + 长 anchor-tail |
| G5 G1+停滞门 | `q:false,g:true,C:C0` | **仅停滞** | 关 | only-no-progress | **5 / 5 / 16** | G，周期旋钮 off |
| G6 G2+停滞门 | `q:true,g:true,C:C0` | **仅停滞** | GC_dist 方法 | only-no-progress | **5 / 5 / 16** | GC_dist，周期旋钮 off |
| G7 G3+停滞门 | `q:false,g:true,C:CL` | **仅停滞** | 关 | only-no-progress | **L_DP / L_FW / 16** | G + 长 anchor/policy tail |
| G8 G4+停滞门 | `q:true,g:true,C:CL` | **仅停滞** | GC_dist 方法 | only-no-progress | **L_DP / L_FW / 16** | GC_dist + 长尾，周期旋钮 off |

**“纯 cache 无 MISS”的执行契约。** 每次新 anchor 提取真实视觉/状态特征，同 task Mahalanobis 排序，取最多 16 个邻居，以 `_kernel_w(distances-distances[0],8)` 归一化混合完整动作块。即使距离大，也不因距离拒绝动作；距离只衰减修正强度。执行后续尾段时复用同一合成块，**不是每个 R 步都重新找最近邻**。G1/G3 的所有已执行动作来自库合成；G2/G4 来自库合成加冻结残差。没有动作头调用、warm start、在线写库或隐藏 fallback；空库、非法值/身份属于无效实验，不能用模型救场后仍报“纯 cache”。视觉特征前向和 CPU 检索/修正仍有成本，NFE=0 不等于完全没有 GPU 工作。[layers/recipe.py:245–271,296–306](/home/weiland/projects/openpi/exp/r17_integration/common/layers/recipe.py:245)；[serving.py:321–365](/home/weiland/projects/openpi/exp/r17_integration/common/serving.py:321)

**修正器。** 用本模型/本 suite/B500 的 library-only LOEO 残差头，motion-only（不改 gripper），每 task 的标准化输入 + 384 RFF、ridge α100、seed0；强度 `0.5*clip((2-r)/1.25,0,1)`，`r=raw_anchor_distance/distance_scale`，step0 不修正。distance scale 用 B 的五折 held-out、按 episode 平衡的距离中位数；这不是 A 适配，也不是“给最近轨迹套一个通用纠偏器”。固定所有参数，不扫描 blend/cap。[layers/recipe.py:137–161,192–229,274–305](/home/weiland/projects/openpi/exp/r17_integration/common/layers/recipe.py:137)；[layers/math.py:7–8,96–125](/home/weiland/projects/openpi/exp/r17_integration/common/layers/math.py:96)

**停滞门。** 用 `NoProgressGuard(n=3,eps=0.5)`：检索 top donor 的归一化轨迹进度，连续区间进度增量乘当前 donor 的 `length-1` ≤0.5 时累积 decision-step span；span≥2 触发 MISS，前次实际 MISS 清空历史；span≥1 可否决盲尾并要求重新看图。它不是机器人位姿速度门，也不使用真实成功标签。阈值来自共享默认与 READY 收据，**不是每库根据 SR 学出的阈值**。长段加门时 C 是最大可承诺长度，门可以提前打断，必须报告实际执行长度分布；不新增不透明的“连续3次相同动作”门。[layers/guard.py:4–28](/home/weiland/projects/openpi/exp/r17_integration/common/layers/guard.py:4)；[layers/recipe.py:287–314](/home/weiland/projects/openpi/exp/r17_integration/common/layers/recipe.py:287)；[teacher_library.py:240–252](/home/weiland/projects/openpi/exp/r17_integration/search/teacher_library.py:240)

## 3. 库、C 和 fit 的冻结规则

使用 `/data/r17_library/{dp_libero,fastwam_libero,cosmos_libero}_{libero_spatial,libero_10}/bpool/` 的完整 **500 episodes/suite，50/task**，不扫 50..400、不按成功过滤、不剪枝。只读 READY 元数据实查如下；这些行数不是有效测试样本数。

| 模型 / suite | B500 rows | 已有 size_500 fit 的 C | fit digest 前12位 | 实物证据 |
|---|---:|---:|---|---|
| DP / spatial | 12,028 | 10 | `0c141049fd68` | [parent READY:1](/data/r17_library/dp_libero_libero_spatial/bpool/READY.json:1)、[fit READY:1](/data/r17_library/dp_libero_libero_spatial/bpool/fits/size_500/READY.json:1) |
| DP / 10 | 31,546 | 10 | `5956e15b3f21` | [parent READY:1](/data/r17_library/dp_libero_libero_10/bpool/READY.json:1)、[fit READY:1](/data/r17_library/dp_libero_libero_10/bpool/fits/size_500/READY.json:1) |
| Fast / spatial | 12,460 | 10 | `aa3440937097` | [parent READY:1](/data/r17_library/fastwam_libero_libero_spatial/bpool/READY.json:1)、[fit READY:1](/data/r17_library/fastwam_libero_libero_spatial/bpool/fits/size_500/READY.json:1) |
| Fast / 10 | 27,806 | 10 | `a021fd5e9721` | [parent READY:1](/data/r17_library/fastwam_libero_libero_10/bpool/READY.json:1)、[fit READY:1](/data/r17_library/fastwam_libero_libero_10/bpool/fits/size_500/READY.json:1) |
| Cosmos / spatial | 3,676 | 16 | `bf58be069b2f` | [parent READY:1](/data/r17_library/cosmos_libero_libero_spatial/bpool/READY.json:1)、[fit READY:1](/data/r17_library/cosmos_libero_libero_spatial/bpool/fits/size_500/READY.json:1) |
| Cosmos / 10 | 8,408 | 16 | `969138458e0c` | [parent READY:1](/data/r17_library/cosmos_libero_libero_10/bpool/READY.json:1)、[fit READY:1](/data/r17_library/cosmos_libero_libero_10/bpool/fits/size_500/READY.json:1) |

每个 parent 的 `rows[].init_pool` 都是 B、500 个唯一 task_uid；size_500 fit 的 `receipt.store_digest` 绑定 **subset selection**，不是 parent 的 content digest，虽都是500集也不能直接互换身份。所有现成 fit 使用完整40项任务表；只筛选 spatial/10 的执行计划，不能重排或压缩 one-hot 表。生成默认 C 的规则为 `max(R,min(10,H))`，解释了现成 DP/Fast C10、Cosmos C16。[teacher_library.py:216–246](/home/weiland/projects/openpi/exp/r17_integration/search/teacher_library.py:216)

| 模型 | 原生 H / 短 C0 | 长 CL 的预定选择 | 门触发时 sampler |
|---|---|---|---|
| DP-T | 16 / **5** | 在 `{10,15,16}` 中，两主 suite 都通过已注册2pp非劣检验的**最长 C**；无合格长 C 时 **10**（R16历史选择，明确标未获本轮 NI 认证） | DDPM K100，实际 NFE照实记 |
| Fast-WAM | 32 / **5** | 在 `{10,16,20,32}` 中按同一规则取最长；无则 **10**（同样未认证） | 默认 K10，实际 NFE照实记 |
| Cosmos | 16 / **16** | **16**，已经执行整个预测块 | 默认 K5，实际 NFE照实记 |

模型几何、默认 C 与候选见 [horizon/STUDY.md:80–82](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/STUDY.md:80)；2pp判定使用该文第(3)节的24比较校正 CP 与 bootstrap 下界、两个 suite 都通过，不能拿汇总 SR 点差选 C。**本次 owner 的“最长通过值”用于缓存研究入组，优先于旧 horizon 方案“首次失败即停”的采用顺序；不回改旧研究结论。** 长 C 的选择及来源分析 SHA 必须在第一局缓存评测前冻结，不能看缓存成绩改 C。复用 A 选择出的 C 再在 A 评测，须注明 horizon 选择复用，不能称为独立 horizon 复现。

Cosmos **G3=G1、G4=G2、G7=G5、G8=G6**：登记八个逻辑标签，实际运行四种配置，每种仍500/suite；报告复用同一组500身份，不能冒充独立重复实验或增大 n。禁止复制最后动作、循环块或把预测长度改成32来制造“长”组。

**C变化也会改变修正器输入/输出维度。** 现成 C10 head 不能直接贴到 DP/Fast C5/C15/C16/C20/C32。执行前以同一 B500、相同 PCA/metric/LOEO 方法生成所需 C 专属 head 与距离标定，写全新 fit generation、冻结 SHA；不改旧 READY，不在评测后训练。G1/G2、G3/G4 各自必须共享相同底层检索数组，关 corrector 只停用头；G5/G6 等只改变 guard。C间头重新适配属于预注册的方法移植，不能把 G4−G2 全部解释为“只多执行几个动作”的纯机械效应。[layers/recipe.py:148–157,203–215,274–278](/home/weiland/projects/openpi/exp/r17_integration/common/layers/recipe.py:148)

## 4. 评测集裁决：B500建库不等于A500测试

| 证据/选择 | 精确含义 | 对配对和解释的影响 |
|---|---|---|
| R15 纯缓存修正器补充 | **official A：10任务×50官方初态=500/suite/arm**，24臂复制原 A500 配对模板；不是 B500。 | 可以和当时的 A/G/GC_dist 做逐初态比较。[pure_corr/HANDBACK.md:3–7,22–30](/home/weiland/projects/openpi/exp/offline_search/rounds/r15/pure_corr/HANDBACK.md:3) |
| R16 | B30/task 是开发下降；phase1/A 与正式 test A 都使用冻结 A500模板。其后存在按 A 结果提档的重测，不能继承为本次合法的预注册选择。 | 明说：**R16正式测试用A，不是建库B500**。[r16/ADAPTER.md:6–12](/home/weiland/projects/openpi/exp/offline_search/rounds/r16/ADAPTER.md:6)；[r16/REPORT.md:39–40](/home/weiland/projects/openpi/exp/offline_search/rounds/r16/REPORT.md:39) |
| cache-prune | A-pool每任务50；实测两suite各500个均为库采集集合外的 common-unseen。 | 也不是用库自身初态证明泛化。[cache_prune_plan.log.md:161–173](/home/weiland/projects/openpi/logs/cache_prune_plan.log.md:161) |
| owner字面B500（当前登记分支） | 使用上述 parent library 的同一500个 B初态；这些初态已经产生库轨迹、metric与修正器训练数据。 | **若直接全库检索，确实包含自己的轨迹，是训练/评测重用；不能当独立测试。** 也不能按“init=0”把 B0 和官方 A0配对。 |
| 若owner确认“全500”指A | B500只建库；评测官方 `--official-50`，仅 spatial/10各500，核验每个init状态字节与库B身份交集。 | 与指定 A 控制组形成500/suite的真实身份配对；仍要解决下面的版本/运行配置差异。 |

**字面B分支的最低合法语义是 R15 风格 serving-LOEO：** 每个查询排除本 task、本 B-init 的整条 library episode，500库存仍保留，供其它query使用；每task实际候选是另外49条轨迹。不是删除一帧，更不能用 query的A/B相同整数索引判断自己。已有 READY 明确要求 `serving_loeo_required_on_B=true`，P6 handback也明确“fit仍含全部B，B serving须排除自身”。即使排除了 donor，PCA/metric/head训练仍用过这个episode，**仍是B开发性评估，不是完整 cross-fit 或独立泛化证据**。完整500份leave-one-out重拟合不属于这轮八组。[HANDBACK_P6.md:496–509](/home/weiland/projects/openpi/exp/r17_integration/handbacks/HANDBACK_P6.md:496)；[r16/ADAPTER.md:7](/home/weiland/projects/openpi/exp/offline_search/rounds/r16/ADAPTER.md:7)

**当前实现缺口必须补，不能伪称现成 YAML 已完成B500接入：** serving调用 `recipe.query(...)` 没传排除身份；而 `query(exclude_episode=...)` 当前会提前返回原始检索，跳过corrector/guard。因此仅“加一个 exclude_episode 参数”会把 G2/G5 等跑错。需在组合query内部排除donor后继续完整修正/门控，按规范 B identity 映射 numeric episode，并有全路径 CPU证明。[serving.py:332–337](/home/weiland/projects/openpi/exp/r17_integration/common/serving.py:332)；[layers/recipe.py:280–308](/home/weiland/projects/openpi/exp/r17_integration/common/layers/recipe.py:280)

## 5. 控制、配对、指标和分析规则

### 5.1 指定控制的使用范围

| 模型 | 指定历史控制 | 冻结运行配置/局限 |
|---|---|---|
| DP | `/data/mw_finetune/eval/hs5dp_ctl_142812`，K100/C5 | local4090，**2副本/B8/5ms/MPS，timan107×32**；使用已封存有效结果，不把早期tether失败直接当任务失败。[RUN_CONFIGS.md:305–340](/home/weiland/projects/openpi/exp/r17_integration/ops/RUN_CONFIGS.md:305) |
| Fast | `/data/mw_finetune/eval/hs6fw_ctl_143332` + `hs6fw_res_152931`，K10/C5 | H100，**2副本/B16/5ms/MPS，timan107×48**；合并1839+161，用Q封存后的唯一2000身份，不能只取首段或重复计resume。[session_handoff.md:100–102](/home/weiland/projects/openpi/logs/session_handoff.md:100)；[RUN_CONFIGS.md:384–417](/home/weiland/projects/openpi/exp/r17_integration/ops/RUN_CONFIGS.md:384) |
| Cosmos | 原 `formal_cosmos_libero_official2000_20261010_100533_5f558380` + resume `formal_cosmos_libero_official2000_20261010_130639_a8ca2043` 的 merged K5/C16 | H100/B1；读取resume目录封存的 SUMMARY、RESUME_MERGE、RECOVERED_RESULT_SEAL。1726局timan108，274局timan107；spatial全108，10有226局108+274局107。不能声称renderer匹配。[HANDBACK_COSMOS.md:903–928](/home/weiland/projects/openpi/exp/r17_integration/handbacks/HANDBACK_COSMOS.md:903) |

若选择A，每个历史控制只抽 spatial/10原始1000身份。按 `(model,suite,canonical task_name,init_pool,original init index,init bytes hash,environment seed/profile)` 配对；arm/YAML/task_uid字符串不同不妨碍配对，A/B同idx却不能配。验证每suite500个唯一完整pair、每task50；保留真实任务失败=0，基础设施错误按固定重试/验收规则处理，缺失不得选择性删掉。

**每卡一个resident server与上述双副本控制不是同一运行配置。** 本设计默认遵守“每卡单resident”，建议另加每模型一个匹配控制：与缓存臂同GPU、batch cap/wait、MPS、worker数/EGL、代码/权重、计时配置；算法仍是DP100/C5、Fast10/C5、Cosmos5/C16。三个指定历史控制仍按同初态给出配对结果，但注明资源/版本与Cosmos renderer差异，不能拿历史R2墙钟当R1缓存节省。若owner要零新增控制并严格沿用既有配置，DP/Fast需保留R2服务拓扑，不能同时声称只有一个物理resident。第8节问题2冻结这一取舍。**任何批处理变化必须同时应用控制和所有组；不得借缓存调用稀疏而给control换batch设置。**

若选择字面B，指定三份A控制仅作背景，不能作为该500对的控制。必须使用同一 B500 初态的新满步控制；建库teacher有capture开销/旧版本，未经核验不能替代。新控制并不建新库、不往现有库写入。不得以cross-pool非配对SR差冒充配对改善。

### 5.2 预先固定的统计与成本口径

| 项目 | 冻结分析 |
|---|---|
| 完整输出 | 每模型×suite×唯一组：n、SR、配对Δ、gains/losses、每task50局；不因中途SR停臂、调门或换C。 |
| 主比较 | 每唯一组 vs 同配置纯推理控制；DP8×2 + Fast8×2 + Cosmos4×2 = **40比较**，独立于此前horizon/step的24比较family。历史控制敏感性另列，不再增加确认性发现。 |
| 若A且控制匹配：2pp规则 | 采用保守配对界 `L_CP=CP_low(g,500,a)-CP_high(l,500,a)`，`a=.05/(2*40)`；按task内成对bootstrap 10,000次、seed20261010，取`.05/40`下分位，共享控制使用同一重采样索引。两个suite的两种下界都严格大于−2pp才记“2pp非劣”；另报未校正95%区间。不能写“零损失”。 |
| 若字面B | 同样输出配对差、g/l、区间与2pp参照线，**只作固定B库条件下的开发性结果**，不发独立测试NI证书；LOEO donor规则和fit重用必须随结果呈现。 |
| 消融效应 | 逐suite配对：corrector G2−G1/G4−G3/G6−G5/G8−G7；long G3−G1/G4−G2/G7−G5/G8−G6；gate G5−G1/G6−G2/G7−G3/G8−G4。全表作描述性分析，不挑最好一个另作未校正显著性主张。Cosmos long效应是结构性相同，不估计四次独立效应。 |
| NFE/decision | 以sampler实际前向receipt为准。分别给fresh-anchor与全部已验收R步请求两个分母：`Σactual_nfe/N_fresh`、`Σactual_nfe/N_requests`，再给`Σactual_nfe/实际执行controls`。G1–G4所有动作头NFE必须0；G5–G8仅guard触发新policy调用，policy tail自身NFE0。 |
| hit/miss | fresh HIT=`source=library`；MISS=`source=policy`且guard=true；另列library_tail/policy_tail。给fresh hit/miss率与全部请求来源占比，禁止把blind tail重复计检索命中，或拿内部空CP1结果当实际MISS。记录非guard MISS、warm starts、空库/越界均须0。 |
| 实测成本 | decisions/episode、fresh retrievals/episode、stage1/2/3 calls、NFE/episode、实际C分布、corrector engagement/strength、guard triggers/tail veto；服务时间区分prepare/features/retrieval/corrector/sampler、queue wait，给每请求与每实际control、每episode、共同成功pairs。墙钟/episode throughput单列。计时缺失就写缺失，不套π/GR的IR系数，不把NFE降幅当总算力降幅。 |
| 采用结论 | 只有有效A配对的SR门通过、且两个suite实测总服务时间/执行control下降，才称“在2pp界内省成本”；其他情形分别报SR证据不足、成本待测或B开发结果。本次不据A结果逐任务提档。 |

逐条决策源码与cost字段来源：[serving.py:272–290,325–365](/home/weiland/projects/openpi/exp/r17_integration/common/serving.py:272)。Cosmos历史比较须补报同timan107的274对与其余对，并列任务贡献；新全timan107控制若补齐，则以新控制为主，历史分层作为敏感性。

## 6. 热切换已有什么、sol还必须接什么

模型入口的concurrent连接工厂与 `load_cache_config` 已有；同一模型的权重常驻，每connection只有自己的recipe/guard/历史；同bundle id禁止装不同内容。新bundle可异步准备，旧episode继续绑定旧bundle，下一局才 `select_bundle`。因此**八组的算法切换可以表达为上面三个YAML字段**；目前缺口是下面的artifact/identity/runner接线，不能把默认纯推理launcher当成已经能跑缓存的runner。[common/serving.py:74–111](/home/weiland/projects/openpi/exp/r17_integration/common/serving.py:74)；[common/websocket_policy_server.py:44–75,127–165](/home/weiland/projects/openpi/exp/r17_integration/common/websocket_policy_server.py:127)；[models/cosmos/serve.py:29–37](/home/weiland/projects/openpi/exp/r17_integration/models/cosmos/serve.py:29)

| 当前代码/元数据事实 | sol需要完成的最小接线（本次不改运行代码） | CPU验收要求 |
|---|---|---|
| `common/recipe.py`只从parent Store重新fit，不消费现成fit文件；没有 `fit_path` YAML字段。P6有 `load_fitted_recipe`，但现成size500绑定indexed subset。[recipe.py:141–181](/home/weiland/projects/openpi/exp/r17_integration/common/recipe.py:141)；[fit.py:132–172](/home/weiland/projects/openpi/exp/r17_integration/search/fit.py:132) | 在共享RecipeConfig/工厂添加**默认None**的frozen fit引用（建议`fitted_recipe_path/digest`，明确是待加字段，不能现在放入运行YAML）；验证parent→subset→fit选择身份、C/head维度、任务表、k/kref/early及数组哈希。提供indexed-store加载或等价只读映射。共享一次加载，connection fork仅重置可变状态；关修正器应复用同metric并停用头。 | 冻结fit reload动作/donor字节一致；corrector-off保持base相同；错C、错model、错subset、坏SHA失败；相同fit不同guard只fork不重fit。禁止首次每个连接重做大规模fit。 |
| 当前所有库的benchmark binding与指定official50控制不同；DP/Fast还存在checkpoint/feature-policy/code identity差异。 | 核验真实权重文件和native feature/sampler实现，给**显式、内容证明的兼容迁移**，保存旧库只读；A/B split只允许经证明的protocol迁移。不能删除identity校验或手改READY的digest。若权重/特征真变了，则现库不能代表当前模型，须新generation建库/fit并重算预算，不能冒称只换YAML。 | 相同字节的身份迁移证明及不兼容fixture拒绝；加载当前模型+旧库应在未提供证书时拒绝。 |
| B500 composed LOEO缺失，详第4节。 | 在完整recipe query路径接入B canonical identity→own episode排除；A查询不因相同idx排除B episode。不得调用当前提前返回raw的接口假装完成。 | 强制自身最近的fixture中，排除整episode且corrector/guard仍生效；A0与B0不混；concurrent/reset无串历史。 |
| `formal_driver.plan()`明确拒绝`config.enabled/config.checkpoints/recipe.enabled`；现有正式launcher写死full.yaml/formal_full_v1。[formal_driver.py:41–81](/home/weiland/projects/openpi/exp/r17_integration/ops/formal_driver.py:41) | 为共享conductor新增默认关闭的cache manifest模式（或独立cache runner），明确arm/bundle身份和各1000局计划；使用`episode_tasks`原任务表，只筛spatial/10、不重编号；独立验收cache/guard/tail/NFE，不能调用纯推理验收器强求每局全部policy。复用资源admission、accepted-attempt、终局执行ACK与resume机制，不另造模型评测循环。 | 每组两suite各500原始身份；Cosmos别名只生成4物理组；切换时旧episode继续旧bundle、新episode到新bundle；noprogress-only强制MISS和零MISS组验收均测；C16/32部分尾、terminal ACK、重试去重均测。 |

身份差异的具体只读快照：DP library protocol `589fd029…` / official `87e3a7dc…`，checkpoint `b3eaa7cc…` / `ed3ae8c6…`；Fast相同protocol差异，checkpoint `9f8304f7…` / `fa3f174e…`；Cosmos protocol `dea3db8f…` / `cfcd22a0…`，其余剔除capability/default_sampler后的模型字段相同。**这证明当前严格工厂会拒绝，不足以单独证明底层权重确实不同**；DP/Fast的feature差异在`policy_digest`。来源：[DP library:1](/data/r17_library/dp_libero_libero_spatial/bpool/READY.json:1)、[DP model_spec:1](/data/mw_finetune/eval/hs5dp_ctl_142812/assets/model_spec.json:1)、[Fast library:1](/data/r17_library/fastwam_libero_libero_spatial/bpool/READY.json:1)、[Fast model_spec:1](/data/mw_finetune/eval/hs6fw_ctl_143332/assets/model_spec.json:1)、[Cosmos library:1](/data/r17_library/cosmos_libero_libero_spatial/bpool/READY.json:1)、[Cosmos model_spec:1](/data/mw_finetune/eval/formal_cosmos_libero_official2000_20261010_130639_a8ca2043/assets/model_spec.json:1)。严格比较逻辑：[recipe.py:91–97,148–157](/home/weiland/projects/openpi/exp/r17_integration/common/recipe.py:148)。

协调器的热切换顺序（控制协议是真实接口；占位值需在上述接线/冻结后生成，**不是当前可发的campaign命令**）：

```json
{"__ctrl__":"load_cache_config","yaml_path":"<server-visible frozen arm.yaml>","yaml_id":"<arm>","bundle_id":"r17cache/<model>/<suite>/<group>/<sha12>","bundle_identity":"<computed>"}
```

先检查ACK身份，再在上一episode的最终execution ACK与episode_end成功后发 `{"__ctrl__":"select_bundle","bundle_id":"<same>"}`，按冻结plan启动下一局；不得半局换YAML。先预载同suite所有组及所需fit，避免试验期间CPU fit占用服务线程；预载RAM不足就按C成组顺序装载，仍保持权重常驻。语义相同的Cosmos别名不能靠不同campaign RNG生成“独立长段组”。

## 7. 两卡顺序、worker与episode预算

**默认排程保持指定控制的GPU类型：DP在本机48GB RTX4090；Fast、Cosmos在80GB H100。** 同一模型全部组留在同卡；每卡每时刻一个resident模型服务，同模型组之间只换YAML。Fast两suite8组全部结束后，H100才换一次Cosmos并完成4个唯一组；不为每组重启权重。DP与Fast并行，空下来的4090不自动迁移Cosmos，因为会破坏与H100 merged K5的GPU类型配对。若批准新增4090 Cosmos控制，可另登记迁移以缩短总墙钟，本版不默认增加这条跨卡分支。

| 顺序/并行lane | GPU / resident | timan107 workers（计划值，启动前admission） | 每suite顺序 | 缓存局数 |
|---|---|---:|---|---:|
| lane L，与H同时开始 | local4090：DP R1，batch cap8 / wait5ms；全部臂固定同MPS设置 | 24 | 可选匹配control → G1→G2→G3→G4→G5→G6→G7→G8；先spatial后10 | 8,000 |
| lane H 第一段 | H100：Fast R1，batch cap16 / wait5ms；全部臂固定同MPS设置 | 24 | 同上；先spatial后10 | 8,000 |
| lane H 第二段 | H100：Cosmos R1/B1 | 24 | 可选匹配control → G1→G2→G5→G6；先spatial后10；其余四组记别名 | 4,000 |

R1新控制路线统一MPS off；若选择保留R2历史控制路线，则DP/Fast必须保留原MPS、B8/B16、worker数与renderer配置，按其已有资源收据重新排程，不用本表R1配置冒充。所有baseline/arm都关闭debug capture；计时配置必须一致。P3测得Fast H100 R1/B16吞吐16.8 calls/s、R2/B16+MPS为28.0，**因此“每卡R1”不是硬件绝对最快**；本表优化的是owner规定R1且同GPU配对条件下的加载/调度开销，不能给出未经测量的全局最优承诺。[session_handoff.md:100](/home/weiland/projects/openpi/logs/session_handoff.md:100)

worker只用timan107。计划两路24+24=48核，按每worker **4096MiB** 预留共192GiB，另保留 `max(8GiB,10% node/cgroup)` 和既有占用；EGL按8张卡的实际可用显存逐卡admission，DP按不少于576MiB、Fast不少于768MiB/worker预留，Cosmos沿其合格收据，不能只看总CPU内存。约250GB是主机总RAM，不是可用RAM；**只有启动前admission允许才采用24+24**。不允许绕过检查来重演两fleet24导致的OOM。若不能入场，先按内存/renderer限制冻结较少worker及匹配control，再启动任何组；不得看到成绩后改并发。[HANDBACK_Q.md:1107–1129](/home/weiland/projects/openpi/exp/r17_integration/qualification/HANDBACK_Q.md:1107)；[RUN_CONFIGS.md:315–319,394–417](/home/weiland/projects/openpi/exp/r17_integration/ops/RUN_CONFIGS.md:315)

**时间预算只能先给可测公式，不能拿纯推理NFE推造cache小时数。** 现成fit预加载/新C fit为CPU准备；实际缓存快慢还取决于视觉前向、B500检索/修正CPU、guard率和episode长度，当前没有这20个组的合格吞吐收据。冻结后用按既定顺序完成的首批episode只更新耗时预测，不据SR改组/停止：`T_arm,suite=500 / 有效episodes每小时`；`T_local=ΣDP T`，`T_H100=ΣFast T+ΣCosmos T+一次换模时间`，总墙钟约`max(T_local,T_H100)`，另加控制与准备。保留实际shared-node排队时间；不把P3的calls/s直接乘成episode/s。所有GPU执行命令由coordinator在ready配置上生成和运行，本文件不向现有queue追加内容。

| 预算 | 计算 | episode |
|---|---|---:|
| owner名义八组 | 3模型×8组×2suite×500 | 24,000标签覆盖 |
| 实际不同缓存配置 | (DP8+Fast8+Cosmos4)×2×500 | **20,000新局** |
| 若R1路线补匹配control，或字面B需要B控制 | 3模型×2suite×500 | **+3,000** |
| 推荐R1完整预算 | cache + 3个匹配control；历史A控制只读复用 | **23,000新局** |
| 若owner确认A且保留合格历史运行配置、接受Cosmos旧renderer限制 | 无新增control，保留历史配对与限制 | **20,000新局**；成本可比性仍须实测核验 |

每个物理组每suite始终500；Cosmos别名复用500对而不是额外500。前置CPU验收无episode；必要资格smoke另列，不纳入500、不自动授权GPU启动。若身份审计证明必须重建库，其采集量必须另报，不能藏在上述20,000内。

## 8. 给owner的开放问题（两项）

1. **“全B集500episode”确实指建库用的B500，还是指像R15/R16一样“用B500库、在官方A500评测”？** 建议后者以便独立测试并配对指定A控制；若字面B，则本版按serving-LOEO跑开发评测、必须配B控制，并披露fit仍用过这些初态。
2. **“每卡一个resident server”是否严格指一个模型副本？** 若是，建议本版R1并补三模型各1000局同配置control（总23,000）；若优先零新增控制和既有吞吐，则DP/Fast保留原R2/B8、R2/B16/MPS/worker配置（总20,000），不要把R1当成同一控制条件。Cosmos继续H100；只有另明确批准4090匹配控制时再改两卡分工。

无需另问的已定事项：前四组零MISS；后四组只停滞MISS；K保持100/10/5；不跑旋钮；B500全库；C按第3节规则冻结；Cosmos长段组结构性合并。以上两项由coordinator在首局前记下owner答复和冻结版本，不以任何缓存结果反推答案。

## 9. 六尺寸扩展：先登记、后看 size<500 结果

**登记时间：2026-10-10 20:12 CDT；本次截至登记只读已有设计、代码和污染审计，不读取任何 size<500 的评测 outcome。** Owner 新决定：所有模型、所有组重复 `N ∈ {50,100,200,300,400,500}`，每组每 suite 仍为官方 A 的原始 500 局。本节不是启动指令；源数据只读，不操作 GPU、campaign 或队列。

### 9.1 尺寸因素与不变因素

| 因素 | 冻结规则 |
|---|---|
| N 的含义 | 每个模型、每个 suite 的 **B 库 episode 数**，不是评测局数、成功轨迹数或库行数。每 suite 10 个任务，各任务分别取 5/10/20/30/40/50 个 B episode；使用已封存的 `subsets/size_N/VIEW.json`、episode_ids、row_indices 和 parent digest，六级必须嵌套。沿 R10 的 `PCG64(seed=20261002+suite-local-task-id)` 排列各任务50个 B episode 后取前 N/10；三模型选择身份相同。包含所选 episode 的全部合格行，不按 A 成绩或 B 成败另选。[library_alignment.py:109–129](/home/weiland/projects/openpi/exp/r17_integration/ops/library_alignment.py:109) |
| 库与拟合 | 三模型分别使用自己的 B 库。每 N×C 的 Mahalanobis/投影、corrector heads、distance scale、stall 校准均只用该 N 的 B episode；保留已有 LOEO/分折方法与超参数。**不能用 N500 拟合的 metric/head/stall 配 N50 检索后声称是 N50。** 同 N×C 的 corrector/gate 开关组共享冻结 fit，关开关不改变检索基础。 |
| 小库校准不足 | 保留现有代码对 B 成功样本不足的显式门停用/校准不可用规则，报告各任务 B 样本数及门可用性；不得用 A 调阈值或暗中借 N500 校准。若工厂不能合法生成该 N 的 fit，记 BLOCKED，不计成已完成组。 |
| 八组与推理 | 沿用 G1–G8；前四组始终零模型 MISS，后四组仅 Q3 停滞门可 MISS，周期/IR 旋钮全关。DP/Fast 的 C0=5、L=10；Cosmos C0=L=16，G3/G4/G7/G8 分别是 G1/G2/G5/G6 的别名。K 固定 DP100、Fast10、Cosmos5；本轮不叠加减步。 |
| 评测与控制 | `libero_spatial`、`libero_10` 各官方 A50/task=500，所有 N/组保留相同 canonical `(suite, task, init)` 与 init-state 身份。复用 §5 指定 DP `hs5dp_ctl_142812`、Fast `hs6fw_res_152931`、Cosmos merged K5 控制；不得把 B 的 50/100…误作 A 评测子集。 |
| 执行条件 | 保持已冻结 GPU 类型、batch、MPS、sampler、renderer 与 seed 规则；按 N 热切 YAML，但每局绑定一个不可变 bundle。workers 仍仅 timan107。Cosmos 历史控制的 renderer 混合限制继续披露；不据中期 SR 改排程、删臂或提早停止。 |

预算：每 N 有 `(8 DP + 8 Fast + 4 Cosmos) × 2 suites × 500 = 20,000` 个不同评测 episode；六尺寸合计 **120,000**，相对原 N500 新增 **100,000**。按八组标签展开为 144,000，Cosmos 别名不重复运行、不重复计入检验。控制只读复用，不乘以六；失败重试不增加独立样本量。

### 9.2 比较、检验族与结果解释

| 等级/检验族 | 固定比较 | 数量与规则 | 允许的结论 |
|---|---|---|---|
| 原主分析 F500（不改） | N500 每个不同组 vs 同模型纯推理控制，逐 suite | **40**：`(8+8+4)×2`；沿 §5 的 familywise α=.05、2pp NI 与成本条件 | 某组相对纯推理是否符合原采用标准。不得因本次扩展追改原 alpha。 |
| 新尺寸主分析 Fsize | **N50 vs N500，同模型、同组、同 suite** | **40**；单独 familywise α=.05，2pp NI；一组须两 suite 的 CP 和 bootstrap family 下界均 **>−2pp** 才称“最大压缩通过 NI” | N50 相对该组 N500 的保真度；**不等于相对纯推理通过 NI**，两个 2pp 界不能传递。 |
| 描述/探索 | 全六尺寸 SR 曲线；N100/200/300/400 vs N500；所有 N<500 vs 纯推理；同 N 的修正器/门/长 C 消融 | SR 的逐点双侧 95% CP 区间；配对差的未校正单侧 95% CP/ bootstrap 下界。**不另作校正后显著/NI 声明**，不从中挑出“已认证最小 N”。 | 展示剂量关系、损失/收益与候选尺寸，供独立确认；不假定 SR 随 N 单调。 |
| 稳健性 | 固定剔除 32 个近邻 A 初态，重做所有曲线和配对比较 | 与主样本分开标记 sensitivity；不形成额外主检验族，不用其通过替代 full-A 的失败 | 检查接近 B 初态是否驱动结果。 |

F500 与 Fsize 回答不同问题，**各自**控制 5% familywise 风险，不宣称两个族合并仍为 5%（两族联合上界为 10%）。即使只封存一条 lane，分母仍是 40；不随到达结果缩小族。Fsize 未收齐时标记 interim/pending，完整族结果仍须公布，不能只报告通过项。尺寸扩展不增加模型/组选择的确认性声明。

具体统计沿 §5：每 suite 按 canonical task/init 配对，令 `g=arm 成功、reference 失败`，`l=arm 失败、reference 成功`，`Δ=(g−l)/n`。族下界 `LB_CP=BetaPPF(.05/(2×40);g,n−g+1) − BetaPPF(1−.05/(2×40);l+1,n−l)`，g=0/l=n 使用端点 0/1；未校正下界把 40 换成 1。按任务内配对重采样 10,000 次，PCG64 seed=20261010，任务/init 排序，族 bootstrap 取 `.05/40` 分位、未校正取 `.05` 分位；相同 cohort 的臂复用同一重采样序列。报告 discordant g/l 与两种下界；缺失配对不能补零或缩成小样本通过。500 对下多重校正较保守，保留逐比较界帮助解释，不能改主判据。

### 9.3 曲线、成本与近邻敏感性

每 model×suite 一张 **SR vs N 曲线**，八组逐条标记（Cosmos 只画四条并列出别名），N 用真实数值轴，标出逐点 CP 区间及纯推理水平线。未封存点留空，不连线跨缺失点，不对缺失/失败组插值。配套逐组表给 n、SR、vs N500 和 vs control 的配对 Δ/下界、full/sensitivity 标记、固定族和状态；同时导出机器可读 JSON/CSV，不能把跨模型或跨 suite 平均作为主终点。

成本逐 N×model×group×suite（全部 accepted episode）报告：`fresh hit rate = library/(library+policy)`、fresh MISS 数、MISS/已执行 control step、实际 NFE/已执行 control step、calls/episode、NFE/fresh decision，以及可比时的 service/queue/host 时间。`library_tail`/`policy_tail` 不当作新 hit/MISS 或额外网络计算；步数以 execution ACK 的实际控制数为准。G1–G4 必须 MISS=NFE=0；有门组重点画 hit-rate 与 MISS/NFE-per-control-step 随 N 的曲线。所有比率使用合计分子/合计分母，补报原始计数，避免 episode 长度差把 calls/episode 冒充计算节省；无可比时延证据只报告调用/NFE，不能推断墙钟加速。若校准不可用造成门实质停用，单列说明。成功局限定成本仅作补充。

污染审计显示六库均无 A/B 精确初态重叠，近邻标记是距离敏感性、不是泄漏定罪，见 [CONTAMINATION_CHECK.md](CONTAMINATION_CHECK.md)。**固定剔除**审计 CSV 中 `flag=True` 的32个 canonical A身份：spatial11、libero_10 21；full各500，sensitivity分别 **489/479**。不因 N50 没收录原 nearest B 而恢复某个 A，也不随 N 重算排除名单、不删 B/重拟合、不看 SR 选阈值。reference 和每个 arm 使用完全同一排除名单，成本同步限于剩余 episode；任务内 bootstrap 保持各任务剩余样本数。

冻结排除依据（分析入口须校验 SHA256）：

* `CONTAMINATION_DISTANCES.csv`: `35bc907a6da58891e206ac19d12b6fcf42eee4cef77e62f9ddf1df09924337c8`。
* `CONTAMINATION_CHECK.json`: `e695537620daad6665ee282c1c5808f9b59639f9183f2a2e76dacfc8b3bfa30a`。

### 9.4 增量分析入口与封存门槛

新增只读 CPU 入口 `exp/offline_search/rounds/r17/cache/analyze_cache.py`。先读取 lane 的 manifest、plan、receipt 等身份/封存元数据；**只有 PASS（或有明确 recovered PASS 及封存证据）、已关闭、且组/suite 原500身份完整**才读取 accepted journal 与逐步证据。PENDING/未封存 lane 仅列状态，不预览其 SR。不凭目录名猜 N：从冻结 VIEW 的 size、episode_ids、parent 与 fit/recipe 身份核验；每个结果绑定 lane、bundle、receipt 与输入哈希。正常封存与合法 resume 可复用，未证明的重试不能合并或择优。若重复封存同一 cell 且来源不同，拒绝自动选成绩，要求明确 lane 输入。

分析按已封存 lane 增量产出 `CACHE_RESULTS.md/.json/.csv` 和曲线；未到齐 reference 的已封存 arm 可报 SR，配对格记 pending。Cosmos 别名复用结果、不重复计检验。每次重跑输出完整可用快照、待到达 cell、固定族大小、所用证据 SHA 和时间；**允许随到达查看，禁止以此提前宣布完整尺寸研究结束或自适应更改后续设计。** 未封存期只输出 pending 也属于正常结果。

实施验收使用 CPU 合成数据：验证 N/原任务身份、嵌套库与 fit 子集错配拒绝、accepted retry 去重、未封存 lane 不读 journal、配对初态哈希冲突拒绝、尾段不重计 NFE、固定32排除、Cosmos别名与固定40族。首次实际运行只选择 `--sizes 500`；本次不读取 size<500 outcome。后续 coordinator/分析员显式传入新 lane 与 `--sizes 50 100 200 300 400 500`，不由本入口启动任何实验。

尺寸白名单约束整个 lane：若同一共享 ledger 含未允许尺寸，整条标为 `SKIPPED_UNSELECTED_SIZE`，不先读行再筛结果。resume 按冻结 `source_pins` 与 retained UID 递归继承，最终只有一个来源可占同一 model/N/group/suite；不得同时传入已被完整 resume 替代的旧 PASS lane。旧三个纯推理控制没有后来新增的计划 pool 标签和嵌套 server-terminal 对象，允许沿其已封存 `r17.execution.v1` ACK 验证；原500 canonical身份、运行时 init字节hash、C/K、accepted attempt、控制数和client计数仍须匹配。新 cache lane 保持完整 server-terminal ACK 要求。

```bash
.venv/bin/python exp/offline_search/rounds/r17/cache/analyze_cache.py --sizes 500 --lane /data/mw_finetune/eval/csl10d --lane /data/mw_finetune/eval/csl10f --lane /data/mw_finetune/eval/csl10c
# 后续只读分析（将明确的已准备/封存 lane 路径追加到 --lane；不自动搜读新尺寸 outcome）：
.venv/bin/python exp/offline_search/rounds/r17/cache/analyze_cache.py --sizes 50 100 200 300 400 500 --lane <size500-lane> --lane <size50-lane> --lane <other-explicit-lane>
.venv/bin/python exp/offline_search/rounds/r17/cache/test_analyze_cache.py
```

### 9.5 实现验收（不含 size<500 评测数据）

* CPU 合成测试覆盖上述9项路径，包括完整500身份的合成封存 lane，生成 full/sensitivity 表格和 CSV；三模型实际 N500 的48个 YAML/10份不同 fit 与对应 B VIEW/校准元数据通过只读校验。
* 三个既有纯推理控制各1000个主套件身份经新入口的 ledger/ACK/初态hash验证通过。未查看任何 size<500 的评测 outcome。
* 本次显式 N500 增量快照见 [CACHE_RESULTS.md](CACHE_RESULTS.md)，输入 SHA、未到达240个 cell 中的剩余列表及收据观察时间见同名 JSON。运行中的/FAIL/缺收据 lane 只列状态；只有封存PASS才有SR。后续重新运行相同入口即可增量纳入新封存 lane。
