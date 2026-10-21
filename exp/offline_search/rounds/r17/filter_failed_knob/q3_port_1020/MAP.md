# Q3 停滞门移植进 R15/R16：现状地图与方案（10-20，只读调研）

只读调研，没有改代码、没有起进程、没有碰 GPU 和凭据，`/data` 上只读了单个文件。
路径默认相对 `/home/weiland/projects/openpi`，其他代码树写绝对路径。
标注：**[代码]** 表示代码这么写；**[产物]** 表示在运行目录、产物或进程里实际看到的；**[推断]** 表示由代码推出、没有实测。

---

## 0. 结论速览

1. Q3 的标定**只用库，在 CPU 上离线做**：拿库里的成功局做留一（LOEO），取经验 q90 残差和 q10 推进量。不用开发集的局，不用 A 集，也不用 B-val 录制。R15/R16 的老库和老 fit 里已经有全部输入，所以**不需要上 GPU 跑开发集**。
2. R15 和 R16 用的是**同一条判定链**：`R11Knob/R13Knob/TaskDescentKnob → R10Recipe.inner = R8 TriggerCommitJudge`（只开 no_progress，`noprog_span`，n=3，eps=.5），再由插件的 `--os-judge guard_only` 把 `os_force_miss` 变成 MISS。最小接法是**只换 `R10Recipe.inner`**，换成一个 Q3 包装：数组和 base 不变，旋钮层代码也一行不改。
3. 有个关键点：`--os-judge guard_only` 本身只是"转发 os_force_miss"的通用判定模式，R6/R7 的 Q3 控制器当年也是靠它跑的（`rounds/r06/ideation_Q1/method_c/emit_specs.py:27-28`）。65b1cc5f 让它直接退出（`closed_loop/plugin.py:257-258`），**也就把 Q3 在老插件里的唯一 MISS 通道一起堵了**。所以插件需要一个新名字的同语义模式，例如 `stall_only`（改 3 行）。
4. R17 的 `pi05_libero/groot_libero` 路径（04987e6b）**没有库，也没有 Q3 产物**，只有用合成库跑的 CPU 测试。formal-A 的机器也没把这两个模型接进来，**不能直接跑**。推荐走"R16/R15 判定器换成 Q3"这条路。
5. 不影响 fa_cs_pi_r 的前提：①只新增文件；②`plugin.py`（两棵树各一份）和 `rounds/r16/analyze.py:83`、`rounds/r16/conductor_analysis.py:44` 这几行改动，等 fa_cs_pi_r 和 fa_cs_gr_r 都 DONE 之后再做，或者放进另一份新副本里改。
6. 要重跑的是 32,000 局旋钮关、各档（旧定义 84,000 局，重跑时按新的旋钮关结果重新选格），再加缓存部件研究的带门四组 32,000 局。

---

## 1. Q3 停滞门：定义、R17 实现、R6/R7 原始实现

### 1.1 owner 的定义 [代码/文档]
- `exp/offline_search/rounds/r17/cache/OWNER_RULINGS.md:6-12`：停滞门就是 R6 停滞模块 / R7 校准停滞触发器。缓存动作下进展停滞时，强制做一次真模型调用（MISS）。规则是"确认慢就调用，模糊就再看一步"，用的是 Q3 的进展可靠性。冷却只跟在停滞触发的调用后面（R6-C-v2）。R7 的结果是：调用预算相同时 +1.15 pp [+0.12, +2.23]（`logs/offline_search_exploration.log.md:1810`）。在带门四组里，门触发的调用是唯一的模型调用，计入成本。
- `OWNER_RULINGS.md:14-17`（10-13）：停滞门只用 Q3，写法是 YAML 里的 `miss_mode: stall_only` 加 `stall_path`。旧的 `NoProgressGuard`（R8 span，n=3，eps=0.5）已经拔掉，LIBERO 和 MetaWorld 都适用。R17 一侧对应 `exp/r17_integration/common/layers/guard.py:1-14`（构造即报错）和 `common/config.py:21`（`guard` 默认 False）。

### 1.2 Q3 估计器本体（R6，R17 原样调用）[代码]
文件：`exp/offline_search/rounds/r06/ideation_Q3/stall/stall.py`
- 拟合 `StallModel.fit(library, metric, commit_controls)`（:174）：
  - 要求 `commit_controls == min(H, 2*exec_steps)`（:177）。
  - 老库的 step 以 `exec_steps` 为单位，换算成控制步是 `step*R`（:189-193）。
  - **只用成功局**（:218-220 跳过失败局）。
  - W = max(2, ceil(.05·T_med))，K = min(5, E−1)（:233-234）。
  - 对每个窗口做留一（排除整局）估计 delta_hat，记下真推进量和残差。
- 判定 `calibrated_status`（:393-409）：
  - 每个参考局取一个最近的上下文；e90 = 残差的 q90，a10 = 推进量的 q10（:404）。
  - `delta+e90<a10` 判 `slow_confirmed`；`delta<a10<=delta+e90` 判 `slow_ambiguous`；否则判 `ok`（:407）。
- `StallTracker.observe(key, control_index)`（:475）：参考局少于 4 个时判 inactive（:486）。时钟必须严格递增。
- 文件头写明"Only successful library episodes enter templates/calibration. No test trajectory…"（:7-8）；provenance 里写着 `calibration_scope="successful library; … no evaluation data"`（:201）。
- 调用与冷却规则在 `rounds/r06/ideation_Q1/method_c/stall_bridge.py`：
  - `call_probability`（:58-66）：冷却锚点上不调用；`slow_confirmed` 时为 1；其他状态取名义彩票概率。
  - `starts_cooldown`（:69-71）：只有停滞触发的调用才开始冷却。
  - `extra_look` / `scheduled_look`（:43-48、:74-75）。

### 1.3 R6/R7 原始部署（老插件里的 C 控制器）[代码]
`rounds/r06/ideation_Q1/method_c/methods.py:37` `CalibratedRescue`：
- 时钟：`control_index = q.step * block_controls`，其中 block=5，也就是**按环境控制步计**（:181）；接口固定为 b=5、h=10（:89-91）。
- 每个新的视觉锚点调一次 `tracker.observe`（:186-193）；冷却规则见 :201、:209-210；彩票部分见 :204-207（rho>0 时）。
- 写出 `os_force_miss` 和 `os_reason` 63/62（:229-231）。
- `slow_ambiguous` 时在下一个 5 步盲槽插一次 LOOK（:213-218、:235-239，返回 `LookReason(8,'c_slow_ambiguous')`）。
- policy tail 用 `CommitJudge` 门面（:246-256）。
- R6 的 C 臂插件参数是 `--os-blind --os-policy-tail --os-policy-tail-blocks 1 --os-judge guard_only`（`method_c/emit_specs.py:27-28`）。

### 1.4 R17 实现（DP / Fast-WAM / Cosmos LIBERO 和全部 MetaWorld）[代码]
- YAML 校验在 `exp/r17_integration/common/recipe.py:29-42`：
  - `miss_mode ∈ {legacy, never, stall_only}`；
  - `stall_only` 必须带 `stall_path`；有 `stall_path` 时必须是 `stall_only`。
- 服务端在 `stall_only` 下只允许停滞门 MISS 或旋钮 MISS，其他 MISS 一律报错（`common/serving.py:375-379`）。
- 触发逻辑在 `exp/r17_integration/search/cache_recipe.py` 的 `CacheStudyRecipe`（:289）：文档写的是"Connection-local R6-C-v2 at nominal lottery probability zero"（:290）。
  - 载入 `stall_path`（`load_stall` :255-287）：读 `READY.json`，核 `store_digest`、`fitted_digest` 和 `calibration_partition=="B_library_successful_LOEO"`（:276-279），再读 `stall.pkl` / `stall.json`（R6 格式）。
  - **按决策步计**：`quantum = model.exec_steps`，调用 `observe(code, decisions*quantum)`（:343-347）；记录里写 `stall_clock_unit='decision'`（:366）。
  - `call = call_probability(state, cooled, 0.)==1`，彩票为 0，所以只有 `slow_confirmed` 会调用（:351）。冷却规则见 :354-355。结果 `force_miss = call or knob`，`os_reason` 63/62（:359-367）。
  - `slow_ambiguous` 只设置 `look_due = query.step+1`（:356）。**在 `exp/r17_integration` 和 `src/openpi` 的非测试代码里，`force_look` 只有定义（`cache_recipe.py:314`、`common/layers/recipe.py:323`），serving 从不调用**。所以 R17 里"模糊就多看一眼"只留记录 `stall_extra_look`，不改变服务行为 [代码]。
- 标定产物的生成，有两处：
  - **C5/C10 初版**：`exp/r17_integration/ops/cache_study.py:273-336` 的 `fit_inputs`。输入是选定的 B 库视图（行上的 task / episode / step / success）加上该格 fit 的检索度量 `metric_mapping(fitted)`，调用 `StallModel.fit(... commit_controls=min(H,2R))`（:322-323）。产物在 `<run>/assets/cache_inputs/<suite>/n<N>_C<c>/stall/`，`READY` 里写 `calibration_partition=B_library_successful_LOEO`、`independent_B_validation_recording=False`（:325-333）。
  - **C10 按段重标定（10-18）**：`rounds/r17/longseg_1017/recalibrate.py:97-135` `fit_q3`。只取段头行（`step % (commit/replan)==0`，:108），按 H=1、exec_steps=1、commit=1 拟合（:115），也就是"一个段 = 一个时间单位"；`clock_unit='segment_decision'`（:120、:130）。各档周期按 M/N 段头重解（:145-215，Q3 调用和冷却都算进 M）。说明在 `longseg_1017/RECALIBRATION.md:3-13`。
- 实际用的 stall_path（读的是单个 YAML 和 READY）[产物]：
  - 成功过滤库 · DP · LIBERO · 减步 · 旋钮关（`/data/mw_finetune/eval/fa_ls_dp_r_f_o_l/assets/cache_yamls/ls_libero_10_100_chrg_koff.yaml:174-178`）：`miss_mode: stall_only`，`stall_path: /data/mw_finetune/eval/fa_ls_dp_r_f_o_l/assets/q3_decision1018/9092527f…`。READY 里是 `clock_unit=segment_decision`、`commit_horizon=10`、`library_size=100`，`calibration_source="selected teacher B100; Q3 successful-episode LOEO empirical residuals"`。重标定前（同名 `.pre_recal1018_*` 文件，:177）指向 `/data/mw_finetune/eval/ffk1015_dp_libero_reduced_formal_A_filtered/assets/cache_inputs/libero_10/n100_C10/stall`。
  - 成功过滤库 · Fast-WAM · LIBERO · 全步 · 旋钮关（`fa_ls_fw_f_f_o_l/.../ls_libero_10_100_chrg_koff.yaml:173-176`）：同样是 `assets/q3_decision1018/d673998d…`，`segment_decision`。
  - 成功过滤库 · Cosmos · LIBERO · 减步 · 各档（`fa_co_r_f_l/.../n100_l10_crg_k10.yaml:174-177`）：`assets/cache_inputs/libero_10/n100_C16/stall`，`commit_horizon=16`，没有 clock_unit。

### 1.5 R6/R7 当初在哪个模型、哪个库上标定 [代码/产物]
- 是 **π0.5 和 GR00T 在 LIBERO 上**：l10 / spatial 各两种库大小，共 8 格（50 和 500）。见 `ideation_Q3/stall/HANDBACK.md:34-45`，源清单是 `ideation_Q3/stall/source_manifest.json`。
  - 度量来自 R4/R5 的 `BlindAWM` fit，例如 `/home/weiland/trace_runs/os_closed_loop/r05_ptail/fits/r5t_p_l10_50_tail1uc.pkl`。
  - 库是 `lib=current`（部署时约 50 局）和 `lib=big`（π0.5 用 `bpool_cs`，GR00T 用 `bpool_all`，各 500 局；`rounds/r02/g1_awm/awm.py:26-27`）。
  - 拟合脚本 `ideation_Q3/stall/fit_models.py:39-70`，库根目录是 `/home/weiland/trace_runs/offline_search_store`（:18）。产物还在 `/tmp/q3_stall_fits/<cell>/`（8 格加 bmech）[产物]。
- `r06_c_cal`（每格 10 局 B-val）**只用来标定 C 的彩票和预算**（R → Ehat 的 a、b，以及 rho 的解，见 `rounds/r06/ANALYSIS.md:353-354`），不进 Q3 阈值。R17 和这次移植都把彩票设为 0，用不到它。
- R6 的这 8 份 Q3 产物**不能直接拿给 R15/R16 用**：R15/R16 的 fit 对每个选定子集重新做了 PCA（`rounds/r10/recipe/fitting.py:257`，`pca='fresh selected-library randomized SVD seed0'`），检索码和 R4/R5 的 BlindAWM 不同，要按各格的 fit 重新拟合（分钟级 CPU，见 §5）。

---

## 2. R17 的 π0.5 / GR00T LIBERO 路径（04987e6b）

- [代码] 支持 `stall_only` 的 'g' 组，但只用 tmp 合成库跑过 CPU 测试：`exp/r17_integration/tests/ops/test_libero_vla.py:107`、`:230`，`test_libero_vla_knob.py:59`、`:88`。交回文档是 `exp/r17_integration/handbacks/HANDBACK_LIBERO_VLA.md`，GPU 命令一律标注"Run these commands **later** … not executed"（:59、:92-146）。
- [产物] **没有 R17 的 LIBERO 库**：`/data/r17_library/` 下只有 `pi05_mw_mt50`、`groot_mw_mt50`，没有 `pi05_libero_*`、`groot_libero_*`。`exp/r17_integration/ops/cache_study/` 下也没有 `libero_pi`、`libero_gr`。所以**这条路径的 Q3 标定从没做过，也没有产物**。
- [产物] `/data/r17_library/libero/{pi05,groot}/<suite>/` 下只有 R14–R16 时代 YAML 用的 `preload_path` 缓存 pkl：
  - π0.5 是 `cp1_spatial_pool_16.pkl`，GR00T 是 `<suite>_sp16_S3.pkl`。
  - 这些文件软链自 `/home/weiland/trace_runs/dual_20260923/libs/...` 和 `/data/libero_cache/libraries/...`（`filter_failed_knob/cache_study_1019/HANDBACK.md:41-44`）。
  - 它们不是检索库，也不是 R17 的 B500 库。
- [产物] R17 路径的纯推理参照已经有了（4090，A 集 spatial+10 各 1000 局）：`pl_a2` 90.8、`pl_k1a2` 90.3、`gl_a3` 88.4、`gl_k1a` 88.5，以及执行 10 步的 `pl_h10` 96.2/88.6、`gl_h10` 90.4/88.8（`logs/session_handoff.md:1593`、`:1649`、`:1386`、`:1378`）。但有两个问题：
  - R17 路径比 R14 路径低约 2 个点，GR00T spatial 低 5 个点，原因没查；
  - "K1 加执行 10 步被启动器拒绝"（`session_handoff.md:1244`）。
- [代码] formal-A 的机器没接这两个模型：`filter_failed_knob/workflow.py:13-14` 的 `TEST_SOURCE` 只有 dp / fastwam / cosmos；`formal_a.py:189-191` 的 π0.5 / GR00T 走 `full_vla`（R15）和 `reduced_vla`（R16）。格子数也不同：`formal_a_queue.py:38` 给 R17 模型 6 个库大小（50–500），给 VLA 4 个（50/100/200/500）。用户说的"6 档库×2 套件"是 R17 的口径；R15/R16 是 **4 个库大小 × 2 套件 = 8 格**，每格旋钮 4 档（ir45/40/32/25）加旋钮关（`formal_a.py:25`）。
- **结论**：这条路径**不能直接跑**。要跑先得做：
  1. GPU 建库：2 模型 × 2 suite × B500 满步教师采集（`HANDBACK_LIBERO_VLA.md:92-108`）；
  2. CPU finish / fit 和 `cache_study fit`，这一步才产出 Q3（:110-137）；
  3. formal-A 接线（TEST_SOURCE / PURE / OFF）；
  4. 修好 K1 加 C10 纯推理参照；
  5. 全部批次按 6 个库大小重跑。

  做完的结果还和 R10–R16 的历史数字不可比。**不推荐**。

---

## 3. R16（减步，K1）

### 3.1 判定器结构，旧门挂在哪 [代码]
各档和旋钮关的 fit 都是 `TaskDescentKnob`。继承链：
- `rounds/r15/descent/recipe.py:127`（LeaveOwnEpisodeMixin）
- → `rounds/r14/descent/recipe.py:20`
- → `R14Knob`（`rounds/r14/knob/recipe.py:53`）
- → `R11Knob`（`rounds/r11/knob/recipe.py:67`）
- → `R10Recipe`（`rounds/r10/recipe/recipe.py:84`）

`R10Recipe.query/blind_step/policy_tail_step` 全部转给 `self.inner`（:125-142）。

`inner` 在 R10 拟合时构造（`rounds/r10/recipe/fitting.py:218-221`）：
`TriggerCommitJudge(base_kwargs=dict(...serving='anchor_tail',budget=1,gates='budget_only'), progress_guard='noprog_span', events='none', stuck_guard='vision_confirmed', policy_tail_gate='lifecycle', monitor='off', disabled_guards=ONLY_NO_PROGRESS)`，
其中 `ONLY_NO_PROGRESS=('stuck','terminal','overtime')`（`r10/recipe/recipe.py:17`），也就是**只留 no_progress 一位**。之后 `judge.base` 换成 `_CorrectedCache`（:236-245），`recipe.inner = judge`（:248）。

TriggerCommitJudge 的继承链：
- `rounds/r08/abl/judge.py:36/40`（_DisabledGuards）
- → `rounds/r06/p2_ablations/judge.py:55/59`（_TriggerMask，**:21 构造即报错**）
- → `CommitJudge`（`rounds/r05/q1_commit/judge.py:91`，policy tail 在 :112）或 `GrootCommitJudge`（`r06/p1_groot_commit/judge.py:48`，只改终止符号）
- → `PolicyTailJudge`（`r04/k10_policy_tail/judge.py:19`，`query` 在 :35-38 里记 `_policy_gate_anchor`）
- → `VisionConfirmedBlindMixedJudge`（`r04/k7_guard/judge.py:23`）
- → `BlindMixedJudge`（`r04/k1_blind/judge.py:12`）
- → `MixedJudge`（`r03/h3_judge/judge.py:93`）

旧门本体在两个地方：
- `BlindMixedJudge`：`progress_guard="noprog_span"`（:15-24），span 算术在 `_progress`（:76-88），**盲尾否决**在 :52-53（无进展就强制看图）。
- `MixedJudge`：guard 位和 `noprog_n=3`、`prog_eps=.5`（`r03/h3_judge/judge.py:18-20`、:100、:254-…）。65b1cc5f 之后，构造（:111）和反序列化（:163-167 `__setstate__`）都会报错。

MISS 的路径：
- inner 写出 `os_force_miss`；
- `R11Knob.query` 里读到 `guard = os_force_miss`（`r11/knob/recipe.py:138`），只在 `not guard` 时才加周期旋钮（:155）；R13 对应 `r13/knob/recipe.py:100`；
- 插件 `--os-judge guard_only` 收到 forced 就判 MISS（`closed_loop/plugin.py:1003`、:1020-1021）。

[产物] 旧的 R16 旋钮关 group 行（例：`openpi_ext/attic/_trash_oldgate_20261020/r16_k1ref_pi05_off/group.json`）：
- `method=…r15.descent.recipe:TaskDescentKnob`；
- `plugin_args` 含 `--os-blind --os-policy-tail --os-policy-tail-blocks 1 --os-judge guard_only`；
- `yaml_patch.miss.num_steps=1`，`judge=guard_only`。

`cache_study_1019/HANDBACK.md:94-96` 核对过 16 个旋钮关 fit，全是 `TriggerCommitJudge`/`TriggerGrootCommitJudge`，参数 `noprog_span`、n=3、eps=.5。

### 3.2 Q3 最小接法（R16 和 R15 通用）
**只换 fit 里的 `method.inner`**，`TaskDescentKnob`、`R13Knob`、`R11Knob` 和 `R10Recipe` 都不改。

1. **新文件**：建议放 `exp/offline_search/rounds/r06/ideation_Q1/method_c/stall_only.py`，和 `stall_bridge.py` 同包，两棵代码树里都有这个包。类名例如 `Q3StallInner`，持有 `base`（原 `_CorrectedCache`；无修正器的组用 BlindAWM）、`C`（原 inner 的 g3 Tables，`R14Knob._diagnostics` 要用 `inner.C.task`，见 `r14/knob/recipe.py:139`、:207）和 `stall_model`。
   - `reset`：调 `base.reset`，新建 `StallTracker(model, task)`，清空冷却锚点；同时照 `methods.py:178-180` 按局身份自动 reset。
   - `query(q)`：
     - `res = base.query(q)`；
     - `tracker.observe(q, int(q.step)*5)`，时钟同 R6-C 的 `methods.py:181`；
     - 原始 QueryView 交给 `StallModel.encode` 投影（`stall.py:276-292`）。R6 审计过：raw 投影和 A 的检索码逐个一致（`ideation_Q3/stall/HANDBACK.md:49`）。也可以照 `methods.py:146-175` 直接抓检索码。
     - `call = call_probability(state, cooled, 0.)==1`，`starts_cooldown` 只在 stall 时开始冷却；
     - 记录 `_policy_gate_anchor`，同 `PolicyTailJudge.query`；
     - extras 写 `os_force_miss=float(call)`、`os_reason=63` 和 `stall_*` 诊断。
   - **`slow_ambiguous` 不插段中 LOOK**。理由：R17 里这一步不生效（§1.4）；10-18 的规则是"停滞门只能在段边界强制重新决策"。而 C10 下每个段头本来就有新的视觉观测。
   - `blind_step`：直接 `base.blind_step`，也就是去掉 `noprog_span` 盲尾否决。`policy_tail_step` 用 `CommitJudge` 门面，照 `methods.py:246-256`。`invalidate_anchor`、`bytes_per_entry`、`last_blind_extras` 透传。
2. **插件判定模式**：加 `stall_only`，语义同原 `guard_only`：forced 判 MISS，否则判 HIT。改 3 行：`closed_loop/plugin.py:77`（JUDGE_MODES）、:259（parse）、:1026（verdict）。R16 副本 `openpi_r16_k1ref_server/exp/offline_search/closed_loop/plugin.py` 是同一组行号（:77、:257-259、:1026）。`guard_only` 继续报错。
3. **分析侧**：`rounds/r16/analyze.py:83` 和 `rounds/r16/conductor_analysis.py:44` 写死了 `--os-judge` 必须是 `guard_only`（R15 的行由 `formal_a_analysis.load_vla` 经 :104 走前者，R16 的 entry 经 :111 走后者），要改成接受 `stall_only`。
4. **新定义**用新目录建，不覆盖旧封存：
   - resident 会对每个 entry 核 fit sha（`rounds/r16/resident.py:94-96`），旧 fit 也已经不能反序列化。
   - 构建脚本里先在本进程临时绕开 `MixedJudge.__setstate__`，取出旧 inner 的 `base` 和 `C`；再装上 `Q3StallInner`；成功过滤库再套一次 `vla.candidates`（`filter_failed_knob/vla.py:25-36`）。
   - 插件参数里把 `guard_only` 换成 `stall_only`。
   - 写出来的新 fit 里没有任何旧判定器对象。
   - 另一个办法：原库直接复用缓存部件研究那份不带门的 `_cr` fit，它的 base 能直接载入，再从 base 合成一个只含 `.task` 的 `C`。

### 3.3 R16 运行时到底 import 哪份代码 [产物：在跑的 fa_cs_pi_r 进程]
- **常驻服务端**：`/home/weiland/projects/openpi_r16_k1ref_server`。
  - 依据：pid 2487923、2607141、2775732 的 cwd 和 PYTHONPATH 都是这棵树，命令是 `-m exp.offline_search.rounds.r16.resident`。
  - 代码里对应 `rounds/r16/conductor_runner.py:19` 的 `SERVER_TREE`，在 :202 换掉 `local_server.TREE`。
  - 启动时载入整组所有 entry 的 fit（`resident.py:91-121`），所以新模块必须在**这棵树**里。
  - 这棵树的 `stall.py`、`stall_bridge.py` 和主仓逐字节相同（diff 为空）；65b1cc5f 改了它的 `plugin.py`、`r03/h3_judge/judge.py`、`r06/p2_ablations/judge.py`，留有 `.before_oldgate_removal_1020` 备份（git status）。
- **conductor runner**：主仓 `/home/weiland/projects/openpi`（pid 2929343，cwd 是主仓，PYTHONPATH=.:src）。
  - 它 import 的有：`rounds/r16/{conductor_runner,conductor,prepare}`；`conductor.py:51/76/204` 懒加载 `analyze`；`prepare.py:18` 带进 `r15/descent/prepare`，再连带 recipe、r13、r14、r10。
  - **在跑期间这些文件一律不改**。
- **worker agent**：
  - 本机和 a100 用 `/data/mw_finetune/qualification/r16_local_island`（pid 2935897，`conductor_runner.py:18`）；
  - timan107 和 timan1 用 `--agent-island /scratch/zixuans8/openpi_r16_conductor_v1`。
  - worker 不跑判定器，不用改。
- `/home/weiland/projects/openpi_r16_k1ref_cpu` 只用于 CPU 验证（`rounds/r16/k1ref/HANDBACK.md:101`），运行时不用。

### 3.4 缓存部件研究 fa_cs_pi_r 与带门四组
- [代码] 定义在 `filter_failed_knob/cache_study_1019/study.py`：
  - `GROUPS` 在 :32（c、cr、ch、chr），`NOT_BUILT` 在 :33（cg、crg、chg、chrg）。
  - `fit_blob`（:53-75）：取原库 · 减步 · 旋钮关的 fit，`R10Recipe.inner = base`；c 和 ch 把类换成 BlindAWM（:64）。
  - `build_model`（:77-129）：C5 不带 `--os-blind`，C10 带 `--os-blind`；不带 `--os-judge`，`full_model=False`（:101-113）。
  - 交回文档 `cache_study_1019/HANDBACK.md`，包装脚本 `filter_failed_knob/commands/{prepare,launch}_fa_cs_{pi,gr}_r.sh`。
- [产物] 运行目录 `/data/mw_finetune/eval/fa_cs_pi_r/group.json`，进度 3,570/16,000（progress.json）。`fa_cs_gr_r` 只有 group.json 和 ROOT.json，还在排队。
- **带门四组的定义**：cg / crg / chg / chrg = c / cr / ch / chr 加 Q3。
  - fit 和 `fit_blob` 一样，只是 `R10Recipe.inner = Q3StallInner(base, C, stall_model)`，不带旋钮层；
  - 插件参数加 `--os-judge stall_only`，`full_model=True`，k=1（MISS 用 K1，GR00T 带 `GROOT_DENOISING_STEPS=1`）；
  - chg 和 chrg 另加 `--os-policy-tail --os-policy-tail-blocks 1`，保证 MISS 也执行满 C10（插件要求 `--os-policy-tail` 必须配 `--os-blind`，见 `plugin.py:207`）；cg 和 crg 是 C5，MISS 就是 5 步；
  - Q3 产物和 formal-A 共用每格一份（§5）；
  - 规模 2 模型 × 8 格 × 4 组 × 500 = **32,000 局**；
  - 和 c/cr/ch/chr 共用 R16 端口，排在 fa_cs_gr_r 之后。成本口径照 `OWNER_RULINGS.md:11-12`，门触发的调用就是唯一的模型调用。

---

## 4. R15（全步）

### 4.1 结构 [代码/产物]
- arm 规格在 `rounds/r15/descent/prepare.py:101-109` 的 `arm_spec`，插件参数写死了 `--os-blind --os-policy-tail --os-policy-tail-blocks 1 --os-judge guard_only --os-fit-artifact …`。formal-A 用这个函数发 arm（`formal_a.py:84-108`）。
- 每格每档用各自的 fit（`vf.original`，见 `vla_full.py:47-48`）。
- [产物] 例如 `_trash_oldgate_20261020/eval/fa_pi_f_f_o_l50/arms.json`：
  - 库大小 50 的旋钮关是 `R11Knob`（method off）；
  - 100–500 的各档是 `R13Knob`（periodic，target .45/.40/.32/.25），fit 在 `formal_a_1016/runs/ffk1015_<model>_libero_full_formal_A_<arm>/fits/<cell>_<level>.pkl`。
- 两者都是 `R10Recipe` 的子类，`.inner` 和 R16 是同一个 R8 TriggerCommitJudge（§3.1）。
- R13Knob 的流程：
  - `off` 和 `periodic` 直接走 `R11Knob.query`（`r13/knob/recipe.py:81-82`）；
  - guard 取 `os_force_miss`（`r11/knob/recipe.py:138`）；
  - 周期旋钮只在没有 guard 时触发（:155）；
  - 插件 `guard_only` 收到 forced 就判 MISS。
- **Q3 最小接法和 R16 完全一样**（§3.2 的 1–4 步）：换 `.inner`，插件改用 `stall_only`。另外 `r15/descent/prepare.py:108` 写死了 `guard_only`，新构建脚本自己替换插件参数就行，不要改这个文件，因为 fa_cs 的 runner 会 import 它（§3.3）。

### 4.2 R15 运行时 import 哪份代码
- [代码] 启动器是 `rounds/r17/filter_failed/legacy.py:24-33`。它包着 `closed_loop/ops/h100/control.py`，把 tether push/pull 改成 rsync，但只覆盖 `plan`、`sync`、`chain`、`collect` 四个动作（:27）。
- **GR00T 的服务端在 4090 本机**：用 `local_server.TREE = HERE.parents[4]`，也就是**主仓** `/home/weiland/projects/openpi`（`closed_loop/ops/h100/local_server.py:24`、:128）。[产物] `_trash…/fa_gr_f_f_o_l50/runs/*/h100_launch.json` 里是 `server_host='local'`。
- **π0.5 的服务端在 H100 节点**：用的是那台机器上的代码副本 `BASE/openpi`，默认 `/data/oscl_h100/openpi`（`closed_loop/ops/h100/assets.py:18`）。这份副本**只在 `control setup` 时刷新**（`control.py:303-323`），刷新时打包 `source_files()`，exp/offline_search 下所有 .py 都会带上（`control.py:151-176`）。但 `setup` 走 tether push（`control.py:78-94`、:136-148），按规定禁止，而 legacy 包装又不允许 `setup`（`legacy.py:27`）。所以要么用 rsync 公网直连把新模块和 plugin.py 推上去，要么给 legacy 加一个 `setup` 动作（同一个 rsync 补丁）。[产物] `_trash…/fa_pi_f_f_o_l50/runs/*/h100_launch.json` 里 `server_host=None`，即默认的 h100，`worker_host=timan107`。H100 上那份副本的现状没有上机核对。
- worker 岛是 timan107 的 `/scratch/zixuans8/openpi_trace`（`assets.py:19`），不跑判定器。

---

## 5. 标定数据：在哪，要跑什么

- [产物] R15/R16 老流程实际用的检索库在 `/home/weiland/trace_runs/offline_search_store/library/`：
  - π0.5：`{pi05_l10,pi05_spatial}/bpool_cs`，各 500 局，成功 436 / 487；
  - GR00T：`{groot_l10,groot_spatial}/bpool_all`，各 500 局，成功 427 / 456；
  - 数字来自 `library/READY`，每格选哪些局在 fit 的 `kwargs.episode_subset` 里（`formal_a_1016/LIBRARIES.json` 记了每格的库路径、库大小和保留的局）。
  - 每个库目录都有 `task_id/episode/step/success.npy` 和 `manifest.json`，π0.5 的 H=10，GR00T 的 H=16，`exec_steps=5`，所以 L=min(H,2R)=10。**Q3 需要的输入全都在这里**。
- "成功过滤库"在磁盘上没有单独的库，它是 fit 层面的候选行过滤：`vla.candidates`（`vla.py:25-36`）只切 `R10_CANDIDATES` 这些候选字段（`filter_failed/row_filter.py:16`），投影不变。
- [推断] Q3 只用成功局做模板和留一（`stall.py:218-220`），过滤又只去掉失败局的候选行，所以**原库和成功过滤库在同一格上得到的 Q3 模板和标定完全一样**。全步和减步的 fit 用同一个 base：R16 经 `r16/prepare.py:264` 调 `adapter.descent_blob`，取的就是 R15/R14 的旋钮关 fit（`r15/descent/prepare.py:70-76`）。结论是**每 (模型, suite, 库大小) 一份 Q3 就够了，共 16 份**，从原库的旋钮关 fit 拟合。
- **要跑什么**：只有 CPU，不需要开发集的局，也不上 GPU。
  1. Q3 拟合：每格一次 `StallModel.fit(library=dict(manifest=<父库 manifest>, task_id/episode/step/success = 选定子集的行), metric=<该格 fit 的 base>, commit_controls=10)`。`_extract_metric` 会顺着 `.base` 找到 A 的度量（`stall.py:110-149`）。R6 的同类拟合，500 局库大小大约 72 s 一格（`ideation_Q3/stall/fit_pi05_l10_500.json`），16 格用进程池几分钟就能做完。
  2. **重解各档周期**：R15 各档周期原来是用旧门掩码解的（`r15/descent/fits.py:55` 里的 `guard=M.guard_flags(...)`，在 :89-93 求解，规则是 `r11/opus/ir_model.py:88-95` 的 noprog_span）。现在把它换成 Q3 留一的强制调用掩码，含一锚冷却，写法照 `longseg_1017/recalibrate.py:145-175` 的 `episodes()`，只取偶数步锚点。`ScheduleReplay.solve('periodic', .45/.40/.32/.25)`（`r11/knob/calibration.py:36-109`）原样不动。16 格的锚点输入齐全（`rounds/r11/opus/out/anchor/<cell>_f{0..4}.npz`），R16 经 descent_blob 读同一份周期。这一步是按 owner 10-18"改了口径 IR 估计也要改"的规定做（handoff:997）。
- [产物] 库大小 50 的格子里，有些任务的成功局不足 4 个，这些任务上 Q3 会判 inactive、不调用（`stall.py:486`）：π0.5 l10 的任务 9、π0.5 spatial 的任务 0、GR00T l10 的任务 6 和 8、GR00T spatial 的任务 0（`LIBRARIES.json` 的 successes_per_task）。
- 新产物按磁盘规定写到 `/nvme/weiland/openpi` 下，原路径留软链；有 driver 在跑时不要在 `/data` 上批量复制。

---

## 6. 要重跑的批次（两份 MANIFEST 内容相同，都是 59 行；纯推理参照保留）

来源是 `/data/mw_finetune/_trash_oldgate_20261020/MANIFEST.txt` 和 `/home/weiland/projects/openpi_ext/attic/_trash_oldgate_20261020/MANIFEST.txt`（`diff` 一致）；单位定义在 `filter_failed_knob/formal_a_1016/QUEUE.json`。局数按旧定义；各档要等新的旋钮关结果出来后，由 `formal_a_queue.build_levels`（:203-216）重新选格，旧的 `formal_a_1016/off_decisions/{pi05,groot}_*.json` 作废。

脚本约定：
- 链入口是 `filter_failed_knob/commands/{prepare,launch}_<名>.sh`，和 `formal_a_1016/commands/` 里的同名文件内容相同。
- 单位级脚本调 `formal_a_queue dispatch prepare/launch --name <名>`。
- R15 的每格脚本转到 `formal_a_1016/commands/*_<格>.native.sh`，再调 `vla_workers prepare/launch --job <名>`。
- R16 的 launch 调 `rounds.r16.conductor_runner run`（`formal_a.py:169-172`）。

| 白话 | 代号 | 旧局数 | prepare / launch |
|---|---|---|---|
| 原库 · π0.5 · LIBERO · 全步 · 旋钮关 | fa_pi_f_u_o（_l50 _l100 _l200 _l500 _s50 _s100 _s200 _s500） | 8×500=4,000 | 单位级 `commands/{prepare,launch}_fa_pi_f_u_o.sh`；每格 `commands/{prepare,launch}_fa_pi_f_u_o_<格>.sh` |
| 成功过滤库 · π0.5 · LIBERO · 全步 · 旋钮关 | fa_pi_f_f_o（8 格） | 4,000 | `…_fa_pi_f_f_o[_<格>].sh` |
| 原库 · GR00T · LIBERO · 全步 · 旋钮关 | fa_gr_f_u_o（8 格） | 4,000 | `…_fa_gr_f_u_o[_<格>].sh` |
| 成功过滤库 · GR00T · LIBERO · 全步 · 旋钮关 | fa_gr_f_f_o（8 格） | 4,000 | `…_fa_gr_f_f_o[_<格>].sh` |
| 原库 · π0.5 · LIBERO · 减步 · 旋钮关 | r16_k1ref_pi05_off + fa_pi_r_u_c（补齐） | 1,500+2,500=4,000 | 前者是 `rounds/r16/k1ref` 的链（handoff:1390 `r16_ops/chain_k1ref.run.sh`）；后者 `commands/{prepare,launch}_fa_pi_r_u_c.sh`。重跑时**合成一个 4,000 局的 group** |
| 成功过滤库 · π0.5 · LIBERO · 减步 · 旋钮关 | ffk1015_pi05_libero_reduced_off | 4,000 | `filter_failed_knob/vla.sh`（在 R16 服务端树里跑 `vla` / `vla_runner`） |
| 原库 · GR00T · LIBERO · 减步 · 旋钮关 | r16_k1ref_groot_off + fa_gr_r_u_c | 2,050+1,950=4,000 | 同上两类 |
| 成功过滤库 · GR00T · LIBERO · 减步 · 旋钮关 | ffk1015_groot_libero_reduced_off | 4,000 | `vla.sh` |
| 原库 · π0.5 · LIBERO · 全步 · 各档 | fa_pi_f_u_l（旧 5 格 l50 l100 l200 s50 s100） | 5×2,000=10,000 | `…_fa_pi_f_u_l[_<格>].sh`（`_dyn` 是别名） |
| 成功过滤库 · π0.5 · LIBERO · 全步 · 各档 | fa_pi_f_f_l（旧 6 格 l50 l100 s50 s100 s200 s500） | 12,000 | `…_fa_pi_f_f_l[_<格>].sh` |
| 原库 · GR00T · LIBERO · 全步 · 各档 | fa_gr_f_u_l（WAITING_OFF，没跑过） | 0（上限 16,000） | `…_fa_gr_f_u_l.sh` |
| 成功过滤库 · GR00T · LIBERO · 全步 · 各档 | fa_gr_f_f_l（旧 5 格 l50 l100 l200 s50 s100） | 10,000 | `…_fa_gr_f_f_l[_<格>].sh` |
| 原库 · π0.5 · LIBERO · 减步 · 各档 | fa_pi_r_u_l（8 格） | 16,000 | `commands/{prepare,launch}_fa_pi_r_u_l.sh` |
| 成功过滤库 · π0.5 · LIBERO · 减步 · 各档 | fa_pi_r_f_l（7 格） | 14,000 | `…_fa_pi_r_f_l.sh` |
| 原库 · GR00T · LIBERO · 减步 · 各档 | fa_gr_r_u_l（6 格） | 12,000 | `…_fa_gr_r_u_l.sh` |
| 成功过滤库 · GR00T · LIBERO · 减步 · 各档 | fa_gr_r_f_l（5 格） | 10,000 | `…_fa_gr_r_f_l.sh` |
| 原库 · π0.5 / GR00T · LIBERO · 减步 · 缓存部件带门四组（新建） | 例如 fa_cs_pi_rg / fa_cs_gr_rg | 2×16,000=32,000 | 照 `cache_study_1019/study.py` 新写，沿用 `commands/*_fa_cs_*` 的形式 |

合计：旋钮关 32,000 局；各档旧定义 84,000 局，上限 8×16,000=128,000，实际按新的旋钮关结果选格；带门四组 32,000 局。

保留、不重跑的参照：
- π0.5 全步：`fa_pi_f_p10` / `fa_pi_f_ps`（`formal_a_1016/PURE_A_OVERRIDES.json`）；
- GR00T 全步：R15 的 C10 纯推理（`rounds/r15/descent/c10_proof.json:75` 的 test_references，经 `formal_a_analysis.py:56-58`）；
- 减步：R16 phase1 的 K1 纯推理 `r16p1_{model}_{suite}_k1_pure_c10`（`rounds/r16/design.json` 的 phase1，经 `formal_a_analysis.py:53-55`）。

---

## 7. 推荐路径与改动点

### 推荐：R16/R15 判定器换成 Q3（换 `.inner`），每格各拟合一份 Q3
新增文件放在 `filter_failed_knob/q3_port_1020/` 下，例如：build（拟合 Q3、重解周期、建定义）、cache_g（带门四组）、可选的 runner（patch `conductor_runner.SERVER_TREE`）。加上 `method_c/stall_only.py`。

| 步 | 内容 | 落在哪 | 何时可做 |
|---|---|---|---|
| 1 | 写 `Q3StallInner`（§3.2 第 1 条） | 新文件 `rounds/r06/ideation_Q1/method_c/stall_only.py` | 现在（只新增） |
| 2 | 16 份 Q3 拟合（CPU） | 新的 build 脚本 | 现在 |
| 3 | 按 Q3 掩码重解 ir45/40/32/25 的周期，写成新目录表（CPU） | 新的 build 脚本（调用 `ScheduleReplay` 和 R17 `episodes()` 的写法，不改原文件） | 现在 |
| 4 | 建旋钮关定义：R15 4 个单位 × 8 格；R16 4 个 group，每个 8 格 × 500；插件参数用 `stall_only` | 新的 build 脚本（复用 `formal_a.full_vla/reduced_vla` 的写法，自己替换插件参数） | 现在（只写新目录） |
| 5 | `plugin.py` 加 `stall_only` | 主仓 `closed_loop/plugin.py:77/259/1026` | 主仓现在就能改（fa_cs 的 runner 静态上不 import 它，只有 `look_recording` 懒加载）。稳妥起见放到 fa_cs DONE 之后 |
| 6 | 同上 | R16 服务端树 `openpi_r16_k1ref_server/.../plugin.py` 加上新模块 | **fa_cs_pi_r 和 fa_cs_gr_r 都 DONE 之后**。也可以另复制一份树，用新 runner 包装 patch `conductor_runner.SERVER_TREE`（代码在 `:19`、:202），不碰旧树 |
| 7 | 分析放行 `stall_only` | `rounds/r16/analyze.py:83`、`rounds/r16/conductor_analysis.py:44` | fa_cs 全部 DONE 之后，因为 conductor 会懒加载 `analyze` |
| 8 | 把 H100 上的 R15 π0.5 代码刷新到新模块和插件 | `/data/oscl_h100/openpi`（node h100），用 rsync 公网直连 | 跑 π0.5 全步之前 |
| 9 | 跑旋钮关 → `formal_a_queue.evaluate` 出新的 off_decisions → `build_levels` 只建 tune 格 → 跑各档 | 现有 `formal_a_queue` 流程，新单位名 | 旋钮关跑完之后 |
| 10 | 带门四组 cg/crg/chg/chrg（§3.4） | 新的 cache_g 脚本 | 定义现在就能建；运行排在 fa_cs_gr_r 之后（R16 端口串行） |

- 时钟按原 R6-C：`q.step×5` 个控制步。老插件 C10 下视觉锚点都在偶数步（段头），所以就等于"每段一个决策"，和 R17 的 `segment_decision` 只差一个常数倍；**拟合用 L=10，原流程即可，不需要像 R17 那样另做按段重标定** [推断]。去掉段中 LOOK 以后，段中再决策只剩插件自己的生命周期 LookReason。
- 运行顺序：
  - 4090 先跑完 fa_cs_pi_r（3,570/16,000）和 fa_cs_gr_r；
  - 然后在 4090 上依次跑 R16 的 Q3 旋钮关、R15 GR00T 的旋钮关、各档、带门四组；
  - π0.5 全步（H100）在第 8 步完成后就能跑，不必等 4090。
- 对 fa_cs_pi_r 的影响：第 1–4、10 步只新增文件，**没有影响**。第 5–7 步动的是 fa_cs 服务端和 runner 的 import 图，按上表等它和 fa_cs_gr_r 都 DONE 后再做，或者在新树里做。在跑批次的停滞门文件不动（R17 的 `q3_decision1018` 都是硬链接，link count 9，**不加链接、不 mv**）。

### 不推荐：改走 R17 的 pi05_libero / groot_libero
理由见 §2：没有库，没有 Q3；要 GPU 建库（2,000 局满步教师采集），还要补 formal-A 接线、K1+C10 参照，格子变成 6 个库大小；R17 路径的纯推理比 R14 路径低 2–5 点、原因没查；结果和 R10–R16 的历史不可比。

### 待协调者拍板的点
1. 是否同意 `slow_ambiguous` 不插段中 LOOK。这样和 R17 的实际行为、10-18 的段边界规则一致；原始 R6-C 是插的（`methods.py:213-218`）。
2. 构建时临时绕开 `MixedJudge.__setstate__` 取出 base 和 C（只用来抽取数组，服务时不构造旧判定器），还是改用 fa_cs 的 `_cr` fit 加合成的 `C`。
3. 插件新模式叫 `stall_only`（和 R17 的 YAML 用词一致）。
