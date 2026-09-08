# RoboCasa365 × warm-start RIT 帕累托 —— 运行进度

> 无人值守。owner 裁定见本文件 §0。进度条在 §1，每到里程碑更新一次。

## 0. 冻结的口径（owner 2026-09-08 裁定，不再重开）

| 项 | 值 |
|---|---|
| 库 | 只用 **S6 / full**，每任务 50 条、共 650 条；不扫库规模维 |
| 检索 | **text-IVF 开**（`text_ivf_knn` + `index_type: text_ivf` + `prompt_emb {enabled, weight 0}`） |
| 权重 | GR00T `grid3_vision_0@12_vision_2@37_robot_state@50`；pi0.5 `grid_vision_1@87_robot_state@12`（各自冻死，全档共用） |
| 阶梯 | GR00T k2={FULL, WARM@0.75} / k3={FULL, WARM@0.75, WARM@0.5}；pi0.5 k2={FULL, WARM@0.3} / k3={FULL, WARM@0.3, WARM@0.5} |
| 帕累托 | 按 **IR 网格**寻址（不是 delta），**6 个点**，四条线取可达区间交集铺公共网格 |
| 部署/报告 | 用**预测**切点发 yaml，用**实测** IR + SR 出图 |
| 标定 | 评测段 `base_seed=1,000,000`，每任务 10 集共 130；owner 裁定不涉及污染问题 |
| 评测 | **每任务 50 集、每个点 650 集**；13 arm/teacher（12 RIT + always_hit），teacher 地板由标定跑充当 |
| 拓扑 | server 只在 weilandserver：GR00T 5 进程 23160-23164；pi0.5 一端口 23170 `--replicas 4`。worker 只在 timan107，30 个 |
| 顺序 | **先 GR00T，全跑完再 pi0.5** |

## 1. 进度条

```
P0a 延迟测量      [##########]  DONE   GR00T 四档全 nsys 认证，s3(k)=5.232+3.006k，R²=0.991
P0b 成本权威      [##########]  DONE   rit_cost_rc.py：按 schedule 方向取剩余步数
P0c RIT 链路移植  [##########]  DONE   shadow 单集冒烟通过（s/winner/三档 y 齐全）
P1a GR00T 标定    [##########]  DONE   130/130 集，13 任务各 10；teacher 地板臂 macro SR 0.7462
P1b GR00T 拟合发射[##########]  DONE   shadow 12,817 行/13 任务/130 集；13 个 arm 已发
P1c GR00T 主跑    [##........]  pnp lane 在跑（3,250 集）；main lane 待从 20 续补到 50
P2  pi0.5 全流程  [..........]  待 P1
P3  四线帕累托图  [..........]  待 P2
```

## 1b. 新增的 owner 裁定（2026-09-08 下午）

- **H-gate 装在正式实验的 arm 上**（标定跑不装）：`score_hysteresis`，`theta_low=theta_high=θ`，
  `j=3 / probe_interval=3 / L=6`（与 LIBERO 线同一组常量）。
- **θ 的确定方法**：`derive_thresholds(标定期全部有限 cp1_score, THETA_TOP_FRACTION=0.85, 0.0)[0]`
  —— 即放行分数最高的 85%，等价于标定分数分布的 15 分位。与 GTP/RIT 线同口径。
- ⚠ `serve_groot_n15.py` 原先三处 `validate_groot_cache_config` 都没传 `allow_hysteresis_gate`，
  H-gate 会被 GR00T 的 load_guard 拒掉。已在三处补上（LIBERO 侧三个装配点本来就传）。

## 3. 关键数字

**GR00T 成本模型**（W2 k 阶梯，4090 独占，CUDA-Graph，nsys 认证，200 次/档）

> owner 裁定：三段值以**既有台账**（Stage A G-M，CUDA Graph，三段各一图，4090）为准
> —— `stage1=8.12 / stage2=9.36 / s3(4)=17.80`。k 阶梯只用来取 stage3 的**每步斜率**，
> 截距锚到 s3(4)=17.80，即 `s3(k) = 5.777 + 3.006k`。
> 本次 k=4 实测 17.585，与台账 17.80 差 1.2%，互为交叉验证。

| 档 | 付什么 | stage3 | 成本 (ms) | 占 MISS | 省 |
|---|---|---|---|---|---|
| FULL_HIT | 只 stage1 | — | **8.12** | 23.02% | 76.98% |
| WARM@0.75（剩 1 步） | s1+s2+s3(1) | 8.78 | **26.26** | 74.44% | 25.56% |
| WARM@0.50（剩 2 步） | s1+s2+s3(2) | 11.79 | **29.27** | 82.96% | 17.04% |
| MISS | s1+s2+s3(4) | 17.80 | **35.28** | 100% | — |

⚠ `s1+s2 = 17.48 ms` 已占 MISS 的 **49.5%** ⇒ 只要落到 warm（必过 LLM），成本就至少是 MISS 的一半；
warm 两档的可省空间只有 25.6% / 17.0%。帕累托的 IR 张力几乎全部来自 FULL_HIT 占比。
⚠ 按旧公式 `start_t × STAGE3` 会算成 WARM@0.75=88.2% > WARM@0.50=75.5% —— **阶梯成本序整个翻转**，
IR 网格的寻址会全错。这是必须新建 RoboCasa 成本权威的原因。
⚠ stage3 的固定头 5.78 ms 占「剩 1 步」成本的 66%：若按 `s3(4)/4=4.45` 给单步计价，
1 步档会被低估 49%（真值 8.78）。这正是必须补 k 阶梯、而不能只用 k=4 一个点的原因。

## 2. 逐步记录

### P0a 延迟测量（IR 网格的寻址依赖它）
- 2026-09-08 11:58 CDT 启动。weilandserver tmux `w2`，隔离工作区 `/tmp/openpi-stageA`（HEAD 代码经 tether 推送，非 git）。
- 跑 GR00T k∈{1,2,3,4} × prompt 0 × 1 进程，CUDA-Graph 档，nsys 认证；拟合 `s2act(k)=a+b·k`。
- 产物 `/tmp/openpi-stageA/exp/robocasa365/data/latency/groot_cg_k{1..4}_p0_r0.json`。


### P0b / P0c 代码（新增，未入库）
| 文件 | 作用 |
|---|---|
| `exp/robocasa365/rit_cost_rc.py` | RoboCasa 成本权威：`StageCost` / 按 schedule 方向算剩余步数的 `tier_cost` / `predicted_ir` / `attainable_range` / `delta_for_ir` / `common_grid`。LP 拟合、cuts、verdict walk 全部 import 自 `rit_pareto.rit_k`，不复制 |
| `exp/robocasa365/rit_shadow.py` | 标定用 shadow：执行 teacher 自己的动作，同时读同一个 orchestrator 的 (score, winner)，用 `run_stage3_from` 逐档算偏差，逐步写 JSONL |
| `exp/robocasa365/emit_rit_rc.py` | `calib` 子命令发标定 cell；`arms` 子命令拟合 → 读可达 IR 区间 → 铺 6 点公共网格 → 反解切点 → 发 `threshold + warm_tiers` 的 arm（带 H-gate） |
| `exp/robocasa365/serve_groot_n15.py`（改） | 三处 load_guard 放行 H-gate；新增 `--rit-shadow-out/--rit-warm-ts/--rit-weights/--rit-h-exec` |

⚠ **为什么不做离线回放**：采集 h5 存的是 `input_embeds` 被 `slice_groot_cp1_fields` 切开后的分片，
state token 与 image-token 的 scatter 位置没有留下 ⇒ 离线重建 LLM 输入需要重新 tokenize 并猜版式。
shadow 在真 stage-1 张量还在手上时直接标注，把这层重建整个消掉。

### 为什么部署形态是 `threshold + warm_tiers` 而不是 `dispatch_surface`
RIT 是 s-only 的：`cut_at(fit, tier, delta)` 出来的就是分数切点，阶梯等价于一组嵌套阈值。
而 GR00T 的 `load_guard._ALLOWED_JUDGE_TYPES = {threshold, always_hit, always_warm_start}`
**不含 `dispatch_surface`** —— 用那个类型 server 起不来。两个 teacher 都用 threshold 形态，形状对称、可比。
`ThresholdJudge` 按 `warm_tiers` 顺序首个命中即返回，所以 tier 必须按**成本升序 = 阈值降序**写。


## 4. GR00T 线的执行管线（脚本已全部就位）

| 步 | 位置 | 命令 |
|---|---|---|
| 标定采集（teacher 地板臂 + h5 备份） | timan107 `tmux ritcal` | `/tmp/rit/launch_calib_groot.sh` |
| normalizer 重标定（W13 库） | weilandserver `tmux ritcalib` | `calibrate_score_normalizers.py --artifact-dir /tmp/rit/w13_full` |
| 库动作权重 | weilandserver | `python -m exp.robocasa365.rit_shadow --library <full.pkl> --out lib_weights_*.npz` |
| 发标定 cell | weilandserver | `emit_rit_rc.py calib --teacher groot_tp ...` |
| shadow 标定跑 | wls `/tmp/rit/start_groot_shadow.sh` + t107 `/tmp/rit/launch_shadow_groot.sh` | 5 server × 5 worker，单连接 |
| 拟合 + 发 arm | weilandserver | `emit_rit_rc.py arms --shadow <jsonl> --cost <json> --n-targets 6` |
| 主跑 | wls `/tmp/rit/start_groot_eval.sh` + t107 `/tmp/rit/launch_eval_groot.sh` | 5 server（concurrent+dynamic bundles）× 30 worker |

**驱动侧的两处改动**（`run_ws_search2.py`，L1）：
- `--run-prefix` 增加 `rit`（`resolve_cells` 与 ws2 同走 index.json）；
- 新增 `--index-provenance`：pnp lane 必须带 `--pinned-objects`，而它会触发 ws2 的双 teacher
  index-digest 前置检查 —— 那个 digest 覆盖两个 teacher、且校验 ws2 emitter 的源码哈希，
  RIT 树不是它发的。改成校验 RIT emitter 自己写的 `provenance.json`（逐 yaml sha256），
  **反漂移保证不降级**，只是换成了拥有这棵树的 emitter 写的冻结记录。


## 5. 逐任务标定原料（owner 2026-09-08 追加要求）

shadow 的每一行本来就带 `task` / `episode_id` / `step_idx` + `s` + 每档 `y_*`，
所以「按任务各自定 threshold」这个变体**不需要第二次标定跑**，直接从同一份 JSONL 分组重拟合即可。
为此做了两件事：

1. shadow 的输出落 **`/data/robocasa365_cache/rit_calib/<teacher>/shadow_g*.jsonl`**（持久盘），不再放 `/tmp`。
2. `emit_rit_rc.py arms` 的记录里新增 `per_task` 段：逐任务的行数、分数分位（0/5/15/50/85/95/100）、
   各档 `y_*` 的均值与 q95，以及**「若按任务各自切」的 gate θ**。本轮部署的仍是**全局单一阶梯**，
   逐任务的 θ 只记录、不启用。

## 6. 已入库

`5ab6084` *Address the RoboCasa warm-start frontier by inference ratio* —— 302 个文件，
含 W13 遗留（manifest 合并器、档位切分器、pnp 钉死判决、权重搜索网页）与本轮 RIT 全套代码。
按 owner 指示排除 `exp/rit_pareto/build_figure.py` 与 `edit_figure.py`（另一会话在改）。已 push 到 `Ziyang`。


## 7. GR00T 标定结果（P1a DONE，2026-09-08 12:53 CDT）

语料 130/130 集（13 任务 × 10），37 GB，落 `/data/robocasa365_cache/calib_rit_w13/groot_tp/`。
这一跑**同时就是 teacher 地板臂**（teacher-only、评测段、与拟合同一批 seed），记录在
`exp/robocasa365/data/calib_rit/teacher_floor_groot_tp.json`：

| 任务 | SR | 任务 | SR |
|---|---|---|---|
| OpenCabinet | 1.00 | PickPlaceCounterToStove | 0.90 |
| OpenStandMixerHead | 0.90 | PickPlaceSinkToCounter | 0.90 |
| CoffeeSetupMug | 0.80 | PickPlaceToasterToCounter | 0.90 |
| CloseBlenderLid | 0.70 | PickPlaceCounterToCabinet | 0.70 |
| CloseFridge | 0.70 | PickPlaceDrawerToCounter | 0.70 |
| OpenDrawer | 0.60 | SlideDishwasherRack | 0.30 |
| TurnOnSinkFaucet | 0.60 | **macro** | **0.7462** |

⚠ 两个跑出来的坑（三个启动脚本都已修）：
1. 两条 lane 共用 `--run-plan-dir` ⇒ pnp 的 b01 与 main 的 b01 撞 plan_hash，驱动拒绝续跑。改成 `$DATA/main` 与 `$DATA/pnp`。
2. `rc=$?` 取的是 `tee` 的退出码，把上面那次失败报成 rc=0。改 `set -o pipefail` + `${PIPESTATUS[0]}`。

## 8. pi0.5 侧的标定路线（与 GR00T 不同，且更省事）

GR00T 必须走在线 shadow，因为它的 `input_embeds` 被切成分片后 scatter 位置没留下、离线无法重建。
**pi0.5 不同**：它的 stage-1 前缀就是 vision token + prompt token 的拼接，
`exp/common/build_in_memory_cache_artifact._build_fake_stage1_with_masks` 正是为此写的，
而 `exp/dispatch_surface/build_dispatch_table.py` 已经把「回放 → s → 逐档 `run_stage3_from` → y」整条实现了。
⇒ **pi0.5 不改 server**：用现成的 `serve_policy --collect` 采标定语料，再写一份 RoboCasa 版建表脚本复用那条回放。
pi0.5 主跑同样用现成的 `serve_policy --cache_config` + 动态 bundle。


## 9. shadow 冒烟的两个观察（要靠全量分布复核）

单集（OpenDrawer，102 步）：

- **分数几乎不散**：s ∈ [0.9965, 0.9986]。但这是单任务内的 top-1，text-IVF 把检索圈在同一个
  prompt 桶里，所以同任务的最近邻本来就很近。全量跑里已见到 0.9329，跨任务分布确实更宽。
  ⚠ 若最终分布仍然极窄，LP 会把 q(s) 压到严格单调下限上，切点就是「被 eps 地板抬出来的」而非数据支撑的
  —— 这正是记录里 `floor.on_eps_floor` 要回答的问题。
- **三档偏差几乎相同**：y_full 6.576 / y_rem1 6.501 / y_rem2 6.423（加权 L2，32 维全活跃）。
  顺序对（重跑越多步越接近 teacher），说明 `run_stage3_from` 确实吃到了查询的条件；
  量级小是 4 步循环的结构决定的：从 t=0.75 续跑只重做全程的 25%，能纠正的本来就有限。
  ⇒ 预期帕累托上 warm 两档相对 FULL_HIT 的精度增益很小，而成本是它的 3.2-3.6 倍。这是结论，不是 bug。

## 10. pi0.5 建表脚本已就位

`exp/robocasa365/build_rit_table_pi05_rc.py`：复用 `_build_fake_stage1_with_masks` +
`_load_pi05_for_llm_extract` + `_load_components`，逐步重建 pi0.5 的 stage-1 前缀 → 生产 key builder
与检索 → 逐档 `run_stage3_from` 算偏差。参考动作用查询步自己的 `clean_action`，与 GR00T shadow 同口径。


## 11. GR00T 阶梯拟合结果（P1b DONE，2026-09-08 13:58 CDT）

标定 **12,817 行 / 13 任务 / 130 集**，全部有分数与三档风险。分布：
`s` 分位 0/5/15/50/85/95/100 = 0.469 / 0.821 / 0.960 / 0.9965 / 0.9981 / 0.9983 / 0.9987。

风险随分数下降但**很弱**：`corr(s, y_full) = −0.133`、`y_rem1 = −0.042`、`y_rem2 = +0.028`。
按分数五分位的 y_full = 7.43 / 7.12 / 6.73 / 6.62 / 6.67 —— 前四段单调、高分段饱和。
三档均值 6.91 / 6.68 / 6.56，到 q95 拉开成 8.80 / 8.09 / 7.64 ⇒ **warm 的收益集中在检索最差的尾部**。

**IR 网格**（k=2 与 k=3 可达区间都是 [23.02, 100]，取交集铺 6 点）：
23.02 / 38.40 / 53.79 / 69.18 / 84.56 / 99.95。**H-gate θ = 0.963402**（15 分位，n=12,817）。

| arm | k | 预测 IR | delta | 各档切点 | 落在 eps 地板上的档 |
|---|---|---|---|---|---|
| `always_hit` | — | 23.02 | — | — | — |
| `k2__ir023.0` | 2 | 23.02 | 10.288 | full 0.3673 / warm75 0.3673 | full,warm75 |
| `k2__ir038.4` | 2 | 38.42 | 8.743 | full 0.9890 / warm75 0.3673 | full,warm75 |
| `k2__ir053.8` | 2 | 53.82 | 8.102 | full 0.9960 / warm75 0.9826 | warm75 |
| `k2__ir069.2` | 2 | 69.19 | 7.975 | full 0.9975 / warm75 0.9952 | full |
| `k2__ir084.6` | 2 | 84.54 | 7.975 | full 0.9984 / warm75 0.9952 | full |
| `k2__ir100.0` | 2 | 99.99 | 7.769 | full ∞ / warm75 0.9987 | warm75 |
| `k3__ir023.0` | 3 | 23.02 | 10.288 | full 0.3673 / warm50 0.3673 / warm75 0.3673 | full,warm50,warm75 |
| `k3__ir038.4` | 3 | 38.41 | 8.743 | full 0.9890 / warm50 0.3673 / warm75 0.3673 | full,warm50,warm75 |
| `k3__ir053.8` | 3 | 53.77 | 7.975 | full 0.9965 / warm50 0.3673 / warm75 0.9952 | full,warm50 |
| `k3__ir069.2` | 3 | 69.17 | 7.975 | full 0.9979 / warm50 0.3673 / warm75 0.9952 | full,warm50 |
| `k3__ir084.6` | 3 | 84.54 | 7.581 | full ∞ / warm50 0.9347 / warm75 ∞ | warm50 |
| `k3__ir100.0` | 3 | 99.96 | 7.579 | full ∞ / warm50 0.9986 / warm75 ∞ | warm50 |

⚠ **多数切点落在 eps 地板上**。这直接来自上面那个弱相关：LP 在数据不支持时把 q(s) 压到严格单调下限，
切点于是由「保证可逆的下限」而非风险差异决定。**这不是拟合失败，是拟合如实反映了信号弱**——
但报告里必须写明哪些切点是地板抬出来的（记录里逐 arm 有 `floor.on_eps_floor`）。

⚠ **不可触发的档已摘除并记录**（`dropped_rungs`）：切点相等意味着上一档已经吃掉全部命中，
下一档永远不会 fire，loader 也拒绝这种形状。摘掉不改成本也不改判决，比把切点硬推开诚实。

### 逐任务 θ（若按任务各自切，仅记录不启用）
差异极大：SlideDishwasherRack **0.812** ↔ PickPlaceSinkToCounter **0.995**，全局单一 θ 是 0.963。
⇒ owner 提的 per-task threshold 变体有很强的动机，原料在 `shadow_all.jsonl` 里，不需重采。


## 12. 评测预算更正（owner 2026-09-08 17:00）

正式口径是**每任务 50 集、每个点 650 集**（13 任务 × 50），不是我先前按的 20 集/任务。
per teacher = 13 点 × 650 = **8,450 集**（main lane 5,200 + pnp lane 3,250）。

已按 20 集/任务跑完的那轮**保留为试跑**（`data/rit/groot_tp/main_pilot20/`，2,080 集），
它的价值是把信噪比问题量化出来了：

- IR=100 的两个 arm 实测 **99.96% / 99.76% 全 MISS**，即纯推理臂，macro SR 0.5687 / 0.5625；
- 全命中臂 0.6000；这 8 个接触任务的 teacher 地板 0.7000。
  ⇒ **整条曲线的 y 动态范围只有约 0.10**。
- 20 集/任务时 macro SE ≈ 0.039，arm 间差别小于 ~0.11（3σ）分辨不出来，
  而实测跨度只有 0.15 ⇒ 曲线在噪声带里随机游走，这就是「为什么还会下跌」。
- 同一批 seed（episode 0-9）下，标定跑的纯 teacher 0.7000 vs IR=100 arm 0.6125 ——
  **同配置的可复现精度本身约 ±0.09**（n=10/任务）。

50 集/任务把 macro SE 降到 ≈ 0.019（13 任务），是能让曲线读出趋势的最低预算。

⚠ 排除项：H-gate 不是原因。`score_hysteresis` 关闭时那一步**跑全量推理**、不复用动作，
`L=6` 还会在连续 6 次 FULL_HIT 后强制插一次新推理；它只省检索、不改执行。

### 续补口径
main lane 的 uid（`rit-<cid>__l1s1_groot_tp__<Task>:eval:<ord>:<idx>`）在 20 集与 50 集两轮**完全一致**
（同任务表、同顺序），所以把试跑的 journal 带进新目录即可让 driver 的 resume 丢弃已完成 uid、
只跑 20-49，补 3,120 集而不是重跑 5,200 集。⚠ **run plan 不能带过去**：它按 episodes 入哈希，
20 与 50 是两个 plan_hash，带过去会被拒绝续跑。

⚠ 另一个踩到的坑：两条 lane 共用 `--data-dir` 时 run plan 按 cell id 命名而不含 lane，
第二条 lane 一启动就撞 plan_hash。已改成每 lane 一个数据目录。
