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
| 评测 | 每任务 20 集、每 cell 260 集；14 cell/teacher（12 RIT + always_hit + teacher 地板） |
| 拓扑 | server 只在 weilandserver：GR00T 5 进程 23160-23164；pi0.5 一端口 23170 `--replicas 4`。worker 只在 timan107，30 个 |
| 顺序 | **先 GR00T，全跑完再 pi0.5** |

## 1. 进度条

```
P0a 延迟测量      [##########]  DONE   GR00T 四档全 nsys 认证，s3(k)=5.232+3.006k，R²=0.991
P0b 成本权威      [##########]  DONE   rit_cost_rc.py：按 schedule 方向取剩余步数
P0c RIT 链路移植  [########..]  进行中 shadow + emitter 已写并合成验算通过，待真数据
P1a GR00T 标定    [####......]  进行中 5 server × 5 worker，130 集，约 53 min
P1b GR00T 拟合发射[..........]  待 P1a
P1c GR00T 主跑    [..........]  待 P1b
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
