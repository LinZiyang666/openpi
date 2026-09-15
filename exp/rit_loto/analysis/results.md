# RIT 离线标定（LOTO）× GR00T N1.5 × LIBERO — 正式运行结果

运行日期：2026-09-14（America/Chicago，17:15–20:35）。代码 `d50621e`（`origin/Ziyang`）。计划与冻结裁定：`logs/rit_loto_calibration_plan.log.md` §2/§5/§8；运行手册 `logs/session_handoff.md` §1。
本文只转录产物里的数字与判读口径；不宣称"LOTO 已证明可替代 shadow"。

## 0. 运行环境与偏差记录

| 项 | 值 |
|---|---|
| 计算节点 | weilandserver（RTX 4090 49 GB）：噪声地板 / parity / 全表 / 拟合 / 池臂 / 任务 3 server；timan107（8×1080）：任务 3 的 10 个 LIBERO client |
| 代码树 | **`/data/openpi_loto`**（`/data/openpi_lg` 的整树拷贝）。23 个 `CODE_FILES` 全部与 HEAD `d50621e` 内容一致（`cat f \| sha256sum` 双侧对账）；本线 18 个文件与提交版一致，**例外**：`ops/run_noise_floor.sh`、`ops/run_loto_table.sh`、`ops/launch_verify_server.sh` 各改一行 `REPO=/data/openpi_loto`（ops 脚本不在代码身份清单内） |
| 为何另起代码树 | `/data/openpi_lg` 被别线当运行树，且别线当日 15:58 推过与 HEAD 不同的 `src/openpi/cache/groot/staged.py`；本线整条链 fail-closed 于 `require_code_identity`，隔离后全程无代码变动 |
| client launcher 修正 | `ops/launch_verify_clients.sh`（本地工作树已改、未提交）：① timan107 无 `/home/zixuans8/miniconda3/bin/conda`，改为直接调用 `/scratch/zixuans8/libero_sim/bin/python` 并 export `CONDA_PREFIX / PATH / MUJOCO_GL=egl`；② timan107 的 tmux default-shell 是 dash，`${PIPESTATUS[0]}` 是坏替换，`LOTOCLI_EXIT` 改为 `> log 2>&1; echo LOTOCLI_EXIT=$? >> log`。smoke 跑在修正①、未修正②的版本上 |
| 岛 venv / ckpt | `/home/weiland/gr00t_n15_venv`，ckpt `/data/ckpt/n15_libero_{spatial,10}`（内容哈希 `bc4dcc7cec3a…` / `aa21e411f374…`） |
| 库 | `/data/libero_cache/libraries_w13/<suite>/<suite>_w13_S3.pkl`（spatial 1,078 条/50 轨迹 `7e8993793489…`；libero_10 2,598 条/50 轨迹 `9ef8d73145d4…`） |
| 语料 | `/archive/libero_cache/build_{spatial,libero10}_w13/<suite>`（各 500 集；manifest `8b9d0af75648…` / `57c3b7181f54…`） |
| 度量 | Eq. 16：W = `library_action_weights(pkl)`（全库共享，`n_active_dims = 32`，GR00T 32 维输出无一维方差低于阈值；W sha `13e5f7647dcd…` / `83025863d328…`），只比前 `H_exec = 5` 步 |
| 阶梯 | `groot_n15_k8_v1`，warm 档 t = 0.75（`y_rem2`）/ 0.5（`y_rem4`），FULL `y_full`；α = 0.05 |
| 成本 | `exp/libero_groot/config/rit/cost_groot_libero_measured.json`（stage1 6.146 / stage2 7.192 / stage3 3.513 ms·step，线性） |

产物根：weilandserver `/data/libero_cache/rit_loto/<suite>/`；本地副本 `exp/rit_loto/data/<suite>/`（目录不入库）。

## 1. 任务 4 噪声地板（`noise_floor.py`，每 suite 500 决策 = 50/任务 × 10，两噪声种子）

| suite | n | D(ref1, ref2) 中位 / p90 / p95 | D(ref1, clean_action) 中位 / p90 | 用时 |
|---|---|---|---|---|
| libero_spatial | 500 | **6.826** / 7.369 / 7.510 | 6.821 / 7.336 | 266 s |
| libero_10 | 500 | **6.926** / 7.540 / 7.715 | 6.905 / 7.514 | 264 s |

判读：两个随机种子的完整推理之间的 D 与 shadow 各档的 D（5.5–9.1）同量级，即 **D 度量的绝对值被教师自身随机性主导**；D(ref1, clean) ≈ D(ref1, ref2) 说明采集时的 `clean_action` 与一次新的完整推理的差异也只是随机性。不据此改任何阈值（计划口径）。

sha256：`noise_floor.json` spatial `3c69b0f6bc95d8a6…`，libero_10 `72683c204a68d1d0…`。

## 2. 任务 1 parity 门与 LOTO 表（`build_loto_table.py`）

### 2.1 parity 门（200 行 = 20/任务；阈值 = 0.1 × 地板中位数）

| suite | 阈值 | FULL p90 | warm75 p90 | warm50 p90 | 状态 |
|---|---|---|---|---|---|
| libero_spatial | 0.6826 | 0.0 | 0.0 | 0.0 | **PASS** |
| libero_10 | 0.6926 | 0.0 | 0.0 | 0.0 | **PASS** |

parity_D 恒为 0：语料里 fp16 存的中间态对 bf16 去噪无损、去噪确定，离线续跑与采集时完全一致（与冒烟一致）。sha256：`parity_gate.json` `045542f5d1af640f…` / `a350ea7dca8293de…`。

### 2.2 全表

| suite | 集数 | 行数 | 库内行 | self_skipped 总数 | orchestrator 自证 | prompt 置换自证 | no_hit | 用时 |
|---|---|---|---|---|---|---|---|---|
| libero_spatial | 500 | **11,838** | 1,078 | 1,890 | 53 / 0 mismatch | 53 / 0 | 0 | 2,293 s（4.9 行/s，与 libero_10 并跑） |
| libero_10 | 500 | **29,318** | 2,598 | 10,584 | 136 / 0 mismatch | 136 / 0 | 0 | 6,383 s |

行字段：`suite, trajectory_id, episode_id, decision_id, task, task_id, orig_init_state_idx, library_id, library_seed, s, candidate_id, y_full, y_rem2, y_rem4, in_library, episode_success, n_self_skipped`（`s` 即 s_t；`y_*` 即 D_a）。库内行 = 1,078 / 2,598，与库条目数相同（每条库条目正是一条库内决策）。

sha256：`loto_table.jsonl` spatial `de8dd6c572076935…`，libero_10 `e1bd1b8c3b79860e…`。

## 3. 任务 2 等价性对比（`fit_loto.py`，只调现有 `fit_ladders` LP，α = 0.05，K ∈ {1,2,3}）

原 shadow（150 集）按 `arm_record.json` 的 knots 复拟合并审计：alpha 0.05、48 臂 / 96 个 cut 复现一致（`shadow.audit`）。行数：

| suite | LOTO 全部 | 库内 | 库外 | 其中失败集行 | shadow | shadow 失败集行 |
|---|---|---|---|---|---|---|
| libero_spatial | 11,838 | 1,078 | 10,760 | 2,200（18.6%） | 3,432 | 396（11.5%） |
| libero_10 | 29,318 | 2,598 | 26,720 | 6,656（22.7%） | 8,383 | 1,248（14.9%） |

### 3.1 LOTO-all vs shadow：曲线原始差值（`compare.json`，公共支撑上的 knot 并集逐点 max|Δq|；τ_b = shadow 拟合行上 |D − q̂| 的 p90）

| suite | K | tier | max\|Δq\| | 位置 s | 符号（LOTO − shadow） | τ_b | max\|Δq\|/τ |
|---|---|---|---|---|---|---|---|
| spatial | 3 | full | 0.359 | 0.9752 | − | 1.548 | 0.23 |
| spatial | 3 | warm75 | 0.244 | 0.6090 | + | 1.348 | 0.18 |
| spatial | 3 | warm50 | 0.236 | 0.6090 | + | 1.333 | 0.18 |
| libero_10 | 3 | full | 0.368 | 0.9721 | + | 1.471 | 0.25 |
| libero_10 | 3 | warm75 | 0.550 | 0.6549 | + | 1.325 | 0.41 |
| libero_10 | 3 | warm50 | 0.245 | 0.6549 | + | 1.296 | 0.19 |

K = 1 / 2 的同 tier 数字与 K = 3 相同（nesting 约束下同一曲线）。所有最大差都出现在低分尾（s ≤ 0.975），即 shadow 样本最稀的区间；在 s ≥ 0.985（spatial 85% 质量）/ s ≥ 0.994（libero_10 81% 质量），同 s 分箱的经验 p95 三档差 ≤ 0.04 / ≤ 0.02（见 3.4）。

### 3.2 参考带覆盖（`bootstrap_band.json`：shadow 拟合的任务分层 episode bootstrap，B = 200、95% 逐点、200/200 有效；LOTO 曲线落在带内的网格点比例，K = 3）

| suite | full | warm75 | warm50 |
|---|---|---|---|
| libero_spatial | 0.746 | 0.690 | 0.675 |
| libero_10 | 0.918 | 0.949 | 0.913 |

带是 shadow **自身抽样变异**的参考，不是 LOTO-vs-shadow 差异的检验（产物 `note` 原文）。spatial 的 FULL 档出带集中在 s ∈ [0.917, 0.975]（LOTO 低 0.1–0.36）；libero_10 的出带集中在 s < 0.97（LOTO 高 0.3–0.55，该区 shadow 带在 0.92 以下无定义）。

### 3.3 ψ(s) 分数边缘

| suite | 对比 | KS | p | mass(s ≥ 0.99) a / b |
|---|---|---|---|---|
| spatial | LOTO-all vs shadow | 0.050 | 2.6e-6 | 0.502 / 0.533 |
| spatial | 库内 vs 库外 | 0.124 | 1.2e-13 | 0.564 / 0.496 |
| libero_10 | LOTO-all vs shadow | 0.059 | 5.2e-20 | 0.923 / 0.958 |
| libero_10 | 库内 vs 库外 | 0.094 | 1.2e-18 | 0.992 / 0.916 |

只取成功集行再算（`scipy.stats.ks_2samp` 复算）：spatial 0.030、libero_10 **0.013**；只取库外成功行：0.027 / 0.017。即 LOTO-all 与 shadow 的 ψ(s) 差异主要来自失败集行占比（18.6% vs 11.5%；22.7% vs 14.9%）：失败轨迹走到低分状态，使离线 ψ(s) 的低分尾更重（5% 分位 spatial 0.883 vs 0.940；libero_10 0.978 vs 0.993），中位数几乎相同（0.9900 vs 0.9902；0.9984 vs 0.9985）。

### 3.4 同 s 分箱的经验 p95（库外行 / shadow；复算脚本见 §6）

libero_spatial：

| s 区间 | n | FULL | warm75 | warm50 |
|---|---|---|---|---|
| [0, 0.95) | 1112 / 187 | 8.49 / 8.64 | 7.73 / 7.66 | 7.62 / 7.56 |
| [0.95, 0.98) | 606 / 167 | 8.24 / 8.57 | 7.55 / 7.62 | 7.48 / 7.59 |
| [0.98, 0.985) | 529 / 179 | 8.18 / 7.84 | 7.57 / 7.48 | 7.52 / 7.47 |
| [0.985, 0.99) | 3180 / 1071 | 7.94 / 7.96 | 7.59 / 7.58 | 7.58 / 7.55 |
| [0.99, 0.992) | 3191 / 1017 | 7.78 / 7.82 | 7.59 / 7.57 | 7.57 / 7.57 |
| [0.992, 0.994) | 2078 / 771 | 7.69 / 7.70 | 7.58 / 7.56 | 7.57 / 7.54 |

libero_10：

| s 区间 | n | FULL | warm75 | warm50 |
|---|---|---|---|---|
| [0, 0.95) | 793 / 99 | 8.51 / 8.09 | 8.00 / 7.76 | 7.78 / 7.61 |
| [0.95, 0.98) | 769 / 113 | 8.64 / 8.21 | 7.96 / 7.95 | 7.75 / 7.63 |
| [0.98, 0.985) | 280 / 65 | 8.50 / 7.97 | 7.75 / 7.63 | 7.59 / 7.62 |
| [0.985, 0.99) | 400 / 76 | 8.37 / 8.12 | 7.78 / 7.57 | 7.67 / 7.55 |
| [0.99, 0.992) | 258 / 43 | 8.36 / 8.20 | 7.71 / 7.43 | 7.63 / 7.38 |
| [0.992, 0.994) | 457 / 77 | 8.33 / 8.20 | 7.80 / 7.70 | 7.72 / 7.66 |
| [0.994, 1.0] | 23763 / 7910 | 7.95 / 7.97 | 7.72 / 7.72 | 7.69 / 7.68 |

无条件 D 分布 KS（库外 vs shadow）：spatial 0.032 / 0.037 / 0.040，libero_10 0.019 / 0.008 / 0.012（FULL / warm75 / warm50）；库内 vs shadow：spatial 0.088 / 0.064 / 0.057，libero_10 0.032 / 0.030 / 0.020。

### 3.5 库内 / 库外分层

| suite | tier | max\|Δq\|（库内 − 库外，公共支撑） | 库内支撑 |
|---|---|---|---|
| spatial | full / warm75 / warm50 | +0.352 / +0.193 / +0.172 | s ∈ [0.941, 0.994] |
| libero_10 | full / warm75 / warm50 | −0.321 / −0.163 / −0.015 | s ∈ [0.967, 0.999] |

库内行（整条轨迹已从库中排除）与库外行的差异比库外 vs shadow 大，且两 suite 方向相反；它们只占 9% 行，`fit_source = loto_all`（冻结裁定 D5）里影响有限，部署对应的总体是库外行。

### 3.6 同一成本模型下的 IR 70 寻址（`rit_cost_rc.delta_for_ir` + `cuts_for`，`gate_theta(s)`；复算脚本见 §6）

| suite | K | 来源 | δ | cuts（FULL, warm75, warm50；None = +∞ 不派发） | θ |
|---|---|---|---|---|---|
| spatial | 2 | LOTO-all | 7.5938 | [0.99453, 0.98925] | 0.98027 |
| spatial | 2 | shadow 复拟合 | 7.5906 | [None, 0.98955] | 0.98463 |
| spatial | 3 | LOTO-all | 7.5722 | [None, None, 0.96768] | 0.98027 |
| spatial | 3 | shadow 复拟合 | 7.5617 | [None, 0.99076, 0.98872] | 0.98463 |
| libero_10 | 2 | **LOTO-all（冻结臂 `loto_k2_ir70`）** | **7.7133** | **[None, 0.99822]** | **0.99640** |
| libero_10 | 2 | shadow 复拟合 | 7.7184 | [None, 0.99831] | 0.99731 |
| libero_10 | 3 | LOTO-all | 7.6884 | [None, None, 0.99494] | 0.99640 |
| libero_10 | 3 | shadow 复拟合 | 7.6779 | [None, 0.99890, 0.99746] | 0.99731 |

K = 1 两 suite 的 cut 差 ≤ 0.0002。原 `arm_record.json` 里的 `l10_rit_k2_ir70`（δ 7.7596，cuts [0.99892, 0.99812]）是用 INTERIM 成本表寻址的，与本表的成本模型不同，不能直接比。

图：按 owner 指示（2026-09-14）只保留 `exp/rit_loto/analysis/figures/main_run_q_libero_10_k3.{png,pdf}`（主实验 shadow 拟合的 libero_10 K=3 q 曲线，x ∈ [25% 分位, max]，y 7.5–8.05）；其余对比图已删，渲染脚本 `analysis/plot_*.py` 保留、不入库。

## 4. 任务 3 闭环验证（LIBERO-10，冻结臂 `loto_k2_ir70`，gate 开，`verify_closed_loop.py`）

冻结记录 `frozen_run.json`（`1e83bbd578d707b6…`）：`fit_source = loto_all`、K = 2、目标 IR 70、α = 0.05；臂 yaml sha `877773f9472ba154…`；池 = A 池排除 shadow cohort 后每任务 5 正式 + 1 smoke（seed 20260913，`verify_pool_manifest.json` `c043005a31ba0d28…`）；`compile_stage1 = false`。server：weilandserver `serve_groot_libero.py --concurrent --cache-config loto_k2_ir70.yaml --loto-log-out … --loto-frozen-record …`（端口 23150）；client：timan107 10 进程（一任务一进程，`--episode-filter` 官方 init 序号，`--replan-steps 5 --resize-size 256`）。

### 4.1 smoke（10 集，run_tag `smoke`，19:49–19:57）

10 个 h5 / 0 个 `.h5.tmp`；每个 attrs 含 `loto_frozen_record_sha256 / loto_arm_yaml_sha256 / loto_library_sha256 / loto_checkpoint_identity_sha256 / loto_fits_sha256 / loto_pool_manifest_sha256 / h_exec / run_tag / orig_init_state_idx`；sidecar 652 行 = Σ `num_steps`；hit 混合 WARM_START 435 / MISS 217（FULL 档在该臂 cut = ∞，故不会出现）；8 / 10 成功。→ ACCEPT。smoke server 按 PID 关闭后另起 verify server（不同 run_tag、不同进程）。

### 4.2 verify 采集（50 集，run_tag `verify`，19:55–20:15）

`collect`：50 集（每任务 5、官方序号与池清单逐一相符）、**3,108 决策**：WARM_START 1,594（51.3%）/ MISS 1,514 / FULL 0；成功 **39 / 50**（每任务 4/5/4/5/3/5/3/4/2/4）；10 个 client 全部 `LOTOCLI_EXIT=0`。抽样（seed 20260914）：base = 均匀 2,000（含 WARM 1,014）+ warm_extra 580（其余全部 WARM）= **2,580 行**；`inclusion_probability` base 0.6435、warm 1.0。

`label`（岛 venv，GPU，709 s）：2,580 行全部找到候选（`no_candidate = 0`）；在线一致性 2,580 行全检，其中 2,316 行有在线 s（`online_confirmed`），离线重算的 s 与在线 s **max |Δs| = 0**；264 行在线未记 s（`offline_only`）。每行同时给 `y_full` 与 `y_rem2`（K = 2 两档标签：WARM 行 = (候选 chunk, 执行 chunk)，MISS 行 = (候选 chunk, 候选 warm)）。

### 4.3 report（`verify.json` `387c31579797cf66…`；α = 0.05、容差 0.05、通过上界 0.10、B = 1000、seed 0；信息门 min_rows 200 / min_episodes 20 / min_event_episodes 5 / min_valid_fraction 0.9）

| tier | 支撑内行数 | 超越数 | 点估计 | 95% cluster bootstrap 区间 | 有集数 / 含超越 / 含非超越 | 判读（原文） |
|---|---|---|---|---|---|---|
| full | 0 | — | — | — | — | **insufficient_evidence**（`no in-support rows`：该臂 FULL cut = ∞，设计内） |
| warm75 | 1,594 | 109 | **0.0684** | **[0.0563, 0.0826]**（1000/1000 有效，非退化） | 50 / 44 / 50 | **within_preset_tolerance_in_support** |

按分数四分位分箱的 warm75 超越率：[0.99822, 0.99847) 0.095（n=399）、[0.99847, 0.99865) 0.065（398）、[0.99865, 0.99883) 0.050（398）、[0.99883, 0.99919] 0.063（399）。按任务（超越/派发）：0.053 / 0.070 / 0.050 / 0.056 / 0.095 / 0.089 / 0.066 / 0.066 / 0.054 / 0.072。派发 WARM 行的 D 中位 6.956、p95 7.774（臂 δ = 7.713）；MISS 行的反事实 warm75 D p95 7.764（高于 δ 的比例 0.059）、反事实 FULL D p95 8.263。

联合复拟合（2,000 base 行，`fit_ladders` 同 LP）：warm75 曲线在派发区比 LOTO 曲线高约 0.05（见图）。运行摘要：`realized_ir_measured_cost = 73.91`（按实测成本表折算，非墙钟；臂预测 70.01，闭环 WARM 命中率 51.3%）。

闭环叠加图未保留（owner 指示只留主实验 q 曲线图）；可用 `analysis/plot_closed_loop.py` 重新渲染。

`reading_legend` 原文：`within_preset_tolerance_in_support` = "interval upper bound <= alpha + tol on in-support rows (approximate cluster-bootstrap diagnostic, not a proof of exact alpha calibration)"。

## 5. 结论口径（只转录，不外推）

1. **噪声地板**：D(ref1, ref2) 中位 6.83 / 6.93（p90 7.37 / 7.54），与各档 D 同量级；Eq. 16 在 GR00T 上的绝对值被教师随机性主导，q 曲线只比地板 p95 高 0.1–1.1。
2. **parity 门**两 suite PASS 且 parity_D 恒 0：离线从存储中间态续跑与采集时逐位一致。
3. **LOTO 全表**两 suite 各 500 集全量（11,838 / 29,318 行），自证全部零 mismatch。
4. **等价性**：LOTO-all 与 shadow 复拟合的 q_a(s) 原始差 max|Δq| ≤ 0.37（≤ 0.25 τ）、warm75 在 libero_10 低分尾 0.55（0.41 τ）；差异集中在 shadow 样本稀的低分区；工作区（spatial s ≥ 0.985、libero_10 s ≥ 0.994）同分箱 p95 差 ≤ 0.04 / ≤ 0.02。参考带覆盖 spatial 0.68–0.75、libero_10 0.91–0.95。ψ(s) 的差异主要是失败集占比（成功行 KS 0.030 / 0.013）。同成本模型下 IR 70 寻址：libero_10 K=2 的 warm75 cut 0.99822（LOTO）vs 0.99831（shadow 复拟合）。
5. **闭环**：LOTO 标定的 K=2/IR70 臂在 50 集 LIBERO-10 上，派发档 warm75 的超越率点估计 0.068、95% cluster bootstrap 区间 [0.056, 0.083]，落在 α + 容差 = 0.10 之内 → `within_preset_tolerance_in_support`；FULL 档无派发 → `insufficient_evidence`。这是"以固定库/曲线/已选决策为条件的近似诊断"，不是精确 α 校准的证明。
6. **未说明的**：本轮没有跑 shadow 标定臂的对照闭环，因此不能说 LOTO 与 shadow 在闭环上等价，只能说 LOTO 臂的闭环超越率在预设容差内。

## 6. 产物清单与复算命令

本地副本（`exp/rit_loto/data/`，不入库；sha256 前 16 位）：

| 文件 | spatial | libero_10 |
|---|---|---|
| noise_floor.json | 3c69b0f6bc95d8a6 | 72683c204a68d1d0 |
| parity_gate.json | 045542f5d1af640f | a350ea7dca8293de |
| loto_table.jsonl | de8dd6c572076935 | e1bd1b8c3b79860e |
| loto_table.jsonl.record.json | dbb94823f33b99dd | 9a44cd3ab3af18ca |
| fits.json | 608703f5d24d3373 | 97c6a1bf8623d839 |
| compare.json | 6dd7abb6a040a9e7 | cec48efc6a9e1600 |
| bootstrap_band.json | 521924d0e17b04f8 | f39436d8f3694501 |
| frozen_run.json | — | 1e83bbd578d707b6 |
| verify_pool_manifest.json | — | c043005a31ba0d28 |
| config/libero_10/loto_k2_ir70.yaml | — | 877773f9472ba154 |
| verify/decisions.jsonl / episodes.json / sample_manifest.json | — | e5957f906aa164cf / c735a164d8996605 / 09939ab5e8ee3025 |
| verify/verify_rows.jsonl / verify_labels.record.json / verify.json | — | 61bad6e495ed8783 / 08c838b209f410ec / 387c31579797cf66 |

weilandserver 原件：`/data/libero_cache/rit_loto/<suite>/`（含 `verify_logs/{smoke,verify}/conn_*/` 的 60 个 H5，不拉回）。

复算（岛上 `REPO=/data/openpi_loto`，主 venv `PYTHONPATH=$REPO/src:$REPO`，岛 venv 另加 gr00t 路径；命令原文见 `logs/session_handoff.md` §1.3 与 `exp/rit_loto/ops/{run_noise_floor,run_loto_table,run_fit,run_pools_arm,launch_verify_server,launch_verify_clients,run_clr}.sh`）：

```
bash ops/run_noise_floor.sh <suite>                      # 步 1
bash ops/run_loto_table.sh <suite> parity                # 步 2（需 noise_floor.json）
bash ops/run_loto_table.sh <suite> full                  # 步 3（需 PASS 的 parity_gate.json）
bash ops/run_fit.sh <suite>                              # 步 4
bash ops/run_pools_arm.sh                                # 步 5（libero_10）
bash ops/launch_verify_server.sh <arm.yaml> smoke|verify <frozen_run.json> 23150 srv0   # 步 6 server
bash ops/launch_verify_clients.sh ziyanglin.com 23150 smoke|verify <pool> <filter.json> 1|5   # 步 6 client（timan107）
bash ops/run_clr.sh collect|label|report                 # 步 7
```

§3.3 的成功集 KS、§3.4 分箱 p95、§3.6 寻址表：`scipy.stats.ks_2samp` / `numpy.quantile` 直接作用于 `loto_table.jsonl` 与 `shadow_rows.jsonl`；寻址用 `fit_loto.deserialize_fit` + `rit_cost_rc.delta_for_ir` / `cuts_for` + `emit_rit_arms.gate_theta`，输入为 `fits.json` 各来源的 `fits[k]` 与 `s_sample`。§3.2 带内比例：`rit_k.predict` 在 `bootstrap_band.json` 的 `grid` 上求值后与 `lo/hi` 比较。
