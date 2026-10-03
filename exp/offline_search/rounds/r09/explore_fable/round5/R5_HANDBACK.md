# Round 5（fable）交接 — 堆叠方案的"看一眼"成本，批次 `r09_fable_r5`（已备好，**未发射**）

## 给 owner 的白话结论
- 这一轮研究"每 10 步看一眼双目相机"这笔固定开销（约占堆叠方案 .16–.18 开销中的 .075）。两条省法：① π0.5 在合适的时候只看腕部相机（便宜一半多）；② 门控跟随：
  条件合适就再多播 5 步缓存动作、跳过那次看一眼。两条都不看任务是谁，只看机器人状态、夹爪阶段结构、节奏落后量。
- 先说盘子大小：全部换腕眼最多省 .034，容易阶段腕眼省 .014，门控跟随省 .013 —— 都比"叫策略"那笔（.085）小。所以成败只看成功率有没有掉。
- 离线证据（R8 的延迟相机影子，π0.5，init 0–29）：腕眼检索在"容易阶段 / 节奏正常"的时候几乎等于双目（动作误差 +.004，夹爪命令零不一致），落后时明显更差（+.032，夹爪不一致 9 %）；
  跟随块比重新看一眼平均差 +.013，阶段门把夹爪错误压到 0 但动作误差不变。R8 闭环（纯缓存，init 20–29）也是这个格局：容易阶段腕眼无损、一直腕眼 L10 掉 10 个点、门控跟随小亏 2–5 个点。
- 准备了 6 条新臂 + 2 条同批对照（r3c 的两条领先堆叠原件）。预测写在 PREDICTION_R5.md（2026-10-02 04:04 CDT，早于发射）：容易阶段腕眼不掉点、一直腕眼掉 ≥5、节奏门居中；
  门控跟随 π0.5 小亏、GR00T 亏 ≥3。全部发射前检查通过（工件断言 6/6、CPU 自测 12/12 零 P2 报错、同步计划 63 文件 16.1 GiB、单元测试 5 + round3 18 全过）。
- 一个坦白的风险：腕眼走的"按请求相机"通道在 R7/R8 只跟纯缓存同台过，这是第一次与判官堆叠同台；CPU 自测 harness 没有这条通道，只有用真实录制查询的单元测试覆盖了切换/恢复/跳过修正。建议先跑 esc_weasy。

## 臂（run root `/home/weiland/trace_runs/os_closed_loop/r09_fable_r5`，manifest 任务 0–9 × init 20–29）
| 臂 | 类 | 做法 | 插件旗标差异 | 预测 SR / IR |
|---|---|---|---|---|
| r9f5_pi05_l10_50_esc | NpGraspEsc3（r3c 工件原件） | 同批对照：守卫+半力度修正+升级 | — | .90 / .162 |
| r9f5_pi05_l10_50_esc_wall | LookCostEsc wrist_gate=all | 所有方法计划的看一眼只用腕眼 | +--os-request-cameras --os-tokens off | .86 / .145 |
| r9f5_pi05_l10_50_esc_weasy | LookCostEsc wrist_gate=easy | R7 容易阶段判据通过才用腕眼 | 同上 | .90 / .148 |
| r9f5_pi05_l10_50_esc_wpace | LookCostEsc wrist_gate=pace, pace_lag 1 | 节奏落后 ≤ 1 决策且链/阈值通过才用腕眼 | 同上 | .88 / .136 |
| r9f5_pi05_l10_50_esc_fg | LookCostEsc follow_blocks=1, stage gate | 门控跟随（阶段门+状态阈值），双目 | — | .89 / .150 |
| r9f5_groot_l10_50_np | NpGraspStackGroot3（r3c 工件原件） | 同批对照：守卫+半力度修正 | — | .87 / .183 |
| r9f5_groot_l10_50_np_fg | LookCostGroot follow_blocks=1, stage gate | 门控跟随 | — | .84 / .176 |
| r9f5_groot_l10_50_np_fv | LookCostGroot follow_blocks=1, valve only | 只开状态阈值的跟随 | — | .83 / .170 |
- 其余 kwargs 与 r3c 堆叠完全一致（onlynp_fit=r8abl_onlynp_*_l10_50，corrected_fit=r9f2_*_l10_50_corr05pt，max_calls 0，升级 lag 12 / deadline 80）；
  π0.5 用 R7 冻结的 stages_pi05_l10_50.pkl 与 wrist_pi05_l10_50.pkl（指纹与修正底座的检索一致）；GR00T 阶段表在预拟合时从库按同一检索拟合。
- 腕眼看一眼时：检索在 R7 腕部 72 维度量里（判官的 os_score_all 由腕部适配器回答），修正器跳过（CorrectedCacheJW，wrist_pass），生命周期看一眼仍是双目（插件规则）。
- 门控跟随：守卫的盲段否决先于扩展；只在基座返回"预算到"看一眼时、门通过且最多一块时播后继块；之后强制看一眼（封顶）。
- 全部 kwargs：`r5_kwargs.json`；臂规格 `arms_in.json`；发射行 `arms.json`（8 行，config/ 16 yaml）；来源 `provenance.json`。

## 发射前证据
- `build_r5 check` 6/6：judge.burst 与源一致为 0、gm_burst 2、max_calls 0、单一判官家族、基座 CorrectedCacheJW 且 blend .5、wrist_pass 复位、与守卫同库、
  阶段表检索指纹 = 基座指纹、参考承诺 2 块、腕部度量 72 维、相机模式初始 full、升级字段正确。
- CPU 自测（`/home/weiland/trace_runs/os_closed_loop/r09_fable_r5/selftest/*.log`，4 集，真实判官链，无相机通道）12/12 PASS、0 条 P2；生产 kwargs：fg 臂 4 集播出 20 个跟随块（fv 28 个），weasy 臂规划了 42 次腕眼（harness 无法执行）；强制触发变体也通过。
- 单元测试 `round5/tools/tests/test_round5.py` 5 通过：参数校验；**真实录制查询**上腕眼看一眼 = 核均值（无修正）、双目看一眼 ≠ 核均值（修正生效）、度量字段原样恢复、wrist_pass 复位、第 0 步拒绝腕眼；
  两个模型的门控跟随在真实盲段查询上：普通盲块 → 后继块（os_sf_source>0，last_step 更新）→ 封顶后强制看一眼，无进展否决优先；批次规格（对照指向 r3c 冻结工件、腕眼臂带相机旗标、无任务索引字段、manifest 为评测切分）。
  round3 测试 18 仍全过（r3c/r4 类未改）。
- 同步计划（`/home/weiland/trace_runs/os_closed_loop/r09_fable_r5/plan.log`）rc 0：63 文件 16.1 GiB（含 r3c 两个对照工件、R7 阶段表与腕部度量、6 个本轮 fits）。
- 代码：新文件 `round5/tools/{methods.py,build_r5.py,look_cost.py}`、tests、`round5/{PREDICTION_R5,DATA_ANALYSIS_R5,R5_HANDBACK}.md`；
  h100 镜像 `/data/oscl_h100/openpi/exp/offline_search/rounds/r09/explore_fable/round5/tools/methods.py` sha **e159ac5d…**（与本地一致）；
  依赖 round3 methods fa881a57…、r07 c2_wrist 1629c979…、r07 c1_follow 1d8739b5…、r04 k3_cost 494f8f98… 两端一致。

## 命令（我不发射）
```bash
cd /home/weiland/projects/openpi
R=/home/weiland/trace_runs/os_closed_loop/r09_fable_r5; OPS=$PWD/exp/offline_search/closed_loop/ops/h100
P=(taskset -c 22-25 env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src .venv/bin/python)
ARMS_P="r9f5_pi05_l10_50_esc r9f5_pi05_l10_50_esc_wall r9f5_pi05_l10_50_esc_weasy r9f5_pi05_l10_50_esc_wpace r9f5_pi05_l10_50_esc_fg"
ARMS_G="r9f5_groot_l10_50_np r9f5_groot_l10_50_np_fg r9f5_groot_l10_50_np_fv"
WORKER_HOST=timan107 SYNC_PORT=23195 bash "$OPS/sync_assets.sh" --concurrent "$R" $ARMS_P $ARMS_G
# suggested order: control, weasy (riskiest new path) first, then the rest
WORKER_HOST=timan107 PORTS=23240,23241,23242,23243 WPS=12 MAX_ATTEMPTS=3 POLL_SECONDS=30 OSCL_MANIFEST=$R/manifests/eval100_inits20_29.json \
  bash "$OPS/chain_h100.sh" "$R" r9f5_pi05_l10_50_esc r9f5_pi05_l10_50_esc_weasy r9f5_pi05_l10_50_esc_wpace r9f5_pi05_l10_50_esc_wall r9f5_pi05_l10_50_esc_fg $ARMS_G
# abort the three wrist arms on any server-log line matching 'camera' + Traceback / 'ContractError' / 'P2 does not support'
"${P[@]}" -m exp.offline_search.rounds.r09.explore_fable.round5.tools.build_r5 score     # SR + owner IR (wrist look .0646), inits 20-29 only
"${P[@]}" -m exp.offline_search.rounds.r09.explore_fable.round2.tools.paired_r2 --ref $R:r9f5_pi05_l10_50_esc $R:r9f5_pi05_l10_50_esc_weasy $R:r9f5_pi05_l10_50_esc_wpace $R:r9f5_pi05_l10_50_esc_wall $R:r9f5_pi05_l10_50_esc_fg
"${P[@]}" -m exp.offline_search.rounds.r09.explore_fable.round2.tools.paired_r2 --ref $R:r9f5_groot_l10_50_np $R:r9f5_groot_l10_50_np_fg $R:r9f5_groot_l10_50_np_fv
```
注意：paired_r2 的 IR 用 summary 账本（不分腕眼价），腕眼臂的 owner IR 以 `build_r5 score` 为准（按 camera_mode=wrist_only 计 .0646）。
