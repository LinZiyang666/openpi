# Round 8（fable）交接 — 500 库纯缓存上的"看一眼"节省，批次 `r09_fable_r8`（已备好，**未发射**）

## 给 owner 的白话结论
- round 7 的结果写在 round7/R7_RESULTS.md：GR00T 把兜底推迟消掉了（第一次叫策略只晚 0.8 个决策），成功率照样掉 11 个点、开销还更贵；π0.5 Spatial-50 的腕眼+跟随把节省全交给了多出来的升级调用，掉 8 个点。
  结论同意协调员：**守卫堆叠里少看一眼这条路已经榨干**。
- 本轮换到 R8 说"少看一眼安全"的地方：500 库纯缓存（推荐配置就是缓存 A，开销 .075，没有任何策略调用）。先核实了前提：R8 日志里四个 500 格的 A / 门控跟随 / 腕眼 / 无门跟随臂调用份额全是 0，
  "兜底被推迟"在构造上不存在；但 round 7 也证明 GR00T 的跟随块本身在 50 库上有害，所以 500 库上 GR00T 跟随安不安全只能看 R8 的直接证据（L10-500 −2、Spatial-500 −2，噪声内）。
- 冻结 8 条臂：四个格子各一条缓存对照（R8 的 A 行原样）+ 一条省看一眼臂：GR00T = R8 的门控跟随臂原样（R7 冻结工件）；π0.5 = 新类 PaceWrist（R7 的"容易阶段腕眼"把判据换成"在节奏上"）叠在门控跟随基座上。
- 预测：Spatial-500 两模型和 GR00T L10-500 都不掉点（节省 π0.5 ≈ 一半、GR00T ≈ .01）；π0.5 L10-500 我预测会掉 ≥ 5 个点（长任务 + 三分之一的看一眼只用腕眼，R8 一直腕眼掉 13）。发射前检查全过。

## 臂（run root `/home/weiland/trace_runs/os_closed_loop/r09_fable_r8`，任务 0–9 × init 20–29，无判官、无策略调用）
| 格子 | 对照 | 省看一眼臂 | 预测 SR / IR（对照 → 变体） |
|---|---|---|---|
| π0.5 L10-500 | r9f8_pi05_l10_500_A（R8 A 行） | r9f8_pi05_l10_500_wpace_fg：PaceWrist(pace_lag 1) over StageFollow E1（阶段门+阈值），R7 wrist_pi05_l10_500 / stages_pi05_l10_500；+--os-request-cameras --os-tokens off | .86/.077 → **.80/.040** |
| π0.5 Spatial-500 | r9f8_pi05_spatial_500_A | r9f8_pi05_spatial_500_wpace_fg（同上，Spatial 工件） | .96/.078 → **.96/.035** |
| GR00T L10-500 | r9f8_groot_l10_500_A | r9f8_groot_l10_500_fg = R8 r8_groot_l10_500_SF1 行原样（StageFollow E1，r7_groot_l10_500_SF1.pkl） | .87/.075 → **.85/.068** |
| GR00T Spatial-500 | r9f8_groot_spatial_500_A | r9f8_groot_spatial_500_fg = R8 r8_groot_spatial_500_SF1 行原样 | .97/.075 → **.96/.064** |
- PaceWrist（round8/tools/methods.py，R7 StageWrist 子类，R7 类未改）：看一眼用腕眼当且仅当链结构有效、状态在库阈值内（R7 的 reason 3/5 不出现）且锚点 top-1 的库步落后 ≤ 1 个决策；不要求夹爪阶段一致（R7 reason 2/4 放过）。检索在 R7 腕部 72 维度量里（StageWrist 原机制）。
- 预测与判据 `PREDICTION_R8.md`（2026-10-02 06:06 CDT，早于预拟合/发射）：K1 Spatial-500 两臂 + GR00T L10-500 的 SR ≥ 同批 A − 3 且 IR ≤ A − .007；K2 π0.5 L10-500 变体 SR ≤ A − 5（预测亏）；K3 π0.5 变体 IR ≤ .045、GR00T ≤ .070；过程断言：8 臂调用份额恒 0。

## 发射前证据
- `build_r8 check` 4/4 OK：π0.5 工件是 PaceWrist(enabled, pace 1) over StageFollow(E1, 阶段门, 阈值, lib big, anchor_tail/budget 1/budget_only)，阶段表与跟随表的检索指纹 = 基座、腕部度量 72 维、相机初始 full、cell 一致；GR00T 工件是 R7 冻结 StageFollow E1（指纹一致、路径在 r07_main/fits）。
- 纯缓存 CPU 自测（--blind，4 集）4/4 PASS、0 条 Traceback/P2：π0.5 两臂各播出 12 / 8 个跟随块、规划 92 次腕眼（harness 不走相机通道）；GR00T 两臂 4 / 8 个跟随块。
- 测试 `round8/tools/tests/test_round8.py` 3 通过：参数校验；真实录制查询上计划 = "reason∈{0,2,4} 且落后 ≤ 1" ⇔ 腕眼、腕眼看一眼返回有效结果且度量字段原样恢复；批次规格（对照逐字段等于 R8 A 行、GR00T 变体等于 R8 SF1 行、π0.5 变体旗标与工件、无判官/策略尾旗标、无任务索引、无守卫/升级字段）。
- 同步计划 rc 0：94 文件 22.5 GiB（`/home/weiland/trace_runs/os_closed_loop/r09_fable_r8/plan.log`；含两个 500 库 pkl、四个 A 工件、两个 R7 SF1 工件、R7 wrist/stages、本轮两个 fits）。
- 代码：新文件 `round8/tools/{methods.py,build_r8.py}`、tests、`round8/{PREDICTION_R8,R8_HANDBACK}.md`、`round7/R7_RESULTS.md`；h100 镜像 `/data/oscl_h100/openpi/exp/offline_search/rounds/r09/explore_fable/round8/tools/methods.py` sha **cc35e98a…**（本地一致）；R7 c2_wrist 1629c979… / c1_follow 1d8739b5… 两端一致。

## 命令（我不发射）
```bash
cd /home/weiland/projects/openpi
R=/home/weiland/trace_runs/os_closed_loop/r09_fable_r8; OPS=$PWD/exp/offline_search/closed_loop/ops/h100
P=(taskset -c 22-25 env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src .venv/bin/python)
ARMS="r9f8_pi05_l10_500_A r9f8_pi05_l10_500_wpace_fg r9f8_pi05_spatial_500_A r9f8_pi05_spatial_500_wpace_fg r9f8_groot_l10_500_A r9f8_groot_l10_500_fg r9f8_groot_spatial_500_A r9f8_groot_spatial_500_fg"
WORKER_HOST=timan107 SYNC_PORT=23195 bash "$OPS/sync_assets.sh" --concurrent "$R" $ARMS
WORKER_HOST=timan107 PORTS=23240,23241,23242,23243 WPS=12 MAX_ATTEMPTS=3 POLL_SECONDS=30 OSCL_MANIFEST=$R/manifests/eval100_inits20_29.json bash "$OPS/chain_h100.sh" "$R" $ARMS
"${P[@]}" -m exp.offline_search.rounds.r09.explore_fable.round8.tools.build_r8 score    # SR + owner IR（腕眼 .0646），init 20–29
for c in pi05_l10_500 pi05_spatial_500; do "${P[@]}" -m exp.offline_search.rounds.r09.explore_fable.round2.tools.paired_r2 --ref $R:r9f8_${c}_A $R:r9f8_${c}_wpace_fg; done
for c in groot_l10_500 groot_spatial_500; do "${P[@]}" -m exp.offline_search.rounds.r09.explore_fable.round2.tools.paired_r2 --ref $R:r9f8_${c}_A $R:r9f8_${c}_fg; done
```
注：纯缓存臂没有 MISS，所以"按请求相机"的"MISS 时补全"路径不会触发；paired_r2 的 IR 不分腕眼价，以 build_r8 score 为准。
