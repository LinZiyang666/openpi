# Round 7（fable）交接 — GR00T "不推迟兜底"的门控跟随 + π0.5 Spatial-50 合并"看一眼"臂，批次 `r09_fable_r7`（已备好，**未发射**）

## 给 owner 的白话结论
- round 6 的结果已写成 round6/R6_RESULTS.md：合并臂 .880 @ .160 对 .840 @ .196，省 18 % 不掉点；预注册三条里两条过，第三条（跟随后腕眼更少）方向反了，
  原因和 GR00T 的问题是同一个现象：跟随块之后检索"看起来进度更快"。
- 设计 GR00T 变体前先查了一件事：GR00T 对照里，每次"盲块之后的看一眼"有 **23.5 %** 的概率要叫策略兜底，而且用锚点时刻的任何信号（落后量、离库远近、夹爪开合）都预测不出来是哪一次。
  这意味着 GR00T 每跳过一次看一眼就丢掉约五分之一的兜底机会，没有门能提前把它挑出来；跟随省下的看一眼开销（最多 ≈ .016）跟任何把兜底拉回来的后检花的调用开销差不多。
  所以本轮的 GR00T 臂是**机制验证**（把兜底拉回来，成功率能否回到对照），不是省钱臂；我在预测里把这点写明了。
- GR00T 变体（新类 LookCostGrootPace）：跟随只在阶段门+状态阈值通过、锚点落后 ≤ 1 个决策、无进展记忆为空时放行；跟随块之后那次看一眼若落后量比锚点时增加 ≥ 2 个决策（15 步只前进 ≤ 1 个库步）就立即叫策略。
  预测 .86 @ .189（对照 .87 @ .194）；判据：第一次叫策略的推迟从 +5.2 个决策降到 ≤ +2、成功率不比对照低 3 个点、开销不更贵。
- π0.5 Spatial-50 合并臂（round 5 的类原样，换 Spatial 的 R7 阶段表与腕部度量）：预测 .95 @ .072（对照 .96 @ .116，省 ≈ 38 %：Spatial 上调用本来少，看一眼占大头）。
- 发射前检查全过；不碰 opus 的门类。

## 臂（run root `/home/weiland/trace_runs/os_closed_loop/r09_fable_r7`，任务 0–9 × init 20–29）
| 臂 | 类 / 工件 | 做法 | 旗标 | 预测 SR / IR |
|---|---|---|---|---|
| r9f7_groot_l10_50_np | NpGraspStackGroot3，r3c 原件 | 同批对照 | 同 r3c | .87 / .194 |
| r9f7_groot_l10_50_np_fgp | **LookCostGrootPace**（round7/tools/methods.py）pace_lag 1，lag_jump 2，follow 1 阶段门 | 跟随放行加节奏与空记忆条件；跟随后看一眼做节奏一致性检查（reason 94，flag bit 12） | 同 r3c | **.86 / .189** |
| r9f7_pi05_sp_50_esc | NpGraspEsc3，r4 原件 | 同批对照（Spatial-50 守卫+修正+升级） | 同 r4 | .95 / .116 |
| r9f7_pi05_sp_50_esc_wpace_fg | LookCostEsc（round 5 原样）wrist pace + follow 1 | 节奏腕眼 + 门控跟随 | +--os-request-cameras --os-tokens off | **.95 / .072** |
- kwargs 与各自对照一致（max_calls 0；GR00T 阶段表预拟合时从库拟合；Spatial 用 R7 冻结 stages_pi05_sp_50 / wrist_pi05_sp_50，指纹与腕部度量 kwargs 已核对一致）。
- 预测与判据 `PREDICTION_R7.md`（2026-10-02 05:39 CDT，早于预拟合/发射）：G1 兜底推迟 ≤ +2 决策、G2 SR ≥ 对照 − 3、G3 IR ≤ 对照；S1 Spatial 臂 SR ≥ 对照 − 3 且 IR ≤ 对照 − .030。

## 发射前证据
- `build_r7 check` 2/2 OK（r3c 断言 + 阶段表指纹 + 门/跟随开关 + GR00T 无腕眼）；CPU 自测 4/4 PASS、0 条 P2（GR00T fgp 4 集：20 个跟随块、10 次跟随后检查、0 次强制调用、0 次撤销；Spatial 臂：20 个跟随块、76 次腕眼规划）。
- 测试：`round7/tools/tests/test_round7.py` 3 通过 —— 参数校验；**真实录制 GR00T 查询**上：撤销只在"落后 > 1 或记忆非空"时发生、跟随后一次查询置 os_lc7_post 并清 pending、强制调用当且仅当落后增加 ≥ 2（reason 94、flag bit）；批次规格（对照指向 r3c/r4 冻结工件、相机旗标只在 π0.5 臂、无任务索引、无 opus 门字段）。round5/round6 测试 6 仍全过。
- 同步计划 rc 0：51 文件 9.2 GiB（`/home/weiland/trace_runs/os_closed_loop/r09_fable_r7/plan.log`）。
- 代码：新文件 `round7/tools/{methods.py,build_r7.py}`、tests、`round7/{PREDICTION_R7,R7_HANDBACK}.md`、`round6/R6_RESULTS.md`；h100 镜像 `/data/oscl_h100/openpi/exp/offline_search/rounds/r09/explore_fable/round7/tools/methods.py` sha **8fe5c484…**（本地一致）；round5 methods e159ac5d…、round3 fa881a57… 两端未变。

## 命令（我不发射）
```bash
cd /home/weiland/projects/openpi
R=/home/weiland/trace_runs/os_closed_loop/r09_fable_r7; OPS=$PWD/exp/offline_search/closed_loop/ops/h100
P=(taskset -c 22-25 env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src .venv/bin/python)
WORKER_HOST=timan107 SYNC_PORT=23195 bash "$OPS/sync_assets.sh" --concurrent "$R" r9f7_groot_l10_50_np r9f7_groot_l10_50_np_fgp r9f7_pi05_sp_50_esc r9f7_pi05_sp_50_esc_wpace_fg
WORKER_HOST=timan107 PORTS=23240,23241,23242,23243 WPS=12 MAX_ATTEMPTS=3 POLL_SECONDS=30 OSCL_MANIFEST=$R/manifests/eval100_inits20_29.json \
  bash "$OPS/chain_h100.sh" "$R" r9f7_groot_l10_50_np r9f7_groot_l10_50_np_fgp r9f7_pi05_sp_50_esc r9f7_pi05_sp_50_esc_wpace_fg
"${P[@]}" -m exp.offline_search.rounds.r09.explore_fable.round7.tools.build_r7 score   # SR + owner IR（腕眼 .0646），init 20–29
"${P[@]}" -m exp.offline_search.rounds.r09.explore_fable.round2.tools.paired_r2 --ref $R:r9f7_groot_l10_50_np $R:r9f7_groot_l10_50_np_fgp
"${P[@]}" -m exp.offline_search.rounds.r09.explore_fable.round2.tools.paired_r2 --ref $R:r9f7_pi05_sp_50_esc $R:r9f7_pi05_sp_50_esc_wpace_fg
# G1 (first-call delay) after the run: python -m exp.offline_search.rounds.r09.explore_fable.round6.tools.follow_analysis with R5 -> R7 and ARMS -> the r7 pair (same parser), or the one-off snippet in R7_HANDBACK
```
