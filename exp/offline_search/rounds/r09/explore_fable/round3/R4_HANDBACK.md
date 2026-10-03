# Round 4（fable）交接 — 六个新格子的泛化批次 `r09_fable_r4`（已备好，**未发射**）

## 给 owner 的白话结论
- 这一轮没有新结果，只是把 LIBERO-10 小库上赢的那两套做法原封不动搬到另外六个格子（两个模型 × LIBERO-10 大库、Spatial 小库、Spatial 大库），
  每格配两条对照（纯缓存、只开"没进展就叫策略"的守卫），共 21 条臂，每条跑 100 集（10 个任务 × 第 20–29 号起始状态）。
- 做法本身不看任务是谁：守卫只看"机器人有没有在动"，动作修正器只看当前图像/状态/缓存动作，升级规则只看落后了多少步。
  修正器的头是在第 0–19 号起始状态上拟合的，评测用第 20–29 号，两边不重叠；第 30–49 号从头到尾没打开。
- 预测已先写死在 `PREDICTION_R4.md`（2026-10-02 03:25 CDT，早于所有预拟合）。我押的是：LIBERO-10 大库和 Spatial 小库上，守卫+修正比只开守卫再高 2–4 个百分点，
  调用开销还略降；两个 Spatial 大库是天花板格（纯缓存已 .96/.98），目标只是"别变差"。
- 发射前的全部检查都过了：9 条堆叠臂的工件逐项断言通过（守卫原参数未被改动、单一判官家族、修正器半力度、同一个库），18 个 CPU 自测全部 PASS、
  0 条上轮那种 "P2 does not support" 报错，同步计划通过（135 个文件，23.7 GiB）。
- 一处坦白：在看一个日志格式时屏幕滚过了一条第 33 号起始状态的单步记录（r8abl_onlynp_p_l10_500，任务 0），没有用于任何数字；此后所有解析都在读取时先按起始编号过滤。

## 批次内容（run root `/home/weiland/trace_runs/os_closed_loop/r09_fable_r4`，manifest `manifests/eval100_inits20_29.json` = 任务 0–9 × init 20–29）
| 格子 | NpCorr（守卫+半力度修正，max_calls 0） | NpCorrEsc（+升级 lag 12 / deadline 80） | 缓存对照（R8 A 行原样） | 只开守卫对照（r8abl 行原样） |
|---|---|---|---|---|
| π0.5 L10-500 | r9f4_pi05_l10_500_np_corr05 (NpGraspStack3) | r9f4_pi05_l10_500_np_corr05_esc (NpGraspEsc3) | r9f4_pi05_l10_500_A ← r8_pi05_l10_500_A | r9f4_pi05_l10_500_onlynp ← r8abl_onlynp_p_l10_500 |
| π0.5 Spatial-50 | r9f4_pi05_spatial_50_np_corr05 | r9f4_pi05_spatial_50_np_corr05_esc | r9f4_pi05_spatial_50_A ← r8_pi05_spatial_50_A | r9f4_pi05_spatial_50_onlynp ← r8abl_onlynp_p_sp_50 |
| π0.5 Spatial-500 | r9f4_pi05_spatial_500_np_corr05 | r9f4_pi05_spatial_500_np_corr05_esc | r9f4_pi05_spatial_500_A ← r8_pi05_spatial_500_A | r9f4_pi05_spatial_500_onlynp ← r8abl_onlynp_p_sp_500 |
| GR00T L10-500 | r9f4_groot_l10_500_np_corr05 (NpGraspStackGroot3) | — | r9f4_groot_l10_500_A ← r8_groot_l10_500_A | r9f4_groot_l10_500_onlynp ← r8abl_onlynp_g_l10_500 |
| GR00T Spatial-50 | r9f4_groot_spatial_50_np_corr05 | — | r9f4_groot_spatial_50_A ← r8_groot_spatial_50_A | r9f4_groot_spatial_50_onlynp ← r8abl_onlynp_g_sp_50 |
| GR00T Spatial-500 | r9f4_groot_spatial_500_np_corr05 | — | r9f4_groot_spatial_500_A ← r8_groot_spatial_500_A | r9f4_groot_spatial_500_onlynp ← r8abl_onlynp_g_sp_500 |
- 对照行复制了源行的 method / kwargs / plugin_args（含源冻结工件路径）/ client_overrides；来源与工件 sha 在 `provenance.json`。
- 堆叠臂 kwargs：`onlynp_fit=<该格 r8abl_onlynp_*.pkl>, corrected_fit=<R4>/fits/r9f4_<cell>_corr05pt.pkl, empty_aperture .001(π0.5)/.0009(GR00T), closed_sign +1/−1, hold_decisions 2, burst 2, max_calls 0, force_trigger_at []`（Esc 另加 lag_threshold 12, deadline 80）；plugin_args 与 r3c 相同（--os-blind --os-policy-tail --os-policy-tail-blocks 1 --os-judge guard_only）。
- 修正底座 `r9f4_<cell>_corr05pt`：CorrectedCache，该格 A 的 base kwargs（500 库 lib big / kref 8；50 库 lib current / kref 5）+ 该格 A 工件 + `round2/out/corrector/head_<cell>_motion_pertask.npz`（本轮新拟合 5 个头，init 0–19；π0.5 Spatial-50 的头沿用 round 2），blend .5，不改夹爪。
- 全部 kwargs：`r4_kwargs.json`；臂规格：`arms_in.json`；发射后的行：`arms.json`（21 行，config/ 42 个 yaml）。

## 预测（摘自 PREDICTION_R4.md，SR / IR）
π0.5 L10-500：NpCorr .88/.150，NpCorrEsc .89/.160 ｜ π0.5 Sp-50：.90/.130，.92/.140 ｜ π0.5 Sp-500：.97/.090，.98/.095 ｜
GR00T L10-500：.92/.165 ｜ GR00T Sp-50：.90/.130 ｜ GR00T Sp-500：.95/.110。对照应复现下表 ±5 pp。
判据 H1/H2/H3 与失败解读预案见该文件；跑完后 `build_r4 score` 自动按判据打分（只读 init 20–29）。

参考（init 20–29，n=100）：
| 格子 | 纯缓存 A | 只开守卫 SR @ IR（look / call 份额） |
|---|---|---|
| π0.5 L10-500 | .86 | .86 @ .155 (.504 / .093) |
| π0.5 Spatial-50 | .79 | .87 @ .134 (.511 / .067) |
| π0.5 Spatial-500 | .96 | .99 @ .090 (.514 / .013) |
| GR00T L10-500 | .87 | .92 @ .170 (.504 / .112) |
| GR00T Spatial-50 | .91 | .86 @ .135 (.510 / .069) |
| GR00T Spatial-500 | .98 | .93 @ .111 (.510 / .042) |
（`/home/weiland/trace_runs/os_closed_loop/r09_fable_r4/ref_onlynp_inits20_29.csv`；服务器决策日志逐条按 init 过滤后汇总）

## 发射前证据
- 工件断言（`build_r4 check`，9/9 OK）：judge.burst == 源 == 0、gm_burst 2、max_calls 0、单一判官家族、base 为 CorrectedCacheJ 且 blend .5、与守卫同一个库、
  disabled_guards = stuck/terminal/overtime、cell 一致、夹爪符号正确、Esc 臂 lag 12 / deadline 80（非 Esc 臂无升级字段）。
- CPU 自测（`/home/weiland/trace_runs/os_closed_loop/r09_fable_r4/selftest/*.log`，每臂 4 集，真实判官链 + policy tail + guard_only）：18/18 PASS，0 条 "P2 does not support"。
  生产 kwargs：空抓触发（reason 93）恒为 0，守卫 miss 2（π0.5 L10-500）/ 4（GR00T L10-500）/ 0（Spatial 四格）；
  强制触发变体（max_calls 2, force_trigger_at [2,6]）：每次 14 条 reason-93 决策、miss 16–18 —— 触发 + 恢复爆发在新格子上也走通。
- 同步计划（`control plan`，`/home/weiland/trace_runs/os_closed_loop/r09_fable_r4/plan.log`）：rc 0，135 个文件 23.7 GiB（含 6 个 R8 冻结 A 工件、6 个 r8abl 守卫工件、15 个本轮 fits、4 个库 pkl、store）。
- 测试：`round3/tools/tests/` 18 通过（test_round3 6、test_round3c 3、test_round3_integration 3、**新增 test_build_r4 6**：21 臂形状、对照逐字段等于冻结行、
  堆叠臂用本格守卫 + 半力度修正、无任务索引字段、manifest = 评测切分、打分器只用 init 20–29）。
- 代码：`round3/tools/methods.py` **未改动**（sha fa881a57…，h100 镜像 `/data/oscl_h100/openpi/exp/offline_search/rounds/r09/explore_fable/round3/tools/methods.py` 已核对相同；
  round2 methods 86774b83…、opus EscalateOnlyNP 2ede2b5f… 两端一致）。新文件只有 `round3/tools/build_r4.py`、`round3/tools/tests/test_build_r4.py`（h100 不需要）。

## 命令（我不发射；按 r3c 脚本模式）
```bash
cd /home/weiland/projects/openpi
R=/home/weiland/trace_runs/os_closed_loop/r09_fable_r4; OPS=$PWD/exp/offline_search/closed_loop/ops/h100
P=(taskset -c 22-25 env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src .venv/bin/python)
ARMS_P="r9f4_pi05_l10_500_np_corr05 r9f4_pi05_l10_500_np_corr05_esc r9f4_pi05_l10_500_A r9f4_pi05_l10_500_onlynp r9f4_pi05_spatial_50_np_corr05 r9f4_pi05_spatial_50_np_corr05_esc r9f4_pi05_spatial_50_A r9f4_pi05_spatial_50_onlynp r9f4_pi05_spatial_500_np_corr05 r9f4_pi05_spatial_500_np_corr05_esc r9f4_pi05_spatial_500_A r9f4_pi05_spatial_500_onlynp"
ARMS_G="r9f4_groot_l10_500_np_corr05 r9f4_groot_l10_500_A r9f4_groot_l10_500_onlynp r9f4_groot_spatial_50_np_corr05 r9f4_groot_spatial_50_A r9f4_groot_spatial_50_onlynp r9f4_groot_spatial_500_np_corr05 r9f4_groot_spatial_500_A r9f4_groot_spatial_500_onlynp"
# h100 fleet (ports 23240-23243, worker timan107, sync port 23195); pi0.5 first, then GR00T (fewer model reloads)
WORKER_HOST=timan107 SYNC_PORT=23195 bash "$OPS/sync_assets.sh" --concurrent "$R" $ARMS_P $ARMS_G
WORKER_HOST=timan107 PORTS=23240,23241,23242,23243 WPS=12 MAX_ATTEMPTS=3 POLL_SECONDS=30 OSCL_MANIFEST=$R/manifests/eval100_inits20_29.json \
  bash "$OPS/chain_h100.sh" "$R" $ARMS_P $ARMS_G
# abort on any 'P2 does not support' line in server logs (none expected; selftests clean)
# scoring after the run (pre-registered H1/H2/H3, inits 20-29 only) and paired analysis per cell
"${P[@]}" -m exp.offline_search.rounds.r09.explore_fable.round3.tools.build_r4 score
for c in pi05_l10_500 pi05_spatial_50 pi05_spatial_500; do
  "${P[@]}" -m exp.offline_search.rounds.r09.explore_fable.round2.tools.paired_r2 --ref $R:r9f4_${c}_onlynp $R:r9f4_${c}_np_corr05 $R:r9f4_${c}_np_corr05_esc $R:r9f4_${c}_A; done
for c in groot_l10_500 groot_spatial_50 groot_spatial_500; do
  "${P[@]}" -m exp.offline_search.rounds.r09.explore_fable.round2.tools.paired_r2 --ref $R:r9f4_${c}_onlynp $R:r9f4_${c}_np_corr05 $R:r9f4_${c}_A; done
```
GR00T 臂带 `resize_size 256`（随源行），π0.5 判官臂 `replan_steps 5`，π0.5 缓存对照随 R8 行带 `resize_size 224`。
