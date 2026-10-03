# Round 6（fable）交接 — 跟随为何在 GR00T 上亏 + π0.5 合并"看一眼"臂 `r09_fable_r6`（已备好，**未发射**）

## 给 owner 的白话结论
- 两份结果写好了：round3/R4_RESULTS.md（六格泛化：三赢两平一小输，升级在 LIBERO-10 大库误触发亏 7 个点；预注册 H1/H2 没过、H3 过）、round5/R5_RESULTS.md（π0.5 三种腕眼与门控跟随都不掉点，节奏腕眼和门控跟随各省 .025 / .032；GR00T 跟随亏 9–15；H1/H3/H4 过、H2 没过）。
- GR00T 为什么亏（round6/DATA_ANALYSIS_R6.md）：跟随块本身不差，亏在**兜底被推迟** —— 跟随让看一眼变少、看到的状态更贴库，"没进展"守卫晚 5 个决策才第一次叫策略；GR00T 堆叠只有这一道兜底，等不了；
  π0.5 有"落后就一直叫策略"的升级，输掉的集里兜底只晚半个决策。只开状态阈值的版本另有直接伤害：44 % 的跟随块挨着夹爪事件、8 % 里夹爪翻转。
- 新臂：π0.5 "节奏腕眼 + 门控跟随"合并成一条（我自己的 LookCostEsc 类，只是把两个开关同时打开；不碰 opus 的守卫-调用门），加上原堆叠作同批对照。预测 .89 @ .135（对照 .88 @ .183），
  判据：不掉 3 个点以上、省 ≥ .035。发射前检查全过。

## 臂（run root `/home/weiland/trace_runs/os_closed_loop/r09_fable_r6`，manifest 任务 0–9 × init 20–29）
| 臂 | 类 / 工件 | 做法 | 插件旗标 | 预测 SR / IR |
|---|---|---|---|---|
| r9f6_pi05_l10_50_esc | NpGraspEsc3，r3c 工件原件 | 守卫+半力度修正+升级 | 同 r3c | .88 / .183 |
| r9f6_pi05_l10_50_esc_wpace_fg | LookCostEsc wrist_gate=pace(pace_lag 1) + follow_blocks=1(阶段门+阈值) | 落后 ≤ 1 且链/阈值通过 → 腕眼；10 步处守门通过 → 播后继块跳过看一眼 | +--os-request-cameras --os-tokens off | **.89 / .135** |
- 其余 kwargs 与 r3c 一致（onlynp_fit r8abl_onlynp_p_l10_50、corrected_fit r9f2_pi05_l10_50_corr05pt、lag 12 / deadline 80、max_calls 0）；R7 冻结的 stages/wrist 工件；无新类（round5 methods.py 不变，h100 镜像 sha e159ac5d… 已核对）。
- 预测（PREDICTION_R6.md，2026-10-02 04:52 CDT，早于发射）：决策构成 双目 .19 / 腕眼 .23 / 调用 .105 / 跟随 .17；H1 SR ≥ 对照 − 3；H2 IR ≤ 对照 − .035；H3 跟随后那次看一眼的腕眼比例低于普通看一眼（≈ 40 % vs 56 %）。

## 发射前证据
- `build_r6 check` 1/1 OK（r3c 断言 + 阶段表指纹 + 腕部度量 72 维 + 门/跟随开关）；CPU 自测 2/2 PASS、0 条 P2（4 集里播出 20 个跟随块、规划 72 次腕眼；harness 无相机通道）；
  同步计划 rc 0：34 文件 11.1 GiB（`/home/weiland/trace_runs/os_closed_loop/r09_fable_r6/plan.log`）；测试：round6 spec 1 + round5 5 全过（builder 只加了注册文件名参数，默认不变）。
- 代码：新文件 `round6/tools/{build_r6.py,follow_analysis.py,follow_offline.py}`、tests、`round6/{PREDICTION_R6,DATA_ANALYSIS_R6,R6_HANDBACK}.md`；`round5/tools/build_r5.py` 三个函数加了 `reg_name` 参数（默认值不变）。

## 命令（我不发射）
```bash
cd /home/weiland/projects/openpi
R=/home/weiland/trace_runs/os_closed_loop/r09_fable_r6; OPS=$PWD/exp/offline_search/closed_loop/ops/h100
P=(taskset -c 22-25 env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src .venv/bin/python)
WORKER_HOST=timan107 SYNC_PORT=23195 bash "$OPS/sync_assets.sh" --concurrent "$R" r9f6_pi05_l10_50_esc r9f6_pi05_l10_50_esc_wpace_fg
WORKER_HOST=timan107 PORTS=23240,23241,23242,23243 WPS=12 MAX_ATTEMPTS=3 POLL_SECONDS=30 OSCL_MANIFEST=$R/manifests/eval100_inits20_29.json \
  bash "$OPS/chain_h100.sh" "$R" r9f6_pi05_l10_50_esc r9f6_pi05_l10_50_esc_wpace_fg
"${P[@]}" -m exp.offline_search.rounds.r09.explore_fable.round6.tools.build_r6 score      # SR + owner IR（腕眼 .0646），init 20–29
"${P[@]}" -m exp.offline_search.rounds.r09.explore_fable.round2.tools.paired_r2 --ref $R:r9f6_pi05_l10_50_esc $R:r9f6_pi05_l10_50_esc_wpace_fg
```
