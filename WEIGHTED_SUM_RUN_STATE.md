# weighted_sum 实验状态 —— ✅ 全部完成（2026-05-26）

> 原 compact 恢复手册（~230 行）在收尾时从磁盘丢失（未 commit 不可恢复，原因不明）。
> 实验已闭环结束，本文件为简洁完成记录。完整结果见 `exp/weighted_sum/RESULTS.md`。

## 交付清单（全部完成）

1. **基线**：136 配置 × 100ep = 13600（jupyter 单机 H200，可信）。SR 62.1%，最优 72%。
2. **3 轮调优**（用户决定做完 3 轮就停）：两个 keybuilder（cp1_max_pool + cp1_spatial_pool_16）都细化。SR 天花板 **~74% 全程收敛**（r1=r2=r3）。
3. **完整分析**：`RESULTS.md` §1–§7（一份连贯分析，非分阶段）+ `data/phase2/all_results.csv`（418 配置）。
4. **2 张图**（`exp/weighted_sum/figures/`，已发聊天室）：fig1 三元热力图（各 keybuilder 权重单纯形 SR）+ fig2 keybuilder 箱线图。不含 a100。
5. **top10 跨 GPU 对比**（`RESULTS.md §7` + `data/phase2/top10_compare.csv`）：top10 在 a100(A100, 3rep) vs jupyter(H200)，a100 平均低 **6.6pp**、最大 18pp、8/10 偏低 → 干净验证跨架构 SR 不可比（印证固定单 GPU 的必要性）。

## 核心结论

- **SR 天花板 ~74%**（always_hit 纯检索，该库密度下）。
- **keybuilder**：`spatial_16`(74%) ≈ `max_pool`(73%) 噪声内平手，均 ≫ `spatial_64`(67%) ≈ `mean`(67%)。保留细空间(4×4)或逐维 max > 过粗(2×2)/全局均值。
- **最优权重区**：低 vision_0(@0.06–0.31) + 中高 vision_1(@0.44–0.50)·robot_state(@0.44–0.50)；三模态 > 双模态 > 单模态（iso_vision_0 最差 29%）。
- **归一化**：三字段 zscore(tanh) 最优（norm2 对照未超）。
- **跨 GPU**：同配置 H200 vs A100 系统性差 ~7pp（§7）。

## 拓扑（最终）

- **jupyter-ziyang10** = H200 server（weiland.top:14000），跑了全部 jupyter 实验。
- **timan107** = driver/client。**本机** /home/weiland/projects/openpi = 留底（config/round_*、data/phase2/*.csv/json、figures/）。
- **a100** = 新机（149.165.152.105, A100-40GB, vla-cache），旧 a100(149.165.151.106) 已退役替换（见 devices.md §2.3）。仅 top10 跨 GPU 对比用过，已停 server。
- broker = pc732（weiland.top）。

## spatial keybuilder 命名 —— 决定不改（2026-05-26 用户定）

扫描发现 `spatial_pool_64` 跨多实验（weighted_sum + trajectory/phase1/verdict）+ 历史 log + pkl，全库 rename 影响面大且触及其它已完成实验/历史记录。**用户决定不改，保持现状。** 代码里 `_4` 已是规范名（output_tokens），`_64` 作向后兼容别名指向 `_4`（key_builder.py `CP1SpatialPool64=CP1SpatialPool4`、config.py registry 标注 legacy alias），功能正常无歧义风险，只是 pkl 文件名 + 各 config 沿用 `_64`。**此事了结，不再 pending。**

## 未提交的代码改动（工作树，未 git commit）

journal.py(livelock fix) / worker.py(py3.8 Callable + 删 START/DONE debug) / agent.py(conda_env) / scheduler.py(eval_concurrency 默认 1→2) / run_phase2.py(--eval-concurrency) / summarize.py(SR 口径) / emit_yamls.py(vdims 全字段) / refine_round.py(新) / build_all_results.py(新) / plot_results.py(新) / RESULTS.md(新) / docs/experiments/conductor_tutorial.md(§12 参数全集) / docs/dist_experiment_control devices.md(a100 换机) / .gitignore(config yaml 排除)。
