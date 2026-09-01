# Session Handoff — Dispatch（论文已转向 RIT 风险索引阈值；三目标夜跑链进行中；2026-08-30 20:30）

> **一句话状态**：论文定位已按 owner 决定转向——**RIT（Risk-Indexed Threshold，风险索引阈值；s-only 校准分位切）= 阈值规则族帕累托前沿的风险索引，GST（Grid-Searched Threshold，网格搜索阈值）= 同族的网格搜索索引**（SV/CRD/H-CRD 全部降为消融/附录素材，CRD 图已删）。夜跑三目标链：**① l10 s-only+gate（sysgate，OOM 后 20:20 续跑，34.9%，预计 ~23:10）→ ② spatial s-only 密扫（已全备、dry-validate 过，①完自动接）→ ③ spatial s-only+gate（已全备，②完接）**。冻结裁决 `stop_before_C` 与其资产不动。
> **术语（owner 2026-08-31 裁定，全库统一）**：GST = Grid-Searched Threshold（网格搜索阈值；旧 threshold / thr / tgrid 族）；RIT = Risk-Indexed Threshold（风险索引阈值；s-only / s0 校准阶梯，(s,v) 版 SV 为其消融）。
> **规则（新增两条加粗）**：**未经 owner 明示绝不 `git add`/暂存任何东西（今日被训诫，一切产出留工作区）**；**不随手写记忆文件（owner 已删我写的）**。其余照旧：commit/push 只按明示、作者 LinZiyang666 无 AI 署名；handoff 只按要求写；定向测试；不碰 pruned 测试集/fresh C；codex 改码逐 diff 核验；杀进程 ps→kill pid；中文回复英文注释。

## 0. 权威文档（今日新增/大改）

| 文档 | 内容 |
|---|---|
| `logs/dispatch_surface_sonly_pivot_report.md` | **转向报告**：§0 新定位、§1 全天讨论证据链（batch D→GTP 对照→参数诊断→per-task 平手→三线区间打平）、§2 完整表述（K=1 让步 / K≥2 解路径命题 / K=3 决胜实验设计 / 措辞纪律）、§3 系统实验、§6 **相关工作初查**（selective prediction；Pareto Testing 2210.07913；MultiRisk 2512.24587；级联影子价格 2605.06350——后两篇投稿前必须精读）与贡献声明层级 |
| `docs/iclr/latex/sonly_note.tex` | 数学文档（仿 dispatch_note 风格）：两种索引、K=1 等价、单乘子解路径、per-task/跨库迁移、不主张范围、四条实验推论；**pdflatex 已过**（sonly_note.pdf） |
| `logs/dispatch_surface_rev2_amendment_result.md` | 交审文档新增 §16–§16.8：batch D 结果/golden/GTP 对照/行为分解/离线预演/同密度区间对照。§15 = codex Round 2 **APPROVED WITH REVIEWER FIXES**（§15.6 未尽：commit + commit 后重建 export record provenance） |
| `logs/dispatch_surface_rev2_confirmation_plan.log.md` | §10 之后按时间追加全天记录（暂停/重启/batch D/清理/OOM 事故等），append-only |
| 已删（owner 目标变更） | `docs/iclr/ICLR_PAPER_BLOCKING_TODO.md`（原冻结 SHA 文档，freeze_record.verify 不受影响）+ 三个 `.old` 规划稿；README 索引与 gate 横幅已清。**保留待表态**：paper_rethink_discussion / actioncache_* / dispatch_defense_plan / latex/paper_outline / experiment_list |

## 1. 论文转向要点（给 codex 立新 G1 的材料已齐）

- 主张三层：(a) 问题域首创（VLA 动作缓存的风险校准多档调度）；(b) 等风险闭式解路径 vs "搜索+检验"（Pareto Testing 一族）；(c) matched-density 前沿贴合 + 调参预算 + 负结果实证学。K=1 与 selective prediction 等价——主动让步。禁语：最优/支配/CMDP/formal guarantee。
- ActionCache 防守评估（详见对话/交审）：三支柱（backbone 前 key 地板 14% vs 45–53%；arXiv-only 豁免；风险索引 vs 他们逐任务手调）；**仍需**：Arm1 CP2-AC 对照（~3000 ep）、pruned-500 确认（方法冻结后 ~2500 ep）、K=3（~4–7k ep）。录用估计 40–55%。截稿 9/16 或 9/25（须再核）。

## 2. l10 数据资产（A′ 开发集，全部 post-hoc 探索）

- **24 臂 dense summary**：`exp/dispatch_surface/data/sgrid/libero_10/sgrid_summary_all.json`（18 臂 sha `78eb6491…` + 加密 6 臂 sha `c7b84be9…` 合并；raw 在 `…/sgrid/libero_10/raw{,2}/`）。关键数：区间内 SV 高 thr +2.0…+4.2 pt、s0 打平；`s0_p86` 0.763@70.7%、`s0_p775` 0.840@83.5%、`s0_p75` **0.870@94.6%**、`sv_p75` 0.840@81.8%；GST（thr）前沿止于 84.5%/0.797；anchor 0.847/67.52ms；s0/sv 在 89–95% 段越过 anchor。
- batch D（H-CRD 消融素材）：`data/crd/libero_10/crd_batchD_summary.json` sha `71008e0c…` + golden 0 违例 + `analysis/crd_offline_screen.py`（校准误差<0.5ms，旋钮余地仅 1–3%）。
- 图：`analysis/figures/libero_10/` 仅 16 文件（dense 系列 + 诊断图；旧版/CRD 版已删）。**画法**：实线=逐点非支配前沿（`_pareto_staircase`），点线=两臂混合包络；`--sgrid-summary` 时不再生成非 dense 旧图（latest-only 已写入 plot main）。报告页 https://claude.ai/code/artifact/5458a825-e32a-45b8-8674-c9289c9aa31e （图1 dense/图1b 24臂/图2 dense/图3–5）。

## 3. 夜跑链操作手册（compact 后最重要）

**① l10 sysgate（在跑）**：15 个 s0 分位 × production gate（θ=0.9928/j3/probe3/L6），矩阵 `/tmp/dsp_shared/config/precheck_libero_10_sysgate/`（24 臂，run 用 `--arms` 15 s0）。20:18 timan107 **系统 OOM**（机队连跑 ~9h 驱动锁页内存耗尽，GTP 已知模式）杀 worker→全体 keepalive 1011→runner 退出；server 无恙（握手 0.04s 验证过）。20:20:26 同 journal 续跑（1525 accepted 无损；污染日志轮换为 `.oom_2020`）。Monitor `bl3yk24jf` + cron `7fca9f2a`（都跨 compact 存活），TOTAL 4500，标记 `SYSGATE_DONE`。**再 OOM → 降 worker 数续跑**。
**DONE 后**：CronDelete → `bash $CLAUDE_JOB_DIR/tmp/pull_sysgate.sh`（→ `data/sysgate/libero_10/raw/`）→ `sgrid_sweep summarize --arm-matrix data/sgrid/libero_10/inputs/precheck_libero_10_sysgate/arm_matrix_sgrid.json --arms <15 s0 臂> --journal/per-step/launch-manifest <raw> --split-manifest data/libero_10/init_pools/split_manifest.json --out data/sysgate/libero_10/sysgate_summary.json` → 系统图（s0+gate 层叠加 24 臂 dense 图，需给 plot 加一层或单独脚本）→ 页面/文档 → **接跑 ②**。
**② spatial 密扫**：`tmux new -s srv0 -d /tmp/dsp_spsgrid_wrap.sh`（先确认无残留、srv0 空）；health `/tmp/dsp_spsgrid_health.sh` TOTAL 4200 标记 `SPSGRID_DONE`；输出 `/tmp/dsp_precheck/libero_spatial_sgrid/`；挂同款 Monitor+cron。DONE 后 pull（仿 pull_sgrid.sh 造 pull_spsgrid.sh：libero_10_sgrid→libero_spatial_sgrid、DST=data/sgrid/libero_spatial/raw）→ summarize（matrix 在 `data/sgrid/libero_spatial/inputs/precheck_libero_spatial_sgrid/`，`--task-suite` 由 matrix 校验、`--arms` 14 s0、split manifest 本地路径 `exp/dispatch_surface/data/aprime_rev1/discipline/libero_spatial_primary/split_manifest.json`）→ spatial 三线图（GST 基线 = GTP spatial 数据 / Rev 1 spatial phase0）→ **接跑 ③**。
**③ spatial sysgate**：`/tmp/dsp_spsysgate_wrap.sh`，TOTAL 4200，`SPSYSGATE_DONE`，θ=0.97174 已在 yaml。流程同 ②。

## 4. spatial 资产（今日新建，全部三机核验）

Rev 1 包 `/tmp/dsp_shared/libero_spatial/rev1_discipline/MANIFEST.json` sha `4f9f79b2…`；lib.pkl `b3f61dc5…`（425MB）；gate θ **0.9717439413070679**（j3/probe3/L6，与 l10 同构、θ 不同）；模板权重 vision_0@6/vision_1@50/robot_state@43（l10 是 56/25/18）；table `/tmp/dsp_shared/libero_spatial/inputs/dispatch_table_fresh.jsonl` sha `9448c115…`（今日从本地推）；新 export 12 分位 {50,55,60,65,70,75,775,85,875,90,925,96} record `352c85bc…` + 旧 exploratory s0{p80,p95} sv{p95,p975}；cfg `precheck_libero_spatial_{sgrid,sysgate}` 各 16 臂（14 s0 + sv 载 contract `dsp_sv_p95`），tgz `0edb3555…` 48 文件三机一致；**两批 dry-validate 均过（14 臂，EXIT=0）**。timan107 侧：split manifest = 仓库相对 `exp/dispatch_surface/data/init_pools/split_manifest.json`、A′ 池同目录 `test_aprime`（rollup `89099b50…`）；quota 同 l10（test=30×10 task）；spatial anchor SR 0.9567/67.52ms；policy fingerprint 与 l10 相同 → **server 不用动**。

## 5. 代码改动（全部未暂存；`sgrid_sweep.py` 已同步两远端并 sha 核验）

- `exp/dispatch_surface/sgrid_sweep.py`：`summarize --arms`（子集）、`crd_params()`、**`emit --gate-layer secondary`**（注入 production gate，θ 自动取包内 rev1 matrix `gate_theta`；新协议 `dispatch_surface_rev2_sysgate_dev`；run/summarize 双协议，gated 走 `LAYER_SECONDARY` 校验）。
- `analysis/plot_budget_amendment.py`：逐点非支配前沿实线 + hull 点线、`--crd-summary`（现已不用）、latest-only（有 `--sgrid-summary` 不再出非 dense 旧图）。
- `analysis/crd_offline_screen.py`（离线参数预演器）。测试：`tests/dispatch_surface/test_sgrid_summarize_subset.py`、`test_sgrid_gate_layer.py`（3 例）等定向全过。
- Git：约 20 文件为 compact 前 owner 指示暂存的旧批次；今日一切改动/新文件/删除均未暂存（含 docs/iclr 删除）。

## 6. 拓扑/运行时（不变项速查）

server：weilandserver tmux `srv0`，`ziyanglin.com:23150` 公网 1:1 直连（proxy→23151–54，4 replica，13:59 起，H-CRD 版代码，pi05_libero，两 suite 通用）。client：timan107 48 worker；**每条命令 `export PATH=/usr/local/bin:/usr/bin:/bin`（runner 加 `/scratch/zixuans8/dsp_bin`）**，HOME 分别 /home/weiland、/home/zixuans8；tether push/pull `--force`、父目录先建、exec 静默 10min 超时→本地轮询；远端克隆独立（tar+push /tmp+sha -c）；`MUJOCO_GL=egl`；吞吐 ~18–19 ep/min；srv0.log 的 InvalidMessage/EOFError = 健康脚本裸探测噪声。

## 7. 坑（新增今日）

- **wrap 的 DONE 标记会在 runner 异常退出时误写**：任何续跑前先 `mv` 轮换日志（sgrid/sysgate 均已踩过）。
- **timan107 OOM 模式**：机队连跑多小时后驱动锁页内存耗尽 → keepalive 1011 连锁；恢复 = 确认 server 健康（真实握手探测）→ 轮换日志 → 原 journal 续跑；复发则降 worker。
- emit 需 ≥1 SV 臂载 contract；同一 journal 只能配同一 matrix sha（补臂 = 大矩阵 + `--arms` 分批）；summarize 子集必须 `--arms` 否则报 incomplete。
- plot 的 merge_sgrid 拒绝重名臂；两 summary 合并用 jq/python 合 `arms` 后传单文件（`sgrid_summary_all.json` 即此法）。
- H-CRD 代码/数据/审查记录保留（消融+审查链），仅图与页面已清；不再向图中传 `--crd-summary`。

## 8. 待 owner / codex

codex 新 G1（转向提案 = 转向报告 + sonly_note + 24 臂证据）；§15.6 未尽（commit reviewer patch、commit 后重建 export record provenance——现 record 的 git_commit 字段两端各记无关 HEAD，已查明）；pruned-500 确认与 Arm1 CP2 对照与 K=3（论文三缺口）；GST+gate 同链 baseline 臂（§16.5 待办）；docs/iclr 剩余文档去留；commit/push 时机。
