# Session handoff

> 多条线共用本文件。§0 为常驻初始化（不动）。**§1 = x₀-head × 标签过滤数据实验线终态（实验完成 2026-09-19 19:41；等 owner 裁 commit）**；§2 = 减步基线线终态（已收工）。旧 §7（key builder × LDA）已删——其状态见记忆 `project_fusion_weight_lda_ablation`。

## 0. 初始化方式（不变）

owner 的常驻指令，逐字有效：

> 「开始进行实验，我离开了，期间由你独断专行，不要问我任何问题，注意监控体系定位，
> corn 负责定时巡检，monitor 负责条件触发，不要职责混淆，停止条件是完全做完实验，不做完不停」

**三条必须保持的纪律**：

1. ⛔ **不得用 git commit / push 同步代码和数据**。要同步任何代码或数据一律走 **tether**。
   git 只在里程碑收口、且 owner 当次明确指示时才动。
2. **server 只在 server 节点跑，worker 只在 timan 族跑**（见 §2）。
3. **读 `experiment-lifecycle` skill 但不挂载**：只取 tether 用法（`dist_experiment_control/docs/usage.md`）
   与设备清单（`docs/devices.md`），**不要执行它的 §0 初始化**，不要索要 agentchat 账号 / token / 房间。

### 开工步骤（照做，不要问 owner）

1. **读文档**，按这个顺序，都读完再动手：
   - `logs/cache_prune_run_handoff.md`（整份，运行入口的唯一权威）
   - 本文件 §1–§5
   - `/home/weiland/projects/dist_experiment_control/docs/devices.md`（拓扑与机器约束）
   - `/home/weiland/projects/dist_experiment_control/docs/usage.md`（tether 命令）
   - 需要时再查 `logs/cache_prune_plan.log.md`（算法与统计定义）
2. **探设备**：`tether node ls -a`，确认 h100 / weilandserver / timan107 / timan108 四台
   都 ONLINE；逐台查 GPU 占用、RAM、磁盘余量、残留 tmux 与进程。⚠ timan108 只有 3 张卡
   （见 §2），别按 4 张排。
3. **对齐代码**：四台的仓要和本地同一 commit。**走 tether push，不走 git push**；
   已存在的文件要 `--force`；`/scratch` 不在 timan 的 allow_roots，经 `/tmp` 中转。
   推完逐文件 `sha256sum` 对账，**比对内容不要带路径**（`cat A B | sha256sum`，
   `sha256sum A B` 会把路径算进去，我为此误判过一次）。
4. **建任务表**（TaskCreate），把 §1 的准备项与三相位拆成条目，边做边更新状态。
5. **搭监控**（§4）：cron 定时一行巡检 + Monitor 条件触发。**先搭好再放量**。
6. **先 smoke 再放量**：任何新拓扑都先跑一个最小 cell 验证到底（起 1 个 server、少量 worker、
   跑通一集并核对产物），再按 §2 的规格扩到目标规模。上一条线每次跳过这步都出事。

**其它长期约束**：

- commit message 全英文；**绝不加 `Co-Authored-By: Claude`** 或任何 AI 署名（作者恒为
  `LinZiyang666 <3177267975@qq.com>`）。未经 owner 当次指示不得 `git add`。
- ⛔ **画图脚本一律不许 commit**（`build_*figure*` / `plot_*` / `render_*` / `edit_*figure*` /
  `*_editor.html`）。`exp/rit_pareto/build_figure.py` 与 `edit_figure.py` 属于另一个 session，**碰都不要碰**。
- ⛔ 不要 `rm -rf`（带 -f 的递归强删）；`rm` / `rm -r` 不带 -f 可以用；删除前先 `wc -l` 核对规模。
- ⛔ **共享机上不要 `pkill -f`** 裸模式：它会匹配到发起命令的 shell 自己。用字符类
  `[w]orker_entry`，且**脚本正文任何地方（含注释、echo）都不能出现裸的匹配串**。
- 无人值守期间**禁起 `run_in_background` 后台任务**（触发审批弹窗阻塞会话），用 Monitor。
- 报时刻用本机本地时间（America/Chicago）。
- **巡检就只巡检**：贴 PROBE 行，不要顺手做额外分析（owner 明确要求过）。

## 1. x₀-head × 标签过滤数据实验线 — 终态（实验完成 2026-09-19 19:41 CDT；只保留结论、资产与待裁项）

> 权威记录：`logs/x0_multimodal_plan.log.md`（§10.1 准备、§10.2 正式矩阵与提速/再平衡时间线、§10.3 Phase 3）；终报 `exp/dp_nfe/analysis/x0_multimodal.md`；记忆 `project_x0_multimodal_line`。

### 1.1 结论（终报 §1/§3）

64 格两机矩阵全部训练+评测（466 记录 0 invalid）。**pusht：H2 supported / H1-data not supported（S_x0 = −0.008 [−0.041, +0.026]，δ 内等价）/ I not supported；square_mh：三项 inconclusive（区间宽，点估计同向）；blockpush：12 格 screening < 0.5 无判决；kitchen 不可用。** 描述性：explore 10/10 任务 ε 头 DDIM-1 归零、x₀ 头一步 ≈ 100 步；x₀ 头锚点普遍低 4–8 pp，多模态/mh 数据上更低（square image M −24 pp）；诊断：ε 一步 x̂₀ 爆炸（89–96% 坐标越界），x₀ 头条件离散度 ≈ 0（确定性回归器）。

### 1.2 owner 裁定（仍有效）

只用 h100 / weilandserver；不 commit/push 除非 owner 当次说（71cc10b 之后**没有**新授权）；画图脚本 `plot_x0.py` 不入库；`data/` 槽不入库；报时 America/Chicago。9-19 owner 放行 wls 杀进程/重切（"我授权你"）与再平衡（"充分平衡"）；两机启用 CUDA MPS（h100 21:49 9-18、wls 13:35 9-19）——实验已结束，daemon 仍在（`echo quit | nvidia-cuda-mps-control` 可关，不影响别的 session 已跑的进程）。

### 1.3 资产

- 本机 `exp/dp_nfe/data/x0_multimodal/`（gitignore）：`wls/`、`h100/`（cells、results_trailing、diagnostics、runs 元数据）、`merged/`（合并矩阵 + `merge_report.json`）、`decisions.json`、`diagnostics_table.{json,md}`、`figures/*.png`。
- 两机：`$X0_DATA/{cells,runs,results_trailing,diagnostics,subsets}`（ckpt lowdim 1 G / image 4 G 每格，共 ≈100 G；wls `/data/dp/x0_multimodal`、h100 `/data/dp_h100/x0_multimodal`）；wls 上 4 个 M ε s43/s44 的 5k 步半成品 run 目录留置（无 summary，merge 按 h100 归属）。
- 工作树未 commit 改动（71cc10b 之后）：§1.6 旧清单 + Phase 3 新增 `analysis/{merge_x0_hosts,diagnostics_table}.py`、`ops/{pull_x0_results,x0_dispersion}.sh`、`tests/dp_nfe/test_x0_{merge,diagnostics_table}.py`、`analysis/x0_multimodal.md`、`logs/README.md` 行、plan §10。本机 `uv run pytest tests/dp_nfe` 114 passed / 2 skipped（终报前跑）。
- job tmp `~/.claude/jobs/9267c51a/tmp/`：`probe_x0.sh`、`watch_phase2.sh`、`lane_rates.sh`、`mps_restart_{h100,wls}.sh`、`h100_takeover.sh`、`rebalance_wls.sh`、`mps_daemon_*.sh`（提速/再平衡脚本，均已执行）。

### 1.4 待 owner 裁

1. commit/push：本线 Phase 2/3 全部改动（一次结构化 commit，英文 message，无 AI 署名）+ 减步基线线遗留未跟踪文件（`exp/{xwam_nfe,cosmos_nfe}`、`exp/dp_nfe/{eval_dp_steps,aggregate_dp}.py`、ladder 脚本、`logs/{xwam_dp_nfe,cosmos_nfe}_plan.log.md`、`logs/cache_transfer/*`、`logs/nfe_baseline_rc365_plan.log.md`）。
2. 两机 MPS daemon 是否保留；两机 ≈100 G checkpoint 是否清理（`runs/*/checkpoints/{final,latest}.ckpt`；identity/manifest/log 已拉回本机）。
3. Jayanth 回信草稿（要点见 plan/对话）。
4. 下一轮若继续：CUDA graph / torch.compile 训练提速（本轮明确不做，留作代码定稿后的 G1/G2 项）；square image 预算不足（40k 锚 0.52 vs 官方 0.72）；blockpush/kitchen 需要不同的数据或预算才能进正式族。

### 1.5 坑（保留，新 session 仍会踩）

- tmux 与 conda `LD_LIBRARY_PATH`：`env -u LD_LIBRARY_PATH tmux`。
- pgrep/pkill 自匹配：模式与字面量不能同在一条命令里；脚本落盘再跑。
- h100 `tether exec` 双执行、`HOME=/home/exouser`、根盘 7.4 G。
- tether pull 只收单文件（目录拒 `not_a_regular_file`）→ 远端 tar 再拉（`pull_x0_results.sh`）。
- 本机自动模式分类器会拦"杀 wls 进程"的脚本（Interfere With Workloads / Modify Shared Resources / Auto-Mode Bypass）；owner 明示授权后放行过一次；只"加"不"杀"的操作（起 daemon、加 lane）一直放行。
- launch-bound 训练（DP UNet，单进程 CPU 100%、GPU 功耗 <50%）：提速 = 多 lane + CUDA MPS（h100 16.7→76 upd/s、4090 21→46），不改代码/identity；时间片下加 lane 到 4 条就饱和，评测器会把训练拖慢一半，MPS 下不会。
- 两机 zarr 子集字节不同（selection 相同）→ 结果记录绑训练主机的 cell yaml sha，合并必须按归属主机取 yaml（`merge_x0_hosts.py`）。
- DataLoader 迭代器创建消耗 RNG；官方配置 `${now:…}`；kitchen npy 掩码非前缀；square image 动作两次转换差 ≤0.04。

## 2. 减步基线线 — 终态（已收工 2026-09-17 07:45；只保留结论与指针）

- 记录：`logs/nfe_baseline_rc365_plan.log.md`、`logs/cosmos_nfe_plan.log.md`、`logs/xwam_dp_nfe_plan.log.md`（X-WAM RoboTwin 2.0 k=10 0.908 / k=1 0.864；DP 官方阶梯 leading 网格）；数据 `exp/{nfe_baseline,cosmos_nfe,xwam_nfe,dp_nfe}/data/`；记忆 `project_nfe_baseline_live_run`。
- 结论：flow / x₀-EDM 头（π0.5、GR00T、Cosmos、X-WAM）减步近无损；RoboCasa365 上 GR00T k=1 −11 pp、π0.5 k=1 −17 pp；DP ε-DDPM 头 1 步归零（leading 与 trailing 网格都如此，见 §1.4 P0）。全部产物未 commit，等 owner。
