# Session handoff

> 多条线共用本文件。§0 为常驻初始化（不动）。**§1–§6 = 离线检索探索线（offline_search）交接，2026-09-26 23:50 CDT 覆写（owner 要求，compact 前）**。step_diag / warm reset 线的交接原文移到附录 A（该线已全部完成、待 owner 裁定提交）。

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



## 1. 现在在哪（2026-09-27 01:50 CDT）—— 离线检索探索线（offline_search），R2 收尾 + R3 闭环进行中

**owner 已睡，协调者全自主推进。目标（/goal 已设）**：
- 把 R2 做完：闭环两组全部跑完，再做分析并提交；
- 接着独立做完一整轮 R3，允许跑闭环；
- 中途不停、不问，暂停点全部跳过，决策记进台账。

**唯一权威**：`logs/offline_search_exploration.log.md`（章程；§8 纪律，§9 owner 裁定，§10 台账）。compact 之后**先读 §8、§9 和 §10 的最后 150 行**，再读本节。

| 阶段 | 状态 |
|---|---|
| R0 / R1 / R2 离线 | ✅ 已提交 |
| R2 闭环 50 集组（16 臂） | ✅ 全部完成（00:51），结果见 §2 |
| R2 分析 | ✅ `rounds/r02/ANALYSIS.md` 已含 50 集组全部结果（提交 85a201c）；**500 集组跑完后要发附录 2**：SendMessage 给 agent `aa44db0c1790670fa`，它的重算脚本在 `tmp/analysis_r02/regen_cl.sh` |
| R2 闭环 500 集组（16 臂） | 🔄 01:45 起由 tmux `oscl_queue` 在跑，预计约 04:30 完成 |
| R3 构思 | ✅ A（fable）、B（fable）、C（codex）三路，选题定稿见 `rounds/r03/SELECTION.md`（H1–H4） |
| R3 编码 | ✅ H1 `AWM3`（含后补的 release_guard）、H2 混合模式插件、H3 `MixedJudge`、H4 `ops/kpi.py` 与 `pilot.sh`（提交 cf2fb8d、8bf2a33） |
| R3 离线全量 | ✅ H1 176 个作业，H3 48 个作业，结果在 `results/r03/` |
| R3 pilot | ✅ 结论见 §2；gr 的 2 个 pilot 臂排在 queue2 开头 |
| R3 全量（500 集）+ MX 混合模式 | ⏳ 由 queue2 在 R2 500 集组之后自动执行 |
| R3 分析与提交 | ⏳ 全部跑完后派 fable 分析 agent，然后提交并向 owner 汇报 |

## 2. 关键结果

**R2 闭环，50 集组**（纯缓存，A 池 500 集）：

| 格子 | CL0 B0 | CL1 前 5 均值 | CL2 AWM | CL3 +恢复 | 纯推理 |
|---|---|---|---|---|---|
| π0.5-sp | .668 | .764 | .800 | .798 | .986 |
| π0.5-l10 | .440 | .428 | .630 | .642 | .844 |
| GR00T-sp | .736 | .852 | .888 | .862 | .940 |
| GR00T-l10 | .468 | .466 | .552 | .606 | .870 |

- 合成层在 spatial 上 +10～+12 pp，在 l10 上为 0；方法层（AWM）+3.6 / +20.2 / +3.6 / +8.6 pp。
- 恢复机制不是通用手段，只有 GR00T-l10 +5.4 pp。
- 历史对照：S3 为 .688 / .456，S6 为 .810 / .516。

**R3 pilot**（100 集，陷阱任务 × init 0–19，与 R2 同集配对比较）：
- **π0.5-sp**：cl2ref .70（与 R2 3/3 持平，重跑噪声约 6%）；借用先验 α.5 为 **.79**（p = .064）；ridge1 .75；α1 .74。
- **两个开关失败**：夹爪承诺（对称）**.10**，因为它挡住了抓取；终止守卫 **.23**，陷阱只是挪到停顿行。
- **π0.5-l10**：cl2ref .59（相对 R2 .52，l10 噪声更大）；α1 .49（任务 4 崩）；α.5 .53，借用先验在 l10 上无益。
- **GR00T-sp**（弱任务 {8,2,3,7,5}）：cl2ref .75，tgp **.60**（任务 3 从 .95 掉到 .30），任何终止行屏蔽都有害。
- **推论**：纯缓存能拿到的只剩 spatial 上的借用先验；终止吸收和死锁要交给混合模式的 MISS。

## 3. 离线要点（陈旧状态 err；四格依次为 π0.5-sp / π0.5-l10 / GR00T-sp / GR00T-l10）

- **三层拆分**：
  1. 合成：B0 .643/.597/.621/.644 → M4 .589/.540/.548/.568；
  2. 方法（50 集库）：AWM .579/.513/.511/.513；
  3. 库（500 集）：AWM .501/.433/.432/.447。
- **R3**：借用先验 α.5 为 .550/.487/.494/.489（标注"借用大库信息"）；ridge1 为 .570/.505/.508/.517。夹爪承诺和终止守卫离线分别 +.006～.031 和 +.028。
- **离线 err 不能给闭环排序**：A、C 和 R2 分析三方都证实了。筛选一律靠闭环 pilot。
- **库体积**：AWM 每条目 580 B，另加每个 suite 约 17 MB 的 PCA；拟合 pickle 在 50 集库上 22–28 MB，在 500 集库上 47–118 MB（MixedJudge 为 26–187 MB）；现役 pkl 431–1103 MB。

## 4. 正在运行的东西（compact 后先核对）

- **tmux `oscl_queue`** 执行 `/home/weiland/.claude/jobs/a607dd74/tmp/next_queue.sh`，日志在同目录的 `next_queue.log`：
  - 步骤：pilot sp → pilot l10 → pilot GR00T（这三步已完成）→ **R2 500 集组的 chain（正在跑）** → 若存在则执行 `next_queue2.sh`。
  - ⛔ 正在执行的 bash 脚本不能原地改。要跳过某个臂，就在对应运行目录写 `state/<arm>.DONE` 加 `.SKIPPED`；`next_queue2.sh` 在被执行之前都可以改，先写 `.tmp` 再 `mv`。
- **queue2**：
  1. gr pilot（sp、l10）；
  2. R3 全量：`r03_full` 的 r3f_{p,g}_{sp_a05, sp_rm1, l10_a05}；
  3. 条件步骤：若 gr 的 pilot 比 a05 高 ≥ 3 pp，追加 a05gr 的全量；
  4. MX l10 核心 5 个臂：b0h70 / awm_h70 / ev_h70 / g / perk3，完整模型 server；
  5. MX sp 核心 5 个臂；
  6. MX h50 共 4 个臂。
- **运行目录**（均在 `/home/weiland/trace_runs/os_closed_loop/` 下）：
  - `r02_g50`：完成；`r02_g500`：在跑；
  - `r03_pilot`、`r03_full`、`r03_mx`：拟合产物、yaml、timan107 端的 yaml 都已就绪。
  - 结构统一为 `runs/<arm>/summary.json`、`runs/chain.log`（EV 行）、`state/`。
- **监控**：
  - Monitor 在 `tail -F` 以上各个 chain.log 和 `next_queue.log`，过滤条件为 `^EV .*(ARM_DONE|ARM_FAILED|SERVER_DIED|GPU_TIGHT|PORT_BUSY|CHAIN_DONE|CHAIN_STOPPED)|Traceback|^QUEUE`，30 分钟到期后需要重挂。
  - cron `44664685` 在每小时第 7/27/47 分做 PROBE。
- **KPI 工具**：
  ```bash
  taskset -c <cpus> .venv/bin/python -m exp.offline_search.closed_loop.ops.kpi \
    --run-root <R> [--run-root <R2>] <arm…> --ref <run:arm> [--pilot | --tasks … --episodes 0-19] \
    --recon AWM3=awm:5 --json … --md …
  ```
- **codex**：companion 的 `status` 在 compact 后按会话过滤、找不到任务。直接读 `/tmp/codex-companion/openpi-*/jobs/<id>.json` 并检查 PID 是否存活。
- **子 agent**：A `a0c7646ef8d80b670`、B `a27181135033e9710`、H1 `afebf8d0cb4cbec0d`、H2 `afc4b1c2ad16e1e8a`、H3 `a064bd459192fc74b`、H4 `ad7ecadd78517868f`、R2 分析 `aa44db0c1790670fa`。

## 5. 下一步（按顺序）

1. **R2 500 集组完成后**（约 04:30）：
   - SendMessage 给 R2 分析 agent 做附录 2：500 集组，库效应按每个 CL 配对；
   - 提交；台账写 R2 小结；task #28 完成。
2. **queue2 各步骤完成时**：用 KPI 工具做配对，结果记台账。
   - R3 全量臂的参照是 R2 同格的 CL2（50 集组）。
   - MX 臂报实际 h、IR（π0.5 为 0.152 + 0.848 × MISS 比例）、SR，以及 MISS 落在失败集中的比例和原因码占比。
3. **全部跑完后**：
   - 派 fable 写 `rounds/r03/ANALYSIS.md`：三层拆分加"控制效应"，两种库规模，SR 与 IR 曲线（点包括纯推理、纯缓存 CL2、MX 各臂）；
   - 提交，向 owner 汇报；task #30 完成。

## 6. 纪律与坑（本线专有，章程 §8/§9 有全文）

- **禁止事项**：
  - ⛔ 视觉必需。
  - ⛔ CPU 38-43,82-87 已让给他线，本线池为 0-37,44-81。闭环 server 用 0-17,44-61，chain 用 34-37,78-81，其余给 agent 和离线任务，一律 `taskset`。
  - ⛔ 不 pkill、不 pgrep -f 自匹配，只按 PID 或 PID 文件 kill。前台不长时间 sleep，等待交给 Monitor；Monitor 只报事件，cron 只做巡检。
- **结论必须带的内容**：三层拆分；两种库规模和体积，并与现役对照；"借用大库信息"要单独标注。
- **每个阶段完成就 commit**：
  - 只加本线路径，被忽略的 json spec 用 `git add -f`。
  - 作者 LinZiyang666，提交信息用英文，不加 AI 署名。不 push。
- **已踩过的坑**：
  - GR00T 需要 `--resize-size 256`（已修）。
  - server 日志要轮转（已修）。
  - 每臂的 chain 失败后要用同一份臂列表重新起链。
  - Monitor 只看最新一次 ARM_START 之后的日志。
  - `xargs` 默认会吃掉引号：并行跑 prefit 脚本时用 `xargs -d '\n' -P N -I{} bash -c '{}'`。
  - 推送 yaml 到 timan107 时不要用 `sync_remote.sh` 重推 `run_arm.sh`，那样会原地改写正在执行的脚本。只推 yaml（先 `tether push` 到 `/tmp/oscl_stage`，再在远端 `cp` 到 `os_cl/cfg/`）。
  - 混合模式的分位数控制器必须给 τ0；l10 上只开守卫就超出了 h=.7 的 MISS 预算，实际 h 约 .62。
- **评测库与工具**：
  - `/dev/shm/offline_search_store`，重启后执行 `stage_shm.sh`。
  - 离线批量用 `harness.batch`，启动脚本在 `tmp/run_r03_h{1,3}.sh`。
  - `tmp/rtable.py` 汇总结果表。
- **本线提交历史**（最新在前）：8bf2a33、85a201c、cf2fb8d、6ce5e60、3dd12f7、411dfd1、34e65b9、80bd7ce、08f7fcb、4fcb316、5bde9de、a0c60a4、21236f5、596b00b、8edc377、897dc8a、cbc7e7a、db5c0bf、e9dcd98、03e6e0e、5ee0058、d975009。

---
## 附录 A：step_diag / warm reset 线交接（2026-09-26 03:20 CDT 版，原文保留，标题降一级）

### 1. 现在在哪（2026-09-26 03:20 CDT，全部完成）

本线：在 RoboCasa365 / LIBERO / MetaWorld 上比较 full、纯减步（plain）、我们的 warm start（精确续跑，只有它叫 warm start）与 warm reset 各变体（cache 起点 / self 起点），GR00T 另有 shoot 消融。记法（owner 定）：T = 起点动作所处 t（0 = 最终动作），N = 实际去噪步数，t = 传给模型的 t，一律 π0.5 记法（1 噪声、0 干净；GR00T 原生 = 1 − t）。self = 同一观测上先做一次 full 推理当起点（不用 cache），每决策 K+N 次前向。

| 线 | 状态 | 结果位置 |
|---|---|---|
| RoboCasa `sdiag_self13`（π0.5 + GR00T self / shoot，step_diag 队列 `sdq`） | ✅ 9/25 23:35 ALL_DONE 312/312，已分析 | 报告 §6.13、§6.14；网页 |
| LIBERO π0.5（step_diag 队列 `sdlq`，spatial + libero_10，9 臂） | ✅ 9/25 03:23 完、已分析 | §6.15 π0.5 部分；网页 |
| MetaWorld MT50 π0.5（新框架，6 臂 × 50 任务 × 20 集，无库） | ✅ 9/26 00:08 完、已分析 | §6.16；网页；数据 `/data/wr_mw/formal/` |
| LIBERO GR00T full / plain_k1 / plain_k2（新框架补跑） | ✅ 9/26 00:18 完，全部准入 | `/data/wr_runs2/`，并入 §6.15 GR00T |
| LIBERO GR00T 26 个 warm 臂（新框架） | ✅ 9/26 03:14 完：39 段全部准入（spatial 13 + libero_10 26，含 01:00 为尾部均衡拆出的 10 个半段），已分析 | `/data/wr_runs/`；§6.15 GR00T 部分；网页 v12 |

- 分析产物：`exp/step_diag/data/analysis/warm_variants_groot_libero_{spatial,10}.{json,md}`、`success_length_groot_libero_{spatial,10}.json`（libero_10 合并 29 个 run dir）。报告：`step_vs_warmstart.md` §6.15（标题改为「π0.5 与 GR00T」）、`success_length.md` 读法 6。
- 机器：h100 / wls / timan107 / timan108 均无本线进程，GPU 显存 ≈0；cron、Monitor、后台等待全部已停。
- **下一步**：等 owner 裁定 §7（提交与清理）。没有在跑或待跑的实验。

### 2. LIBERO GR00T warm 臂运行拓扑（新框架，`exp.warm_reset.run`；已全部停止，留作复现参考）

- 代码钉版 dea5066：driver 树 wls `/data/openpi_wr`；h100 server 树 `/data/openpi_sdlib`；timan worker 树 `/scratch/zixuans8/step_diag/openpi_lib`；timan 上 run dir 在 `/scratch/zixuans8/wr_runs/<run>`（timan `/data` 不可写）。
- 身份与旧 LIBERO 轮一致：A 池 `exp/common/data/db_init/libero/<suite>_apool` idx 0..49、seed 7、replan 5、GR00T K=8（`groot_n15_k8_v1`）、resize 256、namespace `wr_groot_<suite>`；库 `/data/libero_cache/libraries_w13/<suite>/<suite>_w13_S3.pkl`；base yaml `exp/step_diag/config/arms/groot_libero_<suite>/warm_t0.75.yaml`。
- 队列 `/data/wr_runs/queue/queue.json` + `lane.py`（每段新 run dir、跑完 admit、失败重试 ≤3）；LIBERO-10 后段拆成单臂 500 集段，最后 5 个 cache 臂再拆成 init 0–24 / 25–49 半段（queue 字段 `tasks_file`，`tasks/groot_libero_10_i{00_24,25_49}.json`；lane.py 备份 `lane.py.bak_0100`，旧 lane 进程经 `STOP_<lane>` + `relaunch_lane.sh` 换新代码，队列空时自停 server）。
- 15 条跑道（server / driver(wls) / worker）：h100 H1–H4 = 23270–23273 / 23182–23185，H5–H6 = 23260–23261 / 23143–23144，H7–H8 = 23274–23275 / 23186–23187；wls W1–W7 = 23170–23176 / 23140–23142, 23145–23148。worker：timan107（H2, H3, H5, H6, W1–W4；不用 GPU 3）、timan108（H1, H4, H7, H8, W5–W7）。
- 看：`bash /data/wr_runs/wr_status.sh`、`python3 /data/wr_runs/lane.py status`、`tail /data/wr_runs/logs/segments.log`（`SEGMENT … run=<rc> admit=<rc>`）、`grep -E "FAIL|STALL|RETRY|WAIT" /data/wr_runs/logs/lane_events.log`、说明 `/data/wr_runs/NOTES.txt`。停：`touch /data/wr_runs/queue/STOP`（跑完当前段退出）。
- ⚠ 两个精确续跑段（`*_s01_warm`）的 `EVIDENCE_FAIL / PULL_MISMATCH` 是误报：`warm_t*` 按设计不写 server 证据，admit 照常通过。
- full/plain 补跑的另一套：`/data/wr_runs2/`（树 `/data/openpi_wr2` @44c689b、`wr2_*` 会话已全部停）；h100 按 K 起 server（无 cache config），证据在 h100 `/data/wr_evidence2/`。
- MetaWorld：冻结快照 `/data/openpi_mw`（dea5066 + 工作树 diff，`/data/wr_mw/formal/PROVENANCE.txt`），server 在 wls、worker 在 timan108（`/scratch/zixuans8/metaworld_sim` + `openpi_mw`），全部已停。

### 3. 代码与提交状态

- 已推送 origin/Ziyang：`20acb6f` warm reset 一等公民框架、`2f0116c` 实验入口 `exp/warm_reset`、`dea5066` plan log + 迁移研究、`35de051` 框架补全（yaml `miss:` 块 = full/plain 臂，π0.5 按 bundle；GR00T 每 K 端点；`warm_reset.trigger: always` 无库自产 `SELF_ONLY`；钉物体 RC；注册钩子 `exp/<pkg>/warm_reset_env.py`；分析适配器 `exp/warm_reset/analysis.py`）、`fd16d86` MetaWorld 接入、`44c689b` 日志（owner 16:20 豁免 plan/G1、19:57 批准提交）。
- **未提交（等 owner 指示）**：`exp/step_diag/ops/{self13,libero}_queue.py`（共机调度、`rotate_serve_out` 修旧输出误报 SERVER_FAIL、h100 `ram_budget` 185→160 防 OOM）+ 对应测试；`exp/step_diag/evidence.py`（`CHECKPOINT_DIGEST_EQUIVALENTS`）+ 测试；报告 `exp/step_diag/analysis/step_vs_warmstart.md`（§6.13 GR00T、§6.14、§6.15 注、§6.16）与 `success_length.md`；本文件。分析产物在 `exp/step_diag/data/analysis/`（数据目录被 .gitignore 覆盖，要入库需 `git add -f`）。`exp/trace_dual/` 是他线的，勿动。
- 网页构建脚本（`/home/weiland/.claude/jobs/3f6cef91/tmp/web/*.py`）属画图类脚本，**不入库**。

### 4. 结果摘要（报告 / 网页的依据）

- **RoboCasa π0.5**（13 任务，N=2）：full 0.548、plain_k2 0.481、ours 0.277；warmreset 0.708 / self 0.738，resetfinal 0.708 / self 0.735，midfinal 0.545 / self 0.685（+0.14 显著）；成功集调用对 full −16 到 −39 次，self 与 cache 同幅 ⇒ 收益来自 reset 流程。
- **RoboCasa GR00T**（K=4）：self ≈ cache（11 个配置差值均值 +0.003，10 个区间含 0；唯一显著的是 self 更差 T=0/N=1/t=1 −0.042）；无 π0.5 那样的成功集缩短。shoot 全面崩溃（对 full −0.46 到 −0.62，越界越多越差；N=1 shoot 输入与 ours 相同、只是步长大，0.555 → 0.015）⇒ warm reset 的收益来自重置 t。
- **LIBERO π0.5**：plain_k2 ≈ full，warm reset 各变体 ≈ full，ours 显著 −6 到 −7 点，self ≈ cache，长度不变。
- **LIBERO GR00T Spatial**（新框架，29 臂）：全部 0.926–0.948（full 0.944），对 full 配对差 ±2 点内、区间全含 0；成功集决策差 ≤ 0.3。warm_t0.875 与 warm_t0.75 恰都 469/500（待核是否同一路径）。full/plain：Spatial 94.4 / 93.4 / 93.6%，LIBERO-10 88.0 / 84.6 / 84.2%。
- **LIBERO GR00T LIBERO-10**（新框架，29 臂，K=8）：full 0.880；plain_k1 0.846（−0.034 [−0.068, −0.000]）、plain_k2 0.842（−0.038 [−0.068, −0.008]）；ours N=1 0.712（对 full −0.168 [−0.208, −0.128]）、N=2 0.818（−0.062）；缓存 reset 12 配置 0.826–0.868，对同 N plain −0.020 到 +0.022 全含 0（停在 plain 水平，不补回）；self − cache −0.026 到 +0.048（均值 +0.008，仅 1 对显著）；成功集调用差 −0.51 到 +1.10（full 52.5），无缩短。spatial 两个精确续跑臂同为 469/500 已核为巧合（20/20 集结果相反）。
- **MetaWorld π0.5**（50 任务 × 20 集）：full 0.588、plain_k2 0.604、plain_k1 0.537、selfwarmreset 0.571、selfresetfinal 0.544、selfmidfinal 0.482；self 对 plain_k2 三个区间都不含 0，损失在 hard / very hard 组；少步臂成功集都少约 1–1.6 次决策（与 plain 同幅）。RLinf 口径四组平均：full 51.4（RLinf 公布 43.8，K=5）。
- 三 benchmark 合看：reset 式 warm reset 只在减步明显掉点的 RoboCasa 有收益（self 起点同样有效）；LIBERO 上两模型都停在 plain 水平（无收益），MetaWorld 有害；精确续跑（ours）在各处都有害或持平。

### 5. 网页

- https://claude.ai/artifact/3N13AXbWAwZkaUizoo7TLE，源文件 `/home/weiland/projects/openpi_ext/artifacts/warm_reset_robocasa.html`（v12：加了 GR00T LIBERO-Spatial / LIBERO-10 各 N=1、N=2 四个面板 `gl_sp1/gl_sp2/gl_101/gl_102`）；9/26 整理后源文件换了位置，publish 时**必须传 `url=https://claude.ai/artifact/3N13AXbWAwZkaUizoo7TLE`**，否则会生成新链接（不传 icon）。
- 结构：`const DATA` / `const CALLS`（一行 JSON）+ `PANELS`（key、mount、title、budget、note、cmax/cticks、bench、`nTasks`、`cardsLabel/cardsNote`）；`rowsOf` 把 self 臂贴在 `pair` 的 cache 臂下（紧贴、斜纹），无 pair 的 self 单独成行；`family: "shoot"` 前加分组行；calls `mean: null` 显示 "—" / n/a。颜色 token 在 `:root` 与两个 dark 块，含 `shoot1/shootmid/shoot05/plain1`。
- 数据构建：π0.5 self + LIBERO `/home/weiland/.claude/jobs/3f6cef91/tmp/web/build_data.py`；GR00T RC `exp/step_diag/data/analysis/web_groot_self13.json`（分析 agent 生成）；MetaWorld `tmp/web/build_mw.py` → `mw.json`（按难度组卡片）。截图检查 `tmp/web/shot*.py`（`uv run --no-project --python 3.11 --with playwright`，chromium 已装）。
- GR00T LIBERO：`tmp/web/build_groot_libero.py spatial 10` → `groot_libero.json`，`tmp/web/update_page_gl.py` 幂等并入页面（DATA/CALLS、PANELS、lede/bench/注释措辞）；截图 `tmp/web/shot_gl*.py`。
- 网页规矩：英文；图例不提颜色；用 inference calls 不用 env steps；不排名；变化 = 值 − full，按配对检验着色；只有精确续跑叫 warm start。

### 6. 监控与纪律（本会话新增的 owner 规则）

- 监控已全部撤掉（03:15 CDT）：cron `461fbc65` 已删，Monitor 与后台等待已停。
- ⛔ timan 族（107/108）只跑 worker，永不跑 server；server 只放 wls 与 h100。
- ⛔ 不许任何机器干等：某线结束立即把资源重排给剩余线（今晚已做多轮：sdq 结束→h100/wls 加 LIBERO 跑道，MetaWorld 结束→wls 加 W6/W7，full/plain 结束→h100 加 H7/H8）。
- 优先级：MetaWorld π0.5 > GR00T。worker 一律放 timan，本机只放 server（MetaWorld 本地 worker 曾挤爆 4090 显存致 GR00T OOM）。
- h100 GR00T RoboCasa server 常驻会涨到 27–31 GB，7 个会 OOM（dmesg 可证）；已改预算 160。
- 提交只在 owner 当次明确指示时；可分多次提交；英文 message；无 AI 署名。
- 子 agent 常在「等监控通知」时停住不醒：有结果却不动时直接 SendMessage 叫醒；叫回来的 agent 保留上下文。

### 7. 待 owner 裁定 / 杂项

- **home 整理（2026-09-26 完成，`d6ee47e` 已推送）**：本项目在本机 home 的文件只剩主仓 `~/projects/openpi` 与 `~/projects/openpi_ext/{envs,third_party,lines,artifacts,scratch,attic}`（说明 `openpi_ext/README.md`，明细 `MOVES.tsv`，改动前原件 `attic/reorg_rollback_20260926/`）；`~/trace_runs` 按 owner 指定留原地；`~/ckpt_*`、`~/rl_router` 软链已删，引用一律 `/data/ckpt/<名>`、`/data/rl_router`。所有 venv / conda 已改前缀并逐个冒烟（LIBERO 图像 md5 与搬迁前一致；metaworld 17 passed 无 skip；dp_nfe 123 passed；GR00T 岛 10 passed），全量 CPU 测试的失败均为 HEAD 既有。遗留：① run_so_101 的 Mac 端 rsync 目标要改成 `~/projects/openpi_ext/lines/run_so_101/`（其项目记忆里的服务器 IP 192.168.1.150 与本机 192.168.0.200 不符，请核对）；② run_so_101 自己的仓有 15 个路径改写未提交；③ ops 文档里 `~/ops/README.md:86-91`、`HANDOFF.md:65,72-82`、`gpu-fault-2026-08/.../ANALYSIS.md:81`、`setup-scripts/lerobot_venv_setup.sh:4-7` 仍是旧路径（归 ops 会话改）；④ `/data/openpi_*` 冻结树与 `/data/wr_runs/serve_wls.sh` 仍写旧 `~/ckpt_*`，原样复跑需显式 `SD_CKPT=/data/ckpt/...`；⑤ `.claude/settings.local.json:193` 的权限规则指向 `~/projects/dist_experiment_control`，未改。

- 是否提交 §3 的未提交改动（报告 §6.13 GR00T / §6.14 / §6.15 全部 / §6.16、success_length.md、队列修复、evidence 等价表、本文件）；分析产物 JSON/MD 在被 .gitignore 覆盖的数据目录，入库需 `git add -f`。
- 清理：本机 `pull_g13/`（GR00T 分析拉数据解包副本，h100 部分约 3.5 GB 重复）、timan107/108 `/tmp/sdg13/*.tgz`（约 67 MB）、`/data/wr_*` 各 run 目录与 `/data/openpi_mw`、`/data/openpi_wr*` 冻结树的保留期限。
- MetaWorld 运行 agent 自报：19:37 曾对两个未核实的 PID 发 `kill -TERM`（均不存在，未误伤）。
- GR00T K=4（RoboCasa）真模型逐位对等仍缺（本机无 RC 观测 npz）。

---


---

## 附录 B：trace_dual 数据资产

本线的原始数据（8 组 × 500 集 trace）见 `exp/trace_dual/analysis/{results,data_quality}.md`；离线评测库由它抽取（章程 §3）。旧的 trace_dual 交接已被本线 §1–§6 取代。
