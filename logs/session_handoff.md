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



## 1. 现在在哪（2026-09-27 14:40 CDT）—— 离线检索探索线（offline_search），R4 闭环阶段

**目标（/goal 已设）**：前进到 R4 结束，其后再做 R5。
- ⛔ owner 规定**只用 codex agent**，不起 Claude 子 agent，不用 Workflow。
- owner 的要求：思路不受限；时间不是问题；要仔细研究。
- **唯一权威**：`logs/offline_search_exploration.log.md`（章程）。§8 纪律，§9 owner 裁定到第 9 条，§10 台账含 R4 全部条目。**compact 之后先读 §8、§9 和 §10 的最后 150 行，再读本节。**

| 阶段 | 状态 |
|---|---|
| R0–R3 | ✅ `rounds/r02/ANALYSIS.md`、`rounds/r03/ANALYSIS.md` |
| R4 构思 | ✅ 4 份 codex 报告：`rounds/r04/ideation_A/REPORT.md`（看一眼，做几步）、`ideation_B/REPORT.md`（降低单次成本）、`ideation_C/REPORT.md` 和 `REPORT_2.md`（时间偏移检索、分层 pilot、随机化识别调用价值） |
| R4 选题 | ✅ `rounds/r04/SELECTION.md`（中文）、`CODING_BRIEF.md`（英文：文件归属、方法与插件接口、日志字段、成本口径） |
| R4 编码 | ✅ K1–K4 均已交回并提交（6e1cab5），各自的 HANDBACK.md 在 `rounds/r04/k{1_blind,2_serving,3_cost,4_eval}/` |
| R4 第一批闭环（`r04_frontier`） | 🔄 正在跑，见 §4 |
| R4 第二到四批 | ⏳ 第三批已生成并预拟合，见 §5 |
| R4 分析与提交 | ⏳ 派 1 个 codex 分析 agent，然后提交，R4 结束 |

## 2. 关键结果
- **R2 纯缓存 AWM**：50 集库 .800 / .630 / .888 / .552；500 集库 .954 / .768 / .966 / .706（依次为 π0.5-sp / π0.5-l10 / GR00T-sp / GR00T-l10）。纯推理为 .986 / .844 / .940 / .870。
- **R3 混合模式（π0.5，IR 用 owner 口径 .152 + .848m）**：
  - spatial：V7 + 守卫 .980 @ .43。
  - l10：周期 k3 .832 @ .43，k5 .792 @ .32；500 集库只用守卫（g500）**.864 @ .238**。
- **R4 第一批（500 集库，π0.5-l10）**：
  - 守卫 noprog 4 为 .808 @ .214，落在两点连线之下，作废。
  - 周期 8 为 .850 @ .245，低于 g500。
  - g500 仍是这一段的最优点。
- **K3 实测**：
  - dummy_cached 让 stage 1 从 65.8 降到 45.0 ms；wrist_only 让 HIT 的 stage 1 降到 23.9 ms，MISS 时再补 21.6 ms。两者 MISS 动作都与原来逐位一致。
  - stage 2 打包**作废**：bf16 下动作差 0.014。
  - eager serving 的实测成本在 `closed_loop/ops/cost_table.json`：π0.5 三段占比 .142 / .072 / .786，stage 3 每步 36.4 ms。它和 owner 口径是**两套不同的成本基准**，报告时各自单列，主口径仍是 owner 的。
- **K2 实测**：在真实 GPU server 上，盲走决策不跑 stage 1，下发给 client 的动作逐位一致（π0.5 与 GR00T 各 24/24）。真实 GPU 上的 MISS 没有测过。
- **B 的发现**：π0.5 纯推理 K2 与 K10 持平（.996 / .848），见 step_diag 线 §6.15。

## 3. 需要 owner 的事
暂无。R4 / R5 的裁定在 §9 第 9 条：
- 500 集库算可部署，但也要有 50 集库的实验；
- 便宜 key 可用；
- 不在 weilandserver 起 worker；
- 不做系统测量；
- "看一眼，做几步"已放宽。

## 4. 正在运行的东西（compact 后先核对）
- **第一批 chain**：
  - tmux `oscl_r4f`，运行目录 `/home/weiland/trace_runs/os_closed_loop/r04_frontier`。
  - 2 个完整模型 server（端口 23150、23151），每个 32 个 worker；server CPU 0-17,44-61，chain CPU 34-37,78-81。
  - 进度：已完成 `r4_p_l10_g500_np4`（.808）和 `r4_p_l10_per8_500`（.850）；正在跑 `r4_p_l10_per12_500`。
  - 之后依次：`r4_p_l10_g50_np4`、`r4_p_l10_per6_50`、`r4_p_sp_g500`、`r4_p_sp_per12_500`。
  - 4 个纯推理种子臂已写 DONE 加 DEFERRED 推迟。**以后改用 K4 的 seeded `pure_inference` 臂重跑**（`k4_eval/arms_frontier.json` 里有 seeds 1001 / 2001），不要删掉这些 DONE 来重跑老臂。
  - 预计 15:40 左右跑完。
- **Monitor**：
  - 一个合并的无损 watcher：`bash /home/weiland/.claude/jobs/a607dd74/tmp/r4_watch.sh`。它读 `tmp/r4_watch.jobs`（codex job 列表）和 `tmp/r4_watch.runs`（运行目录列表），已报过的事件记在 `tmp/r4_watch.seen`，重挂时既不漏也不重。
  - Monitor 最长 30 分钟，**到期静默重挂即可**（owner 不希望看到反复的"已重挂"消息）。新开运行目录时，把路径追加进 `r4_watch.runs`。
- **codex job**：R4 的 7 个（A、B、C、C 第二遍、K1–K4）全部已结束，目前没有在跑的 codex。
- **显存**：他项目的训练时有时无，时而占 15–20 GB。我方完整模型 server 每个约 9–10 GB。开臂前 chain 会检查 NEED_MB × 端口数（完整模型时 NEED_MB 为 9000）。

## 5. 下一步（按顺序）
1. **盲走闭环 smoke**：用一个单独的运行目录（例如 `r04_bsmoke`），不要用 `r04_blind`，否则 legacy 子集会写 DONE，导致后面的全量被跳过。
   - 臂：`r4b3_p_l10_500_ph2g`（盲走 + 守卫，完整模型）和 `r4b3_p_l10_500_ph2c`（纯缓存盲走，STAGE1_ONLY）。
   - 参数：`OSCL_EPISODES=0 OSCL_TASKS=0,1`，端口 23160，每 server 2 个 worker，SERVER_CPUS 18-25,62-69。
   - 验收：vision 占比明显小于 1，blind 行的 `s1_ms` 为 null，SR 正常，collect 有 cost_ledger。
2. **第三批全量**：运行目录 `/home/weiland/trace_runs/os_closed_loop/r04_blind`，15 臂已生成并预拟合，yaml **还没推到 timan107**。
   - 推送：只推 yaml，逐个 `tether push` 到 `/tmp/oscl_stage`，再在远端 `cp` 到 `/scratch/zixuans8/openpi_trace/os_cl/cfg/`。
   - 等第一批 chain 结束后再起，tmux 名如 `oscl_r4b`，2–3 个 server 按显存定。
   - 臂与对照关系：
     - l10 500 集库：`b0g`（B=0 适配器对照；与 g500 .864 比）、`ph2g`、`ph1g`、`tail1ug`（不门控的剩余块）、`ph2k8`（与 per8 .850 比）、`clk1g`、`ph2c`（纯缓存，与 CL2-500 .768 比）。
     - l10 50 集库：`b0g`、`ph2g`、`ph2k5`（与 perk5 .792 比）、`ph2c`（与 CL2 .630 比）。
     - sp：500 集 `ph2c`（与 .954 比）、500 集 `tail1uc`、50 集 `ph2g`（与 g .888 比）。
     - baseline：`r4b3_p_l10_50_inferL10`（纯推理，L=10）。
   - 列表以 `r04_blind/arms_in.json` 为准。
3. **第二批**（K3 规格，运行目录 `r04_cost`，已预拟合，还要 emit 和推 yaml）：
   - MISS K2：g500、g50、perk5、sp 只用守卫；
   - K2 纯推理 baseline；
   - K4 的 `arms_frontier.json`：seeded 纯推理，以及 L=10 的 K10 / K2 baseline。
4. **第四批**：从 K3 的 `arms_r4.json` 取 dummy_cached / wrist_only × 是否盲走 × 两种库规模，盲走预算按第三批的赢家定。**不用** K1 第四批里带 `--os-pack-prefix` 的行。另加 control_step_library 的 G 与 GS，l10 两种规模。视时间再加 C 第二遍的随机化 CALL / CACHE。
5. **统计与分析**：
   - KPI：`taskset -c <cpus> .venv/bin/python -m exp.offline_search.closed_loop.ops.kpi --run-root <R> ... --ref <run:arm> --json/--md`，新臂带 `cost_ledger`。
   - 派 1 个 codex 分析 agent 写 `rounds/r04/ANALYSIS.md`（brief 仿照 `rounds/r03/ANALYSIS_BRIEF.md`）：SR 对 IR 前沿（owner 口径为主，eager 口径单列），四层拆分外加成本实现层，两种库规模，体积与现役对照。
   - 然后提交，R4 结束。

## 6. 纪律与坑（本线专有，章程 §8/§9 有全文）
- **禁止事项**：
  - ⛔ 只用 codex，不用 Claude 子 agent / Workflow。
  - ⛔ CPU 38-43,82-87 属于他线，本线池为 0-37,44-81。
  - ⛔ 不 pkill、不 pgrep -f 自匹配，只按 PID kill。
  - ⛔ 前台不长时间 sleep；等待交给 Monitor 或 until 循环。
- **codex 调用**：
  - 命令：`node /home/weiland/.claude/plugins/cache/openai-codex/codex/1.0.6/scripts/codex-companion.mjs task --background --write [--full-access] "<prompt>"`。
  - 不带 `--model` 时用默认的 gpt-6-astra xhigh。
  - 状态在 `~/.claude/plugins/data/codex-openai-codex/state/openpi-50fd553c5e274099/{state.json,jobs/}`。
  - 沙箱里看不到 GPU 和 `/dev/shm`，需要 GPU 就加 `--full-access`。
  - ⚠ `task --help` 会被当成 prompt 起一个会话。
  - 只能 `--resume-last`，续跑就另起一个新 job。
  - prompt 结构参考 `rounds/r04/prompts/`（XML 块）。
- **闭环运维**：
  - 推 yaml 不要用 `sync_remote.sh`。推远端脚本用"先写 .new 再 mv"的原子方式。
  - `xargs` 要加 `-d '\n'`。
  - 正在执行的队列脚本不能原地改；要跳过臂，写 DONE 加 SKIPPED / DEFERRED。
  - legacy 子集（`OSCL_EPISODES`）会写 legacy DONE；K4 的清单子集写的是专用 marker，不会误跳过全量。
  - `tether pull` 偶尔返回 75，重跑 collect 即可。
  - collect 里旧的 `ir_measured` 字段受负载延迟影响（会出现 1.40 这种值），不用它；新臂看 `cost_ledger`。
- **结论必须带的内容**：拆分为合成 / 方法 / 库 / 控制四层，外加成本实现层；50 集与 500 集两种库规模；体积与现役对照（π0.5 431 / 1103 MB，GR00T 429 / 1068 MB）；"借用大库信息"单独标注。
- **提交**：每个阶段完成就 commit；只加本线路径（被忽略的 json spec 用 `git add -f`）；作者 LinZiyang666；提交信息用英文，不加 AI 署名，不 push。
- **本线最近的提交**：6e1cab5、dc5e3cb、c0f621d、cd77717、3ede7fc、72dffef、dde14f8、97c6343。

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
