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



## 1. 现在在哪（2026-09-27 22:25 CDT）—— 离线检索探索线（offline_search），R4 闭环收尾 + R5 编码基本完成、闭环排队

**目标（/goal，owner 已离开）**：独自推进 R4 和 R5，中间没有阻塞（章程 §9 第 13 条，跳过所有暂停点）。
- ⛔ 所有 agent 只用 codex；codex 只做开发、调试、离线分析，**不看守长实验**（§9 第 14 条）。闭环的运行与看守一律由协调者自己做。
- **续聊指定的 codex**：job 记录 `~/.claude/plugins/data/codex-openai-codex/state/openpi-50fd553c5e274099/jobs/<id>.json` 里有 `threadId`。该 job 这一轮结束后，用 `codex exec resume <threadId> "<增量指令>" -c sandbox_mode="danger-full-access" -c approval_policy="never" -o <out.md>` 续跑，进程要自己用 `kill -0` 看。正在执行的一轮不能插话。追加工作优先续聊原 agent（Q4 已这样做过）。
- **唯一权威**：`logs/offline_search_exploration.log.md`。§9 的 owner 裁定到第 14 条；§10 台账含 R4 / R5 全部条目。**compact 之后先读 §9 和 §10 最后 300 行，再读本节。**
- 任务列表：#37 R4 分析、#40 K8 复测、#43 R5 总项、#45 R4 剩余闭环、#46 R5 闭环、#47 Q5、#48 R5 分析。

| 阶段 | 状态 |
|---|---|
| R4 编码 K1–K10 | ✅ 已提交（6e1cab5、e8980cd） |
| R4 闭环 | 🔄 只剩 GR00T 纯缓存盲走 8 臂（l10 4 臂在 23160 跑，sp 4 臂在 23150 队列）和 K10 种子 `r4f_p_{l10,sp}_inf_s2001`（l10 在 23150 跑） |
| R4 分析与提交 | ⏳ `rounds/r04/ANALYSIS_BRIEF.md` 初稿要按 §2 更新，然后派 codex 分析 agent |
| R5 构思 A–D、选题、CODING_BRIEF、FINDINGS | ✅（`rounds/r05/`） |
| R5 编码 | Q1 ✅、Q2 ✅（插件已装）、Q3 ✅（结论：一律调用，不开臂）、Q4 ✅（grow250 + 示范规模曲线）、Q6 ✅；**Q5 在做**（GPU 检索影子模式，task-mukmfjxz-i7jxc2，PID 2938164，正在跑安装前回归） |
| R5 闭环 | ⏳ 45 臂排在两条线的 R4 尾巴之后（见 §4）；各 smoke 已通过 |

## 2. 关键结果（owner 口径：vision .152、MISS +.848、盲走 / 块尾 0；每个 MISS 按完整推理计）
- **噪声基准**：stock g500（l10 500 集只用守卫）三次运行 .864 / .850 / .832，**均值 .849 ± .016**，约等于 L=5 纯推理（l10 .848 / .850 / .844）。l10 上 2–3 pp 的差别单跑分不清。
- **执行段长度**：l10 纯推理 L=10 为 **.904 / .900**（两次，IR .5），配对 52 对 24，p=.0018。spatial 上 L=10 为 .986，无差别；spatial 纯推理 L=5 为 .986–.992。
- **盲走（看一眼，走几步）**：
  - K1 的稠密卡住守卫去掉了视觉确认，误触发约 2.8 倍。K7 视觉确认守卫在 B=0 时与 stock 逐位一致。
  - **K7 anchor_tail**：l10 500 **.880 @ .203**（K1 版 .878 @ .223），l10 50 **.806 @ .242**（50 集新前沿），sp 500 **.982 @ .128**；sp 50 在 R5 队列。
  - **K7 phase B=2**：l10 500 **.862 @ .178**；l10 50 为 .700（稀疏库上失败）；sp 500 为 .968 @ .140。B=1 被 B=2 支配。
  - K1 守卫、B=1 时：tail .878 > kernel_clock .868 > phase .838。两时钟方案被支配（.822 / .776）。sp 50 的 K1 phase B=2 为 .884 @ .204。
- **纯缓存（不调用策略）盲走**：phase B=2 在 l10 500 为 .778 @ .091、l10 50 为 .620 @ .098、sp 500 为 .954 @ .083，SR 与 CL2 持平，IR 少 36–46%。**★ 纯缓存 anchor_tail sp 500：.982 @ .078**（stage1_only，0 次 MISS），对 CL2 p=.013，对纯推理 .992 为 p=.27（不可分）。l10 的纯缓存 tail 在 R5 队列（r05_ptail）。
- **wrist_only key（只用守卫，K10 MISS）**：l10 500 .820、l10 50 .734、sp 500 .974、sp 50 .924。配对检验只在 sp50 显著优于双相机（p=.027，来自任务 4、9），l10 500 更差（p=.021），其余持平。IR 按比例折算（假设）。
- **MISS 减步 K2**：SR 与 K10 无差别。owner 裁定不纳入系统（§9 第 12 条），只作附注。
- **K5 因果识别**（守卫 MISS 处 CALL 对 CACHE）：**g500 ΔY=+.034 [+.004, +.066]**，成本持平；**g50 ΔY=−.002 [−.040, +.032]**，后续调用更多。**Q3**：两个规模都"一律调用"，没有任何有支持的节省情境。
- **control_step_library**：G 崩（.222 / .114），GS 持平（.772 / .628），关闭。
- **两时钟、noprog 4、周期类在 500 集库**：都被支配。
- **工程**：K6 修了串行锁；K8 测得 CPU 检索每决策 1.2–2.2 ms（PCA 占 31–49%），原生检索 2.1–7.4 ms；K9 GPU 同图增量 .36–.63 ms，PCIe 131 KB → 1.4 KB，top-1 ≥99.9%。
- **R5 构思要点**：A 发现 K10 的块尾被继承门否决 → C10（Q1）；B 认为离线目标排不了闭环 SR → 受限双损失求解器（r05_b1）；C 认为库增长慢、硬剪枝不行；D 的抓取核验收益很小（Q1 的 D1）。
- **R5 编码与 smoke 结果**：
  - C10 smoke：17 次 MISS 后都发出了策略块尾。D1 smoke：触发 1 次。
  - GR00T G10 smoke：14 次 MISS、13 次策略块尾、39 次缓存 HIT、39 次缓存块尾，owner IR .19。
  - Q6 wrist 盲走 smoke 正常。
  - Q4：grow250（l10 13,121 行 = 50 集 + 208 个成功集；sp 6,240 行）；示范数据 demo100 / 200 / 300（current 不在 bpool_cs 中，所以只有 100 ⊂ 200 ⊂ 300 ⊂ 500 是嵌套的；曲线全程 kref 5，而 500 端点用 kref 8）。

## 3. 需要 owner 的事
无。owner 已离开，/goal 要求不设阻塞。裁定在章程 §9 第 9–14 条。

## 4. 正在运行的东西（compact 后先核对：`tmux ls`，以及各运行目录 `chain_console.log` 的最后几行）
端口与 CPU：23150 / 23151（server CPU 0-17,44-61，chain 34-37,78-81），23160 / 23161（server 18-25,62-69，chain 30-33,74-77），23162 用于 smoke（14-17,58-61）。每个 server 24 个 worker。所有运行目录都在 `/home/weiland/trace_runs/os_closed_loop/`。
- **23160 / 61**：
  - `relay_B5`：正在跑 r04_gblind 的 l10 4 臂（ph2 / tail2u × 500 / 50），csl 已完成。
  - 之后 `relay_R5B4`：r05_ptail 4（纯缓存 tail l10 500 / l10 50 / sp 50，外加 wrist sp50 重复）→ r05_growth 4（grow250 refit / frozen × {l10, sp}，按清单 inits 25–49，每臂 250 集）→ r05_q1 6（C10 × {l10, sp} × {50, 500}、D1 × l10 × {50, 500}）→ r04_k7 的 `r4k7_p_sp_50_tail1ug` → r05_b1 π0.5 6（l10 500、sp 500、sp 50 的 hand / solve 成对）。
- **23150 / 51**：
  - `relay_A4`：正在跑 `r4f_p_l10_inf_s2001`，之后 `r4f_p_sp_inf_s2001`，再到 r04_gblind 的 sp 4 臂。
  - 之后 `relay_R5A4`：r05_q2 6（GR00T CycleTail G10 × {l10, sp} × {50, 500}，GR00T 纯策略 L10 × 2）→ r05_q6 4（wrist + tail + wrist 守卫）→ r05_demo_curve 8（DemoBlindAWM tail：refit 100 / 200 / 300 × 两套件，frozen50 200 × 两套件）→ r05_b1 GR00T 6。
- **codex**：只有 Q5 在跑（`task-mukmfjxz-i7jxc2`，thread `01a0e5d1-1b69-7583-b2ab-4826599464e9`，PID 2938164，CPU 34-37,78-81，可用 GPU ≤6 GB）。它会先跑完所有旧回归测试，再原子安装进插件（Q2 已交回）。影子模式有 2 臂规格（l10 50 / 500 纯缓存 AWM）；K7 tail 明确不支持。
- **Monitor**：合并 watcher `bash /home/weiland/.claude/jobs/a607dd74/tmp/r4_watch.sh`，读 `tmp/r4_watch.jobs` 与 `tmp/r4_watch.runs`（23 个运行目录），已报事件记在 `tmp/r4_watch.seen`。30 分钟到期后**静默重挂**。新运行目录追加进 runs，新 codex job 追加进 jobs。
- 接力脚本在 `/home/weiland/.claude/jobs/a607dd74/tmp/relay_*.sh`。
  - **插队方法**：正在执行的脚本不能原地改。只 `kill` 那个接力的 bash PID（它的子 chain 照跑），用 sed 生成新脚本、新 tmux 会话，等旧 chain 或旧接力结束再接手。`relay_R5A4` 和 `relay_R5B4` 还在等待，可以直接替换。
  - 跳过：写 `state/<arm>.DONE` + `.SKIPPED`。推迟：写 `.DONE` + `.DEFERRED`，由后面的接力删掉 DONE 后再跑。

## 5. 下一步（按顺序）
1. **Q5 交回后**：读 HANDBACK，确认插件已原子安装、旧回归全部通过；在 23162 上做影子 smoke（`--os-gpu-retrieval shadow`，看一致率、延迟和动作逐位一致）；把它的 2 臂追加到某条线的接力末尾。
2. **R4 闭环结束**（GR00T 盲走 8 臂、两个种子臂）后：更新 `rounds/r04/ANALYSIS_BRIEF.md`，加入 §2 的全部 R4 结果（K7 / K10 / wrist / K5 + Q3 / g500 三次 / L10 两次 / 纯缓存盲走与 tail / csl / GR00T 盲走 / K8 / K9；K2 放附注）。然后派 1 个 codex 分析 agent 写 `rounds/r04/ANALYSIS.md`，提交，R4 结束（task #37）。
3. **R5 闭环**：每出一臂就用 v、m 算 owner IR 并记账。r05_b1 用配对（同一批 init）比较 hand 与 solve；r05_growth 用清单子集，与 R2 的 50 / 500 CL2 在同一批 inits 25–49 上比；r05_demo_curve 拼出 50 → 100 → 200 → 300 → 500 的曲线（注意 kref 与 current 的差异）。
4. **R5 闭环结束后**：写 `rounds/r05/ANALYSIS_BRIEF.md`（仿 R4），派 codex 分析 agent 写 `rounds/r05/ANALYSIS.md`，提交，R5 结束（task #48）。然后用 K8 的脚本在空闲机器上复测搜索延迟（task #40）。
5. **每个阶段都提交**：只加本线的 .py / .sh / .md 与臂规格 json（`git add -f`），不加 results/、dev/、before/ 和大于 1 MB 的文件；作者 LinZiyang666，英文，不加 AI 署名，不 push。

## 6. 纪律与坑（本线专有，章程 §8 / §9 有全文）
- ⛔ CPU 38-43,82-87 属于他线；只按 PID kill；不 pkill；前台不长时间 sleep（等待用 Monitor）。
- **成本口径**：owner 口径为主，eager 口径（`closed_loop/ops/cost_table.json`）单列，不混用。owner 口径的 wrist / dummy 成本表在 `rounds/r04/cost_table_owner.json`（比例迁移假设）。**watcher 报的 IR 对 R4 / R5 臂是 eager 账本；报告前自己用 summary.json 里 `cost_ledger` 的 v、m 算**：`.152·v + .848·m`；wrist 为 `.0552·v + m·(.848+.0499)`。
- **配对检验**：journal 的 `task_uid` 是 `arm:eval:<task>:<init>`；用精确 McNemar。K5 估计器是 `rounds/r04/k5_rand/estimate.py`。
- **推 yaml**：`tether push --force` 到 `timan107:/tmp/oscl_stage/`，远端 `cp` 到 `.new` 后 `mv` 到 `/scratch/zixuans8/openpi_trace/os_cl/cfg/`；用 `sha256sum | sort -k2 | sha256sum` 对账。
- **smoke 用单独的运行目录**。codex 沙箱里 `/home/weiland/trace_runs` 只读，拟合落在 `/tmp/<agent>_fits/`，要自己复制并核对 sha；需要写 store 的 agent 用 `--full-access`。`/dev/shm` 剩余约 42 GB（Q4 新增了 18 GB 的库）。
- **store root**：arm 的 plugin_args 带 `--os-root <冷拷贝>` 时，预拟合要用同一个 root。
- **插件现状**：K6（按连接加锁）、K5（随机化）、K10（π0.5 policy tail）、Q2（GR00T policy tail、blocks、CycleTail）都已安装。不带新 flag 时行为逐字节不变。Q5 下一个合入。
- **collect** 偶尔因 `tether pull` 瞬时失败（COLLECT_FAILED），手动重跑 `ops.collect --run-root <R> <arm>`。
- **台账时间戳**：写之前先 `date`，别写超前（犯过一次）。
- **最近的本线提交**：bcd3d1c、e8980cd、44c9136、6e1cab5。

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
