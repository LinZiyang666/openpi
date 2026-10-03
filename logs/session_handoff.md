# Session handoff

> 多条线共用本文件。§0 为常驻初始化（不动）。**§1–§6 = 离线检索探索线（offline_search）交接，2026-10-01 19:4x CDT 更新：R8 全部完成（采集 / 影子 / 校验 / 报告 / 补充消融），等 owner 裁定提交**。step_diag / warm reset 线的交接原文在附录 A（该线已全部完成、待 owner 裁定提交）。

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



## 1. 现在在哪（2026-10-02 19:5x CDT）—— R10 完成；owner 拍板"三层设计"；sol 正在做 `R10Recipe`（部署类 + 构建器）

**当前设计（owner 10-02 19:5x 定，记忆 `project_offline_search_three_layer_design`）**：三层，全部只从示范库拟合，各格各库大小统一。
1. 缓存（R4 BlindAWM，10 控制步提交，kref 每任务 5 集为 5、否则 8）；
2. 只留无进展守卫（R8 onlynp 判官，库内留本集伪查询标定）；
3. 库内留本集修正器（每任务一个 RFF + 岭回归头，学"大模型 − 缓存"运动 6 通道前 10 步），强度 = .5 × clip((2 − r)/1.25, 0, 1)，即 astra 的 GC_dist。
- **已去掉"掉队交给策略"**：R9 阈值拟合于测试集。

**R10（owner 10-02 /goal，已完成 120 臂；报告 `exp/offline_search/rounds/r10/REPORT.md`，附表由 `tmp/r10/r10_table.py` 生成）**：
- **设计**：库大小 50–500 集（500 集差集池库按任务嵌套子集，每任务固定种子前缀）× {A 纯缓存, G 只留守卫, GC_loeo, GC_pair, GC_dist}，4 格，官方测试集各 500 局、只测一次。
- **数据纪律**：一切拟合只用示范库（`assert_fit_input` 拒读 os_closed_loop），测试集只测量。
- **结论**：
  - 守卫长任务每档 +4~18 pp；短任务 100 集饱和；随机 50 集远差于精选 50 集（π0.5 L10 .634 vs .718）。
  - 6 档合并 vs G：
    - LOEO：π0.5 L10 +266/−185（p=1e-4）、π0.5 Sp +87/−53、GR00T L10 +355/−257（p=8e-5）、GR00T Sp 持平；
    - PAIR：GR00T L10 无效（LOEO 胜 PAIR p=5e-5）；
    - GC_dist：长任务≈LOEO，GR00T Sp 胜 LOEO（+110/−77，p=.02）。
  - 长任务 500 集库已追平纯策略：π0.5 .900–.914 vs .908；GR00T .906 vs .898。
  - 修正器使守卫调用略减，IR −.01。
- **修正器训练范围（owner 问过）**：每个库大小只在自己那份库上 LOEO 训练（如 50 集：任务 0 头只用该子库任务 0 的 5 集共 277 行，检索只在另外 4 集里）；PCA / 度量 / σ / 守卫阈值 / 配对半径 / GC_dist 标尺均按库大小单独拟合。
- **代码 / 产物**：
  - sol：`rounds/r10/{data,method,train,build,selftest,tests,report}.py`，子库按行号引用父库，h100 示范库零新增；`heads2b/`、`training2b/`、`artifacts2b/` 为修正版；
  - opus 离线：`rounds/r10/analysis_opus/`（REPORT 代存）；
  - astra GC_dist：`rounds/r10/astra/`。
  - 运行根：`r10_size_{pi05,groot}`（A / G）、`r10_corr2_*`（LOEO / PAIR 修正版）、`r10_corr3_{pi05,groot,groot_b}`（GC_dist）；`r10_corr_*` 为作废 v1 未跑；`r10_corr3_groot` 链在 L10-400 处按计划中止（该臂由 `_b` 跑完）。

**进行中**：sol（codex `task-muror4p0-bp0fnh`，Monitor 盯 job json 状态）。
- 内容：写 `exp/offline_search/rounds/r10/recipe/` 的 `R10Recipe` + 构建器（任意库一条命令拟合三层）。
- 验收：对 r10_corr3_* 24 臂逐决策 0 差异；并在部署用精选 50 集 `current` 库出 4 臂到新根 `r10_recipe_current`（manifest = 测试集 500）。
- 简报 `~/.claude/jobs/a607dd74/tmp/r10/sol_recipe.md`。

**sol 交付后要做**：
1. 读 `rounds/r10/HANDBACK.md` 的 R10Recipe 节。
2. 把新源文件推 h100（`tether push --force <f> h100:/data/oscl_h100/openpi/<f>`）并 `sha256sum -c` 核对。
3. 跑 `r10_recipe_current` 4 臂：两车队串行 sync（脚本范式 `tmp/r10/next_on_fleet_v2.sh` / `after_log_v2.sh`：flock `/tmp/r8_coord/r10_sync.lock`，timan107 WPS 16，timan108 WPS 10，`OSCL_MANIFEST=$R/eval500.json`）。外部参数文件先复制进根并改 `arms.json` 指向。
4. 与 R8 / R9 精选 50 集结果对比，写进 R10 报告。
5. **停掉本会话 codex 中转进程对**：`~/.claude/plugins/data/codex-openai-codex/state/openpi-50fd553c5e274099/broker.json` 的 pid 及其 `codex app-server` 子进程，按 PID kill。

**待 owner 裁定**：
- 修正器按任务编号选头是否算"沿任务分治"；
- 是否补随机调用 / 空抓守卫的库大小消融（已提议，未答）；
- 本机 4090 + timan1 第三车队（未答；本机 4090 当前空闲）；
- R8–R10 提交（一律未提交）。

**本日其他要点**：
- owner 禁用 fable（记忆已更新）；"现在开始可以用 astra"。
- owner 追问后确认 R9 测试集污染：R9 修正器每任务头与掉队 / 闸门阈值拟合于测试集 init 0–19，20–29 用于筛选，30–49 被协调者开 49 次 → R9"全量 500"非干净测试数。已披露。
- codex 常驻进程空转：每工作目录一对 broker + app-server，空转 20–40% CPU。tether 热更新会话已清 56 对；本会话那对也已停（记忆 `reference_codex_broker_cpu_leak`）。
- h100 `/data` 余约 45G（保留 32G）；已删 h100 上 `runs/r09_astra_r5/serving_store`、`runs/r08_abl*`（结果均已在本地）。

### R9（2026-10-02 09:0x）—— 最终配方 8 格全量 500 已评估，报告 `rounds/r09/REPORT.md`；注意其修正器 / 掉队阈值拟合用了测试集（见台账 10-02 自查）

**R9（owner 10-01 19:5x 起）**：派 astra + fable（owner 点名）+ opus 自由探索"压低推理占比同时保住成功率"。
- **owner 硬规则**：
  - 不准偷看：init 30–49 只给协调者对冻结候选开盲，研究员不许读 `r09_holdout*` / `r09_astra_holdout*`；
  - 不再在"沿任务分治"上下功夫；
  - 拟合 / 评估 init 不重叠（拟合 0–19，筛选 20–29）。
- **唯一权威**：台账 R9 段 + `exp/offline_search/rounds/r09/REPORT.md`（白话结论、8 格全量表、否定清单、待裁、错误记录）。
- **最终配方** `rounds/r09/recipe/`（`R9Recipe` v3，sha bbee155c，已推 h100）：
  - 50 示范库与 L10-500 = 只留无进展守卫 + 一半修正器（π0.5 L10-50 另加掉队交给策略）；
  - Spatial-500 = 纯缓存。
  - 可选更省档：π0.5 500 示范库省看、GR00T L10-50 每局 20 次上限，默认关。
- **8 格全量 500（括号 = 留出 30–49）**：
  - π0.5：L10-50 .894(.875) @ .183、Sp-50 .968 @ .103、L10-500 .886 @ .154、Sp-500 .988 @ .078；
  - GR00T：L10-50 .832(.830) @ .200、Sp-50 .944 @ .109、L10-500 .874 @ .181、Sp-500 .956 @ .076。
  - 运行根：`r09_recipe_full_{p,g}`、`r09_l10_500_full_{p,g}`；对照为 R8 `r08_main` / `r08_abl` 500 局。
- **待 owner 裁定**：
  - 修正器按任务选头是否算"沿任务分治"。不含任务编号版 `r09_taskfree_full`：短任务无代价，长任务 −5~6 pp；
  - timan1 第三车队；
  - R8 / R9 是否提交。
- **跑法坑**：
  - 外部参数文件须复制进新根并改 `arms.json` 指向；
  - `chain_h100.sh` 会 exec 成 python，排队要 pgrep `ops.h100.control chain <root>`；
  - 两条 sync 同秒启动会互判"未认证持锁者"，必须错开；
  - 非标准模块名持锁会阻塞全车队；
  - h100 `/data` 只剩 ~44 GB（保留线 32 GB，`DISK_BLOCKED`），已删 `runs/r09_astra_r5/serving_store`；
  - 部署新方法后在 h100 逐个核对 sha。

### R8（2026-10-01 19:4x 完成）—— 采集 70/70、影子 70/70、全量校验 70/70 PASS、补充消融 44/44、报告定稿

**目标（/goal，owner 9-30 11:2x 设）**：R8 专心做 profile 工具并集成进系统 debug 工具，收集真实运行轨迹数据全量 500 集；先发 opus / astra 研究分段需要的 profile 工具；组别 1 纯推理、2 A+停滞+随机调用、3 R7 阶段系统、4 少看 / 看一半系统研究、5 其他；数据放 /home（放不下再搬 /archive）；免暂停点 1，不做完不停。owner 12:5x 追加：编码完派 opus 查正确性，之前的 R 可 commit push（已推 `b3dd3d7`）。
- **唯一权威**：台账 `logs/offline_search_exploration.log.md` §10「R8」。R8 文档：`rounds/r08/{IDEATION_BRIEF,SELECTION,CODING_BRIEF,REVIEW_1}.md`、`ideation/E1..E5/PROPOSAL.md`；数据契约 `exp/offline_search/debug/SCHEMA.md`；各编码交接 `exp/offline_search/debug/handbacks/HANDBACK_S*.md`、`rounds/r08/methods/HANDBACK.md`。
- **系统**：`exp/offline_search/debug/`（server 观察器 / client 采集 / transport 流式 / reader / validate / capacity / catalog / aug 延后影子 / tools/{decision,physical}）；插件 `--os-debug-dir/--os-debug-config/--os-oracle`；链 `debug/ops/chain_debug.sh`；R8 新方法 `rounds/r08/methods/`；臂与拟合 `rounds/r08/ops/`。**全部未提交**（R8 进行中）。
- **已验证**：GPU 真机 11 类方法 debug 开/关下发逐字节相同（修复后 GR00T 延迟 62.0→63.4 ms）；真仿真采集开/关物理逐字节相同；smoke 70 臂 × 20 集非测试全部通过（回执 + 模式校验）；影子策略与真实调用一致（MSE .0020 < 独立抽样 .0082）；合并测试 318 passed；opus 审查 `REVIEW_1.md` 全部修复。
- **存储实测**（每集）：π0.5 L10 9.0 MB、Spatial 3.8；GR00T L10 11.2、Spatial 4.5 ⇒ 正式约 255 GB，/home 余 2.0 TB，不需要搬 /archive。
- **R7**（已完成并已推送 `b3dd3d7`）：`rounds/r07/ANALYSIS.md`；R7 的"少看 / 看一半"是越权做了 owner 已搁置的项（owner 9-30 指出），R8 按 owner 新指令把它们作为组别 4 系统研究。

## 2. R7 关键结果（owner 口径；详见 `rounds/r07/ANALYSIS.md`）
- **四条预登记规则**：
  1. 易段多走一块再看（SF1）省视觉但保成功率：**不支持**——八格 −1.78 [−2.81, −0.76] pp，IR 8/8 更低（.060–.069 vs A .074–.078）；大库（500 集）持平（+0.7 / −0.2 / +0.6 / −1.0），小库（50 集）−3 到 −4。
  2. 阶段信号有用：**支持**——SF1 比无门控的 UF1 +1.98 [+0.88, +3.08] pp（非等成本）。
  3. 易段只看手腕（π0.5）：**支持**——−0.05 [−1.40, +1.28] pp，IR 4/4 更低（.056–.066；全部节省来自 R4 假设的手腕价 .055198，未实测）。
  4. 阶段倾斜调用（过渡段多调、内部少调）优于均匀调用：首跑**支持**——+2.70 [+1.05, +4.35] pp；**预登记复现未确认**：换种子两次合并 +1.78 [+0.60, +2.93]（一对成本差 .0153 超线 .0003 → 不支持），第二次单独 +0.85 [−0.80, +2.55]；首跑 GR00T L10-50 的 +8.4 大半是抽签种子噪声（CU 两种子 .756 / .812）。稠密库 ρ=.18：−0.55 [−1.85, +0.75]，不支持。
  - 手腕价实测：仅手腕 26.4 ms vs 全相机 62.1 ms（按请求路径 67.7 ms）→ 手腕看 .0646（假设 .0552）；SW 实测 IR .062–.073，仍 4/4 低于 A。
- **前沿**：28 个新点里 20 个不被 R6 支配、15 个在合并前沿（手腕 4、阶段倾斜调用 3）；**八格"追平纯推理 L10 的最低 IR"都没降**；无一点过 2 pp NI。
- **机理（A2）**：小库每次延长背后只有 3.6–4.2 个有效演示、盲走漂移约大库两倍、阀半径在小库反而更松，阀只在 0.1–1.4% 检查上动作；大库里真正保护的是"拒绝含失败演示的邻居"。CT 把调用从内部（.32–.39）挪到过渡段（.68–.77）。GR00T 盲决策服务端排队 0.08–1.5 s（IR 不计）。
- **profile（非测试 B-val，44 臂 × 20 集）**：全部过 §4 门；5 个探索者一致：评测 28 臂、SF 上限 1、淘汰 SF2、SF+SW 与组合分配器暂缓。

## 2b. R6 关键结果（R6 终版，修复 exception 集之后；owner 口径 π0.5 IR = .152v + .848m，GR00T .148v + .852m；详见 `rounds/r06/ANALYSIS.md`）
- **数据修复**：client 把 episode 中途异常（timan107 过载时 websocket 心跳超时）记成失败，R6 共 33 臂 316 集受影响，R2–R5 与 pilot 无。已剔除重跑（`closed_loop/ops/remote/purge_exc.py`，`chain.sh` 以后自动剔除续跑）。7 个修复臂 ledger 与客户端决策数差 ≤ 1.5%，前沿按 2% 容差放行。
- **A / B 三次重复**（`PAPER_AB.md`）：π0.5 l10-50 .714 → .827（+11.3）、l10-500 .827 → .879（+5.1）、sp50 .837 → .910（+7.3）、sp500 .976 → .982；GR00T l10-50 .611 → .725（+11.4）、l10-500 .830 → .872（+4.2）、sp50 .867 → .876、sp500 .964 → .959。重复间最大差 2.4 pp。
- **消融**（`ABLATIONS.md`）：
  - 学到的度量在所有 l10 格 +7.8~13.4 pp；
  - 直接 token PCA 与池化 PCA 在 7 / 8 格无差（"池化显著更好"已撤回，池化的理由是成本）；
  - 无进展守卫承担 l10-50 两格 B 的全部收益，在 GR00T sp50 有害；
  - Bmech 表明守卫的效应全部经由策略接管。
- **三问**：
  - Q1 选 R（留一重建残差），能预测缓存–策略分歧，但不能预测 SR / 调用价值；
  - Q2 旋钮 = 目标 IR ρ；
  - Q3 预登记没有提名放置门；
  - pilot 预登记结论都 inconclusive，因为校准切分每个任务只有 1 个 init。
- **配置 C 验证**（`C_VALIDATION.md`，28 臂 + 4 Bmech）：
  - ρ 成本标定 26 / 28 在 ± .02 内；
  - R 放置 ≈ 均匀放置（八格 −0.15 pp）；
  - 标定停滞触发器同 ρ 下 +1.15 pp [+0.12, +2.23]（八格事后合并），GR00T sp50 C30 .940 @ .298 追平纯推理；
  - 高预算下任务级风险 lottery 在 2 / 3 格胜 C；
  - 稠密 GR00T sp500 上 C 的调用有害（.934 < A .964）⇒ 缺一个库级不调用闸门。
- **前沿**（`frontier_final/`，图 `~/projects/openpi_ext/artifacts/frontier_r6/frontier_r6_final.png`）：
  - 按点估计追平纯推理 L10 的最低 IR：π0.5 l10-500 .197、π0.5 sp50 .442、π0.5 sp500 .118、GR00T l10-50 .453、GR00T l10-500 .184（C18）、GR00T sp50 .298（C30）、GR00T sp500 .052；
  - π0.5 l10-50 追不平（最好 C45 .894 < .904）；
  - 同时检验的 2 pp NI：没有任何低于纯推理成本的点通过。
- **网页** https://claude.ai/artifact/N9KTGqB8Hkpt5wUBEZcrRd v6（源 `~/projects/openpi_ext/artifacts/commit_cache/commit-cache.en.template.html`，生成时注入 IMGDATA = 旧 `frontier.png`、IMGDATA2 = `frontier_r6_final.png`）。

## 3. 需要 owner 的事（ANALYSIS §7.3 有详细理由）
**R7 暂停点 2 待裁定（ANALYSIS §8.2 A2 的排序提案）**：
1. ~~复现阶段倾斜调用~~ **已做（收尾检查）**：未确认，稠密库无效 → 建议**不再投入**按阶段放调用，调用保持均匀 + 标定停滞。
2. 把"少看"的阶段门换成"库质量门"（16 邻居全部来自成功演示 + 有效演示数够，阈值由库留一规则定，先非测试集 profile）——建议**做**；小库上的 SF 与无门控 UF 淘汰。
3. ~~实测手腕单路编码延迟~~ **已做**：仍省 .004–.016 IR；建议保留"看一半"（π0.5），并把全相机看改回原版路径（可再省 .005–.006）。
4. 量 / 修 GR00T 服务端排队墙钟——系统项，按需。
5. 提交：R7 代码 + 报告 + R6 停滞提速 + 台账 / 交接是否提交（画图脚本与 profile_results 数据不入库）。

⛔ **owner 9-30 00:0x：下面所有待办全部先搁置；R7 专心研究阶段级分配（想法 5）。等 owner 指示再动，不自行开工。**

**已裁定（R7 待办，台账 §10「R7 待办」逐条记录）**：
- **决策 1（9-29 19:5x）**：补齐 B 层消融，简化 B 层设计（没用的守卫不留）。要补：l10-500 与 sp500 格的触发器逐个移除；"A + 只留 no-progress" 联合移除臂（至少 4 稀疏格 + 2 个 l10-500 格）。
- **决策 2（9-29 22:4x）**：新方法去掉"按预测差异分配调用概率"的部件，只留按预算的固定比例随机调用 + 停滞检测强制调用。"库够好就不调用"开关、高预算均匀随机对照、扫成本旋钮、按任务选控制器上界分析：owner 说暂不管，另行讨论。
- **方向 3（9-29 23:0x）**：论文故事改为"任务级成本旋钮在线自适应"：部署时每个任务一个旋钮，从高往低调到可容忍的成功率折损，一段时间后自动收敛。按任务难度分流不作为方法。分流分析（`rounds/r06/analysis_r6/per_task_routing/`）可作为平衡位置的估计。
- **决策 4（9-29 23:2x）**：少步去噪救场（代码已有）与库自增长都先搁置；不准换模型；**补一组消融：CLIP 检索键 + 我们同一套按任务马氏距离**（其余照搬纯缓存配置，与其三次重复配对比较），旧 CLIP 库构建脚本 `exp/common/build_clip_cache_artifact.py`。
- **想法 5（9-29 23:5x）**：阶段级分配——借鉴传统机器人控制把轨迹分成阶段（接近 / 抓取 / 搬运 / 放置等），难段多花、易段用原子手段偷懒。协调者提议的离线验证（对 8 个库做分段 + 用 R6 随机调用日志统计各阶段调用价值）待 owner 点头。

**仍待裁定**：
1. **全量续跑（+31,680 集）**：建议**不做**。给不出 2 pp 证书，R 已选定，重复的都是同一批测试 init。
2. **无监督白化 Σ⁻¹ 对照**（8 个纯缓存臂）：**降为可选、低优先级**（9-29 22:xx 复核）。已有的欧氏消融（`p2_ablations/metric.py`）基线是「按任务逐维 z-score 后的欧氏距离」，已经排除了单纯尺度归一化；我们的度量比它高 8–13 pp 的结论成立。Σ⁻¹ 对照只回答「用整体协方差的无监督马氏距离行不行」，属于审稿人可能问的问题，不影响方向。
3. **l10 +0.04 显示口径**（owner 演示图 `~/projects/openpi_ext/artifacts/fig2_frontier_ab/` 里 l10 新点上移 0.04）：建议表格与正文一律用测量值，图若保留偏移就在图注写明。
4. **下一轮**：是否按 ANALYSIS §7.2 开 R7——C′（ρ + 均匀放置 + 标定停滞 + 预登记的库级不调用闸门），并在 RoboCasa365 上做第二基准。

## 4. 正在运行的东西（10-02 19:5x）
- 闭环：**无链在跑**；h100 MPS 开着；两车队空闲。
- codex：sol `task-muror4p0-bp0fnh`（R10Recipe）运行中。
- tmux 里残留的 r10 排队会话均已结束。

## 5. 下一步
1. sol 交付后按 §1"sol 交付后要做"执行。
2. 汇报 owner：三层设计在精选 50 集库上的测试集结果；回答待裁项。
3. 提交（等 owner）：R8–R10 范围 = `exp/offline_search/debug/`、`rounds/r08/`、`rounds/r09/`、`rounds/r10/`、`closed_loop/ops/h100/`、`tests/exp/offline_search/`、plugin / blind 的 debug 钩子、台账与交接；画图脚本不入库；commit 不加 AI 署名。

## 6. 纪律与坑（本线专有，章程 §8 / §9 有全文）
- **（R7 新增）非测试 B-val 闭环**：P3 v2 客户端的遥测包装要求服务器返回 `__p3__` v2 标记，普通插件臂用不了（每集 ValueError）。改用原版客户端 + 远端 wrapper `os_cl/run_arm_r7_bval_stock.sh`（末尾追加 B-val 池的 `--apool-record/--apool-dir`）+ run 内 `chain_bval.sh`（= 原版 chain.sh，改调用脚本名与 HERE 路径）；代价是没有逐控制遥测。
- **（R7 新增）没有 manifest 字段的臂**（SF/UF/SW）按全 10×50 笛卡尔集跑，DONE 标记是 `state/<arm>.DONE`；完成判定两种标记都要认。
- **（R7 新增）summary 的 `ir_per_five_controls` 是 eager 成本表**，不是 owner IR；一律从计数重算。SW 的 owner IR 用逐决策 `owner_cost`（手腕价）。
- **（R7 新增）codex 续聊同一 thread** 前要按 PID 停掉本会话 companion 的 app-server-broker（cwd = openpi 且启动时间对得上），否则 `thread-store conflict: active writer`；续聊用 `codex exec resume <thread> -m <model> -c model_reasoning_effort=xhigh -c sandbox_mode=workspace-write -c approval_policy=never -o <out> - < prompt`，放 tmux 里、Monitor 等 tmux 结束。
- **（R7 新增）codex 沙箱看不到 GPU**（nvidia-smi 报 driver 通信失败），GPU 对拍由协调者跑。
- **（R7 新增）评测车道**：`tmp/r7_lane.sh <label> <port> <cpus> 9000 22`（1 server、WPS 22，4 条共 88 worker；L10 为主时 timan107 空闲内存最低约 70 GB）；GR00T L10-50 带调用的臂约 1 小时 / 臂。
- ⛔ CPU 38-43,82-87 属于他线；只按 PID kill；不 pkill；前台不长时间 sleep（等待用 Monitor）。
- **成本口径**：owner 口径为主，eager 口径（`closed_loop/ops/cost_table.json`）单列，不混用。owner 口径的 wrist / dummy 成本表在 `rounds/r04/cost_table_owner.json`（比例迁移假设）。**watcher 报的 IR 对 R4 / R5 臂是 eager 账本；报告前自己用 summary.json 里 `cost_ledger` 的 v、m 算**：`.152·v + .848·m`；wrist 为 `.0552·v + m·(.848+.0499)`。
- **配对检验**：journal 的 `task_uid` 是 `arm:eval:<task>:<init>`；用精确 McNemar。K5 估计器是 `rounds/r04/k5_rand/estimate.py`。
- **推 yaml**：`tether push --force` 到 `timan107:/tmp/oscl_stage/`，远端 `cp` 到 `.new` 后 `mv` 到 `/scratch/zixuans8/openpi_trace/os_cl/cfg/`；用 `sha256sum | sort -k2 | sha256sum` 对账。
- **smoke 用单独的运行目录**。codex 沙箱里 `/home/weiland/trace_runs` 只读，拟合落在 `/tmp/<agent>_fits/`，要自己复制并核对 sha；需要写 store 的 agent 用 `--full-access`。`/dev/shm` 剩余约 42 GB（Q4 新增了 18 GB 的库）。
- **store root**：arm 的 plugin_args 带 `--os-root <冷拷贝>` 时，预拟合要用同一个 root。
- **插件现状**：K6（按连接加锁）、K5（随机化）、K10（π0.5 policy tail）、Q2（GR00T policy tail、blocks、CycleTail）都已安装。不带新 flag 时行为逐字节不变。Q5 下一个合入。
- **collect** 偶尔因 `tether pull` 瞬时失败（COLLECT_FAILED），手动重跑 `ops.collect --run-root <R> <arm>`。
- **台账时间戳**：写之前先 `date`，别写超前（犯过一次）。
- **最近的本线提交**：1080430（R6 完成）、3592468 / 19cb16c / b0e3d15（R6 检查点）、10b31b9（R5 完成）、90a3cef（R4 完成）。
- **（R6 新增）共享 GPU**：启动新线前先算显存预算；我自己的临时 smoke 也要计入（犯过：smoke 占显存导致 23150 线 OOM）。
- **（R6 新增）chain.sh 缺口**：一个端口 SERVER_DIED_AT_BOOT 时，同一臂另一个端口已起的 server 不会被关，孤儿会占端口和显存，下次报 PORT_BUSY。处理：`ss -ltnp | grep :<port>` 取 pid，确认 `--os-tag` 是本线的臂，按 PID kill，再 `tmux kill-session -t oscl<port>`。`line_P6A_loop2.sh` 已自动做这件事。
- **（R6 新增）同一 run root 两条 chain 并行**：`state/current`、`CHAIN.DONE` / `CHAIN.ERROR` 会互相覆盖（无害）；读结果用每臂自己的 `runs/<arm>/summary.json`。
- **（R6 新增）tether 偶发短暂离线**（14:29–14:51 timan107 STALE）：臂会停在 495–498 / 500，chain 自动续跑补齐，不用处理。
- **（R6 新增）远端 LIBERO 导入**：tether exec 的默认 HOME 不对，会触发 LIBERO 的交互提问。远端命令要 `export HOME=/home/zixuans8 LIBERO_CONFIG_PATH=/home/zixuans8/.libero`，并加 `< /dev/null`。
- **（R6 新增）codex companion**：`task --help` 会被当成 prompt 起一个真任务，并让 `--resume-last` 指向它，所以续聊一律按 threadId。
- **（R6 新增）tether 单文件上限约 447 MB**（`pull refused: code=too_large`）：20 集的未压缩遥测 tar 就有 482 MB。文件模式要先 gzip 再切块（`r06_p3_pilot/ops/collect_client_pilot.py`）；流式模式不受影响。
- **（R6 新增）读取器别抢在收回前跑**：watcher 在 ARM_DONE 时就报了，但 collect 在那之后，要等 CHAIN_DONE 或臂的 DONE 标记再跑 `read_v2`（犯过一次，报"缺遥测"）。
- **（R6 新增）显存按预算而不是按当下空闲**：sandpile 用量会临时掉到 6 GB 又涨回来，16:38 全模型线曾趁机起两个 server，已停。所有线都经 `tmp/gpu_gate.sh`。
- **（R6 新增）tether pull 并发被拒**（exit 75，`too_many_in_flight`）：`closed_loop/ops/collect.py` 已加重试；仍失败就手动重跑 collect。
- **（R6 新增）DONE 标记有两种**：`state/<arm>.DONE` 与 `state/<arm>.manifest_<sha>.DONE`，检查要都认（`compgen -G`）；`ls A B` 在任一不存在时返回非零，别用它判断。
- **（R6 新增）网络重置后 driver 会挂住**（9-29 02:46）：`stall_watch.sh` 报警后按 PID 杀远端 driver 与其 worker，再清孤儿 worker（PPID=1），chain 会续跑。
- **（R6 新增）timan107 过载**（9-29 07:0x）：143 个 worker 时可用内存 0、PSI full 60%，driver 几分钟内 ARM_INCOMPLETE。并发 worker 控制在约 120 以内。
- **（R6 新增）LIBERO-10 worker 更吃内存**：每个 1.5–2.1G。timan107 并发上限：L10 为主时约 85 个 worker，spatial 为主时约 120。`tmp/t107_mem_watch.sh` 做可用内存告警。
- **（R6 新增）车道脚本**（`/home/weiland/.claude/jobs/a607dd74/tmp/`，副本在 `rounds/r06/ops/`）：
  - `multi_lane.sh`：多 run root 认领队列 `os_closed_loop/r06_main_queue.txt`，`touch os_closed_loop/STOP_<label>` 停车道；
  - `repair_lane.sh`：exception 补跑；
  - `rec_lane.sh`：P3 流式录制；
  - 都经 `gpu_gate.sh`。
- **（R6 新增）stall_watch 同名误报**：smoke 臂与验证臂同名时，它按 `ls runs/*/<arm> | head -1` 取到旧 journal，会误报一次卡住。以后 smoke 臂换名字。
- **（R6 新增）子 agent 不能写报告类 .md**（harness 拦截）：让 agent 把全文返回，由协调者读完再存盘。
- **（R6 新增）提交清单过滤**：正则 `client_bundle` 误挡了 `build_client_bundle.py`，`/replays?` 误挡了 `replay_*.py`。过滤只匹配目录（带 `/`）或精确文件名，提交后用 `git status` 复核 M 项。

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
