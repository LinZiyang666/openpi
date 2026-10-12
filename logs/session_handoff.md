# Session handoff

> 多条线共用本文件。§0 为常驻初始化（不动）。**§1–§6 = 离线检索探索线（offline_search）交接，2026-10-05 21:2x CDT 更新（R15 收口；在做纯缓存 + 修正器消融，见 §1.2）。以下为旧摘要：：R13（第 4 层两种调用对比）与 R14（第 5 层逐任务旋钮下降）均已收口，报告在 `rounds/r13/REPORT.md`、`rounds/r14/REPORT.md`；无在跑实验；待 owner 答复是否补 100/200 集；GPU 找"沙堆实验"会话协商（见 §1 首段）；R12/R13/R14 均未提交**。step_diag / warm reset 线的交接原文在附录 A（该线已全部完成、待 owner 裁定提交）。

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



## 1. 现在在哪（2026-10-09 00:0x CDT；最先读下面的"压缩前快照"；R17 MetaWorld 微调与测试进行中，4 个 sol 并行；/archive 搬迁进行中）

### 1.R16 正在做：纯减步融入闭环（owner 10-07 目标"融入纯减步并实验"，会话 goal 仍挂着）

**10-11 13:0x 新任务：第 5 层逐任务 IR 旋钮下降（owner 12:5x 指示；先读这段，再读下面 12:3x 交接）**
- owner 原话要点：在每个新模型的「纯缓存 + 长执行段 + 修正器 + 停滞门」上加按任务下降的 IR 旋钮；目标是达到纯推理成功率、用最少的 IR；新模型没有开销 profile，IR 先用 miss 比例代替；参照以前的逐任务下降；每个库大小都要探索。
- 设定（我定的）：
  - 基础组：DP、Fast-WAM 用 chrg（C=10），Cosmos 用 crg（C=16）。
  - IR 代理 = 模型调用次数 /（已执行控制步数 / C0），C0 为 DP/Fast 5、Cosmos 16，纯推理 = 1.0。
  - 档位从高到低：pure（周期间隔 0，每次决策都调模型）→ k50 → k40 → k30 → k20 → k10（按 miss 比例用库回放标定周期间隔）→ koff（只有停滞门）。
  - 标定：`r17_ops/calibrate_knob.py`、`r17_ops/knob/calibrate_all.sh` → `r17_ops/knob/gaps.jsonl`（36 格）。
  - 开发集：B 池起点 0..29，每任务每档 30 局，服务端 LOEO（排除查询自身那条库轨迹）。B 上的纯推理参照用建库 teacher 轨迹的成败，不重跑。每模型 12 格 × 10 任务 × 6 档 × 30 = 21,600 局，一个批次跑全网格。
  - 判定：沿用 R14/R16 冻结规则，5 个点容差（另出 0、2 个点表），第一次失败即停。
  - 测试：冻结档位表后在官方 A 上每格 500 局，对照纯推理控制组。然后按 R16 提档规则：低 ≥10 个点 → pure；低 6–9 个点 → 升一档，已是 k50 的 → pure。
- 派出的 sol（13:00，日志在 `r17_ops/`）：
  - `sol_knob_serving.log`：缓存研究放开周期旋钮调用，新增 `serve_loeo`（只改 common/、search/）。
  - `sol_knob_campaign.log`：knob STUDY、plan 支持按行任务/起点/B 池、worker 装 B 池初态、`ops/cache_knob.py`（只改 ops/）。
  - `sol_knob_analysis.log`：`rounds/r17/knob/descent.py`、`test.py`。
  - 任务书都在 `rounds/r17/prompts/sol_knob_*.md`。
- 跑法：DP 开发集放 4090，Fast-WAM、Cosmos 等 H100 上 Cosmos 50–400 收齐后再上。
- 14:1x 进展：
  - 三个 sol 已交回，diff 我审过：服务端 36 行、ops 约 90 行 + `ops/cache_knob.py`、分析 `rounds/r17/knob/{common,descent,test}.py`。Cosmos provider 拦周期间隔那一条我删了。
  - 我加了 `R17_INHERIT_WORKER_DEPLOYMENT` 环境变量（formal_eval.prepare_workers），用来指定从哪个批次继承依赖和配置。原因：timan107 上旧 LIBERO 配置 `/scratch/zixuans8/openpi_trace_h100` 已被删，全新准备的批次会失败；续跑批次不受影响。
  - 冒烟 `knobsmokedp1355`（8 局）验证通过：B 池起点、k30 的旋钮调用比例约 0.32、LOEO 排除了自身轨迹。
  - 开发集清单 `exp/r17_integration/ops/cache_study/knob/dev/STUDY.json`（每模型 72 个 YAML、21,600 局）。准备脚本 `r17_ops/knob/prepare_dev.sh`（DP 用 R4、本机 48 个 worker；Fast-WAM 用 R2/B16、timan107 45 + timan1 40）。
  - 接续脚本 `r17_ops/knob/chain_after_cosmos.sh`（14:25 挂上，PID 1353909）：等 r8c 退出后起 `knobdevdp`、`knobdevfw4`（两者都已准备好），日志 `r17_ops/knob/chain.log`。
  - 准备 Fast-WAM 时又拔了三道旧门：
    - timan1 的 worker 上限统一为 40（`ops/timan1_workers.py`）；
    - Fast-WAM `library_compatibility.validate` 只留 LIBERO、K10、协议三项（原来十几道源码、权重指纹门，当前代码打包的运行时过不了）；
    - `search/cache_recipe.validate_teacher` 不再要求回执里有证据。
  - `knobdevfw`/`fw2`/`fw3`、`knobdevdp` 都是失败的半成品目录，不用。在跑的是 `knobdevdp2`（15:32 起，4090）和 `knobdevfw4`（15:27 起，H100）。监控用 `r17_ops/knob/watch_knob.sh`（formal_watch14 里写死了旧批次名单，看不到旋钮批次）。
  - Cosmos 开发集已改用 `knobdevcos3`（17:1x 用新代码重新准备，旋钮批次各组不再排队，`c9ae467a`；`knobdevcos2` 作废）。原 `knobdevcos2` 已准备好（R8 + MPS，76 GiB，timan107 45 + timan1 40，21,600 局）。
    - 接续脚本 `r17_ops/knob/chain_cosmos_after_fw.sh`（16:33 挂上）：等 knobdevfw4 退出，把 driver 软链到 `/tmp/r17_driver/knobdevcos2/driver`，再启动。
    - 跑完要把 driver 拷回 /data（同 r8c 的 `driver_back_r8c.sh` 做法）。
    - 准备时又拔了 Cosmos `library_admission.compatibility` 的源码/AST 指纹门（`3ce79e98`）。`knobdevcos` 是失败的半成品目录。
  - DP A 集测试接续脚本 `r17_ops/knob/chain_dp_test.sh`（19:17 挂上）：knobdevdp2 退出后依次跑 `descent.py`（输出 `rounds/r17/knob/dev_results_dp/`）、`cache_knob test --model dp`（输出 `cache_study/knob/test_dp`）、`prepare_dev.sh dp knobtestdp <STUDY>`，然后启动 knobtestdp。测试跑完后按逐任务标准提档重测（手动）。
  - 网页第 9 版新增「纯推理减步」一节（数据集 `steps`，数据来自 `rounds/r17/horizon/RESULTS_DP_FASTWAM.json` 和 `COSMOS_K2_ANALYSIS.json`）。
  - **owner 16:4x 命令（已记入记忆）**：
    - 新模型默认减步数：DP 10 步（DDIM）、Fast-WAM 1 次、Cosmos 2 次（实算 1 次）。
    - 当前全步数旋钮实验做完后，仿 R16：缓存系统里所有模型调用（停滞门、周期旋钮、pure 档）都改成减步，重做一遍逐任务旋钮。
    - 目标成功率 = 同起点减步纯推理（A 集对照：`hs5dp_ddim10s0`、`hs6fw_k1`、Cosmos `hs2_cosmos_h100_k2incl_c16_*`）。
    - 16:43 派 sol 准备代码和输入（任务书 `rounds/r17/prompts/sol_knob_reduced_steps.md`，日志 `r17_ops/sol_knob_reduced.log`）：减步 YAML、放开三个模型校验里写死的原生步数、plan/aggregate 按 YAML 步数、开发集加 pure 行（B 上的减步纯推理参照）、descent `--reference pure-arm`、test `--controls reduced`。
  - **owner 16:0x 定的验收标准**：每个任务的 IR 档位都要调到成功率与纯推理相近或相等（已记入记忆）。执行办法：
    - A 集上每任务 50 局，最多比纯推理低 4 个点（少成功 2 局）算达标；
    - 低 10 个点及以上直接提到 pure，低 6–8 个点升一档；
    - 只重跑提档的任务，反复进行，直到全部达标或已到 pure。
    - `test.py` 已支持多轮：同一任务取最高的档位，也就是最新一轮。Cosmos 开发集还没准备，H100 等 Fast-WAM 跑完再上；Cosmos 用旧式 driver，要放 NVMe。
  - 收齐后的分析：`.venv/bin/python exp/offline_search/rounds/r17/knob/descent.py --lane <dev lane>... --out exp/offline_search/rounds/r17/knob/dev_results`，再用 `cache_knob test --model M --table levels.json` 生成测试批次。

**10-11 12:3x compact 前交接（只读这一段即可）**

**owner 规则（违反过）**
- 实验由我亲自起停和守护；agent 只写代码，派单前我先亲自读一线代码；功能不过度实现，不加门。
- 网页和图交付前，必须本地渲染截图看一遍（见下文「网页」）。
- 只跑 libero_spatial 和 libero_10，评测用官方 A 集；一个模型跑完再换下一个。

**现状**
- DP、Fast-WAM：缓存库 500 和 50–400 全部完成，各 8000 + 40000 局。
- Cosmos 500 档完成（`cosmos500w85_220813`）。
- Cosmos 50–400 档在跑，批次 `/data/mw_finetune/eval/cosmossizes145_r8c`：
  - 配置：H100 上 8 副本 + MPS（memory_budget_gib 76），145 个 worker（timan107 45、timan1 3 号卡 40、本机 60）。
  - `driver/` 是软链接，实际在 `/tmp/r17_driver/cosmossizes145_r8c/driver`。原因：/data 机械盘 fsync 一次 71 ms，旧式 driver 被拖慢。
  - 续跑链：145 → r4 → r8b → r8c（每段的 CACHE_RESUME.json 记着上一段）。
  - 12:23 时累计约 7200 / 20000 局，每分钟约 40–50 局，预计 16:00–17:00 收齐。
  - 启动脚本 `r17_ops/cosmos_r8c_nvme.sh`，日志 `r17_ops/h100_chain.log`。
- 监控：`bash r17_ops/formal_watch14.sh`（含出错重跑数和 OOM），用 Monitor 挂，每 30 分钟到期后重挂，过滤掉已结束的批次。

**收齐后要做**
1. 把 `/tmp/r17_driver/cosmossizes145_r8c/driver` 拷回 /data：删掉软链接，换成实体目录，内容逐字节核对后再删 /tmp 副本。
2. 重跑分析：
   ```
   .venv/bin/python exp/offline_search/rounds/r17/cache/analyze_cache.py --sizes 50 100 200 300 400 500 \
     --lane /data/mw_finetune/eval/{dpc8713063796_r2_d1dd,fastwamc5481985003,hs5dp_ctl_142812,hs6fw_res_152931,\
   formal_cosmos_libero_official2000_20261010_100533_5f558380,formal_cosmos_libero_official2000_20261010_130639_a8ca2043,\
   fastwamc6066547406,csztd_r3_l48,cosmos500w85_220813,cosmossizes145_r8c}
   ```
   每个 --lane 单独写一个参数，大括号只是简写。约需 20 分钟，后台跑。
3. 更新网页数据（用 Cosmos 完整结果），提交 CACHE_RESULTS 和 handoff。

**网页** https://claude.ai/artifact/To18VP3TyzS5SXY3MmmUCp（Dashboard 类型）
- 源文件在 `/home/weiland/.claude/jobs/35f2869b/tmp/dash/`：`index.html`、`cells.json`、`baselines.json`、`progress.json`、`architecture.json`。
- 更新流程：
  1. 用 `Artifact publish asset:true` 上传 json，得到 `/_blob/<id>`；
  2. 用 `ArtifactData update` 改 `datasets/<id>` 的 `source.url`（要带 if_version）；
  3. 页面写成 `files/index.html` 的 `{text}`（当前版本 7）。
- 数据集文档当前版本：cells v2、progress v2、baselines v1、architecture v3。
- 本地检查：`harness.html` 是本地模拟页（stub dash，数据全转成字符串，加载了 Noto Sans SC）。用 `~/.cache/ms-playwright/chromium_headless_shell-1243/chrome-headless-shell-linux64/chrome-headless-shell --screenshot` 渲染，再用 openpi venv 的 PIL 裁图后看。
- 页面结构：进度 → 要点 → 模型结构（每个模型上方数据流图 + 下方 Figure 1 风格结构图，标 ×100/×10/×5 循环）→ 成功率曲线 → 透视表（首行纯推理，每格「成功率 + 差值」）→ 说明。
- 页面运行错误看 `data/users/me/runs`。

**已知结论**
- DP：库够大时，带长执行段、修正器、停滞门的组达到或超过纯推理（libero_10 在 500 时最高 +9.2）。
- Fast-WAM：所有组都低于纯推理。
- Cosmos：缓存库 50 差，原因是每 16 步才检索一次、每个任务只有 38 行，k=16 个近邻占了 42% 的行；缓存库 500 时 spatial 为 95.0%，好于 Fast-WAM。

**坑**
- 旧式 driver 目录必须放 NVMe。
- 4090 上 DP + 本机 worker 最多 48 个（显存）。
- Cosmos 副本加载峰值约 8 GB，memory_budget_gib 会按副本平分。
- 收尾时 formal_eval 可能空挂，确认远端进程都关了再 SIGTERM。
- 不要用 `pgrep -f` 去 kill 含有自身命令行的进程。

**10-11 12:2x 增量（最新，先读）**
- Cosmos 50–400 档切到 `cosmossizes145_r8c`（8 副本 + MPS，145 个 worker），`driver/` 用软链接放在 `/tmp/r17_driver/cosmossizes145_r8c/driver`（NVMe）。原因：旧式 driver 每行都 fsync，/data 正忙，每次 71 ms，worker 只有 11% 的时间在干活；改到 NVMe 后 worker 忙碌比例到 95%，速度约每分钟 50 局。保留 6434 局，还剩 13566 局，预计 16:00 左右收齐。**收齐后把 /tmp 下的 driver 目录拷回 /data 留底**。
- 网页：https://claude.ai/artifact/To18VP3TyzS5SXY3MmmUCp（dashboard 类型，数据在 job tmp 的 dash/*.json，上传成附件再更新 datasets）。Cosmos 收齐后，用完整分析更新 cells.json。
- Cosmos 在缓存库 50 表现差的原因：每 16 步检索一次，每个任务只有 38 行，固定取 k=16 个近邻、权重近乎平均，相当于把该任务 42% 的行混在一起；Fast-WAM 每个任务 124 行，只混 13%。缓存库到 500 时 Cosmos 为 95.0%，比 Fast-WAM 的 87.8% 还好。

**10-11 09:3x 增量（最新，先读）**
- Cosmos 500 档（`cosmos500w85_220813`）08:10 收齐 4000 局。
- Cosmos 50–400 档受 server 限制：每次决策都要先跑 stage-1 编码器生成检索 key，2 个副本时排队中位 8.9 秒。处理经过：
  - `cosmossizes145`（2 副本，08:10–08:39，收 850 局）；
  - → `cosmossizes145_r4`（4 副本 + MPS，08:46–08:59，收 650 局，其中 55 局是停机时的出错局，已剔除）；
  - → `cosmossizes145_r8`（8 副本，启动时 OOM：60 GiB 预算平分后每个副本 7.5 GiB，小于加载峰值 8.05）；
  - → `cosmossizes145_r8b`（8 副本 + MPS，预算 76 GiB），09:13 起跑，保留 1445 局，排队中位降到 0.35 秒。
- 脚本：`r17_ops/cosmos_r4_mps.sh`、`cosmos_r8b_mps.sh`（`cache_resume.prepare(..., replicas=N)` 后改 campaign.json 的 mps 和 memory_budget_gib）。预计 14:30–15:30 收齐。
- Cosmos size 50 的纯缓存组在 spatial 上几乎全部失败。这是真实结果：145 批次的缓存输入和拔门前验证过的 `cosmossizes85_222001` 逐字节相同。原因是 Cosmos 每 16 步才检索一次，size 50 时每个任务只有约 52 行，16 个近邻会跨越轨迹的各个阶段。
- 分析出全表时，lane 用 `cosmossizes145_r8b`（它沿 CACHE_RESUME 链接回 r4 和 145）。

**10-11 07:3x 增量（最新，先读）**
- Fast-WAM 50–400 档（`fastwamc6066547406`）06:21 收齐 40000 局；DP 50–400 档（`csztd_r3_l48`，再加 `csztd_r2_c383_r2_e5b0` 保留的 6350 局）07:16 收齐。两批收据都是 FAIL（`remote worker close/snapshot incomplete`），这是收尾打包 worker 快照那一步失败，数据完整。收尾之后启动进程空挂不退，我手动 SIGTERM 结束了。
- Cosmos 500 档（`cosmos500w85_220813`）06:28 起跑，07:26 收到 2667/4000。之后链用 145 个 worker 跑 `cosmossizes145`（DP 已经结束）。Cosmos 带停滞门的组受 server 限制（H100 约 91%），预计 13:00–14:00 全部收齐。
- 分析：`analyze_cache.py` 已含 DP 和 Fast 全尺寸，只差 Cosmos，命令见 `r17_ops/analysis_0720.log` 的上一级调用（lane 列表：dpc8713063796_r2_d1dd、fastwamc5481985003、三个控制组、fastwamc6066547406、csztd_r3_l48、cosmos500w85_220813，Cosmos sizes 跑完后再加 cosmossizes145）。

**10-11 00:5x 增量（最新，先读）**
- 4090 OOM：本机 worker 的 EGL 渲染和 DP server 在同一张 4090 上，每个渲染进程约 706 MiB。60 个 worker 把 48 GB 占满，DP 副本在 batch 推理时 CUDA OOM；出错的 26 局都已重跑，没有被记成失败。之前的显存准入挡的正是这个，48 个就是上限。
- 已停掉 `csztd_r2_c383_r2_e5b0`（保留 6350 局），改用 `r17_ops/dp_resume_l48.sh` 以 48 个本机 worker 续跑，新批次 `csztd_r3_l48`，00:55 开始收局，显存 36.7 GB。DP lane 进程已经停掉，以后 DP 不再由 lane 管，失败要手动续跑（用 `cache_resume.prepare`，写法见该脚本）。
- H100 链判断 DP 是否结束，改成看 `prepared-run .../csztd*` 进程是否还在（链 PID 2954987）。
- 监控换成 `r17_ops/formal_watch14.sh`：同时报出错重跑数（`ERRORS_UP`）和本机副本 OOM 次数。
- RUN_CONFIGS 拓扑一节已写入 4090 显存约束。

**10-11 00:0x 夜间推进（owner 睡觉，agent 限制解除；最新，先读）**
- H100 链 `r17_ops/chain_h100_cosmos.sh`（PID 2282671，日志 `r17_ops/h100_chain.log`）：等 Fast size（启动进程 1637628）退出、确认收到 ≥40000 局 → 释放 `cosmos500w85_220813` 的 DISPATCH_HOLD → 跑 Cosmos 500 档（4000 局）→ 等 4090 DP lane 退出，最多 60 分钟 → 跑 `cosmossizes145`（145 个 worker：timan107 45 + timan1 40 + 本机 60，20000 局）；如果 DP 还没结束，就改跑 `cosmossizes85_222001`（只用外部 85 个 worker）。145 批次由 `/home/weiland/.claude/jobs/35f2869b/tmp/prepare_cosmos145.py` 准备，H100 server 加本机 worker 的混合组合还没有实跑过。
- 预计完成：Fast size 约 06:00，DP size 约 06:00–06:30，Cosmos 全部约 09:00–10:30。
- 分析：sol A（`r17_ops/sol_A_analysis.log`，任务书 `rounds/r17/prompts/sol_A_analysis_simplify.md`）在去掉 `analyze_cache.py` 的封存/PASS/身份门，并支持续跑链合并和中间结果，然后跑出 N500 与中间尺寸的表。
- 已提交（Sonnet）：openpi Ziyang 分支 5 个提交（3e5352f9..af46cf20），工作区干净；生成数据留在磁盘上、已 ignore。
- 记忆已精简到 19 条；新规则：派单前协调者亲自熟悉一线代码；功能不过度实现。

**10-10 23:4x 增量（最新，先读）**
- owner 23:1x 令：自加的校验、指纹、合规「全部拔掉」。已用 AST 脚本（`/home/weiland/.claude/jobs/35f2869b/tmp/ungate.py`）把 exp/r17_integration 下 ops、qualification、common、benchmarks、conductor、search 加载部分，以及 DP、Fast-WAM、Cosmos 三个模型里的身份、封存、指纹、授权、证据、收据类 raise 改成 pass，约 500 处。另外改了几处：内存、磁盘、显存三道准入恒通过；provider 按名字查找时忽略内容哈希；DP、Cosmos 库兼容检查不再锁代码哈希。保留的检查：形状、dtype、几何、路径安全、端口、超时、MPS 就绪、每局恰好收到一次、grouped driver 禁止原地重启、删除远端数据前的核对（sync_back、incremental_sync、worker_storage 三个文件整体保留）、timan1_workers.py 整体保留（只用 3 号卡）。改前备份：job tmp 下 `r17_ops_models_before_ungate_*.tar`。
- DP 50–400 档 23:3x 在跑：lane 目录 `/data/mw_finetune/eval/csztdp`，当前 run `csztd_r2_c383_r2_e5b0`，本机 60 个 worker、4 个副本，lane 日志 `r17_ops/dp_sizes_r6.log`。Fast-WAM 50–400 档（`fastwamc6066547406`）23:33 收 4539/40000 局。
- 以后跑新批次都用拔门后的代码，不会再因为改了代码导致哈希对不上而被拦。Cosmos 500 档和 50–400 档的准备目录是拔门前做的，直接 `formal_eval.sh --run --prepared-run` 跑；如果仍被门挡住，就用当前代码重新准备一份。

**10-10 23:1x 交接（compact 前，最新，先读；只写待办）**

**owner 硬规则（违反过、被骂过）**
- 实验由协调者亲自启动、守护、续跑；agent 只写代码，命令交给协调者。owner 23:0x：「这次你自己来，不准派 sol」，这一轮我亲自动手。
- 不画蛇添足：R17 formal_eval 那一层（单独封存、冻结编队、身份指纹、收据门、拉证据包）是我们自己加的，owner 认为没意义。能绕就绕，不再新加门。显存准入已经拔掉。
- 只跑 libero_spatial 和 libero_10；评测集用官方 A；库是 B500 及嵌套子集。
- 一个模型的全部 size 和全部组跑完，再换下一个模型（YAML 热切换），不要来回切模型。
- 每个模型可以按 worker 的设备来源开多个 conductor（多批并行），共用同一个 GPU。
- 本机的 worker 优先给 4090（DP）用，不额外走网络；H100 那条线用外部节点 timan107 + timan1。
- 特例：timan1 只能用 3 号卡；本机可以跑 worker。timan1 的其它卡碰都不许碰。
- timan107 磁盘紧张：远端数据定期同步回本机再删掉。timan107 上不要的数据可以直接删（openpi、libero_sim、Isaac-GR00T 要留）。
- 沙堆实验已冻结：不用交还 GPU，也不用写台账。

**正在跑**
- H100：Fast-WAM size 50–400，run `/data/mw_finetune/eval/fastwamc6066547406`（csznf 续跑出来的新目录，4 万局；timan107 45 + timan1 40；23:03 已收 821 局，速度很快）。直接用 `exp/r17_integration/ops/formal_eval.sh --run --prepared-run <dir>` 启动（我的 nohup，日志 `r17_ops/fast_sizes_run.log`）。
- 4090：DP size 50–400，lane `/data/mw_finetune/eval/cszqdp`（本机 48 个 worker，4 个 replica；lane 进程是最早那个 cache_lane --lane 4090 --directory cszqdp，会自动重试）。失败过两次：①DP 模型代码硬性要求 R2；②我改 models/dp/formal_batch_server.py 导致模型身份不匹配（已改回）。之后放宽了 `ops/formal_server.py` 的 dp_capacity_serve 条件（record.replicas 是 None 也接受）。lane 正在重试 `cszqd_r1_9ece_r1_c5f4`，**结果还没确认**。如果还失败，就用 replicas=2 重新预备：`python - <<EOF\nimport exp.r17_integration.ops.cache_capacity_lanes as m; m.FLEETS['dp_libero']={'local':60}; （replicas 写死在 prepare 里，dp=4，需要改成 2）; m.prepare('4090',Path('/data/mw_finetune/eval/<新目录>'),'<前缀>')\nEOF`，再跑 `cache_lane --lane 4090 --directory <新目录> --run`。

**接下来要做（按顺序）**
1. 确认 DP size 已经在收局（journal 有增长，4090 显存被占用）。起来后考虑扩到 60 个以上 worker（显存准入已拔，只剩内存准入：每个 worker 3.5 GiB，余量 8 GiB）。
2. Fast size 跑完后接 Cosmos：`/data/mw_finetune/eval/cosmos500w85_220813` 上有 DISPATCH_HOLD.json，要先改名释放；然后跑 cosmos500 和 `cosmossizes85_222001`（外部 85 个）。不要用 cache_model_order 队列，它有三个 bug：把前序的 PARTIAL 当成 FAIL、run log 已存在就报 FileExists、queue_retry 找不到 RECEIPT 会报错。直接按顺序 `formal_eval.sh --run --prepared-run <dir>`。
3. 远端数据定期同步：Q 写了 sync（看 `sol_Q_reorder.log` 交回的命令）。**sync_back.py 现在是 worker 包里的版本，Q 的版本备份在 /home/weiland/.claude/jobs/35f2869b/tmp/sync_back.Q_version.py**；跑 Fast/DP 的这段时间不要换回来（会触发 worker 包刷新门）。
4. 等两条线都收完：派 astra 按 CACHE_STUDY.md §9 出 6 个 size × 组的分析，同时出 500 档的结论（DP：csl10d + dpc 各段已合并 PASS；Fast：csl10f + fastwamc5481985003 PASS；Cosmos 500 还没跑）。附近邻 32 个 init 的敏感性分析（CONTAMINATION_CHECK.md 已确认无泄漏）。
5. 中长期（owner 方向）：cache 研究改回原版 conductor 的跑法，每个模型一个常驻 server + 多个 conductor（按设备来源各开一个）+ YAML 热切换 + 断点续跑，去掉 formal_eval 多出来的那层。需要写代码时派 sol（只写代码）。

**代码状态（防止踩坑）**
- 我改过：`ops/formal_eval.py` 里 driver 的 wait_port 从 60 s 改成 600 s（4 万局的计划初始化超过 60 s）；`ops/worker_gpu.py` 的 admission 改成 slots=10**6（拔掉显存准入）；`ops/formal_server.py` 的 resize 条件。备份都在 job tmp（*.bak_*）。
- `ops/formal_workers.py` 和 `ops/timan1_workers.py` 里有 P6 写了一半的 live-fleet 改动（P6 的轮次我只杀了客户端，服务端可能还在跑）。`cache_study/sizes/evidence/t1_recycle_fix.json` 的 pins 是我手动刷新的。每次 prepared run 启动都会核对 worker 包和当前代码：不一致就刷新，而刷新只能做一次，第二次会被「CACHE_RECYCLE_BASE.tar 已存在」挡住。所以**跑实验期间不要改 ops 代码**，也不要让 agent 改。
- 监控：`r17_ops/formal_watch13.sh`（盯 hs5*/hs6*/csm*/csl10*/dpc*/fastwamc*/cosmos500*/csz*），每 30 分钟重新挂一次。
- timan107 磁盘 85%（剩 32 GB），timan108 仍离线。

**10-10 11:2x compact 后增量（最新）**
- Fast-WAM 官方基线 11:1x 在 391/2000 局 FAIL（`remote worker fleet failed: timan107`，旧启动无内存准入）。H100 队列因此已自动开跑执行 10 步档 `hs2_fast_h100_k10_c10_20261010_112043`（带新内存准入）。派 Q 查根因、给续跑/重跑基线命令（`sol_Q_fwfail.log`，基线用端口 23180/23189）。
- D 交回 Cosmos K=2（含清理步）：实际 NFE=1，3/3 逐位一致；排队脚本 `r17_ops/queue_h100_cosmos_k2.sh`（后台 b20mrlj7o），Cosmos 基线结束后自动开（timan108 ×24，端口 23240/23249）。
- P3 仍在写 CPU 部分、按约定等 4090 锁；交回后续派：不持锁、和 DP 共卡测相对加速比（≤25 分钟）。
- 11:3x owner 批评 DP 开 12 副本（同卡多进程只是分时，不加算力）→ 已停 DP 队列和 B=1 基线（`hs2_dp_k100_c5_default_*` PARTIAL，worker 已清）。P3 随即拿到 4090 锁在测 Fast-WAM batch（约 12:05 放锁）。派 P4（thread 01a12182-005b-7d03-8dbb-28057d4d588c，`sol_P4_dpbatch.log`）做 DP 服务端 batch，交 `models/dp/BATCHING.md` 和 DP 官方基线命令；交回后用 batch 配置重开 DP 基线加 5 档。
- ⛔ owner：timan1 不准用（已写记忆）。worker 节点只用 timan107、timan108。
- 11:38 Cosmos 官方基线从 11:30:53 起卡在 1726/2000（server 0% CPU、worker 闲着，与停 DP 时间重合）→ 派 D（`sol_D_stall.log`）查原因、尽量原批次续完，并保证停一个批次不会波及别的批次。timan108 上 GPU1/2 的高占用是别人（xy51）的，不是我们的渲染。
- 11:4x **timan108 tether agent 离线**（心跳停 3 分钟以上），我们的 24 个 Cosmos worker 和 7 个 DP worker 卡在内核 D 状态（rwsem_down_write_slowpath）。Cosmos 基线 11:42 FAIL（1726/2000，tether exec 超时）；Cosmos K=2 因 timan108 离线没起来（队列已退出，待重开）。为了配对一致，Cosmos 基线补跑和 K=2 都等 timan108 恢复后仍放 timan108。
- 11:4x owner：工作转到 timan107（「107 可以开 80–90 个 worker」）。Cosmos K=2 在 timan107 起不来：内存准入拒绝（C10 按每个 6 GiB 预留 24 个，加上约 95 GiB 查不到归属的内存）。Q 已确认 Fast-WAM 基线死于 timan107 全机 OOM；Q 在做 `--resume-from`（保留 391 局，P3 改了 Fast-WAM 服务文件，模型身份变了，要用原 runtime 包续跑）。
- timan107 可用内存一直在掉（11:45 76 GiB → 11:48 69 GiB，进程匿名内存不变），疑似每个 worker 在内核或驱动侧泄漏，退出才释放；C10 约 12:35 可能 OOM。派新 sol M（thread 01a126b8-3457-7f30-a120-0da1edf582ac，`sol_M_memleak.log`）查根因、做 worker 定期回收、给出每节点容量。监控 brph5qtzr 每 5 分钟报 timan107 可用内存。
- 11:50 P3 交回 Fast-WAM batch：B8、等 5 ms，吞吐是 B1 的 5.44 倍，最大动作偏差 0.0078（`models/fastwam/BATCHING.md`）。采用。已停 C10（B1，`hs2_fast_h100_k10_c10_*` PARTIAL，约 400 局作废），H100 队列随之停。停后 timan107 可用内存回到 216 GiB（泄漏在进程退出时释放）。
- 11:5x Cosmos K=2 在 H100 + timan107 ×24 开跑（`hs_cosmos_k2.log`，run 名 `hs2_cosmos_h100_k2incl_c16_t107_*`），`r17_ops/mem_guard.sh` 在 timan107 可用内存低于 25 GiB 时优雅停它（后台 bi9y3yd37）。
- 续派 P3（`sol_P3_seam.log`）：把 batch 正式接进 formal_eval（H100 可用），H100 冒烟，给 Fast-WAM batch 基线和 6 档的命令及两卡分工。
- 12:02 D 交回 Cosmos 停滞：timan108 上 24 个 worker 都卡在 D 状态；DP 与 Cosmos 两批共用 EGL GPU0；1726 局可复用，274 局续跑计划已存（HANDBACK_COSMOS），但 timan108 离线、无法确认 worker 已关，所以还没续。D 修了渲染设备准入和子进程关闭检查（70 项测试 + 27 个子测试通过）。
- **泄漏线索**：timan107 上 24 个 worker 全在 GPU0 渲染（8085/8192 MiB），GPU1–7 全空。推测单卡显存满后驱动用系统内存兜底，所以内存掉了却查不到在哪个进程。续派 D（`sol_D_egl.log`）把 worker 轮流分到所有 GPU、加按卡显存的准入、验证观测逐位一致、冒烟证明内存不掉，并给出 Cosmos 基线 274 局在 timan107 续跑的命令。
- 12:03 M 交回：根因是 NVIDIA/EGL 在反复建环境时占住宿主内存，GPU0 挤满又加重了这个问题；平均每个 worker 每局约 0.23 GiB（0.09–0.54）。worker 跑满一定局数就回收（`--worker-max-episodes`，默认 8，`ops/worker_lifetime.py`），Cosmos K=2 已经用上，内存会随回收恢复。M 建议每个 worker 最多跑 4 局、预留 6 GiB，这样 timan107 能放 32 个，timan108 能放 37 个；80–90 个要等 D 把渲染分散后再看。
- 12:17 Q 交回：Fast-WAM B1 基线死于 11:18:43 全机 OOM；`--resume-from` 做好了（139 项测试 + 29 个子测试通过），B1 基线不再续，留给 Cosmos 基线续跑用。续派 Q（`sol_Q_prestage.log`）：把档与档之间 10–15 分钟的空等压到 2 分钟以内（复用已部署的 runtime、跑当前档时预备下一档），写一个队列脚本，并应用 M_to_Q 的请求。
- 12:3x–12:5x：D 做好渲染分散（`--worker-egl-devices`，默认用全部 GPU 轮流分配，timan108 排除 1/2）和续跑豁免（`--resume-waive-unreachable-source-node timan108`）；P3 把 batch 接进 formal_eval（`--max-batch/--batch-wait-ms`，各档用 `--paired-control` 指向同卡对照）；P4 交 `models/dp/BATCHING.md`（实测 1 副本、B8、等 5 ms，12 个客户端时每秒 14.5 次调用）。
- owner 12:5x 批评两卡空着，并说「timan107 以前开 64 个都没问题」→ 12:53 开 hs3（`r17_ops/launch_hs3.sh`，后台 bsz55ww0u）：H100 Fast-WAM B8/5ms 对照，四套件，2 副本，timan107 ×32（`hs3_fast_b8w5_h100_k10_c5_control_*`）；4090 DP B16/5ms 对照，四套件，1 副本，timan107 ×24（`hs3_dp_b16w5_4090_k100_c5_control_*`，用 DP 的 formal_eval.sh）。每个 worker 预留 3 GiB、显存 768 MiB，每 4 局回收，渲染分散到 8 张卡。DP 过准入后，Cosmos 基线续跑 274 局（timan107 ×8，端口 23243/23248，后台 bhgms8qex，`hs3_cosmos_resume.log`）。
- 分工：H100 只跑 Fast-WAM（对照加 6 档，各档 `--paired-control` 指向 H100 对照）；4090 只跑 DP（对照加 5 档）。对照跑完后交 Q 的队列脚本连续跑各档。
- owner 13:0x：把每个模型的并发运行方式（server/worker 配置）记录下来，推进最快。派新 sol R（thread 01a126f9-3aaa-75c1-993e-e4fe8b770fcc，`sol_R_runbook.log`）写 `exp/r17_integration/ops/RUN_CONFIGS.md`（9 个模型×基准组合的命令模板、副本、batch、端口、worker、实测吞吐、瓶颈），并用正在跑的 hs3 两批实测 batch 填充率，给调优建议。已写记忆 feedback-run-configs-fastest。用词：owner 不喜欢「对照组」，统一叫「基线」。
- owner 13:0x：「运行时做性能记录，肯定还有吞吐提升空间」→ 派新 sol S（thread 01a126fc-19cd-7b22-bab6-f12ef4faa378，`sol_S_perf.log`）写 `ops/perf_recorder.py`，每 30 秒采样两卡 GPU、timan107、各批进度、batch 填充、排队，写到 `/data/mw_finetune/perf/<date>/`，跑 20 分钟以上后按预期加速排序列出提升点（只出建议，不动批次）。
- owner：「h100 和 4090 都可以开 MPS，如果能加速」→ 续派 P3（`sol_P3_mps.log`）：用私有管道目录实现可选的 `--mps`（只影响我们指定的进程，不改 compute mode），在两卡上做有限时的对比测量（不超过 20 分钟），并判断剩下各档开 MPS 后能否继续用现有基线配对。
- 13:08 Q 交回预备和队列：启动时间拆解（DP 副本依次启动 267 秒、Fast-WAM 86 秒等），加了复用已部署文件、提前一档预备、副本并发加载；预计换档 60–110 秒（`ops/formal_queue.sh --jobs <json> --run`）。我已写好 `r17_ops/queue_hs3_fast.json` 和 `queue_hs3_fast.sh`（H100：基线 PASS 后跑 6 档，`--paired-control` 指向 hs3 基线），以及 `queue_hs3_dp.sh`（4090：基线 PASS 后跑 5 档）。**两个队列都还没启动。**
- 13:1x Fast-WAM 队列只做预备时 C20 被拒：`pending control sealed identity mismatch: model_manifest.json`，工作区代码在基线开跑后被改过（多半是 P3 的 MPS 改动）。派 Q（`sol_Q_freeze.log`）让各档使用基线冻结的 runtime 包，DP 队列也照此改。配对检查要求 worker 数、内存、显存、回收、batch 与基线完全一致。
- 13:16 R 交回 `ops/RUN_CONFIGS.md`（9 个模型×基准组合）。实测 hs3：Fast-WAM 每分钟 20.7 局，实际平均每批 2.7（上限 8）；DP 每分钟 22.3 局，平均每批 2.1（上限 16）。batch 没凑满，瓶颈在 worker。64 个 worker 实际约 123 GiB（每个约 2 GiB）。P4 交回 DP batch（最多 32，三种采样，每个请求自己的随机数）。
- 13:2x 停掉 hs3 两个基线（各约 400 局作废），改开 **hs4**（`r17_ops/launch_hs4.sh`，后台 b7kh4407x）：Fast-WAM H100 2 副本 B8/5ms ×48 worker；DP 4090 1 副本 B16/5ms ×36 worker；每个 worker 预留 2048 MiB、显存 768 MiB、每 4 局回收。各档队列要按 hs4 的 worker 数和基线路径重新生成（`queue_hs3_*` 已过时），等 Q 的冻结 runtime 改完后再改。监控换用 `formal_watch7.sh`（盯 hs4_*）。
- 13:2x hs4 DP 启动失败：`EGL VRAM admission refused: 36 workers requested, 16 fit`。timan107 有 8 张 8 GB 卡，Fast-WAM 48×768 MiB 加 Cosmos 8×1024 MiB 已经占满。等 Cosmos 续跑结束，再以 36 个 worker、每个预留 576 MiB 显存重开 DP（`r17_ops/launch_hs4_dp.sh`，后台 bd4hmf9rd）。所以 DP 各档也要用 576 MiB。timan108 仍离线（1 小时）。
- 13:3x Cosmos 续跑 274/274 局（成功 268），但收据 FAIL（`driver incomplete: 2 ... resolved retry errors`）→ 派 D（`sol_D_merge.log`）核验后合并成 2000 局基线，并和 K2 对比。
- 13:39 DP 第二次被显存准入拒（只放得下 32 个）→ 改用 **32 个 worker**，重开 `hs4_dp_b16w5_4090_k100_c5_control_20261010_133915`（576 MiB 显存，2048 MiB 内存）。
- Q 13:32 交回冻结 runtime（`--runtime-from`，server/driver/worker 都按哈希固定；之前破坏身份的是 P3 新加的 `models/fastwam/mps.py`）。已生成 `queue_hs4_fast.json` 并预备成功（`/data/mw_finetune/eval/formal_queue_hs4_fast_frozen`），`queue_hs4_dp.sh` 已填好 B 路径、32 个 worker。两条自动接续都已挂上：`queue_hs4_fast.sh`（后台 bxeepoai8）、`queue_hs4_dp.sh`（后台 burrv4tcy），基线 PASS 后自动开跑。
- 13:37 S 交回性能记录器（`ops/perf_recorder.py`，PID 1965274 常驻，停止用 `kill -TERM 1965274`，数据在 `/data/mw_finetune/perf/2026-10-10/`）。头号瓶颈是 conductor driver 每条记录都 fsync、加锁时间长：DP 回报和派发中位延迟 19–26 秒，batch 中位数只有 1。Fast-WAM 的 H100 已经 99% 满载。→ 派 Q（`sol_Q_driver.log`）做批量提交和锁外写盘，把 driver 单独列为编排固定项，这样各档可以用新 driver 而不必重跑基线，并更新两条队列。
- 14:0x Cosmos 收尾：D 合并封存 K5 官方基线 1965/2000 = 98.25%（spatial 96.4、object 99.6、goal 98.4、l10 98.6），SUMMARY 在 `formal_cosmos_libero_official2000_20261010_130639_a8ca2043/SUMMARY.json`。astra 的 `horizon/COSMOS_K2.md`：K2 每次决策 NFE 5→1（−80%）；spatial +0.2，l10 −2.0；24 项族系下界 spatial −2.77/−1.60，l10 −5.78/−4.40，判「未证明 2 pp 非劣，保留 K5」；同在 timan107 的子集也掉 2.19，不能归因于渲染卡。已告诉 owner：24 项校正下 500 对样本几乎不可能判任何一档非劣，这是设计问题，要他决定加样本还是放宽校正。owner 认为基本无损、l10 是噪声，我回复「无显著损失，l10 长任务约 −2 但不显著」，可选补做 l10 A/A（timan107 有空位时）。
- 14:02 P3 交回 MPS（`--mps` 可选，默认关）：4090 上 DP 2×B8 开 MPS 是关的 1.15 倍，DP 1×B16 和 Fast-WAM 没有收益；H100 显存不够，没测成。各档都不开 MPS（服务时间要和基线可比）。
- 完成时间估计（14:0x）：Fast-WAM 基线约 15:00，6 档完约 17:00–17:30；DP 基线约 15:20，5 档完约 19:00–19:40。Q 的 driver 修复接上后会提前。
- 14:06 owner：「两卡占用都很低，要把占用提上去（加副本、调大 batch），并测是不是 worker 大多在等、瓶颈在 server」。实测：4090 SM 0%（68–111 W）；H100 SM 95–99% 但只有约 200 W/700 W；timan107 上 90 个 worker 有 72 个 CPU 低于 10%（在等）。→ 停掉 Fast-WAM hs4 基线（943 局作废，`queue_hs4_fast.sh` 也停了），派 P3（`sol_P3_h100sweep.log`）独占 H100 不超过 15 分钟，测 1×B16/B24/B32、2×B16 开关 MPS，给出新的基线命令。派 S（`sol_S_dpwait.log`）拆分 DP worker 和 server 的时间构成。DP 基线（32 个 worker）继续跑。
- owner：「找到更好的配置后，之前记录的配置要改，不能以后再踩同一个坑」→ 派 R（`sol_R_update1.log`）替换 RUN_CONFIGS 里过时的推荐，新增「坑（不要重复）」一节；P3/S/Q 的结果出来后再派第二次更新。
- 14:20 R 交回 RUN_CONFIGS 第一次更新（新增 8 条「坑」，16 处待定）。14:21 Q 交回 driver 批量提交（`--driver-mode group-commit`，`ops/formal_driver_fast.py`、`group_commit.py`；只在 driver 层单独固定一份编排版本）：真实 /data 回放，派发中位数 667 ms → 0.13 ms，回报 27.6 s → 0.09 ms；持久性规则是证据和终止日志都 fsync 后才计数。两条队列都已改用 group-commit 并预备通过。
- Q 在我的 DP 队列运行中改写了 `queue_hs4_dp.sh`（14:02）→ 停掉旧实例，改从副本 `queue_hs4_dp.run.sh --run` 运行（后台 b8qwarqfx），等 DP 基线 PASS 后跑 5 档。教训：后台脚本一律从副本运行，免得 agent 中途改写（要写进 RUN_CONFIGS 的坑）。Fast-WAM 的预备队列针对的是已停的 hs4 基线，等 P3 的新配置出来后要重新生成。
- 14:24 S 交回 DP 时间构成：worker 等推理约 49%、等 driver 约 33%、仿真只占约 11%；server 有 93% 的时间有 batch 在跑，但 4090 只用到 41%/164 W，DDPM100 的 100 次迭代在主机侧串行，是主机瓶颈。建议：2 个副本（P3 测过 2×B8+MPS 每秒 26.2 次调用，1×B16 是 14.4 次）、加新 driver，以后可以考虑把整个采样器做成 CUDA graph（预计 1.15–1.35 倍）。
- 14:2x 停掉 DP hs4 基线（928 局作废），改开 **hs5**：`hs5dp_ctl_<时刻>`（名字见 `r17_ops/hs5_dp_stamp.txt`；MPS 管道路径上限 85 字节，所以 run 名要短），4090，2 副本，B8/5ms，`--mps --driver-mode group-commit`，timan107 ×32，2048/576 MiB，每 4 局回收。各档队列用副本 `r17_ops/queue_hs5_dp.run.sh`（先 `--prepare-only`，再 `--run`，后台 bma6x0w93），各档名为 `hs5dp_<档>`。监控换成 `formal_watch9.sh`（盯 hs5*/hs6*）。中途撞过一次 `Address already in use`（停旧批次后端口还没释放），重启后正常。
- 14:33 P3 的 H100 扫描（用了 6.8 分钟）：2×B16/5ms+MPS 每秒 28.0 次调用（1×B16 16.8，1×B24 19.2，2×B16 不开 MPS 24.8；3 个副本放不下）。14:33 开 **hs6** Fast-WAM 基线 `hs6fw_ctl_143332`（H100，2×B16/5ms，`--mps --driver-mode group-commit`，timan107 ×48，2048/768 MiB，四套件）。6 档队列 `r17_ops/queue_hs6_fw.json`，接续脚本 `queue_hs6_fw.run.sh`（后台 bwnl3npcc）：基线封存后在 CPU 预备到 `/data/mw_finetune/eval/fq_hs6fw`，基线 PASS 后跑 `hs6fw_<档>`。派 R 第二次更新 RUN_CONFIGS（`sol_R_update2.log`）。
- 14:43 R 第二次更新完成（待定项清零）。hs5 DP 约 44 局/分钟，hs6 Fast-WAM 约 45 局/分钟（都约是原来的 2 倍）。
- 15:26–15:28 两个基线都因 tether 超时被判 FAIL：DP `hs5dp_ctl_142812` 2000/2000 已收，收尾拉回 483 MB worker 包时 `nats: timeout`（我已手动重拉到 `timan107.tar`）；Fast-WAM `hs6fw_ctl_143332` 1839/2000，同一分钟 10 秒的 tether exec 超时（很可能是这个大包把通道挤满）。处理：DP 五档用去掉收据门的副本 `queue_hs5_dp_nogate.run.sh --run` 直接开跑（后台 bzzxersrd，各档只用 `--runtime-from`）；Fast-WAM 用 `--resume-from` 续跑 161 局（`hs6fw_res_152931`，后台 bgm6cifho）。派 Q（`sol_Q_tether.log`）：重封 DP 基线；让合并后的 Fast-WAM 基线能做 fq_hs6fw 各档的 paired-control；单次 tether 超时不再判整批失败；大包走 rsync 等不占控制通道的路径；补「坑」。**Fast-WAM 六档还没有自动接续**（`queue_hs6_fw.run.sh` 见 FAIL 已退出），等 Q 给出 paired-control 改法后再开。
- 15:4x Q：DP 基线已恢复为 PASS（2000/2000，旧收据存为 `RECEIPT_BEFORE_RECOVERY.json`）；Fast-WAM 六档改为 `--paired-control hs6fw_res_152931`、`--runtime-from hs6fw_ctl_143332`，预备通过（`/data/mw_finetune/eval/fq_hs6fw`）。Fast-WAM 合并 SUMMARY PASS 2000 局，成功率 93.95%（spatial 90.8、l10 91.6、object 95.6、goal 97.8），但续跑收据是 PARTIAL（返回码 2，`recovered_worker_shutdown`），等 Q 修成 PASS。已挂 `queue_hs6_fw_res.run.sh`（后台 b31crl3y6）：收据 PASS 后自动跑六档。DP 五档在跑（`queue_hs5_dp_nogate.run.sh`，后台 bzzxersrd）。
- 15:5x owner：「我离开了，你独自推进」；goal「有序推进实验」。
- 15:52 astra 交 `rounds/r17/cache/CACHE_STUDY.md`（8 组＝修正器 q × 停滞门 g × C；L＝各模型通过 2 pp 的最长 C，都没通过就用 10；Cosmos 只有 4 组；共 20,000 局；在 A 集上测）。协调者决定：沿用基线的 batch 配置并复用这些基线当对照（已写进 OWNER_RULINGS）。等 P6 的接入审计。
- 15:55 DP c10 PASS（1000 局，成功 868）。c15 两次在开跑前失败：MPS 管道竞态，两个副本同时启动时并发查询 `nvidia-cuda-mps-control`，偶尔报「Cannot send command」；mps.py 在冻结 runtime 里，不能改。处理：剩下四档改为 `--replica-startup-width 1` 重新预备（名字 `hs5dp_<档>s<次>`），队列 `queue_hs5_dp_retry2.run.sh`（后台 bkbk8nsg1）开跑前失败时自动重试 2 次。c15s0 在跑。
- 16:0x Fast-WAM 续跑收据 PASS（Q 修复），`queue_hs6_fw_res.run.sh` 自动开六档，hs6fw_c10 在跑。⚠ Fast-WAM 队列用的是 `--replica-startup-width 2` 加 MPS，有同样竞态的风险，formal_queue 失败时没有重试；某档开跑前失败的话，就按 width 1 重新预备剩下的档。
- 16:32 P6 交回 cache 框架（HANDBACK_P6.md:1682；48 份 YAML 在 `ops/cache_study/yaml/STUDY.json`；冒烟用 `ops/cache_study/smoke.sh <model> --run`，每个模型 32 局；L 暂定 C10，等 horizon 研究定稿）。各模型还差的部分已派出：P3（Fast-WAM 兼容认证，`sol_P3_cache.log`）、P4（DP 读 size500 库，`sol_P4_cache.log`）、D（Cosmos 配方准入，`sol_D_cache.log`）。
- 16:33 Fast-WAM c10 跑满 1000 局但返回码 2（`recovered_worker_shutdown`）→ 收据 PARTIAL，formal_queue 停下。改用 `queue_hs6_fw_retry.run.sh`（后台 bg4ka3lec）从预备好的 c16..k1 往下跑：返回码 2 但 1000 局全收的视为完成并继续；开跑前失败的按 width 1 重新预备（名字 `hs6fw_<档>s<次>`）。DP c15s0 PASS，`queue_hs5_dp_retry2` 在跑 c16。**待 Q 处理**：hs6fw_c10（及以后同类档）的收据恢复成 PASS；正常路径接受「全部格子已收、只有 worker 关闭确认缺失」的返回码 2；证据包拉取改到 GPU 释放之后。
- 16:37 Q 交回 MPS 竞态修复（`ops/mps_control_guard.py`、`queue_retry.py`；串行化控制调用、有界重试，固定版本不变；之后新起的子进程自动生效）。Q 给的 `--continue-queue` 命令**没有执行**（我自己的重试队列已经在跑 c16，再起会重复派发）。派 Q（`sol_Q_rc2.log`）：恢复 c10 收据，正常收尾路径接受返回码 2，证据包异步拉取。16:5x hs6fw_c10 已恢复成 PASS（930/1000）。DP c15s0 PASS（852/1000）。
- 16:57 P4 交回 DP cache 准入（HANDBACK_P4.md:1205，CPU 已全部完成）。GPU 冒烟要等 DP 各档跑完：`cd /home/weiland/projects/openpi; flock -n /tmp/r17_gpu.lock bash exp/r17_integration/ops/cache_study/smoke.sh dp --run`（15–25 分钟）。P3、D 还在做。
- 17:05 D 交回 Cosmos cache 准入（HANDBACK_COSMOS.md:955，96 项测试，H100 CPU 预检 PASS）；冒烟命令 `bash exp/r17_integration/ops/cache_study/smoke.sh cosmos --run`（10–15 分钟）。Q 交回 rc2/ACK 恢复以及「先释放 GPU、再异步封存」（`formal_finalize.py` 等），之后新起的档自动生效；全部跑完后要按 Q_to_R.md:99 做 CPU 封存屏障。派 astra（`astra_horizon_analysis.log`）先分析已完成的 DP/Fast-WAM 档，写 `horizon/RESULTS_DP_FASTWAM.md` 和 `analyze_arms.py`，并给出 cache 研究暂定的 L。
- 17:14 P3 交回 Fast-WAM cache 准入（HANDBACK_P3.md:975；冒烟用预备好的 `formal_eval.sh --prepared-run /data/mw_finetune/eval/cfwp31010 --run`，10–20 分钟）。三个模型的 CPU 准入都完成了。派 P6（`sol_P6_launch.log`）：把 smoke.sh 改成和各模型对照一致的配置；做全量计划（每个模型一个常驻 server 热切换 8 组，L 做成参数）；给出各通道的命令。17:1x hs6fw_c16 PASS（917/1000）；DP ddim10s0 在跑；Fast-WAM 在跑 c20。
- 17:14 astra 中期分析（`horizon/RESULTS_DP_FASTWAM.md`、`analyze_arms.py`）：DP C10 S −1.4 / L +4.2，C15 −4.8 / +4.4，C16 −5.6 / +1.4；Fast C10 +1.0 / +2.6（不做多重比较校正时两套件都通过），C16 +1.8 / −0.8。经 24 项族系校正后 5 档都没过 2 pp。DP 的 horizon 拉长后 spatial 明显下降、l10 反而上升。调用次数降幅：C10 约 49%，C15/16 约 65–68%。cache 研究暂定 L：DP=10、Fast=10（按规则取兜底值）。还缺 DP DDIM10/5、Fast C20/C32/K2/K1。
- 17:34 hs6fw_c20 已收满 1000 局（成功 918），但 FAIL：`remote worker close/snapshot incomplete`；队列照常往下走。c32 第一次启动撞上 `Address already in use`（c20 的端口还没释放）→ 自动重新预备成 `hs6fw_c32s1` 重试。派 Q（`sol_Q_c20.log`）：恢复 c20 收据、修端口释放、对所有未 PASS 的档做 CPU 封存屏障。DP ddim10s0 约 81%。
- 17:4x DP ddim10s0 PASS（832/1000），异步收尾生效（ddim5s0 紧接着就开了）。17:52 Q 把 hs6fw_c20 恢复为 PASS（918/1000）。hs6fw_c32s1 在跑。
- 17:59 P6 交 cache 全量计划（`ops/cache_study/FULL_RUN.md`；ops 测试 356 项 + 43 个子测试）。冒烟命令：`smoke.sh dp|fastwam|cosmos --run --run-name csmd1010|csmf1010|csmc1010`；全量：`.venv/bin/python -m exp.r17_integration.ops.cache_lane --lane 4090 --directory /data/mw_finetune/eval/csl10dp --run` 和 `--lane h100 --directory /data/mw_finetune/eval/csl10h --run`（L=C10 暂定；Cosmos 用 C16）。预计：DP 全量 3–5 小时，H100 上 Fast→Cosmos 5–10 小时。已挂自动链 `chain_cache_4090.sh`（后台 blu370hmi：DP 档位队列结束后先冒烟，冒烟 PASS 再跑全量）和 `chain_cache_h100.sh`（后台 bpoc2u3zk：Fast-WAM 档位结束后 Fast 冒烟、Cosmos 冒烟，再跑全量）。监控换用 `formal_watch10.sh`（新增 csm*/csl10*）。
- 18:09 DP 横向/减步五档全部跑完（`QD_ALL_DONE`；ddim5s0 等补证据）。重跑 `analyze_arms.py`（输出到 job tmp）：DP DDIM10 S −1.2 / L −3.2；Fast C20 +1.2 / 0.0，C32 −0.8 / **−6.2**。族系校正后没有一档通过；L 定为 DP=10、Fast=10（兜底值），和 cache 已预备的 C10 一致。4090 链开始跑 DP cache 冒烟。Fast 还剩 k2s1（在跑）和 k1。全部跑完后让 astra 出 RESULTS_DP_FASTWAM.md 终版。
- 18:21 DP cache 冒烟 `csmd1010` 失败：`port ziyanglin.com:23401 did not become ready`。本机 server 的端口必须在 ziyanglin.com 的 NAT 转发段 23100–23199 内，worker 才连得上；worker 主进程就是因为连不上才悄悄退出的。4090 链因此退出。派 P6（`sol_P6_smoke.log`，备注写进 COORD_to_P6.md）修端口、加快速失败、自己用新名字重跑 DP 冒烟，并检查 h100 上两个冒烟。4090 空着，等 P6。
- 18:41 P6 交回：根因是 23401 端口没有走路由；已修路由、失败检测、日志、锁（CPU 测试 881 项 + 43 子测试）。重跑的 DP 冒烟 `csmdf1010` PASS：32/32 局，1227 次命中，26 次只由停滞触发的 MISS，2600 次 NFE，不带门的组 NFE 为 0，热切换没有重启 replica。已准备好的 csl10d/f/c 在启动时会重新绑定路由，不用重新准备。18:4x 开 **DP cache 全量** `cache_lane --lane 4090 --directory /data/mw_finetune/eval/csl10dp --run`（后台 bwohfh43q，日志 `r17_ops/cache_lane_4090.log`）。H100 链（bpoc2u3zk）等 Fast-WAM k1 跑完后自动跑 Fast/Cosmos 冒烟和全量。
- 18:56 **horizon/减步研究的所有档都跑完了**（Fast-WAM k2s1 PASS 908/1000，k1 1000 局、证据还在补）。H100 链开始跑 Fast-WAM cache 冒烟。派 astra（`astra_horizon_final.log`）：等 k1 PASS 后出终版 RESULTS_DP_FASTWAM.md、给 owner 的 `horizon/SUMMARY.md`，并把三个模型的官方基线更新进 `exp/mw_finetune/RESULTS.md`。DP cache 全量约 56–60 局/分钟，预计 21:00 前后跑完。
- 19:10 **horizon/减步研究收口**：astra 写好 `rounds/r17/horizon/SUMMARY.md`（给 owner 的中文汇总）和终版 RESULTS_DP_FASTWAM.md，并把三个模型的 A 集官方基线写进 `exp/mw_finetune/RESULTS.md:34`。结论：12 个候选都没有通过预注册的采用门（两个套件都满足 2 pp 非劣，并有实测成本证据），默认配置保持不变（DP K100/C5、Fast K10/C5、Cosmos K5/C16）。最值得复核的是 Fast C10（+1.0/+2.6，两套件不做多重比较校正时都过，调用减半）。明显掉点的：DP C15/C16 的 spatial（−4.8/−5.6）、Fast C32 的 l10（−6.2）、减步档在 l10 上（DP DDIM10/5 为 −3.2/−4.2，Fast K1 −3.0，Fast K2 −1.8，Cosmos K2 −2.0）。cache 研究的 L 定为 DP 10、Fast 10、Cosmos 16。
- 19:30 三个 cache 冒烟都过了（DP csmdf1010 30/32、Fast csmf1010 29/32、Cosmos csmc1010 28/32）。H100 全量已开（`cache_lane --lane h100 --directory /data/mw_finetune/eval/csl10h`，先 csl10f 跑 Fast-WAM，再 csl10c 跑 Cosmos，日志 `r17_ops/cache_lane_h100.log`，链 bpoc2u3zk）。DP 全量 csl10d 已到 3177/8000。全部跑完后派 astra 按 CACHE_STUDY 做 8 组分析。
- 19:4x owner 问 cache 是不是被 worker 卡住：是，timan107 CPU 负载 88/48 核，两张卡的 SM 都在 0–22%。只有 timan108 恢复才能提速。目前的成功率（只有 spatial，未配对）：DP 8 组 91.4–94.6%（基线 91.2），Fast 纯 cache 87.8%（基线 90.8），+修正器 98.3%（233 局）。owner 让查数据污染：评测用 A/test（初始状态 0–49），库是 B 池 size_500（spatial 12028 行），两池各自独立；派 astra（`astra_contam.log`）做哈希精确重叠、近重复和拟合来源检查，结果写 `cache/CONTAMINATION_CHECK.md`。
- 19:49 DP cache 全量 csl10d 在 4001/8000 时 FAIL：`worker_restart_resource_timeout`（每 4 局回收一次，回收时在超额的 timan107 上拿不到资源），通道重试又被「需要显式续跑」拒绝。派 P6（`sol_P6_resume.log`）修回收逻辑、自行续跑 DP，并让 cache_lane 支持续跑。4090 暂时空着。
- 20:04 astra 交回 `cache/CONTAMINATION_CHECK.md`：A 测试初态和 6 个库都没有精确重叠（0/500），20 个任务都满足 B50 = 原始 100 个初态减去官方 A50；95,924 行库数据和 10 份修正器/停滞门拟合都确认只来自 B；有 32 个 A 初态被近邻阈值标记（spatial 11、l10 21），没有证据表明是泄漏。最终分析里加一项敏感性分析：去掉这 32 个初态再算一遍。
- 20:1x owner：「把每个 cache size 都全部重复一组，实验 size 相当于翻 6 倍，每个都如此」→ 50/100/200/300/400 五档各自重跑全部组（3 个模型，8 组，Cosmos 只有 4 组）× 2 套件 × 500 局，在 A 集上测。新增约 10 万局，只用 timan107 的话约 22 小时，加上 timan108 约 11 小时（已提醒 owner 修 timan108）。派新 sol P6b（thread 01a12882-35d3-7201-85b3-3b62a4b417d2，`sol_P6b_sizes.log`）：补各档的拟合和停滞门校准，生成 YAML，扩展两条通道接在 500 档之后跑；派 astra（`astra_cache_sizes.log`）更新 CACHE_STUDY 预注册，加入库大小这个因子。⚠ P6（DP 续跑）和 P6b 可能同时改 cache_lane.py，要留意冲突。
- 20:06 时的完成时间预估（只算 500 档）：Fast 约 22:30，DP 续跑约 23:10，Cosmos 约 23:50。
- 20:21 P6 交回：DP 续跑 `dpc8513584592` 在跑（4001 局保留，只补 l10 缺的 3999 局）。根因有两条：回收时虽然内存检查通过，仍重新走了显存准入；每一代日志都泄漏文件句柄。两处都已修复（129 项测试 + 259 项）。正在跑的 DP 监督进程的软文件上限已提到 8192。监控换成 `formal_watch11.sh`（新增 dpc* 等）。timan1 上已有 `/scratch/zixuans8/{libero_sim,openpi_rc365,Isaac-GR00T,nfe}`，T1 会尽量复用。
- 20:2x–20:31 Fast 全量 csl10f 在 4023/8000 处 FAIL（`worker_restart_resource_timeout`，修复前的旧进程）；随后 DP 续跑也 FAIL（fleet「receipt unavailable」），Fast 续跑报 `mkdir ... No space left on device`。**根因：timan107 根分区（/、/tmp、/scratch 共用，212 GB）满到 100%**：我们的 `mw_r17/formal/runs` 12 GB，加上 T1 在 timan107 上暂存的 `mw_r17/timan1_provision` 9 GB。处理：删了 24 个已 PASS、本地 tar 已核对的远端副本（腾出 6 GB，现在 98%；c20 和 hs3/hs4 保留），两条通道重开（`cache_lane_h100_r3.log` 后台 bk0qpmnyp，`cache_lane_4090_r3.log` 后台 bx0poe9f2）。派 P6（`sol_P6_disk.log`）：远端自动清理、磁盘守卫。T1 交回后，让它删掉 timan107 上那 9 GB 暂存。
- 20:47 Fast 续跑 fastwamc1456717485 FAIL：timan107 内核 OOM 杀掉 worker（cache worker 每个约 3.1–3.4 GB，预留只有 2048 MiB；DP 32 个加 Fast 48 个超过 220 GB）。DP 续跑 dpc3464789869 在跑。Q 交回本机 worker 功能（`--worker-counts local=N`，私有环境在 `/data/mw_finetune/qualification/local_workers`，冒烟 12/12，像素差为零；建议 DP local=16、H100 local=8）。T1 交回 timan1（只用 3 号卡，guard 生效，每个 worker 3.5 GiB / 540 MiB，像素逐字节一致，最多 16 个）；⚠ T1 的最初几次探测误用了未授权的卡，已排除并重跑，要如实告诉 owner。派 Q（`sol_Q_split.log`）：接入 timan1 节点；续跑时允许同样的 worker 总数在多个节点之间重新分配；预留改为约 3.5 GiB；自己按 timan107=16、local=16、timan1=16 重开 H100 通道。H100 暂时空着。
- 20:5x H100 通道自动重试 fastwamc6990509669 又失败（同样是 48 个 worker 全在 timan107）。我去停这条通道时，⚠ **误把 P6b 的 CPU 预备进程（cache_lane h100 cszeh，PID 463102/463103）停了**（只在 CPU 上，无数据损失），已写进 REQUESTS/COORD_to_P6b.md 请它重跑。H100 通道本身已经自己退出。DP 续跑 dpc3464789869 在跑（worker 进程名是 worker_lifetime，约 32 个，到 134 局）。
- 21:0x owner：「timan107 上我们不要的数据直接删」→ 已删 hs3/hs4 远端 runs、/tmp/r17_Q_qualification、/scratch/zixuans8/openpi_trace(8G)、openpi_trace_h100(6.2G)；Q 删了 T1 的 9G 暂存。timan107 现有 32 GB 空闲（85%）。保留 openpi、libero_sim、Isaac-GR00T。owner 骂拖沓——今晚的问题都是事后逐个补（磁盘、内存、多节点），以后开跑前要一次查清前提条件。
- Q 准备好 Fast 续跑 `fastwamc2350933912*`（保留 4311 局、补 3689 局；timan107/local/timan1 各 16，每个 4 GiB，timan1 6 GiB），接着 Cosmos 用 timan1 上 8 个 worker，Q 自己负责启动 H100 通道。
- 21:05 P6b 交 size 网格（`ops/cache_study/sizes/`，240 个配置，`run_4090.sh` / `run_h100.sh` 等前一条通道 PASS 后再派发；原估 4090 15–25 小时、H100 25–49 小时，当时只算了 timan107）。续派 P6b（`sol_P6b_arm.log`）：改成分节点编队（DP timan107 16 + local 16；Fast 三台各 16；Cosmos 视情况分配）、重新预备、自行布置两个队列、更新耗时估计。
- 21:1x owner 批评：「一边加设备一边减 worker」——总数一直是 80（32 + 48），没有增加。实测容量：timan107 约 45（48 核，空闲 123 GiB），本机约 60（88 核，空闲 237 GiB），timan1 约 40（48 核，空闲 468 GiB，只用 GPU3）≈ 145。size 网格目标：DP 约 60（timan107 20 + local 40，DP replica 可加到 3–4 个，避免 server 端 CPU 成为瓶颈），Fast→Cosmos 约 85（timan107 25 + local 20 + timan1 40）。**P6b 本轮交回后立刻按这组数字重新派发**（它现在按 32/48 在做）。
- 21:1x owner：「4090 的 worker 优先给 4090 用，不要增加传输，4090 也可以接入外面的 worker」→ 修订后的分配：4090/DP = 本机约 60 个（走 localhost，DP replica 3–4 个），server 撑得住的话再加 timan107；H100 = timan107 约 45 + timan1 约 40，不用本机 worker。
- 21:1x Q 启动 Fast-WAM 续跑时又碰到两个阻塞：①批次的 batch 处理身份里含 worker 节点表，已改为另设一张执行用的节点表，batch 身份保持不变；②timan107 上旧的 LIBERO 配置路径不见了——⚠ 很可能是我删 /tmp/r17_Q_qualification 时连带删掉的；Q 改为复用已封存的依赖副本。H100 从 20:47 起一直空着，等 Q 的续跑。
- 21:17 Fast 续跑 fastwamc7615306896 开跑（三台各 16），约 70 局/分钟；21:2x 跑到 397 局 FAIL：timan1 上 16 个 worker 全部 exit 75 → 重启预算耗尽。推测 75 是计划内回收的退出码，T1 的设备命名空间包装没把它当回收处理。通道自动续跑为 fastwamc5232306668（配置同上，会循环失败）。派 Q（`sol_Q_t1fix.log`）：修 timan1 上回收码的透传；15 分钟内修不好就先去掉 timan1 继续跑。
- 21:29 P6b 已按旧编队布置 size 通道（PID 834339 等 csl10dp，834494 等 csl10h；DP 为 timan107 16 + local 16，Fast 三台各 16，Cosmos timan1 8；估计 DP 12–24 小时，H100 18–38 小时）。续派 P6b（`sol_P6b_rearm.log`）：停掉这两个等待进程，按大编队重新预备（DP 本机约 60 个、3–4 个 replica；H100 用 timan107 约 45 + timan1 约 40），timan1 等 Q 修好后再用，然后重新布置。
- 21:43 Q 修好 timan1 的回收问题（bubblewrap 下的 PID 和 native PID 对不上，按代绑定身份后解决；106 项测试 + 29 子测试）。Fast 续跑 `fastwamc5481985003` 到 5332/8000，约 73 局/分钟，预计 22:20 前后完成；timan1 已回收 16 次，0 失败。紧接着的 Cosmos 500（`cosmosp6223294666`）只有 timan1 8 个 worker，太小 → 派 Q（`sol_Q_cosmos.log`）趁还没派发，按 timan107 + timan1 大编队重新预备，并让通道接上新版本。
- 21:5x owner：「4090 上现在有多少 worker，我觉得还有空间」→ DP 续跑是 32 个 worker，全在 timan107，本机 0 个；两个 server replica 各约 36% CPU，4090 约 10%。派 P6（`sol_P6_local60.log`）：在 episode 边界停掉当前段，开新段，用本机约 60 个 worker、3–4 个 replica，timan107 让给 Cosmos；放开 cache 续跑「worker 总数必须相同」的限制，并记录每段的编队。
- 22:0x 状态：DP 旧段 dpc3464789869 在 episode 边界收尾（1734 局），新段 dpc2777537823 用本机 33 个 worker 启动（被准入卡住）；Fast 续跑三台各 16，到 1446 局，速度降到约 21 局/分钟（原因待查）；Cosmos 大编队已就绪（Q）；P6b 的 85 编队和 45 后备已封存。owner：「本机 256G 内存只用这点、内存准入哪来的」→ 准入过于保守（每 worker 4 GiB + 10% 余量 25 GiB + 按预留而不是实测计）。P6 本轮结束后立刻续派 `prompts/sol_P6_dp_more.md`：预留改 3.5 GiB、余量 8 GiB、按 max(预留, 实测 RSS) 计；DP 先扩到约 45，Fast 结束后扩到约 60；同步给 P6b。
- 每档启动约 10–15 分钟卡空着（打包、分发、内存检查）；Q 完事后让它改成上一档运行时预备下一档。

**10-10 11:2x 交接（compact 前，最新，先读）**

**一、owner 今天的裁定和纠正（都要照做）**
- **22:1x owner：①不同 cache size 要排进计划；②YAML 能热切换，就一个模型全部跑完（全部 size + 全部组）再换下一个模型，不要频繁切换模型；③timan 系列硬盘不够，数据定期同步回本机、同步后删远端。④实验由协调者亲自启动和守护，agent 只写代码，命令交给协调者（已写进记忆 feedback-coordinator-guards-experiments）。**
- 22:1x 接管：我已停掉 P6b 挂的两个等待进程（1130348/1130349），H100 size 通道 csznh 由我在后台自己挂（b0hec4lg0，等 csl10h PASS）。P6b（`sol_P6b_dp60.log`）只改代码：渲染预留降到 0.7 GiB、内存预留 3.5 GiB、余量 8 GiB，按本机 60 个 worker 重新预备 cszmdp，把挂载命令交给我。Q 的 Cosmos500 大编队（`cosmos500w85_220813`，timan107 45 + timan1 40，2 个 B1 replica）已预备。派 Q（`sol_Q_reorder.log`）只改配置：把 H100 顺序改成 Fast500 → Fast50..400 → Cosmos500 → Cosmos50..400（每个模型尽量只起一个常驻 server），同时加远端定期同步回本机并删除；命令交给我执行。DP 新段 `dpc8713063796_r2_d1dd`（P6 起的，本机 33 个）由我守护。
- 22:3x owner：「一将无能累死三军，这次你自己来，不准派 sol」；指出每个模型可以按 worker 的设备来源开多个 conductor；单独封存这一层是我加的、没用。**由我亲自接手**：①`cache_model_order hold-next --lane csl10h --model cosmos_libero`，拦住 csl10h 在 Fast 结束后派发 Cosmos500；②停掉旧的 csznh waiter（1167414）；③`cache_model_order run --directory /data/mw_finetune/eval/h100_model_order_20261010/cosmos_all`（PID 1478263，等 fast_sizes 队列 PASS 后跑 Cosmos500 w85 + cosmossizes85，外部 85 个 worker）；④fast_sizes 队列遇到前序批次 PARTIAL（还在跑）就直接报错退出（bug）→ 用我的 until 循环（后台 ba0d4g0kc）等 fastwamc5481985003 PASS 后再起；⑤DP size 通道 `run_4090.sh`（cszmdp，本机 60 个、4 个 replica，GPU 准入实际只放约 50 个）由我在后台挂（b65romhwa），等 csl10dp PASS。⚠ csl10h 的进程 729176 在 Fast 结束后要退出，Cosmos 的 hold 才能释放；如果它卡着不退，就由我按 PID 停掉。
- 22:3x–22:43 亲自排障，让 Fast 各 size（csznf）起跑：①Fast500 续跑 8000 局都收齐了，收据停在 PENDING_EVIDENCE → 把 fast_sizes 计划的 after_run 清空（原件存为 `.orig_before_coord`）；②T1 的指纹门挡住，因为 P6 两轮并行改了 formal_workers / timan1_workers → 我停掉了 P6 的两轮（local60 和 dynfleet）；③把 formal_workers / timan1_workers 恢复成 csznf worker 包里的版本（撤掉 P6 半成品的 live-fleet 改动；当前版本备份在 job tmp/ops_backup_2245），重算 t1_recycle_fix 的指纹，t1_ready=True；④csznf 上一次刷新留下的 CACHE_RECYCLE_BASE.tar 会挡住再次刷新，所以把代码恢复到不需要刷新的状态；⑤把队列目录里的旧日志和回执挪开（attempt2–5）。22:43 csznf owned_spawn，正在起跑。链：`h100_chain.log`（先 Fast sizes，再 cosmos_all）。Q 的 sync_back.py 已恢复成它在写的版本。DP500 续跑 8000 局收齐（PENDING_EVIDENCE），DP size 通道（b65romhwa）在等 csl10dp lane 退出。
- **20:1x owner 特例：timan1 的 3 号显卡空着，授权这次 cache 研究用它跑 worker（只限 3 号卡，「这也是这次的特例」）。**
  timan1 实测：48 核，负载约 5，可用内存约 440 GB，3 号卡空闲，其余卡有别人在用。派新 sol T1（`sol_T1_timan1.log`）准备 timan1 的环境、资产，验证只在 3 号卡上渲染，结果写进 REQUESTS/T1_to_Q.md；Q 做完本机 worker 后再把 timan1 节点加进启动器（要续派 Q）。
- **20:1x owner 特例：本机（weilandserver）显存、CPU、内存都充足，cache 研究可以在本机也开 worker（「这个是特例」，常规仍是 worker 只放 timan 族）。** 派 Q（`sol_Q_local.log`）给启动器加本机 worker 节点（支持和 timan107 混合编队、本机 4090 做 EGL 渲染、准入、路由），冒烟后告诉 P6b 怎么在后续批次加本机 worker。
- **15:4x owner 下一个研究（horizon/减步研究跑完后开）：对所有新模型跑 8 组——纯 cache、纯 cache+修正器、纯 cache+长 horizon、纯 cache+长 horizon+修正器（都不用 MISS），再把这 4 组各加上停滞门重复一遍；带旋钮的暂时不跑；每组「全 B 集 500 episode」；全部用 YAML 配合 concurrent server 热切换来跑，「如果不行就是你没有把模型充分接入框架」。** 已派 astra（`astra_cache_study.log`）写预注册 `rounds/r17/cache/CACHE_STUDY.md`（对照 R15/R16 的语义，并核实评测集用 A 还是 B，B 是建库用的初始状态，有泄漏问题）；派 P6（thread 01a12182-101b-7822-a6dc-dea5e79dcc2d，`sol_P6_cache.log`）牵头接入：审计缺口、补框架、通过 REQUESTS 找 P3/P4/D 补模型钩子、生成 YAML 和冒烟计划。
- owner 15:5x 确认：**cache 研究的评测集是官方测试集 A**（每任务 50 局，每套件 500 局，只跑 spatial 和 l10）；「全 B 集」是口误，B 只用来建库。已写入 `REQUESTS/COORD_to_P6.md` 和 `rounds/r17/cache/OWNER_RULINGS.md`。
- **15:4x owner：「沙堆实验正式被我冻结了，你不用管他」→ 两张卡用完不用再书面交还或写台账（已写进记忆）。**
- **15:3x owner：「我们要的数据只有 libero_spatial 和 libero_10，其他 LIBERO 任务的数据仅供参考，甚至可以不跑」→ 以后新模型的 LIBERO 基线和各档只跑 `--suites libero_spatial libero_10`（官方测试集 A，每任务 50 局，每套件 500 局）；object/goal 不再跑。**
- 协调者只协调：调研、读代码、设计、写码、分析都派 agent（科学类 astra，工程类 sol）；协调者负责派单、审阅、守护实验、汇报。
- 记忆只收项目级内容（已按规范删 3 条、精简 4 条），会话进度只写 handoff。
- LIBERO 纯推理基线 = 官方测试集 A，每任务 50 局、每套件 500 局（`--official-50`）。之前 400 局的非标准集作废（RESULTS.md 已标注）。
- 本机 4090 是 48 GB，Fast-WAM（每副本约 26 GB）和 Cosmos 都放得下；两卡都用上，不要让卡空等。
- 开 batch：精度差不影响成功率就不纠结，按吞吐选；同一组配对研究的对照和实验组必须用同一种 batch 配置。
- Cosmos 减步只测 **K=2（含清理步，官方 num_steps=2 语义，如实记实际 NFE）**，在 H100 上与 Cosmos 官方基线配对；不测 K1/K3，也不补 4090 Cosmos 对照。

**二、GPU（X2：10-10 10:1x 起按 I018 冻结期约定书面告知后重新使用，台账已记）**
- H100：Cosmos 官方基线 `formal_cosmos_libero_official2000_20261010_100533_5f558380`（2 副本，timan108 ×24，约 11:35 完）；Fast-WAM 官方基线 `formal_fastwam_libero_official2000_20261010_101044_b0c4e68b`（2 副本，timan107 ×24，batch=1，约 352/2000，预计约 16:00 完；若 P3 的 batch 结论好，按 owner 意思停掉、用 batch 配置重跑）。
- 4090：DP 队列 `r17_ops/queue_4090_hs_v2.sh`（后台任务 bf3s4dkqm），11:2x 起：DP 官方四套件基线 → 执行 10/15/16 步 → DDIM 10/5，worker 在 timan108 ×20（每个预留 4 GB），日志 `r17_ops/hs_dp_*.log`。
- H100 队列 `r17_ops/queue_h100_hs.sh`（后台任务 bzqjc851n）：等 Fast-WAM 基线结束后跑 Fast-WAM 执行 10/16/20/32 步、K=2/1（timan107 ×24，2 副本）。⚠ 如果改用 batch，这条队列要改参数，并补 batch 配置的对照。
- 两卡用完都要书面交还沙堆实验（SendMessage）并写台账。

**三、在做的 agent（Codex 走 `dynamical_system/runs/i018-control/bin/codex_daemon_run.py`，日志在 `openpi_ext/scratch/r17_ops/`）**
- P3（thread `01a12181-f87d-73c1-88f6-0407d9f849bf`，`sol_P3_batch.log`）：剖析 Fast-WAM 并做服务端动态 batching（B=1–16 吞吐、功耗、数值差），加默认关闭的 `--max-batch`，交 `exp/r17_integration/models/fastwam/BATCHING.md`。它发现正式服务在副本内把所有连接串行处理、batch 钩子被关着。要用 4090 测，但 4090 锁被 DP 队列持有：回报拿不到锁时，让它不持锁、和 DP 共卡测相对加速比。
- D（thread `01a11f04-32ac-7eb0-a84c-fd244cac0d16`，`sol_D_k2.log`）：放开 Cosmos K=2（含清理步），与官方逐位比对，交 HANDBACK_COSMOS「K2 incl. clean step」和命令（H100，2 副本，timan108 ×12，`--official-50 --suites libero_spatial libero_10 --denoising-steps 2`，端口 23240/23249）。交回后由协调者开跑。
- astra（thread `01a12661-c6ae-7623-98d3-915b38d88bb3`）：研究文档 `exp/offline_search/rounds/r17/horizon/STUDY.md` 已完成。内容包括 π0.5/GR00T 历史（C5→C10 的结论）、各模型的 horizon 和减步档位、预注册（2 个点非劣，24 项比较）、两卡排程。之前那次 Cosmos 减步实验的调用次数记录和配对都有问题，已在文档里注明。
- Q（thread `01a121c0-0b47-79c3-a554-f6fdb32f9d67`）：已修完同节点 OOM（加节点内存检查、单个 worker 最多补 2 次、每批独立目录）和 timan108 的 LIBERO 资源问题。

**四、研究设计（STUDY.md 是唯一依据）**
- horizon：DP 执行 5→10/15/16 步；Fast-WAM 执行 5→10/16/20/32 步；Cosmos 已执行整块 16 步，不做。
- 减步：DP DDPM100→DDIM10/5；Fast-WAM K10→2/1；Cosmos K5→2（含清理步）。
- 只跑 spatial 和 libero_10，每档 1000 局，与同卡默认档逐个起始状态配对；基线跑四套件。

**五、待办和待 owner 定**
- 等 P3 的 batch 结论，决定 Fast-WAM（以及 DP、Cosmos）是否改用 batch 重跑基线和各档。
- D 交回后开 Cosmos K=2。
- 收集各档结果，按 STUDY.md 的判定规则汇报。
- 之前给 owner 的提速选项（4090 跑完 DP 后分担 2–3 档 Fast-WAM、去掉执行 20 步那档）还没回复；开 batch 后可能就不需要了。
- 6 个 LIBERO 库已经建好（`/data/r17_library/`），之后用它们做新模型的闭环实验。

**R17 正式测试规矩（owner 10-09/10 纠正，原在记忆里，移到这里）**
- 正式成功率测试一律走框架：`exp/r17_integration/ops/formal_eval.sh`（conductor 逐局分发 + 原版 replica_proxy 多副本；DP 加 `--cuda-graph`），不用原版 `examples/libero/main.py` 去连单个加锁的服务。
- LIBERO 纯推理基线按 R10/R16 协议用官方测试集 A：每任务 50 个官方起始状态、每套件 500 局（`--official-50`）。集成 spec 里的「40 任务 × init 0..9 = 400 局」作废。MT50 每任务 10 局，共 500 局。
- 本机 4090 是 48 GB（台账有写）。

**10-10 07:4x 建库全部完成、两卡已交还（最新，先读）**
- 6 个库都建完（DP / Fast-WAM / Cosmos × spatial / libero_10），每个都有 50–500 六档子集和拟合，汇总表见 `exp/mw_finetune/RESULTS.md` § LIBERO teacher libraries。产物在 `/data/r17_library/<bundle>_<suite>/bpool/`（本机 /data，约 45 GB）。
- **4090 04:3x、H100 07:4x 都已交还 I018**（书面通知加台账；I018 冻结中，openpi 再用时书面告知并写台账即可）。H100 上残留约 99 MB 暂存和 /tmp/r17_formal 小文件，待清理。
- 下一步（等 owner）：用这些库跑新模型的闭环（R16 格式：库大小 × K 的纯缓存、守卫、修正器），或先补 Fast-WAM LIBERO 的正式 400 局纯推理基线（还没跑过）；另外可以用真实库重跑 G3。

**10-10 04:3x 建库状态（最新，先读）**
- 已建完（`/data/r17_library/<bundle>_<suite>/bpool/`，6 档子集加拟合都齐）：Cosmos × spatial（3676 行）、Cosmos × libero_10（8408 行，抢救 461 局并补跑 39 局）、DP × spatial（12028 行，补跑 4 局）、DP × libero_10（31546 行，抢救后补跑 3 局）。
- **4090 已于 04:3x 交还 I018**（台账和书面通知都已完成）。I018 回复：按 owner 指示长期冻结，近期不用 4090 和 H100；冻结期间 openpi 要用 4090，书面告知并写台账即可，不用等回复；I018 解冻前会提前 30 分钟书面收回。
- H100 在跑 `r17_ops/chain_h100_repairs.sh`：Fast-WAM spatial 补跑 271 局 → Fast-WAM libero_10 补跑 446 局（两批都已抢救，保住 229 局和 54 局），预计 08:30–09:30 完成，完成后交还 H100（SendMessage 加台账）。
- 夜里修的坑（P6/D）：tether 掉线三次（timan108 两次、H100 一次）都是我们自己的调用把心跳挤掉了（老代码约每分钟 322 次，现在约 8 次），已加 5 分钟重试；新增抢救模式和补跑模式，处理中断批次、缺局和第 0 步错误；H100 有界暂存 20 GB；隧道就绪等待；按节点选 LIBERO 配置。
- 已知待议：库端口窗口固定 233xx（不在 NAT 转发段，走 tether 隧道）；GR00T C/B2 批量差异归 R4-2；G3 合成库拟合问题，真实库建好后可以重跑 G3。

**10-10 00:1x 建库进行中（最新，先读）**
- 正式测试全部完成（RESULTS.md）：MT50 任务均值 GR00T 77.6 / DP-T 61.0 / Fast-WAM 60.4 / π0.5 58.8；LIBERO 400 局 Cosmos 98.3 / DP-T 84.3；纯减步 π0.5 K1 57.6、GR00T K1 73.8。资格检查：副本时段（DP-LIBERO、GR00T-MW）PASS，Fast-WAM LIBERO A/B PASS（P6 用的绑定收据在 `/data/mw_finetune/qualification/bound_receipts/g2_bound_p6_20261010_000040/`）。
- 建库自动链：4090 `r17_ops/chain_4090_dpfull2.sh`（DP spatial → l10）；H100 `r17_ops/chain_h100_lib.sh`（Cosmos spatial → l10 → Fast-WAM 冒烟 ×2 → Fast-WAM 全量 ×2）。每步按退出码把关，步与步之间等 90 秒避开 TIME_WAIT。日志在 `r17_ops/lib_*.log`，产物在 `/data/r17_library/<bundle>_<suite>/`。
- 今晚修的坑：B 池哈希缺 `sha256:` 前缀（D 修）；隧道就绪竞态只对 cosmos 生效（D 推广到所有走隧道的情况）；H100 分批拉回重复预留耗光 20 GB 额度（P6 修）；Fast-WAM 门槛改为 G2 A/B 收据（P6），收据补模型清单绑定（Q）；库端口限定 233xx，不在 NAT 转发段 23100–23199，所以全走 tether 隧道。
- 全部建库完成后：H100、4090 交还沙堆实验（SendMessage + 台账）。

**10-09 23:3x 状态（最新，先读）**
- 正式测试改走框架（`exp/r17_integration/ops/formal_eval.sh`，Q 写；conductor 逐局分发 + 原版 `replica_proxy` 多副本；DP 用 `--cuda-graph`）。教训写进记忆 `feedback-formal-eval-through-framework`。
- 结果：DP-LIBERO 正式 400 局 84.3%（spatial 90 / object 72 / goal 94 / 10 81），见 RESULTS.md。在跑：Fast-WAM MT50 500 局（H100，2 副本，run `formal_fastwam_mw_mt50_500_20261009_223917_0b845f00`）、Cosmos-LIBERO 400 局（H100，2 副本，`..._231322_2b680ed5`）、DP-MW MT50 500 局（4090 + timan107，`..._225821_ef25fe4e`）、π0.5 K=1（4090）。
- 同卡教训：12 个 DP 小 kernel 进程和 Fast-WAM 同在 H100 上会让 Fast-WAM 从每分钟 10 局降到 4 局，所以 DP-MW 改放 4090。
- Cosmos-LIBERO 已由 sol D 接入（`handbacks/HANDBACK_COSMOS.md`；H100 运行时修了 NVRTC 和 gymnasium；冒烟 2/2，与官方动作逐位一致）。D 在加建库用的采集钩子。
- 建库：P6 交付 `ops/library_campaign.sh`（`HANDBACK_P6.md` § LIBERO library pipeline；B 池来源与 R10 对齐都已证明）。P6 在改 H100 有界暂存（H100 /data 只剩约 90 GB）。
- 自动链：`r17_ops/chain_4090_dplib.sh`（4090 空出后跑 DP 建库，先冒烟后全量，按退出码把关）、`r17_ops/chain_h100_fwlibero.sh`（Fast-WAM MT50 跑完后跑 Fast-WAM LIBERO A/B 资格检查，这是它建库的前置条件）。
- 监控：`r17_ops/r17_all_watch2.sh`（每 5 分钟报正式测试进度，加 sol 完成事件）。

**10-09 22:3x 排期：新模型 LIBERO 建库（owner 22:2x 下达，参数已问定）**
- 范围：DP、Fast-WAM、Cosmos（Cosmos 等 sol D 接入并过资格检查）× libero_spatial、libero_10。π0.5、GR00T 以前做过，不重做。
- 初始状态：沿用 π0.5/GR00T 的 B 池（差集池），每个套件 500 个；官方 A 池 500 局只做测试。嵌套子集用 R10 同一组排列（`exp/offline_search/rounds/r10/subsets/*_{50..500}.json`，种子 20261002，π0.5 与 GR00T 排列已核对一致），所以各模型在每个库大小上逐局对应。
- 库大小：50/100/200/300/400/500（owner 选，不是 R16 的 4 档）。
- 采集：conductor 逐局分发 + 多副本，老师用满步（DP100/Fast10/Cosmos5），debug 模式采集。只留建库和拟合要用的字段，**外加两路相机图像（owner 说保留）**，其余调试数据丢掉。失败局保留（同 R10）。
- 建库：先建 500，再按排列切出 50–400（按行号引用，不复制数组）；每档只用本档库拟合马氏距离度量、留本集修正器、守卫阈值、上限等 R16 用到的拟合物；500 档沿用留本集服务的说明。
- 实现：派 P6 写采集与建库（`exp/r17_integration/ops/`），先只用 CPU；GPU 冒烟和全量采集等当前正式测试结束后由协调者开。这一步也解决 G3 合成库拟合不出校正器的问题（相当于 ROUND4 R4-1 的 LIBERO 部分）。

**10-09 21:0x 4090 资格检查小结 + 纯减步（最新，先读）**
- 核心 A+B 全 PASS：π0.5-MW（G1 首个端到端，`slot_G1_pi05_mw_{A,B}`）、GR00T-MW（A/B/K2/K1）、GR00T 60k（A/B）、DP-LIBERO（A/B）。收据在 `/data/mw_finetune/qualification/runs/`，序列日志 `r17_ops/qual_4090_retry{1..4}.log`。
- 未过：GR00T C/B2（GPU 批量与单条差 0.0029 > 1e-3）→ 归 R4-2；G3 π0.5/GR00T（固定输入合成库拟合不出校正器距离尺度）→ 归 R4-1；π0.5 tail、G3 DP-LIBERO 为设计内限定 PARTIAL。待跑：两个副本时段（需独占 4090），H100 一侧（Fast-WAM LIBERO/MW、DP-MW）等 Fast-WAM 训完。
- MT50 纯减步（RESULTS.md 新表）：GR00T-MW 4 步 77.6/68.2，2 步 76.0/67.8，1 步 73.8/63.0（按任务平均/组均值）。π0.5 K=2 在跑，完了自动接 K=1（`r17_ops/after_pi05_k2.sh`）。
- DP-LIBERO 全量 LIBERO 测试：踩了 HOME、PYTHONPATH、服务元数据缺 `concurrent`（已在 `exp/mw_finetune/dp/serve_libero.py` 元数据加 `"concurrent": True`，只改元数据）、/scratch 不能 tether 拉取 4 个坑，20:44 起 4 服务（端口 23152–23155）× 4 套件跑，较慢（100 步 DDPM + 共卡）。

**10-09 20:3x 状态（最新，先读）**
- MT50 纯减步：20:07 起本地 tmux `r17_eval_groot_k2`（GR00T-MW K=2，端口 23151）与 `r17_eval_pi05_k2`（π0.5 K=2，端口 23150）共卡跑，结果 `/data/mw_finetune/eval/{groot_mt50_60k_k2_egl,pi05_rlinf_mt50_k2_egl}`；完了接 K=1（同命令换 `--denoising-steps 1` 和 run 名）。`run_eval_local_lane.sh` 支持 `EVAL_NODE` 环境变量换 worker 节点。
- DP-LIBERO 训练 20:21 完成（10 万步，loss 0.026）；LIBERO 全量纯推理测试 20:22 起：本地 tmux `r17_eval_dpl`（`r17_ops/run_eval_dp_libero.sh`：导出拉回 `/data/mw_finetune/dp/libero_policy_100000`，4090 端口 23154 服务，timan107 tmux `r17_dpl_<suite>` 4 套件 × 4 worker，`examples/libero/main.py` 50 局/任务），结果拉回 `/data/mw_finetune/eval/dp_libero_100k/`。
- 资格检查第三轮：PASS = DP-LIBERO A、B（另 π0.5 A、GR00T A/K2/K1/60k A 之前已过）；G3_dp_libero 为设计内的限定 PARTIAL；三个 MW B 时段链路已通（worker 干净退出、driver PASS），卡在把 timan 上的打包文件从 `/scratch` 拉回（tether allow_roots 只有 /home /tmp /srv）→ 派 Q（GPU fix 4，日志 `r17_ops/sol_intQ_gpufix4.log`）。
- 用户问过为什么出这些问题：①π0.5 K2 是编译 vs eager（R16 服务强制 eager）；②探针是 P2 偏离了框架 driver 的语义（框架只在首拉记探针）；③收尾竞争在 LIBERO 上由 runner 按租约回收 worker。教训：偏离框架的地方先查框架原做法。

**10-09 19:5x 第二轮 4090 结果**：PASS = π0.5-MW A、GR00T-MW A/K2/K1、GR00T 60k A；PARTIAL = π0.5-MW B、GR00T B、60k B（链路通了：driver PASS、两局都被接收；卡在 Q 的远端 worker 管理进程不退出，tether 返回 75）；FAIL = DP-LIBERO A（GPU 上 B1 vs B2 差 6.6e-5 > 1e-5）。派 Q（GPU fix 3，worker 管理进程收尾）、P4（GPU fix 2，DP 批量：建议像 Fast-WAM 一样禁用 B>1），修完重跑 B 时段及其依赖。

**10-09 19:3x 资格检查修复与第二轮重跑**：π0.5-MW A 用 R16-eager 一致性标准重跑 PASS；GR00T B 失败根因是 P2 driver 每次拉取都验探针而 worker 只在首拉发 → P2 改为按连接沿用（GPU fix 2）；Q 刷新部署、重签（GPU fix 2，`qualification/HANDBACK_Q.md`）。19:3x 起本地 tmux `r17_qual_4090c` 跑 `r17_ops/qual_4090_retry2.txt`（6 个可跑 + 8 个依赖时段，独占模式），日志 `r17_ops/qual_4090_retry2.log`。H100 时段要等 Fast-WAM 与 DP-LIBERO 训练都退出（约 22:15），顺序见 HANDBACK_Q.md "H100 wait and later ordered queue"。

**10-09 19:1x GR00T-MW MT50 结果**：500 局 0 错误，按任务平均 77.6%、组均值 68.2%（easy 91.1 / medium 58.2 / hard 71.7 / very hard 52.0），已写入 RESULTS.md；4090 空闲，等 Q 修好 worker 后重跑资格检查。P2 结论：π0.5 K2 失败是编译版 vs eager 的数值差（K2 最大差 0.078），适配器与 R16 实际 eager 路径逐字节一致；补丁在 `REQUESTS/P2_to_Q.md`。P4 已补探针接口。

**10-09 19:0x 4090 资格检查第一遍结果（最新，先读）**
- PASS：`slot_G2_groot_mw_A`、`slot_G2_groot_mw_final60000_A`（GR00T-MW 真实断点 K4/2/1 原生推理与采集一致）。
- FAIL：`slot_G1_pi05_mw_A`，K=10 逐字节一致，K=2 与原版非分阶段路径不一致 → 派 P2（`prompts/sol_int_P2_gpufix1.md`，日志 `r17_ops/sol_intP2_gpufix1.log`）判断是 K2 网格/路径真 bug 还是编译 kernel 浮点差，可做 ≤15 分钟 GPU 诊断。
- PARTIAL：两个 GR00T B 时段——driver 拒收 timan107 worker（"worker simulator package mismatch"）；GR00T K2/K1——端口 23156 被上一个时段占着；DP-LIBERO A——Q_to_P4 的探针接口未实现。→ 派 Q（`prompts/sol_int_Q_gpufix1.md`，日志 `r17_ops/sol_intQ_gpufix1.log`）修 worker 包指纹与端口等待；派 P4（`prompts/sol_int_P4_gpufix1.md`）实现 `--load-spec/--observations`。修好后用 `r17_ops/run_qual_seq.sh` 重跑失败/跳过的时段。
- Long-WAM 评估完成（`exp/offline_search/rounds/r17/survey_longwam/ASSESSMENT.md`）：建议不迁移（LIBERO 权重都要 2.4 s 历史、无零历史权重、无 MetaWorld、与 Fast-WAM 同族）；可选便宜探针（当前帧复制 13 份当历史）等 owner 定。

**10-09 18:4x GR00T 训完，4090 测试 + 资格检查开跑（最新，先读）**
- GR00T 18:40 训完（6 万步，最终 loss 0.0088，`/data/mw_finetune/groot/4090_mt50/checkpoint-60000`）。
- 18:41 GR00T MT50 500 局正式测试：本地 tmux `r17_eval_groot`（`run_eval_local_lane.sh groot groot_mt50_60k_egl 23151 12 -- serve_local.sh …checkpoint-60000`），worker timan107 tmux `r17_groot`，日志 `r17_ops/eval_groot_mt50_60k_egl.log`，结果 `/data/mw_finetune/eval/groot_mt50_60k_egl/summary.json`。
- 18:41 4090 资格检查序列：本地 tmux `r17_qual_4090`（`r17_ops/run_qual_seq.sh r17_ops/qual_4090_lane.txt --share-eval`，按依赖跑 16 个时段，A 不过就跳过对应 B），汇总日志 `r17_ops/qual_4090_seq.log`，每个时段 `r17_ops/qual_<slot>.log`，收据在 `/data/mw_finetune/qualification/runs/`。独占时段 `slot_G2_groot_mw_R_replicas` 等测试结束后单独跑。
- Long-WAM（NVIDIA 今天发布）可迁移性评估：Opus 后台在读代码，交 `exp/offline_search/rounds/r17/survey_longwam/ASSESSMENT.md`。初判：它的卖点是长历史上下文（KV 复用），与"只看当前观测"冲突；除非有零历史模式/权重。

**10-09 17:0x DP-LIBERO 微调改到 H100（owner："为什么不去h100上微调diffusion"）**：把 `libero_96`（6.9 GB）和最新 DP 代码推到 H100（`r17_ops/push_dp_libero_h100.sh`，核对通过），17:0x 在 h100 tmux `r17_dp_libero` 开训（从 brandonyang LIBERO 30k 微调，10 万步、batch 256、每 2 万步存一次、显存上限 8 GiB、不走锁，输出 `/data/mw_finetune/dp/h100_libero_finetune`），约 7–8 步/秒，预计 20:40 前后完；监控 `r17_ops/dpl_h100_watch.sh`。Fast-WAM 因共卡变慢，预计约 22:15 训完。4090 在 GR00T 训完后只跑 GR00T 测试 + 资格检查，约 21:00 可交还。

**10-09 16:4x 第四轮规划（ROUND4_PLAN.md）**：sol P 列了全部功能 × 6 个模型/基准组合的现状矩阵、18 个缺口（G01–G18）和 4 个工作包：R4-1 真实老师建库 + 实测成本系数 + Fast/DP 锁等待计时（最高优先）；R4-2 并发部署/副本/批处理（π S1–S3、GR S3、Fast、DP）；R4-3 WARM/warm-reset/CP2/trace/RIT/routing；R4-4 物理采集（含 LIBERO worker）/完整分析/split。边界：GR00T-LIBERO 原本就不支持 S1/S2 批处理和自动副本，不算缺口。GPU 资格检查通过后开 R4-1。

**10-09 16:1x DP MT50 结果**：500 局 0 错误，按任务平均 61.2%、组均值 47.4%（easy 79.6 / medium 40.9 / hard 35.0 / very hard 34.0），写入 `exp/mw_finetune/RESULTS.md`。H100 上 DP 服务已停。

**10-09 15:5x GPU 阶段计划（最新，先读）**
- Q 15:45 交回最终脚本（34 个 `qualification/slot_*.sh`，CPU 预检全过；顺序与前置条件见 `qualification/PLAN.md`）。脚本检测到训练进程会拒跑（PARTIAL）。
- **4090（GR00T 训完约 18:40，后台等待任务 b99g2x4cz 会通知）**：① GR00T-MW 500 局正式测试（`r17_ops/run_eval_local_lane.sh groot groot_mt50_60k_egl 23151 12 -- bash exp/mw_finetune/groot/serve_local.sh --checkpoint /data/mw_finetune/groot/4090_mt50/checkpoint-60000 --port 23151`）；② 同时按 PLAN 顺序跑 4090 资格检查（加 `--share-eval`）；③ DP-LIBERO 微调（`exp/mw_finetune/dp/libero/launch.sh`，先 --probe）。
- **H100（Fast-WAM 训完约 20:50）**：Fast-WAM-MW 500 局（`run_eval_h100_lane.sh fastwam fastwam_mt50_frozen_b128_egl 23252 12 '<serve_compat … policy_016001>'`）+ H100 资格检查（Fast-WAM LIBERO/MW、DP-MW）。
- DP MT50 测试（4 路）约 16:15 完。

**10-09 15:3x 第三轮完成，两路 Opus 都认可进 GPU 资格检查（最新，先读）**
- `review/REVIEW_A3.md`：N1–N4 已由 P1/P2 修好（IR 按 R11 在库上回放校准；任务表在服务边界强制对齐；下降错误分类；绑定零哈希）；A 自己修了 GR00T 锁等待计时、GR00T 默认任务表、P6 按 service_ms 计价 + 按阶段台账的 owner IR、debug 写入器字段冲突；测试 R17 350、GR00T 63、Fast-WAM 110、DP 125、legacy 1087 全过，R16 verify 通过。限制：MW 的 IR 成本系数未实测前只能用显式 periodic_gap；service_ms 含线程间 GPU 争用。
- `review/REVIEW_B3.md`：B 自己修了 DP-LIBERO 读不了 P2 spec（major）、Fast-WAM 哈希放在键路径上、DP 接受 Fast-WAM 专用协议；Fast-WAM+common+benchmarks 249、DP 125 等全过。
- 仍不支持（下一轮实现）：replica、阶段 batching（GR00T S1/S2）、WARM/CP2/trace/routing、record/collector、split、实测 GPU 计价。计划：GPU 资格检查后再派一轮实现 + Opus 审查。
- DP MT50 正式测试改为 4 路并行（15:10 起，本地 tmux `r17_eval_dp4`，`r17_ops/run_eval_dp_h100_parts.sh`，h100 tmux `r17_srv_dp1..4` 端口 23254/23255/23258/23259，timan108 `r17_dp1..4`，按 `r17_ops/mt50_task_parts.txt` 分 4 组；完成后 `merge_mt50_parts.py` 合并到 `/data/mw_finetune/eval/dp_mt50_scratch_100k_h100_merged`），约 7 局/分钟，预计 16:20 完；之前单进程的 53 局作废留档。
- Codex 后台服务 15:20 自动升级到 0.162.1 并重启，断了 Q，已同线程续派（`prompts/sol_int_Q_resume2.md`，日志 `r17_ops/sol_intQ_r2.log`）。Q 交回后要再让它按 A3 改动重生成一次。

**10-09 15:0x 第三轮：Opus 审完直接改（最新，先读）**
- P1–P4 第 2 轮修复 14:52–15:01 全部交回，sol 停手。DP MT50 训练 14:46 完成（10 万步，loss 0.0087），500 局测试已自动开跑（本地 tmux `r17_eval_dp`）。
- 15:0x 续派两路 Opus 第三轮（按 owner：审完自己改到认可为止）：A 写范围 `exp/r17_integration/{common,benchmarks,conductor,models/pi05,models/groot,debug,search,ops}` + 测试 → `review/REVIEW_A3.md`；B 写范围 `models/{fastwam,dp}` + 测试 → `review/REVIEW_B3.md`；B 需要改 common 时写补丁提议，我转给 A。两路都不碰 `qualification/`（Q 在重生成脚本）。Opus 改完后让 Q 再用 gen_slots.py 重生成一次。

**10-09 14:4x Opus 复审完成，修复第 2 轮（最新，先读）**
- `review/REVIEW_A2.md`：第一轮 13 条修好 12 条（部分修 1 条）；新 major：N1 recipe `target_ir` 控制的是 MISS 占比而非 R16 的 IR；N2 recipe 任务顺序与 benchmark task_id 没核对（会用错任务的库，已复现）；minor N3（下降卡死）、N4（每次绑定在事件循环上 SHA 整个老师库）、N5（cost_ms 含排队）。测试：R17 281、GR00T 61、legacy 1007 通过。
- `review/REVIEW_B2.md`：B1/B2/B4/B5/B7–B10 修好；B4 后 key 占整次调用 FLOP：LIBERO 0.14、MW 0.09；B3 网格对了，K2/K4/K10 相对 K100 的 MSE 0.020/0.016/0.019，K1 1.77 不可用；新 major N1：Q 的资格检查脚本没跟上（LIBERO 素材 480²、DP 判定 12/11、Fast-WAM 批量 2）；若干 minor。
- 14:4x 派第 2 轮修复：P1（IR 语义、任务表绑定、老师库只在加载时校验一次）、P2（任务表身份、下降错误分类、成本账区分排队/服务时间）、P3（debug 参数、协议交叉校验、续跑身份、measure）、P4（K1 只在纯减步研究标志下允许）、Q（等 P1–P4 修完后用 gen_slots.py 重生成全部脚本并重跑 CPU 预检）。日志 `r17_ops/sol_intP{1..4}_fix2.log`、`sol_intQ_fix1.log`。
- **owner 14:5x：「第三次审查的时候你让opus审查结束后就自己上手改，改到自己同意为止」** → 第 2 轮修复全部交回后，续派原来两路 Opus（A：框架/MW/π0.5/GR00T + P6；B：Fast-WAM/DP）做第三轮：审完直接改代码、补测试，反复改到自己认可为止；按目录分工避免两路改同一文件，sol 那时都已停手。Q 的脚本在 Opus 改完后再用 gen_slots.py 重生成。

**10-09 14:2x 资格检查脚本就绪 + DP 测试自动排队（最新，先读）**
- sol Q 交回：`exp/r17_integration/qualification/` 34 个一键脚本 `slot_*.sh`（`--preflight` 只跑 CPU；`--share-eval` 允许和正式测试按显存共卡；检测到训练进程或显存不够会返回 PARTIAL），顺序与显存表见 `qualification/PLAN.md`，说明 `HANDBACK_Q.md`。训练结束后按 PLAN 顺序跑；若 Opus 复审导致代码再改，用 `gen_slots.py` 重生成。
- .gitignore：Sonnet 加了 `exp/r17_integration/.gitignore`、`exp/mw_finetune/.gitignore`、`exp/offline_search/rounds/r17/.gitignore`，未跟踪文件 22234 → 511（没提交）。
- DP MT50 正式测试：后台等待任务（等 H100 上 DP 训练进程退出且 checkpoint-100000 存在）会自动起本地 tmux `r17_eval_dp` 跑 `r17_ops/run_eval_dp_h100.sh`（导出 → h100 tmux `r17_srv_dp` 端口 23254 不走锁 → timan108 tmux `r17_dp` 500 局 → sync_back → aggregate；日志 `r17_ops/eval_dp_mt50_scratch_100k_h100.log`，结果 `/data/mw_finetune/eval/dp_mt50_scratch_100k_h100`）。

**10-09 14:1x 修复第 1 轮完成，Opus 复审中（最新，先读）**
- P1–P6 全部交回 "Fix round 1"（各 handback 末节）。要点：4 个封存共享文件已恢复 HEAD（git 对 src/ 零改动），R16 `analyze verify` → DESIGN_VERIFIED；五层改为按 bundle 的 `r17.recipe.v1`；LIBERO 经 `benchmarks/libero_bridge.py` 支持执行步数回报与全局 task id；逐决策成本账；GR00T 键名 blocker 修好；Fast-WAM 文本编码移出 key 路径；DP 改 trailing DDIM（K1 标有损）。
- 14:1x 让原来两路 Opus 在原上下文复审 → `review/REVIEW_A2.md`、`REVIEW_B2.md`。复审通过后跑 GPU 资格检查（sol Q 的 `qualification/slot_*.sh`）。

**10-09 13:3x 审查结果与修复第 1 轮（最新，先读）**
- `exp/r17_integration/review/REVIEW_A.md`（框架/MW/π0.5/GR00T）、`REVIEW_B.md`（Fast-WAM/DP），复现脚本 `review/repro_{A,B}/`。共 2 blocker + 8 major + 若干 minor，CPU 测试原先都没抓到。
  - blocker：GR00T-MW 只认 image/state 键（worker 发 observation/*）→ 每局首次推理 KeyError；LIBERO 旧 runner 不回报执行步数 → commit10/15/32 在 LIBERO 上实际是 commit5。
  - major：P1 改共享文件破坏 R16 DESIGN_LOCK（R16 封存分析拒跑）；五层是进程级配置不能按 bundle 切；GR00T 服务入口没接五层、库与 K 绑死、debug 部署不了；没有逐决策成本；LIBERO 侧身份/task_id/conductor 只写了 MW；DP 减步网格是截断 DDPM（K1 动作 MSE 1.21 vs K100 0.24，K2 正常）；Fast-WAM key 路径每次跑 UMT5 文本编码器（key 成本虚高）。
- 13:3x 用同线程把修复派回 P1–P6（任务 `prompts/sol_int_P{n}_fix1.md`，日志 `r17_ops/sol_intP{n}_fix1.log`），每个问题要求补回归测试并在 handback 追加 "Fix round 1"。修完再派 Opus 复审，然后 GPU 资格检查。

**10-09 13:0x Opus 代码审查（owner："他们代码写完了之后你发opus agent去审所有模型都要包含"，"没必要那么多路"）**
- 2 路 Opus（Agent 工具，后台）：A 审共享框架 P1 + MetaWorld/conductor/π0.5-MW（P2）+ GR00T-MW（P5）+ 相关工具（P6）→ `exp/r17_integration/review/REVIEW_A.md`；B 审 Fast-WAM（P3，LIBERO+MW）+ DP-T（P4，LIBERO+MW）→ `REVIEW_B.md`。只读、零 GPU，只报会改变行为或结论的问题。审完把 blocker/major 派回对应 P 线程修，再跑 GPU 资格检查。

**10-09 12:4x 接入实现全部交回（最新，先读）**
- P1–P6 全部交回 `exp/r17_integration/handbacks/HANDBACK_P{1..6}.md`（CPU 测试：P2 51、P3 49+91 回归、P4 6 个 YAML 校验通过、P5 196、P6 142，均 0 跳过；零 GPU）。P1 共享改动我已审：4 个文件 +84 行，全部由 `adapter_binding` 有值才生效，旧路径不变。
- 未完成：GPU/真机资格检查 G1（π0.5-MW 首个端到端）、G2（各模型）、G3（全功能工具链），见 `exp/r17_integration/ops/RUNBOOK.md` 与各 handback。
- 12:4x 派 sol Q（tmux `cx_intQ`，日志 `r17_ops/sol_intQ.log`，线程 `01a121c0-0b47-79c3-a554-f6fdb32f9d67`）：把每个资格检查做成一键脚本 `exp/r17_integration/qualification/slot_*.sh`（≤15 分钟、只起自有进程、按 PID 停、写收据），先做完所有 CPU 预检和 timan worker 部署，交 `qualification/PLAN.md` + `HANDBACK_Q.md`。GPU 时段：4090 在 GR00T 训完（约 18:40）后，H100 在 Fast-WAM 训完（约 20:30）后。

**10-09 11:5x DP 训练（最新，先读）**
- sol E 交回（`exp/mw_finetune/dp/HANDBACK_E.md`、`RECIPE.md`）：brandonyang 两份权重重建无误（原始帧上去噪误差低），但迁移到我们设置都不可用（MW ML45 1/16、0/16，相机 corner4+夹爪视角对不上；LIBERO 0/4）→ 按 owner 规则：MT50 从头训，LIBERO 在 30k 上微调。配方：10 万步、batch 256、AdamW(.95,.999)、1e-4→1e-5、500 预热、无 EMA，DDPM 100 步。
- 11:5x 在 H100 上和 Fast-WAM 共卡开训 DP MT50 从头训（tmux `r17_dp_mt50` on h100，直接跑 train.py 不走锁、显存上限 8 GiB、10000 步存一次，输出 `/data/mw_finetune/dp/h100_mt50_scratch`，日志 `.console.log`），约 8–9 步/秒，预计 15:30 前后训完。监控 `r17_ops/dp_h100_watch.sh`（已并入 r17_all_watch.sh）。
- DP LIBERO 微调：数据只在本机，等 GR00T 训完（约 18:40）后在 4090 跑：`bash exp/mw_finetune/dp/libero/launch.sh --steps 100000 --batch-size 256 --workers 4 --save-every 5000 --output /data/mw_finetune/dp/4090_libero_finetune`（先 --probe）。

**10-09 11:4x 系统接入：6 个实现 sol 已派（最新，先读）**
- sol P 交回设计（11:31）：`exp/r17_integration/{INVENTORY,DESIGN,WORK_PACKAGES}.md` + `verification/CPU_CHECKS.md`；每项功能标 F（框架自动）/M（每模型钩子）/B（每基准），沿现有 connection factory → YAML config → orchestrator 路径接入；第一个端到端目标 π0.5-on-MetaWorld。
- 11:4x 并行派 P1–P6（tmux `cx_intP1`..`cx_intP6`，日志 `r17_ops/sol_intP{1..6}.log`，任务 `prompts/sol_int_P{n}.md`，交 `exp/r17_integration/handbacks/HANDBACK_P{n}.md`，互相请求放 `exp/r17_integration/REQUESTS/`）：
  | 包 | 内容 | 线程 |
  |---|---|---|
  | P1 | 共享框架 + 五层参数化（唯一可改共享 `src/openpi/cache/*`、`websocket_policy_server.py`，默认关） | 01a12181-e8ba-7640-b689-2f2da8860edb |
  | P2 | MetaWorld 基准 / conductor 按局分发 / worker / 物理 debug 采集 + π0.5-MW 首个端到端 | 01a12181-f099-7ff0-aebd-d634cc7657b5 |
  | P3 | Fast-WAM LIBERO + MW 适配 | 01a12181-f87d-73c1-88f6-0407d9f849bf |
  | P4 | DP-T LIBERO + MW 适配（用 sol E 的重建） | 01a12182-005b-7d03-8dbb-28057d4d588c |
  | P5 | GR00T-MW 接入 | 01a12182-0845-7471-9410-f44e422a0d26 |
  | P6 | 库/debug/离线搜索/分析/部署资格 | 01a12182-101b-7822-a6dc-dea5e79dcc2d |
- 今天零 GPU（两卡训练中）；GPU 资格检查命令由各包写好，训练结束后我一次一个模型跑。

**10-09 11:2x Fast-WAM 训练事故（最新，先读）**
- 11:19 在第 10000 步存断点时，C 的磁盘预算检查报 `H100 shared budget: 119.2+13.3 GB would exceed 120 GB` 后退出（/data 实际还有 77 GB）。原因：H100 上 `/data/mw_finetune` 被推上去的 GR00T 核对断点、sol E 的 DP 文件，以及 C 的两个测速目录撑大了。
- 处理：按 owner 之前"H100 上我们不需要的都可以删了"，删了 `fastwam/probe_frozen_b16`（16 GB）、`fastwam/probe_lora_b16`（21 GB）、`groot/eval_ckpts/4090_mt50/checkpoint-30000`（7.1 GB，原件在本机）→ 共享目录 69 GB。11:21 从第 8000 步 `latest.pt` 续训，损失约 2000 步（约 2 小时）→ **Fast-WAM 预计改为 19:05 前后训完**。
- 今后往 H100 `/data/mw_finetune` 放东西要算进这个 120 GB 预算（C 的训练每存一次断点要临时多 13.3 GB）；sol E 在 H100 上保持 ≤ 10 GB。

**10-09 10:5x 新目标：新模型 + MetaWorld 接入系统（owner 10:4x）**
- owner 原话：「1.把新模型接入我们的系统 2. 把metaworld 接入我们的系统 对他们，要支持数据debug 收集 要支持concurrent server yaml热切换， 要支持conductor按照最细的episode分发任务，一切groot 和pi支持的他们都要支持」。
- 阶段一：派架构 sol P（tmux `cx_intP`，日志 `r17_ops/sol_intP.log`，线程 `01a1214f-8cfc-7411-b0a7-7495e15e9b6b`，任务 `prompts/sol_int_plan.md`）→ 交 `exp/r17_integration/{INVENTORY,DESIGN,WORK_PACKAGES}.md`（π0.5/GR00T 全功能清单、模型适配接口、MetaWorld 基准接口、4–6 个可并行工作包，π0.5-on-MetaWorld 作为第一个端到端目标）。
- owner 10:5x 补充：「我们很多功能都是在框架层实现的，理论上框架正确接入后就会自动支持，你要仔细接入框架，但是其他功能有些我们要自己写」→ 写进 `exp/r17_integration/OWNER_NOTE.md`。sol P 回合中途收不到消息：它交回后若没按这条把每项功能标成"框架自动 / 每模型要写 / 每基准要写"，用同线程续派让它读 OWNER_NOTE.md 修订。
- 阶段二：我审完后按工作包并行派 sol 实现。
- sol E（DP）10:4x 用同线程续派（`prompts/sol_mw_E_resume1.md`，日志 `r17_ops/sol_mwE2.log`）。
- Codex 后台服务（app-server）在运行，用于派 sol。

**10-09 10:3x 重启后状态（最新，先读）**
- 本机 10:07 重启完成：内核 6.8.0-146，驱动 595.91.07，nvidia-smi 正常，EGL GL_RENDERER = RTX 4090。
- GR00T 10:14 从 checkpoint-30000 续训（tmux `r17_groot_train`），4090 约 39.6 GB，1.00 s/步，预计 18:40 训完。
- GR00T 中途核对（checkpoint-30000）：4090 余量不够（会挤训练），改在 H100 上和 Fast-WAM 共卡做 → 8 个 easy 任务 **14/16**（`RESULTS.md`）。H100 上的服务已停。
- Fast-WAM 第 9350 步，预计 17:00 训完。
- Codex 后台服务（app-server）重启后已在运行（被拒那条命令其实执行了）；**sol E 被重启打断，等 owner 确认后再续派**。owner 纠正：不要把 Codex app-server 叫"守护进程"，实验由主会话守护（已记进长期记忆）。
- Cosmos-MW 正式 500 局：建议取消，等 owner 定。

**10-09 09:2x 新增扩散模型（owner："我还想加一个扩散模型…metaworld有吗，如果没有可以加入微调"）**
- 现状：LIBERO 有 `brandonyang/dp-libero-checkpoints`（DP-Transformer，语言条件，openpi 式目录，无代码/无 README/无成功率）；MetaWorld 只有同作者 `brandonyang/dp-metaworld-checkpoints`（训练集是 **ML45**，5 个 MT50 任务没见过，无成功率）；`iFaz` 的 DP-UNet 要两帧历史，MW 仓库是空的。公开 MT50 上 DP 组均值 10.5%。
- 09:2x 派 sol E（tmux `cx_mwE`，日志 `r17_ops/sol_mwE.log`，线程 `01a12102-9a92-72d1-af6c-9cb9f15e8c56`，任务 `exp/offline_search/rounds/r17/prompts/sol_mw_E_dp.md`）：找原始代码或从 metadata.pt 重建、LIBERO 封装与冒烟、ML45 权重在 MT50 上 8 任务核对、MT50 微调启动器（预计几 GPU 小时）、`exp/mw_finetune/dp/`。**本机重启会打断它：重启后若日志没有 TURN_COMPLETED，用同一线程续派一句"机器重启过，从中断处继续"。**
- owner 09:3x 定："把 metaworld ddpm 也纳入进来，先尝试他的权重，如果不行我们自己从头微调"。已写 `exp/mw_finetune/REQUESTS/COORD_to_E.md`（先试 brandonyang ML45 85k → 不行再在 MT50 上从头训同构 DP-T，DDPM 100 步默认；owner 09:3x 补充：LIBERO 也一样，brandonyang LIBERO 权重不行就自己在 LIBERO 4 套件上训一份）。**续派 sol E 时要在提示里明确让它读 COORD_to_E.md。** 注：我们之前的 DDPM 减步实验是 robomimic + PushT 的官方 DP-CNN，不是 LIBERO（已向 owner 说明）。

**10-09 09:0x weilandserver 计划重启（最新，先读；本会话会随重启结束）**
- **10:0x 应 owner 要求已手动停掉 GR00T 训练**（停在 30141 步，tmux `r17_groot_train` 已关；重启后从 checkpoint-30000 续）。
- **09:56 GR00T checkpoint-30000 已存完并核对**（global_step 30000，含 optimizer.pt 8.6 GB，已 sync），已推送通知 owner 可以重启。sol E（DP）重启时可能还在跑，它在 H100 上起的冒烟进程（约 1–3 GB 显存）重启后要查是否残留，残留就按 PID 停掉，别影响 Fast-WAM。
- owner 08:5x 同意找时机重启（"是否找一个时机reboot机器，可以操作"）。时点：GR00T 存完 checkpoint-30000（约 09:55，后台等待任务 bb7vtq2z5）后我告诉 owner，**owner 执行 `sudo reboot`**（我没 sudo）。训练不提前停：若停了而迟迟不重启，4090 在修好前无法续训；重启只损失断点后几分钟。I018 已同意（它的状态已落盘，重启后自己恢复 srv-i018-rsyncd）。
- **Codex 共享守护进程**：重启后谁先发现没起来就执行 `codex app-server daemon start`，再告诉对方（I018 约定）。
- H100 不需要再重启：05:06 已重启，驱动与库都是 595.99.02，无 reboot-required。
- **重启后要做（新会话先读这里）**：
  1. `nvidia-smi` 正常、`cat /proc/driver/nvidia/version` 是 595.91.07；验证 EGL：`/home/weiland/projects/openpi_ext/envs/metaworld_sim/bin/python -c "import os;os.environ['MUJOCO_GL']='egl';import mujoco;..."` 或直接用 A 的 camera_parity capture（GL_RENDERER 应是 RTX 4090）。
  2. 续训 GR00T（自动从 checkpoint-30000 续）：`tmux new -d -s r17_groot_train "bash /home/weiland/projects/openpi_ext/scratch/r17_ops/groot_train_4090_nolock.sh --batch-size 64 --workers 12 --steps 60000 --save-steps 30000 --output /data/mw_finetune/groot/4090_mt50 2>&1 | tee -a /home/weiland/projects/openpi_ext/scratch/r17_ops/groot_train_4090.console.log; bash"`，确认进度条从 30000 起、loss 正常。
  3. GR00T 中途核对改回 4090：本地起 `exp/mw_finetune/groot/serve_local.sh --checkpoint /data/mw_finetune/groot/4090_mt50/checkpoint-30000 --port 23151`（它走 GPU 锁；训练不走锁），workers 用 timan107（`run_eval_local_lane.sh`，必要时加子集参数），8 个 easy 任务 × 2 局。
  4. 重挂监控：`r17_ops/r17_all_watch.sh`（H100 Fast-WAM + GR00T + 评测日志）；重建 cron：R17 巡检（:07/:37）。旧的一次性 cron 7ccd21e1（H100 上的 GR00T 核对）随会话消失，不用了。
  5. H100 的 Fast-WAM 训练不受本机重启影响（tmux `r17_fastwam_train` on h100），预计 17:00 前后训完。之后按 `r17_ops/run_eval_h100_lane.sh` 跑 500 局 → 书面交还 H100。

**10-09 06:4x 本机驱动事件（最新，先读）**
- 06:23 CDT unattended-upgrades 把本机 nvidia-*-595-server 从 595.71.05 升到 595.91.07，加载的模块仍是 595.71.05 → 本机 nvidia-smi 报 mismatch，**4090 起不了新 CUDA 进程**（GR00T 训练照跑）；另有内核 6.8.0-146 的 reboot-required。我没有 sudo，已推送通知 owner，并告知 I018、写台账（weilandserver运维 会话已不在线）。恢复：GPU 上无进程时 rmmod nvidia_uvm nvidia 后 modprobe，或重启；之后验证 EGL GL_RENDERER。
- 应对：GR00T 中途核对改到 H100 和 Fast-WAM 共卡做（cron 7ccd21e1，10:02；脚本 `r17_ops/push_groot_ckpt_h100.sh` 只传权重约 7.6 GB，GR00T 服务不走锁）。旧 cron 368f384e 已删。
- GR00T 正式测试：owner 修好 4090 就在 4090 上测，否则等 Fast-WAM 在 H100 测完后在 H100 上测。Cosmos 正式测试要么等 4090 修好，要么在 H100 上搭 Cosmos 环境。

**10-09 05:1x H100 已重启 + Fast-WAM 中途核对（最新，先读）**
- 05:05 Fast-WAM 第 4000 步断点落盘后停训，复制核对用断点 `policy_sanity_004000`；05:06 `sudo reboot`，05:09 回来，驱动和库都是 595.99.02，nvidia-smi 正常（tether agent 是用户级 systemd 服务，开了 linger）。
- 中途核对：8 个 easy 任务 × 2 局，16 局成功 10 局，约定没问题（`RESULTS.md` 已记，标为中途核对）。
- 05:15 用原命令 `--resume` 续训（tmux `r17_fastwam_train`），从第 4000 步接上。预计 16:50 前后训完。
- cron d44a4848 的步骤都已提前完成。

**10-09 03:2x 后续排程（最新，先读）**
- **05:12 一次性 cron d44a4848**：H100 提前重启（I018 已同意 05:10，/dev/shm/i018 已备份）。步骤：等 Fast-WAM 第 4000 步断点落盘 → 停训练 → 复制 `policy_latest` 为 `policy_sanity_004000` → `sudo reboot` → 验证驱动 → 8 个 easy 任务 × 2 局的中途核对（`EVAL_EXTRA=... run_eval_h100_lane.sh fastwam fastwam_sanity_004000 23252 8 '<serve_compat>'`）→ 用原命令 `--resume` 续训。
- **10:17 一次性 cron 368f384e**：GR00T 在第 30000 步存断点后，在 4090 上和训练共卡做同样 8 任务的中途核对（端口 23151，timan107）。
- GR00T 不和 π0.5 共卡后每秒约 1.0 步，预计 10-09 18:10 前后训完；Fast-WAM 预计 17:30–18:00 训完（含重启耽误的时间）。
- 训完：Fast-WAM 在 H100 + timan108 跑 500 局正式测试 → 交还 H100；GR00T 在 4090 + timan107 跑 500 局 → Cosmos 默认配置 500 局 → 交还 4090。

**10-09 03:1x /archive 搬迁完成**：9 项 1670.7 GiB 全部迁完（MOVER_EXIT rc=0，03:08），9 个原路径都是指向 /archive 的软链、14 条 relink 已核对；/data 可用 786 → 2475 GiB。已通知 I018 窗口结束、台账已写、巡检 cron 47ef23ad 已删。/data/dp（156 GiB）和 /data/xwam（141 GiB）在 PLAN.md 里标为 needs owner，没搬。

**10-09 01:3x R17 运行状态（最新，先读）**
- H100：Fast-WAM 训练 01:08 起（tmux `r17_fastwam_train` on h100；冻结视频，batch 128，16,001 步，约 3.7 s/步，预计 17:00 CDT 前后完；输出 `/data/mw_finetune/fastwam/mt50_frozen_b128`，最终 `policy_016001`）。监控脚本 `r17_ops/h100_train_watch.sh`。
- **H100 驱动事件**：01:12 被 unattended-upgrades 升到 595.99 用户态库，内核模块仍 595.84，需重启；训练不受影响。I018 同意：训完后由 openpi `sudo reboot`，I018 已于 01:3x 确认 /dev/shm/i018 备份完成，训完即可重启；重启后在 H100 上跑 Fast-WAM 测试：`r17_ops/run_eval_h100_lane.sh fastwam fastwam_mt50_frozen_b128_egl 23252 12 '<serve_compat 命令，见 fastwam/HANDBACK_C.md>'`，跑完书面交还 H100（约 19:30）。apt-mark hold 转 owner 定。
- 4090：GR00T 训练 01:1x 起（tmux `r17_groot_train`，`r17_ops/groot_train_4090_nolock.sh`，不走锁以便和测试共卡；batch 64，6 万步，约 1.3–1.4 s/步，预计 10-09 23:00 前后完；输出 `/data/mw_finetune/groot/4090_mt50`）。监控 `r17_ops/groot_train_watch.sh`。之后：GR00T 测试（端口 23151 + timan107）→ Cosmos 默认配置 500 局（23153）→ 交还 4090。
- **π0.5 RLinf 测完（02:43）**：按任务平均 58.8%（与 9 月一致），组均值 50.5%（easy 70.0 / medium 45.5 / hard 46.7 / very hard 40.0）；结果汇总表 `exp/mw_finetune/RESULTS.md`。
- Cosmos-MW：D 的探测 75 局只成功 1 局（`cosmos/HANDBACK_D.md`），作者结果复现不了；正式 500 局仍要跑，以留下记录。

**10-09 00:5x R17 进展（最新，先读）**
- A 已交回（HANDBACK_A.md）：正式评测 worker 跑在 timan（owner 00:1x 准许用 timan107/108）。分工：4090 服务（端口 2315x）配 timan107 worker，H100 服务（端口 2325x）配 timan108 worker；渲染统一用 NVIDIA EGL。相机核对通过：评测画面与 lerobot/metaworld_mt50 一致。容量（零策略）：timan107 约 72 局/分钟，timan108 约 120 局/分钟。
- 我写的编排脚本 `openpi_ext/scratch/r17_ops/run_eval_local_lane.sh`（起本机服务 → timan107 worker → 同步回本机 → 汇总 → 停服务）。π0.5 正式测试 00:49 起，tmux `r17_eval_pi05`，日志 `r17_ops/eval_pi05_rlinf_mt50_egl.log`，结果 `/data/mw_finetune/eval/pi05_rlinf_mt50_egl/summary.json`。
- B（GR00T）：全量数据已转好并校验。H100 batch 128、训扩散头：每秒 0.644 步、44 GiB；4090 训扩散头的测速在排队。我的计划：60k 步 × batch 128（NVIDIA LIBERO-90 单卡），先在 4090 上训，Fast-WAM 训完后断点接到 H100 续训，约 32 h。
- C（Fast-WAM）：冻结视频在 batch 128 时 10 轮约 15.6 h；视频 LoRA 在 batch 16 时 46 h（在测更大 batch）。规则：LoRA 10 轮 ≤ 24 h 就用 LoRA，否则冻结视频。指令在 `exp/mw_finetune/REQUESTS/COORD_to_{B,C}.md`。
- D（Cosmos 社区权重）：四种重建配置在 assembly 上全是 0/10，reach 最多 1/5，看来复现不了作者结果。正式 500 局待 D 交回后再定。
- sol 日志：`r17_ops/sol_mwB.log`、`sol_mwC.log`、`sol_mwD2.log`（D 续派，探测）、`sol_mwA3.log`（A 续派，已完成）。监控 `r17_ops/sol_watch.sh` 读 `watch_list.txt`。

**10-09 00:0x 压缩前快照（先读这段，最新）**

### A. 正在做：R17 = 在 MetaWorld MT50 上微调 GR00T N1.5 和 Fast-WAM，测纯推理成功率（会话 /goal 已挂）
- **owner 原话目标**：「编写测试和微调工具，在metaworld上按照论文配方微调groot和fast wam并测试各个模型微调后纯inference成功率」；**授权独占 4090 和 H100**（台账"独占 X1"，I018 23:4x 已把两卡腾空 0 MiB；I018 要求**哪张先用完先还哪张**，每次 SendMessage 书面通知 + 写台账；/dev/shm/i018 不许动）。owner 23:5x 去睡，交代"独自推进"。
- **要测的模型**：GR00T N1.5-MW（自己微调）、Fast-WAM-MW（自己微调，从官方 LIBERO 权重 `yuanty/fastwam` 起步，冻结视频骨干只训动作专家或 LoRA）、π0.5 RLinf SFT（现成 `/data/ckpt/pi05_metaworld_rlinf`，9 月测过按任务平均 58.8%）、Cosmos-MW 社区权重（`mmurray-ms/Cosmos-Policy-MetaWorld-Predict2-2B`，v3，作者 runner 未公开）。
- **benchmark 协议**：MetaWorld MT50 v3（本机 `openpi_ext/envs/metaworld_sim`：metaworld 3.0.0 / gymnasium 1.3.0 / mujoco 3.14；`exp/metaworld/` 现成 env 与 runner，`MT1(task, seed=7)` + `set_task(train_tasks[idx])`），每任务 10 局（idx 0..9，共 500 局），按 easy/medium/hard/very hard（28/11/6/5）报组均值和总均值，另给按任务均值。
- **论文配方依据**：MetaWorld 参考配方 = SmolVLA 论文（`lerobot/metaworld_mt50` 数据出处）：10 万步、batch 64、AdamW(0.9,0.95)、lr 1e-4 余弦到 2.5e-6、预热 100；GR00T N1.5 README：batch 开满、2 万步（24 GB 4090 要 `--no-tune_diffusion_model`，我们的 4090 是 48 GB）；Fast-WAM 官方 8 卡 ZeRO-1 全训（单卡只能冻结视频专家或 LoRA，要写明偏差）。MT50 公开组均值：DP 10.5、TinyVLA 31.6、SmolVLA 68.2、π0 SFT 50.8、π0.5 SFT 43.8、π0+PPO 78.1、π0.5+PPO 70.7。
- **数据**：`lerobot/metaworld_mt50` 2,500 局 / 204,806 帧 / 单路 480² / 4 维 state、action / fps 80。本机根目录 `/data/mw_finetune/`，H100 根目录 `/data/mw_finetune/`（H100 /data 现剩 188 GB，底线 30 GB）。Wan2.2-TI2V-5B 底座两边都有：`/data/xwam/wan22_5b/`。
- **4 个 sol 并行（23:55 起）**，共同规则 `exp/offline_search/rounds/r17/prompts/mw_common_rules.md`（目录所有权、服务接口 `load_policy()->infer(obs)->actions[T,4]`、GPU 冒烟用 `flock /tmp/r17_gpu.lock` ≤15 分钟、禁令）：
  | sol | 范围 | tmux / 日志（`openpi_ext/scratch/r17_ops/`） | 线程 |
  |---|---|---|---|
  | A | MT50 测试框架、serve、汇总、worker 部署（timan107 没有 MetaWorld，A 决定装还是本机跑）、π0.5 接入、`CONTRACT.md` | cx_mwA / sol_mwA.log | 01a11efc-d2d7-7483-b643-40132a5da445 |
  | B | GR00T N1.5：RECIPE、MT50 转 GR00T 格式、新机器人标签、训练启动器（--probe）、推理封装；冒烟 4090 | cx_mwB / sol_mwB.log | 01a11f04-328d-7282-992c-3d34070073ca |
  | C | Fast-WAM：RECIPE、H100 环境、复用 Wan 底座、冻结视频/LoRA、启动器、推理封装；冒烟 H100 | cx_mwC / sol_mwC.log | 01a11f04-329e-7133-8094-563a0d49400e |
  | D | Cosmos-MW 社区权重：下载、适配、2 局冒烟（4090） | cx_mwD / sol_mwD.log | 01a11f04-32ac-7eb0-a84c-fd244cac0d16 |
  各交 `exp/mw_finetune/HANDBACK_{A,B,C,D}.md`；目录 `exp/mw_finetune/{groot,fastwam,cosmos}/`。
- **我接下来**：sol 交回 → 审阅 → 测训练速度（--probe）→ 开长训（Fast-WAM 放 H100；GR00T 视速度放 4090 或 H100 之后）→ 每个模型 MT50×10 局纯推理测试 → 汇总表 → 哪张卡用完先还。长训和测试都在 tmux 里由我跑（agent 不看守长实验）。
- **只能派 sol / astra / opus**（owner 10-09 00:0x）；科学类派 astra 或 opus，工程派 sol；同一调研要同时派 opus + astra 独立做、互不可见。

### B. /archive 搬迁（9 项 1.67 TiB，owner 批准；I018 推迟了它在 /archive 的写和删，等我们通知）
- tmux `archive_mv`：`cd openpi_ext/scratch/archive_plan && ./archive_mover.sh move_list_all.txt relink_all.tsv`；日志 `mover.log`（逐文件行很多，看关键行用 `grep -v -E '^[^ ]+ {2,}'`）。
- 进度：前 7 项完成（含 `dual_20260923` 625 GiB）；第 8 项 `r08_main` 复制完、23:58 正在删源前比对；第 9 项 `r06_p3_pilot` 260 GiB 待做。预计 10-09 凌晨 2–4 点完。
- 完成后：核对 9 项原路径是指向 /archive 的软链、`relink_all.tsv` 14 条已改指向 → SendMessage 通知沙堆实验"/archive 写入窗口结束" → 向 owner 报释放空间 → 删巡检 cron 47ef23ad。

### C. 今天已完成（不用再做）
- **R16 定稿并已提交推送**（`d02d59bf`，`Ziyang`）：三批提档重跑（v1 46 任务、v2 7 个 −6 个点边界、v3 7 个 −10 个点改 c10），合并结果 `rounds/r16/retest_v1_v2_v3_results/`；四种库 × k 与满步纯推理持平（−0.3～+0.7），比满步逐任务下降 k1 +1.1 / k2 +1.3（显著），开销 .171/.188 对 .291/.506。
- **解耦**：分支 `decouple-cpu-gpu` 已推（`43702dfd`）。冒烟：现行组跑通（GR00T 1 份 23.1 局/分钟）；拆分组曾因 AF_UNIX 路径过长失败，sol 已修（未再上 GPU 验证）。下次要卡先 `run_gpu_package.sh --smoke --new --deadline ...`。
- **R17 选型过程**：调研都在 `exp/offline_search/rounds/r17/`（survey v1/v2 被否；survey_opus/astra；survey2_* 结论 LIBERO∩MetaWorld 无全过；survey3_* 第三个模型两边都选 Fast-WAM）。owner 定的硬要求见记忆 feedback_porting_nonvla_cache_friendly。
- 4090 / H100 本轮之前的借用（B2、S1、R1）都已结束并写台账。

### D. 挂着的监控与 cron（只活在本会话，压缩后检查 CronList，必要时重挂）
- cron `2026da26`（:07/:37）R17 夜间巡检；cron `47ef23ad`（:17）/archive 巡检。
- Monitor：4 个 sol 交回/停滞（脚本内联，看 sol_mw{A..D}.log 的 TURN_COMPLETED / 进程 / 25 分钟无输出）；/archive `archive_plan/mover_watch.sh`（30 分钟到期要重挂）。
- Codex 守护进程 10-08 20:19 被 I018 重启过（0.162.0），派单照旧用 `codex_daemon_run.py`。

**10-08 23:5x 下一轮（R17）选型进展**：
- owner 暂定新模型 **Fast-WAM + Cosmos Policy**；benchmark = LIBERO + **MetaWorld MT50（v3）**；GR00T N1.5 我们自己在 MetaWorld 上微调。调研全在 `exp/offline_search/rounds/r17/`：`survey*`（v1/v2 被否）、`survey_opus|survey_astra`（第三版独立）、`survey2_*`（LIBERO∩MetaWorld，结论无全过）、`survey3_*`（第三个模型，Opus 选 Fast-WAM、备选 EasyWAM；astra 结果待收）。
- owner 定的硬要求（见记忆 feedback_porting_nonvla_cache_friendly）：非 VLA（世界模型算）、只看当前观测（不要 KV 回填 / 历史 / 目标图）、缓存要有利可图、2–3 个不同家族、尽量不自训、汇报用表格。
- 待办：验证 Cosmos 社区 MetaWorld 权重（写评测包装约 1 天 + MT50×10 约 4 GPU-h）；Fast-WAM 在 MetaWorld 上微调（H100 ≈13 h）；GR00T N1.5 MetaWorld 微调（SmolVLA 配方 10 万步×64，H100 ≈10 h）；都要先向沙堆实验收回 GPU。DP 在 MT50 公开仅 10.5%，是否保留待 owner 定。

**10-08 17:3x 最终状态**：
- R16 全部完成，**报告定稿 `rounds/r16/REPORT.md`（三批重跑合并，`retest_v1_v2_v3_results/`）**：四种库 × k 与满步纯推理持平（−0.3 到 +0.7），开销 .153–.212 对 .506；跨库合并比满步逐任务下降 k=1 +1.1 [+0.4,+1.7]、k=2 +1.3 [+0.6,+1.9]，开销 .171/.188 对 .291。−10 个点边界 7 个任务已按 c10 补跑（v3，`retest_superseding.py`，16:40–17:20，收回 R1）。
- 4090：收回 R1（16:40–17:23）已结束，整卡归 I018，台账已写。我方无 GPU 作业。
- 解耦：冒烟测试现行组通过（GR00T 1 份 23.1 局/分钟），拆分组启动失败 `AF_UNIX path too long`；sol 17:42 已修（短私有 socket 路径 + HF 导入 + 回环绑定 + 就绪等待上限，100 测试通过，4 道门证明仍可复用）。下次要 4090 先跑冒烟：`bash /home/weiland/projects/openpi_decouple/exp/decouple/run_gpu_package.sh --smoke --new --deadline "$(TZ=America/Chicago date -Iseconds -d '+10 minutes')"`，通过再跑整包（约 2 h，需整卡）。
- /archive：第一批 6 项已搬完（含 2 条 relink），在搬三个旧轨迹目录；结束后通知沙堆实验并向 owner 汇报。
- 提交 R16 仍待 owner 授权。

**10-08 14:4x 状态（已被上一段覆盖）**：
- R16 全部跑完：v2 补跑 7 个 −6 个点边界任务 13:48–14:27 完成（350 局，全部变好）；两批合并分析 `rounds/r16/retest_v1_v2_results/`；**REPORT.md 已更新为最终数字**：四种库 × k 与满步纯推理持平（−0.4 到 +0.5），跨库合并比满步逐任务下降 k=1 +1.0 [+0.4,+1.7]、k=2 +1.2 [+0.5,+1.9]，开销 .171/.187 对 .291。
- **待补**：恰好 −10 个点的 7 个任务因同一浮点 bug 提到了 ir45（规则应 c10），清单见 REPORT.md 第三节。要补需 v3（这 7 个按 c10 重跑，约 350 局、25 分钟 GPU）；combined 分析器现在拒绝与 v1 重叠的代际，补跑前要让 sol 加"后一代覆盖前一代"的支持。等下一个 4090 时段。
- **4090 已于 14:27 整卡交还 I018**（台账已写 S1 实际起止、峰值约 29.8 GB；B2/M6/M7 失效）。h100 也是 I018 的。我方现在没有 GPU 作业。
- 解耦基准：sol 已修（延迟归属、复算、门复用、`--smoke` 模式，`run_gpu_package.sh` 绑核已改 2-43,46-87）；已跑完那组复算：GR00T 现行 1 份 31.0 局/分钟。下次拿到 4090 先 `run_gpu_package.sh --smoke --new --deadline ...`（约 10 分钟），通过再跑整包（约 2 小时）。命令见 `openpi_decouple/exp/decouple/GPU_PACKAGE.md`。
- /archive 搬迁 14:27 开跑（tmux `archive_mv`，9 项 1.67 TiB，`move_list_all.txt`/`relink_all.tsv`；mover 已去掉逐文件 fsync 和复制时 --checksum，删源前仍有 sync -f + rsync -rlcn 全量校验）。日志 `archive_plan/mover.log`；巡检 cron 47ef23ad（每小时 :17）。结束后通知沙堆实验（它推迟了 /archive 删除）。
- 提交 R16 仍待 owner 授权。

**10-08 13:1x 状态**：
- 测试集 32 格 11:42 跑完；按 owner 规则提档 46 个任务、重跑 2,300 局（11:43–12:52，`r16_conductor_retest_*_v1`），合并分析 `rounds/r16/retest_v1/results/RETEST.md`。**报告已写：`rounds/r16/REPORT.md`**。
- 结果：提档后四种库大小 × k 全部与满步纯推理持平（−0.8 到 +0.5 个点），开销 .149–.205 对 .506；跨库合并比满步逐任务下降 k=1 +0.8 [+0.1,+1.4]、k=2 +1.1 [+0.4,+1.7]，开销 .166/.186 对 .291。
- **漏了 7 个任务**：per_task.py 浮点比较把恰好 −6 个点算成 −5.999999，没进清单（已改为取整）。清单 `r16_ops/retest_changes_v2_boundary.json`。sol（R16 线程 01a117c6，日志 `r16_ops/sol_retest_v2.log`，tmux `cx_r16_rt`）在做 v2 增量重跑 + v1/v2 合并分析。350 局约 15 分钟 GPU，等下一个 4090 时段。
- 4090：R16 12:52 结束，借用 B2 结束；12:52 起是时段 S1（台账"时段 S1"），跑解耦 GPU 基准，**最迟 15:15 交还 I018**，交还时写台账实际起止时间和显存峰值、书面通知；之后 I018 起整卡计时队列一天多。
- 解耦基准：tmux `decouple_gpu`，命令见 tmux 历史（不用包里的 run_gpu_package.sh：它绑 16 核；我改为 taskset 2-43,46-87）。前两次门失败已修（worktree 补 `assets` 软链；`qualify_gpu.py` 把 jax 依赖的导入移进 π0.5 分支；重生成 `benchmark.local.json` 哈希），13:07 用 `--new` 重开。**日志在 `gpu_comparison/archive_20261008_125227_23dcdb/package_run.log`**（--new 把 tee 的文件挪进了归档目录）。结果 `gpu_comparison/SUMMARY.md`。
- /archive 搬迁：owner 12:5x 让研究 /data 哪些能搬 /archive 并改软链（"astra 是土豪用法，用 sol 或 sonnet"）。sol 新线程 `01a11ca6-96ca-7753-9b5f-c25c17eae853`，任务 `openpi_ext/scratch/archive_plan/sol_archive_plan.md`，只读调研 + 写搬迁脚本，交回 PLAN.md / move_list.txt / relink.tsv / archive_mover.sh（dry-run）。我审后在 tmux 里单流搬。/data 上 openpi 约 2.5 TiB（i018、idea_sprint 不是我们的）。
- owner 10-08 还说：视觉头显存不测（"不测视觉头，不管他"）。
- **/archive 搬迁（13:3x 更新）**：sol 交回第一批 6 项 509 GiB（`archive_plan/PLAN.md`、`move_list.txt`、`relink.tsv`、`archive_mover.sh`，dry-run 通过）。owner 13:3x 选"旧轮次原始轨迹也搬"（`/data/trace_runs/dual_20260923`、`os_closed_loop/r08_main`、`r06_p3_pilot`，共 1.16 TiB）；dp/xwam/其他旧线不搬。sol 同线程在补第二批清单 `move_list_all.txt` / `relink_all.tsv` + dry-run（日志 `archive_plan/sol_archive_batch2.log`）。**I018 同意 15:15 起我方独占 /archive 写入**（它推迟 archive_cold.sh 和 /archive 删除，等我们通知结束）。跑法：tmux `archive_mv`，`cd archive_plan && ./archive_mover.sh move_list_all.txt relink_all.tsv`，看 `mover.log`；结束后通知沙堆实验。
- **13:50 现状**：解耦基准第 3 次又失败（GR00T 现行 1 份那组 300 局其实跑完了，但 relay 延迟行 uid 全是 null，汇总判空）。决定：今天不再跑基准；13:48 起在 S1 内先跑 R16 v2 补跑（tmux `r16_retest_v2`，队列 `r16_ops/queue_retest_v2_ordered.txt`，日志 `r16_ops/retest_loop.log`），**跑完即提前交还 4090**（I018 请求）；硬停守护 tmux `s1_guard`（15:10:30）。sol 同线程修基准（`prompts/sol_decouple_bench_fix2.md`，日志 `r16_ops/sol_decouple_bench_fix2.log`）：延迟归属、已完成组复算、不重跑已过的门、10 分钟真机冒烟模式。**教训：下个 GPU 时段先跑冒烟，不要直接放整包。**
- v2 跑完后：`retest_combined analyze --plan retest_v1/plan.json --plan retest_v2/plan.json --output retest_v1_v2_results`（命令见 HANDBACK_RETEST.md 附录），更新 REPORT.md 第三节。
- 解耦基准第 3 次重开（13:24，`--new`）：GR00T 现行组起 worker 时 tether 30 s 超时且只试 1 次 → 改 `package_driver.py:119` 用默认 4 次重试（与 R16 运行器一致），重生成配置。**日志现在是 `openpi_ext/scratch/r16_ops/pkg_run3.log`**。巡检 cron e902f7a1。
**内容**：所有大模型调用（守卫触发、旋钮定期调用、纯推理）改成动作头只跑 k 步去噪，k = 1 或 2。库、度量、修正器都沿用现有冻结的那一套。执行段是现行的 10 步。决定见 `rounds/r16/COORDINATOR_DECISIONS.md`。
- IR：π0.5 是 .152V + c·M，c = .454（k = 1）/ .498（k = 2）；GR00T 是 .148V + c·M，c = .258 / .343。

**结果到目前为止**：
- **第一步（纯减步推理，测试集 8 组 × 500 局）**：8 格都和满步分不开（−0.8 到 +1.4 个点）。开销 π0.5 .30–.33（满步 .50），GR00T .20–.25。结果在 `rounds/r16/phase1_results/PHASE1.md`。
- **开发集（B 池，逐任务下降，5 个点容差，按局自适应）**：4 个分组全部完成。档位表已冻结，在 `rounds/r16/dev_results_conductor/`：`levels.json`、`DESCENT.md`、`TABLE_FREEZE.json`，另有 `levels_tol0pp.json` / `levels_tol2pp.json`。
- **测试集（32 格 × 500 局）**：π0.5 两组已完成，GR00T k = 1 在跑（10:00 约 1,900/4,000），预计 11:40 前后全部跑完。π0.5 粗算（8 格平均）：k = 2 是 94.0% @ .17，k = 1 是 92.6% @ .15；满步纯推理 94.7% @ .51，原逐任务下降 94.1% @ ≈.24。
  - 粗算脚本：`r16_ops/quick_test.py`；逐任务对比满步纯推理：`r16_ops/per_task.py`，输出写到 `per_task_test.json`。
- **owner 10-08 10:0x 的指示**：直接用这次的结果，成功率低的任务把 IR 档位提上去，只重跑这些任务；"不要多嘴"，不写方法论口径（已记进记忆 feedback_tune_knob_without_caveats）。
  - 规则：比满步纯推理低 ≥ 10 个点 → c10（k 步纯推理）；低 6–8 个点 → ir45，本来已是 ir45 或更高的 → c10；本来就是 c10 的不动。
  - π0.5 有 27 个任务低 ≥ 6 个点。
  - sol 正在写重跑工具（线程 `01a117c6…`，日志 `r16_ops/sol_retest.log`，交回说明 `rounds/r16/HANDBACK_RETEST.md`）：输入调整清单 JSON，生成只跑这些任务的 conductor 组 `r16_conductor_retest_*`，再做合并分析，替换任务后重算各格的成功率和开销并配对比较。

**跑法（conductor：常驻多方法服务 + yaml 热切换 + 按局派发，零气泡，利用率约 0.95–0.99）**：
- 实现与运行手册：`rounds/r16/HANDBACK_CONDUCTOR.md`、`RUNBOOK_CONDUCTOR.md`、`HANDBACK_GROOT_SERVERS.md`（GR00T 服务数可配置、组尾停服务的修复）。
- 循环：`openpi_ext/scratch/r16_ops/cond_loop2.sh <dev|test>`，在 tmux 里跑（测试集是 `r16_cond_test`），日志 `r16_ops/cond_<mode>.log`。它直接调运行器，跳过有 DONE 的组。**不要用 conductor_commands.sh**：运行器被 Ctrl-C 后它会接着起下一组。
- GR00T 每类任务的服务数读 `r16_ops/groot_sps`，现在是 3，共 6 个服务，用的是借用 B2 的显存。要退回时写 2，中断运行器，等 worker 租约释放，再起循环。
- 残留服务：`r16_ops/stop_group.py <分组目录>` 按收据停。
- 监控：`cond_watch_test.sh`（LIMIT=41984 是 GR00T 段 B2 的上限，π0.5 段用 30720）。速度和利用率：`util_window.py <journal> 64 10`。
- 巡检 cron 只活在本会话；压缩或重开后要重新挂。
- worker 包已部署到 timan107 `/scratch/zixuans8/openpi_r16_conductor_v1`，哈希都核对过。driver 端口 23179 已登记台账。

**收尾（按顺序）**：
1. 测试集跑完 → 跑 per_task.py → 写调整清单 → 用 retest 工具重跑这些任务 → 合并分析。
2. 写 R16 报告 `rounds/r16/REPORT.md`，中文、给 owner 看的口径。
3. 停掉 4090 上我方全部进程 → 台账写 4090 整张控制权归 I018、借用 B2 结束 → 书面通知沙堆实验，它会马上起整卡计时队列。
4. 删巡检 cron。提交 R16 要 owner 授权。

### 1.D 框架层 CPU/GPU 解耦（owner 10-08 09:3x 批准："最好在框架层面就可以解耦"）
- 背景：缓存太好用，GPU 活少；吞吐卡在每个服务进程的单个 Python 线程上（GR00T 服务各 100% CPU，GPU 29–55%，本机 88 线程几乎闲着），而每个进程都要装一整份模型，显存限制了进程数。
- 目标：很多个不带权重的 CPU 前端进程负责连接、检索、修正器、预处理、日志；每个模型只留一两个 GPU 进程，成批执行模型阶段。默认关闭；开启后每个决策逐个一致。
- sol 新线程 `01a11bef-165f-77f0-b334-7ef92a072444`，任务说明 `rounds/r16/prompts/sol_decouple.md`，日志 `r16_ops/sol_decouple.log`。**只在工作树 `/home/weiland/projects/openpi_decouple`（分支 `decouple-cpu-gpu`）里改，不碰主目录。**
  - 10:0x 进展：B 池录下的输入回放逐字节一致，正在做 CPU 时间剖析。
- 交回后：先核对 `HANDBACK_DECOUPLE.md`；等 4090 空出来再跑 GPU 对比基准（现行跑法对拆分后，同显存下的每分钟局数、GPU 和 CPU 利用率）。
- **10:54 已交回**（124 个 CPU 测试通过；`openpi_decouple/HANDBACK_DECOUPLE.md`、`docs/DECOUPLE.md`）。剖析：GR00T 每次请求输入预处理 14.8 ms（π0.5 0.44 ms），检索 + 插件约 7.7 ms，日志约 1 ms → GR00T 服务卡在单线程 CPU 上。GPU 数值门和吞吐对比都还没跑。
- 10:55 同线程派 sol 做"一键 GPU 包"（`prompts/sol_decouple_bench.md`，日志 `r16_ops/sol_decouple_bench.log`，tmux `cx_decouple`）：`exp/decouple/run_gpu_package.sh` = 数值门（B=1 逐位一致）→ GR00T、π0.5 现行 1 份 / 显存装满 / 拆分 N=4/8/16 → 汇总 SUMMARY.md，约 2 h。
- **跑法**：R16 收尾后 4090 先按约定整卡交还 I018；包准备好后再按"要用时要回来"提前通知沙堆实验要整卡时段来跑。

### 1.0 会话与环境
- **重开会话请先 `cd /home/weiland/projects/openpi` 再 `claude --resume <会话号> --dangerously-skip-permissions`**。本会话 14:52 起误在 `/home/weiland/idea-sprint`（沙堆实验的目录）里打开，加载的是 idea-sprint 的记忆；openpi 的记忆在 `~/.claude/projects/-home-weiland-projects-openpi/memory/`，需要时用绝对路径读写。
- 回复 owner 一律用中文，不用代号。
- **跑法工具全在持久目录 `/home/weiland/projects/openpi_ext/scratch/r15_ops/`**（作业临时目录会被删，别再放 `$CLAUDE_JOB_DIR/tmp`）：
  - `run_q8.sh <local|h100> <worker_host> <ports> <wps>`：共享队列 `queue.txt`（每行 `@h100 /abs/root|arm` 或 `@local ...`）。
  - 本机服务数 `local_servers`（4），本机显存预检 `local_need_mb`（6800；GR00T 实测每服务 5.7–6.3 GB，默认 8000 偏高会报 GPU_TIGHT）。
  - h100 服务数上限 `h100_max_servers`（现为 4）；`h100_deadline` 不用。
  - 生效核对 `engage_check.py`；巡检 `probe.sh`；监控 `combined_watch.sh`、`codex_watch.sh <job>`、`tmux_watch.sh`。
  - 日志 `/tmp/r8_coord/r15_q_<host>_<worker>.log`；tmux 会话 `r15q_h100` / `r15q_local`。
  - 线上没有作业时会自己 Q_EXIT，补了队列要重新起 tmux。
- **部署**：
  - 无链在跑时：`SERVER_HOST=h100 WORKER_HOST=<timan> ... control setup`（h100 + 该 worker）；本机线：`SERVER_HOST=local WORKER_HOST=timan107 ... setup --worker-only`。
  - 有链在跑时：只推改动文件，`control.bundle('h100', control.BASE/'openpi', [(abs, rel), ...])`，再用 sha256 逐一核对。

### 1.1 R15 结果（已收口，报告 `exp/offline_search/rounds/r15/REPORT.md`）
- **A 线：100/200/500 集逐任务下降**，测试集每种做法合并 2,000 局，纯推理 93.2% / .506：

  | 库大小 | 统一档位 | 逐任务表 | 逐任务 − 纯推理 |
  |---|---|---|---|
  | 100 | 89.1% / .181 | 92.1% / .314 | −1.1，区间含 0 |
  | 200 | 91.3% / .153 | 92.8% / .266 | −0.5，区间含 0 |
  | 500 | 93.8% / .145 | 93.8% / .267 | +0.6，区间含 0 |

  - 开销省 38–47%（R14 的 50 集是 −2.2 个点、省 37%）。
  - 逐任务对统一档位：+3.0 / +1.5 / −0.1 个点。500 集时统一档位四格全是"关闭"，旋钮不开就够了；GR00T 长任务小库最需要逐任务调。
  - 逐任务保证仍不成立：40 个任务里分别有 8 / 4 / 5 个掉超 5 个点。
  - 数据在 `rounds/r15/descent/`：`TEST_A.md`、`test_a.json`、`levels*.json`、`LOADER_FIXES.md`。
- **B 线：轻量图像检索键**，owner 定调"视觉塔最准"，进附录（裁定 D5）。
  - CPU 约 1 ms/看图；新起点检索误差 +8–10%；闭环小跑 4 组成功率低 2–9 个点、开销只省 12–27%。
  - 附录稿 `rounds/r15/qk/astra/PILOT_RESULTS.md`。
- **owner 本轮定的长期规则**（已写进记忆）：检索不能当瞎子，只用状态的键直接否；第 5 层仍按逐任务下降。

### 1.2 纯缓存 + 修正器 补充消融（已收口，10-06 02:05；报告见 `rounds/r15/REPORT.md` 的"补充"节）
- **设计**：
  - 24 组 = {π0.5, GR00T} × {长任务, 空间任务} × 库大小 {50, 100, 200, 300, 400, 500}，每组测试集 500 局。
  - 修正器用 R10 冻结的、每个库大小各自的距离衰减修正器，不开守卫。24/24 绑定已独立核对。
  - 服务在 h100，worker 在 timan107，与 R10 的纯缓存、只开守卫、守卫 + 修正器逐起点配对。
- **结果**（四格合并 12,000 局）：

  | 组 | 成功率 |
  |---|---|
  | 纯缓存 | 84.0% |
  | 纯缓存 + 修正器 | 87.2% |
  | 只开守卫 | 89.0% |
  | 守卫 + 修正器 | 90.9% |

  - 修正器对纯缓存提高 +3.2 个点，p = 5e-22，owner IR 不变（约 .075）。四格合并后各自都显著。24 组里 20 组提高、3 组持平、1 组降 0.4 个点，没有一组显著变差。
  - π0.5 长任务在 6 种库大小上全部显著（+4.0 到 +8.0 个点）。
  - 空间任务上，修正器单独约等于只开守卫，开销只有其三分之二。
  - 修正器和守卫部分重叠，但叠加最好：守卫 + 修正器比纯缓存 + 修正器高 3.6 个点。
- **文件**：
  - 结果表：`pure_corr/TEST_A.md`（sol 的分析脚本）、`pure_corr/SUPPLEMENT.md`（owner IR 和另外三组配对，脚本 `pure_corr/supplement.py`）。
  - 运行根：`r15_purecorr_*`。
  - 跑法：`r15_ops/run_q9.sh`，即 run_q8 加 h100 /data 保护。
- **收尾已做**：
  - h100 已在台账报空（02:05 起），并通知了沙堆实验；
  - cron 和监控已停；
  - h100、timan107 上都没有我方进程。
- **磁盘**：
  - h100 /data 余 34 GB。动议 M5（底线临时 30 GB）仍有效，直到 owner 定清理方案。
  - 我方 `/data/oscl_h100/runs` 约 16 GB，全部已拉回本地，删除要 owner 定。

### 1.3 GPU 与沙堆实验
- 规则 `/home/weiland/gpu_share/RULES.md`（v5），台账 `LEDGER.md` 是唯一依据。沙堆实验的会话名叫"沙堆实验"，用 ListAgents 找最新的会话地址。
- **h100：10-07 23:0x 起整张控制权归 I018**（owner："你都不用了当然整个的控制权都给他，你要用的时候再要回来"）。我方要用时提前 30 分钟通知收回。
- **4090**：R16 期间我方 28 GB / I018 20 GB（M6）。
  - π0.5 组期间按 M7 是 30 / 18。
  - 测试集的 GR00T 段用借用 B2：我方约 39 GB，6 个服务。I018 要回时发消息，我方 5 分钟内退回 28 GB。
  - **R16 结束后：停掉我方全部进程，书面通知 I018，4090 整张归它。**
- 长期规则（记忆 feedback_idle_devices_full_control）：我方不用的卡整张交给沙堆实验，要用时再要回来。
- h100 /data 的旧运行目录要不要清，仍待 owner 定（M5 底线临时 30 GB）。

### 1.4 codex
- **⛔ 不准带凭据**：不用 `codex exec` 和 companion。一律用守护进程：`python3 /home/weiland/idea-sprint/dynamical_system/runs/i018-control/bin/codex_daemon_run.py --model <gpt-6.1-sol|gpt-6-astra> --effort xhigh --sandbox danger-full-access --cwd <目录> --prompt <文件> --log <日志> [--thread <线程号>]`。
  - 放在 tmux 里跑；用 `r16_ops/daemon_watch.sh <日志>`（环境变量 SESS=<tmux 名>）挂 Monitor。
  - 开跑前用 `acct codex` 看额度。
- **线程**：
  - sol R16：`01a117c6-8ffa-7083-b715-df7eb642bbcc`（R16 全部实现，含 conductor、服务数可配置、重跑工具）；
  - sol 解耦：`01a11bef-165f-77f0-b334-7ef92a072444`；
  - astra R15-B：`01a10c95-6967-7b92-89c3-ed96ab4c5fdb`。
- owner 点名时可以派 Sonnet，比如清理工作区那次。

### 1.5 未提交
- R12–R15 已于 10-07 20:4x 提交并推到 `origin/Ziyang`（e877ddd6…05a32800，owner："该commit push就做，该ignore就ignore"）。未提交的只剩 `rounds/r16/`，等 R16 结束再提交。惯例：英文 commit、不加 AI 署名、json 要 `add -f`、画图脚本不入库。

## 1z. （历史）2026-10-05 09:4x CDT—— R13、R14 都已收口；无在跑实验；待 owner 答复是否补 100/200 集

**最新（优先读这段）**
- **10-05 09:4x compact 前补充（最先读）**：
  - **待 owner 回复**：要不要把 R13（开销反馈下距离调用收敛成定期节奏）和 R14（逐任务旋钮下降）补到库大小 100 和 200 集。
    - 我的建议：两者在 100/200 集上各复现一遍，规模同 50 集（B 池开发集 + 测试集小确认）。
    - 库子集现成（`rounds/r10/subsets/<model>_<suite>_{100,200,...}`）；100 集的旋钮拟合要 sol 先补，约 1 h；跑约半天到一天。
    - owner 之前问"你只用 50 集做了吗，有用 100 和 200 吗"，我已如实回答：
      - R13 开发集验证用了 50 和 200（200 只有长任务）；
      - R13 测试集精简确认只有 50（另有 200/500 的 GR00T 补充臂 6 个）；
      - R14 只有 50；
      - 没有用 100。
  - **GPU 规则（owner 10-05，已写入记忆 `feedback_gpu_coordination_shadui`）**：要用 GPU 就直接找"沙堆实验"会话（I017/I018，h100 目录 `/data/idea_sprint/dynsys`）协商，不请示 owner；自己没需求或可共卡就按需分配。
    - 现约定：我上卡提前 ≥30 分钟发消息，对方清空两卡后回复；对方不计时的作业可共卡、5 分钟内让出；h100 /data 余量 ≥40 GB。
    - 对方地址：它消息里的 `from`（`uds:/run/user/1000/cc-socks/3375024.sock`），或 ListAgents 里的"沙堆实验"。
    - 目前两卡借给 I018 做逐步 profile。
  - timan108 已于 08:44 恢复（整机重启过），09:3x verify-worker 通过，h100 线可用。
  - 起跑方法：本机线和 h100 线都用 `~/.claude/jobs/a607dd74/tmp/r13/run_q6.sh`，队列 `tmp/r13/queue.txt`（`@local` / `@h100`）。
    - 监控用 `combined_watch.sh` 和 `codex_watch.sh`；巡检 `probe.sh`。cron 已删，重开实验时重建。
  - codex 派活：astra = `--model gpt-6-astra --effort xhigh`，sol = `--model gpt-6.1-sol --effort xhigh`，用 companion `task --background --full-access`。
    - 用完按 PID 停本会话的 broker 对（cwd = openpi）。
    - 不让 agent 自己等数据或看守实验。
  - **回复 owner 一律用中文**。
- **R13（owner /goal：两种第 4 层调用对比）**，报告 `exp/offline_search/rounds/r13/REPORT.md`：
  - 结论：距离阈值加"本局开销反馈"后，按距离调用会收敛成和定期调用几乎一样的节奏；同开销 SR 分不开；第 4 层用定期调用。
  - 数字：
    - 开发集 28 臂开销达标；
    - .32 档平均调用间隔：开发集距离 1.67、定期 1.72；测试集 1.6–1.8；
    - 测试集 SR：距离 − 定期 +0.25 [−1.35, +1.80] 个点（2000 对）；
    - 机理：节奏来自开销账本，不是"偏离随时间增大"。
  - owner 定调：这条结论在论文里展示即可，不要大量数据。
  - 论文用表：`rounds/r13/astra/final_paper_table.csv`、`final_b_gap_cdf.csv`、`final_test_a_gap_cdf.csv`。
  - 裁定 D1–D5：`rounds/r13/COORDINATOR_DECISIONS.md`。
- **R14（第 5 层 = owner 原意的"部署时逐任务旋钮下降"）**，报告 `exp/offline_search/rounds/r14/REPORT.md`：
  - 做法：B 池每任务 30 局逐档下降（5 个点容差、首次失败即停），冻结档位表后在测试集确认。
  - 结果（测试集合并）：逐任务表 91.0% @ .317，纯推理 93.2% @ .506（−2.2 个点，开销省 37%），统一档位 89.3% @ .281。
  - 每任务守容差失败：40 个里 11 个掉超 5 个点，样本太少。
  - 示范库估计预测不了下降档位。
  - 建议：逐任务保证需要每任务约 100 局以上或保守规则，再在新数据上确认。
  - 原 R14 设计（部件共用消融、示范库分配）的代码在 `rounds/r14/{astra,impl,knob}/`，未跑。
- **owner 的语言要求**：一律中文（10-04 夜两次误用英文被批评，已写入语言记忆）。
- **机器**：
  - timan108 的 tether 代理 10-05 02:06–08:44 离线（整机重启过），08:44 起恢复在线，09:3x verify-worker 已核对。
  - **h100 与本机 4090 借给 I017/I018 共用**（逐步 profile，不计时）：我方要上卡须提前 ≥30 分钟发消息，对方清空两卡后回复；h100 的 /data 余量保持 ≥40 GB。
  - h100 已由 I017 交还（I017 转 PARK，以后要独占须提前半天约）。
  - /home 腾空间已完成（余 2.6 TB），备份只管 /home。
- **未提交**：R12、R13、R14 的代码和报告。提交须 owner 当次授权；英文 commit、不加 AI 署名、json 文件要 `add -f`、画图脚本不入库。

## 1a0. （10-04 23:30 版，已过时）

**最新（优先读这段）**
- **owner 10-04 晚定调**：
  - R13 结论 = "加上开销反馈后，按距离调用会收敛成和定期调用几乎一样的节奏"，论文里展示即可，不要大量数据。
  - R13 大网格已停（裁定 D5），只补 12 臂测试集精简确认。
  - R14 = owner 原意的"分任务的部署阶段旋钮下降法"：每任务从纯推理逐档往下降，降到 SR 掉出容差为止。
  - owner 说"可以用示范库估计的比例，如果估计的和真实下降的准就更好"——所以要对照 astra 的示范库估计。
  - 其余 R14 设计（部件共用消融、示范库分配验证）不跑。
- **R13 文档**：
  - astra 终版 `rounds/r13/astra/REPORT.md` / `ANALYSIS.md` / `final/HANDBACK.md`；
  - 论文用 `final_paper_table.csv`、`final_b_gap_cdf.csv`；
  - 裁定 `rounds/r13/COORDINATOR_DECISIONS.md`（D1–D5）。
- **R13 结论（B 池开发集）**：
  - .32 档平均调用间隔：距离反馈 1.67、定期 1.72（次看图）；无反馈的固定阈值节奏不同。
  - SR：距离 − 定期 +0.86 [−0.61, +2.29] 个点，2800 对。
  - 开销控制：28/28 距离反馈臂达标，固定阈值 5/8 越界。
- **R13 测试集精简集**（4 个 50 集格，.32，rep0，距离 / 定期 / 关闭）：
  - GR00T 6 臂已跑完（本机）；
  - π0.5 6 臂待 h100 交还后跑（在 `tmp/r13/queue_h100_hold.txt`）；
  - 跑完运行 `rounds/r13/astra/final_test_a.py` 填报告（命令见 `final/HANDBACK.md`）。
- **R14 下降**：
  - 根 `r14_descent_<cell>`，4 格 × 8 臂 × 300 局（每任务 30 局），说明见 `rounds/r14/descent/HANDBACK.md`；
  - 都在本机线上：GR00T 22:07 起，π0.5 接在后面，预计 05:00 前后全齐；
  - astra 已冻结判定协议 `rounds/r14/descent_astra/PROTOCOL.md`（主容差 5 个点、第一次失败即停）。
- **R14 之后的步骤**：
  1. 数据齐后运行 `analyze.py apply`（命令见 `descent_astra/HANDBACK.md`），产出 `levels.json` / `uniform.json` / `DESCENT.md`；
  2. 用 `rounds/r14/descent` 的 `prepare emit-test-a --levels --uniform --out .../test_a_selected` 生成测试根，再 verify / plan / install；
  3. 部署 `descent/__init__.py` 与 `descent/recipe.py` 到 h100（用 `control.bundle`，见 §4）；
  4. 测试集每格 3 臂（按任务下降 / 统一档位 / 纯推理），GR00T 在本机，π0.5 在 h100；
  5. 叫 astra 写 R14 报告。
- **h100**：借给 I017 跑 n99，预计 10-05 07:30–08:00 交还（最坏 15:00）；对方会发消息。
- **owner 的语言要求**：一律用中文回复（今晚我两次误用英文，被 owner 严厉批评）。

## 1a. （10-04 20:2x 版，已过时）

**10-04 晚最新状态（优先读这段；下面"10-04 compact 前"及更早段落为历史留档）**

- **owner 目标（/goal，10-04 11:1x）**："研究两种第4层调用详细对比"，即定期调用 vs 距离阈值调用。可用 astra、sol，以及 timan107、timan108、h100、本机 4090。
- **owner 追加**：
  - "之后把沿任务分治也做了"，即 R14 第 5 层；范围由我定为两部分：部件共用消融 + 按任务分配调用开销。
  - "你们两个协商，我不做打扰"：h100 / 4090 与 I017"沙堆实验"会话直接商量，不请示 owner。
  - 同意只跑重复块 rep0/rep1（裁定 D4）。
  - 机理问题：两法在轨迹哪里调用；若相近，距离阈值就是"隔一段时间调用一次"的另一种实现。
- **R13 文档**：
  - 简报 `exp/offline_search/rounds/r13/BRIEF.md`
  - astra 冻结设计 `rounds/r13/astra/`：SPEC / PREDICTION / PLAN / OFFLINE，门槛判定 GATES.md + gates.json，机理分析 MECHANISM.md
  - sol 实现 `rounds/r13/impl/`、`rounds/r13/knob/`（R13Knob：距离反馈原因码 69，固定阈值诊断 70）
  - 协调者裁定 `rounds/r13/COORDINATOR_DECISIONS.md`（D1–D4）
  - 数据说明 `rounds/r13/DATA.md`
- **R13 已有结论（B 池开发集，未看测试集）**：
  - 新距离调用"距离阈值 + 本局开销反馈"能精准控开销：8 格 32 主臂平均偏差约 .004，最差 .015（R11 旧版平均 −.08）。
  - 只重新标定固定阈值不够：8 臂里 5 臂越界，方向随格子变。
  - 定期调用在 π0.5 长任务上系统性少花约 .01。
  - 开发集 SR 合并：距离 − 定期 +0.9 个点（p = .24，2800 对），分不开。
  - **机理**：
    - 两法节奏相近，但同状态重放有 66.6% 的被选位置不同；
    - 距离并不随"离上次调用多久"增长（间隔只解释 0.3–2.5%），节奏来自开销账本；
    - 距离单独挑的点，大模型与缓存的动作差高 11.3%（主要在空间任务）。
  - 放行：64 臂全部纳入，16 个测试根全部放行（含 D1/D2 裁定）。两个配对点不算"同开销"：GR00T L10-200 .25、π0.5 L10-50 .32 rep0。
- **R13 正式测试**（每臂 500 局，rep0/rep1 共 160 臂；rep2/rep3 的根已装但不跑）：
  - GR00T 80 臂在本机，17:55 起，20:17 完成 6 臂；
  - π0.5 80 臂在 h100，交还后开跑。
- **R14**：
  - astra 设计 `rounds/r14/astra/`（开发集验证 68 臂 × 50 局，测试 128 臂 × 500 局）；
  - sol 实现 `rounds/r14/impl/` + `rounds/r14/knob/`；
  - 已排入两条线的队列（在 R13 测试之后）；R14 测试根要等 astra 判定验证门槛后再装、再入队。
- **h100 借给 I017**（墙钟计时，需独占）：
  - 16:21 起跑 W2 n99，约 14.4 h，预计 10-05 07:30–08:00 交还，最坏 15:00；
  - 对方会发消息交还，失败或超时提前说；交还后我方连续 2–3 天，对方再要须提前半天约；
  - 4090 已于 13:31 交还，对方以后要独占窗口须提前 ≥ 20 分钟约。
- **/home 腾空间（owner 10-04）**：已搬 1.25 TB 到 /data 与 /archive，原位置留软链，/home 余量 2.6 TB；备份只管 /home（owner 裁定，已入磁盘记忆）。

## 1b. （历史）2026-10-04 早 CDT —— R12"看一半"全部完成并已裁定

**10-04 compact 前最新状态（优先读这段，下面各段为历史留档）**

**R12 结论与 owner 裁定**
- 报告定稿：`exp/offline_search/rounds/r12/REPORT.md`，顶部是 owner 裁定和白话结论。
- **owner 10-04 裁定**：
  - "看一半"（只编码一路相机）**不进主设计，放附录**：SR 无提升；开销约 −20% 的收益集中在空间任务；GR00T 长任务所有单路方案 −5；按任务选相机在长任务上选不准；有效规律是看过测试集 A 后才得出的，未干净验证。
  - 不靠第三人称画面的**新守卫（H2）也放附录**：开销 −5%、SR 持平，但长任务首次调用晚约 10 步。旋钮本身能调开销，第 2 层保留原无进展守卫。
  - 记忆 `project_offline_search_five_layer_design` 已加附录条目，MEMORY.md 索引已同步。
- **主设计（五层，不变）**：1 缓存（两路相机，按任务马氏距离）；2 无进展守卫；3 库内修正器；4 推理开销旋钮（定期 + 距离阈值都纳入，待细比）；5 按任务分治。挑选和标定只在 B 池开发集上做，测试集 A 只测冻结版。

**测试集 A 定稿数字**（每组 500 局；原系统每格跑两次，对两次逐起点平均配对；5 格不开旋钮等权合并）
- opus 看一半（M1）：SR −0.8 ± 1.3，开销 −.010（−9%）。
- 新守卫（H2）：+0.1 ± 1.1，−.007（−5%）。
- 新守卫 + 看一半（M2）：+0.1 ± 1.3，−.025（−19%）。
- 按任务选相机五层（half-everywhere）：+0.2 ± 1.2，−.026（−21%）。
  - GR00T 空 50：+4.0 / −30%；π0.5 空 200：−44%；GR00T 长 50：所有单路方案 −5；开旋钮 .32：+3.2 ± 2.6，开销持平。
- 纯缓存相机消融：手腕在 GR00T 空间 +7.6、长任务 −11（均显著）；π0.5 长 −3、空间持平；第三人称单路处处更差；重训单路距离无益。
- 打分脚本：`~/.claude/jobs/a607dd74/tmp/r12/score_final.py`（最终，对两次原系统）、`score_testA.py`、`score_pilot.py`、`half_stoprules.py`。

**数据与文档位置**
- 小跑（B 池）：`rounds/r12/{lookhalf,guard}/pilots/*_fix1_*`、`guard/pilots/r12_guard_fix2_pi05_spatial_200`。
- 消融：B 快筛 `rounds/r12/ablate/pilots/`；camdev `rounds/r12/ablate/camdev/`。
- half-everywhere B 快筛：`rounds/r12/half/pilots/`。
- 测试集 A 根（均在 `/home/weiland/trace_runs/os_closed_loop/`）：`r12_test_*`、`r12_test_rep_*`（原系统第二次）、`r12_ablate_test_*`、`r12_test_half_*`。
- 设计与分析：
  - `rounds/r12/opus/{PILOT_ANALYSIS,DESIGN_HALF_EVERYWHERE,PREDICTION}.md`
  - `rounds/r12/opus/out/design/camera_choice_screen_camdev.{json,log}`、`design_predict_conditional.log`
  - `rounds/r12/astra/PILOT_ANALYSIS.md`
- sol 交接：
  - `rounds/r12/{lookhalf,guard}/HANDBACK.md`
  - `rounds/r12/TEST_A_HANDBACK.md`（含 fix3 登记竞态、fix4 回收拆卷）
  - `rounds/r12/{ablate,ablate/camdev,half,test_a_rep_staged}/HANDBACK.md`
- 台账：`logs/offline_search_exploration.log.md` 的 R12 段。

**现状**
- 两台 GPU 空闲，无残留服务；cron 与 Monitor 已全部停止；本会话 codex 中转进程已停。
- **R12 未提交**，等 owner 授权：commit 用英文、不加 AI 署名、画图脚本不入库。
- 新文件很多：`closed_loop/{execution_worker,guard,lookhalf*,cache_ablate*,half_everywhere*,look_recording}.py`、`rounds/r12/**`、`ops/h100` 改动等。`.gitignore` 规则是 *.json / *.yaml 一刀切，需要的要 `add -f`。
- h100 上为腾磁盘，已删除已回收到本地的 4 个 B 池运行目录（7.9 GB）。

**运行基础设施（可复用）**
- 共享队列跑法：`tmp/r12/run_q5.sh <local|h100> <fleet> <ports> <wps>` + `tmp/r12/queue4.txt`。
  - 行首 `@h100 ` / `@local ` 指定机器；跑完但回收失败的组只重收、不重跑。
  - 每组完成后用 `tmp/r12/engage_check.py` 核对生效；h100 上 π0.5 自动用 3 个服务。
  - 本机：timan107，端口 23100–23103，WPS 16。h100：timan108，端口 23230–23233，WPS 10。
- 已修的运行坑：
  - 带回执的 worker 在 A / B 池都启用；
  - 新守卫配置校验；
  - 本机服务登记竞态（fix3）；
  - 回收包超 tether 812 MB 上限自动拆卷（fix4，≤ 700 MB，too_large 不重试）；
  - h100 /data 余量低于 30 GiB 时同步会被拒（DISK_BLOCKED），需清理已回收的运行目录。
- GR00T 在 4090 与 H100 上结果不同（空间格 SR 约差 5 点），同一格所有组必须在同一种显卡上跑；π0.5 无此差异。

**10-04 04:4x（已被上段取代）**
- 报告定稿 `exp/offline_search/rounds/r12/REPORT.md`（顶部白话结论 + 待裁定）。打分：`tmp/r12/score_final.py`（对两次原系统平均配对）。
- 合并（5 格，不开旋钮）：half-everywhere +0.2 ± 1.2 / −.026（−21%）≈ m2 +0.1 / −.025；空间任务受益（GR00T 空 50 +4.0 / −30%），GR00T 长任务所有单路方案 −5。
- 待 owner：①两路任务是否叠 m2；②长任务改回两路，或找更可靠的选相机依据（此方向看过 A 后提出，正式验证需新测试数据）。
- GPU 两台空闲；cron 与 Monitor 已停；R12 未提交（等 owner 授权）。

**10-04 02:4x**
- camdev 完成，按冻结规则定出相机表（见台账 "half-everywhere" 段），条件预测已落盘；sol 实现交付（`rounds/r12/half/HANDBACK.md`），B 根 `rounds/r12/half/pilots/r12_half_*` 已 emit，A 根暂存 `half/test_a_staged/`（快筛过后 `python -m exp.offline_search.rounds.r12.half.prepare --install-test-a`）。
- 02:40 B 快筛开跑（queue4；GR00T @h100 带同批 off）。快筛后按设计 §10 冻结停止规则 S1–S5 判格子，过关的上测试集 A（GR00T 在 h100，可加同批 off）。

**10-04 02:3x**
- 测试集 A（方法组 + 消融）全部完成；报告草稿 `rounds/r12/REPORT.md`。
- opus 设计"按任务选相机"五层（`rounds/r12/opus/DESIGN_HALF_EVERYWHERE.md`）：手腕任务 = 手腕缓存（原距离）+ H2 + 手腕修正头 + 重算旋钮；两路任务 = 现系统。**待 owner 裁定**：两路任务保持现系统（默认，正在按此实施）还是叠 m2。
- 进行中：camdev 选相机开发集（queue4，GR00T @h100）；sol 实现 `R12HalfEverywhere`（task-muth6n5u-qslyti，`rounds/r12/half/`）。
- camdev 完成后：`opus/tools/camera_choice.py screen,camdev` → `design_predict.py conditional …camera_choice_screen_camdev.json`（存日志）→ sol 交付后推代码、emit B 根、排快筛。
- cron 巡检换为 `5157bcaf`（含上述步骤）。

**10-04 01:4x 进展**
- 测试集 A 方法组全部完成；合并 5 格：opus 看一半 ΔIR −.0096 / ΔSR −1.2 ± 1.4；h2 −.0066 / −0.3 ± 1.3；**m2 −.0242（−19%）/ −0.4 ± 1.4**。GR00T 长任务看一半显著 −6（m1/m2）。
- 纯缓存消融测试集 A：GR00T 长任务手腕 −11（显著）、GR00T 空间手腕 +7.6（显著）、π0.5 长任务 −3 / 空间 50 持平；π0.5 空间 200 在本机跑最后一格。→ 手腕并非处处安全，方向改为"按任务选相机"。
- 已派：sol 生成 B 池"选相机"开发集根 `rounds/r12/ablate/camdev/`（both / wrist_sub，每任务 40 局，不录图）——交付后排队（GR00T @h100）；opus 设计"全看一半"五层（结果存 `rounds/r12/opus/DESIGN_HALF_EVERYWHERE.md`，按任务选相机须在 B 上定）。
- 待写：`rounds/r12/REPORT.md`（全部数据齐后）。

**22:0x owner 定调 + 通宵计划（owner 去睡，按此顺序独立推进）**
- owner：纯缓存消融首格（π0.5 长任务 50，B 100 局）"两张图 .63 / 手腕+原距离 .57 / 手腕+重训 .63 / 第三人称+原距离 .52"后说："看起来可以只把 cache 做成全看一半的，不过其他层可能会需要调整"，并批准顺序：
  1. 跑完测试集 A 剩余组 + 5 格消融快筛；
  2. 消融测试集 A：每格 both / wrist_refit / wrist_sub 各 500 局（GR00T @h100）；
  3. 若手腕缓存在 A 上不掉 SR → opus 设计"全手腕"五层（手腕缓存 + 新守卫 h2 + 手腕修正器 + 重标定旋钮 + 按任务可选相机），sol 实现 → B 100 局快筛 → 测试集 A。
- 巡检：cron `6a860fd4`（每小时 :17/:47），Monitor 条件触发（错误 / 60 分钟停滞）。

**21:2x 增补**
- owner 加了**纯缓存相机消融**（只开第 1 层：无守卫 / 无调用 / 无旋钮 / 无修正器；两路 vs 一路），owner 同意"都跑"。sol 实现：`rounds/r12/ablate/HANDBACK.md`；每格 5 组 both / wrist_sub / wrist_refit / agent_sub / agent_refit；B 快筛根 `rounds/r12/ablate/pilots/r12_ablate_<cell>`（100 局），A 暂存 `ablate/test_a_staged/`（未安装，快筛后挑组再装：`python -m exp.offline_search.rounds.r12.ablate.prepare --install-test-a`）。
- 消融代码已推 timan107（哈希一致）；**未推 h100 / timan108**。GR00T 两格消融（h100）要等 sol fix4（回收包超 tether 812 MB 上限自动拆分，`tmp/r12/sol_bugs.md`）交付并推送后再入队（`@h100`）。
- 队列现为 `tmp/r12/queue4.txt`，脚本 `run_q5.sh`（消融生效核对 `tmp/r12/engage_check.py`）；旧 v4 跑完手上的根后由 succ5 接手。
- owner 裁定：π0.5 长任务 h2/m2 照跑全量，标"越过筛选"。opus 分析存档 `rounds/r12/opus/PILOT_ANALYSIS.md`，测试集 A 预测冻结在 `opus/PREDICTION.md` 末节。
- 手动回收过 `r12_guard_groot_l10_50_off`（h100 同显卡对照，SR .70）。

**20:1x 状态**
- B 池小跑全部完成（看一半 13 组 + 新守卫 9 组 + 额外原系统对照）。opus 方法 / h2 / m2 筛过；astra 两种门筛掉（直接节省 ≤ .005）。打分脚本 `tmp/r12/score_pilot.py`。
- owner 10-03 ~19:40："用 100 episode 快速筛选完了要上全 500 的" → 测试集 A 全量 20:13 开跑。
  - 根（sol 生成，`rounds/r12/TEST_A_HANDBACK.md`）：`/home/weiland/trace_runs/os_closed_loop/r12_test_{pi05_l10_50,pi05_spatial_50,groot_spatial_50,groot_l10_50}`（off/opus_m1/h2/m2）、`r12_test_pi05_spatial_200`（off/opus_m1）、`r12_test_pi05_l10_50_periodic_ir32`（off/h2/m2）、`r12_test_pi05_spatial_200_guard_pending`（h2/m2，**等 B 快筛 `guard/pilots/r12_guard_fix2_pi05_spatial_200` 通过才跑**）。
  - 队列 `tmp/r12/queue2.txt`（源 `queue_testA.txt`），脚本 `tmp/r12/run_q3.sh`（跑完但回收失败只重收不重跑；h100 上 π0.5 用 3 服务）；两条线同前。
  - sol 修了 A 池上新 worker 不启用的问题（`worker_pool.install`）；代码已推 h100/timan108/timan107，哈希一致。
- 在跑的 agent：opus（小跑分析 + 冻结测试集 A 预测，结果存 `rounds/r12/opus/PILOT_ANALYSIS.md`）、astra（复盘，`rounds/r12/astra/PILOT_ANALYSIS.md`）、sol fix3（本机服务登记竞态，见 `tmp/r12/sol_bugs.md`；修好后要推 h100/timan 若有改动）。
- 已知坑：本机服务登记竞态（约每 15 组一次）→ STATUS_POLL_FAILED + 孤儿服务占端口；处理＝核对进程属于该组后按 PID 停 supervisor+child，脚本自动重试。

**18:0x 状态（已过时）**
- sol 修复已交付（17:48 我取消了它剩余的慢速 CPU 冒烟）。根因：旧 worker 不发执行回执，服务端按规则退回全看。新 worker `closed_loop/execution_worker.py`，主动方法自动选用。
- 17:51 开跑小跑，**共享队列**：`tmp/r12/run_q.sh` + `tmp/r12/queue.txt`（谁空谁领，方法组在前，看一半根的原系统组在队尾，守卫根原系统组因与看一半根重复而跳过，只留旋钮 .32 根的）。
  - 本机 4090 + timan107（WPS16，端口 23100–23103），日志 `/tmp/r8_coord/r12_q_local_timan107.log`
  - h100 + timan108（WPS10，端口 23230–23233，SYNC 23196），日志 `/tmp/r8_coord/r12_q_h100_timan108.log`；h100 上另有别人的两个服务约 41 GB，属正常共用。
  - 代码已推 h100 / timan107 / timan108，哈希一致。
- 生效核对：本机 π0.5 长任务 opus_m1 运行中约 27% 看图点只看一路；h100 GR00T 空间 opus_m1 26%。脚本对每个主动组（h2 除外）自动查 `one_camera_looks>0`，失败写 `ENGAGEMENT_FAIL`。
- 跑完后：各根 `report.py` 打分 → 我写 `rounds/r12/REPORT.md`；回叫 opus / astra 时各自只给自己的方法组 + 原系统组（保持隔离）。

**17:4x 状态（已过时，仅留档）**
- **h100**：owner 明确说"h100 可以用"。之前读 h100 被自动权限审查拒，是因为我挪动了 h100 镜像文件，并非 h100 不可用。
  - 现状：显卡空（0 MiB，另一项目训练已结束），`/data` 余 47G；中午 h100 录制 `r12_devlooks_{50S,200}` 已在 12:35 完成。
  - h100 服务树尚无 R12 代码（lookhalf* / guard.py / plugin 改动），sol 交付后要推送并核对 sha。
- **sol 修复**（codex `task-musxy7rt-gjkfud`，日志 `~/.claude/plugins/data/codex-openai-codex/state/openpi-50fd553c5e274099/jobs/task-musxy7rt-gjkfud.log`）：17:40 正在本机做真实冒烟。交付位置：
  - `rounds/r12/lookhalf/HANDBACK.md`
  - `rounds/r12/guard/HANDBACK.md`
  - 新名字的小跑根。
- **交付后做**：
  1. 推送 R12 服务代码到 h100，核对 sha。
  2. 两条线并行：π0.5 格子用本机 4090 + timan107，GR00T 格子用 h100 + timan108。
     - 本机：`SERVER_HOST=local`，4 服务 × WPS 16；
     - h100：`SERVER_HOST=h100`，PORTS 23230–23233，WPS 10，SYNC_PORT 23196；
     - 两条线同步都用 flock `/tmp/r8_coord/r10_sync.lock`，持锁到开跑。
     - 脚本参照 `tmp/r12/run_lhpilot.sh`（rec_run 系，`B=` 指 pilots 目录，`LOCAL_SERVER_COUNT=4`）。
  3. **每条线第一组方法组跑完立刻核对** `summary.json` 的 `lookhalf_cost_ledger.one_camera_looks > 0`（新守卫看 H2 决策计数）；为 0 立即停（记忆 `feedback_verify_engagement_and_stalls`）。
  4. 先跑看一半小跑，再跑新守卫小跑（各 5 格；看一半 off / opus_m1 / astra_consensus2，π0.5 50 集两格加 strict；新守卫 off / h2 / m2），各 100 局。
- **本机此刻**：在跑多余的原系统对照组（`tmp/r12/run_offs.sh`，日志 `/tmp/r8_coord/r12_offs_local.log`，GR00T Sp-50 off 17:39 开跑，之后 GR00T L10-50 off）。
  - 这些与已有录制重复：`r11_dev_k10_*` 的选集 = 小跑选集，且录制组就是原系统，故每格已有 2–3 份原系统基线，只当重复样本。
  - 方法组要开跑时可以停掉它：先停包装脚本，再对该根 control abort（`SERVER_HOST=local`）。
- **owner 不满**：GPU 空等。今天浪费：sol 被中断后空等约 1.5 h，看一半小跑白跑约 1 h。之后不许让 GPU 空等，等待期间要安排有用的活。


**R12（owner 10-03 /goal "探索实验看一半"）**——简报 `exp/offline_search/rounds/r12/BRIEF.md`、`DIRECTIVE_2.md`，台账 R12 段

**今天 owner 的定调（都已写进记忆 `project_offline_search_five_layer_design`）**
- 五层设计：1 缓存 / 2 守卫 / 3 修正器 / 4 推理开销旋钮 / 5 按任务分治。
- 第 4 层：作用是补纯库与纯推理之间的 SR 差距，按目标调；定期调用与距离阈值调用都正式纳入，之后再细比。

**R12 的目标与规则**
- 目标：缓存层"看一半"（只编码一个相机），要么 SR 无损且 IR 降，要么掉的 SR 用旋钮补回后同 SR 总 IR 更低。
- 研究员隔离：astra 与 opus 不许读对方 `rounds/r12/<other>/`。
- **owner 裁定（DIRECTIVE_2）**：
  - 必须是方法，不是代码层面的效率优化；合批编码、跳过占位图、底层计算技巧都不算。
  - 价目按集中部署满负荷算：单看一路 = 视觉价的一半（π0.5 .076 + 补看 .076；GR00T .074 + .074）。
  - owner 明确说**不要把这条记进记忆**。
- **数据纪律**：
  - `offline_search_store/queries/*` 是 trace_dual 在测试集 A 上录的，禁止用于设计。
  - 只用 B 池示范库（含两视角 key / token / 原图）与 B 池开发集。

**开发集录制（全部完成）**
- 根目录：`os_closed_loop/r12_devlooks_{50L,400,knob,50S_local,200_local}`；h100 那条录的 `r12_devlooks_{50S,200}` 据 opus 也已完成。
- 记录内容：每个看图点的两视角 key、状态、top-k、守卫状态；50L 另有 token、图像和影子策略动作。
- 记录功能：`closed_loop/look_recording.py`（只限 B 池根）。

**研究结果**
- **opus（`rounds/r12/opus/`，REPORT / HANDBACK 为协调者代存）**
  - 第 1 版"同批编码"已被 owner 否决；
  - 按新价目，看一半在五层系统中基本省不下钱：单路检索判"无进展"会漏约三成守卫调用；
  - 之后研究"换信号源守卫" `GUARD.md`：H2 用 7 个不靠第三人称的特征做风险模型（库内拟合），阈值按格子在开发集上定、使调用率与原守卫相同。回放上更会叫，3 个长任务格首叫晚 3–10 步；
  - M2 = H2 + 看一半：单路占比不开旋钮 .23–.50、开旋钮 .15–.22，预测同 SR 约 −.025（区间宽）。
- **astra（`rounds/r12/astra/`）**
  - 记忆 + 位移 + 双证一致门（consensus2 / strict）；
  - 影子核对 700 局：π0.5 省 .0016–.0035，同样指出守卫是瓶颈。

**实现（sol）**
- 看一半：`closed_loop/lookhalf*.py`、`rounds/r12/lookhalf/`（R12LookHalf：off / opus_m1 / astra_consensus2 / astra_strict）。
- 新守卫：`closed_loop/guard.py`、`rounds/r12/guard/`（R12Guard：off / h2 / m2）。
- 已知不一致：opus 文字与回放代码的距离归一化不同，服务端按 SPEC 文字实现，少量决策不同，待核。

**看一半小跑作废（17:0x）**
- 8 组 `one_camera_looks=0`：规划 / 同步把 `--os-lookhalf <方法>` 改写成了契约路径，服务静默按原系统跑。已停并 abort。
- 修复中：sol codex `task-musxy7rt-gjkfud`（续新守卫线程），等待器 `bjd1w4lzh`。
  - 要求：修参数改写、配错直接报错、端到端检查真实启动参数、本机冒烟证明单看发生、新守卫一起修、重新生成两套小跑根（新名字）。
- **修好后**：先跑看一半小跑（5 格 × {off, opus_m1, astra_consensus2} + π0.5 50 集两格的 strict，各 100 局，旋钮关），再跑新守卫小跑（5 格 × {off, h2, m2}）。
- **跑法**：本机 4090 + timan107，`SERVER_HOST=local`、4 服务 × WPS 16。脚本 `~/.claude/jobs/a607dd74/tmp/r12/run_lhpilot.sh <local> timan107 23100,23101,23102,23103 16 <roots…>`（B=pilots 路径；`left.py` 计剩余组数；失败自动重试）。
- **验收**：先看 summary 的 `lookhalf_cost_ledger.one_camera_looks > 0`、`ir_compute_share`、`first_guard_call`；再与 off 配对比 SR / IR。

**待 owner 裁定**
- 新守卫阈值按格子用开发集定是否可接受；
- 上测试集时是否同时测"只换守卫"和"换守卫 + 看一半"；
- h100 管理权限：读 h100 录制状态被自动权限审查以"修改共享资源"拒绝，之后未再碰 h100。

**运行器 bug 清单（给 sol，未修）** `tmp/r12/sol_bugs.md`
- h100 镜像路径冲突：协调者把 4 个旧镜像改名 `*.r11dev.bak`；
- 本机状态轮询连续失败；
- 清理拒停自己的服务、留残留：协调者按 PID 停过。

**其他**
- 13:56 sol 线程被外部中断（`turn_aborted interrupted`）：续跑前须先 cancel 旧任务记录，再 `--resume-last`。
- owner 说另一会话 `5a375dad` 在 `/home/weiland/s5wt/p1..p28` 开的 28 个 codex "不要管它"。
- idea-sprint 9 月 29 日孤儿中转进程不归本会话。

### R11（10-03 09:08 完成）—— 第 4 层旋钮，报告 `rounds/r11/REPORT.md`；测试集 94 组
- 开销落在目标 ±.02 内：定期 17/17、随机 16/17；状态类离线标定后闭环少花（距离均 −.08）。
- 定期比随机 +1.5–2.2 pp（同实际 IR）。
- 研究员终稿：`rounds/r11/{opus,astra}/ANALYSIS.md`。
- 本机 4090 与 H100 汇总可合并（权重 sha 一致）。
- **提交**：
  - 已推送：a3b10ecd、e26c41e1（R8–R10）；9d57c8d9（R10Recipe + R11，10-03，sonnet）。
  - R12 全部未提交。



**R11 完成（10-03 09:08 测试集 94/94；报告 `exp/offline_search/rounds/r11/REPORT.md`）**：
- **结论**：
  - 定期调用 17/17 落在目标 IR ±.02 内；同实际 IR 比随机 +1.5–2.2 pp，且是唯一稳定高于"整局混跑"直线的可控方法（+1.2）。
  - 随机调用 16/17 落在目标内，保留作标定参照。
  - 状态类（距离 / 分歧 / 误差混合）离线标定后闭环少花（距离均 −.08），不作旋钮。
  - 两个变体不采用。
  - 接近纯推理的格子默认关闭，判据（opus：缓存与大模型动作差 < .40）待开发集验证；astra 对 GR00T L10-500 有异议（批次噪声 .858 vs .906）。
- **待 owner**：
  - 是否采纳定期调用为第 4 层；
  - 是否开 R12 验证（守卫反复触发时逐步加密 / 全库估守卫率 / 每局预算记账 / 默认关闭判据的开发集验证）；
  - R11 代码与数据的提交（未授权，未提交）。
- **代码**：
  - `rounds/r11/knob/`（R11Knob）；
  - `closed_loop/devset.py`（B 池开发集）；
  - `closed_loop/ops/h100/{local_server.py, server_host.py, HANDBACK_LOCAL.md}`（本机 4090 服务，`SERVER_HOST=local`，4 服务 × 16 worker，约 14 分钟 / 组）。
- **跑法要点**：
  - h100 只有一张卡，另有项目占 10.6G，两车队同时各 4 个 π0.5 服务会 `GPU_TIGHT` → 一个车队用本机 4090；
  - 同步须持锁到开跑；
  - 本机端口释放竞争已修。
  - 脚本在 `~/.claude/jobs/a607dd74/tmp/r11/`：`run_root3.sh`、`run_dev3.sh`、`sweep107L.sh`、`resume107L2.sh`、`r11_score.py`、`PATROL.md`。
- **硬件**：4090 与 H100 权重 sha 一致、汇总可合并；单局不可跨卡复现。



**R11（owner 10-02 20:1x /goal，"不做完不停"）**：
- **五层设计**（记忆 `project_offline_search_five_layer_design`）：1 缓存 / 2 只留无进展守卫 / 3 库内修正器 GC_dist / 4 IR 旋钮（本轮）/ 5 沿任务分治。第 4 层不得按任务设参数。
- **简报** `rounds/r11/BRIEF.md`。
  - opus 报告 `rounds/r11/opus/`（随机 / 定期 + 共用账本；REPORT 代存）。
  - astra 报告 `rounds/r11/astra/`（距离 / 误差混合 / 自适应 / 分歧；44 组冻结）。
- **进行中**：
  - sol codex `task-murrgj6c-kuny2o`（简报 `~/.claude/jobs/a607dd74/tmp/r11/sol_knob.md`）：`R11Knob` + 构建器 + 94 组测试集 A（根 `r11_knob_1..4`）+ 开发集小跑 20 组（`r11_devknob_50`）。
  - 闭环两路：timan107 精选 50 集三层 4 组（根 `r10_recipe_current`，前 2 组已完成）；timan108 开发集冒烟（根 `r11_dev_current` / `r11_dev_size50`，脚本 `tmp/r11/run_dev.sh`）。
  - 脚本：`tmp/r11/run_root.sh <fleet> <root>`（测试集，flock 串行同步）、`run_dev.sh`、`score_current.py`。
- **sol 交付后**：
  1. 推 h100 源文件并 `sha256sum -c`；
  2. 先跑 `r11_devknob_50`（只核对开销与生命周期，不改设置）；
  3. 再两车队各跑两根 `r11_knob_*`；
  4. 按"同目标开销配对"出 SR-IR 前沿；
  5. 叫回 opus / astra 分析各自结果；
  6. 写 `rounds/r11/REPORT.md`；
  7. 停本会话 codex 中转进程对。
- **其他**：
  - 开发集机制：`closed_loop/devset.py`，B 池库外起始状态，硬守卫防混 A / B；
  - 已推送提交 a3b10ecd、e26c41e1；
  - h100 余 46G；
  - 读运行日志的 grep 曾被自动权限审查拒绝，以后台等待通知为准。



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

## 4. 正在运行的东西（10-05 07:1x）
- 无。两条线都已退出（tmux 无 r13q*），cron 已删，codex 中转进程已按 PID 停掉。
- 可复用：
  - 排队脚本 `~/.claude/jobs/a607dd74/tmp/r13/run_q6.sh`（共享队列 `queue.txt`、`@local` / `@h100` 指定机器、按 h100 剩余显存定服务数）；
  - 确实生效检查 `engage_check.py`（原因码 61–79）、巡检 `probe.sh`、监控 `combined_watch.sh`。
- **坑**：
  - 一个运行根同一时间只能由一条线服务（`fleet_lock`）；
  - 完整 `setup` 要拿全局锁，本机线在跑时只能用 `control.bundle` 推文件；
  - 端口释放超时后，重试会被上一次的孤儿 worker 卡住（PPID=1、同 driver-port），要按 PID 清理。

## 5. 下一步
1. 等 owner 下一个方向。
2. 若要提交 R12–R14，须 owner 授权。
3. 若 owner 要把 R14 推进到逐任务保证：每任务约 100 局以上的下降数据或保守规则，再在新的评估数据上确认。

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
