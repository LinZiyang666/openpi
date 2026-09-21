# Session handoff

> 多条线共用本文件。§0 为常驻初始化（不动）。**§1–§7 = 减步 vs warm start 诊断线（`exp/step_diag`）运行阶段 live 交接，2026-09-20 21:45 CDT 覆写**。前两条线（x₀-head、减步基线）终态只在记忆 `project_x0_multimodal_line` / `project_nfe_baseline_live_run`。

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


## 1. 现在在哪（2026-09-21 02:05 CDT）— **实验全部跑完，终报已写，等 owner 审阅/commit 指示**

- 代码已 commit `d8e464d`（origin/Ziyang）。plan `logs/step_vs_warmstart_diagnostics_plan.log.md` v3.1（§9.6 已补运行记录）；手册 `exp/step_diag/ops/README.md`。
- **全部 9,760 集正式数据已拉回本地并校验**（server 行 arrays sha 逐文件 0 bad；driver 产物 journal 集数齐全）：shadow 660、Q-B π0.5 2,750、Q-B GR00T 1,350、Q-C.3 5,000。分析产物：`exp/step_diag/data/analysis/{shadow_<env>,qb_pi05,qb_groot}.json`，表 `exp/step_diag/analysis/{shadow_<env>,qb_pi05,qb_groot}.md` + 拼接 `step_vs_warmstart_tables.md`，**手写终报 `exp/step_diag/analysis/step_vs_warmstart.md`**。
- **结论**：Q-B 两 policy 均 `inconclusive` + `harmful_on_flat`（π0.5 cliff macro Δ −0.035 [−0.145,+0.075]、H50 上界 <0；GR00T Δ +0.055 [−0.045,+0.155]；flat PnP 任务 Δ −0.63/−0.78（π0.5）、−0.45（GR00T））；Q-A 两 policy `no_conclusion`（ρ 0.085 / 0.377）；Q-C.3 object/goal 阶梯平坦。全部 cell 准入（complete/equal_nfe/miss 0）。
- **远端状态**：h100 与 weilandserver 全部 server 已停、MPS 已关（GPU 0 MiB）；timan107/108 无 driver/worker 残留（tmux `sdphase_*` 会话已自然退出）。远端树与数据保留（h100 `/data/openpi_sdiag`、wls 同、timan `/scratch/zixuans8/step_diag/openpi`；打包分块在各机 `/tmp/sdiag/pull/`，可删）。
- **已 commit + push `952dc99`（owner 2026-09-21 授权，单提交）**：运行期 7 处代码改动（§6）+ 分析 md（`parity_*.json`、`shadow_*.md`、`qb_*.md`、`step_vs_warmstart_tables.md`、`step_vs_warmstart.md`）+ `config/rc_timan107.env` + `rit_pareto/config/task_order_libero_{object,goal}.json` + handoff/README/plan 日志。**待 owner**：四层设计下一步（终报 §0/§3.3 给了 warm start 必须带门、相似度分数不区分好坏命中的负证据）。

## 2. 拓扑与资产（全部已验）

| 角色 | 主机 | 关键路径 |
|---|---|---|
| RC server π0.5 | h100 `149.165.153.233`，端口 23240-47；tree `/data/openpi_sdiag`（clone d8e464d）；`HOME=/home/exouser`；根盘 7 GB | py `/home/exouser/openpi/.venv/bin/python`；ckpt `/home/exouser/ckpt/pi05_robocasa_pytorch`；W13 库 `/data/robocasa365_cache/cache_artifacts_w13/{pi05,groot_tp}_spatial_pool_16_w13_full.pkl`（侧车已写）；server 产物 `exp/step_diag/data/server/<teacher>/<arm>/`（smoke 在 `server_smoke/`） |
| RC server GR00T | weilandserver `ziyanglin.com` 23150-59；tree `/data/openpi_sdiag`；`HOME=/home/weiland` | serve_groot.sh 默认路径即可（gr00t `/home/weiland/gr00t_n15`，ckpt `/home/weiland/ckpt_n15_robocasa_tp/.../checkpoint-60000`）；每进程 ≈5.8 GB → 4090 最多 7 个 |
| RC worker | timan108（3×24 GB，h100 lane，env `config/rc_timan.env`）；timan107（8×8 GB，weilandserver lane，env `config/rc_timan107.env`） | tree `/scratch/zixuans8/step_diag/openpi`；岛 python `/scratch/zixuans8/Isaac-GR00T/gr00t/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python`；driver 产物 `exp/step_diag/data/rc/<teacher>/<arm>/`（smoke 已挪到 `rc_smoke/`） |
| LIBERO（已完） | weilandserver 23140-43/23150-52 ↔ timan107（sim python `/scratch/zixuans8/libero_sim/bin/python`，A 池 `/scratch/zixuans8/openpi/exp/common/data/db_init/libero/*_apool`，object/goal 池现场 materialize 在 step_diag 树）| 产物已在本地 `exp/step_diag/data/{server,libero,qc3}` |

- 单连接硬约束：π0.5 / GR00T RC 一个 server 进程 ↔ 一个 worker；同一 server 不能同时被 main/pnp 两个 cell 用；跨臂同 lane 槽数一致（每臂每 lane 2 槽）。
- tether：`/scratch` 不在 timan allow_roots → 推 `/tmp/sdiag/` 再 cp；`tether pull` 单文件上限 ≈447 MB（split -b 400m，按 sha 对账再 cat）；h100 无 ssh，数据 tar+pull；weilandserver 走 LAN `rsync weiland@192.168.0.200`。h100 `tether exec` 双执行，脚本全幂等。

## 3. 运行链（脚本在本机 `$CLAUDE_JOB_DIR/tmp/sd/`，远端副本 `/tmp/sdiag/`）

1. **起 server**（幂等，分批加载）：h100 `bash /tmp/sdiag/sd_servers_launch.sh <pi05|groot> sdiag_v1 /data/openpi_sdiag/exp/step_diag/data/server <mode:arm:port:arg>...`；weilandserver `bash /tmp/sdiag/sd_servers_wls_launch.sh sdiag_v1 <out> <env_id:mode:arm:port:arg>...`（arg = yaml 绝对路径 | 步数 | `-`）。就绪 = `sdlaunch_*.log` 出现 `SDSERVERS_EXIT` 且每行 `listening`；完整 phase 表 `sd/qb_phases.txt`。
2. **读 config_sha**：`<out>/<teacher>/<arm>/manifest_<arm>.json` 的 `config_sha`。
3. **起 phase**：`bash /tmp/sdiag/sd_cells_launch.sh sdphase_<tag> /tmp/sdiag/sd_rc_phase.sh <pi05|groot_tp> v1 "<arm>=<sha>=<main ports csv>=<pnp ports csv>" ...`；可加 `SD_LANES=main|pnp`（launcher 会写进 tmux 命令）。脚本自动填正式任务集/集数/flat 100 集、INCOMPLETE 自动 resume ≤3 次、开跑前扫孤儿。日志 `/tmp/sdiag/sdphase_<tag>.log`（`CELL … SDCELL_EXIT=0` / `SDPHASE_EXIT`），cell 日志 `/tmp/sdiag/sdcell_v1_<teacher>_<arm_>_<lane>.log`。
4. **停 server**：`cd /data/openpi_sdiag && bash exp/step_diag/ops/stop_servers.sh <ports>`（释放锁）。换臂前必停旧的（RAM/显存）。
5. **孤儿**：`bash /tmp/sdiag/sd_sweep_orphans.sh`（driver 端口无人监听即杀）。
6. **拉取**：h100 `tar` + `split -b 400m` + `tether pull` 逐块 → `exp/step_diag/data/server/<teacher>/<arm>/`（`ops/pull_server_rows.sh` 的 sha 校验逻辑单独跑）；weilandserver `bash exp/step_diag/ops/pull_server_rows.sh weiland@192.168.0.200:/data/openpi_sdiag/exp/step_diag/data/server/<t>/<arm> exp/step_diag/data/server/<t>/<arm>`；driver 产物 timan tar → `tether pull` → 解到 `exp/step_diag/data/rc/`。
7. **分析**：`uv run python -m exp.step_diag.analysis.aggregate_arms --policy pi05 --arms-root exp/step_diag/data/rc --server-rows exp/step_diag/data/server --shadow-json exp/step_diag/data/analysis/shadow_pi05_rc.json --out-json exp/step_diag/data/analysis/qb_pi05.json --out-md exp/step_diag/analysis/qb_pi05.md`（groot 同理）；或 `ops/analyze_all.sh`。

## 4. 监控

- cron `5660763b`（每小时 :09/:39）：`bash $CLAUDE_JOB_DIR/tmp/sd/sd_probe.sh` 只贴 PROBE 行。
- Monitor（会话级，compact 后要重挂）：timan108 `sdphase_Q*.log` 的 `SDPHASE_EXIT|GAVE_UP` + journal accepted 计数；timan107 `sdphase_G*.log` 同；server 就绪看 `sdlaunch_*.log` 的 `SDSERVERS_EXIT`。server 日志判活别 grep websockets 握手 Traceback。

## 5. 准入要点（分析端会拒什么）

- 三方对账：journal accepted terminal ↔ launch manifest ↔ worker `episode_summary` ↔ server finalize/决策行（`client_stamp` launch/arm/experiment/config_sha、`executed_steps==m`、`n_stage3_calls==1`、hit_type、schedule、arrays sha）。
- 非并发 server 每进程一个 `session_id`；同 uid 两次到访（driver 崩溃/孤儿后 resume，attempt 都从 1 起）靠 finalize 行分 occurrence，用 accepted terminal 的 run_id→launch 选 session，其余记 `stray_sessions` 不门控（本次运行期改的分析逻辑）。
- 跨臂门只看 checkpoint 身份 + 环境合同 + experiment；host/GPU/MPS/worker 岛只作 notes。

## 6. 运行期代码改动（本地工作树，未 commit，四台远端已同步）

1. `tests/exp/step_diag/test_parity_manual.py`：排除 `stage_timing/server_timing`；π0.5 参照钉 `_stage3_action_expert` 的 num_steps。
2. `src/openpi/conductor/driver.py`：`except (TimeoutError, socket.timeout)`（LIBERO driver 跑 py3.8）。
3. `exp/step_diag/ops/run_rc_cell.sh`：tmux 名去掉臂名里的 `.`。
4. `exp/step_diag/run_diag.py`：INCOMPLETE 退出码 1。
5. `exp/step_diag/analysis/{aggregate_arms,analyze_shadow}.py` + `tests/exp/step_diag/test_aggregate.py`：occurrence/session 消歧（85 测试过）。
6. `exp/nfe_baseline/ops/`：**删空闲超时换档 v1**，`ladder_server.sh` = 原 v2（信号切档，切档后消费 kdone 文件），`rc/sig_client.sh` 同时识别 `LANE|RCLANE`（owner 指示）。
7. 新文件：`exp/step_diag/config/rc_timan107.env`、`exp/rit_pareto/config/task_order_libero_{object,goal}.json`、`exp/step_diag/analysis/parity_*.json`、`shadow_*.md`。

## 7. 坑与裁定（必读）

- ⛔ owner 21:15：**任何机器不许干等**——某对机器空出来立刻把剩余队列整块（同 policy）搬过去。
- owner 14:22：Q-B 全程 MPS（两 server 机都已开；结束后 `echo quit | nvidia-cuda-mps-control`）；终报写明 runtime。
- ⛔ timan 上 `pgrep -fa` 里 `tmux new -s X …` 那个进程是 tmux **server**，杀它 = 全部 driver 死、worker 变孤儿霸占单连接端口（1013）。孤儿按 driver 端口无人监听判定。resume-noop 的 cell 也会留孤儿。
- 1013 连败烧光重试 → cell INCOMPLETE；同参数重跑 = resume 只派发缺失身份（已验）。
- 换 server 数目后同 run_id 的 `run_plan_*.json` 会 mismatch 拒启；零集完成时删该目录旧 plan/launch 再起。
- h100 RAM：带库 π0.5 进程 29 GB、GR00T 库进程 ~22 GB；9 个 shadow 进程曾把 avail 压到 11 GB。GPU：π0.5 8 GB/进程。
- 4090：GR00T RC ≈5.8 GB/进程，最多 7 个；π0.5 LIBERO ≈7 GB。
- smoke 与正式共用 driver 产物目录 → `sd_move_smoke.py` 已挪走；server 侧 smoke 在 `server_smoke/`。
- 阶梯（Q-C）只用信号切档；`analyze_shadow` CLI 钉正式参数，smoke 用 `sd_smoke_shadow_check.py`。
- 全仓 `uv run pytest` 需 `--continue-on-collection-errors`；既有失败清单见 plan §9.5。
