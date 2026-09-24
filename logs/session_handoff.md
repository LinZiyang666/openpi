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


## 1. 现在在哪（2026-09-23 01:40 CDT）— **本线全部实验（含 09-22 晚的追加）已跑完、分析与报告写完、全部机器已停；只等 owner 指示 commit**

**owner 指令**：`/goal`「我去睡觉了，你独自值守实验，不做完不停」（09-22 21:30）——已完成。此前各轮（§6.1–6.7）见上一版。commit 只在 owner 当次指示时。

**终报** `exp/step_diag/analysis/step_vs_warmstart.md`：§0 结论（含 §6.8–6.10 一行）；§6.1–6.5 π0.5；§6.5.1 外部审查修正；§6.6 GR00T 对称；§6.7 运行记录与事故（含 09-22 20:55 OOM、09-23 00:01 timan107 tmux 事故）；**§6.8 midfinal 喂入点消融、§6.9 GR00T 2 步补齐 13 任务、§6.10 init_probe 机制探针**。
- π0.5 13 任务（2 次前向）：full .548 / plain_k2 .481 / warm .277 / warmreset .708 / resetfinal .708 / **midfinal .545**（midfinal − resetfinal −0.163 [−0.205, −0.120]）。
- GR00T 13 任务 1 次前向：full .638 / plain_k1 .575 / warm .555 / warmreset .562 / resetfinal .573 / **midfinal .611**（− resetfinal +0.038 [0.000, +0.075]，− full −0.028 含 0）。
- GR00T 13 任务 2 次前向：full .638 / plain_k2 .629 / warmreset .632 / resetfinal .646（全部差值含 0）。
- init_probe（40/40，完整模式）：t=1 喂入时 π0.5 输出保留缓存差异远多于 GR00T；报告 `analysis/init_probe_20260922/complete_results.{zh,en}.md`，合并视图 `data/init_probe_20260922/merged/`，工具 `ops/resume_init_probe.py`。
- 表：`analysis/warm_variants_{pi05,groot}_*.md`（新增 `groot_macro13_t0.5`，`pi05_macro13`/`groot_macro13`/`*_t0.1/t0.2/t0.75/t0.5` 已含 midfinal）；图 `analysis/figures/{macro13_four_arms,groot_macro13_five_arms}.png`、`analysis/init_probe_20260922/*.png`。
- 数据全部在本地 `exp/step_diag/data/`，server 行 arrays sha 全对（midfinal 两臂各 2 条被重跑覆盖的 stray finalize，不门控）。
- 机器：h100 / weilandserver 全部 sd server 停、GPU 0 MiB、MPS 未开；timan107/108 无我们的 cell（timan107 用 `tmux -L sdiag ls` 查）。远端 `/tmp/sdiag/*.tgz`、`initprobe_r1_transfer/` 为临时包，可删，未删。

**本轮新增代码（未 commit）**：`envs.py`（`midfinal` 模式、`MID_ENTRY_T`、macro13 放行 midfinal）、`pi05.py`（`mid_final` 分支、`FINAL_START_VARIANTS`）、`groot.py`（`mid_denoise_loop`、`mid_final`）、`serve_diag_groot.py`、`run_diag.py`、`ops/serve_{pi05,groot}.sh`、`ops/run_rc_cell.sh`（`SD_TMUX_SOCKET`）、`analysis/warm_variants.py`（midfinal 臂 + 外部审查三处修正）、新 `ops/resume_init_probe.py`；测试 `test_warm_variants.py`、`test_groot_warm_variants.py`、新 `test_resume_init_probe.py`（step_diag 套件 120 passed / 1 skipped，09-23 01:40）。另有 Codex 的 `serve_init_probe.py`、`analyze_init_probe.py`、`ops/run_init_probe_pilot.py`、`tests/.../test_init_probe.py` 与 `data/figures/plot_init_probe_*.py`（画图脚本不入库）。

**09-23 上午追加（owner）**：① GR00T 喂入点补充（起点=缓存最终动作）：`midfinal50_t0.75/t0.5`（喂 t=0.5，n=1/2）全 13 任务 + `midfinal_t0.5`（喂 0.75，n=2）补 11 任务，09:35 起跑（h100 23250–59 ↔ timan108 main；wls 23150–56 ↔ timan107 pnp，私有 tmux `-L sdiag`）；`midfinal_t0.5` main 已完 .663。② 排队：GR00T `midreset_t0.75/t0.5`（起点=缓存快照，喂 t=0.75，n=1/2）× 13 任务，代码已同步（sha 5ffe8345，测试 126 passed）；① 的 server 空出即起。

**09-23 12:50 状态（h100 即将关机；本会话在 auto 模式下对「从 h100 拉数据」被会话级安全检查拦截，需切出 auto 模式或新会话继续）**：
- GR00T 喂入点补充全部完成并已拉取分析（`analysis/warm_variants_groot_macro13{,_t0.5}.md`）：起点=缓存最终动作，13 任务 macro：喂 t=1 1 步 .573 / 2 步 .646；喂 0.75 1 步 .611 / 2 步 .643；喂 0.5 1 步 .646 / 2 步 .600（full .638、plain_k1 .575、plain_k2 .629）。
- midreset（快照起点喂 0.75）：main 两格完成（n=1 .5775、n=2 .645，8 任务），h100 server 已全停、GPU 0；PnP 两格仍在 wls 23150–56 ↔ timan107（私有 tmux `-L sdiag`）。**待办**：h100 上 `server_macro13/groot_tp/midreset_t0.{75,5}` 两个目录（约 230 MB）还没拉——用 `/tmp/sdiag/sd_h100_pack.sh` + `/tmp/sdiag/sd_pull_parts.sh` 拉回到 `exp/step_diag/data/server_macro13/groot_tp/`；timan108 `rc_macro13/groot_tp/midreset_*`、timan107 同名目录 tar+pull；wls server 行本地 cp；然后重跑 `warm_variants --policy groot --t 0.75/0.5` 13 任务并写报告 §6.11（起点 × 喂入点 × 步数总表）。
- **h100 数据抢救**：清单 `/archive/h100_rescue/inventory_{data,home}.tsv.gz`（本机逐文件比对：h100 独有 542 GB，其中大部分是 env/cache/代码）；第一级 4.4 GB 已搬完到 `/archive/h100_rescue/h100/<原绝对路径>`（x0 结果除 ckpt、/tmp 日志、openpi 各线独有文件、home/openpi 独有、sdiag smoke；`/archive/h100_rescue/done/*` 记文件数全对）。工具 `~/.claude/jobs/ffa26b09/tmp/h100_rescue.sh <bundle> [gzip|plain]`（读 `/archive/h100_rescue/lists/<bundle>.lst`，tether 分块拉、逐块 sha、续传）。第二级待搬（按价值）：x0 `runs/*/checkpoints/final.ckpt` 38 个 66 GB（plain tar）→ `/data/dp_h100/data` 15 GB → x0 `*.hdf5` 8.5 GB → `/data/xwam/robotwin_data` 20 GB → `/data/libero_cache/corpus_w13` 95 GB。tether 实测 ~5 MB/s、同时仅 1 个传输；更快的通道（本机 rsync over ssh，需 owner 自行把本机 `~/.ssh/id_ed25519.pub` 加到 h100 `authorized_keys`）被权限规则拦，未绕过。

**09-23 14:25 状态（会话已恢复为可执行模式）**：
- midreset 两臂 13 任务完成并分析（`warm_variants_groot_macro13{,_t0.5}.md`，§6.11 已写、§0 已加一行）：快照喂 0.75 n=1 .628 / n=2 .669。
- **在跑**：GR00T `midreset50_t0.75/t0.5`（快照喂 0.5，n=1/2）× 13 任务 × 50（审计 agent 找出的唯一欠账），09-23 14:20 起：h100 23250–54（n=1）/23255–59（n=2）↔ timan108 main；wls 23150–52（n=1）/23153,23155,23156（n=2）↔ timan107 pnp；两台 worker 都用私有 tmux `tmux -L sdiag`。跑完：拉 h100 `server_macro13/groot_tp/midreset50_*`（sd_h100_pack+sd_pull_parts）、wls 本地 cp、两台 `rc_macro13/groot_tp/midreset50_*` tar+pull，重跑 `warm_variants --policy groot --t 0.75/0.5` 13 任务，把快照喂 0.5 一行补进 §6.11 表。
- **h100 搬运改用 rsync+ssh**（owner 授权；本机 `id_ed25519.pub` 临时加在 h100 exouser `authorized_keys`，**搬完要删**）：`/archive/h100_rescue/rsync_tier2.sh lane1|lane2`（tmux `h100_rsync_lane1`，lane2 等 lane1 结束后自动接），清单 `/archive/h100_rescue/lists/{lane1,lane2}.txt` → t2a x0 final.ckpt 117 GB、t2b DP 数据 15 GB、t2c x0 hdf5 8.5 GB、t2d xwam 29 GB、t2e libero corpus 95 GB、t2f cosmos-policy 13 GB、t2g robotwin 26 GB；约 30 MB/s（SMR 盘，单路）；日志 `/archive/h100_rescue/rsync_lane*.log`，完成标记 `/archive/h100_rescue/done/`。
- job 临时目录已被清掉：巡检/监控脚本改在会话 scratchpad `/tmp/claude-1000/-home-weiland-projects-openpi/ffa26b09-*/scratchpad/`。
- 审计 agent 标为「不确定、owner 未点名」未跑：π0.5 快照×步数解耦交叉臂；π0.5 版喂入点扫描；GR00T n=3 阶梯；GR00T warm_t0.5 补 8 任务。

**09-23 16:25 状态**：
- **GR00T 全部实验完成**：midreset50（快照喂 0.5，n=1/2）13 任务补跑完（审计 agent 找出的唯一欠账），§6.11 表已补齐、§0 已更新：13 任务 macro 1 步 final 喂 0.5 .646 / 快照喂 0.5 .637 / 快照喂 0.75 .628；2 步快照喂 0.75 .669 最高，喂 0.5 回落（快照 .632、final .600）。h100 与 wls 上全部 sd server 已停、GPU 0 MiB；两台 worker 无 cell。
- **h100 搬运**：lane1 四包完成（t2a x0 final.ckpt 58/58、t2b DP 数据、t2c x0 hdf5、t2d xwam 52385 文件），lane2（libero corpus 95 GB → cosmos-policy → robotwin）在 tmux `h100_rsync_lane2` 自动续上，约 17:45 完成。**完成后**：删 h100 `/home/exouser/.ssh/authorized_keys` 里本机 `id_ed25519.pub` 那一行；删 h100 `/data/h100rescue/`（清单与空 stage）。
- 监控脚本在 `/archive/h100_rescue/ops/`。审计 agent 标「不确定、owner 未点名」未跑：π0.5 快照×步数解耦交叉臂；π0.5 版喂入点扫描；GR00T n=3 阶梯；GR00T warm_t0.5 补 8 任务。

**09-23 17:55 全部完成**：GR00T 起点×喂入点×步数网格 13 任务全齐（§6.11）；h100 数据抢救完成：195 613 个文件约 288 GB 在 `/archive/h100_rescue/h100/<原绝对路径>`，说明见 `/archive/h100_rescue/README.md`（与 h100 清单逐文件大小核对，仅 6 个当时仍在写的 server 日志不同）；h100 上临时 ssh 公钥已删（ssh 已拒绝）、`/data/h100rescue` 已删。所有 sd server 已停，h100 / weilandserver GPU 0 MiB。cron 巡检已撤。

**剩下**：等 owner 决定是否 commit。

## 2. 拓扑与资产（本机 = weilandserver，hostname 已确认）

| 角色 | 主机 | 关键路径 |
|---|---|---|
| 本机/开发树 | weilandserver `~/projects/openpi`（对本机操作直接做，不走 tether） | 岛树 `/data/openpi_sdiag`（server 用；改动文件用 cp 同步）；数据 `exp/step_diag/data/{rc,rc500,rc_x1m,rc_macro13,server,server500,server_x1m,server_macro13,analysis,figures}`（gitignored） |
| wls server（公网 ziyanglin.com:2314x π0.5 / 2315x GR00T） | 本机 4090 48 GB，MPS 已开 | π0.5 启动器 `/tmp/sdiag/sd_servers_wls{,_launch}.sh`；GR00T 启动器 `/tmp/sdiag/sd_servers_wls_groot{,_launch}.sh`（tmux `sdlaunch_wls_groot`，日志 `/tmp/sdiag/sdlaunch_wls_groot.log`，server 日志 `/tmp/sdiag/sdsrv<port>.log`）；GR00T 上限约 7 进程；π0.5 每进程 GPU 7.5 GB |
| h100 server（149.165.153.233，tether user exouser） | tree `/data/openpi_sdiag`，MPS 已开，根盘 91%（`/tmp/sdiag/pull` 用后即删） | `/tmp/sdiag/sd_servers_launch.sh <pi05|groot> <exp> <out> <mode:arm:port:arg>...`（tmux `sdlaunch_<teacher>`，日志 `/tmp/sdiag/sdlaunch_<teacher>.log`）；单 server 直起：`SD_EXP/SD_OUT/...` 环境 + `nohup bash exp/step_diag/ops/serve_{pi05,groot}.sh ...`（见 §3）；`sd_h100_pack.sh`、`sd_h100_relaunch_all.sh`（事故恢复用）；π0.5 ≤7 进程（RAM），GR00T 端口只能 23250–23259（`rc_timan.env`） |
| worker | timan107 ↔ wls（`rc_timan107.env`：π0.5 23140-47、GR00T 23150-59）；timan108 ↔ h100（`rc_timan.env`：π0.5 23240-47、GR00T 23250-59） | tree `/scratch/zixuans8/step_diag/openpi`；`/tmp/sdiag/run_rc_cell.sh.new`（= 工作树 ops/run_rc_cell.sh，支持 `SD_EXP SD_BASE_SEED SD_OUT_ROOT SD_RC_ENV`）；cell 日志 `/tmp/sdiag/sdcell_v1_<teacher>_<arm>_<lane>[_<exp>].log`，尾行 `SDCELL_EXIT=<code>` |
| 本地拉取 | `/tmp/sdiag/sd_pull_parts.sh <node> <name> /tmp/sdiag/pull`（分块 ≤447 MB、sha 对账、拼回 tgz） | |

tether：`tether exec <node> -- bash -lc '...'`（h100 双执行、10 min 上限；引号坑：多行脚本先落盘再 push）；`tether push <local> <node>:<path> --force`；timan `/scratch` 不在 allow_roots → 推 `/tmp/sdiag/` 再 cp。

## 3. 运行链（剩余 GR00T 工作）

**server 就绪判据**：启动器日志出现 `SERVE <port> ...| listening`（或 `SDSERVERS_EXIT`），`ss -ltnH` 端口在听；`config_sha` 读 `<out>/groot_tp/<arm>/manifest_<arm>.json`。GR00T 加载约 2 min。

1. **起 server**（示例）：
   - h100：`tether exec h100 -- bash -lc 'cd /data/openpi_sdiag && bash exp/step_diag/ops/stop_servers.sh <旧port>; A=/data/openpi_sdiag/exp/step_diag/config/arms; bash /tmp/sdiag/sd_servers_launch.sh groot sdiag_macro13 /data/openpi_sdiag/exp/step_diag/data/server_macro13 resetfinal:resetfinal_t0.75:<port>:$A/groot_rc/warm_t0.75.yaml'`（模式 `full:full:<port>:-`、`plain:plain_k1:<port>:1`、`warm|warmreset|resetfinal:<arm>:<port>:<yaml>`；warm_t0.5.yaml 对应 2 步臂）。若 `sdlaunch_groot` tmux 仍在，改用 nohup 直起：`export HOME=/home/exouser SD_REPO=/data/openpi_sdiag SD_EXP=sdiag_macro13 SD_OUT=.../server_macro13 SD_HOME=/home/exouser SD_GROOT=/home/exouser/gr00t_n15 SD_GROOT_PY=/home/exouser/gr00t_n15_venv/.venv/bin/python SD_CKPT=/home/exouser/ckpt/n15_robocasa_tp/gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000; nohup bash exp/step_diag/ops/serve_groot.sh groot_rc <mode> <arm> <port> <yaml|-> > /tmp/sdiag/serve_<port>.out 2>&1 &`。
   - wls：`bash exp/step_diag/ops/stop_servers.sh <旧port>; bash /tmp/sdiag/sd_servers_wls_groot_launch.sh sdiag_macro13 /data/openpi_sdiag/exp/step_diag/data/server_macro13 <mode:arm:port:arg>...`。
2. **起 cell**：
   - timan108（h100 server）：`tether exec timan108 -- bash -lc "cd /scratch/zixuans8/step_diag/openpi && export SD_EXP=sdiag_macro13 SD_OUT_ROOT=/scratch/zixuans8/step_diag/openpi/exp/step_diag/data/rc_macro13 && bash /tmp/sdiag/run_rc_cell.sh.new groot_tp <arm> <main|pnp> 149.165.153.233:<port>[,...] <tasks csv> 50 - <config_sha> v1"`。
   - timan107（wls server）：同上，加 `SD_RC_ENV=/scratch/zixuans8/step_diag/openpi/exp/step_diag/config/rc_timan107.env`，server 写 `ziyanglin.com:<port>`。1M 段再加 `SD_EXP=sdiag_xseed1m SD_BASE_SEED=1000000 SD_OUT_ROOT=.../rc_x1m`。
   - 待起 cell 的任务串：rf75 main 后半 `OpenDrawer,OpenStandMixerHead,SlideDishwasherRack,TurnOnSinkFaucet`（arm resetfinal_t0.75，需 exp macro13 的 resetfinal_t0.75 server，可复用 23253/23259 空出后）；full pnp `PickPlaceCounterToCabinet,PickPlaceSinkToCounter,PickPlaceToasterToCounter`（server 23256 空出后）；warm75 pnp 同三任务（23157 空出后）；wr5 main / rf5 main `OpenCabinet,SlideDishwasherRack`（23255 / 23254 空出后）。
   - 判据：日志出现 `expected=<n> episodes`；`run-plan mismatch` = 同 lane 换了 server 列表，须把该 lane 的 launch/journal/run_plan/per_step/summary 移走再起（会重跑）；INCOMPLETE 同参数重跑即 resume。
3. **监控**：Monitor 只报终态（`SDCELL_EXIT|INCOMPLETE|DONE arm`，去重），cron `5c94d7c0` 每 15 min PROBE 一行；就绪 Monitor 各起各的。
4. **拉数**（GR00T）：h100 `bash /tmp/sdiag/sd_h100_pack.sh <name> <dir>`（幂等，`<name>.parts` 已存在则要先删 part 文件）对 `server/groot_tp/{warmreset_t0.75,warmreset_t0.5,resetfinal_t0.75,resetfinal_t0.5}`、`server_x1m/groot_tp/*`、`server_macro13/groot_tp/*` → 本地 `sd_pull_parts.sh` → `tar xzf -C exp/step_diag/data/<root>/groot_tp/`；wls 的 `server*/groot_tp/*` 直接 `cp -r --update=none`；timan107/108：`tar czf` 各 root 的 `groot_tp` 子目录 → pull → 解到对应本地 root（同臂两机文件名不冲突，manifest 内容寻址）。校验：`ops/pull_server_rows.sh` 里那段 python 逐 arrays sha。⚠ 拉前确认 cell 已 SDCELL_EXIT，否则要重拉（今天 warmshoot 一次）。
5. **分析**：`uv run python -m exp.step_diag.analysis.warm_variants --policy groot --t 0.75 --tasks TurnOnSinkFaucet,PickPlaceCounterToStove --out-json data/analysis/warm_variants_groot_t0.75.json --out-md analysis/warm_variants_groot_t0.75.md`（阶梯；`--t 0.5` 同）；1M 段加 `--arms-root exp/step_diag/data/rc_x1m --server-rows exp/step_diag/data/server_x1m`；宏观 13 任务 `--arms-root exp/step_diag/data/rc,exp/step_diag/data/rc_macro13 --server-rows exp/step_diag/data/server,exp/step_diag/data/server_macro13 --tasks <13 任务>`（Q-B 5 任务复用 rc/）。分析前删掉本地作废 launch 文件（§7）。
6. **写 §6.6**（GR00T：阶梯表、1M 段、宏观 macro + 逐任务），更新本文件 §1 与记忆；最后 `stop_servers.sh` 全部端口 + `echo quit | nvidia-cuda-mps-control` 两台、GPU 归零。

## 4. 监控

- Monitor = 条件触发（终态/就绪/错误），cron = 定时 PROBE（`5c94d7c0`，会话级）。compact 后两者都要重挂/重建。
- 判 server 活别 grep websockets 握手 Traceback；h100 根盘 91%，`/tmp/sdiag/pull` 分块拉完即删。

## 5. 准入要点

- 变体臂 kind='warm'：每决策 hit_type=WARM_START、start_t=t、executed_steps=n、n_stage3_calls=1；GR00T schedule `groot_n15_k4_v1`，n = K − snapshot_index(t)（t0.75→1，t0.5→2），π0.5 n = ⌊t·10+½⌋。
- 跨臂比较门只看 checkpoint_sha256 + env 契约；同臂跨机 config_sha 不同（ckpt 路径）但可并存一目录。
- 跨 root 合并（`warm_variants.py --arms-root a,b`）要求 task_uid 不重复；不要在两个 root 跑同一 (arm, task)。

## 6. 本次改动清单（未 commit；commit 需 owner 指示，信息英文，无 AI 署名，画图脚本不入库）

- `exp/step_diag/pi05.py`：变体 `reset_t / overshoot / reset_final`（`warm_variant_stage3`、`_final_chunk_like`、变体模式下包 `orch.check`）。
- `exp/step_diag/groot.py`：`groot_warm_variant_stage3`、`install_warm_variant`（升序循环 `denoise_loop(noise=起点, num_steps=n, start_index=0)`）。
- `exp/step_diag/envs.py`：`WARM_VARIANT_MODES`（warmreset/warmshoot/resetfinal）、`warm_t_of/warm_mode_of`、validate_arm 放开（overshoot 仍 pi05-only）、`VAR500_*`、`RC_XCHECK_BASE_SEED/XSEED_*`、`MACRO13_ARMS(_BY_POLICY)`、`XSEED_ARMS/TASKS_BY_POLICY`。
- `exp/step_diag/run_diag.py`：变体臂放行（GR00T 无 overshoot）、`sdiag_var500` / `sdiag_xseed1m`（seed 1M）/ `sdiag_macro13` 三个 exp id 的校验与 out-root 强制。
- `exp/step_diag/serve_diag_pi05.py`、`serve_diag_groot.py`：新模式；`ops/serve_pi05.sh`、`ops/serve_groot.sh`（case + GR00T PYTHONPATH 加 `packages/openpi-client/src`）；`ops/run_rc_cell.sh`（`SD_OUT_ROOT`、tmux 名带 exp 后缀）；`ops/README.md` 两段。
- `exp/step_diag/analysis/aggregate_arms.py`（`want_t`）、新 `analysis/warm_variants.py`（`--t --min-idx`、多 root 合并、macro 分层 bootstrap、`--policy groot`）。
- 外部审查（Codex，16:16）三条统计修正已落 `analysis/warm_variants.py`：重复评测按身份取均值后配对（顺序无关）、逐任务跨臂模型/环境身份门（只比有数据的臂）、配对差全同时 Wilson 退化区间（† 标注）；全部 π0.5 表已重跑，仅宏观 13 任务数字微变（warmreset − full +0.161 [+0.116, +0.205]），报告 §6.5.1 已写。checkpoint 两机全量 sha 一致。
- 测试：`tests/exp/step_diag/test_warm_variants.py`、`test_groot_warm_variants.py`（套件 106 passed / 1 skipped）。
- 报告与产物：`analysis/step_vs_warmstart.md` §6.1–6.5、`analysis/warm_variants_*.md`、`analysis/pi05_macro13_success_rates.{md,csv}`、`analysis/warm_variants_note_for_professor.md`、`analysis/figures/macro13_four_arms.png`；`data/` 下全部 gitignored。
- 别的会话/owner 的文件不动：`logs/README.md`、`logs/cache_trace_mode_plan.log.md`、根目录截图。

## 7. 坑与裁定（必读）

- owner 裁定：本线追加实验 L1 无审查；不跑 row 4；英文对外；Monitor/cron 职责不重合；GR00T 不跑 500 集与 warmshoot；宏观轮四臂都跑（warmshoot 排最后）。
- ⚠ **h100 事故 09-22 07:49**：exouser 全部进程（tmux server、MPS、8 server）同时消失，原因未定（紧随 `stop_servers.sh 23244 23245` + 双执行的 launch 之后）。恢复脚本 `/tmp/sdiag/sd_h100_relaunch_all.sh`（清 lock、重开 MPS、原端口原配置重起，sha 确定性相同）；受影响 cell 同参数 resume；卡死的 driver 按 tmux 会话名 kill + 按 driver 端口定点回收孤儿 worker（⛔ 不 kill tmux server 进程）。h100 π0.5 8 进程时 RAM avail 仅 3 GB → 保持 ≤7。
- ⚠ GR00T venv 的 openpi-client editable 指向已归档旧树 → `serve_groot.sh` PYTHONPATH 已加 `$REPO/packages/openpi-client/src`。
- ⚠ 「部分拉取」会把后来作废的 launch 文件带回本地 → 分析前删掉（`rc_macro13/pi05/full/launch_4355c2da3e.json` 已删）；server 端作废 run 的行成 stray（arrays 被同名新 run 覆盖 → sha 不符，属预期，不门控）。
- ⚠ 拉 server 行必须在 cell 终态之后（warmshoot 500 集一次拉早了 7 集，已重拉）。
- 满载速率比空载慢 2–3×（h100 8 进程 GPU 50%；wls warm server 检索吃 CPU ~10 核/进程）；full 单 server 每集 ~3 min → 多任务 lane 拆成多 cell 或多 server。
- 单 server 单连接：一个 cell 的 server 列表起跑后固定；调度只能靠拆任务子集成多 cell。
- 给教授的核心改口：reset 没抹掉缓存（预测错）；t 是噪声水平（对）；reset 式 = 「缓存初始化的减步推理」，起点用最终动作即可。
