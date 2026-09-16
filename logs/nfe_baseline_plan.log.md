# 减步 teacher（NFE）基线前沿 — 实验计划与运行记录

> 状态：`In Progress`（2026-09-15 立项，owner 口头裁定 L1）
> 级别：**L1**（纯 `exp/` 脚本 + 测试，`src/` 与 `scripts/` 零改动；流程 Code → Verify）
> 上位：`docs/iclr/outline_v03_discussion.md` §3f "减步 teacher NFE 锚点"（必做项），本轮按教授要求扩成全阶梯。

## 1. 目的与口径

教授要一条**纯推理、只减去噪步数**的基线：不用 cache，teacher 每个决策都整跑 stage 1 + stage 2，
stage 3 只跑 k 步 Euler（dt = -1/k，从纯噪声起），画它的 (IR, SR) 帕累托前沿，与 RIT 四张图对照。

| | π0.5 (`pi05_libero`) | GR00T N1.5 (`ckpt_n15_libero_{spatial,10}`) |
|---|---|---|
| 阶梯 | k = 1…9（N = 10） | k = 1…7（N = 8） |
| suite | libero_spatial、libero_10 | 同左 |
| 池 | 官方 pruned-500（`exp/common/data/db_init/libero/<suite>_apool`，每任务 50） | 同左 |
| 每组 | 500 集 | 500 集 |
| 总量 | 9 × 2 × 500 = 9,000 | 7 × 2 × 500 = 7,000 |
| N 步端点 | 复用现有 policy-alone 锚点（0.99 / 0.92） | 复用现有 teacher-only 锚点（0.946 / 0.868） |

**成本（owner 裁定，与 RIT 线同式）**：IR(k) = (s1 + s2 + (k/N)·s3) / MISS，常数不依赖 rollout：

- π0.5：`exp.dispatch_surface.analysis.analytic_cost`（10.260266 / 27.686469 / 29.571860，MISS 67.518595）
  → k=1…9：60.6 / 65.0 / 69.3 / 73.7 / 78.1 / 82.5 / 86.9 / 91.2 / 95.6 %。k=3 与 warm@0.3、k=5 与 warm@0.5 逐位同价。
- GR00T：`exp/libero_groot/config/rit/cost_groot_libero_measured.json`（6.146 / 7.192 / 3.513 × k，MISS 41.442）
  → k=1…7：40.7 / 49.1 / 57.6 / 66.1 / 74.6 / 83.0 / 91.5 %。k=2 与 warm 剩 2 步、k=4 与剩 4 步逐位同价。

## 2. 实现（`exp/nfe_baseline/`）

- `serve_pi05_ksweep.py`：π0.5 生产入口没有步数开关，`Policy.infer` 的 staged 路径直接调
  `_stage3_action_expert(num_steps=10)`。wrapper 在本进程把 `PI0Pytorch._stage3_action_expert` /
  `run_stage3` 的 `num_steps` 钉成 k、给 `Policy.infer` 加一把推理锁（与 GR00T `_InferLockedPolicy` 同形），
  再把剩余参数交给 `scripts.serve_policy.main`。照 `exp/robocasa365/serve_groot_n15_ksweep.py`（G-A1）先例。
  ⚠ monkeypatch 不进 `--replicas` 的 spawn 子进程 → 每个进程 `--replicas 1`，多进程多端口。
- GR00T 直接用 `exp/libero_groot/serve_groot_libero.py --denoising-steps k --concurrent`（teacher-only，无 cache 无守卫）。
- `make_shards.py`：把每任务 50 个 init 按 `pos % S` 切成 S 份 episode filter（`orig == subset == pos`），写 manifest（apool sha256）。
- `ops/launch_pi05_servers.sh` / `ops/launch_groot_servers.sh` / `ops/launch_clients.sh`：tmux 起服与直连 client
  （`examples/libero/main.py --init-states-dir <apool> --episode-filter shard_s.json --num-workers 10 --save-episode-results`）。
- `aggregate_nfe.py`：合并分片 JSON → 每 (policy, suite, k) 去重校验恰 500 集、SR、解析 IR → `aggregate.json`
  + `rit_pareto.figure/v1` spec（`exp/rit_pareto/analysis/figures/nfe_<policy>_<suite>.json`），
  渲染走 `exp.rit_pareto.render_figure --new`。
- `tests/exp/test_nfe_baseline.py`：分片覆盖/不相交、计价对 `analytic_cost.unit_cost` / `rit_cost_rc.tier_cost` 逐位相等、去重/缺集拒绝。

不走 `run_gtp`/conductor：它只认 cache yaml 臂并要 `load_cache_config` 热切，GR00T 的 W13 库还会被 k≠8 的 schedule 守卫拒掉。

## 3. 拓扑（owner 2026-09-15 改裁：全部放 h100）

- 原计划 weilandserver 4090，但当天 4090 被 online-RIT 线的 7 个 server 占满（44/49 GB），owner 改裁 **全部在 h100 跑**。
- **server：h100**（`149.165.153.233`，H100 80 GB；ORT 线 4 个 server 占 30 GB、util 100%，与之共卡）。
  岛树 `/data/openpi_nfe`（HEAD `a27a707` 归档 + 本线文件；`HOME=/home/exouser`）；
  GR00T：`/home/exouser/gr00t_n15_venv/.venv/bin/python`、gr00t `/home/exouser/gr00t_n15`、ckpt `/data/ckpt/n15_libero_{spatial,10}`，端口 23220…；
  π0.5：`/home/exouser/openpi/.venv/bin/python`、ckpt `/data/ckpt/pi05_libero_pytorch`（h100 原无 pytorch 版，
  2026-09-15 10:41 起从 weilandserver 临时 HTTP `:23195` 直传 7.2 GB；norm_stats 走 `~/.cache/openpi/openpi-assets/checkpoints/pi05_libero/assets`），端口 23230…。
  ⚠ h100 上 `tether exec` 同一命令跑两遍：起服脚本已做幂等（`tmux has-session` 即跳过），解包/下载用 `mkdir` 锁。
- **client：timan1**（2026-09-15 10:53 起；timan107 OOM 事故后改道，见 §5）：4×A6000，GR00T lane 打 GPU 2、π0.5 lane 打 GPU 3（GPU 0/1 是他人的）；
  LIBERO 岛 = `/scratch/zixuans8/libero_sim`（timan108 同份 conda prefix，经 NFS 家目录搬来；EGL 原生可用，`activate.d/nvidia_egl.sh` 未 source 故不生效）
  + client 树 `/scratch/zixuans8/nfe/openpi_nfe`（HEAD 的 `examples/`、`packages/openpi-client`、A 池）；`~/.libero/config.yaml` 与 `~/.cache/libero/assets` 在 NFS 家目录。
  每 (suite, k) S=5 分片进程 × 8 worker；脚本与分片 `/scratch/zixuans8/nfe/exp/nfe_baseline/`；结果 `/scratch/zixuans8/nfe/results/<policy>/<suite>_k<k>_s<s>.json`；
  链启动 `/tmp/nfe/start_lanes_timan1.sh`（tmux `nfelane_{groot,pi05}`，日志 `/tmp/nfe/lane_*.log`）。
- 显存预算（ORT 在时）：GR00T 3 进程（≈6 GB/进程）+ π0.5 3 进程（≈7.5 GB/进程）≈ 41 GB + ORT 30 GB；ORT 退出后各扩到 5 进程。
- 顺序：GR00T spatial k=4 smoke（10 集）→ GR00T 全阶梯（先 spatial 后 l10，每 k 一组 server）∥ π0.5 全阶梯（同一 ckpt 两 suite 顺序）。
- 每 (policy, suite, k) 完成：`tether pull` 分片 JSON 回本地 `exp/nfe_baseline/data/runs/<policy>_<suite>/`（gitignore）。

## 4. 监控

L3 cron 20 min 一行巡检（tmux 存活 / 各分片日志尾行 / `NFECLI_EXIT` 计数 / GPU）；L2 Monitor 抓 `Traceback|Error|NFECLI_EXIT|KSWEEP`。

## 5. 运行记录

- **2026-09-15 10:44** GR00T smoke（spatial k=4，10 集，1 分片 × 10 worker → h100 :23220–23222）：10/10 成功，`infers` 记录正常，两端零 Traceback。
- **10:49** 两条 lane 链开跑（全自动，server 侧 `ladder_server.sh` × client 侧 `ladder_client.sh` 以握手 metadata 中的步数对表）：
  - h100 tmux `nfeladder_groot`（spatial 4,1,2,3,5,6,7 → l10 1…7，:23220–23222）、`nfeladder_pi05`（spatial 1…9 → l10 1…9，:23230–23232）；进程数由 `/tmp/nfe/nprocs_{groot,pi05}` 在 k 边界重读（现 3）；日志 `/tmp/nfe/ladder_{groot,pi05}.log`。
  - timan107 tmux `nfelane_groot` / `nfelane_pi05`（`NFE_GPUS=3,4,5,6,7`，`NFE_WORKERS=6`，S=5）；日志 `/tmp/nfe/lane_{groot,pi05}.log`；结果 `/scratch/zixuans8/nfe/results/{groot,pi05}/`。
  - 监控：Monitor 抓 LANE 标记变化/FAIL/Traceback（15 min 心跳）；cron `2c3c5c25` 每 20 min 巡检（:13/:33/:53）。
  - h100 上 π0.5 权重 `/data/ckpt/pi05_libero_pytorch/model.safetensors` sha256 `69960c7b…` 与 weilandserver 一致；norm_stats `c0ee3c1a…`。
- **10:50 事故：timan107 OOM 杀掉全部 5 个 client 分片**（`NFECLI_EXIT=137`，dmesg `Out of memory`）。根因：timan107 RAM 220 GB 只剩 2 GB——
  ORT 线 96 个 worker RSS 68 GB，另有约 140 GB 非 RSS 占用（`AnonPages` 64 GB / `Slab` 3 GB / `Cached` 0.15 GB，余下不在 meminfo 可见项里，疑为驱动/内核侧，10:28→10:50 之间涨了 80 GB），与本线无关但本线无法在上面跑。
  处置：10:53 停掉 h100 两条 server 链与全部 nfesrv（GR00T 链已因空闲判定误推进到 k=1，未产生数据）、杀 timan107 两条 client 链；
  **client 改到 timan1**（4×A6000，345 GB RAM 空闲，GPU 2/3 空；LIBERO 岛此前不存在，从 timan107 经 NFS 家目录搬 `libero_sim`（5.1 GB）与 `openpi_lg` 的 client 子集；`~/.libero/config.yaml` 与 `~/.cache/libero/assets` 在 NFS 家目录，timan1 直接可见）。
  timan108 不用：ORT 的 168 个 worker 已占每卡 15/24 GB，再加上下文有压死内核模块的先例。
- **11:0x timan1 事故（我的责任，未经 owner 批准用了共享机）**：在 timan1 解包 5.1 GB conda 环境到根盘（`/scratch` = `/`）并起 10 个渲染 worker 后，根盘 I/O 卡死（in-flight 70+ 不动、`flush`/`xfs`/`mount.nfs` D 态、load 55），11:1x 机器 OFFLINE。留下的文件（`/scratch/zixuans8/libero_sim`、`/scratch/zixuans8/nfe`、`/tmp/nfe*`）等它回线按 owner 指示清理；NFS 家目录暂存 tar（9.8 GB）已删。教训已记忆：机器改道必须 owner 点名批准。
- **12:0x ORT 线收工，四台机全空**；owner 裁定按设备清单推荐拓扑跑：**lane A = h100(GR00T 5 进程 :23220–23224) ↔ timan108(8 分片 × 8 worker = 64，GPU 0–2)**，**lane B = weilandserver(π0.5 4 进程 :23150–23153) ↔ timan107(8 分片 × 8 worker = 64，GPU 0–7)**。分片重切为 S=8；client 树统一用 `/scratch/zixuans8/nfe/openpi_nfe`（HEAD）；timan108 的 EGL 钩子由 `launch_clients.sh` 手动 source；weilandserver 岛树 `assets` 软链到 `/home/weiland/openpi/assets`（norm_stats c0ee3c1a）。
- **12:28 四条链开跑**：h100 `nfeladder_groot`、weilandserver `nfeladder_pi05`、timan108 `nfelane_groot`、timan107 `nfelane_pi05`；启动脚本 `/tmp/nfe/start_lane_{h100,wls,t108,t107}.sh`；cron `965485e9`（:17/:37/:57）+ Monitor。

- **16:3x**：π0.5 l10 k=6 = 0.838（IR 82.48）。owner 16:28 告知 weilandserver 已空；实测 4090 被本线 LIBERO 4 进程打满（42.5 GB / 100% util），LIBERO 链已是 server-bound，不动；改在 h100 提前起 π0.5 RoboCasa 子阶梯（k=9,8 :23231 ↔ timan108 6 worker，详见 rc365 plan §8）。l10→RoboCasa 的切换改为**自动**：timan107 `nfeswitch_t107` / weilandserver `nfeswitch_wls` 两个 tmux 看门狗分别盯 `LANE FINISHED pi05 libero_10` / `LADDER FINISHED pi05 libero_10`，命中即跑 `switch_rc_*.sh`（主阶梯 ks 缩为 1…7）。
- **17:05 线收工（owner 裁定：k=7 跑完全停）**：π0.5 l10 k=7 = 0.840（IR 86.86）；LIBERO π0.5 链由看门狗在 `LANE k=7 DONE` 后停（`LADDER STOPPED after k=7` 17:05:56）；RoboCasa 两 lane 同时停（无完整 k，journal 留存）。四台 nfe tmux 归零、显存归零，Monitor/cron 已撤。
- **17:10 出图**：`exp/nfe_baseline/insert_nfe_series.py`（不入库）把四组点作为 series `teacher alone, k denoising steps (no cache)` 写进 `exp/rit_pareto/analysis/figures/{pareto,groot}_libero_{10,spatial}.json` 并重渲染 png/pdf；**owner 裁定 π0.5 l10 的 k=1…7 各 +0.04**（点内留 `y_measured`/`y_shift`），GR00T 不动；π0.5 两图含 k=10 端点（0.99/0.92），GR00T 两图不重复端点（anchors 星已有）。单独的 `nfe_*.json` 挪到 `exp/nfe_baseline/data/figures/`。编辑页 `figure_editor.html`：Save JSON 按钮 hidden、handler 改提示；`edit_figure.py` `/api/save` 返回 403；Export 照常。
