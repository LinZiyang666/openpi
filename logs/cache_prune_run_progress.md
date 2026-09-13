# cache_prune 运行进度板

> 运行会话，2026-09-12 起。运行入口与冻结设计见 `logs/cache_prune_run_handoff.md`；
> 常驻指令与拓扑见 `logs/session_handoff.md`。本文只记本轮运行的事实与决策。

## 1. 拓扑决策（2026-09-12 13:00 CDT）

owner 要求四台设备全用；`run_concurrent.py` 原设计只认单机（server 与 sim worker 同机、loopback 端点）。
实施方案：**一个 run 目录、一个 schedule、两条 lane**，每条 lane = 一台 serving 节点 + 一台 sim 节点：

| lane | server 节点（跑 `run_concurrent --lane`） | worker 节点（跑 `run_size_eval --role agent`） | 臂 |
|---|---|---|---|
| A | h100 `149.165.153.233:23280-23284`，driver 23290 | timan108，3×A5000，5×12=60 worker（20/卡） | 20 个 rit50 臂（两 suite） |
| B | weilandserver `ziyanglin.com:23150-23154`，driver 23160 | timan107，8×GTX1080，5×12=60 worker | 20 个 cs500_success 臂（两 suite） |

- server 端不再本机 probe client 环境：**worker 首次 pull 自带 probe**（协议 wait/replan/max_steps、torch/libero 版本、
  两 suite A-pool 逐任务 sha），driver 记 census 写 `per_step.jsonl.workers.json`，wave 校验改用 census
  （`remote_fleet_evidence`）。
- worker 按任务的 suite 建环境（`EpisodeTask.experiment`），init 池按 suite 绑定
  （`--init-states-dir libero_spatial=…,libero_10=…`），所以一支车队常驻跨 suite、跨 wave。
- run.json 跨节点必须一致：两节点仓都在 `/home/weiland/openpi`（h100 是真目录，venv 软链到 `/data/venvs/openpi_weiland`），
  ckpt 在 `/data/openpi/checkpoints/pi05_libero_pytorch`（含 `assets/physical-intelligence/libero/norm_stats.json`，sha c0ee3c1a），
  h100 的 `/data` 由软链改成 bind mount（否则 `resolve()` 路径不一致）。`implementation_identity` 两节点 digest 一致
  （`ca599949…`），模型三文件 sha 一致。
- 合并 = 把 h100 的 `wave_*`/`view_*` 目录按原路径拷回 weilandserver 的 run 目录，再跑 `analyze_prune`。

## 2. 代码改动（未 commit，已 tether 推四机）

- `src/openpi/conductor/{worker,driver,agent}.py`：pull 带 probe；driver census；`WorkerSpec.probe`。
- `examples/libero/worker_entry.py`：按 suite 建环境、按 suite 绑池、`--probe`。
- `exp/ablation_study/cache_size/run_size_eval.py`：`--role all|driver|agent`、`--bind-port`、`--agent-server`、
  `--apool-dir`（worker 机本地池，对记录重哈希）、`--worker-prefix`、census 落盘。
- `exp/ablation_study/cache_prune/{concurrent_plan,run_concurrent,run_prune_eval,emit_prune_arms,prepare_membership}.py`：
  lane spec、按 lane 打包 wave、远端车队证据、`--lane`、按 lane 判完成、`validate_freeze(arms=)` 只重哈希本 lane 的库。
- `exp/ablation_study/cache_prune/ops/`：`launch_fleet.sh` / `stop_fleet.sh`（sim 机）、`run_lane.sh`（server 机）、
  `probe_lane.py`（巡检一行）、`mirror_lane_inputs.py`（按 freeze 把 lane 输入镜像到 h100）。
- 测试：`tests/ablation_study/cache_prune/test_cache_prune_lanes.py`（18）、`tests/conductor/test_census.py`（3）、
  cache_size runner +3；相关目录 1965 passed，仅 `tests/exp/test_prebuilt_matrix_backend.py` 2 个既有失败（HEAD 上就挂）。

## 3. 事实记录

- 13:10 emitter 在 weilandserver tmux `cpemit` 开跑（`/tmp/cp_emit.log`）；spatial rit50 10 点约 15 min，
  spatial cs500 每点约 7 min；l10 cs500 预计每点 20-40 min → freeze 预计 18:30-19:30 CDT。
- 13:40 单 server 试起：`serve_policy --stage1-device cuda:0 --stage2/3 meta`，加载 P05 库 <2 min，
  GPU 2.2 GiB、RSS 3.3 GiB/进程；`load_cache_config` 经公网 hairpin 正常，同路径库只加载一次。
- 13:47 mini 远端 smoke（driver@weilandserver + 2 worker@timan107，spatial P05，10 集）：10/10 完成，8 成功，
  10-24 s/集；census 2 worker、probe 齐全、FULL_HIT 100%、`steps/infers` 与 replan=5 一致；EGL 各占一卡 533 MiB。
  `infer_ms` 约 300 ms/次（含公网往返），episode 仍只需 10-25 s。
- 13:55 h100 新 venv（`uv sync --frozen`）缺 openpi 的 transformers 补丁：起 server 报 `transformers_replace is not installed`；
  按提示 `cp -r src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/` 后正常
  （GPU 2.4 GiB/进程）。这是 venv 层的准备步骤，不在仓库里。
- 吞吐预估：spatial ≈ 15 s/集/worker，60 worker ≈ 4 集/s；l10 约 2 倍；一条 lane 10,000 集约 2-3 h。

## 4. 14:00–14:40 转直跑（owner 指示）

- owner：freeze/验证是画蛇添足，有 YAML + PKL 就跑。串行 emitter 14:05 杀掉，剩余点并行导出
  （`--no-verify`，每点 ~1 min），14:20 全 40 库就绪；`run_concurrent`/`concurrent_plan`/`run_prune_eval`/
  `verify_prune` 及其测试整体删除，`emit_prune_arms` 改为「导出 + 写 YAML」，`analyze_prune` 改为直接读 journal。
- 直跑结构：每 family（suite × regime，10 臂）一次 `run_size_eval --role driver`（`ops/run_lane_direct.sh`），
  5 个独立 `serve_policy --replicas 1`（stage 2/3 在 meta，每进程 GPU ≈2.5–2.9 GB）+ 远端车队 5 端点 × 12 worker；
  lane 内 family 串行（`ops/run_lane_chain.sh`）。数据在各 server 节点
  `/data/openpi/ablation_study/cache_prune/concurrent_v1/direct/<lane>/<suite>_<regime>/{journal,per_step}.jsonl`。
- 14:17 两 lane 起跑（首轮因 `run_lane_direct.sh` 一处引号语法错误空转 1 分钟，修后重启）。
- 14:38 巡检：lane A spatial/rit50 1097/5000、110 集/min、SR 0.69；lane B spatial/cs500 367/5000、37 集/min、SR 0.89；
  两卡 util 99%，无 infra 错误。lane B 慢 3×（1080 渲染 + 大库检索）。
- 与设备清单的差异：清单 LIBERO 参数为 4 replica + 64 worker，现跑 5 进程 + 60 worker；owner 已知，未要求改。
- 14:49 lane A 接 libero_10/rit50 后 3 分钟 driver 退出（exit 1）：60 个 worker 在 `select_bundle` 上连续抛
  `ConnectionClosedError`，19,884 次失败把 4,955 集的重试全部耗尽（这类 uid 不入 journal），driver 以为跑完。
  根因：`LiberoEpisodeRunner._ensure_client` 只在 server key 变化时重连，server 按 family 重启后旧 websocket 成死连接，
  每个后续 episode 毫秒级失败。修：`run()` 任一异常后 `close()` 丢弃连接（`examples/libero/episode_runner.py`），
  推四机；两支车队重启；lane A 以 journal 续跑 libero_10/rit50（45 条已有终态保留）。spatial/rit50 5000/5000 未受影响。
- 14:59 lane B spatial/cs500 完成（5000/5000，exit 0）；15:41 lane A libero_10/rit50 完成（5000/5000，exit 0）→ lane A 全部 20 臂完成。
  lane A 数据（449 MB，压缩 24 MB）经 h100 http → weilandserver `direct/A/`；timan108 车队与 h100 server 已关（h100 显存 0）。
- 15:45 初步分析（`analysis/prelim_1545.json`，三个完整 family，全部 FULL_HIT=1.000）：
  spatial/rit50 P00 SR 0.664 → P09 0.120（dSR −0.54）；l10/rit50 0.454 → 0.246（−0.21）；spatial/cs500 0.808 → 0.606（−0.20）。
  cs500 前 30% 删除内 dSR 的 95% CI 都跨 0；rit50 spatial 到 P07（60%）才显著。lane B l10/cs500 进行中（~2,700/5000）。
- 16:19 lane B libero_10/cs500 完成（5000/5000，exit 0）→ **40 臂 × 500 集全部完成**，基础设施错误 0，FULL_HIT 全 1.000。
- 16:21 最终分析 + 8 张图（weilandserver `concurrent_v1/analysis/`），拉回本地
  `exp/ablation_study/cache_prune/data/direct_20260912/`（journal/per_step/driver.log/census/server 日志，845 MB）；
  分析 JSON 与图放 `exp/ablation_study/cache_prune/analysis/`，报告 `analysis/results.md`。
- 收尾：timan107/108 车队、weilandserver/h100 server、http 服务全部关闭（两卡显存 0）；Monitor ×2 与 cron 已停。
- 结论摘要：删 20–30% 内四个 family 都无可分辨代价；代价出现点随库大小后移（小库 30–60%，大库 50–70%）；
  删 80% 小库崩（spatial/rit50 0.66→0.12）、大库仍留大半（spatial/cs500 0.81→0.61）；失败表现为控制步数拉长。
