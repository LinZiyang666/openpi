# Session Handoff — RIT-Pareto（K=2 四组已完成；K=3 四组进行中，group 1 在跑；2026-09-01 20:25 CDT）

> **一句话状态**：RIT-Pareto 线正在跑 **K=3 阶梯（FULL / WARM@0.3 / WARM@0.5，无 gate）的 4 组帕累托极限**：顺序 ① spatial-RIT（**20:17 起在跑**，timan108 tmux `k3_g1`）→ ② spatial-GST → ③ l10-RIT → ④ l10-GST；每组每臂 500 集（官方 pruned-500 池），RIT 16 臂、GST 34 格。K=2 的四组（spatial/l10 × no-gate/H-gate）今天 05:24–18:45 全部跑完、已本地化、已出图与报告。**owner /goal 授权：独断、不弹阻塞窗口、不做完不停、监控体系必须在。** 所有代码改动**未 commit**（owner 明示后才 commit/push）。
> **术语**：GST = 网格搜索阈值（分位切）；RIT = 风险索引阈值（单 δ 切风险曲线）；RIT-PL = 分段线性估计量；K3 = 三档阶梯。tau1 = 标定参考（winner 的 x₀.₉ 在查询观测下补完 9 步）。
> **规则**：未经 owner 明示绝不 `git add`/commit/push；不写记忆文件（owner 明令）；handoff 只按要求写；作者 LinZiyang666、英文 why 式 commit、无 AI 署名；共享机杀进程 ps→kill pid；中文回复英文注释；**图 = 散点 + 非支配点前沿线（owner 纠正：不逐点连线）**；server 4 replica / client 48 worker 固定（owner）。

## 0. 权威文档

| 文档 | 内容 |
|---|---|
| `exp/rit_pareto/analysis/analysis.md` | **实验报告**：Part I（K=2 四组：设计/溯源/拓扑与事故记录/spatial+l10 结果表与解读/讨论/工件布局）+ Part II（K=3：§9 设计已写，§10–12 结果待填） |
| `exp/rit_pareto/analysis/figures/pareto_rit_{libero_spatial,libero_10}.{png,pdf}` | K=2 前沿图（no gate vs H gate + GST 参考） |
| `logs/rit_pl_ir_ladder_plan.log.md` | RIT-PL 计划（commit `ca15b5e`，G1/G2 记录） |
| `logs/dispatch_surface_sonly_pivot_report.md`、`docs/iclr/latex/sonly_note.tex` | 转向报告与数学 note |
| 本文件 §5–§7 | 拓扑、恢复规程、事故记录 |

## 1. 当前状态：K=3 四组全部完成（2026-09-02 21:34），等 owner 裁决

- **实验已收官（owner 21:40 裁定）**：无任务在跑，cron 已撤、Monitor 已停，TaskList #14/#15 完成；**weilandserver `srv0` server 已于 21:44 关闭**（:23150 不监听、显存 0、tmux 会话消失）；timan108 车队已随 runner 退出（0 worker）。全部改动已作为**单一 commit `Add RIT-Pareto K=2 and K=3 frontier experiments on the pruned-500 pools` push 到 origin/Ziyang**（§6 Verify：全量 pytest 4981 passed / 14 failed / 60 skipped，14 例全为 HEAD 既有或密封 review_tests，与本次改动无关）。
- 四组结果（全部审计 OK：唯一 uid = 总量、每臂 500、0 dup、截断 0、attempt 集合一致）：
  | 组 | 集数 | ok | 本地 raw + aggregate.json |
  |---|---|---|---|
  | g1 spatial RIT | 8000 | 7048 | `exp/rit_pareto/data/runs/libero_spatial_k3_rit/` |
  | g2 spatial GST | 17000 | 16378 | `exp/rit_pareto/data/runs/libero_spatial_k3_gst/` |
  | g3 l10 RIT | 8000 | 5186 | `exp/rit_pareto/data/runs/libero_10_k3_rit/` |
  | g4+g5 l10 GST | 17000 | 13221 | `exp/rit_pareto/data/runs/libero_10_k3_gst/`（含 `contaminated_uids_1011.json`；g4 事故后剔除 30 截断集 + resume 补跑 9111 集，§7） |
- **图（owner 裁定只画图不写报告）**：`exp/rit_pareto/analysis/figures/pareto_k3_libero_spatial.{png,pdf}`（RIT 前沿 13 点 / GST 前沿 9 点 / K=2 虚线）、`pareto_k3_libero_10.{png,pdf}`（RIT 前沿 14 点 / GST 前沿 17 点 / K=2 虚线）；散点画全部臂，折线只连非支配点。重画命令：`uv run python -m exp.rit_pareto.aggregate_rit plot-k3 --suite <suite> --rit …/<suite>_k3_rit/aggregate.json --gst …/<suite>_k3_gst/aggregate.json --k2-nogate …/<suite>_ng/aggregate.json --out-dir exp/rit_pareto/analysis/figures`。
- 两 suite 共同观察（未写入 analysis.md，供 owner 参考）：低/中 IR 段 GST 前沿高于 RIT-K3（spatial IR 40–55：GST 0.91–0.97 vs RIT 0.80–0.82；l10 IR 43–66：GST 0.61–0.78 vs RIT 0.49–0.68），RIT-K3 在该段把 18–29% 决策放到 WARM@0.5，而 GST 最优格几乎不用 W0.5；高 IR 段两法齐平。
- **仍待 owner（非阻塞）**：① 是否要 analysis.md §10–12（owner 已裁只画图）；② 根治僵尸 worker 的 src 改动（`examples/libero/episode_runner.py` 在 ConnectionClosed 时重建 client）是否立项；③ 远端临时文件是否清理：timan108 `/tmp/dsp_precheck/rit_pareto/`（raw 已 sha 校验拉回本地，含 g4 备份 `*.pre_excise_142132`、`*.old` 日志）与 weilandserver `/tmp/dsp_shared/rit_pareto/`（shadow H5 唯一副本 41 GB、`h5_shards/`、`table_tau1_k3.part*`）。
- **timan108 远端 raw**：`/tmp/dsp_precheck/rit_pareto/<suite>_k3_<rule>/` + tgz/sha；本地已 sha 校验一致。

## 2. K=3 实验定义（owner 裁定）

- 阶梯：FULL_HIT → WARM@0.3（3 步）→ WARM@0.5（5 步，新增）→ MISS；verdict 取最便宜的可入档（`ThresholdJudge`：先 `threshold`，再按 `warm_tiers` 列表顺序）。**无 gate**（`always_search`）。
- 成本（`rit_k.tier_cost`，常数来自 `analytic_cost`）：FULL 10.260 / W0.3 46.818 / W0.5 52.733 / MISS 67.519 ms；IR% = Σcost / (N·MISS)。
- 两法同一部署形式（GTP 模板 + `judge: threshold` + `warm_tiers [{θ_w03,0.3},{θ_w05,0.5}]`，同一 ws pkl），只差阈值来源：
  - **RIT-K3**：`exp/rit_pareto/rit_k.py` 三层联合 pinball LP（嵌套 q_w05 ≤ q_w03 ≤ q_full，ε=0.02，α=0.05，结点梯子同 rit_pl；K=2 与 rit_pl 逐位一致有测试），单 δ → 三切点；IR 寻址 20…95 步 5 → 16 臂；切点 +∞ 或被更险档遮蔽的档从 yaml 省略。
  - **GST-K3**：百分比三元组 (fh,w3,w5)，步长 20，fh+w3+w5 ≤ 80，去 (0,0,0) → **34 格**（owner 拍板）；θ = 同一 shadow 表 s 的降序分位（`derive_thresholds` 惯例，累积份额）；部署切点重合的格去重（两 suite 均 0 跳过）。
- 标定：复用 Part I 的 150 集 tau1 shadow cohort（seed 20260901，每 task 15 个 pruned init），`build_dispatch_table --extra-warm-tiers 0.5` 重建表 `table_tau1_k3.jsonl`（多一列 `y_tau5`；guard 证明 s/y7/y10 与 K=2 表逐位一致、行序一致）。spatial 3193 行 / l10 9008 行；y5 中位 < y7 < y10 两 suite 均成立。
- 导出结果：两 suite RIT 16 臂 |gap| ≤ 0.04 pt、GST 34/34 格；l10 RIT ir20 因 full=w03 切点重合只部署 full+warm05。

## 3. 资产位置

- **weilandserver**（server 机，`ziyanglin.com:23150` 1:1 直连；`tether exec` 须 `export PATH=/usr/local/bin:/usr/bin:/bin; export HOME=/home/weiland`）：
  - tmux `srv0`：`serve_policy --replicas 4 --replica-spawn-batch 2 --port 23150`（pi05_libero，~31 GB；`/tmp/srv0.log`；重启脚本 `/tmp/dsp_shared/rit_pareto/start_eval_server.sh`，ready 签名 `replica_proxy listening on`，~4 min）。
  - `/tmp/dsp_shared/rit_pareto/<suite>/`：`shadow_manifest.json`、`shadow_pool/`、`cohort_plan.json`、`cohort_manifest.json`、`h5/shard*/`（150 H5，spatial 11 GB / l10 30 GB，**唯一副本**）、`calibration_retrieval.yaml`、`table_tau1.jsonl`（K=2）、`table_tau1_k3.jsonl`（K=3，+`.weights.npz`）、`export_tau1/`（K=2 RIT 工件）、`arms/`（K=2 yaml）、**`k3/`（K=3：`export_record.json`、`rit/*.yaml`、`gst/*.yaml`、`arm_matrix_{rit,gst}.yaml`）**；l10 另有 `table_tau1_k3.{done,part0-3}.jsonl` 与 `h5_shards/`（并行建表遗留，可删）。
  - `/tmp/dsp_shared/rit_pareto/*.sh`：全部运维脚本；`/data/rit_stage/`：给 timan108 建车队用的 staging 包（可删）；`/data/openpi_dispatch`：代码克隆（与本地工作树同步，含 K3 代码）。
- **timan108**（client 车队，`tether exec` 用 `export PATH=/scratch/zixuans8/dsp_bin:/usr/local/bin:/usr/bin:/bin`）：
  - 环境：`/scratch/zixuans8/libero_sim`（从 weilandserver 搬迁的 conda prefix，EGL 535 hook 指向 `/scratch/zixuans8/nvidia-gl`）、`/scratch/zixuans8/dsp_bin/conda`（shim）、`/scratch/zixuans8/openpi_dispatch`（代码 + `packages/openpi-client`，runner PYTHONPATH 必须前置 `packages/openpi-client/src`）、runner venv `/scratch/zixuans8/openpi/.venv`（py3.12）、A-pool `/scratch/zixuans8/openpi_dispatch/exp/common/data/db_init/libero/<suite>_apool`。
  - `/tmp/dsp_shared/rit_pareto/<suite>/{export_tau1,arms,k3}`：与 weilandserver 同路径同 sha（arm yaml 内为绝对路径）。
  - `/tmp/dsp_precheck/rit_pareto/`：`run_group.sh`（K=2）、`run_group_k3.sh`、`rit_health.sh`、`rit_health_k3.sh`；K=2 四组 raw（`<suite>_{ng,hg}/` + tgz/sha）；K=3 `<suite>_k3_<rule>/`；`libero_10_ng/journal.jsonl.pre_srvdown` 与 `contaminated_uids_srvdown_1130.json`（11:30 事故备份）。
- **timan107**：12:05 重启回归，`/tmp` 清空，环境需重建；本轮不用。
- **本地（全部未 commit）**：`exp/rit_pareto/`（`shadow_cohort.py`、`export_rit.py`、`emit_arms.py`、`rit_k.py`、`export_k3.py`、`aggregate_rit.py`、`config/task_order_*.json`、`ops/*.sh` 21 个运维脚本副本、`analysis/analysis.md`、`analysis/figures/`、`data/shadow/`、`data/runs/{libero_spatial,libero_10}_{ng,hg}/` 四组 raw + aggregate.json）；`tests/rit_pareto/`（3 文件 44 例，全绿）；additive 改动 `exp/gate_threshold_pareto/run_gtp.py`（`--judge-type/--eval-gate/--gpu-ids/--warm-tiers`）、`exp/dispatch_surface/build_dispatch_table.py`（`--noise-sidecar` 可选、`--extra-warm-tiers`）；`logs/README.md`、本文件。定级 L1（exp 脚本 + additive 标志，未动 src）。

## 4. 监控体系（compact/重启后必须重建，session 级）

- **L1** `rit_health_k3.sh <suite> <rit|gst> <total>`（timan108）：一行 progress/ok/runner/workers/server/err/**cexc**（main.py `Caught exception` 次数 = 被截断集数）/**restarts**（worker 重启数）/**bigW**（RSS ≥ 6 GB 的 worker 数）/**zomb**（没有到 :23150 established 连接的 worker 数）/freeGB；`RIT GROUP DONE`；`ALERT runner exited|server DOWN|runner dead|low memory(<30 GB)|ballooned workers(bigW≥4)|zombie workers(zomb≥2)`。err 是累计 traceback 行数，g4 已被 1011 刷到 3.6 万，**看 cexc 不看 err**。
- **L2 Monitor**（persistent，180 s，**owner 22:20 裁定：只做条件触发，不推里程碑**）：每 3 min 跑 L1，只在 ALERT、STALL(6 轮 = 18 min 冻结)、DONE、tether 连续 3 次无回复时发事件；DONE 退出。写法：`tether exec --timeout 60s timan108 -- bash -lc 'bash /tmp/dsp_precheck/rit_pareto/rit_health_k3.sh <suite> <rule> <total>'`，`grep ALERT` 直出、`d==prev` 计 stall、`RIT GROUP DONE` → `K3_GN_<SUITE>_<RULE>_DONE` 退出。
- **L3 cron** `*/20 * * * *`（**定时巡检归 cron**）：同一 L1 + `tether node ls -a`；prompt 含处置规程（§6）；compact 后存活。
- 当前挂载：**无**（21:35 全部撤除）。重启实验时按上述模板重建。TaskList：#10–#13 完成，#14 进行中（描述含当前组，cron 靠它），#15 待办。

## 5. 拓扑速查

server weilandserver 4090 48 GB：4 replica，`--replica-spawn-batch 2`，`OPENPI_SERVER_GPU_MEMORY_LOCK=0`，bundle 由 conductor 每臂热切；client timan108 3×A5000，48 worker（每 EGL worker ≈ 0.65 GB CPU 内存、~8 GB 显存/卡），`--gpus 3`；吞吐 spatial ≈ 95 ep/min、l10 ≈ 30 ep/min。tether exec 静默 10 min 上限，长跑一律 tmux；`tether push` 远端路径须绝对、父目录先建；timan108 `allow_roots=[/home /tmp /srv]`（不能 push 到 /scratch，先 push /tmp 再 mv）。

## 6. 故障处置规程（已实战验证）

- **server 死**（`pgrep -af "[s]erve_policy.py --replicas"` 为 0，`:23150` 不监听）：runner 会把在途集记成 `failed, accepted`（同一 drain ts 的一批）。步骤：停 runner（`kill -INT`，等退出，`kill -9` worker）→ 从 journal/per_step 剔除该批 uid（按 `max(ts of failed)` ±2 s 选 uid，写备份 `.pre_srvdown` 与 uid 清单）→ `start_eval_server.sh` 等 ready → 轮换日志 → 同 journal 续跑（resume 跳过完整臂，episode 级续跑）。
- **weilandserver OOM 根因**（11:30）：另一会话经 tether 在同机跑 tether 仓库 Go 测试（5789 个 `exe` 进程，unit 峰值 247 GB）→ kernel OOM 杀 replica、tmux server、tether agent（systemd `Restart=on-failure` 5 s 自动拉回）。若再发生：同上规程；建议那边加 `systemd-run -p MemoryMax=`。
- **timan108 runner/worker 死或 OOM**：`tail -40 /tmp/rit_k3_<suite>_<rule>.log`、`dmesg -T | grep -i killed`；轮换日志后同 journal 续跑。
- **GPU 被他人占满 → EGL FatalError 假失败**（timan107 事故）：`run_gtp --gpu-ids` 跳过该卡；受污染的 journal 整体作废重跑。
- **timan108 worker 内存膨胀 → OOM → 1011 截断 + 僵尸 worker**（g4 10:46–10:57 实战）：症状 = freeGB 骤降到个位数、bigW 十几个（RSS 7–12 GB，GPU 显存同涨 2–5 GB，渲染器泄漏，起因不明）、dmesg OOM kill、随后 server 端 keepalive ping 20 s 无 pong 批量 1011 关连接（同一秒一批）。后果两类：① main.py 步内 `except Exception → break`，集被截断记成终态 `failed`（client_timing `steps` < 上限）= **污染，必须剔除补跑**；② `episode_start` 在死连接上抛 → worker 报 `episode … raised`，driver 判 retriable 只重派不落 journal，但该 worker 永远拿着死连接空转刷 traceback（僵尸）。**僵尸的真正危害**：`episode_runner._ensure_client → select_bundle` 在死连接上立即抛 → driver 判 retriable 重派 → 同一僵尸再领再抛，每集 `max_episode_retries=3`（4 次）耗尽后 scheduler 静默记 `done_fail` 且**不落 journal**；6 个僵尸 19 分钟烧掉 9081 集，runner 之后以 exit 0 "完成"。因此 **zomb ≥ 2 立即按 PID kill**；runner 若在 progress < total 时退出（`ALERT runner exited`），不是故障而是队列被烧光：audit → excise 截断集 → `launch_k3_group.sh N suite rule` 同 journal resume 即可补齐。处置：`ps`→按 PID `kill -9` 膨胀（≥6 GB）与僵尸（无 :23150 连接，两次采样取交集）worker，conductor 自动重启并重派在途集（无终态记录、无损失）。不要用 pkill。根治需改 `examples/libero/episode_runner.py` 在 ConnectionClosed 时重建 client（src 改动，等 owner 裁）。
- **完整性审计**（每组完成后）：journal terminal 行 = unique uid = 总量、dup 0；failed 集的 per_step 决策数不异常少（spatial <42、l10 <100 判可疑）；per_step attempt 集合 == journal attempt；**failed 且 client_timing.steps < 上限（l10 500 / spatial 200）= 截断污染**（`audit_k3_group.py` 已内置，输出 `truncated_failed_uids`）；四组 K=2 与 K3 g1–g3 均 0 异常。

## 7. 事故记录（时间线）

01:54 K=2 group 1 在 timan107 起跑，GPU 5/7 被占 → 7 次 EGL FatalError 假失败 → 作废重跑（`--gpu-ids 0,1,2,3,4,6`）；02:50 timan107 宕机（12:05 才重启）→ 03:19 weilandserver 本机 20 worker 跑 870 集（唯一偏离 48 的区间）→ 03:37 自建 timan108 车队接管，同 journal 续跑；11:30 weilandserver 被另一会话 Go 测试打爆内存，OOM 杀 replica，group 3 剔除 48 个在途假失败后续跑；K=2 四组 18:45 全部完成，审计 0 异常。K=3：18:56 建表（并行 4 片后 20:07 完成），20:15 srv0 重启，20:16 smoke 三档均出现，20:17 group 1 起跑。

- **2026-09-02 10:41–11:15（g4 l10 GST）**：10:41 freeGB 174 → 11:01 10；19 个 worker RSS 膨胀到 7–12 GB（GPU 显存同涨），10:46–10:49 内核 OOM 杀 5 个；10:52:31/10:52:51/10:53:11/10:55:51/10:57:11 五批 server 1011 keepalive 关连接 → 30 集截断记 failed（臂 f00w20v20）+ 6 僵尸 worker。11:08 按 PID 杀 19 膨胀 worker（free 回 153 GB、显存回 10 GB/卡），11:11 杀 6 僵尸（raised 行停增），conductor 共重启 30 个 worker，cexc 定格 30，进度未中断。健康脚本/审计/剔除脚本随即固化（§4/§6）。**14:19** runner exit 0 提前退出于 7919/17000：事后从 raised 行分布（每 5 万行均匀 2632 条）确认 6 僵尸在 10:52–11:11 已把后续臂的 9081 集全部重试耗尽（36418 raised ≈ 9100 × 4）；14:21 剔除 30 截断集，14:22 `k3_g5` 同 journal resume 补跑 9111 集，**21:34 完成**（resume 段 cexc/restarts/bigW/zomb 全程 0），审计 OK。
## 8. K=2 结果摘要（详见 analysis.md Part I）

spatial：no gate 实测 IR 37→93 %、SR 0.772→0.998；H gate 40→93 %、0.906→0.992；H gate 在 40–60 % 段与 GST 同库参考持平到 −2 pt，>70 % 一致。l10：no gate 27→93 %、0.468→0.872；H gate 40→93 %、0.662→0.860；H gate 在 40–66 % 比 no gate 高 8–20 pt，vs GST 50–62 % 低 2–5 pt、74–86 % 高 2–4 pt（均在 CI 内）；顶点 ≈ 纯 teacher 0.868。
