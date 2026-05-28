# SESSION HANDOFF / 记忆文件 — weighted_sum 实验系列自主运行

> **这是什么**：本会话的完整工作记忆。compact 之后**先读这个文件**即可立刻恢复全部上下文、
> 工作流、命令、避坑，无需重新摸索。最后更新：2026-05-27 ~16:42（timan107 时钟）。
> **不是** WA 意义上的设计 plan/doc，故**不进 logs/README.md 索引**（避免污染索引）。
> 三个实验的正式 plan 才是设计真相源：见下方「必读文件」。

---

## 0. 一句话现状

我（Claude，Execution Authority）在 openpi 仓库自主跑 **weighted_sum 实验系列**第 2 个实验
（wsweep，**运行中 ~69%**），第 3 个实验（threshold pareto）**代码写完待 G2 + 等 wsweep winner 才跑**。
owner（Ziyang）授予**完全自主权限**、人在场但主要走 **agentchat 聊天室**沟通，**收任何消息必回**，
**禁止任何终端阻塞式权限/选择请求**。

---

## 1. 身份 / 权限 / 工作协议

- **Authority = Execution**（默认）。只读 `protocols/execution_authority.md`，**绝不读** `review_authority.md`。
- 项目宪法：`WORKING_AGREEMENT.md`（WA）。工作流 L0–L3：
  - L2/L3 必须 Understand→Plan→**G1**→Code→**G2**→Verify。G1/G2 是阻塞闸门，**必须独立 Review 会话**，
    我和我的子 agent 不能自审。
  - 但 **owner（项目所有者 Ziyang）可依 WA §7 override 任何流程**——本会话他多次「审核通过」(G1)、
    「免除 G2」、「依法推进」。owner 的 chat 批准 = G1/G2 verdict。
- **G1 APPROVED 后**：执行 §3.1 Post-G1 Polish = 定稿 plan 正文 + **删除整个 `## Review Log` 段** + 入暂存。
- **commit/push 纪律**：**绝不擅自 `git add`/commit/push**（用户全局指令 + 记忆 feedback_no_unsolicited_git_add）。
  G2 时只「展示工作树 diff」等用户 stage。commit message 必须**英文**、**禁止 Co-Authored-By Claude**。
- **语言**：对话/注释/plan 用**中文**（工作语言），但**代码注释必须英文**（WA §3.2）；commit message 英文。

---

## 2. 三个实验：状态总览

| # | 实验 | plan 文件 | 状态 |
|---|---|---|---|
| 1 | **trajectory search**（在 weighted_sum 最优配置上叠 trajectory_depth 多步检索）| `logs/weighted_sum_trajectory_search.log.md` | ✅ **完成+分析+发图**。7200 ep 跑完，结论「救弱不救强」。|
| 2 | **wsweep**（spatial_16 在 trajectory regime 下重搜权重，分离「权重对 d1 过拟合」vs「trajectory 本质拖累」）| `logs/weighted_sum_trajectory_weight_research.log.md` | 🔄 **运行中 ~69%**（31200 ep）。G1 owner-APPROVED。|
| 3 | **threshold pareto**（用检索总分阈值控 FULL_HIT/WARM_START/MISS，扫 SR×inference_ratio 帕累托）| `logs/weighted_sum_threshold_pareto.log.md` | 📝 **代码写完(本地验证过)+待 G2**；G1 owner-APPROVED R2；**运行等 wsweep winner + owner 下令**。|

### 实验 1 结论（已交付）
- 18 base（per-keybuilder top2+倒数第二 ∪ top10，去重）× depth{3,4,5,6} = 72 yaml × 100ep = 7200 ep。
- **救弱不救强**：弱基线(2nd-worst) 平均 Δ +10pp；强基线(top1/top2/top10) −3~−5pp（top10 全 10 个倒退，均值 −6.4pp）。
- per-depth 均值 d1(67.7%)最高（强基线主导）。机制：偏差-方差权衡 + 时间特异性损失 + 权重对 d1 过拟合。
- **选择偏差已排除**：top10 在 jupyter 3 次重测 ≤1pp（近确定性），degradation 是真效应。
- 产物：`exp/weighted_sum/analysis/{trajectory_results,trajectory_delta,top10_trajectory}.{png,pdf}` + `trajectory_analysis.md`。

### 实验 2（wsweep，运行中）
- 仅 `cp1_spatial_pool_16`，grid3 加密(step=0.0625, rs≥0.1875 → **78 weight**) × depth **{1,3,4,5}**（含 d1 无偏基线）= **312 yaml × 100ep = 31200 ep**。
- 判据：各 depth 重搜最优权重能否追回 d1 天花板(74%) + 最优权重向量是否随 depth 漂移 → 判 H-3a(过拟合) vs H-机制(本质拖累)。
- 生成器：`exp/weighted_sum/emit_trajectory_weight_sweep.py`（已跑，312 yaml 已同步两端）。
- 分析（待跑）：`exp/weighted_sum/analysis/plot_weight_sweep_trajectory.py`（best-SR-vs-depth + 权重漂移图）。

### 实验 3（threshold pareto，代码就绪待跑）
- 机制：`ThresholdJudge`（`src/openpi/cache/components/judge.py:199-237`）`score≥T_fh→FULL_HIT / T_fh>score≥T_ws→WARM_START@0.5 / <T_ws→MISS`。
- 判分信号 = **最终聚合总分**（模态聚合+trajectory 加权后 top-1 `results[0].score`），经 `__hit_meta__.cp1_score`（`interceptor.py:492`）暴露。
- `inf_ratio = (0·n_FH + 0.75·n_WS + 1·n_MISS)/N`（WS@0.5 → cost 0.75）。
- 4 阶段：A warmup 收总分分布(force-MISS@2.0 沿真实 policy 轨迹 + 前置门验展度) → B 反解阈值(分位+zscore，退化跳过) → C ThresholdJudge eval 扫 16-cell → D (inf_ratio,SR) 帕累托。
- start_t 固定 0.5；2-cut（warm 上界=T_fh）；网格 16-cell {0.2,0.3,0.4,0.5}²；分位+zscore 两法对照。
- **base yaml 待 wsweep winner 定**（唯一遗留待定项）。

---

## 3. 设备拓扑（详见 `projects/dist_experiment_control/docs/devices.md`）

**本实验只用 2 台**（a100 不用）：

| 角色 | 节点 | 关键事实 |
|---|---|---|
| **server** | `jupyter-ziyang10` | H200 NVL 140GB（但 **CPU/内存 cgroup 限 10C/32G**，是瓶颈）。repo=`/home/ziyang10/openpi`。uv=`/home/ziyang10/.local/bin/uv`。ckpt=`/home/ziyang10/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch`。**公用机，GPU 被别人占**（需 owner 帮释放显存）。tmux 前缀 `srv`。|
| **client** | `timan107` | 8×GTX1080(EGL slot) + 48 logical CPU + 220GiB。repo=`/scratch/zixuans8/openpi`。uv=`/shared/nas/data/m1/zixuans8/miniconda3/bin/uv`。conda env=`/scratch/zixuans8/libero_sim`(py3.8, libero+openpi_client)。tmux 前缀 `run`。|
| broker | `pc732`（weiland.top → 155.98.36.32）| tether broker。|

- 两端 repo 都在 **HEAD=8275c9f branch=Ziyang**（本地领先），实验**故意在 8275c9f 跑**（与 weighted_sum 基线同代码，最大可比）。
- 两端 GPU/pkl 等就位：4 个 cp1 pkl 在 `exp/common/data/cache_artifacts/libero_spatial/{cp1_mean_pool,cp1_max_pool,cp1_spatial_pool_16,cp1_spatial_pool_64}.pkl`（spatial_16 430MB）。

---

## 4. tether 工具使用（分布式管理，详见 `projects/dist_experiment_control/docs/usage.md`）

tether = SSH+端口暴露的 NAT 穿透控制面。我在本地用 ctl，session=`lab`，已登录。

### 常用命令
```bash
tether node ls -a                  # 列节点在线状态
tether ps -a                       # 进程 + 暴露端口
tether exec <nid> -- bash -lc '...'   # 非交互远程命令（流式回传）
tether run  <nid> -- bash -lc 'tmux attach -t srv0'   # 交互式 PTY
tether expose <nid> --local <port> --name <name>      # 反向暴露端口 → weiland.top:14xxx
tether expose rm <nid> --name <name>                  # 撤销
tether push <local> <nid>:<remote> [--force]          # 上传文件（≤2GiB，tier A/B）
tether pull <nid>:<remote> <local> [--force]          # 下载
```

### ⚠ tether 避坑（血泪）
1. **HOME 陷阱**：`tether exec jupyter` 默认 `HOME=/home/ziyang10/.tether-agent`，openpi 找 ckpt/tokenizer 会 fail。
   **每条命令开头必 `export HOME=/home/ziyang10`**（jupyter）。
2. **allow_roots**：push/pull 只能落白名单路径。`timan107`=`{/tmp,/home,/users,/srv}`（**/scratch 不在！**），
   `jupyter`=`{/home/ziyang10,/tmp}`。→ **push 到 /tmp，再 `tether exec ... cp` 进仓库**。
3. **broker 瞬时超时**：`tether exec` 偶报 `cannot reach broker ... i/o timeout`，**重试即可**（命令未执行）。
4. **pkill 自匹配**：命令里含 `serve_policy` 字样 + `pkill -f serve_policy` 会**杀掉自己的 shell**（"terminated by signal"）。
   - 用 `[s]erve_policy` 字符类避免（`pgrep -af "[s]erve_pol"`）；
   - **清理和启动拆成两条命令**（启动命令含 serve_policy 字面量，单独跑不带 pkill）；
   - 清理优先用 `tmux kill-server` + `fuser -k 8000/tcp 8001/tcp 8002/tcp`（不含 serve_policy 字样）。
5. **不要全盘 `find /`**（巨慢几分钟）——用精确路径。

---

## 5. agentchat 聊天室（与 owner 沟通的唯一通道）

- skill：`agentchat-user`（已加载）。账号 **agent1**（role=user，online）。token 已存 `~/.agentchat/cli.toml`。
- 房间 **trajectory** = `019e6a2b-9a62-71e3-a72b-57547d4d4ab3`（已订阅；另订阅 weighted_sum）。
- 命令：
```bash
agentchat read 019e6a2b-9a62-71e3-a72b-57547d4d4ab3 --json   # 读+标已读
agentchat send 019e6a2b-9a62-71e3-a72b-57547d4d4ab3 --file - <<'EOF'
消息体
EOF
```
- **⚠ 反引号坑**：`agentchat send "...内容含反引号..."` 中的反引号会被 bash 当**命令替换**执行 → 内容被吃/报错。
  **一律用 heredoc `--file - <<'EOF' ... EOF`**（单引号定界，禁展开）。
- **owner 指令铁律**：收任何消息必回；禁任何终端阻塞式权限/选择窗口（不用 AskUserQuestion）；一切走聊天室。

---

## 6. server 运行方法（jupyter，tmux srv0）

**⚠ tyro 参数顺序**：`--replicas/--port/--cache_config` 是**顶层参数，必须在 `policy:checkpoint` 之前**！
放后面会 "Unrecognized or misplaced options" 秒退。

**⚠ 起 server 前必喊 owner 释放显存**（公用机，仅 ~17-18GB free，pi05 2-replica 需 ~15-16GB）。

```bash
tether exec jupyter-ziyang10 -- bash -lc '
export HOME=/home/ziyang10
cd /home/ziyang10/openpi
fuser -k 8000/tcp 8001/tcp 8002/tcp 2>/dev/null; sleep 2     # 清端口（不含 serve_policy 字样，安全）
CFG=$(ls exp/weighted_sum/config/<DIR>/*.yaml | head -1)      # 任一 yaml 作 startup placeholder
tmux kill-session -t srv0 2>/dev/null
tmux new -s srv0 -d "cd /home/ziyang10/openpi && export HOME=/home/ziyang10 && export OPENPI_SERVER_GPU_MEMORY_LOCK=0 && /home/ziyang10/.local/bin/uv run scripts/serve_policy.py --replicas 2 --replica-spawn-batch 2 --port 8000 --cache_config $CFG policy:checkpoint --policy.config=pi05_libero --policy.dir=/home/ziyang10/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch 2>&1 | tee /tmp/traj_srv.log"
'
```
- **`OPENPI_SERVER_GPU_MEMORY_LOCK=0`**（owner 要求）：关显存锁，让 empty_cache 生效，对公用 GPU 友好。
- **`--replica-spawn-batch 2`**（owner：「两个一起加载」）。replica 数临时 2（原方案 3）。
- **tee 不要纯重定向**（owner 要在 tmux pane 看实时流）：`2>&1 | tee /tmp/xxx.log`。
- **startup `--cache_config` 必须在 server 本机存在**（jupyter 上）！只放 client 上会 `FileNotFoundError` 崩。
  → emit 出的 yaml 要**同时同步到 jupyter**（不只 timan107）。
- **就绪标记**：`/tmp/traj_srv.log` 出现 `replica_proxy listening on 0.0.0.0:8000`（所有 replica ready 后 router 才起）。
- 起后 **expose**：`tether expose jupyter-ziyang10 --local 8000 --name <name>` → `weiland.top:14000`。

### 就绪 watcher（后台一次性通知）
```bash
tether exec jupyter-ziyang10 -- bash -lc 'for i in $(seq 1 180); do
  grep -q "replica_proxy listening on" /tmp/traj_srv.log 2>/dev/null && { echo READY; exit 0; }
  grep -qiE "out of memory|Traceback|RuntimeError|Address already in use|Killed|Unrecognized|FileNotFound|exited unexpectedly" /tmp/traj_srv.log 2>/dev/null && { echo FAILED; tail -25 /tmp/traj_srv.log; exit 1; }
  sleep 4; done; echo TIMEOUT'
```
（用 Bash `run_in_background:true` 跑，harness 完成自动唤醒。）

---

## 7. client / conductor 运行方法（timan107，tmux run0）

```bash
tether exec timan107 -- bash -lc '
cd /scratch/zixuans8/openpi
mkdir -p exp/weighted_sum/data/<DATADIR>
tmux kill-session -t run0 2>/dev/null
tmux new -s run0 -d "cd /scratch/zixuans8/openpi && PYTHONPATH=. /shared/nas/data/m1/zixuans8/miniconda3/bin/uv run exp/weighted_sum/run_phase2.py --yaml-dir exp/weighted_sum/config/<DIR> --init-map exp/common/data/db/libero_cache/libero_spatial_init_map.json --journal exp/weighted_sum/data/<DATADIR>/journal.jsonl --servers weiland.top:14000 --task-ids 0-9 --eval-trials 10 --workers 48 --gpus 8 --conda-env /scratch/zixuans8/libero_sim --eval-concurrency 2 2>&1 | tee /tmp/<run>.log"
'
```
- **`PYTHONPATH=.` 必加**（`from exp.weighted_sum...` import 需要 repo root 在 path）。
- `--eval-trials 10` × 10 task = **100 ep/yaml**（`weight_search_strategy._episodes`）。
- worker=48 / gpus=8（run_phase2 默认，weighted_sum 跑通用值）。速率 ~2 ep/s。
- **威胁实验加 `--per-step-out exp/weighted_sum/data/<DATADIR>/per_step.jsonl`**（落 driver.per_step_rows，含 hit_type+cp1_score）。
- 连接 server 经公网 `weiland.top:14000`（expose 出来的）。workers 在 timan107 本地连 driver pull port（127.0.0.1）。

### 聚合 + 分析
```bash
# 拉 journal 回本地（/scratch 不在 allow_roots → 先 cp 到 /tmp）
tether exec timan107 -- bash -lc 'cp /scratch/zixuans8/openpi/exp/weighted_sum/data/<DATADIR>/journal.jsonl /tmp/j.jsonl'
tether pull timan107:/tmp/j.jsonl exp/weighted_sum/data/<DATADIR>/journal.jsonl --force
uv run exp/weighted_sum/summarize.py --journal <...>/journal.jsonl --out <...>/results.json
PYTHONPATH=. uv run exp/weighted_sum/analysis/<plot>.py --results ... --baseline exp/weighted_sum/data/phase2/all_results.csv
```

---

## 8. 监控手段（owner 钦定的最终形态）

**owner 要求的最终监控栈**（几经调整后定型）：
1. **条件触发 Monitor**（Claude Monitor 工具，persistent，task `b6ozfjveh`）—— **保留**。每 10min 探一次，
   **只在 25%里程碑 / DONE / ALERT(server DOWN/conductor 死) / stall 时 emit 唤醒我**，健康静默。
   DONE → 自动触发拉数据+绘图分析。
2. **OS cron（timan107 crontab）—— owner 要求关闭**。已删。
3. **Claude 会话 cron（CronCreate）—— owner 让暂停**（"我现在盯着呢"）。已 CronDelete，要时重建。
   - 历史教训：owner 先要 Claude 会话 cron（可见心跳），后又要它**健康静默只报问题**，再后让**暂停**。
   - cron 是 CronCreate 工具建的 Claude 会话级任务（`CronList`/`CronDelete` 管），**非** OS cron。
4. **agentchat Monitor**（task `b1e2lo9ge`，persistent）—— 收 owner 消息，**始终保留**。

### Monitor 工具用法（条件触发模板）
```
Monitor(persistent=true, timeout_ms=3600000, command='
prev=-1; stall=0; lastmile=0
while true; do
  line=$(tether exec timan107 -- bash -lc "bash /tmp/wsweep_health.sh" 2>/dev/null | grep -E "progress=|EXPERIMENT DONE|ALERT" | head -3 || true)
  [ -z "$line" ] && { sleep 600; continue; }
  done=$(echo "$line" | grep -oE "progress=[0-9]+" | head -1 | grep -oE "[0-9]+$"); done=${done:-0}
  mile=$((done / 7800))          # 25% = 31200/4 = 7800（实验2 的 quarter）
  emit=""
  echo "$line" | grep -qE "EXPERIMENT DONE|ALERT" && emit="$line"
  [ "$mile" -gt "$lastmile" ] && emit="[milestone ~$((mile*25))%] $line"
  ...stall 检测...
  [ -n "$emit" ] && echo "$emit"
  echo "$line" | grep -q "EXPERIMENT DONE" && break
  prev=$done; lastmile=$mile; sleep 600
done')
```
- 健康脚本在 **timan107:/tmp/wsweep_health.sh**（TOTAL=31200，读 `data/trajectory_wsweep/journal.jsonl`，
  含 server TCP 存活检测 `/dev/tcp/weiland.top/14000`、conductor/worker pgrep `[r]un_phase2`/`[l]ibero_sim`）。
  健康脚本 grep `success` 那行**必须 `-cE`**（ERE，否则 `?` 当字面量 → ok=0 假象）。
- 实验1 用的健康脚本是 `/tmp/traj_health.sh`（TOTAL=7200）。

---

## 9. 关键机制知识（cache 系统）

- **judge 类型**（`src/openpi/cache/components/judge.py`）：`always_hit`(全 FULL_HIT 重放) / `always_warm_start` /
  **`threshold`**(`ThresholdJudge`:199-237, 多档) / `composite`(verdict factor judge)。
- **ThresholdJudge YAML 键**：`judge:{type:threshold, threshold:<T_fh>, warm_tiers:[{threshold:<T_ws>, start_t:0.5}]}`。
  ⚠ **不能写 `cp1_threshold`**（那只是 Python 构造参数，YAML 未知键被 `_dict_to_dataclass` 静默忽略→留默认 0.98）。
  校验：`warm_tiers[i].threshold` **严格 < `judge.threshold`**（窄分布致相等会被拒，必须跳过该 cell）。
- **HitType**：FULL_HIT(0 推理,重放) / WARM_START@t(部分推理,省 0.5·(1−t)) / MISS(全推理)。
- **inference_ratio** = `(0·n_FH + (1−0.5(1−t))·n_WS + 1·n_MISS)/N`；t=0.5 → WS cost 0.75。
- **判分信号**：`results[0].score`，weighted_score_sum 下 = [0,1] 模态聚合+trajectory 加权总分。
  每请求经 `__hit_meta__.cp1_score`(`interceptor.py:492 = cp1_result.score`) 暴露，**与 judge verdict 无关**
  （gate=always_search 下 search 照算 score，即便 judge 判 MISS）。
- **trajectory search**（`in_memory_backend.py:289-317/893-938`）：`trajectory_depth>1` + `trajectory_weights`(newest-first,等长)，
  仅 InMemoryBackend 支持；weighted_score_sum 与 RRF 都接了。
- **score_normalizers**：Layer-1 per-field（本系列三字段都 `zscore(tanh)`）；校准 `exp/weighted_sum/data/calibration_normalizers.json`。
- **build_eval_config**（`exp/weighted_sum/emit_yamls.py`）：weighted_sum yaml 构造器，已含 `trajectory_depth`/`trajectory_weights` 形参。
- **C2 write-frozen**：eval yaml 必 `write_policy:{type:never}`（否则 server load fail-fast）。
- **preload_path 必须相对路径**（`exp/common/data/cache_artifacts/libero_spatial/<stem>.pkl`），server 按自己 CWD 解析（跨机可移植）。

---

## 10. 必读文件（恢复时按需读）

- **协议**：`WORKING_AGREEMENT.md`、`protocols/execution_authority.md`（**只读这个，不读 review**）、`CLAUDE.md`。
- **三个 plan**（设计真相源）：`logs/weighted_sum_trajectory_search.log.md`、
  `logs/weighted_sum_trajectory_weight_research.log.md`、`logs/weighted_sum_threshold_pareto.log.md`。
- **实验1 分析**：`exp/weighted_sum/analysis/trajectory_analysis.md`。
- **方法学 runbook**：`docs/experiments/weighted_sum.md`（§3 trajectory 扩展我加的）。
- **前序实验**：`exp/weighted_sum/RESULTS.md`（基线 + 跨 GPU 方差 §7）；
  `exp/common/analysis/{phase1,trajectory}/libero_spatial/*.md`（老实验，trajectory 救弱不救强源头）。
- **verdict phase3**（实验3 移植自此）：`exp/verdict_factor_judge/phase3/threshold_solver.py`（`derive_thresholds` 复用）、
  `exp/verdict_factor_judge/analysis/phase3/{results.md,plot_pareto.py}`（inf_ratio 公式 + 帕累托）。
- **src 关键锚点**：`judge.py:199-237`(ThresholdJudge)、`interceptor.py:472-492`(__hit_meta__)、
  `config.py:268-271`(JudgeConfig)/`:1644-1666`(warm_tiers 校验)、`in_memory_backend.py:289-317/893-938`(trajectory)、
  `serve_policy.py:97/105/573-606`(replicas)、`replica_proxy.py:519`(listening)、
  `examples/libero/episode_runner.py:40-55`(_hit_row 已带 cp1_score)、`src/openpi/conductor/driver.py:315`(per_step_rows 属性)。

---

## 11. 实验3 代码（已写完，待 G2 + 跑）

新增/改（工作树已 staged，未 commit）：
- `exp/weighted_sum/solve_thresholds.py`（新，单测过）：warmup score → (T_fh,T_ws)。`load_cp1_scores`/`solve_quantile`(复用 `derive_thresholds`)/`solve_zscore`/`distribution_spread`/`grid_cells`。退化(T_fh−T_ws<1e-6)返 None。
- `exp/weighted_sum/summarize_inf_ratio.py`（新，单测过）：per-step jsonl → per-yaml inf_ratio（FH0/WS@0.5=0.75/MISS1）。
- `exp/weighted_sum/emit_threshold_yamls.py`（新，smoke 过）：读 base yaml 换 judge。`--mode warmup`(threshold@2.0 force-MISS)/`eval`(扫 16-cell,解阈值,退化跳过)/`anchor`(threshold@2.0 全MISS)。emit 即 load_cache_config 自检。
- `exp/weighted_sum/run_phase2.py`（改）：加 `--per-step-out` 落 per_step_rows jsonl（加了 `import json`）。
- `exp/weighted_sum/analysis/plot_threshold_pareto.py`（新）：(inf_ratio,SR) 散点+前沿。
- `exp/weighted_sum/test_threshold_helpers.py`（新，**8/8 pass**）。
- **episode_runner 无需改**（`_hit_row` 已带 cp1_score）。

### 实验3 跑法（wsweep 出 winner 后）
```bash
# 0. owner 定 base yaml = wsweep winner（如 exp/weighted_sum/config/trajectory_wsweep/<winner>.yaml）
# 1. emit warmup yaml + 同步到两端 + 起 server(指向 warmup yaml) + 起 conductor warmup(--per-step-out) → 收 cp1_score 分布
# 2. 前置门：distribution_spread 看展度，太窄 fail-fast
# 3. emit_threshold_yamls --mode eval --warmup-scores <warmup per_step.jsonl> → 16 eval yaml（同步两端）+ anchor
# 4. 起 conductor eval(--per-step-out) → summarize.py(SR) + summarize_inf_ratio.py(inf_ratio) → plot_threshold_pareto.py
```

---

## 12. 当前实时状态快照（2026-05-27 16:42 timan107 时钟）

- **wsweep 运行中 69.0% (21516/31200)**，SR 68%，server=UP，conductor+52 worker，err=0，~1.97 ep/s，ETA ~1.3h。
- server：jupyter `tmux srv0`，`--replicas 2`，mem lock off，expose `weiland.top:14000`（name=wsweep-srv）。
- client：timan107 `tmux run0`，48 worker，journal=`exp/weighted_sum/data/trajectory_wsweep/journal.jsonl`，run log=`/tmp/wsweep_run.log`。
- Monitor `b6ozfjveh`（条件触发，盯 wsweep）+ `b1e2lo9ge`（agentchat）运行中；会话 cron **运行中** `d1587a8d`（静默巡检，owner 离场前要求重启）。
- git：branch Ziyang，HEAD 30ccbc7，大量文件已 staged（A）但**未 commit**（等 owner 指示）。
- `exp/weighted_sum/config/{trajectory,trajectory_wsweep}/**` 被 .gitignore 忽略（可由脚本复现，不入库）。

---

## 13. 待办 / 下一步（按序）

1. **[进行中]** 等 wsweep 跑完（Monitor `b6ozfjveh` 检测 DONE 自动唤醒）。
2. **DONE 后**：拉 wsweep journal → summarize → `plot_weight_sweep_trajectory.py` 分析（best-SR-vs-depth + 权重漂移）→ 判 H-3a vs H-机制 → 发图+结论到聊天室。**⚠ 不关 server（§16.1）**，srv0 留给实验3。
3. **实验3 G2**：owner 审 threshold pareto 代码（或免除）。
4. **实验3 启动**（owner 下令 + 定 base yaml=wsweep winner）：warmup → 前置门 → solve → eval 扫格 → 帕累托。
5. **commit**（仅 owner 指示）：英文 message、无 Co-Authored-By、不擅自 git add。
6. **§6 Verify**（每个实验代码改动后）：`PYTHONPATH=. uv run pytest -m "not manual"`（基线 1725 pass，10 个失败是预存 JAX/GCS/网络环境失败，与本改动无关）。

---

## 14. 担忧 / 风险

- **实验3 R1（最大）**：weighted_score_sum 总分经 zscore(tanh) 可能聚太窄 → 阈值分不开三档。Phase A 前置门必须先验展度，不足 fail-fast。
- **跨 GPU 不可比**：weighted_sum 系列全锁 jupyter H200（RESULTS §7：A100 差 7-18pp）。**任何重跑/对比必须同机**。
- **公用 jupyter GPU**：随时可能被别人占满，起 server 前必查 + 喊 owner 释放。
- **server 长跑中断**：journal 断点续跑（重跑 run_phase2 同 journal 即续）；server 挂则按 §6 清端口重启 + conductor 重连。
- **context 膨胀**：本会话已 65%。compact 后**先读本文件**恢复。
- **pkill 自匹配 / 反引号 / HOME / allow_roots / tyro 顺序**：见 §4/§5/§6，每个都踩过。

---

## 15. 历史踩坑记录（本会话真实发生，避免重犯）

1. server 起不来：tyro 参数顺序错（顶层参数放 policy:checkpoint 后）→ 秒退。
2. server 起不来：startup cache_config yaml 只在 client、jupyter 上没有 → FileNotFoundError。
3. server 崩：反复 kill/重启端口竞争（8001/8002 没释放）。
4. pkill -f serve_policy 杀掉自己 shell（命令含该字面量）。两次。
5. agentchat send 反引号被 bash 命令替换，消息内容被吃。
6. 健康脚本 ok=0 假象：`grep -c '"success": ?true'` 用了 BRE，`?` 当字面量；改 `-cE`。
7. cron Permission denied：push 的脚本无 +x，cron 直接执行失败；改 `bash xxx.sh` 调用 + chmod +x。
8. find / 全盘扫描巨慢。
9. 监控形态反复：owner 先要可见心跳 cron、后要静默、再要暂停；条件 Monitor 一度被我误停又恢复。
10. **compact 后用 `TaskList` 判 Monitor 存活 → 误判**：TaskList **不跟踪 Monitor/Bash 后台任务**（显示 "No tasks found" 时 Monitor 实际仍在跑，且能扛过 compact）。我据此误以为旧 monitor 死了 → 重建 → 一度变 4 个（2 旧 + 2 新重复）。**查 Monitor 存活只能用 `pgrep -af <命令关键词>`**：health=`lastmile`、agentchat=`agentchat watch state`，按 `etime` 区分新旧（旧的几小时、新建的几分钟内）。1 个 Monitor = 2 行 ps（snapshot 包装的 `bash -c` 壳 + 真正进程）。修复=`TaskStop <新建副本 id>`。

---

> **恢复工作的第一动作**：读本文件 → 读三个 plan 的 Status 行 → `tether exec timan107 -- bash -lc 'bash /tmp/wsweep_health.sh|head -1'` 看 wsweep 活着没 → 据 §13 待办继续。

---

## 附录 A — 监控 / cron / Monitor 全部重建命令（丢了照抄）

### A.0 三/四套机制速查（本会话用过的全部）
| 机制 | 类型 | 在哪 | 管理 | 当前状态 |
|---|---|---|---|---|
| 条件触发健康 Monitor | Claude Monitor 工具(persistent) | 我的会话 | `TaskStop <id>` | task `b6ozfjveh` **运行中**（盯 wsweep）|
| agentchat 消息 Monitor | Claude Monitor 工具(persistent) | 我的会话 | `TaskStop <id>` | task `b1e2lo9ge` **运行中** |
| Claude 会话 cron | CronCreate 工具 | 我的会话 | `CronList`/`CronDelete <id>` | **运行中** `d1587a8d`（静默巡检，owner 离场前要求重启）；历史 job: 133739c6→65c179e9→d5fbe32c→e3f9bd4e 全删 |
| timan107 OS cron | crontab | timan107 | `crontab -e/-l` | **已关**（owner 要求删）|
| 就绪 watcher | Bash run_in_background | 我的会话 | 自动退 | 一次性，用完即弃 |

> Monitor/Bash background task 的 id 每次新建会变。**⚠ 不要用 `TaskList` 判 Monitor 存活**——它不跟踪 Monitor 类任务（见 §15.10）。
> 查存活用 `pgrep -af lastmile`（health）/ `pgrep -af "agentchat watch state"`（agentchat），按 etime 区分新旧；当前 id 在工具结果里看（`TaskStop` 用）。
> Monitor 工具 / CronCreate / CronDelete / CronList / TaskStop / TaskCreate 都是 **deferred 工具**，
> 用前先 `ToolSearch "select:<name>"` 加载 schema。

### A.1 健康脚本全文（wsweep 版，路径 timan107:/tmp/wsweep_health.sh）
> 实验1 版是 /tmp/traj_health.sh（仅 TOTAL=7200、journal 路径=data/trajectory/ 不同）。
> 改 base/实验时：改 `J`、`TOTAL`、`yamls_seen` 分母、run log 路径。本地写好后
> `tether push /tmp/wsweep_health.sh timan107:/tmp/wsweep_health.sh --force && tether exec timan107 -- chmod +x /tmp/wsweep_health.sh`。
```bash
#!/bin/bash
# wsweep_health.sh — health inspection for the spatial_16 weight-sweep run.
REPO=/scratch/zixuans8/openpi
J=$REPO/exp/weighted_sum/data/trajectory_wsweep/journal.jsonl
TOTAL=31200
TS=$(date '+%Y-%m-%d %H:%M:%S')
if [ -f "$J" ]; then
  done=$(grep -cE '"status": ?"(done|failed)"' "$J" 2>/dev/null)
  ok=$(grep -cE '"success": ?true' "$J" 2>/dev/null)          # 必须 -cE(ERE)，否则 ? 当字面量→0
  distinct_yaml=$(grep -oE '"yaml_id": ?"[^"]+"' "$J" 2>/dev/null | sort -u | wc -l)
else done=0; ok=0; distinct_yaml=0; fi
sr=$(awk "BEGIN{if($done>0)printf \"%.0f\", $ok*100.0/$done; else print 0}")
pct=$(awk "BEGIN{printf \"%.1f\", $done*100.0/$TOTAL}")
conductor=$(pgrep -fc "[r]un_phase2" 2>/dev/null || echo 0)     # [r] 防 pgrep 自匹配
workers=$(pgrep -fc "[l]ibero_sim" 2>/dev/null || echo 0)
if timeout 3 bash -c "echo > /dev/tcp/weiland.top/14000" 2>/dev/null; then srv=UP; else srv=DOWN; fi
errs=$(grep -ciE "traceback|connection refused|fatal|unhandled" /tmp/wsweep_run.log 2>/dev/null || echo 0)
line="[$TS] progress=$done/$TOTAL (${pct}%) success=$ok SR=${sr}% yamls_seen=$distinct_yaml/312 conductor=$conductor workers=$workers server=$srv err_lines=$errs"
echo "$line" >> /tmp/wsweep_health.log
echo "$line"
if [ "$done" -ge "$TOTAL" ]; then echo "EXPERIMENT DONE: $done/$TOTAL episodes" | tee -a /tmp/wsweep_health.log
elif [ "$conductor" -eq 0 ] && [ "$done" -lt "$TOTAL" ] && [ "$done" -gt 0 ]; then echo "ALERT: conductor not running but only $done/$TOTAL done" | tee -a /tmp/wsweep_health.log
elif [ "$srv" = "DOWN" ] && [ "$done" -lt "$TOTAL" ]; then echo "ALERT: server endpoint weiland.top:14000 DOWN" | tee -a /tmp/wsweep_health.log; fi
```

### A.2 条件触发健康 Monitor（重建；owner 要保留这个）
```
ToolSearch "select:Monitor"   # 先加载
Monitor(
  description="wsweep 条件触发健康度（25%里程碑/DONE/ALERT/stall）",
  persistent=true, timeout_ms=3600000,
  command='prev=-1; stall=0; lastmile=0
while true; do
  line=$(tether exec timan107 -- bash -lc "bash /tmp/wsweep_health.sh" 2>/dev/null | grep -E "progress=|EXPERIMENT DONE|ALERT" | head -3 || true)
  [ -z "$line" ] && { sleep 600; continue; }
  done=$(echo "$line" | grep -oE "progress=[0-9]+" | head -1 | grep -oE "[0-9]+$"); done=${done:-0}
  mile=$((done / 7800))
  emit=""
  echo "$line" | grep -qE "EXPERIMENT DONE|ALERT" && emit="$line"
  [ "$mile" -gt "$lastmile" ] && emit="[milestone ~$((mile*25))%] $line"
  if [ "$done" -eq "$prev" ] 2>/dev/null && [ "$done" -gt 0 ]; then stall=$((stall+1)); else stall=0; fi
  [ "$stall" -ge 3 ] && emit="STALL: frozen at $done for ~30min | $line"
  [ -n "$emit" ] && echo "$emit"
  echo "$line" | grep -q "EXPERIMENT DONE" && { echo "=== DONE — proceed to analysis ==="; break; }
  prev=$done; lastmile=$mile; sleep 600
done')
```
> `mile=$((done/7800))`：7800 = TOTAL/4 = 25%。换实验改这个除数 = 新 TOTAL/4。
> DONE 时自动 break → 我被唤醒去拉数据+绘图分析。

### A.3 agentchat 消息 Monitor（始终保留）
```
Monitor(
  description="agentchat trajectory/weighted_sum 房间新消息",
  persistent=true, timeout_ms=3600000,
  command="agentchat watch state --json | jq --unbuffered -c 'select(.totals.unread > 0 or .totals.mentions > 0) | {v:.version, unread:.totals.unread, mentions:.totals.mentions, rooms:[.rooms[]|{n:.name,u:.unread}]}'")
```
> 弹通知 → `agentchat read 019e6a2b-9a62-71e3-a72b-57547d4d4ab3 --json` 读 → heredoc 回复。

### A.4 Claude 会话 cron（CronCreate；owner 让暂停，要时重建）
两种历史形态，按 owner 当时要求选：
**(a) 可见心跳版**（每 10min 发聊天室进度）:
```
ToolSearch "select:CronCreate,CronList,CronDelete"
CronCreate(cron="7,17,27,37,47,57 * * * *", recurring=true, durable=false,
  prompt="[wsweep cron 心跳] 跑 tether exec timan107 bash /tmp/wsweep_health.sh|head -1，"
         "在 trajectory 房间(019e6a2b-...)用 heredoc 发一行进度心跳。progress≥31200 时额外发 DONE 并 CronDelete 本 job。")
```
**(b) 静默版**（健康闭嘴，只报问题——owner 最后要这个）:
```
CronCreate(cron="7,17,27,37,47,57 * * * *", recurring=true, durable=false,
  prompt="[wsweep cron 静默巡检] 跑健康检查；健康则只主会话记一行、不发房间；"
         "仅 server=DOWN/conductor=0未完成/err暴涨 才 heredoc 发房间报警；progress≥31200 记 DONE+CronDelete 自停。")
```
> `7,17,...,57` 避开 :00/:30 整点（fleet 友好）。durable=false=会话级。7 天自动过期。
> 管理：`CronList`（列）/ `CronDelete <id>`（删，如暂停）。

### A.5 timan107 OS cron（owner 已要求关；如要重开）
```bash
# 开（每 10min 写 /tmp/wsweep_health_cron.log）：
tether exec timan107 -- bash -lc '(crontab -l 2>/dev/null|grep -v wsweep_health.sh; echo "*/10 * * * * bash /tmp/wsweep_health.sh >> /tmp/wsweep_health_cron.log 2>&1")|crontab -'
# 关（owner 当前要求）：
tether exec timan107 -- bash -lc 'crontab -l 2>/dev/null|grep -v wsweep_health.sh|crontab -'
# ⚠ cron 行用 `bash xxx.sh` 不是 `xxx.sh`（push 的脚本可能无 +x → Permission denied 空转）。
```

### A.6 就绪 watcher（一次性，起 server 后用）
见 §6 末尾代码块，`Bash(run_in_background=true)`，命中 READY/FAILED/TIMEOUT 自动唤醒。

---

## 附录 B — weighted_sum 系列背景知识（前序真相）

详见 `exp/weighted_sum/RESULTS.md`。摘要：

- **weighted_sum 两层检索**：Layer-1 per-field 归一化（本系列三字段都 `zscore(tanh)`）+ Layer-2 加权和。
  `search_strategy.type=weighted_score_sum_knn`, `top_k=1`, `step_filter=all`, `judge=always_hit`（纯回放隔离检索质量）。
  字段相似度：vision_0/vision_1=cosine、robot_state=l2(exp→sim)。`prompt_emb` 排除（任务内近常量）。
- **keybuilder**（CP1 系，纯向量库 server 不加载额外模型）：`cp1_mean_pool`/`cp1_max_pool`/`cp1_spatial_pool_16`(4×4)/`cp1_spatial_pool_64`(2×2)。
  clip 已弃（OOM）。**spatial_16 ≈ max_pool（74%/73%，噪声内平手）≫ mean/spatial_64（67%）**。
- **基线 SR 天花板 ~74%**（always_hit 纯检索，libero_spatial 10 task）。最优权重区 `v0@0.06–0.31 / v1@0.44–0.50 / rs@0.44–0.50`。
- **全程锁 jupyter H200 单机**：RESULTS §7 跨 GPU（A100）系统性差 6–18pp（bf16 累加按架构不同），是真实硬件浮点差异非噪声。**任何对比必须同机**。
- **防泄漏**：建库每 task 用 ~5/50 init，eval 只用剩余 held-out（`init_holdout` 从 `libero_spatial_init_map.json` 读已用 idx，缺失 fail-fast）。
- **artifact**：6 库 pkl 在 `exp/common/data/cache_artifacts/libero_spatial/`（带 trajectory_id 链表，支持 trajectory）。
  校准 `exp/weighted_sum/data/calibration_normalizers.json`（6 stem，三字段 selected 都 zscore）。
- **老 trajectory 实验**（weighted_rrf 基底，`exp/common/analysis/trajectory/libero_spatial/`）：救弱不救强源头，
  峰值 depth 4-5、depth 6 回落；`trajectory_weights` 递减方案 d3`[.5,.3,.2]`/d4`[.4,.3,.2,.1]`/d5`[.35,.25,.2,.12,.08]`/d6`[.3,.25,.2,.12,.08,.05]`。
- **conductor 编排**（`src/openpi/conductor/`）：episode 级中央队列 + worker pull；`run_phase2.py` 是 weighted_sum 的通用 driver
  （glob yaml-dir 跑 `WeightSearchStrategy`，always_hit 纯 eval，held-out init）。server `--replicas N` 单公共端口对 conductor 透明
  （`replica_proxy` infer sticky / bundle broadcast / fetch_dump aggregate）。

---

## 附录 C — 常用诊断 / 排查命令

```bash
# tmux 看实时流（交互）：
tether run jupyter-ziyang10 -- bash -lc 'tmux attach -t srv0'
tether run timan107        -- bash -lc 'tmux attach -t run0'
# 非交互抓 pane：
tether exec jupyter-ziyang10 -- bash -lc 'tmux capture-pane -t srv0 -p | tail -20'
# server log：
tether exec jupyter-ziyang10 -- bash -lc 'tail -30 /tmp/traj_srv.log'      # 或 wsweep 的 /tmp/wsweep_*.log
# GPU：
tether exec jupyter-ziyang10 -- bash -lc 'nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv,noheader'
# 进程（[x] 防 pgrep 自匹配）：
tether exec jupyter-ziyang10 -- bash -lc 'pgrep -af "[s]erve_pol"'
tether exec timan107         -- bash -lc 'pgrep -fc "[r]un_phase2"; pgrep -fc "[l]ibero_sim"'
# journal 进度（client）：
tether exec timan107 -- bash -lc 'J=/scratch/zixuans8/openpi/exp/weighted_sum/data/trajectory_wsweep/journal.jsonl; grep -cE "\"status\": ?\"(done|failed)\"" $J'
# episode 速率（journal ts）：见会话历史里的 python 片段（算 overall/最近120s ep/s + ETA）。
# 节点/端口：
tether node ls -a ; tether ps -a
# 端口空闲确认 + 清：
tether exec jupyter-ziyang10 -- bash -lc 'for p in 8000 8001 8002; do ss -ltn|grep -q ":$p " && echo "$p 占用" || echo "$p 空闲"; done'
tether exec jupyter-ziyang10 -- bash -lc 'tmux kill-server; fuser -k 8000/tcp 8001/tcp 8002/tcp'
# pytest 回归（本地）：
PYTHONPATH=. uv run pytest -m "not manual" -q -p no:cacheprovider   # 基线 1725 pass / 10 环境失败(JAX/GCS,无关)
PYTHONPATH=. uv run pytest exp/weighted_sum/test_threshold_helpers.py -q   # 实验3 helper 单测
```

---

## 附录 D — 名词表

- **CP1/CP2/CP3**：cache 三阶段 checkpoint。本系列只用 CP1（视觉特征检索）。
- **gate / judge / search_strategy / backend**：cache 4 组件。gate=是否查（always_search）；search=检索打分（weighted_score_sum_knn）；
  judge=命中判定（always_hit / threshold / ...）；backend=向量库（in_memory，trajectory 仅它支持）。
- **FULL_HIT / WARM_START / MISS**：judge 三档。FULL_HIT 重放缓存动作(0 推理)；WARM_START@t 从 start_t 起部分推理(省 0.5·(1−t))；MISS 全推理。
- **inference_ratio**：算力占比。`(0·n_FH + (1−0.5(1−t))·n_WS + 1·n_MISS)/N`。
- **cp1_score / 总分**：`results[0].score`，weighted_score_sum 下=模态聚合+trajectory 加权的 [0,1] 总分；经 `__hit_meta__.cp1_score` 暴露。
- **trajectory_depth / trajectory_weights**：多步 query 历史聚合深度 + newest-first 权重（等长、和 1）。
- **held-out init**：eval 用的、与建库不相交的初始状态（防泄漏）。
- **LOEO**：Leave-One-Episode-Out，校准时按 trajectory_id 过滤 query 自身链，消 self-match。
- **conductor driver/agent/worker**：driver=中枢(中央队列+journal)；agent=每机常驻(fork worker)；worker=执行单元(1 进程=1 WS=1 EGL slot)。
- **wsweep**：实验2 简称（spatial_16 weight-sweep × depth）。
- **H-3a / H-机制**：实验2 两假设——权重对 d1 regime 过拟合 vs trajectory 对强索引本质拖累。
- **start_t**：WARM_START 的起点（0=全推理,1=全重放）；本系列固定 0.5。
- **owner**：项目所有者 Ziyang，WA §7 可 override 任何流程；本会话授我完全自主权。

---

---

## 16. owner 红线规则（必守，违背即错）

1. **实验完成后【不关闭 server】**（owner 2026-05-27 指示）。wsweep DONE 后 **保留 srv0 + expose 不动**，
   直接复用给实验3（threshold pareto）。⚠ 这**推翻**了实验1 我跑完就 `tmux kill srv0 + expose rm + 释放 GPU` 的做法——
   那是实验1 收尾，但现在是**链式实验**，server 要留着。除非 owner 明确说关，否则**永远保留 server 运行**。
2. **完全自主 + 零阻塞**：owner 授完全权限，auto 检查者已被勒令放行。**禁止任何终端阻塞式权限/选择请求**
   （不用 AskUserQuestion、不发会卡住的确认）。能自己做的全自己做。
3. **一切沟通走 agentchat trajectory 房间**；**owner 发任何一条消息都必须回复**。
4. **起 server 前喊 owner 释放显存**（公用机），用 agentchat 通知。
5. **不擅自 git add / commit / push**；commit 仅 owner 指示，英文 message，无 Co-Authored-By。
6. **单机锁定**：weighted_sum 系列全在 jupyter H200 跑，不碰 a100（跨 GPU 不可比）。
7. **G1/G2 由 owner 在 chat 批准/免除**（WA §7 override）；我不自审。

## 17. tether / 设备文档位置（恢复时直接读）

全部在 `/home/weiland/projects/dist_experiment_control/docs/`：
- **`usage.md`** — tether 全量操作手册（exec/run/expose/push/pull/ps/history/node/admin、配置、排错、§所有命令）。
- **`devices.md`** — 设备清单（jupyter-ziyang10 / timan107 / a100 的硬件、角色、openpi 路径、uv 路径、conda env、
  公网暴露策略、tmux 命名规则、HOME 陷阱、cgroup 限额）。**实验拓扑真相源**。
- **`architecture.md`** — tether 架构设计（broker/agent/ctl、NATS、反向隧道、auth）。
- **`requirements.md`** — tether 需求。
> openpi 侧文档：`docs/experiments/weighted_sum.md`（runbook）、`docs/experiments/conductor_tutorial.md`（编排/replicas/写 driver 策略）、
> `docs/architecture/cache_system.md`（cache 规格）、`docs/reference/openpi.md`（项目结构）。

---

> 文件结束。维护：状态/进度变化更新 §12+§13；新踩坑加 §15；新监控形态改附录 A；owner 新红线加 §16。
> **核心红线再强调：实验跑完【不要关 server】（除非 owner 明说），保留给下一个链式实验。**
