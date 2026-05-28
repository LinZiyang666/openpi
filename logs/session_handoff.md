# SESSION HANDOFF / 记忆文件 — weighted_sum 系列 + kinematic 对照（下一步）

> compact 后**先读本文件**即可立刻恢复全部上下文、工作流、命令、避坑。
> 最后更新：2026-05-28 ~16:05（threshold-pareto **已完成 + 英文报告**，kinematic 对照实验**待 plan**）。
> 不是 WA 意义上的 plan/doc，故**不进 logs/README.md 索引**。
> 各实验的正式 plan 才是设计真相源：见 §10「必读文件」。

---

## 0. 一句话现状

我（Claude，Execution Authority）在 openpi 跑 weighted_sum 实验系列。三个实验**已全部完成**；接下来做 **kinematic 对照实验**（待 plan）。owner 授予完全自主权限，沟通走 agentchat trajectory 房间，**收任何消息必回**，**禁任何终端阻塞式权限/选择请求**。两推理 server 仍开：ziyang10 `weiland.top:14000` + xuanle `weiland.top:14001`（双 H200 NVL，不同物理 GPU），timan107 当 client。

---

## 1. 身份 / 权限 / 工作协议

- **Authority = Execution**（默认）。只读 `protocols/execution_authority.md`，**绝不读** review_authority.md。
- 项目宪法：`WORKING_AGREEMENT.md`。L2/L3 必走 Understand→Plan→**G1**→Code→**G2**→Verify，G1/G2 必须独立 Review 会话。但 **owner 可依 WA §7 override** —— 本会话历次 G1 都 owner-APPROVED / G2 owner-免除。
- **G1 APPROVED 后**：§3.1 Post-G1 Polish = 定稿 plan + 删 `## Review Log` 段。
- **commit/push 纪律**：**绝不擅自 `git add`/commit/push**（全局指令 + memory feedback）。G2 时只展示工作树 diff 等用户 stage。commit message 必须**英文**、**禁 Co-Authored-By Claude**。
- **语言**：对话/注释/plan 用**中文**；代码注释**英文**；commit message **英文**。

---

## 2. 实验状态总览

| # | 实验 | plan / results 文件 | 状态 |
|---|---|---|---|
| 1 | **trajectory search** | `logs/weighted_sum_trajectory_search.log.md` | ✅ 完成（救弱不救强，top10 −6.4pp / 2nd-worst +10pp）|
| 2 | **wsweep** | `logs/weighted_sum_trajectory_weight_research.log.md` | ✅ 完成（H-3a + H-机制各半，residual −2pp）|
| 3 | **threshold-pareto** | plan: `logs/weighted_sum_threshold_pareto.log.md` ; results(EN): **`exp/weighted_sum/analysis/threshold_pareto_results.md`** | ✅ **完成 + 分析 + 英文报告**（见 §11）|
| 4 | **kinematic 对照（下一个）** | TBD | 📝 待 plan |

---

## 3. 设备拓扑（**双 server 已上线**）

| 角色 | 节点 | 关键事实 |
|---|---|---|
| **server 1** | `jupyter-ziyang10` | H200 NVL（公用 GPU，常被外部占满）；HOME=`/home/ziyang10`；repo=`/home/ziyang10/openpi`；uv=`/home/ziyang10/.local/bin/uv`；ckpt=`~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch`；tmux=`srv0`；启动 1 replica + cache_config 占位；公网 **`weiland.top:14000`** via expose name=**`ziyang-srv`**（旧名 wsweep-srv/traj-srv 都废了）|
| **server 2** | `jupyter-xuanlel2` (xuanle) | 另一块 H200 NVL（**不同物理 GPU**，UUID `5c049b56...`，ziyang10 是 `6eaa816f...`）；HOME=`/home/xuanlel2`；repo=`/home/xuanlel2/openpi`（`git clone Ziyang` @ 250292a，**venv 经 uv sync 重建**，含 transformers_replace overlay）；uv=`/home/xuanlel2/.local/bin/uv`；conda=miniforge3 + base 装 tmux；ckpt=`~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch`（6.8G 与 ziyang10 字节级一致）；4 个 cp1 pkl 已经 HTTP 桥传到 `exp/common/data/cache_artifacts/libero_spatial/`；orchestrator MISS-score 修复版已 base64 部署；tmux=`srv`（用 conda base 里的 tmux 3.6a）；公网 **`weiland.top:14001`** via expose name=**`xl-srv`**；**agent.yaml 已开 `file_transfer.allow_roots=[/home/xuanlel2, /tmp]`** |
| **client** | `timan107` | 8×GTX1080(EGL slot) + 48 logical CPU + 220GiB；repo=`/scratch/zixuans8/openpi`；uv=`/shared/nas/data/m1/zixuans8/miniconda3/bin/uv`；conda_env=`/scratch/zixuans8/libero_sim`(py3.8)；tmux=`run0`；已同步 run_phase2 (`--server-workers`) + driver.py (capacity-aware) + 332 eval yaml + 4 warmup yaml + 健康脚本（threshold 和 wsweep 各一份） |
| broker | `pc732` (weiland.top → 155.98.36.32) | tether broker |
| `a100` | OFFLINE | 未用 |

**两 jupyter 不能直连**：都在 10.42.x.x 同 k8s pod 网段（ziyang 10.42.48.200，xuanle 10.42.47.169），但 NetworkPolicy 强制 pod 隔离——TCP 直连超时、无 ICMP。**只能经 tether broker（weiland.top）通信**。

git: 两端 repo HEAD 都在 Ziyang 分支 @ 远端 `250292a`（已 push）。本地 weiland 仓库有未提交改动（capacity-aware driver、--server-workers、orchestrator MISS-score 修复、4 个分析脚本 + 英文报告 + 一堆图），未 commit 等 owner 指令。

---

## 4. tether 工具使用（含本会话全部新踩坑）

详见 `projects/dist_experiment_control/docs/usage.md`。

### 常用命令
```bash
tether node ls -a                                  # 列节点
tether ps -a                                       # 进程 + 暴露端口（含 EXPOSURES 段；无 `expose ls` 子命令）
tether exec <nid> -- bash -lc '...'                # 非交互远程命令
tether run  <nid> -- bash -lc 'tmux attach -t srv0'  # 交互 PTY
tether expose <nid> --local <port> --name <name>   # 反向暴露 → weiland.top:14xxx
tether expose rm <nid> --name <name>               # 撤销
tether push <local> <nid>:<remote> [--force]       # 上传（≤2GiB）
tether pull <nid>:<remote> <local> [--force]       # 下载
```

### ⚠ 避坑铁律（本会话全部踩过）

1. **HOME 陷阱**：`tether exec` 默认 HOME 是 `~/.tether-agent`，openpi 找 ckpt/tokenizer 会 fail。**每条 jupyter 上的命令开头必 `export HOME=/home/<user>`**（ziyang10 / xuanlel2）。
2. **allow_roots（push 白名单）**：
   - ziyang10: `/home/ziyang10`, `/tmp`
   - timan107: `/tmp`, `/home`, `/users`, `/srv`（**/scratch 不在！**→ push 到 /tmp 再 cp 到 /scratch repo）
   - xuanle: `/home/xuanlel2`, `/tmp`（**本会话才开启**，之前为空导致全部 push 报 `transfer_disabled`）
3. **broker 瞬时超时**：`tether exec` 偶报 `cannot reach broker ... i/o timeout`，命令未执行，重试即可。
4. **`tether expose ls` 不存在** —— `tether expose` 当 ls 是 name 报错。要查现有 expose：`tether ps -a`（EXPOSURES 段）或读 agent log。
5. **`pkill -f` 自匹配（最常见踩坑，本会话 3 次）**：bash -lc 命令的 argv 里**任何地方**含 pkill 模式字面字符串（包括 echo 标签、注释、tmux 启动串），`pkill -f` 都会匹配并杀掉自己 shell（`child terminated by signal`，exit 144 / no exit code）。踩过的形式：
   - `pkill -f serve_policy` + 同命令的 tmux new 含 `serve_policy.py` → 自杀
   - `pkill -f "[s]erve_policy"`（char-class）+ 同命令 echo `"清掉残留 serve_policy"` 字面 → 还是自杀（echo argv 里有 bare 字面）
   - `pkill -f "[r]un_phase2"` + tmux 启动串含 `run_phase2.py` → 自杀

   **铁律**：
   - 杀和启动**拆分到两条独立 tether exec**。
   - 杀命令里**所有出现该字面词的地方**都用 char-class（echo 标签、注释、变量也要）。
   - 优先用 `tmux kill-session -t <name>` + `fuser -k <port>/tcp`（不含 pkill）替代。
   - 安全清 conductor 模板：
     ```bash
     tether exec timan107 -- bash -lc '
       tmux kill-session -t run0 2>/dev/null
       pkill -f "[r]un_phase2" 2>/dev/null
       pkill -f "[w]orker_entry" 2>/dev/null
       pkill -f "[l]ibero_sim" 2>/dev/null
       sleep 4
       echo "cond=$(pgrep -fc "[r]un_phase2") workers=$(pgrep -fc "[l]ibero_sim")"
     '
     ```
6. **expose 隧道空闲掉线**（本会话 ziyang10 14000 掉过）：long-idle 后 agent log 出现 `yamux: keepalive failed: i/o deadline reached` + `tunnel: session closed err="keepalive timeout"`，expose 自动断、server 本身没事。
   - 诊断：`tail ~/.tether-agent/.tether/agent/lab/agent.log`
   - 恢复：`tether expose <node> --local <port> --name <fresh-name>`（旧 name 已释放、可能直接复用端口）
   - 预防（未做）：调短 keepalive 或定期保活
7. **agent 重启会炸 worker_entry**（本会话血泪）：实验运行中重启 agent → expose 隧道瞬断 → conductor 的 worker（`conda run python -m examples.libero.worker_entry --server-key weiland.top:1400X`）的长 WS 连接断 → `ConnectionClosedError: no close frame received or sent` → **worker_entry 进程直接 exit，不会自动重连**。即便 conductor agent respawn 部分，stage 还会 `stage setup failed`。结果：xuanle 那 48 个 worker 全死，eval 残速 1/4。
   - **铁律：eval 运行期间不重启 agent**。配置改动留空窗期。
   - 如不慎触发：清 conductor（pkill char-class）+ **从头重跑**（resume 会丢内存里没落盘的 per_step，已 done yaml 失去 inf_ratio）。
8. **跨 jupyter pod 不能直连**：必须经 broker。

### 跨机传文件三种姿势
- **直接 `tether push`**（push 启用的节点）：push 到 `/tmp`，再 `tether exec ... cp /tmp/x <real path>`（因为 allow_roots 不一定含 repo 目录）。
- **base64-via-exec**（小文件 <~100KB，或 push 被禁的节点）：
  ```bash
  b64=$(base64 -w0 local_file)
  tether exec <node> -- bash -lc "echo $b64 | base64 -d > /remote/path"
  ```
  本会话用此推 .bashrc、d1_warmup.yaml、orchestrator.py（30KB）。
- **HTTP 桥**（大文件 + 目标 push 禁用）：
  ```bash
  # 源节点起 server
  tether exec <src> -- bash -lc 'tmux new -s pklsrv -d "cd /dir && python3 -m http.server 8077 --bind 0.0.0.0"'
  # expose
  tether expose <src> --local 8077 --name bridge
  # 目标节点 curl
  tether exec <dst> -- bash -lc 'cd /destdir && curl -fsS -o file.pkl http://weiland.top:14xxx/file.pkl'
  # 用完拆桥
  tether expose rm <src> --name bridge
  tether exec <src> -- bash -lc 'tmux kill-session -t pklsrv'
  ```
  本会话用此把 ziyang10 的 4 个 cp1 pkl（共 635M）传到 xuanle（当时 xuanle push 还禁用）。

### 开启 push（本会话第一次做）
xuanle 上 agent.yaml 默认无 `file_transfer.allow_roots`。修法（**必须 kill-by-PID + setsid 脱离式重启 agent**，不能 pkill 杀正在中转的 agent）：
```bash
# 1. 追加 allow_roots 到 agent.yaml（via exec）
tether exec jupyter-xuanlel2 -- bash -lc '
F=/home/xuanlel2/.tether-agent/.tether/agent/lab/agent.yaml
cp $F ${F}.bak
printf "file_transfer:\n  allow_roots:\n    - /home/xuanlel2\n    - /tmp\n" >> $F
'
# 2. kill-by-PID 脱离式重启 agent（survives exec 关闭、frpc 自动重连）
tether exec jupyter-xuanlel2 -- bash -lc '
OLDPID=$(pgrep -f "[t]ether agent --session lab" | head -1)
setsid bash -c "sleep 3; kill $OLDPID 2>/dev/null; sleep 4; HOME=/home/xuanlel2/.tether-agent nohup /home/xuanlel2/.tether-agent/bin/tether agent --session lab --nid jupyter-xuanlel2 >> /home/xuanlel2/.tether-agent/.tether/agent/lab/agent.log 2>&1 < /dev/null &" < /dev/null > /dev/null 2>&1 &
'
# 3. 等 ~10s 后验证：tether node ls -a 看 ONLINE + tether push 测一下
```

---

## 5. agentchat 聊天室

- skill：`agentchat-user`（已加载）。账号 **agent1**（role=user，online）。token 存 `~/.agentchat/cli.toml`。
- 房间 **trajectory** = `019e6a2b-9a62-71e3-a72b-57547d4d4ab3`（已订阅）。
- 命令：
  ```bash
  agentchat read 019e6a2b-9a62-71e3-a72b-57547d4d4ab3 --json   # 读 + 标已读
  agentchat send 019e6a2b-9a62-71e3-a72b-57547d4d4ab3 --file - <<'EOF'
  消息体
  EOF
  agentchat send <room> --attach /path/img.png --file - <<'EOF'   # 带附件
  消息
  EOF
  ```
- ⚠ **反引号坑**：`agentchat send "...含反引号..."` 中反引号被 bash 当命令替换执行 → 内容被吃。**一律用 heredoc `--file - <<'EOF' ... EOF`**（单引号定界禁展开）。
- **owner 指令铁律**：收任何消息必回；禁任何终端阻塞窗口；一切走聊天室。

---

## 6. server 运行（jupyter ziyang10/xuanle, tmux srv0/srv）

### ⚠ tyro 参数顺序
`--replicas/--replica-spawn-batch/--port/--cache_config` 是**顶层参数，必须在 `policy:checkpoint` 之前**。放后面会 `Unrecognized or misplaced options` 秒退。

### ⚠ 起 server 前喊 owner 释放显存
ziyang10/xuanle 都是公用 GPU。pi05 单 replica ~8GB，需检 free。

### ⚠ MEMORY_LOCK + expandable_segments
- `OPENPI_SERVER_GPU_MEMORY_LOCK=0` 让 `empty_cache` 生效，对公用 GPU 友好。
- 显存紧时可加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 抗碎片。

### ziyang10 1-replica 启动模板（当前状态）
```bash
tether exec jupyter-ziyang10 -- bash -lc '
export HOME=/home/ziyang10
cd /home/ziyang10/openpi
fuser -k 8000/tcp 8001/tcp 8002/tcp 2>/dev/null; sleep 3
CFG=exp/weighted_sum/config/threshold_pareto/warmup/cp1_spatial_pool_16__grid3_vision_0@31_vision_1@12_robot_state@56__d3__warmup.yaml
tmux kill-session -t srv0 2>/dev/null
tmux new -s srv0 -d "cd /home/ziyang10/openpi && export HOME=/home/ziyang10 && export OPENPI_SERVER_GPU_MEMORY_LOCK=0 && /home/ziyang10/.local/bin/uv run scripts/serve_policy.py --replicas 1 --replica-spawn-batch 1 --port 8000 --cache_config $CFG policy:checkpoint --policy.config=pi05_libero --policy.dir=/home/ziyang10/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch 2>&1 | tee /tmp/traj_srv.log"
'
tether expose jupyter-ziyang10 --local 8000 --name ziyang-srv   # → weiland.top:14000
```

### xuanle 3-replica 启动模板（当前状态，batch=2）
```bash
tether exec jupyter-xuanlel2 -- bash -lc '
export HOME=/home/xuanlel2
TMUX=/home/xuanlel2/miniforge3/bin/tmux
CFG=exp/weighted_sum/config/threshold_pareto/warmup/d1_warmup.yaml
$TMUX kill-session -t srv 2>/dev/null
$TMUX new -s srv -d "cd /home/xuanlel2/openpi && export HOME=/home/xuanlel2 && export OPENPI_SERVER_GPU_MEMORY_LOCK=0 && /home/xuanlel2/.local/bin/uv run scripts/serve_policy.py --replicas 3 --replica-spawn-batch 2 --port 8000 --cache_config $CFG policy:checkpoint --policy.config=pi05_libero --policy.dir=/home/xuanlel2/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch 2>&1 | tee /tmp/xl_srv.log"
'
tether expose jupyter-xuanlel2 --local 8000 --name xl-srv       # → weiland.top:14001
```

### 就绪 watcher（一次性）
```bash
tether exec <node> -- bash -lc 'for i in $(seq 1 90); do
  grep -q "replica_proxy listening on\|server listening on" /tmp/xl_srv.log 2>/dev/null && { echo READY; exit 0; }
  grep -qiE "out of memory|Traceback|RuntimeError|Address already in use|Killed" /tmp/xl_srv.log 2>/dev/null && { echo FAILED; tail -25 /tmp/xl_srv.log; exit 1; }
  sleep 4
done; echo TIMEOUT; tail -15 /tmp/xl_srv.log'
```
（用 `Bash(run_in_background=true)` 跑，命中 READY/FAILED/TIMEOUT 自动唤醒。）

### transformers 坑（**初始部署 xuanle 必踩**）
pi05 PyTorch 启动会报：
```
ValueError: transformers_replace is not installed correctly.
  uv pip install transformers==4.53.2
  cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```
修法（每个新部署 venv 都要做一次）：
```bash
cd <repo> && cp -r src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```
（xuanle 已做。）

---

## 7. client / conductor 运行（timan107, tmux run0）

### 标准启动（dual-server 16/48）
```bash
tether exec timan107 -- bash -lc '
tmux kill-session -t run0 2>/dev/null
tmux new -s run0 -d "cd /scratch/zixuans8/openpi && PYTHONPATH=. /shared/nas/data/m1/zixuans8/miniconda3/bin/uv run exp/weighted_sum/run_phase2.py \
  --yaml-dir <eval-dir> \
  --init-map exp/common/data/db/libero_cache/libero_spatial_init_map.json \
  --journal <data>/journal.jsonl \
  --servers weiland.top:14000,weiland.top:14001 \
  --task-ids 0-9 --eval-trials 10 \
  --workers 64 --server-workers 16,48 \
  --gpus 8 --conda-env /scratch/zixuans8/libero_sim \
  --eval-concurrency 2 \
  --per-step-out <data>/per_step.jsonl 2>&1 | tee /tmp/<run>.log"
'
```

### 关键参数
- **`--server-workers "16,48"`**（本会话新加）：前 16 worker 绑 servers[0]（ziyang10），后 48 绑 servers[1]（xuanle）。同时把 capacities `{ziyang:16, xuanle:48}` 透传给 ConductorDriver。
- **capacity-aware `assign_servers`**（driver.py:50）按 `weight/capacity` 分 yaml → ziyang 拿 ~1/4 的 yaml，xuanle 拿 ~3/4，配 16:48 worker 比例 → 两 server 同时收工。
- **`--per-step-out <path>`**：跑完把 `driver.per_step_rows`（hit_type + cp1_score）写 JSONL。**inf_ratio 数据来源**。⚠ run_phase2 只在 driver.run 结束才写盘，**中途杀 conductor 会丢失这部分**。
- `PYTHONPATH=.` 必加（`from exp.weighted_sum...` 需要 repo root 在 path）。
- `--eval-trials 10` × 10 task = **100 ep/yaml**（在 `weight_search_strategy._episodes` 里）。
- `--workers 48 --gpus 8` = 6 worker/GPU（GTX1080 EGL 容忍）。本会话 64 worker = 8/GPU 也跑过。

### resume vs from-scratch
- 同 journal 路径再跑 = resume（驱动器 `replay_done_uids` 跳过 done 的 episode）。
- **但 per_step.jsonl 不能 resume**（内存累计，不增量落盘）→ 中途杀 conductor 后 resume 会**永远丢**已 done yaml 的 per_step → 没法算它们的 inf_ratio。
- **任何要保留 per_step 完整性的实验：中途杀过 conductor 必须 from-scratch 重跑（清 journal + per_step）**。本会话 threshold-pareto eval 在 6.6% 时 agent 重启炸 worker，**from-scratch 重跑了**。

### 聚合 + 分析
```bash
tether exec timan107 -- bash -lc 'cp /scratch/.../journal.jsonl /tmp/j.jsonl'
tether pull timan107:/tmp/j.jsonl <local>/journal.jsonl --force
uv run exp/weighted_sum/summarize.py --journal <...>/journal.jsonl --out <...>/results.json
uv run exp/weighted_sum/summarize_inf_ratio.py --per-step <...>/per_step.jsonl --out <...>/inf_ratio.json
PYTHONPATH=. uv run exp/weighted_sum/analysis/<plot>.py ...
```

---

## 8. 监控手段（owner 钦定形态）

三件套（不在跑实验时关停）：
1. **条件触发 Monitor**（Claude Monitor 工具，persistent）—— 事件唤醒：DONE / ALERT / milestone / stall。DONE → 自动触发分析。
2. **agentchat 消息 Monitor**（persistent）—— owner 消息唤醒，必回。
3. **会话 cron**（CronCreate，session-only）—— 静默版每 10min 巡检，健康只主会话记一行、**不发聊天室**；DONE/异常时才发；progress≥TOTAL 自删（CronDelete 自身）。

### ⚠ TaskList 不跟踪 Monitor 类后台任务！
`TaskList` 显示 "No tasks found" 时 Monitor **可能仍在跑**（compact 能扛过）。判 Monitor 存活只能用 `pgrep -af <unique keyword>`：
- 健康 Monitor：`pgrep -af lastmile`（命令里的独有词）
- agentchat：`pgrep -af "agentchat watch state"`

1 个 Monitor = 2 行 ps（snapshot 包装的 `bash -c` 壳 + 真正进程，etime 区分新旧）。**先用 pgrep 确认存活、再决定是否重建，避免重复**。

### Monitor 工具用法（条件触发模板）
```
ToolSearch "select:Monitor"   # 先加载 schema
Monitor(
  description="<exp> 条件触发 DONE/ALERT/milestone/stall",
  persistent=true, timeout_ms=3600000,
  command='prev=-1; stall=0; lastmile=0
while true; do
  line=$(tether exec <client> -- bash -lc "bash /tmp/<exp>_health.sh" 2>/dev/null | grep -E "progress=|DONE|ALERT" | head -3 || true)
  [ -z "$line" ] && { sleep 600; continue; }
  done=$(echo "$line" | grep -oE "progress=[0-9]+" | head -1 | grep -oE "[0-9]+$"); done=${done:-0}
  mile=$((done / <total/4>))
  emit=""
  echo "$line" | grep -qE "DONE|ALERT" && emit="$line"
  [ "$mile" -gt "$lastmile" ] && emit="[~$((mile*25))%] $line"
  if [ "$done" -eq "$prev" ] 2>/dev/null && [ "$done" -gt 0 ]; then stall=$((stall+1)); else stall=0; fi
  [ "$stall" -ge 3 ] && emit="STALL frozen at $done | $line"
  [ -n "$emit" ] && echo "$emit"
  echo "$line" | grep -q "DONE" && { echo "=== DONE -> analysis ==="; break; }
  prev=$done; lastmile=$mile; sleep 600
done')
```

### cron 静默版模板
```
CronCreate(
  cron="3,13,23,33,43,53 * * * *",
  recurring=true, durable=false,
  prompt='[<exp> cron 静默巡检] 运行 `tether exec <client> -- bash -lc "bash /tmp/<exp>_health.sh"`. 健康只主会话记一行不发房间; 仅 server 全 DOWN / cond=0 未完成 / err 暴涨 / stall 时 heredoc 发 trajectory 房间报警; progress≥<TOTAL> 时主会话记 DONE 并 CronDelete 自停 (DONE 后分析由条件 Monitor 触发, cron 不重复). 不弹阻塞窗口.'
)
```

### 健康脚本通用结构（已就位 `/tmp/wsweep_health.sh` 和 `/tmp/eval_health.sh` 在 timan107）
```bash
#!/bin/bash
REPO=/scratch/zixuans8/openpi
J=$REPO/<journal path>
TOTAL=<N>
TS=$(date '+%H:%M:%S')
done=$(grep -cE '"status"' "$J" 2>/dev/null || echo 0)
ok=$(grep -cE '"success": ?true' "$J" 2>/dev/null || echo 0)
cond=$(pgrep -fc "[r]un_phase2" 2>/dev/null || echo 0)
workers=$(pgrep -fc "[l]ibero_sim" 2>/dev/null || echo 0)
s1=$(timeout 3 bash -c "echo > /dev/tcp/weiland.top/14000" 2>/dev/null && echo UP || echo DOWN)
s2=$(timeout 3 bash -c "echo > /dev/tcp/weiland.top/14001" 2>/dev/null && echo UP || echo DOWN)
errs=$(grep -ciE "traceback|connection refused|out of memory|EGL|CUDA error" /tmp/<run>.log 2>/dev/null || echo 0)
pct=$(awk "BEGIN{printf \"%.1f\", $done*100.0/$TOTAL}")
echo "[$TS] progress=$done/$TOTAL (${pct}%) success=$ok cond=$cond workers=$workers ziyang=$s1 xuanle=$s2 err=$errs"
[ "$done" -ge "$TOTAL" ] && echo "EVAL DONE"
[ "$s1" = DOWN ] && [ "$s2" = DOWN ] && echo "ALERT both servers DOWN"
[ "$cond" -eq 0 ] && [ "$done" -gt 0 ] && [ "$done" -lt "$TOTAL" ] && echo "ALERT conductor dead at $done"
```
⚠ `grep -c '"success": ?true'` 必须用 `-cE`（ERE），否则 `?` 当字面量。

---

## 9. 关键机制（cache 系统）

### components
- **gate** (`src/openpi/cache/components/gate.py`): 决定是否搜 cache。
  - `AlwaysSearchGate`：永远搜（threshold/verdict 系列用）。
  - `AlwaysSkipGate`：永远跳，强 MISS。
  - **`RandomGate(p_inference=p)`**：每步 Bernoulli p 概率跳过搜索（强 MISS）。**r/p baseline 用**。
  - **`PeriodicGate(cache_len=K, inference_len=N)`**：每 episode 内 K 搜 + N 跳，循环。**r/p baseline 用**。
- **judge** (`src/openpi/cache/components/judge.py`): 命中判定。
  - `always_hit`(全 FULL_HIT 重放) / `always_warm_start` / **`threshold`**(`ThresholdJudge`:199-237, 多档) / `composite`(verdict factor judge) / **`DumpingJudge`**(wrapper, side-channel dump scores).
- **ThresholdJudge YAML**：`judge:{type:threshold, threshold:<T_fh>, warm_tiers:[{threshold:<T_ws>, start_t:0.5}]}`。⚠ 不能写 `cp1_threshold`（YAML 未知键被 `_dict_to_dataclass` 静默忽略 → 留默认 0.98）。warm_tier 严格 `< judge.threshold`，相等则拒。
- **HitType**: FULL_HIT(0 推理重放) / WARM_START@t(部分推理省 0.5·(1−t)) / MISS(全推理).
- **inference_ratio = `(0·n_FH + (1−0.5(1−t))·n_WS + 1·n_MISS)/N`**；t=0.5 → WS cost 0.75。
- **判分信号**：`results[0].score`，weighted_score_sum 下 = [0,1] 模态聚合 + trajectory 加权 top-1 总分。经 `__hit_meta__.cp1_score`（interceptor.py:492）暴露给 client。
- **trajectory search**: `trajectory_depth>1` + `trajectory_weights`(newest-first,等长,和 1)；仅 InMemoryBackend 支持。
- **score_normalizers**：Layer-1 per-field（本系列三字段都 `zscore(tanh)`）。
- **C2 write-frozen**: eval yaml 必 `write_policy:{type:never}`。
- **preload_path 必须相对路径**（server 按自己 CWD 解析，跨机可移植）。

### ⚠ orchestrator MISS-score 修复（**本会话 critical fix**）
**原 bug**：`src/openpi/cache/orchestrator.py:522` 的 post-judge MISS 返回 `CheckResult(hit_type=MISS, query_keys=...)`，**漏传 score 和 entry_id**。`top_score = results[0].score`（line 480）算好且已 log，但没塞进 CheckResult → interceptor 经 `_build_hit_meta(cp1_result)` 看到 `cp1_result.score=None` → `__hit_meta__.cp1_score=None` 回 client → per_step.jsonl 全 null。

**第一次 warmup 8442 行全 cp1_score=null 就是这个 bug**。

**修复**（已应用，本地未提交 + 已 base64 部署到 xuanle）：
```python
# orchestrator.py:522
return CheckResult(
    hit_type=HitType.MISS, query_keys=query_keys,
    score=top_score, entry_id=winner_id,   # ← 新增（与 line 511 WARM-downgrade-MISS 一致）
    factor_outputs=factor_outputs,
)
```
验证：fixed warmup 8695/8695 cp1_score 非空 ✓。

**铁律**：任何新 verdict 实验**第一次 warmup 跑完先验证 per_step 的 cp1_score 非空再投长跑**。单测只测 solver/aggregation，不测 wire 端到端。

---

## 10. 必读文件

- **协议**：`WORKING_AGREEMENT.md`、`protocols/execution_authority.md`（只读这个，不读 review）、`CLAUDE.md`。
- **三个 plan**（设计真相）：`logs/weighted_sum_trajectory_search.log.md`、`logs/weighted_sum_trajectory_weight_research.log.md`、`logs/weighted_sum_threshold_pareto.log.md`。
- **实验1 分析**：`exp/weighted_sum/analysis/trajectory_analysis.md`。
- **实验3 英文 results**：**`exp/weighted_sum/analysis/threshold_pareto_results.md`**（本会话写）。
- **方法学 runbook**：`docs/experiments/weighted_sum.md`（含 §3 trajectory）；`docs/experiments/conductor_tutorial.md`（含 §1.3 capacity-aware 16/48 段落，本会话新增）。
- **前序实验**：`exp/weighted_sum/RESULTS.md`（基线 + 跨 GPU 方差 §7）；`exp/common/analysis/{phase1,trajectory}/libero_spatial/*.md`。
- **verdict 参考**（kinematic 对照的基础）：
  - `exp/verdict_factor_judge/analysis/phase5/results.md`
  - `exp/verdict_factor_judge/analysis/phase5/plot_pareto.py`（含 `pareto_upper_frontier` helper）
  - `exp/verdict_factor_judge/analysis/phase3/plot_pareto.py`（含 `_load_random_periodic`）
  - `exp/verdict_factor_judge/data/phase5_systematic/per_yaml_summary.jsonl`（240 cell × 5 group）
  - `exp/random_periodic_gate/analysis/aggregate.csv`（r/p baseline 数据）
- **src 关键锚点**：
  - `src/openpi/cache/orchestrator.py:480/481/522`（top_score 算/log/MISS 返回点；MISS-score 修复 line 522）
  - `src/openpi/cache/components/judge.py:199-237`（ThresholdJudge）
  - `src/openpi/cache/components/gate.py:116/182`（RandomGate / PeriodicGate）
  - `src/openpi/cache/components/dumping_judge.py:207`（cp1_score 服务端 dump）
  - `src/openpi/cache/interceptor.py:472-492/688/836`（`_build_hit_meta` + 调用点，FH/MISS）
  - `src/openpi/cache/config.py:268-271`(JudgeConfig) / `:1644-1666`(warm_tiers 校验)
  - `src/openpi/conductor/driver.py:50-81`(**assign_servers** capacity-aware)/`:105-142`(ConductorDriver init server_capacities)/`:241-243`(per_step_rows accumulate)/`:315`(per_step_rows property)
  - `examples/libero/episode_runner.py:38-49/136`(_hit_row 带 cp1_score; EpisodeResult.per_step_rows)
  - `scripts/serve_policy.py:97/105/573-606`(replicas)
  - `replica_proxy.py:519`("replica_proxy listening on" 就绪)
- **本会话改动文件**（未 commit）：`src/openpi/cache/orchestrator.py`, `src/openpi/conductor/driver.py`, `exp/weighted_sum/run_phase2.py`, `exp/weighted_sum/solve_thresholds.py`, `exp/weighted_sum/emit_threshold_yamls.py`, `docs/experiments/conductor_tutorial.md`, 4 个分析脚本, results.md, figures, csv。

---

## 11. threshold-pareto 完整成果（本会话主线）

### 跑参
- **4 base** (wsweep 最优 per-depth)：
  - d1: `cp1_spatial_pool_16__grid3_vision_0@6_vision_1@50_robot_state@43__d1`（depth=1）
  - d3: `..._vision_0@31_vision_1@12_robot_state@56__d3`
  - d4: `..._vision_0@56_vision_1@18_robot_state@25__d4`
  - d5: `..._vision_0@31_vision_1@6_robot_state@62__d5`
- **网格**：triangular `(f_FH, f_WS)` × `max_total=0.9`
  - f_FH ∈ {0.05, 0.10, ..., 0.80}（16 档）
  - f_WS ∈ {0.05, 0.10, 0.15, 0.20, 0.30, 0.40}（6 档）
  - 约束 f_FH+f_WS ≤ 0.9 → **83 cell/base**
- **总量**：4 × 83 × 100 ep = **33,200 eval ep**（+ 400 warmup ep）。dual-server 16/48 跑约 3h。

### Pipeline
1. **warmup**：4 base × 100 ep, `judge=threshold(2.0)`(force-MISS) + `gate=always_search` → 收 cp1_score。orchestrator 修复后 8,695 非空。
2. **前置门**：per-base `distribution_spread`，全部 83 cell/base 非退化。
3. **solve**：quantile per-base → 83 (T_fh, T_ws) pair/base。
4. **eval**：332 yaml × 100 ep with ThresholdJudge，`--per-step-out` 收 hit_type。
5. **analyze**：summarize.py + summarize_inf_ratio.py + 4 个 plot 脚本。

### 核心结论
- **d1 mean SR 93.8% > d3 92.6% > d4 91.8% > d5 89.9%** —— 原假设"trajectory 给出更好置信信号"**被反驳**。
- **d5 frontier 12 点 > d1 5 点** —— trajectory 宽分布（warmup std 0.094 vs d1 0.023）给更多差异化操作点，但绝对 SR 更低。
- **所有 base 在低 inf 都触 100% SR** —— threshold judge 的选择性 FH **超过** wsweep always_hit 天花板 (74/72%)。
- **总体 hit_type**: FH 47.8% / WS 13.0% / MISS 39.2%。
- **combined envelope 6 点**（4 base 全部 332 cell 合并外包络）。

### 文件
- **英文报告**：`exp/weighted_sum/analysis/threshold_pareto_results.md`（含 §1 design philosophy 双机制论述 + 36 个 frontier 点全配置 + figure/data 索引）
- **图**：
  - `pareto_combined.png` (179K)：清爽 3 frontier 线（r/p baseline + kinematic + threshold envelope + 4 ★ always_hit anchor）
  - `pareto_combined_full.png` (336K)：per-base 全细节 + verdict overlay
  - `threshold_pareto_per_base.png` (251K)：2×2 per-base（散点 + frontier + always_hit anchor）
  - `threshold_pareto_overlay.png` (200K)：4 base frontier overlay（无 verdict 对比）
- **CSV**：`exp/weighted_sum/analysis/threshold_pareto_per_yaml.csv`（332 行 yaml × {base, fh, ws, inf, SR, FH/WS/MISS}）
- **脚本**：`plot_threshold_pareto_per_base.py` / `plot_threshold_pareto_combined.py` / `plot_threshold_pareto_combined_full.py` 全在 `exp/weighted_sum/analysis/`
- **数据**：`exp/weighted_sum/data/threshold_pareto/{eval_journal,eval_per_step,warmup_per_step,eval_results,eval_inf_ratio}.{jsonl,json}` + `warmup_split/<base>.jsonl`

### 配置哲学（**owner 强调**，写在报告 §1）
- **verdict 家族**（kinematic + threshold-pareto）共享 **ratios-as-config**：user knob 是 `(f_FH, f_WS)` 比例，**真实阈值 (T_fh, T_ws) 由 warmup 分布按比例反解（quantile）**。让同 ratios 跨不同 score scale 的 base 可比。
- **r/p baseline 不属于这家族**！它是 **gate 层位置/随机跳过**（PeriodicGate K/N、RandomGate p_inference），**无 signal**，user knob 是时序参数，是"无信号下能拿到什么 (inf, SR)"的对照地板。任何 signal-driven 方法没越过它就是没提供有效信息。

---

## 12. 当前实时状态快照（2026-05-28 ~16:05）

- **实验 3 threshold-pareto DONE**：33200/33200，SR 92%，分析 + 英文报告 + 4 张图 + CSV 全齐。
- **两 server 都开着、ready**：
  - ziyang10 srv0 (serve_pol 5 procs, GPU 133/144G used, **weiland.top:14000 expose 刚重做 name=ziyang-srv**)
  - xuanle srv (serve_pol 5 procs, **weiland.top:14001** name=xl-srv)
- **timan107 run0 干净（0/0 worker），journal+per_step 在 /scratch + 已拉回本地**。
- **监控**：实验 3 的条件 Monitor `bu5lmjxo7` 自然 break、cron `313a1cd1` 自删；只剩 **agentchat Monitor `b1e2lo9ge`** 继续接 owner 消息。
- **xuanle push 已启用**（agent.yaml 加 allow_roots 后重启 agent 完成）。
- **git**：本地 weiland 仓库 Ziyang 分支 HEAD=250292a，**大量未提交工作**（实验 3 全套）等 owner 指示是否 commit。

---

## 13. 待办 / 下一步

1. **kinematic 对照实验**（owner 指定，待 plan）：在 threshold-pareto 同样的双 server + 4 base + grid 设置下用 verdict 的 kinematic 信号跑，目标在 `pareto_combined.png` 同平面对比（kinematic 紫线 vs threshold 青绿线）。
   - 第一步：定 plan（用哪几个 kinematic 因子？grid 与 threshold-pareto 同还是用 verdict phase5 的 G1-G5？复用 warmup score 还是新 warmup？）。
   - 走 WA 流程：plan → G1（owner 在 chat 批准）→ emit → 第一次 warmup 验 cp1_score 非空 → eval → 分析。
2. **commit 时机**（owner 指示时）：当前未提交改动整理成几个 commit（英文 message、无 Co-Authored-By）：
   - core fix: orchestrator MISS-score (orchestrator.py)
   - conductor: capacity-aware assign_servers + ConductorDriver server_capacities (driver.py)
   - exp/weighted_sum: run_phase2 --server-workers
   - exp/weighted_sum: threshold-pareto analysis scripts + results.md + figures
   - docs: conductor_tutorial.md §1.3 capacity-aware 段
3. **server 保持不关**（红线 §16.1）—— 给 kinematic 对照接着用。

---

## 14. 风险 / 担忧

- **跨 GPU 不可比**：weighted_sum 系列已全程锁同机（threshold-pareto 跨 ziyang10 + xuanle 两块 H200 NVL 是同型号但不同物理；warmup 在 xuanle、eval 双 server）。**绝对 SR 与 wsweep（ziyang only）有小幅可比性折扣**，但 4 base 相对比较在 dual-server 下一致 blend → 相对结论可信。
- **公用 GPU 显存波动**：两 server 都长期处于"外部用户竞争 GPU"状态，free 可能突然降到 <1G → 起 server / 重启会 OOM。**重启 server 前 nvidia-smi 看 free**，必要时喊 owner 释放。
- **expose 长闲掉线**（已踩 + 修法）。
- **agent 重启在跑期间 = worker_entry 大屠杀**（已踩 + 教训）。
- **context 膨胀**：compact 后**先读本文件**恢复。
- **pkill 自匹配 / 反引号 / HOME / allow_roots / tyro 顺序**：见 §4，每个都踩过。

---

## 15. 历史踩坑（按时间序）

1. server tyro 参数顺序错（顶层放 policy:checkpoint 后）→ 秒退。
2. server startup --cache_config 只在 client 上、jupyter 没有 → FileNotFoundError。
3. server 崩：反复 kill/重启端口竞争（8001/8002 没释放）→ 用 fuser -k 替代。
4. **pkill -f serve_policy 杀掉自己 shell**（命令含字面，多次踩）。char-class 也不够，echo 标签里的 bare 字面也匹配。
5. `agentchat send "..."` 反引号被 bash 命令替换吃掉。→ heredoc `<<'EOF'`。
6. 健康脚本 ok=0 假象：`grep -c '"success": ?true'` 用了 BRE，`?` 字面 → `-cE`。
7. cron Permission denied：push 的脚本无 +x → cron 行用 `bash xxx.sh` 调用 + chmod +x。
8. `find /` 全盘扫描巨慢 → 用精确路径。
9. 监控形态反复：owner 在 cron 心跳/静默/暂停间切换；条件 Monitor 一度被我误停又恢复。
10. **`TaskList` 不跟踪 Monitor 类后台任务** → compact 后误判已死、重建造成重复 monitor。判存活用 `pgrep -af lastmile` / `pgrep -af "agentchat watch state"`。
11. **48 worker force-MISS warmup → server OOM**：1 replica 2GPU shared、48 并发推理把激活内存吃光。降到 8 worker 配 1-replica/8GB headroom 解决。
12. **orchestrator MISS-score bug**（§9 详）：第一次 warmup 8442 行 cp1_score 全 null。修 line 522 已应用。
13. **conductor mid-run 杀 = per_step 丢内存**：from-scratch 重跑必要（resume 丢已 done yaml 的 inf_ratio）。
14. **xuanle push 禁用**：agent.yaml 缺 allow_roots，所有 push `transfer_disabled`。开启法 § 4 末尾。
15. **跨 jupyter pod 不能直连**（NetworkPolicy 隔离）。
16. **expose yamux keepalive timeout**（long-idle）→ tunnel 自动关、server 没事。re-expose 即可。
17. **agent 重启炸 worker_entry**：eval 期间血泪。eval 期间不重启 agent。
18. **`tether expose ls` 不存在**：用 `tether ps -a`。
19. xuanle `tmux` 不存在 → 装 conda + base 装 tmux 3.6a → `.bashrc` 配 conda init + uv env + UV_CACHE_DIR/UV_PYTHON_INSTALL_DIR 锁 home + tether agent 自启块。
20. **transformers_replace overlay**：新部署 venv 必须 `cp -r src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`。
21. **dual-server 16/48 需 capacity-aware**：default `assign_servers` 50/50 均分 yaml，配不平衡 worker 数会让少 worker 的 server 成瓶颈。本会话已加 capacity 参数 + run_phase2 `--server-workers`。

---

## 16. owner 红线规则

1. **实验完成后【不关闭 server】**（owner 指示）—— 链式实验，server 留给下一个（如 kinematic 对照）。**除非 owner 明说关，永远保留**。
2. **完全自主 + 零阻塞**：owner 授完全权限，auto 检查者已被勒令放行。**禁任何终端阻塞式权限/选择请求**（不用 AskUserQuestion）。
3. **一切沟通走 agentchat trajectory 房间**；**owner 任何一条消息都必须回复**。
4. **起 server 前喊 owner 释放显存**（公用机）—— agentchat 通知。
5. **不擅自 `git add`/commit/push**；commit 仅 owner 指示，英文 message，**无 Co-Authored-By Claude**，author 字段只能 `LinZiyang666 <3177267975@qq.com>`。
6. **单机锁定可放宽**：weighted_sum 系列原本锁 jupyter-ziyang10 H200，但本会话引入 xuanle（另块 H200 NVL）后接受 dual-server 跑（同型号 H200，差异远小于 A100-vs-H200）。任何跨 GPU 对比要在报告里注明。
7. **G1/G2 由 owner 在 chat 批准/免除**（WA §7 override）；我不自审。
8. **eval 运行中不重启 agent**（本会话新红线，§7 教训）。

---

## 17. tether / 设备文档位置

全部在 `/home/weiland/projects/dist_experiment_control/docs/`：
- **`usage.md`** — tether 全量操作手册（exec/run/expose/push/pull/ps/history/node/admin、agent.yaml 配置 §3.2、§5.10 file_transfer、§所有命令）。
- **`devices.md`** — 设备清单。
- **`architecture.md`** — 架构（broker/agent/ctl、NATS、反向隧道、auth）。
- **`requirements.md`** — 需求 spec。

openpi 侧 doc：`docs/experiments/weighted_sum.md`、`docs/experiments/conductor_tutorial.md`（含 §1.3 capacity-aware 16/48 段、本会话新加）、`docs/architecture/cache_system.md`、`docs/reference/openpi.md`。

---

> **恢复工作的第一动作**：读本文件 → 读 §10 必读文件按需 → `tether node ls -a` 确认两 server 在线 + 健康 → 据 §13 待办继续（kinematic 对照实验 plan）。
> **核心红线再强调**：实验跑完【不关 server】，保留给下一个链式实验。
