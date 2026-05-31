# SESSION HANDOFF — weighted_sum libero_10 复刻（实验运行中，全量恢复手册）

> ⚠ **compact 后第一件事：从头读完本文件全文**，再读 plan `logs/weighted_sum_libero10_replication.log.md` §14。
> 假设你对之前对话**零记忆**——本文件 = 你的全部上下文。最后全量重写：**2026-05-30 ~17:00 CST**。
> **当前核心**：Stage 2 在 timan107 上跑（双 server ziyang10:14000 + **xuanle:14002**，**80 worker 40,40 + MALLOC**（2026-05-30 17:57 从 96 缩：96w RAM 累积冲 197G/avail 21G 危险→滚动重启泄压+缩 80→稳态 106G/avail 112G），~67%）。监控 5 层 + cron 挂着。集群不稳（jupyter pod ~1h 掉一次，已 3 次，恢复 SOP 见 §6）。**无人值守 mandate 生效**（owner 授权全自主 + server 错误自动重启；commit 仍需 owner 明确发话）。
> compact 后：① §0 验身份/git ② §5 逐项 pgrep 验监控 ③ §1 命令验活，绝不重启在跑的 eval（journal resume）④ §13 验全栈 ⑤ 等 L2 报 STAGE2 DONE → §7 收尾 → Stage 3 → Stage 4(需 owner G2)。

---

## 目录
- §0 身份 / 权限 / git 纪律 / 无人值守 mandate
- §1 ★ 现在正在跑什么（命令 verbatim + 验活 + 重启）
- §2 ★ Stage 1 已完成产出（winner / top10 / all_results）
- §3 ★ 走过的坑 + 修复（全量）
- §4 ★ worker / RAM 调优史 + MALLOC（重点）
- §5 ★ 监控架构（5 层 + cron，全量 id + 重挂）
- §6 ★ 集群不稳 + 故障恢复 SOP（已用 3 次）
- §7 ★ Stage 2 跑完后流程 + Stage 3/4
- §8 数据资产 + 路径
- §9 设备拓扑 + RAM 铁律
- §10 agentchat
- §11 tether 避坑（全踩过）
- §12 未提交改动（diff，等 owner commit）
- §13 命令速查（copy-paste）
- §14 红线 + 恢复 checklist
- §15 实测数据值

---

## §0 身份 / 权限 / git 纪律 / 无人值守 mandate

- **Authority = Execution**（默认）。只读 `protocols/execution_authority.md`，**绝不读** `protocols/review_authority.md`（读对方=违规）。
- 项目宪法：`WORKING_AGREEMENT.md` + `CLAUDE.md`（auto-loaded）。L2 走 Understand→Plan→G1→G2→Verify→Commit。
- 本任务 plan = `logs/weighted_sum_libero10_replication.log.md`：**L2，G1 APPROVED R3 + G2 APPROVED R2**；代码已 commit `9519a79` + push `Ziyang`；§14 = 执行日志。
- **git 纪律（owner 红线）**：**绝不擅自 `git add`/commit/push/stash/reset/rebase/force-push**；commit 仅 owner 明确发话、**英文、无 Co-Authored-By Claude、author=`LinZiyang666 <3177267975@qq.com>`、owner 偏好一次结构化大 commit**。
- **语言**：对话/plan/handoff 中文；代码注释/docstring 英文（0 汉字）；commit message 英文。
- **无人值守 mandate 生效中**（owner 2026-05-30 原文授权）：可自主改任何脚本/跑任何命令；不弹终端阻塞窗口（AskUserQuestion/EnterPlanMode）；所有沟通走 agentchat libero10 房间；owner 任何消息必回。**例外仍需 owner 房间显式发话**：commit/push/git stash/reset 等写历史高危 op。
- **owner 额外授权**：server 因错误崩溃可自主重启（前提节点可达）。
- 当前 working tree **未提交改动**（见 §12）：`src/openpi/conductor/agent.py`（task_suite fix + MALLOC 默认）、`exp/weighted_sum/run_phase2.py`（task_suite fix）、`logs/session_handoff.md`（本文件）。**别 revert！等 owner 发话 commit。**

---

## §1 ★ 现在正在跑什么（compact 后逐项验活，绝不重启在跑的 eval）

### 1.1 ★ Stage 2 eval —— client = timan107，tmux `s2eval`（当前主跑）
**合并 2a+2b = 372 yaml × (10 task × 10 trial) = 37,200 episode。双 server。96 worker。约 61.5% @ 16:55。**

**验活：**
```bash
tether exec timan107 -- bash -lc 'export HOME=/home/zixuans8; bash /tmp/stage2_libero10_health.sh'
#   → [HH:MM:SS] progress=N/37200 (P%) success=K cond=3 s1=UP s2=UP err=0
tether exec timan107 -- bash -lc 'P=run_phase; echo "driver:$(pgrep -f "${P}2"|wc -l)"; W=worker_ent; echo "workers:$(pgrep -f "${W}ry"|wc -l)(=80x2=160)"; tmux ls 2>&1|grep s2eval'
```

**★完整启动命令实体（死了才重发；保留 journal 会 resume；HOME + MALLOC + 14002 缺一不可）：**
```bash
tether exec timan107 -- bash -lc '
  export HOME=/home/zixuans8
  tmux new -s s2eval -d "cd /scratch/zixuans8/openpi && export HOME=/home/zixuans8 && export PYTHONPATH=/scratch/zixuans8/openpi && export MALLOC_ARENA_MAX=2 && export MALLOC_TRIM_THRESHOLD_=134217728 && /shared/nas/data/m1/zixuans8/miniconda3/bin/uv run exp/weighted_sum/run_phase2.py --yaml-dir /tmp/stage2_libero10_yamls --init-map /tmp/libero_10_init_map.json --journal /tmp/stage2_libero10/journal.jsonl --servers weiland.top:14000,weiland.top:14002 --task-ids 0-9 --eval-trials 10 --task-suite libero_10 --server-workers 40,40 --gpus 8 --conda-env /scratch/zixuans8/libero_sim --eval-concurrency 2 2>&1 | tee -a /tmp/stage2_libero10/run.log"
'
```
- **`export HOME=/home/zixuans8` 必带**（§3-A：否则 libero 首次 import input() → EOFError，worker 全起不来）。
- **`export MALLOC_ARENA_MAX=2 && export MALLOC_TRIM_THRESHOLD_=134217728` 必带**（§4：经 agent.py os.environ 透传 worker，把 per-worker RAM 从 3.4G 压到 1.6G，防 OOM）。agent.py `_default_spawn` 现也有这俩默认，但 launch export 保险。
- **`--servers weiland.top:14000,weiland.top:14002`**（xuanle 是 14002 不是 14001！§3-C）。
- **`--server-workers 40,40`（=80 worker）**。⚠ **2026-05-30 17:57 从 96 缩到 80**：96w 长跑后 RAM 累积冲 197G/avail 21G(危险，单 worker 升 4G)→滚动重启泄压(kill 后 197G→14G 全归还 OS)+缩 80→稳态 **106G/avail 112G 安全**。RAM 是**累积型**(深检索 worker 时间累积，MALLOC 压不住工作集累积)，深检索末期若再逼近 → 再缩 64(32,32) 或滚动重启泄压。
- **`PYTHONPATH=/scratch/zixuans8/openpi` 必带**（否则 `ModuleNotFoundError: No module named 'exp'`）。
- journal = `/tmp/stage2_libero10/journal.jsonl`（断点续跑源，**别删**；重启同路径会 skip 已 done）。run.log = `/tmp/stage2_libero10/run.log`。
- 数据已在 timan107:/tmp：`/tmp/stage2_libero10_yamls/`（372 yaml）、`/tmp/libero_10_init_map.json`。
- **清残留 worker（重启前；变量法防 pkill 自杀，见 §3-H）：**
```bash
tether exec timan107 -- bash -lc 'export HOME=/home/zixuans8; tmux kill-session -t s2eval 2>/dev/null; P=run_phase; pkill -9 -f "${P}2" 2>/dev/null; W=worker_ent; pkill -9 -f "${W}ry" 2>/dev/null; sleep 5; rm -f /tmp/stage2_libero10/run.log; echo "driver:$(pgrep -f "${P}2"|wc -l) wkr:$(pgrep -f "${W}ry"|wc -l) journal:$(wc -l < /tmp/stage2_libero10/journal.jsonl)"'
```

### 1.2 Server #1 —— jupyter-ziyang10（tmux `srv0`，3 replica，@9519a79，→14000）
**验活：**
```bash
tether exec jupyter-ziyang10 -- bash -lc 'export HOME=/home/ziyang10; tmux ls 2>&1|grep srv0; tail -3 /tmp/srv0.log; git -C /home/ziyang10/openpi rev-parse --short HEAD'
nc -zv weiland.top 14000 2>&1 | tail -1
```
**★完整启动命令（pod 重启后 serve 没了/placeholder 没了 才重发；placeholder 现放 NFS home 持久）：**
```bash
tether exec jupyter-ziyang10 -- bash -lc '
  export HOME=/home/ziyang10; cd /home/ziyang10/openpi
  tmux kill-session -t srv0 2>/dev/null
  tmux new -s srv0 -d "cd /home/ziyang10/openpi && export HOME=/home/ziyang10 && export OPENPI_SERVER_GPU_MEMORY_LOCK=0 && /home/ziyang10/.local/bin/uv run scripts/serve_policy.py --replicas 3 --replica-spawn-batch 2 --port 8000 --cache_config /home/ziyang10/stage1_placeholder.yaml policy:checkpoint --policy.config=pi05_libero --policy.dir=/home/ziyang10/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch 2>&1 | tee /tmp/srv0.log"
'
```
**expose（pod 重启后隧道死，换新 name 重做；broker 可能给新端口！见 §3-C）：**
```bash
tether expose jupyter-ziyang10 --local 8000 --name ws6-zy   # 看输出实际端口，期望 14000
```
- ready 签名（log 出现即就绪）：`replica_proxy listening on 0.0.0.0:8000 -> [8001, 8002, 8003]`。
- placeholder 缺失则重建：`tether push <local 任一 stage2 spatial_16 d1 yaml> jupyter-ziyang10:/home/ziyang10/stage1_placeholder.yaml`。

### 1.3 Server #2 —— jupyter-xuanlel2（tmux `srv0`，**3 replica（owner 指令，原 2）**，→14002）
**⚠ xuanle 无系统 tmux/fuser → 用全路径 `/home/xuanlel2/miniforge3/bin/tmux`；NFS uid 坑（dump root 用 /tmp）。**
**验活：**
```bash
tether exec jupyter-xuanlel2 -- bash -lc 'export HOME=/home/xuanlel2; /home/xuanlel2/miniforge3/bin/tmux ls 2>&1|grep srv0; tail -3 /tmp/srv0.log'
nc -zv weiland.top 14002 2>&1 | tail -1
```
**★完整启动命令（3 replica）：**
```bash
tether exec jupyter-xuanlel2 -- bash -lc '
  export HOME=/home/xuanlel2; cd /home/xuanlel2/openpi
  TMUX=/home/xuanlel2/miniforge3/bin/tmux
  $TMUX kill-session -t srv0 2>/dev/null
  $TMUX new -s srv0 -d "cd /home/xuanlel2/openpi && export HOME=/home/xuanlel2 && export OPENPI_SERVER_GPU_MEMORY_LOCK=0 && /home/xuanlel2/.local/bin/uv run scripts/serve_policy.py --replicas 3 --replica-spawn-batch 2 --port 8000 --cache_config /home/xuanlel2/stage1_placeholder.yaml policy:checkpoint --policy.config=pi05_libero --policy.dir=/home/xuanlel2/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch 2>&1 | tee /tmp/srv0.log"
'
```
**expose（xuanle agent state_write_failed 偶发，重试即成，broker 给的端口可能不是 14001！见 §3-C）：**
```bash
for i in 1 2 3; do tether expose jupyter-xuanlel2 --local 8000 --name ws7-xl-$i 2>&1 | tail -1 | grep -q exposed && break; sleep 4; done
# 看输出实际端口；若不是 14001/14002，改 health s2 探针 + conductor --servers
```
- ready 签名：`All 3 replicas ready; ... replica_proxy listening on 0.0.0.0:8000 -> [8001, 8002, 8003]`。
- xuanle 3 replica 稳态 host RAM ~26G（cgroup 32G，紧但 fit，无 OOM）。
- placeholder：`tether push <yaml> jupyter-xuanlel2:/home/xuanlel2/stage1_placeholder.yaml`（NFS home 持久）。

### 1.4 监控 5 层 + cron（见 §5 全量；compact 后必逐项 pgrep 验，死了重挂）
一键验全栈：
```bash
echo "L2:$(pgrep -fc 'M25=|STALL_THRESHOLD') L5:$(pgrep -fc 'tail -F /tmp/srv0.log') L4:$(pgrep -fc 'agentchat watch state') RAMv2:$(pgrep -fc 'RAM WARNING|10G thresh')"
# L2≥1(bf3l56qkb), L5≥2(b4dfwbxii), L4≥1, RAMv2≥1(b0v974yde)；CronList 看 b4cf0ba2
```

---

## §2 ★ Stage 1 已完成产出（winner / top10 / all_results）

- **base eval**：136 yaml(4 keybuilder × 34 权重) × 100 ep = 13,600，mean SR 33.9%，零 MISS。
- **refine 只跑 1 轮**（owner 砍 round2/3）：104 yaml(spatial_16 + mean_pool 平局双链，各 52 yaml) × 100 = 10,400，没超过 base 的 52%。
- **★winner = `cp1_spatial_pool_16` d1-best = 52.0%**。两个 52% 并列（都 vision-only，rs=0，zscore）：
  - `v0=0.50 v1=0.50 rs=0.00`（baseline）
  - `v0=0.12 v1=0.87 rs=0.00`（baseline）
  - per-kb max：spatial_16=52% > mean_pool=48% > spatial_64=47% > max_pool=46%。
- **all_results.csv** = `exp/weighted_sum/data/phase2/libero_10/all_results.csv`（240 config = base 136 + r1 104）。列：stage,keybuilder,yaml_id,v0,v1,rs,normalizer,n,success_rate。
- **top10** = `exp/weighted_sum/config/top10/libero_10/`（10 yaml + top10_manifest.json）。mean SR 排序：
  ```
   1. 52% spatial_16 0.12/0.87/0    2. 52% spatial_16 0.50/0.50/0   3. 51% spatial_16 0.62/0.37/0
   4. 48% mean_pool 0.18/0.31/0.50  5. 48% mean_pool 0.25/0.37/0.37 6. 47% spatial_16 0.87/0/0.12
   7. 47% spatial_64 0.12/0.50/0.37 8. 46% max_pool 0.37/0.62/0      9. 46% max_pool 0.75/0/0.25
  10. 46% mean_pool 0.12/0.31/0.56
  ```
- journal/results 本地：`exp/weighted_sum/data/phase2/libero_10/{journal.jsonl,results.json}`、`exp/weighted_sum/data/round_1/libero_10/{journal.jsonl,results.json}`。
- **产出 → 下阶段（plan §3.2）**：top10 → Stage 2a base；winner kb d1-best → Stage 4 base；全实验 mean SR 最高 → Stage 3 base。

---

## §3 ★ 走过的坑 + 修复（全量，别重犯）

### 3-A ★★ HOME 修复（libero input() EOFError，最坑，铁律）
- **症状**：timan107 launch worker 时 `File ".../libero/libero/__init__.py", line 104: answer = input(...)` → `EOFError: EOF when reading a line`，worker 全起不来 journal 空。
- **根因**：tether 默认 HOME=`/srv/local/zixuans8/tether-home`（无 `.libero/config.yaml`）；libero 首次 import 在 config 不存在时交互式 `input()`，非交互 stdin 直接 EOF。config 实际在 `/home/zixuans8/.libero/config.yaml`。
- **修复（铁律）**：**所有** run_phase2/smoke launch（含 tmux 内层命令）必须 `export HOME=/home/zixuans8`。base eval 当年也是这么跑通的，旧 handoff §1.1 漏写。

### 3-B placeholder 移到 NFS home（pod 重启 /tmp 被清）
- **症状**：pod 重启后 `/tmp/stage1_placeholder.yaml` 没了 → serve_policy `--cache_config` 找不到。
- **修复**：placeholder 放 **NFS home** `/home/<user>/stage1_placeholder.yaml`（pod 重启持久）。内容用任一 stage2 spatial_16 d1 yaml（合法 cache config，conductor per-eval 热swap，boot 配置无所谓；preload pkl 相对路径在 server cwd 下可解析）。重建：`tether push <local yaml> <node>:/home/<user>/stage1_placeholder.yaml`。

### 3-C ★★ expose 端口会变（broker 重分配）
- **症状**：pod/隧道掉重新 expose 后，broker 给了**新端口**（ziyang10 还回 14000，但 xuanle 给了 **14002** 而不是 14001——旧 ws3b-xl 死隧道还占着 14001）。
- **影响**：① health 脚本 s2 探针硬编码端口 → 探错端口显示 DOWN。② conductor `--servers` 用错端口 → worker 连不上。
- **修复**：re-expose 后看实际分配端口，然后 ① 改 health 脚本：`sed -i "s#/dev/tcp/weiland.top/<旧>#/dev/tcp/weiland.top/<新>#" /tmp/stage2_libero10_health.sh` ② conductor `--servers` 用新端口重启。**当前 xuanle = 14002。**
- xuanle expose 偶报 `agent_rejected:state_write_failed`（agent state.json 写 race），**重试 1-2 次即成**。

### 3-D ★★ RAM 增长 = MuJoCo worker 时间累积（见 §4 详）
- 简版：每 LIBERO sim worker（MuJoCo+robosuite+离屏渲染）随 episode 累积 RAM，pre-MALLOC ~3.4G/worker 一路爬，60 worker 冲 214G 近 OOM。**MALLOC 调优修复（§4）**：ARENA_MAX=2 → ~1.6G/worker plateau。

### 3-E ★★ 全 MISS bug（task_suite 没转发，已修，fix 未提交）
- **症状**：server log 全 `cp1 judge: MISS (top_score=None)`，检索 0 候选退化纯 live 推理。
- **根因**：`agent.py:_default_spawn` 构造 worker 命令没转发 `--task-suite-name`，worker 默认 libero_spatial，task_key 与库的 libero_10 裸 prompt 不匹配 → `_filter_entries`(in_memory_backend line348 精确匹配) 全过滤。
- **修复（3 行，未提交，见 §12）**：agent.py WorkerSpec 加 `task_suite_name` 字段 + _default_spawn base_cmd 加 `--task-suite-name`；run_phase2 WorkerSpec(...) 加 `task_suite_name=args.task_suite`。验证：server log `FULL_HIT`。

### 3-F ★★ 80 worker OOM → 早期估算（已被 MALLOC 改写，见 §4）
- pre-MALLOC：80 worker OOM（280G>220G）；曾用 50(30,20)。MALLOC 后 per-worker 减半，现 96 worker。

### 3-G init_map / calibration（Stage 0，已完成）
- init_map：plan §4.1 的 map 子命令对本数据坏（非 task-major + uv env 无 libero）。改用本地 libero_sim conda env + LIBERO 权威 task 序 + cache-subset↔full np.isclose 匹配，emit `exp/common/data/db/libero_cache/libero_10_init_map.json`（50 records，held_out 45/task，total_inits=50）。
- calibration：排除 CLIP（符号链接临时目录只含 4 cp1 pkl），全 zscore。`exp/weighted_sum/data/phase1/libero_10/calibration_normalizers.json`。
- §4-B 已验证：built-in libero_10 init-states 与 db_init 完全不相交 → 无泄漏，eval 有效（protocol A，同 libero_spatial）。

### 3-H ★ pkill 自杀（踩过 3 次）
- `pkill -f <pattern>` 匹配自己 shell argv。**铁律**：(a) 变量拼接 `P=run_phase; pkill -9 -f "${P}2"`、`W=worker_ent; pkill -9 -f "${W}ry"`、`P=serve; pkill -9 -f "${P}_policy"`；(b) **echo/同 shell 不得出现字面词** run_phase2/worker_entry/serve_policy；(c) **kill 与 launch 拆两条独立 exec**。

### 3-I L5 monitor 脆断 → 自动重连（见 §5）；L2 曾静默死 → L3 cron 升级 monitor-of-monitors。

### 3-J 流程反模式（已铸成，今后改）：跳过 §3.11 smoke 直接大跑 → 全MISS/OOM 拖到大跑才暴露。今后先 smoke 再大跑。

### 3-L d6 看似停滞 = 调度排尾非卡死（2026-05-30 诊断，会自然消解）
- **症状**：Stage 2 中段 d6 进度长时间停在 700/1500（其他 depth 在涨）。
- **真相**：d6 已完成的 7 个是 **mean_pool/max_pool** 的 d6 yaml(各 100ep)；剩 8 个全是 **spatial_16/64** 的 d6。conductor 按 yaml 字典序展开 cell 队列，`spatial_pool_16` 的 d6 排在**巨量 spatial_16 wsweep(2b 312 个 d1-d5)** 之后 → 要等 spatial_16 的 d1-d5 跑完才轮到 → 落 Stage 2 最末尾。**全 37200 cell 都在队列，conductor 不退出就会跑完，不漏**。别误判卡死、别重启。验证：`grep -oE "[A-Za-z0-9_@]+__d6:eval" journal|sort|uniq -c` 看完成的 d6 yaml 是不是只有 mean/max。

### 3-K ★ keepalive ping timeout = 良性自愈噪音（别误判 fatal！2026-05-30 诊断）
- **症状**：health `err` 从 0 跳到上千；run.log 大量 `websockets ConnectionClosedError: 1011 keepalive ping timeout` + Traceback（`_send_ctrl`/`select_bundle` 栈）。
- **根因**：**depth-5 重检索**（candidates 284-399）单次推理+search 偶尔 >20s，超 websockets 默认 ping timeout → worker ws 连接被服务端关。worker 随即被 agent 重启、episode 被 driver requeue、journal-resume 续上 → **零数据损失**。
- **判定良性的证据**（cron/会话见 err 暴涨先查这些，全满足=良性不干预）：① conductor etime 无异常重启 ② keepalive 密度稳定(~3.5%，`tail -2000 run.log|grep -c keepalive` 占比不飙升) ③ worker 满额(160=80×2) ④ journal 持续涨 ⑤ 两 server FULL_HIT 无 OOM/crash ⑥ 隧道通。
- **决定**：**不重启 server(没崩，重启无益)、不降 worker、不改代码**——系统正确自愈。吞吐因 d5 重检索 ~31/min(低于 d1 标称 40，正常)。
- **可选优化(暂不做)**：调大 `packages/openpi-client/.../websocket_client_policy.py` 的 ws ping_timeout(如 60s)+重启 conductor(journal 不丢) → 消除断连、吞吐回 ~40。有改动风险，owner 发话才做。
- **health err 不触发 L2 ALERT**（L2 只在 health 输出含 "ALERT"/"EVAL DONE" 时 emit；health 仅 server 双 DOWN 或 cond=0 才打 ALERT）→ err 数字大不会误报警，但 cron 人工判断时按上面 6 证据归类良性。

---

## §4 ★ worker / RAM 调优史 + MALLOC（重点章节）

### 4.1 调优历程（owner 全程参与）
- 起始 50 worker(30,20)：27.6 ep/min。
- 加 worker：50→66(33,33)：37.8/min（+37%，6 replica 喂满）。但 pre-MALLOC RAM 冲 214G/3G avail 近 OOM。
- 反复缩/加：66→56→60→… 每次 RAM 撞 10G 看护器就缩。**关键认知：pre-MALLOC RAM 不 plateau，随时间一路爬撞顶**。
- **MALLOC 调优（根治）**：见 4.2。
- 96 worker(48,48) 跑了一段（owner 指令）→ **2026-05-30 17:57 缩到 80 worker(40,40)**：深检索期 RAM 累积冲 197G/avail 21G 危险，滚动重启泄压(197G→14G)+缩 80→稳态 106G/avail 112G。**当前 80 worker(40,40)**。RAM 累积型，末期可能再缩 64。

### 4.2 ★★ MALLOC 调优（根因 + 修复 + 验证）
- **根因**：MuJoCo/robosuite worker 的 env 是每 task 创建、跑完该 task 所有 trial 后 `close()`（main.py:786 建 / 923 close，episode 间复用）。close 后渲染上下文 + sim data 没完全归还 OS，且 **glibc 多线程 malloc arena 囤积**（MuJoCo 渲染多线程狂开 arena），freed 内存留在 arena 不还 OS → 看着像内存一路涨。
- **修复（零正确性风险，纯内存管理）**：
  - `MALLOC_ARENA_MAX=2`：限制 glibc 每进程 arena 数（默认 8×核），治本。
  - `MALLOC_TRIM_THRESHOLD_=134217728`(128M)：freed 块超阈值归还 OS。
  - 经 `agent.py:_default_spawn` 的 `env`（os.environ 透传，只 strip VIRTUAL_ENV/PYTHONPATH/PYTHONHOME）传到 worker。**已写进 agent.py 成默认（setdefault，caller 可覆盖）+ launch export 双保险**。
- **验证（60 worker 16min 测试）**：t0=17G → t4=86G → t8=98G → t12=94G（**98→94 降了，TRIM 在归还**）→ plateau ~95G。对比 pre-MALLOC 同期 174G→冲 214G。**per-worker RAM 3.4G → ~1.6G，减半多**。吞吐 ~33/min 不变（没变慢）。
- **96 worker 预期**：96×1.6 ≈ 154G / ~66G avail，安全。warmup 期盯峰值确认。
- **owner 指令**：上 96 worker + **时刻监视内存**（RAM 看护器 v2 守 10G，破线立马缩）。

---

## §5 ★ 监控架构（5 层 + cron，全量 id + 重挂）

| 层 | 数据源 | id | 职责 | 验活 | 重挂 |
|---|---|---|---|---|---|
| **L1** health | journal+TCP探针+pgrep | `/tmp/stage2_libero10_health.sh`(timan107，**s2 探针=14002**) + symlink `/tmp/current_run_health.sh`→它 | 一行总览 | `tether exec timan107 -- bash -lc 'bash /tmp/stage2_libero10_health.sh'` | 见下脚本 |
| **L2** 条件 Monitor | poll L1 | task `bf3l56qkb`(high=50,TOTAL=37200) | milestone75/100 / ALERT / STALL / **STAGE2 DONE→exit0** | `pgrep -fc 'M25=\|STALL_THRESHOLD'`≥1 | 见 §13 |
| **L3** cron | L1+pgrep | `b4cf0ba2`(每10min 2,12,..52) | **通用 monitor-of-monitors**，查 current_run_health + L2/L5/L4 存活 | `CronList` | CronCreate |
| **L4** agentchat watcher | daemon push | (§0 起的，跨 compact 存活) | owner 房间消息 | `pgrep -fc 'agentchat watch state'`≥1 | §10 模板 |
| **L5** server-log Monitor | tail -F 两 server log | `b4dfwbxii`(自动重连) | 真 fatal 秒推 | `pgrep -fc 'tail -F /tmp/srv0.log'`=2 | §13 |
| **RAM v2** Monitor | poll free RAM | `b0v974yde`(15s) | **avail<10G WARNING→滚动重启/缩；<5G或run.log Killed CRITICAL** | `pgrep -fc 'RAM WARNING\|10G thresh'`≥1 | 见下 |

**L1 health script 全文（`/tmp/stage2_libero10_health.sh`，s2=14002）：**
```bash
#!/bin/bash
JOURNAL=/tmp/stage2_libero10/journal.jsonl
LOG=/tmp/stage2_libero10/run.log
TOTAL=37200
TS=$(date "+%H:%M:%S")
if [ -f "$JOURNAL" ]; then done=$(grep -cE "\"status\"" "$JOURNAL"); ok=$(grep -cE "\"success\": ?true" "$JOURNAL"); else done=0; ok=0; fi
cond=$(pgrep -f "[r]un_phase2" 2>/dev/null | wc -l)
s1=$(timeout 3 bash -c "echo > /dev/tcp/weiland.top/14000" 2>/dev/null && echo UP || echo DOWN)
s2=$(timeout 3 bash -c "echo > /dev/tcp/weiland.top/14002" 2>/dev/null && echo UP || echo DOWN)
err=$(grep -ciE "traceback|connection refused|out of memory|CUDA error|FATAL|Killed" "$LOG" 2>/dev/null); [ -z "$err" ] && err=0
pct=$(awk "BEGIN{printf \"%.1f\", $done*100.0/$TOTAL}")
echo "[$TS] progress=$done/$TOTAL (${pct}%) success=$ok cond=$cond s1=$s1 s2=$s2 err=$err"
[ "$done" -ge "$TOTAL" ] && echo "EVAL DONE"
[ "$s1" = "DOWN" ] && [ "$s2" = "DOWN" ] && echo "ALERT all servers DOWN"
[ "$cond" -eq 0 ] && [ "$done" -gt 0 ] && [ "$done" -lt "$TOTAL" ] && echo "ALERT conductor dead at $done"
```

**L2 Monitor 全文（persistent，timeout 3600000）：**
```bash
prev=-1; stall=0; high=50; INTERVAL=180; STALL_THRESHOLD=8; TOTAL=37200
M25=$((TOTAL/4)); M50=$((TOTAL/2)); M75=$((TOTAL*3/4))
while true; do
  line=$(tether exec timan107 -- bash -lc 'bash /tmp/stage2_libero10_health.sh' 2>/dev/null | head -4)
  [ -z "$line" ] && { sleep "$INTERVAL"; continue; }
  done_n=$(echo "$line" | grep -oE 'progress=[0-9]+' | head -1 | cut -d= -f2); case "$done_n" in ''|*[!0-9]*) done_n=0;; esac
  mile=0; if [ "$done_n" -ge "$TOTAL" ]; then mile=100; elif [ "$done_n" -ge "$M75" ]; then mile=75; elif [ "$done_n" -ge "$M50" ]; then mile=50; elif [ "$done_n" -ge "$M25" ]; then mile=25; fi
  emit=""; echo "$line" | grep -qE "EVAL DONE|ALERT" && emit="$line"
  if [ "$mile" -gt "$high" ]; then emit="[s2 ~${mile}%] $(echo "$line"|head -1)"; high=$mile; fi
  if [ "$done_n" = "$prev" ] && [ "$done_n" -gt 0 ] && [ "$done_n" -lt "$TOTAL" ]; then stall=$((stall+1)); else stall=0; fi
  if [ "$stall" -ge "$STALL_THRESHOLD" ]; then emit="STALL frozen done=$done_n | $(echo "$line"|head -1)"; stall=0; fi
  [ -n "$emit" ] && echo "$emit"
  if echo "$line" | grep -qE "EVAL DONE"; then echo "=== STAGE2 DONE -> summarize + Stage 3 ==="; exit 0; fi
  prev=$done_n; sleep "$INTERVAL"
done
```

**RAM 看护器 v2 全文（persistent，10G 阈值，owner 设定）：**
```bash
while true; do
  line=$(tether exec timan107 -- bash -lc "free -g | awk '/Mem:/{print \$3,\$4,\$7}'" 2>/dev/null)
  used=$(echo "$line"|awk '{print $1}'); free_g=$(echo "$line"|awk '{print $2}'); avail=$(echo "$line"|awk '{print $3}')
  case "$avail" in ''|*[!0-9]*) sleep 15; continue;; esac
  killed=$(tether exec timan107 -- bash -lc 'grep -c -iE "Killed process|out of memory|oom-kill" /tmp/stage2_libero10/run.log 2>/dev/null' 2>/dev/null | tr -d '[:space:]')
  if [ "${killed:-0}" -gt 0 ]; then echo "RAM CRITICAL: OOM/Killed (count=$killed) avail=${avail}G used=${used}G — SHRINK NOW"; fi
  if [ "$avail" -lt 5 ]; then echo "RAM CRITICAL: avail=${avail}G free=${free_g}G used=${used}G <5G — SHRINK NOW"
  elif [ "$avail" -lt 10 ]; then echo "RAM WARNING(10G thresh): avail=${avail}G free=${free_g}G used=${used}G"; fi
  sleep 15
done
```
- **RAM 看护器报 WARNING/CRITICAL 的处理**：96 worker + MALLOC 预期稳态 ~154G/66G avail 不会触发；若意外破 10G → **先滚动重启 conductor（kill+relaunch 重置 RAM，journal-resume，保 worker 数）**；若 MALLOC 失效或仍破 → 缩 worker（48,48→40,40→…）。
- **L3 cron 通用版**（指向 `/tmp/current_run_health.sh` symlink，跨轮/跨 Stage 复用）：全健康只主会话记一行不发房间；monitor 死立即重挂+报房间；server 错误崩溃自主重启(辨真死→恢复 SOP §6)；DONE 由 L2 触发不自删。

---

## §6 ★ 集群不稳 + 故障恢复 SOP（已用 3 次，2026-05-30 下午）

**现象**：jupyter-ziyang10 / jupyter-xuanlel2 的 tether agent **~1 小时掉一次**（pod 容器重启 or pod↔broker 网络中断）。timan107 + broker(pc732) 一直正常。
- node STALE→OFFLINE：agent 心跳断，**exec 不进节点**（node_offline）。隧道随之死（nc 端口 refused）。
- 也可能节点 ONLINE 但隧道单独断（nc refused 但 node ls 显 ONLINE）= 隧道掉，re-expose 即可。

**恢复 SOP（节点回 ONLINE 后自动执行）：**
1. `tether node ls -a | grep -E "ziyang10|xuanlel2"` 看 ONLINE/OFFLINE；`nc -zv weiland.top 14000/14002` 看隧道。
2. 若节点 OFFLINE → 够不到，**urgent 喊 owner 手动重启 pod/agent**（agent 离线我无法自主重启）。等节点回 ONLINE。
3. 节点 ONLINE 但 serve 没了（pod 重启）→ 验 `git rev-parse`(应 9519a79)/ckpt/4pkl 都在(NFS home 持久)、清残留 serve（变量法）、placeholder 在不在(NFS home，不在则 push)。
4. 起 serve_policy（§1.2/§1.3，**ziyang10 3rep / xuanle 3rep**，spawn-batch 2），后台 ready-waiter 等 `replica_proxy listening`。
5. re-expose（新 name；**broker 可能给新端口** → 记下实际端口，改 health s2 探针 + conductor --servers，见 §3-C）。
6. kill 旧 conductor + 清 run.log（保 journal！）+ 重启 conductor（§1.1，新端口，journal 从断点 resume，**零数据损失**）。
7. 重挂 L2（§13）；确认 progress 上涨。
8. 房间报"已恢复"。
- **journal-resume 是核心**：conductor 死/重启不丢 episode（已 done 的跳过，in-flight retry）。**实测多次 0 数据损失。**
- 两节点恢复探测器模板（监测任一回 ONLINE）：见 §13。

---

## §7 ★ Stage 2 跑完后流程（L2 报 STAGE2 DONE 触发）

### 7.1 拉回 + summarize + 拆 2a/2b 分析
```bash
# 1. pull journal
mkdir -p exp/weighted_sum/data/stage2/libero_10
tether pull timan107:/tmp/stage2_libero10/journal.jsonl exp/weighted_sum/data/stage2/libero_10/journal.jsonl
# 2. summarize（per-yaml SR）
uv run exp/weighted_sum/summarize.py --journal exp/weighted_sum/data/stage2/libero_10/journal.jsonl --out exp/weighted_sum/data/stage2/libero_10/results.json
# 3. 按 yaml_id 来源拆开分析：
#    2a(trajectory) = 15 base × depth{3,4,5,6}=60 yaml，id 形如 <base_id>__d{depth}
#    2b(weight sweep) = spatial_16 78权重 × depth{1,3,4,5}=312 yaml
#    dedup_manifest 在 config/stage2/libero_10/dedup_manifest.json（0 dup 记录）
```
- emit 命令（重做用，**module 模式 + libero_10 override**）：
```bash
# 2a:
uv run python -m exp.weighted_sum.emit_trajectory_yamls --calibration exp/weighted_sum/data/phase1/libero_10/calibration_normalizers.json --artifact-dir exp/common/data/cache_artifacts/libero_10 --results-csv exp/weighted_sum/data/phase2/libero_10/all_results.csv --top10-dir exp/weighted_sum/config/top10/libero_10 --output-dir exp/weighted_sum/config/trajectory/libero_10 --depths 3,4,5,6
# 2b:
uv run python -m exp.weighted_sum.emit_trajectory_weight_sweep --calibration exp/weighted_sum/data/phase1/libero_10/calibration_normalizers.json --artifact-dir exp/common/data/cache_artifacts/libero_10 --output-dir exp/weighted_sum/config/trajectory_wsweep/libero_10 --depths 1,3,4,5
# 合并去重到 config/stage2/libero_10（内容签名：keybuilder+非零权重+trajectory_depth）
```

### 7.2 Stage 3 / Stage 4（plan §3.1 + handoff §14 旧版）
- **Stage 3 threshold-pareto**：⚠ **owner 2026-05-30 改设计 = per-depth base**（非原 plan 单 base）。base = **spatial_16 在 depth{1,3,4,5} 各自的 SR 最优 yaml = 4 个 base**（owner 进无人值守前下放细节，agent 采纳建议默认：**(a) 固定 keybuilder=spatial_16**（与 Stage 4 一致，不跨 kb）；**(b) d1 候选池含 Stage 1 全部 spatial_16 d1**——winner spatial_16 d1-best 52% 在内）。d6 不做（owner 只要 1/3/4/5）。
  - 每 base 独立走 `emit_threshold_yamls`(warmup→`solve_thresholds`→eval grid fh+ws≤0.9，**11×5 网格−6 退化=49 cell** 满格，owner "不用调网格")+anchor；`run_phase2 --per-step-out`；`summarize_inf_ratio`。
  - **预算 = 4 base × ~49 cell ≈ 196 cell ≈ 20,800 ep**（vs 原单 base 5,200）。双 server 可承受。
  - **安全阀（无人值守）**：Stage 2 DONE→summarize 后，按规则选出 4 个 base（depth 分组 argmax SR），**先房间通报"选出的 4 base=[...]"给 owner 异步否决窗口，然后继续跑不等**（不阻塞）。若数据异常（如固定 spatial_16 某 depth SR 远低于跨-kb best）→ 房间标注+仍按规则跑，owner 可否决。
- **Stage 4 kinematic 复刻**：base=winner-kb(spatial_16) d1-best；`kinematic/runner.py` 8 mode；237 cell ~24,000 ep；**§8.6 全套路径 override 到 kinematic_phase5/libero_10/**。
  - ★ **Stage 4 必须 owner 回来**：要填 CFG_SPECS 真值(winner 权重+μσ)进 `v2_spec.py` = 代码改动 → **独立 Review session 过 G2**（我 Execution 不能自审，WA 高于用户指令，即使 owner 授权也不能跳 G2）。我最多做到前置(emit-warmup/run-warmup/verify-raw 不涉代码)+代码改好待审。
  - Stage 4 决策点（owner 定）：单 vs 双 server fail-fast；winner≠spatial_16 怎么办(本例 winner=spatial_16 故无碍)；protocol A/B。
- 全程双 server，每 config 100 ep。

---

## §8 数据资产 + 路径

| 资产 | 路径 | 状态 |
|---|---|---|
| 6 CP1 库 pkl | `exp/common/data/cache_artifacts/libero_10/`（本地 + ziyang10 + xuanle 各一份）| mean/max/sp16/sp64 + 2 CLIP(不用) |
| init_map | `exp/common/data/db/libero_cache/libero_10_init_map.json` | 50 records，held_out 45/task |
| calibration | `exp/weighted_sum/data/phase1/libero_10/calibration_normalizers.json` | 4 cp1 全 zscore |
| Stage1 base/r1 yaml | `exp/weighted_sum/config/{phase2,round_1}/libero_10/` | base 136 + r1 104 |
| Stage1 all_results | `exp/weighted_sum/data/phase2/libero_10/all_results.csv` | 240 config |
| top10 | `exp/weighted_sum/config/top10/libero_10/` | 10 yaml + manifest |
| Stage2 yaml | `exp/weighted_sum/config/stage2/libero_10/`(372) + timan107:`/tmp/stage2_libero10_yamls` | 2a60+2b312，0dup |
| Stage2 journal | timan107:`/tmp/stage2_libero10/journal.jsonl` | TOTAL 37200，跑中 |
| placeholder | `/home/ziyang10/stage1_placeholder.yaml`、`/home/xuanlel2/stage1_placeholder.yaml` | NFS home 持久 |
- **gitignore**：`exp/**/data/**` + `exp/weighted_sum/config/**` ignored（yaml/journal/pkl 不入库）；只 code + analysis + plan log + README 入库。

---

## §9 设备拓扑 + RAM 铁律

| 角色 | 节点 | 关键 |
|---|---|---|
| server#1 | jupyter-ziyang10 | H200 NVL/cc9.0；HOME=/home/ziyang10；repo=/home/ziyang10/openpi；uv=/home/ziyang10/.local/bin/uv；ckpt=~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch；tmux=srv0；3 replica；NAT→**weiland.top:14000** |
| server#2 | jupyter-xuanlel2 | 另块 H200 NVL(同构)；HOME=/home/xuanlel2；**无系统tmux/fuser**(用 /home/xuanlel2/miniforge3/bin/tmux + `pkill -9 -f "[s]erve_policy"`+`ss`)；**NFS uid 坑**(dump root /tmp)；cgroup 32G(3rep~26G紧但fit)；3 replica；NAT→**weiland.top:14002** |
| client | timan107 | Intel Xeon E5-2650 v4@2.2GHz/48 逻辑核/**220G DDR4**/8×GTX1080(8G/卡 EGL渲染)；repo=/scratch/zixuans8/openpi；uv=/shared/nas/data/m1/zixuans8/miniconda3/bin/uv；conda_env=/scratch/zixuans8/libero_sim(py3.8)；**HOME 必 export /home/zixuans8**；**allow_roots 不含 /scratch**(传文件走 /tmp 中转) |
| broker | pc732(weiland.top→155.98.36.32) | tether broker；公网端口段 14000-14999 |
| a100 | **OFFLINE** | 公网 149.165.152.105；failover 需 owner 恢复 + 数据双 sync(本例没用) |

**★ RAM 铁律**：MALLOC 后 per-worker ~1.6G（pre-MALLOC 3.4G）。96 worker ≈ 154G/66G avail 安全。**RAM 是增长型**(MuJoCo 累积)，靠 MALLOC 压住 + 看护器守 10G + 必要时滚动重启/缩 worker。

---

## §10 agentchat（唯一 owner 异步通道）

- 账号 `agent1`(role=user, online)；token 在 `~/.agentchat/cli.toml`。**token 是密钥：勿 echo/log/commit/贴房间。**
- **房间 libero10 = `019e749f-c5f4-7ce0-9666-4b9a5d8e9af3`**（已 subscribed；唯一沟通房间）。
- 读：`agentchat read 019e749f-c5f4-7ce0-9666-4b9a5d8e9af3 --json`。发（必 heredoc 单引号定界）：
```bash
agentchat send 019e749f-c5f4-7ce0-9666-4b9a5d8e9af3 --file - <<'EOF'
消息体
EOF
```
- urgent：加 `--priority urgent`。watcher 重挂（先 pgrep 验存活）：`agentchat watch state --json | jq --unbuffered -c '([.rooms[]?|select(.name=="libero10")|.unread]|add//0) as $l10|select($l10>0 or (.totals.mentions//0)>0)'` 放 Monitor persistent。
- **通知规约（owner 2026-05-30 强化：聊天室只发重要事务，别刷屏）**：聊天室**只发** ①需 owner 决策 ②ALERT/危险(OOM/server 崩/数据风险) ③Stage milestone(75%)/DONE ④server 起关。**不主动发**：routine 进度/吞吐补报/诊断细节/处置过程细节（这些只主会话记，owner 问才答）。owner 房间消息**仍必回**（但简短）。〔教训：曾发"80w 吞吐 30.8/min"被 owner 指为不重要〕

---

## §11 tether 避坑（全踩过）
1. **HOME 陷阱**：默认 HOME=`~/.tether-agent`(timan107=/srv/local/zixuans8/tether-home)；ziyang10/xuanle 程序找 ~/.cache 必 `export HOME=/home/<user>`；timan107 worker 找 .libero 必 `export HOME=/home/zixuans8`(§3-A)。
2. **allow_roots**：ziyang10=/home/ziyang10,/tmp；xuanle=/home/xuanlel2,/tmp；timan107=/tmp,/home,/users,/srv(**/scratch 不在**→先 push /tmp 再 cp 进 repo；push 目标必带 `<node>:` 前缀)。
3. **跨 remote 拷文件**：push 首参须 local → 先 pull A→local 再 push local→B。
4. **pkill 自杀**：§3-H 变量法 + echo 不含字面 + 拆 exec。
5. **expose 隧道~10h 闲掉线 / pod 重启隧道死**：换新 name 重 expose；**broker 可能给新端口**(§3-C)→改 health 探针 + conductor --servers。
6. **eval 期间不重启 tether agent**(worker_entry 不 auto-reconnect)。
7. **跨 jupyter pod 不能直连**(k8s NetworkPolicy)→必经 broker。
8. **xuanle 无系统 tmux/fuser** → 全路径 `/home/xuanlel2/miniforge3/bin/tmux` + `pkill -9 -f "[s]erve_policy"` + `ss`。
9. **xuanle expose state_write_failed** 偶发 → 重试 1-2 次即成。
10. **server tyro 顺序**：`--replicas`/`--replica-spawn-batch`/`--port`/`--cache_config` 等顶层 flag **必在 `policy:checkpoint` 之前**。
11. **裸 TCP 探针副作用**：health 的 /dev/tcp 探针让 server 记良性 InvalidMessage Traceback——L5 已改只匹配真 fatal。

---

## §12 未提交改动（diff，等 owner commit；timan107 re-sync 会丢，丢了照此重打）

**文件 1：`src/openpi/conductor/agent.py`** —— 两处改动：
1. `WorkerSpec` dataclass 加字段（task_suite fix）：
```python
    conda_env: str = ""
    # LIBERO benchmark suite ... callers (run_phase2) forward their --task-suite here.
    task_suite_name: str = "libero_spatial"
```
2. `_default_spawn` base_cmd 末尾加（task_suite fix）：
```python
        "--driver-port", str(driver_port),
        "--task-suite-name", spec.task_suite_name,
    ]
```
3. `_default_spawn` 在 `env["CUDA_VISIBLE_DEVICES"] = spec.gpu_id` 之后加（MALLOC 默认）：
```python
    env["CUDA_VISIBLE_DEVICES"] = spec.gpu_id
    # glibc malloc tuning ... setdefault so an explicit caller export still wins.
    env.setdefault("MALLOC_ARENA_MAX", "2")
    env.setdefault("MALLOC_TRIM_THRESHOLD_", "134217728")
```

**文件 2：`exp/weighted_sum/run_phase2.py`** —— WorkerSpec(...) 加 kwarg（task_suite fix）：
```python
        WorkerSpec(worker_id=f"w{i}", server_key=worker_server_keys[i], gpu_id=str(i % args.gpus),
                   conda_env=args.conda_env, task_suite_name=args.task_suite)
```
- 推 timan107（allow_roots 无 /scratch → /tmp 中转）：`tether push src/openpi/conductor/agent.py timan107:/tmp/agent_x.py && tether push exp/weighted_sum/run_phase2.py timan107:/tmp/rp2_x.py && tether exec timan107 -- bash -lc 'cp /tmp/agent_x.py /scratch/zixuans8/openpi/src/openpi/conductor/agent.py; cp /tmp/rp2_x.py /scratch/zixuans8/openpi/exp/weighted_sum/run_phase2.py'`
- **agent.py + run_phase2 已在 timan107 应用**（含 MALLOC）。server 端不需要这些 fix。
- 其它已 commit 改动(@9519a79+)：emit_trajectory_yamls/emit_trajectory_weight_sweep/refine_round/emit_top10/build_all_results/summarize/kinematic/* + v2_spec CFG_SPECS 占位 + tests。

---

## §13 命令速查（copy-paste）

**全栈一键验活：**
```bash
echo "=== nodes ==="; tether node ls -a | grep -E "ziyang10|xuanlel2|timan107"
echo "=== eval health ==="; tether exec timan107 -- bash -lc 'bash /tmp/stage2_libero10_health.sh'
echo "=== monitors ==="; echo "L2:$(pgrep -fc 'M25=|STALL_THRESHOLD') L5:$(pgrep -fc 'tail -F /tmp/srv0.log') L4:$(pgrep -fc 'agentchat watch state') RAMv2:$(pgrep -fc 'RAM WARNING|10G thresh')"
echo "=== RAM ==="; tether exec timan107 -- bash -lc 'free -g | awk "/Mem:/{print \"used=\"\$3\"G avail=\"\$7\"G\"}"'
echo "=== links ==="; nc -zv weiland.top 14000 2>&1|tail -1; nc -zv weiland.top 14002 2>&1|tail -1
echo "=== room ==="; agentchat read 019e749f-c5f4-7ce0-9666-4b9a5d8e9af3 --json | head -20
```
**server ready watcher（重启 server 后等就绪，background bash run_in_background）：**
```bash
for i in $(seq 1 80); do tether exec jupyter-ziyang10 -- bash -lc 'grep -qE "replica_proxy listening on" /tmp/srv0.log' 2>/dev/null && { echo READY; break; }; sleep 15; done
```
**节点恢复探测器（任一回 ONLINE，Monitor persistent）：**
```bash
while true; do st=$(tether node ls -a 2>/dev/null | grep -E "ziyang10|xuanlel2"); zy=$(echo "$st"|grep ziyang10|awk '{print $2}'); xl=$(echo "$st"|grep xuanlel2|awk '{print $2}'); if [ "$zy" = "ONLINE" ] || [ "$xl" = "ONLINE" ]; then echo "NODE RECOVERY zy=$zy xl=$xl"; exit 0; fi; sleep 90; done
```
**smoke test（refine/Stage 前必跑，1 yaml 验链路，HOME 必带）：**
```bash
tether exec timan107 -- bash -lc 'export HOME=/home/zixuans8; rm -rf /tmp/smoke_y && mkdir -p /tmp/smoke_y && cp $(ls /tmp/stage2_libero10_yamls/cp1_spatial_pool_16__*__d3.yaml|head -1) /tmp/smoke_y/; cd /scratch/zixuans8/openpi && export PYTHONPATH=/scratch/zixuans8/openpi && timeout 280 /shared/nas/data/m1/zixuans8/miniconda3/bin/uv run exp/weighted_sum/run_phase2.py --yaml-dir /tmp/smoke_y --init-map /tmp/libero_10_init_map.json --journal /tmp/smoke/journal.jsonl --servers weiland.top:14000 --task-ids 0 --eval-trials 2 --task-suite libero_10 --workers 4 --gpus 4 --conda-env /scratch/zixuans8/libero_sim --eval-concurrency 1 2>&1 | tail -6'
# 期望 "all stages done" + journal done=2；任何 EOFError/MISS(top_score=None)/Traceback → 排错
```
**滚动重启（RAM 看护器报 10G 时，保 worker 数，重置 RAM）：** = §1.1 清残留 + 重启同命令（journal-resume）。

---

## §14 红线 + 恢复 checklist（零记忆）

**恢复 checklist：**
1. 读本文件全文 + plan §14。
2. `git -C /home/weiland/projects/openpi log -1 --oneline`(应 9519a79) + `git status`(**会看到 agent.py/run_phase2.py modified = §12 fix，别 revert！** + session_handoff.md)。
3. `tether node ls -a`(ziyang10/xuanlel2/timan107 ONLINE；a100 OFFLINE)。集群不稳，可能某节点 OFFLINE → §6 恢复。
4. **逐项 pgrep 验 §5 监控存活**（L2 bf3l56qkb / L5 b4dfwbxii / L4 / RAM v2 b0v974yde；CronList 看 b4cf0ba2）；死的重挂 + 报房间。
5. §13 全栈验活；`agentchat read` 看 owner 有无新指令（有必回）。
6. eval 没跑完 → **别重跑**（journal resume），等 L2 报 STAGE2 DONE；跑完 → §7。
7. RAM 盯着（96 worker，看护器守 10G）；RAM 看护器报警 → 滚动重启或缩 worker。

**红线（最重要）：**
1. **不擅自 git add/commit/push/stash/reset**——§12 的 agent.py(task_suite+MALLOC)/run_phase2 fix 等 owner 明确发话才 commit（英文/无 Co-Authored-By/author=LinZiyang666/单大 commit）。
2. **owner 房间任何消息必回**；无人值守 mandate 生效，不弹终端阻塞窗口。
3. **起/关 server 必通知房间**；server 错误崩溃可自主重启(owner 授权，§6 SOP)；隧道瞬断不重启(re-expose)。
4. **monitor 存活必 pgrep 实证、不假设**（L3 cron 自动兜底）。
5. **timan107 worker 用 MALLOC + 当前 80(40,40)**（96 长跑 RAM 累积冲 197G 危险已缩 80，稳态 106G/avail 112G）；RAM 累积型，深检索末期再逼近 → 滚动重启泄压(kill 全归还 OS)或再缩 64(32,32)；看护器破 10G 立马处理。
6. **pkill 必变量法 + echo 不含字面 + kill/launch 拆 exec**。
7. **eval 在跑别重启**（journal resume）；要重启走清残留→同命令(HOME+MALLOC+14002)。
8. **Stage 4 CFG_SPECS 填真值需 owner 开独立 Review session 过 G2**（我不能自审，WA 高于用户授权）。
9. **决策可比性**：双 server 仅因两台同构 H200；只声称 libero_10 内部可比。
10. **HOME=/home/zixuans8 / placeholder NFS home / xuanle=14002 / server 3rep+3rep** —— 这 4 个是本 session 的新铁律，旧正文/旧 handoff 不含。

---

## §15 实测数据值（验证/复现用）

- **Stage 1 winner**：spatial_16 52%（v0/v1/rs = 0.50/0.50/0 与 0.12/0.87/0 并列）。per-kb max 52/48/47/46。
- **MALLOC 效果**：60 worker，pre-MALLOC 15→174G@10min→214G撞顶(3.4G/worker)；MALLOC 后 17→86→98→94G plateau(~1.6G/worker)；吞吐 ~33/min 不变。
- **吞吐**：50w 27.6/min、66w 37.8/min、60w ~33/min（瓶颈本是闭环推理延迟 ~1767ms/step，replica 数是大杠杆，6 replica 已满）。
- **RAM 上限**：220G；MALLOC 后 96w≈154G/66G avail 安全。
- **init_map used-inits（每 task 5 个，eval held-out 排除）**：
  ```
  t0=[19,22,23,42,47] t1=[0,6,11,25,38] t2=[23,38,42,46,48] t3=[0,1,5,7,41] t4=[12,18,22,28,32]
  t5=[0,3,6,13,48] t6=[3,10,13,15,47] t7=[3,20,33,48,49] t8=[6,25,29,32,47] t9=[10,13,14,44,47]
  ```
- **LIBERO libero_10 task 序（task_key=language）**：0 tomato_sauce / 1 cream_cheese+butter / 2 turn_on_stove_moka / 3 black_bowl_drawer / 4 white_mug / 5 book_caddy / 6 white_mug+chocolate / 7 alphabet_soup+cream_cheese / 8 both_moka_stove / 9 yellow_white_mug_microwave。
- **当前进度快照**：~61.5%(22893/37200) @16:55；96 worker warmup 中 RAM 125G→~154G；s1/s2 UP。**恢复时以 §13 实时 health 为准。**

---

---

## §16 当前未决事项 / 待办（compact 后接着做）

1. **Stage 2 跑完**（L2 报 STAGE2 DONE）→ §7.1 pull journal→summarize→拆 2a/2b 分析→**Stage 3（⚠ per-depth：见 §7.2 改后设计，4 base=spatial_16 d{1,3,4,5}-best，196 cell，先房间通报 4 base 再跑）**。这是主线，无需 owner（仅入口通报，不阻塞）。
2. **96 worker RAM 持续盯**：MALLOC 后 96w 稳态 ~130-154G（实测 16:55=130G/88G avail，可能再爬向 ~154G）。owner 要"时刻监视"。RAM 看护器 b0v974yde 守 10G；若意外破线 → 滚动重启(保 worker)优先，再不行缩。
3. **commit（等 owner 发话）**：§12 的 agent.py(task_suite + MALLOC) + run_phase2(task_suite)。owner 偏好一次结构化大 commit、英文、无 Co-Authored-By、author=LinZiyang666。**绝不擅自 commit。**
4. **Stage 4 需 owner 回来开独立 Review session 过 G2**（CFG_SPECS 填 winner 真值=代码改动，我 Execution 不能自审）。我可先做 Stage4 前置(emit-warmup/run-warmup/verify-raw)。
5. **集群随时可能再掉**（已 3 次，~1h 一次）→ §6 SOP；节点 OFFLINE 够不到则 urgent 喊 owner。
6. **owner 最近关注点**：worker 规模/RAM 优化（MALLOC 已解决）、吞吐（96w=40.2/min 最快）。owner 高频参与，房间消息必回。
7. **可选优化（owner 没要，备忘）**：若想再快，bottleneck 是闭环推理延迟非 worker；replica 已 3+3=6 满。再加 worker 边际递减（96 已接近）。

---

> 本文件 ≥500 行全量恢复手册。compact 后：读全文 → §14 checklist → §13 验活 → 集群掉了走 §6 → 等 L2 STAGE2 DONE → §7 → Stage 3 → Stage 4(owner G2)。**HOME=/home/zixuans8 / MALLOC_ARENA_MAX=2+TRIM=128M / xuanle=14002 / server 3rep+3rep / 当前 80 worker(40,40)(96 因 RAM 累积冲 197G 危险已缩 80，稳态 106G/avail 112G) 是本 session 新铁律，务必用。** 实测：96w=40.2 ep/min 但长跑 RAM 累积到 197G 不可持续；80w 稳态 106G 安全。MALLOC 压 glibc 囤积但压不住深检索工作集累积，靠滚动重启泄压(kill 全归还 OS)。
