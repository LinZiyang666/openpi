# Session Handoff — GR00T × LIBERO cache 线（v12 · X16 帕累托主跑中 + X17 计划两次被自审推翻）

> ⚠ **不是 `logs/session_handoff.md`**（那份属 X14/X15 RL Router，勿动）。
> 口径唯一来源：[`logs/libero_groot_gate_pareto_plan.log.md`](libero_groot_gate_pareto_plan.log.md)（X16 门-阈值帕累托）、
> [`logs/libero_groot_ws_search.log.md`](libero_groot_ws_search.log.md)（权重搜索）、
> [`logs/libero_groot_collection.log.md`](libero_groot_collection.log.md)（采集与建库）、
> [`logs/groot_serving_perf_parity_plan.log.md`](groot_serving_perf_parity_plan.log.md)（X17，**待 v3 重写**）。

**Status**（2026-08-25 09:0x CDT）：
- **X16 全链收官**：`GATE-PARETO-DONE` @ 08:51:23，两 suite × 17 臂 × 500 集 = 17,000 集，
  0 失败、partial 0、全部过完整性门 I1–I6。报告见
  [`exp/libero_groot/analysis/gate_pareto/analysis.md`](../exp/libero_groot/analysis/gate_pareto/analysis.md)。
  **最重要的发现：判据值不值钱取决于「执行体 × suite」，四个格子里只有 GR00T×spatial 的 gate-only 是免费的**
  （p=1.000）；l10 上要付 7.8 pp（p=0.001）。
- **X17 v5 §4 Code 完成，现在 G2**（§4 本地测试 advisory：`5 failed = HEAD 基线 / 4041 passed`，新增 52 条测试）。
  ⚠ 此前记成「§6 Verify 过」是错的：章程 §5 禁止在 G2 APPROVED 前进 §6。
- **未提交**：本轮全部改动仍在工作树，路径清单见 §5（工作树四条线共用，须按显式路径提交）。
- **剩余**：X17 的 G-perf 与端到端冒烟（现在整池已空出来，可以做了）。

  ⇒ 三批约 8 h，之后 gate-only + aggregate + plot → `GATE-PARETO-DONE`。spatial 已收官（§2）。
- **X17 v5 的 §4 Code 完成、§6 Verify 过（5 failed = HEAD 基线，零新增；新增 45 条测试），待 G2。**
  v1–v4 连败四轮、共六轮自审；v5 改用本次运行自己的链日志做量化（§4）。
- **未实施**：G-perf 与端到端冒烟需要整池，等 X16 收官。

  l10 已过 warmup(SR 0.850)/solve/emit/smoke+T7，**23:54 进 16 臂 sweep**（预计 3.5–7 h），
  之后 gate-only + 聚合 + 出图 → `GATE-PARETO-DONE`。
- **X17 服务层改造：plan v1、v2 连续被自审判死**，正确方向已查清但**待 owner 定范围**（见 §5）。

---

## 0. 接手第一步

```bash
date                       # 汇报一律本机本地时间 (CDT)
export HOME=/home/weiland
timeout 100 tether exec weilandserver -- bash -lc 'export HOME=/home/weiland;
  echo "chain=$(tmux has-session -t "=gpchain" 2>/dev/null && echo up||echo GONE) \
sched=$(tmux has-session -t "=libgp" 2>/dev/null && echo up||echo down) \
srv=$(tmux ls|grep -c ^libsrv) keep=$(tmux has-session -t "=keepwarm" 2>/dev/null && echo up||echo DEAD)";
  grep -a "GP ===" /tmp/gpchain.log | tail -1'
```

### ⚠⚠ 操作铁律（本会话新踩出来三条，全部有血）

1. **tmux 的 `-t` 目标必须写 `'=name'`**。没有精确匹配时 tmux **回落到前缀匹配且返回 0**：
   `kill-session -t libgp` 会把 `libgpchain` **自己杀掉**，无任何报错。
   症状：链打完相位横幅就凭空消失、日志停在那行、没有 scheduler、没有 status 文件。
   我为此误判三次（先怪 `tee`、再怪 `bash -x`）。实证：`tmux new -s probechain` 后
   `tmux kill-session -t probe` 返回 0 且 `probechain` 消失。
2. **等待循环会自匹配**：`until ! pgrep -f X; do sleep; done` 这条命令自己的 argv 就含 X，
   **永远不退出**。本会话因此空转 25 分钟（pytest 其实 20 秒就结束了）。
   要等后台任务就用 `run_in_background: true`，别写 pgrep 轮询。
3. **仿真 venv 导不进 `openpi`**：客户端 `PYTHONPATH=.` 只有仓库根，而 `openpi` 在 `src/` 下。
   `--per-step-log-dir` 是客户端**唯一**会 import openpi 的地方（`openpi.serving.per_step_recorder`），
   所以只有开逐步捕获才炸，症状是「车队产出 0 行」而非导入错误。已修为 `PYTHONPATH=.:src`。
4. 打 weilandserver 用 `export HOME=/home/weiland`；打 timan107 **在远端** `export HOME=/home/zixuans8`
   （本地 HOME 必须留 `/home/weiland`，否则 tether CLI 找不到 session）。
5. **清场与发车拆两次 tether 调用**；`pgrep -f` 用 `[m]ai[n].py` 双重装甲。
6. **起链用重定向不用 tee，会话名不能以 `libgp` 为前缀**（见第 1 条）。
7. **挂了 Monitor 就别再轮询**——owner 明确打断过一次。

---

## 1. X16 现状与恢复

**拓扑**：weilandserver 起 6 个 server（`libsrv23160..23165`）+ 链 `gpchain`（日志 `/tmp/gpchain.log`）
+ 每相位新起的调度器 `libgp`（日志 `/tmp/libgp.log`）；timan107 出 worker（`lw<port>_<n>`）。
spatial 12 worker/槽（6 槽 = 72），**l10 8 槽（48）**——l10 worker 3.1 GB，72×3.1 > timan107 的 220 GB。

**相位顺序（每 suite）**：template → warmup pool → warmup(100ep) → solve → emit sweep+gate-only
→ smoke(2 臂×20ep)+T7 → sweep(16 臂×500ep) → gate-only(1 臂×500ep) → aggregate。
两 suite 都完再 plot+manifest → `GATE-PARETO-DONE`。

**重挂命令**（脚本按产物续跑，幂等；会话名与重定向都不能改）：
```bash
tmux new -s gpchain -d "cd /home/weiland/openpi && bash exp/libero_groot/run_gate_pareto.sh > /tmp/gpchain.log 2>&1"
```
⚠ 链是 **fail-closed**：`chain=GONE` 且日志无 `GATE-PARETO-DONE` = **真故障不是抖动**，
先读 `tail -20 /tmp/gpchain.log` 找 `FAILED:`、再读 `grep PHASE-FAILED /tmp/libgp.log`，诊断清楚再重挂。
`sched=down` 但 `chain=up` 是正常相位间隙（solve / emit / aggregate / T7），别干预。

**keepwarm 任何情况不许关**（4090 冷态陡热会静默算错）：
```bash
tmux new -s keepwarm -d "cd /home/weiland/openpi && /home/weiland/.local/bin/uv run python /home/weiland/gtp_logs/gpu_keepwarm.py 2>&1 | tee -a /home/weiland/gtp_logs/keepwarm.log"
```

---

## 2. spatial 结果（已收官，这是本轮的科学产出）

17 臂 × 500 集 = 8,500 集、208,020 次门决策，partial 0，聚合过完整性门 I1–I6。

| 指标 | 值 |
|---|---|
| teacher ratio 跨度 | **0.225 – 0.855** |
| 成功率跨度 | 0.822 – 0.940 |
| **gate-only 星标** | **tr 0.155 / SR 0.832** |

**结论 1 —— 与 pi0.5 相反：关掉判据几乎不要钱。** gate-only（tr 0.155/SR 0.832）vs 最宽松的
fh80（tr 0.225/SR 0.834）：**教师调用再少 31%，成功率只掉 0.2 个点**，基本落在前沿上。
pi0.5 那条线的 §3.4 明写 gate-only「严格落在前沿延长线下方」、spatial 掉约 11.8 点。
机制解释：fh80 的教师调用里判据 MISS 只占 0.064、门跳过占 0.160 ——
**宽松端真正在干活的是门不是判据**。这直接关系到「verdict 层值不值得留」，两执行体给了相反答案。

**结论 2 —— 教师来源分解随判据松紧此消彼长**（同一 suite）：

| 臂 | tr | = 门跳过 | + 判据 MISS |
|---|---|---|---|
| fh05 | 0.855 | 0.038 | 0.817 |
| fh50 | 0.366 | 0.139 | 0.227 |
| fh80 | 0.225 | 0.160 | 0.064 |

判据越松，门贡献越大并收敛到 gate-only 地板。这是 `L=6` 的 V2 注入语义决定的
（连续 L 次 FULL_HIT 才注入，严判据不断清零计数器）。

**结论 3 —— 可用工作点（配对符号翻转检验，2000 次置换，同一批 500 个 (task,init)）**：
以 fh05 为参照，fh20（tr 0.590）Δ=−2.4 pp、**p=0.155 测不出差异**；fh30（tr 0.494）Δ=−4.6 pp、p=0.007 是真退让。
沿前沿插值：1 pp 对应 **tr≈0.77**，tr=0.5 处约 3.4 pp（前沿是上包络、偏乐观，真实代价 3.4–4.6 pp）。

⚠ **锚点陷阱**：fh05 的 0.940 **高于**任何教师参照，**不能拿它当"教师上限"算退让**，那会系统性夸大代价。
以官方 A 池 0.92 为锚，fh20 只差 **0.4 pp**（但那个分母 n=50、±4 pp，见 §4）。

**图**：`exp/libero_groot/analysis/gate_pareto/preview/pareto_libero_spatial.png`
（已叠 pi0.5 两条前沿；`preview/` 是子目录，manifest 只摘要上级目录的**文件**，不会污染正式台账）。
库规模三家对齐（约 50 条轨迹 / 约 1050 entries，pi0.5 的 `cs` 臂就是同一个 cache_size S3 构造）
⇒ 差异归因于**执行体与检索配方**，不是库大小。⚠ 我一度把 note 里的 "49 trajectories" 读成每任务、
凭空造了个 10 倍库差，被 owner 当场纠正——引 note 前先看清是总数还是每任务。

---

## 3. 报告口径（写进最终报告，plan §10）

1. 门超参 `j=3/probe=3/L=6` 是 pi0.5 赢点移植，GR00T 上未调；θ 是每库自己 warmup 重解。
   只能说"pi0.5 工作点移植到 GR00T 的表现"，不能说"GR00T 的最优门"。
2. **不出 inference ratio 图**：`0.15195+0.84805·tr` 是 pi0.5 CUDA-Graph 档常数，与 GR00T 无关。
3. 权重是在 A 池（=评测集）上选的，绝对数字带选择偏差；跨臂比较不受影响。
4. 单库（每 suite 只跑 S3）。
5. gate-only 的 `L` 取 6（同主扫描），pi0.5 取 8 ⇒ 两线 gate-only 数值不可直接比。
6. **A 池上没有纯教师基线（owner 2026-08-24 裁定不补）**。可用分母只有官方 46/50=0.92（n=50，±4 pp）
   与我们采集期的 B 池 0.912（B 池更难）。凡"达教师 X%"必须标注分母来源与样本量。
   补法：warmup 同一份 yaml、init 换 A 池、集数 500，约 80 分钟/suite。
7. **名义 f_FH ≠ 实测命中率**：B 池强制全 MISS 标定 vs A 池缓存反馈执行，fh80 名义 0.80 实测 0.914。
   不影响前沿（横轴是实测 tr），只影响"名义→工作点"的映射。

---

## 4. X17（服务层性能）—— **v5 · Code 完成 · 在 G2**

owner 要求「GR00T/RoboCasa 都改成 pi0.5 的高性能方式」、「conductor 也要进来」，并 **override G1**（改自审 agent）。
**四版计划连败、共六轮自审**，每一版的死因都不同，值得逐条记住：

| 版本 | 死因 | 推翻它的锚点 |
|---|---|---|
| v1 | 称自家调度器是「cell 级静态分配」；称拆锁 = 乘性提升 | `orchestrate_search.py:21-23`（共享队列抢活）；`serving_throughput_problem.md:536-543`（**CUDA launch 持 GIL**） |
| v2 | 称 conductor 天然消 bubble | `scheduler.py:246-249` 按 server 过滤 stage；`driver.py:141` 一 yaml 一 bin |
| v3 | sibling-shard 收益为零；会切碎标定；把 `--replicas` 当排除 | 相位墙钟 = `max_lane_episodes × 单集时长`，分片不改变任何一条 lane 的集数 |
| **v4** | **把一次僵尸-worker 事故的墙钟安成一个还没跑过的相位的实测成本** | `ws_search.log.md:141-143`（`infer_ms_per_call` 88→2325 ms，健康值 27 min）；`gate_pareto_plan.log.md:325`（它是 **`eval l10`** 的 min/cell，不是 gate-only） |

⇒ 前三版是「给一个**机制**安上它没有的性质」，v4 是「给一个**数字**安上它没有的身份」。

### v5 的量化（每个数字来自本次运行自己的 `/tmp/gpchain.log`）

| 相位 | 墙钟 | 占用 |
|---|---|---|
| spatial warmup | **16.0 min** | 1 槽 |
| spatial sweep（16 臂） | 138.1 min | 6 槽（已打满） |
| spatial **gate-only** | **32.0 min** | 1 槽 |
| spatial 全程 | 190.5 min | |
| l10 warmup | **36.0 min** | 1 槽 |

⇒ **单臂相位 = 48.0 / 190.5 = 25.2%**，收益约 **18%**（两 suite 合计 1.2–1.5 h）。不多，但 owner 要的是这条框架。

### 拓扑：6 个独立端点，**不是** `--replicas`

`task.py:38-47` 的部署不变量规定的就是「多副本 = 多个独立 `ServerEndpoint`」。
`--replicas` 的增量只有**跨端点 work-stealing**（本线实测 2.3–6.0%），代价是
supervisor 单点（`serve_policy.py:802-811`：任一 child 死则全体 `os._exit(1)`，爆炸半径 1/6→6/6）、
`replica_proxy.py:56` 的 30 s fan-out 硬编码、公共端口丢 `/healthz`。
⇒ 列入 plan §8 未决 1，**是排序不是排除**（v3 正是栽在把它当排除）。

### 已 ship（全部带能失败的测试，共 45 条新增）

- `conductor/sharding.py`：eval yaml 摊成每台 server 一个兄弟 stage；**对 warmup 直接 raise**
  （分片会让 N 次 `fetch_dump` 各拿 1/N、N 次 `publish` 互相覆盖，**不报错**）。
- `TaskGraph.validate()` uid 唯一性：重复 uid = 结果被丢 + `all_done()` 永不成立的**静默死锁**。
- `EpisodeResult.duration_s` + `worker.py` 计时：否则「worker 占用率」在任何产物里都算不出来。
- `WorkerSpec.resize_size/replan_steps/env`：GR00T **必须** `--resize-size 256`，
  否则整队起来后**每一集**都在 wire contract 上失败（`worker_entry.py:52-58`）。
- 两个 GR00T 入口 `--allow-dynamic-bundles`（默认关）：守卫从启动期一次改为**每 bundle 一次**；
  storage **只读不重建**（重建 = 每连接每臂一次 GB 级加载）；`bundle_id` 一律非 `"default"`。
- `staged.py` 无条件 clone：CUDA-graph 静态输出缓冲的别名竞态。⚠ **只堵输出侧**，
  graph 的输入同样是静态缓冲 ⇒ **锁仍是必需的，不给拆锁发通行证**。
  ⚠ 该测试在 v3/v4 各被判过一次「不可能失败」；本轮**实测验证**：去掉 clone 即红。
- `run_conductor.py`：6 端点 + 兄弟分片 + `--role driver|agent|all`，含四条「跑起来才炸」的修复
  （重试的 per-step 行按 attempt 去重否则撞死 I5、`<arm>.json` 从 plan 回填 wire 丢掉的 join 字段、
  merge sidecar 的两个计数取自 **plan 与 journal 两处不同来源**、空分片跳过 setup）。

### 待办

**G-perf 与端到端冒烟需要整池，X16 收官前无法执行。** 基线用上表的 48.0 min。


## 4b. `GATE-PARETO-DONE` 之后的收尾手册（机械执行）

链自己会跑完 aggregate + plot。**但链上跑的是旧版分析器**（不含教师来源分解），
所以收官后要补一次重算。顺序：

```bash
export HOME=/home/weiland
# 1) 确认收官
timeout 100 tether exec weilandserver -- bash -lc 'grep -a GATE-PARETO-DONE /tmp/gpchain.log'

# 2) 推新版分析器（此时才允许推；§9 的封锁到此解除）
tether push exp/libero_groot/analysis/gate_pareto/analyze_gate_pareto.py \
  weilandserver:/home/weiland/openpi/exp/libero_groot/analysis/gate_pareto/analyze_gate_pareto.py

# 3) 逐 suite 重跑 aggregate（纯标准库，就地跑；出 gate_skip_ratio / judge_miss_ratio）
for s in libero_spatial libero_10; do
  timeout 300 tether exec weilandserver -- bash -lc "export HOME=/home/weiland; cd /home/weiland/openpi &&
    PYTHONPATH=.:src .venv/bin/python exp/libero_groot/analysis/gate_pareto/analyze_gate_pareto.py aggregate \
      /data/libero_cache/gate_pareto/$s/eval_results \
      /data/libero_cache/gate_pareto/$s/eval_per_step \
      --expect-ep 500 --yaml-dir <该 suite 的 sweep yaml 目录> --out <该 suite 的 summary.json>"
done

# 4) 重出图 + manifest，拉回 preview/
# 5) 拉回 plot_data.json / MANIFEST.json / png / pdf
```

**四份交付物核对**：两 suite 的 `pareto_<suite>.png` + `.pdf`、`plot_data.json`、`MANIFEST.json`。

**报告收尾**：`exp/libero_groot/analysis/gate_pareto/analysis.md` 的 §4（libero_10）待填，
§2 结论 2 的教师来源分解表**必须用第 3 步的新数字重写**（现表是会话内临时算的，已在文中标注）。
要回答的三个问题写在 §4 的占位里：结论 1（判据在宽松端边际贡献很小）是否在第二个 suite 复现、
教师来源分解的交叉点是否移动、两 suite 的可用工作点是否一致。

**X17 的实机验收**（整池空出来之后才能做）：
- G-perf 基线 = §4 表里的 spatial 单臂 48.0 min；测量脚本
  `exp/libero_groot/analysis/gate_pareto/phase_utilisation.py <journal> --workers N --per-arm`。
- 端到端冒烟：两臂 × 少量 episode 走 `run_conductor.py`，**盯 Xid**，并复用完整性门。

## 5. 资产与 commit 状态

⚠ **共享 checkout**：本工作树同时有 X16 / X15(rl_router) / ICLR / RoboCasa 四条线的未提交改动。
handoff 曾记录过一次误提交 X15 的 33 个文件（未 push，`reset --soft` 回滚）。
⇒ **一律 `git commit -- <显式路径清单>`，禁裸 `git commit`。**

**X17 本轮的显式路径清单**（尚未 `git add`，等 owner 授权）：

```
src/openpi/conductor/sharding.py            # 新增
src/openpi/conductor/task.py
src/openpi/conductor/worker.py
src/openpi/conductor/agent.py
src/openpi/conductor/journal.py
src/openpi/conductor/driver.py
src/openpi/cache/groot/staged.py
exp/libero_groot/run_conductor.py           # 新增
exp/libero_groot/serve_groot_libero.py
exp/robocasa365/serve_groot_n15.py
exp/libero_groot/analysis/gate_pareto/analyze_gate_pareto.py
exp/libero_groot/analysis/gate_pareto/shard_imbalance_probe.py   # 新增
exp/libero_groot/analysis/gate_pareto/phase_utilisation.py       # 新增
exp/libero_groot/analysis/gate_pareto/paired_test.py             # 新增
exp/libero_groot/analysis/gate_pareto/analysis.md                # 新增（最终报告）
exp/libero_groot/analysis/gate_pareto/preview/                   # 新增（6 份交付物）
src/openpi/conductor/protocol.py
tests/conductor/test_sharding.py            # 新增
tests/conductor/test_task.py
tests/conductor/test_agent.py
tests/libero_groot/test_dynamic_bundle_guards.py   # 新增
tests/libero_groot/test_conductor_dispatch.py      # 新增
tests/libero_groot/test_phase_utilisation.py       # 新增
tests/conductor/test_journal.py
tests/conductor/test_protocol.py
src/openpi/conductor/protocol.py
tests/libero_groot/test_gate_pareto_analyze.py
tests/cache/groot/test_compiled_vision.py
docs/architecture/cache_system.md
docs/architecture/experiment_conductor.md
docs/architecture/experiment_conductor.en.md
docs/README.md
logs/README.md
logs/groot_serving_perf_parity_plan.log.md
logs/session_handoff_robocasa365.md
```

**X16 尚未提交的部分**：`exp/libero_groot/run_gate_pareto.sh`（`--reference` 叠图）、
`analyze_gate_pareto.py` 的 pi0.5 叠加改动（与 X17 的改动**在同一文件**，同批提交）、
`exp/libero_groot/analysis/gate_pareto/analysis.md`（报告，l10 章节待填）、`preview/` 预览图。

**已 push**（分支 `Ziyang`）：`5942403`（GR00T×LIBERO 全套）、`97f3f64`、`530b599`（首点火两修复）。


## 6. 设备

| 机器 | 角色 | 要点 |
|---|---|---|
| **weilandserver** | server + 链宿主（4090 48G / 88 核 / 251G） | repo 在 **`/home/weiland/openpi`**（不是 `~/projects/openpi`，那是本机路径）；比 origin **落后 14 个 commit** 且有 57 个未提交改动 ⇒ **别 `git pull`**，用 `tether push` 直传具体文件；GR00T 岛 venv `/home/weiland/gr00t_n15_venv/.venv`；`/data` 3.6T；公网段 `ziyanglin.com:23100-23199` |
| **timan107** | sim 车队（48 核 / 220G / 8×GTX1080） | `/scratch/zixuans8/openpi`（**X15 线共用，别写**）；`/tmp/make_shards.py`、`/tmp/libsearch/shards`、`/tmp/libgp_shards_<tag>` |

**实测容量**：单 server 进程 CPU 仅 **161%**（88 核）、GPU 22%、两机负载都极低 ⇒
单臂慢**不是资源打满**，是只发了 12 个 client；而 `_InferLockedPolicy` 使**同一进程内加再多 worker 也无用**。

**Verify 口径更正（2026-08-24 实测）**：`protocols/execution_authority.md` §6 要的是**裸全量 `uv run pytest`**，
不是 blast-radius 子集（旧记忆写反了，G1/G2 都据章程驳回过）。`tests/serving` **已不再挂起**
（连跑两次 387s/390s 正常收敛）。HEAD 既有失败 5 条：`test_libero_main` 源码锁、
`test_prebuilt_matrix_backend` ×2、`test_robocasa_policy_config` ×2（后者只在 4000 测试全量跑里出现，
单独跑与干净 HEAD 都通过 ⇒ 仓库既有测试隔离缺陷，非本线回归）。
`tests/review_tests/` **在 HEAD 上不存在**（审查方本地探针），Execution **可运行、绝不可读/列/检索**。
