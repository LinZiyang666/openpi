# Session handoff — X9b cache_size（compaction 恢复用）

> 重写于 2026-08-17 22:25 CDT。上一版是「本机重启」场景，已整体作废。
> 本文件是 compaction 后接手的**唯一入口**：先读这里，再按 §2 一条命令确认现场。

---

## 1. 一句话状态

**P7 全部收官（08-19 06:10，ALL GROUPS DONE fail=0，14,000/14,000）+ P8 分析完成 + 正式报告已写**：
两套件 primary 族一致落 **分支 G**（`D-teacher` × `Q-fail` × `P-inconc`）——纯 cache 顶档确证落后
teacher 超过 δ=5 pp（l10 +35.2 pp / spatial +16.4 pp），平台未分辨出。
报告：`exp/ablation_study/cache_size/analysis/analysis.md`（+3 张曲线图）。
**远端已拆除**（server/worker 全部 PID 定点清理，GPU 释放 24.5G，邻居无恙）。
**08-19 追加两项已完成**：① 绘图管线重构（owner 指示）：`emit_plot_data.py` → `analysis/plot_data.json`
（单一数据源，来源 sha256，补充实验挂 `latency.<label>`）→ `plot_size.py` 只读该文件；
② **补充实验 X9b-L**（优化 backend × 12 `all` pkl × 两机）收官：延迟对桶大小严格线性
**10–13 µs/条**，优化栈下 l10/S6 检索 31 ms/call ⇒ 比 teacher 快 ~4.4×，"盈亏平衡点"仅原版 backend
存在；WSL l10/S6 因峰值内存 ≈3× fp16-pkl 被 OOM（真发现）。报告 `analysis/relatency.md`，原始数据 `analysis/relatency_data/`（owner 指示图和数据都归 cache_size）。
⚠ 一处更正记录在 plan §12.3：旧 bench pkl 其实也按 `payload.task_key` 分桶（先前探针漏查 payload 层）。
剩余：owner 审读报告 → commit/push 授权 → P9 M-c2 族化收编（须邻居结束）→ P10 文档同步。

## 2. 接手第一条命令

⚠ **P7 已收官、现场已拆**——健康脚本现在应报 `phase=idle`，那是正常终态不是事故。
先读 `exp/ablation_study/cache_size/analysis/analysis.md` 与 plan §12.2.4。

```bash
tether exec weilandserver -- bash -lc 'bash /home/weiland/cache_size_p1b/cs_phase_health.sh'
```

脚本**自己从活着的进程推断阶段**（recal / eval / srv-idle / idle），不依赖任何需人工更新的标记
——上一版用过 `/tmp/cs_current_suite` 那类标记，标记过期时死掉的跑批看起来和健康的一模一样。

输出形如：
`[MM-DD HH:MM:SS] phase=eval recal=4/4 arms=28/28 srv=2 workers=8 eval_rows=N arms_seen=M gpu_free=..M ram_avail=..G data_free=..G`

三种情形：① `phase=eval` 且 rows 在涨 → 正常，等；② 出现 `ALERT ...` → 见 §6；
③ `phase=idle/srv-idle` 且 rows 未达 14,000 → 跑批已停，见 §6。

## 2b. ⚠ 无人值守授权（owner 2026-08-18 00:1x：「期间不要出现任何打断工作的问题或选择，一切中途遇到的问题由你专断」）

**生效范围**：owner 睡眠期间。所有中途问题**自行裁决并执行**，不弹窗、不停下等指示、不在正文里挂"待裁决"。
事后一次性汇报做了什么、为什么。

**预先冻结的处置表**（照做，不要临场重新发明）：

| 情形 | 处置 |
|---|---|
| 某组跑完（journal 达 3,000 / 4,000） | 自动进下一组（`run_all_groups.sh` 自己会做）。**并立刻对该组跑 P8 分析**（§9 命令），产物落 `/tmp/size_<suite>_<filt>.{json,md}` 后拉回 |
| 逐 episode FULL_HIT 门 fail **且缺口 ≤ 2 条** | 按已有 owner 裁定处理：**当作失败计入分母、不重跑**，在 plan 记披露三件套（缺几条 / 方向是否可证伪 / 影响上界 pp） |
| FULL_HIT 门 fail **且缺口 > 2 条或非收尾位置** | 那是系统性问题不是收尾竞态。**停该组、记录现象、跳到下一组继续**，别重跑整组浪费一夜 |
| server 挂 / driver 没了 | 按 §6.1 **只续跑没跑完的组**（`CS_GROUPS`），已完成组的 journal 不碰 |
| worker 被外部杀 | 不动。conductor 会重试，accepted-attempt 台账保证不丢数据 |
| RAM < 60G 或 GPU 告危 | **降** worker（`WORKERS=2` 续跑），绝不加。别去动邻居 session |
| 邻居 session 占资源 | 一律不碰。只按 `:8030` / 自己的 PID 动手 |
| 四组全跑完 | 出两套件正式报告 → `exp/ablation_study/cache_size/analysis/*.md`（纯 md），图 → 同目录 png |

**明确不在专断范围内（仍需 owner 明确指示）**：`git add` / `commit` / `push`。
代码改动留在工作树里本来就是安全的，不需要靠 commit 来保住；而工作树里还混着别的 session 的改动，
夜里替 owner 决定"哪些进这个 commit"是越权。

## 3. ⚠ session 内资产（compaction / 本机重启会清掉它们，不会清掉远端进程）

⚠ **本机 WSL 重启不影响远端跑批**（08-18 11:0x 实证：重启后 `csmain`/`cssrv` 仍是 08-17 22:14 的原始 session，
driver 活着、journal 连续）。但 cron 与 Monitor 是 session-only，**重启后必须按下表重挂**，ID 会变。

⚠ **08-19 06:1x 已随 P7 收官全部退役**（cron `0c532e74` 已删、Monitor `bzn2bszrp` 已停）——下表仅作历史记录与将来重建参考。

| 资产 | ID | 作用 | 挂掉了怎么重建 |
|---|---|---|---|
| **L3 cron 巡检** | `0c532e74` | `7,27,47 * * * *`（每 20 min）跑 §2 的健康脚本；健康只记一行，ALERT 才展开 | `CronList` 查；没了就按 §3.1 的 prompt 重建 |
| **L5 push Monitor** | `bzn2bszrp` | `tail -F /tmp/cssrv.log /tmp/cseval.log` 只 grep OOM/CUDA error/Killed/No space/Segfault；**静默时零输出** | 重建见 §3.2 |

**分工纪律（owner 明确要求）**：定时巡检归 **cron**；Monitor **只做事件驱动的秒级推送**，
不做周期轮询。先前 Monitor 每 4 分钟报一次进度，既重复 cron 又毁掉信噪比，已废止。

### 3.1 cron prompt 要点（重建时照抄语义）
- 跑 §2 命令；健康 → 主会话一行；`ALERT` → 报警 + 看日志定位，**不擅自重启/kill 任何进程**
- 这台机上还有**另一个 session 的 X14 rl_router**（:8000 / :7002 / :7003）——**一律不碰**
- 要杀本实验的 server 只能 `pgrep -f "[s]erve_policy.py.*8030"` 取 PID **定点** kill；
  **严禁** `pkill -f serve_policy` 或 `pkill -f worker_entry`（会打死 X14）
- 不弹阻塞窗口；不 commit / 不 push（需 owner 明确指示）

### 3.2 Monitor 重建
```
tether exec --timeout 168h weilandserver -- bash -lc 'tail -F -n0 /tmp/cssrv.log /tmp/csmain_trace.log 2>/dev/null | grep --line-buffered -v "^+" | grep --line-buffered -iE "out of memory|CUDA error|Killed|MemoryError|No space left|Segmentation fault|core dumped|Cannot allocate|ALL GROUPS DONE|!! .* FAILED|server did not come up|server FAILED|server TIMEOUT"'
```
persistent: true。⚠ **`grep -v "^+"` 不能省**：trace 日志带 `-x`，`restart_server()` 里"检查有没有 OOM"的
grep 命令本身会被 `set -x` 打印出来（含 "out of memory" 字样），不滤掉就是对自己脚印的连环误报（08-18 已踩）。

## 4. 远端拓扑（weilandserver，单机闭环，client 走 127.0.0.1 不经 broker）

| tmux | 端口 | 归属 | 内容 |
|---|---|---|---|
| **`csmain`** | — | **本实验** | P7 驱动：`run_all_groups.sh`，按组顺序跑并在组间重启 server |
| **`cssrv`** | **:8030** | **本实验** | 评测 server（`--concurrent`，不带 `--cache_config`，yaml 由 conductor 热切换） |
| **`csw2`** | — | **本实验** | **8 个热加入的额外 worker（w4–w11）**，owner 08-18 授权 12 worker。conductor pull 协议不认 worker 身份、只认 server_key ⇒ 起进程连上同一 driver 端口 (46327) 即领活；跑完 driver 对每个拉活者发 MSG_SHUTDOWN ⇒ **随组自动退出，不留孤儿**。若 driver 异常死掉则须手动清：按 `--worker-id w(4\|5\|…\|11)` + `--server-key 127.0.0.1:8030` 定点 kill。⚠ 起它们的脚本 `launch_extra_workers.sh` 曾因 tmux 非登录 shell 无 conda 而全灭——PATH 必须显式 export |
| ~~`rlrsrv` / `rlrsc` / `rlrsc2` / `rlrm6`~~ | ~~:8000 / :7002 / :7003~~ | X14 rl_router | **08-17 22:4x 已全部结束**，端口已释放 |
| `rcg2gsrv` 等 | :8010 / **:8000** | ⛔ **别的 session** | 一律不碰。08-18 10:29 起 :8000 上有 **16 个 worker**，会大幅改变 CPU 余量，加 worker 前必须先看 `uptime` |

⚠ **健康脚本的 worker 计数曾被邻居污染**（08-18 10:29 读出 11，实际 4）：裸 `grep 8030` 会命中邻居命令行里
任何含 8030 的字段。已改为锚 `--server-key 127.0.0.1:8030`（备份 `cs_phase_health.sh.bak`）。
**任何"本实验的 X 有几个"的计数都必须锚在唯一标识上**，不能靠一个数字子串。

⚠ **邻居会换人**：X14 结束的同一分钟就冒出了 `rcg2gsrv`。所以纪律不能写成"避开 X14 的那三个端口"，
而是**只按自己的端口/PID 动手**：`pgrep -f "[s]erve_policy.py.*8030"` 取 PID 定点 kill，
worker 也只认带 `8030` 的那些。**任何宽模式 `pkill -f serve_policy` / `pkill -f worker_entry` 一律禁止**——
无论此刻邻居是谁。

日志：`/tmp/csmain_trace.log`（当前 l10 重跑，带 `-x`）、`/tmp/cssrv.log`、
`/tmp/csmain.log`（spatial 两组的历史日志）。

## 5. 数据与产物（全在 weilandserver 的 `/data`，repo 侧经软链）

```
/data/openpi/ablation_study/cache_size/
├── collect_h5/{libero_spatial,libero_10}/task_N/episode_M.h5   1,000 条，131 G，已验收
├── save_traj/…                                                 客户端副产物
├── artifacts/cache_size_<suite>_<filter>_<tier>.pkl            24 个库，49 G
├── results/                                                    采集台账 + sha ledger
├── eval/journal_<组>.jsonl + per_step_<组>.jsonl               ← P7 正式产物
└── smoke_artifacts/                                            P6 smoke，**已隔离，勿混入分析**
```

repo 侧软链（名字自带指向信息，`.gitignore:6` 命中）：
`exp/ablation_study/cache_size/data/{artifacts,collect_h5}__symlink__slash_data_openpi`

配置（**未被 gitignore，应进 git**）：`exp/ablation_study/cache_size/config/`
`apool_<suite>.yaml`（A 池冻结记录）、`size_grid_<suite>_<filter>.yaml`、`lists_{success,all}/`、
`arms/`（28 个臂）、`recal/`（4 份）、`matrix_<suite>_<filter>.yaml`（4 份）、`entries_*.json`

## 6. 故障处理

| 现象 | 判断 | 动作 |
|---|---|---|
| `tether exec` 报 `i/o timeout to broker` | 控制面瞬断，**不是实验挂了** | `nc -zv linziyang.top 443` + `tether node ls -a` 确认节点 ONLINE，重试即可 |
| `ALERT ... driver is gone` | 会话还在、跑批已死（最阴的形态） | 看 `/tmp/csmain_trace.log` 末尾；按 §6.1 续跑 |
| `ALERT RAM available < 60G` | BackendPool 不 evict，库常驻 | 进一步拆组，勿加 worker |
| worker 被外部杀 | 另一 session 的宽模式 pkill | 数据不丢（conductor 重试 + accepted-attempt 台账），只慢 |

### 6.1 续跑某几组
```bash
tether exec weilandserver -- bash -lc 'cat > /tmp/csgroups.sh <<EOF
export WORKERS=4
export CS_GROUPS="libero_10 success|libero_10 all"
exec bash -x /home/weiland/cache_size_p1b/run_all_groups.sh
EOF
tmux new -s csmain -d "bash /tmp/csgroups.sh > /tmp/csmain_trace.log 2>&1"'
```
⚠ **变量名必须是 `CS_GROUPS`**：`GROUPS` 是 bash 内建（调用者的组 ID 数组），赋值会被静默吃掉、解析出 `1000`。
⚠ **不要在前台手动跑该脚本调试**：它开头就 `tmux kill-session -t cssrv`，会把正在加载模型的 server 杀掉，
把正跑着的实例变成空转（已犯过，连废两次重启）。

## 7. 已完成的结果（spatial 两组，7,000 ep，逐臂校验已做）

两组都做过逐臂核验：**每臂 500/500 accepted、attempt 恒为 1、10 任务 × 50 init 网格完整**。
`all` 组的逐 episode FULL_HIT 门 **PASS**（8/8 臂见证齐全、`stale_rows_ignored=0`）；
`success` 组因下述 1 条被门标 FAILED。

| 档 | 每任务轨迹 | `all`（**primary**） | `success`（次级） |
|---|---:|---:|---:|
| S1 | 1 | 0.5020 | 0.5320 |
| S2 | 2 | 0.4640 | 0.4880 |
| S3 | 5 | 0.6880 | 0.6880 |
| S4 | 10 | 0.7280 | 0.6980 |
| S5 | 20 | 0.7620 | 0.7480 |
| S6 | 45 / 43.9 | **0.8100** | **0.8160** |

teacher 锚 0.974 ⇒ 顶档 gap ≈ 16.4 pp（δ=5 pp）。**确认性落格只能由 P8 的 Holm 出，上表仅是效应量。**

**primary = `all`**（§3.1b 裁定 1 把自变量定义为「采集轨迹数/任务」；4 个 recal 敏感性臂也只挂在 `all` 上）。
敏感性臂（descriptive）：spatial S1 recal−fixed **+0.6 pp**、S6 **+1.0 pp**。⚠ 点估计小 **不等于**等价成立——正式 ±3 pp 检验只有 **S6 通过**，见下。

**owner 裁定**：`success` 组 S6 的最后一条（`:eval:9:49`）无 FULL_HIT 见证 →
**当作失败照常计入，不重跑**；报告须披露「1/3000 无见证 + 影响上界 +0.16pp + 方向对 cache 不利」。
⚠ 原先预判「`all` 组也会缺最后一条」——**实测未发生**，plan §12.2 已更正。

**spatial 主族 P8 已出结论**（`all`，全部门禁一次通过，细节与判读见 plan §12.2.1）：

> **分支 G** — `D-teacher` × `Q-fail` × `P-inconc`，`M-yes = False`。
> `gap = 0.974 − 0.810 = +0.1640`，BCa CI `[+0.1060, +0.2080]`；
> 检验 8 Holm 后 p = **0.0205** ⇒ **确认 gap > δ=5 pp**；检验 6 Holm 后 p = 0.0078 ⇒ teacher 显著在前。

三条判读纪律（别读错）：① **S1→S2 的下降不是非单调性证据**（检验 1 原始 p=0.5766、Holm 后 1.0000）；
② **`P-inconc` 意味着曲线没饱和**（S5–S6 斜率 CI `[-0.20, +9.60] pp`）⇒ **不得**写"更多数据也没用"；
③ 敏感性臂 **S6 等价成立、S1 不成立**（宽度问题，非不等价证据）⇒ 主曲线措辞须限定"在生产标定参数下"。

**l10/success（次级族，descriptive）也已收官并出 P8**（08-18 11:16，13h02m，门禁一次过，
`final snapshot: 0 rows` = 收尾竞态修复实证生效）：

| 档 | S1 | S2 | S3 | S4 | S5 | S6 | teacher |
|---|---:|---:|---:|---:|---:|---:|---:|
| SR | 0.274 | 0.360 | 0.464 | 0.470 | 0.498 | **0.522** | **0.868** |

分支 G（descriptive）：`D-none` × `Q-fail` × `P-inconc`。gap +34.6 pp，CI `[+20.2, +49.4] pp`。
⚠ **`D-none × Q-fail` 可达格真实出现**（检验 8 Holm 后 0.0468 拒、检验 6 Holm 后 0.0615 未拒，两尾不同），
§8.4.1b 预注册读法正确接住；`ci_gate_disagreements` 里那条分歧报告须原样保留。细节 plan §12.2.3。

**两个已落袋的结构性发现**（细节见 plan §12.2）：
1. `spatial/all` 的 S1/S1_recal 在 **task 8 上 0/50**、S2 3/50（`success` 同档 16/50、15/50）。
   两族 S1 列表只差一条，就是 task 8：`all` 取到 `episode_0`（采集台账 `episode_id=400`，失败）。
   **一条失败轨迹足以让 `always_hit` 把整个任务打死**。
2. `l10` 的 S1/S2 两族库**逐字节相同**（555 / 1,094 entries）⇒ 跑完后两者之差是**纯运行噪声**，
   是本实验的经验噪声底，报告须作为管线自检明写。

## 8. 延迟模型（P6 实测，同争用条件两点解出）

**per-call ≈ 126 ms + 44.05 ms / 千 entries**（126 ms = Stage1 前向 + 网络；44 ms/千条 = 暴力检索）

| 档 | l10 entries | per-call | vs teacher 690 ms |
|---|---:|---:|---:|
| S4 | 5,660 | 375 ms | 1.84× |
| S5 | 11,720 | 642 ms | 1.07× |
| S6 | 26,493 | 1,293 ms | **0.53×** |

**盈亏平衡 ~12,700 entries** —— size 轴同时是成本轴，过了那点纯 cache 比 teacher 还慢。
⚠ 斜率含 X14 争用（同点上 search 分量是 shipped bench 的 3.4×），**X14 结束后应用 2 臂 × 10 ep 重测**。

内存实测：库常驻 ≈ **1.22× pkl**（S6 的 9 G pkl → +11 G RSS），最大一组约 30 G，占可用 14%（门 50%）。

## 9. 后续（plan §12）

**P8 已可复现执行**（spatial 主族已跑通，命令原样照抄换 suite/filter 即可）：

```bash
tether exec --timeout 20m weilandserver -- bash -lc 'cd /home/weiland/openpi && PYTHONPATH=. ./.venv/bin/python \
  exp/ablation_study/cache_size/analysis/analyze_size.py \
  --suite <SUITE> --outcome-filter <all|success> \
  --journal        /data/openpi/ablation_study/cache_size/eval/journal_<SUITE>_<FILT>.jsonl \
  --per-step       /data/openpi/ablation_study/cache_size/eval/per_step_<SUITE>_<FILT>.jsonl \
  --launch-record  /data/openpi/ablation_study/cache_size/eval/per_step_<SUITE>_<FILT>.jsonl.launch.json \
  --teacher-anchor exp/ablation_study/data/anchors/<SUITE>_teacher \
  --apool-digest   <config/apool_<SUITE>.yaml 里的 rollup> \
  --grid           exp/ablation_study/cache_size/config/size_grid_<SUITE>_<FILT>.yaml \
  --out-json /tmp/size_<SUITE>_<FILT>.json --out-md /tmp/size_<SUITE>_<FILT>.md'
```

apool digest：spatial `0eeece46a08b958efe7b7db4e6b13d3269b0433be4e20fbae3c0f352bc3aca9c`，
l10 见 `config/apool_libero_10.yaml`。
⚠ **teacher anchor 原本不在远端**（`data/` 被 gitignore），已 push 两套件共 6 个 json 过去；
换机器要重推。anchor 协议三轴（init 集 / seed=7 / replan_steps=5）已实测与评测臂一致，见 plan §12.2。

**跑不了 `spatial/success`**：它的逐 episode FULL_HIT 门会因那 1 条缺见证的 episode 而 fail。
**不要**为它开 `--smoke`（那会连完整性、launch 绑定、attempt 校验一起放掉）。owner 裁定只说"当作失败计入"，
§12.2 的 SR 表已经这么计（0.8160 = 408/500），descriptive 目的已达成，**不必也不应弱化那道门**。

之后：`plot_size.py`（x 轴措辞现按 `outcome_filter` 自动切"collected/successful"）→
**P9 M-c2 族化收编（须 X14 结束）** → **P10 文档同步**。
最终报告落 `exp/ablation_study/cache_size/analysis/*.md`（纯 md），逐轮 log 留 `logs/`。

**ETA 推算**（供判断是否异常，不是承诺）：per-call ≈ `126 ms + 44.05 ms/千 entries`，
l10 每 episode ≈ 52 次调用 + 约 22.8 s 固定开销（由 S1 实测 470 ep/h @4 worker 反解）：

| 组 | 各档 s/ep | 4 worker 下耗时 |
|---|---|---:|
| l10 / success | 31 / 32 / 36 / 42 / 54 / 76 | ≈ 9.4 h |
| l10 / all | 31 / 32 / 36 / 42 / 56 / 90 (+S1r 31, S6r 90) | ≈ 14.1 h |

合计 ≈ **23.5 h**，自 08-17 22:14 起 ⇒ **08-18 21:00–22:00**。X14 若提前结束会更快。

## 10. 本地未提交清单（HEAD 已被别的 session 推进到 `480b2ad`；本实验代码停在 `ceb0158`）

- `exp/ablation_study/cache_size/analysis/`：**`analyze_size.py`（+`--outcome-filter`，复用 `arm_name()`，
  敏感性臂按主/次族区分，产出带 `family_role`）**、**`plot_size.py`（x 轴措辞跟随 filter）**
- `exp/ablation_study/cache_size/`：`verify_collect.py`(新)、`full_hit.py`(新)、
  `verify_apool.py`(+`load_init_states` torch 1.11/2.x 兼容)、`run_size_eval.py`(+`_write_snapshot` 收尾修复、
  A 池重哈希、accepted-attempt 门)、`emit_size_grid.py` / `emit_size_yamls.py` / `emit_arm_matrix.py` /
  `run_recal.py` / `build_size_artifacts.py`（均 +`--outcome-filter` 与命名带 filter）
- `tests/ablation_study/cache_size/`：**`test_cache_size_analysis.py` +5（端到端走 `main()`）、
  `test_cache_size_plot.py` +3（轴标签）** ⇒ 全量 **210 passed / 5 skipped**；`test_cache_size_verify_collect.py`(新)、
  `test_cache_size_build_driver.py`(新)、以及 grid/yamls/recal/runner/apool 的增补
- `logs/cache_size_ablation_plan.log.md` §3.1b（两条 owner 裁定）+ §12.1（执行记录）
- `logs/session_handoff_cache_size.md`（本文件）
- `exp/ablation_study/cache_size/config/`：**全部已从远端拉回**（28 臂 yaml / 4 matrix / 4 recal /
  `lists_{success,all}` / `entries_*.json` / `size_grid_*`，共 72 文件）。`apool_*.yaml` 与运行时
  实际传给 `--apool-record` 的两份 sha256 逐位一致（spatial `d654643f…` / l10 `2be5df50…`）

⚠ 工作树里另有 `exp/rl_router/*`、`src/openpi/cache/components/mlp_router_judge.py`、
`docs/iclr/tier_paper_outline.zh.md` 等改动，**属于别的 session，不要混进本实验的 commit**。
