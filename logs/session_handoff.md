# Session handoff — 转向 LIBERO × GR00T

> 写于 2026-09-11。上一条线（RoboCasa365 warm-start RIT 帕累托）**已全部收官**，机器已清空。
> 本文件是 compact 之后的唯一交接依据：只留仍然有效的东西。RoboCasa 那条线的过程细节不再重复，
> 需要时查 `logs/rc365_rit_run_progress.md`。

---

## 0. 初始化方式（不变）

owner 的常驻指令，逐字有效：

> 「开始进行实验，我离开了，期间由你独断专行，不要问我任何问题，注意监控体系定位，
> corn 负责定时巡检，monitor 负责条件触发，不要职责混淆，停止条件是完全做完实验，不做完不停」

**三条必须保持的纪律**：

1. ⛔ **不得用 git commit / push 同步代码和数据**。期间要同步任何代码或数据一律走 **tether**。
   git 只在里程碑收口、且 owner 当次明确指示时才动。
2. **server 只在 server 节点跑，worker 只在 timan 族跑**（见 §2）。
3. **读 `experiment-lifecycle` skill 但不挂载**：只取 tether 用法（`dist_experiment_control/docs/usage.md`）
   与设备组用法（`docs/devices.md`），**不要执行它的 §0 初始化**，不要索要 agentchat 账号 / token / 房间。

**其它长期约束**：

- commit message 全英文；**绝不加 `Co-Authored-By: Claude`** 或任何 AI 署名。
- 未经 owner 当次指示不得 `git add`。
- ⛔ **画图脚本一律不许 commit**（`build_*figure*` / `plot_*` / `render_*` / `edit_*figure*` /
  `*_editor.html` / `add_k1_series.py`）。`exp/rit_pareto/build_figure.py` 与 `edit_figure.py`
  属于另一个 session，**碰都不要碰**。
- ⛔ **不要 `rm -rf`**（触发审批弹窗打断无人值守）；删除前先 `wc -l` 核对规模。
- ⛔ **共享机上不要 `pkill -f`**：该模式会匹配到发起命令的 shell 自己，本会话已因此误杀过一次
  传输进程。一律按 tmux session 名精确 kill，或先 `pgrep` 拿 PID 再逐个 kill。
- ⛔ **不要自己启动图形编辑器**（`edit_rit_figure.py`），owner 自己起；我起会占 8765。
- 无人值守期间**禁起 `run_in_background` 后台任务**（触发审批弹窗阻塞会话），用 Monitor。
- 报时刻用本机本地时间（America/Chicago），UTC 只作内部对表。

---

## 1. 上一条线的结论（已收官，只留结果）

RoboCasa365 × warm-start RIT，**27,850 集，五阶段，err / missing 全程 0**：
GR00T k=2/k=3+teacher 9,100 · pi0.5 k=2/k=3+teacher 9,100 · GR00T k=1 4,550 · pi0.5 k=1 4,550。

**最值得记的结论**：两个 teacher 的 k=1 纯阈值曲线都在 IR≈78-85 见顶，**并超过 IR=100 的全推理臂**
（GR00T .694 @ 77.5 vs .669 @ 99.9；pi0.5 .575 @ 85.1 vs .566 @ 100）。

产物：`~/tmp_rit/k1_products/`、`~/tmp_rit/frontier_k1_{groot,pi05}_all13.json`、
`exp/robocasa365/analysis/figures/`。已推送 `11c2ed4`、`3af3ef2`。

---

## 2. 设备拓扑与硬规则（当前有效）

权威文件：`/home/weiland/projects/dist_experiment_control/docs/devices.md`（本会话刚精简过）。

### 配对拓扑：一 server 一车队，两条 lane 互不共用

| lane | server | worker | 入口 | 实测链路 |
|---|---|---|---|---|
| A | **h100** | **timan108** | `149.165.153.233:232xx` | 117 MB/s |
| B | **weilandserver** | **timan107** | `ziyanglin.com:231xx` | 31 MB/s |

⛔ **timan 族（timan1 / 107 / 108）只能当 worker，永不起 server** —— 卡再空也不改。

### 按实验线取参数

| 实验线 | server replica | worker / 每台 timan |
|---|---|---|
| RoboCasa365 × pi0.5 | 4 | 30 |
| RoboCasa365 × GR00T | 5 | 30 |
| **LIBERO** | 同上（按 teacher） | **64** |

⚠ 反向也成立：**LIBERO 的 64 不能照搬到 RoboCasa**（厨房场景每个渲染上下文 1.5-1.6 GiB，
64 个要 100 GiB，两台 timan 都装不下）。

### 为什么不能两个车队挤一台 server（本会话实测）

worker 10→20→30，吞吐 5.60→8.50→10.50 集/分，每 worker 效率 0.560→0.425→0.350；
而两台 worker 机 load 仅 8.5/88 与 8.8/48，server 也没有任何硬资源打满
（功耗 413/700 W、显存 77%、CPU 60%）。**排队发生在服务端单卡上，加 worker 只是排更长的队。**

### 机器要点

- **h100**：Jetstream2，公网 `149.165.153.233`，1×H100 80G，**20 vCPU（全场最少）**，230 GiB RAM。
  防火墙全开、不需 expose。根盘仅 58 G（剩 ~11 G，别再放大东西）；**1 TB 卷挂 `/data`**，
  `/scratch/zixuans8` 也软链到同卷，路径与 timan107 一致（venv 可原样搬）。
  ⚠ 起带 cache 的 server 时**内存是上限不是显存**：每 replica ~30 GiB RSS。
- **weilandserver**：88 核 + 4090 49 GiB。⚠ 驱动是 `-server` 计算版**不带图形栈**；本会话补装了
  `libnvidia-gl-595-server`（版本严格匹配，先 `apt-get -s` 模拟确认不动内核模块）才用上 4090。
  之前是 llvmpipe 软件渲染，7.45 ms/帧**且像素与其它机器不一致，混进实验会污染结果**。
  **新机装完驱动务必验 `GL_RENDERER`。** sudo 密码 owner 给过，用 `sudo -S` 从 stdin 喂，
  不写进文件、不回显。
- **timan107**：48 核，8×GTX1080（8 GiB/卡）。sim venv 在
  `/scratch/zixuans8/Isaac-GR00T/gr00t/eval/sim/robocasa365/robocasa365_uv/.venv`，
  repo 在 `/scratch/zixuans8/openpi_rc365`。
- **timan1**：4×A6000 48G、48 核 —— **车队里最大的未开发算力，没装环境**。

---

## 3. ⚠ 遗留问题（新会话开工先处理）

1. **timan108 仍 OFFLINE**（2026-09-10 起）。我在 3 张 A5000 上开了 45 个 worker
   （98.7% 显存、15 个上下文/卡）把 NVIDIA 内核模块压死：进程全进 D 状态、`SIGKILL` 无效，
   随后 ping / ssh 全不通。**要 owner 或 engrit 从 BMC 复位**，我没有通道。
   它本地盘困着 970 集作废数据（已在别处重跑，不影响结论）。
   **按配对拓扑 LIBERO 的 A lane 要用它，开工前必须先确认它回来了。**
2. **`11c2ed4` 里误提交了 6 个画图脚本**（`build_rit_figure_spec` / `plot_rit_pareto` /
   `plot_rit_pareto_four` / `render_rit_figure` / `edit_rit_figure` / `rit_figure_editor.html`），
   已推送。清理方式 `git rm --cached` 后另起 commit，**需 owner 当次授权**。
   同 commit 还有 8 个出图产物，是否也清 owner 未表态。
3. **工作区有未提交改动**（按规矩不提交）：k=1 上图的全套 —— `add_k1_series.py` 新增，
   三个出图/编辑脚本改动，两份 spec 与三张图重生成。
4. **图形编辑器要硬刷新**：owner 浏览器标签页跑的是旧 JS，`Ctrl+Shift+R` 才会拿到
   带 **Save + export PNG/PDF** 按钮和网格吸附修复的新页面。

---

## 4. 下一条线：LIBERO × GR00T

### 已知资产与坑（开工前逐条复核，别想当然认为还成立）

- **采集线已有基础**：6 路拓扑、`launch_collection.sh` 断点续跑、replan5 判决；
  **教师 B 池低于官方 A 池**。
- ⚠ **夹爪约定相反**：GR00T → LIBERO **必须照抄官方 `normalize_gripper_action`（1-2x 再 sign）**。
  漏掉的表现是**静默 0% 成功且每集跑满步数上限**——不报错，只全灭。
- ⚠ **两条线共用 `cache/groot/*`，连采集器都共用**。denoise 步数 **RoboCasa 4 步 / LIBERO 8 步**
  （`groot_n15_k4_v1` vs `groot_n15_k8_v1`），**schedule 主键必须带步数**，
  否则两条线的库会互相"合法"对上。
- ⚠ **LIBERO init 硬预算**：1000/suite = pruned_init 500（测试，只测量）+ 差集池 500
  （内分 450 train / 50 val）。**无法新铸第三池**，标定与 router 标签只能从 B 池内出。
- **LIBERO 并发**：client 1 进程 = 1 WS 连接；`--num-workers` cap 15 是单进程内 EGL 限制。
  要测 server N 并发就开 N 个独立 `main.py` 进程。
- **闭环瓶颈**是每控制步一次同步 websocket 往返；client 与 server 同机比走 broker 快 5-8×。
- ⚠ **孤儿 worker**：kill conductor 不杀 LIBERO worker，它们挂到 systemd 下继续占 CUDA context
  （实测 48 个吃 27 GB → 自家 server OOM）。白名单回收，且杀与起分两条命令。

### RoboCasa 线留下的、对 LIBERO 同样适用的经验

- ⚠ **env 构建锁**（本会话新增于 `exp/robocasa365/episode_runner.py`）：`max_cached_envs=1`
  使每个 worker 换任务时同时销毁+创建一个 EGL 上下文；N 个 worker 就是 N 对并发操作砸向同一张卡，
  驱动按设备加锁，进程卡进 D 状态。修法是给「销毁+创建」整段加跨进程文件锁
  `RC365_ENV_BUILD_LOCK`（默认 `/tmp/rc365_env_build.lock`，置空可关），**只改时序不改 episode 行为**。
  LIBERO 若复用同一 runner，这把锁要挂上；若是另一套 runner，**先确认它有没有同样的问题**。
- ⚠ **`MjRenderContextOffscreen ... has no attribute 'con'` 的计数是最早的告警信号**，
  比 journal 的 `error` 行早得多 —— **一出现就降载，别等后果**。我上次把它判成良性噪声，丢了一台机器。
- **三段判定语义**：`done` = 解决；`failed` 且 `error=None` = **未成功但有效，计入分母**；
  `error` 非空 = ERR（排除）；**完全没有终态记录 = MISSING（重试耗尽）**。
- **ConductorDriver 是 ep 级 resume**，崩溃后 relaunch 直接同 journal，不需过滤。
- **tether 两个坑**：`push` 默认拒绝覆盖（`dst_exists`），重传加 `--force`；
  `exec` 单次约 10 min 硬上限，长任务一律放 tmux 解耦、本地短轮询。
- **rsync daemon 传输**：`munge symlinks` 默认开会改写绝对软链（venv 的解释器链会废掉），
  关掉后又会丢前导斜杠 —— **传完务必验软链**。非 root 起 daemon 要显式把
  `lock file` / `pid file` 指到 `/tmp`，且**不能写 `uid`/`gid`**（会 setgroups 失败）。

### 开工顺序建议

1. 确认 timan108 是否已复位；没回来则 LIBERO 只能单 lane（weilandserver + timan107）。
2. 逐条复核上面的「已知坑」在当前代码里的状态，尤其**夹爪转换**与 **schedule 主键带步数**。
3. 按 §2 配 replica 与 **64 worker**，先起一条 lane 的 smoke：确认 `GL_RENDERER` 是真 GPU、
   `con` 计数为 0、journal 有正常终态记录，再放量。

---

## 5. 监控体系的职责划分（owner 明确要求，不要混淆）

- **cron**：定时巡检，固定一行汇报。健康就记「巡检 OK: <关键数字>」，
  只在 GPU 掉 0 / 进程消失 / 日志出现 Traceback 时展开。
- **Monitor**：条件触发。守段边界、终态标记、真实错误（journal 的 `error` 行、非良性异常、
  `con` 计数、节点失联）。
  ⚠ **过滤器只抓「你会据此采取行动」的信号**：本会话犯过一次，把良性的 `__del__` 噪声也算进告警，
  每 5 分钟假警一次 —— **告警疲劳正是上次忽略真信号的原因**。
- 实验做完后按 owner 指令收尾：**拉数据回本地 → 关 server/worker → 关监控体系（含 cron）→ 停止待命**。
