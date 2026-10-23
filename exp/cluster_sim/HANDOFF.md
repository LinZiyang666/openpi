# 集群部署模拟实验：会话交接（「openpi新实验」会话专用）

compact 后先读本文件，再读 `DESIGN.md`（owner 的需求与决议，权威），需要服务端细节时读 `R16_SERVING_MAP.md`。
共用的 `logs/session_handoff.md` 由看守会话维护，本实验只在那里留了一块「10-21 22:xx 新实验」摘要。

## 我是谁、管什么

- 本会话名「openpi新实验」，负责 **π0.5 · LIBERO-10 集群部署模拟实验**（owner 10-21 晚交代）。
- 在跑正式批次（H100、pod、4090 队列）的看守由兄弟会话 `[75fc7a]` 负责（它的名字是 owner 那句「我们要做一个新实验，是模拟集群部署实验…」的原文）。我不碰那些批次。
- owner 10-21 23:0x 说「我睡觉了，你独自推进」，并设了目标「推进吞吐实验」。

## owner 定下的实验内容（摘要，全文见 DESIGN.md）

- **对比两方**：全步纯推理（K=10、无缓存）vs 完整系统（成功过滤库 · 500 级库 · Q3 实测档位 · 减步 K=1 · 修正器 + 停滞门 · **不开长执行段**）。
- **服务端**：4090 单份模型（不要副本），显存最多 24 GB。可以重写服务、批处理、调度，把 4090 用满。前段（视觉、算缓存键）和后段（未命中才进）可以各有批大小和等待时间。推理代码两边必须相同，享受同样的优化。**两边各自调到最优，正式实验每边一套参数，只变 N。**
- **请求端**：在 4090 上用完整系统真实跑一遍 LIBERO-10 的 500 局，录下所有请求（像 debug 模式，不收模型那一侧的数据），两边回放同一份。回放在本机进行：
  - 用种子打乱 500 条轨迹，N 个槽位依次取轨迹回放；
  - 冷启动时 N 个槽位在一个请求间隔内均匀错开；
  - 机器人收到 action 后等「执行时间 + 2 ms」再发下一条。
- **记录**：每条请求在服务器上「收到 → 发出」的延迟、服务器每秒处理的请求数、每个 N 放完 500 局的总时间。不测成功率。
- **正式 N**：5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100。
- **owner 追加**：
  - CPU 线程要放开（单线程等于没优化）；
  - 讨论时别加防御性核对（记忆 feedback-keep-it-simple 已加这条）；
  - 先写代码，正式实验排在 4090 空出之后。

## 已定的实现选择（我定的，owner 授权「具体手段由你决定」）

- **旋钮档位**（`LEVELS.json`，正式选档规则）：任务 0 用 ir25，任务 4 用 ir40，任务 6 用 ir32，其余 7 个任务旋钮关。
  - 档位分组来自 `fa_pi_r_f_l_q_l500_ir{25,40,32}`；旋钮关分组是 `fa_pi_r_f_o_q_l500_off`。
- **不开长执行段**：去掉插件参数 `--os-blind --os-policy-tail --os-policy-tail-blocks 1`。这样每条请求都先跑第 1 段看图，再由方法判命中或未命中。
- **请求间隔**：LIBERO 实测 20 Hz，R16 worker 每条请求执行 5 步，间隔 = 5 × 50 ms + 2 ms = 252 ms。
- **服务端**（两种模式同一服务树 `/home/weiland/projects/openpi_r16_q3_server`，同一模型、推理拦截器和批处理协调器）：
  - 纯推理 = `scripts.serve_policy --cache`（拦截器不带缓存，依次跑第 1、2、3 段，K=10）。
  - 完整系统 = R16 常驻 `exp.offline_search.rounds.r16.resident` 加四个分组；启动后按 R16 driver 的方式发 load_cache_config 预载这些分组。
  - `serve_entry.py` 里两边共用的改动：
    - 请求计时，写 `requests_<pid>.csv`（`conn,kind,t_recv,t_sent`）；
    - 每段单独的批上限和等待（取到第一条后先取空队列）；
    - 不编码 mask 全空的第三路相机，指令只留 80 个 token（前缀 968 → 592，CPU 上校验动作最大差 0.005）；
    - 请求线程池 512；显存上限 24 GiB；阶段计时器关闭；
    - 可选 `--compile vision,lm,expert`。
  - 不绑核，OMP 默认不设（可用 `--omp N`）。
- **每个分组用自己的 yaml 副本**：放在 `/data/mw_finetune/eval/cs_rec_pi05/definitions/<arm>/config.yaml`，evidence_dir 改到 `<root>/evidence/<arm>`。原件指向 R16 测试共用的证据目录，**不能往那里写**。

## 文件（全在 `/home/weiland/projects/openpi/exp/cluster_sim/`，已推送 origin Ziyang）

| 文件 | 作用 |
|---|---|
| `DESIGN.md` | owner 决议 + 实现说明 + 「10-21 夜里已确认」 |
| `R16_SERVING_MAP.md` | 代理画的 R16 π0.5 服务路径（file:line） |
| `LEVELS.json` | 每个任务的档位和各档成功数 |
| `record_group.py` | 生成录制分组 `/data/mw_finetune/eval/cs_rec_pi05`（实体在 /nvme，已建，500 局；R16 排程 dry-run 40 个阶段、500 局） |
| `record_run.py` + `record_resident.py` | 照常用 Q3 runner，常驻改从 record_resident 启动，收到的每一帧写进 `$CLUSTER_SIM_RAW/c<pid>_<n>.msgpack` |
| `record.py split` | 按 episode_start…episode_end 切成 `ep_<task>_<init>.msgpack`（开头补 select_bundle） |
| `run_record.sh` | 第 1 步一键：录制 + 切分（日志 `/nvme/weiland/openpi/cluster_sim/logs/record_run.log`） |
| `replay.py` | 回放端（`--episodes K` 只放前 K 局，用于调优） |
| `serve.py` / `serve_entry.py` | 启动器 / 服务进程补丁（`CLUSTER_SIM_GPU` 可改可见卡，CPU 测试时设为空） |
| `sweep.py` | 一种模式：起服务 → 预热 20 局（不计入）→ 各 N 回放 → SIGINT 停（启动器转 SIGTERM）→ `analyze.py` 写 `RESULTS.json` |
| `tune.py` + `tune_pure.json` / `tune_cache.json` | 第 2 步：候选配置逐个短跑（N=5、40，各 60 局），汇总 `TUNE.json` |

commits：847ce11a（设计）、f78b9a5f（主体）、408e33fc、49fd5e53、974f07b3、cdfff73e、93a09b80、46de5d40、b9c9664a、5758401e（本文件）、962065e3（LEVELS 和调参候选原先被 .gitignore 的 *.json 挡住没进库，已强制加入；两边各加一个 CUDA graph 候选 `--compile-mode reduce-overhead`）。

## 现状（10-23 06:xx）：实验完成

- 录制、调优、正式回放全部完成；结果和结论在 DESIGN.md「正式实验」，表 `runs/REPORT.md`，图 `runs/report.png`（`/nvme/weiland/openpi/cluster_sim/runs/`）。
- 纯推理 04:08、完整系统 05:54 跑完；两边无失败请求；完整系统各 N 命中率 78.3%（与录制一致）。
- 10-23 06:1x 已把 weilandserver 交还看守会话 [75fc7a]（4090 无本会话进程）。

## 4090 空出后的步骤（照做）

1. **录制**：`bash exp/cluster_sim/run_record.sh`。
   - 它会用 Q3 runner 起 3 个 π0.5 常驻（23150–23152，驱动 23159），worker 走中央调度器。和看守会话 `q3_port_1020.build` 生成的 R16 launch 脚本同一环境前缀、同一 runner（10-22 15:3x 核对过，之后 R16/Q3 代码没改）。
   - 跑完自动切分到 `/nvme/weiland/openpi/cluster_sim/library`。
   - 看 `library/SUMMARY.json` 是否有 500 局。
2. **调优**（两边各自）：
   - `PYTHONPATH=.:src .venv/bin/python -m exp.cluster_sim.tune --mode pure --library /nvme/weiland/openpi/cluster_sim/library --out /nvme/weiland/openpi/cluster_sim/tune/pure --configs exp/cluster_sim/tune_pure.json`
   - cache 同理，用 `tune_cache.json`。
   - 同时用 `~/.local/bin/py-spy record/dump --pid <服务 pid>` 看瓶颈：完整系统重点看 CPU 方法查询、插件决策日志、证据写入的占比；纯推理重点看第 2、3 段。
   - 若埋点占比大，可以考虑两边都去掉：`R14Knob._diagnostics` 里的 `original_corrector_audit` 会重复算一遍距离，约 2 ms/次，只用于写日志。
   - 按结果为每边定一套参数，写进 DESIGN.md。
3. **正式实验**：
   - `PYTHONPATH=.:src .venv/bin/python -m exp.cluster_sim.sweep --mode pure --run /nvme/weiland/openpi/cluster_sim/runs/pure --library /nvme/weiland/openpi/cluster_sim/library <定好的参数>`
   - cache 同理，run 目录 `runs/cache`。
   - 默认 N=5..100、500 局、种子 0。预计纯推理约 4–5 h，完整系统约 2–3 h。
   - 结果在各 run 目录的 `RESULTS.json`（每个 N：requests_per_s、latency_ms mean/p50/p90/p99、total_s）。
4. 汇报 owner，更新 DESIGN.md 和本文件。

## 粗估（供判断瓶颈，未在 GPU 上实测）

- **纯推理**：每条请求要跑第 2 段（Gemma-2B，592 token）约 26 ms，上限约 30–38 条/秒，大约 N≈8–10 就饱和。
- **完整系统**：大多数请求只跑第 1 段（SigLIP 两路相机）约 3.5 ms，加上 CPU 方法查询约 5–8 ms（可并行）。
- 生产日志里查询 30–65 ms 是三个常驻挤在绑定的 32 核上造成的，不代表本实验的情况。

## 踩过的坑（别再犯）

- 服务进程不理 SIGINT（批处理线程不是守护线程）。要按 PID 发 SIGTERM，启动器已经这样处理。
- 合成测试的 episode_start 里 `__task__` 必须是库任务表里的真实 LIBERO 指令，否则插件报 `task ... is not in the pi05_l10 library task map`。
- 启动器探测端口时，server.log 会出几条 `EOFError: connection closed while reading HTTP request line`，无害。
- 只按 PID 杀进程，不用 pkill -f；不用 git stash；提交只加 `exp/cluster_sim` 和 handoff 这些自己的路径。仓库里有看守会话的链在写的 JSON，用 `git commit -- <paths>`。
- 不碰在跑批次和 h100-temp；/data 不新写（`/data/mw_finetune/eval` 本身是指向 /nvme 的软链，cs_rec_pi05 实体在 /nvme）。
