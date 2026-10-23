# 集群部署模拟实验：会话交接（「openpi新实验」会话专用）

compact 后先读本文件，再读 `DESIGN.md`（owner 决议 + 调优过程 + 正式结果，权威）。服务端代码地图见 `R16_SERVING_MAP.md`。
共用的 `logs/session_handoff.md` 由看守会话维护；本实验只在那里有一块「10-21 22:xx 新实验」摘要。

## 我是谁、管什么

- 本会话（listing 里叫 `cluster_sim handoff doc [850a45]`）负责 **π0.5 · LIBERO-10 集群部署模拟实验**（owner 10-21 晚交代）。
- 正式批次（H100、pod、4090 队列）的看守是兄弟会话 `[75fc7a]`。它的名字很长，就是 owner 那句「我们要做一个新实验，是模拟集群部署实验，我现在和你讨论设计…」，SendMessage 的 `to` 用 ListAgents 里那一行的全名加 ` [75fc7a]`。我不碰它的批次和端口（23192/23193/23197/23198/23199、23117/23118）。
- 本会话 10-22 21:3x 以 bypass 权限重开（`claude --bg --resume fcd51a3d-4543-4bf6-a842-38214526e7b5 --dangerously-skip-permissions`）；job 临时目录是 `$CLAUDE_JOB_DIR/tmp`（旧的 `jobs/fcd51a3d` 已删）。

## ⛔ 现在在跑什么、我要接着做什么（10-23 14:3x 写）

4090 借用中：看守会话同意借到 **10-23 19:00 左右**，期间它不排 4090、不碰端口 8000。跑完必须给 `[75fc7a]` 发消息交还（写清：4090 空闲时刻、无本会话进程、端口 8000 空）。

后台链（`setsid` 脱离会话，compact/重启不受影响），依次：
1. `runs/cache_seg`：完整系统 + 长执行段（重跑，修好服务端后），14:19 起，约 1.3 h。
2. `runs/cache_full`：缓存系统、不开长执行段、**不减步**（未命中 K=10），保留旋钮档位，约 1.6 h。
   - 1、2 由 `run_formal.sh`（PID 1345950，`MODES="cache_seg cache_full"`）跑，日志 `/nvme/weiland/openpi/cluster_sim/logs/formal_cache_seg.log`、`formal_cache_full.log`。
3. `runs/cache`：完整系统重跑。等 PID 1345950 退出后，链 PID 1652783 先把旧 `runs/cache` 改名 `runs/cache_v1_rebuild`，再 `MODES=cache` 跑，日志 `logs/formal_v2_cache.out`、`formal_cache.log`。预计 **18:40 前后**全部结束。

查进度：`ls /nvme/weiland/openpi/cluster_sim/runs/*/replay/`、`pgrep -af "run_formal|exp.cluster_sim"`、`nvidia-smi`。每个 run 结束会写 `RESULTS.json`。

全部跑完后照做：
1. 每组核对：没有失败请求（`grep -c "batch failed\|RuntimeError" runs/<x>/server/server.log`）；每个 N 都是 500 局；`stages_*.json` 里 `control_s.factory` 的次数（池化后只有开头几十到一百多次重建）。
2. 新旧对比：`runs/cache_seg` vs `runs/cache_seg_v1_rebuild`、`runs/cache` vs `runs/cache_v1_rebuild`，看高负载段（N≥40）吞吐和延迟变化。
3. 命中率：按各 N 回放时间窗统计 `/data/mw_finetune/eval/cs_rec_pi05/runs/*/server_8000/decisions_*.jsonl` 里 `ev=dec` 的 `hit`（以前：不跳请求 78.3%，长执行段 81.4%）。
4. `PYTHONPATH=.:src .venv/bin/python -m exp.cluster_sim.report --runs /nvme/weiland/openpi/cluster_sim/runs`，生成 `runs/REPORT.md` 和 `runs/report.png`，四条线：纯推理、缓存系统全步、完整系统、完整系统 + 长执行段。亲自看图，再用 SendUserFile 发给 owner。
5. 把结果写进 DESIGN.md（替换「加测结果」里旧的长执行段数字，注明旧数字来自重建缺陷版本），更新本文件。
6. 给 `[75fc7a]` 发消息交还 4090。
7. 用白话汇报 owner：三组倍数（以 500 局用时或稳态吞吐比）、延迟、和 IR 的关系、修了什么。

## owner 定下的实验内容和后来的决定

- 原始需求（全文见 DESIGN.md）：
  - 对比全步纯推理（K=10，无缓存）和完整系统（成功过滤库 · 500 级库 · Q3 档位 · 减步 K=1 · 修正器 + 停滞门 · 不开长执行段）。
  - 4090 单份模型，显存 ≤24 GB，可重写服务。两边推理代码相同、享受同样的优化，各自调到最优；正式实验每边一套参数，只变 N。
  - 录一遍完整系统的 500 局请求，两边回放同一份；种子打乱，N 个槽位依次取，冷启动错开；请求间隔 = 执行 5 步（250 ms）+ 2 ms。
  - 记录服务器端每条请求的延迟、每秒请求数、每个 N 放完 500 局的总时间。N = 5,10,20,…,100。
- 10-22 起的追加：
  - ⛔ **不换精度**（不做 FP8）。
  - CPU 线程不限：每个 BLAS 调用开 2–8 个线程、靠几百个请求线程并行，是实测最快的；要向 owner 讲清楚这不是单线程。
  - astra 可以用。
  - 「你是专业的 serving 工程师，仔细调整」：要按瓶颈调，不要盲扫格子。
  - 「原生的系统 serving 根本不会有你的那一堆问题，你要调整好了」：凡是我的服务实现引入、原生部署不会有的开销，都必须修掉。
- 10-23 加测：
  - `cache_seg`：减步 + 长执行段，在回放端发一条、跳一条（`--stride 2`），间隔 2×250+2 = 502 ms，推理请求减半。
  - `cache_full`：不开长执行段、不减步的缓存系统，保留现有旋钮档位（owner 选的），未命中 K=10（`--miss-steps 10`）。
  - owner 只要跑这两组；我曾擅自加了 `cache` 重跑被 owner 质疑。之后 owner 说「你觉得数据会改变就重跑」，我判断会变（重建缺陷让事件循环约 6–8% 时间被堵），所以排在最后重跑。
- owner 要求回答问题时先答再做；性能数字没测过不当事实说（memory 已记）。

## 正式参数（`run_formal.sh`）

- 共同：`--graphs --compile all --compile-mode default --fuse23`。
- 纯推理：`--batch 1:16,2:16,3:16 --omp 4`。
- 完整系统：`--batch 1:32,2:16,3:16 --priority 1:-3,3:0 --pipe1 --omp 4 --dot --lean`。
- `cache_seg` = 完整系统 + `--stride 2`；`cache_full` = 完整系统 + `--miss-steps 10`。
- sweep：`--robots 5,10,20,30,40,50,60,70,80,90,100 --seed 0 --warmup 40`，500 局。

## 服务端做了什么（`serve_entry.py`，编号段见文件头）

1. 请求计时。
2. 每段批上限/等待；`--units` 分组；`--pipe1` 第 1 段双缓冲；`--priority` 流优先级。另有每批时间线 `batches_<pid>.csv` 和第 1 段分阶段计时。
3. 前缀去掉空相机，指令截到 80 token。
4. 请求线程池 512；**连接栈按分组池化**（换分组复用已建好的拦截器 + 插件会话，不再重建；控制帧仍在事件循环里，原生方式）；记录重建计时 `control_s`。
5. 显存上限；不做显式 `gc.collect`，第一条推理时 `gc.freeze`。
6. 整段 CUDA graph：第 1 段从拼好的请求字典起录图，合并后段把 Gemma 前缀和整个去噪循环录成一张图（按步数和批大小）；inductor default；启动时用替身请求预录（`precapture`），首次编译约 9–18 min，之后走 `/nvme/weiland/openpi/cluster_sim/compile_cache`。
7. 完整系统：缓存键整批算、一次拷回；去掉插件里的整卡同步。
8. `--fuse23`：第 2 段并进第 3 段批（占位 Stage2Output）。
9. `--lean`：跳过只用于日志的修正器审计。
10. `--dot`：拟合方法的大矩阵换成 `@` 走 `np.dot`（逐位相同、放 GIL）。
11. 锁页缓冲区拼批（两套交替）、整段图调用。
12. `--miss-steps K`：未命中步数覆盖。

另外：拟合方法的 share memo 按方法缓存一次（复制更快）。

## 结果（到 14:3x 为止）

- **纯推理**（`runs/pure`，有效，不受换分组问题影响，控制帧 0.1 ms）：饱和 41.6 条/秒（稳态 42.5 ≈ 上限 44 的 96%），500 局约 667 s；N=5 中位 52 ms。
- **完整系统 v1**（重建缺陷版本，将被重跑替换）：饱和约 88–90 条/秒（稳态 91–96），500 局约 310–320 s；命中率 78.3%。
- **长执行段 v1**（重建缺陷版本，`runs/cache_seg_v1_rebuild`）：N=100 时 500 局 181 s，相对纯推理 3.68 倍；命中率 81.4%；同等负载下吞吐比完整系统低约 10%。owner 由此发现服务端问题。
- **池化验证**（`tune/cache5/{seg_pool,cache_pool}`，短跑）：同等负载（长执行段 N=100 vs 完整系统 N=50）稳态吞吐 81.0 vs 83.9，基本对齐。延迟差约 2 倍是饱和闭环的必然：L = N/C − Z。
- **IR 回答**（owner 问过）：
  - 这套配置的正式 IR = 0.120（含长执行段，成功率 93.6%；同步数纯推理参照 0.305），来自 `q3_port_1020/RESULTS_1022.json`。
  - 正式价格反推：看图 0.152、K=1 未命中 0.454（回代各档吻合）。
  - 本实验不开长执行段：0.152 + 0.217 × 0.454 ≈ 0.25，即 4.0×；用服务代码 batch-1 实测是 0.22，即 4.5×。
  - 按服务批大小重算：0.31，即 3.25×；实测 2.1–2.2×，因为完整系统卡在请求端 CPU，只到其 GPU 上限的 64%。
- **回放偏差回答**（owner 问过）：纯推理回放的是完整系统的轨迹。每条请求计算量与内容无关，所以吞吐和延迟无偏。只有 500 局用时受每局请求数影响：纯推理自己的轨迹 27,981 条，录制 27,747 条，差 0.8%。

## 文件（`/home/weiland/projects/openpi/exp/cluster_sim/`，已推送 origin Ziyang）

| 文件 | 作用 |
|---|---|
| `DESIGN.md` | 决议、调优过程（「10-22 晚」「10-22 夜」）、正式结果、加测 |
| `serve.py` / `serve_entry.py` | 启动器（全部选项见 `--help`）/ 服务进程补丁 |
| `replay.py` | 回放端（`--stride`、`--episodes`、`--interval`） |
| `sweep.py` | 起服务 → 预热 → 各 N 回放 → 停 → analyze（`--episodes N:K` 可按 N 指定局数） |
| `analyze.py` | RESULTS.json：吞吐、稳态吞吐、延迟分位、GPU 忙碌占比、各段批统计 |
| `report.py` | 汇总表 + 图 |
| `run_formal.sh` | 正式回放（`MODES` 可选 pure / cache / cache_seg / cache_full） |
| `tune.py`、`tune_*.json` | 调参扫格子 |
| `bench_stages.py`、`check_graphs.py` | 单段计时、整段图与 eager 对比（K=10 动作差 0.0024，K=1 差 0.014） |
| `record*.py`、`run_record.sh` | 录制（已完成，库在 `/nvme/weiland/openpi/cluster_sim/library`） |

数据：`/nvme/weiland/openpi/cluster_sim/{library,runs,tune,diag,bench,logs}`。录制分组 `/data/mw_finetune/eval/cs_rec_pi05`（实体在 /nvme），服务时决策日志写在它的 `runs/*/server_8000/`。

## 踩过的坑（别再犯）

- 服务进程不理 SIGINT。停服务要对 `exp.cluster_sim.serve` 启动器发 SIGTERM（它转发给进程组）。对 sweep 发 SIGTERM 不会跑它的 finally，服务会被留下，要再按 PID 停启动器。
- `until ... ! pgrep -f "<模式>"` 会匹配到等待循环自己的命令行，永远等不完；等待条件只看文件。
- `grep -c` 没有匹配时输出 0 且返回 1，`$(grep -c ... || echo 0)` 会得到两个 0。
- 用 Python `str.replace` 改文件头说明时，同样的文字常常也出现在代码注释里，会被一起替换进代码。替换前先确认只出现一次。
- `exec` 替换服务端函数时要用模块自己的 `__dict__` 作 globals：它会写模块全局 `_current_bundle`。
- 删数据前先看清：曾把跑了 77 分钟、快完成的 `runs/cache` 重跑直接删了。
- 新会话的 job 临时目录会变，`prof.sh` 这类脚本要重建（或放进仓库）。
- 只按 PID 杀进程，不用 pkill；不用 git stash；提交只带自己的路径（`git commit -- <paths>`），作者 LinZiyang666，不加 AI 署名。
- 不碰在跑批次和 h100-temp；/data 不新写。
- 服务启动约 5–7 分钟（读库 14 GB + 预录图）。完整系统 sweep 里预热 40 局。
- server.log 里启动时有 3 个 EOFError traceback，是端口探测，无害。
