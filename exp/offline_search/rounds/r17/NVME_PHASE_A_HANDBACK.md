# /nvme 迁移 A 阶段交回（10-21 00:1x）

做了什么：照 NVME_MIGRATION.md §2/§4A1 改代码，让本机新建的目录都把实体放在 /nvme，原来的 /data 路径只留一个软链。没有移动或复制任何已有数据，没有起实验进程，没有 commit。tether 只用 `exec` 做了只读检查：H100、A100、h100-temp、两个 pod 上都没有 `/nvme/weiland/openpi`，所以这些机器上的行为不变。

## 1. 改了哪些文件

**辅助函数**：`exp/r17_integration/ops/immutable_release.py`
- `nvme_path(p)`（:17）：本机存在 `/nvme/weiland/openpi`，且 p 在 `/data/mw_finetune` 下时，返回 `/nvme/weiland/openpi/<相对路径>`；其它情况原样返回 p。
- `nvme_dir(p, exist_ok=False)`（:26）：在 /nvme 上建实体目录，在 p 处建目录软链，返回的仍是 p（也就是 /data 字符串）。p 已经存在，或者机器上没有 /nvme，就退回普通的 `mkdir(parents=True)`。将来 `/data/mw_finetune/eval` 整体换成软链之后，p 自动就是实体目录，不会再建软链。
- `install()`（:87）：新发布的目录（driver_release 装的 `formal_driver_*`、formal_prestage 和 runtime_fix 装的 `formal_runtime_*`、formal_finalize 装的 `formal_evidence_*`）在 /nvme 上解包，然后 `os.rename` 到 /nvme 的实体位置，原 /data 路径建软链。install 的 lock 文件也放在 /nvme。已经存在的发布目录照旧复用，不复制。如果实体目录已经装好、但软链还没建，就补建软链。

**新建运行目录的地方，统一改成 `nvme_dir`**
- `ops/formal_eval.py:1953`：同时删掉了 :1929-1930 那道检查（见 §2）
- `ops/queue_retry.py:103`（clone）
- `ops/formal_queue.py:303`（同时删掉了 :302 的父目录检查）
- `filter_failed/resume.py:31`
- `filter_failed/build.py:168`
- `filter_failed_knob/build.py:123`
- `filter_failed_knob/formal_a_short.py:73, :105`
- `filter_failed_knob/formal_a_queue.py:149`（R15 cell）、`:171`（R16 组）
- `filter_failed_knob/q3_port_1020/build.py:191`（R15 单元）、`:259`（R16 组）：EVAL 常量不改，组根目录和 group_sha256 里仍是 /data 字符串
- `filter_failed_knob/cache_study_1019/study.py:126`（R16 组）
- `longseg_1017/build.py:107`：`.incomplete_` 改名改为对 /nvme 上的实体目录做，然后删掉旧软链，再新建
- `longseg_1017/pod_copy.py:153`
- `longseg_1017/catalog.py:28`

**本机 worker 目录改到 /nvme**
- `ops/local_workers.py:15`：`ROOT=/nvme/weiland/openpi/qualification/local_workers`
- `ops/worker_storage.py:19`：`LOCAL_RUNS` 跟着改
- `longseg_1017/reset_remote.py:100`、`autoscale_1017/reset_remote.py:100`：local 允许的父目录同时列 /data 旧路径和 /nvme 新路径
- `longseg_1017/vfix.py:236`、`autoscale_1017/reset_zero.py:70`：local 的 run 父目录改为取部署记录里的值，新旧两种路径都能用

**跨盘硬链接**：`longseg_1017/io_budget.py:129-133`。`tree()` 里的 `os.link` 遇到 EXDEV（源在 /data、新 run 在 /nvme）时改用 `shutil.copyfile`。停滞门文件原来就走 `copy()`，没有改动。

**远端地址不再用本机 resolve 后的路径**（软链根目录 resolve 后是 /nvme 路径，但 H100 和 pod 上的副本在 /data 地址）
- `filter_failed/reset.py:185-187`：在 server 节点上改名时，用 `campaign['server_run']` 的父目录，不用 `str(root)`。改之前，软链根目录会让远端命令去 /nvme 路径找，找不到就什么也不做。
- `longseg_1017/vfix.py:218-226`、`autoscale_1017/reset_zero.py:45`：H100 上的根目录取 `server_run` 的父目录，只比较目录名是否一致。

**R15**：`closed_loop/ops/h100/control.py:433, :516, :685`（`_sync` / `collect` / `chain`）和 `assets.py:139`（`build_plan`），把 `.resolve()` 改成 `os.path.abspath`，详见 §3。`rounds/r15/descent/analyze.py:120-122` 原来钉了 assets.py 源码的 sha，已拔掉。

**文档**：`ops/RUN_CONFIGS.md`，改了两处：
- 在 pod GR00T 那一段补充：R2 时显存溢出，续跑 jc1 改成 R1 / 14 GiB。
- MPS 管道路径：新 run 目录 resolve 到 /nvme，路径多 3 字节，run 名最长 46 字符（原来 49）。

**jc1 相关**
- 新增 `filter_failed_knob/commands/{prepare,launch}_fa_ls_gr_f_u_m_jc1.sh`
- `filter_failed/CONTINUATIONS.json` 里由 resume.build 追加了 jc1 一条

**注意：别的代理在同时改同一批文件。** 下面这些 diff 不是我改的，提交时要分开：
- `formal_eval.py` 里的 `SERVER_CORES` / `server_threads`（:672-683、:700-702）
- `vla_workers.py` 里的 `OMP`
- `local_server.py`、`node.py`、`r16/conductor_runner.py`、`common/batch_runtime.py`、`models/groot/provider.py`
- 本次之前就已经改动的那批 JSON（formal_a_1016/*、FORMAL_QUEUE.json、RESULTS.json）

## 2. 拔掉的检查（按规定直接删，没有换成别的校验）
- `formal_eval.py:1929-1930`：resolve 后的父目录不等于 FORMAL_ROOT 就 raise
- `local_workers.py:155-157`：`copy()` 的目的地 resolve 后必须在允许列表里
- `sync_back.py:18`：`atomic_bytes` 只要任一父目录是软链就 raise。只删了父目录这一项，目标文件本身是软链仍然拒绝。
- `formal_queue.py:302`，`queue_retry.py:257-258`（`continued_jobs`）：父目录必须等于 FORMAL_ROOT
- `r15/descent/analyze.py:121`：assets.py 的源码 sha 钉
- 原来就已经拔成 `pass` 的：`immutable_release.verify:37/44`、`formal_prestage.launch_prepared:308`、`evidence_transfer`、`remote_json`、`formal_runtime`、`cache_control`

**没有动、但列出来的同类检查**（不在新批次的写路径上）：
- 只在建库时用：`library_batch.py:15/27/58/141`、`library_staging.py:76/121/248`、`library_init.py:59`、`library_cleanup.py:28/37/48`、`library_salvage.py:116/177`、`library_capture_views.py:31`
- 同步守护进程当前没在用：`incremental_sync.py:54/60/77/202/241/251`
- 下面几处在新布局下不会触发：
  - `sync_back.py:107`：只拒绝目的地本身是软链，assets 和 servers 都是实体目录
  - `formal_prestage.verify_code:46/48`：只在远端执行
  - `worker_storage.py:43/57`：LOCAL_RUNS 已经是 /nvme 上的实体路径
  - `longseg_1017/staging.py:21`
  - `longseg_1017/check.py:101/124`：samefile，不在链路径上
  - r13/r14 的 `runtime_audit` / `release_check`：钉了 control.py 的源码，但都是旧轮次

## 3. R15 单元的处理
- 采用的做法是「远端用原来的 /data 字符串，不取真实路径」。新建的 R15 单元由 q3_port_1020 用 `nvme_dir` 建：/data 下是软链，arms.json 里写的是 /data 字符串（R15 的 emit 不做 resolve，已用现有的 `fa_pi_f_f_o_q_l50/arms.json` 核对过）。
- `assets.build_plan` 和 `control._sync/collect/chain` 改用 abspath 之后，`str(run)` 和 arms.json 里的字符串一致，`remap` 照旧把它映射到 H100 的 `BASE/runs/<name>`；其它 /home、/data 路径照旧走 mirror。
- 旧代码 resolve 之后会把 run 内的文件误当成 mirror 输入，H100 上的输入和输出位置都会错开。
- 对现有的实体 /data 目录，abspath 和 resolve 结果相同。h100q1 链后续每个 cell 新起的进程用新代码，行为不变。
- `legacy_chains`、`chain_lock` 和 registry 一带（:197/:250/:287/:296）没改：R17 的 R15 用 `vla_shared.fleet_lock` 替代了原生 fleet_lock，不写 registry；这些 run 根目录也不在 CHAIN_LOCK 的 glob 范围里。
- 本机的 R15（4090 GR00T）走 `local_server`，两边都 resolve，比较的结果一致。
- 所以按 §4A2 把已经建好、排队中的 8 个 `fa_pi_f_f_o_q_*` 换成软链也是安全的，前提是这份代码已经上线。
- 新风险：新 R15 单元的 run 根目录在 /nvme，`control.py:383` 导出时会把同在 /nvme 上的 fit pkl 硬链接进 `h100_sync/export_*`；以前根目录在 /data，跨盘时走 COPY_FALLBACK 复制。按调研结论，R10Recipe/R15 的 fit 没有 inode stamp，Q3 的 StallModel 是打包在 fit pkl 里的，不是单独的停滞门文件。如果要求完全不碰 /nvme 上的 fit inode，可以把这里改成一律复制。

## 4. 冒烟测试（已清理）
- 在 `/data/mw_finetune/eval/_nvme_smoke_*` 上调用 `nvme_dir`：
  - /data 下是指向 `/nvme/weiland/openpi/eval/_nvme_smoke_*` 的软链。
  - `atomic_bytes` 经过软链父目录写入成功，文件落在 /nvme。
  - 第二次调用报 FileExistsError；加 `exist_ok` 后可以复用。
- 对 `/data/mw_finetune/qualification/_nvme_smoke_rt_*` 调用两次 `install()`：第一次 reused=False，第二次 reused=True；实体在 /nvme，/data 下是软链。第一版代码在复用时会在 /data 上建 lock 文件，已改为 lock 放 /nvme，并重测通过。
- 把 NVME 改成一个不存在的目录，模拟 H100 和 pod：退回普通 mkdir，不建软链。
- 只删除了我自己建的临时目录、软链、lock 文件和空的 `/nvme/weiland/openpi/qualification`。

## 5. jc1（真实命令，36 秒，rc=0）
`R17_H100_NODE=jupyter-ziyang10-02 R17_H100_HOST=149.165.153.233 R17_DRIVER_PORT=23192 .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.resume --source /data/mw_finetune/eval/fa_ls_gr_f_u_m_j --name fa_ls_gr_f_u_m_jc1 --remaining --verify-closed`

**结果**
- `--verify-closed` 通过：SOURCE_CLOSURE.json 为 PASS、all_owned_closed=true，本机 driver 987934 和 a100/timan1/timan107/pod 上的进程都已退出。源批次 RECEIPT 自己记的是 PARTIAL、all_owned_closed=false。
- 保留 1,307 局，补跑 88,693 / 90,000 局。
- prepare 时会按原生流程再做一次关闭证明，然后刷新剩余集合（`refresh_after_source_closure=true`）。
- 源目录只读，00:05 之后没有任何文件被改动。

**位置**：`/data/mw_finetune/eval/fa_ls_gr_f_u_m_jc1` 是指向 `/nvme/weiland/openpi/eval/fa_ls_gr_f_u_m_jc1` 的软链。

**campaign.json 改动**（改前的版本另存为 `*.before_r1_1792559533771332631`，FFK_DEFINITION.json 同样另存了一份）

| 字段 | 改前 | 改后 |
|---|---|---|
| replicas | 2 | 1 |
| replica_startup_width | 1 | 1 |
| memory_budget_gib | 14.0 | 14.0 |
| backend_ports | [23475, 23476] | [23475] |
| ports | [23474, 23475, 23476, 23192] | [23474, 23475, 23192] |
| cache_worker_orchestration.replicas | 2 | 1（只是记录，沿用 pod_copy 的做法） |

- 新增 `pod_replicas_1021` 记录改前、改后的值和原因。
- 不变的字段：public_port 23474、endpoint 149.165.153.233:23474、driver_port 23192、execution_host jupyter-ziyang10-02、mps null。
- 不变的路径字符串：local_assets、assets、server_run 都是 `/data/mw_finetune/eval/fa_ls_gr_f_u_m_jc1/...`；runtime 是 `formal_runtime_abfcde…`。
- 因为 execution_host、driver_port、public_port 都已经和 pod 别名一致，`execution_host()` 在 stage 时不会再改端口。
- FFK_DEFINITION.json 改为 replicas=1，episodes=88693。

**命令**：`filter_failed_knob/commands/prepare_fa_ls_gr_f_u_m_jc1.sh`（filter_failed.stage）和 `launch_fa_ls_gr_f_u_m_jc1.sh`（filter_failed_knob.launch），照 longseg_1017 下 `_j` 的两个脚本写，带同样的 pod 别名环境。没有执行 prepare，也没有 launch。

**prepare 前要知道**
- `assets` 和以前的 jc/mc 续跑一样指向自己的目录。冻结的 YAML 里引用的源 assets 地址，在 pod 上 j 跑过时已经装好了；如果 pod 重启过，需要先重新 setup。
- worker payload 里覆盖了 `local_workers.py`，内容变了，所以 payload identity 会变，各 worker 节点会重新装一份小 payload。
- driver 覆盖的文件我都没改。

## 6. 还没覆盖、仍会写 /data 的地方
- `rounds/r16/conductor_runner.py:18,182`：LOCAL_ISLAND（agent 日志，lease 每 15 秒写一次）。按 §4A3 等 runner 退出后换软链；这个文件别的代理正在改。
- 在跑的 5 个批次和已经建在 /data 上的排队目录：属于存量，按 §4A2、§4B 处理。
- 已有的 /data 发布目录被复用时照旧在原地读。冻结 runtime 里的旧代码只要用未 resolve 的 /data 路径写盘，仍会碰到旧的 `sync_back` 父目录软链检查。不过 launcher、stage 和 finalizer 都会把目录先 resolve，evidence runtime 也会覆盖成新版 sync_back，所以理论上碰不到。
- `longseg_1017/recalibrate.py:299`：给 Q3 alias 建硬链接（跨盘会 EXDEV，而且是停滞门文件）。只在重新标定时用，没改。
- 只在建库时用：`library_batch.py:112,156`、`library_capture_views.py:46` 的 os.link。
- 当前链不用：
  - `incremental_sync.py:23` 的 ROOT
  - `cache_knob.py:149,270`（`.resolve()` 会把 /nvme 路径写进 YAML）
  - `cache_lane.py:23`
- 默认输出在 /data（手动跑才会写）：`perf_recorder.py:768` 和 `dp_wait_analysis.py:208`，默认目录是 `/data/mw_finetune/perf`。
- 这几处会把软链本身挪走，实体留在 /nvme：
  - `longseg_1017/trash.py`
  - `vfix.retry` 和 `reset_zero.reset` 的本机 backup：必须给 resolve 后的同级路径，也就是 /nvme 上的路径
- 旧的 local worker run（/data 路径）收尾清理时，路径对不上，结果是 RETAINED：数据保留，不删。
- 新的 local worker 目录第一次使用时：
  - 会重建 env311 venv。
  - 会从 /home 复制 LIBERO 依赖（约 609 MB）和 assets。
  - 不读 /data 上的旧副本。
