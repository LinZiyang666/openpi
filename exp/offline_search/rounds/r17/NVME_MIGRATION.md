# /data → /nvme 迁移调研（10-20 23:2x–23:4x 只读实测）

范围：owner 10-19/10-20 定的「R17 的库、所有结果和之后的新写入都放 /nvme/weiland/openpi/，原路径留软链；data 不新写；在跑批次不停、跑完再迁」。本文件只做调研，没有改代码、没有动数据、没有起进程。代码路径相对 `/home/weiland/projects/openpi/`。

## 0. 结论

1. 现在还在往 /data 写的有三类：①5 个在跑批次的 driver/日志；②本机 R16 worker island（agent 日志、每 15 s 一次的 lease）；③每次 prepare/收尾新装的 `formal_driver_* / formal_runtime_* / formal_evidence_*` 发布目录。另外，排队批次的目录已经建在 /data 上，prepare 时会原地写。
2. 新批次落 /nvme 的最小改法：**实体目录建在 /nvme，/data 原路径只留一个软链**。不能直接把 `FORMAL_ROOT`/`EVAL` 常量改成 /nvme，因为 campaign 里的 `assets`、`local_assets`、`server_run`、`runtime` 以及 YAML 里的 stall/fit/store 路径同时也是 H100/pod 上的远端路径：H100 没有 /nvme，pod 只绑定了 /data。具体要改的是：6 处建目录的地方、`immutable_release.install` 一处、拔掉 4 道挡软链或跨盘的检查；R15/R16（q3_port_1020、cache_study_1019）的根目录只在本机用，直接改常量即可（见 §2、§4A）。
3. 停滞门 stamp 只存在**服务端进程的内存**里，按 resolve 之后的路径做键，不写进 build/prepare/launch 的任何文件。「复制 → 原子换成软链 → 原目录改名保留」只会让服务端换个路径重新校验一次，**不会**报 `verified stall artifact changed`。会触发报错的只有三种操作：给同一个 inode 加或删硬链接、跨盘 `mv`（等于复制再 unlink）、chmod。现在本机没有任何 R17 服务端进程（在跑的 R17 服务端都在 pod 或 H100 上，读的是它们自己那份副本），所以**这道门不是迁移障碍，不必拔**（§3）。
4. 早上暂停的那次复制停在 eval 的第 ~376/673 项（`ffk1015_cosmos_libero_reduced_off_u`）；库和 r17_frozen 已经复制完。**不要 SIGCONT**：恢复之后它还是 120 MB/s 限速，两个 governor 会继续按 IO 压力暂停/恢复，`mirror_nvme.sh` 还会在初次复制结束后开始每小时全量读一遍 /data。正确做法是全部杀掉，找窗口一次不限速重跑（§4C）。
5. 库：/nvme 上的副本 10-20 04:06 核对过（1,020,332 个文件，300 个 sha256 一致），/data 上库的子目录最后改动在 10-15，本机没有进程在用它 → **现在就可以原子换成软链**（§4C1）。
6. H100、A100、h100-temp 上的 /data 都是各自的 Jetstream 云盘，不是本机这块 sda，不在本次范围内（§4D）。
7. 顺带发现两个问题，与迁移无关，但很快会挡住队列（§5）：①`chain_ffk_stop` 对 R15/R16 单元永远读不到 RECEIPT，h100q1 跑完 `fa_pi_f_u_o_q` 就会 STOPPED，不会继续跑 `fa_pi_f_f_o_q`；②pod 上的 GR00T 已经累计 757 条 error。

## 1. 谁在往 /data 写（23:40 实测：ps + /proc/*/fd + /proc/*/cwd）

### 1.1 在跑批次

| 批次 | 白话 | 服务端 | 本机 /data 写入 | 进度（23:39） | 预计结束 |
|---|---|---|---|---|---|
| fa_ls_v09_1_j4 | DP · LIBERO · 全步 · 缓存变体 | jupyter-ziyang10 | driver 1658438：`driver.log`，`driver/{journal,per_step(12.7 GB),attempts,dispatch,worker_dispatch}.jsonl`；cwd 在 `/data/mw_finetune/qualification/formal_driver_e30ca6…` | 31,644/33,650 | ~00:20 |
| fa_ls_gr_f_u_m_j | 原库 · GR00T · MetaWorld · 全步 · 旋钮下降（FFK_DEFINITION 里 arm=original；handoff 22:5x 写成了「成功过滤库」） | jupyter-ziyang10-02 | driver 987934，写入同上 | 2,500/90,000，另有 757 条 error | 13–30 h |
| fa_ls_v03_0 | GR00T · MetaWorld · 减步 · 缓存变体 | H100 | driver 1839769（23:24 起） | 2,103/24,406 | ~02:30–03:00 |
| fa_cs_gr_r | 原库 · GR00T · LIBERO · 减步 · 缓存部件研究（R16） | 4090 | runner 1789300（23:25 由链 g4090an 重起）写组根目录下的 journal、dispatch、per_step/、runs/、scheduler/、agent_lease；4 个 resident 写 `runs/resident_*/server_*.log` | 1,603/16,000 | 数小时 |
| fa_pi_f_u_o_q_{l50…s500}（8 cell） | 原库 · π0.5 · LIBERO · 全步 · 旋钮关（Q3 门，R15） | H100 | 链 h100q1 23:33 起跑：本机 driver 写各 cell 根目录；`h100_sync/export_*` 会把跨盘输入整份复制进 /data 的 run 根目录（prep.log 里有 `COPY_FALLBACK … key_v0.npy (817889408 bytes)`） | 刚开始，共 4,000 局 | 数小时 |

fa_co_f_f_lc1（22/22）已在 23:1x 结束。

### 1.2 本机 R16 worker island
`/data/mw_finetune/qualification/r16_local_island`：本机 agent 1792055 和 ~46 个 worker 的 cwd 都在这里。持续写入有三项：`r16_agents/<sha>/agent.log`、`lease.json`（每 15 s 一次 os.replace）、`scheduler/campaign.json`。另外每次起 runner 时，`legacy.install` 都会往 island 里写 conductor.py 和 sitecustomize。路径来源：`exp/offline_search/rounds/r16/conductor_runner.py:18`（LOCAL_ISLAND）、`:182`（`islands=dict(..., a100=LOCAL_ISLAND, local=LOCAL_ISLAND)`）。

### 1.3 每次 prepare/收尾新装的发布目录（每个约 10 MB，每批 1–3 个；23:02、23:19、23:21 都新装过）
- `formal_driver_<id>`：`exp/r17_integration/ops/driver_release.py:72`（prepare 时装）
- `formal_runtime_<pin>`：`exp/r17_integration/ops/formal_prestage.py:102`、`exp/offline_search/rounds/r17/longseg_1017/runtime_fix.py:114`。H100 批次在本机也装一份镜像（formal_prestage.py:121-128）。
- `formal_evidence_<id>`：`exp/r17_integration/ops/formal_finalize.py:401`（收尾时装）
- 本机 R17 worker（只有 4090 服务端的批次才用）跑在 `/data/mw_finetune/qualification/local_workers/runs/<run>_local`：`ops/local_workers.py:15`、`ops/worker_storage.py:19`。最近一次是 10-20 19:00，现在没有在跑。

### 1.4 排队、未开始的批次（目录已在 /data；prepare 时 `filter_failed/stage.py:95,118` 对 `--run` 做 resolve，然后原地写）
- H100 链 h100m26（chain_ffk，单项失败后继续）：fa_ls_v03_1 · fa_ls_gr_f_f_m · fa_ls_v05_1_rz（prepare 是在 `fa_ls_v05_1` 上做 vfix retry，用到 /data 上的 `.before_retry_1430` 备份）· fa_ls_v05_2_rz · fa_ls_fw_f_u_m · fa_ls_v02_b · fa_ls_fw_f_f_l_l / fa_ls_fw_f_u_l_l（目录还不存在，prepare 时由 `longseg_1017.workflow` 新建）· fa_ls_v08_1 · fa_ls_fw_r_u_m · fa_ls_fw_f_f_m。fa_ls_gr_f_u_m、fw_r_f_m、v05_3/4/0 已被 pod 或 h100-temp 领走，有 claim 文件，prepare 会 exit 3 跳过。
- 链 h100m27：fa_ls_fw_r_f_mc2（续跑目录已建好）。
- 链 h100q1：fa_pi_f_f_o_q_*（R15，8 个 cell，已建好）。
- 4090 链 g4090am（23:26 因 hold 文件 queue done，等协调者重起）：R16 组 fa_pi_r_u_o_q、fa_pi_r_f_o_q、fa_gr_r_u_o_q、fa_gr_r_f_o_q、fa_cs_pi_rg、fa_cs_gr_rg（目录里只有 group.json 和 ROOT.json；fit 和停滞门已经在 /nvme/weiland/openpi/r17_q3_port_1020）；R15 本机 GR00T 单元 fa_gr_f_u_o_q_*、fa_gr_f_f_o_q_*。
- 已建好的目录每个 1–2 GB（v03_1 1.9 GB，fw_r_f_mc2 1.4 GB），绝大部分是 runtime.tar、workers.tar 和 assets。

### 1.5 链、调度器和其它后台进程
- 链脚本 `openpi_ext/scratch/r17_ops/ff/chain_ffk.run.sh:9`、`chain_ffk_stop.run.sh:15`：`E=/data/mw_finetune/eval`，只读 RECEIPT 和 journal；链日志 `r17_ops/mw/chain.log` 和 prep/run 日志都在 /home。
- `$CLAUDE_JOB_DIR/tmp/t107_mem_guard.sh:9`：扫描 `/data/mw_finetune/eval/*/WORKER_NODE_STATE.json`，触发时会改写这个文件。
- 中央调度器在 A100 上（/home/exouser/scheduler），不写本机 /data。crontab 里没有与 /data 相关的任务。
- 复制残留：copy_to_nvme.sh 3296379（在等待）、rsync 3726105/3726106/3729764（状态 T）、copy_governor 3343372/3951564（状态 T）、**mirror_nvme.sh 3950599**（在等复制结束，结束后每小时全量读 /data，必须停）。
- `mw_speed2.sh` 411978 还在监控已经不存在的 csfw_k3w4，不读也不写 /data，可以停。`h100_disk_guard2.sh` 只操作 H100。

## 2. 代码里写死的 /data（只列会产生新写入的）

| 位置 | 内容 | 用途 | 改法 |
|---|---|---|---|
| `exp/r17_integration/ops/formal_eval.py:28` | `FORMAL_ROOT=/data/mw_finetune/eval` | 本机 run 根目录，同时用于拼远端路径 | 字符串不改 |
| `formal_eval.py:1928-1930` | `directory.resolve().parent != FORMAL_ROOT` → raise | 新建 formal_eval 批次时检查 | **删掉**（软链会 resolve 到 /nvme） |
| `formal_eval.py:1941` | `directory.mkdir(...)` | 新建 run 目录 | 改为 `nvme_dir(directory)` |
| `formal_eval.py:1972-1973` | `assets`、`server_run`、`local_assets` | 远端也用 | 不改 |
| `exp/offline_search/rounds/r17/filter_failed/resume.py:30` | `FORMAL_ROOT/name; root.mkdir` | 续跑新目录（mc2、jc 类） | `nvme_dir` |
| `filter_failed/resume.py:35-36` | assets/server_run/local_assets | 远端也用 | 不改 |
| `exp/offline_search/rounds/r17/filter_failed_knob/common.py:11`、`build.py:122-123` | `EVAL/name; root.mkdir` | ffk 新批次 | `nvme_dir` |
| `exp/offline_search/rounds/r17/longseg_1017/audit.py:16`、`build.py:97,104` | `EVAL/name; root.mkdir` | longseg 新批次和各档（如 fw_f_f_l_l） | `nvme_dir` |
| `longseg_1017/pod_copy.py:121,152` | `EVAL/name; root.mkdir()` | pod 副本 | `nvme_dir` |
| `longseg_1017/pod_copy.py:142` | `driver_runtime='/data/…/formal_driver_'+id` | 字符串；stage 会重建 | 不改 |
| `exp/r17_integration/ops/queue_retry.py:102-103` | `FORMAL_ROOT/name; mkdir` | cache_resume 原生 clone | `nvme_dir` |
| `exp/r17_integration/ops/immutable_release.py:60-81` | `install()` 的 stage 用 `mkdtemp(dir=root.parent)`（:73），然后 `os.rename(stage, root)`（:81） | 三类发布目录都走这里（driver_release.py:72、formal_prestage.py:102、formal_finalize.py:401、runtime_fix.py:114） | 本机有 /nvme 时，实体建到 /nvme，原路径留软链（见 §4A1）；H100 上没有 /nvme，保持原行为 |
| `exp/offline_search/rounds/r17/filter_failed_knob/q3_port_1020/build.py:31` | `EVAL=/data/mw_finetune/eval` | R15 单元根目录（:189）和 R16 组根目录（:249-258，root 写进了 group_sha256） | **直接改成** `Path('/nvme/weiland/openpi/eval')`（只在本机用；R15 的 H100 映射要求路径字符串等于 resolve 后的路径） |
| `filter_failed_knob/cache_study_1019/study.py:27` | `EVAL` | R16 缓存研究组的根目录 | 同上，直接改 |
| `exp/offline_search/rounds/r16/conductor_runner.py:18,182` | `LOCAL_ISLAND` 同时给 local 和 a100 用 | 本机 agent/lease/日志 | 照 r15_island 的先例换软链（§4A3），不改代码 |
| `exp/r17_integration/ops/local_workers.py:15`、`worker_storage.py:19` | `/data/…/qualification/local_workers(/runs)` | 本机 R17 worker 的运行目录、payload、env311 | 改成 `/nvme/weiland/openpi/qualification/local_workers`（先复制，runs/ 除外） |
| `longseg_1017/vfix.py:236`、`reset_remote.py:100` | 同上的字面路径 | reset/vfix | 跟着改 |
| `exp/r17_integration/ops/incremental_sync.py:23,251` | `ROOT` + `is_relative_to(ROOT)` | 只在独立的同步守护进程里用，当前没在用 | 暂不改 |
| `exp/r17_integration/ops/cache_knob.py:149` | `str((study.ROOT/path).resolve())` 写进 YAML | 旧的旋钮生成器，当前链不用 | 如果再用它，就去掉 `.resolve()`（否则库换软链之后会把 /nvme 路径写给 H100） |
| 生成的 `commands/*.sh`（longseg build.py:33、pod_copy.py:57、q3 build）和链脚本里的 E= | `--run /data/mw_finetune/eval/<name>` | 经过软链照样能用 | 不改 |

**为什么不直接改常量**：H100 批次 prepare 时，`stage.py:44-50,67-70` 会把同一份 assets 装到 H100 上的 `campaign['assets']` 和 `campaign['local_assets']` 两个地址；新 Q3 决策的 stall_path 是用 root 拼出来的字符串；pod 上 r17x 只把 /tmp/zl_r17/data 绑定成 /data。如果改成 /nvme 字符串，H100 和 pod 都找不到这些文件（H100 上用户是 exouser，没有 /nvme）。

## 3. 软链风险

### 3.1 停滞门（Q3 stall_path），以及同类的 fit/store stamp
- **在哪里记录**：都在服务端进程里，第一次加载缓存配置（`load_stall`）时，记在进程内存中：`exp/r17_integration/search/cache_recipe.py:255-286`，键 `key=(str(root.resolve()), stall_digest, fitted_digest)`（:260），stamp 取 lstat 和 stat 的 dev/ino/size/mtime/ctime/mode（:21-23）。之后同一进程每次重载（也就是「下一阶段」）都会核对（:261-266 → `verified stall artifact changed`）。fit 同理：`cache_recipe.py:142-180`（READY.json、fit.npz 的 ino/size/mtime/ctime，:151 raise）；store 同理：`common/recipe.py:132-152`（:152 raise）。**build、prepare、launch 都不记录 inode 或 ctime**，只记录内容 sha（READY.json 里的 files pin，在 :276-279 按内容核对，复制出来的副本能通过）。
- **代码冻结在哪里**：在每批的 runtime.tar 里（已核对 `formal_runtime_966ab8…`，也就是 v03_1 用的那份，:260 与仓库一致），改仓库不影响已建和已排队的批次。
- **哪些操作会触发报错**：在同一个 resolve 路径上，inode 的元数据被改。具体是：加或删硬链接（ctime 变，10-20 01:41 那次事故就是这样）、跨盘 `mv`（等于复制再 unlink，源 inode 的 nlink 和 ctime 都会变）、chmod/chown、mv 替换同名文件。还有一种：目录被改名、而原路径上暂时没有东西，下次 `_stamp` 时会 FileNotFoundError。
- **哪些操作不会触发**：rsync/cp 只读（/data 挂载是 noatime，读不改元数据）；原子交换（renameat2 RENAME_EXCHANGE），把目录换成指向 /nvme 副本的软链；原目录改名保留（只改目录本身，里面文件的 inode 不动）。换完之后 `root.resolve()` 指向 /nvme，键变了，缓存不命中，服务端从 /nvme 重新校验一次（内容 sha 相同，能通过）。代价是：如果换的是 store（库数组），服务端会多算一次 checksum，内存里再放一份。
- **现在谁持有这些 stamp**：本机没有任何 R17 服务端（ps 里没有 formal_server 进程）。本机只有 fa_cs_gr_r 的 4 个 R16 resident，它们用的是 R10Recipe：fit 在 /home 的 cache_study_1019，库是 /data/libero_cache 下的 pkl，没有 stamp 代码。pod 和 H100 上的服务端读各自本机的副本。排队中的本机服务端（R16 q3 组、R15 GR00T）的 fit 和停滞门都在 /nvme（经 /home 软链）。**所以现在换 eval 下的任何目录都不会触发停滞门**。
- **还没 prepare 的排队批次，整目录搬到 /nvme、原路径留软链，安全吗**：安全，条件有三条：①用复制，不用 mv；②原子交换，原目录改名留着；③做的时候链还没 prepare 到它（chain.log 里还没有 `prepare <name>`），而且 /proc 里没有进程的 fd 或 cwd 在里面。共享 inode 的问题不用担心：复制不加链接，原 inode 一点不动；/nvme 上的副本是独立的 inode（rsync 不加 -H），以后也不会再连带别的批次。
- **库 /data/r17_library 换成软链**：对在跑批次没有影响（本机没有进程读它；pod 和 H100 读自己那份）。对排队批次：prepare 和 staging 读到的是 /nvme 副本，内容相同；YAML 里的 `/data/r17_library/...` 字符串不变，H100 照旧读 H100 自己的库。唯一要注意的是 `cache_knob.py:149` 这类会把 resolve 之后的库路径写进 YAML 的代码（当前不用）。库里的软链都是相对路径（如 `cosmos_libero_libero_10 -> libero/cosmos/libero_10`），rsync -a 原样保留。
- **是否唯一障碍**：不是。stamp 门只挡「改同一个 inode」，而迁移不需要改 inode。如果 owner 仍然要拔：把 `cache_recipe.py:266`、`:151` 和 `common/recipe.py:152` 的 raise 改成 pass；但这只对之后新冻结的 runtime 生效，已建批次要靠 `longseg_1017/runtime_fix.apply` 这种改写归档的方式才能覆盖，对这次迁移没有帮助。

### 3.2 会因为软链或跨盘而出错的点（仍然有效的）

| 位置 | 问题 | 影响 | 最小改法 |
|---|---|---|---|
| `formal_eval.py:1929-1930` | resolve 后的父目录 ≠ FORMAL_ROOT 就 raise | 新建 formal_eval.sh 批次失败 | 删掉 |
| `ops/local_workers.py:155-157` | `destination.resolve()` 必须落在 `(ROOT,/tmp/…,/data/mw_finetune/eval)` 里 | 收尾时把本机 worker 证据拉进 run 目录（`direct_transfer.py:111-114`）会 raise，结果变成 `evidence_incomplete` | 删掉这道检查（或者把 `/nvme/weiland/openpi` 加进 allowed） |
| `ops/sync_back.py:18-19` | `atomic_bytes` 只要路径的任一父目录是软链就 raise | 用未 resolve 的路径写盘时报错，例如 `formal_eval.py:816` 的 `install_snapshot(..., campaign['local_assets'])`；整个 eval 换软链之后，所有用 /data 字符串的写入都会报错 | 删掉 `any(p.is_symlink() for p in path.parents)` |
| `longseg_1017/io_budget.py:127` | `os.link(src,dst)` | 源在 /data、新 run 在 /nvme 时报 EXDEV，pod_copy 和 longseg build 失败 | 失败时退回 `shutil.copy2(src,dst)` |
| `closed_loop/ops/h100/control.py:431`、`assets.py:138` | R15 的 `_sync` 和 `build_plan` 先 resolve run 根目录，再按 `str(run)` 映射到 H100 的 base/runs | 软链根目录下，arms.json 里的 /data 字符串匹配不上，会被当成 mirror 输入；只影响 **H100 上的 R15 单元**（fa_pi_f_f_o_q_*） | 两处去掉 `.resolve()`（字符串路径和原来一样）；不想改就让这 8 个单元留在 /data 跑完再迁 |
| `ops/formal_queue.py:302`、`queue_retry.py:257` | 和 FORMAL_ROOT 比较后 raise | 旧的 formal_queue 工具，当前链不用 | 暂不改 |
| `longseg_1017/check.py:101,124` | `samefile` 断言（交回时的 CPU 证明） | 不保留硬链接的副本会失败；不在链路径上 | 见到就拔 |
| `longseg_1017/trash.py`、`build.py:103` 的 `.incomplete_` 改名 | 对软链做 rename 只会挪走软链本身 | 实体目录还留在 /nvme | 迁移后挪垃圾或改名要在 /nvme 那一侧做 |

已经被拔成 `pass`、不再挡的检查：`formal_prestage.py:195,233,307`、`formal_runtime.py:33,38,96,135`、`cache_control.py:101`、`immutable_release.verify` 里的软链和 sha 判断（:37,44）。`stage.py:95,118` 和 `launch_prepared`（formal_prestage.py:306）都先 resolve，所以 PREPARED、LAUNCH、RECEIPT 等文件都会写到 /nvme。`finalize` 用的也是 resolve 之后的目录，`incremental_sync.restore` 不会被触发。

其它注意事项：MPS 管道路径要求不超过 85 字节，`/nvme/weiland/openpi/eval/<run>/mps/<6>/p` 比 /data 多 4 字节，run 名控制在 44 字符以内（现在都在 20 字符以内）。eval 里有 2 个绝对路径软链（`cosmosp6223294666*` → `/data/mw_finetune/eval/cosmos500w*`），整个 eval 换软链之后仍然能解析。

## 4. 方案

### A. 立即：新写入一律落 /nvme

**A0 停掉复制残留**（前提：无；风险：rsync 被 KILL 后，在 /nvme 上留一个 `.stall.pkl.VN5Ynb` 临时文件，无害）
```bash
kill -KILL 3729764 3726106 3726105 3343372 3951564 3950599 3296379   # SIGKILL 对 T 状态的进程直接生效
kill 411978                                                            # 监控已不存在批次的 mw_speed2
```

**A1 一次性改代码**（前提：在链进入下一个 prepare 之前提交；已在跑的 launcher 保留内存里的旧代码，不受影响）
1. 在 `immutable_release.py` 里加一个辅助函数（只用 stdlib；H100 通过 release_install.py 拿到的是同一个文件，但那里没有 /nvme，函数不起作用）：
```python
NVME, DATA = Path('/nvme/weiland/openpi'), Path('/data/mw_finetune')
def nvme_dir(path):
    """New local dir: real dir under /nvme/weiland/openpi, the /data path left as a symlink (owner 10-19/10-20)."""
    path = Path(path)
    if not NVME.is_dir() or not path.is_relative_to(DATA):
        path.mkdir(parents=True, exist_ok=False); return path
    real = NVME/path.relative_to(DATA); real.mkdir(parents=True, exist_ok=False)
    if not path.exists(): path.symlink_to(real)   # once /data/mw_finetune/eval itself points to /nvme, path is already real
    return path
```
   在 `install()`（:60-81）里，当本机有 /nvme 且 root 在 /data/mw_finetune 下时，`real=NVME/root.relative_to(DATA)`：lock 和 `mkdtemp` 都放在 `real.parent`，`os.rename(stage, real)`，最后 `root.symlink_to(real)`。已经存在的发布目录照旧复用，不复制。
2. 把 6 处 mkdir 换成 `nvme_dir(...)`：formal_eval.py:1941、filter_failed/resume.py:30、filter_failed_knob/build.py:123、longseg_1017/build.py:104、longseg_1017/pod_copy.py:152、queue_retry.py:103。
3. 改常量：q3_port_1020/build.py:31 和 cache_study_1019/study.py:27 的 `EVAL` 改成 `/nvme/weiland/openpi/eval`；local_workers.py:15、worker_storage.py:19、vfix.py:236、reset_remote.py:100 改成 `/nvme/weiland/openpi/qualification/local_workers`，先 `cp -a /data/mw_finetune/qualification/local_workers/{env311,dependencies,assets,payload_*}` 过去（dependencies 约 609 MB；runs/ 不复制）。
4. 拔检查、修跨盘：formal_eval.py:1929-1930、local_workers.py:155-157、sync_back.py:18-19（父目录软链检查）；io_budget.py:127 加 EXDEV 时退回复制；control.py:431 和 assets.py:138 去掉 `.resolve()`。
5. 不改：campaign、YAML、commands 里的 /data 字符串，链脚本的 `E=`，t107 guard 的 glob（通过软链都能用）。

**A2 排队未开始的目录：搬到 /nvme、原路径留软链（选这个，不改它们的路径）**
- 理由：这些目录的 YAML 字节、task UID、group_sha256（R16 的 root 写在里面）、PREPARED pin 都已封定，路径字符串同时也是 H100/pod 上的远端路径；改路径等于重建。搬过去再留软链，所有字节不变，而且 prepare 会 resolve，写出的东西全落在 /nvme。
- 对象：§1.4 列出的已建目录（H100 链 10 个 + `fa_ls_v05_1` 和它的 `.before_retry_1430` 备份 + fw_r_f_mc2；4090 的 6 个 R16 组 + 16 个 R15 GR00T cell；H100 R15 的 8 个 fa_pi_f_f_o_q cell，前提是已按 A1.4 去掉 resolve）。合计约 30 GB，读 /data 几分钟。
- 每个目录的前提：链还没 prepare 到它；`/proc/*/{cwd,fd}` 里没有进程用这个路径；A1 已经上线。
```bash
busy(){ for p in /proc/[0-9]*; do { readlink $p/cwd; ls -l $p/fd 2>/dev/null|awk '{print $NF}'; } 2>/dev/null | grep -q "^$1\(/\|$\)" && echo ${p#/proc/}; done; }
swap(){ d=$1; t=$2; l=$(dirname $d)/.$(basename $d).nvme_link     # 原子交换：d 变成软链，旧实体目录改名保留
  ln -s "$t" "$l" && python3 -c 'import ctypes,os,sys;c=ctypes.CDLL(None,use_errno=True);sys.exit(0 if c.renameat2(-100,sys.argv[1].encode(),-100,sys.argv[2].encode(),2)==0 else os.strerror(ctypes.get_errno()))' "$l" "$d" && mv "$l" "$d.data_old_1021"; }
for n in fa_ls_v03_1 fa_ls_gr_f_f_m ...; do
  [ -z "$(busy /data/mw_finetune/eval/$n)" ] && ! grep -q "prepare $n$" .../mw/chain.log || { echo skip $n; continue; }
  rsync -a /data/mw_finetune/eval/$n/ /nvme/weiland/openpi/eval/$n/ \
  && [ -z "$(rsync -a --dry-run --itemize-changes /data/mw_finetune/eval/$n/ /nvme/weiland/openpi/eval/$n/)" ] \
  && swap /data/mw_finetune/eval/$n /nvme/weiland/openpi/eval/$n
done
```
- 风险：如果复制过程中链正好开始 prepare 这个目录，副本会过时。对策是开始复制前再看一眼 chain.log；或者先 touch 对应链的 hold 文件，等它停在两项之间。R15 H100 单元如果不去掉 resolve，就不要搬这 8 个。

**A3 本机 R16 island**（前提：fa_cs_gr_r 的 runner 和本机 agent 都已退出，且下一个 R16 runner 还没起；做法和 r15_island 的先例一样，不改代码）
```bash
rsync -a --exclude=/r16_agents/ /data/mw_finetune/qualification/r16_local_island/ /nvme/weiland/openpi/qualification/r16_local_island/
swap /data/mw_finetune/qualification/r16_local_island /nvme/weiland/openpi/qualification/r16_local_island
```
A100 的 island 是它自己 /data 上的同名路径，不受影响。g4090am 目前处于 hold 状态，正好可以在协调者重起下一条 R16 链之前做。

### B. 在跑批次：跑完前不动，跑完后整目录迁移
- 前提：chain.log 已经有 `<name> ended … receipt=<终态>`（不是 PENDING_EVIDENCE），`busy` 为空。如果这个目录是别的排队批次 YAML 引用的源，只要那些批次的服务端不在本机，就不受影响。
- 每个目录：`rsync -a`（不限速）→ dry-run 结果为空，并且 RECEIPT.json、campaign.json、driver/journal.jsonl 加 20 个随机文件的 sha256 两边一致 → `swap`。v09_1_j4 有 13 GB，大约 1.5 分钟高负载读，可以接受（这是「一个不限速的短窗口」）。
- 顺序：fa_ls_v09_1_j4（~00:20）→ fa_ls_v03_0（~03:00）→ fa_pi_f_u_o_q_*（8 个 cell）→ fa_cs_gr_r → fa_ls_gr_f_u_m_j。
- **pod 上的 GR00T（fa_ls_gr_f_u_m_j，13–30 h）可以提前收尾**：照 h100-temp mc1 的先例。A1 上线后：SIGTERM 它的 launcher（目前是 876723，让它平稳停下）→ 等 RECEIPT 定稿 → 用同样的 pod alias 环境（`R17_H100_NODE=jupyter-ziyang10-02 R17_H100_HOST=149.165.153.233 R17_DRIVER_PORT=23192`）执行 `.venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.resume --source /data/mw_finetune/eval/fa_ls_gr_f_u_m_j --name fa_ls_gr_f_u_m_jc1 --remaining --verify-closed`。新目录由 resume.py:30 的 `nvme_dir` 建在 /nvme。然后照 prepare/launch_fa_ls_fw_r_f_mc1.sh 的样子写两个脚本、起链。只读源的 journal（5 MB），不读 per_step。代价：prepare 大约 8 分钟，在途的局重排。好处：最后一个往 /data 写的批次提前十几个小时消失。757 条 error 本来也需要处理，可以借这次一起处理。

### C. 存量
- **C1 库和 r17_frozen**（现在就做）：本机没有进程在用；库在 10-20 04:06 已核对；r17_frozen 复制 rc=0 但没核对过，先做 `rsync -a --dry-run --itemize-changes`（11 GB，很快）。然后 `swap /data/r17_library /nvme/weiland/openpi/r17_library`，r17_frozen 同理。库的 dry-run 核对（要 stat 大约 100 万个文件）放到 C3 的窗口里做；交换后旧实体目录还在，可以对比。注意：库的建库工具（`library_staging.py:76`、`library_init.py:59`、`library_cleanup.py:28` 等）会拒绝软链根目录，但按记忆规定库不再重建。
- **C2 eval 剩余部分**：暂停时已复制 ~533 GB（/data 口径，10:00），还剩 ~190 GB；另外 10-19 22:21 之后新建或当时排除的 ~70 个 fa_* 目录（§1.4 里那些，加上 jc2、gr_r_f_mc1 等），暂停的 rsync 永远不会补上（它的顶层文件列表在启动时就定了）。/nvme/weiland/openpi/eval 现在 551 GB，其中 64 GB 是 54 个后来进了 oldgate 垃圾箱的旧副本。**重新跑一遍**，rsync 对已一致的文件只做 stat：
```bash
cd /data/mw_finetune/eval
find . -maxdepth 1 -mindepth 1 -type l -printf '/%P\n' > /tmp/eval_links.txt      # 软链（A2/B 换掉的和 2 个旧别名）一律排除，否则 rsync 会去替换 /nvme 上的实体目录
printf '/%s/\n' <在跑批次名> >> /tmp/eval_links.txt
rsync -a --info=progress2,stats2 --exclude-from=/tmp/eval_links.txt /data/mw_finetune/eval/ /nvme/weiland/openpi/eval/
```
   不限速、不加 governor；有硬链接的文件在 /nvme 上会展开成独立副本（与已复制部分一致；/nvme 还剩 2.6 TB）。窗口有两种选法：①所有 /data 上的 driver 都结束之后（B 做完；如果 pod GR00T 提前收尾，大约明早）；②A2 和 v09_1_j4 收尾之后，一次跑完，接受这一小时里剩下的 2–3 个快速 driver 慢 2–4 倍。推荐②，不要拖到 30 h 之后。
- **C3 核对**（一次性手工核对，不是新闸门）：无 driver 时再跑一次 `rsync -a --dry-run --itemize-changes`，应当为空；`find -type f | wc -l` 和 `du -sl --apparent-size` 两边对比（-l 按硬链接展开计数，才和 /nvme 口径一致）；随机 300 个文件加上全部 `RECEIPT.json`、`driver/journal.jsonl` 比 sha256；`find /data/mw_finetune/eval -maxdepth 3 -type l -lname '/*'`，看有没有指向 /tmp 或 /data 的绝对软链（旧 Cosmos 的 driver/ 可能是这种）。
- **C4 整个 eval 换软链**：前提是 `busy /data/mw_finetune/eval` 为空，且 A1 的 4 道检查已拔。`swap /data/mw_finetune/eval /nvme/weiland/openpi/eval`，旧目录变成 `/data/mw_finetune/eval.data_old_1021`。之后所有 `EVAL/name` 都会自然落到 /nvme，`nvme_dir` 也不再需要建软链。qualification（发布目录、observations、assets、local_workers、island）在同一个窗口按同样方法处理（driver 的 cwd 在里面，必须等没有 driver 时做）。
- **C5 垃圾箱**：`/data/mw_finetune/_trash_{devknob_20261016,longseg_20261017,oldgate_20261020}`（handoff 10-19 记约 424 GB，之后又加了 oldgate 的 54 个目录）是作废数据，不占用最宝贵的 /nvme。继续留在 /data，或者按任务 #191 等 owner 同意后挪到 /archive。/nvme 上那 54 个过时的 oldgate 副本（64 GB）用 `mv` 收进 `/nvme/weiland/openpi/_trash_oldgate_20261020/eval/`（同盘改名，不删）。
- **C6 范围外但仍是 /data 读依赖**（只读，不写，由 owner 决定）：模型和环境 `/data/mw_finetune/{dp,models,groot,fastwam,cosmos,envs,vendor,hf}`、`/data/ckpt`（54 GB）、R16 库 `/data/libero_cache`（111 GB，q3 的 R16 组在读）。
- 原目录一律改名保留，不删；删除要 owner 同意。

### D. 远端机器（tether exec 只读实测）
- H100：`/data` 是独立的 1 TB Jetstream 云盘（sdb，QEMU HARDDISK），**已用 92%，只剩 82 GB**，没有 /nvme。不是本机那块盘，不在本次范围，但磁盘余量要留意。
- A100：/data 就在它 58 GB 的根盘上（已用 71%）。h100-temp：/data 在 993 GB 的根盘上（已用 95%）。pod：/data 是 r17x 把 /tmp/zl_r17/data 绑定过去的。
- 这些都不是 weilandserver 的 sda（ST4000NM0035），不在本次范围内。

## 5. 顺带发现（与迁移无关，会马上影响队列）
- `chain_ffk_stop.run.sh` 用 `$E/$J/RECEIPT.json` 判断是否 PASS，但 R15 单元（如 fa_pi_f_u_o_q，实际目录是 `_<cell>`）和 R16 组（只写 DONE.json）都没有这个文件，chain.log 里 fa_cs_pi_r/fa_cs_gr_r 都记成 `receipt=`（空）。所以 **h100q1 跑完 fa_pi_f_u_o_q 会 STOPPED，不会继续 fa_pi_f_f_o_q**；g4090am 如果用 stop 版重起，同样只会跑第一组。
- fa_ls_gr_f_u_m_j 的 journal 里已有 757 条 error，最后一条是 `RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED`。
