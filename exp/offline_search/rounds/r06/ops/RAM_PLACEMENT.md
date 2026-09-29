# weilandserver RAM 放置方案（r06 ops，只读调查）

- 调查时间：2026-09-28 20:20–20:50（本机本地时间 CDT，-0500）
- 方式：全程只读（`nice -n 19 ionice -c3 taskset -c 14-17,58-61`），本文件是唯一写入；未移动/复制/删除任何文件，未碰任何进程、tmux、端口、sandpile 文件
- 工具：`/proc/*/io`、`/proc/diskstats`、`iostat`、`pidstat`、`sar`（/var/log/sysstat/sa26–sa28）、`/proc/*/maps|fd|cmdline`、`mincore()`（本机无 `vmtouch`/`fincore`，用 ctypes 调 libc mincore 测 page-cache 驻留，脚本见附录）

---

## 0. 结论速览

1. **没有设备被持续打满。**
   - sdc（/home SATA SSD）是唯一忙的盘：19:00 起读量从 <0.5 MB/s 跳到 22–133 MB/s（10 分钟均值，20:00 峰值 132.8 MB/s、util 41%）；1 秒粒度峰值 185 MB/s、4.2k r/s、util 63%，但 r_await ≤ 1.1 ms，没有饱和。
   - sda（/data HDD）只在每次 groot server 启动时突发：aqu 3–9、await 37–150 ms，这 30–60 秒内局部打满。
   - nvme、sdb 基本空闲。PSI：io some avg300 0.40%，memory some avg300 0.50%。
2. **sdc 的读 95% 以上来自 codex P2 的 token-PCA**，即两个 `token_pca:TokenPCAAWM` fit worker 对 31 GB token 文件每个相机扫 10 遍。这些读几乎全是 page-cache 被挤掉后的 refault：同一分钟内 `workingset_refault_file` 14.2k 页/s ≈ 58 MB/s，而 PCA 进程的磁盘读正好是 58 MB/s。PCA 已于 20:43:44 / 20:46:15 完成。20:47 时的 40–50 MB/s 来自其收尾的 `pca_checks.sh`（还剩 1 个 check + validate + 2 个 smoke，几分钟量级）。
3. **根因是 RAM 被 /dev/shm 占满。**
   - /dev/shm 用了 86.5 GiB：78.9 GiB 在 RAM，约 7.6 GiB 被挤进 swap。
   - swap 已满（7.9/8.0 GiB），其中进程匿名页只有 0.21 GiB，其余几乎都是 shm 页。
   - 结果 file page cache 只剩约 102 GiB，装不下「2×29 GiB token 流 + 常驻热集」，于是 thrash。
4. **`/dev/shm/offline_search_store` 内容上不陈旧，只是没人用。**
   - 它是 09-26 14:44–14:46 从磁盘 store 复制的「无 tok」镜像（tok 目录是指回磁盘的 symlink），09-27 21:2x–21:4x 又加了 demo100/200/300 和 grow250。
   - 与磁盘 store **逐文件一致**：457 个文件中 402 个全量 sha256 一致、55 个大文件抽样 sha256 一致，缺失 0、尺寸不符 0；反向看，磁盘多出的只有 `_run/` 构建日志。**可以直接丢，不需要先拷回。**
5. **目前只有 line_P6C2 的最后 2 个 groot_spatial arm 在用它。**
   - 这两个 arm 是 `r5x_g_sp_50_tail1u_rep3`（运行中，20:39 起）和 `r5x_g_sp_500_tail1u_rep3`（排队中，P6C2 预计约 21:20–21:30 结束）。它们的 arm spec 没写 `--os-root`，plugin 默认取 `/dev/shm/offline_search_store`，且只读 `groot_spatial` 子树（6.6 GiB）。
   - 其余 **80 GiB 没有任何运行中或排队的 server 在用**：所有 pi05 server 和全部 216 个 p3 pilot arm 都显式用磁盘 store；P6A 的 26 个 arm 也全用磁盘 store。
6. **方案。**
   - Phase 1（SAFE-NOW）：丢掉 80 GiB，只保留 groot_spatial。
   - Phase 2（BETWEEN-ARMS，P6C2 结束后）：丢掉剩下的 6.6 GiB。
   - Phase 3（SAFE-NOW）：把 groot ckpt 和 preload pkl（/data HDD 上，共 16.7 GB）**预热进 page cache**。这是性价比最高的 RAM 用法，约省 30 GB 读/GB。
   - **不建议往 /dev/shm 放任何新东西**：建议总量 0 GiB。可选再用 16 GiB 硬 pin groot ckpt，需 owner 裁定、要改代码。
7. **⚠ 必须同步的风险。** codex P2（tmux `cx_p2pca`）的 8 个 `r6p2_direct_*` 闭环 arm spec（`exp/offline_search/rounds/r06/p2_ablations/arms_pca.json`）**都没有 `--os-root`**；`pca_prepare.py` 只在它自己的 prefit 命令里补了 `--os-root`。这些 arm 如果原样进 chain，server 会默认读 /dev/shm；丢掉 shm 后会在启动时报 FileNotFoundError（失败是响亮的，不是静默的）。二选一：emit 时补上 `--os-root /home/weiland/trace_runs/offline_search_store`；或者按 Phase 2b 放一个兼容 symlink（NEEDS-OWNER）。

---

## 1. 当前 IO

### 1.1 设备层

sar 10 分钟均值，2026-09-28：

| 时段 | sdc 读 MB/s | sdc 写 MB/s | sdc util | sda 读 MB/s | sda aqu / await |
|---|---|---|---|---|---|
| 16:00–18:50 | 0.0–0.4 | 0.3–7 | 0.3–7.7% | 0 | – |
| 19:00 | 69.5 | 6.5 | 23.5% | 0 | – |
| 19:10 / 19:20 / 19:30 | 62.4 / 62.9 / 36.0 | 9–24 | 15–29% | 1.3 / 9.7 / 14.1 | 5.6 / 40 ms；8.2 / 88 ms |
| 19:40 / 19:50 | 32.3 / 73.7 | 6 | 16–31% | 0 / 6.7 | 8.7 / 49 ms |
| 20:00 | **132.8** | 5.5 | **41.1%** | 13.5 | 3.8 / **150 ms** |
| 20:10 / 20:20 / 20:30 | 94.8 / 22.3 / 43.3 | 3–15 | 20–33% | 0 / 0.3 / 10.0 | 2.9 / 124 ms |
| 20:40 | ≈47 | – | 24.7% | 4.0 | 2.7 / 37 ms |

- 前几天对照：09-26 的 store 构建期有过 189 MB/s；09-27 01:50–04:10 有约 88 MB/s 的持续读（util 28%）。sdc 的 10 分钟 util 从没超过约 70%。
- 1 秒粒度（20:3x，20 秒）：sdc 读 56–185 MB/s、1.7k–4.2k r/s、r_await 0.42–1.14 ms、util 34–63%。
- 20:30 → 20:43 期间 sda 共读 5.25 GB，正好等于 20:39 groot_spatial 启动时 ckpt 未缓存部分（约 4.8 GB）加上 preload pkl（0.43 GB）。

### 1.2 进程层

`/proc/<pid>/io` 的 read_bytes/write_bytes 做 60 秒差分：

| 窗口 | sdc 读 | 主要读者 | 备注 |
|---|---|---|---|
| ≈20:28–20:29 | 44.7 MB/s（1366 r/s，util 24%） | PCA pi05_l10_500（pid 3361995）29.2 MB/s；PCA groot_l10_500（pid 1815997）13.9 MB/s | 其余进程都 < 2.5 MB/s |
| 20:33:36–20:34:36 | 15.1 MB/s | PCA 8.6 + 2.4；codex app-server daemon 4.1 | – |
| 20:34:36–20:35:36 | 61.5 MB/s | PCA 58.1 + 1.6 | refault_file 14,233 页/s（≈58 MB/s）；pgsteal_kswapd 38.7k/s；pgsteal_direct 1.1k/s → **PCA 的读 ≈ refault，是 thrash** |
| 20:46:55–20:47:05（pidstat） | 42–49 MB/s | `pca_check --arm r6p2_direct_g_sp_500` 40.7 MB/s；`pca_checks.sh` 刚回收的子进程约 174 MB/s | PCA fit 已结束；check 读的是磁盘 store 的 `queries/*_{cache,inf}` key/tok 抽样 |

- 写：sdc 持续 2–10 MB/s，来自各 server 的 decisions/inputs、p3 stream receiver（pid 708338，0.8 MB/s）、chain 日志。偶有峰值（20:47 出现过 62 MB/s 的写，多半是 fit/check 的 pkl 或 npy 落盘）。对 SSD 来说没有压力。
- 新启动的 server 自身磁盘读 ≤ 1 MB/s：pi05 ckpt 和 cp1 pool pkl 已 100% 在 page cache。
- tmux server（pid 1146111）的 rchar 高达 140–327 MB/s，那是 pty 输出流量（CPU 开销），不是磁盘 IO。

### 1.3 内存（20:29 快照；20:46 的 free -g 与之基本一致）

| 项 | 值 |
|---|---|
| MemTotal | 251.8 GiB |
| MemFree / MemAvailable | 13.7 / 123.2 GiB（20:46：26 / 140 GiB） |
| Shmem（RAM 中的 tmpfs） | 78.9 GiB；/dev/shm df 显示 86.5 GiB，差值约 7.6 GiB 在 swap |
| file cache：Active(file) + Inactive(file) | 51.2 + 51.4 = 102.6 GiB |
| AnonPages | 43.3 GiB |
| Slab | 10.6 GiB（可回收 8.8） |
| Swap（/swap.img，在 nvme） | 7.90 / 8.00 GiB 已用；进程 VmSwap 合计只有 0.21 GiB |
| 历史累计（10 天） | pswpout 19.9M 页（约 76 GiB）；workingset_refault_file 326M 页（约 1.2 TiB） |

各组匿名内存 PSS：pi05 server 28.6 GiB（4 个）；groot server 5.0 GiB（2 个）；sandpile 3.0 GiB（17 个进程）；claude/codex agent 2.8 GiB；PCA 1.6 GiB（已退出）；vscode 1.1 GiB。合计 42.5 GiB。

**sandpile（另一项目，未碰）**：5 个 `sandpile.infra.lane`（每个约 0.3 GB）和 1 个 `sandpile.train.train`（1.2 GB RSS，含 compile worker），外加 tmux `i011-limits` / `i011-mirrors` 的 mirror_loop。每次采样里它的磁盘 IO 都约等于 0。它在 /dev/shm 里只有自己进程持有的已删除 torch/semaphore 段（`pym-2452358`、`sem.*`），体量可忽略。它的 `--min_free_mb 13000` 门控看的是 **GPU 显存**（nvidia-smi），与 RAM 方案无关。

---

## 2. /dev/shm 与其它 tmpfs 清单

### 2.1 顶层条目

| 条目 | 大小 | 属主 | 修改时间 | 当前使用 | 分类 |
|---|---|---|---|---|---|
| `offline_search_store/` | 86.5 GiB（88,585 MiB，apparent） | weiland；openpi offline_search 线 | 目录 09-26 14:46；demo/grow 09-27 21:4x | 仅 P6C2 的 2 个 groot_spatial server 在 mmap（pid 3337864、3372491 打开 `library/groot_spatial/bpool_all/action.npy`）；20:30 时是 groot_l10 的上一个 arm | 6.6 GiB 在用；80 GiB **stale（无人用）** |
| `__KMP_REGISTERED_LIB_<pid>_1000` ×5716 | 各 1 KiB，共约 22 MiB（tmpfs 按页计） | weiland；Intel OpenMP 每个进程的注册文件 | 09-18：991 个；09-19：4686 个；09-26：39 个 | 3 个 pid 仍存活；**5713 个 pid 已死** | stale，属卫生项 |
| 已删除但仍打开的段 | 可忽略 | `sem.*`（本项目 pi05/groot server 与 compile worker、sandpile）；`pym-2452358`（sandpile compile worker） | – | 各自进程持有 | 进程退出自动释放，不处理 |

其它 tmpfs：`/run` 8.9 MiB；`/run/user/1000` 88 KiB；`/run/lock` 0；`/run/qemu` 0。`/tmp` 不是 tmpfs（在 nvme 根分区；`/tmp/torchinductor_weiland` 0.5 GB）。没有 SysV 共享内存段。

### 2.2 `offline_search_store` 子树

单位 MiB。「谁引用」一栏：P6C2 = r06_paper 里没写 `--os-root` 的 arm；pi05 server 全部显式使用磁盘 root。

| 子树 | MiB | 修改时间 | 谁引用 / 状态 | 分类 | 磁盘副本 |
|---|---|---|---|---|---|
| library/pi05_l10（current 664、bpool_all 664、bpool_cs 7410、demo100 1450、demo200 2947、demo300 4428、grow250 3300） | 20,860 | 09-26 14:46 / 09-27 21:4x | 默认 root 的 pi05 arm（r5t/r4b3、r6p2_identity_p）**已全部 DONE**；无运行中或排队的消费者 | stale | 一致 |
| library/pi05_spatial | 8,173 | 同上 | 同上 | stale | 一致 |
| library/groot_l10（current 667、bpool_all 7469） | 8,136 | 09-26 14:46 | P6C2 的 r5x_g_l10_*_rep2/rep3 **已全部 DONE**（最后一个 20:38 完成） | stale | 一致 |
| **library/groot_spatial**（current 268、bpool_all 2963） | **3,231** | 09-26 14:45 | **P6C2 运行中 + 排队各 1 个 arm** | **在用** | 一致 |
| queries/pi05_l10_{cache,inf} | 10,187 + 7,465 | 09-26 14:45 | plugin 在线只读 `queries/<cell>/episodes.json`；pi05 无消费者 | stale | 一致 |
| queries/pi05_spatial_{cache,inf} | 3,712 + 2,742 | 同上 | 同上 | stale | 一致 |
| queries/groot_l10_{cache,inf} | 10,154 + 7,440 | 同上 | 无（l10 已完成；`_inf` cell 从不被 server 读） | stale | 一致 |
| queries/groot_spatial_inf | 2,903 | 同上 | 无 | stale | 一致 |
| **queries/groot_spatial_cache** | **3,538** | 同上 | P6C2 启动时读其中的 `episodes.json`；整目录保守保留 | **在用** | 一致 |
| floor/*（4 个 cell） | 44 | 09-26 14:46 | 保留 groot_spatial 那份（13 MiB） | 其余 stale | 一致 |
| `*/tok`（19 个 symlink）、`profile_cache`（symlink） | 0 | – | 指回 `/home/weiland/trace_runs/offline_search_store/...` | 无 RAM 占用 | 链接目标就是磁盘 |

- **一致性核对**（20:4x）：对 shm 中每个文件做「同名磁盘文件存在 + 尺寸相同 + sha256」比对。≤64 MiB 的文件全量 sha256，共 402 个；更大的 55 个做抽样 sha256（头 4 MiB、尾 4 MiB、6 个随机 1 MiB 块，两边同偏移）。结果 0 不一致。核对本身从磁盘读了 1.85 GB（idle ionice，19 秒）。
- **反向核对**：磁盘有而 shm 没有的只有 `library/_run`、`queries/_run`、`floor/_run` 共 19 个构建日志条目。`store.library_names()` 只枚举 `library/<lib_key>/*`，所以两份 root 对 server 来说库集合完全相同。

### 2.3 谁的配置或代码会指向 /dev/shm（丢弃前必须知道）

- **arm spec 中没写 `--os-root`、从而默认 `/dev/shm/offline_search_store` 的 arm**：
  - r06_paper：16/36 个，其中只有 `r5x_g_sp_50_tail1u_rep3`（运行中）和 `r5x_g_sp_500_tail1u_rep3`（排队）未完成。
  - r06_abl：8/24 个（`r6p2_identity_*`），已全部 DONE。
  - r06_p3_pilot：0/216 个。
  - **待 emit 的 `arms_pca.json`：8/8 个 `r6p2_direct_*` 都没写**（见 §0 第 7 条）。
- **代码默认值**：
  - `exp/offline_search/closed_loop/plugin.py:91`（`--os-root` 默认值）
  - `ops/kpi.py:502,1269`（按 server 日志里记下的 root 找 store，找不到再回落 /dev/shm）
  - `verify_logs.py:354`、`replay_client.py:76`、`selftest.py:376`
  - `rounds/r05/q4_growth/common.py:12`（SHM）以及 r01–r03 的大量分析脚本
  - `start_server.sh` 和 `chain.sh` 本身不含 /dev/shm；它们照搬 arm 的 `plugin_args`。
  - `ops/collect.py`（chain 在 arm 结束时生成 summary.json）**不读 store**，因此不受影响。
- **指进 /dev/shm 的 symlink**（丢弃后会悬空，都是历史 verify_work 目录）：
  - `trace_runs/offline_search_store/derived/r02`（6 个）、`derived/r03`（8 个）
  - 仓库内 `exp/offline_search/rounds/r04/k3_cost/results/*/verify_work/library/*`（10 个）
- **正在运行的脚本**（line_P6A_loop5/v2、line_P6C2、p3_pilot_line、chain_p3.stream.frozen.sh、p3_pilot_watch、r4_watch、pca_checks、p3_cell_pipeline、q_cell_analysis）：都不含 /dev/shm。只有 `.claude/jobs/a607dd74/tmp/` 里旧的 `run_r0x_*.sh` / `prefit_g50*.sh` 含有，而它们不在运行。

---

## 3. 什么值得放进 RAM

判别标准：

- **page cache 已能处理**：文件会很快被再读，只要 RAM 不被 shm 挤占，它就会留在 cache 里。这类不需要动，或者最多预热一次。
- **需要 pin**：两次读之间会被挤出去，或者是「写完马上读」的 scratch，且 page cache 保不住。

| 排名 | 候选 | 大小 | 当前驻留 | 重读模式 | RAM 常驻能省的 IO | 结论 |
|---|---|---|---|---|---|---|
| 1 | **释放 /dev/shm store** | 86.5 GiB | 占 RAM 78.9 GiB + swap 7.6 GiB | 只有 6.6 GiB 有人读 | 间接收益：page cache 多出约 72–80 GiB，swap 也被释放，token 流一类的 thrash 随之消失（本轮 PCA 期间 sdc 读的主体就是 refault） | **首要动作**（Phase 1/2） |
| 2 | groot ckpt `/data/ckpt/n15_libero_10`、`n15_libero_spatial`（HDD） | 2 × 7.59 GB | 51.7% / 36.9% | 每次 groot server 启动都读。仍排队的 groot 启动：P6C2 1 个 arm × 2 server；P6A 12 个 arm × 2 server（r6p1 rep3 4 个 + r6p2 groot 8 个）；p3 pilot 81 个 arm × 1 server | 每次 arm 启动实测 3.6–4.8 GB HDD 读（20:39 那次 5.25 GB），启动慢约 60 s，HDD await 40–150 ms。累计约 350–450 GB / 15.2 GB，**约 25–30 GB/GB** | **预热进 page cache**（Phase 3，SAFE-NOW）；若之后仍被挤出，再硬 pin（Phase 4，NEEDS-OWNER） |
| 3 | groot preload pkl `/data/libero_cache/libraries/libero_10/libero_10_sp16_S3.pkl`、`libero_spatial/libero_spatial_sp16_S3.pkl`（HDD，由 yaml 的 `preload_path` 引用） | 1.07 + 0.43 GB | 6.8% / 0.1% | 同上，每次 groot 启动 | 约 1 GB/次，**约 60 GB/GB** | 与第 2 项一起预热 |
| 4 | token 数组 `library/*/tok/v{0,1}.npy`（磁盘 store；本轮热点是 `pi05_l10/bpool_cs` 和 `groot_l10/bpool_all`，各 2 × 30.9 GB） | 195 GB（全部）；本轮热集 62 GB | 当前 v1 各 86%，v0 4–18% | token PCA 每个相机 10 遍；**本轮已结束**（fit 20:43/20:46 落盘） | 若以后再跑 grid ladder（1×1/2×2/8×8 各要重 fit）：释放 shm 后 page cache 装得下两个 29 GiB 文件，读量从约 10 倍降到约 1 倍 | **不拷进 /dev/shm**（拷一次的 IO 约等于剩余能省的 IO，还会挤占 page cache）；开跑前可选顺序预热 |
| 5 | 共享的 base/guard fit（`r05_ptail`、`r05_q1`、`r05_x`、`r06_paper/fits`） | 1.3 GB | 28–100% | 若在启动时被读，则多次重读 | 很小 | page cache 已处理；可顺手预热（可选） |
| 6 | p3 pilot 每个 arm 自己的 fit（`r06_p3_pilot/fits`，216 个） | 25.3 GB | 1% | 每个 arm 只读 1 次（1 个 server） | 约 1 GB/GB，即零收益 | **不放** |
| 7 | pi05 ckpt `~/.cache/openpi/.../pi05_libero_pytorch/model.safetensors`（sdc） | 7.23 GB | **100%** | 每次 pi05 启动 | page cache 已处理 | 不动 |
| 8 | pi05 preload pkl `trace_runs/dual_20260923/libs/*/cp1_spatial_pool_16.pkl` | 1.53 GB | **100%** | 每次 pi05 启动 | 已处理 | 不动 |
| 9 | store 的 `action.npy` 及元数据（server 启动时对该 lib_key 下每个库都 mmap） | 0.24 GB | 12% | 每次启动，量很小 | 可忽略 | page cache |
| 10 | store 的 `key_v*.npy`、`queries/*`、`tok/img*` | 42 + 50 + 14 GB | 0–7% | **server 在线不读**（`key_v*` 已含在 fit pickle 中）；只有离线 prefit、分析、pca_check 偶尔读 | 偶发 | 不放 |
| 11 | `r06_p3_pilot/tables/<cell>/*.csv`（`p3_cell_pipeline.sh` 生成；每个 cell 7–11 GB，其中 `controls.csv` 6.6 GB） | 18.8 GB | 40% | 写完后几分钟内被 q1/q2/q3 读一遍，下个 cell 换新文件 | 写入即在 cache，释放 shm 后更稳 | page cache 已处理；不放 |
| 12 | venv 的 `.so`（openpi 7.7 GB、gr00t 6.5 GB）、`/tmp/torchinductor_weiland` | – | 被映射的部分常驻（11% / 6%） | 启动时 | 已处理（inductor 在空闲的 nvme 上） | 不动 |
| 13 | 写入流（decisions jsonl、inputs npz、client_telemetry 2–4 MB/s） | – | – | 必须持久化 | – | 留在磁盘 |

**预算（owner 要求：给进程和 page cache 至少留约 60 GB 空闲）**

- MemTotal 251.8 GiB。
- 匿名内存峰值估约 65 GiB：6 个 server 槽位（P6A 2 个全模型 pi05，每个 7–13 GiB；P6C2 2 个；pilot L3/L4 2 个，每个 7–13 GiB）+ sandpile 3 + agent 4。
- 内核约 12 GiB。
- 按 owner 规则，/dev/shm 的硬上限约为 251.8 − 65 − 12 − 60 ≈ **115 GiB**。
- 但 page cache 自己也要装下热集：常态约 37 GiB（pi05 ckpt 6.7、groot ckpt 14.1、两类 preload pkl 2.8、当前 cell 的 tables 约 10、venv 映射部分和共享 fit 约 3）；有 token PCA 在流读时还要再加 58 GiB。
- **建议 /dev/shm 上限 20 GiB，建议总量 0 GiB**（Phase 1+2 之后）。仅当 Phase 3 的预热被证明留不住时，才按 Phase 4 用约 16 GiB 硬 pin groot ckpt 和 preload pkl。**不要再整份复制 store（87 GiB）进 /dev/shm。**
- 预期效果：Phase 1 之后 /dev/shm 从 86.5 降到 6.6 GiB，可用 RAM 增加约 72–80 GiB，swap 中的 shm 页一起释放；Phase 2 之后 /dev/shm 为 0。

---

## 4. 具体动作

执行者：coordinator。所有删除都是 `rm -r` 不带 `-f`，删前先计数。

### A0 — SAFE-NOW（可选卫生项）：清理 pid 已死的 KMP 注册文件（约 22 MiB，5713 个 inode）

```bash
# 计数（期望约 5713）
for f in /dev/shm/__KMP_REGISTERED_LIB_*_1000; do b=${f##*/}; p=${b#__KMP_REGISTERED_LIB_}; p=${p%_1000}; [ -d /proc/$p ] || echo "$f"; done | wc -l
# 删除（删每个之前再检查一次 pid，跳过仍存活的）
for f in /dev/shm/__KMP_REGISTERED_LIB_*_1000; do b=${f##*/}; p=${b#__KMP_REGISTERED_LIB_}; p=${p%_1000}; [ -d /proc/$p ] || rm -- "$f"; done
```

- 需要改的配置：无。
- 风险：几乎为零。存在 pid 复用的极小竞态，后果只是 libiomp 的重复加载检测失效一次。

### A1 — SAFE-NOW：Phase 1，丢掉无人使用的 80 GiB（保留 groot_spatial）

前置检查（不满足任何一条就停下）：

```bash
# (a) P6C2 的 l10 arm 都已 DONE（期望列出 4 个文件）
ls /home/weiland/trace_runs/os_closed_loop/r06_paper/state/r5x_g_l10_{50,500}_tail1u_rep{2,3}.DONE
# (b) 没有进程映射或打开要删的子树（期望两行都输出 0）
grep -l -E '/dev/shm/offline_search_store/(library/(pi05_|groot_l10)|queries/(pi05_|groot_l10_|groot_spatial_inf)|floor/(pi05_|groot_l10))' /proc/[0-9]*/maps 2>/dev/null | wc -l
ls -l /proc/[0-9]*/fd 2>/dev/null | grep -c -E '/dev/shm/offline_search_store/(library/(pi05_|groot_l10)|queries/(pi05_|groot_l10_|groot_spatial_inf)|floor/(pi05_|groot_l10))'
# (c) 已通知 cx_p2pca / coordinator：r6p2_direct_* 在 emit 时必须带 --os-root 磁盘路径（或先做 A2b）
```

执行：

```bash
S=/dev/shm/offline_search_store
DROP=($S/library/pi05_l10 $S/library/pi05_spatial $S/library/groot_l10
      $S/queries/pi05_l10_cache $S/queries/pi05_l10_inf $S/queries/pi05_spatial_cache $S/queries/pi05_spatial_inf
      $S/queries/groot_l10_cache $S/queries/groot_l10_inf $S/queries/groot_spatial_inf
      $S/floor/pi05_l10 $S/floor/pi05_spatial $S/floor/groot_l10)
find "${DROP[@]}" | wc -l                 # 期望 452
du -sc -B1M "${DROP[@]}" | tail -1        # 期望约 81,795 MiB
# rm -r 不会跟随 tok -> /home/... 的 symlink，只删除链接本身；--one-file-system 是额外保险
rm -r --one-file-system -- "${DROP[@]}"
ls $S/library $S/queries $S/floor; df -h /dev/shm; free -g
```

- 需要改的配置：无。运行中的进程只用 groot_spatial；pi05 和 pilot 全部显式使用磁盘 root。
- 时机：现在就可以；实验运行中也可以做。
- 风险：
  - 此后任何「不写 root」的调用（上面列出的代码默认值、历史脚本、未加 `--os-root` 的 r6p2_direct arm），只要碰 pi05_* 或 groot_l10，都会 FileNotFoundError。这是响亮失败，不会静默出错。
  - 对 P6C2 以及历史上用 shm root 的 arm 重跑 KPI 或 verify 时，需要显式加 `kpi.py --store /home/weiland/trace_runs/offline_search_store` 或 `verify_logs.py --root ...`（字节已证明完全一致）。

### A2 — BETWEEN-ARMS：Phase 2，P6C2 结束后丢掉剩余 6.6 GiB

```bash
R=/home/weiland/trace_runs/os_closed_loop/r06_paper
ls $R/state/r5x_g_sp_50_tail1u_rep3.DONE $R/state/r5x_g_sp_500_tail1u_rep3.DONE && grep P6C2_DONE $R/line_P6C2_wait.log
grep -l /dev/shm/offline_search_store /proc/[0-9]*/maps 2>/dev/null | wc -l     # 期望 0（23162/23163 的 server 已退出）
find /dev/shm/offline_search_store | wc -l; du -sh /dev/shm/offline_search_store   # 期望约 61 个条目、约 6.7G
rm -r --one-file-system -- /dev/shm/offline_search_store
df -h /dev/shm; free -g
```

- 时机：P6C2 打出 `P6C2_DONE` 之后，预计约 21:20–21:30。在最后一个 arm 进行中删除，对正在运行的 server 本身无害（它们持有 inode），但 chain 在 arm 内重启 server 时会失败，所以要等 arm 之间。
- 风险：同 A1。

### A2b — NEEDS-OWNER：兼容 symlink（可选，在 A2 之后做）

```bash
ln -s /home/weiland/trace_runs/offline_search_store /dev/shm/offline_search_store
```

- 收益：零 RAM。所有默认 root 的调用（plugin 默认值、kpi 回落、verify/replay/selftest、r01–r05 脚本、24 个悬空的 verify_work symlink、未补 `--os-root` 的 8 个 r6p2_direct arm）都会透明地读到逐字节相同的磁盘 store。
- 代价：server startup 行记录的 `root` 仍然是 `/dev/shm/offline_search_store`，但数据实际来自 /home SSD，provenance 字符串有误导性。重启后 symlink 消失（和原来的 store 一样）。
- 替代方案：把上述代码默认值改成磁盘 store。这是 tracked 代码改动，要走 WA 流程。
- 由 owner 二选一。

### A3 — SAFE-NOW（在 A1 之后做）：把 groot ckpt 和 preload pkl 预热进 page cache

```bash
nice -n 19 ionice -c3 taskset -c 14-17,58-61 cat \
  /data/ckpt/n15_libero_10/model-00001-of-00002.safetensors /data/ckpt/n15_libero_10/model-00002-of-00002.safetensors \
  /data/ckpt/n15_libero_spatial/model-00001-of-00002.safetensors /data/ckpt/n15_libero_spatial/model-00002-of-00002.safetensors \
  /data/libero_cache/libraries/libero_10/libero_10_sp16_S3.pkl /data/libero_cache/libraries/libero_spatial/libero_spatial_sp16_S3.pkl \
  > /dev/null
```

- 读量：≤ 16.7 GB 的 HDD 顺序读（部分已在 cache），约 1–2 分钟。尽量避开某个 groot server 正在启动的那 1–2 分钟（看各 chain console 里的 `ARM_START` → `SERVERS_READY`）。mq-deadline 在 6.8 内核上支持 idle 类 ioprio。
- 需要改的配置：无；server 路径不变。
- 复查：2–3 小时后用附录的 mincore 脚本看这 6 个文件的驻留率，期望 ≥ 95%。若再次掉到 < 90%，说明 page cache 仍然保不住，这时考虑 A4。
- 可选：顺手 `cat` 一遍 `r05_ptail`、`r05_q1`、`r05_x` 的 fit（1.3 GB）。
- 风险：无（只是读）。

### A4 — NEEDS-OWNER（仅当 A3 留不住时）：在 /dev/shm 硬 pin groot ckpt（约 15.2 GB）

```bash
mkdir /dev/shm/ckpt_pin
cp -a /data/ckpt/n15_libero_10 /data/ckpt/n15_libero_spatial /dev/shm/ckpt_pin/
( cd /data/ckpt && sha256sum n15_libero_10/*.safetensors n15_libero_10/*.json n15_libero_spatial/*.safetensors n15_libero_spatial/*.json ) > /tmp/ckpt_pin.sha256
( cd /dev/shm/ckpt_pin && sha256sum -c /tmp/ckpt_pin.sha256 )
```

- 代码改动：groot 的 ckpt 路径在 `exp/offline_search/closed_loop/ops/start_server.sh:59` 按 suite 写死，没有环境变量覆盖；所有 lane，包括 frozen 的 p3 chain，都经由 `HERE=.../closed_loop/ops` 调用这份文件（clean，6e1cab5）。需要改成类似 `CK=${GROOT_CKPT_ROOT:-/data/ckpt}` 的形式，并在目录不存在时回落到 /data。然后在 line_P6A_v2.sh、line_P6C2.sh、p3_pilot_line.sh 的 env 中导出 `GROOT_CKPT_ROOT=/dev/shm/ckpt_pin`。这是 tracked 代码改动，走 WA 流程；只影响之后新启动的 server，所以放在 arm 之间做。
- 风险：日志里的 `--checkpoint` 路径会变（provenance）；重启后 pin 丢失，必须保留回落逻辑；额外占用 15.2 GiB shm（仍在 20 GiB 上限内）。
- 替代方案（非 RAM）：把 ckpt 拷到 /home SSD，彻底绕开 HDD。但记忆中的约定是「ckpt 用 /data/ckpt/<名>」，所以同样需要 owner 裁定。

### A5 — 明确不做

- 不 pin p3 pilot 的 fit（每个只读一次）。
- 不 pin store 的 key/queries/img 数组（server 在线不读）。
- 不把本轮 token 数组拷进 /dev/shm（PCA 已完成；以后再跑时依靠释放出的 page cache，必要时开跑前顺序 `cat` 预热对应的 `tok/v{0,1}.npy`）。
- 不动 tables 和 pi05 ckpt（page cache 已处理）。
- 写入流留在 SSD。
- codex 收尾的 `pca_check`/`pca_validate`/`pca_smoke` 也读磁盘 store，只剩几分钟，不值得改。

---

## 5. 假设与局限

- 「排队中」的范围按以下来源认定：`line_P6A_v2.sh`、`line_P6C2.sh` 的 arm 列表；`r06_p3_pilot/pilot_queue.txt`（216 个 arm：65 DONE、2 CLAIMED、149 pending）；`r06_abl` 的 16 个 `r6p2_*` trigger arm。codex P2 的 `arms_pca.json` 还没有被 emit 到任何 run root，视为「计划中」。
- groot 启动次数和节省量是估算：排队的 arm 数 × 实测单次未缓存读量（3.6–5.3 GB）；两个 server 同时启动时共享一次读。
- 大文件是抽样 sha（头、尾加 6 个随机 1 MiB 块，两边同偏移），没有做全量 sha。小文件（≤ 64 MiB，402 个）全部做了全量 sha。
- `mincore` 驻留率是瞬时值，会随负载变化。
- 基于 base_fit/guard_fit 路径的启动读取没有逐一验证：server 从 artifact 反序列化 method，可能根本不读这些路径。因此第 5 项只作为「可选」。
- 没有读 `tests/review_tests/`。

### 附录：mincore 驻留率复查脚本

只读，不产生 page fault。

```bash
cat <<'EOF' | nice -n 19 ionice -c3 taskset -c 14-17,58-61 python3 - /data/ckpt/n15_libero_10/*.safetensors /data/ckpt/n15_libero_spatial/*.safetensors /data/libero_cache/libraries/libero_10/libero_10_sp16_S3.pkl /data/libero_cache/libraries/libero_spatial/libero_spatial_sp16_S3.pkl
import ctypes, os, mmap, sys
libc=ctypes.CDLL('libc.so.6',use_errno=True)
libc.mmap.restype=ctypes.c_void_p
libc.mmap.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_long]
libc.munmap.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
libc.mincore.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_char_p]
for p in sys.argv[1:]:
    sz=os.path.getsize(p); fd=os.open(p,os.O_RDONLY)
    a=libc.mmap(None,sz,mmap.PROT_READ,mmap.MAP_SHARED,fd,0); os.close(fd)
    v=ctypes.create_string_buffer((sz+4095)//4096); libc.mincore(a,sz,v); libc.munmap(a,sz)
    r=v.raw.count(b'\x01')*4096
    print(f'{r/sz*100:6.1f}%  {sz/1e9:6.2f} GB  {p}')
EOF
```
