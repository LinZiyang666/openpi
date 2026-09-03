# weilandserver 三层存储分层整理

**Status**: `Validated`（已完成并逐笔校验；owner 2026-09-03 确认归档，WA §5）
**Level**: L1（节点运维；`src/` 零改动，repo 内仅本 log）
**执行日期**: 2026-09-03
**节点**: `weilandserver`（tether nid，`ziyanglin.com` / 140.177.159.24）
**权威 ledger**: 远端 `/home/weiland/tier_migration/ledger.jsonl`（JSONL，逐笔追加）
**本文件**: 该 ledger 在本机的固化副本 + 决策依据。**日志留本机 repo，不留 weilandserver**（owner 2026-09-03 指令）。

---

## 0. 结果摘要

**45 笔移动 + 1 笔删除，全部通过四判据校验，零失败**（ledger 中 `FAIL` 计数为 0，45 个 `move_done` 对应 45 个 state marker）。

| 层 | 迁移前 | 迁移后 | 变化 |
|---|---|---|---|
| **T1 `/`** NVMe | 382.3 GiB (45%) | **215.0 GiB (25%)** | **−167.4 GiB** |
| **T2 `/data`** CMR | 1571.6 GiB (44%) | **291.8 GiB (9%)** | **−1279.8 GiB** |
| **T3 `/archive`** SMR | 66.3 GiB (1%) | **1282.3 GiB (10%)** | **+1216.0 GiB** |

分阶段：

| 阶段 | 方向 | 笔数 | 体量 | 实测吞吐 |
|---|---|---|---|---|
| Phase 0 | 删除 | 1 | 23.4 GiB | — |
| Phase 1 | T1 → T2 | 9 | 94.0 GiB | 219–536 MB/s |
| Phase 1b | T1 → T3 | 1 | 56.7 GiB | 233 MB/s |
| Phase 2 | T2 → T3 | 8 | 232.9 GiB | 110–145 MB/s |
| Phase 3 | T2 → T3 | 27 | 1030.6 GiB | 130–153 MB/s |

**软链穿透实测**（迁移后核对，不是设计断言）：`exp/robocasa365/data_symlink_to_data_disk/build_l1s1/pi05/CoffeeSetupMug` 经三级软链落到 `/archive/...`，列出 120 项，与迁移前抽样的文件数逐位吻合；`exp/ablation_study/data/checkpoints`、`run_so_101/models` 同样穿透正常。27 条 phase 3 软链全检 `symlink_ok=27 bad=0`，全机无 `.MIGRATING` 残留。

### 0.1 吞吐读数的两点说明

- **Phase 1 里 489 / 536 MB/s 的读数是 write-back 缓存假象**，不是 CMR 的真实写入速度（sda 外圈物理上限 171 MB/s）。本机 251 GiB 内存把小体量写入整个吸收了，rsync 在落盘前就返回。已在 Phase 1 结束后执行 `sync`（耗时 5m37s，exit 0）强制刷盘确认落地。Phase 2/3 的百 GiB 级体量远超缓存吸收能力，读数是真吞吐。
- **校验开销实测仅 ×1.11**（phase 3 前 10 笔的墙钟/拷贝比 ×1.07–×1.22）。执行中我曾口头估计"三到五成"，属高估，以此表为准。


---

## 1. 背景与目标

weilandserver 有三层存储，但项目文件长期只堆在第一层 SSD 上。目标是把不该占 NVMe 的字节下沉，同时**不改变任何消费方看到的路径** —— 每个被搬走的目录在原位留一个**同名软链**，制造"文件都还在 `/` 上"的错觉。

### 1.1 三层实测拓扑（2026-09-03 采集）

| 层 | 设备 | 挂载 | 容量 | 迁移前已用 | 物理特性 |
|---|---|---|---|---|---|
| **T1** | nvme0n1 Samsung 980 PRO | `/` ext4 `relatime` | 913G | 383G (45%) | 随机读写自由 |
| **T2** | sda ST4000NM0035 **CMR** | `/data` ext4 `noatime` | 3.6T | 1.6T (44%) | 顺序 133–171 MB/s，可覆盖写 |
| **T3** | sdb HGST HSH721414AL **HM-SMR** | `/archive` btrfs zoned + zstd:3 | 12.7T | 67G (**1%**) | 读同普通盘；随机写/覆盖写代价极高 |

T3 几乎全空，是全机最大的未动用资源。

### 1.2 分层判据

两条边界的性质**不同**，不能用同一把尺子：

- **T1 ↔ T2 看文件粒度**。小文件海 + 并发随机读必须留 NVMe；大文件顺序读可以下沉。
- **T2 ↔ T3 不看读性能**（SMR 与 CMR 读速同级，都是 HDD），只看**将来还会不会往里覆盖写 / 大量增删**。写一次读少次的就该进 T3。

据此的准入/否决规则：

- **恒留 T1**：git 工作树、所有 venv / conda env / site-packages、`.cache/uv` 与 `.cache/pip`（见 §2.3 硬链接陷阱）、MuJoCo 资产树、正在跑的随机写 scratch。
- **T1 → T2**：大块、顺序读、当前实验线仍在消费的数据（H5 库、pkl 库、ckpt、dump）。
- **T2 → T3**：已收官实验线的原始数据，只为复现/审计留存，永不覆盖写。

---

## 2. 逐项定级依据

### 2.1 实测文件粒度画像

定级不靠目录名，靠"文件数 / 平均大小"实测：

| 路径 | 大小 | 文件数 | 平均 | 定级 |
|---|---|---|---|---|
| `run_so_101/train` | 57G | 115 | 505 MB | 可下沉（纯大文件顺序读） |
| `ckpt_pi05_robocasa_pytorch` | 6.8G | **3** | 2.3 GB | 可下沉 |
| `ckpt_n15_robocasa{,_tp}` | 各 7.1G | 各 19 | 380 MB | 可下沉 |
| `ckpt_n15_libero_spatial` | 7.1G | 25 | 289 MB | 可下沉 |
| `ckpt_pi05_robocasa` | 12G | 61 | 194 MB | 可下沉 |
| `exp/common/data/db/libero_cache` | 13G | 102 | 121 MB | 可下沉 |
| `rl_router` | 41G | 50,283 | 848 KB | 可下沉至 T2 |
| `exp/rl_router/data` | 1.5G | 2,466 | 609 KB | 可下沉 |
| `exp/markov_sufficiency/data` | 319M | 54 | 5.9 MB | 可下沉 |

### 2.2 两项经实测**否决**的候选

初版计划把这两项列入 T1→T2，实测后撤销：

| 候选 | 实测 | 否决理由 |
|---|---|---|
| `Isaac-GR00T/gr00t/eval` 8.6G | 8.6G **全部**是 `sim/robocasa365/robocasa365_uv/.venv` | venv 铁律：小文件海 + `.so` mmap，HDD 上 import 延迟不可接受 |
| `Isaac-GR00T/external_dependencies/robocasa365` 24G | **123,504 个文件**（108,784 个 `.obj` + 5,764 png + 4,206 xml），平均 195 KB，`robocasa/models/assets` 占 23G | 这是 env init 时多 worker **并发随机小文件读**。正撞已知铁律：并发随机读在 HDD 上塌到 25 MB/s |

### 2.3 `.cache/uv` 不可搬（硬链接陷阱）

`.cache/uv` 26G 看似肥肉，实测 `archive-v0` 下 **112,681 个文件中有 108,697 个 link count > 1** —— 与 `openpi/.venv`、`lerobot_venv`、`gr00t_n15_venv` 是同 inode 硬链接。跨文件系统搬走会**打断硬链接**，结果不是省 26G 而是**多占约 24G**。同理 `.venv` 自身也不能单独搬。要瘦身只能 `uv cache prune`。

### 2.4 本轮**未动**的项

| 项 | 大小 | 未动原因 |
|---|---|---|
| `/tmp/dsp_shared` | 42G | owner 2026-09-03 明示"`/tmp` 不用管"。备案：其中 41G 是 `rit_pareto/*/h5` 原始 episode 池，`/data/rit_stage` 仅 2.3G 不构成备份，重启即失 |
| `/archive/VNAT_release_1.zip` (34.5G) 与解压目录 (34G) 并存 | 34.5G | 非本项目文件，且"删哪一份"取决于使用方式；删除不可逆，未获逐项确认前不动 |
| `/data/openpi/ablation_study/cache_size/artifacts` | 49G | pkl cache 库，分析仍在消费，留 T2 |
| `/data/libero_cache/libraries` | 38G | 16 个 pkl 库是下游真正的消费对象，留 T2 |
| `.cache/uv` | 26G | 见 §2.3 |

### 2.5 RoboCasa365 `build_l1s1` 1.1T —— 初版搁置，后经代码取证改判归档

首轮按"owner 未确认 L1S1 采集是否返工"的默认口径搁置。owner 2026-09-03 追问后做了代码级取证，**改判为归档**，编入 Phase 3。

改判的关键在于**换了判据**：初版顾虑是"采集会不会返工"，而真正该问的是"文件建成后还会不会被**原地改写**"。SMR 惩罚的是写，不是随机读 —— 随机读在 SMR 与 CMR 上同为寻道受限，不构成 T2/T3 之分。

取证三条：

1. **结构上 write-once**。写入侧 `src/openpi/collect/data_collector.py:137` 以 `h5py.File(tmp_path, "w")` 写 `.h5.tmp`，第 165 行 `tmp_path.rename(path)` 原子改名。消费侧遍历 `src/` `exp/` `examples/` 的全部 `h5py.File` 调用**无一例外是 `"r"`**；仅有的 `"a"` / `"r+"` 位于 `tests/robocasa365/` 与 `tests/exp/` 的 tmp fixture。SMR 的原地覆盖写惩罚在这条路径上不可达。
2. **重采是新建文件而非改写**。命名含微秒时间戳（`episode_{id:04d}_{ts}.h5`）且走 tmp+rename，补采落到 zoned btrfs 是新 zone 顺序写。这一条直接消解了初版的搁置理由。
3. **粒度全机最优**。抽样 `build_l1s1/pi05/CoffeeSetupMug`：120 文件 / 47.8 GiB / **平均 407.7 MiB**（min 270.5 / max 443.8）。对比已归档的 ablation `collect_h5`（1,000 文件 / 131.2 GiB / 平均 134 MiB）粒度大三倍。全库无 `.h5.tmp` 残留 ⇒ 无半截写入。

两处必须写明的预期修正：

- **无压缩红利**。h5 datasets 建时已带 `compression="lzf"`，btrfs zstd:3 压在已压过的数据上基本收不到东西。按 1.1T 原样落盘计算，不要按"叠瓦+压缩"省空间。
- **残留风险是"删"不是"写"**。SMR 大批删除要等 zone 回收。唯一会疼的场景是将来成批剔除 / 重采同一批 episode；纯追加或只读消费无碍。

`cache_artifacts_text_ivf`（47G）是下游真正消费的 pkl，留 T2 不动。


---

## 3. 迁移机制

脚本：远端 `/home/weiland/tier_migration/tier_migrate.sh`（全文见 §6）。

### 3.1 不变量

1. **单流串行**，全局 `flock` 把关。依据是已固化的 HDD 铁律：并发顺序流互相穿插会塌到 37 MB/s（占空比 ~4%，util 高但 MB/s 低），单一顺序读者可贴 135 MB/s 单流上限。
2. **同名软链回填**：`mv src src.MIGRATING` → `ln -s dst src` → 读穿校验 → 才 `rm -rf src.MIGRATING`。任一步失败即回滚原目录。消费方路径零变化；已有的 `exp/ablation_study/data/* → /data/...` 软链在目标再次被搬后自然形成二级链，**无需重指**。
3. **源目录只在四判据全过后才删**。
4. **全程不用 `du` 做校验** —— 它对硬链接去重，会产生假失配。

### 3.2 四判据

| 判据 | 内容 |
|---|---|
| **V1** | 普通文件数两侧一致（`find -type f \| wc -l`） |
| **V2** | 普通文件字节和两侧一致（`find -printf '%s\n' \| awk` 求和，**非 `du`**） |
| **V3** | `rsync -aHAXn --itemize-changes` 输出为空（无待传项） |
| **V4** | sha256 抽样一致，抽样**刻意超配 link count > 1 的文件**以覆盖硬链接族（6 个多链文件 + 12 个随机文件，去重后逐个比对） |

### 3.3 其它

- `rsync -aHAX`：`-H` 保硬链接族，`-AX` 带 ACL/xattr。**不用 `--inplace`** —— SMR 必须收到写入新 extent 的顺序流。
- 目标盘空间预检（需要 `bytes × 1.02`）不足即 FAIL 且不动源。
- 断点续跑：每笔完成后落 `state/<slug>.done`，重跑自动跳过。
- 源是软链 / 源不存在 → 跳过，不报错。

---

## 4. 逐笔移动 ledger

> 权威副本：远端 `/home/weiland/tier_migration/ledger.jsonl`。下表由该文件固化而来。
> `MB/s` 是 rsync 拷贝段实测吞吐（不含校验段）。

### Phase 0 — 纯删除（不搬）

| 时间 (UTC) | 路径 | 字节 | 文件数 | 方式 | 依据 |
|---|---|---|---|---|---|
| 2026-09-03T15:42:11Z → 15:42:18Z | `/home/weiland/.cache/go-build` | 25,159,862,929 (23.4 GiB) | 89,579 | `go clean -cache` | tether 的 Go 构建缓存，可重建；构建缓存放 HDD 等于自杀，故删不搬 |

`/` avail: 518,919,749,632 → 544,411,361,280（**+23.7 GiB**）

### Phase 1 — T1 NVMe → T2 CMR

<!-- LEDGER:PHASE1 -->
| 时间 (UTC) | 源 (T1) | 目标 (T2) | 大小 | 文件数 | 耗时 | 吞吐 | V1 | V2 | V3 | V4 抽样 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-03T15:45:31Z | `/home/weiland/rl_router` | `/data/rl_router` | 40.7 GiB | 50,283 | 177s | 235 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T15:46:47Z | `/home/weiland/ckpt_pi05_robocasa` | `/data/ckpt/pi05_robocasa` | 11.6 GiB | 61 | 54s | 219 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T15:47:34Z | `/home/weiland/ckpt_n15_robocasa_tp` | `/data/ckpt/n15_robocasa_tp` | 7.1 GiB | 19 | 31s | 233 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T15:48:32Z | `/home/weiland/ckpt_n15_robocasa` | `/data/ckpt/n15_robocasa` | 7.1 GiB | 19 | 33s | 219 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T15:49:05Z | `/home/weiland/ckpt_n15_libero_spatial` | `/data/ckpt/n15_libero_spatial` | 7.1 GiB | 25 | 32s | 226 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T15:50:11Z | `/home/weiland/ckpt_pi05_robocasa_pytorch` | `/data/ckpt/pi05_robocasa_pytorch` | 6.7 GiB | 3 | 31s | 222 MB/s | ✅ | ✅ | ✅ | 3 文件 |
| 2026-09-03T15:50:47Z | `/home/weiland/openpi/exp/common/data/db/libero_cache` | `/data/openpi/exp_common/db/libero_cache` | 12.0 GiB | 102 | 23s | 536 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T15:50:51Z | `/home/weiland/openpi/exp/rl_router/data` | `/data/openpi/exp_rl_router/data` | 1.4 GiB | 2,466 | 3s | 489 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T15:50:52Z | `/home/weiland/openpi/exp/markov_sufficiency/data` | `/data/openpi/exp_markov_sufficiency/data` | 0.3 GiB | 54 | 1s | 317 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| **合计** | | | **94.0 GiB** | **53,032** | | | | | | |


### Phase 1b — T1 NVMe → T3 SMR

<!-- LEDGER:PHASE1B -->
| 时间 (UTC) | 源 (T1) | 目标 (T3) | 大小 | 文件数 | 耗时 | 吞吐 | V1 | V2 | V3 | V4 抽样 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-03T15:55:05Z | `/home/weiland/run_so_101/train` | `/archive/run_so_101/train` | 56.7 GiB | 115 | 248s | 233 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| **合计** | | | **56.7 GiB** | **115** | | | | | | |


### Phase 2 — T2 CMR → T3 SMR

<!-- LEDGER:PHASE2 -->
| 时间 (UTC) | 源 (T2) | 目标 (T3) | 大小 | 文件数 | 耗时 | 吞吐 | V1 | V2 | V3 | V4 抽样 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-03T16:14:43Z | `/data/openpi/ablation_study/cache_size/collect_h5` | `/archive/openpi/ablation_study/cache_size/collect_h5` | 131.2 GiB | 1,000 | 1145s | 117 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T16:16:33Z | `/data/openpi/ablation_study/cache_size/save_traj` | `/archive/openpi/ablation_study/cache_size/save_traj` | 11.5 GiB | 1,000 | 107s | 110 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T16:25:58Z | `/data/openpi/ablation_study/executor_substitution/checkpoints` | `/archive/openpi/ablation_study/executor_substitution/checkpoints` | 67.9 GiB | 514 | 552s | 125 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T16:27:43Z | `/data/openpi/ablation_study/executor_substitution/distill_raw` | `/archive/openpi/ablation_study/executor_substitution/distill_raw` | 11.7 GiB | 1,004 | 102s | 117 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T16:29:22Z | `/data/openpi/ablation_study/executor_substitution/lerobot` | `/archive/openpi/ablation_study/executor_substitution/lerobot` | 10.6 GiB | 121,062 | 89s | 121 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T16:37:28Z | `/data/libero_cache/build_libero10` | `/archive/libero_cache/build_libero10` | 62.9 GiB | 507 | 468s | 137 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T16:40:46Z | `/data/libero_cache/build_spatial` | `/archive/libero_cache/build_spatial` | 25.1 GiB | 507 | 192s | 133 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T16:44:13Z | `/data/run_so_101/models` | `/archive/run_so_101/models` | 22.2 GiB | 21 | 156s | 145 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| **合计** | | | **343.1 GiB** | **125,615** | | | | | | |


### Phase 3 — T2 CMR → T3 SMR（RoboCasa365 `build_l1s1`，见 §2.5）

由**独立脚本** `tier_migrate3.sh` 执行，**不是**改 `tier_migrate.sh`：bash 按文件偏移惰性读取源码，原地修改一个正在执行的脚本会让它跳到垃圾字节上。v3 与 v1 的差异只有三处 —— 阻塞式 `flock`（故意排队等前一轮释放锁，单流不变量不破）、运行时枚举的 `build_phase3()` 移动表、`phase3` case 分支。

移动表按**任务目录逐个**枚举（13 任务 × 2 teacher = 26 笔 + `_archive_t3_manual` = **27 笔**），而非整块 1.1T 一笔：5–6 小时的作业，中途失败只需重做一个任务。

<!-- LEDGER:PHASE3 -->
| 时间 (UTC) | 源 (T2) | 目标 (T3) | 大小 | 文件数 | 耗时 | 吞吐 | V1 | V2 | V3 | V4 抽样 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-03T17:16:57Z | `/data/robocasa365_cache/build_l1s1/pi05/CloseBlenderLid` | `/archive/robocasa365_cache/build_l1s1/pi05/CloseBlenderLid` | 249.0 GiB | 397 | 1836s | 138 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T17:23:56Z | `/data/robocasa365_cache/build_l1s1/pi05/CloseFridge` | `/archive/robocasa365_cache/build_l1s1/pi05/CloseFridge` | 48.9 GiB | 92 | 377s | 132 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T17:30:34Z | `/data/robocasa365_cache/build_l1s1/pi05/CoffeeSetupMug` | `/archive/robocasa365_cache/build_l1s1/pi05/CoffeeSetupMug` | 47.8 GiB | 120 | 364s | 134 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T17:35:35Z | `/data/robocasa365_cache/build_l1s1/pi05/OpenCabinet` | `/archive/robocasa365_cache/build_l1s1/pi05/OpenCabinet` | 32.7 GiB | 67 | 257s | 130 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T17:38:44Z | `/data/robocasa365_cache/build_l1s1/pi05/OpenDrawer` | `/archive/robocasa365_cache/build_l1s1/pi05/OpenDrawer` | 25.0 GiB | 83 | 167s | 153 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T17:44:43Z | `/data/robocasa365_cache/build_l1s1/pi05/OpenStandMixerHead` | `/archive/robocasa365_cache/build_l1s1/pi05/OpenStandMixerHead` | 49.4 GiB | 177 | 332s | 152 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T17:48:28Z | `/data/robocasa365_cache/build_l1s1/pi05/PickPlaceCounterToCabinet` | `/archive/robocasa365_cache/build_l1s1/pi05/PickPlaceCounterToCabinet` | 27.2 GiB | 76 | 186s | 149 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T17:51:07Z | `/data/robocasa365_cache/build_l1s1/pi05/PickPlaceCounterToStove` | `/archive/robocasa365_cache/build_l1s1/pi05/PickPlaceCounterToStove` | 19.0 GiB | 66 | 130s | 149 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T17:57:21Z | `/data/robocasa365_cache/build_l1s1/pi05/PickPlaceDrawerToCounter` | `/archive/robocasa365_cache/build_l1s1/pi05/PickPlaceDrawerToCounter` | 47.2 GiB | 107 | 330s | 146 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T17:59:49Z | `/data/robocasa365_cache/build_l1s1/pi05/PickPlaceSinkToCounter` | `/archive/robocasa365_cache/build_l1s1/pi05/PickPlaceSinkToCounter` | 17.6 GiB | 56 | 121s | 148 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T18:06:18Z | `/data/robocasa365_cache/build_l1s1/pi05/PickPlaceToasterToCounter` | `/archive/robocasa365_cache/build_l1s1/pi05/PickPlaceToasterToCounter` | 51.5 GiB | 129 | 352s | 149 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T18:09:27Z | `/data/robocasa365_cache/build_l1s1/pi05/SlideDishwasherRack` | `/archive/robocasa365_cache/build_l1s1/pi05/SlideDishwasherRack` | 24.5 GiB | 106 | 167s | 150 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T18:12:21Z | `/data/robocasa365_cache/build_l1s1/pi05/TurnOnSinkFaucet` | `/archive/robocasa365_cache/build_l1s1/pi05/TurnOnSinkFaucet` | 21.6 GiB | 63 | 147s | 150 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T18:15:38Z | `/data/robocasa365_cache/build_l1s1/groot_tp/CloseBlenderLid` | `/archive/robocasa365_cache/build_l1s1/groot_tp/CloseBlenderLid` | 25.0 GiB | 72 | 169s | 151 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T18:22:44Z | `/data/robocasa365_cache/build_l1s1/groot_tp/CloseFridge` | `/archive/robocasa365_cache/build_l1s1/groot_tp/CloseFridge` | 57.4 GiB | 124 | 385s | 152 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T18:25:55Z | `/data/robocasa365_cache/build_l1s1/groot_tp/CoffeeSetupMug` | `/archive/robocasa365_cache/build_l1s1/groot_tp/CoffeeSetupMug` | 25.1 GiB | 89 | 169s | 152 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T18:28:56Z | `/data/robocasa365_cache/build_l1s1/groot_tp/OpenCabinet` | `/archive/robocasa365_cache/build_l1s1/groot_tp/OpenCabinet` | 22.0 GiB | 66 | 149s | 150 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T18:31:28Z | `/data/robocasa365_cache/build_l1s1/groot_tp/OpenDrawer` | `/archive/robocasa365_cache/build_l1s1/groot_tp/OpenDrawer` | 19.2 GiB | 68 | 130s | 151 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T18:32:36Z | `/data/robocasa365_cache/build_l1s1/groot_tp/OpenStandMixerHead` | `/archive/robocasa365_cache/build_l1s1/groot_tp/OpenStandMixerHead` | 7.9 GiB | 68 | 55s | 147 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T18:35:59Z | `/data/robocasa365_cache/build_l1s1/groot_tp/PickPlaceCounterToCabinet` | `/archive/robocasa365_cache/build_l1s1/groot_tp/PickPlaceCounterToCabinet` | 25.5 GiB | 86 | 172s | 151 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T18:37:58Z | `/data/robocasa365_cache/build_l1s1/groot_tp/PickPlaceCounterToStove` | `/archive/robocasa365_cache/build_l1s1/groot_tp/PickPlaceCounterToStove` | 14.1 GiB | 66 | 98s | 147 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T18:42:04Z | `/data/robocasa365_cache/build_l1s1/groot_tp/PickPlaceDrawerToCounter` | `/archive/robocasa365_cache/build_l1s1/groot_tp/PickPlaceDrawerToCounter` | 30.9 GiB | 89 | 210s | 150 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T18:44:58Z | `/data/robocasa365_cache/build_l1s1/groot_tp/PickPlaceSinkToCounter` | `/archive/robocasa365_cache/build_l1s1/groot_tp/PickPlaceSinkToCounter` | 20.1 GiB | 64 | 137s | 150 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T18:47:58Z | `/data/robocasa365_cache/build_l1s1/groot_tp/PickPlaceToasterToCounter` | `/archive/robocasa365_cache/build_l1s1/groot_tp/PickPlaceToasterToCounter` | 21.7 GiB | 76 | 151s | 147 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T18:50:38Z | `/data/robocasa365_cache/build_l1s1/groot_tp/SlideDishwasherRack` | `/archive/robocasa365_cache/build_l1s1/groot_tp/SlideDishwasherRack` | 19.9 GiB | 99 | 137s | 149 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T19:00:10Z | `/data/robocasa365_cache/build_l1s1/groot_tp/TurnOnSinkFaucet` | `/archive/robocasa365_cache/build_l1s1/groot_tp/TurnOnSinkFaucet` | 77.6 GiB | 233 | 529s | 150 MB/s | ✅ | ✅ | ✅ | 12 文件 |
| 2026-09-03T19:00:54Z | `/data/robocasa365_cache/_archive_t3_manual` | `/archive/robocasa365_cache/_archive_t3_manual` | 2.8 GiB | 6 | 25s | 113 MB/s | ✅ | ✅ | ✅ | 6 文件 |
| **合计** | | | **1030.6 GiB** | **2,745** | | | | | | |


---

## 5. 迁移前后账

<!-- LEDGER:DF -->
| 层 | 迁移前已用 | 迁移后已用 | Δ | 迁移前可用 | 迁移后可用 |
|---|---|---|---|---|---|
| T1 `/` NVMe | 358.9 GiB | 215.0 GiB | **-144.0 GiB** | 507.0 GiB | 651.0 GiB |

> ⚠ 上表 T1 的「迁移前」= **358.9 GiB**，取自第一轮 run 启动瞬间，此时 §4 Phase 0 的 23.4 GiB 已被删除。**真实基线是 382.3 GiB**（2026-09-03 会话开始时 `df -h` 显示 383G / 45%）。
> 计入 Phase 0 后，T1 的实际净减为 **−167.4 GiB（382.3 → 215.0 GiB，45% → 25%）**。
> T2 / T3 两行不受影响：Phase 0 只动 T1。
| T2 `/data` CMR | 1571.6 GiB | 291.8 GiB | **-1279.8 GiB** | 2057.6 GiB | 3337.4 GiB |
| T3 `/archive` SMR | 66.3 GiB | 1282.3 GiB | **+1216.0 GiB** | 12971.4 GiB | 11752.8 GiB |


---

## 6. 迁移脚本全文

<!-- SCRIPT -->
### 6.1 `tier_migrate.sh`（phase 0–2）

```bash
#!/usr/bin/env bash
#
# tier_migrate.sh -- serialized, verified, resumable storage tiering for weilandserver.
#
# Tier model
#   T1  /         NVMe 980 PRO   small-file / random-read working set (venvs, repos, sim assets)
#   T2  /data     CMR  sda       large sequential-read data that is still consumed
#   T3  /archive  SMR  sdb       write-once / read-seldom archive; NO in-place rewrite
#
# Invariants
#   * One move at a time, under a global flock: concurrent sequential streams on a
#     rotational disk collapse to ~37 MB/s, a single stream sustains ~135 MB/s.
#   * Every moved directory is replaced by a same-name symlink at its original path,
#     so no consumer path ever changes and symlink chains resolve transparently.
#   * The source is deleted ONLY after all four verification criteria pass.
#   * `du` is never used for verification: it de-duplicates hardlinks and would
#     report a false mismatch. V2 sums per-file sizes instead.
#
# Verification criteria
#   V1  regular-file count identical on both sides
#   V2  sum of regular-file sizes identical on both sides
#   V3  `rsync -n --itemize-changes` finds nothing left to transfer
#   V4  sha256 identical over a sample that over-weights multiply-linked files
#
# Usage:  tier_migrate.sh [phase1|phase1b|phase2|all]
#
set -uo pipefail

ROOT=/home/weiland/tier_migration
LEDGER="$ROOT/ledger.jsonl"
STATE="$ROOT/state"
LOGS="$ROOT/rsync_logs"
LOCK="$ROOT/.hdd.lock"
SAMPLE_LINKED=6
SAMPLE_RANDOM=12

mkdir -p "$ROOT" "$STATE" "$LOGS"

# ------------------------------------------------------------------
# Ledger
# ------------------------------------------------------------------

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

emit() {
  local ev="$1"; shift
  local line="{\"ts\":\"$(ts)\",\"event\":\"$ev\""
  local kv k v
  for kv in "$@"; do
    k="${kv%%=*}"; v="${kv#*=}"
    line+=",\"$k\":\"$v\""
  done
  printf '%s}\n' "$line" >> "$LEDGER"
  printf '[%s] %s %s\n' "$(ts)" "$ev" "$*"
}

# ------------------------------------------------------------------
# Measurement helpers
# ------------------------------------------------------------------

count_files() { find "$1" -xdev -type f 2>/dev/null | wc -l; }
sum_bytes()   { find "$1" -xdev -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{printf "%d", s+0}'; }
avail_bytes() { df -B1 --output=avail "$1" 2>/dev/null | tail -1 | tr -d ' '; }

slug() { printf '%s' "${1#/}" | tr '/' '_'; }

# ------------------------------------------------------------------
# One verified move
# ------------------------------------------------------------------

do_move() {
  local SRC="$1" DST="$2"
  local id; id="$(slug "$SRC")"
  local marker="$STATE/$id.done"
  local rlog="$LOGS/$id.rsync.log"

  if [[ -f "$marker" ]]; then
    emit skip id="$id" reason=already_done
    return 0
  fi
  if [[ -L "$SRC" ]]; then
    emit skip id="$id" reason=src_already_symlink
    return 0
  fi
  if [[ ! -d "$SRC" ]]; then
    emit skip id="$id" reason=src_missing
    return 0
  fi

  local sfiles sbytes davail
  sfiles="$(count_files "$SRC")"
  sbytes="$(sum_bytes "$SRC")"

  mkdir -p "$(dirname "$DST")"
  davail="$(avail_bytes "$(dirname "$DST")")"
  if (( davail < sbytes + sbytes / 50 )); then
    emit FAIL id="$id" reason=dst_insufficient_space need="$sbytes" avail="$davail"
    return 1
  fi

  emit move_start id="$id" src="$SRC" dst="$DST" files="$sfiles" bytes="$sbytes"
  local t0; t0="$(date +%s)"

  # -H preserves hardlink families; -AX carry ACLs / xattrs.
  # No --inplace: SMR must receive plain sequential writes into fresh extents.
  if ! rsync -aHAX --info=stats2 "$SRC/" "$DST/" > "$rlog" 2>&1; then
    emit FAIL id="$id" reason=rsync_nonzero log="$rlog"
    return 1
  fi

  local t1 secs mbps
  t1="$(date +%s)"; secs=$(( t1 - t0 )); (( secs == 0 )) && secs=1
  mbps=$(( sbytes / secs / 1048576 ))
  emit copy_done id="$id" seconds="$secs" mb_per_s="$mbps"

  # ---- V1 file count ----
  local dfiles; dfiles="$(count_files "$DST")"
  if [[ "$sfiles" != "$dfiles" ]]; then
    emit FAIL id="$id" reason=V1_file_count src="$sfiles" dst="$dfiles"
    return 1
  fi

  # ---- V2 byte sum (never du: it de-duplicates hardlinks) ----
  local dbytes; dbytes="$(sum_bytes "$DST")"
  if [[ "$sbytes" != "$dbytes" ]]; then
    emit FAIL id="$id" reason=V2_byte_sum src="$sbytes" dst="$dbytes"
    return 1
  fi

  # ---- V3 rsync dry-run must find nothing to transfer ----
  local pending
  pending="$(rsync -aHAXn --itemize-changes "$SRC/" "$DST/" 2>/dev/null | grep -vc '^$')"
  if [[ "$pending" != "0" ]]; then
    emit FAIL id="$id" reason=V3_itemize_nonempty pending="$pending"
    rsync -aHAXn --itemize-changes "$SRC/" "$DST/" 2>/dev/null | head -40 > "$LOGS/$id.itemize.txt"
    return 1
  fi

  # ---- V4 sha256 sample, over-weighting hardlink families ----
  local -a sample
  mapfile -t sample < <(
    {
      find "$SRC" -xdev -type f -links +1 -printf '%P\n' 2>/dev/null | head -"$SAMPLE_LINKED"
      find "$SRC" -xdev -type f -printf '%P\n' 2>/dev/null | shuf -n "$SAMPLE_RANDOM"
    } | sort -u
  )
  local rel a b checked=0
  for rel in "${sample[@]}"; do
    [[ -z "$rel" ]] && continue
    a="$(sha256sum "$SRC/$rel" 2>/dev/null | cut -d' ' -f1)"
    b="$(sha256sum "$DST/$rel" 2>/dev/null | cut -d' ' -f1)"
    if [[ -z "$a" || "$a" != "$b" ]]; then
      emit FAIL id="$id" reason=V4_sha256 rel="$rel"
      return 1
    fi
    checked=$(( checked + 1 ))
  done
  emit verified id="$id" v1_files="$dfiles" v2_bytes="$dbytes" v3_pending=0 v4_sampled="$checked"

  # ---- swap in the symlink, then drop the source ----
  if ! mv "$SRC" "$SRC.MIGRATING"; then
    emit FAIL id="$id" reason=rename_failed
    return 1
  fi
  if ! ln -s "$DST" "$SRC"; then
    mv "$SRC.MIGRATING" "$SRC"
    emit FAIL id="$id" reason=symlink_failed
    return 1
  fi
  # read-through check before the source is destroyed
  if [[ "$(readlink -f "$SRC")" != "$(readlink -f "$DST")" ]] || [[ ! -d "$SRC/" ]]; then
    rm -f "$SRC"; mv "$SRC.MIGRATING" "$SRC"
    emit FAIL id="$id" reason=symlink_readthrough
    return 1
  fi
  rm -rf "$SRC.MIGRATING"

  touch "$marker"
  emit move_done id="$id" src="$SRC" dst="$DST" files="$dfiles" bytes="$dbytes" seconds="$secs" mb_per_s="$mbps"
  return 0
}

run_table() {
  local phase="$1"; shift
  emit phase_start phase="$phase"
  local row src dst rc=0
  for row in "$@"; do
    src="${row%%|*}"; dst="${row##*|}"
    do_move "$src" "$dst" || { rc=1; emit phase_abort phase="$phase" at="$src"; break; }
  done
  emit phase_end phase="$phase" rc="$rc"
  return "$rc"
}

# ------------------------------------------------------------------
# Move tables
# ------------------------------------------------------------------

PHASE1=(  # T1 NVMe -> T2 CMR : large sequential-read data still in use
  "/home/weiland/rl_router|/data/rl_router"
  "/home/weiland/ckpt_pi05_robocasa|/data/ckpt/pi05_robocasa"
  "/home/weiland/ckpt_n15_robocasa_tp|/data/ckpt/n15_robocasa_tp"
  "/home/weiland/ckpt_n15_robocasa|/data/ckpt/n15_robocasa"
  "/home/weiland/ckpt_n15_libero_spatial|/data/ckpt/n15_libero_spatial"
  "/home/weiland/ckpt_pi05_robocasa_pytorch|/data/ckpt/pi05_robocasa_pytorch"
  "/home/weiland/openpi/exp/common/data/db/libero_cache|/data/openpi/exp_common/db/libero_cache"
  "/home/weiland/openpi/exp/rl_router/data|/data/openpi/exp_rl_router/data"
  "/home/weiland/openpi/exp/markov_sufficiency/data|/data/openpi/exp_markov_sufficiency/data"
)

PHASE1B=( # T1 NVMe -> T3 SMR : cold since 2026-07, 115 files averaging 505 MB
  "/home/weiland/run_so_101/train|/archive/run_so_101/train"
)

PHASE2=(  # T2 CMR -> T3 SMR : finished lines, raw data kept for reproducibility only
  "/data/openpi/ablation_study/cache_size/collect_h5|/archive/openpi/ablation_study/cache_size/collect_h5"
  "/data/openpi/ablation_study/cache_size/save_traj|/archive/openpi/ablation_study/cache_size/save_traj"
  "/data/openpi/ablation_study/executor_substitution/checkpoints|/archive/openpi/ablation_study/executor_substitution/checkpoints"
  "/data/openpi/ablation_study/executor_substitution/distill_raw|/archive/openpi/ablation_study/executor_substitution/distill_raw"
  "/data/openpi/ablation_study/executor_substitution/lerobot|/archive/openpi/ablation_study/executor_substitution/lerobot"
  "/data/libero_cache/build_libero10|/archive/libero_cache/build_libero10"
  "/data/libero_cache/build_spatial|/archive/libero_cache/build_spatial"
  "/data/run_so_101/models|/archive/run_so_101/models"
)

# ------------------------------------------------------------------
# Entry
# ------------------------------------------------------------------

main() {
  local what="${1:-all}"
  emit run_start what="$what" host="$(hostname)"
  df -B1 --output=target,used,avail / /data /archive | tail -3 | while read -r t u a; do
    emit df_before target="$t" used="$u" avail="$a"
  done

  case "$what" in
    phase1)  run_table phase1  "${PHASE1[@]}" ;;
    phase1b) run_table phase1b "${PHASE1B[@]}" ;;
    phase2)  run_table phase2  "${PHASE2[@]}" ;;
    all)
      run_table phase1  "${PHASE1[@]}"  &&
      run_table phase1b "${PHASE1B[@]}" &&
      run_table phase2  "${PHASE2[@]}"
      ;;
    *) emit FAIL reason=unknown_phase what="$what"; return 2 ;;
  esac
  local rc=$?

  df -B1 --output=target,used,avail / /data /archive | tail -3 | while read -r t u a; do
    emit df_after target="$t" used="$u" avail="$a"
  done
  emit run_end what="$what" rc="$rc"
  return "$rc"
}

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another tier_migrate.sh holds the lock; refusing to run a second stream" >&2
  exit 3
fi

main "${1:-all}"
```

### 6.2 `tier_migrate3.sh`（phase 3）

与 6.1 的差异仅三处：阻塞式 `flock`、运行时枚举的 `build_phase3()`、`phase3` case 分支。

> 下方脚本是**实际执行的原文**，未作事后修饰。因此它的头部注释仍沿用 v1 的文件名与 `Usage: tier_migrate.sh [phase1|phase1b|phase2|all]` —— 这行 usage 与实际不符（本文件接受 `phase3`）。保留原样是为了让本记录与跑过的字节一致；修正版留待下次复用该脚本时再改。

```bash
#!/usr/bin/env bash
#
# tier_migrate.sh -- serialized, verified, resumable storage tiering for weilandserver.
#
# Tier model
#   T1  /         NVMe 980 PRO   small-file / random-read working set (venvs, repos, sim assets)
#   T2  /data     CMR  sda       large sequential-read data that is still consumed
#   T3  /archive  SMR  sdb       write-once / read-seldom archive; NO in-place rewrite
#
# Invariants
#   * One move at a time, under a global flock: concurrent sequential streams on a
#     rotational disk collapse to ~37 MB/s, a single stream sustains ~135 MB/s.
#   * Every moved directory is replaced by a same-name symlink at its original path,
#     so no consumer path ever changes and symlink chains resolve transparently.
#   * The source is deleted ONLY after all four verification criteria pass.
#   * `du` is never used for verification: it de-duplicates hardlinks and would
#     report a false mismatch. V2 sums per-file sizes instead.
#
# Verification criteria
#   V1  regular-file count identical on both sides
#   V2  sum of regular-file sizes identical on both sides
#   V3  `rsync -n --itemize-changes` finds nothing left to transfer
#   V4  sha256 identical over a sample that over-weights multiply-linked files
#
# Usage:  tier_migrate.sh [phase1|phase1b|phase2|all]
#
set -uo pipefail

ROOT=/home/weiland/tier_migration
LEDGER="$ROOT/ledger.jsonl"
STATE="$ROOT/state"
LOGS="$ROOT/rsync_logs"
LOCK="$ROOT/.hdd.lock"
SAMPLE_LINKED=6
SAMPLE_RANDOM=12

mkdir -p "$ROOT" "$STATE" "$LOGS"

# ------------------------------------------------------------------
# Ledger
# ------------------------------------------------------------------

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

emit() {
  local ev="$1"; shift
  local line="{\"ts\":\"$(ts)\",\"event\":\"$ev\""
  local kv k v
  for kv in "$@"; do
    k="${kv%%=*}"; v="${kv#*=}"
    line+=",\"$k\":\"$v\""
  done
  printf '%s}\n' "$line" >> "$LEDGER"
  printf '[%s] %s %s\n' "$(ts)" "$ev" "$*"
}

# ------------------------------------------------------------------
# Measurement helpers
# ------------------------------------------------------------------

count_files() { find "$1" -xdev -type f 2>/dev/null | wc -l; }
sum_bytes()   { find "$1" -xdev -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{printf "%d", s+0}'; }
avail_bytes() { df -B1 --output=avail "$1" 2>/dev/null | tail -1 | tr -d ' '; }

slug() { printf '%s' "${1#/}" | tr '/' '_'; }

# ------------------------------------------------------------------
# One verified move
# ------------------------------------------------------------------

do_move() {
  local SRC="$1" DST="$2"
  local id; id="$(slug "$SRC")"
  local marker="$STATE/$id.done"
  local rlog="$LOGS/$id.rsync.log"

  if [[ -f "$marker" ]]; then
    emit skip id="$id" reason=already_done
    return 0
  fi
  if [[ -L "$SRC" ]]; then
    emit skip id="$id" reason=src_already_symlink
    return 0
  fi
  if [[ ! -d "$SRC" ]]; then
    emit skip id="$id" reason=src_missing
    return 0
  fi

  local sfiles sbytes davail
  sfiles="$(count_files "$SRC")"
  sbytes="$(sum_bytes "$SRC")"

  mkdir -p "$(dirname "$DST")"
  davail="$(avail_bytes "$(dirname "$DST")")"
  if (( davail < sbytes + sbytes / 50 )); then
    emit FAIL id="$id" reason=dst_insufficient_space need="$sbytes" avail="$davail"
    return 1
  fi

  emit move_start id="$id" src="$SRC" dst="$DST" files="$sfiles" bytes="$sbytes"
  local t0; t0="$(date +%s)"

  # -H preserves hardlink families; -AX carry ACLs / xattrs.
  # No --inplace: SMR must receive plain sequential writes into fresh extents.
  if ! rsync -aHAX --info=stats2 "$SRC/" "$DST/" > "$rlog" 2>&1; then
    emit FAIL id="$id" reason=rsync_nonzero log="$rlog"
    return 1
  fi

  local t1 secs mbps
  t1="$(date +%s)"; secs=$(( t1 - t0 )); (( secs == 0 )) && secs=1
  mbps=$(( sbytes / secs / 1048576 ))
  emit copy_done id="$id" seconds="$secs" mb_per_s="$mbps"

  # ---- V1 file count ----
  local dfiles; dfiles="$(count_files "$DST")"
  if [[ "$sfiles" != "$dfiles" ]]; then
    emit FAIL id="$id" reason=V1_file_count src="$sfiles" dst="$dfiles"
    return 1
  fi

  # ---- V2 byte sum (never du: it de-duplicates hardlinks) ----
  local dbytes; dbytes="$(sum_bytes "$DST")"
  if [[ "$sbytes" != "$dbytes" ]]; then
    emit FAIL id="$id" reason=V2_byte_sum src="$sbytes" dst="$dbytes"
    return 1
  fi

  # ---- V3 rsync dry-run must find nothing to transfer ----
  local pending
  pending="$(rsync -aHAXn --itemize-changes "$SRC/" "$DST/" 2>/dev/null | grep -vc '^$')"
  if [[ "$pending" != "0" ]]; then
    emit FAIL id="$id" reason=V3_itemize_nonempty pending="$pending"
    rsync -aHAXn --itemize-changes "$SRC/" "$DST/" 2>/dev/null | head -40 > "$LOGS/$id.itemize.txt"
    return 1
  fi

  # ---- V4 sha256 sample, over-weighting hardlink families ----
  local -a sample
  mapfile -t sample < <(
    {
      find "$SRC" -xdev -type f -links +1 -printf '%P\n' 2>/dev/null | head -"$SAMPLE_LINKED"
      find "$SRC" -xdev -type f -printf '%P\n' 2>/dev/null | shuf -n "$SAMPLE_RANDOM"
    } | sort -u
  )
  local rel a b checked=0
  for rel in "${sample[@]}"; do
    [[ -z "$rel" ]] && continue
    a="$(sha256sum "$SRC/$rel" 2>/dev/null | cut -d' ' -f1)"
    b="$(sha256sum "$DST/$rel" 2>/dev/null | cut -d' ' -f1)"
    if [[ -z "$a" || "$a" != "$b" ]]; then
      emit FAIL id="$id" reason=V4_sha256 rel="$rel"
      return 1
    fi
    checked=$(( checked + 1 ))
  done
  emit verified id="$id" v1_files="$dfiles" v2_bytes="$dbytes" v3_pending=0 v4_sampled="$checked"

  # ---- swap in the symlink, then drop the source ----
  if ! mv "$SRC" "$SRC.MIGRATING"; then
    emit FAIL id="$id" reason=rename_failed
    return 1
  fi
  if ! ln -s "$DST" "$SRC"; then
    mv "$SRC.MIGRATING" "$SRC"
    emit FAIL id="$id" reason=symlink_failed
    return 1
  fi
  # read-through check before the source is destroyed
  if [[ "$(readlink -f "$SRC")" != "$(readlink -f "$DST")" ]] || [[ ! -d "$SRC/" ]]; then
    rm -f "$SRC"; mv "$SRC.MIGRATING" "$SRC"
    emit FAIL id="$id" reason=symlink_readthrough
    return 1
  fi
  rm -rf "$SRC.MIGRATING"

  touch "$marker"
  emit move_done id="$id" src="$SRC" dst="$DST" files="$dfiles" bytes="$dbytes" seconds="$secs" mb_per_s="$mbps"
  return 0
}

run_table() {
  local phase="$1"; shift
  emit phase_start phase="$phase"
  local row src dst rc=0
  for row in "$@"; do
    src="${row%%|*}"; dst="${row##*|}"
    do_move "$src" "$dst" || { rc=1; emit phase_abort phase="$phase" at="$src"; break; }
  done
  emit phase_end phase="$phase" rc="$rc"
  return "$rc"
}

# ------------------------------------------------------------------
# Move tables
# ------------------------------------------------------------------

PHASE1=(  # T1 NVMe -> T2 CMR : large sequential-read data still in use
  "/home/weiland/rl_router|/data/rl_router"
  "/home/weiland/ckpt_pi05_robocasa|/data/ckpt/pi05_robocasa"
  "/home/weiland/ckpt_n15_robocasa_tp|/data/ckpt/n15_robocasa_tp"
  "/home/weiland/ckpt_n15_robocasa|/data/ckpt/n15_robocasa"
  "/home/weiland/ckpt_n15_libero_spatial|/data/ckpt/n15_libero_spatial"
  "/home/weiland/ckpt_pi05_robocasa_pytorch|/data/ckpt/pi05_robocasa_pytorch"
  "/home/weiland/openpi/exp/common/data/db/libero_cache|/data/openpi/exp_common/db/libero_cache"
  "/home/weiland/openpi/exp/rl_router/data|/data/openpi/exp_rl_router/data"
  "/home/weiland/openpi/exp/markov_sufficiency/data|/data/openpi/exp_markov_sufficiency/data"
)

PHASE1B=( # T1 NVMe -> T3 SMR : cold since 2026-07, 115 files averaging 505 MB
  "/home/weiland/run_so_101/train|/archive/run_so_101/train"
)

PHASE2=(  # T2 CMR -> T3 SMR : finished lines, raw data kept for reproducibility only
  "/data/openpi/ablation_study/cache_size/collect_h5|/archive/openpi/ablation_study/cache_size/collect_h5"
  "/data/openpi/ablation_study/cache_size/save_traj|/archive/openpi/ablation_study/cache_size/save_traj"
  "/data/openpi/ablation_study/executor_substitution/checkpoints|/archive/openpi/ablation_study/executor_substitution/checkpoints"
  "/data/openpi/ablation_study/executor_substitution/distill_raw|/archive/openpi/ablation_study/executor_substitution/distill_raw"
  "/data/openpi/ablation_study/executor_substitution/lerobot|/archive/openpi/ablation_study/executor_substitution/lerobot"
  "/data/libero_cache/build_libero10|/archive/libero_cache/build_libero10"
  "/data/libero_cache/build_spatial|/archive/libero_cache/build_spatial"
  "/data/run_so_101/models|/archive/run_so_101/models"
)

# Phase 3 is enumerated at run time, one move per task directory, so a failure
# five hours in costs one task rather than the whole 1.1 TiB. Every episode file
# is written once by data_collector.py (`h5py.File(tmp, "w")` then rename) and
# only ever reopened "r", so the SMR overwrite penalty is structurally
# unreachable; the h5 datasets already carry lzf compression, so btrfs zstd:3
# is expected to recover essentially nothing.
build_phase3() {
  local d
  for d in /data/robocasa365_cache/build_l1s1/pi05/*/ \
           /data/robocasa365_cache/build_l1s1/groot_tp/*/; do
    [[ -d "$d" ]] || continue
    d="${d%/}"
    printf '%s|/archive%s\n' "$d" "${d#/data}"
  done
  printf '%s\n' "/data/robocasa365_cache/_archive_t3_manual|/archive/robocasa365_cache/_archive_t3_manual"
}

# ------------------------------------------------------------------
# Entry
# ------------------------------------------------------------------

main() {
  local what="${1:-all}"
  emit run_start what="$what" host="$(hostname)"
  df -B1 --output=target,used,avail / /data /archive | tail -3 | while read -r t u a; do
    emit df_before target="$t" used="$u" avail="$a"
  done

  case "$what" in
    phase1)  run_table phase1  "${PHASE1[@]}" ;;
    phase1b) run_table phase1b "${PHASE1B[@]}" ;;
    phase2)  run_table phase2  "${PHASE2[@]}" ;;
    phase3)  mapfile -t P3 < <(build_phase3); run_table phase3 "${P3[@]}" ;;
    all)
      run_table phase1  "${PHASE1[@]}"  &&
      run_table phase1b "${PHASE1B[@]}" &&
      run_table phase2  "${PHASE2[@]}"
      ;;
    *) emit FAIL reason=unknown_phase what="$what"; return 2 ;;
  esac
  local rc=$?

  df -B1 --output=target,used,avail / /data /archive | tail -3 | while read -r t u a; do
    emit df_after target="$t" used="$u" avail="$a"
  done
  emit run_end what="$what" rc="$rc"
  return "$rc"
}

exec 9>"$LOCK"
# Blocking, unlike tier_migrate.sh: this instance is queued on purpose and
# waits for the in-flight run to release the lock. The single-stream
# invariant still holds -- two rsync streams on one spindle collapse to
# ~37 MB/s, so overlapping is never an option.
echo "waiting for the migration lock (queued behind the running phase)..." >&2
flock 9
echo "lock acquired; starting" >&2

main "${1:-all}"
```

---

## Review Log

_（L1，无 G1/G2 门。本节保留以备 owner 追加意见。）_
