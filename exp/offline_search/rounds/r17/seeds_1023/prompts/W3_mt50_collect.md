# W3：MT50 seed 9 / seed 10 采库管线

先读 exp/offline_search/rounds/r17/seeds_1023/prompts/COMMON.md（必须遵守）和 seeds_1023/DESIGN.md。

## 目标
让 π0.5、GR00T、DP、Fast-WAM 四个模型都能用 MetaWorld **seed 9、seed 10**（`MT1(task, seed).train_tasks[idx]`，idx 0–49，50 任务 × 50 = 2,500 局）各采一个 MT50 缓存库。采集口径与现有 seed 8 库完全相同：全步纯推理、commit_horizon 5、同样的捕获内容；现有 seed 8 库不动。

采完能在**任意机器**（不限本机）做 finish（READY.json、`subsets/size_{5,10,20,30,40,50}`、`fits/size_N` C10 拟合），新库放在：
`/data/r17_library/seed9/<m>_mw_mt50/{campaigns,bpool}` 和 `/data/r17_library/seed10/...`（/data/r17_library 实际在 /nvme，可写）。路径必须以 `_mt50/bpool` 结尾（`ops/cache_study.py:136,142` 靠这个后缀判断 250 = size_5）。

## 已知要改的点（来自只读调查，自行核实）
- seed 写死 8：`exp/r17_integration/debug/library_capture.py:52`（服务端捕获 seed≠8 就报错）、`search/teacher_library.py:36`（finish 要求 ep seed==8）、`search/cache_recipe.py:82-84`（加载库要求 init_pool=='seed8' 且 seed==8；改成接受库自己记录的种子，seed 7 测试集照旧拒绝）、`ops/library_campaign.py:31`（seed 和 init_pool 标签取 spec/seed8）。标签：`teacher_library.py:125,223,226`（`seed8`、`subset_seed 8` 改为实际种子）。
- 不要改 `benchmarks/spec.py:41` 的 `library_seed`（它进 benchmark identity，改了会断开所有库与 seed 7 测试集的对应）。种子按任务传（`task.extra['seed']`），由 library_campaign 新参数 `--library-seed`（默认 8）决定，`init_pool` 标签写 `seed<N>`。
- 库根写死、已存在就拒绝：`library_campaign.py:20,370-372` → 加 `--library-root`（默认原值）；有独立根就不需要拒绝覆盖那道门。
- `openpi_ext/scratch/r17_ops/mw/collect_one.v2.run.sh` 写死 campaign 目录 `/data/r17_library/${MODEL}_mt50/campaigns/$RUN` 和 `--subset-counts 10,20,30,40,50`：**新建** `collect_seed.run.sh`（不要改旧脚本），接受种子和库根，subset counts 用 `5,10,20,30,40,50`。
- `ops/library_staging.py:57-59` 只准本机 finish → 拔掉（H100/其他机器也能 finish）；`library_staging.py:23-25,46-54` 的 H100 /data ≥30 GB 门、`library_campaign.py:301-303`（不准本机 worker、timan1≤40）拔掉（与中央调度器冲突）。
- `teacher_library.py:327` / `library_campaign.py:236` 写 pending knob 文件、`store.py:77` 与 `cache_study.fit_inputs` 里算了又不用的哈希：拔。
- `nested_views`（`teacher_library.py:192-231`，目录名写死 `size_N`，已存在且内容不同就报错 :209-213）：加一个可选的输出根参数（另一个任务 W1 要用它在新目录写 LIBERO 新种子的子集视图；行为默认不变）。
- `driver_release.py:44-45` 采集强制旧版 driver：保留（group-commit 采集没跑过）。

## 文件归属（只改这些）
`exp/r17_integration/debug/library_capture.py`、`exp/r17_integration/search/teacher_library.py`、`exp/r17_integration/search/cache_recipe.py`（只改 seed 检查和 DESIGN.md 列出的 cache_recipe 冗余戳/sha，绑定检查保留）、`exp/r17_integration/ops/library_campaign.py`、`exp/r17_integration/ops/library_staging.py`、`exp/r17_integration/common/store.py`（若 store.py 在别处，按实际路径）、新脚本 `openpi_ext/scratch/r17_ops/mw/collect_seed.run.sh`、RUN_CONFIGS.md 的 MetaWorld 采库小节（补一段新种子采库命令）。

## 交付
按 COMMON.md。命令清单要包括：每个模型 × 每个种子的冒烟（`--smoke`，2 任务 × 2 init）和正式采集命令模板（端口留空让协调者填，并注明 π0.5 与 GR00T 同用 23160 窗口、driver 都落 23169 的冲突）、finish 命令、读回核对（随机抽 3 个任务，确认 `extra.seed`、`init_pool`、env 起点与 `MT1(task, seed).train_tasks[idx]` 一致）。
