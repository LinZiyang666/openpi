# Session handoff

## 0. 初始化方式（不变）

owner 的常驻指令，逐字有效：

> 「开始进行实验，我离开了，期间由你独断专行，不要问我任何问题，注意监控体系定位，
> corn 负责定时巡检，monitor 负责条件触发，不要职责混淆，停止条件是完全做完实验，不做完不停」

**三条必须保持的纪律**：

1. ⛔ **不得用 git commit / push 同步代码和数据**。要同步任何代码或数据一律走 **tether**。
   git 只在里程碑收口、且 owner 当次明确指示时才动。
2. **server 只在 server 节点跑，worker 只在 timan 族跑**（见 §2）。
3. **读 `experiment-lifecycle` skill 但不挂载**：只取 tether 用法（`dist_experiment_control/docs/usage.md`）
   与设备清单（`docs/devices.md`），**不要执行它的 §0 初始化**，不要索要 agentchat 账号 / token / 房间。

### 开工步骤（照做，不要问 owner）

1. **读文档**，按这个顺序，都读完再动手：
   - `logs/cache_prune_run_handoff.md`（整份，运行入口的唯一权威）
   - 本文件 §1–§5
   - `/home/weiland/projects/dist_experiment_control/docs/devices.md`（拓扑与机器约束）
   - `/home/weiland/projects/dist_experiment_control/docs/usage.md`（tether 命令）
   - 需要时再查 `logs/cache_prune_plan.log.md`（算法与统计定义）
2. **探设备**：`tether node ls -a`，确认 h100 / weilandserver / timan107 / timan108 四台
   都 ONLINE；逐台查 GPU 占用、RAM、磁盘余量、残留 tmux 与进程。⚠ timan108 只有 3 张卡
   （见 §2），别按 4 张排。
3. **对齐代码**：四台的仓要和本地同一 commit。**走 tether push，不走 git push**；
   已存在的文件要 `--force`；`/scratch` 不在 timan 的 allow_roots，经 `/tmp` 中转。
   推完逐文件 `sha256sum` 对账，**比对内容不要带路径**（`cat A B | sha256sum`，
   `sha256sum A B` 会把路径算进去，我为此误判过一次）。
4. **建任务表**（TaskCreate），把 §1 的准备项与三相位拆成条目，边做边更新状态。
5. **搭监控**（§4）：cron 定时一行巡检 + Monitor 条件触发。**先搭好再放量**。
6. **先 smoke 再放量**：任何新拓扑都先跑一个最小 cell 验证到底（起 1 个 server、少量 worker、
   跑通一集并核对产物），再按 §2 的规格扩到目标规模。上一条线每次跳过这步都出事。

**其它长期约束**：

- commit message 全英文；**绝不加 `Co-Authored-By: Claude`** 或任何 AI 署名（作者恒为
  `LinZiyang666 <3177267975@qq.com>`）。未经 owner 当次指示不得 `git add`。
- ⛔ **画图脚本一律不许 commit**（`build_*figure*` / `plot_*` / `render_*` / `edit_*figure*` /
  `*_editor.html`）。`exp/rit_pareto/build_figure.py` 与 `edit_figure.py` 属于另一个 session，**碰都不要碰**。
- ⛔ 不要 `rm -rf`；删除前先 `wc -l` 核对规模。
- ⛔ **共享机上不要 `pkill -f`** 裸模式：它会匹配到发起命令的 shell 自己。用字符类
  `[w]orker_entry`，且**脚本正文任何地方（含注释、echo）都不能出现裸的匹配串**。
- 无人值守期间**禁起 `run_in_background` 后台任务**（触发审批弹窗阻塞会话），用 Monitor。
- 报时刻用本机本地时间（America/Chicago）。
- **巡检就只巡检**：贴 PROBE 行，不要顺手做额外分析（owner 明确要求过）。

## 1. ActionCache 基线 × GR00T N1.5（LIBERO）— 待执行实验的记忆点（2026-09-13 写）

代码已 commit `ece4362`（`origin/Ziyang`，G1/G2 APPROVED，§6 Verify 全量通过、失败项均为 HEAD 既有）。
**实验已于 2026-09-13 跑完**（plan §8 步骤 2–6，两组 10 臂 × 500 集，raw 在 `exp/actioncache_baseline/data/runs/groot_<suite>_w13s3/`，`aggregate` 门全过；过程记录见 plan Review Log 末尾 Execution note）。下面的路径与门保留，供重跑 / 复现。
运行手册 = `docs/experiments/actioncache_baseline.md` **§9**（每一步的命令都在那里，照抄）。

### 1.1 这条线要什么

- 只跑**同库 W13-S3 组**（spatial / libero_10 各一个 50 轨迹库），**不跑 500 集库版本**；每组 = 2 档
  （n0 = FULL_HIT@CP2，n1 = WARM_START@**0.875**）× (4 目标臂 + 1 参考臂 θ_raw=**0.65**) = **10 臂 × 500 集**。
- key = action head **编码后**的 VLM 输出 [N,2048] pad 到 640 token + 编码后 state [1536] → 稀疏三值投影 d=500、p=0.01、
  **seed 20260904**（两 suite 同 seed，与臂 yaml `key_builder.cp2_groot.seed` 一致）。
- 成本：teacher 表 `exp/libero_groot/config/rit/cost_groot_libero_measured.json`（P=13.338, L=28.104, M=41.442，
  RTX 4090 `GPU-98d36ed2-…`，两 suite 共用，**不动**）+ 每 suite 实测编码单价 E；FULL=P+E，WARM=P+E+L/8，MISS=M+E，分母 M。
  E 只要 ≥ ~1.8 ms，n1 的地板就高于 45% → n1 整档按规则 fallback 到 `t01–t04`，**这是设计内的，不要当 bug**。
- 跑完只做 pull raw + `aggregate` 完整性审计；**不画图、不写 analysis、不写 results**（owner 明令）。

### 1.2 设备与路径（本线用到的）

| 机器 | 角色 | 关键路径 |
|---|---|---|
| weilandserver（4090 48G，GR00T 岛） | 建库 / parity / cohort 采集 / E 标定 / preflight / **lane B 的 5 个 eval server** | 代码树 `/data/openpi_lg`（**独立克隆，改完必须 tether push 并核 sha**）；岛 venv `/home/weiland/gr00t_n15_venv/.venv/bin/python`；gr00t `/home/weiland/gr00t_n15`；ckpt `/home/weiland/ckpt_n15_libero_{spatial,10}`；`PYTHONPATH=$G:$G/examples/Libero:$REPO/src:$REPO`，`HF_HUB_OFFLINE=1`；LIBERO client 用 `conda run -p /home/weiland/libero_sim`（`launch_collection.sh` 的写法，`/home/weiland/miniconda3/bin/conda`）——我的 `ops/run_acb_collect_clients.sh` 传 `<python>`，先试 `/home/weiland/libero_sim/bin/python`，smoke 不通就改成 conda run |
| h100（`149.165.153.233`，直连 `:23210-23214`；`/data`→`/media/volume/OmniData`） | **lane A（spatial）5 个 eval server** | 代码树同路径 `/data/openpi_lg`；库 / yaml / 成本记录要同步过去（sha） |
| timan107（8×1080）/ timan108（**3**×A5000） | worker（lane B 64 / lane A 64） | `/scratch/zixuans8/openpi_lg`，`HOME=/home/zixuans8`，run_gtp python `/scratch/zixuans8/openpi/.venv/bin/python`，`--conda-env /scratch/zixuans8/libero_sim`；`/scratch` 不在 allow_roots，经 `/tmp` 中转 |

数据：源库 `/data/libero_cache/libraries_w13/<suite>/<suite>_w13_S3.pkl`（weilandserver）；W13 H5
`/archive/libero_cache/build_{spatial,libero10}_w13/<suite>/episode_XXXX_<ts>.h5`（500 集，**无原始图像**，故用模板重建）；
RIT shadow manifest + init 池在仓内 `exp/libero_groot/data/rit/shadow/<suite>/{shadow_manifest.json,shadow_pool/*.init}`
（seed 20260901）；RIT 参考臂记录 `exp/libero_groot/config/rit/<suite>/arm_record.json`（compare 用，本轮不执行）。
计划产物位置：CP2 库 `/data/libero_cache/libraries_w13_cp2/<suite>/<suite>_w13_S3_cp2.pkl`；cohort H5
`/data/libero_cache/acb_shadow_h5/<suite>/attempt_<a>/srv<i>/`；E 记录 `exp/libero_groot/config/actioncache/cost_groot_cp2_encoded_<suite>.json`；
preflight `exp/libero_groot/data/actioncache/overhead_<suite>/overhead.json`；臂 `exp/actioncache_baseline/config/groot_<suite>_w13s3/`；
raw 拉回 `exp/actioncache_baseline/data/runs/groot_<suite>_w13s3/`（`config/`、`data/` gitignored）。

### 1.3 顺序与门（每一步 fail-closed，过不了就停，不要绕）

1. **parity 门先行**（`groot_cp2_parity.py`，两 suite 各 20 步 + 合成负例；需要 `--task-map`）→ 通过才建库。
2. `emit_task_map.py` **必须在 LIBERO client 环境**（`libero_sim`）跑：`task_id → (task.name, task.language)`。
   manifest 的 `task_name` 是下划线名（定位 `.init`），H5 的 `task` 是自然语言指令；两者只靠 task_id 绑定，**不能互推**。
3. 建库 `build_cp2_artifact_groot.py` → `verify_cp2_artifact.py --teacher groot_libero` → 同步 h100（同路径，sha）。
4. cohort 采集：`launch_acb_collectors.sh`（5 个**非并发** collector，weilandserver 直连段端口 **23130+i**，tmux `acbsrv<port>`，attempt 目录**不复用**；⛔ LIBERO client 不在 weilandserver 跑——按 §0 纪律放 timan107，`ACB_HOST=ziyanglin.com`，client 终态 JSON 搬回 collector 的 attempt 目录再验收）
   + `run_acb_collect_clients.sh`（client i **串行**跑 task i、i+5，tmux `acbcli<i>`；每 collector 同一时刻只能有一个连接，
   第二个连接会被 1013 拒掉）→ `verify_shadow_h5.py --task-map …`（成功集 `success` 终态；失败集只收 `step_cap` 且
   steps==max_steps+10，即 230/530；同 attempt 重复 = 该集作废；补跑用 `<acc>/retry/filter_task_<t>.json` 起新 attempt）。
   两 suite 顺序做（换 checkpoint 要重启 server）。≈12 min/suite + 起服 4 min。
5. shadow 表 `build_shadow_table_groot.py`（accepted manifest 必须 `ok:true` 且 `task_map_bound:true`；**不要用 `--limit-episodes` 的表发臂**，会被拒）。
6. E 标定 `ops/run_cp2_encoder_cost.sh`（独占 4090 + nsys，N=566 / 30 warmup / 200 iters / reduce-overhead 冻结；
   每次重测，旧记录自动归档；`debug_sampling` 的记录不能认证）+ preflight `ops/run_cp2_overhead.sh`（warm total P95 ≤10 报告 / 10–40 记提示 / >40 停发臂）。
7. 出臂（主 venv）`export_arms.py --teacher groot_libero --lib-tag w13s3 --cost-record … --encoder-cost-record … --preflight-record …`
   → 恰 10 臂 `acb_<sp|l10>_w13s3_<n0|n1>_<t01..t04|ref650>`；yaml/库/成本记录同步两台 server 与两台 sim box（sha）。
8. eval：server `launch_eval_servers.sh`（5 进程/lane，`--allow-dynamic-bundles`）；client 在 sim box 上
   `ops/run_acb_eval_group.sh <suite> <arm_matrix.yaml> <servers> <workers> <gpus> <run dir>`（`--checkpoint cp2 --warm-tiers 0.875`）。
   **先 smoke**（n0+n1 各 10 集，两 suite 共 40 集，主评测之外），再主跑 lane A spatial（h100×5 + timan108 64w）∥ lane B l10（weilandserver×5 + timan107 64w）。
   监控照 pi0.5 线：cron 20 min 巡检 + Monitor 条件触发；每组完成 pull raw（sha）+ `aggregate` 审计；全部完成关 server/worker、撤监控。
   RIT 旧吞吐（lane B ≈18.8 集/分）只作排期参考；encoded CP2 的实际吞吐以 smoke 为准。

### 1.4 踩过的坑（本线代码里已修，跑的时候别再犯）

- backend 检索**不能放在 `runner.session()`（autocast bf16）里**，否则 cosine 是 bf16 的，与在线 `check()` 口径不一致。
- `tests/review_tests/` 是审查方私有材料，**不许拿来自测**（R1 因此被记流程违规）。
- W13 H5 无 `task_id`/`orig_init_state_idx` attrs 不要紧（建库只用 trajectory_id/step_idx）；但 **cohort 采集**的 H5 必须有这两个 attrs
  （client 经 `--episode-filter` + `episode_start` 打上），验收硬性要求。
- `bench_cp2_overhead_groot --mode encoder-cost` 会先核对 GPU uuid 与 teacher 表一致、live schedule 为 k8；checkpoint 允许与表不同（两 suite 各自的 ckpt）。
- 主 venv 的全量 pytest 有 10 个 HEAD 既有失败（robocasa365 / dispatch_surface / exp/test_prebuilt_matrix_backend），与本线无关，别去修。
- parity 第 1 项对 fp16 次正规区只容忍 1 ulp（已修）；E 标定的 compile 区域不能出现 `BatchFeature`/graph break，否则 Dynamo 每次重编译、E 假到 370 ms 而认证照过（已修为张量孪生 + `fullgraph=True`）。
- `pkill -f "[x]yz"` 的字符类只防 pkill 自己，**同一条 `bash -lc '…'` 里别处出现的裸串仍会把整个 shell 杀掉**——kill 前后的命令拆成两次 tether exec。
- 吞吐参考（weilandserver×5 + timan107 64w）：spatial ≈ 46 集/min（1.8 h / 5000 集），libero_10 ≈ 15–23 集/min（5.2 h）。
