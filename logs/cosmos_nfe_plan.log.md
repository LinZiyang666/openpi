# Cosmos Policy × LIBERO 减步基线（cosmos_nfe）— L1 plan

> 2026-09-16 owner 指令：开始 Cosmos 的工作，先在 weilandserver 上试；第一个实验是**单纯减去去噪步数**（不做想象式/规划），LIBERO libero_10 + libero_spatial；获准机器 weilandserver（4090 48 GB）与 timan107。
> 授权：Execution。级别 L1（exp/ 脚本 + 远端环境，src 零改动）。

## 1. 目标

Cosmos-Policy-LIBERO-Predict2-2B（EDM 视频扩散共去噪，整模循环，默认 5 步）在 libero_10 / libero_spatial 官方 50 init × 10 任务 = 500 集上，`--num_denoising_steps_action k`，k ∈ {5（复现）, 3, 2, 1}（有余力加 4），得到 SR(k)；与论文 97.6 / 98.1（H100，seed 195/196/197 均值）对照。判据：k=1 是否掉点（EDM 未蒸馏 vs 我们 π0.5/GR00T 流匹配的"一步无损"）。成本：单卡 4090 实测 1/2/3/5 步的每 chunk 延迟，IR(k) = 实测延迟比（无前缀/头分工，c_pre = VAE + T5 查表）。

## 2. 机件与拓扑

- 代码：`nvlabs/cosmos-policy`（HEAD 18a2acc）。**磁盘分工（owner 11:2x 裁定）**：代码 + venv 在 SSD `/home/weiland/cosmos-policy`（uv 缓存 `~/.cache/uv`），私有 HOME `/home/weiland/cosmos_home`（.libero 配置、LIBERO 资产、HF token）；模型权重 `HF_HOME=/data/cosmos/hf_home`、结果 `/data/cosmos/results`、上游日志 `/data/cosmos/logs` 在 HDD /data；`/archive`（SMR）不用。首次装在 `/data/cosmos/cosmos-policy` 的旧 venv（13 GB）留待 owner 删除。依赖走 NVIDIA 预编译索引（`cosmos-dependencies/v1.2.0/cu128_torch27`：torch 2.7 + flash-attn 2.7.3 + TE 2.2 + natten），`uv sync --extra cu128 --group libero --python 3.10`，**不需要 Docker / nvcc**（host 只有 CUDA 13.2 toolkit、无 nvidia-container-toolkit）。`libero` 0.1.1 来自 PyPI。
- 检查点：HF `nvidia/Cosmos-Policy-LIBERO-Predict2-2B`（含 `libero_dataset_statistics.json`、`libero_t5_embeddings.pkl`），eval 脚本按 repo id 自动下载；`HF_HOME=/data/cosmos/hf_home`。
- 官方 eval：`cosmos_policy.experiments.robot.libero.run_libero_eval`，**串行**跑 10 任务 × 50 集（并行只用于 best-of-N 查询跨 GPU）；chunk 16 整段执行；`--deterministic True --seed 195`；显存 6.8 GB。
- 并行化：wrapper `exp/cosmos_nfe/run_libero_shard.py`（monkeypatch `run_task` 只跑 `--task-ids` 子集，并把每任务 (n, successes) 写 JSON）；单 4090 上 5 个分片进程 × 2 任务（≈ 5 × 7 GB）。timan107 本轮不需要（官方脚本 sim 与模型同进程；后续若接我们的 server/client harness 再用）。
- 渲染：`MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`，weilandserver 有 libEGL_nvidia。

## 3. 步骤

1. 环境：clone + uv sync（tmux `cosmos_setup`，日志 `/tmp/cosmos/uv_sync.log`）。
2. 冒烟：libero_10 task 0 × 2 集，k=5，核对能跑通、显存、每 chunk 延迟、日志格式。
3. 复现：libero_10 k=5 全 500 集（5 分片），对照 97.6。
4. ladder：libero_10 k=3,2,1；libero_spatial k=5,3,2,1。每点 = 10 任务 JSON 合并。
5. 成本：单进程无干扰下测 k=1/2/3/5 每 chunk 延迟各 ≥50 次。
6. 聚合 + 图：写进 `exp/cosmos_nfe/data/`，spec 进 rit_pareto figures（后续与 cache 前沿同图）。

## 4. 运行记录
- **11:04** clone `nvlabs/cosmos-policy` @18a2acc → `/data/cosmos/cosmos-policy`；`uv sync --extra cu128 --group libero --python 3.10`（NVIDIA 预编译索引，无需 nvcc/Docker）。两个坑：① `egl-probe`（robomimic 依赖）要 cmake → `uv tool install cmake`；② cmake 4.x 拒绝 egl-probe 的 `cmake_minimum_required < 3.5` → 装 `cmake<4`（3.31.10）。第三次 sync 进行中（tmux `cosmos_setup`，日志 `/tmp/cosmos/uv_sync2.log`）。
- 代码：`exp/cosmos_nfe/run_libero_shard.py`（monkeypatch 上游 `run_task`/`run_episode`/`get_action`：任务子集、每集成败、每次查询延迟，写 JSON）+ `ops/{run_shard_wls,launch_point_wls,ladder_wls}.sh`；推到 weilandserver `/data/cosmos/openpi_exp/exp/cosmos_nfe`（PYTHONPATH）。结果 `/data/cosmos/results/<suite>/k<k>/shard<i>.json`，日志 `/tmp/cosmos/<suite>_k<k>_s<i>.log`，上游日志 `/data/cosmos/logs/`。
- **11:31–11:44 门控 + 冒烟**：推理 config 从 gated 仓库 `nvidia/Cosmos-Predict2-2B-Video2World` 取 Wan2.1 VAE `tokenizer/tokenizer.pth` 与底座 `model-480p-16fps.pt`（NVIDIA Open Model License）；owner 在网页同意许可并在 weilandserver 上 `hf auth login`（token 在 `/data/cosmos/hf_home/token`，不经我手）。第一次登录后仍 403（未点 Agree），点了即通。冒烟 libero_10 task 0 × 2 集 k=5：**2/2 成功**，每次查询 840 ms（p50 813，4090；论文 H100 0.61 s），显存 6.5 GB，每集 21–32 s，模型实例化 17 s；结果 `/data/cosmos/results_smoke/`。
- **11:21 迁移到 SSD**：代码 + venv `/home/weiland/cosmos-policy`、私有 HOME `/home/weiland/cosmos_home`、exp 包 `/home/weiland/cosmos_exp`；模型留 `/data/cosmos/hf_home`（含底座后 12 GB）。
- **11:44 首次起阶梯失败**：tmux server 由 ladder 会话先起，后续 `tmux new` 不继承调用 shell 的 PYTHONPATH/HOME → `No module named exp`；且 5 个并发 `uv run` 会同时 sync 同一 venv（危险）。改 `run_shard_wls.sh`：env 写进 tmux 命令串、直接用 `.venv/bin/python`。
- **11:46 阶梯起跑**：tmux `cosmos_ladder`，顺序 libero_10 k=5,3,2,1 → libero_spatial k=5,3,2,1，每点 5 分片 × 2 任务（并发 5 进程，GPU ≈ 20.6 GB / 58%）；日志 `/tmp/cosmos/ladder.log`、分片 `/tmp/cosmos/<suite>_k<k>_s<i>.log`；Monitor + cron（:13/:33/:53 PROBE）。
- **12:05 owner 加入 h100，client 在 timan107**（12:0x 指令）。拓扑 B：h100 `exp/cosmos_nfe/serve_cosmos.py`（websocket + msgpack_numpy，加载方式与上游 eval 完全一致，每请求带 k 与 seed，单线程串行）↔ timan107 10 个 client 分片（`run_libero_shard.py --remote ws://149.165.153.233:23240`：stub `get_model`/T5 cache，`get_action` 走远端；仿真 + 上游 episode 循环原样在 timan107，EGL 渲染 GPU 0–7）。脚本 `ops/{serve_h100,stop_h100,run_shard_t107,ladder_t107}.sh`。
  - h100 env：`/data/cosmos/cosmos-policy`（根盘只剩 8 GB，全放 /data），uv sync 1 分钟；TE import 需 `CUDA_HOME=<venv>/nvidia CUDNN_HOME=<venv>/nvidia/cudnn`（否则 `ldconfig -p | grep libnvrtc` 失败）；HF token 由 owner 在聊天里给出并授权，存 `/data/cosmos/hf_home/token`（600），不进仓库/记忆；**建议 owner 事后在 HF 撤销该 token**。
  - timan107 env：`/scratch/zixuans8/cosmos/cosmos-policy`（/scratch 只剩 17 GB），driver 535 + cu128 torch 可用；私有 HOME `/scratch/zixuans8/cosmos/home`；LIBERO 资产自动下载；EGL 渲染探针通过（20 步 7.0 s）。
  - 坑：libero 包首次 import 若无 `~/.libero/config.yaml` 会**交互式提问**卡死 tmux（h100 踩到）→ 三台都预写 config。
- **12:12 远程冒烟通过**（timan107 task 0 × 2 集 → h100 :23240，k=5）：2/2 成功，server 端 524 ms/查询（H100，比论文 0.61 s 快），含网络往返 666 ms（p50 607），每集 20–23 s。SHARD_EXIT=0；进程退出时的 `EGLGLContext.__del__` EGLError 是 robosuite 清理的良性噪声。
- **12:16 h100 开 10 个 replica**（owner："h100 应该 replica 可以开更高"）：`ops/launch_replicas_h100.sh 10 23240` → :23240–23249，每个 ≈5 GB，合计 50 GB/80 GB；修了 `serve_cosmos.py` 断连处理的 `ConnectionClosed` 引用（只影响连接收尾，不影响推理）后 `restart_replicas_h100.sh` 全部重起。
- **12:18 双 lane 正式跑**：weilandserver ladder 改为只跑 libero_10（k=5,3,2,1，5 分片；重挂 driver 不动在跑分片）；timan107 `cosmos_ladder` 跑 libero_spatial k=5,3,2,1，10 分片 × 1 任务，分片 i → h100 :2324i，渲染 GPU i%8。Monitor 三机 + cron 三条 PROBE（:13/:33/:53）。
- **12:41 libero_spatial k=5 完成（h100 lane，500 集，20 分钟）：SR 0.986**（论文 98.1，复现通过）；逐任务 1.0/1.0/1.0/0.98/1.0/0.94/1.0/0.96/0.98/1.0。10 replica 并发下 server 端均 1.6 s/查询、往返 1.54 s（单流 0.52 s → 10 路合计 ≈6 查询/s，3.3× 收益）。本地 `exp/cosmos_nfe/data/results/libero_spatial/k5/`，聚合 `exp/cosmos_nfe/data/aggregate.json`。
- **12:59 libero_spatial k=3：SR 0.984**（500 集，17 分钟；逐任务 1/1/1/0.96/1/0.94/1/0.94/1/1）。k=2 自动起跑。
- **13:17 libero_spatial k=2：SR 0.974**（500 集，16 分钟）。k=1 自动起跑。
- **13:28 libero_10 k=5（weilandserver lane，500 集，100 分钟）：SR 0.98**（论文 97.6，复现通过）。k=3 自动起跑。
- **13:33 libero_spatial k=1：SR 0.974**（500 集）。spatial 阶梯完：0.986/0.984/0.974/0.976（k=5/3/2/1），一步无损。h100 lane 转 libero_10 k=2。
- **13:36 发现：Cosmos 的 "k 步" = k+1 次网络前向。** spatial k=1 与 k=2 逐集结果完全一致（0.974，同一失败集合）、延迟也几乎相同，追到 `res_sampler.py`：`_forward_impl` 跑 nfe 个求解步之后 `sample_clean=True` 再在 σ_min 上做一次 x0 评估。dummy 去噪器实测：nfe=1 → 2 次（σ=80, 0.002），nfe=2 → 3 次（80, 2.52, 0.002），nfe=3 → 4 次，nfe=5 → 6 次。含义：① k=1 已经不是"一步"，是"σ_max 一步 + 清洁一步"，所以 spatial k=1 ≈ k=2 不奇怪；② 成本轴不能用 k/5，要用实测延迟（或 (k+1)/6）；③ 真正的单次前向要把 `sample_clean` 关掉才能测——留作可选补点（`sample_clean=False, nfe=1`），不在 owner 指令范围内，先记着。
- **13:40 owner 裁定**：Cosmos libero_10 k=1 跑完后**全线停止**（不再跑其余点），机器转去把 π0.5 / GR00T 的 RoboCasa365 减步实验跑完，**每个 k 都跑**（GR00T k=1,2,3；π0.5 k=1…9；13 任务 × 50 seed，journal 续跑）。准备：`switch_rc_{wls,t107}.sh` 的 ks 恢复 1…9；`ops/stop_cosmos_{clients,h100}.sh` 就位。拓扑沿用昨天授权：GR00T h100:23230 ↔ timan108 15 worker；π0.5 weilandserver:23170 ↔ timan107 24 worker。
- **14:12 libero_10 k=2：SR 0.972**（500 集；逐任务最低 0.86）。**owner 14:04 改口：k=2 跑完即全线停，不跑 k=1**。14:13 三台 Cosmos 全部下线（timan107/weilandserver client + ladder、h100 10 replica）；weilandserver 的 libero_10 k=3 在 277 集处停（270 成功，未完成、不入表）。
- **Cosmos × LIBERO 终表（500 集/点，5 步 = 6 次前向，k 步 = k+1 次前向）**：spatial k=5/3/2/1 = 0.986/0.984/0.974/0.974；libero_10 k=5/2 = 0.980/0.972（k=3 partial 0.975@277，k=1 未跑）。结论：与 π0.5/GR00T 同，LIBERO 上减步几乎无损，EDM x₀-预测的"一步"是条件均值。

## 5. RoboCasa-2024（24 任务）减步阶梯 — 2026-09-16 晚立项与准备

owner 20:0x 裁定：LIBERO 不再管；删 wls/h100 上的 Cosmos LIBERO 件（`rm -r` 精确路径，7 项 ≈ 21 GB，见 `logs/nfe_baseline_rc365_plan.log.md` 同时段记录）；配 Cosmos × RoboCasa-2024 环境；server 只在 wls/h100，模拟器只在 t107/t108；π0.5 阶梯结束后全线跑 **k=1,2,3,4,5 × 24 任务 × 50 trial**。

- **Checkpoint**：`nvidia/Cosmos-Policy-RoboCasa-Predict2-2B`（3.9 GB .pt + `robocasa_dataset_statistics.json` + `robocasa_t5_embeddings.pkl`），RoboCasa 2024 版 24 atomic 任务、50 demo/任务，论文 67.1%（H100，3 seed 平均）。eval 口径（ROBOCASA.md）：`obj_instance_split=B`（held-out 物体）、5 个测试场景 `((1,1),(2,2),(4,4),(6,9),(7,10))`，trial i 用场景 (i//10)%5、seed = 195·i·256、chunk 32 执行 16、5 步去噪、deterministic。
- **环境**（`exp/cosmos_nfe/ops/rc24/setup_rc24.sh`，四机同一脚本）：`uv sync --extra cu128 --group robocasa` → clone `moojink/robocasa-cosmos-policy` + `uv pip install -e` → **坑①** fork 钉 `numba==0.56.4` 与 venv numpy 2.2.6 不兼容（`robosuite.utils.numba` import 崩）→ 装 `numba>=0.61,<0.62`；**坑②** `uv sync` 是 exact sync，把早上手装的 msgpack/msgpack-numpy/websockets 删了 → sync 后重装；**坑③** `yes | download_kitchen_assets.py` 在 pipefail 下因 SIGPIPE 误报失败 → 改 printf；**坑④** client 不需要 3.9 GB .pt（snapshot_download 加 allow_patterns），t107 /scratch 只剩 6.5 GB（已删多下的 blob、清 uv cache）。client 端 kitchen assets ≈ 8.2 GB（textures/fixtures/objaverse/generative_textures）。
- **代码**：`exp/cosmos_nfe/run_robocasa_shard.py`（`--trials a-b` 一个任务的 trial 区间为一个分片，复用上游 `create_robocasa_env/run_episode/get_action`，跳过视频；`--remote` 同 LIBERO 版）；`serve_cosmos.py --bench robocasa`；ops：`serve_rc24.sh`（按 hostname 选路径）、`launch_replicas_rc24.sh`、`run_shard_rc24.sh`、`ladder_rc24_client.sh`（120 分片/k 的 tmux 作业池、失败重投 3 轮、`RC24 k=<k> UP/DONE/FAIL`）、`arm_full_*.sh`（等 π0.5 lane STOPPED 后自动起）；聚合 `aggregate_cosmos_rc24.py`（多 root 合并、24 任务 macro）。
- **冒烟 20:05**（h100 :23250 replica，t108 GPU2 一个 worker，TurnOffMicrowave trial 0-1，k=5）：2/2 成功，46 次查询，server 827 ms/次（H100，比 LIBERO 的 ~600 ms 重：chunk 32 + 三路图），client 端 985 ms，单集 ~38 s + 建 env ~11 s；replica 显存 6.5 GB。
- **全量拓扑（已武装，自动触发）**：h100 等 `ladder_prc2.log` 出现 `LADDER STOPPED`（π0.5 pair B 跑完 k=6，~23:20）→ 10 replica :23250-23259；wls 等 `ladder_prc.log` `LADDER STOPPED`（pair A 跑完 k=5，~22:35）→ 4 replica **:23180-23183**（公网段）；t108 等自己 lane STOPPED + 10 个 h100 端口可连 → `ladder_rc24_client.sh 1,2,3,4,5 <h100 urls> 30 0,1,2` 跑 trial 区间 0-9/10-19/20-29/30-39（80% 工作量）；t107 等 lane STOPPED + 4 个 wls 端口 → 32 worker 跑 40-49（20%）。估算：每 k 3 万次查询，h100 侧 ~8 q/s、wls 侧 ~2.4 q/s，k=5 约 45 min、k=1 约 15 min，全程 ≈ 2.5 h，两侧约 02:00 同时收工。结果在各 client `/scratch/zixuans8/cosmos/results_rc24/k<k>/<task>_t<a-b>.json`，日志 `/tmp/cosmos/rc24_full.log`。
- **22:15–22:35 server 提速（owner 追问 CUDA graph）**：`exp/cosmos_nfe/bench_cudagraph.py` 在 h100 上对 RoboCasa 模型做 A/B（合成观测、deterministic、同进程重复）：
  - 纯 eager：k=5 821 ms、k=1 565 ms → 每次 DiT 评估只有 ~64 ms，**固定开销 ~440 ms**，大头是 `get_future_images_from_generated_samples` 把整段 11 帧 latent 过 Wan-VAE decoder 生成"未来图像"（只用于可视化/planning，对动作无关）。
  - `--no-future-decode 1`（monkeypatch 该函数返回 `{}`，value 仍从 latent 读）：k=5 576、k=1 306 ms。
  - `--cuda-graph 1`（`exp/cosmos_nfe/cudagraph_net.py`：按输入签名捕获 `model.net` 前向、静态 buffer 复制 + replay，TE/flash-attn 可捕获，0.5 s 捕完）：再到 k=5 470、k=1 295 ms；每次 DiT 评估 64→~45 ms。三种路径动作 **max|Δ|=0（逐位一致）**。剩余固定开销 ~200 ms = 3 图 JPEG 压缩/缩放 + VAE encode + latent 组装，未动。
  - 已并入 `serve_cosmos.py`（两个开关 + 监听前 3 次合成观测暖机/捕图，暖机 prompt 取 T5 cache 里现成的指令，避免触发 T5 加载）和 `serve_rc24.sh`；h100 :23250 冒烟 replica 已换新（显存 5.8 GB）。真机冒烟 TurnOffMicrowave trial 0-1：server 409 ms/查询（原 827）、client 456 ms（原 985）、单集 25 s（原 38），2/2。
  - ⚠ **端到端不是逐次可复现的**：同一优化 server 连跑三次 trial 0-1 得 [359,358]→[365,357]→[500(失败),366]，与优化无关（第一次是未优化 server），是闭环混沌 + 跨进程数值/渲染差异；论文也是 3 seed 平均。50 trial × 24 任务的 macro 口径下两侧同等受影响。
  - 全量估算更新：每 k 3 万次查询，h100 10 replica ≈ 24 q/s、wls 4 replica ≈ 6 q/s → k=5 ≈ 20 min、k=1 ≈ 12 min，全程 ≈ 1.3 h（原估 2.5 h）。
- **22:33 RC24 全量开跑（wls/t107 侧）**，**22:58 h100/t108 侧接上**（π0.5 k=6 信号切档 → `LADDER FINISHED` → 10 replica 起 → t108 探到 10 端口 → 30 分片）。t107 侧 k=1（场景 (7,10)，trial 40-49，24 任务 × 10 集）27 min 跑完：**158/240 = 0.658**（官方 5 步 0.671）；wls 4 replica 被 24 个 worker 打满（server 侧 902 ms/查询含排队、client 端 3.15 s），是这一侧的瓶颈。k=2 23:01 起。
- **23:03 t108 侧 3 个分片起跑即崩**（`AttributeError: 'MjRenderContextOffscreen' object has no attribute 'con'` = EGL 上下文创建失败，30 worker 把 3 张 A5000 顶到 24.2/24.5 GB，RoboCasa-2024 每 worker ≈ 2.5 GB）：杀掉 arm 里的 ladder（跑着的分片不受影响），换 `cosmos_ladder_rc24` tmux 以 W=24 重开同一 ladder（完成的跳过、死的重投）。k=1 t108 侧 22:58 起。
- **23:30 client ladder 换 v2**（`ladder_rc24_client2.sh`，平作业表：k 之间不排空、死分片有空位立刻重投、每分片最多 3 次）：旧版每个 k 收尾要排空池子（wls 侧 k=2 尾巴只剩 4 个分片、4090 掉到 36%），t108 侧 5 个 k=1 分片起跑时 `Offscreen framebuffer is not complete`（显存被顶满时 EGL 建不出帧缓冲）要等一轮结束才重投。两台 client 热切换（跑着的分片不动）。t107 W=24（GTX1080 每卡 3 个，2.7 GB/worker）、t108 W=24。
- t107 侧（场景 (7,10)）：k=1 0.658、**k=2 0.663**（各 240 集）；t108 侧 k=1 跑到 687 集时 0.674。
- **23:56 t108 一个 k=1 分片（OpenDrawer t0-9）挂死 41 min**：起跑时 CUDA/EGL 初始化失败（`cudaGetDeviceCount` unexpected error，显存被顶满那阵）但进程不退出、100% CPU 空转、占着池位。手动杀掉；ladder v2 加了 `reap_stalled`（分片日志 25 min 没增长就杀，之后按 tries≤3 重投），两台 client 再次热切换。
- **00:13 Cosmos k=1 全量出数 → owner 裁定停 Cosmos**（"k=1 没有折损就不用继续了"）。**k=1（24 任务 × 50 trial × 5 场景 = 1200 集）macro 0.6725**（pooled 同值），官方 5 步 0.671；server 721 ms/查询（含排队）、单集 56 s。逐任务最低 TurnOffStove 0.18、PnPCabToCounter 0.26、PnPCounterToMicrowave/PnPMicrowaveToCounter 0.36、CoffeeSetupMug 0.38；最高 CloseDrawer/TurnOffMicrowave 1.00。部分点：k=2 pooled 0.671（997 集、86 分片完）、k=3 0.633（218 集）、k=4 0.564（39 集，太少）。结论与 LIBERO 一致：Cosmos 的 EDM/x₀ 头一步无损（对照 π0.5/GR00T 在 RoboCasa365 上 −17/−11 pp）。聚合 `exp/cosmos_nfe/data/aggregate_rc24.json`（只含完整的 k=1），原始 `results_rc24_{timan107,timan108}/`。四机 Cosmos 全停（replica 0、分片 0、arm 看门全杀）。
