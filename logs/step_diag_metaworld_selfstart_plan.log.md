# step_diag：π0.5 × MetaWorld MT50 自产起点（无 cache）— 计划

> 状态：`Superseded`（2026-09-25 立项；**owner 2026-09-25 16:20 CDT 裁定：不走 step_diag、不写 plan / 不走 G1，MetaWorld 与现行实验直接迁到 warm reset 一等公民框架，由 agent 直接实现**——本文件仅保留 §1–§2 的冒烟结论与仿真约定供实现参考）。级别 **L2**（新增 benchmark 接入：`src/openpi` 新 policy 变换 + 推理配置，`exp/step_diag` 无库自产路径、MetaWorld 驱动 / worker、EnvSpec、队列与分析扩展）。Authority: Execution。
> owner 裁定（2026-09-25）：新 benchmark 用 π0.5、**只跑 full / plain / self，不建 cache 库、不收集 cache**；基础设施用 **step_diag**（不走 warm reset 一等公民框架，理由见 `warm_reset_migration_study.log.md` 与 §7）；优先级**高于 GR00T**（RoboCasa GR00T 与 LIBERO GR00T）；另加 `plain_k1`。
> 上位：[`step_diag_libero_selfstart_plan.log.md`](step_diag_libero_selfstart_plan.log.md)（LIBERO 复现，本计划照搬其臂语义与数据流）、`exp/step_diag/analysis/step_vs_warmstart.md` §6.13 / §6.15。

## 0. 摘要（供无对话上下文的 G1 审查方）

RoboCasa365 上 π0.5 的 reset 式 warm reset（把一个动作起点在 t=1 或 0.9 重新喂入、走 2 步）把成功率从 full 0.548 提到 0.708，且起点换成**当场自产**（同一观测上先做一次 full 推理，不用 cache）同样有效（§6.13）；LIBERO 上则无收益（§6.15，减步本身不掉点）。本计划在第三个 benchmark —— MetaWorld MT50（50 任务，公开 π0.5 SFT checkpoint）—— 上只跑 full、plain_k2、plain_k1 与三个自产臂，回答：自产 warm reset 能否同时胜过 full 与减步。因为不建库，现有自产臂「借 cache 的 WARM_START 分支触发」的实现用不了，需要加一条**不经检索**的自产路径（§4.2）。

## 1. 目的与冒烟结论（Understand 阶段，临时脚本，未入库）

- checkpoint：`RLinf/RLinf-Pi05-MetaWorld-SFT`（HF，openpi PyTorch safetensors，7.47 GB；`metadata.pt` 训练配置 `pi05_metaworld`：`action_horizon=5`、`discrete_state_input=False`、`num_steps=10`、`repo_id=lerobot/metaworld_mt50`），落在 `/data/ckpt/pi05_metaworld_rlinf`，`norm_stats.json` 复制到 `assets/metaworld_mt50/`（`policy_config.create_trained_policy` 从 `<ckpt>/assets/<asset_id>` 读）。严格加载（`safetensors.torch.load_model` 默认 strict）无缺失 / 多余键。
- 约定来源：RLinf 源码（`rlinf/models/embodiment/openpi/policies/metaworld_policy.py`、`dataconfig/metaworld_dataconfig.py`、`rlinf/envs/sim/metaworld/metaworld_env.py`、`toolkits/standalone_eval_scripts/openpi/metaworld_eval.py`）与 LeRobot `envs/metaworld.py` 交叉核对（调研记录见本会话；要点全部固化进 §2）。
- 冒烟（50 任务 × idx 0,1，`MT1(env, seed=0).train_tasks[idx]`，一 server 串行推理）：

| | K=10（full） | K=2（plain） |
|---|---|---|
| 合并成功率（99 集配对） | 0.515 | 0.596 |
| 四难度组简单平均（RLinf 口径；RLinf 公布 43.8，K=5、10 集/任务） | 36.6（easy 69.6 / medium 31.8 / hard 25.0 / very hard 20.0） | 50.0 |
| 不一致对（仅 K10 成功 / 仅 K2 成功） | 5 | 13 |
| 两者皆成功集的推理调用数 | 13.5 | 13.5 |

  结论：接入约定正确（easy 组与 RLinf 几乎相同，合并率差 3 个点）；**MetaWorld 上减步不掉点、K=2 反而更高**（McNemar p≈0.1）。因此本实验问的是「在 full 输出上重新去噪 2 步（自产 warm reset）是否胜过 full 与 plain」，并用 `plain_k1` 看减步到 1 步是否开始掉点。
- 吞吐：客户端时间几乎全在等推理，单次推理约 0.55 s（GPU 被 6 个 GR00T server 占算力；同时段 π0.5 LIBERO 冒烟中位 723 ms/决策）。GR00T 让位后应显著改善；§5 的规模按此保守估计。

## 2. 记法与环境

- 记法同网页：T = 起点动作的 t（0 = 最终动作），N = 实际执行的去噪步数，t = 传给模型的 t（π0.5：1 噪声、0 干净）。
- **环境 `pi05_metaworld`**：policy `pi05`，benchmark `metaworld_mt50`，`action_horizon=5`，`action_dim=32`，`n_executed=4`（xyz + 夹爪），`k_full=10`，schedule `pi05_v1`，`warm_ts=(0.2,)`，`k_set=(1, 2)`；server 与 worker 均在 weilandserver。K=10 与 RoboCasa / LIBERO 一致（RLinf 自评用 5 步，不作为本实验 full）。
- **仿真约定**（worker 侧，全部来自 RLinf standalone 评测，逐条入测试）：`metaworld==3.0.0`；每集 `MT1(env_name, seed=MW_BENCH_SEED)`，`env.set_task(mt1.train_tasks[idx])`（idx = episode 身份，0..49，同一 (任务, idx, seed) 跨臂初始状态相同 ⇒ 逐集配对）；`render_mode="rgb_array"`、`camera_name="corner2"`（断言相机 id 2 即 corner2）、`model.cam_pos[2] = [0.75, 0.075, 0.7]`；渲染 480×480，`img[::-1, ::-1]`（180° 旋转）后 `np.ascontiguousarray`；state = `obs[:4]`（模型不读，仍按 openpi 管线传）；`reset()` 后 15 步零动作沉降；每次推理执行 5 步（H_EXEC=5，整块开环）；每集至多 160 个策略步；`info["success"]` 任一步为真即成功并**结束该集**（与 RLinf success-once 同判定；提前结束用于统计成功集推理调用数）；动作不裁剪不缩放（环境内部 clip 到 [-1, 1]）。
- **prompt**：RLinf `metaworld_config.json` 的 50 条原文（大小写保留；`push-back-v3` = 训练时标签 "Push the puck to a goal"，**不用** LeRobot 2026-09 改后的新文本）。难度分组（easy 28 / medium 11 / hard 6 / very hard 5）同 RLinf。
- **server 侧**：`src/openpi/policies/metaworld_policy.py` 的 `MetaworldInputs`（只 `base_0_rgb` 为真图，两腕位零图、mask False；state 与 prompt 透传）/ `MetaworldOutputs`（`actions[:, :4]`）；`src/openpi/training/config.py` 新增**仅推理**的 `TrainConfig(name="pi05_metaworld", model=Pi0Config(pi05=True, action_horizon=5, discrete_state_input=False), data=<metaworld data factory, AssetsConfig(asset_id="metaworld_mt50")>)`（照 `pi05_robocasa` 条目的写法；归一化由 `create_base_config` 按 PI05 取 quantile，与 RLinf 一致；`max_token_len` 由 pi05 默认 200）。
- 实验 id：`sdiag_mw_self`；数据根独立：驱动 `exp/step_diag/data/metaworld_self`，server `exp/step_diag/data/server_metaworld_self`。`MW_BENCH_SEED = 7`（与 LIBERO env seed 同值，仅为约定）。

## 3. 臂（6 个，与 RoboCasa π0.5 同名同语义）

| 臂 | 起点 | T | N | t 序列 | 每决策 NFE |
|---|---|---|---|---|---|
| `full` | 噪声 | — | 10 | 1, 0.9, …, 0.1 | 10 |
| `plain_k2` | 噪声 | — | 2 | 1, 0.5 | 2 |
| `plain_k1` | 噪声 | — | 1 | 1 | 1 |
| `selfwarmreset_t0.2` | 自产 K=10 运行在 T=0.2 的快照 | 0.2 | 2 | 1, 0.5 | 10 + 2 |
| `selfresetfinal_t0.2` | 自产 K=10 的最终动作 | 0 | 2 | 1, 0.5 | 10 + 2 |
| `selfmidfinal_t0.2` | 自产 K=10 的最终动作 | 0 | 2 | 0.9, 0.45 | 10 + 2 |

自产运行的噪声 = 私有 seed（`DiagSession.self_start_seed`，与 RoboCasa / LIBERO 同一公式：实验 id、env、任务、(seed, idx, pool/bench 身份)、attempt、决策序号、"self"），不动全局 RNG；续跑用的是 `exp.step_diag.pi05.warm_variant_stage3` 同一实现，数值语义与 RoboCasa / LIBERO 自产臂逐位相同（§4.2 测试保证）。

## 4. 实现

1. **policy 与推理配置（src）**：§2 的 `MetaworldInputs` / `MetaworldOutputs` 与 `pi05_metaworld` 推理配置；`ops/serve_pi05.sh` 与 `serve_diag_pi05.main` 的期望配置表加 `pi05_metaworld` → `pi05_metaworld` + ckpt 路径（`serve_diag_pi05.py:170` 目前二选一写死 robocasa / libero）。
2. **无库自产路径（`exp/step_diag`）**：
   - 现状：自产臂在 `CACHE_MODES`（`serve_diag_pi05.py:44`），`--cache-config` 必填（`:69`）；`Pi05DiagInterceptor` 只在 `model.run_stage3_from` 的包装里把起点换成自产（`pi05.py:195-205`），而该入口只在检索判出 WARM_START 时才被调用，没有库就永远走不到。
   - 新增：EnvSpec 标记 `cache_free`（仅 `pi05_metaworld` 为真）。cache_free 环境的自产臂**不接受** `--cache-config`、服务端不装载 cache（与 plain / full 同，`--cache` 仍给出以走 interceptor）；`Pi05DiagInterceptor` 在 `self_start and cache_free` 时把 no-cache 分支的 stage-3 绑定（`self._stage3_fn`，`pi05.py:219`）换成「`_self_start(stage2, like, start_t, num_steps=k_full)` → `warm_variant_stage3(model, stage2, start_x, start_t, num_steps=remaining_steps(start_t), variant)`」。`like` 的形状取模型 `(action_horizon, action_dim)`。
   - 计数与证据：与缓存触发的自产臂同口径——`executed_steps = N`、`n_stage3_calls = 1`、`self_direct_nfe = K`、`self_seed`；`hit_type` 记为新值 `SELF_START`（不伪装成 cache 命中），`start_t = 0.2`。`evidence.manifest_problems` 已只对 `shadow` / `warm*` 臂要求库身份（`evidence.py:52`），自产臂无库可准入；新增：cache_free 环境的自产臂每决策必须 `hit_type == SELF_START` 且带 `self_info`，否则拒收。
   - 其他环境（RoboCasa / LIBERO）行为逐位不变。
3. **envs**：`pi05_metaworld` 行（§2）；MetaWorld 常量（实验 id、`MW_BENCH_SEED`、每任务集数、160 步上限、15 步沉降、50 任务名 + prompt + 难度表）；臂表 `MW_SELF_ARMS`（§3 六臂）；`validate_arm` 为 cache_free 环境放行 full / plain_k{1,2} / 三个自产臂且拒绝 yaml、拒绝其他任何臂（尤其 `warm*` 与缓存 reset 臂）。
4. **MetaWorld episode runner 与驱动**（新文件 `exp/step_diag/run_metaworld_diag.py` + worker 侧 runner）：照 `run_libero_diag.py`（launch 清单 / journal / per_step `episode_summary` / `--run-prefix` / `--tasks` / `--episodes` / 期望身份 = (env, 任务, bench seed, idx)）；worker 用 `~/metaworld_sim`（Python 3.11 + metaworld 3.0.0 + openpi-client，已建；运行时 `PYTHONPATH=<repo>:<repo>/src`，`MUJOCO_GL=egl`，`MUJOCO_EGL_DEVICE_ID` = worker 分到的 GPU）。已核实（2026-09-25 15:00 CDT）：以 `PYTHONPATH=<repo>:<repo>/src` 在该 venv 中 `openpi.conductor{,.worker,.driver,.task,.strategy}`、`exp.step_diag.{envs,worker_entry,run_libero_diag}` 均可导入，无需补依赖；若实现新增依赖，只补进该 venv（不动项目 `.venv`）。
5. **队列**（`ops/libero_queue.py` 扩展，不新开调度器）：把 `pi05_metaworld` 作为 π0.5 作业加入 `sdlq` 的持久状态（`load_state` 对已有状态**追加**缺失作业 id，已完成 / 在跑作业不动）；cell = (臂, 任务) 共 300 个，每 cell 20 集；worker 在 weilandserver 本机启动（新 `ops/run_mw_cell.sh`，不经远端）；server 端口与预算沿用 wls π0.5 的 `COST`。优先级：π0.5 作业已在 `sdlq` 状态里，`self13_queue.libero_pi05_waiting`（`self13_queue.py:201`）会让 RoboCasa GR00T slot 在 cell 间隙让出，LIBERO GR00T 本就等 RoboCasa ⇒ 得到「MetaWorld π0.5 > RoboCasa GR00T > LIBERO GR00T」，无需新机制。另含已在工作树的 serve 输出轮转修复（`rotate_serve_out`，2026-09-25）。
6. **分析**：`warm_variants.py` / `success_length.py` 支持 `pi05_metaworld`（任务 = 50 个 env 名，配对身份 = (env, 任务, bench seed, idx)）；另出 RLinf 口径（四难度组均值及其简单平均）与合并率；self vs full / plain_k2 / plain_k1 配对差与成功集推理调用数。
7. **测试**（CPU，`tests/exp/step_diag/` + `tests/policies/`）：变换（键、mask、4 维切片、图像不被改动）与推理配置（asset_id、quantile、max_token_len）；无库自产路径：在带库环境上构造同一 stage-2、同一 seed，**无库路径输出 == 缓存触发自产路径输出（逐位）**，计数一致；cache_free 臂校验（拒 yaml、拒 warm / 缓存臂）；runner 的观测构造（相机名断言、翻转、沉降步数、160 步上限、成功即停、prompt 表含 push-back 原文、同 (任务, idx, seed) 初始观测跨进程相同）；驱动期望身份；队列追加作业与优先级门；分析在合成数据上的配对与分组均值。

## 5. 运行（G2 + Verify 之后）

- 部署：wls 上 LIBERO 服务树 `/data/openpi_sdlib` 同步到新提交（git），ckpt 已在 `/data/ckpt/pi05_metaworld_rlinf`。冒烟：2 任务 × 2 集 × 6 臂，核对证据准入与每决策 NFE（full 10、plain 2 / 1、自产 12）后正式开跑。
- 规模：6 臂 × 50 任务 × 20 集 = 6,000 集（每任务 20 集：宏观 50 任务、每臂 1,000 集，已足以看 ±3 个点的宏观差；若结论接近再补到 50 集，idx 20..49 追加即可配对）。估算：成功集约 14 次调用、失败集 32 次，平均约 22 次/集，约 13 万次推理（自产臂按 1.2 倍）；按 GR00T 让位后 wls 上 3–5 个 π0.5 server 计，约 6–12 小时。

## 6. 不做的事

- 不建库、不检索、不收集 cache；不跑精确续跑（ours）与缓存起点的 reset 臂；不跑 GR00T MetaWorld。
- 不改 RoboCasa / LIBERO 已跑与在跑的臂语义；不迁移到 warm reset 一等公民框架（迁移研究已列为后续）。

## 7. 为何 step_diag 而非新框架（owner 已裁定，记录理由）

新框架的自产臂同样依赖 WARM_START 检索命中，且 full / plain_k 不在其入口内，要先做「无库自产触发」（L3）与「按 bundle 指定 MISS 步数」（L2）两个框架特性，并写分析适配层；step_diag 只需本计划 §4.2 的局部分支，结果与 RoboCasa / LIBERO 同口径（单连接 B=1），分析脚本与网页直接复用。

## 8. 风险

- 吞吐受 GR00T 共享 GPU 影响（§1）；队列优先级使 GR00T 让位后缓解。
- `metadata.pt` 里 `action_env_dim: 7` 与 4 维 norm stats / 4 维动作不一致：以 RLinf 源码（`[:, :4]`）与 norm stats 为准，冒烟复现已验证。
- MuJoCo EGL 多进程：`MUJOCO_EGL_DEVICE_ID` 必须在 `CUDA_VISIBLE_DEVICES` 内（同 `2f0116c` 修复）；解释器退出时的 EGL 析构报错无害。

## Review Log
