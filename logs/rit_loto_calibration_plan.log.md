# RIT 离线标定（LOTO）与闭环验证：实施计划

> Status: **§5 G2 — R2 APPROVED（owner 授权 Reviewer 直接修复并验收；§6 Verify 待执行）**（验证证据见 `exp/rit_loto/analysis/smoke_evidence.md` 与本文件 G2 R2 Review Log）| Level: **L2**（新实验包 + 服务端日志工厂；本任务不修改 `src/`）| Authority: Review + owner override | 2026-09-14
> 术语：**RIT** = Risk-Indexed Threshold；**LOTO** = leave-one-trajectory-out（留一整条轨迹）；**shadow 标定** = 现行做法，教师驱动 150 集 cohort、缓存在影子中打标（`GrootRitShadow`）。
> 数学权威：论文 Appendix C（`docs/iclr/iclr_paper/iclr2027_conference.tex` §C.1–C.2，Eq. `shadow-deviation` = owner 所称 Eq. 16）。上位记录：[`libero_groot_rit_run_progress.md`](libero_groot_rit_run_progress.md)。
> 审查记录：G1 R1 NEEDS REVISION（B1–B7 / N1–N4，执行方全部采纳）；R2 由 owner 授权审查者直接修订正文后 APPROVED。执行方于 2026-09-14 复核审查者修订并核对其依据的三项代码事实（`ThresholdJudge` 对低于全部切点的候选返回 MISS 且 `winner_id=None`、两份 `arm_record.json` 均无 `alpha` 字段、`EpisodeDataCollector` 额外 attrs 须经 `set_episode_attr`），无异议，全部采纳。G1 Review Log 按执行法 §3.1 已删除，G2 另开新节。

## 0. 需求（owner 2026-09-13 口径，逐条对照）

| # | owner 要求 | 本计划落点 |
|---|---|---|
| R0 | 任务 0：数据盘点（库语料是否在盘、成败计数、字段、主实验日志字段） | §1（已完成，结论在 §1.7） |
| R1 | 任务 1：用已采集轨迹重建 shadow 表，不跑新 episode；库内轨迹整条排除，库外轨迹用完整库；每档 â^(a) 从候选的 h_e^(a) 在当时观测下续跑；â^ref 直接用采集时执行过的 chunk；按 Eq. 16 算 D_a；用现有 LP 拟合 | §2.1 `build_loto_table.py` + §2.2 `fit_loto.py` |
| R2 | 任务 2：LOTO q_a 与原 shadow q_a 同图；ψ(s) 对比；in_library 分层拟合；行数 | §2.2 |
| R3 | 任务 3：LIBERO-10 50 集闭环，部署配置（δ 工作点 + gate），逐决策记观测/s_t/tier/执行 chunk/candidate_id；抽 ~2000 行离线算 ref 与 D_a；判分位带 | §2.3 |
| R4 | 任务 4：~500 观测 × 2 噪声种子的 D 分布（中位数、90 分位） | §2.4 |
| R5 | W 全局共享（非逐档）；只比前 H_exec 步；LOTO 排整条轨迹；失败轨迹只做查询源；不重写拟合代码 | §2.1 的 W 来自 `library_action_weights(S3.pkl)`；`h_exec=5`；排除按 `trajectory_id`；库文件不动；拟合只调 `emit_rit_rc.fit_ladders` |
| 范围裁定 | **只跑 GR00T N1.5 × LIBERO（spatial / libero_10）**，pi0.5 不跑（owner 2026-09-13） | 全文 |
| 设备裁定 | timan107 / timan108 只跑 client worker；weilandserver / h100 跑 server；数据主体在 weilandserver（owner 2026-09-13） | §5 |

## 1. 背景事实（任务 0 盘点，全部实测于 2026-09-13）

### 1.1 建库语料在盘，且失败轨迹保留

W13 采集（2026-09-08，`launch_collection.sh`）对 B 池**每任务全部 50 个 init 各跑一集**，不是"失败就换下一个直到攒够 5 个成功"；库 S3 = 每任务从成功集里按 `seed 0` 抽 5 条（`<suite>_w13_manifest.json`）。位置 weilandserver `/archive/libero_cache/build_{spatial,libero10}_w13/<suite>/episode_<gid>_<ts>.h5`，`gid = task_id*50 + init_idx`。

| suite | 集数 | 成功 | 失败 | 决策总数 | 均/集 | 体量 | 库 S3 轨迹 | 库轨迹全在语料 | 库内含失败 |
|---|---|---|---|---|---|---|---|---|---|
| libero_spatial | 500 | 450 | 50 | 11,838 | 23.7 | 26 GB | 50（5×10） | 50/50 | 0 |
| libero_10 | 500 | 436 | 64 | 29,318 | 58.6 | 63 GB | 50（5×10） | 50/50 | 0 |

逐任务失败数：spatial `12,5,2,1,2,2,2,9,9,6`；libero_10 `8,4,4,2,7,2,12,3,5,17`（task 0..9）。
⇒ 任务 1 查询集：库内 50 条（LOTO）+ 库外 **450 条**（其中成功 400/386、失败 50/64）。库外成功轨迹占多数，但失败轨迹不是零，偏差要在 §2.2 分层里读，而不是靠"只有成功"这条注脚。

### 1.2 每条轨迹存了什么（两 suite 同 schema）

- 文件属性：`task`（自然语言指令）、`task_id`、`orig_init_state_idx`、`episode_id`、`success`、`num_steps`、`denoise_schedule_id=groot_n15_k8_v1`、`denoising_num_steps=8`、`timestamp`。
- 每个 `step_XXXX`（= 一次决策，每 5 个控制步一次）：`vision_0`/`vision_1` `[256,2048] f16`（两路相机的 stage-1 图像 token 段）、`prompt_emb` `[n_tok,2048] f16`（54 / 47 个文本位置）、`robot_state` `[8] f32`、`noise_action_0` `[16,32] f32`（纯噪声起点）、`noise_action_1..7`（Euler 第 i 步消费的 x_t，t=i/8）、`clean_action` `[16,32] f32`（当时执行的教师 chunk，归一化动作空间）。
- **无原始图像**。stage-1 序列可由指令模板 + 存储切片精确重建（`exp/libero_groot/cp2_reconstruct.py`，文本位与状态逐字节断言；已被 CP2 parity 门验证过 key 侧）。动作头续跑一侧的保真度本计划自带自证（§2.1 parity 列）。
- 库条目（S3 pkl）：`id="<stem>:<step>"`、`trajectory_id=stem`、`step_idx`、`payload.action_chunk [16,32]`、`payload.intermediates {0.125..0.875}`（7 个快照）、`denoising_num_steps=8`、`task_key=指令`、`query_keys` vision_0/1/2（32768）+ prompt_emb（2048）+ robot_state（8）。`outcome=None`。

### 1.3 现行 shadow 标定（要被替代的对象）

- cohort：每 suite 150 集，A 池（`libero_<suite>_apool`，官方 pruned 50/任务）按 seed 20260901 抽 5 fit + 10 cal / 任务；教师驱动，`GrootRitShadow` 影子打标。
- 行：`exp/libero_groot/data/rit/shadow/<suite>/shadow_rows.jsonl`（3,432 / 8,383 行；weilandserver `/data/openpi_lg/exp/libero_groot/data/rit/shadow/<suite>/shadow_p*.conn_*.jsonl` 为原始分片），字段 `task, episode_id, step_idx, s, winner_id, y_full, y_rem2, y_rem4, episode_success`。
- **参考 = 该步教师自己的完整推理 chunk**（`GrootRitShadow._label` 里 `ref = teacher`），与任务 1 的 ref 定义一致，任务 2 的比较不混入 ref 口径差。
- `W, active_mask = library_action_weights(S3.pkl)`（全库所有 chunk 所有 16 步的逐维逆标准差，一个矩阵，非逐档）；`h_exec=5`（= client `--replan-steps 5`）。
- 拟合：`exp/robocasa365/emit_rit_rc.fit_ladders(rows, cost, warm_ts=[0.75,0.5], ks=[1,2,3], alpha=0.05, ir_sample=s)` → `rit_cost_rc.fit` → `rit_k.fit_pl_quantile_k`（结点分段线性 pinball LP，ε_total=0.02 严格单调 + 层嵌套，HiGHS）。`arm_record.json` 存了 knots、n_seg_req、每臂 δ/cuts，**没有存 q 值** ⇒ 原曲线由同一代码在 `shadow_rows.jsonl` 上重拟合复现，并以 knots 与各臂 cuts 逐值对账（§2.2）。
- 当时寻址用的成本表是 INTERIM（RoboCasa 借用），只影响 δ→IR，不影响 q_a。本计划一律用现行权威 `exp/libero_groot/config/rit/cost_groot_libero_measured.json`。

### 1.4 主实验（84 臂 × 500 集/suite）日志字段

`run_gtp` 产 `journal.jsonl` + `per_step.jsonl`，在 timan107/108 `/scratch/zixuans8/openpi_lg/exp/libero_groot/data/rit/eval/<suite>_{anchor,hg}/`（本地只有聚合 `exp/libero_groot/data/rit/eval/aggregate_<suite>.json`）。per_step 每决策一行：`yaml_id, task_id, subset_init_state_idx, orig_init_state_idx, episode_id, task_uid, phase, step_idx（控制步，0/5/10…）, hit_type, start_t, winner_id, cp1_score, checkpoint, score, library_sha256, searched, executor, router_outputs, factor_outputs, success, attempt, accepted, run_id`。**没有观测、没有执行的 chunk** ⇒ 任务 3 不能复用现成日志，必须新增服务端逐决策记录（§2.3）。

### 1.5 设备与环境

| 节点 | 角色 | 状态（2026-09-13） | 关键路径 |
|---|---|---|---|
| weilandserver | server + 离线计算 | 4090 49 GB 空闲；88 核 / 251 GB | 语料 `/archive/libero_cache/build_*_w13`；库 `/data/libero_cache/libraries_w13/<suite>/<suite>_w13_S3.pkl`；ckpt `/home/weiland/ckpt_n15_libero_{spatial,10}`；岛 venv `/home/weiland/gr00t_n15_venv/.venv/bin/python`；gr00t `/home/weiland/gr00t_n15`；岛克隆 `/data/openpi_lg`（3af3ef2 + 40 个已推送文件；本计划依赖的 21 个文件与本地逐字节一致，已核）；LIBERO client `conda run -p /home/weiland/libero_sim` |
| h100 | server（备用） | H100 80 GB 空闲 | `/data/openpi_lg`、S3 两库、岛 venv；**无 W13 语料**、ckpt 位置未核 |
| timan107 | client worker | 8×1080 空闲 | `/scratch/zixuans8/openpi_lg`、`HOME=/home/zixuans8`、`--conda-env /scratch/zixuans8/libero_sim` |
| timan108 | client worker | 3×A5000 空闲 | 同上 |

### 1.6 亲验的接口（计划引用的每个符号）

- 检索（与 `exp/libero_groot/analysis/offline_weight_search.py` 同法，该脚本已与生产栈逐条核对 ≤2.4e-5）：`InMemoryBackend(vector_dims=cfg.backend.vector_dims)`；`CacheStorage(backend)`；`storage.batch_insert(entries)`；`backend.freeze()`；`openpi.cache.config._build_search_strategy(ss_cfg, storage, weights: dict, min_top_k_hint=n)`；`SearchContext(query_keys, CheckpointID.CP1, step_idx, task_key)`；`strategy.search(ctx)` → 按分数降序的命中列表，`hit.id / hit.score`。`task_scoped` 默认 True（`config.py:439`）⇒ 候选限同任务。**enabled-fields 契约**：后端 `_iter_active_fields` 把"出现在 query_keys ∩ vector_dims 且 weight>0"的字段都算进融合，未传权重的字段按默认权重参与；模板 yaml 禁用了 `prompt_emb` / `vision_2`，所以查询 key 必须按 `[(name, k.enabled, k.weight) for name, k in _keys_iter(cfg.keys)]` 过滤到 enabled 字段，且 `fusion_weights` 传完整映射（禁用字段显式 0.0），字段顺序取 `_keys_iter` 的固定顺序，归一化取 yaml 的 per_field normalizer（`offline_weight_search.py:93` 同法）。
- 查询 key（与库条目同一条构建链）：`exp.common.build_in_memory_cache_artifact._create_builder("cp1_groot_libero_spatial_pool_16")`（内部映射到 `CP1SpatialPool16KeyBuilder`）；`_build_fake_stage1(group)`；`builder.collect(CheckpointID.CP1, stage1=...)`；`builder.build(CheckpointID.CP1)`；`builder.clear()`。
- 重建与续跑（岛 venv）：`cp2_reconstruct.load_groot_libero_policy(ckpt, denoising_steps=8)`；`GrootStagedRunner(policy.model)`；`TemplateCache(policy, runner).get(task)`（首次未命中会调 `build_template` → `runner.run_stage1`，**必须在 `runner.session()` 内**，`staged.py:438` 对无 bf16 autocast 的调用直接 raise）；同一 session 内 `reconstruct_stage1(template, group)` → `runner.run_stage2_llm(stage1)` → `runner.run_stage3_from(stage2, x_t, t, schedule=runner.live_schedule()).action_pred [1,16,32]`；`runner.run_stage3(stage2, noise=z).action_pred`（显式噪声走 pinned 转写循环）。
- 偏差与权重：`surface_judge.weighted_chunk_deviation(chunk_a, chunk_b, w, active_mask, h_exec) -> float`；`rit_shadow.library_action_weights(pkl) -> (w, active_mask)`。
- schedule：`types.groot_n15_schedule(8)`；`schedule.snapshot_index(t)`、`remaining_steps(t)`、`timestep_set`；`collect.h5_intermediates.episode_schedule(h5)`。
- 拟合与臂：`emit_rit_rc.fit_ladders(rows, cost, warm_ts, ks, alpha, ir_sample=None, knot_sample=None)`（每个 k 返回 `{"fit": PLFitK, "tiers", "s", "ir_s", "n_rows", "ir_range", "knots", "n_seg_req"}`）；`rit_cost_rc.delta_for_ir(fit, s, target, cost)`、`cuts_for(fit, delta)`、`predict(fit, s, tier)`；`emit_rit_arms.load_cost(path) -> StageCost`、`gate_theta(scores)`（0.85 分位）、`_judge_from_cuts(cuts, warm_ts)`、`build_arm(template, judge, layer=LAYER_SECONDARY, theta)`、`write_arm(doc, path)`。
- 服务端日志：`GrootCacheInterceptor(policy, runner, orchestrator=..., timer=...)`，其 `get_action` 依次调 `runner.run_stage1` → `orchestrator.check(CP1, stage1=)` → 按档取 chunk → `orchestrator.broadcast_action(action_cpu)`（**所有档都会广播实际执行的归一化 chunk**）→ `orchestrator.clear()`；返回附 `__hit_meta__ {hit_type, start_t, winner_id, cp1_score, searched, checkpoint, score}`。`slice_groot_cp1_fields(input_embeds, image_token_mask, state, state_mask, enabled=None, expected_state_index=, vision_fields=(VISION_0, VISION_1))` 切出 H5 同款字段。`EpisodeDataCollector(base_dir)`：`on_episode_start(experiment, task, episode_id, episode_name=, extra_metadata=)`、`record_inference(InferenceEmbeddings)`、`set_episode_attr`、`on_episode_end(success)`；`noise_action_steps=[]`、`init_noise=None` 时只落 `clean_action`。`serve_groot_libero._build_shadow_factory` 是 `--concurrent` 下按连接建栈的现成范式（`_InferLockedPolicy`、`_require_default_bundle`、`GrootLiberoPolicyAdapter` 转发 episode 钩子与 `__` 侧信道）。
- A 池子集：`exp.rit_pareto.shadow_cohort.sample_assignment(task_names, seed=, fit_per_task=, cal_per_task=)`、`exp.dispatch_surface.split_init_pools.materialize_pool(apool_dir, out_dir, assignment, keys)`；client `examples/libero/main.py --init-states-dir <dir> --num-trials-per-task N --task-ids t --replan-steps 5 --resize-size 256`（`_load_init_states` 先找 `.pruned_init` 再 `.init`）。

### 1.7 任务 0 结论

1. 语料齐全且含失败轨迹，任务 1 可做，且库外查询源同时含成功与失败（§1.1）。
2. 现行 shadow 的 ref 定义、W、h_exec 与任务 1 一致，任务 2 是干净的"rollout vs 离线"对照（§1.3）。
3. 主实验日志缺观测与执行 chunk，任务 3 需新增服务端日志（§1.4）。
4. 离线计算可全部落在 weilandserver（语料、库、ckpt、岛环境、空闲 4090 都在），任务 3 的 server 也放 weilandserver，client 放 timan107（§1.5）。

## 2. 设计

### 2.1 任务 1：`exp/rit_loto/build_loto_table.py`（岛 venv，GPU）

输入：`--suite`、`--corpus-dir`、`--library-pkl`（S3）、`--template-yaml`（`exp/libero_groot/config/rit/<suite>/template.yaml`，带该 suite 的 normalizer 与权重）、`--checkpoint`、`--warm-ts 0.75,0.5`、`--h-exec 5`、`--out-dir`、`--task-ids`（分片；未合并的单片不能进入正式拟合）、`--limit-episodes`（smoke）、`--parity-sample N`（parity 抽检规模，按任务分层等额，默认 200/suite）、`--parity-only`（只跑 parity 抽检并写门判定）、`--orchestrator-check N`、`--noise-floor-record`（§2.4 产物，parity 门的标尺）。

每条轨迹每个决策点：
1. 查询 key：`_build_fake_stage1(group)` → 同库构建链 → 全字段 key，再**按模板的 enabled 字段过滤**（`_keys_iter(cfg.keys)` 顺序；本线为 `vision_0, vision_1, robot_state`），`fusion_weights` 传 `_keys_iter` 全字段映射且禁用字段为 0.0。
2. 检索：`_build_search_strategy(ss_cfg, storage, fusion_weights, min_top_k_hint=该任务库条目总数)`；`strategy.search(SearchContext(query_keys_enabled, CP1, step_idx, task_key))`，命中按分数降序。**库内轨迹**（`stem ∈ library trajectory_ids`）：取第一个 `entry.trajectory_id != stem` 的命中；**库外轨迹**：取第一个命中。融合分是逐对量（各路 zscore+tanh 后加权和），不依赖候选集合，事后剔除与"从缩小的库检索"逐值相同——这一点不靠推理，由 §6-1b 的真缩库对照测试和 `--orchestrator-check` 的生产栈自证共同锁定。断言库内查询至少剩一条他轨迹（同任务 5 条，恒成立）。记 `n_self_skipped`。
3. 续跑：在**同一个** `runner.session()` 内依次 `template = TemplateCache.get(task)`（首次未命中在此构建）、`stage1 = reconstruct_stage1(template, group)`、`stage2 = run_stage2_llm(stage1)`、对每个 t∈warm_ts `run_stage3_from(stage2, payload.intermediates[t], t, schedule)`；检索（第 1–2 步）在 session 外。任务 3/4 同一契约。
4. 偏差：`ref = group["clean_action"]`；`y_full = dev(payload.action_chunk, ref)`、`y_rem2 = dev(warm@0.75, ref)`、`y_rem4 = dev(warm@0.5, ref)`，`dev = weighted_chunk_deviation(·,·,w,active_mask,h_exec)`，`w` 来自 `library_action_weights(S3.pkl)`，全表一份。
5. **parity 抽检与止损门（在拟合之前、在 §2.4 之后）**：对 `--parity-sample` 个不同决策（按任务分层等额，两 suite 各自独立）从存储的纯噪声重放 full，并从自身快照重放 warm75 / warm50；快照索引均由 `schedule.snapshot_index(t)` 取得。H5 数组先转 torch，full 噪声显式补 batch 维 `[1,16,32]`，所有模型输出统一转为 `[16,32]` CPU fp32 后调用 dev。记录诊断列 `parity_{full,warm75,warm50}_maxabs`（active 维、前 h_exec 步，未加权）与**门用列** `parity_D_{full,warm75,warm50}`（与 D 同一 W / mask / H_exec）。门按 suite、按三种重放分别报 p50/p90，要求 `p90(parity_D_*) ≤ 0.1 × median(D(ref1,ref2))`（同 suite 的 §2.4 标尺），抽检 ≥200 个不同决策且每任务 ≥20 行；所有值须有限、非负，noise-floor 标尺至少 500 观测（每任务 50）且记录行数相符。若地板为零，仅允许全部 parity_D 为零，否则 FAIL；地板小于零或没有活动维属于非法输入。`parity_gate.json` 记录 PASS/FAIL、阈值、样本清单和库/ckpt/模板/语料清单/W/mask/H_exec/schedule/代码版本身份。入口和 `fit_loto.py` 都要求明确 PASS 且身份匹配；缺文件、未知状态或旧门与新输入不一致均拒绝继续。运行顺序固定：§2.4 → `--parity-only` → 门 PASS → 全量任务 1 → 拟合。
6. 每 `--orchestrator-check` 行附检索自证：用模板 yaml 建 `CacheOrchestrator`（always_search / always_hit）对同一 `stage1` 跑 `check`，核对库外查询的 winner 与 score 与第 2 步一致（`build_shadow_table_groot` 的 backend-check 范式）；同时把 `prompt_emb` 查询向量置换成随机向量再跑第 2 步，断言 s 与 winner 不变（禁用字段确实不参与打分）。

输出行（JSONL，主 venv 再转 CSV）：`trajectory_id=stem, episode_id, decision_id(step_idx), task, task_id, orig_init_state_idx, library_id(S3 sha256 前 12 位), library_seed(0), s, candidate_id, y_full, y_rem2, y_rem4, in_library, episode_success, n_self_skipped`；行主键为 `(suite, trajectory_id, decision_id)`，库的完整 sha 留 record，拒绝重复行。parity 抽检另落 `parity_sample.jsonl`（六个误差列 + 行身份）。旁存 `<out>.weights.npz`（w, active_mask）与 `<out>.record.json`（库/模板/ckpt/语料清单 sha、enabled 字段与权重映射、行数、git commit 与实际代码文件 sha、schedule、H_exec、parity 门文件 sha）。`load_library` 先验证 artifact/payload/H5/live runner 的 schedule 一致、两种快照存在且形状合法，不能用 live schedule 给未知来源快照补身份。

### 2.2 任务 1 拟合 + 任务 2 对比：`exp/rit_loto/fit_loto.py`（主 venv，CPU）

- 输入：完整 LOTO 表及 record（必须覆盖 10×50 个不同 task/init；核 corpus/处理集数/实际行数，拒绝分片、重复主键及同一轨迹的身份冲突；K 固定 {1,2,3}）、`parity_gate.json`（身份匹配且明确 PASS 才继续）、原 `shadow_rows.jsonl`、`cost_groot_libero_measured.json`、`arm_record.json`。
- 拟合四份并做一次对账，全部只调 `fit_ladders`（k=1,2,3；alpha 0.05；`ir_sample` = 该份自己的 s）：① LOTO 全部；② LOTO `in_library=True`；③ LOTO `in_library=False`；④ 原 shadow 重拟合；⑤ ④ 的对账。先核 shadow 表/template SHA、`warm_ts`、k 集合和 tier 名；**现有两个 arm_record 均没有 alpha 字段**，不得读取不存在的键后报错或声称已从记录核实 alpha。legacy 路径显式用 `alpha=0.05`（原 emitter 的 DEFAULT_ALPHA），标记 `alpha_source=legacy_reconstruction`，并以全部 knots / cuts 对账确认所复现的曲线；未来带 alpha 的记录则要求值一致。knots 与 `arm_record.fits[k].knots` 逐值相等；对每个 RIT 臂用记录 δ 调 `cuts_for`：仅 `+inf` ↔ JSON `null`，NaN/-inf 非法；有限值差 ≤1e-9，否则失败。⑤ 不过即停。数据缺失/非有限值不静默丢弃；若某个分层确实不满足现有 knot 最低样本要求，记录 `fit_unavailable`、原因及行数，不伪造曲线；主 `loto_all` 或原 shadow 拟合不可用则终止后续发臂。
- **曲线差的精确计算**（N1）：两条分段线性曲线之差的绝对最大值在"公共支持域端点 ∪ 两组拟合 knots 并集（落在公共域内的）"上精确取得；`compare.json` 逐档记 `max|Δq|` 及其所在 s、按原 shadow 的 knot 段分段的 Δq；展示用网格另按两表 s 的经验分位布点；单报各表落在公共支持域外的行数。
- **参考曲线的不确定性显示**（N3）：对原 shadow 按任务分层、有放回重采样完整 episode，固定 B=200 与 seed，每次仍调 `fit_ladders`。同一 episode 被抽中多次时按重数保留行，不按 episode_id 去重。每次在固定展示网格预测；超出该次拟合 knots 支持域的点记不可用，不能用 `predict` 的端点裁剪伪装覆盖。预期的 knot 样本不足可以记失败及原因，意外异常直接报错；不重抽来替换失败样本。每点至少 180/200 个有效重复才展示 2.5/97.5 分位带，其他点留空并报有效数。这是**仅描述原 shadow 估计波动的逐点近似带**，不作为整条曲线的同时置信带或两种方法之差的显著性检验；只报告在哪些点超出参考带及差值，不据此判等价。另报工程尺度 `τ_a := p90_i |D_{i,a} − q̂_a^{shadow}(s_i)|` 与 `max|Δq_a| / τ_a`；τ=0 时仅报原始差值与零尺度标记，不除零。
- ψ(s) 对比：两表 s 的 p05/p50/p95、≥0.99 占比、KS 距离；②③ 之间同法；行数表（按 in_library × episode_success 四格）。
- 产物路径（B7）：数值 → `exp/rit_loto/data/<suite>/{fits.json, compare.json, bootstrap_band.json, parity_gate.json}`；图与 Markdown → `exp/rit_loto/analysis/<suite>/equivalence.png` 与 `exp/rit_loto/analysis/results.md`。`fits.json` 每份保存 source、输入 SHA、knots、各档 q、tiers/y_key/cost、alpha、eps_total、n_seg_req/n_seg、行数与 s 分位、ir_range；反序列化须可还原 PLFitK，不能重新拟合来冒充读取已冻结曲线。渲染脚本 `exp/rit_loto/analysis/plot_equivalence.py` 按 owner 规则不入库。

### 2.3 任务 3：闭环验证（LIBERO-10）

- **拟合来源**（N4 / D5）：本轮依据 owner 授权将既有建议冻结为 `fit_source=loto_all`，任务 3 的 q、s 样本、gate θ、δ 同源；库外拟合保留为任务 2 的对照。未来若更换 fit_source，作为独立配置和 run_tag 运行，不在同一 record 中混用。来源选择、两档标签规则和本节统计容差在读取正式验证结果前冻结。
- **臂**：`exp/rit_loto/emit_verify_arm.py` 用冻结的 loto_all k=2 拟合，在目标 IR=70 处反解 δ：`sol = delta_for_ir(fit, s, 70, cost)` → `cuts_for(fit, sol["delta"])` → `judge, dropped = _judge_from_cuts(cuts, (0.75,))` → `build_arm(template, judge, layer=LAYER_SECONDARY, theta=gate_theta(s))` → `write_arm(...)`。输出 `exp/rit_loto/config/libero_10/loto_k2_ir70.yaml`，由生产 loader 验证。record 记来源拟合 SHA、δ/cuts/θ、dropped_rungs、可达范围与预测 IR；目标不可达或 warm 档退化时报告事实，不静默改目标或声称已验证两档。同目标原臂的 δ/cuts/θ 单列作对照（§8-D1）。
- **冻结记录与准入**：`emit_arm` 写 `frozen_run.json`，固定 suite=libero_10、source=loto_all、K=2、alpha=0.05、IR=70、完整两池及索引、measured cost、eager stage-1（`compile_stage1=False`），并保存拟合、模板、臂 YAML、池清单及各阶段代码 SHA。`fit` 核实际源码和 table 当时使用的 parity gate SHA；emit/server/collect/label/report 均核当前代码与冻结版本相等。代码覆盖生成表、LP、臂、检索、forward、interceptor、日志和报告入口。checkpoint 每次准入顺序读取文件内容计算摘要，缓存命中也必须重新核验字节，不能靠长度或时间戳推断相同。
- **init 池与身份**（B6）：A 池中排除 150 集 shadow cohort 用过的 15 个/任务后，每任务抽 5 个正式 + 1 个 smoke（seed 20260913，smoke 的 1 个与正式 5 个互斥）。`materialize_pool` 分别落 `exp/rit_loto/data/libero_10/verify_pool/` 与 `smoke_pool/`（写入时按官方 index 升序）；同时生成 `verify_filter.json` / `smoke_filter.json`（`[{task_id, subset_init_state_idx, orig_init_state_idx}]`，subset 位置 = 排序后的位置），client 用 `--init-states-dir <pool> --episode-filter <filter>`，`main.py:604/627` 据此把**官方 index** 经 `extra_metadata` 打到 H5 attrs（`orig_init_state_idx`），sidecar 同步记录。清单 `verify_pool_manifest.json` 记两池 index、sha、排除清单 sha。
- **服务端日志** `exp/rit_loto/loto_logger.py`：`GrootLotoLogger` 组合而非复制 interceptor：两个透明代理包住 `runner`（记录最近一次 `run_stage1` 输出）与 `orchestrator`（记录最近一次 `check` 结果与 `broadcast_action` 的实参，其余方法原样转发），交给真正的 `GrootCacheInterceptor`。每次 get_action 开始清空本次捕获；仅成功返回后写一条记录，异常不沿用上一步缓存。session 外将切片和执行 chunk 复制为自有 CPU numpy 数组交给 `EpisodeDataCollector`；不长期持有 GPU stage1 或 inference tensor。H5 每步存 vision_0/1、prompt_emb、robot_state、`clean_action`=广播的执行 chunk，不存 noise；attrs 含 task_id/orig_init_state_idx，并由 `set_episode_attr` 显式写 run_tag、connection_id、臂/库/ckpt/池清单 SHA、schedule 和 H_exec（额外字段不会自动穿过 collector 的 allowlist）。sidecar 存 `episode_id, decision_id, control_step_idx, task, task_id, orig_init_state_idx, s, hit_type, start_t, winner_id, searched, run_tag, connection_id`。decision_id 从 0 连续计数并对应 `step_XXXX`；control_step_idx=5×decision_id。s/winner_id 保留实际返回值，gate-skip 的 null 不补零。CLI 增 `--loto-log-out DIR --loto-run-tag TAG --loto-frozen-record FILE`，需 `--cache-config` + `--concurrent`，与 `--collect-hdf5` / `--rit-shadow-out` / 动态 bundle 模式互斥；只支持本计划 CP1 配方。工厂沿用原装配守卫、推理锁和 episode 生命周期，按连接写 `DIR/<TAG>/conn_<id>/`。
- **拓扑**：server weilandserver `srv0`（端口 23150）；client timan107 10 进程（一任务一进程，`--num-trials-per-task <池大小> --episode-filter ... --replan-steps 5 --resize-size 256`），照 `launch_shadow_clients.sh`。**smoke 与正式分 run_tag、分池、分目录**：先 `smoke`（1 集/任务，smoke_pool）确认 H5 与 sidecar 对齐、attrs 身份正确；再 `verify`（5 集/任务，verify_pool）。
- **合并规则**（B6）：`collect_decisions` 只读 manifest 指定的正式 run_tag；按 `(task_id, orig_init_state_idx)` 对照清单，要求恰好 10×5 个唯一、已结束的 episode（成功与失败均纳入），再按 `(run_tag, connection_id, episode_id, decision_id)` 一一对应 H5 步组和 sidecar。重复 / 缺失 / 不在清单 / attrs 与 sidecar 或冻结身份不一致 / 混入 smoke / 决策缺号或重号均 `SystemExit` 并打印差集。`.h5.tmp`、未结束 episode、额外 sidecar 行不能静默略过；正式数据目录必须独立于 smoke，断线重跑不得直接与旧尝试混合。
- **产物传递校验**：collect 保存 decisions、episodes、sampled 文件 SHA 和 `sample_manifest.json` 的完整 selected_rows（含分组、在线元数据、H5 SHA），label 在模型加载前核样本文件字节及逐条元数据、H5 内容和冻结 attrs/schedule/坐标；label record 绑定整个 sample manifest SHA、完整行数、输入身份和噪声契约。report 核每份产物摘要，重验精确的 50 个 task/init 及完整 decision 序列，并用全部 decisions 重放固定抽样；标签逐行核对 sample_group、在线 s、winner、task/init/outcome、两档有限标签、离线候选一致性和 ref_seed。任何缺失或不一致即拒绝，`--limit` 仅产生独立命名的 partial 文件，不能进正式 report。
- **抽样与 estimand**：从验证集全部 N 个决策中按 seed=20260914 均匀无放回抽 `min(2000,N)` 行，标 `sample_group=base`；再补入所有尚未入选的 WARM 行，标 `sample_group=warm_extra`。保留样本清单、每类原始/抽中行数和纳入概率；同一决策仅一行。联合闭环重拟合只用 base 行，避免 WARM 过采样改变拟合分布。派发 WARM 超越率用 base ∪ warm_extra 中的全部 WARM 行；派发 FULL 超越率只用 base 中的 FULL 行，单列各档抽样比例，不把两个不等概率样本直接合成总体超越率。样本可因 WARM 补样超过 2000，预算记录实际行数。
- **离线候选与 ref**：对每个选中决策从 H5 按 §2.1 的 enabled-fields 契约，在完整 S3 上离线 always-search、top_k=1 得 `candidate_id` 与 `s_offline`，包括实际被 gate 跳过和阈值拒绝的 MISS。真实 `ThresholdJudge` 的 MISS 返回 `winner_id=None`，即使搜索得到候选也是如此；不能依赖不存在的“MISS 有在线 winner”分支。对于实际 FULL/WARM，离线 candidate 必须等于在线 winner，分数差 ≤3e-5（覆盖既有 2.4e-5 检索核验误差）；对于 searched MISS 的有限在线分数同样核对。失败则报告具体决策并停止生成结论；不静默换候选。gate-skip 的离线分数单独标来源，不能冒充当时在线分数。仅离线确实无候选的行没有候选标签，记原因和覆盖数；实际 FULL/WARM 却无候选属于一致性失败。
- 在 session 内（含首次模板构建）重建 stage1 → stage2 → `run_stage3(..., noise=z)` 得 ref。噪声为 `[1,16,32]`，用固定 root seed=20260914 与 `(suite, run_tag, task_id, orig_init_state_idx, decision_id, "verify_ref")` 的规范 JSON UTF-8 编码的 SHA256 前8字节（大端无符号整数）派生种子，不用 Python hash 或遍历次序；用每行独立 `torch.Generator(device="cpu")` 生成标准正态 fp32，runner 按既有路径转到模型设备/dtype。记录 seed、生成契约版本、torch 版本和 shape。同一 ref 供该行两档共用，动作比较前统一为 `[16,32]` CPU fp32。每行标签来源固定如下（c 为离线候选，已对实际 hit 核对身份）：

| 实际派发 | y_full 的动作 | y_rem2 的动作 | dispatched_y_key |
|---|---|---|---|
| FULL_HIT | H5 执行 chunk | c 的 intermediate@0.75 在当前观测下续跑 | y_full |
| WARM_START@0.75 | c 的 action_chunk | H5 执行 chunk | y_rem2 |
| MISS（阈值拒绝或 gate-skip） | c 的 action_chunk | c 的 intermediate@0.75 在当前观测下续跑 | null |

- MISS 的执行 teacher chunk不填入任一候选档。其他 WARM start_t 不符合冻结 K=2 臂，直接拒绝。`verify_rows.jsonl` 存两列 y、`s=s_offline`（联合拟合用）、`online_s`（派发超越率用）、`candidate_id`、在线 `winner_id/searched/hit_type/start_t`、`dispatched_y_key`、sample_group、完整行身份与 ref_seed；缺列、非有限 y 或未知来源在调用 LP 前拒绝，不补零或复制另一档。
- **判据**（B3，`exp/rit_loto/data/libero_10/verify.json`）：
  - 估计量：按上述分档样本，在实际派发行的 online_s 上算**决策加权** `E_a = sum(exceed_i)/n_a`，exceed_i 为 `D_{i,a} > q̂_a(online_s_i)`；q 固定为发臂时的 loto_all 曲线。超出原拟合 knots 支持域的行单列 n 和偏差，不利用 predict 的裁剪宣称覆盖；主判据只对域内行成立，域外风险标为未验证。
  - 区间：以冻结的 **50 个 episode 清单**为重采样总体，在每个任务的 5 个 episode 内有放回抽 5 个，B=1000、seed 固定。每个 episode 带该档选中行的 `(exceed_count, row_count)`，无该档样本的 episode 仍以 `(0,0)` 保留；重复抽中保留重数。每次取计数和的比值，不平均 episode 比例；分母为零的重复记无效并报数量。此区间是以固定库/曲线/已选决策为条件的 **95% 百分位 cluster bootstrap 近似诊断**，不声称精确有限样本覆盖或总体等价。
  - 信息门：每档 ≥200 域内行、≥20 个有该档样本的 episode，且含超越和含未超越的 episode 各 ≥5 个；有效 bootstrap ≥900/1000，区间须非退化。任一不满足则判“证据不足”，仍报告点估计与事件数。**零超越、全超越或所有重复比例相同，不得把退化的 [0,0]/[1,1]/单点当作有覆盖保证的区间**；将可用于判读的区间置 null、记录原因。上述最低数量是预设信息筛选，不是 bootstrap 有效性的数学保证。
  - 三分诊断（α=0.05，超越率容差 τ=0.05，正式结果前冻结）：通过信息门后，上界 ≤0.10 → “域内样本满足预设风险容差（近似诊断）”；下界 >0.10 → “域内风险超出预设容差”；其余 → “证据不足”。不能把允许最多 10% 超越写成已验证 5% 校准；域外仍单独标未验证。报点估计、区间、n 行/n episode/n task、事件 episode 数、抽样比例与有效重复数。s 四分位段仅作同口径诊断，分位 ties 合并并报实际段数，不能把逐段检查当整体多重检验通过。
  - 叠图：仅 base 中有候选、两列齐全的行调 `fit_ladders(ks=[2])`；与冻结曲线在共同支持域显示，并报告无候选及域外覆盖。若样本不满足原 knot 占用门，明确 `fit_unavailable`，保留原始偏差诊断，不填图或为出图放松 LP 门槛。叠图描述闭环状态分布下的候选风险变化，不参与上述对固定曲线的判据。渲染脚本 `analysis/plot_verify.py` 不入库。
- **固定报告参数**：正式 report 只接受 `tol=0.05, n_boot=1000, seed=0, min_rows=200, min_episodes=20, min_event_episodes=5, min_valid_fraction=0.9`；这些参数随臂冻结，CLI 不允许事后放宽。
- 产物路径：H5/sidecar/`verify_rows.jsonl`/`verify.json`/臂 record → `exp/rit_loto/data/libero_10/`；臂 yaml → `exp/rit_loto/config/libero_10/`；图与报告 → `exp/rit_loto/analysis/`。verify record 同时汇总全部正式决策的派发计数、按 measured cost 重计价的 IR、50 集成功数，以及抽样/标签/支持域覆盖；计价 IR 不冒充包含新增日志开销的墙钟延迟测量。

### 2.4 任务 4：`exp/rit_loto/noise_floor.py`（岛 venv，GPU）

每 suite 从语料按 seed 抽 500 个不同决策（每任务 50 个）；重建 → stage2 → `run_stage3(stage2, noise=z1)`、`run_stage3(stage2, noise=z2)` → `D = weighted_chunk_deviation(ref1, ref2, w, mask, 5)`。z1/z2 使用固定 root seed、语料行身份及不同标签 `noise_floor_ref1/ref2` 依 §2.3 的规则派生独立噪声种子，记录每行 seed 与噪声生成契约；模板首建与 forward 都在 session 内。另记 `D(ref1, clean_action)` 作在线采集 chunk 对照。输出 `exp/rit_loto/data/<suite>/{noise_floor_rows.jsonl, noise_floor.json}`，分两组报中位数/p90/p95/均值/n，并记录与 parity 一致的完整输入身份。`D(ref1,ref2)` 的中位数是 parity 的唯一标尺；两种 D 不混合。它描述重建观测下 teacher 随机性的量级，不能宣称所有观测/任一 δ 的数学下界。因此任务 4 先运行提供标尺，parity PASS 后才将其作为正式实验解释。

### 2.5 不做的事

不改 `src/`；不改库 pkl；不重写 LP；不改 `GrootRitShadow` / `GrootCacheInterceptor`；不跑 pi0.5；不动 `exp/libero_groot/config/rit/` 既有臂；不画前沿图。

## 3. 文件清单

| 动作 | 文件 | 内容 |
|---|---|---|
| 新增 | `exp/rit_loto/__init__.py` | 包说明 |
| 新增 | `exp/rit_loto/build_loto_table.py` | §2.1（含 parity 抽检与门） |
| 新增 | `exp/rit_loto/fit_loto.py` | §2.2 拟合、对账、bootstrap 带、对比（数值产物） |
| 新增 | `exp/rit_loto/loto_logger.py` | §2.3 服务端逐决策日志 |
| 新增 | `exp/rit_loto/emit_verify_arm.py` | §2.3 臂 + 正式/smoke 两池 + episode_filter |
| 新增 | `exp/rit_loto/verify_closed_loop.py` | §2.3 合并校验、离线标签、判据 |
| 新增 | `exp/rit_loto/noise_floor.py` | §2.4 |
| 新增 | `exp/rit_loto/ops/{run_noise_floor.sh, run_loto_table.sh, launch_verify_server.sh, launch_verify_clients.sh}` | weilandserver / timan107 启动脚本（tmux `srv0` / `lotocli<t>`，tee 落盘） |
| 修改 | `exp/libero_groot/serve_groot_libero.py` | `--loto-log-out` / `--loto-run-tag` / `--loto-frozen-record` + `_build_loto_factory`（与 `--rit-shadow-out` / `--collect-hdf5` 互斥，需 `--concurrent` + `--cache-config`） |
| 新增 | `tests/exp/test_rit_loto.py` | §6 |
| 新增 | `exp/rit_loto/config/libero_10/loto_k2_ir70.yaml` | 臂 yaml（发臂后；目标 IR 70 由 §8-D1 冻结） |
| 新增（data） | `exp/rit_loto/data/<suite>/{loto_table.jsonl, parity_sample.jsonl, parity_gate.json, fits.json, compare.json, bootstrap_band.json, noise_floor_rows.jsonl, noise_floor.json}`；`exp/rit_loto/data/libero_10/{verify_pool/, smoke_pool/, verify_filter.json, smoke_filter.json, verify_pool_manifest.json, verify_logs/, verify_rows.jsonl, verify.json, frozen_run.json, sample_manifest.json, verify_labels.record.json}` | 运行产物（§2） |
| 新增（analysis） | `exp/rit_loto/analysis/{README.md, results.md, data_inventory.md, <suite>/equivalence.png, libero_10/verify_overlay.png}` | 图与报告（数据盘点报告由 §1 转录） |
| 本地不入库 | `exp/rit_loto/analysis/plot_equivalence.py`、`plot_verify.py` | 渲染（owner 规则） |
| 数据追踪（沿用现有 `.gitignore`） | `exp/rit_loto/data/` | 本计划所有运行数值、权重、manifest 与 init 数据均保留在默认忽略的 data/，不修改 `.gitignore`。weilandserver `/data/libero_cache/rit_loto/` 与本地 data/ 各留副本，跨机显式同步并校验 SHA；入库的 Markdown 报告包含主要数字、数据位置/文件 SHA、代码版本及复算命令。臂 YAML 与分析图片按原文件规则处理。 |
| 修改 | `logs/README.md`、`docs/experiments/README.md`（若加 runbook） | 索引同步 |

## 4. 接口

```python
# exp/rit_loto/build_loto_table.py
def load_library(pkl, *, expected_schedule, warm_ts) -> Library        # entries, trajectory lookup, artifact identity
def enabled_fields(cfg) -> tuple[list[str], dict[str, float]]           # (_keys_iter order enabled names, full weight map with 0.0 for disabled)
def build_retrieval(cfg, storage, weights, n_hint) -> Any               # _build_search_strategy(ss_cfg, storage, weights, min_top_k_hint=n_hint)
def query_keys_for(builder, group, enabled: list[str]) -> dict[str, torch.Tensor]
def loto_winner(hits, query_traj: str, in_library: bool, entry_traj: dict[str, str]) -> tuple[str, float, int]  # (id, score, n_self_skipped)
def label_row(runner, templates, task, group, payload, warm_ts, schedule, w, mask, h_exec) -> dict  # y_full, y_rem2, y_rem4
def parity_row(runner, templates, task, group, schedule, w, mask, h_exec) -> dict  # full / warm75 / warm50 maxabs and weighted D
def parity_gate(parity_rows, noise_floor_record, expected_identity, *, ratio=0.1, min_rows=200) -> dict  # identity + per suite/tier stats + PASS/FAIL
def main() -> None

# exp/rit_loto/fit_loto.py
def load_rows(path, *, in_library: bool | None = None) -> list[dict]
def refit_shadow_and_check(rows, cost, arm_record, *, table_sha, template_sha, legacy_alpha=0.05) -> dict  # legacy alpha provenance + knots + finite/inf/null rules
def curve_max_diff(fit_a, fit_b, tier, s_lo, s_hi) -> dict           # exact on endpoints ∪ knots union
def episode_bootstrap_band(rows, cost, warm_ts, k, alpha, grid, *, n_boot=200, seed=0) -> dict  # pointwise reference variability, effective counts and failures
def score_marginal(s_a, s_b) -> dict                                  # quantiles, mass>=0.99, KS
def main() -> None

# exp/rit_loto/loto_logger.py
class _RecordingRunner:        # __getattr__ 转发；run_stage1 记录输出
class _RecordingOrchestrator:  # __getattr__ 转发；check / broadcast_action 记录
class GrootLotoLogger:
    def __init__(self, policy, runner, *, orchestrator, timer, out_dir: str, experiment: str, run_tag: str) -> None
    def on_task_begin(self) / on_task_end(self) / on_episode_start(...) / on_episode_end(success) -> None
    def get_action(self, observations: dict) -> dict               # 返回 interceptor 的原返回值（含 __hit_meta__）

# exp/rit_loto/emit_verify_arm.py
def emit_arm(suite, fit_source, fits_path, cost, target_ir, template_path, config_out, *, pool_manifest, k=2, original_arm_record=None) -> dict
def load_frozen_record(path) -> tuple[dict, str]                         # validated record and byte digest
def sample_pools(suite, apool_dir, exclude_manifest, per_task_verify, per_task_smoke, seed, out_dir) -> dict  # pools + episode_filter jsons + manifest

# exp/rit_loto/verify_closed_loop.py
def collect_decisions(log_root, manifest, record, record_sha, run_tag="verify") -> tuple[list[dict], dict]   # H5 步组 × sidecar 对齐 + 身份校验（SystemExit）
def sample_decisions(rows, n, seed) -> tuple[list[dict], dict]       # uniform base + warm_extra, manifest with sampling fractions
def offline_candidate(strategy, builder, group, enabled, task_key, decision_id, traj_of) -> dict  # candidate_id / s_offline, independent of online gate/verdict
def k2_labels(runner, templates, task, group, row, payload, executed, w, mask, h_exec, schedule, *, ref_seed) -> dict  # y_full + y_rem2
def exceedance_report(rows, episode_manifest, fit_k2, tiers, *, alpha=0.05, tol=0.05, n_boot=1000, seed=0, min_rows=200, min_episodes=20) -> dict  # information/degeneracy gates, in/out-of-support counts
def main() -> None

# exp/rit_loto/noise_floor.py
def main() -> None
```

## 5. 集成点与运行拓扑

- 数据流：W13 H5（weilandserver `/archive`）→ noise_floor + parity 门 → `build_loto_table.py`（weilandserver 4090，两 suite 各一进程，按显存实测调度）→ JSONL 与 record → tether pull 本地 → `fit_loto.py`（本地主 venv）→ fits/compare/图 → `emit_verify_arm.py`（本地）→ tether push 臂 yaml + 池 + filter/manifest 到 weilandserver / timan107 → server（weilandserver `srv0`）+ clients（timan107）→ H5 + sidecar → `verify_closed_loop.py`（weilandserver）→ pull → 判据与图。所有阶段按完整 SHA 对账，git commit 本身不代表尚未提交的代码版本。
- 代码同步：新文件 `tether push` 到 `/data/openpi_lg`（weilandserver）与 `/scratch/zixuans8/openpi_lg`（timan107，经 `/tmp` 中转），逐文件 `cat | sha256sum` 对账；git 只在里程碑收口且 owner 指示时动。
- 实施顺序：G1 放行 → Code → G2 → Verify → 任务 4（两 suite）→ 任务 1 `--parity-only`（门）→ 任务 1 全量 → 任务 2 → 按冻结的 D1–D5 发臂 + 两池 → 任务 3 smoke → 任务 3 verify → 离线处理。实验 smoke/正式运行保持不同 run_tag；修复 smoke 问题后重新核代码 SHA；任何参与数值链的源码变化都会使旧产物失效，须在最终版本上从 noise-floor/parity 开始重建该链，正式运行前冻结最终版本。
- 估算：原 GPU 粗估 libero_10 ≈35 min、spatial ≈15 min，任务 4 两 suite ≈3 min，50 集闭环 ≈30–40 min；均须按实际吞吐修正。新增 shadow bootstrap 共 2 suite×3 k×200 次 LP，先计时少量重复估算 CPU 总时长，再以有界进程数运行；闭环离线成本按 base + warm_extra 的实际行数及两档标签计算，旧“5 min”不作为承诺。
- 监控：任务 1/4 是有限长离线作业，tmux + 行数探针即可；任务 3 起 server/client 期间挂一个 Monitor 条件触发（server 掉线 / 集数停滞）+ 完成即撤；不起 `run_in_background`。

## 6. 测试策略（`tests/exp/test_rit_loto.py`，主 venv，无 GPU；GPU 侧自证见 §2.1 第 5–6 步）

1. `loto_winner`：合成命中列表，库内查询永不取自身轨迹、取的是他轨迹中分数最高者、`n_self_skipped` 正确；库外查询取首位；全是自身轨迹时 raise。
1b. **真缩库对照**：用 `InMemoryBackend` 装 3 条合成轨迹 × 2 任务的小库，(i) 全库检索 + 事后排除，(ii) 去掉该轨迹重建的库检索，两者 winner 与 score 逐值相等。
1c. **enabled 契约**：用真实 `template.yaml` 的 keys 段构造 cfg，`enabled_fields` 返回 `[vision_0, vision_1, robot_state]` 与含 `prompt_emb: 0.0, vision_2: 0.0` 的权重映射；把 `prompt_emb` 向量随机置换后 `strategy.search` 的 winner/score 不变；把 `prompt_emb` 留在 query_keys 且不传其权重时 score 改变（复现 B1 的失效路径，作为反例锁定）。
2. `label_row` 用桩 runner：三档偏差各自调用 `weighted_chunk_deviation`，`w` 为同一张量、`h_exec=5`；`y_full` 不依赖 runner。
2b. `parity_gate`：构造两活动维，parity 误差仅在第一维、noise-floor 误差仅在第二维；两者都用同一 W，第一维权重 ×100 后 parity_D/地板比及门结论按解析值变化而 maxabs 不变。覆盖三种重放、<200 行、每任务不足20行、任务缺失、零地板、非有限值、身份不匹配、缺文件及未知门状态；不把“任意改变 W 必改变比例”当普遍规律。
2c. **模板首建 session 契约**：桩 runner 的 `run_stage1` 在 session 外调用即 raise；`build_loto_table` / `verify_closed_loop` / `noise_floor` 的逐行函数在首次遇到新任务时不 raise（模板首建被包在 session 内），检索函数在 session 内被调用则测试失败（用桩 storage 检查 autocast 标志）。
3. `curve_max_diff` / `score_marginal`：两份相同行 → Δq 全零、KS 0；构造在某 knot 段平移的行 → max|Δq| 落在该段且等于解析值（分段线性差的极值在 knots 并集上）。
4. `refit_shadow_and_check`：合成小表 + 现有 LP 生成的含 null cuts 记录通过；覆盖无 alpha 的 legacy 记录（明确记录来源）、带 alpha 的匹配/不匹配、有限 cut 篡改、null 改有限、NaN/-inf 与 table/template SHA 错误。用新 fits 序列化往返后核对 predict/cuts 一致。
4b. `episode_bootstrap_band`：检查按任务的 episode 重数保留、给定 seed 可复现；人为制造支持域缩窄和 knot 样本不足，验证点级有效数、失败记录与留空行为，不能用端点裁剪填带。不要求任意一次减少 episode 数都使带宽单调变大。
5. `GrootLotoLogger`：FULL/WARM/MISS 各验证一次 H5 clean_action 等于 broadcast、sidecar 元数据与原返回值一致、所有额外 attrs 确实落盘。再验证连续两步和异常后调用不复用旧捕获、两连接的 episode/decision 计数隔离、钩子仅转发一次且结束后写盘。
6. `sample_pools`：正式 5 + smoke 1 与排除清单零交集、互斥、seed 可复现；`episode_filter` JSON 的 `subset_init_state_idx` 等于排序后位置、`orig_init_state_idx` 为官方 index（含非连续 index 用例）。
6b. `collect_decisions`：合成 verify + smoke 两目录 → 只纳入 verify；重复 episode / 缺一个 / 多一个不在清单 / H5 attr 与 sidecar 不一致 → 各自 SystemExit 并列差集。
7. `k2_labels`：以实际 `ThresholdJudge` 构造“有搜索候选但 MISS 的 winner_id=None”反例，以及 gate-skip；两者离线检索后均可有两列标签。按 §2.3 表为执行 chunk / 候选 full / 候选 warm 设置不同的可解析值，核对每档数值和 runner 调用，不能仅检查键存在。缺列、非有限值、未知 warm_t、hit 身份/分数不匹配明确拒绝；真实无候选只记覆盖；种子在重排/分片后不变。
7b. 抽样：固定 base 不因 WARM 补样改变，补样去重且只入 warm_extra；联合 LP 只收到 base。小于2000行时 base 为全集；分档风险分母和抽样比例可按合成计数手算。
7c. `exceedance_report`：用 episode manifest 包含零贡献 episode，核对任务内抽样与比值聚合。将每条 episode 内全部行等倍复制，在同一重采样序列下点估计及 bootstrap 比例不变；不要求 cluster 区间在所有数据上都宽于 Wilson。零/全超越、退化区间、有效重复不足、事件 episode 不足、`2/20` 小样本均为“证据不足”；另构造通过信息门后上界≤0.10和下界>0.10的两种可手算情形。域外行不进入已验证覆盖声明，分位 ties 不生成空段；闭环 knot 不可用须有明确产物状态。
8. `serve_groot_libero` argparse（照 `tests/libero_groot/test_rit_shadow_factory.py` 范式）：`--loto-log-out` 与 `--collect-hdf5` / `--rit-shadow-out` / 动态 bundle 模式互斥、缺 `--cache-config` / `--concurrent` / `--loto-run-tag` 及非本计划 CP1 配方报错。

## 7. 风险登记

| 风险 | 影响 | 处置 |
|---|---|---|
| 重建 stage-1 的动作头续跑与在线路径有 bf16 级差异 | D 偏移 | §2.1 第 5 步：与 D 同尺度的 `parity_D` 门（p90 ≤ 0.1 × 地板中位数，分 suite / full / warm，≥200 行）先于拟合；FAIL 停下报告 |
| 库内 LOTO 查询删去每任务5条中的1条轨迹，分数分布可变；D 变大的方向并非保证 | 全部合池曲线与完整库部署存在分布差 | §2.2 ②③ 分层单列；本轮主曲线冻结为①，③保留对照，不事后挑选效果更好的来源 |
| 库外成功轨迹占多数（400/450、386/450） | 查询集偏向成功 | 分层 `episode_success` 报行数与 D 均值；不做加权 |
| 闭环 WARM 行少、轨迹内相关、超越事件稀少 | 分档功效不足 / 百分位区间退化 | uniform base 与 warm_extra 分开；50-episode 分层 cluster bootstrap；按 §2.3 信息门/退化门处理，保留“证据不足”，不能以200行冒充200个独立样本 |
| searched MISS 不携带 winner、gate-skip 不携带分数 | 若据在线 null 丢行，闭环拟合丢掉整个 MISS 状态区域 | 对所有选中观测完整库离线检索，在线/离线字段分开；派发超越率只比较实际缓存档 |
| 原 shadow bootstrap 支持域缩窄或 LP 无法满足 knot 占用门 | 端点裁剪造假覆盖、重抽偏向成功拟合 | 点级有效数与失败原因落盘；逐点带仅作参考估计波动显示 |
| 任务 3 用 LOTO 臂而非原臂 | 与既有前沿不可直接比 | record 同时记原臂 δ/cuts；如 owner 要原臂，改一个 yaml 路径即可 |
| 子池 init 身份 / smoke 与正式混合 | 闭环行归属错误 | `--episode-filter` 带官方 index；两池两 run_tag 两目录；合并按清单强校验（§2.3） |
| 禁用字段混入打分 | s 口径漂移，自证失败 | enabled 过滤 + 全字段零权重映射 + 置换自证 + §6-1c 反例测试 |
| timan107 EGL 踩踏（换任务时并发建销上下文） | client 卡死 | 一任务一进程避免换任务；每卡 ≤2 进程 |
| `/tmp` 类路径易被清 | 产物丢失 | 全部产物落 `/data/libero_cache/rit_loto/<suite>/` 与仓内 `exp/rit_loto/data/`（gitignored），当日 pull 回本地 |

## 8. 冻结的实施选择（owner 授权本轮审查者直接修订）

- **D1** 任务 3 臂：K=2、目标 IR 70、cuts 与 gate θ 全部来自 LOTO 拟合；原 `l10_rit_k2_ir70` 仅作 record 对照。IR 70 是无 gate 成本寻址目标，报告另按实际派发份额与 measured cost 计算实测 IR，不要求 gate 后恰好70。
- **D2** 任务 3 init 池：A 池排除 shadow cohort 后每任务正式5个 + smoke1个，两池互斥，seed 20260913；50集正式结果独立于10集smoke。
- **D3** 实验目录名 `exp/rit_loto/`。
- **D4** 任务 2 的"原 shadow 曲线"= 同代码在 `shadow_rows.jsonl` 上重拟合并与 `arm_record` 逐值对账（q 值未落盘，无法直接读）。
- **D5** 主曲线采用既有建议①，即 `fit_source=loto_all`，写入臂/verify record/图注；③只作预先约定的分层对照。此项是本轮受托实施选择，不再留下发臂前必须补答的未定项。
- **D6** 采用执行方已列出的免改白名单方案：所有运行数值/manifest/init 按 data/ 默认忽略、显式同步，报告转录主要数字并记 SHA。`.gitignore` 不列入本计划修改范围。
- **D7** α=0.05、闭环超越率容差0.05、抽样/信息门和 bootstrap 参数按 §2 冻结；“满足≤0.10容差的近似诊断”不等于“精确5%校准”，不足时报告证据不足而不事后放宽。

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-09-14 11:03 CDT

Review type: G2; Authority: Review; Level: L2; target: 本计划的 polished 正文、`exp/rit_loto/` 全部交付文件、`exp/libero_groot/serve_groot_libero.py` 的新增工厂/CLI、`tests/exp/test_rit_loto.py` 与对应索引；checklist: Working Agreement §2.6 / review_authority §4。基线为 `668e91a`，本轮按 Round 1 读取工作树最终实现。此前 owner 授权的 G1 正文修订已被执行方采纳；本轮审查的是执行方编写的代码，Reviewer 未修改被审生产代码。

**Checklist**

| 项目 | 裁决与依据 |
|---|---|
| 与批准计划一致 | **FAIL**。启用字段过滤、整轨迹排除、session 范围、两档动作来源、base/WARM 补样分离、LP 复用均已落地；但冻结身份链、样本完整性、payload schedule、事件信息门与 alpha 约束仍有 B1–B7。 |
| 测试覆盖及通过 | **FAIL**。交付测试及首批相邻测试 53 passed；独立合同反例 14 failed，均复现下列实际漏检，非测试环境失败。现有测试主要覆盖 helper 的正常路径，未覆盖发臂→采样→标签→报告的身份和完整性拒绝路径。 |
| 文档与索引 | **PARTIAL**。计划和 `logs/README.md` 已同步 Code 完成/G2 待审，数据盘点与 analysis 索引齐全；但公共接口 docstring 有 V1，噪声地板说明有 N1，顶部引用的“G2 交付说明”未在本次目标文件中找到，见 N3。依赖正式实验才生成的 YAML/图/结果不作为本轮缺交付。 |
| 无回归 | **PARTIAL**。`src/` 零修改，服务端仍组合真实 interceptor，并沿用连接隔离和推理锁；两套真实 shadow 的 48 臂/96 cuts 各自对账通过。扩大相邻测试 432 passed / 5 skipped / 2 failed；两失败来自 HEAD 已有 gr00t stub 污染，单独运行对应 16 条全部通过，见验证记录。新流程仍存在可产生错误风险结论的 B1/B2，不能据此放行。 |

**Blocking findings**

- [Blocking] [Concern] **B1 / P1 — 发臂、正式日志、离线标签与报告没有强制绑定同一份冻结输入。** `emit_verify_arm.py:130–159` 只核 suite，不核传入 template 与 `fits.identity.template_sha256`；`serve_groot_libero.py:319–323` 只写 checkpoint 路径，池 SHA 可以为空；`verify_closed_loop.py:127/510–513` 的冻结 attrs 检查可整段省略；`cmd_label:281–346` 不把库/ckpt/W/H_exec/schedule 与已收集 H5、发臂 record 对账；`cmd_report:526–550` 不读取 arm/label record，连 fits 的 suite 也不核。独立实测：不同 template 身份仍成功发臂；50 集全部缺少冻结 attrs 仍 collect 成功；把 spatial 曲线传给 `--suite libero_10`，报告仍写 `suite=libero_10` 并给 FULL 档 `within_preset_tolerance_in_support`（CI `[0.005,0.016]`）。— reasoning: 这会把不同模型、检索分数尺度或曲线的结果合成“验证通过”，违反 §2.1/§2.3/§5 的完整 SHA 对账要求。整改：生成一份正式运行冻结记录，将 suite、fit_source/K/alpha、fits SHA、template/arm/library、checkpoint 内容身份、W/mask/H_exec/schedule、pool/run_tag 和相关代码版本贯穿各阶段；必需字段缺失或不一致即拒绝，不能靠可选 `--frozen-attrs` 补救。报告必须核对实际发臂的 fits SHA，并消费 label record；新增 label/fit/logger/emitter 自身代码也须纳入对应版本记录。

- [Blocking] [Concern] **B2 / P1 — 正式标签可截断、缺列或非有限，然后静默缩样本并生成结论。** `verify_closed_loop.py:291–294` 的 `label --limit` 仍写正式 `verify_rows.jsonl`，没有 smoke/partial 标志；`cmd_report:533–543` 不读取采样清单或核对主键集合，无法发现漏标/重复/混入其他 run；`joint_refit:474–475` 将已有 candidate 的缺失/NaN 标签过滤掉；`exceedance_report:376–379` 同样静默过滤派发行；`check_online_consistency:211–220` 的差值判断允许 NaN 绕过。独立实测：应标 1,000 行只提供 500 行，仍产出 FULL 档满足容差（CI `[0.01,0.032]`）；候选 `y_rem2=NaN` 未拒绝而被 LP 前丢弃；FULL 的在线分数为 NaN 也通过一致性检查。— reasoning: §2.3 要求 base 均匀样本、全部 WARM 和两列标签齐全，允许缺标签的唯一明确例外是真正无离线候选。整改：在 LP/超越率之前按采样主键、sample_group 和总数做完整 join，检查每行来源/两列有限值及在线字段；显式隔离调试截断产物，正式报告拒绝 partial、重复、未知来源和有候选却缺标签的行，并在报告中携带采样概率与覆盖计数。

- [Blocking] [Concern] **B3 / P1 — 旧 parity 门能继续认证变化后的代码或 checkpoint。** `build_loto_table.py:95–98/498–500/569` 的比较键没有 `code_sha256`，虽写入 identity 但不比较；`checkpoint_identity:209–222` 对模型只取文件名/大小，只有 `config.json` 做内容 hash。独立实测：将 forward 代码摘要从旧值改成新值，`require_pass_gate` 仍返回 PASS；同名等长权重从 `weights-A` 换成 `weights-B`，checkpoint 摘要完全不变。— reasoning: parity 的用途正是证明本次实际 forward 重建保真；修过 forward 或换过模型后的旧门不能沿用。整改：校验所需代码文件的实际 SHA（缺失不得双方 None/unknown 等价通过），模型身份使用真实权重内容摘要或可验证的内容清单；noise-floor→parity→table→fit 各入口验证相应绑定。计算内容摘要可缓存，但不能以文件大小代替内容身份。

- [Blocking] [Concern] **B4 / P1 — 库加载绕过每条 payload 的 schedule 身份。** `build_loto_table.py:280–301` 只验 artifact schedule 和 `denoising_num_steps`，不验 `payload.schedule_id`，随后在 `label_row` 给这些快照传入 artifact/live schedule。独立实测：artifact 为 `groot_n15_k8_v1`、payload 为 `pi05_v1` 仍加载成功；chunk/snapshot 同为错误的 `[1,32]` 也通过形状检查。— reasoning: 形状相同和步数相同不能证明快照属于同一去噪循环，当前会替错误来源补身份，违反批准计划 §2.1 末段以及 cache_system §5.17 的逐层 schedule 合同。整改：复用 payload 的现有身份验证，严格验证 payload schedule、合法快照、固定 `[16,32]` 动作形状/有限值，并在闭环 label 入口核 H5 schedule；错误应在 GPU 标注之前按 entry/episode 定位拒绝。

- [Blocking] [Concern] **B5 / P2 — collect 的 H5/sidecar 对齐只比较数量，没有比较真实决策坐标。** `verify_closed_loop.py:130–132` 只检查 step group 数等于 num_steps；`144–152` 不检查 `control_step_idx`、task 或 sidecar 的完成状态与 H5 一致，并用 H5 task/success 覆盖 sidecar。独立实测：一集把 `step_0000` 改为 `step_0001` 而 num_steps 保持 1，collect 仍返回不存在的 `step_0000`；sidecar 决策 0 的 control_step 写成 99 也被接受。— reasoning: §2.3 要求 H5 步组和 sidecar 一一对应及连续决策号，合并阶段就应拒绝坏坐标，不能等它恰好被抽中才在 GPU 阶段报 KeyError。整改：核对完整组名集合/decision_id、`control_step_idx=H_exec*decision_id`、task/完成元数据、必需 H5 字段，并验证正式 manifest 为冻结的 10×5 集；逐项输出差集。

- [Blocking] [Concern] **B6 / P2 — 信息门把“包含未超越样本”错写成“整集零超越”。** `verify_closed_loop.py:405–407/437–440` 用 `exceed_count == 0` 统计未超越 episode；批准的 §2.3 要求的是含未超越的 episode，即 `exceed_count < row_count`，混合 episode 可以同时贡献两个信息计数。独立实测：50 集每集 20 行，每集同时包含两类结果，点估计 0.24、1,000 次有效 bootstrap、CI `[0.23,0.25]` 非退化，代码却因“episodes without exceedance 0 < 5”将 interval 置 null 并报证据不足。— reasoning: 明确超出 10% 容差的情形被错误的信息门隐藏；现有高风险测试特意制造 10 个完全零事件 episode，未覆盖这种常见混合集。整改：按包含非事件的定义计数，保留全零/全一和退化门，新增全部 episode 均混合且区间下界 >0.10 的反例。

- [Blocking] [Concern] **B7 / P2 — 原 shadow 对账接受非冻结 alpha，导致不同分位数曲线被直接比较。** `fit_loto.py:167–174` 从 arm_record 直接接受任何 alpha，也从记录推导 ks 而不核固定集合。独立实测：真实 LP 在 alpha=0.20 生成自洽 knots/cuts 后，`refit_shadow_and_check` 全部通过；与此同时主 LOTO `fit_source` 默认仍是 0.05。— reasoning: §2.2 明确规定未来带 alpha 的记录必须与 0.05 一致；曲线能复现只证明记录自洽，不能证明比较的是同一目标分位数。整改：入口强校 alpha=0.05、ks={1,2,3} 及 tier/y_key 契约，保留现有 legacy_reconstruction 路径，增加“非 0.05 但完全自洽”的拒绝测试。

**Constitutional Violation**

- [Blocking] [Concern] **V1 — Working Agreement §3.2 的 public docstring 要求未满足。** 对新包顶层公共函数做 AST 检查，至少 34 个没有 docstring，例如 `build_loto_table.py:498 identity_mismatches`、`verify_closed_loop.py:267 cmd_label`、`:526 cmd_report`、`fit_loto.py:61 load_rows`；这尚未计算类的公开方法。— reasoning: 上位约定明确要求 public classes/functions 有 docstring。补充接口用途、输入/输出与拒绝条件；不要仅为躲规则改成 private 名称。B4 同时违反 WA §8 注册的 cache_system §5.17 身份守卫，详见 B4。

**Non-blocking findings**

- [Non-blocking] [Concern] **N1 — 噪声地板文档重新引入已否决的数学下界说法。** `noise_floor.py:7–9` 声称中位地板 bounds delta from below、其下规则拒绝复用；§2.4 明确仅是随机性尺度，不能推出每个候选/观测都如此。将说明改回批准口径，避免后续结果报告沿用此说法。
- [Non-blocking] [Suggestion] **N2 — 完成捕获后释放引用，并显式拥有 CPU 数据。** `loto_logger.py:205–247` 只在下一次调用开始 reset，返回后仍持有 stage1/check/broadcast，结束 episode 也未释放；`_to_np` 对本来连续的 CPU fp32 张量不会拷贝 numpy 所有权。按 §2.3 在成功/异常的 finally 清理捕获，需留存的 CPU 数组显式 copy；验证写盘前修改原始 buffer 不改变已捕获数据。当前没有据此声称真实线上动作已经被污染。
- [Non-blocking] [Concern] **N3 — 将“29 测试 + 岛上冒烟通过”的证据入口补实。** 本计划顶部引用“G2 交付说明”，但当前正文后直接是空 Review Log，包内两个 Markdown 也未记录该说明。此次 Reviewer 已独立验证 CPU 测试和真实旧 shadow 对账；未据顶部一句话认定当前代码的远端 GPU parity/日志 smoke 通过。执行方应补充 advisory 命令、代码 SHA、日志路径、实测行数/关键数值；实验正式运行仍按批准的 G2→Verify→noise-floor→parity 顺序。
- [Non-blocking] [Suggestion] **N4 — 新测试不要把无 `__spec__` 的 gr00t stub 永久留在 sys.modules。** `tests/exp/test_rit_loto.py:832–835` 的 setdefault 没有 teardown，沿用了 `tests/libero_groot/test_rit_shadow_factory.py:28` 的既有污染模式。改成 pytest monkeypatch 生命周期内替换或去掉不必要的 stub，避免组合测试中的 find_spec 崩溃。

**独立验证记录**

- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q tests/exp/test_rit_loto.py tests/libero_groot/test_rit_shadow_factory.py tests/cache/groot/test_import_isolation.py` → **53 passed**。
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q tests/cache/groot tests/libero_groot tests/robocasa365/test_groot_cache_collector.py tests/robocasa365/test_groot_concurrent_serving.py` → **432 passed / 5 skipped / 2 failed**。两个失败为 `test_frozen_commands_pass_the_new_guards` 的 `gr00t.__spec__ is None`；该运行未包含新增 LOTO 测试，HEAD 已存在的 `test_rit_shadow_factory.py:28` 在 collection 时写入此 stub，故不把这两条归因为本轮服务端改动。独立运行 `tests/robocasa365/test_groot_concurrent_serving.py` → **16 passed**。GPU/compiled 检查的 skip 不算真实 GPU 验证通过。
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q tests/review_tests/test_rit_loto_g2_contracts.py --basetemp=tests/review_tests/tmp_rit_loto_g2 --tb=short` → **14 failed**，逐一对应 B1–B7；失败均为生产函数接受本应拒绝的输入，或 B6 的实际判读错误。测试与临时数据只位于 ignored `tests/review_tests/`，不进入共享 index。
- 用当前 `refit_shadow_and_check`、measured cost 和本地真实 `shadow_rows.jsonl`/template/arm_record 运行：spatial **3,432 行、48 臂、96 cuts**；libero_10 **8,383 行、48 臂、96 cuts**；两者 ks=1/2/3、alpha=0.05 legacy_reconstruction、knots/cuts 对账全部通过。
- 将现有 transitive import guard 应用于新增 build/noise-floor/logger/verify 四个岛入口：无 jax/Pi0.5 禁用导入；四个 ops 脚本 `bash -n` 通过；目标已有 tracked 差分 `git diff --check` 通过。本轮未启动远端实验、未改生产实现、未 commit/push。

**裁决：NEEDS REVISION — B1–B7 与 V1 修复并复审前，G2 不放行。** 本轮审查的代码、测试、计划 Review Log 和已更新的任务索引按 review_authority §5 暂存为下轮差分基线；无关工作区及既有暂存条目保持原状。

### G2 Round 1 — Executor — 2026-09-14

逐条回应。全部采纳；每条列出修改位置与对应新增测试（`tests/exp/test_rit_loto.py`）。

- **B1** — Accepted — 新增**冻结运行记录** `frozen_run.json`（`emit_verify_arm.emit_arm` 产出，`load_frozen_record` 校验协议/必需键）：含 suite、fit_source、K、alpha（强制 0.05）、`fits_sha256`、fits 携带的表身份块（library / checkpoint 内容 / template / W / H_exec / schedule / 表运行代码 sha）、臂 yaml sha、池清单 sha、run_tags、发臂时代码 sha。`emit_arm` 现在核传入 template 的 sha 等于 `fits.identity.template_sha256`，核池清单 suite。`serve_groot_libero` 新增必填 `--loto-frozen-record`（替代 `--loto-pool-manifest`），启动前核臂 yaml sha、库文件 sha、checkpoint 内容身份、`--rit-h-exec` 与冻结值一致，建栈时核 live schedule，并把冻结记录 sha 与各 sha 打进每集 attrs。`collect` 必填 `--frozen-record`，池清单 sha、每集六个 `loto_*` attrs、schedule、h_exec 全部对账；`label` 必填冻结记录 + 采样清单，`bind_label_inputs` 核库/ckpt/template/W/H_exec/schedule；`report` 必填冻结记录 + label record + 采样清单，核 suite、fits sha、label record 非 partial、rows sha、alpha，且 fit_source/K 只从记录读。`CODE_FILES` 纳入 fit_loto / loto_logger / emit_verify_arm / verify_closed_loop。测试：`test_emit_arm_writes_a_frozen_record_bound_to_fits_template_and_pool`、`test_loto_factory_binds_the_frozen_record_before_serving`、`test_collect_decisions_admits_only_the_frozen_run_and_names_every_discrepancy`、`test_report_refuses_partial_labels_and_foreign_curves`（含把 spatial 曲线喂给 libero_10 报告被拒）。
- **B2** — Accepted — `label --limit` 产物改名 `verify_rows.partial.jsonl` + `verify_labels.partial.record.json` 并标 `partial: true`，`report` 拒绝 partial；新增 `validate_labelled_rows`：按采样清单主键 `(run_tag, connection_id, episode_id, decision_id)` 做完整 join（缺一、重复、多出、未知 sample_group、有候选却缺/非有限标签、dispatched_y_key 与 hit_type 不符、派发缓存行无有限在线分数均 SystemExit），只有 `label_reason == no_offline_candidate` 的 MISS 行允许无标签；`exceedance_report` / `joint_refit` 对有候选缺标签的行报错而非过滤；`check_online_consistency` 用 `_finite` 判断，NaN 在线分数被拒。报告携带采样概率与覆盖计数（`sampling` 块）。测试：`test_validate_labelled_rows_requires_the_whole_sample_with_finite_labels`、`test_online_consistency_checks`（NaN 用例）、`test_joint_refit_uses_base_rows_only_and_refuses_missing_labels`、`test_report_refuses_partial_labels_and_foreign_curves`。
- **B3** — Accepted — `IDENTITY_KEYS` 加入 `code_sha256`（dict 逐项比较），`identity_mismatches` 对任一侧缺键即判不一致；`code_sha256()` 对缺文件 SystemExit，不再返回 `missing`；`checkpoint_identity` 改为对目录内全部文件字节做 sha256（路径前缀 + 内容），以 `(path, size, mtime_ns)` 清单键缓存在 `~/.cache/rit_loto/ckpt_identity/`，清单任一项变化即重算，大小从不代替内容。测试：`test_parity_gate_scales_with_w_and_refuses_thin_or_mismatched_samples`（forward 文件 sha 改变即 FAIL）、`test_require_pass_gate_refuses_missing_failed_foreign_or_stale_code_gates`（旧代码 sha / 缺 code_sha256 均拒）、`test_checkpoint_identity_hashes_content_and_caches_by_listing`（同名同长不同字节 → 不同摘要；缓存命中；mtime 变化重算）、`test_code_sha256_refuses_a_missing_file`。
- **B4** — Accepted — `load_library` 对每条 payload 调用现成的 `CachePayload.validate_for_warm_start(expected_schedule, t)`（核 `schedule_id`、`denoising_num_steps`、快照存在/形状/dtype），并额外要求 chunk 与全部快照为有限的 `[16, 32]`；错误按 entry 定位、在任何 GPU 标注之前 SystemExit。W13 两库实测 payload 全部带 `groot_n15_k8_v1`。测试：`test_load_library_validates_every_payloads_schedule_and_shape`（artifact k8 / payload pi05_v1 拒；`[1,32]` 拒）。
- **B5** — Accepted — `collect_decisions` 核 step 组名集合恰为 `step_0000..step_{n-1}`、每组含 vision_0/1、prompt_emb、robot_state、clean_action，sidecar `control_step_idx == h_exec × decision_id`（h_exec 取 H5 attr 并与冻结值对账）、sidecar task 与 `episode_success` 等于文件值、hit_type 合法、H5 schedule 等于冻结值；池清单必须恰为冻结的 `n_tasks × per_task` 集且 sha 等于冻结值；差集逐项打印。测试：`test_collect_decisions_admits_only_the_frozen_run_and_names_every_discrepancy`（组名错位、control_step 错、task 不符、缺字段、h_exec 不符、池不完整各一例）。
- **B6** — Accepted — 信息门改为"含非超越样本的 episode 数"（`exceed_count < row_count`），混合 episode 同时计入两个计数；全零/全一与退化区间仍拒。测试：`test_exceedance_report_gates_readings_and_is_invariant_to_row_duplication` 新增 50 集 × 20 行全部混合、率 0.24 的用例，判读为"域内风险超出预设容差"。
- **B7** — Accepted — `refit_shadow_and_check` 强制记录 alpha（若有）等于 0.05、K 集合恰为 {1,2,3}、每个 K 的 tier 名等于 `rc.ladder` 且 y_key 等于冻结序列；`emit_arm` 拒绝 alpha ≠ 0.05 的 fit source。测试：alpha=0.20 自洽记录被拒、K={1,2} 被拒、alpha=0.20 的 fits 不能发臂。
- **V1** — Accepted — AST 复查后为全部公共函数、类与公共方法补 docstring（含 `identity_mismatches`、`cmd_*`、`load_rows`、logger 的钩子与属性等 42 处），未改私有名规避。
- **N1** — Accepted — `noise_floor.py` 模块说明改回批准口径：仅描述本样本重建观测下教师随机性的量级，不是任一观测/候选的 δ 数学下界。
- **N2** — Accepted — `GrootLotoLogger.get_action` 在 `try/finally` 内清空 runner/orchestrator 捕获（成功与异常路径都不残留），切片与执行 chunk 用 `np.array(..., copy=True)` 显式拥有；测试新增"调用后捕获为 None"与"事后改写 broadcast 张量不影响已落盘 clean_action"。
- **N3** — Accepted — 新增 `exp/rit_loto/analysis/smoke_evidence.md`：本地 advisory 命令与结果、当前代码文件 sha、岛上冒烟链命令/日志路径/实测行数与关键数值（parity_D、noise floor、orchestrator 自证计数、吞吐）；计划头部的"G2 交付说明"改指向该文件。
- **N4** — Accepted — 测试改用 `monkeypatch.setitem(sys.modules, "gr00t", stub)` 且 stub 带 `__spec__`，仅在未存在时注入，随用例生命周期撤销。

复核后的本地证据：`tests/exp/test_rit_loto.py` 35 passed；`tests/libero_groot` 203 passed / 5 skipped；ruff 全部通过。岛上冒烟链按修订后代码重跑，结果见 `smoke_evidence.md`。修订后的代码重新进入 G2。


### G2 Round 2 — Reviewer — APPROVED — 2026-09-14 15:51 CDT

Review type: G2; Authority: Review + owner override; Level: L2; target: 本计划、`exp/rit_loto/`、LOTO 服务工厂与对应测试/索引；基线 HEAD `668e91a`，本轮执行者修订先与 R1 index 比较，再于修复前暂存为本轮原始快照。

**授权与结论边界**

Owner Ziyang Lin 本轮明确要求“审查后直接修改到可以通过；原本暂存；自己的修改留在暂存区外；无需再次询问”。按 Working Agreement 开篇的 owner override，本轮覆盖 review_authority 对生产代码修改、自修复后验收及最终暂存方式的限制。以下 APPROVED 是该授权下的复审、直接修复与验收结论，不冒充对本轮自写补丁的独立第三方审查；没有提交、推送或启动正式实验。保留此前所有 Reviewer / Executor 条目。无未解决的 Constitutional Violation。

**复审发现与直接修复**

- [Blocking → Resolved] [Concern] **B1/B2/B3 的残留准入缺口**：执行者已增加冻结记录和各类 SHA，但 fit/emit/record loader 未将记录的代码与实际源码比较；label 原先只核样本主键，report 未核抽样分组/episode 元数据、原始 population 文件、measured cost 和噪声契约。补 `require_code_identity`，扩大到 LP/发臂/生产执行链；冻结 eager stage-1、完整 50/10 池和正式统计参数；collect 记录 H5 与三个数据文件 SHA，label 在模型加载前核整份样本和 H5，report 重放抽样并逐条 join 在线元数据、分组、候选、两档标签和种子。缺列、NaN、partial、换曲线、改门槛、移换 task/init 现在均拒绝。
- [Blocking → Resolved] [Concern] **B3 的 checkpoint 缓存反例**：独立重放“等长度替换权重并恢复 mtime”，仅靠 size/mtime/ctime/inode 的缓存仍可能返回旧摘要。现改为每次准入重算文件内容摘要；缓存记录只在实际字节摘要相等后标 cache_hit，内容 framing 同时编码相对路径及长度。反例复验通过；代价是每次启动多一次 checkpoint 顺序读取。
- [Blocking → Resolved] [Concern] **全量标定和正式 parity 的样本准入**：task shard 原先可以被当作非 smoke 全表拟合，20 行 noise-floor 冒烟标尺也可支撑后续正式门。fit 现要求 500 个不同 task/init、处理集数/行数一致、固定 K 集合；parity 要求标尺至少 500 行/50 每任务，并拒绝重复 parity 主键和负偏差。`legacy_alpha` 也锁定 0.05；曲线反序列化检查完整 measured tier 元数据、有限结点/纵坐标及合法 shape。
- **B4/B5/B6/B7**：执行者对 payload schedule / [16,32] / 有限值、H5 连续组名 / control_step / task / outcome、混合 episode 的事件/非事件计数、旧 shadow alpha/K 的修订均复核通过；进一步把 H5 schedule/坐标检查提前到 label 模型加载之前，且 pool 不再以任意任务数替代 10×5。
- **V1/N1–N4**：公共 docstring 已补全（含代理方法），AST 无缺失；logger 的 finally 清理与自有 CPU 数组测试通过；新测试 stub 使用 monkeypatch 并适配当前 staged runner 可选参数。历史冒烟证据保留，但明确其 SHA 不代表本轮修改后的代码；收回“教师随机性主导/正式 parity 预期稳过”的过度推断，修正 500/suite 的单位。同步正文接口、产物契约和索引。

**Checklist**

1. **与批准计划一致：PASS。** LOTO 整轨迹排除、全库 MISS 候选恢复、两档参考定义和同 W/H_exec 保持；新增校验落实冻结输入、样本与统计门槛。源码改动集中在新实验包及 LOTO 工厂的一条 eager 守卫；工作区另外的 online RIT / compile 修改按已有并行工作保留。
2. **测试覆盖与通过：PASS。** 完整 50 集/56 决策 fixture 实际调用 collect→label→report（仅 GPU/retrieval 运算用 stub；数值标签另有 runner 单测），小样本正确输出 insufficient_evidence。覆盖原始数据/标签身份被改、同 SHA 配套记录下的语义错误、缺半数标签、非法分组/分数/种子、冻结门槛、旧源码、未合并表与 checkpoint 缓存反例。
3. **文档与索引：PASS。** 正文/接口、`logs/README.md`、analysis 索引及 evidence 已同步；保留此前 Review Log；实验数值完成状态仍未宣称。
4. **无已发现回归：PASS（已运行范围）。** LIBERO 工厂与 GR00T staged/interceptor 组合测试通过；两 suite 真实旧 shadow 均完整复现。未运行全仓 pytest 或本轮修复后的远端 GPU/50 集正式闭环；这些不以 CPU advisory 结果替代。

**验证记录**

- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q tests/exp/test_rit_loto.py tests/review_tests/test_rit_loto_g2_contracts.py tests/libero_groot tests/cache/groot --tb=short -p no:cacheprovider` → **500 passed, 5 skipped**（65.01 s；5 个需要已完成 gate-pareto smoke 产物的检查跳过）。
- 最后补全全量表/parity/legacy-alpha 准入后，定向复验同命令的前两个文件 → **80 passed**（60 交付测试 + 20 复审探针，14.82 s）。
- 真实 `shadow_rows.jsonl` + `arm_record.json` 重拟合：**libero_spatial 3,432 行 / 48 臂 / 96 cuts；libero_10 8,383 行 / 48 臂 / 96 cuts** 全部相等；均 alpha=0.05、K={1,2,3}、alpha_source=legacy_reconstruction。
- `ruff check exp/rit_loto tests/exp/test_rit_loto.py exp/libero_groot/serve_groot_libero.py`、四个 ops 脚本 `bash -n`、`git diff --check`、新包 AST 公共 docstring 检查通过。
- 修复前暂存的 **18 个任务文件 index 对象逐个复核未变**；本轮源代码、测试、计划、证据和结论均未暂存，review_tests 继续 ignored 且从未入 index。共用 server/README 的其他并行修改未回退或重新暂存。

**运行边界**：本次是 G2 代码放行。§6 Verify、正式 noise-floor/parity、全量表与闭环结果尚待执行；由于源码和内容身份契约已变化，旧冒烟/旧 SHA 产物不能沿用为新版本准入。须在最终同步代码上重新生成整条冻结数值链。

**Final verdict: APPROVED — code approved（owner 授权直接修复后验收）。**
