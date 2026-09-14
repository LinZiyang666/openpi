# ActionCache 式 post-backbone（CP2）基线 — runbook

> 设计与 owner 决策：[`logs/actioncache_baseline_plan.log.md`](../../logs/actioncache_baseline_plan.log.md)（v0.6，G1 APPROVED R4）。
> 架构：[`docs/architecture/cache_system.md`](../architecture/cache_system.md) §3 CP2。
> 代码：`exp/actioncache_baseline/`（README 有逐脚本一览）。原文本地副本：`docs/papers/actioncache_2607.06370v2.{pdf,txt}`。

## 0. 这条线在做什么

在 libero_spatial / libero_10 上，用**我们现有的 CP1 库**逐条复刻 ActionCache（arXiv 2607.06370）的检索方式：key = backbone（Stage 2）最终层 prefix 输出 968×2048 → 固定稀疏三值随机投影到 500 维 → 单字段 cosine → 单阈值；suite 内全库检索、无 task 过滤（owner D14）。两档命中：

| 他们的 | 我们的 | 解析成本（CUDA-graph 档，ms） |
|---|---|---|
| N_hit=0 | FULL_HIT@CP2（跳过 Stage 3） | s1+s2 = 37.95 |
| N_hit=1 | WARM_START@0.1（Stage 3 只走最后 1 步） | 40.91 |
| miss | 完整 Stage 1+2+3 | 67.52 |

四组 = 2 suite × 2 库 regime（50 轨迹 `exp/rit_pareto` 库；cache_size S6 全集库），每组 ≤17 臂 × 500 集。切点不是扫阈值，而是用 shadow 分布按 GST K=1 的 IR 寻址反解 θ（§4）。

## 1. 前置条件

- server 端有 pi05 checkpoint（`--checkpoint-dir`），`uv sync` 完成；CP2 投影索引表每进程 +~40 MB CPU / +~40 MB GPU。
- 源 CP1 库 pkl 与其采集 H5（`vision_*` / `prompt_emb` / `robot_state` / `input_images`）在同一台机器上；H5 相对路径或 stem 能被库里的 `trajectory_id` 索引到（`libs.H5Index`）。
- 150 集 dev cohort H5（`exp/rit_pareto` 同一份，weilandserver `/tmp/dsp_shared/rit_pareto/<suite>/h5/`）用于 shadow 表与开销实测。
- 所有脚本以 `uv run python -m exp.actioncache_baseline.<tool>` 调用；`--seed` 是投影种子，四个库**必须用同一个 seed**，且与臂 yaml 的 `key_builder.cp2_vlm.seed` 相同（加载时绑定校验，不一致直接 `ConfigValidationError`）。

## 2. 建 CP2 库（每库一次）

```bash
uv run python -m exp.actioncache_baseline.build_cp2_artifact \
    --source-pkl <cp1_lib.pkl> --h5-root <h5 dir> \
    --checkpoint-dir <pi05 ckpt> --device cuda:0 --seed <SEED> \
    --out-pkl exp/actioncache_baseline/data/cp2_<suite>_<lib>.pkl
```

只改两个字段：`query_keys` 换成 `{"vlm_out": [500]}`、`checkpoint_id` 改为 CP2；`id` / payload / `prev_ids` / `next_ids` / `trajectory_id` / `step_idx` / `outcome` 原样继承（meta `id_policy: inherited_from_source`）。meta 另记 `projection`（seed/d/p/D/nnz/accumulation_dtype/digest）、源 pkl sha256、H5 manifest、`model.weights_digest`（checkpoint 目录**全部字节**的 sha256，约 10 s；shadow 表 / 开销实测 / parity 加载 checkpoint 前都用它做 fail-closed 绑定）、tokenizer、git commit。

**验证（fail-closed，任一项失败即库作废）**：

```bash
uv run python -m exp.actioncache_baseline.verify_cp2_artifact \
    --cp2-pkl <cp2.pkl> --source-pkl <cp1_lib.pkl> --out <verify.json>
```

检查 (a) id 集合一一对应 (a′) 全部 CP2 标签 (a″) 真 `InMemoryBackend` 加载后 CP2 `QuerySpec` 检索非空 (b) payload 逐位相等 (c) 链边闭合 (d) key finite float32 ×500 (e) 每条 `action_chunk` 严格 == (10,32) 且 `intermediates` 含 0.1 (f) `vector_dims` (g) meta 绑定字段（含 64 位 hex 的 `model.weights_digest`）。

**Stage 1 路径（2026-09-04 实测裁定）**：离线重建（H5 `vision_*`/`prompt_emb`，`_build_fake_stage1_with_masks`）会把被 mask 的右腕相机槽也当作有效 token（784 vs serving 的 528），backbone 输出随之偏离，CP2 key 与 serving 的 cosine 仅 0.70（三段 embedding 逐段 cosine 1.0，差在 pad mask）。因此建库 / shadow 表 / 开销实测**一律走在线路径**（`--stage1-path online`：H5 `input_images` + task → `Observation.from_dict` → 真 `run_stage1`，与 serving 同一代码），artifact meta 记 `stage1_path`，shadow / bench 加载时必须与库一致（`exp/actioncache_baseline/stage1_paths.py`）。

**parity 记录（每库抽 200 步，只记录不阻断）**：离线路径与在线路径两条 key 的 cosine（原门槛 0.999）：

```bash
uv run python -m exp.actioncache_baseline.parity_check \
    --h5-root <h5 dir> --checkpoint-dir <ckpt> --seed <SEED> --samples 200 \
    --expect-weights-digest <artifact model.weights_digest> --out <parity.json>
```

不达标 ⇒ 建库改走在线等价路径，不得放宽阈值。

## 3. shadow 表（零 rollout）

```bash
uv run python -m exp.actioncache_baseline.build_shadow_table \
    --cohort-h5-root /tmp/dsp_shared/rit_pareto/<suite>/h5 \
    --library-pkl <cp2.pkl> --checkpoint-dir <ckpt> \
    --out-jsonl exp/actioncache_baseline/data/shadow_<suite>_<lib>.jsonl
```

加载模型前先断言 `--checkpoint-dir` 的 `weights_digest` == 库 meta `model.weights_digest`（不一致直接退出）。每决策：重建 Stage 1 → `run_stage2_capture` → CP2 key → 对全库（`task_scoped: false`）取 top-1 cosine `s`；前 `--backend-check` 个决策同时用真 backend 复核 top-1 一致。输出附 cohort manifest sha256、库 sha256 与模型绑定记录。

## 4. 出臂（GST K=1，IR 寻址）

```bash
uv run python -m exp.actioncache_baseline.export_arms \
    --suite <libero_spatial|libero_10> --lib-tag <lib> \
    --shadow-table <shadow.jsonl> --library-pkl <cp2.pkl> \
    --deploy-library-path </abs/path/on/server/cp2.pkl> \
    --out-dir exp/actioncache_baseline/config/<suite>_<lib>
```

- `IR(θ) = [n(s≥θ)·c_tier + n(s<θ)·c_miss] / (N·c_miss)`；目标阶梯 60,65,…,95（默认），每目标取最近可达分位切；两档各加一条固定参考臂 `θ_raw = 0.85`（他们的默认）。
- **臂数预算机械执行**（`plan_tier_targets`）：n0 ≤ 8、n1 ≤ 7 个目标臂 + 各 1 参考臂 ⇒ 每组 ≤ 17 臂。目标被省略时 `export_record.json` 的 `skipped` 逐条给出原因：`below_tier_floor`（低于该档全放行的 IR 下界，例如 n1 的 60 < 60.6，不被 `--max-gap` 容差救回）、`no_cut_within_max_gap`、`duplicate_cut`（两个目标解到同一 θ）、`tier_budget`（幸存者仍超上限时从低 IR 端丢弃）。`budget` 字段记录每档臂数与总数；超限直接退出。
- 臂名 `acb_<sp|l10>_<lib>_<n0|n1>_<target>`；yaml 里 `threshold = θ_norm = (θ_raw+1)/2`（`affine_clip` 归一），N_hit=1 臂 `threshold: 1.5` + `warm_tiers: [{threshold: θ_norm, start_t: 0.1}]`。
- 每个 yaml 都过 `load_cache_config` 后逐字段断言（checkpoints=={cp2}、builder、keys、vector_dims、strategy、judge），`export_record.json` 记 `theta_raw / theta_norm / predicted_ir / ir_gap / library_sha256`。

## 5. 评测

server 用普通 `scripts/serve_policy.py`（`--cache_config` 指向臂 yaml；serving 装配会拒绝 stage 放置不合法的 CP2 配置）。client 走 gate_threshold_pareto runner，`--checkpoint cp2`：

```bash
uv run python -m exp.gate_threshold_pareto.run_gtp \
    --arm-matrix <out-dir>/arm_matrix.yaml --phase eval --checkpoint cp2 \
    --judge-type threshold --eval-gate always_search --warm-tiers 0.1 \
    --task-suite <suite> --servers <host:port,...> --workers <N> --trials 50 \
    --journal <run>/journal.jsonl --per-step-out <run>/per_step \
    --apool-record <apool.json>
```

`--checkpoint cp2` 让 `validate_arms` 对每个臂 yaml 执行完整 CP2 契约（`libs.cp2_contract_problems`，与导出器的 load-and-assert 同一实现）：checkpoints 恰为 {cp2}、builder / 唯一 key / vector_dims、`top_k=1`、`step_filter=all`、`task_scoped=false`、单字段 cosine + `affine_clip(-1,1)`、always_search、N_hit 两档 judge 形状、`write_policy=never`、无 routing/shadow/collect_meta；并核对臂名的档（n0/n1）与 suite 标签与 yaml 一致。任一条不满足在发出 rollout 前 `SystemExit`。默认 `cp1` 时行为与以前完全一致。噪声不配对（server 全局 RNG，无 seed 选项），两侧只在 `(task_id, init_idx)` 上配对；provenance 记 server commit / torch / CUDA / replica / worker。server 在每条 CP2 `__hit_meta__` 上附加 `library_sha256`（加载库的摘要），client per_step 行同名列落盘。

## 6. 聚合与对照

```bash
uv run python -m exp.actioncache_baseline.aggregate \
    --run-dir <run> --export-record <out-dir>/export_record.json --out <summary.json>
uv run python -m exp.actioncache_baseline.compare_to_reference \
    --cp2-run-dir <run> --ref-run-dir exp/rit_pareto/data/runs/<suite>_ng \
    --export-record <out-dir>/export_record.json --out <comparison.json>
```

- aggregate（fail-closed，`stats.audit_run` 镜像 `exp/rit_pareto/ops/audit_k3_group.py`）：终态行数 == 唯一 uid（0 dup，重复终态直接退出）；per_step `(uid, attempt)` 集合 == journal 终态 `(uid, attempt)` 集合；`failed` 且 `client_timing.steps` < 步数上限（spatial 200 / l10 500）判截断 ⇒ 列出 uid 剔除补跑；`failed` 而 verdict 行少于 42 / 100 同样报错；臂集合 == export record（缺臂 / 多臂都报）；每条 verdict 行的 `library_sha256` == export record 的库摘要（缺失或不等即退出）；每臂恰 `--expect-episodes` 集（**`--allow-partial` 只放宽这一条**）；`checkpoint == "CP2"`；档纯度门（n0 臂 WARM 行 = 0；n1 臂 FULL 行 = 0）。通过后：每臂 SR Wilson 95%、IR 双成本表（CUDA-graph 与 eager 并列）、预测 IR vs 实测 IR 逐臂、审计摘要写入 `aggregate.json`。
- compare（仅 50 库组）：要求同一 `export_record.json`，先复用 aggregate 的完整性、库摘要、arm 集与档纯度门（不能绕过 §3.11 gate），再计算 `ΔSR = SR_CP2 − SR_RIT-reference`。参考 = `exp/rit_pareto` K=2 no-gate 臂的**上凹包**（`stats.reference_hull`：同 cost 取最高 SR → 去支配点 → 弦检验剔除凹陷点；`[(0,.5),(1,.51),(2,.9)]` 在 x=1 插值为 .70 而非 .51）在 CP2 实测 IR 处插值；点估计与每个 bootstrap replicate 共用该实现；两侧按 task 分层 bootstrap（B=2000，逐 replicate 重建参考 hull，`support_miss` >1% 只报 descriptive）；三分裁决 `cp2_higher / reference_higher / indistinguishable`。S6 库组只做 regime 描述，不跨线插值；50 库与 S6 库非嵌套，不做"size 效应"结论。

## 7. CP2 开销实测

```bash
uv run python -m exp.actioncache_baseline.bench_cp2_overhead \
    --suite <suite> --cache-yaml <任一 n0 臂 yaml> --cohort-h5-root <cohort h5> \
    --checkpoint-dir <ckpt> --out-dir exp/actioncache_baseline/data/overhead_<suite>_<lib>
```

真模型 + `build_cache_components` 装真 orchestrator。臂 yaml 自带 `timer.enabled: false`（生产不打探针），harness 强制 `timer.enabled = True` 并把 monitor level 提到 BASIC 让 `SystemTimer` 真正记录；加载模型前先做 `weights_digest` 绑定。`torch.cuda.synchronize()` 包住整个 `check(CP2)`；每决策用 `on_task_begin` / `summary(task_only=True)` 读 orchestrator 自己的探针 `cp2_collect / cp2_gate / cp2_build / cp2_search / cp2_judge / cp2_fetch`，四个核心段任一没记录即中止（harness 故障）。输出 `per_decision.csv`（`episode, step_idx, total_ms` + 每段 `<segment>_ms` 列）+ `overhead.json`（`suite`、库与模型绑定、硬件、cold 前 50 / warm 其余的 median、P95、`per_segment`）。裁决：warm P95 ≤ 10 ms `ok_report`；10–40 ms `report_with_caption`（图注标注）；> 40 ms `halt_profile_segments`（停止发臂，先按分段定位，没有 profile 不预判 backend）。

## 8. 产物落位

`exp/actioncache_baseline/{config,data}/` gitignored（yaml、库、shadow 表、raw、overhead）；`analysis/` 入库（图、结果 .md）。库组成表（来源采集/init 池、轨迹数、entries、成功/失败轨迹、每 task 最少轨迹、horizon）每库必报。

## 9. GR00T N1.5 × LIBERO（同库 W13-S3 组）

> 设计与 owner 决策：[`logs/actioncache_baseline_groot_plan.log.md`](../../logs/actioncache_baseline_groot_plan.log.md)（v0.3，G1 APPROVED R2）。架构：`cache_system.md` §3 CP2 “CP2 on GR00T N1.5”。
> 只跑**同库大小对比**（每 suite 一个 W13-S3 50 轨迹库，与 `exp/libero_groot` RIT 线同一份 pkl 逐条构造），**不跑 500 集库版本**；每组 = 2 档（N_hit=0 / N_hit=1）× (4 目标臂 + 1 参考臂 θ_raw=0.65) = 10 臂 × 500 集。

### 9.1 与 pi0.5 线的差异（其余协议不变）

| | pi0.5（§0–§8） | GR00T N1.5 |
|---|---|---|
| key 来源 | Stage 2 `prefix_out` 968×2048 | action head **编码后**的条件：`process_backbone_output`（vlln + vl_self_attention）[N,2048] + `state_encoder` [1536]（论文 §4.1 / Table 5） |
| builder / 布局 | `cp2_vlm_ternary`，D = 1,982,464 | `cp2_groot_ternary`，`groot_encoded_v1`：token 轴零填充到 640 → D = 640·2048 + 1536 = 1,312,256；`projection` 元数据多一个 `layout` 块 |
| 投影 | 同一实现（seed 20260904, d=500, p=0.01, float32 累加） | 同 |
| N_hit=1 | WARM_START@0.1（10 步降序） | WARM_START@**0.875**（`groot_n15_k8_v1` 升序 8 步，剩 1 个 Euler 步） |
| 参考阈值 | 0.85 | **0.65**（Table 5 GR00T 默认） |
| 成本 | 解析表 CUDA-graph / eager，E=0 | 实测表 `exp/libero_groot/config/rit/cost_groot_libero_measured.json`（P=13.338, L=28.104, M=41.442）+ 每 suite 实测编码单价 **E>0**：FULL = P+E，WARM = P+E+L/8，MISS = M+E；分母 M；全 MISS 臂 IR > 100 % |
| 库构造 | H5 有原始图像 → 在线 Stage 1 | W13 H5 **无原始图像** → 零图模板重建（`stage1_path = groot_reconstructed_template`），逐步断言 `prompt_emb` / `robot_state` 回读一致 |
| 切点 | GST K=1 阶梯 + 档预算 | 首选 {45,60,75,90}；任一不可达（含 E 抬高的 N_hit=1 地板）则**整档** fallback：候选 IR 两端 + 最接近 1/3、2/3 处（并列取高阈值），`t01`–`t04`，`target_ir` = 该切点预测 IR；不足 4 个不同候选即失败 |
| 发臂前置门 | 无 | E 记录（CUDA-Graph certified）+ 决策开销 preflight（warm total P95 ≤10 报告 / 10–40 记提示 / >40 停发臂）缺一不可 |
| shadow cohort | `exp/rit_pareto` 150 集 H5 | RIT 线同一 150 集 init 池，**本机 5 个非并发 collector + 串行 client** 采集 H5，`verify_shadow_h5` 用 client 终态证据验收 |
| 执行环境 | 主 venv（jax） | GR00T 岛 venv（torch，无 jax）：`exp/libero_groot/*_groot.py` 与 `cp2_reconstruct.py` 在 `tests/cache/groot/test_import_isolation.py` 的（含传递）清单里 |
| schedule 守卫 | — | `load_guard`：cp2 配方两档都必须写 `denoise_schedule = groot_n15_k8_v1` 并与 live head 一致（n0 的 MISS 仍是 k=8 teacher），库的 `schedule_id` / `denoising_num_steps` 也在 `validate_artifact_identity` 里绑定 |

### 9.2 步骤（岛上，`PYTHONPATH=<gr00t>:<gr00t>/examples/Libero:<repo>/src:<repo>`，`HF_HUB_OFFLINE=1`）

1. **parity 门（全库建库前置）**：
   ```bash
   <libero client python> -m exp.libero_groot.emit_task_map --suite <suite> --out <acc>/task_map.json
   python -m exp.libero_groot.groot_cp2_parity --suite <suite> --checkpoint <ckpt> --task-map <acc>/task_map.json \
       --h5-root /archive/libero_cache/build_<spatial|libero10>_w13/<suite> --samples 20 --out <parity.json>
   ```
   四项均 fail-closed：合成观测两路（在线 vs 采集切片 fp16 往返重建）序列位同（只容忍 fp16 次正规区 1 ulp 的图像 token 差异——采集格式本身的限制，文本位置 / 状态仍位同）、key 余弦 ≥0.999；状态负例；20 个真实库步的文本/状态断言与 head 编码器直接比对；helper 纯度（stage 2 / action_inputs / RNG 不变）与 MISS、WARM@0.875 路径等价。
2. **建库 + 验证**：
   ```bash
   python -m exp.libero_groot.build_cp2_artifact_groot --source-pkl /data/libero_cache/libraries_w13/<suite>/<suite>_w13_S3.pkl \
       --h5-root <w13 h5 root> --out-pkl /data/libero_cache/libraries_w13_cp2/<suite>/<suite>_w13_S3_cp2.pkl \
       --checkpoint <ckpt> --seed 20260904
   uv run python -m exp.actioncache_baseline.verify_cp2_artifact --teacher groot_libero --cp2-pkl <out> --source-pkl <src>
   ```
   verifier 额外核对顶层/源/payload 的 `schedule_id == groot_n15_k8_v1`、`denoising_num_steps == 8`、`teacher`、`stage1_path`、chunk (16,32)、0.875 快照全覆盖。
3. **cohort H5 采集与验收**（collector 在 weilandserver，LIBERO client 在 sim box 上——`ACB_HOST=<collector host>`，client 终态 JSON 搬回 attempt 目录再验收；每 suite 顺序）：
   ```bash
   # 复用步骤 1 的 task_map.json：task_id -> (task.name, task.language)，来自 benchmark 本身
   python -m exp.libero_groot.verify_shadow_h5 --suite <suite> --shadow-manifest <manifest> --task-map <acc>/task_map.json \
       --attempts-root <root>/<suite> --out-dir <acc> --emit-full-filter                                     # 先写 filter_task_<t>.json
   bash exp/libero_groot/ops/launch_acb_collectors.sh <suite> <ckpt> <repo> <gr00t> <python> 0          # 5 个非并发 collector，8030+i
   ACB_HOST=<collector host> bash exp/libero_groot/ops/run_acb_collect_clients.sh <suite> <repo> <python> 0 <acc>   # sim box 上；client i 串行跑 task i、i+5
   python -m exp.libero_groot.verify_shadow_h5 --suite <suite> --shadow-manifest <manifest> --task-map <acc>/task_map.json \
       --attempts-root <root>/<suite> --out-dir <acc>
   ```
   身份：一个 LIBERO task 有两个名字——manifest 的 `task_name` 是 `task.name`（定位 `.init` 池文件的下划线名），H5 的 `task` attr 是 `task.language`（client 在 `episode_start` 发送、模型实际条件的自然语言指令）。两者只通过 `task_id` 与 task map 绑定，**不从一个推导另一个**；验收记录同时保留 `task_name` 与 `task_language`，下游（shadow 表 / bench / parity）一律用指令建模板。
   验收规则：H5 必须带 client 经 `episode_start` 打上的 `task_id` / `orig_init_state_idx` attrs 且与 client 行、manifest 一致，`episode_id == task_id*15+subset`；step 组恰为 `step_0000..step_{n-1}`；成功集 `termination_reason=success` 且 steps 不超过 `max_steps + 10`；失败集只接受 `step_cap` 且 `client_timing.steps == max_steps + 10`（230 / 530）；H5 `num_steps == step 组数 == infers == ceil((steps-10)/5)`；schedule 正确、数据集齐备。同一 attempt 里同一集出现两条 client 行或两个 H5 ⇒ 该 attempt 对该集无效（不取先/后者）；跨 attempt 两个合法终态报重复；同一 task 的指令串必须唯一。未通过时 `<acc>/retry/filter_task_<t>.json` 只含缺失集，用新的 attempt 编号串行补跑（`attempt_<a>` 目录不复用、不覆盖）。`accepted_shadow_manifest.json` 的 `ok:true` + `task_map_bound:true` 是后续所有步骤的准入。
4. **shadow 表**：`python -m exp.libero_groot.build_shadow_table_groot --suite <suite> --accepted-manifest <acc>/accepted_shadow_manifest.json --library-pkl <cp2 pkl> --checkpoint <ckpt> --out-jsonl <shadow.jsonl>`（前 50 步真 backend 复核；检索在 session 外做，与在线 check 同口径）。
5. **E 与 preflight**（独占 4090）：
   ```bash
   bash exp/libero_groot/ops/run_cp2_encoder_cost.sh <suite> <ckpt> <repo>       # nsys 三步：measure → trace → certify → config/actioncache/cost_groot_cp2_encoded_<suite>.json
   bash exp/libero_groot/ops/run_cp2_overhead.sh <suite> <ckpt> <cp2 pkl> <acc>/accepted_shadow_manifest.json <repo>   # data/actioncache/overhead_<suite>/overhead.json
   ```
   E 的 compile 区域是 `process_backbone_output` 的张量孪生（`vl_self_attention(vlln(x))` + `state_encoder`，先断言与 `run_cp2_key_source` 位同，`fullgraph=True`）：直接 compile 带 `BatchFeature` 的生产路径会 graph break、每次调用重编译，E 会假到 ~370 ms 而 launch 计数认证照过。E 记录与 teacher 表按内容绑定（`libs.groot_cost_record`）：两 suite 共用冻结 teacher 表并分别测 E；摘要的 `teacher_ckpt_sha256` 保留该表的标定 checkpoint，`ckpt_sha256` 保留本 suite 的 E checkpoint，两者允许不同。E 与 teacher 表使用同一 GPU（`gpu_uuid` / 名称）、冻结采样 N=566 / 30 warmup / 200 iters / `reduce-overhead`（两侧都核对）、`certified` 且 `valid`、`cudaGraphLaunch == expected > 0`、有限 `E > 0`、完整 `weights_digest`。measure 阶段若 GPU 与表不符或 live schedule 非 teacher8 则失败，E 的模型 digest 必须与本 suite 库一致；非冻结采样只能作调试（记录 `debug_sampling`），`certify-encoder` 拒绝认证；ops 脚本每次重测并归档旧记录，不凭 `certified:true` 跳过，measure / trace export / certify 任一步失败均返回非零退出码。
6. **出臂**（主 venv）：
   ```bash
   uv run python -m exp.actioncache_baseline.export_arms --teacher groot_libero --suite <suite> --lib-tag w13s3 \
       --shadow-table <shadow.jsonl> --library-pkl <cp2 pkl> --deploy-library-path <server path> \
       --cost-record exp/libero_groot/config/rit/cost_groot_libero_measured.json \
       --encoder-cost-record exp/libero_groot/config/actioncache/cost_groot_cp2_encoded_<suite>.json \
       --preflight-record exp/libero_groot/data/actioncache/overhead_<suite>/overhead.json --out-dir <out>
   ```
   两档必须恰为各一次 `n0,n1`，首选和 fallback 都用固定 `t01`–`t04` 标识以防小数 IR 取整重名；参考阈值固定 0.65，`max_gap` 至多 1 IR 点。全部目标规划完成后才写文件，恰 10 个唯一臂（`acb_<sp|l10>_w13s3_<n0|n1>_<t01..t04|ref650>`），`export_record.json` 记 `cost`（含 E、三档单价与模型/硬件/采样 provenance）、`selection`（首选/fallback 与弃用原因）、`shadow_binding`、`preflight`。写任何臂文件之前的门：shadow 表必须带完整 sidecar（同 teacher/suite/库 sha/projection/schedule/模型，`cohort_episodes == cohort_expected == 150`、`complete` 且非 `limited`，`n_rows` 与 `out_jsonl_sha256` 与正在读的表一致——`--limit-episodes` 的调试表不能发臂）；JSONL 实际覆盖 10 task × 15 subset，原始 init 不重复、每集决策连续且唯一、指令/episode/success 身份一致、cosine 有限，cohort/task-map 摘要是完整 sha256；E 记录的模型 digest == 库的；preflight 记录绑定同 suite/库/accepted cohort/projection/模型，且由 warm total P95 **重算**裁决（有限且非负、`warm.count == n_decisions − 50 > 0`、各核心段样本数 == 决策数且 median/P95 均有限非负），标签不一致或 >40 ms 即拒绝。
7. **评测**：server `launch_eval_servers.sh`（5 进程/lane，`--allow-dynamic-bundles`）；client 在 sim box 上 `bash exp/libero_groot/ops/run_acb_eval_group.sh <suite> <arm_matrix.yaml> <servers> <workers> <gpus> <run dir>`（`--checkpoint cp2 --judge-type threshold --eval-gate always_search --warm-tiers 0.875`）。先 smoke n0+n1 各 10 集。
8. **聚合**：`aggregate.py` 读 export record 的 `teacher`/`cost` 自动按 GR00T 单价计价（只报 `measured` 档），并核对 record 与成本摘要的 `suite` 等于臂 id 推出的 suite（E 按 suite 标定，显式 `--suite` 也不得覆盖臂身份）；同时复核 E 的模型/layout 与库一致、摘要 provenance 的值合法，并重算分母与三档单价以拒绝被改写的价格；档纯度门要求 n1 的 WARM 行全在 0.875，不可恢复的时刻直接拒绝。跨线对照 `compare_to_reference --ref-record <rit arm_record.json>` 要求参考侧 protocol / suite / k=8 与臂集合一致，参考 ledger 用本线实测表按 CP1 路径（无 E）重新计价，RIT 记录里的 interim 成本只记录不使用；**本轮不执行**。
