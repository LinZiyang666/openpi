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

**parity 门（每库抽 200 步）**：离线路径（H5 `vision_*`/`prompt_emb` 重建 Stage 1）与在线等价路径（H5 `input_images` + task + state → 真 `run_stage1`）两条 key 的 cosine ≥ 0.999：

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
