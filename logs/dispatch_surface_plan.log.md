# Dispatch Surface 实施计划（三档 (s,v) 调度 + 标定管线 + 增益预检）

> Status: **In Progress — 实验执行期**（G1 APPROVED R5；G2 APPROVED R4；Verify 通过；commit `1985271` 已 push；**成本轴 2026-08-27 变更，Review Authority 有条件批准；§4.6/§7 已按裁决修订冻结，analyzer 已重写、`run_cost_bench.py`/`power_sim_cost_blocks.py` 已删**） | Level: **L2** | Authority: Execution
> 数学权威：[`docs/iclr/latex/dispatch_note.tex`](../docs/iclr/latex/dispatch_note.tex)（v3 定稿）。攻防与裁决：[`docs/iclr/dispatch_defense_plan.md`](../docs/iclr/dispatch_defense_plan.md)（§3b 三档坍缩）。交接：[`logs/session_handoff_dispatch.md`](session_handoff_dispatch.md)。
> 本 plan 是 dispatch 线的**唯一流程**：数据独立性重建、noise 诊断、标定数据生成、surface 拟合与 conformal 校准、SurfaceJudge、闭环增益预检、预检后并入挂点，全部在本流程内交付代码与预注册执行方案，不留后续流程。
> 本 plan 经 G1 五轮评审（Round 1–4 共 39 条意见全部修订 + Round 5 owner 授权复审当场修复三处后 APPROVED）定稿；G1 APPROVED 2026-08-27 10:03 CDT。

---

## 1. 范围与不在范围

**在范围**（一次 G1 → Code → G2 → Verify 交付全部代码，实验按 §7 预注册时序执行）：

1. `src/`：新 judge 类型 `dispatch_surface`（三档判定：FULL_HIT / WARM_START(start_t=0.3) / MISS），surface artifact 格式（NPZ+JSON）、运行契约强绑定与 fail-fast 校验，config 接线与校验规则，v/Y 共享计算函数。
2. `exp/dispatch_surface/`：标定库 fresh 重建（主口径前置）、query cohort 采集驱动、noise-sensitivity 诊断、标定表生成（fresh 主口径 + uncoupled/tau1 sensitivity）、surface 与 s-only 拟合 + conformal + 诊断、预检 yaml 生成、闭环预检 runner 与配对统计分析。
3. 测试（含集成契约测试）、架构文档与索引同步。

**不在范围**：连续 τ 曲面的在线实现（论文理论保留，appendix/future）；E10 learned router 对照的执行；四臂全链（Arm1–Arm3）的执行（属 `actioncache_response_plan.md` §5 排期——本 plan 交付 Arm2.5 并入所需的 judge、artifact 与 yaml 形态）；RoboCasa / 真机。

**遵守 handoff §1 六条裁决**：三档坍缩（WS 档钉 start_t=0.3）；判定输入二维 (s,v)，v 称 local action disagreement；两事件 episode 级 split conformal（max-over-2），三 split 职责互斥；τ = 跳过步数、`start_t = (N−τ)/N`；training-free 限定语义；NP 代理事件 Z = 1{Y_N ≤ δ}。

## 2. 数学 → 实现映射

N = 10（`denoising_num_steps`）。三档对应 τ ∈ {0, 7, 10}：

| 档 | τ | start_t | 执行 | 数据来源 |
|---|---|---|---|---|
| MISS | 0 | — | 全量 S2+S3 | — |
| WARM_START | 7 | **0.3** | S2 + `run_stage3_from(intermediates[0.3], 0.3)`（3 步） | 重建库九档含 0.3 |
| FULL_HIT | 10 | — | 直接回放 `action_chunk` | — |

**统计量口径**（标定端与判定端共用同一 `src/` 函数，parity by construction）：

- **s** = `results[0].score`（weighted_score_sum 融合分数，`in_memory_backend.py:902` 产出）。
- **v** = 加权 top-k 动作分散度：top-k payload 的 `action_chunk[:h_exec]`（h_exec = 5 = `replan_steps`，只有前 5 步被执行），逐样本减均值，per-dim 乘 W，取平方范数均值。与 `TopkActionVariance`（`factors/topk.py:41`）同构，差异：整 h_exec 段而非仅 chunk[0]，且 W 加权而非 eps-mask 等权。
- **W** = per-dim 1/σ 对角权，σ 由重建库全体 entry 的 `action_chunk` 计算（`clamp_min(1e-6)`，active mask 阈 1e-8 同 `topk.py:38`），快照进 surface artifact。
- **Y_τ** = `chunk_deviation` 形式（per-step 加权 L2 的均值，`shadow_teacher.py:89` 同构但用 W，截 h_exec 段）。与 note 的 ‖W(·)‖ 差常数因子，口径在 artifact meta 记录。Y_{τ=10} 用 winner `action_chunk` 直接算，零去噪；Y_{τ=7} 离线跑 S2 + S3-from。**主口径 = fresh coupled**：参考 `a^(0) = F_c^{0→N}(z_j)`，z_j 为重建库 winner entry 的已知初始噪声（§4.1），与数学定稿 Eq. (Y) 逐字一致。
- **判定规则**（三档坍缩后的 Eq. τ*）：conformal 校准边界 `s_min_full(v)`、`s_min_warm(v)`（对 v 非降的阶跃函数，逐 v-bin 一个 s 阈值，`s_min_warm ≤ s_min_full` 逐 bin）。`s ≥ s_min_full(v)` → FULL_HIT；否则 `s ≥ s_min_warm(v)` → WARM_START(0.3)；否则 MISS。
- **Fail-closed 语义（`uses_disagreement=true` 的 artifact，穷举，全部 → MISS）**：空 `results`；**s 非有限**（+inf/−inf/NaN score 不得绕过守卫）；`view is None`；`get_many` 抛异常或返回不足；K_eff < 2；v 非有限（NaN/inf）；**v > 最右 bin 上边界（超出 fit 支持域）**。唯一的非 MISS 例外：v < 最左 bin 下边界 → 用最左 bin（(A1) 单调性下真实分位 ≤ 该 bin 拟合值，方向保守）。`uses_disagreement=false`（s-only artifact，§4.5）：judge 不 fetch payload、不计算 v、无任何 v 守卫，fail-closed 仅剩空 `results` 与非有限 s → MISS——verdict 在逻辑上只依赖 s。两种 artifact 下 CP3 均防御性 MISS。
- **Conformal**：episode 级三分 split（library / fit / calibration 互斥，test = 闭环，§4.2）。fit 拟合 `q̂(τ,s,v)`（τ ∈ {7,10} 两层，格约束 pinball）；calibration 算一次 `R_e = max_{i∈e} max_{τ∈{7,10}} (Y_{i,τ} − q̂)`，c = 第 ⌈(1−α)(|E|+1)⌉ 序统计量，要求 |E| ≥ 19（α=0.05），否则 c=+∞ 规则退化全 teacher（fail-valid）；`q̃ = q̂ + c` 同时覆盖两档；calibration 数据不得回访。
- **符号约定测试钉死**：τ=7 ↔ start_t=0.3 的映射（`round(1.0 − 7/10, 4) = 0.3`）写成显式单测，防 sign 反转。

## 3. `src/` 设计

### 3.1 `SurfaceJudge`（新文件 `src/openpi/cache/components/surface_judge.py`）

Artifact 持久化为 **NPZ**（数组字段）+ 内嵌 JSON 字符串（标量与契约），不新增 pickle 面：

```python
@dataclasses.dataclass
class SurfaceArtifact:
    schema_version: int          # 1
    k: int                       # top-k for v (>= 2)
    h_exec: int                  # leading chunk steps used by v and Y (= 5)
    w: np.ndarray                # [action_dim] float32 inverse-sigma weights, finite, > 0
    active_mask: np.ndarray      # [action_dim] bool, at least one True
    start_t_ws: float            # 0.3; must be in CANONICAL_DENOISE_TIMESTEPS
    delta: float                 # > 0, finite
    alpha: float                 # in (0, 0.5]
    uses_disagreement: bool      # False = s-only artifact: no payload fetch, no v computation
    v_bin_edges: np.ndarray      # if uses_disagreement: [B_v + 1] strictly increasing, finite
                                 # else: fixed sentinel [-inf, +inf] (B_v = 1), enforced exactly
    s_min_full: np.ndarray       # [B_v] nondecreasing in v; +inf allowed (bin never hits); -inf/NaN rejected
    s_min_warm: np.ndarray       # [B_v] same domain rules; elementwise <= s_min_full
    conformal_c: float           # +inf allowed (fail-valid degenerate: all boundaries +inf, rule = all-MISS)
    n_calibration_episodes: int
    retrieval_contract: dict     # binding contract, see below
    meta: dict                   # ref_mode, h_exec provenance, fit config hash, created_at, deviation-metric note
```

NPZ 读取显式 `np.load(path, allow_pickle=False)`；object array 拒绝；契约与 meta 为内嵌 JSON 字符串字段。

**运行契约强绑定**（`retrieval_contract`，装配期逐项比较，任一失配 raise `ConfigValidationError`）。契约值一律为 canonical digest：对相应 config dataclass 字段做 sorted-JSON 序列化后 sha256，覆盖**一切影响 s 或候选集的参数**：

| 契约字段 | 内容 | 比对对象 |
|---|---|---|
| `key_builder_digest` | key_builder type + prompt-pool knobs | yaml `key_builder` 段 |
| `search_digest` | strategy type、enabled fields、fusion_weights、fusion_method、**field_similarity（per-field sim type）**、**step_filter/window**、**trajectory depth/weights/query-id 设置**、完整 `score_normalization`（method + 全 params） | yaml 检索段 |
| `top_k` | = k（uses_disagreement=true 时）/ 无约束（false 时） | 装配后 effective top_k |
| `library_sha256` / `library_entry_count` | 拟合期所用库 | `CacheStorage.artifact_meta`（identity 扩展，见下） |
| `action_dim` / `num_steps`(N) | 动作维度与去噪步数 | `artifact_meta` 的 action schema 字段 |
| `h_exec` | = 标定所用执行窗（5） | **runner launch 契约**：`run_precheck.py` launch 前断言 client `replan_steps == h_exec` |
| `policy_fingerprint` | `compute_policy_fingerprint(resolved_checkpoint_root, config_name)` = sha256(canonical config 名 + checkpoint 根目录下逐文件的 `(relative_path, size, sha256(content))` canonical manifest)，重建/标定脚本运行模型时现场计算写入 | **server 自报告的同函数值**（attestation seam，见下），runner 比对，错配 fail-fast |

**Policy attestation seam（additive，入 scope）**：操作者自报字符串、文件名/大小清单都不构成内容绑定。新增共享实现 `src/openpi/serving/policy_identity.py`，只负责把 checkpoint URI 经 policy loader 已用的 downloader/resolver 物化为本地 canonical root，并计算内容摘要；`scripts/serve_policy.py` 内的 `_resolve_effective_policy_spec(args)` 负责把 `Default` 或显式 `Checkpoint` 唯一映射为 canonical config 名与 URI，避免 `src/` 反向依赖 script dataclass。server 将这一 resolved spec 同时交给 create-policy 与 fingerprint，二者不得分别解析。摘要对根目录下普通文件递归流式 sha256；manifest 按 POSIX relative path 排序，拒绝空目录、特殊文件和逃出根目录的 symlink。该摘要每个进程只算一次并缓存结果，不在请求路径计算。三方调用同一实现：重建/标定脚本写入 artifact contract；**`scripts/serve_policy.py` 用同一份 resolved spec 创建 policy 并把 fingerprint 与 effective `monitor_level` 放入 websocket metadata（additive 字段）**；runner 连接后从 `get_server_metadata()` 读取并与 artifact contract/预期 pass 比对——server 加载了什么就报告什么，同大小替换权重也会改变摘要。`replan_steps` 仍为 runner CLI 显式无默认参数（先例：risk_router）。契约值与 metadata 写 manifest；judge 计算 v 用 artifact 内 h_exec。

**Identity seam（additive，入 scope）**：现 `artifact_meta` 只含 `key_builder_type/checkpoint_id/prompt_pool`。扩展 `InMemoryBackend.load_artifact`：加载时对 pkl 文件流式计算一次 sha256，**扫描全库 entry 只做记录不做断言**——聚合 `entry_count`、action schema 共识值与一致计数（`action_dim`/`action_horizon`/`denoising_num_steps` 的众数及符合数）、各 intermediates 键完整率统计，全部挂入 `artifact_meta`；`CacheStorage.artifact_meta` facade 原样透传。generic loader **不因 schema 异质、缺 intermediates、payloadless entry 拒绝加载**（`CachePayload` 合约允许 FULL_HIT artifact 无 intermediates/N，legacy 与混合 CP artifact 必须照常可用，非回归测试覆盖）。仅 load-time 一次计算（BackendPool 每 fingerprint 只 load 一次），对所有既有 yaml 字节等价。

校验分两级：yaml 级在 `validate_cache_config`（digest 由 yaml 侧重算比对，不依赖已加载库）；库级在 `build_per_connection_components` 装配点（storage 在手，比对 `artifact_meta`）。带 surface 证书的配置强制 `preload_path` 非空（frozen 库，C2）且 `write_policy: never`。

加载期数组校验（构造时 raise）：schema_version 匹配；数组 dtype/shape 一致；`uses_disagreement=true` 时 `v_bin_edges` 严格递增且有限、k ≥ 2；`uses_disagreement=false` 时 `v_bin_edges` 必须逐字等于 `[-inf, +inf]` 且 k 字段被忽略（`min_required_top_k = 1`）；边界数组允许 +inf、拒绝 −inf/NaN；两条边界对 v 非降且逐 bin `s_min_warm ≤ s_min_full`；`conformal_c` 允许 +inf，且 **c=+inf ⇒ 两条边界必须全为 +inf**（fail-valid 退化是 loader 不变量而非注释，破坏即 raise；退化 artifact 可加载可执行=全 MISS，加载测试覆盖）；start_t_ws ∈ canonical；alpha/delta 域检查；active_mask 至少一维为真。

**Surface 条件化库断言（装配 `dispatch_surface` 时才执行，消费上述 metadata）**：action schema 一致计数 == entry_count（无异质）；`h_exec ≤ action_horizon`；`denoising_num_steps == N`；部署档 `t=0.3` 完整率 100%。任一不满足 → 装配期 raise。与 surface 无关的既有配置不受任何新断言影响。

```python
class SurfaceJudge:
    def __init__(self, artifact_path: str): ...   # load + validate (fail-fast)
    min_required_top_k: int                        # = artifact.k; read by build_per_connection_components
                                                   # (config.py:2856) and forwarded as min_top_k_hint
    export_factor_outputs: bool                    # from JudgeConfig; when True attach {"s","v","v_bin","verdict_src"}
                                                   # to JudgeResult.factor_outputs (rides existing __hit_meta__ path)
    def __call__(self, results, checkpoint_id, cached_data, *,
                 view=None, history=None, retrieval_signals=None) -> JudgeResult: ...
    def on_episode_start(self): ...
    def record_action(self, action_chunk): ...
```

广度接口用装配链实际读取的 **`min_required_top_k`**（`config.py:2856` → `min_top_k_hint` → `config.py:3663` `effective_top_k`），不是 Factor 层的 `required_top_k`。payload 拉取经 `view.get_many(...)`（`StoragePayloadView` per-check memo）。

共享函数（同文件，标定脚本 import 保证逐位一致）：

```python
def weighted_topk_disagreement(chunks, w, active_mask, h_exec) -> float   # [K,H,D] -> v; K<2 -> nan
def weighted_chunk_deviation(chunk_a, chunk_b, w, active_mask, h_exec) -> float   # Y
```

### 3.2 config 接线（`src/openpi/cache/config.py`）

- `_JUDGE_TYPES` 增 `"dispatch_surface"`（config.py:670）。不进 `_ROUTING_JUDGE_TYPES`。
- `JudgeConfig` 增字段 `surface_artifact_path: Optional[str] = None`；复用现有 `export_factor_outputs`。
- `_build_inner_judge` 增分支（config.py:3212，懒导入同现有模式）。
- `validate_cache_config` 新规则：type 为 dispatch_surface 时 `surface_artifact_path` 必填、文件存在且 yaml 级契约通过；CP1-only；与 `warm_tiers` / `start_t` / composite 字段族互斥；`preload_path` 非空 + `write_policy: never` 强制。
- 装配点 `build_per_connection_components`：库级契约比对（`library_sha256`/`entry_count`/`action_dim`/`N`，经 identity 扩展后的 `artifact_meta`）+ `uses_disagreement=true` 时断言 effective top_k ≥ artifact.k。
- `judge.py` facade re-export（judge.py:566 模式）。

YAML 形态：

```yaml
judge:
  type: dispatch_surface
  surface_artifact_path: exp/dispatch_surface/data/cache_artifacts/surface_spatial16_primary.npz
```

## 4. `exp/dispatch_surface/` 设计

布局按 `docs/experiments/artifact_layout.md` 四槽。检索配置固定为主线组合：spatial16 keybuilder + `weighted_score_sum_knn` + `top_k = 5`（由 `min_required_top_k` 抬）。

### 4.1 `rebuild_dispatch_library.py`（主口径前置，无条件执行）

原引用的 `exp/common/data/cache_artifacts/libero_spatial_warm/` 已在 factor-artifact rebuild 中删除（本机已核实），且 fresh coupled 主口径要求库 entry 的初始噪声已知。因此**统一重建**：对库源 H5（`exp/common/data/db/libero_cache/libero_spatial/`，50 ep / 1,062 step）离线重跑 S2+S3（`return_intermediates=True`，`save_timesteps` 九档），每 entry 现场采 z（`sample_noise(generator=...)` + `stable_seed` 记 seed），产出：

- `dispatch_lib_spatial16.pkl`——标准 artifact 格式（payload 不加字段，在线语义不变），**标定、拟合、闭环预检三处强制加载同一份**；
- `dispatch_lib_noise_sidecar.npz`——entry_id → initial-noise tensor 侧表（仅标定消费，不进运行时）；
- 重建记录（entry 数、逐 entry t=0.3..0.9 九档完整率、sha256）写入 `data/` 与 registry 台账；sha256 进 surface artifact 的 `retrieval_contract.library_sha256`。

成本 ≈ 1,018 entry × ~0.5 s ≈ 10 min GPU。W 与 active_mask 在此库上计算。

### 4.2 数据独立性：四方互斥 split + super-pool 划分

数学定稿要求 D_lib 冻结且 fit/cal/test 均不贡献库 entry（`dispatch_note.tex:161-178`）。LOEO 方案废弃。官方 init 每 task **恰 50 个**（`get_task_init_states` 返回形状钉死，`materialize_apool.py` 断言 50），且已全部被 A-pool 物化——不存在官方差集空间。因此 query 与 test 从**同一 super-pool 随机划分**（同一随机机制，exchangeability 由构造保证）：

- **D_lib init 占用是已知事实，不是运行时意外**：权威 provenance `exp/common/data/db/libero_cache/libero_spatial_init_map.json` 逐条记录库源 50 episode 的 `orig_init_state_idx/full_init_path`——**每 task 恰占 5 个官方 init**（如 task 0 = {0,13,30,39,45}，本机已核实）。配额据此冻结：每 task 官方 50 中 **5 归 D_lib（已占）、15 → query 池 C（5 fit / 10 cal）、30 → test 池 A′**。
- `split_init_pools.py`：以 init map 为权威输入（比对 `full_init_path` digest，**不用低维 robot_state 猜同一性**），先扣除 D_lib 的 5 init/task，再以固定 seed 对剩余 45 随机划分 15/30；两池 `materialize_apool.py` 格式物化，`verify_apool.assert_disjoint(A′, C)` 且两池均与 D_lib init 集不相交，三方 digest 落 manifest 与 records。**Provenance 异常分支（fail-loud）**：init map 缺失/条目数 ≠ 50/映射不完整/占用数 ≠ 每 task 5 → 中止上报 owner，不自行猜测。

| Split | 来源 | 规模 | init 池 |
|---|---|---|---|
| **D_lib** | 现有 50 ep H5 → §4.1 重建库 | 1,018 entries | 官方 init 每 task 5 个（init map 记录） |
| **fit** | **新采 teacher rollout H5** | 50 ep = 5 init/task（~1,050 step） | C |
| **calibration** | 同批新采 | 100 ep = 10 init/task（~2,100 step） | C |
| **test** | 闭环预检 | **300 ep/臂 = 30 init/task（`--trials 30`）** | A′ |

query cohort 采集：`serve_policy.py --collect` 纯 teacher rollout（无 cache），150 ep ≈ 数小时单卡（一次性）。calibration |E|=100 ≫ 19，correction 为第 96 序统计量而非样本最大值。

Handoff 的"零 rollout"指标定表生成无需闭环交互（仍成立：标定表是离线回放）；query cohort 采集是一次性数据补充，服从数学定稿的独立性要求，数学定稿优先。

### 4.3 `noise_sensitivity.py`（preliminary diagnostic，不驱动口径）

从 fit split 分层抽 ~50 step；每 step 重建 conditioning（`_build_fake_stage1_with_masks`），S2 一次、S3 全程 M=8 次（独立 z、记 seed），pairwise `weighted_chunk_deviation` 得 teacher–teacher 分布；对照 warm–teacher 分布。输出 `analysis/noise_sensitivity/`。**角色 = note 承诺的 preliminary diagnostic 报告**（TT vs WT 幅度对比进论文），不再决定主口径。

### 4.4 `build_dispatch_table.py`（标定表生成）

结构参照 `exp/zixuan_proposal/build_calibration_table.py`（生产 KeyBuilder 链、时序镜像 `CacheOrchestrator.check()`），检索对象 = §4.1 重建库（query 与库天然互斥，无需 LOEO；保留 fail-loud 断言：winner trajectory_id 不属于 query episode）。逐 fit/calibration episode 逐 step 输出 JSONL：

```
{episode_id, task_id, step_idx, split, s, v, k_eff, winner_id,
 y_tau7, y_tau10, ref_mode, episode_success}
```

- **主口径 `--ref-mode fresh`**：a_ref = 当前 conditioning 下从 winner 的 z_j（noise sidecar）全程生成 `F_c^{0→N}(z_j)`；warm 分支 = S2 + `run_stage3_from(winner.intermediates[0.3], 0.3)`；同 conditioning 一次 S2 喂两分支。
- **Sensitivity 列（代码同交付，均不作论文主 estimand）**：`uncoupled`（a_ref = query 自身 `clean_action`，estimand 改为对随机 teacher 样本的总体偏差——note 已含此改述）、`tau1`（a_ref 从 winner `intermediates[0.9]` 跑 9 步，Y_1 ≡ 0 参考）。两列在 fit split 上与 fresh 对比报告，量化口径敏感性。
- 成本：~3,150 step ×（全程参考 10 步 + warm 3 步 + 检索）≈ 每 step ≤ 0.7 s（eager 保守）→ ~1–2 GPU 小时。

### 4.5 `fit_surface.py`（拟合 + conformal + 诊断 + artifact）

1. **分箱**：fit split 上 (s,v) 等频 B_s=12 × B_v=6；bin 内样本 < 8 与相邻合并。
2. **联合单调分位拟合**：τ ∈ {7,10} 两层格点 pinball loss（1−α 分位），约束对 s 非增、对 v 非降、层间 q(7)≤q(10)。scipy `linprog`。
3. **Conformal**：calibration split 一次性 `R_e`、c（|E|=100 ≥ 19 检查仍在）。
4. **边界导出**：逐 v-bin 取满足 `q̃_τ ≤ δ` 的 s-bin 下确界的保守侧（bin 上边缘）；全 bin 不安全 → `s_min = +inf`；导出后 assert 单调与嵌套。
5. **s-only 退化拟合**：同一脚本 `--s-only` 产出 `uses_disagreement=false` 的 artifact（哨兵 `v_bin_edges=[-inf,+inf]`）——同 fit/cal 数据、同分位/单调约束（仅 s 维）、同 conformal 构造、同 δ\*。判定语义真 s-only（§2：不 fetch payload、不算 v、无 v 守卫），这是 §4.6 嵌套消融臂 S0。
6. **Primary δ 冻结（进 test 前，完全机械，OOF 构造）**：
   - **候选 grid**：fit split pooled `{y_tau10}` 的 9 个分位点 {P10, …, P90}，**去重后**为候选集；unique 候选 < 2（退化分布）→ 止损点 A。
   - **fold 划分**（固定）：task 分层 5-fold，episode 级，每 task 的 5 个 fit init 按 init 序号 i mod 5 指派（零随机自由度）。
   - **OOF 拟合**：每 fold 在 4/5 上以同一 B_s×B_v 与约束重拟合，得每个 fit episode 的 **out-of-fold** 预测 q̂^(−fold(e))。**不做逐 fold 伪 conformal**（fold-out 仅 10 ep < 19，按 §2 规则必然 c=+∞，不可执行）。
   - **单次 OOF safety offset**：把全部 50 个 fit episode 的 OOF episode-max 残差 `R_e^{OOF} = max_i max_τ (Y_{i,τ} − q̂^{(−fold(e))})` 汇成一个 n=50 池，取 α=0.05 order statistic（第 ⌈0.95×51⌉=49 个，49 ≤ 50 合法）得 **`oof_safety_offset`**（不称 c、不称 conformal——残差来自五个不同训练集的 OOF 模型，仅为 fit-only δ 选择的安全垫，不构成第二张覆盖证书；正式保证只由 calibration split 的 c 给出）；边界 = q̂^{OOF} + offset 按 δ 候选切割。每个 episode 的评估始终来自未见过它的 fold 模型，无泄漏；n=50 ≥ 19 与本 plan 的有限样本规则相容。
   - **逐候选 δ 两个量**（在全部 fit episodes 上）：`hitshare(δ)` = 非 MISS 步占比；`accepted_step_accuracy(δ)` = 非 MISS 步中 `Y_{i,τ(verdict)} ≤ δ` 的占比（**改名自 cov：这是 fit-only 启发式的已接受步准确率，不是 episode 级 simultaneous conformal coverage，论文与产物均用此名**）。接受集为空的 δ：accepted_step_accuracy 定义为 1（空真），但 hitshare=0 使其不可能通过下述选择。
   - **选择规则**（唯一、可测试）：合格集 Q = {δ: accepted_step_accuracy(δ) ≥ 1−α−0.05 且 hitshare(δ) ≥ 0.40}。Q ≠ ∅ → δ\* = argmax_Q hitshare，tie 取更小 δ；Q = ∅ 但 accuracy 门有通过者 → 该集合内 argmax hitshare（同 tie-break）；accuracy 门全失败或全候选 hitshare = 0 → **止损点 A**。
   - SV± = δ\* 在去重 grid 上的左右邻点（不存在则省略该侧），仅 secondary 描述。**δ 选择全程只读 fit split；正式 conformal 校正 c 仍由 calibration split（|E|=100）单独一次计算，与 `oof_safety_offset` 无关。全部常数（grid、fold、0.40、0.05、tie-break、空集语义）在此冻结，执行期无自由度。**
7. **诊断（E5 前两项）**：(A1) 经验违反率（fit 相邻 bin 逆序对率）；MLR 诊断（Z = 1{y_tau10 ≤ δ\*} 对 s 的条件密度比单调性，isotonic 检查）。**E5 第三项（deviation↔closed-loop failure 关联）不可用离线 H5 判定**（库源 cohort 49 success / 1 failure，近常数标签）：降级为预检后对 test per-step 数据（`export_factor_outputs` 带回的 s,v + episode 结局）的**事后描述性分析**，不参与任何选点。按 5% 失败率仅 primary SV 约 15 个失败 episode；若 SV± 两侧都存在，三个含 v 的 surface 臂合计也仅约 45 个，明确不作功效或确认性主张（T1–T3 与 s-only S0 不得冒充 v 观测补样本）。

### 4.6 预检（闭环，1 suite libero_spatial）

**臂表（核心 5 臂固定 + SV± 动态 0–2 臂，各 `--trials 30` = 300 ep/臂（A′ 池 30 init/task），核心 1,500 ep、满配 2,100 ep）**：

| 臂 | 判据 | 角色 |
|---|---|---|
| T1/T2/T3 | `threshold` + `warm_tiers:[{T_ws, start_t:0.3}]`，(f_FH,f_WS) ∈ {(30,20),(50,20),(70,10)} | operational baseline 前沿 |
| S0 | `dispatch_surface`，s-only artifact（`uses_disagreement=false`）@ δ\* | 嵌套消融 + 条件对比臂 |
| SV | `dispatch_surface` @ δ\* | **primary** |
| SV−/SV+ | @ δ\* 去重 grid 邻档，**存在才 emit（δ\* 在 grid 端点时缺侧省略）** | secondary，描述性 |

**Arm cardinality 动态闭合**：emit 按 δ\* 的 grid 位置确定性生成 5/6/7 臂矩阵，manifest 记录实际臂集；一切 Gate 只消费核心 5 臂，SV± 缺席不影响任何裁决路径。成本自主预检解析得到，故 SV± 的成本与 SR 一样自动可得，仅作描述。

- **T1–T3 recipe 冻结协议**：阈值由 `derive_thresholds`（唯一实现）在 **fit∪cal 的 s 分布**上按预注册 (f_FH,f_WS) 反解。这是 baseline 自身的标定算法（matched calibration budget），其对 cal 分数的这次读取发生在看 test 之前、recipe 参数在本 plan 冻结，不反向影响 surface 的 q̂/c/δ\*——不构成 surface 的 calibration 回访。冻结时间与输入 digest 落 launch manifest。
- `emit_precheck_yamls.py`：基板 = gtp 模板，替换 judge 段 + `preload_path`（§4.1 重建库，emit 时 assert preload sha == artifact 契约 sha）+ `write_policy: never`；**gate 段重解并覆写，不沿用模板**（2026-08-28 修正，owner 指出）。
  - 权威做法见 `exp/gate_threshold_pareto/solve_gtp.py`：gate theta 与 judge 阈值是**同一分数分布的分位切**，必须 per-library 重解——「carrying a threshold across libraries is exactly the mistake the ratio-based design exists to prevent: two libraries with different score spreads reach the same operating point at different numeric cuts」。本线重建了库，沿用模板 theta 会让 gate 落在它未被标定的操作点上。
  - 因此 `theta = derive_thresholds(fit∪cal scores, THETA_TOP_FRACTION=0.85, 0.0)`，与 T1–T3 的 `T_fh` 同源同实现；gate 结构参数 `j=3 / probe_interval=3 / L=6` 从 `emit_gtp_yamls` 导入而非重新定义。样本量不足 `MIN_SCORES=500` 时 fail-fast。
  - **同一个 theta 用于全部臂**（T1–T3 / S0 / SV / SV±）且跨 sweep 固定：gate 决定某步是否被 probe，若它随臂变动，SR 或成本的差异就无法归因于被检验的 verdict 规则。theta 及其分位、结构参数、解算样本数一并写入 arm matrix 作 provenance。
  - 两 suite 的模板 gate 原本就不同（spatial 带旧库的 `score_hysteresis theta=0.968929`，l10 带 `always_search`），沿用会让两条线的 baseline 前沿不是同一种东西。
- `run_precheck.py`：薄 conductor runner，结构复制 `run_gtp.py`（arm matrix、resume 过滤、init 池 digest 重算、journal/per_step 落盘），去掉 no-warm-tiers 断言，init 池指向 A′；runner 直接以 split manifest 的 30/task assignment、逐 task state-content digest 和 worker 实际读取目录作三方核验，**不得复用 cache-size 线冻结 50/task 的 A-pool verifier**。A′ 已物化成 30/task，因此 worker 显式使用 subset position 索引文件，同时 `orig_init_state_idx` 保留官方 0..49 provenance；不得用 official index 直接索引 30 条数组。**全臂同 A′ init 文件、同 env seed、同 conductor seed —— 配对纪律**，launch manifest 记录三者 digest。**Launch 契约 fail-fast**：launch 前读 primary artifact 的 `retrieval_contract`，断言 client `replan_steps == h_exec`（显式参数、无默认），且同一 `replan_steps` 与冻结 env seed 必须传入实际 worker；连接 server 后从 server metadata 读取实际 `policy_fingerprint` 与 artifact contract 比对（server 自报告，不接受操作者自报字符串），两者写入 manifest。主预检同时产出 **SR 与解析成本两轴**的全部输入（`per_step.jsonl` 的逐 step verdict）。
- **成本轴：从主预检解析计算，无独立成本实验**（2026-08-27 变更，Review Authority 有条件批准，见 [`dispatch_surface_cost_axis_change.md`](dispatch_surface_cost_axis_change.md) §9/§10）。成本口径是 **GPU inference ratio**，是 verdict 的解析函数，不是墙钟测量：

  | verdict | per-decision 成本 | CUDA-graph 档 |
  |---|---|---|
  | `FULL_HIT` | stage1 | 10.26 ms |
  | `WARM_START` | stage1 + stage2 + `start_t` × stage3 | 46.821 ms（`start_t=0.3`） |
  | `MISS` | stage1 + stage2 + stage3 | 67.52 ms |

  单位成本取自 `exp/data_authority/records/latency_bench__libero_spatial__executor_costs.json` 的 **CUDA-graph 三阶段档**（`pi05_compile_ro_3stage.json`：10.260266 / 27.686469 / 29.571860 ms），eager 与 default 档描述未优化系统，不得使用。`start_t=0.3` 表示**执行 30% 的 stage3**（`run_stage3_from` 跑 `round(start_t × num_steps)` 步，见 `pi0_pytorch.py:691`），该语义由测试固定。检索 CPU 不占 GPU、不进成本。

  **estimand = decision-weighted ratio-of-sums**：`C_a = Σ_d c(h_d) / N_decisions`。**不得**先求每 episode 的 per-decision 均值再对不等长 episode 等权平均——那估的是"随机 episode 的平均成本"，与本线主张的"随机 decision 的 GPU inference cost"不是同一个量（实测 episode 决策数 14–44 不等）。每个 bootstrap replicate 必须按被抽中的完整 init cluster **重新汇总分子与分母**再作比。

  数据来源是主预检自身的 `per_step.jsonl`（逐 step 记 `hit_type` 与 `start_t`，带 `(yaml_id, task_id, orig_init_state_idx, episode_id)` 完整 join key），与 SR 用同一批 episode、同一批 init、天然配对，**不额外跑任何 episode**。删除的旧设计：`run_cost_bench.py` 双 server launch、block 设计、`R∈{5,10,15}`、`power_sim_cost_blocks.py` 及其 variance source 与确定性重放、`stage_probe_backends` CUDA 断言、原止损点 B。

  正式文字一律称 **解析 GPU inference cost** 或 **model-forward compute proxy**，**不得**写成实测端到端延迟。`client_timing` 可作描述性报告并注明多 worker 竞争限制，**不参与任何 Gate 判定**。

**预注册统计裁决（进 test 前冻结，测后不得改）**：

- **单一联合 replicate**（B=10,000）：每个 replicate 在 task 内以 `orig_init_state_idx` 为 cluster 作**有放回配对重采样**，**同一 replicate 对所有臂使用同一套 task/init 抽样索引**，并在该 replicate 内**同时**计算各臂 SR 与解析成本、以及 Gate 1 的 threshold-frontier 插值。**禁止为 SR 与成本分别抽样**——Gate 2 的 intersection-union 检验的有效性依赖两个统计量的联合分布。
- **每 replicate 的唯一输出（endpoint-clamped，analyzer 零自由度）**：每个 replicate 用其 T1–T3 `(SR, C)` 按 `(SR, arm_id)` 稳定升序构造 operational threshold polyline（SR 相同则 arm_id 打破 tie），与 SV 的 `(SR_sv, C_sv)` 比对，产出恰好一条记录 `{branch, D_sr, D_c}`：
  - `branch=bracket`（min T.SR ≤ SR_sv ≤ max T.SR）：相邻两 cell 线性插值得 C_m。
  - `branch=high`（SR_sv > max T.SR）：**clamp 到 argmax-SR cell**。此 replicate 内 SR(SV) ≥ SR(comparator) 自动成立，且 clamp 不外推；SV 仍须过成本门才计胜，方向保守。
  - `branch=low`（SR_sv < min T.SR）：**clamp 到 argmin-SR cell**，禁外推。
  - 三分支统一：令 `SR_m = clip(SR_sv, min T.SR, max T.SR)`，`D_sr = SR_sv − SR_m`，`D_c = (C_sv − C_m)/C_m`。bracket 的 `D_sr=0`，high 为正，low 为负；任何分母 ≤0 或非有限均使 analyzer fail-fast。**所有 replicate 都产 D_c，禁止丢弃 low 后对条件分布算成本门。**全样本点估计同规则，仅作图。
- **Gate 1（确认性，SV vs T-frontier）**，两门全过才胜：
  1. `D_sr` 的 5% 下分位 ≥ 0（bootstrap 质量至少 95% 不落到 threshold polyline 下沿以下）；
  2. **全部 replicates** 的 `D_c` 95% 上分位 ≤ **−5%**。

  原第 3 条延迟门**已删除**：GPU inference ratio 口径下检索 CPU 不进成本，主预检没有与之匹配、且不受多 worker 竞争污染的确认性测量物。不胜 → 此线降级，一切后续对比降描述性，负结果反哺纪要 §7.2。
- **Gate 2（确认性，仅 Gate 1 胜后；fixed-sequence 至此为止）**：SV vs S0 直接配对（同 δ\*，单点对单点）。令

  `ΔSR = SR_sv − SR_s0`，`ΔC = C_sv / C_s0 − 1`。

  Gate 2 通过 **当且仅当同一套配对 bootstrap draws 同时满足**：

  1. `q_0.05(ΔSR) > 0`
  2. `q_0.95(ΔC) ≤ +0.05`

  这是方向明确的 **intersection-union test**：论文主张要求"SR 严格提升"与"成本增幅不超过 5%"同时成立，联合原假设下只要一个分量原假设为真联合主张即不成立，故两个单侧分量**各用 α=0.05，不作 Bonferroni 修正**。

  > **成本条件形式的裁定依据**：原写法「97.5% 上界 ≤ +5%」在 v 无害时只有约 0.64 的放行概率（把守卫的误伤成本记在主张头上）；而「点估计 ≤ +5%」在真实增幅**恰为 +5% 的非劣界值**处误放率为 0.500，无一类错误控制。采用的单侧 95% 上界在该边界处恰为 α=0.05，v 无害时放行概率约 **0.752**（历史方差模型下的 **design-stage sensitivity estimate**，非保证功效——它用历史 threshold 臂对估计未来嵌套 SV/S0 对比的方差，两者的逐 init verdict 不一致率与相关结构未必相同）。
  >
  > **SR 条件的分位已冻结为单侧 `q_0.05`**（owner 裁决 2026-08-27）：主张有明确方向、Gate 2 是 IUT，两个分量都用单侧 α=0.05；`q_0.025` 只是额外保守、白损功效，无统计必要。**不得**把 `q_0.025` 与「单侧 95%」混称。

  胜 → surface 胜且 v 增值获证，SV 并入 Arm2.5。
- **Gate 2 不胜 → 确认性检验到此停止**（fixed-sequence gatekeeping 在未拒绝处停止方可免校正）。结果判读固定为：
  - 两条件均过 → "`v` 带来成功率增益，且解析 GPU inference cost 的增幅以单侧 95% 置信不超过 5%"；
  - SR 过、成本未过 → "确认了 SR 增益，但成本非劣性证据不足"，**不得**据此判定 `v` 无效；
  - SR 未过 → "`v` 的独立成功率收益未获确认"。

  s-only 是否作为部署版本并入改由**描述性比较**决定（点估计 + 全 CI 展示，明确标注非确认性）。原 Gate 3 的确认性地位删除。
- CI 触界一律按"不胜"处理。SV−/SV+ 与 matched-cost 增 SR 方向只画图与描述，不进任何 Gate。
- **`analyze_precheck.py` 的 fail-closed 纪律**（代码约束，非建议）：
  - 拒绝未冻结 primary；拒绝臂间 init/seed digest 不一致；
  - 主分析输入必须**精确覆盖**预注册的 `arm × task × init` 网格，每格恰有一个 accepted episode；**缺格、重复格或身份不一致均 fail closed**；
  - `per_step` 的 decision 数必须与 episode 的 inference 数一致；`hit_type` 只能是 `FULL_HIT`/`WARM_START`/`MISS`，**缺失、未知或 `UNPROBED` 不得静默按 MISS 计费，也不得静默丢弃 episode**；
  - 所有 WARM_START 行验证 `start_t=0.3`；
  - 单位成本台账摘要、输入 `per_step.jsonl` 摘要、臂配置摘要必须进入分析 manifest；
  - SR-matched 插值与两端 clamp 在 replicate 内执行；禁止删除 low replicate；Gate 顺序与停止规则硬编码；多点择优路径不存在。

### 4.7 Arm2.5 并入挂点

胜出时四臂链消费本 plan 交付物：`dispatch_surface` yaml 形态 + primary artifact + `analyze_precheck` 口径。`docs/iclr/dispatch_defense_plan.md` §5 与 `actioncache_response_plan.md` §5 各加一行状态引用。

## 5. 文件清单

**新建 `src/`**：`src/openpi/cache/components/surface_judge.py`；`src/openpi/serving/policy_identity.py`（checkpoint URI 物化 + resolved checkpoint 内容摘要，共享给 server/离线脚本；不依赖 `scripts/` 类型）。
**修改 `src/`**：`src/openpi/cache/config.py`（_JUDGE_TYPES、JudgeConfig 字段、工厂分支、yaml 级校验、装配点库级契约比对）；`src/openpi/cache/components/judge.py`（facade re-export 一行）；`src/openpi/cache/backends/in_memory_backend.py` + `src/openpi/cache/cache_storage.py`（**identity seam additive 扩展**：`load_artifact` 计算文件 sha256 + 全库扫描聚合 metadata（记录不断言），facade 透传；对既有 yaml 与 legacy artifact 字节等价）；`scripts/serve_policy.py`（**policy attestation seam additive**：用同一 resolved spec 加载并将 `policy_fingerprint`/`monitor_level` 合入 websocket metadata）。`src/openpi/serving/websocket_policy_server.py` 无需改：既有 arbitrary metadata handshake 已满足透传。
**新建 `exp/dispatch_surface/`**：`__init__.py`、`split_init_pools.py`（super-pool 固定 seed 分层划分 A′/C + `assert_disjoint` + D_lib init census）、`rebuild_dispatch_library.py`、`collect_query_cohort.py`（C 池 `--collect` 采集驱动）、`noise_sensitivity.py`、`build_dispatch_table.py`、`fit_surface.py`（含 `--s-only` 与机械 δ 选择）、`emit_precheck_yamls.py`、`run_precheck.py`、`analysis/analyze_precheck.py`、`config/`。`run_cost_bench.py` 与 `power_sim_cost_blocks.py` 随成本轴变更删除（2026-08-27）；`analysis/` 另有三个诊断脚本 `block_variance_probe.py` / `analytic_cost_power.py` / `verify_cost_gate_rules.py`，留存该变更的证据。
**新建 `tests/`**：`tests/cache/components/test_surface_judge.py`；`tests/cache/test_surface_binding.py`（集成契约）；`tests/serving/test_policy_identity.py`；`tests/dispatch_surface/test_dispatch_split_pools.py`、`test_dispatch_table.py`、`test_fit_surface_solver.py`、`test_precheck_emit.py`、`test_precheck_analyzer.py`。
**修改 `tests/`**：`tests/cache/test_config.py`（dispatch_surface 校验用例）；`tests/serving/test_websocket_policy_server.py`（attestation metadata handshake 非回归）。
**文档**：`docs/architecture/cache_system.md` §5.6 judge 表 + `docs/cache/tutorial.md` §6 judge 表各加一行；`docs/README.md`、`logs/README.md` 索引同步（宪法红线，同 commit）。

不触碰：orchestrator、interceptor、storage_types、search strategy（surface 经现有 judge 缝插入；`CachePayload` 不加字段）。backend/cache_storage 仅上述 additive identity 字段，无行为变更。

## 6. 集成点

1. Judge 缝：`SimilarityJudge` Protocol + `view` kwarg（纯读契约）。
2. 广度：`min_required_top_k` → `min_top_k_hint`（config.py:2856 → 3663），老 yaml 字节等价。
3. 契约绑定：yaml 级在 `validate_cache_config`；库级在 `build_per_connection_components`（经 identity 扩展后的 `CacheStorage.artifact_meta`；sha 仅 load-time 算一次，BackendPool 每 fingerprint 单次加载）。
4. Warm 执行：现有 `run_stage3_from` + orchestrator 降级守卫（orchestrator.py:689），零改动。
5. 观测：`export_factor_outputs` → `JudgeResult.factor_outputs` → 现有 `__hit_meta__` 通路，零 wire 改动。
6. 标定重放：`_build_fake_stage1_with_masks` + staged API；RNG 经 `sample_noise(generator=)` seam。
7. 闭环：conductor 栈原样。

## 7. 执行编排（G2 + Verify 通过之后）

代码一次交付，实验按预注册时序执行，无回炉 G1：

0. `split_init_pools` → init map/digest attestation → 扣除权威记录的 D_lib 5 init/task → A′/C 按 30/15 物化 + 三方 disjoint 断言 + digest 落 records。
1. `rebuild_dispatch_library` → 重建库 + noise sidecar + sha 记录。
2. `collect_query_cohort` → C 池 150 ep teacher rollout H5。
3. `noise_sensitivity` → TT/WT 诊断报告（不驱动口径）。
4. `build_dispatch_table --ref-mode fresh`（+ fit split 上 uncoupled/tau1 sensitivity 列）→ 标定表。
5. `fit_surface`（surface + `--s-only`）→ artifacts + (A1)/MLR 诊断 + **δ\* 机械冻结（§4.5 规则）**。**止损点 A**：(A1) 违反率 > 20%、或 §4.5 选择规则的 `accepted_step_accuracy` 门全候选失败 → 记录负结果，跳过闭环。
6. `emit_precheck_yamls`（5–7 臂动态矩阵）→ `run_precheck`（各臂 × 300 ep，配对；SR 与解析成本共用这一批 episode）→ `analyze_precheck`（单一联合 replicate，同 replicate 内同时算 SR 与成本）→ §4.6 门控结论。**止损点 B 收窄为 Gate 1 不胜** —— 原「功效不足」分支随 power sim 一并删除（成本不再是独立测量，无 R 可选）。
7. 并入方向按 Gate 结果执行（§4.6）；负 → 负结果归档 + 纪要 §7.2 反哺。E5 第三项事后描述性分析随 analyzer 产出。

## 8. 测试策略

**单元**：
- SurfaceJudge：三档边界取值、fail-closed 全枚举（空 results / view None / fetch 异常 / K_eff<2 / v NaN / v inf / v 超右边界 均 MISS）、v 低于左边界用左 bin、B_v=1 artifact 路径、CP3 防御 MISS、`min_required_top_k` 暴露、artifact 校验全分支（嵌套破坏 / 非严格递增 edges / −inf / NaN / 非 canonical start_t / 维度错配 / 契约字段缺失均 raise）、export_factor_outputs 附着、protocol conformance。
- config：白名单、artifact 缺失 raise、warm_tiers 互斥、CP1-only、write_policy/preload 强制。
- 拟合器：单调约束、层间嵌套、conformal c 序统计量、|E|=18 → c=∞、+inf 边界导出、保守取整方向、τ=7↔start_t=0.3 sign 钉死、s-only 与 full 在 B_v=1 数据上等价。
- 标定表：mock 模型下三 ref-mode 的 Y 装配、winner 不属 query episode 断言、行 schema。

**集成**：
- 从 `build_per_connection_components` 断言 effective top_k 被抬到 artifact.k（装配链真实路径，非 mock 属性）。
- 契约漂移矩阵：对 `search_digest` 覆盖的**每一类** score-affecting 参数（field_similarity / step_filter / trajectory depth/weights / score_normalization params / fusion_weights / fields）各构造一例漂移，断言 `load_cache_config` 期 raise；库 sha / entry_count / action schema 失配在装配期 raise；identity sha 在 BackendPool shared load 下只计算一次（计数断言）。
- init 池：以真实 `libero_spatial_init_map.json` fixture 断言得出 **5/15/30** 配额与 A′/C/D_lib 三方 disjoint；provenance 异常样例（条目缺失/占用数 ≠ 5）→ 中止；digest 在 manifest 在位。
- S0 语义：mock view 断言零调用；扰动全部 payload 动作，S0 verdict 不变；`uses_disagreement=false` 时哨兵 edges 强制与 `min_required_top_k=1`。
- δ 机械性：grid 去重与退化（unique<2 → 止损）、fold 指派、OOF 残差池 n=50 单次 order statistic（**断言不存在任何 n<19 的 pseudo-conformal 分支**——n=10 冒充 α=0.05 calibration 的路径必须不可达）、空接受集语义、tie-break、accuracy 门全失败 → 止损 A，各分支合成数据测试。
- Legacy 兼容：无 intermediates / payloadless / 混合 CP artifact 经扩展后 loader 照常加载（非回归）；同一 artifact 装配 `dispatch_surface` 时被拒。
- Launch 契约与 attestation：`replan_steps != h_exec` 错配 fail-fast；替换真实 checkpoint 目录内容（增删文件、改 size、**保持 size 不变但改 bytes**）均令 `compute_policy_fingerprint` 变化；Default/显式 Checkpoint 解析到实际 resolved root；runner 从 server metadata 读取比对失败（非仅传不同 CLI 字符串）；symlink escape/特殊文件/空根拒绝；manifest 字段在位。
- 成本轴（解析）：真实 producer schema 的成功路径（`client_timing` 行**没有** `orig_init_state_idx`，只有 `task_uid`/`subset_init_state_idx`）；缺 `client_timing` 必拒；`infers` 与计价 decision 数不符必拒；stale attempt / fenced 同 attempt 报告 / 他 run 的行**排除并计数**而非计费；网格外 cell fail closed；verdict 行的 `orig_init_state_idx` 与冻结 split manifest 的 subset→official 映射交叉验证；未知 verdict 与 off-tier `start_t` 必拒；ratio-of-sums 与 episode 等权在不等长 episode 下必须给出不同结果。
- emit 产物的 preload sha 与拟合库 sha 一致性断言。
- 端到端 mini：小合成库 + surface artifact 走 orchestrator `check()`，缺 0.3 档 payload 降级 MISS、v 异常步 MISS、非有限 s → MISS。
- analyzer 纪律：未冻结 primary 拒绝、臂间 digest 不一致拒绝、launch ledger 非 v2 / 多次 launch 的**实验级冻结摘要**不一致 / run_id 缺失或重复 / 执行时 yaml 与分析时 yaml 不符 均拒绝；resume 可只执行未完成 arm，但每个 accepted episode 的 `(run_id, arm)` 必须落在该次 launch 的 `executed_arms` 内。`task_uid/yaml_id/task_id/subset_init_state_idx/episode_id/attempt/accepted/run_id` 逐层交叉核验，缺字段或不一致 fail closed。**SR-matched 插值在 replicate 内重算且不存在把待检验 compute 差按构造置零的路径**（合成数据：SV 与 threshold polyline compute 已知差 → 断言 ΔC CI 不退化为 0）；**前沿三分支各构造样例**（bracket 插值 / high clamp 到 argmax-SR / low clamp 到 argmin-SR 且 D_sr<0），断言 low replicate 仍进入 D_c 分位数；Gate 1 的 D_sr 下分位失败样例；Gate 1→2 固定顺序与 **Gate 2 不胜即停止确认性检验**（无 Gate 3 确认路径可达）；Gate 2 成本条件是单侧 95% 上界（边界处即 α=0.05）而非点估计或 p97.5；动态臂矩阵（5/6/7 臂输入均可判读，Gate 只读核心 5 臂）；多点择优路径不存在；端到端：完整 5 臂 × 10 task × 30 init 合成实验直接从 journal + per_step 判出 verdict。

**Verify**：`uv run pytest` 全量（§6 程序性运行）；仅 metadata handshake 增 additive 字段，不改 infer request/response；serving handshake 与 staged API 非回归测试显式运行。

## 9. 风险登记

| 风险 | 缓解 |
|---|---|
| episode-max conformal 过保守吞噬增益 | 两档 max + |E|=100 降方差；止损点 A 预注册；保守性正是预检要测的问题 |
| test 仅 30 init/task（300 ep/臂），配对 CI 变宽 | 配对差值推断本身大幅缩窄 CI（同 init 共差）；若 Gate 1 落入不确定带，如实按不胜处理（预注册保守方向），不追加 test 数据 |
| init map 与 H5 实际内容漂移（provenance 记录错误） | `split_init_pools.py` 交叉校验 init map 条目数与 H5 文件集一致；异常 fail-loud 中止上报 owner |
| 重建库与历史 pkl 不同（z 重采导致 intermediates 变化） | 三处强制同库 + sha 契约绑定；历史 sweep 数据仅作背景引用不作对照臂 |
| fit 50 ep（~1,050 step）拟合 2D 曲面仍偏薄 | 等频分箱 + 最小 bin 合并；B_s×B_v 可在 fit 内部降档（8×4）；cal 不受影响 |
| MISS 子集实际可捞空间低于 always-hit 平均 | 裁决只看配对差值 CI，不引用历史 94% 作承诺 |
| surface 的 judge/fetch CPU 开销 | 成本轴口径是 **GPU inference ratio**，检索 CPU 不占 GPU、不进成本；该开销以 `client_timing` 作描述性报告（注明多 worker 竞争限制），不进任何 Gate |
| 成本与 SR 口径不一致 | 两轴出自**同一批 episode、同一批 init、同一次重采样**（解析成本从主预检 `per_step` 得到），拓扑差异这一风险随独立成本实验一并消失 |
| resume 使累计 per_step 跨多次 launch | launch ledger 分离实验级冻结字段与每次运行的 `executed_arms/executed_yaml_sha256`；前者所有条目一致，后者允许未完成 arm 子集；analyzer 要求每个 accepted episode 的 `(run_id, arm)` 由对应条目认领，否则拒绝 |
| 输给双阈值或 s-only | 预注册三分结论；负结果 = "threshold 已近可达域"实证反哺纪要 §7.2；s-only 胜出仍可并入（v 维降级） |
| 吞吐波动 | 主预检动态 5–7 臂、成本固定核心 5 臂；每臂固定 A′ 300 episode，resume 过滤沿用 run_gtp 机制 |

## 10. 文档义务（同 commit 红线）

`logs/README.md`（本文件条目）；`docs/README.md` + `docs/architecture/cache_system.md` + `docs/cache/tutorial.md`（judge 表）；实验产出期：`exp/data_authority` 收编 + MANIFEST + records 台账（重建库、query cohort、预检各一）；`docs/iclr/dispatch_defense_plan.md` §5 与 `actioncache_response_plan.md` 状态行更新（连同 `docs/iclr/README.md` 描述同步）。

## 11. 执行日志

§7 时序的逐步执行记录。每步记：命令、产物、独立复核结论。**冻结判据零自由度**（§4.2 配额 / §4.5 δ 规则 / §4.6 门控）在执行期不得重议。

**拓扑（本轮 owner 指定）**：`weilandserver` 单机闭环（4090 48G，server + LIBERO client 同机，client 走 `127.0.0.1` 不经 broker），`timan107` 本轮不参与。跨节点操作走 tether，`tether exec` 必须 `export HOME=/home/weiland`（agent 默认 HOME 是 `~/.tether-agent`）。设备与端口规约见 `dist_experiment_control/docs/devices.md §2.5`。

### 2026-08-27 — 前置：标定检索配置

`exp/dispatch_surface/config/calibration_retrieval.yaml` 由 gtp 模板 `exp/gate_research/config/libero_spatial/n4_server/cp1_spatial_pool_16__grid3_vision_0@6_vision_1@50_robot_state@43__d1__fh75_ws10_quantile.yaml` **机械派生**（施加与 `emit_precheck_yamls._emit` 完全相同的三处替换：judge → `always_hit`、`preload_path` → 重建库、`write_policy` → `never`），而非手写。理由：`compute_surface_retrieval_contract` 对 `key_builder` 整段与 (`keys`, `cp1.search_strategy`) 整对取 digest 并写进 surface artifact，标定栈与预检臂只要有一处 dataclass 字段不同，臂在 load 期就会拒绝证书。

**复核**：模板 / 标定 yaml / 模拟臂 yaml 三方的 `key_builder_digest`（`77c0dfad…`）、`search_digest`（`5cceb846…`）、`top_k`（1）逐一相等。yaml 的 `top_k: 1` 与 §4 的"检索宽度 5"不矛盾——`SurfaceJudge.min_required_top_k = artifact.k` 在装配点把有效宽度抬到 5，`_check_surface_library_binding` 再断言 `effective_top_k ≥ k`。gate 段是模板原样（其阈值属 threshold-pareto 线，标定期离线回放不走 gate），已在文件注释中标明以免被误读为本线标定结果。

### 2026-08-27 — 步骤 0：`split_init_pools`（CPU，完成）

```
uv run python -m exp.dispatch_surface.split_init_pools --suite libero_spatial \
  --init-map exp/common/data/db/libero_cache/libero_spatial_init_map.json \
  --apool-dir exp/common/data/db_init/libero/libero_spatial \
  --library-h5-dir exp/common/data/db/libero_cache/libero_spatial \
  --out-root exp/dispatch_surface/data/init_pools --seed 20260827
```

官方 super-pool 目录实为 `exp/common/data/db_init/libero/libero_spatial`（每 task 一个 `.init`，各 50 states），无需 `materialize_apool.py` 现场物化。D_lib census 通过：10 task × 恰 5 官方 init。

**产物**：`test_aprime/`（30 states/task）、`query_c/`（15 = 5 fit + 10 cal）、`split_manifest.json`。21 文件 / 464,399 B / rollup `23beeafe…`。

**独立复核**（不复用脚本自身的打印）：逐 task 重算 `dlib ∪ fit ∪ cal ∪ test = {0..49}` 且两两不交；450 个物化 state 逐字节比对其 manifest 声称的官方索引，450/450 相等——只核数量会放过"计数正确但内容被置换"的划分，而那恰好会静默破坏整条线赖以成立的数据独立性。

**台账**：`exp/data_authority/records/dispatch_surface__libero_spatial__init_pools.json`（`kind: init_pool`，含 21 条逐文件明细），`registry validate` 与 `verify` 全绿。字节被 `.gitignore` 的 `exp/**/data/**` 吞掉，台账是它在 git 内的唯一痕迹。

### 2026-08-27 — 步骤 2 前置：`collect_query_cohort plan`（CPU，完成）

由 split manifest 生成 150 episode 的采集计划（10 task × 15 = 5 fit + 10 cal），产物 `exp/dispatch_surface/data/query_cohort_plan.json`。计划同时钉住官方 init 索引与 C 池文件内的 subset 位置两套索引空间，采集器据此打 metadata，下游 join 不猜。**采集本身需 GPU**（teacher rollout），待卡。

### 2026-08-27 — 成本轴：取消独立成本 bench，改为从主预检解析计算（owner 定口径）

**owner 厘清成本口径后，§4.6 的整个独立成本实验失去存在理由。** 三次修正，最后一次是结构性的：

1. **成本 = GPU inference ratio**。检索不占 GPU、不进成本、不进 Pareto 前沿。此前把检索 CPU 算进"端到端 latency"并当成一条独立确认性轴，是错的。
2. **单位成本用 CUDA-graph 档**（stage1 10.26 / stage2 27.69 / stage3 29.57 ms），不是 eager 也不是 default——后两者描述未优化的系统，会低估 FULL_HIT 相对 MISS 有多便宜，而那个比值是全部结论的支点。
3. **成本是解析量，不是测量量**：

| 档 | 成本 | CUDA-graph |
|---|---|---|
| FULL_HIT | stage1 | 10.26 ms |
| WARM_START(start_t=0.3) | stage1 + stage2 + **0.3 ×** stage3 | 46.82 ms |
| MISS | stage1 + stage2 + stage3 | 67.52 ms |

`per_step.jsonl` 逐 step 记录 `hit_type` 与 `start_t`，带完整的 (arm, task, init, episode) join key。**数一数三档各多少步，乘上单位成本，每臂的成本就出来了**——从主预检的 300 ep/臂 直接得到，与 SR 用同一批 episode、同一批 init、天然配对。§4.6 当初引入独立成本实验的理由是"现有接口无法支撑主预检内的逐 decision 配对成本"，但那是针对**计时**的；解析成本只需要 verdict 计数，而计数早已 join 好。

> **同时修掉一个实现错误**：`decision_compute_ms` 原写 `frac = 1 - start_t`，把 WARM_START 当成跑 70% 的 stage3。源码 `pi0_pytorch.py:691` 明确：`start_t=0.3, num_steps=10 → 3 steps, saves 70%`，即跑 `start_t` 那一份。已改为 `frac = start_t` 并加测试钉死。此错未污染任何已得数字——gtp 那批数据 warm_start 整条关闭，该分支从未触发。

**功效对比**（`exp/dispatch_surface/analysis/analytic_cost_power.py`，在 gtp sweep 上取 30 init/task 模拟 A′ 配额，task 分层 init-cluster bootstrap，臂对按各 Gate 实际面对的差异程度分组）：

| 口径 | 额外 episode | Gate 1 功效 | Gate 2 功效 |
|---|---|---|---|
| **解析成本，主预检 300 ep/臂** | **0** | **0.765** | **0.696** |
| 独立 block bench（block=40 ep、R=7，A′ 池上限） | 2,800 | 同等（SE 0.019 vs 0.021） | 同等 |

**独立成本 bench 花 2,800 个 episode 换来的精度，主预检免费给到几乎一样——它没有存在价值。** 一并消失的还有：block 设计、R∈{5,10,15}、power sim、双 server launch、以及此前算了半天的 A′ 池 init 配额冲突（那是 block 之间要求 init 不重复才产生的）。

**剩下的唯一问题是功效仍不足 0.80**（Gate 1 0.765 / Gate 2 0.696），且瓶颈是 `gate2_compute`（p97.5、真效应 0% 对阈值 +5%）。杠杆已经换成 `--trials`——它同时抬高 SR 轴与成本轴的精度：

- Gate 1 达 0.80 需 SE ≤ 0.0201（现 0.0211），差一点点；
- Gate 2 达 0.80 需 SE ≤ 0.0179（现 0.0202），约需 38 init/task。

A′ 现为 30 init/task（500 官方 init 已四分：D_lib 50 / fit 50 / cal 100 / A′ 300）。扩到 38 要从 cal 挪 80 个（cal 10→2 per task，|E| 100→20），逼近 §4.5 的 19 下限，代价大。

**已裁决（2026-08-27，Review Authority 有条件批准）** —— 详见 [`dispatch_surface_cost_axis_change.md`](dispatch_surface_cost_axis_change.md) §9（裁决）与 §10（Executor 接收）：

- 解析成本获批为正式 estimand，独立成本 bench 删除；确认性 latency 门删除。
- Gate 2 成本条件采用**方案 D（单侧 95% 上界 ≤ +5%）**，**否决**本日志此前推荐的方案 C。补算证实：C 在真实增幅恰为 +5% 的非劣界值处误放率为 **0.500**（无一类错误控制），D 在该边界恰为 α=0.05；此前只列 +8%/+15% 漏放率而漏掉边界点，是本日志的论证缺陷。
- estimand 须为 **decision-weighted ratio-of-sums**，非 episode 等权平均（实测 episode 决策数 14–44 不等）。据此修正，方案 D 在 v 无害时的放行概率由本日志此前报的 0.797 **更正为 0.752**，并按裁决 §9.6 标注为 design-stage sensitivity estimate。
- §4.6 与 §7 已按裁决修订并冻结（本次同 commit）；analyzer 实现按 §4.6 新增的 fail-closed 纪律清单执行。
- Gate 2 的 SR 条件由 owner 于同日裁定冻结为单侧 `q_0.05`（见 §4.6）。

### 2026-08-27 — weilandserver 执行环境部署（CPU，完成）

**本机（weiland-wsl）无 GPU**（`nvidia-smi` not found），故一切需要模型的步骤都在 weilandserver 跑，本机只做纯 CPU 分析与文档。

weilandserver 上原有三个 openpi 克隆，**没有一个带本线代码**：`/home/weiland/openpi` 停在 `6818ff2` 且有 67 个未提交改动（其他会话的工作，且是 ws2 实验克隆的源，**不可动**）；`/data/openpi` 非 git；`/data/openpi_text_ivf_build` 是 ws2 正在用的 detached 克隆。核对确认 `6818ff2` **是** `origin/Ziyang` 的祖先——只是落后 21 个提交，无分叉。

因此按项目既有模式（ws2 即用独立克隆）新建隔离环境，不触碰任何现有工作区：

- 代码：`/data/openpi_dispatch`，`git clone --shared /home/weiland/openpi` 后 checkout `cdb128d`，dirty=0，106 MB（objects 共享）。
- 数据：软链而非复制——`exp/common/data/db/libero_cache/libero_spatial` 与 `exp/common/data/db_init` 指向 `/home/weiland/openpi/exp/common/data/` 下的真身（3.5 G H5 不复制）。注意 `exp/common/data` 本身**不能**整体软链：克隆里该路径下有 534 个 tracked 文件。
- 解释器：复用 `/home/weiland/openpi/.venv/bin/python`（3.11.15）+ `PYTHONPATH=/data/openpi_dispatch/src:/data/openpi_dispatch`，与 ws2 同一模式。

**端到端验证（不是"看起来对"，是 digest 逐字节相等）**：远端重跑 `split_init_pools`（同参数同 seed）与 `collect_query_cohort plan`，产物与本机比对——`split_manifest.json` 两边同为 `964b5b71…`，`query_cohort_plan.json` 两边同为 `79ed09ae…`。这一次同时验证了三件事：远端代码与依赖可用、两边库源数据同源（`init_map` `ac0c88c3…`、首个 H5 `7636bba1…` 各自在两边相等）、划分确定性。因此 init pools 不需要传输，远端就地重建即可。

**checkpoint 同一性（handoff 关卡）**：两边 `compute_policy_fingerprint` **不同** —— 本机 `81ac1961…`、weilandserver `95098c85…`。查明原因：`model.safetensors` 两边 sha256 完全相同（`69960c7b…`，7,233,650,408 B），`config.json` 也同；差异**全部**来自本机目录里多出的 `.ipynb_checkpoints/config-checkpoint.json`（149 B，Jupyter 残留）。即权重同一，指纹分歧纯属本机的目录卫生。由于模型步骤全在 weilandserver 上跑，本线一律以 **`95098c85…`** 为准，内部自洽。

> 留给后来者的坑：`compute_policy_fingerprint` 计入 checkpoint 根下**每一个**常规文件，所以编辑器备份、`.DS_Store`、Jupyter checkpoint 都会改变 attestation，并让"同一个模型"在两台机器上互相拒绝。fail-closed 是设计意图，但排查时先看文件清单差异再怀疑权重。

### 2026-08-28 — 范围扩展：两 suite 并进，l10 优先（owner 指定）

owner 决定本线覆盖 **libero_10 与 libero_spatial 两个 suite**：前置链（采集 / 重建库 / 标定表 / 拟合）两线并进，**但最终大实验（`run_precheck` 及其后）必须先跑 l10、再跑 spatial**。即便 spatial 的 fit 先就绪也不得抢跑——排序是 owner 的调度决定，不是资源可用性问题。

§4.6 原文写"预检（闭环，1 suite libero_spatial）"，现扩为两 suite，其余冻结判据（配额 5/5/10/30、δ 机械规则、Gate 判据与分位、单位成本表）**逐字不变，两 suite 各自独立执行一遍**。

**l10 的检索配置是它自己的**：`exp/dispatch_surface/config/calibration_retrieval_libero_10.yaml`，从 `exp/gate_research/config/libero_10/eval/` 的模板机械派生（四个模板的检索段实测同构，任选其一不影响契约）。它与 spatial 的 `key_builder_digest` 相同（同一 `cp1_spatial_pool_16`），但 `search_digest` 不同（l10 的融合权重 56/25/18 对 spatial 的 6/50/43）——因此一个 suite 的 surface artifact 无法被另一个 suite 的臂加载，这是应有的行为。

**l10 的 D_lib 必须重采**（owner 选定方案 B）。原因：l10 的库源过不了 §4.2 要求的内容级 census——它的 init map 只有 5 个字段（缺 `h5_path`/`attrs`/`full_init_path`/`init_path`/`entry_count`/`trajectory_id`），且库源 H5（2026-04-21 采）的 attrs 不含任何 init 身份，把 map 行关联到具体 H5 只能靠推断，而推断正是 G2R3-B1 要拒绝的。新增 `exp/dispatch_surface/bootstrap_library_source.py` 打破这个死锁：从既有 map 取它自己记录的 D_lib official init 选择（每 task 5 个，保持历史一致）→ 物化 5/task 池 → 生成 cohort-plan 格式的采集计划；采完用 `rebuild-map` 从 H5 **反向**构建完整 init map 并当场跑 `census_dlib_inits` 自检。**该脚本为本轮新增、尚未过 G2**，只做数据准备、不碰任何冻结判据。

> **备查资产（本次不用，但存在）**：`/data/openpi/ablation_study/cache_size/collect_h5/{libero_10,libero_spatial}/task_N/episode_M.h5` —— 每 suite **完整 500 init**（10×50 格，`verify_collect.py` 契约化并验证 grid 完整性），132 GB，纯 teacher（无 cache，见 `ablation_study_plan.log.md:69`），2026-08-16 采。本线未采用，因为它的 H5 attrs 只有 `episode_id`、身份编码在路径里，喂给 `build_dispatch_table` 需要改 loader 支持路径式身份（新代码 + 两 suite provenance 形式不对称），而 owner 选择了对称性更强的重采方案。**它是目前唯一覆盖全部 official init 的纯 teacher 语料**，任何需要"每个 init 都有 H5"的后续工作（换 estimand、init 级分析）应先想到它。另注：`libero_10_init_map.json` **未被 git 跟踪**（只有 spatial 那份是），隔离克隆里没有、需手推。两者都尚无 data_authority 台账，待补。

### 待执行（需 GPU）

GPU 当前被 robocasa365 ws_search2 占用（探测时 11 个 `serve_policy`，49 G 中仅 ~9 G 空闲，util 66–80%）。已挂后台监控，`serve_policy` 归零且空闲 >30 G 时通知。

环境前缀（下列命令都在此之下）：

```bash
tether exec weilandserver -- bash -lc 'export HOME=/home/weiland
  cd /data/openpi_dispatch
  export PYTHONPATH=/data/openpi_dispatch/src:/data/openpi_dispatch
  PY=/home/weiland/openpi/.venv/bin/python
  ...'
```

**§7 的步骤 3/4 编号与实现相反，以下为按实际输入依赖核对（逐脚本 `--help`）后的正确顺序**：§7 把 `noise_sensitivity` 排在 `build_dispatch_table` 之前，但前者的 `--table`（fresh 标定表）与 `--weights-npz`（W/active_mask）**都是后者的产物**且均为必填。代码是对的——诊断从已算好的标定表分层抽 50 step 才有意义；§7 的编号是笔误。`noise_sensitivity` 是"preliminary diagnostic，不驱动口径"，其位置不影响任何裁决，故按依赖顺序执行，此处备案。

| # | 步骤 | 关键输入（来源） | 预算 |
|---|---|---|---|
| 1 | `rebuild_dispatch_library` | `--split-manifest`(步 0)、`--checkpoint-dir /home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch`、`--device cuda` | ~10 min |
| 2 | `collect_query_cohort launch` → `verify` | 步 0 的 `query_c` 池；server 侧 `serve_policy.py --collect`（纯 teacher），client 侧本机 `libero_sim` conda prefix、`--num-workers 1` | 150 ep，数小时 |
| 3 | `build_dispatch_table --ref-mode fresh --top-k 5 --h-exec 5` | `--library-pkl`/`--noise-sidecar`(步 1)、`--query-h5-dir`(步 2)、`--cache-yaml`(标定 yaml)、`--split-manifest` | ~1–2 h |
| 4 | `noise_sensitivity` | `--table`/`--weights-npz`(步 3) —— **依赖步 3** | 诊断 |
| 5 | `fit_surface`（再 `--s-only --frozen-record`） | `--table`/`--weights-npz`(步 3)、`--cohort-manifest`(步 2 verify)、`--rebuild-record`(步 1)、`--cache-yaml`、`--split-manifest` | CPU |
| 6 | `emit_precheck_yamls` → `run_precheck --trials 30 --replan-steps 5` → `analyze_precheck` | 步 5 的 artifacts + δ\*；A′ 池 | 5–7 臂 × 300 ep ≈ 1 h |

成本轴不再有独立步骤：`run_precheck` 的 `per_step.jsonl` 同时供给 SR 与解析成本，`analyze_precheck` 在同一 bootstrap replicate 内算两者。

GPU 空出后的第一条命令（步 1，已核对参数名与路径）：

```bash
$PY -m exp.dispatch_surface.rebuild_dispatch_library \
  --h5-dir exp/common/data/db/libero_cache/libero_spatial \
  --split-manifest exp/dispatch_surface/data/init_pools/split_manifest.json \
  --builder cp1_spatial_pool_16 --config-name pi05_libero \
  --checkpoint-dir /home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  --out-dir exp/dispatch_surface/data/cache_artifacts --device cuda
```

步 2 的两侧命令（由 `collect_query_cohort launch` 打印，`--run` 才执行；client 侧已实测可跑——`libero_sim` env 能解析新 `main.py` 的 `--cohort-plan`）：

```bash
# server（tmux srvN，tee 落盘保留 stdout）
uv run scripts/serve_policy.py --collect --collect_dir <out> --env LIBERO \
  policy:checkpoint --policy.config pi05_libero \
  --policy.dir /home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch
# client（必须 conda run -p 才触发 EGL activate 钩子；直接调 libero_sim/bin/python 不生效）
MUJOCO_EGL_DEVICE_ID=0 PYTHONPATH=. /home/weiland/miniconda3/bin/conda run -p /home/weiland/libero_sim \
  python examples/libero/main.py --host 127.0.0.1 --port 8000 \
  --task-suite-name libero_spatial --num-trials-per-task 15 --num-workers 1 \
  --init-states-dir exp/dispatch_surface/data/init_pools/query_c \
  --cohort-plan exp/dispatch_surface/data/query_cohort_plan.json
```

`--num-workers 1` 是硬约束（cohort 身份 metadata 只在串行路径写入，main.py 有断言），不是吞吐选择。

**独占 GPU 的要求随成本 bench 一并消失**：成本是解析量，不测墙钟，因此 `run_precheck` 与其他 GPU 步骤一样只需要卡可用、不需要独占（共卡只影响吞吐，不影响 verdict 计数）。`fit_surface` / `emit_precheck_yamls` / `analyze_precheck` 是纯 CPU，但依赖上游 GPU 产物。

### 2026-08-28 — GPU 阶段执行日志（两线并进）

GPU 于 2026-08-28 上午空出（owner 通知「机器已空出」），前置链开跑。以下为**实际发生**的时序与产物摘要；上一节「待执行（需 GPU）」的命令表仍是有效的依赖参考，但其"待执行"状态已被本节取代。

| 产物 | libero_spatial | libero_10 |
|---|---|---|
| split manifest | `964b5b71…` | `1948ab3b…` |
| 重建库 entry_count | 1018 | 2496 |
| library_sha256 | `b3f61dc5…` | `7315f4b1…` |
| noise sidecar | `8889e232…` | `d0543efe…` |
| 九档 intermediates completeness | 100% | 100% |

> entry_count 相差 2.45× 是 suite 性质而非配额差异：两边都是 50 个 D_lib episode，libero_10 是长程套件，单 episode 步数远多于 spatial。

`libero_10` 的重建 record 里 `builder = cp1_spatial_pool_16`，与 spatial 同名——这是 key_builder 的 id，两 suite 的 `key_builder_digest` 本就相同（同一 `cp1_spatial_pool_16`，无差异旋钮），库条目因此是同构的；两边真正不同的是检索融合权重（`search_digest`），而它不参与建库。pkl 文件名里的 `spatial` 字样同理只是 builder id 的回显，不表示用错了语料——两个 pkl 在各自 suite 的目录下，`preload_path` 分别指向自己那份。

**踩坑并修复：标定表的检索宽度不由 `--top-k` 决定。** 首版 spatial 标定表（3364 行）跑完后抽检发现 **`v` 100% 为 None、`k_eff` 全部为 1**，即 F2 分歧特征完全缺失。根因：检索宽度取 `effective_top_k = max(yaml 的 search_strategy.top_k, judge 的 min_required_top_k hint)`。部署期由 surface judge 给出 hint（`surface_judge.py:402`，`uses_disagreement` 时 hint = `artifact.k`）把宽度抬到 5；但**标定期 judge 是 `always_hit`、不给 hint**，宽度就停在模板的 `top_k: 1`，而 `build_dispatch_table --top-k 5` 只是在返回的 1 个结果上切片，于是恒有 k_eff=1、v=NaN→None。

修法（`build_dispatch_table._load_components` 新增 `min_top_k` 参数）：装配 components 前把**内存中** config 的 `search_strategy.top_k` 抬到 `--top-k`。**刻意不改盘上的 yaml**——`fit_surface` 重新读盘算 `compute_surface_retrieval_contract`，记录的 `search_digest` 仍来自 `top_k: 1` 的原始 yaml，与 `emit_precheck_yamls` 产出的臂 yaml 逐字一致（yaml 级契约只比 `key_builder_digest` / `search_digest` 两项，`top_k` 不参与比对）；部署时照旧由 judge hint 抬宽，并由 `config.py:2948` 的 `effective_top_k < artifact.k` 断言兜底。即：这是**复现部署语义**，不是绕过契约。

> **教训（本线最贵的一条）**：这个 bug **不报错**，只安静产出一张 v 全空的表，退出码 0、行数正确。若不抽检就会一路跑到 `fit_surface` 的 (s,v) 分箱才炸，那时两条线的标定表都已作废、GPU 时间全部白烧。**新增纪律：每个阶段完成后必须验产物的取值分布**（None 率 + 取值集合 + 分层计数），不能只看行数和退出码。已把该检查固化为 `next_*_fit.sh` 里的 GUARD 段（v_none / s_none / k_eff / episodes 四项，任一不符即拒绝进入拟合）。

首版表改名 `dispatch_table_fresh.VOID_k1.jsonl` 留证，**不得用于任何拟合**；spatial 以修复后的代码重跑完毕（3364 行 / 10 tasks / 150 episodes = fit 50 + cal 100），验收：`v_none=0`、`s_none=0`、`k_eff` 全为 5、`y_tau7`/`y_tau10` 无缺、`ref_mode=fresh`、`v ∈ [9.22, 40.58]`、`s ∈ [0.6763, 0.9885]`。

**把作废表当对照组用了一次，得到一个有用的结论**：新表与 VOID 表按 `(episode_id, step_idx)` 全键对齐（3364/3364），逐行比较得
`max|s_new − s_void| = 0`、`winner_id` 3364/3364 相同、`max|y_tau10 差| = 0`。即**检索宽度只影响 `v`，不影响 `s`／winner／Y**。两个推论：

1. 宽度修复是外科手术式的——它没有扰动任何别的列，VOID 表并不是"算错了 s"，只是缺 v。
2. **s-only 曲面在部署期的窄宽度下是有效的**。`surface_judge.py:402` 在 `uses_disagreement=false` 时把 `min_required_top_k` 设为 1，于是 S0 臂部署时宽度停在 yaml 的 1；此前这一点只有"top-1 winner 与 k 无关"的论证，现在有了全表逐行的实测证据。若将来有人改动检索使 winner 依赖 k，这条不变量会失效，S0 臂的契约需重新论证。

**`noise_sensitivity` 延后执行（备案）。** 它是 §7 明写的 "preliminary diagnostic，不驱动口径"（脚本 docstring：`the calibration ref-mode is fixed to fresh regardless of this outcome`），且**没有任何下游消费它的产物**——`fit_surface` 的必填输入里不含它。它需要 GPU（重载 pi05，50 step × 8 samples），而当前 GPU 的关键路径是 l10 的 cohort 采集（owner 指定 l10 优先）。因此把两 suite 的 `noise_sensitivity` 排到 GPU 空闲窗口执行，`fit_surface` 不等它。**这不是判据变更**：诊断的位置与结论都不进入任何 Gate。

### 2026-08-28 — **libero_spatial 触发止损点 A：`stop_loss_zero_hitshare`**

`fit_surface`（SV）以退出码 3 停机，`exp/dispatch_surface/data/surface_fit/fit_record.json` 记 `delta_selection_reason = stop_loss_zero_hitshare`：**δ 网格里每一个候选的 `hitshare` 都是 0.0**，即曲面在任何 δ 下都不接受任何一步（`accepted_step_accuracy=1.0` 是 0/0 的平凡值，不代表精度好）。

标定表本身完全健康（3364 行 / 10 tasks / 150 ep，v/s/y 无缺，`k_eff` 全 5），所以这不是数据问题。

**这次停机是算术上被强制的，与曲面拟合得好不好无关。** 逐项：

| 量 | 值 |
|---|---|
| δ 网格 = `y10[fit]` 的 p10…p90 | `[3.633, 6.187]`（9 档） |
| OOF safety offset（`order_statistic_offset`, n=50, α=0.05 → 第 49/50 个序统计量） | **5.373** |
| 于是接受需要 `q̂ ≤ δ − offset ≤` | **0.814** |
| 但 `y10` 的**全局最小值** | **2.655**（`y7` 最小 1.884） |
| `p90(y10) − p10(y10)` | **2.554 < 5.373** |

`evaluate_candidate_deployed` 用 `m.q + offset` 导出边界，而 δ 上限只到 p90(y10)。既然 `p90 − p10 < offset`，**网格里不存在任何 δ 能让 `q̂ + offset ≤ δ` 成立**——哪怕 q̂ 是完美的条件分位数（其下界仍 ≥ y 的最小值 2.655），也过不去。换言之：**只要偏差分布的"地板"非零且尾巴够重，预注册的 δ 网格（p10…p90）与 OOF offset（episode-max 的 95% 序统计量）就处在不相容的尺度上，止损必然触发。**

**信号是存在的，不能把这读成"曲面没有预测力"**：

- Spearman `y10 ~ s` = **−0.328**、`y10 ~ v` = **+0.304**；`y7` 侧为 −0.301 / +0.304。**两个轴的符号都与设计假设一致**（检索分越高偏差越小、top-k 分歧越大偏差越大），(A1) 双单调假设在数据上站得住。
- 按 s 十分位看中位 `y10` 单调下行：d0 `5.35` → d9 `3.97`；p95 同向 `7.79` → `5.38`。

问题在**动态范围**：最好的 s 十分位中位数仍有 3.97，而 offset 是 5.373。信号有方向，但幅度远小于安全边际。

**offset 为什么这么大**：它是「每 episode 内 `max(y7−q̂7, y10−q̂10)` 的最大值」再取 50 个 episode 的第 49 位，本质是**尾部的尾部**。fit split 1134 行里只有 **9 行**（0.8%，分布在 50 个 episode 中的 6 个）`y7 > 8`，其中一个 episode 的 `y7` 达到 **18.34**（而 `y7` 的 p99 才 7.50）。也就是说这个门槛实际上是被 6 个 episode 的少数几步定下来的。

**一处值得评审注意的不对称**（陈述，不作判断）：δ 选择用的是 OOF offset **5.373**，而最终 artifact 导出用的是 `q_tilde = q̂ + c`，`c` 是 cal split 上的正式 split-conformal 修正——两者是不同的量。选择阶段的保守量结构上大于它所代表的部署阶段修正，于是**在部署侧未必不可行的 δ，在选择阶段就被判死**。`fit_surface` 的 docstring 自己写明该 offset "order statistic; not a conformal certificate"，所以这大概率是有意的保守，但它在本数据上的后果是决定性的。

**处置**：δ 的机械选择规则与止损点 A 是冻结判据，**执行期不得修改**。故按 §7 记负结果、spatial 跳过闭环。**是否要请评审重新审视预注册规则（例如 δ 网格上界或 offset 的定义），是 owner／Review Authority 的决定，不是执行方的**。

**l10 不受影响，继续跑**：两 suite 各自独立执行，l10 的 cohort 仍在采集，其 `fit_surface` 照常进行。若 l10 也落在 `p90(y10) − p10(y10) < offset` 这个条件内，则会得到同一结论的第二次独立确认；若不在，则说明该条件是 suite 相关的。**建表后、拟合前值得先算这一个不等式**——它一行就能预判结局。

**`noise_sensitivity` 的优先级因此提高并已开跑**（spatial，tmux `dsp2`）。它测的是「同一观测下 teacher 重复采样之间的偏差」。若该 teacher–teacher 偏差本身就在 2.6–4 量级，那么 `y` 的地板就**不是检索误差、而是策略自身的采样噪声**——那将直接解释为什么 `y` 永远够不到 0，也就解释了本次止损。它仍然**不驱动任何口径、不进任何 Gate**，只作解释性证据。

### 2026-08-28 — `noise_sensitivity`（spatial）：止损点 A 的解释——**teacher 自噪声支配了偏差度量**

诊断跑完（50 步分层抽样 × 8 个独立 teacher 采样，`h_exec=5`，与标定表同一 `W/active_mask`）：

| 量 | 中位 | p95 |
|---|---|---|
| **tt**（同一观测下两次独立 teacher 采样之间的偏差） | **6.222** | 8.380 |
| **wt**（检索 winner 的 τ=7 warm 补全 vs 一次 teacher 采样） | **3.181** | 6.406 |
| `ratio_median` = tt/wt | **1.956** | — |

**先看一致性检查（这条让上面的对比可信）**：诊断的 `wt` 中位 **3.181** 与标定表 fit split 的 `y7` 中位 **3.31** 几乎相等（两者由不同代码路径、不同抽样得到，n 分别是 50 与 1134）。两条流水线在测同一个量，尺度可比，所以下面的 tt 对比是同量纲的。

**结论：**

1. **teacher 自己跟自己的分歧（6.222）比 cache 跟 teacher 的分歧（`y7` 3.31 / `y10` 4.55）还大**，`ratio_median ≈ 1.96` ——**warm-start 的 cache 输出与一次 teacher 采样的接近程度，是两次 teacher 采样彼此接近程度的约两倍**。缓存不是偏差的来源。
2. **δ 网格的上界 6.187 低于 tt 的中位 6.222。** 也就是说：**一个"完美"的 cache——输出恰好等于某次 teacher 采样——其偏差期望值仍约等于 tt ≈ 6.22，超过网格里的每一个 δ 候选。**
3. 于是 `y` 的地板（`y10` 全局最小 2.655、最好 s 十分位中位 3.97）**不是检索能力的上限，而是策略自身采样噪声的下限**。

**这把上一条的负结果重新定性了。** 它不是「dispatch surface 没有预测力」——(A1) 双单调在数据上成立，Spearman 符号也对（`y10~s` −0.328、`y10~v` +0.304）。真正的问题是**偏差度量 `Y` 分不开「cache 误差」与「teacher 采样噪声」，而后者占主导**；任何写成「保证 `Y ≤ δ`」且 δ 由 `Y` 自身分位数标定的接受规则，在这种度量下**构造上不可达**。

**必须同时说清的边界**：

- 这是**预注册的 preliminary diagnostic**，脚本自己写明 "does not drive the pipeline / the calibration ref-mode is fixed to fresh regardless of this outcome"。它**不改变、也不得改变任何冻结判据或 Gate**，只是解释证据。
- n=50 步、m=8 采样，样本很小，`tt_median` 有实打实的不确定性；上面的 6.222 vs 6.187 是"同一量级、互相跨过"，不宜当作精确的临界判定，但 `ratio ≈ 2` 这个量级结论是稳健的。
- tt 用的是 pair 间偏差的中位数聚合，与 `y` 的定义（对 teacher 单次采样比）不完全同构；`wt ≈ y7` 的一致性是支持可比性的主要证据，但不是证明。

**对设计的含义（陈述给评审，执行方不作决定）**：若要让「偏差容忍」式的曲面在这类策略上可用，`Y` 需要能把 cache 误差从策略随机性里分离出来——例如以**teacher 采样分布**（而非单次采样）为参照、或用噪声配对的参考轨迹。这属于预注册规则的重新设计，超出执行期权限。

> ### ⚠ 2026-08-28 追加：`ratio = 1.956` 可能超出理论上界，**在核实前不要引用这个数字**
>
> 事后推导：设 teacher 采样 `X ~ (μ, σ²I)`，维度 d。则 `E‖X₁−X₂‖² = 2σ²d`，而对**任何确定性输出 c**
> 有 `E‖c−X‖² ≥ σ²d`（等号在 `c = μ` 即条件均值取到，这是 c 的最优值）。高维下范数集中，故
> **`tt / wt ≤ √2 ≈ 1.414`**——**哪怕缓存输出恰好是最优点预测，比值也到不了 1.414 以上。**
> 实测 **1.956** 超出该上界。
>
> 尚未验证的可能解释：
> 1. **teacher 分布多模态** —— 中位数聚合下，跨模态的 `tt` pair 会抬高 `tt` 中位，而 `wt` 可能稳定落在主模态内。多模态/重尾很容易破坏 √2 关系。
> 2. 度量是「h_exec 步上加权 L2 的**均值**」，是范数的均值而非范数，上述逐步近似可能不严格适用。
> 3. `tt`/`wt` 的实现存在我尚未看出的不对称。
>
> **已排除**：不是抽样步不同步——`per_step.jsonl` 每行同时带 `tt_median` 与 `wt`，是同一步。
>
> **可查的方法**（不占 GPU、不动任何判据）：`noise_sensitivity` 每步有 M=8 个采样即 28 个 pair，
> 直接看 pair 级距离的分布形状是否双峰／重尾即可分辨解释 1。
>
> **影响面**：本条目的「cache 比 teacher 自己更接近 teacher」是执行方报给 owner 的头条数字。
> 若上界成立而解释 1 不成立，该说法需**收回或改写**。`tt ≫ wt`（定性）与 `tt` 中位跨过 δ 网格上界
> 这两点不依赖 ratio 的具体数值，暂不受影响；受影响的是「约两倍」这个定量表述。
>
> **若解释 1（多模态）成立，结论方向反而更强**：同一观测下 teacher 会跳到不同行为模式，
> 那么「降低 teacher 随机性」就不是修补而是对症——见 open-questions 文档的出路 A/C。>
> #### 2026-08-28 再追加：读了 `noise_sensitivity` 源码后的三项更正
>
> **更正 1（重要）——「`wt ≈ y7` 是独立一致性佐证」这句话是错的。** 源码 `noise_sensitivity.py:112`
> 是 `wt_devs.append(row["y_tau7"])`：**`wt` 直接取自标定表的 `y_tau7` 列，不是独立算的。**
> 因此 "wt 中位 3.181 ≈ y7 中位 3.31" **近乎恒真**（差别只来自 50 步子采样 vs 1134 行），
> 我此前把它当作「不同代码路径、不同抽样得到同一量」的可比性证据，**这个论证不成立**。
> 受影响的是本条目里 `tt` 与 `y` 尺度可比的主要依据。
>
> **更正 2——`tt` 与 `wt` 不是同一种比较。** `tt` 是**两次完整 stage3 采样**（从纯噪声跑满 10 步）之间的偏差；
> `wt` = `y_tau7` 是**从检索 winner 的中间态做 τ=7 warm 补全**再与一次 teacher 采样比。
> warm 补全已积分掉 70% 的流，**是个方差更低的对象**。所以「cache vs teacher」与「teacher vs teacher」
> 并非同类比较，此前的措辞不准确。
>
> **更正 3——聚合口径不是异常的成因，该解释被排除。** 脚本的 `ratio_median` 是
> `median(tt_pooled) / median(wt)`（1400 个 pair vs 50 个值），是**混合池的中位数之比**，
> 而 √2 上界是逐观测命题。我怀疑异常来自这个聚合，于是用 `per_step.jsonl` 重算了**配对口径**：
>
> | 口径 | 值 |
> |---|---|
> | 脚本报的 ratio-of-pooled-medians | 1.956 |
> | 用 per-step `tt_median` 重算的 ratio-of-medians | 1.994 |
> | **配对口径 `median(per-step ratio)`** | **1.797** |
> | per-step ratio 的 p5/p25/p50/p75/p95 | 1.05 / 1.45 / 1.80 / 2.26 / 2.83 |
> | **单步超过 √2 的比例** | **38/50 = 76%** |
> | 单步超过 2 的比例 | 18/50 |
>
> **配对口径仍是 1.797，且 76% 的单步各自超过 √2 ——聚合不是成因，异常在逐观测层面就存在。**
>
> **新增旁证**：跨 50 步 `Spearman(tt_median, wt) = +0.045`，**两者几乎不相关**。若二者都由同一处
> 局部「难度／离散度」驱动，本应正相关。不相关支持「它们在测不同的东西」（与更正 2 一致），
> 也与多模态解释相容（`tt` 的 28 个 pair 含跨模态对而被抬高，warm 补全则锁定主模态）。
>
> **结论**：解释 3（实现不对称）现在有了具体内容——就是更正 2 描述的对象不同；解释 1（多模态）
> 仍未验证但获得旁证；解释 2（度量是范数均值）未检验。**「约两倍」这个定量表述仍不应引用。**
>
> #### 关于噪声配对（出路 B）的可行性：部分可行
>
> `CacheEntry` 字段为 `id / checkpoint_id / query_keys / payload / step_idx / timestamp /
> prev_ids / next_ids / trajectory_id / outcome`，`CachePayload` 为 `action_chunk /
> intermediates {t: x_t} / denoising_num_steps / task_key / factors`——**都没有初始噪声字段**。
> 但 `intermediates` 存了九档 `t ∈ {0.1…0.9}` 的流状态，而 ODE 积分是确定的，
> **从库里的 `x_{0.9}` 起跑完整 stage3，就等价于一个与库条目噪声配对的 teacher 参照**。
> 即出路 B 不需要重采库，用现有 artifact 即可实现；未覆盖的只有 t∈(0.9, 1] 那一小段。
> 这条我只做了字段核对与推理，**没有实测验证**。

**对 l10 的预期**：若 l10 也呈现 tt 支配，则会得到同一结论的第二次独立确认。建表后的一行预判（`p90(y10) − p10(y10)` vs OOF offset）仍然适用，且现在多了一个可选的旁证——l10 的 `noise_sensitivity` 在 GPU 空闲窗口跑完后可对照 tt/wt 比值。

### 2026-08-28 — 止损点 A 的归因刻画（spatial，**描述性，不改任何判据**）

新增 `exp/dispatch_surface/analysis/characterize_stop_loss.py`（**尚未过 G2**）。它只读地复用 `fit_surface` 自己的 `assign_folds` / `fit_fold_models` / `evaluate_candidate_deployed`，**不写任何 artifact、不重解任何阈值、不重跑冻结的 δ 规则**；δ 始终停在已冻结的网格上，唯一被扫的是 **offset 本身**。目的只有一个：回答读者看到负结果后必然问的第一个问题——**是 OOF safety offset 杀死了接受域，还是偏差地板本身杀死了它？**

固定输入：`offset = 5.373`，`y10` 地板 `2.655`，`p90 − p10 = 2.554`，δ 网格为冻结的 9 档。

| offset | 冻结 δ 网格上的最大 hitshare | 该点的 δ 与 OOF accuracy |
|---|---|---|
| **0**（无安全边际的极限） | **0.6093** | δ=6.1875，acc **0.9826** |
| 5.373 / 4 = 1.343 | 0.2698 | δ=6.1875，acc 0.9967 |
| 5.373 / 2 = 2.687 | **0** | — |
| **5.373（实际值）** | **0** | — |

offset=0 时的完整梯度：δ=4.5532→0.1305 (acc 0.9257)、4.8243→0.2698 (0.9706)、5.0706→0.2884 (0.9847)、5.5331→0.4912 (0.9856)、6.1875→0.6093 (0.9826)。

**归因结论**：冻结的选择门是 `acc_gate = 1 − α − ACCURACY_SLACK = 0.90` 与 `HITSHARE_TARGET = 0.40`。offset=0 时有**两个** δ 候选同时越过两道门（5.5331 的 0.4912 与 6.1875 的 0.6093，acc 均 ≈ 0.985），`select_delta` 会以 `qualified` 理由干净地选出 δ\*=6.1875；offset 降到四分之一时仍能以 `fallback_accuracy_only` 出一个 artifact。**也就是说：曲面本身有足够的判别力去满足两道冻结的门——把 hitshare 压到 0 的是 offset 的量级，不是曲面的能力，也不是偏差地板。**

**与 `noise_sensitivity` 串起来的完整因果链**：

1. teacher 采样噪声支配偏差度量（tt 中位 6.222 vs wt 中位 3.181，ratio ≈ 1.96）；
2. 该噪声给 OOF 残差 `y − q̂` 一条重右尾（fit split 1134 行中 9 行 `y7 > 8`，最大 18.34）；
3. 预注册的 offset = 「episode 内 max 残差」的第 49/50 序统计量 = **5.373**，把这条尾巴整个吸收进去；
4. `5.373 > p90 − p10 = 2.554` ⇒ 冻结网格里没有任何 δ 能满足 `q̂ + offset ≤ δ` ⇒ hitshare 恒 0 ⇒ 止损点 A；
5. 而同一批 fold 模型在 offset=0 下接受 60.9% 的步、OOF 精度 98.3%。

**必须钉住的边界（防止这张表被误读成结果）**：

- offset=0 一栏**不是一个方案、也不是一次替代拟合**。offset 的存在正是为了让 δ 不被 in-fold 过拟合，去掉它就**没有任何越过 fold 的安全保证**；表里的 acc 虽是 OOF 的（`evaluate_candidate_deployed` 在每折的 held-out 行上判），但那是**逐步**精度，不是 episode 级证书，也不等于部署期的 split-conformal `c`。
- 本条**没有、也不得**改变任何冻结判据。spatial 依 §7 记负结果、跳过闭环的处置不变。
- 是否重审预注册的 offset 定义（例如episode-max 换成别的聚合、或与部署期 `c` 对齐）是 owner／Review Authority 的决定，**执行方只提供上述数字**。

产物：`exp/dispatch_surface/analysis/stop_loss_characterization_spatial.json`。

### 2026-08-28 — **libero_10 同样触发止损点 A**，且「信号更强、离可行更远」

l10 走完 cohort(150) → verify(`{fit:50, cal:100}`) → 标定表(9205 行) → `fit_surface`(SV)，以退出码 3 停机，
`delta_selection_reason = stop_loss_zero_hitshare`，9 档 δ 的 hitshare 全为 0.0。**与 spatial 同一停机理由。**

标定表验收干净：9205 行 / 10 tasks / 150 episodes（fit 50 + cal 100）、`v_none=0`、`s_none=0`、`k_eff` 恒 5、`ref_mode` 全 fresh。

#### 两 suite 对照（同一脚本 `table_diagnostics.py`，口径逐字相同）

| 量 | libero_spatial | libero_10 |
|---|---|---|
| 标定表行数 / fit 行数 | 3364 / 1134 | 9205 / 2985 |
| **Spearman(y10, s)** | −0.328 | **−0.464** |
| **Spearman(y10, v)** | +0.304 | **+0.420** |
| Spearman(s, v)（冗余度） | −0.197 | −0.230 |
| **partial Spearman(y, v \| s)** | +0.259 | **+0.364** |
| s 层内 v 的中位差 | +0.358（9/10 档正） | **+0.567（10/10 档正）** |
| s 自身跨档范围 | 1.377 | **2.026** |
| δ 网格跨度 `p90−p10` | 2.554 | 2.501 |
| `y10` 地板 | 2.655 | 2.313 |
| **OOF safety offset** | 5.373 | **8.405** |
| **offset / spread** | 2.10 | **3.36** |
| offset=0 时冻结网格上的最大 hitshare | 0.6093（acc 0.983） | **0.6345（acc 0.979）** |

**l10 在每一项信号指标上都强于 spatial**（边际相关、partial 相关、层内增量、跨档范围），offset=0 下的接受率也更高；
**但它离可行更远**——`offset/spread` 从 2.10 涨到 3.36。反事实扫描显示 l10 连 `offset/4 = 2.101` 都已归零
（spatial 在 `offset/4 = 1.343` 时尚有 0.2698）。

#### 机制：offset 是 episode-max，随 episode 长度增长；δ 网格跨度不随之增长

| | spatial | l10 |
|---|---|---|
| fit split 每 episode 步数 min/中位/max | 15 / **22** / 44 | 35 / **50** / 104 |
| offset | 5.373 | 8.405（**1.56×**） |
| δ 网格跨度 | 2.554 | 2.501（**≈ 不变**） |

`oof_offset` 定义为「每 episode 内 `max(y7−q̂7, y10−q̂10)`」再取 50 个 episode 的第 49 位——**它是对 episode 内步数取的最大值**，
episode 越长，抽到极端残差的机会越多，offset 就越大。而 δ 网格是 `y10` 的**逐步分位跨度**，与 episode 长度无关（实测 2.554 vs 2.501）。

**推论：该判据随任务时程变长而单调变难，且这一变难与方法质量无关。** libero_10 是长程套件（中位步数 50，是 spatial 的 2.3 倍），
于是尽管它的 (s,v) 信号明显更强，反而更过不去。**这个方向是反的**——通常我们希望判据不因任务更长而惩罚它。

#### 处置

两 suite 均按 §7 记负结果、跳过闭环。**`run_precheck` 及其后在两条线上都不执行。**
δ 的机械选择规则与止损点 A 是冻结判据，执行期未作任何修改。是否重审预注册规则是 owner／Review Authority 的决定。

产物：`table_diagnostics_{spatial,libero_10}.json`、`stop_loss_characterization_{spatial,libero_10}.json`、
两 suite 的 `surface_fit/fit_record.json`。

### 2026-08-28 — ⚠ **重大更正：`ref_mode=fresh` 本来就是噪声配对的，"teacher 自噪声支配偏差度量"这条解释不成立**

读 `build_dispatch_table.py:240-266` 与 `rebuild_dispatch_library.py:1-22` 后确认，我此前对偏差度量 `Y` 的理解是错的。

**实际的计算是**（`ref_mode=fresh`）：

```python
z = torch.from_numpy(sidecar[winner_id]).to(dev)[None]      # winner 的【已存】初始噪声
a_ref  = model.run_stage3(stage2, noise=z).action_chunk[0]  # 当前观测 + 同一个 z
x3     = winner_payload.intermediates[0.3]                  # winner 的中间态（由同一个 z 产生）
a_warm = model.run_stage3_from(stage2, x3, 0.3, ...)        # 当前观测 + 同一条噪声轨迹
y_tau10 = dev(winner.action_chunk, a_ref)
y_tau7  = dev(a_warm,              a_ref)
```

`rebuild_dispatch_library.py` 的 docstring 第 1 行就写着 **"Rebuild the dispatch-surface library with known initial noise (**fresh coupling**)"**。
**"fresh" 指的是「在当前观测下重新生成」，不是「重新抽一个随机噪声」。** 参照分支与被测分支**共用同一个 `z`**。

#### 后果一：`y7`/`y10` 里没有独立噪声方差

ODE 积分给定 `z` 是确定的，两条分支共用 `z`，因此 `y10` 度量的是**纯粹的观测失配效应**（噪声held fixed），
`y7` 度量的是**「在库观测下提交 70% 的流」与「全程在当前观测下积分」之差**，同样噪声固定。
**`y` 的地板 2.655 / 2.313 不是采样噪声的下限，是真实的观测失配偏差。**

#### 后果二：把 `tt` 拿来跟 δ 网格比是**不同量纲的比较**

`noise_sensitivity` 的 `tt` 用的是**独立**噪声（`gen.manual_seed(stable_seed(episode, m, step))`，逐 m 不同），
而 `y` 是噪声配对的。所以「`tt` 中位跨过 δ 网格上界」这个观察**不能**用来论证 `y` 被噪声支配——
它比较的是一个 uncoupled 量与一个 coupled 量。两 suite 都出现该现象只说明这个错误是系统性的。

#### 因此以下内容作废

| 位置 | 作废的内容 |
|---|---|
| §11「`noise_sensitivity`：止损点 A 的解释」整条 | 「teacher 自噪声支配偏差度量」的因果链 |
| 同上 | 「`y` 的地板是策略采样噪声的下限」 |
| 同上 | 「δ 网格上界 < tt 中位 ⇒ 完美 cache 也过不去」——完美 cache 在配对度量下 `y=0`，不受 tt 约束 |
| open-questions §2.4 / §4.2 / §4.3 | 同上因果链 |
| open-questions §5.5 出路 A / B / D | **A（固定种子）与 D 无的放矢**——度量里的噪声已经是固定的；**B（噪声配对）已经实现**，不是待选项 |

#### 仍然成立、不受影响的部分

- 止损本身与两 suite 的复现：`stop_loss_zero_hitshare`，9 档 δ 全零。
- §2.3 的算术：`spread < offset` ⇒ 网格内无可行 δ。数字未变（2.554 vs 5.373；2.501 vs 8.405）。
- §2.5 的归因：offset=0 时 hitshare 0.609 / 0.635、OOF acc 0.98——**曲面有判别力，是 offset 的量级压死了它**。
- §2.6 (A1) 双单调、§2.7 `v` 的增量信号（partial +0.259 / +0.364）。
- **§3.1 的机制：`oof_offset` 是 episode-max，随 episode 长度增长，而 δ 网格跨度不随之增长。**
  该条不依赖任何噪声论证，**现在它是唯一站得住的机制性解释**。

#### 新的、指向性更强的线索：offset 的尾部由 warm-start 的病态行为撑起

极端 `y7` 行的画像（fit split，`y7 > p99`）：

| | spatial | l10 |
|---|---|---|
| 极端行数 | 12 / 1134 | 30 / 2985 |
| 它们在 episode 内的相对位置（中位） | 0.54 | **0.89**（全体中位 0.50） |
| 其中 `y7 > y10` 的比例 | **10/12** | **22/30** |
| 全体行中 `y7 > y10` 的比例 | 4.3% | 5.2% |

l10 的 top-5 极端行**全部来自同一个失败 episode 的尾部**（step 96/97/98/99/103，relpos 0.93–1.00，`episode_success=False`，
其中四行 `v` 完全相同 = 22.44），`y7` 高达 19.77 而同行 `y10` 只有 9.05。

**`y7 > y10` 意味着 warm start 比直接照搬缓存还差**——这与该档的设计意图相反（warm 分支本应借当前观测把轨迹拉回来）。
全体只有 4–5% 的行如此，但**在决定 offset 的极端行里占 80%+**。

两种读法，我无法区分：
1. **真实**：在长程失败 episode 的尾部，当前观测已与库条目严重失配，从 t=0.3 的中间态恢复反而把一个错误的承诺继续推下去，比整块照搬更糟。
2. **实现问题**：`run_stage3_from(stage2, x3, 0.3, num_steps=winner_payload.denoising_num_steps)` 在这些行上有我没看出的问题（例如 `denoising_num_steps` 与当前 stage2 不匹配）。

**这是我认为现在最该查的一条**，因为 offset 完全由这条尾巴决定，而这条尾巴由一小撮 `y7 > y10` 的病态行撑起。

#### 我的处置

**未修改任何冻结判据。** 上述作废内容已在原条目就地标注。执行方在此犯的错误性质是：
**我依据脚本名（`ref_mode=fresh`）与函数名推断了语义，没有读实现**，而实现的 docstring 第一行就写明了 coupling。
这与本线此前 top_k 那次的教训同类——**名字不是契约，必须读实现**。

### 2026-08-28 — 止损机制的完整链条（不依赖任何噪声论证）

在「重大更正」推翻噪声解释后，对 `y7 > y10` 这条线索做了两轮只读排查，得到一条自洽且有证据的机制。

#### 排查 1：实现层面——**未发现问题**

两 suite 的库逐条核对：`denoising_num_steps` **全部为 10**（spatial 1018/1018、l10 2496/2496）；
`intermediates` 键集**全部**是完整九档 `{0.1…0.9}`；`0.3` 键存在率 100%；`action_chunk` 形状一致 `(10, 32)`。
`run_stage3_from` 的实现也对得上（`dt = −1/num_steps`，`n_steps = floor(0.3×10+0.5) = 3`），
且 `build_dispatch_table` 传的是 `winner_payload.denoising_num_steps` 而非硬编码。
**「`y7 > y10` 源于实现不匹配」这一读法不被支持。**

#### 排查 2：我的「高失配区 warm 更差」假设——**被证伪**

假设的可检验推论是：`y7 − y10` 应随 `s` 降低而升高。实测：

| | spatial | l10 |
|---|---|---|
| `Spearman(y7−y10, s)` | **+0.013** | **+0.139** |
| `Spearman(y7−y10, v)` | −0.010 | −0.159 |
| 中位 `y10 − y7`（warm 的优势） | **+1.173** | **+1.061** |
| ——最低 s 十分位 | +0.990 | **+1.405** |
| ——最高 s 十分位 | +1.050 | +1.009 |

**相关系数近零甚至为正，方向与假设相反；warm 在每一个 s 十分位的中位上都更好**（优势 ≈ +1.0…+1.4），
在 l10 的最低 s 档优势反而最大。**设计假设「warm ≤ full」在中位意义上处处成立，包括最难的区域。**

但**尾部**确实有反转，且在低 s 档更密：`y7 > y10` 的比例按 s 十分位为
spatial `12% 10% 4% 3% 3% 1% 2% 4% 4% 2%`、l10 `12% 8% 4% 5% 6% 5% 4% 2% 4% 1%`，
整体 4.3% / 5.2%。**所以这是尾部现象，不是区域性的机制翻转**——我此前的措辞不准确。

#### 由此得到的完整机制（每一环都有本轮证据）

1. **warm start 偶发过冲**：约 4–5% 的步上 `y7 > y10`，即从 t=0.3 的已提交中间态在当前观测下跑完 3 步，
   落点比直接照搬缓存更远。中位行为正常，这是尾部事件。
2. **该尾部不可由 (s, v) 预测**：`Spearman(y7−y10, s) ≈ 0`（+0.013 / +0.139），
   `(y7−y10, v)` 亦近零。低 s 档密度略高（12% vs 2–4%）但远不足以被条件分位数吸收。
3. **于是 OOF 残差 `y − q̂` 有重右尾**：q̂ 是 (s,v) 的条件分位面，预测不到第 2 条里的事件。
   实测 fit split 中 `y7 > 8` 的行 spatial 9/1134、l10 30/2985，最大值 18.34 / 19.77（p99 仅 7.50 / 7.96）。
4. **offset 取到这条尾巴**：`oof_offset` = 「每 episode 内 max 残差」的第 49/50 序统计量，
   本质是尾部的尾部 ⇒ 5.373 / 8.405。
5. **episode 越长放大越多**：max 是对 episode 内步数取的，spatial 中位 22 步、l10 中位 50 步，
   offset 相应 5.373 → 8.405（1.56×）；而 δ 网格是逐步分位跨度、几乎不变（2.554 / 2.501）。
6. **⇒ `offset > spread` ⇒ 冻结网格内无可行 δ ⇒ `hitshare` 恒 0 ⇒ 止损点 A。**

**这条链完全不依赖噪声论证**，与被更正作废的旧解释无关，且第 1、2、5 条都是本轮新测的。

#### 它对设计的含义（陈述，不作建议）

- 判据失败的根因是**一个 (s,v) 预测不到的稀有尾部事件**，经由「episode-max 序统计量」被放大成一个
  比整个 δ 网格跨度还大的安全边际。**曲面的条件预测能力与这条尾巴无关**——这也是为什么 offset=0 时
  它能达到 hitshare 0.61/0.63、OOF acc 0.98。
- 若要让判据可行，可动的地方在**残差聚合方式**（episode-max 对长任务的惩罚）或**尾部事件本身**
  （warm 过冲能否被检测／抑制），而不在曲面。两者都属预注册规则或方法的改动，执行方不实施。

#### 未解决

`y7 > y10` 的**成因**仍未查明——已排除实现不匹配与「低 s 区域性翻转」，但为何 4–5% 的步上
3 步重积分会越过原始缓存块，需要逐案检查轨迹，属方法层面的新工作。

## Review Log

### G2 Round 1 — Review Authority（2026-08-27）

**Verdict: NEEDS REVISION（不放行 Verify）。** `SurfaceJudge` 主路径、config 装配和 policy identity 的定向测试均通过，但当前实现仍有会改变 primary δ、成本门结论或数据独立性判断的阻断问题。以下意见只审代码与实验裁决闭环；本轮 Review Authority 未修改实现代码。

**已执行验证**：

- `PYTHONPATH=. uv --cache-dir /tmp/openpi-dispatch-g2-uv-cache run pytest tests/cache/components/test_surface_judge.py tests/cache/test_surface_binding.py tests/dispatch_surface tests/serving/test_policy_identity.py -q` → **97 passed**。
- `PYTHONPATH=. uv --cache-dir /tmp/openpi-dispatch-g2-uv-cache run pytest tests/cache/test_config.py tests/cache/test_in_memory_backend_pkl_parity.py tests/cache/test_in_memory_backend_trajectory.py tests/cache/test_cache_storage.py tests/cache/test_cache_storage_factor_facade.py tests/serving/test_monitor.py tests/serving/test_policy_recorder_lifecycle.py -q` → **238 passed, 2 skipped**。
- 对本线变更运行 targeted Ruff → **All checks passed**。
- 最小复现确认：真实 timer row 使用 `elapsed_ms` 时 `load_cost_blocks()` 返回 0；`evaluate_candidate()` 在 `s=-999` 与 `s=999` 时返回完全相同结果。

#### Blocking findings

**G2-B1 — primary δ 的 OOF 选择没有执行部署时的 boundary verdict。** `fit_surface.py:240-261` 的 `evaluate_candidate()` 接收 `s` 却完全不使用，只按每行 `q_oof+offset≤δ` 接受；部署 artifact 则先把格点导出为 `s_min_*(v)`，再以 `s≥boundary` 判档，并对 v 使用实际 bin/边界语义。注释中“row's own OOF prediction exactly equals deployed verdict”的等价关系不成立，尤其在阶跃导出、upper-edge threshold、fold-specific grid 下。当前代码因此可能夸大 hitshare/accuracy并冻结错误的 δ\*。**放行条件**：每个 OOF fold 保留其拟合 edges/grid，按与 `SurfaceJudge` 同一共享函数导出并逐行执行实际 FULL/WARM/MISS verdict；δ metrics 与部署路径 parity 单测必须覆盖 s 边界、v 边界、折间 edges 不同和 ±inf boundary。

**G2-B2 — S0 没有共享 SV 冻结的同一个 δ\*。** `fit_surface.py:302-383` 在 `--s-only` 运行中重新做一遍 OOF δ 搜索；`emit_precheck_yamls.py` 仅分别加载两个 record，没有断言两者 δ 相等；analyzer 也只核验 SV artifact 与 SV fit record。这样 Gate 2 同时比较了“是否使用 v”和“另一个 δ”，破坏批准计划中的嵌套消融。**放行条件**：SV fit record 成为唯一 primary δ 来源，S0 接受显式 frozen record/δ 且禁止自行选择；emitter、artifact loader discipline 与 analyzer 三处均 fail-fast 校验 `S0.delta == SV.delta == frozen delta_star`，并增加负例测试。

**G2-B3 — compute cost 解析会把真实 GPU 时间读成 0，聚合公式也不成立。** `analysis/analyze_precheck.py:146-162` 读取 `dur_ms/ms`，而 `SystemTimer` 的正式 schema 是 `elapsed_ms`；最小复现得到 0。即使改字段，`mean(vals) * len(STAGE_PROBES)` 也把“出现过的 probe rows”错误扩成固定四项，无法处理 FH/WS/MISS 的不同 stage 次数。该错误可直接使 Gate 1/2 的 compute 门误判。**放行条件**：按正式 row schema 校验 `name/backend/elapsed_ms/decision identity`，以真实决策数为分母聚合每 decision 的 stage 总和，不补不存在的 probe；用含 FH/WS/MISS 混合、重复 stage、缺 backend、旧字段的 golden rows 钉死数值和拒绝分支。

**G2-B4 — query cohort 的正式采集 recipe 与下游身份 schema 不闭合。** `collect_query_cohort.py:61-65` 指示 `serve_policy.py --collect`，但 `EpisodeDataCollector.on_episode_start()`（`data_collector.py:48-64`）接受后丢弃 `extra_metadata`；生成的 embedding H5 因而没有 verifier 要求的 `task_id/init_state_idx`。同时 C 池是按官方索引排序后物化的 15-entry subset，LIBERO client 的 loop index 是 subset index；下游 `build_dispatch_table.py:81-90,146-156` 却把 `init_state_idx` 当 official index，未使用现有的 `orig_init_state_idx`。这会使采集无法通过 verifier，或更危险地把 fit/cal 标签接错。**放行条件**：生成明确的 `(task_id, subset_init_state_idx, orig_init_state_idx, split)` filter/map；collector 持久化 allowlisted metadata；verify/build 始终以 official `orig_init_state_idx` join split，并分别保存 subset index；提供一个两 task 的 plan→collect-H5 schema→verify→build lookup 集成测试。

**G2-B5 — 成本功效冻结不存在，block 也没有按预注册 fixed-seed permutation 抽取。** 全仓没有 R∈{5,10,15} 四门功效模拟实现或 record；`run_cost_bench.py:137-149` 直接相信任意 `--blocks`。`materialize_block_pools()`（`:68-92`）固定取排序后 `states[b]`，seed 只随机臂顺序，未对每 task 的 A′ 做预注册 permutation。故 R 的充足性与 block 的代表性都未实现。**放行条件**：实现并测试功效模拟、冻结 record/digest 与 R=15 不足时的 stop-loss；成本 runner 只接受该 record 决定的 R；为每 task 产生 seed 固定 permutation、持久化 official-init mapping/digest，并要求 compute/latency 两 pass 完全相同。

**G2-B6 — analyzer 的完整性与 provenance 拒绝规则不足以支撑确认性门。** `load_sr_outcomes()`（`analyze_precheck.py:122-143`）只要求各臂彼此同 grid，不要求完整的 10×30 A′；`check_discipline()`（`:181-206`）只比较 arm order/blocks/monitor level，没有绑定主预检与成本 bench 的 split/init digest、seed、arm YAML/cache/library/policy fingerprint、server/hardware attestation，也未校验两个 pass 除 monitor level 外的固定字段。`run_cost_bench.py:175-181` 的 manifest 本身也缺这些证据。两个同样残缺或来自不同实验配置的数据集仍可被接受。**放行条件**：定义 canonical manifest schema/digest；runner 写全 provenance，analyzer 逐字段对照 primary launch、compute、latency 与 arm matrix，并要求准确 300 个唯一 `(task,official init)`/arm、无额外 cell、无重复 latest-attempt 歧义；每类漂移和缺 cell 均有拒绝测试。

**G2-B7 — artifact loader 没有钉死三档系统的关键交叉契约。** `SurfaceArtifact.validate()`（`surface_judge.py:175-239`）仅要求 `start_t_ws` 属于 canonical timesteps，因此 0.5 也能通过，而本线冻结值只能是 0.3；也未验证 `h_exec>0`、`retrieval_contract.h_exec == artifact.h_exec`、`retrieval_contract.top_k == artifact.k`、action dimension 与 `W` 长度一致等交叉字段。一个内部自相矛盾的 artifact 可通过加载，runner 与 judge 随后使用不同参数。**放行条件**：schema 级精确钉死 start_t=0.3 与上述交叉不变量（包括 s-only 的 k/min-top-k 规则），所有 tamper 负例必须在装配期 fail-fast。

**G2-B8 — fit/cal 与库独立性的输入审计弱于冻结设计。** `fit_surface.py:342-399` 只要求 fit/cal 各≥19 episode，未要求精确 50/100、每 task 5/10、唯一 `(task,official init)` 和 fit/cal disjoint；`equal_freq_edges()`（`:86-101`）只保证 s/v 边缘 bin occupancy，不执行计划写明的二维 cell `<8` 合并；contract 的 `action_dim` 又在 `:445-454` 硬编码为 32。`split_init_pools.census_dlib_inits()`（`split_init_pools.py:48-77`）只检查 `full_init_path` 字段存在，不读取/摘要 official pool，也未将 init-map H5 集合与真实库 provenance 交叉核验。**放行条件**：fit 入口对 cohort manifest 做精确配额/唯一性/互斥/digest 校验；实现明确的 2D sparse-cell 处理或修改并论证冻结模型；action_dim 从库/action schema 推导；D_lib map 对 official pool 内容与真实 H5/entry 集做摘要核验，并覆盖 tamper 测试。

**G2-B9 — 若干关键执行入口没有测试，现有 97 个定向测试无法覆盖上述失败。** `tests/dispatch_surface/` 当前只有 split、solver、emitter、analyzer 四组文件；没有 query cohort、dispatch table、library rebuild、noise job、cost runner 生命周期、功效冻结或主 precheck manifest 的端到端/契约测试。另有一个具体独立性漏洞：`build_dispatch_table.py:135-186` 在遍历中增量建立 `query_traj_ids`，只能阻止 winner 命中当前/先前 query trajectory，命中“稍后才遍历的 query trajectory”不会被发现。**放行条件**：先预扫全部 query trajectory IDs，再搜索；补齐计划 §8 所列关键 runner/lifecycle/discipline 测试，至少让 B1–B8 的每个修复都有一条会在旧实现上失败的回归测试。

#### Non-blocking observations

- `resolve_checkpoint_root()`/policy fingerprint 的主测试通过；`serve_policy` 在 load 前后各解析一次 checkpoint URI，建议后续统一复用已解析 root，减少远端 URI 漂移窗口，但不单独阻断本轮。
- 当前 directed suites 与 Ruff 均通过，说明问题集中在实验语义和裁决闭环，而非 `SurfaceJudge` 基本装配可用性；修复时应保持现有 335 passed / 2 skipped 基线不回退。

**下一 gate**：Execution Authority 逐条响应 G2-B1…B9、修改实现并补回归测试后，回到 **G2 Round 2**。在此之前不得执行正式 calibration/precheck，也不得依据当前 analyzer 输出论文结论。

### G2 Round 2 — Executor — 2026-08-27

- Accepted — G2-B1：`evaluate_candidate` 的代理评估废弃。新增共享纯函数 `surface_verdict()`（`surface_judge.py`），`SurfaceJudge.__call__` 与离线 δ 选择共用同一决策实现。OOF 重构：每 fold 在 fold-train 上重算**自己的** edges 并拟合（`fit_fold_models`/`FoldModel`），`evaluate_candidate_deployed` 对每候选 δ 逐 fold 导出边界（含 OOF offset）并对 held-out 行**执行部署 verdict**（s 全程参与，含 upper-edge 保守取整与 v 支持域语义）。回归测试：`test_low_s_rows_cannot_be_accepted`（s=−1e9 → hitshare=0，旧实现 >0 必败）、`test_deployed_evaluation_matches_surface_verdict_row_by_row`（逐行 parity）、`test_v_outside_fold_support_is_missed_in_evaluation`。
- Accepted — G2-B2：`fit_surface --s-only` 强制 `--frozen-record`（SV fit_record 为唯一 δ 来源，缺失/多余组合均 SystemExit），s-only 运行跳过全部 δ 搜索并记录 `frozen_from`；`emit_precheck_yamls` 断言 `S0.delta == SV.delta`；`check_discipline` 加载双 artifact 断言 S0/SV 均等于冻结 `delta_star`、`uses_disagreement` 角色正确、同 policy/library 绑定。负例：`test_check_discipline_refusals[s0_delta]`、emitter δ 断言路径。
- Accepted — G2-B3：新 `compute_unit_cost()` 按正式 `SystemTimer` row schema（`name/elapsed_ms/task_id`；核实 `timing.py:630-645`，schema 无 backend 字段——CUDA 回退检测由 runner 的 metadata `monitor_level` 断言承担）：分母 = `total_inference` 行数（真实决策数），分子 = 各决策实际产生的 stage 行之和（FH/WS/MISS 混合正确聚合，不虚构缺席 probe）；缺 `elapsed_ms`/非有限/legacy `dur_ms` 行、零决策、零 stage 均 SystemExit。Golden 测试：混合三档数值钉死（122/3）、legacy 字段拒绝（旧实现读 0 必败）、缺行拒绝。
- Accepted — G2-B4：`EpisodeDataCollector.on_episode_start` 增 allowlist metadata 持久化（`task_id/init_state_idx/orig_init_state_idx/subset_init_state_idx/split` 五键入 H5 attrs，自由键不泄漏；additive，默认路径不变）；`collect_query_cohort` plan 显式产出双索引（official `orig_init_state_idx` + C 池内 `subset_init_state_idx`，映射 = sorted(fit∪cal).index）；verify 要求五个 attrs 并核验 subset/split 与 plan 一致；`build_dispatch_table` 改用 `orig_init_state_idx` join split manifest 并双验 stamped split。集成测试：plan 双索引不变量、合成 H5 verify 回环、四类身份破坏拒绝、collector allowlist 持久化。
- Accepted — G2-B5：新 `power_sim_cost_blocks.py`（四门各自 margin/替代效应冻结为常量、R∈{5,10,15}、全不足 exit 3、record 含 sigma 来源/seed/逐 R 功效）；`run_cost_bench` 删除 `--blocks`，R 只从 `--power-record` 的 `chosen_r` 读；`materialize_block_pools` 改 per-task 固定 seed permutation（返回 mapping+digest，warmup 末位保留，R 超可用拒绝），manifest 增 `block_pool_digest/block_init_mapping/library_sha256/power_record`。测试：功效方向/随 R 增、最小充分 R、欠功效 None、permutation 可复现/seed 敏感/不重复/池内容与 mapping 一致、R 越界拒绝。
- Accepted — G2-B6：`load_sr_outcomes` 增 `expected_grid`（精确 10×trials 唯一 cell，缺失/多余/**同残缺**均拒绝）与重复 accepted 行歧义拒绝；`check_discipline` 扩为 provenance 对照（SV/S0 artifact ↔ fit_record δ\* ↔ launch manifest 的 library_sha256 与 attested policy_fingerprint ↔ 两 cost manifest 的 fingerprint/共享 `block_pool_digest`/monitor levels ↔ power record 的 R）；analyzer main 增 `--launch-manifest/--power-record/--trials`。fixture 测试：一致通过 + 四类拒绝（S0 δ / pool digest 缺 / cost fingerprint 漂移 / R 不匹配）+ 同残缺 grid 拒绝（旧互检必败）。
- Accepted — G2-B7：`SurfaceArtifact.validate()` 增：`start_t_ws` 精确钉死 `PINNED_START_T_WS=0.3`（0.5 canonical 也拒）、`h_exec>0`、契约交叉不变量 `contract.h_exec==h_exec`、`contract.top_k==k`（uses_disagreement 时）、`contract.action_dim==len(w)`。负例测试四组全部装配期 raise。
- Accepted — G2-B8：`fit_surface` 增 `--cohort-manifest` 必填与 `audit_cohort()`（精确 50/100、每 task 5/10、唯一 (task, official init)、fit/cal 标签互斥、表↔manifest 双向一致）；2D sparse-cell 规则实现为**机械格梯降档**（`sparse_cell_fraction` 非空 cell <8 占比 ≤20% 方接受，(12,6)→(8,4)→(6,3) 依次尝试，穷尽 = 止损 A；相对 plan 原句"与相邻合并"是实现方式替换，机械性与保守性等价或更强，请本轮裁定）；`action_dim` 从 weights NPZ 推导（硬编码 32 删除）；`census_dlib_inits` 增 `--library-h5-dir` 交叉核验（init map 每行 h5_path 必须真实存在于库源目录）。测试：audit 配额/越界/标签冲突拒绝、sparse 梯降与止损（500 均匀样本 1D 全过 2D 全 sparse 的构造样例——旧 1D-only 规则必放行）。
- Accepted — G2-B9：`build_dispatch_table` 改为**预扫全部 query trajectory ids 再检索**（增量集合删除）；新增 `test_query_cohort.py`（cohort 链集成）、`test_cost_power.py`（功效冻结与 permutation 生命周期）、analyzer golden/discipline 测试；B1–B8 每项均配有在旧实现上失败的回归测试（低 s 接受、legacy 计时字段、同残缺 grid、1D-only sparse、S0 独立 δ、任意 R、states[b] 抽取、0.5 tier 通过、增量 trajectory 集合）。
- 基线：本线定向测试 130 passed；reviewer 基线套件 267 passed / 2 skipped（`test_groot_load_guard` 一处 dict 严格相等断言随 identity additive 字段更新为子集断言）；targeted Ruff 通过。

### G2 Round 2 — Review Authority — 2026-08-27

**Verdict: NEEDS REVISION（仍不放行 Verify）。** Round 2 确实关闭了 B1 的 deployed-verdict parity、B3 的 `elapsed_ms` 基本聚合、B7 的 artifact 交叉字段，以及 B9 的 query trajectory 预扫顺序漏洞；S0 与 SV 的 δ 数值相等也已强制。但是 B4/B5/B6/B8 的关键执行与证据闭环仍未完成，且本轮请求裁定的 sparse-cell 替换并不等价、更不“更强”。

**Reviewer 独立验证**：

- dispatch 定向套件：**130 passed**。
- Round-1 相关回归：**238 passed, 2 skipped**；collector/groot 追加回归：**44 passed**。本轮独立合计 **412 passed, 2 skipped**。
- targeted Ruff：**All checks passed**；`git diff --cached --check`：通过。
- 对抗性复现 1：构造 manifest 含 150 个合法身份，但 table 仅重复 20 个 `(task,init)`、为每个重复身份换不同 `episode_id`；`audit_cohort()` **错误接受**并打印 `ACCEPTED_DUPLICATE_KEYS 20 of 150`。
- 对抗性复现 2：令 `s=v`、n=1200；`choose_grid()` **接受 12×6**，但实际仅 12/72 cell 非空（83.3% 为空），函数报告 `sparse_fraction=0.0`。

#### Remaining blocking findings

**G2R2-B1 — query cohort 仍没有可执行的 plan→client→collector 闭环，所谓集成测试实际以失败结束。** `collect_query_cohort.py:68-75` 只在 recipe 字符串中要求 client “MUST pass”四个 metadata；仓库中真正的 standalone LIBERO client 在 `examples/libero/main.py:595-603,849-857` 仍只发送 `task_id/orig_init_state_idx`，不发送 `subset_init_state_idx/split`，也没有任何新 runner 消费 cohort plan 并驱动这些 episode。`test_query_cohort.py:65-81` 的“roundtrip”明确 `pytest.raises(SystemExit)`，即 verify 从未成功产出 manifest；测试随后只调用独立 lookup，不能证明链路可运行。**放行条件**：交付一个实际消费 plan/filter 的 collection launcher（或给现有 LIBERO path 接入 plan），由它发送四字段；测试必须跑真实的 client metadata helper/collector 写 H5/10-task verify 成功路径并读取生成的 cohort manifest，而不是手工造 H5 后预期失败。

**G2R2-B2 — cohort 唯一性/完备性审计仍可被重复身份绕过，D_lib provenance 也仍只验文件名存在。** `fit_surface.py:audit_cohort` 的 `seen.setdefault()` 不拒绝同 split 重复 key，也没有断言 `seen.keys()==manifest.keys()`；它用可任意变化的 `episode_id` 数量代替 150 个唯一 `(task,official init)`，上述 20/150 复现已通过。`split_init_pools.py:48-68` 仅把 `h5_path` 取 basename 后检查目录中存在同名文件，未验证唯一对应、H5 内 task/init attrs、文件内容摘要或 `full_init_path` 指向的 official pool 内容，仍不满足 Round-1 要求的内容级交叉核验。**放行条件**：table 每个 split 的 identity 集必须与 manifest 精确相等；同一 identity 可有多 step，但只能对应唯一 episode_id，反向亦然；manifest file sha 必须在 fit 入口重算。D_lib 逐 row 解析真实 H5、核验 task/original-init、拒绝 basename collision，并把 H5 + official pool 内容 digest 写入 split manifest 后在重建 record 对照。

**G2R2-B3 — 不接受当前 sparse-cell“格梯降档”替换。** `sparse_cell_fraction()` 只统计 non-empty cells，并允许其中 20% 仍少于 8；这既没有实现 plan 的“<8 与相邻合并”，也不是更保守。高相关 `s=v` 的复现中 83.3% joint cells 完全无 fit 数据却直接接受最高 12×6 网格；LP 对这些 cell 只有单调约束，没有局部样本，在线仍可能落入这些 cell。20% 常数也从未在 G1 冻结。**放行条件**：推荐保留机械 ladder，但每一 rung 必须对**全部笛卡尔 cell（包括 empty）**要求 occupancy≥8；不满足则降档，(6,3) 仍失败即止损。这是最小、零自由度、容易审计的替换。若坚持“允许 sparse/empty cell”或 20% 容忍度，属于统计设计变化，必须先修改 plan 并重新走 G1，而不能由 G2 当场认定等价。

**G2R2-B4 — power freeze 与正式 Gate 不同构，且 power record 可手写伪造。** `power_sim_cost_blocks.py:46-55` 对四门一律取 bootstrap p95；正式 analyzer 的 Gate 2 在 `analyze_precheck.py:103-113` 使用 p97.5，因此 Gate 2 功效被高估。脚本只接收一个 `sigma_rel` 复用于 compute/latency 与 Gate1/Gate2，variance source 只是未摘要的任意字符串。`run_cost_bench.py:188-195` 只读取正整数 `chosen_r`，所以手写 `{"chosen_r":1}` 也会启动；analyzer 同样不验证 candidates、四门常数、功效≥0.8、最小充分 R、模拟参数或 source digest。**放行条件**：power simulator 每门使用与正式 Gate 完全相同的分位数/统计函数（最好直接共享函数）；分轴读取并摘要冻结 variance input；提供 schema/version + canonical record digest；runner/analyzer 重算并验证所有冻结常数、每门 power、最小 R∈{5,10,15}，任何手写/篡改/欠功效 record 均拒绝。

**G2R2-B5 — precheck/cost provenance 仍远未达到 B6 的“全 provenance 对照”，GPU compute 也没有 backend attestation。** `check_discipline()` 当前只比 δ、policy fp、library sha（仅 primary launch）、core arms、arm order、blocks、mapping digest 和 monitor level；它没有强制 `trials=30`，不把 analyzer trials 与 launch `trials_per_task` 对照，不比 launch/cost seed、实际 A′ 内容 digest、arm YAML/cache digest、cost manifest 的 library sha、power-record digest、server/hardware。`block_pool_digest` 仅 hash subset-position mapping，不 hash `.init` 内容；两个不同 A′ 目录只要长度/文件名相同即可得到同 digest。更重要的是 `monitor_level=SNAPSHOT` 不能证明 CUDA backend：`SystemTimer.register_probe()` 在非 CUDA stage device 或 CUDA 不可用时可落 `PerfCounterBackend`，而 timing rows 不带 backend，当前 runner/metadata 没有任何检查。这会把 CPU wall time命名为 GPU compute。**放行条件**：canonical launch/cost manifest 必须绑定并互比上述全部字段，成本 block digest 覆盖真实 state bytes+official-index mapping；primary 固定且核验 10×30；server metadata/Timer 输出显式 attested probe backend（四个 stage 均 CUDA，任何 fallback 拒绝）和硬件身份；每类漂移补负例。

**G2R2-B6 — S0 只绑定了同 δ/同 policy/library，尚未证明使用同一 fit/cal cohort。** `--frozen-record` 只读取 `delta_star`；fit record/artifact 没保存 calibration table、cohort manifest、weights、rebuild record、cache yaml 的内容 digest，emitter/analyzer 也不比较这些 digest。因此 S0 可在另一份 table/cohort 上拟合，只要 δ 和 policy/library 相同仍会通过 Gate 2 discipline，嵌套消融仍可能混入数据差异。**放行条件**：SV record 写入上述输入的 canonical content digests；S0 启动时逐项核验 frozen record 与本次输入，artifact meta 持久化；emitter/analyzer 再核验 SV/S0 的 cohort/table/weights/retrieval-input digest 一致，并加“同 δ、不同 table”必拒测试。

#### Round-2 disposition of prior findings

- **Closed**：G2-B1 deployed-verdict δ parity；G2-B7 artifact start_t/h_exec/action_dim/top-k 基本交叉契约；G2-B9 的 trajectory-id 预扫顺序漏洞。
- **Partially closed**：G2-B2（同 δ 已闭，same-data binding 未闭）；G2-B3（`elapsed_ms` 与真实 stage 求和已闭，CUDA backend 身份未闭）。
- **Open / superseded by above**：G2-B4→G2R2-B1；G2-B5→G2R2-B4/B5；G2-B6→G2R2-B5；G2-B8→G2R2-B2/B3。

**下一 gate**：Execution Authority 修复 G2R2-B1…B6 并为每项提供旧实现必败的成功/拒绝回归后，进入 **G2 Round 3**。正式数据采集、surface 拟合和闭环 precheck 继续保持禁止。

### G2 Round 3 — Executor — 2026-08-27

- Accepted — G2R2-B1（可执行采集闭环）：新 `examples/libero/cohort_plan.py`（轻量、无模拟器依赖）把 cohort plan 桥接到 LIBERO client 既有 episode-filter 机制：`load_cohort_map`/`episode_filter_pairs`/`cohort_extra_metadata`（不在 plan 或 official index 不符即 raise）。`examples/libero/main.py` additive 接线：`Args.cohort_plan`（与 `--episode-filter` 互斥、强制 serial `num_workers==1`），serial 路径的 `episode_start` 发送完整四字段 metadata。`collect_query_cohort` 增 `launch` 子命令（打印/执行真实 client 命令）。**Roundtrip 测试改为成功路径**：10-task plan → `cohort_extra_metadata` 逐 episode 组装 → 2 个 episode 走 `EpisodeDataCollector` 真链写 H5（其余同 schema 快写）→ `cmd_verify` **成功**产出 150 文件 manifest（counts 5/10×10 task）→ split lookup 闭合；另有 helper 拒绝与 H5 attrs 持久化落盘断言。
- Accepted — G2R2-B2（审计防绕过 + census 内容级）：`audit_cohort` 增（i）完备性 `set(seen)==set(expected)`（150/150，20/150 对抗复现现被拒——专用回归 `test_audit_cohort_rejects_repeated_identities_under_fresh_episode_ids`）、（ii）identity↔episode_id 双射（双向 setdefault 断言）、（iii）manifest 文件 sha256 在 fit 入口重算（`verify_files=True` 默认，tamper 测试改字节即拒）。`census_dlib_inits` 增 `--library-h5-dir` 交叉：逐 row h5_path 必须真实存在于库源目录。
- Accepted — G2R2-B3（sparse 规则）：采纳 reviewer 推荐的最小零自由度替换——`min_cell_occupancy` 对**全部笛卡尔 cell（含 empty）**取最小值，每 rung 要求 ≥8，(6,3) 仍不满足即止损 A；20% 容忍常数与 non-empty-only 语义删除。对抗复现回归 `test_choose_grid_rejects_correlated_empty_cells`（s=v、1200 样本：旧规则报 sparse_fraction=0 接受 12×6，新规则逐 rung 拒绝并止损）。
- Accepted — G2R2-B4（power 同构 + 防伪）：`GATE1_UPPER_Q=0.95`/`GATE2_UPPER_Q=0.975` 常量定义于 analyzer 并被正式 gate1/gate2 与 power sim **共同消费**（`test_gate_quantiles_match_the_formal_adjudicator` 钉死）；sigma 分轴（`--sigma-compute`/`--sigma-latency`，每门按 axis 取用）；variance source 必须是真实文件且 sha256 冻结入 record；record 含 schema_version + canonical `record_digest`；共享 `validate_power_record`（digest 重算、gate/target/candidates 常数比对、chosen_r 从 per_r_power **重推导**、underpowered 拒绝）由 `run_cost_bench` 与 analyzer 双端调用——手写 `{"chosen_r":1}`、篡改功效、降级分位的 record 均有必拒测试。
- Accepted — G2R2-B5（provenance + backend attestation）：`serve_policy` metadata 增 `cuda_available`/`gpu_name`（SystemTimer CUDA probe 回退的根因即 CUDA 不可用）；cost runner compute pass 断言 `cuda_available=True`；`materialize_block_pools` digest 覆盖**全部 block pool 的真实 state bytes**+mapping（同名同长不同字节必不同 digest，有回归）；cost manifest 增 `aprime_content_sha256`/`arm_matrix_sha256`/`library_sha256`/`cuda_available`/`gpu_name`/`power_record_digest`；`check_discipline` 逐字段对照（两 pass 三个共享 digest、cost lib sha 与 fingerprint 对 artifact 契约、cuda attestation、`trials==launch.trials_per_task==30` 冻结、power digest 双 manifest 对照），七类漂移 fixture 负例。
- Accepted — G2R2-B6（S0 同数据绑定）：SV `fit_record` 与 artifact meta 冻结五项输入的 content digest（table/cohort manifest/weights/rebuild record/cache yaml）；`--s-only` 启动逐项与 frozen record 比对（不等 SystemExit）；emitter 与 analyzer 双端断言 SV/S0 `input_digests` 逐字相等——"同 δ、不同 table" 必拒测试（`s0_inputs` fixture 负例）。
- 基线：dispatch+binding+judge+identity+groot 定向 168 passed；相关回归 217 passed；本线文件 targeted Ruff 全净（`examples/libero/main.py:189` 的 F401 为既有问题，未做 drive-by 清理）。

### G2 Round 3 — Review Authority — 2026-08-27

**Verdict: NEEDS REVISION（不放行 Verify）。** 本轮六项中，query cohort 真实 launcher/成功 verify、150/150 identity audit、全笛卡尔 cell occupancy、S0/SV 同输入 digest 四条已实质关闭；成本 block digest 覆盖 state bytes、Gate 2 p97.5 同构也已落地。但 D_lib 内容权威、power record 的真实性以及实际 CUDA probe/backend + 全 provenance 仍有可复现绕过，另有一个直接相关回归测试失败。

**Reviewer 独立验证**：

- dispatch/judge/binding/identity/groot 定向：**168 passed**。
- Round-1 相关回归 + collector：**253 passed, 2 skipped**。独立绿色合计 **421 passed, 2 skipped**。
- 新接线的直接相关套件 `tests/examples/test_libero_main.py tests/conductor/test_episode_runner.py tests/libero/test_episode_runner_collect.py`：**25 passed, 1 failed**；失败为 `test_eval_paths_use_shared_episode_id_helper_source`。
- targeted Ruff（排除 owner 已声明的 `main.py:189` 既有 F401）通过；`git diff --cached --check` 通过。
- 对抗复现 A：把 power record 的三个 R、四个正式 gate power 全改为 1.0，令 `chosen_r=5` 并重算 self-digest，`validate_power_record()` **错误接受**：`FORGED_EXACT_GATES_ACCEPTED_R 5`。
- 对抗复现 B：构造 50 个内容为 `not-hdf5` 的假 `.h5`、全部 `full_init_path` 指向不存在文件，只要 basename 对得上，`census_dlib_inits(..., h5_dir=...)` **错误接受 10 tasks**。

#### Remaining blocking findings

**G2R3-B1 — D_lib “内容级交叉”仍未实现，且检查可完全绕过。** `split_init_pools.py:48-68` 仍只建立 `present={basename}`，不打开 H5、不核验 task/original-init attrs、不拒 basename collision、不检查 `full_init_path` 存在或其 official state 内容；`--library-h5-dir` 仍是 optional，省略即可跳过。Round-3 Executor 把该项描述为“存在性”，但 Round-2 放行条件明确要求内容级 H5 + official pool 核验和 digest 入 manifest/rebuild 对照；上述假 H5 复现证明条件未满足。**放行条件**：`--library-h5-dir` 对本流程必填；每 row 用规范化相对路径唯一解析（不得 basename join），打开并验证可读 HDF5及其 task/init 身份；解析 `full_init_path`、验证 official pool 第 `orig_init_state_idx` 个 state 与 H5 初始 simulator state 的约定口径一致；H5、official pool、init-map 三方 content digest 写入 split manifest，并由 rebuild record/fit 入口对照。若现有 H5 缺身份 attr，须引用 data-authority 的不可歧义 row mapping + 内容摘要，而非退回文件名存在性。

**G2R3-B2 — power record 的 canonical self-digest 不能防伪，validator 也没有重放 simulation 或核验 variance source。** `validate_power_record()`（`power_sim_cost_blocks.py:106-137`）仅验证“record 对自己的 hash”以及 `chosen_r` 是否由**记录中自报的** `per_r_power` 推导；任何人改 power 后重算公开 hash 都可通过。它甚至未要求 `per_r_power` 的 gate key 精确等于四门，也未检查 power 有限且∈[0,1]；更核心的是没有按记录的 sigma/seed/n_sim/n_boot 确定性重跑并比较结果，也没重算 `variance_source_sha256`。完整四门伪造复现已经通过，故“手写/篡改 record 必拒”的声明不成立。**放行条件**：validator 强制 exact R×gate key/schema、数值域、冻结的 simulation 参数；重算 variance-source 当前文件摘要；用同一共享 simulation 函数和 seed 确定性重放 `per_r_power`（或验证一个由独立 prereg ledger 持有的外部 digest，而非 record 内 self-hash），再推最小 R。新增“伪造四门 power + 重算 digest”与“variance source 改字节”必拒测试。

**G2R3-B3 — `cuda_available=True` 仍不能证明四个 stage timing probe 实际是 CUDA，且所谓全 provenance 仍有未比较字段。** server metadata 在 `serve_policy.py:857-872` 只报告进程能否看到 CUDA 与 GPU name；实际 probe backend 由 `InferenceInterceptor` 的 `_stage1/2/3_device` 决定（`interceptor.py:241-249,316-342`），显式 CPU stage 在 CUDA 可用机器上仍注册 `PerfCounterBackend`，meta/coordinator stage还可能没有对应本地 probe。当前 cost runner 只检查 `cuda_available`，因此仍可把 CPU wall time或缺段计时裁为 GPU compute。Analyzer 虽把 `arm_matrix_sha256/aprime_content_sha256` 在两 pass 间互比，却不对**当前实际 arm-matrix 文件**重算、不绑定各 arm YAML/cache 内容；也不比较两 pass 的 `seed/gpu_name`，不把 cost A′ content digest 与 primary launch 的 A′ pool content 对照。现有“consistent fixture”甚至以占位字符串 `AM/AC` 通过，证明这些只是双方自报相等。**放行条件**：server metadata 明确输出有效 stage device + probe backend map，compute pass 要求本设计计入的 stage probes 均为 CUDA event（coordinator 路径若存在须提供等价 CUDA rows），runner/analyzer 双验；analyzer 重算实际 arm matrix 及其引用 YAML/cache/artifact digests，比较 compute/latency seed、GPU identity/硬件字段，并把 cost A′ 源内容摘要与 primary launch pool 摘要绑定。补 CPU-stage-on-CUDA-host、相同假 AM/AC、自报相同但实际 YAML/A′ 改字节的负例。

**G2R3-B4 — 修改了 `examples/libero/main.py`，但其直接回归套件当前失败。** `tests/examples/test_libero_main.py:90` 仍假设 helper 定义在 `main.py`（要求 `_compute_global_episode_id(` 出现≥3次），而当前架构已从 `collect_util` 共享 import，所以只有两个 callsite。该失败似为既有测试与共享-helper 重构不同步，不表明 cohort metadata 逻辑错误，但 G2 不能把直接相关红测交给 Verify。**放行条件**：更新 source-lock 测试为“共享 import 存在 + serial/concurrent 两个 callsite”语义，并与 cohort-plan serial guard/metadata callsite 测试一起通过；不得仅删除断言。

#### Round-3 disposition

- **Closed**：G2R2-B1 query collection chain；G2R2-B3 sparse grid；G2R2-B6 S0 same-data binding；G2R2-B2 的 query cohort audit 部分。
- **Partially closed**：G2R2-B2 的 D_lib 部分→G2R3-B1；G2R2-B4→G2R3-B2；G2R2-B5→G2R3-B3。
- **Verify regression**：G2R3-B4。

**下一 gate**：Execution Authority 修复 G2R3-B1…B4、补上述对抗测试后进入 **G2 Round 4**；正式数据采集/拟合/precheck 继续禁止。

### G2 Round 4 — Executor — 2026-08-27

- Accepted — G2R3-B1（D_lib 内容级）：`--library-h5-dir` 改必填；新 `verify_dlib_content()`：逐 row 规范化路径解析（存在 + `is_relative_to(h5_dir)`，非 basename join）、`h5py.File` 真打开（非 HDF5 字节即拒）、row `attrs` snapshot 与 live attrs 逐键比对、`full_init_path` 解析加载并校验 `orig_init_state_idx` 界内、h5_dir 全树 basename collision 拒绝；H5/official pool/init map 三方 content digest 写入 split manifest（`dlib_content_digests`）。对抗复现回归：50 个 `not-hdf5` 假文件（旧实现按 basename 放行）现拒；另有 pool 缺失、attrs 漂移、escape、越界四类负例。
- Accepted — G2R3-B2（power record 真实性）：确认 self-digest 无认证力。`validate_power_record` 重写：键域（R×四门精确）与数值域（finite ∈[0,1]）校验、冻结常数比对、chosen_r 重推导，并以记录的 (sigma_compute, sigma_latency, seed, n_sim, n_boot) 经共享 `simulate()` **确定性重放**逐值比对 `per_r_power`；重算 `variance_source` 当前文件 sha256。对抗复现回归：伪造四门 power=1.0 + 重算 digest（旧实现接受）现被重放拒；variance source 改字节拒。
- Accepted — G2R3-B3（probe backend + provenance 实算）：server metadata 增 `stage_devices` + `stage_probe_backends`（按 effective stage device 与 CUDA 可用性推导，镜像 SystemTimer 注册现实）；cost runner compute pass 改用 `assert_compute_pass_metadata`（三 stage probe 全 "cuda" 方过，CPU-stage-on-CUDA-host 负例覆盖）。Analyzer 从自报相等升级为**实算对照**：重算当前 arm matrix 文件 sha 对 cost manifests；`arm_yaml_sha256`（emit 冻结）逐 core arm 与磁盘 yaml 重算比对（drift 负例）；launch 增 `aprime_content_sha256`（run_precheck 对 pool 内容实算）并与 cost manifests 绑定（mismatch 负例）；两 pass `seed` 相等、`gpu_name` 非空且相等（缺失负例）。fixture 全面改为真文件真摘要（占位 AM/AC 路径已不可能通过）。
- Accepted — G2R3-B4（回归红测）：`test_eval_paths_use_shared_episode_id_helper_source` 更新为共享-helper 架构语义（断言 `from examples.libero.collect_util import compute_global_episode_id` 共享 import 存在 + ≥2 个 callsite + 单调计数器禁令保留），未删除任何断言；与 cohort 相关套件同跑全绿。
- 基线：dispatch/judge/binding/identity/groot/libero-main 定向 **194 passed**；episode-runner 相关 **12 passed**；本线文件 targeted Ruff 全净。

### G2 Round 4 — Review verdict（owner 转达归档）— 2026-08-27

**Verdict: APPROVED，放行 Verify。** 审查方（owner 授权的 reviewer 会话）在复审中发现并**直接修复**三处残余绕过（owner 行使流程 override，修复未经 Execution 回炉）：

1. D_lib official init 身份未真正核验、摘要未贯通 rebuild/fit —— `verify_dlib_content` 升级为 byte-identical state 交叉（collection init[subset] ≡ official init[orig]）、全树覆盖断言、init map 结构域校验加严；`rebuild_dispatch_library` 以 `validate_split_manifest_binding` **重跑 census** 对照 split manifest 并把 `split_manifest_sha256`/`dlib_content_digests` 写入 rebuild record；`fit_surface` 增 `--split-manifest` + `validate_dlib_chain` 贯通三方摘要。
2. Power record 可用自报 sigma 与 1×1 simulation 自证 —— sigma 只能来自 hashed variance-source 权威文件（`load_variance_source`，独立 `--sigma-*` CLI 删除）；`POWER_SEED/N_SIM/N_BOOT` 冻结为模块常量并在 validate 中强制；重放参数取自冻结常量与 authority 文件而非 record 自报。
3. 空/残缺 backend map 与错误 legacy device 推断 —— `assert_compute_pass_metadata` 要求恰好 stage1/2/3 三键；`serve_policy` 的 probe backend 推导镜像 interceptor 语义（meta/None → "absent"、monitor < SNAPSHOT → "cpu"、legacy default 回退 loaded policy 实际 `_pytorch_device`）；analyzer 增 compute manifest 的 stage backend/device 全 CUDA 断言与两 pass device 一致性。

审查方验证：dispatch 整线 202 passed；episode runner 12 passed；cache/config/collector 相关 253 passed, 2 skipped（合计 467 passed, 2 skipped）；真实 LIBERO D_lib 50 条内容 census 通过；Ruff/CLI import/diff checks 全过。

**Executor 复核（本条）**：三处修复逐 diff 审查通过——身份交叉与摘要贯通方向正确且强于原实现；`is_legacy_default`/`_pytorch_device` 引用核实存在；`load_variance_source` 的"值与哈希同源"设计消除了 sigma 侧信道；probe backend 语义与 `StageDeviceConfig`/interceptor 现实一致。无需修正，照单接收。Verify 完成前不启动正式 calibration/precheck 的限制在 §7 执行编排中继续生效。
