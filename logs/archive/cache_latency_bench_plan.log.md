# Cache 延迟回放基准 (cache_latency_bench) — 实现 Plan

> Status: `In Progress` (G1 APPROVED R5 2026-05-30 / §4 Code) · Level: **L2** · Authority: Execution
> 数据预收集已就绪：`exp/common/data/cache_artifacts/libero_10/*.pkl`（库）+ `exp/common/data/db/libero_cache/libero_10/*.h5`（真实 trajectory）。

---

## 0. 背景（自包含）

本项目（openpi PyTorch + Pi0.5 fork）的推理被切成三段：Stage1(vision+tokenize) → Stage2(LLM backbone) → Stage3(action expert)，并在 Stage1 之后插入 **CP1 cache check**。一次请求的端到端延迟 = 模型推理延迟（Stage1/2/3，GPU）+ cache 系统延迟（CP1 的 `check()`，CPU）。我们记 cache 系统这部分为 **t1**。

`CacheOrchestrator.check()` 内部固定分 6 段，且每段已被 `SystemTimer.measure()` 埋点（探针名 `cp1_collect/cp1_gate/cp1_build/cp1_search/cp1_judge/cp1_fetch`，注册于 `orchestrator.py:176-177`，执行于 `:414-495`）：
1. `collect` — KeyBuilder 持 Stage1 张量引用
2. `gate` — 是否检索（当前 `AlwaysSearchGate`）
3. `build` — KeyBuilder reduce + （真 server 中）GPU→CPU D2H 出 `query_keys`
4. `search` — InMemoryBackend 全库 brute-force + 融合 + trajectory
5. `judge` — threshold / composite verdict 判 FULL_HIT/WARM_START/MISS
6. `fetch` — 仅命中时取完整 payload

**目标**：把 t1 从推理栈里抽出来做成一个**轻量、不加载任何模型**的延迟基准——按用户选定的 yaml 像真 server 一样组装 cache 系统，用 H5 里真实采集的 trajectory 数据逐 step 回放驱动 cache "真实工作"（含跨 step 轨迹记忆），同时探针记录**每请求、每部件**的延迟。

**为什么做得到（关键事实）**：cache 的全部状态/记忆都在 `CacheOrchestrator` 及其组件里（SearchStrategy 的 trajectory history、InMemoryBackend 的 score-memo/search-session、Orchestrator 的 verdict 窗口/step_counter），**模型前向不持有任何 cache 记忆**。H5 里存的是已投影的 Stage1 级 token embedding（`vision_0/1/2 [256,2048]` + `prompt_emb [≤200,2048]` + `robot_state [32]`，实测自 `episode_0000…h5`），不是原始图——所以可离线重建 `Stage1Output` 喂真 KeyBuilder，无需 vision tower。`exp/common/build_in_memory_cache_artifact.py` 已证明这条路（`_build_fake_stage1`）。

---

## 1. 范围 / 非目标

**范围**（infra 本体）：一个新 `exp/cache_latency_bench/` 子系统：输入「真实 cache yaml + H5 目录 + 重复次数」，输出「每 step 每部件延迟 CSV + 聚合统计」。零改 `src/`。

**支持的 judge / calibration 范围（含运动学 verdict）**：infra 用 `build_cache_components` 按 yaml 组装真 `CacheOrchestrator`，故 **composite judge（17-factor 运动学 verdict）开箱即用**——4 层（Normalization/Factor/Calibration/Composer）全按 yaml 实例化、用真 backend + 真 history 驱动。
- ✅ `threshold` / `always_hit` / `always_warm_start`；
- ✅ `composite` 运动学 verdict，**前提（harness 启动时显式校验，见下「为何 harness 自校验」）**：(i) backend pkl 已 enrich，含该 yaml 所有 **offline factor** 声明的 `payload.factors` keys；(ii) Layer 3 calibration `samples_source.type == "offline"`（磁盘 jsonl/pkl，infra 启动直接 load）。
- ❌ `composite` + `samples_source.type == "warmup"`：infra 走 `build_cache_components`（`yaml_id=None`），不经 server 的 ctrl `preload_normalizer_buffer` 链路 → 生产路径会在 `_load_calibration_samples` 的 warmup 分支因 `yaml_id` 缺失 `raise ConfigValidationError`（`config.py:2218-2234`）。**harness 预先拒绝**该类 yaml（`__init__` 检测 `samples_source.type==warmup` → 自己 fail-loud，给出「infra 无 WarmupPool 预填链路，请改用 `samples_source.type=offline`」的明确提示），早于、清晰于下游通用错误（运动学 phase5 实验本就用 `samples_source.type=offline + super_warmup_raw.jsonl`，天然落在支持范围内）。

各运动学 factor 的等价性：**state-channel factor（online+offline）+ offline action-channel factor 完全等价**（state 来自真实 query_keys/history.states、factor 读 enriched pkl，不碰 broadcast 的 action）；**online action-channel factor 是「采集-trajectory 回放、非真 server eval 逐调用等价」**（FULL_HIT 步等价；MISS 步真 server 随机重采、WARM_START 步类型错误——§6.4 / R9，三标志标注）。

> **为何 harness 自校验 enrichment（不依赖生产路径报错）**：生产 backend 对未 enrich pkl **不会**启动期干净报错——`InMemoryBackend.load_artifact` 对缺 `library_stats` 会从 entries **回退计算**（warning 后继续，`in_memory_backend.py:263-283`），offline factor 对缺 `payload.factors` 返回 **全 NaN**（`offline.py:105-115`），二者都可能让运动学 verdict 静默跑成 NaN→MISS、延迟数据失真而无声。故 infra 在 `ReplayHarness.__init__`（组装后、run 前）自加一道 `_validate_artifact_enrichment`：**从 config 推 offline keys（不碰 judge 私有实例）**——枚举每个 composite judge 的 `FactorConfig`，凡 **offline-source** factor（判别复用 `config.py:1885-1889` `_collect_offline_writers_from_judges` 的 `compute_for_episode` duck-type），经 factor registry 由其 `(type, params)` 推出声明的 `payload.factors` keys（命名 `<desc>_offline_<ch>__p<P>_f<F>`，`offline.py:88-99`；online factor 不读 `payload.factors`，免校验），**遍历全部 entries**（直接读 `config.backend` 的 `preload_path` pkl 的 `entries`，**不抽样**——offline factor 是 per-winner 读 `payload.factors`、缺 key 按该 entry 返 NaN `offline.py:105-115`，任一会被命中的 entry 缺 key 即静默失真），**任一 entry 缺所需 key 即 `raise`**（指向 `exp/common/factor_postprocess.py`）。确切 registry key-推导方法名于 Code 阶段定。`library_stats` 缺失只发 warning（backend 能回退，enriched pkl 更快更准）。该校验仅适用 in-memory preload artifact（读 `backend.in_memory.preload_path`）；非 in-memory backend 配 composite/offline-factor judge → harness fail-loud（infra 不支持非 in-memory 库做运动学 verdict）。

**非目标**：
- 不设计/不固化任何具体实验矩阵（keybuilder/depth/in-out-DB 扫描）——那是后续实验，本 plan 只交付可被那些实验复用的基础设施。
- 不测模型推理延迟（infra 不跑 Stage1/2/3 GPU 前向）。
- 不改 `src/openpi/`、不改 server 协议、不改 SystemTimer / orchestrator。

---

## 2. 用户已拍板的设计决策（本 plan 的约束前提）

| # | 决策 | 取值 | 影响 |
|---|------|------|------|
| D1 | stage1 张量设备 / build 段真实性 | **CPU**（最轻，本地无卡） | 6 段中 5 段真实；`build` 段**不含** GPU→CPU D2H/CUDA 同步开销 → build 偏低、与真 server 不可比（已知偏差，见 §6） |
| D2 | 回放期间库写入语义 | **冻结只读**（=真 server runtime） | infra 复用 `_enforce_runtime_write_policy` 做 **fail-fast 校验**（与真 server runtime 同一道关卡）：要求输入 yaml 已声明 `write_policy: {type: never}`，否则 `raise ConfigValidationError`（**不静默改写**，契约见 §6.5）。库 = yaml `backend.preload_path` 全程不变 → R 遍完全可比 |
| D3 | 每请求记录详尽度 | **精简** | 每 step 一行：`repeat,episode_id,step_idx,task,hit_type,top_score` + 6 段 + `cp1_total`；末尾聚合 json（per 段 median/p50/p95，按 hit_type 分组）。不采资源快照、不强开 CP3。 |

---

## 3. 总体设计（数据流）

```
run.py (CLI: --cache-config yaml --h5-dir DIR --repeats N --out-dir OUT [--device cpu])
   │
   ▼
ReplayHarness.__init__(yaml):
   _ensure_timing_enabled()                               # ★build 之前！读 monitor.get_monitor_level()，<BASIC 则 set_monitor_level(BASIC) 提级（使后续构造的 SystemTimer enabled）。否则 timer 在 __init__ 被 level<BASIC 无条件置 no-op（timing.py:418-422，默认 OFF monitor.py:107），run 再校验救不回
   config = load_cache_config(yaml)
   config = _enforce_runtime_write_policy(config)         # D2: fail-fast 校验 yaml 已是 never；非 never 直接 raise（与真 server 同，不改写）
   config = <force config.timer.enabled=True>             # 必要条件之二（与上面 level≥BASIC 共同决定 timer 真记录：SystemTimer(enabled=config.timer.enabled) 且 level≥BASIC）
   validate_cache_config(config)
   _reject_warmup_calibration(config)                     # composite + samples_source=warmup → harness fail-loud（infra 无 WarmupPool 链路，§1）
   components = build_cache_components(config)             # 和真 server 同源；此刻构造 SystemTimer（读已升好的 level + config.timer.enabled）
   orchestrator = CacheOrchestrator(**components)          #   9 key（storage/key_builder/gates/judges/search_strategies/timer/write_policy/offline_writers/library_stats）全对应构造参数
   self._timer = components["timer"]                       # 持有同一 timer 引用（per-step 切片用）；此时 enabled=True
   _validate_artifact_enrichment(config, components["storage"])  # offline factor keys 须在 pkl payload.factors；缺→harness raise（生产 backend 不会自己报：回退 library_stats / offline factor 出 NaN，§1）
   # timing self-check（SystemTimer 无 public `enabled`，仅私有 `_enabled` timing.py:432；故用一次真实 probe 验证 timer 确实记录，不碰私有）：
   self._timer.register_probe("__timing_selfcheck__", backend="cpu")   # 先注册避免 unregistered-probe RuntimeWarning（timing.py:544-558）
   with self._timer.measure("__timing_selfcheck__"): pass
   self._timing_ready = ("__timing_selfcheck__" in self._timer.summary(task_only=True))   # measure 在 disabled 时 no-op→summary 无此 key
   if not self._timing_ready: raise RuntimeError("SystemTimer is a no-op (monitor<BASIC or config.timer.enabled=False); cannot collect latency")
   │
   ▼
ReplayHarness.run(H5EpisodeSource, repeats, out_csv):
   assert self._timing_ready                              # __init__ 的 timing self-check 已用真实 probe 验证 timer 记录（SystemTimer 无 public enabled，harness 持自有标志，不碰 timer._enabled）
   orchestrator.on_task_begin()                           # 整个 run = 一个长连接 task
   for r in range(repeats):
     for ep in source.iter_episodes():                    # 一个 H5 文件 = 一个 episode
       orchestrator.on_episode_start(ep.task_key, ep.episode_id)
       for step in ep.steps:                              # 不跳任何 step（gate-skip 也走 check）
         fake_s1 = _build_fake_stage1(step.group)         # CPU float32
         timer.on_task_begin()                            # per-step marker（仅动 timer，不碰 orchestrator）
         t0 = perf_counter()
         res = orchestrator.check(CP1, stage1=fake_s1)    # 跑满 6 段 → timer 记 6 条
         cp1_total = perf_counter() - t0                  # 真 t1（含段间 Python 胶水）
         action = _replay_action(res, step.clean_action)  # FULL_HIT→cached(等价); MISS→H5 clean_action(真 server eval 随机重采→读 action-history 的 judge run-to-run 异); WARM_START→clean_action(类型错误近似); §6.4
         orchestrator.broadcast_action(action)
         orchestrator.buffer_for_write(res.query_keys, action)
         orchestrator.clear()
         row = label(r, ep, step, res, cp1_total) + timer.summary(task_only=True)   # 切出这步 6 段
         writer.write(row)
       orchestrator.on_episode_end()                      # 无参！success 仅 replay metadata（orchestrator.on_episode_end 无 success 形参，interceptor 亦 discard）
   orchestrator.on_task_end()
   │
   ▼
summarize.py: per-step CSV → 聚合 json（per 段 × hit_type 的 count/median/p50/p95）
```

**per-step 切片原理**（零改 src）：`SystemTimer` 每次 `measure()` append 一条 `TimingRecord`（`timing.py:565-571`）。每 step 前调 `timer.on_task_begin()` 把 task marker 移到当前位置（`timing.py:580-590`），`check()` 内部写入 6 条，`timer.summary(task_only=True)` 只返回这 6 条（`timing.py:698-712` → `_get_task_records`）。每 probe count=1，其 `mean_ms`=`p50_ms`= 该步实测值。这套用的全是 SystemTimer 公开 API，不读私有、不改 src。

> 注：`cp1_sum` 探针是 **interceptor 层**加的（`interceptor.py:657`），infra 不经 interceptor，故没有该探针；t1 总值由 infra 自测的 `cp1_total`（`perf_counter` 包 `check()`）给出，6 段之和应 ≈ `cp1_total`（差值 = 段间 Python 胶水）。

---

## 4. 模块与文件清单（files touched — 全为新建，零改 src/）

| 文件 | 职责 |
|------|------|
| `exp/cache_latency_bench/__init__.py` | 包标记 |
| `exp/cache_latency_bench/h5_episode.py` | `H5EpisodeSource`：扫 H5 目录 → 逐 episode/step yield `(fake_stage1, clean_action, meta)` |
| `exp/cache_latency_bench/replay.py` | `ReplayHarness`：按 yaml 组装 orchestrator + 生命周期驱动回放 + per-step 计时切片 + 写 CSV |
| `exp/cache_latency_bench/run.py` | tyro CLI 入口（`--cache-config/--h5-dir/--repeats/--out-dir/--device`） |
| `exp/cache_latency_bench/summarize.py` | per-step CSV → 聚合统计 json |
| `exp/cache_latency_bench/README.md` | 用法 + 已知偏差（D1 的 build 段 caveat）说明 |
| `tests/exp/test_cache_latency_bench.py` | 单元 + 集成测试（见 §8，全 CPU、non-manual、CI 可跑） |
| `docs/experiments/cache_latency_bench.md` | 实验 runbook；同 commit 更新 `docs/README.md` 索引（§9） |
| `logs/README.md` | 加本 plan 条目（已在本提交内同步） |

**复用（不新建、不修改）**：`load_cache_config` / `validate_cache_config` / `build_cache_components`（`src/openpi/cache/config.py`）；`CacheOrchestrator`（`orchestrator.py`）；`CheckpointID`（`types.py`）；`SystemTimer`（`timing.py`）；`_FakeStage1` / `_build_fake_stage1`（`exp/common/build_in_memory_cache_artifact.py`）；`_enforce_runtime_write_policy`（`scripts/serve_policy.py`）。

> `_build_fake_stage1` / `_FakeStage1` 当前是 `exp/common/build_in_memory_cache_artifact.py` 的私有名。infra 在 `exp/` 内部 import 复用（受控的跨模块私有耦合，R6 已登记）；不触 `src/`。

---

## 5. 接口（interfaces）

```python
# h5_episode.py
@dataclass
class StepInput:
    group: "h5py.Group"            # 原始 step group（喂给 _build_fake_stage1）
    clean_action: torch.Tensor     # [H, action_dim]，MISS 路径喂给 broadcast/buffer
    step_idx: int

@dataclass
class EpisodeInput:
    path: str
    task_key: str                  # 来自 H5 root attr "task"
    episode_id: int                # 来自 H5 root attr "episode_id"
    success: bool                  # 来自 H5 root attr "success"
    num_steps: int
    def steps(self) -> Iterator[StepInput]: ...

class H5EpisodeSource:
    def __init__(self, h5_dir: str): ...          # 扫 *.h5，按文件名排序
    def iter_episodes(self) -> Iterator[EpisodeInput]: ...
    def __len__(self) -> int: ...                 # episode 数

# replay.py
class ReplayHarness:
    def __init__(self, cache_config_path: str, *, device: str = "cpu", quiet: bool = True): ...
    def run(self, source: H5EpisodeSource, *, repeats: int = 1, out_csv: str) -> dict: ...
        # 返回聚合统计 dict；同时把 per-step 行写入 out_csv

# run.py (tyro Args)
@dataclass
class Args:
    cache_config: str       # 真实 cache yaml（决定组装与工作方式）
    h5_dir: str             # 真实 trajectory H5 目录
    out_dir: str            # 输出根目录
    repeats: int = 1        # 目录整体重复次数（默认 1）
    device: str = "cpu"     # D1：默认 cpu（GPU 模式留作未来扩展接口）
```

**CSV schema**（per-step，精简）：
`repeat, episode_id, step_idx, task, hit_type, top_score, cp1_collect_ms, cp1_gate_ms, cp1_build_ms, cp1_search_ms, cp1_judge_ms, cp1_fetch_ms, cp1_total_ms, warm_start_action_approx, action_history_approx_active`
（gate-skip 步：search/judge/fetch 留空；MISS 步：fetch 留空。两标志仅在 yaml judge **读 action-history** 时有意义，否则恒 false。`warm_start_action_approx`=true 仅当**该步本身**为 WARM_START（喂 clean_action 冒充部分去噪，类型错误）；`action_history_approx_active`=true 是**累计标志（cumulative，非窗口精确）**：同 episode 内、本步之前 broadcast 过任何**非-FULL_HIT** action（MISS resample 或 WARM_START）即置 true 并保持到 episode 末（`on_episode_start` 重置）——保守提示「本步 verdict 的 action-history 可能含非-真-server-当次 action」，**不**精确追踪该近似 action 是否仍在 `history.actions[-P:]` 窗口内（要精确需按各 online action factor 的最大 P 做滚动窗口；本 plan 取保守 cumulative）。见 §6.4。）

**聚合 json**：`{probe: {hit_type: {count, median, p50, p95}}}` + 全局 `{n_episodes, n_steps, repeats, hit_rate, judge_consumes_action_history}`（末项=true 时，该 run 的 online-action-factor verdict 属「采集-trajectory 回放」、非真 server eval 逐调用等价，§6.4/§7）。

---

## 6. 已知偏差与局限（必须在 README + CSV 头显式标注）

1. **build 段不含 D2H（D1=CPU 的直接后果）**：真 server 中 `KeyBuilder.build()` 在 GPU 上 reduce 后 D2H 到 CPU（含一次 CUDA 同步）；CPU 模式下输入已在 CPU，`.cpu()` 为 no-op，故 `cp1_build` 偏低、**不可与真 server 的 build 段绝对对齐**。其余 5 段（gate/search/judge/fetch + collect）为纯 CPU 逻辑，真实。报告/CSV 须标注 `device=cpu, build_excludes_d2h=true`。GPU 模式接口预留（`--device cuda`，Phase-2 可补，本 plan 不实现）。
2. **绝对延迟与硬件强相关**：CPU 型号/负载影响绝对值；结论应用于「段间相对占比」「随库规模/depth 的标度」「in-DB vs out-of-DB 的 fetch 段差异」，而非跨机绝对毫秒。
3. **t1 ≠ 端到端**：infra 只测 cache 段；模型推理段不测（设计如此）。
4. **broadcast action 的保真度（影响读 action-history 的 composite judge）**：`broadcast_action` 喂入的 action 进 `orchestrator._action_history`，仅被 **composite judge 的 online action-channel factor** 读取（`online.py:158-199`）；对 `threshold`/`always_hit`/`always_warm_start` judge 及只用 state-channel factor 的 composite judge **无影响**（它们不读 action-history，FULL_HIT/MISS/WARM_START 喂什么 action 都不改 verdict/延迟）。对**读 action-history 的 judge**，infra 忠实回放采集 trajectory 的 action 序列，但与真 server eval 当次的关系分三档：
   - **FULL_HIT**：infra 喂 cached `payload.action_chunk`（=库 entry，即 H5 `clean_action`）；真 server FULL_HIT 也喂 cached → **等价**。
   - **MISS（resample 差异）**：infra 喂 H5 `clean_action`（采集时一次真实 MISS 输出样本）；真 server eval MISS 用**新随机 noise** 重采 stage3 得到不同 action（`pi0_pytorch` `sample_noise`→`torch.normal` 全局 RNG，**无固定种子**；`interceptor.py:768-786`）。∴ infra 确定值 vs 真 server 随机值，**run-to-run 不等**（分布内样本，非类型错误）。
   - **WARM_START（类型错误近似）**：真 server 喂模型从 cached `x_t` **部分去噪**的 action（`interceptor.py:721-737, 800-819`），H5 无此量，infra 以**完整去噪**的 `clean_action` 充当 → 类型错误，比 MISS 更严重。
   harness 检测「judge 消费 action-history」时 fail-loud 警告，并标注：per-step `warm_start_action_approx=true`（WARM_START 步）；**累计标志** `action_history_approx_active=true`（cumulative：同 episode 内本步之前 broadcast 过任何**非-FULL_HIT** action——MISS resample 或 WARM_START——即保持 true 到 episode 末，`on_episode_start` 重置；保守提示，**不**精确追踪近似 action 是否仍在 `[-P:]` 窗口内）；run 级 json `judge_consumes_action_history=true`（整个 run 的 online-action-factor verdict 属「采集-trajectory 回放、非真 server eval 逐调用等价」）。
5. **write_policy 契约（fail-fast，非改写）**：`_enforce_runtime_write_policy` 不会把 write-enabled config 静默改成 never，而是对任何非 `never` 的 yaml 直接 `raise ConfigValidationError`（`serve_policy.py:26-48`，docstring 明写 "fails fast"）。这正是真 server runtime 的 C2 契约。**因此 infra 要求输入 yaml 本身已声明 `write_policy: {type: never}`**（与「能在真 server 上跑起来的 yaml」是同一前提）——harness 不复制改写 config，只复用该函数做这道 fail-fast 校验。README/runbook 须写明此前提。
6. **仅驱动 CP1，CP3 段不计入**：harness 只调 `orchestrator.check(CP1)`，不调 `check(CP3)`。这**不破坏 CP1 等价性**（anchor 恒=CP1，CP1 的 gate/judge/strategy/score-memo 与 CP3 是独立 dict 项，CP3 session 开而不用无害不报错，`orchestrator.py:160-163, 291-299`）。但若 yaml 配了 CP3（`checkpoints.cp3.enabled`），其 CP3 段延迟**完全未测**，且真 server 单 step MISS 后含一次 `check(CP3)` 的 CPU 开销（`interceptor.py:794`）不计入 `cp1_total`——故 `cp1_total` ≠「含 CP3 的真 server 单 step cache 总延迟」。README/CSV 头须标注「仅 CP1」。

---

## 7. 真 server 等价性论证（本 infra 的正确性核心）

infra 用**同一个生产 `CacheOrchestrator` + 同一套 yaml 组装的组件**，并按真 server（`InferenceInterceptor`）对 orchestrator 的**完全相同调用序列**驱动：

| 真 server 时机 | 调用 | infra 镜像 |
|---|---|---|
| 连接 open | `on_task_begin()` | run 开始一次 |
| episode 开始 | `on_episode_start(task,ep_id)` | 每 H5 一次 |
| 每 step | `check(CP1,…)` → FULL_HIT: `broadcast_action(cached)`+…；MISS/WARM_START: 真 server `broadcast_action(stage3.action_chunk)`+… | FULL_HIT→cached(**等价**)；MISS→H5 `clean_action`(读 action-history 的 judge: 真 server 随机重采→run-to-run 异)；**WARM_START→clean_action(类型错误近似)**；§6.4 |
| episode 结束 | `on_episode_end()`（**无参**） | 每 H5 一次；`success` 仅 replay metadata，不传 orchestrator |
| 连接 close | `on_task_end()` | run 结束一次 |

唯一差异 =「stage 输出来自 H5 回放」「不跑 stage2/3 GPU」。query_keys、state-history（来自 `query_keys["robot_state"]`）、库、score-memo/search-session/step_counter 全程真实。等价性按「judge 是否读 action-history」分两类：

- **(i) judge 不读 action-history**（`threshold` / `always_hit` / `always_warm_start` / 只用 state-channel factor 的 composite）—— **逐调用等价**：cache 判定只依赖 query_keys + 库 + state-history（全真实），与 broadcast 的 action 数值无关，FULL_HIT/MISS/WARM_START 喂什么 action 都不改 verdict 与延迟。
- **(ii) judge 读 action-history**（composite online action-channel factor）—— **非逐调用等价，是「采集-trajectory 忠实回放」**：infra 喂采集 trajectory 的真实 action 序列——FULL_HIT 步=cached（等价）、MISS 步=采集样本（真 server eval 随机重采、run-to-run 不等，§6.4）、WARM_START 步=类型错误近似（§6.4）。harness fail-loud + `warm_start_action_approx`/`action_history_approx_active`/`judge_consumes_action_history` 三标志，**不**声称逐调用等价。

**四条必守正确性约束**（否则记忆失真）：
- (a) 库填充（`load_artifact`，在 `build_cache_components` 内经 `backend.preload_path` 完成）发生在 `on_episode_start` **之前**——否则 session 活动期 insert 撞 `SearchSessionActiveError`。
- (b) **不跳任何 step**：gate-skip 步也走 `check()`（其内部 `record_query_keys` 保 trajectory 窗口 gap-free）。
- (c) `broadcast_action` 必须在 `check()` **之后**。
- (d) 每 episode 之间 `on_episode_start`/`on_episode_end` 成对，session 正确开/关。

---

## 8. 测试策略（test strategy — 全 CPU / non-manual / CI 可跑）

合成 mini 数据（不依赖 gitignored 的大 pkl/h5）：构造一个 ~3 entry 的 InMemoryBackend pkl + 一个 ~4 step 的合成 H5（含 vision_0/1/2、prompt_emb、robot_state、clean_action），覆盖：

1. **H5 解析**：`H5EpisodeSource` 正确 yield episode meta（task/episode_id/success/num_steps）+ 每 step `_build_fake_stage1` 形状 `prefix_embs=[1,L,2048]`/`state=[1,32]`。
2. **按 yaml 组装 + write_policy fail-fast 契约**：load 一个 `write_policy:never` + always_hit + mean_pool yaml → 组装出 `CacheOrchestrator` 非空；另用一个**非-never** yaml → harness 在 `_enforce_runtime_write_policy` 处 `raise ConfigValidationError`（D2 fail-fast 契约，验证非静默改写）。
3. **生命周期顺序**（spy orchestrator）：`on_task_begin → [on_episode_start → (check→broadcast_action→buffer_for_write→clear)×steps → on_episode_end]×ep → on_task_end`，且 broadcast 在 check 之后。
4. **per-step 切片正确**：每 step CSV 行的段数 = 该步实际走过的段（SEARCH 步 6 段、gate-skip 步 3 段）；`cp1_total ≥ Σ六段 - ε`。
5. **库冻结**：跑完整 run 后 `storage.count()` 不变（D2：on_episode_end 不写库）。
6. **in-DB vs out-of-DB 行为**（呼应需求，judge 用 `threshold`）：喂一条库内 entry 的 query → score≈1≥threshold → FULL_HIT 且 `cp1_fetch` 有值；喂一条扰动/随机向量 → score<threshold → MISS 且 `cp1_fetch` 为空。证实「fetch 段是否发生」随命中率分叉、而 search 段两者都执行。（**不能用 `always_hit`**：它对非空 results 永远 FULL_HIT，brute-force 对随机 query 仍返候选 → 无法分叉，`judge.py:143-145`。）
7. **monitor level auto-elevate（保证有数据）**：用 `monitor.set_monitor_level(MonitorLevel.OFF)`（**不是**改环境变量——`get_monitor_level()` 进程级缓存 `_LEVEL`，改 env 在首次 access 后无效，`monitor.py:126-142`）→ 构造 `ReplayHarness` → 断言 `_ensure_timing_enabled` 已把 level 提到 ≥BASIC、且 harness 的 timing self-check 置 `harness._timing_ready==True`（**经一次真实 `measure()`+`summary()` probe 验证 timer 确实记录**——`SystemTimer` 无 public `enabled`、仅私有 `_enabled` `timing.py:432`，故**不**断言 `_timer.enabled`）。用例 `finally` 里 `set_monitor_level` 复位，避免污染其他用例。
8. **repeats 可比**：repeats=2 下两遍同一 episode 的 per-step 段序列结构一致（库冻结 + 每 episode 记忆重置 → 同初始条件）。
9. **broadcast action 保真度标注**：用一个会产生 WARM_START 且 judge 消费 action-history 的 yaml（**`type: composite`** judge + 读 `history.actions` 的 online action-channel factor + composer 配 `warm_start_threshold`/`warm_start_t` 出 WARM_START；注：judge type 互斥，`always_warm_start` 与 `composite` 不可并存，`config.py:441` / `:2077-2106`）→ 断言：该 WARM_START 行 `warm_start_action_approx=true`、其前出现过 MISS 的后续行 `action_history_approx_active=true`、run 级 `judge_consumes_action_history=true`、harness 发 fail-loud 警告。对照用例：`always_warm_start` 或 `threshold`(warm_tiers) 的 WARM_START（judge **不读** action-history）→ 三标志均不置（恒 false）。**mini 数据构造 5 约束见 §10。**
10. **on_episode_end 无参契约**：spy 断言 harness 调 `orchestrator.on_episode_end()` 不带 `success`（带参会 TypeError，`orchestrator.py:551`）。
11. **运动学 composite verdict 支持范围（harness 自校验）**：(a) enriched mini pkl（含所需 offline factor 的 `payload.factors` keys）+ `composite` judge（offline + online state/action factor）+ `samples_source.type=offline` → 跑通，`cp1_judge` 非零、verdict 三态正常；(b) **未 enrich** mini pkl——含两子用例：**全部 entry 缺** offline keys、以及**仅部分 entry 缺**（其余齐备）——均须 `_validate_artifact_enrichment` `raise`（**全量遍历**，不抽样；部分缺也要抓到），消息指向 `factor_postprocess.py`（注：直接喂生产 backend **不会**自己报——会回退 library_stats / offline factor 出 NaN，故须断言 **harness 自己的**校验报错）；(c) `samples_source.type=warmup` → `_reject_warmup_calibration` `raise`，断言 **harness 自己的**消息含「改用 offline」（而非下游 `config.py:2219` 的通用消息）。

Verify：`uv run pytest tests/exp/test_cache_latency_bench.py` 全绿；并跑一次真实 smoke（`libero_10` 的一个 mean_pool yaml + 一个 H5，本地 CPU）确认端到端产出 CSV+json（smoke 作为 manual 验证记录，不入 CI）。

---

## 9. 风险登记（risk register）

| # | 风险 | 等级 | 缓解 |
|---|------|------|------|
| R1 | build 段缺 D2H 被误读为"真 server build 延迟" | 中 | §6 显式标注；CSV 头写 `build_excludes_d2h=true`；README 警示；GPU 模式接口预留 |
| R2 | per-step 切片依赖 `on_task_begin/summary` 语义，未来若 orchestrator 在 `check` 外产生 record 会污染切片 | 低 | 测试4 断言每步段数；切片紧贴 `check` 前后 |
| R3 | `write_policy=never` 下 composite judge 的 OfflineWriter 仍可能在 `on_episode_end` 改库 | 中 | 测试5 断言 `count()` 不变；若发现 writer 仍写，harness 额外用 backend frozen guard 兜底（C2 已在 ABC `__init_subclass__`，runtime insert 会抛 `BackendFrozenError`） |
| R4 | `OPENPI_MONITOR_LEVEL<BASIC` 致 SystemTimer 构造为 no-op，静默无数据 | 中 | harness `__init__` **构造 timer 前 auto-elevate** 到 ≥BASIC（`_ensure_timing_enabled`）+ 构造后 timing self-check（真实 `measure()`+`summary()` probe 验证记录，不碰私有 `_enabled`）；测试7 覆盖 |
| R5 | yaml keybuilder 维度与 H5 token 布局不符（如 spatial_pool 期望特定 token 数）→ build 报错 | 低 | fail-fast + 清晰错误信息；README 说明 H5 token 布局（vision 256×3 + prompt） |
| R6 | 复用私有 `_build_fake_stage1` 的跨模块耦合 | 低 | §4 备选：提升为 `exp/common` public helper（受控 deviation） |
| R7 | `cp1_llm_layer_extract` keybuilder 需真模型，与"不加载模型"冲突 | 低 | harness fail-fast 拒绝该 keybuilder（与用户"不看 CLIP/不加载模型"一致）；README 列明支持的 keybuilder 白名单 |
| R8 | 大 H5（单文件百 MB 级）逐 step 读盘成为基准噪声 | 低 | H5 读盘在计时窗口**之外**（`_build_fake_stage1` 在 `timer.on_task_begin` 之前完成）；只对 `check()` 计时 |
| R9 | 读 action-history 的 composite judge：infra 的 broadcast action 与真 server eval 当次不同——**MISS** 真 server 随机 noise 重采（infra 喂采集样本，run-to-run 异）、**WARM_START** 真 server 喂部分去噪 action（infra 喂 clean_action，类型错误）；经 `history.actions` 扩散影响后续 verdict | 中 | §6.4/§7 把这类 judge 收窄为「采集-traj 回放、非逐调用等价」；harness fail-loud + `warm_start_action_approx`(WARM_START 步) + `action_history_approx_active`(**cumulative** flag：本步之前出现过非-FULL_HIT action) + run 级 `judge_consumes_action_history`；测试9 覆盖 WARM_START + MISS resample 标注 |
| R10 | 运动学 composite verdict 前提不满足：pkl 未 enrich（缺所需 offline factor `payload.factors` keys）或 yaml 用 `samples_source.type=warmup`——**生产 backend 不会自己启动期报错**（缺 library_stats→回退计算 `in_memory_backend.py:263-283`；缺 factors→offline factor 出 NaN `offline.py:105-115`），可能静默跑成 NaN/MISS、延迟失真 | 中 | infra **不依赖生产路径**：harness `_validate_artifact_enrichment`（offline keys 缺→raise）+ `_reject_warmup_calibration`（warmup→raise，引用 `config.py:2218-2234`），均 harness 自己 fail-loud；测试11(b)(c) 断言 harness 消息 |

---

## 10. 实施步骤（G1 APPROVED 后）

1. `h5_episode.py`：`H5EpisodeSource` + dataclasses + 复用 `_build_fake_stage1`。
2. `replay.py`：`ReplayHarness` 组装（load→enforce never→validate→build→orchestrator）+ 回放循环 + per-step 切片 + CSV writer。
3. `run.py`：tyro CLI。
4. `summarize.py`：CSV → 聚合 json。
5. `tests/exp/test_cache_latency_bench.py`：§8 全部用例（合成 mini 数据）。
6. `README.md` + `docs/experiments/cache_latency_bench.md` + `docs/README.md` 索引同步。
7. §8 Verify（pytest）+ 真实 smoke 记录。

**测试9 / 11(a) 的 mini-data 构造 checklist**（composite + WARM_START + online action factor，§8 最复杂用例，5 约束缺一即 build raise 或静默降级 MISS）：
1. **calibration 样本量**：yaml `calibration.params.window_size` 设极小（如 1），offline jsonl 为**每个** factor key 写 ≥window_size 行 `factor_raw`（否则 `bind_keys` fail-fast，`percentile_rolling.py:71-83`）。
2. **`history.actions ≥ P`**：WARM_START 须在 episode 第 ≥P+1 步（前序 step 经 `broadcast_action` 喂够 `_action_history`）；取 **P=1,F=1** 最省（`online.py:159`）。
3. **`walk_next(winner,k=F)` 返回 F 个 entry**：mini pkl 须建 `prev_ids/next_ids` 链且 winner 非链尾（`payload_view.py:146-189`）；3~4 entry 链够。
4. **WARM_START 不被降级 MISS**：winner payload 须带 `intermediates={warm_start_t: tensor}` + `denoising_num_steps`，composer `warm_start_t` ∈ intermediates keys（`orchestrator.py:497-511`）。
5. **composer 出 WARM_START**：`weighted_sum`(带 `warm_start_threshold`+`warm_start_t`) 或 `weighted_sum_with_warm_fallback`（`config.py:2330-2353`）。
（mini pkl/h5 格式见 §8 引言；enriched 变体在 entry `payload.factors` 填 offline keys。）

---

## 11. 文档与索引同步（constitutional）

- 新建 `docs/experiments/cache_latency_bench.md` → 同 commit 更新 `docs/README.md` experiments 段。
- 本 plan 落 `logs/` → 同 commit 更新 `logs/README.md`（已在本提交同步）。
- 数据产物（CSV/json）落 `exp/cache_latency_bench/data/`，按 `artifact_layout.md` gitignore（本地保留，不入库）；代码 + `analysis/` + README + plan log 入库。

---

## Review Log

### G2 Round 1 — Reviewer — APPROVED — 2026-05-30 21:08 CDT

- [Resolved] Code matches the approved plan's main contract: `ReplayHarness` loads the real cache YAML, reuses `_enforce_runtime_write_policy` for the server-equivalent `write_policy=never` fail-fast gate, forces the timer config on, rejects warmup calibration, validates composite/offline-factor enrichment, builds real cache components, and drives a real `CacheOrchestrator` through the CP1 replay lifecycle (`exp/cache_latency_bench/replay.py:228-301`). The CSV schema and summary/meta outputs cover the agreed per-step probes, `cp1_total`, action-history approximation flags, `judge_consumes_action_history`, `build_excludes_d2h`, and `cp1_only` metadata (`exp/cache_latency_bench/replay.py:61-77`, `exp/cache_latency_bench/run.py:56-67`).
- [Resolved] Test coverage is sufficient for G2: the implementation includes synthetic CPU tests for H5 parsing, write-policy fail-fast, lifecycle ordering, timing segment slicing, frozen-library behavior, hit/miss fetch divergence, monitor auto-elevate, repeats, WARM_START/action-history flags, no-arg `on_episode_end`, composite enrichment validation, warmup calibration rejection, and summary aggregation (`tests/exp/test_cache_latency_bench.py`). Reviewer run: `PYTHONPATH=. uv run pytest tests/exp/test_cache_latency_bench.py -q` → 18 passed. Reviewer real smoke: one `libero_10` mean-pool YAML plus one real H5 episode → 55 steps, `per_step.csv` + `summary.json` produced successfully under `/tmp/cache_latency_bench_g2_out`.
- [Resolved] Documentation and indexes are synchronized for the new subsystem: `exp/cache_latency_bench/README.md` and `docs/experiments/cache_latency_bench.md` document usage, support range, and known deviations; `docs/README.md` and `logs/README.md` have matching index entries.
- [Non-blocking] [Suggestion] Add an explicit startup guard for `key_builder.type == "cp1_llm_layer_extract"` in the harness, even though the mode is documented as unsupported — reasoning: the approved risk register says the harness should fail-fast reject that keybuilder (`logs/cache_latency_bench_plan.log.md:249`), while the current constructor path has no explicit check before `build_cache_components` / replay (`exp/cache_latency_bench/replay.py:237-247`). In practice the unsupported path will fail later because `CP1LLMLayerExtractKeyBuilder` needs model attachment and full Stage1 mask fields (`src/openpi/cache/components/llm_layer_key_builder.py:61-79`, `:175-180`), but an early `NotImplementedError` would make the unsupported contract cleaner. This is non-blocking because the supported keybuilders, docs, tests, and real smoke all cover the delivered benchmark path.
