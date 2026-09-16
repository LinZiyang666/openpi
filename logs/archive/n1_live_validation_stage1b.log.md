# N1 分数滞回门 — Live 验证实现计划（Stage 1b）

- **Status**: Done — live 完成（L2 全流程收官：G1/G2 APPROVED、Verify 797 pass、committed；live 8 run × 500 ep 于 2026-07-05 完成，vs-periodic 及格线 4/4 FAIL 但 N1 vs baseline SR 保真 + net 全正，判决见 `exp/gate_research/analysis/n1_live_results.md` 与 roadmap F9–F12/C10/C11；待 owner 确认后归档）
- **Date**: 2026-07-04
- **Authority**: Execution
- **Level**: L2（新增 correctness-critical 有状态 gate 组件，落 `exp/gate_research/`；L2 依据=正确性关键性，见 §2）
- **上游**: `logs/gate_exploration_roadmap.log.md` §5 Stage 1b；离线前沿结果 `exp/gate_research/analysis/n1_offline_frontier.md`
- **前置产物（已完成）**: Stage 1a 离线扫描（`exp/gate_research/n1_offline_scan.py`）已给出每 config 的 N1 操作点 A（近免费, lost≤1%）与 B（平衡, lost≤4%）；Stage 0 的 `always_search` 500ep×2suite 真 verdict + SR 已采集（`exp/gate_research/data/<suite>/journal.jsonl`），作为配对基线。

---

## 1. 目标与验收

**目标**：把 Stage 1a 离线选出的 N1 操作点在真实闭环 rollout 里跑起来，测量它对成功率（SR）和 inference_ratio 的**真实、可观测**影响——离线 dInf 是"冻结 verdict 序列"的一阶近似（roadmap 公理 C8），错跳命中步会改变执行动作（缓存回放 → 新推理）从而改变轨迹，只有 live 能测 SR。

**N1 规则**（与 1a 同）：客户端维护"最近一次搜索步的 `cp1_score`"；连续 `j` 个已搜索步 `score < θ_low` → 停搜（该步走全推理）；停搜期每 `M` 步 probe 一次；probe `score ≥ θ_high` → 恢复（双阈值滞回）。θ 锚定各 config 的 baked `(ws_thr, fh_thr)`。

**可观测性边界（G1R1-2 修订）**：一旦客户端发 `skip`，server 不搜索 → 该步**没有** always-search 下本会得到的 FH/WS/MISS 反事实标签。故 **live 无法测 `lost%`**（"错跳了多少命中步"需要反事实标签）。live 只报告可观测量：
- **actual SR**（journal 逐 episode 成功）、**skip%**（`searched=False` 步占比）、**actual inference_ratio**（skip 步=全推理 inf 1.0；searched 步按其 `hit_type` 的 inf_value）、**searched-step verdict mix**（仅 searched 步的 FH/WS/MISS 分布）。
- 跳过步的 `hit_type=MISS/cp1_score=None` 是**占位**，分析**绝不**把它当真 MISS verdict（只计为 skip）。lost% 仅在离线（1a，有全标签）成立；live 不报 lost%。可选 shadow-search（skip 步照搜只为拿标签、不改动作）能测 lost 但引入成本/隔离风险，本计划**不默认启用**，仅登记为备选（§8）。

**验收（pass line，roadmap §5 / RPG 坐标系 C6）**：
- **主判据（SR 保真）**：同 config、同 inits 配对下，N1 SR ≥ 该 config `always_search` 基线 SR − 1pp（A 点应几乎无损；B 点检验其 skip 是否引起可见 SR 退化——C8 反事实的真检验）。配对 McNemar 或差值 + Wilson CI。
- **预算判据（打败 periodic，roadmap 强制，G1R1-3 修订）**：**必做** matched-budget periodic 对照——同 skip%（±容差）下 N1 SR ≥ periodic SR。匹配规则见 §6。
- **延迟（C9 三档）**：`net = skip% × SEARCH_MS − Δinf × INFER_MS`，`SEARCH_MS∈{4,34,70}`、`INFER_MS=300`、`Δinf = live_inf_ratio − baseline_inf_ratio`（均**实测**；baseline_inf_ratio 由 analyzer 从 Stage-0 同 config `gate_rows.jsonl` 算，§9 G1R5-3）。
- **live-离线一致性（可选诊断，非 pass-gate；G1R3-5 修订）**：可选地比 live 实测 `skip%`/`searched-step verdict mix` vs 离线预测——该预测由 analyzer 用**共享的可导入 `N1GateState`** 在 Stage-0 `gate_rows.jsonl`（config 过滤 + 去重）上重放产生（与 live 同一状态机，算法自洽）。仅作诊断（系统性偏离 → 离线前沿降粗筛），**不进 pass line**；输入/算法/测试见 §9。**不比 lost%**（不可观测）。

---

## 2. Level 判定（L2）与组件归属（G1R1-4 修订）

**组件归属 = `exp/gate_research/`**（本实验专用机制，按 artifact-layout 默认）。前一版把 N1 放 `examples/libero/` 的依据（"worker 禁 import exp.*"）**系误读**：`agent.py:17` 的 `MUST NOT import exp.*` 约束的是 **agent.py 本身**（src 不依赖 exp），worker 入口由 `WorkerSpec.worker_module` **字符串**命名、spawn 把 repo root 加入 `PYTHONPATH`（`agent.py:113`），故 `exp.gate_research.worker_entry_n1` 作为 worker module **合法可导**（已核 `exp/__init__.py`、`exp/gate_research/__init__.py` 存在；worker conda env 现已能导 `openpi.conductor`/`examples.libero`）。→ **全部新文件落 `exp/gate_research/`，零 src/ 零 examples/ 改动**。

**Level = L2 的依据（非落位，而是正确性关键性）**：N1 是**新增有状态 gate 决策组件**，其输出**改变 live rollout 的执行动作**、且实验结论直接决定 roadmap Stage 2 的 go/no-go。爆炸半径 = **实验有效性**（gate 逻辑 bug → SR 被污染 → 路线图误判），足以要求 G1/G2 独立审查。据 WA「存疑不降级」，即便代码落位已收敛到 exp/（无代码库爆炸半径），仍保 L2 全流程（Plan→G1→Code→G2→Verify），不因落位收敛而降级跳 G2。

---

## 3. 已亲验的架构契约（file:line）

server 端（**不改**）：
- 门：`src/openpi/cache/components/gate.py:242` `ClientControlledGate.__call__(ck, cached_data, request_context)`，读 `request_context["gate_decision"]`，`"skip"→False`/`"search"→True`，缺失/非法 raise。
- 拦截器：`src/openpi/cache/interceptor.py:636` `client_signal = obs.pop("__gate_decision__", None)` → `request_context={"gate_decision": client_signal}`（`interceptor.py:647-649`）；`accepts_client_signal` 不匹配则 fail-loud（`interceptor.py:641-646`）。
- 回传：`interceptor.py:481-513` `_build_hit_meta` → `{"hit_type","start_t","winner_id","cp1_score"}`，挂 `outputs["__hit_meta__"]`（`interceptor.py:734, 889`）；`cp1_result is None`（未搜索）→ `cp1_score=None`（`interceptor.py:492-498`）。
- 配置：`gate.type: client_controlled` 属"legacy 3 类"，**不接受额外字段**（`config.py:1308-1315`, `config.py:2142-2145`）。

client 端（**注入点**）：
- `packages/openpi-client/.../websocket_client_policy.py:47` `infer(obs)->dict`（返回含 `__hit_meta__` 的解包 dict）；`:56` `episode_start(experiment, task="", episode_id=-1, episode_name="", extra_metadata=None)`；`select_bundle`/`episode_end` 亦在此类。
- `examples/libero/episode_runner.py:78` `default_client_factory(server)->WebsocketClientPolicy(host,port)`；`:94-104` `LiberoEpisodeRunner(args, episode_setup, *, client_factory=default_client_factory, run_episode_fn=...)` —— **client_factory 是可注入 kwarg**；`:125` `self._client_factory(task.server)`；`:136` **每 episode 调 `client.episode_start(...)`**（N1 复位钩子）。
- worker 入口：`examples/libero/worker_entry.py:58` `LiberoEpisodeRunner(args, _build_episode_setup(...))`（用默认 factory）；入口模块由 `WorkerSpec.worker_module`（`agent.py:42`，默认 `"examples.libero.worker_entry"`）指定，**可换**；`_build_episode_setup` 可复用。
- worker spawn 的 `base_cmd`（`agent.py:79-93`）参数固定，**不含 N1 参数** → N1 θ 参数经**环境变量**传入 worker（子进程继承 env，conda spawn 亦保留非 VIRTUAL_ENV/PYTHONPATH/PYTHONHOME 的 env，`agent.py:101-104`）；repo root 在 `PYTHONPATH`（`agent.py:113`）→ `exp.gate_research.*` 可作 worker module。
- 每步 infer 在 `examples/libero/main.py:320` `_infer_result = client.infer(element)`；`__hit_meta__` 在 `main.py:327` 读到。N1 wrapper 在 `client.infer` 层做注入/读取，`main._run_episode` **不改**。

采集与记录（**关键**，G1R1-1/2 修订）：
- **绝不开 server collection**：`collection.export_collect_meta` 在 `config.py:1084-1099` 硬性要求 CP1 `always_search`（gate 跳步会造 C5 选择偏置）→ client_controlled run **不设** `--export-collect-meta`。
- **`searched` 记录走既有 recorder seam，零 src/examples 改动**：`main.py:333` 读 `_infer_result.get("__collect_meta__")` → `encode_collect_meta`（`examples/libero/collect_util.py`：对 `{"searched":bool}`（无 "collect" 键）→ `{"collect":None,"searched":True}`，干净）→ `infer_recorder`（`episode_runner.py:167-169`）写 `row["searched"]`。故 **N1 wrapper 客户端合成 `result["__collect_meta__"]={"searched": bool(decision=="search")}`**，`searched` 即落进 per_step_rows，无需 server、不触 C5。
- **`searched` 是唯一权威 skip 信号（G1R3-4 修订）**：`cp1_score is None` **不**等于 skip——searched MISS 在 search 结果为空时 `top_score=None`（`orchestrator.py:489, 531-534`）也回传 `cp1_score=None`。故 skip 判定**只**用客户端决策派生的 `searched`（wrapper 注入，权威）；`cp1_result is None` **不**等同"未搜索"。仅保留**单向** sanity：`searched==False ⇒ cp1_score is None`（反向不成立，不校验）。

---

## 4. 设计（全部新文件，零 src/examples 改动）

| 文件 | 动作 | 责任 |
|---|---|---|
| `exp/gate_research/n1_gate_client.py` | 新建 | **`N1GateState`**（纯状态机，可导入、import 无副作用、无 argv/IO）+ **`N1GateClient`**（包装 inner client）+ `make_n1_client_factory` + `n1_params_from_env`。核心组件。 |
| `exp/gate_research/worker_entry_n1.py` | 新建 | N1 worker 入口：复用 `examples.libero.worker_entry._build_episode_setup` + `main.Args`，构造 `LiberoEpisodeRunner(args, setup, client_factory=make_n1_client_factory(n1_params_from_env()))`，其余同 `worker_entry`。 |
| `exp/gate_research/run_n1_live.py` | 新建 | 1b runner，**按单个 yaml 的 gate.type 分派**（G1R5-2）：`client_controlled` → N1 worker（`worker_module="exp.gate_research.worker_entry_n1"` + 注入 `N1_THETA_LOW/HIGH/N1_J/N1_M`，θ 必填校验）；`periodic` → **默认 worker**（`worker_module="examples.libero.worker_entry"`，不注入 N1 env、不发 client signal），cache_len/inference_len 读自 yaml。两者写**同 schema run manifest**（§9）；`--yaml-dir` 断言恰一个 yaml；复用 run_collect conductor 骨架 + `--per-step-out` 增量 append。 |
| `exp/gate_research/config/<suite>/n1/*.yaml` | 新建 | 复制 Stage-0 对应 config eval yaml，仅 `gate.type: always_search → client_controlled`（其余 baked 阈值/normalizer/preload 不变）；**不设** collection。**首波 Code 只出 2 个 client_controlled yaml**（spatial fh75_ws10 + libero_10 fh5_ws40）；**periodic 对照 yaml 是第二波交付**（`cache_len/inference_len` 由第一波 live 实测 skip% 定，见 §6，G2R1-7 deferred）。 |
| `exp/gate_research/analyze_n1_live.py` | 新建 | 确定性分析器（§9）：读 per_step + journal + **Stage-0 baseline `gate_rows.jsonl`（C9 必需）** + run manifest → 按 task_uid 取**全局 max attempt** 再取该 attempt 全部 step（G1R5-1）→ SR/skip%/live inf_ratio/verdict-mix/C9 三档（baseline_inf_ratio 来自 Stage-0 per-step）+ 跨-run 配对 → 写 `analysis/n1_live_results.md` + result manifest。 |
| `tests/exp/test_n1_gate_client.py` + `tests/exp/test_analyze_n1_live.py` | 新建 | 见 §7：状态机 golden traces + provenance round-trip + 参数/异常/env/单-yaml/config 校验；analyzer 去重/跨-run 配对/journal 重复/periodic searched 重建/offline-replay 诊断。按 artifact-layout 落 `tests/<exp>/`（G1R3-6），非 manual。 |

**`N1GateState`（纯状态机，可导入，供 wrapper 与单测共用）**：
- 字段：`searching:bool, low_run:int, since_probe:int`；参数 `theta_low, theta_high, j, M`。构造即校验（见异常契约）。
- `decide() -> "search"|"skip"`：`searching` → `"search"`；否则 `M is not None and since_probe+1 >= M` → `"search"`（probe）else `"skip"`。
- `observe(decision, score)`：`score is None`（searched 步空搜索 = 合法 MISS，G1R3-4）按 `score=−inf`（必 <θ_low、必 <θ_high）处理。`searching` 分支——`score<θ_low`→`low_run+=1`，`low_run≥j`→`searching=False, since_probe=0`，else `low_run=0`；`skipping` 分支——`decision=="search"`（probe）→`since_probe=0`，`score≥θ_high`→`searching=True, low_run=0`，else 继续 skipping；`decision=="skip"`→`since_probe+=1`（不传 score）。
- `reset()`：`searching=True, low_run=0, since_probe=0`。首步 `decide()` 恒 `"search"`（无历史）。与 1a `n1_sim` 逐步同构（Mealy）；ws_aware（1a 判 0 增益）不进 live。

**`N1GateClient`（wrapper，WA §2.5 组合式）**：
```
N1GateClient(inner, state: N1GateState)
  infer(obs):
      decision = state.decide()
      out_obs = {**obs, "__gate_decision__": decision}      # 不原地改调用方 dict
      result = inner.infer(out_obs)
      score = _read_score(result, decision)                 # 见异常契约
      state.observe(decision, score)
      result["__collect_meta__"] = {"searched": decision == "search"}  # 记录 seam
      return result
  episode_start(*a, **k): state.reset(); return inner.episode_start(*a, **k)
  select_bundle/episode_end/close/__getattr__ -> inner
```

**异常契约（G1R1-5 / G1R3-3/4 修订）**：
- **构造期校验（fail-fast raise）**：`theta_low/theta_high` 为有限 float 且 `theta_high≥theta_low`；`j` 为 int≥1；`M` 为 None 或 int≥1。违反 → `ValueError`（任何 rollout **之前**，worker 启动即崩，正确暴露配置错误，不污染 SR）。
- **searched 步 + 有限 score**：正常 `observe(decision, score)`。
- **searched 步 + `cp1_score is None`（`__hit_meta__` 存在但 score None = 空搜索合法 MISS，G1R3-4）**：按 `score=−inf` 正常 `observe`（进 low_run / 不恢复）——**不是**异常，`logging.debug`。
- **searched 步 + 非有限 score（NaN/±inf）或 `__hit_meta__` 整块缺失（契约违反/数据异常）**：**fail-open 恢复全搜索**——强制 `searching=True, low_run=0, since_probe=0`，下一步必 `"search"`（decide 保证）；`logging.warning` 一次。**绝不 raise**（`main._run_episode` 的 `except Exception` 会把 raise 吞成 episode 失败、污染 SR），**绝不误跳**。两条 trace 分别验证"异常发生在普通 searching 步"与"发生在 probe 步"（G1R3-3：修掉 probe 后残留 `searching=False` 反而继续 skip 的矛盾）。
- **skip 步（decision=="skip"）**：`observe` 走 skip 分支（`since_probe+=1`），不读 score/不更新寄存器。
- **inner.infer 抛异常**：**不捕获**，向上传播（与 base client 一致，由 runner/episode 处理）。

**参数传递**：`run_n1_live.py` 从 CLI 取 θ_low/θ_high/j/M（由 1a 报告操作点填）→ `os.environ` 注入 `N1_THETA_LOW/N1_THETA_HIGH/N1_J/N1_M`；`worker_entry_n1` 读同名 env（`n1_params_from_env`：缺失/非法 → 明确 raise）。driver 侧传**未包装** `default_client_factory`（不 infer），仅 worker wrap。**一次 conductor 调用 = 一个 (config, 操作点)**（单 yaml 单 θ）。

---

## 5. 集成点与不变量

- driver `ctl_factory` = 未包装 `default_client_factory`（只 stage load/preload，不 infer）；N1 wrap 仅 worker 侧（`worker_entry_n1`）。
- `select_bundle`（`episode_runner.py:129`）/`episode_start`（`:136`，每 episode 必调 → N1 复位钩子）/`episode_end` 全透传 inner。
- **记录**：`row["searched"]` 经 `__collect_meta__` seam（§3）落 per_step_rows；`hit_type/cp1_score/start_t/winner_id` 经 `_hit_row` 原样落。**不开 server collection**（无 `--export-collect-meta`）→ 无 C5 冲突。
- **可观测口径（G1R3-4）**：skip 判定**只**用权威 `searched` 字段（非 cp1_score None）；skip% = `searched==False` 占比；verdict mix 仅统计 `searched==True` 步；skip 步 `hit_type=MISS` 是占位、**不计为真 MISS**；periodic run 无 `searched` 字段 → analyzer 按 PeriodicGate 闭式从 decision ordinal 重建（§6/§9）。
- 配对可比性：live run 复用 `WarmupEvalStrategy(skip_warmup=True)` 同 task/init（0..49×10），与 Stage-0 基线**同 conductor/同 env/同 episode_id 方案，仅 gate 不同** → SR 可配对。

---

## 6. 运行范围（Owner 已确认 N1 A/B；periodic 对照按 roadmap 强制加入，G1R1-3）

**N1（Owner 2026-07-04 确认：标准）**：suite 代表 config 各一——libero_spatial=**fh75_ws10**、libero_10=**fh5_ws40**；每 config 跑 **A（近免费）+ B（平衡）** = 4 次 conductor × 500 ep。θ（1a 报告，`θ_low=ws+lo·g, θ_high=θ_low+band·g`）：
- spatial fh75_ws10（ws=0.968929, fh=0.975336, g=0.006407）：A = lo 0 / band 0 / j 3 / M 3；B = lo −0.25 / band 0 / j 1 / M 5。
- libero_10 fh5_ws40（ws=0.997175, fh=0.998383, g=0.001208）：A = lo −0.25 / band 0 / j 3 / M 3；B = lo −0.25 / band 0 / j 2 / M 8。

**matched-budget periodic（必做，roadmap §5 及格线；G1R1-3 / 观测链路 G1R3-2）**：对每个 N1 操作点，取其 live 实测 `skip%` = s，选整数 `(cache_len, inference_len)` 使 `inference_len/(cache_len+inference_len)` 最接近 s（`PeriodicGate`：`inference_len` 步跳搜、`cache_len` 步搜，计数器每 gate 调用 +1，`gate.py:230-232`），容差 **|Δskip%| ≤ 2pp**，`inference_len∈{1,2,3}` 优先小周期。每点一条匹配 periodic run（`gate.type: periodic` yaml）500 ep，比 SR。periodic **用默认 worker**（gate 在 server 侧，client 不发也不能发 `__gate_decision__`——`accepts_client_signal=False` 会 fail-loud）→ per-step 行无 `searched`；analyzer 按 PeriodicGate 闭式**从每 episode 内 decision ordinal 重建** `searched = (ordinal % (cache_len+inference_len)) < cache_len`（ordinal = 按 step_idx 排序序号，与 server 计数器对齐），并加测试。**执行路径（G1R5-2）**：由 `run_n1_live.py` 按 gate.type 分派到默认 worker 启动（§4/§7），写同 schema manifest（含 cache_len/inference_len）。
- **范围含义**：4 N1 + 至多 4 periodic = **至多 8 次 × 500 ep = 4000 ep**（基线免费复用 Stage-0）。periodic 需先拿到 N1 的 live skip% 才能匹配 → 天然是**第二波**。
- **Owner 待确认（第二波启动前）**：periodic 波把算力 2000→4000ep；若 Owner 认为 1a 离线已决定性证明 N1≫periodic 而愿正式**修订 roadmap 及格线**去掉 live periodic，则第二波可免（须同步改 `gate_exploration_roadmap.log.md` §5）。默认按 roadmap 执行（做 periodic）。

---

## 7. 测试策略（G1R1-6 / G1R3 修订：可执行 + 覆盖 wiring/异常/analyzer）

`n1_offline_scan.py` import 即读 `sys.argv` 且 `n1_sim` 只返回聚合量 → **不可导入复用**。逻辑改由可导入纯 `N1GateState`（import 无副作用）承载。测试落 `tests/exp/`（G1R3-6），全部**非 manual**（无 server/GPU）。

`tests/exp/test_n1_gate_client.py`：
- **golden traces**：手工 score 序列 → 断言 `N1GateState` decide/observe 逐步序列（连续 j 低分进 skip、每 M 步 probe、`score≥θ_high` 恢复、滞回带 θ_low<θ_high、reset）。附测内 10 行 mini-ref 重算（**不** import `n1_sim`）。
- **None-as-MISS（G1R3-4）**：searched 步 `score=None`（空搜索）→ 按 −inf 计入 low_run/不恢复，与真低分行为一致。
- **异常契约（G1R3-3）**：非有限 score / `__hit_meta__` 整块缺失 → fail-open（`searching=True, low_run=0, since_probe=0`，下一步 search，不 raise、不误跳）——**两条 trace**：异常发生在普通 searching 步 vs probe 步，均验证"确实继续搜索"。inner.infer 抛 → 传播。
- **参数校验**：θ 非有限 / θ_high<θ_low / j<1 / M=0 → 构造 raise ValueError。
- **provenance round-trip**：fake inner 记录收到 obs → 断言 `out_obs["__gate_decision__"]`==decide()、`result["__collect_meta__"]["searched"]`==(decision=="search")、`episode_start` 触发 reset（跨 episode 不泄漏、连接复用下归零）。
- **env 解析**：`n1_params_from_env` 缺/非法 → raise；合法 → 正确 params。
- **runner/config**：`run_n1_live` 单-yaml 断言（0/>1 → SystemExit）+ manifest 字段完整性；**gate.type 分派（G1R5-2）**：`client_controlled` → 选 `worker_entry_n1` + 要求 N1 env；`periodic` → 选默认 `worker_entry`、不注入 N1 env（防 periodic 一启动即 fail-loud）；加载 n1 yaml → config `validate` 通过且 `accepts_client_signal` True。

`tests/exp/test_analyze_n1_live.py`（G1R3-1/2/5）：
- **per-step 去重（G1R5-1）**：合成含多 attempt + `_kind=episode_summary` 的行，**含"高 attempt 比低 attempt 更短"场景** → 断言按 task_uid 取**全局 max attempt** 再只保留该 attempt 的 step 行（**不**逐 (task_uid,step_idx) 混 attempt）、排除 summary、episode 长度正确。
- **跨-run 配对**：baseline/N1/periodic yaml_id 不同 → 断言按 `(task_id, subset_init_state_idx)` 配对（从 task_uid `rsplit(":",3)` 解析），交集非空、SR 差/McNemar 正确。
- **journal 重复 fail-fast**：合成重复 terminal `task_uid`（journal 无 attempt）→ 断言 analyzer raise（不猜 accepted）。
- **periodic searched 重建**：给 `(cache_len, inference_len)` + 一段 ordinal → 断言重建 `searched`/skip% 与闭式一致。
- **inf_ratio / C9（含 baseline，G1R5-3）**：合成 live searched/skip + hit_type 与一小段 Stage-0 baseline gate_rows → 断言 live inf_ratio（skip→1.0, FH→0, WS→warm_cost, MISS→1.0）、baseline_inf_ratio（Stage-0 全 searched 步均值）、`Δinf` 与三档 net 计算正确。
- **offline-replay 诊断**：给一小段 Stage-0 gate_rows 片段 → 断言共享 `N1GateState` 重放产出的期望 skip%/verdict-mix 与直接调用一致（算法自洽）。

**冒烟（Verify 后，manual）**：重拉 server 后单 config few-ep（fh75_ws10 A，2 task×2 init），核对 worker 实读 θ、`searched` 落库、skip→server MISS、SR 合理、无 fail-loud。
**回归**：`uv run pytest` 全绿；现有 conductor/episode_runner/examples 单测不受影响（全新文件、默认路径字节不变）。

---

## 8. 风险登记

| 风险 | 缓解 |
|---|---|
| N1 wrapper 改 obs 引发 server fail-loud（config 不匹配） | yaml 必须 client_controlled；wrapper 仅 client_controlled run 启用；config 校验测 + 冒烟先验 |
| worker env 未继承 N1 θ（conda spawn 剥 env） | 已核 `agent.py:101-104` 仅剥 VIRTUAL_ENV/PYTHONPATH/PYTHONHOME，N1_* 保留；`n1_params_from_env` 缺失即 raise；冒烟核对 worker 实读 θ |
| `__collect_meta__` seam 依赖 `encode_collect_meta` 对合成 dict 行为 | 已核 `collect_util.encode_collect_meta`：无 "collect" 键 → `{"collect":None,"searched":True}`，`infer_recorder` 记 `searched`；Code 期加断言 + provenance round-trip 测 |
| exp.* 作 worker module 在 conda env 不可导 | 已核 `exp/__init__.py`/`exp/gate_research/__init__.py` 存在、repo root 在 PYTHONPATH、worker 现已导 `openpi.conductor`/`examples.libero`；worker_entry_n1 仅依赖 numpy+openpi_client+这两者；冒烟先验 |
| live skip%/verdict 偏离离线（C8 反事实） | **预期**并要测量；仅比可观测量（skip%、searched-verdict mix），**不比 lost%**；系统性偏离 → 离线前沿降粗筛 |
| 缺 score 时状态机误跳/静默失败 | 异常契约（§4）：searched 步非有限 score/缺 meta → **fail-open 强制恢复全搜索**（不 raise、不误跳）；None-on-searched 当合法 MISS；构造期 fail-fast；单测覆盖 |
| 每 episode 复位漏触发 → 状态跨 episode 泄漏 | 复位挂 `episode_start`（`:136` 每 episode 必调）；provenance round-trip 测覆盖 |
| lost% 不可观测被误当验收 | §1 明确删除 live lost%；分析器不产出 lost%；shadow-search 仅登记备选、不默认 |
| periodic 波倍增算力 | 第二波、启动前 Owner 再确认；或 Owner 正式修订 roadmap 免之 |

---

## 9. 交付与可复现分析（G1R1-7 / G1R3-1/2/5 修订）

- **Code**：§4 全部新文件 diff + plan-conformance 声明。
- **run manifest（每 (config,点) 一份，run_n1_live 写；N1/periodic 同 schema）**：`{run_id, suite, config, point, gate_type, theta_low, theta_high, j, M | cache_len, inference_len, replan_steps, matched_to, yaml_id, baseline_yaml_id, journal_path, per_step_out_path, baseline_journal_path, baseline_gate_rows_path, n_episodes}`。跨-run 配对靠 `(task_id, subset_init_state_idx)`（非 yaml_id-绑定的 task_uid）。**baseline 用确定性直接路径**：`baseline_journal_path`（给 SR）+ `baseline_gate_rows_path`（给 C9），均指向 Stage-0 同 config 产物，按 `baseline_yaml_id` 过滤、缺失/多匹配 fail-fast。**`replan_steps` 是可信 provenance（== worker `main.Args.replan_steps`，本 runner=5）**：analyzer 的"无缺步"校验用它做 expected spacing（`step_idx == [0, sp, 2sp, ...]`），**不从被审数据反推**（否则全运行一致丢行会漏检，G2R5）。`periodic` 的 `matched_to` = 其配对的 N1 `run_id`。
- **`analyze_n1_live.py`（确定性）**：
  1. **per-step 去重（G1R3-1 / G1R5-1）**：排除 `_kind=episode_summary`；**先按 `task_uid` 取全局 max `attempt`，再只保留该 attempt 的全部 step 行**（同 attempt 内按 step_idx 去重）——**不**逐 `(task_uid,step_idx)` 各取 max（否则会把不同 attempt 拼成虚增长度的 franken-episode，污染 skip%/inf_ratio/ordinal）。
  2. **SR / journal（G1R3-1）**：从各 run journal 取逐 episode success；journal 无 attempt，若同一 terminal `task_uid` 出现 >1 次 → **fail-fast raise**（不用 latest-ts 猜 accepted）。
  3. **skip 判定（G1R3-4）**：N1 run 用权威 `searched` 字段；**periodic run** 按 PeriodicGate 闭式从每 episode 内 decision ordinal 重建 `searched`（G1R3-2）。
  4. **baseline_inf_ratio（G1R5-3，C9 必需）**：从 `baseline_gate_rows_path` 的 Stage-0 gate_rows（config 过滤 + 同 §9.1 去重 + 完整性校验：全 `searched=True`、无缺步）算 baseline inf_ratio = 全步 `inf_value(hit_type,start_t)` 均值（always_search → 每步皆 searched）。
  5. **量**：live actual SR / skip%(`searched==False`) / live inf_ratio(skip→1.0, searched→inf_value(hit_type,start_t)) / searched-step verdict mix / C9 三档（`Δinf = live_inf_ratio − baseline_inf_ratio`，均实测）。
  6. **跨-run 配对（G1R3-1）**：N1 vs SR-baseline vs matched-periodic 均按 `(suite, task_id, subset_init_state_idx)` 配对 → SR 差 + McNemar。
  7. **（可选诊断，G1R3-5）** 复用 `baseline_gate_rows_path` 的 gate_rows：用**共享可导入 `N1GateState`** 重放产期望 skip%/verdict-mix，与 live 比（仅诊断，不进 pass line）。
  8. 写 `exp/gate_research/analysis/n1_live_results.md` + result manifest（每点一行 + 来源路径/provenance）。
- **analyzer 测**：见 §7（去重[含高-attempt-更短]/配对/journal 重复/periodic 重建/inf_ratio+baseline/offline-replay 全覆盖，非 manual）。
- **Verify**：`uv run pytest` 全绿 + 冒烟输出。
- **不在本计划内**：G0a hook 服务器化（Stage 1c）；N2（Stage 2）；shadow-search lost% 测量（备选）。

---

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-07-04 08:30 CDT

- [Blocking] [Concern] analyzer 未实现批准计划的 matched-budget periodic 裁决。— reasoning: `analyze()` 只逐 run 计算其相对 always-search baseline 的 SR/Δinf，manifest 没有 N1↔periodic 匹配关系，结果也没有 periodic 的预算差、配对 SR 差或 `N1 SR ≥ periodic SR` 判定；`render_md()` 只是并排列行却宣称“同预算 ≥ periodic”。必须用明确的匹配键/规则把每个 N1 A/B 与对应 periodic run 关联，验证 actual `|Δskip|≤2pp`，在相同 canonical units 上直接计算 N1-vs-periodic SR/McNemar/是否过线，并增加自动化测试。
- [Blocking] [Concern] analyzer 对批准的数据完整性契约执行不足，会把采集断链或 stale baseline 静默变成有效结果。— reasoning: client-controlled 行缺 `searched` 时 `bool(r.get("searched"))` 静默变成 skip；`baseline_inf_ratio()` 只拒绝显式 False、缺字段仍通过；更重要的是 baseline gate rows 未按 §9.1 先选 task_uid 的全局 max attempt，重复 attempt 被一起平均，`replay_offline()` 也有同样问题。Reviewer 独立探针稳定复现三项失败。应统一复用严格 loader/deduper：要求 N1 每行 `searched` 为 bool，baseline 每行 `searched is True`，对 live/baseline/offline replay 均执行 episode-global max-attempt 选择，并覆盖缺字段/多 attempt 测试。
- [Blocking] [Concern] 配对完整性未校验，少跑 episode 会被静默缩成较小交集后仍给出 pass-line 数字。— reasoning: `mcnemar()` 对两个 unit map 直接取交集，不验证集合相等或 manifest `n_episodes`，因此 500-episode 配对要求可能退化成任意 N；`run_metrics()` 也不核对 journal/per-step/config 与 manifest 的 yaml_id、episode count 和 unit set。Reviewer 缺单元探针未触发任何错误。应对 N1、baseline、periodic 的 canonical unit set 与预期 `n_episodes` 做 fail-fast 等集校验，并测试缺失/额外 unit。
- [Blocking] [Concern] `--M none` 与批准接口不兼容，且 runner 未在 driver 侧验证 N1 参数。— reasoning: `_normalize_m()` 把显式 `--M none` 变成 None，随后 `_resolve_worker_and_env()` 又把任何 None 当“未提供”而退出；Reviewer 探针稳定复现。当前 parser 也无法区分 flag 缺失与显式 none。须使用 sentinel/独立 presence 标记保留该区别，并在启动 driver/agent 前调用 `N1GateState` 或等价校验 theta/j/M，避免非法参数导致 worker 被 agent 无限重启。
- [Blocking] [Concern] 新增文件未通过项目静态质量门。— reasoning: `uv run ruff check` 报 5 项：`analyze_n1_live.py` unused `math` + E741 `l`；`test_n1_gate_client.py` unused `math`、unused `inner_sentinel`、E402 中途 import。定向 pytest 虽 87 passed，但 lint/pre-commit 未通过，不能满足 WA §7/G2 tests-passing 条件。
- [Blocking] [Concern] `logs/README.md` 未与 polished plan/当前阶段同步。— reasoning: 目标计划已是 `G1 APPROVED / Code` 并进入 G2，索引仍写 `G1 R1 NEEDS REVISION → R2 Executor applied, 待重审`，属于文档索引不同步；须更新为实际 G1 R7 APPROVED、Code complete/G2 状态并保持描述与最终文件集一致。
- [Non-blocking] [Suggestion] periodic YAML 尚未生成，应在 plan-conformance/deviation 记录中明确其为 live N1 得到 actual skip% 后的第二波产物。— reasoning: 参数依赖第一波结果，当前缺文件可以合理延后；但批准计划文件表写了 periodic YAML，G2 交付说明应明确 deferred 条件，避免被误读为遗漏。

### G2 Round 2 — Executor — 2026-07-04

6 项 blocking + 1 项 non-blocking 全部 Accepted，代码/文档已修（ruff clean，57 tests pass）。

- Accepted（G2-1 matched-periodic 裁决）—— `analyze_n1_live.py` 新增 `match_periodic()`：按 periodic manifest `matched_to==N1 run_id` 配对，验 `|Δskip|≤2pp`，在等集 canonical units 上算 N1-vs-periodic McNemar + `n1_ge_periodic` 判定；`render_md` 显示 vs-periodic 裁决（未配对标 "periodic pending"）；runner 加 `--matched-to` + manifest `matched_to`；测 `test_match_periodic_verdict`/`_none_when_unmatched`。
- Accepted（G2-2 数据完整性,3 项）—— (a) `episode_searched()` 对 client_controlled 行要求 `searched` 为 bool，缺失/非 bool → raise（不再静默变 skip）；(b) `baseline_inf_ratio()` 要求每行 `searched is True`；(c) 统一 `dedup_episodes()`（episode-global max-attempt）复用于 live / baseline / offline-replay。测 `test_run_metrics_missing_searched_raises` / `test_baseline_inf_ratio_dedups_max_attempt` / `_rejects_non_searched`。
- Accepted（G2-3 配对完整性）—— `mcnemar(require_equal=True)` 对 unit 集不等 → fail-fast（列 symmetric-diff）；`run_metrics()` 校验 journal unit 数 == manifest `n_episodes`、per-step/journal 按 manifest `yaml_id` 过滤。测 `test_mcnemar_requires_equal_units` / `test_run_metrics_episode_count_mismatch_raises`。
- Accepted（G2-4 `--M none` + driver 预校验）—— 引入 `M_UNSET` sentinel 区分 flag 缺失 vs 显式 none；`_resolve_worker_and_env` 只对 theta/j 判 None、M 判 sentinel，且在 spawn 前调 `N1GateState(...)` fail-fast（避免非法参数令 agent 无限重启 worker）。测 `test_resolve_worker_client_controlled_m_none_ok` / `_requires_theta_and_m` / `_rejects_bad_params` / `test_normalize_m`。
- Accepted（G2-5 ruff）—— 删 `analyze_n1_live.py` unused `math`、`for l`→`for line`（E741）；删 test unused `math`/`inner_sentinel`、`import pathlib` 提顶（E402）。`ruff check` 全过。
- Accepted（G2-6 README 未同步）—— `logs/README.md` 条目更新为 `In Progress`（G1 R7 APPROVED / §4 Code 完成 / G2 R1→R2 待重审）。
- Accepted（G2-7 periodic yaml 延后）—— §4 文件表明确 periodic yaml 为第二波交付（参数依赖首波 live skip%），首波 Code 只出 2 个 client_controlled yaml。

Verify（advisory）：`ruff check` clean；`pytest tests/exp/test_n1_gate_client.py tests/exp/test_analyze_n1_live.py` → 57 passed。重入 G2。

### G2 Round 3 — Reviewer — NEEDS REVISION — 2026-07-04 09:09 CDT

- [Blocking] [Concern] 数据完整性修复仍未落实批准计划要求的 live/baseline 完整 episode 集与“无缺步”校验。— reasoning: `run_metrics()` 只校验 journal 数量等于 `n_episodes`，没有验证 per-step canonical unit set 与 journal 等集，因此 2 条完整 journal 配 1 个 episode 的 per-step 数据仍会产出指标；`baseline_inf_ratio()` 也不验证 baseline gate rows 覆盖 baseline journal/live 的全部 canonical units，且同一 episode 的 `step_idx=[0,2]` 缺步仍通过。三个独立探针稳定复现。须对选定 max-attempt 后的 live per-step ↔ live journal、baseline gate rows ↔ baseline journal（并与配对 live units）做等集 fail-fast，并验证每个 episode 的 step_idx 从 0 连续无缺口；增加缺 episode、额外 episode、内部缺步测试。
- [Blocking] [Concern] matched-periodic 的 pass-line 裁决仍可能给出错误的 PASS，且匹配关系不唯一时结果依赖输入顺序。— reasoning: 当 `|Δskip|>2pp` 时 `match_periodic()` 虽置 `skip_match_ok=False`，但 `render_md()` 仍只按 `n1_ge_periodic` 输出 `PASS (... OOB)`，违反“同预算”是预算判据前置条件；两个 periodic manifest 同时 `matched_to` 同一 N1 时函数静默取第一个。两个独立探针稳定复现。须要求每个已裁决 N1 恰有一个 matched periodic（并核同 suite/config/provenance），定义并持久化 `periodic_pass = skip_match_ok and n1_ge_periodic`，预算越界明确 FAIL；同时给出 baseline `sr_preservation_ok` 与二者组合的 overall pass（首波 periodic 未运行时应为 pending/不可裁决，而非 pass）。
- [Blocking] [Concern] `logs/README.md` 的交付描述仍与本轮明确的 deferred 文件集不一致。— reasoning: plan §4 与 Executor 回复已明确首波只交付两个 client-controlled YAML、periodic YAML 第二波生成，但索引仍写“client_controlled/periodic yaml + 单测”，会把尚不存在的 periodic YAML 表述为已完成交付；须同步改成 client-controlled YAML 已交付、periodic YAML deferred，并在本轮状态中记录 G2 R3 待整改。

### G2 Round 4 — Executor — 2026-07-04

3 项 blocking 全部 Accepted，代码/文档已修（ruff clean，63 tests pass）。

- Accepted（G2R3-1 完整性:等集 + 无缺步）—— 新增 `check_complete_decisions()`：按**运行级 spacing** 校验每 episode 的 step_idx 为从 0 的完整等差列（真实数据 step_idx=ordinal×replan_steps，如 0,5,10；dropped decision 出现倍距缺口即 fail-fast，不误拒合法数据）；`run_metrics()` 校验 per-step unit 集 == journal unit 集；`analyze()` 校验 **live / baseline-journal / baseline-gate-rows 三方 unit 等集**；`baseline_inf_ratio` 加无缺步校验。测 `test_check_complete_decisions_ok_real_spacing` / `_gap_raises` / `test_run_metrics_unit_mismatch_raises`。
- Accepted（G2R3-2 periodic 裁决 + 唯一性）—— `match_periodic()`：同一 N1 有 >1 matched periodic → raise；校验 periodic 与 N1 同 suite/config；`periodic_pass = skip_match_ok and n1_ge_periodic`（|Δskip|>2pp 预算越界 → **FAIL 不是 pass**）；`analyze()` 加 `vs_baseline.sr_preservation_ok`（ΔSR≥−1pp）+ `overall`（pass 需二者皆成立；**periodic 未跑 = pending 非 pass**）；`render_md` 按 `periodic_pass`/`overall` 显示。测 `test_match_periodic_oob_budget_fails` / `_duplicate_raises` / `_suite_config_mismatch_raises`。
- Accepted（G2R3-3 README 交付描述）—— `logs/README.md` 改为 **2 个 client_controlled yaml 已交付、periodic yaml deferred 第二波**，状态记 G2 R3→R4 待重审。

Verify（advisory）：`ruff check` clean；`pytest tests/exp/test_n1_gate_client.py tests/exp/test_analyze_n1_live.py` → 63 passed。重入 G2。

### G2 Round 5 — Reviewer — NEEDS REVISION — 2026-07-04 09:23 CDT

- [Blocking] [Concern] “无缺步”校验仍从待验证数据自身推断 expected spacing，无法发现全运行一致发生的系统性丢行。— reasoning: `check_complete_decisions()` 取全部观测 diff 的最小值作为 spacing；若每个 episode 都从合法的 `0,5,10` 丢掉 step 5，只剩 `0,10`，函数会把 10 推断成合法 spacing 并通过。独立探针以两个 episode 同时 `[0,10]` 稳定复现。当前 N1/default worker 都构造 `main.Args` 且不开放 replan 参数，可信契约明确为 `replan_steps=5`；该漏检会污染 skip/inf 指标并错位 periodic ordinal。须从可信 provenance（建议 run manifest 显式记录 `replan_steps`，本 runner 为 5）向完整性校验传入 expected spacing，逐 episode 严格核 `step_idx == [0, spacing, ...]`，不得从被审数据反推；增加“所有 episode 同时丢同一 decision”的测试。

### G2 Round 6 — Executor — 2026-07-04

1 项 blocking Accepted，代码/文档已修（ruff clean，65 tests pass）。

- Accepted（G2R5 无缺步用可信 spacing）—— `run_n1_live` 新增 `--replan-steps`（默认 5 == worker `main.Args.replan_steps`）并写入 manifest `replan_steps`；`check_complete_decisions(episodes, spacing)` 改为**接收可信 spacing**、逐 episode 严格核 `step_idx == [0, sp, 2sp, ...]`，**不再从被审数据反推**——全运行一致丢行（如所有 episode `0,5,10 → 0,10,20`）现被 `[0,10,20] ≠ [0,5,10]` 抓到；`run_metrics` / `baseline_inf_ratio` / `replay_offline` / `analyze` 均改传 `manifest["replan_steps"]`。测 `test_check_complete_decisions_uniform_drop_caught`（两 episode 同 `0,10,20` @trusted spacing 5 → raise）+ `_ok_real_spacing` / `_gap_raises` / `_bad_spacing`。

Verify（advisory）：`ruff check` clean；`pytest tests/exp/test_n1_gate_client.py tests/exp/test_analyze_n1_live.py` → 65 passed。重入 G2。

### G2 Round 7 — Reviewer — APPROVED — 2026-07-04 09:36 CDT

- [Resolved] G2R5 的阻塞项已闭环：`run_n1_live` 将可信 `replan_steps` 写入 manifest，`analyze_n1_live` 的 live/baseline/offline-replay 完整性校验均显式使用该 spacing；`check_complete_decisions()` 不再从待审 per-step 数据反推 spacing，因此所有 episode 同时丢相同 decision 的场景会 fail-fast。
- [Verification] 提交回归覆盖已补齐：`test_check_complete_decisions_uniform_drop_caught` 验证 uniform-drop 被拒；Reviewer 独立探针同样覆盖 baseline `0,10` @ trusted spacing 5。`ruff check` All checks passed；相关回归 104 passed；独立审查探针 10 passed。前序 G2 blocking（matched periodic、unit 等集、严格 searched、max-attempt dedup、`--M none`、README/文件集同步）均保持闭环。
