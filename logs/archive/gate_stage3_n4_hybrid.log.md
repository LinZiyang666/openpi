# Stage 3a — N4 混合门 live 原型（Plan）

- **Task**: 实现并 live 验证 N4 混合门（N1 跳预测-MISS 的 V1 分支 + 连续缓存执行 run-length ≥ L 强制注入新推理的 V2 分支），检验其在同 inf_ratio 下能否达到 matched-periodic 的 SR。
- **Level**: L2（多文件 exp 层特征：新客户端状态机 + worker entry + harness 分派 + 分析器适配 + 单测 + 报告；**零 src 改动**）。
- **Authority**: Execution
- **Date**: 2026-07-05（创建）
- **Roadmap 锚点**: `logs/gate_exploration_roadmap.log.md` §5 Stage 3 表 3a 行；净指令（roadmap line 226）：V2 分支用 **uniform / 连续缓存执行 run-length ≥ L 触发（L≈6–8 低剂量）**，**不做危险步靶向**（Stage 2c 否决，AUC~0.52）；V1 分支沿用 N1 跳预测 MISS。
- **前置判决依赖**: Stage 2a（H1 剂量-截断成立 → L 先验 {6,8,12}）、2b（gate 是第四设计轴 → N4 值得做）、2c（危险步不可廉价预测 → 不靶向）。
- **Status**: Done（L2 代码收官：G1/G2 APPROVED R2 + §6 Verify scoped-green，commit `251eddc`）。**Live 已跑并判决（2026-07-05 无人值守）：N4 胜出，赢点 L=6，4/6 pass**；报告 `exp/gate_research/analysis/stage3_n4_live.md`，roadmap §5 已回填。待 owner 确认后归档。

---

## 1. 目标与范围

### 1.1 机制（N4 = V1 ⊕ V2 叠加）

每个 CP1 决策步，N4 客户端状态机产出 `search|skip` 驱动服务器侧 `ClientControlledGate`：

- **默认**：`search`（查缓存：FULL_HIT→回放缓存动作、WARM_START→暖启推理、MISS→全推理）。
- **V1 分支**（延迟，沿用 N1）：N1 双阈值滞回判定预测 MISS → `skip`。skip = 不查缓存直接全推理；因该步本来就是 MISS（全推理），动作分布不变 → **inference-neutral**，只省 search+judge+fetch 开销（roadmap C8：此类跳步离线精确）。
- **V2 分支**（SR 增益，新增）：连续缓存执行（consecutive FULL_HIT 回放）run-length ≥ **L** 时，强制 `skip` → 该步换成一次全新推理（**强制注入**），打断长缓存回放段。F12 机制：长连续缓存回放正是 SR 被压制之处；周期性注入给轨迹"回到自身流形"的机会（Stage 2a H1 剂量曲线证低剂量即饱和 → L 取小）。

**关键实现事实（决定"零 src"成立）**：服务器回传的 `result["__hit_meta__"]` 带 `"hit_type"`（`interceptor.py:500`，∈ {FULL_HIT, WARM_START, MISS}），所以 V2 分支能在**客户端**逐步读 hit_type 数连续 FULL_HIT run-length，无需任何 src 改动。

### 1.2 及格线（roadmap Stage 3 表，C10 轴）

对每个 N4 操作点：
1. **同 inf_ratio 下 SR ≥ matched periodic**（对照取 inf_ratio 最接近的 periodic 点，必要时补 1–2 个 periodic 档）；
2. **net@34 ≥ 0**（延迟净收益，34ms/步 stock 档）；
3. **SR ≥ baseline − 1pp**（不倒退）；
4. 按 C9 三档（优化栈 4ms / stock 34ms / 50k 库 70ms）报告 net。

三条件**合成为分析器单字段 `overall`（N4 分支）**，任一失败即 `fail`（公式与实现见 §5.5——现有 `analyze_n1_live.py:347` 的 N1 `overall` 只含 SR 保真 ∧ periodic_pass、**不含 net**，N4 必须显式补 net 门，否则延迟净负仍误报 pass，违背 C10/C9）。

判决 = N4 是否在某 (suite, L) 上 `overall == "pass"`（三条件同满足）→ "V2 注入可在 live 兑现 SR 增益且延迟不亏"。fallback：若全 L 档 SR 均 < matched periodic → N4 相对纯 periodic 无增量优势，记录并将 V2 触发调度问题回 Stage 4（C1 标定门特征集）。

### 1.3 范围外（明确不做）

- **3b 定型服务器化**：严格下游于 3a 判决（"N4 胜出 → 扩展 ScoreHysteresisGate"），本 plan 不含；3a 判决后另起。
- **危险步靶向注入**：Stage 2c 否（AUC~0.52），净指令排除。
- **src 改动**：N4 全部落 exp 层，复用现成 `ClientControlledGate`。任何 src 触碰即超范围。
- **实际 live 发射**：多 GPU、长时、无人值守，属独立运维动作，须 owner 确认后单独执行（见 §7、feedback_no_background_tasks_unattended）；本 plan 只交付可发射的代码 + 拓扑配方，不在 G1→Code 阶段自动起跑。

---

## 2. 数据资产与可复用 API（均已亲验 file:line）

### 2.1 客户端状态机与封装（复用/镜像源）

- `exp/gate_research/n1_gate_client.py:51` `N1GateState(theta_low, theta_high, j, M)` — Mealy 机：`decide()→"search"|"skip"`（:89）；`observe(decision, score)`（:99，None score = empty-search MISS 作 −inf）；`reset()`（:83，per-episode，首步必 search）；`force_search()`（:125，异常 fail-open）。内部态：`searching/low_run/since_probe`。**N4 将其作为 V1 子机内嵌复用，不改一行**。
- `n1_gate_client.py:41-45` 契约键：`GATE_DECISION_KEY="__gate_decision__"`、`HIT_META_KEY="__hit_meta__"`、`COLLECT_META_KEY="__collect_meta__"`、`SEARCH/SKIP`。
- `n1_gate_client.py:137` `N1GateClient(inner, state)`：`infer(obs)`（:149）decide→注入 `obs["__gate_decision__"]`→`inner.infer`→读 `result["__hit_meta__"]["cp1_score"]`→observe→盖 `result["__collect_meta__"]={"searched":decision==SEARCH}`（:177，**唯一权威 skip 信号**，cp1_score=None 不足判 skip）；`episode_start`（:180）reset 后委托 inner；`_read_score`（:185）返回 `(score, anomaly)`。**N4GateClient 镜像此结构，额外读 hit_type**。
- `n1_gate_client.py:239` `make_n1_client_factory(params, inner_factory=None)`→`factory(server)`；`n1_params_from_env(env)`（:218）读 `N1_THETA_LOW/HIGH/J/M`。

### 2.2 服务器侧门（复用，零改动）

- `src/openpi/cache/components/gate.py:259` `ClientControlledGate.__call__` — 读 `request_context["gate_decision"]` ∈ {"skip","search"}，skip→`False`、search→`True`、否则 raise；无状态。**N1 与 N4 都靠客户端驱动它，server 端完全一致 → 零 src**。
- `gate.py:199` `PeriodicGate(cache_len, inference_len)` — 服务器侧周期跳，skip% = `inference_len/(cache_len+inference_len)`；matched-periodic 对照走默认 worker。
- `src/openpi/cache/interceptor.py:482` `_build_hit_meta(cp1_result)` → `{"hit_type": cp1_result.hit_type.name, "start_t", "winner_id", "cp1_score", [factor_outputs]}`；skip 步 cp1_result=None → MISS 占位（`hit_type="MISS", cp1_score=None`，:492-498）。**hit_type 客户端可读 = V2 数 FULL_HIT run 的依据**。

### 2.3 发射 harness（扩展）

- `exp/gate_research/run_n1_live.py` — 单 (config, 操作点)/次；`_resolve_worker_and_env`（:107）按 YAML `gate.type` 分派：`client_controlled`→`worker_entry_n1`（env 注入阈值）、`periodic`→默认 worker。`build_manifest`（:81）写全 provenance；`single_yaml`（:60）、`gate_info`（:68）；launch = `ConductorDriver` + `WarmupEvalStrategy(skip_warmup=True)`（:209）+ `WorkerAgent`（:257）；崩溃安全增量 append（:273）。
- `exp/gate_research/worker_entry_n1.py` — 构 `LiberoEpisodeRunner(args, setup, client_factory=make_n1_client_factory(params))`；N1 params from env；`worker_module` 名。**N4 镜像出 `worker_entry_n4.py`**。
- 配置 `exp/gate_research/config/{libero_spatial,libero_10}/n1/...quantile.yaml` — `gate.type: client_controlled`（spatial 已验 line 26）。**N4 逐字复用该 YAML**（门在 server 侧、N4 只换客户端状态机），不新增门 YAML。

### 2.4 分析器（小改）

- `exp/gate_research/analyze_n1_live.py`：`episode_searched`（:137）client_controlled 读 per-step `searched` bool（N4 亦产出，直接可用）；`replay_offline`（:215）用 `N1GateState` 重放离线 skip% —— **N1 专用，N4 必须旁路**（否则拿 N1 机器重放 N4 轨迹得错值）；`match_periodic`（:264）经 manifest `matched_to` 配对 + `|Δskip|≤2pp` + `N1 SR≥periodic SR`（:270-290）；`analyze`（:294）分派点 :330 `if gate_type=="client_controlled" and theta_low is not None → replay_offline`。

### 2.5 既有 1b 资产（对照/baseline，join 复用）

- Stage-0 same-config baseline：`exp/gate_research/data/{libero_spatial,libero_10}/journal.jsonl` + `gate_rows.jsonl`（N4 run 的 SR/C9 baseline，manifest 直接指向）。
- 1b periodic 锚点（候选 matched-periodic）：spatial_A_periodic `cache_len=7/inference_len=1`（skip 12.5%）、l10_A/B_periodic 等；均带 `matched_to` 指向对应 N1 run。**N4 因 V2 注入抬高 skip%/inf_ratio，多半落在这些点之上 → 见 §5.3 补档决策规则**。
- 1a A 操作点（N4 沿用的 (θ,j,M)）：spatial fh75_ws10 A = θ 0.968929/0.968929, j=3, M=3；l10 fh5_ws40 A（manifest 内）。

---

## 3. 设计：N4GateState 组合语义（本 plan 核心）

### 3.1 状态

```
N4GateState(theta_low, theta_high, j, M, L, include_ws=False)
  ├─ _n1: N1GateState(theta_low, theta_high, j, M)   # V1 子机，单一真源，复用不改
  ├─ L: int >= 1                                      # V2 注入阈值（连续 FULL_HIT run cap）
  ├─ include_ws: bool = False                         # WARM_START 是否计入缓存执行 run
  ├─ fh_run: int                                      # 当前连续缓存执行 run-length
  └─ _last_v2: bool                                   # decide() 记录上一决策是否为 V2 注入
```

### 3.2 decide()（Mealy 输出）

```
base = _n1.decide()                 # "search" | "skip"
if base == "skip":                  # V1：N1 预测 MISS
    _last_v2 = False; return "skip"
if fh_run >= L:                     # V2：连续缓存执行到顶，强制注入
    _last_v2 = True;  return "skip"
_last_v2 = False; return "search"
```

precedence：V1（N1 skip）优先于 V2。语义上无冲突（两者都产 skip），但决定状态推进路径（§3.3）。V1 skip 时 fh_run 恒为 0（跳步不产生缓存执行），故 V2 判据自然不触发。

### 3.3 observe(decision, hit_type, score)（状态推进）

N4 比 N1 多读一个 hit_type（V2 数 run 用）。三条路径：

```
if decision == "search":
    _n1.observe("search", score)                      # 推进滞回机（同 N1）
    is_cache_exec = (hit_type == "FULL_HIT") or (include_ws and hit_type == "WARM_START")
    fh_run = fh_run + 1 if is_cache_exec else 0        # 非缓存执行(WS默认/MISS)清零
else:  # skip
    fh_run = 0                                         # skip = 全新推理，打断缓存执行段
    if _last_v2:
        pass          # V2 注入：N1 未决定此跳，保持 N1 态冻结（不喂 since_probe，避免污染）
    else:
        _n1.observe("skip", None)                      # V1 跳：同 N1GateClient 现行 observe(SKIP,None)
```

**设计决策与理由（G1 审查焦点）**：

| # | 决策 | 选择 | 理由 |
|---|------|------|------|
| D1 | 缓存执行 run 的定义 | 默认 **仅 FULL_HIT**（`include_ws=False`） | F12/roadmap 的 SR 压制机制指"长纯缓存回放段"；WARM_START 已注入部分新算力（start_t>0），破坏纯回放 → 清零。与 `stage2_common.cache_run_lengths` 默认 `include_ws=False` 一致（同一口径）。设为参数留 sanity 对比。 |
| D2 | V2 注入长度 | **1 步**（单次注入后 fh_run 清零、恢复 search） | roadmap 规则 (ii) 用单数 "skip"；F12 低剂量饱和；periodic 亦 inference_len 可为 1（1b spatial 用 7:1）。留 `inject_len` 扩展位，默认 1。 |
| D3 | V2 注入是否喂 N1.observe | **否**（`_last_v2` 时冻结 N1） | 注入是 N1 未决定的 overlay。若喂 `observe("skip",None)` 会在 N1 searching 态下误增 since_probe（虽当前不影响判定，但语义不洁、埋雷）。冻结 = N1 只看它自己决定的步，V1 golden trace 不变。 |
| D4 | V1/V2 precedence | V1 优先 | 冲突时两者都产 skip，输出无差；precedence 只定状态路径。V1 skip 时 fh_run=0，V2 判据本就不触发，precedence 实为形式化。 |
| D5 | fail-open | 镜像 N1：searched 步异常数据 → `_n1.force_search()` **且** `fh_run=0` | 复用 N1 fail-open 语义 + 清 V2 计数，退化为 always-search，绝不误跳。 |

### 3.4 N4GateClient.infer(obs)（镜像 N1GateClient）

```
decision = state.decide()
out_obs = {**obs, "__gate_decision__": decision}
result = inner.infer(out_obs)
if decision == "search":
    score, anomaly = _read_score(result)      # 复用 N1 的 _read_score 逻辑
    hit_type = _read_hit_type(result)         # 新增：读 result["__hit_meta__"]["hit_type"]
    if anomaly: state.force_search()
    else:       state.observe("search", hit_type, score)
else:
    state.observe("skip", None, None)         # hit_type 忽略（skip 步 fh_run 直接清零）
result["__collect_meta__"] = {"searched": decision == "search"}   # 权威 skip 信号，同 N1
return result
```

`_read_hit_type`：`result["__hit_meta__"]["hit_type"]`，缺失/非法 → 视为非缓存执行（保守清零，不误延长 run）。搜索步 anomaly（缺 hit-meta/非有限 score）仍走 N1 现行 fail-open。

### 3.5 参数校验（fail-fast，worker 启动即崩）

复用 `N1GateState` 的构造校验（θ 有限、theta_high≥theta_low、j 为 int≥1、M 为 None 或 int≥1）；N4 追加 `L` 为 int≥1（拒 bool）、`include_ws` 为 bool。`n4_params_from_env` 读 `N4_THETA_LOW/HIGH/J/M/L`（+ 可选 `N4_INCLUDE_WS`），缺失/畸形 loudly raise（镜像 `n1_params_from_env:218`）。

---

## 4. 涉及文件

### 4.1 新增（exp 层）

| 文件 | 内容 |
|------|------|
| `exp/gate_research/n4_gate_client.py` | `N4GateState`（§3.1-3.3）、`N4GateClient`（§3.4）、`make_n4_client_factory`、`n4_params_from_env`。镜像 `n1_gate_client.py` 结构。 |
| `exp/gate_research/worker_entry_n4.py` | 镜像 `worker_entry_n1.py`：读 N4 env、构 `LiberoEpisodeRunner(client_factory=make_n4_client_factory(params))`。 |
| `tests/exp/test_n4_gate_client.py` | N4GateState golden traces + 组合边界（见 §6）。 |

### 4.2 修改（exp 层）

| 文件 | 改动 | 范围 |
|------|------|------|
| `exp/gate_research/run_n1_live.py` | 加 `--gate-family {n1,n4}`（默认 n1）+ `--L`；`_resolve_worker_and_env` 增 n4 分支（YAML 仍 client_controlled，靠 flag 区分 worker + N4 env）；`build_manifest` 增 `gate_family`、`L` 字段（n1/periodic 为 None）。 | +~30 行，不动 n1/periodic 现行路径 |
| `exp/gate_research/analyze_n1_live.py` | (a) `replay_offline` 分派（:330）对 N4 run（`gate_family=="n4"`）**旁路**（不拿 N1 机器重放）；(b) 新增 `match_periodic_n4`——非变异、按 live inf_ratio 最近选候选（见 §5.3），产 `periodic_pass_n4`；(c) `overall`（:344-348）对 N4 走三条件公式（见 §5.5），N1 分支逐字不变；(d) 结果行携带 `L`/`gate_family` 与所选 periodic 候选 provenance。 | +~55 行，n1 分支向后兼容 |

### 4.3 交付产物（post-run，Code/Verify 阶段生成）

- `exp/gate_research/analysis/stage3_n4_live.md` — N4 三 L 档 × 2 suite 的 (skip%, inf_ratio, SR, net@{4,34,70}) + N4-vs-matched-periodic 判决 + 对 3b 的净指令。
- N4 run 数据落 `exp/gate_research/data/n1_live/<n4_run_id>/`（gitignored，同 1b）。

### 4.4 配置

无新增门 YAML。若需补 periodic 档（§5.3），复用 `config/.../periodic_A|B/` 目录模式加 1–2 个 (cache_len, inference_len) YAML。

---

## 5. Harness 与分析器集成

### 5.1 N4 run 发射（复用 1b 全套 conductor 骨架）

```
uv run exp/gate_research/run_n1_live.py \
  --gate-family n4 --yaml-dir exp/gate_research/config/libero_spatial/n1 \
  --run-id spatial_n4_L8 --point A \
  --theta-low 0.968929 --theta-high 0.968929 --j 3 --M 3 --L 8 \
  --journal .../spatial_n4_L8/journal.jsonl --per-step-out .../rows.jsonl \
  --manifest-out .../manifest.json \
  --baseline-journal exp/gate_research/data/libero_spatial/journal.jsonl \
  --baseline-gate-rows exp/gate_research/data/libero_spatial/gate_rows.jsonl \
  --servers HOST:8000 --workers 48 --gpus 8 --task-ids 0-9 --eval-trials 50 \
  --task-suite libero_spatial --conda-env <env>
```

矩阵：L ∈ {6, 8, 12} × suite ∈ {libero_spatial, libero_10} = **6 个 N4 run**（500 ep 各）。(θ,j,M) 沿用 1a A 点。

### 5.2 manifest 权威字段

`build_manifest` 现有字段 + 新增 `gate_family`（∈ {n1,n4}，默认 n1）、`L`（N4 为 int，N1/periodic 为 None）；N4 run：`gate_family="n4"`、`theta_low/high/j/M` 照填（V1 子机参数）、`cache_len/inference_len=None`。**分派两级**：分析器先按现有 `gate_type`（`periodic` vs `client_controlled`）分（不变），再在 `client_controlled` 内按 `gate_family` 区分 N1/N4；periodic run 的 `gate_family` 无意义（`gate_type=periodic` 即判别键），保持默认不影响。

**`include_ws` 口径（可复现）**：3a **live 固定 `include_ws=False`**（D1 主口径），CLI/env **不暴露** `N4_INCLUDE_WS`，故 manifest 无需该字段、live 结论可复现。`include_ws=True` 仅为单测参数（§6 测 5 覆盖），不进任何 live run；若未来某 sanity live run 需用它，届时须同步把 `include_ws` 写入 manifest 才放行。

### 5.3 matched-periodic 配对规则（非变异，pass line 关键）

N4 的 skip% = V1(MISS-skip) + V2(HIT-skip 注入)，inf_ratio 因 V2 把无推理的 FULL_HIT 步换成推理而**高于 N1**。既有 1b periodic 锚点 skip% 偏低（spatial 12.5%），不一定覆盖 N4。

**唯一非变异配对规则**（新函数 `match_periodic_n4(n4_run, periodic_runs)`，与现有 `match_periodic` 并存、后者对 N1 逐字不变）：

- **候选集** = analyze() 输入清单里**同 `(suite, config)`** 的全部 periodic run。它们的 `matched_to`（指向旧 N1 run_id）**只读、绝不改写**，仅作 provenance 随判决表报出。1b 历史资产零改动 —— 这消除原 step 2「设 `matched_to=<n4>` 或分析器显式配对」的二选一歧义。
- **选择轴 = live `inf_ratio`**（roadmap C10 pass line 的轴，理由见下 D6）。选 `argmin |Δinf_ratio|`。
- **容差**：先验 **0.03**（= C9 半档）。若最近候选 `|Δinf_ratio| > 0.03` → **无有效对照 → `overall="pending"`**（不是 fail，不是 pass；镜像现有 N1 无 periodic → pending 语义，`analyze_n1_live.py:344`）。
- **并列防呆**：若最近两候选彼此 `|Δinf_ratio|` 差 < 1e-9（真并列）→ **raise ValueError**，拒绝静默选边（镜像 `match_periodic:273` 对 >1 匹配的 fail-loud 立场），由 runner 显式加/去候选消歧。
- **`periodic_pass_n4 = (|Δinf_ratio| ≤ 0.03) and (N4_SR ≥ periodic_SR)`**；判决表并报 Δinf + Δskip 双诊断。

**补档程序**（当无候选落进容差时，可选、非变异）：走 `run_n1_live.py --gate-family n1`(periodic 路径) 加 1–2 个 (cache_len, inference_len) YAML（bracketing N4 的 live inf_ratio）各 500 ep，用**全新 run_id**；其 `matched_to` 留 null 或仅信息性（N4 配对靠 inf_ratio 选，不依赖 matched_to），故仍不触碰任何既有 manifest。

**设计决策 D6（G1 焦点）**：为何匹配轴从 N1 的 skip% 改成 N4 的 inf_ratio？—— N1 的 skip 几乎全落 MISS 步（skip%≈inf 增量≈0，两轴等价，故 `match_periodic` 用 skip%）；N4 的 V2 skip 落 HIT 步（skip 抬 inf），skip% 与 inf_ratio **解耦**。roadmap 及格线明写"同 **inf_ratio** 下 SR ≥ periodic"，故对 N4 必须按 inf_ratio 配对才忠实于 C10 轴。N1 分支保留 skip% 配对（`match_periodic` 不动，向后兼容，不改 1b 判决）。

### 5.4 分析器旁路 N1 重放

N4 run 的 `replay_offline`（N1 机器离线重放）无意义（N4≠N1）→ 分派点旁路。N4 的 live skip%/inf/SR 由 per-step rows 直接算（`episode_searched` 读权威 `searched`），matched-periodic 判决走 §5.3。N4 的离线自洽重放（用 N4GateState 重放 gate_rows）**列为可选后续，不入 3a**（live skip% 才是判据；1a 方法论已证离线选点迁移偏差 ≤1.8pp）。

### 5.5 N4 overall pass 公式（三条件，pass line 落地）

现有 `analyze_n1_live.py:344-348` 的 N1 `overall` = `"pass" if vs_baseline["sr_preservation_ok"] and periodic_verdict["periodic_pass"] else "fail"`（无 periodic → `"pending"`），**不含 net 门**——直接照用会让 N4 在延迟净负时误报 pass。N4 分支（`gate_family=="n4"`）改用三条件：

```
overall_n4 =
    "pending"  if periodic_verdict is None (无候选落进 §5.3 容差)
    else "pass" if (periodic_pass_n4                             # §5.3：|Δinf|≤0.03 ∧ N4_SR≥periodic_SR
                    and vs_baseline["sr_preservation_ok"]        # :318  sr_delta_pp >= -1.0  (SR≥baseline−1pp)
                    and net["stock_2.6k"] >= 0)                  # :261  net_row 的 stock 34ms 档；正=省延迟
    else "fail"
```

字段全部来自现有分析器（均已亲验）：`net = net_row(skip_pct, d_inf)`（:260，键 `opt_2.6k/stock_2.6k/opt_50k` = C9 三档 4/34/70ms）；`vs_baseline["sr_preservation_ok"]`（:318）；`periodic_pass`（→ N4 版 `periodic_pass_n4`，§5.3）。markdown 判决表与 result 行**单独列出三个布尔分量**（sr_ok / periodic_pass_n4 / net34_ok）+ 合成 `overall`，使任一失败可定位。N1 分支的 `overall`/`match_periodic` 逐字不改（向后兼容）。

---

## 6. 测试策略（全部 non-manual，小 fixture，镜像 `test_n1_gate_client.py`）

`tests/exp/test_n4_gate_client.py`：

1. **构造校验/fail-fast**：L 非 int/≤0/bool → ValueError；θ/j/M 非法 → 复用 N1 校验路径 raise（worker 启动即崩）。
2. **V1 退化**：`L = 大于测试序列长度的有限 int`（如 `10**9`，绝不触发 V2；**不用 `∞`/sentinel**，因 §3.5 要求 L 为 int≥1）时，N4 的 decide/observe 序列**逐步等于** N1（同 (θ,j,M) 下对同一 score+hit_type 序列）→ 保证 N4 是 N1 的严格超集，不回归 1b。
3. **V2 纯触发**：全 FULL_HIT 序列、N1 恒 searching → decide 序列为 `search×L, skip, search×L, skip, …`（连续缓存执行被 cap 在 L；每注入后 fh_run 清零）。
4. **run 清零**：FULL_HIT 段中插入一个 WARM_START（默认 include_ws=False）/ MISS → fh_run 归零、V2 计数重启。
5. **include_ws=True 对比**：同序列下 WARM_START 计入 run → 触发时机前移，验证参数生效。
6. **V1×V2 组合**：N1 进入 skipping（probe）与 V2 注入交错的序列 → 验证 D3（V2 注入不污染 N1 since_probe：注入前后 N1 的 skipping/probe 相位不变）。
7. **episode reset**：`episode_start`/`reset` 后 fh_run=0、`_last_v2=False`、N1 子机 reset。
8. **权威 searched 盖章**：V1 skip 与 V2 skip 都令 `__collect_meta__.searched=False`；search 步 True。
9. **fail-open**：searched 步 anomaly → force_search + fh_run=0。

分析器测试（并入 `test_analyze_n1_live.py` 或新增小用例，合成 run fixtures）：

10. **旁路 N1 重放**：N4 manifest（`gate_family="n4"`）→ 不触发 `replay_offline`。
11. **inf_ratio 配对**：`match_periodic_n4` 按 live inf_ratio 最近选候选 + `periodic_pass_n4` 布尔正确。
12. **非变异配对（Blocking 2）**：候选 periodic manifest 的 `matched_to` 指向旧 N1 run_id，N4 仍能按最近 inf_ratio 选中该候选；断言 periodic manifest 对象**未被改写**（matched_to 原值保留、仅作 provenance 报出）。
13. **越容差 → pending（Blocking 2）**：唯一候选 `|Δinf_ratio| > 0.03` → `overall == "pending"`（非 fail 非 pass）。
14. **真并列 → raise（Blocking 2）**：两候选 inf_ratio 与 N4 等距（差 < 1e-9）→ `match_periodic_n4` raise ValueError。
15. **overall 三条件（Blocking 1）**：三个失败合成用例，证任一失败 → `overall == "fail"`——
    (a) net-negative：`net["stock_2.6k"] < 0`（构造高 d_inf/低 skip%）而 periodic_pass_n4 与 sr_ok 均真 → fail；
    (b) baseline-fail：`vs_baseline["sr_preservation_ok"] == False`（sr_delta_pp < −1）其余真 → fail；
    (c) periodic-fail：`periodic_pass_n4 == False`（N4_SR < periodic_SR 或越容差）其余真 → fail；
    (d) 三条件全真 → `overall == "pass"`（正对照）。

Runner 分派/provenance 测试（Blocking 3，新增 `tests/exp/test_run_n1_live_n4.py`，镜像 `tests/review_tests/test_n1_live_stage1b_g2.py:114` 的 `_resolve_worker_and_env` 直调模式）：

16. **N4 分派 + env**：`gate_family="n4"` + client_controlled ginfo → `_resolve_worker_and_env` 返回 `N4_WORKER_MODULE` 且设 `N4_THETA_LOW/HIGH/J/M/L`（值校对）。
17. **L fail-fast**：缺 `--L` 或 L 非 int/≤0 → `SystemExit`（driver 内早崩，镜像 N1 的 `run_n1_live.py:120-124` 参数早验）。
18. **向后兼容**：`gate_family="n1"` → `N1_WORKER_MODULE` 且设 `N1_*`（不设 N4_*）；periodic ginfo → `DEFAULT_WORKER_MODULE`——两者与现路径逐字一致（既有 `test_n1_live_stage1b_g2.py` runner 断言仍绿）。
19. **manifest provenance**：`build_manifest` 对 N4 写 `gate_family="n4"`、`L=<int>`；对 N1/periodic 写 `gate_family` 默认（`"n1"`/兼容）、`L=None`——旧 manifest 字段不缺失、既有分析器读取不破。

**Blast radius**：纯 exp/ + tests/exp/，零 src。Verify 用 `uv run pytest tests/exp -m "not manual"`（Stage 2 确认的正确范围，~17s），**不跑全量 GPU 套件**。既有 `tests/review_tests/test_n1_live_stage1b_g2.py` 的 runner/analyzer 断言须仍全绿（N1/periodic 向后兼容的回归护栏）。

---

## 7. 执行拓扑（live run — owner 确认后单独执行，不在 G1→Code 自动起）

- 复用 1b 拓扑模板（memory `project_n1_stage1b_live_run` / `reference_device_topology`）：GPU server（a100 独占 util 有效 / jupyter 公用只看 inf/s）＋ timan107 client；单 server 封顶 3 replica；1 进程 = 1 WS 连接。
- 6 个 N4 run + 可能的 1–2 periodic 补档，各 500 ep；同 conductor / 同 inits，配对键 `(task_id, subset_init_state_idx)`（与 1b 一致，保 N4-vs-periodic 配对纯净）。
- **无人值守红线**：不擅起 `run_in_background`（触发审批弹窗，feedback_no_background_tasks_unattended）；长跑用 owner 确认的会话或 L3 cron 巡检；关键节点（stage DONE / server 起关 / 拓扑变更）才汇报（feedback_chatroom_important_only / feedback_handoff_no_realtime_update）。
- 崩溃恢复：run_n1_live 增量 append + ConductorDriver ep-level resume（同 journal relaunch，reference_conductordriver_ep_resume），跨中断可续。

---

## 8. 风险登记

| # | 风险 | 影响 | 缓解 |
|---|------|------|------|
| R1 | `__hit_meta__.hit_type` 在某判器路径缺失 | V2 数不准 run | `_read_hit_type` 缺失→保守清零（不误延长）；`_build_hit_meta` 已验对 None result 也发 MISS 占位（:492），全路径有值 |
| R2 | V2 注入污染 N1 滞回相位 | V1 行为漂移、回归 1b | D3 冻结 N1（`_last_v2`）；测 6 显式验相位不变；测 2 验 L 大到不触发（`10**9`）时逐步等于 N1 |
| R3 | matched-periodic 覆盖不到 N4 inf_ratio | pass line 无对照 | §5.3 补档规则：先复用近邻，超容差才补 1–2 档；periodic run 廉价（复用 harness） |
| R4 | skip% vs inf_ratio 轴混淆 | 误判 pass | D6 明确 N4 按 inf_ratio 配对；双诊断同报；N1 分支不动 |
| R5 | WARM_START 计入口径争议 | run-length 定义偏差 | D1 默认 FULL_HIT-only（同 stage2_common），include_ws 参数留 sanity；测 5 覆盖 |
| R6 | run_n1_live 加 gate-family 破坏 n1/periodic 现行 | 回归 1b harness | 新分支默认 n1，现行路径不改；测既有 `test_run_gate_sweep`/harness smoke 仍绿 |
| R7 | live 反事实：V2 注入改执行流，离线不可评 | 只能 live 判 | C8/C11：本就设计为 live（Stage 3 定位）；离线仅选 L 先验，SR 判决全 live |
| R8 | L 三档不足以定剂量曲线 | 结论粗 | 先验 {6,8,12} 源自 2a H1 饱和证据；若三档同号可加档，但 F12 已提示低剂量足够 |

---

## 9. 及格线与交付（收敛判据）

- **代码交付**（本 plan G1→Code→G2→Verify 范围）：§4 全部文件 + `tests/exp -m "not manual"` 全绿 + 零 src diff。
- **实验判决**（live run 后，另阶段）：§1.2 三条 pass line 在 (suite, L) 网格上的满足情况 → `stage3_n4_live.md` + roadmap §5 回填 Stage 3 判决 + 对 3b 的净指令（N4 是否胜出、胜出操作点 (θ,j,M,L)）。

---

## 10. 范围外 / 押后

- **3b 定型服务器化**：3a 判 N4 胜出后另起（扩展 `ScoreHysteresisGate` + 操作点 YAML，1c 管道现成）。
- **N4 离线自洽重放**（N4GateState 重放 gate_rows）：可选 sanity，不入 3a。
- **危险步靶向**：2c 否决，永久排除本条线。
- **C1 标定门**（连续缓存执行 run 长度 / 注入相位作特征）：Stage 4，training-free 先行完成后。

---

## Review Log

（G2 外审 append。Executor 逐条回应 Accepted/Rejected，见 execution_authority §10。）

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-07-05 08:20 CDT

#### Verdict

NEEDS REVISION。核心 N4 state/client/runner/analyzer 逻辑和 non-manual scoped tests 通过，但报告输出未满足本 plan §5.5 的 N4 可审计性要求，且 markdown 口径文本仍是 Stage 1b/N1 口径。该问题会污染 Stage 3a live 结论报告，需修正后再放行。

#### Verification

- `python -m py_compile exp/gate_research/n4_gate_client.py exp/gate_research/worker_entry_n4.py exp/gate_research/run_n1_live.py exp/gate_research/analyze_n1_live.py` — PASS
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run pytest tests/exp/test_n4_gate_client.py tests/exp/test_run_n1_live_n4.py tests/exp/test_analyze_n1_live.py -q` — PASS, 80 passed
- 独立渲染检查：构造含 `overall_components={"sr_ok": True, "periodic_pass_n4": True, "net34_ok": False}` 的 N4 row 调 `render_md([row])`，输出仍为 `# N1 Live 验证结果（Stage 1b）`，主表没有 `periodic_pass_n4` / `net34_ok` 分量列，末尾口径仍描述 N1 `|Δskip| <= 2pp` overall，不描述 N4 `|Δinf| <= 0.03` + `net@stock_2.6k >= 0` 三条件。
- 数据形态抽查：现有 `exp/gate_research/data/n1_live/*/manifest.json` 的 periodic/N1 `(suite, config)` 一致，`match_periodic_n4` 的同 `(suite, config)` 非变异候选策略在既有 1b manifest 形态下可找到候选。

#### Blocking issue

1. `analyze_n1_live.render_md()` 未按 §5.5 展示 N4 三个 boolean 分量，且 N4 报告口径仍使用 N1/Stage1b 文案。
   - 代码已把 `overall_components` 写入 result-manifest row（`_strip_units` 会保留），但 markdown 判决表没有单独列出 `sr_ok / periodic_pass_n4 / net34_ok`。
   - `render_md()` 标题仍是 `# N1 Live 验证结果（Stage 1b）`。
   - 口径段仍写 “同预算 N1 SR ≥ periodic（|Δskip| ≤ 2pp）”，这对 N4 是错误口径；N4 需要说明 inf_ratio 配对、`periodic_pass_n4`、SR preservation、`net["stock_2.6k"] >= 0` 三条件。
   - 影响：Stage 3a live report 即使 result JSON 正确，human-facing markdown 也会误导审查者，无法从主报告定位 N4 overall 失败原因（尤其 net34 失败但 periodic/SR 均 pass 的情况）。

#### Required fix

- 修改 `render_md()`：当输出包含 `gate_family == "n4"` 的 row 时，markdown 必须显式展示 `sr_ok / periodic_pass_n4 / net34_ok / overall`。可接受实现：
  - 在主表增加适用于 N4 的组件列；或
  - 增加一个 N4-only verdict components 表（每个 N4 run 一行），但必须和 `overall` 同页可见。
- 同步修正 markdown 标题/口径：混合 N1/N4 时不要称为单纯 “N1 Stage1b”；N4 口径必须写 `|Δinf| <= 0.03`、`N4_SR >= periodic_SR`、`SR >= baseline - 1pp`、`net@stock_2.6k >= 0`。
- 增加一个轻量测试覆盖 `render_md()` 对 N4 row 的输出，至少断言包含 `periodic_pass_n4` 和 `net34_ok`（或等价列名）以及 N4 inf/net 口径，防止后续回归。

#### Non-blocking notes

- `n4_gate_client.py` 的 V1 嵌入、V2 FULL_HIT run cap、V2 skip 不污染 N1 phase、fail-open 和 `searched` 盖章路径与 plan 一致；已有单测覆盖充分。
- `run_n1_live.py` 的 N4 worker/env 分派和 `--L` fail-fast 通过 scoped tests。
- `match_periodic_n4()` 非变异匹配实现符合 G1 决议；现有 periodic manifests 的 `(suite, config)` 与对应 client-controlled runs 一致，候选选择前提成立。

### G2 Round 2 — Executor — 2026-07-05

审查发现属实——我 §4 Code 时把 `overall_components` 只写进了 result-manifest row（JSON），漏了 §5.5 明写的 markdown 单列三分量要求，且沿用了 N1 标题/口径。**Accepted**，已改 `render_md()`（G2 迭代改代码）。

- **Accepted**（Blocking — render_md 未落实 §5.5 N4 可审计性）—
  1. **标题自适应**：`has_n4` 时标题 `# Gate Live 验证结果（N4 混合门 / Stage 3a）`，主表 SR 列头改 `SR(run/base)`（不再把 N4 的 SR 误标 N1）；**N1-only 报表逐字保持** `# N1 Live 验证结果（Stage 1b）` + 原列头（byte-identity，不回归已产出的 Stage 1b 报告）。
  2. **N4 三分量表**：新增 `_n4_components_table()`，`has_n4` 时在主表后渲染 `| N4 run | L | sr_ok | periodic_pass_n4 | net34_ok | overall |`，每个 N4 run 一行、与 `overall` 同页可见；pending（无 inf 内候选）行分量列显 `—`。这样"net34 失败但 periodic/SR 均 pass"可直接从主报告定位。
  3. **N4 口径**：`has_n4` 时在口径段追加 **N4 overall pass 三条件**——matched periodic 按 **|Δinf| ≤ 0.03** 配对（inf_ratio 轴非 skip%）下 **N4_SR ≥ periodic_SR**、**SR ≥ baseline − 1pp**、**net@stock_2.6k ≥ 0**；inf 容差内无候选=pending、并列=raise。
  4. **回归测试**：新增 `test_render_md_n4_shows_components_and_caption`（断言标题含 Stage 3a、含 `periodic_pass_n4`/`net34_ok` 列、含 N4 三条件口径、含定位到 net34 失败的分量行）+ `test_render_md_n1_only_keeps_stage1b_title`（守 N1-only 向后兼容：标题不变、无 N4 列）。

**Non-blocking notes**（审查者确认 N4 state/client/runner/analyzer 逻辑与 plan 一致）：无需改动，接受。

自检（无程序效力）：`tests/exp` **897 passed**（+2 render 测试）、`test_n1_live_stage1b_g2.py` 回归 **10 passed**、ruff clean。修订仅动 `analyze_n1_live.render_md`（+ `_n4_components_table` helper）与 `test_analyze_n1_live.py`。重新提交 G2。

### G2 Round 3 — Reviewer — APPROVED — 2026-07-05 08:52 CDT

#### Verdict

APPROVED。Round 1 blocking 已消除：N4 markdown 报告现在显式展示 `sr_ok / periodic_pass_n4 / net34_ok / overall`，标题和口径也已改为 Stage 3a/N4 语义；N1-only Stage 1b 报告路径保持兼容。无新增重大问题。

#### Verification

- `python -m py_compile exp/gate_research/n4_gate_client.py exp/gate_research/worker_entry_n4.py exp/gate_research/run_n1_live.py exp/gate_research/analyze_n1_live.py` — PASS
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run pytest tests/exp/test_n4_gate_client.py tests/exp/test_run_n1_live_n4.py tests/exp/test_analyze_n1_live.py -q` — PASS, 82 passed
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run pytest tests/review_tests/test_n1_live_stage1b_g2.py -q` — PASS, 10 passed

#### Review notes

- `_n4_components_table()` 覆盖了 pending 与 fail-localization 场景；`test_render_md_n4_shows_components_and_caption` 明确锁住 `periodic_pass_n4`、`net34_ok`、`|Δinf| <= 0.03` 与 `net@stock_2.6k >= 0` 口径。
- `render_md()` 的 N1-only 分支保留 Stage 1b 标题和无 N4 分量列；既有 Stage 1b review regression 通过。
- N4 state/client/runner/analyzer 核心逻辑仍限 `exp/` + `tests/exp/`，零 `src/` blast radius；本轮未运行全量/GPU/manual 测试。
