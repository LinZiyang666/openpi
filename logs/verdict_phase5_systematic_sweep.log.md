# Verdict Phase 5 — Online 因子系统性探索 (5-group × 48-cell × 100ep)

**状态**：`Plan` — G1 APPROVED Round 3 (2026-05-09)；准备进入 §4 Code
**等级**：L2 — 多文件特性（exp/ 新 phase5 子包：spec + runner + 4 mode CLI + analysis；**无 src/ 改动**，复用 phase3 composer + solver）
**职权**：Execution
**负责人**：LinZiyang666
**日期**：2026-05-09
**关联**：
  - 前置 plan：[`logs/verdict_phase4_weight_sweep.log.md`](verdict_phase4_weight_sweep.log.md)
  - 前置数据：
    - `exp/verdict_factor_judge/data/phase3/per_yaml_summary.jsonl`（176 cells × 100ep — recipe + threshold baseline）
    - `exp/verdict_factor_judge/data/phase5/per_yaml_summary.jsonl`（48 cells × 500ep — anchor 真值 + R4 first-measure）⚠️ 注：`data/phase5/` 是 **phase4 stage5 复测**，不是本次 phase5 实验；本次实验数据落 `data/phase5_systematic/`
  - 前置分析：
    - [`exp/verdict_factor_judge/analysis/phase3/results.md`](../exp/verdict_factor_judge/analysis/phase3/results.md)
    - [`exp/verdict_factor_judge/analysis/phase4/stage5/results.md`](../exp/verdict_factor_judge/analysis/phase4/stage5/results.md)

---

## §0 背景

### §0.1 Gate 政策

L2 — G1（plan 审）+ G2（code 审）双门，依 WA §2.1 / `protocols/execution_authority.md` §10。

**无 src/ 改动**：phase 4 已把 composer 改为真加权和（Σ w·contrib / Σ w），solver 已支持 `composer_weights` passthrough。phase 5 不再触碰 `src/openpi/cache/`。

Gate 覆盖的 exp/ 改动：
- `exp/verdict_factor_judge/phase5/__init__.py` — 子包 marker
- `exp/verdict_factor_judge/phase5/spec.py` — 5 group 的 cell 生成器（G1/G2/G3/G4/G5）+ recipe / yaml builder
- `exp/verdict_factor_judge/phase5/runner.py` — 4-mode CLI（`emit-warmup-yamls` / `run-warmup` / `emit-eval-yamls` / `run-eval`）；phase5-local `_run_one_cell_phase5` 与 `_dump_decision_gate_table_phase5`，**结构 mirror** phase4 同名 helper 的 preload-before-load 顺序但**不直接调用** phase4 helper（B2 修订决议；§3.3）
- `exp/verdict_factor_judge/analysis/phase5/plot_pareto.py` — 240 cell × 5 group 染色 Pareto 图（同 phase4 stage5 模板）
- `exp/verdict_factor_judge/analysis/phase5/plot_heatmaps.py` — G1/G5 的 (axis × axis) 2D heatmap
- `exp/verdict_factor_judge/analysis/phase5/results.md` — 跑完后追写

Gate 覆盖的 tests（按 WA §3.1）：
- `tests/exp/test_phase5_spec.py` — 5 group cell 计数 / recipe 生成器形状 / 不重复 yaml_id 不变量 / channel-axis 覆盖
- `tests/exp/test_phase5_runner.py` — 4-mode CLI dispatch / per-group warmup 解决 / `--cell-ids` substring filter / `--group` allowlist filter
- `tests/exp/verdict_factor_judge/test_phase5_yaml_emission.py` — 整链路 yaml_id 唯一性 / factor 集合与 group 设计一致性 smoke

### §0.2 实验目标

Stage 1+2+5 把 **offline factor 维度**几乎扫透：
- offline desc（R2 jerk/dir/disp/path）— 在 noise 内 equivalent
- offline weight α（R1）— α 中点不优于端点（fusion 假设破裂）
- offline W-FUT 双窗（R4）— short-window edge advantage（marginal）
- cascade threshold 4×4 grid — phase3 sweep 过

但 **online factor 维度系统性没扫**：
1. **online 单窗口** — 全锁 _W_K3 = (3,3)，没动过其他 (P, F)
2. **online 多窗口组合** — 一个 online factor 同时配多个窗口 (`windows=[w1, w2, ...]`) 没用过
3. **online 因子权重 R3** — 设计了 5 patterns 但 stage1+2+5 全跳过
4. **多因子组合** — 限定 4 desc × 2 channel × 1 window 的笛卡尔，没探子集
5. **threshold** — phase3 4×4 grid 但每 cell 都用固定 online (3,3)，threshold 与 online 配置耦合未解

phase 5 用 100ep × 240 cell 把这 5 维一次扫完，每 group 内 SR/inf 响应曲线给 phase 6（如有）锁定子优区。

### §0.3 Scope (in / out)

**In**：
- 单 cfg：spatial16（`spatial16_w8_d4`）
- 5 个探索 group，每 group 48 cell，合计 **240 cell × 100ep × 1 seed**
- 每 cell 评估 5 worker，~5 min/cell；6 server 并跑，瓶颈 ~4 h
- 6-server split 4:4:4:3:3:3（**S1=48 / S2=48 / S3=45 / S4=33 / S5=33 / S6=33**）

**Out**：
- 其他 cfg（clip / max_pool）
- multi-seed（每 cell 单 seed；100ep noise floor 已在 §6.1 验明，作 known limitation）
- src/ 改动
- 后续 phase（phase 5 仅 sweep；不改 production gate）

### §0.4 噪声 vs 信号 — 100ep 决议规则

stage 5 已证 100ep 95% CI 半宽 ≈ ±4.4pp（SR=0.95 时）。phase 5 不上多 seed（算力不够），但**每 group 内决议规则降级**：

| group | 决议规则 |
|---|---|
| G1 (单窗口) | 同 (channel, desc) 内 6 windows 排名；只声明 SR top-1 vs top-2 差距 ≥ 5pp 的为 winner，否则报告 inconclusive |
| G2 (多窗口) | 同 (channel, desc, base) 内 12 combos 排名；同样 5pp 阈值 |
| G3 (权重) | 12 patterns 排名；5pp 阈值 |
| G4 (多因子组合) | 12 子集排名；5pp 阈值 |
| G5 (threshold) | 16 cell × 3 recipe = 48 cell；按 (FH, WS) 散点图找 Pareto 前沿，**不**做 winner 排名 |

任何 group 都不强制选 single winner —— phase 5 的产出是**响应曲线 + Pareto 前沿**，不是新 production config。

---

## §1 5 group cell 设计（240 cell 全列出）

### §1.1 共用约定

- **Base offline factor set 取决于 `base_recipe`（不是单一固定集合）**：
  - `base_recipe = p1` → base offline = **g1**（offline_state W-FUT 4 desc × 2 win = 8 keys）
  - `base_recipe = p2` → base offline = **g10**（offline_action W-FUT 4 desc × 2 win = 8 keys）
  - 选哪个 base 决定 offline 这一半的 channel 与 keys；online sweep 只动 online 那一半。G2 单用 p1（理由：keep G2 维度数 = 12×4=48；p2 的多窗口 sweep 移到未来 phase）；G3/G4 都用 p1 + p2 两 base；G1 也都用。**G5 (threshold) 完全不注入 base offline** — 走 phase4 p1/p2 + phase3 g6 三个固定 recipe，对它们各自的现存 factor list 套 16 (FH, WS) cell sweep。
- **yaml_id 命名约定**：`spatial16_w8_d4_phase5_<group>_<axis_summary>__<cell_tag>.yaml`，e.g. `spatial16_w8_d4_phase5_g1_p1_state_jerk__win-3-3.yaml`（base_recipe + channel + desc + window 全编码进 yaml_id 保证唯一性 — 见 §11）
- **Composer**：`weighted_sum_zero_nan` (warm_start_t=0.5, warm cost 0.75, FH/WS 双 thresh)
- **Threshold solver pattern（mirror phase4）**：`phase3.threshold_solver.reconstruct_scores(jsonl_path, recipe, composer_weights=cell.weights)` + `derive_thresholds(scores, fh_ratio, ws_ratio)` — **不调** `solve_recipe`（后者内部 hardcode 等权重，不接 `composer_weights`，G3/G4/G5 任何非等权 cell 用它会拿到错的 thresholds）。详见 §3.3。
- **Locked (FH, WS)**：除 G5 外，全锁 (0.5, 0.5)（phase3 g1 ultra-cheap anchor cell）

### §1.2 G1：online 单窗口 sweep（48 cell）

**axes**：
- 6 windows: `(0,3), (0,5), (1,1), (3,3), (5,5), (7,7)`
- 2 channel: `state`, `action`
- 2 desc: `jerk`, `dispersion`
- 2 base recipe: `p1` (offline = g1 W-FUT all-4-desc / state), `p2` (offline = g10 W-FUT all-4-desc / action)

→ 6 × 2 × 2 × 2 = **48 cell**

每 cell：base offline 8 keys + 1 online factor (1 desc × 1 channel × 1 window) = 9 keys。weight uniform 1/9。

**warmup yaml 数**：每 (channel × desc × window) 组合的 raw 都不同 → 每 cell 1 个独立 warmup yaml。但 base offline 8 keys 共享 → warmup factor list = 8 offline + 1 online；总 48 个 warmup yaml。**单 server 顺序跑 48 × 0.5 min ≈ 24 min warmup overhead/server**。

### §1.3 G2：online 多窗口组合（48 cell）

**axes**：
- 12 multi-window combos:
  - `[(0,3), (3,3)]` — future-near + sym-near
  - `[(0,5), (3,3)]` — future-far + sym-near
  - `[(0,3), (5,5)]` — future-near + sym-far
  - `[(0,3), (3,0)]` — future + past
  - `[(0,5), (5,0)]` — future-far + past-far
  - `[(1,1), (3,3)]` — sym-stack short
  - `[(3,3), (5,5)]` — sym-stack mid
  - `[(5,5), (7,7)]` — sym-stack long
  - `[(1,1), (3,3), (5,5)]` — sym-tower 3
  - `[(0,3), (3,3), (5,5)]` — fut+sym-tower 3
  - `[(0,3), (1,1), (3,3), (5,5)]` — full small
  - `[(0,3), (0,5), (3,3), (5,5), (7,7)]` — full ladder
- 2 channel × 2 desc × 1 base recipe = 4 (用 p1 only，p2 留给 G3-5)

→ 12 × 4 = **48 cell**

每 cell：base offline 8 keys + N online factors (1 desc × 1 channel × M windows) → N keys 总 = 8 + M（M 取 combo 长度，2-5 之间）。weight uniform。

**warmup yaml 数**：每 cell 1 个（48 个 warmup yaml）。

### §1.4 G3：online 因子权重 patterns 拓展（48 cell）

**axes**：
- 12 weight patterns `(jerk_share, disp_share)` — 拓展 phase4 R3_ONLINE_PATTERNS 5 个：
  - `(1, 1)` uniform
  - `(2, 1)` jerk-heavy
  - `(1, 2)` disp-heavy
  - `(1, 0)` jerk-only
  - `(0, 1)` disp-only
  - `(3, 1)` jerk-strong
  - `(1, 3)` disp-strong
  - `(4, 1)` jerk-dominant
  - `(1, 4)` disp-dominant
  - `(2, 3)` slight-disp
  - `(3, 2)` slight-jerk
  - `(5, 1)` jerk-extreme （替换原 `(0.5, 0.5)` —— 后者归一化后与 `(1, 1)` 同构，作 deliberate replicate 价值低，移除让出 cell 给非冗余 pattern）
- 2 channel: `state`, `action`
- 2 base recipe: `p1`, `p2`

→ 12 × 2 × 2 = **48 cell**

每 cell：base offline 8 keys + 2 online keys (jerk + disp on chosen channel, 单窗 (3,3))。**weight 公式（offline / online 各 50% grand total）**：
- offline 8 keys 每个 = `0.5 / 8 = 0.0625`（offline 总权 = 0.5）
- online jerk = `0.5 × pattern[0] / (pattern[0] + pattern[1])`
- online disp = `0.5 × pattern[1] / (pattern[0] + pattern[1])`
- online 总权 = 0.5；grand total = 1.0
- 退化 case：`(1, 0)` 时 disp_w = 0；`(0, 1)` 时 jerk_w = 0；solver 在 `composer_weights` 里把这两个 key 的权写 0，不参与 score 计算（与 `WeightedSumZeroNanComposer` 加权和语义一致）

这与 phase4 R1 α=0.5 等价（offline/online 50/50 split），但额外 sweep online 内部 desc 权重（R3 维度），是 R1×R3 的局部交叉。

**warmup yaml 数**：每 (channel) × base recipe 共享 → 2 channel × 2 recipe = 4 warmup yaml（同一 channel 同一 recipe 的 12 patterns 共享 raw）。

### §1.5 G4：多因子组合（48 cell）

**axes**：
- 12 factor 子集（每子集是一个 (offline_desc_subset, online_desc_subset) 二元组）：
  - 全集：(jerk+dir+disp+path offline) + (jerk+disp online) — full 10
  - drop offline path：(jerk+dir+disp) + (jerk+disp) — 8
  - drop offline disp：(jerk+dir+path) + (jerk+disp) — 8
  - drop offline dir：(jerk+disp+path) + (jerk+disp) — 8
  - drop offline jerk：(dir+disp+path) + (jerk+disp) — 8
  - offline only：(jerk+dir+disp+path) + () — 8
  - online only：() + (jerk+disp) — 2 ← 退化对照（== g6）
  - offline jerk + online jerk：(jerk) + (jerk) — 3 (1 offline + 2 win × 1 desc + 2 channel)
  - offline disp + online disp：(disp) + (disp) — 3
  - W-FUT only offline + full online：(jerk+dir+disp+path W-FUT only-(0,3)) + (jerk+disp) — 6
  - online + offline jerk pair：(jerk W-FUT) + (jerk+disp) — 4
  - 单 jerk 全栈：(jerk W-FUT + W-K3) + (jerk W-K3) — 5
- 2 channel: state, action（适用 online 部分）
- 2 base recipe: p1, p2

→ 12 × 2 × 2 = **48 cell**

每 cell：weight uniform across active keys。

**warmup yaml 数**：每 cell 1 个（factor list 不同 → 不能复用），48 yaml。

### §1.6 G5：threshold (FH, WS) grid × 3 recipe（48 cell）

**axes**：
- 16 (FH, WS) cells: (FH, WS) ∈ {0.2, 0.3, 0.4, 0.5} × {0.2, 0.3, 0.4, 0.5}
- 3 recipe:
  - `p1_state_fut_online_act` (phase4 p1 — 10 keys，α=1.0 即 R2 uniform)
  - `p2_action_fut_online_act` (phase4 p2)
  - `g6_f1a_a_d_jerk_curv_pair` (phase3 pure online winner)

→ 16 × 3 = **48 cell**

每 cell 用对应 recipe 的固定 factor + weight 配置，唯独 (FH, WS) sweep。Threshold 由 solver 在 4×4 grid 上自动派生（用 `reconstruct_scores(jsonl_path, recipe, composer_weights=recipe_weights)` + `derive_thresholds(scores, fh, ws)`，§3.3）。

**warmup raw 来源映射（0 个新 warmup，全复用历史 jsonl）**：

| recipe | factor_raw jsonl 来源 |
|---|---|
| `p1_state_fut_online_act` | `exp/verdict_factor_judge/data/phase4/warmup_factor_raw/p1_state_fut_online_act.jsonl` (phase4 stage1 产出) |
| `p2_action_fut_online_act` | `exp/verdict_factor_judge/data/phase4/warmup_factor_raw/p2_action_fut_online_act.jsonl` (phase4 stage1 产出) |
| `g6_f1a_a_d_jerk_curv_pair` | `exp/verdict_factor_judge/data/phase3/warmup/spatial16_w8_d4_phase3_g6_f1a_a_d_jerk_curv_pair__warmup.jsonl` (phase3 stage1 产出) |

phase5 G5 的 emit-eval-yaml 阶段直接 load 上述路径的 raw 走 solver；run-eval 阶段 server 端 `--warmup-jsonl-dir` 传 phase3 warmup dir + phase4 warmup_factor_raw dir 两个，runner `_run_one_cell_phase5` 在 preload 时按 recipe 找对应 jsonl（与 phase4 stage5 batch1 / batch6 跑 phase3 cell 时复用 `--warmup-jsonl-dir exp/verdict_factor_judge/data/phase3/warmup` 是同一模式）。

### §1.7 cell 总数 sanity

| group | cells |
|---|---:|
| G1 单窗口 | 48 |
| G2 多窗口 | 48 |
| G3 权重 | 48 |
| G4 多因子 | 48 |
| G5 threshold | 48 |
| **总** | **240** |

---

## §2 6-server split（4:4:4:3:3:3）

### §2.1 计算

ratio_unit = 4+4+4+3+3+3 = 21
240 / 21 = 11 整轮 + 9 余
- 11 整轮：S1/S2/S3 各 11×4 = 44, S4/S5/S6 各 11×3 = 33
- 9 余 cell mod 顺序填入：S1+4, S2+4, S3+1（4+4+1=9）

| Server | 机器 | 公网入口 | cell 数 | 4h ETA |
|---|---|---|---:|---|
| S1 | timan107 (frp 8998) | 155.98.36.32:8998 | **48** | 4h ✓ |
| S2 | timan107 (frp 8999) | 155.98.36.32:8999 | **48** | 4h ✓ |
| S3 | timan107 (frp 9000) | 155.98.36.32:9000 | **45** | 3.75h |
| S4 | 直连 8001 | 149.165.151.106:8001 | **33** | 2.75h |
| S5 | 直连 8002 | 149.165.151.106:8002 | **33** | 2.75h |
| S6 | 直连 8003 | 149.165.151.106:8003 | **33** | 2.75h |
| **合计** | | | **240** | 瓶颈 ~4h |

### §2.2 cell-to-server 分配策略

按 yaml_id 字典序（240 cell 全 emit 后排序）round-robin 分到 6 server，权重 4:4:4:3:3:3。

**Pseudo**：
```
slots = ["S1"]*4 + ["S2"]*4 + ["S3"]*4 + ["S4"]*3 + ["S5"]*3 + ["S6"]*3   # 21 slots
allocation = {S: [] for S in 6 servers}
for i, cell in enumerate(sorted_cells):
    allocation[slots[i % 21]].append(cell.yaml_id)
```

→ S1/S2 拿 48 (= 11×4 + 4 余)、S3 拿 45 (= 11×4 + 1)、S4/S5/S6 拿 33 (= 11×3)。**已验算合 240**。

每 server 的 cell list 在 §3.5 由 spec 模块导出 6 个 `--cell-ids` 子串文本（substring allowlist 同 phase4 stage5 §3）。

### §2.3 group × server 交叉

**不**按 group 分 server（避免某 server 全跑 G5 threshold 而别的 server 全跑 G4 重 warmup —— overhead 不均）。240 cell 全字典序混排，每 server 都跑 5 个 group 的混合 → warmup 时间均摊。

---

## §3 spec / runner 设计

### §3.1 phase5/spec.py 公开面

```python
GROUPS = ("g1_single_window", "g2_multi_window", "g3_weight",
          "g4_multi_factor", "g5_threshold")

# axis 枚举
G1_WINDOWS: tuple[tuple[int,int], ...] = ((0,3),(0,5),(1,1),(3,3),(5,5),(7,7))
G1_CHANNELS = ("state", "action")
G1_DESCS = ("jerk", "dispersion")
G1_BASE_RECIPES = ("p1", "p2")

G2_MULTI_COMBOS: tuple[tuple[tuple[int,int], ...], ...] = (...)  # 12 个
G2_CHANNELS = G1_CHANNELS
G2_DESCS = G1_DESCS

G3_WEIGHT_PATTERNS: tuple[tuple[float,float], ...] = (...)  # 12 个

G4_FACTOR_SUBSETS: tuple[dict, ...] = (...)  # 12 子集 (offline_desc_subset, online_desc_subset)

G5_THRESHOLD_GRID: tuple[tuple[float,float], ...] = tuple(
    (fh, ws) for fh in (0.2, 0.3, 0.4, 0.5) for ws in (0.2, 0.3, 0.4, 0.5)
)
G5_RECIPES = ("p1_state_fut_online_act", "p2_action_fut_online_act",
              "g6_f1a_a_d_jerk_curv_pair")

# 入口
def generate_g1_cells() -> list[Cell]: ...   # → 48
def generate_g2_cells() -> list[Cell]: ...   # → 48
def generate_g3_cells() -> list[Cell]: ...   # → 48
def generate_g4_cells() -> list[Cell]: ...   # → 48
def generate_g5_cells() -> list[Cell]: ...   # → 48

def generate_all_cells() -> list[Cell]: ...  # → 240, 字典序排序

def allocate_to_servers(cells: list[Cell],
                        ratio: tuple[int,...] = (4,4,4,3,3,3),
                       ) -> dict[str, list[Cell]]: ...

def build_warmup_yaml_for_cell(cell: Cell) -> dict: ...
def build_eval_yaml_for_cell(cell: Cell, fh_thr: float, ws_thr: float) -> dict: ...
```

`Cell` dataclass：
```python
@dataclass(frozen=True)
class Cell:
    yaml_id: str
    group: str            # "g1_single_window" | ... | "g5_threshold"
    base_recipe: str      # "p1" | "p2" | "g6" (G5)
    factors: tuple[FactorBlock, ...]
    weights: dict[str, float]
    fh_ratio: float
    ws_ratio: float
    declared_keys: tuple[str, ...]
    warmup_yaml_id: str   # 哪些 cell 共享 warmup
```

### §3.2 phase5/runner.py 4-mode CLI（同 phase4）

```bash
# 1. emit warmup yamls (all cells，去重 by warmup_yaml_id)
uv run python -m exp.verdict_factor_judge.phase5.runner \
    --mode emit-warmup-yamls --groups g1,g2,g3,g4,g5

# 2. run warmups (server 必备，本 cmd 一次跑所有去重 warmup)
uv run python -m exp.verdict_factor_judge.phase5.runner \
    --mode run-warmup --groups g1,g2,g3,g4,g5 \
    --host <H> --port <P>

# 3. emit eval yamls (per-cell threshold 由 solver 派生)
uv run python -m exp.verdict_factor_judge.phase5.runner \
    --mode emit-eval-yamls --groups g1,g2,g3,g4,g5

# 4. run eval (per-server，--cell-ids 切片)
uv run python -m exp.verdict_factor_judge.phase5.runner \
    --mode run-eval --groups g1,g2,g3,g4,g5 \
    --cell-ids <substring1> <substring2> ... \
    --host <H> --port <P>
```

`--groups` 接逗号分隔 allowlist（默认全 5 group）。`--cell-ids` 接 substring list（同 phase3/phase4 runner 语义）。

### §3.3 共享代码 / 边界

**直接复用（function-level，无 mutation）**：
- `phase3.threshold_solver.reconstruct_scores(jsonl_path, recipe, *, composer_weights=...)` — phase4 已为它加 `composer_weights` keyword-only；phase5 必须**显式传 `composer_weights=cell.weights`**（不要走 `solve_recipe` — 后者内部不接 weights，回退等权重，G3/G4/G5 任何非等权 cell 解出的 thresholds 会错）
- `phase3.threshold_solver.derive_thresholds(scores, fh_ratio, ws_ratio)` — quantile cut 派生 (FH_thr, WS_thr)
- `phase3.threshold_solver.load_per_key_finite_history` — solver 上游辅助
- `common.run_phase._build_libero_argv` / `_aggregate_sr_from_episode_json` / `_summarize_per_step_log` — server 端 LIBERO worker 启动 + episode 聚合 + per-step writer 摘要
- `common.generate_yamls.write_yaml` — yaml 落盘
- `common.v2_spec.{factor, factor_keys, build_warmup_yaml, build_eval_yaml, ...}` — yaml 块 / factor 键名生成（**不 mutate** 任何 RECIPES dict）

**模板参考但**重写**（phase5-local helper，不直接调用 phase4 的）**：
- `phase5.runner._run_one_cell_phase5(cell, ctl, args)` — 结构 mirror `phase4.runner._run_one_cell` 的 preload-before-load 顺序（`preload_normalizer_buffer(eval_yaml_id, buffer)` → `load_cache_config(eval_yaml)` → server 端 LIBERO 子进程 → 写 summary row），但**不**：
  - 读 `RECIPES_PHASE4` （phase5 用自己的 cell.weights / cell.declared_keys）
  - 走 `data/phase4/warmup_factor_raw/<recipe>.jsonl` 写死路径（phase5 G5 raw 来自 phase3 + phase4 双源，G1-G4 来自 phase5 自己的 warmup dir，路径由 `args.warmup_jsonl_dirs` 拼接）
  - 写 phase4 的 round/recipe summary 字段（phase5 写 `group` / `base_recipe` / `axis_tag` 等 phase5-specific 字段）
- `phase5.runner._dump_decision_gate_table_phase5(cells, summary, group)` — 结构 mirror `phase4.runner._dump_decision_gate_table` 但 **per-group dispatch**（G1/G2/G3/G4 走 5pp-阈值 winner rule，G5 走 Pareto frontier rule，§4）

**新增（全部）**：
- `phase5/__init__.py`
- `phase5/spec.py`
- `phase5/runner.py`（含 `_run_one_cell_phase5` + `_dump_decision_gate_table_phase5` 两个 phase5-local helper；4-mode CLI dispatch）
- `analysis/phase5/{plot_pareto.py, plot_heatmaps.py, results.md}`

**抽 generic helper 的可能性**：当前 plan **不**抽 generic `_run_one_cell` 出来共用 — 那是 cross-phase refactor，scope 超 phase 5，留 phase 6（如有）。phase5 helper 与 phase4 helper 是兄弟，不是父子。

### §3.4 warmup 总数估算

- G1: 48 个（每 cell 1 个）
- G2: 48 个
- G3: 4 个（共享）
- G4: 48 个（每 cell 1 个）
- G5: 0 个（复用 phase3/phase4 的 warmup_factor_raw）

**合计 ~148 个 warmup yaml × ~0.5 min × 5 worker × 5 worker / 6 server = ~2.5 h / server warmup overhead**

→ phase 5 总 wall-clock = 4 h eval + 2.5 h warmup ≈ **6.5 h** wall-clock 瓶颈

### §3.5 per-server cmd book

phase5 spec 跑 `allocate_to_servers()` 后导出 6 个 `--cell-ids` substring list，写入 `logs/verdict_phase5_run_commands.log.md`（在 G2 后产出）。

---

## §4 决议门 / Decision Gates

### §4.1 G1 (单窗口) decision

对每 (channel, desc, base_recipe) 三元组（共 2×2×2=8 个），从 6 windows 中找 SR 最高的 — 但只在 SR_top1 - SR_top2 ≥ 5pp 时声明 winner。

输出：`g1_decision.json` — `{(channel, desc, base): winner_window_or_None}` 8 entries。

### §4.2 G2 (多窗口) decision

类似 G1，但是对每 (channel, desc) 二元组（4 个）从 12 combos 找 winner。5pp 阈值。

输出：`g2_decision.json` — 4 entries。

### §4.3 G3 (权重) decision

对每 (channel, base_recipe) 二元组（4 个）从 12 patterns 找 winner。5pp 阈值。

输出：`g3_decision.json` — 4 entries。

### §4.4 G4 (多因子) decision

对每 (channel, base_recipe) 二元组（4 个）从 12 子集找 winner。5pp 阈值。

输出：`g4_decision.json` — 4 entries。

### §4.5 G5 (threshold) decision

对每 recipe（3 个）从 16 (FH, WS) cells 中：
- 找 SR 最高 cell
- 找 inf 最低且 SR ≥ 0.85 cell
- 找 Pareto upper frontier 全 cells
- **不**做 winner 排名

输出：`g5_decision.json` — 3 entries × `{best_sr, cheapest_above_0.85, frontier_cells}`。

### §4.6 跨 group 综合

phase 5 不强制选 single new production winner —— 输出是 5 个 group 的响应曲线 + 一个 global Pareto plot（240 cell 全部染色）。

---

## §5 测试矩阵

### §5.1 spec 测试 (`tests/exp/test_phase5_spec.py`)

- INV-1：`generate_g1_cells()` 返回 48 cell，且每 cell yaml_id 唯一
- INV-2：5 个 group 合计 240 cell，全 yaml_id 唯一
- INV-3：每 group 内 axis 覆盖完整（G1 6×2×2×2 笛卡尔覆盖；G2 12×2×2 覆盖；etc）
- INV-4：每 cell 的 `declared_keys` ⊆ factor_block 的 keys（factor list 与 weight dict 一致）
- INV-5：G5 cell 的 (FH, WS) ∈ G5_THRESHOLD_GRID 且 recipe ∈ G5_RECIPES
- INV-6：`build_warmup_yaml_for_cell` 与 `build_eval_yaml_for_cell` 调用 v2_spec.build_warmup_yaml / build_eval_yaml（不 mutate phase3.spec.RECIPES）
- INV-7：`allocate_to_servers((4,4,4,3,3,3))` 输出 = 48/48/45/33/33/33
- INV-8：G5 不带 base offline 注入（与 G1-G4 区分）
- INV-9：G3 weight 公式 — 任意 pattern `(a, b)` 解出的 weight dict 满足 `Σ offline = 0.5 ± 1e-9` 且 `Σ online = 0.5 ± 1e-9`；退化 pattern `(0, 1)` 或 `(1, 0)` 时对应 online key weight = 0（不是 NaN，不是缺 key）
- INV-10：G1/G3/G4 base offline keys 与 base_recipe 强绑定 — `base_recipe="p1"` cell 的 offline 部分 keys 全是 `<desc>_offline_state__p<P>_f<F>` 模式；`base_recipe="p2"` 全是 `<desc>_offline_action__p<P>_f<F>` 模式；不混
- INV-11（**B1 关键测试**）：构造同一 cell 但传两种不同 `composer_weights`（uniform vs heavy-skew），调用 `reconstruct_scores(..., composer_weights=...)` + `derive_thresholds(...)` 得到的 (FH_thr, WS_thr) 必须**不同**（否则证明 weights passthrough 没生效）— 防止 phase5 误调 `solve_recipe` 退回等权重的回归

### §5.2 runner 测试 (`tests/exp/test_phase5_runner.py`)

- CLI：`--mode emit-warmup-yamls --groups g1,g3` 只 emit G1 + G3 warmup
- CLI：`--cell-ids win-3-3 jerk-only` substring filter 工作
- **`_run_one_cell_phase5` invariants（phase5-local helper，不调 `phase4.runner._run_one_cell`）**：
  - **R-1 preload-before-load 顺序**：mock `ctl.preload_normalizer_buffer` 与 `ctl.load_cache_config`，跑一个 phase5 cell；assert preload call 在 load call 之前（与 phase4 R1 §0.1 / phase3 line 380-382 一致的 invariant）
  - **R-2 cell.declared_keys 接入**：mock 一个 `Cell(declared_keys=("k1", "k2", "k3"))`，verify _run_one_cell_phase5 把 `cell.declared_keys` 透传给 reconstruct_scores 的 recipe 参数（不能去读 `RECIPES_PHASE4[cell.id]` 或任何全局 dict）
  - **R-3 raw source resolution**：构造 G5 cell（recipe="g6_f1a_a_d_jerk_curv_pair"）+ G1 cell（base_recipe="p1"），各自的 jsonl path 必须按 `args.warmup_jsonl_dirs` 拼接：G5 → `data/phase3/warmup/spatial16_w8_d4_phase3_g6_*__warmup.jsonl`、G1 → `data/phase5_systematic/warmup/<cell.warmup_yaml_id>.jsonl`；**不**走 phase4 hardcoded `data/phase4/warmup_factor_raw/<recipe>.jsonl` 单源路径
  - **R-4 phase5 summary 字段**：写出的 summary jsonl row 必须含 phase5-specific 字段 `group` / `base_recipe` / `axis_tag`（不是 phase4 的 `round` / `recipe` / `alpha_star_for_round`）；缺一项即测试失败
  - **R-5 RECIPES_PHASE4 不读**：用 `monkeypatch.setattr("exp.verdict_factor_judge.phase4.spec.RECIPES_PHASE4", {})` 把全局 dict 清空，跑 _run_one_cell_phase5 仍能成功（因为 phase5 helper 只看 `cell` 对象自身，不去 phase4 全局取数据）— 这是"phase5 与 phase4 是兄弟而非父子"的硬保证
- decision_gate per-group 路由（G1/G2/G3/G4 走 5pp-阈值 winner rule，G5 走 Pareto frontier rule，§4）

### §5.3 yaml emission 集成测试 (`tests/exp/verdict_factor_judge/test_phase5_yaml_emission.py`)

- 240 cell 全 yaml emit 后，yaml_id 集合 size = 240
- 每 yaml 的 factor list 与 weights 字段 sanity（v2_spec validator pass）
- warmup yaml 去重数（per group）符合 §3.4 估算

---

## §6 风险与已知 limitation

### §6.1 100ep noise floor

stage 5 已证 100ep 95% CI 半宽 ≈ ±4.4pp。phase 5 的 G3/G5 内 SR 差距 < 5pp 时**不可分**。决议规则降级为 "winner only if SR_top1 - SR_top2 ≥ 5pp"，否则 inconclusive。

### §6.2 warmup overhead

§3.4 估算 ~2.5 h/server warmup（只 S1/S2/S3 受影响，因为 cell 多）。如果不可接受，G1/G2/G4 的 144 个独立 warmup 可以**预先在本地 emit + 一次性云端跑**（脱机 batch warmup）然后只 push factor_raw jsonl。

### §6.3 G4 子集设计偏 ad-hoc

12 子集是手挑的，没系统覆盖 (offline 4-desc 全子集 = 16 + online 2-desc 全子集 = 4 = 64 个组合)。如果用户要 systematic，G4 应改成 16 × 4 = 64 cell（超 48 预算 → 需调整其他 group 或扩 phase 5 总预算）。

### §6.4 G5 与 phase3 数据重叠

G5 跑 phase4 p1/p2 + phase3 g6 在 4×4 (FH, WS) 上 — phase3 g6 已有 16 cell 数据（100ep）。重跑是**为了在新 data acquisition pipeline 下复测**（phase5 的 per_step writer 与 phase3 略不同？需确认；如果一致，可跳 g6 16 cell → G5 总缩到 32 cell，剩 16 cell 移到 G4 加倍）。

### §6.5 stage5 vs phase5 数据目录歧义

`data/phase5/` 已被 phase4 stage5 真值复测占用（48 cell × 500ep）。本次 phase 5 系统探索数据**必须落不同目录**，方案 `data/phase5_systematic/` 或 `data/phase5_sweep/`，避免 join 时数据混淆。spec.py 默认输出路径设 `data/phase5_systematic/`。

---

## §7 工时估算

| 任务 | 时间 |
|---|---|
| Plan 写完 + G1 review | 0.5 h |
| Code: phase5/spec.py | 1.5 h |
| Code: phase5/runner.py | 1 h |
| Code: 3 个测试文件 ~150 tests | 1.5 h |
| Code: analysis/phase5/plot_*.py | 1 h |
| G2 review + 修订 | 1 h |
| Cmd book 写 | 0.5 h |
| **Code 总计** | **~7 h** |
| 6 server 跑 warmup | 2.5 h |
| 6 server 跑 240 eval | 4 h |
| 数据回收 + 分析 + results.md | 2 h |
| **执行总计** | **~8.5 h wall-clock** |

---

## §8 决策矩阵 / 待用户确认项（Plan 阶段）

| 决议 | 选项 | 推荐 |
|---|---|---|
| D1: G4 子集 12 个还是 16 个 | (a) 12 ad-hoc 守 48 cell 预算；(b) 16 系统全 subsets，G4 总 64 cell，phase 5 总 256 cell（其他 group 缩） | **(a)** — 守预算；ad-hoc 子集已含主要 ablation 模式 |
| D2: G5 跑 phase3 g6 16 cell 是否复用 phase3 数据 | (a) 跳 g6 16 cell（节省 80 ep × 16 = 1280 ep，~13 min/server）；(b) 复测以保 schema 一致 | **(b)** — schema 一致性优先于 13 min |
| D3: warmup 是否预跑到云端 batch 然后只 push factor_raw | (a) 不预跑，让 server 跑（+2.5 h overhead）；(b) 在本地 emit + 一次云端 batch 跑，跑完打包下载 factor_raw | **(a)** — 简单，符合 phase4 stage5 流程 |
| D4: data 目录 | (a) `data/phase5_systematic/`；(b) `data/phase5_sweep/`；(c) `data/phase5/` 内开子目录 `data/phase5/systematic/` | **(a)** — 短，与 stage5 `data/phase5/` 分明 |

---

## §9 入 Code 清单（G1 后）

- [ ] `exp/verdict_factor_judge/phase5/__init__.py`
- [ ] `exp/verdict_factor_judge/phase5/spec.py`（5 group cell 生成器 + Cell dataclass + allocate_to_servers + yaml builder）
- [ ] `exp/verdict_factor_judge/phase5/runner.py`（4-mode CLI；**phase5-local** `_run_one_cell_phase5` + `_dump_decision_gate_table_phase5`，不直接调 phase4 同名 helper）
- [ ] `tests/exp/test_phase5_spec.py`（含 INV-9/10/11，特别 INV-11 验 composer_weights passthrough 真生效）
- [ ] `tests/exp/test_phase5_runner.py`
- [ ] `tests/exp/verdict_factor_judge/test_phase5_yaml_emission.py`
- [ ] `exp/verdict_factor_judge/analysis/phase5/__init__.py`
- [ ] `exp/verdict_factor_judge/analysis/phase5/plot_pareto.py`
- [ ] `exp/verdict_factor_judge/analysis/phase5/plot_heatmaps.py`
- [ ] G2 review + 修订
- [ ] commit + push
- [ ] 写 `logs/verdict_phase5_run_commands.log.md`（cmd book，per-server `--cell-ids` 子串导出）
- [ ] **更新 `logs/README.md`**：在 `### Cache System` table 顶部追加 `verdict_phase5_systematic_sweep.log.md` 入口（plan 完工后）+ 跑完后追加 `verdict_phase5_run_commands.log.md` 入口（cmd book 写完后）。WA §4/§5 把 `logs/README.md` 同步列为 constitutional red line — 必做项不是可选项。

完工后用户在 6 server 上跑，下载 240 cell summary + per_step + episode_results，本地 results.md 分析（5 group 各自 SR/inf 响应 + 总 Pareto + decision_gate.json 5 个 decision file）。

---

## §10 不再考虑（明确 out）

- multi-seed（100ep × 1 seed —— noise floor 是 known limitation；如果 phase5 结论不稳定，phase 6 上 multi-seed）
- src/ composer/solver 改动
- 其他 cfg（clip / max_pool）—— spatial16 only
- Bayesian opt —— grid sweep 优先（同 phase4 §0.3 理由）
- production gate 重写 —— phase5 仅 sweep + 响应曲线

---

## §11 附录：cell_id 命名

```
g1: spatial16_w8_d4_phase5_g1_<base>_<channel>_<desc>__win-<P>-<F>
g2: spatial16_w8_d4_phase5_g2_<base>_<channel>_<desc>__multi-<combo_tag>
g3: spatial16_w8_d4_phase5_g3_<base>_<channel>__pat-<jw>-<dw>
g4: spatial16_w8_d4_phase5_g4_<base>_<channel>__sub-<subset_tag>
g5: spatial16_w8_d4_phase5_g5_<recipe>__fh<FH>_ws<WS>
```

`<combo_tag>` / `<subset_tag>` 是手取的短串（e.g. `fut-near+sym-near`, `drop-path`），spec 里有完整 mapping。

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-05-09 11:47 CDT
- [Blocking] [Concern] G5 p1/p2 cells do not preserve the fixed recipe weight configs required by §1.6. `generate_g5_cells()` sets every declared key to `1/len(declared)` (`exp/verdict_factor_judge/phase5/spec.py:490-491`), so phase4 p1/p2 threshold sweeps give online keys nonzero weight. The approved plan says G5 uses each recipe's fixed factor + weight config and identifies p1 as phase4 `alpha=1.0` / R2 uniform (`logs/verdict_phase5_systematic_sweep.log.md:200,206`); phase4 `generate_r2_weights(..., alpha_star=1.0)` gives online total `1.0 - alpha_star = 0` (`exp/verdict_factor_judge/phase4/spec.py:271-279`). This changes the experiment target and derived thresholds.
- [Blocking] [Concern] G4 subset list diverges from the approved 12-cell design. The plan's final subset is "single jerk full stack" (`logs/verdict_phase5_systematic_sweep.log.md:185`), but `G4_FACTOR_SUBSETS` emits `off-disp-on-full` instead (`exp/verdict_factor_judge/phase5/spec.py:160-163`) and never emits the single-jerk full-stack cell. The count remains 48, but one planned experimental condition is replaced.
- [Blocking] [Concern] Decision-gate output contract differs from §4. The plan requires `g1_decision.json` through `g5_decision.json` (`logs/verdict_phase5_systematic_sweep.log.md:406,412,418,424,434`) and five group decision files (§4.6), while `_dump_decision_gate_table_phase5()` writes only one aggregate `decision_gate.json` (`exp/verdict_factor_judge/phase5/runner.py:614-615`). Downstream command/result steps following the plan will not find the promised files.
- [Blocking] [Concern] New Python files violate Working Agreement §3.2 code-comment/docstring rules. `exp/verdict_factor_judge/analysis/phase5/__init__.py` is empty (no module-level docstring), and new source comments/docstrings contain Chinese text at `exp/verdict_factor_judge/phase5/__init__.py:1`, `exp/verdict_factor_judge/phase5/spec.py:18,536,611,630`, and `exp/verdict_factor_judge/phase5/runner.py:18`. WA §3.2 requires every new file to have a module-level docstring and code comments to be English only.
- [Non-blocking] [Suggestion] Add tests for the plan contracts above: assert the exact G4 subset tags/content, assert G5 p1/p2 weights match the approved fixed phase4 weights, and assert the decision output filenames/content. The current Phase 5 test set passes but covers mostly counts, shapes, and happy-path routing, so it did not catch these mismatches.
- [Non-blocking] [Scope Note] Per owner instruction, pure `155.98.36.13` → `155.98.36.32` IP edits in docs/runners/logs/config are excluded from this G2 review and are not part of this verdict.

### G2 Round 1 — Executor — 2026-05-09 12:05 CDT

- B1 — Accepted — `generate_g5_cells` 不再用 uniform 1/N。新增 `_g5_fixed_weights(recipe_id, declared)`：phase4 p1/p2 → `phase4.spec.generate_r2_weights(rid, alpha_star=1.0, offline_pattern=(1,1,1,1))`（与 stage1+stage5 跑过的 phase4 R2 alpha=1.0 uniform 完全一致；online_total = 1 - alpha = 0 → 2 个 online key 权重 = 0）；phase3 g6 → `{k: 1.0 for k in declared}`（与 phase3 baseline 一致）。新增 INV `test_g5_p1_p2_weights_match_phase4_r2_alpha1_uniform` + `test_g5_g6_weights_match_phase3_baseline` 锁住此契约。
- B2 — Accepted — G4_FACTOR_SUBSETS 的第 12 项从误写的 `off-disp-on-full` 改成 plan §1.5 line 185 写明的 `jerk-full-stack`：`(("jerk",), ("jerk",), ((0,3),(0,5),(3,3)), ((3,3),), "jerk-full-stack")` — offline jerk 走 W-FUT + W-K3 (3 windows) + online jerk W-K3 (1 window) = 4 keys。新增 `test_g4_subset_tags_match_plan_contract`（锁住 12 个 tag 顺序）+ `test_g4_jerk_full_stack_subset_shape`（锁住 4 cell × 3 offline + 1 online = 4 keys）防回归。
- B3 — Accepted — `_dump_decision_gate_table_phase5` 改成写 5 个 per-group 文件（`g1_decision.json` … `g5_decision.json`）而非单一 `decision_gate.json`，与 plan §4.1..§4.5 一致；返回值仍是 aggregate dict 给 callers/tests 用。无对应 group rows 的也写一个空 payload 文件（tests 显式 assert 5 个文件全存在）。新增 `test_decision_gate_writes_5_per_group_files`。
- B4 — Accepted — (a) `analysis/phase5/__init__.py` 加 module docstring `"""Phase 5 analysis — Pareto plot + (FH, WS) heatmaps over 240-cell sweep."""`；(b) `phase5/__init__.py` docstring 翻成英文；(c) `phase5/spec.py` line 18/562/637/656 + `phase5/runner.py` line 18 的中文注释/docstring 翻成英文（"双源"→"dual-source"，"自家"→"own"，"整轮"→"full rotations"，"三个 recipe 各自来源不同"→"each of the three recipes has its own raw source"）。Python `re` 扫描 `[一-鿿]` 在 6 个 phase5 .py 文件中均返回 0 hit。
- S1 — Accepted — 上述 B1/B2/B3 修订各自配套了 contract regression tests（共 4 个新 test）。`test_phase5_spec.py`: `test_g4_subset_tags_match_plan_contract` / `test_g4_jerk_full_stack_subset_shape` / `test_g5_p1_p2_weights_match_phase4_r2_alpha1_uniform` / `test_g5_g6_weights_match_phase3_baseline`；`test_phase5_runner.py`: `test_decision_gate_writes_5_per_group_files`。如果实现退回到 G2 R1 的状态，这些测试会立即失败。
- S2 (Scope Note) — Acknowledged — IP `13` → `32` 的 ack 与回滚均非本轮 G2 修订范围；保留 reviewer 写明的 out-of-scope 标注，不在 Executor 响应中处置。

测试现状：phase5 套 55/55 pass（新增 4 contract tests），phase3/phase4 runner 回归测试也跑通无回归。

### G2 Round 2 — Reviewer — APPROVED — 2026-05-09 12:02 CDT
- [Non-blocking] [Scope Note] Re-review used the working tree delta against the G2 Round 1 staged baseline. Pure IP edits remain excluded per owner instruction.
- [Non-blocking] [Verification] G2 Round 1 blocking items are resolved: G5 p1/p2 weights now delegate to phase4 `generate_r2_weights(..., alpha_star=1.0, offline_pattern=(1,1,1,1))`; G4 now includes the planned `jerk-full-stack` subset; decision output now writes `g1_decision.json` through `g5_decision.json`; new phase5 Python files now have module docstrings and no Chinese comments/docstrings.
- [Non-blocking] [Verification] Independent reviewer run: `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run pytest tests/exp/test_phase5_spec.py tests/exp/test_phase5_runner.py tests/exp/verdict_factor_judge/test_phase5_yaml_emission.py -q` → 55 passed, 1 warning.
- No blocking concerns remain; code approved.
