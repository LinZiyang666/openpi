# X16 — GR00T×LIBERO 门-阈值帕累托（teacher rate 单图）

> **Level**: L2 · **Authority**: Execution · **Stage**: **G1 APPROVED**（2026-08-24 14:27 CDT，R3）→ §4 Code
> **口径依赖**：[`logs/libero_groot_ws_search.log.md`](libero_groot_ws_search.log.md)（权重来源）、
> [`logs/libero_groot_collection.log.md`](libero_groot_collection.log.md)（采集与建库）、
> [`exp/data_authority/analysis/gate_threshold_pareto/analysis.md`](../exp/data_authority/analysis/gate_threshold_pareto/analysis.md)（pi0.5 对照线口径）。
> **章程依赖**：[`docs/experiments/artifact_layout.md`](../docs/experiments/artifact_layout.md) §1（四槽布局）。

---

## 0. 一句话

把 pi0.5 已收官的 `gate_threshold_pareto` 实验原样搬到 GR00T N1.5 上：固定检索栈（用上一轮搜出来的最优权重）、
固定混合门结构，只扫判据阈值 `f_FH` 的 16 档，画 **teacher ratio × 成功率** 的帕累托前沿，外加一个 gate-only 极端点。
**本轮不画 inference ratio 图** —— GR00T 的分段延迟（s1/s2）尚未标定，pi0.5 的仿射常数不可套用（§10.2）。

---

## 1. 背景与动机

上一轮（ws_search）用 `always_hit` + `top_k=1` 隔离出**纯检索质量**，得到两 suite 的最优权重。
那条线的每一步都强制走缓存，因此它回答不了工程上真正的问题：**愿意少调用多少次教师，要付多少成功率**。

pi0.5 侧这个问题已经有答案（教师调用省 63%，spatial 只掉 4.8 个百分点）。
GR00T 侧尚无任何数据点。两者共用同一套门/判据/检索部件，因此这是一次**同实验、换执行体**的对照，
而不是新方法——这也是本轮把 pi0.5 的门超参原样移植（而非重调）的理由（§10.1）。

---

## 2. 实验设计

### 2.1 自变量与固定量

| 项 | 取值 | 来源 |
|---|---|---|
| 自变量 `f_FH` | 16 档 `{0.05, 0.10, …, 0.80}` | `exp/gate_threshold_pareto/solve_gtp.py:FH_GRID` |
| 门 | `score_hysteresis`，`θ_low = θ_high` = 各库 warmup 分数 0.85 分位，`j=3`、`probe_interval=3`、`L=6` | pi0.5 N4 赢点，逐字移植 |
| 判据 | `threshold`，二值（**无 warm tier**） | 两阶段模型无第三阶段可 warm-start |
| 检索栈 | ws_search 赢点权重 + 逐字段 zscore/tanh 归一 + `top_k=1` + `d=1` | §2.2 |
| 库 | 每 suite 一条：`*_sp16_S3.pkl`（5 条成功轨迹/任务） | 与搜索同库，权重才可迁移 |
| 评测集 | A 池 `pruned_init` **全量 500/臂** | owner 裁决（与 ws_search 一致） |
| 客户端 | `--resize-size 256 --replan-steps 5` | 采集/搜索同配方，决策节拍不能变 |

**规模**（scope 已固定，见 §11）：

| 相位 | 臂数 | 集/臂 | 小计 |
|---|---|---|---|
| warmup | 2（每 suite 1） | 100 | 200 |
| eval 主扫描 | 32（2 suite × 16 档） | 500 | 16,000 |
| gate-only 消融 | 2（每 suite 1） | 500 | 1,000 |
| **合计** | 36 | — | **17,200 集** |

### 2.2 模板 yaml（不是新写，是现成赢点）

| suite | 模板 | 权重 (v0/v1/rs) |
|---|---|---|
| libero_spatial | `/data/libero_cache/search/libero_spatial/r2/v0@5_v1@4_rs@3.yaml` | 0.4167 / 0.3333 / 0.2500 |
| libero_10 | `/data/libero_cache/search/libero_10/r2/v0@6_v1@5_rs@1.yaml` | 0.5000 / 0.4167 / 0.0833 |

两份都已带 `key_builder: cp1_groot_libero_spatial_pool_16`、逐字段 zscore 参数、`preload_path` 指向 S3、
`write_policy: never`。照 pi0.5 emitter 的做法，**只动两个字段**：`checkpoints.cp1.gate` 与 `checkpoints.cp1.judge`。
`prompt_emb` 权重为 0 且 `enabled: false`（搜索轴排除，沿用）。模板先从 `/data` 复制进
`exp/libero_groot/config/gate_pareto/<suite>/template.yaml` 并记录 sha256，之后所有生成都以仓内副本为源，
避免 `/data` 侧文件变动使已生成的臂无法追溯。

### 2.3 三个相位

- **warmup**（每 suite 1 臂 × 100 集）：`gate: always_search` + `judge: {type: threshold, threshold: 2.0}`。
  阈值高于 [0,1] 分数域 ⇒ 全 MISS ⇒ 机器人走真实教师轨迹，而 search 照常算分并逐步落盘。
  分数域有界性已亲验：`src/openpi/cache/components/score_normalizers.py:157`「squash to `[0, 1]`… mandatory and bounded」，
  权重和为 1 ⇒ 总分 ∈ [0,1]，故 2.0 是安全哨兵。
- **eval 主扫描**（每 suite 16 臂 × 500 集）：混合门 + 该档解出的 `T_fh`。
- **gate-only 消融**（每 suite 1 臂 × 500 集）：同一混合门 + `judge: {type: threshold, threshold: -1.0}`。
  阈值低于分数域 ⇒ 门放行的检索一律被接受 ⇒ 教师调用只剩**门自己的两个来源**：
  **V2 注入**（连续 `L` 次缓存执行后强制插一次真推理，`gate.py:437`）与 **N1 滞回跳过**
  （连续 `j` 次分数低于 `θ_low` 后停止搜索，期间每 `probe_interval` 拍探一次）。
  测得的 teacher ratio 就是**门的固有干预率**（下界约 `1/(L+1)`，再加滞回贡献）。
  **`L` 取 6，与主扫描一致**（见 §10.5）。

### 2.4 warmup init 池 —— 本轮比 pi0.5 干净的地方

pi0.5 的 warmup 跑在 **A 池每任务前 10 个 init** 上，其 §8 局限 1 自陈这是一次 test-set peeking，
并写明「更干净的替代是用差集池 B 的一个切片，本轮未做是因为会阻塞发射」。

我们没有这个约束，**改用 B 池中未进入 S3 库的 init**：

1. 读 S3 pkl 的全部 entry，取 distinct `trajectory_id`；
2. `episode_identity(traj_id, trials=50)`（`exp/libero_groot/build_size_libraries.py:45`，
   由 `episode_id = task_id * trials + init_idx` 反解）得到入库的 `(task_id, init_idx)` 集合；
3. 每任务在**补集**（约 45 个 init）里按 init_idx 升序取前 10 个 ⇒ 100 集/suite。

结果：标定分布与评测集**零交集**，且与库内轨迹**零自检索**（否则检索会命中自己那条轨迹，θ 被系统性抬高）。
这一步是确定性的，可被复核者用同样三行重算；`warmup_pool_provenance.json` 落盘入库集合与补集。

### 2.5 阈值解法（复用，不重写）

`θ` 与 16 个 `T_fh` 都是同一分布的分位切点，切点约定必须与本项目历史阈值逐字一致，
否则本轮与 pi0.5 线不可比。因此**直接 import 复用** `exp/gate_threshold_pareto/solve_gtp.py`（只读，不改动）：

- `solve_all(per_step_jsonl)` 按 `yaml_id` 分组读 `cp1_score`，对每臂返回 `{theta, cells[16], spread}`；
- 内部走 `exp/verdict_factor_judge/phase3/threshold_solver.py:184`
  `derive_thresholds(scores, fh_ratio, ws_ratio) -> (FH_thr, WS_thr)`（亲验签名）；
- 自带两道前置门：可用分数 `< 500` 或不同取值数 `< 16` 直接 `SystemExit`。

我们的 warmup 逐步行 schema 与它期待的一致（`yaml_id` + `cp1_score`，见 §3 锚点 5），故零适配。
gate-only 臂复用主扫描已解出的同一个 `θ`（读回任一 eval yaml 的 `theta_low`，并断言 `theta_low == theta_high`）。

---

## 3. 可行性亲验（代码锚点）

| # | 结论 | 锚点 | 亲验内容 |
|---|---|---|---|
| 1 | **门对 GR00T 免费可用** | `src/openpi/cache/orchestrator.py:487` `check()`；`:592` gate-skip 返回 | 门在 orchestrator 内部，与执行体无关；skip 分支返回 `CheckResult(MISS, query_keys=…, searched=False)` |
| 2 | GR00T 拦截器不碰门 | `src/openpi/cache/groot/interceptor.py:219` | 只调 `orchestrator.check(CP1, stage1=…)`；MISS ⇒ `run_stage2()` 跑教师，语义正确 |
| 3 | 门可从 yaml 构造 | `src/openpi/cache/config.py:2664` → `:2946` `_build_gate` | `score_hysteresis` ⇒ `ScoreHysteresisGate(theta_low, theta_high, j, probe_interval, L)`；共用 `build_per_connection_components`（`:2608`） |
| 4 | 观测量已对齐 pi0.5 | `src/openpi/cache/groot/interceptor.py:168` `_build_hit_meta` | 字段集与 pi0.5 逐字相同（`hit_type`/`cp1_score`/`searched`/`winner_id`），注释自陈 "so one analysis path reads both" |
| 5 | 客户端能落盘逐步行 | `examples/libero/main.py:114` `--per-step-log-dir` / `:117` `--yaml-id` / `:128` `--phase`；行 618-628 组行 | 行体 = 身份字段 + `**hit_meta` |
| 6 | **gate-skip 行不会被过滤掉** | `examples/libero/main.py:1085-1091` | `filter_searched=_gate_mode`，而 `_gate_mode` 仅由 `--collect-gate-dir` 置真；走 `--per-step-log-dir` 时为 False ⇒ `searched=False` 行保留（teacher ratio 正确性的前提） |
| 7 | 判据类型已在白名单 | `src/openpi/cache/groot/load_guard.py:34` | `_ALLOWED_JUDGE_TYPES = {"threshold", "always_hit"}` ⇒ `threshold` 无需改动 |
| 8 | **唯一硬阻塞** | `src/openpi/cache/groot/load_guard.py:35, 76` | `_REQUIRED_GATE = "always_search"`，装载期硬拒其它门 |
| 9 | **`L` 是 V2 注入上限，不是锁定时长** | `src/openpi/cache/components/gate.py:320-327` 类 docstring；`:437` 判据 | 「once `_fh_run >= L` the gate forces one skip (a fresh inference)」。`L=None` ⇒ 退化为纯 N1。滞回（N1）与注入（V2）是两条独立的教师来源，gate-only 的解读必须区分 |
| 10 | 守卫是**两个执行体共用**的 | `exp/robocasa365/serve_groot_n15.py:225, 337`；`exp/libero_groot/serve_groot_libero.py:153, 307` | 同一函数四处调用；任何默认值放宽都会同时落到 RoboCasa365 线上（§4.1 因此改为 opt-in） |

### 3.1 关于阻塞项的判断

守卫给出的理由是**分析侧**的：「Other gates emit `searched=False` steps, which the downstream analysis would count as real verdicts」。
这描述的是 RoboCasa365 跨场景线的下游假设，不是两阶段拆分的机制限制 —— 锚点 1/2 证明机制上完全成立。

但该假设对 RoboCasa365 **依然有效**（锚点 9）。因此处置**不是**放宽默认白名单，而是
**把例外做成按调用场景显式 opt-in**：守卫默认行为逐字节不变，只有 LIBERO 入口显式声明才放行
`score_hysteresis`，且 opt-in 也只放行这一种门。见 §4.1。

### 3.2 一个必须绕开的坑：逐步文件三重冲突

`PerStepWriterPool`（`src/openpi/serving/per_step_recorder.py:168`）把临时文件命名为
`<out_dir>/_<yaml_id>.worker_<wid>.jsonl`、终稿命名为 `<out_dir>/<yaml_id>.jsonl`，
并在**构造时截断**临时文件。

我们的车队是 **N 个独立 `main.py` 进程、每个 `--num-workers 1`** ⇒ 每个进程都拿 worker 槽 0。
若共用一个 `--per-step-log-dir`，三个层面同时冲突：后启动的进程截断先启动者的临时文件、
N 个进程并发写同一路径、N 份终稿互相覆盖。**症状是行数悄悄变少，不报任何错。**

处置：**每 worker 一个独立目录** `…/per_step/w<i>/`，收尾时在 timan107 上合并 N 份 `<yaml_id>.jsonl`。
合并结果的**完整性由 §6.1 的身份集合等式判定，不靠行数下界**（行数下界证明不了任何东西，见 R2）。零 src 改动。

---

## 4. 文件清单

### 4.1 src/（唯一改动，改为 opt-in）

`src/openpi/cache/groot/load_guard.py`：

```python
#: Gates every GR00T serving entry point may use.
_BASE_ALLOWED_GATES = frozenset({"always_search"})
#: The one documented exception, opted into per entry point (see LIBERO gate-pareto).
_HYSTERESIS_GATE = "score_hysteresis"

def validate_groot_cache_config(
    config: CacheConfig, *, allow_hysteresis_gate: bool = False
) -> None:
```

- 默认 `False` ⇒ 允许集恒为 `{"always_search"}` ⇒ **RoboCasa365 行为逐字节不变**
  （且 `serve_groot_n15.py` 本轮一行不改，它在语言上就不可能传一个新加的 kwarg）。
- `True` 时允许集为 `{"always_search", "score_hysteresis"}` —— **只多这一种**，
  `random`/`periodic`/`client_controlled`/`follow_winner`/`always_skip` 在两种取值下都被拒。
- 用布尔开关而非「调用方传入允许集」：后者能被任意调用方用来夹带任意门，
  守卫就不再拥有"哪些门可能被放行"这一知识。
- 守卫其余四条（CP3 缺席、warm tier 缺席、判据白名单、`write_policy: never`）一字不动。
- 模块 docstring 中「any gate」那段同步订正，写明例外的适用范围与理由。

`exp/libero_groot/serve_groot_libero.py`：两处调用改为
`validate_groot_cache_config(config, allow_hysteresis_gate=True)`，并在附近注释说明
这是 LIBERO 门-帕累托实验的显式例外。

**不改**：`interceptor.py` / `orchestrator.py` / `config.py` / `staged.py` / `serve_groot_n15.py`。

### 4.2 exp/libero_groot/（新增，按四槽布局）

依 [`artifact_layout.md`](../docs/experiments/artifact_layout.md) §1：代码在根、yaml 进 `config/`、
运行产物进 `data/`、**分析工具与图进 `analysis/`**。

| 路径 | 槽 | 作用 |
|---|---|---|
| `gate_pareto_bindings.py` | 根（代码） | 一张表：每 suite 的模板 yaml、S3 库路径、checkpoint、A 池目录、B 池目录、各槽产物根 |
| `emit_gate_yamls.py` | 根（代码） | 由模板生成 warmup(1) + eval(16) + gate_only(1) yaml/suite；每份过 `load_cache_config` 严格自检 + 断言 warm tier 缺席 + 断言 eval/gate_only 臂的 `L` 存在（丢了 `L` 会退化成纯 N1，在结果里与保留 `L` 的臂不可区分） |
| `emit_warmup_pool.py` | 根（代码） | §2.4 的补集推导 + 按 worker 切 episode-filter 分片（schema 同 `make_shards.py`：`{task_id, subset_init_state_idx, orig_init_state_idx}`） |
| `analysis/gate_pareto/analyze_gate_pareto.py` | analysis | 完整性判定（§6.1）+ 聚合 + 出图 + `plot_data.json`。聚合部分纯标准库（在 timan107 就地跑），出图部分单独走 matplotlib |
| `run_gate_pareto.sh` | 根（代码） | 全链驱动 |

**产物路径（钉死）**

| 类别 | 路径 | 说明 |
|---|---|---|
| 生成 yaml | `exp/libero_groot/config/gate_pareto/<suite>/{template.yaml,warmup/,eval/,gate_only/}` | 入 git（`exp/**/config` 未被 gitignore；`gate_threshold_pareto/config` 已入库 78 份，同惯例） |
| 运行产物 | `exp/libero_groot/data/gate_pareto/<suite>/{results,per_step,provenance}/` | `exp/**/data/**` 已 gitignore；仓内 `data` 是指向 `/data/libero_cache` 的软链（owner 的 /data 铁律） |
| 图与台账 | `exp/libero_groot/analysis/gate_pareto/{pareto_<suite>.png,.pdf,plot_data.json,MANIFEST.json,analysis.md}` | 最终报告是纯 `.md`，逐轮记录留 `logs/` |

### 4.3 exp/libero_groot/orchestrate_search.py（改动，opt-in）

复用已经跑过 92 个 cell 的调度器，加四个**默认关闭**的参数，搜索行为逐字节不变：

| 参数 | 默认 | 作用 |
|---|---|---|
| `--per-step-dir` | `""`（关） | 非空时给每 worker 传 `--per-step-log-dir <out>/per_step/w<i> --yaml-id <cell> --phase <phase>`，并在 `collect()` 里合并+回传 |
| `--init-subdir` | `{suite}_apool` | warmup 相位改指 B 池目录 |
| `--shards-dir` | 现值 | warmup 用自己的分片集 |
| `--phase` | `eval` | 写进逐步行，便于三相位共存一份数据 |

改动集中在 `launch_clients()`（`:121`）与 `collect()`（`:147`）两处。

### 4.4 明确不动

- **`exp/gate_threshold_pareto/` 一行不改**：那是已收官（`complete`）的线，其 `analysis.md §6 复现` 是对外承诺。
  本轮只**只读 import** 它的 `solve_gtp`。
- `examples/libero/main.py` 不改（`--per-step-log-dir` 已够用；对该 deprecated 接口的依赖见 §9 R9）。
- `exp/robocasa365/serve_groot_n15.py` 不改（§4.1 的默认值保证它行为不变）。
- `tests/review_tests/` 密封：Execution 全程不读、不列、不检索（`protocols/execution_authority.md` §1）。

---

## 5. 接口

```python
# exp/libero_groot/gate_pareto_bindings.py
@dataclasses.dataclass(frozen=True)
class Binding:
    suite: str; template: str; library: str; checkpoint: str
    apool_dir: str; bpool_dir: str
    config_root: str; data_root: str            # 四槽路径，唯一来源
BINDINGS: tuple[Binding, ...]
def for_suite(suite: str) -> Binding

# exp/libero_groot/emit_gate_yamls.py
FORCE_MISS = 2.0; FORCE_HIT = -1.0
GATE_J = 3; GATE_PROBE_INTERVAL = 3; GATE_L = 6
def build_warmup(b: Binding) -> dict
def build_eval(b: Binding, *, theta: float, t_fh: float) -> dict
def build_gate_only(b: Binding, *, theta: float) -> dict
def emit_warmup(out_root: Path) -> dict[str, str]
def emit_eval(out_root: Path, solved: dict) -> dict[str, str]
def emit_gate_only(out_root: Path) -> dict[str, str]     # theta 从已生成的 eval 臂读回

# exp/libero_groot/emit_warmup_pool.py
def library_inits(pkl_path: str, trials: int) -> dict[int, set[int]]
def warmup_inits(lib: dict[int, set[int]], *, tasks: int, trials: int, per_task: int) -> dict[int, list[int]]
def emit_shards(inits: dict[int, list[int]], lanes: int, out_dir: Path) -> list[Path]

# exp/libero_groot/analysis/gate_pareto/analyze_gate_pareto.py
class IntegrityError(RuntimeError): ...
def check_arm_integrity(results: list[dict], per_step: Iterable[dict], *, expect_ep: int) -> None
def aggregate(results_dir: Path, per_step_dir: Path, *, expect_ep: int) -> dict   # 纯 stdlib
def plot_teacher_pareto(suite: str, arms: dict, gate_only: dict | None, out_dir: Path) -> Path
```

臂命名：warmup `gpw_{sp|l10}`、eval `gp_{sp|l10}_fh{pct:02d}`、gate-only `gpgo_{sp|l10}`
（与 pi0.5 的 `gtp_*` 前缀区分，防两线数据混读）。

---

## 6. 集成点

1. **server**：`serve_groot_libero.py --cache-config <臂 yaml> --concurrent`，一个 cell 一个 server（不走 conductor 的 bundle 热切 —— GR00T server 是 `allow_dynamic_bundles=False`）。
2. **client**：timan107 上 N 个 `examples/libero/main.py`，加 `--per-step-log-dir`/`--yaml-id`/`--phase`。
3. **门**：yaml → `load_cache_config` → `build_per_connection_components` → `_build_gate` → `ScoreHysteresisGate`；运行期由 `orchestrator._feed_verdict_to_gate` 逐步喂裁决。
4. **阈值**：warmup 逐步 jsonl → `solve_gtp.solve_all` → `{theta, cells}` → `emit_gate_yamls.emit_eval` / `emit_gate_only`。
5. **口径**：`teacher_ratio = MISS 决策数 / 总决策数`（决策级，非步级；门每 `replan_steps=5` 个控制步决策一次）；
   `success_rate = 成功集数 / 500`。两者都能用一条 grep 独立复核。

### 6.1 聚合前的完整性判定（fail-closed）

`teacher_ratio` 是本实验的横轴，而它的分母来自逐步证据。逐步数据静默缺失会**在不报错的情况下直接挪动帕累托点**，
因此聚合**必须**先过下列判定，任一不满足即 `IntegrityError` 并拒绝产出该臂的数值（不降级、不估算）：

| 判据 | 内容 |
|---|---|
| I1 结果侧计数 | 结果行数恰为 `expect_ep`（500 / warmup 100） |
| I2 结果侧唯一性 | `(task_id, orig_init_state_idx)` 无重复 |
| I3 身份集合等式 | 逐步侧的 episode 身份集合与结果侧**逐一相等**：无缺失、无额外 |
| I4 每集有裁决 | 每个身份至少一条带 `hit_type` 的行 |
| I5 步唯一性 | `(episode identity, step_idx)` 无重复 |
| I6 逐步文件齐 | 合并前，N 个 worker 目录各自的 `<yaml_id>.jsonl` 均存在（缺文件直接失败，而非让总行数把缺口盖过去） |

I3 是核心：它把"分母的 provenance"建立在集合等式上，而**行数下界做不到这件事**——
某个 worker 的逐步终稿整份缺失时，其他长 episode 的行数完全可能把总数顶过任何下界。

---

## 7. 测试策略

| # | 测试 | 位置 | 断言 |
|---|---|---|---|
| T1 | 守卫 opt-in 矩阵 | `tests/cache/groot/test_groot_load_guard.py` | (a) **默认**（不传 kwarg）拒 `score_hysteresis` —— 这就是 RoboCasa365 走的路径；(b) `allow_hysteresis_gate=True` 通过；(c) `random`/`periodic` 在**两种取值下都拒**（证明 opt-in 没有扩大到那一种之外）；(d) 现有四条守卫在两种取值下均不变；(e) 缺 `theta_low` 的 `score_hysteresis` 由**通用**校验器拒 |
| T2 | emitter 形状 | `tests/libero_groot/test_gate_pareto_emit.py` | warmup 臂 = `always_search` + `threshold 2.0`；eval/gate_only 臂五个门字段齐全且 `L == 6`；gate_only 判据 = `-1.0`；三者 warm tier 均缺席；模板的权重/归一参数**逐字段未被改动**；产物落在 §4.2 钉死的路径 |
| T3 | warmup 池 | `tests/libero_groot/test_warmup_pool.py` | 补集与入库 init 交集为空；每任务恰 `per_task` 个；补集不足时 fail-loud 而非静默取少 |
| T4 | 完整性判定 | `tests/libero_groot/test_gate_pareto_integrity.py` | I1–I6 逐条：缺 worker 文件、缺 episode、多出 episode、重复 step、重复 episode-result、某集零裁决 —— 六种构造各自必须抛 `IntegrityError`；干净输入通过 |
| T5 | 聚合口径 | `tests/libero_groot/test_gate_pareto_analyze.py` | `searched=False` 行**必须**计入 MISS（防回归到"只数判据 MISS"）；空臂返回 `teacher_ratio=None` 而非 0；`f_FH` 从臂名解析正确 |
| T6 | 门状态机在**本轮工作点**上的黄金轨迹 | `tests/cache/components/test_gate.py`（**追加一例，不新建文件**） | 既有覆盖已亲验充分：滞回进入/探测/恢复、band 抗抖、`probe_interval=None`、`None` 分数、`__call__` 纯性（`:345-395`），以及 `L=None/6/3/2` 的 V2 注入（`:540-600`）。**缺口是组合**——带 `L` 的用例都取 `j=1/2`，取 `j=3` 的用例都不带 `L`。因此只补一例：`(j=3, probe_interval=3, L=6)` 的手算黄金轨迹，逐拍断言注入与滞回**互不吞没**。这是"门接通且行为符合定义"的**统计无关**证据 |
| T7 | 冒烟门（实机，`--run-manual`） | `tests/libero_groot/` | **结构性判据**（见 §9 R1）：两臂都产出合法裁决；`searched=false` 行确实出现；不存在恒搜/恒锁；§6.1 完整性判定通过。**不含**任何成功率或 teacher-ratio 的方向性断言 |

**§6 Verify 口径（章程口径，不缩窄）**：按 `protocols/execution_authority.md` §6 与 WA §2.7 跑
**裸 `uv run pytest`** 并留全量输出。配套两条纪律：

1. **先在 HEAD 上取干净 baseline**（进 §4 Code 之前跑一次并存档），Verify 结果与 baseline 逐条比对，
   使"既有失败"与"本轮回归"可分。已知既有失败见 §9 R10。
2. **密封目录的处理**：`testpaths` 覆盖整个 `tests/`，`tests/review_tests/` 会被收集执行。
   Execution 可以**运行**但全程**不读、不列、不检索**其内容（§1）；若该目录内有用例失败，
   记录用例 id 与 pass/fail 后交由 Review Authority 处理，**不得**打开其源码自行修复。
   若 baseline 显示裸跑无法收敛（挂起 / 依赖外部服务），以 baseline 证据向 owner 申请明确的流程 override，
   **不由本计划自行缩小宪法规定的 Verify 范围**。

---

## 8. 运行拓扑与成本

沿用搜索线拓扑：weilandserver（4090 48G）起 6 个 server + 调度器，timan107 出仿真车队，公网段 `ziyanglin.com:23100-23199`。

| 相位 | 规模 | 估时 | 依据 |
|---|---|---|---|
| warmup | 2 × 100 集（全 MISS，纯教师速度） | ~0.5 h | 教师 236 ms/次实测 |
| eval spatial | 16 臂 × 500 集，6 槽并行 ⇒ 3 批 | 1.5–3 h | 搜索实测 27 min/cell，低 `f_FH` 臂教师重载会慢近一倍 |
| eval l10 | 同上 | 3.5–7 h | 搜索实测 62–81 min/cell |
| gate-only | 2 × 500 集（并行占 2 槽） | ~1 h | 高命中 ⇒ 接近纯 cache 速度 |
| **合计** | 17,200 集 | **9–13 h** | 与刚跑完的搜索同量级，一夜可收 |

worker 数沿用实测安全值：spatial 12/槽、**l10 8/槽**（l10 worker 3.1 GB vs spatial 2.6 GB，96 × 3.1 会撑爆 timan107 的 220 GB）。

---

## 9. 风险登记表

| # | 风险 | 症状 | 处置 |
|---|---|---|---|
| R1 | 门在 GR00T 上是**首次真跑** | 门从不触发（等价 always_search）或恒跳过 | **机制层**由既有黄金轨迹测试 + T6 的工作点组合用例锁定；**接通性**由 T7 结构性冒烟门验证（合法裁决 / 有 `searched=false` / 非恒搜非恒锁 / 完整性通过）。小样本的方向性只作诊断记录，**不作硬门**——rollout 随机性与命中导致的轨迹分叉会让低统计功效的方向判据误拒正确实现 |
| R2 | 逐步数据静默缺失 | 行数看着够，实则某 worker 终稿整份丢失 | §6.1 的 I1–I6 fail-closed 判定（身份集合等式，不是行数下界）；T4 用六种构造逐条锁定 |
| R3 | warmup 分布过窄，16 档切不开 | 出现重复工作点，看起来像独立证据 | `solve_gtp.solve_arm` 前置门（`< 500` 样本或 `< 16` distinct ⇒ 直接失败） |
| R4 | 教师重载臂比搜索 cell 慢近一倍 | 撞 `--cell-timeout 7200` 被判 partial | l10 相位把超时提到 14400 |
| R5 | broker `too_many_in_flight`（踩过） | cell 算完但结果没传回 | 沿用全局 pull 锁 + 6 次退避；逐步数据**先在远端做完整性判定与聚合**只回传小 JSON，原始 jsonl 压缩后另拉 |
| R6 | 守卫放宽波及 RoboCasa365 线 | 该线误配置不再被拦 | §4.1 改为 opt-in：默认允许集不变、`serve_groot_n15.py` 一行不改；T1(a)(c)(d) 把"默认仍拒"钉成回归测试 |
| R7 | X15 与本线**共享 checkout** | `config.py` 的 key builder 注册被覆盖过一次 | stage 只按显式路径清单，commit 前回读确认注册仍在 |
| R8 | A 池既是测试集又是上一轮选权重的地方 | 绝对数字带选择偏差 | §10.3 报告口径声明；本轮 warmup 已彻底移出 A 池（§2.4） |
| R9 | 依赖 deprecated 的 `--per-step-log-dir` | 该 flag 被移除后本链路静默失去逐步证据 | 见 §10.6：记录依赖理由与"移除前置条件"，并在本计划中把它列为该 flag 的**现役消费者** |
| R10 | 既有失败混入 Verify 判读 | 把非本轮问题当回归、或反之 | §7 的 HEAD baseline 纪律；已知既有失败：`tests/examples/test_libero_main.py` 源码锁 2≠3、一例 GCS 网络依赖 |
| R11 | **密封边界已发生一次实际破坏（审计证据连续性）** | G2 若把已暴露标识对应的 probe 当作独立秘密证据，审计不对称性是虚的 | Plan 阶段执行方曾运行 `grep -rn "def test.*hysteresis" tests/`，搜索范围含 `tests/review_tests/`，输出暴露了该目录下**一个文件名与一个函数名**；未打开任何该目录下的文件、未使用其内容。**处置**：后续 G2 **不**把已暴露标识对应的 probe 作为独立秘密证据，由 Review Authority 替换或明确排除（处置权属审查方，Execution 不介入）。执行方侧的约束已写入 §4.4，且后续对 `tests/` 的一切检索一律加 `--exclude-dir=review_tests`。本条按 G1 R3 的要求从 Review Log 转记至此，使 G2 的 Review Authority 在 G1 记录被 §3.1 删除后仍可知悉 |

---

## 10. 局限与报告口径（必须写进最终报告）

1. **门超参是 pi0.5 调出来的工作点**：`j=3`/`probe_interval=3`/`L=6` 移植自 pi0.5 的 N4 赢点，GR00T 上未调。
   θ 是每库从自己 warmup 重解的（这条没问题）。结论只能说"pi0.5 工作点移植到 GR00T 的表现"，
   **不能**说成"GR00T 的最优门"。
2. **不出 inference ratio 图**：`analyze_gtp.inference_ratio()` 里的 `0.15195 + 0.84805·tr` 是
   **pi0.5 CUDA-Graph 档**的三段延迟常数，与 GR00T 无关。要补这张图需要先做 GR00T 的分段延迟标定
   （s1 = stage1、s2 = stage2），列为后续项。
3. **权重是在 A 池上选的**（继承自 ws_search）：绝对数字带选择偏差；跨臂比较（不同 `f_FH`）不受影响，
   因为所有臂共用同一套权重、同一评测集、同样的 (task, init) 配对。
4. **单库**：每 suite 只跑 S3（5 条/任务）。pi0.5 那条线有 ws/cs 两库可比，本轮没有——
   库规模轴是独立的开放问题（S1/S2/S4/S5 已建好未评测）。
5. **gate-only 的 `L` 与 pi0.5 不同**：本轮取 6（同主扫描），pi0.5 按 owner 指定取 8。
   `L` 决定 V2 注入的节拍（连续 `L` 次缓存执行强制插一次真推理），因此它直接设定了
   teacher ratio 的下界（约 `1/(L+1)`）。取 6 使星标点与主扫描共用同一个门，
   是**自家前沿的真正左端极限**；pi0.5 的星标点用的是另一个门（下界更低）。
   两线的 gate-only 数值**不可直接对比**，只能各自与自家前沿比较。
6. **A 池上没有纯教师基线（owner 2026-08-24 裁定不补）** —— 评测集是 A 池，但本线从未在 A 池上
   跑过 500 集纯教师。可用的分母只有两个，都不够格当主张的依据：
   官方登记 **46/50 = 0.92（A 池，n=50，±4 pp）**，以及我们采集期的 **0.912（B 池，n=500）**——
   而 B 池是未筛选的剩余、比 A 池难（见 collection log §2），两池不可直接互换。
   **后果**：凡"纯 cache 达教师的 X%"、"与教师持平"一类说法，分母都只能引 n=50 的官方值，
   其自身误差（±4 pp）大于我们要判定的效应（约 1–3 pp）。因此这类句子在报告中
   **必须显式标注分母来源与样本量**，不得写成无条件断言。
   ⚠ 另注意：扫描内最靠教师的臂 fh05（85.5% 教师调用）在 A 池上是 **0.940**，
   **高于**上述任一教师参照 —— 所以**不能拿 fh05 当"教师上限"**来计算退让，
   那会系统性夸大每个臂的代价。跨臂比较（臂与臂之间）不受影响，配对检验照常有效。
   补法（若日后要补）：warmup 的同一份 yaml，init 池换 A 池、集数 500，约 80 分钟/suite。

7. **本链路是 `--per-step-log-dir` 的现役消费者**：canonical 的 `--collect-gate-dir` 走 gate 模式，
   会以 `filter_searched=True` 在合并时丢弃 `searched=false` 行（`main.py:1085-1091`），
   而那正是本实验 teacher ratio 分子的一部分。因此在出现一个"保留 gate-skip 裁决"的受支持
   recorder 模式之前，该 flag **不可移除**；移除前置条件即"canonical 模式提供不过滤开关"。

---

## 11. Scope 裁决（已固定，owner 可推翻）

计划必须是单一可执行规格，因此下列三项由执行方按 §10.2 的实质判断固定；理由随附，owner 可随时推翻并重估预算。

| # | 问题 | **裁决** | 理由 |
|---|---|---|---|
| D1 | 是否加第二条库臂（如 S6/full）？ | **不加** | 成本翻倍，且把"库多大够用"与"阈值怎么选"两个独立问题耦合进同一张图。库规模曲线单独做更清楚 |
| D2 | 是否补 gate-only 消融？ | **加**（已并入 §2.3 / §2.1 规模表 / §5 接口 / §7 T2 / §8 成本） | 2 × 500 集约 1 h，而 pi0.5 那条线三个结论有两个来自它（门的固有干预率不是 0；verdict 的边际价值可读）。`L` 取 6 而非 pi0.5 的 8，理由见 §10.5 |
| D3 | warmup 池用 B 池补集还是照抄 A 池前 10 init？ | **B 池补集** | 同样成本换掉一次 test-set peeking；pi0.5 自陈这是其局限 1，且我们的库 provenance 可用 `episode_identity` 精确反解 |

### 11.1 一项与本计划相邻、需 owner 拍板的既有偏差

`exp/libero_groot/` 现有的 `analyze_search.py` 与 `derive_fine_region.py` 位于包根，
按 `artifact_layout.md` §1 它们属 `analysis/`。二者**尚未 commit**，此刻迁移零代价；
一旦随本轮 commit 落库，就变成需要专门一次迁移才能修的章程偏差。
本计划**不擅自迁移**（属另一条线的产物，且 WA §3.1 禁止顺手清理），建议在同一 commit 内一并归位。

---

## 12. 文档同步清单（WA §4 红线）

索引已存在但内容过期**不构成同步**。本计划每轮改稿完成时、以及 §7 Commit 前，逐项复查：

| 文档 | 需同步的内容 | 时机 |
|---|---|---|
| `logs/README.md` | 本计划条目的**状态**与**摘要**。摘要必须反映当轮的：总集数、src 改动形态、产物落位、口径要点。⚠ 已发生过的失败模式：**状态字段更新了、摘要却仍停在被否决的旧方案** | 每轮改稿完成时 + Commit 前 |
| `logs/libero_groot_gate_pareto_plan.log.md` | 本文件；Review Log append-only，正文随裁决更新 | 每轮 |
| `exp/libero_groot/analysis/gate_pareto/analysis.md` | 最终报告（纯 `.md`），含 §10 的四条局限口径 | 实验收官时 |

**明确不在范围内**（列出以免下一轮再被问）：

- `docs/architecture/cache_system.md` §5.17 —— 亲验该节**未声明门的限制**，原文写的是"真正被复用的是模型无关的那半边：
  Orchestrator / CacheStorage / judge / gate / search strategy"，故 §4.1 的 opt-in 改动与之不矛盾，无过期内容。
  且 WA §2.1 只对 **L3** 要求架构文档更新，本轮是 L2。若 owner 认为该守卫的例外机制值得进架构文档，
  按独立 L0 文档改动处理，不夹带进本轮。
- `docs/cache/tutorial.md` §18 —— 只讲 `cp1_groot_*` KeyBuilder，本轮不改 KeyBuilder。
- `docs/README.md` —— 本轮未在 `docs/` 下新增、修改或移动任何文件。

---

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-08-24 17:26 CDT

- [Blocking] [Concern] §6.1 的 fail-closed 完整性门仍可被缺失证据绕过：`aggregate()` 在 `<arm>.merge.json` 不存在时把 `merge=None` 传入并完全跳过 I6，而且结果目录为空（乃至某个完整臂从未产生结果文件）时会成功返回空/不完整摘要。— reasoning: I6 的唯一证据就是 sidecar 中的 `lanes_expected/lanes_found`，sidecar 缺失时不能证明所有 worker 文件存在；同时按结果目录现有文件枚举无法发现整臂缺席。独立 G2 probes 对“sidecar 缺失”和“空结果目录”两种构造均预期 `IntegrityError`，当前均未抛出（2 failed）。请让正式聚合强制要求 sidecar，并由聚合接口或 driver 对本相位的精确预期 arm 集合/数量做等式校验，不能只校验已经出现的臂。
- [Blocking] [Concern] `run_gate_pareto.sh`/`orchestrate_search.py` 会把失败的相位报告成成功并继续推进：shell 仅 `set -u`，tmux 中的 `python | tee` 没有 `pipefail`，scheduler 的 cell/server/pull 失败只记日志且 `main()` 最终仍正常返回，driver 也只打印 eval 完成数而不断言，末尾无条件打印 `GATE-PARETO-DONE`。— reasoning: warmup、solve、eval、gate-only 或聚合任一失败都可能被后续步骤覆盖/跳过，直接违背本计划的 unattended、resumable 与 fail-closed 目标。请传播非零退出状态，并在每个 phase 边界检查精确结果臂数、每臂 episode 数及 per-step/merge 证据齐全后才进入下一相位；最终 DONE 只能在全部约束满足后打印。
- [Blocking] [Concern] 批准的 T7 实机冒烟门没有真正接入全链，且测试本身允许 false-green：driver 注释写有 `smoke`，实际命令却从 emit 直接进入 16 臂 sweep；`_arms()` 在环境已提供但少于两臂时调用 `pytest.skip`，而“非恒搜”只要求所有臂合计至少出现一次 `searched=false`，没有逐臂拒绝恒搜。— reasoning: §7 T7 与 §9 R1 明确要求在投入 17,200 集前用两臂验证合法裁决、gate skip、非恒搜/非恒锁及 I1–I6；当前零臂/一臂可得到 skipped 而非 failed，一条恒搜臂也可被另一条正常臂掩盖。请在 driver 中先运行明确的两臂小样本并以 `--run-manual` 执行 T7，数据不齐必须失败，且逐臂断言既有 searched 决策也有 gate-skip 决策，通过后才能启动完整 sweep。
- [Blocking] [Concern] “全链驱动”和批准的分析产物清单尚未闭合：driver 只写每 suite 的 `summary.json`，从未调用 analyzer 的 `plot` 子命令，也没有生成承诺的 `MANIFEST.json`；因此 `pareto_<suite>.png/.pdf`、`plot_data.json` 和 manifest 不会由该全链产生。— reasoning: §4.2 把这些文件钉死为交付物并称 `run_gate_pareto.sh` 为全链驱动；仅实现一个永不调用的 plot 子命令不能形成可复现交付链。请在两 suite 聚合都成功后调用 plot，并按计划生成可复核的 manifest（`analysis.md` 可在真实实验收官时填写，但 driver/工具必须提供其余机器生成物的确定性路径与来源/摘要）。

### G2 Round 1 — Executor — 2026-08-24 17:44 CDT

- **Accepted**（完整性门可被绕过）— 两个漏洞都成立，且都是"门在最需要它的场合最弱"的形态。
  (1) `aggregate()` 在 sidecar 缺失时传 `merge=None` 而 `check_arm_integrity` 的 `merge` 有默认值，
  于是 I6 被整段跳过 —— 一个**可选的 fail-closed 检查不是 fail-closed**。
  (2) 按结果目录枚举无法发现整臂缺席：摘要只是少一个点，而穿过其余点的前沿看起来毫无问题。
  修法：`check_arm_integrity` 的 `merge` **改为无默认值的必需关键字**（新增用例断言省略它抛 `TypeError`）；
  `aggregate()` 新增**必需**的 `expect_arms`，先做 `missing`/`extra` 双向集合等式再逐臂检查，
  sidecar 缺失单独抛 `merge sidecar missing`，空预期集抛 `refusing to aggregate nothing`。
  预期臂集来自**被检查目录之外**：新增 `arms_from_config(yaml_dir)` 从已发射的 recipe 读取
  （发射出去的 yaml 才是"本该跑什么"的权威陈述），CLI 强制二选一 `--arms-from` / `--expect-arms`。
  ⚠ **接口变更提示**：`aggregate()` 与 `check_arm_integrity()` 的签名都变了（各多一个必需关键字）。
  按旧签名调用的独立 probe 现在会抛 `TypeError` 而非 `IntegrityError` —— 这正是本条要求的修法所致，
  烦请同步更新 probe 的调用方式；我不读 `tests/review_tests/`，无法自行适配。

- **Accepted**（失败被报成成功）— 成立。调度器**故意**吞掉单 cell 失败（一个坏 cell 不能拖死另外五个槽），
  这意味着"队列排空"根本不是相位成功的证据，而 `main()` 无条件返回 0 把这个缺口一路放行到发布的前沿上。
  三处修复：
  (a) `orchestrate_search.main()` 收尾按产物核对 —— 完成臂集是否覆盖全部 cell、partial 清单、
  开启逐步捕获时每个完成臂的 `.jsonl` 与 `.merge.json` 是否齐全；不满足则打 `PHASE-FAILED` 并 `SystemExit(1)`
  （原先那行 `results={len(glob('*.json'))}` 还把 partial 算进了完成数，一并改掉）。
  (b) driver 改 `set -euo pipefail`；调度器不再内联进 tmux，而是写成脚本文件由 tmux 执行 ——
  `pipefail` 必须设在跑 `python | tee` 的那个 shell 里，否则 tee 的 0 会盖住调度器的退出码 ——
  脚本把退出码写进状态文件，driver 读它，缺文件或非 0 一律 `die`。
  (c) 每个相位末尾加 `verify_phase`：用同一个完整性门按精确臂集与每臂 episode 数复核，通过才进下一相位。
  `GATE-PARETO-DONE` 现在只在两 suite 全绿且四份交付物都非空时才打印。

- **Accepted（结构）/ 部分改形（判据）**（T7 未接入 + false-green）— 接入与 false-green 两条完全成立：
  driver 从 emit 直接跳进 16 臂 sweep，且 `_arms()` 在环境已给但数据不全时 `skip`。
  已修：driver 在 sweep **之前**跑一个两臂小样本相位（`gpgo_<tag>` + `gp_<tag>_fh80`，
  默认每任务 2 集 = 20 集/臂），过 `verify_phase` 后以 `--run-manual` 跑 T7，任一失败即 `die`，
  通过才提交完整 sweep；`_arms()` 改为**环境完全未设才 skip，半配置/缺文件/少于两臂一律 `pytest.fail`**。
  **判据形状按机制改了**，这一点请复核：`L` 是 V2 注入上限，注入需要连续 `L` 次 **FULL_HIT** 裁决，
  因此 gate-skip 是否出现**取决于判据阈值** —— 低 `f_FH` 臂的裁决多为 MISS，`_fh_run` 不断清零，
  V2 永不触发；N1 侧也只在连续 `j` 次低于 θ 时才跳过。所以"每臂都必须有 gate-skip"会把一个
  **合法**的臂判成坏的。改为：逐臂断言 ≥1 决策、≥1 searched 决策（非恒锁）、裁决合法、I1–I6 通过；
  gate-skip 的存在**在 gate-only 臂上逐臂强制** —— 那条臂判据全接受，每次搜索都是 FULL_HIT，
  注入在 `L+1` 次决策内**机制上必然**发生，是唯一能把"门接通"钉死的臂。
  并把 gate-only **设为冒烟集的强制成员**（缺它直接 `pytest.fail`），否则门可以完全惰性而所有断言仍绿。

- **Accepted**（交付物未闭环）— 成立：plot 子命令从未被调用，MANIFEST.json 根本不会产生。
  已修：driver 在两 suite 聚合都成功后调用 `plot`，随后逐个断言
  `pareto_<suite>.{png,pdf}`、`plot_data.json`、`MANIFEST.json` 存在且非空，缺任一即 `die`。
  新增 `write_manifest(out_dir, sources)`：逐文件 sha256 + 字节数 + 每个 suite 的来源 json 路径，
  **manifest 不摘要自身**（否则重写会改变自己的哈希，记录永不可复现）。
  已用合成 17 臂数据实跑一遍出图链，产出 png/pdf/plot_data.json/MANIFEST.json 四件、manifest 记 3 文件；
  顺带修掉图上一处缺陷：前沿可能合法地塌成单点，原先只画线会什么都看不见而图例仍声称有线，已加 marker。
  `analysis.md` 仍留到真实实验收官时写（计划 §12 已如此登记）。

**回归**：`tests/cache tests/libero_groot tests/robocasa365` = **1637 passed / 26 skipped / 0 failed**
（本轮新增 9 个用例：sidecar 必需、空目录、整臂缺席、臂集越界、recipe 集来源、空 recipe 目录、manifest 摘要与自排除）。

### G2 Round 2 — Reviewer — NEEDS REVISION — 2026-08-24 18:11 CDT

- [Blocking] [Concern] 四个 R1 blocking 的逻辑修复均已闭合，但新全链仍绑定到本机不存在的旧 checkout 路径：`run_gate_pareto.sh` 设 `REPO=/home/weiland/openpi` 并由此取 `$REPO/.venv/bin/python`，`orchestrate_search.py` 的 `WEILAND_REPO` 与 `GR00T_PATH` 也使用 `/home/weiland/openpi`；本次权威工作区和实际主机 `Weiland` 上的 checkout 是 `/home/weiland/projects/openpi`，亲验旧目录不存在、旧路径 Python 不可执行、当前工作区 Python 可执行。— reasoning: driver 会在第一条 `$PY` 命令处直接失败，即使绕过该处，scheduler 启 server 时也会 `cd` 到不存在目录并等待超时；因此 17,200 集全链目前不可执行。请不要再复制一个机器绝对 checkout 常量：shell driver 从自身位置解析 repo root（例如基于 `BASH_SOURCE`），Python scheduler 从 `__file__` 解析本地 repo root，并让 GR00T `PYTHONPATH` 复用该解析值；补一条无需 GPU/远端的测试，证明入口解析出的 repo、venv Python 与 server 脚本在当前 checkout 下存在。其余 R1 项已通过复核：开发者回归 `1637 passed / 26 skipped`，密封 fail-closed/scheduler probes `3 passed`，shell 语法检查通过。

### G2 Round 2 — Executor — 2026-08-24 18:26 CDT

- **Accepted（关切成立）/ 更正一处事实前提**（硬编码 checkout 路径）—

  **先更正前提**：`/home/weiland/openpi` 在**部署目标机 weilandserver 上确实存在**，
  且其 `.venv/bin/python` 可执行、`exp/libero_groot/` 已同步（本轮亲验，非转述）；
  该机上反而**没有** `/home/weiland/projects/openpi`。上一轮 92 个 search cell 就是从这条路径跑完的。
  所以"全链启动即失败"这一判断对**本机**成立、对**脚本声明的运行主机**不成立 ——
  两台机器都持有 checkout，只是绝对路径不同（本机 `~/projects/openpi`，服务机 `~/openpi`）。

  **但关切本身完全成立，且比原始判断更普遍**：硬编码的绝对路径把整条链**静默地钉死在一台机器上**，
  代码里没有任何一处声明这一点、也没有任何检查。后果不是导入错误那种一眼可见的失败，而是
  server 永远不 bind、slot 白等满 150 秒启动预算、**每个 cell 重复一次**——
  一个相位可以在没人读日志之前就把 wall clock 烧光。开发者在本机敲一下 driver 更是直接踩中。
  reviewer 给的修法（从入口自身解析）严格更好且零成本，故照办。

  **已改**：
  - `orchestrate_search.py`：`REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]`，
    `WEILAND_REPO` 由它派生（不再是第二处字面量 —— 两处字面量正是它们漂移的方式）；
    `GR00T_PATH` 复用该解析值拼接。
  - GR00T 岛（`gr00t_n15` 源码与 island venv）与本 checkout **无关**，仍是绝对路径，
    但改为**可覆盖**（`GROOT_N15_HOME` / `GROOT_N15_PYTHON`）—— 它是这里唯一一个
    换台机器会合理地放在别处的东西。
  - 新增 `preflight()`：调度器在**派发第一集之前**核对 server 脚本、island 解释器、island 源码，
    缺任一即退出，错误消息同时给出**主机名**与 `__file__` 解析出的 repo root
    （只报"文件缺失"会让操作员去找文件，而不是去看自己站在哪台机器上）。
  - `run_gate_pareto.sh`：`REPO` 由 `BASH_SOURCE` 解析；开头打印 `repo=… host=…` 并逐项核对
    venv 解释器、调度器、分析器、`tether` CLI。
  - 新增 `GP_PREFLIGHT_ONLY=1` 模式：只做解析与主机核对后退出。
    这不是测试专用钩子 —— "这台机器到底有没有这条链需要的东西"本就该在通宵跑之前问，而不是之后。

  **新增测试** `tests/libero_groot/test_gate_pareto_paths.py`（8 例，无 GPU、无远端）：
  调度器解析结果 == 本 checkout、`WEILAND_REPO` 为派生而非字面量、`GR00T_PATH` 含解析根与 `/src`、
  server 脚本存在、venv 解释器存在且可执行、分析器存在、preflight 失败消息同时含主机与解析根、
  以及 **driver 与调度器解析出同一个根** —— 这一条通过 `GP_PREFLIGHT_ONLY` 跑**真脚本**，
  而不是在测试里复述它的解析表达式（自带一份逻辑副本的测试证明不了被测脚本的任何事情）。

  **⚠ 随之而来的部署事项**（请一并复核）：路径改为动态解析后，"driver 在哪运行就驱动哪个 checkout"
  成为显式语义。因此发车前必须把本轮改动同步到 weilandserver 的 `/home/weiland/openpi`
  （该机已有本包的搜索线版本，但没有本轮的五个新模块）。preflight 只能证明**路径**齐全，
  证明不了**版本**新旧 —— 这一步仍是操作纪律，不是代码能兜住的。

**回归**：`tests/cache tests/libero_groot tests/robocasa365` = **1645 passed / 26 skipped / 0 failed**。

### G2 Round 3 — Reviewer — APPROVED — 2026-08-24 18:18 CDT

- [Resolved] R2 的 checkout 路径阻塞已关闭：shell driver 由 `BASH_SOURCE`、Python scheduler 由 `__file__` 解析同一实际 checkout，`GR00T_PATH` 复用该根；旧 `/home/weiland/openpi` 字面量已不再承担本地 repo 定位，GR00T island 路径保留为独立且可覆盖的部署配置。
- [Resolved] 启动前置检查已覆盖 driver 所需 venv/scheduler/analyzer/tether 与 scheduler 所需 server/island，`GP_PREFLIGHT_ONLY=1` 通过真实脚本验证 driver 与 scheduler 根目录一致；新增 8 条路径测试均通过。
- [Evidence] 本轮复跑 `tests/cache tests/libero_groot tests/robocasa365`：**1645 passed / 26 skipped / 0 failed**；路径专项与密封独立 probes：**11 passed**；`bash -n` 与 diff whitespace 检查通过。R1 的 I1–I6/预期臂集、非零状态传播、T7 前置冒烟、plot + manifest 四项修复均保持闭合。
- [Decision] G2 代码批准；可进入 §6 Verify。部署前仍须按执行方记录把已批准快照同步到实际发车 checkout，此为正常发布步骤，不构成代码阻塞。

### Post-G2 Hotfixes — Executor — 2026-08-24 20:00 CDT

两处首点火才暴露的缺陷，均在 G2 APPROVED 之后修改，按 WA §2.6「emergency hotfix 仍需事后 G2 复审」记录在此。
两者都不改变实验设计、口径或产物，只让已批准的设计真的跑得起来。

- **仿真 venv 导不进 `openpi`**（`orchestrate_search.py:launch_clients`）——
  客户端命令原本用 `PYTHONPATH=.`，而 `openpi` 在 `src/` 下。搜索线从未踩到，因为
  `--per-step-log-dir` 是客户端**唯一**会 import `openpi` 的地方
  （`openpi.serving.per_step_recorder`，在 `eval_libero` 里惰性导入）。首次开启逐步捕获后，
  12 个 worker 在 server 起来之后立刻全部 `ModuleNotFoundError` 退出，
  症状呈现为「车队产出 0 行」而不是导入错误。
  修法：仅在启用逐步捕获时用 `PYTHONPATH=.:src`（该 recorder 只依赖 stdlib，
  且 `openpi/__init__.py` 与 `openpi/serving/__init__.py` 均为空命名空间，
  py3.8 的 LIBERO venv 可安全导入）。搜索线路径保持 `.` 不变。
  **计划 §3 锚点 5 的漏洞**：当时只亲验了「main.py 有这个 flag 且会写行」，
  没有亲验「仿真 venv 能否导入它所需要的模块」——契约验到了，运行环境没验。

- **tmux 前缀匹配杀掉了链自己**（`run_gate_pareto.sh`，同类隐患一并加固 `orchestrate_search.py`）——
  `tmux kill-session -t libgp` 在没有精确同名会话时**回落到前缀匹配**，
  于是匹配上链自身的会话 `libgpchain` 并把它杀掉，**退出码 0、无任何输出**。
  表现为：链在打印完某个相位横幅后凭空消失，日志停在那一行，没有 scheduler、没有 status 文件，
  三次复现都被误读成「run_stage 内部失败」。实证：`tmux new -s probechain` 后
  `tmux kill-session -t probe` 返回 0 且 `probechain` 消失。
  修法：全部 `-t` 目标改用 tmux 的精确匹配语法 `'=name'`（`libgp` / `libsrv<port>` / `lw<port>_<n>`）。
  这是一整类 bug：`-t libsrv2316` 同样会误杀 `libsrv23160`。
