# Weighted-Sum Trajectory 每步权重筛选搜索 — libero_10 复制

> **Status**: `In Progress`（G1 APPROVED R3 2026-07-02 / §4 Code 完成 / 待 G2）
> **Level**: L2
> **Authority**: Execution
> **Date**: 2026-07-02
> **关联**: libero_spatial 版 [`logs/weighted_sum_trajectory_weight_alloc.log.md`](weighted_sum_trajectory_weight_alloc.log.md)（G1/G2 APPROVED，code de7c499，主跑完成：171 config × 100 ep，结论「无权重形状显著超 incumbent，d3/d4/d5 均 < d1」）。本计划把**同一实验设计**搬到 **libero_10**，验证该结论是否 dataset-general。设计（搜索矩阵 / 配对分析 / screening 框架）完全复用已批准逻辑；改动只在 **dataset-specific 常数（tracked base manifest + suite 路径）+ emitter/analyze 的 per-suite dispatch**。

---

## 1. 动机与判据

libero_spatial 的筛选结论：171 个 `trajectory_weights` 形状里无一显著超 incumbent（固定递减），且最优 trajectory（0.72-0.73）均 < d1 prior（0.74）→ 合作者「固定递减权重不合理致 d1>d3/d4/d5」的假说**不被支持**。本计划在 **libero_10**（长程 10 任务，更难，d1 天花板 ~0.52）上复制同一筛选，判据同 libero_spatial §1.3：**screening（候选 + 配对不确定度），非裁决**。

**唯一变量**：每 depth 复用 **libero_10 自己的 base**（modality winner + 该 keybuilder + zscore + always_hit + write_policy=never），只搜 `trajectory_weights`。

## 2. 搜索矩阵（完全复用，dataset 无关）

- S1（当前步主导 × 尾形梯度，C_GRID 8 档 × Q_GRID 3 态）、S2（形状格点）、S3（incumbent 锚点）**逐字复用** libero_spatial 的 `build_depth_configs`。
- 权重形状是纯组合，**与 dataset 无关** → 计数不变：**d3=52 / d4=60 / d5=59 / 总 171**（`EXPECTED_UNION={3:52,4:60,5:59}` 复用）。× 100 ep = **17,100 ep**。
- **incumbent `trajectory_weights` 与 libero_spatial 相同**（`emit_trajectory_yamls.DEPTH_WEIGHTS` 亲验为 task-independent 硬编码，`:53-58` 无 suite 分支）：d3 `[0.5,0.3,0.2]`、d4 `[0.4,0.3,0.2,0.1]`、d5 `[0.35,0.25,0.2,0.12,0.08]`。
- 唯一 dataset 差异是**每 depth 的 base modality 权重**（见 §2.1）。

### 2.1 libero_10 per-depth base 真实权重（tracked 非有损 manifest，G1-1 / G1-2R 修正）

**三层来源全有损，唯一权威是实际 base YAML**（G1-2R 核实）：
- (a) CID `@NN` 由 `int(w*100)` 截断——有损；
- (b) `all_results.csv` 的 `v0/v1/rs` **也是**从 yaml_id 正则取整数 ÷100 写出（`build_all_results.py:25-46,63-67`；非读原始 YAML）——与 CID **同样有损**（我 G1-1R 误当它非有损，错了）；
- (c) `grid_/grid3_weight_configs[cid]` 用干净 1/8、1/16 格点重建——对 d3 也错。

三者对 3 个 base **无一全对**（本会话亲验）：

| depth | winner cid | **真实权重 v0/v1/rs（实际 base YAML，非有损）** | CID/csv ÷100 | grid 反推 |
|---|---|---|---|---|
| d3 | `grid_vision_0@62_vision_1@37` | **0.62 / 0.37 / 0.00** | 0.62/0.37 ✓ | 0.625/0.375 ✗ |
| d4 | `grid3_vision_0@25_vision_1@43_robot_state@31` | **0.25 / 0.4375 / 0.3125** | 0.25/0.43/0.31 ✗ | 0.25/0.4375/0.3125 ✓ |
| d5 | `grid_vision_0@50_vision_1@50` | **0.50 / 0.50 / 0.00** | 0.5/0.5 ✓ | 0.5/0.5 ✓ |

**真实值来源（本会话亲验，3 处实际 base YAML 一致）**：d3/d5 = `config/trajectory/libero_10/...__d{3,5}.yaml`；d4 = `config/stage2/libero_10/...__d4.yaml`（与 `trajectory_wsweep/`、`threshold_pareto/.../anchor` 三处逐字一致）。这些 config YAML 虽 gitignored 但实物在本地。
**决定性验证**：用上表真实权重 + libero_10 calibration + 该 depth incumbent `DEPTH_WEIGHTS` 经 `build_eval_config` 重建，与实际 base YAML **deep-diff = 0**（d3/d4/d5 全 0 差异）——真实权重是唯一能精确复现 base 的来源。
**方案（manifest）**：固化一份 **tracked、非有损、per-depth manifest** `LIBERO10_BASE_MANIFEST`（depth → {cid, 完整浮点 weights, 源 base YAML provenance}），emitter 从 manifest 重建；**禁止**从 CID / all_results.csv / grid 反推 libero_10 权重。3 份实际 base YAML 复制为 **tracked 测试 fixture** 供 deep-diff。（真实权重 d3/d5 和=0.99≠1.0；always_hit 排序不受绝对尺度/归一化影响，见 libero_spatial plan §1.2；`validate_cache_config` 只要非负且和>0，rs 被 `_REQUIRED_KEYBUILDER_FIELDS=(vision_0,robot_state)`（`emit_yamls.py:31,75`）强制为 key 字段、权重可 0——已亲验 `load_cache_config` 通过。）

## 3. 代码改动（最小化，向后兼容 libero_spatial）

| 路径 | 改动 |
|---|---|
| `exp/weighted_sum/emit_traj_weight_alloc.py` | **改**：① `WINNER_CID`（单 dict）→ `WINNER_CID_BY_SUITE`（per-depth cid）；② **新增 tracked 非有损 manifest `LIBERO10_BASE_MANIFEST`**（depth→{cid, 完整浮点 weights `{vision_0,vision_1,robot_state}`, 源 base YAML provenance}），值 = 实际 base YAML 真实权重（d3 0.62/0.37/0、d4 0.25/0.4375/0.3125、d5 0.5/0.5/0）；③ `_modality_weights(suite, winner_cid)` dispatch：**libero_10 → 读 manifest**（**绝不** CID/all_results.csv/grid 反推）；libero_spatial → `grid3_weight_configs[cid]`（**逐字不变**）；④ 加 `--task-suite {libero_spatial,libero_10}`；⑤ **CLI `--calibration`/`--artifact-dir`/`--output-dir` default 改 `None`，`parse()` 后按 `SUITE_DEFAULTS[suite]` 补（显式 override 恒优先）**；⑥ `emit()/expected_ids()/_modality_weights()` 接受 winner_cid。libero_spatial 路径逐字不变（回归锁死）。|
| `tests/exp/fixtures/libero10_base/*.yaml` | **新（tracked fixture）**：从实际 base YAML 复制 3 份（d3/d4/d5 winner 的 base）入库，供 §5 deep-diff 测试（emitted incumbent vs 真实 base）。|
| `exp/weighted_sum/analysis/analyze_stepweight.py` | **改（非零改，G1-2 修正）**：① 加 `--task-suite`，选 `WINNER_CID_BY_SUITE[suite]` 并**贯通 `expected_ids(winner_cid, depths)`**（现 `:320` 无条件用 emitter 默认 spatial expected 集，会把 171 合法 libero_10 yaml 判为 missing/extra 而拒绝）；② `D1_NOTE`（`:46` 硬编码「same ziyang10 H200 series」）改 **suite 参数化 provenance**（libero_10 = ziyang10+xuanlel2 两台同构 H200、非同批、Stage-1 部分）；③ `--baseline-csv` default 改 `None`+按 suite 补。|
| `exp/weighted_sum/run_phase2.py` | **零改**（`--task-suite`/`--init-map`/`--yaml-dir`/`--journal` 已参数化）。|
| `exp/weighted_sum/summarize.py` | **零改**（dataset 无关，读 journal）。|
| `tests/exp/test_traj_weight_alloc.py` | **增** libero_10 用例（见 §5）。|
| `logs/README.md` | **改（G1-4）**：同 commit 加本 plan log 索引行（WA §4 Index Sync）；G2 核对状态/链接。|
| 复用零改 | `emit_yamls.{grid_weight_configs,grid3_weight_configs,build_eval_config}`、`emit_trajectory_yamls.DEPTH_WEIGHTS`、`build_depth_configs`（S1/S2/S3）、conductor/server 全不碰。|

### 3.1 emitter / analyze 泛化要点
- **modality 权重来源（分 suite，G1-1 最终）**：`_modality_weights(suite, winner_cid)`——libero_10 → **读 tracked `LIBERO10_BASE_MANIFEST[depth]['weights']`**（真实浮点，如 d3 `{vision_0:0.62, vision_1:0.37, robot_state:0.0}`、d4 `{...:0.4375, ...:0.3125}`），传 `build_eval_config(weights=...)`；libero_spatial → `grid3_weight_configs[cid]`（不动，精确、向后兼容）。**绝不**从 CID / all_results.csv / grid 反推 libero_10 权重（三者皆有损/错，见 §2.1）。manifest 缺 depth / cid 不匹配 → `SystemExit`。
- **动态默认值契约（G1-3）**：`--calibration/--artifact-dir/--output-dir`（emitter）与 `--baseline-csv`（analyze）的 argparse `default` **全设 `None`**；`args = parse()` **之后**按 `--task-suite` 从 `SUITE_DEFAULTS[suite]` 补值；**用户显式传值恒覆盖**。杜绝「只加 `--task-suite libero_10` 但仍静默用 spatial calibration/pkl/output 生成貌似合法的污染批次」。
- yaml_id `{STEM}__{winner_cid[d]}__d{d}__{label}`：grid2（`grid_...`）/grid3（`grid3_...`）cid 都是合法 stem。output_dir 默认 `config/trajectory_weight_alloc/libero_10/eval`。stale-guard 不变（仅 unlink config 下 `*.yaml` + 断言 == expected 171）。

## 4. 运行拓扑与流程

### 4.1 设备（复用 libero_spatial 主跑拓扑）
- **Server = jupyter-ziyang10**：`serve_policy.py --replicas 3 --replica-spawn-batch 2 --port 8000 --cache_config <任一 libero_10 eval yaml> policy:checkpoint --policy.config=pi05_libero --policy.dir=<pi05_libero_pytorch>`。libero_10 pkl `exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl` 已在 ziyang10（1.1GB，keybuilder=cp1_spatial_pool_16 已核）。expose → 公网入口。**单 server 钉死**。
- **Client = timan107**：`run_phase2.py ... --task-suite libero_10 --init-map exp/common/data/db/libero_cache/libero_10_init_map.json --workers 48 --gpus 8 --strategy weight`。conda env `/scratch/zixuans8/libero_sim`。

### 4.2 主跑命令
```bash
PYTHONPATH=. uv run exp/weighted_sum/run_phase2.py \
    --yaml-dir exp/weighted_sum/config/trajectory_weight_alloc/libero_10/eval \
    --init-map exp/common/data/db/libero_cache/libero_10_init_map.json \
    --journal  exp/weighted_sum/data/libero_10/trajectory_weight_alloc/journal.jsonl \
    --servers  <expose-host>:<port> --task-ids 0-9 --eval-trials 10 \
    --workers 48 --gpus 8 --strategy weight --task-suite libero_10 \
    --conda-env /scratch/zixuans8/libero_sim
```
- emit（在 client + server 侧各 emit，或 client emit 后 push；yaml 只需 client 侧）：
  `PYTHONPATH=. uv run exp/weighted_sum/emit_traj_weight_alloc.py --task-suite libero_10`（默认路径已按 suite 切）。

### 4.3 聚合 + 配对分析
```bash
uv run exp/weighted_sum/summarize.py --journal <journal> --out <results.json>
PYTHONPATH=. uv run exp/weighted_sum/analysis/analyze_stepweight.py \
    --task-suite libero_10 \
    --journal <journal> --yaml-dir <libero_10 eval-dir> \
    --out-dir exp/weighted_sum/analysis/libero_10/trajectory_weight_alloc \
    --decision-out exp/weighted_sum/data/libero_10/trajectory_weight_alloc/decision.json
```
- `--task-suite libero_10` 贯通 `expected_ids(WINNER_CID_BY_SUITE['libero_10'])` + suite provenance；`--baseline-csv` 由 suite 默认解析为 `data/libero_10/phase2/all_results.csv`。
- d1 天花板从该 csv 读（best regular-grid zscore cp1_spatial_pool_16 mean SR ≈ **0.52**，analyze 精算），provenance 标注 **ziyang10+xuanlel2 两台同构 H200、非同批、Stage-1 部分（baseline+r1，无 r2/r3）**，非裁决。

### 4.4 主跑前硬门 checklist（skill §3.13）
**拓扑仅 2 台**：server=jupyter-ziyang10、client=timan107。
① ziyang10/timan107 git SHA 一致（含本改动 commit）；② libero_10 pkl 在 ziyang10（keybuilder 已核 + sha256 记录）；③ **171 eval yaml 在 client(timan107)**——`run_phase2` 在 timan107 glob，yaml 内容经 driver 控制 WS 逐 yaml 发 server（架构同 libero_spatial：yaml 只需 client）；**server(ziyang10) 只需 1 个 bootstrap `--cache_config` yaml**（任一 libero_10 yaml，在 ziyang10 就地 emit 一个）+ pkl；**rollup 一致性在 emit 的机器间核**（本地 emit ↔ timan107 emit；ziyang10 若 emit bootstrap 亦一并核 stem）；④ server ready + 链路三层通；⑤ 1-cell smoke（~10 ep 单 yaml 无 Traceback）；⑥ L1/L2/L3 监控就位。任一 ❌ 停。

## 5. 测试策略（`tests/exp/test_traj_weight_alloc.py` 增量）

- **manifest 真实权重 + deep-diff vs 真实 base（G1-1，核心）**：① `LIBERO10_BASE_MANIFEST[d]['weights']` == 上表真实值（d3 0.62/0.37/0、d4 0.25/**0.4375**/0.3125、d5 0.5/0.5/0）；② **deep-diff 派生 incumbent yaml vs tracked base fixture**（`tests/exp/fixtures/libero10_base/*.yaml`）——emitted incumbent（manifest 权重 + incumbent `DEPTH_WEIGHTS`）与真实 base fixture **diff==0**（本会话已亲验 d3/d4/d5 全 0）；S1/S2 config 与 base **仅 `search_strategy.trajectory_weights` 不同**。
- **拒绝所有有损反推（回归，G1-2R）**：断言 manifest 权重 **≠** `grid_weight_configs[cid]`/`grid3_weight_configs[cid]`（d3 0.62≠0.625）**且 ≠** `int(cid_@NN)/100`（d4 0.4375≠0.43、0.3125≠0.31）——守住「manifest 是唯一 source of truth，不得用 CID/all_results.csv/grid 反推」。
- **winner 身份 tracked 交叉核验**：`WINNER_CID_BY_SUITE['libero_10']` 3 个 cid **恰等** `threshold_pareto_per_yaml.csv` per-depth base，且 == manifest 的 cid。
- **grid 家族契约**：d3/d5 rs 权重==0、d4 rs 权重>0；rs 恒为 key 字段（cp1 要求）；全过 `validate_cache_config`。
- **计数锁死**：libero_10 emit 仍 d3=52/d4=60/d5=59/总 171；每向量 长度==depth、全>0；canonical 去重无重复。
- **analyze suite 贯通（G1-2，main 级）**：`--task-suite libero_10` + 完整 libero_10 锁定集 **通过**；混入 spatial ID / 截断集 **SystemExit**；`decision.json.d1_prior_note` provenance == libero_10（ziyang10+xuanlel2）。
- **动态默认值契约（G1-3）**：`--task-suite libero_10` 无其他参数 → emitter 的 calibration/artifact/output 与 analyze 的 baseline-csv **解析为 libero_10 路径**；逐项显式 override 生效（不被 suite 默认覆盖）。
- **向后兼容回归**：`--task-suite libero_spatial`（emitter+analyze）输出与既有**逐字一致**（rollup 不变、`expected_ids` 不变、D1_NOTE 不变）——守住 libero_spatial 已批准行为。
- **派生-YAML 契约**：每 yaml trajectory_depth 正确、ID 唯一、总数==171、d1 不在 eval。
- **§6 Verify**：`uv run pytest` 全绿（改动局限 emitter + analyze + tests；libero_spatial 回归干净）。

## 6. 风险登记

| # | 风险 | 缓解 |
|---|---|---|
| R1 | **base 权重正确性 — 三层来源全有损（CID / all_results.csv / grid 均错或不全对）** | **固化 tracked 非有损 manifest `LIBERO10_BASE_MANIFEST`**（值取自实际 base YAML，本会话 deep-diff=0 亲验）；emitter 只读 manifest；单测 deep-diff emitted incumbent vs tracked base fixture（==0）+ 回归拒绝 CID/csv/grid 反推（G1-1/G1-2R）|
| R2 | **winner 选择 provenance 非机读** — libero_10 trajectory 选 base 的逐 depth SR 只有 pdf/png（`analysis/libero_10/trajectory/*`），无 tracked csv/decision | base **身份**由 threshold csv 证明（downstream，够定 WINNER_CID）；plan 明示 SR-level provenance 不入库；不影响本实验（本实验只需 base 身份）|
| R3 | **baseline csv 部分** — `data/libero_10/phase2/all_results.csv` 只有 baseline+r1（无 r2/r3）| d1 prior 仍可算（best zscore mean≈0.52）；标注为**部分 Stage-1** 的非裁决 prior|
| R4 | **pkl + init_map 未 tracked**（gitignored data，present locally/远端）| 与 libero_spatial 同性质（data/ 天然 gitignore）；主跑前核 pkl 在 ziyang10 + sha256、init_map 在 client；clone 不可复现属既有约束（非本改动引入）|
| R5 | **100 ep 功效 / 多重比较** | 同 libero_spatial：定义为 screening，报候选 + 配对 CI，不裁决；确认性重跑留后续|
| R6 | **可复现** | config 由 emitter 从 tracked calibration + **tracked manifest（真实浮点权重）** 确定性重建（不依赖 gitignored base YAML 或有损 csv/CID）；2 机 emit rollup 核对|
| R7 | **d1 天花板跨批 + 绝对 SR 低（~0.52）** | libero_10 更难本属正常；组内配对（171+incumbent 同 server 同 pkl 同 held-out init）有效；d1 非同批标注|

## 7. 锁定决策（G1 APPROVED）

- **搜索矩阵**：171（52/60/59）× 100 ep = 17,100 ep，与 libero_spatial 同。
- **libero_10 base 真实权重（tracked manifest，非有损；deep-diff=0 亲验）**：d3 `grid_vision_0@62_vision_1@37`=**0.62/0.37/0.00**；d4 `grid3_vision_0@25_vision_1@43_robot_state@31`=**0.25/0.4375/0.3125**；d5 `grid_vision_0@50_vision_1@50`=**0.50/0.50/0.00**。**禁止**从 CID/all_results.csv/grid 反推（三者皆有损/错）。
- **代码**：emitter 加 `--task-suite` + `WINNER_CID_BY_SUITE` + **`LIBERO10_BASE_MANIFEST`（emitter 只读它取 libero_10 权重；libero_spatial 保 grid3）** + **动态默认值(None→post-parse)**；**analyze 非零改**（加 `--task-suite`、贯通 `expected_ids`、参数化 provenance、动态 baseline 默认）；run_phase2/summarize 零改；libero_spatial 向后兼容锁死；**3 份实际 base YAML 入库为 tracked fixture**；**同 commit 更新 `logs/README.md`**。
- **codename**：`trajectory_weight_alloc`（libero_10 子目录）；config gitignore-but-regenerable。
- **定位**：screening；确认性重跑为后续。
- **d1**：不加同批锚点，用 libero_10 既有 SR(~0.52) 作非裁决 prior（provenance=ziyang10+xuanlel2 两台同构 H200、Stage-1 部分）。

---

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-07-02 09:58 CDT

- [Blocking] [Concern] suite 泛化破坏了已批准的 libero_spatial 位置参数接口，未达到“向后兼容锁死”。— reasoning: 变更前 `expected_ids(depths=...)` 接受 `expected_ids((3,))`，`emit(output_dir, calibration, artifact_dir, depths)` 的第 4 个位置参数也是 depths；当前把 `winner_cid` 放到 `expected_ids` 第 1 位、把 `suite` 放到 `emit` 第 4 位。独立 probe 实测 `expected_ids((3,))` 抛 `IndexError`，且 `inspect.signature(emit)` 第 4 参数已变成 `suite`（2 failed）。请保持旧参数顺序，例如 `expected_ids(depths=DEFAULT_DEPTHS, *, winner_cid=WINNER_CID)` 与 `emit(output_dir, calibration, artifact_dir, depths=DEFAULT_DEPTHS, *, suite='libero_spatial', winner_cid=None)`，suite 新参数使用 keyword-only；更新内部调用，并提交旧位置调用回归测试。
- [Blocking] [Concern] G1 锁定的动态默认值/显式 override 测试没有实际实现。— reasoning: 提交的 `test_suite_defaults_resolve_and_override` 只断言 `SUITE_DEFAULTS` 字典字符串包含 suite，既没有调用 `main()`/argparse 后分派，也没有传任何 override，测试名与覆盖内容不符；因此 `args.* or default` 分支发生回归时仍会全绿。独立 probe 证明当前非空 override 行为正确，但该契约需要进入提交测试：mock `emit` 捕获 `main()` 参数，分别验证 `--task-suite libero_10` 的三项默认路径及 `--calibration/--artifact-dir/--output-dir` 显式覆盖。
- [Non-blocking] [Concern] plan 头仍写 `In Progress / §4 Code`，而 `logs/README.md` 已写“Code 完成 / 待 G2”；§7 标题仍是“待 G1 确认”。— reasoning: 不影响实现正确性，但 G2 下一轮前应同步为当前 gate 状态，避免日志与索引互相矛盾。
- [Non-blocking] [Verification] 提交测试 `44 passed`，ruff `All checks passed`；独立端到端 probe 还确认现有 libero_10 eval 目录恰为锁定 171 ID，逐 YAML 与 manifest+calibration 重建完全一致，只有 `trajectory_weights` 随搜索矩阵变化；CLI 默认/非空 override probe 通过。独立兼容 probe总计 `2 failed, 2 passed`，失败即上述旧位置接口回归。

### G2 Round 1 — Executor — 2026-07-02

- **G2-1（位置参数接口破坏，向后兼容未锁死）— Accepted**。属实。已改为审查者建议的签名：`expected_ids(depths=DEFAULT_DEPTHS, *, winner_cid=WINNER_CID)`、`emit(output_dir, calibration, artifact_dir, depths=DEFAULT_DEPTHS, *, suite='libero_spatial', winner_cid=None)`——**depths 复位到原位置槽、suite/winner_cid 改 keyword-only**。更新内部调用：`emit` 内 `expected_ids(depths, winner_cid=winner_cid)`；`main` 内 `emit(..., depths, suite=suite, winner_cid=winner_cid)`；`analyze` 内 `expected_ids(depths, winner_cid=winner_cid)`；修正测试中误用位置传 winner_cid 的 3 处为 keyword。新增回归 `test_backward_compat_positional_interface`：断言 `expected_ids((3,))==expected_ids(depths=(3,))`、`emit` 前 4 位置参数序为 `output_dir/calibration/artifact_dir/depths`、suite/winner_cid 为 KEYWORD_ONLY。libero_spatial emit rollup `50e84868…` 逐字节不变。
- **G2-2（动态默认值/override 测试形同虚设）— Accepted**。属实，原 `test_suite_defaults_resolve_and_override` 只查 dict 字符串、不跑 main。已替换为 `test_emit_main_dynamic_defaults`：**monkeypatch `E.emit` 捕获 `main()` 实参**，(1) `--task-suite libero_10` 无 override → 三项默认（calibration/artifact-dir/output-dir）+ winner_cid 均解析为 libero_10；(2) 显式 `--calibration/--artifact-dir/--output-dir` → override 生效。`args.* or default` 回归会被此测试捕获。
- **G2-3（plan 头/§7 与 README 状态不一致）— Accepted**。plan 头改为「§4 Code 完成 / 待 G2」；§7 标题「待 G1 确认」→「G1 APPROVED」。与 README「Code 完成/待 G2」一致。
- **G2-4（[Verification] 正面）— Noted**。感谢确认 e2e/manifest 重建/171 ID/唯一变量与 CLI probe；两处失败正是 G2-1，已按上修复。复跑：`pytest tests/exp/test_traj_weight_alloc.py` **45 passed**、`ruff` All checks passed、两 suite emit 端到端通过（libero_spatial rollup 不变、libero_10 171）。

### G2 Round 2 — Reviewer — APPROVED — 2026-07-02 10:06 CDT

- [Resolved] 旧位置接口已恢复：`expected_ids((3,))` 正常保留 depths 的首位置槽，`emit(..., depths)` 保留第 4 位置槽；新增 suite/winner 参数均为 keyword-only，分析器和 emitter 内部调用同步正确。
- [Resolved] 提交测试现实际执行 `main()` 并捕获 `emit` 入参，覆盖 libero_10 三项动态默认路径及显式 override，不再只是检查常量字典。
- [Verification] 提交测试 `45 passed`；独立兼容、CLI 分派和现有 libero_10 171-YAML zero-diff probes `4 passed`；ruff `All checks passed`。fixtures 与本地实际 d3/d4/d5 base YAML 逐字一致，日志索引和 gate 状态已同步。
