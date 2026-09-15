# Key builder × LDA 补充实验 plan（π0.5 纯 cache，A 池 500 集）

> 状态：`Plan`（2026-09-15，owner 令草拟；G1 Round 1 NEEDS REVISION → Round 2 修订版交审）。级别 **L1**：只出 yaml、复用现有库/builder/运行链，不改 `src/`；
> 本轮涉及的 `exp/` 文件改动列在 §7。CLIP ViT-L/14 因需要 `src/openpi/cache/config.py` 的 clip 工厂加模型变体接线，**本轮延后**（§2 D4）。
> 上游：`docs/iclr/modality_weight_selection.md`（LDA 主方法、§4.5/§4.6 结果）、`exp/weighted_sum/RESULTS.md`（最初的 pure-cache weighted 探索：有效基线为 4 个 CP1 pool builder × 34 配置，B 池 n=100；CLIP 在该结果记录中已被移除）、`docs/cache/llm_layer_extract.md`（不区分模态的 LLM 层单 key）。

## 1. 目的

最初的探索在 B 池 100 集/格上比较过四种 CP1 pool 降维方式（mean / max / spatial_16 / spatial_64），每种都靠网格搜权重；CLIP 外部视觉模型建过库但未进入有效结果。
现在权重有了零成本闭式规则（LDA），把每种降维方式配上**固定 LDA 规则所得的权重**，在 A 池 500 集上各跑一组，得到"降维方式 × LDA 权重"的同设置对照表；
再加一个**不区分模态、对整个 prefix 直接降维的单 key** 方法，它只有一个参与打分的字段，没有权重可选，直接跑。全部纯 cache（`always_search` + `always_hit` + `top_k 1` + `write_policy never`）。
本表回答"在 LDA 规则下哪种降维方式更好"，不声称 LDA 对任意 builder 都达到该 builder 的最优（§4.6 只在 spatial_16 上验证过）。

## 2. 配置（π0.5 × {LIBERO Spatial, LIBERO-10}，每配置 500 集 = 10 任务 × 50 官方 A 池 init）

| # | 降维方式（key builder） | 库（`exp/common/data/cache_artifacts/<suite>/`） | 参与打分的字段 | 权重 | yaml_id / 臂名 |
|---|---|---|---|---|---|
| 1 | `cp1_mean_pool` | `cp1_mean_pool.pkl`（1018 / 2640） | v0, v1, rs | LDA | `kb_cp1_mean_pool_lda` |
| 2 | `cp1_max_pool` | `cp1_max_pool.pkl` | v0, v1, rs | LDA | `kb_cp1_max_pool_lda` |
| 3 | `cp1_spatial_pool_64`（= 2×2） | `cp1_spatial_pool_64.pkl` | v0, v1, rs | LDA | `kb_cp1_spatial_pool_64_lda` |
| 4 | `clip`（ViT-B/32 openai，外部视觉模型；线上工厂默认变体） | `clip_vit_b_32.pkl`（v 512 维） | v0, v1, rs | LDA | `kb_clip_vit_b_32_lda` |
| 5 | `cp1_llm_layer_extract` + `prefix_mean_pool`，layer 0（**不区分模态**：整段 prefix 过 LLM 第 0 层后均值池化成一个 2048 维 key；Pi0.5 的 state 已离散进 prefix） | spatial：`llm_layer_extract/cp1_llm_l0_prefix_mean_pool.pkl`；**libero_10 需新建**（§4 步骤 2） | 仅 `vision_0`（`robot_state` 按校验要求保持 `enabled: true, weight: 0.0`，零权重不参与检索） | 无 | `kb_llm_l0_prefix_mean_pool` |
| 参考 | `cp1_spatial_pool_16` | 已有 | v0, v1, rs | LDA | §4.5 已跑：0.686 / 0.442；§4.6 同 A 池 1/6 网格最高格 0.688 / 0.486（描述性上界） |

两轮运行（§3.4）：**pool 轮** = 配置 1–4（`matrix_<suite>_pool.yaml`，4 臂）；**LLM 轮** = 配置 5（`matrix_<suite>_llm.yaml`，1 臂）。合计新跑 5 × 2 × 500 = **5,000 集**。

**裁决（已定）**
- **D1**（已定）："不区分模态直接降维" = `cp1_llm_layer_extract` + `prefix_mean_pool`，layer 0，单字段 `vision_0` 参与打分。现有库没有 prefix 的 spatial-16 版；`per_modality_spatial_pool_16` 有模态划分、需要权重，不符合"不用决定权重"。
- **D2**（本轮不做）：prefix key + robot_state 两字段 LDA 对照。`lcw_fit_weights.py` 固定三字段、3×3 协方差，两字段版要改脚本；留作后续，不进本轮矩阵与预算。
- **D3**（已定）：LLM 只取 layer 0。
- **D4**（新）：CLIP ViT-L/14 延后。`config.py:4232` 的 clip 分支只调用 `CLIPKeyBuilder(enabled_fields=...)`，`clip_key_builder.py:103` 默认 `ViT-B-32/openai`；L/14 库 vision 768 维，线上 builder 会产生 512 维 key，与库不匹配。上线需要 `key_builder` 增加模型变体字段、工厂接线并与 artifact 元数据绑定，属 src 改动（L2），另立计划。

## 3. 方法（每个配置）

1. **Phase-1 normalizer**：沿用 `exp/weighted_sum/data/<suite>/phase1/calibration_normalizers.json` 中该 artifact stem 的 `selected`（spatial 六种齐全；libero_10 缺 CLIP B/32 与 LLM 库、spatial 缺 LLM 库 → 用 `exp/common/calibrate_score_normalizers.py --artifact-dir <只含该 pkl 的目录> --output <新 json>` 补标定，LOEO、CPU、分钟级）。
   **兼容门**：`lcw_fit_weights.py` / `lcw_offline_stats.py` 只实现 zscore(tanh)；补标定的 `selected.method` 若不是 `zscore`，脚本须 fail-loud（本轮给 `lcw_fit_weights.py` 加断言），并改为显式用 zscore 的 shortlist 项或另适配，不得静默按 zscore 读 μ/σ。
2. **LDA 权重**（三字段库 1–4）：先用 `exp/weighted_sum/emit_yamls.build_eval_config` 按该 stem 的 selected normalizer 生成一份三字段模板（权重任意），再
   `PYTHONPATH=. uv run python exp/weighted_sum/analysis/lcw_fit_weights.py --library <pkl> --template <模板> --cells <该 suite 的 grid6 cells json> --output exp/weighted_sum/data/keybuilder_lda/fits/<suite>/<stem>`，
   取 `fits["lda@0.05"].w`（脚本同时输出 Fisher/pairwise/动作标签变体，仅 LDA 进入配置；`--cells` 只用于打印，其 leader/SR 不作为本次结果）。
   `weights.json` 逐 builder 记录：库路径与 sha256、标定 json 与 selected 方法/参数、模板路径、`detail@0.05.fisher_lda` 的 d / cov / lda_raw、截断后权重。
3. **yaml**：`exp/weighted_sum/emit_keybuilder_lda.py`（新）调 `build_eval_config` 注入 LDA 权重与 h100 上的 `preload_path`，写 `exp/weighted_sum/config/keybuilder_lda/<suite>/<yaml_id>.yaml`；
   配置 5：`keys.vision_0 {enabled true, weight 1.0}`，`keys.robot_state {enabled true, weight 0.0}`，其余字段关闭，`backend.vector_dims` 保留 artifact 全字段集，`key_builder.type cp1_llm_layer_extract`、`llm_layer 0`、`prefix_reducer.type prefix_mean_pool`（按 `docs/cache/llm_layer_extract.md` §2 的 YAML 模板）。
   出 yaml 后逐份 `load_cache_config` 验收：臂名唯一、`vector_dims` 等于 artifact 全字段集、权重与 weights.json 一致、`preload_path` 在 h100 存在、normalizer 方法为 zscore。
4. **运行**：lane P 配方，`exp/weighted_sum/ops/fw_lane_pi05.sh` 本轮加两个环境变量：`FW_STAGE2_DEVICE`（默认 `meta`）与既有 `FW_OUT` / `FW_CFG_DIR` / `FW_BOOT` / `FW_MATRIX`（新，指定 matrix 文件）。h100 5 server（23280–23284）+ timan108 60 worker（`cache_prune/ops/launch_fleet.sh`），`run_size_eval --role driver`，journal 续跑。
   - **pool 轮**：`FW_MATRIX=matrix_<suite>_pool.yaml`，`FW_BOOT=kb_cp1_mean_pool_lda.yaml`，`FW_STAGE2_DEVICE=meta`，`FW_OUT=/data/openpi/weighted_sum/keybuilder_lda/pool`；CLIP 臂的每端点 worker 数按 §6 B4 预检结果定（默认 12，预检超限则 6）。
   - **LLM 轮**：`FW_MATRIX=matrix_<suite>_llm.yaml`，`FW_BOOT=kb_llm_l0_prefix_mean_pool.yaml`，`FW_STAGE2_DEVICE=cuda:0`，`FW_OUT=/data/openpi/weighted_sum/keybuilder_lda/llm`。
   - smoke 输出 `FW_OUT=/data/openpi/weighted_sum/keybuilder_lda_smoke/<round>`，绝不写进正式目录。
   - 两轮各自 `DRIVER_EXIT=0`、`FULL_HIT rates` 全 1.0 的 driver.log 保留为证据。
5. **判读**：`lcw_ablation_summary.py`（本轮改为同 suite 可传多份 journal 并合并，且断言每臂恰 500 accepted、每任务 50、配对交集 500）读 pool 轮 + LLM 轮 + §4.5 的 spatial_16 LDA 臂 journal（同 lane、同池，作参考臂），出每配置 SR + Wilson 95% 与同 init 配对符号检验；多对比较标注为探索性（不做校正）。
   与最初 B 池探索（4 个 CP1 pool 的 B 池最优）只作方向对照，不并表。LLM 臂的成功率可同 init 比较；其在线代价含 builder 内 layer-0 forward（`llm_layer_key_builder.py:188` 在判定 FULL_HIT 前执行 `_extract`），时延/吞吐另报，不与 pool 臂混算。

## 4. 步骤与预算

| 步 | 内容 | 机器 | 时长 |
|---|---|---|---|
| 1 | **建库**：libero_10 的 `cp1_llm_l0_prefix_mean_pool.pkl`（`exp/common/build_in_memory_cache_artifact.py --builder-type cp1_llm_layer_extract`，原料 = 建 `cp1_spatial_pool_16.pkl` 的同一批 H5）；**验收**：entries 的 `(task_key, trajectory_id, step_idx)` 集合与 `cp1_spatial_pool_16.pkl` 逐条相等，`action_chunk` 逐条 `allclose`，`vector_dims == {vision_0: 2048, robot_state: 32}` | h100（PaliGemma 层 0） | 30 min |
| 2 | 补标定（步骤 1 之后）：libero_10 × clip_b32；两 suite × llm_l0_prefix_mean_pool；记录 selected 方法，非 zscore 触发 §3.1 兼容门 | 本机 CPU | 10 min |
| 3 | LDA 权重 × 4 builder × 2 suite + weights.json；出 10 个 yaml + 4 个 matrix；逐份 load 验收（§3.3） | 本机 | 15 min |
| 4 | 推送（tether，sha 对账）；**预检门**（§6）：CLIP 显存/并发实测、LLM parity manual 测试、每种上线路径的真实 key 构造 smoke | h100 / timan108 | 40 min |
| 5 | pool 轮正式（4,000 集，spatial ≈ 200 集/分、l10 ≈ 110 集/分） | h100 + timan108 | ≈ 45 min |
| 6 | LLM 轮正式（1,000 集，stage 2 在 GPU） | 同上 | ≈ 15 min |
| 7 | 拉回、聚合、写表进纪要 §4.7 与运行记录 | 本机 | 15 min |

总计约 3 小时，其中机器时间约 1.5 小时。

## 5. 产物

- yaml/matrix：`exp/weighted_sum/config/keybuilder_lda/<suite>/{kb_*.yaml, matrix_<suite>_pool.yaml, matrix_<suite>_llm.yaml}`；发射前断言 pool matrix 4 臂、llm matrix 1 臂、臂名全局唯一。
- 权重与来源：`exp/weighted_sum/data/keybuilder_lda/weights.json`、`fits/<suite>/<stem>/lcw_fit_weights.json`；补标定 json 与新库 sha256 记录。
- 原始数据：`exp/weighted_sum/data/keybuilder_lda/{pool,llm}/<suite>/{journal,per_step}.jsonl`（gitignored）；汇总 `summary.json`；预检记录 `preflight.json`（CLIP 显存峰值、连接数、parity 测试结果）。
- 报告：`docs/iclr/modality_weight_selection.md` 新增 §4.7"降维方式 × LDA"表；`logs/fusion_weight_ablation_run.md` 追加运行事实。

## 6. 风险与预检门（正式跑前全部过）

- **B4/R1 CLIP 显存按连接数计**：`scripts/serve_policy.py:535` 每连接 `build_per_connection_components` → `config.py:3931` 新建 builder，`CLIPKeyBuilder._clip_model` 逐实例懒加载（`clip_key_builder.py:125`）。5 server × 12 worker = 60 个连接 ⇒ 最多 60 份 CLIP B/32 权重，不是 5 份。预检：按计划并发起 5 server + 60 worker 跑 CLIP 臂 smoke（每任务 1 集），`nvidia-smi` 记显存峰值与进程数；峰值 > 60 GB 则该臂改 6 worker/端点重测；仍超则 CLIP 臂延后。端口监听与库加载都在 CLIP 懒加载之前，不能替代此检查。另预检 h100 venv `open_clip` 可导入且 `ViT-B-32/openai` 权重能离线加载。
- **B5 表示一致性与配置验收**：① 每份最终 yaml `load_cache_config` 验收（§3.3）；② CLIP 臂与 LLM 臂各做真实 key 构造 + 检索 smoke（10 集，FULL_HIT 1.0，`__hit_meta__` 有 winner）；③ LLM 在 h100 用 `pi05_libero` checkpoint 跑 `uv run pytest tests/cache/test_llm_layer_extract_parity.py -m manual -v`，记录非 skip 的通过结果（线上/离线 pad mask、layer-0 hidden、最终 key 一致）；④ 新 L10 LLM 库按 §4 步骤 1 验收后再标定；⑤ 补标定 selected 非 zscore 时停止 LDA 路径（§3.1）。
- **R2 libero_10 的 LLM 库原料**：建库走"重 tokenize + tokenizer self-check"合同，原料 H5 必须是建 `cp1_spatial_pool_16.pkl` 的那一批（弄清其来源目录并记录 sha 清单）；若原料不在 h100，先从 weilandserver 拉。
- **R3 LDA 字段**：CLIP 库 vision 512 维，normalizer 用其自身 LOEO 标定；三字段库都用 (v0, v1, rs)；配置 5 不走 LDA。
- **R4 与 §4.6 网格的可比性**：pool 轮与网格同 server 设置；LLM 轮 stage 2 在 GPU，纯 cache 的 FULL_HIT 路径不经 `run_stage2`，但 builder 内跑 layer 0，成功率可比、成本不可比，表注写明。
- **R5 同机争用**：只用 h100 + timan108；weilandserver 有另一 session 的 online RIT 线，不碰。

## 7. 本轮改动的文件（全部 `exp/`，无 `src/`）

- 新：`exp/weighted_sum/emit_keybuilder_lda.py`（LDA 注入 + 配置 5 的单 key yaml + 两种 matrix + load 验收）。
- 改：`exp/weighted_sum/ops/fw_lane_pi05.sh`（`FW_STAGE2_DEVICE`、`FW_MATRIX` 环境变量，默认行为不变）；`exp/weighted_sum/analysis/lcw_ablation_summary.py`（同 suite 多 journal 合并 + 完整性断言）；`exp/weighted_sum/analysis/lcw_fit_weights.py`（normalizer 方法断言：非 zscore 即退出）。
- 建库产物：`exp/common/data/cache_artifacts/libero_10/llm_layer_extract/cp1_llm_l0_prefix_mean_pool.pkl`（gitignored）。

## Review Log

### G1 Round 1 — Reviewer — NEEDS REVISION — 2026-09-15 10:01 CDT

Authority: Review。范围：本计划全文及其引用的现有实现；本轮依据工作树审查，无上一轮暂存基线。核对了 `WORKING_AGREEMENT.md`、审查法、docs/logs 索引、cache 架构与 tutorial、数据采集和实验产物布局规则，以及 LDA 纪要、weighted_sum 历史结果、LLM 使用指南和 fusion ablation 运行记录。未运行实验或 G2 独立测试，以下判断依据源码和本地数据的只读清点。

#### Checklist

| 项 | 结论 | 依据 |
|---|---|---|
| 架构一致性 | 原则通过；范围须随 B1 的处置更新 | 复用 CP1 KeyBuilder、weighted_score_sum_knn 和冻结库，符合插件边界；LLM 在 builder 内借用层引用。现有配置接口不能实现 L/14，若补接口，须按计划首段重新评估级别与文件范围。 |
| 接口兼容性 | 不通过 | B1：CLIP 线上工厂固定 B/32；B2：LLM 禁用 robot_state 会被配置校验拒绝；B3：现成 lane 无 Stage-2 GPU 参数入口，分轮选臂和输出目录未闭合。 |
| 风险识别 | 不通过 | B4：CLIP 权重由各连接的 builder 分别加载，不能按每 server 一份估计显存。已有库数量可核对；尚未建出的 L10 LLM 库仍须按原料清单逐条验收。 |
| 测试策略 | 不通过 | B5：每 suite 随取一个配置的 smoke 无法覆盖两种 CLIP 与 LLM 的模型、维度和在线/离线一致性；补标定后的实际 normalizer 也须与 LDA 的实现一致。 |

#### Blocking

- [Blocking] [Concern] **B1 — L/14 无法仅靠现有 YAML 上线（§2 第 5 项、§3.3、R1）。** 请明确选择补充模型变体配置/工厂接线的实现计划，或将 L/14 明确延后并修订本轮矩阵与预算；若补接口，应同时绑定模型名、pretrained 与库元数据，覆盖两个变体的在线 key 维度。— reasoning: `src/openpi/cache/config.py:4232` 的 clip 分支仅调用 `CLIPKeyBuilder(enabled_fields=enabled_fields)`，没有从 YAML 或 artifact 传入模型变体；`components/clip_key_builder.py:103` 默认 `ViT-B-32/openai`。本地两套件 L/14 库元数据均为 `ViT-L-14/openai`、vision 768 维，默认在线 builder 则产生 512 维 key。安装依赖、缓存 L/14 权重或只换 preload_path 都不能修复该接口缺口。

- [Blocking] [Concern] **B2 — 单 key 的 robot_state 配置必须改成可加载的形式（§2 第 6 项、§3.3）。** 在不改 src 的方案内，保留 `robot_state: {enabled: true, weight: 0.0}`，vision_0 权重 1，其余视觉/prompt 字段关闭；backend.vector_dims 保留 artifact 全字段集。明确“单 key”指唯一参与打分的字段。— reasoning: `src/openpi/cache/config.py:2799` 对所有 `cp1_*` 强制要求 vision_0 与 robot_state enabled，LLM 分支没有豁免；`exp/weighted_sum/emit_yamls.py:75` 已用零权重启用解决此约束，计划随后“其余 enabled false”会破坏它。本地 Spatial LLM 库 dims 正是 `{vision_0: 2048, robot_state: 32}`；`in_memory_backend.py:935` 的 active-field 筛选排除 weight<=0，故零权重不参与检索，无需额外选权重。

- [Blocking] [Concern] **B3 — 补全两轮运行的实际入口、选臂与输出绑定（§3.4、§4、§5）。** 给出普通轮与 LLM 轮各自可执行的启动配方：如何设置 Stage 2 GPU、各轮的 `--arms`/独立 matrix 与 BOOT、独立实验的 `FW_OUT`、smoke 输出目录，以及分轮结果如何归并。若使用新的 exp launcher 或调整现有 launcher，把文件与参数接口列入范围。— reasoning: `fw_lane_pi05.sh:44` 将 Stage 2 固定为 meta，尾部参数只传 driver，`FW_BOOT` 不能改变设备；`run_size_eval.py:612` 在不传 `--arms` 时读取整个 matrix，因此一个含六臂的 matrix 不能自动把 LLM 留到第二轮。`fw_lane_pi05.sh:26` 默认写入旧 `fusion_ablation/<suite>`，计划只覆盖 CFG/BOOT 会向旧 journal 追加，并覆盖该目录内 server/driver 日志。需在启动前明确配置，不应依赖运行中临时改脚本。

- [Blocking] [Concern] **B4 — 按实际连接数重算 CLIP 显存与并发预检（R1、§4 预算）。** 将“h100 80 GB 无压力”改为待验证条件，明确每端点 worker 数、CLIP 同时存活的连接/模型数与变体切换方式；正式启动前用计划的并发规模完成实际 collect/build 与显存峰值检查，内存不足时明确减并发或延后该臂。— reasoning: `scripts/serve_policy.py:535` 为每连接调用 `build_per_connection_components`，`config.py:3931` 创建新的 builder；每个 `CLIPKeyBuilder` 的 `_clip_model` 是独立实例，首次 collect 调 `open_clip.create_model_and_transforms`。5 server × 12 worker 的模型副本数可能随约 60 个连接增长，并非仅 5 份。“每进程一种变体”不等于“每进程一份权重”；端口监听/库加载成功也发生在 CLIP 懒加载前，不能替代该检查。

- [Blocking] [Concern] **B5 — 将表示一致性与配置验收加入正式跑前的测试门（§4 第 4 步）。** 至少逐份加载最终 YAML 核验唯一臂名、全字段 dims、权重、库和 normalizer；每种上线的 CLIP 变体及 LLM 路径须各有真实 key 构造/检索 smoke。LLM 在所用 checkpoint 上运行 `tests/cache/test_llm_layer_extract_parity.py` 的 manual parity，并记录非 skip 的结果；新 L10 库验收 `(task, trajectory_id, step_idx)` 与 action 对齐，随后再标定。补标定若选中非 zscore(tanh)，应停止现有 LDA 路径或明确适配实际 normalizer。— reasoning: 当前每 suite 仅一个配置可能只测到 mean_pool，无法发现 B1/B2/B4；LLM 指南 §6 已要求线上/离线 pad mask、layer-0 hidden 和最终 key 一致，建库首步 tokenizer self-check 仅检查 prompt embedding，覆盖范围不同。`lcw_fit_weights.py:151` 直接读取 mu/sigma，`lcw_offline_stats.normalized_scores` 固定 zscore+tanh；`calibrate_score_normalizers.py` 的 selected 可以是其他方法，不能以旧库的选择结果保证新标定。正式每臂 500 个 accepted episode 与 FULL_HIT=1 可沿用 driver 现有检查，但必须保留各轮成功退出的证据。

#### Non-blocking

- [Non-blocking] [Suggestion] **N1 — 完成 LDA 的可复现调用与依赖顺序（§3.2、§4）。** 补上必填 `--output`，明确用于拟合的三字段模板如何先生成、最终权重如何再注入；L10 LLM 建库应先于该库标定。weights.json 同时记录库/标定/模板身份及 `detail@0.05.fisher_lda` 的 d、cov、raw、w。— reasoning: 当前命令缺必填参数；CLI 还会运行多个 Fisher/pairwise/action-label 变体，并非只做一次 LDA，秒级预算不能直接由闭式解的成本推出。`--cells` 不影响 LDA 解，但其中历史 leader/SR 会写入输出 JSON，不能把任意 builder 的对照值当成本次结果。

- [Non-blocking] [Suggestion] **N2 — 为 CLIP 两变体指定独立 artifact_id/yaml_id（§3.3、§5）。** 例如 `kb_clip_vit_b_32_lda` 与 `kb_clip_vit_l_14_lda`，而非均按 builder_type=`clip` 拼接；在产物清单中断言本轮实际 YAML/matrix 臂数。— reasoning: 两库在校准 JSON 中已用不同 artifact stem 区分，`build_eval_config` 返回的 key_builder.type 却相同；`kb_<builder>_lda.yaml` 需要明确采用哪种标识，否则实现时容易覆盖配置或合并 journal 身份。

- [Non-blocking] [Concern] **N3 — 收紧历史描述和结果解释（§1、§3.5、R4）。** 将“最优级权重”写成“固定 LDA 规则所得权重”；说明现有 B 池有效网格结果是四种 CP1 pool，CLIP 在该结果记录中被移除。LLM 的成本说明应计入 builder 内 layer-0 forward；成功率可以同 init 比较，时延与吞吐须另报。— reasoning: `exp/weighted_sum/RESULTS.md` §二/三明确有效基线为 4×34 配置，不能据此声称两种 CLIP 已有可用网格结果。上游 §4.6 也未证明 LDA 对任意 builder 达到最优；`llm_layer_key_builder.py:188` 在判定 FULL_HIT 之前执行 `_extract`，绕过 `run_stage2` 不等于完全不使用 LLM 算力。

- [Non-blocking] [Suggestion] **N4 — 明确可选范围及汇总合同（D1–D3、§3.5、§5）。** 在发射前把 D1/D3 的最终选择写成固定配置；D2 若另开，应增加两字段 LDA 的拟合方案及预算，现有三字段脚本不能直接用。写清多轮 journal 与既有 spatial_16 参考臂的输入映射、每臂 500/每任务 50/配对交集 500 的验收，以及参考最高格为同 A 池网格选出的描述性上界；多对比较可标明探索性或指定校正。— reasoning: `lcw_ablation_summary.py` 能读任意臂名，但同 suite 多次 `--pi05-journal` 会覆盖 report 的同名 suite，不能自动合并分轮输入；它也不会因臂缺集而拒绝汇总。D2 的库只有 vision_0/robot_state，`lcw_fit_weights` 固定读取三字段并构造 3×3 协方差。当前 6,000 集计数未含 D2，应保持可选项与正式矩阵一致。

#### Evidence and disposition

- 本地 artifact 只读清点：Spatial 的 mean/max/spatial_64、CLIP B/32、CLIP L/14、LLM l0 prefix mean 均为 1,018 entries / 49 trajectories / 10 tasks；L10 前五种均为 2,640 / 50 / 10，L10 LLM 文件尚不存在。这里只确认数量与元数据，不宣称已完成逐条 provenance/parity 校验。
- 当前已存在的两套 Phase-1 JSON：Spatial 六种 builder 与 L10 四种 pool 的三字段 selected 均为 zscore(tanh)。L10 CLIP 和两套件 LLM 新标定尚未完成。
- 未发现目标计划已经实施的 Working Agreement 违规；当前 L1 不因主动请求 G1 而自动升级。`logs/README.md` 已有本计划的新增索引项；修订者应在下一次更新正文/状态时同步索引。本轮仅追加 Review Log，不代执行方修改正文或源码。

NEEDS REVISION: B1–B5。执行方须逐项响应并更新计划后再交审；本轮未给出 plan approved。

### G1 Round 1 — Executor — 2026-09-15

- **B1 Accepted** — 核实 `config.py:4232` clip 分支无模型变体参数、builder 默认 `ViT-B-32/openai`，L/14 库 768 维与线上 512 维 key 不匹配。本轮**延后 L/14**（§2 D4），矩阵改为 5 配置 / 5,000 集，B/32 保留并按线上默认变体标明；L/14 的接口接线（`key_builder` 变体字段 + 工厂 + artifact 绑定）另立 L2 计划。
- **B2 Accepted** — 配置 5 改为 `vision_0 {enabled, weight 1.0}` + `robot_state {enabled, weight 0.0}`，其余关闭，`vector_dims` 保留全字段；"单 key"明确为唯一参与打分的字段（§2、§3.3）。
- **B3 Accepted** — §3.4 写出 pool 轮与 LLM 轮各自的启动配方：每轮独立 matrix（4 臂 / 1 臂）、独立 `FW_OUT`、smoke 独立目录、`FW_STAGE2_DEVICE` 与 `FW_MATRIX` 两个新环境变量（§7 列入范围）、两轮 driver.log 作为退出证据；汇总脚本改为多 journal 合并（§3.5、§7）。
- **B4 Accepted** — 核实 CLIP 模型逐连接懒加载，60 连接 ⇒ 最多 60 份权重。"80 GB 无压力"改为待验证：§6 定义按计划并发的实测预检、峰值阈值与降并发/延后规则。
- **B5 Accepted** — §6 增加正式跑前测试门：逐份 yaml load 验收、CLIP/LLM 各自真实 key 构造 smoke、LLM manual parity 测试记录、新 L10 库逐条对齐验收、补标定非 zscore 触发停止；`lcw_fit_weights.py` 加方法断言（§3.1、§7）。
- **N1 Accepted** — §3.2 补全命令（含 `--output`、`--cells` 用途说明）与依赖顺序（建库 → 标定 → LDA，§4）；weights.json 记录库/标定/模板身份与 d / cov / raw / w。
- **N2 Accepted** — 臂名改为 `kb_clip_vit_b_32_lda` 等按 artifact stem 命名，§5 增加 matrix 臂数与臂名唯一断言。
- **N3 Accepted** — §1 改为"固定 LDA 规则所得权重"、说明 B 池有效基线只有四种 CP1 pool、LLM 成本含 builder 内 layer-0 forward、时延另报（§3.5、§6 R4）。
- **N4 Accepted** — D1/D3 写成固定配置，D2 本轮不做并从预算移除；§3.5 写明多轮 journal 与参考臂的输入映射、每臂 500 / 每任务 50 / 配对交集 500 的验收、网格最高格为同 A 池描述性上界、多对比较标注探索性。
