# Key builder × LDA 补充实验 plan（π0.5 纯 cache，A 池 500 集）

> 状态：`In Progress`（2026-09-15；G1 Round 2 **APPROVED**，见 D5；§4 Code 与准备阶段（建库 / 补标定 / LDA / yaml）完成；**G2 Round 3 APPROVED**，owner 授权复审后直接修复，见 D6 与文末 Review Log）。级别 **L1**：复用现有库/builder，新增配置生成与调整实验脚本，不改 `src/`；项目 Verify、GPU 预检与正式实跑尚未完成。
> 本轮拟实施的文件列在 §7。CLIP ViT-L/14 因需要 `src/openpi/cache/config.py` 的 clip 工厂加模型变体接线，**本轮延后**（§2 D4）。
> 上游：`docs/iclr/modality_weight_selection.md`（LDA 主方法、§4.5/§4.6 结果）、`exp/weighted_sum/RESULTS.md`（最初的 pure-cache weighted 探索：有效基线为 4 个 CP1 pool builder × 34 配置，B 池 n=100；CLIP 在该结果记录中已被移除）、`docs/cache/llm_layer_extract.md`（不区分模态的 LLM 层单 key）。

## 1. 目的

最初的探索在 B 池 100 集/格上比较过四种 CP1 pool 降维方式（mean / max / spatial_16 / spatial_64），每种都靠网格搜权重；CLIP 外部视觉模型建过库但未进入有效结果。
现在权重有了零成本闭式规则（LDA），把每种降维方式配上**固定 LDA 规则所得的权重**，在 A 池 500 集上各跑一组，得到"降维方式 × LDA 权重"的同设置对照表；
再加一个**不区分模态、对整个 prefix 直接降维的单 key** 方法，它只有一个参与打分的字段，没有权重可选，直接跑。全部纯 cache（`always_search` + `always_hit` + `top_k 1` + `write_policy never`）。
本表回答"在 LDA 规则下哪种降维方式更好"，不声称 LDA 对任意 builder 都达到该 builder 的最优（§4.6 只在 spatial_16 上验证过）。

## 2. 配置（π0.5 × {LIBERO Spatial, LIBERO-10}，每配置 500 集 = 10 任务 × 50 官方 A 池 init）

| # | 降维方式（key builder） | 库（`exp/common/data/cache_artifacts/<suite>/`） | 参与打分的字段 | 权重 | yaml_id / 臂名 |
|---|---|---|---|---|---|
| 1 | `cp1_mean_pool` | `cp1_mean_pool.pkl`（1018 / 2640） | v0, v1, rs | LDA | `kb_<suite>_cp1_mean_pool_lda` |
| 2 | `cp1_max_pool` | `cp1_max_pool.pkl` | v0, v1, rs | LDA | `kb_<suite>_cp1_max_pool_lda` |
| 3 | `cp1_spatial_pool_64`（= 2×2） | `cp1_spatial_pool_64.pkl` | v0, v1, rs | LDA | `kb_<suite>_cp1_spatial_pool_64_lda` |
| 4 | `clip`（ViT-B/32 openai，外部视觉模型；线上工厂默认变体） | `clip_vit_b_32.pkl`（v 512 维） | v0, v1, rs | LDA | `kb_<suite>_clip_vit_b_32_lda` |
| 5 | `cp1_llm_layer_extract` + `prefix_mean_pool`，layer 0（**不区分模态**：整段 prefix 过 LLM 第 0 层后均值池化成一个 2048 维 key；Pi0.5 的 state 已离散进 prefix） | spatial：`llm_layer_extract/cp1_llm_l0_prefix_mean_pool.pkl`；**libero_10 需新建**（§4 步骤 1） | 仅 `vision_0`（`robot_state` 按校验要求保持 `enabled: true, weight: 0.0`，零权重不参与检索） | 无 | `kb_<suite>_cp1_llm_l0_prefix_mean_pool` |
| 参考 | `cp1_spatial_pool_16` | 已有 | v0, v1, rs | LDA | §4.5 已跑：0.686 / 0.442；§4.6 同 A 池 1/6 网格最高格 0.688 / 0.486（描述性上界） |

两轮运行（§3.4）：**pool 轮** = 配置 1–4（`matrix_<suite>_pool.yaml`，4 臂）；**LLM 轮** = 配置 5（`matrix_<suite>_llm.yaml`，1 臂）。合计新跑 5 × 2 × 500 = **5,000 集**。

`<suite>` 实际展开为 `libero_spatial` 或 `libero_10`，因此十个 yaml_id 全局唯一。CLIP 若按 §6 的环境/显存门延期，两套件统一去掉配置 4，正式清单改为 8 臂 / 4,000 集；在发射前更新两份 pool matrix 和 `active_manifest.json`，注明延期原因，汇总只能按实际清单验收。

**裁决（已定）**

- **D1**（已定）："不区分模态直接降维" = `cp1_llm_layer_extract` + `prefix_mean_pool`，layer 0，单字段 `vision_0` 参与打分。现有库没有 prefix 的 spatial-16 版；`per_modality_spatial_pool_16` 有模态划分、需要权重，不符合"不用决定权重"。
- **D2**（本轮不做）：prefix key + robot_state 两字段 LDA 对照。`lcw_fit_weights.py` 固定三字段、3×3 协方差，两字段版要改脚本；留作后续，不进本轮矩阵与预算。
- **D3**（已定）：LLM 只取 layer 0。
- **D4**（新）：CLIP ViT-L/14 延后。`config.py:4232` 的 clip 分支只调用 `CLIPKeyBuilder(enabled_fields=...)`，`clip_key_builder.py:103` 默认 `ViT-B-32/openai`；L/14 库 vision 768 维，线上 builder 会产生 512 维 key，与库不匹配。上线需要 `key_builder` 增加模型变体字段、工厂接线并与 artifact 元数据绑定，属 src 改动（L2），另立计划。
- **D5（流程记录）**：2026-09-15 owner 授权 Review Authority 会话在 G1 Round 2 直接修订本计划正文并批准（10:20 CDT）；执行方复核其修订（字段名、建库参数、测试入口、一臂一端点的调度合同均已亲验）并只改动一处：pool 轮并发从 2 提到 4。两轮审查往来已按执行法 §3.1 在进入 §4 Code 前删除。
- **D6（本轮流程覆盖）**：2026-09-15 owner Ziyang Lin 明确指示本轮 G2 先复审，再由审查会话直接修至可通过，无需再次询问；执行方原有改动先暂存，审查方随后追加的实现、测试、计划与审查记录均留在暂存区外。本轮据此覆盖审查法的禁止直接修复与最终全部暂存规则；结论为 owner 授权修复后的验收，不将修复部分称作另一作者的独立审查。既有 Review Log 保持原文并仅追加。
## 3. 方法（每个配置）

1. **Phase-1 normalizer**：沿用 `exp/weighted_sum/data/<suite>/phase1/calibration_normalizers.json` 中该 artifact stem 的 `selected`（spatial 六种齐全；libero_10 缺 CLIP B/32 与 LLM 库、spatial 缺 LLM 库 → 用 `exp/common/calibrate_score_normalizers.py --artifact-dir <只含该 pkl 的目录> --output <新 json>` 补标定，LOEO、CPU、分钟级）。
   **兼容门**：本轮三字段 LDA 固定使用 `selected.method=zscore` 且 `params.squash=tanh`；μ/σ 必须有限、σ>0。`lcw_fit_weights.py` 加相应校验，补标定不兼容时停止该 LDA 臂，修订配方后再继续；本轮不在运行中自动改 selected 方法或增加其他 normalizer。单字段 LLM 不进入 LDA，可直接用它自身标定的 selected 单调 normalizer，并记录方法/参数。
2. **LDA 权重**（三字段库 1–4）：先用 `exp/weighted_sum/emit_yamls.build_eval_config` 按该 stem 的 selected normalizer 生成三字段模板，v0/v1/rs 全部启用且权重固定为 1/3，保证三个字段的 normalizer 均存在，再
   `PYTHONPATH=. uv run python exp/weighted_sum/analysis/lcw_fit_weights.py --library <pkl> --template <模板> --cells exp/weighted_sum/data/keybuilder_lda/diagnostic_cells/<suite>.json --output exp/weighted_sum/data/keybuilder_lda/fits/<suite>/<stem>`。
   `emit_keybuilder_lda.py` 的模板阶段先将现有 `data/grid6/summary_grid6.json` 中 `pi05_<suite>.cells` 的 `{a/b/c: sr}` 转为 `{"cells": [{"w": [a/6,b/6,c/6], "sr": sr, "n": 500}, ...]}`；原 summary 不能直接交给拟合脚本。此输入只用于既有脚本的诊断输出，不参与 LDA 解；来源为 spatial_16 网格，其 leader/SR/nearest-cell 字段不得报告为本次 builder 的结果。
   取 `fits["lda@0.05"].w`（脚本还会运行 Fisher/pairwise/动作标签变体，仅 LDA 进入配置）。校验轨迹步编号连续、每字段有限、有效正负类 query 非零，d/cov/raw/w 均有限、截断后正权重之和>0且归一化后和为1；退化时停止，不使用任意兜底权重。
   `weights.json` 逐 builder 记录：库路径与 sha256、标定 json 与 sha256、selected 方法/参数、模板路径与 sha256、`detail@0.05.fisher_lda` 的 d / cov / lda_raw、截断后权重。最终 YAML 实际参与打分的字段，其 normalizer 必须与拟合模板逐字段一致；零权重字段可按现有 emitter 省略 normalizer，拟合模板本身仍须含全部三字段。
   拟合前固定库摘要，模板解析与摘要使用同一份读取字节；计算结束复核库与模板均未变化，否则丢弃本次计算。最终生成阶段只接受匹配这两个摘要的 fit，`weights.json` 沿用 fit 的输入摘要，并在输出来源记录前再次核对当前输入。
3. **yaml**：`exp/weighted_sum/emit_keybuilder_lda.py`（新）调 `build_eval_config` 注入 LDA 权重与 h100 上的 `preload_path`，写 `exp/weighted_sum/config/keybuilder_lda/<suite>/<yaml_id>.yaml`；
   配置 5：`keys.vision_0 {enabled true, weight 1.0}`，`keys.robot_state {enabled true, weight 0.0}`，其余字段关闭，`backend.vector_dims` 保留 artifact 全字段集；使用实际键 `key_builder.extract_layer: 0`、`key_builder.type: cp1_llm_layer_extract`、`key_builder.prefix_reducer.type: prefix_mean_pool`，不使用不存在的 `llm_layer`。
   emitter 分 `templates` 与 `final` 两阶段：前者生成拟合模板与 diagnostic_cells；后者读已落盘 fits 生成正式 YAML/matrix 和 manifest，不将两阶段写成循环依赖。每 matrix 使用现有 `suite` 与 `arms: [{arm, yaml, sidecar: null}]` 格式，yaml_id 等于文件 stem。
   本机逐份 `load_cache_config` 验收结构并额外断言上述实际字段值、臂名唯一、CP1-only、top_k=1、depth=1、always_search/always_hit/write-never、权重和 normalizer 与拟合记录一致。`load_cache_config` 不证明文件存在或 artifact dims 相同：本机另读库元数据核对全字段 dims/模型变体，推送后在 h100 核对文件存在、sha256 与实际 bundle 加载。LDA 字段需 zscore(tanh)，LLM 字段按其自身 selected 验收。
4. **运行**：复用 lane P，`fw_lane_pi05.sh` 新增 `FW_STAGE2_DEVICE`（默认 `meta`，本轮只用 meta/cuda:0）和 `FW_MATRIX`（默认原 `$CFG/matrix_$SUITE.yaml`）；Stage 2 变量只传给 `serve_policy.py`。`FW_CFG_DIR` 指向 `/home/weiland/projects/openpi/exp/weighted_sum/config/keybuilder_lda`，脚本仍追加 suite；`FW_BOOT`、`FW_MATRIX` 的本次调用均传**绝对路径**，不依赖脚本 cd 后的工作目录。保持所有旧调用默认行为。
   - **pool 轮**：`FW_MATRIX=/home/weiland/projects/openpi/exp/weighted_sum/config/keybuilder_lda/<suite>/matrix_<suite>_pool.yaml`，`FW_BOOT=/home/weiland/projects/openpi/exp/weighted_sum/config/keybuilder_lda/<suite>/kb_<suite>_cp1_mean_pool_lda.yaml`，`FW_STAGE2_DEVICE=meta`，`FW_OUT=/data/openpi/weighted_sum/keybuilder_lda/pool`。五端点 23280–23284，每端点 W=12 worker，driver 23290，`FW_EVAL_CONC=4`（四臂各占一个端点同时跑，第五端点空闲；一臂一端点是既有调度合同，见下）；各臂映射由 driver 记录。若 CLIP 显存门需 W=6，**整个 pool 轮**的 driver 容量与 agent 实际 worker 数一起改为6，不宣称支持运行中单臂调车队。
   - **LLM 轮**：`FW_MATRIX=/home/weiland/projects/openpi/exp/weighted_sum/config/keybuilder_lda/<suite>/matrix_<suite>_llm.yaml`，`FW_BOOT=/home/weiland/projects/openpi/exp/weighted_sum/config/keybuilder_lda/<suite>/kb_<suite>_cp1_llm_l0_prefix_mean_pool.yaml`，`FW_STAGE2_DEVICE=cuda:0`，`FW_OUT=/data/openpi/weighted_sum/keybuilder_lda/llm`。仅起 23280 一个 server 与对应12个 worker，driver 23290，`FW_EVAL_CONC=1`。
   - **实际调度合同**：`PureCacheEvalStrategy.plan()` 每臂只建一个 stage，`assign_servers()` 每臂只返回一个端点，没有自动分片。pool 轮四臂分配给四个端点，`FW_EVAL_CONC=4` 时四臂同时跑；第五端点空闲。单 CLIP 臂最多 W 个活动客户端，单 LLM 臂最多12个；“启动60 worker”不代表某一臂使用60连接。这里不增加分片，历史吞吐仅参考，预算按新 smoke 的实际活动端点重估。
   - 顺序：pool 两 suite 串行，再 LLM 两 suite 串行；变更 suite/轮次前等待当前 driver 正常退出，再停止本轮登记的 agent/server。`launch_fleet.sh` 遇到同名 cpag session 会跳过，W 或端点变化前须关闭并重建本实验的对应 session/PID，核对实际 census；不能仅改 driver 的 `--server-workers`。共机时不使用 `stop_fleet.sh` 的全局 worker 匹配清扫。
   - smoke 单独使用 `FW_OUT=/data/openpi/weighted_sum/keybuilder_lda_smoke/<round>/<attempt>`，以及 `--smoke --trials <N> --arms <所测臂>`；正式固定50 trials。保留各 suite/轮次/重试的 driver/server 日志、代码/配置/库/A-pool 身份；driver 按 run 保存 `per_step.jsonl.launch.<run_id>.json` 与 `per_step.jsonl.workers.<run_id>.json`，固定名 launch/workers 文件保留最近一次内容供旧入口使用。
   - 每个正式 driver 须 `DRIVER_EXIT=0`，`assert_accepted_full_hit(..., require_run_id=True)` 与汇总共用 `exp/common/conductor_evidence.py`，按 `(yaml_id, task_uid, run_id, attempt)` 核对每集证据，随后检查 `journal_shortfall` 完整性。打印的 arm-level FULL_HIT rates 包含旧尝试，仅诊断；续跑是否合格以 accepted 证据为准。
5. **判读**：`lcw_ablation_summary.py` 增加可选 `--input-manifest <active_manifest.json>` 严格入口，既有 CLI 默认行为保留。manifest 按 suite 列 pool/LLM journal 与 per_step 路径、允许的 yaml_id、A-pool digest、库/配置身份；仅合并同 suite。参考输入固定为 `exp/weighted_sum/data/fusion_ablation/pi05/<suite>/journal.jsonl`，仅选 `fw_pi05_<suite>_lda`，不把同文件中的 uniform/acterr/leader 三臂一起带入。
   严格入口要求实际臂集合恰等于 manifest（含1个参考臂），不得只校验已出现的臂；正式每臂500个唯一 accepted eval task_uid、每任务50个原始 A-pool init、所有配对交集500。逐 journal 在同 run 内取最高 accepted attempt；同 uid 在两个 run 均被 accepted 即冲突，不能比较重启前后的 attempt 大小。同 attempt 终态及跨文件 winner 只允许完整记录精确重复（包括 status/error），基础设施错误不作 success=false 的正常 episode。新实验每集必须有自身 run/attempt 的非空 FULL_HIT-only 逐步记录，拒绝被 fenced 的证据；跨文件重复 trace 比较 step 编号与整条内容。新 round 的 journal/per_step/launch 必须非空，所有 accepted run 均须绑定通过 A 池检查的 launch；仅冻结参考臂可显式使用 legacy launch，并核对500集覆盖。
   输出每配置 SR + Wilson 95% 与同 init 配对符号检验；多对比较标注探索性（不做校正）。
   与最初 B 池探索（4 个 CP1 pool 的 B 池最优）只作方向对照，不并表。LLM 臂的成功率可同 init 比较；其在线代价含 builder 内 layer-0 forward（`llm_layer_key_builder.py:188` 在判定 FULL_HIT 前执行 `_extract`），时延/吞吐另报，不与 pool 臂混算。

## 4. 步骤与预算

| 步 | 内容 | 机器 | 时长 |
|---|---|---|---|
| 1 | **建库**：libero_10 的 `cp1_llm_l0_prefix_mean_pool.pkl`，原料固定为参考库同一份 H5 清单；调用已有 builder，显式 `--extract-layer 0 --prefix-reducer-type prefix_mean_pool --config-name pi05_libero --checkpoint-dir /data/openpi/checkpoints/pi05_libero_pytorch --device cuda`。**验收**：2,640条、50轨迹、10任务；唯一 `(task_key, trajectory_id, step_idx)` 集合与参考库相等，`action_chunk` 逐条 shape 相同且 `allclose(rtol=0, atol=0)`，`vector_dims == {vision_0: 2048, robot_state: 32}` | h100（PaliGemma 层 0） | 30 min 起，实测重估 |
| 2 | 补标定（步骤 1 之后）：libero_10 × clip_b32；两 suite × llm_l0_prefix_mean_pool；CLIP 三字段适用 §3.1 的 zscore(tanh) 兼容门，LLM 单字段记录并使用自身 selected | 本机 CPU | 10 min |
| 3 | LDA 权重 × 4 builder × 2 suite + weights.json；出 10 个 yaml + 4 个 matrix；逐份 load 验收（§3.3） | 本机 | 15 min |
| 4 | 推送（tether，sha 对账）；**预检门**（§6）：CLIP 显存/并发实测、LLM parity manual 测试、每种上线路径的真实 key 构造 smoke | h100 / timan108 | 40 min |
| 5 | pool 轮正式（4,000 集；CLIP 延期则3,000集）。历史200/110集每分钟仅参考，新预算按各臂实际吞吐与同时活动端点估计 | h100 + timan108 | 预检后确定 |
| 6 | LLM 轮正式（1,000 集，stage 2 在 GPU，单端点12 worker） | 同上 | 预检后确定 |
| 7 | 拉回、聚合、写表进纪要 §4.7 与运行记录 | 本机 | 15 min |

准备/建库/验收先预留约2小时，正式耗时由预检吞吐更新，不再承诺3小时内完成。各项预检计入单独预算，既不计入正式5,000集也不用于调权重。

## 5. 产物

- yaml/matrix：`exp/weighted_sum/config/keybuilder_lda/<suite>/{kb_*.yaml, matrix_<suite>_pool.yaml, matrix_<suite>_llm.yaml}`；发射前按 active_manifest 断言 pool matrix 4 臂（CLIP 延期则3）、llm matrix 1 臂、跨两 suite 的 yaml_id 全局唯一。
- 权重与来源：`exp/weighted_sum/data/keybuilder_lda/weights.json`、`fits/<suite>/<stem>/lcw_fit_weights.json`；补标定 json 与新库 sha256 记录。
- 原始数据：`exp/weighted_sum/data/keybuilder_lda/{pool,llm}/<suite>/{journal,per_step}.jsonl`（gitignored），并拉回 launch/census/各次 driver/server 日志；汇总 `summary.json`；`active_manifest.json` 固定正式臂与参考输入；`preflight.json` 记录显存采样、实际活跃连接/端点、parity 非skip结果及各项证据路径。
- 报告：`docs/iclr/modality_weight_selection.md` 新增 §4.7"降维方式 × LDA"表；`logs/fusion_weight_ablation_run.md` 追加运行事实。

## 6. 风险与预检门（正式跑前全部过）

- **B4/R1 CLIP 显存与实际连接覆盖**：保留逐连接懒加载的风险，但按 §3.4 的真实调度计数：当前单 CLIP 臂分配给一个端点，W=12 时最多12个服务客户端需要 CLIP 模型；不能把整支60-worker车队当成该臂60个已加载实例。两 suite 各做 CLIP 预检，启动与正式 pool 轮相同的五个 stage1 server，`--arms kb_<suite>_clip_vit_b_32_lda --smoke --trials 6`（60集），仅该臂所在端点的 W 个 worker 应工作。
  **通过条件**：①该端点 W 个不同 worker 的 census 均有 `results>0`；②server 的 CLIP load 日志证明相应连接都完成真实 collect/build，且在最后一个首次加载完成时其余连接仍存活（保留带时间戳的连接/加载日志），期间无重连、worker 重启或模型加载错误；③整个预检期间连续采样 `nvidia-smi` 的板卡显存并记录最大值，峰值≤60GB；④完成60集、FULL_HIT=1且检索 winner 非空。只启动进程、只有pull记录或只数server PID均不算覆盖。
  若60集不足以覆盖W个已加载且共存的连接，扩大独立 smoke 工作量后重测，不能判为通过；若 OOM/显存超限，停止并清理本实验旧进程，整个 pool 轮改 W=6（driver容量、agent数量一起改），以同样标准重测。W=6仍失败或离线 `ViT-B-32/openai` 权重/依赖不可用，则两 suite 的 CLIP 统一延期并更新正式 manifest。CLIP吞吐也从该预检记录，不沿用 pool 的旧测值。
- **B5 表示一致性与配置验收**：① 每份最终 YAML 按 §3.3 显式验收，未知键警告视为失败；② 两 suite 每种 pool/LLM 路径至少各10集真实 key 构造+检索 smoke（CLIP用上面的并发 smoke 覆盖），FULL_HIT=1、winner 非空，记录实际 query 维度；③ LLM 使用以下完整命令验证同 checkpoint 的在线/离线 mask、layer-0 hidden 与最终 key，要求测试被收集且全部执行通过，skipped=0；④ 新 L10 库先按 §4.1 逐条验收，再标定；⑤ 三字段拟合/最终 YAML 必须共享通过兼容门的 zscore(tanh)。建库 tokenizer self-check 不能替代③。

  ```bash
  PI05_CHECKPOINT_DIR=/data/openpi/checkpoints/pi05_libero_pytorch \
  PI05_CONFIG_NAME=pi05_libero \
  uv run pytest tests/cache/test_llm_layer_extract_parity.py --run-manual -m manual -v
  ```

  `conftest.py` 默认跳过 manual/env_dependent；单独 `-m manual` 只筛选，不解除skip。

- **R2 libero_10 的 LLM 库原料**：建库走"重 tokenize + tokenizer self-check"合同，原料 H5 必须是建 `cp1_spatial_pool_16.pkl` 的那一批（弄清其来源目录并记录 sha 清单）；若原料不在 h100，先从 weilandserver 拉。
- **R3 LDA 字段**：CLIP 库 vision 512 维，normalizer 用其自身 LOEO 标定；三字段库都用 (v0, v1, rs)；配置 5 不走 LDA。
- **R4 与 §4.6 网格的可比性**：pool 轮与网格同 server 设置；LLM 轮 stage 2 在 GPU，纯 cache 的 FULL_HIT 路径不经 `run_stage2`，但 builder 内跑 layer 0，成功率可比、成本不可比，表注写明。
- **R5 同机争用**：推理与模拟只用 h100 + timan108；发射前读取两节点实际资源/端口占用并登记本实验 PID/session，不能把历史“空闲”当作当前状态。weilandserver 若仍承载另一 session，原料获取限读取已存在的目标 H5，不变更其推理进程。

## 7. 拟实施文件与验证（实验逻辑在 `exp/`，无 `src/` 改动）

- 新：`exp/weighted_sum/emit_keybuilder_lda.py`（templates/final 两阶段、diagnostic_cells 转换、LDA 注入、单 key YAML、两种 matrix、active_manifest 与额外字段/身份/有限值验收）。
- 新：`exp/common/conductor_evidence.py`（driver 与严格汇总共享的 accepted 终态、run/attempt 联接及逐步证据校验）；改：`exp/ablation_study/cache_size/run_size_eval.py`（正式退出门复用该读取器，按 run 留存 launch/census，旧无 run stamp 的直接 helper 调用保持兼容）。
- 改：`exp/weighted_sum/ops/fw_lane_pi05.sh`（`FW_STAGE2_DEVICE`、`FW_MATRIX`，绝对路径存在性检查、保留每次启动日志，默认配置行为不变）；`exp/weighted_sum/analysis/lcw_ablation_summary.py`（`--input-manifest` 严格合并入口，旧CLI保持兼容）；`exp/weighted_sum/analysis/lcw_fit_weights.py`（三字段完整、normalizer方法/参数与进度/有限性/非退化断言）。
- 新：`tests/exp/test_keybuilder_lda.py`。覆盖会污染结果或使准入无效的边界：缺失整臂而其他臂完整、跨文件重复与冲突、suite混入、accepted重试与per_step对应、三字段缺字段/非zscore/退化拟合被拒、LLM真实配置字段与零权重state、CLIP延期后的准确臂集合。另用临时输入和 stub 进程检查 launcher 的两种 Stage-2/绝对路径转发与旧默认值，不在测试中启动GPU/server或结束真实进程。
- 新：`tests/exp/test_keybuilder_lda_review_fixes.py`（D6 授权下的公开回归）：终态冲突不受文件顺序影响、跨 run 的 attempt 不可排序、逐步 payload 冲突、driver 重启与证据缺失、逐 run census、拟合期间输入改变即拒绝。
- 建库产物：`exp/common/data/cache_artifacts/libero_10/llm_layer_extract/cp1_llm_l0_prefix_mean_pool.pkl`（gitignored）。
- 本轮针对性测试、相关回归、静态检查与实际本地产物验收见 G2 Round 3；项目完整 Verify 与 §6 GPU 预检仍待后续执行，正式实验结果尚不存在。写实验报告时同步 `docs/README.md` 与 `logs/README.md`。

## Review Log

### G2 Round 0 — Executor

**改动清单**（工作树，未 staged，等 owner 指示再 `git add`）：

| 文件 | 性质 | 内容 |
|---|---|---|
| `exp/weighted_sum/emit_keybuilder_lda.py` | 新（535 行） | `templates` / `final` 两阶段；`summary_grid6.json` → diagnostic_cells 转换；补标定优先、基础标定兜底的 `resolve_calibration`；三字段 zscore+tanh 兼容门；`read_fit` 拒绝换库/换模板（sha）/退化/离单纯形的拟合；`pool_arm` 断言参与打分字段的 normalizer 与拟合模板逐字段相同；`llm_arm` 用 `build_eval_config` 生成单 key 臂再注入真实键 `key_builder.extract_layer: 0` / `prefix_reducer.type: prefix_mean_pool`；`accept_yaml` 经 `load_cache_config` 回读并断言 builder / 权重 / normalizer / top_k=1 / depth=1 / task_scoped / always_search / always_hit / write never / vector_dims / preload，未知键 warning 视为失败；`library_identity` 读本地库核对 dims、builder、CLIP 变体 = ViT-B-32/openai 并记 sha256；写 `matrix_<suite>_{pool,llm}.yaml`、`weights.json`、`active_manifest.json`；`--skip-clip --skip-clip-reason` 出 3+1 臂。 |
| `exp/weighted_sum/ops/fw_lane_pi05.sh` | 改 | `FW_MATRIX`（默认 `$CFG/matrix_$SUITE.yaml`）、`FW_STAGE2_DEVICE`（默认 meta，只接受 meta / cuda:0，只传给 `serve_policy`）、CFG/MATRIX/BOOT 必须绝对路径且文件存在（在 `cd` 之前检查）、每次启动写 `server_<port>.<stamp>.log` / `driver.<stamp>.log` 不覆盖前次；`FW_ROOT` / `FW_PY`（默认原值，仅供测试桩指向临时目录）。默认调用行为不变（见测试）。 |
| `exp/weighted_sum/analysis/lcw_ablation_summary.py` | 改 | 新增 `--input-manifest` 严格入口（与旧 lane flags 互斥，旧 CLI 不变）：launch 记录绑定 suite / A 池 rollup / trials 50 / 非 smoke / 臂覆盖；`load_accepted` 末次 accepted 语义 + 单次扫描保留 run_id / attempt；run journal 出现 manifest 外的臂即失败，参考 journal 只取指定臂；accepted 行带 worker `error` 即失败；跨文件同 uid 精确重复去重、结果 / attempt / run_id 不同即失败；新臂每集 `assert_full_hit_per_episode`（与 runner 同一函数）；臂集合恰等于 manifest、每臂 500 唯一 uid、`(task, init)` 恰为 10×50、两两交集 500。round 的 journal / per_step / launch 可为列表（续跑到另一目录时），per_step 跨文件同 (uid, attempt) 轨迹不同即失败。 |
| `exp/weighted_sum/analysis/lcw_fit_weights.py` | 改 | `check_template`（三字段齐全、per_field、zscore+tanh、μ 有限 σ>0）、`check_progress`（每轨迹 step 0..len-1）、key 有限性、`fisher_and_lda` / `pairwise_logistic` 空总体不再崩、`safe_normalize` / `degenerate`：主拟合 `lda@0.05` 退化直接 SystemExit 不写兜底，其它变体记 `degenerate: true`；输出记录模板路径 + sha256 + 逐字段 normalizer。 |
| `tests/exp/test_keybuilder_lda.py` | 新（35 用例） | 覆盖 plan §7 列出的边界（见文件头）。launcher 用 PATH 桩（tmux / pkill / ss / 解释器）跑真实脚本，不起 GPU / server、不杀真实进程。 |
| `logs/fusion_weight_ablation_run.md` | 改 | §7.1 准备阶段事实（建库 / 验收 / 补标定 / LDA 权重表 / yaml 验收 / 机器现状）。 |
| `logs/README.md` | 改（未 staged，另一 session 的行同在） | 本 plan 行状态。 |

**计划一致性声明**：代码按已批准的 plan §3 / §4 / §7 实现，另有以下可见偏差，请裁定：

1. `fw_lane_pi05.sh` 新增 `FW_ROOT` / `FW_PY` 两个默认值不变的覆盖项（plan 未列）。用途只有一个：让 §7 要求的"临时输入 + stub 进程"测试能把脚本指向临时目录而不碰 `/home/weiland/projects/openpi`。
2. 日志文件名从固定 `driver.log` / `server_<port>.log` 改为带时间戳（plan §3.4 "不用后一次启动覆盖前次证据"）；拉回时按 stamp 取。
3. `--input-manifest` 的 round 输入允许列表（plan 写单路径）。单路径仍是默认写法，`emit` 产出的 manifest 也是单路径；列表只服务于"续跑落到另一 `FW_OUT`"的情形，跨文件规则按 plan §3.5 实现。
4. plan §4.1 的建库验收用了一次性脚本（`$CLAUDE_JOB_DIR/tmp/accept_llm_library.py`，判据与输出原样记入运行记录 §7.1），未入库；若审查方要求可重复脚本入 `exp/weighted_sum/`，我加。
5. 原料传输：h100 上没有 libero_10 的 H5，按 plan R2 从 weilandserver 拉；用了临时只读 rsync daemon（:23190，用后已停），没有改动另一 session 的任何进程。

**已完成的准备阶段**（plan §4 步 1–3，证据在 `logs/fusion_weight_ablation_run.md` §7.1）：libero_10 LLM 库建成并通过 §4.1 全部验收；三份补标定全 zscore；8 个 LDA 拟合非退化；10 份 yaml + 4 份 matrix + `weights.json` + `active_manifest.json` 生成并逐份回读验收；参考臂真实 journal / launch 过严格入口检查。

**本地测试（执行法 §4 允许的自检，不替代 §6 Verify）**：

```
uv run pytest tests/exp/test_keybuilder_lda.py -q   → 35 passed in 19.27s
bash -n exp/weighted_sum/ops/fw_lane_pi05.sh        → OK
PYTHONPATH=. uv run python exp/weighted_sum/analysis/lcw_ablation_summary.py --pi05-journal libero_spatial=<四臂 journal> --out …
   → uniform 0.666 / lda 0.686 / acterr 0.610 / leader 0.676（与 §4.5 记录一致，旧 CLI 未变）
```

**未做（按 plan 顺序在 G2 之后）**：推送到 h100 并核对 sha / bundle 加载；plan §6 预检门（CLIP 显存与连接覆盖、LLM parity manual 测试、各路径 10 集 smoke、h100 资源普查——h100 现被另一 session 的 4 个 GR00T server 占 30 GB / util 96%）；pool 轮与 LLM 轮正式跑；汇总与纪要 §4.7。

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-09-15 11:26 CDT

**范围与流程**：Review Authority，L1，审查本计划 §7 的工作树实现；基线 HEAD `a27a707a78d3bb3c55149661350aef389eb18633`。已完整读取 polished plan、G2 Round 0 执行方说明、四个实现文件及其 diff、新测试文件、运行记录与相关上游接口。G1 的直接修订授权按 D5 留在该轮；本轮仅审查、运行独立测试并追加本记录，没有修改实现。未发现本轮执行方的独立流程违例；以下为实现与交付问题。

**G2 checklist**：

| 项 | 裁决 | 依据 |
|---|---|---|
| 与批准计划一致 | NEEDS REVISION | 两阶段 emitter、三字段模板、LLM 真实配置、CLIP B/32 限制及两轮 matrix 已实现；§3.5 的 accepted 来源/正式 eval 限制、拟合身份链仍有缺口（B1/B2/B4/B5）。 |
| 测试覆盖与通过 | NEEDS REVISION | 执行方 35 项独立复跑通过；另 128 项相关回归通过。独立反例验证了现有测试未覆盖的来源混用、空 launch、非 eval、拟合身份与脚本入口问题。 |
| 文档与索引同步 | NEEDS REVISION | 运行记录与 logs 索引已更新到 Code 完成；计划中的 LLM 臂名/启动路径与实际产物不一致，按文档启动必然失败（B6）。最终结果文档尚未到交付时点。 |
| 无回归 | NEEDS REVISION | 相关数值/配置/调度回归通过，但旧 summary 脚本入口新增 `PYTHONPATH` 依赖，已对 HEAD 做前后复现（B3）。 |

**Blocking findings**：

- [Blocking] [Concern] **B1 / P1 — FULL_HIT 证据没有绑定 accepted 的 run 身份，跨文件也不是精确记录去重。** `exp/weighted_sum/analysis/lcw_ablation_summary.py:190` 与 `:234` 经共享 helper 只保留 `(task_uid, attempt)` 和 hit_type 列表，丢弃 per-step 的 `run_id`、`accepted` 及用于跨文件比较的 `step_idx`。独立反例：journal 的 500 集均属于新 run/attempt 1，per-step 全来自旧 run/attempt 1，仍返回 500 集 FULL_HIT；把 per-step 全改为 `accepted=false` 或删去 run_id 也能过。两个文件的 step 0 与 step 99 只要都是 FULL_HIT，也被当成同一条 trace 去重。要求新实验保留并匹配 `(yaml_id, task_uid, run_id, attempt)`，只使用该 accepted 运行的证据，按实际 step 身份/内容判定精确重复或冲突；journal 的 winner 与来源也须从同一条记录取得，不能另一次按较短 key 覆盖元数据。 — reasoning: `src/openpi/conductor/driver.py:180` 明确说明重启后 attempt 从 1 重新计数，并已给 journal/per-step 同时盖 run_id；目前仅在输出中列 run_ids，无法防止旧 trace 为没有证据的新运行背书，破坏 §3.5 的纯 cache 验收。

- [Blocking] [Concern] **B2 / P1 — 拟合结果没有绑定拟合时的库，且缺失模板身份会直接跳过校验。** `exp/weighted_sum/analysis/lcw_fit_weights.py:229` 只记录 library 路径；`exp/weighted_sum/emit_keybuilder_lda.py:238` 只比较路径，`:240` 允许整个 template 记录缺失。实际八个 fit 文件均无库 sha256。独立反例中同路径替换库后仍可读取旧 fit，删除 template 记录也仍可通过；最终 `weights.json` 写的是当前库/模板的摘要，会把旧权重重新标成新输入的产物。要求 producer 记录实际拟合输入的库与模板内容摘要，supplement reader 强制存在并核对；不接受无身份的旧 fit，修订后重新生成有可追溯输入身份的拟合产物。 — reasoning: 同路径重建库、重新标定或失败后残留旧 fit 都是这条两阶段运行链的正常操作风险；最终阶段重新算一个当前摘要不能证明已有权重由它算出，违反 §3.2 与上游“离线侧与闭环侧同库同标定”的约束。

- [Blocking] [Concern] **B3 / P2 — 旧 summary 脚本调用方式已经回归。** `exp/weighted_sum/analysis/lcw_ablation_summary.py:40` 新增顶层 `exp.*` 导入。在仓库根目录执行 `env -u PYTHONPATH uv run --no-sync python exp/weighted_sum/analysis/lcw_ablation_summary.py --help` 即报 `ModuleNotFoundError: No module named 'exp'`，所有旧 lane flags 也在解析前失败。同环境下 HEAD 版本的脚本 `--help` 返回 0。要求保留原脚本入口可执行性，并增加移除 PYTHONPATH 的真实子进程 CLI 回归；不能仅靠 pytest 的 `pythonpath = ["."]` 或给旧验收命令临时补 `PYTHONPATH=.` 宣称兼容。 — reasoning: 文件自身 Usage 仍公布直接脚本调用，§3.5/§7 明确承诺旧 CLI 行为保留；当前测试通过掩盖了入口层面的实际失败。

- [Blocking] [Concern] **B4 / P2 — 启动证据校验可被空列表绕过，续跑输入也没有逐来源绑定。** `exp/weighted_sum/analysis/lcw_ablation_summary.py:224` 接受 `launch: []`，独立反例中完全没有对应 launch 仍完成 500 集汇总；多个 journal/per_step/launch 是三组独立列表，合法 launch 不需要对应实际 journal 的 run。要求拒绝空的必需输入，为每份新运行证据保留可核对的 launch/run 关联，缺失或错配即失败；历史参考臂可保留其明确冻结的旧来源合同。同步落实 §3.4 每次 launch/census 的留存：当前仅 server/driver 文本日志带 stamp，`run_size_eval.py:637` 仍覆盖同一 `per_step.jsonl.launch.json`。 — reasoning: 对任意一张合法启动记录检查 A 池摘要，不能证明另一份 journal 用的是该池/正式配置；列表支持和原目录续跑必须保持逐次来源证据，不能把“没有证据”当作校验通过。

- [Blocking] [Concern] **B5 / P2 — 500 集覆盖没有检查 eval phase。** `exp/weighted_sum/analysis/lcw_ablation_summary.py:150` 读取全部 accepted 终态，`:176` 只看 task/episode 两个整数。将一臂所有 journal/per-step 的 uid 从 `:eval:` 改为 `:warmup:`、phase 改为 warmup，仍可通过 500 集覆盖并输出成功率。要求严格入口检查 uid 的 arm/phase 与行内身份一致，只接受正式 eval 终态；非 eval 行按明确规则排除或拒绝，并在排除后重新验足500集。 — reasoning: §3.5 要求的是500个唯一 accepted **eval** task_uid，整数范围相同不证明属于正式评测；不能让预热/其它阶段记录补足分母。

- [Blocking] [Concern] **B6 / P2 — LLM 的文档启动路径不存在。** 计划 §2 配置5和 §3.4 的 FW_BOOT 使用 `kb_<suite>_llm_l0_prefix_mean_pool`，而 `exp/weighted_sum/emit_keybuilder_lda.py:116`、两份实际 matrix、生成 YAML、测试和运行记录都使用 `kb_<suite>_cp1_llm_l0_prefix_mean_pool`。两个 suite 按计划给出的 boot 路径在本机都不存在，launcher 的新存在性检查会退出2。要求统一计划中的臂名/命令与 emitter 产物，并用实际生成的 LLM matrix/boot 跑一次 stub 入口验收。 — reasoning: 名称差异已经导致计划的可执行命令失败，现有 launcher 测试使用自行造的 `boot.yaml`，未覆盖文档和生成器的衔接。

**Non-blocking 与偏差裁定**：

- [Non-blocking] [Suggestion] **N1 — reader 可进一步复核 raw 与 w 的关系。** `read_fit` 对 finite raw 全零/全负但手工填合法 uniform w 的文件仍会接受。建议共享 producer 的非退化判据并验证 w 等于 raw 截断归一化、两处 raw 相同，给出明确错误。 — reasoning: 独立反例说明 reader 的自称验收范围更宽；当前 producer 已阻止这种主拟合输出，因此不把这两个手工不一致输入单列成阻塞项，也不声称已有八个拟合退化。
- [Non-blocking] [Suggestion] **N2 — 在严格汇总产物中显式标记多重比较为探索性、未校正。** — reasoning: §3.5 已约定该口径；当前 `pairs` 只提供 p_sign，补一个清楚的说明可避免下游把全部配对当作确认性结论。

执行方 Round 0 的偏差1（FW_ROOT/FW_PY 测试接口）接受，默认值及真实脚本 stub 测试支持其用途；偏差2（带 stamp 的文本日志）接受其方向，剩余留存缺口见 B4；偏差3（列表续跑）接受其需求，合并实现须修 B1/B4。偏差4不强制新增一次性建库验收脚本，本轮已经独立重验实际 L10 LLM 库；偏差5传输事实按运行记录保留，本轮未远程重演传输或检查已结束的 daemon。D5 中 pool 并发4与四臂各一端点相容；资源可用性仍以 §6 预检为准。

**独立验证证据**：

- `UV_CACHE_DIR=/tmp/openpi-review-uv uv run --no-sync pytest tests/exp/test_keybuilder_lda.py -q`：**35 passed**，19.37s。
- 相关回归：`tests/exp/test_weighted_sum_libero10.py`、`tests/weighted_sum/test_weight_search_strategy.py`、`tests/ablation_study/cache_size/test_cache_size_runner.py`、`tests/ablation_study/cache_size/test_cache_size_analysis.py`、`tests/conductor/test_journal.py`：**128 passed**，2.16s。
- 独立测试仅留 `tests/review_tests/`，已核对 `.gitignore` 排除、不入 index。12 个探针分两次执行：合法对照1通过，11个拒绝/兼容性断言失败（包括 N1 的两个非阻塞反例）；失败均为错误输入未拒绝或 B3 的明确入口错误，非环境缺件导致测试未执行。
- `bash -n exp/weighted_sum/ops/fw_lane_pi05.sh` 与本任务文件的 `git diff --check` 通过。
- 本机实际10份 YAML 经 `load_cache_config` 与 emitter 逐字段断言回读通过，摘要与 active_manifest 一致。两个历史 reference 的真实 launch/A池摘要、仅 lda 选择和500集覆盖通过。八个已有 fit 的 primary raw 均为正。
- 实际 L10 LLM 库独立验收：**2640条 / 50轨迹 / 10任务**，唯一 `(task_key, trajectory_id, step_idx)` 集合与 spatial_16 参考库完全相同，action_chunk 逐条 shape 相同且 `allclose(rtol=0, atol=0)`，vision_0 为有限2048维，artifact dims 与 layer0/prefix_mean_pool 元数据正确。完整 sha256：`93d3d35d9e750d800aa46a87d6c6882be36e7ccb183191ff5fda97bac24bd59b`。
- GPU 预检、LLM manual parity、远端 bundle 与正式 rollout 尚未执行；它们按顺序留给后续 Verify/运行，不因尚未执行本身否决 G2，也不记作已通过。

**裁决**：NEEDS REVISION。关闭 B1–B6 后重交 G2；N1–N2 为建议项。按审查法 §5 将本次相关实现、执行方测试、运行记录、计划及 logs 索引本任务行暂存为下一轮比较基线；其它 session 的修改不纳入。

定位补注：B6 的 `yaml_id` 定义位于 `exp/weighted_sum/emit_keybuilder_lda.py:107`，上文该文件的 `:116` 行号以此为准。

### G2 Round 2 — Executor

逐项回应 Round 1（B1–B6 全部 Accepted，N1–N2 Accepted）。修订后 `tests/exp/test_keybuilder_lda.py` 48 passed（含 13 个新反例），审查方列的相关回归 173 passed；八个拟合已按 B2 重新生成（权重逐位不变）。

- **Accepted — B1**：严格入口不再经过共享 helper 的 `(task_uid, attempt)` 键。`scan_journal` 单次扫描保留每条终态行的 `run_id / attempt / error / phase`；`merge_source` 在同一条记录上选 winner（最高 accepted attempt；同 attempt 的两条 accepted 行只有结果与 run_id 都相同才算重复，否则报不一致），跨文件同 uid 用完整 `Winner(arm, uid, task, init, success, attempt, run_id)` 比较；`scan_per_step` 按 `(yaml_id, task_uid, run_id, attempt)` 收集 `{step_idx: hit_type}`，缺 attempt / run_id / step_idx 的行直接拒绝，`accepted: false` 的行不作证据；`witness_full_hit` 只取 winner 自己 run/attempt 的轨迹，别的 run/attempt 只计入 `stale_rows_ignored`，未被 accepted 的 uid 出现在 per-step 即失败；跨文件同 `(uid, run, attempt)` 逐 step 比对，step 集合或 hit 不同即冲突。反例（审查方三例 + step 0 vs 99）全部进测试并被拒。
- **Accepted — B2**：`lcw_fit_weights.py` 输出新增 `library_sha256`（按内容），模板已有 sha256；`read_fit` 强制两者存在并逐一核对，缺任一身份或不匹配即拒绝（"refit with the current lcw_fit_weights.py"）。八个 fit 已用新版本重算，`weights.json` 里的库 / 模板摘要现在是拟合输入的摘要而非最终阶段重算的当前值（两者相等才会通过）。同路径换库、删 template 记录、旧格式 fit 三个反例进测试。
- **Accepted — B3**：`lcw_ablation_summary.py` 去掉顶层 `exp.*` 导入（严格入口改为自含实现，见 B1），脚本按路径直接可跑；`emit_keybuilder_lda.py` 顶部按 `__file__` 把仓库根加入 `sys.path`。新增子进程回归 `test_scripts_run_by_path_without_pythonpath`：移除 `PYTHONPATH` 后两个脚本 `--help` 退出 0，并用合成 journal 跑通旧 lane flags 全流程。人工复现：`env -u PYTHONPATH uv run --no-sync python exp/weighted_sum/analysis/lcw_ablation_summary.py --pi05-journal …` 输出四臂 0.666 / 0.686 / 0.610 / 0.676。文档中的 `PYTHONPATH=.` 前缀已删。
- **Accepted — B4**：(a) `exp/ablation_study/cache_size/run_size_eval.py` 新增 `write_launch_records`：launch 记录在 driver 构造后写，带 `run_id`（`ConductorDriver.run_id` 正是为此暴露的），同时写固定名 `per_step.jsonl.launch.json`（保留旧合同，续跑会覆盖）和每 run 一份的 `per_step.jsonl.launch.<run_id>.json`（续跑不覆盖）——这是 plan §7 清单之外的一处 `exp/` 改动，只加不改旧行为，单测 `test_driver_launch_records_carry_run_id_and_survive_relaunch`。(b) 汇总侧：round 的 journal / launch / per_step 任一为空即失败（`launch: []` 不再是"无需检查"）；launch 可列表给出或经 `launch_glob` 按 driver 的每 run 命名发现，发现为空同样失败；run source（`launch_binding: run_id`，缺省值）每份 launch 必须带 `run_id`，所有 accepted winner 的 `run_id` 必须 ⊆ 已检查 launch 的 `run_id`，否则报 "no checked launch record"；参考臂在 manifest 中显式声明 `launch_binding: legacy`（附说明），run source 声明 legacy 直接拒绝。emitter 生成的 manifest 已按此写（round 给 `launch_glob`，参考给 legacy）。反例（空 launch、无 run_id、未绑定 run、run source 冒用 legacy）进测试。
- **Accepted — B5**：`scan_journal` 要求 `task_uid` 解析出的 arm / phase 与行内 `yaml_id` / `phase` 一致，且 phase 必须为 `eval`；本设计没有 warm-up，非 eval 终态按"拒绝"处理（提示该记录不可计入），而不是静默排除后再凑数。审查方两个反例（整臂 uid 改 warmup、行 phase 改 warmup）与 uid/yaml_id 错配反例进测试。
- **Accepted — B6**：统一为 emitter 的命名 `kb_<suite>_cp1_llm_l0_prefix_mean_pool`（yaml_id = `kb_<suite>_<stem>`，与库 stem、manifest、weights.json 一致），修订 plan §2 配置 5 与 §3.4 LLM 轮的 `FW_BOOT` 两处正文（G2 阶段改 plan 正文只此两处、只改名字，其余不动；Review Log 未动）。新增 `test_launcher_runs_the_emitted_llm_round`：用 emitter 自己产出的 `matrix_<suite>_llm.yaml` 与 boot yaml 跑 stub launcher；另按审查要求对本机实际生成的两个 suite 的 LLM matrix / boot 各跑一次 stub 入口（rc 0，`--cache-config` / `--arm-matrix` / `--stage2-device cuda:0` 均为实际产物路径）。
- **Accepted — N1**：`read_fit` 现在验证 `w == normalize(clip(raw))`（容差 1e-9）、`lda@0.05.raw == detail@0.05.fisher_lda.lda_raw`、clip 后有质量；手工填 uniform w + 全负 raw、raw 与 detail 不一致两个反例进参数化测试。
- **Accepted — N2**：`--input-manifest` 输出的 `manifest.pairs_note` 明确写明全部配对符号检验为探索性、未做多重比较校正。

**其它**：`logs/fusion_weight_ablation_run.md` §7.1 补注拟合重算与 launch 绑定合同；`logs/README.md` 行状态更新。审查方 Round 1 对偏差 1–5 的裁定照单接受，无异议。本轮改动全部在工作树（未 staged）。


### G2 Round 3 — Reviewer — APPROVED — 2026-09-15 12:27 CDT

**范围与授权**：Review Authority；目标仍为本计划 §7 的 G2 复审，HEAD `a27a707a78d3bb3c55149661350aef389eb18633`。完整复核 Round 2 回复与其相对上轮暂存基线的改动；执行方未拒绝 B1–B6/N1–N2。owner 本轮明确指示“这轮审查后直接修改到你觉得可以通过的地步”“把原本的暂存，把自己的修改留在暂存区外”，按 D6 先完成评估再直接修复。本结论明确包含 owner 授权下的审查方实现，不称作修复部分的独立作者审查；未发现此授权范围外的流程违例。

**复审发现与修复**：执行方的 48 项测试通过；独立反例进一步暴露原修改仍有以下遗漏，均已直接修复并留下公开回归。

- **B1 关闭**：新增 `exp/common/conductor_evidence.py`，汇总与正式 driver 退出门共用 `(arm, uid, run_id, attempt)` 联接。原 driver 仍用短 key，合法重启会被旧 step 判为重复，旧 run 的 FULL_HIT 也可能代替新 run；现在只取 accepted run 的证据，新 driver 显式要求 run stamps。终态记录保留完整内容，同 attempt 的 error/status/其它内容不同即冲突；不同 run 对同 uid 的 accepted 记录不以 attempt 大小排序。跨文件 trace 比较 step 编号与完整行，entry_id 不同也拒绝。完全相同的跨文件记录允许去重；参考文件先按选定臂过滤，再校验所选记录。旧无 run stamps 的直接 helper 调用仍使用历史兼容入口。
- **B2 关闭**：执行方已补齐并重新生成八份 fit 的库/模板 sha256；复审补上生产端时序：库在加载前记录摘要，模板从同一份字节解析与计算摘要，拟合结束再次校验两份输入，变化即拒绝落盘。final 阶段验证当前输入，来源记录使用 fit 内的原始摘要，并在写来源前再次核对，不能把拟合后的新库重新标为旧解输入。变更 library/template 的计算中反例均被拒绝。
- **B3 关闭**：汇总抽取共享模块后用 `__file__` 定位仓库根，直接路径入口仍无需 `PYTHONPATH`；现有两个脚本的子进程 `--help`、旧 lane flags 全流程回归通过。
- **B4 关闭**：执行方已修复空输入、缺 run launch、来源错配与新 round 冒用 legacy；launch 按 run 留存。复审补齐 `per_step.jsonl.workers.<run_id>.json`，连续两次 driver 的 census 均保留，固定名仍指向最新记录。manifest 的新 round 只接受被 launch 逐 run 绑定的 accepted 证据；历史参考保留显式 legacy 合同。
- **B5/B6 关闭**：eval phase 与 uid/yaml_id 一致性反例全部被拒；两 suite 的 LLM boot 命名、正式 YAML 与 matrix 路径一致，launcher stub 回归通过。
- **N1/N2 关闭**：raw/detail 一致、截断后正质量、`w = normalize(clip(raw))` 已校验；汇总显式注明所有配对检验为探索性且未作多重比较校正。

**G2 checklist**：

- **与已批准计划一致 — PASS**：十个新臂、两种运行矩阵、三字段 LDA 与 layer-0 prefix 单 key 配方不变；补齐 plan 原有 run 身份与逐次证据留存合同。共享读取器和 driver 的实际修改已补入 §7；无 `src/` 修改。
- **测试覆盖与通过 — PASS**：针对错配证据、续跑、冲突终态、输入身份等实际失效路径补了 17 项公开回归。相关套件去重后 **458 passed，5 skipped**；5 项为既有 cache-size 高精度统计 manual 参数化测试，未改动、未声称通过。独立审查测试 **18 passed**。
- **文档与索引 — PASS**：本计划正文、运行交接 §7.2、`logs/README.md` 本任务行已同步代码与未运行项；此前 G2 Review Log 逐字保留，仅追加本轮。正式结果与论文表未更新，因为新实验尚未运行。
- **回归与接口 — PASS（本轮证据范围）**：cache-size 全部默认测试、weighted-sum 相关测试及 conductor 默认测试完成；旧 helper/CLI、启动默认值、配置加载、生产 journal/per-step 格式保持兼容。静态检查与 shell 语法检查通过；项目全量 Verify 不由本次相关回归替代。

**验证证据**：

1. 执行方提交版本：`tests/exp/test_keybuilder_lda.py` **48 passed**。修复后的独立审查 + 原 48 项 + cache-size runner：**105 passed**；新增公开回归单独运行 **17 passed**。
2. 相关套件命令：`UV_CACHE_DIR=/tmp/openpi-review-uv uv run --no-sync pytest tests/exp/test_keybuilder_lda.py tests/exp/test_keybuilder_lda_review_fixes.py tests/exp/test_weighted_sum_libero10.py tests/weighted_sum/test_weight_search_strategy.py tests/ablation_study/cache_size tests/conductor -q --tb=short`。沙箱内 **451 passed / 7 failed / 5 skipped**，7 项均被本地 socket 的 `PermissionError: Operation not permitted` 阻止；自动审批允许沙箱外复跑 `tests/conductor/test_census.py tests/conductor/test_integration.py tests/conductor/test_protocol.py`，**15 passed**，覆盖全部 7 项。合计独立测试用例 **458 passed / 5 manual skipped**，无剩余失败。
3. `ruff check --select F,E9` 覆盖五个相关 Python 实现与新增公开测试，通过；两个新文件还通过完整 `ruff check`。`bash -n exp/weighted_sum/ops/fw_lane_pi05.sh` 与 `git diff --check` 通过。
4. 对实际磁盘产物读验：八份 fit 全部通过当前 `read_fit` 的库/模板摘要与非退化检查；从 d/cov 重新求解 `solve(cov + 1e-6 I, d)`，raw 与归一化权重在 `rtol=atol=1e-12` 内一致。十份库/YAML 与权重来源摘要一致，十份 YAML 经生产 loader 与配方断言通过，四份 matrix 的臂名及文件路径一致。
5. 两套件实际 reference launch 的 A 池合同与 500 集覆盖通过；真实生产 per-step 通过新共享读取器：Spatial **500 集 / 14,448 FULL_HIT 步**，L10 **500 集 / 40,131 FULL_HIT 步**，两者 `stale_rows_ignored=0`。这是历史参考证据，不能记作新十臂的正式结果。
6. **未执行**：项目完整 Verify、远端 bundle/sha 核对、§6 GPU 显存与连接预检、LLM manual parity、正式 5,000 集及最终统计。它们仍按计划执行，不记为已过。

**暂存分界**：开始修复前已将执行方本任务八个文件及 `logs/README.md` 的本任务行暂存；快照在 `/tmp/keybuilder_g2_original_v4ukzbd4/`。修复后核对整个 index 的 `git ls-files --stage` 与 `index_entries.txt` 完全一致；本轮审查方实现、17 项公开测试、正文与本记录均未暂存。其它 session 的工作树改动保留；`tests/review_tests/` 保持 ignored 且不进入 index。未 commit/push。

**裁决：APPROVED（D6 owner 授权直接修复后的 G2 验收）。** B1–B6 与 N1–N2 已关闭，无剩余 G2 阻塞项。
