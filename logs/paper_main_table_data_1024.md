# 论文主表：首个种子的数据位置（看守会话，10-24；23:xx 更新：IR 已按只计 GPU 时间重算）

根目录 R = /home/weiland/projects/openpi/exp/offline_search/rounds/r17。首个种子 = LIBERO 库种子 20261002、MT50 seed 8。

## 1. 缓存系统终档（成功过滤库 + 修正器 + Q3 停滞门 + 旋钮下降）

- **长执行段关**（默认段长 C5），14 条线 116 格：`R/longseg_off_1022/RESULTS_OFF.{json,md}`（10-24 定稿 2cb73e94）。
  - 覆盖：DP、FW 的 LIBERO 与 MT50；GR00T MT50；π0.5、GR00T LIBERO；全步 / 减步。
  - `rows[].off` 的字段：sr、n、successes、group_mean / task_mean（MT50）、level_ir、ir（正式 IR）、tuned、per_task_levels、levels。
  - 同格纯推理：pure_sr、pure_n、pure_ir、pure_group_mean、pure_task_mean。
- **长执行段开**（C10），以及 Cosmos（没有长执行段，记为 default）：
  - DP、FW 的 LIBERO 与 MT50，GR00T MT50，Cosmos LIBERO：`R/longseg_1017/RESULTS_1022.{json,md}`，108 行。
    - 每行由 model / benchmark / mode / suite / size 定位，有 `filtered`（成功过滤库，主表用这一臂）和 `original`（原库）两臂。
    - 每臂字段：sr、n、successes、decision_heads、level_ir、ir（正式）、latency_ir、cost_n、controls、no_knob、measured_levels、tasks。
    - 行内另有 `native_pure_target`：同步数 C5 纯推理，是选档目标。
    - 行内另有 `pure_long`：同步数 C10 纯推理，即长执行段开的纯推理参照。
  - π0.5、GR00T LIBERO 的长执行段开：`R/filter_failed_knob/q3_port_1020/RESULTS_1022.json`，64 行。
    - 用 arm 区分库。
    - final 是终档的 SR / IR，pure 是纯推理参照，另有 per_task_levels、tuned、measured_levels。
    - 口径见顶层 ir_definition。
  - π0.5 MT50（没有长执行段，只在 4090 上跑）：`R/knob/mt50/pi05_{full,reduced}_filtered/descent.json`。
    - `cells[]` 的字段：size、group_mean、task_mean、ir、control（即纯推理）。
- 网页用的是同一份数据：`/home/weiland/projects/openpi_ext/scratch/r17_ops/livepage/live_1023/` 下的 knob_1022.json、knob_vla_1022.json、mt50_knob_1022.json、onoff_1023.json、mt50_onoff_1023.json。它们由 build_1022.py 和 build_onoff.py 从上面几份生成，取数口径可以照这两个脚本对照。
- **命中率**：
  - R17 各模型：level_ir = 调用模型的段头数 / 全部段头数（尾段不计）。所以调用数 = level_ir × decision_heads，命中数 = decision_heads − 调用数。
  - π0.5、GR00T：level_ir = 账本里的模型调用数 / 有视觉输入的决策数。

## 2. 纯推理参照

- **长执行段关**（C5，同卡、同步数）：
  - R17 各模型：`R/longseg_off_1022/REFERENCES.json`，键为「模型:benchmark:步数:卡」，值为 run 名。
  - π0.5、GR00T LIBERO：`fa_lo_*_p5_q`（R15）。
  - Cosmos：formal_a.py 里的 PURE。
  - π0.5 MT50：4090 上的 tpi7_0536（全步）和 tpi7k2_1135（减步）。
- **长执行段开**（C10）：
  - R17 各线：RESULTS_1022 每行的 `pure_long`。对应 run 见 `R/longseg_1017/REFERENCE_SELECTION.json` 的 `:long` 键；最终续跑链的覆盖关系见 `R/longseg_1017/FINAL_CHAINS_1022.json`。
  - π0.5、GR00T LIBERO：见 formal_a_queue.references；结果文件里对应 q3 RESULTS_1022 的 pure。
- 上面几份汇总已经带了同格纯推理的 sr / n / ir，一般直接取用即可。

## 3. 分段延迟价格表（10-24 起只计 GPU 时间）

- **口径改了（owner 10-24 批准，b214ee61）**：R17 计时工具原来把 CPU 端的 prepare（输入缩放、归一化）和 host 派发间隙算进各段。现在每段只算 GPU 时间，即该阶段的 CUDA graph 加上同一时间窗里非 graph 的 kernel（取区间并集）。CPU 预处理和 host 空隙两边都不计，和 D5、和文本编码「CPU 上的不计」一致。没有重新测，由现有 nsys 记录算出：`exp/r17_integration/profile/gpu_only.py`。
- **价格表**：`exp/r17_integration/profile/` 和 `R/longseg_1017/profiles/4090/` 下的 `results_gpu/`、`results_mt50_gpu/`、`results_cached_text_gpu/`。每个 json 的 `gpu_only.harness_ms` 记了旧值；旧的 `results*/` 不再用于计价。
- **计价**：`R/longseg_1017/analyze.py` 的 `prices()` → `R/knob/ir_formal.py`。公式不变：IR = (v·(调用 + 命中) + m·调用)·C0 / 控制步数，以全步纯推理一次决策为 1；LIBERO 两个套件共用 libero_10 的计时。DP 不再单独扣 CPU 文本编码（GPU 时间本来就不含它）。
- **π0.5、GR00T LIBERO**：仍用 R15/R16 的 D5 账本（本来就是纯 CUDA graph 时间），数字不变。π0.5 MT50 的价格在 descent.json 里，已按新口径重算。
- 主要变化（ms，旧 → 新）：π0.5 MT50 视觉 38.7 → 9.0；GR00T MT50 视觉 15.4 → 3.1；Cosmos 视觉 106.5 → 88.0；DP 全步 100 步去噪 165 → 104.6；Fast-WAM 视觉约 7.5–9.8 → 4.0–6.7。

## 4. TODO、作废与注意

- 以上全是首个种子。新种子（LIBERO 20261003/4、MT50 9/10）还在跑，跑完由 `R/seeds_1023/results_seeds.py` 出 RESULTS_SEEDS.*，现在不要用。
- RESULTS_1022.json 顶层有 dev_outcomes_used、withdrawn_outcomes_used 两项，填表前先看是否为空。q3 那份有 withdrawn_outcomes_used、load_errors，同样先看。
- FW MT50 全步旋钮批次的回执已封为 PASS（10-24 00:32），RESULTS_OFF.md 已同步。
- 库大小：LIBERO 上 DP、FW、Cosmos 是 50–500，π0.5、GR00T 只有 50/100/200/500；MT50 是 250–2500。
- 10-24 起，已封批次的 driver/per_step.jsonl 原地压成了 .zst，读取走 exp/offline_search/zst_logs.py，结果不变。重跑上面的汇总脚本没有影响；如果要直接读原始 per_step，注意文件名多了 .zst。

## 5. 10-24 23:xx：IR 已按新口径重出

- 重出的文件（同名覆盖，旧版本在旁边 `*.before_gpuonly_1024`）：`R/longseg_1017/RESULTS_1022.{json,md}`、`R/longseg_off_1022/RESULTS_OFF.{json,md}`、`R/knob/mt50/pi05_{full,reduced}{,_filtered}/{descent.json,test.json,DESCENT.md}`、`R/filter_failed_knob/results/pi05_mt50_full/dev/`，以及网页数据 `live_1023/{knob_1022,mt50_knob_1022,onoff_1023,mt50_onoff_1023}.json`。
- 只有 IR 字段变了（ir、latency_ir、pure_ir 及网页的 ir_u/ir_f/pure_ir*）。SR、n、successes、选档、决策计数逐值不变。`q3_port_1020/RESULTS_1022.json`（π0.5/GR00T LIBERO，D5）不变。
- 新旧 IR 对照表：`/nvme/weiland/openpi/gpuonly_1024/`（ir_table.py 的输出；比对脚本 compare.py）。
- ⛔ 论文侧要重跑：`openpi_ext/scratch/mlsys_paper_ops/fill_main_table.py` → `main_table_cells.json` → `appendix_main_tables.tex`（22:01–22:03 用的是旧 IR）；其中算延迟的 UNIT 也要改读 `*_gpu` 档案，并去掉 DP 的 CPU 文本扣减。
