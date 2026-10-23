# A1：新种子实验的结果汇总脚本

先读 exp/offline_search/rounds/r17/seeds_1023/prompts/COMMON.md（必须遵守：本机禁止任何运算，只改文件和 py_compile，不提交）和 seeds_1023/DESIGN.md。

## 目标
写 `exp/offline_search/rounds/r17/seeds_1023/results_seeds.py`（只建新文件，可在 seeds_1023/ 下加辅助模块），在新种子批次跑完后生成 `seeds_1023/RESULTS_SEEDS.{json,md}`：
- 维度：模型 × benchmark（LIBERO 两个 suite / MT50）× 步数（全步/减步）× 段长（长执行段开/关；Cosmos LIBERO 与 π0.5 MT50 只有默认段长）× 库大小 × 库种子。
- 种子：LIBERO 20261002（现有结果）、20261003、20261004；MT50 seed 8（现有）、9、10。现有种子的结果从现有汇总读：长执行段关 `longseg_off_1022/RESULTS_OFF.json`、长执行段开 `longseg_1017/RESULTS_1022.json` 与 `filter_failed_knob/q3_port_1020/RESULTS_1022.json`（π0.5/GR00T LIBERO 开）。
- 每格：两步链的最终 SR（旋钮关达标则用旋钮关，否则逐档下降到的最终档）、正式 IR（4090 profile 口径，同 results_off.py）、同卡同步数纯推理参照 SR、是否达标、链（off → 各档）。MT50 SR 先报四个难度组等权均值，再报按任务均值。
- 跨种子：每格 3 个种子的均值、最小、最大（或标准差），以及每个种子单独的数。
- 计算口径**完全复用**现有代码（`longseg_off_1022/results_off.py` 及其调用的 IR/SR 函数），不要自己重写 IR；新种子批次的发现方式按 W1/W2 的命名（W1 R17 线批次名、W2 VLA 单位名 `_x3/_x4` 之类，最终名字以它们交付为准——把「批次名 → (模型, benchmark, 步数, 段长, 种子)」做成一张显式表 `SEEDS_INDEX.json` 由协调者填，你的脚本只读这张表）。
- 并行：多进程（本机交还后协调者会在 88 线程机器上跑；默认 8 进程，可 `--workers`）。加 `--md-only` 从 json 重建 md。

## 文件归属
只建 `exp/offline_search/rounds/r17/seeds_1023/results_seeds.py`、`seeds_1023/results_lib/*`（如需要）、`seeds_1023/SEEDS_INDEX.example.json`。不改任何现有文件（需要现有函数加参数时，在交付里写明）。

## 交付
按 COMMON.md；附上用现有种子（只读现有汇总 + 现有批次）验证脚本口径的命令：现有种子那一列重算出来要和 RESULTS_OFF.json / RESULTS_1022.json 逐格一致。
