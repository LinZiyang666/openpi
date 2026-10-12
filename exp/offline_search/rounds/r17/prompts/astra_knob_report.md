<task>
为 R17 第 5 层「逐任务 IR 旋钮」写一份结果报告初稿，写到 `exp/offline_search/rounds/r17/knob/REPORT.md`（中文）。
已完成三组：DP 全步数、Fast-WAM 全步数、DP 减步（DDIM 10 步）。Cosmos 全步数、Fast-WAM 减步、Cosmos 减步还在跑，报告里给它们留好位置、写「进行中」。
</task>

<inputs>
- 设计与经过：`logs/session_handoff.md` 开头几段（10-11 13:0x 起的旋钮任务说明，10-12 各增量）。
- 测试结果（最终一轮）：`exp/offline_search/rounds/r17/knob/test_results_dp_r3/`、`test_results_fw_r2/`、`test_results_dp_red_r7/`，各有 TEST_A.md 和 test.json；更早的轮次 `test_results_*_r1..` 记录了每轮提档了哪些任务。
- 开发集下降：`dev_results_dp/`、`dev_results_fw/`、`dev_results_dp_red/`（DESCENT.md、levels.json）。
- 正式 IR：`exp/offline_search/rounds/r17/knob/ir_formal/IR_FORMAL.md`、`ir_formal.json`（计算方法见 `ir_formal.py` 文档串）。
- 延迟实测：`exp/r17_integration/profile/README.md` 末尾的 Results 一节。
</inputs>

<content>
1. 一段话说清做了什么：完整组（纯缓存 + 长执行段 + 修正器 + 停滞门）上按任务选调用频率档位；开发集下降规则；A 集测试与提档规则（每任务 50 局，最多少 2 局算达标；少 5 局及以上提到纯推理，少 3–4 局升一档；只重测提档任务，反复到全部达标）；减步版的定义和对照。
2. 每组一张表：库大小 × 套件，列出成功率、对照成功率、差值、正式 IR（服务端现状、文本每局一次两种口径）、档位分布（只用停滞门 / 中间 / 纯推理）。
3. 每组提档经过：第几轮提了几个任务。
4. 已在纯推理档仍偏低的任务：列出来，说明纯推理档每次决策都调模型、不用缓存（IR 代理 1.0、库命中 0），与对照是同一系统；统计判断由另一个任务给出，这里引用 `PURE_LEVEL_NOISE.md`（如果还没写好就写「待补」）。
5. 延迟实测系数表与方法偏离（Fast-WAM 的 CPU 张量、Cosmos 的 Transformer Engine）。
6. 需要 owner 决定的事：正式 IR 用哪种文本编码口径。
</content>

<rules>
- 只读数据、只写 `REPORT.md` 这一个文件；不改代码、不启动任何实验、不提交 git。只用 CPU。
- 所有数字从上面的文件里取，不要凭印象；表格里不用缩写，任务套件写「空间任务 / 长任务」。
- 写法：直接、具体，像给导师看的实验报告；不写版本号和修订记录。
</rules>

<done>
`REPORT.md` 写好，最后回复一句话说明写了哪些节、还有哪些「待补」。
</done>
