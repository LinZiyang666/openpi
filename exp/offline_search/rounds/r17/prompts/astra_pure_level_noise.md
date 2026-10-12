<task>
判断 R17 逐任务旋钮测试里「已在纯推理档、仍比纯推理对照少 3–7 局」的任务是不是两次纯推理运行之间的抽样波动，结论写到 `exp/offline_search/rounds/r17/knob/PURE_LEVEL_NOISE.md`（中文）。
</task>

<facts>
- 每个任务在官方 A 集起点 0–49 上跑 50 局，与同起点的纯推理对照配对。对照是一次单独的纯推理运行，同一模型、同一套件的所有库大小共用同一份对照。
- 纯推理档每次决策都调用模型、不用缓存（IR 代理 1.0、库命中 0），与对照是同一系统；不同之处只在运行时的随机性（采样噪声、批处理、仿真时序）。
- 数据：`exp/offline_search/rounds/r17/knob/test_results_dp_r3/test.json`、`test_results_fw_r2/test.json`、`test_results_dp_red_r7/test.json`。每个 cell 的 tasks 里有 level、successes、control.successes、paired_n、missing_inits。逐局结果可以用 `exp/offline_search/rounds/r17/knob/common.py` 的加载函数从各批次目录（/data/mw_finetune/eval/knobtest*）读出来。
- 典型例子：DP 减步里 LIVING_ROOM_SCENE2 alphabet_soup 在库 50/100/200/400 四格都在纯推理档，成功局数 38/38/36/39，共用对照 43。
</facts>

<goal>
用你认为合适的方法（例如：同一任务多次纯推理档运行之间的离散程度、配对二项检验、对照偏高的概率、多重比较校正）回答：
1. 单个任务少 3–7 局，在两次同系统运行之间出现的概率有多大；
2. 上面那个四格共用一份对照的例子，更像「对照这次偏高」还是「纯推理档系统性偏低」；
3. 120 个任务 × 3 组里出现这么多个（DP 全步数 3、Fast-WAM 4、DP 减步 5）是否在预期之内。
给出数字和简短结论；方法自定，但要写明假设。
</goal>

<rules>
- 只读数据、只写 `PURE_LEVEL_NOISE.md`（可以在 `exp/offline_search/rounds/r17/knob/noise_analysis.py` 放你的分析脚本）；不改其他代码、不启动实验、不提交 git。只用 CPU。
- 不要为了让结论好看而挑数据；结论不支持「抽样波动」就照实写。
</rules>

<done>
文件写好，最后回复三句话：三个问题各自的结论。
</done>
