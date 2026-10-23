# 共同规矩（四个实现任务都适用）

- 仓库 /home/weiland/projects/openpi（main 分支）。先读 exp/offline_search/rounds/r17/seeds_1023/DESIGN.md（要做什么、静默出错点、拔/留的防御）和 exp/r17_integration/ops/RUN_CONFIGS.md 相关节。
- ⛔ 本机（weilandserver）被别的实验占满：**不准在本机做任何运算**。只准读文件、改文件、对改过的文件跑 `python -m py_compile`。不跑 pytest、不拟合、不解析大文件、不起多进程、不连远程机器、不用 tether。需要运行验证的地方，写进交付里的「协调者要跑的命令」（命令 + 估计时间/内存 + 预期结果）。
- 不提交 git（协调者审完 diff 后提交）。不动在跑批次的目录（/data/mw_finetune/eval/ 下已有的 fa_*、ffk1015_* 等）、冻结运行时（/data/mw_finetune/qualification/formal_runtime_*）、链脚本（openpi_ext/scratch/r17_ops/ff/*.run.sh）、exp/cluster_sim/*。
- 只改自己任务书里归属给你的文件；新代码尽量放新模块 exp/offline_search/rounds/r17/seeds_1023/ 下。需要改别人归属的文件时，不要改，在交付里写明要什么改动。
- 保持简单：用现成函数，别过度实现。代码风格、注释密度、命名跟周围代码一致。
- 路径上不影响数字的冗余防御（DESIGN.md「拔的防御」）顺手拔掉；「保留」一栏的不准拔。DESIGN.md「会静默出错的地方」凡是落在你的路径上，必须改成带种子的路径或缺了就报错。
- 交付（最后的回答）：改了哪些文件（每个一两句为什么）、新文件、协调者要跑的命令（按顺序，标 CPU/GPU、机器、估计时长）、第一次运行前必须做的回归检查、没解决的问题。中文、简洁、带 file:line。
