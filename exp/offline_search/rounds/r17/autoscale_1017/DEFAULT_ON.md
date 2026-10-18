当前状态：全局 OFF；下表为通过真实 lane 验收之后的候选默认（2026-10-18）

ENABLED=off。所有普通新 prepare 使用原生 worker 数、原生 payload、原生回收，不冻结 autoscale overlay。不要用旧报告中的“默认开启”作为启动依据。

| benchmark | timan1 GPU2 | timan107 | local |
|---|---:|---:|---:|
| LIBERO |83|104|83|
| MetaWorld |332|440|332|

上表仅在全局 ON 时生效，覆盖旧手调小人数；更大的配置保留。小硬上限必须带 worker_hard_limits.maximum/reason。当前正式清单无此技术例外。R15/R16 保留节点选择流程，若只选择 timan107，期望最大值为104。timan108 排除。

| 首批队列项 | 路径 | 当前 OFF | 候选 ON |
|---|---|---|---|
| fa_co_f_f_oc1 | R17 continuation | native fixed | 保持 fixed，源冻结 pins |
| fa_ls_fw_r_p_m | R17 stage | 原生配置 | MetaWorld 332/440 |
| fa_ls_gr_r_p_m | R17 stage | 原生 20/20 | MetaWorld 332/440 |
| fa_pi_r_f_l | R16 | 原生64 | 所选 timan107 104 |
| fa_gr_r_u_l | R16 | 原生64 | 所选 timan107 104 |
| fa_ls_dp_r_b_l | R17 stage | 原生24/24 | LIBERO 83/104 |
| fa_ls_dp_r_u_m | R17 stage | 原生60/60 | MetaWorld 332/440 |

CPU 测试显式隔离 ON 候选，不修改全局文件。7 项 dry run 读取真实元数据和计划，输出重定向到交付目录，传输/install/probe 均替换。OFF 另验证 fa_pi_f_u_o（R15：每 arm 39）、fa_pi_r_f_l（R16：64）、fa_ls_gr_r_p_m（R17：20/20），以及原生封包源码哈希。它们不等于真实 prepare 或 GPU 验收。

最新故障恢复、复现、reset 命令见 followup_1018/URGENT_FIX.md；两条 owner 执行的 256 集真实 lane 命令见 TRIAL.md。两份验收 PASS 之前禁止打开全局开关。prepared/失败运行不能仅改模式字段；保留证据重置并核对科学计划。

sol检查器新增 `--local-only`，不改变默认检查行为。该选项对本机模型仍执行冻结解释器CPU校验；H100仅复检本地plan/UID、bootstrap pin和语法、不可变输入inode，并将既有远端CPU回执标为cached，不冒充fresh。`--out`同时重定向逐项回执，保留sol原始证据。执行命令：

```bash
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.longseg_1017.check --local-only --out exp/offline_search/rounds/r17/autoscale_1017/longseg_check_1018/CPU_PROOF.json
```

前后回归采用 baseline_sources 的只读19文件覆盖层：每份备份先匹配 PATCH_BASE.json；Python导入及源文件读取均返回备份字节，__file__ 与绝对 source pins 保留原路径，其余完整当前依赖不变。子 Python 进程继承 sitecustomize 覆盖层。早期物理镜像结果仅用于排查，不用作最终表。cpu_only 阻止实际远端/GPU命令，逐例定时并记录超时栈；最终 ops 两边均使用单线程 BLAS 和180秒上限，其余套件逐例60秒。H100的真实 rsync daemon 用例两边均排除，因为本轮禁止 expose。


本轮最终结果：147个autoscale测试、27个longseg单元测试通过；60项local-only检查（33 fresh、27 PASS_LOCAL）。交接与reset命令见 [HANDOFF_OFF.md](HANDOFF_OFF.md)，全局仍OFF。

最新完整前后表写入 followup_1018/off_final/BEFORE_AFTER.md，按用例对照，不将既有失败或端口保护阻止项算通过。当前这轮所有套件逐例上限180秒；以表内实测超时为准。
