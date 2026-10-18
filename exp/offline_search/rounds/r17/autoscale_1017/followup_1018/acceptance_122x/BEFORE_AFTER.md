# 2026-10-18 12:2x acceptance 接手

| 项目 | 前 | 后 / CPU 证据 |
|---|---|---|
| 10-09 R17 worker archive | 无 worker_installs；immutable bootstrap KeyError | 新 ops-only overlay 补 worker_installs + supervisor 所需 worker_disk；真实 bootstrap 首装、复用、隔离 worker import probe PASS |
| 原任务 | 必须保留 source plan | native archived planner + 新 driver overlay 全部 256 task 字段相等（UID/init/seed/extra），源 campaign/runtime 字节不变 |
| zero_oom 起点 | 已有 counter reader，但旧实跑缺起点；无 dmesg cursor | fleet 构造、worker spawn 前持久化 boot/time/vmstat/user-slice counters + dmesg last-record cursor/hash；只在收尾比较 |
| counter 与日志冲突 | 可读日志 PASS 可能先于 counter fallback | covering counter 非零 delta 永远 FAIL；历史计数45不当作本次增加；轮转可由 counter pair 覆盖；重启 UNKNOWN |
| legacy 并发 | 依赖人工排期 | 两条显式 trial launch 前只读检查 timan107 legacy worker/driver/agent，忙或探测失败均不启动 |
| 全局 ENABLED | off | off（未写入）；普通 OFF prepares 保持原生 |

全量 top-level autoscale tests：加载既有 cpu_only 防护插件后 **207 passed**（after.xml）。首次全量命令遗漏该插件，因此仅“防护插件拒绝 production bind”自检失败；更正调用后通过，没有放松该用例。首次 bind 是无 listen 的短暂测试 socket，不是服务。

全量后进一步加入 worker_disk 闭包与真实隔离 import probe，最终针对受影响的两个测试复核 **2 passed**（12.98s）；其他源码未变化。最初针对本任务的66项亦通过。不将 CPU 替身声称为远端 staging 成功：测试里的 remote、transfer、stage transport 被禁止/替代，真实 bootstrap、不可变本地安装、源 planner 和 driver overlay 执行了真实代码；未执行任何真实 prepare/launch、tether 远端写入或 rsync。

命令在 [TRIAL.md](../../TRIAL.md)。新目录：fa_asv_r17_fw_1018b（timan1 GPU2×4，N32，H100 R2）及 fa_asv_r15_pi05_1018b（timan107×8，N4，H100 R3）。先等现有 timan107 R16 GR00T LIBERO fleet 退出，串行 R17→R15。旧 R15 256/256 实跑的 UNKNOWN 保留，不追补历史基线、不改其审计。

/data 单读者检查：所有大文件读前使用既有 wait_io；测试物理源码/归档仅数 MiB；未读 per_step，没有移动数据、改链或改运行中的 campaign。
