本次 R17 OFF 修复已在工作树，`ENABLED=off`。未启动、重置、结束任何 campaign 或改写运行链。09:40 优先修复阶段没有远端调用；较早验收证据阶段仅按 owner 授权对 timan107 做两次只读查询。

根因是原生打包之外的两个入口。`filter_failed/timan1_gpu2.py:30` 的 authorize 不看开关就写 MT50 N32；`:235` 的 bootstrap 直接读取当前工作树文本注入 fleet。失败 campaign 的 campaign.json 为 N0、autoscale=null，timan1.json 却为 N32（LIVE_SOURCE_PROOF.json）。原生 worker payload 没有动态代际身份，32 局后以同一个 slot ID 重启，PID 921639→952777，于是严格 validator 正确拒绝。

修复：GPU2 的 authorize/route 仅在全局 ON 且 campaign 动态时采用 N32/1800 秒；OFF 恢复 N0、原人数容量检查。bootstrap 通过 `switch.source` 取原生全文；driver hook、worker hook、capacity probe 通过 `switch.function_source` 取原生函数。R16 GPU2 bootstrap 同样修正。`worker_resilience.allocate` OFF 分支恢复补丁前的请求数量及缺额分配算法。endpoint、模型、runtime、GPU 与 probe 校验没有放宽。

实际路径证明在 `tests/test_off_r17_native.py`：真实 MetaWorldConductorDriver、原生 WorkerLoop、formal_workers.worker_command、worker_lifetime.limited_loop；修复前第 33 次 pull 抛出相同 ValueError（red.xml），修复后单代完成 40 个不同 UID，且主动改变 PID 仍被拒绝。新增七项覆盖 by-value 注入、native placement、OFF capacity、所有 worker 模块 SHA，以及同一真实 frozen runtime 输入下 driver_runtime.tar 的前后逐字节相等。本目录最终全集 174 项通过（own_final.xml）。

SOURCE_POLICY.json 核对 19 个补丁原文件：18 个 OFF 字节与 PATCH_BASE.json 相同；control.py 的唯一保留差异为 owner 的 SO_REUSEADDR 修复。原生 archive 之外的模型、任务和其他已冻结旧版本保持来源原字节，不将它们误称为当前 checkout 版本。

既有回归逐例表见 BEFORE_AFTER.md/JSON：原版 474 passed / 23 failed，当前 483 passed / 14 failed；无新增失败、缺失用例或 timeout。包含 filter_failed、vla worker/shared、R17/core conductor、ops、LIBERO recycle 和 h100/control。当前 14 个非通过项为 12 个两侧相同的既有失败，加两个试图绑定生产端口而被 CPU 插件阻止的测试；不冒充全仓通过。每例上限 180 秒。原版通过 19 文件 SHA 验证备份只读加载，未覆盖工作树。

续跑只读选择结果：fa_ls_gr_r_p_m 的 2500 个 UID 分成 939 retained 和 1561 missing，无交集；原 receipt 尚未证明 owner 全部关闭，必须 `--remaining --verify-closed` 创建新 continuation。原 campaign 不 reset，不修改 journal。FW zero-progress reset 新增 `--remote-backup`，检查存活进程及大小写 receipt identity 后改名保存整个 H100 run 目录（含 mps_control），保留本地完整备份，绝不删除。完整 owner 命令在 RECOVERY_COMMANDS.md。

较早验收修复也已通过 CPU：旧 10-09 FW 来源缺 runtime_identity，通过原 runtime.tar 的成员哈希建立内容 identity，服务端 archive 字节和 256 个任务均不变；现有未封存验收 lane 直接重试 prepare。kernel collector 新增启动前/结束后普通用户 counters；无可比基线仍 UNKNOWN。fa_asv_r15_pi05_1018 的新审计在 `../acceptance_fix/r15_reaudit/`，256/256、0 errors、60 回收代均通过，但历史 oom_kill=15 没有启动基线，zero_oom 不能证明。全局开关仍不得开启。
