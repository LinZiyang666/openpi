Slot4 两项故障已 CPU 复现并修复，ENABLED 保持 off。未执行 continuation/reset/prepare/launch，没有杀进程，没有修改运行链或 tmp/t107_mem_guard.sh。

- `exp/r17_integration/ops/library_salvage.py:46`：闭合核验读取节点上既有 process receipt 文件；命令只传路径及原 PID/start_ticks/pgid pin，每批引用限制 8 KiB。完整 argv/env 不再进 tether argv，不需要传输文件或写远端。读取的身份必须与源回执吻合；存活 leader、group member 或 campaign argv 仍拒绝。真实 fa_ls_gr_r_p_m receipt（2.34 MB）在隔离节点文件 fixture 中复现原 Linux E2BIG，修复后所有节点通过相同闭合代码；另测 150 文件分批、身份变化拒绝和 cmdline 不可读时仍拒绝已证明的进程组。
- `autoscale_1017/reset_remote.py`：逐项捕获不可读 identity/cmdline/cwd，跳过其他 UID；cmdline 与 cwd 分别核对，一个不可读不会掩盖另一个的 run 归属。能读且 PID/start_ticks 一致的 receipt 继续阻止 reset；跳过计数写入备份回执。仅测试目录执行 rename，未接触真实 run。
- `kernel_evidence.py`：timan107 普通 dmesg 可读，原 `--time-format raw` 返回 `unknown time format: raw`。现优先普通 dmesg，用 boot-relative 时间限定 lane，记录读取错误和最早/最晚保留时间；不能覆盖旧窗口时仍为 UNKNOWN，不能用后来 OOM 污染早先窗口，也不能把缺日志当作零 OOM。

只读重审 `fa_asv_r15_pi05_1018` 的结果在 `r15_dmesg_final/`：256/256 accepted，0 errors，60 回收代，其余检查通过；zero_oom 仍未能证明，audit FAIL。记录窗口为 Chicago 09:06:38–09:29:17；此次 dmesg 最早保留约 10:37:34，晚于验收窗口。没有启动 counter，当前 oom_kill=45，无法追补 delta。source 明确记录 dmesg、coverage=false，而不是声称无权限。

Coordinator 报告：10:54 timan107 上 H100 R15 LIBERO 39 workers 与 4090 R16 LIBERO 64 workers 重叠，真实 global OOM，oom_kill 37→45，MemAvailable=0；已停止 R15，并启动低于 30 GiB 时退役 R17 的 guard。本轮只读查询确认当前计数 45、dmesg 可见 global_oom；没有改动 guard、运行数量或开关。普通 OFF prepare 仍为原生行为。

所有精确重试命令见 RECOVERY_COMMANDS.md。原 939 个已接受 UID 保留，GR 续跑剩余 1561；FW 的本地/H100 整目录备份仍包含 mps_control。先前失败留下的 retry_build 暂存目录保留不删。

原始复现 `red.xml` 为 8 failed（含真实 E2BIG 和 PermissionError）；修复对应 `green.xml` 为 8 passed；扩展 kernel/正式辅助测试 `proof.xml` 为 27 passed。完整前后结果见 BEFORE_AFTER.json/md。第一次基线有一个 180 秒超时，栈在等待生产 fa_ls_fw_f_b_l 持有的 /tmp/r17_longseg_data_reader.lock；没有干预生产锁。释放后该同源用例重跑通过（before_retry.xml），原始超时日志保留。
