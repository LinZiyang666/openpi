2026-10-18 R16 生命周期测试误传输取证

漏 mock 的调用为 `formal_eval.transfer`，发生于 `test_coordinator_owned_lifecycle_and_lease_without_launches` 的 autoscale release staging。`control.remote`、模型启动/停止和 driver 启动均已 mock，因此未执行远端解包或启动。

| 项目 | 核实结果 |
|---|---|
| 节点 | timan107；未联系 H100 |
| 唯一远端遗留文件 | `/tmp/openpi_autoscale_e9f9bbe5134a2fa64b1366ba3c904f8860dadb57fc777a1177a56241a1f97e7d.tar` |
| 文件大小 | 409,600 bytes；网络协议总字节没有记录 |
| SHA256 | `94d4c3086976beb0c5214a31bcee44c9872ca46b74625cd23bec83630690e2a9` |
| 本地 tar mtime／远端保留 mtime | 2026-10-18 00:31:41.277641851 -0500 |
| 远端 ctime | 2026-10-18 00:31:42.569994496 -0500 |
| 本地完成回执 mtime | 2026-10-18 00:31:43.082643 -0500 |

实际传输动作包括远端 `mkdir -p /tmp` 和 rsync 下载。计划解包的 `/scratch/zixuans8/openpi_autoscale_releases/e9f9bbe5134a2fa64b1366ba3c904f8860dadb57fc777a1177a56241a1f97e7d` 与测试用 `/scratch/zixuans8/openpi_r16_conductor_synthetic` 均不存在。代码调用链和定点 stat 证据支持：没有替换 R16/R15 科学岛、运行或排队 lane 的冻结 payload，也没有接触 H100。没有传输前的 stat，不能证明同名 `/tmp` tar 原先不存在；不能把 ctime 说成精确传输开始时间。

还必须记录本地副作用：direct_transfer 短暂开启了端口 23118 的 rsync daemon，打开 `/tmp/r17_direct_rsync.lock`，随后该函数按自己的配置/PID 清理 daemon 和临时导出目录。这违反了原先禁止 expose 的边界，不能只描述为一次无启动的文件复制。未启动模型、worker 或 campaign。后续 CPU 测试统一使用阻止远端/GPU/传输命令的插件，并补全此生命周期测试的 mock。

本次只进行两次获准的只读远端取证，原文为 `remote_stat_01.txt` 与 `remote_ls_02.txt`；没有删除文件。完整字段、包内文件清单及本地 tar 路径见 `INCIDENT.json`。该远端 tar 留给 coordinator 决定处理。

本地测试目录遗留文件及共享传输锁逐项列于 `LOCAL_FILES.json`，未删除。
