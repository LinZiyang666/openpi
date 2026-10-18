| 套件 | 补丁前通过/失败 | 当前通过/失败 | 新增失败 |
|---|---:|---:|---:|
| h100 (含 control.py) | 216/11 | 225/2 | 0 |
| filter_failed | 43/6 | 43/6 | 0 |
| filter_failed_knob workers/shared | 1/1 | 1/1 | 0 |
| LIBERO recycle | 3/0 | 3/0 | 0 |
| R17 conductor | 70/3 | 70/3 | 0 |
| R17 ops test_ops | 14/2 | 14/2 | 0 |
| core conductor | 127/0 | 127/0 | 0 |

原始 19 文件以 SHA 核对备份作只读 overlay；其他依赖相同。每例 timeout=180 秒；无超时、无缺失用例。测试插件阻止真实远端/GPU/transfer 及 23100–23199 绑定。h100 的两个剩余失败都是约束性阻止：真实 rsync daemon 测试试图绑定 23180，fake-process lifecycle 测试试图绑定 23100；两侧均在 bind 前拒绝。另有 12 个两侧相同的既有失败，不能称为全部套件通过。

本目录已有测试：本轮验收修复前的五个模块只读 overlay 为 **147 passed / 0 failed**，当前 **174 passed / 0 failed**（新增 27 项），逐例新增失败/缺失/超时均为 0。before/after 均使用相同的隔离夹具；夹具补齐来源的 local-worker 回执，并以原 FFK_DEFINITION 资源配置重放首次 prepare，避免读取事故后的 failover 人数作为初始输入。
