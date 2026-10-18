| 套件 | 修复前 通过/失败 | 修复后 通过/失败 | 新增失败 |
|---|---:|---:|---:|
| autoscale tests | 174/0 | 186/0 | 0 |
| test_resume | 9/0 | 9/0 | 0 |
| test_timan1_gpu2 | 1/1 | 1/1 | 0 |
| test_h100_argv_batch | 11/5 | 11/5 | 0 |
| test_library_salvage | 22/0 | 22/0 | 0 |
| test_local_library_salvage | 8/0 | 8/0 | 0 |

逐例无新增失败、缺失用例；最终复核无超时。原始 before.xml 含一次生产读取锁竞争超时，原代码在 before_retry.xml 重跑通过，未干预锁。原始 after.xml 中两项新测试因测试进程 import 后源码最后修改导致 inspect.getsource 行偏移错位；停止修改后新解释器在 after_retry.xml 复核全部新用例和受影响 salvage/resume 套件。所有原始报告保留，verified XML 只合并真实复核结果，不手改测试状态。

每例 timeout=180s；CPU 插件禁止真实远端/GPU/传输及 23100–23199 绑定。6 个剩余失败两边相同。
