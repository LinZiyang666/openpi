# fa_pi_f_u_o 原生重试核查（2026-10-18）

结论：按现有队列普通 re-prepare + launch 即可，不需要 reset。全局 `ENABLED=off` 保持不变。此结论基于本地已有关闭记录和实际重试代码；本轮没有远程查询、传输、启动或修改 campaign/chain。

已有证据 `/data/mw_finetune/eval/fa_pi_f_u_o_l50/runs/chain.log:20–27`：06:05:52 driver STOPPED；06:05:53 journal 为 complete=0、success=0、rows=0；06:05:54–57 三个服务器均 STOPPED。不是仅凭 complete=0 推断 journal 为空。八个 arm 的本地 state 均没有 DONE，其他七个没有失败运行日志。

`control.py:830` 在 OFF 时不读取 LEGACY_AUTOSCALE.json；新 driver spec 不含 overlay/governor 环境变量，继续使用原生 13×3=39 workers。`legacy_release.py:68` 按新 spec 选择原生 node helper；旧 autoscale/ 目录与 recycle JSON 不参与原生调度。动态 overlay 原本单独部署，没有覆盖原生 island。

`control.py:888` 每次尝试创建新 launch_id；`node.py:228` 仅在旧 spec 与新 spec 完全相同且完成/停止时返回 ALREADY_FINISHED。旧动态 receipt 不会抑制新原生启动。如果旧会话仍存在，`node.py:222` 拒绝覆盖不同 spec。旧 .ERROR 不阻止重试；成功后原生 chain 清除对应错误标记。driver.log 继续追加，owner/job/launch receipts 按原生重试规则更新；旧 autoscale/ 证据无需删除。

CPU 回归（真实 control.chain / node.launch，所有 RPC、进程创建均 mock；每例 60 秒超时；生产端口 bind 防护开启）：

| 检查 | PATCH_BASE 原始 control.py | 当前 control.py |
|---|---:|---:|
| 旧动态 release + 旧 ERROR，第一次零进度后原生重试，39 workers，新 launch_id | 1 PASS | 1 PASS |

另有当前真实 node.launch 两例 PASS：关闭的旧动态 receipt 允许原生启动；仍存活的旧会话拒绝覆盖。两例均验证旧 autoscale/ 文件不被修改。总计 **4 passed，0 failed，0 timeout，0.35 秒**。本轮未修改任何生产代码，无需重复已完成的全套 before/after 回归。

测试文件 `test_native_retry.py`，结果 `results.xml`。全局开关仍须等 owner 的 R17、legacy 两条实跑验收全部通过后才可开启。
