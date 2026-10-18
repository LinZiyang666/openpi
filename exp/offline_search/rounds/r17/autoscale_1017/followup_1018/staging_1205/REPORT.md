# 12:05 Fast-WAM 启动失败与闭合修复

本次已完成闭合，未执行 reset、prepare 或 launch。ENABLED 保持 off。

12:05 启动在 `formal_eval.py:1347` 首次启动 MPS guard 时，命中
`formal_process.py:103` 的已有 receipt 拒绝；此检查先于 lease/Popen。
H100 新建了 `mps_control` 包装脚本目录，但没有 mps、MPS_READY.json、servers；
driver journal 不存在，owned_local/owned_remote 均为空。workers 的启动循环在
`formal_eval.py:1372` 之后，未执行。finally 却已设置 mps_ready，读取不存在的
MPS_READY.json 后写入 PARTIAL/closed=false；这不是实际的遗留进程证据。

只读核验 H100、本地、timan1、timan107 的 campaign 路径、PID/start_ticks、
进程组、argv/cwd：全部 PASS。H100 被协调者改名的旧 staging 中 MPS guard、
proxy、两个 replica 的 child/supervisor 全部死亡，receipt 均为 05:22 CLOSED。
timan1 还保留旧 workers.process.json 和 worker run；其所有已登记进程也已死亡。
没有探测 timan108，没有信号、删除或远程写入。

`prestart_closure.py` 经 CPU 测试后执行，只更新该失败 campaign 的闭合记录。
原 receipt 原字节保存在 `RECEIPT.before_prestart_closure.<sha前12位>.json`；
原错误及清理错误保留，新增 PRESTART_CLOSURE.json，当前 receipt 为
FAIL/all_owned_closed=true。计划 SHA 与 11:59 reset 记录相同：
`e3bcb40abdefa51a352a6e29ef9ec549a450847826de61dac273305da3b2119b`。

reset_zero 新增 `--remote-stage-backup`；未给时使用 --remote-backup 的同名
/tmp/r17_formal 备份目录。同时按已登记 deployment 备份各 worker 的 campaign
staging 与 run，避免下一次 native spawn/fleet mkdir 再次碰撞。所有节点先只读
检查，再逐项重新检查并 rename；保留全部内容，有活 PID/进程组则拒绝。
对已由协调者移走的 H100 staging 返回 ABSENT，不移动其 `.stale_0519_moved_1210`。
现有共享 payload、依赖、其他 campaign 和链脚本均不在归档范围内。

已对 RETRY_COMMANDS.sh 的全部 8 个节点路径执行只读预检，均 READY/ABSENT，
目标备份无碰撞。该命令文件本身未执行，由协调者在 H100 空闲边界执行。

测试详见 BEFORE_AFTER.md/json：现有 42/42 保持通过，新增 14/14 通过；
其中原生 spawn 的真实已有回执异常在 CPU 上复现，归档后可到达下一步 Popen
测试哨兵（不启动 GPU 或 worker）。每项超时 90 秒，无挂起，无新增失败。
control.py 未改，保留协调者的 SO_REUSEADDR 修改。
