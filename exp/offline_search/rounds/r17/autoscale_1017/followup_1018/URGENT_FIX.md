紧急恢复交接：全局 OFF，修复保留，等待 owner 实跑验收

1. ENABLED=off，缺失也 OFF。普通 R17/R16/R15 prepare 原生封包/人数/回收；三个指定入口的 CPU 准备已验证。owner 的 control.py SO_REUSEADDR 修复保留。没有修改或终止正在运行的 campaign/chain。
2. R17 renderer.pid 冲突：真实 driver+fleet 先复现（GENERATION_BEFORE.xml），新进程通信 worker ID 增加 UUID，slot ID 保持。driver/probe 验证未放宽。
3. Legacy 零进展 N4：真实 AgentGovernor+WorkerLoop+LiberoEpisodeRunner 先复现四个 AttributeError(task_id) 零步任务（LEGACY_RED02.xml）；不是计数从4开始，而是 client factory 的闭包被 episode_setup 覆盖。固定捕获 factory 后两代共8个 UID 正常完成。
4. 最大值提升、公平账本和两项修复均已安装在同一 OFF 开关后。BUGS_GREEN02.xml 与后续 own_all.xml 证明 CPU 行为；不是实跑许可。原 [autoscale] fixed 不再意味着允许启用动态正式队列。

以下 reset 只由 owner 执行；本轮未运行。R17 原目录整体备份，只有 FAIL、all_owned_closed=true、空 accepted journal 才继续；科学 plan 前后 SHA 必须相同。不可变 cache_inputs 延续原硬链接，不重读/复制大数组；manifest 等可变元数据与备份独立，其他复制走既有 4 MiB/s 串行 I/O 预算。

```bash
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.autoscale_1017.reset_zero --run /data/mw_finetune/eval/fa_ls_fw_r_p_m --backup /data/mw_finetune/eval/fa_ls_fw_r_p_m.before_autoscale_off_20261018 --reprepare
```

R15 一次检查并备份全部8个本地 arm 目录 fa_pi_f_u_o_{s,l}{50,100,200,500}；不删除或修改远端证据。owner-confirmed-closed-zero 是 owner 对远端均已结束且零 accepted 的明确声明，因为本地失败目录不一定收到了远端 journal；工具不能从缺失 journal 推断零进度。已发现 accepted / DONE / 活动本地 owner / 备份已存在均拒绝。

```bash
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.autoscale_1017.reset_legacy_zero --unit fa_pi_f_u_o --backup-suffix .before_autoscale_off_20261018 --owner-confirmed-closed-zero
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.formal_a_queue dispatch prepare --name fa_pi_f_u_o
```

两条新建 fa_asv_* 256集验收方案见 ../TRIAL.md（此链接相对本文件）。完整打印命令：

```bash
bash exp/offline_search/rounds/r17/autoscale_1017/trial_commands.sh
```

全局文件仍 OFF，隔离验收入口只对自己的新 payload 封装 ON。两条均通过 UID/错误/回收/关闭/内存/内核证据检查且前后回归无新增失败后，最终由 owner 决定是否开启。未执行任何真实 lane。
