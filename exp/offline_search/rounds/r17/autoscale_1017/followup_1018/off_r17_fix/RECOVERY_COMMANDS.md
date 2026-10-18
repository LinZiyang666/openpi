以下仅供 coordinator 在队列边界执行；本轮没有执行任何 reset、prepare 或 launch。保持 `autoscale_1017/ENABLED=off`。

在 `/home/weiland/projects/openpi` 执行。939 条结果走新 continuation，绝不能零进度重置。第一条命令只读核对旧进程已关闭；若发现存活 owner，会拒绝继续。原 FAIL receipt、journal、计划全部保留。已经存在同名 continuation 时不要重跑 build。

```bash
export PYTHONPATH=.:src
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.resume --source /data/mw_finetune/eval/fa_ls_gr_r_p_m --name fa_ls_gr_r_p_m_c1 --remaining --verify-closed
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.stage --run /data/mw_finetune/eval/fa_ls_gr_r_p_m_c1
.venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.launch --run /data/mw_finetune/eval/fa_ls_gr_r_p_m_c1
```

上面的准备命令只在上一条成功后继续。当前只读选择证明：2500 = 939 retained + 1561 missing；保留完整 UID/init/seed。准备时在原进程关闭后重新核对集合。

FW 的 09:30 尝试在 driver 启动前失败，0 accepted。下面同时保留本地和 H100 上的完整失败目录；远端 `mps_control` 随整个目录改名保存，没有删除。两边备份使用相同路径文字，但位于不同机器。任何仍存活的 campaign 进程/receipt identity、已存在的备份目录都会使 reset 拒绝。

```bash
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python -m exp.offline_search.rounds.r17.autoscale_1017.reset_zero --run /data/mw_finetune/eval/fa_ls_fw_r_p_m --backup /data/mw_finetune/eval/fa_ls_fw_r_p_m.before_native_retry_0930 --remote-backup /data/mw_finetune/eval/fa_ls_fw_r_p_m.before_native_retry_0930 --reprepare
.venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.launch --run /data/mw_finetune/eval/fa_ls_fw_r_p_m
```

只有 reset 成功且输出 `RESET_PREPARED` 才执行 launch。不会改动任何正在运行的 chain 脚本。

旧 FW 来源的 R17 动态验收 lane 已修复缺失 runtime_identity 的兼容性；它复用同一 frozen runtime.tar，核对成员哈希后建立新内容 pin，科学字段不变。此前失败的 prepare 没有 PREPARED/LAUNCH，可直接重试（无需重新 build）：

```bash
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python -m exp.offline_search.rounds.r17.autoscale_1017.trial_lane prepare --run /data/mw_finetune/eval/fa_asv_r17_fw_1018
.venv/bin/python -m exp.offline_search.rounds.r17.autoscale_1017.trial_lane launch --run /data/mw_finetune/eval/fa_asv_r17_fw_1018
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python -m exp.offline_search.rounds.r17.autoscale_1017.trial_lane audit --run /data/mw_finetune/eval/fa_asv_r17_fw_1018
```

验收入口仅对指定 `fa_asv_*` 的独立进程使用动态代码，不写全局开关；普通队列保持 native。R15 旧验收缺启动前 OOM counter，事后 oom_kill=15 无法推出窗口 delta=0；重审仍 UNKNOWN，不能据此打开全局开关。
