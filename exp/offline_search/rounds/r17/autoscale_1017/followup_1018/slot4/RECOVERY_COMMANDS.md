本轮没有执行以下命令。保持 ENABLED=off；不要改运行链或 tmp/t107_mem_guard.sh。每条成功后再执行下一条。

项目根目录 `/home/weiland/projects/openpi`：

```bash
export PYTHONPATH=.:src
# GR：保留 939，补跑 1561；读取原节点 receipt 文件并核对所有权。
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.resume --source /data/mw_finetune/eval/fa_ls_gr_r_p_m --name fa_ls_gr_r_p_m_c1 --remaining --verify-closed
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.stage --run /data/mw_finetune/eval/fa_ls_gr_r_p_m_c1
.venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.launch --run /data/mw_finetune/eval/fa_ls_gr_r_p_m_c1

# FW：原失败目录（本地和 H100）均整体改名备份，包含 mps_control；不删除。
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python -m exp.offline_search.rounds.r17.autoscale_1017.reset_zero --run /data/mw_finetune/eval/fa_ls_fw_r_p_m --backup /data/mw_finetune/eval/fa_ls_fw_r_p_m.before_native_retry_0930 --remote-backup /data/mw_finetune/eval/fa_ls_fw_r_p_m.before_native_retry_0930 --reprepare
.venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.launch --run /data/mw_finetune/eval/fa_ls_fw_r_p_m
```

本轮只读检查时 c1 与所选本地备份名均不存在，上次两项失败没有完成 reset/build。先前留下的 `*.retry_build_*` 暂存目录未删除，本次使用新 UUID 构建。任何真正匹配 run argv/cwd 或 PID/start_ticks 的存活进程仍会使工具拒绝继续；receipt 身份变化/缺失也不伪装闭合。
