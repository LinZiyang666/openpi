# π0.5 H100 启动失败与临时传输修复（2026-10-18）

**根因已确定并完成CPU修复：四个 Hugging Face 下载 metadata 的时间戳不同，导致 frozen adapter 的整个 checkpoint 内容身份不匹配。不是缺失 inputs/bundle，也不是 GPU 算术错误。** 未启动GPU模型、worker或正式campaign；共享checkpoint、frozen runtime、lane、chain均未修改，无sudo。

## 1. 原因与证据

两次失败目录为H100 `/home/exouser/xgpu_1017/results/h100_pi05_concurrent_20261018T155736Z`、`…155753Z`。各自 `replica_0.log:20` 附近最终异常是 `ValueError: immutable checkpoint identity mismatch`，调用 frozen `exp/r17_integration/models/pi05/adapter.py:375–377`。该检查发生在 `create_trained_policy` 导入与权重加载前；adapter `:330–349` 会递归hash checkpoint下**所有文件**，包括下载metadata。源协调员日志：`/home/weiland/projects/openpi_ext/scratch/r17_ops/ff/slot4_xgpu_pi05_1.log:1`。

| 相对 `/data/ckpt/pi05_metaworld_rlinf/` 的文件 | H100与封存副本差异 |
|---|---|
| `.cache/huggingface/download/.gitattributes.metadata` | 仅第3行时间戳 |
| `.cache/huggingface/download/lerobot/metaworld_mt50/norm_stats.json.metadata` | 仅第3行时间戳 |
| `.cache/huggingface/download/metadata.pt.metadata` | 仅第3行时间戳 |
| `.cache/huggingface/download/model.safetensors.metadata` | 仅第3行时间戳 |

四者前两行commit/blob身份相同，其余小文件和tokenizer SHA相同；权重随后经限速CPU全量hash确认匹配 `6e877ae8…ca7221`。证据：[失败与清理收据](evidence/pi05_h100_failure_1018.json:1)、[metadata逐行差异](evidence/pi05_metadata_difference_1018.json:1)、封存 `concurrent/bundles/pi05/assets/model_manifest.json:81`。两次失败均 `daemon_closed=true`，replica_0分别exit1，检查时29600系列无监听；没有停止其他MPS或服务。

## 2. 已完成的修复与CPU验证

H100专用目录：`/data/mw_finetune/xgpu_1017/pi05_ckpt_4008289bc944`。在同一文件系统硬链接7,473,091,464-byte权重，其他15个小文件复制封存内容；不修改、chmod或覆盖共享权重/metadata，不用逃出目录的symlink，不复制大权重。全目录逻辑大小7,473,099,921 bytes，新增文件内容仅 **8,457 bytes**（不含目录/收据开销）。精确checkpoint身份恢复为 `4008289bc944015dbf3eba6cea0a701f6ef9dbb53e0bfd47dea7b55585d8b62e`，与4090 PIN相同。

[完整checkpoint收据](evidence/temporary_rsync_pi05_20261018T160936Z_14bca0/PI05_CHECKPOINT_VIEW.json:1)记录每文件SHA、inode/device/size/mtime及硬链接证明；收据放在view外，不进入原adapter的递归hash。run入口先核对收据和文件状态，再显式添加checkpoint路径relocation，原身份检查保持。缺view即在MPS/子进程启动前拒绝。

CPU预检PASS：3份输入内容身份、cache依赖前后SHA、冻结runtime release及相关代码、tokenizer、Transformers replacement检查、loader依赖导入。H100环境为Torch2.7.1、Transformers4.53.2、NumPy1.26.4、sentencepiece0.2.1。见 [预检](evidence/temporary_rsync_pi05_20261018T160936Z_14bca0/PI05_PREFLIGHT.json:1) 与[修正命令CPU计划](evidence/pi05_h100_corrected_CPU_plan_1018.jsonl:1)。所有检查禁用CUDA可见设备，不加载模型。

老asset manifest里的adapter source digest早于实际冻结runtime；两卡该runtime文件完全相同，最终预检以 frozen `RUNTIME_RELEASE.json` 为依据。两者差别记录在[代码核对](evidence/pi05_runtime_code_audit_1018.json:1)，没有为消除该差别编辑runtime或asset PIN。

CLI新增checkpoint relocation并纳入新的执行harness快照，**下一次π0.5双卡使用本次同版本工具**。4090此前没有本研究π0.5成功轮次；旧FW/DP结果的不可变harness快照未变。

## 3. 临时只读rsync脚本

`temporary_h100_stage.py`支持 `pi05`、`fw` 两种模式；不带`--run`只生成allowlist/端口检查计划。实际运行时：

1. 用本机及H100的`ss -H -lntu`和活跃`/proc`命令所引用的campaign配置取并集，排除所有已监听端口、每个活跃campaign的整个`ports`列表（即便未监听），以及23177和正式传输用23118。大campaign仅读前后各128KiB，不能读出完整ports则拒绝。
2. 从公网已转发23100…23199中挑选空闲端口；可显式`--port`，同样接受完整拒绝检查。启动前重新检查并尝试bind，竞态导致bind失败就退出，不抢占或杀已有进程。本次实际避开活跃campaign driver **23199**，选择 **23198**。
3. 每次生成独立配置和精确文件allowlist，仅read-only；保留`hosts allow = 149.165.152.232 149.165.153.233`、`hosts deny = *`。无目录通配暴露、无credential、无SSH、无tether push/pull/expose，也不改旧daemon配置。
4. H100通过公网rsync单流拉取，`--bwlimit=16384`，本机IO avg10≥5%时拒绝继续。π仅工具/bundle/reference/3输入/该cache依赖/15个小checkpoint文件，最多约164MB；复用现有runtime、env、权重。FW仅两次完成的4090请求记录及收据，305,991,551 bytes，不传bulk batch ledger。
5. `finally`关闭本次自有daemon/process group；失败、Ctrl-C/SIGTERM也执行清理，留存`DAEMON_CLOSED.json`。只按自有PID处理，不按端口或进程名杀服务。**传输结束立即关闭daemon，再做CPU校验/配对**。

本次π补传及FW配对均已实际执行并COMPLETE，临时daemon均关闭，未使用23177。π首次新增CPU预检因采用老asset source digest而拒绝，daemon同样正常关闭；改为核对实际冻结release后通过，失败收据保留。9项新CPU测试与25项并发回归通过，见 `evidence/tests_temporary_stage_1018.log:1`、`evidence/tests_stage_concurrent_retry_1018.log:1`。

## 4. 协调员命令

**CPU补传已完成，以下为可重跑的一键stage：GPU 0GiB，预计3–5分钟**（首次小文件/依赖增量+64MiB/s权重hash；复用已有大权重）。脚本自动选择新端口并关闭，不需要手动恢复23177：

```bash
cd /home/weiland/projects/openpi
.venv/bin/python exp/offline_search/rounds/r17/xgpu_1017/temporary_h100_stage.py pi05 --run
```

**H100 lane边界执行下式两次：每次44GiB空闲，预计15–25分钟，watchdog25分钟。** 保持R4/B8/112客户端、40GiB allocator、cache/protocol、正式bootstrap与原数值设置。GPU运行尚未由本研究验证；这次交付解决的是已观测的CPU身份失败及依赖准入。

```bash
tether exec --timeout 30m h100 -- /home/exouser/openpi/.venv/bin/python /home/exouser/xgpu_1017/tools/run_concurrent.py --host h100 --model pi05 --checkpoint-view /data/mw_finetune/xgpu_1017/pi05_ckpt_4008289bc944 --batch-telemetry --run
```

**FW跨卡配对已完成；复现命令GPU 0GiB，预计2–6分钟。** 同一临时daemon机制完成传输、自动关闭、只读H100 CPU比较，将派生统计写回本地：

```bash
cd /home/weiland/projects/openpi
.venv/bin/python exp/offline_search/rounds/r17/xgpu_1017/temporary_h100_stage.py fw --run
```

结果：跨卡2,283/2,304超暂定wire预算，145/2,304超逐请求5×同卡floor（全cache）；top1稳定一致928/960=96.6667%。见更新后的 [Fast-WAM报告](FASTWAM_4090_H100_1018.md:1)。π→GR→Cosmos后续lane边界次序及闭环pilot计划不变。
