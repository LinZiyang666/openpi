最新 ruling（2026-10-18 12:2x，覆盖旧side-lane段落）：4/8上限没有owner授权，已移除。只对正式 fa_ls_fw_f_p_m / fa_pi_f_u_l / fa_pi_f_f_l 提供 _dyn 包装器；全局OFF，现有链/包装器/运行目录不改。详见 [FORMAL_ACCEPTANCE.md](FORMAL_ACCEPTANCE.md)。过滤臂用原生journal续跑，不重置；107内存预算96GiB+非ledger峰值增长，10秒一个spawn，guard无需跳过任何动态campaign。

当前交接：全局 OFF 已验证，普通队列使用原生路径。两个故障修复、物理 maxima 和共享账本公平分配已在工作树中，均处于 OFF 开关之后；只有 owner 的两条真实 lane 都 PASS 后才允许开启。

所有普通 prepare 入口都继承同一开关：R17 formal_eval、ff/ffk stage/resume/launch、formal_a_queue dispatch/build-levels、R16 conductor/runner、R15 sync/run_arm。OFF 不封装 autoscale overlay，不注入 governor/observer，也不改变原生回收。CPU dry run 证明 fa_pi_f_u_o 每 arm 39、fa_pi_r_f_l 为64、fa_ls_gr_r_p_m 为20/20；原生封包源码匹配补丁前 SHA。细节与测试替身边界见 [OFF_PREPARE_DRY_RUN.json](OFF_PREPARE_DRY_RUN.json)。timan108 仍排除，owner 的 control.py SO_REUSEADDR 修复保留。

故障修复：R17 每代通信 worker ID 带 UUID，同代重连身份稳定，slot ID 保持；现有 probe/endpoint 校验不放宽。Legacy 的根因是 client factory 闭包被 episode_setup 覆盖，产生四个零步错误后触发 N4；固定捕获 factory 后真实 LiberoEpisodeRunner CPU 路径完成两代八个不同 UID。先失败后通过的记录见 [RED_GREEN_SUMMARY.json](followup_1018/RED_GREEN_SUMMARY.json)。

候选 ON 的物理期望最大值为 LIBERO timan1/timan107=83/104、MetaWorld=332/440；旧小固定人数被提升，较大的配置保留。正式7项中没有小技术硬上限。fa_co_f_f_oc1 仍固定，因为 continuation 必须保留源 pins。CPU ON 候选7项证明见 new_chain_dry_run/，每份都记录真实生产开关 off；公平共享覆盖双 lane 同时起量、平台借用、归还与结束。

147 个 autoscale 测试、27 个 longseg 单元测试通过。sol 60 项 local-only 检查通过，其中33项本机 fresh、27项 PASS_LOCAL 使用明确标记的历史远端回执；本轮不重复远端探测。8组既有套件逐例对照无新增失败、无超时；既有失败和3个被生产端口保护阻止的用例仍列失败，真实 rsync daemon 用例两侧排除。原 cache-lane 卡住用例前/后为8.461/7.848秒。

[完整前后表](followup_1018/off_final/BEFORE_AFTER.md) · [逐例证据](followup_1018/off_final/BEFORE_AFTER.json) · [最终验证与源码 SHA](VERIFICATION.json)

以下重置命令仅由 owner 执行，本轮未执行。R17 要求 FAIL、all_owned_closed=true、空 accepted journal，整目录备份并校验 plan SHA。不可变缓存保持硬链接；其他文件按既有4 MiB/s预算串行复制，可变元数据不与备份共用 inode。

```bash
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.autoscale_1017.reset_zero --run /data/mw_finetune/eval/fa_ls_fw_r_p_m --backup /data/mw_finetune/eval/fa_ls_fw_r_p_m.before_autoscale_off_20261018 --reprepare
```

R15 命令备份全部8个 fa_pi_f_u_o_{s,l}{50,100,200,500} 本地目录。owner-confirmed-closed-zero 表示 owner 已确认远端全部关闭且零 accepted；本地没有完整远端 journal 时工具不会自行推断。远端旧证据不删除、不改写。

```bash
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.autoscale_1017.reset_legacy_zero --unit fa_pi_f_u_o --backup-suffix .before_autoscale_off_20261018 --owner-confirmed-closed-zero
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.formal_a_queue dispatch prepare --name fa_pi_f_u_o
```

[两条256集真实验收的完整 build/prepare/launch/audit 命令](TRIAL.md)。仅显式新建 fa_asv_* lane 的独立进程与新 payload 启用动态；全局文件保持 OFF。验收核对原 UID 列表 SHA、恰好一次接受、零错误、多代回收、关闭证据、内存最低值及覆盖整个区间的零 OOM 证据，并输出 AUTOSCALE_ACTIVE.csv。打印命令，不执行：

```bash
bash exp/offline_search/rounds/r17/autoscale_1017/trial_commands.sh
```

此前误传输：timan107 的 /tmp/openpi_autoscale_e9f9bbe5134a2fa64b1366ba3c904f8860dadb57fc777a1177a56241a1f97e7d.tar，409600 bytes，2026-10-18 00:31:42.569994496 CDT 为远端 ctime；网络协议总字节未记录。只读取证时未发现解包后的 release/测试岛；未联系 H100，没有替换 R16/R15 科学岛或冻结 payload。没有传输前 stat，不能证明同名 tmp 文件此前不存在。该次传输还短暂开启过本地23118 rsync daemon，违反了原禁止 expose 的边界；后续已用 CPU 命令及端口保护阻止重演。完整哈希、时间、远端遗留和48项本地遗留见 [事件报告](followup_1018/incident/INCIDENT.md)、[本地文件列表](followup_1018/incident/LOCAL_FILES.json)。没有删除这些文件。


2026-10-18 12:2x 接手：旧 R17 worker payload 的 immutable bootstrap / supervisor import 缺项（worker_installs、worker_disk）已在新 autoscale overlay 补齐；新增启动 dmesg cursor，保留 vmstat/cgroup counter pair，并禁止可读日志掩盖计数器增长。两条私有验收 launch 先只读检查 timan107 legacy 空闲。新目录/精确命令见 [TRIAL.md](TRIAL.md)；ENABLED 保持 off，旧 R15 UNKNOWN 不追补。
