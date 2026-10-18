# 验收方案已替换为真实正式 campaign

2026-10-18 coordinator ruling：不再使用 256 集 side lane。此前 4/8 的人为上限没有 owner 授权，已从代码移除；不再将其描述为 owner 的限制。旧 256 集结果只作历史诊断，不能作为本次正式验收。

当前方案、所有 _dyn 包装器、prepare/launch/audit 命令与内存依据见 [FORMAL_ACCEPTANCE.md](FORMAL_ACCEPTANCE.md)。全局 ENABLED 保持 off；只有明确选定 campaign 的包装进程和新冻结 worker/driver overlay 私有启用动态。

原手交保留在 followup_1018/formal_dynamic_122x/before/TRIAL.md，已被本 ruling 撤销，不要执行其 side-lane 命令。

```bash
cat exp/offline_search/rounds/r17/autoscale_1017/FORMAL_ACCEPTANCE.md
```
