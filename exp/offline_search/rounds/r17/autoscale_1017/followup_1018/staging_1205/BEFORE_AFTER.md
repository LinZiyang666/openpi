| 测试 | 修改前通过/失败 | 修改后通过/失败 |
|---|---:|---:|
| test_acceptance_fixes | 20/0 | 20/0 |
| test_reset_zero | 10/0 | 10/0 |
| test_slot4 | 12/0 | 12/0 |
| test_staging_reset | 7/7 | 14/0 |

现有 42 项全部继续通过；新增 14 项全部通过。每项超时 90 秒，无挂起、无新增失败。旧源通过导入快照测试，不还原共享工作树。control.py SHA 不变，保留 SO_REUSEADDR。
