前后测试对照（通过 / 失败；逐例超时）

| 套件 | 备份源码 | 当前源码 | 新增回归 | 当前超时 |
|---|---:|---:|---:|---:|
| ops | 492 / 39 | 495 / 36 | 0 | 0 |
| r17_conductor | 70 / 3 | 70 / 3 | 0 | 0 |
| core_conductor | 127 / 0 | 127 / 0 | 0 | 0 |
| ff | 44 / 5 | 45 / 4 | 0 | 0 |
| ffk | 23 / 1 | 23 / 1 | 0 | 0 |
| r16 | 213 / 5 | 213 / 5 | 0 | 0 |
| h100 | 216 / 9 | 225 / 0 | 0 | 0 |
| other_importers | 341 / 34 | 341 / 34 | 0 | 0 |

新增 post-apply CPU 测试：83 通过；0 失败。

19 份备份均核对 PATCH_BASE.json SHA；只读覆盖导入与源码读取，保留绝对路径和完整当前依赖。两边使用同一组当前测试。ops 每例180秒，其余每例60秒；CUDA禁用，BLAS单线程。

初轮期间 cache_study.py 增加了 task_namespace 支持。最终 ops 备份侧已按 dependencies_ops_after/PINS.json 重跑，四个共享依赖与当前侧测试的源码 SHA 一致；初轮备份结果只作排查记录。

H100 真实 rsync daemon 用例两边均 deselect；ops 的 rsync daemon 用例被 CPU guard 在 spawn 前阻断，两边都记失败。这两项不是产品回归结论。

完整逐例状态、错误与耗时见 BEFORE_AFTER.json；实际命令、日志、JUnit 路径见 BEFORE_AFTER_INPUTS.json 及各 regression_* 目录。未运行 GPU 验收。

原报告卡住用例：备份侧 7.965 秒，当前侧 8.090 秒。
