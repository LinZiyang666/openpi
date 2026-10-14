# MT50 缓存长段审核（本次新增描述统计）

逐 campaign 只纳入 accepted=true、无非空 error 的 done/failed；按原计划逐 uid 对照。
GR00T 续跑只继承 cache_resume_selection.retained_task_uids，不能把所有旧失败尝试混入。
完整 cell 必须有 50任务×50初态、seed7；数字均为百分数。组均值/按任务均值的定义同 MT50_AUDIT.md。
配对键为 (task_name, original_init_idx)，只有两侧完整且同 K 才显示长段对照；未做新推断检验。
Fast-WAM 合并 csfw_k3→w4→w5 的保留 UID；以计划及 study 的实际 num_steps 为准，续跑未齐局。

| 模型 | 库大小 | 组 | K | C | n / success | 组均值 / 任务均值 | 完整 | 原始来源 |
|---|---:|---|---:|---:|---|---|---|---|
| DP | 250 | c | 10 | 5 | 2500 / 1084 | 27.87 / 43.36 | True | [driver/journal.jsonl:1](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:1); [csdp_1504/EPISODE_PLAN.json:456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:456) |
| DP | 250 | cg | 10 | 5 | 2500 / 1123 | 28.82 / 44.92 | True | [driver/journal.jsonl:10001](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:10001); [csdp_1504/EPISODE_PLAN.json:420456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:420456) |
| DP | 250 | ch | 10 | 10 | 2500 / 976 | 23.22 / 39.04 | True | [driver/journal.jsonl:5001](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:5001); [csdp_1504/EPISODE_PLAN.json:210456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:210456) |
| DP | 250 | chg | 10 | 10 | 2500 / 977 | 23.13 / 39.08 | True | [driver/journal.jsonl:15001](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:15001); [csdp_1504/EPISODE_PLAN.json:630456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:630456) |
| DP | 250 | chr | 10 | 10 | 2500 / 1015 | 24.90 / 40.60 | True | [driver/journal.jsonl:7501](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:7501); [csdp_1504/EPISODE_PLAN.json:315456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:315456) |
| DP | 250 | chrg | 10 | 10 | 2500 / 1027 | 25.15 / 41.08 | True | [driver/journal.jsonl:17501](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:17501); [csdp_1504/EPISODE_PLAN.json:735456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:735456) |
| DP | 250 | cr | 10 | 5 | 2500 / 1150 | 30.39 / 46.00 | True | [driver/journal.jsonl:2501](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:2501); [csdp_1504/EPISODE_PLAN.json:105456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:105456) |
| DP | 250 | crg | 10 | 5 | 2500 / 1166 | 30.80 / 46.64 | True | [driver/journal.jsonl:12501](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:12501); [csdp_1504/EPISODE_PLAN.json:525456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:525456) |
| DP | 500 | c | 10 | 5 | 2500 / 1234 | 35.89 / 49.36 | True | [driver/journal.jsonl:20001](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:20001); [csdp_1504/EPISODE_PLAN.json:840456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:840456) |
| DP | 500 | cg | 10 | 5 | 2500 / 1272 | 37.65 / 50.88 | True | [driver/journal.jsonl:30001](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:30001); [csdp_1504/EPISODE_PLAN.json:1260456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:1260456) |
| DP | 500 | ch | 10 | 10 | 2500 / 1154 | 32.17 / 46.16 | True | [driver/journal.jsonl:25001](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:25001); [csdp_1504/EPISODE_PLAN.json:1050456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:1050456) |
| DP | 500 | chg | 10 | 10 | 2500 / 1176 | 32.41 / 47.04 | True | [driver/journal.jsonl:35001](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:35001); [csdp_1504/EPISODE_PLAN.json:1470456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:1470456) |
| DP | 500 | chr | 10 | 10 | 2500 / 1167 | 32.45 / 46.68 | True | [driver/journal.jsonl:27501](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:27501); [csdp_1504/EPISODE_PLAN.json:1155456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:1155456) |
| DP | 500 | chrg | 10 | 10 | 2500 / 1200 | 33.25 / 48.00 | True | [driver/journal.jsonl:37501](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:37501); [csdp_1504/EPISODE_PLAN.json:1575456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:1575456) |
| DP | 500 | cr | 10 | 5 | 2500 / 1279 | 38.43 / 51.16 | True | [driver/journal.jsonl:22501](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:22501); [csdp_1504/EPISODE_PLAN.json:945456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:945456) |
| DP | 500 | crg | 10 | 5 | 2500 / 1313 | 39.99 / 52.52 | True | [driver/journal.jsonl:32501](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:32501); [csdp_1504/EPISODE_PLAN.json:1365456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:1365456) |
| DP | 1000 | c | 10 | 5 | 2500 / 1373 | 40.38 / 54.92 | True | [driver/journal.jsonl:40001](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:40001); [csdp_1504/EPISODE_PLAN.json:1680456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:1680456) |
| DP | 1000 | cg | 10 | 5 | 2500 / 1393 | 41.17 / 55.72 | True | [driver/journal.jsonl:50001](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:50001); [csdp_1504/EPISODE_PLAN.json:2100456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:2100456) |
| DP | 1000 | ch | 10 | 10 | 2500 / 1279 | 35.97 / 51.16 | True | [driver/journal.jsonl:45001](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:45001); [csdp_1504/EPISODE_PLAN.json:1890456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:1890456) |
| DP | 1000 | chg | 10 | 10 | 2500 / 1289 | 36.29 / 51.56 | True | [driver/journal.jsonl:55001](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:55001); [csdp_1504/EPISODE_PLAN.json:2310456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:2310456) |
| DP | 1000 | chr | 10 | 10 | 2500 / 1306 | 36.83 / 52.24 | True | [driver/journal.jsonl:47501](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:47501); [csdp_1504/EPISODE_PLAN.json:1995456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:1995456) |
| DP | 1000 | chrg | 10 | 10 | 2500 / 1301 | 36.79 / 52.04 | True | [driver/journal.jsonl:57501](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:57501); [csdp_1504/EPISODE_PLAN.json:2415456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:2415456) |
| DP | 1000 | cr | 10 | 5 | 2500 / 1419 | 42.84 / 56.76 | True | [driver/journal.jsonl:42501](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:42501); [csdp_1504/EPISODE_PLAN.json:1785456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:1785456) |
| DP | 1000 | crg | 10 | 5 | 2500 / 1432 | 42.92 / 57.28 | True | [driver/journal.jsonl:52501](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:52501); [csdp_1504/EPISODE_PLAN.json:2205456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:2205456) |
| DP | 1500 | c | 10 | 5 | 2500 / 1431 | 41.60 / 57.24 | True | [driver/journal.jsonl:60001](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:60001); [csdp_1504/EPISODE_PLAN.json:2520456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:2520456) |
| DP | 1500 | cg | 10 | 5 | 2500 / 1434 | 42.23 / 57.36 | True | [driver/journal.jsonl:70001](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:70001); [csdp_1504/EPISODE_PLAN.json:2940456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:2940456) |
| DP | 1500 | ch | 10 | 10 | 2500 / 1371 | 38.89 / 54.84 | True | [driver/journal.jsonl:65001](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:65001); [csdp_1504/EPISODE_PLAN.json:2730456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:2730456) |
| DP | 1500 | chg | 10 | 10 | 2500 / 1375 | 39.02 / 55.00 | True | [driver/journal.jsonl:75001](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:75001); [csdp_1504/EPISODE_PLAN.json:3150456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:3150456) |
| DP | 1500 | chr | 10 | 10 | 2500 / 1390 | 39.70 / 55.60 | True | [driver/journal.jsonl:67501](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:67501); [csdp_1504/EPISODE_PLAN.json:2835456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:2835456) |
| DP | 1500 | chrg | 10 | 10 | 2500 / 1386 | 39.83 / 55.44 | True | [driver/journal.jsonl:77501](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:77501); [csdp_1504/EPISODE_PLAN.json:3255456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:3255456) |
| DP | 1500 | cr | 10 | 5 | 2500 / 1455 | 42.62 / 58.20 | True | [driver/journal.jsonl:62501](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:62501); [csdp_1504/EPISODE_PLAN.json:2625456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:2625456) |
| DP | 1500 | crg | 10 | 5 | 2500 / 1457 | 43.38 / 58.28 | True | [driver/journal.jsonl:72501](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:72501); [csdp_1504/EPISODE_PLAN.json:3045456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:3045456) |
| DP | 2000 | c | 10 | 5 | 2500 / 1436 | 42.19 / 57.44 | True | [driver/journal.jsonl:80001](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:80001); [csdp_1504/EPISODE_PLAN.json:3360456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:3360456) |
| DP | 2000 | cg | 10 | 5 | 2500 / 1444 | 43.14 / 57.76 | True | [driver/journal.jsonl:90001](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:90001); [csdp_1504/EPISODE_PLAN.json:3780456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:3780456) |
| DP | 2000 | ch | 10 | 10 | 2500 / 1412 | 40.36 / 56.48 | True | [driver/journal.jsonl:85001](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:85001); [csdp_1504/EPISODE_PLAN.json:3570456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:3570456) |
| DP | 2000 | chg | 10 | 10 | 2500 / 1412 | 40.60 / 56.48 | True | [driver/journal.jsonl:95001](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:95001); [csdp_1504/EPISODE_PLAN.json:3990456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:3990456) |
| DP | 2000 | chr | 10 | 10 | 2500 / 1411 | 40.73 / 56.44 | True | [driver/journal.jsonl:87501](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:87501); [csdp_1504/EPISODE_PLAN.json:3675456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:3675456) |
| DP | 2000 | chrg | 10 | 10 | 2500 / 1406 | 40.68 / 56.24 | True | [driver/journal.jsonl:97501](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:97501); [csdp_1504/EPISODE_PLAN.json:4095456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:4095456) |
| DP | 2000 | cr | 10 | 5 | 2500 / 1453 | 42.73 / 58.12 | True | [driver/journal.jsonl:82501](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:82501); [csdp_1504/EPISODE_PLAN.json:3465456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:3465456) |
| DP | 2000 | crg | 10 | 5 | 2500 / 1468 | 43.87 / 58.72 | True | [driver/journal.jsonl:92501](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:92501); [csdp_1504/EPISODE_PLAN.json:3885456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:3885456) |
| DP | 2500 | c | 10 | 5 | 2500 / 1445 | 43.02 / 57.80 | True | [driver/journal.jsonl:100001](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:100001); [csdp_1504/EPISODE_PLAN.json:4200456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:4200456) |
| DP | 2500 | cg | 10 | 5 | 2500 / 1470 | 44.58 / 58.80 | True | [driver/journal.jsonl:110001](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:110001); [csdp_1504/EPISODE_PLAN.json:4620456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:4620456) |
| DP | 2500 | ch | 10 | 10 | 2500 / 1414 | 40.81 / 56.56 | True | [driver/journal.jsonl:105001](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:105001); [csdp_1504/EPISODE_PLAN.json:4410456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:4410456) |
| DP | 2500 | chg | 10 | 10 | 2500 / 1405 | 40.91 / 56.20 | True | [driver/journal.jsonl:115001](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:115001); [csdp_1504/EPISODE_PLAN.json:4830456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:4830456) |
| DP | 2500 | chr | 10 | 10 | 2500 / 1433 | 41.79 / 57.32 | True | [driver/journal.jsonl:107501](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:107501); [csdp_1504/EPISODE_PLAN.json:4515456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:4515456) |
| DP | 2500 | chrg | 10 | 10 | 2500 / 1418 | 41.49 / 56.72 | True | [driver/journal.jsonl:117501](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:117501); [csdp_1504/EPISODE_PLAN.json:4935456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:4935456) |
| DP | 2500 | cr | 10 | 5 | 2500 / 1462 | 44.26 / 58.48 | True | [driver/journal.jsonl:102501](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:102501); [csdp_1504/EPISODE_PLAN.json:4305456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:4305456) |
| DP | 2500 | crg | 10 | 5 | 2500 / 1484 | 45.36 / 59.36 | True | [driver/journal.jsonl:112501](/data/mw_finetune/eval/csdp_1504/driver/journal.jsonl:112501); [csdp_1504/EPISODE_PLAN.json:4725456](/data/mw_finetune/eval/csdp_1504/EPISODE_PLAN.json:4725456) |
| GR00T | 250 | c | 2 | 5 | 2500 / 903 | 23.04 / 36.12 | True | [driver/journal.jsonl:1](/data/mw_finetune/eval/csgr_h5/driver/journal.jsonl:1); [csgr_h5/EPISODE_PLAN.json:193](/data/mw_finetune/eval/csgr_h5/EPISODE_PLAN.json:193) |
| GR00T | 250 | cg | 2 | 5 | 2500 / 981 | 25.65 / 39.24 | True | [driver/journal.jsonl:10001](/data/mw_finetune/eval/csgr_h5/driver/journal.jsonl:10001); [csgr_h5/EPISODE_PLAN.json:420193](/data/mw_finetune/eval/csgr_h5/EPISODE_PLAN.json:420193) |
| GR00T | 250 | ch | 2 | 10 | 2500 / 919 | 24.45 / 36.76 | True | [driver/journal.jsonl:5001](/data/mw_finetune/eval/csgr_h5/driver/journal.jsonl:5001); [csgr_h5/EPISODE_PLAN.json:210193](/data/mw_finetune/eval/csgr_h5/EPISODE_PLAN.json:210193) |
| GR00T | 250 | chg | 2 | 10 | 2500 / 943 | 24.96 / 37.72 | True | [driver/journal.jsonl:15001](/data/mw_finetune/eval/csgr_h5/driver/journal.jsonl:15001); [csgr_h5/EPISODE_PLAN.json:630193](/data/mw_finetune/eval/csgr_h5/EPISODE_PLAN.json:630193) |
| GR00T | 250 | chr | 2 | 10 | 2500 / 966 | 26.19 / 38.64 | True | [driver/journal.jsonl:7501](/data/mw_finetune/eval/csgr_h5/driver/journal.jsonl:7501); [csgr_h5/EPISODE_PLAN.json:315193](/data/mw_finetune/eval/csgr_h5/EPISODE_PLAN.json:315193) |
| GR00T | 250 | chrg | 2 | 10 | 2500 / 995 | 26.93 / 39.80 | True | [driver/journal.jsonl:17501](/data/mw_finetune/eval/csgr_h5/driver/journal.jsonl:17501); [csgr_h5/EPISODE_PLAN.json:735193](/data/mw_finetune/eval/csgr_h5/EPISODE_PLAN.json:735193) |
| GR00T | 250 | cr | 2 | 5 | 2500 / 1017 | 27.45 / 40.68 | True | [driver/journal.jsonl:2501](/data/mw_finetune/eval/csgr_h5/driver/journal.jsonl:2501); [csgr_h5/EPISODE_PLAN.json:105193](/data/mw_finetune/eval/csgr_h5/EPISODE_PLAN.json:105193) |
| GR00T | 250 | crg | 2 | 5 | 2500 / 1062 | 28.65 / 42.48 | True | [driver/journal.jsonl:12501](/data/mw_finetune/eval/csgr_h5/driver/journal.jsonl:12501); [csgr_h5/EPISODE_PLAN.json:525193](/data/mw_finetune/eval/csgr_h5/EPISODE_PLAN.json:525193) |
| GR00T | 500 | c | 2 | 5 | 2500 / 1152 | 33.71 / 46.08 | True | [driver/journal.jsonl:20001](/data/mw_finetune/eval/csgr_h5/driver/journal.jsonl:20001); [csgr_h5/EPISODE_PLAN.json:840193](/data/mw_finetune/eval/csgr_h5/EPISODE_PLAN.json:840193) |
| GR00T | 500 | cg | 2 | 5 | 2500 / 1209 | 35.89 / 48.36 | True | [driver/journal.jsonl:5595](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:5595); [csgr_h6/EPISODE_PLAN.json:235141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:235141) |
| GR00T | 500 | ch | 2 | 10 | 2500 / 1032 | 29.63 / 41.28 | True | [driver/journal.jsonl:595](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:595); [csgr_h6/EPISODE_PLAN.json:25141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:25141) |
| GR00T | 500 | chg | 2 | 10 | 2500 / 1060 | 30.29 / 42.40 | True | [driver/journal.jsonl:10595](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:10595); [csgr_h6/EPISODE_PLAN.json:445141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:445141) |
| GR00T | 500 | chr | 2 | 10 | 2500 / 1115 | 32.81 / 44.60 | True | [driver/journal.jsonl:3095](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:3095); [csgr_h6/EPISODE_PLAN.json:130141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:130141) |
| GR00T | 500 | chrg | 2 | 10 | 2500 / 1140 | 33.98 / 45.60 | True | [driver/journal.jsonl:13095](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:13095); [csgr_h6/EPISODE_PLAN.json:550141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:550141) |
| GR00T | 500 | cr | 2 | 5 | 2500 / 1243 | 36.47 / 49.72 | True | [driver/journal.jsonl:22501](/data/mw_finetune/eval/csgr_h5/driver/journal.jsonl:22501); [csgr_h5/EPISODE_PLAN.json:945193](/data/mw_finetune/eval/csgr_h5/EPISODE_PLAN.json:945193); [driver/journal.jsonl:1](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:1); [csgr_h6/EPISODE_PLAN.json:193](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:193) |
| GR00T | 500 | crg | 2 | 5 | 2500 / 1285 | 38.07 / 51.40 | True | [driver/journal.jsonl:8095](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:8095); [csgr_h6/EPISODE_PLAN.json:340141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:340141) |
| GR00T | 1000 | c | 2 | 5 | 2500 / 1310 | 38.52 / 52.40 | True | [driver/journal.jsonl:15595](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:15595); [csgr_h6/EPISODE_PLAN.json:655141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:655141) |
| GR00T | 1000 | cg | 2 | 5 | 2500 / 1352 | 40.41 / 54.08 | True | [driver/journal.jsonl:25595](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:25595); [csgr_h6/EPISODE_PLAN.json:1075141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:1075141) |
| GR00T | 1000 | ch | 2 | 10 | 2500 / 1256 | 37.25 / 50.24 | True | [driver/journal.jsonl:20595](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:20595); [csgr_h6/EPISODE_PLAN.json:865141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:865141) |
| GR00T | 1000 | chg | 2 | 10 | 2500 / 1277 | 38.13 / 51.08 | True | [driver/journal.jsonl:30595](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:30595); [csgr_h6/EPISODE_PLAN.json:1285141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:1285141) |
| GR00T | 1000 | chr | 2 | 10 | 2500 / 1307 | 39.91 / 52.28 | True | [driver/journal.jsonl:23095](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:23095); [csgr_h6/EPISODE_PLAN.json:970141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:970141) |
| GR00T | 1000 | chrg | 2 | 10 | 2500 / 1329 | 40.55 / 53.16 | True | [driver/journal.jsonl:33095](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:33095); [csgr_h6/EPISODE_PLAN.json:1390141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:1390141) |
| GR00T | 1000 | cr | 2 | 5 | 2500 / 1393 | 42.25 / 55.72 | True | [driver/journal.jsonl:18095](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:18095); [csgr_h6/EPISODE_PLAN.json:760141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:760141) |
| GR00T | 1000 | crg | 2 | 5 | 2500 / 1417 | 43.11 / 56.68 | True | [driver/journal.jsonl:28095](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:28095); [csgr_h6/EPISODE_PLAN.json:1180141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:1180141) |
| GR00T | 1500 | c | 2 | 5 | 2500 / 1404 | 42.08 / 56.16 | True | [driver/journal.jsonl:35595](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:35595); [csgr_h6/EPISODE_PLAN.json:1495141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:1495141) |
| GR00T | 1500 | cg | 2 | 5 | 2500 / 1439 | 44.21 / 57.56 | True | [driver/journal.jsonl:45595](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:45595); [csgr_h6/EPISODE_PLAN.json:1915141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:1915141) |
| GR00T | 1500 | ch | 2 | 10 | 2500 / 1389 | 42.13 / 55.56 | True | [driver/journal.jsonl:40595](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:40595); [csgr_h6/EPISODE_PLAN.json:1705141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:1705141) |
| GR00T | 1500 | chg | 2 | 10 | 2500 / 1394 | 42.40 / 55.76 | True | [driver/journal.jsonl:50595](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:50595); [csgr_h6/EPISODE_PLAN.json:2125141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:2125141) |
| GR00T | 1500 | chr | 2 | 10 | 2500 / 1404 | 42.42 / 56.16 | True | [driver/journal.jsonl:43095](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:43095); [csgr_h6/EPISODE_PLAN.json:1810141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:1810141) |
| GR00T | 1500 | chrg | 2 | 10 | 2500 / 1412 | 42.91 / 56.48 | True | [driver/journal.jsonl:53095](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:53095); [csgr_h6/EPISODE_PLAN.json:2230141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:2230141) |
| GR00T | 1500 | cr | 2 | 5 | 2500 / 1478 | 45.35 / 59.12 | True | [driver/journal.jsonl:38095](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:38095); [csgr_h6/EPISODE_PLAN.json:1600141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:1600141) |
| GR00T | 1500 | crg | 2 | 5 | 2500 / 1508 | 47.12 / 60.32 | True | [driver/journal.jsonl:48095](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:48095); [csgr_h6/EPISODE_PLAN.json:2020141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:2020141) |
| GR00T | 2000 | c | 2 | 5 | 2500 / 1436 | 43.88 / 57.44 | True | [driver/journal.jsonl:55595](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:55595); [csgr_h6/EPISODE_PLAN.json:2335141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:2335141) |
| GR00T | 2000 | cg | 2 | 5 | 2500 / 1481 | 46.29 / 59.24 | True | [driver/journal.jsonl:65595](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:65595); [csgr_h6/EPISODE_PLAN.json:2755141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:2755141) |
| GR00T | 2000 | ch | 2 | 10 | 2500 / 1442 | 43.95 / 57.68 | True | [driver/journal.jsonl:60595](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:60595); [csgr_h6/EPISODE_PLAN.json:2545141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:2545141) |
| GR00T | 2000 | chg | 2 | 10 | 2500 / 1457 | 44.67 / 58.28 | True | [driver/journal.jsonl:70595](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:70595); [csgr_h6/EPISODE_PLAN.json:2965141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:2965141) |
| GR00T | 2000 | chr | 2 | 10 | 2500 / 1493 | 46.81 / 59.72 | True | [driver/journal.jsonl:63095](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:63095); [csgr_h6/EPISODE_PLAN.json:2650141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:2650141) |
| GR00T | 2000 | chrg | 2 | 10 | 2500 / 1498 | 47.21 / 59.92 | True | [driver/journal.jsonl:73095](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:73095); [csgr_h6/EPISODE_PLAN.json:3070141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:3070141) |
| GR00T | 2000 | cr | 2 | 5 | 2500 / 1504 | 46.93 / 60.16 | True | [driver/journal.jsonl:58095](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:58095); [csgr_h6/EPISODE_PLAN.json:2440141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:2440141) |
| GR00T | 2000 | crg | 2 | 5 | 2500 / 1538 | 49.22 / 61.52 | True | [driver/journal.jsonl:68095](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:68095); [csgr_h6/EPISODE_PLAN.json:2860141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:2860141) |
| GR00T | 2500 | c | 2 | 5 | 2500 / 1483 | 46.55 / 59.32 | True | [driver/journal.jsonl:75595](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:75595); [csgr_h6/EPISODE_PLAN.json:3175141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:3175141) |
| GR00T | 2500 | cg | 2 | 5 | 2500 / 1524 | 48.45 / 60.96 | True | [driver/journal.jsonl:85595](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:85595); [csgr_h6/EPISODE_PLAN.json:3595141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:3595141) |
| GR00T | 2500 | ch | 2 | 10 | 2500 / 1454 | 44.48 / 58.16 | True | [driver/journal.jsonl:80595](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:80595); [csgr_h6/EPISODE_PLAN.json:3385141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:3385141) |
| GR00T | 2500 | chg | 2 | 10 | 2500 / 1464 | 44.61 / 58.56 | True | [driver/journal.jsonl:90595](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:90595); [csgr_h6/EPISODE_PLAN.json:3805141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:3805141) |
| GR00T | 2500 | chr | 2 | 10 | 2500 / 1497 | 46.69 / 59.88 | True | [driver/journal.jsonl:83095](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:83095); [csgr_h6/EPISODE_PLAN.json:3490141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:3490141) |
| GR00T | 2500 | chrg | 2 | 10 | 2500 / 1496 | 46.47 / 59.84 | True | [driver/journal.jsonl:93095](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:93095); [csgr_h6/EPISODE_PLAN.json:3910141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:3910141) |
| GR00T | 2500 | cr | 2 | 5 | 2500 / 1553 | 49.59 / 62.12 | True | [driver/journal.jsonl:78095](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:78095); [csgr_h6/EPISODE_PLAN.json:3280141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:3280141) |
| GR00T | 2500 | crg | 2 | 5 | 2500 / 1575 | 50.51 / 63.00 | True | [driver/journal.jsonl:88095](/data/mw_finetune/eval/csgr_h6/driver/journal.jsonl:88095); [csgr_h6/EPISODE_PLAN.json:3700141](/data/mw_finetune/eval/csgr_h6/EPISODE_PLAN.json:3700141) |
| Fast-WAM(partial) | 250 | c | 3 | 5 | 2500 / 894 | 21.46 / 35.76 | True | [driver/journal.jsonl:1](/data/mw_finetune/eval/csfw_k3/driver/journal.jsonl:1); [csfw_k3/EPISODE_PLAN.json:236](/data/mw_finetune/eval/csfw_k3/EPISODE_PLAN.json:236) |
| Fast-WAM(partial) | 250 | cg | 3 | 5 | 2500 / 956 | 22.62 / 38.24 | True | [driver/journal.jsonl:10001](/data/mw_finetune/eval/csfw_k3/driver/journal.jsonl:10001); [csfw_k3/EPISODE_PLAN.json:420236](/data/mw_finetune/eval/csfw_k3/EPISODE_PLAN.json:420236) |
| Fast-WAM(partial) | 250 | ch | 3 | 10 | 2500 / 806 | 19.55 / 32.24 | True | [driver/journal.jsonl:5001](/data/mw_finetune/eval/csfw_k3/driver/journal.jsonl:5001); [csfw_k3/EPISODE_PLAN.json:210236](/data/mw_finetune/eval/csfw_k3/EPISODE_PLAN.json:210236) |
| Fast-WAM(partial) | 250 | chg | 3 | 10 | 2500 / 848 | 20.39 / 33.92 | True | [driver/journal.jsonl:15001](/data/mw_finetune/eval/csfw_k3/driver/journal.jsonl:15001); [csfw_k3/EPISODE_PLAN.json:630236](/data/mw_finetune/eval/csfw_k3/EPISODE_PLAN.json:630236) |
| Fast-WAM(partial) | 250 | chr | 3 | 10 | 2500 / 920 | 23.60 / 36.80 | True | [driver/journal.jsonl:7501](/data/mw_finetune/eval/csfw_k3/driver/journal.jsonl:7501); [csfw_k3/EPISODE_PLAN.json:315236](/data/mw_finetune/eval/csfw_k3/EPISODE_PLAN.json:315236) |
| Fast-WAM(partial) | 250 | chrg | 3 | 10 | 2500 / 952 | 24.28 / 38.08 | True | [driver/journal.jsonl:17501](/data/mw_finetune/eval/csfw_k3/driver/journal.jsonl:17501); [csfw_k3/EPISODE_PLAN.json:735236](/data/mw_finetune/eval/csfw_k3/EPISODE_PLAN.json:735236) |
| Fast-WAM(partial) | 250 | cr | 3 | 5 | 2500 / 948 | 23.69 / 37.92 | True | [driver/journal.jsonl:2501](/data/mw_finetune/eval/csfw_k3/driver/journal.jsonl:2501); [csfw_k3/EPISODE_PLAN.json:105236](/data/mw_finetune/eval/csfw_k3/EPISODE_PLAN.json:105236) |
| Fast-WAM(partial) | 250 | crg | 3 | 5 | 2500 / 985 | 24.23 / 39.40 | True | [driver/journal.jsonl:12501](/data/mw_finetune/eval/csfw_k3/driver/journal.jsonl:12501); [csfw_k3/EPISODE_PLAN.json:525236](/data/mw_finetune/eval/csfw_k3/EPISODE_PLAN.json:525236) |
| Fast-WAM(partial) | 500 | c | 3 | 5 | 2500 / 995 | 25.32 / 39.80 | True | [driver/journal.jsonl:20001](/data/mw_finetune/eval/csfw_k3/driver/journal.jsonl:20001); [csfw_k3/EPISODE_PLAN.json:840236](/data/mw_finetune/eval/csfw_k3/EPISODE_PLAN.json:840236) |
| Fast-WAM(partial) | 500 | cg | 3 | 5 | 2500 / 1031 | 25.94 / 41.24 | True | [driver/journal.jsonl:30001](/data/mw_finetune/eval/csfw_k3/driver/journal.jsonl:30001); [csfw_k3/EPISODE_PLAN.json:1260236](/data/mw_finetune/eval/csfw_k3/EPISODE_PLAN.json:1260236) |
| Fast-WAM(partial) | 500 | ch | 3 | 10 | 2500 / 1002 | 25.18 / 40.08 | True | [driver/journal.jsonl:25001](/data/mw_finetune/eval/csfw_k3/driver/journal.jsonl:25001); [csfw_k3/EPISODE_PLAN.json:1050236](/data/mw_finetune/eval/csfw_k3/EPISODE_PLAN.json:1050236) |
| Fast-WAM(partial) | 500 | chg | 3 | 10 | 2500 / 1004 | 25.36 / 40.16 | True | [driver/journal.jsonl:35001](/data/mw_finetune/eval/csfw_k3/driver/journal.jsonl:35001); [csfw_k3/EPISODE_PLAN.json:1470236](/data/mw_finetune/eval/csfw_k3/EPISODE_PLAN.json:1470236) |
| Fast-WAM(partial) | 500 | chr | 3 | 10 | 2500 / 1054 | 27.23 / 42.16 | True | [driver/journal.jsonl:27501](/data/mw_finetune/eval/csfw_k3/driver/journal.jsonl:27501); [csfw_k3/EPISODE_PLAN.json:1155236](/data/mw_finetune/eval/csfw_k3/EPISODE_PLAN.json:1155236) |
| Fast-WAM(partial) | 500 | chrg | 3 | 10 | 2500 / 1053 | 27.26 / 42.12 | True | [driver/journal.jsonl:37501](/data/mw_finetune/eval/csfw_k3/driver/journal.jsonl:37501); [csfw_k3/EPISODE_PLAN.json:1575236](/data/mw_finetune/eval/csfw_k3/EPISODE_PLAN.json:1575236) |
| Fast-WAM(partial) | 500 | cr | 3 | 5 | 2500 / 1056 | 27.54 / 42.24 | True | [driver/journal.jsonl:22501](/data/mw_finetune/eval/csfw_k3/driver/journal.jsonl:22501); [csfw_k3/EPISODE_PLAN.json:945236](/data/mw_finetune/eval/csfw_k3/EPISODE_PLAN.json:945236) |
| Fast-WAM(partial) | 500 | crg | 3 | 5 | 2500 / 1093 | 28.19 / 43.72 | True | [driver/journal.jsonl:32501](/data/mw_finetune/eval/csfw_k3/driver/journal.jsonl:32501); [csfw_k3/EPISODE_PLAN.json:1365236](/data/mw_finetune/eval/csfw_k3/EPISODE_PLAN.json:1365236) |
| Fast-WAM(partial) | 1000 | c | 3 | 5 | 2500 / 1118 | 30.00 / 44.72 | True | [driver/journal.jsonl:40001](/data/mw_finetune/eval/csfw_k3/driver/journal.jsonl:40001); [csfw_k3/EPISODE_PLAN.json:1680236](/data/mw_finetune/eval/csfw_k3/EPISODE_PLAN.json:1680236) |
| Fast-WAM(partial) | 1000 | cg | 3 | 5 | 2500 / 1161 | 30.64 / 46.44 | True | [driver/journal.jsonl:4235](/data/mw_finetune/eval/csfw_k3w4/driver/journal.jsonl:4235); [csfw_k3w4/EPISODE_PLAN.json:178064](/data/mw_finetune/eval/csfw_k3w4/EPISODE_PLAN.json:178064) |
| Fast-WAM(partial) | 1000 | ch | 3 | 10 | 2500 / 1087 | 28.71 / 43.48 | True | [driver/journal.jsonl:45001](/data/mw_finetune/eval/csfw_k3/driver/journal.jsonl:45001); [csfw_k3/EPISODE_PLAN.json:1890236](/data/mw_finetune/eval/csfw_k3/EPISODE_PLAN.json:1890236); [driver/journal.jsonl:1](/data/mw_finetune/eval/csfw_k3w4/driver/journal.jsonl:1); [csfw_k3w4/EPISODE_PLAN.json:236](/data/mw_finetune/eval/csfw_k3w4/EPISODE_PLAN.json:236) |
| Fast-WAM(partial) | 1000 | chg | 3 | 10 | 2500 / 1094 | 29.35 / 43.76 | True | [driver/journal.jsonl:2401](/data/mw_finetune/eval/csfw_k3w5/driver/journal.jsonl:2401); [csfw_k3w5/EPISODE_PLAN.json:101036](/data/mw_finetune/eval/csfw_k3w5/EPISODE_PLAN.json:101036) |
| Fast-WAM(partial) | 1000 | chr | 3 | 10 | 2500 / 1131 | 29.60 / 45.24 | True | [driver/journal.jsonl:1735](/data/mw_finetune/eval/csfw_k3w4/driver/journal.jsonl:1735); [csfw_k3w4/EPISODE_PLAN.json:73064](/data/mw_finetune/eval/csfw_k3w4/EPISODE_PLAN.json:73064) |
| Fast-WAM(partial) | 1000 | chrg | 3 | 10 | 2500 / 1118 | 29.78 / 44.72 | True | [driver/journal.jsonl:4901](/data/mw_finetune/eval/csfw_k3w5/driver/journal.jsonl:4901); [csfw_k3w5/EPISODE_PLAN.json:206036](/data/mw_finetune/eval/csfw_k3w5/EPISODE_PLAN.json:206036) |
| Fast-WAM(partial) | 1000 | cr | 3 | 5 | 2500 / 1173 | 31.56 / 46.92 | True | [driver/journal.jsonl:42501](/data/mw_finetune/eval/csfw_k3/driver/journal.jsonl:42501); [csfw_k3/EPISODE_PLAN.json:1785236](/data/mw_finetune/eval/csfw_k3/EPISODE_PLAN.json:1785236) |
| Fast-WAM(partial) | 1000 | crg | 3 | 5 | 2468 / 1182 | 不报未齐 cell 的 SR | False | [driver/journal.jsonl:6735](/data/mw_finetune/eval/csfw_k3w4/driver/journal.jsonl:6735); [csfw_k3w4/EPISODE_PLAN.json:283064](/data/mw_finetune/eval/csfw_k3w4/EPISODE_PLAN.json:283064); [driver/journal.jsonl:1](/data/mw_finetune/eval/csfw_k3w5/driver/journal.jsonl:1); [csfw_k3w5/EPISODE_PLAN.json:236](/data/mw_finetune/eval/csfw_k3w5/EPISODE_PLAN.json:236) |
| Fast-WAM(partial) | 1500 | c | 3 | 5 | 574 / 306 | 不报未齐 cell 的 SR | False | [driver/journal.jsonl:7401](/data/mw_finetune/eval/csfw_k3w5/driver/journal.jsonl:7401); [csfw_k3w5/EPISODE_PLAN.json:311036](/data/mw_finetune/eval/csfw_k3w5/EPISODE_PLAN.json:311036) |

## 完整的短段→长段对照（同初态配对，描述性）

| 模型 | 库大小 | 组 | K | 组均值短→长 | 任务均值短→长 | g/l |
|---|---:|---|---:|---|---|---|
| DP | 250 | c→ch | 10 | 27.87→23.22 | 43.36→39.04 | 95/203 |
| DP | 500 | c→ch | 10 | 35.89→32.17 | 49.36→46.16 | 104/184 |
| DP | 1000 | c→ch | 10 | 40.38→35.97 | 54.92→51.16 | 131/225 |
| DP | 1500 | c→ch | 10 | 41.60→38.89 | 57.24→54.84 | 155/215 |
| DP | 2000 | c→ch | 10 | 42.19→40.36 | 57.44→56.48 | 154/178 |
| DP | 2500 | c→ch | 10 | 43.02→40.81 | 57.80→56.56 | 156/187 |
| DP | 250 | cr→chr | 10 | 30.39→24.90 | 46.00→40.60 | 79/214 |
| DP | 500 | cr→chr | 10 | 38.43→32.45 | 51.16→46.68 | 98/210 |
| DP | 1000 | cr→chr | 10 | 42.84→36.83 | 56.76→52.24 | 142/255 |
| DP | 1500 | cr→chr | 10 | 42.62→39.70 | 58.20→55.60 | 169/234 |
| DP | 2000 | cr→chr | 10 | 42.73→40.73 | 58.12→56.44 | 161/203 |
| DP | 2500 | cr→chr | 10 | 44.26→41.79 | 58.48→57.32 | 161/190 |
| DP | 250 | cg→chg | 10 | 28.82→23.13 | 44.92→39.08 | 75/221 |
| DP | 500 | cg→chg | 10 | 37.65→32.41 | 50.88→47.04 | 112/208 |
| DP | 1000 | cg→chg | 10 | 41.17→36.29 | 55.72→51.56 | 136/240 |
| DP | 1500 | cg→chg | 10 | 42.23→39.02 | 57.36→55.00 | 161/220 |
| DP | 2000 | cg→chg | 10 | 43.14→40.60 | 57.76→56.48 | 167/199 |
| DP | 2500 | cg→chg | 10 | 44.58→40.91 | 58.80→56.20 | 147/212 |
| DP | 250 | crg→chrg | 10 | 30.80→25.15 | 46.64→41.08 | 74/213 |
| DP | 500 | crg→chrg | 10 | 39.99→33.25 | 52.52→48.00 | 114/227 |
| DP | 1000 | crg→chrg | 10 | 42.92→36.79 | 57.28→52.04 | 137/268 |
| DP | 1500 | crg→chrg | 10 | 43.38→39.83 | 58.28→55.44 | 162/233 |
| DP | 2000 | crg→chrg | 10 | 43.87→40.68 | 58.72→56.24 | 149/211 |
| DP | 2500 | crg→chrg | 10 | 45.36→41.49 | 59.36→56.72 | 148/214 |
| GR00T | 250 | c→ch | 2 | 23.04→24.45 | 36.12→36.76 | 151/135 |
| GR00T | 500 | c→ch | 2 | 33.71→29.63 | 46.08→41.28 | 107/227 |
| GR00T | 1000 | c→ch | 2 | 38.52→37.25 | 52.40→50.24 | 131/185 |
| GR00T | 1500 | c→ch | 2 | 42.08→42.13 | 56.16→55.56 | 147/162 |
| GR00T | 2000 | c→ch | 2 | 43.88→43.95 | 57.44→57.68 | 148/142 |
| GR00T | 2500 | c→ch | 2 | 46.55→44.48 | 59.32→58.16 | 139/168 |
| GR00T | 250 | cr→chr | 2 | 27.45→26.19 | 40.68→38.64 | 118/169 |
| GR00T | 500 | cr→chr | 2 | 36.47→32.81 | 49.72→44.60 | 99/227 |
| GR00T | 1000 | cr→chr | 2 | 42.25→39.91 | 55.72→52.28 | 117/203 |
| GR00T | 1500 | cr→chr | 2 | 45.35→42.42 | 59.12→56.16 | 127/201 |
| GR00T | 2000 | cr→chr | 2 | 46.93→46.81 | 60.16→59.72 | 157/168 |
| GR00T | 2500 | cr→chr | 2 | 49.59→46.69 | 62.12→59.88 | 132/188 |
| GR00T | 250 | cg→chg | 2 | 25.65→24.96 | 39.24→37.72 | 128/166 |
| GR00T | 500 | cg→chg | 2 | 35.89→30.29 | 48.36→42.40 | 108/257 |
| GR00T | 1000 | cg→chg | 2 | 40.41→38.13 | 54.08→51.08 | 130/205 |
| GR00T | 1500 | cg→chg | 2 | 44.21→42.40 | 57.56→55.76 | 147/192 |
| GR00T | 2000 | cg→chg | 2 | 46.29→44.67 | 59.24→58.28 | 144/168 |
| GR00T | 2500 | cg→chg | 2 | 48.45→44.61 | 60.96→58.56 | 123/183 |
| GR00T | 250 | crg→chrg | 2 | 28.65→26.93 | 42.48→39.80 | 114/181 |
| GR00T | 500 | crg→chrg | 2 | 38.07→33.98 | 51.40→45.60 | 108/253 |
| GR00T | 1000 | crg→chrg | 2 | 43.11→40.55 | 56.68→53.16 | 118/206 |
| GR00T | 1500 | crg→chrg | 2 | 47.12→42.91 | 60.32→56.48 | 111/207 |
| GR00T | 2000 | crg→chrg | 2 | 49.22→47.21 | 61.52→59.92 | 146/186 |
| GR00T | 2500 | crg→chrg | 2 | 50.51→46.47 | 63.00→59.84 | 128/207 |
| Fast-WAM(partial) | 250 | c→ch | 3 | 21.46→19.55 | 35.76→32.24 | 78/166 |
| Fast-WAM(partial) | 500 | c→ch | 3 | 25.32→25.18 | 39.80→40.08 | 116/109 |
| Fast-WAM(partial) | 1000 | c→ch | 3 | 30.00→28.71 | 44.72→43.48 | 97/128 |
| Fast-WAM(partial) | 250 | cr→chr | 3 | 23.69→23.60 | 37.92→36.80 | 106/134 |
| Fast-WAM(partial) | 500 | cr→chr | 3 | 27.54→27.23 | 42.24→42.16 | 115/117 |
| Fast-WAM(partial) | 1000 | cr→chr | 3 | 31.56→29.60 | 46.92→45.24 | 102/144 |
| Fast-WAM(partial) | 250 | cg→chg | 3 | 22.62→20.39 | 38.24→33.92 | 98/206 |
| Fast-WAM(partial) | 500 | cg→chg | 3 | 25.94→25.36 | 41.24→40.16 | 101/128 |
| Fast-WAM(partial) | 1000 | cg→chg | 3 | 30.64→29.35 | 46.44→43.76 | 96/163 |
| Fast-WAM(partial) | 250 | crg→chrg | 3 | 24.23→24.28 | 39.40→38.08 | 133/166 |
| Fast-WAM(partial) | 500 | crg→chrg | 3 | 28.19→27.26 | 43.72→42.12 | 108/148 |
