# W4：R17 启动器——下一批接到还活着的常驻

先读 exp/offline_search/rounds/r17/seeds_1023/prompts/COMMON.md（必须遵守）和 seeds_1023/DESIGN.md「常驻复用」。

## 目标（owner 10-23）
同一台机器上，前后两批 R17 campaign 的模型和步数 K 相同时，后一批不卸载、不重新加载模型：前一批结束时常驻和代理保持运行，后一批直接接上，只把自己的 YAML/缓存产物注册给常驻（`load_cache_config` 已能按 YAML 热注册，`formal_eval.py:1540-1547`），起自己的 driver 和 worker。K 或模型不同则照常停旧起新。

## 现状（只读调查结论，自行核实）
- 没有接管模式；`cache_model_order.py:72` 明确不接管活常驻；`live_fleets` 只管 worker。
- 最小改动方向：`formal_eval.run_campaign` 加两个开关——
  - 保留：finally（约 :1618-1655）不停常驻/代理，记录常驻信息（节点、端口、副本 pid、K、模型身份、MPS 状态）到一个常驻登记文件，`formal_finalize` 的 all_owned_closed 对「移交出去的常驻」放行；
  - 接管：跳过端口准入（:1401-1403）、MPS guard（:1404-1440）、起副本与校验（:1446-1500）、起代理（:1515-1521），核对登记文件里的模型身份/K/端口与本批一致后，注册 YAML（:1540-1547）、起 driver、起 worker。最后一批（或 K 变了）负责真正停常驻。
- 风险点：成功过滤 selection 目前只在常驻启动时读一次（`filter_failed_knob/runtime.py:22-24,82`）——W1 会改成每个 fit 加载时从旁边的 selection 文件惰性读取、缺了报错；你的接管模式依赖这点，交付里写明依赖。后一批的证据写到哪个目录要说清（不能写进前一批目录）；常驻内存只增不减，给出简单的上限判断（例如装的 YAML 数或显存余量）。
- 拔防御：`formal_prestage.py:46` 软链拒绝可拔；`formal_prestage.py:172-173,325` 的 LAUNCH.json 独占认领**保留**。

## 文件归属（只改这些）
`exp/r17_integration/ops/formal_eval.py`、`exp/r17_integration/ops/formal_finalize.py`、`exp/r17_integration/ops/formal_prestage.py`、`exp/r17_integration/ops/formal_ports.py`（若需要）、`exp/offline_search/rounds/r17/filter_failed_knob/launch.py`（加透传开关）、`openpi_ext/scratch/r17_ops/ff/` 下**新建**一个链脚本变体（不改现有 chain_ffk.run.sh）用来串「保留 → 接管 → … → 最后停」。

## 交付
按 COMMON.md。协调者命令要包括一个最小真机验证方案（例如在空闲的 sean 车道 2 或 pod 上，用两个 DP 小批次：第一批 keep、第二批 attach，核对第二批没有重新加载模型、YAML 注册成功、成功过滤生效、两批 journal 局数完整），以及失败时的回退方式。
