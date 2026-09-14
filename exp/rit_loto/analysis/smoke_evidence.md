# G2 交付证据（advisory，执行法 §4 许可的自检；无 §6 Verify 效力）

记录时间：2026-09-14（America/Chicago）。以下是执行者提交 G2 Round 2 之前的历史 advisory 证据；执行者当时记录了与 weilandserver `/data/openpi_lg` 副本的双侧 SHA 对账。本轮 owner 授权修复已改变代码与内容摘要契约，下表及远端数值不代表修复后的版本，也不能作为新版本的正式 parity 准入记录。

## 1. 代码版本

| 文件 | sha256 |
|---|---|
| `exp/rit_loto/build_loto_table.py` | `7425c84c90b9a0cd755cbfded6465f745fc35f9edc3f883fd339d82b66cc48b5` |
| `exp/rit_loto/noise_floor.py` | `4eab643936c6535aaa2e96a0b7355f48f79617ed0db7db139595e6a916311601` |
| `exp/rit_loto/fit_loto.py` | `0cfc0d0319a50ffbf86687b21dc5314ce8e83790ef69609dacd4ee4139747c07` |
| `exp/rit_loto/loto_logger.py` | `36212818d978a774ae697a3ca83aa6c7216c9b67299d44d2a094e037db23bdd3` |
| `exp/rit_loto/emit_verify_arm.py` | `b4e805412a686956a290a6b5578decb096a450b54aa8cfefd5c43801f3c87f9c` |
| `exp/rit_loto/verify_closed_loop.py` | `1ee029778170ed7bcb1a73dafe37101ccdd52993c431de2a0a3414cbb04b233a` |
| `exp/libero_groot/serve_groot_libero.py` | `54e91647f1f9659d2f23711ae93182c3706b5df94173ebeb590624a5ac1d641d` |
| `tests/exp/test_rit_loto.py` | `bb4afebb68367798daa2f566a61094546e39487a2abfae8e3d46cb622951d2c4` |
| `exp/rit_loto/ops/run_noise_floor.sh` | `adca1a1b3cf8a5b39c68178490d44ab579ad749b76fef693eda3288b61570c96` |
| `exp/rit_loto/ops/run_loto_table.sh` | `e93ac8845b012ef5abc123c1352d3f53817317ee5d6aa7445316fead675c492e` |
| `exp/rit_loto/ops/launch_verify_server.sh` | `570feb899f38909187390cf75a75fa8a99e735543ea4aee428fed7a414d8663d` |
| `exp/rit_loto/ops/launch_verify_clients.sh` | `0af7462c71d0d3269f2d37b37f408b75d8ca9bbed1d75069ae3fcaef29965356` |

## 2. 本地（主 venv，无 GPU）

```
uv run pytest tests/exp/test_rit_loto.py tests/libero_groot -q --color=no -p no:cacheprovider
  -> 238 passed, 5 skipped   （其中 tests/exp/test_rit_loto.py 35 passed）
uv run ruff check exp/rit_loto tests/exp/test_rit_loto.py exp/libero_groot/serve_groot_libero.py
  -> All checks passed!
```

`tests/exp/test_rit_loto.py` 覆盖 G2 R1 的每条 Blocking：B1（冻结记录：发臂 template 不符 / 服务端臂·库·ckpt·H_exec·schedule 不符 / collect 缺冻结 attrs / report 换 suite 曲线、换 fits）、B2（缺一半标签、重复、NaN 标签、partial 产物、NaN 在线分数）、B3（forward 文件 sha 变化、缺 code_sha256、同名同长不同字节的 checkpoint）、B4（payload `pi05_v1` 混入 k8 库、`[1,32]` 形状）、B5（组名错位、control_step 错、task 不符、缺字段、h_exec 不符、池不完整）、B6（50 集全部混合、率 0.24 → 风险超标）、B7（alpha 0.20 自洽记录、K={1,2}）。

## 3. weilandserver 岛上冒烟链（GR00T island venv，RTX 4090）

命令：`tmux new -s lotosmk -d "bash /tmp/loto_smoke_chain.sh 2>&1 | tee /tmp/lotosmk2.log"`（脚本内容 = 本仓 `exp/rit_loto/ops/run_noise_floor.sh` / `run_loto_table.sh` 的三段展开，suite=libero_spatial，输出根 `/data/libero_cache/rit_loto_smoke/libero_spatial/`）。日志 `/tmp/lotosmk2.log`。

| 段 | 参数 | 结果 |
|---|---|---|
| `noise_floor` | `--per-task 2`（20 行） | `D(ref1,ref2)` n=20 median 6.8729 / p90 7.4061 / p95 7.6145；`D(ref1,clean)` median 6.9100；8 s；`NF_EXIT=0` |
| `build_loto_table --parity-only` | `--parity-sample 20` | 20 行 7 s；`parity_D_{full,warm75,warm50}` p50/p90 全为 0.0000；gate **FAIL**（原因仅为样本量：20 行 < 200、每任务 2 行 < 20，即预期）；`PAR_EXIT=0` |
| `build_loto_table`（表） | `--limit-episodes 2 --allow-ungated-smoke --orchestrator-check 3` | 2 集 42 行（库内 16 行、自身轨迹跳过 18 次）；orchestrator 自证 8/8 一致、prompt 置换 8/8 一致、no_hit 0；5.3 行/s；`TAB_EXIT=0` |

产物身份（`loto_table.smoke.jsonl.record.json` → `identity`）：checkpoint 内容 sha `8e51aa33ae6bdec7…`（`/data/ckpt/n15_libero_spatial`，7.6 GB safetensors 内容哈希，缓存于 `~/.cache/rit_loto/ckpt_identity/`）、库 `7e8993793489ad06…`、template `2f834f9ebc0450e3…`、W `13e5f7647dcd7e36…`、语料清单 `8b9d0af7564872f0…`；`code_sha256` 与 §1 的对应文件一致。

读数说明：两次独立完整推理之间的 `D` 中位数 6.87 与 shadow 表各档 `D`（5.5–9.1）同量级，提示教师随机性可能贡献较大；20 行不足以确定整套任务的主导误差来源。该样本的 parity_D 为 0，仅说明这些重放行数值一致，不能推断正式 200 行门会通过。完整 noise floor 和 parity 必须在最终代码版本上重跑。

## 4. 尚未运行

正式 noise floor（500/suite，50/任务）、parity 门（200 行）、全表、fit/compare/bootstrap、发臂与两池、闭环 smoke/verify、label/report。按计划 §5 顺序在 G2 APPROVED 与 §6 Verify 之后进行。

## 5. G2 Round 2：owner 授权修复后的本地验收（2026-09-14 15:51 CDT）

修复前版本保存在 git index；本轮修复全部在工作区。完整复审结论与授权边界见 `logs/rit_loto_calibration_plan.log.md` 的 G2 R2。下表只对应本轮实际修改的新包与交付测试；共用 server 的 LOTO 变化为冻结 eager stage-1 守卫，其他并行工作保留。

| 文件 | 本轮最终 sha256 |
|---|---|
| `exp/rit_loto/__init__.py` | `7caae508df46c66def005598a9ba91b495269a5db53c7db692f1d037a5cb5340` |
| `exp/rit_loto/build_loto_table.py` | `319bf0026500c7f92030405d38dce5afe02e7c31a67ec9f2563825833f3533d3` |
| `exp/rit_loto/emit_verify_arm.py` | `ec14362bcb65136e1fa0bd7dad4e74bcbba934426cab2c9f8ce6948452113194` |
| `exp/rit_loto/fit_loto.py` | `7b8baa80095dbc8facacd5535d5bd9d039f7ad9d2bbff137eeb3a6cbaf28ec81` |
| `exp/rit_loto/loto_logger.py` | `6b049d253fda2d0ae4179b4f2edc53a037a3a9bed22769292e403e782a2aca62` |
| `exp/rit_loto/noise_floor.py` | `4eab643936c6535aaa2e96a0b7355f48f79617ed0db7db139595e6a916311601` |
| `exp/rit_loto/verify_closed_loop.py` | `fb79a39289bf0c1b28bc559b15373bfea98fae25fb3ad2cbc709bf7f7f18445d` |
| `tests/exp/test_rit_loto.py` | `85ca49ef39dff602e4d2cb2bdd8c21efc63587687f2d3f971cf29d5049fdd0eb` |

- 组合回归（LOTO + 复审探针 + `tests/libero_groot` + `tests/cache/groot`）：**500 passed、5 skipped**，65.01 s。
- 最后准入修复后定向复验：**80 passed**（60 交付测试 + 20 复审探针），14.82 s。
- 真实旧 shadow 重建：spatial 3,432 行、libero_10 8,383 行，各 **48 臂 / 96 cuts** 精确一致。
- ruff、四个启动脚本 bash 语法、diff 空白检查、新包公共 docstring AST 检查通过。
- 完整 collect→label→report 命令链使用 50 集/56 决策合成日志，GPU/retrieval 运算由 stub 提供，正确报证据不足；这不是远端模型/闭环证据。

正式 noise-floor 必须达到 500/suite、50/任务；正式拟合必须是完整 500 集表，单个 task shard 不可直接使用。checkpoint 准入每次重读内容核 SHA；旧缓存元数据不能替代内容。因源码变动，§3 的远端历史冒烟需在最终版本上重跑，正式实验仍待 Verify 后执行。
