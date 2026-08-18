# Baseline provenance — RoboCasa365 orphan-file rescue (T1)

这些文件是**基线证据**，不是可运行代码：字节级照搬远端原件（不加 header、不跑 formatter），
使 T2 的等价性判据（`tests/robocasa365/test_pi05_stack_parity_manual.py`）失败时可以精确回溯到
「与哪一版基线不一致」。**不得被任何运行时代码 import**；测试只把它们当契约夹具按文本读取。

⚠ 位置说明：plan 原定落 `exp/robocasa365/data/`，但该目录被 `.gitignore:6`（`exp/**/data/**`）
整体忽略、无法入库，而修改 `.gitignore` 属高危操作需 owner 逐次同意 —— 故改落本目录
（`baselines/`，可跟踪）。此偏差在统一临时 G2 一并审。

| 文件 | 源主机 | 源绝对路径 | sha256 | pull 时间 (UTC-6) |
|---|---|---|---|---|
| `serve_robocasa_pi05_ORIGINAL.py` | weilandserver | `/home/weiland/step0b_artifacts/serve_robocasa_pi05.py` | `e125f8e648aa2c5d0432dd8843216a119638e9afab8999c39a16a05aec2d6f29` | 2026-08-17 18:00 |
| `pi05_step0b_client_ORIGINAL.py` | weilandserver | `/home/weiland/step0b_artifacts/step0b_v2.py` | `0e0e749cc8873c7ebe670576376d5e98bdd16071316b8ac54e0520e2a7dd170d` | 2026-08-17 18:00 |
| `../data/pi05_analyze_step0b_ORIGINAL.py`（先例，untracked） | weilandserver | `/home/weiland/step0b_artifacts/analyze_step0b.py` | `1480cb887e9cc7ab7b1752c2ec0c5f41a0421d0d4d67291cd2ae2fea1192ae1d` | 2026-08-16（前一轮 session） |

另一份被抢救的**活代码**（非存档，允许两处形式差异：新增模块 docstring、清理行尾空白）：

| 文件 | 源主机 | 源绝对路径 | 源 sha256 |
|---|---|---|---|
| `src/openpi/policies/robocasa_policy.py` | weilandserver | `/home/weiland/openpi/src/openpi/policies/robocasa_policy.py`（untracked） | `60af69ba36d36c6743d5a594c83c67efcbc4e2f5d4d698f1d426371c8f746bd0` |

校验方式：`sha256sum exp/robocasa365/baselines/*_ORIGINAL.py` 与上表逐条比对。
