# R17 integration review B，第 3 轮：复核 Fix round 2，并直接修复剩余问题

日期 2026-10-09，分支 `Ziyang`。按 owner 指示，本轮审查结束后由我直接改代码，改到我自己同意为止。写入范围限于 `exp/r17_integration/models/{fastwam,dp}` 及其测试。没有改 common、benchmarks、conductor、qualification、四个封存的 src 文件、R16/harness 或 `exp/mw_finetune`；`git status src/` 为空。只用 CPU，不用 GPU，不 commit，不碰任何进程。

脚本和输出放在 `repro_B3/`，scratch 在 `/tmp/reviewB3`。

## 1. REVIEW_B2 发现的复核

| # | 结论 | 证据 |
|---|---|---|
| N1 qualification 脚手架（Q 负责） | **已由 Q 修复（CPU 层面；最后复核于 15:16）** | LIBERO 观测改为从 LeRobot LIBERO parquet 中抽取的原始 256² 成对帧，每帧有 hash（`qualification/recorded_libero.py`，不缩放、不合成腕部视角）；`run_slot.py:212` 对 LIBERO 断言 256；DP 判定改为 `status=='pass' and skipped==0 and passed==len(checks)`（`:351-354`）；两个 B2 槽 `batch2:false`，`native_command` 遇到 Fast-WAM B>1 直接抛 Partial（`:132-133`）；`model_runtime.py:77` 缺腕部视角就报错，不再合成。本轮实跑：Q 的 `recorded_libero.validate` 对 `libero_original256_fix2_v1` 返回 PASS；Q 的真实 `specs/libero_benchmark_spec.json`（官方 40 任务、wire 256）被 Fast-WAM 和 DP 的 spec 校验接受（`repro_B3/q_end_check.py`）。 |
| N2 Fast-WAM 部署时的 debug 采集 | **已修复** | `serve.py:34-35,70-86`：`--debug-directory` 与 `--capture-sink` 互斥，走按连接绑定的 `make_capture_sink_factory`，在 finally 中关闭。 |
| N3 模型协议与 P2 spec 的交叉校验 | **已修复** | `protocols.py:48-116`：加载前用 P2 文档重建 spec，校验 identity 和 task table，并逐项比对 replan、settle、wire、caps、seed、seed_mode（MW 要求完全等于 canonical spec）。我用 P2 真实 `manifest()` 输出和 Q 的真实官方 spec 各验了一遍，都通过。 |
| N4 snapshot scope 只看 prompt | **已修复，但修复带来了成本回归，本轮由我修正（F1）** | scope 改为对 prompt、CPU 图像和 proprio 求 SHA，修复本身正确；但这个 hash 在每次 stage1 都算，包括 HIT 和不开 capture 的 MISS：LIBERO **3.66 ms**、MW **1.84 ms** CPU（`repro_B3/scope_cost.py`），正好加在 B4 刚优化过的 key 路径上。 |
| N5 measure 的口径 | **已修复** | `measure.py`：`key_ms=prepare+stage1`，另报 `hit_path_ms=key+wire` 及两种占比；retrieval 记 null 并注明排除；有确定性时钟测试。 |
| N6 DP K1 只标注没拦住 | **已修复** | `admission.py` 通过 P1 hook 拒绝未标注的 K1；标注的 K1 只能是 plain、commit5 的 study，不能带 CP1/recipe/teacher；K1 teacher 被拒。我独立用真实 P1 binding 探测（`repro_B3/k1_gate.py`）：未标注拒、标注且 commit5 接受、commit10 拒、带 CP1 拒、K2 正常。 |
| B6 Fast-WAM loader 等价性 | **CPU 部分已修复；数值等价性仍待 GPU** | 重跑 `loader_parity --header-only`：PASS，1,651 个参数形状、24 个推理配置字段、dtype 全部一致，0 次读取 tensor 本体（`repro_B3/b6_loader_header.json`）。读过 GPU 比对路径（strict load、tokenizer 和 mask 精确比对、有限容差），没发现缺陷，但 CPU 上无法执行。 |
| B11 合成腕部视角 | **已修复** | P4 的 `fixtures.py` 写入并读回真实成对 96² 帧；Q 侧的修复见 N1。 |
| B3 残留（采样器族） | **说明，不是缺陷** | 降步档是确定性 trailing DDIM，基线是随机 DDPM100；P4 handback 已写明，报告时必须同时说明步数和采样器族都变了。 |

## 2. 本轮新发现与修复（全部由我修复，并附回归测试）

| # | 严重度 | 问题 | 证据 | 修复（文件） | 回归测试 |
|---|---|---|---|---|---|
| F1 | minor（成本回归，由 N4 修复引入） | Fast-WAM 的 conditioning-scope hash 在每次 stage1 都执行，HIT 和不开 capture 的 MISS 也一样：LIBERO 3.66 ms、MW 1.84 ms CPU，而它只在创建 snapshot 或校验 resume 时才需要。 | `repro_B3/scope_cost.py`。 | `models/fastwam/staged.py`：拆出 `_scope_digest`；新增 `stage_conditioning_scope(data)`，惰性计算并按 stage 缓存；stage1 只保存请求自有的 CPU 输入引用（`conditioning_source`），并提前检查它们在 CPU 上；`resume` 和 `_snapshot` 改用惰性 scope。`models/fastwam/batcher.py`：warm 路径同样改用惰性 scope。scope 的取值和协议（`r17.fastwam.conditioning.v2`）都没变。 | `tests/models/fastwam/test_fastwam_review_b3.py`（5 项）：HIT/key 和 capture-off MISS 的 hash 调用为 0；captured 时每个请求只算 1 次，且等于 eager 值；独立准备的相同观测可以精确 resume；state 变了即使 prompt 相同也在调用 head 之前被拒；GPU 上的输入在 stage1 被拒。在内存中重新插入旧的 eager hash：2/2 失败（`repro_B3/neg_f1.py`）。修复后 key 路径 hash 调用次数为 0。 |
| F2 | major（formal DP-LIBERO 加载失败） | DP-LIBERO 用 `LiberoBenchmarkSpec(**document)` 直接构造 spec，P2 sealed 出来的 manifest（`libero_bridge --output`，带 `protocol_identity`、`task_table_identity` 等附加键）会直接 **TypeError**；而 formal DP-LIBERO 又必须提供 spec。这个格式正是 conductor/worker 用的格式。另外原来没有任何 identity 校验。 | `repro_B3/dp_spec_repro.before.txt`：`TypeError: ... unexpected keyword argument 'protocol_identity'`。 | 新增 `models/dp/protocols.py`：`libero_spec_from_document` 与 P2 自己的 `load_benchmark_spec` 一致（按字段过滤、list 转 tuple，若文档带有则校验 `protocol_identity`、`task_table_identity`、`ordered_tasks`）；`models/dp/staged.py` 改为调用它。原来的 dataclass 文档同样能解析。 | `tests/models/dp/test_dp_review_b3.py`：wire 96/256 的 P2 原始 manifest 都被接受且 wire 尺寸被绑定（2 项）；伪造的三个 identity 字段都被拒（3 项）。 |
| F3 | minor（DP-LIBERO 协议绑定放行） | DP 接受任何 LIBERO spec，包括 Fast-WAM 专用的发布版参考协议（固定 seed42、replan32、spatial280）。replan5 的 DP 会静默跑出混合协议：每次都用固定噪声，client replan 也不匹配。 | `dp_spec_repro.before.txt`：released spec 被 ACCEPTED。 | 同一个 `protocols.py`：要求 DP-T 的评测协议，即 replan 等于 layout 的 5、settle 10、seed 7、paired_semantic、caps 220/280/300/520（E 当时就是 `main.py --replan-steps 5 --seed 7` 的默认设置）。不符时报错并列出不一致的字段。 | 同一测试文件（6 项，manifest 和 dataclass 两种形式各测）：released、replan1、settle0、seed8、fixed seed mode、caps 不符都被拒。重新插入旧解析方式：11/11 失败（`repro_B3/neg_f2.py`）。修复后的 repro 输出见 `dp_spec_repro.after.txt`。 |
| F4 | 附带：DP sample 重新 seal | DP 源码改动后，P4 的 `fix_round2_release_*` 固定 identity 按设计 fail closed（`identity mismatch`）。 | `fix_round2_validate_samples.py` 在旧 pin 上 exit 1。 | 新增 `models/dp/samples/review_b3_release_{libero,metaworld}_cpu/`（本轮真实 CPU probe 13/13 的输出：load_cpu.json 和 K1/2/4/10/100 YAML）；`fix_round2_validate_samples.py` 接受一个可选的 prefix 参数，默认仍是 P4 的那一套。P4 的旧 receipt 保留作为历史。 | `fix_round2_validate_samples.py review_b3_release`：**18/18 pass**（真实 475 tensor 的固定加载、10 个 YAML binding、K1 拒绝）。 |

**有争议的项**：无。

**需要 A 改的**：本轮没有必须由 A 修改的问题。可选建议：P1/P2 可以提供一个公用的 `benchmark_spec_from_document(dict)`，供 Fast-WAM 和 DP 共用；目前两边各自实现了 P2 `load_benchmark_spec` 的解析逻辑，以后 P2 改格式时容易不同步。

## 3. 最终测试结果（修复后实跑，CPU）

| 套件 | 结果 |
|---|---|
| `tests/models/fastwam` + `tests/common` + `tests/benchmarks`（.venv） | **249 passed**，0 failed（基线 244，新增 5） |
| `tests/models/dp`（DP island） | **125 passed**，0 failed（基线 114，新增 11） |
| legacy/default-off 九个目标（含 `test_default_off.py`、serving、MW env/runner、R16 conductor/groot servers），排除 EGL | **92 passed, 6 skipped, 1 deselected** |
| `test_replica_proxy.py -m env_dependent --run-manual` | **6 passed**（覆盖上面的 6 个 skip） |
| DP island 上的 legacy serving + default-off | **49 passed** |
| 真实 DP probe（CPU，strict 加载 475 个 tensor） | LIBERO **13/13**，MetaWorld **13/13** |
| DP 固定 sample 校验（`review_b3_release`） | **18/18** |
| Fast-WAM loader header 审计 | PASS，0 次读取 tensor 本体 |

## 4. 最终分模型结论

- **Fast-WAM-LIBERO**：模型侧没有剩余未修的 blocker/major/minor。分段与 vendor 逐位一致；HIT 路径只有 VAE、patchify 和 pool（FLOP 份额约 0.14），不再含 UMT5，F1 修复后也没有了 CPU hash；只接受 256 原帧；ack、身份和 explicit seed 都经 P1/P2 校验；system 和 released 协议与 P2 spec 有交叉校验；debug 按连接采集；B>1 禁用。**结论：同意进入 GPU 资格测试。** 仍待 GPU 的：raw-Wan 与发布版 loader 的数值等价性、native parity、key/full 实测、真实 LIBERO renderer 和官方 task map 的 episode。K1（一步 Euler）的质量没有 CPU 上可行的代理指标，尚未验证。
- **Fast-WAM-MW**：没有剩余问题。**结论：同意进入 GPU 资格测试和计价**（FLOP 份额约 0.09）。正式结果还要等 immutable MT50 delta export。
- **DP-LIBERO**：F2、F3 修复后，可以直接加载 P2 sealed 的 LIBERO manifest，并只接受 DP-T 自己的评测协议；K1 只作为标注过的 plain study；K2/4/10 的 trailing DDIM 相对 K100 的代理误差约 0.016–0.020（降步档质量尚未 qualified）；真实 probe 13/13。**结论：同意进入 GPU 资格测试。** 注意：降步档同时换了采样器族（DDIM 对 DDPM）。
- **DP-MW**：没有剩余问题，真实 probe 13/13。**结论：同意进入 GPU 资格测试。** 正式 SR 要等 E 的 MT50 export（目前仍是 ML45 诊断权重）。
