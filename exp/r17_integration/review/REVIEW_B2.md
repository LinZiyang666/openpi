# R17 integration review B，第 2 轮：Fast-WAM 与 DP-T 的 Fix round 1 复核

复核日期 2026-10-09，分支 `Ziyang`。对象是 HANDBACK_P1–P4 中的 “Fix round 1” 改动，以及与之相关的 P6/Q qualification 代码。按 REVIEW_B 的 B1–B11 逐项复核，再查修复是否引入了新的、会改变行为的缺陷。规则同第 1 轮：只用 CPU，不改代码，不用 GPU，不碰进程，不在 GPU 上加载完整 checkpoint。

**本轮实际运行**（脚本和输出见 `repro_B2/`，scratch 在 `/tmp/reviewB2`）：

- `tests/models/fastwam` 和 `tests/common`：**147 passed**；`tests/models/dp`（DP island）：**79 passed**；`tests/benchmarks`：**27 passed**。
- 真实 DP probe（CPU，各自 strict 加载 475 个 tensor）：LIBERO **13/13 pass**，MetaWorld **13/13 pass**。
- `repro_B2/libero_context_repro_r2.py`：用修复后的 P1 `ConnectionPolicy`（非 diagnostic 模式）重放第 1 轮的 LIBERO 场景，再按 P2 bridge 的身份和 ack 格式各跑一遍。
- `repro_B2/dp_lowk_r2.py`：与第 1 轮相同的 48 个真实成对 LIBERO 训练帧和相同的 seed0 B48 噪声。K100 用 E 未改动的 DDPM；K∈{1,2,4,10} 用 adapter 自己的 `SamplerProfile`/`step`，并同时与 diffusers 的 trailing DDIM、`DPAdapter.sample_batch` 逐位比对。
- `repro_B2/ddim_nondivisor_oracle.py`：精确 MMSE oracle 检查新的 DDIM 更新式，K 取 1/2/3/7/10/13/29/99/100。
- `repro_B2/fastwam_flops.py`：在 meta 设备上（不加载权重）对 raw Wan2.2 VAE38 编码和 UMT5-XXL encoder 计 FLOP，填上 B4 成本拆分里原来未知的 VAE 项。

## 1. 第 1 轮发现的复核

| # | 原严重度 | 结论 | 证据（本轮实测或通读代码） | 残留 |
|---|---|---|---|---|
| B1 LIBERO commit 静默降级 | blocker | **已修复（CPU 层面）** | P1 `serving.py:169-170,277-278,284-285`：对 extended commit，episode 必须声明 `ack_protocol=r17.execution.v1`；缺 ack 时直接报错，不再静默作废 chunk。`episode_end` 要求最终 ack（`:185-186`）。P2 `libero_bridge.py:232-279` 包住原 kernel，每次 infer 带上一决策的实际执行步数，按服务端返回长度执行 `min(replan, len)` 步，最终 ack 走 `episode_end`。重放结果：旧 runner 元数据加 commit10 在 `episode_start` 就报 `ack_protocol` 错；使用 bridge 格式时第 2 次决策为 `policy_tail (0,0,0) cursor=5`，第 3 次漏 ack 报错 `cannot silently downgrade`。P2 测试覆盖 commit 10/15/32 的原 LIBERO kernel 加真实本地 WS（27 passed）。 | 尚未在真实 LIBERO renderer 和真实模型上跑过。 |
| B2 LIBERO 身份/任务号/conductor | major | **已修复（CPU 层面）** | P1 `serving.py:161-166`：识别 `orig_init_state_idx`；`:202-207`：非 diagnostic 模式缺 campaign/task/原始 idx/seed/explicit_seed 时报错。P2 `LiberoBenchmarkSpec` 提供 40 个全局 task id 和 `suite/task` 规范名，per-decision 的 `explicit_seed=sha256(libero:seed:task:idx:decision)`；driver 和 worker entry 有了 LIBERO 分支。重放：init 0 和 7 的 seed 分别为 `961241716` 和 `2566406359`。descent 代码与 benchmark 无关（只是类名仍叫 MetaWorld*）。 | 官方 task map 要在 LIBERO island 里导出；LIBERO 物理采集明确不支持；LIBERO descent 没有专门测试。 |
| B3 DP K1/低 K 网格 | major | **数学已修复；K1 仍不可用，只是被标注** | `sampler.py:20-31,45-51,73-93`：K<100 改为 trailing DDIM、eta 0；每个 K 恰好 K 个点（K29 也是）；predecessor 用真实网格（K 不整除 100 时也正确）。显式请求 reduced DDPM 或 leading DDIM 会被拒绝。**按要求重测（`dp_lowk_r2.out.json`）**，单位是外部 quantile 归一化动作，前 5 步：K1 `[99]` 相对 K100 的 MSE **1.767**，相对数据集 1.948，内部 clip 比例 98.3%；K2 `[99,49]` **0.0200**/0.228；K4 `[99,74,49,24]` **0.0160**/0.235；K10 **0.0190**/0.240；K100 相对数据集 0.239。四个 K 的结果都与 diffusers trailing DDIM 以及 `DPAdapter.sample_batch` 逐位相等，与 P4 的表完全一致。nondivisor oracle：所有 K 的均值都是 0.400，没有偏差。 | K1 仍然实际不可用（见 N6），`lossy` 标签没有任何地方强制执行。另外，降步档是确定性 DDIM，基线是随机 DDPM100：比较时步数和采样器族同时变了，报告里要写清楚。 |
| B4 Fast-WAM key 含 UMT5 | major | **已修复** | `staged.py:89-129`：stage1 只做 VAE→`patchify`→pool，锁里不再有文本编码；`:137-166`：UMT5 和 proprio 条件、未改动的 vendor `pre_dit`（通过 `_ReusePatch` 复用已算好的卷积结果，没有替换任何常驻方法）、mask、prefill 都移到 stage2，只在 MISS 时执行。通读代码确认 `patches.flatten(2).transpose(1,2)` 等同 vendor 的 `rearrange`，dtype 转换和 timestep 也相同；P3 的 `test_b4_*` 在真实小 torch 模块上逐位对比通过。**重新推导成本拆分**（FLOP 计数：VAE 和 UMT5 用 FlopCounter 实测，DiT 用参数×token 估算）：VAE 首帧 LIBERO 224×448 **0.466** TFLOP、MW 224² **0.233**；UMT5 128 token **1.192**；video prefill ≈0.98（LIBERO）/0.49（MW）；K10 action ≈0.653；patchify 约 0.0001。key/full：**LIBERO 0.50→0.14，MW 0.55→0.09**。按 FLOP 计，每次 HIT 省掉整次调用的 86%（LIBERO）/91%（MW）。 | 这是 FLOP 份额，不是延迟。B=1 下 DiT 和 T5 主要受权重带宽限制，VAE 主要受激活限制；实际比例要等 `measure.py` 的 GPU 槽。另见 N5（measure 的口径）。 |
| B5 LIBERO 输入分辨率 | minor | **已修复** | Fast-WAM `wire_policy.py:110-111` 只接受 256²，每个视角用 torch antialias 缩到 224；DP `staged.py` 只接受成对的 96² 或 256²，并与 P2 spec 的 `wire_size` 绑定（formal LIBERO 必须提供 spec）。P2 runner 要求 `resize_size==wire_size`；`resize_with_pad` 在 256 时原样返回（`image_tools.py:28-29`），96 时与 E 的 PIL 路径完全相同。 | 无。 |
| B6 Fast-WAM 发布版协议 / loader | minor | **部分修复** | `protocols.py:40-43` 新增显式的 `released_libero` profile（replan/commit 32、K10、每次 seed 42、settle 10、caps 280/280/300/520、256 原帧），作为独立的 layout 和 identity；`wire_policy.py:96-97` 拒绝非 42 的 seed，`staged.py:76-77` 拒绝 K≠10。P2 有对应的 `--released-reference` spec。 | raw-Wan 与发布版 sidecar/Diffusers loader 的等价性仍未验证（metadata 已明确标出）。另见 N3。 |
| B7 能力元数据高报 | minor | **已修复** | Fast-WAM `manifest.py:75-83`：cp2 和 native warm 改为 `implemented_unverified`，batch_stage* 改为 `unsupported`。DP `staged.py`：warm、library、recipe 改为 `implemented_unverified`，k_step 注明 K1 lossy。 | 无。 |
| B8 DP 服务入口缺 hook | minor | **已修复** | DP `serve.py:54-57,74-85` 有了 `--recipe-factory`、`--capture-sink`、`--debug-directory`；默认的 per-bundle recipe factory 在 `provider.py:36`。 | 无。（Fast-WAM 侧见 N2。） |
| B9 采集影响计时 | minor | **已修复** | Fast-WAM 只有显式 `capture_cp2=True` 才计算 CP2；captured 结果标 `pricing_eligible=False`（`staged.py:175,262-264`）。DP 的 conditioning digest 每个请求只算一次（`staged.py:218`）。P1 ledger 对任何 capture 都标为不可计价（`serving.py:343-344`）。 | 无。 |
| B10 Fast-WAM B>1 | minor | **已修复（禁用）** | `serve.py:51-52` 拒绝；`provider.py:11-12` 拒绝 coordinator；adapter 每个 stage 都调用 `_check_batch`。 | qualification 的 B2 slot 没有同步更新（N1）。 |
| B11 DP-LIBERO 合成腕部视角 | minor | **部分修复** | `fixtures.py` 从 `libero_96` 读真实成对 PNG；probe 和 quality 都改用它（probe 检查 `real_paired_LIBERO_fixture` 通过）。 | `qualification/model_runtime.py:75` 仍用 `image[::-1]` 合成腕部视角（见 N1）。 |

## 2. 新发现

| # | 严重度 | 位置（file:line） | 问题 | 证据 | 修复 |
|---|---|---|---|---|---|
| N1 | **major（qualification 结论）** | `qualification/assets.py:177-197`；`qualification/run_slot.py:196,405,409-410`；`qualification/specs/slot_G2_fastwam_{libero,mw}_C_B2.json:4`；`qualification/model_runtime.py:75` | Q/P6 的 qualification 脚手架没有跟上 P3/P4 的修复，接下来的 GPU 槽会必然失败或误报失败：(a) `assets.observations` 对 LIBERO 也生成 480² 合成帧，`run_slot` 还断言 480²。Fast-WAM-LIBERO 的 native 槽（`qualify`→`prepare`）会因 “LIBERO expects original 256x256” 失败；DP-LIBERO 的 episode/G3 阶段会因 96/256 检查拒绝这些帧。(b) DP native 槽断言 `passed==(12 if MW else 11)`，但 probe 现在两个域都是 13 项，**probe 通过也会被判 FAIL**。(c) 两个 `*_C_B2` 槽仍设 `batch2: true`，而 B>1 已在 `qualify.py:24-25` 被禁止，会立即失败。(d) DP-LIBERO 仍用合成腕部视角，等于把 B11 带回 G3。 | 本轮 CPU 真实 probe：LIBERO 和 MW 都是 `pass 13/0`；断言、spec 和资产代码见左列（`run_slot.py` 的修改时间是 14:11，仍是 12/11）。 | 用真实成对帧按 sealed wire 尺寸重新生成 LIBERO 资产（Fast-WAM 256，DP 96/256），可以直接用 P4 的 `fixtures.libero_observation`；DP 判定改成 `status=="pass" and skipped==0`，不要写死数目；删除或重新标注 B2 槽；`model_runtime` 去掉合成腕部视角。下一个 LIBERO/DP 的 GPU 槽之前完成。 |
| N2 | minor（Fast-WAM 采集部署） | `models/fastwam/serve.py:31-32,67-68` | π0.5、GR00T、DP 的入口都有 `--debug-directory`，内部用 `make_capture_sink_factory(provider, dir)`，按连接绑定实际 sampler。Fast-WAM 只有进程级的 `--capture-sink module:function`：一个普通函数没法在每个连接上构造 P6 的 writer，按 P1 handback，K2/K1 bundle 的 capture 就无法绑定到实际 sampler。部署后的 Fast-WAM debug 采集要么做不了，要么绑错 grid。 | `grep capture_sink_factory` 在 Fast-WAM 目录无结果，而 `pi05/serve.py:23,53`、`groot/serve.py:86-92`、`dp/serve.py:57,77,85` 都有。 | 照 DP 的写法加 `--debug-directory`（与 `--capture-sink` 互斥），把 `capture_sink_factory` 传给 P1 factory，服务退出时 close。 |
| N3 | minor（Fast-WAM-LIBERO 协议绑定） | `models/fastwam/protocols.py:36-43`；`models/fastwam/loader.py:236-243` | Fast-WAM 模型侧的 `ExecutionProtocol`（replan、settle、caps、wire、seed 规则）和 P2 的 `LiberoBenchmarkSpec` 之间没有任何交叉校验，两者只靠 allowlist 里的 protocol identity 字符串联系。例如 system 模型配上 released spec，会静默跑成 seed42 固定、replan5 的混合协议。另外，system profile 的 `seed_rule="semantic_request_rng"` 已经不对：P1 现在要求 explicit_seed，实际噪声来自 P2 的 `decision_seed`。DP 至少把 wire 尺寸绑到了 spec（`libero_benchmark_specs`）。 | 在 conductor、benchmarks、common、qualification 里 grep `execution_protocol`/`suite_control_caps` 都没有结果。 | 加载时要求传入 P2 spec 文档（与 DP 的 `libero_benchmark_specs` 一样），断言 replan、settle、caps、wire、seed_mode 一致；把 seed 标签改为 `explicit_benchmark_seed`。 |
| N4 | minor（第 1 轮漏掉的已有问题） | `models/fastwam/staged.py:124,196` | Fast-WAM native snapshot 的 `conditioning_scope` 只是 `sha256(prompt)`，而 stage2 的条件（KV 和 proprio token）还取决于图像和 state。`resume()` 会接受同一 prompt 下来自另一个观测的 snapshot，静默拼出一条混合轨迹。DP 用的是 conditioning 张量的精确 digest。目前只有直接调用 native API 时可达（通用执行器拒绝 warm 路径）。 | 代码路径如左。 | 把 scope 改成 digest(prompt, proprio, patch tokens)，或者直接对 KV/context 做 digest。 |
| N5 | minor（计价口径） | `models/fastwam/measure.py:76,81` | `key_over_full = key_ms/full_ms` 的分子不含 `prepare_ms`（LIBERO 在 CPU 上做两次 antialias resize），也不含 retrieval 和 `to_wire`，而这些 HIT 都要付。结果会低估 HIT 的成本比例。 | 代码路径如左。 | 报 `hit_path_ms = prepare + stage1 + to_wire` 及其比例；正式计价以 P1 ledger 的每决策 host wall 为准。 |
| N6 | minor（DP K1 只标注、没拦住） | `models/dp/sampler.py:65-68`；`models/dp/staged.py:78` | `quality_status="lossy"` 只出现在 sampler config 和 capability 文本里，binding、bundle validator 和 descent 都不读它。K1 bundle 照常绑定，可以进入 plain 或 descent arm。真实帧代理指标：相对 K100 的 MSE 1.77，98.3% 的内部值顶到 clip 边界，基本相当于噪声。 | `grep quality_status\|lossy` 的唯一消费者是 P4 自己的 sample 校验脚本。 | 只有 YAML 显式写 `allow_lossy_sampler: true` 才接受 K1，否则拒绝；DP 的 descent 档位从 K2 开始。 |

本轮没有发现修复改变了 Fast-WAM 或 DP 在 MW、LIBERO 上 full/K 档的动作数值：Fast-WAM 的 stage 切分与 vendor 逐位一致，DP 的 K100 与 E 逐位一致。

## 3. 更新后的分模型结论

- **Fast-WAM-LIBERO**：B1、B2、B4、B5、B7、B9、B10 已在 CPU 层面修复：key 不再含 UMT5（FLOP 份额约 0.14）；LIBERO 有 ack、身份和 explicit seed；commit 10/15/32 可协商；输入只接受 256 原帧。发布版协议成为一个独立、带标签的 profile，但 loader 与发布版的等价性未验证。仍待：N1（其 qualification 槽会因 480² 帧和 B2 设置必然失败）、N2（部署时没法按连接采集 debug）、N3（模型协议与 P2 spec 没有交叉校验）。**结论：CPU 上已可接 commit/五层/descent；第一个 GPU 槽之前必须修好 N1；GPU parity 和真实 renderer 仍未验证。**
- **Fast-WAM-MW**：B4 已修复（FLOP 份额约 0.09），本轮没有发现 MW 特有的新问题；P2 runner 的 ack 和 context 路径不变。N2、N5 同样适用，N1(c) 也影响它的 B2 槽。**结论：接入正确，可以进 GPU 计价和资格测试；用 measure.py 实测 key/full 时注意 N5 的口径。** 仍待最终 immutable MT50 export。
- **DP-LIBERO**：B3 的数学已修复，K2/4/10 的代理误差约 0.016–0.020（相对 K100）；B5、B8、B9、B11（自有部分）已修复；LIBERO 的 ack 和身份随 P1/P2 修复；真实 probe 13/13。仍待：N1（DP-LIBERO 的槽会被 12/11 断言和 480² 帧判为失败，G3 仍用合成腕部视角）、N6（K1 没被拦住）。**结论：K100 和 K2–K10 可用（降步档质量尚未 qualified）；K1 应在 binding 层拒绝；先修 qualification 脚手架。**
- **DP-MW**：B3、B7、B8、B9 已修复；真实 probe 13/13；P2 MW runner 路径不变。仍待：N1(b)（12 项断言会误报失败）、N6；正式结果要等 E 的 MT50 export（目前仍是 ML45 诊断权重）。**结论：接入正确；修好 N1(b) 和 N6 后可以进 GPU 资格测试。**

## 4. INVENTORY 不支持项的变化（相对 REVIEW_B）

- 从“不支持”变为“CPU 已支持”的：两个 LIBERO 模型的 P11/P12 commit/tail、P05/P07/P08 所需的 task/init 身份、C01–C05 的 R17 LIBERO 派发、worker 和 spec；DP 入口的五层和 debug（B8）。
- 四个模型仍然都缺：S04/S05、S06/S07（Fast-WAM 明确禁用 B>1）、S13、P04（只允许 threshold 等 stock judge）、P13/P14/P15/P16（通用执行器拒绝）、D01/D02/D03/D04/D07/D10/D12、D13/D14（还没有真实 teacher 库）、A01（没有真实价格）、X01。
- LIBERO 仍缺：D09 物理采集（P2 明确不支持）；官方 task map 和 renderer 资格。
- 各模型仍缺：Fast-WAM 的 D08 部署时按连接采集（N2）；DP 的 P10 中 K1 仍是 lossy（N6）；DP-MW 的正式 checkpoint。
