# Pi0.5 High-Level Autoregressive Decode — 实施计划

**Status**: Plan
**Level**: L2
**Authority**: Execution
**Author**: Execution Authority session
**Target branch**: feature branch（待定）
**Relates to**: openpi 上游 issue 建议 "run the PaliGemma.llm module on your high level prompt and start generating the low-level command"；π0.5 论文 §V.E / Fig 13 implicit-HL vs explicit-HL 消融

---

## 1. Background

### 1.1 当前推理路径

`PI0Pytorch.sample_actions` (`src/openpi/models_pytorch/pi0_pytorch.py:704`) 只走三段 pipeline：

1. Stage 1 `_stage1_token_prep`：SigLIP 编图像 + Paligemma embed 语言 → prefix tokens。
2. Stage 2 `_stage2_llm_backbone`：LLM backbone 一次 bidirectional prefill，`use_cache=True`，产出 KV cache。
3. Stage 3 `_stage3_action_expert`：action expert 读 KV cache，在 suffix（noisy action + time）上跑 10 步 Euler flow matching。

Stage 2 **不使用** `paligemma.lm_head`，不做任何 autoregressive 解码。π0.5 论文提到的"VLM 生成 low-level language command"对应的是 Knowledge Insulation 训练阶段 text loss；openpi 公开推理路径把它省略了（README 明确写 "we currently only support the flow matching head for both π0.5 training and inference"）。

### 1.2 可行性基础（已核查）

- `PaliGemmaForConditionalGeneration` 在 `gemma_pytorch.py:57` 被完整实例化，`paligemma.lm_head: nn.Linear(width, 257152)` 存在。
- `stage_device_placement.py:40-41, 285-295` 验证了 `lm_head.weight is embed_tokens.weight`（tied）。`embed_tokens` 在 Stage 1 被调用，必然从 checkpoint 载入，所以 `lm_head` 自动有可用权重。
- `embed_prefix` (`pi0_pytorch.py:284-333`) 采用的 PrefixLM mask 约定（prefix 全 `att_masks=0`，suffix 开 `[1]` 新 block）由 `make_att_2d_masks` (`pi0_pytorch.py:149-178`) cumsum 构造——**这恰好是 PaliGemma 预训练和 π0.5 KI text-loss 的监督结构**，因此 prefix bidirectional + AR suffix causal 语义上**没有冲突**。

### 1.3 为什么需要新函数

`_stage2_llm_backbone` 是一次性 prefill，形状为 `[B, 1, N, N]` 方阵 mask + `position_ids=[B, N]`。增量 decode 每步只喂 1 token，需要 `[B, 1, 1, cache_len+1]` 行 mask + `[B, 1]` position id，**不能复用**该函数。但 prefill 产出的 `DynamicCache` 可以作为 decode 起点直接复用。

### 1.4 ROI 判断

π0.5 论文 Fig 13 显示 implicit HL 已经接近完整性能；LIBERO 是 single-task 短周期场景，HL 预期收益比论文 setting 更小。因此本计划**以 Phase A probe 为 gate**：先验证加载的 checkpoint（`pi05_base` 或 `pi05_libero`）的 `lm_head` 是否能产生有意义 text，再决定是否进入 Phase B 完整集成。Phase A 不过 → Phase B 不做。

---

## 2. Goal

在 π0.5 PyTorch 推理路径上新增**可选**的 HL autoregressive decode 模式，默认关闭，行为与当前一致；开启时：

1. 用高层 prompt + images 触发 `paligemma.language_model` + `lm_head` 做贪心/top-k 自回归解码，产出 low-level command token 序列。
2. 把生成文本拼回 prompt 模板，再走完整 Stage 1→2→3 flow matching。
3. 所有对外 API 保持向后兼容。

---

## 3. Scope

### In scope

- 新方法 `PI0Pytorch.run_stage2b_ar_decode`（incremental decode）。
- 新辅助 `_decode_one_step`（单步 forward + logits 采样）。
- 新数据结构 `Stage2bOutput`。
- `Pi0Config` 新字段 `enable_hl_decode: bool = False`（默认 False）。
- `sample_actions` 在 `enable_hl_decode=True` 时走 Stage 1 → 2 → 2b → 重建 prompt → Stage 1' → 2' → 3 的两遍 pipeline。
- Phase A probe 脚本 `exp/hl_ar_decode_probe/probe_lm_head.py`。
- 单元测试 `tests/models_pytorch/test_pi0_ar_decode.py`。

### Out of scope

- 训练 / fine-tune `lm_head`（任何监督信号改动）。
- 对 `lm_head` 生成结果的 downstream 缓存或跨帧复用。
- 一遍 pipeline 的优化（复用 Stage 2b 的 extended KV 直接喂 Stage 3，跳过第二次 prefill）——作为 follow-up。
- batched beam search / nucleus sampling（仅实现 greedy + 可选 top-k=1）。
- JAX 路径（WA §1 已禁用）。
- 修改 LIBERO eval 脚本之外的任何 inference pipeline。

---

## 4. Phase 分解

### Phase A — Probe（pre-gate，L1 scoped within this L2）

**目的**：验证 `pi05_base` / `pi05_libero` checkpoint 的 `lm_head` 在 LIBERO 真实 prompt + image 输入下能否产出"像自然语言低层指令"的 token 序列。结果决定 Phase B 是否值得做。

**交付物**：

- 新文件 `exp/hl_ar_decode_probe/probe_lm_head.py`：
  - 加载指定 checkpoint 走现有 Stage 1 + 自定义 AR loop。
  - 输入 5-10 条 LIBERO task prompt + agentview/wrist image。
  - greedy 解 ≤ 30 token，打印原文 + 解 token id。
  - 命令行参数：`--checkpoint`, `--task_suite`, `--num_prompts`。
- 新文件 `exp/hl_ar_decode_probe/README.md`：如何运行 + 示例输出。
- 输出样例 dump：`exp/hl_ar_decode_probe/data/sample_decodes.txt`（run 后产出，不入 git，`.gitignore`）。

**Gate 条件**：user 审阅 probe 输出。若生成的文本包含运动类动词 / 空间方位 / 与 prompt 语义相关的词，判为 pass，进入 Phase B。若是无关 pretrain 分布漂移（乱码、无关 token 拼凑），判为 fail，归档本 plan 为 `Historical`，整个改动终止。

**Phase A 不改动**：`src/openpi/` 下任何文件。只新增 `exp/` 脚本和 README。

### Phase B — Integration（Phase A 通过后启动；本 plan 的 L2 主体）

#### B.1 新增推理态数据类

**文件**：`src/openpi/models_pytorch/pi0_pytorch.py`

```python
@dataclass
class Stage2bOutput:
    """Output of Stage 2b: autoregressive text decode from Stage 2 KV cache."""
    stage2: Stage2Output
    generated_tokens: torch.Tensor   # [B, T_gen] (int64) on model device
    generated_text: list[str]        # per-batch decoded text
    extended_past_key_values: Any    # DynamicCache containing prefix + generated tokens
    prefix_len: int                  # length of the original prefix (before decode)
    generated_len: int               # T_gen
```

#### B.2 新增核心方法

**`_decode_one_step`**（私有 helper）：

- 输入：`past_key_values`, `last_token_id: torch.Tensor[B, 1]`, `cache_len: int`
- 动作：
  1. `new_emb = self.paligemma_with_expert.embed_language_tokens(last_token_id) * sqrt(D)`
  2. `position_ids = torch.full((B, 1), cache_len, device=..., dtype=long)`
  3. `attention_mask_4d = torch.zeros(B, 1, 1, cache_len+1, dtype=float32)` （0.0 表示全可见；因为每步只喂 1 token，"当前 token 不能看未来"的约束自动满足）
  4. 调用 `self.paligemma_with_expert.forward(inputs_embeds=[new_emb, None], attention_mask=attention_mask_4d, position_ids=position_ids, past_key_values=past_key_values, use_cache=True)`
  5. 取 `hidden = prefix_output[:, -1]`，经 `self.paligemma_with_expert.paligemma.lm_head` 得 logits
  6. 返回 `(logits, updated_past_key_values)`

**`run_stage2b_ar_decode`**（公开）：

- 签名：`run_stage2b_ar_decode(stage2: Stage2Output, *, max_new_tokens: int = 20, eos_token_id: int = <tokenizer.eos>, stop_token_ids: tuple[int, ...] = (), greedy: bool = True, temperature: float = 0.0, top_k: int = 0) -> Stage2bOutput`
- 动作：
  1. `cache_len = stage2.stage1.prefix_embs.shape[1]`；起始 `past_key_values = stage2.past_key_values`
  2. 首 token 来源：用 tokenizer 编码 `"\nAction: "` 的首 token？**待定：在 Phase A probe 时确定合适的 "开始自回归"初始 token**；若 prompt 已以 `"Action:"` 结尾则不需要再塞，直接对 `past_key_values` 调用 lm_head 拿 prefix 最后一个位置的 logits 作为第一步 next-token 分布。
  3. 循环最多 `max_new_tokens` 步调用 `_decode_one_step`；采样策略分支 greedy / top-k。
  4. 命中 EOS / stop token 或 `";"` 分隔符时提前退出。
  5. 收集 `generated_tokens`，用 `PaligemmaTokenizer.decode`（新引入调用）还原为文本。
  6. 返回 `Stage2bOutput`。

**注意**：`_stage2_llm_backbone` 在首次 prefill 时硬写 `_attn_implementation = "eager"`；decode loop 必须保持同一实现，不得切换为 flash / sdpa（否则 4D 加性 mask 路径失配）。

#### B.3 把 Stage 2b 接入 `sample_actions`

**文件**：`src/openpi/models_pytorch/pi0_pytorch.py`

```python
@torch.no_grad()
def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
    if not self.config.enable_hl_decode:
        # existing path — UNCHANGED
        ...
        return self._stage3_action_expert(...)

    # new HL-decode path (two-pass prefill)
    stage1 = self.run_stage1(observation)
    stage2 = self.run_stage2(stage1)
    stage2b = self.run_stage2b_ar_decode(stage2, ...)

    # rebuild observation with enriched prompt, run Stage 1-3 again
    enriched_obs = self._inject_hl_text(observation, stage2b.generated_text)
    stage1b = self.run_stage1(enriched_obs)
    stage2c = self.run_stage2(stage1b)
    return self._stage3_action_expert(
        stage1b.state, stage1b.prefix_pad_masks, stage2c.past_key_values, noise, num_steps
    )
```

**`_inject_hl_text`**（新 helper）：接受原 observation 和生成的 HL 文本，按 π0.5 tokenizer 约定 (`src/openpi/models/tokenizer.py:22-28`) 的模板拼装新 prompt：

```
Task: {original_task} {hl_command}, State: {state};
Action:
```

通过 `PaligemmaTokenizer.tokenize` 重新得到 `tokenized_prompt` / `tokenized_prompt_mask`，替换到 observation 中。

**注**：两遍 prefill 有明显 latency 代价（约 2x Stage 1+2）；文档和 docstring 中明确标注。一遍版本作为 follow-up。

#### B.4 Config 开关

**文件**：`src/openpi/models/pi0_config.py`（或 `pi0_config.py` 对应位置；以实际存在的为准）

新增字段：

```python
enable_hl_decode: bool = False
hl_max_new_tokens: int = 20
hl_greedy: bool = True
```

默认全部 False / 保守值，保证现有配置 zero impact。

#### B.5 LIBERO CLI 开关（可选，默认不开启）

**文件**：`examples/libero/main.py`

`Args` 新增 `enable_hl_decode: bool = False`；在 policy 构造处转发到 `Pi0Config`。只影响 client 侧；server 端模型 config 决定是否启用（client 侧 flag 仅当独立加载模型的 collect/probe 路径使用）。

> 若现有部署架构中 client 不直接构造模型（WebSocket policy），则此 flag 仅作为 probe / 离线评估入口。最终 wiring 细节在 Phase A 后与 user 确认。

---

## 5. Files Touched

| 路径 | 改动类型 | 说明 |
|------|---------|------|
| `src/openpi/models_pytorch/pi0_pytorch.py` | 修改 | 新增 `Stage2bOutput`、`_decode_one_step`、`run_stage2b_ar_decode`、`_inject_hl_text`；`sample_actions` 加 branch |
| `src/openpi/models/pi0_config.py` | 修改 | 新增 3 个 HL 字段（路径以实际为准，Phase B 开工前定） |
| `exp/hl_ar_decode_probe/probe_lm_head.py` | 新增 | Phase A probe |
| `exp/hl_ar_decode_probe/README.md` | 新增 | probe 使用说明 |
| `exp/hl_ar_decode_probe/.gitignore` | 新增 | 忽略 `data/` |
| `tests/models_pytorch/test_pi0_ar_decode.py` | 新增 | 单元 + smoke 测试 |
| `examples/libero/main.py` | 修改（可选） | 新增 `--enable_hl_decode` flag（仅本地 probe 用） |
| `logs/README.md` | 修改 | 登记本 plan |
| `docs/README.md` | 不动 | 不新增架构文档（单方法扩展，按 §9 "Minor change derivable from code → do NOT create a document"） |

---

## 6. Interfaces Introduced / Modified

### 新增

- `PI0Pytorch.run_stage2b_ar_decode(stage2, *, max_new_tokens, eos_token_id, stop_token_ids, greedy, temperature, top_k) -> Stage2bOutput`
- `PI0Pytorch._decode_one_step(past_key_values, last_token_id, cache_len) -> (logits, past_key_values)`
- `PI0Pytorch._inject_hl_text(observation, hl_text) -> observation`
- `Stage2bOutput` dataclass

### 修改

- `PI0Pytorch.sample_actions`：增加 `enable_hl_decode` branch（默认关闭保持现有行为）。
- `Pi0Config`：新增 3 个字段，全部 default False / 保守值。

### 不修改

- `PaliGemmaWithExpertModel.forward` 签名 — decode loop 通过现有 `[prefix, suffix]` 分支结构，`inputs_embeds=[new_token_emb, None]` 形式复用。
- `embed_prefix` / `embed_suffix` 签名。
- Stage 3 `_stage3_action_expert` / `denoise_step` 完全不动。

---

## 7. Integration Points

1. **KV cache 复用**：`run_stage2` 产出的 `DynamicCache` 是 decode loop 的起点；HF DynamicCache 的 `update(k, v, layer_idx, cache_kwargs)` 约定由 eager attention 路径调用，`paligemma_with_expert.forward` 已经正确传递 `use_cache=True` → KV append。
2. **Tokenizer**：`PaligemmaTokenizer`（`src/openpi/models/tokenizer.py`）已有 `tokenize`；新调用点需要 `decode` 能力——若现有 tokenizer 未暴露 decode，借用 `self._tokenizer.decode`（`sentencepiece.SentencePieceProcessor.decode`）即可。
3. **二次 prefill 的 observation 构造**：`_inject_hl_text` 走的是和现有训练/推理完全一致的 `PaligemmaTokenizer.tokenize` 模板，保证 Stage 1-3 对 enriched prompt 的处理和原路径对 base prompt 的处理在 tokenization / embedding 层面同构。

---

## 8. Test Strategy

**文件**：`tests/models_pytorch/test_pi0_ar_decode.py`

| 测试 | 验证点 |
|------|--------|
| `test_stage2b_output_shapes` | decode 5 步后 `generated_tokens.shape == (B, 5)`，`extended_past_key_values` 层内 K/V 长度 `== prefix_len + 5` |
| `test_decode_matches_full_prefill` | 同一 prompt + 同 3 个 greedy token，比对：(prefill → 1-step decode × 3) 的 last-token logits 与 (prefill prefix+3tokens 一次性) 的对应位置 logits，`max abs diff < 1e-3`（eager bf16） |
| `test_stop_on_eos` | 注入 mock `lm_head` 使其一步即出 EOS；`generated_len <= 1` |
| `test_disable_flag_preserves_original_path` | `enable_hl_decode=False` 时 `sample_actions` 输出与旧版 `sample_actions` 输出 bit-equivalent（固定 noise + seed） |
| `test_sample_actions_smoke_with_hl_decode`（`@pytest.mark.manual`） | 真 checkpoint 上 `enable_hl_decode=True`，batch=1，断言输出形状 `(1, action_horizon, action_dim)` 且 finite |

模型层测试使用小尺寸随机权重的 `PaliGemmaWithExpertModel`（参考现有 `gemma_pytorch.py` 测试模式），避免真实 checkpoint 依赖；端到端 smoke 打 `manual` marker 仅在 GPU 本地跑。

**Verify 阶段（§6）**：`uv run pytest` 全通过 + `test_disable_flag_preserves_original_path` 明确证明 OFF 路径零 regression；staged API 测试全通过。

---

## 9. Risk Register

| ID | 风险 | 影响 | 缓解 |
|----|------|------|------|
| R1 | `lm_head` 对 LIBERO 场景输出乱码 | Phase B 产出负收益或无收益 | Phase A probe gate；gate fail → 终止 |
| R2 | 增量 decode mask 写错导致 KV 污染 | 生成文本错误、动作退化 | `test_decode_matches_full_prefill` 严格比对一次性 prefill |
| R3 | `_attn_implementation` 切换 / dtype 漂移 | 数值不稳或报错 | decode loop 全程复用 Stage 2 的 eager path，不切换；bf16 混合精度由 `to_bfloat16_for_selected_params` 已处理 |
| R4 | 二次 prefill 让 latency x2 | 单次推理慢一倍 | docstring 标注；Phase C follow-up 考虑单次 prefill 方案 |
| R5 | `_inject_hl_text` 文本拼装破坏 tokenizer 模板语义 | Stage 3 输出偏移 | 测试 pin 住模板格式（字节级 diff） |
| R6 | 新字段进 `Pi0Config` 触发已有 checkpoint 加载失败 | 运行时崩 | 默认值保守、`dataclass` 向后兼容检查；测试覆盖 load-path |
| R7 | Tokenizer decode 路径不稳定（特殊 token / UTF-8） | 解出文本无法重新 tokenize | `_inject_hl_text` 做 sanity round-trip（tokenize(decode(ids)) 与原 ids 对比）；不一致时 fallback 到"跳过 HL"记录日志 |

---

## 10. Rollout

1. **Phase A 落盘**：实现 probe → 跑 10 条 LIBERO prompt → 提交 sample 输出给 user。
2. **Gate 决策点**：user 判定 probe 输出是否具备"低层指令"语义。
   - 不具备 → plan 归档 `logs/archive/pi05_hl_ar_decode_plan.log.md` 状态 `Historical`，流程终止。
   - 具备 → 进入 §3 G1（本 plan 进入 G1 gate）。
3. **§3 G1**：独立 Review Authority session 审本 plan；APPROVED 后进入 Phase B 代码实现。
4. **Phase B 落盘 + §5 G2**：完成 B.1–B.5 + 测试；暂存但不提交；独立 Review session 做 G2；APPROVED 后进入 §6 Verify。
5. **§6 Verify**：`uv run pytest` 全通过 + `@pytest.mark.manual` 用 `pi05_base` 真 checkpoint 跑一次 smoke。
6. **§7 Commit**（user 显式指令触发）：commit 含 src 改动 + 测试 + logs/README.md index 更新。
7. **§8 Push**：user 显式指令触发。

---

## 11. Non-goals

- 本 plan 不承诺"HL decode 能提升 LIBERO 成功率"。论文证据显示收益可能小；本工程目标是**提供机制**并验证其正确性，不是性能声索。
- 不更改 WebSocket policy server 协议 / 缓存系统 / warm start。
- 不为本功能新建架构文档（按 WA §9 "Minor change derivable from code → do NOT create a document"）。若 user 要求 paper / 实验文档另起 log。

---

## 12. Open Questions / TBD

以下 3 项在 Phase A 结果出来后、进 G1 前与 user 敲定：

- **O1**：首步 decode 的起始 token。选项 (a) 不塞任何新 token，直接对 prefill 末位置 lm_head 出 logits 采首 token；(b) 塞一个 `"\nAction: "` 编码后的首 token 作为 decode 种子。倾向 (a)。
- **O2**：HL 文本的停止符。PaliGemma tokenizer 的 `";"` token id + EOS id；在 probe 阶段观察实际收敛模式后定最终 stop set。
- **O3**：`Pi0Config` 字段的精确位置与命名——以仓库现有 `pi0_config.py` 风格为准，Phase B 开工前读一遍现有 fields list 后定案。

以上三项不影响 plan 主干，user 可在 G1 Review Log 里一并指示。

---

## Review Log

(empty — awaiting G1)
