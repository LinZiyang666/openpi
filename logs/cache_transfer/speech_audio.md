# 跨领域迁移调研：语音与音频生成/理解

> 调研对象：tiered experience cache + RIT + stateful gate 能否迁到语音/音频（flow-matching / diffusion TTS、流式 speech-to-speech、流式 ASR 迭代精炼、音乐/音效生成、codec/vocoder 分层）。
> 依据：`logs/cache_transfer/briefing.md`、`docs/iclr/iclr_paper/arxivd.tex` §64–344、`logs/session_handoff.md` §1.2；文献与开源代码见各条 URL。
> 日期：2026-09-15。只读仓库、未跑任何实验；凡未查到的数字一律标"不确定"。

---

## 0. 一页结论（先读这个）

- **结构上能对上**：两段式 TTS（CosyVoice 2 = AR LLM → 语义 token → flow-matching token2mel → HiFT；MaskGCT = AR T2S → 迭代 masked S2A → codec decoder）天然就是"共享前缀 + 迭代头"，tap point 处的语义 token + speaker embedding 是白拿的 query。F5-TTS/E2-TTS 这类纯 NAR DiT **没有前缀**，整条网络都是头。
- **成本余量方向对了但量不够**：延迟口径下 CosyVoice 2 的头只占约 30–36%（LLM 占 64–70%），hit 的 IR 地板 ≈ 0.64–0.70，比 VLA 的 0.15 差得多；FLOPs/吞吐口径下头占 ≈ 85%，地板才降到 ≈ 0.15。
- **"减 NFE 傻基线"在语音确实会掉点**（与 LIBERO 不同）：F5-TTS 32→5 NFE 时 UTMOS 3.93→2.91、SIM-o 0.66→0.59；朴素一步蒸馏 WER 2.8→4.8、SIM 0.66→0.52。**但**蒸馏（DSFlow / IntMeanFlow / MeanFlow token2wav / FlashTTS）已把 1–3 NFE 做到接近 teacher（token2wav 一步 RTF 0.0775→0.0046，17×），也就是说 warm 档的对手不是"朴素少步"而是"蒸馏一步"，一旦头被蒸成 1 步，系统又退化成 hit-or-miss 二元，和我们在 VLA 上遇到的死结一样。
- **闭环性弱**：质量是分级量（WER / SIM / UTMOS / MOS），不是成功/失败事件；WER/SIM/UTMOS 全部可**离线**对单条输出直接算，shadow 信号就是成功度量本身，"必须闭环测"的叙事在这里立不住。仅有的漂移对应物是流式 token2wav 用上一 chunk 的 mel 作 prompt 上下文，被替换的 chunk 会污染下一 chunk 的条件，但被文本条件强锚定、每句重置。
- **novelty 地形拥挤**：跨请求近似缓存（NIRVANA NSDI'24、Chorus 2026）已经把"相似 prompt → 复用中间去噪态 → 相似度阈值 → 命中率"整套做完，只是没做语音；intra-request 特征缓存、步数剪枝、蒸馏、AR TTS 投机解码（VADUSA 3×）、检索式投机解码（REST）、精确文本缓存（Home Assistant / Rhasspy / pipecat）、以及 2000 年代的 unit selection（target cost + join cost = 我们的 score + drift）全在。
- **结论：NO**（若非要做，只剩"流式 token2wav 的跨请求 warm-start 缓存"一篇系统短文，预期收益按 NIRVANA/Chorus 类推 ≤ 20–25%，且会被"直接蒸馏一步"一句话打掉）。

---

## 1. 领域内最贴合的 2–3 个具体系统

### 1.1 CosyVoice 2（最贴合，两段式 + 流式 chunk）

- 结构：文本 → Qwen2-0.5B AR LLM 输出 supervised semantic speech token（FSQ，基于 SenseVoice ASR encoder，25 Hz）→ chunk-aware causal flow matching（DiT，token2mel，**NFE 10、CFG β=0.7**）→ HiFT vocoder。开源（Apache-2.0）：<https://github.com/FunAudioLLM/CosyVoice>；论文 <https://arxiv.org/abs/2412.10117>。
- 成本拆分（实测，vllm-omni RFC，RTX 5080 单流，H100 复核）：TTFA 357 ms = LLM prefill 40.5 ms（11.3%）+ LLM decode 24 步 142.3 ms（39.9%）+ flow 首 chunk 41 token 129.0 ms（36.1%）；稳态 LLM ≈ 6.6 ms/token（launch-bound，GPU util 21.8%）vs flow+HiFT 每 chunk "106 ms + 0.50 ms/token"，摊到 ≈ 2.65 ms/token。<https://github.com/vllm-project/vllm-omni/issues/6870>
  → **延迟口径头占比 ≈ 29–36%**。
- FLOPs 口径（估计）：flow 150M 参数 × NFE 10 × CFG 2 ≈ 3 G 参数-次/token，LLM 0.5B × 1/token → 头 ≈ 85%。这是"launch-bound vs 算力-bound"的差别，与我们在 VLA 上的观察同源（`reference_vla_latency_launch_bound`）。
- 另一份拆分（FlashTTS，RTX 4090）：CosyVoice2 首包 843 ms；FlashTTS（Qwen2.5-0.5B + mean-flow 2-NFE 头）首包 325 ms = LLM 195 ms + token→wav 123 ms（头 38%）。<https://arxiv.org/html/2606.09141v1>
- token2wav 单独的 RTF（H20，FP16，bs=1）：CosyVoice2 10-step token2wav RTF 0.0775；一步 MeanFlow 0.0046（DiT 0.0016 + VAE 0.0030）。<https://arxiv.org/pdf/2606.18072>
- Benchmark：Seed-TTS-eval（test-en / test-zh，<https://github.com/BytedanceSpeech/seed-tts-eval>）、LibriSpeech-PC test-clean；度量 WER（Whisper-large-v3 / Paraformer）、SIM-o（WavLM-SV）、UTMOS、CMOS/SMOS。

### 1.2 MaskGCT（两段式 + 迭代 masked 头，S2A 是 FLOPs 大头）

- 结构：T2S（AR，Base 315M / Large 695M）→ 语义 token → S2A（353M，masked generative，逐 RVQ 层迭代，默认步数 [40,16,1×10]，可减到 [10,1×10]）→ codec decoder。开源（Amphion）：<https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct>；论文 <https://arxiv.org/html/2409.00750>。
- 减步数据（Seed-TTS test-en）：默认 SIM-o 0.728 / WER 2.466 / FSD 0.159 → 减步 0.709 / 2.796 / 0.164；test-zh 0.777/2.183/0.101 → 0.766/2.268/0.111。**减步掉点温和**。
- 成本拆分：论文未给 RTF 分段（不确定）。FLOPs 估计：10 s 语音 ≈ 500 语义 token，T2S 500 步 AR（KV cache）≈ 0.35 G 参数-次；S2A 66 次全序列 pass × 353M × 500 token ≈ 11.6 G 参数-次 → S2A ≈ 97% FLOPs，但 AR 段是延迟瓶颈。
- 与我们 tiers 的天然对应：S2A 本来就以 prompt acoustic token 作已知位置、只生成 masked 位置——"hit"= 把检索到的匹配 span 当 prompt 填上，"warm"= 从部分填充状态少迭代几步。

### 1.3 F5-TTS / E2-TTS（纯 NAR DiT，无前缀）

- 结构：文本 ConvNeXt 嵌入 + prompt mel 直接进 DiT，flow matching 生成 mel（NFE 32，Euler，CFG 2.0，sway sampling），Vocos 解码。开源 <https://github.com/SWivid/F5-TTS>；论文 <https://arxiv.org/html/2410.06885v1>。
- 成本：RTF 0.123 @32 NFE（RTX 3090，含 vocoder）；16 NFE 0.065；7 NFE 0.030（<https://arxiv.org/html/2505.19931v2>）。另一测量 32 NFE RTF 0.46（<https://arxiv.org/html/2509.08696>）。**没有昂贵前缀**：query 只能取文本/prompt mel（本来就免费），hit 的成本地板 = vocoder（估计 < 10%，不确定）。
- 问题：mel 长度由 duration 决定、与文本对齐，一段不同文本的中间态无法直接做 warm start；只有整句完全相同才有意义。

### 1.4 流式 speech-to-speech 对话（GLM-4-Voice / Qwen2.5-Omni / Moshi / Step-Audio）

- GLM-4-Voice：9B LLM + CosyVoice 式 flow 解码器（<https://github.com/zai-org/GLM-4-Voice>）；Qwen2.5-Omni：Thinker-Talker + sliding-window DiT + BigVGAN，Qwen3-TTS 把 Code2Wav 简化为轻量 causal ConvNet（<https://arxiv.org/html/2601.15621v1>）；Moshi：7B Temporal Transformer + 小 Depth Transformer + Mimi codec，全 AR、**无迭代头**（<https://kyutai.org/Moshi.pdf>）。
- 头占比：LLM 越大头占比越小（GLM-4-Voice 9B vs 解码器几百 M），估计 < 15%（不确定，未查到官方拆分）。头越来越被做成一步 ConvNet，迭代头正在消失。

### 1.5 其它子领域（一句话判定）

- 流式 ASR 非自回归迭代精炼（Mask-CTC / Align-Refine）：迭代 1–4 步已收敛（Align-Refine k=3 test-other WER 9.0，<https://arxiv.org/pdf/2010.14233>），编码器是前缀且占绝对大头，头没有可省的；**NO**。
- 音乐/音效生成（LAFMA、MusicFlow、FlashAudio）：步数极敏感（LAFMA FAD 9.47@5 步 → 1.48@100 步，<https://arxiv.org/pdf/2406.08203>），但 query 几乎不重复（每个 prompt 独一无二）、无闭环；**NO**。
- codec/vocoder 分层（RVQ 多层）：不是从噪声迭代，是逐层残差；与 tiers 形似神不似；**NO**。

---

## 2. 映射表（以 CosyVoice 2 流式 token2wav 为主线，MaskGCT 为副线）

| 我们的符号 | 语音对应物 | 备注 |
|---|---|---|
| 共享前缀 s1+s2 | AR LLM（文本 + speaker prompt → 语义 token）；MaskGCT 的 T2S | 延迟占 64–70%，FLOPs 占 ≈ 15% |
| tap point | LLM 输出的一个 chunk（CosyVoice 2 默认 15 token ≈ 0.6 s）的语义 token 序列 + speaker embedding（CAM++/ERes2Net）+ 上一 chunk mel 尾部 | 离散 token 可以**精确匹配**（n-gram 哈希），比 VLA 的连续相机特征友好得多 |
| 头 s3 | chunk-aware flow matching token2mel（NFE 10 × CFG 2 = 20 次 DiT 前向）+ HiFT | MaskGCT：S2A 66 次 pass |
| 中间去噪状态 x_t | 该 chunk 在 t = ρ_a 处的 noisy mel | 与 `run_stage3_from(start_t)` 同构；MaskGCT 是"部分 unmask 的 token 网格" |
| tier K（hit） | 直接回放缓存的 mel/wav chunk，只付 LLM 成本 | 精确文本 + 同 speaker 的整句命中 = Home Assistant 式缓存 |
| tier a（warm） | 从缓存 x_t 出发、以**当前** speaker embedding + 当前上一 chunk mel 为条件跑剩余 ρ_a 步 | ρ 能否 < 0.5 取决于同 token 序列下 mel 的方差 |
| key 字段 f | (i) 语义 token n-gram（精确/编辑距离）(ii) speaker embedding cosine (iii) 上一 chunk mel 尾 20 帧的 cosine（对应 `mel_cache_len=20`） | (iii) 是 join cost 的现代版 |
| IVF cell | speaker id（部署里通常只有 1–几个音色） | 比 VLA 的指令 cell 更干净 |
| 分数 s_t | 同式 (norm)/(fuse)，μ/τ 由库对自身留一估 | 可用 |
| 偏差 D_a | tier a 的 mel/wav 与从纯噪声全算参考的 mel 距离（标准化）；**更好的替代**：直接对 tier a 输出算 WER/SIM-o/UTMOS 差 | 后者就是成功度量本身，无需 rollout |
| 成功度量 SR | 无二元事件；用 WER ≤ τ_w ∧ SIM-o ≥ τ_s ∧ UTMOS ≥ τ_u 定义"utterance 合格率"作 SR 的代理 | ε 对应 CMOS 可感知阈值（约 ±0.1–0.2） |
| 成本模型 c_a | 延迟：c_pre ≈ 0.64–0.70 c_0；FLOPs：c_pre ≈ 0.15 c_0；Δ_j = 一步 DiT（×CFG） | 两个口径必须分开报 |
| 闭环性 | chunk t 的 mel 是 chunk t+1 的 prompt 上下文（`mel_overlap_dict` / `hift_cache_dict`，Hanning cross-fade）；替换 t 改变 t+1 的条件 | 存在但弱：文本条件强锚定、句末重置、无失败事件 |
| stateful gate | 同一句内连续 chunk 的命中相关性（整句命中 ⇒ 所有 chunk 命中，几乎平凡） | 跨句：同一会话同一音色，cell 不变 |

---

## 3. 打分表 A–G

| 项 | 分 | 一句依据 | 引用 |
|---|---|---|---|
| A 结构匹配 | **2** | CosyVoice 2 / MaskGCT 是标准"AR 前缀 + 迭代头 + 中间态"，tap point 的语义 token + speaker embedding 免费且离散可精确匹配；但 F5-TTS 类无前缀、s2s 系统头正在被做成一步 ConvNet | <https://arxiv.org/abs/2412.10117>, <https://arxiv.org/html/2409.00750>, <https://arxiv.org/html/2601.15621v1> |
| B 成本余量 | **1** | 延迟口径 hit 地板 ≈ 0.64–0.70（比 VLA 0.15 差）；FLOPs 口径 ≈ 0.15；朴素少步确实掉点（F5-TTS 5 NFE UTMOS 3.93→2.91），但蒸馏 1–3 NFE 已接近 teacher（token2wav 17×、DSFlow 1 步 MOS-N 4.32 vs teacher 4.43），warm 档存在理由被蒸馏吃掉 | <https://github.com/vllm-project/vllm-omni/issues/6870>, <https://arxiv.org/html/2505.19931v2>, <https://arxiv.org/pdf/2602.09041>, <https://arxiv.org/pdf/2606.18072> |
| C 局部性 | **2** | 客服/IVR/agent 的问候、确认、fallback 话术高度重复（社区反复要求加 TTS 缓存）；部署内 speaker 固定 ⇒ cell 平凡；但**没有公开的重复率数字**（不确定）；T2I/T2V 的跨请求缓存实测命中 42.6–58.9%（Chorus 1k 请求）、生产 >90%（NIRVANA 1.5M 库） | <https://github.com/pipecat-ai/pipecat/issues/2629>, <https://www.home-assistant.io/integrations/tts/>, <https://arxiv.org/pdf/2604.04451>, <https://arxiv.org/abs/2312.04429> |
| D 容忍度与闭环 | **1** | 近似输出可接受（听感容差），但质量是分级量而非成功事件；漂移对应物只有"上一 chunk mel 作 prompt"的弱耦合，被文本锚定、逐句重置；WER/SIM/UTMOS 全部离线可算 ⇒ 闭环叙事不成立 | <https://deepwiki.com/FunAudioLLM/CosyVoice>, <https://github.com/vllm-project/vllm-omni/pull/7521>, <https://arxiv.org/html/2412.10117v1> |
| E RIT 可用性 | **2** | shadow 信号离线可构造且比 VLA 更直接（对 tier 输出直接算 WER/SIM/UTMOS）；偏差随 token 匹配长度/speaker cosine 单调可信；一个 δ 切所有 cut、按预算反解都能照搬；gate 对应物平凡（整句命中） | 本报告 §2 映射；方法见 `arxivd.tex` §method:tiers |
| F novelty 地形 | **1** | NIRVANA（NSDI'24）+ Chorus（2026）已做完"跨请求相似度阈值 → 复用中间态 → 命中率/成本"；intra-request 缓存、EPSS 剪枝、蒸馏、VADUSA/PCG 投机解码、REST 检索式投机解码、精确文本缓存、Apple 混合 unit selection（target+join cost）全在；只剩"语音流式 token2wav 的跨请求 warm-start"这一小格 | <https://www.usenix.org/conference/nsdi24/presentation/agarwal-shubham>, <https://arxiv.org/pdf/2604.04451>, <https://arxiv.org/html/2509.08696>, <https://arxiv.org/abs/2509.09748>, <https://arxiv.org/abs/2410.21951>, <https://github.com/FasterDecoding/REST>, <https://machinelearning.apple.com/research/siri-voices> |
| G 我们做得动 | **3** | CosyVoice 2 / F5-TTS / MaskGCT 全开源、单卡 4090 可跑；Seed-TTS-eval 约 1–2k 句，一个前沿点数分钟到十几分钟；WER/SIM/UTMOS 评测器现成 | <https://github.com/FunAudioLLM/CosyVoice>, <https://github.com/BytedanceSpeech/seed-tts-eval> |

**合计 12/21；B、D、F 三项低分是结构性的，不是工程量能补的。**

---

## 4. 该领域的"减迭代次数"傻基线：文献结果

有人做过，而且很系统。结论：**朴素减步在 NFE ≤ 5–6 明显掉点，掉的是自然度/音色（UTMOS、SIM-o、MOS）而非可懂度（WER）；蒸馏后 1–3 NFE 追回。**

| 系统 / 设置 | NFE | WER | SIM-o | UTMOS / MOS | 来源 |
|---|---|---|---|---|---|
| F5-TTS 基线（LibriSpeech-PC） | 32 | 2.37 | 0.66 | 3.93 | EPSS Table 3 |
| F5-TTS + EPSS | 16 | 2.29 | 0.67 | 3.97 | 同上 |
| F5-TTS + EPSS | 7 | 2.45 | 0.66 | 3.84 | 同上 |
| F5-TTS 均匀/sway 采样（无 EPSS） | 7 | 4.16 | 0.60 | 3.36 | EPSS Table 1 |
| F5-TTS + EPSS | 5 | 2.55 | 0.59 | **2.91** | EPSS Table 3；原文："performance sharply declines below 6 NFE" |
| F5-TTS Base（Seed-TTS test-en） | 32 | 1.87 | 0.67 | 3.70 | IntMeanFlow Table 1 |
| F5-TTS + IntMeanFlow 蒸馏（无步采样优化） | 1 | 7.27 | 0.48 | 1.84 | 同上 |
| 同上 | 2 | 4.48 | 0.59 | 3.35 | 同上 |
| F5-TTS + IntMeanFlow + O3S | 3 | 1.60 | 0.65 | 3.79 | 同上（追平 teacher） |
| StepTTS teacher（LibriSpeech test-clean） | 10 | 2.8 | 0.66 | MOS-N 4.43 | DSFlow Table 1 |
| 朴素 endpoint 一步蒸馏 | 1 | 4.8 | 0.52 | MOS-N 3.56 | 同上 |
| DSFlow 一步 | 1 | 3.1 | 0.66 | MOS-N 4.32 | 同上 |
| CosyVoice2 flow teacher | 10 | 2.9 | 0.64 | MOS-N 4.41 | 同上 |
| DSFlow（CosyVoice2 学生） | 1 | 3.1 | 0.63 | MOS-N 4.23 | 同上 |
| CosyVoice2 token2wav（LibriSpeech test-clean） | 10 | 3.18 | 0.940 | UTMOS 3.76 / MOS 4.05 | MeanFlow token2wav Table 1 |
| Latent MeanFlow 一步 | 1 | 3.41 | 0.932 | UTMOS 3.64 / MOS 3.85 | 同上，RTF 0.0775→0.0046 |
| Voicebox（α=0） | 2→32 | 2.8→3.1（几乎不变） | NFE ≤ 4 时 SIM-r 更低（高 α 下） | FSD 随 NFE 单调改善 | <https://arxiv.org/pdf/2306.15687> §5.6 |
| MaskGCT S2A（Seed-TTS test-en） | [40,16,1…]→[10,1…] | 2.466→2.796 | 0.728→0.709 | FSD 0.159→0.164 | MaskGCT Table 8 |

来源 URL：EPSS <https://arxiv.org/html/2505.19931v2>；IntMeanFlow <https://arxiv.org/pdf/2510.07979>；DSFlow <https://arxiv.org/pdf/2602.09041>；MeanFlow token2wav <https://arxiv.org/pdf/2606.18072>；Voicebox <https://arxiv.org/pdf/2306.15687>；MaskGCT <https://arxiv.org/html/2409.00750>。

解读：

1. 与简报 §2 的"一步 Euler ≈ 条件均值"一致，语音里这个均值叫 over-smoothing / mean collapse：韵律是多峰的，回归到均值会得到"糊"的 mel，UTMOS 和 MOS 先掉，WER 后掉（可懂度只需要均值）。文献明确指出 F5/E2 这类 NAR 系统"collapse toward mean predictions of highly multimodal and diverse prosodic patterns"（<https://arxiv.org/html/2509.19928v1>）。
2. 所以语音**满足**"傻基线会掉点"这一条——比 LIBERO 强。但满足的方式是"蒸馏即可修复"，这让 warm 档要打的对手变成蒸馏一步头；蒸馏成本：DSFlow 约 72 h on A100 40GB（<https://arxiv.org/pdf/2602.09041>），IntMeanFlow 用 LibriTTS 585 h 即可把 CosyVoice2 token2mel 做到 1-NFE（WER 2.18 / SIM 0.63 / UTMOS 4.28 vs teacher 2.17 / 0.66 / 4.36）。
3. 头一旦是 1 NFE，Δ_j 只剩一步，tier 只剩 hit/miss；hit 的价值 = 省掉这一步 DiT + 省掉 vocoder（token2wav 一步 RTF 0.0046，占整条 pipeline 延迟不到 5%，估计）。

---

## 5. 预期收益（尽量用文献数字外推）

- **hit 率**：无语音专属数字（不确定）。可借用同类跨请求缓存：Chorus 在 1k 请求 trace 上命中 42.6%（冷）/ 58.9%（热，CLIP 相似度阈值 0.75），前 200 请求 < 20%，1500 请求后 ≈ 50%（<https://arxiv.org/pdf/2604.04451>）；NIRVANA 生产库 1.5M 条命中 > 90%（<https://arxiv.org/html/2508.20424>）。客服/IVR 的**精确整句重复**（问候/确认/fallback）估计能到几十个百分点，但这部分用哈希表就够，与我们方法无关；**chunk 级近重复**（同一短语嵌在不同句子里）的命中率才是我们方法的地盘，没有任何文献数字，需要自己在 MultiWOZ/SGD 系统话术上量。
- **IR 能压到多少**：
  - 延迟口径（单流 serving，CosyVoice 2）：hit 地板 0.64–0.70；若 chunk 级 hit 率 50%、warm 30%（ρ=0.5）、miss 20%：IR ≈ 0.5×0.67 + 0.3×(0.67+0.165) + 0.2×1.0 ≈ 0.79。**只省 21%**，与 NIRVANA 的 21% GPU / 19.8% 延迟 / 19% 美元惊人一致（<https://arxiv.org/abs/2312.04429>）。
  - FLOPs/吞吐口径（批量离线合成）：hit 地板 0.15，同样混合 → IR ≈ 0.5×0.15 + 0.3×0.575 + 0.2×1.0 ≈ 0.45。这个口径有看头，但离线批合成里"重复文本"直接去重就行，不需要近似缓存。
- **SR 损失**：warm@ρ=0.5 的质量没有文献直测（不确定）。可类比 EPSS 的 NFE 5（UTMOS −1.0）作为悲观下界、IntMeanFlow O3S 3-NFE（几乎无损）作为乐观上界。hit 的质量损失 = 同 token 序列下的 mel 方差，主要体现在韵律/上下文协同发音，用 SIM-o/WER 几乎测不出、用 CMOS 才测得出——这意味着**我们的成功度量必须上主观或 UTMOS 类打分**，评测成本上升。
- **与 VLA 对比**：VLA hit 地板 0.15、减步基线不掉点；语音（延迟口径）hit 地板 0.67、减步基线掉点但蒸馏可修。两边的"帕累托前沿相对减步基线的增量"都小：VLA 因为没有 counterfactual，语音因为蒸馏把 counterfactual 抹平、且前缀太贵。

---

## 6. novelty 地形与最强反驳

### 已有工作（按与我们的重叠度排序）

1. **跨请求近似缓存（几乎就是我们的 tiers + 阈值 + 命中率）**：NIRVANA（Adobe，NSDI'24）用文本相似度 > 0.65 命中，从缓存的中间噪声态继续去噪（跳过前 20 步），生产 21% GPU 节省，LCBFU 淘汰策略（<https://www.usenix.org/conference/nsdi24/presentation/agarwal-shubham>）；Chorus（2026）在视频 DiT 上做三阶段（stage-1 全复用 latent = 我们的 hit/warm、stage-2 区域复用、stage-3 全算修复不连续），并明确在 4 步蒸馏模型上 intra-request 缓存失效而 inter-request 仍有 1.23–1.45×（<https://arxiv.org/pdf/2604.04451>）。**没有语音版本**——这是唯一的空格。
2. **intra-request 缓存 / 剪枝（训练无关，与我们正交但抢同一块蛋糕）**：Transformer layer caching for F5-TTS（32 NFE RTF 0.46→0.26，WER 2.03→2.06，<https://arxiv.org/html/2509.08696>）；DiTReducio（F5-TTS / MegaTTS 3，FLOPs −75.4%，RTF +37.1%，<https://arxiv.org/abs/2509.09748>）；EPSS（7 NFE 4×，<https://arxiv.org/html/2505.19931v2>）；扩散缓存综述 <https://arxiv.org/pdf/2510.19755>。
3. **蒸馏到 1–3 NFE（把头的成本直接抹掉）**：DMOSpeech、CoMoSpeech（<https://arxiv.org/pdf/2305.06908>）、FlashSpeech（<https://arxiv.org/pdf/2404.14700>）、IntMeanFlow、DSFlow、Latent MeanFlow token2wav（17×）、FlashTTS（首包 843→325 ms）。
4. **AR 段加速**：VADUSA 投机解码 ≈ 3×（<https://arxiv.org/abs/2410.21951>）、PCG 粗粒度接受（<https://arxiv.org/pdf/2511.13732>）、REST 检索式投机解码（无训练、精确 n-gram 检索、2.33×，<https://github.com/FasterDecoding/REST>）——REST 恰好是"用缓存给 AR 前缀做 warm start 且零质量损失"的现成答案，把我们在前缀侧能做的事也堵死了。
5. **精确文本缓存（工程常识）**：Home Assistant（<https://www.home-assistant.io/integrations/tts/>）、Rhasspy（<https://rhasspy.readthedocs.io/en/latest/text-to-speech/>）、pipecat 提案（key = text + voice + model + 参数，命中 ~1 ms，<https://github.com/pipecat-ai/pipecat/issues/2629>）、elizaOS 甚至讨论把短语缓存限制在 ≤ 5 词（<https://github.com/elizaOS/eliza/issues/30680>）。
6. **祖先：unit selection / hybrid unit selection**：target cost（= 我们的 score）+ concatenation/join cost（= 我们担心的 drift/不连续）+ Viterbi 选路，Apple Siri 2017 用 deep MDN 预测两种 cost 并部署到数亿设备（<https://machinelearning.apple.com/research/siri-voices>, <https://www.isca-archive.org/interspeech_2017/capes17_interspeech.html>）。审稿人会说：你们把 unit selection 装进 flow matching 里，join cost 换成了 warm start。
7. 相邻：FlowEdit 用 associative memory 做发音自适应（<https://arxiv.org/pdf/2606.20518>）；Voicebox/E2 的 in-context infilling 本身就是"给定部分 mel 生成其余"，与 MaskGCT 一样天然支持"hit span 当 prompt"。

### 最强反驳（按杀伤力）

1. **"把头蒸成一步就没有 warm 档了"**：token2wav 一步 RTF 0.0046（H20），头占整条 pipeline 延迟 < 5%；你省的是零头的零头。回答不了，因为我们的方法是训练无关而蒸馏是一次性成本。
2. **"前缀才是大头，而前缀已有 prompt cache / REST / VADUSA"**：延迟口径 LLM 占 64–70%，hit 地板 0.67；NIRVANA 在头占 100% 的 T2I 上都只省 21%。
3. **"精确重复用哈希表，近似重复听得出来"**：语音听感对 join 不连续和韵律不匹配极敏感（unit selection 20 年的教训），近似命中的容差 ε 要用 CMOS 测，而 CMOS 的可感知阈值 ±0.1 会把可用命中率压得很低。
4. **"你的成功度量离线就能算，闭环故事是多余的"**：WER/SIM-o/UTMOS 逐句可算，shadow 信号 = 成功度量，RIT 退化为 conformal 阈值化，"必须闭环测"这一核心论点没有对应物。
5. **"这是 NIRVANA/Chorus 换个模态"**：tiers、相似度阈值、命中率随库增长、末段全算修复，全部已发表。

---

## 7. 首个实验方案（≤ 2 周，只做傻基线 + 成本拆分 + 命中率普查，不碰 RIT）

- **模型**：CosyVoice 2（0.5B，Apache-2.0，`FunAudioLLM/CosyVoice`），流式模式，chunk 15 token；备选 F5-TTS 做"无前缀"对照。
- **Benchmark**：Seed-TTS-eval test-en（≈ 1k 句，WER by Whisper-large-v3、SIM-o by WavLM-SV、UTMOS）。评测成本：4090 上单个前沿点 ≈ 10–20 min（估计，按 RTF 0.2–0.5 × 1k 句 × 平均 8 s）。
- **算力**：1×4090 足够；H100 空闲时做 batch=8 的吞吐口径。
- **Day 1–3：成本拆分实测**。CUDA 同步计时 LLM prefill / LLM decode / flow（按 NFE）/ HiFT，batch=1 与 batch=8 两口径，复现 vllm-omni 的 36%/64% 拆分并给出我们机器的 c_pre/c_0。
- **Day 3–6：傻基线**。flow NFE ∈ {1,2,3,5,7,10}（CFG 保持 0.7）扫 WER/SIM-o/UTMOS，画 (IR, 质量) 曲线；同时跑 IntMeanFlow/DSFlow 若有公开 checkpoint（不确定是否放出）作"蒸馏一步"锚点。**若 NFE=3 已在 UTMOS −0.1 以内，直接终止**：warm 档没有生存空间。
- **Day 6–9：命中率普查（零 GPU）**。取 MultiWOZ 2.2 / SGD 的系统话术，统计 (i) 整句精确重复率 (ii) 用 CosyVoice 2 tokenizer 离散化后 15-token chunk 的精确重复率与 ≥ 10-token 后缀匹配率（模拟 REST）。这给出 C 项的真实数字。
- **Day 9–14：warm start 的 counterfactual 探针**。对同一 speaker、同一语义 token chunk 的两次合成，把第二次从第一次的 x_{t=0.5} 出发（以第二次的上一 chunk mel 为条件）跑剩余 5 步，对比 (a) 全算 (b) NFE=5 从噪声 (c) 直接回放。度量 WER/SIM-o/UTMOS + 20 句小规模 CMOS。**判据**：(a)≈(warm) 且 (warm) 明显优于 (b)，才有 counterfactual；否则收工。
- 输出：一张 (IR, UTMOS/SIM-o) 图、一张成本拆分表、一个重复率表；不写 RIT、不写 gate。

---

## 8. 结论

**NO。** 结构对得上（两段式 TTS 的语义 token + speaker embedding 是免费且可精确匹配的 query），减步傻基线也确实会掉点，但三件事把收益封死：延迟口径下 hit 只能省掉约三分之一（LLM 前缀占 64–70%，与 VLA 的 15% 地板反向）；蒸馏已把头做到 1–3 NFE 接近 teacher，warm 档失去 counterfactual；质量度量逐句离线可算、无成功/失败事件，"必须闭环测"的核心叙事没有对应物。加上 NIRVANA/Chorus 已发表同构方法，剩下的只是"语音流式 token2wav 的跨请求近似缓存"一篇系统短文，预期收益 ≈ 20%，且一句"直接蒸馏一步"即可反驳。
