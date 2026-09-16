# 跨领域迁移调研：LLM / VLM 推理服务与 agent 工作流

> 调研日期 2026-09-15。输入：`logs/cache_transfer/briefing.md`、`docs/iclr/iclr_paper/arxivd.tex` L64–344、`logs/session_handoff.md` §1.2。
> 领域范围：semantic cache、prefix/KV cache 精确与近似复用、检索式 speculative decoding、多轮助手 / RAG / 代码补全 / GUI-web agent、reranker / embedding 服务。
> 本文所有数字均来自引用 URL；写"不确定"的地方就是文献里没找到数字。

## 0. 先给答案：LLM 世界里哪些子场景能容忍 lossy 分档

我们的系统有三个要素——**tiers（含 warm start）、RIT（一个 δ 切全部 cut）、stateful gate**——它们各自需要的前提不同：warm 档需要"有中间状态可续跑、且减迭代会掉点"；RIT 需要"可离线算的偏差信号 + 随分数单调"；gate 需要"命中成串"。按这三个前提把领域切开：

| 子场景 | 能否容忍 lossy | 已被谁做掉 | 我们还剩什么 |
|---|---|---|---|
| **Autoregressive（AR）agent 的重复工具调用 / 计划**（coding agent、web agent、embodied planner） | 能：动作错了环境会报错或后续步纠正，成功度量 = task SR | 计划级 cache（Agentic Plan Caching、AgenticCache、AgentReuse）、tool 返回值 cache（TVCache）、prefix KV cache（生产 90%+ 命中） | 没有 warm 档的中间状态（AR 没有"去噪一半"的东西）；能剩下的是 **"回放 vs 验证式回放 vs 全算"三档 + 闭环漂移测量**。novelty 一般 |
| **GUI agent（截图→VLM→动作）**，AR 版（UI-TARS、Qwen-VL 系） | 能：动作粒度小、错了能重试；SR 在 AndroidWorld/OSWorld 上可测 | 视觉 token 剪枝 / KV 跨帧复用（TRACE、GUI-KV）；memory 类工作（AWM、Mem-W） | 前缀（视觉编码）占大头 ⇒ hit 地板高，除非用便宜的截图 hash 做 key 连视觉编码一起跳过；无中间状态 ⇒ 仍是 hit-or-miss |
| **Diffusion-LM 驱动的 agent**（LLaDA-UI、LLaDA/Dream 文本 agent） | 能，同上 | dLLM-Cache / Fast-dLLM 只做特征缓存，**没有检索式 warm start**；Speculative Correction 证明"从完整草稿出发精炼"远好于"从全 mask 出发" | **结构与我们几乎同构**：前缀（视觉/prompt 编码）+ 迭代去噪头 + 减步会大掉点。tiers + RIT + gate 三件套全有落点。这是唯一 GO 的支线 |
| **分类式 / 路由 / 结构化抽取输出**（固定标签集） | 能（错标签可容忍 ε） | vCache（ICLR 2026）：逐条目在线阈值 + 用户给定错误率上界——**就是单档版 RIT** | 只有 hit-or-miss；没有迭代头，warm 档无对象 |
| **Reranker / embedding 服务** | reranker 分数偏差可容忍；embedding 是确定性函数，精确 cache 即可 | 级联排序 + cross-encoder early exit（SIGIR 2025 SEE）、Adaptive Re-Ranking | 无共享前缀 + 迭代头结构，无闭环；**NO** |
| **开放式 chat / RAG 答案** | 只能 hit-or-miss（用户直接读输出，近似答案可感知） | GPTCache / MeanCache / vCache / Redis LangCache；近似 KV 复用（CacheBlend、EPIC、SemShareKV） | 无闭环、无成功度量、无中间状态；**NO** |
| **代码补全 / 输入接地生成（摘要、context-QA）** | 不能 lossy（token 级要精确） | prefix caching（Copilot 93% token 被缓存）+ lossless 检索式 SD（REST、LLMA、prompt lookup、SuffixDecoding） | 已被无损方法吃干净；lossy 接受（Judge Decoding、FLy、AdaptiveSpec）是 SD 社区自己的活 |
| **推理模型（thinking）** | "减迭代" = 减 thinking token | NoThinking / budget forcing 证明低预算下少想不掉点甚至更好 | 与我们 LIBERO 的教训同构：**减步基线不掉点 ⇒ warm 档无 counterfactual** |

## 1. 领域内最贴合的 2–3 个具体系统

### 1.1 Diffusion VLM GUI agent：LLaDA-UI（首选）
- 模型：16.7B MoE、block-wise diffusion 的视觉语言 GUI agent，语言骨干 LLaDA2.0-mini-base，视觉编码 SigLIP；项目页 / GitHub / HuggingFace 均给出，权重与代码公开。https://arxiv.org/html/2609.13287
- Benchmark：ScreenSpot-V2 90.7%、ScreenSpot-Pro 52.9%、AndroidWorld 53.5%、MobileWorld 25.6%、OSWorld-Verified 29.4%、WebVoyager 56.9%（同上）。
- **减步证据**（就是我们要的"傻基线"）：32 步 / block 32 在 AndroidWorld 42.7%；16 步 / block 64 掉到 33.0%（同上，消融表）。
- 成本拆分：论文只报相对 Qwen3-VL-8B 的 3.6–9.0× 端到端加速，没给"视觉编码 vs 去噪步"的拆分——**需要实测**（§7 第 0 步）。
- 同族文本 dLLM 可作便宜替身：LLaDA-8B-Instruct / Dream-7B（开源）。减步证据：LLaDA base 在 GSM8K 256 步 ≈69%，压到 ≈48 步（与 dLLM-Cache 同 FLOPs）≈22%（https://arxiv.org/html/2506.06295v3 Fig.5c）；Dream-7B 上"朴素截步显著掉点，consistency 蒸馏后才能 4.1–7.7× 减步"（https://www.together.ai/blog/consistency-diffusion-language-models）。

### 1.2 AR coding / tool agent：SuffixDecoding + 计划级 cache 这条线
- SuffixDecoding（NeurIPS 2025 spotlight）：用 suffix tree 缓存历史 prompt 与输出，模型无关 speculative decoding；agentic 负载最高 5.3×，比 EAGLE-2/3 快至 2.8×。https://arxiv.org/abs/2411.04975
- Agentic Plan Caching（NeurIPS 2025）：缓存 planning 阶段的结构化计划模板，关键词匹配，轻量模型适配；平均省 50.31% 成本、27.28% 延迟，"性能保持"。https://arxiv.org/pdf/2506.14852
- AgenticCache（embodied，多 agent）：缓存"计划转移"，利用"下一计划基本由当前计划可预测"的 plan locality；12 组配置平均 SR +22%、仿真延迟 −65%、token −50%；命中直接回放，**后台异步用 LLM 校验刷新**。https://arxiv.org/abs/2604.24039
- TVCache（ICML 2026）：RL post-training 里缓存 tool 返回值，按完整 tool-call 历史做最长前缀匹配才算命中（保证环境状态一致）；命中率最高 70%，中位 tool 执行时间 −6.9×，reward 无损。https://arxiv.org/abs/2602.10986v1
- 成本拆分（真实 agent 负载）：有 context cache 时 decode 占 LLM 时间 91.0–98.6%，prefill 仅 1.4–9.0%；经验 cache 命中 84.6–99.5%。https://arxiv.org/html/2605.26297v1
- GitHub Copilot 生产 trace：每次调用中位 68K prompt token，其中 63K 命中前缀缓存（≈93%）；轮内命中 90%、轮边界降到 55%；中位输出 247 token，输入:输出 > 275:1。https://arxiv.org/html/2608.00101

### 1.3 AR GUI agent：UI-TARS / Qwen-VL 系 + TRACE
- UI-TARS（开源）：https://arxiv.org/pdf/2501.12326 ；GUI-Owl-1.5-8B、Qwen3-VL 亦开源。
- 成本拆分：4096² 截图下视觉编码器占 prefill 时间的 86% / 72% / 75%（Qwen3-VL 2B / 4B / 8B）；ScreenSpot-Pro 单页 ViT 625 ms。https://arxiv.org/html/2604.00886
- 跨帧复用（最近的竞品）：TRACE 复用视觉 token 的 KV cache 并按预算裁剪；GUI-Owl-1.5-8B 在 OmniGUI 上 TTFT 1116.8→452.7 ms、prefill 328.6→140.6 ms、视觉 KV 697→292 MB；温和预算（当前帧 50% / 历史 10%）只保住 dense 性能的 78.7%，紧预算 61.1%。https://arxiv.org/html/2609.10297
- 每步延迟量级：典型移动端 agent > 20 s/步，V-Droid 4.3 s/步；OSWorld-Human 报 agent 步数是人类轨迹的 2.7–4.3×。https://arxiv.org/html/2609.02309

## 2. 映射表

两条 lane 分开写：**D-lane**（diffusion-LM agent，LLaDA-UI / LLaDA 文本 agent）和 **A-lane**（AR agent）。

| 我们的符号 | D-lane（dLLM agent） | A-lane（AR agent） |
|---|---|---|
| 共享前缀 / c_pre | SigLIP 视觉编码 + prompt/历史 prefill（dLLM-Cache 已把 prompt 特征长间隔缓存，说明它是"前缀"） | 视觉编码（GUI）+ 本轮新增 token 的 prefill；历史上下文的 KV 由精确 prefix cache 免费提供（读价 0.1×，https://platform.claude.com/docs/en/docs/build-with-claude/prompt-caching） |
| tap point / 免费 key | 前缀算完处：视觉 token 池化 + 指令/历史 embedding 均值 + 上一动作 | 同上；另可加一个**比前缀更便宜**的 key（截图 pHash / a11y 树 hash / tool-call 历史 hash，TVCache 式），命中时连视觉编码也跳过 |
| 头 / 迭代 | 对动作 block 的 N 步去噪（LLaDA-UI 32 步/block） | 逐 token decode（无迭代精炼；agent 输出短：Copilot 中位 247 token） |
| 中间状态 | 部分 unmask 的 token 序列（第 j 步的状态） | **不存在**。替代物 = 检索到的历史输出当 draft，用一次并行 forward 验证（LLMA / REST 机制） |
| tier 定义 | tier K = 回放库里动作；tier a = 把库动作按比例 ρ_a 重新 mask，在当前截图条件下续跑剩余步；tier 0 = 全 mask 全算 | tier K = 无验证回放；tier a = 回放 + 按宽松度分级的 lossy 验证（margin 阈值，AdaptiveSpec 式；被拒后从拒绝点续 decode）；tier 0 = 严格验证或不用 draft |
| key 字段 | 截图 token map、指令 embedding、上一动作、（可选）a11y 树 | context embedding（最后 hidden 均值）、tool-call 历史前缀、当前 tool 返回摘要 |
| 分数 s_t | 与 eq.norm/fuse 相同：逐字段 tanh 标准化后加权 | 同上；A-lane 可直接沿用 semantic cache 的 cosine 阈值范式 |
| 偏差 D_a | tier a 动作串 vs 全算动作串的标准化 edit distance；参数字段 exact-match 指示（坐标动作用像素距离） | 同上；工具调用 = (name, args) 结构化比较 |
| 成功度量 SR | AndroidWorld / OSWorld / WebVoyager task SR；ALFWorld/WebShop（文本 dLLM 替身） | SWE-bench / τ-bench / WebArena task SR |
| 成本模型 | c_a = c_pre + Σ_{j≥a}Δ_j，Δ_j = 单步去噪 forward；实测待做 | c_K ≈ c_key（可低于 c_pre）；c_a = c_pre + verify(L_draft) + decode(拒绝后余量)，后者随机 |
| 闭环性 / 漂移 | 有：回放的动作改变屏幕/环境，后续决策落在策略不会自己到达的状态 | 有，同上；TVCache 强调"tool 输出依赖先前交互引起的环境状态"正是此性质 |
| 状态门对应物 | 同一 app/页面内连续命中（session 局部性） | 轮内 prefix 命中 90% vs 轮边界 55%（Copilot trace）；99.28%（chat）/ 99.96%（agent）的复用块来自同一 session（https://arxiv.org/pdf/2605.18825）——"成串"结构天然存在 |

## 3. 打分表 A–G（0–3）

| 项 | 分 | 依据 | URL |
|---|---|---|---|
| A 结构匹配 | **2**（D-lane 3 / A-lane 1） | dLLM agent = 前缀 + 迭代去噪头 + 部分 unmask 中间状态，与 flow-matching 头同构；AR 没有中间状态，warm 档只能退化成"draft + 验证" | https://arxiv.org/html/2609.13287 ；https://arxiv.org/abs/2304.04487 |
| B 成本余量 | **2**（D-lane 3 / A-lane 1） | dLLM 减步掉点：LLaDA GSM8K 256→≈48 步 69%→22%；LLaDA-UI 32→16 步 42.7%→33.0%。AR 的"减迭代"= 少想 / 少输出，NoThinking 在低预算反而更好（AMC23 @700 token 51.3 vs 28.9），warm 档无 counterfactual。AR agent 有 cache 时 prefill 仅 1.4–9%，hit 省得多但 GUI 视觉编码占 prefill 72–86%，hit 地板高 | https://arxiv.org/html/2506.06295v3 ；https://arxiv.org/html/2609.13287 ；https://arxiv.org/html/2504.09858 ；https://arxiv.org/html/2605.26297v1 ；https://arxiv.org/html/2604.00886 |
| C 局部性 | **2** | 生产 semantic cache 命中 20–45%（Portkey RAG ≈20%，EdTech ≈45%，开放 chat 10–20%）；agent 请求 30% 相同或相似（AgentReuse）；tool 调用命中最高 70%（TVCache，平行 rollout 场景）；session 内 prefix 复用 90%+。跨 episode 的"同屏同动作"命中率**文献无数字，不确定** | https://dev.to/gauravdagde/llm-semantic-caching-the-95-hit-rate-myth-and-what-production-data-actually-shows-8ga ；https://arxiv.org/abs/2512.21309 ；https://arxiv.org/abs/2602.10986v1 ；https://arxiv.org/html/2608.00101 |
| D 容忍度与闭环 | **2**（agent 3 / chat-RAG 0） | agent 动作错了会被环境报错/后续步纠正，SR 可测，且回放改变环境状态 ⇒ 漂移存在（AgenticCache 需后台校验刷新、TVCache 需全历史匹配都是此性质的旁证）；chat/RAG 输出直接给人读，只能 hit-or-miss | https://arxiv.org/abs/2604.24039 ；https://arxiv.org/abs/2602.10986v1 |
| E RIT 可用性 | **2** | 离线 D_a 易构造（edit distance / 结构化 args 比较 / embedding 距离 / LLM judge）；但 vCache 已做"逐条目在线阈值 + 用户给定错误率上界"（LMArena：δ=0.02 命中≈46%、错≈1.1%；GPTCache 静态 0.98 阈值命中 53%、错 4.1%），审稿人会问 RIT 比 vCache 多什么——答案只能是"多档 + warm 档偏差随 ρ 的 order restriction + 按预算反解"。成串结构有（见 C） | https://arxiv.org/html/2502.03771 |
| F novelty 地形 | **1**（D-lane 2 / A-lane 1） | 拥挤：semantic cache（GPTCache/MeanCache/vCache/Krites 两层 static+dynamic）、计划 cache（APC/AgenticCache/AgentReuse）、tool cache（TVCache）、检索式 SD（REST/LLMA/prompt lookup/SuffixDecoding）、lossy 验证（Judge Decoding/FLy/AdaptiveSpec）、dLLM 特征缓存（dLLM-Cache/Fast-dLLM）。**空白**：dLLM 的检索式 warm start（Speculative Correction 只从小模型草稿精炼，不从库）；以及"cache 引起的闭环漂移"没人测 | https://arxiv.org/pdf/2602.13165 ；https://arxiv.org/abs/2411.04975 ；https://arxiv.org/html/2511.22972 ；https://arxiv.org/abs/2608.02625 |
| G 做得动 | **2** | LLaDA-UI 16.7B MoE bf16 单 H100 可放；LLaDA-8B/Dream-7B 4090 可放。AndroidWorld 需 Android 模拟器，每步秒级、每集分钟级，一条前沿（116 任务×3 seed×4 点）估计数十 GPU 小时——**不确定**；文本 dLLM + ALFWorld 便宜得多可先跑 | https://arxiv.org/html/2609.13287 ；https://nvlabs.github.io/Fast-dLLM/ |

## 4. 该领域的"减迭代次数"傻基线

四种对应物，证据分化明显：

1. **dLLM 减去噪步**（真正同构的对应物）：LLaDA base GSM8K 256 步≈69%、≈48 步≈22%（https://arxiv.org/html/2506.06295v3）；LLaDA-UI AndroidWorld 32 步 42.7%、16 步 33.0%（https://arxiv.org/html/2609.13287）；Dream-7B "朴素截步显著掉点"，须 consistency 蒸馏才能 4.1–7.7× 减步（https://www.together.ai/blog/consistency-diffusion-language-models）。**结论：会掉点，warm 档有存在理由。** 机制与 VLA 不同：不是"单峰均值无损"，而是并行 unmask 的 token 独立性假设在少步时破坏一致性——所以 warm start（从一条真实、自洽的历史输出出发只重 mask 一部分）恰好补这个洞，Speculative Correction 已证"从完整草稿精炼 ≫ 从全 mask 起"（GSM8K 0.848→0.899，https://arxiv.org/abs/2608.02625）。
2. **AR 减 decode 长度 / 少想**：NoThinking 低预算（<3k token）一致优于 Thinking（AMC23 @700 token 51.3 vs 28.9；AIME24 @3500 77.3 vs 73.3，https://arxiv.org/html/2504.09858）；GSM8K/MATH-500 在 256 个 thinking token 就达到无上限精度的 95%（https://arxiv.org/html/2607.21433v1）。**结论：常常不掉点**——和我们 LIBERO k=1 = teacher 的处境一样，warm 档没戏。
3. **AR early exit（减层数）**：Llama2/3、Qwen2/3 上 50% 深度退出掉 5–15%，25% 深度掉 15–30%，而实际墙钟只省 10–20%（https://arxiv.org/pdf/2603.23701）。掉点但省得少，且不是我们能 warm start 的维度。
4. **少几轮 draft / lossy 接受**：lossy 验证（AdaptiveSpec 用 target 概率比阈值放行，FLy 用熵门 + 延迟窗口）以"最小精度代价"换吞吐（https://arxiv.org/html/2609.02897 ；https://arxiv.org/html/2511.22972）；"Revisiting Lossy Verification" 系统地列出失效模式（https://arxiv.org/html/2607.26627）。这是 A-lane 里 tiers 的天然实现，但已是 SD 社区的主战场。

## 5. 预期收益

- **hit 率**：D-lane 无直接数字。可参考的上界/下界：agent 请求 30% 相同或相似（https://arxiv.org/abs/2512.21309）；生产 semantic cache 20–45%（https://dev.to/gauravdagde/llm-semantic-caching-the-95-hit-rate-myth-and-what-production-data-actually-shows-8ga）；GUI 连续帧高度时序冗余（TRACE 把历史帧预算压到 10% 仍保 78.7%，https://arxiv.org/html/2609.10297）。LIBERO 式"同任务多 seed"的重复度在 AndroidWorld（116 任务，参数随机化）上应低于 LIBERO（同场景同物体），我估计 hit 率 20–40%，**不确定**。
- **IR 能压到多少**：D-lane 取决于 c_pre/c_0，LLaDA-UI 未报拆分。若视觉编码像 AR GUI 那样占 prefill 72–86%（https://arxiv.org/html/2604.00886）而去噪 32 步/block 又是大头，c_pre/c_0 可能落在 20–40% 区间（估计，需实测）；用截图 hash 做 hit key 可把 hit 成本压到接近 0，但 warm 档仍付 c_pre。A-lane：有 prefix cache 时 prefill 仅 1.4–9%（https://arxiv.org/html/2605.26297v1），hit 几乎省全部，但 warm 档只剩验证式回放。
- **SR 损失**：无 lossy 分档的直接数字。类比证据：AgenticCache 回放计划 + 后台校验 SR 反而 +22%（https://arxiv.org/abs/2604.24039）——说明 agent 领域对回放的容忍度比 VLA 高；TRACE 紧预算掉到 dense 的 61%（https://arxiv.org/html/2609.10297）——说明"省视觉 token"这条路掉点凶，反衬"整条动作回放"可能更划算。
- **与 VLA 对比**：VLA 的问题是"减步不掉点 ⇒ warm 档无意义"，dLLM 恰好相反（减步大掉点），这是迁移的主要理由；坏处是 GUI 前缀占比可能比 π0.5 的 15% 高得多，hit 省得少，得靠便宜 key 绕过。

## 6. novelty 地形与最强反驳

已有：
- semantic cache 阈值学：vCache 用户定错误率 + 逐条目在线阈值（ICLR 2026，https://arxiv.org/html/2502.03771）；Krites 两层 static/dynamic + 异步 LLM judge（https://arxiv.org/pdf/2602.13165）；Generative Caching 对结构相似 prompt 做"模板 + 改写"式 lossy 命中，agentic 工作流命中 83%（https://arxiv.org/pdf/2511.17565）。
- 近似 KV 复用：CacheBlend（F1/Rouge-L 差 ≤0.02，TTFT −2.2–3.3×，https://arxiv.org/abs/2405.16444）但独立复现显示比全 prefill 低 7–18%（https://arxiv.org/html/2603.20218v1）；EPIC（ICML 2025，https://arxiv.org/abs/2410.15332）；SemShareKV 语义相似 prompt 的 KV 共享 6.25×（https://arxiv.org/abs/2509.24832）。
- 检索式 SD：REST 1.62–2.36×（https://arxiv.org/abs/2311.08252）、LLMA >2× 无损（https://arxiv.org/abs/2304.04487）、prompt lookup 2–4× 无损（https://github.com/apoorvumang/prompt-lookup-decoding）、SuffixDecoding 5.3×（https://arxiv.org/abs/2411.04975）。
- agent cache：APC、AgenticCache、AgentReuse、TVCache、ReCache（https://arxiv.org/html/2608.19662）、Continuum/CacheTTL（https://arxiv.org/html/2511.02230v5）。
- dLLM 加速：dLLM-Cache 4.28× 无损（https://arxiv.org/html/2506.06295v3）、Fast-dLLM 27.6× 吞吐（https://nvlabs.github.io/Fast-dLLM/）。

最强反驳（审稿人会打的点）：
1. **"vCache 已经是带保证的阈值标定，你的 RIT 只是把它推广到多档"**——须用 warm 档的 order-restricted 分位曲线和按预算反解回答，单档场景无优势。
2. **"AR agent 里你的 warm 档就是 lossy speculative decoding"**——A-lane 站不住，只能承认并让位。
3. **"dLLM 还没人在生产用、agent 版刚出（LLaDA-UI 2026-09）"**——目标模型的代表性弱；且 consistency 蒸馏（CDLM）一旦普及，减步基线又不掉点，warm 档再度失去 counterfactual（与 VLA 同一命运）。
4. **"计划级 cache 已报 SR +22% 与 −50% token"**——粒度更粗、收益更大；我们的动作级 tiers 需证明在计划 cache 之上还能叠加。
5. **闭环漂移**是我们独有的测量，但要有正例（连续回放确实漂）才成立；GUI 环境里回放错动作往往立即被界面状态暴露，漂移可能比机器人更短命——是利也是弊。

## 7. 首个实验方案（≤2 周）

目标：在 D-lane 上验证三件事——成本拆分、减步基线掉点、库回放/warm start 的 shadow 偏差随分数单调。

- **模型**：主选 LLaDA-UI（单 H100）；替身 LLaDA-8B-Instruct（4090）跑 ALFWorld 文本环境，用来在模拟器就绪前把 pipeline 跑通。
- **Benchmark**：AndroidWorld（116 任务，SR 有官方 harness）；替身 ALFWorld。
- **第 0 步（第 1–3 天）成本拆分实测**：每个动作 block 的 SigLIP 编码 / prompt prefill / 每步去噪 forward 各自 ms，得到 c_pre/c_0 与 Δ_j；同时测截图 pHash 的成本作为"便宜 key"的地板。
- **第 1 步（第 3–7 天）傻基线**：去噪步数 k ∈ {32, 16, 8, 4, 2, 1}/block 的闭环 SR，116 任务 × 1 seed 先看趋势（论文已有 32 vs 16 两点：42.7 → 33.0）。若 k=4 仍不掉点，则本线 NO，停。
- **第 2 步（第 7–12 天）collect + shadow**：k=32 全算跑成功集入库（存截图 token 池化、指令 embedding、上一动作、各 ρ 对应的部分 unmask 状态与最终动作串）；再跑校准集，离线检索打分，对 tier ρ ∈ {0（hit）, 0.25, 0.5} 各算 D_a = 动作串标准化 edit distance（坐标类动作加像素距离），拟合 q_a(s) 看单调性与档间 order。产出：shadow AUROC（上一步分数预测本步命中）判断 gate 是否有对应物。
- **第 3 步（第 12–14 天）闭环 3 点**：δ 三档，各 116 任务 × 1 seed，画 (IR, SR) 与 k 基线同图。
- **评测成本**：AndroidWorld 单集分钟级（agent 每步秒级 + 模拟器），116 集/点 估 2–4 h/点（**不确定**，取决于模拟器并发）；ALFWorld 替身每点 < 30 min。
- 2 周内 A-lane 不开工；若要做，唯一便宜且有信息量的实验是：在 τ-bench/WebArena 的成功轨迹库上，离线统计"按上下文相似度检索到的历史 tool call 与全算 tool call 完全相同"的比例随 s 的曲线（就是单档 shadow），成本纯 CPU，但它只回答 hit-or-miss。

## 8. 结论

**MAYBE**（D-lane GO / A-lane 及 chat-RAG-reranker-embedding NO）：diffusion-LM agent（LLaDA-UI 一类）与我们的"前缀 + 迭代头 + 减步掉点 + 闭环 SR"四条件全部对上，且检索式 warm start 在那里还是空白；但 AR 世界里 warm 档没有中间状态、减步不掉点、hit-or-miss 已被 vCache/计划 cache/无损 SD 做掉，只剩"闭环漂移测量"一条可写。
