# Retrieval-Augmented Inference for Vision-Language-Action Models — Paper Workbench

> Living document for idea development, method drafts, and narrative shaping.
> Section status: `draft` → `wip` → `solid`. Update as ideas mature.
>
> Related bibliographies:
> - [inference_cache_related_work.md](inference_cache_related_work.md)
> - [cloud_edge_deployment.md](cloud_edge_deployment.md)

---

## 1. Elevator Pitch `wip`

We introduce an inference cache system for VLA robot policies — analogous to what semantic caching (GPTCache, etc.) does for LLM serving, but for continuous robot control. By retrieving and reusing cached intermediate representations for visually similar observations, we skip expensive model forward passes at inference time. This is deployment-agnostic: on edge it cuts latency for better control; on cloud it cuts compute so fewer GPUs serve more robots.

**One-liner draft:** "Semantic caching for robot brains — skip the model when you've seen this before."

---

## 2. Introduction Narrative (Problem → Gap → Contribution) `wip`

### P1 — The scaling dilemma: bigger models, thinner edges

VLA models are becoming the standard for generalist robot policies. Scaling laws hold — larger backbones yield more capable, more generalizable policies. But this creates a fundamental tension at the edge:

- **Edge hardware is expensive and under-utilized.** Equipping each robot with a high-VRAM GPU (A100, H100) is prohibitively costly. Utilization is structurally low — a robot arm is idle between tasks, during resets, and while waiting for human input, yet the GPU sits reserved. CapEx scales linearly with fleet size; utilization rarely exceeds 20–30%.
- **Models are outgrowing edge budgets.** A 2B-parameter VLA barely fits on a consumer GPU; the next generation will not. Edge deployment forces a choice between capability (run the big model slowly) and responsiveness (run a small model fast). Neither is satisfactory.
- **Cloud deployment is the natural answer** — and the trend. FogROS2, RoboOS, SPO, and industry players (Watney, Physical Intelligence) are all moving policy inference to remote GPU clusters, where hardware can be shared across robots, scaled elastically, and upgraded without touching the fleet. Cloud robotics is no longer speculative; it is the emerging default.

### P2 — The dual bottleneck of cloud serving

Cloud deployment trades the CapEx problem for two operational bottlenecks:

- **Latency.** VLA models (Pi0, Pi0.5, RT-2, Octo, etc.) take tens to hundreds of milliseconds per forward pass. On a remote GPU, network round-trip (5–50 ms LAN, 50–200 ms WAN) adds on top. For manipulation tasks requiring 10+ Hz control, the combined budget is razor-thin. Missed deadlines cause jerky motion, failed grasps, and unsafe contact.
- **Cost at scale.** Every robot in a fleet demands its own continuous stream of GPU inference. A single H100 can serve only a handful of robots at 10 Hz. As fleets grow from 10 to 1,000 to 10,000 robots, GPU cost and energy consumption scale linearly.

### P3 — The structural redundancy no one exploits (gap)

Most of robot tasks are highly repetitive. A policy executing pick-and-place in the same workspace encounters similar observations thousands of times. Adjacent frames within an episode are nearly identical; across episodes of the same task, the observation distribution is tightly clustered. Yet every inference request triggers a full model forward pass. This redundancy is wasted compute — a systematic inefficiency that current VLA serving stacks do not address.

### P4 — The LLM analogy: semantic caching

LLM serving faced a structurally similar problem — many queries are semantically equivalent — and the community responded with semantic caching layers (GPTCache, LangChain cache, Momento) that intercept requests, check embedding similarity, and return cached responses when appropriate. This has become standard infrastructure for production LLM deployments. **No equivalent exists for robot policy serving.**

### Why now (supporting signals)

- VLA models are getting bigger (2B+ parameters), making per-step inference more expensive.
- Cloud robotics is becoming the default deployment pattern (FogROS2, RoboOS, SPO).
- Robot fleets are scaling from lab to production (warehouses, restaurants, homes).
- The gap between model capability and real-time control budget is widening.

### P5 — Our contribution

We introduce [System Name], an inference cache system for VLA robot policies. By caching intermediate model representations and retrieving them for visually similar observations, the system skips expensive forward passes at inference time. The approach is deployment-agnostic:

- **On cloud servers**: cache hits free GPU cycles, increasing throughput — fewer GPUs serve more robots, reducing cost and energy.
- **On edge devices**: cache hits bypass model inference entirely, cutting latency and raising control frequency.

The system is model-agnostic, training-free, and plugs into existing VLA serving pipelines as a drop-in middleware layer.

### Contribution bullets (draft)

1. We identify the structural observation redundancy in robot policy serving and formalize the inference caching problem for VLA models.
2. We propose [System Name], a modular cache architecture featuring: (a) cascaded KeyBuilder gating — a lightweight CLIP check intercepts before the model runs, a precise vision-token check intercepts after Stage 1, yielding per-request adaptive caching depth from full skip to full inference; (b) graduated compute reuse across cache levels (CP1 / warm-start / CP3); (c) quality-aware decision pipeline (Gate → Search → Judge → WritePolicy).
3. We demonstrate that on [benchmark], the system achieves [X]% cache hit rate with negligible task success degradation, yielding [Y]× latency reduction (edge) and [Z]× throughput improvement (cloud).



---

## 3. Method Sketch `wip`

### 3.1 System overview: cache as pipeline middleware

The cache system wraps the VLA inference pipeline as a middleware layer. It intercepts observations before they reach the model and decides — per request — how much of the pipeline to skip.

```
Observation (raw images + state)
  │
  ▼
┌─────────────────────────────────────────────────────┐
│                CACHE MIDDLEWARE                       │
│                                                      │
│  ┌─────────────┐                                    │
│  │ CLIP KeyBld  │◄── raw image (no model needed)    │
│  └──────┬──────┘                                    │
│         │ query cache                                │
│         ▼                                            │
│  ┌─────────────┐     ┌──────────┐                   │
│  │    Gate      │────►│  Search  │                   │
│  └─────────────┘     └────┬─────┘                   │
│                           │ candidates               │
│                           ▼                          │
│                     ┌──────────┐                     │
│                     │  Judge   │                     │
│                     └────┬─────┘                     │
│                          │                           │
│              ┌───────────┼───────────┐               │
│              ▼           ▼           ▼               │
│         High conf    Mid conf    Low conf / miss     │
│          CP3        Warm-start    Continue to ──►    │
│       (skip all)   (skip S1+K)   Stage 1 below      │
│              │           │                           │
│              ▼           ▼                           │
│           Return      Return                         │
│           actions    partial +                       │
│                      run S2/3                        │
└──────────────────────────┬──────────────────────────┘
                           │ cache miss or low conf
                           ▼
                    ┌──────────────┐
                    │   Stage 1    │ Vision Encoder
                    └──────┬───────┘
                           │ vision tokens
                           ▼
                ┌─────────────────────┐
                │ Vision-token KeyBld  │◄── model's own representation
                └──────────┬──────────��
                           │ query cache (2nd chance)
                           ▼
                     ┌──────────┐
                     │  Judge   │ (more precise than CLIP)
                     └────┬─────┘
                          │
                ┌─────────┼──────────┐
                ▼         ▼          ▼
           High conf   Mid conf   Miss
            CP3       Warm-start   Full inference
          (skip S2/3) (skip K     (run S2+S3)
                       steps)
                          │
                          ▼
                   ┌──────────────┐
                   │  Stage 2/3   │ Flow Matching
                   └──────┬───────┘
                          │
                          ▼
                     ┌──────────┐
                     │WritePolicy│ → store result in cache
                     └──────────┘
                          │
                          ▼
                       Actions
```

### 3.2 Cascaded KeyBuilder gating (MoE KeyBuilder)

The key architectural insight: **which KeyBuilder you use determines where in the pipeline the cache check happens**, and mixing them creates a cascaded decision structure.

#### Two KeyBuilder types

| KeyBuilder | Input | Runs before | Embedding quality | Cost |
|-----------|-------|------------|-------------------|------|
| CLIP KeyBuilder | Raw image | Stage 1 (no model needed) | Good general-purpose visual similarity | Cheap (~5ms, small CLIP encoder) |
| Vision-token KeyBuilder (MeanPool/SpatialPool) | Stage 1 output (vision tokens) | Stage 2/3 (requires Stage 1) | Task-specific, model-native representation | Requires Stage 1 to have run |

#### Cascaded decision logic

The system implements a two-stage gating cascade:

**Stage A — CLIP gate (cheap, coarse, before model):**
1. Encode observation with CLIP KeyBuilder (lightweight, independent of VLA model).
2. Search cache for similar entries.
3. If high-confidence hit → skip entire pipeline (CP3 or full warm-start). Return cached result.
4. If uncertain or miss → fall through to Stage 1.

**Stage B — Vision-token gate (precise, after Stage 1):**
1. Run Stage 1 (vision encoder) — this is necessary anyway if CLIP didn't yield a high-confidence hit.
2. Encode vision tokens with Vision-token KeyBuilder (model's own representation).
3. Search cache again with this more precise key.
4. If hit → skip Stage 2/3 (warm-start from cached intermediate $x_t$, or CP3 with cached final actions).
5. If miss → full inference through Stage 2/3.

#### Why cascading beats single-KeyBuilder

- **CLIP-only** catches easy cases (identical or near-identical observations) cheaply, but may false-positive on visually similar but semantically different scenes (e.g., same workspace, different object pose).
- **Vision-token-only** is more precise (uses the VLA model's own learned representation), but requires running Stage 1 first — losing the biggest compute-saving opportunity.
- **Cascaded** gets the best of both: CLIP filters easy cases at near-zero cost; vision-token handles hard cases with model-grade precision. The system never pays for precision it doesn't need.




---

## 4. Experiment Plan `draft`

<!--
What experiments are needed to support each claim?
| Claim | Experiment | Metric | Status |
|-------|-----------|--------|--------|
-->



---

## 5. Related Work & Positioning `wip`

### 5.1 Taxonomy of existing caching approaches

We categorize prior work by the relationship between cache and model at inference time:

#### A. Retrieval-as-replacement (cache replaces model)

On cache hit, the policy model is completely bypassed; retrieved actions are executed directly.

| Work | What is cached | Retrieval key | Quality gate | Architecture |
|------|---------------|---------------|-------------|--------------|
| RT-Cache (2025) | Raw action trajectories | DINOv2 + SigLIP embedding → Qdrant | None — hit = replay | Model-free (no policy needed) |
| VINN (RSS 2022) | Demonstration (obs, action) pairs | BYOL embedding → kNN | None — weighted avg of k neighbors | Model-free |

**Limitation:** binary cache-or-compute. If the retrieved match is imperfect, the action is imperfect — no model correction. Quality depends entirely on retrieval precision and demonstration coverage.

#### B. Intra-trajectory token/feature reuse (cache inside model, within one episode)

Exploits redundancy between adjacent frames or adjacent denoising steps within a single inference trajectory. Does not reuse historical experience across episodes.

**B1. Frame-level KV reuse (across control cycles within one episode):**

| Work | What is cached | Scope | Model still runs? |
|------|---------------|-------|-------------------|
| VLA-Cache (2025) | KV of static visual tokens across adjacent frames | Intra-episode, frame-to-frame | Yes — full model with partial KV reuse |
| VLN-Cache (2026) | Visual/semantic token KV across navigation steps | Intra-episode, step-to-step | Yes |
| AC2-VLA (2026) | Temporal/spatial/depth redundancy within model | Intra-episode | Yes — adaptive layer/token skip |

**B2. Denoising-step-level block reuse (within a single action generation):**

| Work | What is cached | Scope | Model still runs? |
|------|---------------|-------|-------------------|
| BAC (2025) | Transformer block outputs (SA/CA/FFN activations) | Intra-generation, step-to-step | Yes — each step still runs, but cached blocks are skipped |
| SAG (2026) | Transformer block activations (zig-zag cross timestep+block) | Intra-generation | Yes — pruned blocks use cached activations |

**Precise mechanism of BAC/SAG (important — often confused with our approach):**
- BAC and SAG cache **internal network activations** (self-attention, cross-attention, FFN outputs), **NOT** the intermediate noisy action $x_t$ being denoised.
- They do **NOT** skip entire denoising steps. Every denoising step still executes; only specific transformer blocks within a step are replaced by cached outputs from a prior step.
- They are **intra-generation only** — each new action chunk generation starts cold with fresh noise. No cross-episode memory.
- This is fundamentally **network-level computation pruning**, not trajectory-level step skipping or experience reuse.

**Limitation:** no cross-trajectory knowledge reuse. Every new episode starts cold. The "cache" here is a computation shortcut within one forward pass sequence, not a memory of past experience.

#### C. Speculative decoding with retrieval draft (cache provides draft, model verifies)

Retrieves historical actions as speculative drafts; the VLA model still runs for verification (with optional relaxation).

| Work | Draft source | Verification | Architecture constraint |
|------|-------------|--------------|----------------------|
| HeiSD (2026) | Historical action sequences from Qdrant | VLA model verifies (with verify-skip on high-similarity) | **Autoregressive VLA only** — explicitly excludes diffusion/flow matching |
| Spec-VLA (2025) | Small drafter model | VLA model verifies (relaxed acceptance) | Autoregressive VLA only |

**Limitation:** tied to the draft-then-verify paradigm of speculative decoding, which requires autoregressive token generation. Cannot apply to flow matching models (Pi0, Pi0.5) or diffusion policies.

#### D. Approximate caching of intermediate denoising states (cross-request, skip denoising steps)

Caches the intermediate noisy latent $x_t$ at a specific denoising step and reuses it for similar new requests, skipping early denoising steps entirely.

| Work | Domain | What is cached | Similarity key | Steps skipped | Quality gate |
|------|--------|---------------|---------------|---------------|-------------|
| NIRVANA (NSDI'24) | Text-to-image diffusion | Intermediate latent $I_K$ at steps K∈{5,10,15,20,25} | CLIP text embedding cosine similarity | Up to 25/50 steps; K chosen by similarity threshold (>0.95→25, >0.9→20, >0.85→15) | CLIPScore α=0.9 threshold; human eval 79% satisfaction |

Approximate caching of intermediate diffusion states was first formalized by NIRVANA (NSDI'24) for text-to-image serving.

**Related robot-domain warm-start methods (within-episode, not cross-trajectory):**

| Work | Source of warm-start | Cross-trajectory? | Requires retraining? |
|------|---------------------|-------------------|---------------------|
| Streaming Diffusion Policy (SDP, 2024) | Previous control cycle's partially denoised action buffer | No — within-episode only | No |
| STEP (2026) | Predicted action from spatiotemporal consistency predictor | No — within-episode only | Yes (predictor training) |
| Action-to-Action Flow Matching (A2A, 2026) | Previous step's executed action in latent space | No — within-episode only | Yes (full model retraining) |

**Key distinction:** the robot warm-start methods above initialize from **temporal continuity** (previous control cycle within the same episode). Our warm-start extends this to **cross-trajectory retrieval** — the cached intermediate $x_t$ can come from a completely different episode of a similar task, found via observation embedding similarity. This enables warm-start even at episode start (no prior actions to warm-start from) and across different demonstrations of the same task.

**Relationship to our system:** warm-start is one operating mode within our graduated caching depth, sitting between CP1 and CP3:

| Mode | What is cached | What still runs | Steps saved |
|------|---------------|-----------------|-------------|
| CP1 | Stage 1 (vision encoding) | Stage 2 + 3 (all flow matching steps) | Vision encoder (~60% of compute) |
| **Warm start** | Stage 1 + intermediate $x_t$ at step K | Stage 2/3 from step K onward | Vision encoder + K flow matching steps |
| CP3 | Full pipeline (final actions) | Nothing | Everything |

#### E. Ours: stage-level cache with organic model integration

Cache operates at the boundary between model stages (e.g., after vision encoder, before action decoder). On cache hit, **the model still participates** — but only for the stages not covered by the cache. This enables configurable partial compute reuse:

| Cache level | What is cached | What still runs | Compute saved | Quality impact |
|------------|---------------|-----------------|---------------|---------------|
| CP1 (conservative) | Stage 1 output (vision encoding) | Stage 2 + 3 (action decoding via flow matching) | Heaviest stage skipped; actions still model-generated | Minimal — action decoder runs fresh on cached representation |
| CP3 (aggressive) | Full pipeline output (actions) | Nothing — direct replay | Maximum | Depends on retrieval quality |

**Key differentiators:**

1. **Graduated compute reuse, not binary.** CP1 caches the vision encoder output but still runs the action decoder — the most expensive stage is skipped, yet actions are still generated by the model with quality guarantees. CP3 goes full bypass for maximum speed. The operator chooses the trade-off per deployment.

2. **Quality-aware structured decision pipeline.** Gate (should we even search?) → Search (find candidates) → Judge (is the match good enough to trust?) → WritePolicy (should we remember this result?). No prior work has an explicit, configurable quality gate before cache retrieval. RT-Cache has no gate; HeiSD's verify-skip is post-hoc.

3. **Architecture-agnostic.** Works for flow matching (Pi0.5), diffusion policies, and autoregressive VLAs — any model with a staged encoder-decoder structure. HeiSD and Spec-VLA explicitly cannot handle non-autoregressive models.

4. **Cross-trajectory experience reuse.** Unlike VLA-Cache/BAC/SAG (intra-trajectory only), our cache accumulates experience across episodes, building a growing library of (observation, representation) pairs. More experience → higher hit rate → more compute saved.

### 5.2 Comparison matrix

|  | Cross-trajectory | Model-in-the-loop | Quality gate | Graduated reuse | Flow matching compatible | Domain |
|--|-----------------|-------------------|-------------|----------------|------------------------|--------|
| RT-Cache | Yes | **No** (full bypass) | No | No (all-or-nothing) | N/A | Robot |
| VINN | Yes | **No** (full bypass) | No | No | N/A | Robot |
| VLA-Cache | **No** (frame-to-frame) | Yes | No | No | Yes | Robot |
| BAC / SAG | **No** (step-to-step, block-level) | Yes (partial block skip) | No | No | Yes (diffusion) | Robot |
| HeiSD | Yes | Yes (verify) | Partial | No | **No** (AR only) | Robot |
| NIRVANA | Yes | Yes (re-condition) | Yes (CLIPScore) | Yes (K by similarity) | Yes (DDPM) | **Image gen** |
| SDP / STEP / A2A | **No** (within-episode) | Yes | No | Partial | Yes | Robot |
| **Ours** | **Yes** | **Yes** | **Yes** | **Yes (CP1–warm start–CP3)** | **Yes** | **Robot** |

### 5.3 Positioning statement (draft)

Existing robot-domain approaches treat caching and policy inference as mutually exclusive — either bypass the model entirely (RT-Cache, VINN) or optimize redundancy within a single forward pass (VLA-Cache, BAC/SAG). Speculative decoding methods (HeiSD, Spec-VLA) bridge the gap partially but are restricted to autoregressive architectures. In the image generation domain, NIRVANA (NSDI'24) pioneered approximate caching of intermediate diffusion states for cross-request serving, but targets one-shot image generation — not closed-loop robot control where cumulative errors and real-time stability constraints apply.

We propose the first system that integrates cross-trajectory experience caching into a robot policy's staged inference pipeline, enabling graduated compute reuse — from conservative (cache vision encoder, run action decoder fresh) through warm-start (cache intermediate flow matching state, skip early denoising steps) to aggressive (cache final actions) — with quality-aware retrieval decisions, across any VLA architecture including flow matching models.



---

## 6. Limitations & Venue Gap Analysis `wip`

### Current story ceiling

As it stands, the story is a **systems contribution**: we identified a gap (no semantic cache for robot policy serving), designed a system to fill it (cascaded gating, stage-level caching, quality pipeline), and applied it to VLA models. This is sufficient for systems venues (MLSys, NSDI, CoRL, ICRA) but insufficient for top ML venues (NeurIPS, ICML, ICLR), which expect a scientific insight — not just a well-designed system.

### What's missing for NeurIPS / ICLR

The gap is not more modules or features. It's a **deeper question** that the cache system helps answer. Three candidate directions, each answering a different fundamental question:

**Direction A — Quantify "how much compute is wasted"** (easiest to execute)

Run a battery of robot tasks, record every (observation, action) pair during inference. Post-hoc analysis: how many observation pairs are "similar enough"? How close are their corresponding actions? If the finding is "70% of inference calls in pick-and-place produce actions within ε of a previously computed result" — that number alone is a contribution no one has reported. The cache system then becomes the natural consequence: "we proved this much redundancy exists, so of course we should exploit it."

Core question: *How much of robot policy inference is actually redundant?*

**Direction B — Provide a safety guarantee for reuse** (hardest, highest payoff)

Reviewers will ask: "how do you know cached actions won't crash the robot?" Direction B answers this: if two observations' embedding distance < ε, then the model's action output difference is bounded by δ. Prove this mathematically (even a loose Lipschitz-type bound), or at minimum empirically measure the ε→δ mapping. This turns Gate/Judge thresholds from hand-tuned hyperparameters into theoretically grounded decisions.

Core question: *Under what conditions is a cached action safe to reuse?*

**Direction C — Justify why the cut point is correct** (medium difficulty, clean story)

We cache at the Stage 1 / Stage 2 boundary — but why there and not elsewhere? Direction C quantifies "representation stability" per stage: perturb the observation slightly, measure how much Stage 1 output changes, how much Stage 2 intermediate states change, how much the final action changes. If Stage 1 output is stable (drift < 1%) but Stage 2/3 output is sensitive (drift > 10%), that's a principled proof: the Stage 1 boundary is a natural cache point dictated by the model's internal representation structure, not an arbitrary design choice.

Core question: *Why does stage-level caching work at this boundary and not others?*

### Direction comparison

| | Core question | Difficulty | Contribution type | What it proves |
|--|--------------|-----------|-------------------|---------------|
| A | How much redundancy exists? | Low (data collection + statistics) | Empirical finding | The problem is worth solving |
| B | When is reuse safe? | High (math + experiments) | Theoretical + empirical | The solution is principled |
| C | Why cache at this boundary? | Medium (ablation experiments) | Empirical insight | The design is well-motivated |

Pursuing any one of these would elevate the paper from systems contribution to scientific contribution. A+C together is likely the most practical combination for a NeurIPS submission.

### Venue recommendations

Assuming work cannot be completed before end of June 2026. Deadlines marked (est.) are estimated from prior years; official dates not yet announced.

#### Tier 1 — Best fit

| Conference | Field | Tier | Deadline | Fit | Notes |
|-----------|-------|------|----------|-----|-------|
| **CoRL 2026** | Robot Learning | Top | **2026-05-29** (paper) | **Highest** | Reviewers understand VLA natively. Sim experiments standard. 8 pages. Austin, Nov 9–12. **6 weeks — very tight, likely X unless core work nearly done.** |
| **ICRA 2027** | Robotics & Automation | Top | **~2026-09** (est.) | High | Largest robotics venue, broad scope, system contributions welcome. 6 pages. Deadline typically mid-September. |

**Why these:** robot-domain venues where reviewers understand the problem without needing education. CoRL is particularly friendly to learning-system work; ICRA has the widest acceptance surface.

#### Tier 2 — Possible with extra work

| Conference | Field | Tier | Deadline | Fit | Notes |
|-----------|-------|------|----------|-----|-------|
| ~~NeurIPS 2026~~ | ML (general) | Top | ~~2026-05-06~~ | Medium | **X — deadline in 3 weeks, not ready.** Needs Direction A/B/C. |
| **ICLR 2027** | ML (general) | Top | **~2026-10** (est.) | Medium | More open to interesting ideas than ICML. Still needs scientific insight beyond system design. Deadline typically late September/early October. |
| ~~IROS 2026~~ | Intelligent Robots & Systems | Strong | ~~2026-03-02~~ | High | **X — already passed.** IROS 2027 deadline ~March 2027, fallback option. |
| **RSS 2027** | Robotics (selective) | Top (small) | **~2027-01** (est.) | Medium-low | Very selective (~100 papers/year). Prefers deeper insight over systems. Hard sell without Direction B or C. |

#### Tier 3 — Mismatch or timing issues

| Conference | Field | Tier | Deadline | Fit | Notes |
|-----------|-------|------|----------|-----|-------|
| ~~MLSys 2026~~ | ML Systems | Mid | ~~2025-10-30~~ | Medium | **X — already passed.** MLSys 2027 ~Oct 2026. Prestige limited. |
| NSDI 2027 | Networked Systems | Top | **~2026-09** (est.) | Low | Workload scale insufficient for systems reviewers. Cloud robotics is still niche, not a production workload. Would need real multi-robot serving testbed. |
| ~~ICML 2026~~ | ML (methods) | Top | ~~Passed~~ | Low | **X — already passed.** Needs theoretical contribution anyway. |

#### Recommended timeline

| Target | Deadline | Feasibility |
|--------|----------|-------------|
| ~~CoRL 2026~~ | ~~May 29, 2026~~ | **X — 6 weeks, user confirmed not ready by end of June.** |
| ICRA 2027 | ~Sep 2026 | **Best target** — 5 months to complete system + experiments. |
| ICLR 2027 | ~Oct 2026 | Stretch goal — needs Direction A or C on top of system work. |
| IROS 2027 | ~Mar 2027 | Safe fallback if ICRA doesn't work out. |

### Other risks

- **Warm-start correctness**: cross-trajectory $x_t$ reuse for flow matching is an unverified assumption. If it doesn't hold, graduated depth collapses to two levels (CP1 / CP3), weakening the story.
- **RT-Cache comparison**: need to demonstrate scenarios where model-in-the-loop caching outperforms full-bypass retrieval, otherwise reviewers will ask "why not just use RT-Cache?"
- **"Deployment-agnostic" claim**: claiming both cloud throughput and edge latency benefits without separate evaluations for each will not convince reviewers. Pick one as the main story, treat the other as bonus.

---

## 7. Open Questions `draft`

- [ ] Warm-start feasibility: does cross-trajectory $x_t$ actually produce valid flow matching continuations?
- [ ] Which of Direction A/B/C to pursue for NeurIPS-level framing?
- [ ] Primary deployment story: cloud throughput or edge latency?
- [ ] System name — [System Name] placeholder needs a real name
- [ ] Benchmark selection for evaluation


