# Inference Cache / Retrieval-Augmented Control — Related Work

> Broad bibliography for papers relevant to the OpenPI fork's inference cache system. Organized by proximity to our setup (cross-trajectory, inference-time stage caching for Pi0.5 VLA via observation embedding retrieval).
>
> Compiled: 2026-04-16. Excludes LLM KV-cache literature. Dates below are the first-arXiv-submission month (YYYY-MM), derived from arXiv IDs; conference venues are noted where known.

---

## 1. Closest: Training-Free Retrieval as Control (inference-time, cross-trajectory)

Same paradigm as our `cache/` subsystem — embed observation → vector-DB lookup → reuse past actions.

| Paper | First Posted | Venue | Link |
|-------|--------------|-------|------|
| RT-Cache: Training-Free Retrieval for Real-Time Manipulation | 2025-05 | arXiv 2505.09040 | https://arxiv.org/html/2505.09040v3 |
| VINN: The Surprising Effectiveness of Representation Learning for Visual Imitation | 2021-12 | RSS 2022 (arXiv 2112.01511) | https://arxiv.org/abs/2112.01511 |
| MCNN: Memory-Consistent Neural Networks for Imitation Learning | 2023-10 | arXiv 2310.06171 | https://arxiv.org/html/2310.06171v2 |
| RoboRouter: Training-Free Policy Routing for Robotic Manipulation | 2026-03 | arXiv 2603.07892 | https://arxiv.org/html/2603.07892 |
| In-Context Imitation Learning via Next-Token Prediction | 2024-08 | arXiv 2408.15980 | https://arxiv.org/html/2408.15980v1 |

**Why closest:** these replace per-step model calls with retrieval-and-replay from a memory of past (observation, action) pairs, no fine-tuning required — identical mechanism to our cache path.

---

## 2. VLA Inference Acceleration (token / KV caching within a trajectory)

Shares the "VLA inference cache" naming space but caches tokens/KV inside one trajectory, not across trajectories.

| Paper | First Posted | Venue | Link |
|-------|--------------|-------|------|
| VLA-Cache: Efficient VLA Manipulation via Adaptive Token Caching | 2025-02 | arXiv 2502.02175 | https://arxiv.org/abs/2502.02175 |
| VLN-Cache: Token Caching for VLN with Visual/Semantic Dynamics Awareness | 2026-03 | arXiv 2603.07080 | https://arxiv.org/html/2603.07080 |
| BLURR: Boosted Low-Resource Inference for VLA | 2025-12 | arXiv 2512.11769 | https://arxiv.org/html/2512.11769v1 |
| AC2-VLA: Action-Context-Aware Adaptive Computation in VLA | 2026-01 | arXiv 2601.19634 | https://arxiv.org/html/2601.19634 |
| Fast ECoT: Efficient Embodied Chain-of-Thought via Thoughts Reuse | 2025-06 | arXiv 2506.07639 | https://arxiv.org/html/2506.07639v1 |
| ActionFlow: Pipelined Action Acceleration for VLMs on Edge | 2025-12 | arXiv 2512.20276 | https://arxiv.org/html/2512.20276 |
| A Survey on Efficient VLA Models | 2025-10 | arXiv 2510.24795 | https://arxiv.org/html/2510.24795v1 |
| Efficient VLA for Embodied Manipulation: A Systematic Survey | 2025-10 | arXiv 2510.17111 | https://arxiv.org/html/2510.17111v3 |

**Disambiguation priority:** VLA-Cache is the most frequently confused work. Must explicitly contrast axis of caching (intra-trajectory token vs. cross-trajectory stage output).

---

## 3. Diffusion / Flow Policy Denoising-Step Caching (continuous control)

Directly relevant to Pi0.5's 10-step Euler ODE flow matching segment.

| Paper | First Posted | Venue | Link |
|-------|--------------|-------|------|
| Block-wise Adaptive Caching (BAC) for Accelerating Diffusion Policy | 2025-06 | arXiv 2506.13456 | https://arxiv.org/html/2506.13456 |
| Sparse ActionGen (SAG): Accelerating Diffusion Policy with Real-time Pruning | 2026-01 | arXiv 2601.12894 | https://arxiv.org/abs/2601.12894 |
| Real-Time Chunking (RTC) — Execution of Action Chunking Flow Policies | 2025-06 | arXiv 2506.07339 (Physical Intelligence) | https://arxiv.org/abs/2506.07339 |
| Temporal Action Selector (TAS) for Action Chunking | 2025-11 | arXiv 2511.04421 | https://arxiv.org/html/2511.04421 |
| One-Step Flow Policy (OFP): Self-Distillation for Fast Visuomotor Policies | 2026-03 | arXiv 2603.12480 | https://arxiv.org/html/2603.12480 |
| Asynchronous Robot Inference: Decoupling Action Prediction and Execution | 2025 | HuggingFace blog | https://huggingface.co/blog/async-robot-inference |

**Note:** RTC is from Physical Intelligence (same lab as Pi0/Pi0.5) — mandatory related work citation.

---

## 4. Behavior Retrieval for Imitation Learning (training-time, same mechanism)

Same observation-similarity retrieval mechanism, but applied to training-data selection rather than inference-time action reuse.

| Paper | First Posted | Venue | Link |
|-------|--------------|-------|------|
| Behavior Retrieval: Few-Shot IL by Querying Unlabeled Datasets | 2023-04 | RSS 2023 (arXiv 2304.08742) | https://arxiv.org/abs/2304.08742 |
| SAILOR: Learning and Retrieval from Prior Data for Skill-based IL | 2022-10 | CoRL 2022 (arXiv 2210.11435) | https://arxiv.org/abs/2210.11435 |
| STRAP: Robot Sub-Trajectory Retrieval for Augmented Policy Learning | 2024-12 | arXiv 2412.15182 | https://arxiv.org/html/2412.15182v2 |
| IWR: Data Retrieval with Importance Weights for Few-Shot IL | 2025-09 | arXiv 2509.01657 | https://arxiv.org/html/2509.01657v1 |
| Collage: Adaptive Fusion-based Retrieval for Augmented Policy Learning | 2025-08 | arXiv 2508.01131 | https://arxiv.org/html/2508.01131 |
| ReMoBot: Retrieval-Based Few-Shot IL for Mobile Manipulation | 2024-08 | arXiv 2408.15919 | https://arxiv.org/html/2408.15919 |
| DataMIL: Selecting Data for Robot IL with Datamodels | 2025-05 | arXiv 2505.09603 | https://arxiv.org/html/2505.09603 |

---

## 5. Approximate Caching of Intermediate Denoising States (warm-start lineage)

Cross-request/cross-trajectory reuse of intermediate noisy states to skip early denoising steps.

| Paper | First Posted | Venue | Link |
|-------|--------------|-------|------|
| NIRVANA: Approximate Caching for Efficiently Serving Diffusion Models | 2023-12 | NSDI 2024 (arXiv 2312.04429) | https://arxiv.org/abs/2312.04429 |
| Streaming Diffusion Policy (SDP): Fast Policy Synthesis with Variable Noise | 2024-06 | arXiv 2406.04806 | https://arxiv.org/abs/2406.04806 |
| STEP: Warm-Started Visuomotor Policies with Spatiotemporal Consistency Prediction | 2026-02 | arXiv 2602.08245 | https://arxiv.org/abs/2602.08245 |
| Action-to-Action Flow Matching (A2A) | 2026-02 | arXiv 2602.07322 | https://arxiv.org/html/2602.07322 |
| Warm-Start Flow Matching for Guaranteed Fast Text/Image Generation | 2026-03 | arXiv 2603.19360 | https://arxiv.org/html/2603.19360 |

**NIRVANA** is the origin of the "approximate caching" idea: cache intermediate latent $I_K$ at specific denoising steps, match new requests by CLIP text similarity, skip up to 25/50 steps. Results: 21% GPU savings, 19.8% latency reduction on production workloads.

**SDP/STEP/A2A** apply warm-start to robot policies but are **within-episode only** (temporal continuity from previous control cycle), not cross-trajectory retrieval.

---

## 6. Background: Classical Memory / Primitive-Library Lineage

Older roots of the retrieval-based control idea.

| Paper | First Posted | Venue | Link |
|-------|--------------|-------|------|
| STITCHER: Real-Time Trajectory Planning with Motion Primitive Search | 2024-12 | arXiv 2412.21180 | https://arxiv.org/html/2412.21180v1 |
| A Framework for Learning and Reusing Robotic Skills | 2024-04 | arXiv 2404.18383 | https://arxiv.org/html/2404.18383 |
| Episodic RL with Expanded State-Reward Space | 2024-01 | AAMAS 2024 (arXiv 2401.10516) | https://arxiv.org/abs/2401.10516 |

---

## Positioning Summary

When writing related work, differentiate along three axes:

1. **Axis of caching** — we cache *stage outputs* across trajectories; VLA-Cache / VLN-Cache / BLURR / AC2-VLA / Fast ECoT cache *tokens/KV* within a trajectory.
2. **Intent** — we replace inference compute; Behavior Retrieval / SAILOR / STRAP / IWR / Collage use retrieval to select *training* data.
3. **Closest competitor** — RT-Cache (2025-05) and VINN (RSS 2022) share our exact mechanism. Differentiators to emphasize: Pi0.5 stage-level granularity (CP1/CP3), structured decision pipeline (Gate / Judge / WritePolicy), and meta-device support for skipping Stage 2/3 entirely.
