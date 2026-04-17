# Cloud-Edge Deployment & Inference Efficiency for Robot Policies — Related Work

> Bibliography for papers on robot policy cloud/edge deployment, hierarchical brain-cerebellum architectures, fleet-level serving, and compute/energy efficiency. These provide the deployment-context motivation for the OpenPI inference cache system.
>
> Compiled: 2026-04-16. Dates are first-arXiv-submission month (YYYY-MM) where applicable.

---

## 1. Cloud Robotics Frameworks (policy entirely or partially in the cloud)

Core infrastructure for offloading robot inference to remote servers.

| Paper | First Posted | Venue | Link |
|-------|--------------|-------|------|
| FogROS2: An Adaptive Platform for Cloud and Fog Robotics Using ROS 2 | 2023-07 | ICRA 2023 | https://www.researchgate.net/publication/372120857 |
| FogROS2-FT: Fault Tolerant Cloud Robotics | 2024-12 | IROS 2024 (arXiv 2412.05408) | https://arxiv.org/html/2412.05408v1 |
| FogROS2-Config: Choosing Server Configurations for Cloud Robotics | 2023-11 | ICRA 2024 (arXiv 2311.05600) | https://arxiv.org/html/2311.05600v2 |
| FogROS2-PLR: Probabilistic Latency-Reliability for Cloud Robotics | 2024-10 | arXiv 2410.05562 | https://arxiv.org/html/2410.05562 |
| A Fog Robotics Approach to Deep Robot Learning (Decluttering) | 2019 | ICRA 2019 (Berkeley) | https://goldberg.berkeley.edu/pubs/ICRA2019-ajay-fog-robotics-decluttering-final.pdf |

**Key results:** FogROS2 reduces SLAM latency 50%, grasp planning from 14s→1.2s, motion planning 28×. Now in official ROS 2 distribution. FogROS2-FT adds multi-cloud fault tolerance via stateless service replication.

---

## 2. Latency-Resilient Cloud-Robot Manipulation

Directly addresses the problem: when policy is remote, network latency kills control frequency.

| Paper | First Posted | Venue | Link |
|-------|--------------|-------|------|
| Speculative Policy Orchestration (SPO): A Latency-Resilient Framework for Cloud-Robotic Manipulation | 2026-03 | arXiv 2603.19418 | https://arxiv.org/abs/2603.19418 |
| Asynchronous Robot Inference: Decoupling Action Prediction and Execution | 2025 | HuggingFace blog | https://huggingface.co/blog/async-robot-inference |

**SPO highlights:** Cloud-hosted world model pre-computes and streams future waypoints to a local edge buffer; edge ε-tube verifier bounds kinematic deviation. Reduces network-induced idle time >60% vs blocking remote inference. Discards ~60% fewer predictions than static caching baselines.

**Relevance to our work:** SPO is a "speculative" approach (predict future, verify locally); our cache is a "retrieval" approach (find similar past, replay). Both solve the same latency problem from different angles. SPO needs a world model; our cache needs a demonstration library.

---

## 3. Hierarchical Brain-Cerebellum Architectures (heavy cloud + light edge)

"大脑在云，小脑在本地" — high-level reasoning remote, low-level control local.

| Paper | First Posted | Venue | Link |
|-------|--------------|-------|------|
| RoboOS: A Hierarchical Embodied Framework for Cross-Embodiment and Multi-Agent Collaboration | 2025-05 | arXiv 2505.03673 | https://arxiv.org/abs/2505.03673 |
| Hierarchical Policy Blending as Inference for Reactive Robot Control | 2022-10 | arXiv 2210.07890 | https://arxiv.org/html/2210.07890v3 |
| RoboMatrix: A Skill-centric Hierarchical Framework for Scalable Robot Task Planning and Execution | 2024-12 | arXiv 2412.00171 | https://arxiv.org/html/2412.00171v1 |
| Hierarchical Generative Modelling for Autonomous Robots | 2023 | Nature Machine Intelligence 2023 | https://www.nature.com/articles/s42256-023-00752-z |

**RoboOS:** Brain (RoboBrain MLLM, cloud) → Cerebellum Skill Library (edge, plug-and-play) → Real-Time Shared Memory (multi-agent sync). Validated in restaurants, households, supermarkets across single-arm, dual-arm, humanoid, and wheeled robots.

**Relevance:** Our stage-split architecture (stage1 GPU / stage2+3 meta) is a single-machine version of this pattern. The cache system enables the "cerebellum" to operate independently when the "brain" is unavailable.

---

## 4. On-Device / Edge Policy Efficiency

Making policies small/fast enough to run locally — the other side of the cloud-edge coin.

| Paper | First Posted | Venue | Link |
|-------|--------------|-------|------|
| On-Device Diffusion Transformer Policy for Efficient Robot Manipulation | 2025 | ICCV 2025 | https://openaccess.thecvf.com/content/ICCV2025/papers/Wu_On-Device_Diffusion_Transformer_Policy_for_Efficient_Robot_Manipulation_ICCV_2025_paper.pdf |
| FLOWER: Democratizing Generalist Robot Policies with Efficient VLA Flow Policies | 2025-09 | arXiv 2509.04996 | https://arxiv.org/html/2509.04996v1 |
| One-Step Flow Policy (OFP): Self-Distillation for Fast Visuomotor Policies | 2026-03 | arXiv 2603.12480 | https://arxiv.org/html/2603.12480 |
| Edge Computing and its Application in Robotics: A Survey | 2025-07 | arXiv 2507.00523 | https://arxiv.org/pdf/2507.00523 |

**Relevance:** These compress the model itself; our cache skips the model entirely for familiar observations. Complementary — a cache-miss can still fall back to a compressed on-device model.

---

## 5. Fleet-Level Serving & Scalability (one server → many robots)

Cache directly increases per-GPU throughput: cache hits cost ~0 FLOPs, freeing GPU for cache misses from other robots.

| Paper | First Posted | Venue | Link |
|-------|--------------|-------|------|
| Robot Fleet Learning via Policy Merging | 2024 | NeurIPS 2024 | https://openreview.net/forum?id=IL71c1z7et |
| RobotFleet: An Open-Source Framework for Centralized Multi-Robot Task Planning | 2025-10 | arXiv 2510.10379 | https://arxiv.org/html/2510.10379 |
| Managing a Fleet of AMRs using Cloud Robotics Platform | 2017-06 | arXiv 1706.08931 | https://arxiv.org/pdf/1706.08931 |
| Scalable Heterogeneous Robot Fleet-Based Task Automation in Hospital Environments | 2022 | Frontiers in Robotics and AI 2022 | https://www.frontiersin.org/articles/10.3389/frobt.2022.922835 |

---

## 6. Compute & Energy Efficiency (Green AI angle)

Inference cache as a "green AI" technique — skip redundant compute, save energy, serve more devices per watt.

| Paper | First Posted | Venue | Link |
|-------|--------------|-------|------|
| Small is Sufficient: Reducing the World AI Energy Consumption Through Model Selection | 2025-10 | arXiv 2510.01889 | https://arxiv.org/html/2510.01889 |
| Green AI Techniques for Reducing Energy Consumption in AI Systems | 2025 | ScienceDirect | https://www.sciencedirect.com/science/article/pii/S2590005625002796 |
| Towards Sustainable AI: A Comprehensive Framework for Green AI | 2024 | Discover Sustainability (Springer) | https://link.springer.com/article/10.1007/s43621-024-00641-4 |
| A Systematic Review of Green AI | 2023 | WIREs DMKD | https://wires.onlinelibrary.wiley.com/doi/10.1002/widm.1507 |
| Neural Circuit Policies for Estimating Energy Consumption | 2025-04 | arXiv 2504.02781 | https://arxiv.org/abs/2504.02781 |

**Framing for our work:** Model selection saves ~28% energy globally; our cache achieves the extreme case — for cache hits, inference energy is **zero** (only a vector lookup). At fleet scale with high hit rates, this translates to fewer GPUs powering the same number of robots.

---

## 7. Multi-Stage AI Inference Pipelines (systems perspective)

General infrastructure for optimizing pipelines with heterogeneous stages — directly maps to our 3-stage Pi0.5 pipeline.

| Paper | First Posted | Venue | Link |
|-------|--------------|-------|------|
| Understanding and Optimizing Multi-Stage AI Inference Pipelines (Hermes) | 2025 | MIT CSAIL | https://people.csail.mit.edu/suvinay/pubs/2025.hermes.arxiv.pdf |
| Orchestrating Embodied Systems through the Embodied Context Protocol | 2025 | Research (Science Partner Journal) | https://spj.science.org/doi/10.34133/research.1047 |

---

## Positioning Summary

Our inference cache is **deployment-agnostic** — it provides value in all three deployment topologies:

| Deployment | Cache Role | Primary Benefit |
|------------|-----------|-----------------|
| **Cloud (remote GPU)** | Cache on server, skip model for familiar observations | Higher throughput → fewer GPUs → lower cost & energy per robot |
| **Edge (local GPU)** | Cache on device, skip model for familiar observations | Lower latency → higher control frequency on limited hardware |
| **Cloud-Edge split** | Cache on edge as local "cerebellum" fallback | Latency resilience when network degrades; graceful degradation |

Key differentiators vs existing work:
- **vs SPO**: SPO predicts future speculatively (needs world model); we retrieve past (needs demonstration library). Complementary.
- **vs FogROS2**: FogROS2 optimizes the network/infra layer; we optimize the compute layer. Orthogonal — can stack.
- **vs On-Device compression**: They make the model smaller; we skip the model entirely for cache hits. Complementary — cache miss falls back to compressed model.
- **vs RoboOS brain-cerebellum**: Similar philosophy; our stage-split (stage1 GPU / stage2+3 meta) + cache is a concrete, single-machine implementation of this pattern.
