# OpenPI Model & Training Reference

Detailed reference for project structure, model architecture, training configs, and deployment. For development workflow see [CLAUDE.md](../CLAUDE.md).

---

## Project Structure

```
openpi/
├── src/openpi/
│   ├── models/                  # JAX/Flax model implementations
│   │   ├── pi0.py               #   Pi0/Pi0.5 (flow matching VLA)
│   │   ├── pi0_fast.py          #   Pi0-FAST (autoregressive FSQ variant)
│   │   ├── pi0_config.py        #   Pi0Config (pi05=True/False flag)
│   │   ├── gemma.py             #   Gemma LLM backbone
│   │   └── siglip.py            #   SigLIP vision encoder
│   │
│   ├── models_pytorch/          # PyTorch model port (mirrors JAX)
│   │   ├── pi0_pytorch.py       #   PI0Pytorch - main model (staged API: embed_prefix/suffix/denoise_step)
│   │   ├── gemma_pytorch.py     #   PaliGemmaWithExpertModel (HF-based)
│   │   └── transformers_replace/#   Modified HF modules for adaRMSNorm
│   │
│   ├── policies/                # Inference wrappers (model + transforms)
│   │   ├── policy.py            #   Policy class (handles JAX & PyTorch via is_pytorch flag)
│   │   └── policy_config.py     #   create_trained_policy() factory
│   │
│   ├── cache/                   # [Fork] Inference cache system
│   │   ├── interceptor.py       #   InferenceInterceptor - cache-aware Policy wrapper
│   │   ├── orchestrator.py      #   CacheOrchestrator - CP1/CP3 end-to-end loop
│   │   ├── config.py            #   CacheConfig / SearchConfig dataclass tree + YAML loading
│   │   ├── timing.py            #   SystemTimer - CUDA Event / PerfCounter timing
│   │   ├── types.py             #   CheckpointID, field name constants
│   │   ├── storage_types.py     #   CacheEntry, CachePayload, QuerySpec
│   │   ├── backend_base.py      #   VectorStoreBackend ABC
│   │   ├── cache_storage.py     #   CacheStorage facade
│   │   ├── backends/
│   │   │   ├── in_memory_backend.py  # InMemoryBackend - pickle artifact, cosine/L2 search
│   │   │   └── qdrant_backend.py     # QdrantBackend - Qdrant vector DB
│   │   └── components/
│   │       ├── key_builder.py   #   KeyBuilder ABC + MeanPool/SpatialPool reducers
│   │       ├── clip_key_builder.py # CLIPKeyBuilder - open_clip vision encoder
│   │       ├── search_strategy.py  # SearchStrategy ABC + WeightedRrfKnn/WeightedScoreSum
│   │       ├── gate.py          #   Gate ABC + AlwaysSearch/ThresholdGate
│   │       ├── judge.py         #   Judge ABC + AlwaysHit/ThresholdJudge
│   │       └── write_policy.py  #   WritePolicy ABC + AlwaysWrite
│   │
│   ├── collect/                 # [Fork] Data collection via forward hooks
│   │   ├── collection_policy.py #   CollectionPolicy - captures embeddings per inference
│   │   └── data_collector.py    #   EpisodeDataCollector - buffers & writes HDF5
│   │
│   ├── training/                # Training infrastructure
│   │   └── config.py            #   TrainConfig, named configs (_CONFIGS)
│   │
│   ├── serving/
│   │   └── websocket_policy_server.py
│   │
│   └── transforms.py            # Transform pipeline (repack → normalize → tokenize → model → unnormalize)
│
├── scripts/
│   ├── train.py                 # JAX training
│   ├── train_pytorch.py         # PyTorch DDP training [Fork]
│   ├── serve_policy.py          # Policy server (supports --collect)
│   └── compute_norm_stats.py
│
├── exp/                         # [Fork] Experiment scripts, grouped by experiment
│   ├── common/                                   # Shared helpers (RPC, subprocess, run state, unit key)
│   ├── cache_experiment/                         # CP1 cache experiment pipeline
│   │   ├── build_in_memory_cache_artifact.py     # Build InMemoryBackend pickle from HDF5 (mean/spatial pool)
│   │   ├── build_clip_cache_artifact.py          # Build InMemoryBackend pickle from HDF5 (CLIP encoder)
│   │   ├── generate_cache_run_yamls.py           # Generate YAML configs for cache experiment grid
│   │   ├── run_cache_experiments.py              # Automated cache experiment runner (Phase 1/1.5/2)
│   │   ├── analyze_cache_results.py              # Parse experiment results from state JSON
│   │   ├── calibrate_robot_state_tau.py          # Calibrate L2→similarity tau for robot_state
│   │   └── calibrate_score_sum_stats.py          # Calibrate percentile stats for WeightedScoreSum
│   ├── trajectory_deviation/                     # Trajectory deviation corrective experiment
│   │   ├── run_step1b_gt.py                      # Step 1b: collect GT trajectories
│   │   ├── compute_deviate_scores.py             # Step 2: compute per-cycle deviate scores
│   │   ├── run_spawn_experiment.py               # Step 3: spawn from high-score cycles
│   │   └── analyze_deviation_results.py          # Plot deviate-score analyses
│   ├── temporal_prune/                           # Temporal prune experiment
│   │   └── generate_temporal_prune_yamls.py      # Batch-generate temporal-prune YAML configs
│   └── qdrant_step_knn/                          # Qdrant step-KNN retrieval experiment
│       ├── qdrant_ingest_openpi.py               # Ingest HDF5 → Qdrant
│       ├── qdrant_step_knn_experiment.py         # KNN retrieval benchmark
│       ├── qdrant_verify_openpi.py               # Verify ingested collection self-query
│       ├── toy_stage1_server.py                  # Stage1-only server for retrieval experiments
│       └── toy_qdrant_server.py                  # Qdrant query server
│
├── packages/openpi-client/      # Standalone client library (minimal deps)
├── examples/                    # Robot-specific examples (aloha, libero, droid, etc.)
├── docs/                        # Architecture & usage guides (see docs/README.md)
└── logs/                  # Implementation logs (see logs/README.md)
```

## Three Model Variants

| Model | Type | Action Representation | Config Flag |
|-------|------|-----------------------|-------------|
| **Pi0** | Flow matching | Continuous (Euler ODE) | `pi05=False` |
| **Pi0.5** | Flow matching + co-training | Continuous (Euler ODE) | `pi05=True` |
| **Pi0-FAST** | Autoregressive | Discrete tokens (FSQ) | `Pi0FASTConfig` |

All share: **PaliGemma** (SigLIP 400M + Gemma 2B) + **Action Expert** (Gemma 300M).

## Pi0 / Pi0.5 Core Architecture

```
Input: images + language prompt + robot state + noise x_t (t=1.0)
  |
  +-- [once] embed_prefix() -> KV Cache
  |     Images -> SigLIP -> visual tokens
  |     Language prompt -> Gemma embedding -> text tokens
  |     (Pi0.5: state discretized into text tokens here)
  |
  +-- while t in [1.0 -> 0.0]:  (default 10 Euler steps)
         embed_suffix(x_t, t) -> action tokens
         Gemma([prefix KV cache | suffix]) -> v_t
         x_t = x_t + dt * v_t
  |
Output: x_0 (denoised actions, shape [B, action_horizon, action_dim])
```

## Pi0 vs Pi0.5 Key Differences

| Aspect | Pi0 | Pi0.5 |
|--------|-----|-------|
| **State input** | Continuous vector in suffix | Discretized to text tokens in prefix (256 bins) |
| **Timestep** | Concatenated with actions -> MLP | Separate MLP -> adaRMSNorm on every layer |
| **Suffix tokens** | state(1) + action(50) = 51 | action(50) only |
| **max_token_len** | 48 | 200 |
| **Normalization** | z-score | Quantile (q01/q99) |

## Key Code Paths

| File | Key Methods |
|------|-------------|
| `models/pi0.py` (JAX) | `embed_prefix()`, `embed_suffix()`, `compute_loss()`, `sample_actions()` |
| `models_pytorch/pi0_pytorch.py` | `embed_prefix()`, `embed_suffix()`, `forward()`, `sample_actions()`, `denoise_step()` |

## Transform Pipeline

```
Raw data -> repack -> data_transforms -> Normalize -> model_transforms
  -> Model.sample_actions()
  -> model_transforms.out -> Unnormalize -> data_transforms.out -> repack.out -> action
```

## Pi0.5 Two-Stage Training (from paper)

**Stage 1 - Pre-training (280k steps):** Standard autoregressive next-token prediction on heterogeneous data:
- Mobile manipulator data (MM, ~400h from ~100 homes)
- Non-mobile robot data (ME, diverse environments)
- Cross-embodiment lab data (CE)
- High-level subtask prediction (HL)
- Web data (WD: captioning, VQA, object localization)
- Actions represented as discrete FAST tokens

**Stage 2 - Post-training (80k steps):** Specializes for mobile manipulation:
- Adds action expert (random init) for flow matching
- Joint training: next-token for text + flow matching for actions (alpha=10.0)
- Uses MM + ME data, web data (WD), high-level labels (HL), verbal instructions (VI)
- At inference: first predicts high-level subtask (text), then low-level actions (flow matching)

## Pi0.5 Hierarchical Inference (Paper Concept — Not in Current PyTorch Path)

> **Note:** The paper describes a two-stage hierarchical inference. The current PyTorch implementation (`models_pytorch/pi0_pytorch.py`) does **not** include the high-level autoregressive subtask generation. Only low-level flow matching is implemented. See `logs/archive/step1.log` for details.

As described in the paper, Pi0.5 would perform two-stage inference with the **same model**:
1. **High-level:** Given observation + high-level prompt (e.g., "clean the kitchen"), auto-regressively predict a subtask (e.g., "pick up the plate")
2. **Low-level:** Given observation + subtask as prompt, run flow matching to produce continuous action chunks

## Attention Structure

```
Prefix (images + text + discrete state)        Suffix (action tokens)
+------------------------------------+         +------------------------+
| img_tok ... lang_tok ...           | <------ | act_0  act_1 ... act_49|
| Bidirectional attention (ar=False) |         | Causal attention       |
| PaliGemma 2B processes these       |    X    | Action Expert 300M     |
|                                    |         | + adaRMSNorm(time_emb) |
+------------------------------------+         +------------------------+
      ^ KV Cache (computed once)                       ^ Recomputed each step

Action tokens attend to prefix (via KV cache). Prefix tokens cannot attend to action tokens.
```

## Flow Matching Training

- Noise schedule: `t ~ Beta(1.5, 1)` clipped to [0.001, 0.999]
- Interpolation: `x_t = t * noise + (1-t) * clean_actions`
- Target: velocity field `u_t = noise - clean_actions`
- Loss: MSE between predicted and target velocity field

## Named Training Configs

Defined in `src/openpi/training/config.py` (_CONFIGS list). Key configs:

**Inference-ready (pre-trained checkpoints):**
- `pi0_aloha`, `pi05_aloha` — ALOHA robot
- `pi0_aloha_sim`, `pi05_aloha_sim` — ALOHA simulation
- `pi0_droid`, `pi05_droid`, `pi0_fast_droid` — DROID robot
- `pi0_aloha_towel`, `pi0_aloha_tupperware` — Task-specific ALOHA

**Fine-tuning:**
- `pi0_libero`, `pi05_libero` — LIBERO full fine-tuning
- `pi0_libero_low_mem_finetune` — LIBERO LoRA fine-tuning (Pi0)
- `pi0_fast_libero`, `pi0_fast_libero_low_mem_finetune` — FAST variant for LIBERO
- `pi0_aloha_pen_uncap`, `pi05_aloha_pen_uncap` — ALOHA pen uncap
- `pi05_full_droid_finetune`, `pi05_droid_finetune` — DROID fine-tuning
- `pi0_fast_full_droid_finetune` — DROID FAST fine-tuning

**Debug:**
- `debug`, `debug_pi05` — Small models for testing

**Base checkpoints (GCS):**
- `gs://openpi-assets/checkpoints/pi0_base/params`
- `gs://openpi-assets/checkpoints/pi05_base/params`

## Deployment Options

**1. Direct inference:**
```python
from openpi.policies.policy_config import create_trained_policy
policy = create_trained_policy(config, checkpoint_dir)
actions = policy.infer(observation)
```

**2. Client-server (WebSocket):**
```python
# Server
from openpi.serving import WebsocketPolicyServer
server = WebsocketPolicyServer(policy, port=8000)

# Client (separate machine, minimal deps)
from openpi_client import WebsocketClientPolicy
client = WebsocketClientPolicy(host="server_ip", port=8000)
actions = client.infer(obs)
```

**3. Runtime loop (for robot control):**
```python
from openpi_client.runtime import Runtime
runtime = Runtime(environment, agent, subscribers, max_hz=10)
runtime.run()
```

## Hardware Requirements

- **Inference (full model):** 8GB+ VRAM
- **Inference (stage1-only with cache):** ~2GB VRAM (stage2/3 on meta)
- **LoRA fine-tuning:** 22.5GB+ VRAM
- **Full fine-tuning:** 70GB+ VRAM (multi-GPU recommended)
- **Multi-GPU:** JAX uses FSDP; PyTorch uses DDP via torchrun

### Per-Stage Device Placement

`serve_policy.py` supports assigning each inference stage to a different device:

```bash
# Stage1-only GPU (cache always-hit mode, ~2GB VRAM)
uv run scripts/serve_policy.py \
    --stage1_device cuda:0 --stage2_device meta --stage3_device meta \
    --cache_config cache.yaml ...

# Stage1 GPU + Stage2/3 CPU (low-speed fallback)
uv run scripts/serve_policy.py \
    --stage1_device cuda:0 --stage2_device cpu --stage3_device cpu \
    --cache ...
```

Devices: `cuda:N`, `cpu`, `meta` (zero memory, not callable).
Constraints: all three must be set together; `stage1` cannot be `meta`;
split/meta requires `--cache` or `--cache_config`; meta requires `--cache_config`.
See `StageDeviceConfig` in `src/openpi/models_pytorch/stage_device_placement.py`.

## Dependencies

- **Primary:** JAX 0.5.3, Flax 0.10.2 (original framework)
- **Secondary:** PyTorch 2.7.1, Transformers 4.53.2 (PyTorch port)
- **Vision:** SigLIP (via PaliGemma)
- **Data:** LeRobot for dataset management, Orbax for checkpoints
- **Tracking:** WandB
- **CLI:** Tyro for config-as-CLI
