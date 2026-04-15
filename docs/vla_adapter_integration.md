# VLA-Adapter Integration Guide

VLA-Adapter runs on ICRN while the LIBERO simulator runs on local WSL2, communicating via WebSocket — the same architecture used for π0.5.

```
[Local WSL2]                        [GPU Server — ICRN]
  LIBERO simulator  <--websocket-->  VLA-Adapter inference
  main.py                            serve_vla_adapter.py
  port 9001 (via frp)                port 8001
```

---

## 1. What Is VLA-Adapter

VLA-Adapter is a lightweight Vision-Language-Action model by the OpenHelix Team. It achieves competitive performance using only a 0.5B-parameter Qwen2.5 backbone with a Bridge Attention mechanism, compared to π0.5's 2B+ PaliGemma backbone with flow matching.

Pre-trained LIBERO checkpoints are available on HuggingFace:

- `VLA-Adapter/LIBERO-Spatial-Pro`
- `VLA-Adapter/LIBERO-Object-Pro`
- `VLA-Adapter/LIBERO-Goal-Pro`
- `VLA-Adapter/LIBERO-Long-Pro` (equivalent to libero_10)

---

## 2. Model Comparison: π0.5 vs VLA-Adapter

| Aspect | π0.5 | VLA-Adapter |
|--------|------|-------------|
| Backbone | PaliGemma (SigLIP 400M + Gemma 2B) | Qwen2.5-0.5B + DINOv2 + SigLIP |
| Action Representation | Flow matching (10 denoising steps) | L1 regression with action chunking |
| Parameters | ~2.7B | ~0.5B backbone + ~97M policy |
| Action Horizon | 50 steps | 8 steps (num_open_loop_steps) |
| Action Dimension | 32 | 7 (LIBERO-specific) |
| State Input | Discretized to text tokens (256 bins) | 8-dim proprio vector |
| Framework | JAX + PyTorch port | PyTorch only |
| Serving | serve_policy.py (WebSocket built-in) | Direct script — WebSocket to be added |

---

## 3. What Is Similar

### Image Preprocessing
Both models use identical preprocessing for LIBERO observations:

```python
img = obs["agentview_image"][::-1, ::-1]  # rotate 180 degrees to match training
```

Both use a wrist camera image (`robot0_eye_in_hand_image`) with the same rotation. No conversion needed in the server wrapper.

### Robot State
Both use the same 8-dimensional proprioceptive state vector:
- `eef_pos`: 3 dimensions
- `quat2axisangle(eef_quat)`: 3 dimensions
- `gripper_qpos`: 2 dimensions

The `quat2axisangle` conversion function is also identical in both codebases.

### LIBERO Environment
Both use the same `OffScreenRenderEnv`, the same task suite structure, and the same `max_steps` per suite. No changes to the WSL2 LIBERO environment are needed.

### Communication
The existing WebSocket + FRP tunnel is fully reusable. The WSL2 `main.py` connects to either server identically — only the host/port changes. No modifications to `main.py` are needed.

### Action Chunking
Both return a chunk of actions and execute them open-loop before requerying. The `main.py` action queue pattern (`collections.deque`) handles both naturally.

---

## 4. What Is Different

### Action Dimension
π0.5 outputs 32-dimensional actions (padded for generality). VLA-Adapter outputs 7-dimensional actions for LIBERO. This needs to be verified against what `main.py` passes to `env.step()`.

### Model Loading
π0.5 uses `policy_config.create_trained_policy()`. VLA-Adapter uses its own `initialize_model()` which loads the VLM, `action_head`, and `proprio_projector` as separate components — all must be loaded at server startup.

### Inference Call
π0.5:
```python
policy.infer(observation)  # returns dict with 'actions' key
```

VLA-Adapter:
```python
get_action(cfg, model, observation, task_description,
           action_head=action_head,
           proprio_projector=proprio_projector)  # returns numpy action array directly
```

### No Built-in WebSocket Server
π0.5 has `serve_policy.py` which handles WebSocket serving and episode lifecycle. VLA-Adapter runs inference locally via `torchrun` with no equivalent server. A new `serve_vla_adapter.py` must be written.

### Dependencies
VLA-Adapter requires a separate conda environment (Python 3.10) with `torch==2.2.0`, `flash-attn==2.5.5`, and the prismatic VLM library. It cannot share the OpenPI `uv` environment.

---

## 5. Implementation Plan

### Repository Structure

VLA-Adapter is cloned into `third_party/` to stay consistent with existing structure:

```
openpi/
├── third_party/
│   ├── libero/
│   └── VLA-Adapter/      ← cloned here
└── scripts/
    ├── serve_policy.py
    └── serve_vla_adapter.py   ← new file
```

### serve_vla_adapter.py

The wrapper has three responsibilities:

1. **Model loading** — calls `initialize_model()` at startup, loading the VLM, `action_head`, and `proprio_projector`
2. **WebSocket server** — accepts connections from WSL2 `main.py` on port 8001 via FRP tunnel
3. **Observation conversion** — receives openpi-format observations, converts to VLA-Adapter format, calls `get_action()`, returns action chunk

### Observation Conversion

The conversion is minimal because the formats are nearly identical:

| Field | Sent by main.py | Expected by VLA-Adapter | Conversion |
|-------|----------------|------------------------|------------|
| Primary image | `observation/image` (224x224 uint8) | agentview rotated 180 | None — already preprocessed |
| Wrist image | `observation/wrist_image` (224x224 uint8) | eye_in_hand rotated 180 | None — already preprocessed |
| Robot state | `observation/state` (8-dim float) | 8-dim proprio vector | None — same dimensions |
| Task prompt | `prompt` (string) | `task_description` (string) | None — pass through |

---

## 6. Running

Start the VLA-Adapter server on ICRN:

```bash
conda activate vla_adapter
cd ~/openpi
python scripts/serve_vla_adapter.py \
    --task_suite libero_spatial \
    --port 8001
```

Run the LIBERO simulator on local (no changes to main.py needed):

```bash
MUJOCO_GL=egl python examples/libero/main.py \
    --args.host 155.98.36.13 \
    --args.port 9001 \
    --args.task-suite-name libero_spatial \
    --args.num-trials-per-task 10
```

---

## 7. References

- VLA-Adapter GitHub: https://github.com/OpenHelix-Team/VLA-Adapter
- VLA-Adapter HuggingFace: https://huggingface.co/VLA-Adapter
- OpenPI Reference: `docs/openpi_reference.md`
