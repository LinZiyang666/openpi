# OpenPI & VLA-Adapter Integration Architecture

**AI Assistance Acknowledgment:** Portions of the codebase modifications and this documentation were developed with debugging and architectural assistance from Google Gemini.

---

## 1. Project Overview

This document outlines the setup, architecture, and code modifications required to successfully run the VLA-Adapter inference model within the OpenPI LIBERO evaluation infrastructure. By bridging these two distinct architectural paradigms, we enable seamless physical evaluation of autoregressive vision-language models within diffusion-first simulators.

---

## 2. Environment Setup

VLA-Adapter relies on specific dependencies that differ from the core OpenPI environment. It must be isolated in a dedicated Conda environment.

| Component | Installation Command |
| :--- | :--- |
| **Environment** | `conda create -n vla-adapter python=3.10 -y`| 
| **Activation** |`conda activate vla-adapter` |
| **Core ML** | `pip install torch==2.2.0 torchvision torchaudio` |
| **Data & Networking** | `pip install websockets tyro h5py numpy msgpack msgpack-numpy requests` |
| **Vision & Transformers** | `pip install transformers accelerate opencv-python-headless` |

**Repository Sourcing:**
The VLA-Adapter codebase is treated as an external module. It is cloned directly into the OpenPI `third_party` directory.

---

## 3. Core Serving Logic & Action Pipeline

The custom serving wrapper bridges OpenPI's WebSocket server and the VLA-Adapter. 

### Observation Dictionary Extraction
The VLA-Adapter's extraction utility strictly expects the key `"state"` in the observation dictionary. If missing, it triggers a `KeyError`. The wrapper explicitly maps `observation/state` to `"state"` to satisfy this requirement.

### Action Normalization & Gripper Post-Processing
To perfectly mirror the official >90% success rate evaluation pipeline, we rely on the model's native `dataset_statistics.json` applied internally by `get_action`, avoiding our own normalization errors. 
* **Gripper Thresholding:** We route the raw output through the official `normalize_gripper_action(binarize=True)` to fix continuous-to-binary latency.
* **Convention Alignment:** We utilize `invert_gripper_action` to resolve the mechanical mismatch between the RLDS dataloader (0 = close, 1 = open) and the OpenPI/MuJoCo physics engine (-1 = open, 1 = close).

---

## 4. The "True Hidden States" Monkey Patch

During integration, severe architectural mismatches were discovered deep within the Hugging Face `modeling_prismatic.py` implementation when compared to OpenPi's expectations. To bypass these without permanently altering the external library, we dynamically intercepted and patched `self.action_head.predict_action`.

| Technical Issue | Deployed Resolution |
| :--- | :--- |
| **Intermediate Layer Loss** | Hugging Face's default `forward` pass discards intermediate reasoning layers, blinding the Action Head's spatial awareness. | Implement a "Brain Scanner" interception to capture `_last_hidden_states_tuple` from the language model, preserving all 24 transformer blocks for the Action Head. |
| **Proprioception Drop** | The base model drops the `proprio` argument during the forward pass, causing a `NoneType` reshape crash. | Re-inject the formatted tensor manually from the policy class. |
| **Projector Mocking** | Vision-only checkpoints lack a dedicated `proprio_projector` layer. | Dynamically generate a `lambda` buffer to prevent "NoneType not callable" errors. |

---

## 5. HDF5 Data Collection Bypass

OpenPI’s native collector (`EpisodeDataCollector`) expects Diffusion-style denoising steps. Because the VLA-Adapter is an autoregressive model, the data pipeline required a translation wrapper (`VLACollectionPolicy`).

**Implementation Strategy:**
* Pass an empty list to `noise_action_steps`.
* Inject `np.zeros((1, 2048))` dummy arrays for `vision_embs` and `prompt_emb`.
* This prevents HDF5 writer crashes while cleanly preserving and writing the high-fidelity `robot_state` and `clean_action` data to the disk.

---

## 6. Running the Model
**6. Running the Model**

* Once your conda enviornment is active, boot up frpc as usual: 
```bash start_inner_services.sh ```

* Then, return to openpi directory and execute:
```MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH="src:third_party/libero" python scripts/serve_vla_adapter.py     --collect     --task-suite libero_spatial     --checkpoint-dir VLA-Adapter/LIBERO-Spatial-Pro```

*In the example above, libero_spatial was used, but you can replace this with the same openpi task suite.

* You may see some errors regarding accepting external code, you will have to manually adjust  each of these only the first time the model fails to load. All of these will be found in the VLA-Adapter thirdparty folder

* Launch as usual from local simulation, data will be exported into same data folder on ICRN.

## ---

