# CLIP retrieval-key ablation — r08_abl

Four pi05 arms are ready with fresh fit artifacts. All eight definitions are emitted; the four GR00T arms are blocked because the original libraries and sampled build HDF5s contain no images. Do not schedule those four arms. The shared encoder itself works in both h100 serving venvs without installing packages.

## Exact A specification

`r06/p2_ablations/make_arms.py:sources()` supplies the eight kind-A rows. Only `name`, `method`, and the existing `--os-fit-artifact` value change. `arms_clip.json` is separate from the concurrent guard-ablation `arms_in.json`; all original kwargs, flag order, absent fields and cost-ledger choices are retained.

A is `BlindAWM` with joint features, fit_data=same, full-rank codes, early step-0 metric, k=16, nn=3, lam=.1, state_scale=1, lam_c=.5, no insurance, serving=anchor_tail, budget=1 and gates=budget_only. kref=5 for current libraries and 8 for big libraries. LOOK synthesizes the full cached chunk; the next blind decision executes its second five-control block. This preserves A’s ten-control commit.

| Arm | Library | Library rows / demos | kref | Fit |
|---|---|---:|---:|---|
| `r8abl_clip_p_l10_50` | `current` | 2640 / 50 | 5 | ready |
| `r8abl_clip_p_l10_500` | `bpool_cs` | 29472 / 500 | 8 | ready |
| `r8abl_clip_p_sp_50` | `current` | 1018 / 49 | 5 | ready |
| `r8abl_clip_p_sp_500` | `bpool_cs` | 10909 / 500 | 8 | ready |
| `r8abl_clip_g_l10_50` | `current` | 2645 / 50 | 5 | blocked |
| `r8abl_clip_g_l10_500` | `bpool_all` | 29631 / 500 | 8 | blocked |
| `r8abl_clip_g_sp_50` | `current` | 1063 / 50 | 5 | blocked |
| `r8abl_clip_g_sp_500` | `bpool_all` | 11751 / 500 | 8 | blocked |

The nominal Spatial-50 pi05 source has 49 demos and 1,018 rows; it is retained exactly. Scale labels are the source A labels.

## Encoder and preprocessing proof

The model is OpenAI CLIP ViT-B/32 (768-wide, 12 blocks, 12 attention heads, 32-pixel patches, 512-D image output). The cached timm `vit_base_patch32_clip_224.openai` file is an open_clip-format safetensors checkpoint. Its SHA256 is `e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31`; every Encoder load verifies this exact hash and strictly loads every visual tensor.

OpenAI weights require QuickGELU (`x*sigmoid(1.702*x)`). The reference explicitly uses `force_quick_gelu=True`; open_clip’s untrained ViT-B-32 ordinary GELU default is inappropriate for these weights. `encoder.py` implements the image tower in plain torch, avoiding open_clip in GR00T. All three runtimes use this same file. Float32 preprocessing and weights are explicit; ambient autocast is disabled during encoding, TF32 is disabled for its CUDA operations, and caller backend settings are restored afterward.

CLIP preprocessing is PIL bicubic shortest-edge resize to 224, round-offset center crop to 224, RGB conversion, float32 /255 and OpenAI mean/std normalization, then L2-normalized image embedding. Camera order is agentview/base_0_rgb first, wrist/left_wrist_0_rgb second. Each camera has its own centered PCA-64 fitted using A’s `rsvd_k128_o32_p3_seed0_keep64` recipe; the original eight state values are appended.

Pi05 library frames are the full-row-aligned `tok/img0.npy` and `tok/img1.npy` uint8 224×224 RGB arrays. Twenty-four sampled camera-frame checks match source `input_images/base_0_rgb` and `input_images/left_wrist_0_rgb` bit-for-bit. Forty unique recorded server camera frames (init 0 across ten tasks per suite) match raw wire images to A’s post-transform inputs with max abs difference 0. Thus the pi05 wire-to-library pipeline is already 224×224; the CLIP resize/crop is identity before normalization. Evidence: `results/source_audit.json`.

| Comparison (same six images) | Preprocess max abs diff | Embedding max abs diff |
|---|---:|---:|
| Local plain torch vs open_clip | 0 | 0 |
| h100 pi05 venv vs local open_clip | 0 | 2.682209e-07 |
| h100 GR00T venv vs local open_clip | 0 | 2.682209e-07 |

The six-image proof uses four actual library frames plus 256×256 and rectangular synthetic images. Local comparison is bit-equal; both h100 comparisons are below 1e-5. Remote comparisons used CPU with torch 2.7.1+cu126 (pi05) and 2.5.1+cu124 (GR00T). h100 CUDA encoder parity, actual policy-server loading and network serving are unverified.

The h100 weights are at `/home/exouser/r08abl_clip_sc/open_clip_model.safetensors`, with an identical-hash cache link at `~/.cache/huggingface/hub/models--timm--vit_base_patch32_clip_224.openai/snapshots/r08abl_clip_sc/open_clip_model.safetensors`. The runner’s HOME=/home/exouser resolves this automatically; no arm env/kwargs changes are needed. Disk observation after copying: /data 92 GiB free, root 26 GiB free. The existing `/home/exouser/openpi` checkout and both venvs were not modified.

## Fit and online behavior

`ClipAWM` binds the unchanged `AWM.fit` code to private globals substituting only the PCA source. This follows the R6 TokenPCAAWM precedent and does not mutate the AWM module. Stock `fit_metric`, per-task z-scores, action-supervised cross-demo neighbours, full and early Mahalanobis metrics, confidence calibration, top-16 weights, synthesis and blind-tail behavior remain inherited. The cached state/action/blind arrays and nonvisual settings are bit-equal to each source A artifact; all 80 full/early task metric refits are bit-equal.

The online path is `PluginSession.image -> OnlineQueryView.img0/img1 -> shared Encoder -> fitted PCA GEMV -> inherited AWM metric/query`. Across 29 sampled library rows per ready cell (116 rows, 232 embeddings), max embedding difference is 2.2873282e-06 and max projected key difference is 3.4347177e-06. All 116 full query payload comparisons to stock BlindAWM on identical CLIP features and all 116 blind-tail comparisons pass bit-for-bit. This exercises real plugin accessors in process; it is not a running websocket/policy-server smoke.

The encoder is a process singleton; fitted pickles contain only numerical fitted state, never the GPU model or lock. Connection clones share immutable fit arrays and get independent episode history. LOOK extras insert `os_clip_encode_ms` first, so the decision-log scalar cap retains it. The field measures both cameras’ preprocessing, transfer, inference, GPU completion and D2H. Initial model loading is excluded. Blind decisions do not encode or add this field.

## Discovery retrieval sanity

Candidate leave-one-demo-out excludes every row of the query demonstration, uses <=5 evenly spaced rows per demo, and queries only recorded init indices 0–29. There is no success filter. Deployed PCA and supervised metrics remain frozen from the complete A library; this is descriptive retrieval sanity, not refitted cross-validation. Neighbour agreement is CLIP vs A top-1 and top-16 overlap; action error is sigma-normalized RMS against the library chunk, on the full executed ten controls.

| Cell | Demos / queries | Top-1 agreement | Top-16 overlap | CLIP RMS10 | A RMS10 |
|---|---:|---:|---:|---:|---:|
| `r8abl_clip_p_l10_500` | 300 / 1500 | 0.2333 | 0.4542 | 0.341687 | 0.284986 |
| `r8abl_clip_p_sp_50` | 26 / 130 | 0.7231 | 0.9139 | 0.412206 | 0.408426 |
| `r8abl_clip_p_sp_500` | 300 / 1500 | 0.2733 | 0.5235 | 0.349811 | 0.305769 |

LIBERO-10 current has unknown init indices for all 50 demos, so its discovery-only LODO is unverified. Separate logged inference-query sanity uses 90 query rows, inits [0, 10, 20], and unchanged candidates: top-1 agreement 0.2778, top-16 overlap 0.7076, CLIP/A RMS10 0.427718/0.396430. It is explicitly not LODO. Inits 30–49 are not queried for either retrieval analysis. Full row identities and five-control error results are in `results/validation.json`.

The CLIP retrieval key gives higher offline action error in both 500-demo cells. These results do not establish a closed-loop success rate; the coordinator must run the authorized real evaluation separately.

## Local GPU LOOK latency

RTX 4090 shared with the coordinator; 5 warmups and 50 measured in-process LOOKs per ready cell. Median/p95 timings include two cameras; full method adds PCA, metric, synthesis and anchor recording, but excludes policy stage 1, network, simulator and initial model loading. Long tails are observations on a shared GPU, not isolated H100 latency estimates.

| Cell | CLIP encode median / p95 ms | Full method LOOK median / p95 ms |
|---|---:|---:|
| `r8abl_clip_p_l10_50` | 9.462 / 10.216 | 10.252 / 11.095 |
| `r8abl_clip_p_l10_500` | 9.302 / 190.550 | 10.113 / 191.746 |
| `r8abl_clip_p_sp_50` | 11.138 / 174.784 | 11.925 / 175.704 |
| `r8abl_clip_p_sp_500` | 9.234 / 10.011 | 9.956 / 10.708 |

Owner IR remains pi05 `.152*look + .848*call`, GR00T `.148*look + .852*call`; measured CLIP time is separate. A’s original cost_ledger presence/absence is preserved.

Before each local GPU job the guard required >=8,000 MiB free and set an allocator cap of 3 GiB. Encoding used batches of four frames (eight camera images) and one job at a time under `.gpu_job.lock`. Encoding peak reserved GPU memory was 425,721,856 bytes; latest latency peak was 404,750,336 bytes. Actual checks and progress are in `results/encode.log`, derived `encoded.json` markers and `results/latency.log`. GPU use was limited to library encoding and LOOK latency; no policy/simulator jobs were launched.

## Emission, pairing and remaining limits

The common emitter already merges old arms by name. Its read/merge/config/write is now protected by `.emit_arms.lock`, and arms.json is replaced atomically; CLIP `emit.py` additionally verifies every pre-existing other row and config hash. Eight CLIP rows were added while retaining 24 other rows and 48 config files byte-for-byte, producing 32 total rows. The common emitter change is covered by a concurrent-emission unit test. `arms_in.json` owned by the guard task is untouched.

All eight rows parse/configure in standard mode, without debug/oracle/input capture. The official manifest contains exactly 500 pairs (tasks 0–9 × inits 0–49). All 24 A replicate references have exactly that accepted terminal pair set; comparisons are separately paired to A replicate 1/2/3 in each cell. See `results/arm_validation.json` and run `clip/pairing.json`.

Run `clip/ready_arms.json` lists only the four runnable pi05 arms; `clip/blocked_arms.json` lists the GR00T definitions. GR00T image arrays are absent and the sampled source step groups contain only clean_action, prompt_emb, robot_state, vision_0 and vision_1. No RGB reconstruction or substitute rollout was used. Recover the exact original per-frame camera inputs before building GR00T keys; its 256-wire/0.95-crop/224-resize model pipeline also requires validation against those recovered inputs. There is no GR00T fit or retrieval result.

Refresh the h100 runner’s isolated source snapshot before selecting CLIP, so it includes the new module files. Use its explicit four-arm selection and the full official manifest. The snapshot runner can reuse the verified HOME cache weights; derived embeddings/PCA arrays are not required for serving a prefit artifact. No real closed-loop arm, remote server, LIBERO worker or running foreign process was started or stopped by this task. The temporary rsync daemon owned by this task on 23197 was stopped; receivers on 23198/23199 were left alone.
