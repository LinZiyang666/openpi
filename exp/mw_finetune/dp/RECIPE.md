# R17 Diffusion Policy: reconstruction and MT50/LIBERO training

Agent E builds tools and bounded smokes; the coordinator runs long training and
formal evaluation. No commit/push or long training was launched. All E artifacts
are under this directory or `/data/mw_finetune/dp`. Shared registration request:
[`../REQUESTS/E_to_A_dp.md`](../REQUESTS/E_to_A_dp.md).

## Code and weights provenance

The author's DP wrapper was **not found**. Audited public trees include
branyang02/openpi-eval (main and both release/world-model branches), openpi-cuda,
openpi-trajectory, sim-evals, ricl_openpi_latent_action, and ricl-vla/ricl_openpi.
All287 commits reachable in the cloned openpi-eval repository were searched for
DP paths and commit descriptions; none contain its checkpoint wrapper. Tree
snapshots and revisions are retained in `reports/source_audit.json` and
`reports/*_tree.json`; the read-only source clone is in E's data root.

| Artifact | Pinned revision | Selected files |
|---|---|---|
| [DP LIBERO](https://huggingface.co/brandonyang/dp-libero-checkpoints/tree/3c48ffaab98faacb119beaf2ee12f943503e6492) | `3c48ffaab98faacb119beaf2ee12f943503e6492` | `dp_libero_lang_v1/30000`:367,444,556-byte model, inert metadata, norm stats |
| [DP MetaWorld](https://huggingface.co/brandonyang/dp-metaworld-checkpoints/tree/46c26c0630024f7645f19398cc5be105870ca96f) | `46c26c0630024f7645f19398cc5be105870ca96f` | `dp_metaworld_lang_v1/85000`:367,423,896-byte model, metadata, ML45 stats |
| [Stanford DP Transformer](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/model/diffusion/transformer_for_diffusion.py) | `5ba07ac6661db573af695b419a7947ecb704690f` | Vendored Transformer, sinusoidal embedding and module mixin; MIT license retained |
| [robomimic FiLM/SpatialSoftmax](https://github.com/ARISE-Initiative/robomimic/blob/d309eaecc18acf4152a830a895a6984b8ac71b05/robomimic/models/base_nets.py) | `d309eaecc18acf4152a830a895a6984b8ac71b05` | Reconstructed `FiLMLayer`, `ResNet18ConvFiLM`, `SpatialSoftmax`; MIT license retained |
| [CLIP-L/14](https://huggingface.co/openai/clip-vit-large-patch14/tree/32bd64288804d66eefd0ccbe215aa642df71cc41) | `32bd64288804d66eefd0ccbe215aa642df71cc41` | Projected768-D text embeddings; compact494,624,560-byte text-only export |
| [ML45 source data](https://huggingface.co/datasets/brandonyang/metaworld_ml45/tree/7c3ffbc972fc6b1b0f701882841c2d17009b71b8) | `7c3ffbc972fc6b1b0f701882841c2d17009b71b8` | Only small metadata/task files downloaded; published dataset actually lists44 task texts |
| [MT50 source data](https://huggingface.co/datasets/lerobot/metaworld_mt50/tree/b46a1c185a2368660650a82ac5a5083a56722f48) | `b46a1c185a2368660650a82ac5a5083a56722f48` | A's existing full local dataset; C's audited vector index reused read-only |

Public `metadata.pt` is read with an allowlisted inert unpickler; no public
optimizer pickle is executed. Source weight SHA256s: LIBERO
`3ab4fb34d819bf1bb3e20b78dc2e41560ef47fe663213b17120d23ef2cf78103`,
MetaWorld `98353004131950a0a087a65270414e67067719e4ef7d701b9da77711ba136c16`.

## Reconstruction evidence and its limits

Both models strictly load **all 475 tensors**, with zero missing/unexpected keys.
The complete module tree has91,845,153 scalar parameters for LIBERO and
91,839,988 for MetaWorld, including fixed normalization parameters. Architecture:
two independent ResNet18s with eight FiLM blocks each, GroupNorm(c/16 groups),
32-keypoint spatial softmax and64-D projected visual features per camera;
12 decoder layers,4 conditioning encoder layers,8 heads,512 hidden width.
The condition tokens are timestep plus **one current observation**, never history.
LIBERO uses8-D state/7-D actions; MetaWorld4-D state/actions. Horizon16,
n_obs_steps1, n_action_steps16,96² images,84² crop (centered at eval).

The768-D language convention is directly identified: projected CLIP-L/14
embeddings for **all 44 ML45 task texts** reproduce the stored per-dimension min
and max within **1.90735e-6**. Unprojected pooled embeddings have~1-unit mean
absolute disagreement and are rejected. `verify.py` reproduces this check.

Feature concatenation is `[base64,wrist64,state,language768]`. The unusually
large condition-projection column norms at128..131 support state at that offset
in MetaWorld; the shape alone does not prove concatenation order. Visual math
follows robomimic base_nets.py:618–740 and1113 onward, plus obs_nets.py's default
ReLU after each visual core. The checkpoint's core naming/pooling layout matches
these components. Camera repacking, resize implementation and nonparameter
activations are reconstructed conventions because the original DP wrapper is
unpublished. **Strict loading proves tensor compatibility, not unpublished
source-code identity or a reported benchmark success rate.**

There are two normalization layers. External openpi **quantile** normalization
is followed by the checkpoint's DP affine limit normalization. The embedded
base_config field says `use_quantile_norm=false`, but the author's published
[data factory](https://github.com/branyang02/openpi-eval/blob/ff78053a5557517a64e1b7566e8d47214a21adb8/src/openpi/training/config.py#L177)
sets it from model type; internal state/action statistics support the quantile
interpretation (not z-score). Decode reverses both layers. Images enter as
CHW float[0,1]; the internal96-element RGB affine broadcasts over width, exactly
as Stanford's last_n_dims1 normalizer. Physical actions are never clipped,
scaled, binarized or inverted after decoding. DDPM's configured internal
normalized x0 clip is retained.

Default inference uses **100-step DDPM**, squaredcos_cap_v2, epsilon prediction,
fixed_small variance and clip_sample_range1. A's optional denoising_steps is an
explicit override, never a hidden speed shortcut. `reset()` is a no-op; no
action queue, previous observations, prediction reuse or mutable language cache
is used by serving. MetaWorld uses a fresh private generator from obs.seed;
LIBERO's plain wire uses a fresh fixed-seed0 generator when no seed is supplied,
so even the RNG has no cross-call dependency.
CPU checks show bitwise repeated outputs with the same explicit seed, changed
outputs when task language changes, finite16×7/16×4 outputs, and strict loading.

## Why train MT50

The author's published
[MetaWorld data repacker](https://github.com/branyang02/openpi-eval/blob/ff78053a5557517a64e1b7566e8d47214a21adb8/src/openpi/training/config.py#L305)
maps **corner4.image → base** and **gripperPOV.image → wrist**. The original
ML45 checkpoint's nonconstant wrist pixel statistics confirm a real second
camera. MT50 has one 480² corner2 camera; A's parity report verifies its view,
position(0.75,0.075,0.7),180° orientation and state obs[:4] against the source.
Our wrapper receives A's already-oriented image and does not rotate it again.
The direct ML45 diagnostic fills the unavailable wrist with zeros and preserves
source normalization; this is a camera/domain transfer test, not a faithful
ML45 reproduction. `second_camera=duplicate` is an explicit alternative only.

The owner decision in `../REQUESTS/COORD_to_E.md` supersedes the earlier
fine-tuning proposal: **MT50 fallback trains DP from scratch**. The launcher
reads the ML45 metadata architecture but never opens its DP weights in scratch
mode. Both ResNet18s, FiLM, observation encoder and action Transformer start
from random initialization; generic CLIP text weights remain frozen. The
original visual initialization is unspecified (robomimic FiLM defaults to
`pretrained=False`), so no ImageNet initialization is added.

85k/zero-view achieved1/16;30k/duplicate-view achieved0/16, both through A's
8-easy-task×2 EGL loop with full 100 DDPM steps. 30k/zero-view confirmation also completed0/16 without errors. These are transfer sanity measurements,
not formal MT50 SR. Source-frame teacher-forced epsilon MSE on8 spread ML45
frames is0.00937 at85k and0.01599 at30k with the correct source camera pair;
at85k it increases to0.139 with zero wrist and0.325 with duplicated wrist.
This supports the reconstructed conventions and the camera-transfer diagnosis.
One episode cannot establish source-domain performance.

Nominal ML45's five held-out tasks are bin-picking, box-close, hand-insert,
door-lock and door-unlock ([official environment table](https://github.com/Farama-Foundation/Metaworld/blob/main/metaworld/env_dict.py)).
The published44 task descriptions additionally omit dial-turn. Source-annotation
coverage is inferred semantically and does not prove checkpoint training coverage.
`aggregate_coverage.py` reports the five nominal held-out tasks, additional dial
and44 annotated tasks separately for any later direct-checkpoint evaluation.

## Training recipe

Values come first from the actual checkpoint `metadata.pt`, rather than
substituting VLA optimizer settings:

| Setting | Implemented |
|---|---|
| Updates/global batch |100,000 /256; measured microbatch256, accumulation1 |
| AdamW |betas(.95,.999), eps1e-8, weight_decay1e-6, global gradient clip10 |
| LR |linear500 warmup from1e-4/501 to1e-4; cosine to 1e-5 at update99,500, flat thereafter |
| Precision |BF16 autocast activations; FP32 master policy parameters/Adam moments |
| EMA |none, matching `ema_decay=null` |
| Diffusion training |100 noise timesteps; uniform timestep; epsilon MSE over16 actions |
| Crop |one random84² crop per current96² image during training; fixed center at serving |
| Seed |42; deterministic sampler keyed by(seed,optimizer update,microstep) |
| Saves/logs |immutable checkpoint each5k plus final; JSONL every optimizer update |

LR endpoint semantics follow the author's published
[train_pytorch.py:472–480](https://github.com/branyang02/openpi-eval/blob/ff78053a5557517a64e1b7566e8d47214a21adb8/scripts/train_pytorch.py#L472)
(decay_steps includes warmup). Original DP trainer details beyond the embedded
config remain unknown. The [SmolVLA paper §4.3](https://arxiv.org/html/2506.01844v1#S4.SS3)
uses100k updates, batch64, AdamW(.9,.95),100 warmup and cosine1e-4→2.5e-6 for
simulation. Its Table2 reports DP's MT50 equal-difficulty-group mean10.5%; it is
context, not evidence for these community weights. We keep the explicit DP
metadata recipe. A's formal500 episodes and difficulty groups follow the
shared contract; five-action replanning differs from SmolVLA's one-action
simulation replanning and must be disclosed.

Every implementation deviation/choice:

1. New complete MT50 data, random DP initialization and a fresh100k schedule,
   following the owner fallback decision. Frozen pretrained CLIP supplies text;
   no ML45/LIBERO policy tensors or public optimizer snapshot initialize MT50.
2. Single MT50 corner2 view plus **zero second camera**, replacing the source's
   corner4+gripperPOV. The second branch starts randomly for MT50 and
   remains language-conditioned/trainable; no historical frame is supplied.
3. External quantiles and internal limits are refit on all MT50 vectors/pixels
   and50 task embeddings, rather than reusing ML45 ranges. Equal-task language
   min/max is sufficient for the affine; exact original fitting sample counts
   are unknown. No action clipping is added.
4. PIL bilinear480→96 directly, with no extra flip/rotation. The original DP
   resizing code is unavailable; native checkpoint crop geometry is preserved.
5. Episode tail actions repeat the final action, matching standard LeRobot/DP
   action-window padding. Original unpublished padding cannot be verified.
6. FP32 master weights/moments with BF16 autocast; metadata reveals BF16 training
   precision but not its master-weight implementation. No distributed training
   or activation checkpointing is necessary at this model size.
7. Static frozen training-text embeddings are cached once for transfer/resume
   parity across library versions. Serving encodes the current request's text.
8. JSONL/probe JSON replaces W&B; all5k snapshots are retained (no deletion),
   whereas metadata has keep_period10k. Dataset shuffle is reproducible keyed
   random sampling with replacement; the original loader sequence is unknown.
9. Formal eval uses the requested160-step/5-action historical protocol. The
   LIBERO mechanical smoke uses local OSMesa; full SR should use established
   EGL workers and all task/init-state identities.
10. The seedless LIBERO wire uses a fresh seed0 generator per call to honor the
    owner's no-cross-call-state rule. The original wrapper RNG convention is
    unavailable. MetaWorld instead uses the contract's explicit decision seed.

The complete96² prepared set has492 PNG SQLite shards,204,806 frames,2,500
episodes,50 tasks and80 Hz. C's raw vector index SHA256 is
`7bd1c896d574117c4d8d7db3ba5c97c6b131b1824e97fb6d4a10b96130bf70f1`.
All shards pass quick_check/count audit. Twelve spread first-to-last samples
exactly match original480² source after the stated resize; raw state/action
values match bitwise. Windows cannot cross episode boundaries. The source
push-back wording differs from A's historical prompt; preserve source text for
training and disclose that one-task eval prompt shift, as B/C do.

## Resource and resume evidence

H100 full-data BF16 **scratch200-update batch256×1** probe, sharing with
Fast-WAM and bounded E serving: **6.90391 updates/s**,0.144846s/update,
peak4.80824 GiB allocated/5.00391 GiB reserved, **4.02349h compute projection**. First5 updates are
excluded; timing includes data wait and synchronized GPU completion, excludes
checkpoint saves. This is a projection, not a completed long run. 64×4 measured
2.48930 updates/s,2.54010/2.57031 GiB and11.1589h, so256×1 is selected.
The scratch report is `reports/h100_scratch_mt50_b256_200.json`; older
ML45-initialized probes are retained under `reports/h100_probe*.json` and logs in E's data root.
The reboot fixed the local driver, but GR00T now occupies~39.6/48 GB;
**no local GPU probe until it finishes (~18:40 CDT)**. Re-probe then. A shared-GPU H100 measurement does not predict4090 throughput.

The probe allocator cap is8 GiB (8.59 GB), leaving context headroom under 10 GB;
no training GPU lock was taken by smokes/probes, per the owner's explicit
exception. Long launch.sh runs take the lock and are coordinator-only. H100
E artifacts total~5.87 GB after the reboot cleanup. Three E-owned transfer
archives were removed only after SHA256 parity with retained local copies,
under the owner's explicit cleanup instruction; the single optimizer probe
checkpoint and extracted data/weights remain. **Long training commands target local disk**;
the H10010 GB E budget cannot hold20 full optimizer snapshots. A serving-only
export is~367MB and can be copied to H100 within the remaining budget using
one small archive plus one extracted copy. Do not resync the2.7 GB dataset again
under a new name without checking the budget.

Snapshots contain model, optimizer, CPU/CUDA RNG, implicit sampler update,
recipe/data/init/language/architecture hashes and complete marker. Writes are
fsynced and installed by rename; latest is an atomic symlink. Resume rejects
incompatible recipes. A dedicated DataLoader generator prevents iterator
creation from advancing the saved model RNG. A two-update CPU run and a
one-update/save/reload/one-update run match **all 475 tensors bitwise**; both
optimizer steps, loss and gradient norm also agree. CPU tests/data audits and
checkpoint smoke exports are mechanics evidence, not trained success-rate
measurements. Older interrupted/diagnostic artifacts remain preserved.


## LIBERO validation and fallback

The native wrapper accepts two current camera images and raw8-D EEF/axis-angle/
gripper state, and returns16 physical7-D actions using the existing LIBERO
websocket/msgpack interface. Same image rotation/state/action conventions as
`examples/libero/main.py`; never apply GR00T's gripper inversion. Source/live
256² views were visually checked with the same native camera geometry; current
wire includes its180° rotation already. The completed fixed-seed0 smoke is
**0/4**, one task/init state from each suite. Each episode ran its full suite
limit without interface errors; OSMesa rendering is explicitly diagnostic.
Older0/2 spatial smoke predates the fixed seed and remains separate.

Native source LIBERO episode0 yields epsilon MSE0.01393 and quantile action
MSE0.02845 on8 spread frames. Missing/duplicated/swapped camera variants give
0.361/0.296/0.273 epsilon MSE; z-score normalization gives0.694. All100-step
sampled actions are finite. This is positive mechanics/convention evidence.
**A normalizer coverage surprise:** only source instructions 0..8 (nine LIBERO-10
instructions) reproduce the stored language extrema, max error2.861e-6;
all 40 instructions differ by up to2.107. The claimed four-suite training mixture
cannot be verified from metadata alone. This neither proves nine-task training
nor supports claiming usable four-suite SR. Reports preserve both observations.

Owner-requested fallback tooling is under `libero/`. It fine-tunes the strict
LIBERO30k DP tensors, refits both normalization layers on all 40 tasks and keeps
the original100k/batch256/AdamW/LR/no-EMA recipe. `--init scratch` is supported
if the coordinator chooses random DP initialization instead. This LIBERO
initialization choice is distinct from the required MT50 scratch fallback.

No local requested dataset was found in allowed `/data` and HF cache locations;
search evidence is retained. Source is pinned
[physical-intelligence/libero](https://huggingface.co/datasets/physical-intelligence/libero/tree/a4336d589d589045d1c56423ffdf3b88a0e19b1f):
1,693 episodes,273,465 frames,40 tasks,10 Hz,256² paired images,8-D state/7-D
actions (~34.9 GB parquet). The resumable preparer writes paired96² PNG SQLite
shards, preserves raw vectors and source orientation, repeats episode-tail
actions and never reads observation history. Full runs reject incomplete data.
`--probe-episodes` explicitly marks partial data; training accepts it only with
both `--probe N` and `--allow-partial-probe`. Full local preparation is complete;1,693 paired shards pass SQLite audits,
and12 spread samples match both resized views and raw8/7 vectors exactly.
Commands are in HANDBACK_E. No LIBERO data is copied in full to H100.

A200-update H100 LIBERO fine-tuning probe on **one source episode** measured
7.19309 updates/s,4.80617 GiB allocated/5.00586 GiB reserved, projecting3.86173h
compute. This checks backward/optimizer mechanics and memory; one-episode
cache timing does not predict full mixed-suite data throughput. Re-probe full
local data after GR00T finishes. Full LIBERO SR requires all 4 suites×10 tasks×
50 official initial states, pinned simulator/assets, native suite limits and
per-suite reporting on established EGL workers. No full SR claim is made.

Additional LIBERO deviations: new four-suite source normalization; resumed
fresh optimizer/schedule instead of unpublished optimizer state; source-native
PIL256→96 resize; no verified reproduction of original sampling/normalizer-fit
subset; local OSMesa for diagnostics; and the owner-required seedless-call
stateless RNG. The same padding/precision/log/snapshot/sampler deviations above
also apply. CPU scratch save/reload matches all 475 tensors, full Adam state and
CPU RNG bitwise (reports/cpu_scratch_resume_parity.json); GPU bitwise resume
has not been claimed.
