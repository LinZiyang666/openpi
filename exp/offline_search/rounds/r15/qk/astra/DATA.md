# R15 light image key: data inventory

Initial inventory: 2026-10-05. Reproduce with `inventory.py` and `prepare.py` in this directory, using the CPU/environment prefix in `HANDBACK.md`. Machine evidence is `inventory.json`; prepared arrays and shard SHA256 lists are under `/home/weiland/trace_runs/offline_search_store/derived/r15_qk/`.

## Build libraries and the hard boundary

| Library | Episodes | Rows | Stored RGB | Stored action |
|---|---:|---:|---|---|
| pi05_l10/bpool_cs | 500 | 29,472 | Both cameras, uint8 224×224 | 10×32 |
| pi05_spatial/bpool_cs | 500 | 10,909 | Both cameras, uint8 224×224 | 10×32 |
| groot_l10/bpool_all | 500 | 29,631 | Absent | 16×32 |
| groot_spatial/bpool_all | 500 | 11,751 | Absent | 16×32 |

All four `tok/rows.npy` arrays are unique, cover every library row, and equal `arange(L)` (checked). Both camera tokens are 256×2048 fp16; pooled keys are 32,768 fp32. Training uses only the requested cell's whole-episode subset in `rounds/r10/subsets/`; the subset is checked against the R10 fitted artifact and actual selected row IDs. Library-wide metadata is used for exclusion checks, not as extra training observations. Action sigma and teacher PCA/AWM parameters come from that cell's library-only R10 fit. No external pretrained image weights.

Prepared L10 training sets: size 50 = 2,925 rows; size 200 = 11,437 rows. Both use nearest 224→56 camera sampling (`image[::4,::4]`), matching `LookRecorder` for a 224-pixel wire image. `build_library_store.py` identifies these RGB arrays as the exact HDF5 `input_images/{base_0_rgb,left_wrist_0_rgb}` inputs. Camera order is retained.

## Existing B-dev recordings

Counts below are manifest inventory counts, including all indexed attempts. Evaluation preparation additionally filters to the journal's final accepted, error-free attempt and hashes each selected shard. It does not condition on success.

| Root | Available cells: look count | Images | Policy labels |
|---|---|---|---|
| r12_devlooks_50L | pi05 L10 3,198; GR00T L10 3,384 | Both, 56² | Shadow at every look |
| r12_devlooks_50S | pi05 Spatial 1,244; GR00T Spatial 1,220 | Both, 56² | Actual policy only on MISS |
| r12_devlooks_50S_local | pi05 Spatial 1,221; GR00T Spatial 1,270 | Both, 56² | Actual policy only on MISS |
| r12_devlooks_200 | pi05 L10 2,787; Spatial 1,102 | Both, 56² | Actual policy only on MISS |
| r12_devlooks_200_local | pi05 L10 2,902; Spatial 1,093 | Both, 56² | Actual policy only on MISS |
| r12_devlooks_400 | pi05 L10 2,615; GR00T L10 2,880 | Both, 56² | Actual policy only on MISS |
| r12_devlooks_knob | pi05 L10 1,349; GR00T L10 1,450 | Both, 56² | Actual policy only on MISS |
| r13_devrec_50 | pi05 L10 8,780 across collected arms | None | Off arm has shadows |
| r13_devrec_pi05_spatial_50 | pi05 Spatial 1,210 | None | Shadows |

Do not pair R13 labels with R12 images: their trajectories differ. HIT `action` without `shadow_policy` is a cache output, not a policy label. MISS-only error has policy-selection bias and is secondary evidence. No B-dev image or label enters an optimizer or normalization fit.

The L10 50L evaluation has 100 accepted episodes, all with images and shadow labels. Exact unseen filtering against each cell's original B init IDs gives:

| Served size | Unseen episodes | Labeled looks |
|---:|---:|---:|
| 50 | 100 | 3,198 |
| 100 | 91 | 2,938 |
| 200 | 73 | 2,289 |
| 500 | 0 | 0 |

At 200 the per-task episode counts are 7,8,8,7,8,6,8,7,7,7. These are size-50 controller trajectories evaluated against a different candidate library, not size-200 closed-loop outcomes. Other recording roots require the same explicit exclusion check; the root name alone does not certify unseen status. Size 500 has no unseen B init; any future leave-own-episode-out analysis must be labeled in-library and must not be used as unseen validation.

## GR00T RGB and replay status

The original library HDF5s are locally accessible at `/data/libero_cache/build_libero10/libero_10/` and `/data/libero_cache/build_spatial/libero_spatial/`. The inspected first B episode of each suite contains only `clean_action`, `prompt_emb`, `robot_state`, `vision_0`, `vision_1`; no image or full simulator state. This agrees with the source schema in `r0/build_library_store.py` and the earlier R8 RGB blocker.

The requested `dual_20260923/runs/tr_groot_*` directories each have 500 HDF5 and 500 trace JSONL filenames. Their September trace filenames differ from the August build-library episode filenames and timestamps. Filenames were inventoried only; unknown-pool trace payloads were not opened. They cannot supply training frames merely by sharing a task/init number: the hard rule requires the actual library trajectory. In particular, do not use evaluation trace frames to fill this gap.

Deterministic replay is a possible recovery procedure, not currently verified data: recover the original B init mapping, simulator/controller versions, reset/warmup procedure, action denormalization and gripper convention; replay the first five controls of each stored chunk; compare every pre-action 8-D robot state with the stored row and verify episode length and camera geometry. Robot-state agreement is necessary but insufficient (it omits object state); preserve/render full simulator state and, with coordinator resources, compare re-encoded recovered frames to both stored camera keys/tokens. Abort an episode at the first mismatch. Spatial metadata has missing `orig_init_state_idx` values, so resolve the authoritative build mapping before replay. No simulator, policy server, replay trajectory, remote operation, or unverified GR00T training was launched here.

## Completed exclusion and source checks

`recording_exclusions.py` now checks every inventoried arm using deployed `devset.subset_record`, including its SHA-bound GR00T identity certificates. `recording_exclusions.csv` gives every arm. Selected episode counts unseen at sizes **50 / 100 / 200 / 500** are:

- All R12 size-200 and size-400 recording selections: **100 / 100 / 100 / 0**.
- All R12 size-50 and R13 recording selections: **100 / 91 / 73 / 0**.
- R12 knob recording selections: **50 / 45 / 36 / 0**.

These are selection-identity counts; `prepare.py` reports actual accepted look counts separately. GR00T Spatial's missing init attributes are resolved by the existing `rounds/r11/devset/identities/groot_spatial_bpool_all.json` certificate through `devset.library_episodes`; no index-based guess is needed. Replay still needs the original reset, physics, controller and object-state fidelity.

Further source inspection settled the September trace question: `/home/weiland/trace_runs/dual_20260923/ops/run_group.sh:34` explicitly passes `--apool-record` and `--apool-dir`. Those are **A-pool evaluation traces**, not build-library RGB. Only launcher source and filenames were read; no trace payload was opened. They are ineligible for this work.

`verification.json` checks 20 selected π0.5 source episodes (one per task in each suite), both cameras: all 40 downsampled source images, states and actions match prepared rows exactly. All three prepared cells have exact library-only action sigma and verified subset membership. Spatial-50 has 1,161 training rows and 1,221 accepted B-dev looks, but only 189 policy-labeled MISS looks; the other 1,032 HIT actions are not policy labels.

Per `OWNER_ADDENDUM.md`, `loeo.py` evaluates every library row against other episodes of the same task before further offline selection. Its encoder and metric remain fitted to the full cell library, as in deployed confidence calibration. This is donor-episode exclusion, not encoder-held-out generalization. B-dev remains essential.

References: `r15/BRIEF.md`; `qk/OWNER_ADDENDUM.md`; `r12/prep/HANDBACK.md`; `r13/DATA.md`; `closed_loop/look_recording.py`; `closed_loop/devset.py`; `r0/build_library_store.py`; local `inventory.json`, `recording_exclusions.json`, `verification.json` and derived `*_meta.json`.
