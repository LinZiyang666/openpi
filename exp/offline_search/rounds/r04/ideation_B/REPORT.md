# R4 ideation B — cheaper vision, cheaper MISS, and the low-IR frontier

**Recommendation:** first test two-step π0.5 MISSes and finish the sparse-MISS schedule frontier. Then remove provably irrelevant padded work, subject to numerical parity. Treat a one-camera or intermediate-layer key as a separate, higher-risk experiment. Keep AWM selection and kernel synthesis fixed while testing these changes.

**Measurement limit:** the requested RTX 4090 measurements could not be made in this execution environment. `nvidia-smi` exits **9**, unable to communicate with the driver; `torch.cuda.is_available()` is false. The admission check therefore started **no GPU work**. `/dev/shm/offline_search_store` is absent, so diagnostics used the supplied read-only cold copy. I completed same-library AWM refits for camera deletion and coarser pooling at **both library scales, both models, both suites, both query regimes**. Intermediate-layer and resized keys have a CPU feasibility probe, but **their full-library retrieval quality and CUDA-graph costs remain unmeasured**. No CPU ratio is substituted for a 4090 ratio below. Evidence: `inventory.py` → `inventory.json`; `vision_probe.py` → `vision_gpu.json`, `vision_cpu.json`.

All new files are in `exp/offline_search/rounds/r04/ideation_B/`. No original code/data was changed, no external model was used, no server or LIBERO worker was started, and no git command was run. Every Python invocation used the assigned `12-23,56-67` affinity and OMP/OPENBLAS/MKL = 1. Diagnostic pools use four workers apiece and stayed below the 24-process total limit. No large arrays were written.

## 1. Facts driving the proposals

### 1.1 Cost accounting: do not credit cheap vision twice

Use the binding R4 shares, **a=.152, b=.410, c=.438**, and **F=67.52 ms** from 10.26 + 27.69 + 29.57. Let `m` be the realized MISS share, `K` the number of denoising evaluations, and `r=Ccheap/10.26` the measured cheap-key cost relative to full stage 1. The rounded shares and millisecond constants differ slightly on division; tables consistently use the owner’s rounded shares for IR.

- Full keys: `IR = .152 + m(.410 + .438 K/10)`.
- A cheap key whose intermediate computation is retained and **continued exactly** on MISS: `IR = (1-m).152r + m(.562 + .438 K/10)`.
- A separate cheap pass followed by **recomputation of full stage 1** on MISS: `IR = .152r + m(.562 + .438 K/10)`.
- Thus reusable cheap vision saves only `(1-m).152(1-r)`. A duplicated cheap pass additionally costs `m .152r`. For K=10, the duplicated branch helps only when `m < 1-r`.

A lower-resolution pass ordinarily belongs to the **recompute** case. Full-resolution early layers and a complete single-camera pass can belong to the **continue** case if the server retains their tensors. Simply changing `keys.vision_0.enabled` or the CP1 reducer saves **no tower computation** with the existing interceptor.

`make_tables.py` produces all arithmetic in `cost_tables.json` and `COST_TABLES.md`. The honest baseline MISS prices are:

| K | Stage 2 + 3, ms | Whole MISS, ms | Whole-MISS IR |
|---:|---:|---:|---:|
| 10 | 57.260 | 67.520 | 1.0000 |
| 5 | 42.475 | 52.735 | .7810 |
| 4 | 39.518 | 49.778 | .7372 |
| 2 | 33.604 | 43.864 | .6496 |
| 1 | 30.647 | 40.907 | .6058 |

These scale **stage 3 only**. K=2 does not make a MISS cost .2, and reducing K=2 to K=1 saves only `.0438m` of total IR: **.00445** at the observed g500 MISS rate. Stage 2 is the dominant remaining MISS cost at K=2.

### 1.2 Existing step evidence is stronger than the early step-ladder summary

`exp/step_diag/analysis/step_vs_warmstart.md` §6.15 contains a later, formal matched LIBERO experiment. I recomputed the aggregate SRs from `exp/step_diag/data/analysis/warm_variants_{pi05,groot}_libero_{spatial,10}.json`; see `make_tables.py` → `step_evidence.json`. Each listed cell has **500 episodes**, complete/equal-NFE flags true.

| Model / suite | Full inference | Plain K=2 | Plain K=1 | Exact cached continuation |
|---|---:|---:|---:|---:|
| π0.5 spatial | .986 (K10) | .996 | not in this matched experiment | .926 (2 steps) |
| π0.5 l10 | .844 (K10) | .848 | not in this matched experiment | .778 (2 steps) |
| GR00T spatial | .944 (K8) | .936 | .934 | .938 (1 or 2 steps) |
| GR00T l10 | .880 (K8) | .842 | .846 | .712 (1 step), .818 (2 steps) |

The earlier summary’s l10 `.824–.850` row is an **old K7 result**, not evidence for K1; the text at the beginning of the same file makes that explicit. Do not quote that row as a matched one-step l10 ladder. On π0.5, plain K2 is already supported on l10. What still needs measurement is **K2 on cache-perturbed MISS states**, K1 under the current matched protocol, per-task/seed variation, and K2’s actual graph cost. These questions must be tested at 50 and 500 library episodes. The pure-inference K2 baseline itself has no library and must be drawn on both library panels.

Warm reset does not earn a separate priority: §6.15 reports no systematic improvement over plain reduced steps on LIBERO. Exact continuation loses **6.0 / 6.6 pp** on π0.5 spatial/l10. GR00T l10’s K2 loss is **3.8 pp**, so π0.5’s benign step reduction must not be transferred to GR00T as fact. The GR00T .880 reference above is a different run from R4’s .870 reference; preserve that distinction.

### 1.3 The low-IR schedules: new read-only log measurements

`schedule_replay.py` reads `r03_mx/runs/<arm>/client/journal.jsonl`, includes accepted normal `done` **and `failed`** episodes without errors, matches the accepted attempt, and deduplicates server rows by `(uid,step)`. All six inspected arms contain **500 accepted episodes**. The original guard rule replay reproduces the recorded MISS count exactly in all three guard arms.

| Recorded arm | Library | Successes / 500 | Decisions | MISSes | MISS share | IR K10 | IR if identical decisions use K2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| π0.5 l10 `g500` | 500 | 432 | 29,340 | 2,983 | .101670 | .238216 | .202591 |
| π0.5 l10 `g` | 50 | 370 | 32,967 | 6,664 | .202142 | .323416 | .252586 |
| π0.5 spatial `g` | 49 | 444 | 11,981 | 1,617 | .134964 | .266449 | .219158 |
| π0.5 l10 `perk5` | 50 | 396 | 31,044 | 5,962 | .192050 | .314858 | .247564 |

Of g500’s 2,983 MISSes, primary reasons are **1,899 no-progress, 699 stuck, 250 terminal, 135 overtime**. The no-progress bit is present on **2,158** MISSes when overlaps are retained. Primary reason counts alone cannot price removal of a guard. At 50 episodes the corresponding primary counts are **3,479 / 2,204 / 565 / 416**; the no-progress bit occurs on **4,838** MISSes.

In `MixedJudge.query`, `noprog_n=3` actually requires **two nonadvancing transitions**, i.e. three observations; the logged `extras.noprog_n` counts transitions. The following replay preserves all other guard bits and checks this off-by-one correctly:

| Observed trajectory source | New frozen-observation rule | Replay m | Replay IR K10 | Replay IR K2 |
|---|---|---:|---:|---:|
| l10, 500 | no-progress 4 observations | .055624 | .199169 | .179678 |
| l10, 500 | no-progress 5 observations | .042570 | .188099 | .173183 |
| l10, 500 | no-progress 4 + HIT cap 8 | .144138 | .274229 | .223723 |
| l10, 500 | reset only progress memo after MISS | .086367 | .225239 | .194976 |
| l10, 50 | no-progress 4 observations | .145297 | .275212 | .224300 |
| l10, 50 | no-progress 5 observations | .122152 | .255585 | .212783 |
| l10, 50 | no-progress 4 + HIT cap 8 | .219037 | .337744 | .260993 |
| l10, 50 | reset only progress memo after MISS | .165286 | .292163 | .234247 |
| spatial, 49 | no-progress 4 observations | .103748 | .239978 | .203625 |

**These are budget replays, not new closed-loop outcomes or estimates of their SR.** A changed action changes the later scene, retrieval, guards, and episode length. Raising no-progress to four removes about 45% of observed g500 MISSes but only about 28% at 50 episodes. Adding cap 8 can **increase** total IR: on the 50-library l10 trajectories the K10 replay is .33774, outside the requested .15–.30 interval. Resetting the memo does not mean forcing a HIT: stuck, terminal and overtime remain active.

The old closed-loop evidence remains the ranking evidence: periodic k5 beats guard-only on l10 at 50 episodes (**.792 vs .740**, +5.2 pp, p=.020); targeting beats periodic k3 on spatial (**.980 vs .938**, +4.2 pp). At 500 episodes l10 guard-only is .864 at .238 while the quantile arm buys only .008 SR for roughly .20 more IR. Sources: R3 `ANALYSIS.md` §§2.1–2.7 and the counts above.

### 1.4 Same-library cheaper-key AWM refits, with all candidates retained

`retrieval_diag.py` evaluates **80 cells/variants**: full key, only camera 0, only camera 1, 2×2 spatial pool, global mean pool × two models × two suites × two libraries × inf/cache query regimes. It uses the **entire** candidate library for PCA and per-task AWM fitting; query evaluation is the stored tok subset (50 episodes/cell). There is **no borrowed big-library information** in these fits.

Recipe: library PCA64 per retained camera; concatenate valid state8; refit the original action-neighbor whitening (three nearest heads in other library episodes, ridge .1), including the original step≤2 early fit; same top16 kernel with kref5 at 50 and kref8 at 500; same fresh-after-MISS continuity term .5. All action distances/error use `[:5,:7]` and the current-library sigma. Camera deletion reduces the feature dimension from 136 to 72; it is not zeroing one block under the old metric. Coarser pools are rebuilt on library and query keys by spatially pooling the stored 4×4×2048 features, which commutes with the corresponding token averages. No query `a_inf` participates in fitting.

Validation against the actual `AWM.query` covers **1,974 queries** including step0, fresh and stale at every model/suite/library combination. Minimum per-cell top1 agreement is **.992063**, minimum top16 overlap **.998984**; worst per-cell mean action difference **0.00009750 sigma RMS** (maximum individual action-coordinate difference .009905). The diagnostic uses batched float64 distance arithmetic while online AWM keeps float32 codes, so it is a close reproduction, not bit-identical. Evidence: `validate_retrieval.py` → `retrieval_validation.json`.

**π0.5 stale/cache query results**; error is mean RMS sigma vs `a_inf`, overlap = intersection size / 16 against the same-library full-key AWM. Camera0 is the fixed view; camera1 is the wrist.

| Suite | Library | Full error | cam0 error / overlap16 | cam1 error / overlap16 | pool2 error / overlap16 | pool1 error / overlap16 |
|---|---:|---:|---:|---:|---:|---:|
| spatial | 49 | .53315 | .55325 / .7992 | .55668 / .8347 | .53417 / .8857 | .55189 / .8322 |
| spatial | 500 | .45404 | .50249 / .5883 | .47665 / .6531 | .45567 / .8177 | .46106 / .7217 |
| l10 | 50 | .55726 | .60254 / .6532 | .57473 / .6860 | .56641 / .8160 | .56896 / .7202 |
| l10 | 500 | .47042 | .53072 / .4889 | .48590 / .5446 | .46691 / .7633 | .46780 / .6344 |

N=1,393 spatial and 4,176 l10 cache decisions. **Fresh-state agreement hides the problem:** wrist-only top16 overlap is .9654/.9176 on spatial inf at 49/500 episodes, .9551/.9163 on l10 inf at 50/500, versus .8347/.6531 and .6860/.5446 on cache states. Inf-cell mean errors are almost unchanged (l10 full .32575/.28037, wrist .32563/.28010). This is consistent with AWM’s strong previous-policy-tail term after MISS and is not evidence that a cheaper key will survive sustained HITs.

Gripper split = `abs(sum kernel_weight * sign(gripper_at_head0)) < .8`. Event subsets below are **teacher-requested sign transitions**, compared with the previous actually executed gripper and including intra-head transitions in `a_inf[:5,6]`. They are metric-side grasp/release labels, not claims of physical object contact. A stuck cache can produce repeated requested releases. π0.5 positive means close; GR00T’s action sign convention is reversed. The event labels never enter retrieval.

| π0.5 cache cell | Requested grasp / release N | Full grasp split | Wrist grasp split | Full release split | Wrist release split |
|---|---:|---:|---:|---:|---:|
| spatial 49 | 137 / 245 | .3723 | .2920 | .1551 | .1184 |
| spatial 500 | 137 / 245 | .6350 | .5182 | .3796 | .3143 |
| l10 50 | 295 / 832 | .5864 | .5729 | .5325 | .5048 |
| l10 500 | 295 / 832 | .7017 | .6949 | .6058 | **.6418** |

On l10-500, wrist-only changes the vote sign relative to the full-key selector on **13.56% of requested grasps and 15.38% of requested releases**. A reduced split rate can mean a more decisive **wrong** vote; it is not a success proxy. The l10 pool2 result illustrates the same warning: mean error improves .00350 while release splits increase **.6058→.6875**. R3 already established that offline error does not rank SR.

**GR00T stale/cache diagnostics**, independently refitted at both scales:

| Suite | Library | Full error | cam0 error / overlap16 | cam1 error / overlap16 |
|---|---:|---:|---:|---:|
| spatial | 50 | .51715 | .54904 / .8264 | .53744 / .8373 |
| spatial | 500 | .44321 | .44799 / .5744 | .47576 / .6303 |
| l10 | 50 | .47998 | .50425 / .6552 | .53278 / .6689 |
| l10 | 500 | .43055 | .45159 / .4703 | .47118 / .5293 |

There is no universal camera winner across models. GR00T l10 requested-grasp splits, full→cam0→cam1, are **.7109→.8418→.4844** at 50 and **.8672→.7656→.5566** at 500; requested-release splits are **.6201→.6127→.6495** and **.6985→.7255→.6961** (N=512 grasps, 408 releases). All 80 full tables, step0 subsets, medians, gripper mismatch, severe splits <.5, top1 agreement and per-task results are in `RETRIEVAL_TABLES.md` and `retrieval_<model>_<suite>_<library>.json`.

### 1.5 What the encoder paths actually compute

**π0.5 PyTorch path** (the CUDA cost model applies here):

1. `src/openpi/models_pytorch/pi0_pytorch.py:_stage1_token_prep` (line 537) preprocesses observations and calls `embed_prefix` (348). Its image loop calls the policy’s `paligemma_with_expert.embed_image` for **every** supplied image before applying masks.
2. `src/openpi/models_pytorch/gemma_pytorch.py:84` calls PaliGemma `get_image_features`: policy SigLIP vision tower, post-layernorm, learned 1152→2048 multimodal projector. The checkpoint has **27 blocks, width1152, MLP4304, patch14**, and 224×224 inputs give 256 patches. The JAX counterpart is `src/openpi/models/pi0.py:embed_prefix`/`_stage1_embed_prefix` and `models/siglip.py`; do not benchmark that path against the PyTorch reference.
3. `src/openpi/cache/components/key_builder.py:_slice_cp1_fields`, `_spatial_pool_tokens`, `CP1SpatialPool16KeyBuilder` slice the finished prefix and reduce 16×16 patches to **4×4×2048=32768 per camera**. `cp1_spatial_pool_4` and mean-pool change this reducer **after** tower evaluation. Existing temporal token pruning is also after stage1; it does not reuse transformer computation.
4. `src/openpi/cache/interceptor.py:1737` calls `_stage1_fn` before CP1 and thus before `PluginStrategy.search`. A cheaper-key Method alone cannot avoid stage1. `run_stage2` fills the VLM KV; `run_stage3` performs K Euler evaluations. MISS count is already exposed through `MissConfig.num_steps` (`cache/config.py:873`), `_miss_steps` and the MISS call around interceptor line 2062.

**GR00T:** `src/openpi/cache/groot/staged.py:GrootStagedRunner.run_stage1` (460) runs Eagle’s `extract_feature(pixel_values)`, embeds text and scatters image tokens. `cache/groot/key_builder.py:slice_groot_cp1_fields` uses contiguous image-token masks, **256 tokens/view**, with `GrootLibero*` two-camera builders. Its reduced keys must stay consistent with Eagle’s projector/scatter geometry; changing SigLIP in π0.5 says nothing about this path. `run_stage2_llm` (646) and `run_stage3` make the other two stages; `cache/groot/interceptor.py:_require_miss_steps` requires the bundle K to equal the live action-head step count. Therefore use separate GR00T endpoints configured with `--denoising-steps K` for a K ladder. The current `exp/offline_search/closed_loop/ops/start_server.sh:62` hardcodes eight steps: the coordinator must expose K in an experimental launcher and match the bundle configuration. A YAML-only change is insufficient. Read `exp/libero_groot/bench_profile.py` for the 256-wire-image → 224-model preprocessing contract.

The installed Eagle implementation, `/home/weiland/.cache/huggingface/modules/transformers_modules/eagle2_hg_model/modeling_eagle2_5_vl.py:311`, also reveals an early-key trap: setting `select_layer` calls the **entire** vision model with `output_hidden_states=True` and then indexes the requested layer. That produces an intermediate representation but saves no tower execution. A cheaper intermediate key requires actually stopping the tower at that layer and retaining the state for MISS completion.

A certified **existing** GR00T CUDA-graph cost asset is available: `exp/libero_groot/config/rit/cost_groot_libero_measured.json`, backed by `data/latency/libero_cg_k8_p0_r0.json`. It records **6.145863 / 7.191965 / 28.104 ms**, total **41.441827 ms**, shares **.148301 / .173544 / .678155**; stage3 uses the owner’s replacement measurement, not the bench cell’s 24.419450 ms. Certification reports 2,000 graph launches, 200 iterations, 30 warmups, K8, two-camera LIBERO geometry. This was Torch2.5.1; the accessible environment is Torch2.7.1. Use it as an explicitly labeled historical model, and confirm the current serving split before declaring a new GR00T frontier. R3’s 58/232 ms smoke and loaded multi-server timings are not this split. No π0.5 IR is assigned to GR00T as if measured.

### 1.6 Cheaper-key cost evidence and missing measurements

The checkpoint-only CPU probe ran three stored spatial images per variant, FP32, one thread. Its full-key reconstruction on the first image has cosine **.99999529** with the stored key and relative RMS **.00289893**; this supports the loading/preprocessing path, not CUDA bit parity. These are feasibility numbers only:

| Variant, one view | CPU median ms | CPU/full | Analytic transformer MAC/full | 4090 stage1 ratio | Whole-library quality |
|---|---:|---:|---:|---|---|
| 27 blocks, 224, 4×4 pool | 4770.362 | 1 | 1 | unmeasured | full reference measured |
| first 6 blocks, 224, raw 4×4×1152 | 1025.611 | .2150 | .2222 | unmeasured | **unmeasured at 50 and 500** |
| first 12 blocks | 2054.066 | .4306 | .4444 | unmeasured | **unmeasured at 50 and 500** |
| first 18 blocks | 3059.356 | .6413 | .6667 | unmeasured | **unmeasured at 50 and 500** |
| 27 blocks, 168 | 2878.184 | .6033 | .5533 | unmeasured | **unmeasured at 50 and 500** |
| 27 blocks, 112 | 1425.035 | .2987 | .2430 | unmeasured | **unmeasured at 50 and 500** |
| final 2×2 pool | 4769.127 | .9997 | 1 | unmeasured; no tower work removed | measured, both scales |
| final global pool | 4875.843 | 1.0221 | 1 | unmeasured; no tower work removed | measured, both scales |
| one complete camera | not separately timed | — | roughly half the two active towers | **unmeasured** | measured, both scales |

MAC ratios from `semantic_audit.py` use `depth*(4ND² + 2NDM + 2N²D)`; they exclude patch embedding, projection, pooling, memory and launch costs. Resizing uses bilinear antialiasing and interpolates the tower’s learned positional embedding. Intermediate keys are raw hidden states pooled before the final-only normalization/projector: applying a final-layer projector to an untrained early representation is not assumed valid. A library-only PCA can map that 18432-D key to 64-D per view.

**Exact previous-image reuse:** `image_reuse.py` found **zero exact consecutive-image matches** on either active camera for every query cell. π0.5 pair counts are **1,075 / 1,343 / 3,020 / 4,126** (sp-inf/sp-cache/l10-inf/l10-cache); GR00T **1,064 / 1,327 / 2,625 / 3,884**. The benefit is zero on these data at either library size. Approximate patch reuse is not exact after self-attention: changed patches influence unchanged ones in subsequent layers. It would need a separate approximation, consistent library reconstruction, and error/loop evaluation. Bounded vision-free stretches are agent A’s topic.

**Required completion experiment when the coordinator has GPU access:** one vision-only process, ≥16 GiB free before admission, allocation cap <8 GiB, release after each measurement, stop permanently on OOM. Benchmark full production stage1 and each branch on the same tok images, batch1, matched dtype/backend, warmup30 and 200 CUDA-graph replays; include preprocessing, pooling and the graph-boundary/resume overhead. Measure cheap-HIT, cheap+completion-MISS, and cheap+recompute-MISS separately. Rebuild **all 1,018/2,640 and 10,909/29,472 π0.5 library keys**, refit PCA/AWM separately at each scale, and evaluate the same 1,393/4,176 cache and 1,125/3,070 inf queries. Save large arrays only under the owner’s derived-data destination. `vision_probe.py --device cuda` is a guarded eager feasibility probe, **not a completed CUDA-graph benchmark or full-library extraction tool**.

### 1.7 Storage: both scales, with units explicit

`make_tables.py` → `fit_bytes.json` stats the existing deployment fit artifacts; decimal MB below. Some R3 tables called MiB “MB”, explaining their 63/135 versus 66/141 figures. The owner-specified deployed pkl comparison is retained as nominal **431/1103 MB π0.5**, **429/1068 MB GR00T**.

| Cell | Episodes, current / large | Entries, current / large | Existing AWM pkl MB, current / large | Existing MixedJudge pkl MB, current / large | Owner deployed pkl MB |
|---|---|---|---|---|---:|
| π0.5 spatial | 49 / 500 | 1,018 / 10,909 | 21.342 / 46.993 | 26.036 / 66.084 | 431 |
| π0.5 l10 | 50 / 500 | 2,640 / 29,472 | 24.631 / 94.143 | 32.602 / 141.227 | 1103 |
| GR00T spatial | 50 / 500 | 1,063 / 11,751 | 22.249 / 58.157 | no R3 mixed artifact | 429 |
| GR00T l10 | 50 / 500 | 2,645 / 29,631 | 26.673 / 117.304 | no R3 mixed artifact | 1068 |

AWM retrieval code is **580 B/entry**, MixedJudge’s stated code budget **626 B/entry**; valid full actions are **280 B/entry π0.5**, **448 B/entry GR00T**. Existing pickles retain padded `(H,32)` actions and auxiliary arrays, sometimes duplicated, so do not confuse these compact counts with actual pickle sizes.

A one-camera 64-PC+state8 metric needs **324 B/entry** (72-D code, state8, one early-code norm), a **8.519680 MB PCA mean/basis**, and approximately **.633600 MB** for the conservatively counted ten task metric blocks. A compact full-key design has 17.039360 MB PCA plus 2.241280 MB metric blocks. These are **calculated representation budgets**, not serialized new artifacts. With full valid actions retained:

| Cell | Full compact MB, current / large | One-camera compact MB, current / large |
|---|---:|---:|
| π0.5 spatial | 20.156 / 28.662 | 9.768 / 15.742 |
| π0.5 l10 | 21.551 / 44.627 | 10.748 / 26.954 |
| GR00T spatial | 20.373 / 31.361 | 9.974 / 18.225 |
| GR00T l10 | 22.000 / 49.741 | 11.195 / 32.028 |

Camera-aware guard wrappers add their own calibrated state/vision tables; the camera compact figures do not silently include a serialized MixedJudge. Smaller final pooling reduces the fixed PCA matrices, not the 580-B AWM code or the tower cost. The 500-episode option is deployable under the owner ruling; no scheme here requires retaining the 262-KB raw key per entry online.

## 2. Ranked proposals

### B1 — `miss_k2`: make each π0.5 rescue cheaper

**Pitch / mechanism.** Keep the proven HIT selector and decision rule, but use two ordinary full-range Euler steps when the decision is a MISS. Plain K2 already matches full inference on both LIBERO suites. Unlike a cached warm start, it starts from fresh noise and conditions on the complete current prefix. Risk: states reached after a string of HITs may demand more precise denoising than ordinary inference states.

**Algorithm and integration.** Keep `AWM(lib=current,kref=5)` or `AWM(lib=big,kref=8)` and the R3 method flags. Add the existing YAML `miss: {num_steps: 2, evidence_dir: <arm-local-dir>}`; use `write_policy.type: never` and full-model serving (`STAGE1_ONLY=0`). `src/openpi/cache/config.py:_miss_errors` also requires no `warm_reset` or routing block and disables trace/shadow-teacher mode for this MISS evidence path. `scripts/serve_policy.py` already installs the MISS runtime; the interceptor passes K through `_miss_steps()` and records `miss_nfe`. No Method/API or `src/` edit is necessary. The plugin should include `miss_nfe`/K in collected decision evidence. Both `closed_loop/ops/collect.py:34` and `ops/kpi.py:48` currently hardcode K10 IR; the experimental collector must use `.152+m(.410+.438K/10)`. Static K2 per arm first; a per-reason K2/K10 policy requires a new per-request dispatch contract and is not the initial implementation.

**Predicted SR/IR, explicitly hypotheses.** At 50 episodes, forecast no systematic >2–3 pp loss from the matched K10 schedule: l10 perk5 around its .792 reference, guard-only around .740; spatial guard-only around .888. At 500, forecast l10 guard-only remains near .864. These are mechanism-based expectations, not CIs. At unchanged m, IR is **.24756 / .25259 / .21916 / .20259**, respectively. Spatial-500 has no measured guard-only m/SR: retain pure-cache .954 at .152 as its anchor and do not invent that missing point. For GR00T both scales, the pure-policy l10 step loss makes reduced-MISS SR uncertain; start at K4 with K8 control, then K2 only if the paired loss is small. Its historical-cost full K2 IR is .49138, K1 .40661; mixed rates must use GR00T’s own shares.

**Tier / bytes.** T0 inference configuration atop T1 AWM. No extra library bytes: use the exact 50/500 artifact row in §1.7; policy weights are already required by mixed mode. This changes the **control/inference-cost layer**, not synthesis, fitting or library size.

**Cheapest diagnostic.** Existing §6.15 evidence is already sufficient to justify a pilot. Inspect executed NFE and same-noise K2/K10 actions on the first few permitted coordinator MISSes; offline action closeness is only a wiring/collapse check. No GPU denoising experiment was run by this agent because our GPU authorization was for keys only.

**Closed-loop pilot.** π0.5 l10: libraries 50 and 500 × guard-only K10/K2; additionally 50 and 500 × periodic5 K10/K2 (periodic5-500 is also a new schedule control). Spatial: both libraries × guard-only K10/K2, retaining full-key pure cache. Screen catastrophic failures with all ten tasks × inits0–9, then use all ten × inits0–24 (250) for l10 triage and all ten ×0–49 (500) for confirmation. Run the pure-inference K2 baseline concurrently in the evaluation plan, not on this agent’s machine. K1 is a second-stage variant; its tiny saving at m≈.10 does not justify starting there.

**Kill criterion.** Do not adopt K2 if the paired 500-init comparison shows a reproducible SR loss >2 pp without compensating improvement at matched IR, or if it increases m/episode length enough to erase the cost saving. Report confidence intervals and task losses; a ±7-pp-noise 100-init l10 pilot cannot establish a 2-pp claim. Test K4 as the conservative fallback.

### B2 — `sparse_miss_frontier`: periodic controls, delayed progress guard, bounded HIT runs

**Pitch / mechanism.** Most g500 budget is no-progress intervention, while at the 50-episode l10 library blind periodic MISSes already beat targeting. Map how much intervention is necessary before changing the selector. Relaxing no-progress may avoid false alarms; HIT caps bound the time that drift can accumulate, while a memo reset avoids treating a policy intervention as continued cache stagnation.

**Algorithm / integration.** Reuse `MixedJudge(guards=True, events='none', burst=0)` and `--os-judge guard_only`. Sweep only `noprog_n ∈ {3,4,5}` initially; stuck≥2, terminal∧closed and overtime∧lag remain unchanged. Use `--os-judge-cap 8` or `12` for caps. A cap8 permits eight consecutive HIT decisions and MISSes the ninth if no guard fires; it is **not** periodic8. Periodic arms use `--os-judge periodic:k` with plain AWM; periodic mode deliberately ignores guard flags. Step0 keeps the existing judged/HIT behavior but still computes vision. The reset variant clears only the progress comparison memo when `q.prev_hit is False`, seeds it with the current proposal and restarts counting; it does not clear visual stuck state or suppress hard guards. That variant needs an experimental Method subclass, no `src/` change. No hard forced-HIT cap on consecutive MISSes is recommended before testing this less aggressive reset.

**Exact low-IR map to run, both libraries.** At K10 the asymptotic periodic IRs are:

| k | 6 | 8 | 12 | 16 | 24 | 32 |
|---:|---:|---:|---:|---:|---:|---:|
| IR | .29333 | .25800 | .22267 | .20500 | .18733 | .17850 |
| K2 counterpart | .23493 | .21420 | .19347 | .18310 | .17273 | .16755 |

Pure cache is .152. Finite episodes have `m = sum floor(N_episode/k) / sum N_episode`, below 1/k; e.g. on observed l10 g500 lengths periodic8/12 replay to **.25293/.21588**, not .258/.22267. Report realized m after new rollouts. The existing full-key system cannot reach IR=.150 exactly without changing vision cost; .152 is its lower endpoint.

**Pilot allocation.** Core l10, each library: full-key pure cache; periodic6/8/12/24 K10; guard3/4/5 K10; guard4+cap8 K10; guard3+progress-reset K10. Existing paired guard3 and pure-cache results provide historical controls, but re-run the winning comparator to estimate current noise. Add periodic16/32 only to fill a demonstrated gap, rather than launching every grid point. Then cross K2 with guard3/4 and the best periodic arm. At spatial-500, start with missing guard3 plus periodic12/24 and pure cache; at spatial-50 use guard3/4 and periodic8/12. Add cap12 if cap8 overspends. All-task 100-init screening, 250-init l10 triage, and paired 500-init confirmation as in B1. Include GR00T l10 periodic8/12 and its own library-calibrated guard3/4 at both scales only after validating its motion guard thresholds; the π0.5 guard budget cannot be assumed to transfer.

**Predicted effect / decision targets.** Hypothesis: g500 guard4 preserves most of .864 SR while moving toward replay IR **.19917** (or **.17968** with K2). A useful target is **SR≥.844 at IR≤.21**; this is a target, not a measured forecast. At 50 episodes, test periodic6 for keeping SR within 3 pp of .792 at IR<.30; guard4 is lower priority there because its same-scale guard3 reference is only .740. Guard4+cap8 is a reliability variant, not automatically a lower-cost variant (§1.3). Spatial-500 targets .95 or better near .18–.22, anchored by .954 pure cache at .152; spatial-50 targets a gain over .800 pure cache with less than .30 IR. GR00T SR change is unknown at both scales until its first mixed pilot.

**Tier / bytes.** T0 schedule or small Method-state change atop T1 AWM. Periodic uses the AWM artifacts; guards use MixedJudge artifacts in §1.7. No extra candidate rows, no fit borrowing. This is the control layer; synthesizer and library stay fixed.

**Cheapest diagnostic / kill.** The frozen-observation replay is complete. Keep the K10 schedule arms only if their closed-loop point is above the same-library existing frontier/chord, with uncertainty. At 500 l10 the endpoints are (.152,.768) and (.238216,.864): at IR=.199169 the chord is **about .8205**. A new guard4 point below that is uninteresting; `.844 at ≤.21` is the stronger adoption target. If periodic8/12 is within 2 pp of a guard variant at matched realized IR, prefer the simpler periodic rule. Do not claim a targeting win from unequal budgets. No reintroduction of early event bursts, terminal masking, or gripper-sign commitment.

### B3 — `exact_padding_elision`: reuse the absent camera, optionally pack masked VLM tokens

**Pitch / mechanism.** Remove work on information that is fixed or fully masked before throwing away an active view. This has the cleanest SR hypothesis, but no GPU saving is credited until measured.

**Source evidence.** `src/openpi/policies/libero_policy.py:59–68` supplies two real views plus a zero `right_wrist_0_rgb` with mask false for π0.5. `embed_prefix` still encodes all three; key-builder field enablement only controls later slicing. The default π0.5 prefix has **3×256+200=968** token slots. Fixed dummy-camera packing makes **712**, before any optional text-padding elimination. The transformer receives no attention from valid queries to masked keys (`make_att_2d_masks`), and `denoise_step` positions its suffix by the **sum of valid prefix masks**, not allocated sequence length.

**Algorithm / server changes.** Implement in an experimental serving wrapper under `exp/`, not by modifying `src/`:

1. Cache the exact model/dtype/device-specific dummy-image embedding once after preprocessing. Supply that tensor in the third prefix slot on every decision; preserve all CP1 offsets, masks, positions and outputs. Do not silently assume an all-zero normalized image: cache the actual preprocessed constant. Retain the original layout and return type. This needs a stage1 dispatch override before the plugin query, not a Method-only change.
2. Optional independent MISS variant: after CP1 has judged MISS, wrap `_stage2_fn` with a packed `Stage1Output`, deleting only positions512:768 from embeddings, pad masks, both attention axes, and position IDs. Run the original stage2/3 functions on that packed structure, so KV and `stage2.stage1.prefix_pad_masks` have matching lengths. Preserve the original stage1 for CP1 bookkeeping. Do not pack before the fixed-offset CP1 builder; do not drop any active image patches; initially leave variable text padding untouched for stable graph shapes.

`semantic_audit.py` verifies mask submatrices, position IDs and suffix offsets for five text-valid lengths: **15/15 checks pass**. That is algebraic evidence only. Changed GEMM/attention shapes can change floating-point rounding, so full-prefix KV/action parity with fixed noise and both graph paths is still mandatory. This proposal is unavailable as-is on GR00T: its LIBERO path already has two active cameras and mask-based image scatter. No GR00T savings at either library scale are asserted.

**What stage2 can and cannot save without altering model semantics.** Reusing a task’s whole VLM KV is invalid: `embed_prefix` deliberately makes image and text prefix tokens bidirectional, so text KV after the first attention layer depends on the current images. Smaller images, fewer active tokens, or reusing old image KV changes the policy. Static text **embedding lookup** can be reused, but that belongs to stage1 and does not save the 27.69-ms VLM pass. Deleting genuinely masked positions after CP1 is the narrow semantic-preserving opportunity identified here. Its linear token-work ratio is **712/968=.735537**, quadratic attention-work ratio **.541015**; neither is a latency measurement. Unmodified serving offers no switch for this packing, so the conservative cost tables credit **zero stage2 saving**.

**Predicted SR/IR and cost.** Hypothesis at both 50 and 500: exactly the same mathematical retrieval/action function; practical SR should remain within rerun noise if parity passes. Constant-image caching alone might reduce stage1 toward roughly two-thirds if three similarly priced towers dominate it; this is **an unmeasured forecast**, not .05067 IR earned. Unlike an approximate key, this reuse can also save on MISS because the full representation is still produced. With illustrative `r_all=2/3`, the observed l10 guard K2 points become **.20192 at 50**, **.15192 at 500**, spatial-50 **.16849**. The relevant full-key K2 values are .25259/.20259/.21916. If packed stage2 later measures ratio q, subtract a further `m .410(1-q)`; no value of q is assumed in the recommended table.

**Tier / bytes.** T0 exact computation reuse. Same libraries and fit bytes at both scales (§1.7); one 256×2048 dummy embedding is approximately **1.05 MB bf16** process-level storage, not per entry. Packing reduces temporary prefix/KV allocation, not library size. This is a cost implementation change, with no synthesis, metric or library effect.

**Cheapest diagnostic / pilot / kill.** Coordinator: fixed-input, fixed-noise parity for dummy-only and pack-only separately, including state/task changes, two concurrent connections under the existing lock, and graph-buffer lifetime. Clone retained graph outputs or otherwise ensure they cannot be overwritten by the next request. Then both π0.5 suites × both library sizes on unchanged guard3 and pure-cache arms; 100 all-task inits screen integration, 500 paired inits check any apparent SR difference. Kill on a meaningful same-noise action discrepancy, stale cross-connection state, or less than .02 net measured IR saving for the dummy branch. Packing is optional; do not couple its numerical risk to the easy dummy-cache test.

### B4 — `own_vision_early_key`: wrist-first π0.5 or an intermediate tower, with full MISS completion

**Pitch / mechanism.** Lower the every-HIT vision floor while keeping a visual anchor **on every decision**. Start with a complete single-camera key because its same-library diagnostics are available. In parallel qualify two-view intermediate-layer keys, which retain scene coverage and can continue to full stage1 on MISS.

**Algorithm.** First π0.5 candidate: current wrist view only, 4×4 pool → PCA64, concatenate state8, AWM whitening refitted from the **same 50 or 500 candidate library**, same step0 branch, top16 and kr5/kr8, unchanged synthesis. Return normal `api.Result` rows/actions and log the key variant. Implement a camera-aware AWM adapter rather than presenting missing-camera zeros to the unchanged 136-D fit. At query time no teacher information is used. Refit V7 and the visual-still guard in the reduced space; using the old two-camera calibration or minimum-camera rule is invalid. Initially combine with periodic schedules, which avoid confounding a changed key with an uncalibrated judge. Compare guard-only only after the new guard is calibrated from the same library.

For intermediate variants, use both current views through block6/12/18; pool raw hidden states to4×4×1152, fit PCA64 per camera, then the identical AWM recipe. At MISS continue the retained full-resolution hidden states through remaining blocks, post-layernorm and projector, restore all original prefix components and run unchanged stage2. Store per-request intermediates under the existing connection/dispatch ownership; never reuse another connection’s CUDA graph output. For 168/112 resolution, interpolate positions consistently at build/query time and **pay a complete original-resolution stage1 on MISS**. A resumable early branch and a resized branch must not share the same IR formula.

**Plugin/server requirements.** The existing method runs only after full stage1, so a new experimental interceptor wrapper is necessary: cheap vision → CP1 method/judge → HIT returns library chunk / MISS completes or recomputes stage1 → existing stage2/3 and `on_executed`. Keep `prev_hit`, executed full policy chunk, step numbering and history semantics intact. Extend plugin key buffers to the actual dimension or feed preprojected64-D camera features through an explicitly defined Method adapter; current `OnlineQueryView`/buffers assume standard fields and cannot magically expose early keys. Disable/replace the full-key native shadow search for these arms so instrumentation does not silently compute the expensive key. Log cheap vision, completion/recompute, stage2, stage3/NFE separately and charge all policy work.

**Predictions at both scales.** No defensible SR number can be inferred from §1.4’s action errors. The explicit hypothesis is a ≤2-pp paired SR loss from the matched full-key arm at enough vision saving to move the frontier. At 50 episodes wrist-only looks safer than fixed-only on l10 (error +.01747 vs +.04528); at 500, +.01549 vs +.06030, but the release-vote changes make a substantial SR loss plausible. Spatial gives no universal camera preference at 50; wrist is the cheaper-key pilot candidate, not a declared winner. GR00T should test camera0 and camera1 independently at both scales; its offline pattern differs, and library-side raw images are absent, so resized/intermediate GR00T keys require regeneration from original library observations rather than pretending the stored final tokens contain earlier layers.

At a **hypothetically measured r=.5** with exact MISS completion, K2 l10 guard rates give **IR .19195 (50) / .13432 (500)**; if the cheap pass must be recomputed, **.20731 / .14204**. At r=.25 the reusable figures are **.16163/.10018**. These are conditional cost calculations, not measured operating points; the changed selector can change m. With the unmodified current plugin the actual saving is zero because stage1 already ran.

**Tier / bytes.** T1 refit, no new trained model. Camera code324 B/entry and calculated totals9.768/10.748 MB current π0.5 sp/l10,15.742/26.954 MB large (§1.7), versus431/1103 MB deployed. Early two-camera codes stay580 B/entry; a raw-width1152 pool has fixed PCA mean/basis **9.584640 MB** for both cameras before metric blocks. Resized final projected pools retain32768-D raw keys and their17.039360-MB PCA basis unless the pooling shape is separately reduced. All fits must be repeated at50 and500; using500 covariance/PCA for current candidates would be **borrowed big-library information** and must be a distinct arm.

**Cheapest diagnostic / pilot / kill.** The camera refit diagnostic is complete; the missing CUDA timing and intermediate/resolution refits in §1.6 are prerequisites for those variants. Pilot π0.5 both suites × both library sizes: full-key vs wrist-only under the **same periodic8 K2** schedule, plus the full-key guard3 K2 reference; then test the best qualifying intermediate candidate under that same schedule. All-task100 init collapse screen, l10≥250 triage,500 confirmation. Look especially at spatial task6 and l10 tasks0/6/8 plus all-task release failures. Reject a variant for deployment if r≥.8, if duplicate work erases net saving, or if a paired SR loss >2 pp remains at matched IR. Large offline error/vote changes are reasons to prioritize a collapse check, not a proof of low SR. Do not add gripper commitment or release confirmation to “repair” the key; those mechanisms are already dead.

## 3. Stacked recommendation and required baselines

All rows below use **observed or frozen-replay m**, not a predicted m after changing the policy/key. `r=.5` is illustrative pending a 4090 measurement. The table deliberately exposes the duplicated-MISS penalty. No padding/packing saving is credited in these main columns.

| π0.5 combination | Library | m source | Full key K10 | Full key K2 | Reusable r=.5 key + K2 | Duplicate r=.5 key + K2 |
|---|---:|---:|---:|---:|---:|---:|
| l10 guard3 | 500 | observed .101670 | .23822 | **.20259** | .13432 | .14204 |
| l10 guard4 | 500 | replay .055624 | .19917 | **.17968** | .10791 | .11213 |
| l10 guard4 + cap8 | 500 | replay .144138 | .27423 | .22372 | .15868 | .16963 |
| l10 guard3 + progress reset | 500 | replay .086367 | .22524 | .19498 | .12554 | .13210 |
| l10 periodic5 | 50 | observed .192050 | .31486 | **.24756** | .18616 | .20076 |
| l10 guard3 | 50 | observed .202142 | .32342 | .25259 | .19195 | .20731 |
| l10 guard4 | 50 | replay .145297 | .27521 | .22430 | .15934 | .17038 |
| l10 guard4 + cap8 | 50 | replay .219037 | .33774 | .26099 | .20164 | .21829 |
| spatial guard3 | 49 | observed .134964 | .26645 | .21916 | .15342 | .16367 |
| spatial guard4 | 49 | replay .103748 | .23998 | .20362 | .13551 | .14339 |
| pure cache, either suite | 50 and500 | m=0 | .15200 | .15200 | .07600 | .07600 |
| all MISS, either suite | library irrelevant | m=1 | 1.00000 | .64960 | .64960 | .72560 |

The last row is a useful accounting check: an exact reusable cheap prefix gives **no approximate-key saving on MISS**; a duplicate cheap pass makes pure inference worse. Skipping the unnecessary cheap branch on known periodic-MISS decisions can avoid duplication, but must still account for any Method proposal/guard bookkeeping that is retained. Do not price lower-resolution MISSes as reusable by default.

**Mandatory SR-vs-IR plot companions, shown on both 50/500 panels:**

| π0.5 baseline | Denoise K | Execute L control steps/call | IR per five controls |
|---|---:|---:|---:|
| pure inference, independent policy-seed replicates | 10 | 5 | **1.0000 each** |
| reduced steps only | 4 | 5 | .7372 |
| reduced steps only | 2 | 5 | **.6496** |
| reduced steps only | 1 | 5 | **.6058** |
| longer chunk only | 10 | 8 | .6250 |
| longer chunk only | 10 | 10 | **.5000** |
| reduced + longer chunk | 2 | 8 | .4060 |
| reduced + longer chunk | 2 | 10 | **.3248** |
| reduced + longer chunk | 1 | 10 | **.3029** |

These pure-policy baselines have **zero retrieval-library bytes**, in both panels. L=10 is π0.5’s stored/configured chunk horizon; do not invent an L=20 baseline without changing the model contract. Execution of a longer chunk is agent A’s topic, but its baseline is necessary here: K1+L10 approaches the .30 edge without retrieval. Use a control-normalized denominator when L changes: nominal `IR=(5/L)(.562+.438K/10)`, and report actual total modeled compute per executed control step, including truncated final chunks, plus controls/episode. Merely using the new request count in both sides of the old per-request formula would conceal the saving. Lower SR can shorten/lengthen episodes, so give the per-episode totals too.

For GR00T using the explicitly historical split in §1.5, pure K2/K1 at L5 are **.49138/.40661**; K2 at L10/L16 **.24569/.15356**, K1 at L10/L16 **.20331/.12707**; K8 at L10/L16 **.5000/.3125**. These are calculated baselines, **SR unmeasured for longer execution at either library scale**. They are particularly strong controls for any claimed GR00T low-IR cache point. Confirm current stage costs before a definitive combined plot.

Run at least **three independent pure-inference policy seeds** on the same500 inits, holding environment inits/seed fixed. Report the mean and between-run variation and pair each candidate against matching seeds where available. Three evaluation replicates do not turn each deployment point into IR3: each executes one policy sample and costsIR1. Best-of-three or action averaging would be a different, more expensive policy. The single .844 π0.5 l10 sample is insufficient to label .864 “better than the policy”; R3’s paired p=.36 already says that.

### Effects kept separate

- **Synthesis:** held fixed at AWM’s original kernel16 mean. No proposed gain is assigned to synthesis, and gripper commitment is excluded.
- **Method at fixed library:** B4’s camera/intermediate representation and separately refitted metric. Compare it to full-key AWM with the **same candidates, kref, schedule and K**. B3 aims for no method change at all.
- **Library:** repeat each accepted change at50 and500, preserving the independent fit source. R2’s library gains remain +15.4/+13.8 pp π0.5 sp/l10 and +7.8/+15.4 pp GR00T; they are not gains from cheap vision or the judge. The established kr5-at50/kr8-at500 mismatch should be held within-scale, or add a single kr5-large control if isolating the library effect exactly.
- **Control/inference:** B1/B2 and longer-chunk baselines, with every MISS/vision/denoising call priced. B1×B2 interactions and changed m must be measured; multiplying isolated SR gains is invalid.

## 4. Rejected ideas and precise reasons

1. **“Pool fewer final patches to save stage1.”** The source path has already run all27 vision blocks before either pool. The CPU probe gives pool2/full **.99974**, pool1/full **1.02211**. The owner’s stage-forward cost model therefore gets no demonstrated reduction. Refit results at both scales can be reasonable (π0.5 l10-500 pool2 error .46691 vs .47042), but that is a representation/memory change, not a cheaper vision pass; its release-split increase also blocks an SR claim. Keep pooling only as a compact-basis ablation or as part of a genuinely truncated tower.
2. **“Memoize the previous active camera exactly.”** Zero exact image matches in **9,564 π0.5** and **8,900 GR00T** adjacent pairs across both regimes/suites; hence zero observed active-tower calls eliminated, independently of50/500 candidates. Small pixel differences are not equality, and transformer-global attention prevents exact reuse of later unchanged patches. The fixed dummy camera is the useful exception, isolated in B3.
3. **“Use cached exact warm continuation instead of plain cheap MISS.”** Matched LIBERO π0.5 two-step continuation is .926/.778 versus plainK2 .996/.848 (−7.0pp in both suites); GR00T l10 one-step continuation .712 versus plainK1 .846 (−13.4pp). Those warm experiments use their own recorded libraries, not both R4 library scales, so they do not prove every rebuilt50/500 AWM-start variant impossible. They are nevertheless strong evidence against spending this round’s budget before the supported fresh-noise K2 control. Warm-reset variants did not systematically beat plain reduced steps on LIBERO, either. No warm gain is claimed for either R4 library scale.

## 5. Reproduction and remaining unknowns

Run from the repository root. Every script reads original assets and writes only its local outputs; `retrieval_diag.py` keeps its large temporary arrays in memory. Representative commands (the affinity/thread prefix is required for **each** invocation):

```bash
taskset -c 12-23,56-67 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r04/ideation_B/inventory.py
taskset -c 12-23,56-67 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r04/ideation_B/retrieval_diag.py
taskset -c 12-23,56-67 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r04/ideation_B/validate_retrieval.py
taskset -c 12-23,56-67 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r04/ideation_B/schedule_replay.py
taskset -c 12-23,56-67 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r04/ideation_B/image_reuse.py
taskset -c 12-23,56-67 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r04/ideation_B/vision_probe.py --device cpu
taskset -c 12-23,56-67 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r04/ideation_B/semantic_audit.py
taskset -c 12-23,56-67 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r04/ideation_B/make_tables.py
```

Remaining unknowns are explicit: **4090 CUDA-graph cost for every cheaper-key branch; full-library early/resized-key refits at50 and500; numerical/action parity and real benefit of padding elimination; K2/K1 MISS efficacy on drifted l10 states; new schedule SR and realized m; current GR00T stage split and mixed guard calibration; longer-chunk baseline SR; pure-policy seed variance.** They require the coordinator’s permitted measurements and closed-loop pilots. The new offline tables are completed diagnostics, not replacements for those experiments.
