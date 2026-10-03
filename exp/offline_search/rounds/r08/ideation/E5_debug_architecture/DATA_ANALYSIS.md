# E5 — R8 debug architecture: campaign audit

**All 70 arms pass capture validation; scientific capability remains partial.**

2026-10-01. Capture validation: 06:12–07:30 UTC. Augmentation coverage is explicitly partial.
Run root: `/home/weiland/trace_runs/os_closed_loop/r08_main`.

## 1. What I ran

I reread the brief, my original proposal, `debug/SCHEMA.md`, both tool READMEs, and the relevant reader, validator, writer, observer, transport, augmentation, and physical-adapter code. This is an audit of the collected campaign; no implementation, GPU, server, worker, simulator, remote operation, or git operation was run.

All outputs are under `/tmp/r8cb_E5_debug_architecture/`. Reproducible scripts are beside this report. From the repository root, all Python commands use:

```bash
E5_PY=(taskset -c 18-21,62-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
       MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
       PYTHONPATH=.:src .venv/bin/python)
E5_DIR=exp/offline_search/rounds/r08/ideation/E5_debug_architecture
"${E5_PY[@]}" "$E5_DIR/callback_validate.py" --shard 0
"${E5_PY[@]}" "$E5_DIR/callback_validate.py" --shard 1
"${E5_PY[@]}" "$E5_DIR/callback_inventory.py"
"${E5_PY[@]}" "$E5_DIR/callback_quality.py"
"${E5_PY[@]}" "$E5_DIR/callback_profiles.py"
"${E5_PY[@]}" "$E5_DIR/callback_validate.py" --shards 1 --shard 0 --reverse --limit 20 --workers 3 --avoid-front
"${E5_PY[@]}" "$E5_DIR/callback_validate.py" --shards 70 --shard 58
"${E5_PY[@]}" "$E5_DIR/callback_aggregate.py"
```

At most three Python processes ran concurrently. Every reader instance had `cache_enabled=False`; this is necessary because the standard reader and decision CLI otherwise write derived caches under the capture root.

- **All 70 arms:** existing `validate.validate_arm(expected_pairs=500, require_aug=False)`; exact task/init coverage; full receipt hashing, block decoding, wire-action/control joins, selected-snapshot checks; `capacity.capacity`; writer/sender/status inventories; catalog row/task integrity and source-fingerprint cross-checks.
- **Physical smoke audit:** every task at init 0 in every arm, 700 episodes / 162,116 controls; numeric finiteness, built `physical.selfcheck.check`, and 3,988 snapshot component inventories. These are 20 distinct suite/task initial configurations, repeated across arms.
- **Research profiles:** built `provenance` and `stage_ledger` on all eight A cells, each restricted to 300 discovery episodes; `call_value` on GR00T L10-50 CU/CT/IP and π0.5 L10-50 IP; `exposure_hazard` on GR00T L10-50 FL. The latter use 100 bootstraps as tool checks, not final scientific intervals.
- **Augmentation-dependent smoke profiles:** `divergence` and `follow_vs_look`, GR00T L10 A at 50/500 libraries, init 0 in all ten tasks only. Two episode cards use task 0/init 0 in GR00T L10-50 A and π0.5 Spatial-500 A.
- **Parity:** re-compared existing saved debug-off/on response arrays in `/tmp/r8_S1/gpu_parity2`; no replay or inference was launched.

All-arm coverage comprises eight each of A/CU/CT/FL/SF1; four each of IP/O5b/P10/SW/W10/W5; and two each of SHIFT/A5/O5a. Every arm contributes 500 accepted pairs; `arms.csv` lists the exact names and per-arm counts.

Inits 30–49 were used only for campaign integrity/capacity and permitted aggregate bookkeeping. No holdout outcome selected a segmentation or threshold. Discovery filtering occurs before research tools see episodes/outcomes.

Augmentation is a moving publication. The starting snapshot had 14 `AUG_DONE` arms; `r8_groot_l10_500_CU` was subsequently quarantined as `aug__mixed_610613bf`, and its marker withdrawn. Its missing augmentation is excluded from usable coverage. The **13 audited usable arms** are:

```text
r8_groot_l10_500_{A,CT,FL}
r8_groot_l10_50_{A,CT,CU,FL}
r8_groot_spatial_{50,500}_{A,CT,CU}
```

This is an explicit frozen audit subset, not a claim about the final campaign's augmentation coverage. It contains 293,990 policy shadows and shadow looks, plus 9,379 sampled three-draw records; their ID sets match the expected sets, without duplicates or nonfinite chunk values. This checks IDs and finite chunks, not every input hash or code/checkpoint binding. π0.5 camera augmentation was outside this usable subset.

Evidence: `summary.json`, `arms.csv`, `profile_summary.json`, per-arm `validate/`, `inventory/`, `capacity/`, `quality/`, `profiles/`, `parity.json`, `action_metadata.json`, and `unaccepted_prefixes.jsonl` under the scratch root.

## 2. Do the tools/data answer the original requests?

| Original requested tool | Judgment | Concrete result or remaining gap |
|---|---|---|
| `debug validate` | Works for capture; partial semantic admission | 70/70 arms PASS, covering 35,000/35,000 accepted episodes with zero missing capture entries. It does not reject truncated scores, mismatched hash domains, absent action semantics, or absent pre-assignment stages. |
| `debug parity` | Partial | Saved actions are byte-identical for 100 episode tapes / 2,530 decisions across GR00T A/CU/Policy and π0.5 A/CU. Core observer/plugin/schema hashes match production. This does not certify all R8 wrappers under production concurrency, client overhead, or restored branches. |
| `debug provenance` | Works for support; partial stage labels | Exact member weights, demo ESS, and unknown/tail mass work. Dense-library stage coverage is incomplete. Server/catalog `lib_sha` cannot be joined directly; I separately checked source-array fingerprints. |
| `debug divergence` | Partial | Same-observation comparisons work on 1,363/1,363 sampled decisions; noise-floor draws cover 41/1,363. Full policy shadows remain pending elsewhere. Missing gripper metadata prevents gripper-flip output; normalized RMS is not a calibrated physical error or rescue effect. |
| `debug timeline` | Works for inspection; partial validation | Both episode cards render physical timelines, decision tables, and captured camera images (2/2). Physical stage/onset labels remain heuristics, not human-validated truth. |
| `debug budget` | Works for live work; partial overhead accounting | Live look/call counts, applied prefixes, and request/control denominators reconcile. Receiver latency and full capture overhead are not measured. No real camera-completion event occurred, so that accounting branch is unexercised. |
| `debug value` | Partial | Exact coins and propensities support overall randomized contrasts. No accepted record supplies a certified `stage_pre`/`pre_stage`, so the built tools cannot report certified stage-specific effects without an additional prefix-only stage join. |
| `debug camera_follow` | Partial | Follow-versus-fresh-look gaps and member spread work on 678/678 blind decisions in the two GR00T smoke arms. Camera contrasts were not assessed on unfinished π0.5 augmentation; gripper semantics and exact library-physics backfills remain missing. |
| `debug boundary_export` | Partial | Shared reader, forensics control/decision tables, and segmentation-bench exports cover much of the intent; code-reviewed, not a new segmentation benchmark here. There is no general split-enforcing feature-export contract. |
| `debug snapshot_audit` | Missing as a dedicated tool | Validator/reader plus my audit expose coverage and skipped components. Snapshot files contain simulator/controller numerics and client RNG state, but no complete server/method state or restoration certification. |
| `debug capacity` | Works for bytes; partial system forecast | Exact on-disk attribution and field totals are useful. Final augmentation bytes are not yet known; receiver counters and end-to-end capture latency are absent. |

The design choices that worked were a method-independent observer/client path, exact float64 wire actions, accepted-attempt joins, native entity IDs, lossless blocks, explicit nonapplicability, and deferred private-RNG shadows. All 70 variants/cells share the core capture code. Sampling raw keys and avoiding repeated before/after state made my original 50 MB/episode estimate far too conservative. R8 relaxed my proposed full live raw-key retention to a 1/16 hash sample: only 53,581/855,498 live looks retain raw keys. Exact images and live PCA keys preserve the present profiles; unsampled raw-feature analysis would require re-encoding and appropriate provenance.

The proposal under-specified how to enforce semantic metadata and publication invariants. A structural PASS and a tool-level “available” flag do not establish that every requested scientific quantity is defined. The gripper, stage-time, hash-domain, and augmentation-generation gaps below need explicit admission checks.

## 3. Findings for stage segmentation and lazy-lever allocation

### Capture and joins

**All 70 capture validations PASS**, with exactly 500 accepted episodes per arm and zero reported capture-missing issues. Receipts, required payloads, selected snapshots, and served-wire/applied-control joins reconcile. Augmentation is excluded from this verdict (`require_aug=False`); its separate, partial audit is described above.

The accepted population contains **35,000 episodes, 1,592,477 decisions, and 8,253,989 controls**, including 350,000 settling controls and 7,903,989 active controls, plus 202,448 snapshots. Every arm has exactly tasks 0–9 × inits 0–49. All 500 pair identities in each suite agree across arms on environment seed 7, selected initial-state hash, reset-state hash, and ten settling controls: 38 L10 arms and 32 Spatial arms. This supports valid initial-condition pairing; equal decision numbers later do not mean equal physical states.

All **25,479,632** recorded member references resolve to their declared task; no unknown rows, wrong-task members, duplicate accepted decision IDs, negative weights, or weight-sum errors at the audit's 1e-5 numerical check were found. Ten captured source-array digests per process match the catalog (700/700); catalog parquet hashes also match (70/70). `ep_len` and `progress` lack independent live-source digests.

All five client capabilities are available in 35,000/35,000 accepted episode manifests, with no accepted capture errors. The 700-episode physical smoke audit passes all 3,500 applicable checks; all inspected numeric fields are finite. The absent redundant before-state arrays are correctly marked not applicable. Server sequence sets are complete; 13,322 adjacent file-order inversions reflect concurrent publication rather than holes.

| Field/operation | Available / relevant denominator | Interpretation |
|---|---:|---|
| Both wire RGB images, served actions, state | 1,592,477 decisions | Full capture, including blind requests; no new encoder calls needed |
| Live wrist PCA key | 855,498 / 855,498 looks | Remaining requests explicitly nonapplicable |
| Live third-person PCA key | 707,984 / 855,498 looks | 147,514 wrist-only looks do not compute it |
| Live policy chunk | 122,015 / 122,015 policy calls | Other 1,470,462 decisions explicitly nonapplicable |
| Raw keys | 53,581 materialized / 99,364 hash-selected requests | 45,783 selected requests performed no encoder work |
| Auditable interior-probability call coins | 177,591 anchors | Zero coin/treatment inversion mismatches |
| Pre-assignment stage label | 0 / 1,592,477 decisions | Offline prefix-only reconstruction needs a certified join |
| Camera completions | 0 / 1,592,477 decisions | Cannot validate real completion-path behavior |

Live `k_eff` and `unsupported_mass` are explicitly unsupported on all 1,592,477 decisions. The provenance tool reconstructs support statistics from catalog rows and weights; these are derived quantities, not missing live observations silently imputed as zero.

The data now support physical event alignment that R7 lacked. Native predicates, poses, contacts, initial conditions, and exact executed prefixes can distinguish grasp/transport/release timing from a library-gripper proxy. However, a derived stage must use only controls preceding the decision; the existing `truth_stage_pre` export offers a starting point. Do not substitute the majority stage of subsequently applied controls into randomized-call estimation.

All eight A provenance profiles cover **104,458/104,458 decisions from 2,400 discovery episodes**. Per-cell support and dense-library stage coverage are:

| Cell | Decisions, 50 / 500 library | Mean effective demos, 50 / 500 | Descriptive stage available, 500 library |
|---|---:|---:|---:|
| GR00T L10 | 21,251 / 17,642 | 3.293 / 7.228 | 6,710 / 17,642 |
| GR00T Spatial | 7,539 / 6,577 | 4.059 / 10.057 | 4,163 / 6,577 |
| π0.5 L10 | 20,033 / 17,353 | 3.364 / 7.060 | 8,322 / 17,353 |
| π0.5 Spatial | 7,644 / 6,419 | 3.829 / 9.574 | 6,014 / 6,419 |

Sparse-library descriptive stage coverage is 56,467/56,467; dense coverage is 25,209/47,991. The 22,782 dense decisions marked unknown contain positive mass on unlabeled library rows, not missing row IDs. `StageTable.fit` labels successful demonstration rows and leaves other rows at -1. Dropping this unknown population would systematically change the stage-analysis population. The tool correctly retains it.

The eight A stage ledgers cover the same 104,458 decisions, including unknown stages. Request IR (`work / decisions`) is .07430–.07812; active-control IR (`5 × work / active controls`) is .07458–.07957 because final chunks can be partial. For example, GR00T L10-500 has 8,875 looks / 17,642 decisions and 87,702 active controls: .074453 request IR versus .074884 control IR. Logging actual applied prefixes resolves this denominator difference.

Catalog stages are ordinal gripper-run labels, not a common semantic manipulation ontology. For example, GR00T L10-50 A has descriptive catalog bins 0–6 in discovery. Tiny positive weights can make a kernel “cross-stage”; this alone is not a failure or call-value signal. The useful architectural result is that support and stage mass can now be measured without reconstructing missing trajectories.

On discovery inits, call-value support is 6,967/18,827 decisions in GR00T L10-50 CU, 5,596/18,561 in CT, 10,070/20,041 in GR00T IP, and 9,313/18,506 in π0.5 IP (300 episodes each). CU excludes 2,834 forced anchors and CT excludes 4,013; the other excluded decisions are nonfresh continuations. IP retains all fresh anchors in these cells. This confirms why exact override/propensity fields matter for allocation research. Each tool exports 12 endpoint/estimand contrasts, but **all 48 are in stage `unknown`** because pre-assignment certification is absent; these are overall opportunities, not stage allocation recommendations.

The GR00T L10-50 FL exposure tool supports 5,650/7,939 candidate fresh anchors over 300 discovery episodes / 21,325 decisions. Predicate change and EE displacement are available on 21,325/21,325 decisions; object-relative demo drift, robot-demo drift, and certified stall-entry endpoints are unavailable on all 21,325. The missing library-physics/frame alignment is explicit, correctly preventing EE motion from masquerading as demo drift. Next-look distance is censored on 489/21,325 terminal cases. These capability limits, plus unknown pre-assignment stages, prevent a stage-specific follow-allocation conclusion here.

### Capacity

The 70-arm capacity snapshots, collected approximately 06:15–06:33 UTC, total **259.01 GB**: **257.10 GB capture/metadata plus 1.91 GB augmentation present at scan time**. This covers `runs/<arm>/debug` excluding derived caches and unfinished `.part` files; shared catalog maps add approximately 1.92 MB, while preexisting model/library artifacts are outside this accounting. The scans include augmentation later quarantined and are not a forecast of completed augmentation. Retained unaccepted material accounts for 28.88 MB, shared metadata for 3.68 MB. All 70 arm means are below 20 MB/episode; the largest attributed episode is 24.16 MB. Filesystem free space at inspection was approximately 1.925 TB.

Decimal units below. Mean/p95 describe accepted episode attribution, including augmentation present during each arm's scan; p95 is pooled over episodes, not an average of arm p95s.

| Model × suite | Arms / episodes | Decisions | Capture GB | Aug GB at scan | Mean MB/episode | p95 MB/episode |
|---|---:|---:|---:|---:|---:|---:|
| GR00T L10 | 17 / 8,500 | 554,512 | 98.47 | 1.470 | 11.76 | 21.92 |
| GR00T Spatial | 13 / 6,500 | 153,215 | 28.12 | 0.386 | 4.39 | 7.98 |
| π0.5 L10 | 21 / 10,500 | 665,747 | 97.06 | 0.053 | 9.25 | 17.17 |
| π0.5 Spatial | 19 / 9,500 | 219,003 | 33.46 | 0 | 3.52 | 6.48 |

Compressed field MB per accepted episode: cell-wide retained field bytes divided by 500 × arms, including the six small rejected prefixes in the numerator:

| Cell | Third RGB | Wrist RGB | Raw keys, both | Decision JSONL | Actuator substeps | Contact forces | Entire client capture |
|---|---:|---:|---:|---:|---:|---:|---:|
| GR00T L10 | 5.099 | 4.486 | .431 | .435 | .230 | .093 | .888 |
| GR00T Spatial | 1.798 | 1.638 | .140 | .168 | .088 | .094 | .490 |
| π0.5 L10 | 4.049 | 3.477 | .248 | .422 | .224 | .087 | .871 |
| π0.5 Spatial | 1.458 | 1.271 | .084 | .156 | .086 | .093 | .483 |

Field figures are ZIP member compressed bytes (container overhead excluded); the client column overlaps its listed fields. Full per-field inventories are in `capacity/*.json`. Images total 208.75 GB, about 80.6% of footprint. Raw keys total 7.96 GB; actuator commands 5.70 GB; contact forces 3.19 GB. Preserve physical detail: image storage is the main porting capacity risk. Re-measure resolution, camera count, scene compressibility, and episode duration for each new benchmark.

### Writer, receiver, and timing

All 70 writers drained with zero errors, zero oversized items, and **1,592,660 accepted = written** records, including rejected prefixes. Writer counters record 578.43 GB raw arrays and 221.74 GB serialized payload across 192,933 blocks. Peak queued bytes were 43.19 MB, 8.04% of the 512 MiB limit. This exceeds my proposed 8 MiB server queue: the deployed larger bound provided useful headroom. Individual queue peaks cannot be summed into a concurrent-memory measurement. Recorded queue waits total 7.64 seconds over all writer submissions; the accepted-decision mean is 0.00480 ms. Background serialization totals 46,758.8 seconds across writers; those overlapping spans are not additive serving latency.

All 35,000 accepted attempts have sender-stat snapshots. Recorded peak sender queue is 276,934 bytes; 11 attempts report two connections, the others one, and no snapshot reports spilling. The field called `reconnects` includes the first connection. There are 35,006 receipted attempts in total, including the six CT error prefixes. Receiver durability is supported by receipts and payload hashes, but there are no receiver throughput, ACK-latency, fsync-latency, or saturation counters.

| Cell | Decisions timed | Mean `observer_ms` | Range of per-arm p95, ms |
|---|---:|---:|---:|
| GR00T L10 | 554,512 | 1.202 | 1.348–2.226 |
| GR00T Spatial | 153,215 | 1.195 | 1.395–2.161 |
| π0.5 L10 | 665,747 | 1.297 | 2.272–4.819 |
| π0.5 Spatial | 219,003 | 1.313 | 2.222–5.407 |

The aggregate observer mean is 1.256 ms, maximum 491.604 ms. This is **not full instrumentation overhead**: `observer.finish` timestamps before its final diagnostic split and writer submission/JSON sizing; metric-tap copies occur inside `infer_ms`; client control capture/compression has no per-control timer. Transport/client delays and production scheduling require separate measurements.

Saved parity tapes give the following **inner inference** mean timings, debug off → on: GR00T A 42.83→44.99 ms (496 decisions), CU 96.42→100.67 (496), Policy 268.52→274.96 (496); π0.5 A 44.67→49.35 (521), CU 169.62→171.82 (521). Differences range +1.3% to +10.5%, but these are single sequential runs without matched-load repetition, and exclude outer observer work. They establish action equality for those tapes, not a causal production-latency estimate. The coordinator log also records six real-simulator action-tape parity cases; I did not rerun them.

## 4. Bugs and data-quality problems

| Issue | Concrete file/field and observed scope | Consequence |
|---|---|---|
| Retrieval score truncation | `closed_loop/plugin.py` uses `TOPK_SAVE`; `debug/server/observer.py` records `d.scores`. All 1,592,477 records have 16 rows/weights but 10 scores, including all 855,498 fresh retrievals. | Member-score alignment is incomplete; current capture validation misses this. Save all scores or an explicit scored-row map. |
| Ambiguous library hash | Every decision's `lib_sha` differs from its catalog's `lib_sha`; `observer.py:123` chooses the original pickle source hash, while catalog IDs hash store content. | A nominal `(lib_sha,row)` join fails. Recorded source-array fingerprints match the catalog; unify content identity and distinguish source ancestry. |
| Missing action semantics | 0/70 server manifests expose gripper dimension, threshold, or action scale; 70/70 expose valid indices 0–6. `decision/common.py:dimensions` and `metrics.py:action_metrics` return null gripper flips but include all seven channels in `motion_rms`. | Motion-only and gripper questions are not answered by the apparent metric name. Require named motion/gripper dimensions and normalization metadata. |
| Incomplete live stochastic provenance | The schema/observer and all 70 field inventories contain no live policy-noise tensor or server RNG snapshot; RNG arrays are client snapshot state and private augmentation seeds. Live checkpoint/normalizer content identity is not explicit in server manifests. | Saved actions remain exact, but deferred private draws cannot reproduce the realized stochastic serving draw. Preserve explicit live provenance when reproduction is required. |
| Physical storage vs semantic status | `array_status.policy_chunk=available` on 1,470,462 decisions whose `policy_chunk_status=not_applicable`. PCA/cache placeholders have the same two-layer issue. | Array existence must not be interpreted as valid measurement. Make one authoritative semantic validity mask available through the reader. |
| Uncertified stage timing | No `stage_pre`/`pre_stage` in any accepted decision; `common.stages` falls back to descriptive catalog labels. | Randomization is auditable, but stage-conditioned causal use requires explicit pre-decision evidence/provenance. |
| Mutable augmentation publication | Starting `AUG_DONE` for GR00T L10-500 CU was withdrawn; directory renamed `aug__mixed_610613bf`. Coordinator/E2 report 361 intermediate-version shadow-look shards and failed live-anchor agreement. | Completion marker alone is insufficient; read an immutable, validated generation. My coverage excludes this arm. |
| Derived report identity | `decision/common.py:write_report` records tool name/schema/runtime but does not bind tool source/config, capture manifest, split, and augmentation generation hashes. | Persisted reports cannot independently establish which evolving inputs/code produced them. This callback adds split labels and keeps its scripts, but that is not a general provenance contract. |
| Read-only analysis violation | `reader.ArmData.cache_enabled=True` by default; decision `common.cli` does not disable it, despite its README's read-only claim. | Ordinary profile runs write `debug/derived`; wrappers were needed to respect this callback's rules. |
| Retry census hides failures | `validate.unaccepted_attempts=0` follows the scientific journal, yet six receipted CT error attempts contain 183 decisions / 945 controls. All report `ValueError: stage masses/occupancies must be finite probabilities`. | Keep a separate operational attempt ledger and retry reason; do not equate scientific admission with zero serving faults. |
| Incomplete timing/transport counters | `observer.py:205`, `writer.py:put`, `client/capture.py:flush_controls`, and `transport/sink.py:close`. Sender stats are copied into the completion frame before final drain/ACK or a possible close-time spill. | Recorded stats are snapshots, not final counters; neither overhead nor final transport behavior is fully measured. |
| Snapshot scope | All required snapshots are uncertified; none of 3,988 inspected snapshots contains complete server/method state. Controller object references, names/index lists, and nested fields are listed as skipped. | Numeric simulator snapshots alone cannot support controlled policy/cache branch counterfactuals. |

The CT exceptions are serving-method failures, not silent capture corruption. One preserved error record is `runs/r8_groot_l10_500_CT/debug/server_23175/decisions_2001113.jsonl:9414`; its legacy `_last_log` refers to the previous successful query while current anchor arrays are retained separately. Exception-phase and diagnostic-freshness tags would prevent misleading interpretation. I did not infer the bad probability value or select a numerical/segmentation threshold from these failures.

Some R8 gaps can be repaired through attested analysis sidecars: canonical library identity and action conventions can reference frozen source artifacts, and prefix-only stage tables can join existing pre-decision physics. Do not rewrite raw captures or silently invent missing values. Total historical capture overhead and restoration certification cannot be recovered from the current counters alone.

## 5. What R9 should do

1. **Tighten semantic admission before adding arms.** Check score/row/weight lengths, source-content IDs, required action semantics, semantic finite masks, diagnostic freshness, and metadata completeness. Publish separate capture-completeness and scientific-capability results. Preserve rejected attempts and reasons alongside the accepted population.
2. **Publish immutable augmentation generations.** A manifest should bind capture digest, exact accepted IDs, per-kind expected IDs, code/checkpoint/normalizer/fit hashes, resolved batch/stage mode, private noise domain, part hashes, and live-anchor agreement checks. Mark a generation complete atomically; replacements get a new generation ID. Tools record and pin that ID.
3. **Make research readers read-only by default.** Put optional caches under an explicit scratch/cache root, with content-based invalidation. Add `--split discovery|holdout|all`; fitting/threshold selection requires a frozen split manifest. Bind each report to tool source/config, capture, split, catalog, and augmentation generation hashes. Supply a shared prefix-only stage table with rule/fit hashes and `max_control_idx_used < control_idx_start`, including privileged-vs-deployable labeling. Namespace `stage_run`, mode, event, and physical labels separately; distinguish the labeler's fit population from its label coverage, retaining unknown mass.
4. **Measure actual capture overhead.** Record monotonic receive/start/finish/send timestamps, observer copy/tap/serialize/enqueue spans, client capture/compression spans, control deadlines, and final drain durations. Add final sender and receiver byte/frame/ACK/fsync/retry counters. Repeat debug off/on under matched production load for each dispatch family, including FL/IP/wrist/oracle paths. Keep action/RNG parity distinct from timing claims.
5. **Finish snapshot auditing before branching.** Publish component coverage and restore prerequisites; add policy context/history, method latch/budget/camera state, action queue/cursor, server RNG, and environment/controller state through explicit adapters. Certify a restored multi-step trajectory before using branch outcomes as call-value evidence.
6. **Port through explicit benchmark adapters.** `debug/server/manifest.py:model_manifest` currently defaults to two cameras/eight states/seven valid actions. In contrast, local `src/openpi/policies/robocasa_policy.py` requires three real π0.5 cameras and emits 12 actions; `metaworld_policy.py` uses one real camera and four actions. Require explicit camera/input-use masks and action/state schemas, preserving wire dtypes and shapes. RoboCasa365 needs fixture/articulation identity, goal subconditions, destination frames, and native reset/termination semantics; MetaWorld needs its own robot/controller/action/camera and success adapter. Declare units/frames/quaternion order, normalizer/output mapping and live checkpoint content IDs, control/physics rates, camera timestamps, entity roles/full ID maps, and separate robot/object/contact/predicate capabilities. Avoid LIBERO defaults or goal-name parsing. Real-robot adapters should mark unavailable object truth/contact/snapshots honestly.
7. **Use small porting admission runs first.** Require exact action/control joins, lossless inputs, deterministic initial-state identity where supported, capability-aware physical checks, and debug-off/on action parity. Re-measure field sizes and latency at native camera resolution; preserve full trajectories rather than dropping long or difficult episodes to meet storage limits.

No new segmentation or allocation policy is selected by this audit. R8 has supplied the missing physical and action evidence; R9's debug priority is to make its semantic identity, temporal meaning, and publication state as dependable as its byte-level collection.
