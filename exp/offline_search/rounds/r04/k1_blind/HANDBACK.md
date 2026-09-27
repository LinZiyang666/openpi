# K1 R4 handback

Completed verification: 2026-09-27T19:32:35.576869+00:00.

All implementation, tests, arm specs, and documentation are under `exp/offline_search/rounds/r04/k1_blind/`. No existing R1–R3, harness, src, profile, shared plugin, or shared operations file was edited. No GPU, server, LIBERO worker, network operation, git command, or closed-loop chain was run. The RAM store was absent; all real-store tests used `/home/weiland/trace_runs/offline_search_store`.

## Files and installation

Methods: `__init__.py`, `blind_awm.py`, `judge.py`, `control_step.py`, `wrist.py`. Each method was developed in `dev/` and copied to a sibling temporary file before one rename. The installed methods and development copies match.

```text
2026-09-27T19:06:46Z __init__.py blind_awm.py control_step.py judge.py atomically installed from dev/
2026-09-27T19:09Z wrist.py atomically installed from dev/ (minute precision)
2026-09-27T19:12:58Z blind_awm.py control_step.py atomically replaced (dataclass parity, action-repeat diagnostics)
2026-09-27T19:21:14Z control_step.py atomically replaced (analytic chord rounding, lower-offset ties)
2026-09-27T19:28:26Z judge.py atomically replaced (legacy noprog_n explicitly requests vision)
```

Handoff: `arms_r4.json`, `arm_plan.json`, `batch.json`, `prefit.sh`, `README.md`, `INTEGRATION.md`, `HANDBACK.md`. `make_arms.py` and `validate_arms.py` regenerate/validate these specs. `checks.py`, `run_checks.py`, `test_contract.py`, `test_variants.py`, `test_control_geometry.py`, `run_smokes.py`, `run_existing_checks.sh`, `run_plugin_blind.sh`, `measure_fits.py`, and `make_handback.py` reproduce verification. Detailed logs, metric JSON/NPZ files, fit measurements and command lists are in `results/`. Large pickles are outside the repo at `/tmp/k1_blind_fits/`.

## Switches and behavior

Module prefix: `exp.offline_search.rounds.r04.k1_blind.`

| Method | Exact switches |
|---|---|
| `blind_awm:BlindAWM` | `lib="current", kref=5` for 50, `lib="big", kref=8` for 500; `serving="phase_particles", "kernel_clock", "top1_clock", "anchor_tail"`; `budget=0..4`; `gates="all", "budget_only"`; `residual_threshold=.25, .5, 1` |
| `blind_awm:BlindAWM3` | Same blind switches plus the inherited AWM3 switches; ordinary query keeps AWM3 behavior |
| `judge:BlindMixedJudge` | `base_kwargs={...above...}`, `progress_guard="noprog_span"` or `"noprog_n"`, `memo_reset_after_miss=false or true`; ordinary MixedJudge kwargs including `noprog_n`, `guards`, `events`, `burst` pass through |
| `judge:MemoResetMixedJudge` | Defaults to `progress_guard="noprog_n", memo_reset_after_miss=true`; only progress comparisons reset after executed MISS |
| `control_step:ControlStepLibrary` | Same library kwargs; `ablation="G", "GS"`, `offsets=[0,1,2,3,4]` (coarse `[0,2,4]` optional); pure-cache only |
| `wrist:BlindWristAWM`, `wrist:BlindWristMixedJudge` | K3 wrist metric composed with the blind / gap guard adapter; the judge accepts `base_kwargs` |

Enable blind serving through K2 with `plugin_args: ["--os-blind", ...]`. Without that flag the plugin uses ordinary queries. The method captures every anchor row/weight in both `query()` and `os_synth()`, resets per episode, rejects first/after-MISS/invalid/task-change/discontinuous requests, and optionally supports immediate `invalidate_anchor()` feedback. Look codes 1–6 follow the brief; the span guard additionally requests code 8 (`noprog_span`). The unchanged `noprog_n` guard explicitly requests code 8 (`noprog_n_requires_vision`) on otherwise eligible blind requests because its adjacent-vision assumption is unchanged.

The phase kernel keeps 16 members and fixed weights; each advances only within its own episode using offsets h−1/h/h+1, penalty .05, monotone phase and at most two rows per blind decision. Gates use nominal clock advance. Tail eligibility is one blind block for H=10 and two for H=16; deterministic wire padding never extends eligibility. State std (floor .05) and normalized motion percentiles come only from the deployed library.

Intentional differences are explicit: after a blind gap, AWM’s unavailable adjacent-camera `still` diagnostic is omitted; retrieval/action/confidence are unchanged. The span judge uses dense normalized proprioception and elapsed anchor intervals, so its guards/confidence can differ from stock MixedJudge; `dense_motion_guard=1` labels that change. It corrects the terminal-closed sign for GR00T; the legacy passthrough retains stock semantics. G/GS intentionally change stale ranking and (GS) action alignment; their library-LOEO confidence scales are recalculated, so fresh/step-zero confidence can change even though selection/actions remain exact AWM. Wrist parity is against K3 WristAWM because deleting a camera intentionally changes the metric. No threshold or phase-rule change from A §1.1–§1.3 was needed. Control-step projection uses C’s analytic chord formula and lower-offset rounding on ties.

## Final verification

- **26,880 vision comparisons passed bit for bit**, covering all 120 combinations of serving × B=0..4 × gates × residual threshold on both regimes, both models/suites and both library scales. Compared top-k, scores, full action, confidence and extras. `results/parity_*.json`.
- **728 stock guard comparisons**, **728 memo-reset checks**, **728 span-guard action comparisons**, **120 AWM3 comparisons** (plain, ridge, gripper options), **96 wrist comparisons** passed. `results/variant_parity.json`.
- **60/60 harness smokes passed**, 6,246 decisions, two episodes each, with fresh-fit/reversed-episode determinism checks: BlindAWM and G/GS on all 16 cells/scales; span/stock/memo judges on π0.5 l10 inf/cache at both scales. All 32 G/GS smokes were repeated after final analytic-offset installation. Full err/AURC/gripper metrics and timing: `results/verification_summary.json`, `results/smokes.json`, `results/smokes/`.
- **86 edge checks** and four fake-driver sequences passed (π0.5/GR00T × 50/500): vision → blind → blind → vision → MISS → vision, anchors, lifecycle reasons `[6,1,6]`, dense commits, fresh policy-tail action equality, no-progress spans, memo reset, pickle/reset, tail limits, phase bounds and splice/tie behavior. `results/contract.json`.
- **731,376 store anchor/target windows**, **1,462,752 blind serving comparisons**, h=1/h=2 and phase/clock, passed against ideation A. Maximum mean-error difference **2.8e-10**, tolerance 2e-6. Saved full 16-member ideation anchors are the replay input; independent scalar query parity is tested separately. Their documented batched-anchor versus scalar maximum action reconstruction difference remains 0.000298366. Inf replay retains the ideation fixed-path hypothetical-HIT assumption; these are not counterfactual rollouts. `results/replay_*.json`.
- **10/10 installed plugin blind selftests passed**, 480 decisions, 216 blind, 72 MISS, 264 stage-1 calls, exactly 480 broadcasts. These use real CPU orchestrators, interleaved connections, four episodes/run, duplicate rejection and exact log replay. Mixed integration uses ncal=64 for its fake-driver calibration; the smoke/fit tests use default ncal=3000. `results/plugin_*/selftest_report.json` and `verify_blind.json`.
- **6/6 existing plugin selftests passed**, 235 decisions: old ProbeB0, B=0 adapter, and every-decision MISS/inf replay on both models. Top-k/scores/confidence/action/extras equality 100%. **4/4 existing G3 contract checks passed** (both models × both libraries), 100% kernel/confidence/action agreement. `results/selftest_*/selftest_report.json`, `results/g3_*.log`.
- **528 G/GS queries** checked: 280 step-zero/fresh queries retained AWM rows/scores/actions exactly; all 248 sampled stale queries selected nonzero offsets and changed the splice. G/GS rows and scores were identical. This verifies the intervention, not SR. `results/control_geometry.json`.
- **116 arm specs / 116 CacheConfigs validated**, including plugin argument parsing, MISS K=2 YAML and L=10 client overrides. `results/arms_validation.json`.

### Replay numbers (cache l10; normalized RMS action error)

| Model | Library | h=1 phase / clock | h=2 phase / clock | h=2 phase−clock |
|---|---:|---:|---:|---:|
| pi05 | 50 | 0.538429942 / 0.546822209 | 0.573280674 / 0.595741357 | -0.022460684 |
| pi05 | 500 | 0.460682752 / 0.472142538 | 0.503848633 / 0.530470837 | -0.026622204 |
| groot | 50 | 0.518537215 / 0.534360849 | 0.541339369 / 0.575305557 | -0.033966188 |
| groot | 500 | 0.450643620 / 0.474936469 | 0.492055526 / 0.529312311 | -0.037256785 |

## Measured fresh fits and deployed footprint

Protocol-4 plugin payloads, decimal MB. These are fresh fits, including serialization round trips; fit seconds exclude serialization and were measured while other assigned CPU tests ran. They are not throughput-under-load conclusions. Full detail and paths: `results/fits.json`. The same fitted arrays support serving/gate/budget variants; per-arm metadata must match exact kwargs. Compact retrieval bytes/entry: BlindAWM 586, span judge 632, G/GS 588, wrist blind 330, wrist span judge 376. Valid action payload is another 280 B/row for π0.5 or 448 B/row for GR00T; measured pickle bytes include padded chunks and all auxiliary arrays.

| Cell | Scale | Episodes / rows | Blind representation bytes | Blind pickle bytes (MB) | Fit s | Owner deployed pkl MB |
|---|---:|---:|---:|---:|---:|---:|
| groot_l10 | 50 | 50 / 2645 | 1,549,970 | 28,135,563 (28.136) | 0.227 | 1068 |
| groot_l10 | 500 | 500 / 29631 | 17,363,766 | 118,430,992 (118.431) | 10.417 | 1068 |
| groot_spatial | 50 | 50 / 1063 | 622,918 | 22,841,960 (22.842) | 0.127 | 429 |
| groot_spatial | 500 | 500 / 11751 | 6,886,086 | 58,604,408 (58.604) | 2.249 | 429 |
| pi05_l10 | 50 | 50 / 2640 | 1,547,040 | 26,091,313 (26.091) | 0.287 | 1103 |
| pi05_l10 | 500 | 500 / 29472 | 17,270,592 | 95,264,479 (95.264) | 10.997 | 1103 |
| pi05_spatial | 50 | 49 / 1018 | 596,548 | 21,909,555 (21.910) | 0.129 | 431 |
| pi05_spatial | 500 | 500 / 10909 | 6,392,674 | 47,408,961 (47.409) | 1.352 | 431 |

| Method | Cell | Scale | Pickle bytes (MB) | Fit s |
|---|---|---:|---:|---:|
| cslG | groot_l10 | 50 | 28,055,181 (28.055) | 0.619 |
| cslG | groot_l10 | 500 | 117,541,066 (117.541) | 26.824 |
| cslG | groot_spatial | 50 | 22,809,047 (22.809) | 0.212 |
| cslG | groot_spatial | 500 | 58,250,882 (58.251) | 6.253 |
| cslG | pi05_l10 | 50 | 26,011,081 (26.011) | 0.515 |
| cslG | pi05_l10 | 500 | 94,379,323 (94.379) | 26.802 |
| cslG | pi05_spatial | 50 | 21,877,992 (21.878) | 0.286 |
| cslG | pi05_spatial | 500 | 47,080,695 (47.081) | 4.070 |
| cslGS | groot_l10 | 50 | 28,055,183 (28.055) | 0.833 |
| cslGS | groot_l10 | 500 | 117,541,068 (117.541) | 30.909 |
| cslGS | groot_spatial | 50 | 22,809,049 (22.809) | 0.365 |
| cslGS | groot_spatial | 500 | 58,250,884 (58.251) | 9.310 |
| cslGS | pi05_l10 | 50 | 26,011,083 (26.011) | 0.735 |
| cslGS | pi05_l10 | 500 | 94,379,325 (94.379) | 29.968 |
| cslGS | pi05_spatial | 50 | 21,877,994 (21.878) | 0.295 |
| cslGS | pi05_spatial | 500 | 47,080,697 (47.081) | 4.857 |
| mixed | pi05_l10 | 50 | 32,704,278 (32.704) | 11.490 |
| mixed | pi05_l10 | 500 | 142,348,317 (142.348) | 48.607 |
| mixed | pi05_spatial | 50 | 26,076,273 (26.076) | 4.026 |
| mixed | pi05_spatial | 500 | 66,500,307 (66.500) | 25.440 |
| wrist | pi05_l10 | 50 | 14,613,531 (14.614) | 0.158 |
| wrist | pi05_l10 | 500 | 70,048,763 (70.049) | 9.702 |
| wrist | pi05_spatial | 50 | 11,262,378 (11.262) | 0.090 |
| wrist | pi05_spatial | 500 | 31,697,501 (31.698) | 1.221 |
| wrist_mixed | pi05_l10 | 50 | 21,226,723 (21.227) | 8.108 |
| wrist_mixed | pi05_l10 | 500 | 117,132,593 (117.133) | 38.871 |
| wrist_mixed | pi05_spatial | 50 | 15,429,125 (15.429) | 2.840 |
| wrist_mixed | pi05_spatial | 500 | 50,788,631 (50.789) | 16.409 |

## Exact reproduction commands

Run from `/home/weiland/projects/openpi`. All Python invocations (including scripts’ child processes) used the following prefix; scripts never exceeded eight Python processes in the assigned eight-CPU mask.

```bash
PY=(taskset -c 18-21,62-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /home/weiland/projects/openpi/.venv/bin/python)
K=exp.offline_search.rounds.r04.k1_blind
"${PY[@]}" -m "$K.checks" parity --key pi05_l10 --scale 50
"${PY[@]}" -m "$K.checks" replay --key pi05_l10 --scale 50
"${PY[@]}" -m "$K.run_checks"
"${PY[@]}" -m "$K.run_smokes"
"${PY[@]}" -m "$K.measure_fits"
# Final implementation checks after analytic chord installation:
"${PY[@]}" -m "$K.measure_fits" --only-control
"${PY[@]}" -m "$K.run_smokes" --only-control
"${PY[@]}" -m "$K.test_contract"
"${PY[@]}" -m "$K.test_variants"
"${PY[@]}" -m "$K.test_control_geometry"
bash exp/offline_search/rounds/r04/k1_blind/run_existing_checks.sh
bash exp/offline_search/rounds/r04/k1_blind/run_plugin_blind.sh
"${PY[@]}" -m "$K.make_arms"
"${PY[@]}" -m "$K.validate_arms"
"${PY[@]}" -m "$K.make_handback"
```

`results/smokes.json` records every exact smoke command; the two shell scripts contain every exact existing/G3/plugin test command. All runs were CPU-only. `results/verification_summary.json` contains final counters and all per-smoke metrics. The final method files are SHA-256 listed in `results/source_manifest.json`.

## Coordinator next steps and caveats

1. `arms_r4.json` is an ordered candidate set: 88 batch-three rows and 28 batch-four rows (116 total), with π0.5 l10 first, then spatial; GR00T rows are pure-cache phase/tail controls. `arm_plan.json` labels matched controls and conditional combinations. It includes B=0, phase B=1/2, clock B=1, gated/ungated tail, periodic k=5 at 50 and k=8 at 500, and policy L=10 controls. Batch four includes B=1/2 plus MISS K2, then dummy-camera caching/prefix packing, then wrist-only/prefix packing, plus both G and GS l10 arms at both scales. This grid makes alternatives reviewable; it does not claim all candidates should be run or that a winner is known.
2. Replace every `<RUN>` in the JSON with the chosen coordinator run directory, then use the installed K4 emitter. Mixed rows set `full_model:true`. K2 stacks use `yaml_patch.miss.num_steps=2`, arm-local `miss.evidence_dir`, and `write_policy.type=never`. Redundancy flags are `--os-stage1-mode dummy_cached --os-pack-prefix`; wrist rows use `--os-stage1-mode wrist_only --os-pack-prefix --os-tokens off`. All use `--os-no-shadow-native`. L=10 rows use `replan_steps:10` and `--os-judge periodic:1`; normalize their IR per five controls.
3. Prefit exact arm artifacts with `RUN=/path/to/run bash exp/offline_search/rounds/r04/k1_blind/prefit.sh`. Each generated command uses the CPU prefix above and `--os-fit-artifact "$RUN/fits/<arm>.pkl"`. The actual measured representative artifacts are in `/tmp/k1_blind_fits`; they are useful for review, but cannot be renamed blindly into per-arm artifacts because the plugin validates spec/kwargs/cell metadata. `batch.json` provides deduplicated offline method/cell rows.
4. Only the coordinator should start full-model servers, run stratified collapse screens / the 500 paired episodes, push remote scripts, or make SR/IR decisions. No closed-loop SR, GPU transform parity, prefix-packing numerical parity, cheap-vision savings, or hardware latency claim is made by K1. K3’s validated stage implementation and K2’s real-model transform/bypass acceptance remain prerequisites for those combined arms. The CPU plugin tests validate serving/history mechanics with recorded normalized state and actions.
5. Interpretation: phase versus clock isolates continuation at fixed anchor synthesis/library; top1 removes kernel mixing; gated versus ungated tail changes look control. G versus AWM changes index geometry and GS versus G changes aligned synthesis. 50 versus 500 changes the actual deployed library. No fit here borrows larger-library data for the 50-episode arm. The small π0.5 spatial library has 49 episodes. Static MixedJudge cannot judge spliced heads consistently, so G/GS explicitly refuse G3 mixed wrapping. Full online success/failure and control-cost effects remain unmeasured by this agent.
