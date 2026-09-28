# Q2 hand-back: GR00T CycleTail with bounded chunk tails

Final verification UTC: 2026-09-28T02:34:29.833431+00:00.

Implemented and atomically installed the five owned shared files below. CycleTail, six arm specs, four exact prefits, CPU wire/lifecycle tests and eight-connection parity are complete. The full K10-installed regression recipe (K2/K1/K4/K5/K6/K7/K10) passed against the final shared files. No shared source changed after installation. No server, port, GPU workload, LIBERO worker, closed-loop chain, remote command, git command, subagent or review-test read was used. All Python work used repository `.venv/bin/python`, CPUs `30-33,74-77`, BLAS/OMP threads 1 and CUDA hidden. The store was read-only.

## Installed files and new deliverables

| Shared file | Installed UTC | SHA256 |
|---|---|---|
| blind.py | 2026-09-28T01:36:42.365914+00:00 | `2494f2cfacaea730246754c5d40a910b663369be2c0e1ecb5becf06ef05e021c` |
| plugin.py | 2026-09-28T01:36:42.370055+00:00 | `f9e65a39d834fa6dabda7025075cb9d58e4503443d0ab29125121b61f7b31e8b` |
| verify_logs.py | 2026-09-28T01:36:42.375170+00:00 | `09ec291ca3aa3e24a69810d1906272fc02fa4291aa322b40b4e6676611fc256f` |
| selftest.py | 2026-09-28T01:36:42.378324+00:00 | `3f5bced7261ce8e8c0ce67a25361dde7b7747152e4c3aea5730fafe63cd29183` |
| replay_client.py | 2026-09-28T01:36:42.381707+00:00 | `fb4af10ea5a5d65eaaf8aa48a83f9eb80943f46037b05172966c9c2d23320a99` |

`results/install.json` records preimage hashes, byte sizes and the five atomic rename times. Installation parsed each candidate, checked every shared preimage, wrote/fsynced a same-directory temporary file, used one `os.replace` per shared file (dependency `blind.py` first), and fsynced the directory. `before/` and `dev/` retain the preimages and installed candidates. K1, K7, K10 method files, `src/`, `harness/`, `profile/`, `ops/`, and other owners' files were not modified.

New deliverables under this directory: `judge.py`, `__init__.py`, `arms_q2.json`, `prefit_commands.json`, `prefit.sh`, `prepare_coordinator.py`, `make_arms.py`, `new_tests.py`, `cpu_transforms.py`, `contract_tests.py`, `concurrency_test.py`, `make_concurrency.py`, `tail_parity.py`, `run_final.py`, `validate_arms.py`, `summarize.py`, `write_handback.py`, `REPRODUCE.md`, preparation/install helpers, relocated original regression recipes in `regression/`, and evidence in `results/`. `results/deliverable_hashes.json` gives final SHA256 and UTC modification times of primary new source/spec/documentation files, including this hand-back; `results/file_inventory.json` covers all added files except itself and active report stdout logs. The unchanged original K10 assertions run from relocated copies; original round-4 files were not edited.

## Switches and semantics

Method: `exp.offline_search.rounds.r05.q2_groot.judge:CycleTail`.

G10 kwargs at 50 episodes: `{"lib":"current","kref":5,"cycle_k":4,"tail_blocks":1}`; at 500: `{"lib":"big","kref":8,"cycle_k":4,"tail_blocks":1}`.

Required serving switches: `--os-blind --os-policy-tail --os-policy-tail-blocks 1 --os-judge guard_only --os-no-shadow-native`, with a full-model server and a five-control client. G15 uses method `tail_blocks=2` and `--os-policy-tail-blocks 2`. G15 is implemented/tested, but is not added to the requested six-arm pilot. `--os-policy-tail-blocks` accepts only 1/2, requires `--os-policy-tail`, and refuses two blocks for π0.5's H10. `--os-policy-tail` alone retains its one-tail default and the observed exact π0.5 log bytes. With flags absent all six observed legacy log modes remain byte-identical.

CycleTail subclasses K1 BlindAWM and fixes `serving="anchor_tail", gates="budget_only", budget=tail_blocks`. Real visual anchors are numbered from zero per internal episode: anchors 0, 4, 8, ... force a MISS, others HIT. Blind requests never advance that counter. Both sources execute their own original chunk: head 0–4, tail 5–9, and optionally 10–14. Offset 15 cannot provide a complete five-control response. After the configured tail count, real vision is required. K1 already implements both H16 cache-tail offsets; no K1 edit or new cache-anchor mechanism was needed.

There is no MixedJudge, progress, terminal or stuck guard in CycleTail. Blind rows retain NaN visual-key histories; no visual observation is fabricated. GR00T normalized gripper <0 means closed. The existing GR00T wire adapter converts openness to LIBERO positive-close exactly once. The method only adds a correctly signed previous-gripper diagnostic and never alters action signs. It uses K4's `seed_process()` hook on real queries, including after unpickling. This seeds the serving process; request ordering can still change stochastic model sampling.

The plugin retains copies of normalized and already transformed wire policy chunks with per-connection internal episode identity, original step and five-control cursor. Valid length is H minus that cursor. A second policy tail uses the original MISS chunk at offset 10, even though the preceding request was itself a HIT. It verifies the method's full normalized H×32 chunk against the exact original slice/padding, then returns the original wire slice/padding without reapplying transforms. Cache tails use the existing GR00T CPU adapter. All tail selection/capture/consumption, method state, histories and lifecycle calls remain under the K6 connection lock. No runtime lock surrounds method work or policy inference; the pre-existing GR00T model lock is unchanged.

Step 0, reset (even with repeated external UID), task change, stale episode snapshots, non-five execution audits, invalid state, missing/invalid method output, exhausted tails, burst/cap, or globally due periodic MISS require vision. Duplicate last request IDs are rejected before reservation/commit and preserve the cursor; an admitted failure invalidates it. GR00T mixed implicit task changes may reset the request ID to zero. K6's duplicate contract remains an immediate-last-ID check, not a replay cache of arbitrary historical IDs. A MISS at the last episode decision cannot leak its tail into the next episode.

An absent `__extra__.executed_steps` means the configured five-control protocol, as in K10. The server cannot infer unreported actual execution counts. Keep mixed arms at client L5; only the library-independent pure-policy controls use client L10. Terminal partial chunks end the episode and have no continuation.

## Logging and cost

Policy-tail rows have `src=source="policy_tail"`, `vision=false`, `hit=true`, null stage timings/`miss_k`, and no search/shadow. The normalized full chunk, original wire actions and audit fields are in `--os-log-inputs` NPZs. GR00T/explicit-block startup and NPZ metadata add `policy_tail_blocks`. CycleTail extras identify source anchor step, tail offset and remaining valid rows; policy-tail rows' library rows/weights are proposal provenance, not their action source. `verify_logs.py`, the fake selftest and wire replay verifier now resolve the original anchor across either tail length.

The owner cost is `(0.152*vision_decisions + 0.848*MISSes) / five_control_slots`, with every MISS fully priced (GR00T K8 is a full policy call, not 8/10 of a π0.5 call). Blind policy/cache tails cost zero. K4's existing explicit vision/hit ledger independently reports zero vision, zero MISS and zero total cost for every tested policy-tail-only set. K4's automatic GR00T measured stage table is a separate basis; do not label it as the owner .152/.848 basis. L10 pure-policy controls use two five-control slots per request. No closed-loop SR, realized rollout IR, GPU transform parity or model throughput claim is made here.

## Final verification

| Check | Configurations/checks | Decisions/comparisons | Result |
|---|---|---|---|
| K2 original plugin matrix | 21 | 963 | PASS |
| K1 original plugin matrix | 10 | 480 | PASS |
| K4 original plugin matrix | 4 | 156 | PASS |
| K5 real randomized replays | 4 | 1368 | PASS |
| K5 overlay | 1 | 104 | PASS |
| K6 concurrency | 12 | 2196 | PASS, plus serialized side |
| K6 edges | 1 | 23 | PASS |
| K7 plugin arms | 5 | 240 | PASS |
| K7 scalar comparisons | 4 | 15272 | PASS |
| K7 full-cell count parity | 4 | 189904 | PASS |
| K10 plugin matrix | 9 | 432 | PASS |
| K10 concurrency | 9 | 1647 | PASS, plus serialized side |
| K10 original transform/lifecycle/ledger | 20 | 114 | PASS |
| Q2 GR00T selftests | 8 | 384 | PASS |
| Q2 lifecycle/wire integration | 8 | 420 | PASS |
| Q2 concurrency | 8 | 1464 | PASS, plus serialized side |
| Q2 invalid options/horizon contracts | 23 | — | PASS |
| Six arms / four exact prefits | 6 | 4 | PASS |

Q2's eight lifecycle runs contain 128 checks, 420 decisions and 68 policy-tail ledger rows, all at zero charged tail cost. Its fake selftests contain 384 decisions, 224 blind decisions and 64 policy tails. The original K10 edge run contains 20 checks, 114 decisions, 42 policy tails and 400 exact L10 control comparisons. The original K10 method test contains 72 vision comparisons, 66 blind comparisons, six vetoes and 21 lifecycle checks. K10's concurrency matrix contains 340 policy tails per side; its nine plugin selftests contain 118 policy tails. K7 has 13 synthetic edge checks and 150 blind-gap queries.

K7 edges and the K5 planted/null estimator, forged-log, incomplete/cross-scale, export and cost-ledger checks also passed through the original recipes. K10's inherited HIT-path/method lifecycle comparisons passed unchanged. K7's last two plugin arms reused its exact existing prefits after method/kwargs/cell validation (paths/hashes in `results/k7_fit_reuse.json`); independent replay fitting remained unchanged. The final structured counters are in `results/final_summary.json`; exact Q2 subprocess commands and return codes are in `results/final_commands.json`. `regression/run_installed.sh` and its constituent recipes give every legacy command. See `REPRODUCE.md` for the exact top-level commands and fresh-output requirements.

Byte comparisons freeze time/PID and use identical invocation/output paths, including startup records:

| Mode | Bytes | JSONL rows |
|---|---|---|
| pure | 63350 | 45 |
| mixed | 91774 | 45 |
| r4 | 145755 | 45 |
| blind | 163447 | 54 |
| rand1 | 744786 | 212 |
| rand2 | 744003 | 212 |
| guard_only | 208289 | 54 |
| threshold:inf | 248517 | 54 |
| periodic:5 | 220101 | 54 |

Q2 eight-connection parity (each threaded run replayed serially in a fresh process using its observed reservation order):

| Cell / scale / executed controls | Decisions per side | Vision | MISS | Policy tails | Exact parity |
|---|---|---|---|---|---|
| spatial_50_G10 | 183 | 93 | 32 | 32 | True |
| spatial_50_G15 | 183 | 64 | 16 | 32 | True |
| spatial_500_G10 | 183 | 93 | 32 | 32 | True |
| spatial_500_G15 | 183 | 64 | 16 | 32 | True |
| l10_50_G10 | 183 | 93 | 32 | 32 | True |
| l10_50_G15 | 183 | 64 | 16 | 32 | True |
| l10_500_G10 | 183 | 93 | 32 | 32 | True |
| l10_500_G15 | 183 | 64 | 16 | 32 | True |

Every compared non-timing decision row, returned action, verdict, dense history, mutable anchor and cursor matched. All new concurrency configurations reached eight overlapping fake stage-1 calls. These are fixed-input CPU correctness checks; timings are not a live GR00T throughput forecast.

Wire tests use 256 evenly spaced actual full inference chunks from each of `queries/groot_spatial_inf/a_inf.npy` and `queries/groot_l10_inf/a_inf.npy`: 512 unique stored chunks. Each compares exact action bytes against both L10 and L15 deque consumption. The eight suite/scale/tail tests repeat these same samples, totaling 51200 control comparisons; repetitions are not independent evidence. They use actual GR00T inverse action components and `/data/ckpt/n15_libero_{spatial,10}/experiment_cfg/metadata.json`, plus the installed production output adapter. The production CPU state adapter's masked float32/bfloat16 casts and changing-mask rejection also pass. The installed K10 test separately retains actual π0.5 output-transform and state-dependent transform checks.

Initial full GR00T transform imports under repository Python failed first on missing pytorch3d, then on a transformers VideoInput incompatibility. The final test path loads the actual action inverse components via the installed GR00T dependency paths and omits its identity model inverse and unrelated image transforms; no VLM processor or model is constructed. Two new fixture errors (required empty video concat order and required 256×256 image shape) were corrected before successful final runs. Failed development logs are preserved. Full image-transform/model execution remains unverified here.

## Arms, exact prefits and footprint

`arms_q2.json` is emit_arms format with literal `<RUN>` placeholders: four G10 arms at {spatial,l10}×{50,500}, plus `r5q2_g_spatial_policy_L10` and `r5q2_g_l10_policy_L10`. All are full model, cost_ledger=true, full ordinary MISS schedule, resize_size=256. Mixed clients replan every five controls; pure controls every ten. Pure controls use `pure_inference=true` and server_seed=5101. Mixed CycleTail uses the existing per-server `--os-seed` and seeds through its query hook; the emitter restricts the separate server_seed field to pure_inference.

Exact coordinator prefit commands, including all kwargs/flags and CPU/environment prefixes, are in `prefit_commands.json`. The following actual commands generated the four final artifacts, which were subsequently loaded by the four final G10 plugin selftests:

```bash
cd /home/weiland/projects/openpi
bash exp/offline_search/rounds/r05/q2_groot/prefit.sh
```

Expanded coordinator templates (replace literal `<RUN>` with the same run root):

```bash
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.closed_loop.plugin --os-method exp.offline_search.rounds.r05.q2_groot.judge:CycleTail --os-kwargs '{"lib":"current","kref":5,"cycle_k":4,"tail_blocks":1}' --os-cell groot_spatial_cache --os-log-dir '<RUN>/prefit_logs/r5q2_g_spatial_50_G10' --os-tag r5q2_g_spatial_50_G10 --os-root /home/weiland/trace_runs/offline_search_store --os-blind --os-policy-tail --os-policy-tail-blocks 1 --os-judge guard_only --os-no-shadow-native --os-fit-artifact '<RUN>/fits/r5q2_g_spatial_50_G10.pkl'
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.closed_loop.plugin --os-method exp.offline_search.rounds.r05.q2_groot.judge:CycleTail --os-kwargs '{"lib":"big","kref":8,"cycle_k":4,"tail_blocks":1}' --os-cell groot_spatial_cache --os-log-dir '<RUN>/prefit_logs/r5q2_g_spatial_500_G10' --os-tag r5q2_g_spatial_500_G10 --os-root /home/weiland/trace_runs/offline_search_store --os-blind --os-policy-tail --os-policy-tail-blocks 1 --os-judge guard_only --os-no-shadow-native --os-fit-artifact '<RUN>/fits/r5q2_g_spatial_500_G10.pkl'
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.closed_loop.plugin --os-method exp.offline_search.rounds.r05.q2_groot.judge:CycleTail --os-kwargs '{"lib":"current","kref":5,"cycle_k":4,"tail_blocks":1}' --os-cell groot_l10_cache --os-log-dir '<RUN>/prefit_logs/r5q2_g_l10_50_G10' --os-tag r5q2_g_l10_50_G10 --os-root /home/weiland/trace_runs/offline_search_store --os-blind --os-policy-tail --os-policy-tail-blocks 1 --os-judge guard_only --os-no-shadow-native --os-fit-artifact '<RUN>/fits/r5q2_g_l10_50_G10.pkl'
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.closed_loop.plugin --os-method exp.offline_search.rounds.r05.q2_groot.judge:CycleTail --os-kwargs '{"lib":"big","kref":8,"cycle_k":4,"tail_blocks":1}' --os-cell groot_l10_cache --os-log-dir '<RUN>/prefit_logs/r5q2_g_l10_500_G10' --os-tag r5q2_g_l10_500_G10 --os-root /home/weiland/trace_runs/offline_search_store --os-blind --os-policy-tail --os-policy-tail-blocks 1 --os-judge guard_only --os-no-shadow-native --os-fit-artifact '<RUN>/fits/r5q2_g_l10_500_G10.pkl'
```

| Artifact in /tmp/q2_fits/ | Episodes / rows | Library top-level NPY bytes | CycleTail pickle bytes | Native source pickle bytes |
|---|---|---|---|---|
| r5q2_g_spatial_50_G10.pkl | 50 / 1063 | 280896097 | 22842135 | 429351282 |
| r5q2_g_spatial_500_G10.pkl | 500 / 11751 | 3105168033 | 58604579 | 429351282 |
| r5q2_g_l10_50_G10.pkl | 50 / 2645 | 698934851 | 28135738 | 1068314575 |
| r5q2_g_l10_500_G10.pkl | 500 / 29631 | 7829904393 | 118431163 | 1068314575 |

Library NPY bytes above include top-level arrays (including raw full-resolution keys), excluding token subdirectories; recursive NPY byte totals and full native pickle paths are in `results/arms_validation.json`. CycleTail's compact representation is 586 bytes/entry, plus action/fixed arrays included in its actual pickle size. Each 50-library fit uses its own current library; no borrowed big-library fitting or outcome information is used.

| Artifact | SHA256 |
|---|---|
| r5q2_g_spatial_50_G10.pkl | `f5b1a273b76c108b7caa3b56c7e52c73c7e75e86f1701513fbc3898c024dce16` |
| r5q2_g_spatial_500_G10.pkl | `f298f9203383a58041fe2126774f8e434c85c5e26f577e5f4ebc867b569e2cb1` |
| r5q2_g_l10_50_G10.pkl | `97a3f8846baf2cfb957cc6c14366749a8f04e7c518eca6b252f0db5e15113fc9` |
| r5q2_g_l10_500_G10.pkl | `8baf158a5ac65901e1c1d614e9946b7a16dac625713e04072dd03879133aacdd` |

## Coordinator next step (unexecuted live smoke recipe)

Use the installed plugin on the next normal server start; existing live processes retain their imported code. Copy the four exact artifacts and emit all six specs with:

```bash
cd /home/weiland/projects/openpi
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r05/q2_groot/prepare_coordinator.py --run-root <RUN>
```

This preparation script validates exact method/kwargs/cell metadata, copies by temporary-file rename, and invokes emit_arms; it does not launch a server. Its local validation target was `/tmp/q2_coordinator_prepare`. Do not rename a BlindAWM or K10 pickle as CycleTail. If temporary fits are unavailable, expand each `prefit_commands.json` template with the intended `<RUN>` and fit again.

Coordinator only: run the ordinary full-model GR00T L10 controls and G10 arms on ten tasks × inits 0–9 in both suites and both library scales; retain five-control mixed clients and capture `--os-log-inputs` on a short instrumented subset. If using the existing pilot helper, explicitly set both `PILOT_TASKS=0,1,2,3,4,5,6,7,8,9` and `PILOT_EPISODES=0,1,2,3,4,5,6,7,8,9`: its defaults select five trap tasks. Use a separate pilot run root/name from the eventual 500-init run, since DONE markers otherwise skip the completed pilot arm. Confirm first-anchor MISS, subsequent offsets, actual MISS/vision counts and no stale response after reset; compare the L10 policy control before attributing an effect to cache control. Promote surviving G10 arms to the established 500 paired initializations. G15 is available for a later screen, not requested in this pilot spec. No smoke, rollout, remote sync, server or chain was run by Q2; live SR/IR and actual GR00T GPU behavior remain unverified.
