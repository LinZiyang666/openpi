# R12 sol look-half implementation

The corrected pilot has five fresh local-serving B-dev roots with 17 arms. The prior eight Spatial-50/L10-50 arms are preserved and invalid as method engagement tests; retain them only as coordinator-designated extra baseline replicates. No chains, real fleet syncs, or test-A work were launched. The only live work is the bounded engagement smoke described below.

## Engagement fix (sol_fix.md)

The retained evidence does **not** reproduce the reported argv rewrite: all **32 startup records** from the eight completed arms have `--os-lookhalf` equal to the constructor and startup method. All **34,359 decisions** have `execution_audit=false`; none encodes wrist first. `FIX_FINDINGS.json` lists the exact source logs and per-arm counts. Camera planning correctly invalidated memory without a client acknowledgment, but the old pilot used the unchanged LIBERO worker, which never supplied those acknowledgments. Zero one-camera looks therefore did not test the intended acquisition methods.

`closed_loop/lookhalf_config.py` now validates exact flag values at emitted-arm planning, after relocation, in final server command construction, in startup parsing and against the loaded prefit. Missing/empty/unknown/duplicate values, missing constructor selection, mismatches, and misconfigured aliased R12 classes fail loudly before model serving. The planner also asserts method preservation across relocation; local fit relocation replaces only the exact artifact token, never a substring of an unrelated value. No method-token rewrite was found or claimed repaired.

For active methods, `control.chain` sets `OSCL_EXECUTION_AUDIT=1` and the B-attested driver selects `closed_loop.execution_worker`. This opt-in worker runs the unchanged LIBERO rollout through proxies that count **actual successful environment steps**, exclude initial settling, and send `executed_steps`, `exec_ok`, and sequential `decision_id` on each infer request. Partial/failed execution is reported as such and conservatively completes acquisition. An active R12 server refuses missing or malformed receipts **before any encoding**. Explicit off and non-R12 workers retain their existing path.

`closed_loop/test_r12_pipeline.py` reads actual freshly emitted arms, runs plan and sync with only external transport mocked, then checks the **entire final server argv**, not a pre-plan approximation. It covers every local arm (17 look-half + 15 guard), all seven method variants through remote dry sync, recording contracts/SHA values, real startup parsing, bad configuration, actual/partial execution counts, worker entry selection, and prefit mismatch. Content hashes are calculated from real assets and cached only while inode/size/mtime/ctime stay unchanged. Evidence: `fix_final_tests.log`, `fix_env_tests_final.log` and `FIX_VERIFICATION.json`: **127 CPU checks passed** (98 pipeline/regression checks, 2 coordinator environment checks, 14 look-half and 13 guard selftests).

Bounded live evidence is being finalized in `smokes/fix4/ENGAGEMENT.json`. `live_smoke.py` uses real pi05 weights and real LIBERO B-dev simulator prefixes, one server and one worker at a time, the same two held-out B inits for each method, at most 150 controls per episode and 25 minutes per method. Unchanged bf16 GPU vision / float32 CPU policy placement keeps physical GPU use below 4096 MiB; a 3100 MiB torch allocator cap leaves room for CUDA/EGL contexts. The parent checks `nvidia-smi`, monitors total physical memory and stops only its owned PIDs. These truncated engagement checks do not estimate SR or all-GPU numerical equivalence. The first attempt (`smokes/fix1`) stopped before inference because the smoke client had not loaded its bundle. The second (`smokes/fix2`) confirmed audited wrist-first requests, then was stopped by owned PID because this AVX2 CPU emulates bf16 policy GEMMs too slowly. The third attempt (`smokes/fix3`) exposed an attention-mask dtype mismatch in the smoke-only float32 adapter; no pilot source or fit changed. All attempts remain preserved; the final smoke uses the trained policy weights in float32 on CPU, with the genuine GPU encoder unchanged.

## Serving behavior

The opt-in class is `exp.offline_search.closed_loop.lookhalf:R12LookHalf`, with constructor `method=off|opus_m1|astra_consensus2|astra_strict`, `knob_method=off|periodic`, and the existing library/whole-episode subset arguments. Active periodic knobs also require the R11 target and library-only calibration. Pilots use knob off.

Use `--os-lookhalf METHOD` with matching constructor kwargs, `--os-blind --os-judge guard_only --os-no-shadow-native --os-tokens off`. The wrapper retains R10Recipe layers 1–3+5 and R11's off/periodic layer 4. Randomized verdicts, GPU retrieval, and external cap/burst/step-zero call overrides are refused on this transaction. Absent flags preserve the default dispatch and serialized action/log bytes. Explicit `off` delegates to unchanged R10/R11 and uses stock full stage 1.

`opus_m1` follows Opus SPEC §6: tau .10, margin 1, lag at most eight; planned full at step zero, a known periodic call, or after a policy-calling look. Other valid looks encode wrist first. W/WL stall or consistency rejection completes the missing camera before ordinary full retrieval. Accepted synthesis is WL with its donors/progress, corrected by the W head using W distance attenuation. Frozen lag fits are 2/4/6/8; an unexpected unsupported cadence completes conservatively rather than inventing a metric. Missing or failed exact five-control execution acknowledgments force full.

`astra_consensus2` uses the selected-library two-slot successor safety/state atlas, H proposal, corrected W witness, and corrected full-metric witness with weighted library camera innovations. Tau is .10; strict changes only tau to .05. It applies support .99, safe mass .75, state RMS .2, progress difference .5 and H advance greater than .5. Half never refreshes genuine-full donor memory. The valid prior-full-policy/tail exception requires both explicit execution acknowledgments and the accepted policy-tail source.

`NeedCamera` leaves guard, knob and anchor untouched. The plugin encodes the missing base camera once, refreshes live CP1 keys/history, and repeats the same query with the deployed full stack. An accepted half proposal enters the ordinary guard/knob once and installs its corrected authoritative tail. A half-to-policy call completes before stage 2 and cannot charge or execute a second completion. Failed requests invalidate camera memory and cached proposal state; connection clones keep lifecycle state independent.

π0.5 uses the genuine wrist-only Pi05Override path and the initial full look's validated masked-dummy embedding. GR00T calls `extract_feature` at image batch one for wrist and at image batch one for completion, updating the request-owned prefix. It retains the separately encoded wrist; it does not joint-reencode on fallback. Full looks use the original joint path. Both cameras are present for every policy call.

Relevant implementation: `closed_loop/lookhalf.py`, `lookhalf_math.py`, `lookhalf_cameras.py`, `lookhalf_numerics.py`, `lookhalf_ledger.py`, with narrow integration in `plugin.py`, `stage_overrides.py`, `look_recording.py` and `ops/collect.py`.

## Fits and evidence

`build.py` recomputes every added metric/head/scale/atlas from selected whole B-library episodes, including failures. It never imports explorer fitting modules or reads B-dev outcomes. The default preserves the attested R10 library-only full artifact; `--fresh-base` rebuilds that stack too. Read provenance and frozen post-freeze B-dev constants are in each `artifacts/CELL/build.json` and prefit `fit_info`. Added arrays are shared read-only across connection clones. Nothing is indexed by task beyond the existing partition/head convention.

`SELFTEST.json`: **14 CPU tests passed**. Across five cells, **600 decisions** compare `off` and forced-full active queries against R10: rows, scores, actions, confidence, extras and guard state agree exactly, and active duplicate queries are idempotent. Tests also cover hidden-current-camera poisoning, rejection without guard mutation, plugin completion before one guard commit, authoritative corrected tails, connection isolation, pure periodic peeking, successor support, ledger identities, B-only boundary refusal, and all emitted root contracts.

Astra's atlas safe/successor arrays match exactly in every cell. Maximum affine-metric difference is .00018311; H-head coefficient difference is at most .00001145; H scales differ by at most .00000382. W heads/scales are independently checked against the explorer's inert fits. Small floating-point differences arise from bounded PCA projection gather batches. Opus main metric fits match the reference arithmetic exactly on the same selected-library features; its W scale uses the unchanged R10 five-fold recipe.

Existing regressions: **25 recording tests + 8 R7 camera-dispatch tests passed**, including historical default action/decision/episode-log byte equality, per-request split/stack/relocation and idempotent live completion. Total: **47 CPU tests passed**.

Real-weight bounded encoder smokes checked `nvidia-smi` first and capped the allocator at 4096 MiB. π0.5 peak reserved memory was **874 MiB**; wrist/completion batches were `[1,1]` and the completed prefix matched stock exactly. GR00T peak was **2666 MiB**; batches were `[1,1]`, completion happened once and retained the wrist. A separate joint batch-two diagnostic gave token max-abs .796875, RMS .03253423, relative L2 .03935825. Reports: `ENCODER_SMOKE_pi05.json`, `ENCODER_SMOKE_groot.json`. These smokes use a selected B-library RGB row and synthetic text/state scaffolding with real vision weights; they establish encoder dispatch/completion behavior, not live policy equivalence or rollout accuracy.

## Costs and GR00T recording

Every accepted decision has `lookhalf_cost`, camera initial/completion/total counts, availability bits, acquisition plan and completion reason. New compute-share prices are π0.5 .076 per camera plus .848 for policy; GR00T .074 per camera plus .852 for policy. Full/completed policy calls cost 1. Blind reuse costs 0. `os_sw_*` and `owner_cost` carry the owner price on opt-in rows. R4 legacy fields and collector ledger remain available; the new ledger also gives explicit `legacy_work` using π0.5 R4 prices .055198/.049890 and proportional GR00T .074/.074. Those legacy comparison prices are distinct from the required compute-share basis.

The collector adds `lookhalf_cost_ledger` and `mixed.ir_compute_share` to the usual `runs/ARM/summary.json`. It retains accepted-attempt selection and reports encoded-camera counts, completion reasons, work per five controls, legacy comparison IR and first guard call per episode. CPU plan time is `lookhalf.plan_us`; query/search timing remains in the usual fields.

GR00T roots set `--os-groot-path-check-every 20`: first real look and then every twentieth real look per episode. The B-only writer saves exact encoder-input values as float32 (exact for bf16) plus original dtype, camera order, input ids/masks, joint/single CP1 keys, identical explicit noise, joint/separately completed policy chunks, and max-abs/RMS/relative-L2 deltas in hashed NPZ shards. Inputs come from the same normalized observation used by stage 1; GR00T `prepare_input` only wraps and casts these tensors. The 56-pixel wire thumbnails are not diagnostic encoder inputs. Diagnostic work does not enter serving IR, and forked RNG preserves production draws. The CPU recorder test detects deliberately different paths and verifies identical noise plus unchanged RNG. Live policy deltas remain pending until pilots run; no numerical-equivalence claim is made.

After pilots, `report.py ROOT` verifies B shard hashes and emits `ROOT/numerical_paths.json`. Reports are already emitted with explicit pending status and zero live samples.

## Pilot roots

Fresh roots are under `exp/offline_search/rounds/r12/lookhalf/pilots/` with prefix **`r12_lookhalf_fix1_`**. Old `r12_lookhalf_*` roots and results remain untouched. `pilot_roots.json` and each `ROOT.json` bind source selections, fit hashes, B contracts and arm lists. Each arm has **100 episodes, K=10 per task**, disjoint from its served subset; selections are exact byte copies from `r11_dev_k10_size50` or `size200`. Policy-tail reuse is one block, look recording is on, and 56-pixel wire thumbnails are on. Strict appears only on the two π0.5 size-50 cells.

| Root suffix | Arms |
| --- | --- |
| `r12_lookhalf_fix1_pi05_l10_50` | off, opus_m1, astra_consensus2, astra_strict |
| `r12_lookhalf_fix1_pi05_spatial_50` | off, opus_m1, astra_consensus2, astra_strict |
| `r12_lookhalf_fix1_pi05_spatial_200` | off, opus_m1, astra_consensus2 |
| `r12_lookhalf_fix1_groot_spatial_50` | off, opus_m1, astra_consensus2 |
| `r12_lookhalf_fix1_groot_l10_50` | off, opus_m1, astra_consensus2 |

## Exact coordinator commands

Commands below are handed back, not executed here. Install the updated audited worker on the selected idle island using `setup --worker-only` below; an older worker is now refused by active serving. `sync` prepares SHA-bound recording contracts and B configurations; `chain` issues the corresponding stamped B tasks and collects each arm. The supplied script explicitly lists every root and all 17 arms, runs roots sequentially, uses local serving, and defaults to timan107 with **four servers × WPS 16**. `LOCAL_SERVER_COUNT` and `PORTS` are read from the environment; unset ports are derived starting at 23110. The script checks port count and preserves ports 23100–23103. Port ownership and fleet locks remain enforced by the existing controller.

```bash
cd /home/weiland/projects/openpi
export SERVER_HOST=local WORKER_HOST="${WORKER_HOST:-timan107}"
export LOCAL_SERVER_COUNT="${LOCAL_SERVER_COUNT:-4}" WPS="${WPS:-16}"
# PORTS from the environment is preserved; if unset the script derives
# 23110,23111,23112,23113 for the default count.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LOCAL_SERVER_CPUS=22-37,66-81 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src
unset OSCL_MANIFEST OSCL_TASKS OSCL_EPISODES OSCL_INIT_POOL

# Required once on an idle worker island to install the audited worker.
taskset -c 22-37,66-81 env CUDA_VISIBLE_DEVICES= .venv/bin/python -B \
  -m exp.offline_search.closed_loop.ops.h100.control setup --worker-only

bash exp/offline_search/rounds/r12/lookhalf/coordinator_commands.sh plan
bash exp/offline_search/rounds/r12/lookhalf/coordinator_commands.sh sync
bash exp/offline_search/rounds/r12/lookhalf/coordinator_commands.sh chain
bash exp/offline_search/rounds/r12/lookhalf/coordinator_commands.sh collect
bash exp/offline_search/rounds/r12/lookhalf/coordinator_commands.sh report
```

`chain` already collects; the separate collect phase repeats collection only if needed. Choose an idle owned worker fleet and free forwarded ports before coordinator launch. No test-A command is included.

CPU reproduction and library rebuild (no launches):

```bash
cd /home/weiland/projects/openpi
P=(taskset -c 22-37,66-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 /home/weiland/projects/openpi/.venv/bin/python -B)
"${P[@]}" -m exp.offline_search.rounds.r12.lookhalf.selftest
"${P[@]}" -m pytest -q -p no:cacheprovider \
  --basetemp=exp/offline_search/rounds/r12/lookhalf/scratch/pytest_coordinator_new \
  exp/offline_search/closed_loop/test_look_recording.py
"${P[@]}" -m unittest exp.offline_search.rounds.r07.c2_wrist.test_dispatch
"${P[@]}" -m exp.offline_search.rounds.r12.lookhalf.build
# Optional: rebuild the full library-only R10 stack as well.
"${P[@]}" -m exp.offline_search.rounds.r12.lookhalf.build --fresh-base
```

A rebuild changes artifact bytes/timings; prepare a fresh destination before changing live roots. `prepare.py` intentionally refuses to overwrite existing pilot roots. Current frozen artifacts and roots require no rebuild for the commands above.

Fresh-root regeneration (intentionally refuses existing destinations):

```bash
"${P[@]}" -m exp.offline_search.rounds.r12.lookhalf.prepare --prefix r12_lookhalf_fix1
"${P[@]}" -m pytest -q -p no:cacheprovider \
  --basetemp=exp/offline_search/rounds/r12/lookhalf/scratch/pytest_fix_coordinator \
  exp/offline_search/closed_loop/test_r12_pipeline.py
# Optional repetition of ONLY the authorized bounded smoke, on an idle GPU:
taskset -c 22-37,66-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python -B -m exp.offline_search.rounds.r12.lookhalf.live_smoke \
  --out exp/offline_search/rounds/r12/lookhalf/smokes/coordinator_fresh
```

Use a new prefix/output when repeating either emission or smoke. The exact fresh root names are the table entries and `pilot_roots.json`; no old root is reused by the coordinator script.
