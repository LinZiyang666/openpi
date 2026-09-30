# Coordinator PROFILE run: ordinary plugin arms on non-test B states

**Prepared only. No command that launches a receiver/server/worker/chain, uses
tether, or accesses a remote host was run by C4.** CPU preparation, emit checks,
eight representative prefit/reload checks and offline analyzers were run locally.

Use a fresh run root. All 44 ordinary plugin arms use the same two B-val states
per task, 20 episodes per cell/variant. There is no P3 server method or deployment
policy shadow. The existing P3 v2 client supplies deterministic environment seed
1703, reset attestation and applied-control logs; snapshots are disabled (`P=0`)
because this phase needs ordinary allocation/cost telemetry. Streaming retains
the existing dispatch fence and avoids oversized whole-arm telemetry archives.

The private `chain_profile.sh` is derived from current `p3_profiling/chain_p3.sh`:
only the B-val launcher, helper affinity and stream-start guard change. It retains
manifest-hash DONE markers, accepted-attempt counts, server admission, ordinary
collection, streamed-client verification and failure behavior. A missing/unhealthy
receiver stops the chain. `launch_source.json` binds the inspected source hash.
The stock `chain.sh` alone cannot select B initial states: its remote launcher's
default `--apool-dir` points to the official measurement pool.

## 1. Render exact specs and manifests; stage dependency artifacts

From `/home/weiland/projects/openpi`; replace the example RUN with a fresh durable
run root before fitting. C4's rendering writes only its scratch; the coordinator
copies it to RUN. Specs carry exact final-path kwargs for every arm.

```bash
C4DIR=exp/offline_search/rounds/r07/c4_profile
C4MOD=exp.offline_search.rounds.r07.c4_profile
C4PY=(taskset -c 10-13,54-57 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
RUN=/home/weiland/trace_runs/os_closed_loop/r07_profile_bval1
PLAN=/tmp/r7_C4/rendered_profile_bval1

"${C4PY[@]}" -m "$C4MOD.prepare_profile" --call-specs exp/offline_search/rounds/r07/c3_calls/arms_eval500.json
"${C4PY[@]}" -m "$C4MOD.render_profile" --run-root "$RUN" --out "$PLAN"
mkdir -p "$RUN/fits" "$RUN/cal" "$RUN/stall" "$RUN/manifests"
cp "$PLAN/arms_profile.json" "$RUN/arms_profile.json"
cp "$PLAN/arms_eval500.json" "$RUN/arms_eval500.json"
cp "$PLAN/schedule_v2.json" "$RUN/schedule_v2.json"
cp "$PLAN/chain_profile.sh" "$RUN/chain_profile.sh"
cp "$PLAN/run_arm_bval.sh" "$RUN/run_arm_bval.sh"
cp "$PLAN/launch_source.json" "$RUN/launch_source.json"
cp "$PLAN/manifests/"* "$RUN/manifests/"
cp "$PLAN/prefit_profile.sh" "$RUN/prefit_profile.sh"
cp "$PLAN/prefit_eval500.sh" "$RUN/prefit_eval500.sh"
cp "$PLAN/arm_names_profile.txt" "$RUN/arm_names_profile.txt"
cp "$PLAN/pending_variants.json" "$RUN/pending_variants.json"

# C2: dependency banks, not its final-arm pickles with /tmp/r7_C2 kwargs.
for p in /tmp/r7_C2/fits/wrist_*.pkl /tmp/r7_C2/fits/stages_*.pkl; do
  cp "$p" "$RUN/fits/"
done
# C3: copy each complete calibration directory, including stages and R bank.
cp -a /tmp/r7_C3/cal/. "$RUN/cal/"
for cell in pi05_l10_50 pi05_spatial_50 groot_l10_50 groot_spatial_50; do
  short=${cell/spatial/sp}
  cp -a "/home/weiland/trace_runs/os_closed_loop/r06_c_cal/stall/$short" "$RUN/stall/$cell"
done
```

C3 already provides feasible rho=.30 fits for all four sparse cells. If those
artifacts are unavailable, the exact production recalibration commands are:

```bash
for cell in pi05_l10_50 pi05_spatial_50 groot_l10_50 groot_spatial_50; do
  "${C4PY[@]}" -m exp.offline_search.rounds.r07.c3_calls.calibrate --cell "$cell" --rho .30 --out "/tmp/r7_C3/cal/$cell"
done
```

Run that loop only into fresh C3 destinations: its fitter refuses an existing
fit log. C4 never ran it or wrote C3's paths. Prefer the delivered fits and
inspect their `feasibility.json` before copying. The copied stall directory's
contents, not its directory spelling, bind calibration compatibility.

## 2. Prefit every PROFILE arm against those final paths

```bash
# 44 exact lines: SF1/SF2/UF1 ×8, SW ×4, CU/CT(.30) ×4, paired A ×8.
# C4 prefit freezes A's existing retrieval rather than fitting a new representation.
bash "$RUN/prefit_profile.sh"
cp /tmp/r7_C4/prefits_profile/*.pkl /tmp/r7_C4/prefits_profile/*.json "$RUN/fits/"
"${C4PY[@]}" -m exp.offline_search.closed_loop.ops.emit_arms --run-root "$RUN" --spec "$RUN/arms_profile.json"
```

The generated `prefit_eval500.sh` has exact commands for 32 currently implemented
evaluation arms: SF1 ×8, UF1 ×8, SW ×4, SF+SW ×4, CU/CT ×4. Do not execute it
as an evaluation freeze: profile analysis must first prune variants and choose
the SF cap. For E=2 use C1's literal SF2 rows, then regenerate the selected spec
and prefit commands. C1/C3 explicitly defer SA to composition and a separate
cadence/camera budget re-solve; the four SA rows are pending, with no fabricated
method string or prefit command. [C1 composition contract](../c1_follow/COMPOSITION.md)
and [C3 composition contract](../c3_calls/COMPOSITION.md) state the dependency.

## 3. Deploy the existing P3 client bundle and the B wrapper/pools

This is coordinator-only remote work. Do not invoke shared `ops/sync_remote.sh`:
it replaces the remote shared launcher that the P3 installer protects. The
run-specific bundle contains the ordinary arm YAML/matrix files; the P3 client
does not require a P3 serving method. Its installer preserves the existing stock
launcher and runs the real remote Python 3.8 compatibility preflight.

```bash
P3DIR=exp/offline_search/rounds/r06/p3_profiling
"${C4PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.build_client_bundle --run-root "$RUN" --out "$RUN/client_bundle"
bash "$P3DIR/deploy_client.sh" "$RUN/client_bundle" > "$RUN/client_deploy.log" 2>&1

tether push --force /tmp/q1_method_c_fits/bval_pools.tar timan107:/tmp/r7_profile_bval_pools.tar
tether push --force "$RUN/manifests/bval_pool_l10.yaml" timan107:/tmp/r7_profile_bval_l10.yaml
tether push --force "$RUN/manifests/bval_pool_spatial.yaml" timan107:/tmp/r7_profile_bval_spatial.yaml
tether push --force "$RUN/run_arm_bval.sh" timan107:/tmp/r7_run_arm_bval.sh
tether exec timan107 -- bash -lc 'set -e; D=/scratch/zixuans8/openpi_trace/exp/common/data/db_init/libero; I=/scratch/zixuans8/openpi_trace/os_cl; mkdir -p "$D" "$I/r7_profile_bval"; tar -xf /tmp/r7_profile_bval_pools.tar -C "$D"; cp /tmp/r7_profile_bval_l10.yaml "$I/r7_profile_bval/bval_pool_l10.yaml"; cp /tmp/r7_profile_bval_spatial.yaml "$I/r7_profile_bval/bval_pool_spatial.yaml"; cp /tmp/r7_run_arm_bval.sh "$I/run_arm_r7_bval.sh"; sha256sum "$D"/libero_10/*.init "$D"/libero_spatial/*.init'
```

Compare the 20 printed file hashes to the staged pool YAMLs. No `.pruned_init`
may shadow either B directory. The wrapper accepts only `r7_*_profile`, requires
an exact manifest, and appends `--apool-record` and `--apool-dir` after the stock
defaults. Argparse's final value selects the hash-bound B pool; original B indices
are preserved. The driver still attests all 50 states/task. Numeric equality of
a B index and an official-test index is not state overlap.

## 4. Coordinator receiver, plumbing smoke, then remaining arms

Use coordinator-admitted free ports and CPU allocation. These values are required
inputs, not defaults chosen by C4. The CPU array affects preparation/receiver only;
it does not export an empty CUDA mask into serving processes.

```bash
: "${PORTS:?coordinator-admitted server port list}"
: "${P3_STREAM_PORT:?coordinator-admitted streaming receiver port}"
: "${SERVER_CPUS:?coordinator-admitted serving CPUs}"
export PORTS P3_STREAM_PORT SERVER_CPUS
mkdir -p "$RUN/state"
nohup "${C4PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.stream_receiver --run-root "$RUN" --port "$P3_STREAM_PORT" --ready-file "$RUN/state/receiver_ready.json" > "$RUN/receiver.log" 2>&1 &
echo "$!" > "$RUN/state/receiver.pid"
echo "$P3_STREAM_PORT" > "$RUN/state/P3_STREAM_PORT"
for i in $(seq 1 100); do test -f "$RUN/state/receiver_ready.json" && break; sleep .1; done
test -f "$RUN/state/receiver_ready.json"

# Eight ordinary arms, 20 B episodes each; both models and every follow variant.
WPS=2 STAGE1_ONLY=1 P3_PHASE=pilot \
  bash "$RUN/chain_profile.sh" "$RUN" r7_pi05_l10_50_A_profile r7_pi05_l10_50_SF1_profile r7_pi05_l10_50_SF2_profile r7_pi05_l10_50_UF1_profile r7_groot_l10_50_A_profile r7_groot_l10_50_SF1_profile r7_groot_l10_50_SF2_profile r7_groot_l10_50_UF1_profile

"${C4PY[@]}" -m "$C4MOD.profile_report" --run-root "$RUN" \
  --arms r7_pi05_l10_50_SF1_profile r7_pi05_l10_50_SF2_profile r7_pi05_l10_50_UF1_profile r7_groot_l10_50_SF1_profile r7_groot_l10_50_SF2_profile r7_groot_l10_50_UF1_profile \
  --references r7_pi05_l10_50_A_profile r7_groot_l10_50_A_profile \
  --keys "$C4DIR/telemetry_keys.json" --legacy-full-camera --out /tmp/r7_C4/PROFILE_smoke

```

Inspect 20 unique accepted pairs per smoke arm, exact
reset hashes, complete controls, extension provenance, fresh anchors after LOOK,
no policy/shadow calls on A/SF, and reconciled cost. Reusing the identical manifest
lets the chain skip those eight DONE arms. Its `full_model` rows automatically load
full weights for SW/CU/CT, irrespective of STAGE1_ONLY. These checks do not decide
SR. The chain uploads each exact per-arm manifest; no Cartesian test-init flags
are needed. Ordinary logs cannot be passed to P3 `read_v2`, whose server side
requires `p3_anchor/p3_decision` records; C4's analyzer reuses its control join.

Before any SW rollout, the coordinator must run C2's guarded real GPU parity
command. These are C2's allocated CPUs; no empty CUDA mask is applied. The
script admits one model only if at least 12288 MiB is free. A failure stops SW.

```bash
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c2_wrist.parity_pi05 --out /tmp/r7_C2/gpu_parity.json
```

Require all twelve recorded-observation comparisons to pass. Then satisfy the
[C2 hand-back's mixed-judge smoke](../c2_wrist/HANDBACK.md): a wrist-origin MISS
must log `camera_completion_calls=1` before policy output, and full-origin MISS
must log zero. The pure SW arms do not make policy calls, so running them alone
cannot satisfy this gate. C2 has not supplied a frozen mixed-judge smoke spec;
that prerequisite remains with C2/coordinator. A periodic judge forces a full
camera LOOK before its MISS and cannot substitute for that test. No unverified
smoke arm has been added to the 44-arm experiment.

Only after both camera gates and the eight plumbing smokes pass:

```bash
mapfile -t ARMS < "$RUN/arm_names_profile.txt"
WPS=2 STAGE1_ONLY=1 P3_PHASE=pilot bash "$RUN/chain_profile.sh" "$RUN" "${ARMS[@]}"
```

## 5. Paired final PROFILE report and decision gate

```bash
REFS=(r7_pi05_l10_50_A_profile r7_pi05_l10_500_A_profile r7_pi05_spatial_50_A_profile r7_pi05_spatial_500_A_profile r7_groot_l10_50_A_profile r7_groot_l10_500_A_profile r7_groot_spatial_50_A_profile r7_groot_spatial_500_A_profile r7_pi05_l10_50_CU30_profile r7_pi05_spatial_50_CU30_profile r7_groot_l10_50_CU30_profile r7_groot_spatial_50_CU30_profile)
"${C4PY[@]}" -m "$C4MOD.profile_report" --run-root "$RUN" --references "${REFS[@]}" \
  --keys "$C4DIR/telemetry_keys.json" --legacy-full-camera --reps 2000 --worst 3 --out /tmp/r7_C4/PROFILE_report
```

CU is the owner's simplified base C (uniform calls plus calibrated stall);
CT is paired with that exact CU arm. A references are rerun on these same B states,
not joined to historic test-init A outcomes. `--legacy-full-camera` applies only
to absent camera fields; SW's `camera_mode`, completion counters and `owner_cost`
always take precedence. If hand-backs rename extras, update the logical dotted
paths in a copied keys JSON. Missing actual camera/completion data cannot be
replaced by the method's next-camera forecast.

Apply SELECTION's PROFILE gates: parity, >=5% extension eligibility, >=.005 actual
IR saving against paired A, SW key/policy-input parity, and CU/CT IR in [.27,.33].
Report SR descriptively. Resume the explorers for keep/drop/cap recommendations,
freeze the evaluation list, then compose and recalibrate SA if retained. The B
starts were dense-library acquisition states; this is an allocation/plumbing
profile, not a held-out-bank generalization experiment. No test episode is
authorized by this launch document.
