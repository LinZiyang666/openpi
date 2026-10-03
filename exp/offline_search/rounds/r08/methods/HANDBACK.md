# R8 S5 handback — methods, arm emission, frozen prefits

## Delivered

Implemented all S5 variants in `methods.py`: `FollowLottery`, `WristEveryLook`
(five- and ten-control cadence), `EveryFiveAWM`, `ShiftedAWM`,
`IdentificationProbe`, `PolicyEveryTen`, and `OracleGraspCalls` (5a/5b).
Each new method exposes a detached, JSON-safe, read-only `debug_record()`.

- Lottery support is exactly R7 FollowExtension structural support for E=1/2,
  including zero-weight members. E is uniform over supported {0,1,2}; logs
  include support, propensities, coin/domain/seed, E, successor cursor, stage
  admission by E, and delta/absolute state residuals. Stage/valve never veto.
  `force_e=0` preserves A's Result, action, extras, rows, weights and cadence.
- Wrist uses the frozen independent R7 wrist metric and per-request camera
  path. Step zero and lifecycle/forced looks use full cameras. Stage admission
  is removed. W5 changes only the controller budget to zero; the metric is
  unchanged. Full-camera A5 reuses BlindAWM's existing budget=0 path.
- Shifted A shortens its first commitment to five controls, then resumes A's
  ten-control cadence. `shifted=False` is exactly A.
- IP uses p=.25 at every fresh anchor with no stall/cooldown; P10 uses p=1.
  Both preserve CU's lifecycle-only policy-tail hook, one five-control tail,
  and compute a live cache proposal from frozen A. `AnchorCalls(p=0)` is A.
- Oracle 5a calls at every eligible fresh anchor. 5b additionally requires
  distance <=.05 m and caps at two calls per goal-object ID per episode.
  Lifted, satisfied, or unknown-predicate objects are excluded. Missing truth
  yields a recorded error and no oracle call. These are privileged diagnostics.

`ops/arms_in.json` contains exactly **70 arms / 35,000 episodes**, ordered P0
then P1 then P2, GR00T L10 first: **32 / 34 / 4** arms. All request five-control
decisions and require `osdebug.v1`; stock environment seed is 7. The six oracle
specs carry `--os-oracle`; S2's config derives `OSDEBUG_ORACLE=1` from that flag.

CU/CT retain the frozen R7 budget solves: rho=.30 (50) and .18 (500), original
stall/cooldown rules, lambda and stage tilt. Their new shared paired coin seed
is **26093005**, distinct from R7's 26092903 and replicate-2 26092904; the
existing cell-specific R7 randomization keys are retained. Other variants are
fixed collection probes, not newly calibrated budget controllers.

`ops/r7_sources.json` pins all source fits, dependencies and official/non-test
manifests by SHA256. Unchanged A/SF1/SW use their source artifacts directly by
SHA. A is the identical frozen A artifact used by R7 (originally published in
R5). CU/CT get new serialized artifacts solely to change their seed/metadata.
All new controllers reuse library-only frozen retrieval/stage/wrist statistics;
no evaluation outcome is read for fitting or allocation.

**50 new fits** are complete under `/tmp/r8_S5/fits/` (3,138,840,789 bytes), plus
**20 SHA-reused fits**. `/tmp/r8_S5/prefit_report.json` records all 70 paths/hashes.
Normal plugin `pickle.load`, class/spec/kwargs/cell, frozen retrieval and wrist
fingerprints, call calibration digests, and current method-code SHA validate for
all 70 (`/tmp/r8_S5/artifact_validation.json`). Fits use the established plugin
pickle format; no debug data format is introduced.

## Files

- `methods/{__init__.py,methods.py,test_methods.py,test_library_replay.py,test_plugin_parity.py,HANDBACK.md}`.
- `ops/{__init__.py,emit_arms.py,prefit.py,validate_arms.py,gpu_commands.py,test_ops.py,r7_sources.json,arms_in.json}`.
- Scratch only: `/tmp/r8_S5/{fits/,prefit_report.json,artifact_validation.json,library_replay.json,summary.json}`;
  logs, emitted YAML/matrices/arms and non-test smoke configs are also below `/tmp/r8_S5/`.

No existing file outside S5's owned directories was edited. `/home/weiland/trace_runs`
was only read. No GPU, simulator, server, socket, worker, remote command, tmux,
process-management command or git-state operation was run.

## Run / integrate

From `/home/weiland/projects/openpi`, use this CPU prefix:

```bash
P=(taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
"${P[@]}" -m exp.offline_search.rounds.r08.ops.prefit
"${P[@]}" -m exp.offline_search.rounds.r08.ops.validate_arms
"${P[@]}" -m exp.offline_search.rounds.r08.ops.emit_arms --run-root /tmp/r8_S5/emitted --fit-root /tmp/r8_S5/fits --out /tmp/r8_S5/emitted_specs.json --emit-config
"${P[@]}" -m exp.offline_search.rounds.r08.ops.emit_arms --smoke --variants A FL W10 W5 IP P10 O5a O5b SHIFT A5 --run-root /tmp/r8_S5/smoke --fit-root /tmp/r8_S5/fits --out /tmp/r8_S5/smoke_specs.json --emit-config
```

`prefit` verifies/resumes existing fits; `--rebuild` is required after a method-code
change. `--cells` / `--variants` allow partial preparation. The emitter's optional
`--refresh-source-lock` rechecks published R7 sidecar hashes before repinning.
Emission/preparation write only to owned ops files or `/tmp/r8_S5/`.

The prepared full config is `/tmp/r8_S5/emitted/arms.json` with 70 YAMLs/matrices.
The prepared smoke has 42 arm specs / 840 non-test episodes, using the frozen R7
`NONTEST_BVAL` manifests and separate `_smoke` arm identities; its fits are the
same validated artifacts. Full arms use the SHA-pinned official R7 10x50 manifest.
The coordinator can copy new fits and emit input specs with a campaign fit-root;
the normal closed-loop emitter consumes those specs. S5 `--emit-config` also
preserves the `r8` priority/provenance metadata in its resulting `arms.json`.

Launch through **S2's `debug/ops/chain_debug.sh`**. It supplies the per-port
`--os-debug-dir` and capture config; ordinary chain execution does not enable
capture just because a spec declares `debug_required`.

S1/S2 integration needs no new hook: the oracle method reads **`q.oracle`**,
which S1 exposes only with `--os-oracle`, and accepts S2's `objects` payload with
`object_id,in_window,distance,lifted,satisfied,predicate_known`. The optional
`set_oracle(payload)` supports isolated CPU replays; `distance_m` is also accepted.

## Tests and results

```bash
"${P[@]}" -m pytest exp/offline_search/rounds/r08/methods exp/offline_search/rounds/r08/ops -p no:cacheprovider --basetemp /tmp/r8_S5/final_tests -q -s
"${P[@]}" -m unittest exp.offline_search.rounds.r08.methods.test_methods exp.offline_search.rounds.r08.ops.test_ops -v
```

- Full suite: **23 passed**, 91.66 s, one third-party `pynvml` deprecation warning.
  The subsequently added command-generator test is covered by the final unit run.
- Final unit run: **17 passed** (synthetic actual AWM metric/chains, identities,
  support/propensities/coin distribution, shadow gates, exact policy tails,
  oracle state/caps, camera forcing, SHA/manifest rejection, output bounds,
  command generation). Output: `/tmp/r8_S5/unit_tests.log`.
- Real libraries: all eight cells; **5,391 byte-exact identity comparisons**
  across E=0, p=0 and disabled shift; **1,797 query-only lottery anchors** with
  all E values represented in every cell. Actual cadence, policy/tail padding,
  IP p=.25, oracle cap, and all four wrist banks were exercised. Fixed recorded
  observations, one demonstration per task, at most 24 decisions per episode;
  no closed-loop SR claim. Output: `/tmp/r8_S5/library_replay.json`.
- Real `_ConnPolicy`/session/observer with CPU fake encoder/policy: FL, W10,
  W5, IP, P10 and O5b, **220 requests each**, debug off/on. Response bytes,
  method/history digests including R8 state, RNG states and legacy records
  identical; writer drains without errors. Uses S1's parity harness on
  pi05-Spatial-50. Reports below `/tmp/r8_S5/final_tests/`.
- All **70** artifacts validate through the plugin loader and unchanged frozen
  retrievals. 70 full YAML/matrix configs and 42 non-test smoke configs emitted.
- New Python files parse using Python 3.8 grammar. Server dependencies are
  existing serving dependencies; no new code is installed in the Python 3.8 client.

## Coordinator GPU / remote admission commands (not executed by S5)

Print exact quoted full-model server and recorded-image client commands for
debug off/on; the generator itself only prints, and uses ports outside 23100–23199:

```bash
"${P[@]}" -m exp.offline_search.rounds.r08.ops.gpu_commands --arm r8_pi05_l10_P10_smoke --port 23240 --log-root /tmp/r8_S5/gpu/pi05_off --mode off
"${P[@]}" -m exp.offline_search.rounds.r08.ops.gpu_commands --arm r8_pi05_l10_P10_smoke --port 23240 --log-root /tmp/r8_S5/gpu/pi05_on --mode on
"${P[@]}" -m exp.offline_search.rounds.r08.ops.gpu_commands --arm r8_groot_l10_P10_smoke --port 23241 --log-root /tmp/r8_S5/gpu/groot_off --mode off
"${P[@]}" -m exp.offline_search.rounds.r08.ops.gpu_commands --arm r8_groot_l10_P10_smoke --port 23241 --log-root /tmp/r8_S5/gpu/groot_on --mode on
```

Coordinator: run one generated server at a time and its printed replay command
in another terminal, restarting with the same seed/batch schedule for off/on.
Repeat with FL/IP and pi05 W10/W5/A5 and GR00T SHIFT. Compare per-episode input
archives' `wire_actions,a_exec,hit,has_vision,topk` and method records, in addition
to the replay report's wire-versus-log/tail checks. Replay timing fields are not
expected to match on real hardware. For stochastic policy parity use S1's
supplied-noise/controlled-schedule validation; a fresh stochastic sample or a
changed batch schedule is not a byte-parity certificate. Recorded-image replay
has no privileged simulator channel; test actual oracle call coverage with the
debug client smoke below.

After S2 deployment, coordinator-only real GPU + remote LIBERO smoke:

```bash
PORTS=23240 WPS=2 SERVER_CPUS=30-33,74-77 OPS_CPUS=30-33,74-77 bash exp/offline_search/debug/ops/chain_debug.sh /tmp/r8_S5/smoke r8_groot_l10_50_A_smoke r8_groot_l10_50_FL_smoke r8_groot_l10_P10_smoke r8_groot_l10_50_IP_smoke r8_groot_l10_50_O5a_smoke r8_groot_l10_50_O5b_smoke r8_groot_spatial_50_O5b_smoke
PORTS=23240 WPS=2 SERVER_CPUS=30-33,74-77 OPS_CPUS=30-33,74-77 bash exp/offline_search/debug/ops/chain_debug.sh /tmp/r8_S5/smoke r8_pi05_l10_50_A_smoke r8_pi05_l10_50_FL_smoke r8_pi05_l10_P10_smoke r8_pi05_l10_50_W10_smoke r8_pi05_l10_50_W5_smoke r8_pi05_l10_50_O5a_smoke r8_pi05_l10_50_O5b_smoke r8_pi05_spatial_50_O5b_smoke
```

Require complete capture/receipts, first/full/forced-camera behavior, one policy
tail per call, source/call/coin ledger agreement, and oracle caps before eval500.
Then complete SELECTION §3's physical parity, capacity and deferred augmentation
admission with S1/S2/S3. Real GPU numerics, real simulator behavior and transport
were not validated by S5; those are the coordinator's remaining integration checks.

## Fix round 1

Read `REVIEW_1.md` in full. Fixed **m5, method side** in `OracleGraspCalls`:
top-level `available` and `partial` payloads can supply usable objects. Each
object must have status `available` (omitted status remains compatible with old
resolved records). `unsupported`/`error` objects never become candidates, even
if their remaining fields look eligible. Their ID, status and reason remain in
`diag.oracle_objects`; malformed resolved entries have an explicit object error.
Top-level `unsupported`/`error` still prohibit every oracle call. Diagnostics
preserve the supplied aggregate status. Tight distance/window/lift/goal rules
and the two-call limit per resolved object per episode are unchanged.

**S2 contract verified:** the current `MujocoAdapter.oracle()` emits
`status="partial"`, with `objects` containing `status="available"` resolved
entries and `status="unsupported"` unresolved entries plus reasons. The new
producer/consumer regression constructs a fake environment with one mapped mug
and one unmapped goal subject, feeds the adapter's payload directly to the
method, and checks that only the mug receives a call. No adapter file was edited.
Legacy resolved objects, `distance`/`distance_m`, the live `q.oracle` facade and
the isolated `set_oracle` path remain covered.

Owned parity consumers now glob `decisions*.jsonl` and `writer_stats*.json`.
They skip and report an unterminated final journal line, including a syntactically
valid JSON object without its terminating newline. Fixtures cover both legacy
single-file and multiple process files. Read-only inspection of the **plain**
current smoke directory found 1,380 readable server decisions, one legacy stats
file and one legacy metadata file, with no unterminated final lines at that
instant (`/tmp/r8_S5/fix1_smoke_read.json`). This is a file-reader check, not a
closed-loop completeness or oracle-physics certificate.

Changed files: `methods.py`, `test_methods.py`, `test_library_replay.py`,
`test_plugin_parity.py`, and this handback, all under S5's methods directory.
The coordinator's `R8_EMIT_ROOT` emitter change was preserved. No other owner's
file, campaign data, source artifact or git state was changed.

All **50 new prefits** were rebuilt in `/tmp/r8_S5/fits/` to refresh the method
code SHA; the **20 unchanged fits** still reuse their frozen source hashes.
Updated artifact paths/hashes are in `/tmp/r8_S5/prefit_report.json`; all **70**
pass `/tmp/r8_S5/fix1_artifact_validation.json`. The coordinator should copy the
refreshed new fits before launching oracle arms. Preparation uses only frozen
library statistics and reads no evaluation outcomes.

Commands run with the CPU prefix above:

```bash
"${P[@]}" -m exp.offline_search.rounds.r08.ops.prefit --rebuild
"${P[@]}" -m exp.offline_search.rounds.r08.ops.validate_arms --out /tmp/r8_S5/fix1_artifact_validation.json
"${P[@]}" -m pytest exp/offline_search/rounds/r08/methods exp/offline_search/rounds/r08/ops -p no:cacheprovider --basetemp /tmp/r8_S5/fix1_tests -q -s
"${P[@]}" -m unittest exp.offline_search.rounds.r08.methods.test_methods exp.offline_search.rounds.r08.ops.test_ops -v
"${P[@]}" -m pytest exp/offline_search/rounds/r08/methods/test_plugin_parity.py -p no:cacheprovider --basetemp /tmp/r8_S5/fix1_process_parity -q -s
```

- Full owned suite: **33 passed**, six subtests passed, 96.35 s. The only warning
  is the existing third-party `pynvml` deprecation. Log: `/tmp/r8_S5/fix1_tests.log`.
- Final unit run against S2's updated producer: **22 passed**, 3.312 s.
  Log: `/tmp/r8_S5/fix1_unit_tests.log`.
- All eight real library cells still pass **5,391 byte-exact identity
  comparisons**; partial-payload oracle caps pass in all four library-50 cells.
  Report: `/tmp/r8_S5/library_replay.json`.
- FL/W10/W5/IP/P10/O5b each pass **220-request** debug off/on parity for response
  bytes, method/history state, RNG and legacy records. O5b uses a mixed-status
  partial payload throughout. Reports: `/tmp/r8_S5/fix1_tests/`.
- Repeated parity after S1's per-process writer landed: **10 passed**, 59.56 s,
  including all six 220-request tapes and four legacy/process/truncation fixtures.
  Actual output is `decisions_<pid>.jsonl`, `meta_<pid>.json` and
  `writer_stats_<pid>.json`; every writer drains without errors or skipped lines.
  Log/reports: `/tmp/r8_S5/fix1_process_parity.log`, `/tmp/r8_S5/fix1_process_parity/`.
- All **11** owned Python files parse with Python 3.8 grammar.

Machine-readable delivery summary: `/tmp/r8_S5/fix1_summary.json`.

No GPU, server, worker, remote command, simulator, socket, tmux, forbidden CPU,
process-management command or prohibited test directory was used. Real oracle
window coverage remains a coordinator-only non-test smoke check with S2's
deployed adapter; the existing CPU tests establish the payload/call/cap contract.
