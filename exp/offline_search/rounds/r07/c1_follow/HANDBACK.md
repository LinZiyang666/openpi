# R7 C1 hand-back — SF/UF and shared stages

Written 2026-09-30T06:34:38.800488+00:00. Owned paths only; CPU-only work on 22–25,66–69.

Shared `stages/stages.py` and `stages/STAGES_API.md` first landed 2026-09-30 01:03:30 CDT
(06:03:30 UTC), before follow code and fits. No shared plugin file was changed.

## Deliverables and exact methods

* `stages/stages.py`: E1 two-means/three-row gripper stages, true successors, unknown failed rows,
  content fingerprints, successful-library scales, E3 frozen-A candidate-LOEO valve/deviation tables.
* `c1_follow/methods.py`: `StageFollow` (SF), `UniformFollow` (UF), and importable `FollowExtension`.
* `arms_profile.json`: SF(E=1), SF(E=2), UF(E=1) ×8 = 24 non-test B-val profile arms.
* `arms_eval500.json`: the same 24 candidate full-evaluation arms, subject to profiling pruning/cap freeze.
* `/tmp/r7_C1/stages/`: eight independent frozen tables; `/tmp/r7_C1/fits/`: 24 embedded-table method fits.
* `replay.py`, `test_follow.py`, `verify_specs.py`: reproducible recorded-stream, component and packaging checks.
* `COMPOSITION.md`: plan/check/serve hook, compatible with C3’s `install_follow_extension` bridge.

SF method: `exp.offline_search.rounds.r07.c1_follow.methods:StageFollow`.
UF method: `exp.offline_search.rounds.r07.c1_follow.methods:UniformFollow`.
SF1 kwargs add `extend_blocks=1, stage_gate=true, state_valve=true`;
SF2 changes cap to 2; UF1 adds `extend_blocks=1, stage_gate=false, state_valve=false`.
All use selected A’s `serving="anchor_tail", budget=1, gates="budget_only"`.
Sparse cells use `lib="current", kref=5`; dense π0.5 uses `lib="bpool_cs", kref=8`;
dense GR00T uses `lib="bpool_all", kref=8`. The wrapper accepts those explicit names and
maps them to historical A’s `big` constructor internally, checking the resolved library.

Plugin flags in every row: `--os-root /home/weiland/trace_runs/offline_search_store`,
`--os-no-shadow-native --os-blind --os-fit-artifact <RUN>/fits/r7_<cell>_<variant>.pkl`.
Profile additionally uses `--os-log-inputs` and C4’s `<RUN>/manifests/<model>_<suite>_bval20.json`.
All rows use five-control client decisions, manifest geometry internally, and cost_ledger=true.
There is no call judge or policy-tail flag on these pure-cache arms.

π0.5 extensions execute true `next²`/`next³` member heads with original float32 weights.
GR00T E=1 executes its anchor action controls 10:15 exactly; E=2 next executes `next³` heads.
All members require support, even zero-weight members. No terminal clamp/drop/renormalization.
E=2 is granted in full or rejected; no silent cap degradation. Policy-origin decisions always LOOK.

## Identity and precise requirement limitation

Disabled `extend_blocks=0` delegates directly to A, including Result/blind extras, and introduces
no new fit/calibration dependency. Both disabled SF and disabled UF were checked on every stream.
Enabled SF anchors rejected by structural/stage gates preserve A’s actions, verdicts and vision
flags; they add stage telemetry. The following identity scope is proven:

| cell | P3 decisions | B-val decisions | exact archived actions | rejected-anchor blind/LOOK checks |
|---|---:|---:|---:|---:|
| pi05_l10_50 | 1269 | 688 | 1957 | 1014 |
| pi05_l10_500 | 1167 | 546 | 1713 | 1166 |
| pi05_spatial_50 | 499 | 301 | 800 | 468 |
| pi05_spatial_500 | 439 | 264 | 703 | 303 |
| groot_l10_50 | 1456 | 727 | 2183 | 1450 |
| groot_l10_500 | 1226 | 577 | 1803 | 1447 |
| groot_spatial_50 | 459 | 262 | 721 | 369 |
| groot_spatial_500 | 448 | 241 | 689 | 395 |

**10,569 decisions / 52,449 recorded controls / 240 episodes, zero failures.**
Every archived action chunk, recorded vision flag and cache verdict matches A; both disabled
classes independently match A bitwise (including scores, confidence, rows, weights and extras).
Enabled structural/stage rejections pass 6,612 blind/LOOK comparisons.

**Cannot simultaneously satisfy two literal clauses:** immediate outside-radius LOOK at the
first blind check can change A before any extra block is served. Thus “identity whenever no
extension is ultimately granted/served” cannot also hold for those early valve aborts.
The delivered implementation checks the valve at every in-budget blind decision of an
extension-eligible SF commitment and immediately LOOKs on a breach; structurally/stage-rejected
anchors retain A and do not run an effective valve. No identity claim covers early valve aborts.
B-val observed first-blind aborts: SF1 π0.5 L10-50=2, GR00T L10-500=1; SF2 each of those=1;
all other cells=0. This is an explicit unresolved specification conflict, not a parity claim.

## B-val replay eligibility, valve and modeled cost

Each cell has 10 existing non-test B-val episodes (one per task), not the new 20-episode profile.
Grant share is frozen structural/stage eligibility at the anchor, before the dynamic valve.
Extra blocks and extension share report what survives the recorded-state valve.
Valve rate is fired checks / effective eligible-commitment valve checks; UF has no valve.
IR below is vision-cost / mean completed fixed-anchor cycle length, with zero calls;
missing future states/cap LOOKs are censored. It is neither renewed-controller realized IR nor
a causal SR estimate. Overlapping anchors are not independent. A comparison uses .076/.074.

| cell | arm | anchors | grants | grant % | extra blocks | valve fires/checks | censored | modeled IR | mean / p95 blind μs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pi05_l10_50 | SF1 | 346 | 156 | 45.09 | 153 | 2/308 | 12 | 0.06214 | 232.2 / 471.2 |
| pi05_l10_50 | SF2 | 346 | 126 | 36.42 | 250 | 1/376 | 10 | 0.05545 | 233.1 / 443.4 |
| pi05_l10_50 | UF1 | 346 | 251 | 72.54 | 250 | 0/0 | 12 | 0.05542 | 186.6 / 359.1 |
| pi05_l10_500 | SF1 | 274 | 76 | 27.74 | 75 | 0/151 | 10 | 0.06655 | 195.8 / 413.7 |
| pi05_l10_500 | SF2 | 274 | 65 | 23.72 | 128 | 0/193 | 11 | 0.06131 | 197.2 / 389.6 |
| pi05_l10_500 | UF1 | 274 | 260 | 94.89 | 259 | 0/0 | 11 | 0.05099 | 169.9 / 318.8 |
| pi05_spatial_50 | SF1 | 152 | 57 | 37.50 | 56 | 1/114 | 10 | 0.06348 | 190.4 / 383.1 |
| pi05_spatial_50 | SF2 | 152 | 47 | 30.92 | 92 | 1/140 | 10 | 0.05740 | 187.4 / 359.6 |
| pi05_spatial_50 | UF1 | 152 | 87 | 57.24 | 87 | 0/0 | 10 | 0.05818 | 156.3 / 304.4 |
| pi05_spatial_500 | SF1 | 135 | 66 | 48.89 | 65 | 1/132 | 11 | 0.06041 | 205.0 / 419.6 |
| pi05_spatial_500 | SF2 | 135 | 52 | 38.52 | 102 | 1/155 | 10 | 0.05398 | 203.0 / 388.4 |
| pi05_spatial_500 | UF1 | 135 | 104 | 77.04 | 104 | 0/0 | 11 | 0.05370 | 162.5 / 323.8 |
| groot_l10_50 | SF1 | 365 | 111 | 30.41 | 111 | 0/222 | 10 | 0.06400 | 168.8 / 376.5 |
| groot_l10_50 | SF2 | 365 | 94 | 25.75 | 188 | 0/282 | 10 | 0.05851 | 171.1 / 348.3 |
| groot_l10_50 | UF1 | 365 | 312 | 85.48 | 311 | 0/0 | 10 | 0.05146 | 126.4 / 300.9 |
| groot_l10_500 | SF1 | 290 | 64 | 22.07 | 63 | 1/127 | 10 | 0.06662 | 172.9 / 387.0 |
| groot_l10_500 | SF2 | 290 | 56 | 19.31 | 110 | 1/166 | 10 | 0.06194 | 174.5 / 368.2 |
| groot_l10_500 | UF1 | 290 | 274 | 94.48 | 273 | 0/0 | 10 | 0.04975 | 131.0 / 314.8 |
| groot_spatial_50 | SF1 | 133 | 59 | 44.36 | 59 | 0/118 | 10 | 0.05969 | 175.7 / 386.2 |
| groot_spatial_50 | SF2 | 133 | 53 | 39.85 | 106 | 0/159 | 10 | 0.05172 | 181.7 / 361.4 |
| groot_spatial_50 | UF1 | 133 | 87 | 65.41 | 87 | 0/0 | 10 | 0.05467 | 129.8 / 293.9 |
| groot_spatial_500 | SF1 | 122 | 55 | 45.08 | 55 | 0/110 | 11 | 0.05952 | 177.8 / 386.6 |
| groot_spatial_500 | SF2 | 122 | 49 | 40.16 | 98 | 0/147 | 13 | 0.05204 | 186.6 / 358.0 |
| groot_spatial_500 | UF1 | 122 | 102 | 83.61 | 102 | 0/0 | 12 | 0.05087 | 130.9 / 302.1 |

Timing measures the full `blind_step`, including ordinary A tail checks and budget LOOKs;
it excludes archive reads, constructing histories, vision, policy and wire transforms.
Exact timing sample counts and age breakdowns are in `/tmp/r7_C1/<cell>_replay.json`.
All eligibility shares exceed 5%; every modeled saving exceeds .005. Realized IR and SR kill
criteria must still be measured on the coordinator’s new B-val profile episodes.

## Library calibration

Successful source rows only; even-row A cadence; source episode excluded from the frozen A
candidate set. PCA/metric are not refit per held-out episode. Full support at both reference
checks is required; each row sample is max(D_delta at five and ten controls). Row p95 is default.
Episode p95 is E3’s quantile of per-episode maxima, not inverse-episode-length row weighting.
The two conventions are explicitly separate in STAGES_API.md. CT p75 is anchor absolute
state residual by task, with strictly-above empirical occupancy. No state-unit threshold floor.

| cell | successful rows | full-support LOEO anchors | episodes | row p95 | episode p95 |
|---|---:|---:|---:|---:|---:|
| pi05_l10_50 | 2640 | 1172 | 50 | 0.398759 | 0.840703 |
| pi05_l10_500 | 22816 | 10569 | 436 | 0.264306 | 0.796196 |
| pi05_spatial_50 | 1018 | 390 | 49 | 0.342309 | 0.608316 |
| pi05_spatial_500 | 10337 | 4521 | 487 | 0.272642 | 0.472069 |
| groot_l10_50 | 2645 | 1164 | 50 | 0.367592 | 1.171952 |
| groot_l10_500 | 22039 | 10164 | 427 | 0.369569 | 0.894898 |
| groot_spatial_50 | 1063 | 415 | 50 | 0.321672 | 0.470349 |
| groot_spatial_500 | 9815 | 4332 | 456 | 0.285405 | 0.477272 |

## Exact commands and checks

Run from `/home/weiland/projects/openpi`:

```bash
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c1_follow.prefit --emit-specs
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c1_follow.prefit --cells all
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m unittest exp.offline_search.rounds.r07.c1_follow.test_follow -v
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c1_follow.replay --cells all
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c1_follow.verify_specs
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c1_follow.assemble_handback
```

`prefit_commands.sh` enumerates each of the 24 variant × cell rebuild commands using the delivered
stage tables. Initial calibration command was `prefit --cells all`; final packaging rebuilds used
`prefit --cells all --reuse-stages`, with identical stage/retrieval fingerprints.
Final replays used CPUs 22–25; independent fit/test/packaging work used 66–69; all thread limits
and environment variables above were retained. At most three CPU processes ran concurrently.

Component tests: **13 passed**. Coverage includes E1 segmentation parity, isolated reversal,
content/library fingerprints, true successor failures, unknown/zero-weight unanimity, exact
π0.5/native-GR00T synthesis, non-five-control geometry, every eligible blind valve age, and
disabled-fit dependency identity. Unit log: `/tmp/r7_C1/unit_tests.log`.
Packaging: **48 specs / 24 fits / 48 YAMLs / 48 matrices checked**, **24 policy-origin LOOK checks**.
Replay logs: `/tmp/r7_C1/replays.log`; consolidated machine-readable output:
`/tmp/r7_C1/results.json`; each cell also has replay JSON and per-anchor timeline JSON.
Fits were library-only adaptations of provenance-checked deployed A fits; no recorded outcomes
were used in fitting. B-val replay reproduces acquisition-episode exclusion via the existing
`/tmp/q1_method_c_fits/<cell>/recording_exclusions.json` maps. No test-init calibration.

## Telemetry

Anchor keys: `os_sf_mode0`, `os_sf_mode1`, `os_sf_unanimous`, `os_sf_event_mass`,
`os_sf_rows_to_event`, `os_sf_unknown`, `os_sf_stage_gate`, `os_sf_state_valve`, `os_sf_cap`,
`os_sf_structural`, `os_sf_stage_ok`, `os_sf_granted`.
Eligible blind keys additionally: `os_sf_age`, `os_sf_extension`, `os_sf_valve_checked`,
`os_sf_valve_fire`, `os_sf_delta`, `os_sf_radius`, `os_sf_look`, `os_sf_source`.
Source 1 = unchanged native anchor tail, source 2 = true successor heads.
LOOK 1 = cap/budget, 6 = A lifecycle, 11 = follow state valve; names stay on LookReason.
New scalars precede legacy scalars in eligible first-tail results and fit within the plugin’s
24-scalar pure-cache budget. Complete provenance remains in blind_extras. Disabled results
remain untouched. Stage-rejected results add plan telemetry to blind_extras.
The unchanged plugin logs `vision`, `source`, `look_reason`, executed heads, rows and weights.
SF/UF introduce no camera or call decision; those fields belong to C2/C3 when composed.

## Coordinator next steps

1. Copy `/tmp/r7_C1/fits/*.pkl` to `<RUN>/fits/` and `/tmp/r7_C1/stages/*.pkl` to `<RUN>/stages/`.
   Fitted tables are embedded, and artifact kwargs contain no scratch/stage path; copying needs
   no metadata rewrite. Use the exact C1 method strings/kwargs. Profile specs reuse these same
   fit filenames. C4’s aggregate profile specs use *_profile.pkl: copy/rename matching fits
   if selecting that aggregate list; spec/kwargs/cell still match.
2. Bind only `<RUN>` in plugin paths and C4 manifest paths, then use emit_arms with
   `arms_profile.json`. C4 prepares the audited 20 B-val pairs per cell and their launch route.
3. Smoke one B-val episode each for π0.5 SF1/SF2/UF1 and GR00T SF1/SF2/UF1 before the full
   20-per-cell profile. Check full-hit verdicts, five-control wire heads, native/successor source,
   terminal LOOKs, valve LOOK re-entry and actual stage1 counts/cost ledger.
4. Profile all 24 variants × cells plus A on the same B-val pairs with full decision telemetry.
   Apply SELECTION §4 kill rules and select the SF cap before any R7 test episode. Then emit
   surviving eval500 rows; C1’s SF2 evaluation rows are candidate cap alternatives, not an
   instruction to test both caps before freezing the choice.
5. C3 can install `FollowExtension` via its documented bridge; SA requires a separate budget
   solve and profile. No SA composed class, calibration or arm is claimed here.

No closed loop, model inference, server, worker, chain, tmux, port, remote-host, git or destructive
command was run. `/home/weiland/trace_runs` was read-only. The unresolved identity/early-valve
conflict above is the only stated contract exception; realized profile/eval outcomes remain
coordinator work by instruction.

## File SHA256 and artifact inventory

SHA256s below cover added code/spec/docs except this self-referential hand-back. Its digest
and all files are also recorded in `FILES.sha256` after generation.

```text
db8d0345022e43ad450e4ef3837de425a0d0993cc31a2459da206878da1c65ad  exp/offline_search/rounds/r07/c1_follow/COMPOSITION.md
343f4d517aa7fda15784f65dc0cd725cc09ad8b57feacddc4157dc51db393dab  exp/offline_search/rounds/r07/c1_follow/arms_eval500.json
b56f8de32b5cd3ad337da868e6686b82283f987ebe77136d3e228dc23b55c5b7  exp/offline_search/rounds/r07/c1_follow/arms_profile.json
dd00610a3c81e5a9c0a7e8831bed88875a8a9ccdd75ad3836069f37583cdfb22  exp/offline_search/rounds/r07/c1_follow/assemble_handback.py
1d8739b53cafffb3eacf097c79bb174d8dd9019e9b3b86f9727d7ffbdd727a9a  exp/offline_search/rounds/r07/c1_follow/methods.py
97b12034fa51823f755fd4a461a06e8cdf4c3ae951137718769cd3b324caf73d  exp/offline_search/rounds/r07/c1_follow/prefit.py
bf211096e3e389eaebe25144026c31189ae6e6bccec16dbabdc445bb20bcb0b0  exp/offline_search/rounds/r07/c1_follow/prefit_commands.sh
01f9e204f12978feaaa95bb70b72f6db2ab1f71382831ebbf8ab8d200a7fc9a6  exp/offline_search/rounds/r07/c1_follow/replay.py
c11b69222d16722507f43829d07814df6f3ea3361c27fd653eb99d6db076a5a9  exp/offline_search/rounds/r07/c1_follow/test_follow.py
a1af3fc6f07734670962b3f898460fc3caaa9b9e57120012b789f4e20ce6eb50  exp/offline_search/rounds/r07/c1_follow/verify_specs.py
89c16a94420bcfda24ab90a0a5699873f5510c163b5c4fefdb670dfc3447e0c9  exp/offline_search/rounds/r07/stages/STAGES_API.md
ad6ee8015370b111f1a337b667c07ac7bad20573217d39cc7e4a353e75e99e06  exp/offline_search/rounds/r07/stages/stages.py
```

Artifact SHA256s and byte sizes: `/tmp/r7_C1/packaging.json` (all 24 fits),
`/tmp/r7_C1/<cell>_fit.json` (eight tables and three fits each). Frozen-table digests:

```text
e597f90055f98e6ed955ba91f84c1cc93db25fa5fb40ecb47a4ffa81c2b35503  /tmp/r7_C1/stages/pi05_l10_50.pkl
0df3778bfd12b7f17518b79188824bb7534c2bf5543579f35e86b22b78e53056  /tmp/r7_C1/stages/pi05_l10_500.pkl
cbeb7c231ceff28d139e6d2b8cb1c8e17225e2068030149f374736aa4da749e4  /tmp/r7_C1/stages/pi05_spatial_50.pkl
3ab47b303008f46b484005d0984142da6ba8abbe0fc4362148a53cfb4f8dfe8c  /tmp/r7_C1/stages/pi05_spatial_500.pkl
0115634e409d9c67165d639d81f8ce3b2ed237cfbc1dead3f01f1ce679aaf113  /tmp/r7_C1/stages/groot_l10_50.pkl
6728a597ddcc598efdb0a97e9ab403c2cfd162fb7d2dce23ad575f9cb8c8087f  /tmp/r7_C1/stages/groot_l10_500.pkl
046b21923f80fba9342e06aba36b2ffee2aad0a68c89ada69b07f96cb9eb2514  /tmp/r7_C1/stages/groot_spatial_50.pkl
f425167530668e8e6c19c9b0f90cd7bb829381de36786ed6d4a750bad6a0d3fd  /tmp/r7_C1/stages/groot_spatial_500.pkl
```
