# K10 hand-back: opt-in policy tail

Completed UTC: 2026-09-28T01:19:21.016453+00:00.

Implemented and atomically installed the plugin path, K7 subclass, fake-policy/selftest and replay verification support. No K1/K7, src, harness, profile, ops, cost-table or other owner files were edited. No GPU, server, port, LIBERO worker, chain, remote host, git command or review_tests content was used. Python ran on CPUs 26–29,70–73, CUDA disabled, OMP/OpenBLAS/MKL threads 1. No agent delegation was used.

## Design and contract

`--os-policy-tail` requires `--os-blind` and π0.5. Without the new switch, no policy response is retained for tail reuse and legacy logs keep their exact schema and values (fixed-clock/PID comparisons below). Existing methods are unaffected. The optional method hook is `policy_tail_step(BlindQueryView) -> BlindResult | LookReason`; an absent hook requests vision. It receives the actual previous MISS via `prev_hit=False`, `prev_a_exec`, and dense histories, with no visual fields.

The plugin saves copies of both the normalized policy chunk and the original returned wire actions, tagged with the actual per-connection EpisodeView identity and decision step. Only the immediately following decision can consume them. Step 0, reset/task change, a snapshot from another episode, invalid current state, non-five execution audit, a second blind after a policy tail, judge burst/cap or globally due periodic MISS all require vision. A rejected duplicate reserves no index and preserves the saved response; an admitted failure clears it. The same per-connection K6 lock covers selection, response capture, commit and lifecycle. No runtime lock was added around method or policy work.

For H=10, serve normalized `policy_chunk[5:10]` followed by five repeats of its last row, preserving every one of the 32 columns and float32 dtype. The wire chunk separately uses `previous_response.actions[5:10]` followed by five repeats of its final wire row, preserving its dtype and columns. The plugin enforces exact normalized equality against the saved MISS regardless of hook output. No output transform is reapplied to the policy tail. This preserves the same controls as the L=10 client queue even if transforms depend on the old observation state. Only five controls of the tail response may execute. For H>10 the helper retains all remaining rows but still allows only one five-control blind decision before vision.

`PolicyTailJudge` subclasses K7 `VisionConfirmedBlindMixedJudge`. HIT `blind_step` is inherited unchanged. It retains the last vision proposal separately for gate evaluation while the ordinary K1 anchor is still invalidated on MISS. The policy hook temporarily passes that proposal and a `prev_hit=True` gate facade through the inherited budget, span guard and base gates, then restores the invalid anchor. Actual `hist_hit`, action history, vision mask and guard memos are never changed. The candidate library rows are gate provenance only; the returned action is the policy tail. Supplied arms use anchor_tail, budget 1, budget_only. Budget 0 is available for tests; budgets >1 and non-tail serving are rejected. K7 stuck confirmation and no-progress span consume the policy-tail row as a blind gap; visual keys remain NaN.

### Execution-count contract

The existing LIBERO client sends no per-request execution count and requests again when its action queue empties. As in the existing blind protocol, absent `__extra__.executed_steps` means the configured five-control client contract. A supplied value other than 5 requests vision; terminal partial blocks end the episode and cannot supply a tail in the next episode. K10 rows expose `previous_executed_steps` and `execution_audit` so a protocol assumption is distinguishable from an explicit audit. The supplied arms retain L=5. Do not use the switch with an L=10 or otherwise non-five client that omits this audit: the server cannot infer unreported physical execution.

### Logs and costs

Policy-tail rows are `src=source="policy_tail"`, `vision=false`, `hit=true`, `judge="blind"`, `look_reason=null`, `blind_age=0` (consecutive blind decisions BEFORE this request), `miss_k=null`, both stage timings null, searched/shadow_available false, and `exec_ok=true`. The next vision row has blind_age 1. `hit=true` means a served response requiring no new policy inference, not a library retrieval. Wire hit_type is FULL_HIT. Normalized served_head is exact, and NPZ records retain full normalized and wire chunks, source and audit fields. Policy-tail extras identify `policy_tail=1` and the source decision.

The existing K4 ledger reads the explicit vision and hit flags: a tail costs zero and adds neither a vision nor a MISS. Member rows/weights/lib describe the last vision proposal used for gates, consistent with proposal diagnostics on MISSes; they must not be interpreted as the action source. Existing KPI action metrics use served_head correctly; library spell/terminal diagnostics on these rows remain proposal-based. No cost-table or ops change was needed.

## Atomic installation

Every shared file was developed in dev/, preimage-checked, parsed, written/fsynced to a same-directory temporary file and installed with one `os.replace` (atomic rename/mv), then directory-fsynced. Dependency order was blind.py before plugin.py. No shared edits followed these installs.

| File | Installed UTC | SHA256 |
|---|---|---|
| `exp/offline_search/closed_loop/blind.py` | 2026-09-28T00:25:06.244148+00:00 | `40ef0112fa2e50cd01c813ba9f04ddec496185f53366bd54b9698101a4f65c9b` |
| `exp/offline_search/closed_loop/plugin.py` | 2026-09-28T00:25:06.247961+00:00 | `0697a6b9b72a0d8919030e3bfaf395fe5e2ff950a379367b52e74b180ff20d0b` |
| `exp/offline_search/closed_loop/verify_logs.py` | 2026-09-28T00:25:06.251579+00:00 | `08910bde0b0044a01237af9c5b2e58e6bf921284dfb7ddf0b9f2760a037621a3` |
| `exp/offline_search/closed_loop/selftest.py` | 2026-09-28T00:25:06.254651+00:00 | `62403c4cc3dce6e19b5e9490b388e3e04841c0fad4da789b39205ac6ea0b886c` |
| `exp/offline_search/closed_loop/replay_client.py` | 2026-09-28T00:25:06.257664+00:00 | `6691532d63ac27ae90702becd0635795363c41167cb09dd96a14f506a2740b6d` |

Primary new deliverables (UTC file availability / final modification time):

| File | UTC | SHA256 |
|---|---|---|
| `judge.py` | 2026-09-28T00:15:20.030061+00:00 | `4c3567bcd22ab2ddddcfb5c4a3b74bb9b8208e8bf82ccf722a19815659f869c3` |
| `__init__.py` | 2026-09-28T00:15:20.033061+00:00 | `a6c314557a2e79c7477fbed2688336ceea6c768e9cb164fa64a7b20ee7736ee7` |
| `arms_k10.json` | 2026-09-28T00:21:17.808309+00:00 | `a4e94e6f2fac28fbbae3c7ded14bae68fb6e0a911e076866975bab6c3537379f` |
| `prefit.sh` | 2026-09-28T00:21:17.809309+00:00 | `1edb4b458a4004e04bedf030c1ae4d8fa09df40ad32048db1ca2ce91797c7abc` |
| `prefit_commands.json` | 2026-09-28T00:21:17.808309+00:00 | `5da1375657c37cc417784d61214f5ab76a6663210adba18a91fa944e4b33e462` |

`results/install.json` also records preimage hashes and sizes. `results/deliverable_hashes.json` records the added top-level source/spec/report deliverables and their UTC modification times/hashes (excluding the checksum sidecar). `before/` contains the captured shared preimages; `dev/` the installed candidates and adapted recipes. Primary new files are judge.py, arms_k10.json, prefit.sh, prefit_commands.json, policy_tail_test.py, method_test.py, tail_concurrency_test.py, run_tail_matrix.py, validation/installation runners and this handback.

## Installed verification results

All reported final checks used the installed shared files. Compact evidence is in results/installed/ and results/final_summary.json; full logs/NPZs are in the /tmp paths listed below.

| Check | Runs / checks | Decisions / comparisons | Result |
|---|---:|---:|---|
| K2 original plugin matrix | 21 | 963 decisions | PASS |
| K1 original plugin matrix | 10 | 480 decisions | PASS |
| K4 original plugin matrix | 4 | 156 decisions | PASS |
| K5 four real MixedJudge randomized replays | 4 | 1368 decisions | PASS |
| K5 injected overlay lifecycle/history | 1 | 104 decisions | PASS |
| K6 eight-connection serialized/concurrent parity | 12 configurations | 2196 threaded + same serialized | PASS |
| K7 existing plugin arms | 5 | 240 decisions | PASS |
| K7 scalar parity | 4 cell/scales | 15272 complete query comparisons | PASS |
| K7 full-cell count parity | 4 cell/scales | 189904 decisions | PASS |
| K7 edges | 13 synthetic checks | 150 gap queries | PASS |
| K6 edges | 8 connections | 23 decisions, 25 reservations, 64 quantile transactions | PASS |
| K10 existing blind driver, three cells × three judges | 9 | 432 decisions, 118 policy tails | PASS |
| K10 eight-connection serialized/concurrent parity | 9 configurations | 1647 threaded + same serialized; 347 tails per side | PASS |
| K10 transform/L10/lifecycle/guard/ledger | 20 checks | 114 decisions, 42 tails, 400 L10 controls | PASS |
| K10 vs K7 HIT path | 3 cell/scales | 72 exact query results, 66 exact blind results, 6 identical vetoes | PASS |
| Arm emitter and parser | 3 | exact kwargs / full model / cost ledger / L5 | PASS |

K6 edge_test reran the original deterministic race/failure/partial-write/quantile/clone checks; see results/installed/k6_edges.log and /tmp/k10_installed_k6_edges. K7 original edges, scalar parity and full-cell rate checks reran through k7_unit.py with only output paths relocated; their complete original counters are in results/installed/k7_unit/results/ and copied in final_summary.json. K5 estimator validation reran 200 planted + 200 null datasets, 500 init clusters each, 499 bootstrap draws; exact effects, forged logs, incomplete/cross-scale pairs, raw/collect/KPI/export consistency, historical layouts and four-arm cost ledger passed. These are regression checks, not rollout outcomes.

Flag-off JSONL comparisons use identical invocation/output paths, frozen clock and PID, against before/; all bytes, including startup rows, were equal:

| Mode | Bytes | Rows |
|---|---:|---:|
| pure | 63,350 | 45 |
| mixed | 91,774 | 45 |
| r4 | 145,755 | 45 |
| blind | 163,447 | 54 |
| rand1 | 744,786 | 212 |
| rand2 | 744,003 | 212 |

The new edge test recorded 42 policy-tail decisions with ledger vision_decisions=0, misses=0, total_cost=0.0 and IR per five controls=0.0. It used the installed pi05 output transform stack and normalization assets at `/home/weiland/projects/openpi/assets/pi05_libero`. The client comparison extended/popped action queues exactly as examples/libero/main.py does; a separate state-dependent AbsoluteActions case demonstrated why retaining the original wire response matters.

Development findings: the first transform fixture tried the checkpoint-local assets path (absent); it was corrected to the existing repository normalization-asset fallback. The all-MISS duplicate test exposed loss of a cached tail on duplicate rejection; this was fixed before installation by invalidating only admitted requests. A method-test fixture initially used dataclasses.replace on EpisodeView, which is not a dataclass; corrected to an explicit test identity. The initial transform and duplicate failures are retained in results/dev/. Final results above passed.

Guard-only concurrency replays served zero policy tails at all three cell/scales: guards vetoed reuse after their 7, 4 and 10 MISSes respectively. Forced-MISS and periodic configurations exercised 347 tails per threaded/serialized side. This is an implementation/gate check, not evidence that the feature improves rollout SR or IR; realized policy-tail share must be measured by the coordinator.

## Arms and exact coordinator prefits

`arms_k10.json` is emit_arms format with literal `<RUN>` placeholders. Exact base kwargs match K7 tail1ug: l10 500 and spatial 500 use lib=big/kref=8; l10 50 uses lib=current/kref=5. All use anchor_tail/budget=1/gates=budget_only, noprog_span, events=none, vision_confirmed. All are full-model, cost_ledger=true, with `--os-blind --os-policy-tail --os-judge guard_only --os-no-shadow-native`. MISS remains the ordinary K=10 policy path; no YAML miss patch. K10 deployment prefits were not run here and are intentionally left to the coordinator. Do not relabel K7 pickles as K10: the artifact validates exact method/kwargs/cell metadata.

Run `RUN=/absolute/run/root bash exp/offline_search/rounds/r04/k10_policy_tail/prefit.sh` from the repository. The exact expanded template command for each arm (replace literal `<RUN>` with the same run root) is:

```bash
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.closed_loop.plugin --os-method exp.offline_search.rounds.r04.k10_policy_tail.judge:PolicyTailJudge --os-kwargs '{"base_kwargs":{"lib":"big","kref":8,"serving":"anchor_tail","budget":1,"gates":"budget_only"},"progress_guard":"noprog_span","events":"none","stuck_guard":"vision_confirmed"}' --os-cell pi05_l10_cache --os-log-dir '<RUN>/prefit_logs/r4k10_p_l10_500_tail1ug' --os-tag r4k10_p_l10_500_tail1ug --os-root /home/weiland/trace_runs/offline_search_store --os-blind --os-policy-tail --os-judge guard_only --os-no-shadow-native --os-fit-artifact '<RUN>/fits/r4k10_p_l10_500_tail1ug.pkl'
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.closed_loop.plugin --os-method exp.offline_search.rounds.r04.k10_policy_tail.judge:PolicyTailJudge --os-kwargs '{"base_kwargs":{"lib":"current","kref":5,"serving":"anchor_tail","budget":1,"gates":"budget_only"},"progress_guard":"noprog_span","events":"none","stuck_guard":"vision_confirmed"}' --os-cell pi05_l10_cache --os-log-dir '<RUN>/prefit_logs/r4k10_p_l10_50_tail1ug' --os-tag r4k10_p_l10_50_tail1ug --os-root /home/weiland/trace_runs/offline_search_store --os-blind --os-policy-tail --os-judge guard_only --os-no-shadow-native --os-fit-artifact '<RUN>/fits/r4k10_p_l10_50_tail1ug.pkl'
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.closed_loop.plugin --os-method exp.offline_search.rounds.r04.k10_policy_tail.judge:PolicyTailJudge --os-kwargs '{"base_kwargs":{"lib":"big","kref":8,"serving":"anchor_tail","budget":1,"gates":"budget_only"},"progress_guard":"noprog_span","events":"none","stuck_guard":"vision_confirmed"}' --os-cell pi05_spatial_cache --os-log-dir '<RUN>/prefit_logs/r4k10_p_sp_500_tail1ug' --os-tag r4k10_p_sp_500_tail1ug --os-root /home/weiland/trace_runs/offline_search_store --os-blind --os-policy-tail --os-judge guard_only --os-no-shadow-native --os-fit-artifact '<RUN>/fits/r4k10_p_sp_500_tail1ug.pkl'
```

Emitter/parser validation wrote /tmp/k10_emitted and results/arms_validation.json. It did not launch servers or run prefits. Test ncal=64 configurations are verification-only; supplied arms retain K7 default calibration kwargs. The separate HIT parity test reads existing K7 fits as references, never as deployment K10 artifacts. No new learned representation or borrowed cross-scale controller information is introduced.

## Exact verification commands

Commands ran from /home/weiland/projects/openpi. The shell recipes preserve original owner test logic, changing affinity/loaders and owned output paths. Their expanded command records are retained alongside reports.

```bash
P=(taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
B=exp/offline_search/rounds/r04/k10_policy_tail
"${P[@]}" "$B/install.py"
bash "$B/run_installed.sh" > "$B/results/installed_run.log" 2>&1
"${P[@]}" "$B/method_test.py" > "$B/results/installed/method_test.log" 2>&1
"${P[@]}" "$B/validate_arms.py" > "$B/results/arms_validation.log" 2>&1
"${P[@]}" "$B/write_handback.py"
```

For a fresh rerun, choose fresh /tmp outputs in the recipes; blind logs append and concurrency/edge drivers refuse existing outputs. Do not rerun install.py just to repeat tests: it deliberately requires the original preimages. Full test commands are in dev/installed_{k2,k1,k4}.sh, run_installed.sh, results/installed/replay_commands_installed.json, k7_commands.json and tail_matrix.json.

## Coordinator next steps and limits

1. Run the three exact prefits, resolve <RUN> in arms_k10.json and emit the arms with the existing emitter. Keep the client at five controls/request and the full model loaded. Use K=10 MISS, no miss YAML patch.
2. Let current arms finish. New servers import the installed plugin; already running servers retain their imported code. Any separate checkout needs all five shared files and the K10 method module. The plugin never restarts running services.
3. Run the coordinator-owned live smoke, then paired evaluation at both library scales and spatial. Audit src=policy_tail, non-vision/non-MISS counts, actual client L=5, and the ledger. No live GPU, simulator, closed-loop SR/IR, stochastic rollout equality or hardware throughput claim is made here.
4. L10 equality is verified for supplied normalized policy chunks and the installed CPU output transforms, not for newly sampled GPU inference or simulator dynamics. Endpoint-confirmed K7 guards retain their documented blind-gap limitations. A method may request vision after a MISS; policy-tail counts therefore need not equal MISS counts.
