# Round 3b (fable) — fixed stacks, new run root r09_fable_r3b (NOT launched)

## Root cause of the r3 abort
`NpGraspStackGroot(NpGraspStack, TriggerGrootCommitJudge)` linearized BOTH frozen R8 judge families (π0.5 `TriggerCommitJudge` and
GR00T `TriggerGrootCommitJudge`) into one MRO (verified: both R8 ablation classes appear in `NpGraspStackGroot.__mro__`), so one
decision ran two judge query chains over the same per-episode state; the frozen P2 trigger mask then found a nonzero burst/return
window and raised `ContractError: P2 does not support active burst/return windows` (traceback: `round3/tools/methods.py:129 query →
r06/p2_ablations/judge.py:40`). The episodes that crashed were exactly those where the guard/mask path was exercised, so the partial
r3 result is invalid and was discarded. The π0.5 classes had a clean MRO but never ran.

## Fix (new classes only; r3 classes left byte-identical)
`round3/tools/methods.py` v2: `_GraspStack2` mixin + `NpGraspStack2` (π0.5 judge family), `NpGraspStackGroot2` (GR00T family only;
refuses `closed_sign != -1`), `NpGraspEsc2` (opus escalation + corrected base + optional recovery). `_stack_fit` now refuses any class
whose MRO contains both R8 judge classes. Debug hook `force_trigger_at` (default empty; production arms use `[]`) fires the empty-grasp
trigger at given decision indices so tests can exercise guard misses and recovery bursts in the same episode. h100 mirror updated:
`exp/offline_search/rounds/r09/explore_fable/round3/tools/methods.py` sha 583ab645… (opus `round2/methods.py` 2ede2b5f…, fable
`round2/tools/methods.py` 86774b83… unchanged).

## Tests
- Unit (`tests/test_round3.py`, 6 pass): trigger burst/cap/extras contract, sign handling, parameter validation, MRO family check
  (asserts the r3 GR00T class mixes two families and every v2 class exactly one), forced-trigger hook.
- Integration (`tests/test_round3_integration.py`, 3 cases): the CPU plugin selftest (`--blind --policy-tail --judge guard_only`,
  recorded store keys, real judge chain) for NpGraspStack2 / NpGraspEsc2 / NpGraspStackGroot2 with `force_trigger_at=[2, 6]`;
  asserts PASS, no P2 error, and ≥ 2 policy misses per run.
- CPU plugin selftests of all six arms (production kwargs) and forced-trigger variants (`r09_fable_r3b/selftest/*.log`), 12/12 PASS:
  π0.5 np_corr05_gm / _esc / _gm_esc: prod 2 guard misses + 2 tails each, forced 16/16; GR00T np_corr05 and np_corr05_gmS: prod 4
  guard misses + 4 tails, forced 18/18; GR00T corr05_gmS (round-2 class): PASS.

## Arms (`/home/weiland/trace_runs/os_closed_loop/r09_fable_r3b`, manifest tasks 0–9 × inits 20–29; `control plan` passes: 57 files, 16.0 GiB logical)
| arm | class | stack | prediction (PREDICTION_R3.md, unchanged) |
|---|---|---|---|
| r9f3b_groot_l10_50_np_corr05 | NpGraspStackGroot2, max_calls 0 | only-no-progress + half corrector | .83 / IR .17 |
| r9f3b_groot_l10_50_corr05_gmS | round-2 GraspMissCallsSigned | half corrector + empty-grasp recovery | .81 / .082 |
| r9f3b_groot_l10_50_np_corr05_gmS | NpGraspStackGroot2, max_calls 2 | guard + corrector + recovery | .85 / .18 |
| r9f3b_pi05_l10_50_np_corr05_esc | NpGraspEsc2, max_calls 0 | guard + corrector + opus escalation | .88 / .20 |
| r9f3b_pi05_l10_50_np_corr05_gm | NpGraspStack2, max_calls 2 | guard + corrector + recovery | .875 / .185 |
| r9f3b_pi05_l10_50_np_corr05_gm_esc | NpGraspEsc2, max_calls 2 | full stack | .89 / .21 |

```bash
cd /home/weiland/projects/openpi
P=(taskset -c 22-25 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src .venv/bin/python)
R3B=/home/weiland/trace_runs/os_closed_loop/r09_fable_r3b
ARMS=(r9f3b_groot_l10_50_np_corr05 r9f3b_groot_l10_50_corr05_gmS r9f3b_groot_l10_50_np_corr05_gmS r9f3b_pi05_l10_50_np_corr05_esc r9f3b_pi05_l10_50_np_corr05_gm r9f3b_pi05_l10_50_np_corr05_gm_esc)
WORKER_HOST=<fleet> SYNC_PORT=23195 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control sync --concurrent "$R3B" "${ARMS[@]}"
WORKER_HOST=<fleet> PORTS=<4 ports> WPS=12 MAX_ATTEMPTS=2 POLL_SECONDS=30 OSCL_MANIFEST=$R3B/manifests/eval100_inits20_29.json \
  "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control chain "$R3B" "${ARMS[@]}"
"${P[@]}" -m exp.offline_search.rounds.r09.explore_fable.round2.tools.paired_r2 --ref grown:r9f_ctrlA_p_l10_50 $R3B:r9f3b_pi05_l10_50_np_corr05_esc $R3B:r9f3b_pi05_l10_50_np_corr05_gm $R3B:r9f3b_pi05_l10_50_np_corr05_gm_esc r9f2_pi05_l10_50_np_corr05pt
"${P[@]}" -m exp.offline_search.rounds.r09.explore_fable.round2.tools.paired_r2 --ref r9f2_groot_l10_50_corr0 $R3B:r9f3b_groot_l10_50_np_corr05 $R3B:r9f3b_groot_l10_50_corr05_gmS $R3B:r9f3b_groot_l10_50_np_corr05_gmS r9f2_groot_l10_50_corr05pt
```
All six arms are full-model (π0.5 9 GB, GR00T 8 GB per port). If a server log ever shows `P2 does not support`, abort: it means the
deployed module is not sha 583ab645….
