# Round 3 (fable) — stacks: no-progress guard × half corrector × empty-grasp recovery × escalation

Rules: inits 0–29 only; every fitted component (corrector heads, aperture thresholds) from inits 0–19; evaluation manifest =
tasks 0–9 × inits 20–29; nothing task-indexed; no chain launched by me (both fleets busy with the coordinator's holdout runs).
Opus's and astra's round-2 outputs were read after independence ended; opus's `EscalateOnlyNP` is imported, not copied.

## Code
`round3/tools/methods.py` (new file; 3 unit tests pass in `round3/tools/tests/test_round3.py`):
- `NpGraspStack` (π0.5) / `NpGraspStackGroot` (GR00T): copy the frozen R8 "A + only no-progress" judge field by field (as opus's
  `_EscalateJudge` does), swap its cache base for the round-2 `CorrectedCache` artifact (half strength, heads fitted on 0–19), add the
  empty-grasp recovery trigger (close held ≥ 2 decisions and finger aperture < .0010 π0.5 / .0009 GR00T; `closed_sign` +1 / −1;
  2-call burst, ≤ `max_calls` triggers; `max_calls=0` = guard + corrector only). Forced misses use the existing contract
  (`os_force_miss`, `os_reason` 93, `_s['flag']`), so CU's policy-tail lifecycle commits 10 controls per call.
- `NpGraspEsc` (π0.5): the same on top of opus's `EscalateOnlyNP` (pace-lag 12, deadline 80, persistent escalation), unmodified.
Deployed to h100 (new files only): `exp/offline_search/rounds/r09/explore_fable/round3/{__init__.py,tools/__init__.py,tools/methods.py}`
(sha f19671eb…). Also required on h100 and already present: `explore_fable/round2/tools/methods.py` (86774b83…) and
`explore_opus/round2/methods.py` (2ede2b5f…, identical to the local file).

## Arms (run root `/home/weiland/trace_runs/os_closed_loop/r09_fable_r3`, manifest `manifests/eval100_inits20_29.json`)
| arm | class | stack | predicted SR / IR (PREDICTION_R3.md) |
|---|---|---|---|
| `r9f3_groot_l10_50_np_corr05` | NpGraspStackGroot, max_calls 0 | only-no-progress + half corrector | .83 / .17 |
| `r9f3_groot_l10_50_corr05_gmS` | round2 GraspMissCallsSigned | half corrector + empty-grasp recovery | .81 / .082 |
| `r9f3_groot_l10_50_np_corr05_gmS` | NpGraspStackGroot, max_calls 2 | guard + corrector + recovery | .85 / .18 |
| `r9f3_pi05_l10_50_np_corr05_esc` | NpGraspEsc, max_calls 0 | guard + corrector + opus escalation | .88 / .20 |
| `r9f3_pi05_l10_50_np_corr05_gm` | NpGraspStack, max_calls 2 | guard + corrector + recovery | .875 / .185 |
| `r9f3_pi05_l10_50_np_corr05_gm_esc` | NpGraspEsc, max_calls 2 | full stack | .89 / .21 |
Same-fleet references on these pairs: π0.5 cache .740 / corr05 .810 / np_corr05 .860 @ .177 / policy .930; GR00T cache .560 /
corr05 .770 / gmS_corr0 .630 @ .086; GR00T pure policy on this fleet not yet run (timan108: .870).

## Commands (coordinator)
```bash
cd /home/weiland/projects/openpi
P=(taskset -c 22-25 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src .venv/bin/python)
R3=/home/weiland/trace_runs/os_closed_loop/r09_fable_r3
ARMS=(r9f3_groot_l10_50_np_corr05 r9f3_groot_l10_50_corr05_gmS r9f3_groot_l10_50_np_corr05_gmS r9f3_pi05_l10_50_np_corr05_esc r9f3_pi05_l10_50_np_corr05_gm r9f3_pi05_l10_50_np_corr05_gm_esc)
WORKER_HOST=timan107 SYNC_PORT=23195 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control sync --concurrent "$R3" "${ARMS[@]}"
WORKER_HOST=timan107 PORTS=23240,23241,23242,23243 WPS=12 MAX_ATTEMPTS=2 POLL_SECONDS=30 OSCL_MANIFEST=$R3/manifests/eval100_inits20_29.json \
  "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control chain "$R3" "${ARMS[@]}"
# analysis (paired, per task; add r3 to ROOTS or pass the full root path)
"${P[@]}" -m exp.offline_search.rounds.r09.explore_fable.round2.tools.paired_r2 --ref grown:r9f_ctrlA_p_l10_50 $R3:r9f3_pi05_l10_50_np_corr05_esc $R3:r9f3_pi05_l10_50_np_corr05_gm $R3:r9f3_pi05_l10_50_np_corr05_gm_esc r9f2_pi05_l10_50_np_corr05pt
"${P[@]}" -m exp.offline_search.rounds.r09.explore_fable.round2.tools.paired_r2 --ref r9f2_groot_l10_50_corr0 $R3:r9f3_groot_l10_50_np_corr05 $R3:r9f3_groot_l10_50_corr05_gmS $R3:r9f3_groot_l10_50_np_corr05_gmS r9f2_groot_l10_50_corr05pt
```
All six arms are full-model (policy calls); GR00T servers need 8 GB, π0.5 9 GB per port. Expected wall time ≈ 6 × 4 min.

## Decision rule (preregistered)
A stack is a 300-pair / holdout candidate for its cell if it beats the best single component on the same pairs by ≥ 3 pp paired
(McNemar) and its IR is ≤ .21 (π0.5) / ≤ .20 (GR00T); among qualifying stacks prefer the lowest IR within 2 pp of the best SR.
