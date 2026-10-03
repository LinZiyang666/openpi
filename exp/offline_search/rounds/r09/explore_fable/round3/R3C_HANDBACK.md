# Round 3c (fable) — real root cause found and fixed; new run root r09_fable_r3c (NOT launched)

## ⚠ Cancel the 3 queued GR00T r3b arms: they carry the same defect (their fitted artifacts have judge.burst = 2) and will crash the same way.

## Root cause (verified on the fitted artifacts, not a guess)
The r3/r3b stack mixin stored its recovery-burst length as `self.burst` and kept it over `vars(src)`; the frozen only-no-progress
judge has its own `burst` (= 0 in `r8abl_onlynp_*`; k7 arms its burst/return windows **only when `self.burst > 1`**). Every fitted
r3/r3b stack therefore had `judge.burst = 2` (checked: r9f3b_pi05_l10_50_np_corr05_esc, r9f3b_groot_l10_50_np_corr05,
r9f3b_groot_l10_50_np_corr05_gmS all show `burst 2`; opus's EscalateOnlyNP artifact shows `burst 0`). After the first no-progress
firing k7 set `_s['burst_end'] = step + 2`, which is never cleared, and the P2 trigger mask raises `P2 does not support active
burst/return windows` as soon as a disabled guard (stuck / terminal / overtime) fires later — exactly the guard-path episodes.
The MRO issue reported for r3b was real but secondary; the attribute collision is what killed both r3 and r3b.

## Second defect found while fixing (affects a round-2 number)
Judges retrieve through the G3 base contract (`os_score_all` / `os_synth`), never `base.query`, so a swapped-in `CorrectedCache`
base never applied its correction inside any judge stack. Consequence: round-2 `r9f2_pi05_l10_50_np_corr05pt` (.860 @ .177) was
**only-no-progress alone on my fleet**, not "guard + corrector". v3 fixes this with `CorrectedCacheJ` (correction applied in
`os_synth`; verified by a test that the judge-path chunk differs from the plain kernel mean on the motion channels only).

## Fix (new classes only; r3/r3b classes byte-identical)
`round3/tools/methods.py` v3: `CorrectedCacheJ`, `_GraspStack3` (all stack state under `gm_*`; fit refuses any attribute name that
exists on the frozen judge, refuses mixed judge families, and asserts `judge.burst` unchanged), `NpGraspStack3`, `NpGraspStackGroot3`,
`NpGraspEsc3`. h100 mirror: `exp/offline_search/rounds/r09/explore_fable/round3/tools/methods.py` sha **fa881a57…**.

## Tests
- `tests/test_round3c.py` (3, pass; use the real frozen artifacts): (1) v2 artifact has `burst == 2`, v3 `burst == 0`; (2) **reproduction**:
  with the fitted objects, k7's window-arming line (verbatim condition `if self.burst > 1`) after a no-progress firing, then the REAL
  `_TriggerMask.query` with a disabled-guard firing → v2 raises `P2 does not support active burst/return windows`, v3 masks the guard
  cleanly; (3) corrector active on the judge path. `tests/test_round3.py` (6, pass) incl. MRO family checks.
- CPU plugin selftests (recorded store keys, real judge chain, `--blind --policy-tail --judge guard_only`, 6 episodes) for all six r3c arms
  with production kwargs and forced-trigger variants (`r09_fable_r3c/selftest/*.log`): 11/11 of the stack runs PASS with 0 P2 errors
  (π0.5 prod: 2 guard misses; forced: 16 misses; GR00T prod: 4; forced: 18); the round-2 `corr05_gmS` arm's log is listed separately below.
  Caveat (honest): the earlier 40-episode selftests did NOT reproduce the live crash because the fake driver never produced a disabled-guard
  firing after a no-progress firing; the reproduction is the artifact-level test above, not a closed-loop replay.

## Arms (`/home/weiland/trace_runs/os_closed_loop/r09_fable_r3c`, manifest tasks 0–9 × inits 20–29; `control plan` passes: 57 files, 16.0 GiB)
| arm | class | stack | prediction (PREDICTION_R3.md) — now a true corrector-under-judge stack, so the np_corr05 rows are untested territory |
|---|---|---|---|
| r9f3c_groot_l10_50_np_corr05 | NpGraspStackGroot3, max_calls 0 | guard + half corrector | .83 / .17 |
| r9f3c_groot_l10_50_corr05_gmS | round-2 GraspMissCallsSigned (unchanged) | half corrector + recovery | .81 / .082 |
| r9f3c_groot_l10_50_np_corr05_gmS | NpGraspStackGroot3, max_calls 2 | guard + corrector + recovery | .85 / .18 |
| r9f3c_pi05_l10_50_np_corr05_esc | NpGraspEsc3, max_calls 0 | guard + corrector + escalation | .88 / .20 |
| r9f3c_pi05_l10_50_np_corr05_gm | NpGraspStack3, max_calls 2 | guard + corrector + recovery | .875 / .185 |
| r9f3c_pi05_l10_50_np_corr05_gm_esc | NpGraspEsc3, max_calls 2 | full stack | .89 / .21 |
Artifacts verified: `judge.burst 0`, `gm_burst 2`, base `CorrectedCacheJ` blend .5.

```bash
cd /home/weiland/projects/openpi
P=(taskset -c 22-25 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src .venv/bin/python)
R3C=/home/weiland/trace_runs/os_closed_loop/r09_fable_r3c
ARMS=(r9f3c_groot_l10_50_np_corr05 r9f3c_groot_l10_50_corr05_gmS r9f3c_groot_l10_50_np_corr05_gmS r9f3c_pi05_l10_50_np_corr05_esc r9f3c_pi05_l10_50_np_corr05_gm r9f3c_pi05_l10_50_np_corr05_gm_esc)
WORKER_HOST=<fleet> SYNC_PORT=23195 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control sync --concurrent "$R3C" "${ARMS[@]}"
WORKER_HOST=<fleet> PORTS=<4 ports> WPS=12 MAX_ATTEMPTS=2 POLL_SECONDS=30 OSCL_MANIFEST=$R3C/manifests/eval100_inits20_29.json \
  "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control chain "$R3C" "${ARMS[@]}"
"${P[@]}" -m exp.offline_search.rounds.r09.explore_fable.round2.tools.paired_r2 --ref grown:r9f_ctrlA_p_l10_50 $R3C:r9f3c_pi05_l10_50_np_corr05_esc $R3C:r9f3c_pi05_l10_50_np_corr05_gm $R3C:r9f3c_pi05_l10_50_np_corr05_gm_esc r9f2_pi05_l10_50_np_corr05pt
"${P[@]}" -m exp.offline_search.rounds.r09.explore_fable.round2.tools.paired_r2 --ref r9f2_groot_l10_50_corr0 $R3C:r9f3c_groot_l10_50_np_corr05 $R3C:r9f3c_groot_l10_50_corr05_gmS $R3C:r9f3c_groot_l10_50_np_corr05_gmS r9f2_groot_l10_50_corr05pt
```
Before launching, confirm on h100: `sha256sum .../explore_fable/round3/tools/methods.py` = fa881a57…; abort on any `P2 does not support` line.
