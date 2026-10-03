# Round 2b (fable) — empty-grasp line deepened; frozen arms in r09_fable_r2c

Rules: inits 0–29 only; fits on 0–19, evaluation on 20–29; nothing per task; explore_astra/round2* and explore_opus not read;
no chain launched by me (r2c is emitted, prefitted and `control plan` passes; launch commands below).

## 1. Live closed-loop results (chain 1, inits 20–29, 100 pairs; `RESULTS_LIVE.md` has the full table)
π0.5 LIBERO-10-50, same-topology cache .740 @ .0765, pure policy .930 @ .504:
- corrector FULL strength (per-task heads fitted on 0–19): .630 (−11 pp); single task-agnostic head full strength: .540 (−20 pp) → full strength is harmful closed loop although best offline; keep ≤ .5.
- **empty-grasp recovery on the plain cache (`gm_corr0`): .790 @ .0863**, +5 pp [−1, +11], discordant +8/−3; 0.74 recovery calls per episode (+.010 IR). Same SR with the full-strength corrector underneath (`gm_corr1pt` .790 @ .0835; +16 pp over its own base .630).
- GR00T L10-50 and Spatial-50 trigger arms, and batch 2 (half-strength corrector ± trigger, no-progress judge on the half-corrected base, trigger + uniform coin .15) are running/queued under the coordinator.

## 2. Offline deepening (fit 0–19 / eval 20–29; `tools/aperture.py`, inline scripts recorded in HANDBACK)

**2.1 What the policy does after an empty grasp (shadows at flagged decisions, pure-cache arm, inits 20–29).**
π0.5: in the 2 decisions after the flag the policy holds the gripper OPEN 66% of the controls (≥ 50% in 74% of
flags) and would close again ~2 decisions later → its recovery is "open, re-approach, re-close": an open-and-retry
response is aligned with the policy for π0.5. GR00T: only 26% open (closes again immediately) → GR00T's recovery is
a re-position with the gripper closed; a cache-side "open-and-retry" is not what its policy does.

**2.2 How many calls, how soon (independent-coin arms, inits 0–29, observational).** π0.5 L10-50 flagged episodes
(n 63, SR .21): a call within 2 decisions of the flag → SR .37 (n 19) vs .14 without (n 44); ≥ 2 calls within 8
decisions .38 (n 13) vs 1 call .14 (n 35) vs 0 calls .20 (n 15). GR00T (n 69, SR .13): 1 call .19 (n 31) vs 0 calls
.04 (n 24); ≥ 2 calls .14 (n 14). → Call immediately; for π0.5 a longer burst may help; for GR00T one call already
helps and more does not obviously add. Hence the r2c sweep: burst 1 vs burst 3 (max 3).

**2.3 The failures the flag does not catch (inits 20–29, pure-cache arms) and a second robot-only signal.**
| cell | failures | uncaught by empty-grasp | labels of the uncaught | "long carry" (close held ≥ 150 controls): failures / successes flagged | "stall" (eef path < 1 cm over 30 controls) | union of the three |
|---|---|---|---|---|---|---|
| π0.5 L10-50 | 27 | 5 | fixture_not_done 3, unknown_release 1, held_not_placed 1 | 10/27 / **0/73** | 5/27 / 5/73 | 25/27 failures vs 6/73 successes |
| GR00T L10-50 | 38 | 10 | misplace 4, grasp_miss 2, undone 2, unknown_release 2 | 14/38 / **0/62** | 16/38 / 1/62 | 36/38 vs 5/62 |
| π0.5 Sp-50 | 21 | 2 | misplace 2 | 1/21 / 0/79 | 2/21 / 0/79 | 19/21 vs 0/79 |
The long-carry signal (object held but never placed) has zero false alarms on held-out successes and catches the
place-side failures (fixture / held_not_placed / some misplace); it fires late (≈ 160 controls after the forensic
onset) but long before the step cap. Failures also show 3× more close attempts (7.3 vs 2.0 per episode) and 2×
the gripper toggles — repeated closing is itself a third candidate signal (≥ 3 closes: 21/27 failures vs 22/73
successes on π0.5 — too unspecific alone).

**2.4 Earlier detection from the approach phase.** Not found in this round: at onsets the gripper command agrees
with the policy and the motion gap at the two decisions before the close is at the episode average; the empty
grasp is the first robot-observable event. (The flag already precedes the forensic onset by 30–70 controls because
that onset is the window entry of the *final* attempt.)

## 3. Frozen arms, run root `/home/weiland/trace_runs/os_closed_loop/r09_fable_r2c` (not launched)
π0.5 LIBERO-10-50, manifest tasks 0–9 × inits 20–29 (`manifests/eval100_inits20_29.json`), plain cache base
(blend 0, no fitted component, so the same artifacts can go to all 300 discovery pairs), serving class
`round2/tools/methods.py` (`GraspMissCalls`, new `GraspMissCalls2`; pushed to the h100 tree, sha 539b501d…).
| arm | trigger | response | cap |
|---|---|---|---|
| `r9f2c_pi05_l10_50_gm_corr0_b1` | empty grasp (aperture < .0010 after a 10-control close) | **1** policy call | 2 triggers |
| `r9f2c_pi05_l10_50_gm_corr0_b3m3` | empty grasp | **3**-call burst | 3 triggers |
| `r9f2c_pi05_l10_50_gm2_corr0` | empty grasp **or** long carry (close held 30 decisions = 150 controls) | 2-call burst | 3 triggers |
Reference arms on the same pairs: `gm_corr0` (burst 2, cap 2) .790 @ .0863; cache .740; policy .930.
Decision rule: pick the variant with the best SR at IR ≤ .10; if `gm2` adds ≥ 3 pp over `gm_corr0`, carry the
second trigger to the GR00T and 300-pair runs. Expected: b1 ≈ .77–.79 @ ≈ .082; b3m3 ≈ .79–.83 @ ≈ .095; gm2 ≈ .80–.84 @ ≈ .095.

```bash
cd /home/weiland/projects/openpi
P=(taskset -c 22-25 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src .venv/bin/python)
C=/home/weiland/trace_runs/os_closed_loop/r09_fable_r2c
ARMS=(r9f2c_pi05_l10_50_gm_corr0_b1 r9f2c_pi05_l10_50_gm_corr0_b3m3 r9f2c_pi05_l10_50_gm2_corr0)
WORKER_HOST=timan107 SYNC_PORT=23195 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control sync --concurrent "$C" "${ARMS[@]}"
WORKER_HOST=timan107 PORTS=23240,23241,23242,23243 WPS=12 MAX_ATTEMPTS=2 POLL_SECONDS=30 OSCL_MANIFEST=$C/manifests/eval100_inits20_29.json \
  "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control chain "$C" "${ARMS[@]}"
"${P[@]}" -m exp.offline_search.rounds.r09.explore_fable.round2.tools.paired_r2 --ref grown:r9f_ctrlA_p_l10_50 r2c:... # add ROOTS entry or pass the full root path as root:arm
```
(`paired_r2.py` accepts `<root path>:<arm>`; e.g. `/home/weiland/trace_runs/os_closed_loop/r09_fable_r2c:r9f2c_pi05_l10_50_gm_corr0_b1`.)

## 4. Next idea, not prepared (needs new serving code): cache-side open-and-retry for π0.5
On a flag, serve "open gripper + lift 3 cm" for one commit and invalidate the anchor so the next look re-retrieves
from the pre-grasp region, no policy call (IR unchanged). Aligned with the π0.5 policy's own response (2.1); not
aligned with GR00T's. Worth one 100-pair arm after the r2c sweep.
