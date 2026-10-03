# Round 2c (fable) — cache-side open-and-retry; frozen arms in r09_fable_r2d

Rules kept: inits 0–29 only; thresholds/maps calibrated on inits 0–19, sanity on 20–29; nothing per task; explore_astra/round2* and
explore_opus not read; no chain launched by me. New serving code = new classes appended to `round2/tools/methods.py`
(`GraspMissReanchor`, `GraspMissCallsSigned`; the earlier classes are byte-identical); 11 unit tests pass; CPU plugin selftest of
`GraspMissReanchor` (blind serving) PASS. Module pushed to the h100 mirror of my directory (sha 86774b83…).

## 0. Correction to report first: GR00T gripper sign
Verified on recorded apertures (inits < 30): π0.5 normalized gripper **+1 = close**; GR00T normalized gripper **+1 = open, −1 =
close** (wire is +1 = close for both; the GR00T wire map flips the sign). Consequences: (i) my GR00T per-phase table in
DATA_ANALYSIS §1 has the phase names inverted for GR00T (numbers unchanged); (ii) the GR00T `GraspMissCalls` arms of chain 1 and
batch 2 (`r9f2_groot_l10_50_gm_corr1pt/gm_corr0`, `r9f2_groot_l10_50_gm_corr05pt`) test "closed" with π0.5's sign and never
trigger — they equal their bases (confirmed: gm_corr1pt .690 vs corr1pt .700). Sign-aware replacements are in r2d (below). The
π0.5 arms (gm_corr0 .790 etc.) are unaffected. The aperture detector study itself used wire commands and is correct for both.

## 1. Cache-side open-and-retry (`GraspMissReanchor`, no policy call, owner IR unchanged)
At a look decision whose history shows the close command held ≥ 2 decisions while the finger aperture < threshold (fingers met
nothing), the served 10-control commit is replaced by a **retreat**: the last 2 executed decisions replayed in reverse with negated
motion in wire units (affine normalized→wire map fitted exactly, residual 0, from recorded served/wire pairs, inits < 30) and the
gripper commanded open; the anchor is set to that chunk so the blind decision serves its tail, and the next decision is a fresh look
from the retreated open-gripper state → retrieval re-attempts the grasp. Cap 2 retries/episode, cooldown 3 decisions. Alignment
with the policy: after an empty grasp π0.5's own shadow opens the gripper (66% of the next 10 controls) and re-closes ~2 decisions
later — this is what the retreat does; GR00T's policy keeps the gripper closed and re-positions (26% open), so for GR00T the
open-and-retry is a hypothesis with weaker support.
Union variant (`rru`): triggers empty grasp **or** long carry (close held 30 decisions = 150 controls; 0 false alarms on held-out
successes) **or** stall (normalized end-effector path over 6 decisions below p1 of successful windows on inits 0–19: .002 π0.5 /
.017 GR00T; held-out 20–29: flags 1/27 & 0/73 (π0.5), 10/38 & 0/62 (GR00T)); carry/stall retreat with the gripper state kept
(dropping a held object would be worse); cap 3.

**Offline sanity on recorded pure-cache histories, inits 20–29 (the method's own `flag`/`retreat_chunk` on the logged executed
chunks and states):**
| cell | failing episodes with ≥ 1 retreat | successful episodes with ≥ 1 retreat | retreats per failing / successful episode | triggers in failures: empty / carry / stall | retreat translation per control vs typical served |
|---|---|---|---|---|---|
| π0.5 L10-50 | 23 / 27 | 1 / 73 | 2.33 / 0.03 | 60 / 3 / 0 | .307 vs .277 (normalized) |
| GR00T L10-50 | 36 / 38 | 4 / 62 | 2.74 / 0.06 | 78 / 14 / 12 | .284 vs .232 |
(These counts are on the *recorded* trajectories, where nothing was retried; closed loop the first retreat changes the rest.)

## 2. Frozen arms, run root `/home/weiland/trace_runs/os_closed_loop/r09_fable_r2d` (emitted, prefitted, `control plan` passes: 57 files, 15.9 GiB logical, mostly on h100 already; NOT launched)
Manifest tasks 0–9 × inits 20–29 (`manifests/eval100_inits20_29.json`); all bases are the plain cache (blend 0) unless noted.
| arm | class | trigger(s) | response | cap | IR |
|---|---|---|---|---|---|
| `r9f2d_pi05_l10_50_rr_corr0` | GraspMissReanchor | empty grasp (ap < .0010, closed_sign +1) | retreat + open, fresh look | 2 | = cache |
| `r9f2d_pi05_l10_50_rru_corr0` | GraspMissReanchor | empty ∪ long carry 30 ∪ stall (.002) | retreat (open only for empty) | 3 | = cache |
| `r9f2d_groot_l10_50_rr_corr0` | GraspMissReanchor | empty (ap < .0009, closed_sign −1) | retreat + open | 2 | = cache |
| `r9f2d_groot_l10_50_rru_corr0` | GraspMissReanchor | empty ∪ carry 30 ∪ stall (.017) | retreat | 3 | = cache |
| `r9f2d_groot_l10_50_gmS_corr0` | GraspMissCallsSigned | empty (sign-corrected) | 2-call policy burst | 2 | ≈ cache + .01 |
| `r9f2d_groot_l10_50_gmS_corr1pt` | GraspMissCallsSigned on the FULL-strength corrector (GR00T corr1pt = .700 vs cache .560 on these pairs) | empty (sign-corrected) | 2-call burst | 2 | ≈ +.01 |
References on the same 100 pairs: π0.5 cache .740 / `gm_corr0` .790 @ .0863 / policy .930; GR00T cache .560 / corr1pt .700.
Decision rule: a re-anchor arm is worth extending to 300 pairs if it beats its cache by ≥ 5 pp paired at unchanged IR; the GR00T
sign-corrected call arms replace the invalid chain-1/batch-2 GR00T trigger arms and are judged like `gm_corr0` on π0.5 (≥ +5 pp at ≤ +.03 IR).

```bash
cd /home/weiland/projects/openpi
P=(taskset -c 22-25 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src .venv/bin/python)
D4=/home/weiland/trace_runs/os_closed_loop/r09_fable_r2d
ARMS=(r9f2d_pi05_l10_50_rr_corr0 r9f2d_pi05_l10_50_rru_corr0 r9f2d_groot_l10_50_rr_corr0 r9f2d_groot_l10_50_rru_corr0 r9f2d_groot_l10_50_gmS_corr0 r9f2d_groot_l10_50_gmS_corr1pt)
WORKER_HOST=timan107 SYNC_PORT=23195 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control sync --concurrent "$D4" "${ARMS[@]}"
WORKER_HOST=timan107 PORTS=23240,23241,23242,23243 WPS=12 MAX_ATTEMPTS=2 POLL_SECONDS=30 OSCL_MANIFEST=$D4/manifests/eval100_inits20_29.json \
  "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control chain "$D4" "${ARMS[@]}"
"${P[@]}" -m exp.offline_search.rounds.r09.explore_fable.round2.tools.paired_r2 --ref grown:r9f_ctrlA_p_l10_50 $D4:r9f2d_pi05_l10_50_rr_corr0 $D4:r9f2d_pi05_l10_50_rru_corr0
"${P[@]}" -m exp.offline_search.rounds.r09.explore_fable.round2.tools.paired_r2 --ref r9f2_groot_l10_50_corr0 $D4:r9f2d_groot_l10_50_rr_corr0 $D4:r9f2d_groot_l10_50_rru_corr0 $D4:r9f2d_groot_l10_50_gmS_corr0 $D4:r9f2d_groot_l10_50_gmS_corr1pt
```
(The h100 chain hashes its code tree at start; `round2/tools/methods.py` on h100 is the current local file.)

## 3. Live results added this round (RESULTS_LIVE.md)
GR00T L10-50 (inits 20–29): cache .560 @ .0742; corrector FULL strength (per-task heads fitted on 0–19) **.700 (+14 pp)** — the
opposite sign to π0.5 (−11 pp), consistent with GR00T's cache gap being 10× its policy noise; gm_corr1pt .690 (inverted trigger,
equals its base). Remaining chain-1 arms (GR00T gm_corr0 — invalid as a trigger test; Spatial-50 cache/corr1pt/gm_corr1pt), batch 2
and r2c are under the coordinator; score with `paired_r2.py` as they land.
