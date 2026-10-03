# R9 round 2 — hand-back (fable)

Written 2026-10-02 (CDT). Directory `exp/offline_search/rounds/r09/explore_fable/round2/`. Rules followed: inits 30–49 never
read (every loader filters `init < 30`; journals/forensics/server logs filtered in code); heads fitted on inits 0–19,
all evaluation on inits 20–29; no per-task switch at decision time; the other researcher's `round2/` was not read before
`PROPOSALS.md` was written. No git state changed; no existing file outside my directory edited (new files were pushed
to the h100 mirror of my own directory only).

## Deliverables
`REPORT.md` (owner section in Chinese first), `DATA_ANALYSIS.md`, `PROPOSALS.md`, this file, `tools/` (`corrector.py`,
`aperture.py`, `methods.py`, `paired_r2.py`, `tests/test_round2.py` — 6 tests pass), `out/corrector/` (heads + held-out
evaluations), `out/r2_paired.json` (closed-loop paired analysis).

## Closed loop that ran (my fleet)
Run root `/home/weiland/trace_runs/os_closed_loop/r09_fable_r2`: 11 arms × 100 pairs (tasks 0–9 × inits 20–29,
`manifests/eval100_inits20_29.json`), h100 ports 23240–23243, WORKER_HOST=timan107, WPS=12, one chain (tmux `r9f_r2`),
sync on reserved port 23195 (`sync --concurrent`). Arms: see `arms.json`; prefits in `fits/` (built with the plugin prefit
CLI, logs in `prefit_logs/`). Deployed to h100 (new files only): `exp/offline_search/rounds/r09/explore_fable/round2/
{__init__.py,tools/__init__.py,tools/methods.py}` (sha a4e1f253…).

Reproduce:
```
P="taskset -c 22-37,66-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src .venv/bin/python"
T2=exp.offline_search.rounds.r09.explore_fable.round2.tools
$P -m $T2.corrector --per-task ; $P -m $T2.corrector ; $P -m $T2.corrector --gripper      # heads + held-out tables
$P -m $T2.aperture                                                                          # empty-grasp detector study
bash /tmp/fable_r2_prefit.sh ; bash /tmp/fable_r2_gm.sh   # (kwargs in <run>/corr_kwargs.json, gm_kwargs.json)
$P -m exp.offline_search.closed_loop.ops.emit_arms --run-root $R2 --spec $R2/arms_in.json
WORKER_HOST=timan107 SYNC_PORT=23195 $P -m exp.offline_search.closed_loop.ops.h100.control sync --concurrent $R2 <arms>
WORKER_HOST=timan107 PORTS=23240,23241,23242,23243 WPS=12 OSCL_MANIFEST=$R2/manifests/eval100_inits20_29.json $P -m exp.offline_search.closed_loop.ops.h100.control chain $R2 <arms>
$P -m $T2.paired_r2 --ref grown:r9f_ctrlA_p_l10_50 r9f2_pi05_l10_50_corr1pt r9f2_pi05_l10_50_corr1single r9f2_pi05_l10_50_gm_corr1pt r9f2_pi05_l10_50_gm_corr0 grown:r9f_P10_p_l10
```

## Results at hand-back

**Status at hand-back (2026-10-02 00:0x CDT): chain `r9f_r2` (tmux, self-terminating) has finished 2 of 11 arms; the remaining 9 (grasp-miss trigger arms, GR00T L10-50, Spatial-50) complete over the next ~30 min. Read them with `paired_r2` (HANDBACK.md).**

| arm (π0.5 L10-50, inits 20–29, 100 pairs) | SR @ IR | paired vs cache .740 (same pairs/topology) |
|---|---|---|
| pure cache (round-1 control) | .740 @ .0765 | — |
| pure policy (round-1 control) | .930 @ .504 | +19 pp [+10, +27] |
| corrector **full strength**, per-task head fitted on 0–19 | **.630** @ .0762 | **−11 pp** [−23, 0], +14/−25, p = .11 |
| corrector full strength, single task-agnostic head | **.540** @ .076 | **−20 pp** |
| grasp-miss trigger (+ full corrector / + plain cache), GR00T, Spatial | pending | pending |

**Reading.** The full-strength corrector — the offline optimum on held-out inits (−31% gap) — is clearly harmful
closed loop on LIBERO-10 (−11 and −20 pp), while the half strength of round 1 was +1.4 pp. Offline distance to the
policy mispredicts closed loop for the third time this round (grown library, full corrector). The per-task pattern
(task 2: 1.0 → .6, task 3: 1.0 → .8, task 0: .4 → .6) repeats the grown-library pattern: a stronger pull toward the
policy helps the tasks the demos handle badly and breaks the tasks they handle well. Consequence: correction strength
must be shrunk (≤ .5) and the remaining LIBERO-10 gap cannot be closed by correcting the cached action — the
grasp-miss recovery call (observation-keyed) is the live candidate; its arms are the pending ones, plus batch 2
(half-strength base + trigger, no-progress + half corrector, trigger + uniform coin) already prefitted and emitted in
the same run root (`arms_in_batch2.json`, `arms_in_batch2b.json`).

Batch 2 (6 arms, prefitted + emitted, NOT yet synced/launched; run after chain 1 ends): `r9f2_{pi05,groot}_l10_50_corr05pt`, `r9f2_{pi05,groot}_l10_50_gm_corr05pt`, `r9f2_pi05_l10_50_np_corr05pt`, `r9f2_pi05_l10_50_gm_corr05pt_pu15` (drop `np_corr1pt` and `gm_corr1pt_pu15`: full-strength bases). Commands: `WORKER_HOST=timan107 SYNC_PORT=23195 $P -m ...control sync --concurrent $R2 <batch2 arms>` then the same chain command with those arms.
