# R4 input: what R2 and R3 established (2026-09-27 13:0x CDT)

Full reports: `rounds/r02/ANALYSIS.md` (32 pure-cache closed-loop arms), `rounds/r03/ANALYSIS.md` (pilots, full pure-cache
arms, 18 mixed HIT/MISS arms, SR-vs-IR frontier), ledger `logs/offline_search_exploration.log.md` §10 (every decision
and number). Cells: π0.5-sp / π0.5-l10 / GR00T-sp / GR00T-l10. Closed loop = LIBERO, A-pool 500 pruned inits per arm,
paired on the same inits; pure-inference reference SR .986 / .844 / .940 / .870 (single stochastic sample).

## Cost model (owner definition; π0.5 CUDA-graph stage latencies)
Stage 1 (vision encoder -> pooled keys) 10.26 ms, stage 2 (VLM prefix) 27.69 ms, stage 3 (action expert denoising,
10 steps) 29.57 ms; shares .152 / .410 / .438. Inference ratio IR = (N_req·s1 + N_miss·(s2+s3)) / (N_req·Σs)
= 0.152 + 0.848·MISS share. Every decision pays stage 1 today (the retrieval key); a HIT pays nothing else. GR00T's
split is not measured (measure it if you need it: plugin logs `s1_ms` / `s23_ms` per decision).

## What works (closed loop)
- **AWM** (action-whitened joint metric over [PCA-64 cam0, PCA-64 cam1, robot state 8], per-task closed-form
  whitening from action-similar cross-episode row pairs, kernel mean of the 16 nearest with bandwidth set by the 5th):
  pure cache 50-episode library .800 / .630 / .888 / .552 vs deployed B0 .668 / .440 / .736 / .468; 500-episode library
  .954 / .768 / .966 / .706. 580 B/entry vs 262 KB/entry deployed.
- **Library size** is the largest layer (+8…+16 pp for AWM from 50 to 500 episodes).
- **Mixed HIT/MISS** (full model runs on MISS), π0.5, IR = 0.152 + 0.848·MISS share:
  - spatial: V7 confidence + guards at h≈.67 -> .980 @ IR .43 (policy .986); periodic MISS every 3rd -> .938; guard-only
    .888 @ .27; h=.5 -> .986 @ .61.
  - l10 (50-ep library): V7+guards .816 @ .44; periodic k=3 .832 @ .43; periodic k=5 .792 @ .32; guard-only .740 @ .32;
    h=.5 .868 @ .60. Targeting does NOT beat periodic on l10 (it does on spatial, +4.2 pp).
  - l10 (500-ep library): guard-only **.864 @ IR .24** (= policy .844, paired +2.0 n.s.); V7+guards .872 @ .44.
  - B0 as selector −11.8 pp, B0 score as judge −8.2 pp (both dead).
  - Guards = MISS when: stuck (no motion ≥2 decisions), terminal library row picked with gripper closed, overtime∧lag,
    library progress not advancing 3 decisions.
- Pure-cache metric tweaks at 50 episodes: borrowed big-library metric prior α=.5 +4.8 / +4.4 pp (π0.5), ridge 1.0 +4.0
  pp (π0.5-sp, no borrowing); nothing significant on GR00T.

## Failure anatomy (why things fail)
- Pure-cache failures are all step-cap timeouts caused by deadlock "spells": robot stops -> observation unchanged ->
  retrieval returns the same row -> same action. Traps: grasp / release moments (split gripper vote in the kernel ->
  hover), terminal "hold" rows of library episodes, pause rows, hubs.
- In AWM's closed loop, consecutive picks stay in the same library episode 44–71 % of decisions but hit the exact
  successor row only 18–29 %; the same row repeats 15–40 % (stalls). The kernel mixes 3–4 library episodes.
- Offline action error does NOT rank closed-loop methods (Spearman across tasks −.44…+.62; every R3 proxy failed).
  100-init l10 pilots cannot see effects below ±7 pp (run-to-run noise: 6 % / 13 % / 2 % discordant episodes).

## Dead (measured in closed loop; do not re-propose)
Vision-free selection throughout the episode (R1); library-side recovery / trajectory exclusion (CL3); symmetric gripper
commitment (SR .10: blocks the grasp) and release guard (l10 SR .01: blocks the release); terminal-row masking in any
form (.23 / GR00T .60: the trap moves); early events + bursts in the mixed judge; B0 selector or B0 score judge in mixed
mode; mode-only (one gripper class) averaging; success-only libraries; T2 learned metrics (MLKR/MLP) — not worth it.

## Owner rulings in force for R4 / R5 (protocol §9 items 8, 9)
- Only codex agents are used for delegated work.
- The 500-episode library counts as deployable, but every conclusion must also have the 50-episode library experiment.
- A cheaper vision key is accepted (the policy's own vision encoder, but e.g. fewer layers / lower resolution than the
  full stage-1 output), with the library keys rebuilt the same way.
- **"Look once, act several steps"**: the vision-mandatory rule is relaxed — a decision with vision is still required as
  an anchor, and vision-free decisions in between are allowed if bounded and triggered back to vision by conditions
  (state, phase, time). Whole-episode blindness (R1) stays dead.
- LIBERO workers only on timan107 (48 cores; 64 workers = saturated; closed loop is the bottleneck). weilandserver's 4090
  hosts the servers and is shared with another project's training (it has priority; we yield).
- System measurements (throughput / latency under load) are out of scope for these rounds.
