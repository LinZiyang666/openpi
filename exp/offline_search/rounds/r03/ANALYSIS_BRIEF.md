# R3 analysis brief (fable analysis agent)

Read: `logs/offline_search_exploration.log.md` (protocol §8, §9 owner rulings, §10 ledger — the R3 entries are the
coordinator's running record of every decision and pilot result), `rounds/r03/SELECTION.md`, `rounds/r03/FINDINGS.md`,
the ideation originals `rounds/r03/NOTES_ideation_{A,B}.md` + `ideation_C/REPORT.md`, `rounds/r03/CODING_BRIEF.md`, the
four family READMEs (`h1_trap/README.md`, `h2_mixed/README.md`, `h3_judge/README.md`, `h4_kpi/r02_kpi.md`),
`closed_loop/README.md` (incl. "Mixed HIT/MISS mode" and "KPI tool"), and `rounds/r02/ANALYSIS.md` (the R2 closed-loop
reference, incl. its 500-episode addendum).

Data:
- Offline R3: `exp/offline_search/results/r03/<method>/<cell>.{npz,json}` (H1 AWM3 variants, H3 MXJ variants);
  R2 reference `results/r02/`, `results/scoreboard.csv`.
- Closed loop (all under `/home/weiland/trace_runs/os_closed_loop/`):
  - `r02_g50/`, `r02_g500/` — R2 pure-cache arms (50- and 500-episode libraries; CL0–CL3 × 4 cells).
  - `r03_pilot/` — 100-episode pilots (π0.5 trap tasks sp {6,9,0,4,1} / l10 {0,4,6,8,7}, GR00T-sp {8,2,3,7,5}; ep_idx
    0–19); arms with `state/<arm>.SKIPPED` were skipped on purpose (reason inside).
  - `r03_full/` — 500-episode pure-cache arms of the pilot winners (borrowed prior α=.5 `a05`, non-borrowing ridge 1.0
    `rm1`, possibly release-guard `a05gr`).
  - `r03_mx/` — mixed HIT/MISS arms (full model on MISS): B0 quantile h=.7 control, MixedJudge V7+guards (h .7/.5),
    V7+guards+events+burst (h .7/.5), guard-only, periodic k=3 control; π0.5 sp and l10, 50-episode library, AWM base.
  - Pure-inference reference on the same inits (trace_dual): SR .986 / .844 / .940 / .870, 21.6 / 58.8 / 22.7 / 58.1
    decisions per episode.
- Tool: `taskset -c <your range> .venv/bin/python -m exp.offline_search.closed_loop.ops.kpi --run-root … <arms> --ref
  <run:arm> [--pilot | --tasks … --episodes 0-19] --recon AWM3=awm:5 --json … --md …` (validated against A/C numbers;
  mixed-mode block: realized h, IR, MISS share in failed episodes, reason codes, first MISS vs first spell).

Deliver `rounds/r03/ANALYSIS.md` (English) and return the same text:
1. **Pure-cache results**: pilot table (paired vs R2 and vs the pilot's own CL2 re-run; the run-to-run noise floor is
   ~6 % discordant pairs on spatial, ~13 % on l10 — quantify it), full 500-episode table (paired vs R2 CL2 of the same
   cell), per-task and trap-class KPIs; verdict per switch (borrowed prior α, ridge, gripper commitment symmetric /
   release-guard, terminal guard). Explain the two collapses mechanistically (symmetric commit blocks the grasp;
   terminal masking moves the trap) with log evidence.
2. **Mixed-mode results**: per arm realized h, IR (π0.5: 0.152 + 0.848·MISS share), SR with Wilson CI, paired vs pure
   cache CL2 (h=1) on the same inits; the SR-vs-IR frontier with the pure-inference and pure-cache endpoints; judge
   comparison at matched realized h (B0 score vs V7+guards vs +events vs guard-only vs periodic control) — is targeting
   worth anything over periodic MISS? MISS timing (first MISS vs first spell), reason-code shares, interrupted
   successful episodes, s1/s23 latency.
3. **Four-layer decomposition**: synthesis / method at fixed library / library / control (handoff) effect, with the
   policy's inference counted as cost; label every "borrowed big-library information" fit.
4. **Library scale** for every configuration (episodes, entries, bytes/entry, fit pickle MB vs deployed pkl 431 / 1103
   / 429 / 1068 MB) at the 50- and 500-episode libraries.
5. **What R4 should do** (ranked, concrete), and what is now settled / dead.

Rules: CPU only; `taskset -c <range in your prompt>`, ≤ that many processes, BLAS threads 1; read-only except
`rounds/r03/ANALYSIS.md` and scratch `/home/weiland/.claude/jobs/a607dd74/tmp/analysis_r03/`; no git, no GPU, no
timan107, never touch running chains / servers / tmux. Be quantitative and skeptical: paired tests on the same inits,
CIs, noise floor, realized vs target h, and whether an SR gain is just more policy inference.
