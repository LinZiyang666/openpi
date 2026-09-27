# R2 analysis brief (fable analysis agent)

Read: `logs/offline_search_exploration.log.md` (protocol: §4 metrics, §6 step 5, §8, §9 owner rulings, §10 ledger —
R2 entries), `rounds/r02/SELECTION.md` (method list + owner rulings: vision mandatory, two closed-loop groups,
three-layer decomposition), `rounds/r02/NOTES_ideation_{A,B}.md`, each family's code / README under
`rounds/r02/g{1..4}_*/`, `rounds/r03/FINDINGS.md` (the coordinator's digest of R2 so far), `rounds/r03/NOTES_ideation_A.md`
(closed-loop trace mining of the π0.5 50-episode arms — reuse, do not redo; scripts in `rounds/r03/diag_A/`),
`harness/README.md`, `profile/README.md`, `closed_loop/README.md`.

Data:
- Offline: `exp/offline_search/results/r02/<method>/<cell>.{npz,json}` (+ `results/r00/` baselines B0–B4),
  `results/scoreboard.csv` (latest run_ts per method×cell). Store `/dev/shm/offline_search_store`.
- Closed loop (pure cache, A-pool 500 inits per arm, servers log every decision):
  `/home/weiland/trace_runs/os_closed_loop/r02_g50/` (50-episode library; arms `oscl50_{p,g}_{sp,l10}_cl{0..3}`) and
  `r02_g500/` (500-episode library; arms `oscl500_*`); per arm `runs/<arm>/summary.json`,
  `runs/<arm>/server_<port>/decisions_*.jsonl`, pulled journal / per_step; run summary `summary.json`; arm definitions
  `arms.json` (CL0 = B0 native / B0-formula top-1 over the 10× candidates, CL1 = B0 ranking + mean of top-5,
  CL2 = AWM, CL3 = CL2 + V6 stuck recovery with ot_still). Status: `runs/chain.log` (EV lines). Arms still running
  when you start are listed there; analyse what is complete and state what was missing.
- Pure-inference reference on the same inits (trace_dual): SR .986 / .844 / .940 / .870 (π0.5-sp / π0.5-l10 /
  GR00T-sp / GR00T-l10). History (π0.5, B0-style top-1, `exp/ablation_study/cache_size/analysis/analysis.md`):
  spatial S3 .688 → S6 .810, l10 S3 .456 → S6 .516.

Deliver (write `rounds/r02/ANALYSIS.md` and return the same text):
1. **Closed-loop table** (the exam): SR with Wilson 95 % CI per arm × cell for both groups, paired comparisons on the
   same inits (S→F / F→S counts, exact McNemar p) CL1 vs CL0, CL2 vs CL1, CL3 vs CL2, and 500 vs 50 per CL; per-task
   SR; decisions / episode; per-decision latency (q_us, infer_ms) and the native-agreement rate.
2. **Three-layer decomposition in closed loop and offline, side by side**: synthesis (CL1 − CL0), method at fixed
   library (CL2 − CL1), library (500 − 50 for each CL), with the offline stale / fresh err and AURC deltas of the same
   methods. Label every fit that uses data beyond the deployed candidate library as "borrowed big-library information".
3. **Offline ↔ closed-loop mapping**: where they agree / disagree, per model and suite; is GR00T's pattern the same as
   π0.5's (spells, trap classes — run `diag_A/load_cl.py` + `a2_stuck.py` style KPIs on the GR00T and 500-episode arms)?
4. **Library scale** for every row: episodes, entries, bytes/entry, fit pickle MB, total vs deployed pkl
   (π0.5 431 / 1103 MB, GR00T 429 / 1068 MB; 262 KB key per entry); both the current (~50 episodes) and 10× library.
5. **Per-method verdict** for the offline families (G1 AWM variants, G2 V4/V5/M8x, G3 V6/V7, G4 T2): keep / refine /
   drop with the deciding numbers, fresh and stale regimes, per model.
6. **Risks and what R3 should know** (R3 is already under way — read `rounds/r03/SELECTION.md`; say where the R2 data
   supports or contradicts its choices).

Rules: CPU only; `taskset -c <range in your prompt>`, ≤ that many processes, BLAS threads 1; read-only except
`rounds/r02/ANALYSIS.md` and scratch under `/home/weiland/.claude/jobs/a607dd74/tmp/analysis_r02/`; no git; no repo
edits elsewhere; no timan107, no GPU, never touch the running chains / servers. Be quantitative and skeptical (paired
tests on the same inits, CIs, regime identity, duplicate variants, library overlap with queries, noise floor).
