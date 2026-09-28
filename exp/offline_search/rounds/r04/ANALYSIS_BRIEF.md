# R4 analysis brief (codex analysis agent)

Read first: `logs/offline_search_exploration.log.md` — §8 protocol, §9 owner rulings (items 9–12 are new for R4/R5:
500-library deployable but every conclusion also at 50; cheaper vision keys allowed; "look once, act several steps"
allowed with bounded blind stretches and mandatory vision anchors; search latency measured as a pinpoint exception;
**MISS step reduction (K2) is NOT part of the system**), and the whole §10 R4 ledger (the coordinator's running record
of every decision, anomaly and result). Then `rounds/r04/SELECTION.md`, `CODING_BRIEF.md`, the ideation reports
`ideation_{A,B,C}/REPORT*.md`, every hand-back `k1_blind/ k2_serving/ k3_cost/ k4_eval/ k5_rand/ k6_concurrency/
k7_guard/HANDBACK.md`, `k8_search_latency/REPORT.md`, `k9_gpu_retrieval/REPORT.md` (if present), `rounds/r03/ANALYSIS.md`.

## Data (all under /home/weiland/trace_runs/os_closed_loop/; every arm is 500 paired episodes, 10 tasks × inits 0–49)
| run root | content |
|---|---|
| `r02_g50`, `r02_g500` | R2 pure-cache references (CL2 AWM), both models/suites |
| `r03_mx` | R3 mixed arms (g = 50-lib guard-only .740@.323, g500 .864@.238, perk5, perk3, V7 …) |
| `r04_frontier` | batch 1: l10 500 guard noprog4, periodic 8/12; l10 50 guard noprog4, periodic 6; sp 500 guard-only, periodic 12 (4 DEFERRED seed arms have no data) |
| `r04_cost` | K2 arms (appendix only, owner ruling 12), seeded K10 pure inference (s1001/s2001), L=10 K10 pure inference; SKIPPED rows have no data |
| `r04_blind` | batch 3, K1 blind methods with the K1 *dense* stuck guard: ph2g, b0g, ph1g, tail1ug (500) … plus the deferred pure-cache blind arms (ph2c, clk1g, ph2k8, 50: ph2k5, ph2c; sp: ph2c, tail1uc, 50 ph2g; inferL10). 50_ph2g/50_b0g SKIPPED (replaced by K7) |
| `r04_k7` | K7 vision-confirmed stuck guard: l10 500 b0g/ph2g/ph1g/tail1ug, l10 50 ph2g/tail1ug, sp 500 ph2g/tail1ug |
| `r04_b4k` | all SKIPPED (K2 stacking, owner ruling 12) |
| `r04_b4w` | wrist-only stage-1 key, guard-only, K10: l10/sp × 50/500 |
| `r04_csl` | control_step_library G/GS, l10 × 50/500 (pure cache) |
| `r04_gblind` | GR00T pure-cache blind: {ph2, tail2u} × {l10, sp} × {50, 500} |
| `r04_rep` | two replicates of stock `r3mx_p_l10_g500` (noise floor / reference) |
| `r04_k5` | randomized CALL/CACHE (K5), l10 guard-only, 50 and 500 × replicate 1/2 |
| `r04_bsmoke`, `r04_k5_smoke` | smokes (not for conclusions) |
Pure-inference references: trace_dual SR .986 / .844 / .940 / .870 (π0.5-sp / π0.5-l10 / GR00T-sp / GR00T-l10) plus
R4 seeded K10 arms.

## Cost bases (report both, never mix; owner basis is primary)
- Owner basis (π0.5 CUDA graph): vision decision .152, MISS +.848 (K10); blind decision 0; wrist-only / dummy_cached
  stage 1 priced by `rounds/r04/cost_table_owner.json` (ratio-transfer ASSUMPTION — label it). IR per 5 control steps.
- Eager basis (K3 measured, `closed_loop/ops/cost_table.json`): report separately.
- Search cost (K8): method-only CPU ms per vision / blind decision from `k8_search_latency/REPORT.md` (controlled
  table), added per decision as a third column "owner IR + search"; say that the GPU-resident design (K9) would drive
  it toward zero.
- dummy_cached (bit-exact, K3): no closed-loop arm; reprice vision decisions of the relevant arms analytically and say so.
- K2 arms: appendix only (K2 vs K10 SR indistinguishable), never on the main frontier.

## Tool
`taskset -c <your range> .venv/bin/python -m exp.offline_search.closed_loop.ops.kpi --run-root <R> [--run-root …]
<arms> --ref <run:arm> [--ledger] [--cost-table rounds/r04/cost_table_owner.json] --json … --md …`; K5 estimator:
`rounds/r04/k5_rand/estimate.py` (see its hand-back; separately per library scale, with `--baseline r3mx_p_l10_g /
r3mx_p_l10_g500`). Collect summaries: `<run>/summary.json` (`cost_ledger` has v, m, K, L).

## Deliver `rounds/r04/ANALYSIS.md` (English) and return the same text
1. **Noise floor first**: from `r04_rep` (two stock g500 replicates) + the original g500, the run-to-run SR spread and
   paired discordance on l10; use it for every comparison (paired McNemar/bootstrap on the same inits, Wilson CIs).
   State which R4 "wins" survive (e.g. K7 ph2g .862 @ .178 vs g500; K1 tail1ug .878 @ .223; K7 b0g .830).
2. **SR-vs-IR frontier** per cell (π0.5 l10 and spatial; both library scales; GR00T pure-cache blind), owner basis
   primary, with pure inference, pure cache, R3 best points and every R4 arm; mark dominated points; second panel with
   search cost added; third with eager basis.
3. **Blind stepping ("look once, act several steps")**: phase_particles B=1/B=2 vs kernel_clock vs anchor_tail, gated
   vs budget-only; the K1 dense guard failure (stuck fires ~2.8×; b0g .842 @ .271 vs K7 b0g .830 @ .245) and the K7
   fix; realized vision share, blind age, look-reason mix, MISS reasons; why the 50-library blind arm loses (library
   density) while the 500-library one wins; pure-cache blind vs R2 CL2; GR00T blind.
4. **Cheaper vision (wrist-only key)** and exact dummy_cached repricing.
5. **Control-step library (G/GS)** vs CL2.
6. **Randomized CALL/CACHE (K5)**: ITT and per-landmark/context effects ΔY, ΔN, ΔM with clustered intervals, ΔCρ at
   the g50/g500 operating points; whether any context shows a supported effect; what this implies for a learned gate.
7. **Four-layer decomposition** (synthesis / method at fixed library / library / control) plus the cost-implementation
   layer; label borrowed big-library information.
8. **Library scale and bytes** for every configuration (entries, bytes/entry, fit pickle MB vs deployed 431 / 1103 /
   429 / 1068 MB).
9. **Engineering findings**: R4 serving serialization (K6, timing only), search latency (K8), GPU-resident retrieval
   (K9).
10. **What R5 should do** (ranked, concrete), including the owner's R5 directions: offline solving of hyperparameters
    (LOTO/LOEO, closed form, as RIT/LDA were solved; §9 item 10), GPU-fused retrieval, a wrist-only vision-confirmed
    guard for blind + wrist, library growth/pruning, GR00T mixed mode; and what is settled / dead.

Rules: CPU only; `taskset -c <range in your prompt>`, ≤ that many processes, BLAS threads 1; read-only except
`rounds/r04/ANALYSIS.md` and scratch `/home/weiland/.claude/jobs/a607dd74/tmp/analysis_r04/`; no git, no GPU, no
timan107; never touch running chains / servers / tmux. Be quantitative and skeptical: paired tests on the same inits,
CIs, the measured noise floor, realized vs intended shares, and whether an SR gain is just more policy inference.
