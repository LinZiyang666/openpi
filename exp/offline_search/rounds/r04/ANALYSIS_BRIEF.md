# R4 analysis brief (fable analysis agent; owner ruling §9 item 15)

Read first: `logs/offline_search_exploration.log.md` — §8 protocol, §9 owner rulings (items 9–15; for R4/R5: the
500-episode library is deployable but every conclusion also at 50; cheaper vision keys allowed; "look once, act several
steps" allowed with bounded blind stretches and mandatory vision anchors; search latency measured as a pinpoint
exception; **MISS step reduction (K2) is NOT part of the system**, appendix only), and the whole §10 R4 ledger (the
coordinator's running record of every decision, anomaly and result, with timestamps in CDT). Then
`rounds/r04/SELECTION.md`, `CODING_BRIEF.md`, the ideation reports `ideation_{A,B,C}/REPORT*.md`, every hand-back
`k1_blind/ k2_serving/ k3_cost/ k4_eval/ k5_rand/ k6_concurrency/ k7_guard/ k10_policy_tail/HANDBACK.md`,
`k8_search_latency/REPORT.md`, `k9_gpu_retrieval/REPORT.md`, `rounds/r03/ANALYSIS.md`. R5 was already planned from the
ledger while R4 ran: `rounds/r05/FINDINGS.md`, `SELECTION.md`, `ideation_{A,B,C,D}/REPORT.md`, `q3_callvalue/` (Q3
= K5 follow-up, conclusion "call everywhere") — read them so §10 below can say whether the R4 evidence supports those
choices.

## Data (all under /home/weiland/trace_runs/os_closed_loop/; every arm is 500 paired episodes, 10 tasks × inits 0–49)
| run root | content |
|---|---|
| `r02_g50`, `r02_g500` | R2 pure-cache references (CL2 AWM), both models/suites |
| `r03_mx` | R3 mixed arms (`r3mx_p_l10_g` 50-lib guard-only .740@.323, `r3mx_p_l10_g500` .864@.238, perk5, perk3, V7 …) |
| `r04_frontier` | batch 1: l10 500 guard noprog4, periodic 8/12; l10 50 guard noprog4, periodic 6; sp 500 guard-only, periodic 12. The four `r4_p_*_inf_s1/s2` seed arms are DEFERRED with no data (replaced by `r04_cost` s1001/s2001) |
| `r04_cost` | K10 seeded pure inference `r4f_p_{l10,sp}_inf_s1001` and `_s2001`; K10 L=10 pure inference `r4f_p_{l10,sp}_inf_k10_L10` (seed 3001); K2 arms (appendix only, owner ruling 12); SKIPPED rows have no data |
| `r04_blind` | batch 3, K1 blind methods with the K1 *dense* stuck guard: l10 500 ph2g, b0g, ph1g, tail1ug, clk1g (kernel_clock B=1), ph2k8 (two-clock, periodic 8); pure-cache blind (no policy) ph2c l10 500 / l10 50 / sp 500, **tail1uc sp 500 (pure-cache anchor_tail, stage-1-only server, zero MISS)**; l10 50 ph2k5 (two-clock); sp 50 ph2g; `r4b3_p_l10_50_inferL10` = second, unseeded pure-inference L=10 run. `50_ph2g` / `50_b0g` SKIPPED (replaced by K7) |
| `r04_k7` | K7 vision-confirmed stuck guard: l10 500 b0g/ph2g/ph1g/tail1ug, l10 50 ph2g/tail1ug, sp 500 ph2g/tail1ug (the sp 50 tail1ug cell is queued in R5, not available) |
| `r04_b4w` | wrist-only stage-1 key, guard-only, K10 MISS: l10/sp × 50/500 |
| `r04_b4k` | all SKIPPED (K2 stacking, owner ruling 12); `r04_b4` is empty |
| `r04_csl` | control_step_library G/GS, l10 × 50/500 (pure cache) |
| `r04_rep` | two replicates `r4rep_p_l10_g500_{a,b}` of stock `r3mx_p_l10_g500` (noise floor) |
| `r04_k5` | randomized CALL/CACHE (K5), l10 guard-only, 50 and 500 × replicates r1/r2 |
| `r04_gblind` | GR00T pure-cache blind: {ph2, tail2u} × {l10, sp} × {50, 500} — **STILL RUNNING** (see below) |
| `r04_bsmoke`, `r04_k5_smoke` | smokes (not for conclusions) |

Pure-inference references: trace_dual SR .986 / .844 / .940 / .870 (π0.5-sp / π0.5-l10 / GR00T-sp / GR00T-l10), the R4
seeded K10 arms (L=5), and the two L=10 runs (l10 .904 seeded, .900 unseeded; sp .986).

**Not yet available when you start** (the coordinator will message you when they land; write everything else first,
leave clearly marked slots for these and fill them in on the follow-up): the 8 `r04_gblind` GR00T arms, and
`r4f_p_l10_inf_s2001` / `r4f_p_sp_inf_s2001`. Check `<run>/state/<arm>.DONE` and `<run>/summary.json` before using an arm.

**K10 policy tail was coded but NOT run in closed loop**: the inherited K1 `blind_step` vetoes it inside a no-progress
span, which covers 88% / 94% of l10 MISSes (50 / 500 library; ideation A's finding). R5 Q1 (C10 CommitJudge,
`policy_tail_gate="lifecycle"`) fixes this and is queued. Report the K10 hand-back as engineering + the veto finding.

## Cost bases (report both, never mix; owner basis is primary)
- Owner basis (π0.5 CUDA graph): vision decision .152, MISS +.848 (K10 full inference); blind decision 0; wrist-only /
  dummy_cached stage 1 priced by `rounds/r04/cost_table_owner.json` (ratio-transfer ASSUMPTION — label it; wrist:
  .0552·v + m·(.848 + .0499)). GR00T owner basis: historical .148 / .174 / .678 (same file). IR per 5 control steps;
  L=10 pure inference = .5.
- **The watcher / summary "IR" for R4 arms is the eager ledger** — recompute owner IR yourself from `cost_ledger` v, m
  in each `summary.json`: IR = .152·v + .848·m.
- Eager basis (K3 measured, `closed_loop/ops/cost_table.json`): report separately.
- Search cost (K8): method-only CPU ms per vision / blind decision from `k8_search_latency/REPORT.md` (controlled
  table), added per decision as a third column "owner IR + search"; K9 measured the GPU-resident in-graph increment
  (.36–.63 ms, top-1 ≥99.9%) — say how that changes the column.
- dummy_cached (bit-exact, K3): no closed-loop arm; reprice vision decisions of the relevant arms analytically and say so.
- K2 arms: appendix only (K2 vs K10 SR indistinguishable), never on the main frontier.

## Tools
`taskset -c 26-29,70-73 .venv/bin/python -m exp.offline_search.closed_loop.ops.kpi --run-root <R> [--run-root …]
<arms or run:arm> --ref <run:arm> [--ledger] [--cost-table rounds/r04/cost_table_owner.json] [--manifest …] --json …
--md …` (paired comparison vs `--ref`, bootstrap); K5 estimator `rounds/r04/k5_rand/estimate.py` (see its hand-back;
separately per library scale, `--baseline r3mx_p_l10_g / r3mx_p_l10_g500`). Journals: `task_uid` is
`<arm>:eval:<task>:<init>`; use exact McNemar on discordant pairs. Summaries: `<run>/summary.json`.

## Deliver `rounds/r04/ANALYSIS.md` (English) and return a one-paragraph summary
1. **Noise floor first**: stock g500 three runs (.864 original, .850, .832 replicates): run-to-run SR spread and paired
   discordance on l10; use it for every comparison (paired McNemar / bootstrap on the same inits, Wilson CIs). State
   which R4 "wins" survive against the mean .849, not against .864.
2. **Execution-length effect**: l10 pure inference L=10 (.904 / .900) vs L=5 (seeded K10 arms, trace_dual .844) with
   per-task paired counts; spatial null. Separate "execute the whole chunk" from "save vision" in every blind result
   (anchor_tail executes steps 5–9 of the synthesized chunk). This is the most important confound of R4.
3. **SR-vs-IR frontier** per cell (π0.5 l10 and spatial; both library scales; GR00T pure-cache blind), owner basis
   primary, with pure inference at L=5 and L=10, pure cache, R3 best points and every R4 arm; mark dominated points;
   second panel with search cost added; third with eager basis.
4. **Blind stepping ("look once, act several steps")**: phase_particles B=1/B=2 vs kernel_clock vs anchor_tail, gated
   vs budget-only vs two-clock; the K1 dense-guard failure (stuck fires ~2.8×; b0g .842) and the K7 fix (B=0 bit-equal
   to stock); realized vision share, blind age, look-reason mix, MISS reasons; why phase_particles loses on the
   50-episode library (.700) while anchor_tail holds (.806 @ .242); pure-cache blind vs R2 CL2 (same SR, 36–46% less
   IR); **pure-cache anchor_tail sp 500 .982 @ .078 with zero MISS vs pure inference .992 (p=.27)** — check it hard
   (per-task, per-init discordance, whether any episode used the policy); GR00T blind.
5. **Cheaper vision (wrist-only key)**: paired tests vs the two-camera guard-only arms (sp50 .924 vs .888 p=.027 from
   tasks 4 and 9; l10 500 worse p=.021; others equal) — is sp50 a real effect or a multiple-comparison artefact (four
   cells tested)? Exact dummy_cached repricing.
6. **Control-step library (G/GS)** vs CL2 (G collapses .222 / .114; GS neutral .772 / .628): why.
7. **Randomized CALL/CACHE (K5) and Q3**: ITT and per-landmark/context ΔY, ΔN, ΔM with clustered intervals, ΔCρ at the
   g50/g500 operating points (g500 ΔY +.034 [+.004, +.066] cost-neutral; g50 ΔY −.002 [−.040, +.032]); Q3 found no
   supported saving context — what this implies for any learned gate.
8. **Four-layer decomposition** (synthesis / method at fixed library / library / control incl. execution length) plus
   the cost-implementation layer; label borrowed big-library information.
9. **Library scale and bytes** for every configuration (entries, bytes/entry, fit pickle MB vs deployed 431 / 1103 /
   429 / 1068 MB).
10. **Engineering findings**: R4 serving serialization (K6, timing only), search latency (K8: CPU method 1.2–2.2 ms per
    decision, PCA 31–49%), GPU-resident retrieval in the stage-1 graph (K9), K10 policy-tail serving path and its veto.
11. **R5 check and R6 proposals**: for each R5 selection (Q1 C10 + D1, Q2 GR00T CycleTail, Q3 call-everywhere, Q4
    library growth + demo scaling curve, Q5 GPU shadow, Q6 wrist + blind guard, B1 constrained offline solver for
    hyperparameters, owner §9 item 10) say whether R4 evidence supports it; what is settled / dead; ranked concrete
    ideas beyond R5.

Rules: CPU only; `taskset -c 26-29,70-73`, ≤ 8 processes, BLAS threads 1 (OMP/OPENBLAS/MKL_NUM_THREADS=1); read-only
except `rounds/r04/ANALYSIS.md` and scratch `/home/weiland/.claude/jobs/a607dd74/tmp/analysis_r04/`; no git, no GPU, no
timan107; never touch running chains / servers / tmux / relays (the coordinator runs all experiments). Do not read
`tests/review_tests/`. Be quantitative and skeptical: paired tests on the same inits, CIs, the measured noise floor,
realized vs intended shares, whether an SR gain is just more policy inference or longer execution.
