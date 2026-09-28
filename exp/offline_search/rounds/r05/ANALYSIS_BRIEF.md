# R5 analysis brief (codex analysis agent; owner ruling §9 item 15)

Read first: `logs/offline_search_exploration.log.md` — §8 protocol, §9 owner rulings (items 9–15: 500-episode library
deployable but every conclusion also at 50; "look once, act several steps" with bounded blind stretches and mandatory
vision anchors; MISS step reduction (K2) is NOT part of the system; owner R5 direction item 10 = solve hyperparameters
offline, LOTO/LOEO or closed form, as RIT/LDA were), and the §10 ledger from the R5 entries on (every decision, smoke,
result and anomaly, times in CDT). Then `rounds/r05/SELECTION.md` (including the two addenda at the end — the final
queue), `FINDINGS.md`, `CODING_BRIEF.md`, `ideation_{A,B,C,D}/REPORT.md`, every hand-back
`q1_commit/ q2_groot/ q3_callvalue/ q4_growth/ q5_gpu/ q6_wrist_blind/HANDBACK.md`, and **`rounds/r04/ANALYSIS.md`** —
R5 builds directly on its conclusions (noise floor, execution-length confound, frontier, §11 R5 check). Do not
re-derive R4; cite it and extend it.

## Data (all under /home/weiland/trace_runs/os_closed_loop/; 500 paired episodes per arm = 10 tasks × inits 0–49, unless noted)
| run root | arms | question |
|---|---|---|
| `r05_ptail` | `r5t_p_l10_500_tail1uc`, `r5t_p_l10_50_tail1uc`, `r5t_p_sp_50_tail1uc` (π0.5 pure-cache anchor_tail, stage-1-only servers, zero MISS); `r5t_p_sp_g50_wrist_rep` (replicate of R4 wrist sp50 guard-only) | cache-side execution-length effect on l10 and at 50; does the R4 wrist sp50 p=.027 replicate? |
| `r05_growth` | `r5q4_p_{l10,sp}_grow250_{refit,frozen}` — **250-pair manifest, inits 25–49 only** (`rounds/r05/q4_growth/evaluation_pairs.json`) | library grown with 250 paid policy episodes (inits 0–24); compare with R2 CL2-50 / CL2-500 restricted to the same pairs |
| `r05_demo_curve` | `r5q4d_p_{l10,sp}_{100,200,300}_refit_tail1`, `r5q4d_p_{l10,sp}_200_frozen50_tail1` | demo-data scaling curve with the pure-cache anchor_tail controller (kref 5); endpoints: 50 = `r05_ptail` 50 arms (current library, not in bpool_cs), 500 = `r05_ptail` / R4 `tail1uc` (kref 8) — state both caveats |
| `r05_q1` | C10 CommitJudge `r5q1_c10_p_{l10,sp}_{500,50}`; D1 GraspCheck `r5q1_d1_p_l10_{500,50}` | policy-side commit (execute the policy's own steps 5–9 after a MISS, lifecycle gate) and grasp check |
| `r04_k7` | `r4k7_p_sp_50_tail1ug` (R4 cell filled in R5) | K7 tail at sp-50 |
| `r05_x` | `r5x_p_l10_500_k7tail_rep{a,b}` (two replicates of R4 `r4k7_p_l10_500_tail1ug`); GR00T one-block (10-step) pure-cache tail `r5x_g_{l10,sp}_{500,50}_tail1u` | three-run pooled K7 tail (acceptance rule); GR00T "commit" vs "commit for 15 steps" |
| `r05_q2` | GR00T CycleTail `r5q2_g_{l10,spatial}_{500,50}_G10` (MISS every 4th anchor, 10-step chunks from both sources); GR00T pure policy L=10 `r5q2_g_{l10,spatial}_policy_L10` | GR00T mixed mode; GR00T execution-length reference |
| `r05_q6` | `r5q6_p_{l10,spatial}_{500,50}_tail` (wrist-confirmed stuck guard + wrist-only vision + anchor_tail) | stacking the cheaper wrist key on the blind tail |
| `r05_b1` | `r5b_{p,g}_{cell}_{hand,solve}` for π0.5 (K7 tail controller) sp-50, sp-500, l10-500 and GR00T (pure cache) sp-50, sp-500, l10-500 | owner item 10: the constrained offline solver's parameters vs hand-tuned, one change set per cell, frozen before the loop |
| `r05_q5` | `r5q5_p_l10_{50,500}_cl2_shadow` (GPU retrieval in shadow; served actions are the CPU ones) | live GPU/CPU agreement and latency under load; SR must equal R2 CL2 within noise |
| `r05_*_smoke` | smokes, not for conclusions | |
References: every R4 run root listed in `rounds/r04/ANALYSIS_BRIEF.md` (R2 CL2, R3 stock g/g500, R4 frontier, seeded
pure inference L=5 s1001/s2001 and L=10, K7, K5, wrist, GR00T blind), trace_dual pure inference
(.986 / .844 / .940 / .870 for π0.5-sp / π0.5-l10 / GR00T-sp / GR00T-l10).
**Replicates hidden in B1** (verified from `r05_b1/arms_in.json`): the `*_hand` arms re-run existing configurations —
π0.5 l10-500 / sp-500 hand = R4 K7 tail (`r4k7_p_{l10,sp}_500_tail1ug`), GR00T l10-500 hand = R2 CL2
(`oscl500_g_l10_cl2`); check the others. Pool them as replicates (K7 tail l10-500 then has four runs: R4, B1 hand,
`r05_x` rep a/b). Each `*_solve` changes one thing (π0.5 500: kref 8→5; π0.5 sp-50: ridge .1→1.0; GR00T l10-500:
`state_scale` 1→3; read the specs for the rest).
Use per-arm `runs/<arm>/summary.json` (the root `summary.json` can miss an arm when two chains share a run root —
`r05_b1` and `r05_x` do). Only use an arm with `state/<arm>.DONE`; list any arm that is missing or SKIPPED.

## Cost bases (owner basis primary; never mix)
- π0.5: vision decision .152, MISS +.848 (full inference), blind / policy-tail decisions 0 → IR = .152·v + .848·m.
  GR00T: .148 / +.852 (.174 + .678) → IR = .148·v + .852·m. Pure inference L=5 = 1.0, L=10 = .5.
- **Recompute owner IR from `cost_ledger` v, m in each per-arm summary** — the summary's own IR fields are the eager
  ledger or a measured ratio. Wrist-only stage 1: `rounds/r04/cost_table_owner.json` (.0552·v + m·(.848 + .0499),
  ratio-transfer ASSUMPTION — label it). dummy_cached: −.048·v exact (R4 §3).
- grow250 is paid policy data: report its acquisition cost (Q4 hand-back: 14,849 / 5,310 policy calls) separately
  from the serving IR, and the lifecycle IR if the growth episodes are counted.
- Search cost: K8 CPU table as in R4; for `r05_q5` report the measured live GPU vs CPU retrieval latency.

## Tools
`taskset -c <range> .venv/bin/python -m exp.offline_search.closed_loop.ops.kpi --run-root <R> [--run-root …] <run:arm …>
--ref <run:arm> [--manifest rounds/r05/q4_growth/evaluation_pairs.json] --json … --md …`; R4's analysis scripts in
`/home/weiland/.claude/jobs/a607dd74/tmp/analysis_r04/` (`an.py`, `blind_stats.py`, `blind_runs.py`, `frontier.py`) can
be copied and extended; `rounds/r05/q5_gpu/summarize_shadow.py --log-dir <run>/runs/<arm>/server_<port>` for Q5.
Journals: `task_uid` = `<arm>:eval:<task>:<init>`; exact McNemar on discordant pairs; init-clustered bootstrap.

## Deliver `rounds/r05/ANALYSIS.md` (English) and print a one-paragraph summary
1. **Noise floor and acceptance rule**: g500 three runs (R4) plus K7 tail l10-500 three runs (R4 + `r05_x` reps): run
   spread, discordance, and whether K7 tail's +3.1 pp over stock survives the pooled three-run comparison.
2. **Execution length, cache side and policy side**: pure-cache tail vs CL2 on l10 500 / 50 and spatial 50 (with R4
   sp500); GR00T one-block (10) vs two-block (15) pure-cache tail vs GR00T policy L=10 vs trace_dual L=5 — is the
   variable "commit" or "commit length", and does it differ by suite?
3. **C10 policy commit** vs K7 tail (pooled three runs) and vs L=10 pure inference at both scales and both suites;
   realized policy-tail share; **D1** (reach, triggers, effect; say plainly if one run cannot resolve it).
4. **GR00T mixed mode**: CycleTail G10 vs pure-cache tails (one- and two-block) vs policy L=10, per cell; where the
   scheduled policy call pays (50 library) and where it does not (500, spatial).
5. **Wrist stack (Q6)** vs K7 tail and vs R4 wrist guard-only; the wrist sp50 replicate.
6. **Library growth**: grow250 refit / frozen vs CL2-50 and CL2-500 on the same 250 pairs (same controller; state the
   controller mismatch if any); demo scaling curve 50 → 100 → 200 → 300 → 500 with the tail controller (kref and
   data-source caveats), refit vs frozen50 at 200; bytes and rows per point; what a deployment should do.
7. **B1 solver vs hand** (owner item 10): paired per cell, pooled; the parameters that changed; whether the offline
   objective predicted the sign; a direct answer to "can hyperparameters be solved offline" with the R4 + R5 evidence.
8. **Q5 GPU shadow**: live agreement (top-1, top-16, chunk, confidence), latency p50/p90/max vs CPU, failures; SR vs R2
   CL2 (must be equal: shadow serves CPU actions); what is needed before `serve`.
9. **Frontier per cell** (π0.5 l10 / spatial × 50 / 500; GR00T same) with every R4 and R5 point, owner basis primary,
   plus search cost and dummy_cached repricing panels; mark dominated points; the best deployable point per cell and
   its bytes.
10. **Four-layer decomposition** (synthesis / method / library / control incl. execution length) + cost layer, R4 → R5.
11. **What is settled, what is dead, and ranked R6 proposals** (R5 is the last round in the current owner goal; write
    the recommendations for the owner to decide on).

Rules: CPU only, `taskset -c <range in your prompt>`, ≤ that many processes, OMP/OPENBLAS/MKL_NUM_THREADS=1,
CUDA_VISIBLE_DEVICES=''. Read-only except `rounds/r05/ANALYSIS.md` and your scratch directory. No git, no `rm -rf`,
never `pkill -f`, no servers / chains / tmux / timan107 / ports 23150-23169 (the coordinator runs every experiment). Do
not read `tests/review_tests/`. Be quantitative and skeptical: paired tests on the same inits, the measured noise floor,
realized vs intended shares, and whether an SR gain is just more policy inference or longer execution.
