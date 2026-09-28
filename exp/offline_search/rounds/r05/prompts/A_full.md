<context>
Repository: /home/weiland/projects/openpi (branch Ziyang). This is round 5 (R5) of a multi-round exploration of action-cache
retrieval methods for VLA policies (π0.5 and GR00T N1.5) on LIBERO (suites libero_spatial and libero_10). A server
answers each decision (every 5 control steps) either from a library of stored action chunks (HIT, cheap) or by running
the policy (MISS, expensive). The exam is closed-loop task success rate (SR) versus inference ratio (IR).
Read, in this order (all paths relative to the repo root):
1. exp/offline_search/rounds/r05/FINDINGS.md  (what R2–R4 established, the cost model, dead ends, owner rulings — binding)
2. exp/offline_search/rounds/r04/ (SELECTION.md, CODING_BRIEF.md, hand-backs k1..k10, k8 and k9 reports, ideation A/B/C reports) and rounds/r03/ANALYSIS.md
3. exp/offline_search/rounds/r02/ANALYSIS.md  (R2: 32 pure-cache closed-loop arms, three-layer decomposition)
4. logs/offline_search_exploration.log.md sections 8, 9 and the R4 part of section 10 (protocol, owner rulings, ledger; Chinese)
5. exp/offline_search/IDEATION_BRIEF.md (proposal format and settled negatives)
6. exp/offline_search/harness/README.md (offline store layout, Method API, metrics, valid action/state dims)
7. exp/offline_search/closed_loop/README.md (server plugin incl. mixed HIT/MISS mode, KPI tool, how arms are run)
8. Code you may build on: exp/offline_search/rounds/r02/g1_awm/awm.py (AWM), rounds/r03/h1_trap/awm3.py (AWM3),
   rounds/r03/h3_judge/judge.py (MixedJudge: V7 confidence + guards), rounds/r02/g3_recovery/ (V6/V7 wrappers), rounds/r04/k1_blind/ (blind methods), rounds/r04/k7_guard/judge.py (vision-confirmed guard), rounds/r04/k9_gpu_retrieval/ (GPU retrieval prototype),
   exp/offline_search/closed_loop/plugin.py and ops/ (kpi.py, chain.sh).
Data (read-only):
- Offline store: /dev/shm/offline_search_store (hot copy; cold copy /home/weiland/trace_runs/offline_search_store).
  queries/<model>_<suite>_<inf|cache>/ = recorded decisions of full-inference and pure-cache runs with keys, robot state,
  executed chunks and the policy's own action a_inf; library/<model>_<suite>/{current (≈50 episodes), bpool_cs (π0.5,
  500), bpool_all (GR00T, 500)} with episode/step/progress/next/prev arrays; tok/ subsample has stage-1 tokens and images.
- Closed-loop runs: /home/weiland/trace_runs/os_closed_loop/{r02_g50, r02_g500, r03_pilot, r03_full, r03_mx, r04_*}/runs/<arm>/
  (server_*/decisions_*.jsonl = one row per decision with picks, scores, confidence, extras, hit/miss, latencies;
  client/journal.jsonl + per_step.jsonl; summary.json). Arm definitions in each run root's arms.json.
- Related line (read-only): exp/step_diag/analysis/step_vs_warmstart.md (reduced denoising steps / warm start results).
</context>

<hard_constraints>
- Codex agents only: do all the work yourself (no sub-agents needed).
- Python: /home/weiland/projects/openpi/.venv/bin/python, run from the repo root. Prefix EVERY python/numpy command with
  `taskset -c 30-33,74-77` and set OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; never run more processes
  than logical CPUs in your range. CPUs 38-43 and 82-87 belong to another project: never use them.
- GPU: none (CUDA_VISIBLE_DEVICES='').
- Never touch running processes, tmux sessions, ports or servers you did not start; never start LIBERO workers or use
  timan107; no closed-loop runs (the coordinator runs all closed loop).
- No git commands. No `rm -rf`. Never `pkill -f`. Do not read tests/review_tests/. Do not modify any existing file.
  Write ONLY inside exp/offline_search/rounds/r05/ideation_A/ (scripts, outputs, report). Large arrays
  (> 50 MB) go to /tmp/r05_ideation_A/ (/home/weiland/trace_runs is read-only in your sandbox).
- Owner rulings (FINDINGS.md) are binding: a vision anchor is required in every episode's control loop (bounded
  vision-free stretches are allowed); no external models (CLIP/DINO/…); report every conclusion at the 50-episode AND
  the 500-episode library, with library bytes vs the deployed pkl (π0.5 431/1103 MB, GR00T 429/1068 MB); split effects
  into synthesis / method at fixed library / library / control (policy calls counted as cost); label any fit that uses
  data beyond the deployed library as "borrowed big-library information".
</hard_constraints>

<research_mode>
Time is not a constraint: study the material carefully, run the diagnostics you need on the store and the closed-loop
logs, and do not stop at the first plausible idea. Offline action error does NOT rank closed-loop methods (see
FINDINGS.md), so ground proposals in mechanisms visible in the closed-loop logs and in cost arithmetic, and say which
closed-loop pilot would decide each one.
</research_mode>

<grounding_rules>
Ground every claim in files you read or numbers you computed; give script paths and exact numbers. Label hypotheses and
forecasts as such. If a needed fact is missing, compute it or state exactly what is unknown.
</grounding_rules>

<structured_output_contract>
Write your final report to exp/offline_search/rounds/r05/ideation_A/REPORT.md (self-contained, English):
1. Key measured facts that drive your proposals (numbers + script paths).
2. Up to 4 proposals, ranked by expected closed-loop value per unit IR, each with: name, pitch, hypothesis/mechanism,
   algorithm precise enough to implement against the harness Method API and the closed-loop plugin (say exactly what
   the plugin/server must change, if anything), predicted effect on SR and IR at 50 and 500 episodes (IR with the cost
   model in FINDINGS.md), cost tier and library bytes, kill criterion, cheapest diagnostic, closed-loop pilot design
   (arms, cells, inits), variants.
3. 2–3 rejected ideas with measured reasons.
Then print a one-paragraph summary as your final message.
</structured_output_contract>

<task>
You are R5 ideation agent A: execution horizon and "when to look again".
R4's strongest closed-loop facts are about how long an action chunk is executed before the next look: pure-inference
l10 improves from SR .845–.850 (5 executed steps per call) to .904 (all 10 steps); anchor_tail (after a vision HIT,
serve steps 5–9 of the same synthesized chunk without vision) reached .880 @ IR .203 (500-episode l10), .806 @ .242
(50-episode l10) and .982 @ .128 (500-episode spatial); per-neighbour phase continuation fails at 50 episodes; an
opt-in "policy tail" (after a MISS, execute steps 5–9 of the policy's own chunk without vision) is being implemented
(rounds/r04/k10_policy_tail/). Study this mechanism and design R5's best controller:
1. Why does executing whole chunks help on l10 (and does it on spatial)? Use the closed-loop logs of the L=5 vs L=10
   pure-inference arms (/home/weiland/trace_runs/os_closed_loop/r04_cost: r4f_p_l10_inf_s1001, r4b2_p_l10_inf_k2_s1101
   (L=5), r4f_p_l10_inf_k10_L10 (L=10); spatial counterparts as they complete), per task / per failure type, stalls,
   re-planning jitter at chunk boundaries (compare consecutive chunks' overlap), gripper events.
2. How far can the look cadence go? Beyond one chunk (e.g. stitching the next library rows after the chunk's end,
   or the control-step library of C), adaptive look timing from proprioception / phase / predicted error, and what the
   guards need to see. Separate the execution-length effect from the vision-saving effect in every forecast.
3. GR00T (H=16 chunks, 8-step executes? check the GR00T LIBERO settings) has never been run in mixed mode: propose
   GR00T mixed arms with tails, both suites and library scales.
4. Anything in the MISS path that interacts: e.g. when a MISS happens mid-chunk, is it better to execute the policy's
   whole chunk (policy tail) or the rest of a cache chunk?
Give concrete arms (method kwargs / plugin flags) for R5's closed loop.
</task>
