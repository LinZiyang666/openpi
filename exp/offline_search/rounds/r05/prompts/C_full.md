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
  `taskset -c 10-13,54-57` and set OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; never run more processes
  than logical CPUs in your range. CPUs 38-43 and 82-87 belong to another project: never use them.
- GPU: none (CUDA_VISIBLE_DEVICES='').
- Never touch running processes, tmux sessions, ports or servers you did not start; never start LIBERO workers or use
  timan107; no closed-loop runs (the coordinator runs all closed loop).
- No git commands. No `rm -rf`. Never `pkill -f`. Do not read tests/review_tests/. Do not modify any existing file.
  Write ONLY inside exp/offline_search/rounds/r05/ideation_C/ (scripts, outputs, report). Large arrays
  (> 50 MB) go to /tmp/r05_ideation_C/ (/home/weiland/trace_runs is read-only in your sandbox).
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
Write your final report to exp/offline_search/rounds/r05/ideation_C/REPORT.md (self-contained, English):
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
You are R5 ideation agent C: the library as a model component — growth, pruning, routing and GPU residency.
R4 showed library density matters (per-neighbour phase continuation fails at 50 episodes but works at 500; anchor_tail
works at both) and that retrieval can live on the GPU inside stage 1's CUDA graph (rounds/r04/k9_gpu_retrieval/:
in-place append keeps graph addresses). Study:
1. Online self-growth: start from the 50-episode library and insert what deployment produces (policy MISS chunks with
   their keys/state; possibly successful cache segments). Using the recorded closed-loop decisions (MISS rows carry the
   policy chunk and the query inputs), simulate growth offline and estimate how fast (episodes of deployment) the
   50-library approaches 500-library behaviour; what to insert, what not (failed episodes, stalls), dedup, and the
   fit/calibration consequences (PCA/whitening frozen vs refit; K9 requires frozen bases for in-place append).
2. Pruning / coreset: from usage statistics over ≈1.6 M logged decisions (which rows are ever top-16, which lead to
   stalls/traps), how small can the 500 library get at equal closed-loop behaviour; bytes vs deployed pkl.
3. Routing / priors: task-level and scene-switch priors (owner's earlier idea), library partitioning, cross-task
   donors only where measured useful (C2 found cross-task donors rarely win).
4. GPU deployment details that change method design (fixed capacity, tie policy, float precision).
Label every borrowed-information fit; forecast SR / IR at both scales; give R5 arms.
</task>
