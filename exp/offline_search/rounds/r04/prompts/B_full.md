<task>
You are R4 ideation agent B. Your topic is lowering the cost of every decision type other than "look once, act several
steps" (agent A owns that): the per-decision vision cost and the per-MISS cost, plus the best schedules at the low-IR end.
Cost model (π0.5, CUDA-graph): stage 1 vision encoder 10.26 ms (.152), stage 2 VLM prefix 27.69 ms (.410), stage 3
denoising 29.57 ms (.438). Study and rank concrete methods, including:
1. Cheaper retrieval key from the policy's OWN vision encoder (owner-approved): e.g. keys pooled from an intermediate
   layer of the vision tower, lower input resolution, fewer cameras or pooled patches, reusing part of the previous
   decision's computation. Find where stage 1 is computed (src/openpi: the π0.5 model's image embedding path; the
   closed-loop key builder `cp1_*` and the interceptor's stage split; the GR00T path under src/openpi/cache/groot/ and
   exp/libero_groot/) and measure, on the stored tok-subsample images (π0.5 has images in the store), (a) the cost of
   each cheaper key relative to full stage 1 on the 4090, and (b) retrieval quality with AWM refitted on the cheaper
   keys (library keys rebuilt the same way): top-k overlap with the full-key AWM, action error vs a_inf, and gripper
   vote splits at grasp/release. A MISS still needs the full stage 1, so the cheaper key saves only on HITs: give the
   net IR.
2. Cheaper MISS: fewer denoising steps on MISS (stage 3 only; use exp/step_diag/analysis/ for what is known about
   LIBERO step ladders and warm starts, and state what must still be measured on libero_10), whether anything in stage 2
   can be saved on a MISS without changing src/ semantics, and the resulting MISS cost.
3. The low-IR end of the SR-vs-IR frontier at the 500- and the 50-episode library: which schedules (periodic MISS every
   k, guard variants, HIT-run caps) should be run to map IR .15–.30, and which strong baselines must share the plot
   (reduced steps only, executing a longer part of each chunk, pure inference with more seeds) — with their IR.
Give the stacked IR table for the combinations you recommend. Your CPU range: 12-23,56-67 (24 logical CPUs, at most 24
processes). Your letter: B.
</task>

<context>
Repository: /home/weiland/projects/openpi (branch Ziyang). This is round 4 (R4) of a multi-round exploration of action-cache
retrieval methods for VLA policies (π0.5 and GR00T N1.5) on LIBERO (suites libero_spatial and libero_10). A server
answers each decision (every 5 control steps) either from a library of stored action chunks (HIT, cheap) or by running
the policy (MISS, expensive). The exam is closed-loop task success rate (SR) versus inference ratio (IR).
Read, in this order (all paths relative to the repo root):
1. exp/offline_search/rounds/r04/FINDINGS.md  (what R2/R3 established, the cost model, dead ends, owner rulings — binding)
2. exp/offline_search/rounds/r03/ANALYSIS.md  (full R3 closed-loop analysis, incl. SR-vs-IR frontier and failure anatomy)
3. exp/offline_search/rounds/r02/ANALYSIS.md  (R2: 32 pure-cache closed-loop arms, three-layer decomposition)
4. logs/offline_search_exploration.log.md sections 8, 9 and the R3/R4 part of section 10 (protocol, owner rulings, ledger)
5. exp/offline_search/IDEATION_BRIEF.md (proposal format and settled negatives)
6. exp/offline_search/harness/README.md (offline store layout, Method API, metrics, valid action/state dims)
7. exp/offline_search/closed_loop/README.md (server plugin incl. mixed HIT/MISS mode, KPI tool, how arms are run)
8. Code you may build on: exp/offline_search/rounds/r02/g1_awm/awm.py (AWM), rounds/r03/h1_trap/awm3.py (AWM3),
   rounds/r03/h3_judge/judge.py (MixedJudge: V7 confidence + guards), rounds/r02/g3_recovery/ (V6/V7 wrappers),
   exp/offline_search/closed_loop/plugin.py and ops/ (kpi.py, chain.sh).
Data (read-only):
- Offline store: /dev/shm/offline_search_store (hot copy; cold copy /home/weiland/trace_runs/offline_search_store).
  queries/<model>_<suite>_<inf|cache>/ = recorded decisions of full-inference and pure-cache runs with keys, robot state,
  executed chunks and the policy's own action a_inf; library/<model>_<suite>/{current (≈50 episodes), bpool_cs (π0.5,
  500), bpool_all (GR00T, 500)} with episode/step/progress/next/prev arrays; tok/ subsample has stage-1 tokens and images.
- Closed-loop runs: /home/weiland/trace_runs/os_closed_loop/{r02_g50, r02_g500, r03_pilot, r03_full, r03_mx}/runs/<arm>/
  (server_*/decisions_*.jsonl = one row per decision with picks, scores, confidence, extras, hit/miss, latencies;
  client/journal.jsonl + per_step.jsonl; summary.json). Arm definitions in each run root's arms.json.
- Related line (read-only): exp/step_diag/analysis/step_vs_warmstart.md (reduced denoising steps / warm start results).
</context>

<hard_constraints>
- Codex agents only: do all the work yourself (no sub-agents needed).
- Python: /home/weiland/projects/openpi/.venv/bin/python, run from the repo root. Prefix EVERY python/numpy command with
  `taskset -c 12-23,56-67` and set OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; never run more processes
  than logical CPUs in your range. CPUs 38-43 and 82-87 belong to another project: never use them.
- GPU: allowed for the cheaper-key measurements only: at most 1 GPU process at a time, at most 8 GB, and only when nvidia-smi shows at least 16 GB free (the 4090 is shared with another project's training, which has priority; if you ever see an out-of-memory error, stop GPU work and continue on CPU). Release the GPU as soon as a measurement is done.
- Never touch running processes, tmux sessions, ports or servers you did not start; never start LIBERO workers or use
  timan107; no closed-loop runs (the coordinator runs all closed loop).
- No git commands. No `rm -rf`. Never `pkill -f`. Do not read tests/review_tests/. Do not modify any existing file.
  Write ONLY inside exp/offline_search/rounds/r04/ideation_B/ (scripts, outputs, report). Large arrays
  (> 50 MB) go to /home/weiland/trace_runs/offline_search_store/derived/r04/ideation_B/.
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
Write your final report to exp/offline_search/rounds/r04/ideation_B/REPORT.md (self-contained, English):
1. Key measured facts that drive your proposals (numbers + script paths).
2. Up to 4 proposals, ranked by expected closed-loop value per unit IR, each with: name, pitch, hypothesis/mechanism,
   algorithm precise enough to implement against the harness Method API and the closed-loop plugin (say exactly what
   the plugin/server must change, if anything), predicted effect on SR and IR at 50 and 500 episodes (IR with the cost
   model in FINDINGS.md), cost tier and library bytes, kill criterion, cheapest diagnostic, closed-loop pilot design
   (arms, cells, inits), variants.
3. 2–3 rejected ideas with measured reasons.
Then print a one-paragraph summary as your final message.
</structured_output_contract>
