<task>
You are R4 ideation agent A. Your topic is the owner's new direction "look once, act several steps" (看一眼，做几步):
a decision WITH vision (stage-1 keys, AWM retrieval) acts as an anchor; the following decisions are taken WITHOUT running
the vision encoder (stage 1 skipped -> these decisions cost ~0 IR) until a condition sends the loop back to vision.
Today every decision pays stage 1 (IR floor .152); at the best l10 point (500-episode library, guard-only mixed mode,
IR .238) that floor is 64 % of the cost. Design, measure and rank concrete methods for this, including:
- What to serve on a vision-free decision. The measured closed-loop facts (FINDINGS.md) say AWM's consecutive picks hit
  the exact successor row only 18–29 % of the time and stall on the same row 15–40 %, and the kernel mixes 3–4 library
  episodes: so naive "follow the top-1 demo in time" is suspect. Evaluate alternatives such as advancing every kernel
  member along its own library episode and re-synchronising each member's phase with the robot's proprioceptive state
  (robot_state valid dims) — or anything better you find. Quantify on the offline store (all four cells, inf and cache
  query cells, 50- and 500-episode libraries) how the served action degrades with the number of blind decisions
  1..4, split by task phase (near library gripper transitions vs not, early/mid/late), and compare with (a) vision every
  decision (AWM), (b) simply executing a longer part of the anchor's chunk (the "execute more steps" baseline).
- When to look again: bounded blind budget, phase-aware triggers known from the library without vision (e.g. a gripper
  open/close ahead in the members' episodes, near-terminal rows), proprioceptive triggers (no motion, deviation from the
  expected library state), after any MISS. Estimate from the store / closed-loop logs how often each trigger would fire
  and at which phases the pure-cache failures (deadlock spells, grasp/release traps) start relative to them.
- How it combines with the existing mixed HIT/MISS mode (guards / periodic MISS) and with the 500-episode library, and
  the resulting IR (cost model: vision decision .152, vision-free decision ≈ 0, MISS 1.0).
- What the closed-loop server/plugin must do: read exp/offline_search/closed_loop/plugin.py (in particular the
  per-connection policy wrapper `_ConnPolicy.infer` and how `on_search` / bookkeeping work) and state precisely how a
  vision-free decision can be served without the interceptor running stage 1, what the client must send, and how the
  QueryView history stays consistent. Do not implement it; specify it.
Your CPU range: 0-11,44-55 (24 logical CPUs, at most 24 processes). Your letter: A.
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
  `taskset -c 0-11,44-55` and set OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; never run more processes
  than logical CPUs in your range. CPUs 38-43 and 82-87 belong to another project: never use them.
- GPU: CPU only (CUDA_VISIBLE_DEVICES=''). Do not use the GPU.
- Never touch running processes, tmux sessions, ports or servers you did not start; never start LIBERO workers or use
  timan107; no closed-loop runs (the coordinator runs all closed loop).
- No git commands. No `rm -rf`. Never `pkill -f`. Do not read tests/review_tests/. Do not modify any existing file.
  Write ONLY inside exp/offline_search/rounds/r04/ideation_A/ (scripts, outputs, report). Large arrays
  (> 50 MB) go to /home/weiland/trace_runs/offline_search_store/derived/r04/ideation_A/.
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
Write your final report to exp/offline_search/rounds/r04/ideation_A/REPORT.md (self-contained, English):
1. Key measured facts that drive your proposals (numbers + script paths).
2. Up to 4 proposals, ranked by expected closed-loop value per unit IR, each with: name, pitch, hypothesis/mechanism,
   algorithm precise enough to implement against the harness Method API and the closed-loop plugin (say exactly what
   the plugin/server must change, if anything), predicted effect on SR and IR at 50 and 500 episodes (IR with the cost
   model in FINDINGS.md), cost tier and library bytes, kill criterion, cheapest diagnostic, closed-loop pilot design
   (arms, cells, inits), variants.
3. 2–3 rejected ideas with measured reasons.
Then print a one-paragraph summary as your final message.
</structured_output_contract>
