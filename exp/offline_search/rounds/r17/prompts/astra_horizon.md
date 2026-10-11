<task>
Research question from the owner: for the new models (DP-T, Fast-WAM, Cosmos Policy), can we simply increase the action horizon, i.e. execute more actions per inference call with fewer calls, the way we did for π0.5 and GR00T, without losing success rate?

1. Find out exactly what was done for π0.5 and GR00T:
   - which "horizon" knob (executed actions per call / replan steps / chunk length; e.g. the "C10 = 10 steps executed per call" full-policy reference used in R14–R16);
   - which values were tried, on which test set and episode counts, and what the SR/cost results and conclusion were.
   Sources: `/home/weiland/projects/openpi/exp/offline_search/rounds/` (r10–r16 REPORT/DESIGN/COORDINATOR_DECISIONS), `/home/weiland/projects/openpi/logs/` (*plan.log.md, session_handoff.md), `docs/`. Cite file:line for every claim.
2. For each new model, establish from code and official sources:
   - native chunk length;
   - the official/default executed horizon (DP-T horizon 16; Fast-WAM; Cosmos LIBERO executes 16 of a 16×7 chunk);
   - what our R17 serve entries and formal launcher currently use (`exp/r17_integration/models/*`, `exp/r17_integration/ops/formal_eval.py`, `exp/mw_finetune/*`; e.g. DP100/replan5, Fast10/replan5, Cosmos5/replan16);
   - any model-specific constraint on executing longer, e.g. Cosmos already executes its whole chunk, so "larger" may require a longer predicted chunk, which may not be a pure knob.
3. Design a study matching the π0.5/GR00T protocol:
   - horizon values per model;
   - LIBERO official test A (50 official inits per task, 500 per suite; `--official-50`), at least libero_spatial and libero_10;
   - paired comparison vs each model's default horizon;
   - the same decision rule used for π0.5/GR00T, otherwise a pre-registered one;
   - MetaWorld MT50 only if the π/GR00T study included it.
   State the expected GPU time per arm.
4. If the formal launcher lacks a horizon/replan parameter, have it added: either write the minimal default-off flag yourself with a CPU test, or specify exactly what sol must add. Do not change default behaviour.
</task>

<constraints>
- GPUs are busy until this evening with official-test-A baselines (DP on 4090, Cosmos + Fast-WAM on H100). Do NOT launch GPU runs. Total GPU >15 min must be handed to the coordinator as exact commands. A 2-episode CPU/GPU smoke ≤5 min is OK only if a GPU is idle (check `nvidia-smi`), never killing anything.
- Read-only on existing results. Do not touch running processes.
</constraints>

<compact_output_contract>
Write `/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/STUDY.md` (Chinese, tables). Sections:
(1) what π0.5/GR00T did, with citations;
(2) per-model horizon facts and constraints;
(3) the pre-registered study design and decision rule;
(4) exact coordinator commands with GPU-time estimates;
(5) open questions for the owner, if any.
Final message: 6 lines max.
</compact_output_contract>
