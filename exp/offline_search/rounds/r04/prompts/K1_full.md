<task>
You are R4 coding agent K1: the method side of "look once, act several steps" plus two pure-cache/guard methods.
Directory: exp/offline_search/rounds/r04/k1_blind/ (create; add __init__.py). Build on
exp/offline_search/rounds/r02/g1_awm/awm.py (AWM), rounds/r03/h1_trap/awm3.py (AWM3) and rounds/r03/h3_judge/judge.py
(MixedJudge) by subclassing/wrapping; never edit them. Implement, following ideation A §1.1–§1.3 and §2 exactly unless
a test forces a change (document any change):
1. Blind-capable AWM methods (usable pure-cache and inside the mixed judge), implementing
   `blind_step(bq) -> BlindResult | LookReason` from the CODING_BRIEF contract (types in
   exp.offline_search.closed_loop.blind, created by K2; if it does not exist yet when you start, define identical
   dataclasses in your package behind a try/except import so both work): serving variants `phase_particles`
   (fixed 16-member anchor kernel and weights, each member advanced along its own library episode, per-member phase
   chosen from offsets {h-1,h,h+1} by the proprioceptive cost with the .05 offset penalty, monotone, ≤2 rows per blind
   step), `kernel_clock`, `top1_clock`, `anchor_tail` (serve the unexecuted part of the anchor chunk; π0.5 H=10 → 1
   blind decision, GR00T H=16 → 2); budget B ∈ {1,2,3,4}; look gates: gripper event ahead (≥.20 anchor mass), near
   terminal (≥.20 mass on the last two rows), two low-motion intervals (library 10th percentile), displacement
   residual > .5 (thresholds .25/1 as variants), lifecycle (first decision / after MISS / invalid anchor); a
   `gates="budget_only"` variant for the ungated tail baseline. State scales/percentiles fitted from the deployed
   library only. Every variant must also run through the ordinary `query()` (vision decisions) exactly like AWM.
2. A gap-aware MixedJudge subclass (`noprog_span`: retrieval progress measured only on vision anchors, accumulated over
   elapsed decisions; motion guard from dense proprioception), wrapping the blind-capable base so the plugin's mixed
   mode can run blind stretches; plus a MixedJudge "memo reset after MISS" variant (ideation B2) and plain `noprog_n`
   passthrough.
3. `control_step_library` (ideation C REPORT.md proposal 1) as a pure-cache method with its ablation G (virtual-distance
   ranking, unshifted actions) and GS (spliced actions).
Tests: offline harness smoke on π0.5 and GR00T cells (inf and cache) at both library sizes; bit-for-bit equality of
every variant's vision `query()` with AWM (or documented, justified differences); a blind-step replay on the store
reproducing ideation A's h=1/h=2 phase-vs-clock numbers to within reconstruction tolerance; a CPU mixed-sequence test of
blind_step with a fake plugin driver (vision → blind → blind → vision → MISS → vision) checking anchors, invalidation
and gate reasons. Deliver `k1_blind/arms_r4.json`: emit_arms spec rows (with `<RUN>` placeholder for fit artifacts) for
the third- and fourth-batch arms in SELECTION.md at both library sizes (π0.5 l10 first, then spatial; GR00T pure-cache
tail/phase), plus prefit commands and measured pickle sizes/fit times.
</task>

<context>
Repository /home/weiland/projects/openpi (branch Ziyang). Round 4 (R4) of an action-cache exploration for VLA policies
(π0.5, GR00T N1.5) on LIBERO. You are one of four coding agents working in parallel. Read first:
exp/offline_search/rounds/r04/CODING_BRIEF.md (binding: file ownership, live-system rule, the method<->plugin contract,
log fields, cost ledger, rules), then exp/offline_search/rounds/r04/SELECTION.md, the ideation reports in
exp/offline_search/rounds/r04/ideation_{A,B,C}/ (REPORT.md; C also REPORT_2.md), exp/offline_search/closed_loop/README.md,
exp/offline_search/harness/README.md, and the code you build on (named in your task).
Store: /dev/shm/offline_search_store (hot) or /home/weiland/trace_runs/offline_search_store (cold copy, same content).
Python: /home/weiland/projects/openpi/.venv/bin/python from the repo root (GR00T servers use their own venv, see
closed_loop/ops/start_server.sh).
</context>

<hard_constraints>
- Your CPU range: 18-21,62-65. Prefix every python/numpy command with `taskset -c 18-21,62-65` and set
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; never more processes than threads in your range.
  CPUs 38-43 and 82-87 belong to another project: never use them.
- GPU: none. Run with CUDA_VISIBLE_DEVICES=''.
- Edit only the files you own (CODING_BRIEF.md "File ownership"). Develop in a copy, keep existing behaviour
  byte-identical when your new options are absent, install each shared file atomically (temp file + mv), and re-run the
  existing selftests after installing.
- No git. No `rm -rf`. Never `pkill -f`; kill only processes you started, by PID. No timan107, no LIBERO workers, no
  closed-loop chains. Never touch ports, processes or tmux sessions you did not start. Do not read tests/review_tests/.
  Do not modify src/, harness/, profile/ or rounds/r01..r03/.
</hard_constraints>

<completeness_contract>
Resolve your task fully before stopping: implementation, tests, and the hand-back file. Do not stop at the first
working version; check edge cases (episode reset, step 0, after a MISS, both models where required, both library sizes
50 and 500 where required).
</completeness_contract>

<verification_loop>
Before finishing, verify every deliverable against the task and the brief with real runs (selftests, parity checks,
smoke runs as allowed). If a check fails, fix and re-run; report the final numbers, not the first attempt.
</verification_loop>

<grounding_rules>
Report only what you ran and observed; give exact commands, numbers and file paths. Label anything unverified.
</grounding_rules>

<structured_output_contract>
Write rounds/r04/k1_blind/HANDBACK.md (files changed/added with install times; exact switches/arm-spec fields; tests
and their numbers; what the coordinator must do next; caveats), then print a one-paragraph summary as your final
message.
</structured_output_contract>
