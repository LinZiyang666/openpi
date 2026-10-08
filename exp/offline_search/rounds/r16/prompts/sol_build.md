<task>
R16 (owner goal 2026-10-07 "融入纯减步并实验"): every policy call at deployment runs the action expert with only k denoising steps (k ∈ {1, 2}; π0.5 default 10, GR00T N1.5 default 8). Build everything needed to run the two phases in exp/offline_search/rounds/r16/COORDINATOR_DECISIONS.md (read it first; D1–D6 are binding). Repository: /home/weiland/projects/openpi.
</task>

<context>
- Deployed five-layer system: rounds/r10/recipe (cache + no-progress guard + distance corrector), rounds/r11/knob (periodic knob), rounds/r14/descent (TaskDescentKnob, descent roots, test-A emitter), rounds/r14/descent_astra (frozen PROTOCOL.md + analysis), rounds/r15/descent (sizes 100/200/500: dev roots, shared dev C10, leave-own-episode-out at 500, analyze.py / analyze_test_a.py with per-model stage and combine, LOADER_FIXES.md). Reuse these; R16 must be R14/R15 with only the policy-call step count (and its cost) changed.
- Existing MISS step reduction: closed_loop/stage_overrides.py `miss_steps_from_yaml` (yaml `miss: {num_steps: k}`; GR00T requires the live head K to match, e.g. serve_groot.py `--denoising-steps` / GROOT_DENOISING_STEPS), used by R4 K3 "MISS K2" arms (rounds/r04/k3_cost/make_arms.py yaml_patch). The plugin logs `miss_k` per MISS decision and the cost ledger knows `miss_steps`. Verify that every policy call path honours k: guard MISS, periodic knob MISS, the policy-tail commit (executes the k-step chunk's steps 5–9 blind), and pure-inference (C10) arms.
- Phase 1: pure inference with k steps, same identities/seeds/template as R14's test-A pure C10 arms (find them; R15 reused them as its test reference). 8 arms (2 models × 2 suites × k 1/2).
- Phase 2: sizes 50/100/200/500 × 2 models × 2 suites × k 1/2 = 32 cells. Dev descent arms per cell: periodic ir45/ir40/ir32/ir25/off with k-step policy calls, on the R15 D2 dev inits (size 50: R14's dev inits). Tolerance reference = full-step pure C10 on the same dev inits: reuse R14/R15 dev C10 if identities/template match (prove it), else emit the missing full-step C10 dev arms. Task fallback when ir45 fails = k-step pure inference (needs no dev run unless the protocol requires its SR; say which).
- Cost (D5): π0.5 IR = (.152 V + c(k) M)/N, c(1)=.454, c(2)=.498; GR00T IR = (.148 V + c(k) M)/N, c(1)=.258, c(2)=.343; compute c(k) from the cited stage-cost sources, not hand-typed constants.
- Hosts (D6): every root runs servers on the local 4090 (`@local`), workers on timan107. Queue format for my runner: one arm per line, `@local /abs/root|arm`; phase 1 first, then phase 2 dev ordered so each cell's levels finish together.
</context>

<deliverables>
1. Runtime: k-step policy calls for both models through the existing mechanism, default-off; with k = default steps every decision and action is byte-identical to today (CPU replay proof on recorded or synthetic inputs, both models). Any shared-runtime edit atomic, minimal, tested. Decision logs must show the k actually used (miss_k or equivalent) so I can check it after the first arm.
2. Roots and arms: phase-1 roots, phase-2 dev roots, queue files rounds/r16/queue_phase1.txt and rounds/r16/queue_dev.txt. Roots under /home/weiland/trace_runs/os_closed_loop/r16_*; never bind a fleet inside a root.
3. Analysis: (a) phase 1 paired table vs R14 test C10 (SR, Δ with paired CI, IR with D5 pricing); (b) dev descent → frozen per-task level tables per cell (and uniform level per cell for reference), same rules as R14/R15; (c) a test-A emitter that builds the per-task arms from the frozen tables (refuses to run before the table freeze), queue_test.txt; (d) test-A analysis: per cell vs full-step pure C10 and vs the existing full-step per-task descent arm of the same cell (R14 size 50, R15 sizes 100/200/500), pooled per size and per k. Test the analysis on synthetic outcomes only.
4. Deployment list: files to sync to timan107 and to the local server tree, with sha256; per-server GPU memory estimate for π0.5 and GR00T full-model servers at k steps (my quota on the 4090 is 28 GB).
5. Real-environment smoke commands for me (not run by you): per model, 1 arm × 2 episodes for phase 1 k=1 and one phase-2 dev arm, with the expected log fields.
6. rounds/r16/HANDBACK.md (short): what exists, proofs, commands, open risks.
</deliverables>

<constraints>
CPU only: taskset -c 2-9,46-53 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python. No servers, GPU, tether/remote operations, sync/chain launches, git, pkill, rm -rf. No symlinks under exp/offline_search; pytest with -p no:cacheprovider and scratch outside the repo. Write under exp/offline_search/rounds/r16/ and the new r16_* roots; shared edits only if unavoidable (default-off). Test A is measure-only: do not read test-A outcomes except R14/R15 summary metadata needed for pairing and hashes. Refit nothing.
</constraints>

<default_follow_through_policy>
Do not stop to ask routine questions; choose the option most faithful to R14/R15 and record the choice in HANDBACK.md. Stop and report only if a decision in COORDINATOR_DECISIONS.md cannot be implemented as written.
</default_follow_through_policy>
