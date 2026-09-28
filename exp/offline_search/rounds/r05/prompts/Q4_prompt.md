<task>
You are R5 coding agent Q4: a closed-loop-testable library growth snapshot (the owner's "增删 / self-growth" direction),
following rounds/r05/ideation_C/REPORT.md proposal 1 (§1c frozen-fit growth proxy, §2 algorithm) but WITHOUT plugin
changes: grow the library offline from already-recorded policy data, publish it as a new store library, and give the
coordinator arm specs to test it in closed loop on held-out inits.
1. Build grown libraries for π0.5 libero_10 and libero_spatial (GR00T optional, same recipe): start from the deployed
   50-episode library (`current`) and append every SUCCESSFUL episode of the recorded full-inference traces
   (`queries/pi05_<suite>_inf/` in the store — policy actions, keys, robot state, per-episode success) whose init index
   is 0–24 (all ten tasks); never use inits 25–49 (they are the evaluation inits). Admission exactly as C's report
   (successful episodes only, exact-duplicate removal, stable IDs, per-episode step/progress/next/prev with explicit
   handling of missing edges — a missing edge is "unknown continuation", not "terminal"). Store the result as new
   library names in BOTH store copies (/dev/shm/offline_search_store and /home/weiland/trace_runs/offline_search_store),
   e.g. `library/pi05_l10/grow250/`, in the exact on-disk format the existing libraries use (read harness/README.md and
   the store code; validate with the existing store loaders and library validators; do not modify existing libraries).
   Record provenance (source episodes, counts, bytes) — this is paid policy data, NOT a 50-episode library: label it
   "50 + 250 policy episodes" and report its row count.
2. Make AWM (pure cache, CL2 recipe) able to use it in two variants: (a) `refit`: PCA / whitening / calibration fitted on
   the grown library as for any library; (b) `frozen`: PCA, task centers, whitening and calibration taken from the
   deployed 50-library fit, grown rows projected through them (this mirrors in-place append on a GPU-resident bank).
   Use existing kwargs if the AWM code already accepts a library name; otherwise add a thin subclass in your own
   directory (rounds/r05/q4_growth/) — do not edit existing method files.
3. Offline check on the evaluation inits' recorded queries (25–49): nearest-neighbour density and action / successor
   losses for static-50, grown-frozen, grown-refit and the 500 library, as in C's library_study (no SR claims).
4. Arm specs rounds/r05/q4_growth/arms_q4.json (emit_arms format, `<RUN>` placeholders, pure cache, no judge, STAGE1
   only): {refit, frozen} × {l10, spatial} = 4 arms, plus a manifest file of the evaluation pairs (all ten tasks ×
   inits 25–49; the K4 manifest format `[[task_id, init], ...]` is accepted by `OSCL_MANIFEST` / the arm field
   `manifest`). The coordinator will compare against the existing full-run arms restricted to the same pairs (R2
   static 50 and 500 CL2). Give exact prefit commands and run the prefits yourself into
   /home/weiland/trace_runs/os_closed_loop/r05_growth/fits/ (report sizes and sha256).
</task>

<context>
Repository /home/weiland/projects/openpi (branch Ziyang). Round 5 of an action-cache exploration for VLA policies
(π0.5, GR00T N1.5) on LIBERO. Read first: exp/offline_search/rounds/r05/CODING_BRIEF.md (binding rules), rounds/r05/
SELECTION.md, rounds/r05/FINDINGS.md, rounds/r05/ideation_C/REPORT.md (+ its library_study.py), exp/offline_search/
harness/README.md (store layout, Method API), exp/offline_search/rounds/r02/g1_awm/awm.py, closed_loop/README.md
(manifests, arms). You own only rounds/r05/q4_growth/, the new store library directories you create, and
/home/weiland/trace_runs/os_closed_loop/r05_growth/. Closed-loop chains are running on this machine: never touch
servers, ports 23150-23169, tmux sessions or processes you did not start.
Python: /home/weiland/projects/openpi/.venv/bin/python from the repo root.
</context>

<hard_constraints>
- Your CPU range: 26-29,70-73. Prefix every python/numpy command with `taskset -c 26-29,70-73` and set
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; never more processes than threads in your range. CPUs 38-43
  and 82-87 belong to another project: never use them.
- GPU: none (CUDA_VISIBLE_DEVICES='').
- Never modify or delete anything that already exists in the store; only add new library directories. Check free space
  on /dev/shm before writing (report it) and keep the grown libraries' size reasonable.
- No git. No `rm -rf`. Never `pkill -f`. No servers, timan107, LIBERO workers or chains. Do not read tests/review_tests/.
</hard_constraints>

<completeness_contract>
Resolve fully: library build + validation in both store copies, method variants, offline check, arm specs, manifest,
prefits, hand-back. State any cell you could not build and why.
</completeness_contract>

<verification_loop>
Re-run the store validators and a fresh-process load/fit/query of every prefit artifact before finishing.
</verification_loop>

<grounding_rules>
Report only what you ran and observed; exact commands, numbers and file paths. Label anything unverified.
</grounding_rules>

<structured_output_contract>
Write exp/offline_search/rounds/r05/q4_growth/HANDBACK.md (provenance and row counts, files, switches, offline table,
arm specs, prefits, coordinator next steps, caveats), then print a one-paragraph summary.
</structured_output_contract>
