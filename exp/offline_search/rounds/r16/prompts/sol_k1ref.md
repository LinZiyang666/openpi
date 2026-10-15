<task>
Rerun of R16's k = 1 per-task IR knob for π0.5 and GR00T on LIBERO, with the reference changed from full-step pure inference to k = 1 pure inference (owner order 2026-10-13). Repo /home/weiland/projects/openpi. You write and CPU-test code and hand back exact commands; the coordinator launches every server, worker, chain and monitor.

Scope: k = 1 only; π0.5 and GR00T × libero_10 / libero_spatial × library sizes 50 / 100 / 200 / 500 = 16 cells. Everything else is unchanged from R16: library pkls, fits, guard, corrector, knob levels, cost pricing (`rounds/r16/cost.py`, D5), horizon, identities, UIDs, seeds, pools. Read first: `rounds/r16/{COORDINATOR_DECISIONS.md,REPORT.md,RUNBOOK_CONDUCTOR.md,HANDBACK_CONDUCTOR.md,HANDBACK_RETEST.md}`, `analyze.py`, `conductor*.py`, `retest*.py`, `dev_results_conductor/`, `retest_v1_v2_v3_results/`.

Owner rules (binding):
- The reference is pure inference with the same steps: k = 1 pure C10. Dev reference = `/home/weiland/trace_runs/os_closed_loop/r16_dev_c10_{model}_{suite}_50_k1` (300 B episodes, all 10 tasks; pure inference does not use the library, so it serves every size). Test reference = R16 phase-1 k = 1 test-A arms (500 A episodes per model × suite, paired by init).
- Dev descent: per task, order c10 → ir45 → ir40 → ir32 → ir25 → off, 30 B inits; a level passes iff 100·(level successes − k1 C10 successes) ≥ −5·30 (integers); the first failure keeps the previous level; c10 is the top (k1 pure).
- Test raising: per task (50 A episodes), deficit d = reference successes − task successes. d ≤ 2 passes; d = 3–4 → one level up; d ≥ 5 → c10. Re-run only raised tasks, repeat until every task passes or sits at c10. Integer comparisons only.
- Full-system rule: per cell, the full system with the knob off (every task at `off`) on test A. If its cell SR ≥ the k1 pure test SR of the same model × suite (500 episodes, paired), that cell is not tuned: it reports the knob-off SR and IR.
- Reuse every outcome that has the identical arm definition, level, k, identities and UIDs: R16 dev cohorts, R16 test v1 arms and retest generations v1/v2/v3. Measure only what is missing. Never rewrite a frozen R16 root, table or output.

My CPU pass from `dev_results_conductor/dev.json` (check it): with the k1 reference, 31 of 160 (cell, task) levels change among measured levels, and 19 tasks need dev levels never measured (R16 stopped at its old first failure). Your code must reproduce this from source data and list the 19 with the next level to measure.

Deliver:
1. A new output tree `rounds/r16/k1ref/` (code + results), not touching R16 frozen files.
2. Dev supplement: adaptive groups that measure only the missing (cell, task, level) cohorts, one level at a time per task with first-failure stop, on the existing conductor machinery; then a freeze of the 16 k1-reference tables.
3. Knob-off test groups: for each of the 16 cells, the tasks whose `off` outcome on test A is not already measured (50 A episodes each).
4. Test groups for (cell, task, frozen level) not already measured on test A; then the raise loop (generation-by-generation emit from the previous analysis, like retest v1–v3).
5. Analysis: per cell SR / IR / paired difference vs k1 pure test (same estimators as R16), the full-system-rule decision per cell, per-task final levels, and a table in the format of `retest_v1_v2_v3_results/RETEST.md`. Also a JSON with one row per model × suite × size: sr, baseline_sr (k1 pure), diff_pp, ir, pure_ir, rule (knob | full_system).
6. Runtime: server on the local RTX 4090 (whole card is available; keep R16's process layout: π0.5 3 processes, GR00T 4, unless you measure otherwise), workers on timan107 through the existing island `/scratch/zixuans8/openpi_r16_conductor_v1` (its `worker.sha256` verifies today). Local code: three files changed after R16 (`packages/openpi-client/src/openpi_client/msgpack_numpy.py`, `src/openpi/cache/groot/batcher.py`, `src/openpi/cache/groot/staged.py`, all R17 opt-in changes). Run the server from an immutable snapshot in which `conductor_deployment/local.sha256` verifies fully (e.g. a `git worktree` at HEAD under /home/weiland/projects/ with the three files restored from 231a1cf7 / 30505063 and the R16 groups/fits included); give the build command.
7. Ports: driver 23159, servers 23150..23155 (RUN_CONFIGS legacy R16 allocation). 23179 is held by a live Fast-WAM campaign driver: refuse it.
8. CPU tests, then a handback `rounds/r16/k1ref/HANDBACK.md` with the exact commands in order: snapshot build/verify, smoke (2 episodes per arm, 4 workers), knob-off test, dev supplement, freeze, test, each raise generation, analysis.
</task>

<rules>
- CPU only: no servers, GPU, remote commands, git commits, pkill, rm -rf. Never touch credentials.
- tether: never push/pull/expose. Worker files reach timan107 only by rsync over the public network; if the island needs a change, list the files and the rsync command and leave it to me.
- Keep it simple: reuse R16 modules; add the smallest new code.
- If a source fact contradicts this brief, stop and report it instead of guessing.
</rules>

<output>
Final message: what was built, the reproduced 31 / 19 counts with the 19-task list, the number of episodes each phase will run, and the handback path.
</output>
