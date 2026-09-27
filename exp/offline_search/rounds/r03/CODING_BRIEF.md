# R3 coding brief (common to the 4 coding agents H1–H4)

Read first: `logs/offline_search_exploration.log.md` (protocol: §3.1 valid dims, §5 harness, §8 discipline, §9 owner
rulings), `exp/offline_search/rounds/r03/SELECTION.md` (the approved list — your family's row is your task),
`rounds/r03/FINDINGS.md` (R2 numbers), the ideation originals `rounds/r03/NOTES_ideation_A.md` (closed-loop trace
mining: deadlock spells, trap taxonomy, proposals P1–P4), `NOTES_ideation_B.md` (mixed HIT/MISS judge: facts F1–F11,
proposals P1–P4, mixed-mode plan), `rounds/r03/ideation_C/REPORT.md` (codex ideation C), and their scripts in
`rounds/r03/diag_A/`, `diag_B/`, `ideation_C/`. Code you build on: `rounds/r02/g1_awm/awm.py` (AWM + the G3 base
contract), `rounds/r02/g3_recovery/` (V6 `StuckRecovery`, V7 `DriftCalibratedConfidence`, README = exact algorithms
and the base contract), `exp/offline_search/closed_loop/` (plugin, ops, README), `harness/README.md`.

Everything from `rounds/r02/CODING_BRIEF.md` still holds (Method API, valid dims, online-legal inputs, extras, speed,
memory, smoke commands, no git, no `rm -rf`, never `pkill -f`, do not read `tests/review_tests/`) unless overridden
here.

## What changed since R2 (why R3 looks like this)
- **Closed loop is the exam; offline mean err is NOT the gate.** No offline proxy orders both suites (A §B). R3
  methods are screened by a 100-episode closed-loop **pilot** per π0.5 suite (5 trap tasks × 20 inits: spatial
  tasks {6, 9, 0, 4, 1}, l10 tasks {0, 4, 6, 8, 7}; `OSCL_EPISODES=0,…,19 OSCL_TASKS=<tasks>` in `chain.sh`), then
  the winners run 500 episodes on all four cells. The offline harness batch still runs (three-layer decomposition,
  library scale, regression check), but a method is not dropped for a small offline err loss.
- Pure-cache failures are step-cap timeouts caused by deadlock spells at grasp / release / terminal rows; library-side
  recovery does not escape them (CL3 ≈ CL2). Pure cache must **prevent** trap entry (H1); a mixed system must **MISS**
  at trap entry (H2 + H3).
- Mixed HIT/MISS is allowed in R3 (owner): a MISS runs the full model on the server (stage 2/3), the executed chunk is
  then the policy's.

## Where / how
- Code: `exp/offline_search/rounds/r03/<family_dir>/` (`h1_trap/`, `h2_mixed/` for anything that is not a plugin edit,
  `h3_judge/`, `h4_kpi/`; add `__init__.py`). H2 and H4 may edit / add files under `exp/offline_search/closed_loop/`
  (the plugin and ops are this line's code; keep every existing flag and the pure-cache behaviour byte-identical when
  the new flags are absent — R2 arms are still running on it, see "Live system" below). Nobody edits `harness/`,
  `profile/`, `rounds/r01|r02/` (subclass / wrap instead) or anything under `src/`.
- Offline deliverable (H1, H3): `<family_dir>/batch.json` in the R2 format (`method`, `kwargs`, `family`, `cells`,
  `subsample`, `allow_gpu_fit`) for every variant of the full offline round; the coordinator runs it with
  `harness.batch`. Do not run full-scale yourself.
- Closed-loop deliverable (H1, H3, H2 for its control arms): `<family_dir>/arms_pilot.json` (and `arms_full.json` where
  relevant) = `emit_arms.py` spec rows (`name` [A-Za-z0-9_], `model`, `suite`, `mode: plugin`, `method`, `kwargs`,
  `plugin_args` incl. `--os-fit-artifact <RUN>/fits/<name>.pkl` and, for mixed arms, the `--os-judge …` flags), with
  `<RUN>` as a literal placeholder the coordinator substitutes, plus a prefit note (command, time, pickle size). Name
  arms `r3p_<p|g>_<sp|l10>_<short>` (pilot) / `r3f_…` (full) / `r3mx_…` (mixed); ≤ 40 chars.
- Store: `/dev/shm/offline_search_store` (hot arrays in RAM). Big arrays you create go under
  `/home/weiland/trace_runs/offline_search_store/derived/r03/<family_dir>/`, never in the repo tree.
- Closed-loop logs to mine (read-only): `/home/weiland/trace_runs/os_closed_loop/r02_g50/runs/<arm>/` (server
  decision logs `server_<port>/decisions_*.jsonl`, pulled journal / per_step, `summary.json`), arms
  `oscl50_{p,g}_{sp,l10}_cl{0,1,2,3}` (π0.5 all done; GR00T arriving), and later `r02_g500/…` (`oscl500_*`).

## Rules for R3 methods (on top of R2's)
- ⛔ Vision mandatory in every regime (step 0, after a MISS, after a HIT): selection and/or confidence must use the
  policy's own visual keys. State / continuity / gripper / progress signals only as auxiliary terms or guards.
- **Mixed-mode correctness.** Any per-episode state that depends on what was *executed* (served gripper sign, anchor
  row, run length, dwell counters) must be derived from `q.prev_a_exec` / `q.prev_hit` / `q.hist_*`, not from the
  method's own last output — in mixed mode the executed chunk after a MISS is the policy's (`prev_hit=False`).
  Offline cache cells (always HIT) and inf cells (always MISS) exercise both paths; smoke both.
- Keep the G3 base contract (`os_score_all`, `os_library`, `os_fit_library`, `synth_k`, `synth_T`, `os_synth`,
  `os_confidence`) on every selector you build, so V7 / guards can wrap it; check with
  `rounds/r02/g3_recovery/tools/contract_check.py` (must print `"PASS": true`).
- Three-layer decomposition (§9) in every hand-back: synthesis effect / method effect at fixed library / library
  effect; any fit using data beyond the deployed candidate library is labelled **"borrowed big-library information"**
  (`fit_data="big"`, prior from bpool) and must have a non-borrowing twin.
- Library scale (§9) in every hand-back: episodes, entries, bytes/entry, total MB (fit pickle) at the current
  (~50-episode) and 10× (500-episode) library vs the deployed pkl (π0.5 431 / 1103 MB, GR00T 429 / 1068 MB).
- Closed-loop readiness as in R2: `query()` ≲ 5 ms single-thread on the 10× library; picklable fit artifact;
  `closed_loop/selftest.py` passes for your method (pure-cache mode; H2 adds mixed mode).

## ⛔ CPU / GPU / network budget (owner rule, mandatory)
- Your logical-CPU range is in your prompt. Prefix EVERY python/numpy command with `taskset -c <range>`; processes ≤
  threads in the range; OMP/MKL/OPENBLAS threads = 1 per process (one BLAS-heavy one-off may use the whole range).
- ⛔ CPUs 38-43,82-87 belong to another project; never use them. The closed-loop servers use 0-17,44-61 and the chain
  34-37,78-81 — never put work there.
- GPU (H2 only, smoke): weilandserver 4090, 49 GB; the R2 chain's 4 stage-1 servers use ≈ 7.5–9 GB. You may run at
  most 2 full-model smoke servers at once (π0.5 ≈ 7.6 GB, GR00T ≈ 5.7 GB) after checking `nvidia-smi` free memory.
  Ports: scan `ss -ltnH | grep -oE ':231[0-9]{2}\b'` and use free ports in 23160–23189 only (never 23150–23153; other
  sessions share the 23100–23199 range and tmux names — never touch a port / tmux session you did not create).
- timan107 (48 cores) is saturated by the R2 chain's 64 workers: H2 may run ONE tiny closed-loop smoke there
  (≤ 4 workers total, ≤ 10 episodes, its own run root `/home/weiland/trace_runs/os_closed_loop/r03_smoke`, its own
  `tmux -L oscl` session names `oscl_r3s_*`), nothing else. Everybody else: no timan107.

## Live system (do not break it)
The R2 closed-loop chains (`tmux oscl_chain_g50c`, then `oscl_chain_g500b`) run the current plugin / ops code from
this working tree for every new arm (servers import `exp/offline_search/closed_loop/plugin.py` at start; `chain.sh`,
`start_server.sh`, `collect.py` are re-read per arm; `run_arm.sh` / `count.py` live on timan107). Therefore:
- Edits to `closed_loop/*.py` / `ops/*` must keep the pure-cache path byte-identical when the new flags are absent and
  must never be left in a broken intermediate state: develop in a copy (e.g. `closed_loop/_dev/` or your family dir),
  run the selftests there, then install the final version in one step (write to a temp file in the same dir and
  `mv` over the original) and re-run `selftest.py` for the pure-cache mode immediately.
- Do not run `sync_remote.sh` for the R2 run roots; for your smoke push only into your own run root's arms.

## Hand-back (final message)
File list; batch.json / arms json summary (variants, names); smoke numbers per method (err, AURC, regime split,
gripper-sign-faithful err for synthesized actions, the KPI your family targets); fit time, ms/query, pickle size;
three-layer and library-scale statements; caveats. Keep it factual; the coordinator runs the full offline batch and
all closed-loop arms.
