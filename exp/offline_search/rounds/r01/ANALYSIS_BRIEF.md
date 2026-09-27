# R1 analysis brief (fable analysis agent)

Read: `logs/offline_search_exploration.log.md` (protocol, esp. §4 metrics, §6 step 5, §10 ledger), `rounds/r01/FINDINGS.md`
(R0 facts), `rounds/r01/SELECTION.md` (what was run and why), the ideation originals `NOTES_ideation_{A,B}.md`, each
family's code/notes under `rounds/r01/f{1..4}_*/` (F4 also has `m9_tables.md`, `m9_results.json`), `harness/README.md`
(output formats) and `profile/README.md` (tools).

Data: run outputs `exp/offline_search/results/r01/<method>/<cell>.{npz,json}` (+ R0 baselines in `results/r00/`),
scoreboard `exp/offline_search/results/scoreboard.csv` (round r00/r01; take the latest run_ts per method×cell),
coverage caches under the store `profile_cache/`. Store root `/dev/shm/offline_search_store`.
Regimes: inf cells = every step≥1 decision follows a MISS (fresh tail); cache cells = every step≥1 decision follows a
HIT (stale tail). A deployed cache alternates; report both and reason about the mix.

Deliver (write to `rounds/r01/ANALYSIS.md`, and return the same text):
1. Per-method verdict table: keep / refine / drop, with the deciding numbers (err mean & median, regret vs current and
   vs big-library oracle, AURC, risk@30/50, grip_mis, bad rate, ms/query at timing_concurrency 4, bytes/entry, fit_s),
   separately for fresh (inf) and stale (cache) regimes and per model. Use the profile tools:
   `compare` (episode-bootstrap CIs) of each family's best variant vs B0 and vs the best overall; `breakdown` slices
   (third, grip transition, outcome, conf decile); `timeline` (stay/track/switch) for the top methods; `explain` on the
   worst decisions of the top methods; `coverage` for the big libraries.
2. Cross-method findings: what actually carries the gain (library size vs representation vs continuity vs synthesis vs
   confidence), where each regime's error now comes from, and what the M9 vision probe says about the value of visual
   information and which representation captures it.
3. Failure profile of the current best per regime (which slices dominate the remaining error; gripper transitions,
   late steps, specific tasks, drifted states).
4. Risks the offline harness cannot see (closed-loop compounding, regime mixing, confidence scale across regimes) and
   what cheap offline proxy could partially address each.
5. Recommendations for round 2: which ideas to refine (with concrete parameter/mechanism changes), which to drop, and
   2–4 new directions suggested by the data; plus whether any closed-loop check should be proposed to the owner now.

Rules: CPU only; run tools with `taskset -c <range given in your prompt>` and ≤ that many processes; read-only except
`rounds/r01/ANALYSIS.md` and scratch under `/home/weiland/.claude/jobs/a607dd74/tmp/analysis_r01/`; no git; no repo
edits elsewhere. Be quantitative and skeptical: check that apparent gains are not artefacts (regime identity, the
metric's executed-segment definition, duplicate variants, library overlap with queries, teacher-noise floor).

## ⛔ Library scale in every conclusion (owner rule)
Every result or comparison must state the library it used: episodes, entries, key bytes/entry, action bytes/entry,
total library volume (+ fixed overhead such as PCA bases), next to the current deployed library (pkl on disk:
π0.5 431 MB spatial / 1103 MB l10, GR00T 429 / 1068 MB; 262 KB key per entry). Until the owner settles whether the
10× library is deployable, report every conclusion both at current library size (~50 episodes) and on the 10× library.
