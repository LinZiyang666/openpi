<task>
Make the R17 cache-study analysis actually produce numbers now, by removing its gates, then run it on CPU and hand back the tables.

File: `exp/offline_search/rounds/r17/cache/analyze_cache.py` (752 lines; test file `test_analyze_cache.py`). Spec of the statistics: `exp/offline_search/rounds/r17/cache/CACHE_STUDY.md` §5 and §9.2–9.3. The statistics are right; the problem is the gating. Today it reports 0/240 cells because:
- `receipt_snapshot()` only admits lanes whose RECEIPT is `PASS` + `all_owned_closed` + cleanup confirmed; everything else is `PENDING_SEAL` and never read.
- `load_lane()` additionally requires a PASS `SUMMARY.json`, recovered-result seals, and many `require(...)` identity checks (`validate_bundles`, `check_fit_selection`, plan/manifest identity, `Sources.sha` pins).
- `evidence_rows()` requires resume sources to be unchanged (`Resume source changed`) and closed (`Resume parent not closed`).
- `accepted_terminal()` aborts on a duplicate accepted row instead of keeping one.

The owner has ruled that this kind of defensive gating is not wanted. Results must be readable for finished and running lanes alike.
</task>

<what_to_change>
1. Read any lane regardless of receipt status. No PASS / closed / SUMMARY / seal / sha requirement. Record the receipt status in the report as information only.
2. Resume chains: a lane's `campaign.json` → `cache_resume_selection` (or `CACHE_RESUME.json` → `source`) names the previous segment; keep following it. Union accepted terminal rows across the chain. If the same `task_uid` is accepted in more than one segment, keep the latest by `ts`. Do not require sources to be unchanged or closed.
3. Duplicate accepted rows within one journal: keep the latest by `ts`, do not abort.
4. Remove identity/pin `require`s (bundle, fit selection, plan/manifest identity, sha pins, nested-view checks). Keep only what is needed to map each planned task to (model, library size N, group, suite, canonical task/init key): read it from the plan's `extra` fields (`library_size`, `cache_group`, etc.) and `assets/CACHE_STUDY.json` as today.
5. Interim cells: a cell with fewer than 500 accepted episodes is reported with its actual n and status INTERIM. Paired comparisons use only keys present in both arms and print the pair count. Do not invent or impute missing outcomes.
6. Per-step cost evidence (decisions/ACKs) may be missing for episodes still running: compute costs over the episodes that have it, report the count, and do not abort.
7. Keep unchanged: CP and bootstrap paired statistics, the 40-comparison families (F500: each group vs pure-inference control; Fsize: N50 vs N500), the fixed 32-init sensitivity exclusion from `CONTAMINATION_DISTANCES.csv`, Cosmos aliases (ch/chr/chg/chrg reuse c/cr/cg/crg), cost metrics, the md/json/csv outputs and the plots.
8. Do not add any new checks, hashes, seals or validation layers. Delete code that becomes dead. Update `test_analyze_cache.py` only as needed so it passes (drop tests that only exercised removed gates).
</what_to_change>

<lanes>
All under `/data/mw_finetune/eval/` (read-only):
- DP N500: final resume segment `dpc8713063796_r2_d1dd` (chain back to `csl10d`).
- Fast-WAM N500: final segment `fastwamc5481985003` (chain back to `csl10f`).
- Pure-inference controls: DP `hs5dp_ctl_142812`, Fast `hs6fw_res_152931`, Cosmos `formal_cosmos_libero_official2000_20261010_100533_5f558380` plus its resume `formal_cosmos_libero_official2000_20261010_130639_a8ca2043`.
- Running, sizes 50–400 (interim): Fast `fastwamc6066547406`, DP `csztd_r2_c383_r2_e5b0` (chain back to `csztd`).
- Cosmos has no cache lane yet.
If the current CLI's control discovery (`--eval-root`) does not find these controls, make it take them explicitly; keep it simple.
</lanes>

<run>
CPU only, read-only on /data. Run:
`.venv/bin/python exp/offline_search/rounds/r17/cache/analyze_cache.py --sizes 50 100 200 300 400 500 --lane <each lane above>`
and the test file. Output stays in `exp/offline_search/rounds/r17/cache/` (CACHE_RESULTS.md/.json/.csv + svg).
</run>

<action_safety>
Experiments are running on the GPUs and import code from this repository. Do not start, stop, resume or touch any campaign, process, queue, GPU, or remote node. Do not edit anything outside `exp/offline_search/rounds/r17/cache/`. Do not git commit.
</action_safety>

<handback>
Reply with: (a) a short summary of what you removed and what you kept; (b) the N500 table per model × group × suite: n, SR, Δ vs control with discordant g/l and the family CP lower bound; (c) the interim size table (n and SR per cell so far); (d) the exact command to rerun later with more lanes; (e) anything that looked wrong in the data (missing evidence, duplicate rows, odd counts).
</handback>
