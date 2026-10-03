# Opus round 4 — hand-back: escalation window stacked on the half-strength corrector

Nothing launched. All local work on CPUs 2-9,46-53; no holdout root touched (prefix refusal in every reader); no git change.
Homing (round 3) is recorded as negative in `round3/DATA_ANALYSIS.md` §5 and `round3/PROPOSALS.md`, and dropped.

## What is frozen
Run root `/home/weiland/trace_runs/os_closed_loop/r09_opus_r4`, manifest `manifests/eval_inits20_29.json`
(tasks 0–9 × inits 20–29, 100 pairs). Predictions were written first: `PREDICTION.md` (02:49:59 CDT).

| arm | what | method / artifact |
|---|---|---|
| `r9o4_pi05_l10_50_cache`, `r9o4_groot_l10_50_cache` | pure cache, R8 spec | BlindAWM, `r05_ptail/fits/r5t_p_l10_50_tail1uc.pkl` / `r05_x/fits/r5x_g_l10_50_tail1u.pkl` |
| `r9o4_pi05_l10_50_corr`, `r9o4_groot_l10_50_corr` | corrector alone: **exactly** fable's `r9f2_*_l10_50_corr05pt` spec and fitted artifact | fable `CorrectedCache` (blend .5, motion only, per-task heads fitted on inits 0–19), `r09_fable_r2/fits/r9f2_*_l10_50_corr05pt.pkl` |
| `r9o4_pi05_l10_50_corr_esc_w24`, `r9o4_groot_l10_50_corr_esc_w24` | the same fitted corrector as the cache + escalation window (lag 12, deadline 80, window 24; not retuned) | `round4/methods.py:CorrectedEscalateCalls`, `round4/artifacts/r9o4_*_corr_esc_w24.pkl` |

Correction path: `CorrectedEscalateCalls` is an R8 `AnchorCalls` controller; every fresh decision calls
`self.base.query(q)` = `CorrectedCache.query`, which applies the head and re-remembers the anchor, so cache decisions
serve the corrected chunk and the blind decision serves the corrected tail. No judge, no `os_score_all` / `os_synth`
(the plugin uses those only with `--os-gpu-retrieval`, not set). Verified:
- unit tests (`round4/tools/tests/test_round4.py`, 7 pass): on real library queries for both models the stack's served
  chunk is byte-identical to fable's standalone corrector artifact, has the same top-k as the plain cache, **differs from
  the plain cache on motion channels 0–5 of the first 10 rows only** (gripper, padding and rows ≥ 10 identical; non-zero
  difference on ≥ 15 of 20 queries), the blind decision serves the corrected tail; a forced trigger at decision 4 calls
  the policy at decisions 4–26 exactly (24 decisions) then stops; the frozen artifacts reproduce fable's corrector
  output exactly; the fit refuses a mismatched blend or cell.
- CPU plugin selftest (real per-connection stack, `--blind --policy-tail --judge guard_only`): see the table below.

| selftest | result |
|---|---|
| `r9o4_pi05_l10_50_corr` (frozen) | PASS, 48 decisions, 0 MISS |
| `r9o4_pi05_l10_50_corr_esc_w24` (frozen) | PASS, 0 MISS (no episode falls behind pace in 12 replayed decisions) |
| π0.5 stack, forced trigger at decision 2, window 24 (test twin) | PASS, 20 MISS + 20 policy tails (calls at decisions 2–10 of each 12-decision episode) |
| π0.5 stack, forced trigger at decision 2, window 4 (test twin) | PASS, 8 MISS + 8 policy tails (decisions 2 and 4), corrected cache afterwards |
| `r9o4_groot_l10_50_corr` (frozen) | PASS, 0 MISS |
| `r9o4_groot_l10_50_corr_esc_w24` (frozen) | PASS, 0 MISS |
| GR00T stack, forced trigger at decision 2, window 24 / window 4 (test twins) | PASS, 20 MISS + 20 policy tails / 8 MISS + 8 policy tails |

Plugin-level correction check (same replayed episodes, paired by episode uid and decision): the frozen stack serves
chunks byte-identical to the frozen `_corr` arm at all 48 decisions per model, and differs from the frozen `_cache` arm
on the motion channels at all 24 fresh decisions per model (median max |Δ| .069 π0.5, .053 GR00T) with **zero**
difference on the gripper channel.

`force_trigger_at` exists only in the test twins (`/tmp/opus_r4_test_*.pkl`); the frozen kwargs do not contain it
(asserted by a test).

## h100 files to deploy / verify (sha256)
New:
- `exp/offline_search/rounds/r09/explore_opus/round4/__init__.py` e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
- `exp/offline_search/rounds/r09/explore_opus/round4/methods.py` e3816f4bdcc21cea4d91557eec6d2b5d6f800eb9dad412b10ed519d92606829c

Must already be present (imported):
- `exp/offline_search/rounds/r09/explore_opus/__init__.py` e3b0c442…b855
- `exp/offline_search/rounds/r09/explore_opus/round2/__init__.py` e3b0c442…b855
- `exp/offline_search/rounds/r09/explore_opus/round2/methods.py` 2ede2b5f16a59700bd62b4927aaf763d54da566f5a38abcb2980ceb44d03ef0b
- `exp/offline_search/rounds/r09/explore_fable/__init__.py`, `explore_fable/round2/__init__.py`,
  `explore_fable/round2/tools/__init__.py` e3b0c442…b855 (empty)
- `exp/offline_search/rounds/r09/explore_fable/round2/tools/methods.py` 86774b83bb24309b38e64bd419f901c17ccad07c1195e8a438136eda1aefc72b
  (the version fable reported pushing; contains `CorrectedCache`)

Artifacts (synced by `control sync` from the plugin args):
- `round4/artifacts/r9o4_pi05_l10_50_corr_esc_w24.pkl` a123099239bd4d469fd572cde819d92d6d4268d4daf7701ddb399adf8ced6457
- `round4/artifacts/r9o4_groot_l10_50_corr_esc_w24.pkl` f781fa2722ca7eafd3b5a5a6beaa5deae06546362d6b4155232cbdfe99853b53
- fable's `r9f2_pi05_l10_50_corr05pt.pkl` 59e5e288cfae66dfd44465da99ee2a79c4df33156c7edb6012591f26ccebc992,
  `r9f2_groot_l10_50_corr05pt.pkl` 90733b3a2be1bd824d0f3a1167f1ef36e7fb5bd56236ffff7533f385724fd44c (control arms; the
  stack artifacts embed the same fitted corrector and do not read these files or the head npz at serve time)

`round4/tools/deploy_sources.py` prints this list; `--execute` installs with the same refuse-on-difference rule as
before.

## Commands (coordinator)
```bash
cd /home/weiland/projects/openpi
P=(taskset -c 2-9,46-53 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
R=/home/weiland/trace_runs/os_closed_loop/r09_opus_r4
ARMS=(r9o4_groot_l10_50_cache r9o4_groot_l10_50_corr r9o4_groot_l10_50_corr_esc_w24 r9o4_pi05_l10_50_cache r9o4_pi05_l10_50_corr r9o4_pi05_l10_50_corr_esc_w24)
"${P[@]}" -m exp.offline_search.rounds.r09.explore_opus.round4.tools.deploy_sources            # list + shas
"${P[@]}" -m exp.offline_search.rounds.r09.explore_opus.round4.tools.deploy_sources --execute  # install new files
WORKER_HOST=<fleet> "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control plan "$R" "${ARMS[@]}"
WORKER_HOST=<fleet> SYNC_PORT=<port> "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control sync --concurrent "$R" "${ARMS[@]}"
WORKER_HOST=<fleet> PORTS=<p1,..,p4> WPS=<n> MAX_ATTEMPTS=2 POLL_SECONDS=30 OSCL_MANIFEST="$R/manifests/eval_inits20_29.json" \
  "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control chain "$R" "${ARMS[@]}"
# readout: headline, measured window success, and the both-crossed cross-tab against the corrector control
"${P[@]}" -m exp.offline_search.rounds.r09.explore_opus.round3.tools.dissect --run-root "$R" --prefix r9o4 \
  --pairs pi05:50:corr_esc_w24:corr,pi05:50:corr:cache,groot:50:corr_esc_w24:corr,groot:50:corr:cache
```
The stack arms log `r9o_lag`, `r9o_escalated`, `r9o_esc_step` in every server decision record (the cross-tab reads them).
Local dry plan: exit 0, 53 files, 15.78 GiB logical (`r09_opus_r4/h100_sync/plan.json`).
