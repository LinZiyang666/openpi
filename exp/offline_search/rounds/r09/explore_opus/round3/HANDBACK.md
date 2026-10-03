# Opus round 3 — hand-back

## What ran
- Local CPU only (no GPU, remote, sync, server, worker or chain). No process left running. No git change.
- Rules: holdout roots refused by prefix (`r09_holdout*`, `r09_astra_holdout*`; tested); my round-2 catalog ran at
  00:33 CDT, before those roots existed. Screen analysis uses only my own inits 20–29 runs; everything fitted or chosen
  (trigger constants from round 2, homing maps) uses inits 0–19. Nothing task-indexed.
- **CPU slip:** one analysis command (`tools/evidence.py`, 31 s) ran with `taskset -c 10-13`, outside my assigned
  CPUs 2-9,46-53. All other commands used my range.
- `REPORT.md` may be blocked by the agent harness (it blocks report files for subagents); its text is in my final
  hand-back message — please save it as `round3/REPORT.md`.

## Files (`exp/offline_search/rounds/r09/explore_opus/round3/`)
| path | content |
|---|---|
| `DATA_ANALYSIS.md` | screen dissection, rejected fixes, homing evidence, predictions |
| `PROPOSALS.md` | fixes + new lever, frozen arms, adoption bars |
| `methods.py` | `HomingEscalation` |
| `tools/dissect.py`, `tools/evidence.py`, `tools/action_map.py` | per-episode screen dissection; evidence (crosstab, proximity, futility, early, guard calls); model↔wire / wire→eef maps |
| `tools/prepare_arms.py`, `tools/deploy_sources.py`, `tools/reproduce.sh` | arms + artifacts; source install helper; full reproduction (≈ 1 min) |
| `tools/tests/test_round3.py` | 9 tests (23 with round 2) |
| `artifacts/r9o3_*.pkl` | 6 frozen prefits |
| `out/` | `dissect.json`, `episodes_*.parquet`, `evidence.json`, `action_map_*.json`, `prepare_checks.json`, `selftest_frozen_arms.log` |

Round-2 fixes also in this turn: `round2/tools/common.py` refuses every `r09_holdout*` root (test added);
`round2/tools/screen_analysis.py` reads GR00T trigger rows from server logs.

## Runs requested (coordinator; nothing launched)
Run root `/home/weiland/trace_runs/os_closed_loop/r09_opus_r3`, manifest `manifests/eval_inits20_29.json` (100 pairs).

| batch | arms |
|---|---|
| 1 (priority) | `r9o3_pi05_l10_50_cache r9o3_pi05_l10_50_esc_w24 r9o3_pi05_l10_50_home_w24 r9o3_pi05_l10_50_home_cache` (+ optional `r9o3_pi05_l10_P10`) |
| 2 | `r9o3_groot_l10_50_cache r9o3_groot_l10_50_esc_w24 r9o3_groot_l10_50_home_w24 r9o3_groot_l10_50_home_cache` (+ optional `r9o3_groot_l10_P10`) |

```bash
cd /home/weiland/projects/openpi
P=(taskset -c 2-9,46-53 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
R=/home/weiland/trace_runs/os_closed_loop/r09_opus_r3
ARMS=(r9o3_pi05_l10_50_cache r9o3_pi05_l10_50_esc_w24 r9o3_pi05_l10_50_home_w24 r9o3_pi05_l10_50_home_cache)
# 1. install sources in the h100 tree (new round3 files; round2 methods.py unchanged, sha 2ede2b5f...)
"${P[@]}" -m exp.offline_search.rounds.r09.explore_opus.round3.tools.deploy_sources            # plan
"${P[@]}" -m exp.offline_search.rounds.r09.explore_opus.round3.tools.deploy_sources --execute  # coordinator
#    round3/methods.py sha256 3ca5d59ed00ad3b8c8d580045c5247fc01bebbbf8cb4248668c572ee5402c410
# 2. plan + sync (your fleet and sync port)
WORKER_HOST=<fleet> "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control plan "$R" "${ARMS[@]}"
WORKER_HOST=<fleet> SYNC_PORT=<port> "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control sync --concurrent "$R" "${ARMS[@]}"
# 3. chain
WORKER_HOST=<fleet> PORTS=<p1,..,p4> WPS=<n> MAX_ATTEMPTS=2 POLL_SECONDS=30 OSCL_MANIFEST="$R/manifests/eval_inits20_29.json" \
  "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control chain "$R" "${ARMS[@]}"
# 4. readout: headline + both-crossed cross-tab
"${P[@]}" -m exp.offline_search.rounds.r09.explore_opus.round2.tools.screen_analysis --run-root "$R"
```
All homing / escalation arms are full-model with the pure-policy plugin flags (`--os-blind --os-policy-tail
[--os-policy-tail-blocks 1] --os-judge guard_only`); `_cache` is the R8 pure-cache spec. Homing arms log
`r9o3_mode` (0 cache, 1 homing, 2 policy window, 3 after), `r9o3_dist`, `r9o3_t_trig`, `r9o3_t_win` and `r9o_lag` in
every server decision record. Cross-tab on the new run root:
`"${P[@]}" -m exp.offline_search.rounds.r09.explore_opus.round3.tools.dissect --run-root "$R" --prefix r9o3 --pairs
pi05:50:esc_w24:cache,pi05:50:home_w24:cache,pi05:50:home_cache:cache,pi05:50:home_w24:esc_w24,groot:50:esc_w24:cache,groot:50:home_w24:cache,groot:50:home_cache:cache,groot:50:home_w24:esc_w24`
(writes `out/dissect_r9o3.json`; for homing arms the trigger step is `r9o3_t_trig`).

Estimated 25–40 min per batch. Bars and predictions: `PROPOSALS.md`.
