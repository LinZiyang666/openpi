# Round 5 (opus) — predictions, written 2026-10-02 04:18 CDT, BEFORE any arm was emitted

Screen: `/home/weiland/trace_runs/os_closed_loop/r09_opus_r5`, LIBERO-10, 50-demo library, tasks 0-9 x inits 20-29
(100 pairs per arm). Thresholds were chosen on inits 0-19 only (`tools/fit_thresholds.py`): **L0 = 2, C = 20** for both
models. Basis: `out/predict.json` (IR from the stacks' own screen ledgers minus the calls the gates drop; SR from the
fit-inits prefix simulation `out/gatesim.json`, cross-checked on inits 20-29 with the corrector-only runs as the gated
prefix).

| arm | π0.5 SR | π0.5 IR | GR00T SR | GR00T IR |
|---|---|---|---|---|
| `stack` (fable r3c, re-run as same-batch control) | ≈ .89–.92 (r3c run .920; its sibling run .880) | .162 | ≈ .87–.89 (r3c .890; sibling .880) | .183 |
| `stack_P2` (on-pace silence) | stack − 0.6 pp (stack-specific cross-check) … − 2.6 pp (fit sim, 90% [−4.7, −1.0]) | **.135 (−16%)** | stack + 0.6 pp (fit sim, 90% [−1.4, +2.7]); cross-check +1.3 pp | **.146 (−20%)** |
| `stack_P2C20` (+ 20-call budget) | stack − 2.1 pp (cross-check) … − 3.3 pp (fit sims added) | **.126 (−22%)** | stack + 0.2 pp (fit sims added); cross-check +1.3 pp | **.136 (−26%)** |

Mechanism predictions (checkable in the server logs, `extras.r9o5_gate`):
- gate P drops ≈ 1.7 (π0.5) / 2.5 (GR00T) no-progress calls per episode in expectation (it fires on ≈ 2.4 / 3.0
  looks per episode; ≈ 20–30 % of those stalls come back as a call one look later);
- gate C binds in ≈ 6 (π0.5) / 10 (GR00T) episodes, almost all of them failures that latch to the step cap;
  at most 1 success in each model needed more than 20 calls in the r3c run;
- the escalation rule (π0.5) is untouched: gate P never drops an escalation call; only gate C can end a takeover;
- remaining no-progress calls should sit at pace lag ≥ 3 (none at lag ≤ 2 in either gated arm).

Verdict rule I will apply to the screen: the gates are a success if IR falls by at least the predicted amount
(± 0.01) and the same-batch SR difference to `stack` is within the run-to-run churn (≤ 5 pp on 100 pairs).
π0.5 does **not** reach the requested 25 % cut at a predicted loss under ~1 pp; GR00T does.
