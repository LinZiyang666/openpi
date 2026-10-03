# R11 dev-set baselines (knob off = R10Recipe 3-layer), coordinator, 2026-10-02 21:5x CDT

Non-test closed loop: B-pool inits outside each served subset library (sol dev-set machinery, `closed_loop/devset.py`), 5 inits/task × 10 tasks = 50 episodes per arm, seed 20261003. `guard_call_share` = m / v from the cost ledger = share of real looks that called the policy (all calls here are guard calls). Pure-policy dev reference = stored parent-library outcomes on the same inits (sol `pure_policy_reference.json`).

| root | arm | SR | owner IR | guard_call_share | pure-policy ref |
|---|---|---|---|---|---|
| r11_dev_current | pi05 l10 curated-50 | .78 | .182 | .248 | .84 |
| | pi05 spatial curated-50 | .86 | .135 | .133 | 1.00 |
| | groot l10 curated-50 | .78 | .209 | .315 | .82 |
| | groot spatial curated-50 | .84 | .140 | .149 | .90 |
| r11_dev_size50 | pi05 l10 50 | .84 | .164 | .204 | .84 |
| | pi05 spatial 50 | .86 | .149 | .167 | .98 |
| | groot l10 50 | .74 | .208 | .311 | .88 |
| | groot spatial 50 | .88 | .132 | .130 | .90 |
| r11_dev_size200 | pi05 l10 200 | .86 | .149 | .170 | .76 |
| | pi05 spatial 200 | .96 | .109 | .071 | .96 |
| | groot l10 200 | .88 | .184 | .255 | .90 |
| | groot spatial 200 | .92 | .118 | .096 | .88 |

Use: checking the knob → IR map's only offline unknown (closed-loop guard rate). Knob settings stay library-only (owner: LOEO-constructed knob); these numbers must not feed back into settings for the frozen grid. n = 50 per arm, so SR here is ±7 pp noise.
