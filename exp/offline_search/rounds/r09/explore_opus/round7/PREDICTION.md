# Round 7 (opus) — predictions, written 2026-10-02 05:41 CDT, BEFORE any arm was emitted (run root did not exist)

Arms (`/home/weiland/trace_runs/os_closed_loop/r09_opus_r7`, LIBERO-10 / 50, tasks 0-9 x inits 20-29): the leading r3c
stacks as same-batch controls, and the same stacks with ONLY the per-episode guard-call budget (no pace gate):
C = 20 (fitted on inits 0-19 in round 5) and C = 15 (second dose point; its fit-simulated loss was above the 1 pp rule).
Class: round-5 `GatedNpGraspEsc3` / `GatedNpGraspStackGroot3` with `pace_lag_max=None` (gate P off). Gate C counts every
committed guard call (no-progress, escalation) and afterwards hands the rest of the episode to the corrected cache.
Basis: `out/predict7.json` (stack screen ledgers: fable r3c run and the round-5 control), round-5 `gatesim.json`
(fit inits), round-5 screen (P2C20 vs P2: +1 / −1 pp at −.029 / −.012 IR).

| arm | SR vs same-batch stack | IR vs stack | why |
|---|---|---|---|
| π0.5 `C20` | 0 … −3 pp (fit sim −0.8 [−1.2, −0.4]; screen hint +1) | **−6 … −10 %** (.162→.152 on the r3c run, .184→.166 on the r5 control) | 7-14 episodes exceed 20 calls, mostly escalated failures; 2-5 of them succeeded after > 20 calls (late escalation rescues) and are at risk |
| π0.5 `C15` | −1 … −5 pp (fit sim −1.3 [−1.9, −0.8]) | **−10 … −16 %** | 10-17 episodes over budget, 4-7 successes among them |
| GR00T `C20` | 0 … −1 pp (fit sim −0.4 [−0.8, −0.1]; screen hint −1) | **−7 … −8 %** (.183→.168, .187→.175) | 11-13 episodes over budget, all failures in both screen runs |
| GR00T `C15` | 0 … −2 pp (fit sim −1.8 [−2.4, −1.1]) | **−12 … −13 %** | 12-14 over budget, 1 success among them |

Mechanism check (server extras): `r9o5_gate == 2` only (no gate-1 decisions); calls per episode never exceed C;
budget-gated decisions concentrated in episodes that run to the step cap. π0.5's IR cut is larger on the round-5
control batch because that batch escalated more (4.2 escalation calls per episode vs 2.2 in the r3c run).
