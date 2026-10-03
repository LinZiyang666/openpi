# Round 6 (opus) — predictions, written 2026-10-02 05:36 CDT, BEFORE any arm was emitted (run root did not exist)

Screen: `/home/weiland/trace_runs/os_closed_loop/r09_opus_r6`, LIBERO-10 / 50 demos, tasks 0-9 x inits 20-29.
Trigger fitted on inits 0-19 (`out/exhaust.json`): the top-1 row is the LAST row of its demonstration (end_rows 0)
and the episode is still running >= 2 decisions after the first such look (dwell 2) -> 24 decisions of policy at every
fresh decision, then back to the stack; once per episode. Basis: `out/exhaust.log`, `out/predict6.json` (replay of the
fitted trigger on the three screen runs of each stack, 300 episodes per model).

**Headline prediction: the takeover arms will be indistinguishable from their same-batch controls.** In the leading
stacks the no-progress guard already reacts to an exhausted script 2 decisions after the demo end (median), i.e. at
the same look the fitted trigger would (any trigger firing earlier also fires in 44-87 % of successful episodes), and
from then on the policy already holds 97 % (π0.5) / 82 % (GR00T) of the decisions. Inside the would-be takeover windows
the stacks already call at 99 % (π0.5) / ~92 % (GR00T) of fresh decisions.

| arm | SR | IR | mechanism checks (server extras `r9o6_*`) |
|---|---|---|---|
| `r9o6_pi05_l10_50_stack` (control) | ≈ .88–.92 | ≈ .18 | — |
| `r9o6_pi05_l10_50_stack_X` (+ takeover, escalation kept) | control ± churn (Δ 0, |Δ| ≤ 5 pp) | control + .0004 | fires in ≈ 14 % of episodes (≈ 55 % of them successes finishing inside the window); ≈ 0.2 extra calls per firing |
| `r9o6_pi05_l10_50_npcorr_X` (escalation replaced by takeover) | control − 0 … 4 pp | ≈ control − .005 (−3 %) | escalation calls outside the window (≈ 3 per episode) disappear; the no-progress guard covers most of them (88 % call rate after the end) |
| `r9o6_groot_l10_50_stack` (control) | ≈ .85–.89 | ≈ .19 | — |
| `r9o6_groot_l10_50_stack_X` (+ takeover) | control + 0 … 2 pp (only the ~8 % cache chunks in exhausted failures change) | control + .0015 | fires in ≈ 10 % of episodes; ≈ 1 extra call per firing |

Why no SR gain is expected: once the demo has been played to its end without success, recovery is poor whoever holds
control (still running 6 decisions after the end: pure cache 3-24 %, guard stacks 16-35 %, full escalation
takeover 8-18 % on the same inits), and it falls with the time left (fit only-no-progress + B: first end at <= 30 ->
75-86 % recover, 40-50 -> 25-43 %, > 60 -> 0-15 %). The failure is produced before the end of the script; detecting
the end earlier than the guard already does is impossible without false alarms.
