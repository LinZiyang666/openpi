# Round-4 predictions (written 2026-10-02 02:49:59 CDT, before any r09_opus_r4 arm exists or runs)

Stack: half-strength corrector (fable round 2, `CorrectedCache`, per-task heads fitted on inits 0–19, blend .5, the
exact fitted artifacts of `r9f2_{pi05,groot}_l10_50_corr05pt`) as the cache, plus my bounded escalation window
(lag ≥ 12 at a fresh decision ≤ 80 → policy at every fresh decision for 24 decisions → corrector cache again; constants
unchanged since round 2). 100 pairs, tasks 0–9 × inits 20–29.

Basis (`tools/predict.py`, `out/predictions.json`): exact-prefix simulation on fable's corrector runs (inits 20–29,
timan107): π0.5 corrector .81 @ .076, 26% of episodes cross the rule, the corrector recovers .31 of them by itself;
GR00T corrector .77 @ .074, 34% cross, self-recovery .32. Takeover success r bracketed by my round-2 screen's
window on triggered episodes (π0.5 .46, GR00T .40) ± .10.

| arm | π0.5 long-50 | GR00T long-50 |
|---|---|---|
| `r9o4_*_cache` (same-run control) | .70–.76 @ .076 | .55–.62 @ .074 |
| `r9o4_*_corr` (corrector alone) | .76–.83 @ .076 | .70–.78 @ .074 |
| `r9o4_*_corr_esc_w24` (stack) | **.82–.88 @ .11–.13** (point .85) | **.76–.83 @ .12–.14** (point .80) |

Paired stack − corrector: π0.5 +1 to +7 pp (point +4), GR00T −1 to +6 pp (point +3); the gain is confined to episodes
that fall behind pace (expected 20–35% of pairs). Paired stack − cache: π0.5 +8 to +15 pp, GR00T +15 to +25 pp.
Falsifiable: (a) stack ≥ corrector on both cells; (b) the stack never changes an episode that does not cross the rule
(both arms identical there apart from run-to-run noise); (c) stack IR ≤ .14 on both cells; (d) no stack reaches pure
policy (π0.5 ≈ .92, GR00T ≈ .87) on 100 pairs.
