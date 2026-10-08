# R12 opus HANDBACK — revised per DIRECTIVE_2 (saved by the coordinator; first version in HANDBACK_v1.md)

**Task:** R12 look-half explorer (opus), revised per DIRECTIVE_2 and the owner's "method, not code-level efficiency" ruling.

**Frozen method:** M1, the guarded one-camera look (SPEC §6).
- Library-fitted parts: W metric; lag-matched wrist-plus-remembered-agentview metrics (L = 2/4/6/8); wrist-only LOEO corrector head; W distance scale.
- Dev-calibrated constants: τ = .10 and m = 1.0.
- Full look at step 0, at planned knob calls, and on the look right after any call.
- Otherwise wrist first. Complete if min(top-1 progress advance of wrist and wrist+memory) ≤ .5+m, or consistency > τ, or the remembered agentview is more than 8 decisions old.
- Otherwise serve the wrist-plus-memory chunk with the wrist head correction.
- On GR00T the wrist is encoded alone and used as is.
- Plugin needs:
  - a wrist-only stage 1 for π0.5 and GR00T;
  - completion-for-retrieval;
  - the planned full look declared before stage 1.

**Withdrawn:**
- S (batched camera stage 1): owner ruling.
- C and H0: superseded, because closed-loop dev showed the guard problem.

**Predictions:** PREDICTION.md Section B, 13:11 CDT.
- Knob off, one-camera share: pi05 L10-50 .23, Sp-50 .30, Sp-200 .45, groot L10-50 .12.
- ΔIR −.007 to −.018; ΔSR −0.15 to −0.55 pp.
- Matched-SR ΔIR: pi05 L10-50 −.003, Sp-50 −.008, Sp-200 −.004; dense L10 and GR00T worse. Knob on: 0.
- Pooled over the gain candidates about −.002, 80% interval [−.012, +.015]: no distinguishable saving expected.

**Experiment plan:** SPEC §7.
- Phase A: own-loop dev pilot, 4 cells × 2 arms × 100 episodes, look recording on. Stop rule: first guard call ≥ 2 decisions later, or call rate −20%.
- Phase B: test A, knob off, pi05 L10-50 / Sp-50 / Sp-200, groot Sp-50, plus groot L10-50 as control, baseline vs M1 (10 arms).
- No knob-on arms: M1 has no one-camera looks there.

**Data discipline and incidents:**
- Dev recordings (50L, 50S, 200, 400, knob) were used as B-pool dev evidence. The constants calibrated on them are labelled.
- No test A, no astra directory, no git, no launches, no `pkill`, no `rm -rf`.
- GPU: one GR00T mismatch run under 7 GB (others were using about 30 GB at the time); earlier tower runs under 4 GB on the idle GPU.
- Earlier incident: a stray `rounds/r12/__init__.py` and its pycache, which was created outside the explorer directory and deleted immediately.
- `out/replay*/` (about 1.2 GB) and `cache/` (about 0.6 GB) are disposable and should not be committed.
