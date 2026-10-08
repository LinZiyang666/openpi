# R12 opus HANDBACK (saved by the coordinator; the harness blocks report files from subagents)

**Task:** R12 look-half explorer (opus). Brief: `rounds/r12/BRIEF.md`.

**Frozen proposals (SPEC.md)**
- **S — batched camera stage 1** (π0.5 serving layer, not a cache change):
  - embed base and wrist in one `embed_image` call; take the masked slot from the cached dummy;
  - assemble the prefix exactly as stock;
  - parity gate: step-0 top-1 agreement ≥ .90;
  - re-price the ledger by measurement.
- **C — wrist-first consistency cascade** (layer 1, plus a wrist-only head in layers 3/5):
  - everything in the deployed R10 fit stays bit-identical;
  - added per task: W metric on [v1, rs]; WL_L metrics for L = 2/4/6/8; a wrist-only LOEO head; a W distance scale;
  - online: step 0 and planned knob calls are full looks; every other look, including the one right after a MISS, is wrist-first;
  - compute W and WL_lag, score = motion RMS of |W − WL| plus gripper-sign disagreement over the first 5 steps;
  - complete if score > τ or lag > 8; otherwise serve W with the own guard, and complete on a MISS;
  - needs a plugin "completion-for-retrieval" path (a `NeedCamera` sentinel and re-query);
  - τ: frozen .15·√(50/N); dev-calibrated τ' = .10·√(50/N), which is the one to use.
- **H0 — wrist-only at every non-call look:** a calibration arm.

**Predictions** (PREDICTION.md, frozen 12:15 CDT, Addendum A 12:28):
- **P0, prices:**
  - batched two-camera stage 1 ≤ 1.25 × wrist-only (the frozen ms range 27–32 refers to the R7 stack);
  - GR00T one-image ≥ .8 × two-image — confirmed by the coordinator.
- **P1, S:** ΔIR −.037 to −.043 per slot; ΔSR 0; step-0 agreement ≥ .9.
- **P2, C with τ':**
  - L10-50 and Sp-50: one-camera share .40, ΔIR owner −.029, fair −.010, ΔSR about −0.15 to −0.2 pp;
  - dense cells: share .10–.25, owner −.021 to −.025, fair −.003 to −.006;
  - pooled ΔSR over 5 cells −0.2 pp, 80% interval [−1.2, +0.8].
- **P3, H0:** −0.7 / −0.8 pp; owner −.032 / −.041; fair −.013 / −.022.
- **P4, GR00T:** none (no price to save).

**Experiment plan** (SPEC §4):
- **Phase 0** (prices, local GPU idle) applies the frozen rule: adopt S if batched ≤ 1.25 × wrist.
- **Phase 1** dev pilot, 50 episodes per arm, implementation and cost checks only; nothing is refit.
- **Phase 2** test A, 500 episodes per arm, same-batch baselines:
  - S: π0.5 L10-50, Sp-50, L10-200, L10-500 and Sp-500 knob off, plus L10-50 and Sp-50 at knob P@.32 (14 arms);
  - or, if C is built: the same 5 cells baseline vs C, H0 on two cells, and knob-on {baseline P@.32, C P@.32, C retargeted to the baseline's IR}.
  - Statistics: paired McNemar, pooled stratified bootstrap, noninferiority bound −1 pp.

**Data discipline and incidents**
- Fitting and selection used only B-pool libraries and B-pool dev recordings. τ' was calibrated on dev data after the freeze and is labelled as such.
- The R6 B-val recordings were used for agreement statistics only.
- No `offline_search_store/queries`, no test-A root, no astra directory.
- The `r12_devlooks_*` directory names were visible before the freeze. `COORDINATOR.md`, `prep/HANDBACK.md` and the `50L`/`50S` recordings were read only after the freeze. The 200, 400 and knob roots were never opened.
- **Incident:** `rounds/r12/__init__.py` and its `__pycache__` were accidentally created in the shared r12 directory and deleted immediately with plain `rm` / `rmdir`.
- No git, no launches, no `pkill`, no `rm -rf`.
- GPU use: the tower/low-res/parity runs used under 4 GB VRAM, all on the idle 4090 (before 12:16). After the GR00T recording servers started, the GPU was not used for any measurement.
- **Open item:** Request 0, the π0.5 batched stage-1 price, decides everything downstream.
- `out/replay*/` (about 1.2 GB) and `cache/` (about 0.6 GB) are disposable and should not be committed.
