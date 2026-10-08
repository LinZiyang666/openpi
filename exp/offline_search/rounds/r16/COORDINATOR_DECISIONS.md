# R16 coordinator decisions

All recorded before any R16 outcome exists. Owner goal (2026-10-07): "融入纯减步并实验" — replace every full-step policy call (MISS) with plain step-reduced inference of the action head.

## D1: what changes
Every policy call made at deployment uses k denoising steps of the action expert (flow matching, Euler from noise), k ∈ {1, 2}, for both π0.5 (default 10) and GR00T N1.5 (default 8). This covers guard-triggered MISS, periodic knob calls and the pure-inference arms. Nothing else changes: library data, PCA/metric fits, guard, corrector heads and knob caps are the existing frozen artifacts, because collection used the full model. The executed horizon stays the current one (a policy call commits its full 10-step chunk; the cache commits 10 steps). Owner: "收集阶段还是完整的模型，你只把miss换成单纯减步k=1或2".

## D2: why old step-reduction data cannot be reused
The 09-15 step ladder replanned every 5 steps. The current system executes 10 steps per call, so both the full-step reference and the effect of fewer steps on the chunk's second half differ. Owner: "之前的数据我们没有改变horizon，改horizon后会变". All R16 numbers are measured at the current horizon; no 09-15 number is used as a reference.

## D3: phase 1 — pure step-reduced inference
{π0.5, GR00T} × {LIBERO-10, LIBERO-Spatial} × k ∈ {1, 2} = 8 arms, official test A, 500 episodes each, measure only. Same identities, seeds and template as R14's test-A pure C10 arms (full policy, 10 steps executed per call), which are reused as the paired full-step reference.

## D4: phase 2 — full system with per-task knob descent
- Library sizes 50 / 100 / 200 / 500 (owner: "可以"), both models, both suites, k ∈ {1, 2}: 32 cells.
- The per-task descent is re-run on B-dev with k-step policy calls (owner: "2。可以"). Protocol = R14 frozen `descent_astra/PROTOCOL.md` verbatim (30 B inits per task, 5 pp primary tolerance, first-failure stop, order ir45 → ir40 → ir32 → ir25 → off); dev inits and size-500 leave-own-episode-out exactly as R15 D2.
- **Reference for the tolerance = full-step pure inference** (10 steps executed per call) on the same dev inits — the existing R14/R15 dev C10 references, reused if they are provably the same identities/template; otherwise rerun once per model × suite.
- The top fallback level of each task (when even ir45 fails) is k-step pure inference, consistent with D1.
- Test A: per cell, the frozen per-task arm × 500 episodes. Compared (paired) with full-step pure inference (R14 test C10) and with the existing full-step per-task descent arms of the same cell (R14 size 50, R15 sizes 100/200/500).

## D5: cost
Owner IR keeps the vision coefficient and prices a k-step policy call by its measured stage costs (owner: "是的把miss换成纯减步的"):
- π0.5: IR = (.152·V + c(k)·M) / N with c(k) = (27.686 + (k/10)·29.572) / 67.519 → c(1) = .454, c(2) = .498 (full .848).
- GR00T: IR = (.148·V + c(k)·M) / N with c(k) = (7.192 + (k/8)·28.104) / 41.442 → c(1) = .258, c(2) = .343 (full .852).
Stage costs: π0.5 `exp.dispatch_surface.analysis.analytic_cost`; GR00T `exp/libero_groot/config/rit/cost_groot_libero_measured.json`. Full-step numbers keep the old coefficients.

## D6: hosts and devices
Owner: workers on timan107; one server host only (h100 or local 4090); return it and tell 沙堆实验 it owns the whole device right after the experiment. Choice: **local 4090 for every root** — GR00T has a 4090-vs-H100 host effect and all R14/R15 GR00T references (test C10, dev C10, per-task descent) ran on the 4090; π0.5 has no host effect. Quota per GPU-sharing rules v4 (openpi 28 GB on the 4090); h100 is left entirely to I018 during R16.

## D7: agents
Implementation by sol (gpt-6.1-sol, xhigh) through the shared Codex app-server daemon (`codex_daemon_run.py`; no agent carries credentials). The coordinator runs every server, chain and monitor.
