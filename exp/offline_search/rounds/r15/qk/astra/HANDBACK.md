# Round-2 handback — pilot candidates frozen

Read [PILOT.md](PILOT.md) for exact artifacts, loader configs, eight arms and baseline reuse. [PROTOTYPE2.md](PROTOTYPE2.md) gives all hold-out comparisons, image audits and the unchanged numerical gate report. Recommendation: **continue with the owner-authorized paired B-dev pilot**; no online run or GPU use was launched by astra.

- **pi05_l10_50: state_drop_aug, 40 epochs.** B-dev balanced error is +9.83% versus the deployed key. Shuffling images raises error 169.0%; black images raise it 117.5%. Controlled CPU key median 1.028 ms. Offline B-dev gate: FAIL; image audit: PASS.
- **pi05_l10_200: state_drop_aug, 120 epochs.** B-dev balanced error is +8.02% versus the deployed key. Shuffling images raises error 217.5%; black images raise it 131.9%. Controlled CPU key median 0.958 ms. Offline B-dev gate: FAIL; image audit: PASS.

Selection used two within-library episode hold-outs over five recipes and 40/80/120 epochs, before selected-candidate B-dev evaluation. Every comparison uses the same symmetric AWM/early metric and 16-NN kernel. State-only remains rejected. The original student also passes the image audit, so its proximity to a state-only clean score was not evidence that it ignored images.

GATES.md is unchanged (SHA256 `188d3cc1d3beb807174c3bff84696542be72f00bcb66ac0cdc66cbc913a2d937`). Its old offline-veto/100-episode-first authorization is superseded by owner directive 4 and the coordinator's 300-episode plan. Numerical failures remain visible in [gate_report.json](gate_report.json). Sol's full runtime verification and H100 timing remain separate from these verified CPU encoder artifacts.

Use the R15 list outside size 200 for both cells, current/new × off/ir25 × 300 episodes. Size-200 current baselines can reuse the corresponding R15 descent arms after completion/equivalence checks; no R15 size-50 baseline exists. Fixed ir25 caps are 2.6125 (50) and 2.0 (200). No other rollout or A work is requested.

Stable model format, Tensor-output loader setting (`output_index: null`), selected epoch counts, manifest hashes and host CPU-binding notes are in PILOT.md. [round2/verification.json](round2/verification.json) records the actual sol-loader checks and artifact/data boundaries. [ROUND2_MANIFEST.json](ROUND2_MANIFEST.json) freezes this handoff. Round 1 remains preserved in `round2/round1_snapshot/` and the original final manifest.

Open issue: the remaining unseen-state error may still reduce closed-loop SR or cause extra policy calls. Image sensitivity and faster key extraction do not settle that question; the bounded matched pilot is the next discriminating test.
