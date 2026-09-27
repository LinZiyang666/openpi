# R4 K1 methods

`blind_awm:BlindAWM` subclasses the immutable R2 AWM; `BlindAWM3` composes the same blind adapter with R3 AWM3. Use the dotted prefix `exp.offline_search.rounds.r04.k1_blind.`. The ordinary query and G3 synthesis paths capture all 16 anchor members without changing retrieval. The plugin must opt into `--os-blind` to invoke `blind_step` before stage 1.

| Setting | Values |
|---|---|
| `lib`, `kref` | `current`, 5 for ~50 episodes; `big`, 8 for 500 |
| `serving` | `phase_particles`, `kernel_clock`, `top1_clock`, `anchor_tail` |
| `budget` | 0 (adapter control), 1, 2, 3, 4 |
| `gates` | `all` (default), `budget_only` (tail control / diagnostic replay) |
| `residual_threshold` | .5 default; .25 and 1 sensitivity variants |

Phase uses fixed anchor weights, absolute normalized state cost, .05 offset penalty and monotone advancement of at most two library rows. Events and terminal gates use nominal clock rows, including for phase serving. Tail eligibility uses the original real chunk horizon, never its wire padding. Blind results zero padded action dimensions. `last_blind_extras` retains diagnostics when a look is requested. An optional `invalidate_anchor()` permits immediate MISS feedback; otherwise the next `prev_hit=False` invalidates it before blind service.

`judge:BlindMixedJudge` takes `base_kwargs` above, `progress_guard="noprog_span"` (default) or `"noprog_n"`, and `memo_reset_after_miss` (default false). Other kwargs pass to MixedJudge, including `noprog_n`, `guards`, `events`, and `burst`. Defaults use guards and no early events. The new span guard adds the elapsed decision gap after nonadvancing real vision anchors; after its first nonadvancing transition, code 8 (`noprog_span`) requests consecutive vision until advancement. It measures motion from dense normalized proprioception. `noprog_n` is the exact legacy guard passthrough; it requests vision on every decision because its unchanged guard needs adjacent visual observations (look code 8, `noprog_n_requires_vision`). `MemoResetMixedJudge` defaults to `noprog_n` plus progress-only memo reset after MISS. Stuck, terminal and overtime guards are retained. GR00T terminal-closed sign is corrected only in the explicitly new span variant; the stock passthrough retains stock behavior.

`control_step:ControlStepLibrary(ablation="G"|"GS")` is pure-cache only. It retains real AWM codes and one fractional offset per parent, with the ordinary step-zero/fresh selection and action branches. G applies virtual-distance ranking to unshifted chunks; GS splices actually executed five-control heads across valid episode edges. Offset zero keeps the original full chunk. Nonzero offsets hold the last executed library control if the recorded sequence ends. `offsets=[0,2,4]` is the optional coarse ablation. Confidence scales are refitted by library LOEO, including the dispersion of aligned heads. Static G3 guard tables cannot represent splices, so this method explicitly refuses mixed wrapping.

`wrist:BlindWristAWM` and `BlindWristMixedJudge` compose K3's wrist-only metric with these adapters. They depend on K3's installed module and server flags. Wrist query parity is against WristAWM, because camera deletion intentionally changes the metric.

`arms_r4.json` contains batch-three controls and batch-four combination candidates, ordered π0.5 l10, spatial, then GR00T pure cache. `arm_plan.json` identifies batches and conditional candidates; it does not rank or declare a winner. Replace `<RUN>` before passing the spec to K4's `emit_arms`. `prefit.sh` writes exact per-arm plugin artifacts after setting `RUN`; it uses the assigned K1 CPU range and no GPU. Large verification pickles live in `/tmp/k1_blind_fits`, outside the repository.

All verification commands and measured results are in `HANDBACK.md` and `results/`. No closed-loop success-rate result is claimed by these CPU tests.
