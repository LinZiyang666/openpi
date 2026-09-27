K1 method contract (2026-09-27):
- Method spec exp.offline_search.rounds.r04.k1_blind.blind_awm:BlindAWM, kwargs lib=current,kref=5 (50) or lib=big,kref=8 (500), serving=phase_particles|kernel_clock|top1_clock|anchor_tail, budget=0..4, gates=all|budget_only, residual_threshold=.25|.5|1.
- Mixed spec exp.offline_search.rounds.r04.k1_blind.judge:BlindMixedJudge; base_kwargs above; guards=true, events=none, progress_guard=noprog_span (dense motion guard) or noprog_n (stock MixedJudge passthrough); memo_reset_after_miss=false|true.
- Method remembers anchors both in query and os_synth (MixedJudge path); reset clears, blind_step detects prev_hit=False and invalidates. Optional invalidate_anchor() supported for immediate plugin MISS feedback.
- Need --os-blind to activate K2 bypass. last_blind_extras exposes rejected look-gate diagnostics. Result.extras contains phase, gates, normalized state and expected displacement; BlindResult carries every served row and fixed weight.
- No-progress observed vision transition forces look code 8 (name noprog_span) until progress advances. Codes 1..6 per common contract.
- Optional fourth-batch composition exp.offline_search.rounds.r04.k1_blind.wrist:BlindWristAWM / BlindWristMixedJudge uses K3's wrist metric; requires --os-stage1-mode wrist_only --os-no-shadow-native --os-tokens off. K3's camera module must be installed. The gap wrapper recalibrates its vision features through WristView.
- ControlStepLibrary is pure cache, intentionally rejects the G3 wrapper path (static guard tables do not describe splices).
