# TRACER Phase 5 — calibration report (libero_10)

- **Verdict**: FAIL ❌
- exit gate: BHR↓ (False) AND SR_calibrated ≥ SR_base − ε (False) AND IR↓/= (True)
- SR_base(I_val)=0.8560  SR_illustrative=0.8240  SR_calibrated=0.5360  (ε=0.02)

## Metrics (paired, same I_val + seed)
- BadHitRate: illustrative 0.4330 (1376/3178) -> calibrated 0.4893 (5530/11303)
- FFR: illustrative 0.8344 (9079/10881) -> calibrated 0.0000 (0/7518)
- IR: illustrative 0.7944 -> calibrated 0.3171 (c_warm=0.75)
- verdicts illustrative=15457 (incomplete 0) / calibrated=19582 (incomplete 0)
