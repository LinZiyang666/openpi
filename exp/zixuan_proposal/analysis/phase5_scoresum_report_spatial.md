# TRACER Phase 5 — calibration report (libero_spatial)

- **Verdict**: FAIL ❌
- exit gate: BHR↓ (True) AND SR_calibrated ≥ SR_base − ε (False) AND IR↓/= (True)
- SR_base(I_val)=0.9720  SR_illustrative=0.7760  SR_calibrated=0.7760  (ε=0.02)

## Metrics (paired, same I_val + seed)
- BadHitRate: illustrative 0.4312 (1916/4443) -> calibrated 0.3670 (1509/4112)
- FFR: illustrative 0.4241 (1861/4388) -> calibrated 0.0000 (0/4141)
- IR: illustrative 0.3516 -> calibrated 0.2831 (c_warm=0.75)
- verdicts illustrative=6852 (incomplete 0) / calibrated=6605 (incomplete 0)
