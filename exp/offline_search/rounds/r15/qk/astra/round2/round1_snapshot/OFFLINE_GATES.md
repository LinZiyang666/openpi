# Offline gate application

**NO-GO for online work.** The primary CPU-trained candidate fails unseen B-dev accuracy. The prototype has not refitted or verified a complete five-layer plugin, and H100 deployment-host latency is unmeasured. No approval is implied by this report.

| Cell | LOEO balanced ratio | Worst LOEO task ratio | B-dev balanced ratio | Worst B-dev task ratio | Top-16 pass | CPU cost pass locally |
|---|---:|---:|---:|---:|---|---|
| pi05_l10_50 | 0.9950 | 1.0006 | 1.1250 | 1.1730 | False | True |
| pi05_l10_200 | 1.0168 | 1.0355 | 1.1000 | 1.2304 | False | True |

Exact booleans and ratios: `gate_report.json`; definitions: frozen `GATES.md`. Library LOEO conditions on the final trained representation. A library pass cannot substitute for unseen B-dev. Spatial lacks all-look labels; GR00T lacks exact library RGB, so neither is eligible.
