# Post-freeze implementation clarification

After the prospective grid/prediction freeze, the reference controller gained explicit static off/all-call endpoint handling for scores outside the fitted library range. The self-check now covers this. None of the proposed static arms uses either endpoint; all frozen doses, thresholds, forecasts and analysis payloads are unchanged. No closed-loop output was read. The original code hashes remain recorded in `freeze.json`; the handback manifest records the final reference-code hashes.

The knob-off baseline should bypass the layer-4 adapter entirely. The active nonfinite-score fallback still requests inference and charges the actual ledger. A disabled knob adds no calls.
