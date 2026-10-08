# R14 offline design status

Offline deliverables completed and frozen 2026-10-04T17:35:06.095796+00:00: `OFFLINE.md`, `SPEC.md`, `PREDICTION.md`, early `REQUESTS.md`, executable CPU studies, exact pilot identities/grids and the economical test grid. No new variant was run closed loop. No test-A data or reserved R13 validation identity file was opened. No GPU, remote action or rollout was used.

Final pilot: 68 arms / 3400 episode runs. Final test plan: 128 arms / 64,000 episode runs. The early pilot request was expanded to isolate the additional continuity scale discovered in the code audit; the current CSV is authoritative.

Important limitations: production full-row pooled fits and real-plugin parity remain for sol. Shadow policy labels currently support only the two small long-horizon cells. All final B budget refits use existing development data; earlier sparse-split failures remain visible. Size-full-library lacks a B complement. Runtime is a B-timing scenario, not a measured H100 benchmark.

Open questions for the coordinator: when the R13 pilot releases its reservation; when already requested spatial/middle-size shadows become available; whether the planned host/client capacity is available; whether the rollout seeds affect actual policy/simulator randomness. No answer is needed to review or implement this frozen design. Do not refit silently after these answers or results arrive.
