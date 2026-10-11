# Owner note (2026-10-09 10:5x) — read before writing DESIGN.md

「我们很多功能都是在框架层实现的，理论上框架正确接入后就会自动支持，你要仔细接入框架，但是其他功能有些我们要自己写」

Implications for the design:
1. Identify precisely which features live in the **framework layer** (the shared policy/serving/conductor framework that π0.5 and GR00T plug into) and therefore come for free once a model is integrated through the framework's model/policy interface correctly — and which features are **model-specific** or **benchmark-specific** code that we must write ourselves for each new model / MetaWorld.
2. INVENTORY.md must mark every feature as `framework (automatic once integrated)` / `per-model hook (must implement)` / `per-benchmark (must implement)`, with the exact framework entry point and the hook a new model must provide.
3. The primary deliverable of DESIGN.md is a careful, faithful framework integration for each new model (same way π0.5 and GR00T are integrated — follow their integration path exactly, file by file), plus the short list of extra pieces to write.
