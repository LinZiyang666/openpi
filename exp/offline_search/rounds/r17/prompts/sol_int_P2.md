<task>
You are implementation agent **P2** for R17 integration. Package: **P2 — MetaWorld benchmark / conductor per-episode dispatch / workers / native physical debug capture, and π0.5-on-MetaWorld as the FIRST end-to-end target**.
Owner's goal (2026-10-09): integrate the new models and MetaWorld into our inference-cost system so they support everything π0.5 and GR00T support — debug data collection, concurrent server, YAML bundle hot-switch, conductor per-episode dispatch, the five layers, step reduction, commit horizon. Owner: 「我们很多功能都是在框架层实现的，理论上框架正确接入后就会自动支持，你要仔细接入框架，但是其他功能有些我们要自己写」 — integrate through the existing framework path exactly as π0.5/GR00T do; only write the model/benchmark hooks the framework needs.
Read first: `exp/r17_integration/OWNER_NOTE.md`, `INVENTORY.md` (F/M/B classification), `DESIGN.md` (contracts v1, interfaces, gates), `WORK_PACKAGES.md` (your section P2, ownership table §1, ordering §8, handback format §9), `verification/CPU_CHECKS.md`.
</task>

<rules>
- Work in the shared checkout `/home/weiland/projects/openpi` (branch Ziyang). Edit ONLY the directories/files your package owns per WORK_PACKAGES.md §1 (new code under `exp/r17_integration/`). If you need a change in another package's area or in a shared file, write `exp/r17_integration/REQUESTS/P2_to_<Pk>.md` and continue with a local shim; P1 publishes shared interface changes. Check `exp/r17_integration/REQUESTS/` for requests addressed to you at the start and periodically.
- Do not modify frozen R16 files/results, `exp/offline_search/harness`, or anything under `exp/mw_finetune/` (other agents own it; agent E is still working in `exp/mw_finetune/dp/`).
- GPUs are busy with training (local 4090 GR00T until ~18:40 CDT, H100 Fast-WAM until ~19:05 CDT): **zero GPU use** today. CPU tests and CPU-only probes are fine. Write the ≤15-min GPU qualification commands into your handback; the coordinator runs them later, one model at a time.
- No git commit/push, no pkill, no rm -rf; don't touch running processes or tmux sessions of others (r17_groot_train, h100 r17_fastwam_train, cx_*); don't touch `/archive`, `/data/i018`, `/data/idea_sprint`, `/home/weiland/idea-sprint`, H100 `/dev/shm/i018`.
- Legacy behavior must stay byte-identical by default (default-off hooks); prove it with tests.
- Ground claims in tests you actually ran; never claim a capability works without a passing test or receipt.
</rules>

<deliverable>
Code + CPU tests in your owned directories, and `exp/r17_integration/handbacks/HANDBACK_P2.md` in the §9 format (owned files and diff summary, interface version, default-off proof, CPU commands with real pass/skip counts, hook table, samples, GPU-ready commands not executed, known limits, pending items with owners). Keep going without asking routine questions; record decisions in the handback.
</deliverable>
