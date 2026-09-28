Follow-up task (same rules, CPU range, store discipline and hand-back style as before). The owner adds: "offline library
growth can also be fitted — we have all 500 demo episodes in h5; build the library up gradually and enlarge it".
Extend your work with a demo-data scaling curve, in rounds/r05/q4_growth/ (append a section to HANDBACK.md):
1. Nested demo libraries for π0.5 libero_10 and libero_spatial: subsets of the 500-episode demonstration library
   (`bpool_cs`; state whether the deployed 50-episode `current` episodes are contained in it) with 100, 200 and 300
   episodes (10 / 20 / 30 per task, nested, deterministic and documented). Publish each as a new store library (e.g.
   `demo100`) in both store copies, validated by the existing loaders; distinct from your grow250 directories.
2. Two fit variants per size: `refit` (PCA / whitening / calibration on that library) and `frozen50` (deployed 50-library
   representation, larger library's rows projected through it — incremental growth without refit).
3. Offline curve vs size (50 current, 100, 200, 300, 500): density and held-out action / successor losses, both variants,
   both suites, on recorded cache and inference queries.
4. Arm specs (append to arms_q4.json or a new arms_q4_demo.json): pure-cache anchor_tail (K1 BlindAWM, serving anchor_tail,
   budget 1, gates budget_only, `--os-blind`, STAGE1 only, no judge) with `refit` at 100 / 200 / 300 × {l10, spatial}
   (6 arms; state the kref rule) and `frozen50` at 200 × {l10, spatial} (2 arms); full 500-init runs (no manifest: demo
   libraries contain no evaluation-init data). Run the prefits into /home/weiland/trace_runs/os_closed_loop/
   r05_demo_curve/fits/ (sizes, sha256). Print a one-paragraph summary at the end.
