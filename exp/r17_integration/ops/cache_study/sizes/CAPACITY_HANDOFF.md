# Size lanes re-armed, 2026-10-10

Old waiting PIDs834339/834494 stopped with fenced SIGINT, zero children/dispatch, closed receipts and released locks. Existing running DP/H100 lanes were not signalled.

4090: PID1130348, /data/mw_finetune/eval/cszmdp, local32, DP4 replicas. The requested local60 cannot pass the existing1024MiB renderer reservations on49140MiB VRAM plus8192MiB server growth and10% GPU headroom. Reserve was retained; no extra timan107 workers were added. Four replicas retain the same cached actions/native inputs. The70% per-replica CPU target needs live measurement after PASS.

H100: PID1130349, /data/mw_finetune/eval/csznh. Fast-WAM85 then Cosmos85, each timan10745+timan140, local0,2 replicas. T1 GPU3 UUID/PCI/minor guard remains pinned,6144MiB RAM/1024MiB renderer per worker; other workers4096MiB RAM. T1 planned-recycle fix passed4-generation/zero-crash-budget regression and Q actual bwrap proof; deployed payload pins audited. Prestaged45-worker timan107-only fallbacks are selected if fix/namespace verification fails, and T1 is reconsidered at the Cosmos boundary.

Both arms are WAITING_PREDECESSOR: csl10dp / csl10h respectively. Exact predecessor PID generations must exit, lane ownership release, and every predecessor result seal PASS/all-owned-closed. CPU disk>=8GiB/RAM/renderer admission then waits before any resident/fleet launch, preserving any older local H100 reservation. Existing P6 verified-local-seal cleanup remains active.

Order: DP50,100,200,300,400. H100 Fast-WAM50..400, then Cosmos50..400. Official-A task UIDs/plans and native runtime/assets unchanged; canonical batching topology retained with actual execution counts recorded separately. All original csze preparations complete, all5 new preparations CPU-validated, worker rosters record nodes. The frozen drivers plan the exact original160000 cells across full/fallback variants,100000 dispatched scientific cells. Library audit remains PASS240 YAMLs/960 queries; all per-size B fits/calibrations remain sealed.

Tests:124 broad passed;38 focused passed plus2 subtests (overlap). Final worker-helper refresh and real-library/driver-plan audits PASS.

Planning durations after predecessor PASS and resource admission: DP12–24h; Fast-WAM8–18h + Cosmos4–10h = H10012–28h; T1-free fallback18–36h. Not measured85-worker throughput; timan10745 approaches its48-core capacity; model MISS, initialization and sealing can dominate.

Launch scripts (already nohup-armed; do not launch duplicates): run_4090.sh and run_h100.sh here. Machine records: ARMED.json, CAPACITY_FLEETS.json.
