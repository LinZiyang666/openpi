# Stage A drivers (plan `logs/robocasa365_warmstart_plan.log.md` §3.0 / §6)

Run scripts used on the frozen measurement machine (weilandserver) and the
RoboCasa sim client host (timan107). They run from the isolated clone
`/tmp/openpi-stageA` on weilandserver, never from `/home/weiland/openpi`.

| script | host | what |
|---|---|---|
| `run_w2.sh <prompts> <ks> <procs>` | weilandserver | W2 G-M cells: `nsys profile` → `bench_groot_stages.py --mode measure` → `nsys stats` (cuda_api_sum + nvtx_pushpop_sum) → `--mode certify`; serial, resumable, logs to `/tmp/stageA/w2.log` |
| `run_pi05_recal.sh` | weilandserver | pi0.5 same-card recalibration of the ledger's CUDA-Graph tier, 3 repeats |
| `ga1_launch_server.sh <k>` | weilandserver | W3 G-A1 teacher-only server at k denoise steps via `serve_groot_n15_ksweep.py`, port 23160 |
| `ga1_launch_client.sh <k>` | timan107 | W3 G-A1 client: the W8 floor-arm recipe (5 PickPlace tasks × 50, layout/style (1,1), base_seed 1000000, pinned objects, 14 workers), run-prefix `ga1k<k>` |

Outputs: cell JSONs in `exp/robocasa365/data/latency/` of the clone (pulled to
the same path locally; gitignored), traces in `/tmp/w2_traces/`.
