# R13 dev recording handback

Root: `/home/weiland/trace_runs/os_closed_loop/r13_devrec_50`.

28 R11Knob arms, each explicitly `dev=true, init_pool=B`, library size 50,
100 episodes (10 inits/task). Each cell's seven arms use byte-identical copies
of its `r11_dev_k10_size50` manifest: selection seed 20261004, simulator seed 7,
plugin run seed 0. All record looks without tokens or images; only the four off
arms record shadow policy chunks. Full models, guard-only judge, policy tail,
replan steps 5, and GR00T resize 256 match R11. Frozen R11 fits are referenced
in place, without refitting; each `pred_IR_lib` comes from its artifact.

CPU verification passed for arm/config consistency, all 28 frozen artifact
SHA256 values against R11 calibration receipts, exact paired K=10 selections,
100 episodes/arm, and disjointness from the actual served subsets. Dev contracts
are saved in the root. Both real configuration-only planner paths passed with
remote/launch functions blocked; each generated 28 SHA-bound look contracts and
excluded test-A query metadata. Evidence is in `evidence/verification.json`,
`evidence/plan_local_audit.json`, `evidence/plan_h100_audit.json`, and their logs.
R11Knob selects the ordinary worker with `OSCL_RECORD_LOOKS=1` and
`OSCL_REPLAN_STEPS=5`; no R12 execution-audit worker is required.

Caveat: remote runtime compatibility, GPU capacity, and rollout outcomes were
not checked. Use the existing R11/R12 runtime deployment. H100 plans perform
the standard path-only pickle relocation and retain the original source SHA;
local plans preserve original fit bytes. R11's existing UID-derived RNG behavior
is preserved. No runtime code was edited, and no remote operation or server ran.

Run order (keep every GR00T cell's seven arms on one fleet):

```text
r13_devrec_pi05_l10_50_off_shadow
r13_devrec_pi05_l10_50_periodic_ir25
r13_devrec_pi05_l10_50_periodic_ir32
r13_devrec_pi05_l10_50_periodic_ir40
r13_devrec_pi05_l10_50_distance_ir25
r13_devrec_pi05_l10_50_distance_ir32
r13_devrec_pi05_l10_50_distance_ir40
r13_devrec_pi05_spatial_50_off_shadow
r13_devrec_pi05_spatial_50_periodic_ir25
r13_devrec_pi05_spatial_50_periodic_ir32
r13_devrec_pi05_spatial_50_periodic_ir40
r13_devrec_pi05_spatial_50_distance_ir25
r13_devrec_pi05_spatial_50_distance_ir32
r13_devrec_pi05_spatial_50_distance_ir40
r13_devrec_groot_l10_50_off_shadow
r13_devrec_groot_l10_50_periodic_ir25
r13_devrec_groot_l10_50_periodic_ir32
r13_devrec_groot_l10_50_periodic_ir40
r13_devrec_groot_l10_50_distance_ir25
r13_devrec_groot_l10_50_distance_ir32
r13_devrec_groot_l10_50_distance_ir40
r13_devrec_groot_spatial_50_off_shadow
r13_devrec_groot_spatial_50_periodic_ir25
r13_devrec_groot_spatial_50_periodic_ir32
r13_devrec_groot_spatial_50_periodic_ir40
r13_devrec_groot_spatial_50_distance_ir25
r13_devrec_groot_spatial_50_distance_ir32
r13_devrec_groot_spatial_50_distance_ir40
```

Coordinator commands, provided only. Initialize once, then choose one fleet
block for this root; both blocks address the same arms and sync receipt.

```bash
cd /home/weiland/projects/openpi
P=(taskset -c 22-37,66-81 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 "$PWD/.venv/bin/python" -B)
unset OSCL_MANIFEST OSCL_EPISODES OSCL_TASKS OSCL_INIT_POOL OSCL_POOL_CONTRACT OSCL_POOL_CONTRACT_SHA256 OSCL_RECORD_LOOKS OSCL_EXECUTION_AUDIT
RUN=/home/weiland/trace_runs/os_closed_loop/r13_devrec_50
mapfile -t ARMS < <("${P[@]}" -c 'import json,sys; print("\n".join(r["arm"] for r in json.load(open(sys.argv[1]))))' "$RUN/arms.json")
```

Local RTX 4090 serving, timan107 workers, ports 23100–23103:

```bash
SERVER_HOST=local WORKER_HOST=timan107 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control plan "$RUN" "${ARMS[@]}"
SERVER_HOST=local WORKER_HOST=timan107 SYNC_PORT=23197 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control sync --concurrent "$RUN" "${ARMS[@]}"
SERVER_HOST=local WORKER_HOST=timan107 PORTS=23100,23101,23102,23103 WPS=4 OMP=1 LOCAL_SERVER_CPUS=22-37,66-81 MAX_ATTEMPTS=3 POLL_SECONDS=60 \
  "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control chain "$RUN" "${ARMS[@]}"
SERVER_HOST=local WORKER_HOST=timan107 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control collect "$RUN" "${ARMS[@]}"
```

H100 serving, timan108 workers, ports 23230–23233:

```bash
SERVER_HOST=h100 WORKER_HOST=timan108 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control plan "$RUN" "${ARMS[@]}"
SERVER_HOST=h100 WORKER_HOST=timan108 SYNC_PORT=23198 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control sync --concurrent "$RUN" "${ARMS[@]}"
SERVER_HOST=h100 WORKER_HOST=timan108 PORTS=23230,23231,23232,23233 WPS=4 OMP=1 MAX_ATTEMPTS=3 POLL_SECONDS=60 \
  "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control chain "$RUN" "${ARMS[@]}"
SERVER_HOST=h100 WORKER_HOST=timan108 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control collect "$RUN" "${ARMS[@]}"
```

CPU check reproduction, from the initialized repository shell above:

```bash
"${P[@]}" exp/offline_search/rounds/r13/devrec/build_verify.py verify
"${P[@]}" exp/offline_search/rounds/r13/devrec/build_verify.py plan-local
"${P[@]}" exp/offline_search/rounds/r13/devrec/build_verify.py plan-h100
```

## Separate pi05 spatial root for H100

Root: `/home/weiland/trace_runs/os_closed_loop/r13_devrec_pi05_spatial_50`.
Exactly the seven `r13_devrec_pi05_spatial_50_*` arms listed above, with the same
names, frozen R11 fits, byte-identical paired K=10 B manifest, and selection /
simulator / run seeds 20261004 / 7 / 0. All seven record looks; `off_shadow`
also records shadow policy chunks; tokens and images remain disabled.
100 episodes/arm, 700 total. The running `r13_devrec_50` root was untouched;
its seven pi05 spatial arms are intentionally unused. Use this block for H100
instead of the original root's H100 block, so the fleets have separate root locks.

CPU verification passed, including frozen fit SHA256 checks, paired selections,
served-subset disjointness, configurations, and recording contracts. Both real
local and H100 planner paths passed with remote/launch and fleet-lock entry
points blocked, each producing seven SHA-bound look contracts. Evidence and
logs: `evidence/pi05_spatial_50/{verification.json,build_verify.log,plan_local_audit.json,plan_h100_audit.json,plan_local.log,plan_h100.log}`.
No rollout, sync, remote operation, or server was launched.

H100 serving / timan108 workers (commands provided only):

```bash
cd /home/weiland/projects/openpi
P=(taskset -c 22-37,66-81 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 "$PWD/.venv/bin/python" -B)
unset OSCL_MANIFEST OSCL_EPISODES OSCL_TASKS OSCL_INIT_POOL OSCL_POOL_CONTRACT OSCL_POOL_CONTRACT_SHA256 OSCL_RECORD_LOOKS OSCL_EXECUTION_AUDIT
RUN=/home/weiland/trace_runs/os_closed_loop/r13_devrec_pi05_spatial_50
mapfile -t ARMS < <("${P[@]}" -c 'import json,sys; print("\n".join(r["arm"] for r in json.load(open(sys.argv[1]))))' "$RUN/arms.json")
SERVER_HOST=h100 WORKER_HOST=timan108 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control plan "$RUN" "${ARMS[@]}"
SERVER_HOST=h100 WORKER_HOST=timan108 SYNC_PORT=23198 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control sync --concurrent "$RUN" "${ARMS[@]}"
SERVER_HOST=h100 WORKER_HOST=timan108 PORTS=23230,23231,23232,23233 WPS=4 OMP=1 MAX_ATTEMPTS=3 POLL_SECONDS=60 \
  "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control chain "$RUN" "${ARMS[@]}"
SERVER_HOST=h100 WORKER_HOST=timan108 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control collect "$RUN" "${ARMS[@]}"
```

CPU-only reproduction with the same initialized shell:

```bash
"${P[@]}" exp/offline_search/rounds/r13/devrec/build_verify.py verify --pi05-spatial-only
"${P[@]}" exp/offline_search/rounds/r13/devrec/build_verify.py plan-local --pi05-spatial-only
"${P[@]}" exp/offline_search/rounds/r13/devrec/build_verify.py plan-h100 --pi05-spatial-only
```
