#!/usr/bin/env bash
# Closed-loop evidence for a G3 wrapper: the plugin's CPU selftest (real per-connection stack, recorded keys) with
# --os-log-inputs, then verify_logs = offline harness replay of the LOGGED online inputs (the method's own served
# chunks as history) -> every decision's topk / scores / confidence / synthesized action / extras (incl. the V6 state
# machine: stuck_n, esc_n, level, anchor, ...) must be bit-identical online and offline.
# Optional 6th arg "artifact": prefit once with single-threaded BLAS (--os-fit-artifact) and serve from the pickle.
#
# usage: taskset -c <cpus> bash tools/cl_evidence.sh <tag> <cell> <yaml> <method spec> '<kwargs json>' [artifact] [episodes]
set -u
cd /home/weiland/projects/openpi
TAG=$1; CELL=$2; YAML=$3; M=$4; KW=$5; ART=${6:-}; NEP=${7:-20}
D=/home/weiland/trace_runs/offline_search_store/derived/r02/g3_recovery/cl
mkdir -p $D/fits
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
EXTRA=()
if [ "$ART" = artifact ]; then
  P=$D/fits/$TAG.pkl
  [ -e $P ] && rm $P
  .venv/bin/python -m exp.offline_search.closed_loop.plugin --os-method "$M" --os-kwargs "$KW" --os-cell $CELL \
      --os-log-dir $D/fits --os-fit-artifact $P > $D/prefit_$TAG.log 2>&1 || { echo "$TAG PREFIT FAILED"; exit 1; }
  EXTRA=(--fit-artifact $P)
fi
.venv/bin/python -m exp.offline_search.closed_loop.selftest --cell $CELL --yaml "$YAML" --method "$M" --kwargs "$KW" \
    --episodes $NEP --out $D/selftest_$TAG "${EXTRA[@]}" > $D/selftest_$TAG.log 2>&1
rc1=$?
.venv/bin/python -m exp.offline_search.closed_loop.verify_logs --log-dir $D/selftest_$TAG --work $D/verify_work \
    --out $D/verify_$TAG.json > $D/verify_$TAG.log 2>&1
rc2=$?
.venv/bin/python - "$D/selftest_$TAG" "$D/verify_$TAG.json" "$TAG" "$rc1" "$rc2" <<'EOF'
import glob, json, sys
import numpy as np
d, vj, tag, rc1, rc2 = sys.argv[1:]
st = json.load(open(f"{d}/selftest_report.json"))
v = json.load(open(vj))
cov = {}
COLS = {"level1": ("x_level", 1), "level2": ("x_level", 2), "level3": ("x_level", 3), "blend": ("x_blend", 1),
        "terminal": ("x_terminal", 1), "anchor_exec": ("x_anchor_src", 1)}
for f in glob.glob(f"{d}/inputs/*.npz"):
    z = np.load(f)
    for k, (col, val) in COLS.items():
        if col in z.files:
            cov[k] = cov.get(k, 0) + int((z[col] == val).sum())
print(json.dumps({"tag": tag, "selftest_rc": int(rc1), "verify_rc": int(rc2), "decisions": st["decisions"],
                  "exec_ok": st["exec_ok"], "selftest_offline_equal(first decisions)": st["offline_equal"],
                  "verify_decisions": v["decisions"], "verify_offline_equal(all decisions)": v["offline_equal"],
                  "executed_equals_selected": v["executed_equals_selected"], "q_us_p50": v["q_us"]["p50"],
                  "state_machine_coverage": cov}))
EOF
