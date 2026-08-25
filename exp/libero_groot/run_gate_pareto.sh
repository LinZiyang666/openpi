#!/usr/bin/env bash
# Drive the GR00T gate-threshold Pareto end to end, unattended.
#
#   template -> warmup pool -> warmup run -> solve -> emit sweep + gate-only
#   -> SMOKE (2 arms, small sample, T7) -> sweep -> gate-only -> aggregate
#   -> plot + manifest
#
# Runs on weilandserver in tmux so it outlives the operator's session. Every
# stage is resumable by artifact: the scheduler skips cells whose complete
# results already exist, and each emit step is idempotent, so a re-entered
# stage costs only the cells that were in flight when it died.
#
# Failure propagation is the property this script exists to guarantee. A phase
# is not "done" because its tmux session ended -- the scheduler deliberately
# swallows per-cell failures so one bad cell cannot strand the other five. Each
# phase therefore ends with an explicit artifact check, the scheduler's exit
# status is carried out of tmux through a status file, and `set -e` turns any
# of that into a stop. GATE-PARETO-DONE is printed only if every phase passed.
#
# The warmup is the one phase that does NOT run on the A pool: its shards come
# from emit_warmup_pool.py, which selects B-pool inits whose trajectories are
# absent from the library. Thresholds fitted on the evaluation set would be a
# test-set peek, and thresholds fitted on library episodes would be biased by
# self-retrieval.
set -euo pipefail
export HOME=${HOME:-/home/weiland}
# Resolved from this script's own location, never restated: the control-plane
# box and the serving box hold the checkout at different absolute paths, so a
# hardcoded root is wrong on one of them by construction. Running this file is
# therefore the statement of which checkout the chain drives.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=$REPO/.venv/bin/python
export PYTHONPATH=$REPO:$REPO/src

ROOT=/data/libero_cache/gate_pareto
PORTS=23160,23161,23162,23163,23164,23165
LOG=/tmp/libgp.log
RUNDIR=/tmp/libgp_run
SUITES=${SUITES:-"libero_spatial libero_10"}
SMOKE_EP=${SMOKE_EP:-2}          # episodes per task -> 20 per arm over 10 tasks

mkdir -p "$RUNDIR"
say() { echo "[$(date '+%m-%d %H:%M:%S')] GP $*"; }
die() { say "FAILED: $*"; exit 1; }

# Fail here, naming the host, rather than at the first phase: an unattended run
# that dies on a missing interpreter three hours in has already held six GPU
# slots for nothing.
say "repo=$REPO host=$(hostname)"
[ -x "$PY" ] || die "no openpi venv interpreter at $PY (repo resolved from \$BASH_SOURCE)"
[ -f "$REPO/exp/libero_groot/orchestrate_search.py" ] || die "scheduler missing under $REPO"
[ -f "$REPO/exp/libero_groot/analysis/gate_pareto/analyze_gate_pareto.py" ] || die "analyzer missing under $REPO"
command -v tether >/dev/null || die "tether CLI not on PATH; this chain drives timan107"
# An operator (or a test) can ask for the resolution and the host check alone.
# Worth having as a real mode rather than a test-only hook: "does this box even
# hold what the chain needs" is the question to answer before an overnight run,
# not after it.
if [ -n "${GP_PREFLIGHT_ONLY:-}" ]; then
  say "preflight OK"
  exit 0
fi

# spatial workers hold ~2.6 GB each, libero_10's ~3.1 GB (longer horizon,
# larger scenes). 12/slot took timan107 to 4 GB free on l10, one balloon away
# from an OOM cascade that silently drops whole shards.
workers_for() { [ "$1" = "libero_10" ] && echo 8 || echo 12; }
# A teacher-heavy arm (low f_FH) runs at roughly half the throughput of the
# pure-cache cells the search measured, and l10 episodes are 2.4x longer.
timeout_for() { [ "$1" = "libero_10" ] && echo 14400 || echo 7200; }
tag_for() { [ "$1" = "libero_10" ] && echo l10 || echo sp; }

arm_list() {  # $@ = yaml dirs -> comma-separated arm names
  local arms=()
  for d in "$@"; do
    for f in "$d"/*.yaml; do
      [ -e "$f" ] || continue
      arms+=("$(basename "$f" .yaml)")
    done
  done
  [ ${#arms[@]} -gt 0 ] || die "no arm recipes in $*"
  (IFS=,; echo "${arms[*]}")
}

# ---------------------------------------------------------------------------
# One scheduler phase, with the exit status carried out of tmux.
# ---------------------------------------------------------------------------
run_stage() {  # $1=label $2=suite $3=yaml dir $4=results $5=per-step $6=expect_ep $7=trials $8=extra
  local label=$1 suite=$2 yamls=$3 results=$4 per_step=$5 expect=$6 trials=$7 extra=$8
  local ckpt status script
  ckpt=$($PY -c "
from exp.libero_groot import gate_pareto_bindings as g
print(g.for_suite('$suite').checkpoint)")
  status=$RUNDIR/${label}.status
  script=$RUNDIR/${label}.sh
  rm -f "$status"
  mkdir -p "$results" "$per_step"

  # A script file rather than an inline tmux command: `pipefail` has to be set
  # inside the shell that runs `python | tee`, or tee's exit status hides the
  # scheduler's, and the status file has to be written by that same shell.
  cat > "$script" <<EOS
set -o pipefail
export HOME=/home/weiland
export PYTHONPATH=$REPO:$REPO/src
cd $REPO
$PY exp/libero_groot/orchestrate_search.py \\
  --yaml-dir $yamls --results-dir $results --suite $suite --checkpoint $ckpt \\
  --ports $PORTS --workers $(workers_for "$suite") --expect $expect \\
  --trials $trials --cell-timeout $(timeout_for "$suite") \\
  --per-step-dir $per_step $extra 2>&1 | tee -a $LOG
echo \$? > $status
EOS

  # '=' forces an exact match. Without it tmux falls back to PREFIX matching,
  # so `-t libgp` with no session of that exact name matches this chain's own
  # session (libgpchain) and kills it -- silently, exit code 0, no output. The
  # chain then vanishes right after announcing a phase, with an empty log and
  # no scheduler, which reads as a failure inside run_stage rather than as the
  # kill it actually is.
  tmux kill-session -t '=libgp' 2>/dev/null || true
  sleep 2
  tmux new -s libgp -d "bash $script"
  while tmux has-session -t '=libgp' 2>/dev/null; do sleep 120; done

  [ -f "$status" ] || die "$label: scheduler left no status file (tmux died?)"
  local rc; rc=$(cat "$status")
  [ "$rc" = "0" ] || die "$label: scheduler exited $rc (see $LOG for PHASE-FAILED)"
  say "$label: scheduler exited 0"
}

# ---------------------------------------------------------------------------
# Artifact-level phase gate. The scheduler's own check is per-cell; this one is
# per-phase, and it is what decides whether the next phase may start.
# ---------------------------------------------------------------------------
verify_phase() {  # $1=label $2=results $3=per-step $4=expect_ep $5=arm csv
  local label=$1 results=$2 per_step=$3 expect=$4 arms=$5
  $PY exp/libero_groot/analysis/gate_pareto/analyze_gate_pareto.py aggregate \
    "$results" "$per_step" "$RUNDIR/${label}.summary.json" \
    --expect-ep "$expect" --expect-arms "$arms" \
    || die "$label: integrity gate rejected the phase"
  say "$label: integrity gate passed ($(echo "$arms" | tr ',' '\n' | wc -l) arms)"
}

for suite in $SUITES; do
  tag=$(tag_for "$suite")
  cfg=$REPO/exp/libero_groot/config/gate_pareto/$suite
  data=$ROOT/$suite
  workers=$(workers_for "$suite")

  say "=== $suite: template ==="
  $PY -m exp.libero_groot.emit_gate_yamls --mode template --suite "$suite"

  say "=== $suite: warmup pool + arm ==="
  $PY -m exp.libero_groot.emit_warmup_pool --suite "$suite" \
    --lanes "$workers" --out-dir "/tmp/libgp_shards_$tag"
  tether exec timan107 -- bash -lc \
    "export HOME=/home/zixuans8; rm -rf /tmp/libgp_shards_$tag; mkdir -p /tmp/libgp_shards_$tag"
  for f in /tmp/libgp_shards_$tag/*.json; do
    tether push "$f" "timan107:/tmp/libgp_shards_$tag/$(basename "$f")" --force >/dev/null
  done
  $PY -m exp.libero_groot.emit_gate_yamls --mode warmup --suite "$suite"

  say "=== $suite: warmup run (B pool, 100 ep) ==="
  run_stage "warmup_$tag" "$suite" "$cfg/warmup" "$data/warmup_results" \
    "$data/warmup_per_step" 100 50 \
    "--phase warmup --init-subdir $suite --shards-dir /tmp/libgp_shards_$tag --shard-prefix gpw_$tag --skip-shard-prep"
  verify_phase "warmup_$tag" "$data/warmup_results" "$data/warmup_per_step" 100 "gpw_$tag"

  say "=== $suite: solve thresholds ==="
  cp "$data/warmup_per_step/gpw_$tag.jsonl" "$data/warmup_scores.jsonl"
  $PY -m exp.gate_threshold_pareto.solve_gtp \
    --warmup-per-step "$data/warmup_scores.jsonl" --out "$data/solved.json"

  say "=== $suite: emit sweep + gate-only ==="
  $PY -m exp.libero_groot.emit_gate_yamls --mode eval --suite "$suite" --solved "$data/solved.json"
  $PY -m exp.libero_groot.emit_gate_yamls --mode gate_only --suite "$suite"

  # -------------------------------------------------------------------------
  # Smoke: two arms, a small sample, before 17,200 episodes are committed. The
  # gate-only arm is mandatory here -- its judge accepts every search, so the
  # V2 injection is guaranteed to fire and gate-skip presence becomes a sound
  # per-arm assertion rather than a probabilistic one.
  # -------------------------------------------------------------------------
  say "=== $suite: smoke (2 arms x $((SMOKE_EP * 10)) ep) ==="
  rm -rf "$cfg/smoke" "$data/smoke_results" "$data/smoke_per_step"
  mkdir -p "$cfg/smoke"
  cp "$cfg/gate_only/gpgo_$tag.yaml" "$cfg/eval/gp_${tag}_fh80.yaml" "$cfg/smoke/"
  run_stage "smoke_$tag" "$suite" "$cfg/smoke" "$data/smoke_results" \
    "$data/smoke_per_step" $((SMOKE_EP * 10)) "$SMOKE_EP" ""
  verify_phase "smoke_$tag" "$data/smoke_results" "$data/smoke_per_step" \
    $((SMOKE_EP * 10)) "gpgo_$tag,gp_${tag}_fh80"
  GP_SMOKE_RESULTS_DIR="$data/smoke_results" \
  GP_SMOKE_PER_STEP_DIR="$data/smoke_per_step" \
  GP_SMOKE_EXPECT_EP=$((SMOKE_EP * 10)) \
  GP_SMOKE_ARMS="gpgo_$tag,gp_${tag}_fh80" \
    $REPO/.venv/bin/python -m pytest "$REPO/tests/libero_groot/test_gate_pareto_smoke.py" \
      -q --run-manual || die "smoke_$tag: T7 structural gate failed -- the gate is not connected"
  say "smoke_$tag: T7 passed, committing the full sweep"

  say "=== $suite: sweep (16 arms x 500 ep) ==="
  run_stage "eval_$tag" "$suite" "$cfg/eval" "$data/eval_results" "$data/eval_per_step" 500 50 ""

  say "=== $suite: gate-only (1 arm x 500 ep) ==="
  run_stage "go_$tag" "$suite" "$cfg/gate_only" "$data/eval_results" "$data/eval_per_step" 500 50 ""

  say "=== $suite: aggregate ==="
  arms=$(arm_list "$cfg/eval" "$cfg/gate_only")
  $PY exp/libero_groot/analysis/gate_pareto/analyze_gate_pareto.py aggregate \
    "$data/eval_results" "$data/eval_per_step" "$data/summary.json" \
    --expect-ep 500 --expect-arms "$arms" || die "$suite: final integrity gate rejected the sweep"
  say "$suite: summary at $data/summary.json"
done

# ---------------------------------------------------------------------------
# Deliverables. Only reached when every phase of every suite passed, so the
# figures can never depict a partially-failed run.
# ---------------------------------------------------------------------------
say "=== plot + manifest ==="
SPECS=()
for suite in $SUITES; do
  [ -f "$ROOT/$suite/summary.json" ] || die "$suite: summary.json missing at plot time"
  SPECS+=(--suite "$suite=$ROOT/$suite/summary.json")
done
$PY exp/libero_groot/analysis/gate_pareto/analyze_gate_pareto.py plot \
  "${SPECS[@]}" --out-dir "$REPO/exp/libero_groot/analysis/gate_pareto" \
  --status "gate-threshold Pareto, $(date '+%Y-%m-%d'), teacher-rate axis only"

for suite in $SUITES; do
  for f in "pareto_$suite.png" "pareto_$suite.pdf"; do
    [ -s "$REPO/exp/libero_groot/analysis/gate_pareto/$f" ] || die "deliverable missing: $f"
  done
done
[ -s "$REPO/exp/libero_groot/analysis/gate_pareto/plot_data.json" ] || die "plot_data.json missing"
[ -s "$REPO/exp/libero_groot/analysis/gate_pareto/MANIFEST.json" ] || die "MANIFEST.json missing"

say "GATE-PARETO-DONE"
