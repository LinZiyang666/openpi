#!/usr/bin/env bash
# usage: analyze_all.sh <rows_root> <weights_dir> <rc_arms_root> <json_out> [gaps_dir] [libero_arms_root]
set -euo pipefail
ROWS=${1:?rows}; W=${2:?weights}; ARMS=${3:?RC arms}; OUT=${4:?json out}; GAPS=${5:-}
LIB_ARMS=${6:-exp/step_diag/data/libero}
REPORTS=${SD_REPORTS:-exp/step_diag/analysis}
mkdir -p "$OUT" "$REPORTS"
REPORT_FILES=()
for env in pi05_rc groot_rc pi05_libero_spatial pi05_libero_10 groot_libero_spatial groot_libero_10; do
  case "$env" in pi05_*) teacher=pi05 ;; *) teacher=groot_tp ;; esac
  d="$ROWS/$teacher/shadow_$env"
  [ -d "$d" ] || { echo "skip $env (no $d)"; continue; }
  extra=()
  case "$env" in
    *_rc)
      driver="$ARMS/$teacher/shadow"
      if [ -n "$GAPS" ] && [ -f "$GAPS/gaps_${env%_rc}_rc.json" ]; then
        extra+=(--gaps "$GAPS/gaps_${env%_rc}_rc.json")
      fi ;;
    *) driver="$LIB_ARMS/$teacher/shadow_$env" ;;
  esac
  uv run python -m exp.step_diag.analysis.analyze_shadow analyze --env-id "$env" --rows "$d" \
    --driver-dir "$driver" --weights "$W/weights_$env.npz" "${extra[@]}" \
    --out-json "$OUT/shadow_$env.json" --out-md "$REPORTS/shadow_$env.md"
  REPORT_FILES+=("$REPORTS/shadow_$env.md")
done
for policy in pi05 groot; do
  extra=()
  [ ! -f "$OUT/shadow_${policy}_rc.json" ] || extra+=(--shadow-json "$OUT/shadow_${policy}_rc.json")
  uv run python -m exp.step_diag.analysis.aggregate_arms --policy "$policy" --arms-root "$ARMS" --server-rows "$ROWS" \
    "${extra[@]}" --out-json "$OUT/qb_$policy.json" --out-md "$REPORTS/qb_$policy.md"
  REPORT_FILES+=("$REPORTS/qb_$policy.md")
done
# generated tables only; the interpreted report exp/step_diag/analysis/step_vs_warmstart.md is written by hand
cat "${REPORT_FILES[@]}" > "$REPORTS/step_vs_warmstart_tables.md"
