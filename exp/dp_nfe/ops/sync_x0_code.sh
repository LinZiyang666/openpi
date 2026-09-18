#!/bin/bash
# Local: push the exp/dp_nfe code + tests to a host's DP layout (<root>/openpi_exp/exp/dp_nfe, <root>/openpi_exp/tests/dp_nfe)
# so the DP env can import exp.dp_nfe.*  usage: sync_x0_code.sh <host> <dp base dir e.g. /home/weiland/dp or /data/dp_h100>
set -euo pipefail
H=${1:?host}; B=${2:?dp base}
cd "$(dirname "$0")/../../.."
tether exec --timeout 30s $H -- bash -lc "mkdir -p $B/openpi_exp/exp/dp_nfe/analysis $B/openpi_exp/exp/dp_nfe/config/x0_multimodal $B/openpi_exp/exp/dp_nfe/ops $B/openpi_exp/tests/dp_nfe; touch $B/openpi_exp/exp/__init__.py" >/dev/null 2>&1
for f in exp/dp_nfe/__init__.py exp/dp_nfe/dp_sampler.py exp/dp_nfe/x0_workspace.py exp/dp_nfe/train_x0.py exp/dp_nfe/eval_dp_steps_v2.py \
         exp/dp_nfe/mode_filter_datasets.py exp/dp_nfe/x0_cells.py exp/dp_nfe/x0_identity.py exp/dp_nfe/x0_normalizer.py \
         exp/dp_nfe/x0_queue.py exp/dp_nfe/smoke_x0.py exp/dp_nfe/analysis/__init__.py \
         exp/dp_nfe/analysis/dispersion_index.py exp/dp_nfe/analysis/aggregate_x0.py exp/dp_nfe/analysis/plot_x0.py exp/dp_nfe/config/x0_multimodal/tasks.yaml \
         exp/dp_nfe/config/x0_multimodal/hosts.yaml exp/dp_nfe/ops/dp_x0_env.sh exp/dp_nfe/ops/x0_queue.sh \
         tests/__init__.py tests/dp_nfe/__init__.py tests/dp_nfe/dp_stubs.py tests/dp_nfe/test_x0_sampler.py tests/dp_nfe/test_x0_aggregate.py \
         tests/dp_nfe/test_x0_data.py tests/dp_nfe/test_x0_workspace.py tests/dp_nfe/test_x0_workspace_cpu.py \
         tests/dp_nfe/test_x0_eval_keying.py tests/dp_nfe/test_x0_queue.py tests/dp_nfe/test_x0_cli.py tests/dp_nfe/test_x0_review_regressions.py; do
  [ -f "$f" ] || continue
  tether push "$f" "$H:$B/openpi_exp/$f" --force
done
echo "synced to $H:$B/openpi_exp"
