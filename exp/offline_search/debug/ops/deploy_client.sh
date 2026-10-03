#!/usr/bin/env bash
# Coordinator only. Usage: deploy_client.sh <bundle-directory> [--dry-run]
set -euo pipefail
BUNDLE=$(realpath "${1:?bundle-directory}")
cd "$BUNDLE"
sha256sum -c client_bundle.sha256
digest=$(sha256sum client_bundle.tar | cut -d' ' -f1)
command="set -euo pipefail
test \"\$(sha256sum /tmp/osdebug_client_stage | cut -d' ' -f1)\" = '$digest'
stage=\$(mktemp -d /tmp/osdebug_unpack.XXXXXXXX)
tar -xf /tmp/osdebug_client_stage -C \"\$stage\"
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /scratch/zixuans8/libero_sim/bin/python \"\$stage/install_client_payload.py\" --bundle \"\$stage\"
cd /scratch/zixuans8/openpi_trace
LIBERO_CONFIG_PATH=/home/zixuans8/.libero PYTHONPATH=/scratch/zixuans8/openpi_trace/packages/openpi-client/src:/scratch/zixuans8/openpi_trace/src:/scratch/zixuans8/openpi_trace PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /scratch/zixuans8/libero_sim/bin/python -m exp.offline_search.debug.client.preflight --stock-imports"
if [ "${2:-}" = --dry-run ]; then
  printf '%s\n' "tether push --force $BUNDLE/client_bundle.tar timan107:/tmp/osdebug_client_stage" "$command"
  exit 0
fi
[ $# -eq 1 ] || { echo 'unexpected argument' >&2; exit 2; }
timeout 300 tether push --force "$BUNDLE/client_bundle.tar" timan107:/tmp/osdebug_client_stage
timeout 600 tether exec timan107 -- bash -c "$command"
echo OSDEBUG_CLIENT_DEPLOY_OK
