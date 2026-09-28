#!/usr/bin/env bash
# COORDINATOR ONLY. One push and one remote exec; never replace shared launchers.
# /tmp/p3_stage is the staged TAR FILE, so its parent /tmp already exists.
# Usage: deploy_client.sh <bundle-directory> [--dry-run]
set -euo pipefail
BUNDLE=$(realpath "${1:?bundle-directory}")
cd "$BUNDLE"
sha256sum -c client_bundle.sha256
digest=$(sha256sum client_bundle.tar | cut -d' ' -f1)
echo "local $digest $BUNDLE/client_bundle.tar"
while IFS= read -r -d '' f; do sha256sum "$f"; done < <(find payload -type f -print0 | sort -z)
# Preflight imports only: no workers, driver loops, env construction or network.
command="set -euo pipefail
test \"\$(sha256sum /tmp/p3_stage | cut -d' ' -f1)\" = '$digest'
echo 'remote $digest /tmp/p3_stage'
stage=\$(mktemp -d /tmp/p3_unpack.XXXXXXXX)
tar -xf /tmp/p3_stage -C \"\$stage\"
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /scratch/zixuans8/libero_sim/bin/python \"\$stage/install_client_payload.py\" --bundle \"\$stage\"
cd /scratch/zixuans8/openpi_trace
export HOME=/home/zixuans8 LIBERO_CONFIG_PATH=/home/zixuans8/.libero
PYTHONPATH=/scratch/zixuans8/openpi_trace/packages/openpi-client/src:/scratch/zixuans8/openpi_trace/src:/scratch/zixuans8/openpi_trace PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /scratch/zixuans8/libero_sim/bin/python -m exp.offline_search.rounds.r06.p3_profiling.client_preflight --stock-imports < /dev/null"
if [ "${2:-}" = --dry-run ]; then
  printf '%s\n' "tether push --force $BUNDLE/client_bundle.tar timan107:/tmp/p3_stage" "$command"
  exit 0
fi
[ $# -eq 1 ] || { echo 'unexpected argument' >&2; exit 2; }
timeout 300 tether push --force "$BUNDLE/client_bundle.tar" timan107:/tmp/p3_stage
timeout 600 tether exec timan107 -- bash -c "$command"
echo P3_CLIENT_DEPLOY_OK
