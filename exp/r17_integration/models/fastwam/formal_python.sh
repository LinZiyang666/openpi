#!/bin/bash
# P3 local shim: redirect only P6's resident entry, preserve all other commands.
set -euo pipefail
forwarded=("$@")
for ((index=0; index+1<${#forwarded[@]}; index++)); do
    if [[ "${forwarded[index]}" == '-m' && "${forwarded[index+1]}" == 'exp.r17_integration.ops.formal_server' ]]; then
        forwarded[index+1]='exp.r17_integration.models.fastwam.formal_batch_server'
        break
    fi
done
exec /data/mw_finetune/envs/fastwam/bin/python "${forwarded[@]}"
