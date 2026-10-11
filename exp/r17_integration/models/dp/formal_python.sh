#!/bin/bash
# Redirect only the resident entry. All other P6 commands pass through.
set -euo pipefail
forwarded=("$@")
for ((index=0; index+1<${#forwarded[@]}; index++)); do
    if [[ "${forwarded[index]}" == '-m' && "${forwarded[index+1]}" == 'exp.r17_integration.ops.formal_server' ]]; then
        forwarded[index+1]='exp.r17_integration.models.dp.formal_batch_server'
        break
    fi
done
exec /data/mw_finetune/dp/env/bin/python "${forwarded[@]}"
