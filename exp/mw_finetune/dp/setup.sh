#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
DP_ROOT=${DP_ROOT:-/data/mw_finetune/dp}
DP_BASE_PYTHON=${DP_BASE_PYTHON:-$PWD/.venv/bin/python}
mkdir -p "$DP_ROOT"
if [[ ! -x "$DP_ROOT/env/bin/python" ]]; then
  "$DP_BASE_PYTHON" -m venv --without-pip "$DP_ROOT/env"
fi
# Reuse installed packages read-only; keep all E installation files in E's root.
DP_BASE_SITE=$("$DP_BASE_PYTHON" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
DP_OWN_SITE=$("$DP_ROOT/env/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
printf '%s\n' "$DP_BASE_SITE" > "$DP_OWN_SITE/dp_readonly_base.pth"
export PYTHONPATH="$PWD:$PWD/src:$PWD/packages/openpi-client/src"
export HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
"$DP_ROOT/env/bin/python" -m exp.mw_finetune.dp.setup --root "$DP_ROOT"
"$DP_ROOT/env/bin/python" -c 'import importlib.metadata as m; print("\n".join(sorted("{}=={}".format(d.metadata["Name"], d.version) for d in m.distributions())))' > "$DP_ROOT/environment.txt"
