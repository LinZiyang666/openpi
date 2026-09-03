#!/bin/bash
# setup_t108.sh — build the timan108 client fleet from weilandserver's staging server.
# Phase A (code + data, fast): openpi_dispatch clone, A-pools, RIT arms/artifacts.
# Phase B (LIBERO conda prefix, 2 GB): /scratch/zixuans8/libero_sim, EGL hook rewritten for timan108.
set -uo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
S=http://ziyanglin.com:23161
R=/scratch/zixuans8/openpi_dispatch
mkdir -p $R /tmp/dsp_shared/rit_pareto /tmp/dsp_precheck/rit_pareto /scratch/zixuans8/dsp_bin
cd /tmp
step() { echo "=== [$(date +%H:%M:%S)] $*"; }
step code
curl -sS -m 120 -o /tmp/code_t108.tgz $S/code_t108.tgz && curl -sS -m 60 -o /tmp/code_t108.sha $S/code_t108.sha || { echo SETUP_FAILED code-download; exit 1; }
cd $R && tar -xzf /tmp/code_t108.tgz && echo "code mismatch: $(sha256sum -c /tmp/code_t108.sha 2>&1 | grep -vc ': OK$')"
step apools
curl -sS -m 120 -o /tmp/apools.tgz $S/apools.tgz && mkdir -p $R/exp/common/data/db_init/libero && tar -xzf /tmp/apools.tgz -C $R/exp/common/data/db_init/libero && ls $R/exp/common/data/db_init/libero
step arms
curl -sS -m 120 -o /tmp/rit_arms_both.tgz $S/rit_arms_both.tgz && tar -xzf /tmp/rit_arms_both.tgz -C /tmp/dsp_shared/rit_pareto && ls /tmp/dsp_shared/rit_pareto/libero_spatial/arms | head -3
step shim
cat > /scratch/zixuans8/dsp_bin/conda <<'SHIM'
#!/bin/bash
# conda shim (timan108): supports only `conda run [--no-capture-output] -p <prefix> <cmd...>`.
# Sources the prefix's activate.d hooks, then execs the command with the prefix's bin first on PATH.
args=("$@")
[ "${args[0]}" = "run" ] || { echo "conda shim: only 'run' is supported" >&2; exit 2; }
shift
[ "${1:-}" = "--no-capture-output" ] && shift
[ "${1:-}" = "-p" ] || { echo "conda shim: expected -p <prefix>" >&2; exit 2; }
prefix=$2; shift 2
export CONDA_PREFIX="$prefix"
export PATH="$prefix/bin:$PATH"
for f in "$prefix"/etc/conda/activate.d/*.sh; do [ -f "$f" ] && . "$f"; done
exec "$@"
SHIM
chmod +x /scratch/zixuans8/dsp_bin/conda
step done-phase-A
echo SETUP_A_DONE
